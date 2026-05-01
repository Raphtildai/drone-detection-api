# -*- coding: utf-8 -*-
"""
inference_test_loader.py  (v17 patch)
──────────────────────────────────────
Changes from v16
────────────────
_detect_format()
  Improved heuristic: now checks for flat WAV files (no sub-folder
  structure) and tries to infer triplets from common naming conventions
  before defaulting to uavirbase.  Prints a list of the first 20 file
  names found so you can see what the zip actually contains.

_index_uavirbase()
  Now also searches for flat (non-sub-folder) session layouts where the
  zip root contains wav files directly.

_index_flat_wavs()  [NEW]
  Handles zips where all files are in the root with names like:
    session_001_mic1.wav  session_001_mic2.wav  session_001_mic3.wav
    session_001_label.json
  or any pattern where 3+ WAVs share a common stem prefix.
  Falls back to grouping all WAVs into consecutive triplets when no
  naming pattern is found (useful for unlabelled test sets).

Everything else unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


# ── lazy imports ─────────────────────────────────────────────────────────────
def _get_cfg(cfg=None):
    if cfg is not None:
        return cfg
    from .config import config
    return config

def _get_ap(cfg):
    from .audio_processing import AudioProcessor
    return AudioProcessor(cfg)

def _parse_label(raw: bytes):
    from .datasets import parse_label_json
    return parse_label_json(raw)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TestSession:
    session_id:   str
    wav_paths:    List[str]
    azimuth_deg:  Optional[float] = None
    distance_m:   Optional[float] = None
    height_m:     Optional[float] = None
    metadata:     Dict = field(default_factory=dict)

    @property
    def has_label(self): return self.azimuth_deg is not None


@dataclass
class TestDataset:
    sessions:       List[TestSession]
    source_zip:     str
    dataset_format: str
    extract_dir:    str
    n_labelled:     int = 0
    n_unlabelled:   int = 0

    def __len__(self): return len(self.sessions)
    def __repr__(self):
        return (f"TestDataset(n={len(self)}, labelled={self.n_labelled}, "
                f"format={self.dataset_format!r}, "
                f"zip={Path(self.source_zip).name!r})")


# ── ZIP extraction ────────────────────────────────────────────────────────────

def _safe_extract_zip(zip_path: str, dest: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            safe_name = Path(member.filename).as_posix().lstrip("/")
            if ".." in safe_name:
                continue
            target = dest / safe_name
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"✅ Extracted to {dest}")


def _find_audio_files(root: Path, exts=(".wav",".mp3",".flac",".ogg")) -> List[Path]:
    found = []
    for ext in exts:
        found.extend(root.rglob(f"*{ext}"))
    return sorted(found)


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(root: Path) -> str:
    all_files = list(root.rglob("*"))
    names     = [f.name for f in all_files if f.is_file()]

    # Show what's in the zip to help debugging
    print(f"   ZIP contents ({len(names)} files, first 20):")
    for n in names[:20]:
        print(f"     {n}")
    if len(names) > 20:
        print(f"     … and {len(names)-20} more")

    name_set = set(names)

    # PannoniaFS: Grp-prefixed WAVs or audio_files/ directory
    if any("Grp" in n or "grp" in n for n in names):
        return "pannoniafs"
    if any("audio_files" in str(f) for f in all_files if f.is_dir()):
        return "pannoniafs"

    # UaVirBASE: sub-folders each containing output.wav + label.json
    if ("output.wav" in name_set or "audio.wav" in name_set) and \
       ("label.json" in name_set or "annotation.json" in name_set):
        return "uavirbase"

    # Generic triplet: explicit _ch0/_ch1/_ch2 naming
    if any("_ch0" in n for n in names):
        return "generic_triplet"

    # Flat WAVs with any naming — try to form triplets
    wav_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".wav"]
    if len(wav_files) >= 3:
        return "flat_wavs"

    print("   ⚠️  Could not detect format — defaulting to 'uavirbase'")
    return "uavirbase"


# ── Indexers ──────────────────────────────────────────────────────────────────

def _index_uavirbase(root: Path, cfg) -> List[TestSession]:
    ap       = _get_ap(cfg)
    sessions = []
    mic_idx  = getattr(cfg, "UAVIRBASE_MIC_INDICES", [0, 1, 2])
    AUDIO_C  = {"output.wav", "audio.wav"}
    LABEL_C  = {"label.json", "annotation.json"}

    for session_dir in sorted(root.rglob("*")):
        if not session_dir.is_dir():
            continue
        files      = {f.name for f in session_dir.iterdir() if f.is_file()}
        audio_name = next((n for n in AUDIO_C if n in files), None)
        label_name = next((n for n in LABEL_C if n in files), None)
        if audio_name is None:
            continue

        audio_path = session_dir / audio_name
        session_id = session_dir.name
        az = di = ht = None
        if label_name:
            try:
                parsed = _parse_label((session_dir / label_name).read_bytes())
                if parsed: az, di, ht = parsed
            except Exception:
                pass

        out_dir = root / "_split" / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_paths = []
        try:
            channels = ap.load_channels(str(audio_path), channel_indices=mic_idx)
            for i, ch in enumerate(channels):
                p = str(out_dir / f"ch{i}.wav")
                sf.write(p, ap.pad_or_truncate(ch), cfg.SR)
                wav_paths.append(p)
        except Exception as e:
            print(f"   ⚠️  {session_id}: channel split failed ({e})")
            continue
        if len(wav_paths) < 3:
            continue
        sessions.append(TestSession(session_id=session_id, wav_paths=wav_paths[:3],
                                    azimuth_deg=az, distance_m=di, height_m=ht))
    return sessions


def _index_generic_triplet(root: Path, cfg) -> List[TestSession]:
    sessions = []
    for ch0 in sorted(root.rglob("*_ch0.wav")):
        stem = ch0.name.replace("_ch0.wav", "")
        d    = ch0.parent
        ch1  = d / f"{stem}_ch1.wav"
        ch2  = d / f"{stem}_ch2.wav"
        if not ch1.exists() or not ch2.exists():
            continue
        lf  = d / f"{stem}_label.json"
        az = di = ht = None
        if lf.exists():
            try:
                parsed = _parse_label(lf.read_bytes())
                if parsed: az, di, ht = parsed
            except Exception:
                pass
        sessions.append(TestSession(session_id=stem,
                                    wav_paths=[str(ch0), str(ch1), str(ch2)],
                                    azimuth_deg=az, distance_m=di, height_m=ht))
    return sessions


def _index_flat_wavs(root: Path, cfg) -> List[TestSession]:
    """
    Handle zips with flat WAV files (no sub-folder session structure).

    Strategy
    ────────
    1. Try to group WAVs by shared stem prefix separated by common
       delimiters (_mic, _ch, _m, digits at end).
    2. If exactly 3 WAVs share a prefix, treat them as one session.
    3. If no grouping works, sort all WAVs and take consecutive triplets.
    4. Look for a matching <stem>_label.json or labels.csv in the root.
    """
    ap       = _get_ap(cfg)
    wav_files = sorted(_find_audio_files(root, exts=(".wav",)))
    if not wav_files:
        return []

    # Try to load a labels.csv if present
    # Expected columns: session_id or filename, azimuth_deg, distance_m, height_m
    label_lookup: Dict[str, Tuple] = {}
    labels_csv = root / "labels.csv"
    if labels_csv.exists():
        try:
            with open(labels_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("session_id") or row.get("filename") or
                           row.get("stem") or "").strip()
                    try:
                        label_lookup[key] = (
                            float(row["azimuth_deg"]),
                            float(row["distance_m"]),
                            float(row["height_m"]),
                        )
                    except (KeyError, ValueError):
                        pass
            print(f"   📋 Loaded {len(label_lookup)} labels from labels.csv")
        except Exception as e:
            print(f"   ⚠️  Could not read labels.csv: {e}")

    # Try to group by prefix
    import re
    groups: Dict[str, List[Path]] = {}
    for wav in wav_files:
        stem = wav.stem
        # Strip trailing mic/channel indicators: _mic1, _ch1, _m1, or ending digits
        base = re.sub(r"[_\-](mic|ch|m|channel|mic)[_\-]?\d+$", "", stem,
                      flags=re.IGNORECASE)
        base = re.sub(r"[_\-]\d+$", "", base)   # strip trailing _01 _1 etc
        if not base:
            base = stem
        groups.setdefault(base, []).append(wav)

    # Find groups with exactly 3 members — these are clean triplets
    sessions   = []
    used        = set()
    out_dir     = root / "_split"
    out_dir.mkdir(exist_ok=True)

    for base, group_wavs in sorted(groups.items()):
        if len(group_wavs) == 3:
            session_id = base
            wav_paths  = _resample_and_pad(group_wavs, out_dir, session_id, ap, cfg)
            if len(wav_paths) < 3:
                continue
            az, di, ht = label_lookup.get(session_id, (None, None, None))
            # Also check individual JSON
            for w in group_wavs:
                lf = w.parent / f"{w.stem}_label.json"
                if lf.exists():
                    try:
                        p = _parse_label(lf.read_bytes())
                        if p: az, di, ht = p
                    except Exception:
                        pass
                    break
            sessions.append(TestSession(session_id=session_id,
                                        wav_paths=wav_paths,
                                        azimuth_deg=az, distance_m=di,
                                        height_m=ht))
            for w in group_wavs:
                used.add(str(w))

    # Leftover WAVs — take as consecutive triplets
    leftover = [w for w in wav_files if str(w) not in used]
    for i in range(0, len(leftover) - 2, 3):
        triple     = leftover[i:i+3]
        session_id = f"session_{i//3:04d}"
        wav_paths  = _resample_and_pad(triple, out_dir, session_id, ap, cfg)
        if len(wav_paths) < 3:
            continue
        az, di, ht = label_lookup.get(session_id, (None, None, None))
        sessions.append(TestSession(session_id=session_id, wav_paths=wav_paths,
                                    azimuth_deg=az, distance_m=di, height_m=ht))

    return sessions


def _resample_and_pad(wav_files: List[Path], out_dir: Path,
                      session_id: str, ap, cfg) -> List[str]:
    """Load, resample to cfg.SR, pad/truncate, and write mono WAVs."""
    paths = []
    for i, w in enumerate(wav_files):
        try:
            y = ap.load(str(w), mono=True)
            y = ap.pad_or_truncate(y)
            dest = out_dir / f"{session_id}_ch{i}.wav"
            sf.write(str(dest), y, cfg.SR)
            paths.append(str(dest))
        except Exception as e:
            print(f"   ⚠️  Could not load {w.name}: {e}")
    return paths


def _index_pannoniafs(root: Path, cfg) -> List[TestSession]:
    """PannoniaFS: audio_files/Grp*.wav + optional markers + position CSVs."""
    ap = _get_ap(cfg)
    audio_dir = root / "audio_files"
    if not audio_dir.exists():
        matches = [d for d in root.rglob("audio_files") if d.is_dir()]
        audio_dir = matches[0] if matches else root

    channel_names = ["Grp1-1", "Grp1-2", "Grp1-3"]
    channel_files: Dict[str, Optional[Path]] = {n: None for n in channel_names}
    for wav in _find_audio_files(audio_dir):
        for name in channel_names:
            if name in wav.stem:
                channel_files[name] = wav
    missing = [n for n, p in channel_files.items() if p is None]
    if missing:
        all_wavs = sorted(_find_audio_files(audio_dir))
        if len(all_wavs) < 3:
            raise RuntimeError("PannoniaFS: fewer than 3 WAV files found")
        for i, name in enumerate(channel_names):
            channel_files[name] = all_wavs[i]

    # Parse hover markers
    marker_files = list(root.rglob("*markers_samples.txt"))
    hover_events: List[Tuple[str, int, Optional[int]]] = []
    if marker_files:
        with open(marker_files[0]) as f:
            lines = [l.strip() for l in f if l.strip()]
        parsed_markers = []
        for line in lines:
            parts = line.lstrip("#").split(None, 2)
            try:
                if len(parts) >= 3:
                    sample_str, label = parts[1], " ".join(parts[2:])
                elif len(parts) == 2:
                    sample_str, label = parts
                else:
                    continue
                parsed_markers.append((int(sample_str), label.strip()))
            except (ValueError, IndexError):
                continue
        for i, (sample, label) in enumerate(parsed_markers):
            if "lebeg" in label.lower() or "hover" in label.lower():
                end_sample = (parsed_markers[i+1][0] if i+1 < len(parsed_markers)
                              else sample + int(20 * 192_000))
                end_sample = min(end_sample, sample + int(20 * 192_000))
                hover_events.append((label, sample, end_sample))
    if not hover_events:
        hover_events = [("full_file", 0, None)]

    sessions = []
    out_dir   = root / "_split"
    out_dir.mkdir(exist_ok=True)
    native_sr = getattr(cfg, "UAVIRBASE_ORIG_SR", 192_000)

    for event_label, start_smp, end_smp in hover_events:
        session_id = f"pannoniafs_{event_label.replace(' ','_')}"
        wav_paths  = []
        try:
            for ch_name in channel_names:
                src  = channel_files[ch_name]
                info = sf.info(str(src))
                nf   = (end_smp - start_smp) if end_smp else info.frames - start_smp
                y_n, _ = sf.read(str(src), start=start_smp,
                                  frames=min(nf, info.frames - start_smp),
                                  dtype="float32")
                if y_n.ndim > 1: y_n = y_n[:, 0]
                import librosa
                y_r = librosa.resample(y_n, orig_sr=native_sr, target_sr=cfg.SR)
                y_o = ap.pad_or_truncate(y_r)
                dest = out_dir / f"{session_id}_{ch_name}.wav"
                sf.write(str(dest), y_o, cfg.SR)
                wav_paths.append(str(dest))
        except Exception as e:
            print(f"   ⚠️  {session_id}: {e}"); continue
        if len(wav_paths) < 3: continue
        sessions.append(TestSession(session_id=session_id, wav_paths=wav_paths[:3],
                                    metadata={"event_label": event_label}))
    return sessions


# ── Public API — load ─────────────────────────────────────────────────────────

def load_test_dataset_zip(
    zip_path: str,
    cfg=None,
    extract_dir: Optional[str] = None,
    dataset_format: str = "auto",
    max_sessions: Optional[int] = None,
) -> TestDataset:
    """
    Extract a zipped test dataset and return an indexed TestDataset.

    Parameters
    ──────────
    zip_path        : path to the ZIP file
    cfg             : Config instance
    extract_dir     : where to unzip (temp dir if None)
    dataset_format  : "auto" | "uavirbase" | "pannoniafs" |
                      "generic_triplet" | "flat_wavs"
    max_sessions    : cap the number of sessions (useful for quick tests)

    ZIP layouts supported
    ─────────────────────
    uavirbase       sub-folder per session, each with output.wav + label.json
    pannoniafs      audio_files/Grp*.wav + optional markers/ + position/
    generic_triplet <stem>_ch0.wav + <stem>_ch1.wav + <stem>_ch2.wav
    flat_wavs       any flat WAV files — grouped into triplets automatically;
                    add a labels.csv with columns
                    session_id,azimuth_deg,distance_m,height_m for ground truth

    If your zip contains 89 files with none of the above naming conventions,
    use dataset_format="flat_wavs" explicitly. The loader will sort all WAVs
    alphabetically and take consecutive groups of 3 as sessions.
    """
    cfg      = _get_cfg(cfg)
    zip_path = str(zip_path)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}\n"
            "In Colab: upload via the Files panel first."
        )
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="drone_test_")
        print(f"📂 Extracting to temp dir: {extract_dir}")
    root = Path(extract_dir)
    print(f"📦 Loading test dataset from {Path(zip_path).name} …")
    _safe_extract_zip(zip_path, root)

    if dataset_format == "auto":
        dataset_format = _detect_format(root)
        print(f"   Detected format: {dataset_format!r}")

    print(f"   Indexing sessions (format={dataset_format!r}) …")
    if dataset_format == "uavirbase":
        sessions = _index_uavirbase(root, cfg)
    elif dataset_format == "pannoniafs":
        sessions = _index_pannoniafs(root, cfg)
    elif dataset_format == "generic_triplet":
        sessions = _index_generic_triplet(root, cfg)
    elif dataset_format == "flat_wavs":
        sessions = _index_flat_wavs(root, cfg)
    else:
        raise ValueError(
            f"Unknown dataset_format={dataset_format!r}. "
            "Choose from: uavirbase, pannoniafs, generic_triplet, flat_wavs, auto"
        )

    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    n_lab   = sum(1 for s in sessions if s.has_label)
    n_unlab = len(sessions) - n_lab
    print(f"✅ Indexed {len(sessions)} sessions  ({n_lab} labelled, {n_unlab} unlabelled)")

    if len(sessions) == 0:
        print("\n💡 Tip: if your ZIP has flat WAV files, try:")
        print("   test_ds = load_test_dataset_zip(zip_path, config, dataset_format='flat_wavs')")
        print("   If it has 3 WAVs per session with a shared name prefix, that will work.")
        print("   To add ground-truth labels, include a labels.csv:")
        print("   session_id,azimuth_deg,distance_m,height_m")

    return TestDataset(sessions=sessions, source_zip=zip_path,
                       dataset_format=dataset_format, extract_dir=extract_dir,
                       n_labelled=n_lab, n_unlabelled=n_unlab)


# ── Public API — evaluate ─────────────────────────────────────────────────────

def run_test_dataset_evaluation(
    test_ds: TestDataset,
    cfg=None,
    show_plots: bool = True,
    save_csv: Optional[str] = None,
) -> Dict:
    """
    Run detection + localisation over every session in test_ds.
    Prints a per-session table and a summary block.
    Returns a dict with mae_az_deg, mae_dist_m, mae_ht_m, detection_rate.
    """
    cfg = _get_cfg(cfg)

    from .inference import (
        load_detection_model, load_localization_model,
        load_3ch, detect, localize,
    )
    from .utils import angular_error_deg

    print(f"\n{'='*65}")
    print(f"  Test dataset evaluation: {Path(test_ds.source_zip).name}")
    print(f"  Format: {test_ds.dataset_format}  |  Sessions: {len(test_ds)}")
    print(f"{'='*65}")

    if len(test_ds) == 0:
        print("  ⚠️  No sessions to evaluate.")
        return {"sessions": [], "mae_az_deg": float("nan"),
                "mae_dist_m": float("nan"), "mae_ht_m": float("nan"),
                "detection_rate": 0.0, "n_sessions": 0, "n_labelled": 0}

    try:
        load_detection_model(cfg)
        load_localization_model(cfg)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model checkpoint not found: {e}\n"
            "Run train_localization() before evaluating."
        ) from e

    session_results = []
    az_errors = []; dist_errors = []; ht_errors = []
    n_detected = 0

    header = (f"{'Session':28s}  {'Det':3s}  {'Prob':5s}  "
              f"{'CNN_P':5s}  {'Heur_P':6s}  "
              f"{'Az_p':7s}  {'Az_t':7s}  {'Err':6s}  "
              f"{'Di_p':6s}  {'Di_t':6s}  {'Ht_p':5s}")
    print(header); print("-" * len(header))

    for sess in test_ds.sessions:
        try:
            channels = load_3ch(sess.wav_paths, cfg)
            det_res  = detect(channels, cfg)
            detected = det_res["detected"]
            prob     = det_res["probability"]
            cnn_prob = det_res["cnn_probability"]
            heur_prob = det_res["heuristic_probability"]
            if detected: n_detected += 1
            az_pred = dist_pred = ht_pred = float("nan")
            if detected:
                try:
                    loc_res  = localize(channels, cfg)
                    az_pred  = loc_res["azimuth_deg"]
                    dist_pred = loc_res["distance_m"]
                    ht_pred  = loc_res["height_m"]
                except Exception:
                    pass
            az_err = dist_err = ht_err = float("nan")
            if sess.has_label and not math.isnan(az_pred):
                az_err   = float(angular_error_deg(
                    np.array([az_pred]), np.array([sess.azimuth_deg]))[0])
                dist_err = abs(dist_pred - sess.distance_m)
                ht_err   = abs(ht_pred   - sess.height_m)
                az_errors.append(az_err); dist_errors.append(dist_err)
                ht_errors.append(ht_err)
            det_str = "YES" if detected else "no "
            az_t    = f"{sess.azimuth_deg:7.1f}" if sess.has_label else "      -"
            az_e    = f"{az_err:6.1f}" if not math.isnan(az_err) else "     -"
            di_t    = f"{sess.distance_m:6.2f}" if sess.has_label else "     -"
            print(f"{sess.session_id:28.28s}  {det_str}  {prob:.3f}  "
                  f"{cnn_prob:5.3f}  {heur_prob:6.3f}  "
                  f"{az_pred:7.1f}  {az_t}  {az_e}  "
                  f"{dist_pred:6.2f}  {di_t}  {ht_pred:5.2f}")
            session_results.append({
                "session_id": sess.session_id, "detected": detected,
                "probability": prob, "az_pred_deg": az_pred,
                "az_true_deg": sess.azimuth_deg, "az_err_deg": az_err,
                "dist_pred_m": dist_pred, "dist_true_m": sess.distance_m,
                "dist_err_m": dist_err, "ht_pred_m": ht_pred,
                "ht_true_m": sess.height_m, "ht_err_m": ht_err,
                "cnn_probability": det_res.get("cnn_probability", float("nan")),
            })
        except Exception as e:
            print(f"  ⚠️  {sess.session_id}: {e}")
            session_results.append({"session_id": sess.session_id, "error": str(e)})

    mae_az   = float(np.mean(az_errors))   if az_errors   else float("nan")
    mae_dist = float(np.mean(dist_errors)) if dist_errors else float("nan")
    mae_ht   = float(np.mean(ht_errors))   if ht_errors   else float("nan")
    det_rate = n_detected / max(len(test_ds), 1)

    print(f"\n{'='*65}")
    print(f"  Sessions evaluated : {len(test_ds)}")
    print(f"  Detection rate     : {det_rate:.1%}  ({n_detected}/{len(test_ds)})")
    if az_errors:
        print(f"  MAE azimuth        : {mae_az:.1f}°  (n={len(az_errors)})")
        print(f"  MAE distance       : {mae_dist:.2f} m")
        print(f"  MAE height         : {mae_ht:.2f} m")
    else:
        print("  (no labelled sessions — MAE not computed)")
    print(f"{'='*65}\n")

    if save_csv:
        try:
            keys = [k for k in session_results[0] if k != "error"]
            with open(save_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys + ["error"])
                w.writeheader()
                for row in session_results:
                    w.writerow({k: row.get(k, "") for k in keys + ["error"]})
            print(f"💾 Results saved to {save_csv}")
        except Exception as e:
            print(f"⚠️  Could not save CSV: {e}")

    if show_plots and az_errors:
        try:
            from .visualization import plot_polar_azimuth
            pred_azs = [r["az_pred_deg"] for r in session_results
                        if not math.isnan(r.get("az_pred_deg", float("nan")))]
            true_azs = [r["az_true_deg"] for r in session_results
                        if r.get("az_true_deg") is not None]
            if pred_azs:
                plot_polar_azimuth(pred_azs, true_azimuths=true_azs or None, cfg=cfg)
        except Exception as e:
            print(f"⚠️  Plot failed: {e}")

    return {"sessions": session_results, "mae_az_deg": mae_az,
            "mae_dist_m": mae_dist, "mae_ht_m": mae_ht,
            "detection_rate": det_rate, "n_sessions": len(test_ds),
            "n_labelled": len(az_errors)}


# ── Colab one-liner ───────────────────────────────────────────────────────────

def upload_and_evaluate(cfg=None, dataset_format="auto",
                        max_sessions=None, show_plots=True,
                        save_csv=None) -> Dict:
    """Upload a ZIP in Colab and evaluate immediately."""
    if "google.colab" not in sys.modules:
        raise RuntimeError(
            "upload_and_evaluate() requires Google Colab.\n"
            "Outside Colab:\n"
            "  test_ds = load_test_dataset_zip('/path/test.zip', config)\n"
            "  results = run_test_dataset_evaluation(test_ds, config)"
        )
    from google.colab import files
    print("📁 Please upload your test dataset ZIP.")
    uploaded = files.upload()
    zip_files = [p for p in uploaded if p.lower().endswith(".zip")]
    if not zip_files:
        raise ValueError("No ZIP file found in upload.")
    zip_path = zip_files[0]
    test_ds  = load_test_dataset_zip(zip_path, cfg=cfg,
                                     dataset_format=dataset_format,
                                     max_sessions=max_sessions)
    return run_test_dataset_evaluation(test_ds, cfg=cfg,
                                       show_plots=show_plots, save_csv=save_csv)