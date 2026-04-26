#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dunakeszi_segment_extractor.py
───────────────────────────────
Extract ground-truth audio segments from any Dunakeszi polywav file and
package them as a test ZIP ready for the drone_detection pipeline.

USAGE — drop any WAV file(s) from the repository and run:

    python dunakeszi_segment_extractor.py --segments ground_truth/ground_truth_segments.json --wav-dir   wavs/ --output-dir dunakeszi_test_segments

The extractor inspects EVERY .wav file in --wav-dir, determines its
time-slot from the filename suffix (no suffix=slot 0, A=slot 1, B=slot 2 …
U=slot 21 … and so on for the full ~28-slot, 3-hour recording), computes
which ground-truth segments overlap that slot, extracts them, and skips
segments whose files are absent.  You never need to download all files at
once.

Timing model
────────────
  The recording started at 13:36:00 local (CEST) = onset_from_rec_s = 0.
  Each polywav file is exactly 4 GB → 399.46 s @ 192 kHz, 14 ch, float32.
  File suffix → slot index:

      251020VITEMOROM1AT01.wav   → slot  0  (13:36 – 13:43)
      251020VITEMOROM1AT01A.wav  → slot  1  (13:43 – 13:49)
      251020VITEMOROM1AT01B.wav  → slot  2  (13:49 – 13:56)
      …
      251020VITEMOROM1AT01I.wav  → slot  9  (14:35 – 14:42)
      251020VITEMOROM1AT01J.wav  → slot 10  (14:42 – 14:49)
      …
      251020VITEMOROM1AT01U.wav  → slot 21  (15:55 – 16:02)
      …

  slot_start_s = slot_index × 399.46      (seconds from recording start)
  within_file_sample = (onset_from_rec_s − slot_start_s) × NATIVE_SR

Channel mapping (0-indexed in the 14-ch polywav)
──────────────────────────────────────────────────
  ch 0, 1    = Mix L/R (not used)
  ch 2, 3, 4 = BK-6-W  E, H, B  (Scorpio ch 3, 4, 5)  ← TDOA subset used here
  ch 5–7     = BK-6-W  J, F, L
  ch 8, 9, 10= BK-6-E  E, H, B  (Scorpio ch 9, 10, 11) ← alternative TDOA subset
  ch 11–13   = BK-6-E  J, F, L

Output format: generic_triplet (directly loadable by the drone_detection pipeline)
    <session_id>_ch0.wav   — BK-6-W E  (22 050 Hz, mono, 3.0 s)
    <session_id>_ch1.wav   — BK-6-W H
    <session_id>_ch2.wav   — BK-6-W B
    <session_id>_label.json
    labels.csv
    manifest.json
    dunakeszi_test_segments.zip
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed.  Run: pip install librosa")
    sys.exit(1)

# ── Audio constants ────────────────────────────────────────────────────────────

NATIVE_SR      = 192_000
TARGET_SR      = 22_050
TARGET_DUR_S   = 3.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR_S)   # 66 150

# Exact 4 GB polywav → duration per file
_BYTES_PER_FRAME = 14 * 4          # 14 channels × float32
CHUNK_DUR_S      = (4 * 1024**3) / (_BYTES_PER_FRAME * NATIVE_SR)  # ≈ 399.4613 s

# BK-6-W TDOA subset: Scorpio ch 3,4,5 → 0-indexed polywav channels 2,3,4
POLYWAV_CHANNELS = [2, 3, 4]

# Recording reference (onset_from_rec_s = 0 corresponds to this local time)
REC_REF_LOCAL_S = 13 * 3600 + 36 * 60   # 48 960 s


# ── File-slot resolver ─────────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(
    r"251020VITEMOROM1AT01([A-Z]*)\.wav$", re.IGNORECASE
)

def wav_slot(path: Path) -> Optional[int]:
    """
    Return the slot index (0-based) for a polywav file, or None if not recognised.

    Slot 0  = no suffix (13:36–13:43)
    Slot 1  = suffix A
    Slot 9  = suffix I  (14:35–14:42)
    Slot 10 = suffix J  (14:42–14:49)
    Slot 21 = suffix U  (15:55–16:02)
    Slot 25 = suffix Y  (16:25–16:32)  ← show_10 long-range
    """
    m = _SUFFIX_RE.match(path.name)
    if m is None:
        return None
    suffix = m.group(1).upper()
    if suffix == "":
        return 0
    if len(suffix) == 1:
        return ord(suffix) - ord('A') + 1
    # Two-char suffix: AA=27, AB=28 … (future-proof)
    if len(suffix) == 2:
        return 26 + (ord(suffix[0]) - ord('A')) * 26 + (ord(suffix[1]) - ord('A') + 1)
    return None


def slot_time_range(slot: int) -> Tuple[float, float]:
    """Return (start_s, end_s) from recording start for a slot."""
    return slot * CHUNK_DUR_S, (slot + 1) * CHUNK_DUR_S


def slot_local_hms(slot: int) -> str:
    s = int(slot * CHUNK_DUR_S) + REC_REF_LOCAL_S
    return f"{s//3600:02d}:{(s%3600)//60:02d}"


# ── Audio I/O ──────────────────────────────────────────────────────────────────

def read_polywav_window(
    wav_path: Path,
    start_sample: int,
    n_frames: int,
    channels: List[int],
) -> np.ndarray:
    """
    Read n_frames samples from wav_path starting at start_sample,
    selecting the given channel indices.
    Returns (len(channels), n_frames) float32, zero-padded if past EOF.
    """
    info = sf.info(str(wav_path))
    actual_start  = max(0, min(start_sample, info.frames - 1))
    actual_frames = min(n_frames, max(0, info.frames - actual_start))

    if actual_frames > 0:
        data, _ = sf.read(str(wav_path), start=actual_start,
                          frames=actual_frames, dtype="float32", always_2d=True)
        out = data[:, channels].T.astype(np.float32)
    else:
        out = np.zeros((len(channels), 0), dtype=np.float32)

    if actual_frames < n_frames:
        pad = np.zeros((len(channels), n_frames - actual_frames), dtype=np.float32)
        out = np.concatenate([out, pad], axis=1)
    return out


def extract_overlap(
    seg: dict,
    wav_path: Path,
    slot: int,
) -> Tuple[np.ndarray, int]:
    """
    Extract the portion of `seg` that overlaps `slot` from `wav_path`.
    Returns (audio_chunk, offset_samples_from_seg_start).
    audio_chunk is (3, n_native) float32; may be empty if no overlap.
    """
    slot_start, slot_end = slot_time_range(slot)
    seg_onset = float(seg["onset_from_rec_s"])
    seg_end   = seg_onset + float(seg["duration_s"])

    ovlp_start = max(seg_onset, slot_start)
    ovlp_end   = min(seg_end,   slot_end)
    if ovlp_end <= ovlp_start:
        return np.zeros((3, 0), dtype=np.float32), 0

    within_file_s = ovlp_start - slot_start
    start_sample  = int(within_file_s * NATIVE_SR)
    n_frames      = int((ovlp_end - ovlp_start) * NATIVE_SR)

    audio = read_polywav_window(wav_path, start_sample, n_frames, POLYWAV_CHANNELS)
    offset_samples = int((ovlp_start - seg_onset) * NATIVE_SR)
    return audio, offset_samples


def assemble_segment(
    seg: dict,
    available: Dict[int, Path],
) -> Optional[np.ndarray]:
    """
    Assemble the full segment from whatever slot files are available.
    Returns (3, n_native) float32, or None if nothing could be extracted.
    Gaps where a file is missing are filled with silence.
    """
    seg_onset   = float(seg["onset_from_rec_s"])
    seg_dur     = float(seg["duration_s"])
    need_native = int(seg_dur * NATIVE_SR)

    start_slot = int(seg_onset          / CHUNK_DUR_S)
    end_slot   = int((seg_onset+seg_dur) / CHUNK_DUR_S)

    buf = np.zeros((3, need_native), dtype=np.float32)
    any_data = False

    for slot in range(start_slot, end_slot + 1):
        if slot not in available:
            continue
        chunk, offset = extract_overlap(seg, available[slot], slot)
        if chunk.shape[1] == 0:
            continue
        end_idx = min(offset + chunk.shape[1], need_native)
        src_len = end_idx - offset
        if src_len > 0:
            buf[:, offset:end_idx] = chunk[:, :src_len]
            any_data = True

    return buf if any_data else None


def resample_and_pad(audio_native: np.ndarray) -> np.ndarray:
    """(3, n) @ NATIVE_SR → (3, TARGET_SAMPLES) @ TARGET_SR, float32."""
    out = []
    for ch in audio_native:
        r = librosa.resample(ch, orig_sr=NATIVE_SR, target_sr=TARGET_SR)
        if len(r) >= TARGET_SAMPLES:
            r = r[:TARGET_SAMPLES]
        else:
            r = np.pad(r, (0, TARGET_SAMPLES - len(r)))
        out.append(r.astype(np.float32))
    return np.stack(out, axis=0)


# ── Label writer ───────────────────────────────────────────────────────────────

def make_label_json(seg: dict) -> dict:
    """Build label.json matching the format parse_label_json() expects."""
    az   = seg.get("azimuth_deg_onset")
    dist = seg.get("distance_xy_m_onset")
    ht   = float(seg.get("altitude_m") or seg.get("distance_3d_m_onset") or 0.0)
    if az   is None: az   = 0.0
    if dist is None: dist = float(seg.get("distance_3d_m_onset") or 0.0)
    return {
        "drone": {
            "azimuth":  float(az),
            "distance": float(dist),
            "height":   ht,
        },
        "segment_id":    seg["id"],
        "session":       seg["session"],
        "maneuver_type": seg["maneuver_type"],
        "flight_phase":  seg["flight_phase"],
        "n_drones":      seg["n_drones"],
        "split":         seg["split"],
        "speed_mps":     seg.get("speed_mps"),
        "radius_m":      seg.get("radius_m"),
        "duration_s":    seg["duration_s"],
    }


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_from_wav_dir(
    segments_json: Path,
    wav_dir: Path,
    output_dir: Path,
    splits: Optional[List[str]] = None,
    skip_unlabelled: bool = False,
    verbose: bool = True,
) -> List[dict]:

    with open(segments_json) as f:
        all_segments: List[dict] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover WAV files ────────────────────────────────────────────────────
    available: Dict[int, Path] = {}
    for wp in sorted(wav_dir.glob("*.wav")):
        slot = wav_slot(wp)
        if slot is None:
            if verbose:
                print(f"  [skip] unrecognised filename: {wp.name}")
            continue
        available[slot] = wp

    if not available:
        print(f"ERROR: no recognised polywav files in {wav_dir}")
        print("  Expected names like  251020VITEMOROM1AT01.wav  /  ...I.wav  /  ...U.wav")
        sys.exit(1)

    print(f"\nFound {len(available)} polywav file(s):")
    for slot in sorted(available):
        s, e = slot_time_range(slot)
        print(f"  slot {slot:2d}  ({s:7.0f}–{e:7.0f} s from rec start / "
              f"local {slot_local_hms(slot)})  {available[slot].name}")

    covered_slots = set(available.keys())

    # ── Filter segments ───────────────────────────────────────────────────────
    candidates = []
    for seg in all_segments:
        if splits and seg.get("split") not in splits:
            continue
        if skip_unlabelled and seg.get("azimuth_deg_onset") is None:
            continue
        onset   = float(seg["onset_from_rec_s"])
        dur     = float(seg["duration_s"])
        ss      = int(onset         / CHUNK_DUR_S)
        es      = int((onset + dur) / CHUNK_DUR_S)
        needed  = set(range(ss, es + 1))
        if needed & covered_slots:
            candidates.append(seg)

    print(f"\n{len(candidates)} segment(s) overlap the provided file(s)"
          + (f"  [splits={splits}]" if splits else ""))
    if not candidates:
        print("  The provided WAV file(s) don't cover any segments in the "
              "requested splits.")
        return []

    # ── Extract ───────────────────────────────────────────────────────────────
    manifest: List[dict] = []
    skipped = 0

    for seg in candidates:
        sid        = seg["id"]
        split      = seg.get("split", "unknown")
        maneuver   = seg.get("maneuver_type", "unknown")
        session_id = f"seg_{sid:03d}_{maneuver}_{split}"
        flags      = list(seg.get("quality_flags", []))
        onset      = float(seg["onset_from_rec_s"])
        dur        = float(seg["duration_s"])

        if verbose:
            print(f"  → {session_id}  onset={onset:.1f}s  dur={dur:.1f}s", end="  ")

        # Flag partial coverage
        ss     = int(onset        / CHUNK_DUR_S)
        es     = int((onset+dur)  / CHUNK_DUR_S)
        needed = set(range(ss, es + 1))
        missing = needed - covered_slots
        if missing:
            flags.append("partial_coverage")
            if verbose:
                print(f"[partial: slots {sorted(missing)} absent]", end="  ")

        # Assemble audio from available files
        audio_native = assemble_segment(seg, available)
        if audio_native is None:
            if verbose:
                print("SKIP (no audio)")
            skipped += 1
            continue

        audio_model = resample_and_pad(audio_native)
        rms_vals = [float(np.sqrt(np.mean(ch**2))) for ch in audio_model]
        max_rms  = max(rms_vals)

        if max_rms < 1e-7:
            if verbose:
                print("SKIP (silent — file does not cover this segment)")
            skipped += 1
            continue

        # Write outputs
        for i, ch in enumerate(audio_model):
            sf.write(str(output_dir / f"{session_id}_ch{i}.wav"),
                     ch, TARGET_SR, subtype="FLOAT")
        with open(output_dir / f"{session_id}_label.json", "w") as f:
            json.dump(make_label_json(seg), f, indent=2)

        rms_db = round(20 * np.log10(max_rms + 1e-10), 1)
        entry = {
            "session_id":       session_id,
            "segment_id":       sid,
            "session":          seg["session"],
            "split":            split,
            "maneuver_type":    maneuver,
            "flight_phase":     seg.get("flight_phase"),
            "n_drones":         seg.get("n_drones", 1),
            "drones":           seg.get("drones", []),
            "onset_from_rec_s": onset,
            "duration_s":       dur,
            "altitude_m":       seg.get("altitude_m"),
            "speed_mps":        seg.get("speed_mps"),
            "radius_m":         seg.get("radius_m"),
            "azimuth_deg":      seg.get("azimuth_deg_onset"),
            "distance_xy_m":    seg.get("distance_xy_m_onset"),
            "distance_3d_m":    seg.get("distance_3d_m_onset"),
            "rms_ch0":          rms_vals[0],
            "rms_ch1":          rms_vals[1],
            "rms_ch2":          rms_vals[2],
            "rms_max_db":       rms_db,
            "quality_flags":    flags,
            "source_files":     [available[s].name for s in sorted(needed & covered_slots)],
            "array":            "BK-6-W",
            "array_geometry":   "gp2",
            "polywav_channels": POLYWAV_CHANNELS,
            "wav_ch0":          f"{session_id}_ch0.wav",
            "wav_ch1":          f"{session_id}_ch1.wav",
            "wav_ch2":          f"{session_id}_ch2.wav",
            "label_json":       f"{session_id}_label.json",
        }
        manifest.append(entry)
        if verbose:
            print(f"✓  rms={rms_db} dB")

    print(f"\n✅ Extracted {len(manifest)} segments  ({skipped} skipped)")
    return manifest


# ── Output writers ─────────────────────────────────────────────────────────────

def write_labels_csv(manifest: List[dict], output_dir: Path):
    import csv
    fieldnames = ["session_id", "azimuth_deg", "distance_m", "height_m",
                  "split", "maneuver_type", "n_drones", "altitude_m", "speed_mps"]
    with open(output_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in manifest:
            w.writerow({
                "session_id":    r["session_id"],
                "azimuth_deg":   r["azimuth_deg"]   if r["azimuth_deg"]   is not None else "",
                "distance_m":    r["distance_xy_m"]  if r["distance_xy_m"] is not None else "",
                "height_m":      r["altitude_m"]     if r["altitude_m"]    is not None else "",
                "split":         r["split"],
                "maneuver_type": r["maneuver_type"],
                "n_drones":      r["n_drones"],
                "altitude_m":    r["altitude_m"]     if r["altitude_m"]    is not None else "",
                "speed_mps":     r["speed_mps"]      if r["speed_mps"]     is not None else "",
            })
    print(f"📋 labels.csv written")


def write_manifest(manifest: List[dict], output_dir: Path):
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 manifest.json written  ({len(manifest)} segments)")


def create_zip(output_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(output_dir.iterdir()):
            if fp.is_file():
                zf.write(fp, arcname=fp.name)
    mb = zip_path.stat().st_size / (1024**2)
    print(f"📦 ZIP → {zip_path}  ({mb:.1f} MB)")


def print_summary(manifest: List[dict]):
    from collections import Counter
    print("\n" + "═" * 65)
    print("  EXTRACTION SUMMARY")
    print("═" * 65)
    print(f"  Total extracted : {len(manifest)} segments")
    for sp, n in sorted(Counter(r["split"] for r in manifest).items()):
        print(f"    {sp:8s}: {n}")
    print(f"\n  Maneuver types:")
    for m, n in sorted(Counter(r["maneuver_type"] for r in manifest).items(), key=lambda x: -x[1]):
        print(f"    {m:20s}: {n}")
    print(f"\n  Source files:")
    for f in sorted({f for r in manifest for f in r["source_files"]}):
        n = sum(1 for r in manifest if f in r["source_files"])
        print(f"    {f}  ({n} seg{'s' if n!=1 else ''})")
    labelled = sum(1 for r in manifest if r["azimuth_deg"] is not None)
    print(f"\n  Labelled (az+dist+ht) : {labelled}/{len(manifest)}")
    rms_dbs = [r["rms_max_db"] for r in manifest]
    print(f"  RMS range             : {min(rms_dbs):.1f} … {max(rms_dbs):.1f} dB")
    print("═" * 65)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Dunakeszi ground-truth segments from any subset of polywav files"
    )
    ap.add_argument("--segments",    required=True,
                    help="Path to ground_truth_segments.json")
    ap.add_argument("--wav-dir",     required=True,
                    help="Directory containing polywav .wav file(s)")
    ap.add_argument("--output-dir",  default="dunakeszi_test_segments",
                    help="Output directory (default: dunakeszi_test_segments)")
    ap.add_argument("--splits",      nargs="+", choices=["train", "val", "test"],
                    default=None, help="Restrict to specific splits")
    ap.add_argument("--no-zip",      action="store_true",
                    help="Skip ZIP creation")
    ap.add_argument("--skip-unlabelled", action="store_true",
                    help="Skip segments with no azimuth label")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print what would be extracted without writing files")
    args = ap.parse_args()

    segments_json = Path(args.segments)
    wav_dir       = Path(args.wav_dir)
    output_dir    = Path(args.output_dir)

    for p, name in [(segments_json, "segments JSON"), (wav_dir, "wav directory")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}"); sys.exit(1)

    print("═" * 65)
    print("  Dunakeszi Ground-Truth Segment Extractor")
    print("═" * 65)
    print(f"  Segments JSON  : {segments_json}")
    print(f"  WAV directory  : {wav_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Splits         : {args.splits or 'all'}")
    print(f"  Resample       : {NATIVE_SR} Hz → {TARGET_SR} Hz")
    print(f"  Output length  : {TARGET_DUR_S} s  ({TARGET_SAMPLES} samples)")
    print(f"  Channels       : polywav idx {POLYWAV_CHANNELS}  (BK-6-W E, H, B)")
    print(f"  Slot duration  : {CHUNK_DUR_S:.4f} s per file")

    if args.dry_run:
        with open(segments_json) as f:
            all_segs = json.load(f)
        available = {wav_slot(wp): wp for wp in sorted(wav_dir.glob("*.wav"))
                     if wav_slot(wp) is not None}
        print(f"\nDRY RUN — {len(available)} file(s), slots {sorted(available.keys())}")
        for seg in all_segs:
            if args.splits and seg.get("split") not in args.splits:
                continue
            onset = float(seg["onset_from_rec_s"])
            dur   = float(seg["duration_s"])
            ss    = int(onset        / CHUNK_DUR_S)
            es    = int((onset+dur)  / CHUNK_DUR_S)
            slots = set(range(ss, es+1))
            has   = bool(slots & set(available.keys()))
            print(f"  seg_{seg['id']:03d}_{seg['maneuver_type']:15s} "
                  f"slots={sorted(slots)}  "
                  f"{'✓ extractable' if has else '-- not covered (download slot '+str(sorted(slots))+' files)'}")
        return

    manifest = extract_from_wav_dir(
        segments_json=segments_json,
        wav_dir=wav_dir,
        output_dir=output_dir,
        splits=args.splits,
        skip_unlabelled=args.skip_unlabelled,
        verbose=True,
    )

    if not manifest:
        print("\nNothing extracted. Check that --wav-dir contains files covering "
              "the required time slots, or use --dry-run to preview coverage.")
        sys.exit(0)

    write_labels_csv(manifest, output_dir)
    write_manifest(manifest, output_dir)
    print_summary(manifest)

    if not args.no_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        create_zip(output_dir, zip_path)
        print(f"\n🎯  Ready for model evaluation:")
        print(f"    test_ds = load_test_dataset_zip('{zip_path}', config,")
        print(f"              dataset_format='generic_triplet')")
        print(f"    results = run_test_dataset_evaluation(test_ds, config)")


if __name__ == "__main__":
    main()