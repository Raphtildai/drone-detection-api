# repository_loader.py
# -*- coding: utf-8 -*-
"""
repository_loader.py
────────────────────
Streams drone-audio dataset segments directly from an online repository ZIP
via HTTP range-requests, driving RepositoryRealtimeSession in the live dashboard.

LIVE SERVER STORAGE POLICY
───────────────────────────
No archive files are ever downloaded or written to disk.  Dataset ZIPs
(UaVirBASE ≈ 4 GB, Dunakeszi ≈ 4 GB per polywav chunk) are far too large
for deployment server storage.  The `allow_download` and `cache_zip`
parameters are accepted for API compatibility with offline callers (Colab
notebooks, local scripts) but are silently ignored on this server.

Fallback chain (no disk I/O at any step)
─────────────────────────────────────────
1. RemoteZip streaming   — HTTP range-requests into the remote ZIP, fetching
                           only the bytes needed per session (~100 KB each).
                           Requires the `remotezip` package (auto-installed).
2. Synthetic fallback    — if the repo is unreachable and
                           allow_synthetic_fallback=True, generates segments
                           via synthesise_drone() using real BPF profiles.
→  RuntimeError          — if both strategies fail / are disabled.

Public API
──────────
stream_repository_segments(cfg, ...)
    → Generator[Tuple[List[np.ndarray], dict], None, None]

Each iteration yields:
    channels : List[np.ndarray]  — 3 float32 arrays at cfg.SR
    label    : dict              — {segment_id, azimuth_deg, distance_m, height_m,
                                    has_position, maneuver_type, split, source, ...}

Supported dataset_type values
──────────────────────────────
  "uavirbase"   — UaVirBASE Zenodo ZIP (sub-folders with output.wav + label.json)
  "dunakeszi"   — Dunakeszi outdoor ZIP (192 kHz polywav, array parameter selects mic group)
  Any other value is treated as "uavirbase" with a warning.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger("drone_v2.repository_loader")


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

# Audio filenames that may contain the multi-channel recording
_AUDIO_CANDIDATES = {"output.wav", "audio.wav", "recording.wav"}
# Label filenames that may carry the position metadata
_LABEL_CANDIDATES = {"label.json", "annotation.json", "metadata.json"}


def _ensure_remotezip() -> None:
    """Install remotezip if not already importable."""
    try:
        import remotezip  # noqa: F401
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "remotezip"]
        )


def _native_sr_for(dataset_type: str, cfg) -> int:
    """Return the native sample-rate of the dataset before resampling."""
    if dataset_type == "dunakeszi":
        return 192_000
    return getattr(cfg, "UAVIRBASE_ORIG_SR", 96_000)


def _parse_label_bytes(raw: bytes) -> Optional[dict]:
    """
    Decode a label.json / annotation.json byte string.

    Returns a normalised dict with keys:
        azimuth_deg, distance_m, height_m, has_position,
        maneuver_type, source, extra (raw JSON)
    or None if the file is an ambient/non-drone record or fails to parse.
    """
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

    # Skip ambient / background-only recordings
    drone_block = data.get("drone", data)
    src = drone_block.get("sound_source", "")
    if isinstance(src, str) and "ambient" in src.lower():
        return None

    # --- Extract azimuth ---
    az = None
    for key in ("azimuth_deg", "azimuth", "az", "bearing", "heading",
                "direction_deg", "direction"):
        if key in data:
            try:
                az = float(data[key]); break
            except (TypeError, ValueError):
                pass
    if az is None and "drone" in data:
        for key in ("azimuth_deg", "azimuth", "az", "bearing"):
            if key in data["drone"]:
                try:
                    az = float(data["drone"][key]); break
                except (TypeError, ValueError):
                    pass

    # --- Extract distance ---
    dist = None
    for key in ("distance_m", "distance", "dist", "range", "range_m",
                "horizontal_distance", "slant_range"):
        if key in data:
            try:
                dist = float(data[key]); break
            except (TypeError, ValueError):
                pass
    if dist is None and "drone" in data:
        for key in ("distance_m", "distance", "dist", "range"):
            if key in data["drone"]:
                try:
                    dist = float(data["drone"][key]); break
                except (TypeError, ValueError):
                    pass

    # --- Extract height ---
    ht = None
    for key in ("height_m", "height", "alt", "altitude", "z",
                "elevation", "altitude_m", "z_m", "height_agl"):
        if key in data:
            try:
                ht = float(data[key]); break
            except (TypeError, ValueError):
                pass
    if ht is None and "drone" in data:
        for key in ("height_m", "height", "alt", "altitude", "z"):
            if key in data["drone"]:
                try:
                    ht = float(data["drone"][key]); break
                except (TypeError, ValueError):
                    pass

    maneuver = (
        data.get("maneuver_type")
        or data.get("maneuver")
        or data.get("flight_pattern")
        or (data.get("drone") or {}).get("maneuver_type")
        or None
    )

    return {
        "azimuth_deg":   az,
        "distance_m":    dist,
        "height_m":      ht,
        "has_position":  (az is not None and dist is not None),
        "maneuver_type": maneuver,
        "source":        "real",
        "extra":         data,
    }


def _load_multichannel_bytes(
    audio_bytes: bytes,
    native_sr: int,
    mic_indices: List[int],
    cfg,
    ap,
) -> Optional[List[np.ndarray]]:
    """
    Write audio_bytes to a temp file, load the requested channels,
    resample to cfg.SR, and pad/truncate to cfg.TARGET_DURATION.

    Returns a list of 3 float32 arrays, or None on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = tf.name
        try:
            channels = ap.load_channels(tmp_path, channel_indices=mic_indices)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not channels:
            return None

        # Pad / truncate each channel to exactly TARGET_DURATION
        out = [ap.pad_or_truncate(ch) for ch in channels]

        # Ensure we have exactly 3 channels (replicate last if needed)
        while len(out) < 3:
            out.append(out[-1].copy())

        return out[:3]

    except Exception as exc:
        log.debug("_load_multichannel_bytes failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Session indexer (remote ZIP central directory)
# ══════════════════════════════════════════════════════════════════════════════

def _index_remote_sessions(
    rz,
    dataset_type: str,
    required_split: Optional[str],
) -> List[Dict]:
    """
    Scan the RemoteZip central directory and return a list of session dicts:
        {session_id, audio_path, label_path, split}

    The split is inferred from path components (train/val/test) when the
    remote ZIP was created with split sub-folders; otherwise it is None.

    For Dunakeszi ZIPs that contain the pipeline-ready triplet layout
    (<stem>_ch0.wav / _ch1.wav / _ch2.wav / <stem>_label.json), each triplet
    is returned as a separate session with is_dunakeszi=True so that
    _load_multichannel_bytes can handle it correctly.
    """
    all_names = rz.namelist()
    norm = [n.replace("\\", "/") for n in all_names]

    # ── Dunakeszi pipeline-ready layout: flat triplets ────────────────────────
    if dataset_type == "dunakeszi":
        label_paths = [p for p in norm if p.endswith("_label.json")]
        sessions: List[Dict] = []
        for lp in label_paths:
            stem_path = lp[: -len("_label.json")]   # e.g. "dunakeszi/seg_0001"
            stem      = stem_path.split("/")[-1]     # e.g. "seg_0001"
            ch0_path  = f"{stem_path}_ch0.wav"
            if ch0_path not in norm:
                continue   # no matching audio → skip

            # Infer split from path components
            split = None
            for part in stem_path.split("/"):
                if part in ("train", "val", "test", "validation"):
                    split = "val" if part == "validation" else part
                    break

            if required_split is not None and split != required_split and split is not None:
                continue

            sessions.append({
                "session_id":   stem,
                "audio_path":   ch0_path,   # loader will derive ch1/ch2 by stem
                "audio_stem":   stem,
                "audio_prefix": stem_path,  # full path prefix inside the ZIP
                "label_path":   lp,
                "split":        split,
                "is_dunakeszi": True,
            })

        if sessions:
            log.info(
                "Remote ZIP (dunakeszi): %d pipeline-ready triplets found (split=%s)",
                len(sessions), required_split,
            )
            return sessions

        # No triplets found — the ZIP may be the raw polywav layout.
        # We cannot stream 4 GB polywavs via RemoteZip efficiently; log clearly
        # and return empty so the caller falls through to synthetic.
        raw_wavs = [p for p in norm if p.lower().endswith(".wav")]
        log.warning(
            "Remote ZIP (dunakeszi): no pipeline-ready triplets found. "
            "The ZIP contains %d .wav file(s). "
            "Expected '<stem>_label.json' + '<stem>_ch0.wav' pairs. "
            "Raw polywav streaming is not supported via RemoteZip — "
            "use Nextcloud range-request streaming (Strategy 0) instead.",
            len(raw_wavs),
        )
        return []

    # ── UaVirBASE / generic layout: output.wav + label.json in sub-folders ────
    # Group entries by their parent directory
    session_map: Dict[str, Dict] = {}
    for path_str in norm:
        p = Path(path_str)
        parts = p.parts
        if len(parts) < 2:
            continue
        # Use the immediate parent directory as the session key
        session_dir = "/".join(parts[:-1])
        entry = session_map.setdefault(session_dir, {
            "session_id": parts[-2] if len(parts) >= 3 else parts[0],
            "audio_path": None,
            "label_path": None,
            "split":      None,
        })
        fname = parts[-1].lower()
        if fname in _AUDIO_CANDIDATES:
            entry["audio_path"] = path_str
        elif fname in _LABEL_CANDIDATES:
            entry["label_path"] = path_str

        # Try to read split from path components
        for comp in parts:
            if comp in ("train", "val", "test", "validation"):
                entry["split"] = "val" if comp == "validation" else comp
                break

    # Keep only paired sessions (audio + label)
    paired = [
        s for s in session_map.values()
        if s["audio_path"] and s["label_path"]
    ]

    # Filter by required split
    if required_split is not None:
        paired = [s for s in paired if s["split"] == required_split or s["split"] is None]

    log.info("Remote ZIP: %d paired sessions found (split=%s)", len(paired), required_split)
    return paired


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — RemoteZip streaming
# ══════════════════════════════════════════════════════════════════════════════

def _stream_via_remotezip(
    url: str,
    cfg,
    ap,
    dataset_type: str,
    mic_indices: List[int],
    native_sr: int,
    required_split: Optional[str],
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Indefinitely stream sessions from a remote ZIP without downloading it.

    On each pass through the session list the order is shuffled so the
    live dashboard sees variety even on short datasets.
    """
    _ensure_remotezip()
    from remotezip import RemoteZip  # type: ignore

    RZ_KWARGS = {"initial_buffer_size": 64 * 1024 * 1024}

    log.info("Opening remote ZIP: %s", url)
    with RemoteZip(url, **RZ_KWARGS) as rz:
        sessions = _index_remote_sessions(rz, dataset_type, required_split)

    if not sessions:
        raise RuntimeError(
            f"No paired sessions found in remote ZIP at {url!r} "
            f"(split={required_split!r})"
        )

    log.info("Streaming %d sessions from remote ZIP (looping)", len(sessions))

    while True:
        random.shuffle(sessions)
        with RemoteZip(url, **RZ_KWARGS) as rz:
            for sess in sessions:
                try:
                    label_raw   = rz.read(sess["label_path"])
                    label_meta  = _parse_label_bytes(label_raw)
                    if label_meta is None:
                        continue  # ambient / unreadable

                    # ── Dunakeszi pipeline-ready triplet ──────────────────────
                    if sess.get("is_dunakeszi"):
                        prefix = sess["audio_prefix"]   # e.g. "dir/seg_0001"
                        out_chs: List[np.ndarray] = []
                        for ch_idx in range(3):
                            ch_path = f"{prefix}_ch{ch_idx}.wav"
                            try:
                                ch_bytes = rz.read(ch_path)
                            except Exception:
                                break
                            # Each file is mono; load and resample
                            import io as _io
                            import soundfile as _sf
                            data, sr = _sf.read(_io.BytesIO(ch_bytes), dtype="float32",
                                                always_2d=False)
                            if sr != cfg.SR:
                                import librosa as _lr
                                data = _lr.resample(data, orig_sr=sr, target_sr=cfg.SR)
                            from .audio_processing import AudioProcessor as _AP
                            ap_local = _AP(cfg)
                            out_chs.append(ap_local.pad_or_truncate(data))
                        if len(out_chs) == 0:
                            continue
                        while len(out_chs) < 3:
                            out_chs.append(out_chs[-1].copy())
                        channels = out_chs[:3]

                    # ── UaVirBASE / generic single-file session ───────────────
                    else:
                        audio_bytes = rz.read(sess["audio_path"])
                        channels    = _load_multichannel_bytes(
                            audio_bytes, native_sr, mic_indices, cfg, ap
                        )
                        if channels is None:
                            continue

                    extra = label_meta.get("extra", {}) or {}
                    # Dunakeszi-specific rich metadata from make_label_json()
                    label = {
                        "segment_id":    sess["session_id"],
                        "split":         sess["split"],
                        "source":        "real",
                        "dataset_type":  dataset_type,
                        "azimuth_deg":   label_meta["azimuth_deg"],
                        "distance_m":    label_meta["distance_m"],
                        "height_m":      label_meta["height_m"],
                        "has_position":  label_meta["has_position"],
                        "maneuver_type": label_meta["maneuver_type"],
                        # Dunakeszi extras (None for UaVirBASE)
                        "session":       extra.get("session"),
                        "array":         extra.get("array"),
                        "n_drones":      extra.get("n_drones"),
                        "speed_mps":     extra.get("speed_mps"),
                        "radius_m":      extra.get("radius_m"),
                        "duration_s":    extra.get("duration_s"),
                        "clip_start_s":  extra.get("clip_start_s_in_seg"),
                        "flight_phase":  extra.get("flight_phase"),
                        # Trajectory waypoints for GPS map overlay
                        "trajectory":    extra.get("trajectory"),
                    }
                    yield channels, label

                except Exception as exc:
                    log.debug("Session %s skipped: %s", sess["session_id"], exc)
                    continue


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — Full ZIP download + local iteration
# ══════════════════════════════════════════════════════════════════════════════

def _stream_from_local_zip(
    *args,
    **kwargs,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    DISABLED — full ZIP downloads are not permitted on this server.

    Dataset archives (UaVirBASE ≈ 4 GB, Dunakeszi ≈ 4 GB per chunk) are
    too large to store on a deployment server.  Raises immediately so the
    caller falls through to the synthetic fallback.

    For offline / local testing with pre-extracted data, use
    _stream_from_local_extracted_dir() directly with a pre-extracted path.
    """
    raise RuntimeError(
        "Full ZIP download is disabled (live server storage policy). "
        "Only RemoteZip streaming or synthetic fallback are permitted."
    )
    yield  # makes this a generator function; never reached



def _index_local_sessions(
    root: Path,
    dataset_type: str,
    required_split: Optional[str],
) -> List[Dict]:
    """
    Walk the extracted directory and build a list of session dicts.

    Supports two layouts:

    1. UaVirBASE layout  (one sub-folder per session):
          <session_id>/output.wav  +  <session_id>/label.json

    2. Dunakeszi pipeline-ready layout  (flat directory, triplets):
          <stem>_ch0.wav  +  <stem>_ch1.wav  +  <stem>_ch2.wav  +  <stem>_label.json
       All three channel files are stored; _ch0 is selected as the primary audio
       path (the loader picks all three explicitly by mic_indices in the caller).
       A session record is emitted for each unique <stem>.
    """
    sessions: List[Dict] = []

    # ── Layout 2: Dunakeszi flat triplets ─────────────────────────────────────
    if dataset_type == "dunakeszi":
        # Collect all _label.json files directly under root (or one level deep)
        label_files = sorted(root.rglob("*_label.json"))
        for label_file in label_files:
            stem = label_file.name[: -len("_label.json")]   # e.g. "seg_0001"
            d    = label_file.parent

            # Require at least _ch0.wav to exist
            ch0 = d / f"{stem}_ch0.wav"
            if not ch0.exists():
                continue

            label_meta = _parse_label_bytes(label_file.read_bytes())
            if label_meta is None:
                continue

            split = None
            for part in d.parts:
                if part in ("train", "val", "test", "validation"):
                    split = "val" if part == "validation" else part
                    break

            if required_split is not None and split != required_split and split is not None:
                continue

            sessions.append({
                "session_id": stem,
                # Store the directory + stem so _load_channels_from_disk_dunakeszi
                # can load all three channels by index.  We encode this as the
                # ch0 path; the loader detects _ch0/_ch1/_ch2 via the stem.
                "audio_path":    str(ch0),
                "audio_dir":     str(d),
                "audio_stem":    stem,
                "label_meta":    label_meta,
                "split":         split,
                "is_dunakeszi":  True,
            })

        if sessions:
            log.info(
                "Dunakeszi local index: %d triplet sessions found under %s",
                len(sessions), root,
            )
            return sessions

        # Fall through to generic layout scan if no triplets found
        log.warning(
            "No *_label.json + *_ch0.wav triplets found under %s — "
            "trying generic layout scan", root,
        )

    # ── Layout 1: generic output.wav + label.json sub-folder layout ───────────
    for session_dir in sorted(root.rglob("*")):
        if not session_dir.is_dir():
            continue
        files      = {f.name.lower(): f for f in session_dir.iterdir() if f.is_file()}
        audio_name = next((n for n in _AUDIO_CANDIDATES if n in files), None)
        label_name = next((n for n in _LABEL_CANDIDATES if n in files), None)
        if audio_name is None or label_name is None:
            continue

        label_meta = _parse_label_bytes(files[label_name].read_bytes())
        if label_meta is None:
            continue

        split = None
        for part in session_dir.parts:
            if part in ("train", "val", "test", "validation"):
                split = "val" if part == "validation" else part
                break

        if required_split is not None and split != required_split and split is not None:
            continue

        sessions.append({
            "session_id": session_dir.name,
            "audio_path": str(files[audio_name]),
            "label_meta": label_meta,
            "split":      split,
        })
    return sessions


def _load_channels_from_disk(
    audio_path: str,
    mic_indices: List[int],
    cfg,
    ap,
    session: Optional[Dict] = None,
) -> Optional[List[np.ndarray]]:
    """
    Load a multi-channel file from disk, select channels, pad/truncate.

    For Dunakeszi pipeline-ready sessions (is_dunakeszi=True), the segment
    has already been extracted into three mono files:
        <stem>_ch0.wav, <stem>_ch1.wav, <stem>_ch2.wav
    These are loaded directly (no channel-index slicing needed).

    For UaVirBASE / generic sessions the original multi-channel WAV is sliced
    by mic_indices as before.
    """
    try:
        # ── Dunakeszi triplet: load three pre-extracted mono files ─────────
        if session and session.get("is_dunakeszi"):
            d    = Path(session["audio_dir"])
            stem = session["audio_stem"]
            out  = []
            for ch_idx in range(3):
                ch_path = d / f"{stem}_ch{ch_idx}.wav"
                if not ch_path.exists():
                    break
                data, sr = sf.read(str(ch_path), dtype="float32", always_2d=False)
                # Resample if needed
                if sr != cfg.SR:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=cfg.SR)
                out.append(ap.pad_or_truncate(data))
            if len(out) == 0:
                return None
            while len(out) < 3:
                out.append(out[-1].copy())
            return out[:3]

        # ── Generic / UaVirBASE: slice mic channels from polywav ───────────
        channels = ap.load_channels(audio_path, channel_indices=mic_indices)
        if not channels:
            return None
        out = [ap.pad_or_truncate(ch) for ch in channels]
        while len(out) < 3:
            out.append(out[-1].copy())
        return out[:3]

    except Exception as exc:
        log.debug("_load_channels_from_disk failed (%s): %s", audio_path, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3 — Synthetic fallback
# ══════════════════════════════════════════════════════════════════════════════

def _stream_synthetic(
    cfg,
    n_synthetic: int,
    max_dist: float,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Generate synthetic drone segments via synthesise_drone() and yield them
    in a loop.  Positions are drawn uniformly on a disk of radius max_dist/2.
    """
    from .audio_processing import synthesise_drone

    log.warning(
        "Repository unreachable — falling back to %d synthetic segments", n_synthetic
    )

    DRONE_TYPES   = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]
    DRONE_WEIGHTS = [0.30, 0.30, 0.25, 0.15]
    cx, cy        = float(cfg.ARRAY_CENTER[0]), float(cfg.ARRAY_CENTER[1])
    max_r         = max_dist / 2.0

    segments = []
    for i in range(n_synthetic):
        r      = random.uniform(1.0, max_r)
        theta  = random.uniform(0, 2 * math.pi)
        src_xy = [cx + r * math.cos(theta), cy + r * math.sin(theta)]

        drone_type = random.choices(DRONE_TYPES, weights=DRONE_WEIGHTS)[0]
        channels   = synthesise_drone(
            mic_positions = cfg.MIC_POSITIONS,
            src_xy        = src_xy,
            drone_type    = drone_type,
            noise_profile = "mixed",
            cfg           = cfg,
        )
        az  = math.degrees(math.atan2(src_xy[1] - cy, src_xy[0] - cx))
        label = {
            "segment_id":    f"synthetic_{i:05d}",
            "split":         None,
            "source":        "synthetic",
            "azimuth_deg":   round(az, 2),
            "distance_m":    round(r, 3),
            "height_m":      round(random.uniform(2.0, 20.0), 2),
            "has_position":  True,
            "maneuver_type": "synthetic_hover",
        }
        segments.append((channels, label))

    log.info("Synthetic fallback: generated %d segments", len(segments))

    while True:
        random.shuffle(segments)
        for channels, label in segments:
            # Re-synthesise with fresh noise on each loop pass so the audio
            # is not byte-identical every cycle
            src_xy     = [
                cx + label["distance_m"] * math.cos(math.radians(label["azimuth_deg"])),
                cy + label["distance_m"] * math.sin(math.radians(label["azimuth_deg"])),
            ]
            drone_type = random.choices(DRONE_TYPES, weights=DRONE_WEIGHTS)[0]
            fresh_chs  = synthesise_drone(
                mic_positions = cfg.MIC_POSITIONS,
                src_xy        = src_xy,
                drone_type    = drone_type,
                noise_profile = "mixed",
                cfg           = cfg,
            )
            yield fresh_chs, label


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def _stream_from_local_extracted_dir(
    root_path: str,
    cfg,
    ap,
    dataset_type: str,
    mic_indices: List[int],
    required_split: Optional[str],
    segment_id: Optional[int] = None,
    loop: bool = True,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Stream sessions from a pre-extracted directory structure.

    Handles both layouts automatically:
    - UaVirBASE: sub-folder per session with output.wav + label.json
    - Dunakeszi: flat directory with <stem>_ch0/1/2.wav + <stem>_label.json

    Parameters
    ----------
    segment_id : if set, only yield the session matching this integer id
    loop       : if True (default), loop indefinitely; if False, yield each
                 session once then return (generator exhausted)
    """
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    sessions = _index_local_sessions(root, dataset_type, required_split)
    if not sessions:
        raise RuntimeError(f"No valid sessions found in {root_path}")

    # Filter to a single segment when segment_id is specified
    if segment_id is not None:
        def _match(sess):
            name = sess["session_id"]
            # Try numeric suffix match
            import re as _re
            m = _re.search(r"(\d+)$", name)
            return int(m.group(1)) == segment_id if m else False
        sessions = [s for s in sessions if _match(s)]
        if not sessions:
            all_ids = sorted(set(
                int(m.group(1))
                for s in _index_local_sessions(root, dataset_type, None)
                for m in [__import__("re").search(r"(\d+)$", s["session_id"])]
                if m
            ))
            raise RuntimeError(
                f"Segment id={segment_id} not found in {root_path}. "
                f"Available ids: {all_ids}"
            )
        log.info("Single-segment mode: playing segment id=%d (%d match(es))",
                 segment_id, len(sessions))
        # Always play once for a specific segment selection
        loop = False

    log.info(
        "Streaming %d sessions from extracted directory: %s (loop=%s)",
        len(sessions), root_path, loop,
    )

    first_pass = True
    while True:
        if not first_pass:
            if not loop:
                log.info("Segment playback complete (loop=False) — generator exhausted")
                return
            random.shuffle(sessions)
        first_pass = False

        for sess in sessions:
            channels = _load_channels_from_disk(
                sess["audio_path"], mic_indices, cfg, ap, session=sess
            )
            if channels is None:
                continue

            label_meta = sess["label_meta"]
            extra      = label_meta.get("extra", {}) or {}
            label = {
                "segment_id":    sess["session_id"],
                "split":         sess["split"],
                "source":        "real",
                "dataset_type":  dataset_type,
                "azimuth_deg":   label_meta["azimuth_deg"],
                "distance_m":    label_meta["distance_m"],
                "height_m":      label_meta["height_m"],
                "has_position":  label_meta["has_position"],
                "maneuver_type": label_meta["maneuver_type"],
                # Dunakeszi-specific extras (None for UaVirBASE)
                "session":       extra.get("session"),
                "array":         extra.get("array"),
                "n_drones":      extra.get("n_drones"),
                "speed_mps":     extra.get("speed_mps"),
                "radius_m":      extra.get("radius_m"),
                "duration_s":    extra.get("duration_s"),
                "clip_start_s":  extra.get("clip_start_s_in_seg"),
                "flight_phase":  extra.get("flight_phase"),
                "trajectory":    extra.get("trajectory"),
                "audio_file":    Path(sess["audio_path"]).name,
            }
            yield channels, label

        if not loop:
            log.info("All %d sessions played once (loop=False) — generator exhausted",
                     len(sessions))
            return


def stream_repository_segments(
    cfg,
    url:                      Optional[str]  = None,
    dataset_type:             str            = "uavirbase",
    array:                    str            = "BK-6-E",
    max_dist:                 float          = 100.0,
    required_split:           Optional[str]  = None,
    allow_download:           bool           = True,   # accepted but IGNORED — see storage policy
    allow_synthetic_fallback: bool           = True,
    n_synthetic:              int            = 200,
    cache_zip:                Optional[str]  = None,   # accepted but IGNORED — see storage policy
    segment_id:               Optional[int]  = None,   # play only this GT segment id
    loop:                     bool           = True,    # False → stop after one full pass
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Yield (channels, label) tuples streamed directly from an online repository.

    The generator loops indefinitely, re-shuffling each pass, so a live
    dashboard session never runs dry.

    STORAGE POLICY (live server)
    ────────────────────────────
    No archive files are ever written to disk.  Data access is exclusively
    via RemoteZip HTTP range-requests, which fetch only the bytes needed for
    each individual session (~100 KB) without storing the multi-gigabyte ZIP.

    `allow_download` and `cache_zip` are accepted for API compatibility with
    other callers (notebooks, offline scripts) but are silently ignored here.
    If RemoteZip streaming fails the function falls straight through to the
    synthetic fallback — it will NEVER trigger a download.

    Fallback chain
    ──────────────
    1. RemoteZip streaming   (no disk writes, HTTP range-requests only)
    2. Synthetic data        (when allow_synthetic_fallback=True and
                              the repository is unreachable)
    → RuntimeError           (if both above fail / are disabled)

    Parameters
    ──────────
    cfg                      : drone_detection Config instance
    url                      : Remote ZIP URL; None → cfg.UAVIRBASE_ZIP_URL
    dataset_type             : "uavirbase" | "dunakeszi"
    array                    : Dunakeszi array group (ignored for UaVirBASE)
    max_dist                 : Max source distance (m) for synthetic positions
    required_split           : Restrict to "train" | "val" | "test" | None
    allow_download           : Accepted for compatibility — ignored on server
    allow_synthetic_fallback : Yield synthetic data if repo unreachable
    n_synthetic              : Pool size for synthetic fallback
    cache_zip                : Accepted for compatibility — ignored on server

    Yields
    ──────
    channels : List[np.ndarray]  — 3 float32 arrays (cfg.SR × cfg.TARGET_DURATION,)
    label    : dict              — {segment_id, split, source, azimuth_deg,
                                    distance_m, height_m, has_position, maneuver_type}
    """
    if allow_download or cache_zip:
        log.debug(
            "allow_download/cache_zip ignored — live server storage policy "
            "prohibits ZIP downloads (datasets are 4 GB+). "
            "Using RemoteZip streaming only."
        )

    # ── Resolve mic_indices, ap, and native_sr FIRST ──────────────────────────
    # These must be defined before any early-return branch (extracted dir, etc.)
    # that calls _stream_from_local_extracted_dir() or _load_channels_from_disk().
    if dataset_type not in ("uavirbase", "dunakeszi", "mems"):
        log.warning("Unknown dataset_type=%r — treating as 'uavirbase'", dataset_type)
        dataset_type = "uavirbase"

    if dataset_type == "dunakeszi":
        _ARRAY_CH  = getattr(cfg, "DUNAKESZI_ARRAY_CHANNELS",
                             {"BK-6-E": [8, 9, 10], "BK-6-W": [2, 3, 4]})
        mic_indices = list(_ARRAY_CH.get(array, [8, 9, 10]))
    else:
        mic_indices = list(getattr(cfg, "UAVIRBASE_MIC_INDICES", [0, 1, 3]))

    native_sr = _native_sr_for(dataset_type, cfg)

    # Lazy-import to avoid circular imports at module load time
    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    # ── Resolve local path info (used by Strategy 2 fallback below) ─────────────
    local_zip_path = None
    extracted_path = None
    if dataset_type == "dunakeszi":
        local_zip_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)

        # DUNAKESZI_LOCAL_PATH may point directly to the extracted directory
        # (not a .zip file), so we handle both cases:
        #   a) it IS a directory → use it directly as extracted_path
        #   b) it IS a .zip file → derive extracted_path by stripping .zip
        if local_zip_path:
            if os.path.isdir(local_zip_path):
                extracted_path = local_zip_path          # (a) already extracted
            elif os.path.isfile(local_zip_path):
                extracted_path = local_zip_path.replace(".zip", "")  # (b) zip
                if not os.path.isdir(extracted_path):
                    extracted_path = None  # extracted dir doesn't exist yet
                # Queue the local ZIP for RemoteZip fallback
                if not url:
                    url = f"file://{os.path.abspath(local_zip_path)}"

    # ── Check whether Nextcloud is configured ────────────────────────────────
    _nc_configured = bool(
        dataset_type == "dunakeszi"
        and getattr(cfg, "NEXTCLOUD_BASE_URL", None)
        and getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", None)
    )

    # ── Strategy 0: Nextcloud HTTP range-request streaming ───────────────────
    # Tried FIRST when Nextcloud credentials are present, so that the live
    # polywav files are used instead of the local pipeline-ready directory.
    if _nc_configured:
        log.info(
            "Nextcloud configured — attempting Nextcloud range-request stream "
            "(local dir will be used as fallback only)"
        )
        try:
            from .dunakeszi_nextcloud import iter_nextcloud_segments
        except ImportError:
            try:
                from dunakeszi_nextcloud import iter_nextcloud_segments
            except ImportError:
                log.warning(
                    "dunakeszi_nextcloud not importable — skipping Nextcloud strategy"
                )
                iter_nextcloud_segments = None  # type: ignore

        if iter_nextcloud_segments is not None:
            try:
                gen   = iter_nextcloud_segments(cfg, ap, required_split=required_split,
                                                segment_id=segment_id, loop=loop)
                first = next(gen)
                log.info("Nextcloud streaming: first segment OK — live stream active")
                yield first
                yield from gen
                return
            except StopIteration:
                log.warning("Nextcloud generator yielded nothing — falling back")
            except Exception as exc:
                log.warning("Nextcloud streaming failed: %s — trying next strategy", exc)

    # ── Strategy 0b: local polywav directory ─────────────────────────────────
    # Used when DUNAKESZI_LOCAL_POLYWAV_DIR is set (raw polywav files downloaded
    # locally).  Applies identical segment metadata + byte-seek logic as the
    # Nextcloud strategy but reads from disk — no credentials or network needed.
    _local_polywav_dir = getattr(cfg, "DUNAKESZI_LOCAL_POLYWAV_DIR", None)
    if dataset_type == "dunakeszi" and _local_polywav_dir and os.path.isdir(_local_polywav_dir):
        log.info(
            "Local polywav directory configured — streaming from disk: %s",
            _local_polywav_dir,
        )
        try:
            from .dunakeszi_nextcloud import iter_local_polywav_segments
        except ImportError:
            try:
                from dunakeszi_nextcloud import iter_local_polywav_segments
            except ImportError:
                log.warning(
                    "dunakeszi_nextcloud not importable — skipping local polywav strategy"
                )
                iter_local_polywav_segments = None  # type: ignore

        if iter_local_polywav_segments is not None:
            try:
                gen = iter_local_polywav_segments(
                    cfg, ap,
                    local_polywav_dir = _local_polywav_dir,
                    required_split    = required_split,
                    segment_id        = segment_id,
                    loop              = loop,
                    array             = array,
                )
                first = next(gen)
                log.info(
                    "Local polywav streaming: first segment OK — live stream active"
                )
                yield first
                yield from gen
                return
            except StopIteration:
                log.warning("Local polywav generator yielded nothing — falling back")
            except Exception as exc:
                log.warning(
                    "Local polywav streaming failed: %s — trying next strategy", exc
                )

    def _mono_to_3ch(mono: np.ndarray, native_sr: int, cfg, ap) -> List[np.ndarray]:
        if native_sr != cfg.SR:
            import librosa
            mono = librosa.resample(mono, orig_sr=native_sr, target_sr=cfg.SR)
        y = ap.pad_or_truncate(mono)
        return [y, y, y]
             
    # ── Strategy 0-MEMS: MEMS Nextcloud streaming ────────────────────────────
    if dataset_type == "mems":
        _nc_configured_mems = bool(
            getattr(cfg, "NEXTCLOUD_BASE_URL", None)
            and getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", None)
        )
        if _nc_configured_mems:
            try:
                from .dunakeszi_nextcloud import iter_mems_segments
            except ImportError:
                from dunakeszi_nextcloud import iter_mems_segments
            try:
                gen = iter_mems_segments(cfg, required_split=required_split,
                                          segment_id=segment_id, loop=loop)
                mono, native_sr, label = next(gen)
                yield _mono_to_3ch(mono, native_sr, cfg, ap), label
                for mono, native_sr, label in gen:
                    yield _mono_to_3ch(mono, native_sr, cfg, ap), label
                return
            except StopIteration:
                log.warning("MEMS generator yielded nothing — falling back to synthetic")
            except Exception as exc:
                log.warning("MEMS streaming failed: %s — falling back to synthetic", exc)

        if allow_synthetic_fallback:
            log.warning("MEMS repository unavailable — using synthetic fallback (mono-style)")
            yield from _stream_synthetic(cfg, n_synthetic, max_dist)
            return
        raise RuntimeError("MEMS streaming unavailable and allow_synthetic_fallback=False")

    # ── Strategy 1: local pre-extracted directory (Dunakeszi pipeline-ready) ──
    # Used when Nextcloud is not configured OR after Nextcloud fails.
    if extracted_path and os.path.isdir(extracted_path):
        log.info(
            "Using local extracted Dunakeszi directory%s: %s",
            " (Nextcloud fallback)" if _nc_configured else "",
            extracted_path,
        )
        try:
            gen = _stream_from_local_extracted_dir(
                extracted_path, cfg, ap, dataset_type, mic_indices, required_split,
                segment_id=segment_id, loop=loop,
            )
            first = next(gen)
            yield first
            yield from gen
            return
        except Exception as exc:
            log.warning("Failed to use extracted directory: %s", exc)

    # ── Resolve the URL for RemoteZip (Strategy 2) ───────────────────────────
    if url is None:
        if dataset_type == "dunakeszi":
            url = getattr(cfg, "DUNAKESZI_ZIP_URL", None)
            if url is None and local_zip_path is None:
                log.warning(
                    "dataset_type='dunakeszi' but no DUNAKESZI_ZIP_URL in config, "
                    "no local path, and no url= passed explicitly. "
                    "Falling back to synthetic data."
                )
        else:
            url = getattr(cfg, "UAVIRBASE_ZIP_URL", None)

    if url is None and local_zip_path is None:
        if allow_synthetic_fallback:
            log.warning(
                "No repository URL or local file for dataset_type=%r — "
                "using synthetic fallback", dataset_type
            )
            yield from _stream_synthetic(cfg, n_synthetic, max_dist)
            return
        raise RuntimeError(
            f"No repository URL or local file configured for dataset_type={dataset_type!r} "
            "and allow_synthetic_fallback=False. "
            "Set cfg.DUNAKESZI_ZIP_URL, cfg.DUNAKESZI_LOCAL_PATH, or pass url= explicitly."
        )

    # ── Strategy 2: RemoteZip streaming (supports local file:// URLs) ────────
    if url is not None:
        try:
            log.info("Attempting RemoteZip streaming from %s", url)
            gen = _stream_via_remotezip(
                url            = url,
                cfg            = cfg,
                ap             = ap,
                dataset_type   = dataset_type,
                mic_indices    = mic_indices,
                native_sr      = native_sr,
                required_split = required_split,
            )
            first = next(gen)
            log.info("RemoteZip streaming: first segment OK — live stream active")
            yield first
            yield from gen
            return
        except StopIteration:
            log.warning("RemoteZip generator yielded nothing — falling back to synthetic")
        except Exception as exc:
            log.warning("RemoteZip streaming failed: %s — falling back to synthetic", exc)

    # ── Strategy 3: Synthetic fallback (no network, no disk) ─────────────────
    if allow_synthetic_fallback:
        log.warning(
            "All real-data strategies failed — streaming %d synthetic segments. "
            "Results will not reflect real recordings.",
            n_synthetic,
        )
        yield from _stream_synthetic(cfg, n_synthetic, max_dist)
        return

    raise RuntimeError(
        f"All streaming strategies failed for dataset_type={dataset_type!r} "
        f"(Nextcloud configured={_nc_configured}, url={url!r}, "
        f"local={extracted_path!r}) and allow_synthetic_fallback=False."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Public API — segment browser (for dashboard "select files from repository")
# ══════════════════════════════════════════════════════════════════════════════

def list_repository_segments(
    cfg,
    dataset_type: str = "dunakeszi",
    array: str = "BK-6-E",
    required_split: Optional[str] = None,
) -> List[Dict]:
    """
    Return a list of available segment dicts from the local dataset directory.

    Each dict contains:
        segment_id, split, has_position, azimuth_deg, distance_m, height_m,
        maneuver_type, source, dataset_type
        + Dunakeszi extras: session, array, n_drones, flight_phase, ...

    This is used by the dashboard to populate a segment-selection dropdown
    so the user can replay a specific recording rather than a random shuffle.

    Returns an empty list if no local dataset is configured / found.
    """
    if dataset_type == "dunakeszi":
        local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
    else:
        local_path = str(getattr(cfg, "UAVIRBASE_RAW", ""))

    if not local_path or not os.path.isdir(local_path):
        log.info(
            "list_repository_segments: no local directory for dataset_type=%r",
            dataset_type,
        )
        return []

    root = Path(local_path)
    sessions = _index_local_sessions(root, dataset_type, required_split)

    result = []
    for sess in sessions:
        lm    = sess["label_meta"]
        extra = lm.get("extra", {}) or {}
        result.append({
            "segment_id":    sess["session_id"],
            "split":         sess["split"],
            "dataset_type":  dataset_type,
            "source":        "real",
            "has_position":  lm["has_position"],
            "azimuth_deg":   lm["azimuth_deg"],
            "distance_m":    lm["distance_m"],
            "height_m":      lm["height_m"],
            "maneuver_type": lm["maneuver_type"],
            # Dunakeszi extras
            "session":       extra.get("session"),
            "array":         extra.get("array"),
            "n_drones":      extra.get("n_drones"),
            "flight_phase":  extra.get("flight_phase"),
            "speed_mps":     extra.get("speed_mps"),
        })

    log.info(
        "list_repository_segments: %d segments available for dataset_type=%r",
        len(result), dataset_type,
    )
    return result


def stream_single_segment(
    segment_id: str,
    cfg,
    dataset_type: str = "dunakeszi",
    array: str = "BK-6-E",
) -> Optional[Tuple[List[np.ndarray], dict]]:
    """
    Load and return a single segment by segment_id.

    Used by the dashboard when the user selects a specific segment from the
    browser rather than letting the system pick randomly.

    Returns (channels, label) or None if the segment is not found.
    """
    if dataset_type == "dunakeszi":
        local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
        _ARRAY_CH  = getattr(cfg, "DUNAKESZI_ARRAY_CHANNELS",
                             {"BK-6-E": [8, 9, 10], "BK-6-W": [2, 3, 4]})
        mic_indices = list(_ARRAY_CH.get(array, [8, 9, 10]))
    else:
        local_path  = str(getattr(cfg, "UAVIRBASE_RAW", ""))
        mic_indices = list(getattr(cfg, "UAVIRBASE_MIC_INDICES", [0, 1, 3]))

    if not local_path or not os.path.isdir(local_path):
        log.warning("stream_single_segment: no local directory for dataset_type=%r", dataset_type)
        return None

    root     = Path(local_path)
    sessions = _index_local_sessions(root, dataset_type, required_split=None)
    sess     = next((s for s in sessions if s["session_id"] == segment_id), None)
    if sess is None:
        log.warning("stream_single_segment: segment_id=%r not found", segment_id)
        return None

    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    channels = _load_channels_from_disk(
        sess["audio_path"], mic_indices, cfg, ap, session=sess
    )
    if channels is None:
        return None

    lm    = sess["label_meta"]
    extra = lm.get("extra", {}) or {}
    label = {
        "segment_id":    sess["session_id"],
        "split":         sess["split"],
        "source":        "real",
        "dataset_type":  dataset_type,
        "azimuth_deg":   lm["azimuth_deg"],
        "distance_m":    lm["distance_m"],
        "height_m":      lm["height_m"],
        "has_position":  lm["has_position"],
        "maneuver_type": lm["maneuver_type"],
        "session":       extra.get("session"),
        "array":         extra.get("array"),
        "n_drones":      extra.get("n_drones"),
        "flight_phase":  extra.get("flight_phase"),
        "trajectory":    extra.get("trajectory"),
    }
    return channels, label

# ══════════════════════════════════════════════════════════════════════════════
# 1. Rich segment browser
# ══════════════════════════════════════════════════════════════════════════════
 
def list_dunakeszi_segments_rich(
    cfg,
    split_filter:   Optional[str] = None,
    session_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return the full segment-browser payload for the Dunakeszi dataset.
 
    Used by GET /api/v2/repository/segments.
 
    Response shape (matches index_v2.html loadSegmentBrowser expectations):
    {
        "segments": [
            {
                "id":              int,
                "session":         str,    e.g. "show_8"
                "show_number":     int,
                "wall_clock":      str,    e.g. "14:41"
                "split":           str,    "train" | "val" | "test"
                "local_start_hms": str,    "14:41:00"
                "local_end_hms":   str,
                "duration_s":      float,
                "maneuver_type":   str,
                "flight_phase":    str,
                "description":     str,
                "n_drones":        int,
                "drones":          list[str],
                "altitude_m":      float | None,
                "speed_mps":       float | None,
                "radius_m":        float | None,
                "azimuth_deg":     float | None,   bearing from North at onset
                "distance_m":      float | None,   XY distance at onset
                "local_available": bool,    pipeline-ready triplet on disk
                "bk_available":    bool,    BK-6 polywav stream possible
                "mems_available":  bool,    MEMS file covers this segment
                "quality_flags":   list[str],
                "session_description": str,
            },
            …
        ],
        "sessions": [
            {
                "session_id":  str,
                "show_number": int,
                "wall_clock":  str,
                "description": str,
                "n_drones":    int,
            },
            …
        ],
        "total":  int,
        "source": str,   "ground_truth" | "local_index" | "empty"
    }
    """
    try:
        from .dunakeszi_nextcloud import build_segment_browser_response
    except ImportError:
        try:
            from dunakeszi_nextcloud import build_segment_browser_response
        except ImportError:
            log.warning("dunakeszi_nextcloud not importable — using local index fallback")
            return _list_from_local_only(cfg, split_filter, session_filter)
 
    return build_segment_browser_response(cfg, split_filter, session_filter)
 
 
def _list_from_local_only(
    cfg,
    split_filter:   Optional[str],
    session_filter: Optional[str],
) -> Dict[str, Any]:
    """
    Fallback: scan DUNAKESZI_LOCAL_PATH for *_label.json files and return a
    minimal browser payload.  Works without dunakeszi_nextcloud installed.
    """
    import json, re
 
    local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
    if not local_path or not os.path.isdir(local_path):
        return {"segments": [], "sessions": [], "total": 0, "source": "empty"}
 
    root = Path(local_path)
    sessions_seen: Dict[str, dict] = {}
    segments_out:  List[dict]      = []
 
    for lf in sorted(root.rglob("*_label.json")):
        try:
            raw = json.loads(lf.read_text())
        except Exception:
            continue
 
        stem = lf.name[: -len("_label.json")]
        if not (lf.parent / f"{stem}_ch0.wav").exists():
            continue
 
        split   = raw.get("split")
        session = raw.get("session") or "unknown"
        if split_filter   and split   != split_filter:   continue
        if session_filter and session != session_filter: continue
 
        m      = re.search(r"seg_(\d+)", stem)
        seg_id = int(m.group(1)) if m else None
        sessions_seen.setdefault(session, {"session_id": session, "description": session})
 
        segments_out.append({
            "id":             seg_id,
            "session":        session,
            "split":          split,
            "local_start_hms": None,
            "maneuver_type":  raw.get("maneuver_type"),
            "description":    raw.get("note") or stem,
            "n_drones":       raw.get("n_drones", 1),
            "altitude_m":     raw.get("height_m"),
            "azimuth_deg":    raw.get("original_bearing_deg"),
            "distance_m":     raw.get("distance_m"),
            "quality_flags":  [],
            "local_available": True,
            "bk_available":   True,
            "mems_available": False,
        })
 
    return {
        "segments": segments_out,
        "sessions": list(sessions_seen.values()),
        "total":    len(segments_out),
        "source":   "local_index",
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. Stream a single segment by ground-truth ID
# ══════════════════════════════════════════════════════════════════════════════
 
def stream_segment_by_gt_id(
    segment_id: int,
    cfg,
    array: str = "BK-6-E",
    ap=None,
) -> Optional[Tuple[List[np.ndarray], dict]]:
    """
    Load a Dunakeszi maneuver segment by its ground-truth integer ID.
 
    Fallback chain:
        1. Local pipeline-ready file (dunakeszi_pipeline_ready_B)
        2. Nextcloud HTTP range-request (requires NEXTCLOUD_* config)
        3. Returns None → caller uses synthetic fallback
 
    Parameters
    ----------
    segment_id : int  — ground-truth MANEUVER_SEGMENTS id
    cfg        : Config instance
    array      : "BK-6-E" | "BK-6-W"
    ap         : AudioProcessor instance (created if None)
 
    Returns
    -------
    (channels, label) or None
    """
    if ap is None:
        from .audio_processing import AudioProcessor
        ap = AudioProcessor(cfg)
 
    # ── Strategy 1: local pipeline-ready triplet ──────────────────────────────
    result = _stream_from_local_by_gt_id(segment_id, cfg, array, ap)
    if result is not None:
        log.info("segment_id=%d: served from local pipeline-ready files", segment_id)
        return result
 
    # ── Strategy 2: Nextcloud range-request ───────────────────────────────────
    nc_configured = bool(
        getattr(cfg, "NEXTCLOUD_BASE_URL", None)
        and getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", None)
    )
    if not nc_configured:
        log.warning(
            "segment_id=%d: Nextcloud not configured "
            "(NEXTCLOUD_BASE_URL=%r, NEXTCLOUD_SHARE_TOKEN=%s) — "
            "set env vars or pass nextcloud_url/nextcloud_token in the POST body",
            segment_id,
            getattr(cfg, "NEXTCLOUD_BASE_URL", None),
            "***" if getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", None) else None,
        )
    if nc_configured:
        try:
            from .dunakeszi_nextcloud import stream_segment_from_nextcloud
        except ImportError:
            try:
                from dunakeszi_nextcloud import stream_segment_from_nextcloud
            except ImportError:
                log.warning("dunakeszi_nextcloud not importable — skipping range-read strategy")
                return None
 
        result = stream_segment_from_nextcloud(cfg, segment_id, array=array, ap=ap)
        if result is not None:
            log.info("segment_id=%d: served from Nextcloud range-read", segment_id)
            return result
 
    log.warning("segment_id=%d: not found in local files or Nextcloud", segment_id)
    return None
 
 
def _stream_from_local_by_gt_id(
    segment_id: int,
    cfg,
    array: str,
    ap,
) -> Optional[Tuple[List[np.ndarray], dict]]:
    """
    Look for a pipeline-ready triplet whose stem starts with 'seg_NNNN' where
    NNNN matches the zero-padded segment_id.  Loads and returns it.
    """
    import json
    import soundfile as sf
 
    local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
    if not local_path or not os.path.isdir(local_path):
        return None
 
    root = Path(local_path)
    prefix = f"seg_{segment_id:04d}"
 
    # Find a label file whose stem starts with the prefix
    candidates = sorted(root.rglob(f"{prefix}*_label.json"))
    if not candidates:
        # Also try without leading zeros (older naming)
        candidates = sorted(root.rglob(f"seg_{segment_id}_*_label.json"))
    if not candidates:
        return None
 
    lf   = candidates[0]
    stem = lf.name[: -len("_label.json")]
 
    out_channels = []
    for ch_idx in range(3):
        ch_path = lf.parent / f"{stem}_ch{ch_idx}.wav"
        if not ch_path.exists():
            break
        try:
            data, sr = sf.read(str(ch_path), dtype="float32", always_2d=False)
            if sr != cfg.SR:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=cfg.SR)
            out_channels.append(ap.pad_or_truncate(data))
        except Exception as exc:
            log.debug("_stream_from_local_by_gt_id ch%d failed: %s", ch_idx, exc)
            break
 
    if not out_channels:
        return None
    while len(out_channels) < 3:
        out_channels.append(out_channels[-1].copy())
 
    try:
        raw = json.loads(lf.read_text())
    except Exception:
        raw = {}
 
    label = {
        "segment_id":    segment_id,
        "split":         raw.get("split"),
        "source":        "real",
        "dataset_type":  "dunakeszi",
        "array":         raw.get("array", array),
        "azimuth_deg":   raw.get("azimuth_deg"),
        "distance_m":    raw.get("distance_m"),
        "height_m":      raw.get("height_m"),
        "has_position":  raw.get("has_position", False),
        "maneuver_type": raw.get("maneuver_type"),
        "flight_phase":  raw.get("flight_phase"),
        "n_drones":      raw.get("n_drones", 1),
        "speed_mps":     raw.get("speed_mps"),
        "radius_m":      raw.get("radius_m"),
        "duration_s":    raw.get("duration_s"),
        "session":       raw.get("session"),
        "audio_file":    f"{stem}_ch0.wav",
    }
    return out_channels[:3], label
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. File browser tables (polywav / MEMS)
# ══════════════════════════════════════════════════════════════════════════════
 
def get_dunakeszi_file_browser(cfg, file_type: str = "polywav") -> Dict[str, Any]:
    """
    Return metadata table for the Dunakeszi file browser panel.
 
    file_type: "polywav" | "mems"
 
    Returns a dict with:
        files   : list of file metadata dicts
        total   : int
        type    : str
    """
    try:
        from .dunakeszi_nextcloud import polywav_file_info, mems_file_info
    except ImportError:
        try:
            from dunakeszi_nextcloud import polywav_file_info, mems_file_info
        except ImportError:
            return {"files": [], "total": 0, "type": file_type, "error": "dunakeszi_nextcloud not installed"}
 
    if file_type == "mems":
        files = mems_file_info(cfg)
    else:
        files = polywav_file_info(cfg)
 
    return {
        "files": files,
        "total": len(files),
        "type":  file_type,
    }