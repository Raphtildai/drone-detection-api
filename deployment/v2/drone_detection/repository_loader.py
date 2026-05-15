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
    """
    all_names = rz.namelist()
    norm = [n.replace("\\", "/") for n in all_names]

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
    """Walk the extracted directory and build a list of session dicts."""
    sessions = []
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
) -> Optional[List[np.ndarray]]:
    """Load a multi-channel file from disk, select channels, pad/truncate."""
    try:
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
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Stream sessions from a pre-extracted directory structure.
    """
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    
    sessions = _index_local_sessions(root, dataset_type, required_split)
    if not sessions:
        raise RuntimeError(f"No valid sessions found in {root_path}")
    
    log.info("Streaming %d sessions from extracted directory: %s", len(sessions), root_path)
    
    while True:
        random.shuffle(sessions)
        for sess in sessions:
            channels = _load_channels_from_disk(
                sess["audio_path"], mic_indices, cfg, ap
            )
            if channels is None:
                continue
            
            label_meta = sess["label_meta"]
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
            }
            yield channels, label

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

    # Check for local ZIP file first (for Dunakeszi)
    local_zip_path = None
    if dataset_type == "dunakeszi":
        local_zip_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
        # Check for extracted directory (without .zip extension)
        extracted_path = local_zip_path.replace('.zip', '') if local_zip_path else None
        if local_zip_path and os.path.isfile(local_zip_path):
            log.info(f"Using local Dunakeszi ZIP file: {local_zip_path}")
            # Use local ZIP via RemoteZip (which supports file:// URLs)
            if not url:
                url = f"file://{os.path.abspath(local_zip_path)}"

        if extracted_path and os.path.isdir(extracted_path):
            log.info(f"Using extracted Dunakeszi directory: {extracted_path}")
            try:
                gen = _stream_from_local_extracted_dir(
                    extracted_path, cfg, ap, dataset_type, mic_indices, required_split
                )
                first = next(gen)
                yield first
                yield from gen
                return
            except Exception as exc:
                log.warning(f"Failed to use extracted directory: {exc}")

    # Resolve the URL — use dataset-specific config key
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
            log.warning("No repository URL or local file for dataset_type=%r — using synthetic fallback", dataset_type)
            yield from _stream_synthetic(cfg, n_synthetic, max_dist)
            return
        raise RuntimeError(
            f"No repository URL or local file configured for dataset_type={dataset_type!r} "
            "and allow_synthetic_fallback=False. "
            "Set cfg.DUNAKESKI_ZIP_URL, cfg.DUNAKESZI_LOCAL_PATH, or pass url= explicitly."
        )

    # Resolve mic indices and dataset native SR
    mic_indices = list(getattr(cfg, "UAVIRBASE_MIC_INDICES", [0, 1, 3]))
    if dataset_type not in ("uavirbase", "dunakeszi"):
        log.warning("Unknown dataset_type=%r — treating as 'uavirbase'", dataset_type)
        dataset_type = "uavirbase"
    native_sr = _native_sr_for(dataset_type, cfg)

    # Lazy-import to avoid circular imports at module load time
    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    # ── Strategy 1: RemoteZip streaming (supports local file:// URLs) ──────────
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

    # ── Strategy 2: Synthetic fallback (no network, no disk) ──────────────────
    if allow_synthetic_fallback:
        log.warning(
            "Repository unreachable — streaming %d synthetic segments. "
            "Results will not reflect real recordings.",
            n_synthetic,
        )
        yield from _stream_synthetic(cfg, n_synthetic, max_dist)
        return

    raise RuntimeError(
        f"RemoteZip streaming failed for URL={url!r} and "
        "allow_synthetic_fallback=False. "
        "Check network connectivity and the repository URL."
    )