# -*- coding: utf-8 -*-
"""
repository_loader.py
────────────────────
Reads the Dunakeszi drone-audio dataset **directly from the online repository**
(Zenodo ZIP for UaVirBASE, or any HTTP-hosted ZIP for Dunakeszi-style data)
and runs each segment through the exact same processing pipeline that the
offline extractor + prepare scripts would have applied.

Fallback chain (in order)
──────────────────────────
1. Remote ZIP streaming  — reads individual files from the ZIP over HTTP
   without downloading the whole archive (requires `remotezip`).
2. Full download         — downloads the entire ZIP once, then processes locally.
3. Synthetic data        — if both network paths fail, generates synthetic
   3-channel segments using the drone BPF profiles in config.

Public API
──────────
load_repository_dataset(cfg, ...)
    → RepositoryDataset  (torch.utils.data.Dataset)

stream_repository_segments(cfg, ...)
    → generator of (channels: List[np.ndarray], label: dict)

prepare_repository_to_disk(cfg, output_dir, ...)
    → Path   (local directory in dunakeszi_pipeline_ready format)

RepositoryDataset
    torch Dataset wrapping stream_repository_segments(); can be passed
    directly to a DataLoader.

Design notes
────────────
- All label conversion is done in-process using the same logic as
  prepare_dunakeszi_for_pipeline.py (bearing_to_pipeline_az + convert_label).
- Channel extraction uses the same ARRAY_CHANNELS mapping as
  dunakeszi_segment_extractor_fixed.py.
- Audio resampling uses AudioProcessor.load() (librosa, float32, TARGET_SR).
- The module is self-contained: it only imports from the standard drone_detection
  package and the Python stdlib + numpy/soundfile.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Inline copies of the two pure-Python helpers from prepare_dunakeszi… and
# dunakeszi_segment_extractor…  so this module has zero coupling to those
# scripts being present on disk.
# ──────────────────────────────────────────────────────────────────────────────

# Channel indices for each Dunakeszi array (matches ARRAY_CHANNELS in extractor)
_ARRAY_CHANNELS: Dict[str, List[int]] = {
    "BK-6-E": [8, 9, 10],
    "BK-6-W": [2, 3, 4],
}
_DEFAULT_ARRAY = "BK-6-E"

# UaVirBASE: which of the 8-mic polywav channels to pick (0-indexed)
# ch1(N)=0, ch2(E)=1, ch4(W)=3  → same as config.UAVIRBASE_MIC_INDICES
_UAVIRBASE_MIC_INDICES = [0, 1, 3]

# Audio constants matching the pipeline
_NATIVE_SR = 192_000      # Dunakeszi polywav native sample rate
_TARGET_SR = 22_050       # pipeline sample rate

# Exact 4 GB polywav chunk duration
_BYTES_PER_FRAME  = 14 * 4                               # 14ch × float32
_CHUNK_DUR_S      = (4 * 1024 ** 3) / (_BYTES_PER_FRAME * _NATIVE_SR)  # ≈ 399.46 s


def _bearing_to_pipeline_az(bearing_deg: float) -> float:
    """Geographic bearing (from-North) → pipeline math angle (from-East)."""
    math_az = 90.0 - bearing_deg
    return (math_az + 180.0) % 360.0 - 180.0


def _convert_label(raw: dict, max_dist: float = 100.0) -> dict:
    """
    Convert a Dunakeszi extractor label dict to the pipeline flat format.
    Mirrors prepare_dunakeszi_for_pipeline.prepare_dunakeszi.convert_label().
    """
    drone    = raw.get("drone", {})
    bearing  = drone.get("azimuth")
    distance = drone.get("distance")
    height   = drone.get("height")

    # Support already-converted flat labels (azimuth_deg / distance_m / height_m)
    if bearing is None:
        bearing  = raw.get("azimuth_deg") or raw.get("azimuth") or raw.get("bearing")
    if distance is None:
        distance = raw.get("distance_m") or raw.get("distance") or raw.get("dist")
    if height is None:
        height   = raw.get("height_m") or raw.get("height") or raw.get("alt")

    has_position = (
        bearing is not None
        and distance is not None
        and height is not None
        and not (bearing == 0.0 and distance == 0.0)
    )

    if has_position:
        pipeline_az = _bearing_to_pipeline_az(float(bearing))
        distance_m  = float(distance)
        height_m    = float(height)
    else:
        pipeline_az = distance_m = height_m = None

    return {
        "azimuth_deg":          pipeline_az,
        "distance_m":           distance_m,
        "height_m":             height_m,
        "source":               raw.get("source", "dunakeszi"),
        "original_bearing_deg": float(bearing) if bearing is not None else None,
        "segment_id":           raw.get("segment_id"),
        "session":              raw.get("session"),
        "maneuver_type":        raw.get("maneuver_type"),
        "flight_phase":         raw.get("flight_phase"),
        "n_drones":             raw.get("n_drones"),
        "split":                raw.get("split"),
        "array":                raw.get("array"),
        "speed_mps":            raw.get("speed_mps"),
        "radius_m":             raw.get("radius_m"),
        "duration_s":           raw.get("duration_s"),
        "has_position":         has_position,
    }


def _parse_uavirbase_label(raw_bytes: bytes) -> Optional[Tuple[float, float, float]]:
    """
    Parse a UaVirBASE label.json, returning (az_deg, dist_m, height_m) or None.
    Matches parse_label_json() in datasets.py.
    """
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    drone = data.get("drone", {})
    if isinstance(drone, dict):
        src = drone.get("sound_source", "")
        if isinstance(src, str) and "ambient" in src.lower():
            return None
        try:
            az = drone.get("azimuth")
            di = drone.get("distance")
            ht = drone.get("height")
            if az is not None and di is not None and ht is not None:
                return float(az), float(di), float(ht)
        except (TypeError, ValueError):
            pass

    # Flat keys fallback
    _AZ   = ["azimuth_deg", "azimuth", "az"]
    _DIST = ["distance_m",  "distance", "dist"]
    _HT   = ["height_m",    "height",  "alt", "altitude"]
    az = next((data[k] for k in _AZ   if k in data), None)
    di = next((data[k] for k in _DIST if k in data), None)
    ht = next((data[k] for k in _HT   if k in data), None)
    if az is not None and di is not None and ht is not None:
        return float(az), float(di), float(ht)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ──────────────────────────────────────────────────────────────────────────────

def _make_synthetic_segments(
    cfg,
    n: int = 200,
) -> List[Tuple[List[np.ndarray], dict]]:
    """
    Generate `n` synthetic 3-channel drone segments using the pipeline's own
    synthesise_drone() function.  Covers all BPF drone types.
    """
    from .audio_processing import synthesise_drone

    rng = np.random.default_rng(cfg.SEED)
    segments = []
    drone_types = list(cfg.SYNTH_DRONE_TYPES)
    weights     = np.asarray(cfg.SYNTH_DRONE_WEIGHTS, dtype=float)
    weights    /= weights.sum()

    for _ in range(n):
        az_deg   = rng.uniform(-180, 180)
        dist_m   = rng.uniform(5.0, cfg.MAX_LOCALIZATION_DIST * 0.8)
        height_m = rng.uniform(5.0, 40.0)
        x = dist_m * math.cos(math.radians(az_deg)) + cfg.ARRAY_CENTER[0]
        y = dist_m * math.sin(math.radians(az_deg)) + cfg.ARRAY_CENTER[1]

        d_type = str(rng.choice(drone_types, p=weights))

        channels = synthesise_drone(
            mic_positions  = cfg.MIC_POSITIONS,
            src_xy         = [x, y],
            drone_type     = d_type,
            noise_level    = float(rng.uniform(0.01, 0.08)),
            cfg            = cfg,
        )
        label = {
            "azimuth_deg": az_deg,
            "distance_m":  dist_m,
            "height_m":    height_m,
            "source":      "synthetic",
            "drone_type":  d_type,
            "has_position": True,
        }
        segments.append((channels, label))

    log.info("Synthetic fallback: generated %d segments.", n)
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Remote-ZIP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_remotezip() -> bool:
    """Return True if remotezip is importable, otherwise try to install it."""
    try:
        import remotezip  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "remotezip"],
            timeout=60,
        )
        import remotezip  # noqa: F401
        return True
    except Exception as exc:
        log.warning("Could not install remotezip: %s", exc)
        return False


class _RemoteZipReader:
    """
    Streams individual files out of a remote ZIP without downloading the whole
    archive.  Falls back gracefully when remotezip is not available.
    """

    def __init__(self, url: str, buffer_mb: int = 64):
        self._url    = url
        self._buf    = buffer_mb * 1024 * 1024
        self._rz     = None

    def __enter__(self):
        if not _ensure_remotezip():
            raise ImportError("remotezip is not available.")
        from remotezip import RemoteZip
        self._rz = RemoteZip(self._url, initial_buffer_size=self._buf)
        self._rz.__enter__()
        return self

    def __exit__(self, *args):
        if self._rz:
            self._rz.__exit__(*args)

    def namelist(self) -> List[str]:
        return [n.replace("\\", "/") for n in self._rz.namelist()]

    def read(self, name: str) -> bytes:
        return self._rz.read(name)


# ──────────────────────────────────────────────────────────────────────────────
# Core streaming generators
# ──────────────────────────────────────────────────────────────────────────────

def _stream_uavirbase_remote(
    url: str,
    cfg,
    max_sessions: int = 0,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Stream (channels, label) pairs from the UaVirBASE remote ZIP.

    Each session folder contains:
      output.wav  — multi-channel WAV (8 mics, 96 kHz)
      label.json  — position annotation

    We pick channels 0, 1, 3 (North, East, West) matching
    config.UAVIRBASE_MIC_INDICES and resample to config.SR.
    """
    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    AUDIO_NAMES = {"output.wav", "audio.wav"}
    LABEL_NAMES = {"label.json", "annotation.json"}

    with _RemoteZipReader(url) as rz:
        names = rz.namelist()
        # Group by session directory
        sessions: Dict[str, Dict[str, str]] = {}
        for name in names:
            parts = name.split("/")
            if len(parts) < 2:
                continue
            session = "/".join(parts[:-1])
            fname   = parts[-1]
            sessions.setdefault(session, {})
            if fname in AUDIO_NAMES:
                sessions[session]["audio"] = name
            elif fname in LABEL_NAMES:
                sessions[session]["label"] = name

        paired = [
            (s, v["audio"], v["label"])
            for s, v in sessions.items()
            if "audio" in v and "label" in v
        ]
        log.info("UaVirBASE: %d paired sessions found in remote ZIP.", len(paired))
        if max_sessions > 0:
            paired = paired[:max_sessions]

        n_ok = n_skip = 0
        for session_id, audio_path, label_path in paired:
            try:
                label_bytes = rz.read(label_path)
                parsed = _parse_uavirbase_label(label_bytes)
                if parsed is None:
                    n_skip += 1
                    continue
                az_deg, dist_m, height_m = parsed

                audio_bytes = rz.read(audio_path)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(audio_bytes)
                    tmp = tf.name
                try:
                    channels = ap.load_channels(
                        tmp, channel_indices=cfg.UAVIRBASE_MIC_INDICES
                    )
                finally:
                    os.unlink(tmp)

                channels = [ap.pad_or_truncate(c) for c in channels]
                label = {
                    "azimuth_deg": az_deg,
                    "distance_m":  dist_m,
                    "height_m":    height_m,
                    "source":      "uavirbase_remote",
                    "session":     Path(session_id).name,
                    "has_position": True,
                }
                n_ok += 1
                yield channels, label
            except Exception as exc:
                log.warning("Session %s failed: %s", session_id, exc)
                n_skip += 1

    log.info("UaVirBASE remote: %d ok, %d skipped.", n_ok, n_skip)


def _stream_dunakeszi_remote(
    url: str,
    cfg,
    array: str = _DEFAULT_ARRAY,
    max_dist: float = 100.0,
    max_segments: int = 0,
    required_split: Optional[str] = None,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Stream (channels, label) pairs from a Dunakeszi-style remote ZIP.

    Expected ZIP layout (dunakeszi_pipeline_ready or equivalent):
      <stem>_ch0.wav
      <stem>_ch1.wav
      <stem>_ch2.wav
      <stem>_label.json

    All resampling / label conversion is done here in-process using the same
    logic as prepare_dunakeszi_for_pipeline.py.
    """
    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    with _RemoteZipReader(url) as rz:
        names_raw = rz.namelist()
        # Build stem → {ch0, ch1, ch2, label} map
        stem_map: Dict[str, Dict[str, str]] = {}
        for name in names_raw:
            fname = Path(name).name
            if fname.endswith("_label.json"):
                stem = fname[: -len("_label.json")]
                stem_map.setdefault(stem, {})["label"] = name
            elif fname.endswith("_ch0.wav"):
                stem = fname[: -len("_ch0.wav")]
                stem_map.setdefault(stem, {})["ch0"] = name
            elif fname.endswith("_ch1.wav"):
                stem = fname[: -len("_ch1.wav")]
                stem_map.setdefault(stem, {})["ch1"] = name
            elif fname.endswith("_ch2.wav"):
                stem = fname[: -len("_ch2.wav")]
                stem_map.setdefault(stem, {})["ch2"] = name

        complete = {
            s: v for s, v in stem_map.items()
            if all(k in v for k in ("label", "ch0", "ch1", "ch2"))
        }
        log.info("Dunakeszi remote: %d complete triplets found.", len(complete))

        items = list(complete.items())
        if max_segments > 0:
            items = items[:max_segments]

        n_ok = n_skip = 0
        for stem, files in items:
            try:
                label_raw = json.loads(rz.read(files["label"]).decode("utf-8"))
                label     = _convert_label(label_raw, max_dist)

                if required_split and label.get("split") not in (None, required_split):
                    continue

                channels = []
                for ch_key in ("ch0", "ch1", "ch2"):
                    wav_bytes = rz.read(files[ch_key])
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(wav_bytes)
                        tmp = tf.name
                    try:
                        ch = ap.load(tmp, mono=True)
                    finally:
                        os.unlink(tmp)
                    channels.append(ap.pad_or_truncate(ch))

                n_ok += 1
                yield channels, label
            except Exception as exc:
                log.warning("Segment %s failed: %s", stem, exc)
                n_skip += 1

    log.info("Dunakeszi remote: %d ok, %d skipped.", n_ok, n_skip)


def _stream_from_local_zip(
    zip_path: Path,
    cfg,
    max_dist: float = 100.0,
    dataset_type: str = "dunakeszi",
    required_split: Optional[str] = None,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Stream segments from a locally-downloaded ZIP.
    Handles both dunakeszi_pipeline_ready and UaVirBASE layouts.
    """
    from .audio_processing import AudioProcessor
    ap = AudioProcessor(cfg)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]

        if dataset_type == "uavirbase":
            # UaVirBASE: session/output.wav + session/label.json
            sessions: Dict[str, Dict[str, str]] = {}
            for name in names:
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                session = "/".join(parts[:-1])
                fname   = parts[-1]
                sessions.setdefault(session, {})
                if fname in ("output.wav", "audio.wav"):
                    sessions[session]["audio"] = name
                elif fname in ("label.json", "annotation.json"):
                    sessions[session]["label"] = name

            for session_id, files in sessions.items():
                if "audio" not in files or "label" not in files:
                    continue
                try:
                    label_bytes = zf.read(files["label"])
                    parsed = _parse_uavirbase_label(label_bytes)
                    if parsed is None:
                        continue
                    az, di, ht = parsed
                    audio_bytes = zf.read(files["audio"])
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(audio_bytes); tmp = tf.name
                    try:
                        channels = ap.load_channels(tmp, cfg.UAVIRBASE_MIC_INDICES)
                    finally:
                        os.unlink(tmp)
                    channels = [ap.pad_or_truncate(c) for c in channels]
                    yield channels, {
                        "azimuth_deg": az, "distance_m": di, "height_m": ht,
                        "source": "uavirbase_local", "session": Path(session_id).name,
                        "has_position": True,
                    }
                except Exception as exc:
                    log.warning("UaVirBASE session %s: %s", session_id, exc)

        else:
            # Dunakeszi-style: stem_ch0/1/2.wav + stem_label.json
            stem_map: Dict[str, Dict[str, str]] = {}
            for name in names:
                fname = Path(name).name
                if fname.endswith("_label.json"):
                    stem = fname[: -len("_label.json")]
                    stem_map.setdefault(stem, {})["label"] = name
                elif fname.endswith("_ch0.wav"):
                    stem = fname[: -len("_ch0.wav")]
                    stem_map.setdefault(stem, {})["ch0"] = name
                elif fname.endswith("_ch1.wav"):
                    stem = fname[: -len("_ch1.wav")]
                    stem_map.setdefault(stem, {})["ch1"] = name
                elif fname.endswith("_ch2.wav"):
                    stem = fname[: -len("_ch2.wav")]
                    stem_map.setdefault(stem, {})["ch2"] = name

            for stem, files in stem_map.items():
                if not all(k in files for k in ("label", "ch0", "ch1", "ch2")):
                    continue
                try:
                    raw   = json.loads(zf.read(files["label"]).decode("utf-8"))
                    label = _convert_label(raw, max_dist)
                    if required_split and label.get("split") not in (None, required_split):
                        continue
                    channels = []
                    for ch_key in ("ch0", "ch1", "ch2"):
                        wav_bytes = zf.read(files[ch_key])
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                            tf.write(wav_bytes); tmp = tf.name
                        try:
                            ch = ap.load(tmp, mono=True)
                        finally:
                            os.unlink(tmp)
                        channels.append(ap.pad_or_truncate(ch))
                    yield channels, label
                except Exception as exc:
                    log.warning("Segment %s: %s", stem, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Full-download fallback
# ──────────────────────────────────────────────────────────────────────────────

def _download_zip(url: str, dest: Path, chunk_bytes: int = 4 * 1024 * 1024) -> Path:
    """Download `url` to `dest` with a progress log every 50 MB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    log.info("Downloading %s → %s …", url, dest)
    written = 0
    t0      = time.time()
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(chunk_bytes)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
            mb = written / (1024 ** 2)
            if int(mb) % 50 == 0:
                log.info("  … %.0f MB  (%.0f s)", mb, time.time() - t0)
    tmp.rename(dest)
    log.info("Download complete: %.1f MB in %.0f s", written / 1e6, time.time() - t0)
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Public generator
# ──────────────────────────────────────────────────────────────────────────────

def stream_repository_segments(
    cfg=None,
    url: Optional[str] = None,
    dataset_type: str = "uavirbase",
    array: str = _DEFAULT_ARRAY,
    max_dist: float = 100.0,
    max_segments: int = 0,
    required_split: Optional[str] = None,
    cache_zip: Optional[Path] = None,
    allow_download: bool = True,
    allow_synthetic_fallback: bool = True,
    n_synthetic: int = 200,
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Yield ``(channels, label)`` tuples read directly from the online repository.

    Parameters
    ──────────
    cfg               : Config instance (defaults to the module singleton)
    url               : URL of the remote ZIP.  Defaults to cfg.UAVIRBASE_ZIP_URL
                        when dataset_type == 'uavirbase'.
    dataset_type      : 'uavirbase' | 'dunakeszi'
    array             : Dunakeszi array to use ('BK-6-E' or 'BK-6-W')
    max_dist          : MAX_LOCALIZATION_DIST for label normalisation info
    max_segments      : Stop after this many segments (0 = no limit)
    required_split    : If given, only yield segments whose split == this value
    cache_zip         : If given and the file exists, stream from it instead of
                        the network; if given and absent, download to this path.
    allow_download    : Whether to fall back to a full download if remotezip fails
    allow_synthetic_fallback : Whether to yield synthetic data if all network
                               paths fail
    n_synthetic       : How many synthetic segments to generate as fallback

    Yields
    ──────
    channels : List[np.ndarray]  — 3 × (N,) float32 arrays at cfg.SR
    label    : dict              — pipeline-format label (azimuth_deg, distance_m,
                                  height_m, has_position, source, …)
    """
    from .config import config as _default_cfg
    cfg = cfg or _default_cfg

    if url is None and dataset_type == "uavirbase":
        url = cfg.UAVIRBASE_ZIP_URL

    # ── 1. Local cache ZIP already exists ─────────────────────────────────────
    if cache_zip and Path(cache_zip).exists():
        log.info("Streaming from cached ZIP: %s", cache_zip)
        yield from _stream_from_local_zip(
            Path(cache_zip), cfg,
            max_dist=max_dist,
            dataset_type=dataset_type,
            required_split=required_split,
        )
        return

    # ── 2. Remote-ZIP streaming (no full download) ─────────────────────────────
    if url:
        try:
            log.info("Attempting remote-ZIP streaming from %s …", url)
            if dataset_type == "uavirbase":
                gen = _stream_uavirbase_remote(url, cfg, max_segments=max_segments)
            else:
                gen = _stream_dunakeszi_remote(
                    url, cfg, array=array, max_dist=max_dist,
                    max_segments=max_segments, required_split=required_split,
                )
            for item in gen:
                yield item
            return
        except Exception as exc:
            log.warning("Remote-ZIP streaming failed: %s. Trying full download …", exc)

    # ── 3. Full download fallback ──────────────────────────────────────────────
    if url and allow_download:
        try:
            zip_dest = cache_zip if cache_zip else Path(
                tempfile.mktemp(suffix=".zip", prefix="drone_repo_")
            )
            _download_zip(url, Path(zip_dest))
            log.info("Streaming from downloaded ZIP: %s", zip_dest)
            yield from _stream_from_local_zip(
                Path(zip_dest), cfg,
                max_dist=max_dist,
                dataset_type=dataset_type,
                required_split=required_split,
            )
            if not cache_zip:
                Path(zip_dest).unlink(missing_ok=True)
            return
        except Exception as exc:
            log.warning("Full download failed: %s. Falling back to synthetic.", exc)

    # ── 4. Synthetic fallback ──────────────────────────────────────────────────
    if allow_synthetic_fallback:
        print(
            "⚠️  Repository unreachable — using SYNTHETIC data "
            f"({n_synthetic} segments). Results will not reflect real recordings."
        )
        for channels, label in _make_synthetic_segments(cfg, n=n_synthetic):
            yield channels, label


# ──────────────────────────────────────────────────────────────────────────────
# Disk preparation helper (mirrors prepare_dunakeszi_for_pipeline.py output)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_repository_to_disk(
    cfg=None,
    output_dir: Optional[Path] = None,
    url: Optional[str] = None,
    dataset_type: str = "uavirbase",
    array: str = _DEFAULT_ARRAY,
    max_dist: float = 100.0,
    max_segments: int = 0,
    required_split: Optional[str] = None,
    cache_zip: Optional[Path] = None,
    allow_download: bool = True,
    allow_synthetic_fallback: bool = True,
    n_synthetic: int = 200,
) -> Path:
    """
    Download and convert repository data to the dunakeszi_pipeline_ready layout
    on disk, then return the output directory path.

    Output layout:
        output_dir/
            <stem>_ch0.wav
            <stem>_ch1.wav
            <stem>_ch2.wav
            <stem>_label.json
            labels.csv

    The output is a drop-in replacement for a LocalizationDataset split
    directory and is compatible with load_test_dataset_zip(generic_triplet).
    """
    import csv
    from .config import config as _default_cfg
    cfg = cfg or _default_cfg

    if output_dir is None:
        output_dir = cfg.PROCESSED_DIR / "repository_cache" / dataset_type
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already populated
    existing = list(output_dir.glob("*_label.json"))
    if len(existing) > 10:
        print(f"✅ Repository cache already populated ({len(existing)} segments): {output_dir}")
        return output_dir

    gen = stream_repository_segments(
        cfg=cfg, url=url, dataset_type=dataset_type, array=array,
        max_dist=max_dist, max_segments=max_segments,
        required_split=required_split, cache_zip=cache_zip,
        allow_download=allow_download,
        allow_synthetic_fallback=allow_synthetic_fallback,
        n_synthetic=n_synthetic,
    )

    csv_rows = []
    n_written = 0
    for channels, label in gen:
        stem = f"seg_{n_written:05d}"
        if label.get("segment_id"):
            stem = str(label["segment_id"])
        elif label.get("session"):
            stem = f"{label['session']}_{n_written:04d}"

        for i, ch in enumerate(channels):
            sf.write(str(output_dir / f"{stem}_ch{i}.wav"), ch, cfg.SR)
        (output_dir / f"{stem}_label.json").write_text(
            json.dumps(label, indent=2)
        )
        csv_rows.append({
            "stem":         stem,
            "azimuth_deg":  label.get("azimuth_deg"),
            "distance_m":   label.get("distance_m"),
            "height_m":     label.get("height_m"),
            "source":       label.get("source", ""),
            "has_position": label.get("has_position", False),
            "split":        label.get("split", ""),
            "maneuver_type": label.get("maneuver_type", ""),
        })
        n_written += 1

    if csv_rows:
        csv_path = output_dir / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  📋 labels.csv written ({len(csv_rows)} rows)")

    print(f"✅ Prepared {n_written} segments → {output_dir}")
    return output_dir


# ──────────────────────────────────────────────────────────────────────────────
# torch Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


if _TORCH_OK:
    class RepositoryDataset(Dataset):
        """
        A torch Dataset that reads segments directly from the online repository.

        Usage
        ─────
        .. code-block:: python

            from drone_detection.repository_loader import RepositoryDataset
            from drone_detection import config
            from torch.utils.data import DataLoader

            config.set_array_geometry('gp2')
            config.MAX_LOCALIZATION_DIST = 100.0

            ds     = RepositoryDataset(config, dataset_type='dunakeszi',
                                       max_segments=500)
            loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

            mel, ipd, label_t = next(iter(loader))
            # mel:     (B, 3, N_MELS, T)
            # ipd:     (B, 3) or (B, 4)
            # label_t: (B, 4) → [sin_az, cos_az, dist/max, ht/max]

        Parameters
        ──────────
        cfg             : Config instance
        url             : Remote ZIP URL (defaults to cfg.UAVIRBASE_ZIP_URL)
        dataset_type    : 'uavirbase' | 'dunakeszi'
        array           : Dunakeszi array name
        max_dist        : MAX_LOCALIZATION_DIST for label normalisation
        max_segments    : Limit dataset size (0 = all)
        required_split  : 'train' | 'val' | 'test' | None
        cache_zip       : Path to cache the downloaded ZIP locally
        allow_download  : Fall back to full download if remotezip fails
        allow_synthetic_fallback : Fall back to synthetic if network fails
        n_synthetic     : Size of the synthetic fallback
        augment         : Apply waveform augmentation during __getitem__
        """

        def __init__(
            self,
            cfg=None,
            url: Optional[str] = None,
            dataset_type: str = "uavirbase",
            array: str = _DEFAULT_ARRAY,
            max_dist: float = 100.0,
            max_segments: int = 0,
            required_split: Optional[str] = None,
            cache_zip: Optional[Path] = None,
            allow_download: bool = True,
            allow_synthetic_fallback: bool = True,
            n_synthetic: int = 200,
            augment: bool = False,
        ):
            from .config import config as _default_cfg
            self.cfg       = cfg or _default_cfg
            self.augment   = augment
            self.max_dist  = max_dist
            self._segments: List[Tuple[List[np.ndarray], dict]] = []

            print(f"📥 RepositoryDataset: loading '{dataset_type}' data …")
            for channels, label in stream_repository_segments(
                cfg=self.cfg, url=url, dataset_type=dataset_type,
                array=array, max_dist=max_dist, max_segments=max_segments,
                required_split=required_split, cache_zip=cache_zip,
                allow_download=allow_download,
                allow_synthetic_fallback=allow_synthetic_fallback,
                n_synthetic=n_synthetic,
            ):
                self._segments.append((channels, label))

            print(f"  Loaded {len(self._segments)} segments  "
                  f"({'with' if augment else 'without'} augmentation)")

        def __len__(self) -> int:
            return len(self._segments)

        def __getitem__(self, idx: int):
            from .utils import compute_ipd_features
            from .audio_processing import AudioProcessor

            ap = AudioProcessor(self.cfg)
            channels, label = self._segments[idx]

            # Optional waveform augmentation
            if self.augment:
                try:
                    from .utils import augment_waveform
                    channels = [augment_waveform(c, self.cfg.SR) for c in channels]
                except Exception:
                    pass

            # Mel feature tensor: (3, N_MELS, T)
            mels  = [ap.mel(c) for c in channels]
            mel_t = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)

            # IPD feature tensor: (3,) or (4,)
            ipd_raw = compute_ipd_features(channels, self.cfg)
            if getattr(self.cfg, "BPF_ENERGY_RATIO_AS_FEATURE", False):
                try:
                    bpf_hz = 200.0  # reasonable default if unknown
                    ratio  = ap.compute_bpf_energy_ratio(channels[0], bpf_hz)
                except Exception:
                    ratio = 0.0
                ipd_raw = np.append(ipd_raw, float(ratio)).astype(np.float32)
            ipd_t = torch.tensor(ipd_raw, dtype=torch.float32)

            # Label tensor: [sin_az, cos_az, dist/max, ht/max]
            az   = label.get("azimuth_deg") or 0.0
            dist = label.get("distance_m")  or 0.0
            ht   = label.get("height_m")    or 0.0
            az_r = math.radians(az)
            label_t = torch.tensor([
                math.sin(az_r),
                math.cos(az_r),
                float(np.clip(dist / self.max_dist, 0.0, 1.0)),
                float(np.clip(ht   / self.max_dist, 0.0, 1.0)),
            ], dtype=torch.float32)

            return mel_t, ipd_t, label_t

else:
    # Dummy class so the module imports cleanly even without torch
    class RepositoryDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "torch is required for RepositoryDataset. "
                "Install it with: pip install torch"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def load_repository_dataset(
    cfg=None,
    url: Optional[str] = None,
    dataset_type: str = "uavirbase",
    array: str = _DEFAULT_ARRAY,
    max_dist: float = 100.0,
    max_segments: int = 0,
    required_split: Optional[str] = None,
    cache_zip: Optional[Path] = None,
    allow_download: bool = True,
    allow_synthetic_fallback: bool = True,
    n_synthetic: int = 200,
    augment: bool = False,
) -> "RepositoryDataset":
    """
    One-liner factory for RepositoryDataset.

    Example — UaVirBASE (default, Zenodo):
    ─────────────────────────────────────────
    .. code-block:: python

        from drone_detection.repository_loader import load_repository_dataset
        from drone_detection import config

        ds = load_repository_dataset(config, max_segments=500)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

    Example — Dunakeszi hosted ZIP:
    ────────────────────────────────
    .. code-block:: python

        config.set_array_geometry('gp2')
        config.MAX_LOCALIZATION_DIST = 100.0

        ds = load_repository_dataset(
            config,
            url='https://example.com/dunakeszi_pipeline_ready.zip',
            dataset_type='dunakeszi',
            max_dist=100.0,
            required_split='test',
        )

    Example — no network (synthetic only):
    ────────────────────────────────────────
    .. code-block:: python

        ds = load_repository_dataset(config, allow_download=False,
                                     allow_synthetic_fallback=True,
                                     n_synthetic=500)
    """
    return RepositoryDataset(
        cfg=cfg, url=url, dataset_type=dataset_type, array=array,
        max_dist=max_dist, max_segments=max_segments,
        required_split=required_split, cache_zip=cache_zip,
        allow_download=allow_download,
        allow_synthetic_fallback=allow_synthetic_fallback,
        n_synthetic=n_synthetic, augment=augment,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper — quick smoke-test / disk preparation
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    ap = argparse.ArgumentParser(
        description="Stream / download drone repository data and convert to pipeline format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
# Stream first 50 UaVirBASE segments and print labels
python repository_loader.py --dataset uavirbase --max-segments 50

# Download & convert a Dunakeszi hosted ZIP to disk
python repository_loader.py \\
    --dataset dunakeszi \\
    --url https://example.com/dunakeszi_pipeline_ready.zip \\
    --output-dir ./dunakeszi_cache \\
    --max-dist 100.0

# Dry-run with synthetic data only (no network)
python repository_loader.py --no-download --synthetic-only --n-synthetic 100
        """,
    )
    ap.add_argument("--dataset",      choices=["uavirbase", "dunakeszi"], default="uavirbase")
    ap.add_argument("--url",          default=None, help="Remote ZIP URL")
    ap.add_argument("--array",        default=_DEFAULT_ARRAY, choices=list(_ARRAY_CHANNELS))
    ap.add_argument("--max-dist",     type=float, default=100.0)
    ap.add_argument("--max-segments", type=int,   default=10,
                    help="Max segments to stream (0 = all; default 10 for smoke-test)")
    ap.add_argument("--split",        default=None, choices=["train", "val", "test"],
                    help="Only yield segments of this split")
    ap.add_argument("--cache-zip",    default=None, help="Cache the downloaded ZIP here")
    ap.add_argument("--output-dir",   default=None,
                    help="If given, write pipeline-ready files to this directory")
    ap.add_argument("--no-download",  action="store_true",
                    help="Disable full-download fallback (remote-zip only or synthetic)")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Skip all network attempts; use synthetic data only")
    ap.add_argument("--n-synthetic",  type=int, default=200)
    ap.add_argument("--array-geometry", default=None,
                    help="Override array geometry (uavirbase | gp1 | gp2)")
    args = ap.parse_args()

    # Set up config
    try:
        from drone_detection.config import config as cfg
    except ImportError:
        from config import config as cfg  # fallback when run from package dir

    if args.array_geometry:
        cfg.set_array_geometry(args.array_geometry)
    cfg.MAX_LOCALIZATION_DIST = args.max_dist

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    url = None if args.synthetic_only else args.url

    if args.output_dir:
        out = prepare_repository_to_disk(
            cfg=cfg, output_dir=Path(args.output_dir), url=url,
            dataset_type=args.dataset, array=args.array,
            max_dist=args.max_dist, max_segments=args.max_segments,
            required_split=args.split,
            cache_zip=Path(args.cache_zip) if args.cache_zip else None,
            allow_download=not args.no_download,
            allow_synthetic_fallback=True,
            n_synthetic=args.n_synthetic,
        )
        print(f"\n✅ Pipeline-ready data at: {out}")
    else:
        print(f"\n{'─'*60}")
        print(f"  Smoke-test: streaming up to {args.max_segments} segments")
        print(f"{'─'*60}")
        for i, (channels, label) in enumerate(stream_repository_segments(
            cfg=cfg, url=url, dataset_type=args.dataset, array=args.array,
            max_dist=args.max_dist, max_segments=args.max_segments,
            required_split=args.split,
            cache_zip=Path(args.cache_zip) if args.cache_zip else None,
            allow_download=not args.no_download,
            allow_synthetic_fallback=True,
            n_synthetic=args.n_synthetic,
        )):
            ch_shapes = [c.shape for c in channels]
            print(
                f"  [{i+1:3d}] channels={ch_shapes}  "
                f"az={label.get('azimuth_deg','?'):6}°  "
                f"dist={label.get('distance_m','?')}m  "
                f"src={label.get('source','?')}"
            )
        print(f"\n{'─'*60}")
        print("Smoke-test complete.")


if __name__ == "__main__":
    _cli()