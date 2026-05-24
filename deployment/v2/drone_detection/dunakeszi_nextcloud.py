# -*- coding: utf-8 -*-
"""
dunakeszi_nextcloud.py
──────────────────────
HTTP range-request reader for the Dunakeszi dataset files hosted on Nextcloud.

Provides:
  - WebDAV PROPFIND-based file listing (polywav + MEMS)
  - HTTP Range-request byte-window streaming (no full download)
  - GPX parsing from the share
  - Segment-browser metadata API (merges ground-truth + local availability)

No files are ever downloaded in full.  Each call fetches only the bytes
it needs (e.g. a 3-second window of a 4 GB polywav = ~3.2 MB read).

Configuration keys expected on the `cfg` object
────────────────────────────────────────────────
  NEXTCLOUD_BASE_URL      "https://your.nextcloud.host"
  NEXTCLOUD_SHARE_TOKEN   "AbCdEf1234567"   (public share token)
  NEXTCLOUD_POLYWAV_PATH  "/polywav"        (path inside the share)
  NEXTCLOUD_MEMS_PATH     "/mems"           (path inside the share)
  NEXTCLOUD_GPX_PATH      "/DRON-GPX"       (path inside the share)
  DUNAKESZI_LOCAL_PATH    "/path/to/dunakeszi_pipeline_ready_B"  (optional)

If Nextcloud credentials are absent the module degrades gracefully:
  list_remote_files() → []
  stream_segment_range() raises RemoteUnavailableError (caught by caller)
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

log = logging.getLogger("drone_v2.dunakeszi_nextcloud")


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class RemoteUnavailableError(RuntimeError):
    """Raised when the Nextcloud share is unreachable or misconfigured."""


class RangeRequestError(RuntimeError):
    """Raised when an HTTP range request fails."""


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

POLYWAV_SR       = 192_000          # native sample-rate of polywav files
POLYWAV_CHANNELS = 14               # total channels in each polywav file
POLYWAV_DTYPE    = np.float32       # 32-bit float
POLYWAV_BYTES_PER_SAMPLE = 4        # bytes per sample per channel
POLYWAV_BYTES_PER_FRAME  = POLYWAV_CHANNELS * POLYWAV_BYTES_PER_SAMPLE  # 56

MEMS_SR_ASSUMED       = 48_000
MEMS_CHANNELS_ASSUMED = 4
MEMS_BITS_ASSUMED     = 24          # 24-bit signed int → 3 bytes/sample
MEMS_BYTES_PER_FRAME  = MEMS_CHANNELS_ASSUMED * 3  # 12

# Channel indices for the two BK-6 arrays inside the polywav (0-indexed)
BK6E_CHANNELS = [8, 9, 10]    # East array: E-E, E-H, E-B
BK6W_CHANNELS = [2, 3, 4]     # West array: W-E, W-H, W-B

# WebDAV namespace
_DAV_NS = "DAV:"

# WAV header size in bytes (standard 44-byte PCM header)
_WAV_HEADER_BYTES = 44


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _requests():
    """Lazy-import requests (avoids import cost at module load)."""
    try:
        import requests as _req
        return _req
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
        import requests as _req
        return _req


def _share_dav_url(cfg) -> str:
    """
    Build the WebDAV root URL for a Nextcloud public share.

    Nextcloud public share WebDAV path pattern:
      https://<host>/public.php/webdav
    with HTTP Basic Auth:  user=<token>  password=""
    """
    base = getattr(cfg, "NEXTCLOUD_BASE_URL", "").rstrip("/")
    if not base:
        raise RemoteUnavailableError("NEXTCLOUD_BASE_URL not configured")
    return f"{base}/public.php/webdav"


def _auth(cfg) -> Tuple[str, str]:
    token = getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", "")
    if not token:
        raise RemoteUnavailableError("NEXTCLOUD_SHARE_TOKEN not configured")
    return (token, "")


def _file_url(cfg, remote_path: str) -> str:
    """Build full URL for a file inside the share."""
    base = _share_dav_url(cfg)
    path = remote_path.lstrip("/")
    return f"{base}/{path}"


# ══════════════════════════════════════════════════════════════════════════════
# WebDAV PROPFIND — list files in a share directory
# ══════════════════════════════════════════════════════════════════════════════

def list_remote_files(
    cfg,
    remote_subpath: str = "",
    depth: int = 1,
    extensions: Optional[Tuple[str, ...]] = None,
) -> List[Dict[str, Any]]:
    """
    List files in a Nextcloud share directory via WebDAV PROPFIND.

    Parameters
    ----------
    cfg            : Config object with NEXTCLOUD_* keys
    remote_subpath : Sub-path inside the share (e.g. "/polywav")
    depth          : WebDAV Depth header (1 = immediate children)
    extensions     : Filter by file extension, e.g. (".wav",). None = all.

    Returns
    -------
    List of dicts: {name, path, size_bytes, content_type, last_modified}
    """
    req = _requests()
    try:
        base   = _share_dav_url(cfg)
        auth   = _auth(cfg)
        url    = base.rstrip("/") + ("/" + remote_subpath.strip("/") if remote_subpath else "")
        headers = {
            "Depth": str(depth),
            "Content-Type": "application/xml",
        }
        body = (
            '<?xml version="1.0"?>'
            '<d:propfind xmlns:d="DAV:">'
            '  <d:prop>'
            '    <d:displayname/>'
            '    <d:getcontentlength/>'
            '    <d:getcontenttype/>'
            '    <d:getlastmodified/>'
            '  </d:prop>'
            '</d:propfind>'
        )
        resp = req.request(
            "PROPFIND", url,
            auth=auth,
            headers=headers,
            data=body,
            timeout=30,
        )
        resp.raise_for_status()

    except Exception as exc:
        log.warning("WebDAV PROPFIND failed (%s): %s", remote_subpath, exc)
        return []

    # Parse the multi-status XML
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        log.warning("WebDAV XML parse error: %s", exc)
        return []

    ns = {"d": _DAV_NS}
    results: List[Dict[str, Any]] = []

    for response in root.findall(".//d:response", ns):
        href_el = response.find("d:href", ns)
        if href_el is None:
            continue
        href = href_el.text or ""

        # Skip collection (directory) entries
        ctype_el = response.find(".//d:getcontenttype", ns)
        ctype = (ctype_el.text or "") if ctype_el is not None else ""
        if "directory" in ctype or href.endswith("/"):
            continue

        # Extract filename from href
        name = href.rstrip("/").split("/")[-1]
        if not name:
            continue

        # Extension filter
        if extensions is not None:
            if not any(name.lower().endswith(ext.lower()) for ext in extensions):
                continue

        size_el = response.find(".//d:getcontentlength", ns)
        size = int(size_el.text) if size_el is not None and size_el.text else 0

        lmod_el = response.find(".//d:getlastmodified", ns)
        lmod = lmod_el.text if lmod_el is not None else None

        results.append({
            "name":          name,
            "path":          href,
            "size_bytes":    size,
            "content_type":  ctype,
            "last_modified": lmod,
        })

    log.info("WebDAV PROPFIND %s: %d files found", remote_subpath, len(results))
    return results


def list_polywav_files(cfg) -> List[Dict[str, Any]]:
    """List PolyWav files from the Nextcloud share."""
    path = getattr(cfg, "NEXTCLOUD_POLYWAV_PATH", "/polywav")
    files = list_remote_files(cfg, remote_subpath=path, extensions=(".wav",))
    # Sort by filename so chunk order is preserved
    files.sort(key=lambda f: f["name"].lower())
    return files


def list_mems_files(cfg) -> List[Dict[str, Any]]:
    """List MEMS files from the Nextcloud share."""
    path = getattr(cfg, "NEXTCLOUD_MEMS_PATH", "/mems")
    files = list_remote_files(cfg, remote_subpath=path, extensions=(".wav",))
    files.sort(key=lambda f: f["name"].lower())
    return files


def list_gpx_files(cfg, show_folder: str = "") -> List[Dict[str, Any]]:
    """List GPX files for a given show folder from the Nextcloud share."""
    base_path = getattr(cfg, "NEXTCLOUD_GPX_PATH", "/DRON-GPX")
    sub = f"{base_path}/{show_folder}".rstrip("/")
    return list_remote_files(cfg, remote_subpath=sub, extensions=(".gpx",))


# ══════════════════════════════════════════════════════════════════════════════
# HTTP Range-request reader
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_byte_range(
    url: str,
    auth: Tuple[str, str],
    byte_start: int,
    byte_end: int,
    timeout: int = 60,
) -> bytes:
    """
    Fetch [byte_start, byte_end) from a remote URL via HTTP Range request.

    Returns raw bytes.  Raises RangeRequestError on failure.
    """
    req = _requests()
    headers = {"Range": f"bytes={byte_start}-{byte_end - 1}"}
    try:
        resp = req.get(url, auth=auth, headers=headers, timeout=timeout)
        if resp.status_code not in (200, 206):
            raise RangeRequestError(
                f"Range request failed: HTTP {resp.status_code} for {url} "
                f"(range {byte_start}-{byte_end})"
            )
        return resp.content
    except RangeRequestError:
        raise
    except Exception as exc:
        raise RangeRequestError(f"Range request error: {exc}") from exc


def read_polywav_window(
    cfg,
    remote_path: str,
    start_s: float,
    duration_s: float,
    channel_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Read a time window from a remote 192 kHz / 14-channel polywav via HTTP range.

    Parameters
    ----------
    cfg            : Config with NEXTCLOUD_* keys
    remote_path    : Path inside the share (e.g. "/polywav/251020VITEMOROM1AT01.wav")
    start_s        : Start offset in seconds within this file
    duration_s     : Window length in seconds
    channel_indices: List of 0-indexed channel numbers to extract (default: all 14)

    Returns
    -------
    audio   : np.ndarray  shape (n_channels, n_samples)  float32
    sample_rate : int     always POLYWAV_SR (192 000)
    """
    if channel_indices is None:
        channel_indices = list(range(POLYWAV_CHANNELS))

    auth = _auth(cfg)
    url  = _file_url(cfg, remote_path)

    # PolyWav is raw PCM float32 preceded by a WAV header
    # Byte layout: header | frame0_ch0…ch13 | frame1_ch0…ch13 | …
    frame_start  = int(start_s    * POLYWAV_SR)
    n_frames     = int(duration_s * POLYWAV_SR)

    byte_start = _WAV_HEADER_BYTES + frame_start * POLYWAV_BYTES_PER_FRAME
    byte_end   = byte_start + n_frames * POLYWAV_BYTES_PER_FRAME

    log.debug(
        "Range-reading polywav: %s  t=[%.2f, %.2f)s  bytes=[%d, %d)",
        remote_path, start_s, start_s + duration_s, byte_start, byte_end,
    )

    raw = _fetch_byte_range(url, auth, byte_start, byte_end)

    # Interpret as float32, shape (n_frames, POLYWAV_CHANNELS)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, POLYWAV_CHANNELS)

    # Select requested channels
    selected = arr[:, channel_indices].T   # shape (n_ch, n_frames)
    return selected.copy(), POLYWAV_SR


def read_mems_window(
    cfg,
    remote_path: str,
    start_s: float,
    duration_s: float,
    channel_indices: Optional[List[int]] = None,
    sr: int = MEMS_SR_ASSUMED,
    n_channels: int = MEMS_CHANNELS_ASSUMED,
    bits: int = MEMS_BITS_ASSUMED,
) -> Tuple[np.ndarray, int]:
    """
    Read a time window from a remote MEMS WAV file via HTTP range.

    MEMS files are 48 kHz / 4-channel / 24-bit signed integer PCM.
    The exact format should be verified via verify_mems_format() once before use.

    Returns
    -------
    audio : np.ndarray shape (n_channels, n_samples)  float32 normalised [-1, 1]
    sample_rate : int
    """
    if channel_indices is None:
        channel_indices = list(range(n_channels))

    bytes_per_sample = (bits + 7) // 8
    bytes_per_frame  = n_channels * bytes_per_sample

    auth = _auth(cfg)
    url  = _file_url(cfg, remote_path)

    frame_start = int(start_s    * sr)
    n_frames    = int(duration_s * sr)
    byte_start  = _WAV_HEADER_BYTES + frame_start * bytes_per_frame
    byte_end    = byte_start + n_frames * bytes_per_frame

    log.debug("Range-reading MEMS: %s  t=[%.2f, %.2f)s", remote_path, start_s, start_s + duration_s)
    raw = _fetch_byte_range(url, auth, byte_start, byte_end)

    # 24-bit signed integer: unpack 3 bytes per sample
    if bits == 24:
        n_samples = len(raw) // bytes_per_frame * n_channels
        arr = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_samples):
            b = raw[i * 3: i * 3 + 3]
            if len(b) < 3:
                break
            val = int.from_bytes(b, "little", signed=True)
            arr[i] = val
        arr_2d = arr.reshape(-1, n_channels).T   # (n_ch, n_frames)
        audio  = arr_2d.astype(np.float32) / (2 ** 23)  # normalise to [-1, 1]
    elif bits == 16:
        arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels).T
        audio = arr.astype(np.float32) / 32768.0
    elif bits == 32:
        audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, n_channels).T
    else:
        raise ValueError(f"Unsupported bit depth: {bits}")

    selected = audio[channel_indices, :]
    return selected.copy(), sr


# ══════════════════════════════════════════════════════════════════════════════
# GPX parser
# ══════════════════════════════════════════════════════════════════════════════

_GPX_NS_PATS = [
    "http://www.topografix.com/GPX/1/1",
    "http://www.topografix.com/GPX/1/0",
    "",
]

def parse_gpx_bytes(raw: bytes) -> List[Dict[str, float]]:
    """
    Parse a GPX file and return a list of trackpoints.

    Each item: {lat, lon, ele, time_iso}
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        log.warning("GPX parse error: %s", exc)
        return []

    # Try each namespace pattern
    points: List[Dict] = []
    for ns_uri in _GPX_NS_PATS:
        ns = {"g": ns_uri} if ns_uri else {}
        prefix = "g:" if ns_uri else ""
        trkpts = root.findall(f".//{prefix}trkpt", ns)
        if not trkpts:
            trkpts = root.findall(f".//{prefix}wpt", ns)
        for pt in trkpts:
            try:
                lat = float(pt.attrib.get("lat", 0))
                lon = float(pt.attrib.get("lon", 0))
                ele_el = pt.find(f"{prefix}ele", ns)
                ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
                time_el = pt.find(f"{prefix}time", ns)
                t_iso = time_el.text if time_el is not None else None
                points.append({"lat": lat, "lon": lon, "ele": ele, "time_iso": t_iso})
            except (ValueError, AttributeError):
                continue
        if points:
            break

    return points


def fetch_gpx(cfg, remote_path: str) -> List[Dict[str, float]]:
    """
    Fetch a GPX file from the Nextcloud share and parse its trackpoints.
    """
    req = _requests()
    auth = _auth(cfg)
    url  = _file_url(cfg, remote_path)
    try:
        resp = req.get(url, auth=auth, timeout=30)
        resp.raise_for_status()
        return parse_gpx_bytes(resp.content)
    except Exception as exc:
        log.warning("GPX fetch failed (%s): %s", remote_path, exc)
        return []


def gpx_to_xy_waypoints(
    gpx_points: List[Dict[str, float]],
    origin_lat: float,
    origin_lon: float,
) -> List[Dict[str, float]]:
    """
    Convert GPX trackpoints to local XY metres relative to origin.

    Returns list of {x_m, y_m, z_m, time_iso}.
    X = East, Y = North, Z = Up (altitude).
    """
    R = 6_371_000.0
    cos_lat = math.cos(math.radians(origin_lat))
    out = []
    for pt in gpx_points:
        dx = (pt["lon"] - origin_lon) * math.pi / 180.0 * R * cos_lat
        dy = (pt["lat"] - origin_lat) * math.pi / 180.0 * R
        out.append({
            "x_m":     round(dx, 2),
            "y_m":     round(dy, 2),
            "z_m":     round(pt.get("ele", 0.0), 2),
            "time_iso": pt.get("time_iso"),
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Segment browser — rich metadata for /api/v2/repository/segments
# ══════════════════════════════════════════════════════════════════════════════

def build_segment_browser_response(
    cfg,
    split_filter:   Optional[str] = None,
    session_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the complete response for the /api/v2/repository/segments endpoint.

    Merges ground-truth metadata (from dunakeszi_ground_truth_fixed.py) with
    local pipeline-ready file availability (DUNAKESZI_LOCAL_PATH).

    Returns a dict with:
        segments : list — one per maneuver segment, matching the HTML browser schema
        sessions : list — one per show, for the session dropdown
        total    : int
        source   : str  "ground_truth" | "local_index" | "empty"
    """
    # Import the ground-truth module.  It lives alongside this file.
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            log.warning("dunakeszi_ground_truth_fixed not importable — falling back to local index")
            return _build_from_local_index(cfg, split_filter, session_filter)

    # Build enriched sessions and segments
    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    enriched_segs  = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)

    # Build local-availability index
    local_stems = _local_stems(cfg)

    # Build sessions list for dropdown
    sessions_out = []
    for s in sorted(sessions_by_id.values(), key=lambda x: x.get("show_number", 99)):
        sessions_out.append({
            "session_id":   s["session_id"],
            "show_number":  s.get("show_number"),
            "wall_clock":   s.get("wall_clock"),
            "description":  s.get("description"),
            "n_drones":     s.get("n_drones", 1),
            "drones":       s.get("drones", []),
            "mems_recording": s.get("mems_recording", False),
        })

    # Build segments list
    segments_out = []
    for seg in enriched_segs:
        # Apply filters
        if split_filter and seg.get("split") != split_filter:
            continue
        if session_filter and seg.get("session") != session_filter:
            continue

        sess = sessions_by_id.get(seg["session"], {})

        # Check local availability (pipeline-ready triplet exists)
        seg_stem_prefix = f"seg_{seg['id']:04d}"
        local_ok = any(s.startswith(seg_stem_prefix) for s in local_stems)

        segments_out.append({
            # ── Identity ─────────────────────────────────────────────────────
            "id":              seg["id"],
            "session":         seg["session"],
            "show_number":     sess.get("show_number"),
            "wall_clock":      sess.get("wall_clock"),
            "split":           seg.get("split"),
            # ── Timing ───────────────────────────────────────────────────────
            "local_start_hms": seg.get("local_start_hms"),
            "local_end_hms":   seg.get("local_end_hms"),
            "onset_from_rec_s": seg.get("onset_from_rec_s"),
            "duration_s":      seg.get("duration_s"),
            "within_session_offset_s": seg.get("within_session_offset_s"),
            # ── Maneuver ─────────────────────────────────────────────────────
            "maneuver_type":   seg.get("maneuver_type"),
            "flight_phase":    seg.get("flight_phase"),
            "description":     seg.get("description"),
            "n_drones":        seg.get("n_drones", 1),
            "drones":          seg.get("drones", []),
            # ── Position ─────────────────────────────────────────────────────
            "altitude_m":      seg.get("altitude_m"),
            "speed_mps":       seg.get("speed_mps"),
            "radius_m":        seg.get("radius_m"),
            "azimuth_deg":     seg.get("azimuth_deg_onset"),
            "distance_m":      seg.get("distance_xy_m_onset"),
            "start_coord":     seg.get("start_coord"),
            "end_coord":       seg.get("end_coord"),
            # ── Availability ─────────────────────────────────────────────────
            "local_available": local_ok,
            "bk_available":    seg.get("bk_available", True),
            "mems_available":  seg.get("mems_available", False),
            # ── Quality ──────────────────────────────────────────────────────
            "quality_flags":   seg.get("quality_flags", []),
            # ── Data pointers ────────────────────────────────────────────────
            "gpx_folder":      seg.get("gpx_folder"),
            "log_files":       seg.get("log_files", {}),
            # ── Show metadata (for info strip) ────────────────────────────────
            "session_description": sess.get("description"),
        })

    return {
        "segments": segments_out,
        "sessions": sessions_out,
        "total":    len(segments_out),
        "source":   "ground_truth",
        "filters":  {"split": split_filter, "session": session_filter},
    }


def _local_stems(cfg) -> set:
    """
    Return the set of stem prefixes found in the local pipeline-ready directory.
    e.g. {"seg_0001_circle_train", "seg_0002_hover_val", ...}
    """
    local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
    if not local_path or not os.path.isdir(local_path):
        return set()
    stems = set()
    for f in Path(local_path).iterdir():
        if f.suffix == ".wav" and "_ch0" in f.name:
            stem = f.name.replace("_ch0.wav", "")
            stems.add(stem)
    return stems


def _build_from_local_index(
    cfg,
    split_filter:   Optional[str],
    session_filter: Optional[str],
) -> Dict[str, Any]:
    """
    Fallback: build browser response from local pipeline-ready directory only.
    Returns fewer metadata fields but always works without ground-truth import.
    """
    local_path = getattr(cfg, "DUNAKESZI_LOCAL_PATH", None)
    if not local_path or not os.path.isdir(local_path):
        return {"segments": [], "sessions": [], "total": 0, "source": "empty"}

    import json

    root = Path(local_path)
    segments_out = []
    sessions_seen: Dict[str, dict] = {}

    for label_file in sorted(root.rglob("*_label.json")):
        try:
            raw = json.loads(label_file.read_text())
        except Exception:
            continue

        stem = label_file.name[: -len("_label.json")]
        ch0  = label_file.parent / f"{stem}_ch0.wav"
        if not ch0.exists():
            continue

        split   = raw.get("split")
        session = raw.get("session") or raw.get("session_id", "unknown")
        seg_id_str = raw.get("segment_id", stem)
        # Try to parse numeric id from stem
        m = re.search(r"seg_(\d+)", stem)
        seg_id = int(m.group(1)) if m else None

        if split_filter and split != split_filter:
            continue
        if session_filter and session != session_filter:
            continue

        sessions_seen.setdefault(session, {"session_id": session, "description": session})

        segments_out.append({
            "id":             seg_id,
            "session":        session,
            "split":          split,
            "maneuver_type":  raw.get("maneuver_type"),
            "description":    raw.get("description", stem),
            "n_drones":       raw.get("n_drones", 1),
            "altitude_m":     raw.get("height_m"),
            "azimuth_deg":    raw.get("azimuth_deg"),
            "distance_m":     raw.get("distance_m"),
            "local_available": True,
            "bk_available":   True,
            "mems_available": False,
            "quality_flags":  [],
        })

    return {
        "segments": segments_out,
        "sessions": list(sessions_seen.values()),
        "total":    len(segments_out),
        "source":   "local_index",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Streaming audio from Nextcloud into pipeline channels
# ══════════════════════════════════════════════════════════════════════════════

def stream_segment_from_nextcloud(
    cfg,
    segment_id: int,
    array: str = "BK-6-E",
    ap=None,
) -> Optional[Tuple[List[np.ndarray], dict]]:
    """
    Stream a single maneuver segment from the Nextcloud polywav files using
    HTTP range requests, resampling to cfg.SR on the fly.

    Returns (channels, label) where:
        channels : List[np.ndarray]  3 × float32 at cfg.SR
        label    : dict  (ground-truth metadata)

    Returns None if the segment cannot be loaded.
    """
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            log.error("dunakeszi_ground_truth_fixed not importable")
            return None

    # Look up the segment
    seg = next((s for s in gt.MANEUVER_SEGMENTS if s["id"] == segment_id), None)
    if seg is None:
        log.warning("stream_segment_from_nextcloud: segment_id=%d not found", segment_id)
        return None

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    enriched = gt._enrich_segments([seg], sessions_by_id)[0]

    # Determine which polywav file and byte offset to read
    onset_s    = enriched["onset_from_rec_s"]  # seconds from rec start (show_1 trigger)
    duration_s = float(enriched.get("duration_s") or 3.0)

    # Cap to 30 s for streaming
    read_dur_s = min(duration_s, 30.0)

    # Which polywav chunk?
    chunk_dur = gt.PW_CHUNK_DUR_S   # ~399.46 s
    chunk_idx = int(onset_s // chunk_dur)
    offset_in_chunk = onset_s - chunk_idx * chunk_dur

    if chunk_idx < 0 or chunk_idx >= len(gt.POLYWAV_FILES):
        log.warning("Segment %d onset %.1fs maps to chunk %d — out of range", segment_id, onset_s, chunk_idx)
        return None

    pw_filename = gt.POLYWAV_FILES[chunk_idx]
    pw_path_cfg = getattr(cfg, "NEXTCLOUD_POLYWAV_PATH", "/polywav")
    remote_path = f"{pw_path_cfg}/{pw_filename}"

    # Channel selection
    ch_map = getattr(cfg, "DUNAKESZI_ARRAY_CHANNELS", {"BK-6-E": BK6E_CHANNELS, "BK-6-W": BK6W_CHANNELS})
    channel_indices = ch_map.get(array, BK6E_CHANNELS)

    try:
        log.info(
            "Streaming segment %d from %s  onset=%.1fs  dur=%.1fs  ch=%s",
            segment_id, pw_filename, offset_in_chunk, read_dur_s, channel_indices,
        )
        audio_raw, native_sr = read_polywav_window(
            cfg, remote_path, offset_in_chunk, read_dur_s, channel_indices
        )
    except RemoteUnavailableError as exc:
        log.warning("Nextcloud unavailable for segment %d: %s", segment_id, exc)
        return None
    except RangeRequestError as exc:
        log.warning("Range request failed for segment %d: %s", segment_id, exc)
        return None

    # Resample to cfg.SR and pad/truncate via AudioProcessor
    if ap is None:
        from .audio_processing import AudioProcessor
        ap = AudioProcessor(cfg)

    out_channels = []
    for ch_row in audio_raw:
        if native_sr != cfg.SR:
            import librosa
            ch_row = librosa.resample(ch_row, orig_sr=native_sr, target_sr=cfg.SR)
        out_channels.append(ap.pad_or_truncate(ch_row))

    while len(out_channels) < 3:
        out_channels.append(out_channels[-1].copy())
    out_channels = out_channels[:3]

    # Build label from ground-truth metadata
    from math import atan2, degrees, hypot
    sc = enriched.get("start_coord")
    if sc and sc[0] is not None:
        bearing = round(degrees(atan2(sc[0], sc[1])), 1)   # geographic bearing (N=0, E=90)
        dist    = round(hypot(sc[0], sc[1]), 1)
        ht      = round(float(sc[2]) if len(sc) > 2 else enriched.get("altitude_m", 0), 1)
        # Convert to pipeline math angle (from-East, CCW)
        pipeline_az = (90.0 - bearing + 180.0) % 360.0 - 180.0
    else:
        pipeline_az = None
        dist        = None
        ht          = enriched.get("altitude_m")

    label = {
        "segment_id":    segment_id,
        "session":       enriched.get("session"),
        "split":         enriched.get("split"),
        "source":        "real",
        "dataset_type":  "dunakeszi",
        "array":         array,
        "maneuver_type": enriched.get("maneuver_type"),
        "flight_phase":  enriched.get("flight_phase"),
        "description":   enriched.get("description"),
        "n_drones":      enriched.get("n_drones", 1),
        "drones":        enriched.get("drones", []),
        "azimuth_deg":   pipeline_az,
        "distance_m":    dist,
        "height_m":      ht,
        "has_position":  pipeline_az is not None and dist is not None,
        "speed_mps":     enriched.get("speed_mps"),
        "radius_m":      enriched.get("radius_m"),
        "duration_s":    read_dur_s,
        "local_start_hms": enriched.get("local_start_hms"),
        "show_number":   sessions_by_id.get(enriched.get("session", ""), {}).get("show_number"),
        "audio_file":    pw_filename,
        "polywav_chunk": chunk_idx,
        "polywav_offset_s": offset_in_chunk,
    }

    return out_channels, label


# ══════════════════════════════════════════════════════════════════════════════
# Polywav file info (for the file browser table)
# ══════════════════════════════════════════════════════════════════════════════

def iter_nextcloud_segments(
    cfg,
    ap,
    required_split: Optional[str] = None,
    segment_id:     Optional[int] = None,
    loop:           bool           = True,
    array:          str            = "BK-6-E",
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Yield (channels, label) tuples by streaming individual maneuver segments
    from the Nextcloud polywav files via HTTP range requests.

    Called by stream_repository_segments() as Strategy 0 whenever
    NEXTCLOUD_BASE_URL and NEXTCLOUD_SHARE_TOKEN are configured.

    Parameters
    ----------
    cfg            : Config with NEXTCLOUD_* keys
    ap             : AudioProcessor instance (already constructed by caller)
    required_split : restrict to "train" | "val" | "test" | None
    segment_id     : if set, yield only this segment once (loop forced False)
    loop           : if True, cycle through segments indefinitely
    array          : "BK-6-E" | "BK-6-W"

    Raises
    ------
    RemoteUnavailableError  if Nextcloud credentials are absent
    RuntimeError            if ground-truth module is missing or no segments match
    """
    # Validate credentials early so the caller gets a clean exception
    _auth(cfg)   # raises RemoteUnavailableError if not configured

    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            raise RuntimeError(
                "dunakeszi_ground_truth_fixed not importable — cannot iterate "
                "Nextcloud segments without the ground-truth metadata module."
            )

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    all_segs       = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)

    # Apply filters
    if required_split is not None:
        all_segs = [s for s in all_segs if s.get("split") == required_split]
    if segment_id is not None:
        all_segs = [s for s in all_segs if s["id"] == segment_id]
        loop     = False   # single-segment → always one shot

    if not all_segs:
        raise RuntimeError(
            f"iter_nextcloud_segments: no segments match "
            f"split={required_split!r}, segment_id={segment_id!r}"
        )

    log.info(
        "iter_nextcloud_segments: %d segment(s) to stream from Nextcloud "
        "(loop=%s, split=%r, array=%s)",
        len(all_segs), loop, required_split, array,
    )

    import random as _random

    first_pass = True
    while True:
        if not first_pass:
            if not loop:
                return
            _random.shuffle(all_segs)
        first_pass = False

        for seg in all_segs:
            try:
                result = stream_segment_from_nextcloud(
                    cfg, seg["id"], array=array, ap=ap
                )
            except Exception as exc:
                log.debug(
                    "iter_nextcloud_segments: segment %d skipped (%s)",
                    seg["id"], exc,
                )
                continue

            if result is None:
                log.debug(
                    "iter_nextcloud_segments: segment %d returned None — skipping",
                    seg["id"],
                )
                continue

            channels, label = result
            yield channels, label

        if not loop:
            return


def polywav_file_info(cfg) -> List[Dict[str, Any]]:
    """
    Return metadata about each polywav chunk: name, duration, shows covered,
    and whether the file is available on the remote Nextcloud share.

    Tries to list files from Nextcloud; if unavailable falls back to the
    ground-truth POLYWAV_FILES list with remote_available=None.
    """
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            return []

    # Try remote listing
    try:
        remote_files = list_polywav_files(cfg)
        remote_names = {f["name"]: f for f in remote_files}
    except RemoteUnavailableError:
        remote_names = {}

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    all_segs = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)

    result = []
    for idx, fname in enumerate(gt.POLYWAV_FILES):
        chunk_start = idx * gt.PW_CHUNK_DUR_S
        chunk_end   = chunk_start + gt.PW_CHUNK_DUR_S

        # Which segments does this chunk overlap?
        covered = [
            s["id"] for s in all_segs
            if s["onset_from_rec_s"] < chunk_end
            and s["onset_from_rec_s"] + s["duration_s"] > chunk_start
        ]
        shows = sorted({
            all_segs[next((i for i, s in enumerate(all_segs) if s["id"] == sid), 0)]["session"]
            for sid in covered
            if any(s["id"] == sid for s in all_segs)
        })

        ri = remote_names.get(fname)
        result.append({
            "chunk_index":      idx,
            "filename":         fname,
            "chunk_start_s":    round(chunk_start, 2),
            "chunk_end_s":      round(chunk_end, 2),
            "duration_s":       round(gt.PW_CHUNK_DUR_S, 2),
            "size_bytes":       ri["size_bytes"] if ri else 4 * 1024 ** 3,
            "remote_available": ri is not None if remote_names else None,
            "segment_ids":      covered,
            "shows":            shows,
        })

    return result


def mems_file_info(cfg) -> List[Dict[str, Any]]:
    """
    Return metadata about each MEMS audio file: name, assumed timing,
    which segments it covers, and remote availability.
    """
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            return []

    try:
        remote_files = list_mems_files(cfg)
        remote_names = {f["name"]: f for f in remote_files}
    except RemoteUnavailableError:
        remote_names = {}

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    all_segs = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)

    mems_start_rec_s = gt.MEMS_START_LOCAL_S - gt.RECORDING_REF_LOCAL_S
    file_dur_s = gt.MEMS_ASSUMED_FORMAT["duration_s"]

    result = []
    for idx, fname in enumerate(gt.MEMS_FILES):
        fstart = mems_start_rec_s + idx * file_dur_s
        fend   = fstart + file_dur_s

        covered = [
            s["id"] for s in all_segs
            if s["onset_from_rec_s"] < fend
            and s["onset_from_rec_s"] + s["duration_s"] > fstart
            and s.get("mems_available", False)
        ]

        ri = remote_names.get(fname)
        result.append({
            "file_index":        idx,
            "filename":          fname,
            "mems_start_rec_s":  round(fstart, 2),
            "mems_end_rec_s":    round(fend, 2),
            "duration_s":        round(file_dur_s, 2),
            "size_bytes":        ri["size_bytes"] if ri else None,
            "remote_available":  ri is not None if remote_names else None,
            "segment_ids":       covered,
        })

    return result