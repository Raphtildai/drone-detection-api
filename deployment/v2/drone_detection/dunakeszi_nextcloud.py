# dunakeszi_nextcloud.py
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

# Bytes that are illegal in XML 1.0 content (outside TAB/LF/CR) — Nextcloud
# has been observed to emit these unescaped inside <d:href>/<d:displayname>
# when a filename on the share contains them, which breaks ET.fromstring().
_XML_ILLEGAL_BYTES_RE = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
# Fallback-only defaults, used when the WAV header can't be probed at all.
# The BRUEL/Norsonic recorder actually writes 14-channel, 24-bit signed PCM
# (wFormatTag=1, not IEEE float) — confirmed against a real polywav header,
# which also carries bext/iXML metadata chunks pushing the data offset well
# past the naive 44/80-byte assumption. Real files are always probed via
# _parse_wav_format()/_get_wav_format(), which return the true channel count
# and sample format; these constants only matter if that probe fails outright.
POLYWAV_CHANNELS = 14               # total channels in each polywav file
POLYWAV_BYTES_PER_SAMPLE = 3         # 24-bit PCM → 3 bytes per sample per channel
POLYWAV_BYTES_PER_FRAME  = POLYWAV_CHANNELS * POLYWAV_BYTES_PER_SAMPLE  # 42

MEMS_SR_ASSUMED       = 48_000
MEMS_CHANNELS_ASSUMED = 4
MEMS_BITS_ASSUMED     = 24          # 24-bit signed int → 3 bytes/sample
MEMS_BYTES_PER_FRAME  = MEMS_CHANNELS_ASSUMED * 3  # 12

# Channel indices for the two BK-6 arrays inside the polywav (0-indexed)
BK6E_CHANNELS = [8, 9, 10]    # East array: E-E, E-H, E-B
BK6W_CHANNELS = [2, 3, 4]     # West array: W-E, W-H, W-B

# WebDAV namespace
_DAV_NS = "DAV:"

# WAV header size in bytes (standard 44-byte PCM header).
# NOTE: this constant is kept for reference only.  Multi-channel / non-PCM
# WAV files produced by the polywav recorder carry an extended fmt chunk
# and often a "fact" chunk, pushing the data offset to 60–80+ bytes.
# Always use _get_wav_data_offset() instead of this literal.
_WAV_HEADER_BYTES = 44

# Polywav-specific fallback data offset.
#
# The Dunakeszi polywav files are 14-channel, 24-bit signed-PCM WAV/RF64 files
# recorded by the BRUEL multi-channel system — NOT IEEE float32 as originally
# assumed here. Real files also carry bext/iXML broadcast-metadata chunks of
# variable size (observed: ds64 + bext + iXML + fmt ≈ 6 KB) before the data
# chunk, so there is no fixed "true" offset — this 80-byte value is only a
# last-resort guess for when the header can't be probed at all; real reads
# always go through _parse_wav_format()/_get_wav_format(), which walk the
# actual RIFF chunks to find the real offset, channel count and sample format.
#
# This value is used as the fallback when the remote server cannot serve the
# first 256 bytes via an HTTP 206 range response (e.g. when it returns 200 OK
# for small ranges but honours Range for large data reads).
_POLYWAV_DATA_OFFSET = 80

# Cache: url → byte offset of the "data" chunk payload start
_WAV_DATA_OFFSET_CACHE: Dict[str, int] = {}

# BRUEL/Norsonic polywav recorders write float32 max (0x7F7FFFFF ≈ 3.4e38) as a
# sentinel value for channels with no connected microphone or clipped/invalid
# samples.  numpy.isfinite() returns True for FLT_MAX, so a plain isfinite()
# check silently passes these sentinels through — causing inf RMS, overflow
# in mel-spectrogram computation, and zero detections.
# Any |sample| > _SENTINEL_THRESHOLD is treated as invalid and zeroed.
_SENTINEL_THRESHOLD = np.float32(1e10)


def _sanitize_audio(arr: np.ndarray, label: str) -> Tuple[np.ndarray, int]:
    """
    Replace NaN, Inf, and BRUEL FLT_MAX sentinel values (|x| > 1e10) with 0.

    Returns (sanitized_array, n_bad_samples).
    Operates in-place on a copy; never modifies the input.
    """
    bad_mask = ~np.isfinite(arr) | (np.abs(arr) > _SENTINEL_THRESHOLD)
    n_bad    = int(np.sum(bad_mask))
    if n_bad:
        arr = arr.copy()
        arr[bad_mask] = 0.0
    return arr, n_bad


def _decode_pcm_frames(raw: bytes, n_channels: int, bytes_per_sample: int, is_float: bool) -> np.ndarray:
    """
    Decode interleaved raw audio bytes into shape (n_frames, n_channels) float32.

    Handles the sample formats actually seen in the wild here: 32-bit IEEE
    float (polywav's originally-assumed format) and 16/24/32-bit signed PCM
    (the format the BRUEL recorder actually writes). 24-bit has no native
    numpy dtype, so it's unpacked from 3 raw bytes per sample and sign-extended
    — done vectorised (no per-sample Python loop) since polywav windows can be
    millions of frames at 192 kHz.
    """
    frame_bytes = n_channels * bytes_per_sample
    usable = (len(raw) // frame_bytes) * frame_bytes
    raw = raw[:usable]

    if bytes_per_sample == 4 and is_float:
        return np.frombuffer(raw, dtype="<f4").reshape(-1, n_channels)
    if bytes_per_sample == 2:
        arr = np.frombuffer(raw, dtype="<i2").reshape(-1, n_channels)
        return arr.astype(np.float32) / 32768.0
    if bytes_per_sample == 3:
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, n_channels, 3)
        as_i32 = (
            buf[..., 0].astype(np.int32)
            | (buf[..., 1].astype(np.int32) << 8)
            | (buf[..., 2].astype(np.int32) << 16)
        )
        as_i32 = np.where(as_i32 & 0x800000, as_i32 - 0x1000000, as_i32)
        return as_i32.astype(np.float32) / 8388608.0
    if bytes_per_sample == 4 and not is_float:
        arr = np.frombuffer(raw, dtype="<i4").reshape(-1, n_channels)
        return arr.astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported sample format: bytes_per_sample={bytes_per_sample} is_float={is_float}")


# ============================================================================
# Cache: url/path → (data_offset, n_channels, bytes_per_sample, is_float)
# ============================================================================
_WAV_FORMAT_CACHE: Dict[str, Tuple[int, int, int, bool]] = {}

def _parse_wav_format(header: bytes) -> Tuple[int, int, int, bool]:
    """Parse a WAV/RF64 header. Returns (data_offset, n_channels, bytes_per_sample, is_float).
    Falls back to (44, POLYWAV_CHANNELS, POLYWAV_BYTES_PER_SAMPLE, False) if parsing fails."""
    import struct
    n_channels       = POLYWAV_CHANNELS         # fallback only
    bytes_per_sample = POLYWAV_BYTES_PER_SAMPLE # fallback only
    is_float         = False                    # fallback only
    data_offset      = _WAV_HEADER_BYTES

    def _fallback():
        return data_offset, n_channels, bytes_per_sample, is_float

    try:
        if len(header) < 12:
            return _fallback()
        riff_id = header[:4]
        wave_id = header[8:12]
        if riff_id not in (b"RIFF", b"RF64", b"BW64") or wave_id != b"WAVE":
            return _fallback()

        pos = 12
        while pos + 8 <= len(header):
            chunk_id   = header[pos:pos + 4]
            chunk_size = struct.unpack_from("<I", header, pos + 4)[0]

            if chunk_id == b"fmt " and pos + 8 + 16 <= len(header):
                # WAVEFORMATEX layout: wFormatTag(2) nChannels(2) nSamplesPerSec(4)
                #                       nAvgBytesPerSec(4) nBlockAlign(2) wBitsPerSample(2)
                fmt_tag, n_channels_p, _, _, _, bits_per_sample = struct.unpack_from(
                    "<HHIIHH", header, pos + 8
                )
                n_channels       = n_channels_p
                bytes_per_sample = max(1, bits_per_sample // 8)
                # 1 = WAVE_FORMAT_PCM (int), 3 = WAVE_FORMAT_IEEE_FLOAT.
                # 0xFFFE (WAVE_FORMAT_EXTENSIBLE) carries the real tag in a
                # GUID subformat we don't parse — treat as int PCM, the
                # common case for multichannel recorders that use it.
                is_float = fmt_tag == 3

            if chunk_id == b"data":
                data_offset = pos + 8
                return data_offset, n_channels, bytes_per_sample, is_float

            pos += 8 + chunk_size
            if chunk_size % 2:
                pos += 1

        return _fallback()
    except Exception as exc:
        log.debug("_parse_wav_format error: %s", exc)
        return _fallback()


def _get_wav_format(url: str, auth: Tuple[str, str]) -> Tuple[int, int, int, bool]:
    """Same probing strategy as _get_wav_data_offset(), but also returns
    n_channels, bytes_per_sample and is_float."""
    if url in _WAV_FORMAT_CACHE:
        return _WAV_FORMAT_CACHE[url]

    header_bytes: Optional[bytes] = None
    try:
        header_bytes = _fetch_byte_range(url, auth, 0, 8192, timeout=15)
    except Exception as exc:
        log.debug("_get_wav_format: range probe failed for %s: %s", url, exc)

    if header_bytes is None:
        try:
            req  = _requests()
            resp = req.get(url, auth=auth, timeout=20, stream=True)
            if resp.status_code in (200, 206):
                header_bytes = b""
                for chunk in resp.iter_content(chunk_size=512):
                    header_bytes += chunk
                    if len(header_bytes) >= 8192:
                        break
                header_bytes = header_bytes[:8192]
            resp.close()
        except Exception as exc:
            log.debug("_get_wav_format: streaming probe error for %s: %s", url, exc)

    if header_bytes is None:
        log.warning("_get_wav_format: all probes failed for %s — using fallback %d ch / %d bytes",
                     url, POLYWAV_CHANNELS, _POLYWAV_DATA_OFFSET)
        result = (_POLYWAV_DATA_OFFSET, POLYWAV_CHANNELS, POLYWAV_BYTES_PER_SAMPLE, False)
        _WAV_FORMAT_CACHE[url] = result
        return result

    data_offset, n_channels, bytes_per_sample, is_float = _parse_wav_format(header_bytes)
    if data_offset == _WAV_HEADER_BYTES:
        data_offset = _POLYWAV_DATA_OFFSET

    log.info(
        "WAV format for %s: data_offset=%d bytes, n_channels=%d, bytes_per_sample=%d, is_float=%s",
        url, data_offset, n_channels, bytes_per_sample, is_float,
    )
    result = (data_offset, n_channels, bytes_per_sample, is_float)
    _WAV_FORMAT_CACHE[url] = result
    return result

def _get_wav_data_offset(url: str, auth: Tuple[str, str]) -> int:
    """
    Return the byte offset at which the PCM payload starts in a remote WAV file.

    Strategy
    --------
    1. Try HTTP 206 range request for bytes 0–511 (fast, no full download).
    2. If the server returns 200 (ignores Range) or any non-206, fall back to a
       streaming GET that reads only the first 512 bytes from the response body
       and immediately closes the connection — so we never download the full file.
    3. If both attempts fail, use _POLYWAV_DATA_OFFSET (80 bytes) which is the
       correct layout for BRUEL 14-channel IEEE float32 WAV files.

    The result is cached so each remote file is probed only once per process.
    """
    if url in _WAV_DATA_OFFSET_CACHE:
        return _WAV_DATA_OFFSET_CACHE[url]

    header_bytes: Optional[bytes] = None

    # ── Attempt 1: proper range request (206) ─────────────────────────────────
    try:
        header_bytes = _fetch_byte_range(url, auth, 0, 8192, timeout=15)
    except RangeRequestError as exc:
        log.debug(
            "_get_wav_data_offset: range probe returned non-206 for %s (%s) "
            "— trying streaming fallback",
            url, exc,
        )
    except Exception as exc:
        log.debug("_get_wav_data_offset: range probe error for %s: %s", url, exc)

    # ── Attempt 2: streaming GET — read first 512 bytes, close immediately ────
    if header_bytes is None:
        try:
            req  = _requests()
            resp = req.get(url, auth=auth, timeout=20, stream=True)
            if resp.status_code in (200, 206):
                header_bytes = b""
                for chunk in resp.iter_content(chunk_size=512):
                    header_bytes += chunk
                    if len(header_bytes) >= 8192:
                        break
                header_bytes = header_bytes[:8192]
            resp.close()
        except Exception as exc:
            log.debug("_get_wav_data_offset: streaming probe error for %s: %s", url, exc)

    if header_bytes is None:
        log.warning(
            "_get_wav_data_offset: all probe attempts failed for %s — "
            "using polywav-specific fallback offset %d bytes",
            url, _POLYWAV_DATA_OFFSET,
        )
        _WAV_DATA_OFFSET_CACHE[url] = _POLYWAV_DATA_OFFSET
        return _POLYWAV_DATA_OFFSET

    offset = _parse_wav_data_offset(header_bytes)

    # If parsing still fell back to 44 (e.g. the streaming probe returned HTML
    # because the URL requires a session cookie), override with the known polywav
    # layout rather than using the wrong value.
    if offset == _WAV_HEADER_BYTES:
        log.warning(
            "_get_wav_data_offset: header parse returned fallback 44 for %s "
            "— overriding with polywav-specific offset %d bytes",
            url, _POLYWAV_DATA_OFFSET,
        )
        offset = _POLYWAV_DATA_OFFSET

    log.debug("WAV data offset for %s: %d bytes", url, offset)
    _WAV_DATA_OFFSET_CACHE[url] = offset
    return offset


def _parse_wav_data_offset(header: bytes) -> int:
    import struct
    try:
        if len(header) < 12:
            log.debug("_parse_wav_data_offset: header too short, using 44")
            return _WAV_HEADER_BYTES

        riff_id = header[:4]
        wave_id = header[8:12]

        # RF64/BW64: RIFF variant used once a file exceeds ~4 GiB (the 32-bit
        # RIFF size field overflows). Structurally identical past byte 12 —
        # carries a 'ds64' chunk (real 64-bit sizes) before fmt/data, which
        # the generic chunk walker below skips over like any other chunk.
        if riff_id not in (b"RIFF", b"RF64", b"BW64") or wave_id != b"WAVE":
            log.debug(
                "_parse_wav_data_offset: unrecognised container %r/%r, using 44",
                riff_id, wave_id,
            )
            return _WAV_HEADER_BYTES

        pos = 12
        while pos + 8 <= len(header):
            chunk_id   = header[pos: pos + 4]
            chunk_size = struct.unpack_from("<I", header, pos + 4)[0]
            if chunk_id == b"data":
                return pos + 8
            pos += 8 + chunk_size
            if chunk_size % 2:
                pos += 1

        log.warning(
            "_parse_wav_data_offset: 'data' chunk not found in first %d bytes; "
            "using fallback offset 44 — audio may be misaligned",
            len(header),
        )
        return _WAV_HEADER_BYTES

    except Exception as exc:
        log.debug("_parse_wav_data_offset error: %s", exc)
        return _WAV_HEADER_BYTES


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
    """
    Build full URL for a file inside the share for WebDAV operations
    (PROPFIND, MKCOL, etc.).  Uses /public.php/webdav/<path>.
    """
    base = _share_dav_url(cfg)
    rpath = remote_path.lstrip("/")
    return f"{base}/{rpath}"


# Cache: base_url → working download URL template ("dav" | "legacy")
_NC_DOWNLOAD_STYLE_CACHE: Dict[str, str] = {}


def _file_download_url(cfg, remote_path: str) -> str:
    """
    Build the direct-download URL for a file on a Nextcloud public share.

    This is DISTINCT from the WebDAV URL used for PROPFIND.  Range requests
    for binary data must go to the download endpoint, otherwise Nextcloud
    may return an HTML redirect/login page instead of raw bytes.

    Two URL patterns are tried (and the working one is cached per host):

      Modern  (NC 25+):  /public.php/dav/files/<token>/<path>
      Legacy  (NC <25):  /public.php/webdav/<path>   (same as WebDAV)

    The modern URL does not require the share token in Basic Auth — the token
    is embedded in the path — but we send it anyway for compatibility.
    """
    base  = getattr(cfg, "NEXTCLOUD_BASE_URL", "").rstrip("/")
    token = getattr(cfg, "NEXTCLOUD_SHARE_TOKEN", "")
    rpath = remote_path.lstrip("/")

    cached = _NC_DOWNLOAD_STYLE_CACHE.get(base)
    if cached == "share_dl":
        # /index.php/s/<token>/download?path=<dir>&files=<filename>
        # Split rpath into directory and filename for the query params.
        parts    = rpath.rsplit("/", 1)
        dl_dir   = "/" + parts[0] if len(parts) > 1 else "/"
        dl_file  = parts[-1]
        return f"{base}/index.php/s/{token}/download?path={dl_dir}&files={dl_file}"
    if cached == "dav":
        return f"{base}/public.php/dav/files/{token}/{rpath}"
    if cached == "legacy":
        return f"{base}/public.php/webdav/{rpath}"

    # ── Auto-probe: find the first URL style that returns HTTP 206 ────────────
    #
    # We MUST see 206 Partial Content — a 200 OK means the server ignored the
    # Range header and returned the full file, which would corrupt audio decoding.
    #
    # Priority order:
    #   1. /index.php/s/<token>/download  — public-share download endpoint; this
    #      is the only endpoint guaranteed to honour Range on all NC versions.
    #   2. /public.php/dav/files/<token>/ — modern DAV (NC 25+), often range-capable.
    #   3. /public.php/webdav/            — legacy WebDAV fallback (rarely range-ok).

    req  = _requests()
    auth = _auth(cfg)

    parts   = rpath.rsplit("/", 1)
    dl_dir  = "/" + parts[0] if len(parts) > 1 else "/"
    dl_file = parts[-1]

    candidates = [
        ("share_dl", f"{base}/index.php/s/{token}/download?path={dl_dir}&files={dl_file}"),
        ("dav",      f"{base}/public.php/dav/files/{token}/{rpath}"),
        ("legacy",   f"{base}/public.php/webdav/{rpath}"),
    ]

    for style, probe_url in candidates:
        try:
            resp = req.head(
                probe_url, auth=auth, timeout=10,
                headers={"Range": "bytes=0-3"}, allow_redirects=True,
            )
            if resp.status_code == 206:
                log.info(
                    "Nextcloud download URL style: %s (%s) — %s",
                    style, probe_url.split("?")[0], base,
                )
                _NC_DOWNLOAD_STYLE_CACHE[base] = style
                # Return the correct URL for the actual rpath
                if style == "share_dl":
                    return probe_url   # already built for this rpath
                if style == "dav":
                    return f"{base}/public.php/dav/files/{token}/{rpath}"
                return f"{base}/public.php/webdav/{rpath}"
        except Exception:
            continue

    # No style returned 206.  Log a clear warning and fall back to share_dl
    # anyway — it is the most likely to work even if HEAD is blocked.
    log.warning(
        "_file_download_url: no URL style returned HTTP 206 for %s — "
        "falling back to share_dl (index.php/s/<token>/download).  "
        "If range requests keep failing, verify that your Nextcloud allows "
        "Range requests on public shares (check nginx/Apache proxy config).",
        base,
    )
    _NC_DOWNLOAD_STYLE_CACHE[base] = "share_dl"
    return f"{base}/index.php/s/{token}/download?path={dl_dir}&files={dl_file}"


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

    # Parse the multi-status XML. Strip bytes that are illegal in XML 1.0
    # content first (see _XML_ILLEGAL_BYTES_RE) rather than losing the whole
    # directory listing over one bad filename.
    content = _XML_ILLEGAL_BYTES_RE.sub(b"", resp.content)

    # This share's server has been observed to return the PROPFIND response
    # body duplicated — a second `<?xml ...?>` declaration appears mid
    # document, which is illegal (only one is allowed, at position 0) and
    # makes the whole thing unparseable. Discard everything from the second
    # declaration onward and parse just the first (complete) document.
    xml_decl_positions = [m.start() for m in re.finditer(rb"<\?xml\b", content)]
    if len(xml_decl_positions) > 1:
        content = content[:xml_decl_positions[1]]

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        pos = getattr(exc, "position", None)
        snippet = ""
        if pos:
            line_no, col_no = pos
            lines = content.decode("utf-8", errors="replace").splitlines()
            if 0 <= line_no - 1 < len(lines):
                snippet = lines[line_no - 1][max(0, col_no - 40):col_no + 40]
        log.warning("WebDAV XML parse error: %s | near: %r", exc, snippet)
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

    IMPORTANT: we require HTTP 206 Partial Content.  HTTP 200 means the server
    ignored our Range header and returned the full file; accepting 200 here would
    cause read_polywav_window to decode the WAV header (bytes 0-N of the full
    file) as PCM audio instead of the requested PCM window — producing millions
    of NaN/Inf values.
    """
    req = _requests()
    headers = {"Range": f"bytes={byte_start}-{byte_end - 1}"}
    try:
        resp = req.get(url, auth=auth, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            raise RangeRequestError(
                f"Server returned HTTP 200 (full file) instead of 206 Partial "
                f"Content for {url} — the server does not honour Range requests "
                f"on this endpoint.  byte_start={byte_start}"
            )
        if resp.status_code != 206:
            raise RangeRequestError(
                f"Range request failed: HTTP {resp.status_code} for {url} "
                f"(range {byte_start}-{byte_end - 1})"
            )
        return resp.content
    except RangeRequestError:
        raise
    except Exception as exc:
        raise RangeRequestError(f"Range request error: {exc}") from exc


def read_polywav_window(
    cfg, remote_path: str, start_s: float, duration_s: float,
    channel_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    if channel_indices is None:
        channel_indices = list(range(POLYWAV_CHANNELS))

    auth = _auth(cfg)
    url  = _file_download_url(cfg, remote_path)
    data_offset, n_channels, bytes_per_sample, is_float = _get_wav_format(url, auth)

    if max(channel_indices) >= n_channels:
        raise RangeRequestError(
            f"read_polywav_window: requested channels {channel_indices} but file "
            f"{remote_path} only has {n_channels} channels"
        )

    bytes_per_frame = n_channels * bytes_per_sample

    frame_start  = int(start_s    * POLYWAV_SR)
    n_frames     = int(duration_s * POLYWAV_SR)

    byte_start = data_offset + frame_start * bytes_per_frame
    byte_end   = byte_start + n_frames * bytes_per_frame

    log.debug(
        "Range-reading polywav: %s  t=[%.2f, %.2f)s  bytes=[%d, %d)",
        remote_path, start_s, start_s + duration_s, byte_start, byte_end,
    )

    raw = _fetch_byte_range(url, auth, byte_start, byte_end)

    expected_bytes = n_frames * bytes_per_frame
    actual_bytes   = len(raw)
    if actual_bytes < expected_bytes:
        log.debug(
            "read_polywav_window: server returned %d bytes, expected %d — "
            "trimming to %d complete frames",
            actual_bytes, expected_bytes, actual_bytes // bytes_per_frame,
        )
    usable = (actual_bytes // bytes_per_frame) * bytes_per_frame
    if usable == 0:
        raise RangeRequestError(
            f"read_polywav_window: no complete frames in server response "
            f"({actual_bytes} bytes, frame_size={bytes_per_frame})"
        )

    arr = _decode_pcm_frames(raw[:usable], n_channels, bytes_per_sample, is_float)

    # ── Select the channels we actually care about FIRST ──────────────────────
    selected_raw = arr[:, channel_indices]   # shape (n_frames, n_selected_ch)

    # Sanitize + measure invalid rate ONLY over the channels we're using.
    # Other channels in the 14-ch polywav may be legitimately unconnected
    # (BRUEL sentinel FLT_MAX) and would otherwise pollute this check and
    # cause spurious cache invalidation even when our channels are perfectly
    # aligned and clean.
    selected_clean, n_bad = _sanitize_audio(selected_raw, remote_path)
    if n_bad:
        n_tot = selected_clean.size
        pct   = 100.0 * n_bad / n_tot
        log.warning(
            "read_polywav_window: %d / %d invalid sample(s) (%.1f%%) in "
            "selected channels %s at t=%.2f s (NaN/Inf/sentinel) — "
            "replacing with zeros (data_offset=%d, file=%s)",
            n_bad, n_tot, pct, channel_indices, start_s, data_offset, remote_path,
        )
        if pct > 5.0:
            log.warning(
                "read_polywav_window: invalid rate %.1f%% > 5%% in selected "
                "channels %s — data offset %d may be wrong.  Invalidating "
                "cache for %s.",
                pct, channel_indices, data_offset, url,
            )
            _WAV_DATA_OFFSET_CACHE.pop(url, None)

    selected = selected_clean.T   # shape (n_ch, n_frames)
    return selected.copy(), POLYWAV_SR


def read_polywav_window_local(
    local_path: str,
    start_s: float,
    duration_s: float,
    channel_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Read a time window from a LOCAL 192 kHz / 14-channel polywav file.

    Identical semantics to read_polywav_window() but reads from disk via
    numpy mmap — no HTTP, no auth, no range-request machinery.

    Parameters
    ----------
    local_path      : Absolute path to the .wav file on disk
    start_s         : Start offset in seconds within the file
    duration_s      : Window length in seconds
    channel_indices : 0-indexed channels to extract (default: all 14)

    Returns
    -------
    audio       : np.ndarray  shape (n_channels, n_samples)  float32
    sample_rate : int         always POLYWAV_SR (192 000)
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"read_polywav_window_local: file not found: {local_path}")

    # ── Determine PCM format from the WAV header ──────────────────────────────
    # Uses the same chunk-walking probe as the remote path (_parse_wav_format),
    # which — unlike the old local-only probe — actually detects the real
    # channel count and sample format (the BRUEL recorder writes 24-bit PCM,
    # not the IEEE float32 originally assumed) instead of hardcoding them.
    cache_key = str(path)
    if cache_key in _WAV_FORMAT_CACHE:
        data_offset, n_channels, bytes_per_sample, is_float = _WAV_FORMAT_CACHE[cache_key]
    else:
        try:
            with open(path, "rb") as fh:
                header_bytes = fh.read(8192)
            data_offset, n_channels, bytes_per_sample, is_float = _parse_wav_format(header_bytes)
            if data_offset == _WAV_HEADER_BYTES:
                log.warning(
                    "read_polywav_window_local: header parse returned 44 for %s "
                    "— overriding with polywav-specific offset %d bytes",
                    local_path, _POLYWAV_DATA_OFFSET,
                )
                data_offset = _POLYWAV_DATA_OFFSET
        except Exception as exc:
            log.warning(
                "read_polywav_window_local: header probe failed for %s (%s) "
                "— using polywav-specific fallback format",
                local_path, exc,
            )
            data_offset, n_channels, bytes_per_sample, is_float = (
                _POLYWAV_DATA_OFFSET, POLYWAV_CHANNELS, POLYWAV_BYTES_PER_SAMPLE, False,
            )
        _WAV_FORMAT_CACHE[cache_key] = (data_offset, n_channels, bytes_per_sample, is_float)
        log.info(
            "Local WAV format for %s: data_offset=%d bytes, n_channels=%d, "
            "bytes_per_sample=%d, is_float=%s",
            local_path, data_offset, n_channels, bytes_per_sample, is_float,
        )

    if channel_indices is None:
        channel_indices = list(range(n_channels))
    if max(channel_indices) >= n_channels:
        raise RangeRequestError(
            f"read_polywav_window_local: requested channels {channel_indices} but file "
            f"{local_path} only has {n_channels} channels"
        )

    bytes_per_frame = n_channels * bytes_per_sample
    frame_start = int(start_s    * POLYWAV_SR)
    n_frames    = int(duration_s * POLYWAV_SR)
    byte_start  = data_offset + frame_start * bytes_per_frame
    byte_end    = byte_start + n_frames * bytes_per_frame

    log.debug(
        "Local-reading polywav: %s  t=[%.2f, %.2f)s  bytes=[%d, %d)",
        path.name, start_s, start_s + duration_s, byte_start, byte_end,
    )

    # ── Read the byte window directly from disk ───────────────────────────────
    file_size = path.stat().st_size
    if byte_start >= file_size:
        raise RangeRequestError(
            f"read_polywav_window_local: byte_start={byte_start} is beyond "
            f"file size {file_size} for {local_path}"
        )
    actual_end = min(byte_end, file_size)

    with open(path, "rb") as fh:
        fh.seek(byte_start)
        raw = fh.read(actual_end - byte_start)

    actual_bytes = len(raw)
    usable = (actual_bytes // bytes_per_frame) * bytes_per_frame
    if usable == 0:
        raise RangeRequestError(
            f"read_polywav_window_local: no complete frames read "
            f"({actual_bytes} bytes, frame_size={bytes_per_frame}) "
            f"from {local_path}"
        )

    arr = _decode_pcm_frames(raw[:usable], n_channels, bytes_per_sample, is_float)

    # ── Select the channels we actually care about FIRST ──────────────────────
    selected_raw = arr[:, channel_indices]   # shape (n_frames, n_selected_ch)

    # Sanitize + measure invalid rate ONLY over the channels we're using.
    # Other channels in the 14-ch polywav may be legitimately unconnected
    # (BRUEL sentinel FLT_MAX) and would otherwise pollute this check and
    # cause spurious cache invalidation even when our channels are perfectly
    # aligned and clean.
    selected_clean, n_bad = _sanitize_audio(selected_raw, local_path)
    if n_bad:
        n_tot = selected_clean.size
        pct   = 100.0 * n_bad / n_tot
        log.warning(
            "read_polywav_window_local: %d / %d invalid sample(s) (%.1f%%) "
            "in selected channels %s at t=%.2f (NaN/Inf/sentinel) — "
            "replacing with zeros (data_offset=%d, file=%s)",
            n_bad, n_tot, pct, channel_indices, start_s, data_offset, local_path,
        )
        if pct > 5.0:
            log.warning(
                "read_polywav_window_local: invalid rate %.1f%% > 5%% in "
                "selected channels %s — WAV format detection may be wrong.  "
                "Invalidating cache for %s.",
                pct, channel_indices, cache_key,
            )
            _WAV_FORMAT_CACHE.pop(cache_key, None)

    selected = selected_clean.T   # shape (n_ch, n_frames)
    return selected.copy(), POLYWAV_SR


def stream_segment_from_local_polywav(
    cfg,
    segment_id: int,
    local_polywav_dir: str,
    array: str = "BK-6-E",
    ap=None,
) -> Optional[Tuple[List[np.ndarray], dict]]:
    """
    Stream a single maneuver segment from LOCAL polywav files on disk, using
    the same ground-truth metadata and channel logic as the Nextcloud version.

    This is the drop-in offline replacement for stream_segment_from_nextcloud():
    it reads bytes directly from a local directory containing the raw polywav
    files (e.g. downloaded copies of the BRUEL 192 kHz recordings).

    Parameters
    ----------
    cfg               : Config with DUNAKESZI_LOCAL_POLYWAV_PATH or passed directly
    segment_id        : Ground-truth MANEUVER_SEGMENTS id
    local_polywav_dir : Directory containing the polywav .wav files
                        (e.g. /data/Dunakeszi_BRUEL_VIDEO/192KHZ_MULTIWAV_AUDIO_12X_BRUEL4053)
    array             : "BK-6-E" | "BK-6-W"
    ap                : AudioProcessor instance (created if None)

    Returns
    -------
    (channels, label) or None
    """
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            log.error("dunakeszi_ground_truth_fixed not importable")
            return None

    seg = next((s for s in gt.MANEUVER_SEGMENTS if s["id"] == segment_id), None)
    if seg is None:
        log.warning("stream_segment_from_local_polywav: segment_id=%d not found", segment_id)
        return None

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    enriched = gt._enrich_segments([seg], sessions_by_id)[0]

    onset_s    = enriched["onset_from_rec_s"]
    duration_s = float(enriched.get("duration_s") or 3.0)

    if onset_s < 0:
        log.warning(
            "stream_segment_from_local_polywav: segment %d has negative onset "
            "%.2f — skipping", segment_id, onset_s,
        )
        return None
    if onset_s < 1.0:
        log.warning(
            "stream_segment_from_local_polywav: segment %d onset %.2f is within "
            "the first second — skipping to avoid WAV header region",
            segment_id, onset_s,
        )
        return None

    read_dur_s = min(duration_s, 30.0)
    chunk_dur  = gt.PW_CHUNK_DUR_S
    chunk_idx  = int(onset_s // chunk_dur)
    offset_in_chunk = onset_s - chunk_idx * chunk_dur

    chunk_remaining = chunk_dur - offset_in_chunk
    if read_dur_s > chunk_remaining:
        read_dur_s = max(chunk_remaining, 0.5)

    if chunk_idx < 0 or chunk_idx >= len(gt.POLYWAV_FILES):
        log.warning(
            "stream_segment_from_local_polywav: segment %d onset %.1fs maps "
            "to chunk %d — out of range", segment_id, onset_s, chunk_idx,
        )
        return None

    pw_filename = gt.POLYWAV_FILES[chunk_idx]
    local_file  = str(Path(local_polywav_dir) / pw_filename)

    if not Path(local_file).exists():
        log.warning(
            "stream_segment_from_local_polywav: file not found: %s", local_file,
        )
        return None

    ch_map = getattr(cfg, "DUNAKESZI_ARRAY_CHANNELS",
                     {"BK-6-E": BK6E_CHANNELS, "BK-6-W": BK6W_CHANNELS})
    channel_indices = ch_map.get(array, BK6E_CHANNELS)

    try:
        log.info(
            "Local-streaming segment %d from %s  onset=%.1fs  dur=%.1fs  ch=%s",
            segment_id, pw_filename, offset_in_chunk, read_dur_s, channel_indices,
        )
        audio_raw, native_sr = read_polywav_window_local(
            local_file, offset_in_chunk, read_dur_s, channel_indices,
        )
    except Exception as exc:
        log.warning(
            "stream_segment_from_local_polywav: read failed for segment %d: %s",
            segment_id, exc,
        )
        return None

    if ap is None:
        from .audio_processing import AudioProcessor
        ap = AudioProcessor(cfg)

    out_channels = []
    for ch_row in audio_raw:
        if native_sr != cfg.SR:
            import librosa
            ch_row = librosa.resample(ch_row, orig_sr=native_sr, target_sr=cfg.SR)
            ch_row, n_bad = _sanitize_audio(ch_row, f"seg{segment_id}")
            if n_bad:
                log.warning(
                    "stream_segment_from_local_polywav: %d invalid sample(s) "
                    "after resampling segment %d — replaced with zeros",
                    n_bad, segment_id,
                )
        out_channels.append(ap.pad_or_truncate(ch_row))

    while len(out_channels) < 3:
        out_channels.append(out_channels[-1].copy())
    out_channels = out_channels[:3]

    from math import atan2, degrees, hypot
    sc = enriched.get("start_coord")
    if sc and sc[0] is not None:
        bearing     = round(degrees(atan2(sc[0], sc[1])), 1)
        dist        = round(hypot(sc[0], sc[1]), 1)
        ht          = round(float(sc[2]) if len(sc) > 2 else enriched.get("altitude_m", 0), 1)
        pipeline_az = (90.0 - bearing + 180.0) % 360.0 - 180.0
    else:
        pipeline_az = None
        dist        = None
        ht          = enriched.get("altitude_m")

    label = {
        "segment_id":      segment_id,
        "session":         enriched.get("session"),
        "split":           enriched.get("split"),
        "source":          "real",
        "dataset_type":    "dunakeszi",
        "array":           array,
        "maneuver_type":   enriched.get("maneuver_type"),
        "flight_phase":    enriched.get("flight_phase"),
        "description":     enriched.get("description"),
        "n_drones":        enriched.get("n_drones", 1),
        "drones":          enriched.get("drones", []),
        "azimuth_deg":     pipeline_az,
        "distance_m":      dist,
        "height_m":        ht,
        "has_position":    pipeline_az is not None and dist is not None,
        "speed_mps":       enriched.get("speed_mps"),
        "radius_m":        enriched.get("radius_m"),
        "duration_s":      read_dur_s,
        "local_start_hms": enriched.get("local_start_hms"),
        "show_number":     sessions_by_id.get(enriched.get("session", ""), {}).get("show_number"),
        "audio_file":      pw_filename,
        "polywav_chunk":   chunk_idx,
        "polywav_offset_s": offset_in_chunk,
    }
    return out_channels, label


def iter_local_polywav_segments(
    cfg,
    ap,
    local_polywav_dir: str,
    required_split: Optional[str] = None,
    segment_id: Optional[int] = None,
    loop: bool = True,
    array: str = "BK-6-E",
) -> Generator[Tuple[List[np.ndarray], dict], None, None]:
    """
    Yield (channels, label) by reading segments from LOCAL polywav files.

    Exact drop-in replacement for iter_nextcloud_segments() — same ground-truth
    metadata, same channel extraction, same resampling — but reads from disk
    instead of Nextcloud HTTP range requests.

    Parameters
    ----------
    cfg               : Config instance
    ap                : AudioProcessor instance
    local_polywav_dir : Directory containing POLYWAV_FILES (e.g. the
                        192KHZ_MULTIWAV_AUDIO_12X_BRUEL4053 folder)
    required_split    : "train" | "val" | "test" | None
    segment_id        : If set, yield only this segment once (loop forced False)
    loop              : If True, cycle through segments indefinitely
    array             : "BK-6-E" | "BK-6-W"
    """
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        try:
            import dunakeszi_ground_truth_fixed as gt
        except ImportError:
            raise RuntimeError(
                "dunakeszi_ground_truth_fixed not importable — cannot iterate "
                "local polywav segments without the ground-truth metadata module."
            )

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    all_segs       = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)

    if required_split is not None:
        all_segs = [s for s in all_segs if s.get("split") == required_split]
    if segment_id is not None:
        all_segs = [s for s in all_segs if s["id"] == segment_id]
        loop     = False

    if not all_segs:
        raise RuntimeError(
            f"iter_local_polywav_segments: no segments match "
            f"split={required_split!r}, segment_id={segment_id!r}"
        )

    # Filter to segments whose polywav file actually exists on disk
    polywav_dir = Path(local_polywav_dir)
    available_segs = []
    for seg in all_segs:
        chunk_idx  = int(seg["onset_from_rec_s"] // gt.PW_CHUNK_DUR_S)
        if 0 <= chunk_idx < len(gt.POLYWAV_FILES):
            fpath = polywav_dir / gt.POLYWAV_FILES[chunk_idx]
            if fpath.exists():
                available_segs.append(seg)

    if not available_segs:
        raise RuntimeError(
            f"iter_local_polywav_segments: no polywav files found in "
            f"{local_polywav_dir} for the requested segments "
            f"(split={required_split!r}).  "
            f"Expected files like: {gt.POLYWAV_FILES[0] if gt.POLYWAV_FILES else '?'}"
        )

    log.info(
        "iter_local_polywav_segments: %d/%d segment(s) have local polywav files "
        "(loop=%s, split=%r, array=%s, dir=%s)",
        len(available_segs), len(all_segs), loop, required_split, array, local_polywav_dir,
    )

    import random as _random

    first_pass = True
    while True:
        if not first_pass:
            if not loop:
                return
            _random.shuffle(available_segs)
        first_pass = False

        for seg in available_segs:
            result = stream_segment_from_local_polywav(
                cfg, seg["id"], local_polywav_dir, array=array, ap=ap,
            )
            if result is None:
                log.debug(
                    "iter_local_polywav_segments: segment %d returned None — skipping",
                    seg["id"],
                )
                continue
            yield result

        if not loop:
            return


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
    url  = _file_download_url(cfg, remote_path)

    # Probe the actual WAV data-chunk offset (MEMS files are 4-ch 24-bit WAV,
    # which carry an 18-byte fmt chunk + 12-byte fact chunk → data at ~58 bytes,
    # not the 44-byte PCM standard assumed by _WAV_HEADER_BYTES).
    data_offset = _get_wav_data_offset(url, auth)

    frame_start = int(start_s    * sr)
    n_frames    = int(duration_s * sr)
    byte_start  = data_offset + frame_start * bytes_per_frame
    byte_end    = byte_start + n_frames * bytes_per_frame

    log.debug("Range-reading MEMS: %s  t=[%.2f, %.2f)s", remote_path, start_s, start_s + duration_s)
    raw = _fetch_byte_range(url, auth, byte_start, byte_end)

    # 24-bit signed integer: unpack 3 bytes per sample (interleaved channels)
    if bits == 24:
        n_frames_actual = len(raw) // bytes_per_frame
        arr = np.zeros((n_frames_actual, n_channels), dtype=np.int32)
        for f in range(n_frames_actual):
            frame_base = f * bytes_per_frame
            for ch in range(n_channels):
                sample_base = frame_base + ch * 3
                b = raw[sample_base: sample_base + 3]
                if len(b) < 3:
                    break
                arr[f, ch] = int.from_bytes(b, "little", signed=True)
        audio = arr.T.astype(np.float32) / (2 ** 23)  # (n_ch, n_frames), [-1, 1]
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
    url  = _file_download_url(cfg, remote_path)
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

    # Determine which polywav file and byte offset to read.
    #
    # onset_from_rec_s  = seconds elapsed since the polywav recording started
    #                     (the global "rec start" reference, show_1 trigger).
    #                     This IS the correct timeline for chunk selection.
    #
    # within_session_offset_s = onset_from_rec_s minus the session (show) start
    #                           inside the polywav — only useful for cross-checks.
    #
    # The polywav chunks are contiguous from t=0 of the recording, so
    # onset_from_rec_s maps directly onto (chunk_idx, offset_in_chunk).
    # This is correct as long as gt.RECORDING_REF_LOCAL_S and PW_CHUNK_DUR_S
    # are calibrated to the polywav clock (verified in dunakeszi_ground_truth_fixed).
    onset_s    = enriched["onset_from_rec_s"]  # seconds from polywav t=0
    duration_s = float(enriched.get("duration_s") or 3.0)

    # Sanity-check: warn if onset is suspiciously early (likely a calibration issue)
    if onset_s < 0:
        log.warning(
            "Segment %d has negative onset_from_rec_s=%.2f — "
            "ground-truth timing may be miscalibrated; skipping",
            segment_id, onset_s,
        )
        return None

    # onset=0.0 means the range request would start at the WAV header itself
    # (before the first audio frame), producing garbage float32 values.
    # Any real drone event must start at least 1 second into the recording.
    if onset_s < 1.0:
        log.warning(
            "Segment %d onset_from_rec_s=%.2f is within the first second — "
            "this points to a ground-truth calibration issue (the drone was not "
            "yet airborne at t=0). Skipping to avoid reading WAV header as audio.",
            segment_id, onset_s,
        )
        return None

    # Cap to 30 s for streaming
    read_dur_s = min(duration_s, 30.0)

    # Which polywav chunk covers this onset?
    chunk_dur = gt.PW_CHUNK_DUR_S   # ~399.46 s
    chunk_idx = int(onset_s // chunk_dur)
    offset_in_chunk = onset_s - chunk_idx * chunk_dur

    # Guard: offset_in_chunk + duration must not spill past the chunk boundary.
    # If it does, cap the read to the chunk end (the pipeline pads to TARGET_DURATION).
    chunk_remaining = chunk_dur - offset_in_chunk
    if read_dur_s > chunk_remaining:
        log.debug(
            "Segment %d read window (%.1f s) would cross chunk boundary at %.1f s — "
            "capping to %.1f s",
            segment_id, read_dur_s, chunk_remaining, chunk_remaining,
        )
        read_dur_s = max(chunk_remaining, 0.5)  # keep at least 0.5 s

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
            # Clamp sentinels and NaN/Inf that can survive or be introduced by resampling
            ch_row, n_bad = _sanitize_audio(ch_row, f"seg{segment_id}")
            if n_bad:
                log.warning(
                    "stream_segment_from_nextcloud: %d invalid sample(s) "
                    "after resampling segment %d ch — replaced with zeros",
                    n_bad, segment_id,
                )
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

def stream_segment_from_mems(
    cfg,
    segment_id: int,
    ap=None,
    _enriched: Optional[dict] = None,   # skip re-derivation when caller already has it
) -> Optional[Tuple[np.ndarray, int, dict]]:
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        import dunakeszi_ground_truth_fixed as gt

    if _enriched is not None:
        enriched = _enriched
    else:
        # Slow path (only used when called standalone, e.g. from a future
        # single-segment endpoint). Recompute from the FULL segment list —
        # enriching a lone segment in isolation drops mems_available, so we
        # must always enrich the whole batch and pick this one out.
        sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
        all_enriched = gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)
        enriched = next((s for s in all_enriched if s["id"] == segment_id), None)
        if enriched is None:
            log.warning("stream_segment_from_mems: segment_id=%d not found", segment_id)
            return None
        sessions_by_id_for_label = sessions_by_id  # used below for show_number lookup

    if not enriched.get("mems_available", False):
        log.debug("stream_segment_from_mems: segment %d has no MEMS coverage", segment_id)
        return None

    onset_s    = enriched["onset_from_rec_s"]
    duration_s = float(enriched.get("duration_s") or 3.0)
    if onset_s < 1.0:
        log.warning("stream_segment_from_mems: segment %d onset too early — skipping", segment_id)
        return None

    mems_start_rec_s = gt.MEMS_START_LOCAL_S - gt.RECORDING_REF_LOCAL_S
    file_dur_s        = gt.MEMS_ASSUMED_FORMAT["duration_s"]
    rel_s             = onset_s - mems_start_rec_s
    if rel_s < 0:
        log.warning("stream_segment_from_mems: segment %d predates MEMS recording start", segment_id)
        return None

    file_idx  = int(rel_s // file_dur_s)
    offset_in_file = rel_s - file_idx * file_dur_s
    if file_idx < 0 or file_idx >= len(gt.MEMS_FILES):
        log.warning("stream_segment_from_mems: segment %d maps to out-of-range MEMS file %d",
                    segment_id, file_idx)
        return None

    read_dur_s = min(duration_s, 30.0, file_dur_s - offset_in_file)
    if read_dur_s < 0.5:
        read_dur_s = 0.5

    mems_filename = gt.MEMS_FILES[file_idx]
    remote_path   = f"{getattr(cfg, 'NEXTCLOUD_MEMS_PATH', '/mems')}/{mems_filename}"

    try:
        audio_mc, native_sr = read_mems_window(cfg, remote_path, offset_in_file, read_dur_s)
    except (RemoteUnavailableError, RangeRequestError) as exc:
        log.warning("stream_segment_from_mems: read failed for segment %d: %s", segment_id, exc)
        return None

    mono = audio_mc.mean(axis=0).astype(np.float32)
    mono, n_bad = _sanitize_audio(mono, f"mems_seg{segment_id}")

    label = {
        "segment_id":    segment_id,
        "session":       enriched.get("session"),
        "split":         enriched.get("split"),
        "source":        "real",
        "dataset_type":  "mems",
        "array":         "MEMS",
        "maneuver_type": enriched.get("maneuver_type"),
        "flight_phase":  enriched.get("flight_phase"),
        "description":   enriched.get("description"),
        "n_drones":      enriched.get("n_drones", 1),
        "duration_s":    read_dur_s,
        "local_start_hms": enriched.get("local_start_hms"),
        "audio_file":    mems_filename,
        "mems_file_index": file_idx,
        "mems_offset_s": offset_in_file,
        "localization_method": "spectral_proxy",
        "has_position":  False,
    }
    return mono, native_sr, label


def iter_mems_segments(
    cfg,
    required_split: Optional[str] = None,
    segment_id: Optional[int] = None,
    loop: bool = True,
) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
    _auth(cfg)
    try:
        from . import dunakeszi_ground_truth_fixed as gt
    except ImportError:
        import dunakeszi_ground_truth_fixed as gt

    sessions_by_id = {s["session_id"]: s for s in gt._enrich_sessions(gt.SESSIONS)}
    all_segs = [s for s in gt._enrich_segments(gt.MANEUVER_SEGMENTS, sessions_by_id)
                if s.get("mems_available", False)]

    if required_split is not None:
        all_segs = [s for s in all_segs if s.get("split") == required_split]
    if segment_id is not None:
        all_segs = [s for s in all_segs if s["id"] == segment_id]
        loop = False

    if not all_segs:
        raise RuntimeError(
            f"iter_mems_segments: no MEMS-covered segments match split={required_split!r}, "
            f"segment_id={segment_id!r}"
        )

    import random as _random
    first_pass = True
    while True:
        if not first_pass:
            if not loop:
                return
            _random.shuffle(all_segs)
        first_pass = False

        n_yielded_this_pass = 0
        for seg in all_segs:
            result = stream_segment_from_mems(cfg, seg["id"], _enriched=seg)   # <-- pass it through
            if result is None:
                continue
            n_yielded_this_pass += 1
            yield result

        # Safety: if an entire pass over known-MEMS-covered segments produced
        # zero playable results, something is structurally wrong (bad timing
        # calc, missing remote files, etc.) — stop instead of spinning at
        # 100% CPU forever with no visible error.
        if n_yielded_this_pass == 0:
            raise RuntimeError(
                f"iter_mems_segments: {len(all_segs)} candidate segment(s) were "
                f"marked mems_available=True but none could be loaded — check "
                f"Nextcloud MEMS path/credentials or ground-truth timing fields."
            )

        if not loop:
            return

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
        # Build the unique set of sessions (shows) that appear in this chunk.
        # Use a dict keyed by seg id for O(1) lookup to avoid the default-index bug.
        segs_by_id = {s["id"]: s for s in all_segs}
        shows = sorted({
            segs_by_id[sid]["session"]
            for sid in covered
            if sid in segs_by_id
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