"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DUNAKESZI DRONE DATASET — DEEP FORENSIC ANALYSIS  v2                     ║
║   Stage 1 of N: Understand everything before touching a model               ║
║                                                                              ║
║   KEY CHANGES vs v1                                                          ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   • BRUEL files now downloaded selectively (AT01I + AT01J only)            ║
║     using HTTP Range — pulls only the drone flight window (~7 min)         ║
║   • Flight window baked in from drone log: 12:57:21–13:04:09 UTC           ║
║   • GPX telemetry parsed with xml.etree (not fragile regex)                ║
║   • 192 kHz aware: N_FFT, SR_TARGET, BPF bands all rescaled                ║
║   • Brüel multi-WAV channel splitting (12-ch interleaved PCM)              ║
║   • Skip-if-exists on every download (safe to re-run)                      ║
║   • Drone-specific plots: per-channel PSD, BPF harmonic ladder,            ║
║     flight-window spectrogram, SNR timeline                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

QUICK START
───────────
  Step 1 — set SHARE_URL to the Nextcloud root share link
  Step 2 — python3 dunakeszi_audit_v2.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import gc
import os, re, json, wave, struct, io, sys, time, hashlib, math
import warnings, logging, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from urllib.parse import unquote, urlparse, urlencode, quote
import base64

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── third-party ───────────────────────────────────────────────────────────────
import requests
import numpy as np
import pandas as pd
from scipy import signal, fft, stats
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =============================================================================
#  ▶▶  USER SETTINGS  ◀◀
# =============================================================================

SHARE_URL       = "https://nc.ek-cer.hu/index.php/s/6xBE46A9HQCCQ8F?dir=/Dunakeszi_2025_10_25"   # ← paste token here
SCRIPT_DIR      = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = SCRIPT_DIR / "dunakeszi_data"

# ── Download limits ───────────────────────────────────────────────────────────
MAX_MEMS_FILES  = 60
MAX_MEMS_MB     = 600
TRIM_SECONDS    = 30           # seconds to pull per MEMS file via Range header

# ── Brüel selective download ──────────────────────────────────────────────────
# These two files bracket the entire drone flight (12:57:21 – 13:04:09 UTC).
# AT01I ends at ~12:59:22, AT01J covers 12:59:28 – 13:08:16.
BRUEL_FILES_TO_DOWNLOAD = {
    "251020VITEMOROM1AT01I.wav",   # tail of pre-flight + start of flight
    "251020VITEMOROM1AT01J.wav",   # main flight window
}
# We only need seconds 0 … BRUEL_RANGE_END of each file.
# AT01I: flight starts at offset ~(12:57:21 - 12:50:34) = 407 s into the file
# AT01J: flight ends   at offset ~(13:04:09 - 12:59:28) = 281 s into the file
# Pull 600 s from AT01I (enough margin) and 360 s from AT01J.
BRUEL_BYTE_CAPS = {
    "251020VITEMOROM1AT01I.wav": 600,   # seconds
    "251020VITEMOROM1AT01J.wav": 360,   # seconds
}
# Brüel WAV specs: 192 kHz, 16-bit, 12 channels interleaved
BRUEL_SR        = 192_000
BRUEL_CHANNELS  = 14
BRUEL_BIT_DEPTH = 24           # Brüel 4053 records 24-bit; adjust if needed

# ── Drone flight window (from messages.log) ───────────────────────────────────
FLIGHT_DATE          = "2025-10-20"
FLIGHT_TAKEOFF_UTC   = datetime(2025, 10, 20, 12, 57, 21)
FLIGHT_LANDING_UTC   = datetime(2025, 10, 20, 13,  4,  9)
FLIGHT_DURATION_SEC  = (FLIGHT_LANDING_UTC - FLIGHT_TAKEOFF_UTC).total_seconds()

# AT01J file starts at 12:59:28 UTC (from last_modified minus ~8.8 min segment)
# AT01I file starts at 12:50:34 UTC
BRUEL_FILE_STARTS = {
    "251020VITEMOROM1AT01I.wav": datetime(2025, 10, 20, 12, 50, 34),
    "251020VITEMOROM1AT01J.wav": datetime(2025, 10, 20, 12, 59, 28),
}

# Folders to INDEX only (no download)
DOWNLOAD_META_ONLY_FOLDERS = {"VIDEO", "FOTO"}   # BRUEL now handled separately

# ── Output ────────────────────────────────────────────────────────────────────
WORKSPACE  = SCRIPT_DIR / "dunakeszi_audit_ws"
OUTPUT_DIR = SCRIPT_DIR / "dunakeszi_audit_output"
PLOT_DIR   = OUTPUT_DIR / "plots"
for d in [WORKSPACE, OUTPUT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Signal processing — 192 kHz aware ────────────────────────────────────────
SR_TARGET   = BRUEL_SR          # 192 000 Hz
N_FFT       = 65536             # ~341 ms window at 192 kHz → 2.93 Hz resolution
HOP         = 16384
N_MELS      = 256               # more mel bands to cover 0–96 kHz
SEED        = 42
np.random.seed(SEED)

# ── Drone acoustic bands (Brüel @ 192 kHz) ───────────────────────────────────
# DJI-class drone blade-pass fundamental: typically 70–100 Hz
# Harmonics extend well into the ultrasonic range at 192 kHz
DRONE_BPF_HZ     = 82          # expected blade-pass fundamental (Hz)
DRONE_HARMONICS  = 20          # number of harmonics to mark on plots
MOTOR_HZ         = 440         # approximate motor electrical frequency (Hz)

# ── Visuals ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
C = ["#00FFCC","#FF4C6A","#FFD700","#7B68EE","#FF8C00",
     "#00BFFF","#ADFF2F","#FF69B4","#40E0D0","#FF6347",
     "#DA70D6","#98FB98","#F0E68C","#87CEEB","#DEB887"]
sns.set_palette(C)

SEP  = "─" * 72
SEP2 = "═" * 72

print(SEP2)
print("  DUNAKESZI DRONE DATASET  —  FORENSIC ANALYSIS  v2")
print(f"  Flight window: {FLIGHT_TAKEOFF_UTC} → {FLIGHT_LANDING_UTC}")
print(f"  Duration     : {FLIGHT_DURATION_SEC:.0f} s  ({FLIGHT_DURATION_SEC/60:.1f} min)")
print(SEP2)


# =============================================================================
#  SECTION 0  ──  NEXTCLOUD WebDAV DOWNLOADER  (skip-if-exists)
# =============================================================================

_DAV_NS = "DAV:"

def _nc_parse_share_url(url: str):
    p = urlparse(url)
    host_base = f"{p.scheme}://{p.netloc}"
    m = re.search(r"/s/([A-Za-z0-9]+)", p.path)
    if not m:
        raise ValueError(f"Cannot extract share token from URL: {url}")
    token = m.group(1)
    subpath = ""
    if "dir=" in (p.query or ""):
        for part in p.query.split("&"):
            if part.startswith("dir="):
                subpath = unquote(part[4:]).lstrip("/")
    return host_base, token, subpath

def _nc_webdav_base(host_base, token):
    return f"{host_base}/public.php/webdav"

def _nc_auth_header(token):
    creds = base64.b64encode(f"{token}:".encode()).decode()
    return {"Authorization": f"Basic {creds}"}

def _nc_propfind(session, webdav_base, token, remote_path="", depth=1):
    url = webdav_base.rstrip("/")
    if remote_path:
        url += "/" + remote_path.strip("/")
    headers = {
        **_nc_auth_header(token),
        "Depth": str(depth),
        "Content-Type": "application/xml; charset=utf-8",
    }
    body = (b'<?xml version="1.0"?>'
            b'<d:propfind xmlns:d="DAV:">'
            b'  <d:prop><d:displayname/><d:getcontentlength/>'
            b'    <d:getcontenttype/><d:resourcetype/>'
            b'    <d:getlastmodified/></d:prop>'
            b'</d:propfind>')
    r = session.request("PROPFIND", url, headers=headers, data=body, timeout=30)
    if r.status_code not in (207, 200):
        r.raise_for_status()
    root_xml   = ET.fromstring(r.text)
    items      = []
    base_href  = None
    for resp in root_xml.findall(f"{{{_DAV_NS}}}response"):
        href         = unquote(resp.findtext(f"{{{_DAV_NS}}}href", ""))
        if base_href is None:
            base_href = href; continue
        prop_ok = None
        for ps in resp.findall(f"{{{_DAV_NS}}}propstat"):
            if "200" in ps.findtext(f"{{{_DAV_NS}}}status", ""):
                prop_ok = ps.find(f"{{{_DAV_NS}}}prop"); break
        if prop_ok is None: continue
        rt     = prop_ok.find(f"{{{_DAV_NS}}}resourcetype")
        is_dir = rt is not None and rt.find(f"{{{_DAV_NS}}}collection") is not None
        name   = prop_ok.findtext(f"{{{_DAV_NS}}}displayname", "") or href.rstrip("/").split("/")[-1]
        try:    size = int(prop_ok.findtext(f"{{{_DAV_NS}}}getcontentlength", "0"))
        except: size = 0
        items.append({
            "href": href, "name": name,
            "size": size, "size_mb": size / 1024**2,
            "content_type": prop_ok.findtext(f"{{{_DAV_NS}}}getcontenttype", ""),
            "is_dir": is_dir,
            "last_modified": prop_ok.findtext(f"{{{_DAV_NS}}}getlastmodified", ""),
            "remote_path": href.split("/webdav/", 1)[-1] if "/webdav/" in href else name,
        })
    return items

def _nc_collect_files(session, webdav_base, token, remote_path="", depth=4):
    items = _nc_propfind(session, webdav_base, token, remote_path, depth=1)
    files = []
    for it in items:
        if not it["is_dir"]:
            files.append(it)
        elif depth > 0:
            files.extend(_nc_collect_files(session, webdav_base, token,
                                            it["remote_path"], depth - 1))
    return files

def _nc_download(session, webdav_base, token, remote_path, dest, byte_cap=None):
    """Download with skip-if-exists. byte_cap = max bytes (Range header)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"      ↩  SKIP (exists)  {dest.name}")
        return True
    url     = webdav_base.rstrip("/") + "/" + remote_path.lstrip("/")
    headers = _nc_auth_header(token)
    if byte_cap:
        headers["Range"] = f"bytes=0-{byte_cap - 1}"
    try:
        r = session.get(url, headers=headers, stream=True, timeout=300)
        if r.status_code not in (200, 206):
            r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):   # 1 MB chunks
                if chunk: f.write(chunk)
        actual_mb = dest.stat().st_size / 1024**2
        print(f"      ✔  {dest.name:<55}  {actual_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"      ⚠  {dest.name}: {e}")
        if dest.exists(): dest.unlink()
        return False

def _bytes_for_seconds(seconds, sr=BRUEL_SR, channels=BRUEL_CHANNELS,
                        bit_depth=BRUEL_BIT_DEPTH):
    """Calculate WAV body bytes for a given duration, plus 44-byte header."""
    return 44 + int(seconds * sr * channels * (bit_depth // 8))

def _folder_key(name):
    u = name.upper()
    for k in ["MEMS","DRON","JEGY","VIDEO","FOTO","BRUEL"]:
        if k in u: return k
    return u

def download_dataset(workspace: Path) -> dict:
    folder_map = {}

    if LOCAL_DATA_ROOT.exists():
        audio_files = [f for f in LOCAL_DATA_ROOT.rglob("*")
                       if f.suffix.lower() == ".wav"]
        if audio_files:
            print(f"  ✔  Local data found at {LOCAL_DATA_ROOT}")
            print(f"     {len(audio_files)} WAV files detected — skipping download")
            for p in LOCAL_DATA_ROOT.iterdir():
                if p.is_dir(): folder_map[p.name] = p
            folder_map["_root"] = LOCAL_DATA_ROOT
            return folder_map

    if not SHARE_URL or "..." in SHARE_URL:
        print("\n  ⚠  SHARE_URL not configured — set it at the top of the script.\n")
        folder_map["_root"] = workspace
        return folder_map

    host_base, token, subpath = _nc_parse_share_url(SHARE_URL)
    webdav_base = _nc_webdav_base(host_base, token)
    session     = requests.Session()
    session.headers["User-Agent"] = "DunakesziAudit/2.0"

    print(f"\n  🔗  Nextcloud: {host_base}  token={token}")
    root_items = _nc_propfind(session, webdav_base, token,
                              remote_path=subpath, depth=1)
    print(f"  Found {len(root_items)} top-level items\n")

    for item in root_items:
        fname = item["name"]
        fkey  = _folder_key(fname)
        local = workspace / fname
        local.mkdir(exist_ok=True)
        folder_map[fname] = local

        if not item["is_dir"]:
            _nc_download(session, webdav_base, token,
                         item["remote_path"], local / fname)
            continue

        meta_only = any(k in fkey for k in DOWNLOAD_META_ONLY_FOLDERS)
        is_bruel  = "BRUEL" in fkey
        is_mems   = "MEMS" in fkey

        print(f"\n  {'📋 INDEX' if meta_only else '⬇  DOWNLOAD'}  {fname} …")
        try:
            all_files = _nc_collect_files(session, webdav_base, token,
                                          item["remote_path"], depth=4)
        except Exception as e:
            print(f"      ⚠  Could not list {fname}: {e}"); continue

        print(f"      {len(all_files)} files found")

        if meta_only:
            idx = [{"name": f["name"], "size_mb": round(f["size_mb"], 3),
                    "last_modified": f["last_modified"],
                    "remote_path": f["remote_path"]} for f in all_files]
            with open(local / "_index.json", "w") as fh:
                json.dump(idx, fh, indent=2)
            print(f"      Indexed {len(idx)} files → _index.json")
            continue

        dl_count, dl_mb = 0, 0.0
        audio_exts = {".wav", ".flac", ".w64"}

        for f in all_files:
            ext     = Path(f["name"]).suffix.lower()
            fname_f = f["name"]

            # ── Brüel: selective download of flight-window files only ─────────
            if is_bruel:
                if fname_f not in BRUEL_FILES_TO_DOWNLOAD:
                    continue   # skip all other Brüel files
                secs     = BRUEL_BYTE_CAPS.get(fname_f, 400)
                byte_cap = _bytes_for_seconds(secs)
                print(f"      🎯  Brüel target: {fname_f}  "
                      f"({secs}s = {byte_cap/1024**2:.0f} MB)")
                rel   = Path(f["remote_path"]).parent.name
                dest  = (local / rel / fname_f) if rel and rel != fname else (local / fname_f)
                ok    = _nc_download(session, webdav_base, token,
                                     f["remote_path"], dest, byte_cap=byte_cap)
                if ok:
                    dl_count += 1; dl_mb += dest.stat().st_size / 1024**2
                continue

            # ── MEMS: standard capped download ────────────────────────────────
            if is_mems:
                if ext not in audio_exts: continue
                if dl_count >= MAX_MEMS_FILES:
                    print(f"      ⚡ MEMS file cap ({MAX_MEMS_FILES})"); break
                if dl_mb >= MAX_MEMS_MB:
                    print(f"      ⚡ MEMS size cap ({MAX_MEMS_MB} MB)"); break
                byte_cap = int(TRIM_SECONDS * 192 * 1024)
            else:
                byte_cap = None   # DRON/JEGY — full download

            rel      = Path(f["remote_path"]).parent.name
            dest_dir = (local / rel) if rel and rel != fname else local
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest     = dest_dir / fname_f
            ok       = _nc_download(session, webdav_base, token,
                                    f["remote_path"], dest, byte_cap=byte_cap)
            if ok:
                dl_count += 1; dl_mb += dest.stat().st_size / 1024**2

        print(f"\n      ✔  {dl_count} files  /  {dl_mb:.1f} MB")

    return folder_map


# =============================================================================
#  SECTION 1  ──  FILE-SYSTEM AUDIT
# =============================================================================

AUDIO_EXT = {".wav",".flac",".mp3",".ogg",".w64",".pcm",".raw",".aif",".aiff"}
TEXT_EXT  = {".txt",".log",".md",".csv",".json",".xml",".kml",".gpx",".nmea"}
VIDEO_EXT = {".mp4",".avi",".mov",".mkv",".mts",".m2ts",".wmv"}
IMAGE_EXT = {".jpg",".jpeg",".png",".tiff",".tif",".bmp"}
DATA_EXT  = {".csv",".json",".xml",".xlsx",".xls",".mat",".h5",".hdf5",".npy"}

def _classify_ext(ext):
    if ext in AUDIO_EXT: return "audio"
    if ext in VIDEO_EXT: return "video"
    if ext in IMAGE_EXT: return "image"
    if ext in DATA_EXT:  return "data"
    if ext in TEXT_EXT:  return "text"
    return "other"

def _md5_prefix(path, n_bytes=65536):
    h = hashlib.md5()
    try:
        with open(path,"rb") as f: h.update(f.read(n_bytes))
        return h.hexdigest()[:12]
    except: return ""

def audit_filesystem(root: Path) -> pd.DataFrame:
    print(f"\n  🔍  Scanning {root} …")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        stat  = p.stat()
        ext   = p.suffix.lower()
        rel   = p.relative_to(root)
        parts = rel.parts
        rows.append({
            "path":       str(p), "rel_path": str(rel),
            "filename":   p.name, "stem": p.stem, "extension": ext,
            "folder_l1":  parts[0] if len(parts)>1 else "_root",
            "folder_l2":  parts[1] if len(parts)>2 else "",
            "depth":      len(parts)-1,
            "size_bytes": stat.st_size,
            "size_kb":    stat.st_size/1024,
            "size_mb":    stat.st_size/1024**2,
            "mtime":      datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type":  _classify_ext(ext),
            "is_hidden":  p.name.startswith("."),
            "md5_prefix": _md5_prefix(p),
        })
    df = pd.DataFrame(rows)
    print(f"     ✔  {len(df)} files / {df['folder_l1'].nunique()} folders")
    return df


# =============================================================================
#  SECTION 2  ──  AUDIO FILE FORENSICS
# =============================================================================

def read_wav_info(path: Path) -> dict:
    info = {"path": str(path), "readable": False}
    try:
        with wave.open(str(path), "r") as wf:
            info.update({
                "readable":     True,
                "n_channels":   wf.getnchannels(),
                "sample_rate":  wf.getframerate(),
                "bit_depth":    wf.getsampwidth() * 8,
                "n_frames":     wf.getnframes(),
                "duration_sec": wf.getnframes() / max(wf.getframerate(), 1),
                "format": "WAV/PCM", "comptype": wf.getcomptype(),
            })
    except Exception:
        try:
            sr, data = wavfile.read(str(path))
            info.update({
                "readable":     True,
                "n_channels":   1 if data.ndim==1 else data.shape[1],
                "sample_rate":  sr,
                "bit_depth":    data.dtype.itemsize * 8,
                "n_frames":     data.shape[0],
                "duration_sec": data.shape[0] / max(sr, 1),
                "format": f"WAV/{data.dtype}", "comptype": "NONE",
            })
        except Exception as e:
            info["error"] = str(e)
    return info

def read_wav_signal(path: Path, max_seconds=30.0, channel=0):
    """
    Load one channel from a WAV file.
    Returns (float32 array, sample_rate) or (None, None).
    """
    try:
        sr, data = wavfile.read(str(path))
        orig_dtype = data.dtype

        if data.ndim > 1:
            ch = min(channel, data.shape[1] - 1)
            data = data[:, ch]

        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            maxval = max(abs(info.min), info.max)
            data = data.astype(np.float32) / maxval
        else:
            data = data.astype(np.float32)

        max_frames = int(max_seconds * sr)
        if len(data) > max_frames:
            data = data[:max_frames]

        return data, int(sr)
    except Exception:
        return None, None

def _parse_wav_header(f):
    """
    Robust WAV/RF64 chunk scanner.
    RF64 layout: RF64 | size | WAVE | ds64 chunk | fmt chunk | [other] | data chunk
    The ds64 chunk carries the real 64-bit data size for RF64 files.
    """
    f.seek(0)
    magic = f.read(4)
    is_rf64 = (magic == b'RF64')
    if magic not in (b'RIFF', b'RF64'):
        raise ValueError(f"Not a WAV file (magic: {magic})")

    f.read(4)   # RIFF/RF64 size (unreliable for RF64)
    wave = f.read(4)
    if wave != b'WAVE':
        raise ValueError(f"Not a WAVE file (got: {wave})")

    n_ch = sr = bit_depth = 0
    data_offset = data_size = 0
    ds64_data_size = 0  # real 64-bit size from ds64 chunk

    while True:
        chunk_id = f.read(4)
        if len(chunk_id) < 4:
            break
        chunk_size = int.from_bytes(f.read(4), "little")
        chunk_pos  = f.tell()   # position of chunk body

        if chunk_id == b'ds64':
            # RF64 size extension: riff_size(8) + data_size(8) + sample_count(8) + ...
            f.read(8)  # riff size (64-bit), skip
            ds64_data_size = int.from_bytes(f.read(8), "little")
            # don't need sample_count, stop reading this chunk

        elif chunk_id == b'fmt ':
            f.read(2)   # audio format (1=PCM, 3=float, 0xFFFE=extensible)
            n_ch      = int.from_bytes(f.read(2), "little")
            sr        = int.from_bytes(f.read(4), "little")
            f.read(4)   # byte rate
            f.read(2)   # block align
            bit_depth = int.from_bytes(f.read(2), "little")

        elif chunk_id == b'data':
            data_offset = chunk_pos
            data_size   = chunk_size
            break  # data chunk found — stop scanning

        # Advance to next chunk; WAV chunks are word (2-byte) aligned
        next_pos = chunk_pos + chunk_size + (chunk_size % 2)
        f.seek(next_pos)

    # For RF64, data chunk size field = 0xFFFFFFFF — use ds64 value instead
    if data_size == 0xFFFFFFFF and ds64_data_size > 0:
        data_size = ds64_data_size
    # Final fallback: estimate from actual file size on disk
    if data_size == 0 or data_size == 0xFFFFFFFF:
        cur = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur)
        data_size = file_size - data_offset

    return n_ch, sr, bit_depth, data_offset, data_size

def _bruel_cache_path(path: Path) -> Path:
    return OUTPUT_DIR / f"{path.stem}_flight_window.npy"

def _load_bruel_segments(bruel_results: list, channels=None):
    """
    Load cached Brüel flight-window arrays as memmaps and concatenate.
    channels:
      - None => all channels
      - int  => one channel only
      - list/tuple => selected channels
    """
    arrays = []
    sr = None
    n_ch = None

    ok_results = [r for r in bruel_results if r.get("ok") and r.get("segment_cache")]
    if not ok_results:
        return None, None, None

    for r in ok_results:
        arr = np.load(r["segment_cache"], mmap_mode="r")
        if channels is None:
            sub = arr
        elif isinstance(channels, int):
            sub = arr[:, channels]
        else:
            sub = arr[:, channels]

        arrays.append(np.asarray(sub, dtype=np.float32))
        if sr is None:
            sr = r["sr"]
            n_ch = arr.shape[1] if arr.ndim == 2 else 1

    if len(arrays) == 1:
        combined = arrays[0]
    else:
        combined = np.concatenate(arrays, axis=0)

    return combined, sr, n_ch

def _downsample_1d(sig: np.ndarray, sr: int, target_sr: int):
    if sig is None or len(sig) == 0:
        return sig, sr
    decim = max(1, sr // target_sr)
    if decim <= 1:
        return sig.astype(np.float32, copy=False), sr
    sig_ds = signal.decimate(sig, decim, ftype="fir", zero_phase=True).astype(np.float32)
    return sig_ds, sr // decim

def _downsample_2d(arr: np.ndarray, sr: int, target_sr: int):
    if arr is None or len(arr) == 0:
        return arr, sr
    decim = max(1, sr // target_sr)
    if decim <= 1:
        return arr.astype(np.float32, copy=False), sr
    arr_ds = signal.decimate(arr, decim, axis=0, ftype="fir", zero_phase=True).astype(np.float32)
    return arr_ds, sr // decim

def read_bruel_flight_window(path: Path, file_start: datetime) -> dict:
    result = {"path": str(path), "ok": False}
    try:
        with open(path, "rb") as f:
            n_ch, sr, bit_depth, data_offset, data_size = _parse_wav_header(f)

            if n_ch == 0 or sr == 0 or bit_depth == 0:
                raise ValueError(f"Invalid WAV header: n_ch={n_ch} sr={sr} bit={bit_depth}")

            bytes_per_sample = bit_depth // 8
            bytes_per_frame = bytes_per_sample * n_ch

            if data_size == 0 or data_size == 0xFFFFFFFF:
                file_size = path.stat().st_size
                data_size = file_size - data_offset

            n_frames_total = data_size // bytes_per_frame

            print(f"     📋  {path.name}: {sr}Hz {bit_depth}bit {n_ch}ch "
                  f"{n_frames_total/sr:.1f}s  data@{data_offset}")

            result.update({
                "sr": sr,
                "n_ch": n_ch,
                "bit_depth": bit_depth,
                "shape": (n_frames_total, n_ch),
            })

            file_end = file_start + timedelta(seconds=n_frames_total / sr)
            overlap_start = max(file_start, FLIGHT_TAKEOFF_UTC)
            overlap_end   = min(file_end, FLIGHT_LANDING_UTC)

            if overlap_start >= overlap_end:
                result["note"] = "No overlap with flight window"
                print(f"     ⚠  {path.name}: no overlap  "
                      f"(file {file_start}→{file_end})")
                return result

            t0_offset = (overlap_start - file_start).total_seconds()
            t1_offset = (overlap_end   - file_start).total_seconds()
            s0 = int(t0_offset * sr)
            s1 = int(t1_offset * sr)

            out_frames = s1 - s0
            byte_start = data_offset + s0 * bytes_per_frame

            cache_path = _bruel_cache_path(path)

            # Create disk-backed output array
            segment_mm = np.lib.format.open_memmap(
                cache_path, mode="w+", dtype=np.float32, shape=(out_frames, n_ch)
            )

            f.seek(byte_start)

            # Process about ~1 second at a time to keep RAM low
            # chunk_frames = 192000
            chunk_frames = 48000
            out_pos = 0

            while out_pos < out_frames:
                this_frames = min(chunk_frames, out_frames - out_pos)
                this_bytes = this_frames * bytes_per_frame
                raw = f.read(this_bytes)

                if len(raw) != this_bytes:
                    raise ValueError(
                        f"Short read in {path.name}: expected {this_bytes} bytes, got {len(raw)}"
                    )

                if bit_depth == 24:
                    raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)

                    samples = (
                        raw_u8[:, 0].astype(np.int32)
                        | (raw_u8[:, 1].astype(np.int32) << 8)
                        | (raw_u8[:, 2].astype(np.int32) << 16)
                    )
                    samples[samples >= 2**23] -= 2**24
                else:
                    dtype = {16: np.int16, 32: np.int32}.get(bit_depth, np.int16)
                    samples = np.frombuffer(raw, dtype=dtype)

                samples_2d = samples.reshape(this_frames, n_ch)
                segment_mm[out_pos:out_pos + this_frames] = (
                    samples_2d.astype(np.float32) / (2 ** (bit_depth - 1))
                )

                out_pos += this_frames

                del raw, samples, samples_2d
                if bit_depth == 24:
                    del raw_u8
                gc.collect()

            segment_mm.flush()
            del segment_mm
            gc.collect()

            result.update({
                "flight_offset_sec": t0_offset,
                "flight_duration_sec": t1_offset - t0_offset,
                "overlap_start_utc": overlap_start.isoformat(),
                "overlap_end_utc": overlap_end.isoformat(),
                "segment_cache": str(cache_path),
                "ok": True,
            })

            print(f"     ✔  {path.name}: {t0_offset:.1f}–{t1_offset:.1f}s  "
                  f"({t1_offset-t0_offset:.1f}s, {n_ch}ch, {sr}Hz)")

    except Exception as e:
        result["error"] = str(e)
        print(f"     ⚠  {path.name}: {e}")

    return result

def diagnose_audio(data, sr, path=""):
    d = {"path": path}
    if data is None or len(data) == 0:
        d["error"] = "unreadable"; return d
    d["rms"]            = float(np.sqrt(np.mean(data**2)))
    d["peak"]           = float(np.max(np.abs(data)))
    d["dc_offset"]      = float(np.mean(data))
    d["dynamic_range"]  = float(20 * np.log10(d["peak"] + 1e-12))
    clip_thresh = 0.99
    d["clip_pct"]       = float(np.mean(np.abs(data) > clip_thresh) * 100)
    d["clipped"]        = d["clip_pct"] > 0.1
    frame = int(0.1 * sr)
    n_fr  = len(data) // frame
    if n_fr > 0:
        frames   = data[:n_fr*frame].reshape(n_fr, frame)
        fr_rms   = np.sqrt(np.mean(frames**2, axis=1))
        fr_db    = 20 * np.log10(fr_rms + 1e-12)
        d["silence_pct"] = float(np.mean(fr_db < -50) * 100)
        d["active_pct"]  = 100 - d["silence_pct"]
        d["rms_db_mean"] = float(fr_db.mean())
        d["rms_db_std"]  = float(fr_db.std())
        noise_rms  = float(np.percentile(fr_rms, 10)) + 1e-12
        signal_rms = float(np.percentile(fr_rms, 90)) + 1e-12
        d["snr_db"]       = float(20 * np.log10(signal_rms / noise_rms))
        d["noise_floor_db"]= float(20 * np.log10(noise_rms))
    else:
        d["silence_pct"] = d["active_pct"] = d["snr_db"] = d["noise_floor_db"] = 0.0
    f_psd, psd = signal.welch(data, fs=sr, nperseg=min(N_FFT, len(data)//4 or 4096))
    d["spectral_centroid_hz"] = float(np.sum(f_psd*psd) / (np.sum(psd)+1e-10))
    d["spectral_flatness"]    = float(
        np.exp(np.mean(np.log(psd+1e-10))) / (np.mean(psd)+1e-10))
    peaks, props = signal.find_peaks(psd, height=psd.max()*0.05, distance=3)
    if len(peaks):
        top = peaks[np.argmax(props["peak_heights"])]
        d["dominant_freq_hz"] = float(f_psd[top])
    else:
        d["dominant_freq_hz"] = 0.0
    d["zcr"]          = float(np.mean(np.abs(np.diff(np.sign(data)))) / 2)
    d["kurtosis"]     = float(stats.kurtosis(data))
    d["crest_factor"] = d["peak"] / (d["rms"] + 1e-9)
    idx = (f_psd > 50) & (f_psd < sr/2 * 0.9)
    if idx.sum() > 5:
        slope, *_ = np.polyfit(np.log10(f_psd[idx]+1e-6),
                               np.log10(psd[idx]+1e-10), 1)
        d["noise_slope"]  = float(slope)
        d["noise_colour"] = ("pink"     if -1.5 < slope < -0.5 else
                             "brownian" if slope < -1.5 else "white")
    else:
        d["noise_slope"] = 0.0; d["noise_colour"] = "unknown"
    return d

def audit_audio_files(fs_df: pd.DataFrame):
    print(f"\n  🎙  Audio forensics …")
    # Only diagnose MEMS files — skip Brüel (too large, handled separately)
    audio_rows = fs_df[
        (fs_df["file_type"] == "audio") &
        (fs_df["folder_l1"].str.contains("MEMS", case=False, na=False))
    ].copy()
    if len(audio_rows) == 0:
        print("     ⚠  No MEMS audio files"); return pd.DataFrame(), pd.DataFrame()
    header_rows, diag_rows = [], []
    for _, row in audio_rows.iterrows():
        p   = Path(row["path"])
        print(f"     → {row['filename']}", end="\r", flush=True)
        hdr = read_wav_info(p) if row["extension"] == ".wav" else \
              {"path": str(p), "readable": False}
        hdr["filename"] = row["filename"]
        hdr["folder"]   = row["folder_l1"]
        header_rows.append(hdr)
        if row["extension"] == ".wav":
            data, sr = read_wav_signal(p, max_seconds=60)
            diag = diagnose_audio(data, sr, str(p))
            diag.update({"filename": row["filename"], "folder": row["folder_l1"],
                         "sr": sr})
            diag_rows.append(diag)
    hdr_df  = pd.DataFrame(header_rows)
    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame()
    print(f"     ✔  {len(hdr_df)} headers  |  {len(diag_df)} WAV diagnosed")
    return hdr_df, diag_df


# =============================================================================
#  SECTION 3  ──  TELEMETRY AUDIT  (proper GPX XML parsing)
# =============================================================================

def _parse_gpx(path: Path) -> pd.DataFrame:
    """Parse GPX trackpoints via xml.etree — handles all namespace variants."""
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        # Handle namespaced and bare GPX
        ns_match = re.match(r'\{(.*?)\}', root.tag)
        ns = ns_match.group(1) if ns_match else ""
        prefix = f"{{{ns}}}" if ns else ""
        rows = []
        for trkpt in root.iter(f"{prefix}trkpt"):
            lat = trkpt.get("lat")
            lon = trkpt.get("lon")
            ele = trkpt.findtext(f"{prefix}ele")
            tim = trkpt.findtext(f"{prefix}time")
            spd = trkpt.findtext(f"{prefix}speed") or \
                  trkpt.findtext(f"{prefix}extensions/{prefix}speed")
            rows.append({
                "lat":       float(lat) if lat else None,
                "lon":       float(lon) if lon else None,
                "elevation": float(ele) if ele else None,
                "time":      tim,
                "speed":     float(spd) if spd else None,
                "source":    path.name,
            })
        if rows:
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            return df
    except Exception as e:
        print(f"     ⚠  GPX parse error {path.name}: {e}")
    return pd.DataFrame()

def _is_valid_gpx_time(ts):
    if pd.isna(ts):
        return False
    # Reject obvious junk timestamps
    if ts.year < 2024 or ts.year > 2026:
        return False
    return True

def audit_telemetry(folder_map: dict) -> dict:
    print(f"\n  🛸  Telemetry audit …")
    result = {"files": [], "combined_df": None, "issues": [], "gpx_df": None}
    dron_dir = None
    for k, v in folder_map.items():
        if "DRON" in k.upper() or "ADATOK" in k.upper():
            dron_dir = Path(v)
            break
    if dron_dir is None or not dron_dir.exists():
        result["issues"].append("No DRON-ADATOK folder found")
        return result

    dfs, gpx_dfs = [], []

    fw_start = pd.Timestamp(FLIGHT_TAKEOFF_UTC, tz="UTC") - pd.Timedelta(minutes=10)
    fw_end   = pd.Timestamp(FLIGHT_LANDING_UTC, tz="UTC") + pd.Timedelta(minutes=10)

    for f in sorted(dron_dir.rglob("*")):
        if not f.is_file():
            continue

        ext = f.suffix.lower()
        if ext not in {".gpx", ".csv", ".json", ".txt", ".log"}:
            continue   # skip WAV, EXE, SKYC, XLSX, etc.

        info = {
            "name": f.name,
            "size_kb": f.stat().st_size / 1024,
            "ext": ext,
            "columns": [],
            "n_rows": 0,
            "issues": [],
        }

        try:
            if ext == ".gpx":
                gdf = _parse_gpx(f)
                info["n_coords"] = len(gdf)
                info["columns"] = list(gdf.columns) if not gdf.empty else []

                if not gdf.empty and "time" in gdf.columns:
                    gdf = gdf[gdf["time"].apply(_is_valid_gpx_time)].copy()

                    if not gdf.empty:
                        info["time_range"] = f"{gdf['time'].min()} → {gdf['time'].max()}"

                        # keep only points near the real flight
                        gdf_fw = gdf[(gdf["time"] >= fw_start) & (gdf["time"] <= fw_end)].copy()

                        if not gdf_fw.empty:
                            gpx_dfs.append(gdf_fw)
                            info["n_coords_flight_window"] = len(gdf_fw)
                        else:
                            info["issues"].append("No valid points near flight window")
                    else:
                        info["issues"].append("All GPX timestamps invalid after filtering")

            elif ext == ".csv":
                df = pd.read_csv(f, on_bad_lines="skip", nrows=5000)
                info["columns"] = list(df.columns)
                info["n_rows"] = len(df)
                dfs.append(df)

            elif ext == ".json":
                raw = json.loads(f.read_text(errors="replace"))
                flat = pd.json_normalize(raw if isinstance(raw, list) else [raw])
                info["columns"] = list(flat.columns)
                info["n_rows"] = len(flat)
                dfs.append(flat)

            elif ext in {".txt", ".log"}:
                lines = f.read_text(errors="replace").splitlines()
                info["n_lines"] = len(lines)

        except Exception as e:
            info["issues"].append(str(e))

        result["files"].append(info)
        print(f"     📄  {f.name:<45}  {f.stat().st_size/1024:>8.1f} KB  "
              f"rows/coords={info.get('n_rows') or info.get('n_coords', 0)}")

    if dfs:
        result["combined_df"] = pd.concat(dfs, ignore_index=True)

    if gpx_dfs:
        result["gpx_df"] = pd.concat(gpx_dfs, ignore_index=True)
        print(f"     ✔  GPX(valid+windowed): {len(result['gpx_df'])} trackpoints from {len(gpx_dfs)} files")
    else:
        print("     ⚠  No GPX data survived validation/window filtering")

    return result

# =============================================================================
#  SECTION 4  ──  LOGBOOK
# =============================================================================

def audit_logbook(folder_map: dict) -> dict:
    print(f"\n  📋  Logbook audit …")
    result = {"entries": [], "raw_text": ""}
    jegy_dir = None
    for k, v in folder_map.items():
        if "JEGY" in k.upper(): jegy_dir = Path(v); break
    if jegy_dir is None or not jegy_dir.exists():
        result["issues"] = "No logbook folder"; return result
    full_text = []
    for f in sorted(jegy_dir.rglob("*")):
        if not f.is_file(): continue
        try:
            text = f.read_text(errors="replace")
            full_text.append(f"=== {f.name} ===\n{text}")
            for line in text.splitlines():
                m = re.search(r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)", line)
                if m:
                    result["entries"].append({
                        "source": f.name,
                        "timestamp": m.group(1),
                        "text": line.strip(),
                    })
            print(f"     📄  {f.name:<45}  {f.stat().st_size/1024:>6.1f} KB")
        except Exception as e:
            print(f"     ⚠  {f.name}: {e}")
    result["raw_text"] = "\n\n".join(full_text)
    print(f"     ✔  {len(result['entries'])} timestamped entries")
    return result


# =============================================================================
#  SECTION 5  ──  CONSISTENCY ANALYSIS
# =============================================================================

def consistency_analysis(hdr_df, diag_df):
    issues, report = [], {}
    if hdr_df.empty: return {"issues": ["No audio headers"]}
    for col, label in [("sample_rate","Sample rates"), ("bit_depth","Bit depths"),
                       ("n_channels","Channel counts")]:
        if col in hdr_df.columns:
            counts = hdr_df[col].value_counts().to_dict()
            report[col.replace("sample_rate","sample_rates")
                      .replace("bit_depth","bit_depths")
                      .replace("n_channels","channel_counts")] = counts
            if len(counts) > 1:
                issues.append(f"MIXED {label}: {counts}")
    if "duration_sec" in hdr_df.columns:
        dur = hdr_df["duration_sec"].dropna()
        report["duration_stats"] = {
            "min_sec":float(dur.min()), "max_sec":float(dur.max()),
            "mean_sec":float(dur.mean()), "std_sec":float(dur.std()),
        }
        if dur.std() > dur.mean() * 0.5:
            issues.append("High duration variance — possible truncation")
    if not diag_df.empty and "clipped" in diag_df.columns:
        n = int(diag_df["clipped"].sum())
        report["clipped_files"] = n
        if n: issues.append(f"{n} files have clipping")
    if not diag_df.empty and "silence_pct" in diag_df.columns:
        n = int((diag_df["silence_pct"] > 70).sum())
        if n: issues.append(f"{n} files are >70% silence")
    report["issues"]   = issues
    report["n_issues"] = len(issues)
    return report


# =============================================================================
#  SECTION 6  ──  TEMPORAL ALIGNMENT
# =============================================================================

def temporal_alignment(diag_df, tele_result, log_result):
    report = {"aligned": [], "unmatched_audio": [], "unmatched_tele": []}
    report["audio_timestamps_found"] = 0
    report["tele_range"] = "unknown"
    report["log_entries"] = len(log_result.get("entries", []))

    gpx_df = tele_result.get("gpx_df")
    if gpx_df is not None and not gpx_df.empty and "time" in gpx_df.columns:
        times = gpx_df["time"].dropna()
        if len(times):
            report["tele_range"] = f"{times.min()} → {times.max()}"

            fw_start = pd.Timestamp(FLIGHT_TAKEOFF_UTC, tz="UTC")
            fw_end   = pd.Timestamp(FLIGHT_LANDING_UTC, tz="UTC")

            in_window = gpx_df[(gpx_df["time"] >= fw_start) & (gpx_df["time"] <= fw_end)]
            report["gpx_points_in_flight_window"] = len(in_window)
            report["gpx_files_with_flight_data"] = in_window["source"].nunique() if "source" in in_window.columns else 0

            if len(in_window):
                report["aligned"].append("Validated GPX timestamps overlap known flight window")

            print(f"     📍  {len(in_window)} GPX trackpoints overlap the flight window")

    return report
# =============================================================================
#  SECTION 7  ──  DRONE-SPECIFIC PLOTS
# =============================================================================

def estimate_rotor_rpm(bruel_results, ch=0, fmin=50, fmax=150, n_blades=2):
    """
    Estimate rotor RPM over time from the acoustic blade-pass fundamental.
    Returns a DataFrame with time_s, bpf_hz, rpm.
    """
    sig_ds, sr_ds = _concat_downsampled_channel(bruel_results, ch=ch, target_sr=4000)
    if sig_ds is None:
        return pd.DataFrame()

    # STFT tuned for low-frequency tracking
    nperseg = 2048
    noverlap = 1536
    f, t, Zxx = signal.stft(sig_ds, fs=sr_ds, nperseg=nperseg, noverlap=noverlap)

    mag = np.abs(Zxx)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return pd.DataFrame()

    f_band = f[band]
    mag_band = mag[band, :]

    peak_idx = np.argmax(mag_band, axis=0)
    bpf_track = f_band[peak_idx]

    rpm_track = (60.0 * bpf_track) / n_blades

    df = pd.DataFrame({
        "time_s": t,
        "bpf_hz": bpf_track,
        "rpm": rpm_track,
    })

    # optional smoothing
    df["bpf_hz_smooth"] = df["bpf_hz"].rolling(7, center=True, min_periods=1).median()
    df["rpm_smooth"] = df["rpm"].rolling(7, center=True, min_periods=1).median()

    out = OUTPUT_DIR / f"rotor_rpm_ch{ch}.csv"
    df.to_csv(out, index=False)
    print(f"   🌀  Rotor RPM track → {out}")

    del sig_ds, Zxx, mag, mag_band
    gc.collect()

    return df

def _save(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   📊  {name}")

def _iter_bruel_arrays(bruel_results, channels=None):
    """
    Yield (arr, sr, n_ch, result_dict) from cached npy files using mmap.
    channels:
      None -> all channels
      int -> one channel
      list/tuple -> selected channels
    """
    ok_results = [r for r in bruel_results if r.get("ok") and r.get("segment_cache")]
    for r in ok_results:
        arr = np.load(r["segment_cache"], mmap_mode="r")
        sr = r["sr"]
        n_ch = arr.shape[1] if arr.ndim == 2 else 1

        if channels is None:
            sub = arr
        elif isinstance(channels, int):
            sub = arr[:, channels]
        else:
            sub = arr[:, channels]

        yield sub, sr, n_ch, r


def _iter_bruel_channel_chunks(arr, ch, chunk_frames=240000):
    """
    Iterate one channel from a memmapped 2D array in chunks.
    """
    n = arr.shape[0]
    for s in range(0, n, chunk_frames):
        e = min(s + chunk_frames, n)
        yield np.asarray(arr[s:e, ch], dtype=np.float32)


def _iter_bruel_chunks_1d(arr_1d, chunk_frames=240000):
    n = len(arr_1d)
    for s in range(0, n, chunk_frames):
        e = min(s + chunk_frames, n)
        yield np.asarray(arr_1d[s:e], dtype=np.float32)


def _running_rms_from_chunks(chunks):
    total_sq = 0.0
    total_n = 0
    for x in chunks:
        total_sq += float(np.sum(x.astype(np.float64) ** 2))
        total_n += len(x)
    return math.sqrt(total_sq / max(total_n, 1))


def _concat_downsampled_channel(bruel_results, ch=0, target_sr=8000):
    """
    Load a single channel across files, downsample each file separately,
    then concatenate only the reduced arrays.
    """
    out = []
    sr_out = None
    for arr, sr, _, _ in _iter_bruel_arrays(bruel_results, channels=ch):
        x = np.asarray(arr, dtype=np.float32)
        x_ds, sr_ds = _downsample_1d(x, sr, target_sr)
        out.append(x_ds)
        sr_out = sr_ds
        del x, x_ds
        gc.collect()

    if not out:
        return None, None

    if len(out) == 1:
        return out[0], sr_out
    return np.concatenate(out), sr_out


def _concat_downsampled_multich(bruel_results, target_sr=4000):
    """
    Downsample each cached multichannel file separately, then concatenate
    only the reduced arrays.
    """
    out = []
    sr_out = None
    n_ch_out = None

    for arr, sr, n_ch, _ in _iter_bruel_arrays(bruel_results, channels=None):
        x = np.asarray(arr, dtype=np.float32)
        x_ds, sr_ds = _downsample_2d(x, sr, target_sr)
        out.append(x_ds)
        sr_out = sr_ds
        n_ch_out = x_ds.shape[1]
        del x, x_ds
        gc.collect()

    if not out:
        return None, None, None

    if len(out) == 1:
        return out[0], sr_out, n_ch_out
    return np.concatenate(out), sr_out, n_ch_out

# ── PLOT A: Brüel flight window — all 12 channels waveform ───────────────────
def plot_bruel_channels(bruel_results: list):
    """One subplot per channel — per-file downsample then concatenate reduced data."""
    plot_sig, plot_sr, n_ch = _concat_downsampled_multich(bruel_results, target_sr=2000)
    if plot_sig is None:
        print("   ⚠  No Brüel flight-window data — skipping channel plot")
        return

    # RMS from downsampled version is good enough for display label
    rms_per_ch = [float(np.sqrt(np.mean(plot_sig[:, ch] ** 2))) for ch in range(plot_sig.shape[1])]

    t = np.arange(len(plot_sig)) / plot_sr

    fig, axes = plt.subplots(plot_sig.shape[1], 1, figsize=(22, 1.8 * plot_sig.shape[1]),
                             facecolor="#0d0d0d", sharex=True)
    if plot_sig.shape[1] == 1:
        axes = [axes]

    fig.suptitle(
        f"Brüel {plot_sig.shape[1]}-Ch Array — Drone Flight Window\n"
        f"{FLIGHT_TAKEOFF_UTC.strftime('%H:%M:%S')} → "
        f"{FLIGHT_LANDING_UTC.strftime('%H:%M:%S')} UTC  "
        f"({FLIGHT_DURATION_SEC:.0f}s)  [plot downsampled to {plot_sr/1000:.1f} kHz]",
        fontsize=13, color="white", fontweight="bold"
    )

    for ch, ax in enumerate(axes):
        ax.plot(t, plot_sig[:, ch], color=C[ch % len(C)], lw=0.25, alpha=0.85)
        ax.set_ylabel(f"Ch{ch}\n{rms_per_ch[ch]*1000:.2f}mRMS",
                      color="white", fontsize=7, rotation=0, labelpad=40)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelsize=6)
        ax.set_ylim(-0.05, 0.05)

    axes[-1].set_xlabel("Time into flight window (s)", color="white")
    plt.tight_layout()

    del plot_sig
    gc.collect()

    _save(fig, "A1_bruel_channels_flight_window.png")

def plot_bruel_psd_harmonics(bruel_results: list):
    """Per-channel PSD with file-wise accumulation to avoid giant concatenation."""
    ok_results = [r for r in bruel_results if r.get("ok") and r.get("segment_cache")]
    if not ok_results:
        print("   ⚠  No Brüel flight-window data — skipping PSD plot")
        return

    psd_accum = {}
    psd_count = {}
    f_ref = None
    sr_ref = None
    n_ch_ref = None

    for arr, sr, n_ch, r in _iter_bruel_arrays(bruel_results, channels=None):
        sr_ref = sr
        n_ch_ref = n_ch

        # use smaller nperseg to reduce memory
        nperseg = min(16384, arr.shape[0] // 4)
        if nperseg < 2048:
            nperseg = min(arr.shape[0], 2048)

        for ch in range(arr.shape[1]):
            x = np.asarray(arr[:, ch], dtype=np.float32)
            f_psd, psd = signal.welch(x, fs=sr, nperseg=nperseg)

            if f_ref is None:
                f_ref = f_psd

            if ch not in psd_accum:
                psd_accum[ch] = psd.astype(np.float64)
                psd_count[ch] = 1
            else:
                psd_accum[ch] += psd
                psd_count[ch] += 1

            del x, psd
            gc.collect()

    if f_ref is None:
        print("   ⚠  PSD computation failed")
        return

    fig, ax = plt.subplots(figsize=(20, 8), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"Brüel {n_ch_ref}-Channel PSD — Flight Window  (sr={sr_ref/1000:.0f} kHz)\n"
        f"BPF={DRONE_BPF_HZ} Hz  |  {DRONE_HARMONICS} harmonics marked",
        fontsize=13, color="white"
    )

    for ch in sorted(psd_accum):
        mean_psd = psd_accum[ch] / psd_count[ch]
        ax.semilogy(f_ref / 1000, mean_psd, color=C[ch % len(C)],
                    lw=0.9, alpha=0.75, label=f"Ch{ch}")

    for h in range(1, DRONE_HARMONICS + 1):
        f_h = DRONE_BPF_HZ * h / 1000
        if f_h > sr_ref / 2000:
            break
        ax.axvline(f_h, color="white", lw=0.6, ls="--", alpha=0.45)
        if h <= 10:
            ax.text(f_h, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e-5,
                    f"H{h}", color="white", fontsize=6, rotation=90,
                    va="bottom", ha="center")

    ax.axvline(MOTOR_HZ / 1000, color=C[1], lw=1.2, ls=":", alpha=0.7,
               label=f"Motor ~{MOTOR_HZ}Hz")
    ax.set_xlabel("Frequency (kHz)", color="white", fontsize=11)
    ax.set_ylabel("PSD (V²/Hz)", color="white", fontsize=11)
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white",
              ncol=4, loc="upper right")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1, which="both")
    ax.set_xlim(0, min(sr_ref / 2000, 20))

    gc.collect()
    _save(fig, "A2_bruel_psd_harmonics.png")

def plot_bruel_spectrogram(bruel_results: list):
    """Memory-safe spectrogram of channel 0."""
    sig_ds, sr_ds = _concat_downsampled_channel(bruel_results, ch=0, target_sr=8000)
    if sig_ds is None:
        print("   ⚠  No Brüel flight-window data — skipping spectrogram")
        return

    fig, axes = plt.subplots(2, 1, figsize=(22, 10), facecolor="#0d0d0d")
    fig.suptitle(
        f"Brüel Ch0 Spectrogram — Drone Flight Window "
        f"(downsampled to {sr_ds/1000:.1f} kHz)",
        fontsize=13, color="white", fontweight="bold"
    )

    ax = axes[0]
    f, t_stft, Zxx = signal.stft(sig_ds, fs=sr_ds, nperseg=2048, noverlap=1536)
    db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
    del Zxx
    gc.collect()

    im = ax.pcolormesh(
        t_stft, f, db,
        shading="auto",
        cmap="inferno",
        vmin=np.percentile(db, 5),
        vmax=np.percentile(db, 99)
    )
    ax.set_ylabel("Freq (Hz)", color="white")
    ax.set_ylim(0, min(4000, sr_ds // 2))
    ax.set_title("Full spectrum", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")
    plt.colorbar(im, ax=ax, label="dB", format="%+.0f")

    ax = axes[1]
    f2, t2, Zxx2 = signal.stft(sig_ds, fs=sr_ds, nperseg=1024, noverlap=768)
    db2 = 10 * np.log10(np.abs(Zxx2) ** 2 + 1e-10)
    del Zxx2
    gc.collect()

    freq_mask = f2 <= 500
    db2_zoom = db2[freq_mask]

    im2 = ax.pcolormesh(
        t2, f2[freq_mask], db2_zoom,
        shading="auto",
        cmap="magma",
        vmin=np.percentile(db2_zoom, 5),
        vmax=np.percentile(db2_zoom, 99)
    )

    for h in range(1, 8):
        f_h = DRONE_BPF_HZ * h
        if f_h > 500:
            break
        ax.axhline(f_h, color="cyan", lw=0.8, alpha=0.6)
        ax.text(t2[-1] * 0.98, f_h + 2, f"H{h}", color="cyan", fontsize=7, ha="right")

    ax.set_ylabel("Freq (Hz)", color="white")
    ax.set_xlabel("Time into flight window (s)", color="white")
    ax.set_ylim(0, 500)
    ax.set_title("Zoom: 0–500 Hz", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a1a")
    plt.colorbar(im2, ax=ax, label="dB", format="%+.0f")

    del sig_ds, db, db2, db2_zoom
    gc.collect()

    plt.tight_layout()
    _save(fig, "A3_bruel_spectrogram_flight_window.png")

def plot_bruel_snr_timeline(bruel_results: list):
    """Per-channel RMS and BPF-band RMS timeline using per-file downsample then concatenate."""
    combined_ds, sr_ds, n_ch = _concat_downsampled_multich(bruel_results, target_sr=2000)
    if combined_ds is None:
        print("   ⚠  No Brüel flight-window data — skipping SNR timeline")
        return

    frame = int(0.5 * sr_ds)
    n_fr = len(combined_ds) // frame
    t_frames = np.arange(n_fr) * 0.5

    nyq = sr_ds / 2
    low = max(DRONE_BPF_HZ - 50, 1) / nyq
    high = min((DRONE_BPF_HZ + 50) / nyq, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")

    rms_db_all = []
    bprms_db_all = []

    for ch in range(combined_ds.shape[1]):
        ch_sig = combined_ds[:, ch]

        frames = ch_sig[:n_fr * frame].reshape(n_fr, frame)
        rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
        rms_db_all.append(rms_db)

        ch_bp = signal.filtfilt(b, a, ch_sig)
        bp_fr = ch_bp[:n_fr * frame].reshape(n_fr, frame)
        bprms = 20 * np.log10(np.sqrt(np.mean(bp_fr ** 2, axis=1)) + 1e-10)
        bprms_db_all.append(bprms)

    del combined_ds
    gc.collect()

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle(
        f"Brüel Array — RMS & Band-pass RMS Timeline "
        f"(downsampled to {sr_ds} Hz)",
        fontsize=13, color="white"
    )

    ax = axes[0]
    for ch, rms_db in enumerate(rms_db_all):
        ax.plot(t_frames, rms_db, color=C[ch % len(C)], lw=0.8, alpha=0.75, label=f"Ch{ch}")
    ax.set_ylabel("RMS Level (dBFS)", color="white")
    ax.set_title("Broadband RMS", color="white")
    ax.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white", ncol=7)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)

    ax = axes[1]
    for ch, bprms_db in enumerate(bprms_db_all):
        ax.plot(t_frames, bprms_db, color=C[ch % len(C)], lw=0.8, alpha=0.75, label=f"Ch{ch}")
    ax.set_ylabel("Band-pass RMS (dBFS)", color="white")
    ax.set_xlabel("Time into flight window (s)", color="white")
    ax.set_title(f"Band-pass RMS ({DRONE_BPF_HZ}±50 Hz)", color="white")
    ax.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white", ncol=7)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    _save(fig, "A4_bruel_snr_timeline.png")

# ── PLOT E: GPX flight track ──────────────────────────────────────────────────
def plot_gpx_track(tele_result: dict):
    gpx_df = tele_result.get("gpx_df")
    if gpx_df is None or gpx_df.empty:
        print("   ⚠  No GPX data — skipping track plot"); return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0d0d0d")
    fig.suptitle("GPX Telemetry — Flight Track", fontsize=13, color="white")

    # All tracks coloured by file
    ax = axes[0]
    sources = gpx_df["source"].unique() if "source" in gpx_df.columns else ["all"]
    for i, src in enumerate(sources[:20]):   # cap at 20 for readability
        sub = gpx_df[gpx_df["source"]==src] if "source" in gpx_df.columns else gpx_df
        sub = sub.dropna(subset=["lat","lon"])
        ax.plot(sub["lon"], sub["lat"], color=C[i%len(C)],
                lw=0.8, alpha=0.6, label=src[:20])
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.set_title("All GPX Tracks", color="white", fontsize=10)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)

    # Flight window only
    ax = axes[1]
    if "time" in gpx_df.columns:
        fw_start = pd.Timestamp(FLIGHT_TAKEOFF_UTC, tz="UTC")
        fw_end   = pd.Timestamp(FLIGHT_LANDING_UTC, tz="UTC")
        fw = gpx_df[(gpx_df["time"] >= fw_start) & (gpx_df["time"] <= fw_end)]
        fw = fw.dropna(subset=["lat","lon"])
        if len(fw):
            sc = ax.scatter(fw["lon"], fw["lat"],
                            c=range(len(fw)), cmap="plasma",
                            s=20, alpha=0.9, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Time index")
            ax.set_title(f"Flight Window Only\n({len(fw)} trackpoints)",
                         color="white", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No GPX data\nin flight window",
                    ha="center", va="center", color="white",
                    transform=ax.transAxes)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    _save(fig, "A5_gpx_flight_track.png")

# ── Standard plots (abbreviated, same as v1 but adapted) ─────────────────────
def plot_filesystem_overview(fs_df):
    if fs_df.empty: return
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle("File System Overview", fontsize=16, color="white")
    tc = fs_df["file_type"].value_counts()
    axes[0,0].bar(tc.index, tc.values, color=C[:len(tc)], edgecolor="white", lw=0.5)
    axes[0,0].set_title("Files by Type", color="white")
    axes[0,0].set_facecolor("#1a1a1a"); axes[0,0].tick_params(colors="white")
    sz = fs_df.groupby("file_type")["size_mb"].sum().sort_values(ascending=False)
    axes[0,1].bar(sz.index, sz.values, color=C[:len(sz)], edgecolor="white", lw=0.5)
    axes[0,1].set_title("Size by File Type (MB)", color="white")
    axes[0,1].set_facecolor("#1a1a1a"); axes[0,1].tick_params(colors="white")
    fc = fs_df.groupby("folder_l1")["size_mb"].sum().sort_values()
    axes[0,2].barh(fc.index, fc.values, color=C[:len(fc)], edgecolor="white", lw=0.5)
    axes[0,2].set_title("Size by Folder (MB)", color="white")
    axes[0,2].set_facecolor("#1a1a1a"); axes[0,2].tick_params(colors="white")
    fcount = fs_df.groupby("folder_l1").size().sort_values(ascending=False)
    axes[1,0].bar(fcount.index, fcount.values, color=C[:len(fcount)], edgecolor="white")
    axes[1,0].set_title("File Count per Folder", color="white")
    axes[1,0].set_facecolor("#1a1a1a"); axes[1,0].tick_params(colors="white", axis="x", rotation=30, labelsize=8)
    axes[1,0].tick_params(colors="white", axis="y")
    for i, (ft, grp) in enumerate(fs_df.groupby("file_type")):
        axes[1,1].hist(grp["size_mb"].values, bins=30, alpha=0.65,
                       color=C[i%len(C)], label=ft, edgecolor="black")
    axes[1,1].set_xscale("log"); axes[1,1].set_title("File Size Distribution (log)", color="white")
    axes[1,1].legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    axes[1,1].set_facecolor("#1a1a1a"); axes[1,1].tick_params(colors="white")
    ext_c = fs_df["extension"].value_counts().head(10)
    axes[1,2].pie(ext_c.values, labels=ext_c.index, colors=C[:len(ext_c)],
                  autopct="%1.1f%%", textprops={"color":"white","fontsize":8})
    axes[1,2].set_title("Top-10 Extensions", color="white")
    plt.tight_layout()
    _save(fig, "01_filesystem_overview.png")

def plot_audio_psd_overview(diag_df):
    """Average PSD per folder — adapted for 192 kHz range."""
    if diag_df.empty: return
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#0d0d0d")
    fig.suptitle("Average PSD per Folder (all audio files)", fontsize=14, color="white")
    ax.set_facecolor("#1a1a1a")
    for i, fld in enumerate(folders):
        sub  = diag_df[diag_df["folder"]==fld] if "folder" in diag_df.columns else diag_df
        psds, sr_ref = [], None
        for _, row in sub.head(5).iterrows():
            data, sr = read_wav_signal(Path(row["path"]))
            if data is None: continue
            if sr_ref is None: sr_ref = sr
            if sr != sr_ref:
                data = signal.resample(data, int(len(data)*sr_ref/sr)).astype(np.float32)
            f, psd = signal.welch(data, fs=sr_ref, nperseg=min(N_FFT, len(data)//4 or 4096))
            psds.append(psd)
        if not psds: continue
        mean_psd = np.mean(psds, axis=0)
        ax.semilogy(f/1000, mean_psd, color=C[i%len(C)], lw=2, label=str(fld))
    ax.set_xlabel("Frequency (kHz)", color="white", fontsize=11)
    ax.set_ylabel("PSD", color="white", fontsize=11)
    ax.legend(fontsize=9, facecolor="#1a1a1a", labelcolor="white")
    ax.tick_params(colors="white"); ax.grid(True, alpha=0.12, which="both")
    for h in range(1, 8):
        f_h = DRONE_BPF_HZ * h / 1000
        ax.axvline(f_h, color="cyan", lw=0.8, ls=":", alpha=0.5)
        ax.text(f_h, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"H{h}", color="cyan", fontsize=6, rotation=90, va="top")
    plt.tight_layout()
    _save(fig, "02_psd_overview.png")

def plot_signal_health(diag_df):
    if diag_df.empty: return
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle("Signal Health Dashboard", fontsize=15, color="white")
    def _hist(ax, col, title):
        if col not in diag_df.columns: return
        folders = diag_df.get("folder", pd.Series(["all"]*len(diag_df)))
        for i, (fld, idx) in enumerate(
                diag_df.groupby(folders if "folder" in diag_df.columns
                                else pd.Series(["all"]*len(diag_df))).groups.items()):
            ax.hist(diag_df.loc[idx, col].dropna(), bins=25, alpha=0.7,
                    color=C[i%len(C)], label=str(fld), edgecolor="black")
        ax.set_title(title, color="white", fontsize=10)
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    _hist(axes[0,0], "snr_db",         "SNR per File (dB)")
    _hist(axes[0,1], "rms_db_mean",    "Mean RMS Level (dB)")
    _hist(axes[0,2], "noise_floor_db", "Noise Floor (dB)")
    _hist(axes[1,0], "clip_pct",       "Clipping %")
    _hist(axes[1,1], "silence_pct",    "Silence %")
    _hist(axes[1,2], "dominant_freq_hz","Dominant Freq (Hz)")
    plt.tight_layout()
    _save(fig, "03_signal_health.png")

def plot_gpx_altitude(tele_result):
    gpx_df = tele_result.get("gpx_df")
    if gpx_df is None or gpx_df.empty or "elevation" not in gpx_df.columns: return
    fig, ax = plt.subplots(figsize=(18, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    fig.suptitle("GPX Altitude Profile — All Files", fontsize=13, color="white")
    if "time" in gpx_df.columns:
        gpx_sorted = gpx_df.dropna(subset=["time","elevation"]).sort_values("time")
        ax.plot(gpx_sorted["time"], gpx_sorted["elevation"],
                color=C[0], lw=0.8, alpha=0.85)
        ax.fill_between(gpx_sorted["time"], gpx_sorted["elevation"],
                        alpha=0.2, color=C[0])
        # Mark flight window
        fw_s = pd.Timestamp(FLIGHT_TAKEOFF_UTC, tz="UTC")
        fw_e = pd.Timestamp(FLIGHT_LANDING_UTC, tz="UTC")
        ax.axvspan(fw_s, fw_e, color=C[1], alpha=0.15, label="Flight window")
        ax.axvline(fw_s, color=C[1], lw=1.5, ls="--")
        ax.axvline(fw_e, color=C[1], lw=1.5, ls="--")
        ax.legend(fontsize=9, facecolor="#1a1a1a", labelcolor="white")
    ax.set_ylabel("Elevation (m)", color="white")
    ax.tick_params(colors="white"); ax.grid(True, alpha=0.1)
    plt.tight_layout()
    _save(fig, "04_gpx_altitude.png")

def plot_rotor_rpm(rpm_df, n_blades=2):
    if rpm_df is None or rpm_df.empty:
        print("   ⚠  No RPM data — skipping RPM plot")
        return

    fig, ax = plt.subplots(figsize=(18, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    ax.plot(rpm_df["time_s"], rpm_df["rpm"], alpha=0.35, lw=0.8, label="Instant RPM")
    ax.plot(rpm_df["time_s"], rpm_df["rpm_smooth"], lw=2.0, label="Smoothed RPM")

    ax.set_title(f"Estimated Rotor RPM from Acoustic BPF (assuming {n_blades} blades)",
                 color="white")
    ax.set_xlabel("Time into flight window (s)", color="white")
    ax.set_ylabel("RPM", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor="#1a1a1a", labelcolor="white")

    plt.tight_layout()
    _save(fig, f"A6_rotor_rpm_{n_blades}blades.png")

# =============================================================================
#  SECTION 8  ──  REPORT
# =============================================================================

def write_report(fs_df, hdr_df, diag_df, consist, tele_result,
                 log_result, align_result, bruel_results, t_elapsed):
    report = {
        "generated":  datetime.now().isoformat(),
        "elapsed_sec": round(t_elapsed, 1),
        "flight_window": {
            "takeoff_utc":  FLIGHT_TAKEOFF_UTC.isoformat(),
            "landing_utc":  FLIGHT_LANDING_UTC.isoformat(),
            "duration_sec": FLIGHT_DURATION_SEC,
        },
        "filesystem": {
            "total_files":  int(len(fs_df)) if not fs_df.empty else 0,
            "total_size_gb": float(fs_df["size_mb"].sum()/1024) if not fs_df.empty else 0,
            "by_type":  fs_df["file_type"].value_counts().to_dict() if not fs_df.empty else {},
            "by_folder": fs_df["folder_l1"].value_counts().to_dict() if not fs_df.empty else {},
        },
        "audio": {
            "total_wav":  int(len(hdr_df)) if not hdr_df.empty else 0,
            "sample_rates": (hdr_df["sample_rate"].value_counts().to_dict()
                             if not hdr_df.empty and "sample_rate" in hdr_df.columns else {}),
        },
        "signal_quality": {
            "mean_snr_db": float(diag_df["snr_db"].mean())
                           if not diag_df.empty and "snr_db" in diag_df.columns else None,
            "clipped_files": int(diag_df["clipped"].sum())
                             if not diag_df.empty and "clipped" in diag_df.columns else 0,
            "noise_colours": (diag_df["noise_colour"].value_counts().to_dict()
                              if not diag_df.empty and "noise_colour" in diag_df.columns else {}),
        },
        "consistency": consist,
        "telemetry": {
            "n_files": len(tele_result.get("files",[])),
            "n_gpx_points": len(tele_result["gpx_df"]) if tele_result.get("gpx_df") is not None else 0,
            "n_gpx_in_flight": align_result.get("gpx_points_in_flight_window", 0),
        },
        "logbook": {"n_entries": len(log_result.get("entries",[]))},
        "alignment": {k:v for k,v in align_result.items() if k != "gpx_df"},
        "bruel": {
            "files_downloaded": len(bruel_results),
            "flight_window_ok": sum(1 for r in bruel_results if r.get("ok")),
            "total_flight_sec": sum(r.get("flight_duration_sec",0) for r in bruel_results),
            "channels": bruel_results[0]["n_ch"] if bruel_results and bruel_results[0].get("ok") else 0,
            "sample_rate": bruel_results[0]["sr"] if bruel_results and bruel_results[0].get("ok") else 0,
        },
    }
    json_path = OUTPUT_DIR / "audit_report_v2.json"
    txt_path  = OUTPUT_DIR / "audit_report_v2.txt"
    with open(json_path,"w") as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        "=" * 72,
        "  DUNAKESZI DRONE DATASET — FORENSIC AUDIT REPORT  v2",
        f"  Generated : {report['generated']}",
        f"  Elapsed   : {report['elapsed_sec']} s",
        "=" * 72, "",
        "FLIGHT WINDOW",
        "─" * 40,
        f"  Takeoff : {FLIGHT_TAKEOFF_UTC}",
        f"  Landing : {FLIGHT_LANDING_UTC}",
        f"  Duration: {FLIGHT_DURATION_SEC:.0f}s  ({FLIGHT_DURATION_SEC/60:.1f} min)",
        "",
        "BRÜEL RECORDINGS",
        "─" * 40,
        f"  Files downloaded       : {report['bruel']['files_downloaded']}",
        f"  Flight window segments : {report['bruel']['flight_window_ok']}",
        f"  Total flight audio (s) : {report['bruel']['total_flight_sec']:.1f}",
        f"  Channels               : {report['bruel']['channels']}",
        f"  Sample rate            : {report['bruel']['sample_rate']} Hz",
        "",
        "FILE SYSTEM",
        "─" * 40,
        f"  Total files : {report['filesystem']['total_files']}",
        f"  Total size  : {report['filesystem']['total_size_gb']:.3f} GB",
        f"  By type     : {report['filesystem']['by_type']}",
        "",
        "SIGNAL QUALITY (MEMS files)",
        "─" * 40,
        f"  Mean SNR       : {report['signal_quality']['mean_snr_db']} dB",
        f"  Clipped files  : {report['signal_quality']['clipped_files']}",
        f"  Noise colours  : {report['signal_quality']['noise_colours']}",
        "",
        "TELEMETRY (GPX)",
        "─" * 40,
        f"  GPX files          : {report['telemetry']['n_files']}",
        f"  Total trackpoints  : {report['telemetry']['n_gpx_points']}",
        f"  In flight window   : {report['telemetry']['n_gpx_in_flight']}",
        "",
        "CONSISTENCY ISSUES",
        "─" * 40,
    ] + [f"  {'⚠' if i else '✔'}  {iss}"
         for i, iss in enumerate(consist.get("issues",["No issues found"]))] + [
        "=" * 72,
    ]
    txt_path.write_text("\n".join(lines))
    print(f"\n  📝  {json_path}")
    print(f"  📝  {txt_path}")
    return report


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t0 = time.time()

    print(f"\n{SEP}\n  STEP 0 — DATA ACQUISITION\n{SEP}")
    folder_map = download_dataset(WORKSPACE)

    print(f"\n{SEP}\n  STEP 1 — FILE-SYSTEM AUDIT\n{SEP}")
    root  = folder_map.get("_root", WORKSPACE)
    fs_df = audit_filesystem(Path(root))
    fs_df.to_csv(OUTPUT_DIR / "filesystem_audit.csv", index=False)

    print(f"\n{SEP}\n  STEP 2 — AUDIO FILE FORENSICS (MEMS)\n{SEP}")
    hdr_df, diag_df = audit_audio_files(fs_df)
    hdr_df.to_csv(OUTPUT_DIR / "audio_headers.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(OUTPUT_DIR / "audio_diagnostics.csv", index=False)

    print(f"\n{SEP}\n  STEP 3 — BRÜEL FLIGHT WINDOW EXTRACTION\n{SEP}")
    bruel_results = []
    for fname, file_start in BRUEL_FILE_STARTS.items():
        # Search in workspace for the downloaded Brüel file
        hits = list(WORKSPACE.rglob(fname))
        if not hits:
            print(f"     ⚠  {fname} not found in workspace — check download")
            continue
        result = read_bruel_flight_window(hits[0], file_start)
        bruel_results.append(result)

    print(f"\n{SEP}\n  STEP 4 — TELEMETRY AUDIT (proper GPX)\n{SEP}")
    tele_result = audit_telemetry(folder_map)
    if tele_result.get("gpx_df") is not None:
        tele_result["gpx_df"].to_csv(OUTPUT_DIR / "gpx_combined.csv", index=False)

    print(f"\n{SEP}\n  STEP 5 — LOGBOOK\n{SEP}")
    log_result = audit_logbook(folder_map)

    print(f"\n{SEP}\n  STEP 6 — CONSISTENCY\n{SEP}")
    consist = consistency_analysis(hdr_df, diag_df)

    print(f"\n{SEP}\n  STEP 7 — TEMPORAL ALIGNMENT\n{SEP}")
    align_result = temporal_alignment(diag_df, tele_result, log_result)

    print(f"\n{SEP}\n  STEP 8 — GENERATING PLOTS\n{SEP}")
    # Standard overview plots
    plot_filesystem_overview(fs_df)
    plot_audio_psd_overview(diag_df)
    plot_signal_health(diag_df)
    plot_gpx_altitude(tele_result)
    # Drone-specific Brüel plots
    plot_bruel_channels(bruel_results)
    plot_bruel_psd_harmonics(bruel_results)
    plot_bruel_spectrogram(bruel_results)
    plot_bruel_snr_timeline(bruel_results)
    plot_gpx_track(tele_result)

    rpm_df_2 = estimate_rotor_rpm(bruel_results, ch=0, fmin=50, fmax=150, n_blades=2)
    plot_rotor_rpm(rpm_df_2, n_blades=2)

    rpm_df_3 = estimate_rotor_rpm(bruel_results, ch=0, fmin=50, fmax=150, n_blades=3)
    plot_rotor_rpm(rpm_df_3, n_blades=3)

    print(f"\n{SEP}\n  STEP 9 — REPORT\n{SEP}")
    report = write_report(fs_df, hdr_df, diag_df, consist, tele_result,
                          log_result, align_result, bruel_results,
                          time.time() - t0)

    print(f"\n{SEP2}")
    print("  AUDIT v2 COMPLETE")
    print(SEP2)
    print(f"  Elapsed         : {report['elapsed_sec']} s")
    print(f"  Files audited   : {report['filesystem']['total_files']}")
    print(f"  Brüel segments  : {report['bruel']['flight_window_ok']}")
    print(f"  GPX trackpoints : {report['telemetry']['n_gpx_points']}")
    print(f"  In flight win.  : {report['telemetry']['n_gpx_in_flight']}")
    print(f"  Plots → {PLOT_DIR}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()