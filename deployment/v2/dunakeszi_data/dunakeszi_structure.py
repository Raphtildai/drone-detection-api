"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DUNAKESZI DRONE DATASET — DEEP FORENSIC ANALYSIS                         ║
║   Stage 1 of N: Understand everything before touching a model               ║
║                                                                              ║
║   Supports:  Nextcloud  (nc.ek-cer.hu)  via WebDAV — NO login needed       ║
║              Local path (if already downloaded)                             ║
║                                                                              ║
║   What this script does (NO training, NO models):                           ║
║     0.  Nextcloud WebDAV selective download                                 ║
║     1.  Full file-system audit       (every file, every byte)               ║
║     2.  Audio file forensics         (codec, sr, bit-depth, channels, dur)  ║
║     3.  Waveform inspection          (DC offset, clipping, silence, RMS)    ║
║     4.  Spectral deep-dive           (PSD, dominant freqs, harmonics)       ║
║     5.  Noise floor characterisation (SNR, pink vs white, floor level)      ║
║     6.  Cross-file consistency       (sr mismatch, duration spread, gaps)   ║
║     7.  Telemetry structure audit    (columns, units, GPS coverage, gaps)   ║
║     8.  Logbook parsing              (sessions, annotations, timestamps)    ║
║     9.  Temporal alignment           (audio ↔ telemetry timestamp match)    ║
║    10.  20 publication-quality plots                                        ║
║    11.  Plain-English findings report (JSON + TXT)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

QUICK START
───────────
  Step 1 — paste your Nextcloud share link into SHARE_URL below
           e.g.  "https://nc.ek-cer.hu/index.php/s/6xBE46A9HQCCQ8F"

  Step 2 — python3 dunakeszi_audit.py

  The script will:
    • List every file in all 6 sub-folders via WebDAV PROPFIND
    • Download MEMS .wav files (up to MAX_MEMS_FILES / MAX_MEMS_MB)
    • Download full DRON-ADATOK + JEGYZOKONYV
    • Index BRUEL_VIDEO / FOTÓ / VIDEÓ without downloading them
    • Run all 8 audit stages and generate 20 plots

  Option B — Local path (already downloaded):
    Set  LOCAL_DATA_ROOT  to your folder — SHARE_URL is ignored
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, json, wave, struct, io, sys, time, hashlib, math
import warnings, logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from urllib.parse import urlencode, quote

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── third-party ───────────────────────────────────────────────────────────────
import requests
import numpy as np
import pandas as pd
from scipy import signal, fft, stats
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatter, AutoMinorLocator
import seaborn as sns

# =============================================================================
#  ▶▶  USER SETTINGS  ◀◀  — edit these before running
# =============================================================================

# ── Nextcloud share link ──────────────────────────────────────────────────────
# Paste the share link for the ROOT Dunakeszi folder
# e.g.  SHARE_URL = "https://nc.ek-cer.hu/index.php/s/6xBE46A9HQCCQ8F"
SHARE_URL = "https://nc.ek-cer.hu/index.php/s/...."

# ── Local fallback (if already downloaded) ───────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

LOCAL_DATA_ROOT = SCRIPT_DIR / "dunakeszi_data"

# ── Download limits  (prevents accidentally pulling 165 GB) ──────────────────
MAX_MEMS_FILES   = 60      # max number of MEMS .wav files to download
MAX_MEMS_MB      = 600     # hard MB cap for the MEMS folder total
TRIM_SECONDS     = 30      # first N seconds to download per large audio file
                           # (uses HTTP Range header — no need to download full file)

# Folders to INDEX (list all files) but NOT download
DOWNLOAD_META_ONLY_FOLDERS = {"VIDEO", "FOTO", "BRUEL"}

# Output
WORKSPACE  = SCRIPT_DIR / "dunakeszi_audit_ws"
OUTPUT_DIR = SCRIPT_DIR / "dunakeszi_audit_output"
PLOT_DIR   = OUTPUT_DIR / "plots"

for d in [WORKSPACE, OUTPUT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Signal processing
SR_TARGET  = 22050
N_FFT      = 4096
HOP        = 1024
N_MELS     = 128
SEED       = 42
np.random.seed(SEED)

# Visuals
plt.style.use("dark_background")
C = ["#00FFCC","#FF4C6A","#FFD700","#7B68EE","#FF8C00",
     "#00BFFF","#ADFF2F","#FF69B4","#40E0D0","#FF6347",
     "#DA70D6","#98FB98","#F0E68C","#87CEEB","#DEB887"]
sns.set_palette(C)

SEP  = "─" * 72
SEP2 = "═" * 72

print(SEP2)
print("  DUNAKESZI DRONE DATASET  —  FORENSIC ANALYSIS STAGE 1")
print(SEP2)

# =============================================================================
#  SECTION 0  ──  NEXTCLOUD WebDAV DOWNLOADER
# =============================================================================
#
#  Nextcloud exposes a standard WebDAV endpoint for public shares:
#
#    PROPFIND  https://<host>/public.php/webdav/<path>
#              Authorization: Basic <token> ""     (token = share token, empty pw)
#              Depth: 1
#    → returns XML <multistatus> with all children
#
#    GET       https://<host>/public.php/webdav/<path>/<filename>
#              Authorization: Basic <token> ""
#    → returns file bytes  (Range header supported for partial download)
#
# =============================================================================

import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse
import base64

# WebDAV XML namespaces
_DAV_NS   = "DAV:"
_NC_NS    = "http://nextcloud.org/ns"
_OC_NS    = "http://owncloud.org/ns"


def _nc_parse_share_url(url: str):
    """
    Extract (host_base, share_token) from a Nextcloud share URL.
    Handles both:
      https://nc.example.com/index.php/s/TOKEN
      https://nc.example.com/index.php/s/TOKEN?dir=/SubFolder
    """
    p = urlparse(url)
    host_base = f"{p.scheme}://{p.netloc}"
    # token is the path component after /s/
    m = re.search(r"/s/([A-Za-z0-9]+)", p.path)
    if not m:
        raise ValueError(f"Cannot extract share token from URL: {url}")
    token = m.group(1)
    # optional sub-path encoded in ?dir=
    subpath = ""
    if "dir=" in (p.query or ""):
        for part in p.query.split("&"):
            if part.startswith("dir="):
                subpath = unquote(part[4:]).lstrip("/")
    return host_base, token, subpath


def _nc_webdav_base(host_base: str, token: str) -> str:
    return f"{host_base}/public.php/webdav"


def _nc_auth_header(token: str) -> dict:
    """Nextcloud public share: user=token, password=empty string."""
    creds = base64.b64encode(f"{token}:".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _nc_propfind(session: requests.Session, webdav_base: str,
                 token: str, remote_path: str = "", depth: int = 1) -> list:
    """
    PROPFIND a directory; returns list of dicts:
      {href, name, size, content_type, is_dir, last_modified}
    """
    url  = webdav_base.rstrip("/")
    if remote_path:
        url += "/" + remote_path.strip("/")

    headers = {
        **_nc_auth_header(token),
        "Depth":        str(depth),
        "Content-Type": "application/xml; charset=utf-8",
    }
    body = (b'<?xml version="1.0"?>'
            b'<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">'
            b'  <d:prop>'
            b'    <d:displayname/><d:getcontentlength/>'
            b'    <d:getcontenttype/><d:resourcetype/>'
            b'    <d:getlastmodified/>'
            b'  </d:prop>'
            b'</d:propfind>')

    r = session.request("PROPFIND", url, headers=headers,
                        data=body, timeout=30)
    if r.status_code not in (207, 200):
        r.raise_for_status()

    root_xml = ET.fromstring(r.text)
    items    = []
    base_href = None

    for resp in root_xml.findall(f"{{{_DAV_NS}}}response"):
        href = resp.findtext(f"{{{_DAV_NS}}}href", "")
        href_decoded = unquote(href)

        # First response is the directory itself — record its href to skip it
        if base_href is None:
            base_href = href_decoded
            continue

        prop_ok = resp.find(
            f".//{{{_DAV_NS}}}propstat[{{{_DAV_NS}}}status='HTTP/1.1 200 OK']"
            f"/{{{_DAV_NS}}}prop")
        if prop_ok is None:
            # Try without namespace prefix in status text
            for ps in resp.findall(f"{{{_DAV_NS}}}propstat"):
                status = ps.findtext(f"{{{_DAV_NS}}}status", "")
                if "200" in status:
                    prop_ok = ps.find(f"{{{_DAV_NS}}}prop")
                    break

        if prop_ok is None:
            continue

        rt    = prop_ok.find(f"{{{_DAV_NS}}}resourcetype")
        is_dir = rt is not None and rt.find(f"{{{_DAV_NS}}}collection") is not None

        name  = prop_ok.findtext(f"{{{_DAV_NS}}}displayname", "")
        if not name:
            # Fall back to last path component of href
            name = href_decoded.rstrip("/").split("/")[-1]

        size_text = prop_ok.findtext(f"{{{_DAV_NS}}}getcontentlength", "0")
        try:    size = int(size_text)
        except: size = 0

        ctype = prop_ok.findtext(f"{{{_DAV_NS}}}getcontenttype", "")
        lmod  = prop_ok.findtext(f"{{{_DAV_NS}}}getlastmodified", "")

        items.append({
            "href":          href_decoded,
            "name":          name,
            "size":          size,
            "size_mb":       size / 1024**2,
            "content_type":  ctype,
            "is_dir":        is_dir,
            "last_modified": lmod,
            # Remote path relative to webdav root
            "remote_path":   href_decoded.split("/webdav/", 1)[-1] if "/webdav/" in href_decoded else name,
        })

    return items


def _nc_collect_files(session, webdav_base, token,
                      remote_path="", depth=4) -> list:
    """Recursively list all files under remote_path."""
    items = _nc_propfind(session, webdav_base, token, remote_path, depth=1)
    files = []
    for it in items:
        if not it["is_dir"]:
            files.append(it)
        elif depth > 0:
            files.extend(_nc_collect_files(
                session, webdav_base, token,
                it["remote_path"], depth - 1))
    return files


def _nc_download(session: requests.Session, webdav_base: str,
                 token: str, remote_path: str,
                 dest: Path, byte_cap: int = None) -> bool:
    """Download one file from Nextcloud WebDAV; optional byte cap."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        return True  # already downloaded

    url     = webdav_base.rstrip("/") + "/" + remote_path.lstrip("/")
    headers = _nc_auth_header(token)
    if byte_cap:
        headers["Range"] = f"bytes=0-{byte_cap - 1}"

    try:
        r = session.get(url, headers=headers, stream=True, timeout=120)
        if r.status_code not in (200, 206):
            r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk: f.write(chunk)
        return True
    except Exception as e:
        print(f"\n      ⚠  {dest.name}: {e}")
        if dest.exists(): dest.unlink()
        return False


def _folder_key(name: str) -> str:
    """Map a folder display name to one of the known dataset keys."""
    u = name.upper()
    for k in ["MEMS", "DRON", "JEGY", "VIDEO", "FOTO", "BRUEL"]:
        if k in u:
            return k
    return u


def download_dataset(workspace: Path) -> dict:
    """
    Main entry point.
    Returns folder_map {folder_name: local_Path} for every folder processed.
    """
    folder_map = {}

    # ── Option B: already on disk ─────────────────────────────────────────────
    if LOCAL_DATA_ROOT.exists():
        audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".w64"}
        audio_files = [f for f in LOCAL_DATA_ROOT.rglob("*")
                       if f.suffix.lower() in audio_exts]
        if audio_files:
            print(f"  ✔  Local data found at {LOCAL_DATA_ROOT}")
            print(f"     {len(audio_files)} audio files detected — skipping download")
            for p in LOCAL_DATA_ROOT.iterdir():
                if p.is_dir():
                    folder_map[p.name] = p
            folder_map["_root"] = LOCAL_DATA_ROOT
            return folder_map

    # ── Option A: Nextcloud WebDAV ────────────────────────────────────────────
    if not SHARE_URL:
        print("""
  ┌─ NO DATA SOURCE CONFIGURED ──────────────────────────────────────────┐
  │                                                                       │
  │  Set SHARE_URL at the top of this script to your Nextcloud link.     │
  │  e.g.  SHARE_URL = "https://nc.ek-cer.hu/index.php/s/TOKEN"         │
  │                                                                       │
  │  Running on SYNTHETIC demo data for now.                             │
  └───────────────────────────────────────────────────────────────────────┘
""")
        _build_demo_data(workspace)
        folder_map["_root"] = workspace
        for d in workspace.iterdir():
            if d.is_dir():
                folder_map[d.name] = d
        return folder_map

    # ── Parse share URL ───────────────────────────────────────────────────────
    host_base, token, subpath = _nc_parse_share_url(SHARE_URL)
    webdav_base = _nc_webdav_base(host_base, token)
    session     = requests.Session()
    session.headers["User-Agent"] = "DunakesziAudit/1.0"

    print(f"\n  🔗  Nextcloud host  : {host_base}")
    print(f"  🔑  Share token     : {token}")
    print(f"  📡  WebDAV endpoint : {webdav_base}")
    if subpath:
        print(f"  📂  Sub-path        : /{subpath}")

    # ── List root ─────────────────────────────────────────────────────────────
    print(f"\n  📂  Listing root …")
    try:
        root_items = _nc_propfind(session, webdav_base, token,
                                  remote_path=subpath, depth=1)
    except Exception as e:
        raise RuntimeError(
            f"\n  WebDAV PROPFIND failed: {e}\n"
            f"  URL tried: {webdav_base}\n"
            f"  Token:     {token}\n"
            f"  Check that the share link is public (no password required)."
        ) from e

    print(f"  Found {len(root_items)} top-level items:\n")
    for it in root_items:
        ico = "📁" if it["is_dir"] else "📄"
        print(f"      {ico}  {it['name']:<45}  {it['size_mb']:>8.1f} MB"
              f"  [{it['last_modified'][:16]}]")

    # ── Process each sub-folder ───────────────────────────────────────────────
    for item in root_items:
        fname = item["name"]
        fkey  = _folder_key(fname)
        local = workspace / fname
        local.mkdir(exist_ok=True)
        folder_map[fname] = local

        meta_only = any(k in fkey for k in DOWNLOAD_META_ONLY_FOLDERS)

        if not item["is_dir"]:
            # Top-level file (e.g. README) — download directly
            dest = local / fname
            _nc_download(session, webdav_base, token,
                         item["remote_path"], dest)
            continue

        # List all files inside this sub-folder
        print(f"\n  {'📋 INDEX' if meta_only else '⬇  DOWNLOAD'}  {fname} …")
        try:
            all_files = _nc_collect_files(
                session, webdav_base, token, item["remote_path"], depth=4)
        except Exception as e:
            print(f"      ⚠  Could not list {fname}: {e}")
            continue

        print(f"      {len(all_files)} files found")

        if meta_only:
            # Save an index JSON — no actual file download
            idx = [{"name": f["name"], "size_mb": round(f["size_mb"], 3),
                    "last_modified": f["last_modified"],
                    "remote_path": f["remote_path"]}
                   for f in all_files]
            with open(local / "_index.json", "w") as fh:
                json.dump(idx, fh, indent=2)
            print(f"      Indexed {len(idx)} files → _index.json")
            continue

        # Download with limits
        is_mems      = "MEMS" in fkey
        dl_count     = 0
        dl_mb        = 0.0
        audio_exts   = {".wav", ".flac", ".w64", ".pcm", ".raw"}

        for f in all_files:
            ext     = Path(f["name"]).suffix.lower()
            size_mb = f["size_mb"]

            if is_mems:
                if ext not in audio_exts:
                    continue
                if dl_count >= MAX_MEMS_FILES:
                    print(f"      ⚡ File cap reached ({MAX_MEMS_FILES} files)")
                    break
                if dl_mb >= MAX_MEMS_MB:
                    print(f"      ⚡ Size cap reached ({MAX_MEMS_MB} MB)")
                    break
                # Byte cap: estimate bytes for first TRIM_SECONDS
                # Assumes worst-case 48kHz 16-bit stereo = 192 kB/s
                byte_cap = int(TRIM_SECONDS * 192 * 1024)
                byte_cap = min(byte_cap, int(size_mb * 1024**2))
            else:
                byte_cap = None  # download fully

            # Preserve sub-folder structure locally
            rel_path  = f["remote_path"]
            subfolder = Path(rel_path).parent.name
            dest_dir  = local / subfolder if subfolder and subfolder != fname else local
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f["name"]

            ok = _nc_download(session, webdav_base, token,
                              rel_path, dest, byte_cap=byte_cap)
            if ok:
                actual_mb  = dest.stat().st_size / 1024**2
                dl_count  += 1
                dl_mb     += actual_mb
                print(f"      ✔  {f['name'][:52]:<52}  "
                      f"{actual_mb:>5.1f}/{size_mb:.1f} MB", end="\r")

        print(f"\n      ✔  {dl_count} files  /  {dl_mb:.1f} MB downloaded")

    return folder_map


def _build_demo_data(ws: Path):
    """Realistic synthetic Dunakeszi-style dataset for offline testing."""
    from scipy.io import wavfile as wf
    print("  ⚙  Building synthetic demo dataset …")
    sr = 48000  # MEMS recorders typically record at 48 kHz

    configs = [
        # (sub-folder,          label,       bpf, harm, motor_hz, dist_m, snr_db, n)
        ("MEMS/drone_pass",     "drone",      82,  8,  440, 15, 12, 10),
        ("MEMS/drone_hover",    "drone",      78,  8,  430,  8, 18,  8),
        ("MEMS/background",     "background",  0,  0,    0,  0, 99,  8),
        ("MEMS/wind_noise",     "background",  0,  0,    0,  0, 99,  5),
        ("MEMS/car_passby",     "vehicle",    65,  3,  200, 20,  8,  5),
    ]
    for (sub, lbl, bpf, harm, mhz, dist, snr_db, n) in configs:
        d = ws / sub; d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            dur = float(np.random.uniform(3, 10))
            t   = np.linspace(0, dur, int(sr * dur), endpoint=False)
            sig = np.zeros_like(t)
            if bpf > 0:
                att = 1.0 / max(dist, 1)
                for h in range(1, harm + 1):
                    fm   = 1 + 0.015 * np.sin(2*np.pi*1.2*t + np.random.rand()*2*np.pi)
                    sig += att * np.sin(2*np.pi*bpf*h*fm*t) / (h**1.1)
                for h in range(1, 4):
                    sig += att * 0.25 * np.sin(2*np.pi*mhz*h*t) / h
            noise = np.random.randn(len(t))
            ff    = fft.rfftfreq(len(t), 1/sr); ff[0] = 1e-6
            pink  = np.real(fft.irfft(fft.rfft(noise) / np.sqrt(ff), n=len(t)))
            noise = 0.6*noise + 0.4*pink / (np.std(pink) + 1e-9)
            sp    = float(np.mean(sig**2))
            np_   = sp / (10**(snr_db/10)) if sp > 0 else 1e-4
            sig  += noise * np.sqrt(np_ / (float(np.mean(noise**2)) + 1e-9))
            sig   = sig / (float(np.max(np.abs(sig))) + 1e-9)
            wf.write(str(d / f"{lbl}_{i:03d}_sr{sr}.wav"),
                     sr, (sig * 32767).astype(np.int16))

    # Demo telemetry
    td = ws / "DRON"; td.mkdir(exist_ok=True)
    t0 = datetime(2024, 10, 24, 10, 0, 0)
    rows = []
    lat0, lon0 = 47.6255, 19.1352
    theta = np.linspace(0, 3*np.pi, 180)
    for i, th in enumerate(theta):
        rows.append({
            "timestamp":   (t0 + timedelta(seconds=i*3)).isoformat(),
            "latitude":    lat0 + 0.0008*np.sin(th),
            "longitude":   lon0 + 0.0012*np.cos(th),
            "altitude_m":  35 + 12*np.sin(th/2),
            "speed_ms":    4 + 2*np.cos(th),
            "heading_deg": (np.degrees(np.arctan2(
                np.gradient([0.0008*np.sin(x) for x in theta])[i],
                np.gradient([0.0012*np.cos(x) for x in theta])[i])) + 360) % 360,
            "battery_pct": max(0, 100 - i*0.5),
            "drone_id":    "DJI-MINI3-001",
        })
    pd.DataFrame(rows).to_csv(td / "flight_log_20241024.csv", index=False)

    # Demo logbook
    ld = ws / "JEGY"; ld.mkdir(exist_ok=True)
    (ld / "session_notes.txt").write_text(
        "Dunakeszi field recording session 2024-10-24\n"
        "Location: Dunakeszi sports field, 47.6255N 19.1352E\n"
        "Weather: overcast, wind 2-4 m/s from NW\n"
        "Equipment: MEMS array (4-mic cross, 5cm spacing), Brüel & Kjær\n"
        "10:00 - MEMS array calibration\n"
        "10:15 - DJI Mini 3 Pro takeoff, circular 80m radius path\n"
        "10:45 - hover test at 8m, 15m, 30m altitude\n"
        "11:00 - background noise recording (no drone)\n"
        "11:20 - car pass-by recordings\n"
        "12:00 - session end\n"
    )
    print(f"  ✔  Demo data written to {ws}")


# =============================================================================
#  SECTION 1  ──  FILE-SYSTEM FORENSIC AUDIT
# =============================================================================

AUDIO_EXT = {".wav",".flac",".mp3",".ogg",".w64",".pcm",".raw",".aif",".aiff"}
TEXT_EXT  = {".txt",".log",".md",".csv",".json",".xml",".kml",".gpx",".nmea"}
VIDEO_EXT = {".mp4",".avi",".mov",".mkv",".mts",".m2ts",".wmv"}
IMAGE_EXT = {".jpg",".jpeg",".png",".tiff",".tif",".bmp"}
DATA_EXT  = {".csv",".json",".xml",".xlsx",".xls",".mat",".h5",".hdf5",".npy"}


def audit_filesystem(root: Path) -> pd.DataFrame:
    """Walk every file under root and collect exhaustive metadata."""
    print(f"\n  🔍  Scanning {root} …")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file(): continue
        stat  = p.stat()
        ext   = p.suffix.lower()
        rel   = p.relative_to(root)
        parts = rel.parts
        depth = len(parts) - 1
        rows.append({
            "path":          str(p),
            "rel_path":      str(rel),
            "filename":      p.name,
            "stem":          p.stem,
            "extension":     ext,
            "folder_l1":     parts[0] if len(parts) > 1 else "_root",
            "folder_l2":     parts[1] if len(parts) > 2 else "",
            "depth":         depth,
            "size_bytes":    stat.st_size,
            "size_kb":       stat.st_size / 1024,
            "size_mb":       stat.st_size / 1024**2,
            "mtime":         datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type":     _classify_ext(ext),
            "is_hidden":     p.name.startswith("."),
            "md5_prefix":    _md5_prefix(p),   # first 64 KB hash (fast)
        })
    df = pd.DataFrame(rows)
    print(f"     ✔  {len(df)} files found across "
          f"{df['folder_l1'].nunique()} top-level folders")
    return df


def _classify_ext(ext):
    if ext in AUDIO_EXT:   return "audio"
    if ext in VIDEO_EXT:   return "video"
    if ext in IMAGE_EXT:   return "image"
    if ext in DATA_EXT:    return "data"
    if ext in TEXT_EXT:    return "text"
    return "other"


def _md5_prefix(path: Path, n_bytes=65536) -> str:
    h = hashlib.md5()
    try:
        with open(path,"rb") as f:
            h.update(f.read(n_bytes))
        return h.hexdigest()[:12]
    except: return ""


# =============================================================================
#  SECTION 2  ──  AUDIO FILE FORENSICS
# =============================================================================

def read_wav_info(path: Path) -> dict:
    """Extract header metadata WITHOUT loading the full signal."""
    info = {"path": str(path), "readable": False}
    try:
        with wave.open(str(path), "r") as wf:
            info.update({
                "readable":     True,
                "n_channels":   wf.getnchannels(),
                "sample_rate":  wf.getframerate(),
                "bit_depth":    wf.getsampwidth() * 8,
                "n_frames":     wf.getnframes(),
                "duration_sec": wf.getnframes() / wf.getframerate(),
                "format":       "WAV/PCM",
                "comptype":     wf.getcomptype(),
            })
    except Exception:
        # Try scipy for non-standard WAVs
        try:
            sr, data = wavfile.read(str(path))
            info.update({
                "readable":     True,
                "n_channels":   data.ndim,
                "sample_rate":  sr,
                "bit_depth":    data.dtype.itemsize * 8,
                "n_frames":     data.shape[0],
                "duration_sec": data.shape[0] / sr,
                "format":       f"WAV/{data.dtype}",
                "comptype":     "NONE",
            })
        except Exception as e:
            info["error"] = str(e)
    return info


def read_wav_signal(path: Path, max_seconds: float = 30.0):
    """Load audio signal — returns (float32 array, sample_rate) or (None, None)."""
    try:
        sr, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)           # multi-channel → mono for analysis
        data = data.astype(np.float64)
        # normalise
        if data.dtype != np.float32:
            maxval = {1: 128, 2: 32768, 3: 8388608, 4: 2147483648}.get(
                np.dtype(data.dtype).itemsize, 32768)
            data /= maxval
        # trim
        max_frames = int(max_seconds * sr)
        if len(data) > max_frames:
            data = data[:max_frames]
        return data.astype(np.float32), int(sr)
    except Exception:
        return None, None


def diagnose_audio(data: np.ndarray, sr: int, path: str) -> dict:
    """Compute a full suite of audio health diagnostics."""
    d = {"path": path}
    if data is None or len(data) == 0:
        d["error"] = "unreadable"; return d

    # Basic stats
    d["rms"]           = float(np.sqrt(np.mean(data**2)))
    d["peak"]          = float(np.max(np.abs(data)))
    d["dc_offset"]     = float(np.mean(data))
    d["dynamic_range"] = float(20 * np.log10(d["peak"] + 1e-12))

    # Clipping detection (samples within 1% of max)
    clip_thresh = 0.99
    d["clip_pct"]      = float(np.mean(np.abs(data) > clip_thresh) * 100)
    d["clipped"]       = d["clip_pct"] > 0.1

    # Silence detection (RMS below -60 dB in any 100ms frame)
    frame = int(0.1 * sr)
    n_fr  = len(data) // frame
    if n_fr > 0:
        frames    = data[:n_fr*frame].reshape(n_fr, frame)
        fr_rms    = np.sqrt(np.mean(frames**2, axis=1))
        fr_db     = 20 * np.log10(fr_rms + 1e-12)
        d["silence_pct"]   = float(np.mean(fr_db < -50) * 100)
        d["active_pct"]    = 100 - d["silence_pct"]
        d["rms_db_mean"]   = float(fr_db.mean())
        d["rms_db_std"]    = float(fr_db.std())
        d["rms_db_min"]    = float(fr_db.min())
        d["rms_db_max"]    = float(fr_db.max())
    else:
        d["silence_pct"] = d["active_pct"] = 0.0

    # SNR (quiet 10th percentile = noise; loud 90th = signal)
    if n_fr > 0:
        noise_rms  = float(np.percentile(fr_rms, 10)) + 1e-12
        signal_rms = float(np.percentile(fr_rms, 90)) + 1e-12
        d["snr_db"] = float(20 * np.log10(signal_rms / noise_rms))
        d["noise_floor_db"] = float(20 * np.log10(noise_rms))
    else:
        d["snr_db"] = d["noise_floor_db"] = 0.0

    # Spectral features
    f_psd, psd = signal.welch(data, fs=sr, nperseg=min(4096, len(data)))
    d["spectral_centroid_hz"] = float(
        np.sum(f_psd * psd) / (np.sum(psd) + 1e-10))
    d["spectral_flatness"]    = float(
        np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10))

    # Dominant frequency (highest power peak)
    peaks, props = signal.find_peaks(psd, height=psd.max()*0.05, distance=3)
    if len(peaks):
        top = peaks[np.argmax(props["peak_heights"])]
        d["dominant_freq_hz"] = float(f_psd[top])
        d["dominant_freq_db"] = float(10 * np.log10(props["peak_heights"].max() + 1e-10))
    else:
        d["dominant_freq_hz"] = d["dominant_freq_db"] = 0.0

    # ZCR
    d["zcr"] = float(
        np.mean(np.abs(np.diff(np.sign(data)))) / 2)

    # Kurtosis (impulsiveness)
    d["kurtosis"] = float(stats.kurtosis(data))

    # Crest factor
    d["crest_factor"] = d["peak"] / (d["rms"] + 1e-9)

    # Noise colour (slope of PSD in log-log space)
    idx = (f_psd > 50) & (f_psd < sr/2 * 0.9)
    if idx.sum() > 5:
        slope, *_ = np.polyfit(np.log10(f_psd[idx] + 1e-6),
                               np.log10(psd[idx] + 1e-10), 1)
        d["noise_slope"] = float(slope)
        # -1 = pink, -2 = brownian, 0 = white
        d["noise_colour"] = ("pink" if -1.5 < slope < -0.5
                             else "brownian" if slope < -1.5
                             else "white")
    else:
        d["noise_slope"] = 0.0
        d["noise_colour"] = "unknown"

    return d


def audit_audio_files(fs_df: pd.DataFrame) -> pd.DataFrame:
    """Run full forensics on every audio file."""
    print(f"\n  🎙  Audio forensics …")
    audio_rows = fs_df[fs_df["file_type"] == "audio"].copy()
    if len(audio_rows) == 0:
        print("     ⚠  No audio files found"); return pd.DataFrame()

    header_rows, diag_rows = [], []
    for _, row in audio_rows.iterrows():
        p   = Path(row["path"])
        ext = row["extension"]
        # Header info
        if ext == ".wav":
            hdr = read_wav_info(p)
        else:
            hdr = {"path": str(p), "readable": False,
                   "note": f"Non-WAV ({ext}) — header not parsed"}
        hdr["filename"] = row["filename"]
        hdr["folder"]   = row["folder_l1"]
        header_rows.append(hdr)

        # Signal diagnostics (WAV only for now)
        if ext == ".wav":
            data, sr = read_wav_signal(p)
            diag = diagnose_audio(data, sr, str(p))
            diag["filename"] = row["filename"]
            diag["folder"]   = row["folder_l1"]
            diag["sr"]       = sr
            diag_rows.append(diag)

    hdr_df  = pd.DataFrame(header_rows)
    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame()

    print(f"     ✔  {len(hdr_df)} audio files  |  "
          f"{len(diag_df)} WAV files fully diagnosed")
    if len(diag_df) and "sample_rate" in hdr_df.columns:
        srs = hdr_df["sample_rate"].value_counts()
        print(f"     Sample rates found: {dict(srs)}")
        if len(srs) > 1:
            print("     ⚠  MIXED SAMPLE RATES — will need resampling before training!")

    return hdr_df, diag_df


# =============================================================================
#  SECTION 3  ──  TELEMETRY STRUCTURE AUDIT
# =============================================================================

def audit_telemetry(folder_map: dict) -> dict:
    """Parse every data file in DRON folder; return summary dict."""
    print(f"\n  🛸  Telemetry audit …")
    result = {"files": [], "combined_df": None, "issues": []}

    dron_dir = None
    for k, v in folder_map.items():
        if "DRON" in k.upper() or "ADATOK" in k.upper():
            dron_dir = Path(v); break

    if dron_dir is None or not dron_dir.exists():
        result["issues"].append("No DRON-ADATOK folder found")
        return result

    dfs = []
    for f in sorted(dron_dir.rglob("*")):
        if not f.is_file(): continue
        ext  = f.suffix.lower()
        info = {"name": f.name, "size_kb": f.stat().st_size/1024, "ext": ext,
                "columns": [], "n_rows": 0, "issues": []}
        try:
            if ext == ".csv":
                df = pd.read_csv(f, on_bad_lines="skip", nrows=5000)
                info["columns"] = list(df.columns)
                info["n_rows"]  = len(df)
                info["dtypes"]  = {c: str(df[c].dtype) for c in df.columns}
                info["nulls"]   = df.isnull().sum().to_dict()
                info["sample"]  = df.head(3).to_dict()
                # Try to identify GPS columns
                for col in df.columns:
                    cl = col.lower()
                    if any(x in cl for x in ["lat","gps_lat","latitude"]):
                        info["lat_col"] = col
                    if any(x in cl for x in ["lon","lng","longitude"]):
                        info["lon_col"] = col
                    if any(x in cl for x in ["alt","altitude","height"]):
                        info["alt_col"] = col
                    if any(x in cl for x in ["time","ts","timestamp","date"]):
                        info["time_col"] = col
                dfs.append(df)

            elif ext == ".json":
                raw  = json.loads(f.read_text(errors="replace"))
                flat = pd.json_normalize(raw if isinstance(raw, list) else [raw])
                info["columns"] = list(flat.columns)
                info["n_rows"]  = len(flat)
                dfs.append(flat)

            elif ext == ".kml":
                coords = re.findall(
                    r"<coordinates>(.*?)</coordinates>", f.read_text(), re.DOTALL)
                n = sum(len(c.strip().split()) for c in coords)
                info["n_coords"] = n
                info["note"]     = "KML trajectory"

            elif ext == ".gpx":
                pts = re.findall(r'<trkpt lat="([\d.\-]+)" lon="([\d.\-]+)"',
                                 f.read_text())
                info["n_coords"] = len(pts)
                info["note"]     = "GPX trajectory"

            elif ext in {".txt", ".log"}:
                lines = f.read_text(errors="replace").splitlines()
                info["n_lines"] = len(lines)
                info["preview"] = lines[:5]

        except Exception as e:
            info["issues"].append(str(e))

        result["files"].append(info)
        print(f"     📄  {f.name:<45}  {f.stat().st_size/1024:>8.1f} KB  "
              f"cols={len(info.get('columns',[]))}")

    if dfs:
        result["combined_df"] = pd.concat(dfs, ignore_index=True)
        print(f"     ✔  Combined telemetry: "
              f"{len(result['combined_df'])} rows × "
              f"{len(result['combined_df'].columns)} columns")

    return result


# =============================================================================
#  SECTION 4  ──  LOGBOOK / NOTES AUDIT
# =============================================================================

def audit_logbook(folder_map: dict) -> dict:
    """Read all text files in JEGY folder."""
    print(f"\n  📋  Logbook audit …")
    result = {"entries": [], "raw_text": ""}

    jegy_dir = None
    for k, v in folder_map.items():
        if "JEGY" in k.upper() or "EGYZOKONYV" in k.upper():
            jegy_dir = Path(v); break

    if jegy_dir is None or not jegy_dir.exists():
        result["issues"] = "No logbook folder found"
        return result

    full_text = []
    for f in sorted(jegy_dir.rglob("*")):
        if not f.is_file(): continue
        try:
            text = f.read_text(errors="replace")
            full_text.append(f"=== {f.name} ===\n{text}")
            # Heuristic: extract time-stamped lines
            for line in text.splitlines():
                m = re.search(
                    r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)", line)
                if m:
                    result["entries"].append({
                        "source":    f.name,
                        "timestamp": m.group(1),
                        "text":      line.strip(),
                    })
            print(f"     📄  {f.name:<45}  {f.stat().st_size/1024:>6.1f} KB")
            print(f"         Preview: {text[:200].replace(chr(10),' ')[:120]} …")
        except Exception as e:
            print(f"     ⚠  {f.name}: {e}")

    result["raw_text"] = "\n\n".join(full_text)
    print(f"     ✔  {len(result['entries'])} timestamped entries extracted")
    return result


# =============================================================================
#  SECTION 5  ──  CROSS-FILE CONSISTENCY ANALYSIS
# =============================================================================

def consistency_analysis(hdr_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    """Check for sample rate mismatches, duration spread, possible duplicates."""
    issues = []

    if hdr_df.empty: return {"issues": ["No audio headers to analyse"]}

    report = {}

    # Sample rate distribution
    if "sample_rate" in hdr_df.columns:
        sr_counts = hdr_df["sample_rate"].value_counts().to_dict()
        report["sample_rates"] = sr_counts
        if len(sr_counts) > 1:
            issues.append(f"MIXED SAMPLE RATES: {sr_counts} — normalise before training")

    # Bit depth
    if "bit_depth" in hdr_df.columns:
        bd_counts = hdr_df["bit_depth"].value_counts().to_dict()
        report["bit_depths"] = bd_counts
        if len(bd_counts) > 1:
            issues.append(f"Mixed bit depths: {bd_counts}")

    # Channel count
    if "n_channels" in hdr_df.columns:
        ch_counts = hdr_df["n_channels"].value_counts().to_dict()
        report["channel_counts"] = ch_counts
        if 1 in ch_counts and max(ch_counts.keys()) > 1:
            issues.append("Mixed mono/multi-channel files — check MEMS array layout")

    # Duration spread
    if "duration_sec" in hdr_df.columns:
        dur = hdr_df["duration_sec"].dropna()
        report["duration_stats"] = {
            "min_sec":  float(dur.min()),
            "max_sec":  float(dur.max()),
            "mean_sec": float(dur.mean()),
            "std_sec":  float(dur.std()),
        }
        if dur.std() > dur.mean() * 0.5:
            issues.append("High duration variance — some files may be truncated/padded")

    # Duplicate detection (MD5 prefix)
    if "md5_prefix" in hdr_df.columns:
        dup = hdr_df[hdr_df.duplicated("md5_prefix", keep=False)]
        if len(dup) > 0:
            issues.append(f"{len(dup)} potentially duplicate files detected")
            report["duplicates"] = dup[["filename","md5_prefix","size_kb"]].to_dict()

    # Clipping
    if not diag_df.empty and "clipped" in diag_df.columns:
        n_clipped = int(diag_df["clipped"].sum())
        if n_clipped:
            issues.append(f"{n_clipped} files have clipping (amplitude > 99%)")
        report["clipped_files"] = n_clipped

    # High silence ratio
    if not diag_df.empty and "silence_pct" in diag_df.columns:
        mostly_silent = diag_df[diag_df["silence_pct"] > 70]
        if len(mostly_silent):
            issues.append(f"{len(mostly_silent)} files are >70% silence")

    report["issues"]   = issues
    report["n_issues"] = len(issues)

    return report


# =============================================================================
#  SECTION 6  ──  TEMPORAL ALIGNMENT ANALYSIS
# =============================================================================

def temporal_alignment(diag_df: pd.DataFrame, tele_result: dict,
                       log_result: dict) -> dict:
    """Check whether audio filenames/mtimes align with telemetry timestamps."""
    report = {"aligned": [], "unmatched_audio": [], "unmatched_tele": []}

    if diag_df.empty:
        return report

    # Parse timestamps from audio filenames (common patterns)
    ts_pattern = re.compile(
        r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})"    # date
        r"[-_T ]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})?")  # time

    audio_times = {}
    for _, row in diag_df.iterrows():
        m = ts_pattern.search(row.get("filename",""))
        if m:
            try:
                yr,mo,dy = int(m.group(1)),int(m.group(2)),int(m.group(3))
                hr,mn    = int(m.group(4)),int(m.group(5))
                sc       = int(m.group(6)) if m.group(6) else 0
                audio_times[row["filename"]] = datetime(yr,mo,dy,hr,mn,sc)
            except: pass

    # Telemetry time range
    tele_range = None
    if tele_result.get("combined_df") is not None:
        df = tele_result["combined_df"]
        for col in df.columns:
            if any(x in col.lower() for x in ["time","ts","timestamp","date"]):
                try:
                    t = pd.to_datetime(df[col], errors="coerce").dropna()
                    if len(t) > 0:
                        tele_range = (t.min(), t.max())
                        break
                except: pass

    report["audio_timestamps_found"] = len(audio_times)
    report["tele_range"]             = str(tele_range) if tele_range else "unknown"
    report["log_entries"]            = len(log_result.get("entries", []))

    # Check overlap
    if audio_times and tele_range:
        t0, t1 = tele_range
        for fn, at in audio_times.items():
            at_utc = pd.Timestamp(at)
            if t0 <= at_utc <= t1:
                report["aligned"].append(fn)
            else:
                report["unmatched_audio"].append(fn)

    return report


# =============================================================================
#  SECTION 7  ──  20 PLOTS
# =============================================================================

def _save(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   📊  {name}")


def plot_01_filesystem_overview(fs_df):
    """Total file counts and sizes by type and top-level folder."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle("File System Overview — Dunakeszi Dataset", fontsize=16,
                 color="white", fontweight="bold")

    # File type counts
    ax = axes[0,0]
    tc = fs_df["file_type"].value_counts()
    ax.bar(tc.index, tc.values, color=C[:len(tc)], edgecolor="white", lw=0.5)
    ax.set_title("Files by Type", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    [ax.text(i, v+0.2, str(v), ha="center", color="white", fontsize=9)
     for i,v in enumerate(tc.values)]

    # Size by type
    ax = axes[0,1]
    sz = fs_df.groupby("file_type")["size_mb"].sum().sort_values(ascending=False)
    ax.bar(sz.index, sz.values, color=C[:len(sz)], edgecolor="white", lw=0.5)
    ax.set_title("Total Size by File Type (MB)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_ylabel("MB", color="white")

    # Folder breakdown
    ax = axes[0,2]
    fc = fs_df.groupby("folder_l1")["size_mb"].sum().sort_values(ascending=True)
    bars = ax.barh(fc.index, fc.values, color=C[:len(fc)], edgecolor="white", lw=0.5)
    ax.set_title("Size by Top-Level Folder (MB)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_xlabel("MB", color="white")

    # File count per folder
    ax = axes[1,0]
    fcount = fs_df.groupby("folder_l1").size().sort_values(ascending=False)
    ax.bar(fcount.index, fcount.values, color=C[:len(fcount)],
           edgecolor="white", lw=0.5)
    ax.set_title("File Count per Folder", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", axis="y")
    ax.tick_params(colors="white", axis="x", rotation=30, labelsize=8)

    # File size distribution (log scale)
    ax = axes[1,1]
    for i, (ft, grp) in enumerate(fs_df.groupby("file_type")):
        vals = grp["size_mb"].values
        if len(vals): ax.hist(vals, bins=30, alpha=0.65,
                              color=C[i%len(C)], label=ft, edgecolor="black")
    ax.set_xscale("log"); ax.set_title("File Size Distribution (log MB)", color="white")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Extension pie
    ax = axes[1,2]
    ext_c = fs_df["extension"].value_counts().head(10)
    wedges, texts, atexts = ax.pie(
        ext_c.values, labels=ext_c.index,
        colors=C[:len(ext_c)], autopct="%1.1f%%",
        textprops={"color":"white","fontsize":8},
        pctdistance=0.8, startangle=90)
    ax.set_title("Top-10 Extensions", color="white")
    ax.set_facecolor("#0d0d0d")

    plt.tight_layout()
    _save(fig, "01_filesystem_overview.png")


def plot_02_folder_tree(fs_df):
    """Sunburst-style nested bar showing folder → subfolder structure."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0d0d0d")
    fig.suptitle("Folder Structure & Depth", fontsize=14, color="white")

    # Files per folder/subfolder heatmap
    ax = axes[0]
    pivot = fs_df.groupby(["folder_l1","folder_l2"]).size().unstack(fill_value=0)
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.5,
                    linecolor="#333", annot=True, fmt="d", cbar_kws={"label":"File count"})
        ax.set_title("Files per Folder → Subfolder", color="white", fontsize=11)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_facecolor("#1a1a1a")

    # Depth distribution
    ax = axes[1]
    for i, (ft, grp) in enumerate(fs_df.groupby("file_type")):
        ax.hist(grp["depth"], bins=range(0, grp["depth"].max()+2),
                alpha=0.7, color=C[i%len(C)], label=ft, edgecolor="black",
                align="left")
    ax.set_title("File Depth Distribution", color="white")
    ax.set_xlabel("Directory Depth", color="white")
    ax.set_ylabel("File Count", color="white")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_xticks(range(0, int(fs_df["depth"].max())+1))

    plt.tight_layout()
    _save(fig, "02_folder_tree.png")


def plot_03_audio_header_stats(hdr_df):
    """Sample rate, bit depth, channel, duration distributions."""
    if hdr_df.empty: return
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle("Audio File Header Statistics", fontsize=15, color="white")

    # Sample rate
    ax = axes[0,0]
    if "sample_rate" in hdr_df.columns:
        sr_v = hdr_df["sample_rate"].dropna().value_counts()
        ax.bar([str(int(x)) for x in sr_v.index], sr_v.values,
               color=C[:len(sr_v)], edgecolor="white", lw=0.5)
        ax.set_title("Sample Rate Distribution (Hz)", color="white")
        ax.set_xlabel("Sample Rate (Hz)", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        [ax.text(i, v+0.1, str(v), ha="center", color="white", fontsize=9)
         for i,v in enumerate(sr_v.values)]

    # Bit depth
    ax = axes[0,1]
    if "bit_depth" in hdr_df.columns:
        bd_v = hdr_df["bit_depth"].dropna().value_counts()
        ax.bar([str(int(x)) for x in bd_v.index], bd_v.values,
               color=C[:len(bd_v)], edgecolor="white", lw=0.5)
        ax.set_title("Bit Depth Distribution", color="white")
        ax.set_xlabel("Bit Depth", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Channel count
    ax = axes[0,2]
    if "n_channels" in hdr_df.columns:
        ch_v = hdr_df["n_channels"].dropna().value_counts()
        ax.bar([str(int(x)) for x in ch_v.index], ch_v.values,
               color=C[:len(ch_v)], edgecolor="white", lw=0.5)
        ax.set_title("Channel Count Distribution", color="white")
        ax.set_xlabel("Channels", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Duration histogram
    ax = axes[1,0]
    if "duration_sec" in hdr_df.columns:
        dur = hdr_df["duration_sec"].dropna()
        ax.hist(dur, bins=40, color=C[0], edgecolor="black", alpha=0.85)
        ax.axvline(dur.mean(), color=C[1], lw=2, linestyle="--",
                   label=f"Mean: {dur.mean():.1f}s")
        ax.axvline(dur.median(), color=C[2], lw=2, linestyle=":",
                   label=f"Median: {dur.median():.1f}s")
        ax.set_title("Recording Duration Distribution", color="white")
        ax.set_xlabel("Duration (s)", color="white")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Duration vs file size scatter
    ax = axes[1,1]
    if "duration_sec" in hdr_df.columns and "size_mb" in hdr_df.columns:
        merged = hdr_df.dropna(subset=["duration_sec","size_mb"])
        if "folder" in merged.columns:
            for i, (fld, grp) in enumerate(merged.groupby("folder")):
                ax.scatter(grp["duration_sec"], grp["size_mb"],
                           color=C[i%len(C)], alpha=0.75, s=60,
                           label=fld, edgecolors="white", lw=0.3)
        else:
            ax.scatter(merged["duration_sec"], merged["size_mb"],
                       color=C[0], alpha=0.75, s=60)
        ax.set_xlabel("Duration (s)", color="white")
        ax.set_ylabel("File Size (MB)", color="white")
        ax.set_title("Duration vs File Size", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Compression types
    ax = axes[1,2]
    if "comptype" in hdr_df.columns:
        ct = hdr_df["comptype"].value_counts()
        ax.bar(ct.index, ct.values, color=C[:len(ct)], edgecolor="white", lw=0.5)
        ax.set_title("Compression / Codec", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout()
    _save(fig, "03_audio_header_stats.png")


def plot_04_signal_health(diag_df):
    """Clipping, silence, DC offset, RMS overview across all files."""
    if diag_df.empty: return
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor="#0d0d0d")
    fig.suptitle("Signal Health Dashboard", fontsize=15, color="white")

    def _hist(ax, col, title, unit="", log=False):
        if col not in diag_df.columns: return
        vals = diag_df[col].dropna()
        folders = diag_df.get("folder", pd.Series(["all"]*len(diag_df)))
        for i, (fld, idx) in enumerate(diag_df.groupby(
                folders if "folder" in diag_df.columns else
                pd.Series(["all"]*len(diag_df))).groups.items()):
            ax.hist(diag_df.loc[idx, col].dropna(), bins=25, alpha=0.7,
                    color=C[i%len(C)], label=str(fld), edgecolor="black")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel(f"{unit}", color="white"); ax.set_ylabel("Count", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        if log and vals.min() > 0: ax.set_xscale("log")

    _hist(axes[0,0], "snr_db",        "SNR per File (dB)",         "dB")
    _hist(axes[0,1], "rms_db_mean",   "Mean RMS Level (dB)",       "dBFS")
    _hist(axes[0,2], "noise_floor_db","Noise Floor (dB)",          "dBFS")
    _hist(axes[1,0], "clip_pct",      "Clipping % of Samples",     "%")
    _hist(axes[1,1], "silence_pct",   "Silence % (< -50 dB)",     "%")
    _hist(axes[1,2], "dc_offset",     "DC Offset",                 "amplitude")

    plt.tight_layout()
    _save(fig, "04_signal_health.png")


def plot_05_snr_deep(diag_df):
    """SNR vs noise floor vs dominant frequency — 2D scatter matrix."""
    if diag_df.empty: return
    key_cols = ["snr_db","noise_floor_db","dominant_freq_hz",
                "spectral_centroid_hz","crest_factor","kurtosis"]
    key_cols = [c for c in key_cols if c in diag_df.columns]
    n = len(key_cols)
    if n < 2: return

    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n), facecolor="#0d0d0d")
    fig.suptitle("Signal Quality — Pairwise Feature Matrix", fontsize=14,
                 color="white", y=1.01)

    folders = (diag_df["folder"].values
               if "folder" in diag_df.columns
               else np.array(["all"]*len(diag_df)))
    uniq = sorted(set(folders))

    for i, ci in enumerate(key_cols):
        for j, cj in enumerate(key_cols):
            ax = axes[i,j]
            ax.set_facecolor("#1a1a1a")
            if i == j:
                for k, fld in enumerate(uniq):
                    mask = folders == fld
                    ax.hist(diag_df.loc[mask, ci].dropna(), bins=20,
                            color=C[k%len(C)], alpha=0.75, edgecolor="black")
                ax.set_ylabel(ci.replace("_"," "), color="white", fontsize=7)
            else:
                for k, fld in enumerate(uniq):
                    mask = folders == fld
                    xi   = diag_df.loc[mask, cj].values
                    yi   = diag_df.loc[mask, ci].values
                    valid = np.isfinite(xi) & np.isfinite(yi)
                    ax.scatter(xi[valid], yi[valid], s=30, alpha=0.7,
                               color=C[k%len(C)], edgecolors="white", lw=0.2)
            ax.tick_params(colors="white", labelsize=5)
            if i == n-1: ax.set_xlabel(cj.replace("_"," "), color="white", fontsize=7)

    # Legend
    patches = [plt.matplotlib.patches.Patch(color=C[k%len(C)], label=fld)
               for k,fld in enumerate(uniq)]
    fig.legend(handles=patches, loc="lower right",
               facecolor="#1a1a1a", labelcolor="white", fontsize=8,
               bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    _save(fig, "05_signal_quality_matrix.png")


def plot_06_noise_colour(diag_df):
    """Noise colour classification and PSD slope analysis."""
    if diag_df.empty or "noise_colour" not in diag_df.columns: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d0d0d")
    fig.suptitle("Noise Floor Characterisation", fontsize=14, color="white")

    # Colour pie
    ax = axes[0]
    nc = diag_df["noise_colour"].value_counts()
    colour_map = {"pink": "#FF69B4", "white": "#FFFFFF",
                  "brownian": "#8B4513", "unknown": "#888888"}
    wedge_cols = [colour_map.get(x, C[i]) for i,x in enumerate(nc.index)]
    ax.pie(nc.values, labels=nc.index, colors=wedge_cols, autopct="%1.0f%%",
           textprops={"color":"white"}, startangle=90)
    ax.set_title("Noise Colour Classification", color="white")
    ax.set_facecolor("#0d0d0d")

    # PSD slope distribution
    ax = axes[1]
    if "noise_slope" in diag_df.columns:
        ax.hist(diag_df["noise_slope"].dropna(), bins=25,
                color=C[0], edgecolor="black", alpha=0.85)
        ax.axvline(-1, color=C[1], lw=2, ls="--", label="Pink noise (-1)")
        ax.axvline(-2, color=C[2], lw=2, ls="--", label="Brownian (-2)")
        ax.axvline( 0, color=C[3], lw=2, ls="--", label="White noise (0)")
        ax.set_title("PSD Slope Distribution", color="white")
        ax.set_xlabel("Log-log slope", color="white")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Noise floor vs SNR coloured by folder
    ax = axes[2]
    if "noise_floor_db" in diag_df.columns and "snr_db" in diag_df.columns:
        folders = diag_df.get("folder", pd.Series(["all"]*len(diag_df)))
        for i, (fld, grp) in enumerate(diag_df.groupby(folders)):
            ax.scatter(grp["noise_floor_db"], grp["snr_db"],
                       color=C[i%len(C)], label=str(fld),
                       alpha=0.8, s=70, edgecolors="white", lw=0.3)
        ax.set_xlabel("Noise Floor (dBFS)", color="white")
        ax.set_ylabel("SNR (dB)", color="white")
        ax.set_title("Noise Floor vs SNR", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        ax.grid(True, alpha=0.1)

    plt.tight_layout()
    _save(fig, "06_noise_floor.png")


def plot_07_sample_waveforms(diag_df, n_each=3):
    """Show sample waveforms from each folder."""
    if diag_df.empty: return
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]
    rows    = []
    for fld in folders:
        sub = diag_df[diag_df["folder"]==fld] if "folder" in diag_df.columns else diag_df
        rows += sub.head(n_each).to_dict("records")

    n = len(rows)
    if n == 0: return
    fig, axes = plt.subplots(n, 1, figsize=(18, 2.8*n), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("Sample Waveforms per Folder", fontsize=14, color="white")

    for ax, row in zip(axes, rows):
        data, sr = read_wav_signal(Path(row["path"]), max_seconds=10)
        if data is None: continue
        t   = np.linspace(0, len(data)/sr, len(data))
        fld = row.get("folder","?")
        col = C[list(folders).index(fld) % len(C)]
        ax.plot(t, data, color=col, lw=0.3, alpha=0.85)
        ax.fill_between(t, data, alpha=0.1, color=col)
        # Mark clipping
        clip_thresh = 0.99
        clips = np.where(np.abs(data) > clip_thresh)[0]
        if len(clips):
            ax.scatter(t[clips], data[clips], color="red", s=5,
                       zorder=5, alpha=0.7)
        snr   = row.get("snr_db", "?")
        noise = row.get("noise_floor_db","?")
        title = (f"[{fld}]  {Path(row['path']).name[:60]}"
                 f"  |  SNR={snr:.1f}dB  noise={noise:.1f}dBFS"
                 if isinstance(snr, float) else
                 f"[{fld}]  {Path(row['path']).name[:60]}")
        ax.set_title(title, color="white", fontsize=8)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel("Amp", color="white", fontsize=7)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white", labelsize=7)
        ax.axhline(0, color="white", lw=0.3, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", color="white")
    plt.tight_layout()
    _save(fig, "07_sample_waveforms.png")


def plot_08_psd_gallery(diag_df, n_each=3):
    """Average PSD per folder — key for understanding drone frequency bands."""
    if diag_df.empty: return
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#0d0d0d")
    fig.suptitle("Average Power Spectral Density per Folder", fontsize=14,
                 color="white")
    ax.set_facecolor("#1a1a1a")

    for i, fld in enumerate(folders):
        sub = (diag_df[diag_df["folder"]==fld]
               if "folder" in diag_df.columns else diag_df)
        psds, sr_ref = [], None
        for _, row in sub.head(n_each).iterrows():
            data, sr = read_wav_signal(Path(row["path"]))
            if data is None: continue
            if sr_ref is None: sr_ref = sr
            if sr != sr_ref:
                n_out = int(len(data) * sr_ref / sr)
                data  = signal.resample(data, n_out)
            f, psd = signal.welch(data, fs=sr_ref, nperseg=N_FFT)
            psds.append(psd)
        if not psds: continue
        mean_psd = np.mean(psds, axis=0)
        std_psd  = np.std(psds, axis=0)
        ax.semilogy(f/1000, mean_psd, color=C[i%len(C)], lw=2, label=str(fld))
        ax.fill_between(f/1000,
                        np.maximum(mean_psd - std_psd, 1e-12),
                        mean_psd + std_psd,
                        alpha=0.15, color=C[i%len(C)])

    ax.set_xlabel("Frequency (kHz)", color="white", fontsize=11)
    ax.set_ylabel("Power Spectral Density", color="white", fontsize=11)
    ax.legend(fontsize=9, facecolor="#1a1a1a", labelcolor="white")
    ax.tick_params(colors="white"); ax.grid(True, alpha=0.12, which="both")
    # Annotate likely drone bands
    for f_hz, lbl in [(80,"BPF ~80Hz"), (160,"2nd harm"), (420,"Motor")]:
        ax.axvline(f_hz/1000, color="white", lw=0.8, ls=":", alpha=0.5)
        ax.text(f_hz/1000, ax.get_ylim()[1]*0.5, lbl,
                color="white", fontsize=7, rotation=90, va="top")
    plt.tight_layout()
    _save(fig, "08_psd_gallery.png")


def plot_09_spectrograms(diag_df):
    """STFT spectrogram for one file from each folder."""
    if diag_df.empty: return
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]
    n = len(folders)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("STFT Spectrograms — One Sample per Folder",
                 fontsize=13, color="white")

    for ax, fld in zip(axes, folders):
        sub = (diag_df[diag_df["folder"]==fld].iloc[0]
               if "folder" in diag_df.columns else diag_df.iloc[0])
        data, sr = read_wav_signal(Path(sub["path"]), max_seconds=15)
        if data is None: ax.text(0.5,0.5,"N/A",ha="center",color="white"); continue
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=N_FFT,
                                 noverlap=N_FFT-HOP)
        db = 10*np.log10(np.abs(Zxx)**2 + 1e-10)
        im = ax.pcolormesh(t, f/1000, db, shading="auto",
                           cmap="inferno", vmin=np.percentile(db,5), vmax=db.max())
        ax.set_title(f"[{fld}]\n{Path(sub['path']).name[:30]}",
                     color="white", fontsize=8)
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Freq (kHz)", color="white")
        ax.set_ylim(0, min(sr/2000, 11))  # cap at 11 kHz for clarity
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    _save(fig, "09_spectrograms.png")


def plot_10_mel_spectrograms(diag_df):
    """Mel spectrogram for one file from each folder."""
    if diag_df.empty: return
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]
    n = len(folders)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("Mel-Spectrograms — One Sample per Folder",
                 fontsize=13, color="white")

    from scipy.io import wavfile as wf2
    sr_ref = SR_TARGET
    fb = None

    for ax, fld in zip(axes, folders):
        sub  = (diag_df[diag_df["folder"]==fld].iloc[0]
                if "folder" in diag_df.columns else diag_df.iloc[0])
        data, sr = read_wav_signal(Path(sub["path"]), max_seconds=15)
        if data is None: ax.text(0.5,0.5,"N/A",ha="center",color="white"); continue
        if sr != sr_ref:
            data = signal.resample(data, int(len(data)*sr_ref/sr)).astype(np.float32)
            sr   = sr_ref
        # Build filterbank once
        if fb is None:
            fb = _mel_filterbank(sr, N_FFT, N_MELS)
        _, _, Zxx = signal.stft(data, fs=sr, nperseg=N_FFT, noverlap=N_FFT-HOP)
        mel  = fb @ np.abs(Zxx)
        mel_db = 10*np.log10(np.maximum(mel, 1e-10))
        im = ax.pcolormesh(np.arange(mel_db.shape[1]),
                           np.arange(N_MELS), mel_db,
                           shading="auto", cmap="magma")
        ax.set_title(f"[{fld}]\n{Path(sub['path']).name[:30]}",
                     color="white", fontsize=8)
        ax.set_xlabel("Frame", color="white"); ax.set_ylabel("Mel Band", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax, label="dB")

    plt.tight_layout()
    _save(fig, "10_mel_spectrograms.png")


def _mel_filterbank(sr, n_fft, n_mels):
    low_hz, high_hz = 20.0, sr/2.0
    lm = 2595*np.log10(1+low_hz/700); hm = 2595*np.log10(1+high_hz/700)
    pts = 700*(10**(np.linspace(lm,hm,n_mels+2)/2595)-1)
    bins = np.floor((n_fft+1)*pts/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1))
    for m in range(1,n_mels+1):
        for k in range(bins[m-1],bins[m]):
            if bins[m]!=bins[m-1]: fb[m-1,k]=(k-bins[m-1])/(bins[m]-bins[m-1])
        for k in range(bins[m],bins[m+1]):
            if bins[m+1]!=bins[m]: fb[m-1,k]=(bins[m+1]-k)/(bins[m+1]-bins[m])
    return fb


def plot_11_dominant_frequencies(diag_df):
    """Where are the dominant spectral peaks across all files?"""
    if diag_df.empty or "dominant_freq_hz" not in diag_df.columns: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d0d0d")
    fig.suptitle("Dominant Frequency Analysis", fontsize=14, color="white")

    # Distribution
    ax = axes[0]
    folders = diag_df["folder"].unique() if "folder" in diag_df.columns else ["all"]
    for i, fld in enumerate(folders):
        sub = (diag_df[diag_df["folder"]==fld]
               if "folder" in diag_df.columns else diag_df)
        vals = sub["dominant_freq_hz"].dropna()
        ax.hist(vals, bins=50, alpha=0.7, color=C[i%len(C)],
                label=str(fld), edgecolor="black")
    ax.set_title("Dominant Frequency Distribution (Hz)", color="white")
    ax.set_xlabel("Frequency (Hz)", color="white")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    # Mark known drone bands
    for f_hz, lbl in [(80,"BPF"),(160,"H2"),(240,"H3"),(420,"Motor")]:
        ax.axvline(f_hz, color="white", lw=1, ls="--", alpha=0.6)
        ax.text(f_hz+2, ax.get_ylim()[1]*0.85, lbl,
                color="white", fontsize=7)

    # Scatter: dominant freq vs SNR
    ax = axes[1]
    for i, fld in enumerate(folders):
        sub = (diag_df[diag_df["folder"]==fld]
               if "folder" in diag_df.columns else diag_df)
        ax.scatter(sub["dominant_freq_hz"], sub["snr_db"],
                   color=C[i%len(C)], label=str(fld),
                   alpha=0.75, s=70, edgecolors="white", lw=0.3)
    ax.set_xlabel("Dominant Frequency (Hz)", color="white")
    ax.set_ylabel("SNR (dB)", color="white")
    ax.set_title("Dominant Frequency vs SNR", color="white")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    _save(fig, "11_dominant_frequencies.png")


def plot_12_duration_timeline(hdr_df):
    """Timeline of recordings sorted by modification time."""
    if hdr_df.empty or "mtime" not in hdr_df.columns: return
    df = hdr_df.copy()
    df["mtime_dt"] = pd.to_datetime(df["mtime"], errors="coerce")
    df = df.dropna(subset=["mtime_dt","duration_sec"]).sort_values("mtime_dt")
    if len(df) == 0: return

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), facecolor="#0d0d0d")
    fig.suptitle("Recording Timeline", fontsize=14, color="white")

    # Timeline bars
    ax = axes[0]
    folders = df["folder"].unique() if "folder" in df.columns else ["all"]
    for i, fld in enumerate(folders):
        sub = df[df["folder"]==fld] if "folder" in df.columns else df
        ax.barh(np.arange(len(sub)), sub["duration_sec"].values,
                left=[x.timestamp() for x in sub["mtime_dt"]],
                height=0.6, color=C[i%len(C)], alpha=0.75,
                label=str(fld), edgecolor="black", lw=0.3)
    ax.set_title("Files by Modification Time & Duration", color="white")
    ax.set_xlabel("Unix Timestamp", color="white")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Gap analysis
    ax = axes[1]
    df_sorted = df.sort_values("mtime_dt")
    gaps = df_sorted["mtime_dt"].diff().dt.total_seconds().dropna()
    ax.hist(np.log10(gaps[gaps > 0] + 1), bins=40,
            color=C[0], edgecolor="black", alpha=0.85)
    ax.set_title("Time Gaps Between Files (log10 seconds)", color="white")
    ax.set_xlabel("log10(gap in seconds)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    for xs, lbl in [(0,"1s"),(1,"10s"),(2,"1min"),(3,"~17min"),(4,"~3hr")]:
        ax.axvline(xs, color="white", lw=0.7, ls=":", alpha=0.5)
        ax.text(xs+0.03, ax.get_ylim()[1]*0.85, lbl, color="white", fontsize=7)

    plt.tight_layout()
    _save(fig, "12_recording_timeline.png")


def plot_13_channel_comparison(diag_df):
    """Compare multi-channel files — are channels balanced?"""
    if diag_df.empty: return
    multi_ch = (diag_df[diag_df.get("n_channels", pd.Series(dtype=int)) > 1]
                if "n_channels" in diag_df.columns else pd.DataFrame())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d0d0d")
    fig.suptitle("Multi-Channel MEMS Array Analysis", fontsize=14, color="white")

    ax = axes[0]
    if len(multi_ch) > 0:
        # Load one multi-channel file and compare channels
        path = Path(multi_ch.iloc[0]["path"])
        try:
            sr, data = wavfile.read(str(path))
            if data.ndim > 1:
                for ch in range(min(data.shape[1], 8)):
                    ch_data = data[:, ch].astype(np.float32)
                    ch_data /= (np.max(np.abs(ch_data)) + 1e-9)
                    f_p, psd = signal.welch(ch_data, fs=sr, nperseg=N_FFT)
                    ax.semilogy(f_p/1000, psd, color=C[ch%len(C)],
                                lw=1.5, alpha=0.85, label=f"Ch {ch}")
                ax.set_title(f"Per-Channel PSD\n{path.name[:40]}",
                             color="white", fontsize=9)
                ax.set_xlabel("Frequency (kHz)", color="white")
                ax.set_ylabel("PSD", color="white")
                ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
                ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
                ax.grid(True, alpha=0.1)
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not load\nmulti-ch file:\n{e}",
                    ha="center", va="center", color="white", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No multi-channel\nfiles found",
                ha="center", va="center", color="white", transform=ax.transAxes,
                fontsize=12)
    ax.set_facecolor("#1a1a1a")

    # Channel count breakdown
    ax = axes[1]
    if "n_channels" in diag_df.columns:
        ch_counts = diag_df["n_channels"].value_counts().sort_index()
        ax.bar([str(int(x)) for x in ch_counts.index], ch_counts.values,
               color=C[:len(ch_counts)], edgecolor="white", lw=0.5)
        ax.set_title("Channel Count Distribution", color="white")
        ax.set_xlabel("Number of Channels", color="white")
        ax.set_ylabel("File Count", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        [ax.text(i, v+0.1, str(v), ha="center", color="white")
         for i,v in enumerate(ch_counts.values)]
    else:
        ax.text(0.5, 0.5, "Channel info\nnot available",
                ha="center", va="center", color="white", transform=ax.transAxes)
    ax.set_facecolor("#1a1a1a")

    plt.tight_layout()
    _save(fig, "13_channel_analysis.png")


def plot_14_telemetry(tele_result):
    """Visualise telemetry structure and coverage."""
    df = tele_result.get("combined_df")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d0d0d")
    fig.suptitle("Telemetry Structure & Coverage", fontsize=14, color="white")

    if df is None or len(df) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No telemetry data found",
                    ha="center", va="center", color="white", transform=ax.transAxes)
            ax.set_facecolor("#1a1a1a")
        plt.tight_layout(); _save(fig, "14_telemetry.png"); return

    # GPS track
    ax = axes[0,0]
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce").dropna()
        lons = pd.to_numeric(df[lon_col], errors="coerce").dropna()
        n    = min(len(lats), len(lons))
        sc   = ax.scatter(lons[:n], lats[:n], c=range(n), cmap="plasma",
                          s=15, alpha=0.8, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Time index")
        ax.set_title(f"GPS Track\n({lon_col} vs {lat_col})", color="white", fontsize=9)
        ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    else:
        ax.text(0.5, 0.5, "No lat/lon columns", ha="center", va="center",
                color="white", transform=ax.transAxes)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Altitude
    ax = axes[0,1]
    alt_col = next((c for c in df.columns
                    if any(x in c.lower() for x in ["alt","height","elev"])), None)
    if alt_col:
        alt = pd.to_numeric(df[alt_col], errors="coerce").dropna()
        ax.plot(alt.values, color=C[1], lw=1.5)
        ax.fill_between(range(len(alt)), alt, alpha=0.2, color=C[1])
        ax.set_title(f"Altitude Profile ({alt_col})", color="white", fontsize=9)
        ax.set_xlabel("Record index", color="white")
        ax.set_ylabel("Altitude (m)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Column null-map
    ax = axes[1,0]
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    ax.barh(range(len(null_pct)), null_pct.values, color=C[2], edgecolor="black")
    ax.set_yticks(range(len(null_pct)))
    ax.set_yticklabels(null_pct.index, fontsize=7, color="white")
    ax.set_title("Missing Data % per Column", color="white", fontsize=9)
    ax.set_xlabel("% null", color="white")
    ax.axvline(20, color=C[1], lw=1, ls="--", alpha=0.7)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Data type summary
    ax = axes[1,1]
    dtype_counts = df.dtypes.astype(str).value_counts()
    ax.bar(dtype_counts.index, dtype_counts.values,
           color=C[:len(dtype_counts)], edgecolor="white", lw=0.5)
    ax.set_title("Column Data Types", color="white", fontsize=9)
    ax.set_xlabel("Dtype", color="white"); ax.set_ylabel("Count", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout()
    _save(fig, "14_telemetry.png")


def plot_15_per_file_heatmap(diag_df):
    """Per-file diagnostic heatmap — overview of all files at once."""
    if diag_df.empty: return
    cols = ["snr_db","rms_db_mean","noise_floor_db","dominant_freq_hz",
            "clip_pct","silence_pct","spectral_centroid_hz",
            "kurtosis","crest_factor","zcr"]
    cols = [c for c in cols if c in diag_df.columns]
    if len(cols) < 2: return

    data = diag_df[cols].copy()
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Normalise each column 0–1 for visual
    for col in data.columns:
        rng = data[col].max() - data[col].min()
        if rng > 0:
            data[col] = (data[col] - data[col].min()) / rng

    labels = (diag_df["filename"].str[:25].values
              if "filename" in diag_df.columns else range(len(diag_df)))

    fig_h = max(8, len(data) * 0.35)
    fig, ax = plt.subplots(figsize=(16, fig_h), facecolor="#0d0d0d")
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_"," ") for c in cols],
                       color="white", rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="white", fontsize=6)
    ax.set_title("Per-File Diagnostic Heatmap (normalised, green=good)",
                 color="white", fontsize=12)
    plt.colorbar(im, ax=ax, label="Normalised value", shrink=0.5)
    fig.patch.set_facecolor("#0d0d0d")
    plt.tight_layout()
    _save(fig, "15_per_file_heatmap.png")


def plot_16_pca(diag_df):
    """PCA on diagnostic features — which files cluster together?"""
    if diag_df.empty: return
    cols = ["snr_db","rms_db_mean","noise_floor_db","dominant_freq_hz",
            "spectral_centroid_hz","spectral_flatness","kurtosis",
            "crest_factor","zcr","silence_pct","clip_pct"]
    cols = [c for c in cols if c in diag_df.columns]
    if len(cols) < 3: return

    X = diag_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X = StandardScaler().fit_transform(X)

    pca    = PCA(n_components=min(3, len(cols)), random_state=SEED)
    X_pca  = pca.fit_transform(X)
    ev     = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="#0d0d0d")
    fig.suptitle("PCA of Audio Diagnostic Features", fontsize=14, color="white")

    folders = (diag_df["folder"].values
               if "folder" in diag_df.columns
               else np.array(["all"]*len(diag_df)))
    uniq    = sorted(set(folders))

    for ax, (c1, c2) in zip(axes[:2], [(0,1),(0,2)]):
        for i, fld in enumerate(uniq):
            mask = folders == fld
            ax.scatter(X_pca[mask,c1], X_pca[mask,c2],
                       color=C[i%len(C)], label=str(fld),
                       alpha=0.8, s=70, edgecolors="white", lw=0.3)
        ax.set_xlabel(f"PC{c1+1} ({ev[c1]*100:.1f}%)", color="white")
        ax.set_ylabel(f"PC{c2+1} ({ev[c2]*100:.1f}%)", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Scree
    ax = axes[2]
    all_ev = PCA(random_state=SEED).fit(X).explained_variance_ratio_
    ax.bar(range(1,len(all_ev)+1), all_ev*100,
           color=C[0], edgecolor="white", lw=0.5)
    ax.plot(range(1,len(all_ev)+1), np.cumsum(all_ev)*100,
            "o-", color=C[1], lw=2, ms=5)
    ax.axhline(80, color="white", ls="--", lw=1, alpha=0.5)
    ax.set_title("Scree Plot", color="white")
    ax.set_xlabel("PC #", color="white"); ax.set_ylabel("Variance Explained (%)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout()
    _save(fig, "16_pca_diagnostics.png")


def plot_17_tsne(diag_df):
    """t-SNE of audio diagnostic features."""
    if len(diag_df) < 5: return
    cols = ["snr_db","rms_db_mean","noise_floor_db","dominant_freq_hz",
            "spectral_centroid_hz","kurtosis","crest_factor","zcr"]
    cols = [c for c in cols if c in diag_df.columns]
    X    = diag_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X    = StandardScaler().fit_transform(X)
    print("   ⏳  t-SNE …", end=" ", flush=True)
    ts   = TSNE(n_components=2, perplexity=min(20,len(X)//3+1),
                random_state=SEED, max_iter=800).fit_transform(X)
    print("done")

    fig, ax = plt.subplots(figsize=(10,8), facecolor="#0d0d0d")
    folders = (diag_df["folder"].values
               if "folder" in diag_df.columns
               else np.array(["all"]*len(diag_df)))
    uniq = sorted(set(folders))
    for i, fld in enumerate(uniq):
        mask = folders == fld
        ax.scatter(ts[mask,0], ts[mask,1], color=C[i%len(C)],
                   label=str(fld), alpha=0.85, s=80,
                   edgecolors="white", lw=0.4)
    ax.set_title("t-SNE — Audio File Clusters", color="white", fontsize=14)
    ax.legend(fontsize=9, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.grid(True, alpha=0.08)
    _save(fig, "17_tsne_clusters.png")


def plot_18_correlation_heatmap(diag_df):
    """Pearson correlation of all numerical diagnostic features."""
    if diag_df.empty: return
    num_df = diag_df.select_dtypes(include=np.number)
    if len(num_df.columns) < 3: return
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d0d0d")
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                linewidths=0.3, linecolor="#333",
                xticklabels=[c.replace("_"," ") for c in corr.columns],
                yticklabels=[c.replace("_"," ") for c in corr.columns],
                annot=len(corr) < 20, fmt=".2f", annot_kws={"size":6})
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=7)
    fig.patch.set_facecolor("#0d0d0d")
    _save(fig, "18_correlation_heatmap.png")


def plot_19_consistency_report(consist, hdr_df):
    """Visual summary of data consistency issues."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0d0d0d")
    fig.suptitle("Data Consistency Report", fontsize=14, color="white")

    # Issue list
    ax = axes[0,0]
    ax.axis("off")
    issues = consist.get("issues", ["No issues found ✔"])
    text   = "\n".join(f"  {'⚠' if i>0 else '✔'}  {issue}"
                       for i, issue in enumerate(issues))
    ax.text(0.05, 0.95, "DATA QUALITY ISSUES\n" + "─"*40 + "\n" + text,
            transform=ax.transAxes, color="white", fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

    # SR distribution
    ax = axes[0,1]
    if "sample_rate" in hdr_df.columns:
        sr_v = hdr_df["sample_rate"].dropna().value_counts()
        ax.bar([str(int(x)) for x in sr_v.index], sr_v.values,
               color=C[:len(sr_v)], edgecolor="white", lw=0.5)
        ax.set_title("Sample Rate Distribution", color="white")
        ax.set_xlabel("Hz", color="white"); ax.set_ylabel("Count", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Duration spread
    ax = axes[1,0]
    if "duration_sec" in hdr_df.columns:
        dur = hdr_df["duration_sec"].dropna()
        folders = hdr_df.get("folder", pd.Series(["all"]*len(hdr_df)))
        for i, (fld, idx) in enumerate(hdr_df.groupby(folders).groups.items()):
            ax.hist(hdr_df.loc[idx,"duration_sec"].dropna(), bins=30,
                    alpha=0.7, color=C[i%len(C)], label=str(fld), edgecolor="black")
        ax.set_title("Duration Spread per Folder", color="white")
        ax.set_xlabel("Duration (s)", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # File size vs expected (sr*dur*channels*bytedepth)
    ax = axes[1,1]
    if all(c in hdr_df.columns for c in
           ["size_bytes","sample_rate","n_channels","bit_depth","n_frames"]):
        hdr_c = hdr_df.dropna(subset=["size_bytes","sample_rate",
                                       "n_channels","bit_depth","n_frames"])
        expected = (hdr_c["n_frames"] * hdr_c["n_channels"] *
                    hdr_c["bit_depth"] / 8 + 44)  # +44 for WAV header
        actual   = hdr_c["size_bytes"]
        ratio    = (actual / (expected + 1)).clip(0, 3)
        ax.hist(ratio, bins=30, color=C[0], edgecolor="black", alpha=0.85)
        ax.axvline(1.0, color=C[1], lw=2, ls="--", label="Expected = Actual")
        ax.set_title("File Size / Expected Size Ratio", color="white")
        ax.set_xlabel("Ratio (1.0 = perfect)", color="white")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        truncated = int((ratio < 0.7).sum())
        if truncated:
            ax.text(0.05, 0.9, f"⚠ {truncated} files may be truncated",
                    transform=ax.transAxes, color=C[1], fontsize=9)

    plt.tight_layout()
    _save(fig, "19_consistency_report.png")


def plot_20_findings_summary(fs_df, diag_df, consist, tele_result,
                              log_result, align_result):
    """Final plain-English summary dashboard."""
    fig = plt.figure(figsize=(20, 12), facecolor="#0d0d0d")
    fig.suptitle("DUNAKESZI DATASET — FINDINGS SUMMARY",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    # Build text blocks
    n_audio  = int((fs_df["file_type"]=="audio").sum()) if not fs_df.empty else 0
    n_video  = int((fs_df["file_type"]=="video").sum()) if not fs_df.empty else 0
    n_data   = int((fs_df["file_type"]=="data").sum())  if not fs_df.empty else 0
    n_text   = int((fs_df["file_type"]=="text").sum())  if not fs_df.empty else 0
    total_gb = fs_df["size_mb"].sum()/1024              if not fs_df.empty else 0

    avg_snr  = float(diag_df["snr_db"].mean())         if not diag_df.empty and "snr_db" in diag_df.columns else 0
    avg_dur  = float(diag_df.get("sr", pd.Series([0])).mean()) if not diag_df.empty else 0
    n_clip   = int(diag_df["clipped"].sum())            if not diag_df.empty and "clipped" in diag_df.columns else 0
    n_issues = consist.get("n_issues", 0)

    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    blocks = [
        ("📁  FILE SYSTEM",
         f"Total files    : {len(fs_df)}\n"
         f"Total size     : {total_gb:.2f} GB\n"
         f"Audio files    : {n_audio}\n"
         f"Video files    : {n_video}\n"
         f"Data/log files : {n_data + n_text}\n"
         f"Top folders    : {', '.join(fs_df['folder_l1'].unique()[:4]) if not fs_df.empty else 'N/A'}"),

        ("🎙  AUDIO QUALITY",
         f"Mean SNR           : {avg_snr:.1f} dB\n"
         f"Clipped files      : {n_clip}\n"
         f"Noise colours      : "
         + (diag_df['noise_colour'].value_counts().to_string().replace('\n',' | ')
            if not diag_df.empty and 'noise_colour' in diag_df.columns
            else "N/A") + "\n"
         f"Sample rates found : "
         + (str(diag_df['sr'].value_counts().to_dict())
            if not diag_df.empty and 'sr' in diag_df.columns
            else "N/A")),

        ("🛸  TELEMETRY",
         f"Files found    : {len(tele_result.get('files', []))}\n"
         f"Combined rows  : "
         + (str(len(tele_result['combined_df']))
            if tele_result.get('combined_df') is not None else "0") + "\n"
         f"Columns found  : "
         + (str(list(tele_result['combined_df'].columns)[:5])
            if tele_result.get('combined_df') is not None else "N/A") + "\n"
         f"Issues         : {'; '.join(tele_result.get('issues',[]) or ['none'])}"),

        ("📋  LOGBOOK",
         f"Entries found  : {len(log_result.get('entries', []))}\n"
         f"Preview:\n"
         + "\n".join(("  " + e.get("text","")[:60])
                     for e in log_result.get("entries",[])[:5])),

        ("⚙  CONSISTENCY",
         f"Issues found   : {n_issues}\n"
         + "\n".join(f"  • {iss[:65]}" for iss in consist.get("issues",[])[:6])),

        ("🕐  ALIGNMENT",
         f"Audio w/ timestamps : {align_result.get('audio_timestamps_found',0)}\n"
         f"Telemetry range     : {align_result.get('tele_range','unknown')}\n"
         f"Aligned files       : {len(align_result.get('aligned',[]))}\n"
         f"Unmatched audio     : {len(align_result.get('unmatched_audio',[]))}\n"
         f"Logbook entries     : {align_result.get('log_entries',0)}"),
    ]

    for idx, (title, body) in enumerate(blocks):
        r, c = divmod(idx, 3)
        ax   = fig.add_subplot(gs[r, c])
        ax.axis("off")
        ax.set_facecolor("#0d1117")
        ax.text(0.05, 0.97, title, transform=ax.transAxes,
                color=C[idx%len(C)], fontsize=12, fontweight="bold", va="top")
        ax.text(0.05, 0.80, body, transform=ax.transAxes,
                color="white", fontsize=8, va="top", fontfamily="monospace",
                wrap=True)
        for sp in ax.spines.values():
            sp.set_edgecolor(C[idx%len(C)]); sp.set_linewidth(1.5)

    _save(fig, "20_findings_summary.png")


# =============================================================================
#  SECTION 8  ──  TEXT REPORT
# =============================================================================

def write_report(fs_df, hdr_df, diag_df, consist, tele_result,
                 log_result, align_result, t_elapsed):
    """Write JSON + plain-text findings report."""

    # ── JSON ──────────────────────────────────────────────────────────────────
    report = {
        "generated":     datetime.now().isoformat(),
        "elapsed_sec":   round(t_elapsed, 1),
        "filesystem": {
            "total_files":  int(len(fs_df)),
            "total_size_gb": float(fs_df["size_mb"].sum()/1024) if not fs_df.empty else 0,
            "by_type":      fs_df["file_type"].value_counts().to_dict() if not fs_df.empty else {},
            "by_folder":    fs_df["folder_l1"].value_counts().to_dict() if not fs_df.empty else {},
            "extensions":   fs_df["extension"].value_counts().to_dict() if not fs_df.empty else {},
        },
        "audio": {
            "total_wav":    int(len(hdr_df)) if not hdr_df.empty else 0,
            "sample_rates": (hdr_df["sample_rate"].value_counts().to_dict()
                             if not hdr_df.empty and "sample_rate" in hdr_df.columns else {}),
            "bit_depths":   (hdr_df["bit_depth"].value_counts().to_dict()
                             if not hdr_df.empty and "bit_depth" in hdr_df.columns else {}),
            "channel_counts":(hdr_df["n_channels"].value_counts().to_dict()
                             if not hdr_df.empty and "n_channels" in hdr_df.columns else {}),
            "duration_stats":({"min": float(hdr_df["duration_sec"].min()),
                               "max": float(hdr_df["duration_sec"].max()),
                               "mean": float(hdr_df["duration_sec"].mean())}
                             if not hdr_df.empty and "duration_sec" in hdr_df.columns else {}),
        },
        "signal_quality": {
            "mean_snr_db":    float(diag_df["snr_db"].mean()) if not diag_df.empty and "snr_db" in diag_df.columns else None,
            "clipped_files":  int(diag_df["clipped"].sum()) if not diag_df.empty and "clipped" in diag_df.columns else 0,
            "noise_colours":  (diag_df["noise_colour"].value_counts().to_dict()
                              if not diag_df.empty and "noise_colour" in diag_df.columns else {}),
            "mean_noise_floor_db": float(diag_df["noise_floor_db"].mean()) if not diag_df.empty and "noise_floor_db" in diag_df.columns else None,
        },
        "consistency":   consist,
        "telemetry": {
            "n_files":    len(tele_result.get("files",[])),
            "n_rows":     len(tele_result["combined_df"]) if tele_result.get("combined_df") is not None else 0,
            "columns":    (list(tele_result["combined_df"].columns)
                          if tele_result.get("combined_df") is not None else []),
            "issues":     tele_result.get("issues",[]),
        },
        "logbook": {
            "n_entries":  len(log_result.get("entries",[])),
        },
        "alignment":     align_result,
    }

    json_path = OUTPUT_DIR / "audit_report.json"
    with open(json_path,"w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Plain text ─────────────────────────────────────────────────────────────
    txt_path = OUTPUT_DIR / "audit_report.txt"
    lines    = [
        "=" * 72,
        "  DUNAKESZI DRONE DATASET — FORENSIC AUDIT REPORT",
        f"  Generated : {report['generated']}",
        f"  Elapsed   : {report['elapsed_sec']} s",
        "=" * 72, "",
        "SECTION 1 — FILE SYSTEM",
        "─" * 40,
        f"  Total files     : {report['filesystem']['total_files']}",
        f"  Total size      : {report['filesystem']['total_size_gb']:.3f} GB",
        f"  By type         : {report['filesystem']['by_type']}",
        f"  By folder       : {report['filesystem']['by_folder']}",
        "",
        "SECTION 2 — AUDIO FILES",
        "─" * 40,
        f"  WAV files parsed        : {report['audio']['total_wav']}",
        f"  Sample rates (Hz)       : {report['audio']['sample_rates']}",
        f"  Bit depths              : {report['audio']['bit_depths']}",
        f"  Channel counts          : {report['audio']['channel_counts']}",
        f"  Duration stats (s)      : {report['audio']['duration_stats']}",
        "",
        "SECTION 3 — SIGNAL QUALITY",
        "─" * 40,
        f"  Mean SNR                : {report['signal_quality']['mean_snr_db']} dB",
        f"  Clipped files           : {report['signal_quality']['clipped_files']}",
        f"  Noise colours           : {report['signal_quality']['noise_colours']}",
        f"  Mean noise floor        : {report['signal_quality']['mean_noise_floor_db']} dBFS",
        "",
        "SECTION 4 — CONSISTENCY ISSUES",
        "─" * 40,
    ] + [f"  {'⚠' if i else '✔'}  {iss}"
         for i, iss in enumerate(consist.get("issues",["No issues found"]))] + [
        "",
        "SECTION 5 — TELEMETRY",
        "─" * 40,
        f"  Files         : {report['telemetry']['n_files']}",
        f"  Records       : {report['telemetry']['n_rows']}",
        f"  Columns       : {report['telemetry']['columns']}",
        f"  Issues        : {report['telemetry']['issues']}",
        "",
        "SECTION 6 — TEMPORAL ALIGNMENT",
        "─" * 40,
        f"  Audio files with timestamps : {align_result.get('audio_timestamps_found',0)}",
        f"  Telemetry range             : {align_result.get('tele_range','unknown')}",
        f"  Aligned                     : {len(align_result.get('aligned',[]))}",
        f"  Unmatched audio             : {len(align_result.get('unmatched_audio',[]))}",
        "",
        "NEXT STEPS RECOMMENDED",
        "─" * 40,
        "  1. Confirm MEMS array geometry (number of channels per file, spacing)",
        "  2. Verify sample rate consistency before concatenating files",
        "  3. Re-examine clipped files — consider re-recording or windowing",
        "  4. Align audio timestamps with telemetry using logbook session notes",
        "  5. Annotate drone vs background segments using telemetry altitude/speed",
        "  6. Check high-silence-ratio files — may be pad artefacts",
        "=" * 72,
    ]
    txt_path.write_text("\n".join(lines))

    print(f"\n  📝  audit_report.json  →  {json_path}")
    print(f"  📝  audit_report.txt   →  {txt_path}")
    return report


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t0 = time.time()

    # 0. Download / locate data
    print(f"\n{SEP}\n  STEP 0 — DATA ACQUISITION\n{SEP}")
    folder_map = download_dataset(WORKSPACE)

    # 1. File-system audit
    print(f"\n{SEP}\n  STEP 1 — FILE-SYSTEM AUDIT\n{SEP}")
    root = folder_map.get("_root", WORKSPACE)
    fs_df = audit_filesystem(Path(root))
    fs_df.to_csv(OUTPUT_DIR / "filesystem_audit.csv", index=False)
    print(f"  Saved → filesystem_audit.csv")

    # 2. Audio forensics
    print(f"\n{SEP}\n  STEP 2 — AUDIO FILE FORENSICS\n{SEP}")
    hdr_df, diag_df = audit_audio_files(fs_df)
    hdr_df.to_csv(OUTPUT_DIR / "audio_headers.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(OUTPUT_DIR / "audio_diagnostics.csv", index=False)
    print(f"  Saved → audio_headers.csv, audio_diagnostics.csv")

    # 3. Telemetry
    print(f"\n{SEP}\n  STEP 3 — TELEMETRY AUDIT\n{SEP}")
    tele_result = audit_telemetry(folder_map)
    if tele_result.get("combined_df") is not None:
        tele_result["combined_df"].to_csv(
            OUTPUT_DIR / "telemetry_combined.csv", index=False)

    # 4. Logbook
    print(f"\n{SEP}\n  STEP 4 — LOGBOOK AUDIT\n{SEP}")
    log_result = audit_logbook(folder_map)
    if log_result.get("entries"):
        pd.DataFrame(log_result["entries"]).to_csv(
            OUTPUT_DIR / "logbook_entries.csv", index=False)
    if log_result.get("raw_text"):
        (OUTPUT_DIR / "logbook_raw.txt").write_text(log_result["raw_text"])

    # 5. Consistency
    print(f"\n{SEP}\n  STEP 5 — CONSISTENCY ANALYSIS\n{SEP}")
    consist = consistency_analysis(hdr_df, diag_df)
    print(f"  Issues found: {consist.get('n_issues', 0)}")
    for iss in consist.get("issues", []):
        print(f"  ⚠  {iss}")

    # 6. Temporal alignment
    print(f"\n{SEP}\n  STEP 6 — TEMPORAL ALIGNMENT\n{SEP}")
    align_result = temporal_alignment(diag_df, tele_result, log_result)

    # 7. All 20 plots
    print(f"\n{SEP}\n  STEP 7 — GENERATING 20 PLOTS\n{SEP}")
    plot_01_filesystem_overview(fs_df)
    plot_02_folder_tree(fs_df)
    plot_03_audio_header_stats(hdr_df)
    plot_04_signal_health(diag_df)
    plot_05_snr_deep(diag_df)
    plot_06_noise_colour(diag_df)
    plot_07_sample_waveforms(diag_df)
    plot_08_psd_gallery(diag_df)
    plot_09_spectrograms(diag_df)
    plot_10_mel_spectrograms(diag_df)
    plot_11_dominant_frequencies(diag_df)
    plot_12_duration_timeline(hdr_df)
    plot_13_channel_comparison(diag_df)
    plot_14_telemetry(tele_result)
    plot_15_per_file_heatmap(diag_df)
    plot_16_pca(diag_df)
    plot_17_tsne(diag_df)
    plot_18_correlation_heatmap(diag_df)
    plot_19_consistency_report(consist, hdr_df)
    plot_20_findings_summary(fs_df, diag_df, consist, tele_result,
                             log_result, align_result)

    # 8. Report
    print(f"\n{SEP}\n  STEP 8 — WRITING REPORT\n{SEP}")
    report = write_report(fs_df, hdr_df, diag_df, consist,
                          tele_result, log_result, align_result,
                          time.time() - t0)

    print(f"\n{SEP2}")
    print("  AUDIT COMPLETE")
    print(SEP2)
    print(f"  Elapsed        : {report['elapsed_sec']} s")
    print(f"  Files audited  : {report['filesystem']['total_files']}")
    print(f"  WAV diagnosed  : {report['audio']['total_wav']}")
    print(f"  Issues found   : {consist.get('n_issues',0)}")
    print(f"  Plots saved    : {PLOT_DIR}")
    print(f"  Reports        : {OUTPUT_DIR}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()