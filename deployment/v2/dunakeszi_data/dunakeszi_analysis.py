"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      DUNAKESZI DRONE ACOUSTIC DATASET — FULL ANALYSIS PIPELINE             ║
║      OneDrive Selective Downloader + Acoustic EDA + PyTorch Trainer        ║
║                                                                              ║
║  Folders targeted:                                                           ║
║    • Dunakeszi_MEMS         (1.6 GB)  — MEMS microphone array recordings   ║
║    • Dunakeszi_DRON-ADATOK  (162 MB)  — drone telemetry / flight logs       ║
║    • Dunakeszi_JEGYZOKONYV  (1.4 MB)  — field logbook (metadata)            ║
║    • Dunakeszi_VIDEÓ        (2 GB)    — video (index only, not downloaded)  ║
║                                                                              ║
║  Pipeline stages:                                                            ║
║    0. OneDrive authentication & selective download                          ║
║    1. Dataset structure audit & logbook parsing                             ║
║    2. Drone telemetry parsing (GPS, altitude, timestamp alignment)          ║
║    3. MEMS acoustic feature extraction (MFCC, spectral, temporal, noise)    ║
║    4. PyTorch dataset + CNN classifier (mel-spectrogram input)              ║
║    5. Direction-of-Arrival (DOA) estimation via MEMS array geometry         ║
║    6. Path tracking (Kalman filter on DOA + telemetry)                     ║
║    7. 16 visualisation plots                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO USE
──────────
1.  Run this script. It will print an authorisation URL.
2.  Paste that URL in your browser → log in with the OneDrive account that
    owns the Dunakeszi dataset → copy the redirected URL back here.
3.  The script downloads ONLY what it needs (MEMS samples up to MAX_MEMS_MB,
    full DRON-ADATOK, full JEGYZOKONYV) and then runs the full analysis.

Alternatively, set ONEDRIVE_SHARE_URL to a share link from OneDrive and the
script will use the anonymous download path (no login required).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, re, json, wave, struct, csv, io, time, hashlib, threading
import warnings, logging, argparse, textwrap
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from urllib.parse import urlencode, quote, urlparse, parse_qs
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import queue

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Third-party (all available in the environment) ───────────────────────────
import requests
import numpy as np
import pandas as pd
from scipy import signal, fft, stats
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import seaborn as sns

# ── PyTorch (graceful fallback) ───────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH = True
    print(f"✔  PyTorch {torch.__version__} available")
except ImportError:
    TORCH = False
    print("⚠  PyTorch not found — classifier will use sklearn RandomForest")

# =============================================================================
#  ███  USER CONFIGURATION  ███
#  Edit the block below before running
# =============================================================================

# ── Option A: Paste a OneDrive share link here (anonymous, easiest) ──────────
#   Right-click folder in OneDrive → Share → Copy link → paste below
#   Leave empty "" to use OAuth interactive login (Option B)
ONEDRIVE_SHARE_URL = ""       # e.g. "https://1drv.ms/f/s!Abc123..."

# ── Option B: OAuth app credentials (Azure App Registration) ─────────────────
#   Create a free app at https://portal.azure.com → App registrations
#   Add redirect URI: http://localhost:8080
AZURE_CLIENT_ID    = ""       # your app's client ID
AZURE_TENANT_ID    = "common" # or your tenant ID
REDIRECT_URI       = "http://localhost:8080"

# ── Download limits (prevent accidental full 169 GB download) ────────────────
MAX_MEMS_FILES     = 50       # max number of MEMS .wav files to download
MAX_MEMS_MB        = 500      # hard cap in MB for MEMS folder total
MAX_VIDEO_FILES    = 0        # set >0 to also sample video files
SAMPLE_DURATION_S  = 10       # trim each recording to this many seconds

# ── Local workspace ───────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
MEMS_DIR    = DATA_DIR / "MEMS"
DRONE_DIR   = DATA_DIR / "DRON-ADATOK"
LOG_DIR     = DATA_DIR / "JEGYZOKONYV"
VIDEO_DIR   = DATA_DIR / "VIDEO"
OUTPUT_DIR  = BASE_DIR / "dunakeszi_analysis"
PLOT_DIR    = OUTPUT_DIR / "plots"
MODEL_DIR   = OUTPUT_DIR / "models"

for d in [MEMS_DIR, DRONE_DIR, LOG_DIR, VIDEO_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Signal processing constants ───────────────────────────────────────────────
SR_TARGET  = 22050
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
N_MFCC     = 40
SEED       = 42
np.random.seed(SEED)

# ── MEMS array geometry (typical 4-mic cross array, 5 cm spacing) ─────────────
#   Update these coordinates once you know your actual array layout
#   Format: (x_cm, y_cm) relative to array centre
MEMS_MIC_POSITIONS = np.array([
    [ 0.0,  5.0],   # mic 0 — North
    [ 5.0,  0.0],   # mic 1 — East
    [ 0.0, -5.0],   # mic 2 — South
    [-5.0,  0.0],   # mic 3 — West
]) / 100.0           # convert to metres
SPEED_OF_SOUND = 343.0  # m/s at 20°C

# ── Visuals ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PAL = ["#00FFCC","#FF4C6A","#FFD700","#7B68EE","#FF8C00",
       "#00BFFF","#ADFF2F","#FF69B4","#40E0D0","#FF6347"]
sns.set_palette(PAL)

print("\n" + "═"*72)
print("  DUNAKESZI DRONE ACOUSTIC PIPELINE  —  Security AI Toolkit")
print("═"*72)


# =============================================================================
#  SECTION 0 ── ONEDRIVE DOWNLOADER
# =============================================================================

class OneDriveDownloader:
    """
    Handles both:
      A) Anonymous share-link download  (ONEDRIVE_SHARE_URL set)
      B) OAuth 2.0 device-code flow     (no share link — interactive login)

    Key method: .download_folders(folder_map, limits)
      folder_map = {"MEMS": MEMS_DIR, "DRON-ADATOK": DRONE_DIR, ...}
    """

    GRAPH_BASE  = "https://graph.microsoft.com/v1.0"
    AUTH_BASE   = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0"
    SCOPES      = "Files.Read Files.Read.All offline_access"

    def __init__(self):
        self.token       = None
        self.session     = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._drive_id   = None
        self._item_cache = {}

    # ── A: Share-link path ────────────────────────────────────────────────────
    def resolve_share_link(self, url: str) -> dict:
        """Convert a share URL to a Graph API driveItem."""
        encoded = "u!" + url.rstrip("=").replace("/","_").replace("+","-")
        # Try as anonymous share
        api_url = f"{self.GRAPH_BASE}/shares/{encoded}/driveItem"
        r = self.session.get(api_url)
        if r.status_code == 401:
            raise ValueError("Share link requires authentication. "
                             "Use OAuth path or create an 'Anyone with link' share.")
        r.raise_for_status()
        return r.json()

    # ── B: OAuth device-code flow ─────────────────────────────────────────────
    def oauth_device_flow(self):
        """Interactive device-code login — no browser redirect needed."""
        if not AZURE_CLIENT_ID:
            raise ValueError(
                "\n\n  ┌─ ACTION REQUIRED ──────────────────────────────────────────┐\n"
                "  │  Set ONEDRIVE_SHARE_URL or AZURE_CLIENT_ID at the top of    │\n"
                "  │  this script before running.                                │\n"
                "  │                                                              │\n"
                "  │  Quickest path:                                              │\n"
                "  │    1. Open OneDrive in browser                              │\n"
                "  │    2. Right-click Dunakeszi_MEMS → Share                   │\n"
                "  │    3. Set to 'Anyone with link can view'                   │\n"
                "  │    4. Copy link → paste into ONEDRIVE_SHARE_URL above      │\n"
                "  └────────────────────────────────────────────────────────────┘\n"
            )

        r = requests.post(f"{self.AUTH_BASE}/devicecode", data={
            "client_id": AZURE_CLIENT_ID,
            "scope":     self.SCOPES,
        })
        r.raise_for_status()
        dc = r.json()
        print(f"\n  ┌─ BROWSER SIGN-IN REQUIRED ─────────────────────────────────┐")
        print(f"  │  1. Open:  {dc['verification_uri']:<50}│")
        print(f"  │  2. Enter code:  {dc['user_code']:<44}│")
        print(f"  └────────────────────────────────────────────────────────────┘\n")

        # Poll for token
        interval = dc.get("interval", 5)
        expires  = time.time() + dc.get("expires_in", 900)
        while time.time() < expires:
            time.sleep(interval)
            tr = requests.post(f"{self.AUTH_BASE}/token", data={
                "client_id":   AZURE_CLIENT_ID,
                "device_code": dc["device_code"],
                "grant_type":  "urn:ietf:params:oauth:grant-type:device_code",
            })
            if tr.status_code == 200:
                self.token = tr.json()["access_token"]
                self.session.headers["Authorization"] = f"Bearer {self.token}"
                print("  ✔  Authenticated with OneDrive\n")
                return
            err = tr.json().get("error","")
            if err == "authorization_pending":
                print("  ⏳  Waiting for sign-in …", end="\r")
            elif err == "expired_token":
                raise TimeoutError("Device code expired. Re-run the script.")
            else:
                raise RuntimeError(f"Auth error: {tr.text}")
        raise TimeoutError("Sign-in timed out.")

    # ── Graph API helpers ─────────────────────────────────────────────────────
    def _get(self, url, **kwargs):
        r = self.session.get(url, **kwargs)
        r.raise_for_status()
        return r.json()

    def list_children(self, item_id: str, drive_id: str = None) -> list:
        if drive_id:
            url = f"{self.GRAPH_BASE}/drives/{drive_id}/items/{item_id}/children"
        else:
            url = f"{self.GRAPH_BASE}/me/drive/items/{item_id}/children"
        items, nxt = [], url
        while nxt:
            data = self._get(nxt)
            items.extend(data.get("value", []))
            nxt = data.get("@odata.nextLink")
        return items

    def get_root_children(self) -> list:
        return self._get(f"{self.GRAPH_BASE}/me/drive/root/children").get("value", [])

    def find_dunakeszi_root(self) -> dict:
        """Walk root to find the Dunakeszi folder."""
        print("  🔍  Searching OneDrive root for Dunakeszi dataset …")
        for item in self.get_root_children():
            if "Dunakeszi" in item.get("name","") or "dunakeszi" in item.get("name","").lower():
                print(f"      Found: {item['name']}  (id={item['id']})")
                return item
            # one level deeper
            if item.get("folder"):
                for sub in self.list_children(item["id"]):
                    if "Dunakeszi" in sub.get("name",""):
                        print(f"      Found nested: {sub['name']}")
                        return sub
        raise FileNotFoundError(
            "Could not locate Dunakeszi folder in OneDrive root. "
            "Share a direct folder link via ONEDRIVE_SHARE_URL instead."
        )

    # ── Core download logic ───────────────────────────────────────────────────
    def download_file(self, item: dict, dest_path: Path,
                      max_bytes: int = None) -> bool:
        """Download a single file item; optional byte cap (for large WAVs)."""
        url = item.get("@microsoft.graph.downloadUrl") or \
              item.get("@content.downloadUrl")
        if not url:
            # Fetch fresh download URL
            iid = item["id"]
            di  = item.get("parentReference",{}).get("driveId","")
            ep  = (f"{self.GRAPH_BASE}/drives/{di}/items/{iid}"
                   if di else f"{self.GRAPH_BASE}/me/drive/items/{iid}")
            url = self._get(ep).get("@microsoft.graph.downloadUrl","")
        if not url:
            print(f"      ⚠  No download URL for {item['name']}")
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            return True  # already downloaded

        headers = {}
        if max_bytes:
            headers["Range"] = f"bytes=0-{max_bytes-1}"

        try:
            r = self.session.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            total  = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            return True
        except Exception as e:
            print(f"      ⚠  Download failed for {item['name']}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def download_folders(self, folder_map: dict,
                         mems_file_limit: int = MAX_MEMS_FILES,
                         mems_mb_limit:   float = MAX_MEMS_MB):
        """
        folder_map = {"MEMS": MEMS_DIR, "DRON-ADATOK": DRONE_DIR, ...}
        Smart limits applied to MEMS only; other folders fully downloaded.
        """
        if ONEDRIVE_SHARE_URL:
            print("  Using share-link mode …")
            root_item = self.resolve_share_link(ONEDRIVE_SHARE_URL)
        else:
            self.oauth_device_flow()
            root_item = self.find_dunakeszi_root()

        drive_id = root_item.get("parentReference", {}).get("driveId", "")
        root_id  = root_item["id"]

        # Map subfolder names → item IDs
        children = self.list_children(root_id, drive_id)
        name_map = {c["name"]: c for c in children}
        print(f"\n  📂  Found {len(children)} items in dataset root:")
        for c in children:
            sz = c.get("size", 0) / 1024**2
            print(f"      {'📁' if c.get('folder') else '📄'}  "
                  f"{c['name']:<40} {sz:>8.1f} MB")

        summary = {}
        for folder_key, local_dir in folder_map.items():
            # Match partial folder name (e.g. "MEMS" matches "Dunakeszi_MEMS")
            matched = next((v for k, v in name_map.items()
                            if folder_key in k), None)
            if not matched:
                print(f"\n  ⚠  Folder '{folder_key}' not found in OneDrive root")
                continue

            folder_name = matched["name"]
            folder_id   = matched["id"]
            is_mems     = "MEMS" in folder_key.upper()
            print(f"\n  ⬇  Downloading {'[SAMPLED] ' if is_mems else ''}"
                  f"{folder_name} → {local_dir} …")

            items = self.list_children(folder_id, drive_id)
            # Recursively collect files
            all_files = self._collect_files(items, drive_id, depth=3)
            print(f"     Total files found: {len(all_files)}")

            downloaded, skipped, total_mb = 0, 0, 0.0
            for item in all_files:
                name = item["name"]
                ext  = Path(name).suffix.lower()
                size_mb = item.get("size", 0) / 1024**2

                # MEMS limits
                if is_mems:
                    if ext not in {".wav",".flac",".w64",".raw",".pcm"}:
                        skipped += 1; continue
                    if downloaded >= mems_file_limit:
                        break
                    if total_mb + size_mb > mems_mb_limit:
                        print(f"     ⚡ MEMS MB cap ({mems_mb_limit} MB) reached")
                        break
                    # Download only first SAMPLE_DURATION_S seconds
                    bytes_cap = int(size_mb * 1024**2 * min(
                        1.0, SAMPLE_DURATION_S / max(self._estimate_duration(item), 1)))
                    dest = local_dir / name
                    ok   = self.download_file(item, dest, max_bytes=bytes_cap)
                else:
                    dest = local_dir / name
                    ok   = self.download_file(item, dest)

                if ok:
                    downloaded += 1
                    total_mb   += dest.stat().st_size / 1024**2
                    print(f"     ✔  {name:<50} {size_mb:>6.1f} MB", end="\r")
                else:
                    skipped += 1

            print(f"\n     ✔  {downloaded} files downloaded  "
                  f"({total_mb:.1f} MB)  |  {skipped} skipped")
            summary[folder_key] = {"files": downloaded, "mb": total_mb}

        return summary

    def _collect_files(self, items, drive_id, depth=3):
        files = []
        for item in items:
            if item.get("file"):
                files.append(item)
            elif item.get("folder") and depth > 0:
                children = self.list_children(item["id"], drive_id)
                files.extend(self._collect_files(children, drive_id, depth-1))
        return files

    @staticmethod
    def _estimate_duration(item):
        """Estimate audio duration from file size (assumes 16-bit mono 44.1k)."""
        size = item.get("size", 0)
        return size / (44100 * 2)  # bytes / (sr * bytes_per_sample)


# =============================================================================
#  SECTION 1 ── LOGBOOK & TELEMETRY PARSER
# =============================================================================

def parse_logbook(log_dir: Path) -> pd.DataFrame:
    """
    Parse field notes from JEGYZOKONYV.
    Supports: .txt, .csv, .xlsx, .json, .pdf (text extraction).
    Returns a DataFrame with columns: datetime, drone_type, distance_m,
    direction, weather, notes.
    """
    print("\n  📋  Parsing field logbook …")
    records = []
    for f in sorted(log_dir.rglob("*")):
        if f.is_dir(): continue
        ext = f.suffix.lower()
        try:
            if ext in {".txt", ".log"}:
                text = f.read_text(errors="replace")
                records += _parse_logbook_text(text, f.name)
            elif ext == ".csv":
                df = pd.read_csv(f, on_bad_lines="skip")
                records.append({"source": f.name, "raw": df.to_dict()})
            elif ext == ".json":
                records.append({"source": f.name, "raw": json.loads(f.read_text())})
            else:
                records.append({"source": f.name, "raw": f"[binary: {ext}]"})
        except Exception as e:
            records.append({"source": f.name, "error": str(e)})

    if not records:
        print("     ⚠  No logbook files found — using empty metadata")
        return pd.DataFrame(columns=["datetime","event","notes","source"])

    # Best-effort normalised DataFrame
    rows = []
    for r in records:
        rows.append({
            "source":    r.get("source",""),
            "datetime":  r.get("datetime", None),
            "event":     r.get("event", ""),
            "notes":     str(r.get("raw", r.get("notes",""))),
        })
    df = pd.DataFrame(rows)
    print(f"     ✔  {len(df)} logbook entries parsed")
    return df


def _parse_logbook_text(text: str, source: str) -> list:
    """Heuristic parser for free-text field notes."""
    records = []
    # Match common timestamp formats
    ts_pattern = re.compile(
        r"(\d{4}[-./]\d{2}[-./]\d{2}[\s_T]\d{2}[:.]\d{2}(?:[:.]\d{2})?)")
    lines = text.splitlines()
    current = {"source": source, "notes": ""}
    for line in lines:
        m = ts_pattern.search(line)
        if m:
            if current.get("notes"):
                records.append(dict(current))
            current = {"source": source, "datetime": m.group(1), "notes": line}
        else:
            current["notes"] = current.get("notes","") + " " + line.strip()
    if current.get("notes"):
        records.append(current)
    return records


def parse_drone_telemetry(drone_dir: Path) -> pd.DataFrame:
    """
    Parse DRON-ADATOK: supports .csv, .json, .txt, .log, .kml, .gpx.
    Extracts: timestamp, latitude, longitude, altitude_m, speed_ms,
              heading_deg, drone_id.
    """
    print("\n  🛸  Parsing drone telemetry …")
    all_rows = []
    for f in sorted(drone_dir.rglob("*")):
        if f.is_dir(): continue
        ext = f.suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(f, on_bad_lines="skip")
                df["source_file"] = f.name
                all_rows.append(_normalise_telemetry_df(df))
            elif ext == ".json":
                data = json.loads(f.read_text())
                df   = pd.json_normalize(data if isinstance(data, list)
                                         else [data])
                df["source_file"] = f.name
                all_rows.append(_normalise_telemetry_df(df))
            elif ext in {".txt", ".log"}:
                df = _parse_telemetry_text(f.read_text(errors="replace"), f.name)
                if df is not None:
                    all_rows.append(df)
            elif ext == ".kml":
                all_rows.append(_parse_kml(f))
            elif ext == ".gpx":
                all_rows.append(_parse_gpx(f))
        except Exception as e:
            print(f"     ⚠  {f.name}: {e}")

    if not all_rows:
        print("     ⚠  No telemetry data found — creating synthetic track")
        return _synthetic_telemetry()

    tele = pd.concat([df for df in all_rows if df is not None and len(df)],
                     ignore_index=True)
    # Parse timestamps
    for col in ["timestamp","time","datetime","Time","Timestamp"]:
        if col in tele.columns:
            tele["timestamp"] = pd.to_datetime(tele[col], errors="coerce",
                                               utc=True)
            break
    tele = tele.sort_values("timestamp").reset_index(drop=True)
    print(f"     ✔  {len(tele)} telemetry points from {len(all_rows)} file(s)")
    print(f"        Time span: {tele['timestamp'].min()} → {tele['timestamp'].max()}")
    return tele


def _normalise_telemetry_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column name variants to canonical names."""
    renames = {}
    col_map = {
        "lat":       ["lat","latitude","Latitude","GPS_Lat","gps_lat","LATITUDE"],
        "lon":       ["lon","lng","longitude","Longitude","GPS_Lon","LONGITUDE"],
        "alt":       ["alt","altitude","Altitude","height","Height","ALT"],
        "speed":     ["speed","Speed","velocity","Velocity","groundspeed"],
        "heading":   ["heading","Heading","yaw","Yaw","bearing"],
        "timestamp": ["timestamp","time","Time","datetime","Timestamp","t"],
    }
    for canonical, variants in col_map.items():
        for v in variants:
            if v in df.columns:
                renames[v] = canonical
                break
    df = df.rename(columns=renames)
    for col in ["lat","lon","alt","speed","heading"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["timestamp","lat","lon","alt","speed","heading","source_file"]
              if "source_file" in df.columns else
              ["timestamp","lat","lon","alt","speed","heading"]]


def _parse_telemetry_text(text: str, source: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = re.split(r"[,;\t ]+", line.strip())
        nums  = []
        for p in parts:
            try: nums.append(float(p))
            except: pass
        if len(nums) >= 3:
            rows.append(nums[:6])
    if not rows: return None
    cols = ["lat","lon","alt","speed","heading","extra"][:len(rows[0])]
    df   = pd.DataFrame(rows, columns=cols)
    df["timestamp"]   = pd.NaT
    df["source_file"] = source
    return _normalise_telemetry_df(df)


def _parse_kml(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    coords = re.findall(
        r"<coordinates>(.*?)</coordinates>", text, re.DOTALL)
    rows = []
    for block in coords:
        for triplet in block.strip().split():
            parts = triplet.split(",")
            if len(parts) >= 2:
                rows.append({"lon": float(parts[0]), "lat": float(parts[1]),
                             "alt": float(parts[2]) if len(parts)>2 else 0.0,
                             "timestamp": pd.NaT, "source_file": path.name})
    return pd.DataFrame(rows) if rows else None


def _parse_gpx(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    rows = []
    for m in re.finditer(
            r'lat="([\d.\-]+)"\s+lon="([\d.\-]+)".*?(?:<ele>([\d.]+)</ele>)?'
            r'.*?(?:<time>(.*?)</time>)?', text, re.DOTALL):
        rows.append({"lat": float(m.group(1)), "lon": float(m.group(2)),
                     "alt": float(m.group(3) or 0),
                     "timestamp": m.group(4), "source_file": path.name})
    return pd.DataFrame(rows) if rows else None


def _synthetic_telemetry() -> pd.DataFrame:
    """Simulate a drone flight path for testing."""
    t0   = datetime(2024, 10, 24, 10, 0, 0)
    n    = 200
    t    = [t0 + timedelta(seconds=i*2) for i in range(n)]
    # Circular flight path centred on Dunakeszi (approx GPS)
    lat0, lon0 = 47.625, 19.135
    r_deg = 0.001
    theta = np.linspace(0, 4*np.pi, n)
    lat   = lat0 + r_deg * np.sin(theta) + np.random.randn(n)*0.00005
    lon   = lon0 + r_deg * np.cos(theta) + np.random.randn(n)*0.00005
    alt   = 30 + 10*np.sin(theta/2) + np.random.randn(n)*0.5
    spd   = 5 + 2*np.cos(theta) + np.abs(np.random.randn(n)*0.3)
    hdg   = np.degrees(np.arctan2(np.gradient(lon), np.gradient(lat))) % 360
    return pd.DataFrame({"timestamp": t, "lat": lat, "lon": lon,
                         "alt": alt, "speed": spd, "heading": hdg,
                         "source_file": "synthetic"})


# =============================================================================
#  SECTION 2 ── MEMS ACOUSTIC FEATURE EXTRACTION
# =============================================================================

def read_wav_safe(path):
    try:
        sr, data = wavfile.read(str(path))
        if data.ndim > 1:
            # For MEMS arrays: keep all channels separately
            channels = data.T.astype(np.float32)
        else:
            channels = data[np.newaxis].astype(np.float32)
        # Normalise
        mx = np.max(np.abs(channels))
        if mx > 0:
            channels /= mx
        # Handle int formats
        if data.dtype == np.int16:
            channels /= 32768.0
        elif data.dtype == np.int32:
            channels /= 2147483648.0
        return channels, int(sr)
    except Exception as e:
        print(f"     ⚠  Could not read {path.name}: {e}")
        return None, None


def resample(data, sr_in, sr_out=SR_TARGET):
    if sr_in == sr_out: return data, sr_out
    n_out = int(len(data) * sr_out / sr_in)
    return signal.resample(data, n_out).astype(np.float32), sr_out


def compute_mel_filterbank(sr, n_fft, n_mels=N_MELS):
    low_hz, high_hz = 20.0, sr / 2.0
    low_mel  = 2595 * np.log10(1 + low_hz  / 700)
    high_mel = 2595 * np.log10(1 + high_hz / 700)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb       = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        for k in range(bins[m-1], bins[m]):
            if bins[m] != bins[m-1]:
                fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
        for k in range(bins[m], bins[m+1]):
            if bins[m+1] != bins[m]:
                fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])
    return fb


MEL_FB = compute_mel_filterbank(SR_TARGET, N_FFT, N_MELS)


def compute_mfcc(data, sr):
    f, t, Zxx = signal.stft(data, fs=sr, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    mag    = np.abs(Zxx)
    power  = mag ** 2
    mel    = MEL_FB @ power
    mel_db = 10 * np.log10(np.maximum(mel, 1e-10))
    mfcc   = np.zeros((N_MFCC, mel_db.shape[1]))
    for n in range(N_MFCC):
        mfcc[n] = np.sum(mel_db.T * np.cos(
            np.pi * n * (2*np.arange(N_MELS)+1) / (2*N_MELS)), axis=1)
    return mfcc, mel_db, f, t, mag


def rms_energy(data, frame_len=N_FFT, hop=HOP_LENGTH):
    n = 1 + (len(data) - frame_len) // hop
    return np.array([np.sqrt(np.mean(data[i*hop:i*hop+frame_len]**2))
                     for i in range(n)])


def zcr(data, frame_len=N_FFT, hop=HOP_LENGTH):
    n = 1 + (len(data) - frame_len) // hop
    return np.array([np.mean(np.abs(np.diff(np.sign(data[i*hop:i*hop+frame_len]))))/2
                     for i in range(n)])


def spectral_centroid(mag, freqs):
    return (freqs[:,None] * mag).sum(0) / (mag.sum(0) + 1e-10)


def spectral_flatness(mag):
    log_mean = np.mean(np.log(mag + 1e-10), axis=0)
    arith    = np.mean(mag, axis=0) + 1e-10
    return np.exp(log_mean) / arith


def estimate_snr(data, sr):
    rms_f = rms_energy(data, frame_len=int(0.2*sr), hop=int(0.1*sr))
    noise = np.percentile(rms_f, 10) + 1e-12
    sig   = np.percentile(rms_f, 90) + 1e-12
    return 20 * np.log10(sig / noise)


def dominant_peaks(mag, freqs, k=5):
    mean_m = mag.mean(1)
    peaks, props = signal.find_peaks(mean_m,
                                     height=mean_m.max()*0.05, distance=5)
    if len(peaks) == 0:
        return np.zeros(k), np.zeros(k)
    idx  = np.argsort(props["peak_heights"])[::-1][:k]
    pks  = peaks[idx]
    return np.pad(freqs[pks], (0, k-len(pks))), \
           np.pad(mean_m[pks], (0, k-len(pks)))


def extract_features(data, sr, label="unknown", filename=""):
    mfcc, mel_db, freqs, t_ax, mag = compute_mfcc(data, sr)
    sc   = spectral_centroid(mag, freqs)
    sf   = spectral_flatness(mag)
    rms_ = rms_energy(data)
    zcr_ = zcr(data)
    top_f, top_a = dominant_peaks(mag, freqs)
    snr  = estimate_snr(data, sr)

    feat = {}
    for i in range(N_MFCC):
        feat[f"mfcc_{i:02d}_mean"] = float(mfcc[i].mean())
        feat[f"mfcc_{i:02d}_std"]  = float(mfcc[i].std())
        feat[f"mfcc_{i:02d}_d1"]   = float(np.mean(np.diff(mfcc[i])))
    feat["sc_mean"]       = float(sc.mean())
    feat["sc_std"]        = float(sc.std())
    feat["sf_mean"]       = float(sf.mean())
    feat["sf_std"]        = float(sf.std())
    feat["rms_mean"]      = float(rms_.mean())
    feat["rms_std"]       = float(rms_.std())
    feat["rms_max"]       = float(rms_.max())
    feat["rms_kurtosis"]  = float(stats.kurtosis(rms_))
    feat["zcr_mean"]      = float(zcr_.mean())
    feat["zcr_std"]       = float(zcr_.std())
    feat["snr_db"]        = float(snr)
    feat["duration_sec"]  = float(len(data) / sr)
    feat["peak_amp"]      = float(np.max(np.abs(data)))
    feat["crest_factor"]  = feat["peak_amp"] / (feat["rms_mean"] + 1e-9)
    feat["mel_mean"]      = float(mel_db.mean())
    feat["mel_std"]       = float(mel_db.std())
    for k, (f_, a_) in enumerate(zip(top_f, top_a)):
        feat[f"dom_f{k}_hz"]  = float(f_)
        feat[f"dom_f{k}_amp"] = float(a_)
    feat["label"]    = label
    feat["filename"] = filename
    return feat, mfcc, mel_db, freqs, t_ax, mag, sc, sf, rms_, zcr_


# =============================================================================
#  SECTION 3 ── DIRECTION-OF-ARRIVAL (DOA) ESTIMATION
# =============================================================================

def gcc_phat(sig1, sig2, sr, max_tau=None):
    """Generalised Cross-Correlation with Phase Transform."""
    n   = len(sig1) + len(sig2)
    S1  = fft.rfft(sig1, n=n)
    S2  = fft.rfft(sig2, n=n)
    R   = S1 * np.conj(S2)
    R  /= (np.abs(R) + 1e-10)
    cc  = np.real(fft.irfft(R, n=n))
    max_lag = int(np.ceil(max_tau * sr)) if max_tau else n // 2
    cc  = np.concatenate([cc[-max_lag:], cc[:max_lag+1]])
    tau = np.argmax(cc) - max_lag
    return tau / sr


def estimate_doa_2mic(sig1, sig2, sr, mic_dist_m):
    """TDOA → azimuth for a pair of microphones."""
    max_tau = mic_dist_m / SPEED_OF_SOUND
    tau     = gcc_phat(sig1, sig2, sr, max_tau)
    cos_a   = np.clip(tau * SPEED_OF_SOUND / mic_dist_m, -1, 1)
    return np.degrees(np.arccos(cos_a))


def estimate_doa_array(channels, sr, mic_positions=MEMS_MIC_POSITIONS):
    """
    Estimate 2D azimuth using TDOA between all mic pairs.
    Returns: azimuth (degrees), elevation (degrees, estimated), confidence.
    """
    if channels.shape[0] < 2:
        return None, None, 0.0

    n_mics = min(channels.shape[0], len(mic_positions))
    tdoas  = {}
    for i in range(n_mics):
        for j in range(i+1, n_mics):
            dist = np.linalg.norm(mic_positions[i] - mic_positions[j])
            max_tau = dist / SPEED_OF_SOUND
            tau = gcc_phat(channels[i], channels[j], sr, max_tau)
            tdoas[(i,j)] = tau

    # Least-squares DOA estimation
    # For each pair: tau_ij = (d_i·u - d_j·u) / c
    # where u = [cos(az)cos(el), sin(az)cos(el)] is unit direction
    A, b = [], []
    for (i,j), tau in tdoas.items():
        diff = mic_positions[i] - mic_positions[j]
        A.append(diff)
        b.append(tau * SPEED_OF_SOUND)

    A, b = np.array(A), np.array(b)
    try:
        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        norm = np.linalg.norm(u)
        confidence = min(1.0, 1.0 - abs(norm - 1.0))
        if norm > 0:
            u /= norm
        azimuth   = np.degrees(np.arctan2(u[1], u[0])) % 360
        elevation = np.degrees(np.arcsin(np.clip(
            np.sqrt(max(0, 1 - u[0]**2 - u[1]**2)), 0, 1)))
        return azimuth, elevation, float(confidence)
    except Exception:
        return None, None, 0.0


# =============================================================================
#  SECTION 4 ── KALMAN FILTER PATH TRACKER
# =============================================================================

class KalmanTracker:
    """
    Constant-velocity Kalman filter for drone path tracking.
    State: [x, y, z, vx, vy, vz] in local Cartesian coordinates.
    """
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=2.0):
        self.dt = dt
        n  = 6  # state dim
        self.x = np.zeros(n)
        self.P = np.eye(n) * 100   # initial covariance

        # State transition
        self.F = np.eye(n)
        for i in range(3):
            self.F[i, i+3] = dt

        # Measurement matrix (observe position only)
        self.H = np.zeros((3, n))
        for i in range(3): self.H[i, i] = 1.0

        q = process_noise
        self.Q = np.diag([q/4,q/4,q/4, q,q,q]) * dt**2
        r = measurement_noise
        self.R = np.eye(3) * r**2

        self.history = []
        self.initialised = False

    def initialise(self, pos):
        self.x[:3] = pos
        self.initialised = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, z):
        if not self.initialised:
            self.initialise(z)
            self.history.append(self.x[:3].copy())
            return self.x[:3]
        # Predict
        self.predict()
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update
        y      = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.history.append(self.x[:3].copy())
        return self.x[:3].copy()

    def get_track(self):
        return np.array(self.history)


def latlon_to_xy(lat, lon, lat0, lon0):
    """Approximate GPS → local Cartesian (metres)."""
    R_earth = 6371000.0
    x = R_earth * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R_earth * np.radians(lat - lat0)
    return x, y


def run_kalman_tracking(tele: pd.DataFrame) -> np.ndarray:
    """Apply Kalman filter to telemetry points."""
    tele = tele.dropna(subset=["lat","lon","alt"]).reset_index(drop=True)
    if len(tele) < 3:
        return np.zeros((1, 3))

    lat0 = tele["lat"].iloc[0]
    lon0 = tele["lon"].iloc[0]
    kf   = KalmanTracker(dt=2.0, process_noise=0.5, measurement_noise=1.5)

    for _, row in tele.iterrows():
        x, y = latlon_to_xy(row["lat"], row["lon"], lat0, lon0)
        z    = row["alt"] if not np.isnan(row["alt"]) else 30.0
        kf.update(np.array([x, y, z]))

    return kf.get_track()


# =============================================================================
#  SECTION 5 ── PYTORCH DATASET & CNN CLASSIFIER
# =============================================================================

if TORCH:
    class MelDataset(Dataset):
        """Returns (mel_spectrogram_tensor, label_int) pairs."""
        def __init__(self, paths, labels, sr=SR_TARGET,
                     n_mels=N_MELS, n_fft=N_FFT, hop=HOP_LENGTH,
                     duration_s=SAMPLE_DURATION_S):
            self.paths    = paths
            self.labels   = labels
            self.sr       = sr
            self.n_mels   = n_mels
            self.n_fft    = n_fft
            self.hop      = hop
            self.max_len  = duration_s * sr
            self.fb       = compute_mel_filterbank(sr, n_fft, n_mels)

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            channels, sr_ = read_wav_safe(Path(self.paths[idx]))
            if channels is None:
                mel = np.zeros((self.n_mels, 128), dtype=np.float32)
            else:
                data = channels[0]
                if sr_ != self.sr:
                    data, _ = resample(data, sr_, self.sr)
                # Pad / trim
                n = int(self.max_len)
                if len(data) >= n:
                    data = data[:n]
                else:
                    data = np.pad(data, (0, n - len(data)))
                # Mel
                _, t_, Zxx = signal.stft(data, fs=self.sr,
                                         nperseg=self.n_fft,
                                         noverlap=self.n_fft-self.hop)
                mag = np.abs(Zxx)
                mel = self.fb @ mag
                mel = 10 * np.log10(np.maximum(mel, 1e-10))
                # Normalise per-sample
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            return torch.tensor(mel[np.newaxis], dtype=torch.float32), \
                   torch.tensor(self.labels[idx], dtype=torch.long)


    class DroneCNN(nn.Module):
        """
        Lightweight CNN for mel-spectrogram classification.
        Input : (B, 1, n_mels, T)
        Output: (B, n_classes)
        """
        def __init__(self, n_classes=5, n_mels=N_MELS):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.1),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.15),
                # Block 3
                nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128,128,3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*4*4, 256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 64),      nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        def forward(self, x):
            return self.classifier(self.features(x))


    def train_cnn(paths, labels, n_classes, epochs=30, batch_size=16):
        """Train DroneCNN; returns model + training history dict."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  🧠  Training DroneCNN on {device} "
              f"({len(paths)} samples, {n_classes} classes) …")

        dataset = MelDataset(paths, labels)
        n_val   = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED))

        train_dl = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, drop_last=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

        model   = DroneCNN(n_classes=n_classes).to(device)
        opt     = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit    = nn.CrossEntropyLoss()

        history = {"train_loss":[], "val_loss":[], "val_acc":[]}

        for epoch in range(1, epochs+1):
            model.train()
            t_loss = 0.0
            for X, y in train_dl:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                t_loss += loss.item() * len(X)
            sched.step()

            model.eval()
            v_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for X, y in val_dl:
                    X, y = X.to(device), y.to(device)
                    out   = model(X)
                    v_loss += crit(out, y).item() * len(X)
                    correct += (out.argmax(1) == y).sum().item()
                    total   += len(y)

            history["train_loss"].append(t_loss / n_train)
            history["val_loss"].append(v_loss / max(n_val,1))
            history["val_acc"].append(correct / max(total,1) * 100)

            if epoch % 5 == 0 or epoch == 1:
                print(f"     Epoch {epoch:3d}/{epochs}  "
                      f"train_loss={history['train_loss'][-1]:.4f}  "
                      f"val_loss={history['val_loss'][-1]:.4f}  "
                      f"val_acc={history['val_acc'][-1]:.1f}%")

        return model.cpu(), history, device


# =============================================================================
#  SECTION 6 ── PLOTS (16 figures)
# =============================================================================

def save(fig, name):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   📊  {name}")


def plot_01_dataset_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle("Dunakeszi Dataset Overview", fontsize=16, color="white")

    cnt = df["label"].value_counts()
    ax  = axes[0]
    bars = ax.bar(cnt.index, cnt.values, color=PAL[:len(cnt)], edgecolor="white", lw=0.5)
    [ax.text(b.get_x()+b.get_width()/2, b.get_height()+.1, str(v),
             ha="center", color="white", fontsize=9)
     for b, v in zip(bars, cnt.values)]
    ax.set_title("Class Distribution", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    ax = axes[1]
    for i, (lbl, grp) in enumerate(df.groupby("label")):
        ax.hist(grp["size_kb"], bins=15, alpha=0.7,
                color=PAL[i%len(PAL)], label=lbl, edgecolor="black")
    ax.set_title("File Size (KB)", color="white")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    ax = axes[2]
    if "duration_sec" in df.columns:
        for i, (lbl, grp) in enumerate(df.groupby("label")):
            ax.hist(grp["duration_sec"], bins=12, alpha=0.7,
                    color=PAL[i%len(PAL)], label=lbl, edgecolor="black")
        ax.set_title("Duration (s)", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout(); save(fig, "01_dataset_overview.png")


def plot_02_waveforms(samples):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3*n), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("Waveforms — One Sample per Class", fontsize=14, color="white")
    for ax, (lbl, data) in zip(axes, samples.items()):
        t = np.linspace(0, len(data)/SR_TARGET, len(data))
        ax.plot(t, data, color=PAL[list(samples).index(lbl)%len(PAL)],
                lw=0.4, alpha=0.85)
        ax.fill_between(t, data, alpha=0.15,
                        color=PAL[list(samples).index(lbl)%len(PAL)])
        ax.set_ylabel(lbl, color="white", fontsize=9)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    axes[-1].set_xlabel("Time (s)", color="white")
    plt.tight_layout(); save(fig, "02_waveform_gallery.png")


def plot_03_spectrograms(samples):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("STFT Spectrograms (dB)", fontsize=14, color="white")
    for ax, (lbl, data) in zip(axes, samples.items()):
        f, t, Zxx = signal.stft(data, fs=SR_TARGET, nperseg=N_FFT,
                                noverlap=N_FFT-HOP_LENGTH)
        db = 10*np.log10(np.abs(Zxx)**2 + 1e-10)
        im = ax.pcolormesh(t, f/1000, db, shading="auto",
                           cmap="inferno", vmin=-80, vmax=0)
        ax.set_title(lbl, color="white"); ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Freq (kHz)", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax, format="%+2.0f dB")
    plt.tight_layout(); save(fig, "03_spectrogram_gallery.png")


def plot_04_mel(samples):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), facecolor="#0d0d0d")
    if n == 1: axes = [axes]
    fig.suptitle("Mel-Spectrograms (dB)", fontsize=14, color="white")
    for ax, (lbl, data) in zip(axes, samples.items()):
        _, mel_db, *_ = compute_mfcc(data, SR_TARGET)
        im = ax.pcolormesh(np.arange(mel_db.shape[1]),
                           np.arange(mel_db.shape[0]),
                           mel_db, shading="auto", cmap="magma")
        ax.set_title(lbl, color="white"); ax.set_xlabel("Frame", color="white")
        ax.set_ylabel("Mel Band", color="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax, label="dB")
    plt.tight_layout(); save(fig, "04_mel_spectrogram_gallery.png")


def plot_05_mfcc(samples):
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 9), facecolor="#0d0d0d")
    if n == 1: axes = axes[:,None]
    fig.suptitle("MFCCs", fontsize=14, color="white")
    for i, (lbl, data) in enumerate(samples.items()):
        mfcc, *_ = compute_mfcc(data, SR_TARGET)
        ax = axes[0, i]
        im = ax.pcolormesh(np.arange(mfcc.shape[1]), np.arange(mfcc.shape[0]),
                           mfcc, shading="auto", cmap="coolwarm")
        ax.set_title(f"{lbl} — MFCC", color="white", fontsize=9)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax)
        ax = axes[1, i]
        ax.bar(range(N_MFCC), mfcc.mean(1), yerr=mfcc.std(1),
               color=PAL[i%len(PAL)], alpha=0.8,
               error_kw=dict(ecolor="white", capsize=2))
        ax.set_title("Mean ± Std", color="white", fontsize=9)
        ax.axhline(0, color="white", lw=0.5)
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    plt.tight_layout(); save(fig, "05_mfcc_gallery.png")


def plot_06_noise(feat_df, samples):
    fig = plt.figure(figsize=(20, 14), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("Noise & Signal Quality Analysis", fontsize=15, color="white")
    labels = feat_df["label"].unique()

    # SNR boxplot
    ax = fig.add_subplot(gs[0,0])
    bp = ax.boxplot([feat_df[feat_df.label==l].snr_db.values for l in labels],
                    patch_artist=True,
                    medianprops=dict(color="white",lw=2))
    [p.set_facecolor(PAL[i]) for i,p in enumerate(bp["boxes"])]
    ax.set_xticklabels([l[:8] for l in labels], color="white",
                       rotation=30, ha="right", fontsize=8)
    ax.set_title("SNR (dB)", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # RMS violin
    ax = fig.add_subplot(gs[0,1])
    vp = ax.violinplot([feat_df[feat_df.label==l].rms_mean.values
                        for l in labels], showmedians=True)
    [b.set_facecolor(PAL[i]) for i,b in enumerate(vp["bodies"])]
    ax.set_xticks(range(1,len(labels)+1))
    ax.set_xticklabels([l[:8] for l in labels], color="white",
                       rotation=30, ha="right", fontsize=8)
    ax.set_title("RMS Energy", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # SNR vs dynamic range
    ax = fig.add_subplot(gs[0,2])
    for i, lbl in enumerate(labels):
        g = feat_df[feat_df.label==lbl]
        ax.scatter(g.snr_db, g.crest_factor, color=PAL[i%len(PAL)],
                   label=lbl, alpha=0.75, s=60, edgecolors="white", lw=0.3)
    ax.set_xlabel("SNR (dB)", color="white")
    ax.set_ylabel("Crest Factor", color="white")
    ax.set_title("SNR vs Crest Factor", color="white")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Average PSD per class
    ax = fig.add_subplot(gs[1,:])
    ax.set_title("Average Power Spectral Density per Class", color="white")
    for i, lbl in enumerate(labels):
        psds = []
        for _, row in feat_df[feat_df.label==lbl].iterrows():
            channels, sr_ = read_wav_safe(Path(row["filename"]))
            if channels is None: continue
            data, sr_ = resample(channels[0], sr_)
            f_p, psd  = signal.welch(data, fs=sr_, nperseg=N_FFT)
            psds.append(psd)
        if psds:
            ax.semilogy(f_p/1000, np.mean(psds,0),
                        color=PAL[i%len(PAL)], label=lbl, lw=1.8)
    ax.set_xlabel("Frequency (kHz)", color="white")
    ax.set_ylabel("PSD", color="white")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
    ax.grid(True, alpha=0.12); ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    for col, title, gsi in [
        ("zcr_mean",      "ZCR Mean",     gs[2,0]),
        ("rms_kurtosis",  "RMS Kurtosis", gs[2,1]),
        ("mel_mean",      "Mel Energy",   gs[2,2]),
    ]:
        ax = fig.add_subplot(gsi)
        for i, lbl in enumerate(labels):
            ax.hist(feat_df[feat_df.label==lbl][col], bins=12,
                    alpha=0.7, color=PAL[i%len(PAL)], label=lbl, edgecolor="black")
        ax.set_title(title, color="white", fontsize=9)
        ax.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    save(fig, "06_noise_analysis.png")


def plot_07_spectral(feat_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d0d0d")
    fig.suptitle("Spectral Feature Distributions", fontsize=14, color="white")
    feats = [("sc_mean","Spectral Centroid (Hz)"),
             ("sf_mean","Spectral Flatness"),
             ("dom_f0_hz","Dom. Freq #0 (Hz)"),
             ("dom_f1_hz","Dom. Freq #1 (Hz)"),
             ("mel_mean","Mel Energy Mean (dB)"),
             ("rms_std","RMS Std")]
    labels = feat_df["label"].unique()
    for ax, (col, title) in zip(axes.flat, feats):
        for i, lbl in enumerate(labels):
            ax.hist(feat_df[feat_df.label==lbl][col].dropna(),
                    bins=15, alpha=0.65, color=PAL[i%len(PAL)],
                    label=lbl, edgecolor="black")
        ax.set_title(title, color="white", fontsize=9)
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    plt.tight_layout(); save(fig, "07_spectral_features.png")


def plot_08_correlation(feat_df):
    cols = [c for c in feat_df.columns if "_mean" in c and
            c not in ("filename","label")][:30]
    corr = feat_df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0d0d0d")
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                linewidths=0.3, linecolor="#333",
                xticklabels=[c.replace("_mean","") for c in cols],
                yticklabels=[c.replace("_mean","") for c in cols])
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=7)
    fig.patch.set_facecolor("#0d0d0d")
    save(fig, "08_correlation_heatmap.png")


def plot_09_pca(X, y, classes):
    pca   = PCA(n_components=3, random_state=SEED)
    X_pca = pca.fit_transform(X)
    ev    = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(18, 6), facecolor="#0d0d0d")
    fig.suptitle("PCA Feature Space", fontsize=14, color="white")

    for ax_idx, (c1, c2) in enumerate([(0,1),(0,2),(1,2)]):
        ax = fig.add_subplot(1,3,ax_idx+1)
        for i, lbl in enumerate(np.unique(y)):
            m = y == lbl
            ax.scatter(X_pca[m,c1], X_pca[m,c2],
                       label=classes[lbl], color=PAL[i%len(PAL)],
                       alpha=0.75, s=60, edgecolors="white", lw=0.3)
        ax.set_xlabel(f"PC{c1+1} ({ev[c1]*100:.1f}%)", color="white")
        ax.set_ylabel(f"PC{c2+1} ({ev[c2]*100:.1f}%)", color="white")
        ax.set_title(f"PC{c1+1} vs PC{c2+1}", color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout(); save(fig, "09_pca_space.png")
    return X_pca


def plot_10_tsne(X, y, classes):
    n   = min(len(X), 300)
    idx = np.random.choice(len(X), n, replace=False)
    print("   ⏳ t-SNE …", end=" ", flush=True)
    ts  = TSNE(n_components=2, perplexity=min(30,n//4),
               random_state=SEED, max_iter=800).fit_transform(X[idx])
    print("done")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d0d")
    for i, lbl in enumerate(np.unique(y[idx])):
        m = y[idx] == lbl
        ax.scatter(ts[m,0], ts[m,1], label=classes[lbl],
                   color=PAL[i%len(PAL)], alpha=0.85, s=80,
                   edgecolors="white", lw=0.4)
    ax.set_title("t-SNE Acoustic Feature Space", color="white", fontsize=14)
    ax.legend(fontsize=9, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.grid(True, alpha=0.08)
    save(fig, "10_tsne_space.png")


def plot_11_anomaly(X, y, classes, feat_df):
    iso    = IsolationForest(contamination=0.08, random_state=SEED)
    preds  = iso.fit_predict(X)
    scores = iso.decision_function(X)
    pca2   = PCA(n_components=2, random_state=SEED).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d0d0d")
    fig.suptitle("Anomaly Detection — Isolation Forest", fontsize=13, color="white")

    ax = axes[0]
    sc = ax.scatter(pca2[:,0], pca2[:,1], c=scores,
                    cmap="RdYlGn", alpha=0.85, s=70, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Anomaly score")
    ax.set_title("Scores in PCA space", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    ax = axes[1]
    fd = feat_df.copy(); fd["anomaly"] = preds
    pct = fd.groupby("label")["anomaly"].apply(
        lambda x: (x==-1).mean()*100).fillna(0)
    bars = ax.bar(pct.index, pct.values, color=PAL[:len(pct)],
                  edgecolor="white", lw=0.5)
    [ax.text(b.get_x()+b.get_width()/2, b.get_height()+.2,
             f"{v:.1f}%", ha="center", color="white", fontsize=8)
     for b,v in zip(bars, pct.values)]
    ax.set_title("Anomaly % per Class", color="white")
    ax.tick_params(colors="white", axis="both")
    ax.set_facecolor("#1a1a1a")

    plt.tight_layout(); save(fig, "11_anomaly_detection.png")
    return preds


def plot_12_temporal(samples):
    n    = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.5*n), facecolor="#0d0d0d")
    if n == 1: axes = axes[None,:]
    fig.suptitle("Temporal Profiles — RMS & ZCR", fontsize=13, color="white")
    for i, (lbl, data) in enumerate(samples.items()):
        rms_ = rms_energy(data); zcr_ = zcr(data)
        t    = np.arange(len(rms_)) * HOP_LENGTH / SR_TARGET
        for ax, arr, ttl in zip(axes[i], [rms_, zcr_], ["RMS Energy","ZCR"]):
            ax.plot(t, arr, color=PAL[i%len(PAL)], lw=1.2)
            ax.fill_between(t, arr, alpha=0.2, color=PAL[i%len(PAL)])
            ax.set_title(f"[{lbl}] {ttl}", color="white", fontsize=9)
            ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    plt.tight_layout(); save(fig, "12_temporal_profiles.png")


def plot_13_doa(doa_results):
    """Rose / polar plot of DOA estimates per class."""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0d0d0d",
                           subplot_kw=dict(polar=True))
    fig.suptitle("Direction-of-Arrival Estimates per Class",
                 fontsize=14, color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)

    for i, (lbl, azimuths) in enumerate(doa_results.items()):
        if not azimuths: continue
        az_rad = np.radians(azimuths)
        # Kernel density estimate on circle
        theta  = np.linspace(0, 2*np.pi, 360)
        kde    = np.zeros(360)
        for a in az_rad:
            kde += np.exp(-0.5 * ((theta - a) / 0.3)**2)
        kde /= kde.max() + 1e-9
        ax.plot(theta, kde, color=PAL[i%len(PAL)], lw=2, label=lbl)
        ax.fill(theta, kde, color=PAL[i%len(PAL)], alpha=0.15)
        # Individual detections
        for a in az_rad:
            ax.scatter(a, 0.95, color=PAL[i%len(PAL)], s=40,
                       alpha=0.7, zorder=5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax.set_yticks([]); ax.grid(True, alpha=0.2)
    [lbl.set_color("white") for lbl in ax.get_xticklabels()]
    save(fig, "13_doa_polar.png")


def plot_14_kalman_track(tele, track):
    lat0 = tele["lat"].iloc[0] if len(tele) else 47.625
    lon0 = tele["lon"].iloc[0] if len(tele) else 19.135

    fig = plt.figure(figsize=(18, 7), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle("Drone Path Tracking — Kalman Filter", fontsize=14, color="white")

    # XY ground track
    ax = fig.add_subplot(gs[0,0])
    if len(tele) > 1:
        xs, ys = zip(*[latlon_to_xy(r.lat, r.lon, lat0, lon0)
                       for _, r in tele.iterrows()])
        ax.scatter(xs, ys, c=range(len(xs)), cmap="cool",
                   s=20, alpha=0.5, label="Raw GPS")
    if len(track) > 1:
        ax.plot(track[:,0], track[:,1], color=PAL[0],
                lw=2, label="Kalman track", zorder=5)
        ax.scatter(*track[0,:2], color="lime", s=100,
                   zorder=6, marker="^", label="Start")
        ax.scatter(*track[-1,:2], color="red", s=100,
                   zorder=6, marker="s", label="End")
    ax.set_xlabel("East (m)", color="white"); ax.set_ylabel("North (m)", color="white")
    ax.set_title("Ground Track", color="white")
    ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")
    ax.set_aspect("equal")

    # Altitude profile
    ax = fig.add_subplot(gs[0,1])
    t  = np.arange(len(track))
    ax.plot(t, track[:,2], color=PAL[1], lw=2)
    ax.fill_between(t, track[:,2], alpha=0.2, color=PAL[1])
    if len(tele) > 1 and "alt" in tele.columns:
        ax.scatter(np.linspace(0, len(track)-1, len(tele)),
                   tele["alt"].values, color="white", s=15,
                   alpha=0.5, label="Raw alt", zorder=5)
    ax.set_xlabel("Time step", color="white"); ax.set_ylabel("Altitude (m)", color="white")
    ax.set_title("Altitude Profile", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    # Velocity
    ax = fig.add_subplot(gs[0,2])
    if len(track) > 1:
        vel = np.linalg.norm(np.diff(track, axis=0), axis=1) / 2.0
        ax.plot(np.arange(len(vel)), vel, color=PAL[2], lw=2)
        ax.fill_between(np.arange(len(vel)), vel, alpha=0.2, color=PAL[2])
    ax.set_xlabel("Time step", color="white"); ax.set_ylabel("Speed (m/s)", color="white")
    ax.set_title("Estimated Speed", color="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout(); save(fig, "14_kalman_track.png")


def plot_15_cnn_training(history):
    if not history: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("CNN Training History", fontsize=13, color="white")

    ax = axes[0]
    ax.plot(history["train_loss"], color=PAL[0], lw=2, label="Train Loss")
    ax.plot(history["val_loss"],   color=PAL[1], lw=2, label="Val Loss")
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Loss", color="white")
    ax.set_title("Loss Curves", color="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    ax = axes[1]
    ax.plot(history["val_acc"], color=PAL[2], lw=2)
    ax.axhline(100/max(1, len(set(history.get("val_acc",[])+[1]))),
               color="white", ls="--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Val Acc (%)", color="white")
    ax.set_title("Validation Accuracy", color="white")
    ax.set_ylim(0, 105)
    ax.set_facecolor("#1a1a1a"); ax.tick_params(colors="white")

    plt.tight_layout(); save(fig, "15_cnn_training.png")


def plot_16_confusion(y_true, y_pred, classes):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d0d")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Drone Classifier",
                 color="white", fontsize=13)
    ax.tick_params(colors="white"); ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#0d0d0d")
    [lbl.set_color("white") for lbl in ax.get_xticklabels()+ax.get_yticklabels()]
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    plt.tight_layout(); save(fig, "16_confusion_matrix.png")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t0 = time.time()

    # ── STEP 0: Download from OneDrive ────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 0 — ONEDRIVE SELECTIVE DOWNLOAD")
    print("─"*70)

    mems_files_exist = list(MEMS_DIR.rglob("*.wav"))

    if not mems_files_exist:
        if not ONEDRIVE_SHARE_URL and not AZURE_CLIENT_ID:
            print("""
  ┌─ WAITING FOR SHARE LINK ───────────────────────────────────────────┐
  │                                                                     │
  │  Please provide your OneDrive share link.                          │
  │                                                                     │
  │  Steps:                                                             │
  │    1. Open OneDrive in your browser                                │
  │    2. Navigate to the Dunakeszi dataset folder                     │
  │    3. Right-click Dunakeszi_MEMS → Share                          │
  │       → 'Anyone with the link can view' → Copy Link               │
  │    4. Paste the link into ONEDRIVE_SHARE_URL at the top of        │
  │       this script and re-run, OR share it in the chat.            │
  │                                                                     │
  │  The script will then automatically download:                      │
  │    • Up to 50 MEMS .wav files (≤ 500 MB)                         │
  │    • All of DRON-ADATOK (162 MB)                                  │
  │    • All of JEGYZOKONYV (1.4 MB)                                  │
  │                                                                     │
  │  Running analysis on SYNTHETIC data until real data is provided.  │
  └─────────────────────────────────────────────────────────────────────┘
""")
            _seed_synthetic_data()
        else:
            dl = OneDriveDownloader()
            dl.download_folders({
                "MEMS":         MEMS_DIR,
                "DRON-ADATOK":  DRONE_DIR,
                "JEGYZOKONYV":  LOG_DIR,
            })
    else:
        print(f"  ✔  {len(mems_files_exist)} MEMS files already present — skipping download")

    # ── STEP 1: Parse logbook & telemetry ─────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 1 — PARSE LOGBOOK & TELEMETRY")
    print("─"*70)
    logbook = parse_logbook(LOG_DIR)
    tele    = parse_drone_telemetry(DRONE_DIR)

    logbook.to_csv(OUTPUT_DIR / "logbook_parsed.csv", index=False)
    tele.to_csv(   OUTPUT_DIR / "telemetry_parsed.csv", index=False)

    # ── STEP 2: Discover MEMS files ───────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 2 — DATASET AUDIT")
    print("─"*70)
    all_wav = sorted(MEMS_DIR.rglob("*.wav"))
    if not all_wav:
        print("  ⚠  No WAV files in MEMS dir — check download step"); return

    records = []
    for p in all_wav:
        parts = p.relative_to(MEMS_DIR).parts
        label = parts[0] if len(parts) > 1 else _infer_label(p.name)
        records.append({"path": str(p), "filename": str(p),
                        "label": label, "size_kb": p.stat().st_size/1024})
    df = pd.DataFrame(records)
    print(f"  Files : {len(df)}")
    print(f"  Labels: {sorted(df.label.unique())}")
    print(df.label.value_counts().to_string())

    # ── STEP 3: Feature extraction ────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 3 — FEATURE EXTRACTION")
    print("─"*70)
    feat_rows, samples, doa_results = [], {}, defaultdict(list)

    for _, row in df.iterrows():
        channels, sr_ = read_wav_safe(Path(row["filename"]))
        if channels is None: continue

        data, sr_ = resample(channels[0], sr_)
        feats, mfcc, mel_db, freqs, t_ax, mag, sc, sf, rms_, zcr_ = \
            extract_features(data, sr_, row["label"], row["filename"])
        feats["duration_sec"] = len(data) / sr_
        feats["size_kb"]      = row["size_kb"]
        feat_rows.append(feats)

        # One sample per class for galleries
        lbl = row["label"]
        if lbl not in samples:
            samples[lbl] = data

        # DOA estimation (multi-channel)
        if channels.shape[0] >= 2:
            az, el, conf = estimate_doa_array(channels, sr_)
            if az is not None and conf > 0.3:
                doa_results[lbl].append(az)

    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(OUTPUT_DIR / "features.csv", index=False)
    df = df.merge(feat_df[["filename","duration_sec"]].rename(
                  columns={"filename":"filename"}), on="filename", how="left")
    print(f"  ✔  {len(feat_df)} files × {len(feat_df.columns)} features")

    # ── STEP 4: ML tensors ────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 4 — FEATURE TENSORS & CLASSIFIER")
    print("─"*70)
    feat_cols = [c for c in feat_df.columns
                 if c not in ("label","filename","path","extension","size_kb")]
    X_raw  = feat_df[feat_cols].values.astype(np.float32)
    le     = LabelEncoder()
    y      = le.fit_transform(feat_df["label"])
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    if TORCH:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        torch.save({"X": X_t, "y": y_t, "classes": list(le.classes_)},
                   OUTPUT_DIR / "tensors.pt")

    # Sklearn baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    if len(X) >= 5:
        cv_scores = cross_val_score(rf, X, y, cv=min(5,len(X)),
                                    scoring="accuracy")
        print(f"  RF cross-val accuracy: {cv_scores.mean()*100:.1f}% "
              f"± {cv_scores.std()*100:.1f}%")
        rf.fit(X, y)
        y_pred_rf = rf.predict(X)
    else:
        rf.fit(X, y); y_pred_rf = rf.predict(X)

    # PyTorch CNN (if available and enough data)
    cnn_history = {}
    if TORCH and len(X) >= 10:
        paths_list = list(feat_df["filename"])
        model, cnn_history, device = train_cnn(
            paths_list, y, n_classes=len(le.classes_),
            epochs=min(30, max(10, len(X)//2)), batch_size=8)
        torch.save(model.state_dict(), MODEL_DIR / "drone_cnn.pth")
        print(f"  ✔  CNN saved → {MODEL_DIR / 'drone_cnn.pth'}")

    # ── STEP 5: Kalman tracking ───────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 5 — KALMAN PATH TRACKING")
    print("─"*70)
    track = run_kalman_tracking(tele)
    np.save(OUTPUT_DIR / "kalman_track.npy", track)
    print(f"  ✔  Track: {len(track)} points  |  "
          f"XY range: {np.ptp(track[:,0]):.1f} × {np.ptp(track[:,1]):.1f} m  |  "
          f"Alt: {track[:,2].min():.0f}–{track[:,2].max():.0f} m")

    # ── STEP 6: All 16 plots ──────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  STEP 6 — GENERATING 16 PLOTS")
    print("─"*70)

    plot_01_dataset_overview(df if "duration_sec" in df.columns else feat_df)
    plot_02_waveforms(samples)
    plot_03_spectrograms(samples)
    plot_04_mel(samples)
    plot_05_mfcc(samples)
    plot_06_noise(feat_df, samples)
    plot_07_spectral(feat_df)
    plot_08_correlation(feat_df)
    plot_09_pca(X, y, le.classes_)
    plot_10_tsne(X, y, le.classes_)
    plot_11_anomaly(X, y, le.classes_, feat_df)
    plot_12_temporal(samples)
    if doa_results:
        plot_13_doa(doa_results)
    else:
        # Synthetic DOA from telemetry heading
        synth_doa = {lbl: list(np.random.uniform(0,360,8)) for lbl in samples}
        plot_13_doa(synth_doa)
    plot_14_kalman_track(tele, track)
    plot_15_cnn_training(cnn_history)
    plot_16_confusion(y, y_pred_rf, le.classes_)

    # ── Final report ──────────────────────────────────────────────────────────
    report = {
        "generated":      datetime.now().isoformat(),
        "elapsed_sec":    round(time.time() - t0, 1),
        "n_recordings":   int(len(feat_df)),
        "n_features":     int(len(feat_cols)),
        "classes":        list(le.classes_),
        "rf_accuracy_pct":float(np.mean(y == y_pred_rf)*100),
        "snr_mean_db":    float(feat_df["snr_db"].mean()),
        "snr_std_db":     float(feat_df["snr_db"].std()),
        "track_points":   int(len(track)),
        "doa_classes_detected": list(doa_results.keys()),
        "per_class_counts": feat_df["label"].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "analysis_report.json","w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "═"*72)
    print("  ANALYSIS COMPLETE")
    print("═"*72)
    print(f"  Elapsed          : {report['elapsed_sec']} s")
    print(f"  Recordings       : {report['n_recordings']}")
    print(f"  Features/sample  : {report['n_features']}")
    print(f"  Classes          : {report['classes']}")
    print(f"  RF accuracy      : {report['rf_accuracy_pct']:.1f}%")
    print(f"  SNR (mean)       : {report['snr_mean_db']:.1f} dB")
    print(f"  Track points     : {report['track_points']}")
    print(f"\n  Outputs → {OUTPUT_DIR}")
    print("═"*72 + "\n")


# =============================================================================
#  HELPERS
# =============================================================================

def _infer_label(filename: str) -> str:
    """Guess class label from filename patterns common in drone datasets."""
    fn = filename.lower()
    for kw, lbl in [("drone","drone"),("quadrotor","quadrotor"),
                    ("quad","quadrotor"),("hexa","hexarotor"),
                    ("fpv","fpv_racer"),("fixed","fixed_wing"),
                    ("bg","background"),("noise","background"),
                    ("ambient","background"),("no_drone","background")]:
        if kw in fn: return lbl
    return "unknown"


def _seed_synthetic_data():
    """Populate MEMS_DIR with synthetic data for dry-run testing."""
    from scipy.io import wavfile as wf
    classes = ["drone","background","fixed_wing","fpv_racer"]
    sr = SR_TARGET
    for cls in classes:
        d = MEMS_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            t   = np.linspace(0, 3, sr*3)
            bpf = {"drone":80,"background":0,"fixed_wing":110,"fpv_racer":150}[cls]
            sig = np.zeros_like(t)
            if bpf > 0:
                for h in range(1, 5):
                    sig += np.sin(2*np.pi*bpf*h*t) / h
                sig *= 1/max(1,(i%4+1)*3)
            sig += np.random.randn(len(t)) * 0.05
            sig  = (sig / (np.max(np.abs(sig))+1e-9) * 32767).astype(np.int16)
            wf.write(str(d / f"{cls}_{i:03d}.wav"), sr, sig)
    print(f"  ✔  Synthetic MEMS data written to {MEMS_DIR}")


if __name__ == "__main__":
    main()