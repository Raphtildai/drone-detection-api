#!/usr/bin/env python3
# dunakeszi_benchmark.py  v2
"""
Dunakeszi Field Dataset — Modular Benchmark Evaluator
======================================================
All key decisions (audio source, ground-truth strategy, model path) are
runtime arguments.  Nothing is hard-wired.

Quick starts
------------
  # Heuristic only, discover everything automatically:
  python dunakeszi_benchmark.py --dataset-root ./dunakeszi_data

  # Full CNN run, MEMS only, flight-window GT:
  python dunakeszi_benchmark.py \\
      --dataset-root ./dunakeszi_data \\
      --model-dir    /path/to/drone_v15/models \\
      --source       mems \\
      --gt-strategy  flight_window

  # Both sources, ALL four GT strategies, side-by-side report:
  python dunakeszi_benchmark.py \\
      --dataset-root ./dunakeszi_data \\
      --source       both \\
      --gt-strategy  all

Audio sources
-------------
  mems   12 × 8 kHz WAV files (Dunakeszi_MEMS/Audio/)
  bruel  14-ch 192 kHz flight-window .npy caches (dunakeszi_audit_output/)
  both   run both tracks independently

Ground-truth strategies
-----------------------
  flight_window   segments overlapping 12:57:21–13:04:09 UTC = drone
  gpx_altitude    drone when GPX elevation > --gpx-alt-threshold (default 5 m AGL)
  gpx_speed       drone when GPX speed > --gpx-speed-threshold (default 1 m/s)
  conservative    segment must satisfy ALL available GT signals simultaneously
  all             run all four strategies and produce a comparison table

Outputs (--output dir, default ./dunakeszi_benchmark_out)
---------------------------------------------------------
  {source}_{gt_strategy}_segments.csv    per-segment scores + labels
  {source}_{gt_strategy}_report.json     F1 / AUROC / PR / confusion numbers
  plots/
    {source}_{gt_strategy}_timeline.png
    {source}_{gt_strategy}_roc_pr.png
    {source}_{gt_strategy}_confusion.png
    {source}_{gt_strategy}_score_dist.png
    comparison_table.png                 (only when --gt-strategy all)
  benchmark_summary.json                 top-line numbers for all runs
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, csv, json, math, time, random, warnings, argparse, logging
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, confusion_matrix,
        roc_curve, precision_recall_curve,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


# =============================================================================
#  FLIGHT CONSTANTS  (from audit_v2 / messages.log)
# =============================================================================

FLIGHT_TAKEOFF_UTC = datetime(2025, 10, 20, 12, 57, 21, tzinfo=timezone.utc)
FLIGHT_LANDING_UTC = datetime(2025, 10, 20, 13,  4,  9,  tzinfo=timezone.utc)
FLIGHT_DURATION_S  = (FLIGHT_LANDING_UTC - FLIGHT_TAKEOFF_UTC).total_seconds()

BRUEL_SR       = 192_000
BRUEL_CHANNELS = 14
MEMS_SR        = 8_000
MODEL_SR       = 22_050

# Brüel file wall-clock starts (from audit_v2 config)
BRUEL_FILE_STARTS: Dict[str, datetime] = {
    "251020VITEMOROM1AT01I.wav": datetime(2025, 10, 20, 12, 50, 34, tzinfo=timezone.utc),
    "251020VITEMOROM1AT01J.wav": datetime(2025, 10, 20, 12, 59, 28, tzinfo=timezone.utc),
}

# ── Plot theme ────────────────────────────────────────────────────────────────
P = {
    "bg":    "#0d1117", "panel": "#161b22", "grid":  "#21262d",
    "text":  "#c9d1d9", "muted": "#484f58",
    "drone": "#00d4ff", "ok":    "#3fb950", "err":   "#f85149",
    "warn":  "#d29922", "purp":  "#a371f7",
}
plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["panel"],
    "axes.edgecolor": P["grid"], "axes.labelcolor": P["text"],
    "xtick.color": P["text"], "ytick.color": P["text"],
    "text.color": P["text"], "grid.color": P["grid"],
    "grid.alpha": 0.4, "legend.facecolor": P["panel"],
    "legend.edgecolor": P["grid"], "savefig.facecolor": P["bg"],
    "font.family": "monospace",
})


# =============================================================================
#  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dunakeszi benchmark evaluator — all decisions at runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Paths
    p.add_argument("--dataset-root", default=".",
                   help="Root folder of the Dunakeszi dataset (auto-discovers sub-dirs)")
    p.add_argument("--mems-dir",  default=None,
                   help="Override: path to MEMS Audio WAV folder")
    p.add_argument("--bruel-npy", default=None,
                   help="Override: path to dunakeszi_audit_output (*.npy files)")
    p.add_argument("--bruel-wav", default=None,
                   help="Override: path to raw Brüel WAV folder (for negatives)")
    p.add_argument("--gpx-csv",   default=None,
                   help="Override: path to gpx_combined.csv from audit output")
    p.add_argument("--model-dir", default=None,
                   help="Path to drone_v15 models/ folder (best_detection.pth)")
    p.add_argument("--output",    default="./dunakeszi_benchmark_out",
                   help="Output directory for all results")

    # What to evaluate
    p.add_argument("--source",      default="both",
                   choices=["mems", "bruel", "both"],
                   help="Audio source(s) to evaluate")
    p.add_argument("--gt-strategy", default="all",
                   choices=["flight_window", "gpx_altitude",
                             "gpx_speed", "conservative", "all"],
                   help="Ground-truth labelling strategy")

    # GT thresholds
    p.add_argument("--gpx-alt-threshold",   type=float, default=5.0,
                   help="AGL elevation (m) above which segment = drone [gpx_altitude]")
    p.add_argument("--gpx-speed-threshold", type=float, default=1.0,
                   help="Speed (m/s) above which segment = drone [gpx_speed]")

    # Segmentation
    p.add_argument("--segment-sec", type=float, default=1.0)
    p.add_argument("--hop-sec",     type=float, default=0.5)
    p.add_argument("--context-min", type=float, default=10.0,
                   help="Minutes of non-flight audio to include from MEMS files")
    p.add_argument("--mems-start",  default=None,
                   help="UTC ISO datetime when MEMS recording started "
                        "(e.g. 2025-10-20T10:00:00).  Parsed from filename if absent.")

    # Brüel channel
    p.add_argument("--bruel-channel", type=int, default=-1,
                   help="Brüel channel index (-1 = average all)")

    # Model
    p.add_argument("--no-model",    action="store_true",
                   help="Skip CNN; heuristic scoring only")
    p.add_argument("--cnn-weight",  type=float, default=0.80,
                   help="Weight of CNN score in fused probability (0–1)")
    p.add_argument("--threshold",   type=float, default=None,
                   help="Fixed decision threshold (default: find best F1 on data)")

    p.add_argument("--seed", type=int, default=42)
    return p


# =============================================================================
#  PATH DISCOVERY
# =============================================================================

def discover_paths(args) -> Dict[str, Optional[Path]]:
    """
    Walk the dataset root and try to find the standard sub-directories
    produced by audit_v2.  User overrides always win.
    """
    root = Path(args.dataset_root)

    def find(candidates: List[Path]) -> Optional[Path]:
        for c in candidates:
            if c.exists():
                return c
        return None

    mems_dir = (Path(args.mems_dir) if args.mems_dir else
                find([root / "Dunakeszi_MEMS" / "Audio",
                      root / "Dunakeszi_MEMS",
                      root / "MEMS" / "Audio",
                      root / "MEMS"]))

    bruel_npy = (Path(args.bruel_npy) if args.bruel_npy else
                 find([root / "dunakeszi_audit_output",
                       root / "dunakeszi_audit_ws",
                       root / "audit_output"]))

    bruel_wav = (Path(args.bruel_wav) if args.bruel_wav else
                 find([root / "Dunakeszi_BRUEL_VIDEO",
                       root / "BRUEL",
                       root / "dunakeszi_data" / "Dunakeszi_BRUEL_VIDEO"]))

    gpx_csv = (Path(args.gpx_csv) if args.gpx_csv else
               find([root / "dunakeszi_audit_output" / "gpx_combined.csv",
                     root / "gpx_combined.csv"]))

    model_dir = (Path(args.model_dir) if args.model_dir else
                 find([Path("/tmp/drone_v15/models"),
                       Path.home() / "drone_v15" / "models",
                       root / "models"]))

    paths = {
        "mems_dir":  mems_dir,
        "bruel_npy": bruel_npy,
        "bruel_wav": bruel_wav,
        "gpx_csv":   gpx_csv,
        "model_dir": model_dir,
    }

    print("\n  Discovered paths:")
    for k, v in paths.items():
        status = "✓" if (v and v.exists()) else "✗"
        print(f"    {status}  {k:12s}: {v}")
    return paths


# =============================================================================
#  GPX LOADER
# =============================================================================

def load_gpx_csv(gpx_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if gpx_csv is None or not gpx_csv.exists():
        return None
    try:
        df = pd.read_csv(gpx_csv, parse_dates=["time"])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        print(f"  GPX: {len(df)} trackpoints loaded from {gpx_csv.name}")
        return df
    except Exception as e:
        print(f"  ⚠️  GPX load failed: {e}")
        return None


def gpx_value_at(gpx_df: pd.DataFrame, ts: datetime,
                 col: str, window_s: float = 5.0) -> Optional[float]:
    """Interpolate a GPX column value at a given UTC timestamp."""
    if gpx_df is None or col not in gpx_df.columns:
        return None
    ts_pd = pd.Timestamp(ts)
    lo = ts_pd - pd.Timedelta(seconds=window_s)
    hi = ts_pd + pd.Timedelta(seconds=window_s)
    sub = gpx_df[(gpx_df["time"] >= lo) & (gpx_df["time"] <= hi)].dropna(subset=[col])
    if sub.empty:
        return None
    return float(sub[col].median())


# =============================================================================
#  GROUND-TRUTH STRATEGIES
# =============================================================================

GT_STRATEGIES = ["flight_window", "gpx_altitude", "gpx_speed", "conservative"]


def assign_gt_label(
    seg_mid_utc: datetime,
    strategy: str,
    gpx_df: Optional[pd.DataFrame],
    alt_threshold: float,
    speed_threshold: float,
) -> int:
    """
    Return 1 (drone) or 0 (non_drone) for a segment midpoint timestamp.

    flight_window  — inside takeoff…landing UTC window
    gpx_altitude   — GPX elevation at that timestamp > alt_threshold
    gpx_speed      — GPX speed > speed_threshold
    conservative   — ALL available signals agree it is drone
    """
    fw = FLIGHT_TAKEOFF_UTC <= seg_mid_utc <= FLIGHT_LANDING_UTC

    if strategy == "flight_window":
        return int(fw)

    alt = gpx_value_at(gpx_df, seg_mid_utc, "elevation") if gpx_df is not None else None
    spd = gpx_value_at(gpx_df, seg_mid_utc, "speed")     if gpx_df is not None else None

    if strategy == "gpx_altitude":
        if alt is None:
            return int(fw)          # fall back to flight_window if no GPX data
        return int(alt > alt_threshold)

    if strategy == "gpx_speed":
        if spd is None:
            return int(fw)
        return int(spd > speed_threshold)

    if strategy == "conservative":
        # Must satisfy ALL signals that are available
        signals = [fw]
        if alt is not None:
            signals.append(alt > alt_threshold)
        if spd is not None:
            signals.append(spd > speed_threshold)
        return int(all(signals))

    raise ValueError(f"Unknown GT strategy: {strategy}")


# =============================================================================
#  AUDIO UTILITIES
# =============================================================================

def load_wav(path: Path, max_sec: float = None) -> Tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if max_sec:
        data = data[:int(max_sec * sr)]
    return data, int(sr)


def resample(y: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return y
    from math import gcd
    g = gcd(src, dst)
    return sp_signal.resample_poly(y, dst // g, src // g).astype(np.float32)


def windows(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    starts = list(range(0, max(1, n - win + 1), hop))
    if n > win and starts[-1] != n - win:
        starts.append(n - win)
    elif n <= win:
        starts = [0]
    return [(s, min(s + win, n)) for s in starts]


def pad_or_trim(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def parse_mems_start(path: Path, fallback: datetime) -> datetime:
    import re
    m = re.search(r"(\d{8})[_\-]?(\d{6})", path.stem)
    if m:
        try:
            return datetime.strptime(
                m.group(1) + m.group(2), "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback


# =============================================================================
#  HEURISTIC SCORING  (no librosa dependency)
# =============================================================================

def heuristic_score(y: np.ndarray, sr: int) -> float:
    if len(y) == 0:
        return 0.0
    y = y.astype(np.float64)
    rms    = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    peak   = float(np.max(np.abs(y)) + 1e-8)
    rms_db = 20.0 * math.log10(rms)
    cf     = peak / rms                        # crest factor

    if cf > 8.5:                               # impulsive → not drone
        return 0.10

    nperseg = min(512, max(32, len(y) // 4))
    f, psd  = sp_signal.welch(y, fs=sr, nperseg=nperseg)
    total   = psd.sum() + 1e-10
    centroid = float(np.sum(f * psd) / total)
    flatness = float(np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd + 1e-10))

    band = (f >= 80) & (f <= 2500)
    drone_ratio = float(psd[band].sum() / total) if band.any() else 0.0

    dom_freq = float(f[np.argmax(psd)])

    e_sc  = float(np.clip((rms_db + 45) / 25, 0, 1))
    c_sc  = 1.0 if 120 <= centroid <= 4000 else 0.35
    fl_sc = float(np.clip(1 - flatness / 0.6, 0, 1))
    d_sc  = 1.0 if 70 <= dom_freq <= 600 else 0.3
    b_sc  = float(np.clip(drone_ratio / 0.45, 0, 1))

    raw  = 0.25*e_sc + 0.20*c_sc + 0.15*fl_sc + 0.20*d_sc + 0.20*b_sc
    prob = 1.0 / (1.0 + math.exp(-8.0 * (raw - 0.50)))
    return float(np.clip(prob, 0, 1))


# =============================================================================
#  CNN MODEL
# =============================================================================

_model_cache: Dict[str, Any] = {}


def load_model(model_dir: Optional[Path]):
    if not TORCH_OK or not LIBROSA_OK or model_dir is None:
        return None, None, 0.62

    ckpt = model_dir / "best_detection.pth"
    if not ckpt.exists():
        print(f"  ⚠️  No checkpoint at {ckpt}")
        return None, None, 0.62

    if str(ckpt) in _model_cache:
        return _model_cache[str(ckpt)]

    import torch
    import torch.nn as nn

    class DetectionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                *[self._b(i, o) for i, o in [(3,32),(32,64),(64,128),(128,256)]])
            self.attn = nn.Sequential(
                nn.Conv2d(256,64,(1,1)), nn.ReLU(),
                nn.Conv2d(64,1,(1,1)), nn.Softmax(dim=2))
            self.gap  = nn.AdaptiveAvgPool2d((1,1))
            self.cls  = nn.Sequential(
                nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,2))
        @staticmethod
        def _b(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co),
                nn.ReLU(), nn.MaxPool2d(2))
        def forward(self, x):
            f = self.enc(x); a = self.attn(f)
            f = (f * a).sum(dim=2, keepdim=True)
            return self.cls(self.gap(f).flatten(1))

    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(str(ckpt), map_location=dev)
    m    = DetectionCNN().to(dev)
    m.load_state_dict(data["model_state"])
    m.eval()
    thr  = float(data.get("best_threshold", 0.62))
    print(f"  ✅ Model loaded  ep={data.get('epoch','?')}  "
          f"f1={data.get('best_val_f1','?')}  thr={thr:.3f}")
    _model_cache[str(ckpt)] = (m, dev, thr)
    return m, dev, thr


def cnn_score(y: np.ndarray, sr: int, model, dev) -> float:
    try:
        import torch
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64,
            fmin=20, fmax=min(8000, sr // 2))
        m = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        m = (m - m.mean()) / (m.std() + 1e-6)
        t = torch.tensor(np.stack([m, m, m]), dtype=torch.float32
                         ).unsqueeze(0).to(dev)
        with torch.no_grad():
            return float(torch.softmax(model(t), dim=1)[0, 1].item())
    except Exception:
        return 0.0


def score_segment(y: np.ndarray, sr: int,
                  model=None, dev=None, cnn_w: float = 0.80) -> Dict:
    h = heuristic_score(y, sr)
    if model is not None and LIBROSA_OK:
        c = cnn_score(y, sr, model, dev)
        fused = cnn_w * c + (1 - cnn_w) * h
        if c > 0.45 and h > 0.45:
            fused = min(1.0, fused + 0.06)
    else:
        c, fused = float("nan"), h
    return {
        "heuristic_prob": round(h, 5),
        "cnn_prob":       None if math.isnan(c) else round(c, 5),
        "fused_prob":     round(float(np.clip(fused, 0, 1)), 5),
    }


# =============================================================================
#  SEGMENT EXTRACTION  — MEMS
# =============================================================================

def extract_mems_segments(
    mems_dir: Path,
    gpx_df: Optional[pd.DataFrame],
    gt_strategies: List[str],
    args,
    model=None, dev=None, model_thr: float = 0.62,
) -> pd.DataFrame:
    wav_files = sorted(mems_dir.glob("*.wav")) if mems_dir and mems_dir.exists() else []
    if not wav_files:
        print(f"  ⚠️  No WAV files in {mems_dir}")
        return pd.DataFrame()

    mems_start_default = (
        datetime.fromisoformat(args.mems_start).replace(tzinfo=timezone.utc)
        if args.mems_start
        else datetime(2025, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    )

    ctx = timedelta(minutes=args.context_min)
    win_n = int(args.segment_sec * MODEL_SR)
    hop_n = int(args.hop_sec    * MODEL_SR)

    rows = []
    for fi, wf in enumerate(wav_files, 1):
        print(f"  [{fi}/{len(wav_files)}] MEMS: {wf.name}", flush=True)
        try:
            rec_start = parse_mems_start(wf, mems_start_default)
            y_raw, raw_sr = load_wav(wf)
        except Exception as e:
            print(f"    ⚠️  {e}"); continue

        # Trim to context window
        ctx_s0 = max(0, (FLIGHT_TAKEOFF_UTC - ctx - rec_start).total_seconds())
        ctx_s1 = min(len(y_raw) / raw_sr,
                     (FLIGHT_LANDING_UTC + ctx - rec_start).total_seconds())
        if ctx_s1 <= ctx_s0:
            print(f"    ⚠️  File window doesn't overlap context — "
                  f"rec_start={rec_start}, ctx=[{ctx_s0:.0f}s,{ctx_s1:.0f}s]")
            continue

        y_ctx = y_raw[int(ctx_s0 * raw_sr): int(ctx_s1 * raw_sr)]
        y_rs  = resample(y_ctx, raw_sr, MODEL_SR)

        for ws, we in windows(len(y_rs), win_n, hop_n):
            clip  = pad_or_trim(y_rs[ws:we], win_n)
            t_off = ctx_s0 + ws / MODEL_SR          # offset from file start (s)
            mid   = rec_start + timedelta(seconds=t_off + args.segment_sec / 2)
            rms   = float(20 * math.log10(
                np.sqrt(np.mean(clip.astype(np.float64) ** 2)) + 1e-8))

            sc = score_segment(clip, MODEL_SR, model, dev, args.cnn_weight)

            row: Dict[str, Any] = {
                "source_file": wf.name,
                "offset_sec":  round(t_off, 3),
                "seg_mid_utc": mid.isoformat(),
                "rms_db":      round(rms, 2),
                **sc,
            }
            for strat in gt_strategies:
                row[f"gt_{strat}"] = assign_gt_label(
                    mid, strat, gpx_df, args.gpx_alt_threshold, args.gpx_speed_threshold)
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#  SEGMENT EXTRACTION  — BRÜEL
# =============================================================================

def load_bruel_flight_npy(npy_dir: Path, channel: int) -> Optional[np.ndarray]:
    files = sorted(npy_dir.glob("*_flight_window.npy")) if npy_dir else []
    if not files:
        return None
    arrays = []
    for nf in files:
        arr = np.load(str(nf), mmap_mode="r")
        sub = arr.mean(axis=1) if (arr.ndim == 2 and channel < 0) else \
              arr[:, min(channel, arr.shape[1]-1)] if arr.ndim == 2 else arr
        arrays.append(np.asarray(sub, dtype=np.float32))
        print(f"    {nf.name}: {sub.shape[0]/BRUEL_SR:.1f}s")
    return np.concatenate(arrays)


def load_bruel_pre_flight_wav(bruel_wav_dir: Optional[Path],
                              channel: int, duration_s: float = 90.0) -> Optional[np.ndarray]:
    if not bruel_wav_dir or not bruel_wav_dir.exists():
        return None
    for wf in sorted(bruel_wav_dir.glob("*.wav")):
        if "AT01I" not in wf.name:
            continue
        try:
            y, sr = load_wav(wf, max_sec=duration_s + 5)
            # Take only the pre-flight portion (before flight_window starts in this file)
            file_start = BRUEL_FILE_STARTS.get(wf.name)
            if file_start:
                pre_s = (FLIGHT_TAKEOFF_UTC - file_start).total_seconds()
                pre_s = max(0, min(pre_s, duration_s))
            else:
                pre_s = duration_s
            y = y[:int(pre_s * sr)]
            print(f"    Pre-flight from {wf.name}: {len(y)/sr:.1f}s")
            return resample(y, sr, MODEL_SR)
        except Exception as e:
            print(f"    ⚠️  {e}")
    return None


def extract_bruel_segments(
    npy_dir: Optional[Path],
    bruel_wav_dir: Optional[Path],
    gpx_df: Optional[pd.DataFrame],
    gt_strategies: List[str],
    args,
    model=None, dev=None,
) -> pd.DataFrame:
    y_flight = load_bruel_flight_npy(npy_dir, args.bruel_channel)
    if y_flight is None:
        print("  ⚠️  No Brüel .npy files found")
        return pd.DataFrame()

    y_flight_rs = resample(y_flight, BRUEL_SR, MODEL_SR)
    del y_flight

    win_n = int(args.segment_sec * MODEL_SR)
    hop_n = int(args.hop_sec    * MODEL_SR)

    rows = []

    # ── flight-window segments ─────────────────────────────────────────────
    # .npy files span exactly FLIGHT_TAKEOFF_UTC → FLIGHT_LANDING_UTC
    for ws, we in windows(len(y_flight_rs), win_n, hop_n):
        clip  = pad_or_trim(y_flight_rs[ws:we], win_n)
        t_off = ws / MODEL_SR
        mid   = FLIGHT_TAKEOFF_UTC + timedelta(seconds=t_off + args.segment_sec / 2)
        rms   = float(20 * math.log10(
            np.sqrt(np.mean(clip.astype(np.float64) ** 2)) + 1e-8))
        sc = score_segment(clip, MODEL_SR, model, dev, args.cnn_weight)
        row: Dict[str, Any] = {
            "source":      "bruel_flight",
            "offset_sec":  round(t_off, 3),
            "seg_mid_utc": mid.isoformat(),
            "rms_db":      round(rms, 2),
            **sc,
        }
        for strat in gt_strategies:
            row[f"gt_{strat}"] = assign_gt_label(
                mid, strat, gpx_df, args.gpx_alt_threshold, args.gpx_speed_threshold)
        rows.append(row)

    # ── pre-flight negatives ───────────────────────────────────────────────
    y_pre = load_bruel_pre_flight_wav(bruel_wav_dir, args.bruel_channel)
    if y_pre is not None:
        pre_start = FLIGHT_TAKEOFF_UTC - timedelta(seconds=len(y_pre) / MODEL_SR)
        for ws, we in windows(len(y_pre), win_n, hop_n):
            clip  = pad_or_trim(y_pre[ws:we], win_n)
            t_off = ws / MODEL_SR
            mid   = pre_start + timedelta(seconds=t_off + args.segment_sec / 2)
            rms   = float(20 * math.log10(
                np.sqrt(np.mean(clip.astype(np.float64) ** 2)) + 1e-8))
            sc = score_segment(clip, MODEL_SR, model, dev, args.cnn_weight)
            row = {
                "source":      "bruel_pre_flight",
                "offset_sec":  round((mid - FLIGHT_TAKEOFF_UTC).total_seconds(), 3),
                "seg_mid_utc": mid.isoformat(),
                "rms_db":      round(rms, 2),
                **sc,
            }
            for strat in gt_strategies:
                row[f"gt_{strat}"] = assign_gt_label(
                    mid, strat, gpx_df, args.gpx_alt_threshold, args.gpx_speed_threshold)
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#  METRICS
# =============================================================================

def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    best_f1, best_thr = 0.0, 0.62
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (y_score >= thr).astype(int)
        tp = int(np.sum((pred==1) & (y_true==1)))
        fp = int(np.sum((pred==1) & (y_true==0)))
        fn = int(np.sum((pred==0) & (y_true==1)))
        p = tp / max(tp+fp, 1); r = tp / max(tp+fn, 1)
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def compute_metrics(df: pd.DataFrame, gt_col: str,
                    score_col: str = "fused_prob",
                    fixed_thr: Optional[float] = None) -> Dict:
    if df.empty or gt_col not in df.columns or score_col not in df.columns:
        return {}
    y_true  = df[gt_col].values.astype(int)
    y_score = df[score_col].fillna(0).values.astype(float)
    n_pos = int(y_true.sum()); n_neg = int((y_true==0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"n_segments": len(df), "n_drone": n_pos, "n_non_drone": n_neg,
                "note": "single-class — metrics not meaningful"}

    if fixed_thr is not None:
        thr = fixed_thr
        _, best_f1 = best_threshold(y_true, y_score)
    else:
        thr, best_f1 = best_threshold(y_true, y_score)

    pred = (y_score >= thr).astype(int)
    tp   = int(np.sum((pred==1)&(y_true==1))); fp = int(np.sum((pred==1)&(y_true==0)))
    tn   = int(np.sum((pred==0)&(y_true==0))); fn = int(np.sum((pred==0)&(y_true==1)))
    pre  = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
    f1   = 2*pre*rec/(pre+rec) if (pre+rec) > 0 else 0.0
    acc  = (tp+tn) / max(tp+tn+fp+fn, 1)

    out = {
        "n_segments": len(df), "n_drone": n_pos, "n_non_drone": n_neg,
        "threshold":  round(thr, 3),
        "accuracy":   round(acc,  4),
        "precision":  round(pre,  4),
        "recall":     round(rec,  4),
        "f1":         round(f1,   4),
        "best_f1_at_best_thr": round(best_f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    if SKLEARN_OK and len(np.unique(y_true)) > 1:
        out["auroc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["auprc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return out


# =============================================================================
#  PLOTTING
# =============================================================================

def _ax(ax):
    ax.grid(True, alpha=0.3, lw=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def plot_timeline(df: pd.DataFrame, gt_col: str, score_col: str,
                  thr: float, title: str, out: Path):
    y_true  = df[gt_col].values.astype(int)
    y_score = df[score_col].fillna(0).values.astype(float)
    t       = df["offset_sec"].values if "offset_sec" in df.columns else np.arange(len(df))

    pred = (y_score >= thr).astype(int)
    tp_m = (pred==1)&(y_true==1); fp_m = (pred==1)&(y_true==0)
    tn_m = (pred==0)&(y_true==0); fn_m = (pred==0)&(y_true==1)

    fig, axes = plt.subplots(2, 1, figsize=(22, 8), facecolor=P["bg"])
    fig.suptitle(title, fontsize=12, color=P["drone"], fontweight="bold")

    ax = axes[0]
    ax.scatter(t[y_true==0], y_score[y_true==0], c=P["muted"], s=7, alpha=0.35, label="non_drone GT")
    ax.scatter(t[y_true==1], y_score[y_true==1], c=P["drone"], s=11, alpha=0.8,  label="drone GT")
    ax.axhline(thr, color=P["warn"], lw=1.4, ls="--", label=f"thr={thr:.2f}")
    ax.set_ylim(-0.05, 1.10); ax.set_ylabel("Probability"); ax.legend(fontsize=8); _ax(ax)

    ax = axes[1]
    kw = dict(s=11, alpha=0.8)
    ax.scatter(t[tp_m], y_score[tp_m], c=P["ok"],   label="TP", **kw)
    ax.scatter(t[fp_m], y_score[fp_m], c=P["warn"],  label="FP", **kw)
    ax.scatter(t[tn_m], y_score[tn_m], c=P["muted"], label="TN", s=7, alpha=0.3)
    ax.scatter(t[fn_m], y_score[fn_m], c=P["err"],   label="FN", **kw)
    ax.axhline(thr, color=P["warn"], lw=1.4, ls="--")
    ax.set_ylim(-0.05, 1.10); ax.set_ylabel("Probability")
    ax.set_xlabel("Offset (s)"); ax.legend(fontsize=8, ncol=4); _ax(ax)

    plt.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_roc_pr(df: pd.DataFrame, gt_col: str, score_col: str,
                metrics: Dict, title: str, out: Path):
    if not SKLEARN_OK:
        return
    y_true  = df[gt_col].values.astype(int)
    y_score = df[score_col].fillna(0).values.astype(float)
    if len(np.unique(y_true)) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=P["bg"])
    fig.suptitle(title, fontsize=11, color=P["drone"])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, color=P["drone"], lw=2,
                 label=f"AUC={metrics.get('auroc',0):.4f}")
    axes[0].plot([0,1],[0,1],"--",color=P["muted"],lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC"); axes[0].legend(fontsize=9); _ax(axes[0])

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(rec, prec, color=P["ok"], lw=2,
                 label=f"AUC={metrics.get('auprc',0):.4f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall"); axes[1].legend(fontsize=9); _ax(axes[1])

    plt.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_confusion(df: pd.DataFrame, gt_col: str, score_col: str,
                   thr: float, title: str, out: Path):
    y_true = df[gt_col].values.astype(int)
    y_pred = (df[score_col].fillna(0).values >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dk", [P["panel"], P["drone"]])
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    im = ax.imshow(cm, cmap=cmap)
    plt.colorbar(im, ax=ax)
    lbls = ["non_drone", "drone"]
    ax.set_xticks([0,1]); ax.set_xticklabels(lbls)
    ax.set_yticks([0,1]); ax.set_yticklabels(lbls)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    tot = cm.sum(axis=1, keepdims=True) + 1e-8
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({100*cm[i,j]/tot[i,0]:.1f}%)",
                    ha="center", va="center", color=P["text"],
                    fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=11, color=P["drone"])
    plt.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_score_dist(df: pd.DataFrame, gt_col: str, score_col: str,
                    title: str, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    for lv, col, nm in [(0, P["muted"], "non_drone"), (1, P["drone"], "drone")]:
        sub = df[df[gt_col]==lv][score_col].dropna()
        if len(sub):
            ax.hist(sub.values, bins=40, alpha=0.65, color=col,
                    label=f"{nm} (n={len(sub)})", density=True)
    ax.set_xlabel("Fused probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=9); _ax(ax)
    fig.suptitle(title, fontsize=11, color=P["drone"])
    plt.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_comparison_table(all_results: List[Dict], out: Path):
    """
    Grid plot comparing F1 / AUROC / Recall across all
    (source × gt_strategy) combinations.
    """
    if not all_results:
        return

    sources   = sorted({r["source"]      for r in all_results})
    strategies= sorted({r["gt_strategy"] for r in all_results})
    metrics   = ["f1", "auroc", "precision", "recall", "accuracy"]
    colors    = [P["drone"], P["ok"], P["purp"], P["warn"], P["err"]]

    n_s = len(sources); n_st = len(strategies); n_m = len(metrics)
    fig, axes = plt.subplots(n_m, 1, figsize=(max(10, 2*n_s*n_st), 4*n_m),
                              facecolor=P["bg"])
    if n_m == 1:
        axes = [axes]
    fig.suptitle("Dunakeszi Benchmark — Strategy × Source Comparison",
                 fontsize=13, color=P["drone"], fontweight="bold")

    for mi, (metric, col) in enumerate(zip(metrics, colors)):
        ax = axes[mi]
        ax.set_facecolor(P["panel"])
        labels, vals = [], []
        for src in sources:
            for strat in strategies:
                match = [r for r in all_results
                         if r["source"]==src and r["gt_strategy"]==strat]
                v = match[0].get("metrics", {}).get(metric) if match else None
                labels.append(f"{src}\n{strat}")
                vals.append(float(v) if v is not None else 0.0)
        xs = np.arange(len(labels))
        bars = ax.bar(xs, vals, color=col, alpha=0.75, edgecolor=P["grid"], width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=P["text"])
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(metric.upper()); ax.set_ylim(0, 1.15); _ax(ax)

    plt.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


# =============================================================================
#  SAVE HELPERS
# =============================================================================

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"  💾 {path.name}  ({len(df)} rows)")


def save_json(obj: Dict, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  💾 {path.name}")


# =============================================================================
#  ONE EVALUATION RUN
# =============================================================================

def run_one(
    df: pd.DataFrame,
    source: str,
    gt_strategy: str,
    fixed_thr: Optional[float],
    output_dir: Path,
) -> Dict:
    tag = f"{source}_{gt_strategy}"
    gt_col    = f"gt_{gt_strategy}"
    score_col = "fused_prob"

    if df.empty or gt_col not in df.columns:
        print(f"  ⚠️  {tag}: no data or missing GT column")
        return {}

    # Save segment CSV
    cols_out = [c for c in df.columns
                if c not in [f"gt_{s}" for s in GT_STRATEGIES] + [gt_col]
                or c == gt_col]
    save_csv(df[[c for c in df.columns
                 if not c.startswith("gt_") or c == gt_col]],
             output_dir / f"{tag}_segments.csv")

    m = compute_metrics(df, gt_col, score_col, fixed_thr)
    save_json(m, output_dir / f"{tag}_report.json")

    thr = float(m.get("threshold", fixed_thr or 0.62))
    title_base = f"{source.upper()} | GT={gt_strategy} | thr={thr:.2f}"

    print(f"  {tag}:  F1={m.get('f1','?')}  AUROC={m.get('auroc','?')}  "
          f"P={m.get('precision','?')}  R={m.get('recall','?')}")

    pdir = output_dir / "plots"
    plot_timeline(df, gt_col, score_col, thr,
                  f"Timeline — {title_base}", pdir / f"{tag}_timeline.png")
    plot_roc_pr(df, gt_col, score_col, m,
                f"ROC / PR — {title_base}", pdir / f"{tag}_roc_pr.png")
    plot_confusion(df, gt_col, score_col, thr,
                   f"Confusion — {title_base}", pdir / f"{tag}_confusion.png")
    plot_score_dist(df, gt_col, score_col,
                    f"Score Distributions — {title_base}",
                    pdir / f"{tag}_score_dist.png")
    print(f"    📊 4 plots → plots/{tag}_*.png")

    return {"source": source, "gt_strategy": gt_strategy, "metrics": m}


# =============================================================================
#  MAIN
# =============================================================================

SEP  = "─" * 72
SEP2 = "═" * 72


def main():
    args = build_parser().parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    output_dir = Path(args.output)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(SEP2)
    print("  DUNAKESZI BENCHMARK  v2")
    print(f"  source       : {args.source}")
    print(f"  gt_strategy  : {args.gt_strategy}")
    print(f"  segment_sec  : {args.segment_sec}s  hop={args.hop_sec}s")
    print(f"  output       : {output_dir}")
    print(SEP2)

    # ── Discover paths ────────────────────────────────────────────────────────
    paths = discover_paths(args)

    # ── Decide which strategies to run ───────────────────────────────────────
    if args.gt_strategy == "all":
        strategies = list(GT_STRATEGIES)
    else:
        strategies = [args.gt_strategy]

    # ── Load GPX ──────────────────────────────────────────────────────────────
    gpx_df = load_gpx_csv(paths["gpx_csv"])
    if gpx_df is None and any(s != "flight_window" for s in strategies):
        print("  ⚠️  GPX CSV not found — "
              "gpx_altitude / gpx_speed will fall back to flight_window")

    # ── Load model ────────────────────────────────────────────────────────────
    model = dev = None
    model_thr = args.threshold or 0.62
    if not args.no_model and TORCH_OK and LIBROSA_OK:
        model, dev, model_thr = load_model(paths["model_dir"])

    fixed_thr = args.threshold   # None → find best per run

    # ── Extract segments ──────────────────────────────────────────────────────
    df_mems  = pd.DataFrame()
    df_bruel = pd.DataFrame()

    if args.source in ("mems", "both"):
        print(f"\n{SEP}\n  EXTRACTING MEMS SEGMENTS\n{SEP}")
        if paths["mems_dir"]:
            df_mems = extract_mems_segments(
                paths["mems_dir"], gpx_df, strategies, args,
                model, dev, model_thr)
        else:
            print("  ⚠️  mems_dir not found — skipping")

    if args.source in ("bruel", "both"):
        print(f"\n{SEP}\n  EXTRACTING BRÜEL SEGMENTS\n{SEP}")
        if paths["bruel_npy"]:
            df_bruel = extract_bruel_segments(
                paths["bruel_npy"], paths["bruel_wav"], gpx_df,
                strategies, args, model, dev)
        else:
            print("  ⚠️  bruel_npy not found — skipping")

    # ── Save raw segment data ─────────────────────────────────────────────────
    if not df_mems.empty:
        save_csv(df_mems,  output_dir / "mems_all_segments.csv")
    if not df_bruel.empty:
        save_csv(df_bruel, output_dir / "bruel_all_segments.csv")

    # ── Run evaluations ───────────────────────────────────────────────────────
    print(f"\n{SEP}\n  EVALUATING\n{SEP}")
    all_results = []

    for source, df in [("mems", df_mems), ("bruel", df_bruel)]:
        if df.empty:
            continue
        for strat in strategies:
            result = run_one(df, source, strat, fixed_thr, output_dir)
            if result:
                all_results.append(result)

    # ── Comparison plot (when multiple runs) ──────────────────────────────────
    if len(all_results) > 1:
        plot_comparison_table(all_results,
                              output_dir / "plots" / "comparison_table.png")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "generated":     datetime.utcnow().isoformat(),
        "args": {
            "source":      args.source,
            "gt_strategy": args.gt_strategy,
            "segment_sec": args.segment_sec,
            "hop_sec":     args.hop_sec,
            "scoring":     "cnn+heuristic" if model else "heuristic_only",
        },
        "flight_window": {
            "takeoff_utc":  FLIGHT_TAKEOFF_UTC.isoformat(),
            "landing_utc":  FLIGHT_LANDING_UTC.isoformat(),
            "duration_sec": FLIGHT_DURATION_S,
        },
        "runs": all_results,
    }
    save_json(summary, output_dir / "benchmark_summary.json")

    # ── Print top-line table ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  RESULTS SUMMARY")
    print(f"  {'source':<8} {'gt_strategy':<16} {'F1':>6} {'AUROC':>6} "
          f"{'Prec':>6} {'Recall':>6} {'n_drone':>8} {'n_neg':>8}")
    print("  " + "─"*68)
    for r in all_results:
        m = r.get("metrics", {})
        print(f"  {r['source']:<8} {r['gt_strategy']:<16} "
              f"{m.get('f1','—'):>6} {m.get('auroc','—'):>6} "
              f"{m.get('precision','—'):>6} {m.get('recall','—'):>6} "
              f"{m.get('n_drone','—'):>8} {m.get('n_non_drone','—'):>8}")
    print(SEP2)
    print(f"  Outputs → {output_dir}\n")


if __name__ == "__main__":
    main()
