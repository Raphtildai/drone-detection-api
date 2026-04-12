#!/usr/bin/env python3
# vitemor_indoor_benchmark.py
"""
Vitemor Indoor Drone Dataset — Benchmark Evaluator
===================================================
Session  : 250703_Vitemor_Dron
Date     : 2025-07-03
Location : Pannónia Film Studio (indoor)
Audio    : 12-ch array (Grp1×6 + Grp2×6), 192 kHz 24-bit WAV + 4-ch group3
Telemetry: mavic_pro.csv / mavic_pro2.csv / mavic mini.csv  (x_rel/y_rel/z_rel)

Design
------
This script is a PURPOSE-BUILT SIBLING of dunakeszi_benchmark.py.
It reuses the EXACT same scoring stack (heuristic_score, feature_stack_v15,
cnn_score, score_segment, load_model) and the same output schema so that
results from both sessions can be compared directly.

Differences from the outdoor benchmark
---------------------------------------
1. Ground truth comes from 75 SAMPLE-INDEX markers (not a UTC flight window).
2. Audio is sliced by marker sample offsets via soundfile seeks (no .npy caches).
3. Telemetry uses x_rel/y_rel/z_rel in metres (not GPS lat/lon).
4. Every segment carries drone_model / phase / maneuver / nominal_altitude_m.
5. q4_features.csv BPF values are used to set per-drone heuristic band centres.

Marker-based GT
---------------
For each inter-marker interval [marker_i, marker_i+1]:
  - label = 1 (drone) if the interval is part of an active flight phase (1–3)
             and neither marker is a takeoff/landing boundary of silence
  - label = 0 (non_drone) for pre-flight silence and post-landing windows
  - label = -1 (ambiguous) for phase 4 free-run (no scripted positions)
    → excluded from metrics by default (--include-free-run to override)

CLI quick starts
----------------
  # Heuristic only, auto-discover paths:
  python vitemor_indoor_benchmark.py --dataset-root ./kutatas

  # Full CNN run, all channels averaged:
  python vitemor_indoor_benchmark.py \\
      --dataset-root ./kutatas \\
      --model-dir    /path/to/drone_v15/models \\
      --channel      -1

  # Only hover segments (best for initial validation):
  python vitemor_indoor_benchmark.py \\
      --dataset-root ./kutatas \\
      --maneuver     hover

  # Cross-domain comparison (after running outdoor benchmark):
  python vitemor_indoor_benchmark.py \\
      --dataset-root    ./kutatas \\
      --outdoor-summary ./dunakeszi_benchmark_out/benchmark_summary.json

Outputs (--output, default ./vitemor_benchmark_out)
----------------------------------------------------
  indoor_segments.csv          per-segment: scores + GT + metadata
  indoor_report.json           F1/AUROC/PR/confusion per maneuver type
  indoor_per_drone_report.json per-drone breakdown
  plots/
    timeline_{drone}.png       probability timeline per drone
    roc_pr.png
    confusion.png
    score_dist_by_maneuver.png
    bpf_vs_score.png           BPF energy ratio vs heuristic score scatter
    cross_domain_comparison.png  (only when --outdoor-summary is given)
  benchmark_summary.json
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, math, json, csv, time, random, warnings, argparse, re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False
    print("⚠️  soundfile not found — install with: pip install soundfile")

from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
#  SESSION CONSTANTS
# =============================================================================

SESSION_ID      = "250703_Vitemor_Dron"
SESSION_DATE    = "2025-07-03"
DOMAIN          = "indoor"
INDOOR_SR       = 192_000       # WAV sample rate
MODEL_SR        = 22_050        # target SR for CNN features (matches v15 Config.SR)
GROUP1_CHANNELS = 6             # Grp1-1 … Grp1-6
GROUP2_CHANNELS = 6             # Grp2-1 … Grp2-6
GROUP3_CHANNELS = 4             # g3-1 … G3-4  (shorter duration)

# DAW recording start times (from position CSV headers and analysis.md)
DAW_START_UTC = {
    "mavic_pro":  datetime(2025, 7, 3, 11, 29,  3, tzinfo=timezone.utc),
    "mavic_pro2": datetime(2025, 7, 3, 11, 49, 57, tzinfo=timezone.utc),
    "mavic_mini": datetime(2025, 7, 3, 11, 37, 56, tzinfo=timezone.utc),
}

# Sync tap: global anchor (sample 1,646,731 at 192 kHz)
SYNC_TAP_SAMPLE = 1_646_731

# Phase definitions — maps phase number → drone key
PHASE_DRONE = {1: "mavic_pro", 2: "mavic_pro2", 3: "mavic_mini", 4: "mavic_pro2"}
PHASE_LABEL  = {1: "Mavic Pro", 2: "Mavic 2 Pro", 3: "Mavic Mini", 4: "Mavic 2 Pro (free run)"}

# Maneuver classification by event text patterns
MANEUVER_MAP = {
    "lebeg":    "hover",
    "felszall": "takeoff",
    "leszall":  "landing",
    "negyzet":  "square",
    "kor":      "circle",
    "elore":    "forward",
    "hatra":    "backward",
    "jobbra":   "right",
    "balra":    "left",
    "forgas":   "rotation",
    "repules":  "flyover",
    "bh":       "diagonal",
    "je":       "diagonal",
    "waypoint": "waypoint",
    "free run": "free_run",
}

# BPF Hz per drone model (from q4_features.csv analysis)
BPF_HZ = {
    "mavic_pro":  209.0,   # median across altitudes (ignoring 2m anomaly)
    "mavic_pro2": 195.0,   # median 193-199 Hz
    "mavic_mini": 360.0,   # median 355-363 Hz
}

# Altitude at each hover event (from overview.md)
HOVER_ALTITUDES_M = [1, 2, 3, 4]

# ── Plot palette (matches outdoor benchmark) ──────────────────────────────────
P = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "muted": "#484f58",
    "drone": "#00d4ff", "ok": "#3fb950", "err": "#f85149",
    "warn": "#d29922", "purp": "#a371f7",
    "pro": "#00d4ff", "mv2": "#3fb950", "mini": "#a371f7",
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
        description="Vitemor indoor drone benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-root", default=".",
                   help="Root folder containing audio_files/, markers/, position/")
    p.add_argument("--audio-dir",   default=None, help="Override: path to audio_files/")
    p.add_argument("--markers-dir", default=None, help="Override: path to markers/")
    p.add_argument("--position-dir",default=None, help="Override: path to position/")
    p.add_argument("--q4-csv",      default=None, help="Override: path to q4_features.csv")
    p.add_argument("--model-dir",   default=None, help="drone_v15 models/ folder")
    p.add_argument("--outdoor-summary", default=None,
                   help="Path to outdoor benchmark_summary.json for cross-domain plot")
    p.add_argument("--output",      default="./vitemor_benchmark_out")

    p.add_argument("--channel",  type=int, default=-1,
                   help="Channel index to use (-1 = average all channels in group)")
    p.add_argument("--group",    default="grp1",
                   choices=["grp1", "grp2", "grp3", "all"],
                   help="Which microphone group to use")
    p.add_argument("--phase",    type=int, default=None, choices=[1,2,3,4],
                   help="Only evaluate a specific phase (default: phases 1-3)")
    p.add_argument("--maneuver", default=None,
                   choices=["hover","square","circle","forward","backward",
                             "diagonal","flyover","rotation","free_run","all"],
                   help="Filter to a specific maneuver type")
    p.add_argument("--include-free-run", action="store_true",
                   help="Include phase 4 free-run segments (label=-1 → excluded by default)")

    p.add_argument("--segment-sec", type=float, default=1.0)
    p.add_argument("--hop-sec",     type=float, default=0.5)
    p.add_argument("--no-model",    action="store_true")
    p.add_argument("--cnn-weight",  type=float, default=0.80)
    p.add_argument("--threshold",   type=float, default=None)
    p.add_argument("--seed",        type=int,   default=42)
    return p


# =============================================================================
#  PATH DISCOVERY
# =============================================================================

def discover_paths(args) -> Dict[str, Optional[Path]]:
    root = Path(args.dataset_root)

    def find(*candidates):
        for c in candidates:
            if c and Path(c).exists():
                return Path(c)
        return None

    audio_dir    = find(args.audio_dir,
                        root/"audio_files", root/"kutatas"/"audio_files")
    markers_dir  = find(args.markers_dir,
                        root/"markers",     root/"kutatas"/"markers")
    position_dir = find(args.position_dir,
                        root/"position",    root/"kutatas"/"position")
    q4_csv       = find(args.q4_csv,
                        root/"q4_features.csv",
                        root/"output"/"q4_features.csv",
                        Path("q4_features.csv"))
    model_dir    = find(args.model_dir,
                        Path("/tmp/drone_v15/models"),
                        Path.home()/"drone_v15"/"models")

    paths = {
        "audio_dir":    audio_dir,
        "markers_dir":  markers_dir,
        "position_dir": position_dir,
        "q4_csv":       q4_csv,
        "model_dir":    model_dir,
    }
    print("\n  Discovered paths:")
    for k, v in paths.items():
        status = "✓" if v and v.exists() else "✗"
        print(f"    {status}  {k:14s}: {v}")
    return paths


# =============================================================================
#  MARKER LOADING
# =============================================================================

def load_markers(markers_dir: Path) -> pd.DataFrame:
    """
    Parse _samples.txt — returns DataFrame with columns:
      index, sample, event_en, event_hu, phase, drone_key,
      maneuver, is_takeoff, is_landing, nominal_altitude_m
    """
    f = markers_dir / "250703_Vitemor_Dron_markers_samples.txt"
    if not f.exists():
        # Try any _samples.txt in the folder
        candidates = list(markers_dir.glob("*samples*"))
        if not candidates:
            raise FileNotFoundError(f"No markers_samples.txt in {markers_dir}")
        f = candidates[0]

    rows = []
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expected format: "  1  1646731  taps sync"  (tab or multi-space)
            parts = re.split(r"\s{2,}|\t", line)
            if len(parts) < 3:
                continue
            try:
                idx     = int(parts[0])
                sample  = int(parts[1].replace(",", ""))
                event   = " ".join(parts[2:]).strip()
            except ValueError:
                continue
            rows.append({"index": idx, "sample": sample, "event_raw": event})

    if not rows:
        raise ValueError(f"No markers parsed from {f}")

    df = pd.DataFrame(rows)

    # Assign phase and drone from known marker ranges (overview.md)
    def phase_of(idx):
        if   idx <= 19: return 1
        elif idx <= 50: return 2
        elif idx <= 66: return 3
        else:           return 4

    df["phase"]     = df["index"].apply(phase_of)
    df["drone_key"] = df["phase"].map(PHASE_DRONE)
    df["drone_label"] = df["phase"].map(PHASE_LABEL)

    # Classify maneuver from Hungarian event text
    def classify_maneuver(ev):
        ev_lower = ev.lower()
        for token, name in MANEUVER_MAP.items():
            if token in ev_lower:
                return name
        return "other"

    df["maneuver"]   = df["event_raw"].apply(classify_maneuver)
    df["is_takeoff"] = df["event_raw"].str.lower().str.contains("felszall")
    df["is_landing"] = df["event_raw"].str.lower().str.contains("leszall")

    # Nominal altitude from hover events: "1m lebeg" → 1
    def parse_altitude(ev):
        m = re.search(r"(\d+)\s*m", ev.lower())
        return int(m.group(1)) if m else None
    df["nominal_altitude_m"] = df["event_raw"].apply(parse_altitude)

    print(f"  Markers: {len(df)} events parsed from {f.name}")
    return df


# =============================================================================
#  INDOOR POSITION LOADING
# =============================================================================

def load_indoor_position(position_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Returns dict: {"mavic_pro": df, "mavic_pro2": df, "mavic_mini": df}
    Each DataFrame has: timestamp (datetime), x_rel, y_rel, z_rel,
                        accel_x, accel_y, pitch_rel, roll_rel
    """
    file_map = {
        "mavic_pro":  "mavic_pro.csv",
        "mavic_pro2": "mavic_pro2.csv",
        "mavic_mini": "mavic mini.csv",
    }
    result = {}
    for key, fname in file_map.items():
        path = position_dir / fname
        if not path.exists():
            print(f"  ⚠️  Position file not found: {path}")
            continue
        try:
            df = pd.read_csv(path, comment="#", skipinitialspace=True)
            # Normalise column names
            df.columns = [c.strip() for c in df.columns]
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            result[key] = df
            print(f"  Position [{key}]: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠️  Failed to load {fname}: {e}")
    return result


def position_at_sample(
    pos_dfs: Dict[str, pd.DataFrame],
    drone_key: str,
    sample_mid: int,
    sr: int = INDOOR_SR,
    window_s: float = 5.0,
) -> Optional[Dict]:
    """
    Look up drone position at a given sample index.
    Uses the DAW→GPS clock alignment from analysis.md:
      - mavic_pro:  reliable (±0 s)
      - mavic_pro2: degraded (±40 s) — z_rel from hover altitude used instead
      - mavic_mini: degraded (±20 s) — z_rel from hover altitude used instead
    Returns dict with z_rel (altitude), x_rel, y_rel or None.
    """
    df = pos_dfs.get(drone_key)
    if df is None or "timestamp" not in df.columns:
        return None
    daw_start = DAW_START_UTC.get(drone_key)
    if daw_start is None:
        return None
    wall_time = daw_start + timedelta(seconds=sample_mid / sr)
    lo = wall_time - timedelta(seconds=window_s)
    hi = wall_time + timedelta(seconds=window_s)
    sub = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].dropna(subset=["z_rel"])
    if sub.empty:
        return None
    row = sub.iloc[len(sub) // 2]
    return {
        "x_rel":      float(row.get("x_rel", 0)),
        "y_rel":      float(row.get("y_rel", 0)),
        "z_rel":      float(row.get("z_rel", 0)),
        "pitch_rel":  float(row.get("pitch_rel", 0)),
        "roll_rel":   float(row.get("roll_rel", 0)),
    }


# =============================================================================
#  INDOOR GT ASSIGNMENT
# =============================================================================

# Silence windows (sample ranges where no drone is flying)
# Pre-flight: 0 → marker #2 (takeoff) = 0 → 20,449,142
# Between Phase 1 landing (marker 19) and Phase 2 takeoff (marker 20):
#   127,360,430 → 276,930,583
# Between Phase 2 landing (marker 50) and Phase 3 takeoff (marker 51):
#   468,975,616 → 663,479,722
# After Phase 3 landing (marker 66) but before Phase 4 (marker 67):
#   715,968,682 → 797,442,048
# Phase 4 is labelled ambiguous (-1) unless --include-free-run

SILENCE_WINDOWS = [
    (0, 20_449_142),
    (127_360_430, 276_930_583),
    (468_975_616, 663_479_722),
    (715_968_682, 797_442_048),
]

PHASE4_START = 797_442_048
PHASE4_END   = 964_679_513   # marker 75 + some margin


def assign_gt_indoor(
    sample_mid: int,
    markers_df: pd.DataFrame,
    include_free_run: bool = False,
) -> int:
    """
    Returns:
       1  = drone active (phases 1–3)
       0  = silence / no drone
      -1  = ambiguous (phase 4 free run)
    """
    # Phase 4 free run
    if PHASE4_START <= sample_mid <= PHASE4_END:
        return -1 if not include_free_run else 1

    # Silence windows
    for lo, hi in SILENCE_WINDOWS:
        if lo <= sample_mid <= hi:
            return 0

    # Inside a flight phase (between takeoff and landing)
    # Find the interval the sample falls in
    samples = markers_df["sample"].values
    idx = np.searchsorted(samples, sample_mid, side="right") - 1
    if idx < 0 or idx >= len(markers_df):
        return 0

    row = markers_df.iloc[idx]
    phase = int(row["phase"])

    # Phase 4 → ambiguous
    if phase == 4:
        return -1 if not include_free_run else 1

    # Phases 1–3: 1 unless the interval starts with a landing
    if row["is_landing"]:
        return 0

    return 1


# =============================================================================
#  AUDIO LOADING  — marker-based slicing
# =============================================================================

def list_group_files(audio_dir: Path, group: str) -> List[Path]:
    """Return sorted list of WAV files for the requested group."""
    patterns = {
        "grp1":  ["Grp1-*.wav", "grp1-*.wav"],
        "grp2":  ["Grp2-*.wav", "grp2-*.wav"],
        "grp3":  ["g3-*.wav",   "G3-*.wav",   "g3*.wav"],
        "all":   ["*.wav"],
    }
    files = []
    for pat in patterns.get(group, ["*.wav"]):
        files.extend(audio_dir.glob(pat))
    return sorted(set(files))


def load_wav_slice_sf(path: Path, start_sample: int, end_sample: int,
                      channel: int = -1) -> Tuple[np.ndarray, int]:
    """
    Load a slice of a multichannel WAV using soundfile's seek capability.
    channel = -1 → average all channels.
    Returns (float32 mono array, sr).
    """
    if not SF_OK:
        raise RuntimeError("soundfile required for indoor audio loading")
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        n  = f.frames
        s0 = max(0, min(start_sample, n - 1))
        s1 = max(s0 + 1, min(end_sample, n))
        f.seek(s0)
        data = f.read(s1 - s0, dtype="float32", always_2d=True)
    if channel < 0:
        mono = data.mean(axis=1)
    else:
        ch = min(channel, data.shape[1] - 1)
        mono = data[:, ch]
    return mono.astype(np.float32), int(sr)


def average_group(audio_dir: Path, group: str,
                  start_sample: int, end_sample: int,
                  channel: int) -> Optional[np.ndarray]:
    """
    Load the same sample range from all files in a group and average them.
    Falls back gracefully if some files are missing or too short.
    """
    files = list_group_files(audio_dir, group)
    if not files:
        return None
    arrays = []
    sr_out = None
    for f in files:
        try:
            y, sr = load_wav_slice_sf(f, start_sample, end_sample, channel)
            if len(y) > 0:
                arrays.append(y)
                sr_out = sr
        except Exception:
            pass
    if not arrays:
        return None
    # Pad shorter arrays to the longest
    max_len = max(len(a) for a in arrays)
    padded  = [np.pad(a, (0, max_len - len(a))) for a in arrays]
    return np.mean(padded, axis=0).astype(np.float32)


# =============================================================================
#  SCORING STACK  (identical to outdoor benchmark — kept here for standalone use)
# =============================================================================

def _safe_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def feature_stack_v15(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """[log-mel, PCEN, delta-mel] → (3, 64, T) matching v15 training cache."""
    if not LIBROSA_OK:
        return None
    try:
        M   = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256,
                                              n_mels=64, fmin=20, fmax=min(8000, sr//2))
        m   = librosa.power_to_db(M, ref=np.max).astype(np.float32)
        pcen= librosa.pcen(M, sr=sr, hop_length=256, gain=0.8, bias=10.0,
                           power=0.25, time_constant=0.4, eps=1e-6).astype(np.float32)
        dlt = np.diff(m, axis=1, prepend=m[:, :1]).astype(np.float32)
        return np.stack([_safe_standardize(m),
                         _safe_standardize(pcen),
                         _safe_standardize(dlt)], axis=0)
    except Exception:
        return None


def _per_frame_entropy(S: np.ndarray, min_ef: float = 0.01) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64) + 1e-10
    fe = S.sum(axis=0)
    active = fe >= fe.max() * min_ef
    P = S / fe[np.newaxis, :]
    H = -np.sum(P * np.log2(P + 1e-12), axis=0)
    H_norm = (H / math.log2(S.shape[0])).astype(np.float32)
    H_norm[~active] = np.nan
    return H_norm


def _harmonic_comb_score(y: np.ndarray, sr: int,
                          f0_min=80.0, f0_max=500.0, n_harm=6) -> float:
    if not LIBROSA_OK:
        return 0.0
    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        Sm = S.mean(axis=1)
        fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        best = 0.0
        for f0 in np.linspace(f0_min, f0_max, 30):
            he = 0.0
            for k in range(1, n_harm+1):
                t = f0 * k
                if t > sr/2: break
                idx = int(np.argmin(np.abs(fr - t)))
                w   = max(1, int(20/(fr[1]-fr[0])))
                he += float(Sm[max(0, idx-w):idx+w+1].max())
            sc = he / (float(Sm.sum()) + 1e-10) / n_harm * n_harm
            if sc > best: best = sc
        return float(min(best, 1.0))
    except Exception:
        return 0.0


def heuristic_score(y: np.ndarray, sr: int,
                    bpf_hz: Optional[float] = None) -> float:
    """
    Full v15-faithful heuristic. Accepts optional bpf_hz to shift the
    drone-band energy window (default 80–2500 Hz) to ±20 Hz around the
    known BPF and its first three harmonics.
    """
    if len(y) == 0:
        return 0.0
    y = np.asarray(y, dtype=np.float32)
    rms    = float(np.sqrt(np.mean(y**2)) + 1e-8)
    rms_db = 20 * math.log10(rms)
    peak   = float(np.max(np.abs(y)) + 1e-8)
    cf     = peak / rms

    centroid = rolloff = bandwidth = 0.0
    median_ent = p10_ent = 0.5

    if LIBROSA_OK:
        try:
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
            centroid  = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
            rolloff   = float(np.mean(librosa.feature.spectral_rolloff(
                S=S, sr=sr, roll_percent=0.85)))
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
            fe = _per_frame_entropy(S, 0.01)
            ve = fe[~np.isnan(fe)]
            if len(ve) == 0: return 0.0
            median_ent = float(np.median(ve))
        except Exception:
            pass
    else:
        nperseg = min(512, max(32, len(y)//4))
        f, psd  = sp_signal.welch(y.astype(np.float64), fs=sr, nperseg=nperseg)
        total   = psd.sum() + 1e-10
        centroid = float(np.sum(f * psd) / total)

    if median_ent < 0.38: return float(np.clip(median_ent / 0.38 * 0.25, 0, 0.25))
    if cf > 8.5 and median_ent < 0.52: return 0.10

    voiced_ratio = f0_med = f0_std = 0.0
    if LIBROSA_OK:
        try:
            f0, _, _ = librosa.pyin(y, fmin=50.0, fmax=500.0, sr=sr,
                                     hop_length=256, fill_na=0.0)
            f0 = np.nan_to_num(f0, nan=0.0)
            voiced_ratio = float(np.mean(f0 > 0))
            vf = f0[f0 > 0]
            f0_med = float(np.median(vf)) if len(vf) else 0.0
            f0_std = float(np.std(vf))    if len(vf) else 0.0
        except Exception:
            pass

    comb = _harmonic_comb_score(y, sr) if (LIBROSA_OK and len(y) >= sr) else 0.0

    # BPF band energy — use per-drone band if available
    if bpf_hz is not None and LIBROSA_OK:
        try:
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
            fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
            total_p = (S**2).mean(axis=1).sum() + 1e-10
            band_e = 0.0
            for k in range(1, 5):
                fc = bpf_hz * k
                if fc > sr/2: break
                w = 20
                mask = (fr >= fc-w) & (fr <= fc+w)
                band_e += float((S**2).mean(axis=1)[mask].sum())
            drone_ratio = float(np.clip(band_e / total_p, 0, 1))
        except Exception:
            drone_ratio = 0.35
    else:
        drone_ratio = 0.35

    e_sc  = float(np.clip((rms_db+45)/25, 0, 1))
    f0_sc = (1.0 if 80 <= f0_med <= 500 else 0.3 if 50 <= f0_med < 80 else 0.0)
    v_sc  = float(np.clip(voiced_ratio/0.50, 0, 1))
    st_sc = (1.0 - float(np.clip(f0_std/80, 0, 1)) if f0_med > 0 else 0.2)
    c_sc  = 1.0 if 120 <= centroid <= 4000 else 0.35
    r_sc  = 1.0 if 300 <= rolloff  <= 8000 else 0.4
    b_sc  = 1.0 if 150 <= bandwidth <= 4000 else 0.5
    ent_sc= float(np.clip((median_ent-0.35)/0.35, 0, 1))
    cb_sc = float(np.clip(comb*3, 0, 1))
    band_sc = float(np.clip(drone_ratio/0.45, 0, 1))

    # v15 weights + extra band score (weighted in, slightly lower ent)
    score = (0.12*e_sc + 0.16*f0_sc + 0.10*v_sc + 0.08*st_sc +
             0.09*c_sc + 0.06*r_sc + 0.06*b_sc  + 0.14*ent_sc +
             0.10*cb_sc + 0.09*band_sc)
    prob = 1.0 / (1.0 + math.exp(-8.0*(score - 0.50)))
    return float(np.clip(prob, 0, 1))


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
    import torch, torch.nn as nn
    class DetectionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                *[self._b(i, o) for i, o in [(3,32),(32,64),(64,128),(128,256)]])
            self.attn = nn.Sequential(
                nn.Conv2d(256,64,(1,1)), nn.ReLU(),
                nn.Conv2d(64,1,(1,1)),  nn.Softmax(dim=2))
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            self.cls = nn.Sequential(
                nn.Linear(256,256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256,2))
        @staticmethod
        def _b(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        def forward(self, x):
            f = self.enc(x); a = self.attn(f)
            return self.cls(self.gap((f*a).sum(dim=2, keepdim=True)).flatten(1))
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(str(ckpt), map_location=dev)
    m    = DetectionCNN().to(dev)
    m.load_state_dict(data["model_state"]); m.eval()
    thr  = float(data.get("best_threshold", 0.62))
    f1   = data.get("best_val_f1", data.get("best_val_acc", "?"))
    print(f"  ✅ Model  ep={data.get('epoch','?')}  f1={f1}  thr={thr:.3f}")
    _model_cache[str(ckpt)] = (m, dev, thr)
    return m, dev, thr


def cnn_score(y: np.ndarray, sr: int, model, dev) -> float:
    feat = feature_stack_v15(y, sr)
    if feat is None: return 0.0
    try:
        import torch
        t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            return float(torch.softmax(model(t), dim=1)[0, 1].item())
    except Exception:
        return 0.0


def resample(y: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst: return y
    from math import gcd
    g = gcd(src, dst)
    return sp_signal.resample_poly(y, dst//g, src//g).astype(np.float32)


def windows(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    starts = list(range(0, max(1, n-win+1), hop))
    if n > win and starts[-1] != n-win: starts.append(n-win)
    elif n <= win: starts = [0]
    return [(s, min(s+win, n)) for s in starts]


def pad_or_trim(y: np.ndarray, n: int) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n-len(y)))


def score_segment(y: np.ndarray, sr: int,
                  model=None, dev=None, cnn_w: float = 0.80,
                  bpf_hz: Optional[float] = None) -> Dict:
    h = heuristic_score(y, sr, bpf_hz=bpf_hz)
    if model is not None and LIBROSA_OK:
        c = cnn_score(y, sr, model, dev)
        fused = cnn_w*c + (1-cnn_w)*h
        if c > 0.45 and h > 0.45: fused = min(1.0, fused + 0.06)
    else:
        c, fused = float("nan"), h
    c_val = None if (isinstance(c, float) and math.isnan(c)) else round(c, 5)
    f_val = round(float(np.clip(fused, 0, 1)), 5)
    return {
        "heuristic_prob": round(h, 5), "cnn_prob": c_val, "fused_prob": f_val,
        "heuristic_probability": round(h, 5),
        "cnn_probability": c_val, "probability": f_val,
    }


# =============================================================================
#  MAIN SEGMENT EXTRACTOR
# =============================================================================

def extract_indoor_segments(
    audio_dir: Path,
    markers_df: pd.DataFrame,
    pos_dfs: Dict[str, pd.DataFrame],
    q4_df: Optional[pd.DataFrame],
    args,
    model=None, dev=None,
) -> pd.DataFrame:
    """
    For each inter-marker interval, slide a window and score each segment.
    Attaches full metadata (phase, drone, maneuver, altitude, GT label).
    """
    win_n = int(args.segment_sec * MODEL_SR)
    hop_n = int(args.hop_sec    * MODEL_SR)
    rows  = []

    # Build BPF lookup from q4_features.csv if available
    bpf_lookup: Dict[str, float] = dict(BPF_HZ)  # fallback to constants
    if q4_df is not None and "drone" in q4_df.columns and "bpf_hz" in q4_df.columns:
        drone_map = {"Pro": "mavic_pro", "Mv2": "mavic_pro2", "Mini": "mavic_mini"}
        for _, qr in q4_df.iterrows():
            key = drone_map.get(str(qr["drone"]))
            if key:
                # Use altitude-matched value when available
                alt = qr.get("altitude_m")
                bpf = float(qr["bpf_hz"])
                if alt == 3:   # representative altitude
                    bpf_lookup[key] = bpf

    n_markers = len(markers_df)
    phases_to_run = [args.phase] if args.phase else [1, 2, 3]
    if args.include_free_run:
        phases_to_run.append(4)

    for i in range(n_markers - 1):
        row_i   = markers_df.iloc[i]
        row_i1  = markers_df.iloc[i + 1]
        phase   = int(row_i["phase"])

        if phase not in phases_to_run:
            continue

        drone_key   = PHASE_DRONE[phase]
        drone_label = PHASE_LABEL[phase]
        maneuver    = str(row_i["maneuver"])

        if args.maneuver and args.maneuver != "all" and maneuver != args.maneuver:
            continue

        s_start = int(row_i["sample"])
        s_end   = int(row_i1["sample"])
        dur_s   = (s_end - s_start) / INDOOR_SR

        if dur_s < args.segment_sec * 0.5 or dur_s > 600:
            continue

        gt_label = assign_gt_indoor(
            (s_start + s_end) // 2, markers_df, args.include_free_run)

        bpf_hz = bpf_lookup.get(drone_key)

        print(f"  [{phase}] {drone_label:<20} {maneuver:<12} "
              f"{dur_s:5.1f}s  gt={gt_label}", end="\r", flush=True)

        # Load and resample audio
        y_raw = average_group(audio_dir, args.group, s_start, s_end, args.channel)
        if y_raw is None or len(y_raw) == 0:
            continue
        y_rs = resample(y_raw, INDOOR_SR, MODEL_SR)

        for ws, we in windows(len(y_rs), win_n, hop_n):
            clip = pad_or_trim(y_rs[ws:we], win_n)
            t_off = s_start / INDOOR_SR + ws / MODEL_SR
            sample_mid = s_start + int(ws / MODEL_SR * INDOOR_SR)

            rms_db = float(20 * math.log10(
                np.sqrt(np.mean(clip.astype(np.float64)**2)) + 1e-8))

            sc = score_segment(clip, MODEL_SR, model, dev,
                               args.cnn_weight, bpf_hz=bpf_hz)

            # Position lookup
            pos = position_at_sample(pos_dfs, drone_key, sample_mid)

            row: Dict[str, Any] = {
                # Identity
                "session":           SESSION_ID,
                "domain":            DOMAIN,
                "drone_model":       drone_label,
                "drone_key":         drone_key,
                "phase":             phase,
                "maneuver":          maneuver,
                "nominal_altitude_m": row_i.get("nominal_altitude_m"),
                # Timing
                "marker_index_start": int(row_i["index"]),
                "marker_index_end":   int(row_i1["index"]),
                "sample_start":       sample_mid,
                "offset_sec":         round(t_off, 3),
                # Acoustics
                "rms_db":             round(rms_db, 2),
                "bpf_hz_ref":         round(bpf_hz, 1) if bpf_hz else None,
                # Position (if available)
                "z_rel":   round(pos["z_rel"],   3) if pos else None,
                "x_rel":   round(pos["x_rel"],   3) if pos else None,
                "y_rel":   round(pos["y_rel"],   3) if pos else None,
                # GT
                "gt_label": gt_label,
                # Scores
                **sc,
            }
            rows.append(row)

    print()  # newline after \r progress
    return pd.DataFrame(rows)


# =============================================================================
#  METRICS
# =============================================================================

def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    best_f1, best_thr = 0.0, 0.62
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (y_score >= thr).astype(int)
        tp = int(np.sum((pred==1)&(y_true==1)))
        fp = int(np.sum((pred==1)&(y_true==0)))
        fn = int(np.sum((pred==0)&(y_true==1)))
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        if f1 > best_f1: best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def compute_metrics(df: pd.DataFrame, fixed_thr: Optional[float] = None,
                    score_col: str = "fused_prob") -> Dict:
    if df.empty or score_col not in df.columns: return {}
    sub = df[df["gt_label"] != -1].copy()
    if sub.empty: return {"note": "all ambiguous"}
    y_true  = sub["gt_label"].values.astype(int)
    y_score = sub[score_col].fillna(0).values.astype(float)
    n_pos = int(y_true.sum()); n_neg = int((y_true==0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"n_segments": len(sub), "n_drone": n_pos, "n_non_drone": n_neg,
                "note": "single-class"}
    if fixed_thr is not None:
        thr = fixed_thr; _, bf = best_threshold(y_true, y_score)
    else:
        thr, bf = best_threshold(y_true, y_score)
    pred = (y_score >= thr).astype(int)
    tp = int(np.sum((pred==1)&(y_true==1))); fp = int(np.sum((pred==1)&(y_true==0)))
    tn = int(np.sum((pred==0)&(y_true==0))); fn = int(np.sum((pred==0)&(y_true==1)))
    pre = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0.0
    out = {"n_segments":len(sub), "n_drone":n_pos, "n_non_drone":n_neg,
           "threshold":round(thr,3), "accuracy":round((tp+tn)/max(tp+tn+fp+fn,1),4),
           "precision":round(pre,4), "recall":round(rec,4), "f1":round(f1,4),
           "best_f1_at_best_thr":round(bf,4), "tp":tp,"fp":fp,"tn":tn,"fn":fn}
    if SKLEARN_OK and len(np.unique(y_true)) > 1:
        out["auroc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["auprc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return out


# =============================================================================
#  PLOTTING
# =============================================================================

def _ax(ax):
    ax.grid(True, alpha=0.3, lw=0.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)


DRONE_COLORS = {"mavic_pro": P["pro"], "mavic_pro2": P["mv2"],
                "mavic_mini": P["mini"]}


def plot_timeline_per_drone(df: pd.DataFrame, thr: float, output_dir: Path):
    if df.empty or "drone_key" not in df.columns: return
    sub = df[df["gt_label"] != -1].copy()
    drones = sub["drone_key"].unique()
    fig, axes = plt.subplots(len(drones), 1,
                             figsize=(22, 5*len(drones)), facecolor=P["bg"])
    if len(drones) == 1: axes = [axes]
    fig.suptitle("Indoor Benchmark — Detection Timeline per Drone",
                 fontsize=12, color=P["drone"], fontweight="bold")
    for ax, dk in zip(axes, drones):
        sdf = sub[sub["drone_key"] == dk]
        t   = sdf["offset_sec"].values
        sc  = sdf["fused_prob"].fillna(0).values
        gt  = sdf["gt_label"].values
        pred= (sc >= thr).astype(int)
        tp_m=(pred==1)&(gt==1); fp_m=(pred==1)&(gt==0)
        tn_m=(pred==0)&(gt==0); fn_m=(pred==0)&(gt==1)
        ax.scatter(t[gt==0], sc[gt==0], c=P["muted"], s=7, alpha=0.35)
        ax.scatter(t[gt==1], sc[gt==1], c=DRONE_COLORS.get(dk, P["drone"]),
                   s=11, alpha=0.8)
        ax.scatter(t[fn_m], sc[fn_m],   c=P["err"],  s=14, zorder=5, marker="x")
        ax.scatter(t[fp_m], sc[fp_m],   c=P["warn"], s=14, zorder=5, marker="^")
        ax.axhline(thr, color=P["warn"], lw=1.2, ls="--")
        ax.set_title(PHASE_LABEL.get(
            int(sdf["phase"].mode()[0]), dk), color=P["drone"])
        ax.set_ylabel("Probability"); ax.set_ylim(-0.05, 1.10); _ax(ax)
    axes[-1].set_xlabel("Offset within session (s)")
    plt.tight_layout()
    out = output_dir / "plots" / "timeline_per_drone.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


def plot_roc_pr(df: pd.DataFrame, metrics: Dict, output_dir: Path):
    if not SKLEARN_OK or df.empty: return
    sub = df[df["gt_label"] != -1]
    y_true  = sub["gt_label"].values.astype(int)
    y_score = sub["fused_prob"].fillna(0).values.astype(float)
    if len(np.unique(y_true)) < 2: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=P["bg"])
    fig.suptitle("Indoor Benchmark — ROC & PR Curves", fontsize=11, color=P["drone"])
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
    out = output_dir / "plots" / "roc_pr.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


def plot_confusion(df: pd.DataFrame, thr: float, output_dir: Path):
    if df.empty: return
    sub  = df[df["gt_label"] != -1]
    yt   = sub["gt_label"].values.astype(int)
    yp   = (sub["fused_prob"].fillna(0).values >= thr).astype(int)
    cm   = confusion_matrix(yt, yp)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dk", [P["panel"], P["drone"]])
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    im = ax.imshow(cm, cmap=cmap); plt.colorbar(im, ax=ax)
    lbls = ["non_drone","drone"]
    ax.set_xticks([0,1]); ax.set_xticklabels(lbls)
    ax.set_yticks([0,1]); ax.set_yticklabels(lbls)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    tot = cm.sum(axis=1, keepdims=True) + 1e-8
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({100*cm[i,j]/tot[i,0]:.1f}%)",
                    ha="center", va="center", color=P["text"],
                    fontsize=12, fontweight="bold")
    fig.suptitle(f"Indoor Confusion Matrix (thr={thr:.2f})",
                 fontsize=11, color=P["drone"])
    plt.tight_layout()
    out = output_dir / "plots" / "confusion.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


def plot_score_dist_by_maneuver(df: pd.DataFrame, output_dir: Path):
    if df.empty or "maneuver" not in df.columns: return
    sub      = df[df["gt_label"] != -1]
    maneuvers= sub["maneuver"].value_counts().index.tolist()[:8]
    colors   = [P["drone"], P["ok"], P["mv2"], P["warn"], P["err"],
                P["purp"], P["muted"], P["pro"]]
    fig, ax  = plt.subplots(figsize=(14, 6), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    for i, mv in enumerate(maneuvers):
        s = sub[sub["maneuver"]==mv]["fused_prob"].dropna()
        if len(s): ax.hist(s.values, bins=30, alpha=0.6, color=colors[i%len(colors)],
                           label=f"{mv} (n={len(s)})", density=True)
    ax.set_xlabel("Fused probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); _ax(ax)
    fig.suptitle("Score Distribution by Maneuver Type",
                 fontsize=11, color=P["drone"])
    plt.tight_layout()
    out = output_dir / "plots" / "score_dist_by_maneuver.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


def plot_bpf_vs_score(df: pd.DataFrame, q4_df: Optional[pd.DataFrame],
                      output_dir: Path):
    """Scatter: BPF energy ratio (from q4_features.csv) vs heuristic score."""
    if q4_df is None or df.empty: return
    if "bpf_energy_ratio" not in q4_df.columns: return
    # Merge on drone + altitude
    drone_map = {"Pro": "mavic_pro", "Mv2": "mavic_pro2", "Mini": "mavic_mini"}
    q4 = q4_df.copy()
    q4["drone_key"] = q4["drone"].map(drone_map)
    merged = df.merge(
        q4[["drone_key","altitude_m","bpf_hz","bpf_energy_ratio"]],
        left_on=["drone_key","nominal_altitude_m"],
        right_on=["drone_key","altitude_m"],
        how="left")
    if merged.empty or "bpf_energy_ratio" not in merged.columns: return
    sub = merged.dropna(subset=["bpf_energy_ratio", "heuristic_prob"])
    if sub.empty: return
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    for dk, col in DRONE_COLORS.items():
        s = sub[sub["drone_key"]==dk]
        if not s.empty:
            ax.scatter(s["bpf_energy_ratio"], s["heuristic_prob"],
                       c=col, alpha=0.6, s=25,
                       label=PHASE_LABEL.get(PHASE_DRONE.get(
                           [k for k,v in PHASE_DRONE.items() if v==dk and k<4][0], 1), dk))
    ax.set_xlabel("BPF Energy Ratio (q4_features.csv)")
    ax.set_ylabel("Heuristic Drone Score")
    ax.legend(fontsize=9); _ax(ax)
    fig.suptitle("BPF Energy Ratio vs Heuristic Score (Hover Segments)",
                 fontsize=11, color=P["drone"])
    plt.tight_layout()
    out = output_dir / "plots" / "bpf_vs_score.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


def plot_cross_domain(indoor_metrics: Dict,
                      outdoor_summary_path: Optional[str],
                      output_dir: Path):
    """Side-by-side bar chart: indoor vs outdoor top-line metrics."""
    if outdoor_summary_path is None: return
    try:
        with open(outdoor_summary_path) as f:
            outdoor = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Could not load outdoor summary: {e}"); return

    # Extract best outdoor metrics across all runs
    outdoor_runs = outdoor.get("runs", [])
    if not outdoor_runs: return
    best_outdoor = max(outdoor_runs,
                       key=lambda r: float(r.get("metrics",{}).get("f1",0) or 0))
    out_m = best_outdoor.get("metrics", {})

    metric_names = ["f1", "auroc", "precision", "recall", "accuracy"]
    ind_vals = [float(indoor_metrics.get(m) or 0) for m in metric_names]
    out_vals = [float(out_m.get(m) or 0)           for m in metric_names]

    x  = np.arange(len(metric_names))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    bars_i = ax.bar(x - w/2, ind_vals, w, color=P["drone"], alpha=0.8, label="Indoor (Vitemor)")
    bars_o = ax.bar(x + w/2, out_vals, w, color=P["ok"],    alpha=0.8, label="Outdoor (Dunakeszi)")
    for bars in [bars_i, bars_o]:
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=8, color=P["text"])
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.legend(fontsize=9, loc="lower right"); _ax(ax)
    fig.suptitle("Cross-Domain Comparison: Indoor vs Outdoor",
                 fontsize=12, color=P["drone"], fontweight="bold")
    plt.tight_layout()
    out = output_dir / "plots" / "cross_domain_comparison.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  📊 {out.name}")


# =============================================================================
#  PER-DRONE AND PER-MANEUVER REPORT
# =============================================================================

def per_drone_report(df: pd.DataFrame, fixed_thr: Optional[float]) -> Dict:
    report = {}
    for dk in df["drone_key"].unique():
        sub = df[df["drone_key"] == dk]
        m   = compute_metrics(sub, fixed_thr)
        report[dk] = m
        print(f"  {dk:<16}  F1={m.get('f1','?')}  AUROC={m.get('auroc','?')}  "
              f"n_drone={m.get('n_drone','?')}  n_neg={m.get('n_non_drone','?')}")
    return report


def per_maneuver_report(df: pd.DataFrame, fixed_thr: Optional[float]) -> Dict:
    report = {}
    for mv in df["maneuver"].unique():
        sub = df[df["maneuver"] == mv]
        m   = compute_metrics(sub, fixed_thr)
        report[mv] = m
    return report


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
    print(f"  VITEMOR INDOOR BENCHMARK")
    print(f"  Session  : {SESSION_ID}")
    print(f"  Source   : {args.group}  channel={args.channel}")
    print(f"  Phase    : {args.phase or '1-3'}")
    print(f"  Maneuver : {args.maneuver or 'all'}")
    print(f"  Segment  : {args.segment_sec}s  hop={args.hop_sec}s")
    print(f"  Output   : {output_dir}")
    print(SEP2)

    # ── Discover paths ────────────────────────────────────────────────────────
    paths = discover_paths(args)

    # ── Load markers ──────────────────────────────────────────────────────────
    if paths["markers_dir"] is None:
        print("❌ markers/ directory not found — cannot continue")
        sys.exit(1)
    print(f"\n{SEP}\n  LOADING MARKERS\n{SEP}")
    markers_df = load_markers(paths["markers_dir"])

    # ── Load telemetry ────────────────────────────────────────────────────────
    pos_dfs: Dict[str, pd.DataFrame] = {}
    if paths["position_dir"]:
        print(f"\n{SEP}\n  LOADING TELEMETRY\n{SEP}")
        pos_dfs = load_indoor_position(paths["position_dir"])

    # ── Load q4_features ─────────────────────────────────────────────────────
    q4_df = None
    if paths["q4_csv"]:
        try:
            q4_df = pd.read_csv(paths["q4_csv"])
            print(f"\n  q4_features: {len(q4_df)} rows loaded")
        except Exception as e:
            print(f"  ⚠️  q4_features load failed: {e}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = dev = None
    model_thr = args.threshold or 0.62
    if not args.no_model and TORCH_OK and LIBROSA_OK:
        model, dev, model_thr = load_model(paths["model_dir"])

    # ── Extract segments ──────────────────────────────────────────────────────
    print(f"\n{SEP}\n  EXTRACTING SEGMENTS\n{SEP}")
    if paths["audio_dir"] is None:
        print("❌ audio_files/ directory not found — cannot continue")
        sys.exit(1)

    df = extract_indoor_segments(
        audio_dir=paths["audio_dir"],
        markers_df=markers_df,
        pos_dfs=pos_dfs,
        q4_df=q4_df,
        args=args,
        model=model, dev=dev,
    )

    if df.empty:
        print("❌ No segments extracted — check paths and marker file")
        sys.exit(1)

    print(f"\n  Extracted: {len(df)} segments  "
          f"drone={int((df['gt_label']==1).sum())}  "
          f"non_drone={int((df['gt_label']==0).sum())}  "
          f"ambiguous={int((df['gt_label']==-1).sum())}")

    df.to_csv(output_dir / "indoor_segments.csv", index=False)
    print(f"  💾 indoor_segments.csv")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print(f"\n{SEP}\n  METRICS\n{SEP}")
    overall = compute_metrics(df, args.threshold)
    thr      = float(overall.get("threshold", model_thr))

    (output_dir / "indoor_report.json").write_text(
        json.dumps(overall, indent=2, default=str))
    print(f"\n  Overall:  F1={overall.get('f1','?')}  "
          f"AUROC={overall.get('auroc','?')}  thr={thr:.3f}")

    print(f"\n  Per-drone:")
    drone_rep = per_drone_report(df, args.threshold)
    (output_dir / "indoor_per_drone_report.json").write_text(
        json.dumps(drone_rep, indent=2, default=str))

    maneuver_rep = per_maneuver_report(df, args.threshold)
    (output_dir / "indoor_per_maneuver_report.json").write_text(
        json.dumps(maneuver_rep, indent=2, default=str))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  PLOTS\n{SEP}")
    plot_timeline_per_drone(df, thr, output_dir)
    plot_roc_pr(df, overall, output_dir)
    plot_confusion(df, thr, output_dir)
    plot_score_dist_by_maneuver(df, output_dir)
    plot_bpf_vs_score(df, q4_df, output_dir)
    plot_cross_domain(overall, args.outdoor_summary, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "generated":   datetime.utcnow().isoformat(),
        "session":     SESSION_ID,
        "domain":      DOMAIN,
        "args": {
            "group":       args.group,
            "channel":     args.channel,
            "phase":       args.phase,
            "maneuver":    args.maneuver,
            "segment_sec": args.segment_sec,
            "hop_sec":     args.hop_sec,
            "scoring":     "cnn+heuristic" if model else "heuristic_only",
        },
        "overall_metrics":  overall,
        "per_drone_metrics": drone_rep,
        "per_maneuver_metrics": maneuver_rep,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"  💾 benchmark_summary.json")

    print(f"\n{SEP2}")
    print("  INDOOR BENCHMARK COMPLETE")
    print(SEP2)
    print(f"  F1={overall.get('f1','?')}  AUROC={overall.get('auroc','?')}  "
          f"Prec={overall.get('precision','?')}  Recall={overall.get('recall','?')}")
    print(f"  Outputs → {output_dir}\n")


if __name__ == "__main__":
    main()
