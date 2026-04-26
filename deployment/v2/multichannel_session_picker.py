#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multichannel_session_picker.py
══════════════════════════════
Multichannel-aware drone session picker for the Dunakeszi 12-channel rig.

Uses the FULL drone_detection pipeline (CNN + heuristic) for detection
rather than simple RMS/BPF heuristics, plus:

  • Sliding-window DPA detection over the full recording
  • Per-channel spectrogram display for each candidate segment
  • Ground truth extraction from GPX + RPM sidecar files
  • Interactive trim and save of clean sessions with rich _meta.json

Channel layout
──────────────
  Ch  0–4  : DPA stand A  (professional capsules, linear rail, 5 cm spacing)
  Ch  5–9  : DPA stand B  (same — cross-stand strategies need separation set)
  Ch 10–11 : XMOS MEMS   (independent USB clock — NOT synced with DPA)

Usage
─────
  python multichannel_session_picker.py \\
      --wav    251020VITEMOROM1AT01U.wav \\
      --output sessions/ \\
      --gpx    gpx_combined.csv \\
      --rpm    rotor_rpm_ch0.csv \\
      --window-sec 5.0 \\
      --stride-sec 2.5 \\
      --strategy stand_a

  # With known stand separation (enables cross-stand localisation):
  python multichannel_session_picker.py --wav FILE.wav --stand-separation-m 1.35

  python multichannel_session_picker.py --wav 251020VITEMOROM1AT01U.wav --output sessions_out/ --gpx Dunakeszi_Data/output/gpx_combined.csv --rpm Dunakeszi_Data/output/rotor_rpm_ch0.csv --flight-takeoff 2025-10-20T12:57:21 --flight-landing 2025-10-20T13:04:09 --window-sec 5.0 --stride-sec 2.5

  python3 -c "
import pandas as pd
df = pd.read_csv('dunakeszi_data/audit/dunakeszi_audit_output/gpx_combined.csv', parse_dates=['time'])
print(df[(df.time > '2025-10-20 12:58') & (df.time < '2025-10-20 13:10')].head(20))
"
Dependencies
────────────
  numpy scipy soundfile matplotlib pandas
  + drone_detection package in same directory (drone_detection/)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for saving
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    pd = None           # type: ignore
    PANDAS_OK = False

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

# ── insert v2/ into path so drone_detection is importable ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ══════════════════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WINDOW_SEC   = 5.0
DEFAULT_STRIDE_SEC   = 2.5
DEFAULT_SAVE_SR      = 22050
DEFAULT_DPA_STRATEGY = "stand_a"
DEFAULT_MAX_DRONES   = 3

BRUEL_FILE_STARTS: Dict[str, datetime] = {
    "251020VITEMOROM1AT01I": datetime(2025, 10, 20, 12, 50, 34, tzinfo=timezone.utc),
    "251020VITEMOROM1AT01J": datetime(2025, 10, 20, 12, 59, 28, tzinfo=timezone.utc),
    "251020VITEMOROM1AT01U": datetime(2025, 10, 20, 13,  0,  0, tzinfo=timezone.utc),
}

TAKEOFF_RAMP_SEC = 30
LANDING_RAMP_SEC = 30

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline loader (lazy — avoids import errors if models missing)
# ══════════════════════════════════════════════════════════════════════════════

_cfg         = None
_detect_fn   = None
_localize_fn = None
_ap          = None


def _load_pipeline(models_dir: Optional[Path] = None):
    global _cfg, _detect_fn, _localize_fn, _ap
    if _detect_fn is not None:
        return

    from drone_detection.config import config
    _cfg = config
    if models_dir is not None:
        _cfg.DRIVE_MODELS = Path(models_dir)

    from drone_detection.inference import load_detection_model, detect
    load_detection_model(_cfg)
    _detect_fn = detect

    from drone_detection.multidrone import localize_multi_drone
    _localize_fn = localize_multi_drone

    from drone_detection.audio_processing import AudioProcessor
    _ap = AudioProcessor(_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Audio I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resample(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from math import gcd
    g = gcd(src_sr, dst_sr)
    return signal.resample_poly(y, dst_sr // g, src_sr // g).astype(np.float32)


def load_channels(
    path: Path,
    indices: List[int],
    target_sr: int,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> Tuple[List[np.ndarray], int]:
    """Load specific channels from a WAV, optionally sliced, resampled."""
    info   = sf.info(str(path))
    src_sr = info.samplerate
    n_ch   = info.channels
    s0     = int(start_s * src_sr)
    s1     = int(end_s * src_sr) if end_s is not None else info.frames

    data, _ = sf.read(str(path), start=s0, stop=s1,
                      dtype="float32", always_2d=True)
    channels = []
    for idx in indices:
        if idx >= n_ch:
            raise ValueError(f"Channel {idx} does not exist (file has {n_ch} ch)")
        ch = data[:, idx]
        if src_sr != target_sr:
            ch = _resample(ch, src_sr, target_sr)
        channels.append(ch.astype(np.float32))
    return channels, target_sr


def get_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def get_native_sr(path: Path) -> int:
    return sf.info(str(path)).samplerate


# ══════════════════════════════════════════════════════════════════════════════
# Sliding-window detection
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window_detect(
    wav_path:    Path,
    dpa_channel: int,
    window_sec:  float,
    stride_sec:  float,
    total_sec:   float,
) -> List[Dict]:
    """
    Run detect() on overlapping windows of a single DPA channel.
    Returns a list of window result dicts with added 'start_s' / 'end_s'.
    """
    results = []
    t = 0.0
    sr = _cfg.SR

    print(f"  Scanning {total_sec:.1f}s in {window_sec}s windows "
          f"(stride {stride_sec}s)…")

    while t < total_sec:
        end = min(t + window_sec, total_sec)
        try:
            (ch,), _ = load_channels(wav_path, [dpa_channel], sr,
                                     start_s=t, end_s=end)
            det = _detect_fn([ch], _cfg)
            det["start_s"] = round(t, 3)
            det["end_s"]   = round(end, 3)
            results.append(det)

            prob = det["probability"]
            mark = "🚁" if det["detected"] else "·"
            print(f"  {mark}  {t:7.1f}s–{end:7.1f}s  prob={prob:.3f}",
                  flush=True)
        except Exception as e:
            print(f"  ⚠  window {t:.1f}s: {e}")
        t += stride_sec
        gc.collect()

    return results


def merge_detected_windows(
    windows: List[Dict],
    merge_gap_sec: float = 2.0,
    min_duration_sec: float = 2.0,
) -> List[Dict]:
    """
    Merge overlapping/adjacent detected windows into candidate segments.
    Returns list of dicts: start_s, end_s, peak_prob, mean_prob, n_windows.
    """
    hot = [w for w in windows if w["detected"]]
    if not hot:
        return []

    segments = []
    cur_s    = hot[0]["start_s"]
    cur_e    = hot[0]["end_s"]
    probs    = [hot[0]["probability"]]

    for w in hot[1:]:
        if w["start_s"] - cur_e <= merge_gap_sec:
            cur_e = max(cur_e, w["end_s"])
            probs.append(w["probability"])
        else:
            segments.append({
                "start_s":  cur_s,
                "end_s":    cur_e,
                "peak_prob": round(max(probs), 4),
                "mean_prob": round(float(np.mean(probs)), 4),
                "n_windows": len(probs),
            })
            cur_s = w["start_s"]
            cur_e = w["end_s"]
            probs = [w["probability"]]

    segments.append({
        "start_s":  cur_s,
        "end_s":    cur_e,
        "peak_prob": round(max(probs), 4),
        "mean_prob": round(float(np.mean(probs)), 4),
        "n_windows": len(probs),
    })

    return [s for s in segments if (s["end_s"] - s["start_s"]) >= min_duration_sec]


# ══════════════════════════════════════════════════════════════════════════════
# Spectrogram display
# ══════════════════════════════════════════════════════════════════════════════

CHANNEL_LABELS = {
    0: "DPA-A0", 1: "DPA-A1", 2: "DPA-A2", 3: "DPA-A3", 4: "DPA-A4",
    5: "DPA-B0", 6: "DPA-B1", 7: "DPA-B2", 8: "DPA-B3", 9: "DPA-B4",
    10: "MEMS-0", 11: "MEMS-1",
}


def plot_segment_spectrograms(
    wav_path:    Path,
    seg:         Dict,
    display_chs: List[int],
    save_path:   Path,
    title:       str = "",
    drones:      Optional[List[Dict]] = None,
) -> None:
    """
    Plot spectrograms for all display_chs over the candidate segment.
    Saves a PNG to save_path.
    """
    if not MATPLOTLIB_OK:
        print("  (matplotlib not available — skipping spectrogram)")
        return

    sr      = _cfg.SR if _cfg else 22050
    start_s = seg["start_s"]
    end_s   = seg["end_s"]
    dur     = end_s - start_s
    n_ch    = len(display_chs)

    # Add 1s padding each side for context
    pad     = 1.0
    load_s  = max(0.0, start_s - pad)
    load_e  = end_s + pad

    try:
        channels, _ = load_channels(wav_path, display_chs, sr,
                                    start_s=load_s, end_s=load_e)
    except Exception as e:
        print(f"  ⚠  Spectrogram load error: {e}")
        return

    # ── layout ────────────────────────────────────────────────────────────────
    fig_h   = max(3 * n_ch, 8)
    fig, axes = plt.subplots(n_ch, 1, figsize=(18, fig_h), sharex=True)
    if n_ch == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    cmap = "inferno"

    for row, (ch_idx, ch_audio) in enumerate(zip(display_chs, channels)):
        ax    = axes[row]
        label = CHANNEL_LABELS.get(ch_idx, f"Ch{ch_idx}")

        # spectrogram
        nperseg = min(2048, len(ch_audio) // 8)
        noverlap = nperseg * 3 // 4
        f_arr, t_arr, Sxx = signal.spectrogram(
            ch_audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
            scaling="density",
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmin   = float(np.percentile(Sxx_db, 5))
        vmax   = float(np.percentile(Sxx_db, 99))

        t_abs = t_arr + load_s   # absolute time in file

        ax.pcolormesh(t_abs, f_arr, Sxx_db,
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")

        # highlight the detected segment
        ax.axvspan(start_s, end_s, color="#39d353", alpha=0.08, zorder=2)
        ax.axvline(start_s, color="#39d353", lw=1.2, ls="--", alpha=0.7)
        ax.axvline(end_s,   color="#39d353", lw=1.2, ls="--", alpha=0.7)

        # clock domain separator
        if ch_idx >= 10:
            ax.text(load_s + 0.1, sr * 0.45,
                    "⚠ INDEPENDENT CLOCK",
                    color="#f0a500", fontsize=7, va="top",
                    fontfamily="monospace")

        ax.set_ylim(0, min(sr / 2, 8000))
        ax.set_ylabel(label, color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── drone annotation on top panel ─────────────────────────────────────────
    if drones:
        ax0 = axes[0]
        for i, d in enumerate(drones, 1):
            az   = d.get("azimuth_deg", 0)
            dist = d.get("distance_m", "?")
            ax0.text(
                end_s + 0.1, 6500 - i * 600,
                f"Drone {i}: az={az:.0f}° dist={dist:.1f}m",
                color="#58a6ff", fontsize=8, va="top",
            )

    axes[-1].set_xlabel("Time in file (s)", color="#c9d1d9", fontsize=9)

    prob_str = f"peak={seg['peak_prob']:.3f}  mean={seg['mean_prob']:.3f}"
    fig.suptitle(
        f"{title}   |   {start_s:.1f}s – {end_s:.1f}s  ({dur:.1f}s)   |   {prob_str}",
        color="#e6edf3", fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Spectrogram saved: {save_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth extraction
# ══════════════════════════════════════════════════════════════════════════════

def file_start_utc(wav_path: Path) -> Optional[datetime]:
    stem = wav_path.stem.upper()
    for key, dt in BRUEL_FILE_STARTS.items():
        if key.upper() in stem:
            return dt
    return None


def extract_gpx_window(
    gpx_df,
    t0_utc: datetime,
    t1_utc: datetime,
) -> List[Dict]:
    if gpx_df is None or gpx_df.empty:
        return []
    ts0 = pd.Timestamp(t0_utc, tz="UTC")
    ts1 = pd.Timestamp(t1_utc, tz="UTC")
    sub = gpx_df[(gpx_df["time"] >= ts0) & (gpx_df["time"] <= ts1)].copy()
    if sub.empty:
        return []
    sub["time"] = sub["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    keep = [c for c in ("time","lat","lon","elevation","speed","source")
            if c in sub.columns]
    return sub[keep].replace({float("nan"): None}).to_dict(orient="records")


def extract_rpm_window(
    rpm_df,
    t0_utc: datetime,
    t1_utc: datetime,
    takeoff_utc: datetime,
) -> Dict:
    if rpm_df is None or rpm_df.empty:
        return {}
    s0 = (t0_utc - takeoff_utc).total_seconds()
    s1 = (t1_utc - takeoff_utc).total_seconds()
    sub = rpm_df[(rpm_df["time_s"] >= s0) & (rpm_df["time_s"] <= s1)]
    if sub.empty:
        return {}
    out: Dict = {}
    for col in ("rpm", "rpm_smooth", "bpf_hz", "bpf_hz_smooth"):
        if col in sub.columns:
            vals = sub[col].dropna()
            if not vals.empty:
                out[f"{col}_mean"] = round(float(vals.mean()), 1)
                out[f"{col}_min"]  = round(float(vals.min()), 1)
                out[f"{col}_max"]  = round(float(vals.max()), 1)
    return out


def infer_flight_phase(
    t0: datetime, t1: datetime,
    takeoff: Optional[datetime], landing: Optional[datetime],
) -> str:
    if takeoff is None or landing is None:
        return "unknown"
    mid = t0 + (t1 - t0) / 2
    if mid < takeoff:
        return "pre-flight"
    if mid > landing:
        return "post-flight"
    if mid <= takeoff + timedelta(seconds=TAKEOFF_RAMP_SEC):
        return "takeoff"
    if mid >= landing - timedelta(seconds=LANDING_RAMP_SEC):
        return "landing"
    return "cruise"


# ══════════════════════════════════════════════════════════════════════════════
# Signal metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_signal_metrics(y: np.ndarray, sr: int) -> Dict:
    y   = np.asarray(y, dtype=np.float64)
    rms = float(np.sqrt(np.mean(y ** 2)))
    pk  = float(np.max(np.abs(y)))
    out: Dict = {
        "rms_dbfs":    round(20 * math.log10(rms  + 1e-12), 2),
        "peak_dbfs":   round(20 * math.log10(pk   + 1e-12), 2),
        "crest_factor": round(pk / (rms + 1e-9), 3),
    }
    frame = max(int(0.1 * sr), 64)
    n_fr  = len(y) // frame
    if n_fr >= 4:
        frames = y[:n_fr * frame].reshape(n_fr, frame)
        fr_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        noise  = float(np.percentile(fr_rms, 10)) + 1e-12
        sig    = float(np.percentile(fr_rms, 90)) + 1e-12
        out["snr_db"] = round(20 * math.log10(sig / noise), 2)
    try:
        nperseg = min(4096, max(256, len(y) // 8))
        f_psd, psd = signal.welch(y, fs=sr, nperseg=nperseg)
        denom = float(np.sum(psd)) + 1e-10
        out["spectral_centroid_hz"] = round(float(np.sum(f_psd * psd) / denom), 1)
        peaks, props = signal.find_peaks(
            psd, height=psd.max() * 0.05, distance=3)
        if len(peaks):
            top = peaks[np.argmax(props["peak_heights"])]
            out["dominant_freq_hz"] = round(float(f_psd[top]), 2)
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Localization for a segment
# ══════════════════════════════════════════════════════════════════════════════

def localize_segment(
    wav_path:    Path,
    seg:         Dict,
    strategy:    str,
    max_drones:  int,
    stand_sep:   Optional[float],
) -> List[Dict]:
    """Run multi-drone localization on the segment's DPA triplet."""
    if _localize_fn is None or _cfg is None:
        return []

    # Import multichannel helpers
    try:
        from multichannel_inference import (
            build_dpa_triplet, _make_cfg_for_triplet,
            DPA_MIC_POSITIONS_M, DEFAULT_CHANNEL_MAP,
            set_stand_separation,
        )
        if stand_sep is not None:
            set_stand_separation(stand_sep)
    except ImportError as e:
        print(f"  ⚠  multichannel_inference not importable: {e}")
        return []

    sr       = _cfg.SR
    ch_map   = DEFAULT_CHANNEL_MAP
    dpa_all  = ch_map.get("dpa_a", []) + ch_map.get("dpa_b", [])

    try:
        channels, _ = load_channels(
            wav_path, dpa_all, sr,
            start_s=seg["start_s"], end_s=seg["end_s"],
        )
    except Exception as e:
        print(f"  ⚠  Load for localization failed: {e}")
        return []

    all_ch_dict = {idx: ch for idx, ch in zip(dpa_all, channels)}

    try:
        triplet_chs, triplet_idx, _ = build_dpa_triplet(
            dpa_all, all_ch_dict, strategy=strategy,
        )
        cfg2   = _make_cfg_for_triplet(_cfg, triplet_idx, DPA_MIC_POSITIONS_M)
        drones = _localize_fn(triplet_chs, cfg2, max_drones)
        return drones
    except Exception as e:
        print(f"  ⚠  Localization error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Save clean session
# ══════════════════════════════════════════════════════════════════════════════

def save_session(
    wav_path:    Path,
    seg:         Dict,
    start_s:     float,
    end_s:       float,
    out_dir:     Path,
    save_sr:     int,
    drones:      List[Dict],
    gpx_data:    List[Dict],
    rpm_data:    Dict,
    flight_phase: str,
    detection_windows: List[Dict],
    spec_path:   Optional[Path],
) -> Dict:
    """Write .wav + _meta.json for one accepted session clip."""
    out_dir.mkdir(parents=True, exist_ok=True)

    stem    = wav_path.stem
    tag     = f"{stem}_{int(start_s*1000):07d}_{int(end_s*1000):07d}"
    wav_out = out_dir / f"{tag}.wav"
    meta_out= out_dir / f"{tag}_meta.json"

    # load at save_sr
    try:
        (ch0,), _ = load_channels(wav_path, [0], save_sr,
                                  start_s=start_s, end_s=end_s)
    except Exception as e:
        print(f"  ⚠  Save load error: {e}")
        return {}

    # normalize
    peak  = float(np.max(np.abs(ch0)) + 1e-8)
    norm  = 0.98 / peak
    ch0_n = np.clip(ch0 * norm, -1.0, 1.0).astype(np.float32)
    sf.write(str(wav_out), ch0_n, save_sr)

    sig = compute_signal_metrics(ch0, save_sr)

    # serialize drones (ndarray → list)
    drones_serial = []
    for d in drones:
        dd = {k: (v.tolist() if hasattr(v, "tolist") else v)
              for k, v in d.items()}
        drones_serial.append(dd)

    meta = {
        "clip": {
            "source_file":   str(wav_path),
            "output_wav":    str(wav_out),
            "start_s":       round(start_s, 4),
            "end_s":         round(end_s,   4),
            "duration_s":    round(end_s - start_s, 4),
            "sample_rate":   save_sr,
            "flight_phase":  flight_phase,
            "normalization_gain_db": round(20 * math.log10(norm), 3),
        },
        "detection": {
            "peak_prob":   seg["peak_prob"],
            "mean_prob":   seg["mean_prob"],
            "n_windows":   seg["n_windows"],
            "threshold":   _cfg.DETECTION_THRESHOLD if _cfg else 0.945,
            "windows":     [
                {"start_s": w["start_s"], "end_s": w["end_s"],
                 "probability": w["probability"], "detected": w["detected"]}
                for w in detection_windows
                if w["start_s"] >= start_s - 1 and w["end_s"] <= end_s + 1
            ],
        },
        "localization": {
            "n_drones": len(drones_serial),
            "drones":   drones_serial,
        },
        "signal_metrics": sig,
        "ground_truth": {
            "gpx_n_points":  len(gpx_data),
            "gpx_trackpoints": gpx_data,
            "rpm":           rpm_data,
        },
        "spectrogram_png": str(spec_path) if spec_path else None,
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    row = {
        "source_file":  str(wav_path),
        "output_wav":   str(wav_out),
        "output_meta":  str(meta_out),
        "start_s":      round(start_s, 4),
        "end_s":        round(end_s,   4),
        "duration_s":   round(end_s - start_s, 4),
        "flight_phase": flight_phase,
        "peak_prob":    seg["peak_prob"],
        "n_drones":     len(drones_serial),
        "gpx_points":   len(gpx_data),
        **{f"sig_{k}": v for k, v in sig.items()},
    }
    print(f"  💾 Saved: {wav_out.name}")
    print(f"      meta: {meta_out.name}")
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Playback
# ══════════════════════════════════════════════════════════════════════════════

def try_play(y: np.ndarray, sr: int):
    if len(y) == 0:
        print("  (nothing to play)"); return
    if SOUNDDEVICE_OK:
        try:
            sd.stop(); sd.play(y.astype(np.float32), sr, blocking=True); return
        except Exception as e:
            print(f"  sounddevice error: {e}")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(str(tmp), y.astype(np.float32), sr)
        for cmd in (
            ["ffplay", "-nodisp", "-autoexit", str(tmp)],
            ["mpv", "--no-video", str(tmp)],
            ["aplay", str(tmp)],
        ):
            try:
                subprocess.run(cmd, check=False,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue
        print("  (no audio player found — install ffplay or sounddevice)")
    finally:
        tmp.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Range parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_ranges(text: str, total_sec: float) -> List[Tuple[float, float]]:
    out = []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        if ":" not in part:
            raise ValueError(f"Use start:end format — got '{part}'")
        a, b = part.split(":", 1)
        s = max(0.0, float(a.strip()))
        e = min(total_sec, float(b.strip()))
        if e <= s:
            raise ValueError(f"end must be > start: '{part}'")
        out.append((s, e))
    return out


def print_help():
    print("""
  Commands:
    Enter / y        accept segment as-is
    A:B              trim to A–B seconds (relative to segment start)
    A:B, C:D         save multiple sub-clips
    p                play segment (ch 0)
    p A B            play sub-region A..B (relative to segment start)
    l                run localization on this segment
    s                show/re-save spectrogram
    n                skip this segment
    q                quit and write manifest
    h                this help
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main interactive loop
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: WAV not found: {wav_path}"); sys.exit(1)

    out_dir    = Path(args.output)
    sessions_d = out_dir / "sessions"
    specs_d    = out_dir / "spectrograms"
    manifest_d = out_dir / "manifests"
    det_d      = out_dir / "detection"
    for d in (sessions_d, specs_d, manifest_d, det_d):
        d.mkdir(parents=True, exist_ok=True)

    total_sec = get_duration(wav_path)
    native_sr = get_native_sr(wav_path)

    print("=" * 70)
    print(f"  MULTICHANNEL SESSION PICKER")
    print(f"  File   : {wav_path.name}")
    print(f"  Length : {total_sec:.1f}s  ({total_sec/60:.1f} min)")
    print(f"  SR     : {native_sr} Hz  |  channels: auto-detect from header")
    print("=" * 70)

    # ── load pipeline ─────────────────────────────────────────────────────────
    print("\n  Loading detection pipeline…")
    models_dir = Path(args.models) if args.models else Path(__file__).parent / "models"
    try:
        _load_pipeline(models_dir)
        print(f"  ✅ Pipeline ready  (threshold={_cfg.DETECTION_THRESHOLD:.3f})")
    except Exception as e:
        print(f"  ✖  Pipeline load failed: {e}")
        sys.exit(1)

    # ── stand separation ──────────────────────────────────────────────────────
    stand_sep = args.stand_separation_m
    if stand_sep is not None:
        print(f"  Stand separation: {stand_sep:.3f} m → cross-stand strategies enabled")
    else:
        print("  Stand separation: unknown → single-stand strategy only")

    # ── aux data (GPX, RPM) ───────────────────────────────────────────────────
    gpx_df = None
    if args.gpx and Path(args.gpx).is_file() and PANDAS_OK:
        try:
            gpx_df = pd.read_csv(args.gpx, parse_dates=["time"])
            gpx_df["time"] = pd.to_datetime(gpx_df["time"], utc=True, errors="coerce")
            print(f"  GPX: {len(gpx_df)} trackpoints loaded")
        except Exception as e:
            print(f"  ⚠  GPX load failed: {e}")

    rpm_df = None
    if args.rpm and Path(args.rpm).is_file() and PANDAS_OK:
        try:
            rpm_df = pd.read_csv(args.rpm)
            print(f"  RPM: {len(rpm_df)} rows loaded")
        except Exception as e:
            print(f"  ⚠  RPM load failed: {e}")

    takeoff_utc: Optional[datetime] = None
    landing_utc: Optional[datetime] = None
    f_start_utc = file_start_utc(wav_path)

    def _parse_dt(s: str) -> datetime:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ValueError(f"Cannot parse datetime: {s!r}")

    if args.flight_takeoff:
        takeoff_utc = _parse_dt(args.flight_takeoff)
    if args.flight_landing:
        landing_utc = _parse_dt(args.flight_landing)

    # ── display channels ──────────────────────────────────────────────────────
    n_file_ch = sf.info(str(wav_path)).channels
    dpa_chs   = [i for i in range(min(10, n_file_ch))]
    mems_chs  = [i for i in [10, 11] if i < n_file_ch]
    display_chs = dpa_chs[:5] + mems_chs   # show stand-A + MEMS in spectrograms
    detect_ch = 2   # middle of stand A triplet

    print(f"\n  DPA channels  : {dpa_chs}")
    print(f"  MEMS channels : {mems_chs}")
    print(f"  Detection ch  : {detect_ch}")
    print(f"  Spectrogram   : channels {display_chs}")

    # ── sliding-window detection ───────────────────────────────────────────────
    print(f"\n  Running sliding-window detection…")
    windows = sliding_window_detect(
        wav_path, detect_ch,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        total_sec=total_sec,
    )

    # save raw detection log
    det_log = det_d / f"{wav_path.stem}_windows.json"
    det_log.write_text(json.dumps([
        {k: v for k, v in w.items() if k not in ("heuristic_features",)}
        for w in windows
    ], indent=2))
    print(f"  Detection log: {det_log.name}")

    # merge into candidate segments
    segments = merge_detected_windows(
        windows,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
    )
    print(f"\n  {'━'*60}")
    print(f"  Found {len(segments)} candidate segment(s)")
    print(f"  {'━'*60}\n")

    if not segments:
        print("  No drone activity detected in this file.")
        print("  Tip: lower --window-sec or adjust detection threshold in config.py")
        return

    # ── interactive segment review ─────────────────────────────────────────────
    print_help()
    all_rows: List[Dict] = []

    for seg_idx, seg in enumerate(segments, 1):
        dur = seg["end_s"] - seg["start_s"]

        # ground truth for this segment
        gpx_data:   List[Dict] = []
        rpm_data:   Dict       = {}
        flight_phase           = "unknown"

        if f_start_utc is not None:
            t0_utc = f_start_utc + timedelta(seconds=seg["start_s"])
            t1_utc = f_start_utc + timedelta(seconds=seg["end_s"])
            if gpx_df is not None:
                gpx_data = extract_gpx_window(gpx_df, t0_utc, t1_utc)
            if rpm_df is not None and takeoff_utc is not None:
                rpm_data = extract_rpm_window(rpm_df, t0_utc, t1_utc, takeoff_utc)
            flight_phase = infer_flight_phase(t0_utc, t1_utc, takeoff_utc, landing_utc)

        print(f"\n  ┌─ Segment {seg_idx}/{len(segments)} {'─'*48}")
        print(f"  │  Range  : {seg['start_s']:.2f}s → {seg['end_s']:.2f}s  ({dur:.1f}s)")
        print(f"  │  Prob   : peak={seg['peak_prob']:.3f}  mean={seg['mean_prob']:.3f}  "
              f"windows={seg['n_windows']}")
        print(f"  │  Phase  : {flight_phase}")
        print(f"  │  GPX pts: {len(gpx_data)}")
        if rpm_data:
            bpf = rpm_data.get("bpf_hz_mean", "?")
            rpm = rpm_data.get("rpm_mean", "?")
            print(f"  │  RPM    : {rpm}  BPF: {bpf} Hz")
        print(f"  └{'─'*54}")

        # auto-generate spectrogram
        spec_path = specs_d / f"{wav_path.stem}_seg{seg_idx:03d}.png"
        drones: List[Dict] = []

        plot_segment_spectrograms(
            wav_path, seg, display_chs, spec_path,
            title=f"{wav_path.stem} — seg {seg_idx}",
            drones=drones,
        )

        while True:
            cmd = input(
                f"\n  ▶  [Enter]=accept  A:B=trim  p=play  l=localize  "
                f"s=spectrogram  n=skip  q=quit  h=help : "
            ).strip()

            if cmd.lower() == "h":
                print_help(); continue

            if cmd.lower() in ("n", "skip"):
                print("  → Skipped."); break

            if cmd.lower() == "q":
                print("  Quit requested.")
                _write_manifest(all_rows, manifest_d)
                return

            # ── localize ──────────────────────────────────────────────────────
            if cmd.lower() == "l":
                print("  Running localization…")
                drones = localize_segment(
                    wav_path, seg, args.strategy,
                    args.max_drones, stand_sep,
                )
                if drones:
                    for i, d in enumerate(drones, 1):
                        print(f"  Drone {i}: az={d['azimuth_deg']:.1f}°  "
                              f"dist={d['distance_m']:.1f}m  "
                              f"conf_r={d.get('confidence_radius', float('nan')):.2f}m")
                    # re-save spectrogram with drone annotations
                    plot_segment_spectrograms(
                        wav_path, seg, display_chs, spec_path,
                        title=f"{wav_path.stem} — seg {seg_idx} — {len(drones)} drone(s)",
                        drones=drones,
                    )
                else:
                    print("  No drones localized.")
                continue

            # ── spectrogram ───────────────────────────────────────────────────
            if cmd.lower() == "s":
                plot_segment_spectrograms(
                    wav_path, seg, display_chs, spec_path,
                    title=f"{wav_path.stem} — seg {seg_idx}",
                    drones=drones,
                )
                continue

            # ── playback ──────────────────────────────────────────────────────
            if cmd.lower().startswith("p"):
                parts = cmd.split()
                sr    = _cfg.SR
                try:
                    (ch,), _ = load_channels(wav_path, [detect_ch], sr,
                                             start_s=seg["start_s"],
                                             end_s=seg["end_s"])
                except Exception as e:
                    print(f"  ⚠  Load failed: {e}"); continue

                if len(parts) == 1:
                    print(f"  ♪ Playing {dur:.1f}s…")
                    try_play(ch, sr)
                elif len(parts) == 3:
                    try:
                        ps = max(0.0, float(parts[1]))
                        pe = min(dur, float(parts[2]))
                        try_play(ch[int(ps*sr):int(pe*sr)], sr)
                    except ValueError:
                        print("  Use: p A B  (seconds relative to segment start)")
                del ch; gc.collect()
                continue

            # ── accept as-is ──────────────────────────────────────────────────
            if cmd in ("", "y"):
                row = save_session(
                    wav_path, seg,
                    start_s=seg["start_s"], end_s=seg["end_s"],
                    out_dir=sessions_d,
                    save_sr=args.save_sr,
                    drones=drones,
                    gpx_data=gpx_data,
                    rpm_data=rpm_data,
                    flight_phase=flight_phase,
                    detection_windows=windows,
                    spec_path=spec_path,
                )
                if row:
                    all_rows.append(row)
                break

            # ── trim ranges ───────────────────────────────────────────────────
            try:
                ranges = parse_ranges(cmd, dur)
            except Exception as e:
                print(f"  ⚠  {e}"); continue

            saved = 0
            for sub_s, sub_e in ranges:
                abs_s = seg["start_s"] + sub_s
                abs_e = seg["start_s"] + sub_e
                sub_seg = {**seg, "start_s": abs_s, "end_s": abs_e}
                row = save_session(
                    wav_path, sub_seg,
                    start_s=abs_s, end_s=abs_e,
                    out_dir=sessions_d,
                    save_sr=args.save_sr,
                    drones=drones,
                    gpx_data=gpx_data,
                    rpm_data=rpm_data,
                    flight_phase=flight_phase,
                    detection_windows=windows,
                    spec_path=spec_path,
                )
                if row:
                    all_rows.append(row); saved += 1
            if saved:
                break

    _write_manifest(all_rows, manifest_d)
    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  Sessions saved : {len(all_rows)}")
    print(f"  Output         : {out_dir}")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Manifest writer
# ══════════════════════════════════════════════════════════════════════════════

def _write_manifest(rows: List[Dict], manifest_d: Path):
    if not rows:
        print("  (no sessions saved — no manifest written)")
        return
    mj = manifest_d / "session_manifest.json"
    mc = manifest_d / "session_manifest.csv"
    mj.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    keys = sorted({k for r in rows for k in r})
    with open(mc, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n  📄 Manifest JSON : {mj}")
    print(f"  📄 Manifest CSV  : {mc}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Multichannel drone session picker — CNN detection + "
                    "spectrogram + ground truth extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--wav",      required=True,
                    help="Path to multichannel WAV")
    ap.add_argument("--output",   default="sessions_out",
                    help="Output directory (default: sessions_out/)")
    ap.add_argument("--models",   default=None,
                    help="Path to models/ dir (default: ./models/)")
    ap.add_argument("--gpx",      default=None,
                    help="GPX trackpoints CSV (columns: time, lat, lon, elevation)")
    ap.add_argument("--rpm",      default=None,
                    help="Rotor RPM CSV (columns: time_s, rpm, bpf_hz …)")
    ap.add_argument("--flight-takeoff", default=None,
                    help="Takeoff UTC e.g. 2025-10-20T12:57:21")
    ap.add_argument("--flight-landing", default=None,
                    help="Landing UTC e.g. 2025-10-20T13:04:09")
    ap.add_argument("--strategy", default=DEFAULT_DPA_STRATEGY,
                    choices=["stand_a","stand_b","max_baseline",
                             "equilateral","cross_stand","best_snr"],
                    help=f"DPA triplet strategy (default: {DEFAULT_DPA_STRATEGY})")
    ap.add_argument("--stand-separation-m", type=float, default=None,
                    help="Stand A–B distance in metres (enables cross-stand)")
    ap.add_argument("--window-sec",  type=float, default=DEFAULT_WINDOW_SEC,
                    help=f"Detection window length (default: {DEFAULT_WINDOW_SEC}s)")
    ap.add_argument("--stride-sec",  type=float, default=DEFAULT_STRIDE_SEC,
                    help=f"Detection stride (default: {DEFAULT_STRIDE_SEC}s)")
    ap.add_argument("--merge-gap-sec",    type=float, default=2.0,
                    help="Merge windows within this gap (default: 2.0s)")
    ap.add_argument("--min-duration-sec", type=float, default=2.0,
                    help="Minimum segment duration (default: 2.0s)")
    ap.add_argument("--max-drones", type=int, default=DEFAULT_MAX_DRONES,
                    help=f"Max drones for localization (default: {DEFAULT_MAX_DRONES})")
    ap.add_argument("--save-sr",    type=int, default=DEFAULT_SAVE_SR,
                    help=f"Sample rate for saved clips (default: {DEFAULT_SAVE_SR})")
    run(ap.parse_args())


if __name__ == "__main__":
    main()