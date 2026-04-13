# -*- coding: utf-8 -*-
"""
drone_detection/utils.py
────────────────────────
# Pure math/signal helpers (sigmoid, gcc_phat, etc.)
Pure helper functions: math, signal processing, file utilities.
No ML framework imports — keeps import time low when only utilities
are needed.
"""

from __future__ import annotations

import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.signal


# ── Numeric / angular helpers ─────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def wrap_angle_deg(a: float) -> float:
    """Wrap angle to (−180, +180]."""
    return float((a + 180.0) % 360.0 - 180.0)


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Element-wise shortest angular distance in degrees."""
    diff = np.asarray(pred) - np.asarray(target)
    return np.abs((diff + 180.0) % 360.0 - 180.0)


def safe_slug(name: str) -> str:
    """Convert *name* to a filesystem-safe ASCII slug."""
    return (
        "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        .strip("_") or "out"
    )


def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, np.float32) ** 2)) + 1e-8)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, np.float32)
    m = float(np.max(np.abs(y)) + 1e-8)
    return np.clip(y * (peak / m), -1.0, 1.0).astype(np.float32)


def classify_detection_score(score: float, cfg) -> str:
    if score >= cfg.DETECTION_THRESHOLD:      return "drone"
    if score >= cfg.DETECTION_THRESHOLD_LOW:  return "possible_drone"
    if score >= cfg.DETECTION_THRESHOLD_WEAK: return "weak_possible_drone"
    return "non_drone"


def safe_prob_average(values, default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def xy_to_azimuth_deg(xy: np.ndarray, center: np.ndarray) -> float:
    return wrap_angle_deg(
        np.degrees(np.arctan2(float(xy[1] - center[1]), float(xy[0] - center[0])))
    )


def azimuth_deg_to_xy(az_deg: float, dist_m: float, center: np.ndarray) -> np.ndarray:
    r = np.radians(az_deg)
    return np.array(
        [center[0] + dist_m * np.cos(r), center[1] + dist_m * np.sin(r)],
        dtype=np.float32,
    )


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x.astype(np.float32)
    k   = max(1, int(k))
    pad = k // 2
    xp  = np.pad(x, (pad, pad), mode="edge")
    w   = np.ones(k) / k
    return np.convolve(xp, w, mode="valid")[: len(x)].astype(np.float32)


def robust_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float32)
    return np.gradient(x.astype(np.float64), dt).astype(np.float32)


def random_crop_or_loop(y: np.ndarray, target_n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return np.zeros(target_n, dtype=np.float32)
    if len(y) == target_n:
        return y
    if len(y) > target_n:
        start = random.randint(0, len(y) - target_n)
        return y[start : start + target_n]
    reps = int(np.ceil(target_n / max(len(y), 1)))
    return np.tile(y, reps)[:target_n]


def mix_at_snr(drone_y: np.ndarray, bg_y: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix drone + background at a target SNR (dB)."""
    drone_y = np.asarray(drone_y, np.float32)
    bg_y    = np.asarray(bg_y, np.float32)
    d_rms   = rms_energy(drone_y)
    b_rms   = rms_energy(bg_y)
    if b_rms < 1e-8:
        return drone_y
    scale = (d_rms / (10 ** (snr_db / 20.0))) / b_rms
    return normalize_peak(drone_y + bg_y * scale)


# ── Spectral / signal helpers ─────────────────────────────────────────────────

def safe_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)


def compute_delta_2d(x: np.ndarray) -> np.ndarray:
    """First-order finite difference along the time axis (axis=1)."""
    x = np.asarray(x, np.float32)
    return np.diff(x, axis=1, prepend=x[:, :1]).astype(np.float32)


# ── GCC-PHAT TDOA estimation ──────────────────────────────────────────────────

def _parabolic_peak(y: np.ndarray, x: int) -> float:
    """Sub-sample peak refinement via parabolic interpolation."""
    if x <= 0 or x >= len(y) - 1:
        return float(x)
    y1, y2, y3 = y[x - 1], y[x], y[x + 1]
    d = y1 - 2 * y2 + y3
    if abs(d) < 1e-12:
        return float(x)
    return x + 0.5 * (y1 - y3) / d


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int,
    max_tau: float,
    interp: int = 4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Generalised Cross-Correlation with Phase Transform.

    Returns
    -------
    tau : estimated delay (seconds)
    lags : lag axis (seconds)
    cc  : cross-correlation magnitude
    """
    n   = len(sig) + len(ref)
    S   = np.fft.rfft(sig, n=n)
    R   = np.fft.rfft(ref, n=n)
    if np.max(np.abs(S)) < 1e-10 or np.max(np.abs(R)) < 1e-10:
        nlags = 2 * int(interp * fs * max_tau) + 1
        return 0.0, np.zeros(nlags), np.zeros(nlags)
    X        = S * np.conj(R)
    den      = np.abs(X); den[den < 1e-10] = 1e-10; X /= den
    cc       = np.fft.irfft(X, n=interp * n)
    ms       = min(int(interp * n / 2), int(interp * fs * max_tau))
    cc       = np.concatenate((cc[-ms:], cc[:ms + 1]))
    pk       = int(np.argmax(np.abs(cc)))
    pk_f     = _parabolic_peak(np.abs(cc), pk)
    tau      = (pk_f - ms) / (interp * fs)
    lags     = np.arange(-ms, ms + 1) / (interp * fs)
    return tau, lags, np.abs(cc)


def gcc_phat_peaks(
    x1: np.ndarray,
    x2: np.ndarray,
    sr: int,
    max_tau: float,
    n_peaks: int,
) -> list[tuple[float, float]]:
    """Return the *n_peaks* strongest GCC-PHAT peaks as (tau, strength) pairs."""
    n    = len(x1) + len(x2) - 1
    X1   = np.fft.rfft(x1, n=n)
    X2   = np.fft.rfft(x2, n=n)
    cc   = X1 * np.conj(X2); cc /= (np.abs(cc) + 1e-8)
    r    = np.fft.irfft(cc, n=n)
    ml   = int(max_tau * sr)
    rt   = np.concatenate([r[-ml:], r[:ml + 1]])
    lags = np.arange(-ml, ml + 1) / sr
    peaks: list[tuple[float, float]] = []
    rw   = rt.copy()
    for _ in range(n_peaks):
        idx = int(np.argmax(rw))
        peaks.append((float(lags[idx]), float(rw[idx])))
        lo = max(0, idx - 3); hi = min(len(rw), idx + 4)
        rw[lo:hi] = 0.0
    return peaks


def _fractional_delay(signal_: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a fractional-sample delay via a Lagrange interpolation filter."""
    if delay_samples < 0:
        delay_samples = 0.0
    int_d = int(np.floor(delay_samples))
    frac  = delay_samples - int_d
    taps  = np.arange(-3, 5)
    h     = np.sinc(taps - frac) * np.hanning(len(taps))
    h    /= h.sum() + 1e-12
    filt  = np.convolve(signal_.astype(np.float64), h, mode="full")[: len(signal_)]
    if int_d > 0:
        return np.concatenate([np.zeros(int_d), filt])[: len(signal_)].astype(np.float32)
    return filt.astype(np.float32)


def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = scipy.signal.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return scipy.signal.sosfilt(sos, y).astype(np.float32)


# ── RNG seeding ───────────────────────────────────────────────────────────────

def set_seed(s: int) -> None:
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Dependency helpers ────────────────────────────────────────────────────────

def ensure_remotezip() -> None:
    try:
        import remotezip  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "remotezip"]
        )


# ── Group-split helpers (used by both datasets.py and custom builder) ─────────

def infer_group_id(path: Path, strategy: str = "source_file") -> str:
    stem = path.stem.lower()
    patterns = [
        r"(session[_\-]?\d+)", r"(flight[_\-]?\d+)", r"(take[_\-]?\d+)",
        r"(clip[_\-]?\d+)",    r"(yt[_\-]?[a-z0-9]+)", r"(fs[_\-]?\d+)",
        r"(bbc[_\-]?[a-z0-9]+)", r"(xc[_\-]?\d+)",    r"(sb[_\-]?\d+)",
    ]
    for p in patterns:
        m = re.search(p, stem)
        if m:
            return m.group(1)
    parts = re.split(r"[_\-]", stem)
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def grouped_split_paths(
    files: list[Path], seed: int = 42
) -> Dict[str, list[Path]]:
    """Assign whole *source groups* to train/val/test to prevent leakage."""
    rng = random.Random(seed)
    group_to_files: Dict[str, list[Path]] = {}
    for f in files:
        gid = infer_group_id(f)
        group_to_files.setdefault(gid, []).append(f)
    groups = list(group_to_files.items())
    rng.shuffle(groups)
    n     = len(groups)
    n_tr  = int(0.70 * n)
    n_val = int(0.15 * n)
    split_map: Dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for i, (_, flist) in enumerate(groups):
        if i < n_tr:
            split_map["train"].extend(flist)
        elif i < n_tr + n_val:
            split_map["val"].extend(flist)
        else:
            split_map["test"].extend(flist)
    return split_map


# ── Waveform augmentation (used by DetectionDataset & LocalizationDataset) ────

def random_eq_tilt(y: np.ndarray, sr: int, strength: float = 0.35) -> np.ndarray:
    y    = np.asarray(y, np.float32)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr); freqs[0] = 1.0
    tilt  = random.uniform(-strength, strength)
    mag   = np.power(np.maximum(freqs / 1000.0, 1e-3), tilt)
    out   = np.fft.irfft(spec * mag, n=len(y))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def random_bandlimit(y: np.ndarray, sr: int) -> np.ndarray:
    y  = np.asarray(y, np.float32)
    lo = random.uniform(20, 250)
    hi = random.uniform(2500, min(8000, sr / 2 - 200))
    if hi <= lo + 300:
        return y
    sos = scipy.signal.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return np.clip(scipy.signal.sosfilt(sos, y), -1.0, 1.0).astype(np.float32)


def random_reverb(y: np.ndarray, sr: int) -> np.ndarray:
    y      = np.asarray(y, np.float32)
    ir_len = int(random.uniform(0.03, 0.20) * sr)
    if ir_len < 8:
        return y
    t     = np.linspace(0, 1, ir_len, endpoint=False)
    decay = np.exp(-t * random.uniform(8, 25))
    ir    = decay * np.random.randn(ir_len).astype(np.float32)
    ir[0] += 1.0; ir /= np.max(np.abs(ir)) + 1e-8
    out = scipy.signal.fftconvolve(y, ir, mode="full")[: len(y)]
    return normalize_peak(out).astype(np.float32)


def random_codec_like(y: np.ndarray) -> np.ndarray:
    step = random.choice([64, 128, 256, 512])
    return np.clip(np.round(np.asarray(y, np.float32) * step) / step, -1.0, 1.0).astype(np.float32)


def random_dropout_chunks(y: np.ndarray, max_chunks: int = 3) -> np.ndarray:
    y = np.asarray(y, np.float32).copy()
    n = len(y)
    for _ in range(random.randint(0, max_chunks)):
        w = random.randint(max(8, n // 100), max(16, n // 20))
        s = random.randint(0, max(0, n - w))
        y[s : s + w] *= random.uniform(0.0, 0.25)
    return y.astype(np.float32)


def augment_waveform(y: np.ndarray, cfg) -> np.ndarray:
    """Full waveform augmentation chain used by DetectionDataset."""
    from drone_detection.audio import AudioProcessor  # local import to avoid circulars

    y = np.asarray(y, np.float32)
    if random.random() < 0.90:
        y = np.clip(y * db_to_gain(random.uniform(-10, 10)), -1, 1).astype(np.float32)
    if random.random() < 0.60:
        y = AudioProcessor(cfg).add_noise(y, random.uniform(0, 25))
    if random.random() < 0.35:
        y = random_eq_tilt(y, cfg.SR, 0.45)
    if random.random() < 0.35:
        y = random_bandlimit(y, cfg.SR)
    if random.random() < 0.25:
        y = random_reverb(y, cfg.SR)
    if random.random() < 0.25:
        y = random_codec_like(y)
    if random.random() < 0.30:
        y = random_dropout_chunks(y)
    if random.random() < 0.50:
        y = np.roll(y, random.randint(0, len(y) // 4)).astype(np.float32)
    if random.random() < 0.20:
        clip_val = random.uniform(0.35, 0.9)
        y = np.clip(y, -clip_val, clip_val) / clip_val
    return np.clip(y, -1, 1).astype(np.float32)