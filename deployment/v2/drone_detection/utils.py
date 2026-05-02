# -*- coding: utf-8 -*-
"""
utils.py
────────
Pure utility functions shared across the pipeline.

Contents
────────
- Math / scalar helpers
- Audio loading helpers (librosa + pydub fallback)
- Signal processing: GCC-PHAT, bandpass, fractional delay
- Waveform augmentation (EQ tilt, bandlimit, reverb, codec simulation)
- Grouped-split helpers for leak-free train/val/test splits
- File I/O helpers (safe_slug, rms_energy, normalize_peak …)
"""

import math
import os
import random
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

try:
    from pydub import AudioSegment
    _PYDUB_OK = True
except ImportError:
    _PYDUB_OK = False

try:
    import yt_dlp as ytdlp
    _YTDLP_OK = True
except ImportError:
    _YTDLP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Scalar helpers
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def wrap_angle_deg(a: float) -> float:
    """Wrap an angle into (-180, 180]."""
    return float((a + 180.0) % 360.0 - 180.0)


def safe_slug(name: str) -> str:
    """Convert a string to a filesystem-safe identifier."""
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
    return (y * (peak / m)).astype(np.float32) if m > 0 else y


def angular_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Wrapped absolute angular error in degrees."""
    diff = (np.asarray(pred) - np.asarray(true))
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def xy_to_azimuth_deg(xy: np.ndarray, center: np.ndarray) -> float:
    return wrap_angle_deg(
        np.degrees(np.arctan2(float(xy[1] - center[1]), float(xy[0] - center[0])))
    )


def azimuth_deg_to_xy(az: float, dist: float, center: np.ndarray) -> np.ndarray:
    r = np.radians(az)
    return np.array(
        [center[0] + dist * np.cos(r), center[1] + dist * np.sin(r)],
        dtype=np.float32,
    )


def classify_detection_score(score: float, cfg) -> str:
    """Map a fused probability to a human-readable label."""
    if score >= cfg.DETECTION_THRESHOLD:
        return "drone"
    if score >= cfg.DETECTION_THRESHOLD_LOW:
        return "possible_drone"
    if score >= cfg.DETECTION_THRESHOLD_WEAK:
        return "weak_possible_drone"
    return "non_drone"


def safe_prob_average(values, default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x.astype(np.float32)
    k = max(1, int(k))
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")[: len(x)].astype(np.float32)


def robust_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float32)
    return np.gradient(x.astype(np.float64), dt).astype(np.float32)


def random_crop_or_loop(y: np.ndarray, target_n: int) -> np.ndarray:
    """Return a random crop if y is long enough, otherwise tile to fill target_n."""
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


def mix_at_snr(
    drone_y: np.ndarray, bg_y: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix drone + background at a target SNR (dB)."""
    drone_y = np.asarray(drone_y, dtype=np.float32)
    bg_y    = np.asarray(bg_y,    dtype=np.float32)
    d_rms   = rms_energy(drone_y)
    b_rms   = rms_energy(bg_y)
    if b_rms < 1e-8:
        return drone_y
    scale = (d_rms / (10 ** (snr_db / 20.0))) / b_rms
    return normalize_peak(drone_y + bg_y * scale)


def safe_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + eps)).astype(np.float32)


def compute_delta_2d(x: np.ndarray) -> np.ndarray:
    """First-order temporal derivative of a 2-D feature matrix."""
    x = np.asarray(x, dtype=np.float32)
    return np.diff(x, axis=1, prepend=x[:, :1]).astype(np.float32)


def _set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# Audio I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_any(path: Path, sr: int) -> np.ndarray:
    """
    Load any audio file to a float32 mono array resampled to sr.
    Tries librosa first, falls back to pydub.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y.astype(np.float32)
        except Exception:
            pass
    if _PYDUB_OK:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            AudioSegment.from_file(str(path)).export(str(tmp), format="wav")
            y, _ = librosa.load(str(tmp), sr=sr, mono=True)
            return y.astype(np.float32)
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Cannot load {path}")


def convert_to_wav(src: Path, dst: Path):
    """Convert any audio file to WAV at the original sample rate."""
    if src.suffix.lower() == ".wav":
        shutil.copy2(str(src), str(dst))
    elif _PYDUB_OK:
        AudioSegment.from_file(str(src)).export(str(dst), format="wav")
    else:
        raise RuntimeError(f"pydub not available for {src.suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# Signal processing
# ══════════════════════════════════════════════════════════════════════════════

def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = scipy.signal.butter(
        4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos"
    )
    return scipy.signal.sosfilt(sos, y).astype(np.float32)


def _parabolic_peak(y: np.ndarray, x: int) -> float:
    """Sub-sample peak refinement via parabolic interpolation."""
    if x <= 0 or x >= len(y) - 1:
        return float(x)
    y1, y2, y3 = y[x - 1], y[x], y[x + 1]
    d = y1 - 2 * y2 + y3
    if abs(d) < 1e-12:
        return float(x)
    return x + 0.5 * (y1 - y3) / d


def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: float, interp: int = 4):
    n   = len(sig) + len(ref)
    S   = np.fft.rfft(sig, n=n); R = np.fft.rfft(ref, n=n)
    if np.max(np.abs(S)) < 1e-10 or np.max(np.abs(R)) < 1e-10:
        nlags = 2*int(interp*fs*max_tau)+1
        return 0.0, np.zeros(nlags), np.zeros(nlags)
    X   = S * np.conj(R); den = np.abs(X); den[den < 1e-10] = 1e-10; X /= den
    cc  = np.fft.irfft(X, n=interp*n)
    ms  = min(int(interp*n/2), int(interp*fs*max_tau))
    cc  = np.concatenate((cc[-ms:], cc[:ms+1]))
    pk  = int(np.argmax(np.abs(cc))); pk_f = _parabolic_peak(np.abs(cc), pk)
    tau = (pk_f - ms) / (interp * fs)
    lags = np.arange(-ms, ms+1) / (interp*fs)
    return tau, lags, np.abs(cc)


def gcc_phat_peaks(
    x1: np.ndarray,
    x2: np.ndarray,
    sr: int,
    max_tau: float,
    n_peaks: int,
) -> List[Tuple[float, float]]:
    """
    Return the top-n GCC-PHAT peaks as (tau_s, amplitude) tuples.
    Used by multi-drone localization to find multiple TDOAs.
    """
    n    = len(x1) + len(x2) - 1
    X1   = np.fft.rfft(x1, n=n)
    X2   = np.fft.rfft(x2, n=n)
    cc   = X1 * np.conj(X2)
    cc  /= np.abs(cc) + 1e-8
    r    = np.fft.irfft(cc, n=n)
    ml   = int(max_tau * sr)
    rt   = np.concatenate([r[-ml:], r[:ml + 1]])
    lags = np.arange(-ml, ml + 1) / sr
    peaks: List[Tuple[float, float]] = []
    rw = rt.copy()
    for _ in range(n_peaks):
        idx = int(np.argmax(rw))
        peaks.append((float(lags[idx]), float(rw[idx])))
        lo = max(0, idx - 3)
        hi = min(len(rw), idx + 4)
        rw[lo:hi] = 0.0
    return peaks


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    delay_samples = max(0.0, float(delay_samples))
    n     = len(signal)
    int_d = int(np.floor(delay_samples))
    frac  = delay_samples - int_d

    # Fractional part via windowed-sinc interpolation
    if frac > 1e-6:
        taps  = np.arange(-3, 5)
        h     = np.sinc(taps - frac) * np.hanning(len(taps))
        h    /= h.sum() + 1e-12
        sig_f = np.convolve(signal.astype(np.float64), h, mode='full')[:n]
    else:
        sig_f = signal.astype(np.float64).copy()

    # Integer part via zero-pad + truncate
    if int_d > 0:
        out          = np.empty(n, dtype=np.float32)
        out[:int_d]  = 0.0
        out[int_d:]  = sig_f[:n - int_d]
        return out

    return sig_f.astype(np.float32)


def compute_ipd_features(channels: List[np.ndarray], cfg) -> np.ndarray:
    """
    Compute 3 Inter-microphone Phase Delay (IPD) features via GCC-PHAT,
    bounded to the physically reachable lag window for this array geometry.
    Searching the full signal length (old behaviour) caused noise peaks far
    outside the physical range to dominate, making the IPD branch useless.
    """
    sr = cfg.SR
    c  = getattr(cfg, "SPEED_OF_SOUND", 343.0)
    # Maximum inter-mic distance → maximum physical lag in samples (+10% margin)
    max_dist = float(np.max([
        np.linalg.norm(np.array(cfg.MIC_POSITIONS[i]) - np.array(cfg.MIC_POSITIONS[j]))
        for i in range(len(cfg.MIC_POSITIONS))
        for j in range(i + 1, len(cfg.MIC_POSITIONS))
    ]))
    max_tau_samples = int(np.ceil(max_dist / c * sr * 1.1))

    pairs = [(0, 1), (0, 2), (1, 2)]
    ipds  = []
    for i, j in pairs:
        xi = channels[i].astype(np.float64)
        xj = channels[j].astype(np.float64)
        n  = max(len(xi), len(xj))
        Xi = np.fft.rfft(xi, n=n)
        Xj = np.fft.rfft(xj, n=n)
        cc = Xi * np.conj(Xj)
        cc /= np.abs(cc) + 1e-8
        tau = np.fft.irfft(cc, n=n)
        # Only search within ±max_tau_samples (positive lags at start, negative at end)
        search   = np.concatenate([tau[:max_tau_samples + 1], tau[-max_tau_samples:]])
        pk_local = int(np.argmax(np.abs(search)))
        pk       = pk_local if pk_local <= max_tau_samples else pk_local - len(search)
        ipds.append(float(pk) / sr)
    return np.array(ipds, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Waveform augmentation (v15)
# ══════════════════════════════════════════════════════════════════════════════

def random_eq_tilt(y: np.ndarray, sr: int, strength: float = 0.35) -> np.ndarray:
    """Random spectral tilt ±strength octaves/decade."""
    y    = np.asarray(y, np.float32)
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    freqs[0] = 1.0
    tilt = random.uniform(-strength, strength)
    mag  = np.power(np.maximum(freqs / 1000.0, 1e-3), tilt)
    out  = np.fft.irfft(spec * mag, n=len(y))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def random_bandlimit(y: np.ndarray, sr: int) -> np.ndarray:
    """Randomly restrict bandwidth to simulate varied recording conditions."""
    y  = np.asarray(y, np.float32)
    lo = random.uniform(20, 250)
    hi = random.uniform(2500, min(8000, sr / 2 - 200))
    if hi <= lo + 300:
        return y
    sos = scipy.signal.butter(
        4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos"
    )
    return np.clip(scipy.signal.sosfilt(sos, y), -1.0, 1.0).astype(np.float32)


def random_reverb(y: np.ndarray, sr: int) -> np.ndarray:
    """Convolve with a short synthetic room impulse response."""
    y      = np.asarray(y, np.float32)
    ir_len = int(random.uniform(0.03, 0.20) * sr)
    if ir_len < 8:
        return y
    t     = np.linspace(0, 1, ir_len, endpoint=False)
    decay = np.exp(-t * random.uniform(8, 25))
    noise = np.random.randn(ir_len).astype(np.float32)
    ir    = decay * noise
    ir[0] += 1.0
    ir   /= np.max(np.abs(ir)) + 1e-8
    out   = scipy.signal.fftconvolve(y, ir, mode="full")[: len(y)]
    return normalize_peak(out).astype(np.float32)


def random_codec_like(y: np.ndarray) -> np.ndarray:
    """Simulate quantization artefacts from lossy codec compression."""
    y    = np.asarray(y, np.float32)
    step = random.choice([64, 128, 256, 512])
    return np.clip(np.round(y * step) / step, -1.0, 1.0).astype(np.float32)


def random_dropout_chunks(y: np.ndarray, max_chunks: int = 3) -> np.ndarray:
    """Randomly attenuate short segments to simulate signal drop-outs."""
    y = np.asarray(y, np.float32).copy()
    n = len(y)
    for _ in range(random.randint(0, max_chunks)):
        w = random.randint(max(8, n // 100), max(16, n // 20))
        s = random.randint(0, max(0, n - w))
        y[s : s + w] *= random.uniform(0.0, 0.25)
    return y.astype(np.float32)


def augment_waveform(y: np.ndarray, cfg) -> np.ndarray:
    """
    Apply a random chain of waveform augmentations (v15).

    Augmentations (each applied probabilistically):
        gain jitter · additive noise · EQ tilt · bandlimit ·
        reverb · codec simulation · dropout chunks · time shift · clipping
    """
    y = np.asarray(y, np.float32)
    if random.random() < 0.90:
        y = np.clip(y * db_to_gain(random.uniform(-10, 10)), -1, 1).astype(np.float32)
    if random.random() < 0.60:
        # Inline add_noise to avoid circular import
        sr_val = rms_energy(y)
        noise  = np.random.randn(len(y)).astype(np.float32)
        noise /= (rms_energy(noise) + 1e-8)
        snr_db = random.uniform(0, 25)
        scale  = sr_val / (10 ** (snr_db / 20.0))
        y      = np.clip(y + noise * scale, -1.0, 1.0).astype(np.float32)
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


def perturb_multichannel(channels: List[np.ndarray], cfg) -> List[np.ndarray]:
    """
    Apply correlated + per-channel perturbations to a 3-mic recording.
    Used in LocalizationDataset augmentation.

    NOTE: independent per-channel fractional delay was intentionally removed.
    Applying a different random delay to each channel destroys the inter-channel
    time difference (ITD) that GCC-PHAT and the IPD branch rely on for azimuth.
    Gain jitter, additive noise, and EQ tilt are safe because they do not
    alter the relative timing between channels.
    """
    out       = []
    base_gain = random.uniform(-3.0, 3.0)
    for ch in channels:
        x  = np.asarray(ch, np.float32).copy()
        x *= db_to_gain(base_gain + random.uniform(-1.5, 1.5))
        if random.random() < 0.5:
            sr_val = rms_energy(x)
            n      = np.random.randn(len(x)).astype(np.float32)
            n     /= rms_energy(n) + 1e-8
            snr    = random.uniform(0, 20)
            x      = np.clip(x + n * sr_val / (10 ** (snr / 20.0)), -1, 1).astype(np.float32)
        if random.random() < 0.25:
            x = random_eq_tilt(x, cfg.SR, 0.25)
        out.append(np.clip(x, -1, 1).astype(np.float32))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Grouped-split helpers (v15 — reduces train/val leakage)
# ══════════════════════════════════════════════════════════════════════════════

def infer_group_id(path: Path) -> str:
    """
    Infer a logical group ID from a filename so that clips from the same
    recording session are kept together during the train/val/test split.
    """
    stem     = path.stem.lower()
    patterns = [
        r"(session[_\-]?\d+)", r"(flight[_\-]?\d+)", r"(take[_\-]?\d+)",
        r"(clip[_\-]?\d+)",    r"(yt[_\-]?[a-z0-9]+)", r"(fs[_\-]?\d+)",
        r"(bbc[_\-]?[a-z0-9]+)", r"(xc[_\-]?\d+)", r"(sb[_\-]?\d+)",
    ]
    for p in patterns:
        m = re.search(p, stem)
        if m:
            return m.group(1)
    parts = re.split(r"[_\-]", stem)
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def grouped_split_paths(
    files: List[Path], seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Assign files to train / val / test by group (session), not randomly,
    so related files never span across splits.
    """
    rng = random.Random(seed)
    group_to_files: Dict[str, List[Path]] = {}
    for f in files:
        gid = infer_group_id(f)
        group_to_files.setdefault(gid, []).append(f)

    groups = list(group_to_files.items())
    rng.shuffle(groups)
    n      = len(groups)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)

    split_map: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for i, (_, flist) in enumerate(groups):
        if i < n_tr:
            split_map["train"].extend(flist)
        elif i < n_tr + n_val:
            split_map["val"].extend(flist)
        else:
            split_map["test"].extend(flist)
    return split_map


# ══════════════════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_remotezip():
    """Install remotezip if not already available."""
    try:
        import remotezip  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "remotezip"]
        )


def show_and_save(fig, path=None, dpi: int = 150):
    """Save a matplotlib figure and display it inline (Jupyter / Colab)."""
    from IPython.display import display
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved: {path}")
    display(fig)