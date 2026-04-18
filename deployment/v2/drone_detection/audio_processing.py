# -*- coding: utf-8 -*-
"""
audio_processing.py
───────────────────
AudioProcessor — core class for loading, padding, and feature extraction.
Also contains synthesise_drone() for generating synthetic 3-mic recordings.

Includes:
  synthesise_drone()
    - Fundamentals now drawn from measured BPF profiles (per-drone bands from
      PannoniaFS Q2: Mavic Pro 209 Hz, Mavic 2 193 Hz, Mavic Mini 360 Hz)
      when a drone_type is passed.
    - Noise floor is injected from real measurement profiles:
        indoor  — flat ≈ −82.7 dB with resonance peaks at 627/1637/4363 Hz
        outdoor — Brownian (f⁻²) scaled to ≈ −78 dB median
      Selected via noise_profile argument.
    - Source signal generated ONCE, then delayed per mic (fractional delay),
      matching the physics patch.

  AudioProcessor
    - compute_bpf_energy_ratio() — new method returning the fraction of
      signal power in the BPF band(s), matching the Q4 feature definition.
      Used as the optional 4th IPD scalar.
    - feature_stack(log-mel, PCEN, delta-mel).
"""

import random
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

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

from .config import (
    Config, config,
    DRONE_BPF_PROFILES,
    NOISE_FLOOR_INDOOR,
    NOISE_FLOOR_OUTDOOR,
)
from .utils import (
    _fractional_delay,
    normalize_peak,
    safe_standardize,
    compute_delta_2d,
)


class AudioProcessor:
    """
    Handles all audio loading and feature-extraction operations.
    New method:
    compute_bpf_energy_ratio(y, bpf_hz, bw_hz) — scalar BPF energy ratio
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or config

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, path, mono: bool = True) -> np.ndarray:
        path = str(path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                y, _ = librosa.load(path, sr=self.cfg.SR, mono=mono)
                return y.astype(np.float32)
            except Exception:
                pass
        if _PYDUB_OK:
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            try:
                AudioSegment.from_file(path).export(str(tmp), format="wav")
                y, _ = librosa.load(str(tmp), sr=self.cfg.SR, mono=mono)
                return y.astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Cannot load {path}: {e}")
            finally:
                tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot load {path} (pydub not installed)")

    def load_channels(
        self, path, channel_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=self.cfg.SR, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        idx = channel_indices if channel_indices is not None else list(range(y.shape[0]))
        return [y[i].astype(np.float32) for i in idx if i < y.shape[0]]

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        n = int(self.cfg.SR * self.cfg.TARGET_DURATION)
        if len(y) >= n:
            return y[:n].astype(np.float32)
        return np.pad(y, (0, n - len(y))).astype(np.float32)

    def add_noise(self, y: np.ndarray, snr_db: float) -> np.ndarray:
        from .utils import rms_energy
        sr_val = rms_energy(y)
        noise  = np.random.randn(len(y)).astype(np.float32)
        noise /= (rms_energy(noise) + 1e-8)
        scale  = sr_val / (10 ** (snr_db / 20.0))
        return np.clip(y + noise * scale, -1.0, 1.0).astype(np.float32)

    # ── Feature extraction ────────────────────────────────────────────────────

    def mel_power(self, y: np.ndarray) -> np.ndarray:
        return librosa.feature.melspectrogram(
            y=y, sr=self.cfg.SR, n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH, n_mels=self.cfg.N_MELS,
            fmin=20, fmax=8000,
        ).astype(np.float32)

    def mel(self, y: np.ndarray) -> np.ndarray:
        M = self.mel_power(y)
        return librosa.power_to_db(M, ref=np.max).astype(np.float32)

    def pcen(self, y: np.ndarray) -> np.ndarray:
        M = self.mel_power(y)
        P = librosa.pcen(
            M, sr=self.cfg.SR, hop_length=self.cfg.HOP_LENGTH,
            gain=0.8, bias=10.0, power=0.25, time_constant=0.4, eps=1e-6,
        )
        return P.astype(np.float32)

    def feature_stack(self, y: np.ndarray) -> np.ndarray:
        """3-channel feature tensor: [log-mel, PCEN, delta-mel]."""
        m = safe_standardize(self.mel(y))
        p = safe_standardize(self.pcen(y))
        d = safe_standardize(compute_delta_2d(m))
        return np.stack([m, p, d], axis=0).astype(np.float32)

    def compute_bpf_energy_ratio(
        self,
        y: np.ndarray,
        bpf_hz: float,
        bw_hz: float = 20.0,
        n_harmonics: int = 4,
    ) -> float:
        """
        Compute the BPF energy ratio: fraction of total signal power
        concentrated in ±bw_hz bandpass filters centred on the fundamental
        and its first (n_harmonics−1) overtones.

        This matches the Q4 feature definition from analysis.md and the
        measured values in q4_features.csv (Mavic 2 Pro: 0.36–0.54,
        Mavic Pro: 0.10–0.46, Mavic Mini: 0.20–0.40).

        Returns a float in [0, 1].
        """
        y = np.asarray(y, dtype=np.float64)
        sr = self.cfg.SR
        nyq = sr / 2.0
        total_power = float(np.mean(y ** 2)) + 1e-10

        bpf_power = 0.0
        for k in range(1, n_harmonics + 1):
            fc = bpf_hz * k
            if fc + bw_hz >= nyq:
                break
            lo = max(fc - bw_hz, 1.0)
            hi = min(fc + bw_hz, nyq - 1.0)
            sos = scipy.signal.butter(
                4, [lo / nyq, hi / nyq], btype="band", output="sos"
            )
            band = scipy.signal.sosfilt(sos, y)
            bpf_power += float(np.mean(band ** 2))

        return float(np.clip(bpf_power / total_power, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Noise floor generators (real measurement profiles)
# ══════════════════════════════════════════════════════════════════════════════

def _make_indoor_noise(n: int, sr: int, amplitude: float = 0.008) -> np.ndarray:
    """
    Synthesise indoor noise matching the PannoniaFS pre-flight measurements:
    - Broadband noise floor at ≈ −82.7 dB (RMS ≈ 0.007)
    - Tonal interference peaks at 627, 1637, 4363, 2565, 1061, 3746 Hz
      with prominences 22.8, 14.1, 13.3, 11.7, 10.8, 10.8 dB above the floor.
    """
    t    = np.linspace(0, n / sr, n, endpoint=False)
    rng  = np.random.default_rng()
    # Broadband component (slightly pink-ish, not pure white)
    wn   = rng.standard_normal(n).astype(np.float32)
    b, a = scipy.signal.butter(1, 0.05)
    noise = scipy.signal.lfilter(b, a, wn).astype(np.float32)
    noise *= amplitude / (np.std(noise) + 1e-8)

    # Tonal peaks (freq_hz, prominence_dB above floor)
    peaks = [
        (627,  22.8), (1637, 14.1), (4363, 13.3),
        (2565, 11.7), (1061, 10.8), (3746, 10.8),
    ]
    for freq, prom_db in peaks:
        if freq >= sr / 2:
            continue
        gain = amplitude * (10 ** (prom_db / 20.0))
        phase = rng.uniform(0, 2 * np.pi)
        noise += (gain * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)

    return noise.astype(np.float32)


def _make_outdoor_noise(n: int, sr: int, amplitude: float = 0.010) -> np.ndarray:
    """
    Synthesise outdoor Brownian (f⁻²) noise matching Dunakeszi measurements.
    Median PSD approximately −78 dBFS; wind-dominated low-frequency content.
    """
    rng  = np.random.default_rng()
    wn   = rng.standard_normal(n).astype(np.float64)
    # Integrate white noise → Brownian (cumsum = 1/f, power = 1/f²)
    brown = np.cumsum(wn)
    # High-pass at 5 Hz to avoid DC drift
    sos   = scipy.signal.butter(2, 5.0 / (sr / 2), btype="high", output="sos")
    brown = scipy.signal.sosfilt(sos, brown).astype(np.float32)
    brown *= amplitude / (np.std(brown) + 1e-8)
    return brown


def _make_noise(
    n: int,
    sr: int,
    profile: str = "mixed",
    amplitude: Optional[float] = None,
) -> np.ndarray:
    """
    Return a noise array matching the specified measurement profile.

    profile : "indoor" | "outdoor" | "mixed" (random choice per call)
    """
    if profile == "mixed":
        profile = random.choice(["indoor", "outdoor"])
    if profile == "indoor":
        amp = amplitude if amplitude is not None else 0.008
        return _make_indoor_noise(n, sr, amp)
    # outdoor
    amp = amplitude if amplitude is not None else 0.010
    return _make_outdoor_noise(n, sr, amp)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic multi-channel drone signal generator
# ══════════════════════════════════════════════════════════════════════════════

def synthesise_drone(
    mic_positions: np.ndarray,
    src_xy,
    fundamental: Optional[int] = None,
    noise_level: float = 0.03,
    duration: Optional[float] = None,
    sr: Optional[int] = None,
    drone_type: Optional[str] = None,
    noise_profile: str = "mixed",
    cfg: Optional[Config] = None,
) -> List[np.ndarray]:
    """
    Synthesise a physically realistic 3-channel drone recording.
    - If drone_type is given, the fundamental is drawn from the measured
      BPF profile for that drone type (config.sample_fundamental_hz).
    - Noise is generated from real measurement profiles (indoor/outdoor/mixed)
      instead of the simple pink noise.
    - The source signal is still generated once and then delayed per mic
      using fractional-sample delay (correct physics).

    Parameters
    ──────────
    mic_positions : (N, 2) array, mic XY in metres
    src_xy        : [x, y] drone position in metres
    fundamental   : BPF fundamental in Hz; overrides drone_type if given
    noise_level   : relative amplitude of noise (0.03 default)
    duration      : clip length in seconds
    sr            : sample rate
    drone_type    : one of config.DRONE_BPF_PROFILES keys; ignored if
                    fundamental is explicitly supplied
    noise_profile : "indoor" | "outdoor" | "mixed"
    cfg           : Config instance (defaults to module singleton)
    """
    _cfg = cfg or config
    sr   = sr   or _cfg.SR
    dur  = duration or _cfg.TARGET_DURATION
    n    = int(sr * dur)
    t    = np.linspace(0, dur, n, endpoint=False)
    c    = _cfg.SPEED_OF_SOUND
    src  = np.asarray(src_xy, dtype=np.float64)

    # ── Determine fundamental frequency ──────────────────────────────────
    if fundamental is None:
        if drone_type is not None and getattr(_cfg, "SYNTHETIC_USE_MEASURED_BPF", True):
            fundamental = int(round(_cfg.sample_fundamental_hz(drone_type)))
        else:
            fundamental = random.choice([80, 90, 100, 110, 120, 130])

    # ── Build the SOURCE signal ONCE ──────────────────────────────────────
    _, _, _, n_harmonics = _cfg.get_bpf_profile(drone_type or "generic_quad")
    y_src = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        amp  = 1.0 / (k ** 1.3) * (0.9 + 0.2 * random.random())
        ph   = random.uniform(0, 2 * np.pi)
        jit  = 1.0 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
        y_src += amp * np.sin(2 * np.pi * fundamental * k * jit * t + ph)

    # ── Noise from real measurement profile ───────────────────────────────
    noise_amp = max(noise_level, 0.05)
    noise     = _make_noise(n, sr, profile=noise_profile,
                             amplitude=noise_amp * 0.5).astype(np.float64)
    y_src    += noise_amp * noise

    # ── Apply propagation delay + attenuation per mic ─────────────────────
    channels = []
    for mic in mic_positions:
        dist      = max(float(np.linalg.norm(src - mic)), 0.01)
        y_mic     = y_src / (dist ** 0.6 + 0.1)
        delay_smp = dist / c * sr
        if delay_smp > 0:
            y_mic = _fractional_delay(y_mic.astype(np.float32), delay_smp)
        channels.append(y_mic.astype(np.float32))

    return channels