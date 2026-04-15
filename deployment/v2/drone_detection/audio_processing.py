# -*- coding: utf-8 -*-
"""
audio_processing.py
───────────────────
AudioProcessor — core class for loading, padding, and feature extraction.
Also contains synthesise_drone() for generating synthetic 3-mic recordings.

v15 features
────────────
  feature_stack() returns a 3-channel tensor [log-mel, PCEN, delta-mel]
  instead of the old 3× repeated log-mel.  This richer representation
  substantially improves generalisation to real-world recordings.
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

from .config import Config, config
from .utils import (
    _fractional_delay,
    normalize_peak,
    safe_standardize,
    compute_delta_2d,
)


class AudioProcessor:
    """
    Handles all audio loading and feature-extraction operations.

    Parameters
    ──────────
    cfg : Config
        Pipeline configuration.  Defaults to the module-level singleton.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or config

    # ──────────────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, path, mono: bool = True) -> np.ndarray:
        """
        Load an audio file to a float32 array resampled to cfg.SR.
        Falls back to pydub for non-native formats.
        """
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
        self,
        path,
        channel_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Load a multi-channel file and return selected channels as a list."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=self.cfg.SR, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        idx = channel_indices if channel_indices is not None else list(range(y.shape[0]))
        return [y[i].astype(np.float32) for i in idx if i < y.shape[0]]

    # ──────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        """Crop or zero-pad y to exactly cfg.TARGET_DURATION samples."""
        n = int(self.cfg.SR * self.cfg.TARGET_DURATION)
        if len(y) >= n:
            return y[:n].astype(np.float32)
        return np.pad(y, (0, n - len(y))).astype(np.float32)

    def add_noise(self, y: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white noise at a given SNR (dB)."""
        from .utils import rms_energy
        sr_val = rms_energy(y)
        noise  = np.random.randn(len(y)).astype(np.float32)
        noise /= (rms_energy(noise) + 1e-8)
        scale  = sr_val / (10 ** (snr_db / 20.0))
        return np.clip(y + noise * scale, -1.0, 1.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Feature extraction (v15)
    # ──────────────────────────────────────────────────────────────────────────

    def mel_power(self, y: np.ndarray) -> np.ndarray:
        """Compute a raw power mel-spectrogram (not in dB)."""
        return librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.SR,
            n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH,
            n_mels=self.cfg.N_MELS,
            fmin=20,
            fmax=8000,
        ).astype(np.float32)

    def mel(self, y: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram (dB, referenced to maximum power)."""
        M = self.mel_power(y)
        return librosa.power_to_db(M, ref=np.max).astype(np.float32)

    def pcen(self, y: np.ndarray) -> np.ndarray:
        """
        Per-Channel Energy Normalisation (PCEN).
        More robust than log-mel under varying background noise.
        """
        M = self.mel_power(y)
        P = librosa.pcen(
            M,
            sr=self.cfg.SR,
            hop_length=self.cfg.HOP_LENGTH,
            gain=0.8,
            bias=10.0,
            power=0.25,
            time_constant=0.4,
            eps=1e-6,
        )
        return P.astype(np.float32)

    def feature_stack(self, y: np.ndarray) -> np.ndarray:
        """
        v15 3-channel feature tensor: [log-mel, PCEN, delta-mel].

        Each channel is independently standardised (zero-mean, unit-variance)
        so the CNN sees balanced gradients regardless of absolute energy.

        Returns
        ───────
        np.ndarray  shape (3, N_MELS, T)
        """
        m = safe_standardize(self.mel(y))
        p = safe_standardize(self.pcen(y))
        d = safe_standardize(compute_delta_2d(m))
        return np.stack([m, p, d], axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic multi-channel drone signal generator
# ══════════════════════════════════════════════════════════════════════════════

def synthesise_drone(
    mic_positions: np.ndarray,
    src_xy,
    fundamental: int = 100,
    noise_level: float = 0.03,
    duration: Optional[float] = None,
    sr: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Synthesise a physically realistic 3-channel drone recording.

    Models
    ──────
    - Harmonic comb at `fundamental` Hz with amplitude 1/k^1.3
    - Per-harmonic Doppler jitter (±0.3 %)
    - Pink-noise propwash
    - Propagation delay (fractional sample)
    - 1/r^0.6 distance attenuation

    Parameters
    ──────────
    mic_positions : (3, 2) array, mic XY coordinates in metres
    src_xy        : [x, y] drone position in metres
    fundamental   : blade-pass fundamental frequency (Hz)
    noise_level   : relative amplitude of propwash noise
    duration      : clip length in seconds (defaults to cfg.TARGET_DURATION)
    sr            : sample rate (defaults to cfg.SR)

    Returns
    ───────
    List[np.ndarray]  one float32 array per microphone
    """
    sr  = sr  or config.SR
    dur = duration or config.TARGET_DURATION
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    c   = config.SPEED_OF_SOUND
    src = np.asarray(src_xy, dtype=np.float64)
    # ── Build the SOURCE signal ONCE ──────────────────────────────────────
    # Harmonic comb
    y_src = np.zeros(n, dtype=np.float64)
    for k in range(1, 9):
        amp = 1.0 / (k ** 1.3) * (0.9 + 0.2 * random.random())
        ph  = random.uniform(0, 2 * np.pi)
        jit = 1.0 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
        y_src += amp * np.sin(2 * np.pi * fundamental * k * jit * t + ph)

    # Pink-ish propwash — generated ONCE so the delay is preserved across mics
    # noise_level enforced >= 0.05 so GCC-PHAT has broadband content to lock onto
    noise_amp = max(noise_level, 0.05)
    wn   = np.random.randn(n)
    b, a = scipy.signal.butter(1, 0.05)
    pink = scipy.signal.lfilter(b, a, wn)
    y_src += noise_amp * pink + noise_amp * 0.5 * np.random.randn(n)
    # ── End source ────────────────────────────────────────────────────────

    channels = []
    for mic in mic_positions:
        dist = max(float(np.linalg.norm(src - mic)), 0.01)

        # Distance attenuation
        y_mic = y_src / (dist ** 0.6 + 0.1)

        # Sub-sample fractional delay (replaces integer-only shift)
        delay_samp = dist / c * sr
        if delay_samp > 0:
            y_mic = _fractional_delay(y_mic.astype(np.float32), delay_samp)

        channels.append(y_mic.astype(np.float32))

    return channels