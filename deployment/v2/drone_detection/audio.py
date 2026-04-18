# -*- coding: utf-8 -*-
"""
drone_detection/audio.py
────────────────────────
Audio I/O and processing:
  - AudioProcessor  (load, mel, PCEN, feature_stack, pad/truncate, noise)
  - synthesise_drone()
  - compute_ipd_features()
  - AudioWebScraper
  - collect_background_pool()
  - convert helpers
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import scipy.signal
import soundfile as sf

from .config import AUDIO_EXTS, config as _default_cfg
from .utils import (
    _fractional_delay,
    normalize_peak,
    safe_standardize,
    compute_delta_2d,
)

# ── Optional deps ─────────────────────────────────────────────────────────────
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

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


# ── AudioProcessor ────────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Wraps all audio I/O and feature-extraction operations.

    The v15 feature stack is:
        channel 0 — standardised log-mel
        channel 1 — standardised PCEN
        channel 2 — standardised Δ-mel (first-order temporal difference)

    This replaces the v13 3×log-mel repetition for better generalisation.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or _default_cfg

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, path, mono: bool = True) -> np.ndarray:
        """Load and resample to cfg.SR. Falls back to pydub for exotic formats."""
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

    def load_channels(self, path, channel_indices=None) -> List[np.ndarray]:
        """Load a multi-channel file; return list of mono arrays."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=self.cfg.SR, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        idx = channel_indices if channel_indices is not None else list(range(y.shape[0]))
        return [y[i].astype(np.float32) for i in idx if i < y.shape[0]]

    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        n = int(self.cfg.SR * self.cfg.TARGET_DURATION)
        if len(y) >= n:
            return y[:n].astype(np.float32)
        return np.pad(y, (0, n - len(y))).astype(np.float32)

    def add_noise(self, y: np.ndarray, snr_db: float) -> np.ndarray:
        from .utils import rms_energy
        sr  = rms_energy(y)
        n   = np.random.randn(len(y)).astype(np.float32)
        n  /= (rms_energy(n) + 1e-8)
        scale = sr / (10 ** (snr_db / 20.0))
        return np.clip(y + n * scale, -1.0, 1.0).astype(np.float32)

    # ── Feature extraction ────────────────────────────────────────────────────

    def mel_power(self, y: np.ndarray) -> np.ndarray:
        return librosa.feature.melspectrogram(
            y=y, sr=self.cfg.SR, n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH, n_mels=self.cfg.N_MELS,
            fmin=20, fmax=8000,
        ).astype(np.float32)

    def mel(self, y: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram (dB, ref=max)."""
        M = self.mel_power(y)
        return librosa.power_to_db(M, ref=np.max).astype(np.float32)

    def pcen(self, y: np.ndarray) -> np.ndarray:
        """Per-Channel Energy Normalisation — more robust to level changes."""
        M = self.mel_power(y)
        P = librosa.pcen(
            M, sr=self.cfg.SR, hop_length=self.cfg.HOP_LENGTH,
            gain=0.8, bias=10.0, power=0.25, time_constant=0.4, eps=1e-6,
        )
        return P.astype(np.float32)

    def feature_stack(self, y: np.ndarray) -> np.ndarray:
        """
        Build the v15 3-channel feature tensor (C, F, T):
          [standardised log-mel, standardised PCEN, standardised Δ-mel]
        """
        m = safe_standardize(self.mel(y))
        p = safe_standardize(self.pcen(y))
        d = safe_standardize(compute_delta_2d(m))
        return np.stack([m, p, d], axis=0).astype(np.float32)


# ── Standalone audio helpers ──────────────────────────────────────────────────

def load_audio_any(path, sr: int) -> np.ndarray:
    """Load any audio file to float32 mono; uses pydub as fallback."""
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


def convert_to_wav(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".wav":
        shutil.copy2(str(src), str(dst))
    elif _PYDUB_OK:
        AudioSegment.from_file(str(src)).export(str(dst), format="wav")
    else:
        raise RuntimeError(f"pydub not available for {src.suffix}")


def compute_ipd_features(channels: List[np.ndarray], cfg=None) -> np.ndarray:
    """
    Inter-Microphone Phase Differences for pairs (0,1), (0,2), (1,2).
    Returns a float32 array of shape (3,).
    """
    cfg = cfg or _default_cfg
    sr  = cfg.SR
    pairs = [(0, 1), (0, 2), (1, 2)]
    ipds  = []
    for i, j in pairs:
        xi = channels[i].astype(np.float64)
        xj = channels[j].astype(np.float64)
        n  = max(len(xi), len(xj))
        Xi = np.fft.rfft(xi, n=n); Xj = np.fft.rfft(xj, n=n)
        cc = Xi * np.conj(Xj); cc /= (np.abs(cc) + 1e-8)
        tau = np.fft.irfft(cc, n=n)
        pk  = int(np.argmax(np.abs(tau)))
        if pk > n // 2:
            pk -= n
        ipds.append(float(pk) / sr)
    return np.array(ipds, dtype=np.float32)


def synthesise_drone(
    mic_positions: np.ndarray,
    src_xy,
    fundamental: int     = 100,
    noise_level: float   = 0.03,
    duration: float      = None,
    sr: int              = None,
) -> List[np.ndarray]:
    """
    Simulate drone audio on a microphone array using harmonic synthesis
    with physically-correct TDOA delays.

    Parameters
    ----------
    mic_positions : array (n_mics, 2) — mic XY positions in metres
    src_xy        : [x, y] source position in metres
    fundamental   : blade-pass fundamental frequency (Hz)
    noise_level   : amplitude of pink + white noise floor
    duration      : clip duration (default cfg.TARGET_DURATION)
    sr            : sample rate (default cfg.SR)

    Returns
    -------
    List of float32 arrays, one per microphone.
    """
    sr  = sr  or _default_cfg.SR
    dur = duration or _default_cfg.TARGET_DURATION
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    c   = _default_cfg.SPEED_OF_SOUND
    src = np.asarray(src_xy, dtype=np.float64)

    channels = []
    for mic in mic_positions:
        dist = max(float(np.linalg.norm(src - mic)), 0.01)
        sd   = int(dist / c * sr)
        y    = np.zeros(n, dtype=np.float64)

        # Harmonic blade-pass series
        for k in range(1, 9):
            amp  = 1.0 / (k ** 1.3) * (0.9 + 0.2 * random.random())
            ph   = random.uniform(0, 2 * np.pi)
            jit  = 1.0 + 0.003 * np.sin(2 * np.pi * 0.5 * t)
            y   += amp * np.sin(2 * np.pi * fundamental * k * jit * t + ph)

        # Distance attenuation + noise
        y /= (dist ** 0.6 + 0.1)
        wn = np.random.randn(n)
        b, a = scipy.signal.butter(1, 0.05)
        pink = scipy.signal.lfilter(b, a, wn)
        y   += noise_level * pink + noise_level * 0.5 * np.random.randn(n)

        # Propagation delay
        if sd > 0:
            y = np.concatenate([np.zeros(sd), y[:-sd]])
        channels.append(y.astype(np.float32))

    return channels


def perturb_multichannel(
    channels: List[np.ndarray], cfg=None
) -> List[np.ndarray]:
    """Light per-channel augmentation used by LocalizationDataset."""
    cfg  = cfg or _default_cfg
    ap   = AudioProcessor(cfg)
    from .utils import db_to_gain, random_eq_tilt
    out  = []
    base = random.uniform(-3.0, 3.0)
    for ch in channels:
        x = np.asarray(ch, np.float32).copy()
        x *= db_to_gain(base + random.uniform(-1.5, 1.5))
        if random.random() < 0.5:
            x = ap.add_noise(x, random.uniform(0, 20))
        if random.random() < 0.25:
            x = _fractional_delay(x, random.uniform(0.0, 2.0))
        if random.random() < 0.25:
            x = random_eq_tilt(x, cfg.SR, 0.25)
        out.append(np.clip(x, -1, 1).astype(np.float32))
    return out


# ── Multi-source audio web scraper ────────────────────────────────────────────

class AudioWebScraper:
    """
    Download drone and non-drone audio from multiple free public sources:
      • BBC Sound Effects   (no key)
      • xeno-canto          (no key)
      • SoundBible          (no key)
      • FreeSound.io        (no key)
      • Freesound.org       (optional API key)
      • yt-dlp YouTube      (opt-in)
    """

    AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")
    _BBC_DRONE = ["drone", "buzz", "propeller", "rotor"]
    _BBC_NON   = ["wind", "crowd", "traffic", "rain", "birds", "urban"]
    _SOUNDBIBLE = {"non_drone": ["1575", "1480", "1350", "1288", "1192"]}

    def __init__(self, cfg=None):
        self.cfg      = cfg or _default_cfg
        self.api_key  = getattr(self.cfg, "FREESOUND_API_KEY", "")
        self.out_root = self.cfg.RAW_DIR / "scraped_audio"
        if _REQUESTS_OK:
            import requests
            self.sess = requests.Session()
            self.sess.headers.update({"User-Agent": "DroneDetectionResearch/1.0"})
        else:
            self.sess = None

    def download(self, force: bool = False) -> None:
        if not _REQUESTS_OK:
            print("⚠️  requests not installed — scraping skipped")
            return
        for label in ["drone", "non_drone"]:
            (self.out_root / label).mkdir(parents=True, exist_ok=True)
        n = self._count()
        if n > 0 and not force:
            print(f"✅ Scraped audio already exists ({n} files) — skipping.")
            return
        print("🌐 Multi-source audio scraping …")
        self._scrape_freesound()
        self._scrape_freesound_io()
        self._scrape_bbc()
        self._scrape_xeno_canto()
        self._scrape_soundbible()
        if getattr(self.cfg, "SCRAPE_YTDLP_ENABLED", False) and _YTDLP_OK:
            self._scrape_youtube()
        print(f"✅ Scraping done — {self._count()} files collected.")

    def _count(self) -> int:
        n = 0
        for label in ["drone", "non_drone"]:
            d = self.out_root / label
            if d.exists():
                for ext in self.AUDIO_EXTS:
                    n += len(list(d.glob(f"*{ext}")))
        return n

    def _scrape_freesound(self) -> None:
        if not self.api_key:
            print("  ℹ️  No FREESOUND_API_KEY — skipping Freesound.org")
            return
        import re as _re
        print("  🔎 Freesound.org …")
        base = "https://freesound.org/apiv2/search/text/"
        queries = {
            "drone":     ["drone flying", "quadcopter", "uav sound", "drone propeller"],
            "non_drone": ["wind", "car passing", "crowd noise", "bird chirping"],
        }
        for label, terms in queries.items():
            sd = self.out_root / label
            for term in terms:
                try:
                    data = self.sess.get(
                        base,
                        params={
                            "query": term,
                            "filter": "duration:[2 TO 15]",
                            "fields": "id,name,previews,duration",
                            "page_size": self.cfg.SCRAPE_MAX_PER_QUERY,
                            "token": self.api_key,
                        },
                        timeout=10,
                    ).json()
                except Exception as e:
                    print(f"    ⚠️  FS ({term}): {e}"); continue
                for s in data.get("results", []):
                    d = s.get("duration", 0)
                    if not (self.cfg.SCRAPE_MIN_DURATION <= d <= self.cfg.SCRAPE_MAX_DURATION):
                        continue
                    fp = sd / f"fs_{s['id']}.mp3"
                    if fp.exists():
                        continue
                    try:
                        c = self.sess.get(s["previews"]["preview-hq-mp3"], timeout=10).content
                        if len(c) > 5000:
                            fp.write_bytes(c)
                    except Exception:
                        pass

    def _scrape_freesound_io(self) -> None:
        import re as _re
        print("  🔎 FreeSound.io (key-free) …")
        base = "https://freesound.io/api/sounds/search"
        queries = {
            "drone":     ["drone", "uav", "quadcopter"],
            "non_drone": ["wind", "crowd", "ambient"],
        }
        for label, terms in queries.items():
            sd = self.out_root / label
            for term in terms:
                try:
                    resp  = self.sess.get(base, params={"query": term, "limit": 20}, timeout=8)
                    data  = resp.json()
                    items = data.get("results", data if isinstance(data, list) else [])
                    for item in items:
                        url = item.get("download_url") or item.get("url", "")
                        if not url or not any(url.endswith(e) for e in self.AUDIO_EXTS):
                            continue
                        fid = _re.sub(r"[^\w]", "_", url[-20:])
                        ext = Path(url).suffix or ".mp3"
                        fp  = sd / f"fsio_{fid}{ext}"
                        if fp.exists():
                            continue
                        try:
                            c = self.sess.get(url, timeout=10).content
                            if len(c) > 5000:
                                fp.write_bytes(c)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"    ⚠️  FSio ({term}): {e}")

    def _scrape_bbc(self) -> None:
        print("  🔎 BBC Sound Effects …")
        base = "https://sound-effects.bbcrewind.co.uk/api/search"
        qs   = {"drone": self._BBC_DRONE, "non_drone": self._BBC_NON}
        for label, terms in qs.items():
            sd = self.out_root / label
            for term in terms:
                try:
                    resp   = self.sess.get(base, params={"q": term, "limit": 15}, timeout=8)
                    data   = resp.json()
                    sounds = data.get("results", data.get("sounds", []))
                    for s in sounds:
                        sid = s.get("id", s.get("assetId", ""))
                        if not sid:
                            continue
                        url = f"https://sound-effects-media.bbcrewind.co.uk/zip/{sid}.wav"
                        fp  = sd / f"bbc_{sid}.wav"
                        if fp.exists():
                            continue
                        try:
                            c = self.sess.get(url, timeout=12).content
                            if len(c) > 10000:
                                fp.write_bytes(c)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"    ⚠️  BBC ({term}): {e}")

    def _scrape_xeno_canto(self) -> None:
        print("  🔎 xeno-canto …")
        sd = self.out_root / "non_drone"
        for term in ["wind", "rain", "stream", "ambient"]:
            try:
                data = self.sess.get(
                    "https://xeno-canto.org/api/2/recordings",
                    params={"query": term, "page": 1},
                    timeout=8,
                ).json()
                for rec in data.get("recordings", [])[:10]:
                    url = "https:" + rec.get("file", "")
                    if url == "https:":
                        continue
                    fp = sd / f"xc_{rec.get('id', '')}.mp3"
                    if fp.exists():
                        continue
                    try:
                        c = self.sess.get(url, timeout=12).content
                        if len(c) > 5000:
                            fp.write_bytes(c)
                    except Exception:
                        pass
            except Exception as e:
                print(f"    ⚠️  XC ({term}): {e}")

    def _scrape_soundbible(self) -> None:
        print("  🔎 SoundBible …")
        sd = self.out_root / "non_drone"
        for sid in self._SOUNDBIBLE["non_drone"]:
            fp = sd / f"sb_{sid}.mp3"
            if fp.exists():
                continue
            try:
                c = self.sess.get(
                    f"https://soundbible.com/grab.php?id={sid}&type=mp3", timeout=10
                ).content
                if len(c) > 5000:
                    fp.write_bytes(c)
            except Exception:
                pass

    def _scrape_youtube(self) -> None:
        if not _YTDLP_OK:
            print("  ⚠️  yt-dlp not installed.")
            return
        print("  🔎 yt-dlp (YouTube) …")
        sd   = self.out_root / "drone"
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(sd / "yt_%(id)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "download_sections": {"*": "00:00:00-00:00:30"},
            "quiet": True, "no_warnings": True,
        }
        for url in getattr(self.cfg, "SCRAPE_YTDLP_URLS", []):
            try:
                with ytdlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"    ⚠️  yt-dlp ({url}): {e}")


# ── Background pool helpers ───────────────────────────────────────────────────

def incorporate_scraped_audio(cfg=None, force: bool = False) -> None:
    """Copy scraped audio into the detection processed directories."""
    cfg     = cfg or _default_cfg
    scraped = cfg.RAW_DIR / "scraped_audio"
    if not scraped.exists():
        print("ℹ️  No scraped audio — skipping.")
        return
    det = cfg.PROCESSED_DIR / "detection"
    ap  = AudioProcessor(cfg)
    for label in ["drone", "non_drone"]:
        src = scraped / label
        if not src.exists():
            continue
        files = []
        for ext in AudioWebScraper.AUDIO_EXTS:
            files.extend(src.glob(f"*{ext}"))
        if not files:
            continue
        random.shuffle(files)
        si  = int(len(files) * 0.85)
        splits = {"train": files[:si], "val": files[si:]}
        added = skipped = failed = 0
        for split, fl in splits.items():
            dst = det / split / label
            dst.mkdir(parents=True, exist_ok=True)
            for f in fl:
                out = dst / f"{f.stem}.wav"
                if out.exists() and not force:
                    skipped += 1; continue
                try:
                    y = ap.pad_or_truncate(load_audio_any(f, cfg.SR))
                    sf.write(str(out), y, cfg.SR)
                    added += 1
                except Exception as e:
                    failed += 1; print(f"   ⚠️  {f.name}: {e}")
        print(f"   ✅ {label}: added={added} skip={skipped} fail={failed}")


def collect_background_pool(cfg=None) -> Dict[str, list]:
    """
    Build a categorised pool of background files for mixing.
    Includes processed non_drone WAVs, scraped audio, and the
    custom-builder background pool (if present).
    """
    cfg  = cfg or _default_cfg
    pool = {k: [] for k in ["speech", "crowd", "wind", "traffic", "non_drone"]}

    for split in ["train", "val", "test"]:
        d = cfg.PROCESSED_DIR / "detection" / split / "non_drone"
        if not d.exists():
            continue
        for f in d.glob("*.wav"):
            pool["non_drone"].append(f)
            nm = f.stem.lower()
            for kws, bkt in [
                (["speech", "talk", "voice"], "speech"),
                (["crowd", "market"],         "crowd"),
                (["wind", "breeze"],          "wind"),
                (["traffic", "car", "road"],  "traffic"),
            ]:
                if any(k in nm for k in kws):
                    pool[bkt].append(f)

    # Scraped audio
    scraped = cfg.RAW_DIR / "scraped_audio" / "non_drone"
    if scraped.exists():
        for f in scraped.glob("*.*"):
            if f.suffix.lower() not in AudioWebScraper.AUDIO_EXTS:
                continue
            pool["non_drone"].append(f)

    # Custom-builder background pool
    custom_bg = Path(getattr(cfg, "CUSTOM_DATASET_IMPORTED_ROOT", "")) / "background_pool"
    if custom_bg.exists():
        from .config import AUDIO_EXTS as _AEXTS
        for f in custom_bg.rglob("*"):
            if f.is_file() and f.suffix.lower() in _AEXTS:
                pool["non_drone"].append(f)

    # Fill empty sub-pools from non_drone
    for k in ["speech", "crowd", "wind", "traffic"]:
        if not pool[k]:
            pool[k] = list(pool["non_drone"])

    # De-duplicate
    for k in pool:
        seen, uniq = set(), []
        for p in pool[k]:
            s = str(p)
            if s not in seen:
                uniq.append(p); seen.add(s)
        pool[k] = uniq

    print("📚 Background pool:")
    for k in pool:
        print(f"   {k:10s}: {len(pool[k])}")
    return pool