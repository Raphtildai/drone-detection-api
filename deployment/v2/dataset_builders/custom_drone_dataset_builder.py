#!/usr/bin/env python3
# custom_drone_dataset_builder.py

import os
import sys
import time
import math
import json
import random
import shutil
import tempfile
import tarfile
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False


AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff", ".m4a")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuilderConfig:
    sr: int = 22050
    segment_window_sec: float = 1.25
    segment_hop_sec: float = 0.35
    detect_threshold: float = 0.52
    weak_threshold: float = 0.34
    merge_gap_sec: float = 0.40
    min_segment_sec: float = 0.90
    max_segment_sec: float = 4.00
    segment_overlap_sec: float = 0.25

    snr_min_db: float = -5.0
    snr_max_db: float = 15.0
    drone_gain_min_db: float = -6.0
    drone_gain_max_db: float = 6.0
    bg_gain_min_db: float = -6.0
    bg_gain_max_db: float = 6.0

    clean_peak: float = 0.98
    trim_top_db: float = 28.0
    min_peak: float = 0.02
    min_clip_sec: float = 0.4
    augments_per_segment: int = 4
    val_fraction: float = 0.15
    seed: int = 42

    # anti-silence / anti-speech
    silence_rms_db_threshold: float = -40.0
    silence_peak_threshold: float = 0.015
    min_active_frame_ratio: float = 0.18

    speech_voiced_ratio_threshold: float = 0.72
    speech_centroid_max_hz: float = 2200.0
    speech_bandwidth_max_hz: float = 1800.0
    speech_f0_max_hz: float = 280.0

    # ── Freesound (replaces Pixabay) ──────────────────────────────────────────
    # Get a free API key at https://freesound.org/apiv2/apply/
    # You can also pass it via the FREESOUND_API_KEY environment variable.
    freesound_api_key: str = ""
    freesound_timeout_connect: int = 6
    freesound_timeout_read: int = 30
    freesound_page_size: int = 15          # results per query page
    freesound_max_per_query: int = 10      # clips to actually download per query
    freesound_max_total_backgrounds: int = 60
    # Queries sent to Freesound – feel free to extend this list
    freesound_queries: Tuple[str, ...] = (
        "wind outdoor",
        "rain ambience",
        "traffic city",
        "crowd outdoor",
        "construction noise",
        "birds forest",
        "city ambience",
        "engine idle",
        "park ambience",
        "market background",
    )

    # ── UrbanSound8K ─────────────────────────────────────────────────────────
    use_urbansound8k: bool = True
    # Official Zenodo download URL (no login required, ~6 GB)
    urbansound8k_url: str = (
        "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
    )
    urbansound8k_max_total_backgrounds: int = 80
    urbansound8k_allowed_classes: Tuple[str, ...] = (
        "air_conditioner",
        "engine_idling",
        "jackhammer",
        "drilling",
        "siren",
        "car_horn",
        "street_music",
    )

    # ── Synthetic fallback ────────────────────────────────────────────────────
    # When no real backgrounds are found, generate N synthetic noise clips so
    # the augmentation pipeline can still run.
    synth_fallback_count: int = 30
    synth_clip_sec: float = 6.0

    # Global caps
    max_background_pool_total: int = 150


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_slug(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name]
    return "".join(keep).strip("_") or "output"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float32) ** 2)) + 1e-8)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y)) + 1e-8)
    return np.clip(y * (peak / m), -1.0, 1.0).astype(np.float32)


def random_crop_or_loop(y: np.ndarray, target_n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return np.zeros(target_n, dtype=np.float32)
    if len(y) == target_n:
        return y
    if len(y) > target_n:
        start = random.randint(0, len(y) - target_n)
        return y[start:start + target_n]
    reps = int(np.ceil(target_n / max(1, len(y))))
    return np.tile(y, reps)[:target_n].astype(np.float32)


def mix_at_snr(drone_y: np.ndarray, bg_y: np.ndarray, snr_db: float) -> np.ndarray:
    drone_y = np.asarray(drone_y, dtype=np.float32)
    bg_y = np.asarray(bg_y, dtype=np.float32)
    d_rms = rms_energy(drone_y)
    b_rms = rms_energy(bg_y)
    if b_rms < 1e-8:
        return normalize_peak(drone_y)
    scale = (d_rms / (10 ** (snr_db / 20.0))) / b_rms
    return normalize_peak(drone_y + bg_y * scale)


def load_audio_any(path: Path, sr: int) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        if PYDUB_OK:
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            try:
                AudioSegment.from_file(str(path)).export(str(tmp), format="wav")
                y, _ = librosa.load(str(tmp), sr=sr, mono=True)
                return y.astype(np.float32)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
        raise


def trim_silence_edges(y: np.ndarray, top_db: float = 28.0) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32)
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt.astype(np.float32) if len(yt) > 0 else y.astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def sliding_window_ranges(
    n_samples: int, sr: int, window_sec: float, hop_sec: float
) -> List[Tuple[int, int]]:
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if n_samples <= 0 or win <= 0 or hop <= 0:
        return []

    starts = list(range(0, max(1, n_samples - win + 1), hop))
    if n_samples > win and starts and starts[-1] != n_samples - win:
        starts.append(n_samples - win)
    elif n_samples <= win:
        starts = [0]

    return [(s, min(s + win, n_samples)) for s in starts]


def merge_regions(
    regions: List[Dict], max_gap_sec: float, min_duration_sec: float
) -> List[Dict]:
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: r["start_s"])
    merged = [regions[0].copy()]

    for r in regions[1:]:
        prev = merged[-1]
        gap = r["start_s"] - prev["end_s"]
        if gap <= max_gap_sec:
            prev["end_s"] = max(prev["end_s"], r["end_s"])
            prev["max_score"] = max(prev["max_score"], r["max_score"])
            prev["mean_score"] = float((prev["mean_score"] + r["mean_score"]) / 2.0)
            prev["windows"] += r.get("windows", 1)
        else:
            merged.append(r.copy())

    out = []
    for r in merged:
        dur = r["end_s"] - r["start_s"]
        if dur >= min_duration_sec:
            r["duration_s"] = dur
            out.append(r)
    return out


def split_long_region(
    start_s: float, end_s: float, max_chunk_sec: float, overlap_sec: float
) -> List[Tuple[float, float]]:
    dur = end_s - start_s
    if dur <= max_chunk_sec:
        return [(start_s, end_s)]

    step = max(0.2, max_chunk_sec - overlap_sec)
    out = []
    t = start_s
    while t < end_s:
        t2 = min(t + max_chunk_sec, end_s)
        out.append((t, t2))
        if t2 >= end_s:
            break
        t += step
    return out


def analyze_window_quality(y: np.ndarray, sr: int, cfg: BuilderConfig) -> Dict:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return {"rms_db": -120.0, "peak": 0.0, "active_ratio": 0.0, "is_silent": True}

    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    rms_db = 20.0 * math.log10(rms + 1e-8)
    peak = float(np.max(np.abs(y)) + 1e-8)

    frame = max(256, int(0.03 * sr))
    hop = max(128, int(0.01 * sr))

    energies = []
    for start in range(0, max(1, len(y) - frame + 1), hop):
        seg = y[start:start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        energies.append(float(np.sqrt(np.mean(seg ** 2)) + 1e-8))

    energies = np.asarray(energies, dtype=np.float32)
    active_ratio = (
        float(np.mean(energies > (10 ** (cfg.silence_rms_db_threshold / 20.0))))
        if len(energies) > 0 else 0.0
    )

    is_silent = (
        rms_db < cfg.silence_rms_db_threshold
        or peak < cfg.silence_peak_threshold
        or active_ratio < cfg.min_active_frame_ratio
    )

    return {
        "rms_db": rms_db,
        "peak": peak,
        "active_ratio": active_ratio,
        "is_silent": bool(is_silent),
    }


def looks_like_speech(features: Dict, cfg: BuilderConfig) -> bool:
    voiced_ratio = float(features.get("voiced_ratio", 0.0))
    centroid = float(features.get("centroid_hz", 0.0))
    bandwidth = float(features.get("bandwidth_hz", 0.0))
    f0 = float(features.get("f0_median_hz", 0.0))
    f0_std = float(features.get("f0_std_hz", 0.0))

    return bool(
        voiced_ratio >= cfg.speech_voiced_ratio_threshold
        and centroid <= cfg.speech_centroid_max_hz
        and bandwidth <= cfg.speech_bandwidth_max_hz
        and 70.0 <= f0 <= cfg.speech_f0_max_hz
        and f0_std <= 80.0
    )


def _make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Return a requests Session with automatic retry logic."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(
    url: str,
    out_path: Path,
    timeout: Tuple[int, int] = (10, 60),
    session: Optional[requests.Session] = None,
    headers: Optional[Dict] = None,
) -> bool:
    """Stream-download *url* to *out_path*, using a .part temp file."""
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    _session = session or requests.Session()
    try:
        with _session.get(
            url, stream=True, timeout=timeout, headers=headers or {}
        ) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(out_path)
        return True
    except Exception as e:
        print(f"  ⚠️  Download failed for {url}: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_waveform_and_mel_plot(
    audio_path: Path, y: np.ndarray, sr: int, out_path: Path, title: str
):
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, linewidth=0.6)
    ax1.set_title(f"{title} — Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(2, 1, 2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = ax2.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        extent=[0, len(y) / sr, 0, sr / 2 / 1000],
        cmap="magma",
    )
    ax2.set_title(f"{title} — Mel Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (kHz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_mix_comparison_plot(
    drone_y: np.ndarray,
    bg_y: np.ndarray,
    mixed_y: np.ndarray,
    sr: int,
    out_path: Path,
    title: str = "Drone + Background Mix",
):
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(15, 10))
    for i, (name, sig) in enumerate(
        [("Drone Segment", drone_y), ("Background", bg_y), ("Mixed Output", mixed_y)], 1
    ):
        ax = fig.add_subplot(3, 1, i)
        t = np.arange(len(sig)) / sr
        ax.plot(t, sig, linewidth=0.6)
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic drone scoring
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_drone_score(y: np.ndarray, sr: int) -> Dict:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return {"score": 0.0, "features": {}}

    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    rms_db = 20.0 * math.log10(rms + 1e-8)

    try:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)))
    except Exception:
        centroid = rolloff = bandwidth = flatness = zcr = 0.0

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            f0, _, _ = librosa.pyin(
                y, fmin=50.0, fmax=500.0, sr=sr, hop_length=256, fill_na=0.0
            )
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        voiced_ratio = float(np.mean(f0 > 0.0))
        voiced_f0 = f0[f0 > 0.0]
        f0_median = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception:
        voiced_ratio = f0_median = f0_std = 0.0

    try:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        power = np.mean(S ** 2, axis=1)

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            total = float(np.sum(power) + 1e-8)
            return float(np.sum(power[mask]) / total)

        low_mid_ratio = band_ratio(80, 1200)
        drone_band_ratio = band_ratio(80, 2500)
        speech_band_ratio = band_ratio(100, 4000)
    except Exception:
        low_mid_ratio = drone_band_ratio = speech_band_ratio = 0.0

    energy_score = float(np.clip((rms_db + 45.0) / 25.0, 0.0, 1.0))
    f0_score = 1.0 if 70 <= f0_median <= 260 else (0.6 if 50 <= f0_median <= 350 else 0.0)
    voiced_score = float(np.clip(voiced_ratio / 0.55, 0.0, 1.0))
    stability_score = 1.0 - float(np.clip(f0_std / 80.0, 0.0, 1.0)) if f0_median > 0 else 0.2
    centroid_score = 1.0 if 120 <= centroid <= 3500 else 0.35
    rolloff_score = 1.0 if 400 <= rolloff <= 6500 else 0.40
    bandwidth_score = 1.0 if 180 <= bandwidth <= 3200 else 0.50
    texture_score = 1.0 - float(np.clip(flatness / 0.5, 0.0, 1.0))
    drone_band_score = float(np.clip(drone_band_ratio / 0.45, 0.0, 1.0))

    speech_penalty = 0.0
    if (
        voiced_ratio > 0.72
        and 70 <= f0_median <= 280
        and centroid < 2200
        and bandwidth < 1800
    ):
        speech_penalty = 0.25

    score = (
        0.16 * energy_score
        + 0.18 * f0_score
        + 0.12 * voiced_score
        + 0.12 * stability_score
        + 0.09 * centroid_score
        + 0.07 * rolloff_score
        + 0.07 * bandwidth_score
        + 0.05 * texture_score
        + 0.14 * drone_band_score
    ) - speech_penalty

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "features": {
            "rms_db": rms_db,
            "centroid_hz": centroid,
            "rolloff_hz": rolloff,
            "bandwidth_hz": bandwidth,
            "flatness": flatness,
            "zcr": zcr,
            "voiced_ratio": voiced_ratio,
            "f0_median_hz": f0_median,
            "f0_std_hz": f0_std,
            "low_mid_ratio": low_mid_ratio,
            "drone_band_ratio": drone_band_ratio,
            "speech_band_ratio": speech_band_ratio,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background source 1 – Freesound.org  (replaces the broken Pixabay scraper)
# ─────────────────────────────────────────────────────────────────────────────

class FreesoundBackgroundDownloader:
    """
    Downloads ambient/background sounds from Freesound using their public API v2.

    How to get an API key (free):
      1. Register at https://freesound.org/
      2. Go to https://freesound.org/apiv2/apply/
      3. Create an application – you'll get a Client secret which is your API key.
      4. Pass it via BuilderConfig.freesound_api_key  or the
         FREESOUND_API_KEY environment variable.
    """

    BASE_URL = "https://freesound.org/apiv2"

    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.api_key = (
            cfg.freesound_api_key
            or os.environ.get("FREESOUND_API_KEY", "")
        ).strip()
        self.session = _make_session()

    def enabled(self) -> bool:
        return bool(self.api_key)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        params["token"] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(
                url,
                params=params,
                timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  ⚠️  Freesound API error: {e}")
            return None

    def _search(self, query: str) -> List[Dict]:
        """Return a list of sound metadata dicts for *query*."""
        data = self._get(
            "/search/text/",
            {
                "query": query,
                "fields": "id,name,previews,license,duration",
                "filter": "duration:[2 TO 30] type:(wav OR mp3 OR ogg)",
                "page_size": self.cfg.freesound_page_size,
                "sort": "downloads_desc",
            },
        )
        if data is None:
            return []
        return data.get("results", [])

    def _download_preview(self, sound: Dict, out_path: Path) -> bool:
        """Download the HQ preview MP3 (no OAuth needed)."""
        previews = sound.get("previews", {})
        # prefer high-quality; fall back to low-quality
        url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not url:
            return False
        return download_file(
            url,
            out_path,
            timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read),
            session=self.session,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def download_backgrounds(
        self,
        save_dir: Path,
        queries: Optional[List[str]] = None,
        force: bool = False,
        max_total: Optional[int] = None,
    ) -> Dict[str, int]:
        if not self.enabled():
            print(
                "ℹ️  No Freesound API key found.\n"
                "   Set BuilderConfig.freesound_api_key or the FREESOUND_API_KEY env var.\n"
                "   Get a free key at https://freesound.org/apiv2/apply/\n"
                "   Skipping Freesound download."
            )
            return {"downloaded": 0, "queries": 0, "skipped_existing": 0}

        ensure_dir(save_dir)
        queries = list(queries or self.cfg.freesound_queries)
        max_total = max_total or self.cfg.freesound_max_total_backgrounds
        downloaded = skipped = searched = 0

        print(f"\n🌐 Freesound: downloading up to {max_total} background clips …")

        for query in queries:
            if downloaded >= max_total:
                break

            searched += 1
            print(f"  🔎 Query: '{query}'")
            results = self._search(query)

            kept = 0
            for sound in results:
                if downloaded >= max_total or kept >= self.cfg.freesound_max_per_query:
                    break

                sid = sound.get("id")
                name = safe_slug(sound.get("name", f"sound_{sid}"))
                out_path = save_dir / f"freesound_{sid}_{name[:40]}.mp3"

                if out_path.exists() and not force:
                    skipped += 1
                    continue

                ok = self._download_preview(sound, out_path)
                if ok:
                    downloaded += 1
                    kept += 1
                    print(f"    ✅ {out_path.name} ({sound.get('duration', '?'):.1f}s)")

            time.sleep(0.25)  # be polite to the API

        print(
            f"✅ Freesound: downloaded={downloaded}, "
            f"queries={searched}, skipped_existing={skipped}"
        )
        return {"downloaded": downloaded, "queries": searched, "skipped_existing": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Background source 2 – UrbanSound8K  (now with auto-download)
# ─────────────────────────────────────────────────────────────────────────────

class UrbanSound8KImporter:
    """
    Imports (and optionally auto-downloads) UrbanSound8K background clips.

    Auto-download:
      If the dataset is not found at *dataset_root*, the importer will try to
      download the ~6 GB tar.gz from Zenodo and extract it automatically.
      Pass  auto_download=True  (or set  cfg.use_urbansound8k=True  and call
      import_backgrounds with  auto_download=True).
    """

    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.session = _make_session()

    # ── layout detection ─────────────────────────────────────────────────────
    # Three folder layouts are supported:
    #
    # Layout A – standard extracted archive (default from Zenodo):
    #   <root>/UrbanSound8K/audio/fold1/…
    #   <root>/UrbanSound8K/metadata/UrbanSound8K.csv
    #
    # Layout B – already inside the inner folder:
    #   <root>/audio/fold1/…
    #   <root>/metadata/UrbanSound8K.csv
    #
    # Layout C – pre-sorted by class (output of sort_urbansound8k.py):
    #   <root>/air_conditioner/*.wav
    #   <root>/engine_idling/*.wav
    #   … (any of the allowed class names as subfolders)

    _KNOWN_CLASSES = {
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer",
        "siren", "street_music",
    }

    def _detect_layout(self, dataset_root: Path):
        """Return (layout, audio_root_or_None, meta_csv_or_None).
        layout is 'A', 'B', 'C', or None (not found).
        """
        # Layout A
        a_audio = dataset_root / "UrbanSound8K" / "audio"
        a_csv   = dataset_root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        if a_audio.exists() and a_csv.exists():
            return "A", a_audio, a_csv

        # Layout B
        b_audio = dataset_root / "audio"
        b_csv   = dataset_root / "metadata" / "UrbanSound8K.csv"
        if b_audio.exists() and b_csv.exists():
            return "B", b_audio, b_csv

        # Layout C – class-named subfolders present
        subdirs = {d.name for d in dataset_root.iterdir() if d.is_dir()}
        if subdirs & self._KNOWN_CLASSES:
            return "C", dataset_root, None

        return None, None, None

    def _dataset_present(self, dataset_root: Path) -> bool:
        layout, _, _ = self._detect_layout(dataset_root)
        return layout is not None

    def _auto_download(self, dataset_root: Path) -> bool:
        ensure_dir(dataset_root)
        tar_path = dataset_root / "UrbanSound8K.tar.gz"

        if not tar_path.exists():
            print(f"\n📥 UrbanSound8K not found – downloading (~6 GB) …")
            print(f"   URL : {self.cfg.urbansound8k_url}")
            print(f"   Dest: {tar_path}")
            print("   This may take a while depending on your connection.")
            ok = download_file(
                self.cfg.urbansound8k_url,
                tar_path,
                timeout=(30, 600),
                session=self.session,
            )
            if not ok:
                print("❌ UrbanSound8K download failed. Skipping.")
                return False
        else:
            print(f"  📦 Found existing archive: {tar_path}")

        print(f"  📂 Extracting {tar_path.name} …")
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(dataset_root)
            print("  ✅ Extraction complete.")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

    # ── public API ────────────────────────────────────────────────────────────

    def import_backgrounds(
        self,
        dataset_root: Path,
        save_dir: Path,
        force: bool = False,
        max_total: Optional[int] = None,
        auto_download: bool = False,
    ) -> Dict[str, int]:
        """
        Import background clips from UrbanSound8K into *save_dir*.

        Supports three folder layouts automatically:
          A  Standard extracted archive  (<root>/UrbanSound8K/audio/fold1/…)
          B  Already inside inner folder (<root>/audio/fold1/…)
          C  Pre-sorted by class         (<root>/engine_idling/*.wav  etc.)
             This is the output of sort_urbansound8k.py – no CSV needed.
        """
        ensure_dir(save_dir)
        max_total = max_total or self.cfg.urbansound8k_max_total_backgrounds

        layout, audio_root, metadata_csv = self._detect_layout(dataset_root)

        if layout is None:
            if auto_download:
                success = self._auto_download(dataset_root)
                layout, audio_root, metadata_csv = self._detect_layout(dataset_root)
                if layout is None:
                    print("❌  UrbanSound8K still not found after download attempt.")
                    return {"imported": 0, "scanned": 0}
            else:
                print(
                    f"ℹ️  UrbanSound8K not found at {dataset_root}.\n"
                    "   Supported layouts:\n"
                    "     A) standard archive extract: <root>/UrbanSound8K/audio/fold1/\n"
                    "     B) inside inner folder:      <root>/audio/fold1/\n"
                    "     C) pre-sorted by class:      <root>/engine_idling/*.wav  etc.\n"
                    "   Pass --urbansound8k-auto-download to fetch it automatically (~6 GB)."
                )
                return {"imported": 0, "scanned": 0}

        allowed = set(self.cfg.urbansound8k_allowed_classes)
        imported = scanned = 0
        print(f"\n🎵 UrbanSound8K (layout {layout}): importing up to {max_total} clips from {dataset_root} …")

        # ── Layout C: class-named subfolders, no CSV needed ───────────────────
        if layout == "C":
            candidates = []
            for cls_dir in sorted(dataset_root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                if cls not in allowed:
                    continue
                for f in cls_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                        candidates.append((cls, f))

            random.shuffle(candidates)
            for cls, src in candidates:
                if imported >= max_total:
                    break
                scanned += 1
                dst = save_dir / f"us8k_{cls}_{src.name}"
                stem, suffix = dst.stem, dst.suffix
                idx = 1
                while dst.exists() and not force:
                    dst = save_dir / f"{stem}_{idx}{suffix}"
                    idx += 1
                try:
                    shutil.copy2(str(src), str(dst))
                    imported += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to import {src.name}: {e}")

        # ── Layout A / B: fold-based structure with CSV ───────────────────────
        else:
            import csv
            rows = []
            with open(metadata_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("class", "").strip() in allowed:
                        rows.append(row)

            random.shuffle(rows)
            for row in rows:
                if imported >= max_total:
                    break
                fold  = row["fold"]
                fname = row["slice_file_name"]
                cls   = row["class"].strip()
                src   = audio_root / f"fold{fold}" / fname
                if not src.exists():
                    continue
                scanned += 1
                dst = save_dir / f"us8k_{cls}_{fname}"
                stem, suffix = dst.stem, dst.suffix
                idx = 1
                while dst.exists() and not force:
                    dst = save_dir / f"{stem}_{idx}{suffix}"
                    idx += 1
                try:
                    shutil.copy2(str(src), str(dst))
                    imported += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to import {fname}: {e}")

        print(f"✅ UrbanSound8K: imported={imported}, scanned={scanned}")
        return {"imported": imported, "scanned": scanned}


# ─────────────────────────────────────────────────────────────────────────────
# Background source 3 – Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBackgroundGenerator:
    """
    Generates simple synthetic noise backgrounds when no real ones are available.
    Produces a variety of coloured-noise textures so augmented clips are not all
    identical.
    """

    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg

    def generate(self, save_dir: Path, count: Optional[int] = None) -> List[Path]:
        ensure_dir(save_dir)
        count = count or self.cfg.synth_fallback_count
        sr = self.cfg.sr
        n = int(self.cfg.synth_clip_sec * sr)

        saved: List[Path] = []
        print(f"\n🔧 Generating {count} synthetic background clips …")

        for i in range(count):
            kind = i % 5
            rng = np.random.default_rng(self.cfg.seed + i)

            if kind == 0:
                # White noise
                y = rng.standard_normal(n).astype(np.float32)
            elif kind == 1:
                # Pink noise (approximate)
                white = rng.standard_normal(n).astype(np.float32)
                b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], \
                       [1.0, -2.494956002, 2.017265875, -0.522189400]
                from scipy.signal import lfilter
                try:
                    y = lfilter(b, a, white).astype(np.float32)
                except Exception:
                    y = white
            elif kind == 2:
                # Low-frequency rumble
                freqs = rng.uniform(40, 200, 8).astype(np.float32)
                t = np.linspace(0, self.cfg.synth_clip_sec, n, dtype=np.float32)
                y = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)
                y += 0.15 * rng.standard_normal(n).astype(np.float32)
            elif kind == 3:
                # Band-pass noise (traffic-like 100–800 Hz)
                white = rng.standard_normal(n).astype(np.float32)
                from scipy.signal import butter, sosfilt
                try:
                    sos = butter(4, [100 / (sr / 2), 800 / (sr / 2)], btype="band", output="sos")
                    y = sosfilt(sos, white).astype(np.float32)
                except Exception:
                    y = white
            else:
                # High-frequency hiss
                white = rng.standard_normal(n).astype(np.float32)
                from scipy.signal import butter, sosfilt
                try:
                    sos = butter(4, 3000 / (sr / 2), btype="high", output="sos")
                    y = sosfilt(sos, white).astype(np.float32)
                except Exception:
                    y = white

            y = normalize_peak(y, peak=0.85)
            out_path = save_dir / f"synth_bg_{i:04d}_kind{kind}.wav"
            sf.write(str(out_path), y, sr)
            saved.append(out_path)

        print(f"✅ Synthetic backgrounds generated: {len(saved)}")
        return saved


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

class CustomDroneDatasetBuilder:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    def analyze_file(self, audio_path: Path) -> Dict:
        y = load_audio_any(audio_path, self.cfg.sr)
        total_s = len(y) / self.cfg.sr

        windows = sliding_window_ranges(
            len(y), self.cfg.sr, self.cfg.segment_window_sec, self.cfg.segment_hop_sec
        )

        window_details = []
        candidate_regions = []

        for i, (s, e) in enumerate(windows):
            clip = y[s:e]
            target_len = int(self.cfg.segment_window_sec * self.cfg.sr)
            if len(clip) < target_len:
                clip = np.pad(clip, (0, target_len - len(clip)))

            quality = analyze_window_quality(clip, self.cfg.sr, self.cfg)
            res = heuristic_drone_score(clip, self.cfg.sr)
            score = float(res["score"])
            start_s = s / self.cfg.sr
            end_s = e / self.cfg.sr

            speech_like = looks_like_speech(res["features"], self.cfg)
            is_silent = quality["is_silent"]

            keep = False
            if not is_silent and not speech_like:
                if score >= self.cfg.detect_threshold:
                    keep = True
                elif (
                    score >= self.cfg.weak_threshold
                    and res["features"].get("drone_band_ratio", 0.0) >= 0.18
                    and quality["active_ratio"] >= self.cfg.min_active_frame_ratio
                ):
                    keep = True

            window_details.append(
                {
                    "window_index": i,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "score": score,
                    "quality": quality,
                    "speech_like": bool(speech_like),
                    "is_silent": bool(is_silent),
                    "kept": bool(keep),
                    "features": res["features"],
                }
            )

            if keep:
                candidate_regions.append(
                    {
                        "start_s": float(start_s),
                        "end_s": float(end_s),
                        "max_score": score,
                        "mean_score": score,
                        "windows": 1,
                    }
                )

        merged = merge_regions(
            candidate_regions,
            max_gap_sec=self.cfg.merge_gap_sec,
            min_duration_sec=self.cfg.min_segment_sec,
        )

        final_segments = []
        for region in merged:
            start_s = max(0.0, region["start_s"] - 0.15)
            end_s = min(total_s, region["end_s"] + 0.15)
            for cs, ce in split_long_region(
                start_s, end_s,
                max_chunk_sec=self.cfg.max_segment_sec,
                overlap_sec=self.cfg.segment_overlap_sec,
            ):
                final_segments.append(
                    {
                        "start_s": float(cs),
                        "end_s": float(ce),
                        "duration_s": float(ce - cs),
                        "score": float(region["max_score"]),
                    }
                )

        return {
            "file": str(audio_path),
            "duration_s": float(total_s),
            "windows_total": len(window_details),
            "detected_regions": merged,
            "segments": final_segments,
            "window_details": window_details,
        }

    def extract_segments_from_file(
        self,
        audio_path: Path,
        analysis: Dict,
        out_dir: Path,
        plot_dir: Optional[Path] = None,
    ) -> List[Path]:
        ensure_dir(out_dir)
        if plot_dir is not None:
            ensure_dir(plot_dir)

        y = load_audio_any(audio_path, self.cfg.sr)
        saved = []

        for i, seg in enumerate(analysis.get("segments", []), 1):
            s = int(seg["start_s"] * self.cfg.sr)
            e = int(seg["end_s"] * self.cfg.sr)
            clip = y[s:e].astype(np.float32)

            if len(clip) == 0:
                continue

            clip = trim_silence_edges(clip, top_db=self.cfg.trim_top_db)
            quality = analyze_window_quality(clip, self.cfg.sr, self.cfg)
            res = heuristic_drone_score(clip, self.cfg.sr)

            if quality["is_silent"]:
                continue
            if looks_like_speech(res["features"], self.cfg):
                continue
            if len(clip) < int(self.cfg.min_clip_sec * self.cfg.sr):
                continue

            peak = float(np.max(np.abs(clip)) + 1e-8)
            if peak < self.cfg.min_peak:
                continue

            clip = normalize_peak(clip, peak=self.cfg.clean_peak)

            out_name = (
                f"{safe_slug(audio_path.stem)}"
                f"_seg_{i:03d}"
                f"_{int(seg['start_s'] * 1000):07d}"
                f"_{int(seg['end_s'] * 1000):07d}.wav"
            )
            out_path = out_dir / out_name
            sf.write(str(out_path), clip, self.cfg.sr)
            saved.append(out_path)

            if plot_dir is not None:
                plot_path = plot_dir / f"{out_path.stem}.png"
                save_waveform_and_mel_plot(
                    audio_path=out_path,
                    y=clip,
                    sr=self.cfg.sr,
                    out_path=plot_path,
                    title=f"Extracted Segment: {out_path.stem}",
                )

        return saved

    def extract_from_folder(
        self,
        source_root: Path,
        clean_out_dir: Path,
        debug_out_dir: Path,
        recursive: bool = True,
    ) -> Dict:
        ensure_dir(clean_out_dir)
        ensure_dir(debug_out_dir)

        pattern_iter = source_root.rglob("*") if recursive else source_root.glob("*")
        audio_files = [
            f for f in pattern_iter if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ]

        if not audio_files:
            print(f"⚠️  No audio files found in {source_root}")
            return {"files": 0, "segments": 0, "saved": 0}

        total_segments = total_saved = 0

        for fpath in audio_files:
            try:
                analysis = self.analyze_file(fpath)

                source_plot_dir = debug_out_dir / "source_plots"
                ensure_dir(source_plot_dir)
                y_src = load_audio_any(fpath, self.cfg.sr)
                save_waveform_and_mel_plot(
                    audio_path=fpath,
                    y=y_src,
                    sr=self.cfg.sr,
                    out_path=source_plot_dir / f"{safe_slug(fpath.stem)}_source.png",
                    title=f"Source File: {fpath.name}",
                )

                dbg_path = debug_out_dir / f"{safe_slug(fpath.stem)}_analysis.json"
                dbg_path.write_text(json.dumps(analysis, indent=2))

                total_segments += len(analysis.get("segments", []))
                saved = self.extract_segments_from_file(
                    audio_path=fpath,
                    analysis=analysis,
                    out_dir=clean_out_dir,
                    plot_dir=debug_out_dir / "segment_plots",
                )
                total_saved += len(saved)

                print(
                    f"🎯 {fpath.name}: "
                    f"windows={analysis['windows_total']} "
                    f"segments={len(analysis.get('segments', []))} "
                    f"saved={len(saved)}"
                )
            except Exception as e:
                print(f"⚠️  Failed processing {fpath.name}: {e}")

        print(
            f"\n✅ Extraction complete: "
            f"files={len(audio_files)}, segments={total_segments}, saved={total_saved}"
        )
        return {"files": len(audio_files), "segments": total_segments, "saved": total_saved}

    def build_background_pool(
        self,
        background_source_dir: Path,
        background_pool_dir: Path,
        max_total: Optional[int] = None,
    ) -> List[Path]:
        ensure_dir(background_pool_dir)
        max_total = max_total or self.cfg.max_background_pool_total

        all_candidates = [
            f for f in background_source_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ]
        random.shuffle(all_candidates)

        copied = 0
        bg_files: List[Path] = []

        for f in all_candidates:
            if copied >= max_total:
                break

            dst = background_pool_dir / f.name
            stem, suffix = dst.stem, dst.suffix
            idx = 1
            while dst.exists():
                dst = background_pool_dir / f"{stem}_{idx}{suffix}"
                idx += 1

            try:
                shutil.copy2(str(f), str(dst))
                bg_files.append(dst)
                copied += 1
            except Exception as e:
                print(f"⚠️  Failed to copy background {f.name}: {e}")

        print(f"✅ Background pool ready: {copied} files")
        return bg_files

    def augment_clean_segments(
        self,
        clean_drone_dir: Path,
        background_pool_dir: Path,
        train_out_dir: Path,
        val_out_dir: Path,
        manifest_path: Optional[Path] = None,
        plot_dir: Optional[Path] = None,
        max_plot_examples: int = 20,
    ) -> Dict:
        ensure_dir(train_out_dir)
        ensure_dir(val_out_dir)
        if plot_dir is not None:
            ensure_dir(plot_dir)

        drone_files = sorted(clean_drone_dir.glob("*.wav"))
        bg_files = [
            f for f in background_pool_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ]

        if not drone_files:
            print(f"⚠️  No clean drone clips found in {clean_drone_dir}")
            return {"train": 0, "val": 0, "clean": 0}

        if not bg_files:
            print(f"⚠️  No background clips found in {background_pool_dir}")
            return {"train": 0, "val": 0, "clean": 0}

        n_train = n_val = n_clean = 0
        manifest_rows = []
        plotted = 0

        for drone_path in drone_files:
            try:
                drone_y = load_audio_any(drone_path, self.cfg.sr)
                drone_y = trim_silence_edges(drone_y, top_db=self.cfg.trim_top_db)

                if len(drone_y) < int(self.cfg.min_clip_sec * self.cfg.sr):
                    continue

                # always save a clean copy
                clean_target = train_out_dir / f"clean_{drone_path.stem}.wav"
                if not clean_target.exists():
                    sf.write(str(clean_target), normalize_peak(drone_y), self.cfg.sr)
                    n_clean += 1

                for j in range(self.cfg.augments_per_segment):
                    bg_path = random.choice(bg_files)
                    bg_y = load_audio_any(bg_path, self.cfg.sr)
                    bg_y = random_crop_or_loop(bg_y, len(drone_y))

                    dy = np.clip(
                        drone_y * db_to_gain(
                            random.uniform(self.cfg.drone_gain_min_db, self.cfg.drone_gain_max_db)
                        ),
                        -1.0, 1.0,
                    ).astype(np.float32)

                    by = np.clip(
                        bg_y * db_to_gain(
                            random.uniform(self.cfg.bg_gain_min_db, self.cfg.bg_gain_max_db)
                        ),
                        -1.0, 1.0,
                    ).astype(np.float32)

                    snr_db = random.uniform(self.cfg.snr_min_db, self.cfg.snr_max_db)
                    mixed = mix_at_snr(dy, by, snr_db)

                    if random.random() < 0.35:
                        noise = np.random.randn(len(mixed)).astype(np.float32)
                        noise /= np.max(np.abs(noise)) + 1e-8
                        mixed = normalize_peak(mixed + 0.008 * noise)

                    split = "val" if random.random() < self.cfg.val_fraction else "train"
                    out_dir = val_out_dir if split == "val" else train_out_dir
                    out_name = (
                        f"aug_{drone_path.stem}_{j:02d}_{safe_slug(bg_path.stem)}.wav"
                    )
                    out_path = out_dir / out_name
                    sf.write(str(out_path), mixed, self.cfg.sr)

                    manifest_rows.append(
                        {
                            "source_clean": str(drone_path),
                            "background": str(bg_path),
                            "output": str(out_path),
                            "split": split,
                            "snr_db": round(float(snr_db), 3),
                        }
                    )

                    if plot_dir is not None and plotted < max_plot_examples:
                        save_mix_comparison_plot(
                            drone_y=dy,
                            bg_y=by,
                            mixed_y=mixed,
                            sr=self.cfg.sr,
                            out_path=plot_dir / f"{out_path.stem}_mix.png",
                            title=(
                                f"Mix | {drone_path.stem} + {bg_path.stem} "
                                f"| SNR={snr_db:.2f} dB"
                            ),
                        )
                        plotted += 1

                    if split == "val":
                        n_val += 1
                    else:
                        n_train += 1

            except Exception as e:
                print(f"⚠️  Failed augmentation for {drone_path.name}: {e}")

        if manifest_path is not None:
            ensure_dir(manifest_path.parent)
            manifest_path.write_text(json.dumps(manifest_rows, indent=2))

        print(f"✅ Augmentation complete: train={n_train}, val={n_val}, clean={n_clean}")
        return {"train": n_train, "val": n_val, "clean": n_clean}


# ─────────────────────────────────────────────────────────────────────────────
# Storage / Colab helpers
# ─────────────────────────────────────────────────────────────────────────────

def running_in_colab() -> bool:
    return "google.colab" in sys.modules


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def upload_to_colab(target_dir: Path, prompt: str = "Upload files"):
    if not running_in_colab():
        raise RuntimeError("upload_to_colab() only works inside Google Colab.")
    from google.colab import files

    ensure_dir(target_dir)
    print(f"\n📤 {prompt}")
    uploaded = files.upload()

    saved_paths = []
    for name, content in uploaded.items():
        out_path = target_dir / name
        with open(out_path, "wb") as f:
            f.write(content)
        saved_paths.append(out_path)

    print(f"✅ Uploaded {len(saved_paths)} file(s) to {target_dir}")
    return saved_paths


def get_drive_base_dir() -> Path:
    if running_in_colab():
        candidates = [
            Path("/content/drive/MyDrive"),
            Path("/content/drive/drone_data/MyDrive"),
        ]
        for c in candidates:
            if c.exists():
                base = c / "custom_drone_builder"
                ensure_dir(base)
                return base
        raise RuntimeError(
            "Google Drive appears not mounted. "
            "Mount it first: drive.mount('/content/drive')"
        )
    base = Path.cwd() / "custom_drone_builder"
    ensure_dir(base)
    return base


def get_project_dirs(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    if base_dir is None:
        base_dir = get_drive_base_dir()

    dirs = {
        "base": base_dir,
        "inputs": base_dir / "inputs",
        "source": base_dir / "inputs" / "source",
        "backgrounds": base_dir / "inputs" / "backgrounds",
        "output": base_dir / "output",
        "clean": base_dir / "output" / "clean_drone_segments",
        "debug": base_dir / "output" / "analysis_debug",
        "background_pool": base_dir / "output" / "background_pool",
        "train": base_dir / "output" / "train" / "drone",
        "val": base_dir / "output" / "val" / "drone",
        "plots": base_dir / "output" / "plots",
        "source_plots": base_dir / "output" / "plots" / "source",
        "segment_plots": base_dir / "output" / "plots" / "segments",
        "mix_plots": base_dir / "output" / "plots" / "mixes",
        "manifest": base_dir / "output" / "augmentation_manifest.json",
    }

    for k, p in dirs.items():
        if k != "manifest":
            ensure_dir(p)

    return dirs


def upload_sources_to_persistent_drive():
    dirs = get_project_dirs()
    return upload_to_colab(dirs["source"], prompt="Upload source drone recordings")


def upload_backgrounds_to_persistent_drive():
    dirs = get_project_dirs()
    return upload_to_colab(dirs["backgrounds"], prompt="Upload background recordings")


def clear_project_inputs():
    dirs = get_project_dirs()
    for key in ["source", "backgrounds"]:
        for f in dirs[key].rglob("*"):
            if f.is_file():
                f.unlink()
    print("✅ Cleared persistent input folders.")


def clear_project_output():
    dirs = get_project_dirs()
    out = dirs["output"]
    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)
    print("✅ Cleared output folder.")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_builder(
    source_root=None,
    backgrounds_root=None,
    output_root=None,
    threshold=0.52,
    weak_threshold=0.34,
    window_sec=1.25,
    hop_sec=0.35,
    augments_per_segment=4,
    val_frac=0.15,
    freesound_api_key: str = "",
    use_urbansound8k: bool = False,
    urbansound8k_auto_download: bool = False,
    use_synth_fallback: bool = True,
):
    """
    Main entry point.

    Background sources (tried in order, all results pooled):
      1. Freesound.org  – pass freesound_api_key or set FREESOUND_API_KEY env var
      2. UrbanSound8K   – set use_urbansound8k=True; the dataset must already be
                          extracted at <backgrounds_root>/urbansound8k/, unless
                          you also pass urbansound8k_auto_download=True (~6 GB)
      3. Synthetic      – always-available coloured-noise fallback (use_synth_fallback=True)
      4. Any audio files you manually drop into <backgrounds_root>/

    Parameters
    ----------
    freesound_api_key : str
        API key for Freesound. Free at https://freesound.org/apiv2/apply/
        Can also be set via the FREESOUND_API_KEY environment variable.
    use_urbansound8k : bool
        Whether to try importing clips from UrbanSound8K.
    urbansound8k_auto_download : bool
        If True, downloads UrbanSound8K automatically when not present (~6 GB).
    use_synth_fallback : bool
        Generate synthetic noise backgrounds when the real pool is thin.
    """
    cfg = BuilderConfig(
        detect_threshold=threshold,
        weak_threshold=weak_threshold,
        segment_window_sec=window_sec,
        segment_hop_sec=hop_sec,
        augments_per_segment=augments_per_segment,
        val_fraction=val_frac,
        freesound_api_key=freesound_api_key or os.environ.get("FREESOUND_API_KEY", ""),
        use_urbansound8k=use_urbansound8k,
    )

    builder = CustomDroneDatasetBuilder(cfg)

    # Only fall back to Colab/Drive-based dirs when paths were not supplied AND
    # we are actually inside Colab. On a normal local machine we use the paths
    # exactly as given (main() already resolves sensible script-relative defaults).
    if source_root is None or backgrounds_root is None or output_root is None:
        if running_in_colab():
            dirs = get_project_dirs()
            source_root      = source_root      or dirs["source"]
            backgrounds_root = backgrounds_root or dirs["backgrounds"]
            output_root      = output_root      or dirs["output"]
        else:
            # Shouldn't normally reach here – main() always supplies all three.
            # Provide safe local fallbacks for when run_builder() is called directly.
            script_dir = Path(__file__).resolve().parent
            source_root      = source_root      or script_dir / "source"
            backgrounds_root = backgrounds_root or script_dir / "backgrounds"
            output_root      = output_root      or script_dir / "output"

    source_root      = Path(source_root)
    backgrounds_root = Path(backgrounds_root)
    output_root      = Path(output_root)

    ensure_dir(output_root)
    clean_dir     = output_root / "clean_drone_segments"
    debug_dir     = output_root / "analysis_debug"
    bg_pool_dir   = output_root / "background_pool"
    train_dir     = output_root / "train" / "drone"
    val_dir       = output_root / "val"   / "drone"
    plots_dir     = output_root / "plots"
    mix_plots_dir = plots_dir  / "mixes"
    manifest_path = output_root / "augmentation_manifest.json"
    for p in [clean_dir, debug_dir, bg_pool_dir, train_dir, val_dir, plots_dir, mix_plots_dir]:
        ensure_dir(p)

    print("=" * 70)
    print("  CUSTOM DRONE DATASET BUILDER")
    print("=" * 70)
    print(f"📁 Source       : {source_root}")
    print(f"📁 Backgrounds  : {backgrounds_root}")
    print(f"📁 Output       : {output_root}")

    # ── Step 1: extract clean drone segments ──────────────────────────────────
    ext_stats = builder.extract_from_folder(
        source_root=source_root,
        clean_out_dir=clean_dir,
        debug_out_dir=debug_dir,
        recursive=True,
    )

    # ── Step 2: Freesound backgrounds ─────────────────────────────────────────
    freesound_dir = backgrounds_root / "freesound"
    freesound = FreesoundBackgroundDownloader(cfg)
    freesound_stats = freesound.download_backgrounds(
        save_dir=freesound_dir,
        max_total=cfg.freesound_max_total_backgrounds,
    )

    # ── Step 3: UrbanSound8K ──────────────────────────────────────────────────
    us8k_stats = {"imported": 0, "scanned": 0}
    if cfg.use_urbansound8k:
        # Accept the dataset at either:
        #   <backgrounds>/urbansound8k/   (explicit subfolder – most common)
        #   <backgrounds>/                (user pointed --backgrounds straight at the dataset)
        us8k = UrbanSound8KImporter(cfg)
        _us8k_candidates = [
            backgrounds_root / "urbansound8k",
            backgrounds_root,
        ]
        urbansound_root = next(
            (p for p in _us8k_candidates if us8k._detect_layout(p)[0] is not None),
            backgrounds_root / "urbansound8k",   # fallback for auto-download
        )
        print(f"\n🗂️  UrbanSound8K root resolved to: {urbansound_root}")
        us8k_stats = us8k.import_backgrounds(
            dataset_root=urbansound_root,
            save_dir=backgrounds_root / "urbansound8k_imported",
            max_total=cfg.urbansound8k_max_total_backgrounds,
            auto_download=urbansound8k_auto_download,
        )

    # ── Step 4: Synthetic fallback if pool is still thin ──────────────────────
    synth_stats = {"generated": 0}
    if use_synth_fallback:
        existing_bg = list(backgrounds_root.rglob("*"))
        existing_count = sum(
            1 for f in existing_bg if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        )
        if existing_count < cfg.synth_fallback_count:
            synth_dir = backgrounds_root / "synthetic"
            gen = SyntheticBackgroundGenerator(cfg)
            synth_clips = gen.generate(save_dir=synth_dir)
            synth_stats["generated"] = len(synth_clips)
            print(
                f"ℹ️  Real background pool has {existing_count} files – "
                f"added {len(synth_clips)} synthetic clips."
            )

    # ── Step 5: Build capped background pool ──────────────────────────────────
    builder.build_background_pool(
        background_source_dir=backgrounds_root,
        background_pool_dir=bg_pool_dir,
        max_total=cfg.max_background_pool_total,
    )

    # ── Step 6: Augment ───────────────────────────────────────────────────────
    aug_stats = builder.augment_clean_segments(
        clean_drone_dir=clean_dir,
        background_pool_dir=bg_pool_dir,
        train_out_dir=train_dir,
        val_out_dir=val_dir,
        manifest_path=manifest_path,
        plot_dir=mix_plots_dir,
        max_plot_examples=24,
    )

    print("\n📊 Final summary")
    print(f"   files analyzed      : {ext_stats.get('files', 0)}")
    print(f"   segments detected   : {ext_stats.get('segments', 0)}")
    print(f"   clean clips saved   : {ext_stats.get('saved', 0)}")
    print(f"   freesound downloads : {freesound_stats.get('downloaded', 0)}")
    print(f"   us8k imported       : {us8k_stats.get('imported', 0)}")
    print(f"   synth clips added   : {synth_stats.get('generated', 0)}")
    print(f"   train augmented     : {aug_stats.get('train', 0)}")
    print(f"   val augmented       : {aug_stats.get('val', 0)}")
    print(f"   clean copies        : {aug_stats.get('clean', 0)}")
    print(f"   output root         : {output_root}")

    return {
        "extract": ext_stats,
        "augment": aug_stats,
        "freesound": freesound_stats,
        "urbansound8k": us8k_stats,
        "synth": synth_stats,
        "output_root": str(output_root),
        "clean_dir": str(clean_dir),
        "debug_dir": str(debug_dir),
        "background_pool_dir": str(bg_pool_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "plots_dir": str(plots_dir),
        "manifest_path": str(manifest_path),
    }


def main(argv=None):
    # Resolve the directory the script itself lives in so that relative-path
    # defaults ("source", "backgrounds", "output") are anchored there rather
    # than wherever the user happens to call the script from.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Standalone custom drone sound extractor and augmenter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "When --source / --backgrounds / --output are omitted the script\n"
            "looks for  <script_dir>/source  and  <script_dir>/backgrounds  and\n"
            "writes results to  <script_dir>/output  automatically.\n\n"
            "Examples:\n"
            "  # All folders next to the script (simplest)\n"
            "  python custom_drone_dataset_builder.py\n\n"
            "  # Explicit paths\n"
            "  python custom_drone_dataset_builder.py \\\n"
            "    --source ./my_drones --backgrounds ./my_bg --output ./out\n\n"
            "  # With Freesound downloads\n"
            "  python custom_drone_dataset_builder.py \\\n"
            "    --freesound-api-key YOUR_KEY\n"
        ),
    )

    parser.add_argument(
        "--source",
        default=None,
        help=f"Folder with source drone recordings  [default: <script_dir>/source]",
    )
    parser.add_argument(
        "--backgrounds",
        default=None,
        help=f"Folder with background/noise recordings  [default: <script_dir>/backgrounds]",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output root folder  [default: <script_dir>/output]",
    )
    parser.add_argument("--threshold",            type=float, default=0.52,
                        help="Drone detection score threshold  [default: 0.52]")
    parser.add_argument("--weak-threshold",       type=float, default=0.34,
                        help="Weak detection threshold  [default: 0.34]")
    parser.add_argument("--window-sec",           type=float, default=1.25,
                        help="Analysis window size in seconds  [default: 1.25]")
    parser.add_argument("--hop-sec",              type=float, default=0.35,
                        help="Analysis hop size in seconds  [default: 0.35]")
    parser.add_argument("--augments-per-segment", type=int,   default=4,
                        help="Number of augmented copies per clean segment  [default: 4]")
    parser.add_argument("--val-frac",             type=float, default=0.15,
                        help="Fraction of augmented clips sent to validation  [default: 0.15]")
    parser.add_argument(
        "--freesound-api-key",
        default="",
        help=(
            "Freesound API key (or set FREESOUND_API_KEY env var). "
            "Free at https://freesound.org/apiv2/apply/"
        ),
    )
    parser.add_argument(
        "--use-urbansound8k",
        action="store_true",
        help="Import UrbanSound8K backgrounds from <backgrounds>/urbansound8k/",
    )
    parser.add_argument(
        "--urbansound8k-auto-download",
        action="store_true",
        help="Auto-download UrbanSound8K from Zenodo if not found (~6 GB).",
    )
    parser.add_argument(
        "--no-synth-fallback",
        action="store_true",
        help="Disable synthetic noise background generation.",
    )

    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"⚠️  Ignoring unknown arguments: {unknown}")

    # ── resolve paths, falling back to script-relative defaults ──────────────
    # Relative paths are resolved relative to the script's directory so the
    # command works the same regardless of where the user calls it from.
    def _resolve(arg, default_name):
        p = Path(arg) if arg else Path(default_name)
        return p if p.is_absolute() else (script_dir / p).resolve()

    source_root      = _resolve(args.source,      "source")
    backgrounds_root = _resolve(args.backgrounds, "backgrounds")
    output_root      = _resolve(args.output,      "output")

    # Warn clearly if the source folder looks empty / missing
    if not source_root.exists():
        print(f"⚠️  Source folder not found, creating it: {source_root}")
        source_root.mkdir(parents=True, exist_ok=True)
        print(
            "   Drop your drone recording files into that folder and re-run.\n"
            "   Supported formats: .wav .mp3 .ogg .flac .aif .aiff .m4a"
        )

    if not backgrounds_root.exists():
        print(f"ℹ️  Backgrounds folder not found, creating it: {backgrounds_root}")
        backgrounds_root.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        source_root=source_root,
        backgrounds_root=backgrounds_root,
        output_root=output_root,
        threshold=args.threshold,
        weak_threshold=args.weak_threshold,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        augments_per_segment=args.augments_per_segment,
        val_frac=args.val_frac,
        freesound_api_key=args.freesound_api_key,
        use_urbansound8k=args.use_urbansound8k,
        urbansound8k_auto_download=args.urbansound8k_auto_download,
        use_synth_fallback=not args.no_synth_fallback,
    )

    # Notebook / Colab: honour Drive-based project dirs if inside Colab;
    # otherwise use the resolved paths above.
    if running_in_colab():
        print("\n🧭 Google Colab detected – using Drive-based project folders.")
        dirs = get_project_dirs()
        print(f"📁 Source folder      : {dirs['source']}")
        print(f"📁 Backgrounds folder : {dirs['backgrounds']}")
        print(f"📁 Output folder      : {dirs['output']}")
        print("\nUseful helpers:")
        print("  upload_sources_to_persistent_drive()")
        print("  upload_backgrounds_to_persistent_drive()")
        print("  clear_project_inputs()  /  clear_project_output()")
        print("  run_builder(freesound_api_key='YOUR_KEY')")
        common_kwargs.update(
            source_root=dirs["source"],
            backgrounds_root=dirs["backgrounds"],
            output_root=dirs["output"],
        )

    return run_builder(**common_kwargs)


if __name__ == "__main__":
    main()

# Usage
# # Simplest — just drop files in source/ and backgrounds/ next to the script
# python custom_drone_dataset_builder.py

# # With explicit source + backgrounds, auto output
# python custom_drone_dataset_builder.py --source source --backgrounds backgrounds

# # Fully explicit
# python custom_drone_dataset_builder.py \
#   --source source --backgrounds backgrounds --output output

# # With UrbanSound8K you've already downloaded
# python custom_drone_dataset_builder.py \
#   --use-urbansound8k \
#   --source source --backgrounds backgrounds
