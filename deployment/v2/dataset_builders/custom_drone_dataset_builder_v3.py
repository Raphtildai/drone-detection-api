#!/usr/bin/env python3
# custom_drone_dataset_builder_v2.py

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
import hashlib
from pathlib import Path
from dataclasses import dataclass
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
    test_fraction: float = 0.10
    seed: int = 42

    silence_rms_db_threshold: float = -40.0
    silence_peak_threshold: float = 0.015
    min_active_frame_ratio: float = 0.18

    speech_voiced_ratio_threshold: float = 0.72
    speech_centroid_max_hz: float = 2200.0
    speech_bandwidth_max_hz: float = 1800.0
    speech_f0_max_hz: float = 280.0

    freesound_api_key: str = ""
    freesound_timeout_connect: int = 6
    freesound_timeout_read: int = 30
    freesound_page_size: int = 15
    freesound_max_per_query: int = 10
    freesound_max_total_backgrounds: int = 60
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

    use_urbansound8k: bool = True
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

    synth_fallback_count: int = 30
    synth_clip_sec: float = 6.0
    max_background_pool_total: int = 150

    # New in v2
    split_mode: str = "grouped"  # grouped | random
    include_test_split: bool = True
    segment_group_strategy: str = "source_file"  # source_file | source_prefix
    background_group_strategy: str = "source_family"
    write_csv_manifest: bool = True
    use_source_full_file_as_fallback_positive: bool = False

    # New in v3: purity-aware filtering so speech/silence do not become positives
    quarantine_enabled: bool = True
    trim_edge_margin_sec: float = 0.12
    window_label_drone_threshold: float = 0.58
    window_label_speech_threshold: float = 0.60
    min_drone_window_ratio: float = 0.72
    max_speech_window_ratio: float = 0.12
    max_silence_window_ratio: float = 0.18
    max_uncertain_window_ratio: float = 0.25
    purity_weak_join_threshold: float = 0.44


def safe_slug(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name]
    return "".join(keep).strip("_") or "output_v3"


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


def sliding_window_ranges(n_samples: int, sr: int, window_sec: float, hop_sec: float) -> List[Tuple[int, int]]:
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


def merge_regions(regions: List[Dict], max_gap_sec: float, min_duration_sec: float) -> List[Dict]:
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


def split_long_region(start_s: float, end_s: float, max_chunk_sec: float, overlap_sec: float) -> List[Tuple[float, float]]:
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
    active_ratio = float(np.mean(energies > (10 ** (cfg.silence_rms_db_threshold / 20.0)))) if len(energies) > 0 else 0.0
    is_silent = (
        rms_db < cfg.silence_rms_db_threshold
        or peak < cfg.silence_peak_threshold
        or active_ratio < cfg.min_active_frame_ratio
    )
    return {"rms_db": rms_db, "peak": peak, "active_ratio": active_ratio, "is_silent": bool(is_silent)}


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
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(url: str, out_path: Path, timeout: Tuple[int, int] = (10, 60), session: Optional[requests.Session] = None, headers: Optional[Dict] = None) -> bool:
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    _session = session or requests.Session()
    try:
        with _session.get(url, stream=True, timeout=timeout, headers=headers or {}) as resp:
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


def save_waveform_and_mel_plot(audio_path: Path, y: np.ndarray, sr: int, out_path: Path, title: str):
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
    img = ax2.imshow(mel_db, origin="lower", aspect="auto", extent=[0, len(y) / sr, 0, sr / 2 / 1000], cmap="magma")
    ax2.set_title(f"{title} — Mel Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (kHz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_mix_comparison_plot(drone_y: np.ndarray, bg_y: np.ndarray, mixed_y: np.ndarray, sr: int, out_path: Path, title: str = "Drone + Background Mix"):
    ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(15, 10))
    for i, (name, sig) in enumerate([("Drone Segment", drone_y), ("Background", bg_y), ("Mixed Output", mixed_y)], 1):
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
        S = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            f0, _, _ = librosa.pyin(y, fmin=50.0, fmax=500.0, sr=sr, hop_length=256, fill_na=0.0)
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        voiced_ratio = float(np.mean(f0 > 0.0))
        voiced_f0 = f0[f0 > 0.0]
        f0_median = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception:
        voiced_ratio = f0_median = f0_std = 0.0
    try:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        power = np.mean(S ** 2, axis=1) if S is not None else np.zeros(513, dtype=np.float32)
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
    if voiced_ratio > 0.72 and 70 <= f0_median <= 280 and centroid < 2200 and bandwidth < 1800:
        speech_penalty = 0.25
    score = (
        0.16 * energy_score + 0.18 * f0_score + 0.12 * voiced_score + 0.12 * stability_score
        + 0.09 * centroid_score + 0.07 * rolloff_score + 0.07 * bandwidth_score
        + 0.05 * texture_score + 0.14 * drone_band_score
    ) - speech_penalty
    return {"score": float(np.clip(score, 0.0, 1.0)), "features": {
        "rms_db": rms_db, "centroid_hz": centroid, "rolloff_hz": rolloff, "bandwidth_hz": bandwidth,
        "flatness": flatness, "zcr": zcr, "voiced_ratio": voiced_ratio, "f0_median_hz": f0_median,
        "f0_std_hz": f0_std, "low_mid_ratio": low_mid_ratio, "drone_band_ratio": drone_band_ratio,
        "speech_band_ratio": speech_band_ratio,
    }}


class FreesoundBackgroundDownloader:
    BASE_URL = "https://freesound.org/apiv2"
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.api_key = (cfg.freesound_api_key or os.environ.get("FREESOUND_API_KEY", "")).strip()
        self.session = _make_session()
    def enabled(self) -> bool:
        return bool(self.api_key)
    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        params["token"] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read))
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  ⚠️  Freesound API error: {e}")
            return None
    def _search(self, query: str) -> List[Dict]:
        data = self._get("/search/text/", {
            "query": query,
            "fields": "id,name,previews,license,duration",
            "filter": "duration:[2 TO 30] type:(wav OR mp3 OR ogg)",
            "page_size": self.cfg.freesound_page_size,
            "sort": "downloads_desc",
        })
        return [] if data is None else data.get("results", [])
    def _download_preview(self, sound: Dict, out_path: Path) -> bool:
        previews = sound.get("previews", {})
        url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not url:
            return False
        return download_file(url, out_path, timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read), session=self.session)
    def download_backgrounds(self, save_dir: Path, queries: Optional[List[str]] = None, force: bool = False, max_total: Optional[int] = None) -> Dict[str, int]:
        if not self.enabled():
            print("ℹ️  No Freesound API key found. Skipping Freesound download.")
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
                    duration = sound.get("duration", 0.0)
                    try:
                        duration_str = f"{float(duration):.1f}s"
                    except Exception:
                        duration_str = "?s"
                    print(f"    ✅ {out_path.name} ({duration_str})")
            time.sleep(0.25)
        print(f"✅ Freesound: downloaded={downloaded}, queries={searched}, skipped_existing={skipped}")
        return {"downloaded": downloaded, "queries": searched, "skipped_existing": skipped}


class UrbanSound8KImporter:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.session = _make_session()
    def _dataset_present(self, dataset_root: Path) -> bool:
        audio_root = dataset_root / "UrbanSound8K" / "audio"
        meta_csv = dataset_root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        return audio_root.exists() and meta_csv.exists()
    def _auto_download(self, dataset_root: Path) -> bool:
        ensure_dir(dataset_root)
        tar_path = dataset_root / "UrbanSound8K.tar.gz"
        if not tar_path.exists():
            print(f"\n📥 UrbanSound8K not found – downloading (~6 GB) …")
            ok = download_file(self.cfg.urbansound8k_url, tar_path, timeout=(30, 600), session=self.session)
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
    def import_backgrounds(self, dataset_root: Path, save_dir: Path, force: bool = False, max_total: Optional[int] = None, auto_download: bool = False) -> Dict[str, int]:
        ensure_dir(save_dir)
        max_total = max_total or self.cfg.urbansound8k_max_total_backgrounds
        if not self._dataset_present(dataset_root):
            if auto_download:
                success = self._auto_download(dataset_root)
                if not success or not self._dataset_present(dataset_root):
                    return {"imported": 0, "scanned": 0}
            else:
                print(f"ℹ️  UrbanSound8K not found at {dataset_root}.")
                return {"imported": 0, "scanned": 0}
        audio_root = dataset_root / "UrbanSound8K" / "audio"
        metadata_csv = dataset_root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        import csv
        rows = []
        allowed = set(self.cfg.urbansound8k_allowed_classes)
        with open(metadata_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("class", "").strip() in allowed:
                    rows.append(row)
        random.shuffle(rows)
        imported = scanned = 0
        print(f"\n🎵 UrbanSound8K: importing up to {max_total} clips …")
        for row in rows:
            if imported >= max_total:
                break
            fold = row["fold"]
            fname = row["slice_file_name"]
            cls = row["class"].strip()
            src = audio_root / f"fold{fold}" / fname
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


class SyntheticBackgroundGenerator:
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
                y = rng.standard_normal(n).astype(np.float32)
            elif kind == 1:
                white = rng.standard_normal(n).astype(np.float32)
                b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1.0, -2.494956002, 2.017265875, -0.522189400]
                from scipy.signal import lfilter
                try:
                    y = lfilter(b, a, white).astype(np.float32)
                except Exception:
                    y = white
            elif kind == 2:
                freqs = rng.uniform(40, 200, 8).astype(np.float32)
                t = np.linspace(0, self.cfg.synth_clip_sec, n, dtype=np.float32)
                y = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)
                y += 0.15 * rng.standard_normal(n).astype(np.float32)
            elif kind == 3:
                white = rng.standard_normal(n).astype(np.float32)
                from scipy.signal import butter, sosfilt
                try:
                    sos = butter(4, [100 / (sr / 2), 800 / (sr / 2)], btype="band", output="sos")
                    y = sosfilt(sos, white).astype(np.float32)
                except Exception:
                    y = white
            else:
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


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_source_group(path: Path, strategy: str = "source_file") -> str:
    stem = safe_slug(path.stem.lower())
    if strategy == "source_prefix":
        toks = stem.split("_")
        return "_".join(toks[:2]) if len(toks) >= 2 else stem
    return stem


def infer_background_group(path: Path, strategy: str = "source_family") -> str:
    stem = safe_slug(path.stem.lower())
    parts = [p.lower() for p in path.parts]
    parent = safe_slug(path.parent.name.lower()) if path.parent else "root"
    if strategy == "source_family":
        if "freesound" in stem or "freesound" in parent or "freesound" in parts:
            return stem.split("_")[0] + "_" + (stem.split("_")[1] if len(stem.split("_")) > 1 else parent)
        if stem.startswith("us8k_"):
            toks = stem.split("_")
            return "_".join(toks[:2]) if len(toks) >= 2 else "us8k"
        if stem.startswith("synth_bg_"):
            toks = stem.split("_")
            return "_".join(toks[:3]) if len(toks) >= 3 else "synthetic"
        return f"{parent}_{stem.split('_')[0] if stem else 'bg'}"
    return stem


def assign_groups_to_splits(group_ids: List[str], cfg: BuilderConfig) -> Dict[str, str]:
    uniq = sorted(set(group_ids))
    rng = random.Random(cfg.seed)
    rng.shuffle(uniq)
    n = len(uniq)
    if n == 0:
        return {}
    n_test = int(round(cfg.test_fraction * n)) if cfg.include_test_split else 0
    n_val = int(round(cfg.val_fraction * n))
    if n >= 3 and cfg.include_test_split:
        n_test = max(1, n_test)
    if n >= 2:
        n_val = max(1, n_val)
    if n_test + n_val >= n:
        if cfg.include_test_split and n >= 3:
            n_test = 1
            n_val = 1
        elif n >= 2:
            n_test = 0
            n_val = 1
        else:
            n_test = n_val = 0
    out = {}
    for i, gid in enumerate(uniq):
        if i < n_test:
            out[gid] = "test"
        elif i < n_test + n_val:
            out[gid] = "val"
        else:
            out[gid] = "train"
    return out


def write_json(path: Path, obj: Dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def maybe_write_csv_from_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    import csv
    ensure_dir(path.parent)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def classify_window_purity(clip: np.ndarray, sr: int, cfg: BuilderConfig) -> Dict:
    quality = analyze_window_quality(clip, sr, cfg)
    res = heuristic_drone_score(clip, sr)
    feats = res.get("features", {})
    drone_score = float(res.get("score", 0.0))
    speech_like = looks_like_speech(feats, cfg)
    silence_like = bool(quality.get("is_silent", False))
    periodicity = 0.0
    try:
        ac = librosa.autocorrelate(np.asarray(clip, dtype=np.float32), max_size=min(len(clip), int(sr * 0.03)))
        if len(ac) > 4:
            ac = ac / (np.max(np.abs(ac)) + 1e-8)
            periodicity = float(np.max(ac[1:]))
    except Exception:
        periodicity = 0.0
    if silence_like:
        label = "silence"
    elif speech_like and drone_score < cfg.window_label_speech_threshold:
        label = "speech"
    elif drone_score >= cfg.window_label_drone_threshold and not speech_like:
        label = "drone"
    elif drone_score >= cfg.purity_weak_join_threshold and not silence_like:
        label = "uncertain"
    else:
        label = "uncertain" if not silence_like else "silence"
    return {
        "label": label,
        "score": drone_score,
        "speech_like": bool(speech_like),
        "silence_like": bool(silence_like),
        "periodicity": periodicity,
        "quality": quality,
        "features": feats,
    }


def build_candidate_regions_from_window_labels(windows: List[Dict], max_gap_sec: float) -> List[Tuple[int, int]]:
    candidate_idxs = [i for i, w in enumerate(windows) if w.get("window_label") in ("drone", "uncertain")]
    if not candidate_idxs:
        return []
    groups = []
    start = prev = candidate_idxs[0]
    for idx in candidate_idxs[1:]:
        gap = float(windows[idx]["start_s"]) - float(windows[prev]["end_s"])
        if gap <= max_gap_sec:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def evaluate_region_purity(region_windows: List[Dict], cfg: BuilderConfig) -> Dict:
    n = max(1, len(region_windows))
    labels = [w.get("window_label", "uncertain") for w in region_windows]
    drone_ratio = sum(l == "drone" for l in labels) / n
    speech_ratio = sum(l == "speech" for l in labels) / n
    silence_ratio = sum(l == "silence" for l in labels) / n
    uncertain_ratio = sum(l == "uncertain" for l in labels) / n
    scores = [float(w.get("score", 0.0)) for w in region_windows]
    mean_drone_score = float(np.mean(scores)) if scores else 0.0
    max_drone_score = float(np.max(scores)) if scores else 0.0
    accept = (
        drone_ratio >= cfg.min_drone_window_ratio
        and speech_ratio <= cfg.max_speech_window_ratio
        and silence_ratio <= cfg.max_silence_window_ratio
        and uncertain_ratio <= cfg.max_uncertain_window_ratio
        and mean_drone_score >= cfg.weak_threshold
        and max_drone_score >= cfg.detect_threshold
    )
    return {
        "accept": bool(accept),
        "drone_ratio": float(drone_ratio),
        "speech_ratio": float(speech_ratio),
        "silence_ratio": float(silence_ratio),
        "uncertain_ratio": float(uncertain_ratio),
        "mean_drone_score": mean_drone_score,
        "max_drone_score": max_drone_score,
    }


def trim_region_edges_by_labels(region_windows: List[Dict], cfg: BuilderConfig):
    keep_idx = [i for i, w in enumerate(region_windows) if w.get("window_label") == "drone"]
    if not keep_idx:
        return None
    first, last = keep_idx[0], keep_idx[-1]
    start_s = max(0.0, float(region_windows[first]["start_s"]) - cfg.trim_edge_margin_sec)
    end_s = float(region_windows[last]["end_s"]) + cfg.trim_edge_margin_sec
    if end_s <= start_s:
        return None
    return float(start_s), float(end_s)


def decide_output_bucket(purity: Dict) -> str:
    if purity.get("accept", False):
        return "clean"
    if float(purity.get("speech_ratio", 0.0)) > 0.25:
        return "rejected_speech"
    if float(purity.get("silence_ratio", 0.0)) > 0.35:
        return "rejected_silence"
    return "quarantine_uncertain"


class CustomDroneDatasetBuilder:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    def analyze_file(self, audio_path: Path) -> Dict:
        y = load_audio_any(audio_path, self.cfg.sr)
        total_s = len(y) / self.cfg.sr
        windows = sliding_window_ranges(len(y), self.cfg.sr, self.cfg.segment_window_sec, self.cfg.segment_hop_sec)
        window_details = []
        for i, (s, e) in enumerate(windows):
            clip = y[s:e]
            target_len = int(self.cfg.segment_window_sec * self.cfg.sr)
            if len(clip) < target_len:
                clip = np.pad(clip, (0, target_len - len(clip)))
            cls = classify_window_purity(clip, self.cfg.sr, self.cfg)
            window_details.append({
                "window_index": i,
                "start_s": float(s / self.cfg.sr),
                "end_s": float(e / self.cfg.sr),
                "score": float(cls["score"]),
                "quality": cls["quality"],
                "speech_like": bool(cls["speech_like"]),
                "is_silent": bool(cls["silence_like"]),
                "window_label": cls["label"],
                "periodicity": float(cls.get("periodicity", 0.0)),
                "features": cls["features"],
                "kept": bool(cls["label"] == "drone"),
            })
        candidate_ranges = build_candidate_regions_from_window_labels(window_details, max_gap_sec=self.cfg.merge_gap_sec)
        merged = []
        final_segments = []
        quarantined_segments = []
        for ridx, (a, b) in enumerate(candidate_ranges, 1):
            region_windows = window_details[a:b+1]
            purity = evaluate_region_purity(region_windows, self.cfg)
            trimmed = trim_region_edges_by_labels(region_windows, self.cfg)
            raw_start_s = float(region_windows[0]["start_s"])
            raw_end_s = float(region_windows[-1]["end_s"])
            bucket = decide_output_bucket(purity)
            region_info = {
                "region_index": ridx,
                "start_s": raw_start_s,
                "end_s": raw_end_s,
                "duration_s": float(raw_end_s - raw_start_s),
                "windows": len(region_windows),
                "purity": purity,
                "bucket": bucket,
            }
            if trimmed is not None:
                region_info["trimmed_start_s"] = float(trimmed[0])
                region_info["trimmed_end_s"] = float(trimmed[1])
            merged.append(region_info)
            if trimmed is None:
                continue
            start_s, end_s = trimmed
            if (end_s - start_s) < self.cfg.min_segment_sec:
                continue
            target_list = final_segments if purity.get("accept", False) else quarantined_segments
            for cs, ce in split_long_region(start_s, end_s, max_chunk_sec=self.cfg.max_segment_sec, overlap_sec=self.cfg.segment_overlap_sec):
                seg = {
                    "start_s": float(cs),
                    "end_s": float(ce),
                    "duration_s": float(ce - cs),
                    "score": float(purity.get("max_drone_score", 0.0)),
                    "bucket": bucket,
                    "purity": purity,
                }
                target_list.append(seg)
        return {
            "file": str(audio_path),
            "duration_s": float(total_s),
            "windows_total": len(window_details),
            "detected_regions": merged,
            "segments": final_segments,
            "quarantine_segments": quarantined_segments,
            "window_details": window_details,
        }

    def extract_segments_from_file(self, audio_path: Path, analysis: Dict, out_dir: Path, plot_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
        ensure_dir(out_dir)
        if plot_dir is not None:
            ensure_dir(plot_dir)
        bucket_dirs = {
            "clean": out_dir,
            "quarantine_uncertain": out_dir.parent / "quarantine_uncertain",
            "rejected_speech": out_dir.parent / "rejected_speech",
            "rejected_silence": out_dir.parent / "rejected_silence",
        }
        for p in bucket_dirs.values():
            ensure_dir(p)
        y = load_audio_any(audio_path, self.cfg.sr)
        saved = {k: [] for k in bucket_dirs.keys()}
        segment_candidates = list(analysis.get("segments", [])) + list(analysis.get("quarantine_segments", []))
        for i, seg in enumerate(segment_candidates, 1):
            s = int(seg["start_s"] * self.cfg.sr)
            e = int(seg["end_s"] * self.cfg.sr)
            clip = y[s:e].astype(np.float32)
            if len(clip) == 0:
                continue
            clip = trim_silence_edges(clip, top_db=self.cfg.trim_top_db)
            quality = analyze_window_quality(clip, self.cfg.sr, self.cfg)
            res = heuristic_drone_score(clip, self.cfg.sr)
            purity = dict(seg.get("purity", {}))
            bucket = seg.get("bucket") or decide_output_bucket(purity)
            if quality["is_silent"]:
                bucket = "rejected_silence"
            elif looks_like_speech(res["features"], self.cfg) and float(res.get("score", 0.0)) < self.cfg.window_label_speech_threshold:
                bucket = "rejected_speech"
            if len(clip) < int(self.cfg.min_clip_sec * self.cfg.sr):
                bucket = "rejected_silence" if quality["is_silent"] else "quarantine_uncertain"
            peak = float(np.max(np.abs(clip)) + 1e-8)
            if peak < self.cfg.min_peak:
                bucket = "rejected_silence"
            clip = normalize_peak(clip, peak=self.cfg.clean_peak)
            out_name = f"{safe_slug(audio_path.stem)}_seg_{i:03d}_{int(seg['start_s'] * 1000):07d}_{int(seg['end_s'] * 1000):07d}.wav"
            out_path = bucket_dirs[bucket] / out_name
            sf.write(str(out_path), clip, self.cfg.sr)
            saved[bucket].append(out_path)
            if plot_dir is not None:
                plot_subdir = plot_dir / bucket
                ensure_dir(plot_subdir)
                plot_path = plot_subdir / f"{out_path.stem}.png"
                save_waveform_and_mel_plot(audio_path=out_path, y=clip, sr=self.cfg.sr, out_path=plot_path, title=f"{bucket}: {out_path.stem}")
        return saved

    def extract_from_folder(self, source_root: Path, clean_out_dir: Path, debug_out_dir: Path, recursive: bool = True) -> Dict:
        ensure_dir(clean_out_dir)
        ensure_dir(debug_out_dir)
        quarantine_dir = clean_out_dir.parent / "quarantine_uncertain"
        rejected_speech_dir = clean_out_dir.parent / "rejected_speech"
        rejected_silence_dir = clean_out_dir.parent / "rejected_silence"
        for p in [quarantine_dir, rejected_speech_dir, rejected_silence_dir]:
            ensure_dir(p)
        pattern_iter = source_root.rglob("*") if recursive else source_root.glob("*")
        audio_files = [f for f in pattern_iter if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        if not audio_files:
            print(f"⚠️  No audio files found in {source_root}")
            return {"files": 0, "segments": 0, "saved": 0, "quarantine": 0, "rejected_speech": 0, "rejected_silence": 0}
        total_segments = total_saved = total_quarantine = total_rej_speech = total_rej_silence = 0
        segment_manifest = []
        for fpath in audio_files:
            try:
                analysis = self.analyze_file(fpath)
                source_plot_dir = debug_out_dir / "source_plots"
                ensure_dir(source_plot_dir)
                y_src = load_audio_any(fpath, self.cfg.sr)
                save_waveform_and_mel_plot(audio_path=fpath, y=y_src, sr=self.cfg.sr, out_path=source_plot_dir / f"{safe_slug(fpath.stem)}_source.png", title=f"Source File: {fpath.name}")
                dbg_path = debug_out_dir / f"{safe_slug(fpath.stem)}_analysis.json"
                dbg_path.write_text(json.dumps(analysis, indent=2))
                total_segments += len(analysis.get("segments", [])) + len(analysis.get("quarantine_segments", []))
                saved_map = self.extract_segments_from_file(audio_path=fpath, analysis=analysis, out_dir=clean_out_dir, plot_dir=debug_out_dir / "segment_plots")
                src_group = infer_source_group(fpath, self.cfg.segment_group_strategy)
                for bucket, paths in saved_map.items():
                    for out_path in paths:
                        segment_manifest.append({
                            "segment_path": str(out_path),
                            "segment_name": out_path.name,
                            "bucket": bucket,
                            "source_file": str(fpath),
                            "source_group": src_group,
                            "source_sha1": file_sha1(fpath),
                            "segment_sha1": file_sha1(out_path),
                        })
                total_saved += len(saved_map.get("clean", []))
                total_quarantine += len(saved_map.get("quarantine_uncertain", []))
                total_rej_speech += len(saved_map.get("rejected_speech", []))
                total_rej_silence += len(saved_map.get("rejected_silence", []))
                if not saved_map.get("clean") and self.cfg.use_source_full_file_as_fallback_positive:
                    y = trim_silence_edges(y_src, top_db=self.cfg.trim_top_db)
                    if len(y) >= int(self.cfg.min_clip_sec * self.cfg.sr):
                        out_path = clean_out_dir / f"{safe_slug(fpath.stem)}_full.wav"
                        sf.write(str(out_path), normalize_peak(y, peak=self.cfg.clean_peak), self.cfg.sr)
                        segment_manifest.append({
                            "segment_path": str(out_path), "segment_name": out_path.name, "bucket": "clean", "source_file": str(fpath),
                            "source_group": src_group, "source_sha1": file_sha1(fpath), "segment_sha1": file_sha1(out_path),
                            "fallback_full_file": True,
                        })
                        total_saved += 1
                print(f"🎯 {fpath.name}: windows={analysis['windows_total']} clean={len(saved_map.get('clean', []))} quarantine={len(saved_map.get('quarantine_uncertain', []))} speech_rej={len(saved_map.get('rejected_speech', []))} silence_rej={len(saved_map.get('rejected_silence', []))}")
            except Exception as e:
                print(f"⚠️  Failed processing {fpath.name}: {e}")
        write_json(debug_out_dir / "segment_manifest.json", {"rows": segment_manifest})
        maybe_write_csv_from_rows(debug_out_dir / "segment_manifest.csv", segment_manifest)
        print(f"\n✅ Extraction complete: files={len(audio_files)}, clean={total_saved}, quarantine={total_quarantine}, rejected_speech={total_rej_speech}, rejected_silence={total_rej_silence}")
        return {"files": len(audio_files), "segments": total_segments, "saved": total_saved, "quarantine": total_quarantine, "rejected_speech": total_rej_speech, "rejected_silence": total_rej_silence}

    def build_background_pool(self, background_source_dir: Path, background_pool_dir: Path, max_total: Optional[int] = None) -> List[Path]:
        ensure_dir(background_pool_dir)
        max_total = max_total or self.cfg.max_background_pool_total
        all_candidates = [f for f in background_source_dir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        random.shuffle(all_candidates)
        copied = 0
        bg_files: List[Path] = []
        bg_manifest = []
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
                bg_manifest.append({
                    "background_path": str(dst),
                    "original_path": str(f),
                    "background_group": infer_background_group(f, self.cfg.background_group_strategy),
                    "background_sha1": file_sha1(dst),
                })
                copied += 1
            except Exception as e:
                print(f"⚠️  Failed to copy background {f.name}: {e}")
        write_json(background_pool_dir / "_background_manifest.json", {"rows": bg_manifest})
        maybe_write_csv_from_rows(background_pool_dir / "_background_manifest.csv", bg_manifest)
        print(f"✅ Background pool ready: {copied} files")
        return bg_files

    def _load_segment_manifest(self, clean_drone_dir: Path) -> Dict[str, Dict]:
        manifest_json = clean_drone_dir.parent / "analysis_debug" / "segment_manifest.json"
        if manifest_json.exists():
            try:
                rows = json.loads(manifest_json.read_text()).get("rows", [])
                return {Path(r["segment_path"]).name: r for r in rows if "segment_path" in r}
            except Exception:
                pass
        return {}

    def _load_background_manifest(self, background_pool_dir: Path) -> Dict[str, Dict]:
        manifest_json = background_pool_dir / "_background_manifest.json"
        if manifest_json.exists():
            try:
                rows = json.loads(manifest_json.read_text()).get("rows", [])
                return {Path(r["background_path"]).name: r for r in rows if "background_path" in r}
            except Exception:
                pass
        return {}

    def augment_clean_segments(self, clean_drone_dir: Path, background_pool_dir: Path, train_out_dir: Path, val_out_dir: Path, test_out_dir: Optional[Path] = None, manifest_path: Optional[Path] = None, plot_dir: Optional[Path] = None, max_plot_examples: int = 20) -> Dict:
        ensure_dir(train_out_dir)
        ensure_dir(val_out_dir)
        if self.cfg.include_test_split and test_out_dir is not None:
            ensure_dir(test_out_dir)
        if plot_dir is not None:
            ensure_dir(plot_dir)
        drone_files = sorted(clean_drone_dir.glob("*.wav"))
        bg_files = [f for f in background_pool_dir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        if not drone_files:
            print(f"⚠️  No clean drone clips found in {clean_drone_dir}")
            return {"train": 0, "val": 0, "test": 0, "clean": 0}
        if not bg_files:
            print(f"⚠️  No background clips found in {background_pool_dir}")
            return {"train": 0, "val": 0, "test": 0, "clean": 0}

        seg_manifest = self._load_segment_manifest(clean_drone_dir)
        bg_manifest = self._load_background_manifest(background_pool_dir)

        # Split by clean drone source group, not by augmentation instance.
        drone_group_ids = []
        drone_group_for_file = {}
        for drone_path in drone_files:
            row = seg_manifest.get(drone_path.name, {})
            gid = row.get("source_group") or infer_source_group(drone_path, self.cfg.segment_group_strategy)
            drone_group_ids.append(gid)
            drone_group_for_file[drone_path] = gid
        group_to_split = assign_groups_to_splits(drone_group_ids, self.cfg) if self.cfg.split_mode == "grouped" else {}

        n_train = n_val = n_test = n_clean = 0
        manifest_rows = []
        plotted = 0

        for drone_path in drone_files:
            try:
                drone_y = load_audio_any(drone_path, self.cfg.sr)
                drone_y = trim_silence_edges(drone_y, top_db=self.cfg.trim_top_db)
                if len(drone_y) < int(self.cfg.min_clip_sec * self.cfg.sr):
                    continue

                drone_gid = drone_group_for_file[drone_path]
                split = group_to_split.get(drone_gid, "train") if group_to_split else ("val" if random.random() < self.cfg.val_fraction else "train")
                if split == "test" and (test_out_dir is None or not self.cfg.include_test_split):
                    split = "val"

                split_dir = train_out_dir if split == "train" else val_out_dir if split == "val" else test_out_dir
                assert split_dir is not None

                clean_target = split_dir / f"clean_{drone_path.stem}.wav"
                if not clean_target.exists():
                    sf.write(str(clean_target), normalize_peak(drone_y), self.cfg.sr)
                    n_clean += 1
                    manifest_rows.append({
                        "kind": "clean", "source_clean": str(drone_path), "output_v3": str(clean_target), "split": split,
                        "source_group": drone_gid,
                    })

                for j in range(self.cfg.augments_per_segment):
                    bg_path = random.choice(bg_files)
                    bg_y = load_audio_any(bg_path, self.cfg.sr)
                    bg_y = random_crop_or_loop(bg_y, len(drone_y))
                    dy = np.clip(drone_y * db_to_gain(random.uniform(self.cfg.drone_gain_min_db, self.cfg.drone_gain_max_db)), -1.0, 1.0).astype(np.float32)
                    by = np.clip(bg_y * db_to_gain(random.uniform(self.cfg.bg_gain_min_db, self.cfg.bg_gain_max_db)), -1.0, 1.0).astype(np.float32)
                    snr_db = random.uniform(self.cfg.snr_min_db, self.cfg.snr_max_db)
                    mixed = mix_at_snr(dy, by, snr_db)
                    if random.random() < 0.35:
                        noise = np.random.randn(len(mixed)).astype(np.float32)
                        noise /= np.max(np.abs(noise)) + 1e-8
                        mixed = normalize_peak(mixed + 0.008 * noise)
                    out_name = f"aug_{drone_path.stem}_{j:02d}_{safe_slug(bg_path.stem)}.wav"
                    out_path = split_dir / out_name
                    sf.write(str(out_path), mixed, self.cfg.sr)
                    bg_row = bg_manifest.get(bg_path.name, {})
                    manifest_rows.append({
                        "kind": "augmented", "source_clean": str(drone_path), "background": str(bg_path), "output_v3": str(out_path),
                        "split": split, "snr_db": round(float(snr_db), 3), "source_group": drone_gid,
                        "background_group": bg_row.get("background_group") or infer_background_group(bg_path, self.cfg.background_group_strategy),
                    })
                    if plot_dir is not None and plotted < max_plot_examples:
                        save_mix_comparison_plot(drone_y=dy, bg_y=by, mixed_y=mixed, sr=self.cfg.sr, out_path=plot_dir / f"{out_path.stem}_mix.png", title=f"Mix | {drone_path.stem} + {bg_path.stem} | SNR={snr_db:.2f} dB")
                        plotted += 1
                    if split == "val":
                        n_val += 1
                    elif split == "test":
                        n_test += 1
                    else:
                        n_train += 1
            except Exception as e:
                print(f"⚠️  Failed augmentation for {drone_path.name}: {e}")

        if manifest_path is not None:
            write_json(manifest_path, {"rows": manifest_rows})
            if self.cfg.write_csv_manifest:
                maybe_write_csv_from_rows(manifest_path.with_suffix(".csv"), manifest_rows)
        print(f"✅ Augmentation complete: train={n_train}, val={n_val}, test={n_test}, clean={n_clean}")
        return {"train": n_train, "val": n_val, "test": n_test, "clean": n_clean}


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
        candidates = [Path("/content/drive/MyDrive"), Path("/content/drive/drone_data/MyDrive")]
        for c in candidates:
            if c.exists():
                base = c / "custom_drone_builder"
                ensure_dir(base)
                return base
        raise RuntimeError("Google Drive appears not mounted. Mount it first.")
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
        "output": base_dir / "output_v3",
        "clean": base_dir / "output_v3" / "clean_drone_segments",
        "quarantine": base_dir / "output_v3" / "quarantine_uncertain",
        "rejected_speech": base_dir / "output_v3" / "rejected_speech",
        "rejected_silence": base_dir / "output_v3" / "rejected_silence",
        "debug": base_dir / "output_v3" / "analysis_debug",
        "background_pool": base_dir / "output_v3" / "background_pool",
        "train": base_dir / "output_v3" / "train" / "drone",
        "val": base_dir / "output_v3" / "val" / "drone",
        "test": base_dir / "output_v3" / "test" / "drone",
        "plots": base_dir / "output_v3" / "plots",
        "source_plots": base_dir / "output_v3" / "plots" / "source",
        "segment_plots": base_dir / "output_v3" / "plots" / "segments",
        "mix_plots": base_dir / "output_v3" / "plots" / "mixes",
        "manifest": base_dir / "output_v3" / "augmentation_manifest.json",
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
    out = dirs["output_v3"]
    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)
    print("✅ Cleared output folder.")


def run_builder(source_root=None, backgrounds_root=None, output_root=None, threshold=0.52, weak_threshold=0.34, window_sec=1.25, hop_sec=0.35, augments_per_segment=4, val_frac=0.15, test_frac=0.10, freesound_api_key: str = "", use_urbansound8k: bool = False, urbansound8k_auto_download: bool = False, use_synth_fallback: bool = True, include_test_split: bool = True):
    cfg = BuilderConfig(
        detect_threshold=threshold,
        weak_threshold=weak_threshold,
        segment_window_sec=window_sec,
        segment_hop_sec=hop_sec,
        augments_per_segment=augments_per_segment,
        val_fraction=val_frac,
        test_fraction=test_frac,
        freesound_api_key=freesound_api_key or os.environ.get("FREESOUND_API_KEY", ""),
        use_urbansound8k=use_urbansound8k,
        include_test_split=include_test_split,
    )
    builder = CustomDroneDatasetBuilder(cfg)
    dirs = get_project_dirs()
    source_root = Path(source_root) if source_root else dirs["source"]
    backgrounds_root = Path(backgrounds_root) if backgrounds_root else dirs["backgrounds"]
    if output_root is not None:
        output_root = Path(output_root)
        ensure_dir(output_root)
        clean_dir = output_root / "clean_drone_segments"
        quarantine_dir = output_root / "quarantine_uncertain"
        rejected_speech_dir = output_root / "rejected_speech"
        rejected_silence_dir = output_root / "rejected_silence"
        debug_dir = output_root / "analysis_debug"
        bg_pool_dir = output_root / "background_pool"
        train_dir = output_root / "train" / "drone"
        val_dir = output_root / "val" / "drone"
        test_dir = output_root / "test" / "drone"
        plots_dir = output_root / "plots"
        mix_plots_dir = plots_dir / "mixes"
        manifest_path = output_root / "augmentation_manifest.json"
        for p in [clean_dir, quarantine_dir, rejected_speech_dir, rejected_silence_dir, debug_dir, bg_pool_dir, train_dir, val_dir, test_dir, plots_dir, mix_plots_dir]:
            ensure_dir(p)
    else:
        output_root = dirs["output_v3"]
        clean_dir = dirs["clean"]
        quarantine_dir = dirs["quarantine"]
        rejected_speech_dir = dirs["rejected_speech"]
        rejected_silence_dir = dirs["rejected_silence"]
        debug_dir = dirs["debug"]
        bg_pool_dir = dirs["background_pool"]
        train_dir = dirs["train"]
        val_dir = dirs["val"]
        test_dir = dirs["test"]
        plots_dir = dirs["plots"]
        mix_plots_dir = dirs["mix_plots"]
        manifest_path = dirs["manifest"]
    print("=" * 70)
    print("  CUSTOM DRONE DATASET BUILDER v3")
    print("=" * 70)
    print(f"📁 Source       : {source_root}")
    print(f"📁 Backgrounds  : {backgrounds_root}")
    print(f"📁 Output       : {output_root}")
    ext_stats = builder.extract_from_folder(source_root=source_root, clean_out_dir=clean_dir, debug_out_dir=debug_dir, recursive=True)
    freesound_dir = backgrounds_root / "freesound"
    freesound = FreesoundBackgroundDownloader(cfg)
    freesound_stats = freesound.download_backgrounds(save_dir=freesound_dir, max_total=cfg.freesound_max_total_backgrounds)
    us8k_stats = {"imported": 0, "scanned": 0}
    if cfg.use_urbansound8k:
        urbansound_root = backgrounds_root / "urbansound8k"
        us8k = UrbanSound8KImporter(cfg)
        us8k_stats = us8k.import_backgrounds(dataset_root=urbansound_root, save_dir=backgrounds_root / "urbansound8k_imported", max_total=cfg.urbansound8k_max_total_backgrounds, auto_download=urbansound8k_auto_download)
    synth_stats = {"generated": 0}
    if use_synth_fallback:
        existing_bg = list(backgrounds_root.rglob("*"))
        existing_count = sum(1 for f in existing_bg if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        if existing_count < cfg.synth_fallback_count:
            synth_dir = backgrounds_root / "synthetic"
            gen = SyntheticBackgroundGenerator(cfg)
            synth_clips = gen.generate(save_dir=synth_dir)
            synth_stats["generated"] = len(synth_clips)
            print(f"ℹ️  Real background pool has {existing_count} files – added {len(synth_clips)} synthetic clips.")
    builder.build_background_pool(background_source_dir=backgrounds_root, background_pool_dir=bg_pool_dir, max_total=cfg.max_background_pool_total)
    aug_stats = builder.augment_clean_segments(clean_drone_dir=clean_dir, background_pool_dir=bg_pool_dir, train_out_dir=train_dir, val_out_dir=val_dir, test_out_dir=test_dir, manifest_path=manifest_path, plot_dir=mix_plots_dir, max_plot_examples=24)
    print("\n📊 Final summary")
    print(f"   files analyzed      : {ext_stats.get('files', 0)}")
    print(f"   segments detected   : {ext_stats.get('segments', 0)}")
    print(f"   clean clips saved   : {ext_stats.get('saved', 0)}")
    print(f"   quarantine clips    : {ext_stats.get('quarantine', 0)}")
    print(f"   rejected speech     : {ext_stats.get('rejected_speech', 0)}")
    print(f"   rejected silence    : {ext_stats.get('rejected_silence', 0)}")
    print(f"   freesound downloads : {freesound_stats.get('downloaded', 0)}")
    print(f"   us8k imported       : {us8k_stats.get('imported', 0)}")
    print(f"   synth clips added   : {synth_stats.get('generated', 0)}")
    print(f"   train augmented     : {aug_stats.get('train', 0)}")
    print(f"   val augmented       : {aug_stats.get('val', 0)}")
    print(f"   test augmented      : {aug_stats.get('test', 0)}")
    print(f"   clean copies        : {aug_stats.get('clean', 0)}")
    print(f"   output root         : {output_root}")
    return {
        "extract": ext_stats, "augment": aug_stats, "freesound": freesound_stats, "urbansound8k": us8k_stats, "synth": synth_stats,
        "output_root": str(output_root), "clean_dir": str(clean_dir), "debug_dir": str(debug_dir), "background_pool_dir": str(bg_pool_dir),
        "train_dir": str(train_dir), "val_dir": str(val_dir), "test_dir": str(test_dir), "plots_dir": str(plots_dir), "manifest_path": str(manifest_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone custom drone sound extractor and augmenter (purity-aware v3)")
    parser.add_argument("--source", help="Folder with long source recordings")
    parser.add_argument("--backgrounds", help="Folder with background/noise recordings")
    parser.add_argument("--output_v3", help="Output root folder")
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--weak-threshold", type=float, default=0.34)
    parser.add_argument("--window-sec", type=float, default=1.25)
    parser.add_argument("--hop-sec", type=float, default=0.35)
    parser.add_argument("--augments-per-segment", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--freesound-api-key", default="")
    parser.add_argument("--use-urbansound8k", action="store_true")
    parser.add_argument("--urbansound8k-auto-download", action="store_true")
    parser.add_argument("--no-synth-fallback", action="store_true")
    parser.add_argument("--no-test-split", action="store_true")
    parser.add_argument("--min-drone-window-ratio", type=float, default=0.72)
    parser.add_argument("--max-speech-window-ratio", type=float, default=0.12)
    parser.add_argument("--max-silence-window-ratio", type=float, default=0.18)
    args, unknown = parser.parse_known_args(argv)
    if unknown and not running_in_notebook():
        print(f"⚠️  Ignoring unknown arguments: {unknown}")
    common_kwargs = dict(
        threshold=args.threshold,
        weak_threshold=args.weak_threshold,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        augments_per_segment=args.augments_per_segment,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        freesound_api_key=args.freesound_api_key,
        use_urbansound8k=args.use_urbansound8k,
        urbansound8k_auto_download=args.urbansound8k_auto_download,
        use_synth_fallback=not args.no_synth_fallback,
        include_test_split=not args.no_test_split,
    )
    if args.source and args.backgrounds and args.output_v3:
        return run_builder(source_root=args.source, backgrounds_root=args.backgrounds, output_root=args.output_v3, **common_kwargs)
    if running_in_notebook():
        dirs = get_project_dirs()
        return run_builder(source_root=dirs["source"], backgrounds_root=dirs["backgrounds"], output_root=dirs["output_v3"], **common_kwargs)
    parser.error("--source, --backgrounds, and --output are required when not running interactively.")


if __name__ == "__main__":
    main()

# # Usage
# python custom_drone_dataset_builder_v3.py \
#   --source source \
#   --backgrounds backgrounds \
#   --output output_v3 \
#   --use-urbansound8k \
#   --augments-per-segment 4 \
#   --val-frac 0.15 \
#   --test-frac 0.10