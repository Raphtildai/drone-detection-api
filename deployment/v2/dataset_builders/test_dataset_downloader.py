#!/usr/bin/env python3
# test_dataset_downloader.py
"""
Independent downloader for building a strictly held-out drone test dataset.

Features
- Keeps test data separate from training code
- Downloads from YouTube via yt-dlp using URL lists
- Downloads outdoor non-drone audio from Xeno-canto
- Downloads drone/non-drone previews from Freesound (optional API key)
- Imports your own local raw recordings
- Normalizes folder structure and writes manifests

Output structure
output/
├── drone/
├── non_drone/
└── manifests/
"""

"""
Independent downloader for a strictly held-out drone test dataset,
with optional automatic filtering.

What auto filtering does
- rejects near-silence
- rejects speech-like clips
- optionally clips long files into drone-dominant windows
- can quarantine uncertain files instead of discarding them

This is still independent from the training pipeline.
It does NOT augment or mix audio.
"""

import os
import csv
import json
import time
import math
import argparse
import warnings
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False


AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff", ".webm", ".opus")


@dataclass
class DownloaderConfig:
    sr: int = 22050
    normalize_audio: bool = False
    trim_silence: bool = False
    trim_top_db: float = 28.0
    min_duration_sec: float = 1.0
    max_duration_sec: float = 1800.0

    youtube_enabled: bool = True
    yt_audio_format: str = "bestaudio/best"
    yt_extract_audio_codec: str = "wav"
    yt_limit_per_list: int = 500

    xenocanto_enabled: bool = False
    xenocanto_queries: Tuple[str, ...] = (
        "wind",
        "rain",
        "stream",
        "ambient",
        "birds",
        "forest",
        "field recording",
    )
    xenocanto_max_total: int = 120
    xenocanto_per_query: int = 20

    freesound_enabled: bool = False
    freesound_api_key: str = ""
    freesound_queries_drone: Tuple[str, ...] = (
        "drone flyby",
        "quadcopter",
        "uav",
        "drone hovering",
    )
    freesound_queries_non_drone: Tuple[str, ...] = (
        "wind outdoor",
        "traffic city",
        "crowd outdoor",
        "park ambience",
        "construction noise",
        "generator hum",
        "motorbike passby",
        "helicopter distant",
    )
    freesound_max_total_drone: int = 80
    freesound_max_total_non_drone: int = 160
    freesound_page_size: int = 15
    freesound_timeout_connect: int = 8
    freesound_timeout_read: int = 30

    retries: int = 3
    backoff: float = 0.5
    user_agent: str = "TestDatasetDownloader/2.0"

    # Auto filtering
    auto_filter: bool = False
    quarantine_uncertain: bool = True
    split_long_files: bool = True
    filter_window_sec: float = 0.80
    filter_hop_sec: float = 0.25
    filter_merge_gap_sec: float = 0.30
    filter_max_clip_sec: float = 8.0
    filter_edge_margin_sec: float = 0.10

    silence_rms_db_threshold: float = -40.0
    silence_peak_threshold: float = 0.015
    min_active_frame_ratio: float = 0.18

    speech_voiced_ratio_threshold: float = 0.72
    speech_centroid_max_hz: float = 2200.0
    speech_bandwidth_max_hz: float = 1800.0
    speech_f0_max_hz: float = 280.0

    strong_drone_threshold: float = 0.58
    weak_drone_threshold: float = 0.44
    min_drone_window_ratio: float = 0.70
    max_speech_window_ratio: float = 0.15
    max_silence_window_ratio: float = 0.20
    max_uncertain_window_ratio: float = 0.30


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name]
    return "".join(keep).strip("_") or "output"


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y)) + 1e-8)
    return np.clip(y * (peak / m), -1.0, 1.0).astype(np.float32)


def trim_silence_edges(y: np.ndarray, top_db: float = 28.0) -> np.ndarray:
    if not LIBROSA_OK:
        return y.astype(np.float32)
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt.astype(np.float32) if len(yt) > 0 else y.astype(np.float32)
    except Exception:
        return y.astype(np.float32)


def make_session(cfg: DownloaderConfig) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=cfg.retries,
        backoff_factor=cfg.backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": cfg.user_agent})
    return sess


def read_text_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def load_audio_any(path: Path, sr: int) -> np.ndarray:
    if LIBROSA_OK:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                y, _ = librosa.load(str(path), sr=sr, mono=True)
            return y.astype(np.float32)
        except Exception:
            pass

    if PYDUB_OK:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            AudioSegment.from_file(str(path)).export(str(tmp), format="wav")
            if LIBROSA_OK:
                y, _ = librosa.load(str(tmp), sr=sr, mono=True)
                return y.astype(np.float32)
            data, _ = sf.read(str(tmp))
            if data.ndim > 1:
                data = data.mean(axis=1)
            return np.asarray(data, dtype=np.float32)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    data, _ = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def analyze_window_quality(y: np.ndarray, sr: int, cfg: DownloaderConfig) -> Dict:
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


def looks_like_speech(features: Dict, cfg: DownloaderConfig) -> bool:
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
    except Exception:
        S = None
        centroid = rolloff = bandwidth = flatness = 0.0

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
        if S is None:
            raise RuntimeError("no_stft")
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        power = np.mean(S ** 2, axis=1)
        total = float(np.sum(power) + 1e-8)

        def band_ratio(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(np.sum(power[mask]) / total)

        drone_band_ratio = band_ratio(80, 2500)
    except Exception:
        drone_band_ratio = 0.0

    energy_score = float(np.clip((rms_db + 45.0) / 25.0, 0.0, 1.0))
    f0_score = 1.0 if 70 <= f0_median <= 260 else (0.6 if 50 <= f0_median <= 350 else 0.0)
    voiced_score = float(np.clip(voiced_ratio / 0.55, 0.0, 1.0))
    stability_score = 1.0 - float(np.clip(f0_std / 80.0, 0.0, 1.0)) if f0_median > 0 else 0.2
    centroid_score = 1.0 if 120 <= centroid <= 3500 else 0.35
    bandwidth_score = 1.0 if 180 <= bandwidth <= 3200 else 0.50
    texture_score = 1.0 - float(np.clip(flatness / 0.5, 0.0, 1.0))
    drone_band_score = float(np.clip(drone_band_ratio / 0.45, 0.0, 1.0))

    speech_penalty = 0.0
    if voiced_ratio > 0.72 and 70 <= f0_median <= 280 and centroid < 2200 and bandwidth < 1800:
        speech_penalty = 0.25

    score = (
        0.18 * energy_score
        + 0.18 * f0_score
        + 0.12 * voiced_score
        + 0.12 * stability_score
        + 0.10 * centroid_score
        + 0.08 * bandwidth_score
        + 0.08 * texture_score
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
            "voiced_ratio": voiced_ratio,
            "f0_median_hz": f0_median,
            "f0_std_hz": f0_std,
            "drone_band_ratio": drone_band_ratio,
        },
    }


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


def split_long_region(start_s: float, end_s: float, max_chunk_sec: float, overlap_sec: float = 0.25) -> List[Tuple[float, float]]:
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


def classify_window(clip: np.ndarray, sr: int, cfg: DownloaderConfig, label_hint: str) -> Dict:
    quality = analyze_window_quality(clip, sr, cfg)
    silence_like = bool(quality["is_silent"])

    if label_hint == "drone":
        res = heuristic_drone_score(clip, sr)
        feats = res["features"]
        drone_score = float(res["score"])
        speech_like = looks_like_speech(feats, cfg)

        if silence_like:
            label = "silence"
        elif speech_like and drone_score < 0.60:
            label = "speech"
        elif drone_score >= cfg.strong_drone_threshold and not speech_like:
            label = "drone"
        elif drone_score >= cfg.weak_drone_threshold and not speech_like:
            label = "uncertain"
        else:
            label = "uncertain"

        return {
            "window_label": label,
            "score": drone_score,
            "quality": quality,
            "features": feats,
        }

    # non_drone filtering is simpler: reject silence, optionally reject speech-only
    if silence_like:
        label = "silence"
    else:
        res = heuristic_drone_score(clip, sr)
        feats = res["features"]
        speech_like = looks_like_speech(feats, cfg)
        if speech_like:
            label = "speech"
        else:
            label = "non_drone"

    return {
        "window_label": label,
        "score": float(res["score"]) if 'res' in locals() else 0.0,
        "quality": quality,
        "features": feats if 'feats' in locals() else {},
    }


def build_candidate_regions(window_rows: List[Dict], cfg: DownloaderConfig, label_hint: str) -> List[Tuple[int, int]]:
    if label_hint == "drone":
        keep_idxs = [i for i, w in enumerate(window_rows) if w["window_label"] in ("drone", "uncertain")]
    else:
        keep_idxs = [i for i, w in enumerate(window_rows) if w["window_label"] == "non_drone"]

    if not keep_idxs:
        return []

    groups = []
    start = keep_idxs[0]
    prev = keep_idxs[0]
    for idx in keep_idxs[1:]:
        gap = window_rows[idx]["start_s"] - window_rows[prev]["end_s"]
        if gap <= cfg.filter_merge_gap_sec:
            prev = idx
        else:
            groups.append((start, prev))
            start = idx
            prev = idx
    groups.append((start, prev))
    return groups


def evaluate_region(region_windows: List[Dict], cfg: DownloaderConfig, label_hint: str) -> Dict:
    labels = [w["window_label"] for w in region_windows]
    n = max(1, len(labels))
    drone_ratio = sum(l == "drone" for l in labels) / n
    speech_ratio = sum(l == "speech" for l in labels) / n
    silence_ratio = sum(l == "silence" for l in labels) / n
    uncertain_ratio = sum(l == "uncertain" for l in labels) / n
    non_drone_ratio = sum(l == "non_drone" for l in labels) / n
    mean_score = float(np.mean([w["score"] for w in region_windows])) if region_windows else 0.0
    max_score = float(np.max([w["score"] for w in region_windows])) if region_windows else 0.0

    if label_hint == "drone":
        accept = (
            drone_ratio >= cfg.min_drone_window_ratio
            and speech_ratio <= cfg.max_speech_window_ratio
            and silence_ratio <= cfg.max_silence_window_ratio
            and uncertain_ratio <= cfg.max_uncertain_window_ratio
            and mean_score >= cfg.weak_drone_threshold
            and max_score >= cfg.strong_drone_threshold
        )
    else:
        accept = (
            non_drone_ratio >= 0.65
            and silence_ratio <= 0.25
        )

    return {
        "accept": bool(accept),
        "drone_ratio": float(drone_ratio),
        "speech_ratio": float(speech_ratio),
        "silence_ratio": float(silence_ratio),
        "uncertain_ratio": float(uncertain_ratio),
        "non_drone_ratio": float(non_drone_ratio),
        "mean_score": mean_score,
        "max_score": max_score,
    }


def trim_region_edges(region_windows: List[Dict], cfg: DownloaderConfig, label_hint: str):
    target = "drone" if label_hint == "drone" else "non_drone"
    keep_idxs = [i for i, w in enumerate(region_windows) if w["window_label"] == target]
    if not keep_idxs:
        return None
    first = keep_idxs[0]
    last = keep_idxs[-1]
    start_s = max(0.0, region_windows[first]["start_s"] - cfg.filter_edge_margin_sec)
    end_s = region_windows[last]["end_s"] + cfg.filter_edge_margin_sec
    if end_s <= start_s:
        return None
    return float(start_s), float(end_s)


def auto_filter_segments(y: np.ndarray, cfg: DownloaderConfig, label_hint: str) -> List[Dict]:
    windows = sliding_window_ranges(len(y), cfg.sr, cfg.filter_window_sec, cfg.filter_hop_sec)
    if not windows:
        return []

    rows = []
    for i, (s, e) in enumerate(windows):
        clip = y[s:e]
        target_len = int(cfg.filter_window_sec * cfg.sr)
        if len(clip) < target_len:
            clip = np.pad(clip, (0, target_len - len(clip)))
        cls = classify_window(clip, cfg.sr, cfg, label_hint)
        rows.append({
            "window_index": i,
            "start_s": float(s / cfg.sr),
            "end_s": float(e / cfg.sr),
            **cls,
        })

    candidate_ranges = build_candidate_regions(rows, cfg, label_hint)
    segments = []
    for ridx, (a, b) in enumerate(candidate_ranges, 1):
        region_windows = rows[a:b+1]
        purity = evaluate_region(region_windows, cfg, label_hint)
        trimmed = trim_region_edges(region_windows, cfg, label_hint)
        if trimmed is None:
            continue
        s0, e0 = trimmed
        if (e0 - s0) < cfg.min_duration_sec:
            continue

        pieces = split_long_region(s0, e0, cfg.filter_max_clip_sec)
        for piece_id, (ps, pe) in enumerate(pieces, 1):
            if (pe - ps) < cfg.min_duration_sec:
                continue
            bucket = "accepted" if purity["accept"] else "quarantine"
            segments.append({
                "region_id": ridx,
                "piece_id": piece_id,
                "start_s": float(ps),
                "end_s": float(pe),
                "duration_s": float(pe - ps),
                "bucket": bucket,
                "purity": purity,
            })
    return segments


def prepare_audio_file(src: Path, dst: Path, cfg: DownloaderConfig, label_hint: str, quarantine_dir: Optional[Path] = None) -> List[Dict]:
    try:
        y = load_audio_any(src, cfg.sr)
    except Exception as e:
        return [{"ok": False, "reason": f"load_failed:{e}", "source_path": str(src)}]

    if cfg.trim_silence:
        y = trim_silence_edges(y, cfg.trim_top_db)

    duration = len(y) / float(cfg.sr)
    if duration < cfg.min_duration_sec:
        return [{"ok": False, "reason": f"too_short:{duration:.3f}s", "source_path": str(src)}]
    if duration > cfg.max_duration_sec:
        return [{"ok": False, "reason": f"too_long:{duration:.3f}s", "source_path": str(src)}]

    if cfg.normalize_audio:
        y = normalize_peak(y)

    rows = []

    if not cfg.auto_filter:
        ensure_dir(dst.parent)
        sf.write(str(dst), y, cfg.sr)
        rows.append({
            "ok": True,
            "source_path": str(src),
            "output_path": str(dst),
            "duration_s": float(duration),
            "bucket": "accepted",
        })
        return rows

    segments = auto_filter_segments(y, cfg, label_hint)
    if not segments:
        rows.append({
            "ok": False,
            "source_path": str(src),
            "reason": "no_segments_after_filter",
        })
        return rows

    accepted_count = 0
    for seg in segments:
        s = int(seg["start_s"] * cfg.sr)
        e = int(seg["end_s"] * cfg.sr)
        ys = y[s:e].astype(np.float32)
        if len(ys) == 0:
            continue
        if cfg.normalize_audio:
            ys = normalize_peak(ys)

        base = f"{safe_slug(src.stem)}_r{seg['region_id']:03d}_p{seg['piece_id']:02d}.wav"
        if seg["bucket"] == "accepted":
            out_path = dst.parent / base
            ensure_dir(out_path.parent)
            sf.write(str(out_path), ys, cfg.sr)
            accepted_count += 1
            rows.append({
                "ok": True,
                "source_path": str(src),
                "output_path": str(out_path),
                "duration_s": float(seg["duration_s"]),
                "bucket": "accepted",
                **{f"purity_{k}": v for k, v in seg["purity"].items()},
            })
        elif cfg.quarantine_uncertain and quarantine_dir is not None:
            out_path = quarantine_dir / label_hint / base
            ensure_dir(out_path.parent)
            sf.write(str(out_path), ys, cfg.sr)
            rows.append({
                "ok": True,
                "source_path": str(src),
                "output_path": str(out_path),
                "duration_s": float(seg["duration_s"]),
                "bucket": "quarantine",
                **{f"purity_{k}": v for k, v in seg["purity"].items()},
            })

    if accepted_count == 0:
        rows.append({
            "ok": False,
            "source_path": str(src),
            "reason": "no_accepted_segments_after_filter",
        })
    return rows


def save_manifest(rows: List[Dict], out_json: Path, out_csv: Path):
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


class LocalImporter:
    def __init__(self, cfg: DownloaderConfig):
        self.cfg = cfg

    def import_tree(self, src_root: Path, dst_root: Path, label: str, quarantine_dir: Optional[Path]) -> List[Dict]:
        rows = []
        if not src_root or not src_root.exists():
            return rows
        ensure_dir(dst_root)

        files = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        for p in files:
            dst = dst_root / f"local_{safe_slug(p.stem)}.wav"
            item_rows = prepare_audio_file(p, dst, self.cfg, label_hint=label, quarantine_dir=quarantine_dir)
            for row in item_rows:
                row.update({"source_type": "local_import", "label": label})
            rows.extend(item_rows)
        return rows


class YoutubeDownloader:
    def __init__(self, cfg: DownloaderConfig):
        self.cfg = cfg

    def available(self) -> bool:
        try:
            subprocess.run(["yt-dlp", "--version"], check=False, capture_output=True)
            return True
        except Exception:
            return False

    def download_from_list(self, url_list_path: Path, dst_root: Path, label: str, quarantine_dir: Optional[Path]) -> List[Dict]:
        rows = []
        urls = read_text_lines(url_list_path)
        if not urls:
            return rows
        ensure_dir(dst_root)

        if not self.available():
            print("⚠️ yt-dlp not found. Install with: pip install yt-dlp")
            return rows

        for i, url in enumerate(urls[: self.cfg.yt_limit_per_list], 1):
            temp_pattern = dst_root / f"yt_tmp_{label}_{i:05d}.%(ext)s"
            cmd = [
                "yt-dlp",
                "-f", self.cfg.yt_audio_format,
                "--extract-audio",
                "--audio-format", self.cfg.yt_extract_audio_codec,
                "--no-playlist",
                "-o", str(temp_pattern),
                url,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                produced = list(dst_root.glob(f"yt_tmp_{label}_{i:05d}.*"))
                if not produced:
                    rows.append({
                        "source_type": "youtube",
                        "label": label,
                        "source_url": url,
                        "ok": False,
                        "reason": "no_output_found",
                    })
                    continue
                src = produced[0]
                dst = dst_root / f"youtube_{label}_{i:05d}.wav"
                item_rows = prepare_audio_file(src, dst, self.cfg, label_hint=label, quarantine_dir=quarantine_dir)
                for row in item_rows:
                    row.update({"source_type": "youtube", "label": label, "source_url": url})
                rows.extend(item_rows)
            except subprocess.CalledProcessError as e:
                rows.append({
                    "source_type": "youtube",
                    "label": label,
                    "source_url": url,
                    "ok": False,
                    "reason": f"yt_dlp_failed:{e.returncode}",
                })
            finally:
                for p in dst_root.glob(f"yt_tmp_{label}_{i:05d}.*"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
        return rows


class XenoCantoDownloader:
    BASE_URL = "https://xeno-canto.org/api/2/recordings"

    def __init__(self, cfg: DownloaderConfig):
        self.cfg = cfg
        self.sess = make_session(cfg)

    def download_non_drone(self, dst_root: Path, quarantine_dir: Optional[Path]) -> List[Dict]:
        rows = []
        ensure_dir(dst_root)
        total = 0
        for query in self.cfg.xenocanto_queries:
            if total >= self.cfg.xenocanto_max_total:
                break
            try:
                resp = self.sess.get(self.BASE_URL, params={"query": query, "page": 1}, timeout=(8, 30))
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                rows.append({
                    "source_type": "xeno_canto",
                    "label": "non_drone",
                    "query": query,
                    "ok": False,
                    "reason": f"search_failed:{e}",
                })
                continue

            for rec in data.get("recordings", [])[: self.cfg.xenocanto_per_query]:
                if total >= self.cfg.xenocanto_max_total:
                    break
                url = "https:" + rec.get("file", "")
                if not url or url == "https:":
                    continue
                tmp = dst_root / f"xc_tmp_{rec.get('id', total)}.mp3"
                try:
                    r = self.sess.get(url, timeout=(8, 60))
                    r.raise_for_status()
                    tmp.write_bytes(r.content)
                    dst = dst_root / f"xenocanto_{rec.get('id', total)}_{safe_slug(rec.get('en', 'bird'))}.wav"
                    item_rows = prepare_audio_file(tmp, dst, self.cfg, label_hint="non_drone", quarantine_dir=quarantine_dir)
                    for row in item_rows:
                        row.update({
                            "source_type": "xeno_canto",
                            "label": "non_drone",
                            "query": query,
                            "recording_id": rec.get("id", ""),
                            "species": rec.get("en", ""),
                            "source_url": url,
                        })
                    rows.extend(item_rows)
                    total += 1
                except Exception as e:
                    rows.append({
                        "source_type": "xeno_canto",
                        "label": "non_drone",
                        "query": query,
                        "recording_id": rec.get("id", ""),
                        "source_url": url,
                        "ok": False,
                        "reason": f"download_failed:{e}",
                    })
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
            time.sleep(0.2)
        return rows


class FreesoundDownloader:
    BASE_URL = "https://freesound.org/apiv2"

    def __init__(self, cfg: DownloaderConfig):
        self.cfg = cfg
        self.api_key = (cfg.freesound_api_key or os.environ.get("FREESOUND_API_KEY", "")).strip()
        self.sess = make_session(cfg)

    def enabled(self) -> bool:
        return bool(self.cfg.freesound_enabled and self.api_key)

    def _search(self, query: str) -> List[Dict]:
        url = f"{self.BASE_URL}/search/text/"
        params = {
            "query": query,
            "fields": "id,name,previews,license,duration",
            "filter": "duration:[1 TO 1800] type:(wav OR mp3 OR ogg)",
            "page_size": self.cfg.freesound_page_size,
            "sort": "downloads_desc",
            "token": self.api_key,
        }
        resp = self.sess.get(
            url,
            params=params,
            timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read),
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def _download_preview(self, sound: Dict, tmp_path: Path):
        previews = sound.get("previews", {})
        url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not url:
            raise RuntimeError("no_preview_url")
        r = self.sess.get(
            url,
            timeout=(self.cfg.freesound_timeout_connect, self.cfg.freesound_timeout_read),
        )
        r.raise_for_status()
        tmp_path.write_bytes(r.content)

    def download_label(self, queries: Tuple[str, ...], dst_root: Path, label: str, max_total: int, quarantine_dir: Optional[Path]) -> List[Dict]:
        rows = []
        ensure_dir(dst_root)
        total = 0
        for query in queries:
            if total >= max_total:
                break
            try:
                sounds = self._search(query)
            except Exception as e:
                rows.append({
                    "source_type": "freesound",
                    "label": label,
                    "query": query,
                    "ok": False,
                    "reason": f"search_failed:{e}",
                })
                continue

            for sound in sounds:
                if total >= max_total:
                    break
                sid = sound.get("id", total)
                tmp = dst_root / f"fs_tmp_{label}_{sid}.mp3"
                try:
                    self._download_preview(sound, tmp)
                    dst = dst_root / f"freesound_{label}_{sid}_{safe_slug(sound.get('name', 'sound'))[:40]}.wav"
                    item_rows = prepare_audio_file(tmp, dst, self.cfg, label_hint=label, quarantine_dir=quarantine_dir)
                    for row in item_rows:
                        row.update({
                            "source_type": "freesound",
                            "label": label,
                            "query": query,
                            "sound_id": sid,
                            "source_name": sound.get("name", ""),
                        })
                    rows.extend(item_rows)
                    total += 1
                except Exception as e:
                    rows.append({
                        "source_type": "freesound",
                        "label": label,
                        "query": query,
                        "sound_id": sid,
                        "ok": False,
                        "reason": f"download_failed:{e}",
                    })
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
            time.sleep(0.2)
        return rows


class TestDatasetDownloader:
    def __init__(self, cfg: DownloaderConfig):
        self.cfg = cfg
        self.local_importer = LocalImporter(cfg)
        self.youtube = YoutubeDownloader(cfg)
        self.xeno = XenoCantoDownloader(cfg)
        self.freesound = FreesoundDownloader(cfg)

    def run(
        self,
        output_root: Path,
        youtube_drone_list: Optional[Path] = None,
        youtube_non_drone_list: Optional[Path] = None,
        import_local_drone: Optional[Path] = None,
        import_local_non_drone: Optional[Path] = None,
        download_xenocanto: bool = False,
        download_freesound: bool = False,
    ) -> Dict:
        ensure_dir(output_root)
        drone_dir = output_root / "drone"
        non_dir = output_root / "non_drone"
        quarantine_dir = output_root / "quarantine" if self.cfg.quarantine_uncertain else None
        manifests_dir = output_root / "manifests"
        ensure_dir(drone_dir)
        ensure_dir(non_dir)
        ensure_dir(manifests_dir)
        if quarantine_dir is not None:
            ensure_dir(quarantine_dir / "drone")
            ensure_dir(quarantine_dir / "non_drone")

        rows: List[Dict] = []

        if import_local_drone:
            print("📥 Importing local drone files...")
            rows.extend(self.local_importer.import_tree(import_local_drone, drone_dir, "drone", quarantine_dir))

        if import_local_non_drone:
            print("📥 Importing local non-drone files...")
            rows.extend(self.local_importer.import_tree(import_local_non_drone, non_dir, "non_drone", quarantine_dir))

        if self.cfg.youtube_enabled and youtube_drone_list and youtube_drone_list.exists():
            print("📺 Downloading YouTube drone files...")
            rows.extend(self.youtube.download_from_list(youtube_drone_list, drone_dir, "drone", quarantine_dir))

        if self.cfg.youtube_enabled and youtube_non_drone_list and youtube_non_drone_list.exists():
            print("📺 Downloading YouTube non-drone files...")
            rows.extend(self.youtube.download_from_list(youtube_non_drone_list, non_dir, "non_drone", quarantine_dir))

        if download_xenocanto and self.cfg.xenocanto_enabled:
            print("🐦 Downloading Xeno-canto non-drone files...")
            rows.extend(self.xeno.download_non_drone(non_dir, quarantine_dir))

        if download_freesound and self.freesound.enabled():
            print("🌐 Downloading Freesound drone files...")
            rows.extend(self.freesound.download_label(
                self.cfg.freesound_queries_drone, drone_dir, "drone", self.cfg.freesound_max_total_drone, quarantine_dir
            ))
            print("🌐 Downloading Freesound non-drone files...")
            rows.extend(self.freesound.download_label(
                self.cfg.freesound_queries_non_drone, non_dir, "non_drone", self.cfg.freesound_max_total_non_drone, quarantine_dir
            ))
        elif download_freesound and not self.freesound.enabled():
            print("⚠️ Freesound requested but not enabled or API key missing.")

        save_manifest(
            rows,
            manifests_dir / "download_manifest.json",
            manifests_dir / "download_manifest.csv",
        )

        summary = self._summary(rows, output_root)
        (manifests_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary

    @staticmethod
    def _summary(rows: List[Dict], output_root: Path) -> Dict:
        ok_rows = [r for r in rows if r.get("ok")]
        fail_rows = [r for r in rows if not r.get("ok", False)]

        def count_by(key: str) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for r in rows:
                k = str(r.get(key, "unknown"))
                out[k] = out.get(k, 0) + 1
            return out

        quarantine_root = output_root / "quarantine"
        return {
            "output_root": str(output_root),
            "total_manifest_rows": len(rows),
            "successful_rows": len(ok_rows),
            "failed_rows": len(fail_rows),
            "accepted_drone_files": len(list((output_root / "drone").glob("*.wav"))),
            "accepted_non_drone_files": len(list((output_root / "non_drone").glob("*.wav"))),
            "quarantine_drone_files": len(list((quarantine_root / "drone").glob("*.wav"))) if quarantine_root.exists() else 0,
            "quarantine_non_drone_files": len(list((quarantine_root / "non_drone").glob("*.wav"))) if quarantine_root.exists() else 0,
            "sources_breakdown": count_by("source_type"),
            "labels_breakdown": count_by("label"),
        }


def main():
    ap = argparse.ArgumentParser(description="Independent downloader for a held-out drone test dataset with auto filtering")
    ap.add_argument("--output", required=True, help="Output dataset root")
    ap.add_argument("--youtube-drone-list", help="Text file with YouTube URLs for drone test audio")
    ap.add_argument("--youtube-non-drone-list", help="Text file with YouTube URLs for non-drone test audio")
    ap.add_argument("--import-local-drone", help="Folder with your own raw drone test recordings")
    ap.add_argument("--import-local-non-drone", help="Folder with your own raw non-drone test recordings")
    ap.add_argument("--download-xenocanto", action="store_true", help="Download Xeno-canto non-drone audio")
    ap.add_argument("--download-freesound", action="store_true", help="Download Freesound audio")
    ap.add_argument("--enable-freesound", action="store_true", help="Enable Freesound downloader")
    ap.add_argument("--enable-xenocanto", action="store_true", help="Enable Xeno-canto downloader")
    ap.add_argument("--freesound-api-key", default="", help="Freesound API key")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--normalize-audio", action="store_true")
    ap.add_argument("--trim-silence", action="store_true")
    ap.add_argument("--trim-top-db", type=float, default=28.0)
    ap.add_argument("--min-duration-sec", type=float, default=1.0)
    ap.add_argument("--max-duration-sec", type=float, default=1800.0)

    ap.add_argument("--auto-filter", action="store_true", help="Enable automatic filtering")
    ap.add_argument("--no-quarantine", action="store_true", help="Discard uncertain clips instead of quarantining them")
    ap.add_argument("--strong-threshold", type=float, default=0.58)
    ap.add_argument("--weak-threshold", type=float, default=0.44)
    ap.add_argument("--min-drone-ratio", type=float, default=0.70)
    ap.add_argument("--max-speech-ratio", type=float, default=0.15)
    ap.add_argument("--max-silence-ratio", type=float, default=0.20)
    ap.add_argument("--max-uncertain-ratio", type=float, default=0.30)

    args = ap.parse_args()

    cfg = DownloaderConfig(
        sr=args.sr,
        normalize_audio=args.normalize_audio,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        freesound_enabled=args.enable_freesound,
        freesound_api_key=args.freesound_api_key,
        xenocanto_enabled=args.enable_xenocanto,
        auto_filter=args.auto_filter,
        quarantine_uncertain=not args.no_quarantine,
        strong_drone_threshold=args.strong_threshold,
        weak_drone_threshold=args.weak_threshold,
        min_drone_window_ratio=args.min_drone_ratio,
        max_speech_window_ratio=args.max_speech_ratio,
        max_silence_window_ratio=args.max_silence_ratio,
        max_uncertain_window_ratio=args.max_uncertain_ratio,
    )

    downloader = TestDatasetDownloader(cfg)
    downloader.run(
        output_root=Path(args.output),
        youtube_drone_list=Path(args.youtube_drone_list) if args.youtube_drone_list else None,
        youtube_non_drone_list=Path(args.youtube_non_drone_list) if args.youtube_non_drone_list else None,
        import_local_drone=Path(args.import_local_drone) if args.import_local_drone else None,
        import_local_non_drone=Path(args.import_local_non_drone) if args.import_local_non_drone else None,
        download_xenocanto=args.download_xenocanto,
        download_freesound=args.download_freesound,
    )


if __name__ == "__main__":
    main()

# # Usage
# # With Youtube lists
# python test_dataset_downloader.py \
#   --output ./test_dataset \
#   --youtube-drone-list drone_urls.txt \
#   --youtube-non-drone-list non_drone_urls.txt \
#   --enable-xenocanto \
#   --download-xenocanto \
#   --auto-filter

# # stricter drone filtering
# python test_dataset_downloader.py \
#   --output ./test_dataset \
#   --youtube-drone-list drone_urls.txt \
#   --auto-filter \
#   --strong-threshold 0.62 \
#   --weak-threshold 0.48 \
#   --min-drone-ratio 0.78 \
#   --max-speech-ratio 0.08 \
#   --max-silence-ratio 0.12

# # uncertain clips discarded instead of quarantined
# python test_dataset_downloader.py \
#   --output ./test_dataset \
#   --youtube-drone-list drone_urls.txt \
#   --auto-filter \
#   --no-quarantine