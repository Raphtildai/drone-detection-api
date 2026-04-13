# -*- coding: utf-8 -*-
"""
drone_detection/datasets.py
────────────────────────────
All data loading, caching, and management:
  - MelCacheManager              (builds the v15 feature cache on disk)
  - MelCachedDataset             (reads cached .npy files)
  - DetectionDataset             (raw WAV, on-the-fly features)
  - LocalizationDataset          (3-ch WAV sessions + JSON labels)
  - SyntheticLocDataset          (fast synthetic samples)
  - SyntheticLocDatasetV2        (physics-aware, conditioned on real grid)
  - DroneAudioDatasetManager     (download/prepare DroneAudioDataset)
  - UaVirBASEDatasetManager      (download/prepare UaVirBASE, pos-grouped split)
  - DatasetManifest              (thesis reproducibility record)
  - generate_mixed_drone_training_audio()
  - inject_synthetic_det_data()
  - get_det_dataloaders()
  - report_detection_split_counts()
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm.auto import tqdm

from drone_detection.config import AUDIO_EXTS, config as _default_cfg
from drone_detection.audio import (
    AudioProcessor,
    load_audio_any,
    synthesise_drone,
    compute_ipd_features,
    perturb_multichannel,
)
from drone_detection.utils import (
    wrap_angle_deg,
    random_crop_or_loop,
    mix_at_snr,
    normalize_peak,
    db_to_gain,
    augment_waveform,
    ensure_remotezip,
    grouped_split_paths,
    infer_group_id,
    safe_slug,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Label parsing helpers (UaVirBASE)
# ═══════════════════════════════════════════════════════════════════════════════

_AZ_KEYS   = ["azimuth_deg","azimuth","az","Azimuth","AZ","bearing","heading","direction_deg","direction"]
_DIST_KEYS = ["distance_m","distance","dist","Distance","range","range_m","horizontal_distance","slant_range"]
_HT_KEYS   = ["height_m","height","alt","altitude","Height","z","elevation","Elevation","altitude_m","z_m","height_agl"]


def _get_scalar(d: dict, keys: list) -> Optional[float]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                return float(v)
            if isinstance(v, list) and len(v) > 0:
                arr = [float(x) for x in v if x is not None and not math.isnan(float(x))]
                return float(np.median(arr)) if arr else None
    return None


def _cartesian_to_az_dist_ht(obj: dict) -> Optional[Tuple[float, float, float]]:
    def _gv(d, keys):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, (int, float)) and not math.isnan(float(v)): return float(v)
                if isinstance(v, list) and v:
                    vals = [float(x) for x in v if x is not None and not math.isnan(float(x))]
                    return float(np.median(vals)) if vals else None
        return None
    x = _gv(obj, ["x","X","east","East","east_m","pos_x","x_m"])
    y = _gv(obj, ["y","Y","north","North","north_m","pos_y","y_m"])
    z = _gv(obj, ["z","Z","up","Up","up_m","pos_z","z_m","height","height_m","altitude","alt"])
    if x is not None and y is not None and z is not None:
        return (
            wrap_angle_deg(float(math.degrees(math.atan2(y, x)))),
            float(math.sqrt(x**2 + y**2)),
            float(abs(z)),
        )
    return None


def parse_label_json(raw: bytes) -> Optional[Tuple[float, float, float]]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "drone" in data and isinstance(data["drone"], dict):
        drone = data["drone"]
        if isinstance(drone.get("sound_source", ""), str) and "ambient" in drone.get("sound_source", "").lower():
            return None
        try:
            az, di, ht = float(drone["azimuth"]), float(drone["distance"]), float(drone["height"])
            if not (math.isnan(az) or math.isnan(di) or math.isnan(ht)):
                return az, di, ht
        except (KeyError, TypeError, ValueError):
            pass
        r = _cartesian_to_az_dist_ht(drone)
        if r:
            return r
    az = _get_scalar(data, _AZ_KEYS)
    di = _get_scalar(data, _DIST_KEYS)
    ht = _get_scalar(data, _HT_KEYS)
    if az is not None and di is not None and ht is not None:
        return float(az), float(di), float(ht)
    for sk in ["uav","target","labels","annotation","data","position"]:
        if sk not in data: continue
        sub = data[sk]
        if isinstance(sub, dict):
            r = _cartesian_to_az_dist_ht(sub)
            if r: return r
    return None


def _probe_label_schema(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode())
    except Exception:
        return "invalid JSON"
    if isinstance(data, list):
        sample = data[0] if data else {}
        return f"list[{len(data)}] | sample keys={list(sample.keys())[:8] if isinstance(sample,dict) else '?'}"
    if isinstance(data, dict):
        return f"dict | top-level keys={list(data.keys())[:8]}"
    return f"unknown ({type(data).__name__})"


# ═══════════════════════════════════════════════════════════════════════════════
# MelCacheManager — builds the v15 feature cache
# ═══════════════════════════════════════════════════════════════════════════════

class MelCacheManager:
    """
    Converts all processed WAV files to v15 3-channel .npy tensors and stores
    them in cfg.MEL_CACHE_DIR for fast DataLoader access.
    """

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(self.cfg)

    def build(self, force: bool = False) -> None:
        cache_root = self.cfg.MEL_CACHE_DIR
        n_existing = len(list(cache_root.rglob("*.npy")))
        if not force and n_existing > 100:
            print(f"✅ Mel cache already exists ({n_existing} files) — skipping.")
            return
        if force and cache_root.exists():
            shutil.rmtree(str(cache_root))
        print("🎵 Building mel cache from processed WAVs [v15 features] …")
        det_root = self.cfg.PROCESSED_DIR / "detection"
        wavs = []
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                src = det_root / split / label
                if src.exists():
                    wavs.extend((split, label, wav) for wav in src.glob("*.wav"))
        total = 0
        for split, label, wav in tqdm(wavs, desc="Mel cache"):
            dst = cache_root / split / label
            dst.mkdir(parents=True, exist_ok=True)
            out = dst / f"{wav.stem}.npy"
            if out.exists() and not force:
                continue
            try:
                y = self.ap.pad_or_truncate(self.ap.load(wav))
                np.save(str(out), self.ap.feature_stack(y))
                total += 1
            except Exception as e:
                print(f"   ⚠️  {wav.name}: {e}")
        print(f"✅ Mel cache built ({total} new files).")

    def count(self) -> Dict[str, int]:
        out = {}
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                d = self.cfg.MEL_CACHE_DIR / split / label
                out[f"{split}/{label}"] = len(list(d.glob("*.npy"))) if d.exists() else 0
        return out

    def _inject_synthetic(self, cache: Path, force: bool = False) -> None:
        n = self.cfg.SYNTHETIC_DET_SAMPLES
        if n <= 0:
            return
        out = cache / "train" / "drone"
        out.mkdir(parents=True, exist_ok=True)
        if len(list(out.glob("synth_det_*.npy"))) >= n and not force:
            print("  ✅ Synthetic injection already done.")
            return
        print(f"  🔬 Injecting {n} synthetic feature tensors …")
        rng   = np.random.default_rng(self.cfg.SEED)
        cx, cy = self.cfg.ARRAY_CENTER
        ap    = AudioProcessor(self.cfg)
        for i in tqdm(range(n), desc="SynthInject", leave=False):
            r     = rng.uniform(0.5, self.cfg.MAX_LOCALIZATION_DIST)
            theta = rng.uniform(0, 2 * np.pi)
            xy    = [cx + r * np.cos(theta), cy + r * np.sin(theta)]
            fund  = int(rng.choice([80, 90, 100, 110, 120, 130]))
            chs   = synthesise_drone(
                self.cfg.MIC_POSITIONS, xy, fundamental=fund,
                noise_level=float(rng.uniform(0.01, 0.08)),
            )
            y = ap.pad_or_truncate(chs[0])
            np.save(str(out / f"synth_det_{i:06d}.npy"), ap.feature_stack(y))


def inject_synthetic_det_data(cfg=None, force: bool = False) -> None:
    """Inject synthetic drone mel tensors into the detection cache train split."""
    cfg       = cfg or _default_cfg
    cache_dir = cfg.MEL_CACHE_DIR / "train" / "drone"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_samples = cfg.SYNTHETIC_DET_SAMPLES
    if n_samples <= 0:
        return

    # Reduce synthetic count when real custom data is present
    real_custom = len(list((cfg.PROCESSED_DIR / "detection" / "train" / "drone").glob("custom*.wav")))
    if getattr(cfg, "CUSTOM_DATASET_SKIP_SYNTH_IF_PRESENT", False) and real_custom > 0:
        n_samples = min(n_samples, max(0, real_custom // 4))
    if n_samples <= 0:
        print("ℹ️  Synthetic injection skipped (real custom data present).")
        return

    existing = len(list(cache_dir.glob("synth_det_*.npy")))
    if existing >= n_samples and not force:
        print(f"✅ Synthetic detection cache already present ({existing} files).")
        return

    ap      = AudioProcessor(cfg)
    rng     = np.random.default_rng(cfg.SEED + 1)
    r       = rng.uniform(0.3, cfg.MAX_LOCALIZATION_DIST, n_samples)
    theta   = rng.uniform(0, 2 * np.pi, n_samples)
    funds   = rng.choice([80, 90, 100, 110, 120, 130], n_samples)
    noises  = rng.uniform(0.02, 0.10, n_samples)
    cx, cy  = cfg.ARRAY_CENTER
    positions = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)

    print(f"🔬 Injecting {n_samples} synthetic drone mels …")
    for i in tqdm(range(n_samples)):
        out_path = cache_dir / f"synth_det_{i:06d}.npy"
        if out_path.exists() and not force:
            continue
        chs  = synthesise_drone(cfg.MIC_POSITIONS, positions[i],
                                fundamental=int(funds[i]), noise_level=float(noises[i]))
        mono = np.mean(np.stack(chs, axis=0), axis=0)
        np.save(str(out_path), ap.feature_stack(ap.pad_or_truncate(mono)))
    print("✅ Synthetic injection done.")


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Datasets
# ═══════════════════════════════════════════════════════════════════════════════

class MelCachedDataset(Dataset):
    """Reads pre-computed .npy feature tensors from the mel cache."""

    def __init__(self, cache_root: Path, split: str, augment: bool = False) -> None:
        self.augment = augment
        self.files: list[Path] = []
        self.labels: list[int] = []
        for idx, cls in enumerate(["non_drone", "drone"]):
            d = cache_root / split / cls
            if d.exists():
                for f in d.glob("*.npy"):
                    self.files.append(f); self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No cached mels in {cache_root}/{split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(str(self.files[idx])).astype(np.float32)
        if self.augment:
            x = self._spec_augment(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    @staticmethod
    def _spec_augment(x: np.ndarray) -> np.ndarray:
        _, F, T = x.shape
        if random.random() < 0.50:
            x *= random.uniform(0.85, 1.15)
        if random.random() < 0.35:
            t = random.randint(2, max(2, T // 10)); s = random.randint(0, max(0, T - t))
            x[:, :, s:s + t] *= random.uniform(0.0, 0.2)
        if random.random() < 0.25:
            f = random.randint(2, max(2, F // 8)); s = random.randint(0, max(0, F - f))
            x[:, s:s + f, :] *= random.uniform(0.0, 0.2)
        if random.random() < 0.30:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.03
        return x


class DetectionDataset(Dataset):
    """On-the-fly feature extraction from raw WAV files with full waveform augmentation."""

    def __init__(
        self, root: Path, split: str, augment: bool = False, cfg=None
    ) -> None:
        self.ap      = AudioProcessor(cfg or _default_cfg)
        self.cfg     = cfg or _default_cfg
        self.augment = augment
        self.files: list[Path] = []; self.labels: list[int] = []
        for idx, cls in enumerate(["non_drone", "drone"]):
            d = root / split / cls
            if d.exists():
                for f in d.glob("*.wav"):
                    self.files.append(f); self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No files in {root}/{split}")

    def __len__(self) -> int: return len(self.files)

    def __getitem__(self, idx):
        y = self.ap.pad_or_truncate(self.ap.load(self.files[idx]))
        if self.augment:
            y = augment_waveform(y, self.cfg)
        return (
            torch.tensor(self.ap.feature_stack(y), dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class LocalizationDataset(Dataset):
    """Real 3-channel localization sessions (UaVirBASE format)."""

    def __init__(
        self, root: Path, split: str, augment: bool = False, cfg=None
    ) -> None:
        self.cfg     = cfg or _default_cfg
        self.ap      = AudioProcessor(self.cfg)
        self.augment = augment
        self.sessions: list[Tuple[list, Path]] = []
        d = root / split
        if not d.exists():
            raise RuntimeError(f"Localization split not found: {d}")
        for lf in d.glob("*_label.json"):
            sid  = lf.stem.replace("_label", "")
            chs  = [d / f"{sid}_ch{i}.wav" for i in range(3)]
            if all(c.exists() for c in chs):
                self.sessions.append((chs, lf))
        if not self.sessions:
            raise RuntimeError(f"No complete sessions in {d}")

    def __len__(self) -> int: return len(self.sessions)

    def __getitem__(self, idx):
        chs_paths, lf = self.sessions[idx]
        channels = [self.ap.pad_or_truncate(self.ap.load(p)) for p in chs_paths]
        if self.augment:
            channels = perturb_multichannel(channels, self.cfg)

        ipd_cache = lf.parent / (lf.stem.replace("_label", "") + "_ipd.npy")
        if not self.augment and ipd_cache.exists():
            ipd = np.load(str(ipd_cache))
        else:
            ipd = compute_ipd_features(channels, self.cfg)
            if not self.augment:
                try: np.save(str(ipd_cache), ipd)
                except Exception: pass

        mels  = [self.ap.mel(c) for c in channels]
        mel_t = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)
        ipd_t = torch.tensor(ipd, dtype=torch.float32)

        label  = json.loads(lf.read_text())
        az_deg = float(label["azimuth_deg"])
        di_m   = float(label["distance_m"])
        ht_m   = float(label["height_m"])

        if self.augment:
            az_deg = wrap_angle_deg(az_deg + random.gauss(0, 15.0))
            di_m   = max(0.5, di_m + random.gauss(0, 1.5))
            ht_m   = max(0.5, ht_m + random.gauss(0, 1.0))

        az_rad   = math.radians(az_deg)
        max_dist = self.cfg.MAX_LOCALIZATION_DIST
        lbl_t    = torch.tensor([
            math.sin(az_rad), math.cos(az_rad),
            np.clip(di_m / max_dist, 0, 1.5), np.clip(ht_m / max_dist, 0, 1.5),
        ], dtype=torch.float32)
        return mel_t, ipd_t, lbl_t


class SyntheticLocDataset(Dataset):
    """Fast fully-synthetic localisation dataset for training augmentation."""

    def __init__(self, cfg=None, n_samples: int = 500, augment: bool = True) -> None:
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(self.cfg)
        self.n   = n_samples; self.augment = augment
        rng      = np.random.default_rng(self.cfg.SEED)
        r        = rng.uniform(0.3, self.cfg.MAX_LOCALIZATION_DIST, n_samples)
        theta    = rng.uniform(0, 2 * np.pi, n_samples)
        height   = rng.uniform(0.5, 5.0, n_samples)
        cx, cy   = self.cfg.ARRAY_CENTER
        self.positions    = np.stack([cx + r*np.cos(theta), cy + r*np.sin(theta), height], axis=1)
        self.fundamentals = rng.choice([80, 90, 100, 110, 120, 130], n_samples)

    def __len__(self) -> int: return self.n

    def __getitem__(self, idx):
        pos  = self.positions[idx]; fund = int(self.fundamentals[idx])
        chs  = synthesise_drone(self.cfg.MIC_POSITIONS, pos[:2], fundamental=fund,
                                noise_level=0.04 if self.augment else 0.01)
        chs  = [self.ap.pad_or_truncate(c) for c in chs]
        if self.augment and random.random() < 0.3:
            chs = [self.ap.add_noise(c, random.uniform(5, 20)) for c in chs]
        mels  = [self.ap.mel(c) for c in chs]
        mel_t = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)
        ipd_t = torch.tensor(compute_ipd_features(chs, self.cfg), dtype=torch.float32)
        from drone_detection.utils import xy_to_azimuth_deg
        az_deg = xy_to_azimuth_deg(pos[:2], self.cfg.ARRAY_CENTER)
        az_rad = math.radians(az_deg)
        cx, cy = self.cfg.ARRAY_CENTER
        dist_m = math.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2)
        max_d  = self.cfg.MAX_LOCALIZATION_DIST
        lbl_t  = torch.tensor([
            math.sin(az_rad), math.cos(az_rad),
            np.clip(dist_m / max_d, 0, 1.5), np.clip(pos[2] / max_d, 0, 1.5),
        ], dtype=torch.float32)
        return mel_t, ipd_t, lbl_t


# ── Physics-aware synthetic dataset (v2 patch) ────────────────────────────────

# Real UaVirBASE grid constants
_REAL_AZ_DEG  = [0, 45, 90, 135, 180, 225, 270, 315]
_REAL_DIST_M  = [10.0, 20.0]
_REAL_HT_M    = [10.0, 20.0]
_DRONE_FUNDAMENTALS = {
    "dji_mavic":    [87,  174, 261],
    "dji_phantom":  [100, 200, 300],
    "parrot":       [73,  146, 219],
    "generic_quad": [110, 220, 330],
    "hexarotor":    [65,  130, 195],
}


def _synthesise_drone_v2(
    mic_positions: np.ndarray, src_xy: np.ndarray, height_m: float,
    fundamental: int = 100, drone_type: str = "generic_quad",
    wind_speed_ms: float = 0.0, ground_reflection: bool = True,
    noise_level: float = 0.03, duration: float = None, sr: int = None,
) -> List[np.ndarray]:
    """
    Physics-aware drone synthesis:
    adds ground reflection, motor whine, wind noise, and drone-type-specific
    harmonic envelopes on top of the basic synthesise_drone() model.
    """
    import scipy.signal as _ss
    sr  = sr  or _default_cfg.SR
    dur = duration or _default_cfg.TARGET_DURATION
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    c   = _default_cfg.SPEED_OF_SOUND
    src_3d = np.array([src_xy[0], src_xy[1], height_m], dtype=np.float64)
    harmonics = _DRONE_FUNDAMENTALS.get(drone_type, _DRONE_FUNDAMENTALS["generic_quad"])
    channels  = []
    for mic_xy in mic_positions:
        mic_3d = np.array([mic_xy[0], mic_xy[1], 0.0], dtype=np.float64)
        dist   = max(float(np.linalg.norm(src_3d - mic_3d)), 0.01)
        sd     = int(dist / c * sr)
        y      = np.zeros(n, dtype=np.float64)
        for k in range(1, 10):
            amp  = (1.0 / (k**1.4)) * (0.85 + 0.3 * random.random())
            ph   = random.uniform(0, 2 * np.pi)
            jit  = 1.0 + 0.004 * np.sin(2 * np.pi * random.uniform(0.3, 1.2) * t)
            y   += amp * np.sin(2 * np.pi * fundamental * k * jit * t + ph)
        for hf in harmonics[1:3]:
            y += 0.15 * (0.8 + 0.4 * random.random()) * np.sin(2 * np.pi * hf * t + random.uniform(0, 2*np.pi))
        y /= (dist**0.65 + 0.1)
        if ground_reflection and height_m > 0.5:
            src_img = np.array([src_3d[0], src_3d[1], -src_3d[2]])
            dist_img = max(float(np.linalg.norm(src_img - mic_3d)), 0.01)
            sd_img   = int(dist_img / c * sr)
            refl     = np.zeros(n, dtype=np.float64)
            rg       = 0.35 / (dist_img**0.65 + 0.1)
            for k in range(1, 6):
                refl += (1.0 / (k**1.6)) * rg * np.sin(2*np.pi*fundamental*k*t + random.uniform(0, 2*np.pi))
            if sd_img > 0: refl = np.concatenate([np.zeros(sd_img), refl[:-sd_img]])
            y += refl * 0.4
        b, a = _ss.butter(2, [200/(sr/2), 3000/(sr/2)], btype="band")
        y   += _ss.lfilter(b, a, np.random.randn(n)) * noise_level
        if wind_speed_ms > 0.5:
            bw, aw = _ss.butter(1, 150/(sr/2), btype="low")
            y     += _ss.lfilter(bw, aw, np.random.randn(n)) * wind_speed_ms * 0.003
        if sd > 0: y = np.concatenate([np.zeros(sd), y[:-sd]])
        channels.append(y.astype(np.float32))
    return channels


class SyntheticLocDatasetV2(Dataset):
    """
    Physics-aware synthetic localisation dataset conditioned on the real
    UaVirBASE measurement grid. Used in train_localization_v2().

    Parameters
    ----------
    grid_fraction : fraction of samples placed near real grid positions;
                    the rest fill in interpolated / random positions.
    """

    def __init__(
        self, cfg=None, n_samples: int = 2000, grid_fraction: float = 0.55,
        augment: bool = True,
        real_az_values: list = None, real_dist_values: list = None, real_ht_values: list = None,
        az_jitter_deg: float = 18.0, dist_jitter_m: float = 3.0, ht_jitter_m: float = 2.5,
        seed: int = 7777,
    ) -> None:
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(self.cfg)
        self.n   = n_samples; self.augment = augment
        real_az   = real_az_values   or _REAL_AZ_DEG
        real_dist = real_dist_values or _REAL_DIST_M
        real_ht   = real_ht_values   or _REAL_HT_M
        rng = np.random.default_rng(seed)
        n_grid = int(n_samples * grid_fraction); n_free = n_samples - n_grid
        az_grid   = rng.choice(real_az,   n_grid).astype(float) + rng.uniform(-az_jitter_deg,   az_jitter_deg,   n_grid)
        dist_grid = np.clip(rng.choice(real_dist, n_grid).astype(float) + rng.uniform(-dist_jitter_m, dist_jitter_m, n_grid), 1.0, self.cfg.MAX_LOCALIZATION_DIST)
        ht_grid   = np.clip(rng.choice(real_ht,   n_grid).astype(float) + rng.uniform(-ht_jitter_m,   ht_jitter_m,   n_grid), 0.5, self.cfg.MAX_LOCALIZATION_DIST)
        az_free   = rng.uniform(0, 360, n_free)
        dist_free = np.exp(rng.uniform(np.log(2.0), np.log(self.cfg.MAX_LOCALIZATION_DIST), n_free))
        ht_free   = rng.uniform(1.0, 25.0, n_free)
        az_all   = np.concatenate([az_grid, az_free])
        dist_all = np.concatenate([dist_grid, dist_free])
        ht_all   = np.concatenate([ht_grid, ht_free])
        az_rad   = np.radians(az_all)
        cx, cy   = self.cfg.ARRAY_CENTER
        self.positions = np.stack([cx + dist_all*np.cos(az_rad), cy + dist_all*np.sin(az_rad), ht_all], axis=1).astype(np.float32)
        drone_types      = list(_DRONE_FUNDAMENTALS.keys())
        self.drone_types  = [rng.choice(drone_types) for _ in range(n_samples)]
        self.fundamentals = [int(rng.choice(_DRONE_FUNDAMENTALS[dt][0:1])) for dt in self.drone_types]
        self.wind_speeds  = rng.uniform(0.0, 4.0, n_samples).astype(np.float32)
        self.is_grid      = np.array([True]*n_grid + [False]*n_free, dtype=bool)

    def __len__(self) -> int: return self.n

    def __getitem__(self, idx):
        pos   = self.positions[idx]
        chs   = _synthesise_drone_v2(
            self.cfg.MIC_POSITIONS, pos[:2], float(pos[2]),
            fundamental=self.fundamentals[idx], drone_type=self.drone_types[idx],
            wind_speed_ms=float(self.wind_speeds[idx]) if self.augment else 0.0,
            noise_level=0.03 + 0.07*random.random() if self.augment else 0.03,
        )
        chs   = [self.ap.pad_or_truncate(c) for c in chs]
        if self.augment:
            chs = perturb_multichannel(chs, self.cfg)
        ipd_t = torch.tensor(compute_ipd_features(chs, self.cfg), dtype=torch.float32)
        mel_t = torch.tensor(np.stack([self.ap.mel(c) for c in chs], axis=0), dtype=torch.float32)
        cx, cy = self.cfg.ARRAY_CENTER
        az_deg = wrap_angle_deg(float(np.degrees(np.arctan2(pos[1]-cy, pos[0]-cx))))
        dist_m = float(np.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2))
        az_rad = math.radians(az_deg)
        max_d  = self.cfg.MAX_LOCALIZATION_DIST
        lbl_t  = torch.tensor([
            math.sin(az_rad), math.cos(az_rad),
            float(np.clip(dist_m/max_d, 0, 1.5)), float(np.clip(pos[2]/max_d, 0, 1.5)),
        ], dtype=torch.float32)
        return mel_t, ipd_t, lbl_t


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetManifest (for thesis reproducibility)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetManifest:
    """Full record of dataset decisions for thesis appendix."""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    system_version: str = "drone_v15 + loc_patch_v2"
    real_source: str = "UaVirBASE (Zenodo 15391924, Microphone_array.zip)"
    real_total_sessions: int = 0
    real_ambient_sessions: int = 0
    real_drone_sessions: int = 0
    real_unique_positions: int = 0
    real_az_values_deg: list = field(default_factory=list)
    real_dist_values_m: list = field(default_factory=list)
    real_ht_values_m: list = field(default_factory=list)
    real_split_strategy: str = "position-grouped (whole position groups assigned to splits)"
    real_train_sessions: int = 0; real_val_sessions: int = 0; real_test_sessions: int = 0
    synthetic_generator: str = "_synthesise_drone_v2 (blade pass + motor whine + ground reflection + wind noise)"
    synthetic_conditioned_on_grid: bool = True
    synthetic_grid_fraction: float = 0.55
    synthetic_az_jitter_deg: float = 18.0; synthetic_dist_jitter_m: float = 3.0
    synthetic_ht_jitter_m: float = 2.5
    synthetic_drone_types: list = field(default_factory=lambda: list(_DRONE_FUNDAMENTALS.keys()))
    synthetic_n_train: int = 0; synthetic_n_val: int = 0; synthetic_rng_seed: int = 7777
    model_architecture: str = "LocalizationCNN / LocalizationCNNLite"
    loss_function: str = "localization_loss (2*MSE_az + SmoothL1_dist + 0.7*SmoothL1_ht)"
    epochs: int = 0; batch_size: int = 0; learning_rate: float = 0.0; cfg_seed: int = 42
    eval_on_real_only: bool = True
    eval_note: str = (
        "Primary evaluation uses ONLY real held-out sessions. "
        "Synthetic data is used for training augmentation only. "
        "Val/test positions were never seen during training (position-grouped split)."
    )
    limitations: list = field(default_factory=lambda: [
        "Dataset contains only 1 drone type recorded on a single day at a single outdoor location.",
        "All real positions at exactly 10 m or 20 m distance; no near-field (< 10 m) real data.",
        "Only 8 discrete azimuths (multiples of 45°); no real continuous azimuth ground truth.",
        "Multi-drone evaluation is entirely synthetic; no real simultaneous multi-drone recordings exist.",
        "Synthetic acoustic model approximates but does not fully capture real drone acoustics.",
        "Kalman tracker parameters are heuristic, not fitted to real trajectories.",
    ])

    def save(self, path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2)
        print(f"📄 Dataset manifest saved: {path}")
        return str(path)

    def print_thesis_summary(self) -> None:
        print("\n" + "="*70)
        print("  DATASET MANIFEST — paste into thesis Chapter 3 / Appendix")
        print("="*70)
        print(f"  Real data source   : {self.real_source}")
        print(f"  Real sessions      : {self.real_drone_sessions} drone + {self.real_ambient_sessions} ambient")
        print(f"  Real positions     : {self.real_unique_positions} "
              f"({self.real_az_values_deg} az / {self.real_dist_values_m} m dist / {self.real_ht_values_m} m ht)")
        print(f"  Real split         : train={self.real_train_sessions} val={self.real_val_sessions} test={self.real_test_sessions} (position-grouped)")
        print(f"  Synthetic train    : {self.synthetic_n_train}")
        print(f"  Evaluation         : {self.eval_note}")
        print("\n  Limitations acknowledged in thesis:")
        for i, lim in enumerate(self.limitations, 1):
            print(f"    {i}. {lim}")
        print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# Position-grouped split (replaces random split for UaVirBASE)
# ═══════════════════════════════════════════════════════════════════════════════

def position_grouped_split(
    usable: list,
    train_frac: float = 0.625,
    val_frac: float   = 0.187,
    seed: int         = 42,
) -> Dict[str, list]:
    """
    Group sessions by (az, dist, ht) bin and assign WHOLE GROUPS to splits.
    Guarantees val/test positions were never seen during training.
    """
    pos_groups: Dict[tuple, list] = defaultdict(list)
    for item in usable:
        az, di, ht = item[2]
        key = (round(az / 45) * 45 % 360, round(di), round(ht))
        pos_groups[key].append(item)
    groups = sorted(pos_groups.items(), key=lambda x: x[0])
    rng = random.Random(seed); rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(n * train_frac)))
    n_val   = max(1, int(round(n * val_frac)))
    n_test  = max(1, n - n_train - n_val)
    while n_train + n_val + n_test > n: n_test -= 1
    split_map: Dict[str, list] = {"train": [], "val": [], "test": []}
    for i, (pos_key, sessions) in enumerate(groups):
        if   i < n_train:            split_map["train"].extend(sessions)
        elif i < n_train + n_val:    split_map["val"].extend(sessions)
        else:                         split_map["test"].extend(sessions)
    print(f"\n📐 Position-grouped split ({n} unique positions):")
    for sp in ["train", "val", "test"]:
        pos_in = sorted({(round(s[2][0]), round(s[2][1]), round(s[2][2])) for s in split_map[sp]})
        print(f"   {sp:5s}: {len(split_map[sp]):3d} sessions | {len(pos_in):2d} positions")
    return split_map


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset managers (download + process)
# ═══════════════════════════════════════════════════════════════════════════════

class DroneAudioDatasetManager:
    """Download and prepare the DroneAudioDataset binary classification dataset."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(self.cfg)

    def prepare(self) -> bool:
        dest = self.cfg.DRONEDS_RAW
        proc = self.cfg.PROCESSED_DIR / "detection"

        def _count(p): return len(list(p.glob("*.wav"))) if p.exists() else 0
        counts = {
            "train_drone":     _count(proc/"train"/"drone"),
            "train_non_drone": _count(proc/"train"/"non_drone"),
            "val_drone":       _count(proc/"val"/"drone"),
            "val_non_drone":   _count(proc/"val"/"non_drone"),
            "test_drone":      _count(proc/"test"/"drone"),
            "test_non_drone":  _count(proc/"test"/"non_drone"),
        }
        ready = (counts["train_drone"] > 20 and counts["train_non_drone"] > 20
                 and counts["val_drone"] > 0 and counts["val_non_drone"] > 0
                 and counts["test_drone"] > 0 and counts["test_non_drone"] > 0)
        if ready:
            print(f"✅ Detection dataset ready ({sum(counts.values())} files)")
            return True
        print("⚠️ Detection dataset incomplete. Rebuilding …")
        if proc.exists(): shutil.rmtree(proc)
        self._download(dest); self._process(dest, proc)
        return True

    def _download(self, dest: Path) -> None:
        archive = dest / "drone_dataset.zip"; dest.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print("📥 Downloading DroneAudioDataset …")
            urllib.request.urlretrieve(self.cfg.DRONEDS_ZIP_URL, str(archive))
        if any(d.is_dir() and d.name == "Binary_Drone_Audio" for d in dest.rglob("*")):
            return
        print("📦 Extracting …")
        with zipfile.ZipFile(archive) as z: z.extractall(str(dest))

    def _process(self, src: Path, dst: Path) -> None:
        binary = next((d for d in src.rglob("Binary_Drone_Audio") if d.is_dir()), None)
        if binary is None: print("❌ Binary_Drone_Audio not found"); return
        mapping = {"yes_drone":"drone","unknown":"non_drone","Drone":"drone","noDrone":"non_drone"}
        all_files = {"drone": [], "non_drone": []}
        for cls_dir in binary.iterdir():
            if not cls_dir.is_dir(): continue
            label = mapping.get(cls_dir.name, "non_drone")
            all_files[label].extend([f for f in cls_dir.glob("*.*") if f.is_file()])
        import librosa as _lib
        for label, files in all_files.items():
            random.shuffle(files); n = len(files)
            if n == 0: continue
            splits = {"train": files[:int(n*.70)], "val": files[int(n*.70):int(n*.85)], "test": files[int(n*.85):]}
            for split, flist in splits.items():
                out = dst/split/label; out.mkdir(parents=True, exist_ok=True)
                for f in flist:
                    tgt = out/f"{f.stem}.wav"
                    if tgt.exists(): continue
                    try:
                        y, _ = _lib.load(str(f), sr=self.cfg.SR, mono=True)
                        sf.write(str(tgt), y, self.cfg.SR)
                    except Exception as e: print(f"   ⚠️  {f.name}: {e}")
        print("✅ Detection dataset processed")


class UaVirBASEDatasetManager:
    """
    Download and prepare the UaVirBASE localization dataset.
    Uses position-grouped splits to avoid position leakage between
    training and evaluation.
    """

    AUDIO_FILENAME = "output.wav"
    LABEL_FILENAME = "label.json"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(self.cfg)

    def prepare(self) -> bool:
        proc = self.cfg.PROCESSED_DIR / "localization"
        if (proc/"train").exists():
            n = len(list(proc.rglob("*_label.json")))
            if n > 20:
                print(f"✅ Localization dataset ready ({n} sessions)"); return True
        url = self.cfg.UAVIRBASE_ZIP_URL
        if url is None:
            print("⚠️  UAVIRBASE_ZIP_URL is None → using synthetic fallback.")
            self._write_synthetic(proc); return True
        if getattr(self.cfg, "UAVIRBASE_FULL", False):
            print("📥 FULL download mode …")
            try:
                self._download_full(self.cfg.UAVIRBASE_RAW)
                self._process_local(self.cfg.UAVIRBASE_RAW, proc)
            except Exception as e:
                print(f"⚠️  Full download failed ({e}) → synthetic fallback.")
                self._write_synthetic(proc)
        else:
            n_sess = getattr(self.cfg, "UAVIRBASE_N_SESSIONS", 500)
            print(f"📥 PARTIAL download: {n_sess} sessions via remotezip …")
            try:
                ensure_remotezip()
                self._download_partial(url, proc, n_sess)
            except Exception as e:
                print(f"⚠️  Partial download failed ({e}) → synthetic fallback.")
                self._write_synthetic(proc)
        return True

    def _download_partial(self, url: str, proc: Path, n_sessions: int) -> None:
        """Download sessions and assign them via position-grouped split."""
        from remotezip import RemoteZip
        RZ_KW = {"initial_buffer_size": 64*1024*1024}
        AUDIO_CANDIDATES = {"output.wav", "audio.wav"}
        LABEL_CANDIDATES = {"label.json", "annotation.json"}

        print("   Reading remote ZIP central directory …")
        with RemoteZip(url, **RZ_KW) as rz:
            all_names = rz.namelist()

        norm_names  = [n.replace("\\", "/") for n in all_names]
        session_map: Dict[str, list] = {}
        for path in norm_names:
            p = Path(path)
            if len(p.parts) < 2: continue
            session_map.setdefault(str(Path(*p.parts[:-1])), []).append(p.name)

        paired = []
        for session_dir, files in session_map.items():
            a = next((f for f in files if f in AUDIO_CANDIDATES), None)
            l = next((f for f in files if f in LABEL_CANDIDATES), None)
            if a and l:
                paired.append((f"{session_dir}/{a}", f"{session_dir}/{l}"))

        print(f"   Candidate paired sessions: {len(paired)}")
        if not paired:
            raise RuntimeError("No paired audio/json sessions found.")

        print("   Validating labels …")
        usable: list = []; ambient_skipped = parse_failed = 0
        with RemoteZip(url, **RZ_KW) as rz:
            for audio_path, label_path in tqdm(paired, desc="Validating"):
                try:
                    raw    = rz.read(label_path)
                    parsed = parse_label_json(raw)
                    if parsed is None:
                        try:
                            content = json.loads(raw.decode("utf-8"))
                            if "ambient" in content.get("drone", {}).get("sound_source", "").lower():
                                ambient_skipped += 1; continue
                        except Exception: pass
                        parse_failed += 1; continue
                    usable.append((audio_path, label_path, parsed))
                except Exception: parse_failed += 1

        print(f"   Usable: {len(usable)} | Ambient: {ambient_skipped} | Failed: {parse_failed}")
        if not usable: raise RuntimeError("No usable sessions after validation.")

        # ── position-grouped split ────────────────────────────────────────────
        split_map = position_grouped_split(usable, seed=self.cfg.SEED)
        for s in ["train", "val", "test"]: (proc / s).mkdir(parents=True, exist_ok=True)

        downloaded = failed = 0
        all_items  = ([(s, "train") for s in split_map["train"]] +
                      [(s, "val")   for s in split_map["val"]]   +
                      [(s, "test")  for s in split_map["test"]])

        with RemoteZip(url, **RZ_KW) as rz:
            for (audio_path, label_path, parsed), split in tqdm(all_items, desc="Downloading"):
                sid = Path(audio_path).parent.name
                try:
                    az, di, ht    = parsed
                    audio_bytes   = rz.read(audio_path)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(audio_bytes); tmp_path = tf.name
                    try:
                        channels = self.ap.load_channels(tmp_path, self.cfg.UAVIRBASE_MIC_INDICES)
                    finally:
                        if os.path.exists(tmp_path): os.unlink(tmp_path)
                    out = proc / split
                    for i, ch in enumerate(channels):
                        sf.write(str(out/f"{sid}_ch{i}.wav"), self.ap.pad_or_truncate(ch), self.cfg.SR)
                    (out/f"{sid}_label.json").write_text(
                        json.dumps({"azimuth_deg":az,"distance_m":di,"height_m":ht,"source":"real"})
                    )
                    downloaded += 1
                except Exception as e:
                    failed += 1; print(f"   ⚠️  {sid}: {e}")

        print(f"\n✅ Downloaded {downloaded} sessions ({failed} failed)")
        for split in ["train","val","test"]:
            print(f"   {split}: {len(list((proc/split).glob('*_label.json')))} sessions")
        if downloaded == 0:
            raise RuntimeError("Zero sessions saved.")

    def _download_full(self, dest: Path) -> None:
        archive = dest/"uavirbase.zip"
        if not archive.exists():
            urllib.request.urlretrieve(self.cfg.UAVIRBASE_ZIP_URL, str(archive))
        with zipfile.ZipFile(archive) as z: z.extractall(str(dest))
        archive.unlink()

    def _process_local(self, src: Path, dst: Path) -> None:
        sessions = [p.parent for p in src.rglob(self.LABEL_FILENAME)] or \
                   [p.parent for p in src.rglob("annotation.json")]
        if not sessions: print("⚠️  No label files found."); return
        random.shuffle(sessions); n = len(sessions)
        splits = {"train": sessions[:int(n*.70)], "val": sessions[int(n*.70):int(n*.85)], "test": sessions[int(n*.85):]}
        for split, sl in splits.items():
            out = dst/split; out.mkdir(parents=True, exist_ok=True)
            for sess in tqdm(sl, desc=split):
                audio_file = next((sess/x for x in [self.AUDIO_FILENAME, "audio.wav"] if (sess/x).exists()), None)
                if audio_file is None:
                    wavs = list(sess.glob("*.wav"))
                    if not wavs: continue
                    audio_file = wavs[0]
                label_file = next((sess/x for x in [self.LABEL_FILENAME, "annotation.json"] if (sess/x).exists()), None)
                if label_file is None: continue
                try:
                    parsed = parse_label_json(label_file.read_bytes())
                    if parsed is None: continue
                    az, di, ht = parsed
                    channels = self.ap.load_channels(audio_file, self.cfg.UAVIRBASE_MIC_INDICES)
                    for i, ch in enumerate(channels):
                        sf.write(str(out/f"{sess.name}_ch{i}.wav"), self.ap.pad_or_truncate(ch), self.cfg.SR)
                    (out/f"{sess.name}_label.json").write_text(
                        json.dumps({"azimuth_deg":az,"distance_m":di,"height_m":ht,"source":"real"})
                    )
                except Exception as e: print(f"   ⚠️  {sess.name}: {e}")
        print("✅ Full dataset processed")

    def _write_synthetic(self, proc: Path, n_total: int = 2400) -> None:
        print(f"🔬 Generating {n_total} synthetic localization samples …")
        rng    = np.random.default_rng(self.cfg.SEED)
        dists  = np.exp(rng.uniform(np.log(0.3), np.log(self.cfg.MAX_LOCALIZATION_DIST), n_total))
        az_rad = rng.uniform(-np.pi, np.pi, n_total)
        heights = rng.uniform(0.5, 5.0, n_total)
        funds  = rng.choice([80, 90, 100, 110, 120, 130], n_total)
        noises = rng.uniform(0.01, 0.08, n_total)
        cx, cy = self.cfg.ARRAY_CENTER
        xs = cx + dists*np.cos(az_rad); ys = cy + dists*np.sin(az_rad)
        idx_tr = int(n_total*.70); idx_val = int(n_total*.85)
        splits = ["train"]*idx_tr + ["val"]*(idx_val-idx_tr) + ["test"]*(n_total-idx_val)
        for s in ["train","val","test"]: (proc/s).mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n_total), desc="Synthetic loc"):
            sid  = f"synth_{i:06d}"
            chs  = synthesise_drone(self.cfg.MIC_POSITIONS, [xs[i],ys[i]],
                                    fundamental=int(funds[i]), noise_level=float(noises[i]))
            out  = proc/splits[i]
            for j, ch in enumerate(chs):
                sf.write(str(out/f"{sid}_ch{j}.wav"), self.ap.pad_or_truncate(ch), self.cfg.SR)
            (out/f"{sid}_label.json").write_text(
                json.dumps({"azimuth_deg": float(np.degrees(az_rad[i])),
                            "distance_m":  float(dists[i]),
                            "height_m":    float(heights[i])})
            )
        print(f"✅ Synthetic fallback: {len(list(proc.rglob('*_label.json')))} sessions.")


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed-drone training audio generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mixed_drone_training_audio(cfg=None, force: bool = False) -> None:
    """
    Generate mixed (drone + background) training audio for the detection model.
    Sources drone audio from the clean train split and custom-builder clips.
    """
    from drone_detection.audio import collect_background_pool
    cfg = cfg or _default_cfg
    ap  = AudioProcessor(cfg)
    train_dir = cfg.PROCESSED_DIR/"detection"/"train"/"drone"
    val_dir   = cfg.PROCESSED_DIR/"detection"/"val"/"drone"
    if not train_dir.exists():
        print("⚠️ No train/drone directory found."); return

    existing = list(train_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav")) + \
               list(val_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"))
    if len(existing) >= cfg.MIXED_DRONE_SAMPLES and not force:
        print(f"✅ Mixed drone audio already exists ({len(existing)} files) — skipping."); return

    if force:
        for d in [train_dir, val_dir]:
            if d.exists():
                for f in d.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"):
                    try: f.unlink()
                    except Exception: pass

    drone_files = [f for f in train_dir.glob("*.wav") if not f.stem.startswith(cfg.MIX_CACHE_PREFIX)]
    custom_clean = Path(getattr(cfg, "CUSTOM_DATASET_IMPORTED_ROOT", "")) / "clean_drone_segments"
    if getattr(cfg, "CUSTOM_DATASET_INCLUDE_CLEAN_IN_MIXING", False) and custom_clean.exists():
        drone_files.extend(f for f in custom_clean.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    # de-duplicate
    seen = set(); drone_files = [f for f in drone_files if str(f) not in seen and not seen.add(str(f))]

    if not drone_files:
        print("⚠️ No clean drone WAVs found for mixing."); return

    bg_pool = collect_background_pool(cfg)
    usable_bg = [k for k in cfg.MIX_BACKGROUND_LABELS if len(bg_pool.get(k, [])) > 0]
    if not usable_bg:
        print("⚠️ No background pool."); return

    n_total = int(cfg.MIXED_DRONE_SAMPLES)
    n_val   = int(n_total * cfg.MIXED_DRONE_VAL_FRAC)
    n_train = n_total - n_val
    print(f"🎛️ Generating mixed audio: total={n_total} train={n_train} val={n_val}")

    for i in tqdm(range(n_total), desc="Mixing drone+background"):
        split   = "val" if i < n_val else "train"
        out_dir = val_dir if split == "val" else train_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        drone_p = random.choice(drone_files)
        bg_lbl  = random.choice(usable_bg)
        bg_p    = random.choice(bg_pool[bg_lbl])
        out_p   = out_dir / f"{cfg.MIX_CACHE_PREFIX}_{i:06d}_{bg_lbl}_{safe_slug(drone_p.stem)}.wav"
        if out_p.exists() and not force: continue
        try:
            dy = ap.pad_or_truncate(load_audio_any(drone_p, cfg.SR))
            by = random_crop_or_loop(load_audio_any(bg_p, cfg.SR), len(dy))
            dy = np.clip(dy * db_to_gain(random.uniform(*cfg.MIX_GAIN_RANGE_DB)), -1, 1).astype(np.float32)
            by = np.clip(by * db_to_gain(random.uniform(*cfg.MIX_BG_GAIN_RANGE_DB)), -1, 1).astype(np.float32)
            mixed = mix_at_snr(dy, by, random.uniform(*cfg.MIX_SNR_DB_RANGE))
            if random.random() < 0.35:
                mixed = np.roll(mixed, random.randint(0, len(mixed)//10)).astype(np.float32)
            if random.random() < 0.40:
                noise = np.random.randn(len(mixed)).astype(np.float32)
                noise /= np.max(np.abs(noise)) + 1e-8
                mixed = normalize_peak(mixed + 0.01 * noise)
            sf.write(str(out_p), mixed, cfg.SR)
        except Exception as e:
            print(f"   ⚠️ Failed mix {i}: {e}")

    made = (len(list(train_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"))) +
            len(list(val_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"))))
    print(f"✅ Mixed drone generation complete ({made} files).")


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader builder + reporting
# ═══════════════════════════════════════════════════════════════════════════════

def get_det_dataloaders(cfg=None):
    cfg   = cfg or _default_cfg
    cache = cfg.MEL_CACHE_DIR
    tr_ds = MelCachedDataset(cache, "train", augment=True)
    va_ds = MelCachedDataset(cache, "val",   augment=False)
    try:
        te_ds = MelCachedDataset(cache, "test", augment=False)
    except RuntimeError:
        print("⚠️ No mel cache test split — using val as fallback.")
        te_ds = MelCachedDataset(cache, "val", augment=False)

    labels  = np.array(tr_ds.labels)
    counts  = np.bincount(labels); counts[counts == 0] = 1
    weights = (1.0 / counts)[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def _col(batch): xs, ys = zip(*batch); return torch.stack(xs), torch.stack(ys)

    bs  = cfg.BATCH_SIZE
    nw  = min(4, os.cpu_count() or 2)
    pin = cfg.DEVICE == "cuda"
    kw  = dict(collate_fn=_col, num_workers=nw, pin_memory=pin,
               persistent_workers=(nw > 0),
               prefetch_factor=2 if nw > 0 else None)

    tr_l = DataLoader(tr_ds, batch_size=bs, sampler=sampler, **kw)
    va_l = DataLoader(va_ds, batch_size=bs, shuffle=False,   **kw)
    te_l = DataLoader(te_ds, batch_size=bs, shuffle=False,   **kw)
    return tr_l, va_l, te_l


def report_detection_split_counts(cfg=None) -> None:
    cfg  = cfg or _default_cfg
    root = cfg.PROCESSED_DIR / "detection"
    print("\n📊 Detection WAV split counts")
    for split in ["train", "val", "test"]:
        for label in ["drone", "non_drone"]:
            d = root / split / label
            n = len(list(d.glob("*.wav"))) if d.exists() else 0
            print(f"   {split:5s} / {label:10s}: {n}")


def audit_localization_labels(cfg=None) -> None:
    cfg  = cfg or _default_cfg
    proc = cfg.PROCESSED_DIR / "localization"
    all_labels = []
    for split in ["train", "val", "test"]:
        for lf in (proc/split).glob("*_label.json") if (proc/split).exists() else []:
            d = json.loads(lf.read_text())
            all_labels.append((d["azimuth_deg"], d["distance_m"], d["height_m"]))
    if not all_labels: print("No labels found."); return
    az, dist, ht = zip(*all_labels)
    print(f"  Sessions : {len(all_labels)}")
    print(f"  Azimuth  : min={min(az):.1f}°  max={max(az):.1f}°  mean={np.mean(az):.1f}°")
    print(f"  Distance : min={min(dist):.2f}m  max={max(dist):.2f}m  mean={np.mean(dist):.2f}m")
    print(f"  Height   : min={min(ht):.2f}m  max={max(ht):.2f}m  mean={np.mean(ht):.2f}m")