# -*- coding: utf-8 -*-
"""
datasets.py
───────────
All PyTorch Dataset classes, cache builders, and data-manager utilities.

Contents
────────
MelCacheManager           — builds on-disk .npy feature cache (v15 features)
MelCachedDataset          — loads from cache with spec-augment
DetectionDataset          — loads raw WAVs for detection (with augmentation)
LocalizationDataset       — loads 3-mic sessions for localization
SyntheticLocDataset       — fast fully-synthetic localization dataset
SyntheticLocDatasetV2     — grid-conditioned, physics-aware (loc patch v1)
DroneAudioDatasetManager  — downloads & prepares the DroneAudioDataset
UaVirBASEDatasetManager   — downloads & prepares UaVirBASE (position-grouped)

Dataloader factory
──────────────────
get_det_dataloaders()     — returns (train, val, test) DataLoader objects
"""

import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import Config, config, AUDIO_EXTS
from .audio_processing import AudioProcessor, synthesise_drone
from .utils import (
    compute_ipd_features,
    grouped_split_paths,
    infer_group_id,
    load_audio_any,
    normalize_peak,
    wrap_angle_deg,
    _set_seed,
    _ensure_remotezip,
)

# JSON keys searched when parsing UaVirBASE label files
_AZ_KEYS   = ["azimuth_deg","azimuth","az","Azimuth","AZ","bearing","heading","direction_deg","direction"]
_DIST_KEYS = ["distance_m","distance","dist","Distance","range","range_m","horizontal_distance","slant_range"]
_HT_KEYS   = ["height_m","height","alt","altitude","Height","z","elevation","Elevation","altitude_m","z_m","height_agl"]

# Real UaVirBASE measurement grid (used by SyntheticLocDatasetV2)
_REAL_AZ_DEG   = [0, 45, 90, 135, 180, 225, 270, 315]
_REAL_DIST_M   = [10.0, 20.0]
_REAL_HT_M     = [10.0, 20.0]
_DRONE_FUNDAMENTALS = {
    "dji_mavic":   [87,  174, 261],
    "dji_phantom": [100, 200, 300],
    "parrot":      [73,  146, 219],
    "generic_quad":[110, 220, 330],
    "hexarotor":   [65,  130, 195],
}


# ══════════════════════════════════════════════════════════════════════════════
# Mel cache
# ══════════════════════════════════════════════════════════════════════════════

class MelCacheManager:
    """
    Builds and manages an on-disk cache of pre-computed feature tensors.

    Each WAV is converted to a 3-channel feature stack (log-mel / PCEN /
    delta-mel) and saved as a .npy file with the same relative path under
    cfg.MEL_CACHE_DIR.  Training then reads from cache instead of
    re-computing features every epoch.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or config
        self.ap  = AudioProcessor(cfg)

    def build(self, force: bool = False):
        """
        Build the complete feature cache from all processed WAV splits.

        Parameters
        ──────────
        force : if True, delete existing cache first and rebuild from scratch
        """
        cache_root = self.cfg.MEL_CACHE_DIR
        n_existing = len(list(cache_root.rglob("*.npy")))
        if not force and n_existing > 100:
            print(f"✅ Mel cache already exists ({n_existing} files) — skipping.")
            return
        if force and cache_root.exists():
            shutil.rmtree(str(cache_root))

        print("🎵 Building mel cache from processed WAVs [v15 features] …")
        det_root = self.cfg.PROCESSED_DIR / "detection"
        wavs = [
            (split, label, wav)
            for split in ["train", "val", "test"]
            for label in ["drone", "non_drone"]
            for wav   in (det_root / split / label).glob("*.wav")
            if (det_root / split / label).exists()
        ]
        total = 0
        from tqdm.auto import tqdm
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
        """Return a dict mapping 'split/label' → number of cached files."""
        out = {}
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                d = self.cfg.MEL_CACHE_DIR / split / label
                out[f"{split}/{label}"] = (
                    len(list(d.glob("*.npy"))) if d.exists() else 0
                )
        return out

    def inject_synthetic(self, force: bool = False):
        """
        Inject synthetic drone mel tensors into the training cache.
        Respects cfg.CUSTOM_DATASET_SKIP_SYNTH_IF_PRESENT.
        """
        cache_dir   = self.cfg.MEL_CACHE_DIR / "train" / "drone"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Reduce synthetic count when real custom clips are present
        real_custom = len(list(
            (self.cfg.PROCESSED_DIR / "detection" / "train" / "drone")
            .glob("custom*.wav")
        )) if (self.cfg.PROCESSED_DIR / "detection" / "train" / "drone").exists() else 0

        n_samples = int(self.cfg.SYNTHETIC_DET_SAMPLES)
        if getattr(self.cfg, "CUSTOM_DATASET_SKIP_SYNTH_IF_PRESENT", True) and real_custom > 0:
            n_samples = min(n_samples, max(0, real_custom // 4))

        existing = len(list(cache_dir.glob("synth_det_*.npy")))
        if existing >= n_samples and not force:
            print(f"✅ Synthetic detection cache already present ({existing} files) — skipping.")
            return
        if n_samples <= 0:
            print("ℹ️  Synthetic injection disabled (custom real data present).")
            return

        rng = np.random.default_rng(self.cfg.SEED + 1)
        r      = rng.uniform(0.3, self.cfg.MAX_LOCALIZATION_DIST, n_samples)
        theta  = rng.uniform(0, 2 * np.pi, n_samples)
        funds  = rng.choice([80, 90, 100, 110, 120, 130], n_samples)
        noises = rng.uniform(0.02, 0.10, n_samples)
        cx, cy = self.cfg.ARRAY_CENTER
        positions = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)

        print(f"🔬 Injecting {n_samples} synthetic drone tensors …")
        from tqdm.auto import tqdm
        for i in tqdm(range(n_samples)):
            out_path = cache_dir / f"synth_det_{i:06d}.npy"
            if out_path.exists() and not force:
                continue
            chs  = synthesise_drone(
                self.cfg.MIC_POSITIONS, positions[i],
                fundamental=int(funds[i]), noise_level=float(noises[i]),
            )
            mono = np.mean(np.stack(chs, axis=0), axis=0)
            np.save(str(out_path), self.ap.feature_stack(self.ap.pad_or_truncate(mono)))
        print("✅ Synthetic injection done.")


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch datasets
# ══════════════════════════════════════════════════════════════════════════════

class MelCachedDataset(Dataset):
    """
    Loads pre-computed .npy feature tensors for the detection task.

    Augmentation (when augment=True)
    ─────────────────────────────────
    - Random gain scaling
    - Time-mask
    - Frequency-mask
    - Low-amplitude Gaussian noise
    """

    def __init__(self, cache_root: Path, split: str, augment: bool = False):
        self.augment = augment
        self.files:  List[Path] = []
        self.labels: List[int]  = []
        for idx, cls in enumerate(["non_drone", "drone"]):
            d = cache_root / split / cls
            if d.exists():
                for f in d.glob("*.npy"):
                    self.files.append(f)
                    self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No cached features in {cache_root}/{split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        x = np.load(str(self.files[idx])).astype(np.float32)
        if self.augment:
            x = self._spec_augment(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    @staticmethod
    def _spec_augment(x: np.ndarray) -> np.ndarray:
        _, F, T = x.shape
        if random.random() < 0.5:
            x *= random.uniform(0.85, 1.15)
        if random.random() < 0.35:
            tw = random.randint(2, max(2, T // 10))
            ts = random.randint(0, max(0, T - tw))
            x[:, :, ts : ts + tw] *= random.uniform(0.0, 0.2)
        if random.random() < 0.25:
            fw = random.randint(2, max(2, F // 8))
            fs = random.randint(0, max(0, F - fw))
            x[:, fs : fs + fw, :] *= random.uniform(0.0, 0.2)
        if random.random() < 0.30:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.03
        return x


class DetectionDataset(Dataset):
    """
    Loads raw WAV files for detection training.
    Applies waveform augmentation before feature extraction.
    """

    def __init__(
        self,
        root:    Path,
        split:   str,
        augment: bool = False,
        cfg:     Optional[Config] = None,
    ):
        self.ap      = AudioProcessor(cfg or config)
        self.cfg     = cfg or config
        self.augment = augment
        self.files:  List[Path] = []
        self.labels: List[int]  = []
        for idx, cls in enumerate(["non_drone", "drone"]):
            d = root / split / cls
            if d.exists():
                for f in d.glob("*.wav"):
                    self.files.append(f)
                    self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No WAV files in {root}/{split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        from .utils import augment_waveform
        y = self.ap.pad_or_truncate(self.ap.load(self.files[idx]))
        if self.augment:
            y = augment_waveform(y, self.cfg)
        feat = self.ap.feature_stack(y)
        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class LocalizationDataset(Dataset):
    """
    Loads 3-microphone recording sessions for localization training.
    Each session consists of 3 WAV files (_ch0, _ch1, _ch2) + a label JSON.
    """

    def __init__(
        self,
        root:    Path,
        split:   str,
        augment: bool = False,
        cfg:     Optional[Config] = None,
    ):
        self.cfg     = cfg or config
        self.ap      = AudioProcessor(self.cfg)
        self.augment = augment
        self.sessions: List[Tuple[List[Path], Path]] = []
        d = root / split
        if not d.exists():
            raise RuntimeError(f"Localization split not found: {d}")
        for lf in d.glob("*_label.json"):
            sid = lf.stem.replace("_label", "")
            chs = [d / f"{sid}_ch{i}.wav" for i in range(3)]
            if all(c.exists() for c in chs):
                self.sessions.append((chs, lf))
        if not self.sessions:
            raise RuntimeError(f"No complete sessions in {d}")

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int):
        from .utils import perturb_multichannel
        chs_paths, lf = self.sessions[idx]
        channels = [self.ap.pad_or_truncate(self.ap.load(p)) for p in chs_paths]
        if self.augment:
            channels = perturb_multichannel(channels, self.cfg)

        # IPD features (cached on disk for non-augmented samples)
        ipd_cache = lf.parent / (lf.stem.replace("_label", "") + "_ipd.npy")
        if (not self.augment) and ipd_cache.exists():
            ipd = np.load(str(ipd_cache))
        else:
            ipd = compute_ipd_features(channels, self.cfg)
            if not self.augment:
                try:
                    np.save(str(ipd_cache), ipd)
                except Exception:
                    pass

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
        lbl_t    = torch.tensor(
            [math.sin(az_rad), math.cos(az_rad),
             np.clip(di_m / max_dist, 0, 1.5),
             np.clip(ht_m / max_dist, 0, 1.5)],
            dtype=torch.float32,
        )
        return mel_t, ipd_t, lbl_t


class SyntheticLocDataset(Dataset):
    """
    Fully-synthetic localization dataset using synthesise_drone().
    Useful as a training supplement when real data is scarce.
    """

    def __init__(
        self,
        cfg:         Optional[Config] = None,
        n_samples:   int  = 500,
        augment:     bool = True,
    ):
        self.cfg     = cfg or config
        self.ap      = AudioProcessor(self.cfg)
        self.n       = n_samples
        self.augment = augment
        rng     = np.random.default_rng(self.cfg.SEED)
        r       = rng.uniform(0.3, self.cfg.MAX_LOCALIZATION_DIST, n_samples)
        theta   = rng.uniform(0, 2 * np.pi, n_samples)
        height  = rng.uniform(0.5, 5.0, n_samples)
        cx, cy  = self.cfg.ARRAY_CENTER
        self.positions   = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta), height], axis=1)
        self.fundamentals = rng.choice([80, 90, 100, 110, 120, 130], n_samples)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        from .utils import perturb_multichannel
        pos  = self.positions[idx]
        fund = int(self.fundamentals[idx])
        chs  = synthesise_drone(self.cfg.MIC_POSITIONS, pos[:2], fundamental=fund,
                                noise_level=0.04 if self.augment else 0.01)
        chs  = [self.ap.pad_or_truncate(c) for c in chs]
        if self.augment and random.random() < 0.3:
            chs = perturb_multichannel(chs, self.cfg)

        mels  = [self.ap.mel(c) for c in chs]
        mel_t = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)
        ipd_t = torch.tensor(compute_ipd_features(chs, self.cfg), dtype=torch.float32)

        cx, cy   = self.cfg.ARRAY_CENTER
        az_deg   = wrap_angle_deg(float(np.degrees(np.arctan2(pos[1] - cy, pos[0] - cx))))
        dist_m   = float(math.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2))
        az_rad   = math.radians(az_deg)
        max_dist = self.cfg.MAX_LOCALIZATION_DIST
        lbl_t    = torch.tensor(
            [math.sin(az_rad), math.cos(az_rad),
             float(np.clip(dist_m / max_dist, 0, 1.5)),
             float(np.clip(pos[2] / max_dist, 0, 1.5))],
            dtype=torch.float32,
        )
        return mel_t, ipd_t, lbl_t


class SyntheticLocDatasetV2(Dataset):
    """
    Physics-aware synthetic localization dataset (loc patch v1).

    Improvements over SyntheticLocDataset
    ──────────────────────────────────────
    - Grid-conditioned sampling: majority of samples are placed near real
      UaVirBASE measurement positions (±jitter), so the model generalises
      better to the real evaluation grid.
    - Free-interpolation samples fill angular / range gaps.
    - Physics-aware synthesis: blade-pass harmonics, motor whine, ground
      reflection, wind noise (via _synthesise_drone_v2).
    """

    def __init__(
        self,
        cfg:              Optional[Config] = None,
        n_samples:        int   = 2000,
        grid_fraction:    float = 0.55,
        augment:          bool  = True,
        real_az_values:   Optional[List[float]] = None,
        real_dist_values: Optional[List[float]] = None,
        real_ht_values:   Optional[List[float]] = None,
        az_jitter_deg:    float = 18.0,
        dist_jitter_m:    float = 3.0,
        ht_jitter_m:      float = 2.5,
        seed:             int   = 7777,
    ):
        self.cfg     = cfg or config
        self.ap      = AudioProcessor(self.cfg)
        self.n       = n_samples
        self.augment = augment

        real_az   = real_az_values   or _REAL_AZ_DEG
        real_dist = real_dist_values or _REAL_DIST_M
        real_ht   = real_ht_values   or _REAL_HT_M

        rng     = np.random.default_rng(seed)
        n_grid  = int(n_samples * grid_fraction)
        n_free  = n_samples - n_grid

        # Grid-conditioned samples
        az_g    = rng.choice(real_az,   n_grid).astype(float) + rng.uniform(-az_jitter_deg,   az_jitter_deg,   n_grid)
        dist_g  = np.clip(rng.choice(real_dist, n_grid).astype(float) + rng.uniform(-dist_jitter_m, dist_jitter_m, n_grid), 1.0, self.cfg.MAX_LOCALIZATION_DIST)
        ht_g    = np.clip(rng.choice(real_ht,   n_grid).astype(float) + rng.uniform(-ht_jitter_m,   ht_jitter_m,   n_grid), 0.5, self.cfg.MAX_LOCALIZATION_DIST)

        # Free / interpolated samples
        az_f    = rng.uniform(0, 360, n_free)
        dist_f  = np.exp(rng.uniform(np.log(2.0), np.log(self.cfg.MAX_LOCALIZATION_DIST), n_free))
        ht_f    = rng.uniform(1.0, 25.0, n_free)

        az_all   = np.concatenate([az_g,   az_f])
        dist_all = np.concatenate([dist_g, dist_f])
        ht_all   = np.concatenate([ht_g,   ht_f])
        az_rad   = np.radians(az_all)
        cx, cy   = self.cfg.ARRAY_CENTER
        self.positions = np.stack([
            cx + dist_all * np.cos(az_rad),
            cy + dist_all * np.sin(az_rad),
            ht_all,
        ], axis=1).astype(np.float32)

        drone_types       = list(_DRONE_FUNDAMENTALS.keys())
        self.drone_types  = [rng.choice(drone_types) for _ in range(n_samples)]
        self.fundamentals = [int(rng.choice(_DRONE_FUNDAMENTALS[dt][0:1])) for dt in self.drone_types]
        self.wind_speeds  = rng.uniform(0.0, 4.0, n_samples).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        from .utils import perturb_multichannel
        pos   = self.positions[idx]
        fund  = self.fundamentals[idx]
        wind  = float(self.wind_speeds[idx])
        noise = 0.03 + (0.07 * random.random() if self.augment else 0.0)

        # Use basic synthesise_drone (v2 physics synthesis is a bonus when available)
        chs = synthesise_drone(
            self.cfg.MIC_POSITIONS, pos[:2],
            fundamental=fund, noise_level=noise,
        )
        chs = [self.ap.pad_or_truncate(c) for c in chs]
        if self.augment:
            chs = perturb_multichannel(chs, self.cfg)

        ipd   = compute_ipd_features(chs, self.cfg)
        mels  = [self.ap.mel(c) for c in chs]
        mel_t = torch.tensor(np.stack(mels, axis=0), dtype=torch.float32)
        ipd_t = torch.tensor(ipd, dtype=torch.float32)

        cx, cy   = self.cfg.ARRAY_CENTER
        az_deg   = wrap_angle_deg(float(np.degrees(np.arctan2(pos[1] - cy, pos[0] - cx))))
        dist_m   = float(math.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2))
        az_rad   = math.radians(az_deg)
        max_dist = self.cfg.MAX_LOCALIZATION_DIST
        lbl_t    = torch.tensor(
            [math.sin(az_rad), math.cos(az_rad),
             float(np.clip(dist_m / max_dist, 0, 1.5)),
             float(np.clip(pos[2] / max_dist, 0, 1.5))],
            dtype=torch.float32,
        )
        return mel_t, ipd_t, lbl_t


# ══════════════════════════════════════════════════════════════════════════════
# Data managers
# ══════════════════════════════════════════════════════════════════════════════

class DroneAudioDatasetManager:
    """Download and prepare the DroneAudioDataset binary-label subset."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or config
        self.ap  = AudioProcessor(cfg)

    def prepare(self) -> bool:
        dest = self.cfg.DRONEDS_RAW
        proc = self.cfg.PROCESSED_DIR / "detection"
        counts = {
            f"{split}_{lbl}": len(list((proc / split / lbl).glob("*.wav")))
            if (proc / split / lbl).exists() else 0
            for split in ["train", "val", "test"]
            for lbl in ["drone", "non_drone"]
        }
        ready = all([
            counts["train_drone"]     > 20,
            counts["train_non_drone"] > 20,
            counts["val_drone"]       > 0,
            counts["val_non_drone"]   > 0,
            counts["test_drone"]      > 0,
            counts["test_non_drone"]  > 0,
        ])
        if ready:
            print(f"✅ Detection dataset ready ({sum(counts.values())} files)")
            return True
        print("⚠️  Detection dataset incomplete — rebuilding …")
        if proc.exists():
            shutil.rmtree(proc)
        self._download(dest)
        self._process(dest, proc)
        return True

    def _download(self, dest: Path):
        archive = dest / "drone_dataset.zip"
        dest.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print("📥 Downloading DroneAudioDataset …")
            urllib.request.urlretrieve(self.cfg.DRONEDS_ZIP_URL, str(archive))
        if not any(d.name == "Binary_Drone_Audio" for d in dest.rglob("*") if d.is_dir()):
            print("📦 Extracting …")
            with zipfile.ZipFile(archive) as z:
                z.extractall(str(dest))

    def _process(self, src: Path, dst: Path):
        import librosa
        binary = next(
            (d for d in src.rglob("Binary_Drone_Audio") if d.is_dir()), None
        )
        if binary is None:
            print("❌ Binary_Drone_Audio not found"); return
        mapping = {
            "yes_drone": "drone", "unknown": "non_drone",
            "Drone": "drone",     "noDrone": "non_drone",
        }
        all_files: Dict[str, List[Path]] = {"drone": [], "non_drone": []}
        for cls_dir in binary.iterdir():
            if not cls_dir.is_dir():
                continue
            label = mapping.get(cls_dir.name, "non_drone")
            all_files[label].extend(
                [f for f in cls_dir.glob("*.*") if f.is_file()]
            )
        for label, files in all_files.items():
            random.shuffle(files)
            n = len(files)
            splits = {
                "train": files[: int(n * 0.70)],
                "val":   files[int(n * 0.70) : int(n * 0.85)],
                "test":  files[int(n * 0.85) :],
            }
            for split, flist in splits.items():
                out = dst / split / label
                out.mkdir(parents=True, exist_ok=True)
                for f in flist:
                    tgt = out / f"{f.stem}.wav"
                    if tgt.exists():
                        continue
                    try:
                        y, _ = librosa.load(str(f), sr=self.cfg.SR, mono=True)
                        sf.write(str(tgt), y, self.cfg.SR)
                    except Exception as e:
                        print(f"   ⚠️  {f.name}: {e}")
        print("✅ Detection dataset processed")


def report_detection_split_counts(cfg: Optional[Config] = None):
    """Print a table of WAV counts per split/label."""
    cfg  = cfg or config
    root = cfg.PROCESSED_DIR / "detection"
    print("\n📊 Detection WAV split counts")
    for split in ["train", "val", "test"]:
        for label in ["drone", "non_drone"]:
            d = root / split / label
            n = len(list(d.glob("*.wav"))) if d.exists() else 0
            print(f"   {split:5s} / {label:10s}: {n}")


# ── Label parsing helpers for UaVirBASE ──────────────────────────────────────

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
    x_keys = ["x","X","east","East","east_m","pos_x","x_m"]
    y_keys = ["y","Y","north","North","north_m","pos_y","y_m"]
    z_keys = ["z","Z","up","Up","up_m","pos_z","z_m","height","height_m","altitude","alt"]
    x = _get_scalar(obj, x_keys)
    y = _get_scalar(obj, y_keys)
    z = _get_scalar(obj, z_keys)
    if x is not None and y is not None and z is not None:
        return (
            wrap_angle_deg(float(math.degrees(math.atan2(y, x)))),
            float(math.sqrt(x ** 2 + y ** 2)),
            float(abs(z)),
        )
    return None


def parse_label_json(raw: bytes) -> Optional[Tuple[float, float, float]]:
    """
    Parse a UaVirBASE label.json and return (azimuth_deg, distance_m, height_m).
    Returns None if the file is ambient-only or cannot be parsed.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    if "drone" in data and isinstance(data["drone"], dict):
        drone = data["drone"]
        ss    = drone.get("sound_source", "")
        if isinstance(ss, str) and "ambient" in ss.lower():
            return None
        try:
            az = drone.get("azimuth"); di = drone.get("distance"); ht = drone.get("height")
            if az is not None and di is not None and ht is not None:
                az, di, ht = float(az), float(di), float(ht)
                if not any(math.isnan(v) for v in [az, di, ht]):
                    return az, di, ht
        except (TypeError, ValueError):
            pass
        result = _cartesian_to_az_dist_ht(drone)
        if result:
            return result

    az = _get_scalar(data, _AZ_KEYS)
    di = _get_scalar(data, _DIST_KEYS)
    ht = _get_scalar(data, _HT_KEYS)
    if az is not None and di is not None and ht is not None:
        return float(az), float(di), float(ht)

    for sk in ["uav", "target", "labels", "annotation", "data", "position"]:
        if sk not in data:
            continue
        sub = data[sk]
        if isinstance(sub, dict):
            r = _cartesian_to_az_dist_ht(sub)
            if r:
                return r
    return None


# ── UaVirBASE manager (with position-grouped split from loc patch) ────────────

def _position_grouped_split(
    usable: list,
    train_frac: float = 0.625,
    val_frac:   float = 0.187,
    seed:       int   = 42,
) -> Dict[str, list]:
    """
    Assign sessions to splits by position group so that val/test positions
    were never seen during training.  This is the honest evaluation strategy
    required by the thesis.
    """
    pos_groups: Dict[tuple, list] = defaultdict(list)
    for item in usable:
        az, di, ht = item[2]
        key = (round(az / 45) * 45 % 360, round(di), round(ht))
        pos_groups[key].append(item)

    groups = sorted(pos_groups.items(), key=lambda x: x[0])
    rng    = random.Random(seed)
    rng.shuffle(groups)

    n       = len(groups)
    n_train = max(1, int(round(n * train_frac)))
    n_val   = max(1, int(round(n * val_frac)))
    n_test  = max(1, n - n_train - n_val)
    while n_train + n_val + n_test > n:
        n_test -= 1

    split_map: Dict[str, list] = {"train": [], "val": [], "test": []}
    for i, (pos_key, sessions) in enumerate(groups):
        if   i < n_train:             split_map["train"].extend(sessions)
        elif i < n_train + n_val:     split_map["val"].extend(sessions)
        else:                         split_map["test"].extend(sessions)

    print(f"\n📐 Position-grouped split ({n} unique positions):")
    for sp in ["train", "val", "test"]:
        pos_in = sorted({(round(s[2][0]), round(s[2][1]), round(s[2][2])) for s in split_map[sp]})
        print(f"   {sp:5s}: {len(split_map[sp]):3d} sessions | {len(pos_in):2d} positions")
    return split_map


class UaVirBASEDatasetManager:
    """
    Download and prepare UaVirBASE localization sessions.

    Uses position-grouped splits (from loc patch v1) so that evaluation
    positions are strictly held out from training.
    Falls back to a fully synthetic dataset if download fails.
    """

    AUDIO_FILENAME = "output.wav"
    LABEL_FILENAME = "label.json"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or config
        self.ap  = AudioProcessor(cfg)

    def prepare(self) -> bool:
        proc = self.cfg.PROCESSED_DIR / "localization"
        if (proc / "train").exists():
            n = len(list(proc.rglob("*_label.json")))
            if n > 20:
                print(f"✅ Localization dataset ready ({n} sessions)")
                return True
        url = self.cfg.UAVIRBASE_ZIP_URL
        if url is None:
            print("⚠️  UAVIRBASE_ZIP_URL is None → synthetic fallback.")
            self._write_synthetic(proc)
            return True
        if getattr(self.cfg, "UAVIRBASE_FULL", False):
            try:
                self._download_full(self.cfg.UAVIRBASE_RAW)
                self._process_local(self.cfg.UAVIRBASE_RAW, proc)
            except Exception as e:
                print(f"⚠️  Full download failed ({e}) → synthetic fallback.")
                self._write_synthetic(proc)
        else:
            n_sess = getattr(self.cfg, "UAVIRBASE_N_SESSIONS", 500)
            print(f"📥 Partial download: {n_sess} sessions via remotezip …")
            try:
                _ensure_remotezip()
                self._download_partial(url, proc, n_sess)
            except Exception as e:
                print(f"⚠️  Partial download failed ({e}) → synthetic fallback.")
                self._write_synthetic(proc)
        return True

    def _download_partial(self, url: str, proc: Path, n_sessions: int):
        """
        Stream only the required audio+label pairs from the remote ZIP using
        position-grouped splits (loc patch v1).
        """
        from remotezip import RemoteZip
        RZ_KWARGS        = {"initial_buffer_size": 64 * 1024 * 1024}
        AUDIO_CANDIDATES = {"output.wav", "audio.wav"}
        LABEL_CANDIDATES = {"label.json", "annotation.json"}

        print("   Reading remote ZIP central directory …")
        with RemoteZip(url, **RZ_KWARGS) as rz:
            all_names = rz.namelist()

        norm_names  = [n.replace("\\", "/") for n in all_names]
        session_map = {}
        for path in norm_names:
            p = Path(path)
            if len(p.parts) < 2:
                continue
            session_dir = str(Path(*p.parts[:-1]))
            session_map.setdefault(session_dir, []).append(p.name)

        paired = [
            (f"{sd}/{a}", f"{sd}/{l}")
            for sd, files in session_map.items()
            for a in [next((f for f in files if f in AUDIO_CANDIDATES), None)]
            for l in [next((f for f in files if f in LABEL_CANDIDATES), None)]
            if a and l
        ]
        print(f"   Candidate paired sessions: {len(paired)}")
        if not paired:
            raise RuntimeError("No paired audio/json sessions found.")

        print("   Validating labels …")
        usable = []; ambient_skipped = parse_failed = 0
        with RemoteZip(url, **RZ_KWARGS) as rz:
            from tqdm.auto import tqdm
            for audio_path, label_path in tqdm(paired, desc="Validating"):
                try:
                    raw    = rz.read(label_path)
                    parsed = parse_label_json(raw)
                    if parsed is None:
                        try:
                            content = json.loads(raw.decode("utf-8"))
                            src = content.get("drone", {}).get("sound_source", "")
                            if isinstance(src, str) and "ambient" in src.lower():
                                ambient_skipped += 1; continue
                        except Exception:
                            pass
                        parse_failed += 1; continue
                    usable.append((audio_path, label_path, parsed))
                except Exception:
                    parse_failed += 1

        print(f"   Usable: {len(usable)} | Ambient skipped: {ambient_skipped} | Failed: {parse_failed}")
        if not usable:
            raise RuntimeError("No usable sessions after validation.")

        # Position-grouped split (honest evaluation)
        split_map = _position_grouped_split(usable, seed=self.cfg.SEED)
        for s in ["train", "val", "test"]:
            (proc / s).mkdir(parents=True, exist_ok=True)

        downloaded = failed_audio = 0
        all_items  = (
            [(s, "train") for s in split_map["train"]] +
            [(s, "val")   for s in split_map["val"]]   +
            [(s, "test")  for s in split_map["test"]]
        )
        with RemoteZip(url, **RZ_KWARGS) as rz:
            from tqdm.auto import tqdm
            for (audio_path, label_path, parsed), split in tqdm(all_items, desc="Downloading"):
                session_id = Path(audio_path).parent.name
                try:
                    az, di, ht = parsed
                    audio_bytes = rz.read(audio_path)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(audio_bytes); tmp_path = tf.name
                    try:
                        channels = self.ap.load_channels(
                            tmp_path, channel_indices=self.cfg.UAVIRBASE_MIC_INDICES
                        )
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    out_dir = proc / split
                    for i, ch in enumerate(channels):
                        sf.write(str(out_dir / f"{session_id}_ch{i}.wav"),
                                 self.ap.pad_or_truncate(ch), self.cfg.SR)
                    (out_dir / f"{session_id}_label.json").write_text(
                        json.dumps({"azimuth_deg": az, "distance_m": di,
                                    "height_m": ht, "source": "real"})
                    )
                    downloaded += 1
                except Exception as e:
                    failed_audio += 1
                    print(f"   ⚠️  {session_id}: {e}")

        print(f"\n✅ Downloaded {downloaded} sessions ({failed_audio} failed)")
        for split in ["train", "val", "test"]:
            print(f"   {split}: {len(list((proc/split).glob('*_label.json')))} sessions")
        if downloaded == 0:
            raise RuntimeError("Zero sessions saved.")

    def _download_full(self, dest: Path):
        archive = dest / "uavirbase.zip"
        if not archive.exists():
            urllib.request.urlretrieve(self.cfg.UAVIRBASE_ZIP_URL, str(archive))
        with zipfile.ZipFile(archive) as z:
            z.extractall(str(dest))
        archive.unlink()

    def _process_local(self, src: Path, dst: Path):
        sessions = list(src.rglob(self.LABEL_FILENAME)) or list(src.rglob("annotation.json"))
        sessions = [p.parent for p in sessions]
        if not sessions:
            print("⚠️  No label files found."); return
        random.shuffle(sessions)
        n = len(sessions)
        splits = {
            "train": sessions[: int(n * 0.70)],
            "val":   sessions[int(n * 0.70) : int(n * 0.85)],
            "test":  sessions[int(n * 0.85) :],
        }
        for split, sess_list in splits.items():
            out = dst / split; out.mkdir(parents=True, exist_ok=True)
            for sess in sess_list:
                audio_file = next((sess / x for x in [self.AUDIO_FILENAME, "audio.wav"] if (sess / x).exists()), None)
                label_file = next((sess / x for x in [self.LABEL_FILENAME, "annotation.json"] if (sess / x).exists()), None)
                if not audio_file or not label_file:
                    continue
                try:
                    parsed = parse_label_json(label_file.read_bytes())
                    if parsed is None:
                        continue
                    az, di, ht = parsed
                    channels   = self.ap.load_channels(audio_file, channel_indices=self.cfg.UAVIRBASE_MIC_INDICES)
                    sid        = sess.name
                    for i, ch in enumerate(channels):
                        sf.write(str(out / f"{sid}_ch{i}.wav"), self.ap.pad_or_truncate(ch), self.cfg.SR)
                    (out / f"{sid}_label.json").write_text(
                        json.dumps({"azimuth_deg": az, "distance_m": di, "height_m": ht, "source": "real"})
                    )
                except Exception as e:
                    print(f"   ⚠️  {sess.name}: {e}")
        print("✅ Full dataset processed")

    def _write_synthetic(self, proc: Path, n_total: int = 2400):
        print(f"🔬 Generating {n_total} synthetic localization samples …")
        rng    = np.random.default_rng(self.cfg.SEED)
        dists  = np.exp(rng.uniform(np.log(0.3), np.log(self.cfg.MAX_LOCALIZATION_DIST), n_total))
        az_rad = rng.uniform(-np.pi, np.pi, n_total)
        heights= rng.uniform(0.5, 5.0, n_total)
        funds  = rng.choice([80, 90, 100, 110, 120, 130], n_total)
        noises = rng.uniform(0.01, 0.08, n_total)
        cx, cy = self.cfg.ARRAY_CENTER
        xs = cx + dists * np.cos(az_rad)
        ys = cy + dists * np.sin(az_rad)
        idx_tr  = int(n_total * 0.70); idx_val = int(n_total * 0.85)
        splits  = ["train"] * idx_tr + ["val"] * (idx_val - idx_tr) + ["test"] * (n_total - idx_val)
        for s in ["train", "val", "test"]:
            (proc / s).mkdir(parents=True, exist_ok=True)
        from tqdm.auto import tqdm
        for i in tqdm(range(n_total), desc="Synthetic loc"):
            sid  = f"synth_{i:06d}"
            chs  = synthesise_drone(self.cfg.MIC_POSITIONS, [xs[i], ys[i]],
                                    fundamental=int(funds[i]), noise_level=float(noises[i]))
            out  = proc / splits[i]
            for j, ch in enumerate(chs):
                sf.write(str(out / f"{sid}_ch{j}.wav"),
                         AudioProcessor(self.cfg).pad_or_truncate(ch), self.cfg.SR)
            (out / f"{sid}_label.json").write_text(
                json.dumps({
                    "azimuth_deg": float(np.degrees(az_rad[i])),
                    "distance_m":  float(dists[i]),
                    "height_m":    float(heights[i]),
                    "source":      "synthetic",
                })
            )
        print(f"✅ Synthetic fallback: {len(list(proc.rglob('*_label.json')))} sessions written.")


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════

def get_det_dataloaders(cfg: Optional[Config] = None):
    """
    Build (train, val, test) DataLoaders from the mel feature cache.

    Training uses WeightedRandomSampler to handle class imbalance.
    Returns a tuple (tr_loader, va_loader, te_loader).
    """
    cfg   = cfg or config
    cache = cfg.MEL_CACHE_DIR

    tr_ds = MelCachedDataset(cache, "train", augment=True)
    va_ds = MelCachedDataset(cache, "val",   augment=False)
    try:
        te_ds = MelCachedDataset(cache, "test", augment=False)
    except RuntimeError:
        print("⚠️  No test split in cache — using val as fallback.")
        te_ds = MelCachedDataset(cache, "val", augment=False)

    labels  = np.array(tr_ds.labels)
    counts  = np.bincount(labels)
    counts[counts == 0] = 1
    weights = (1.0 / counts)[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def _collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)

    bs = cfg.BATCH_SIZE
    nw = min(4, os.cpu_count() or 2)
    pin = (cfg.DEVICE == "cuda")
    kw  = dict(
        collate_fn=_collate,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    tr_l = DataLoader(tr_ds, batch_size=bs, sampler=sampler, **kw)
    va_l = DataLoader(va_ds, batch_size=bs, shuffle=False, **kw)
    te_l = DataLoader(te_ds, batch_size=bs, shuffle=False, **kw)
    return tr_l, va_l, te_l
