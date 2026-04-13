# -*- coding: utf-8 -*-
"""
drone_detection/config.py
─────────────────────────
# Config, AUDIO_EXTS
Central configuration object and global constants.
Import `config` from here everywhere else; do not scatter
magic numbers across modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


# ── Audio file extensions recognised across the whole pipeline ────────────────
AUDIO_EXTS: tuple[str, ...] = (
    ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff"
)


class Config:
    """
    Single source of truth for every hyper-parameter and path.

    Usage
    -----
    >>> from drone_detection.config import config
    >>> config.SR
    22050
    """

    def __init__(self) -> None:
        self.IN_COLAB: bool = "google.colab" in sys.modules
        if self.IN_COLAB:
            self._mount_drive()
        self._setup_paths()

        # ── Audio ──────────────────────────────────────────────────────────────
        self.SR: int              = 22050
        self.TARGET_DURATION: float = 3.0
        self.N_MELS: int          = 64
        self.HOP_LENGTH: int      = 256
        self.N_FFT: int           = 1024

        # ── Microphone array geometry (metres) ────────────────────────────────
        self.MIC_POSITIONS: np.ndarray = np.array(
            [[0.00, 0.00], [0.20, 0.00], [0.10, 0.1732]], dtype=np.float32
        )
        self.SPEED_OF_SOUND: float  = 343.0
        self.ARRAY_CENTER: np.ndarray = self.MIC_POSITIONS.mean(axis=0)
        self.MAX_LOCALIZATION_DIST: float = 25.0

        # ── UaVirBASE dataset ─────────────────────────────────────────────────
        self.UAVIRBASE_ZIP_URL: str = (
            "https://zenodo.org/records/15391924/files/"
            "Microphone_array.zip?download=1"
        )
        self.UAVIRBASE_MIC_INDICES: list[int] = [0, 1, 2]
        self.UAVIRBASE_ORIG_SR: int           = 96000
        self.UAVIRBASE_FULL: bool             = False
        self.UAVIRBASE_N_SESSIONS: int        = 2000

        # ── DroneAudioDataset ─────────────────────────────────────────────────
        self.DRONEDS_ZIP_URL: str = (
            "https://github.com/saraalemadi/DroneAudioDataset/"
            "archive/refs/heads/master.zip"
        )

        # ── Web scraping (optional) ───────────────────────────────────────────
        self.FREESOUND_API_KEY: str      = ""
        self.SCRAPE_MAX_PER_QUERY: int   = 50
        self.SCRAPE_MIN_DURATION: float  = 2.0
        self.SCRAPE_MAX_DURATION: float  = 15.0
        self.SCRAPE_YTDLP_URLS: list[str] = [
            "https://www.youtube.com/watch?v=5PSNL5qB3xQ",
            "https://www.youtube.com/watch?v=ZKvZv8sLPAo",
        ]
        self.SCRAPE_YTDLP_ENABLED: bool = False

        # ── Synthetic data ────────────────────────────────────────────────────
        self.SYNTHETIC_DET_SAMPLES: int = 500
        self.SYNTHETIC_SAMPLES: int     = 2000

        # ── Training ─────────────────────────────────────────────────────────
        self.BATCH_SIZE: int   = 32
        self.NUM_EPOCHS: int   = 30
        self.LR: float         = 5e-4
        self.SEED: int         = 42
        self.DEVICE: str       = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_AMP: bool     = True

        self._gpu_mem_gb: float  = self._detect_gpu_mem()
        self.USE_LITE_LOC: bool  = (self._gpu_mem_gb < 4.0)

        # ── Detection thresholds ─────────────────────────────────────────────
        self.DETECTION_THRESHOLD: float      = 0.62
        self.DETECTION_THRESHOLD_LOW: float  = 0.35
        self.DETECTION_THRESHOLD_WEAK: float = 0.18
        self.CNN_WEIGHT: float               = 0.80
        self.HEURISTIC_WEIGHT: float         = 0.20

        # ── External file inference ───────────────────────────────────────────
        self.EXTERNAL_INFER_THRESHOLD: float   = 0.35
        self.EXTERNAL_MIN_POS_SEGMENTS: int    = 1
        self.EXTERNAL_SEGMENT_SEC: float       = 1.0
        self.EXTERNAL_SEGMENT_OVERLAP: float   = 0.5
        self.EXTERNAL_AGG_MODE: str            = "mean_topk"
        self.EXTERNAL_TOPK: int                = 3

        # ── Mixed-drone augmentation ──────────────────────────────────────────
        self.MIXED_DRONE_SAMPLES: int          = 1200
        self.MIXED_DRONE_VAL_FRAC: float       = 0.15
        self.MIX_SNR_DB_RANGE: tuple[float, float]      = (-5.0, 15.0)
        self.MIX_GAIN_RANGE_DB: tuple[float, float]     = (-8.0, 8.0)
        self.MIX_BG_GAIN_RANGE_DB: tuple[float, float]  = (-6.0, 6.0)
        self.MIX_BACKGROUND_LABELS: list[str] = [
            "speech", "crowd", "wind", "traffic", "non_drone"
        ]
        self.MIX_CACHE_PREFIX: str = "mixdrone"

        # ── Kalman tracker ────────────────────────────────────────────────────
        # NOTE: multidrone_localization_patch_v2 widens these at runtime.
        self.LOC_CONFIDENCE_CAP: float = 5.0
        self.KF_PROCESS_NOISE: float   = 0.5
        self.KF_MEASURE_NOISE: float   = 0.3
        self.KF_MAX_COAST: int         = 5
        self.KF_MIN_HITS: int          = 2   # patched to 1 for multi-drone
        self.KF_MATCH_GATE: float      = 2.0  # patched to 8.0 for multi-drone
        self.MAX_DRONES: int           = 3
        self.TDOA_DEDUP_MS: float      = 0.029e-3  # v2 patch value (was 0.05e-6)

        # ── Custom-builder dataset integration ───────────────────────────────
        self.CUSTOM_DATASET_ENABLED: bool  = False
        self.CUSTOM_DATASET_ROOT: str      = ""
        self.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE: bool = True
        self.CUSTOM_DATASET_INCLUDE_CLEAN_IN_TRAIN: bool        = True
        self.CUSTOM_DATASET_INCLUDE_CLEAN_IN_MIXING: bool       = True
        self.CUSTOM_DATASET_PREFER_BUILDER_SPLITS: bool         = True
        self.CUSTOM_DATASET_MIN_TEST_FRACTION: float            = 0.10
        self.CUSTOM_DATASET_SKIP_SYNTH_IF_PRESENT: bool         = True
        self.CUSTOM_DATASET_MANIFEST_NAME: str    = "augmentation_manifest.json"
        self.CUSTOM_DATASET_CLEAN_SUBDIR: str     = "clean_drone_segments"
        self.CUSTOM_DATASET_TRAIN_SUBDIR: str     = "train/drone"
        self.CUSTOM_DATASET_VAL_SUBDIR: str       = "val/drone"
        self.CUSTOM_DATASET_BGPOOL_SUBDIR: str    = "background_pool"
        self.CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR: str = "manual_clean/clean_drone_sections"
        self.CUSTOM_DATASET_IMPORTED_ROOT: str    = ""   # set in ensure_dirs()

        self.ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD: bool = True
        self.ALLOW_EXTERNAL_AUDIO_DOWNLOADS: bool           = False
        self.ALLOW_UAVIRBASE_DOWNLOAD: bool                 = True

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu_mem() -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0

    def _setup_paths(self) -> None:
        base  = "/content/drone_v15"       if self.IN_COLAB else "/tmp/drone_v15"
        drive = "/content/drive/MyDrive/drone_v15" if self.IN_COLAB else "/tmp/drone_v15"
        B, D  = Path(base), Path(drive)

        self.LOCAL_BASE: Path    = B
        self.RAW_DIR: Path       = B / "raw"
        self.PROCESSED_DIR: Path = B / "processed"
        self.MEL_CACHE_DIR: Path = B / "mel_cache"

        self.DRIVE_ROOT: Path   = D
        self.DRIVE_MODELS: Path = D / "models"
        self.DRIVE_LOGS: Path   = D / "logs"
        self.DRIVE_TRACKS: Path = D / "tracks"
        self.DRIVE_PLOTS: Path  = D / "logs/plots"

        self.UAVIRBASE_RAW: Path = B / "uavirbase"
        self.DRONEDS_RAW: Path   = B / "droneds"

    def _mount_drive(self) -> None:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"Drive mount failed: {e}")

    def ensure_dirs(self) -> None:
        """Create all local and Drive directories."""
        for p in [
            self.RAW_DIR, self.PROCESSED_DIR, self.MEL_CACHE_DIR,
            self.UAVIRBASE_RAW, self.DRONEDS_RAW,
        ]:
            os.makedirs(str(p), exist_ok=True)

        for p in [
            self.DRIVE_ROOT, self.DRIVE_MODELS, self.DRIVE_LOGS,
            self.DRIVE_TRACKS, self.DRIVE_PLOTS,
        ]:
            try:
                os.makedirs(str(p), exist_ok=True)
            except OSError as e:
                print(f"⚠️  {p}: {e}")

        # Derived path that depends on LOCAL_BASE being set
        self.CUSTOM_DATASET_IMPORTED_ROOT = str(
            self.RAW_DIR / "custom_builder_import"
        )


# ── Module-level singleton ────────────────────────────────────────────────────
config = Config()