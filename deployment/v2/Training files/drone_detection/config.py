# -*- coding: utf-8 -*-
"""
config.py
─────────
Central configuration for the drone detection & localization pipeline.

Usage:
    from config import Config, AUDIO_EXTS
    cfg = Config()
    cfg.ensure_dirs()
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch


# ── Audio file extensions accepted everywhere in the pipeline ─────────────────
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff")


class Config:
    """
    Single source of truth for every hyper-parameter, path, and flag used
    across training, inference, and deployment.

    Attribute sections
    ──────────────────
    Environment / paths · Audio · Mic array · Dataset URLs ·
    Web scraping · Synthetic data · Training · Detection thresholds ·
    External inference · Mixed-audio augmentation · Kalman tracker ·
    Custom builder dataset
    """

    def __init__(self):
        self.IN_COLAB = "google.colab" in sys.modules
        if self.IN_COLAB:
            self._mount_drive()
        self._setup_paths()

        # ── Audio ──────────────────────────────────────────────────────────
        self.SR               = 22_050    # sample rate (Hz)
        self.TARGET_DURATION  = 3.0       # clip length fed to models (s)
        self.N_MELS           = 64        # mel bands
        self.HOP_LENGTH       = 256       # STFT hop
        self.N_FFT            = 1024      # STFT window size

        # ── Microphone array (equilateral triangle, 20 cm baseline) ────────
        self.MIC_POSITIONS = np.array(
            [[0.00, 0.00], [0.20, 0.00], [0.10, 0.1732]], dtype=np.float32
        )
        self.SPEED_OF_SOUND        = 343.0
        self.ARRAY_CENTER          = self.MIC_POSITIONS.mean(axis=0)
        self.MAX_LOCALIZATION_DIST = 25.0   # metres

        # ── Dataset download URLs ──────────────────────────────────────────
        self.UAVIRBASE_ZIP_URL = (
            "https://zenodo.org/records/15391924/files/"
            "Microphone_array.zip?download=1"
        )
        self.UAVIRBASE_MIC_INDICES = [0, 1, 2]
        self.UAVIRBASE_ORIG_SR     = 96_000
        self.UAVIRBASE_FULL        = False       # True = download full dataset
        self.UAVIRBASE_N_SESSIONS  = 2000

        self.DRONEDS_ZIP_URL = (
            "https://github.com/saraalemadi/DroneAudioDataset/"
            "archive/refs/heads/master.zip"
        )

        # ── Web scraping ───────────────────────────────────────────────────
        self.FREESOUND_API_KEY    = ""           # optional
        self.SCRAPE_MAX_PER_QUERY = 50
        self.SCRAPE_MIN_DURATION  = 2.0
        self.SCRAPE_MAX_DURATION  = 15.0
        self.SCRAPE_YTDLP_URLS    = [
            "https://www.youtube.com/watch?v=5PSNL5qB3xQ",
            "https://www.youtube.com/watch?v=ZKvZv8sLPAo",
        ]
        self.SCRAPE_YTDLP_ENABLED = False

        # ── Synthetic data ─────────────────────────────────────────────────
        self.SYNTHETIC_DET_SAMPLES = 500
        self.SYNTHETIC_SAMPLES     = 2000

        # ── Training ──────────────────────────────────────────────────────
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 30
        self.LR         = 5e-4
        self.SEED       = 42
        self.DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_AMP    = True

        # ── Detection thresholds (v15: auto-searched on val, stored in ckpt) ─
        self.DETECTION_THRESHOLD      = 0.62
        self.DETECTION_THRESHOLD_LOW  = 0.35
        self.DETECTION_THRESHOLD_WEAK = 0.18
        self.CNN_WEIGHT               = 0.80
        self.HEURISTIC_WEIGHT         = 0.20

        # ── External / sliding-window inference ───────────────────────────
        self.EXTERNAL_INFER_THRESHOLD  = 0.35
        self.EXTERNAL_MIN_POS_SEGMENTS = 1
        self.EXTERNAL_SEGMENT_SEC      = 1.0
        self.EXTERNAL_SEGMENT_OVERLAP  = 0.5
        self.EXTERNAL_AGG_MODE         = "mean_topk"
        self.EXTERNAL_TOPK             = 3

        # ── Mixed-audio augmentation ──────────────────────────────────────
        self.MIXED_DRONE_SAMPLES   = 1200
        self.MIXED_DRONE_VAL_FRAC  = 0.15
        self.MIX_SNR_DB_RANGE      = (-5.0, 15.0)
        self.MIX_GAIN_RANGE_DB     = (-8.0, 8.0)
        self.MIX_BG_GAIN_RANGE_DB  = (-6.0, 6.0)
        self.MIX_BACKGROUND_LABELS = [
            "speech", "crowd", "wind", "traffic", "non_drone"
        ]
        self.MIX_CACHE_PREFIX = "mixdrone"

        # ── Kalman tracker (v2 patch values) ──────────────────────────────
        self.LOC_CONFIDENCE_CAP = 5.0
        self.KF_PROCESS_NOISE   = 0.5
        self.KF_MEASURE_NOISE   = 0.3
        self.KF_MAX_COAST       = 4      # v2: was 5
        self.KF_MIN_HITS        = 1      # v2: was 2 (lower so single detections confirm)
        self.KF_MATCH_GATE      = 8.0    # v2: was 2.0 m (wider for noisy TDOA)
        self.MAX_DRONES         = 3
        self.TDOA_DEDUP_MS      = 0.029e-3   # v1: was 0.05e-3 (580× too tight)

        # ── Custom builder dataset integration ────────────────────────────
        self._init_custom_dataset_defaults()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu_mem() -> float:
        """Return available GPU VRAM in GiB; 0.0 when no GPU is present."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0

    def _setup_paths(self):
        """Populate all local and Drive path attributes."""
        base  = "/content/drone_v15" if self.IN_COLAB else "/tmp/drone_v15"
        drive = (
            "/content/drive/MyDrive/drone_v15"
            if self.IN_COLAB else "/tmp/drone_v15"
        )
        B, D = Path(base), Path(drive)

        self.LOCAL_BASE    = B
        self.RAW_DIR       = B / "raw"
        self.PROCESSED_DIR = B / "processed"
        self.MEL_CACHE_DIR = B / "mel_cache"

        self.DRIVE_ROOT   = D
        self.DRIVE_MODELS = D / "models"
        self.DRIVE_LOGS   = D / "logs"
        self.DRIVE_TRACKS = D / "tracks"
        self.DRIVE_PLOTS  = D / "logs" / "plots"

        self.UAVIRBASE_RAW = B / "uavirbase"
        self.DRONEDS_RAW   = B / "droneds"

    def _mount_drive(self):
        """Mount Google Drive (Colab only); swallows errors gracefully."""
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"Drive mount failed: {e}")

    def _init_custom_dataset_defaults(self):
        """Set custom builder dataset attributes if not already present."""
        defaults = {
            "CUSTOM_DATASET_ENABLED":                       False,
            "CUSTOM_DATASET_ROOT":                          "",
            "CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE": True,
            "CUSTOM_DATASET_INCLUDE_CLEAN_IN_TRAIN":        True,
            "CUSTOM_DATASET_INCLUDE_CLEAN_IN_MIXING":       True,
            "CUSTOM_DATASET_PREFER_BUILDER_SPLITS":         True,
            "CUSTOM_DATASET_MIN_TEST_FRACTION":             0.10,
            "CUSTOM_DATASET_SKIP_SYNTH_IF_PRESENT":         True,
            "CUSTOM_DATASET_MANIFEST_NAME":                 "augmentation_manifest.json",
            "CUSTOM_DATASET_CLEAN_SUBDIR":                  "clean_drone_segments",
            "CUSTOM_DATASET_TRAIN_SUBDIR":                  "train/drone",
            "CUSTOM_DATASET_VAL_SUBDIR":                    "val/drone",
            "CUSTOM_DATASET_BGPOOL_SUBDIR":                 "background_pool",
            "CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR":           "manual_clean/clean_drone_sections",
            "CUSTOM_DATASET_IMPORTED_ROOT":                 str(
                self.RAW_DIR / "custom_builder_import"
            ),
            "ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD":     True,
            "ALLOW_EXTERNAL_AUDIO_DOWNLOADS":               False,
            "ALLOW_UAVIRBASE_DOWNLOAD":                     True,
        }
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def ensure_dirs(self):
        """Create all required local and (optional) Drive directories."""
        for p in [
            self.RAW_DIR, self.PROCESSED_DIR, self.MEL_CACHE_DIR,
            self.UAVIRBASE_RAW, self.DRONEDS_RAW,
        ]:
            os.makedirs(str(p), exist_ok=True)

        for p in [
            self.DRIVE_ROOT, self.DRIVE_MODELS,
            self.DRIVE_LOGS, self.DRIVE_TRACKS, self.DRIVE_PLOTS,
        ]:
            try:
                os.makedirs(str(p), exist_ok=True)
            except OSError as e:
                print(f"⚠️  {p}: {e}")

    def __repr__(self) -> str:
        return (
            f"Config(device={self.DEVICE}, sr={self.SR}, "
            f"threshold={self.DETECTION_THRESHOLD:.2f}, "
            f"use_lite_loc={self.USE_LITE_LOC})"
        )


# ── Module-level singleton — used as the default argument everywhere ──────────
config = Config()
