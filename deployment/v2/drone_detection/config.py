# -*- coding: utf-8 -*-
"""
config.py
─────────
Central configuration for the drone detection & localization pipeline.
Includes:
  - Multi-array support: MIC_POSITIONS can be switched via set_array_geometry()
  - Per-drone BPF bands derived from real PannoniaFS Q2 measurements
  - Two noise floor profiles (indoor / outdoor Brownian)
  - Learning rate and regularisation defaults tuned
  - BPF_ENERGY_RATIO_AS_FEATURE flag for the improved localization model
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff")

# ── Known array geometries ────────────────────────────────────────────────────
# All positions in metres (x, y).  Origin is the geometric centroid.
# GP1: equilateral triangle, 2165 mm leg (PannoniaFS Group-1 Schoeps/Brüel)
_GP1_LEG = 2.165
_GP1_POSITIONS = np.array([
    [0.0,                 _GP1_LEG * 2 / 3 * np.sin(np.pi / 2)],   # front
    [-_GP1_LEG / 2,      -_GP1_LEG / 3 * np.sin(np.pi / 2)],       # back-left
    [ _GP1_LEG / 2,      -_GP1_LEG / 3 * np.sin(np.pi / 2)],       # back-right
], dtype=np.float32)

# GP2: equilateral triangle, 2500 mm baseline (PannoniaFS Group-2 Brüel)
_GP2_LEG = 2.500
_GP2_POSITIONS = np.array([
    [0.0,                 _GP2_LEG * 2 / 3 * np.sin(np.pi / 2)],
    [-_GP2_LEG / 2,      -_GP2_LEG / 3 * np.sin(np.pi / 2)],
    [ _GP2_LEG / 2,      -_GP2_LEG / 3 * np.sin(np.pi / 2)],
], dtype=np.float32)

# UaVirBASE compact array: equilateral triangle, 200 mm baseline
_UAVIRBASE_POSITIONS = np.array(
    [[0.00, 0.00], [0.20, 0.00], [0.10, 0.1732]], dtype=np.float32
)

ARRAY_GEOMETRIES = {
    "uavirbase": _UAVIRBASE_POSITIONS,
    "gp1":       _GP1_POSITIONS,
    "gp2":       _GP2_POSITIONS,
}

# ── Per-drone BPF bands from PannoniaFS Q2 measurements ─────────────────────
# Format: (f_min_hz, f_center_hz, f_max_hz, n_harmonics)
# Source: q4_features.csv + analysis.md Q2 results
DRONE_BPF_PROFILES = {
    "mavic_pro":   (160, 209, 340, 4),   # measured 209 Hz at 1/3/4 m
    "mavic_2_pro": (160, 195, 340, 4),   # measured 193–199 Hz, very stable
    "mavic_mini":  (260, 360, 620, 4),   # measured 356–363 Hz
    "dji_phantom": (90,  100, 200, 4),   # legacy estimate
    "generic_quad":(80,  110, 350, 4),   # fallback
    "hexarotor":   (60,   75, 195, 6),
}

# BPF energy ratios measured per drone (mean across altitudes, from Q2)
DRONE_BPF_ENERGY_RATIOS = {
    "mavic_pro":   0.258,   # mean of 0.455, 0.098, 0.122, 0.337
    "mavic_2_pro": 0.446,   # mean of 0.541, 0.454, 0.432, 0.358
    "mavic_mini":  0.307,   # mean of 0.398, 0.344, 0.282, 0.202
}

# ── Noise floor profiles from real measurements ──────────────────────────────
# indoor: PannoniaFS pre-flight silence window (analysis.md Q1)
NOISE_FLOOR_INDOOR = {
    "median_db":   -82.7,
    "type":        "complex_broadband",  # not simple pink/white
    "peaks_hz":    [627, 1637, 4363, 2565, 1061, 3746, 2315, 35, 23],
    "peak_prominences_db": [22.8, 14.1, 13.3, 11.7, 10.8, 10.8, 11.0, 11.6, 12.3],
}
# outdoor: Dunakeszi Brüel (dunakeszi_analysis_report.pdf §4.3)
NOISE_FLOOR_OUTDOOR = {
    "median_db":   -78.0,   # approximate from PSD plots
    "type":        "brownian",           # f^-2 slope
    "snr_in_bpf_band_db": 5.0,          # ~5 dB above floor in 50–150 Hz
}


class Config:
    def __init__(self):
        self.IN_COLAB = "google.colab" in sys.modules
        if self.IN_COLAB:
            self._mount_drive()
        self._setup_paths()

        # ── Audio ──────────────────────────────────────────────────────────
        self.SR               = 22_050
        self.TARGET_DURATION  = 3.0
        self.N_MELS           = 64
        self.HOP_LENGTH       = 256
        self.N_FFT            = 1024

        # ── Active array geometry (switchable at runtime) ──────────────────
        self._array_name = "uavirbase"
        self.MIC_POSITIONS = _UAVIRBASE_POSITIONS.copy()
        self.SPEED_OF_SOUND        = 343.0
        self.ARRAY_CENTER          = self.MIC_POSITIONS.mean(axis=0)
        self.MAX_LOCALIZATION_DIST = 25.0

        # ── Dataset download URLs ──────────────────────────────────────────
        self.UAVIRBASE_ZIP_URL = (
            "https://zenodo.org/records/15391924/files/"
            "Microphone_array.zip?download=1"
        )
        self.UAVIRBASE_MIC_INDICES = [0, 1, 2]
        self.UAVIRBASE_ORIG_SR     = 96_000
        self.UAVIRBASE_FULL        = False
        self.UAVIRBASE_N_SESSIONS  = 2000

        self.DRONEDS_ZIP_URL = (
            "https://github.com/saraalemadi/DroneAudioDataset/"
            "archive/refs/heads/master.zip"
        )

        # ── Web scraping ───────────────────────────────────────────────────
        self.FREESOUND_API_KEY    = ""
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
        # Use measured BPF profiles in synthesis 
        self.SYNTHETIC_USE_MEASURED_BPF = True
        # Inject indoor + outdoor noise profiles during synthesis
        self.SYNTHETIC_NOISE_PROFILE = "mixed"  # "indoor" | "outdoor" | "mixed"
        # Append BPF energy ratio as extra scalar to IPD feature vector
        self.BPF_ENERGY_RATIO_AS_FEATURE = True

        # ── Training ──────────────────────────────────────────────────────
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 30
        # slightly lower base LR; cosine schedule handles warmup
        self.LR         = 3e-4
        self.SEED       = 42
        self.DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_AMP    = True
        # L2 weight decay (AdamW)
        self.WEIGHT_DECAY = 1e-3   # increased from 1e-4 for better regularisation
        # Gradient clip norm
        self.GRAD_CLIP    = 1.0    # tighter than 2.0

        # ── Detection thresholds ──────────────────────────────────────────
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

        # ── Kalman tracker ─────────────────────────────────────────────────
        self.LOC_CONFIDENCE_CAP = 5.0
        self.KF_PROCESS_NOISE   = 0.5
        self.KF_MEASURE_NOISE   = 0.3
        self.KF_MAX_COAST       = 4
        self.KF_MIN_HITS        = 1
        self.KF_MATCH_GATE      = 8.0
        self.MAX_DRONES         = 3
        self.TDOA_DEDUP_MS      = 0.029e-3

        # ── Custom builder dataset ────────────────────────────────────────
        self._init_custom_dataset_defaults()

    # ──────────────────────────────────────────────────────────────────────────
    # Array geometry helpers
    # ──────────────────────────────────────────────────────────────────────────

    def set_array_geometry(self, name: str):
        """
        Switch the active microphone array geometry at runtime.

        Parameters
        ──────────
        name : one of "uavirbase", "gp1", "gp2"
                "gp1" = PannoniaFS Group-1 (2165 mm leg, Schoeps/Brüel)
                "gp2" = PannoniaFS Group-2 (2500 mm base, Brüel)
        """
        if name not in ARRAY_GEOMETRIES:
            raise ValueError(
                f"Unknown array '{name}'. Choose from: "
                + ", ".join(ARRAY_GEOMETRIES)
            )
        self._array_name   = name
        self.MIC_POSITIONS = ARRAY_GEOMETRIES[name].copy()
        self.ARRAY_CENTER  = self.MIC_POSITIONS.mean(axis=0)
        print(
            f"✅ Array geometry set to '{name}'  "
            f"baseline ≈ {self._array_baseline_m():.3f} m"
        )

    def _array_baseline_m(self) -> float:
        """Return the maximum inter-mic distance for the active array."""
        mics = self.MIC_POSITIONS
        return float(max(
            np.linalg.norm(mics[i] - mics[j])
            for i in range(len(mics))
            for j in range(i + 1, len(mics))
        ))

    @property
    def array_name(self) -> str:
        return self._array_name

    # ──────────────────────────────────────────────────────────────────────────
    # BPF helpers (used by synthetic generator)
    # ──────────────────────────────────────────────────────────────────────────

    def get_bpf_profile(self, drone_type: str) -> tuple:
        """Return (f_min, f_center, f_max, n_harmonics) for a drone type."""
        return DRONE_BPF_PROFILES.get(
            drone_type, DRONE_BPF_PROFILES["generic_quad"]
        )

    def sample_fundamental_hz(self, drone_type: str, rng=None) -> float:
        """
        Sample a blade-pass fundamental frequency consistent with real
        measurements.  Draws from a tight Gaussian around f_center.
        """
        f_min, f_center, f_max, _ = self.get_bpf_profile(drone_type)
        if rng is None:
            import random
            sigma = (f_max - f_min) / 6.0  # ±3σ fits within band
            return float(np.clip(
                random.gauss(f_center, sigma), f_min, f_max
            ))
        sigma = (f_max - f_min) / 6.0
        return float(np.clip(
            rng.normal(f_center, sigma), f_min, f_max
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu_mem() -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0

    def _setup_paths(self):
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
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"Drive mount failed: {e}")

    def _init_custom_dataset_defaults(self):
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
            f"array={self._array_name}, "
            f"threshold={self.DETECTION_THRESHOLD:.2f})"
        )


config = Config()