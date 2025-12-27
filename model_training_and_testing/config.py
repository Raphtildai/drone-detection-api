import sys
from pathlib import Path
import numpy as np
import torch

class Config:
    def __init__(self):
        # Detect environment
        self.IN_COLAB = 'google.colab' in sys.modules
        if self.IN_COLAB:
            self._mount_drive()

        # Base path: Drive in Colab, local otherwise
        self.BASE = Path("/content/drive/MyDrive/drone_project") if self.IN_COLAB else Path("./drone_project_local")

        # Directories (all auto-resolve to Drive if in Colab)
        self.RAW_DIR = self.BASE / "raw"
        self.PROCESSED_DIR = self.BASE / "processed"
        self.LOCAL_MELCACHE = self.BASE / "mel_cache"
        self.MODELS_DIR = self.BASE / "models"
        self.LOGS_DIR = self.BASE / "logs"
        self.TBOARD_DIR = self.BASE / "tensorboard"
        self.BACKUP_DIR = self.BASE / "backup"

        # Dataset
        self.GITHUB_ZIP_URL = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"

        # Audio processing
        self.SR = 22050
        self.TARGET_DURATION = 3.0
        self.N_MELS = 64
        self.HOP_LENGTH = 256
        self.N_FFT = 1024
        self.SYNTHETIC_DATA_SAMPLES = 2000

        # Training
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10
        self.LR = 1e-4
        self.SEED = 42
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Localization
        self.SPEED_OF_SOUND = 343.0
        self.MIC_POSITIONS = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.1, 0.2 * np.sqrt(3)/1.0]
        ])

        # Files
        self.LOSS_JSON = self.LOGS_DIR / "loss_history.json"

        # Augmentation
        self.USE_AMP = True
        self.AUG_SPEC_PROB = 0.5
        self.FREQ_MASK_PARAM = 8
        self.TIME_MASK_PARAM = 32

        self.DEBUG = True

        # Ensure all directories exist
        self.ensure_dirs()

    def _mount_drive(self):
        """Mount Google Drive in Colab"""
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except Exception as e:
            print("⚠️ Drive mount failed:", e)

    def ensure_dirs(self):
        """Create all necessary directories"""
        dir_paths = [
            self.RAW_DIR, self.PROCESSED_DIR, self.LOCAL_MELCACHE,
            self.MODELS_DIR, self.LOGS_DIR, self.TBOARD_DIR, self.BACKUP_DIR
        ]
        for d in dir_paths:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Could not create directory {d}: {e}")

    def path(self, category: str, filename: str = "") -> Path:
        """
        Return the correct path for a given category and optional filename.
        Automatically ensures the parent directory exists.
        Categories: 'raw', 'processed', 'melcache', 'models', 'logs', 'tensorboard', 'backup'
        Example:
            config.path('models', 'best_model.pt')
        """
        category_map = {
            'raw': self.RAW_DIR,
            'processed': self.PROCESSED_DIR,
            'melcache': self.LOCAL_MELCACHE,
            'models': self.MODELS_DIR,
            'logs': self.LOGS_DIR,
            'tensorboard': self.TBOARD_DIR,
            'backup': self.BACKUP_DIR
        }

        if category not in category_map:
            raise ValueError(f"Unknown category '{category}'. Valid: {list(category_map.keys())}")

        base_path = category_map[category]
        full_path = base_path / filename if filename else base_path

        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

# Global config
config = Config()