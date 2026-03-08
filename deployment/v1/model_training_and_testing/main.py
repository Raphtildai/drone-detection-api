# main.py

import random
import shutil
import numpy as np
import torch

from config import config
from dataset_manager import DatasetManager
from mel_cache import MelCacheManager
from dataset_loader import get_dataloaders
from model import SimpleDroneDetector
from training import TrainingManager
from tensorboard_logger import TensorBoardLogger
from synthetic import inject_synthetic_3ch_data

# ==================== MAIN EXECUTION ====================
def main(num_epochs=10):
    # Initialize configuration
    config.ensure_dirs()

    # Set random seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print("🚀 Starting Drone Detection Training")
    print(f"📍 Device: {config.DEVICE}")

    # Initialize managers
    dataset_mgr = DatasetManager(config)
    mel_cache_mgr = MelCacheManager(config)

    # Prepare dataset
    if not dataset_mgr.prepare_dataset():
        raise RuntimeError("❌ Failed to prepare dataset")

    # Create mel cache
    if config.LOCAL_MELCACHE.exists():
        shutil.rmtree(config.LOCAL_MELCACHE)

    print("🎵 Adding synthetic data...")
    inject_synthetic_3ch_data(num_samples=config.SYNTHETIC_DATA_SAMPLES) # Pass num_samples explicitly or let it default to None

    print("🎵 Creating mel cache...")
    mel_cache_mgr.create_mel_cache()

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config.LOCAL_MELCACHE)

    # Initialize model
    device = torch.device(config.DEVICE)
    model = SimpleDroneDetector(in_channels=3).to(device)

    # Setup TensorBoard
    tb_logger = TensorBoardLogger(config)
    writer = tb_logger.start()

    # Train model
    training_mgr = TrainingManager(config, model, device)
    training_mgr.train_and_evaluate(train_loader, val_loader, test_loader, num_epochs, tb_logger)

    # Cleanup
    tb_logger.close()

    print("✅ Training completed!")

if __name__ == "__main__":
    main()
