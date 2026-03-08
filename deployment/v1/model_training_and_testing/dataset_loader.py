# dataset_loader.py

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import config


# ==================== DATA LOADERS ====================
class MelCachedDataset(Dataset):
    def __init__(self, melcache_root: Path, split="train", augment=False):
        self.files = []
        self.labels = []
        self.augment = augment

        split_dir = Path(melcache_root) / split
        for idx, lbl in enumerate(["non_drone", "drone"]):
            folder = split_dir / lbl
            if folder.exists():
                for f in folder.glob("*.npy"):
                    self.files.append(f)
                    self.labels.append(idx)

        if len(self.files) == 0:
            raise RuntimeError(f"No mel files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])
        if mel.ndim == 2:
            mel = np.expand_dims(mel, 0)  # Add channel dimension

        if self.augment and random.random() < 0.5:
            mel = self._spec_augment(mel)

        return torch.tensor(mel, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def _spec_augment(self, mel):
        """Simple spec augmentation"""
        C, n_mels, T = mel.shape
        if random.random() < 0.3:
            f = random.randint(1, 8)
            f0 = random.randint(0, n_mels - f)
            mel[:, f0:f0+f, :] = 0.0
        if random.random() < 0.3:
            t = random.randint(1, 32)
            t0 = random.randint(0, T - t)
            mel[:, :, t0:t0+t] = 0.0
        return mel

def get_dataloaders(melcache_root: Path):
    """Get balanced dataloaders"""
    train_ds = MelCachedDataset(melcache_root, "train", augment=True)
    val_ds = MelCachedDataset(melcache_root, "val", augment=False)
    test_ds = MelCachedDataset(melcache_root, "test", augment=False)

    # Use weighted sampling for balance
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def collate_fn(batch):
        Xs, ys = zip(*batch)
        # Ensure 3 channels for all samples
        Xs = [x.repeat(3, 1, 1) if x.shape[0] == 1 else x for x in Xs]
        return torch.stack(Xs), torch.stack(ys)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"📊 Training samples: {len(train_ds)}")
    return train_loader, val_loader, test_loader