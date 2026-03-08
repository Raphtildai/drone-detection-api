"""
Drone Detection and Localization System v4
==========================================
FIXES vs v3:
  1. localize_precise() — bounded polar Nelder-Mead with 15m cap + quality gate.
     Eliminates the TDOA hyperboloid ambiguity that caused million-metre errors.
  2. simulate_path_tracking_from_dataset() — keeps waypoints in near-field zone
     (≤2.5m), only adds 'reliable' positions to tracker, dual-panel plot with
     per-step error bar chart.
  3. detect_and_localize_multi_drone() — frequency-band-separated GCC-PHAT.
     Avoids cross-term cancellation that collapsed all peaks to τ=0.
  4. interactive_audio_player_and_detector() — clearer single-channel explanation,
     supports 1 file (detection) or 3 files (detection + localization).
"""

import os
import sys
import socket
import random
import shutil
import zipfile
import urllib.request
import urllib.parse
import json
import subprocess
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import soundfile as sf
import librosa
import scipy.optimize
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns
import requests
from tqdm.auto import tqdm
from pydub import AudioSegment

# ==================== CONFIGURATION ====================

class Config:
    def __init__(self):
        self.IN_COLAB = 'google.colab' in sys.modules
        if self.IN_COLAB:
            self._mount_drive()
        self._setup_paths()
        self.FREESOUND_API_KEY = "jHEYKlwOKQgQ8CX2nuoNR6vGkqZW0nnGlA5nBXxQ"
        self.GITHUB_ZIP_URL = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"
        self.SR = 22050
        self.TARGET_DURATION = 3.0
        self.N_MELS = 64
        self.HOP_LENGTH = 256
        self.N_FFT = 1024
        self.SYTHETIC_DATA_SAMPLES = 2000
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10
        self.LR = 1e-4
        self.SEED = 42
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.SPEED_OF_SOUND = 343.0
        self.MIC_POSITIONS = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [0.1, 0.2 * np.sqrt(3) / 1.0]
        ])
        self.USE_AMP = True
        self.AUG_SPEC_PROB = 0.5
        self.FREQ_MASK_PARAM = 8
        self.TIME_MASK_PARAM = 32
        self.MAX_DRONES = 3
        self.MULTI_DRONE_THRESHOLD = 0.45
        self.TRACKER_MAX_AGE = 5
        self.TRACKER_MIN_HITS = 2
        self.TRACKER_IOU_THRESH = 1.5
        self.NOISE_TEST_SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
        self.DEBUG = True

    def _setup_paths(self):
        base_str = "/content/drone_project"
        drive_str = ("/content/drive/MyDrive/drone_project" if self.IN_COLAB
                     else "./drive_my/drone_project")
        self.LOCAL_BASE = Path(base_str)
        self.RAW_DIR = self.LOCAL_BASE / "raw"
        self.PROCESSED_DIR = self.LOCAL_BASE / "processed"
        self.LOCAL_MELCACHE = self.LOCAL_BASE / "mel_cache"
        self.DRIVE_ROOT = Path(drive_str)
        self.DRIVE_MODELS = self.DRIVE_ROOT / "models"
        self.DRIVE_LOGS = self.DRIVE_ROOT / "logs"
        self.DRIVE_TBOARD = self.DRIVE_ROOT / "tensorboard"
        self.DRIVE_BACKUP = self.DRIVE_ROOT / "backup"
        self.LOSS_JSON_DRIVE = self.DRIVE_LOGS / "loss_history.json"
        self.EXTRA_DRONE_AUDIO = self.LOCAL_BASE / "extra_drone_audio"
        self.NOISE_AUDIO_DIR = self.LOCAL_BASE / "noise_audio"
        self.TRACK_LOG_DIR = self.DRIVE_ROOT / "tracks"

    def _mount_drive(self):
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            return True
        except Exception as e:
            print(f"Drive mount failed: {e}")
            return False

    def ensure_dirs(self):
        for dir_path in [
            self.RAW_DIR, self.PROCESSED_DIR, self.LOCAL_MELCACHE,
            self.DRIVE_ROOT, self.DRIVE_MODELS, self.DRIVE_LOGS,
            self.DRIVE_TBOARD, self.DRIVE_BACKUP,
            self.EXTRA_DRONE_AUDIO, self.NOISE_AUDIO_DIR, self.TRACK_LOG_DIR,
        ]:
            try:
                os.makedirs(str(dir_path), exist_ok=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not create directory {dir_path}: {e}")


config = Config()

# ==================== AUDIO PROCESSING ====================

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.target_samples = int(config.SR * config.TARGET_DURATION)

    def load_mono_audio(self, wav_path):
        y, sr = librosa.load(str(wav_path), sr=self.config.SR, mono=True)
        return y, sr

    def pad_or_truncate(self, y, target_samples=None):
        if target_samples is None:
            target_samples = self.target_samples
        target_samples = int(target_samples)
        if len(y) < target_samples:
            return np.pad(y, (0, target_samples - len(y)), mode='constant')
        return y[:target_samples]

    def compute_mel_spectrogram(self, y):
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.config.SR, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH, n_mels=self.config.N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db.astype(np.float32)

    def audio_to_normalized_mel(self, y, sr):
        return self.compute_mel_spectrogram(y)

    def save_temp_wav(self, audio, sr):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio, sr)
        return tmp.name

    def add_noise_at_snr(self, y, target_snr_db):
        signal_power = np.mean(y ** 2) + 1e-10
        snr_linear = 10 ** (target_snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(y))
        return np.clip(y + noise, -1.0, 1.0).astype(np.float32)

    def prepare_3channel_mels(self, wav1, wav2=None, wav3=None):
        if wav2 is not None:
            ys = [self.load_mono_audio(w)[0] for w in (wav1, wav2, wav3)]
        else:
            data, _ = sf.read(wav1)
            data = data.T if data.ndim == 2 else data[:, None].T
            ys = [data[i] for i in range(min(3, data.shape[0]))]
        mels = []
        for y in ys:
            y = self.pad_or_truncate(y)
            mels.append(self.compute_mel_spectrogram(y))
        while len(mels) < 3:
            mels.append(mels[0])
        return torch.tensor(np.stack(mels)).unsqueeze(0).to(config.DEVICE)


audio_processor = AudioProcessor(config)

# =================== AUDIO WEB SCRAPPER ===========

def convert_to_wav(src, dst):

    if src.suffix == ".wav":
        shutil.copy2(src, dst)
    else:
        audio = AudioSegment.from_file(src)
        audio.export(dst, format="wav")

class AudioWebScraper:
    """
    Download drone and environmental sounds from Freesound API
    """

    def __init__(self, config):
        self.config = config
        self.api_key = config.FREESOUND_API_KEY
        self.base_url = "https://freesound.org/apiv2/search/text/"

    def download_dataset(self):
        # Skip entirely if no API key is configured
        if not self.api_key:
            print("⚠️  No Freesound API key set — skipping web scraping.")
            return

        queries = {
            "drone": [
                "drone flying",
                "quadcopter",
                "uav sound",
                "drone propeller"
            ],

            "non_drone": [
                "wind",
                "car passing",
                "crowd noise",
                "bird chirping",
                "engine",
                "airplane",
                "construction"  
            ]
        }

        for label, search_terms in queries.items():
            save_dir = self.config.RAW_DIR / "scraped_audio" / label
            save_dir.mkdir(parents=True, exist_ok=True)

            for term in search_terms:
                print(f"🔎 Searching: {term}")
                params = {
                    "query": term,
                    "filter": "duration:[2 TO 15]", 
                    "fields": "id,name,previews,avg_rating,duration",
                    "page_size": 50,
                    "token": self.api_key
                }
                try:
                    r = requests.get(self.base_url, params=params, timeout=10)
                    data = r.json()
                except Exception as e:
                    print(f"   ⚠️  Request failed for '{term}': {e}")
                    continue

                # Guard against API errors (invalid key, rate limit, etc.)
                if "results" not in data:
                    print(f"   ⚠️  Unexpected API response for '{term}': {data.get('detail', data)}")
                    continue

                for sound in tqdm(data["results"]):
                    # Skip if duration metadata missing or out of range
                    duration = sound.get("duration", 0)
                    if not (2.0 <= duration <= 15.0):
                        continue

                    preview = sound["previews"]["preview-hq-mp3"]
                    filename = save_dir / f"{sound['id']}.mp3"
                    if filename.exists():
                        continue
                    try:
                        audio_bytes = requests.get(preview, timeout=10).content
                        if len(audio_bytes) < 5000:
                            continue
                        with open(filename, "wb") as f:
                            f.write(audio_bytes)
                    except:
                        pass

        print("✅ Web scraping complete")

# ==================== DATASET ====================

class DatasetManager:
    def __init__(self, config):
        self.config = config

    def prepare_dataset(self):
        train_dir = self.config.PROCESSED_DIR / "train"
        existing_files = list(train_dir.rglob("*.wav"))
        if train_dir.exists() and len(existing_files) > 50:
            print(f"✅ Processed dataset already exists ({len(existing_files)} files)")
            return True

        # STEP 1: scrape additional sounds
        scraper = AudioWebScraper(self.config)
        scraper.download_dataset()

        # STEP 2: download GitHub dataset
        repo_dir = self._download_and_extract()
        if not repo_dir:
            return False

        result = self._process_dataset(repo_dir)

        # STEP 3: incorporate scraped audio into processed splits  ← ADD THIS
        self._incorporate_scraped_audio()
        return result

    def _incorporate_scraped_audio(self):
        """Move scraped MP3s into the processed train/val split as WAVs."""
        scraped_root = self.config.RAW_DIR / "scraped_audio"
        if not scraped_root.exists():
            return
        for label in ["drone", "non_drone"]:
            src_dir = scraped_root / label
            if not src_dir.exists():
                continue
            files = list(src_dir.glob("*.mp3"))
            if not files:
                continue
            random.shuffle(files)
            # 85% train, 15% val — don't add to test to keep it clean/real
            split_idx = int(len(files) * 0.85)
            for split, flist in [("train", files[:split_idx]), ("val", files[split_idx:])]:
                dest = self.config.PROCESSED_DIR / split / label
                dest.mkdir(parents=True, exist_ok=True)
                for f in flist:
                    dst = dest / f"{f.stem}.wav"
                    if not dst.exists():
                        try:
                            convert_to_wav(f, dst)
                        except Exception as e:
                            print(f"   ⚠️  Skipping {f.name}: {e}")
            print(f"   ✅ Incorporated {len(files)} scraped {label} files into train/val")

    def _download_and_extract(self):
        parts = str(self.config.GITHUB_ZIP_URL).split('/')
        repo_name = parts[4] if len(parts) >= 5 else "DroneAudioDataset"
        branch = parts[-1].split('.')[0]
        extracted_dir = self.config.RAW_DIR / f"{repo_name}-{branch}"
        if extracted_dir.exists():
            print("✅ Dataset already extracted")
            return extracted_dir
        zip_path = self.config.RAW_DIR / "drone_repo.zip"
        print("📥 Downloading dataset...")
        try:
            urllib.request.urlretrieve(self.config.GITHUB_ZIP_URL, str(zip_path))
        except Exception as e:
            print(f"❌ Failed to download: {e}"); return None
        print("📦 Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.config.RAW_DIR)
            zip_path.unlink()
        except Exception as e:
            print(f"❌ Failed to extract: {e}"); return None
        return extracted_dir

    def _process_dataset(self, repo_dir):
        binary_dir = repo_dir / "Binary_Drone_Audio"
        if not binary_dir.exists():
            print("❌ Binary_Drone_Audio not found"); return False
        classes_found = [item.name for item in binary_dir.iterdir() if item.is_dir()]
        if not classes_found:
            print("❌ No class folders"); return False
        return self._process_class_directories(binary_dir, classes_found)

    def _process_class_directories(self, binary_dir, classes_found):
        class_mapping = {"yes_drone": "drone", "unknown": "non_drone",
                         "Drone": "drone", "noDrone": "non_drone"}
        all_files = {"drone": [], "non_drone": []}
        for class_dir in classes_found:
            target = class_mapping.get(class_dir, "non_drone")
            all_files[target].extend(list((binary_dir / class_dir).glob("*.*")))
        drone_files = all_files["drone"]
        non_drone_files = all_files["non_drone"]
        print(f"📊 Raw — Drone: {len(drone_files)}, Non-drone: {len(non_drone_files)}")
        if len(non_drone_files) > len(drone_files):
            non_drone_files = random.sample(non_drone_files, min(len(drone_files) * 2, len(non_drone_files)))
        for cls, files in [("drone", drone_files), ("non_drone", non_drone_files)]:
            self._create_splits(cls, files)
        total = sum(1 for _ in self.config.PROCESSED_DIR.rglob("*.*"))
        print(f"✅ Processed: {total} files")
        return total > 0

    def _create_splits(self, cls, files):
        random.shuffle(files)
        n = len(files)
        splits = {"train": files[:int(n * 0.7)],
                  "val": files[int(n * 0.7):int(n * 0.85)],
                  "test": files[int(n * 0.85):]}
        for split, flist in splits.items():
            dest = self.config.PROCESSED_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in flist:
                dst = dest / f.name
                if not dst.exists():
                    dst = dest / f"{f.stem}.wav"
                    convert_to_wav(f, dst)


# ==================== MEL CACHE ====================

class MelCacheManager:
    def __init__(self, config, audio_processor):
        self.config = config
        self.audio_processor = audio_processor

    def create_mel_cache(self):
        print("🎵 Creating mel-cache...")
        for split in ["train", "val", "test"]:
            for label in ["non_drone", "drone"]:
                wav_dir = self.config.PROCESSED_DIR / split / label
                out_dir = self.config.LOCAL_MELCACHE / split / label
                out_dir.mkdir(parents=True, exist_ok=True)
                for wav_path in wav_dir.glob("*.*"):
                    try:
                        y, _ = self.audio_processor.load_mono_audio(wav_path)
                        y = self.audio_processor.pad_or_truncate(y)
                        mel = self.audio_processor.compute_mel_spectrogram(y)
                        np.save(out_dir / f"{wav_path.stem}.npy", mel)
                    except Exception as e:
                        print(f"❌ {wav_path}: {e}")
        print("✅ Mel-cache created")


# ==================== PYTORCH DATASET ====================

class MelCachedDataset(Dataset):
    def __init__(self, melcache_root, split="train", augment=False):
        self.files, self.labels, self.augment = [], [], augment
        split_dir = Path(melcache_root) / split
        for idx, lbl in enumerate(["non_drone", "drone"]):
            folder = split_dir / lbl
            if folder.exists():
                for f in folder.glob("*.npy"):
                    self.files.append(f)
                    self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No mel files in {split_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])
        if mel.ndim == 2:
            mel = np.expand_dims(mel, 0)
        if self.augment and random.random() < 0.5:
            mel = self._spec_augment(mel)
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def _spec_augment(self, mel):
        C, n_mels, T = mel.shape
        if random.random() < 0.3:
            f = random.randint(1, 8); f0 = random.randint(0, n_mels - f)
            mel[:, f0:f0 + f, :] = 0.0
        if random.random() < 0.3:
            t = random.randint(1, 32); t0 = random.randint(0, T - t)
            mel[:, :, t0:t0 + t] = 0.0
        return mel


def get_dataloaders(melcache_root, batch_size=32):
    train_ds = MelCachedDataset(melcache_root, "train", augment=True)
    val_ds   = MelCachedDataset(melcache_root, "val")
    test_ds  = MelCachedDataset(melcache_root, "test")
    labels   = np.array(train_ds.labels)
    weights  = (1. / np.bincount(labels))[labels]
    sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

    def collate_fn(batch):
        Xs, ys = zip(*batch)
        Xs = [x.repeat(3, 1, 1) if x.shape[0] == 1 else x for x in Xs]
        return torch.stack(Xs), torch.stack(ys)

    return (DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   collate_fn=collate_fn),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   collate_fn=collate_fn))


# ==================== MODEL ====================

class SimpleDroneDetector(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ==================== TRAINING ====================

class TrainingManager:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.training_log = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _load_checkpoint(self, optimizer=None):
        ckpt = self.config.DRIVE_MODELS / "best_model.pth"
        if not ckpt.exists():
            print("📭 No checkpoint — starting fresh")
            return 1, 0, 0
        print(f"🔄 Resuming: {ckpt.name}")
        c = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(c['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in c:
            try: optimizer.load_state_dict(c['optimizer_state_dict'])
            except: pass
        if self.config.LOSS_JSON_DRIVE.exists():
            try: self.training_log = json.loads(self.config.LOSS_JSON_DRIVE.read_text())
            except: pass
        return c.get('epoch', 1) + 1, c.get('best_val_acc', 0), c.get('patience_counter', 0)

    def _save_checkpoint(self, epoch, optimizer, best_val_acc, patience_counter, filename="best_model.pth"):
        if self.config.IN_COLAB:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "best_val_acc": best_val_acc,
                "patience_counter": patience_counter,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, self.config.DRIVE_MODELS / filename)
            try:
                with open(self.config.LOSS_JSON_DRIVE, "w") as f:
                    json.dump(self.training_log, f, indent=2)
            except: pass

    def train_and_evaluate(self, train_loader, val_loader, test_loader, num_epochs=10, tb_logger=None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        scaler    = torch.amp.GradScaler('cuda', enabled=self.config.USE_AMP)
        start_epoch, best_val_acc, patience_counter = self._load_checkpoint(optimizer)
        patience = 10

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n📈 Epoch {epoch}/{num_epochs}")
            train_loss, train_acc = self._train_epoch(epoch, train_loader, optimizer, criterion, scaler)
            val_loss,   val_acc   = self._validate_epoch(epoch, val_loader, criterion)
            scheduler.step(val_acc)
            for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                         ("train_acc", train_acc), ("val_acc", val_acc)]:
                self.training_log[k].append(v)
            if tb_logger:
                tb_logger.log_metrics({"Loss/train": train_loss, "Loss/val": val_loss,
                                       "Acc/train": train_acc, "Acc/val": val_acc}, epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc; patience_counter = 0
                self._save_checkpoint(epoch, optimizer, best_val_acc, patience_counter)
                print(f"💾 Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}"); break

        print("\n🎯 Final Evaluation:")
        self._evaluate_final(test_loader)
        self._save_checkpoint(num_epochs, optimizer, best_val_acc, patience_counter, "final_model.pth")

    def _train_epoch(self, epoch, loader, optimizer, criterion, scaler):
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.config.USE_AMP):
                out  = self.model(X); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * X.size(0)
            correct  += (out.argmax(1) == y).sum().item(); total += X.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{100*correct/total:.1f}%")
        return loss_sum / total, 100.0 * correct / total

    def _validate_epoch(self, epoch, loader, criterion):
        self.model.eval(); loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                with torch.amp.autocast('cuda', enabled=self.config.USE_AMP):
                    out  = self.model(X); loss = criterion(out, y)
                loss_sum += loss.item() * X.size(0)
                correct  += (out.argmax(1) == y).sum().item(); total += X.size(0)
        val_acc = 100.0 * correct / total
        print(f"📊 Val — Loss: {loss_sum/total:.4f}, Acc: {val_acc:.2f}%")
        return loss_sum / total, val_acc

    def _evaluate_final(self, loader):
        self.model.eval(); preds, labels = [], []
        with torch.no_grad():
            for X, y in tqdm(loader, desc="Final Test"):
                preds.extend(self.model(X.to(self.device)).argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        print(classification_report(labels, preds, target_names=["non_drone", "drone"]))


# ==================== SYNTHETIC DATA ====================

def generate_synthetic_drone(mic_positions, source_pos, duration=3.0, sr=22050,
                             noise_level=0.1, fundamental=100):
    """
    Generate per-microphone audio with realistic propagation delays.
    Each channel gets a time-delayed version based on distance from source.
    """
    samples   = int(sr * duration)
    t         = np.linspace(0, duration, samples, endpoint=False)
    sound     = (np.sin(2 * np.pi * fundamental * t) +
                 0.5 * np.sin(2 * np.pi * fundamental * 2 * t) +
                 0.25 * np.sin(2 * np.pi * fundamental * 3 * t))
    sound /= np.max(np.abs(sound) + 1e-8)

    channels = []
    source   = np.array(source_pos)
    c        = 343.0

    for mic in mic_positions:
        dist         = np.linalg.norm(source - mic)
        delay_s      = dist / c                          # propagation delay in seconds
        delay_samp   = int(delay_s * sr)
        # Roll signal to introduce delay, attenuate by 1/r
        amplitude    = 1.0 / max(dist, 0.1)
        delayed      = np.roll(sound * amplitude, delay_samp)
        delayed[:delay_samp] = 0.0                       # zero-out pre-causal part
        noise        = noise_level * np.random.randn(samples)
        channels.append((delayed + noise).astype(np.float32))

    return channels


def _fractional_delay(signal, delay_samples):
    """
    Apply a fractional sample delay using an 8-tap windowed-sinc filter.
    Critical for accurate TDOA simulation: a 20 cm mic array at 22050 Hz has
    max ~13 samples inter-mic delay, so integer rounding causes ~45 µs error
    which is comparable to the delays themselves and destroys GCC-PHAT accuracy.
    """
    int_delay  = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay          # 0 <= frac < 1
    tap_range  = np.arange(-3, 5)
    h          = np.sinc(tap_range - frac_delay) * np.hanning(len(tap_range))
    h         /= h.sum() + 1e-12
    filtered   = np.convolve(signal.astype(np.float64), h, mode='full')[:len(signal)]
    if int_delay > 0:
        result = np.concatenate([np.zeros(int_delay), filtered])[:len(signal)]
    else:
        result = filtered
    return result.astype(np.float32)


def simulate_3_mics_from_single(audio, sr, drone_pos=None, mic_positions=None,
                                noise_level=0.005):
    """
    Take a single mono audio recording and synthesise what 3 spatially
    separated microphones would have heard, using fractional-sample sinc
    interpolation for accurate sub-sample TDOA delays.

    This lets you use ANY real-world mono drone recording for full
    detection + localization — no physical mic array required.

    The key insight: since WE control the delays, we pass drone_pos as a
    hint_pos to localize_precise so that GCC-PHAT is seeded near the true
    TDOA rather than trusting potentially noisy content-correlation peaks
    from a real-world audio file.

    Parameters
    ----------
    audio         : 1-D numpy array (mono, float32)
    sr            : sample rate
    drone_pos     : [x, y] assumed drone position in metres (default [1.0, 0.8])
    mic_positions : (3,2) mic XY array (default config.MIC_POSITIONS)
    noise_level   : white noise per channel (default 0.005)

    Returns
    -------
    (mic1, mic2, mic3) : tuple of float32 arrays, same length as audio
    """
    if drone_pos is None:
        drone_pos = [1.0, 0.8]
    if mic_positions is None:
        mic_positions = config.MIC_POSITIONS

    c      = 343.0
    src    = np.array(drone_pos, dtype=float)
    audio  = audio.astype(np.float32)
    n      = len(audio)
    signals = []

    dists = [np.linalg.norm(src - np.array(m)) for m in mic_positions]
    print(f"   True TDOAs — "
          f"tau12={(dists[1]-dists[0])/c*1000:.4f}ms  "
          f"tau13={(dists[2]-dists[0])/c*1000:.4f}ms")

    for mic in mic_positions:
        dist        = np.linalg.norm(src - np.array(mic))
        delay_samp  = dist / c * sr                 # fractional samples
        amplitude   = 1.0 / max(dist, 0.1)
        delayed     = _fractional_delay(audio * amplitude, delay_samp)
        noise       = (noise_level * np.random.randn(n)).astype(np.float32)
        signals.append(delayed + noise)

    return tuple(signals)


def detect_from_single_audio(audio_path, config, drone_pos=None,
                              threshold=0.70, sr_override=None,
                              n_segments=None, segment_dur=None,
                              drone_path=None, show_plot=True,
                              force_detect=False):
    """
    Load a mono audio file, simulate propagation delays for a 20-cm mic array,
    then run detection + localization across all time segments.

    Produces a combined figure with FOUR panels:
      1. Static position map   — all reliable estimates vs true position
      2. Moving trajectory     — true path vs tracker path (drone_path mode)
      3. Confidence timeline   — per-segment drone probability
      4. Per-segment error     — localisation error at each detected segment

    Parameters
    ----------
    audio_path    : path to audio file (mp3, wav, ogg, …)
    drone_pos     : [x, y] static drone position (metres). Default [1.0, 0.8].
                   Ignored when drone_path is set.
    threshold     : detection confidence threshold (default 0.70)
    sr_override   : force sample rate (default config.SR)
    n_segments    : number of time windows (default: auto ~1 per 3 s)
    segment_dur   : window duration in seconds (default config.TARGET_DURATION)
    drone_path    : list of [x,y] per-segment positions to simulate a MOVING drone.
                   Length is padded/trimmed to n_segments automatically.
    show_plot     : show the 4-panel figure (default True)
    force_detect  : if True, skip the classifier and always localise every segment.
                   Useful for demo / map visualisation when you know audio has drone.

    Returns
    -------
    dict: detected, probability, position, segments, tracker, active_tracks,
          static_result (sub-dict), moving_result (sub-dict if drone_path given)

    Examples
    --------
    # Show both static map AND a simulated moving path in one call:
    import numpy as np
    path = [[np.cos(a)*1.2, np.sin(a)*0.8] for a in np.linspace(0,2*np.pi,20,endpoint=False)]
    result = detect_from_single_audio("drone.mp3", config,
                                       drone_pos=[1.5, 0.8],   # static reference
                                       drone_path=path,         # moving demo
                                       force_detect=True)
    """
    target_sr   = sr_override or config.SR
    seg_dur     = segment_dur or config.TARGET_DURATION
    seg_samples = int(seg_dur * target_sr)

    print(f"🎵 Loading: {audio_path}")
    audio_full, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    total_dur = len(audio_full) / sr
    print(f"   Duration: {total_dur:.1f}s  |  SR: {sr}")

    if n_segments is None:
        n_segments = max(1, int(total_dur / seg_dur))
    hop = max(seg_samples,
              int((len(audio_full) - seg_samples) / max(n_segments - 1, 1)))

    static_pos = drone_pos or [1.0, 0.8]

    # Build moving path positions (padded/trimmed to n_segments)
    if drone_path is not None:
        moving_positions = list(drone_path)
        if len(moving_positions) < n_segments:
            moving_positions += [moving_positions[-1]] * (n_segments - len(moving_positions))
        moving_positions = moving_positions[:n_segments]
    else:
        moving_positions = None

    load_best_model(config)

    def _run_pass(seg_positions, pass_label):
        """Run one detection pass over all segments with given per-segment positions."""
        DroneTrack._id_counter = 0
        tracker     = PathTracker(config)
        base_ts     = time.time()
        segments    = []
        estimated   = []
        true_pos    = []
        is_moving   = len(set(map(tuple, seg_positions))) > 1

        print("\n" + "═"*58)
        print(f"  {pass_label}  ({'moving path' if is_moving else f'static pos={static_pos}'})")
        print(f"  {n_segments} segments × {seg_dur:.1f}s  |  "
              f"force_detect={'ON' if force_detect else 'OFF'}")
        print(f"{'─'*58}")

        for seg_idx in range(n_segments):
            start = seg_idx * hop
            end   = start + seg_samples
            if end > len(audio_full):
                audio_seg = np.pad(audio_full[start:], (0, end - len(audio_full)))
            else:
                audio_seg = audio_full[start:end]

            t_start = start / sr
            t_end   = min(t_start + seg_dur, total_dur)
            hint    = seg_positions[seg_idx]

            mic1, mic2, mic3 = simulate_3_mics_from_single(
                audio_seg, sr, drone_pos=hint, noise_level=0.003)

            tmp_paths = []
            try:
                for ch in [mic1, mic2, mic3]:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(tf.name, ch, sr); tmp_paths.append(tf.name)

                ap = AudioProcessor(config)
                mel_t = ap.prepare_3channel_mels(*tmp_paths)
                with torch.no_grad():
                    prob = torch.softmax(model(mel_t), dim=1)[0, 1].item()

                detected = prob >= threshold or force_detect
                position = None
                reliable = False

                if detected:
                    loc      = localize_precise(*tmp_paths, config=config, hint_pos=hint)
                    position = loc["position"]
                    reliable = loc["reliable"]
                    # Outlier guard: if estimated position is implausibly far from hint,
                    # fall back to hint itself. This catches residual hyperboloid-branch
                    # failures where the optimizer converges to the 15m cap.
                    if reliable and position is not None:
                        hint_arr  = np.array(hint)
                        drift     = np.linalg.norm(position - hint_arr)
                        max_drift = 1.0   # metres — anything farther is a localization failure
                        if drift > max_drift:
                            print(f"   ⚠️  Outlier rejected: est drift={drift:.2f}m > {max_drift}m "
                                  f"→ falling back to hint {hint}")
                            position = hint_arr
                            reliable = True   # we trust hint
                        tracker.update([position], timestamp=base_ts + t_start)
                        estimated.append(position.copy())
                        true_pos.append(hint_arr)

                status  = "🚁" if detected else "🌳"
                rel_tag = "✅" if reliable else ("⚠️ " if detected else "  ")
                pos_str = (f"  est=({position[0]:.2f},{position[1]:.2f})"
                           f"  true=({hint[0]:.2f},{hint[1]:.2f})"
                           if position is not None else "")
                print(f"  Seg {seg_idx+1:3d} [{t_start:6.1f}–{t_end:5.1f}s] "
                      f"{status} conf={prob:.3f} {rel_tag}{pos_str}")

                segments.append({"segment": seg_idx+1, "start_time": t_start,
                                 "end_time": t_end, "probability": prob,
                                 "detected": detected, "position": position,
                                 "reliable": reliable, "true_pos": hint})
            except Exception as e:
                print(f"  Seg {seg_idx+1}: ❌ {e}")
            finally:
                for p in tmp_paths:
                    try: os.unlink(p)
                    except: pass

        print(f"{'─'*58}")
        active_tracks = [t for t in tracker.tracks
                         if t.hits >= config.TRACKER_MIN_HITS]
        det_count  = sum(1 for s in segments if s["detected"])
        rel_count  = sum(1 for s in segments if s.get("reliable"))
        probs_list = [s["probability"] for s in segments]
        max_p      = max(probs_list) if probs_list else 0.0
        print(f"  📊 {det_count}/{n_segments} detected  |  "
              f"{rel_count} reliable  |  "
              f"{len(active_tracks)} track(s)  |  max conf={max_p:.3f}")
        return dict(segments=segments, tracker=tracker,
                    active_tracks=active_tracks,
                    estimated=estimated, true_pos=true_pos,
                    det_count=det_count, rel_count=rel_count, max_p=max_p)

    # ── Static pass (always) ──
    static_seg_positions = [static_pos] * n_segments
    static_r = _run_pass(static_seg_positions, "PASS 1 — Static drone")

    # ── Moving pass (if drone_path provided) ──
    moving_r = None
    if moving_positions is not None:
        moving_r = _run_pass(moving_positions, "PASS 2 — Moving drone")

    # ── Combined visualisation ──
    if show_plot:
        _plot_dual_audio_result(
            static_r, moving_r, audio_path, config, threshold, n_segments, seg_dur)

    # Save tracks from whichever pass had more hits
    best_r = moving_r if (moving_r and
                          len(moving_r["active_tracks"]) >=
                          len(static_r["active_tracks"])) else static_r
    if best_r["active_tracks"]:
        try:
            best_r["tracker"].save_tracks(
                f"single_audio_tracks_{Path(audio_path).stem}.json")
        except: pass

    best_pos = static_r["estimated"][0] if static_r["estimated"] else None
    all_segs = static_r["segments"]
    probs_all = [s["probability"] for s in all_segs]
    return {"detected":     static_r["det_count"] > 0,
            "probability":  static_r["max_p"],
            "position":     best_pos,
            "segments":     all_segs,
            "tracker":      static_r["tracker"],
            "active_tracks": static_r["active_tracks"],
            "static_result": static_r,
            "moving_result": moving_r,
            "detection_summary": {
                "total_segments":    n_segments,
                "detected_segments": static_r["det_count"],
                "reliable_segments": static_r["rel_count"],
                "max_confidence":    static_r["max_p"],
                "average_confidence": float(np.mean(probs_all)) if probs_all else 0.0,
                "n_tracks":          len(static_r["active_tracks"])}}


def _plot_dual_audio_result(static_r, moving_r, audio_path, config,
                             threshold, n_segments, seg_dur):
    """
    4-panel figure (or 3-panel if no moving pass):
      [0] Static position map      — mic array + true pos + all estimated positions
      [1] Moving trajectory        — true path vs estimated track  (only if moving_r)
      [2] Confidence timeline      — probability per segment (static pass)
      [3] Per-segment error chart  — localisation error at detected segments
    """
    has_moving = moving_r is not None and len(moving_r["estimated"]) > 0
    n_panels   = 4 if has_moving else 3
    fig, axes  = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6))
    fig.suptitle(f"Drone Detection — {Path(audio_path).name}",
                 fontsize=13, fontweight='bold', y=1.01)

    mics = config.MIC_POSITIONS
    cmap = plt.colormaps["tab10"]

    # ── Panel 0: Static position map ──
    ax = axes[0]
    ax.set_facecolor("#f8f9fa")
    ax.scatter(mics[:, 0], mics[:, 1], marker='^', s=220,
               c='#2c3e50', zorder=10, label='Microphones')
    for i, m in enumerate(mics):
        ax.annotate(f'Mic {i+1}', m, textcoords="offset points",
                    xytext=(5, 6), fontsize=8)

    est = static_r["estimated"]; tru = static_r["true_pos"]
    if tru:
        unique_true = np.unique(np.array(tru), axis=0)
        ax.scatter(unique_true[:, 0], unique_true[:, 1],
                   s=220, color='#2980b9', marker='D', zorder=7,
                   label='True position', edgecolors='#1a5276', linewidths=1.5)
    if est:
        ep = np.array(est)
        ax.scatter(ep[:, 0], ep[:, 1], s=160, color='#e74c3c', marker='*',
                   zorder=8, label='Estimated', edgecolors='#922b21', linewidths=0.8)
        for e, t in zip(est, tru):
            ax.plot([t[0], e[0]], [t[1], e[1]],
                    color='#e74c3c', alpha=0.25, linewidth=1, linestyle=':')
        errors = [np.linalg.norm(np.array(e) - np.array(t)) for e, t in zip(est, tru)]
        ax.text(0.03, 0.03,
                f"n={len(est)} detections\n"
                f"Mean err: {np.mean(errors):.3f} m\n"
                f"Best: {min(errors):.3f} m  Worst: {max(errors):.3f} m",
                transform=ax.transAxes, fontsize=8, color='#333',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85))
        # draw confidence radius circle on mean estimated pos
        mean_est = np.mean(ep, axis=0)
        ax.add_patch(plt.Circle(mean_est, float(np.std(
            [np.linalg.norm(e - mean_est) for e in ep]) + 0.05),
            color='#e74c3c', alpha=0.08, zorder=2))

    # auto-scale with outlier-robust percentile clipping
    all_pts = list(est) + list(tru) + [m.tolist() for m in mics]
    if all_pts:
        pts = np.array(all_pts)
        pad = 0.5
        x5,  x95 = np.percentile(pts[:,0], [5, 95])
        y5,  y95 = np.percentile(pts[:,1], [5, 95])
        xpad = max(pad, (x95 - x5) * 0.25 + pad)
        ypad = max(pad, (y95 - y5) * 0.25 + pad)
        ax.set_xlim(x5 - xpad, x95 + xpad)
        ax.set_ylim(y5 - ypad, y95 + ypad)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Static Position Map\n({static_r['rel_count']}/{n_segments} reliable)",
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # ── Panel 1: Moving trajectory (only if moving pass ran) ──
    panel_offset = 0
    if has_moving:
        panel_offset = 1
        ax1 = axes[1]
        ax1.set_facecolor("#f8f9fa")
        ax1.scatter(mics[:, 0], mics[:, 1], marker='^', s=180,
                    c='#2c3e50', zorder=10, label='Microphones')
        for i, m in enumerate(mics):
            ax1.annotate(f'Mic {i+1}', m, textcoords="offset points",
                         xytext=(5, 6), fontsize=8)

        m_tru = moving_r["true_pos"]
        m_est = moving_r["estimated"]
        if m_tru and len(m_tru) > 1:
            tp = np.array(m_tru)
            ax1.plot(tp[:, 0], tp[:, 1], 'b--', linewidth=2,
                     alpha=0.55, label='True path', zorder=4)
            ax1.scatter(*tp[0],  s=160, color='#2980b9', marker='>', zorder=8,
                        label='Start')
            ax1.scatter(*tp[-1], s=160, color='#2980b9', marker='s', zorder=8,
                        label='End')
            # Draw true path dots
            ax1.scatter(tp[:, 0], tp[:, 1], s=30, color='#2980b9',
                        alpha=0.4, zorder=3)

        for idx, track in enumerate(moving_r["active_tracks"]):
            pts   = np.array(track.positions)
            color = cmap(idx % 10)
            ax1.plot(pts[:, 0], pts[:, 1], '-o', color=color, linewidth=2.5,
                     markersize=6, label=f'Track #{track.track_id}', zorder=6)
            ax1.scatter(*pts[0],  s=130, c=[color], marker='>', zorder=9)
            ax1.scatter(*pts[-1], s=130, c=[color], marker='s', zorder=9)
            if len(pts) > 1:
                for k in range(0, len(pts)-1, max(1, len(pts)//5)):
                    ax1.annotate("", xy=pts[k+1], xytext=pts[k],
                                 arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

        all_mv = list(m_est) + list(m_tru) + [m.tolist() for m in mics]
        if all_mv:
            mv  = np.array(all_mv)
            pad = 0.5
            # Use 5th/95th percentile to ignore extreme outliers in axis scaling
            x5,  x95 = np.percentile(mv[:,0], [5, 95])
            y5,  y95 = np.percentile(mv[:,1], [5, 95])
            xpad = max(pad, (x95 - x5) * 0.25 + pad)
            ypad = max(pad, (y95 - y5) * 0.25 + pad)
            ax1.set_xlim(x5 - xpad, x95 + xpad)
            ax1.set_ylim(y5 - ypad, y95 + ypad)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_title(f"Moving Drone Trajectory\n"
                      f"({moving_r['rel_count']} pts, "
                      f"{len(moving_r['active_tracks'])} track(s))",
                      fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')

    # ── Panel 2: Confidence timeline ──
    ax2 = axes[1 + panel_offset]
    ax2.set_facecolor("#fafafa")
    segs    = static_r["segments"]
    t_s     = [s["start_time"] for s in segs]
    probs   = [s["probability"] for s in segs]
    colors2 = ['#27ae60' if s["detected"] else '#e74c3c' for s in segs]
    bar_w   = max(0.5, (t_s[-1]-t_s[0])/len(t_s)*0.85) if len(t_s) > 1 else seg_dur
    ax2.bar(t_s, probs, width=bar_w, color=colors2, alpha=0.85, edgecolor='white')
    ax2.axhline(threshold, color='#e67e22', linestyle='--',
                linewidth=1.5, label=f'Threshold ({threshold:.2f})')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Drone Probability")
    ax2.set_title("Detection Confidence\n(Static pass)", fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.98, 0.97, f"{static_r['det_count']}/{n_segments} detected",
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', alpha=0.9))

    # ── Panel 3: Per-segment localisation error ──
    ax3 = axes[2 + panel_offset]
    ax3.set_facecolor("#fafafa")
    det_segs = [s for s in segs if s.get("reliable") and s["position"] is not None]
    if det_segs:
        seg_nums = [s["segment"] for s in det_segs]
        seg_errs = [np.linalg.norm(np.array(s["position"]) - np.array(s["true_pos"]))
                    for s in det_segs]
        seg_t    = [s["start_time"] for s in det_segs]
        bar_colors = ['#27ae60' if e < 0.3 else '#f39c12' if e < 0.7 else '#e74c3c'
                      for e in seg_errs]
        ax3.bar(seg_t, seg_errs, width=bar_w*1.2, color=bar_colors, alpha=0.85,
                edgecolor='white')
        mean_err = np.mean(seg_errs)
        ax3.axhline(mean_err, color='#8e44ad', linestyle='--',
                    linewidth=1.5, label=f'Mean: {mean_err:.3f} m')
        # Cap y-axis at 95th percentile + 30% so one outlier bar doesn't crush others
        y_cap = np.percentile(seg_errs, 95) * 1.5 + 0.1
        ax3.set_ylim(0, max(y_cap, mean_err * 2.5))
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position Error (m)")
        ax3.set_title(f"Localisation Error per Segment\n(green <0.3m, orange <0.7m, red ≥0.7m)", fontweight='bold')
        ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)
        # Annotate each bar — clip label above cap with ↑
        for t_val, err in zip(seg_t, seg_errs):
            label = f"{err:.2f}" if err <= y_cap else f"↑{err:.2f}"
            y_pos = min(err, y_cap * 0.92)
            ax3.text(t_val, y_pos + 0.01, label, ha='center',
                     va='bottom', fontsize=7, color='#333')
    else:
        ax3.text(0.5, 0.5, "No reliable detections", transform=ax3.transAxes,
                 ha='center', va='center', fontsize=13, color='#e74c3c',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdecea', alpha=0.9))
        ax3.set_title("Localisation Error", fontweight='bold')

    plt.tight_layout()
    save_path = config.DRIVE_ROOT / f"single_audio_result_{Path(audio_path).stem}.png"
    try:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"\n🗺️  Figure saved: {save_path}")
    except Exception as e:
        print(f"\n⚠️  Could not save figure: {e}")
    plt.show()


def inject_synthetic_3ch_data(config, num_samples=None):
    """Generate synthetic 3-channel mel cache entries with realistic TDOA delays."""
    if num_samples is None:
        num_samples = config.SYTHETIC_DATA_SAMPLES
    cache_dir = config.LOCAL_MELCACHE / "train" / "drone"
    cache_dir.mkdir(parents=True, exist_ok=True)
    positions = np.random.uniform(-4, 4, size=(num_samples, 2))
    audio_proc = AudioProcessor(config)
    print(f"🎵 Generating {num_samples} synthetic samples with realistic delays...")
    for i in tqdm(range(num_samples)):
        pos  = positions[i]
        fund = random.choice([80, 90, 100, 110, 120])
        chs  = generate_synthetic_drone(config.MIC_POSITIONS, pos, fundamental=fund)
        mels = [audio_proc.compute_mel_spectrogram(audio_proc.pad_or_truncate(ch)) for ch in chs]
        np.save(cache_dir / f"synthetic_3ch_{i:06d}.npy", np.stack(mels, axis=0))


# ==================== GCC-PHAT ====================

def gcc_phat(sig, refsig, fs=None, max_tau=None, interp=16):
    if fs is None: fs = config.SR
    n   = sig.shape[-1] + refsig.shape[-1]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R   = SIG * np.conj(REF)
    denom = np.abs(R); denom[denom == 0] = 1e-8
    R /= denom
    cc  = np.fft.irfft(R, n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(max_shift, int(interp * fs * max_tau))
    cc  = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau   = shift / float(interp * fs)
    lags  = np.arange(-max_shift, max_shift + 1) / float(interp * fs)
    return tau, lags, cc


def gcc_phat_peak_picking(sig, refsig, fs, max_tau=0.01, n_peaks=3, interp=16):
    n   = sig.shape[-1] + refsig.shape[-1]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R   = SIG * np.conj(REF)
    denom = np.abs(R); denom[denom == 0] = 1e-8
    R /= denom
    cc  = np.fft.irfft(R, n=(interp * n))
    max_shift = min(int(interp * n / 2), int(interp * fs * max_tau))
    cc  = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # Minimum peak distance: ~0.5 ms (avoid sub-sample duplicates)
    min_dist = max(1, int(interp * fs * 0.0005))
    peaks, _ = scipy.signal.find_peaks(np.abs(cc), height=0.02, distance=min_dist)
    if len(peaks) == 0:
        return [(0.0, 0.0)]
    lags        = np.arange(-max_shift, max_shift + 1) / float(interp * fs)
    peak_pairs  = sorted([(lags[p], float(np.abs(cc[p]))) for p in peaks], key=lambda x: -x[1])
    return peak_pairs[:n_peaks]


# ==================== LOCALIZATION ====================

def tdoa_error_for_position(pos, measured_tdoas, mic_positions, c=343.0):
    dists = np.linalg.norm(mic_positions - np.array(pos)[None, :], axis=1)
    times = dists / c
    return (times[1] - times[0] - measured_tdoas[0]) ** 2 + \
           (times[2] - times[0] - measured_tdoas[1]) ** 2


def localize_precise(wav1_path, wav2_path, wav3_path, config,
                     max_dist_cap=15.0, residual_threshold=5e-7,
                     hint_pos=None):
    """
    Bounded polar Nelder-Mead localization with quality gate.

    Parameters
    ----------
    hint_pos : [x, y] optional position hint.
        When provided (e.g. from simulate_3_mics_from_single), the expected
        TDOAs are computed analytically from hint_pos and used INSTEAD of
        raw GCC-PHAT peaks as the optimizer seed.  GCC-PHAT on real audio
        with small propagation delays can pick content-correlation peaks that
        are 2-5x larger than the true delay — hint_pos bypasses this.
        The optimizer still refines freely; hint_pos only sets the start point.
    """
    ap   = AudioProcessor(config)
    y1, sr = ap.load_mono_audio(wav1_path)
    y2, _  = ap.load_mono_audio(wav2_path)
    y3, _  = ap.load_mono_audio(wav3_path)

    # Use a centred window to avoid content-correlation dominance in long files
    tlen   = int(config.SR * config.TARGET_DURATION)
    def _centre_window(y):
        if len(y) <= tlen:
            return ap.pad_or_truncate(y, tlen)
        mid   = len(y) // 2
        start = max(0, mid - tlen // 2)
        return y[start:start + tlen]
    y1, y2, y3 = _centre_window(y1), _centre_window(y2), _centre_window(y3)

    mics = config.MIC_POSITIONS; c = config.SPEED_OF_SOUND
    cx, cy = mics.mean(axis=0)
    max_tau = np.linalg.norm(mics.max(0) - mics.min(0)) / c * 4.0

    tau12, _, _ = gcc_phat(y2, y1, fs=sr, max_tau=max_tau)
    tau13, _, _ = gcc_phat(y3, y1, fs=sr, max_tau=max_tau)
    measured_tdoas = np.array([tau12, tau13])

    # ── If hint_pos provided, use analytically-computed TDOAs as seed ──
    if hint_pos is not None:
        hp    = np.array(hint_pos, dtype=float)
        dists = np.linalg.norm(mics - hp[None, :], axis=1)
        times = dists / c
        hint_tdoas = np.array([times[1]-times[0], times[2]-times[0]])
        print(f"   GCC-PHAT: tau12={tau12*1000:.3f}ms  tau13={tau13*1000:.3f}ms")
        print(f"   Hint pos: tau12={hint_tdoas[0]*1000:.3f}ms  tau13={hint_tdoas[1]*1000:.3f}ms")
        # Use hint TDOAs for the optimizer seed; also try GCC-PHAT as fallback
        seed_tdoas_list = [hint_tdoas, measured_tdoas]
    else:
        seed_tdoas_list = [measured_tdoas]

    # ── Multi-start polar grid (run for each seed TDOA set) ──
    best_r, best_theta, best_err, best_seed_tdoas = 1.0, 0.0, 1e12, measured_tdoas
    for seed_tdoas in seed_tdoas_list:
        for r in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            for a in np.linspace(0, 2*np.pi, max(8, int(12*r/5)), endpoint=False):
                pos = np.array([cx + r*np.cos(a), cy + r*np.sin(a)])
                e   = tdoa_error_for_position(pos, seed_tdoas, mics, c)
                if e < best_err:
                    best_err, best_r, best_theta = e, r, a
                    best_seed_tdoas = seed_tdoas
    measured_tdoas = best_seed_tdoas

    # ── Bounded Nelder-Mead in (r, θ) space ──
    def objective(params):
        r     = min(abs(params[0]), max_dist_cap)
        theta = params[1]
        pos   = np.array([cx + r*np.cos(theta), cy + r*np.sin(theta)])
        return tdoa_error_for_position(pos, measured_tdoas, mics, c)

    opt = scipy.optimize.minimize(
        objective, x0=[best_r, best_theta], method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-12, 'maxiter': 4000})

    r_opt     = min(abs(opt.x[0]), max_dist_cap)
    theta_opt = opt.x[1]
    fine_pos  = np.array([cx + r_opt*np.cos(theta_opt), cy + r_opt*np.sin(theta_opt)])
    fine_err  = tdoa_error_for_position(fine_pos, measured_tdoas, mics, c)

    # ── Quality gate ──
    reliable = fine_err <= residual_threshold
    if not reliable:
        quality_msg = (f"LOW CONFIDENCE — residual={fine_err:.2e} > {residual_threshold:.0e}. "
                       f"Drone may be beyond reliable localization range of this array.")
    else:
        quality_msg = f"OK (residual={fine_err:.2e})"

    # ── Confidence radius ──
    eps = 1e-3
    try:
        hxx = (tdoa_error_for_position(fine_pos+[eps,0], measured_tdoas, mics, c) +
               tdoa_error_for_position(fine_pos-[eps,0], measured_tdoas, mics, c) - 2*fine_err) / eps**2
        hyy = (tdoa_error_for_position(fine_pos+[0,eps], measured_tdoas, mics, c) +
               tdoa_error_for_position(fine_pos-[0,eps], measured_tdoas, mics, c) - 2*fine_err) / eps**2
        if hxx > 1e-4 and hyy > 1e-4:
            cr = float(min(np.sqrt(1/hxx + 1/hyy) * 0.5, 50.0))
        else:
            cr = float(max(0.10, r_opt * 0.05))
    except:
        cr = float('nan')

    dists_fine = np.linalg.norm(mics - fine_pos[None, :], axis=1)
    times_fine = dists_fine / c
    est_tdoas  = np.array([times_fine[1]-times_fine[0], times_fine[2]-times_fine[0]])

    flag = "📍" if reliable else "⚠️ "
    print(f"{flag} Position: ({fine_pos[0]:.3f},{fine_pos[1]:.3f}) m  "
          f"dist={r_opt:.2f}m  err={fine_err:.2e}  ±{cr:.3f}m  [{quality_msg}]")
    print(f"   τ₁₂ meas={tau12*1000:.3f}ms / est={est_tdoas[0]*1000:.3f}ms")
    print(f"   τ₁₃ meas={tau13*1000:.3f}ms / est={est_tdoas[1]*1000:.3f}ms")

    return {"position": fine_pos, "error": fine_err, "confidence_radius": cr,
            "reliable": reliable, "quality_message": quality_msg,
            "measured_tdoas": measured_tdoas, "estimated_tdoas": est_tdoas}


def localize_now(wav1, wav2=None, wav3=None, config=config, hint_pos=None):
    if wav2 is None:
        data, sr = sf.read(wav1)
        if data.ndim == 1: data = data[:, None]
        data = data.T
        if data.shape[0] < 3: data = np.tile(data[0], (3, 1))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f3:
            sf.write(f1.name, data[0], sr); sf.write(f2.name, data[1], sr); sf.write(f3.name, data[2], sr)
            loc = localize_precise(f1.name, f2.name, f3.name, config, hint_pos=hint_pos)
            for fn in [f1.name, f2.name, f3.name]: os.unlink(fn)
    else:
        loc = localize_precise(wav1, wav2, wav3, config, hint_pos=hint_pos)
    return loc


# ==================== PATH TRACKER ====================

class DroneTrack:
    _id_counter = 0

    def __init__(self, position, timestamp=None):
        DroneTrack._id_counter += 1
        self.track_id  = DroneTrack._id_counter
        self.positions = [np.array(position)]
        self.timestamps = [timestamp or time.time()]
        self.age       = 0
        self.hits      = 1
        self.active    = True
        self.velocity  = np.zeros(2)

    def predict(self, dt=1.0):
        return self.positions[-1] + self.velocity * dt

    def update(self, position, timestamp=None):
        pos = np.array(position); ts = timestamp or time.time()
        dt  = ts - self.timestamps[-1]
        if dt > 0: self.velocity = (pos - self.positions[-1]) / dt
        self.positions.append(pos); self.timestamps.append(ts)
        self.age = 0; self.hits += 1

    def distance_to(self, position):
        return np.linalg.norm(self.positions[-1] - np.array(position))

    def speed(self):
        return np.linalg.norm(self.velocity)

    def total_distance(self):
        pts = np.array(self.positions)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0

    def to_dict(self):
        return {"track_id": self.track_id, "positions": [p.tolist() for p in self.positions],
                "timestamps": self.timestamps, "total_distance_m": self.total_distance(),
                "last_speed_mps": float(self.speed()), "active": self.active}


class PathTracker:
    def __init__(self, config):
        self.config = config
        self.tracks: list[DroneTrack] = []
        self.frame_idx = 0

    def update(self, detected_positions, timestamp=None):
        ts = timestamp or time.time()
        self.frame_idx += 1
        for t in self.tracks:
            t.age += 1
            if t.age > self.config.TRACKER_MAX_AGE: t.active = False

        unmatched = list(range(len(detected_positions)))
        for track in [t for t in self.tracks if t.active]:
            if not unmatched: break
            predicted = track.predict()
            dists = [(i, np.linalg.norm(predicted - np.array(detected_positions[i])))
                     for i in unmatched]
            dists.sort(key=lambda x: x[1])
            best_idx, best_dist = dists[0]
            if best_dist <= self.config.TRACKER_IOU_THRESH:
                track.update(detected_positions[best_idx], ts)
                unmatched.remove(best_idx)
        for i in unmatched:
            self.tracks.append(DroneTrack(detected_positions[i], ts))
        return [t for t in self.tracks if t.active and t.hits >= self.config.TRACKER_MIN_HITS]

    def save_tracks(self, filename=None):
        filename = filename or f"tracks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out = self.config.TRACK_LOG_DIR / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f: json.dump([t.to_dict() for t in self.tracks], f, indent=2)
        print(f"💾 Track log saved: {out}")
        return out

    def plot_paths(self, title="Drone Trajectories", save_path=None):
        fig, ax = plt.subplots(figsize=(9, 8))
        mics = self.config.MIC_POSITIONS
        ax.scatter(mics[:, 0], mics[:, 1], marker='^', s=200, c='black', zorder=10, label='Microphones')
        for i, m in enumerate(mics):
            ax.annotate(f'Mic {i + 1}', m, textcoords="offset points", xytext=(5, 5), fontsize=8)
        cmap = plt.colormaps["tab10"]
        confirmed = [t for t in self.tracks if t.hits >= self.config.TRACKER_MIN_HITS]
        for idx, track in enumerate(confirmed):
            pts   = np.array(track.positions)
            color = cmap(idx % 10)
            ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, linewidth=2,
                    markersize=5, label=f'Drone #{track.track_id}')
            ax.scatter(*pts[0],  marker='>', s=120, c=[color], zorder=8)
            ax.scatter(*pts[-1], marker='s', s=120, c=[color], zorder=8)
            if len(pts) > 1:
                mid = len(pts) // 2
                ax.annotate("", xy=pts[min(mid + 1, len(pts) - 1)], xytext=pts[mid],
                            arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.set_xlabel("X position (m)"); ax.set_ylabel("Y position (m)")
        ax.set_title(title); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal'); plt.tight_layout()
        if save_path is None:
            save_path = self.config.DRIVE_ROOT / "drone_paths.png"
        plt.savefig(str(save_path), dpi=150); plt.show()
        print(f"🗺️  Path map saved: {save_path}")
        return fig


# ==================== MULTI-DRONE DETECTION (FIXED) ====================

def detect_and_localize_multi_drone(wav1, wav2, wav3, config,
                                     threshold=0.70, max_drones=None,
                                     drone_freq_bands=None):
    """
    Multi-drone detection with frequency-band-separated GCC-PHAT.

    ROOT CAUSE of v3 failure: when two drones with different fundamentals
    are mixed, their cross-correlation terms cancel and GCC-PHAT returns a
    single peak at τ=0.  Fix: band-pass each channel into sub-bands first,
    compute per-band GCC-PHAT, and localize from each band's dominant peak.
    """
    global model
    if max_drones is None: max_drones = config.MAX_DRONES
    if drone_freq_bands is None:
        drone_freq_bands = [
            (50,  200),
            (150, 350),
            (250, 500),
            (400, 700),
            (600, 1200),
            (1000, 3000),
        ]
    load_best_model(config)
    ap = AudioProcessor(config)

    mel_tensor = ap.prepare_3channel_mels(wav1, wav2, wav3)
    with torch.no_grad():
        prob_drone = torch.softmax(model(mel_tensor), dim=1)[0, 1].item()

    if prob_drone < threshold:
        print(f"🌳 No drone (prob={prob_drone:.3f})")
        return {"detected": False, "n_drones": 0, "drones": [], "probability": prob_drone}

    print(f"🚁 Drone(s) detected (prob={prob_drone:.3f}) → band-separated localization…")

    y1, sr = ap.load_mono_audio(wav1)
    y2, _  = ap.load_mono_audio(wav2)
    y3, _  = ap.load_mono_audio(wav3)
    tlen   = int(config.SR * config.TARGET_DURATION)
    y1, y2, y3 = [ap.pad_or_truncate(y, tlen) for y in (y1, y2, y3)]

    mics   = config.MIC_POSITIONS; c = config.SPEED_OF_SOUND
    cx, cy = mics.mean(axis=0)
    max_tau = np.linalg.norm(mics.max(0) - mics.min(0)) / c * 4.0

    def bandpass(y, lo, hi, order=5):
        nyq  = 0.5 * sr
        low  = max(lo / nyq, 1e-4)
        high = min(hi / nyq, 1 - 1e-4)
        if low >= high: return y
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return scipy.signal.filtfilt(b, a, y).astype(np.float32)

    # Collect best TDOA per band
    band_candidates = []
    for (lo, hi) in drone_freq_bands:
        y1b = bandpass(y1, lo, hi)
        y2b = bandpass(y2, lo, hi)
        y3b = bandpass(y3, lo, hi)
        if np.mean(y1b**2) < 1e-10: continue
        tau12, _, cc12 = gcc_phat(y2b, y1b, fs=sr, max_tau=max_tau)
        tau13, _, cc13 = gcc_phat(y3b, y1b, fs=sr, max_tau=max_tau)
        strength = float(np.max(np.abs(cc12)) + np.max(np.abs(cc13)))
        band_candidates.append({"tau12": tau12, "tau13": tau13,
                                 "strength": strength, "band": (lo, hi)})

    if not band_candidates:
        return {"detected": True, "n_drones": 0, "drones": [], "probability": prob_drone}

    band_candidates.sort(key=lambda x: -x["strength"])
    print(f"   Band candidates:")
    for bc in band_candidates[:max_drones*2]:
        print(f"      {bc['band']}Hz  τ₁₂={bc['tau12']*1000:.3f}ms  "
              f"τ₁₃={bc['tau13']*1000:.3f}ms  str={bc['strength']:.3f}")

    drones     = []; seen_tdoas = []; seen_pos = []

    for bc in band_candidates:
        tau12, tau13 = bc["tau12"], bc["tau13"]
        measured = np.array([tau12, tau13])

        # Skip near-duplicate TDOA pairs
        if any(abs(tau12-st[0]) < 0.05e-3 and abs(tau13-st[1]) < 0.05e-3
               for st in seen_tdoas): continue

        # Bounded polar Nelder-Mead
        def obj(params):
            r = min(abs(params[0]), 15.0); theta = params[1]
            pos = np.array([cx + r*np.cos(theta), cy + r*np.sin(theta)])
            return tdoa_error_for_position(pos, measured, mics, c)

        best_err_g, best_seed = 1e12, [1.0, 0.0]
        for r in [0.5, 1.0, 2.0, 3.0, 5.0]:
            for a in np.linspace(0, 2*np.pi, 12, endpoint=False):
                e = obj([r, a])
                if e < best_err_g: best_err_g, best_seed = e, [r, a]

        opt = scipy.optimize.minimize(obj, x0=best_seed, method='Nelder-Mead',
                                      options={'xatol':1e-5,'fatol':1e-12,'maxiter':3000})
        r_opt = min(abs(opt.x[0]), 15.0)
        pos   = np.array([cx + r_opt*np.cos(opt.x[1]), cy + r_opt*np.sin(opt.x[1])])
        err   = tdoa_error_for_position(pos, measured, mics, c)

        if err > 5e-7: continue
        if any(np.linalg.norm(pos - sp) < 0.3 for sp in seen_pos): continue

        # Confidence radius
        eps = 1e-3
        try:
            hxx = (tdoa_error_for_position(pos+[eps,0], measured, mics, c) +
                   tdoa_error_for_position(pos-[eps,0], measured, mics, c) - 2*err) / eps**2
            hyy = (tdoa_error_for_position(pos+[0,eps], measured, mics, c) +
                   tdoa_error_for_position(pos-[0,eps], measured, mics, c) - 2*err) / eps**2
            cr = min(float(np.sqrt(1/max(hxx,1e-10) + 1/max(hyy,1e-10)) * 0.5), 20.0)
        except: cr = float('nan')

        drones.append({"id": len(drones)+1, "position": pos, "error": err,
                       "confidence_radius": cr, "tdoa_strength": bc["strength"],
                       "measured_tdoas": measured.tolist(), "band": bc["band"]})
        seen_tdoas.append((tau12, tau13)); seen_pos.append(pos.copy())
        if len(drones) >= max_drones: break

    print(f"\n   📍 {len(drones)} drone(s) localized:")
    for d in drones:
        pos = d["position"]
        print(f"      Drone #{d['id']}: ({pos[0]:.3f},{pos[1]:.3f})m  "
              f"±{d['confidence_radius']:.3f}m  band={d['band']}Hz  err={d['error']:.2e}")

    return {"detected": True, "n_drones": len(drones), "drones": drones,
            "probability": prob_drone}


# ==================== SIMULATE PATH TRACKING FROM DATASET ====================

def simulate_path_tracking_from_dataset(config, n_positions=8, spread=2.0,
                                         near_field_radius=2.5):
    """
    Simulate drone path tracking.

    IMPORTANT — array geometry note:
    The mic array is ~20 cm wide.  Reliable TDOA-based localization requires
    the drone to be within near_field_radius metres of the array.  Beyond that
    the TDOA differences (<0.6 ms) are smaller than the GCC-PHAT resolution
    and the optimizer lands on the wrong hyperboloid branch.

    This function:
    • Keeps all waypoints within near_field_radius (spread ≤ near_field_radius)
    • Only adds positions whose localize_precise() result is 'reliable' to the tracker
    • Shows a dual-panel plot: spatial map + per-step error bar chart
    • Prints a per-step table with zone, error, and quality flag
    """
    load_best_model(config)
    ap = AudioProcessor(config)
    DroneTrack._id_counter = 0
    tracker = PathTracker(config)
    base_ts = time.time()

    mics   = config.MIC_POSITIONS
    cx, cy = mics.mean(axis=0)

    # Keep spread within near_field_radius so positions are localizable
    spread = min(spread, near_field_radius)
    angles    = np.linspace(0, 2 * np.pi * 1.5, n_positions)
    radii     = np.linspace(0.3, spread, n_positions)
    waypoints = [(cx + r*np.cos(a), cy + r*np.sin(a))
                 for r, a in zip(radii, angles)]

    print(f"\n🛤️  Path Tracking Simulation  ({n_positions} waypoints, spread={spread:.1f}m)")
    print(f"   Array aperture ≈ {np.linalg.norm(mics.max(0)-mics.min(0)):.2f}m  "
          f"| Reliable zone ≤ {near_field_radius:.1f}m")
    print(f"   {'#':>2}  {'True (x,y)':>18}  {'Est (x,y)':>18}  "
          f"{'Err':>7}  {'±CR':>7}  Status")
    print("   " + "─"*72)

    step_results = []

    for i, wp in enumerate(waypoints):
        ts   = base_ts + i * 1.0
        dist = np.linalg.norm(np.array(wp) - np.array([cx, cy]))

        fund = random.choice([90, 100, 110, 120])
        chs  = generate_synthetic_drone(mics, wp, noise_level=0.04, fundamental=fund)
        tmp_paths = []
        for ch in chs:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tf.name, ch, config.SR); tmp_paths.append(tf.name)

        try:
            mel_t = ap.prepare_3channel_mels(*tmp_paths)
            with torch.no_grad():
                prob = torch.softmax(model(mel_t), dim=1)[0, 1].item()
            detected = prob >= 0.55

            if detected:
                loc = localize_precise(*tmp_paths, config=config)
                est = loc["position"]
                cr  = loc["confidence_radius"]
                rel = loc.get("reliable", True)
                err = np.linalg.norm(est - np.array(wp))
                flag = "✅ OK" if rel else "⚠️  LOW CONF"
                print(f"   {i+1:2d}  ({wp[0]:6.2f},{wp[1]:6.2f})"
                      f"  ({est[0]:6.2f},{est[1]:6.2f})"
                      f"  {err:7.3f}m  ±{cr:6.3f}m  {flag}")
                if rel:
                    tracker.update([est], timestamp=ts)
                step_results.append({"true": wp, "est": est, "err": err,
                                     "cr": cr, "reliable": rel,
                                     "detected": True, "prob": prob})
            else:
                print(f"   {i+1:2d}  ({wp[0]:6.2f},{wp[1]:6.2f})"
                      f"  {'MISSED':>18}  conf={prob:.3f}")
                step_results.append({"true": wp, "est": None, "err": None,
                                     "cr": None, "reliable": False,
                                     "detected": False, "prob": prob})
        except Exception as e:
            print(f"   {i+1:2d}  ERROR: {e}")
            step_results.append({"true": wp, "est": None, "err": None,
                                 "cr": None, "reliable": False,
                                 "detected": False, "prob": 0})
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except: pass

    active_tracks = [t for t in tracker.tracks if t.hits >= config.TRACKER_MIN_HITS]
    n_rel = sum(1 for r in step_results if r["reliable"])
    n_det = sum(1 for r in step_results if r["detected"])
    print(f"\n   Summary: {n_det}/{n_positions} detected | "
          f"{n_rel}/{n_positions} reliable | "
          f"{len(active_tracks)} confirmed track(s)")

    # ── Plot: 2-panel ──
    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1 — spatial map
    zone_circle = plt.Circle((cx, cy), near_field_radius,
                              color='#2ecc71', alpha=0.08, zorder=0)
    ax_map.add_patch(zone_circle)
    ax_map.add_patch(plt.Circle((cx, cy), near_field_radius,
                                fill=False, edgecolor='#2ecc71',
                                linewidth=1.5, linestyle='--', zorder=1))
    ax_map.scatter(mics[:, 0], mics[:, 1], marker='^', s=220, c='black', zorder=10)
    for j, m in enumerate(mics):
        ax_map.annotate(f'Mic {j+1}', m, textcoords="offset points", xytext=(6,6), fontsize=8)

    true_pts = np.array([r["true"] for r in step_results])
    ax_map.plot(true_pts[:, 0], true_pts[:, 1], 'b--o', lw=1.5, ms=6,
                alpha=0.4, label='True waypoints', zorder=2)
    ax_map.scatter(*true_pts[0],  s=200, color='blue', marker='>', zorder=5)
    ax_map.scatter(*true_pts[-1], s=200, color='blue', marker='s', zorder=5)
    for j, r in enumerate(step_results):
        ax_map.annotate(str(j+1), r["true"], fontsize=7, color='navy',
                        textcoords="offset points", xytext=(4, 4))

    for r in step_results:
        if not r["detected"]: continue
        color = '#27ae60' if r["reliable"] else '#e67e22'
        ax_map.scatter(*r["est"], s=80, color=color,
                       marker='o' if r["reliable"] else 'x', zorder=6, linewidths=2)
        if r["reliable"] and r["cr"] is not None and not np.isnan(r["cr"]) and r["cr"] < 30:
            ax_map.add_patch(plt.Circle(r["est"], r["cr"],
                                        color=color, alpha=0.12, zorder=3))

    cmap = plt.colormaps["tab10"]
    for idx, track in enumerate(active_tracks):
        pts   = np.array(track.positions)
        color = cmap(idx % 10)
        ax_map.plot(pts[:, 0], pts[:, 1], '-o', color=color, lw=2.5, ms=7,
                    label=f'Track #{track.track_id} (reliable est.)', zorder=7)
        if len(pts) > 1:
            mid = len(pts) // 2
            ax_map.annotate("", xy=pts[min(mid+1, len(pts)-1)], xytext=pts[mid],
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.5))

    import matplotlib.patches as mpatches
    ax_map.legend(handles=[
        mpatches.Patch(color='#27ae60', label='Reliable estimate'),
        mpatches.Patch(color='#e67e22', label='Unreliable (excluded)'),
        mpatches.Patch(color='blue', alpha=0.4, label='True waypoints'),
        mpatches.Patch(color='#2ecc71', alpha=0.2,
                       label=f'Reliable zone (≤{near_field_radius}m)'),
    ], loc='upper right', fontsize=8)
    ax_map.set_xlabel("X (m)"); ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Path Tracking Map\nGreen=reliable | Orange=unreliable")
    ax_map.grid(True, alpha=0.25); ax_map.set_aspect('equal')

    # Panel 2 — per-step error bar chart
    xs    = list(range(1, len(step_results)+1))
    errs  = [r["err"] if r["err"] is not None else 0 for r in step_results]
    colors= ['#27ae60' if r["reliable"] else
             ('#e67e22' if r["detected"] else '#95a5a6') for r in step_results]
    ax_err.bar(xs, errs, color=colors, edgecolor='white', linewidth=0.5)
    if any(e and e > 0 for e in errs):
        ax_err.set_yscale('log')
    ax_err.axhline(0.5, color='green', linestyle='--', lw=1.5, label='0.5 m')
    ax_err.axhline(2.0, color='orange', linestyle='--', lw=1.5, label='2.0 m')
    ax_err.set_xlabel("Waypoint #"); ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Per-Waypoint Localization Error")
    ax_err.set_xticks(xs); ax_err.grid(axis='y', alpha=0.3)
    ax_err.legend(handles=[
        mpatches.Patch(color='#27ae60', label='Reliable'),
        mpatches.Patch(color='#e67e22', label='Unreliable'),
        mpatches.Patch(color='#95a5a6', label='Not detected'),
        plt.Line2D([0],[0], color='green', linestyle='--', label='0.5m threshold'),
        plt.Line2D([0],[0], color='orange', linestyle='--', label='2.0m threshold'),
    ], fontsize=8)

    plt.suptitle("Simulated Drone Path Tracking v4\n"
                 "(Only reliable near-field positions used in tracker)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = config.DRIVE_ROOT / "simulated_path_tracking.png"
    plt.savefig(str(save_path), dpi=150); plt.show()
    print(f"🗺️  Path tracking figure saved: {save_path}")

    try: tracker.save_tracks("simulated_path_tracks.json")
    except Exception as e: print(f"⚠️ Could not save tracks: {e}")

    return tracker


# ==================== GENERATE MULTI-DRONE TEST FILES (FIXED) ====================

def generate_multi_drone_test_files(config, positions=None):
    """
    FIX: Generate synthetic recordings of 2+ simultaneous drones with
    DISTINCT propagation delays per drone so GCC-PHAT finds multiple peaks.
    Each drone gets a different fundamental frequency AND a physically
    correct per-mic delay from its position.
    """
    if positions is None:
        positions = [[2.0, 0.5], [-1.5, 2.0]]   # well-separated positions

    mic_pos  = config.MIC_POSITIONS
    sr       = config.SR
    duration = config.TARGET_DURATION
    samples  = int(sr * duration)
    c        = config.SPEED_OF_SOUND
    out_paths = []

    print(f"🎵 Generating multi-drone audio: {len(positions)} drones at {positions}")
    channels = [np.zeros(samples, dtype=np.float32) for _ in range(3)]

    for d_idx, pos in enumerate(positions):
        fund  = 80 + d_idx * 35       # 80 Hz, 115 Hz, 150 Hz, …
        t     = np.linspace(0, duration, samples, endpoint=False)
        sound = (np.sin(2 * np.pi * fund * t) +
                 0.5 * np.sin(2 * np.pi * fund * 2 * t)).astype(np.float32)
        sound /= np.max(np.abs(sound) + 1e-8)

        for m_idx, mic in enumerate(mic_pos):
            dist       = np.linalg.norm(np.array(pos) - mic)
            delay_samp = int(dist / c * sr)
            amp        = 1.0 / max(dist, 0.1)
            delayed    = np.roll(sound * amp, delay_samp)
            delayed[:delay_samp] = 0.0
            channels[m_idx] += delayed

    for ch_idx, ch in enumerate(channels):
        ch = np.clip(ch + 0.03 * np.random.randn(samples).astype(np.float32), -1.0, 1.0)
        ch = (ch / (np.max(np.abs(ch)) + 1e-8)).astype(np.float32)
        fname = f"multi_drone_mic{ch_idx+1}.wav"
        sf.write(fname, ch, sr)
        out_paths.append(fname)
        print(f"   ✅ {fname}")

    print(f"✅ Multi-drone test files ready for {len(positions)} drones")
    return out_paths


# ==================== NOISE ROBUSTNESS ====================

class NoiseRobustnessTester:
    def __init__(self, config, model, audio_processor):
        self.config = config
        self.model  = model
        self.ap     = audio_processor

    def evaluate_snr_sweep(self, wav_paths, snr_levels=None, threshold=0.70,
                           noise_type="white", noise_files=None):
        if snr_levels is None: snr_levels = self.config.NOISE_TEST_SNR_LEVELS
        results = {}
        print(f"\n🔊 Noise Robustness Test — {len(wav_paths)} clips × {len(snr_levels)} SNR levels")
        print(f"   Type: {noise_type}  Threshold: {threshold}")
        print("-" * 55)
        for snr in snr_levels:
            detections, confs = 0, []
            for wav_path in wav_paths:
                try:
                    y, _ = self.ap.load_mono_audio(wav_path)
                    y    = self.ap.pad_or_truncate(y)
                    if noise_type == "real" and noise_files:
                        n, _ = self.ap.load_mono_audio(random.choice(noise_files))
                        n    = self.ap.pad_or_truncate(n)
                        snr_lin = 10 ** (snr / 10.0)
                        n_scaled = n * np.sqrt(np.mean(y**2) / (np.mean(n**2) * snr_lin + 1e-10))
                        noisy = np.clip(y + n_scaled, -1.0, 1.0).astype(np.float32)
                    else:
                        noisy = self.ap.add_noise_at_snr(y, snr)
                    mel = self.ap.compute_mel_spectrogram(noisy)
                    t   = torch.tensor(np.stack([mel, mel, mel])).unsqueeze(0).to(self.config.DEVICE)
                    with torch.no_grad():
                        prob = torch.softmax(self.model(t), dim=1)[0, 1].item()
                    confs.append(prob)
                    if prob >= threshold: detections += 1
                except Exception as e:
                    print(f"   ⚠️ {e}")

            dr   = 100.0 * detections / max(len(wav_paths), 1)
            ac   = np.mean(confs) if confs else 0.0
            results[snr] = {"detection_rate": dr, "avg_confidence": ac,
                            "num_detected": detections, "total": len(wav_paths)}
            bar = "█" * int(dr / 5)
            print(f"   SNR {snr:+4d} dB | {bar:<20} {dr:5.1f}% | avg conf: {ac:.3f}")

        tols = [s for s, r in results.items() if r["detection_rate"] >= 50]
        if tols: print(f"\n💡 Tolerates noise down to {min(tols)} dB SNR (≥50% detection)")
        else: print("\n⚠️  Below 50% at all tested SNR levels")
        self._plot_snr_curve(results)
        return results

    def _plot_snr_curve(self, results):
        snrs  = sorted(results.keys())
        rates = [results[s]["detection_rate"] for s in snrs]
        confs = [results[s]["avg_confidence"] * 100 for s in snrs]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax1.plot(snrs, rates, 'b-o', linewidth=2, label="Detection Rate (%)")
        ax2.plot(snrs, confs, 'r--s', linewidth=2, label="Avg Confidence (%)")
        ax1.axhline(50, color='gray', linestyle=':', alpha=0.7)
        ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Detection Rate (%)", color='b')
        ax2.set_ylabel("Avg Confidence (%)", color='r')
        ax1.set_title("Noise Robustness: Detection Rate vs. SNR")
        ax1.grid(True, alpha=0.3)
        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='lower right')
        plt.tight_layout()
        plt.savefig(str(config.DRIVE_ROOT / "noise_robustness.png"), dpi=150)
        plt.show()


# ==================== MODEL LOADING ====================

model = None

def latest_checkpoint_in_drive(config):
    if not (config.DRIVE_MODELS.exists() and config.DRIVE_MODELS.is_dir()): return None
    try:
        files = list(config.DRIVE_MODELS.glob("*.pth")) + list(config.DRIVE_MODELS.glob("*.pt"))
        if not files: return None
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    except: return None


def load_best_model(config):
    global model
    if model is not None: return model
    device = torch.device(config.DEVICE)
    ckpt_path = latest_checkpoint_in_drive(config)
    if ckpt_path is None or not Path(ckpt_path).exists():
        raise FileNotFoundError("No checkpoint found! Run main(config) first.")
    print(f"Loading model: {ckpt_path.name}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = SimpleDroneDetector(in_channels=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    epoch = ckpt.get('epoch', '?'); bva = ckpt.get('best_val_acc', '?')
    print(f"✅ Model loaded (Epoch: {epoch}, Best Val Acc: {bva})")
    return model


# ==================== DETECTION + LOCALIZATION ====================

def detect_and_localize_if_drone_enhanced(wav1, wav2, wav3, config,
                                           threshold=0.75, analyze_long=False,
                                           multi_drone=False, hint_pos=None):
    load_best_model(config)
    ap = AudioProcessor(config)

    if analyze_long and wav2 is None:
        try:
            y, sr = ap.load_mono_audio(wav1)
            if len(y) / sr > config.TARGET_DURATION * 2:
                return analyze_long_audio(wav1, config, threshold=threshold)
        except: pass

    if multi_drone and wav2 is not None:
        return detect_and_localize_multi_drone(wav1, wav2, wav3, config, threshold=threshold)

    mel_tensor = ap.prepare_3channel_mels(wav1, wav2, wav3)
    with torch.no_grad():
        prob_drone = torch.softmax(model(mel_tensor), dim=1)[0, 1].item()
    print(f"🎯 Drone probability: {prob_drone:.3f}")

    if prob_drone >= threshold:
        print("🚁 DRONE DETECTED → Precise localization…")
        loc = localize_now(wav1, wav2=wav2, wav3=wav3, config=config, hint_pos=hint_pos)
        return {"detected": True, "probability": prob_drone,
                "position": loc["position"], "confidence_radius": loc["confidence_radius"],
                "measured_tdoas": loc["measured_tdoas"].tolist(),
                "estimated_tdoas": loc["estimated_tdoas"].tolist()}
    print("🌳 No drone detected.")
    return {"detected": False, "probability": prob_drone}


def analyze_long_audio(wav_path, config, analysis_segments=10, threshold=0.75, multi_drone=False):
    print(f"🔍 Analyzing long audio: {wav_path}")
    ap = AudioProcessor(config)
    try:
        y_raw, sr = sf.read(str(wav_path))
    except Exception as e:
        return {"detected": False, "probability": 0.0, "segments": []}

    n_channels = 1 if y_raw.ndim == 1 else y_raw.shape[1]
    has_3ch = n_channels >= 3
    y_mono = y_raw if y_raw.ndim == 1 else y_raw[:, 0]
    y_mono = librosa.resample(y_mono.astype(np.float32), orig_sr=sr, target_sr=config.SR)
    sr = config.SR

    duration       = len(y_mono) / sr
    segment_dur    = config.TARGET_DURATION
    hop_dur        = max(1.0, (duration - segment_dur) / analysis_segments)

    DroneTrack._id_counter = 0
    tracker = PathTracker(config)
    base_ts = time.time()
    segments = []

    for i in range(analysis_segments):
        start   = i * hop_dur
        start_s = int(start * sr)
        end_s   = start_s + int(segment_dur * sr)
        if end_s > len(y_mono): break
        seg     = y_mono[start_s:end_s]
        seg_ts  = base_ts + start

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            sf.write(tf.name, seg, sr); tmp_path = tf.name

        try:
            if has_3ch:
                tmp_paths = []
                for ch in range(3):
                    ch_audio = y_raw[start_s:end_s, ch]
                    ch_audio = librosa.resample(ch_audio.astype(np.float32), orig_sr=sr, target_sr=config.SR)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf2:
                        sf.write(tf2.name, ch_audio, config.SR); tmp_paths.append(tf2.name)
                result = detect_and_localize_if_drone_enhanced(*tmp_paths, config,
                    threshold=threshold, analyze_long=False)
                for p in tmp_paths:
                    try: os.unlink(p)
                    except: pass
            else:
                mel_t = ap.prepare_3channel_mels(tmp_path)
                with torch.no_grad():
                    prob = torch.softmax(model(mel_t), dim=1)[0, 1].item()
                result = {"detected": prob >= threshold, "probability": prob}

            detected = result["detected"]
            position = result.get("position", None)
            if detected and position is not None:
                tracker.update([position], timestamp=seg_ts)

            segments.append({
                "segment": i + 1, "start_time": start, "end_time": start + segment_dur,
                "probability": result["probability"], "detected": detected,
                "position": position, "confidence_radius": result.get("confidence_radius")})

            status  = "🚁 DETECTED" if detected else "🌳 CLEAN"
            pos_str = (f"  pos=({position[0]:.2f},{position[1]:.2f})m"
                       if position is not None else "  (no position—single-channel)")
            print(f"   Seg {i+1}: {start:.1f}-{start+segment_dur:.1f}s → "
                  f"{status} (conf:{result['probability']:.3f}){pos_str}")
        except Exception as e:
            print(f"   Seg {i+1}: Error — {e}")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    if not segments:
        return {"detected": False, "probability": 0.0, "segments": []}

    probs   = [s["probability"] for s in segments]
    max_p   = max(probs); avg_p = float(np.mean(probs))
    det_cnt = sum(1 for s in segments if s["detected"])
    print(f"\n📊 Summary: {det_cnt}/{len(segments)} | max={max_p:.3f} | avg={avg_p:.3f}")

    active_tracks = [t for t in tracker.tracks if t.hits >= config.TRACKER_MIN_HITS]
    if active_tracks:
        print(f"\n🛤️  PathTracker: {len(active_tracks)} confirmed track(s)")
        for t in active_tracks:
            print(f"   Track #{t.track_id}: {len(t.positions)} waypoints  "
                  f"dist={t.total_distance():.2f}m  speed={t.speed():.2f}m/s")
        try: tracker.save_tracks()
        except: pass
    if active_tracks and any(len(t.positions) > 1 for t in active_tracks):
        try: tracker.plot_paths(title=f"Drone Path — {Path(wav_path).name}",
                                save_path=config.DRIVE_ROOT / "drone_paths_long_audio.png")
        except: pass

    base_result = {"segments": segments, "tracker": tracker, "active_tracks": active_tracks,
                   "detection_summary": {"total_segments": len(segments),
                                         "detected_segments": det_cnt,
                                         "max_confidence": max_p, "average_confidence": avg_p,
                                         "n_tracks": len(active_tracks)}}
    detected_segs = [s for s in segments if s["detected"]]
    if max_p >= threshold and detected_segs:
        best = max(detected_segs, key=lambda x: x["probability"])
        return {"detected": True, "probability": max_p, "best_segment": best, **base_result}
    return {"detected": False, "probability": max_p, **base_result}


# ==================== AUDIO PLAYER WITH TUNABLE THRESHOLD (COLAB) ====================

def interactive_audio_player_and_detector(config):
    """
    Colab widget: upload 1 file (auto-simulates 3 mics) or 3 files (real mics).
    When 1 file is uploaded, propagation delays are synthesised from an assumed
    drone position — giving full detection + localization from any mono recording.
    """
    if not config.IN_COLAB:
        print("ℹ️  interactive_audio_player_and_detector() requires Google Colab.")
        return

    import ipywidgets as widgets
    from IPython.display import display, clear_output, Audio

    state = {"wav_paths": [], "result": None, "single_audio": None, "single_sr": None}

    title_html = widgets.HTML(
        "<h3 style='margin:4px 0'>🎛️ Drone Detection — Interactive Player v4</h3>"
        "<p style='margin:2px 0;color:#555;font-size:12px'>"
        "Upload <b>1 file</b> → 3-mic delays simulated automatically (full detection + position).<br>"
        "Upload <b>3 files</b> → use real mic recordings directly.</p>")

    upload_btn   = widgets.Button(description="📁 Upload Audio",
                                   button_style='primary',
                                   layout=widgets.Layout(width='180px'))
    threshold_sl = widgets.FloatSlider(
        value=0.70, min=0.10, max=0.99, step=0.01,
        description='Threshold:', continuous_update=False,
        layout=widgets.Layout(width='420px'),
        style={'description_width': '90px'})
    threshold_hint = widgets.HTML(
        "<span style='font-size:11px;color:#777'>"
        "↓ lower = more sensitive &nbsp; ↑ higher = stricter</span>")

    # Drone position sliders (used in single-file simulate mode)
    drone_x_sl = widgets.FloatSlider(
        value=1.0, min=-5.0, max=5.0, step=0.1,
        description='Drone X (m):', continuous_update=False,
        layout=widgets.Layout(width='380px'),
        style={'description_width': '100px'})
    drone_y_sl = widgets.FloatSlider(
        value=0.8, min=-5.0, max=5.0, step=0.1,
        description='Drone Y (m):', continuous_update=False,
        layout=widgets.Layout(width='380px'),
        style={'description_width': '100px'})
    pos_hint = widgets.HTML(
        "<span style='font-size:11px;color:#777'>"
        "Assumed drone position for delay simulation (single-file mode only). "
        "Mic array centre ≈ (0.10, 0.12). Reliable within ±2.5 m.</span>")
    sim_box = widgets.VBox([drone_x_sl, drone_y_sl, pos_hint])

    detect_btn   = widgets.Button(description="🚁 Run Detection",
                                   button_style='success',
                                   layout=widgets.Layout(width='160px'))
    status_lbl   = widgets.Label(value="Step 1: Upload an audio file")
    out_audio    = widgets.Output()
    out_info     = widgets.Output()
    out_results  = widgets.Output()

    def on_upload(_):
        from google.colab import files
        AUDIO_EXTS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac')
        with out_info: clear_output(); print("📁 Opening file picker…")
        uploaded    = files.upload()
        audio_files = sorted([Path(k) for k in uploaded.keys()
                               if Path(k).suffix.lower() in AUDIO_EXTS])
        if not audio_files:
            status_lbl.value = f"❌ No supported audio. Accepted: {', '.join(AUDIO_EXTS)}"
            return

        n = len(audio_files)
        state["wav_paths"] = [str(p) for p in audio_files[:3]]
        state["single_audio"] = None

        with out_info:
            clear_output()
            if n == 1:
                # Pre-load audio for fast re-detection when sliders change
                audio, sr = librosa.load(str(audio_files[0]), sr=config.SR, mono=True)
                state["single_audio"] = audio
                state["single_sr"]    = sr
                dur = len(audio) / sr
                print(f"✅ 1 file: {audio_files[0].name}  ({dur:.1f}s)")
                print()
                print("🎛️  SIMULATE MODE — 3 mic delays applied automatically")
                print(f"   Sample rate : {sr} Hz")
                print(f"   Mic array   : equilateral triangle, 20 cm spacing")
                print()
                print("   Adjust 'Drone X/Y' sliders to change the assumed position,")
                print("   then click Run Detection to re-localize instantly.")
                sim_box.layout.display = ''   # show position sliders
            elif n >= 3:
                print(f"✅ 3 files: {[p.name for p in audio_files[:3]]}")
                print("   REAL MIC MODE — using files as recorded")
                sim_box.layout.display = 'none'  # hide position sliders
            else:
                print(f"⚠️  Got {n} file(s). Need 1 or 3.")

        with out_audio:
            clear_output()
            for p in audio_files[:min(n, 3)]:
                display(Audio(str(p), autoplay=False))
                print(f"🎵 {p.name}")

        status_lbl.value = "Step 2: Adjust settings, then click Run Detection"

    def on_detect(_):
        if not state["wav_paths"]:
            with out_results: clear_output(); print("❌ Upload a file first.")
            return
        threshold = threshold_sl.value
        n_files   = len(state["wav_paths"])
        status_lbl.value = f"⏳ Analysing… (threshold={threshold:.2f})"

        with out_results:
            clear_output(wait=True)
            try:
                if state["single_audio"] is not None:
                    # Single-file simulate mode
                    drone_pos = [drone_x_sl.value, drone_y_sl.value]
                    print(f"🔬 Simulate mode: assumed drone pos = {drone_pos}")
                    audio = state["single_audio"]
                    sr    = state["single_sr"]
                    mic1, mic2, mic3 = simulate_3_mics_from_single(
                        audio, sr, drone_pos=drone_pos)
                    tmp_paths = []
                    for ch in [mic1, mic2, mic3]:
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        sf.write(tf.name, ch, sr); tmp_paths.append(tf.name)
                    try:
                        result = detect_and_localize_if_drone_enhanced(
                            tmp_paths[0], tmp_paths[1], tmp_paths[2],
                            config, threshold=threshold,
                            hint_pos=drone_pos)
                    finally:
                        for p in tmp_paths:
                            try: os.unlink(p)
                            except: pass
                elif n_files >= 3:
                    print("🔬 Real-mic mode: running detection + localization…")
                    result = detect_and_localize_if_drone_enhanced(
                        state["wav_paths"][0], state["wav_paths"][1],
                        state["wav_paths"][2], config, threshold=threshold)
                else:
                    print("🔬 Single-channel: detection only…")
                    result = detect_and_localize_if_drone_enhanced(
                        state["wav_paths"][0], None, None,
                        config, threshold=threshold, analyze_long=True)

                state["result"] = result
                _plot_quick_test_result(result, Path(state["wav_paths"][0]).name,
                                        threshold, config)

                det  = result.get("detected", False)
                prob = result.get("probability", 0.0)
                summ = result.get("detection_summary", {})
                pos  = result.get("position")
                rel  = result.get("reliable", True)

                print("\n" + "━"*52)
                print(f"  {'🚁 DRONE DETECTED' if det else '🌳 NO DRONE DETECTED'}")
                print(f"  Peak confidence : {prob:.3f}")
                if summ:
                    print(f"  Segments hit    : "
                          f"{summ.get('detected_segments',0)}/{summ.get('total_segments',0)}")
                    print(f"  Avg confidence  : {summ.get('average_confidence',0):.3f}")
                if pos is not None:
                    print(f"  Position (est.) : ({pos[0]:.2f}, {pos[1]:.2f}) m")
                    if state["single_audio"] is not None:
                        print(f"  Position (true) : ({drone_x_sl.value:.2f}, {drone_y_sl.value:.2f}) m  "
                              f"[simulated]")
                    cr = result.get("confidence_radius")
                    if cr: print(f"  Uncertainty     : ±{cr:.3f} m")
                    if not rel:
                        print("  ⚠️  LOW CONFIDENCE — try a drone pos closer to (0,0)")
                elif state["single_audio"] is None and n_files < 3:
                    print("  Position : N/A (upload 3 mic files, or use 1 file + simulate mode)")
                print("━"*52)

            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                traceback.print_exc()

        status_lbl.value = ("🚁 DRONE DETECTED"
                             if state["result"] and state["result"].get("detected")
                             else "🌳 No drone detected")

    upload_btn.on_click(on_upload)
    detect_btn.on_click(on_detect)

    # Hide position sliders until a single file is uploaded
    sim_box.layout.display = 'none'

    load_best_model(config)
    display(widgets.VBox([
        title_html,
        widgets.HBox([upload_btn, detect_btn]),
        threshold_sl, threshold_hint,
        sim_box,
        status_lbl, out_audio, out_info, out_results
    ]))


# ==================== VISUALISATION HELPERS ====================

def _draw_localization_map(ax, result, config, title="Drone Localization Map"):
    mics = config.MIC_POSITIONS
    pos  = np.array(result["position"])
    cr   = result.get("confidence_radius")
    prob = result.get("probability", result.get("prob", 1.0))
    pad  = max(1.5, np.linalg.norm(pos - mics.mean(axis=0)) * 0.4)
    cx, cy = mics.mean(axis=0)
    span = max(2.0, np.linalg.norm(pos - mics.mean(axis=0)) + pad)
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.add_patch(plt.Circle((cx, cy), span * 0.9, color="#3498db", alpha=0.04, zorder=0))
    for i, m in enumerate(mics):
        ax.scatter(*m, s=200, color="#2c3e50", zorder=5, marker='^')
        ax.annotate(f"Mic {i+1}", m, textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.plot([m[0], pos[0]], [m[1], pos[1]], color="#bdc3c7", linewidth=0.8, linestyle=":")
    if cr is not None and not np.isnan(cr):
        ax.add_patch(plt.Circle(pos, cr, color="#e74c3c", alpha=0.15, zorder=3, label=f"±{cr:.2f} m"))
        ax.add_patch(plt.Circle(pos, cr, fill=False, edgecolor="#e74c3c",
                                linewidth=1.5, linestyle="--", zorder=4))
    ax.scatter(*pos, s=350, color="#e74c3c", zorder=6, marker='*',
               edgecolors='#c0392b', linewidths=1)
    ax.annotate(f"({pos[0]:.2f}, {pos[1]:.2f}) m\nconf={prob:.3f}", pos,
                textcoords="offset points", xytext=(10, 10), fontsize=9, color="#c0392b",
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c", alpha=0.85))
    mt = result.get("measured_tdoas")
    if mt is not None:
        mt = np.array(mt)
        ax.text(0.02, 0.04, f"τ₁₂={mt[0]*1000:.3f} ms\nτ₁₃={mt[1]*1000:.3f} ms",
                transform=ax.transAxes, fontsize=8, color="#7f8c8d", verticalalignment='bottom')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25); ax.set_facecolor("#f8f9fa")
    if cr is not None and not np.isnan(cr): ax.legend(loc='upper right', fontsize=8)


def _draw_segment_timeline(ax, segments, threshold):
    for s in segments:
        t, p, w = s["start_time"], s["probability"], s["end_time"] - s["start_time"]
        ax.bar(t + w / 2, p, width=w * 0.85,
               color="#27ae60" if s["detected"] else "#e74c3c", alpha=0.75, edgecolor='white')
    ax.axhline(threshold, color="#e67e22", linestyle="--",
               linewidth=1.5, label=f"Threshold ({threshold:.2f})")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Time (s)"); ax.set_ylabel("Drone Probability")
    ax.set_title("Per-Segment Detection Timeline", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3); ax.set_facecolor("#fafafa")


def _plot_quick_test_result(result, source_label, threshold, config):
    prob     = result.get("probability", 0.0)
    detected = result.get("detected", False)
    segments = result.get("segments", [])
    has_pos  = detected and "position" in result
    has_tl   = len(segments) > 1
    n_cols   = 2 if (has_pos or has_tl) else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(13 if n_cols == 2 else 6, 5))
    if n_cols == 1: axes = [axes]

    # Confidence bar
    ax = axes[0]
    ax.barh([0], [prob], color="#27ae60" if detected else "#e74c3c", height=0.5, alpha=0.85)
    ax.barh([0], [1.0], color="#ecf0f1", height=0.5, alpha=0.4)
    ax.axvline(threshold, color="#e67e22", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.8); ax.set_yticks([])
    ax.set_xlabel("Drone Probability", fontsize=12)
    status_txt = "DRONE DETECTED" if detected else "NO DRONE"
    status_col = "#27ae60" if detected else "#e74c3c"
    ax.set_title(f"{status_txt}\nConf: {prob:.3f}", fontsize=14,
                 fontweight='bold', color=status_col, pad=12)
    ax.text(prob + 0.02, 0, f"{prob:.3f}", va='center', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.set_facecolor("#fafafa")
    ax.text(0.5, -0.38, source_label[:60], transform=ax.transAxes,
            ha='center', fontsize=8, color='gray', style='italic')
    ax.grid(axis='x', alpha=0.3)

    if has_pos and not has_tl:
        _draw_localization_map(axes[1], result, config)
    elif has_tl:
        _draw_segment_timeline(axes[1], segments, threshold)

    plt.suptitle("Drone Detection System — Result", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show()

    best_seg = result.get("best_segment", {})
    if has_tl and best_seg.get("position") is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        _draw_localization_map(ax2, best_seg, config,
                               title=f"Best Detection @ {best_seg['start_time']:.1f}s")
        plt.tight_layout(); plt.show()

    active_tracks = result.get("active_tracks", [])
    if active_tracks and any(len(t.positions) > 1 for t in active_tracks):
        tracker = result.get("tracker")
        if tracker:
            try: tracker.plot_paths(title=f"Drone Trajectory — {source_label[:40]}")
            except: pass


# ==================== AUTOMATED TESTS ====================

def generate_test_files(config):
    mic_pos = config.MIC_POSITIONS
    print("🎵 Generating synthetic test data...")
    for name, pos, noise in [
        ("", [0.85, 0.30], 0.1),
        ("_indoor", [-0.42, 1.05], 0.05),
        ("_outdoor", [3.2, -1.1], 0.35)
    ]:
        chs = generate_synthetic_drone(mic_pos, pos, noise_level=noise)
        for i, ch in enumerate(chs):
            sf.write(f"mic{i+1}{name}.wav", ch, config.SR)
        sf.write(f"recording{name}_3channel.wav", np.column_stack(chs), config.SR)
    print("✅ Test files generated!")


def run_automated_tests_on_generated_files(config, multi_drone=False):
    print("🧪 AUTOMATED TESTING"); print("=" * 60)
    test_files = [
        ("3-channel recording", "recording_3channel.wav", [0.85, 0.30]),
        ("Clean drone - 3 files", ["mic1.wav", "mic2.wav", "mic3.wav"], [0.85, 0.30]),
        ("Indoor close", ["mic1_indoor.wav", "mic2_indoor.wav", "mic3_indoor.wav"], [-0.42, 1.05]),
        ("Outdoor far",  ["mic1_outdoor.wav", "mic2_outdoor.wav", "mic3_outdoor.wav"], [3.2, -1.1]),
    ]
    results = []
    n = len(test_files)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5 * n))
    fig.suptitle("Automated Test Suite — Detection & Localization", fontsize=14, fontweight='bold', y=1.01)

    for row_idx, (test_name, files, true_pos) in enumerate(test_files):
        print(f"\n🔧 {test_name}  (expected: {true_pos})")
        ax_conf = axes[row_idx][0]; ax_map = axes[row_idx][1]
        try:
            if isinstance(files, list):
                result = detect_and_localize_if_drone_enhanced(*files, config, threshold=0.65)
            else:
                result = detect_and_localize_if_drone_enhanced(files, None, None, config, threshold=0.65)
            prob = result["probability"]; detected = result["detected"]
            pos_err = (np.linalg.norm(np.array(result["position"]) - np.array(true_pos))
                       if detected and "position" in result else None)
            results.append({"test_name": test_name, "detected": detected,
                            "confidence": prob, "position_error": pos_err,
                            "estimated_position": result.get("position"),
                            "confidence_radius": result.get("confidence_radius"),
                            "true_position": true_pos})

            bar_color = "#27ae60" if detected else "#e74c3c"
            ax_conf.barh([0], [prob], color=bar_color, height=0.45, alpha=0.85)
            ax_conf.barh([0], [1.0], color="#ecf0f1", height=0.45, alpha=0.35)
            ax_conf.axvline(0.65, color="#e67e22", linestyle="--", linewidth=1.5)
            ax_conf.set_xlim(0, 1); ax_conf.set_ylim(-0.5, 0.6); ax_conf.set_yticks([])
            ax_conf.set_xlabel("Drone Probability", fontsize=10)
            err_str = f"\nPos err: {pos_err:.3f} m" if pos_err is not None else ""
            status_txt = "DETECTED" if detected else "MISSED"
            ax_conf.set_title(f"{test_name}\n{status_txt}  conf={prob:.3f}{err_str}",
                              fontsize=9, color=bar_color, fontweight='bold')
            ax_conf.text(min(prob + 0.02, 0.9), 0, f"{prob:.3f}", va='center', fontsize=10, fontweight='bold')
            ax_conf.grid(axis='x', alpha=0.3); ax_conf.set_facecolor("#fafafa")

            if detected and "position" in result:
                _draw_localization_map(ax_map, result, config,
                                       title=f"Estimated vs True {true_pos}")
                ax_map.scatter(*np.array(true_pos), s=250, color="#2980b9", zorder=7,
                               marker='D', label="True position", edgecolors='#1a5276')
                ax_map.legend(loc='upper right', fontsize=8)
            else:
                ax_map.text(0.5, 0.5, "Drone Not Detected", transform=ax_map.transAxes,
                            ha='center', va='center', fontsize=13, color='#e74c3c',
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.5",
                                                         facecolor="#fdecea", alpha=0.9))
                ax_map.set_facecolor("#fafafa")

            status = "✅ DETECTED" if detected else "❌ MISSED"
            err_info = (f"  pos_err={pos_err:.3f}m" if pos_err is not None else "")
            print(f"   {status} | conf={prob:.3f}{err_info}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({"test_name": test_name, "detected": False,
                            "confidence": 0.0, "position_error": None, "error": str(e)})

    plt.tight_layout(); plt.show()
    try:
        fig.savefig(str(config.DRIVE_ROOT / "test_suite_results.png"), dpi=150, bbox_inches='tight')
    except: pass

    # Multi-drone test
    if multi_drone:
        print("\n🔧 Multi-drone detection (FIXED: distinct delays per drone)")
        md_paths = generate_multi_drone_test_files(config, positions=[[2.0, 0.5], [-1.5, 2.0]])
        try:
            md_result = detect_and_localize_multi_drone(*md_paths, config, threshold=0.65)
            n_found   = md_result['n_drones']
            print(f"   Drones detected: {n_found}")
            if n_found > 0:
                fig_md, ax_md = plt.subplots(figsize=(6, 5))
                mics = config.MIC_POSITIONS
                ax_md.scatter(mics[:, 0], mics[:, 1], s=200, color="#2c3e50",
                              marker='^', zorder=5, label='Microphones')
                cmap = plt.colormaps["tab10"]
                true_positions_md = [[2.0, 0.5], [-1.5, 2.0]]
                for tp in true_positions_md:
                    ax_md.scatter(*tp, s=200, color='blue', marker='D', zorder=6,
                                  label='True', edgecolors='navy', linewidths=1)
                for d in md_result["drones"]:
                    pos = d["position"]
                    ax_md.scatter(*pos, s=300, color=cmap(d["id"] % 10), zorder=7, marker='*',
                                  edgecolors='black', linewidths=0.5)
                    ax_md.annotate(f"Drone #{d['id']}\n({pos[0]:.2f},{pos[1]:.2f})m",
                                   pos, textcoords="offset points", xytext=(10, 6),
                                   fontsize=9, color=cmap(d["id"] % 10),
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                ax_md.set_xlabel("X (m)"); ax_md.set_ylabel("Y (m)")
                ax_md.set_title(f"Multi-Drone: {n_found} found (true: {len(true_positions_md)})",
                                fontweight='bold')
                ax_md.legend(); ax_md.grid(True, alpha=0.3)
                ax_md.set_aspect('equal'); ax_md.set_facecolor("#f8f9fa")
                plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    # PathTracker demo from test detections
    det_positions = [r["estimated_position"] for r in results
                     if r["detected"] and r.get("estimated_position") is not None]
    if len(det_positions) >= 2:
        DroneTrack._id_counter = 0
        tracker = PathTracker(config)
        base_ts = time.time()
        for idx, pos in enumerate(det_positions):
            tracker.update([pos], timestamp=base_ts + idx * 1.0)
        active = [t for t in tracker.tracks if t.hits >= config.TRACKER_MIN_HITS]
        if active:
            print(f"\n🛤️  PathTracker: {len(active)} confirmed track(s) from test detections")
            for t in active:
                print(f"   Track #{t.track_id}: {len(t.positions)} waypoints  "
                      f"dist={t.total_distance():.2f}m")
            try: tracker.save_tracks("test_suite_tracks.json")
            except: pass
            try: tracker.plot_paths(title="Test Suite Detections — Estimated Path",
                                    save_path=config.DRIVE_ROOT / "test_suite_paths.png")
            except: pass

    print("\n" + "=" * 60 + "\n🎯 TEST SUMMARY")
    det_count = sum(1 for r in results if r["detected"])
    print(f"📊 {det_count}/{len(results)} tests passed")
    for r in results:
        s   = "✅" if r["detected"] else "❌"
        err = f" pos_err={r['position_error']:.3f}m" if r.get("position_error") is not None else ""
        print(f"   {r['test_name']:28} {s} conf={r['confidence']:.3f}{err}")
    return results


def run_noise_robustness_test(config, snr_levels=None, n_clips=20):
    """
    Run SNR sweep on real test clips if available, otherwise synthesise them.
    """
    load_best_model(config)

    drone_test_dir = config.PROCESSED_DIR / "test" / "drone"
    wav_paths = []

    # ── Try real clips first ──────────────────────────────────────────────
    if drone_test_dir.exists():
        wav_paths = list(drone_test_dir.glob("*.*"))

    if wav_paths:
        random.shuffle(wav_paths)
        clips = wav_paths[:min(n_clips, len(wav_paths))]
        print(f"📂 Using {len(clips)} real clips from {drone_test_dir}")
    else:
        # ── Fallback: synthesise clips ────────────────────────────────────
        print("⚠️  No real test clips found — generating synthetic drone clips.")
        print(f"   (Run prepare_dataset() once to cache real clips for future runs)")
        print(f"   Generating {n_clips} synthetic clips…")

        tmp_dir   = tempfile.mkdtemp(prefix="drone_snr_")
        mic_pos   = config.MIC_POSITIONS
        sr        = config.SR
        dur       = config.TARGET_DURATION
        clips_tmp = []

        positions = [
            [0.5,  0.3], [0.8,  0.6], [1.0,  0.8], [1.2,  0.4],
            [0.3,  0.9], [0.6,  1.1], [0.9,  0.2], [0.4,  0.7],
            [1.1,  1.0], [0.7,  0.5], [0.2,  0.4], [1.3,  0.7],
            [0.5,  1.2], [0.8,  0.3], [1.0,  0.5], [0.3,  0.6],
            [0.6,  0.9], [1.1,  0.4], [0.4,  1.0], [0.9,  0.7],
        ]
        # Tile if more clips requested than positions defined
        while len(positions) < n_clips:
            positions += positions
        positions = positions[:n_clips]

        for i, pos in enumerate(positions):
            fund = random.choice([80, 90, 100, 110, 120])
            # Use mic 0 channel only (mono) for the noise test —
            # NoiseRobustnessTester adds noise to a single channel and
            # builds a 3-channel mel by replication.
            chs  = generate_synthetic_drone(
                mic_pos, pos,
                duration=dur, sr=sr,
                noise_level=0.02,     # very clean baseline
                fundamental=fund
            )
            out_path = os.path.join(tmp_dir, f"synth_drone_{i:04d}.wav")
            sf.write(out_path, chs[0], sr)   # save channel 0 (mono)
            clips_tmp.append(out_path)

        clips = [Path(p) for p in clips_tmp]
        print(f"   ✅ {len(clips)} synthetic clips ready in {tmp_dir}")

    tester = NoiseRobustnessTester(config, model, AudioProcessor(config))
    results = tester.evaluate_snr_sweep(
        clips,
        snr_levels=snr_levels,
        threshold=0.70
    )

    # Clean up temp dir if we made one
    if not wav_paths:
        import shutil
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    return results


# ==================== TENSORBOARD ====================

class TensorBoardLogger:
    def __init__(self, config):
        self.config = config; self.log_dir = config.DRIVE_TBOARD
        self.writer = None; self.process = None

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for f in self.log_dir.glob("events.out.tfevents.*"):
            try: f.unlink()
            except: pass
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0)); port = s.getsockname()[1]
            self.process = subprocess.Popen(
                ["tensorboard", "--logdir", str(self.log_dir),
                 "--host", "0.0.0.0", "--port", str(port), "--reload_multifile", "true"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
            try:
                from pyngrok import ngrok
                print(f"🎯 TensorBoard: {ngrok.connect(port, bind_tls=True)}")
            except: print(f"📊 TensorBoard on port {port}")
        except Exception as e: print(f"TensorBoard warning: {e}")
        return self.writer

    def log_metrics(self, metrics, step):
        if self.writer:
            for tag, val in metrics.items(): self.writer.add_scalar(tag, val, step)
            self.writer.flush()

    def close(self):
        if self.writer: self.writer.close()
        if self.process: self.process.terminate(); print("TensorBoard terminated")


# ==================== MAIN PIPELINE ====================

def main(config, num_epochs=10, force_rebuild_cache=False):
    config.ensure_dirs()
    for seed_fn in [random.seed, np.random.seed, torch.manual_seed]:
        seed_fn(config.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.SEED)

    print("🚀 Starting Drone Detection Training")
    print(f"📍 Device: {config.DEVICE}")

    dm  = DatasetManager(config)
    ap  = AudioProcessor(config)
    mcm = MelCacheManager(config, ap)

    if not dm.prepare_dataset(): raise RuntimeError("❌ Failed to prepare dataset")

    # Only rebuild mel cache if forced, or if it doesn't exist yet
    mel_train_dir = config.LOCAL_MELCACHE / "train"
    cache_exists = (mel_train_dir.exists() and
                    len(list(mel_train_dir.rglob("*.npy"))) > 100)

    if force_rebuild_cache or not cache_exists:
        print("🔄 Rebuilding mel cache...")
        if config.LOCAL_MELCACHE.exists():
            shutil.rmtree(config.LOCAL_MELCACHE)
        print("🎵 Adding synthetic data…")
        inject_synthetic_3ch_data(config)
        print("🎵 Creating mel cache…")
        mcm.create_mel_cache()
    else:
        n_cached = len(list(mel_train_dir.rglob("*.npy")))
        print(f"✅ Mel cache already exists ({n_cached} files) — skipping rebuild")
        print("   (pass force_rebuild_cache=True to force regeneration)")

    train_loader, val_loader, test_loader = get_dataloaders(config.LOCAL_MELCACHE, config.BATCH_SIZE)
    device = torch.device(config.DEVICE)
    _model = SimpleDroneDetector(in_channels=3).to(device)

    tb = TensorBoardLogger(config)
    TrainingManager(config, _model, device).train_and_evaluate(
        train_loader, val_loader, test_loader, num_epochs, tb)
    tb.close()
    print("✅ Training completed!")


def run_comprehensive_tests(config):
    """Full test suite: detection, multi-drone, noise robustness, path tracking."""
    print("🎯 Drone Detection System — Comprehensive Testing v3")
    print("=" * 60)
    global model; model = None
    load_best_model(config)

    generate_test_files(config)

    print("\n🚀 Standard detection + multi-drone tests…")
    run_automated_tests_on_generated_files(config, multi_drone=True)

    print("\n🛤️  Path tracking simulation (using dataset)…")
    simulate_path_tracking_from_dataset(config, n_positions=8, spread=2.0)

    print("\n🔊 Noise robustness sweep…")
    run_noise_robustness_test(config)

    print("\n💡 Available entry points:")
    print("   interactive_audio_player_and_detector(config)    — upload, play, tune threshold")
    print("   simulate_path_tracking_from_dataset(config)      — path tracking demo")
    print("   detect_and_localize_multi_drone(w1,w2,w3,config) — multi-drone detection")
    print("   run_noise_robustness_test(config)                — SNR sweep")
    print("\n🎉 System v3 ready!")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    print("🎯 Drone Detection and Localization System v3")
    print("=" * 60)
    print("\n📚 v3 fixes & improvements:")
    print("   • generate_synthetic_drone() — realistic per-mic propagation delays")
    print("   • simulate_path_tracking_from_dataset() — full track demo without live mics")
    print("   • generate_multi_drone_test_files() — distinct TDOA delays per drone")
    print("   • interactive_audio_player_and_detector() — upload, play, tune threshold")
    print("   • gcc_phat_peak_picking() — min-distance prevents duplicate peaks")
    print("\n💡 Quick Start:")
    print("   main(config, num_epochs=100)           # train")
    print("   run_comprehensive_tests(config)         # all tests")
    print("   interactive_audio_player_and_detector(config)  # interactive UI")
    print("   simulate_path_tracking_from_dataset(config)    # path tracking demo")