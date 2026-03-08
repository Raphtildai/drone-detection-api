# mel_cache.py

import numpy as np
import librosa
from pathlib import Path

# ==================== MEL CACHE MANAGER ====================
class MelCacheManager:
    def __init__(self, config):
        self.config = config

    def create_mel_cache(self):
        """Create mel-spectrogram cache"""
        print("🎵 Creating mel-cache...")

        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                self._process_split(split, label)

        print("✅ Mel-cache created")

    def _process_split(self, split, label):
        """Process a single split"""
        wav_dir = self.config.PROCESSED_DIR / split / label
        out_dir = self.config.LOCAL_MELCACHE / split / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in wav_dir.glob("*.wav"):
            self._process_single_file(wav_path, out_dir)

    def _process_single_file(self, wav_path, out_dir):
        """Process single audio file to mel"""
        try:
            y, sr = librosa.load(str(wav_path), sr=self.config.SR, mono=True)
            y = self._pad_or_truncate(y, self.config.SR * self.config.TARGET_DURATION)
            mel = self._audio_to_normalized_mel(y, sr)
            out_path = out_dir / f"{wav_path.stem}.npy"
            np.save(out_path, mel)
        except Exception as e:
            print(f"❌ Error processing {wav_path}: {e}")

    def _pad_or_truncate(self, y, target_samples):
        target_samples = int(target_samples)
        if len(y) < target_samples:
            return np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            return y[:target_samples]

    def _audio_to_normalized_mel(self, y, sr):
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db.astype(np.float32)