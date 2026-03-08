# synthetic.py

import numpy as np
import librosa
from tqdm import tqdm

from config import config

# ==================== SYNTHETIC DATA ====================
def generate_synthetic_drone(mic_positions, source_pos, duration=3.0, sr=22050, noise_level=0.1):
    """Generate synthetic drone sound with noise level parameter"""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False)

    # Simple drone sound
    fundamental = 100
    sound = np.sin(2 * np.pi * fundamental * t) + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
    sound = sound / np.max(np.abs(sound))

    # Apply delays and noise
    channels = []
    for _ in range(3):
        noise = noise_level * np.random.randn(samples)  # Use noise_level parameter
        channels.append((sound + noise).astype(np.float32))

    return channels

def inject_synthetic_3ch_data(num_samples=None):
    """Inject synthetic data for training"""
    if num_samples is None:
        num_samples = config.SYTHETIC_DATA_SAMPLES

    cache_dir = config.LOCAL_MELCACHE / "train" / "drone"
    cache_dir.mkdir(parents=True, exist_ok=True)

    positions = np.random.uniform(-4, 4, size=(num_samples, 2))

    print(f"🎵 Generating {num_samples} synthetic samples...")
    for i in tqdm(range(num_samples)):
        pos = positions[i]
        chs = generate_synthetic_drone(config.MIC_POSITIONS, pos)
        mels = []
        for ch in chs:
            mel = librosa.feature.melspectrogram(y=ch, sr=config.SR, n_fft=config.N_FFT,
                                               hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            mels.append(mel_db.astype(np.float32))
        mel_stack = np.stack(mels, axis=0)
        np.save(cache_dir / f"synthetic_3ch_{i:06d}.npy", mel_stack)