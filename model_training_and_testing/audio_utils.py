# audio_utils.py

import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
from config import config

def load_mono_audio(wav_path):
    y, sr = librosa.load(str(wav_path), sr=config.SR, mono=True)
    return y, sr


def save_temp_wav(audio, sr):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, sr)
    return tmp.name


def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def compute_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SR,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def prepare_3ch_mels(wav1, wav2=None, wav3=None):
    target_len = int(config.SR * config.TARGET_DURATION)

    if wav2 is not None:
        ys = [librosa.load(w, sr=config.SR, mono=True)[0] for w in (wav1, wav2, wav3)]
    else:
        data, _ = sf.read(wav1)
        data = data.T if data.ndim == 2 else data[:, None].T
        ys = [data[i] for i in range(min(3, data.shape[0]))]

    mels = []
    for y in ys:
        y = pad_or_trim(y, target_len)
        mels.append(compute_mel(y))

    while len(mels) < 3:
        mels.append(mels[0])

    return torch.tensor(np.stack(mels)).unsqueeze(0).to(config.DEVICE)