import unittest
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
from unittest.mock import patch, MagicMock

from audio_utils import (
    load_mono_audio,
    save_temp_wav,
    pad_or_trim,
    compute_mel,
    prepare_3ch_mels,
)


class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.duration = 2
        self.audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, self.sr * self.duration))

    def test_load_mono_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, self.audio, self.sr)
            y, sr = load_mono_audio(tmp.name)
            self.assertEqual(sr, self.sr)
            self.assertTrue(len(y) > 0)

    def test_save_temp_wav(self):
        tmp_path = save_temp_wav(self.audio, self.sr)
        self.assertTrue(tmp_path.endswith(".wav"))
        data, sr = sf.read(tmp_path)
        self.assertEqual(sr, self.sr)
        np.testing.assert_array_almost_equal(data, self.audio, decimal=4)

    def test_pad_or_trim_padding(self):
        y = np.array([1, 2, 3])
        target_len = 5
        result = pad_or_trim(y, target_len)
        self.assertEqual(len(result), target_len)
        np.testing.assert_array_equal(result[:3], y)

    def test_pad_or_trim_trimming(self):
        y = np.array([1, 2, 3, 4, 5])
        target_len = 3
        result = pad_or_trim(y, target_len)
        self.assertEqual(len(result), target_len)
        np.testing.assert_array_equal(result, y[:3])

    def test_compute_mel(self):
        mel_db = compute_mel(self.audio)
        self.assertEqual(mel_db.dtype, np.float32)
        self.assertEqual(mel_db.ndim, 2)

    @patch('audio_utils.librosa.load')
    def test_prepare_3ch_mels_multi_file(self, mock_load):
        mock_load.return_value = (self.audio, self.sr)
        result = prepare_3ch_mels("wav1.wav", "wav2.wav", "wav3.wav")
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)

    @patch('audio_utils.sf.read')
    def test_prepare_3ch_mels_single_file(self, mock_read):
        stereo_audio = np.stack([self.audio, self.audio, self.audio])
        mock_read.return_value = (stereo_audio, self.sr)
        result = prepare_3ch_mels("audio.wav")
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)


if __name__ == '__main__':
    unittest.main()