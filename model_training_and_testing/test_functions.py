import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from functions import DroneDetectionSystemFunctions
from config import Config

class TestDroneDetectionSystemFunctions:
    
    @pytest.fixture
    def system(self):
        return DroneDetectionSystemFunctions()
    
    def test_init_default_config(self):
        system = DroneDetectionSystemFunctions()
        assert system.config is not None
        assert system.model is None
        assert system.device is None
    
    def test_init_custom_config(self):
        config = Config()
        system = DroneDetectionSystemFunctions(config=config)
        assert system.config == config
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_load_best_model_cpu(self, mock_cuda, system):
        with patch.object(system, 'get_latest_checkpoint', return_value=Path('fake_model.pth')):
            with patch('torch.load') as mock_load:
                mock_load.return_value = {'model_state_dict': {}}
                with patch('model.SimpleDroneDetector'):
                    try:
                        system.load_best_model()
                    except:
                        pass
                    assert system.device == torch.device('cpu')
    
    def test_get_latest_checkpoint_no_files(self, system):
        with patch('pathlib.Path.exists', return_value=False):
            checkpoint = system.get_latest_checkpoint()
            assert checkpoint is None
    
    def test_latest_checkpoint_in_dir_empty(self, system):
        with patch('pathlib.Path.glob', return_value=[]):
            result = system._latest_checkpoint_in_dir(Path('.'))
            assert result is None
    
    @patch('librosa.load')
    @patch('librosa.feature.melspectrogram')
    def test_preprocess_audio_file(self, mock_mel, mock_load, system):
        mock_load.return_value = (np.random.randn(16000), 16000)
        mock_mel.return_value = np.random.randn(128, 100)
        
        result = system.preprocess_audio_file('test.wav')
        assert result.dtype == np.float32
        assert result.shape == (128, 100)
    
    def test_localize_now_single_channel(self, system):
        with patch.object(system, 'localize_from_three_wavs', return_value=([0.5, 0.5], 1e-3)):
            with patch('soundfile.read', return_value=(np.random.randn(16000), 16000)):
                result = system.localize_now('test.wav')
                assert result == [0.5, 0.5]
    
    @patch('torch.no_grad')
    def test_detect_and_localize_no_drone(self, mock_grad, system):
        system.model = Mock()
        system.device = torch.device('cpu')
        mock_logits = torch.tensor([[1.0, 0.1]])
        system.model.return_value = mock_logits
        
        with patch.object(system, 'preprocess_audio_file', return_value=np.random.randn(128, 100)):
            with patch('soundfile.read', return_value=(np.random.randn(16000), 16000)):
                result = system.detect_and_localize_if_drone('test.wav', threshold=0.75)
                assert result['detected'] == False
                assert 'probability' in result