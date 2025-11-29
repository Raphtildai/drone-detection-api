import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import zoom
import logging
import io

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sr = 22050
        self.target_duration = 3.0
        self.n_mels = 64
        self.hop_length = 256
        self.n_fft = 1024
        self.target_samples = int(self.sr * self.target_duration)
        self.expected_time_frames = 259  # From training data
    
    def load_audio_from_fileobj(self, file_obj):
        """Load audio using soundfile"""
        try:
            file_obj.seek(0)
            audio_data, original_sr = sf.read(file_obj)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if original_sr != self.sr:
                audio_data = self.resample_audio(audio_data, original_sr, self.sr)
            
            return audio_data, self.sr
            
        except Exception as e:
            raise Exception(f"Audio loading failed: {str(e)}")
    
    def resample_audio(self, audio_data, orig_sr, target_sr):
        """Resample audio using scipy"""
        try:
            duration = len(audio_data) / orig_sr
            new_length = int(duration * target_sr)
            resampled = signal.resample(audio_data, new_length)
            return resampled
        except Exception as e:
            raise Exception(f"Resampling failed: {str(e)}")
    
    def preprocess_audio(self, y, sr):
        """Preprocess audio: pad/truncate to target duration"""
        target_samples = int(sr * self.target_duration)
        
        if len(y) < target_samples:
            return np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            return y[:target_samples]
    
    def extract_features(self, y, sr):
        """Extract mel-like spectrogram features with FIXED dimensions"""
        try:
            # Preprocess audio - ensure exact length
            y_processed = self.preprocess_audio(y, sr)
            
            # Create spectrogram with exact parameters to match training
            f, t, Sxx = signal.spectrogram(
                y_processed,
                fs=sr,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                window='hann',
                scaling='density',
                mode='magnitude'
            )
            
            # Convert to mel-like scale
            mel_spectrum = self.linear_to_mel(Sxx, sr, self.n_mels)
            
            # Convert to dB scale
            mel_db = 10 * np.log10(mel_spectrum + 1e-10)
            
            # CRITICAL FIX: Ensure exact dimensions (64, 259)
            current_frames = mel_db.shape[1]
            if current_frames != self.expected_time_frames:
                logger.info(f"Resizing features from {mel_db.shape} to (64, {self.expected_time_frames})")
                mel_db = self.resize_to_target(mel_db, self.expected_time_frames)
            
            # Normalize
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            
            # Repeat to 3 channels
            mel_db = np.expand_dims(mel_db, 0)  # Add channel dimension
            mel_3channel = np.repeat(mel_db, 3, axis=0)  # Repeat to 3 channels
            
            logger.info(f"Final features shape: {mel_3channel.shape}")
            return mel_3channel.astype(np.float32)
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {str(e)}")
    
    def resize_to_target(self, features, target_frames):
        """Resize features to target time frames"""
        current_frames = features.shape[1]
        
        if current_frames == target_frames:
            return features
        
        # Use interpolation to resize
        zoom_factor = (1.0, target_frames / current_frames)
        resized = zoom(features, zoom_factor, order=1)  # Linear interpolation
        
        # If we got more frames than needed, truncate
        if resized.shape[1] > target_frames:
            resized = resized[:, :target_frames]
        # If we got fewer, pad with the last frame
        elif resized.shape[1] < target_frames:
            padding = target_frames - resized.shape[1]
            last_frame = resized[:, -1:]
            pad_frames = np.repeat(last_frame, padding, axis=1)
            resized = np.concatenate([resized, pad_frames], axis=1)
        
        return resized
    
    def linear_to_mel(self, spectrogram, sr, n_mels):
        """Convert linear spectrogram to mel-like scale"""
        mel_filters = self.create_mel_filterbank(sr, n_mels, spectrogram.shape[0])
        mel_spectrum = np.dot(mel_filters, spectrogram)
        return mel_spectrum
    
    def create_mel_filterbank(self, sr, n_mels, n_fft_bins):
        """Create mel filter bank"""
        low_freq = 0
        high_freq = sr / 2
        
        low_mel = self.hz_to_mel(low_freq)
        high_mel = self.hz_to_mel(high_freq)
        
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        
        fft_bins = np.floor((n_fft_bins + 1) * hz_points / sr).astype(int)
        
        filters = np.zeros((n_mels, n_fft_bins))
        
        for i in range(n_mels):
            left = fft_bins[i]
            center = fft_bins[i + 1]
            right = fft_bins[i + 2]
            
            if left < center < right:
                filters[i, left:center] = np.linspace(0, 1, center - left)
                filters[i, center:right] = np.linspace(1, 0, right - center)
        
        # Normalize filters
        enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
        filters *= enorm[:, np.newaxis]
        
        return filters
    
    def hz_to_mel(self, frequency):
        """Convert Hz to mel scale"""
        return 2595.0 * np.log10(1.0 + frequency / 700.0)
    
    def mel_to_hz(self, mel):
        """Convert mel to Hz scale"""
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)