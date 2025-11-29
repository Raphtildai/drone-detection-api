# real_time_audio.py
import pyaudio
import numpy as np
import threading
import queue
import time
from collections import deque

class MultiChannelAudioCapture:
    def __init__(self, sample_rate=22050, channels=3, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = deque(maxlen=sample_rate * 10)  # 10-second buffer
        
        # PyAudio configuration for multi-channel input
        self.audio = pyaudio.PyAudio()
        
    def find_multichannel_device(self):
        """Find audio device that supports multiple channels"""
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] >= self.channels:
                print(f"Found device: {device_info['name']} with {device_info['maxInputChannels']} channels")
                return i
        return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for real-time audio capture"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Reshape to (chunk_size, channels)
            audio_data = audio_data.reshape(-1, self.channels)
            
            # Add to buffer and queue for processing
            self.audio_buffer.extend(audio_data)
            self.audio_queue.put(audio_data.copy())
            
        return (None, pyaudio.paContinue)
    
    def start_capture(self):
        """Start real-time audio capture"""
        device_index = self.find_multichannel_device()
        if device_index is None:
            raise Exception("No multi-channel audio device found")
        
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        print(f"🎤 Started recording from {self.channels} channels")
    
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
    
    def get_recent_audio(self, duration=3.0):
        """Get recent audio for analysis"""
        samples_needed = int(self.sample_rate * duration)
        recent_audio = list(self.audio_buffer)
        
        if len(recent_audio) >= samples_needed:
            audio_array = np.array(recent_audio[-samples_needed:])
            return audio_array
        return None

# Real-time processing class
class RealTimeDroneDetector:
    def __init__(self, config):
        self.config = config
        self.audio_capture = MultiChannelAudioCapture(channels=3)
        self.is_monitoring = False
        self.detection_callback = None
        
    def start_monitoring(self, detection_callback=None):
        """Start real-time monitoring"""
        self.detection_callback = detection_callback
        self.is_monitoring = True
        self.audio_capture.start_capture()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.audio_capture.stop_capture()
        
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_monitoring:
            try:
                # Get recent audio for analysis
                audio_data = self.audio_capture.get_recent_audio(duration=3.0)
                
                if audio_data is not None and audio_data.shape[0] > 0:
                    # Process the audio
                    result = self.process_audio_chunk(audio_data)
                    
                    # Call callback if drone detected
                    if result and result.get('detected') and self.detection_callback:
                        self.detection_callback(result)
                
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)
    
    def process_audio_chunk(self, audio_data):
        """Process a chunk of multi-channel audio"""
        try:
            # Ensure we have proper multi-channel data
            if audio_data.ndim != 2 or audio_data.shape[1] < 3:
                return None
            
            # Extract features and detect
            features = self.extract_features(audio_data)
            detection_result = self.detect_drone(features)
            
            # Localize if drone detected
            if detection_result['is_drone']:
                localization_result = self.localize_drone(audio_data)
                detection_result.update(localization_result)
            
            return detection_result
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def extract_features(self, audio_data):
        """Extract features from multi-channel audio"""
        # Your existing feature extraction code
        # This should handle multi-channel input properly
        pass
    
    def detect_drone(self, features):
        """Detect drone presence"""
        # Your existing detection code
        pass
    
    def localize_drone(self, audio_data):
        """Localize drone using multi-channel audio"""
        # Calculate TDOA using real multi-channel data
        tdoas, error = self.calculate_tdoa_real(audio_data)
        
        if tdoas is not None:
            position, loc_error = self.localize_drone_real(tdoas)
            return {
                'localized': True,
                'position': position,
                'tdoas': tdoas,
                'error': loc_error
            }
        
        return {'localized': False}
    
    def calculate_tdoa_real(self, audio_data):
        """Calculate TDOA from real multi-channel audio"""
        try:
            tdoas = []
            
            # Use channel 0 as reference
            ref_channel = audio_data[:, 0]
            
            for i in range(1, 3):
                # Cross-correlation between reference and other channels
                correlation = np.correlate(ref_channel, audio_data[:, i], mode='full')
                peak_idx = np.argmax(correlation) - (len(ref_channel) - 1)
                tdoa = peak_idx / self.audio_capture.sample_rate
                tdoas.append(tdoa)
            
            return np.array(tdoas), None
            
        except Exception as e:
            return None, str(e)
    
    def localize_drone_real(self, tdoas):
        """Localize drone using real TDOA measurements"""
        # Your existing localization code with real microphone positions
        pass