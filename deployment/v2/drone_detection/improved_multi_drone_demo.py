# -*- coding: utf-8 -*-
"""
improved_multi_drone_demo.py - Enhanced Version
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import Config, config
from .audio_processing import AudioProcessor
from .tracking import KalmanTracker
from .inference import run_pipeline
from .visualization import plot_multi_drone_positions, plot_track_trajectory, create_comparison_plot


@dataclass
class DroneAudioSource:
    file_path: str
    drone_type: Optional[str] = None
    fundamental_hz: Optional[float] = None
    source_position: Optional[Tuple[float, float]] = None
    gain_db: float = 0.0

class NonPhysicsMultiDroneMixer:
    """
    Multi-drone mixer without physical propagation.
    
    Uses:
    - Frequency band separation
    - Channel gain encoding
    - Artificial delays (for pseudo-TDOA)
    """

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or config
        self.ap = AudioProcessor(cfg)

    def load_and_preprocess(self, file_path: str, fundamental: Optional[float], idx: int) -> np.ndarray:
        audio = self.ap.load(file_path, mono=True)
        audio = self.ap.pad_or_truncate(audio)

        # --- Frequency band assignment ---
        if fundamental and fundamental > 0:
            try:
                import scipy.signal

                nyquist = self.cfg.SR / 2

                # Create NON-overlapping bands per drone
                band_width = 80
                center = fundamental + (idx * 120)  # shift each drone

                low = max(center - band_width, 20)
                high = min(center + band_width, nyquist - 1)

                sos = scipy.signal.butter(
                    4,
                    [low / nyquist, high / nyquist],
                    btype='band',
                    output='sos'
                )
                audio = scipy.signal.sosfilt(sos, audio)

            except Exception as e:
                print(f"    Warning: Filtering failed: {e}")

        return audio

    def generate_channel_gains(self, idx: int, n_channels: int) -> List[float]:
        """
        Create unique gain pattern per drone across channels.
        """
        base_patterns = [
            [1.0, 0.6, 0.3],
            [0.3, 1.0, 0.6],
            [0.6, 0.3, 1.0],
        ]

        pattern = base_patterns[idx % len(base_patterns)]

        # Slight variation for more drones
        variation = 1.0 - (idx * 0.05)

        return [max(p * variation, 0.1) for p in pattern[:n_channels]]

    def generate_artificial_delays(self, idx: int, n_channels: int) -> List[float]:
        """
        Small unique delays per drone (pseudo spatial signature)
        """
        base = 0.0002 * (idx + 1)

        return [
            base,
            base + 0.00015,
            base + 0.0003
        ][:n_channels]

    def apply_fractional_delay(self, audio: np.ndarray, delay_sec: float) -> np.ndarray:
        if delay_sec <= 0:
            return audio.copy()

        sr = self.cfg.SR
        delay_samples = delay_sec * sr

        if delay_samples >= len(audio):
            return np.zeros_like(audio)

        int_delay = int(delay_samples)
        frac_delay = delay_samples - int_delay

        delayed = np.roll(audio, int_delay)
        delayed[:int_delay] = 0

        if frac_delay > 1e-6:
            delayed[:-1] = (1 - frac_delay) * delayed[:-1] + frac_delay * delayed[1:]

        return delayed.astype(np.float32)

    def mix_drones(self, sources: List[DroneAudioSource]) -> Tuple[List[np.ndarray], List[Dict]]:
        n_channels = len(self.cfg.MIC_POSITIONS)
        n_samples = int(self.cfg.SR * self.cfg.TARGET_DURATION)

        mixture = [np.zeros(n_samples, dtype=np.float32) for _ in range(n_channels)]
        metadata = []

        for idx, source in enumerate(sources):
            print(f"  Mixing drone {idx+1}: {Path(source.file_path).name}")

            audio = self.load_and_preprocess(
                source.file_path,
                source.fundamental_hz,
                idx
            )

            gains = self.generate_channel_gains(idx, n_channels)
            delays = self.generate_artificial_delays(idx, n_channels)

            for ch in range(n_channels):
                delayed = self.apply_fractional_delay(audio, delays[ch])
                delayed *= gains[ch]

                mixture[ch] += delayed

            metadata.append({
                "drone_idx": idx,
                "gains": gains,
                "delays": delays,
                "fundamental": source.fundamental_hz
            })

            print(f"    Gains: {['%.2f' % g for g in gains]}")
            print(f"    Delays (ms): {['%.2f' % (d*1000) for d in delays]}")

        # Normalize
        max_peak = max(np.max(np.abs(ch)) for ch in mixture)
        if max_peak > 0:
            mixture = [ch / max_peak * 0.95 for ch in mixture]

        return mixture, metadata
    
class EnhancedMultiDroneMixer:
    """Enhanced mixer with better frequency separation and gain staging."""
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or config
        self.ap = AudioProcessor(cfg)
        self.speed_of_sound = cfg.SPEED_OF_SOUND
        
    def load_and_preprocess(self, file_path: str, fundamental: Optional[float]) -> np.ndarray:
        """Load audio and apply frequency filtering for better separation."""
        audio = self.ap.load(file_path, mono=True)
        audio = self.ap.pad_or_truncate(audio)
        
        # Apply bandpass filter around the fundamental frequency if known
        if fundamental and fundamental > 0:
            try:
                import scipy.signal
                nyquist = self.cfg.SR / 2
                # Wider band for better detection (fundamental ± 150 Hz)
                low = max(fundamental - 150, 20)
                high = min(fundamental + 150, nyquist - 1)
                sos = scipy.signal.butter(4, [low/nyquist, high/nyquist], 
                                         btype='band', output='sos')
                audio = scipy.signal.sosfilt(sos, audio)
            except Exception as e:
                print(f"    Warning: Filtering failed: {e}")
        
        return audio
    
    def calculate_time_delays(self, source_pos: Tuple[float, float]) -> List[float]:
        """Calculate precise time delays for each microphone."""
        delays = []
        source = np.array(source_pos, dtype=np.float64)
        
        for mic_pos in self.cfg.MIC_POSITIONS:
            distance = np.linalg.norm(source - mic_pos)
            delay = distance / self.speed_of_sound
            delays.append(delay)
        
        # Normalize to smallest delay
        min_delay = min(delays)
        delays = [d - min_delay for d in delays]
        
        return delays
    
    def apply_fractional_delay(self, audio: np.ndarray, delay_sec: float) -> np.ndarray:
        """Apply fractional sample delay using interpolation."""
        if delay_sec <= 0:
            return audio.copy()
        
        sr = self.cfg.SR
        delay_samples = delay_sec * sr
        
        if delay_samples >= len(audio):
            return np.zeros(len(audio), dtype=np.float32)
        
        # Integer and fractional parts
        int_delay = int(delay_samples)
        frac_delay = delay_samples - int_delay
        
        # Apply integer delay
        delayed = np.roll(audio, int_delay)
        delayed[:int_delay] = 0
        
        # Apply fractional delay using linear interpolation
        if frac_delay > 1e-6:
            weights = np.array([1 - frac_delay, frac_delay])
            delayed_frac = np.zeros_like(delayed)
            delayed_frac[:-1] = delayed[:-1] * weights[0] + delayed[1:] * weights[1]
            delayed_frac[-1] = delayed[-1]
            delayed = delayed_frac
        
        return delayed.astype(np.float32)
    
    def mix_drones(self, sources: List[DroneAudioSource]) -> Tuple[List[np.ndarray], List[Dict]]:
        """Mix drones with proper gain staging and frequency separation."""
        n_channels = len(self.cfg.MIC_POSITIONS)
        n_samples = int(self.cfg.SR * self.cfg.TARGET_DURATION)
        
        mixture = [np.zeros(n_samples, dtype=np.float32) for _ in range(n_channels)]
        source_metadata = []
        
        # Calculate per-channel gain for each source
        for idx, source in enumerate(sources):
            print(f"  Mixing drone {idx+1}: {Path(source.file_path).name}")
            
            # Load and preprocess audio
            audio = self.load_and_preprocess(source.file_path, source.fundamental_hz)
            
            # Get position
            position = source.source_position or (0, 0)
            
            # Calculate delays
            delays = self.calculate_time_delays(position)
            
            # Calculate distance-based gains (inverse square law)
            source_pos = np.array(position)
            gains = []
            for mic_pos in self.cfg.MIC_POSITIONS:
                distance = np.linalg.norm(source_pos - mic_pos)
                # More aggressive attenuation for distant drones
                gain = 1.0 / (max(distance, 0.5) ** 1.2)
                gains.append(gain)
            
            # Normalize gains to prevent clipping
            max_gain = max(gains)
            if max_gain > 0:
                gains = [g / max_gain * 0.8 for g in gains]
            
            # Apply delays and gains to each channel
            for ch in range(n_channels):
                # Apply delay
                delayed = self.apply_fractional_delay(audio, delays[ch])
                
                # Apply gain
                delayed *= gains[ch]
                
                # Add to mixture
                mixture[ch] = np.clip(mixture[ch] + delayed, -1.0, 1.0)
            
            source_metadata.append({
                'drone_idx': idx,
                'file': source.file_path,
                'position': position,
                'fundamental': source.fundamental_hz,
                'gains': gains
            })
            
            print(f"    Position: ({position[0]:.2f}, {position[1]:.2f}) m")
            print(f"    Gains: [{', '.join(f'{g:.2f}' for g in gains)}]")
        
        # Final normalization
        max_peak = max(np.max(np.abs(ch)) for ch in mixture)
        if max_peak > 0:
            mixture = [ch / max_peak * 0.95 for ch in mixture]
        
        return mixture, source_metadata


def detect_drone_fundamentals_enhanced(audio_files: List[str], cfg: Config) -> List[Optional[float]]:
    """Enhanced fundamental frequency detection with harmonic analysis."""
    fundamentals = []
    ap = AudioProcessor(cfg)
    
    for file_path in audio_files:
        try:
            audio = ap.load(file_path, mono=True)
            
            # Compute spectrogram for better frequency resolution
            f, t, Sxx = signal.spectrogram(audio, cfg.SR, nperseg=2048, noverlap=1024)
            
            # Average over time
            power = np.mean(Sxx, axis=1)
            
            # Look for peaks in drone frequency range
            mask = (f >= 50) & (f <= 500)
            f_masked = f[mask]
            power_masked = power[mask]
            
            # Find top 3 peaks
            peak_indices = signal.find_peaks(power_masked, height=np.max(power_masked) * 0.3)[0]
            if len(peak_indices) > 0:
                # Take the lowest frequency peak as fundamental (usually the strongest)
                fundamental = f_masked[peak_indices[0]]
                fundamentals.append(fundamental)
                print(f"  Detected fundamental: {fundamental:.1f} Hz for {Path(file_path).name}")
            else:
                fundamentals.append(None)
                print(f"  Could not detect fundamental for {Path(file_path).name}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            fundamentals.append(None)
    
    return fundamentals


def multi_drone_demo_enhanced(
    audio_files: List[str],
    cfg: Optional[Config] = None,
    positions: Optional[List[Tuple[float, float]]] = None,
    visualize: bool = True,
) -> dict:
    """
    Enhanced multi-drone demo with better localization.
    
    Args:
        audio_files: List of audio file paths
        cfg: Configuration object
        positions: Manual positions for drones [(x1,y1), (x2,y2), ...]
        visualize: Generate visualization plots
    
    Returns:
        Detection results dictionary
    """
    cfg = cfg or config
    # mixer = EnhancedMultiDroneMixer(cfg)
    mixer = NonPhysicsMultiDroneMixer(cfg)
    
    print("\n" + "="*70)
    print("🚁 ENHANCED MULTI-DRONE DETECTION DEMO")
    print("="*70)
    print(f"Processing {len(audio_files)} drone audio files...\n")
    
    # Detect fundamental frequencies
    print("📡 Analyzing drone audio characteristics:")
    fundamentals = detect_drone_fundamentals_enhanced(audio_files, cfg)
    
    # Set positions if not provided
    if positions is None:
        # Spread drones in a circle for better separation
        n_drones = len(audio_files)
        positions = []
        for i in range(n_drones):
            angle = i * (360 / n_drones)
            radius = 6.0  # Further away for better separation
            positions.append((radius * np.cos(np.radians(angle)), 
                             radius * np.sin(np.radians(angle))))
    
    # Create audio sources with gain adjustments
    sources = []
    for idx, (file_path, pos, fund) in enumerate(zip(audio_files, positions, fundamentals)):
        # Adjust gain based on distance
        distance = np.linalg.norm(pos)
        gain_db = -10 * np.log10(max(distance, 1.0))  # 10dB loss per doubling of distance
        
        sources.append(DroneAudioSource(
            file_path=file_path,
            source_position=pos,
            fundamental_hz=fund,
            gain_db=gain_db
        ))
        
        print(f"\n  Drone {idx+1}: {Path(file_path).name}")
        print(f"    Fundamental: {fund:.1f} Hz" if fund else "    Fundamental: unknown")
        print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f}) m")
        print(f"    Gain adjustment: {gain_db:.1f} dB")
    
    # Mix the audio
    print("\n🎛️  Creating enhanced multi-drone audio mixture...")
    mixture, metadata = mixer.mix_drones(sources)
    
    # Save to temporary files
    tmp_paths = []
    for idx, channel_audio in enumerate(mixture):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tf.name, channel_audio, cfg.SR)
        tmp_paths.append(tf.name)
    
    print(f"\n💾 Saved {len(tmp_paths)} channel files to temporary location")
    
    # Run the pipeline with multi-drone mode
    print("\n🔍 Running detection and localization pipeline...")
    print("   (This may take a few seconds)")
    
    try:
        # Configure tracker for multi-drone
        tracker = KalmanTracker(cfg)
        
        # Run pipeline with multi-drone detection
        result = run_pipeline(
            tmp_paths, 
            cfg, 
            tracker=tracker, 
            multi_drone=True
        )
        
        # Display results
        print("\n" + "="*70)
        print("📊 LOCALIZATION RESULTS")
        print("="*70)
        print(f"Detection confidence: {result['probability']:.3f}")
        print(f"Drones localized: {len(result['drones'])} / {len(audio_files)}")
        
        if result['drones']:
            print("\n📍 Position Comparison:")
            print("-" * 70)
            print(f"{'Drone':<8} {'Simulated (m)':<20} {'Detected (m)':<20} {'Error (m)':<12}")
            print("-" * 70)
            
            errors = []
            for i, (sim_pos, drone) in enumerate(zip(positions, result['drones'])):
                det_xy = drone["xy_position"]
                error = np.linalg.norm(det_xy - np.array(sim_pos))
                errors.append(error)
                
                print(f"  {i+1:<7} ({sim_pos[0]:>5.2f},{sim_pos[1]:>5.2f})     "
                      f"({det_xy[0]:>5.2f},{det_xy[1]:>5.2f})     {error:>5.2f}")
            
            print("-" * 70)
            print(f"Average localization error: {np.mean(errors):.2f} m")
            print(f"Best localization error: {np.min(errors):.2f} m")
            print(f"Worst localization error: {np.max(errors):.2f} m")
            
            # Generate visualizations
            if visualize:
                print("\n" + "="*70)
                print("📈 GENERATING VISUALIZATIONS")
                print("="*70)
                
                # Plot multi-drone positions
                print("  Plotting multi-drone positions...")
                plot_multi_drone_positions(result['drones'], cfg, save=True)
                
                # Plot track trajectories
                tracks = tracker.all_confirmed()
                if tracks:
                    print("  Plotting track trajectories...")
                    plot_track_trajectory(tracks, cfg, save=True)
                
                # Also create a comparison plot showing simulated vs detected
                create_comparison_plot(positions, result['drones'], 
                                       audio_files, fundamentals, cfg)
                
                # Also print track information
                print(f"\n  Confirmed tracks: {len(tracks)}")
                for track in tracks:
                    xy = track.predicted_xy()
                    print(f"    Track #{track.track_id}: ({xy[0]:.2f}, {xy[1]:.2f}) m, "
                          f"hits={track.hits}")
        else:
            print("\n⚠️  No drones were localized!")
            print("\nTroubleshooting tips:")
            print("  1. Ensure audio files contain clear drone sounds")
            print("  2. Try increasing TARGET_DURATION in config")
            print("  3. Adjust KF_MATCH_GATE for wider association")
            print("  4. Try using synthetic test files first")
        
        return result
        
    finally:
        # Cleanup temporary files
        for p in tmp_paths:
            try:
                os.unlink(p)
            except:
                pass


def quick_test_enhanced():
    """Quick test with synthetic drones."""
    print("\n🧪 Testing with synthetic drones...")
    
    # Create synthetic test scenario
    test_files = []
    positions = [(3.0, 2.0), (-2.0, 3.0), (1.0, -3.0)]
    
    from .audio_processing import synthesise_drone
    
    for idx, pos in enumerate(positions):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        channels = synthesise_drone(
            mic_positions=config.MIC_POSITIONS,
            src_xy=pos,
            drone_type=['mavic_pro', 'mavic_2_pro', 'mavic_mini'][idx],
            noise_level=0.02,
            duration=3.0
        )
        sf.write(tf.name, channels[0], config.SR)
        test_files.append(tf.name)
        print(f"  Created test drone {idx+1} at ({pos[0]:.1f}, {pos[1]:.1f}) m")
    
    try:
        result = multi_drone_demo_enhanced(
            audio_files=test_files,
            positions=positions,
            visualize=True
        )
        return result
    finally:
        for f in test_files:
            try:
                os.unlink(f)
            except:
                pass


if __name__ == "__main__":
    quick_test_enhanced()