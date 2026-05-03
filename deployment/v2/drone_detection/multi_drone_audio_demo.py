# -*- coding: utf-8 -*-
"""
multi_drone_audio_demo.py
─────────────────────────
Extended demo for real multi-drone audio processing with Kalman tracking.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .test_multi_drone_audio import create_test_audio_files, find_drone_audio_files
import numpy as np
import soundfile as sf
from dataclasses import dataclass
import warnings

from .config import Config, config
from .audio_processing import AudioProcessor
from .tracking import KalmanTracker
from .config import Config, config, DRONE_BPF_PROFILES, DRONE_BPF_ENERGY_RATIOS
from .orchestration import run_pipeline 


@dataclass
class DroneAudioSource:
    """Represents a single drone audio recording with its characteristics."""
    file_path: str
    drone_type: Optional[str] = None
    fundamental_hz: Optional[float] = None
    source_position: Optional[Tuple[float, float]] = None  # (x, y) in meters
    gain_db: float = 0.0  # Gain adjustment in dB


class MultiDroneAudioMixer:
    """
    Handles mixing multiple real drone audio recordings with proper gain staging
    and spatial positioning simulation.
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or config
        self.ap = AudioProcessor(cfg)
        
    def load_and_prepare_audio(self, source: DroneAudioSource) -> np.ndarray:
        """
        Load a drone audio file and prepare it for mixing.
        
        Args:
            source: DroneAudioSource with file path and parameters
            
        Returns:
            Prepared audio array (mono)
        """
        # Load audio
        audio = self.ap.load(source.file_path, mono=True)
        
        # Pad or truncate to target duration
        audio = self.ap.pad_or_truncate(audio)
        
        # Apply gain adjustment
        if source.gain_db != 0:
            gain_linear = 10 ** (source.gain_db / 20.0)
            audio = audio * gain_linear
        
        # Optionally filter to keep only BPF band (if needed)
        if source.fundamental_hz:
            audio = self._bandpass_filter(audio, source.fundamental_hz)
        
        return audio
    
    def _bandpass_filter(self, audio: np.ndarray, fundamental_hz: float) -> np.ndarray:
        """Apply bandpass filter around the fundamental frequency."""
        try:
            import scipy.signal
            nyquist = self.cfg.SR / 2
            # Filter around fundamental ± 100 Hz
            low = max(fundamental_hz - 100, 20)
            high = min(fundamental_hz + 100, nyquist - 1)
            sos = scipy.signal.butter(4, [low/nyquist, high/nyquist], 
                                      btype='band', output='sos')
            return scipy.signal.sosfilt(sos, audio)
        except:
            return audio
    
    def mix_multi_drone_audio(
        self, 
        sources: List[DroneAudioSource],
        snr_db: Optional[float] = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Mix multiple drone audio sources as if recorded by the microphone array.
        
        Each source is treated as coming from a different drone, and their
        contributions are summed for each microphone channel.
        
        Args:
            sources: List of DroneAudioSource objects
            snr_db: Optional SNR for added background noise
            
        Returns:
            Tuple of (mixed_channels, source_metadata)
        """
        n_channels = len(self.cfg.MIC_POSITIONS)
        n_samples = int(self.cfg.SR * self.cfg.TARGET_DURATION)
        
        # Initialize mixture for each channel
        mix = [np.zeros(n_samples, dtype=np.float32) for _ in range(n_channels)]
        
        source_metadata = []
        
        for idx, source in enumerate(sources):
            # Load and prepare audio
            audio = self.load_and_prepare_audio(source)
            
            # For now, we'll use a default position if not provided
            # In a real scenario, you'd simulate direction-dependent filtering
            position = source.source_position or (0, 0)
            
            # Apply simple panning based on position
            panned_channels = self._apply_spatial_panning(audio, position)
            
            # Add to mix
            for ch in range(n_channels):
                mix[ch] = np.clip(mix[ch] + panned_channels[ch], -1, 1)
            
            source_metadata.append({
                'source_idx': idx,
                'drone_type': source.drone_type,
                'fundamental_hz': source.fundamental_hz,
                'position': position
            })
        
        # Normalize to prevent clipping
        peak = max(float(np.max(np.abs(ch))) for ch in mix) + 1e-8
        mix = [(ch / peak * 0.95).astype(np.float32) for ch in mix]
        
        # Add noise if requested
        if snr_db is not None:
            mix = [self.ap.add_noise(ch, snr_db) for ch in mix]
        
        return mix, source_metadata
    
    def _apply_spatial_panning(self, audio: np.ndarray, position: Tuple[float, float]) -> List[np.ndarray]:
        """
        Apply simple spatial panning based on position.
        
        Args:
            audio: Mono audio source
            position: (x, y) position in meters from array center
            
        Returns:
            List of channel arrays
        """
        n_channels = len(self.cfg.MIC_POSITIONS)
        channels = []
        
        # Convert position to angle and distance
        x, y = position
        angle = np.arctan2(y, x)  # azimuth angle in radians
        distance = np.sqrt(x**2 + y**2)
        
        # Apply doppler effect if moving (simplified)
        # For static simulation, just apply gain based on distance and panning
        
        for mic_idx, mic_pos in enumerate(self.cfg.MIC_POSITIONS):
            # Calculate gain based on distance from source to microphone
            mic_x, mic_y = mic_pos
            distance_to_mic = np.sqrt((x - mic_x)**2 + (y - mic_y)**2)
            
            # Inverse square law + minimum distance to avoid infinite gain
            gain = 1.0 / (max(distance_to_mic, 0.5) ** 0.6)
            
            # Simple panning (more directional for better localization)
            # This is a simplified model; real TDOA would come from actual delays
            mic_angle = np.arctan2(mic_y, mic_x)
            angle_diff = np.abs(angle - mic_angle)
            directional_gain = np.cos(angle_diff / 2) ** 2
            
            final_gain = gain * (0.5 + 0.5 * directional_gain)
            
            # Apply gain
            channel_audio = audio * final_gain
            channels.append(channel_audio.astype(np.float32))
        
        return channels


def detect_drone_type_from_audio(audio: np.ndarray, sr: int) -> Optional[str]:
    """
    Detect drone type from audio based on BPF characteristics.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Detected drone type or None
    """
    from .audio_processing import AudioProcessor
    from .config import DRONE_BPF_PROFILES, DRONE_BPF_ENERGY_RATIOS
    
    ap = AudioProcessor()
    
    best_match = None
    best_ratio = 0
    
    # Check against known drone BPF profiles (use module-level constant)
    for drone_type, (f_min, f_center, f_max, n_harmonics) in DRONE_BPF_PROFILES.items():
        # For each drone type, compute BPF energy ratio
        ratio = ap.compute_bpf_energy_ratio(audio, f_center, bw_hz=30, n_harmonics=n_harmonics)
        
        # Compare to expected ratios (if available)
        expected_ratio = DRONE_BPF_ENERGY_RATIOS.get(drone_type, 0.3)
        score = 1.0 - min(1.0, abs(ratio - expected_ratio) / expected_ratio)
        
        if score > best_ratio:
            best_ratio = score
            best_match = drone_type
    
    return best_match if best_ratio > 0.4 else None


def multi_drone_demo_from_files(
    audio_files: List[str],
    cfg: Optional[Config] = None,
    auto_detect_types: bool = True,
    known_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    known_fundamentals: Optional[List[Optional[float]]] = None,
) -> dict:
    """
    Run multi-drone detection and tracking using real audio files.
    
    Args:
        audio_files: List of paths to real drone audio recordings
        cfg: Configuration object
        auto_detect_types: Automatically detect drone types from audio
        known_positions: Optional list of positions for each source (for panning)
        known_fundamentals: Optional list of fundamental frequencies
    
    Returns:
        Detection results dictionary
    """
    cfg = cfg or config
    mixer = MultiDroneAudioMixer(cfg)
    
    print("\n🚁 MULTI-DRONE REAL AUDIO DEMO")
    print("─" * 50)
    print(f"Processing {len(audio_files)} drone audio files...")
    
    # Create audio sources
    sources = []
    for idx, file_path in enumerate(audio_files):
        drone_type = None
        fundamental = None
        
        if auto_detect_types:
            # Load a short segment for detection
            ap = AudioProcessor(cfg)
            audio_sample = ap.load(file_path, mono=True)
            drone_type = detect_drone_type_from_audio(audio_sample, cfg.SR)
            if drone_type:
                f_min, f_center, f_max, _ = cfg.get_bpf_profile(drone_type)
                fundamental = f_center
        
        # Override with provided values
        if known_fundamentals and known_fundamentals[idx] is not None:
            fundamental = known_fundamentals[idx]
        if known_positions and known_positions[idx] is not None:
            position = known_positions[idx]
        else:
            # Simulate positions in a circle around the array
            angle = (idx * 2 * np.pi / len(audio_files))
            radius = 3.0  # meters
            position = (radius * np.cos(angle), radius * np.sin(angle))
        
        sources.append(DroneAudioSource(
            file_path=file_path,
            drone_type=drone_type,
            fundamental_hz=fundamental,
            source_position=position,
            gain_db=0.0
        ))
        
        print(f"  Drone {idx+1}: {Path(file_path).name}")
        if drone_type:
            print(f"    Type: {drone_type}, Fund: {fundamental:.1f}Hz")
        print(f"    Simulated position: ({position[0]:.2f}, {position[1]:.2f})m")
    
    # Mix the audio sources
    print("\n🎛️  Mixing audio sources...")
    mixed_audio, source_metadata = mixer.mix_multi_drone_audio(sources, snr_db=20)
    
    # Save mixed audio to temporary files
    tmp_paths = []
    for idx, channel_audio in enumerate(mixed_audio):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tf.name, channel_audio, cfg.SR)
        tmp_paths.append(tf.name)
    
    print(f"💾 Mixed {len(mixed_audio)} channels to temporary files")
    
    # Run the pipeline
    try:
        tracker = KalmanTracker(cfg)
        result = run_pipeline(tmp_paths, cfg, tracker=tracker, multi_drone=True)
        
        # Post-process results
        print(f"\n📊 Detection Results:")
        print(f"  Detected: {result['detected']}  (conf={result['probability']:.3f})")
        print(f"  Drones found: {len(result['drones'])}")
        
        for i, drone in enumerate(result['drones']):
            xy = drone["xy_position"]
            print(f"    Drone {i+1}: ({xy[0]:.2f},{xy[1]:.2f})m  "
                  f"az={drone['azimuth_deg']:.1f}°  dist={drone['distance_m']:.2f}m")
            
            # Compare to simulated positions if available
            if i < len(sources) and sources[i].source_position:
                true_pos = sources[i].source_position
                error = np.linalg.norm(xy - np.array(true_pos))
                print(f"      Error from simulation: {error:.3f}m")
        
        # Display track information
        tracks = tracker.all_confirmed()
        print(f"\n🎯 Confirmed tracks: {len(tracks)}")
        for track in tracks:
            xy = track.predicted_xy()
            print(f"    Track #{track.track_id}: pos=({xy[0]:.2f},{xy[1]:.2f})m  "
                  f"hits={track.hits}  dist={track.total_distance():.3f}m")
        
        return result
        
    finally:
        # Cleanup temporary files
        for p in tmp_paths:
            try:
                os.unlink(p)
            except:
                pass


def batch_process_multi_drone_scenarios(
    audio_file_groups: List[List[str]],
    cfg: Optional[Config] = None,
) -> List[dict]:
    """
    Process multiple multi-drone scenarios.
    
    Args:
        audio_file_groups: List of scenarios, each with list of audio files
        cfg: Configuration object
        
    Returns:
        List of results for each scenario
    """
    results = []
    
    for idx, audio_files in enumerate(audio_file_groups):
        print(f"\n{'='*60}")
        print(f"Scenario {idx + 1}/{len(audio_file_groups)}")
        print(f"{'='*60}")
        
        try:
            result = multi_drone_demo_from_files(audio_files, cfg)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing scenario {idx+1}: {e}")
            results.append(None)
    
    # Summary statistics
    successful = sum(1 for r in results if r and r.get('detected', False))
    total_drones = sum(len(r.get('drones', [])) for r in results if r)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"  Scenarios processed: {len(audio_file_groups)}")
    print(f"  Successful detections: {successful}/{len(audio_file_groups)}")
    print(f"  Total drones tracked: {total_drones}")
    print(f"{'='*60}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Drone Audio Demo")
    parser.add_argument("--audio-files", nargs="+", help="Paths to drone audio files")
    parser.add_argument("--audio-dir", type=str, help="Directory containing drone audio files")
    parser.add_argument("--num-drones", type=int, default=2, help="Number of drones to simulate if no files provided")
    parser.add_argument("--auto-detect", action="store_true", help="Automatically detect drone types from audio")
    args = parser.parse_args()

    # Gather audio files    
    if args.audio_files:
        audio_files = args.audio_files
    elif args.audio_dir:
        audio_files = find_drone_audio_files(args.audio_dir)
        if len(audio_files) < args.num_drones:
            print(f"⚠️  Only found {len(audio_files)} files, but requested {args.num_drones}")
        audio_files = audio_files[:args.num_drones]
    else:
        print("No audio files provided, generating synthetic test files...")
        audio_files = create_test_audio_files(args.num_drones)
    
    if not audio_files:
        print("❌ No audio files found or generated!")
        return
    result = multi_drone_demo_from_files(
        audio_files=audio_files,
        auto_detect_types=args.auto_detect
    )
    return result

if __name__ == "__main__":
    main()