# test_multi_drone_audio.py
"""
Test script for multi-drone audio detection and tracking.
"""

import argparse
from pathlib import Path
from typing import List

def find_drone_audio_files(directory: str) -> List[str]:
    """Find all drone audio files in a directory."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(directory).glob(f"*{ext}"))
    
    return [str(f) for f in audio_files]

def main():
    parser = argparse.ArgumentParser(description='Multi-drone audio detection demo')
    parser.add_argument('--audio-dir', type=str, help='Directory containing drone audio files')
    parser.add_argument('--audio-files', nargs='+', help='Specific audio files to use')
    parser.add_argument('--num-drones', type=int, default=2, help='Number of drones to simulate')
    parser.add_argument('--auto-detect', action='store_true', default=True, 
                       help='Auto-detect drone types')
    
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
        # Create synthetic test files if no real audio is provided
        print("No audio files provided, generating synthetic test files...")
        audio_files = create_test_audio_files(args.num_drones)
    
    if not audio_files:
        print("❌ No audio files found or generated!")
        return
    
    # Run the multi-drone demo
    from multi_drone_audio_demo import multi_drone_demo_from_files
    
    result = multi_drone_demo_from_files(
        audio_files=audio_files,
        auto_detect_types=args.auto_detect
    )
    
    return result

def create_test_audio_files(num_files: int) -> List[str]:
    """Create synthetic test audio files for demonstration."""
    import tempfile
    import numpy as np
    import soundfile as sf
    from audio_processing import synthesise_drone
    from config import config
    
    test_files = []
    positions = [[i-1, 1] for i in range(num_files)]  # Spread them out
    funds = [100 + 30*i for i in range(num_files)]    # Different fundamentals
    
    for idx in range(num_files):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        # Generate synthetic drone
        channels = synthesise_drone(
            mic_positions=config.MIC_POSITIONS,
            src_xy=positions[idx],
            fundamental=funds[idx],
            noise_level=0.02,
            drone_type=['mavic_pro', 'mavic_2_pro'][idx % 2]
        )
        # Save only first channel (mono)
        sf.write(tf.name, channels[0], config.SR)
        test_files.append(tf.name)
        print(f"  Created test file: {tf.name} (fund={funds[idx]}Hz)")
    
    return test_files

if __name__ == "__main__":
    main()