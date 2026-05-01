#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_extracted_segments.py
───────────────────────────────
Visualizes extracted Dunakeszi segments to verify they contain valid drone audio.

Usage:
    python validate_extracted_segments.py --segments-dir dunakeszi_test_segments --output-dir validation_plots
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Try to import for spectrogram
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not installed - spectrograms disabled")

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'drone': '#2E86AB',
    'background': '#A23B72',
    'azimuth': '#F18F01',
    'warning': '#C73E1D',
}


def load_manifest(segments_dir: Path) -> List[dict]:
    """Load manifest.json from extracted segments directory."""
    manifest_path = segments_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def load_labels(segments_dir: Path, session_id: str) -> dict:
    """Load label.json for a specific segment."""
    label_path = segments_dir / f"{session_id}_label.json"
    if not label_path.exists():
        return {}
    with open(label_path) as f:
        return json.load(f)


def plot_segment_validation(
    session_id: str,
    audio_ch0: np.ndarray,
    audio_ch1: np.ndarray,
    audio_ch2: np.ndarray,
    label: dict,
    sample_rate: int = 22050,
    save_path: Optional[Path] = None,
):
    """
    Create a comprehensive validation plot for a single segment.
    
    Panels:
        1. Waveform (3 channels stacked)
        2. Spectrogram (channel 0)
        3. RMS envelope
        4. Text panel with ground truth info
    """
    duration = len(audio_ch0) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_ch0))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Segment Validation: {session_id}", fontsize=14, fontweight='bold')
    
    # Panel 1: Waveforms (3 channels)
    ax1 = axes[0, 0]
    ax1.plot(time_axis, audio_ch0, label='Ch0 (E)', alpha=0.7, linewidth=0.8)
    ax1.plot(time_axis, audio_ch1, label='Ch1 (H)', alpha=0.7, linewidth=0.8)
    ax1.plot(time_axis, audio_ch2, label='Ch2 (B)', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform - BK-6-W (E, H, B)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Spectrogram (channel 0 - E direction)
    ax2 = axes[0, 1]
    if HAS_LIBROSA:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_ch0)), ref=np.max)
        img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax2)
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Spectrogram (Channel 0 - E direction)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        # Add drone frequency annotation (typical drone noise ~40-200Hz for propellers)
        ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Propeller ~50Hz')
        ax2.axhline(200, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Motor ~200Hz')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Install librosa\nfor spectrogram', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Spectrogram (unavailable)')
    
    # Panel 3: RMS envelope and detection
    ax3 = axes[1, 0]
    hop_length = 512
    rms = librosa.feature.rms(y=audio_ch0, hop_length=hop_length)[0] if HAS_LIBROSA else np.ones(100)
    rms_time = np.linspace(0, duration, len(rms))
    ax3.plot(rms_time, rms, color=COLORS['drone'], linewidth=1.5)
    ax3.fill_between(rms_time, 0, rms, alpha=0.3, color=COLORS['drone'])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('RMS Energy')
    ax3.set_title('RMS Envelope (Channel 0)')
    ax3.grid(True, alpha=0.3)
    
    # Add threshold line (heuristic)
    threshold = np.median(rms) * 2 if len(rms) > 0 else 0.01
    ax3.axhline(threshold, color=COLORS['warning'], linestyle='--', 
                linewidth=1, label=f'Threshold ({threshold:.3f})')
    ax3.legend(fontsize=9)
    
    # Panel 4: Ground truth information
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Extract label info
    drone_info = label.get('drone', {})
    azimuth = drone_info.get('azimuth', 'N/A')
    distance = drone_info.get('distance', 'N/A')
    height = drone_info.get('height', 'N/A')
    
    info_text = f"""
    📋 GROUND TRUTH INFORMATION
    ─────────────────────────────────
    
    Segment ID:     {label.get('segment_id', 'N/A')}
    Session:        {label.get('session', 'N/A')}
    Split:          {label.get('split', 'N/A')}
    
    ─────────────────────────────────
    🎯 Drone Position:
       Azimuth:      {azimuth}° from North
       Distance:     {distance:.1f} m (XY plane)
       Height:       {height:.1f} m
    
    ─────────────────────────────────
    🚁 Maneuver:
       Type:         {label.get('maneuver_type', 'N/A')}
       Phase:        {label.get('flight_phase', 'N/A')}
       Speed:        {label.get('speed_mps', 'N/A')} m/s
       Radius:       {label.get('radius_m', 'N/A')} m
       Drones:       {label.get('n_drones', 'N/A')}
    
    ─────────────────────────────────
    📊 Audio Stats:
       Duration:     {duration:.2f} s
       Sample Rate:  {sample_rate} Hz
       Channels:     3 (E, H, B)
    """
    
    ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}")
    
    plt.close(fig)


def plot_azimuth_compass(azimuth_deg: float, save_path: Optional[Path] = None):
    """Plot a polar compass showing the expected drone direction."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    # Convert to radians (0° = North)
    rad = np.radians(90 - azimuth_deg)
    
    # Plot arrow pointing to drone
    ax.arrow(0, 0, rad, 0.8, head_width=0.1, head_length=0.1, 
             fc=COLORS['azimuth'], ec=COLORS['azimuth'], linewidth=2)
    
    # Plot circle at drone position
    ax.scatter(rad, 0.9, s=200, c=COLORS['drone'], marker='o', zorder=5)
    
    # Add text annotation
    ax.text(rad, 1.1, f"{azimuth_deg}°", ha='center', va='center', 
            fontsize=12, fontweight='bold', color=COLORS['azimuth'])
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(1.2)
    ax.set_rticks([])
    ax.set_title(f'Expected Drone Direction: {azimuth_deg}° from North', fontsize=12, pad=20)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved compass: {save_path}")
    
    plt.close(fig)


def create_validation_report(manifest: List[dict], segments_dir: Path, output_dir: Path):
    """Generate a summary report of all extracted segments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EXTRACTED SEGMENTS VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for entry in manifest:
        session_id = entry['session_id']
        
        # Load audio
        audio_files = [
            segments_dir / f"{session_id}_ch0.wav",
            segments_dir / f"{session_id}_ch1.wav",
            segments_dir / f"{session_id}_ch2.wav",
        ]
        
        if not all(f.exists() for f in audio_files):
            report_lines.append(f"❌ {session_id}: Missing audio files")
            continue
        
        # Load audio data
        audio_ch0, sr = sf.read(audio_files[0])
        audio_ch1, _ = sf.read(audio_files[1])
        audio_ch2, _ = sf.read(audio_files[2])
        
        # Load labels
        label = load_labels(segments_dir, session_id)
        
        # Compute audio metrics
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio_ch0**2)) + 1e-10)
        peak_db = 20 * np.log10(np.max(np.abs(audio_ch0)) + 1e-10)
        
        # Check if audio has energy (not silent)
        has_energy = rms_db > -50  # -50dB threshold
        is_valid = has_energy
        
        # Create validation plots
        plot_path = output_dir / f"{session_id}_validation.png"
        plot_segment_validation(
            session_id, audio_ch0, audio_ch1, audio_ch2, label, sr, plot_path
        )
        
        # Create compass plot
        azimuth = label.get('drone', {}).get('azimuth')
        if azimuth is not None:
            compass_path = output_dir / f"{session_id}_compass.png"
            plot_azimuth_compass(azimuth, compass_path)
        
        # Add to report
        status = "✅ VALID" if is_valid else "⚠️  LOW ENERGY"
        report_lines.append(f"\n{status}: {session_id}")
        report_lines.append(f"   Maneuver: {label.get('maneuver_type', 'N/A')} / {label.get('session', 'N/A')}")
        report_lines.append(f"   Azimuth: {label.get('drone', {}).get('azimuth', 'N/A')}°")
        report_lines.append(f"   Distance: {label.get('drone', {}).get('distance', 'N/A')} m")
        report_lines.append(f"   Audio RMS: {rms_db:.1f} dB, Peak: {peak_db:.1f} dB")
        report_lines.append(f"   Duration: {len(audio_ch0)/sr:.2f} s")
        report_lines.append(f"   Plots: {plot_path.name}, {compass_path.name if azimuth else 'N/A'}")
    
    # Write report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📋 Validation report saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    valid_count = sum(1 for line in report_lines if "✅ VALID" in line)
    total_count = sum(1 for line in report_lines if "✅ VALID" in line or "⚠️" in line)
    print(f"  Total segments: {total_count}")
    print(f"  Valid (has energy): {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    print(f"  Reports saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Validate extracted Dunakeszi segments")
    parser.add_argument("--segments-dir", type=str, default="dunakeszi_test_segments",
                        help="Directory containing extracted segments")
    parser.add_argument("--output-dir", type=str, default="validation_plots",
                        help="Directory to save validation plots")
    args = parser.parse_args()
    
    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    
    if not segments_dir.exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        return
    
    print(f"🔍 Validating segments in: {segments_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Load manifest
    try:
        manifest = load_manifest(segments_dir)
        print(f"✅ Found {len(manifest)} segments in manifest")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Create validation report and plots
    create_validation_report(manifest, segments_dir, output_dir)
    
    print("\n🎯 To manually verify:")
    print("   1. Open the validation plots in the output directory")
    print("   2. Check spectrograms for drone harmonic structure")
    print("   3. Verify RMS envelope shows activity (not silence)")
    print("   4. Confirm compass direction matches expected maneuver")
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
    # Example usage:
    # python validate_extracted_segments.py --segments-dir dunakeszi_test_segments --output-dir validation_plots