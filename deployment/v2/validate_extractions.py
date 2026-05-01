#!/usr/bin/env python3
# validate_extractions.py - Run validation on extracted segments

import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import argparse
import csv

def validate_segment(ch0_path: Path, label_path: Path) -> dict:
    """Validate a single extracted segment."""
    try:
        audio, sr = sf.read(str(ch0_path))
        
        # FFT analysis
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Drone frequency range
        drone_range = (freqs >= 30) & (freqs <= 300)
        if np.any(drone_range):
            peak_idx = np.argmax(fft[drone_range])
            dom_freq = freqs[drone_range][peak_idx]
            drone_energy = np.sum(fft[drone_range]**2)
        else:
            dom_freq = np.nan
            drone_energy = 0
        
        total_energy = np.sum(fft**2)
        energy_ratio = drone_energy / total_energy if total_energy > 0 else 0
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8)
        
        # Try YIN for fundamental frequency
        try:
            f0_yin = librosa.yin(audio, fmin=30, fmax=300, sr=sr)
            f0_median = np.nanmedian(f0_yin)
        except:
            f0_median = np.nan
        
        is_valid = (
            dom_freq >= 30 and dom_freq <= 300 and
            rms_db > -40 and
            energy_ratio > 0.3
        )
        
        return {
            "valid": is_valid,
            "dom_freq_hz": float(dom_freq) if not np.isnan(dom_freq) else None,
            "f0_yin_hz": float(f0_median) if not np.isnan(f0_median) else None,
            "rms_db": float(rms_db),
            "energy_ratio": float(energy_ratio)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Validate extracted drone segments")
    parser.add_argument("--segments-dir", type=str, required=True,
                       help="Directory containing extracted segments")
    parser.add_argument("--output-csv", type=str, default="validation_report.csv",
                       help="Output CSV file")
    args = parser.parse_args()
    
    segments_dir = Path(args.segments_dir)
    manifest_path = segments_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"❌ manifest.json not found in {segments_dir}")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    results = []
    print(f"\n🔍 Validating {len(manifest)} segments...\n")
    
    for entry in manifest:
        session_id = entry['session_id']
        ch0_path = segments_dir / f"{session_id}_ch0.wav"
        label_path = segments_dir / f"{session_id}_label.json"
        
        print(f"  {session_id}...", end=" ")
        
        if not ch0_path.exists():
            print("❌ missing audio")
            continue
        
        validation = validate_segment(ch0_path, label_path)
        
        # Load ground truth
        with open(label_path) as f:
            label = json.load(f)
        
        result = {
            "session_id": session_id,
            "valid": validation['valid'],
            "dom_freq_hz": validation.get('dom_freq_hz'),
            "f0_yin_hz": validation.get('f0_yin_hz'),
            "rms_db": validation.get('rms_db'),
            "energy_ratio": validation.get('energy_ratio'),
            "session": label.get('session'),
            "maneuver_type": label.get('maneuver_type'),
            "radius_m": label.get('radius_m'),
            "n_drones": label.get('n_drones'),
            "azimuth_deg": label.get('drone', {}).get('azimuth'),
            "distance_m": label.get('drone', {}).get('distance'),
        }
        
        results.append(result)
        
        status = "✅ VALID" if validation['valid'] else "⚠️ INVALID"
        print(f"{status} (dom_freq={validation.get('dom_freq_hz', 'N/A')}Hz)")
    
    # Write CSV report
    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = ['session_id', 'valid', 'dom_freq_hz', 'f0_yin_hz', 'rms_db', 
                     'energy_ratio', 'session', 'maneuver_type', 'radius_m', 
                     'n_drones', 'azimuth_deg', 'distance_m']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total segments: {len(results)}")
    print(f"  Valid: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"  Report saved: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()