#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dunakeszi_for_pipeline.py
──────────────────────────────────
Converts Dunakeszi extracted segments into the exact format expected by
the drone_detection pipeline, fixing all four incompatibilities:

  FIX 1 — Label JSON keys: nested drone.azimuth → flat azimuth_deg / distance_m / height_m
  FIX 2 — Azimuth convention: bearing-from-North (GT fixed script) → math angle (pipeline)
  FIX 3 — Label JSON printed to Colab tells you the config patch to apply
  FIX 4 — Produces a labels.csv for use with inference_test_loader's flat_wavs format

Usage
─────
    python prepare_dunakeszi_for_pipeline.py \
        --input  extracted_J/ \
        --output dunakeszi_pipeline_ready/ \
        [--max-dist 100.0]

    python prepare_dunakeszi_for_pipeline.py --input  extracted_J/ --output dunakeszi_pipeline_ready/ --max-dist 100.0
    

The output directory is a drop-in replacement for a LocalizationDataset
split directory and also works with load_test_dataset_zip (generic_triplet).

Convention reference
────────────────────
  Ground-truth fixed script:   azimuth = atan2(x, y)   x=East, y=North
                                0°=North, +90°=East   (geographic/bearing)
  Pipeline convention:         azimuth = atan2(y, x)   x=East, y=North
                                0°=East,  +90°=North  (math/Cartesian)
  Conversion:                  pipeline_az = 90 - bearing_from_north
                               then wrap to (-180, 180]

  Examples (X=East, Y=North coordinate frame):
    drone due West  → x=-60, y=0   → bearing=-90°  → pipeline=+180° (or -180°, same dir)
    drone SW        → x=-60, y=-60 → bearing=-135° → pipeline=-135°  (diagonal, unchanged)
    drone NE        → x=+60, y=+60 → bearing=+45°  → pipeline=+45°   (diagonal, unchanged)
    drone NW        → x=-60, y=+60 → bearing=-45°  → pipeline=+135°
    drone SE        → x=+60, y=-60 → bearing=+135° → pipeline=-45°

Config patch
────────────
  Apply these two lines in your Colab before inference/training on Dunakeszi:

    config.set_array_geometry('gp2')    # 2500mm baseline matches BK-6-E
    config.MAX_LOCALIZATION_DIST = 100.0  # covers all orbit radii up to 60m

  NOTE: if your model was TRAINED on uavirbase (200mm array), localization
  accuracy on Dunakeszi will be limited because the IPD feature distribution
  is different. Detection accuracy is unaffected (uses channel 0 only).
"""

import argparse
import json
import math
import shutil
import csv
from pathlib import Path
from typing import Optional


# ── Azimuth conversion ────────────────────────────────────────────────────────

def bearing_to_pipeline_az(bearing_deg: float) -> float:
    """
    Convert a geographic bearing (from-North, CW positive) to the
    pipeline's math angle (from-East, CCW positive), wrapped to (-180, 180].

    This reverses the atan2(x,y) → atan2(y,x) change made in the fixed
    ground truth script.  Diagonal segments (±45°, ±135°) are unaffected.
    """
    math_az = 90.0 - bearing_deg
    return (math_az + 180.0) % 360.0 - 180.0


def angular_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two angles (degrees)."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


# ── Label conversion ──────────────────────────────────────────────────────────

def convert_label(raw: dict, max_dist: float) -> Optional[dict]:
    """
    Convert a Dunakeszi extractor label JSON to the pipeline's flat format.

    Input (from dunakeszi_segment_extractor_fixed.py make_label_json):
      {
        "drone": {"azimuth": <bearing_deg>, "distance": <m>, "height": <m>},
        "segment_id": ..., "session": ..., "maneuver_type": ...,
        "n_drones": ..., "split": ..., ...
      }

    Output (pipeline LocalizationDataset / inference_test_loader format):
      {
        "azimuth_deg":  <pipeline math angle>,
        "distance_m":   <metres>,
        "height_m":     <metres>,
        "source":       "dunakeszi",
        "original_bearing_deg": <bearing used in GT fixed script>,
        "segment_id":   ...,
        "session":      ...,
        "maneuver_type": ...,
        "n_drones":     ...,
        "split":        ...,
        "array":        ...,
        "note":         <human-readable convention info>
      }

    Returns None if the label does not contain usable position data.
    """
    drone = raw.get("drone", {})
    bearing = drone.get("azimuth")     # bearing-from-North (fixed GT script)
    distance = drone.get("distance")
    height   = drone.get("height")

    # Null-position segments (figure-8, survey) — distance=0, azimuth=0
    # These are valid for detection but not for localization regression.
    has_position = (
        bearing is not None
        and distance is not None
        and height is not None
        and not (bearing == 0.0 and distance == 0.0)
    )

    if not has_position:
        pipeline_az  = None
        distance_m   = None
        height_m     = None
        note = "no_position: figure-8 or survey — valid for detection, not localization"
    else:
        pipeline_az  = bearing_to_pipeline_az(float(bearing))
        distance_m   = float(distance)
        height_m     = float(height)
        note = (
            f"bearing_from_north={bearing:.1f}° → pipeline_math_az={pipeline_az:.1f}°; "
            f"dist={distance_m:.1f}m (norm@{max_dist:.0f}m={distance_m/max_dist:.3f})"
        )

    return {
        "azimuth_deg":           pipeline_az,
        "distance_m":            distance_m,
        "height_m":              height_m,
        "source":                "dunakeszi",
        "original_bearing_deg":  float(bearing) if bearing is not None else None,
        "segment_id":            raw.get("segment_id"),
        "session":               raw.get("session"),
        "maneuver_type":         raw.get("maneuver_type"),
        "flight_phase":          raw.get("flight_phase"),
        "n_drones":              raw.get("n_drones"),
        "split":                 raw.get("split"),
        "array":                 raw.get("array"),
        "speed_mps":             raw.get("speed_mps"),
        "radius_m":              raw.get("radius_m"),
        "duration_s":            raw.get("duration_s"),
        "has_position":          has_position,
        "note":                  note,
    }


# ── Main conversion ───────────────────────────────────────────────────────────

def prepare_dunakeszi(
    input_dir:  Path,
    output_dir: Path,
    max_dist:   float = 100.0,
    dry_run:    bool  = False,
):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    label_files = sorted(input_dir.glob("*_label.json"))
    if not label_files:
        print(f"❌ No *_label.json files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    n_ok = n_no_pos = n_skip = 0

    print(f"\n{'═'*68}")
    print(f"  Dunakeszi → pipeline adapter")
    print(f"{'═'*68}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  MAX_LOCALIZATION_DIST to use: {max_dist:.0f} m")
    print(f"  Azimuth convention: bearing-from-N → math angle (atan2 swap)")
    print(f"{'─'*68}\n")

    for label_path in label_files:
        stem = label_path.stem.replace("_label", "")

        # Check all 3 channel WAVs exist
        ch_paths = [input_dir / f"{stem}_ch{i}.wav" for i in range(3)]
        missing  = [p for p in ch_paths if not p.exists()]
        if missing:
            print(f"  ⚠️  {stem}: missing {[p.name for p in missing]} — skipping")
            n_skip += 1
            continue

        # Load and convert label
        raw = json.loads(label_path.read_text())
        converted = convert_label(raw, max_dist)

        if not dry_run:
            # Copy WAV files unchanged (already 3s @ 22050Hz float32)
            for ch_path in ch_paths:
                shutil.copy2(ch_path, output_dir / ch_path.name)

            # Write converted label JSON
            out_label = output_dir / f"{stem}_label.json"
            out_label.write_text(json.dumps(converted, indent=2))

        # Status
        if converted["has_position"]:
            az_b = converted["original_bearing_deg"]
            az_p = converted["azimuth_deg"]
            dist = converted["distance_m"]
            ht   = converted["height_m"]
            print(
                f"  ✅ {stem}\n"
                f"      bearing={az_b:+7.1f}° → pipeline_az={az_p:+7.1f}°  "
                f"dist={dist:.1f}m  ht={ht:.1f}m  "
                f"[{converted.get('maneuver_type','?')} {converted.get('session','?')}]"
            )
            n_ok += 1
        else:
            print(
                f"  ⚪ {stem}  → no position  [{converted.get('note','')}]"
            )
            n_no_pos += 1

        # Accumulate CSV row
        csv_rows.append({
            "session_id":             stem,
            "azimuth_deg":            converted["azimuth_deg"]           if converted["has_position"] else "",
            "distance_m":             converted["distance_m"]            if converted["has_position"] else "",
            "height_m":               converted["height_m"]              if converted["has_position"] else "",
            "original_bearing_deg":   converted["original_bearing_deg"]  if converted["has_position"] else "",
            "has_position":           converted["has_position"],
            "session":                converted["session"]               or "",
            "maneuver_type":          converted["maneuver_type"]         or "",
            "n_drones":               converted["n_drones"]              or "",
            "split":                  converted["split"]                 or "",
            "radius_m":               converted["radius_m"]              or "",
            "speed_mps":              converted["speed_mps"]             or "",
            "duration_s":             converted["duration_s"]            or "",
            "wav_ch0":                f"{stem}_ch0.wav",
            "wav_ch1":                f"{stem}_ch1.wav",
            "wav_ch2":                f"{stem}_ch2.wav",
        })

    # Write labels.csv (used by inference_test_loader flat_wavs + run_test_dataset_evaluation)
    if not dry_run and csv_rows:
        csv_path = output_dir / "labels.csv"
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  📋 labels.csv written ({len(csv_rows)} rows) → {csv_path}")

    # Summary
    total = n_ok + n_no_pos + n_skip
    print(f"\n{'═'*68}")
    print(f"  CONVERSION SUMMARY")
    print(f"{'─'*68}")
    print(f"  Converted with position  : {n_ok}")
    print(f"  Converted (no position)  : {n_no_pos}  (detection-only segments)")
    print(f"  Skipped (missing WAVs)   : {n_skip}")
    print(f"  Total processed          : {total}")
    print(f"{'═'*68}")

    # Print the config patch the user needs to apply
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  REQUIRED CONFIG PATCH — apply in Colab before using this data  ║
╚══════════════════════════════════════════════════════════════════╝

  from drone_detection import config

  # Switch to BK-6-E array geometry (2500mm baseline, gp2)
  config.set_array_geometry('gp2')

  # Extend localization range to cover all Dunakeszi orbit radii
  config.MAX_LOCALIZATION_DIST = {max_dist:.1f}

  # (Optional) if your model checkpoint uses BPF energy ratio:
  # config.BPF_ENERGY_RATIO_AS_FEATURE = True  ← already True by default

  IMPORTANT: if your saved model was trained with uavirbase (200mm array),
  the IPD branch will be out of distribution for Dunakeszi data.
  Detection (CNN branch, uses only channel 0) will still work correctly.
  Localization accuracy will be degraded until you retrain or fine-tune
  on Dunakeszi data with gp2 geometry.
""")

    # Print the usage instructions
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  HOW TO USE THE OUTPUT DIRECTORY IN YOUR PIPELINE              │
└─────────────────────────────────────────────────────────────────┘

Option A — run_test_dataset_evaluation (generic_triplet, recommended):
─────────────────────────────────────────────────────────────────
  import zipfile, os
  from drone_detection import config, load_test_dataset_zip, run_test_dataset_evaluation

  config.set_array_geometry('gp2')
  config.MAX_LOCALIZATION_DIST = {max_dist:.1f}

  # Zip the output folder, then:
  test_ds = load_test_dataset_zip(
      'dunakeszi_pipeline_ready.zip',
      cfg=config,
      dataset_format='generic_triplet',
  )
  results = run_test_dataset_evaluation(test_ds, cfg=config, show_plots=True)

Option B — LocalizationDataset for training/fine-tuning:
─────────────────────────────────────────────────────────────────
  from drone_detection.datasets import LocalizationDataset
  from torch.utils.data import DataLoader

  config.set_array_geometry('gp2')
  config.MAX_LOCALIZATION_DIST = {max_dist:.1f}

  # Output dir is already in the right structure for LocalizationDataset
  # (just treat it as the 'split' directory):
  ds = LocalizationDataset(
      root=Path('{output_dir}').parent,
      split='{output_dir.name}',
      augment=False,
      cfg=config,
  )
  loader = DataLoader(ds, batch_size=4, shuffle=False)
  mel, ipd, label = next(iter(loader))
  # mel:   (4, 3, N_MELS, T)  — 3-channel mel spectrogram
  # ipd:   (4, 3) or (4, 4)   — TDOA features (gp2 range: ±8ms)
  # label: (4, 4)              — [sin(az), cos(az), dist/max, ht/max]

Option C — manual inference on one segment:
─────────────────────────────────────────────────────────────────
  from drone_detection import config, detect, localize, load_3ch

  config.set_array_geometry('gp2')
  config.MAX_LOCALIZATION_DIST = {max_dist:.1f}

  channels = load_3ch([
      '{output_dir}/seg_020_circle_train_ch0.wav',
      '{output_dir}/seg_020_circle_train_ch1.wav',
      '{output_dir}/seg_020_circle_train_ch2.wav',
  ], config)

  det = detect(channels, config)
  print('detected:', det['detected'], 'prob:', det['probability'])

  if det['detected']:
      loc = localize(channels, config)
      print('azimuth:', loc['azimuth_deg'], 'distance:', loc['distance_m'])
""")


# ── Verification helper ───────────────────────────────────────────────────────

def verify_output(output_dir: Path, max_dist: float = 100.0):
    """
    Spot-check the converted output: verify label keys, azimuth range,
    WAV file count, and distance normalisation.
    """
    output_dir = Path(output_dir)
    label_files = sorted(output_dir.glob("*_label.json"))

    print(f"\n{'─'*60}")
    print(f"  Verification: {len(label_files)} label files in {output_dir.name}")
    print(f"{'─'*60}")

    n_ok = n_warn = 0
    for lf in label_files:
        stem = lf.stem.replace("_label", "")
        label = json.loads(lf.read_text())

        # Required keys check
        required = ["azimuth_deg", "distance_m", "height_m", "source"]
        missing_keys = [k for k in required if k not in label]

        # WAV check
        wavs = [output_dir / f"{stem}_ch{i}.wav" for i in range(3)]
        missing_wavs = [w.name for w in wavs if not w.exists()]

        if missing_keys or missing_wavs:
            print(f"  ✗ {stem}: missing_keys={missing_keys}  missing_wavs={missing_wavs}")
            n_warn += 1
            continue

        az   = label["azimuth_deg"]
        dist = label["distance_m"]
        ht   = label["height_m"]
        hp   = label["has_position"]

        if hp:
            in_range = (
                az is not None and -180.0 <= az <= 180.0
                and dist is not None and dist >= 0
                and ht is not None and ht >= 0
            )
            norm_dist = dist / max_dist if dist else 0
            status = "✅" if in_range else "⚠️ "
            print(f"  {status} {stem}: az={az:+7.1f}°  dist={dist:.1f}m (norm={norm_dist:.3f})  ht={ht:.1f}m")
            if not in_range: n_warn += 1
            else: n_ok += 1
        else:
            print(f"  ⚪ {stem}: no position  [{label.get('note','')}]")
            n_ok += 1

    print(f"\n  OK: {n_ok}  Warnings: {n_warn}")
    if n_warn == 0:
        print("  ✅ All outputs look correct.\n")
    else:
        print("  ⚠️  Some issues found — review warnings above.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert Dunakeszi extracted segments to pipeline format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--input", "-i", required=True,
        help="Directory containing *_ch0.wav, *_ch1.wav, *_ch2.wav, *_label.json "
             "(e.g. extracted_J/)",
    )
    ap.add_argument(
        "--output", "-o", default="dunakeszi_pipeline_ready",
        help="Output directory (default: dunakeszi_pipeline_ready/)",
    )
    ap.add_argument(
        "--max-dist", type=float, default=100.0,
        help="MAX_LOCALIZATION_DIST to use for normalisation info (default: 100.0 m). "
             "Set this to the same value you pass to config.MAX_LOCALIZATION_DIST.",
    )
    ap.add_argument(
        "--verify-only", action="store_true",
        help="Only verify an already-converted output directory; do not convert.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = ap.parse_args()

    if args.verify_only:
        verify_output(Path(args.output), args.max_dist)
    else:
        prepare_dunakeszi(
            input_dir  = Path(args.input),
            output_dir = Path(args.output),
            max_dist   = args.max_dist,
            dry_run    = args.dry_run,
        )
        if not args.dry_run:
            verify_output(Path(args.output), args.max_dist)


if __name__ == "__main__":
    main()