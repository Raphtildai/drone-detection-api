#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dunakeszi_local.py
══════════════════════════════════════════════════════════════════════════════
Local (non-Colab) runner for the full Dunakeszi pipeline.

Stages
──────
  ground_truth  build ground-truth JSON/CSV from embedded metadata
  extract       extract per-segment WAVs from polywav files
  prepare       convert to pipeline-ready format (azimuth fix + flat keys)
  analyse       detect + localize + evaluate + plots

--pipeline-dir lets you skip stages 1–3 entirely and point 'analyse'
directly at any existing flat segment directory, regardless of which script
produced it.  Both CSV formats are accepted:

  Extractor format  (dunakeszi_segment_extractor_fixed.py output)
    columns: azimuth_at_clip_mid_deg, distance_xy_at_mid_m, altitude_at_mid_m
    → bearing-from-North convention; converted to pipeline math angle here

  Prepare format  (prepare_dunakeszi_for_pipeline.py output)
    columns: azimuth_deg, distance_m, height_m
    → already in pipeline convention; used as-is

Usage examples
──────────────
  # Point directly at an already-extracted directory:
  python run_dunakeszi_local.py \
      --work-dir     /data/dunakeszi/run_01 \
      --pipeline-dir ../dunakeszi_test_segments_B \
      --stage        analyse \
      --multi-drone

  # Full run from scratch:
  python run_dunakeszi_local.py \
      --wav-dir   /data/dunakeszi/polywav_J \
      --work-dir  /data/dunakeszi/run_01 \
      --array     BK-6-E

  # Resume from stage 4 (segments inside work-dir/pipeline_ready/):
  python run_dunakeszi_local.py \
      --work-dir  /data/dunakeszi/run_01 \
      --stage     analyse \
      --splits    test \
      --multi-drone

  # Long-range test:
  python run_dunakeszi_local.py \
      --work-dir     /data/dunakeszi/run_01 \
      --pipeline-dir ../dunakeszi_test_segments_B \
      --stage        analyse \
      --splits       test \
      --max-dist     350
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# §0  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _header(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def _check_file(path: Path, label: str):
    if not path.exists():
        print(f"  ✗  {label} not found: {path}")
        sys.exit(1)
    print(f"  ✓  {label}: {path}")


def _run(cmd: list, label: str):
    print(f"\n  ▶  {label}")
    print(f"     {' '.join(str(c) for c in cmd)}\n")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\n  ✗  {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✓  {label} done  ({time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# §1  CSV format detection and normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _csv_header(path: Path) -> set:
    with open(path, newline="") as f:
        return set(next(csv.reader(f)))


def _detect_format(labels_csv: Path) -> str:
    """Return 'prepare' or 'extractor'."""
    cols = _csv_header(labels_csv)
    if "azimuth_deg" in cols and "distance_m" in cols:
        return "prepare"
    if "azimuth_at_clip_mid_deg" in cols:
        return "extractor"
    raise ValueError(
        f"Unrecognised labels.csv format.\n"
        f"  Expected 'azimuth_deg'/'distance_m'  (prepare format)\n"
        f"  or       'azimuth_at_clip_mid_deg'   (extractor format).\n"
        f"  Columns found: {sorted(cols)}"
    )


def _bearing_to_pipeline(bearing_deg: float) -> float:
    """Geographic bearing (N=0, CW+) → pipeline math angle (E=0, CCW+)."""
    math_az = 90.0 - bearing_deg
    return (math_az + 180.0) % 360.0 - 180.0


def _normalise_extractor_csv(src: Path, segment_dir: Path) -> Path:
    """
    Convert extractor-format labels.csv to prepare-format, writing
    labels_normalised.csv alongside it.  Applies the bearing→math-angle
    conversion that prepare_dunakeszi_for_pipeline.py would have done.
    """
    out_path = segment_dir / "labels_normalised.csv"
    out_fields = [
        "session_id", "azimuth_deg", "distance_m", "height_m",
        "original_bearing_deg", "has_position",
        "session", "maneuver_type", "flight_phase", "n_drones", "split",
        "radius_m", "speed_mps", "duration_s",
        "wav_ch0", "wav_ch1", "wav_ch2",
    ]

    rows = []
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("session_id", "")

            def _f(v):
                try:
                    return float(v) if v not in (None, "") else None
                except (ValueError, TypeError):
                    return None

            # Position priority: mid-clip → clip-start → segment onset.
            # Symmetric transits (e.g. SW→origin→NE) have d=0 at the clip
            # midpoint even though the drone is well within range.  In that
            # case we fall back to the clip-start position, which is always
            # on the approach side and has a valid non-zero distance.
            mid_bearing   = _f(row.get("azimuth_at_clip_mid_deg"))
            mid_dist      = _f(row.get("distance_xy_at_mid_m"))
            mid_ht        = _f(row.get("altitude_at_mid_m") or row.get("altitude_m"))
            start_bearing = _f(row.get("azimuth_at_clip_start_deg"))
            start_dist    = _f(row.get("distance_xy_at_start_m"))
            onset_bearing = _f(row.get("onset_azimuth_deg"))
            onset_dist    = _f(row.get("onset_distance_xy_m"))

            if mid_dist is not None and mid_dist > 0.5:
                bearing, dist = mid_bearing, mid_dist
            elif start_dist is not None and start_dist > 0.5:
                bearing, dist = start_bearing, start_dist
            elif onset_dist is not None and onset_dist > 0.5:
                bearing, dist = onset_bearing, onset_dist
            else:
                bearing, dist = None, None   # hover at origin or unknown

            ht      = mid_ht
            has_pos = bearing is not None and dist is not None
            pip_az  = round(_bearing_to_pipeline(bearing), 4) if has_pos else None

            rows.append({
                "session_id":           sid,
                "azimuth_deg":          pip_az   if pip_az  is not None else "",
                "distance_m":           round(dist, 4) if dist is not None else "",
                "height_m":             round(ht,   4) if ht   is not None else "",
                "original_bearing_deg": round(bearing, 4) if bearing is not None else "",
                "has_position":         has_pos,
                "session":              row.get("session", ""),
                "maneuver_type":        row.get("maneuver_type", ""),
                "flight_phase":         row.get("flight_phase", ""),
                "n_drones":             row.get("n_drones", ""),
                "split":                row.get("split", ""),
                "radius_m":             row.get("radius_m", ""),
                "speed_mps":            row.get("speed_mps", ""),
                "duration_s":           row.get("clip_dur_s") or row.get("duration_s", ""),
                "wav_ch0":              f"{sid}_ch0.wav",
                "wav_ch1":              f"{sid}_ch1.wav",
                "wav_ch2":              f"{sid}_ch2.wav",
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r["has_position"])
    print(f"  ✓  Normalised {len(rows)} rows "
          f"({n_pos} with position) → {out_path.name}")
    return out_path


def resolve_labels_csv(segment_dir: Path) -> tuple:
    """
    Return (path_to_labels_csv, format_str).
    Normalises extractor format if needed; caches the result.
    """
    raw  = segment_dir / "labels.csv"
    norm = segment_dir / "labels_normalised.csv"

    if not raw.exists():
        print(f"  ✗  No labels.csv found in: {segment_dir}")
        print("     The directory must contain a labels.csv produced by")
        print("     dunakeszi_segment_extractor_fixed.py or")
        print("     prepare_dunakeszi_for_pipeline.py")
        sys.exit(1)

    fmt = _detect_format(raw)
    print(f"  ✓  labels.csv format: '{fmt}'")

    if fmt == "prepare":
        return raw, fmt

    # Extractor format — normalise (use cached copy if present)
    if norm.exists():
        # Invalidate cache if source is newer (e.g. after a code fix)
        if norm.stat().st_mtime >= raw.stat().st_mtime:
            print(f"  ✓  Cached normalised CSV found: {norm.name}")
            return norm, "prepare"
        else:
            norm.unlink()
            print("  ℹ  Stale normalised CSV removed — regenerating")

    print("  ℹ  Converting extractor CSV to pipeline format …")
    return _normalise_extractor_csv(raw, segment_dir), "prepare"


# ══════════════════════════════════════════════════════════════════════════════
# §2  Stage runners
# ══════════════════════════════════════════════════════════════════════════════

def stage_ground_truth(args, dirs: dict):
    _header("Stage 1 — Ground truth")
    script = Path(__file__).parent / "dunakeszi_ground_truth_fixed.py"
    _check_file(script, "ground truth script")

    cmd = [sys.executable, script, "--out_dir", dirs["gt"]]
    if args.verify_mems:
        cmd += ["--verify_mems", args.verify_mems]

    _run(cmd, "dunakeszi_ground_truth_fixed.py")
    _check_file(dirs["gt"] / "ground_truth_segments.json", "segments JSON")
    print(f"\n  Ground truth written to: {dirs['gt']}")


def stage_extract(args, dirs: dict):
    _header("Stage 2 — Segment extraction")
    gt_dir  = dirs["gt"]
    ext_dir = dirs["extracted"]

    if not (gt_dir / "ground_truth_segments.json").exists():
        print("  ✗  Ground truth not found — run stage 'ground_truth' first.")
        sys.exit(1)
    if not args.wav_dir:
        print("  ✗  --wav-dir is required for the extract stage.")
        sys.exit(1)

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"  ✗  --wav-dir not found: {wav_dir}")
        sys.exit(1)

    script = Path(__file__).parent / "dunakeszi_segment_extractor_fixed.py"
    _check_file(script, "extractor script")

    cmd = [
        sys.executable, script,
        "--segments",      gt_dir / "ground_truth_segments.json",
        "--wav-dir",       wav_dir,
        "--output-dir",    ext_dir,
        "--array",         args.array,
        "--clip-duration", args.clip_duration,
        "--clip-position", args.clip_position,
    ]
    if args.splits:
        cmd += ["--splits"] + args.splits
    if args.skip_unlabelled:
        cmd.append("--skip-unlabelled")
    if args.no_validate:
        cmd.append("--no-validate")
    if args.no_zip:
        cmd.append("--no-zip")

    _run(cmd, "dunakeszi_segment_extractor_fixed.py")
    print(f"\n  Extracted {len(list(ext_dir.glob('*_ch0.wav')))} segment(s) to: {ext_dir}")


def stage_prepare(args, dirs: dict):
    _header("Stage 3 — Prepare for pipeline")
    ext_dir   = dirs["extracted"]
    ready_dir = dirs["pipeline_ready"]

    if not list(ext_dir.glob("*_label.json")):
        print("  ✗  No label JSONs in extracted dir — run stage 'extract' first.")
        sys.exit(1)

    script = Path(__file__).parent / "prepare_dunakeszi_for_pipeline.py"
    _check_file(script, "prepare script")

    cmd = [
        sys.executable, script,
        "--input",    ext_dir,
        "--output",   ready_dir,
        "--max-dist", args.max_dist,
    ]
    _run(cmd, "prepare_dunakeszi_for_pipeline.py")
    print(f"\n  {len(list(ready_dir.glob('*_label.json')))} segment(s) in: {ready_dir}")


def stage_analyse(args, dirs: dict):
    _header("Stage 4 — Inference + evaluation")

    # Choose segment directory: --pipeline-dir overrides work-dir/pipeline_ready/
    if args.pipeline_dir:
        segment_dir = Path(args.pipeline_dir).resolve()
        print(f"  Using --pipeline-dir: {segment_dir}")
    else:
        segment_dir = dirs["pipeline_ready"]

    if not segment_dir.exists():
        print(f"  ✗  Segment directory not found: {segment_dir}")
        sys.exit(1)

    results_dir = dirs["results"]
    plots_dir   = dirs["plots"]

    # Detect and normalise CSV format
    norm_csv, _ = resolve_labels_csv(segment_dir)

    # If we normalised, temporarily present it as labels.csv so that
    # analyse_dunakeszi (which always opens 'labels.csv') finds the right file.
    raw_csv     = segment_dir / "labels.csv"
    needs_swap  = (norm_csv != raw_csv)
    backup_csv  = segment_dir / "_labels_original_backup.csv"

    if needs_swap:
        # Back up original, copy normalised in
        shutil.copy2(raw_csv, backup_csv)
        shutil.copy2(norm_csv, raw_csv)
        print(f"  ℹ  Swapped normalised CSV into labels.csv for analysis")

    try:
        pkg_parent = Path(__file__).parent.parent
        if str(pkg_parent) not in sys.path:
            sys.path.insert(0, str(pkg_parent))
        from drone_detection import config
        from dunakeszi_pipeline_analysis import analyse_dunakeszi
    except ImportError as e:
        if needs_swap:
            shutil.copy2(backup_csv, raw_csv)
            backup_csv.unlink(missing_ok=True)
        print(f"  ✗  Import failed: {e}")
        print("     Ensure drone_detection is installed or on PYTHONPATH.")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    config.DRIVE_PLOTS = plots_dir

    gt_csv = dirs["gt"] / "ground_truth_segments.csv"

    try:
        out = analyse_dunakeszi(
            pipeline_dir = str(segment_dir),
            cfg          = config,
            gt_csv       = str(gt_csv) if gt_csv.exists() else None,
            splits       = args.splits or None,
            multi_drone  = args.multi_drone,
            max_dist     = float(args.max_dist),
            show_plots   = False,
            save_plots   = True,
            save_csv     = True,
            plots_dir    = str(plots_dir),
        )
    finally:
        if needs_swap and backup_csv.exists():
            shutil.copy2(backup_csv, raw_csv)
            backup_csv.unlink(missing_ok=True)
            print("  ℹ  Restored original labels.csv")

    # Move results CSV into work-dir/results/
    src = segment_dir / "dunakeszi_inference_results.csv"
    if src.exists():
        dst = results_dir / "dunakeszi_inference_results.csv"
        src.replace(dst)
        print(f"\n  💾 Results CSV: {dst}")

    print(f"  📊 Plots: {plots_dir}")
    print(f"\n  {out['n_detected']}/{out['n_segments']} segments detected.")


# ══════════════════════════════════════════════════════════════════════════════
# §3  Summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(args, dirs: dict, t_start: float):
    _header("Run complete")
    print(f"  Work dir      : {dirs['work']}")
    if args.pipeline_dir:
        print(f"  Pipeline dir  : {args.pipeline_dir}")
    print(f"  Splits        : {args.splits or 'all'}")
    print(f"  Max dist      : {args.max_dist} m")
    print(f"  Total time    : {time.time()-t_start:.1f}s\n")
    for label, d in [
        ("Ground truth",   dirs["gt"]),
        ("Extracted",      dirs["extracted"]),
        ("Pipeline-ready", dirs["pipeline_ready"]),
        ("Results",        dirs["results"]),
        ("Plots",          dirs["plots"]),
    ]:
        print(f"  {'✓' if d.exists() else '–'}  {label:<18s}: {d}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Local runner for the full Dunakeszi pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Paths
    ap.add_argument("--work-dir", required=True,
                    help="Root working directory; all outputs go here.")
    ap.add_argument("--wav-dir", default=None,
                    help="Polywav .wav directory (required for 'extract' stage).")
    ap.add_argument(
        "--pipeline-dir", default=None, metavar="DIR",
        help=(
            "Flat directory with already-extracted segments (WAVs + labels.csv). "
            "Accepted: extractor format (azimuth_at_clip_mid_deg) or prepare "
            "format (azimuth_deg). Format is auto-detected; azimuth conversion "
            "is applied automatically if needed. Bypasses work-dir/pipeline_ready/."
        ),
    )

    # Stage
    ap.add_argument("--stage", choices=["ground_truth","extract","prepare","analyse","all"],
                    default="all", help="Stage(s) to run (default: all).")

    # Extraction
    ap.add_argument("--array", choices=["BK-6-E","BK-6-W"], default="BK-6-E")
    ap.add_argument("--clip-duration", type=float, default=3.0,
                    help="Clip length in seconds (0 = full segment).")
    ap.add_argument("--clip-position", type=float, default=0.5,
                    help="Clip position 0=start 0.5=centre 1=end.")
    ap.add_argument("--splits", nargs="+", choices=["train","val","test"], default=None)
    ap.add_argument("--skip-unlabelled", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--no-zip", action="store_true")

    # Ground truth
    ap.add_argument("--verify-mems", metavar="WAV_PATH", default=None)

    # Analysis
    ap.add_argument("--max-dist", type=float, default=100.0,
                    help="MAX_LOCALIZATION_DIST metres (use 350 for show_10).")
    ap.add_argument("--multi-drone", action="store_true")

    args = ap.parse_args()

    work = Path(args.work_dir)
    dirs = {
        "work":           work,
        "gt":             work / "ground_truth",
        "extracted":      work / "extracted",
        "pipeline_ready": work / "pipeline_ready",
        "results":        work / "results",
        "plots":          work / "results" / "plots",
    }
    work.mkdir(parents=True, exist_ok=True)

    stages = (["ground_truth","extract","prepare","analyse"]
              if args.stage == "all" else [args.stage])

    t0 = time.time()
    _header(f"Dunakeszi local pipeline  —  {', '.join(stages)}")
    print(f"  Work dir      : {work}")
    if args.pipeline_dir:
        print(f"  Pipeline dir  : {args.pipeline_dir}")
    if args.wav_dir:
        print(f"  WAV dir       : {args.wav_dir}")
    print(f"  Array         : {args.array}")
    print(f"  Splits        : {args.splits or 'all'}")
    print(f"  Clip          : {args.clip_duration}s @ {args.clip_position}")
    print(f"  Max dist      : {args.max_dist} m")

    for stage in stages:
        {"ground_truth": stage_ground_truth,
         "extract":      stage_extract,
         "prepare":      stage_prepare,
         "analyse":      stage_analyse}[stage](args, dirs)

    _print_summary(args, dirs, t0)


if __name__ == "__main__":
    main()

    # python run_dunakeszi_local.py --work-dir ../dunakeszi_test_segments_P --pipeline-dir ../dunakeszi_test_segments_P --stage analyse --multi-drone