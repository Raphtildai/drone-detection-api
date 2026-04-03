#!/usr/bin/env python3
"""
sort_urbansound8k.py
────────────────────
Sorts UrbanSound8K clips into per-class folders and gives every file a
self-describing name:

    <class>/<class>_fold<F>_<freesound_id>_<slice_idx>.wav
    e.g.  dog_bark/dog_bark_fold3_100032_0.wav

The original files are NEVER deleted – the script copies by default.
Pass --move to move instead of copy (faster, saves disk space).

Usage
─────
  python sort_urbansound8k.py --dataset /path/to/UrbanSound8K

Optional flags
──────────────
  --output DIR           Where to write sorted files  [default: ./UrbanSound8K_sorted]
  --move                 Move files instead of copying
  --dry-run              Print what would happen without touching any files
  --classes cls1 cls2    Only process these classes  (space-separated)
  --no-fold-prefix       Omit fold number from output filename
  --overwrite            Re-copy/move even if destination already exists

UrbanSound8K class list
───────────────────────
  air_conditioner · car_horn · children_playing · dog_bark · drilling
  engine_idling   · gun_shot · jackhammer       · siren    · street_music
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


# ── the 10 official UrbanSound8K classes (classID 0-9) ───────────────────────
ALL_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sort & rename UrbanSound8K clips by class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", required=True,
        help=(
            "Root of the extracted UrbanSound8K folder "
            "(the one that contains 'audio/' and 'metadata/')."
        ),
    )
    p.add_argument(
        "--output", default="UrbanSound8K_sorted",
        help="Destination root folder  [default: ./UrbanSound8K_sorted]",
    )
    p.add_argument("--move",           action="store_true", help="Move instead of copy")
    p.add_argument("--dry-run",        action="store_true", help="Preview only, no file operations")
    p.add_argument("--overwrite",      action="store_true", help="Overwrite existing destination files")
    p.add_argument("--no-fold-prefix", action="store_true", help="Omit fold number from output filenames")
    p.add_argument(
        "--classes", nargs="+", metavar="CLASS",
        help="Only process these classes (default: all 10)",
    )
    return p.parse_args(argv)


def load_metadata(metadata_csv: Path) -> list:
    rows = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_dest_name(row: dict, include_fold: bool) -> str:
    """
    Build a self-describing filename from one CSV row.

    Original UrbanSound8K filename format:
        <fsID>-<classID>-<occurrenceID>-<sliceID>.wav
        e.g.  100032-0-0-0.wav

    Output format (with fold):
        <class>_fold<F>_<fsID>_<sliceID>.wav
        e.g.  dog_bark_fold3_100032_0.wav

    Output format (without fold):
        <class>_<fsID>_<sliceID>.wav
        e.g.  dog_bark_100032_0.wav
    """
    cls_name  = row["class"].strip()
    fold      = row["fold"].strip()
    orig_stem = Path(row["slice_file_name"]).stem   # strip .wav
    parts     = orig_stem.split("-")

    fs_id     = parts[0] if len(parts) > 0 else orig_stem
    slice_idx = parts[3] if len(parts) > 3 else "0"

    if include_fold:
        return f"{cls_name}_fold{fold}_{fs_id}_{slice_idx}.wav"
    return f"{cls_name}_{fs_id}_{slice_idx}.wav"


def print_stats(counters: dict):
    print("\n📊 Sort summary")
    header = f"  {'Class':<22} {'done':>10} {'skipped':>9} {'missing':>10}"
    print(header)
    print("  " + "─" * 54)
    total_done = total_skip = total_miss = 0
    for cls in sorted(counters):
        c = counters[cls]
        print(f"  {cls:<22} {c['done']:>10} {c['skipped']:>9} {c['missing']:>10}")
        total_done += c["done"]
        total_skip += c["skipped"]
        total_miss += c["missing"]
    print("  " + "─" * 54)
    print(f"  {'TOTAL':<22} {total_done:>10} {total_skip:>9} {total_miss:>10}")


def preview_tree(output_root: Path, max_classes: int = 5, samples: int = 3):
    print("\n📂 Output structure preview:")
    shown = 0
    all_dirs = sorted(d for d in output_root.iterdir() if d.is_dir())
    for cls_dir in all_dirs:
        files = sorted(cls_dir.glob("*.wav"))
        print(f"   {cls_dir.name}/  ({len(files)} files)")
        for f in files[:samples]:
            print(f"     └─ {f.name}")
        if len(files) > samples:
            print(f"     └─ … {len(files) - samples} more")
        shown += 1
        if shown >= max_classes:
            remaining = len(all_dirs) - max_classes
            if remaining > 0:
                print(f"   … and {remaining} more class folder(s)")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)

    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root  = Path(args.output).expanduser().resolve()

    # ── locate audio root ─────────────────────────────────────────────────────
    # Supports both:
    #   /UrbanSound8K/audio/fold1/…   (standard extracted layout)
    #   /audio/fold1/…                (user already inside the inner folder)
    audio_root = next(
        (c for c in [
            dataset_root / "UrbanSound8K" / "audio",
            dataset_root / "audio",
        ] if c.exists()),
        None,
    )
    if audio_root is None:
        print(
            f"❌  Could not find an 'audio/' folder under:\n   {dataset_root}\n\n"
            "    Make sure --dataset points at the extracted archive root,\n"
            "    e.g.  --dataset /data/UrbanSound8K"
        )
        sys.exit(1)

    # ── locate metadata CSV ───────────────────────────────────────────────────
    meta_csv = next(
        (c for c in [
            dataset_root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv",
            dataset_root / "metadata" / "UrbanSound8K.csv",
        ] if c.exists()),
        None,
    )
    if meta_csv is None:
        print(
            f"❌  Could not find metadata/UrbanSound8K.csv under:\n   {dataset_root}"
        )
        sys.exit(1)

    print(f"📁  Dataset audio : {audio_root}")
    print(f"📄  Metadata CSV  : {meta_csv}")
    print(f"📁  Output root   : {output_root}")
    mode_label = "DRY RUN" if args.dry_run else ("move" if args.move else "copy")
    print(f"⚙️   Mode          : {mode_label}\n")

    # ── load and filter metadata ──────────────────────────────────────────────
    all_rows = load_metadata(meta_csv)
    print(f"   Loaded {len(all_rows)} metadata rows.")

    target_classes = set(args.classes) if args.classes else set(ALL_CLASSES)
    unknown = target_classes - set(ALL_CLASSES)
    if unknown:
        print(f"⚠️   Unknown class(es) ignored: {', '.join(sorted(unknown))}")
        target_classes -= unknown

    rows = [r for r in all_rows if r["class"].strip() in target_classes]
    print(
        f"   Processing {len(rows)} rows for: "
        f"{', '.join(sorted(target_classes))}\n"
    )

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    # ── counters ──────────────────────────────────────────────────────────────
    counters = {cls: {"done": 0, "skipped": 0, "missing": 0} for cls in target_classes}
    manifest = []

    # ── process every row ─────────────────────────────────────────────────────
    for row in rows:
        cls_name = row["class"].strip()
        fold     = row["fold"].strip()
        src_name = row["slice_file_name"].strip()
        src_path = audio_root / f"fold{fold}" / src_name

        if not src_path.exists():
            print(f"  ⚠️   Missing source: {src_path}")
            counters[cls_name]["missing"] += 1
            continue

        dest_name = build_dest_name(row, include_fold=not args.no_fold_prefix)
        dest_dir  = output_root / cls_name
        dest_path = dest_dir / dest_name

        if dest_path.exists() and not args.overwrite:
            counters[cls_name]["skipped"] += 1
            continue

        if args.dry_run:
            print(f"  [DRY]  {src_name}  →  {cls_name}/{dest_name}")
            counters[cls_name]["done"] += 1
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                if args.move:
                    shutil.move(str(src_path), str(dest_path))
                else:
                    shutil.copy2(str(src_path), str(dest_path))
                counters[cls_name]["done"] += 1
            except Exception as e:
                print(f"  ⚠️   Failed {src_name}: {e}")
                counters[cls_name]["missing"] += 1

        manifest.append({
            "original_file": src_name,
            "sorted_file":   f"{cls_name}/{dest_name}",
            "class":         cls_name,
            "class_id":      int(row["classID"]),
            "fold":          int(fold),
            "freesound_id":  row["fsID"],
            "salience":      row.get("salience", ""),
        })

    # ── write manifest ────────────────────────────────────────────────────────
    if not args.dry_run:
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n📄  Manifest written → {manifest_path}")

    # ── stats & tree ──────────────────────────────────────────────────────────
    print_stats(counters)

    total_done = sum(c["done"] for c in counters.values())
    op = "would process" if args.dry_run else ("moved" if args.move else "copied")
    print(f"\n✅  {op.capitalize()} {total_done} files  →  {output_root}\n")

    if not args.dry_run and output_root.exists():
        preview_tree(output_root)

# # Usage
# # Copy and sort (safe – originals untouched)
# python sort_urbansound8k.py --dataset /path/to/UrbanSound8K

# # Specify a custom output folder
# python sort_urbansound8k.py --dataset /path/to/UrbanSound8K --output ./sorted_sounds

# # Preview first without touching any files
# python sort_urbansound8k.py --dataset /path/to/UrbanSound8K --dry-run

# # Only extract the classes relevant to your drone project
# python sort_urbansound8k.py --dataset /path/to/UrbanSound8K \
#     --classes air_conditioner engine_idling jackhammer drilling siren

# # Save disk space by moving instead of copying
# python sort_urbansound8k.py --dataset /path/to/UrbanSound8K --move

if __name__ == "__main__":
    main()
