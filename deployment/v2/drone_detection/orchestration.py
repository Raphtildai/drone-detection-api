# -*- coding: utf-8 -*-
"""
notebook.py
───────────
High-level entry points for Colab / Jupyter notebooks.

Training
────────
train_detection()        — full detection training pipeline
train_localization()     — full localization training pipeline
train_all()              — run both in sequence

Custom dataset setup
────────────────────
configure_custom_dataset()          — normalise an uploaded dataset root
quickstart_notebook_setup()         — one-liner Colab setup
quickstart_notebook_setup_v2()      — stricter setup with structure validation
validate_custom_dataset_structure() — check folder layout before training
describe_custom_dataset()           — print file counts per subfolder
get_notebook_setup_template()       — return a copy-pastable code snippet
print_notebook_setup_template()     — print the snippet to stdout

Data helpers
────────────
upload_custom_dataset_artifacts()   — Colab file upload + optional ZIP extract
import_custom_builder_dataset()     — copy cleaned clips into the pipeline dirs
generate_mixed_drone_training_audio() — mix drone + background at random SNR
collect_background_pool()           — gather all background WAVs from disk

Diagnostics
───────────
report_detection_split_counts()
audit_localization_labels()
diagnose_uavirbase()
verify_tdoa_accuracy()
quick_demo()
launch_ui()
"""

import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from .config import Config, config, AUDIO_EXTS
from .audio_processing import AudioProcessor, synthesise_drone
from .utils import (
    bandpass, gcc_phat,
    db_to_gain, load_audio_any, normalize_peak,
    random_crop_or_loop, mix_at_snr,
    safe_slug, _set_seed, _ensure_remotezip,
    infer_group_id, grouped_split_paths,
)
from .datasets import (
    DroneAudioDatasetManager,
    MelCacheManager,
    UaVirBASEDatasetManager,
    get_det_dataloaders,
    report_detection_split_counts,
    parse_label_json,
    _position_grouped_split,
)
from .training import (
    DetectionTrainer,
    LocalizationTrainer,
    train_localization,
)
from .inference import (
    load_detection_model,
    load_localization_model,
    detect,
    localize,
    run_pipeline,
)
from .tracking import KalmanTrack, KalmanTracker
from .multidrone import localize_multi_drone
from .visualization import plot_multi_drone_positions, plot_track_trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Custom dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

def _list_audio_files(root: Optional[Path], exts=AUDIO_EXTS) -> List[Path]:
    if root is None or not Path(root).exists():
        return []
    files = []
    for ext in exts:
        files.extend(Path(root).rglob(f"*{ext}"))
    return sorted([p for p in files if p.is_file()])


def _copy_audio_file(src: Path, dst: Path, sr: int, force: bool = False) -> bool:
    """Copy src → dst, resampling to sr.  Returns True if file was written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return False
    y = load_audio_any(src, sr)
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return False
    sf.write(str(dst), normalize_peak(y), sr)
    return True


def _copy_tree_audio(src: Path, dst: Path, exts=AUDIO_EXTS) -> int:
    """Recursively copy audio files from src to dst (unique names)."""
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        out = dst / f.name
        stem, suf = out.stem, out.suffix
        k = 1
        while out.exists():
            out = dst / f"{stem}_{k}{suf}"
            k += 1
        shutil.copy2(str(f), str(out))
        n += 1
    return n


def _copy_grouped_files(
    files:        List[Path],
    split_to_dir: Dict[str, Path],
    sr:           int,
    force:        bool  = False,
    test_fraction: float = 0.10,
    val_fraction:  float = 0.15,
    seed:         int   = 42,
    prefix:       str   = "custom",
) -> Dict[str, int]:
    stats = {"train": 0, "val": 0, "test": 0, "failed": 0, "skipped": 0}
    if not files:
        return stats
    rng = random.Random(seed)
    gid_to_files: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        gid_to_files[infer_group_id(f)].append(f)
    grouped = list(gid_to_files.items())
    rng.shuffle(grouped)
    n       = len(grouped)
    n_test  = max(1, int(round(n * test_fraction))) if n >= 3 else 0
    n_val   = max(1, int(round(n * val_fraction)))   if n >= 4 else min(1, max(0, n - n_test - 1))
    n_train = max(0, n - n_val - n_test)
    custom  = {"train": [], "val": [], "test": []}
    for i, (_, flist) in enumerate(grouped):
        if i < n_train:                 custom["train"].extend(flist)
        elif i < n_train + n_val:       custom["val"].extend(flist)
        else:                           custom["test"].extend(flist)
    for split, flist in custom.items():
        out_dir = split_to_dir[split]
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in flist:
            dst = out_dir / f"{prefix}_{safe_slug(src.stem)}.wav"
            try:
                changed = _copy_audio_file(src, dst, sr=sr, force=force)
                stats[split if changed else "skipped"] += (1 if changed else 1)
            except Exception:
                stats["failed"] += 1
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# collect_background_pool / generate_mixed_drone_training_audio
# ══════════════════════════════════════════════════════════════════════════════

def collect_background_pool(cfg: Optional[Config] = None) -> Dict[str, List[Path]]:
    """
    Gather all available background (non-drone) audio files into a pool.
    Also includes custom builder backgrounds when present.
    """
    cfg  = cfg or config
    pool = {k: [] for k in ["speech", "crowd", "wind", "traffic", "non_drone"]}

    for split in ["train", "val", "test"]:
        d = cfg.PROCESSED_DIR / "detection" / split / "non_drone"
        if not d.exists():
            continue
        for f in d.glob("*.wav"):
            pool["non_drone"].append(f)
            nm = f.stem.lower()
            for kws, bkt in [
                (["speech", "talk", "voice"],         "speech"),
                (["crowd", "market"],                 "crowd"),
                (["wind", "breeze"],                  "wind"),
                (["traffic", "car", "road", "engine", "siren"], "traffic"),
            ]:
                if any(k in nm for k in kws):
                    pool[bkt].append(f)

    # Custom builder backgrounds
    custom_bg = Path(getattr(cfg, "CUSTOM_DATASET_IMPORTED_ROOT", "")) / "background_pool"
    if custom_bg.exists():
        for f in _list_audio_files(custom_bg, AUDIO_EXTS):
            pool["non_drone"].append(f)

    # Fill empty sub-categories from the general pool
    for k in ["speech", "crowd", "wind", "traffic"]:
        if not pool[k]:
            pool[k] = list(pool["non_drone"])

    # De-duplicate each list
    for k in pool:
        seen, uniq = set(), []
        for p in pool[k]:
            s = str(p)
            if s not in seen:
                uniq.append(p); seen.add(s)
        pool[k] = uniq

    print("📚 Background pool:")
    for k in pool:
        print(f"   {k:10s}: {len(pool[k])}")
    return pool


def generate_mixed_drone_training_audio(
    cfg:   Optional[Config] = None,
    force: bool = False,
):
    """
    Mix clean drone clips with random backgrounds at random SNR and write
    them into the detection train/val directories.

    Uses custom builder clean clips when available (cfg.CUSTOM_DATASET_INCLUDE_CLEAN_IN_MIXING).
    """
    cfg = cfg or config
    ap  = AudioProcessor(cfg)
    train_drone_dir = cfg.PROCESSED_DIR / "detection" / "train" / "drone"
    val_drone_dir   = cfg.PROCESSED_DIR / "detection" / "val"   / "drone"

    if not train_drone_dir.exists():
        print("⚠️  No train/drone directory — run dataset preparation first."); return

    existing = (list(train_drone_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav")) +
                list(val_drone_dir.glob(  f"{cfg.MIX_CACHE_PREFIX}_*.wav")))
    if len(existing) >= cfg.MIXED_DRONE_SAMPLES and not force:
        print(f"✅ Mixed drone audio already exists ({len(existing)} files) — skipping."); return

    if force:
        for d in [train_drone_dir, val_drone_dir]:
            if d.exists():
                for f in d.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"):
                    try: f.unlink()
                    except Exception: pass

    drone_files = [f for f in train_drone_dir.glob("*.wav") if not f.stem.startswith(cfg.MIX_CACHE_PREFIX)]

    # Include custom clean clips
    if getattr(cfg, "CUSTOM_DATASET_INCLUDE_CLEAN_IN_MIXING", True):
        custom_clean = Path(getattr(cfg, "CUSTOM_DATASET_IMPORTED_ROOT", "")) / "clean_drone_segments"
        if custom_clean.exists():
            drone_files.extend(_list_audio_files(custom_clean, AUDIO_EXTS))

    # De-duplicate
    seen, dedup = set(), []
    for p in drone_files:
        if str(p) not in seen:
            dedup.append(p); seen.add(str(p))
    drone_files = dedup

    if not drone_files:
        print("⚠️  No clean drone WAVs found for mixing."); return

    bg_pool = collect_background_pool(cfg)
    usable_bg_labels = [k for k in cfg.MIX_BACKGROUND_LABELS if len(bg_pool.get(k, [])) > 0]
    if not usable_bg_labels:
        print("⚠️  No usable background pool — skipping mixed-drone generation."); return

    n_total = int(cfg.MIXED_DRONE_SAMPLES)
    n_val   = int(n_total * cfg.MIXED_DRONE_VAL_FRAC)
    n_train = n_total - n_val
    print(f"🎛️  Generating mixed drone audio: total={n_total} train={n_train} val={n_val} sources={len(drone_files)}")

    from tqdm.auto import tqdm
    for i in tqdm(range(n_total), desc="Mixing drone+background"):
        split   = "val" if i < n_val else "train"
        out_dir = val_drone_dir if split == "val" else train_drone_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        drone_path = random.choice(drone_files)
        bg_label   = random.choice(usable_bg_labels)
        bg_path    = random.choice(bg_pool[bg_label])
        out_path   = out_dir / f"{cfg.MIX_CACHE_PREFIX}_{i:06d}_{bg_label}_{safe_slug(drone_path.stem)}.wav"
        if out_path.exists() and not force:
            continue
        try:
            drone_y = ap.pad_or_truncate(load_audio_any(drone_path, cfg.SR))
            bg_y    = random_crop_or_loop(load_audio_any(bg_path, cfg.SR), len(drone_y))
            drone_y = np.clip(drone_y * db_to_gain(random.uniform(*cfg.MIX_GAIN_RANGE_DB)), -1, 1).astype(np.float32)
            bg_y    = np.clip(bg_y    * db_to_gain(random.uniform(*cfg.MIX_BG_GAIN_RANGE_DB)), -1, 1).astype(np.float32)
            mixed   = mix_at_snr(drone_y, bg_y, random.uniform(*cfg.MIX_SNR_DB_RANGE))
            if random.random() < 0.35:
                mixed = np.roll(mixed, random.randint(0, len(mixed) // 10)).astype(np.float32)
            if random.random() < 0.40:
                noise  = np.random.randn(len(mixed)).astype(np.float32)
                noise /= (np.max(np.abs(noise)) + 1e-8)
                mixed  = normalize_peak(mixed + 0.01 * noise)
            sf.write(str(out_path), mixed, cfg.SR)
        except Exception as e:
            print(f"   ⚠️  Failed mix {i}: {e}")

    made = (len(list(train_drone_dir.glob(f"{cfg.MIX_CACHE_PREFIX}_*.wav"))) +
            len(list(val_drone_dir.glob(  f"{cfg.MIX_CACHE_PREFIX}_*.wav"))))
    print(f"✅ Mixed drone generation complete ({made} files).")


# ══════════════════════════════════════════════════════════════════════════════
# Custom dataset import
# ══════════════════════════════════════════════════════════════════════════════

def import_custom_builder_dataset(
    cfg:                                Optional[Config] = None,
    builder_root:                       Optional[Path]   = None,
    force:                              bool  = False,
    include_background_pool_as_non_drone: Optional[bool] = None,
    include_clean_in_train:             Optional[bool]   = None,
    prefer_builder_splits:              Optional[bool]   = None,
    min_test_fraction:                  Optional[float]  = None,
) -> Dict[str, Any]:
    """
    Copy cleaned drone clips and optional background clips from a custom
    builder dataset root into the detection pipeline directories.

    Returns a summary dict with counts of imported files.
    """
    cfg = cfg or config
    builder_root = Path(builder_root or getattr(cfg, "CUSTOM_DATASET_ROOT", "")) or None
    if builder_root is None or not builder_root.exists():
        print("ℹ️  No custom builder dataset root provided — skipping.")
        return {"enabled": False, "imported": 0}

    include_background_pool_as_non_drone = (
        getattr(cfg, "CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE", True)
        if include_background_pool_as_non_drone is None else include_background_pool_as_non_drone
    )
    include_clean_in_train = (
        getattr(cfg, "CUSTOM_DATASET_INCLUDE_CLEAN_IN_TRAIN", True)
        if include_clean_in_train is None else include_clean_in_train
    )
    prefer_builder_splits = (
        getattr(cfg, "CUSTOM_DATASET_PREFER_BUILDER_SPLITS", True)
        if prefer_builder_splits is None else prefer_builder_splits
    )
    min_test_fraction = (
        getattr(cfg, "CUSTOM_DATASET_MIN_TEST_FRACTION", 0.10)
        if min_test_fraction is None else min_test_fraction
    )

    det_root      = cfg.PROCESSED_DIR / "detection"
    imported_root = Path(getattr(cfg, "CUSTOM_DATASET_IMPORTED_ROOT", str(cfg.RAW_DIR / "custom_builder_import")))
    imported_clean = imported_root / "clean_drone_segments"
    imported_bg    = imported_root / "background_pool"
    imported_clean.mkdir(parents=True, exist_ok=True)
    imported_bg.mkdir(parents=True, exist_ok=True)

    clean_subdir  = getattr(cfg, "CUSTOM_DATASET_CLEAN_SUBDIR",  "clean_drone_segments")
    manual_subdir = getattr(cfg, "CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR", "manual_clean/clean_drone_sections")
    train_subdir  = getattr(cfg, "CUSTOM_DATASET_TRAIN_SUBDIR",  "train/drone")
    val_subdir    = getattr(cfg, "CUSTOM_DATASET_VAL_SUBDIR",    "val/drone")
    bg_subdir     = getattr(cfg, "CUSTOM_DATASET_BGPOOL_SUBDIR", "background_pool")

    stats = {
        "enabled": True, "builder_root": str(builder_root),
        "train_added": 0, "val_added": 0, "test_added": 0,
        "clean_imported": 0, "background_imported": 0,
        "failed": 0, "skipped": 0,
    }

    # Import builder train/val positives
    for split_name, src_subdir in [("train", train_subdir), ("val", val_subdir)]:
        files = _list_audio_files(builder_root / src_subdir, AUDIO_EXTS)
        dst_dir = det_root / split_name / "drone"
        for src in files:
            dst = dst_dir / f"custommix_{safe_slug(src.stem)}.wav"
            try:
                changed = _copy_audio_file(src, dst, sr=cfg.SR, force=force)
                stats[f"{split_name}_added" if changed else "skipped"] += 1
            except Exception:
                stats["failed"] += 1

    # Import clean drone clips
    clean_files = _list_audio_files(builder_root / clean_subdir, AUDIO_EXTS)
    if not clean_files and (builder_root / manual_subdir).exists():
        clean_files = _list_audio_files(builder_root / manual_subdir, AUDIO_EXTS)

    for src in clean_files:
        dst = imported_clean / f"clean_{safe_slug(src.stem)}.wav"
        try:
            changed = _copy_audio_file(src, dst, sr=cfg.SR, force=force)
            stats["clean_imported" if changed else "skipped"] += 1
        except Exception:
            stats["failed"] += 1

    if include_clean_in_train and clean_files:
        split_dirs = {
            "train": det_root / "train" / "drone",
            "val":   det_root / "val"   / "drone",
            "test":  det_root / "test"  / "drone",
        }
        if prefer_builder_splits:
            split_stats = _copy_grouped_files(
                clean_files, split_dirs, sr=cfg.SR, force=force,
                test_fraction=min_test_fraction,
                val_fraction=getattr(cfg, "MIXED_DRONE_VAL_FRAC", 0.15),
                seed=cfg.SEED, prefix="customclean",
            )
        else:
            split_stats = {"train": 0, "val": 0, "test": 0, "failed": 0, "skipped": 0}
            for src in clean_files:
                dst = split_dirs["train"] / f"customclean_{safe_slug(src.stem)}.wav"
                try:
                    changed = _copy_audio_file(src, dst, sr=cfg.SR, force=force)
                    split_stats["train" if changed else "skipped"] += 1
                except Exception:
                    split_stats["failed"] += 1
        for k in ["train", "val", "test", "failed", "skipped"]:
            stats[f"{'train_added' if k=='train' else k+'_added' if k in ['val','test'] else k}"] += split_stats.get(k, 0)

    # Import background pool as non-drone negatives
    if include_background_pool_as_non_drone and (builder_root / bg_subdir).exists():
        bg_files = _list_audio_files(builder_root / bg_subdir, AUDIO_EXTS)
        for src in bg_files:
            try:
                changed = _copy_audio_file(src, imported_bg / f"bg_{safe_slug(src.stem)}.wav", sr=cfg.SR, force=force)
                stats["background_imported" if changed else "skipped"] += 1
            except Exception:
                stats["failed"] += 1
        split_dirs = {
            "train": det_root / "train" / "non_drone",
            "val":   det_root / "val"   / "non_drone",
            "test":  det_root / "test"  / "non_drone",
        }
        _copy_grouped_files(bg_files, split_dirs, sr=cfg.SR, force=force,
                            test_fraction=min_test_fraction,
                            val_fraction=getattr(cfg, "MIXED_DRONE_VAL_FRAC", 0.15),
                            seed=cfg.SEED + 17, prefix="custombg")

    print("📦 Custom builder import summary:")
    for k, v in stats.items():
        if k not in ("enabled", "builder_root"):
            print(f"   {k:24s}: {v}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Custom dataset validation / description
# ══════════════════════════════════════════════════════════════════════════════

def describe_custom_dataset(
    cfg:          Optional[Config] = None,
    dataset_root: Optional[str]    = None,
) -> Dict[str, int]:
    cfg  = cfg or config
    root = Path(dataset_root or getattr(cfg, "CUSTOM_DATASET_ROOT", ""))
    if not root.exists():
        print(f"⚠️  Custom dataset root not found: {root}"); return {}
    mapping = {
        "clean":          root / getattr(cfg, "CUSTOM_DATASET_CLEAN_SUBDIR",  "clean_drone_segments"),
        "train_drone":    root / getattr(cfg, "CUSTOM_DATASET_TRAIN_SUBDIR",  "train/drone"),
        "val_drone":      root / getattr(cfg, "CUSTOM_DATASET_VAL_SUBDIR",    "val/drone"),
        "background_pool":root / getattr(cfg, "CUSTOM_DATASET_BGPOOL_SUBDIR", "background_pool"),
        "manual_clean":   root / getattr(cfg, "CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR", "manual_clean/clean_drone_sections"),
    }
    print("📦 Custom dataset contents")
    stats = {}
    for name, path in mapping.items():
        n = len(_list_audio_files(path, AUDIO_EXTS)) if path.exists() else 0
        stats[name] = n
        print(f"   {name:15s}: {n}")
    print(f"   root           : {root}")
    return stats


def validate_custom_dataset_structure(
    cfg:               Optional[Config] = None,
    dataset_root:      Optional[str]    = None,
    require_any_positive: bool  = True,
    min_positive_files:   int   = 1,
    min_background_files: int   = 0,
    verbose:              bool  = True,
) -> Dict[str, Any]:
    """
    Validate that the custom dataset has enough positive files.
    Returns a dict with counts, is_valid flag, and warnings.
    """
    cfg  = cfg or config
    root = Path(dataset_root or getattr(cfg, "CUSTOM_DATASET_ROOT", ""))
    if not root.exists():
        raise FileNotFoundError(f"Custom dataset root does not exist: {root}")

    mapping = {
        "clean_drone_segments": root / getattr(cfg, "CUSTOM_DATASET_CLEAN_SUBDIR",  "clean_drone_segments"),
        "manual_clean_sections":root / getattr(cfg, "CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR", "manual_clean/clean_drone_sections"),
        "train_drone":          root / getattr(cfg, "CUSTOM_DATASET_TRAIN_SUBDIR",  "train/drone"),
        "val_drone":            root / getattr(cfg, "CUSTOM_DATASET_VAL_SUBDIR",    "val/drone"),
        "test_drone":           root / "test/drone",
        "background_pool":      root / getattr(cfg, "CUSTOM_DATASET_BGPOOL_SUBDIR", "background_pool"),
    }
    counts = {name: len(_list_audio_files(path, AUDIO_EXTS)) if path.exists() else 0
              for name, path in mapping.items()}

    positive_count   = sum(counts[k] for k in ["clean_drone_segments","manual_clean_sections","train_drone","val_drone","test_drone"])
    background_count = counts["background_pool"]

    warnings_list = []
    if require_any_positive and positive_count < min_positive_files:
        warnings_list.append(f"Not enough positive files. Required ≥ {min_positive_files}, found {positive_count}.")
    if background_count < min_background_files:
        warnings_list.append(f"Not enough background_pool files. Required ≥ {min_background_files}, found {background_count}.")

    preferred_positive_root = None
    for k in ["manual_clean_sections", "clean_drone_segments", "train_drone"]:
        if counts[k] > 0:
            preferred_positive_root = str(mapping[k]); break

    result = {
        "root": str(root), "counts": counts,
        "positive_count": int(positive_count), "background_count": int(background_count),
        "is_valid": len(warnings_list) == 0, "warnings": warnings_list,
        "preferred_positive_root": preferred_positive_root,
    }
    if verbose:
        print("🧪 Custom dataset validation")
        print(f"   root                    : {root}")
        for k, v in counts.items():
            print(f"   {k:24s}: {v}")
        print(f"   positive_count          : {positive_count}")
        print(f"   background_count        : {background_count}")
        print(f"   preferred_positive_root : {preferred_positive_root}")
        if warnings_list:
            print("⚠️  Validation warnings")
            for w in warnings_list:
                print(f"   - {w}")
        else:
            print("✅ Structure looks usable")
    return result


def configure_custom_dataset(
    cfg:          Optional[Config] = None,
    dataset_root: Optional[str]    = None,
    zip_path:     Optional[str]    = None,
    extract_dir:  Optional[str]    = None,
    include_background_pool_as_non_drone: bool = True,
    prefer_manual_clean: bool = True,
) -> str:
    """Normalise and validate a custom dataset root; update cfg accordingly."""
    cfg = cfg or config
    base_in = Path(dataset_root) if dataset_root else Path(extract_dir or (cfg.LOCAL_BASE / "custom_dataset_input"))
    if zip_path:
        base_in.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_in)
    if not base_in.exists():
        raise FileNotFoundError(f"Custom dataset root does not exist: {base_in}")
    cfg.CUSTOM_DATASET_ROOT    = str(base_in)
    cfg.CUSTOM_DATASET_ENABLED = True
    if include_background_pool_as_non_drone is not None:
        cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE = bool(include_background_pool_as_non_drone)
    print(f"✅ Custom dataset configured: {base_in}")
    return str(base_in)


# ══════════════════════════════════════════════════════════════════════════════
# Training entry points
# ══════════════════════════════════════════════════════════════════════════════

def train_detection(
    cfg:                              Optional[Config] = None,
    epochs:                           Optional[int]   = None,
    resume:                           bool  = True,
    force_rebuild_cache:              bool  = False,
    force_regen_mixed_audio:          bool  = False,
    custom_builder_root:              Optional[str]   = None,
    use_custom_builder:               Optional[bool]  = None,
    import_custom_backgrounds_as_non_drone: Optional[bool] = None,
    download_builtin_detection_dataset: Optional[bool] = None,
    download_external_audio:          Optional[bool]  = None,
):
    cfg = cfg or config
    cfg.ensure_dirs()
    _set_seed(cfg.SEED)

    if custom_builder_root is not None:
        cfg.CUSTOM_DATASET_ROOT = str(custom_builder_root)
    if use_custom_builder is None:
        use_custom_builder = bool(getattr(cfg, "CUSTOM_DATASET_ROOT", ""))
    if import_custom_backgrounds_as_non_drone is not None:
        cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE = bool(import_custom_backgrounds_as_non_drone)
    if download_builtin_detection_dataset is None:
        download_builtin_detection_dataset = bool(getattr(cfg, "ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD", True))
    if download_external_audio is None:
        download_external_audio = bool(getattr(cfg, "ALLOW_EXTERNAL_AUDIO_DOWNLOADS", False))

    print("=" * 70)
    print("  STAGE 1 — Detection Model")
    print("=" * 70)

    # ── SOURCE 1: Builtin GitHub DroneAudioDataset ────────────────────────
    if download_builtin_detection_dataset:
        print("\n📥 [1/3] Builtin DroneAudioDataset …")
        DroneAudioDatasetManager(cfg).prepare()
    else:
        print("ℹ️  [1/3] Skipping built-in dataset download.")
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                (cfg.PROCESSED_DIR / "detection" / split / label).mkdir(parents=True, exist_ok=True)

    # ── SOURCE 2: External scraped audio ─────────────────────────────────
    if download_external_audio:
        print("\n🌐 [2/3] External audio scraping …")
        from .dataset_builder import AudioWebScraper, _incorporate_scraped_audio
        AudioWebScraper(cfg).download(force=False)
        _incorporate_scraped_audio(cfg, force=False)
    else:
        print("ℹ️  [2/3] Skipping external audio scraping.")

    # ── SOURCE 3: Custom builder dataset ─────────────────────────────────
    if use_custom_builder:
        print("\n📦 [3/3] Custom builder dataset …")
        import_custom_builder_dataset(
            cfg, getattr(cfg, "CUSTOM_DATASET_ROOT", None), force=False,
            include_background_pool_as_non_drone=cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE,
        )
    else:
        print("ℹ️  [3/3] No custom builder dataset configured.")

    # ── Audit combined sources before proceeding ──────────────────────────
    print("\n📊 Combined dataset audit (pre-mix):")
    det = cfg.PROCESSED_DIR / "detection"
    total_drone = total_non_drone = 0
    for split in ["train", "val", "test"]:
        for label in ["drone", "non_drone"]:
            d = det / split / label
            if not d.exists():
                print(f"   ⚠️  MISSING: {split}/{label}")
                continue
            all_wavs  = list(d.glob("*.wav"))
            builtin   = [f for f in all_wavs if not f.stem.startswith(("custom", "mixdrone", "synth"))]
            custom    = [f for f in all_wavs if f.stem.startswith("custom")]
            mixed     = [f for f in all_wavs if f.stem.startswith("mixdrone")]
            print(f"   {split:5s}/{label:10s}  total={len(all_wavs):4d}  "
                  f"builtin={len(builtin):4d}  custom={len(custom):4d}  mixed={len(mixed):4d}")
            if label == "drone":
                total_drone     += len(all_wavs)
            else:
                total_non_drone += len(all_wavs)

    # Hard-stop if no drone files at all — nothing to train on
    if total_drone == 0:
        raise RuntimeError(
            "❌ No drone audio files found after all import steps. "
            "Check download flags and custom dataset root."
        )
    if total_non_drone == 0:
        raise RuntimeError(
            "❌ No non-drone audio files found after all import steps. "
            "Check download flags and background pool."
        )
    print(f"\n   ✅ Total drone={total_drone}  non_drone={total_non_drone}  "
          f"combined={total_drone + total_non_drone}")

    # ── Mixed augmentation, cache, training ──────────────────────────────
    generate_mixed_drone_training_audio(cfg, force=force_regen_mixed_audio)
    report_detection_split_counts(cfg)

    mcm = MelCacheManager(cfg)
    mcm.build(force=force_rebuild_cache)
    mcm.inject_synthetic(force=False)

    counts = mcm.count()
    print("\n📊 MEL CACHE CLASS BALANCE")
    for key, n in counts.items():
        print(f"  {key:28s}  {n:6d}")
    print()

    try:
        tr_l, va_l, te_l = get_det_dataloaders(cfg)
    except RuntimeError as e:
        print(f"❌ Could not build dataloaders: {e}"); return

    tr = DetectionTrainer(cfg)
    tr._set_loaders(tr_l, va_l, te_l)
    tr.run(epochs=epochs, resume=resume)


def train_localization(
    cfg:                     Optional[Config] = None,
    epochs:                  Optional[int]    = None,
    use_synthetic_fallback:  bool  = True,
    resume:                  bool  = True,
    reset_best:              bool  = False,
):
    """Full localization training pipeline using the standard trainer."""
    cfg = cfg or config
    cfg.ensure_dirs()
    _set_seed(cfg.SEED)

    print("=" * 65)
    print("  STAGE 2 — Localization Model (UaVirBASE)")
    print("=" * 65)
    um = UaVirBASEDatasetManager(cfg)
    try:
        um.prepare()
    except Exception as e:
        print(f"⚠️  UaVirBASE prepare failed: {e}")

    if reset_best:
        ckpt = cfg.DRIVE_MODELS / "best_localization.pth"
        if ckpt.exists():
            import torch
            ck = torch.load(ckpt, map_location=cfg.DEVICE)
            ck["best_val_loss"] = 1e9
            torch.save(ck, ckpt)
            print("🔄 best_val_loss reset to 1e9")

    tr = LocalizationTrainer(cfg)
    tr.run(cfg.PROCESSED_DIR / "localization",
           epochs=epochs, use_synthetic_fallback=use_synthetic_fallback, resume=resume)


def train_all(
    cfg:                    Optional[Config] = None,
    det_epochs:             int   = 5,
    loc_epochs:             int   = 5,
    use_synthetic_loc:      bool  = True,
    resume:                 bool  = True,
    force_rebuild_cache:    bool  = False,
    custom_builder_root:    Optional[str]  = None,
    use_custom_builder:     Optional[bool] = None,
    import_custom_backgrounds_as_non_drone: Optional[bool] = None,
    download_builtin_detection_dataset: Optional[bool] = None,
    download_external_audio: Optional[bool] = None,
):
    """Run both training stages in sequence."""
    train_detection(
        cfg, det_epochs, resume=resume,
        force_rebuild_cache=force_rebuild_cache, force_regen_mixed_audio=False,
        custom_builder_root=custom_builder_root, use_custom_builder=use_custom_builder,
        import_custom_backgrounds_as_non_drone=import_custom_backgrounds_as_non_drone,
        download_builtin_detection_dataset=download_builtin_detection_dataset,
        download_external_audio=download_external_audio,
    )
    train_localization(cfg, loc_epochs, use_synthetic_loc, resume=resume)
    print("\n✅ Both models trained.")


# ══════════════════════════════════════════════════════════════════════════════
# Quickstart helpers
# ══════════════════════════════════════════════════════════════════════════════

def quickstart_notebook_setup(
    cfg:                             Optional[Config] = None,
    dataset_root:                    Optional[str]    = None,
    zip_path:                        Optional[str]    = None,
    include_background_pool_as_non_drone: bool = True,
    download_builtin_detection_dataset:   bool = False,
    download_external_audio:              bool = False,
):
    """One-liner Colab notebook setup (configure + print summary)."""
    cfg = cfg or config
    if dataset_root or zip_path:
        configure_custom_dataset(cfg, dataset_root=dataset_root, zip_path=zip_path,
                                 include_background_pool_as_non_drone=include_background_pool_as_non_drone)
    cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE = bool(include_background_pool_as_non_drone)
    cfg.ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD     = bool(download_builtin_detection_dataset)
    cfg.ALLOW_EXTERNAL_AUDIO_DOWNLOADS               = bool(download_external_audio)
    print("✅ Notebook setup complete")
    print(f"   CUSTOM_DATASET_ROOT                : {getattr(cfg,'CUSTOM_DATASET_ROOT','')}")
    print(f"   BACKGROUND_POOL_AS_NON_DRONE       : {cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE}")
    print(f"   DOWNLOAD_BUILTIN_DETECTION_DATASET : {cfg.ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD}")
    print(f"   DOWNLOAD_EXTERNAL_AUDIO            : {cfg.ALLOW_EXTERNAL_AUDIO_DOWNLOADS}")
    return cfg


def quickstart_notebook_setup_v2(
    cfg:                             Optional[Config] = None,
    dataset_root:                    Optional[str]    = None,
    zip_path:                        Optional[str]    = None,
    include_background_pool_as_non_drone: bool = True,
    download_builtin_detection_dataset:   bool = False,
    download_external_audio:              bool = False,
    validate_structure:                   bool = True,
    min_positive_files:                   int  = 1,
    min_background_files:                 int  = 0,
):
    """Stricter version of quickstart_notebook_setup with structure validation."""
    cfg  = cfg or config
    root = dataset_root
    if dataset_root or zip_path:
        root = configure_custom_dataset(cfg, dataset_root=dataset_root, zip_path=zip_path,
                                        include_background_pool_as_non_drone=include_background_pool_as_non_drone)
    elif getattr(cfg, "CUSTOM_DATASET_ROOT", ""):
        root = cfg.CUSTOM_DATASET_ROOT
    if root:
        describe_custom_dataset(cfg, root)
        if validate_structure:
            validate_custom_dataset_structure(cfg, root,
                                               require_any_positive=True,
                                               min_positive_files=min_positive_files,
                                               min_background_files=min_background_files)
    quickstart_notebook_setup(cfg, dataset_root=root,
                               include_background_pool_as_non_drone=include_background_pool_as_non_drone,
                               download_builtin_detection_dataset=download_builtin_detection_dataset,
                               download_external_audio=download_external_audio)
    return root


def get_notebook_setup_template(
    cfg:                              Optional[Config] = None,
    dataset_root:                     Optional[str]    = None,
    zip_path:                         Optional[str]    = None,
    include_background_pool_as_non_drone: bool = True,
    download_builtin_detection_dataset:   bool = False,
    download_external_audio:              bool = False,
) -> str:
    """Return a copy-pastable Colab/local notebook setup snippet."""
    dr = repr(dataset_root) if dataset_root else "None"
    zp = repr(zip_path)     if zip_path     else "None"
    return f"""# === Custom dataset setup ===
DATASET_ROOT = {dr}
ZIP_PATH = {zp}

root = configure_custom_dataset(config, dataset_root=DATASET_ROOT, zip_path=ZIP_PATH,
    include_background_pool_as_non_drone={include_background_pool_as_non_drone})
describe_custom_dataset(config, root)
validate_custom_dataset_structure(config, root, require_any_positive=True)

quickstart_notebook_setup_v2(config, dataset_root=root,
    include_background_pool_as_non_drone={include_background_pool_as_non_drone},
    download_builtin_detection_dataset={download_builtin_detection_dataset},
    download_external_audio={download_external_audio})

# Detection only:
train_detection(config, epochs=5, resume=False, force_rebuild_cache=True,
    use_custom_builder=True,
    import_custom_backgrounds_as_non_drone={include_background_pool_as_non_drone},
    download_builtin_detection_dataset={download_builtin_detection_dataset})

# Full pipeline:
# train_all(config, det_epochs=5, loc_epochs=5, resume=False, force_rebuild_cache=True)
"""


def print_notebook_setup_template(**kwargs) -> str:
    t = get_notebook_setup_template(**kwargs)
    print(t)
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def audit_localization_labels(cfg: Optional[Config] = None):
    """Print statistics over all localization labels to spot data quality issues."""
    cfg  = cfg or config
    proc = cfg.PROCESSED_DIR / "localization"
    all_labels = []
    for split in ["train", "val", "test"]:
        for lf in (proc / split).glob("*_label.json"):
            d = json.loads(lf.read_text())
            all_labels.append((d["azimuth_deg"], d["distance_m"], d["height_m"]))
    if not all_labels:
        print("No labels found."); return
    az, dist, ht = zip(*all_labels)
    print(f"  Sessions : {len(all_labels)}")
    print(f"  Azimuth  : min={min(az):.1f}°  max={max(az):.1f}°  mean={np.mean(az):.1f}°  std={np.std(az):.1f}°")
    print(f"  Distance : min={min(dist):.2f}m  max={max(dist):.2f}m  mean={np.mean(dist):.2f}m")
    print(f"  Height   : min={min(ht):.2f}m  max={max(ht):.2f}m  mean={np.mean(ht):.2f}m")
    round_dist = sum(1 for d in dist if d == round(d, 0))
    if round_dist > len(all_labels) * 0.3:
        print(f"  ⚠️  {round_dist}/{len(all_labels)} sessions have integer distance — possible placeholders")


def diagnose_uavirbase(cfg: Optional[Config] = None, n_probe: int = 5):
    """Probe the remote UaVirBASE ZIP to inspect the label schema."""
    cfg = cfg or config
    url = cfg.UAVIRBASE_ZIP_URL
    if url is None:
        print("❌ UAVIRBASE_ZIP_URL is not set."); return
    _ensure_remotezip()
    from remotezip import RemoteZip
    print(f"🔍 Diagnosing UaVirBASE archive …\n   URL: {url}\n")
    with RemoteZip(url, initial_buffer_size=5 * 1024 * 1024) as rz:
        all_names = rz.namelist()
    print(f"   Total entries: {len(all_names)}")
    audio_entries = [n for n in all_names if n.endswith("/output.wav") or n.endswith("/audio.wav")]
    label_entries = [n for n in all_names if n.endswith("/label.json") or n.endswith("/annotation.json")]
    print(f"   Audio files : {len(audio_entries)}")
    print(f"   Label files : {len(label_entries)}")
    if not label_entries:
        print("   ⚠️  No label files found."); return
    print(f"\n   Probing first {min(n_probe, len(label_entries))} label files …")
    with RemoteZip(url, initial_buffer_size=5 * 1024 * 1024) as rz:
        for entry in label_entries[:n_probe]:
            raw    = rz.read(entry)
            parsed = parse_label_json(raw)
            status = (f"✅ az={parsed[0]:.1f}°  dist={parsed[1]:.2f}m  ht={parsed[2]:.2f}m"
                      if parsed else "❌ PARSE FAILED")
            print(f"\n   {entry}\n     Result : {status}")
            if not parsed:
                print(f"     Raw    : {raw[:300]}")
    print("\n   ✅ Diagnosis complete.")


def verify_tdoa_accuracy(cfg: Optional[Config] = None) -> bool:
    """Synthesise drone signals and verify TDOA measurement accuracy (< 0.1 ms)."""
    cfg  = cfg or config
    mics = cfg.MIC_POSITIONS
    c    = cfg.SPEED_OF_SOUND
    sr   = cfg.SR
    ap   = AudioProcessor(cfg)
    positions = [[1.0, 0.8], [2.0, 0.5], [-1.5, 2.0], [0.3, 0.2]]
    max_tau   = np.max([np.linalg.norm(mics[i] - mics[j]) for i in range(3) for j in range(i+1, 3)]) / c * 1.5
    ds        = max(1, sr // 16000); fs_ds = sr // ds; hi_ds = min(5000, fs_ds // 2 - 100)
    print("=" * 60); print("  TDOA ACCURACY VERIFICATION"); print("=" * 60)
    all_ok = True
    for pos in positions:
        src   = np.array(pos)
        dists = np.linalg.norm(mics - src[None, :], axis=1)
        exp12 = (dists[1] - dists[0]) / c
        chs   = synthesise_drone(mics, pos, fundamental=100, noise_level=0.005)
        y1    = bandpass(ap.pad_or_truncate(chs[0]), sr, 80, hi_ds)[::ds]
        y2    = bandpass(ap.pad_or_truncate(chs[1]), sr, 80, hi_ds)[::ds]
        meas, _, _ = gcc_phat(y2, y1, fs=fs_ds, max_tau=max_tau, interp=16)
        err = abs(meas - exp12) * 1000
        ok  = err < 0.1
        if not ok: all_ok = False
        print(f"  {str(pos):15s}  expected={exp12*1000:+12.4f}ms  measured={meas*1000:+13.4f}ms  err={err:7.4f}ms  {'✅' if ok else '❌'}")
    print()
    print("  ✅ All accurate." if all_ok else "  ❌ Some failed — check _fractional_delay()")
    print("=" * 60)
    return all_ok


def quick_demo(cfg: Optional[Config] = None) -> dict:
    """Synthesise a 2-drone scenario and run the full pipeline."""
    cfg = cfg or config
    print("\n🚁 QUICK DEMO — Multi-drone synthetic test"); print("─" * 50)
    positions = [[1.5, 0.4], [-1.0, 1.8]]
    funds     = [100, 280]
    ap        = AudioProcessor(cfg)
    n         = int(cfg.SR * cfg.TARGET_DURATION)
    mix       = [np.zeros(n, dtype=np.float32) for _ in range(3)]
    for pos, fund in zip(positions, funds):
        chs = synthesise_drone(cfg.MIC_POSITIONS, pos, fundamental=fund, noise_level=0.03)
        for i, ch in enumerate(chs):
            mix[i] = np.clip(mix[i] + ap.pad_or_truncate(ch), -1, 1).astype(np.float32)
    peak = max(float(np.max(np.abs(ch))) for ch in mix) + 1e-8
    mix  = [(ch / peak * 0.95).astype(np.float32) for ch in mix]
    tmp_paths = []
    for ch in mix:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tf.name, ch, cfg.SR); tmp_paths.append(tf.name)
    tracker = KalmanTracker(cfg)
    result  = run_pipeline(tmp_paths, cfg, tracker=tracker, multi_drone=True)
    for p in tmp_paths:
        os.unlink(p)
    print(f"  Detected: {result['detected']}  (conf={result['probability']:.3f})")
    print(f"  Drones found: {len(result['drones'])}")
    for i, d in enumerate(result["drones"]):
        xy   = d["xy_position"]
        true = positions[i] if i < len(positions) else None
        err  = f"  err={np.linalg.norm(xy - np.array(true)):.3f}m" if true else ""
        print(f"    Drone {i+1}: ({xy[0]:.2f},{xy[1]:.2f})m  az={d['azimuth_deg']:.1f}°  dist={d['distance_m']:.2f}m{err}")
    if result["drones"]:
        plot_multi_drone_positions(result["drones"], cfg)
    # ── Tracks ────────────────────────────────────────────────────────────
    tracks = tracker.all_confirmed()
    print(f"  Confirmed tracks: {len(tracks)}")
    for t in tracks:
        xy = t.predicted_xy()
        print(f"    Track #{t.track_id}: pos=({xy[0]:.2f},{xy[1]:.2f})m  "
              f"hits={t.hits}  dist={t.total_distance():.3f}m")
    if tracks:
        plot_track_trajectory(tracks, cfg)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Colab UI
# ══════════════════════════════════════════════════════════════════════════════

def upload_custom_dataset_artifacts(
    cfg:         Optional[Config] = None,
    extract_dir: Optional[str]    = None,
    accept_zip:  bool = True,
) -> str:
    """Upload files or ZIP in Colab and return the extraction directory path."""
    cfg = cfg or config
    if "google.colab" not in sys.modules:
        raise RuntimeError("upload_custom_dataset_artifacts() requires Google Colab.")
    from google.colab import files
    root = Path(extract_dir or (cfg.LOCAL_BASE / "uploaded_custom_dataset"))
    root.mkdir(parents=True, exist_ok=True)
    uploaded = files.upload()
    saved = []
    for name, content in uploaded.items():
        tmp_path = root / name
        with open(tmp_path, "wb") as f:
            f.write(content)
        saved.append(str(tmp_path))
        if accept_zip and tmp_path.suffix.lower() == ".zip":
            unzip_dir = root / safe_slug(tmp_path.stem)
            unzip_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(unzip_dir)
    print(f"✅ Uploaded {len(saved)} file(s) into {root}")
    return str(root)


def launch_ui(cfg: Optional[Config] = None):
    """Launch the full ipywidgets UI (Colab only)."""
    if "google.colab" not in sys.modules:
        print("ℹ️  launch_ui() requires Google Colab."); return

    import ipywidgets as w
    from IPython.display import display, clear_output, Audio, HTML
    from .visualization import (
        plot_polar_azimuth, plot_multi_drone_positions,
        plot_track_trajectory, plot_training_logs,
    )
    cfg = cfg or config

    def _card(ok, title, rows):
        col = "#10b981" if ok else "#ef4444"; ic = "✅" if ok else "❌"
        trs = "".join(f"<tr><td style='color:#94a3b8;padding:2px 8px'>{k}</td>"
                      f"<td style='color:#e2e8f0;padding:2px 8px'>{v}</td></tr>" for k, v in rows)
        return HTML(f"<div style='background:#0f172a;border-left:3px solid {col};border-radius:8px;"
                    f"padding:12px 16px;margin:8px 0;font-family:monospace;font-size:12px'>"
                    f"<b style='color:{col}'>{ic} {title}</b>"
                    f"<table style='margin-top:8px;border-collapse:collapse'>{trs}</table></div>")

    state = {}

    # ── Tab 1: Single File ─────────────────────────────────────────────────
    t1_up  = w.Button(description="📁 Upload Audio", button_style="primary", layout=w.Layout(width="190px"))
    t1_thr = w.FloatSlider(value=0.70, min=0.10, max=0.99, step=0.01, description="Threshold:",
                           layout=w.Layout(width="400px"), continuous_update=False)
    t1_run = w.Button(description="🚁 Run", button_style="success", layout=w.Layout(width="120px"), disabled=True)
    t1_lbl = w.Label("Step 1: upload a file"); t1_out_audio = w.Output(); t1_out = w.Output()

    def _t1_up(_):
        from google.colab import files
        up = files.upload()
        af = [p for p in up if p.lower().endswith(AUDIO_EXTS)]
        if not af: t1_lbl.value = "❌ No audio"; return
        p = af[0]; state["t1_path"] = p
        with t1_out_audio: clear_output(); display(Audio(p, autoplay=False))
        t1_run.disabled = False; t1_lbl.value = f"Ready: {p}"

    def _t1_run(_):
        if "t1_path" not in state: return
        cfg.DETECTION_THRESHOLD = t1_thr.value
        with t1_out:
            clear_output(wait=True)
            try:
                from .inference import analyse_audio_file
                result = analyse_audio_file(state["t1_path"], cfg, show_plot=True)
                display(_card(result["detected"], "Analysis Result", [
                    ("File", state["t1_path"]), ("Prob", f"{result['probability']:.3f}"),
                    ("Detected", str(result["detected"])), ("Duration", f"{result['duration_sec']:.2f}s"),
                ]))
            except Exception as e:
                import traceback; traceback.print_exc(); display(_card(False, f"Error: {e}", []))

    t1_up.on_click(_t1_up); t1_run.on_click(_t1_run)
    tab1 = w.VBox([w.HTML("<b style='color:#00d4ff;font-family:monospace'>Single-File Detection + Analysis</b>"),
                   w.HBox([t1_up, t1_run]), t1_thr, t1_lbl, t1_out_audio, t1_out])

    # ── Tab 2: 3-Mic Real ─────────────────────────────────────────────────
    t2_up  = w.Button(description="📁 Upload 3 Mic Files", button_style="primary", layout=w.Layout(width="210px"))
    t2_thr = w.FloatSlider(value=0.70, min=0.10, max=0.99, step=0.01, description="Threshold:",
                           layout=w.Layout(width="380px"), continuous_update=False)
    t2_run = w.Button(description="📡 Localize", button_style="success", layout=w.Layout(width="130px"), disabled=True)
    t2_lbl = w.Label("Upload 3 files (sorted = Mic1, Mic2, Mic3)"); t2_out = w.Output()

    def _t2_up(_):
        from google.colab import files
        up = files.upload()
        af = sorted([p for p in up if p.lower().endswith(AUDIO_EXTS)])[:3]
        if len(af) < 3: t2_lbl.value = f"❌ Need 3, got {len(af)}"; return
        state["t2_paths"] = af; t2_run.disabled = False
        t2_lbl.value = str([Path(p).name for p in af])

    def _t2_run(_):
        if "t2_paths" not in state: return
        cfg.DETECTION_THRESHOLD = t2_thr.value
        from .inference import load_3ch
        chs = load_3ch(state["t2_paths"], cfg)
        with t2_out:
            clear_output(wait=True)
            try:
                det = detect(chs, cfg)
                rows = [("Detected", str(det["detected"])), ("Prob", f"{det['probability']:.3f}"),
                        ("CNN",  f"{det['cnn_probability']:.3f}"), ("Heuristic", f"{det['heuristic_probability']:.3f}")]
                if det["detected"]:
                    loc = localize(chs, cfg)
                    rows += [("Azimuth", f"{loc['azimuth_deg']:.1f}°"),
                             ("Distance", f"{loc['distance_m']:.3f} m"),
                             ("XY", f"({loc['xy_position'][0]:.3f}, {loc['xy_position'][1]:.3f}) m")]
                    plot_polar_azimuth([loc["azimuth_deg"]], cfg=cfg)
                display(_card(det["detected"], "3-Mic Real Result", rows))
            except Exception as e:
                display(_card(False, f"Error: {e}", []))

    t2_up.on_click(_t2_up); t2_run.on_click(_t2_run)
    tab2 = w.VBox([w.HTML("<b style='color:#00d4ff;font-family:monospace'>3-Mic Real Recording</b>"),
                   w.HBox([t2_up, t2_run]), t2_thr, t2_lbl, t2_out])

    # ── Tab 3: Multi-Drone ────────────────────────────────────────────────
    t3_run = w.Button(description="🚁🚁 Run Synthetic Multi-Drone", button_style="success", layout=w.Layout(width="280px"))
    t3_max = w.IntSlider(value=2, min=1, max=3, step=1, description="Max drones:",
                         layout=w.Layout(width="300px"), continuous_update=False)
    t3_out = w.Output()

    def _t3_run(_):
        with t3_out:
            clear_output(wait=True)
            try:
                true_pos = [[1.5, 0.4], [-1.0, 1.8]]
                mix = [np.zeros(int(cfg.SR * cfg.TARGET_DURATION), dtype=np.float32) for _ in range(3)]
                ap  = AudioProcessor(cfg)
                for pos, fund in zip(true_pos, [100, 280]):
                    for i, ch in enumerate(synthesise_drone(cfg.MIC_POSITIONS, pos, fundamental=fund, noise_level=0.03)):
                        mix[i] = np.clip(mix[i] + ap.pad_or_truncate(ch), -1, 1).astype(np.float32)
                peak = max(float(np.max(np.abs(ch))) for ch in mix) + 1e-8
                chs  = [(ch / peak * 0.95).astype(np.float32) for ch in mix]
                drones = localize_multi_drone(chs, cfg, t3_max.value)
                rows = [("True positions", str(true_pos)), ("Drones found", str(len(drones)))]
                for i, d in enumerate(drones):
                    xy = d["xy_position"]
                    rows.append((f"Drone #{i+1}", f"({xy[0]:.3f},{xy[1]:.3f})m  az={d['azimuth_deg']:.1f}°  ±{d.get('confidence_radius',0):.3f}m"))
                if drones: plot_multi_drone_positions(drones, cfg)
                display(_card(len(drones) > 0, f"Multi-Drone: {len(drones)} found", rows))
            except Exception as e:
                display(_card(False, f"Error: {e}", []))

    t3_run.on_click(_t3_run)
    tab3 = w.VBox([w.HTML("<b style='color:#00d4ff;font-family:monospace'>Multi-Drone Detection & Localization</b>"),
                   t3_max, t3_run, t3_out])

    # ── Tab 4: Plots ──────────────────────────────────────────────────────
    t4_run = w.Button(description="📈 Training Curves", button_style="info", layout=w.Layout(width="200px"))
    t4_out = w.Output()

    def _t4_run(_):
        with t4_out: clear_output(wait=True); plot_training_logs(cfg)

    t4_run.on_click(_t4_run)
    tab4 = w.VBox([w.HTML("<b style='color:#00d4ff;font-family:monospace'>Training & Evaluation Plots</b>"),
                   t4_run, t4_out])

    tabs = w.Tab(children=[tab1, tab2, tab3, tab4])
    for i, ttl in enumerate(["🎵 Single File", "📡 3-Mic Real", "🚁🚁 Multi-Drone", "📈 Plots"]):
        tabs.set_title(i, ttl)

    header = w.HTML("""
<div style='background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:10px;
            padding:16px 20px;margin-bottom:12px;font-family:monospace'>
  <div style='font-size:18px;font-weight:bold;color:#00d4ff'>
    🚁 Drone Detection v15 — Localisation &amp; Visualisation
  </div>
  <div style='color:#64748b;font-size:11px;margin-top:4px'>
    Features: [log-mel, PCEN, delta-mel] · FocalLoss · v2 Cartesian TDOA solver ·
    Position-grouped evaluation · 6-panel dark dashboard
  </div>
</div>""")
    display(w.VBox([header, tabs]))
