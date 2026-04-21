#!/usr/bin/env python3
# build_training_datasets_and_figures.py
"""
Full pipeline mirror — detection + localization dataset build + thesis figures
==============================================================================
This script mirrors EXACTLY what happens when train_all() is called in the
notebook, but stops before the actual gradient updates so you can inspect
and report the dataset that the models will be trained on.

Execution order mirrors train_detection() → train_localization():

  ┌─ DETECTION DATASET ─────────────────────────────────────────────────────┐
  │  1a. Builtin DroneAudioDataset (GitHub ZIP download)                    │
  │  1b. External scraped audio           (opt-in, off by default)          │
  │  1c. Custom builder dataset           (Dunakeszi picker output)         │
  │  1d. Mixed drone+background augmentation                                │
  │  1e. Mel cache build                                                    │
  │  1f. Synthetic detection injection   (MelCacheManager.inject_synthetic) │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ LOCALIZATION DATASET ──────────────────────────────────────────────────┐
  │  2a. UaVirBASE download + label parse                                   │
  │  2b. SyntheticLocDatasetV2 generation (grid-biased + free positions)    │
  │  2c. Real + synthetic ConcatDataset   (same as trainer)                 │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ FIGURES ───────────────────────────────────────────────────────────────┐
  │  fig1  Detection class balance (drone vs non-drone, all sources)        │
  │  fig2  Detection source breakdown (builtin / custom / mixed / synth)    │
  │  fig3  Mel spectrogram grid (real clips: MEMS, Brüel, builtin)         │
  │  fig4  BPF energy ratio distribution (real clips by source type)        │
  │  fig5  Mixed augmentation SNR distribution                              │
  │  fig6  Detection mel cache class balance                                │
  │  fig7  Localization dataset spatial coverage (real + synthetic)         │
  │  fig8  Localization distance & height histograms                        │
  │  fig9  Localization drone-type BPF energy ratio (synthetic)             │
  │  fig10 Noise profile PSD: MEMS vs Brüel vs synthetic models             │
  └─────────────────────────────────────────────────────────────────────────┘

Usage
-----
python build_training_datasets_and_figures.py \\
    --picker-output   output_v3/manual_review \\
    --base-dir        /tmp/drone_v15 \\
    --out-figures     ./thesis_figures \\
    --gpx             dunakeszi_audit_output/gpx_combined.csv \\
    --bpf-hz          82 \\
    --split-fracs     0.70 0.15 0.15 \\
    --no-train

python build_training_datasets_and_figures.py --picker-output output_v3/manual_review --base-dir C:/tmp/drone_v15 --out-figures ./thesis_figures --bpf-hz 82 --no-train

Pass --no-train to only build datasets + figures without running gradient updates.
Omit it to do the full train_all() run after figures are generated.

The script is self-contained: it imports drone_detection from the same directory
(or any path on sys.path) and calls the same functions the notebook uses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal as sp_signal
import soundfile as sf

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

DPI = 300
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff", ".m4a")

PALETTE = {
    "drone":     "#3B6BBF",
    "non_drone": "#E07B39",
    "builtin":   "#2E7D32",
    "custom":    "#6A1B9A",
    "mixed":     "#E65100",
    "synth":     "#0277BD",
    "MEMS":      "#00ACC1",
    "BRUEL":     "#F4511E",
    "real":      "#1976D2",
    "synthetic": "#F57C00",
    "train":     "#2E7D32",
    "val":       "#1565C0",
    "test":      "#6A1B9A",
    "indoor":    "#5E35B1",
    "outdoor":   "#00897B",
}


# =============================================================================
#  HELPERS
# =============================================================================

def _save(fig: plt.Figure, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅  {path.name}")


def _list_wavs(folder: Path, recursive: bool = True) -> List[Path]:
    if not folder or not folder.exists():
        return []
    fn = folder.rglob if recursive else folder.glob
    return sorted(p for p in fn("*") if p.suffix.lower() in AUDIO_EXTS)


def _load_mono(path: Path, sr: int, max_sec: float = 5.0) -> np.ndarray:
    if _LIBROSA_OK:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_sec)
        return y.astype(np.float32)
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = data[:int(max_sec * file_sr), 0]
    if file_sr != sr:
        from math import gcd
        g = gcd(file_sr, sr)
        y = sp_signal.resample_poly(y, sr // g, file_sr // g).astype(np.float32)
    return y


def _make_mel(y, sr, n_fft=1024, hop=256, n_mels=64):
    if _LIBROSA_OK:
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                            hop_length=hop, n_mels=n_mels)
        return librosa.power_to_db(M, ref=np.max)
    f, t, Sxx = sp_signal.spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return 10 * np.log10(Sxx[:n_mels] + 1e-10)


def _bpf_ratio(y, sr, bpf_hz=82.0, bw_hz=20.0, n_harm=4):
    y = y.astype(np.float64)
    nyq = sr / 2.0
    total = float(np.mean(y**2)) + 1e-10
    power = 0.0
    for k in range(1, n_harm + 1):
        fc = bpf_hz * k
        if fc + bw_hz >= nyq: break
        lo, hi = max(fc - bw_hz, 1.0), min(fc + bw_hz, nyq - 1.0)
        sos = sp_signal.butter(4, [lo/nyq, hi/nyq], btype="band", output="sos")
        power += float(np.mean(sp_signal.sosfilt(sos, y)**2))
    return float(np.clip(power / total, 0, 1))


def _avg_psd(wavs, sr, n=10):
    psds, f_ref = [], None
    for w in wavs[:n]:
        try:
            y = _load_mono(w, sr, 3.0)
            f, psd = sp_signal.welch(y, fs=sr, nperseg=min(4096, len(y)//4 or 512))
            if f_ref is None: f_ref = f
            psds.append(psd)
        except Exception:
            pass
    if not psds:
        return np.array([]), np.array([])
    return f_ref, 10 * np.log10(np.mean(psds, axis=0) + 1e-20)


def _infer_source(path: Path) -> str:
    """Infer which source a WAV in processed/detection came from."""
    s = path.stem.lower()
    if s.startswith("mixdrone"):      return "mixed"
    if s.startswith("synth_det"):     return "synth"
    if "mems" in s or "bruel" in s:   return "custom"
    if s.startswith("custom"):        return "custom"
    return "builtin"


# =============================================================================
#  STEP 0 — Patch Config and build paths
# =============================================================================

def _make_cfg(base_dir: Path, colab: bool, picker_output: Path,
               split_fracs: Tuple[float, float, float]):
    """
    Import drone_detection, patch Config paths to base_dir,
    configure the custom builder dataset.
    Returns (cfg, builder_root).

    Works whether the script is placed:
      • inside  the package  (deployment/v2/drone_detection/build_…py)  ← typical
      • outside the package  (deployment/v2/build_…py)
    """
    script_dir = Path(__file__).resolve().parent

    # If __init__.py is in the same folder, this script lives INSIDE the package.
    # Python needs the PARENT on sys.path so `import drone_detection` resolves.
    if (script_dir / "__init__.py").exists():
        pkg_parent = script_dir.parent   # e.g. deployment/v2/
        pkg_name   = script_dir.name     # e.g. "drone_detection"
    else:
        # Script lives outside the package — add its own directory
        pkg_parent = script_dir
        pkg_name   = "drone_detection"   # package must be a subfolder

    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))

    # Validate we can find it before proceeding
    import importlib
    spec = importlib.util.find_spec(pkg_name)
    if spec is None:
        raise ImportError(
            f"Cannot find package '{pkg_name}' on sys.path.\n"
            f"  sys.path[0] = {pkg_parent}\n"
            f"  Script is at: {Path(__file__).resolve()}\n"
            "  Place the script either inside or directly alongside the "
            "'drone_detection' package folder."
        )

    from .config import Config
    cfg = Config()

    # Override all path attributes
    B = base_dir
    cfg.LOCAL_BASE    = B
    cfg.RAW_DIR       = B / "raw"
    cfg.PROCESSED_DIR = B / "processed"
    cfg.MEL_CACHE_DIR = B / "mel_cache"
    cfg.DRIVE_ROOT    = B
    cfg.DRIVE_MODELS  = B / "models"
    cfg.DRIVE_LOGS    = B / "logs"
    cfg.DRIVE_TRACKS  = B / "tracks"
    cfg.DRIVE_PLOTS   = B / "logs" / "plots"
    cfg.UAVIRBASE_RAW = B / "uavirbase"
    cfg.DRONEDS_RAW   = B / "droneds"

    builder_root = B / "raw" / "dunakeszi_builder"
    cfg.CUSTOM_DATASET_ROOT    = str(builder_root)
    cfg.CUSTOM_DATASET_ENABLED = True
    cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE = False
    cfg.CUSTOM_DATASET_IMPORTED_ROOT = str(cfg.RAW_DIR / "custom_builder_import")
    cfg.CUSTOM_DATASET_MANUAL_CLEAN_SUBDIR = "manual_clean/clean_drone_sections"
    cfg.IN_COLAB = colab

    cfg.ensure_dirs()
    return cfg, builder_root


# =============================================================================
#  STEP 1 — Build builder-root layout from picker output
# =============================================================================

def _build_builder_root(picker_output: Path, builder_root: Path,
                         train_frac=0.70, val_frac=0.15, seed=42):
    # No drone_detection import needed here — uses only stdlib + numpy + shutil
    clips_dir  = picker_output / "clean_drone_sections"
    mems_wavs  = _list_wavs(clips_dir / "MEMS")
    bruel_wavs = _list_wavs(clips_dir / "BRUEL")
    all_wavs   = mems_wavs + bruel_wavs

    if not all_wavs:
        print(f"  ⚠  No WAV clips found under {clips_dir} — "
              "continuing without custom dataset")
        return {}

    print(f"\n  Found {len(mems_wavs)} MEMS + {len(bruel_wavs)} Brüel clips")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_wavs))
    n_tr  = max(1, int(len(all_wavs) * train_frac))
    n_val = max(1, int(len(all_wavs) * val_frac))
    split_map = {}
    for i in idx[:n_tr]:               split_map[all_wavs[i]] = "train"
    for i in idx[n_tr:n_tr + n_val]:   split_map[all_wavs[i]] = "val"
    for i in idx[n_tr + n_val:]:       split_map[all_wavs[i]] = "test"

    dirs = {
        "clean":  builder_root / "clean_drone_sections",
        "manual": builder_root / "manual_clean" / "clean_drone_sections",
        "train":  builder_root / "train" / "drone",
        "val":    builder_root / "val"   / "drone",
        "test":   builder_root / "test"  / "drone",
        "bgpool": builder_root / "background_pool",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (dirs["clean"] / "MEMS").mkdir(exist_ok=True)
    (dirs["clean"] / "BRUEL").mkdir(exist_ok=True)

    counts = defaultdict(int)
    for wav, split in split_map.items():
        src_type = "BRUEL" if wav in set(bruel_wavs) else "MEMS"

        # clean_drone_sections/{MEMS,BRUEL}/
        dst = dirs["clean"] / src_type / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst); counts["clean"] += 1

        # manual_clean fallback (flat)
        dst_m = dirs["manual"] / f"{src_type}_{wav.name}"
        if not dst_m.exists():
            shutil.copy2(wav, dst_m); counts["manual"] += 1

        # train / val / test
        dst_s = dirs[split] / f"custom_{src_type}_{wav.name}"
        if not dst_s.exists():
            shutil.copy2(wav, dst_s); counts[split] += 1

        # sidecar JSON
        meta_src = wav.parent / (wav.stem + "_meta.json")
        if meta_src.exists():
            dst_j = dirs[split] / f"custom_{src_type}_{wav.stem}_meta.json"
            if not dst_j.exists():
                shutil.copy2(meta_src, dst_j)

    print(f"  Builder root: {builder_root}")
    print(f"  train={counts['train']}  val={counts['val']}  "
          f"test={counts['test']}  clean={counts['clean']}")

    (builder_root / "build_manifest.json").write_text(json.dumps({
        "n_mems": len(mems_wavs), "n_bruel": len(bruel_wavs),
        "n_total": len(all_wavs),
        "splits": {str(w): s for w, s in split_map.items()},
    }, indent=2))
    return dict(counts)


# =============================================================================
#  STEP 2 — Run the exact detection pipeline sequence
# =============================================================================

def run_detection_pipeline(cfg, builder_root: Path,
                             download_builtin: bool,
                             download_external: bool) -> Dict:
    """
    Mirrors train_detection() but stops before DetectionTrainer.run().
    Returns a summary of all counts at each stage.
    """
    from .datasets import (
        DroneAudioDatasetManager, MelCacheManager, report_detection_split_counts
    )
    from .orchestration import (
        import_custom_builder_dataset, generate_mixed_drone_training_audio
    )
    from .utils import _set_seed
    _set_seed(cfg.SEED)

    print("\n" + "="*68)
    print("  STAGE 1 — Detection Dataset")
    print("="*68)

    # 1a. Builtin DroneAudioDataset
    if download_builtin:
        print("\n📥 [1/3] DroneAudioDataset (GitHub) …")
        try:
            DroneAudioDatasetManager(cfg).prepare()
        except Exception as e:
            print(f"  ⚠  DroneAudioDataset prepare failed: {e}")
    else:
        print("ℹ️  [1/3] Skipping built-in dataset download.")
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                (cfg.PROCESSED_DIR / "detection" / split / label).mkdir(
                    parents=True, exist_ok=True)

    # 1b. External scraping
    if download_external:
        print("\n🌐 [2/3] External audio scraping …")
        try:
            from .dataset_builder import (
                AudioWebScraper, _incorporate_scraped_audio
            )
            AudioWebScraper(cfg).download(force=False)
            _incorporate_scraped_audio(cfg, force=False)
        except Exception as e:
            print(f"  ⚠  Scraping failed: {e}")
    else:
        print("ℹ️  [2/3] Skipping external scraping.")

    # 1c. Custom builder
    print("\n📦 [3/3] Custom builder dataset (Dunakeszi) …")
    if builder_root.exists() and _list_wavs(builder_root / "train" / "drone"):
        try:
            import_custom_builder_dataset(
                cfg, builder_root=builder_root, force=False,
                include_background_pool_as_non_drone=False,
            )
        except Exception as e:
            print(f"  ⚠  Custom import failed: {e}")
    else:
        print("  ⚠  No custom builder clips found — skipping.")

    # 1d. Count before augmentation
    det = cfg.PROCESSED_DIR / "detection"
    stage_counts: Dict[str, Dict] = {}
    for split in ["train", "val", "test"]:
        stage_counts[split] = {}
        for label in ["drone", "non_drone"]:
            wavs = _list_wavs(det / split / label, recursive=False)
            by_src = defaultdict(int)
            for w in wavs:
                by_src[_infer_source(w)] += 1
            stage_counts[split][label] = dict(by_src)
            stage_counts[split][label]["total"] = len(wavs)

    _print_det_audit(stage_counts, "Pre-augmentation counts")

    total_drone     = sum(stage_counts[s]["drone"]["total"]     for s in stage_counts)
    total_non_drone = sum(stage_counts[s]["non_drone"]["total"] for s in stage_counts)

    if total_drone == 0:
        print("  ❌  No drone files — cannot continue."); return stage_counts
    if total_non_drone == 0:
        print("  ❌  No non-drone files — cannot continue."); return stage_counts

    # 1e. Mixed augmentation
    print("\n🎛️  [4/4] Mixed drone+background augmentation …")
    try:
        generate_mixed_drone_training_audio(cfg, force=False)
    except Exception as e:
        print(f"  ⚠  Mixed audio generation failed: {e}")

    # 1f. Mel cache + synthetic injection
    print("\n🎵  Building mel cache …")
    mcm = MelCacheManager(cfg)
    try:
        mcm.build(force=False)
        mcm.inject_synthetic(force=False)
    except Exception as e:
        print(f"  ⚠  Mel cache build failed: {e}")

    # Post-augmentation counts
    post_counts: Dict[str, Dict] = {}
    for split in ["train", "val", "test"]:
        post_counts[split] = {}
        for label in ["drone", "non_drone"]:
            wavs = _list_wavs(det / split / label, recursive=False)
            by_src = defaultdict(int)
            for w in wavs:
                by_src[_infer_source(w)] += 1
            post_counts[split][label] = dict(by_src)
            post_counts[split][label]["total"] = len(wavs)

    mel_counts = {}
    try:
        mel_counts = mcm.count()
    except Exception:
        pass

    _print_det_audit(post_counts, "Post-augmentation counts (final detection dataset)")
    if mel_counts:
        print("\n📊 MEL CACHE")
        for k, v in mel_counts.items():
            print(f"   {k:28s}: {v}")

    return {"pre": stage_counts, "post": post_counts, "mel": mel_counts}


def _print_det_audit(counts, title):
    print(f"\n  ── {title} ──")
    for split in ["train", "val", "test"]:
        for label in ["drone", "non_drone"]:
            d = counts.get(split, {}).get(label, {})
            total = d.get("total", 0)
            detail = "  ".join(f"{k}={v}" for k, v in d.items() if k != "total")
            print(f"   {split:5s}/{label:10s}: total={total:4d}  {detail}")


# =============================================================================
#  STEP 3 — Run the exact localization pipeline sequence
# =============================================================================

def run_localization_pipeline(cfg) -> Dict:
    """
    Mirrors train_localization() — downloads UaVirBASE + builds
    SyntheticLocDatasetV2. Stops before optimizer step.
    Returns dataset size summary.
    """
    from .datasets import (
        UaVirBASEDatasetManager, LocalizationDataset,
        SyntheticLocDatasetV2
    )
    from .utils import _set_seed
    _set_seed(cfg.SEED)

    print("\n" + "="*68)
    print("  STAGE 2 — Localization Dataset")
    print("="*68)

    # 2a. UaVirBASE
    print("\n📥 UaVirBASE download + label parse …")
    um = UaVirBASEDatasetManager(cfg)
    try:
        um.prepare()
    except Exception as e:
        print(f"  ⚠  UaVirBASE prepare failed: {e}")

    # 2b. SyntheticLocDatasetV2 — same params as train_localization
    n_synth_train = int(cfg.SYNTHETIC_SAMPLES)
    n_synth_val   = max(200, n_synth_train // 5)
    print(f"\n🔬 Building SyntheticLocDatasetV2  "
          f"train={n_synth_train}  val={n_synth_val}  …")
    synth_train = SyntheticLocDatasetV2(cfg, n_samples=n_synth_train,
                                         grid_fraction=0.55, augment=True)
    synth_val   = SyntheticLocDatasetV2(cfg, n_samples=n_synth_val,
                                         grid_fraction=0.55, augment=False)

    # 2c. Try real data
    proc = cfg.PROCESSED_DIR / "localization"
    real_sizes = {"train": 0, "val": 0, "test": 0}
    try:
        real_train = LocalizationDataset(proc, "train", augment=True,  cfg=cfg)
        real_val   = LocalizationDataset(proc, "val",   augment=False, cfg=cfg)
        real_test  = LocalizationDataset(proc, "test",  augment=False, cfg=cfg)
        real_sizes = {"train": len(real_train), "val": len(real_val),
                      "test": len(real_test)}
        print(f"  Real UaVirBASE: train={len(real_train)}  "
              f"val={len(real_val)}  test={len(real_test)}")
    except RuntimeError:
        print("  ⚠  No real localization data — synthetic only")

    total_train = real_sizes["train"] + n_synth_train
    total_val   = real_sizes["val"]   + n_synth_val

    print(f"\n  Total localization: train={total_train}  val={total_val}")

    return {
        "synth_train": n_synth_train,
        "synth_val":   n_synth_val,
        "real":        real_sizes,
        "total_train": total_train,
        "total_val":   total_val,
        "synth_dataset_train": synth_train,
        "synth_dataset_val":   synth_val,
    }


# =============================================================================
#  FIGURES
# =============================================================================

# ── Fig 1: Detection class balance ───────────────────────────────────────────

def fig1_detection_class_balance(post_counts: Dict, out_dir: Path):
    splits = ["train", "val", "test"]
    labels_nice = ["Train", "Validation", "Test"]
    drone_n    = [post_counts[s].get("drone",     {}).get("total", 0) for s in splits]
    nd_n       = [post_counts[s].get("non_drone", {}).get("total", 0) for s in splits]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(splits)); w = 0.35
    b1 = ax.bar(x - w/2, drone_n, w, label="Drone",     color=PALETTE["drone"],     alpha=0.85)
    b2 = ax.bar(x + w/2, nd_n,    w, label="Non-drone", color=PALETTE["non_drone"], alpha=0.85)
    for bars, vals in ((b1, drone_n), (b2, nd_n)):
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels_nice)
    ax.set_ylabel("Sample count")
    ax.set_title("Detection dataset — drone vs non-drone per split\n"
                 "(all sources: builtin + custom + mixed + synthetic)")
    ax.legend()
    total_d  = sum(drone_n)
    total_nd = sum(nd_n)
    ratio = total_d / (total_d + total_nd + 1e-6)
    ax.text(0.98, 0.97,
            f"Total drone:     {total_d}\n"
            f"Total non-drone: {total_nd}\n"
            f"Drone fraction:  {ratio:.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    _save(fig, out_dir / "fig1_detection_class_balance.png")


# ── Fig 2: Source breakdown ───────────────────────────────────────────────────

def fig2_detection_source_breakdown(pre_counts: Dict, post_counts: Dict,
                                     out_dir: Path):
    sources = ["builtin", "custom", "mixed", "synth"]
    src_colours = [PALETTE[s] for s in sources]
    splits = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Detection dataset — source breakdown (drone class)",
                 fontsize=13, fontweight="bold")

    for ax, counts, title in zip(axes,
            [pre_counts, post_counts],
            ["Pre-augmentation", "Post-augmentation (final)"]):
        x = np.arange(len(splits)); width = 0.55
        bottoms = np.zeros(len(splits))
        for src, col in zip(sources, src_colours):
            vals = [counts.get(s, {}).get("drone", {}).get(src, 0) for s in splits]
            bars = ax.bar(x, vals, width, bottom=bottoms, label=src,
                          color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
            # Label non-zero segments
            for bar, v, bot in zip(bars, vals, bottoms):
                if v > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bot + v/2, str(v),
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            bottoms += np.array(vals, dtype=float)
        # Total labels on top
        for xi, tot in zip(x, bottoms):
            ax.text(xi, tot + 2, str(int(tot)), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(["Train", "Val", "Test"])
        ax.set_ylabel("Clip count")
        ax.set_title(title)
        ax.legend(fontsize=9)
    _save(fig, out_dir / "fig2_detection_source_breakdown.png")


# ── Fig 3: Mel spectrogram grid ───────────────────────────────────────────────

def fig3_mel_grid(cfg, builder_root: Path, out_dir: Path):
    sr = cfg.SR

    def _pick(folder, label, n=2):
        wavs = _list_wavs(folder)
        random.seed(42)
        return random.sample(wavs, min(n, len(wavs)))

    det = cfg.PROCESSED_DIR / "detection"

    clips = []
    # MEMS custom
    for w in _pick(builder_root / "clean_drone_sections" / "MEMS", "MEMS", 2):
        clips.append((w, "MEMS (Dunakeszi)"))
    # Brüel custom
    for w in _pick(builder_root / "clean_drone_sections" / "BRUEL", "Brüel (Dunakeszi)", 2):
        clips.append((w, "Brüel (Dunakeszi)"))
    # Builtin drone
    builtin_wavs = [w for w in _list_wavs(det / "train" / "drone")
                    if _infer_source(w) == "builtin"]
    for w in builtin_wavs[:1]:
        clips.append((w, "DroneAudioDataset"))
    # Mixed
    mixed_wavs = [w for w in _list_wavs(det / "train" / "drone")
                  if _infer_source(w) == "mixed"]
    for w in mixed_wavs[:1]:
        clips.append((w, "Mixed (drone+BG)"))

    n_clips = len(clips)
    if n_clips == 0:
        print("  ⚠  No clips for mel grid — skipping fig3"); return

    cols = min(n_clips, 3)
    rows = math.ceil(n_clips / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("Mel spectrograms — all detection dataset sources",
                 fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    for idx, (wav_path, label) in enumerate(clips):
        ax = axes_flat[idx]
        try:
            y   = _load_mono(wav_path, sr)
            mel = _make_mel(y, sr)
            im  = ax.imshow(mel, origin="lower", aspect="auto",
                            extent=[0, len(y)/sr, 0, sr/2/1000],
                            cmap="inferno", vmin=-80, vmax=0)
            plt.colorbar(im, ax=ax, format="%+.0f dB", pad=0.02, shrink=0.9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Freq (kHz)", fontsize=8)

        # Overlay sidecar info if present
        meta_p = wav_path.parent / (wav_path.stem + "_meta.json")
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text())
                phase = m.get("clip", {}).get("flight_phase", "")
                snr   = m.get("signal_metrics", {}).get("snr_db", "")
                ann   = f"phase={phase}" if phase else ""
                if snr: ann += f"  SNR={snr:.1f}dB"
                if ann:
                    ax.text(0.02, 0.97, ann, transform=ax.transAxes,
                            fontsize=6.5, color="white", va="top",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="black", alpha=0.5))
            except Exception:
                pass

    for idx in range(n_clips, len(axes_flat)):
        axes_flat[idx].axis("off")

    _save(fig, out_dir / "fig3_mel_grid.png")


# ── Fig 4: BPF energy ratio distribution ─────────────────────────────────────

def fig4_bpf_ratio(cfg, builder_root: Path, out_dir: Path, bpf_hz: float = 82.0):
    sr = cfg.SR
    det = cfg.PROCESSED_DIR / "detection"

    # Sources to compare
    source_wavs = {
        "MEMS (Dunakeszi)":   _list_wavs(builder_root / "clean_drone_sections" / "MEMS")[:60],
        "Brüel (Dunakeszi)":  _list_wavs(builder_root / "clean_drone_sections" / "BRUEL")[:60],
        "DroneAudioDataset":  [w for w in _list_wavs(det / "train" / "drone")[:60]
                                if _infer_source(w) == "builtin"],
        "Mixed":              [w for w in _list_wavs(det / "train" / "drone")[:30]
                                if _infer_source(w) == "mixed"],
    }
    source_wavs = {k: v for k, v in source_wavs.items() if v}

    if not source_wavs:
        print("  ⚠  No clips for BPF ratio — skipping fig4"); return

    print(f"  Computing BPF ratios ({bpf_hz} Hz) …")
    ratios_by_src = {}
    for src, wavs in source_wavs.items():
        ratios = []
        for w in wavs:
            try:
                y = _load_mono(w, sr, 5.0)
                ratios.append(_bpf_ratio(y, sr, bpf_hz=bpf_hz))
            except Exception:
                pass
        ratios_by_src[src] = np.array(ratios)

    cols = [PALETTE["MEMS"], PALETTE["BRUEL"], PALETTE["builtin"], PALETTE["mixed"]]
    bins = np.linspace(0, 1, 31)
    fig, axes = plt.subplots(1, len(ratios_by_src), figsize=(5 * len(ratios_by_src), 5))
    fig.suptitle(f"BPF energy ratio — real clips by source  (BPF={bpf_hz} Hz, 4 harmonics)",
                 fontsize=13, fontweight="bold")
    if len(ratios_by_src) == 1:
        axes = [axes]

    for ax, (src, vals), col in zip(axes, ratios_by_src.items(), cols):
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No clips", ha="center", va="center",
                    transform=ax.transAxes); continue
        ax.hist(vals, bins=bins, color=col, alpha=0.80, edgecolor="white", lw=0.4)
        ax.axvline(np.median(vals), color="red",    lw=1.8, ls="--",
                   label=f"Median={np.median(vals):.3f}")
        ax.axvline(np.mean(vals),   color="orange", lw=1.4, ls=":",
                   label=f"Mean={np.mean(vals):.3f}")
        ax.set_title(f"{src}\n(n={len(vals)})", fontsize=9)
        ax.set_xlabel("BPF energy ratio")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    _save(fig, out_dir / "fig4_bpf_ratio_distribution.png")


# ── Fig 5: Mixed augmentation SNR distribution ────────────────────────────────

def fig5_mix_snr_distribution(cfg, out_dir: Path):
    """
    Show the distribution of SNRs used in the mixed augmentation,
    and visualise one example clean vs mixed clip side by side.
    """
    det = cfg.PROCESSED_DIR / "detection"
    mixed_wavs = [w for w in _list_wavs(det / "train" / "drone")
                  if _infer_source(w) == "mixed"]

    if not mixed_wavs:
        print("  ⚠  No mixed clips found — skipping fig5"); return

    sr = cfg.SR
    snr_range   = cfg.MIX_SNR_DB_RANGE
    gain_range  = cfg.MIX_GAIN_RANGE_DB

    # Sample theoretical distribution
    rng = np.random.default_rng(42)
    n_sim = 2000
    sim_snrs  = rng.uniform(*snr_range,  n_sim)
    sim_gains = rng.uniform(*gain_range, n_sim)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mixed drone+background augmentation parameters",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(sim_snrs,  bins=30, color=PALETTE["mixed"], alpha=0.80)
    ax.axvline(0, color="black", lw=1.2, ls="--", label="0 dB (equal power)")
    ax.set_xlabel("SNR (dB)  [drone relative to background]")
    ax.set_ylabel("Simulated count")
    ax.set_title(f"SNR range: [{snr_range[0]}, {snr_range[1]}] dB")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(sim_gains, bins=30, color=PALETTE["drone"], alpha=0.80)
    ax.set_xlabel("Drone gain (dB)")
    ax.set_ylabel("Simulated count")
    ax.set_title(f"Drone gain range: [{gain_range[0]}, {gain_range[1]}] dB")

    # Mel of one real mixed clip vs a clean clip
    ax = axes[2]
    try:
        random.seed(1)
        w = random.choice(mixed_wavs)
        y = _load_mono(w, sr, 5.0)
        mel = _make_mel(y, sr)
        im = ax.imshow(mel, origin="lower", aspect="auto",
                       extent=[0, len(y)/sr, 0, sr/2/1000],
                       cmap="inferno", vmin=-80, vmax=0)
        plt.colorbar(im, ax=ax, format="%+.0f dB", pad=0.02)
        ax.set_title(f"Sample mixed clip\n({w.name[:30]})", fontsize=9)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (kHz)")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                transform=ax.transAxes)

    _save(fig, out_dir / "fig5_mix_snr_distribution.png")


# ── Fig 6: Mel cache class balance ────────────────────────────────────────────

def fig6_mel_cache_balance(mel_counts: Dict, out_dir: Path):
    if not mel_counts:
        print("  ⚠  No mel cache counts — skipping fig6"); return

    splits = ["train", "val", "test"]
    drone_n = [mel_counts.get(f"{s}/drone",     0) for s in splits]
    nd_n    = [mel_counts.get(f"{s}/non_drone", 0) for s in splits]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(splits)); w = 0.35
    b1 = ax.bar(x - w/2, drone_n, w, label="Drone",     color=PALETTE["drone"],     alpha=0.85)
    b2 = ax.bar(x + w/2, nd_n,    w, label="Non-drone", color=PALETTE["non_drone"], alpha=0.85)
    for bars, vals in ((b1, drone_n), (b2, nd_n)):
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_ylabel("Cached feature tensors")
    ax.set_title("Mel cache class balance\n"
                 "(real WAVs + synthetic injections — this is the actual DetectionCNN input)")
    ax.legend()
    _save(fig, out_dir / "fig6_mel_cache_balance.png")


# ── Fig 7: Localization spatial coverage ─────────────────────────────────────

def fig7_loc_spatial_coverage(cfg, loc_result: Dict, out_dir: Path):
    synth_ds = loc_result.get("synth_dataset_train")
    real_sz  = loc_result.get("real", {})

    if synth_ds is None:
        print("  ⚠  No localization dataset — skipping fig7"); return

    cx, cy = cfg.ARRAY_CENTER
    pos = synth_ds.positions  # (N, 3): x, y, ht
    azs   = np.degrees(np.arctan2(pos[:, 1] - cy, pos[:, 0] - cx))
    dists = np.sqrt((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2)
    hts   = pos[:, 2]

    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
    fig.suptitle("Localization dataset spatial coverage\n"
                 f"(SyntheticLocDatasetV2 n={len(pos):,} + "
                 f"UaVirBASE real={real_sz.get('train', 0)})",
                 fontsize=12, fontweight="bold")

    # Polar
    ax0 = fig.add_subplot(gs[0], polar=True)
    sc0 = ax0.scatter(np.radians(azs), dists,
                      c=dists, cmap="plasma", alpha=0.35, s=6, linewidths=0)
    ax0.set_theta_zero_location("N"); ax0.set_theta_direction(-1)
    ax0.set_title("Azimuth / distance\n(polar)", pad=14, fontsize=10)
    plt.colorbar(sc0, ax=ax0, pad=0.12, shrink=0.65, label="Dist (m)")

    # Cartesian
    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(pos[:, 0] - cx, pos[:, 1] - cy,
                c=hts, cmap="viridis", alpha=0.25, s=4, linewidths=0)
    mics = cfg.MIC_POSITIONS
    for mi, (mx, my) in enumerate(mics):
        ax1.scatter(mx - cx, my - cy, s=120, color=["#E53935","#43A047","#1976D2"][mi],
                    zorder=5, edgecolors="white", lw=0.8, label=f"M{mi}")
    ax1.set_aspect("equal"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title("Cartesian scatter\n(colour = height)", fontsize=10)
    ax1.legend(fontsize=7)

    # Distance histogram
    ax2 = fig.add_subplot(gs[2])
    ax2.hist(dists, bins=40, color=PALETTE["synthetic"], alpha=0.80,
             label=f"Synthetic (n={len(dists):,})")
    # Overlay UaVirBASE real distances if known
    real_dist_m = [10.0, 20.0]
    for d in real_dist_m:
        ax2.axvline(d, color=PALETTE["real"], lw=1.5, ls="--",
                    label=f"UaVirBASE grid {d:.0f}m")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distance distribution", fontsize=10)
    ax2.legend(fontsize=8)

    _save(fig, out_dir / "fig7_loc_spatial_coverage.png")


# ── Fig 8: Localization distance & height histograms ─────────────────────────

def fig8_loc_dist_height(cfg, loc_result: Dict, out_dir: Path):
    synth_ds = loc_result.get("synth_dataset_train")
    if synth_ds is None:
        print("  ⚠  No localization dataset — skipping fig8"); return

    cx, cy = cfg.ARRAY_CENTER
    pos  = synth_ds.positions
    dists = np.sqrt((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2)
    hts   = pos[:, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Localization dataset — distance & height distributions",
                 fontsize=12, fontweight="bold")

    ax1.hist(dists, bins=40, color=PALETTE["synthetic"], alpha=0.85)
    ax1.set_xlabel("Drone distance from array (m)")
    ax1.set_ylabel("Sample count")
    ax1.set_title(f"Distance  (min={dists.min():.1f}m  max={dists.max():.1f}m  "
                  f"mean={dists.mean():.1f}m)")

    ax2.hist(hts, bins=30, color=PALETTE["outdoor"], alpha=0.85)
    ax2.set_xlabel("Drone height (m)")
    ax2.set_ylabel("Sample count")
    ax2.set_title(f"Height  (min={hts.min():.1f}m  max={hts.max():.1f}m  "
                  f"mean={hts.mean():.1f}m)")

    _save(fig, out_dir / "fig8_loc_dist_height.png")


# ── Fig 9: Synthetic BPF energy ratio by drone type ──────────────────────────

def fig9_synth_bpf_by_type(cfg, loc_result: Dict, out_dir: Path):
    from .config import DRONE_BPF_PROFILES, DRONE_BPF_ENERGY_RATIOS
    from .audio_processing import AudioProcessor

    synth_ds = loc_result.get("synth_dataset_train")
    if synth_ds is None:
        print("  ⚠  No localization dataset — skipping fig9"); return

    ap      = AudioProcessor(cfg)
    n_per   = 30
    types   = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]
    colours = ["#1976D2", "#E53935", "#43A047", "#8E24AA"]

    print("  Sampling synthetic localization clips for BPF ratios …")
    bpf_data = {}
    idx_by_type = {t: [] for t in types}
    for i, dt in enumerate(synth_ds.drone_types):
        if str(dt) in idx_by_type:
            idx_by_type[str(dt)].append(i)

    for drone_type, colour in zip(types, colours):
        idxs = idx_by_type[drone_type][:n_per]
        if not idxs:
            continue
        f_lo, f_mid, f_hi, _ = DRONE_BPF_PROFILES[drone_type]
        ratios = []
        for i in idxs:
            try:
                item = synth_ds[i]
                # item = (mel_tensor, ipd_tensor, label_tensor)
                # We need the audio — re-synthesise cheaply
                from .audio_processing import synthesise_drone
                pos = synth_ds.positions[i]
                chs = synthesise_drone(
                    cfg.MIC_POSITIONS, pos[:2],
                    noise_level=0.03, drone_type=drone_type,
                    noise_profile="mixed", cfg=cfg
                )
                y = ap.pad_or_truncate(chs[0])
                ratios.append(ap.compute_bpf_energy_ratio(y, f_mid))
            except Exception:
                pass
        bpf_data[drone_type] = ratios

    if not bpf_data:
        print("  ⚠  No BPF ratio data — skipping fig9"); return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Synthetic localization clips — BPF energy ratio by drone type\n"
                 "(dashed lines = measured Q2 means from config)",
                 fontsize=12, fontweight="bold")

    labels  = list(bpf_data.keys())
    data    = [bpf_data[l] for l in labels]
    colours_used = colours[:len(labels)]

    bp = ax.boxplot(data, labels=[l.replace("_", "\n") for l in labels],
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    whiskerprops=dict(color="gray"),
                    capprops=dict(color="gray"),
                    flierprops=dict(marker=".", color="gray", markersize=4))
    for patch, col in zip(bp["boxes"], colours_used):
        patch.set_facecolor(col); patch.set_alpha(0.65)

    # Overlay measured means
    for xi, lbl in enumerate(labels, start=1):
        if lbl in DRONE_BPF_ENERGY_RATIOS:
            ax.axhline(DRONE_BPF_ENERGY_RATIOS[lbl],
                       xmin=(xi-1.4)/len(labels),
                       xmax=(xi-0.6)/len(labels),
                       color="black", lw=1.5, ls="--", alpha=0.7)

    ax.set_ylabel("BPF energy ratio")
    _save(fig, out_dir / "fig9_synth_bpf_by_type.png")


# ── Fig 10: Noise profile PSD comparison ─────────────────────────────────────

def fig10_noise_profiles(cfg, builder_root: Path, out_dir: Path):
    from .audio_processing import _make_indoor_noise, _make_outdoor_noise

    sr  = cfg.SR
    dur = 4.0
    n   = int(sr * dur)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Noise profile comparison — real recordings vs synthetic models",
                 fontsize=12, fontweight="bold")

    # Real MEMS
    mems_wavs = _list_wavs(builder_root / "clean_drone_sections" / "MEMS")[:8]
    if mems_wavs:
        f_m, psd_m = _avg_psd(mems_wavs, sr)
        if len(f_m):
            ax.plot(f_m, psd_m, color=PALETTE["MEMS"], lw=1.5,
                    label=f"MEMS real avg (n={len(mems_wavs)})")

    # Real Brüel
    bruel_wavs = _list_wavs(builder_root / "clean_drone_sections" / "BRUEL")[:8]
    if bruel_wavs:
        f_b, psd_b = _avg_psd(bruel_wavs, sr)
        if len(f_b):
            ax.plot(f_b, psd_b, color=PALETTE["BRUEL"], lw=1.5, ls="--",
                    label=f"Brüel real avg (n={len(bruel_wavs)})")

    # Synthetic indoor model
    n_indoor = _make_indoor_noise(n, sr, amplitude=0.015).astype(np.float32)
    f_si, psd_si = _avg_psd([None], sr)  # placeholder
    f_si, psd_si = sp_signal.welch(n_indoor, fs=sr, nperseg=4096)
    ax.plot(f_si, 10 * np.log10(psd_si + 1e-20), color=PALETTE["indoor"],
            lw=1.0, ls=":", alpha=0.85, label="Synthetic indoor model")

    # Synthetic outdoor model
    n_outdoor = _make_outdoor_noise(n, sr, amplitude=0.015).astype(np.float32)
    f_so, psd_so = sp_signal.welch(n_outdoor, fs=sr, nperseg=4096)
    ax.plot(f_so, 10 * np.log10(psd_so + 1e-20), color=PALETTE["outdoor"],
            lw=1.0, ls=":", alpha=0.85, label="Synthetic outdoor model")

    # BPF harmonic markers (82 Hz)
    for h in range(1, 8):
        fh = 82.0 * h
        if fh < sr / 2:
            ax.axvline(fh, color="cyan", lw=0.6, ls=":", alpha=0.5)
            ax.text(fh + 30, -100, f"H{h}", fontsize=7, color="cyan", rotation=90)

    ax.set_xscale("log"); ax.set_xlim(20, sr / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out_dir / "fig10_noise_profiles.png")


# =============================================================================
#  OPTIONAL: full train_all() run
# =============================================================================

def run_training(cfg, det_epochs: int, loc_epochs: int):
    from .orchestration import train_all
    print("\n" + "="*68)
    print(f"  TRAINING  det_epochs={det_epochs}  loc_epochs={loc_epochs}")
    print("="*68)
    train_all(
        cfg,
        det_epochs=det_epochs,
        loc_epochs=loc_epochs,
        resume=True,
        force_rebuild_cache=False,
        custom_builder_root=cfg.CUSTOM_DATASET_ROOT,
        use_custom_builder=True,
        import_custom_backgrounds_as_non_drone=False,
        download_builtin_detection_dataset=True,
        download_external_audio=False,
    )


# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Mirror of train_all() — build datasets + thesis figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--picker-output",    required=True,
                    help="Root of manual_drone_section_picker_v3 output")
    ap.add_argument("--base-dir",         default=None,
                    help="drone_detection LOCAL_BASE (default /tmp/drone_v15)")
    ap.add_argument("--out-figures",      default="./thesis_figures",
                    help="Output directory for figures")
    ap.add_argument("--gpx",              default=None)
    ap.add_argument("--bpf-hz",           type=float, default=82.0,
                    help="Blade-pass fundamental Hz for Dunakeszi clips (default 82)")
    ap.add_argument("--split-fracs",      type=float, nargs=3,
                    default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--no-train",         action="store_true",
                    help="Only build datasets + figures, skip actual training")
    ap.add_argument("--det-epochs",       type=int, default=30)
    ap.add_argument("--loc-epochs",       type=int, default=30)
    ap.add_argument("--download-builtin", action="store_true", default=True,
                    help="Download DroneAudioDataset from GitHub (default True)")
    ap.add_argument("--no-download-builtin", dest="download_builtin",
                    action="store_false")
    ap.add_argument("--download-external", action="store_true", default=False)
    ap.add_argument("--colab",            action="store_true")
    ap.add_argument("--figs",             type=int, nargs="+",
                    default=list(range(1, 11)),
                    help="Which figures to generate (default: all 1-10)")
    args = ap.parse_args()

    picker_output = Path(args.picker_output)
    out_dir       = Path(args.out_figures)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = (Path(args.base_dir) if args.base_dir
                else (Path("/content/drone_v15") if args.colab
                      else Path("/tmp/drone_v15")))

    print("=" * 68)
    print("  Training dataset mirror + thesis figures")
    print("=" * 68)
    print(f"  Picker output  : {picker_output}")
    print(f"  Base dir       : {base_dir}")
    print(f"  Output figures : {out_dir}")

    # ── Config ────────────────────────────────────────────────────────────
    try:
        cfg, builder_root = _make_cfg(base_dir, args.colab, picker_output,
                                       tuple(args.split_fracs))
    except (ImportError, ModuleNotFoundError) as e:
        print(f"\n\u274c Import error: {e}")
        print("\n  Diagnosis:")
        print(f"  Script location : {Path(__file__).resolve()}")
        print("  The script must be placed either:")
        print("    INSIDE  the package  →  deployment/v2/drone_detection/build_training_datasets_and_figures.py")
        print("    OUTSIDE the package  →  deployment/v2/build_training_datasets_and_figures.py")
        print("             (with drone_detection/ as a subfolder next to it)")
        sys.exit(1)

    # ── Step 1: Builder root layout ───────────────────────────────────────
    print(f"\n{'─'*68}\n  STEP 1 — Picker output → builder root layout\n{'─'*68}")
    _build_builder_root(
        picker_output, builder_root,
        train_frac=args.split_fracs[0], val_frac=args.split_fracs[1],
    )

    # ── Step 2: Detection pipeline ────────────────────────────────────────
    det_result = run_detection_pipeline(
        cfg, builder_root,
        download_builtin=args.download_builtin,
        download_external=args.download_external,
    )
    pre_counts  = det_result.get("pre",  {})
    post_counts = det_result.get("post", {})
    mel_counts  = det_result.get("mel",  {})

    # ── Step 3: Localization pipeline ─────────────────────────────────────
    loc_result = run_localization_pipeline(cfg)

    # ── Step 4: Figures ───────────────────────────────────────────────────
    print(f"\n{'─'*68}\n  GENERATING FIGURES  →  {out_dir}\n{'─'*68}")

    fig_map = {
        1:  lambda: fig1_detection_class_balance(post_counts, out_dir),
        2:  lambda: fig2_detection_source_breakdown(pre_counts, post_counts, out_dir),
        3:  lambda: fig3_mel_grid(cfg, builder_root, out_dir),
        4:  lambda: fig4_bpf_ratio(cfg, builder_root, out_dir, args.bpf_hz),
        5:  lambda: fig5_mix_snr_distribution(cfg, out_dir),
        6:  lambda: fig6_mel_cache_balance(mel_counts, out_dir),
        7:  lambda: fig7_loc_spatial_coverage(cfg, loc_result, out_dir),
        8:  lambda: fig8_loc_dist_height(cfg, loc_result, out_dir),
        9:  lambda: fig9_synth_bpf_by_type(cfg, loc_result, out_dir),
        10: lambda: fig10_noise_profiles(cfg, builder_root, out_dir),
    }

    for n in args.figs:
        if n in fig_map:
            print(f"\n  Generating figure {n} …")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    fig_map[n]()
                except Exception as e:
                    print(f"  ⚠  Figure {n} failed: {e}")
        else:
            print(f"  ⚠  No figure {n} defined.")

    # ── Step 5 (optional): full training ─────────────────────────────────
    if not args.no_train:
        run_training(cfg, args.det_epochs, args.loc_epochs)

    print(f"\n{'='*68}")
    print("  DONE")
    print(f"  Figures  → {out_dir}")
    print(f"  Base dir → {base_dir}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()