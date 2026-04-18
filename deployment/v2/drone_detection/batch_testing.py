# -*- coding: utf-8 -*-
"""
batch_testing.py
────────────────
Utilities for batch-testing and diagnosing audio files.

batch_test_audio()   — run analyse_audio_file() on a drone / non_drone folder
diagnose_file()      — print a per-segment feature breakdown for one file
_verify_rms_gate()   — self-test that the RMS gate correctly mutes near-silence
"""

import math
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import random
import torch

from .config import Config, config
from .audio_processing import AudioProcessor
from .inference import heuristic_detect, load_detection_model, detect, analyse_audio_file
from .visualization import PLOT_STYLE, _apply_dark_style, _show_inline, _save_plot


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


# ══════════════════════════════════════════════════════════════════════════════
# Batch tester
# ══════════════════════════════════════════════════════════════════════════════

def batch_test_audio(
    test_data_path,
    cfg:               Optional[Config] = None,
    max_per_category:  int   = 10,
    n_segments:        int   = 8,
    threshold_override: Optional[float] = None,
    show_plot:         bool  = True,
    shuffle:           bool  = True,
    seed:              int   = 42,
) -> dict:
    """
    Batch-test audio files from a folder with drone/ and non_drone/ subfolders.

    Parameters
    ──────────
    test_data_path    : root folder containing 'drone/' and/or 'non_drone/'
    max_per_category  : maximum files to test per category
    n_segments        : segments per file passed to analyse_audio_file
    threshold_override: temporarily override cfg.DETECTION_THRESHOLD
    show_plot         : show 6-panel dashboard after each file
    shuffle           : shuffle files before limiting

    Returns
    ───────
    dict with per-category and overall statistics
    """
    import random as _random
    cfg  = cfg or config
    root = Path(test_data_path)

    # Discover category directories
    cat_files = {}
    for cat in ["drone", "non_drone"]:
        cat_dir = root / cat
        if not cat_dir.exists():
            cat_dir = root / cat.replace("_", " ")   # space variant
        if cat_dir and cat_dir.is_dir():
            files = [p for p in cat_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
            if shuffle:
                rng = _random.Random(seed); rng.shuffle(files)
            cat_files[cat] = files[:max_per_category]
        else:
            cat_files[cat] = []

    total_files = sum(len(v) for v in cat_files.values())
    if total_files == 0:
        print(f"❌ No audio files found under {root}")
        print(f"   Expected subfolders: drone, non_drone")
        return {}

    print("=" * 65)
    print(f"  Batch Test — {root.name}")
    print(f"  max_per_category={max_per_category}  n_segments={n_segments}")
    print(f"  threshold={threshold_override or cfg.DETECTION_THRESHOLD:.3f}")
    for cat, files in cat_files.items():
        print(f"  {cat:12s}: {len(files)} files queued")
    print("=" * 65)

    summary = {}

    for cat, files in cat_files.items():
        if not files:
            continue
        true_label = 1 if cat == "drone" else 0
        results    = []

        print(f"\n{'━'*65}")
        print(f"  Category: {cat.upper()}  ({len(files)} files)")
        print(f"{'━'*65}")

        for file_idx, fpath in enumerate(files):
            print(f"\n[{file_idx+1}/{len(files)}] {fpath.name}")
            try:
                r = analyse_audio_file(
                    str(fpath), cfg,
                    n_segments=n_segments,
                    threshold_override=threshold_override,
                    show_plot=show_plot,
                )
                predicted = 1 if r["detected"] else 0
                correct   = int(predicted == true_label)
                results.append({
                    "file": fpath.name, "prob": r["probability"],
                    "detected": r["detected"], "correct": correct,
                    "true": true_label, "predicted": predicted,
                    "duration": r["duration_sec"],
                    "n_det_segs": sum(s["detected"] for s in r["segments"]),
                    "n_segs":    len(r["segments"]),
                })
                verdict = "✅ correct" if correct else "❌ wrong"
                print(f"  → prob={r['probability']:.3f}  detected={r['detected']}  {verdict}")
            except Exception as e:
                traceback.print_exc()
                print(f"  ⚠️  Error on {fpath.name}: {e}")

        if results:
            n       = len(results)
            n_corr  = sum(r["correct"] for r in results)
            probs   = [r["prob"] for r in results]
            tp      = sum(1 for r in results if r["predicted"] == 1 and r["true"] == 1)
            fp      = sum(1 for r in results if r["predicted"] == 1 and r["true"] == 0)
            tn      = sum(1 for r in results if r["predicted"] == 0 and r["true"] == 0)
            fn      = sum(1 for r in results if r["predicted"] == 0 and r["true"] == 1)

            print(f"\n  ── {cat} summary ──")
            print(f"  Files tested : {n}")
            print(f"  Correct      : {n_corr}/{n}  ({n_corr/n*100:.1f}%)")
            print(f"  Mean prob    : {float(np.mean(probs)):.3f}  ± {float(np.std(probs)):.3f}")
            print(f"  Min/Max prob : {min(probs):.3f} / {max(probs):.3f}")
            print(f"  TP/FP/TN/FN  : {tp}/{fp}/{tn}/{fn}")

            _plot_category_summary(cat, results, cfg)

            summary[cat] = {
                "n": n, "accuracy": n_corr / n, "correct": n_corr,
                "mean_prob": float(np.mean(probs)), "std_prob": float(np.std(probs)),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "results": results,
            }

    # Overall summary
    all_results = [r for v in summary.values() for r in v["results"]]
    if all_results:
        _plot_overall_summary(summary, cfg)
        n_tot  = len(all_results)
        n_corr = sum(r["correct"] for r in all_results)
        tp  = sum(1 for r in all_results if r["predicted"] == 1 and r["true"] == 1)
        fp  = sum(1 for r in all_results if r["predicted"] == 1 and r["true"] == 0)
        tn  = sum(1 for r in all_results if r["predicted"] == 0 and r["true"] == 0)
        fn  = sum(1 for r in all_results if r["predicted"] == 0 and r["true"] == 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"\n{'='*65}")
        print(f"  OVERALL: {n_corr}/{n_tot} correct ({100*n_corr/n_tot:.1f}%)")
        print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
        print(f"  TP/FP/TN/FN: {tp}/{fp}/{tn}/{fn}")
        print(f"{'='*65}")
        summary["overall"] = {
            "n": n_tot, "correct": n_corr, "accuracy": n_corr / n_tot,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Diagnose file
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_file(
    audio_path: str,
    cfg:        Optional[Config] = None,
    n_segs:     int = 4,
):
    """
    Print a detailed feature breakdown for each segment of a file.
    Useful for understanding why a file is misclassified.

    Per-segment output
    ──────────────────
    t_start, rms_db, CNN prob, heuristic prob, fused prob,
    median_entropy, crest_factor, comb_score, f0_median, veto reason
    """
    cfg = cfg or config
    ap  = AudioProcessor(cfg)
    y   = ap.load(audio_path, mono=True)
    total_s = len(y) / cfg.SR
    seg_n   = int(cfg.TARGET_DURATION * cfg.SR)
    hop     = max(seg_n, int((len(y) - seg_n) / max(n_segs - 1, 1)))

    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS: {Path(audio_path).name}  ({total_s:.2f}s)")
    print(f"{'='*70}")
    print(f"  {'Seg':>3}  {'t':>5}  {'rms':>6}  {'CNN':>5}  {'Heur':>5}  "
          f"{'Fused':>5}  {'ent':>5}  {'CF':>5}  {'comb':>5}  {'f0':>5}  veto")
    print(f"  {'-'*66}")

    load_detection_model(cfg)
    m = load_detection_model(cfg)

    for i in range(n_segs):
        start = min(i * hop, max(0, len(y) - seg_n))
        seg   = y[start : start + seg_n]
        if len(seg) < seg_n:
            seg = np.pad(seg, (0, seg_n - len(seg)))

        rms_db = float(20 * math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-8))
        h      = heuristic_detect(seg, cfg)
        feats  = h.get("features", {})
        ent    = feats.get("median_frame_entropy", float("nan"))
        cf     = feats.get("crest_factor",         float("nan"))
        comb   = feats.get("comb_score",           float("nan"))
        f0     = feats.get("f0_median_hz",         float("nan"))
        veto   = feats.get("veto", "-")

        try:
            feat = ap.feature_stack(ap.pad_or_truncate(seg))
            x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
            with torch.no_grad():
                cnn_p = float(torch.softmax(m(x), dim=1)[0, 1].item())
        except Exception:
            cnn_p = float("nan")

        fused = 0.80 * cnn_p + 0.20 * h["probability"]
        t_s   = start / cfg.SR

        def _fmt(v, fmt): return fmt.format(v) if not math.isnan(v) else "  nan"
        print(f"  {i+1:>3}  {t_s:>5.1f}  {rms_db:>6.1f}  "
              f"{cnn_p:>5.3f}  {h['probability']:>5.3f}  {fused:>5.3f}  "
              f"{_fmt(ent,'{:.3f}'):>5}  {_fmt(cf,'{:.2f}'):>5}  "
              f"{_fmt(comb,'{:.3f}'):>5}  {_fmt(f0,'{:.0f}'):>5}  {veto}")

    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

def _verify_rms_gate(cfg: Optional[Config] = None):
    """Verify that near-silent audio is correctly gated to probability=0."""
    import scipy.signal
    cfg = cfg or config
    sr  = cfg.SR
    t   = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)

    loud = (np.sin(2 * np.pi * 800 * t).astype(np.float32) * 0.0025)
    loud /= np.max(np.abs(loud)) + 1e-8
    loud *= 10 ** (-52 / 20)

    result = detect([loud, loud, loud], cfg)
    print("=== RMS gate self-test ===")
    print(f"  Near-silent audio (-52 dB): prob={result['probability']:.3f}  "
          f"veto={result['heuristic_features'].get('veto','none')}")
    assert result["probability"] == 0.0, \
        f"Expected 0.0 for -52 dB audio, got {result['probability']:.3f}"

    normal  = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    normal *= 10 ** (-12 / 20)
    result2 = detect([normal, normal, normal], cfg)
    print(f"  Normal audio (-12 dB):      prob={result2['probability']:.3f}  (should be > 0)")
    assert result2["probability"] > 0.0, "Normal audio should not be gated"

    print("=== RMS gate test PASSED ✅ ===\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers (internal)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_category_summary(cat: str, results: list, cfg: Config):
    """Probability histogram + per-file bar chart for one test category."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n     = len(results)
    probs = [r["prob"] for r in results]
    names = [r["file"][:28] + "…" if len(r["file"]) > 30 else r["file"] for r in results]
    cols  = [PLOT_STYLE["ok"] if r["correct"] else PLOT_STYLE["err"] for r in results]

    fig = plt.figure(figsize=(14, 4), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"Category: {cat.upper()}", color=PLOT_STYLE["accent"],
                 fontsize=12, fontweight="bold")
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    _apply_dark_style(fig, axes)

    ax = axes[0]
    ax.barh(range(n), probs, color=cols, alpha=0.8)
    ax.axvline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"], lw=1.5, ls="--",
               label=f"thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Probability"); ax.set_title("Per-file score")
    ax.legend(facecolor=PLOT_STYLE["panel"], fontsize=8)

    ax = axes[1]
    ax.hist(probs, bins=np.linspace(0, 1, 21), color=PLOT_STYLE["purple"],
            edgecolor=PLOT_STYLE["bg"], alpha=0.85)
    ax.axvline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"], lw=1.5, ls="--")
    ax.axvline(float(np.mean(probs)), color=PLOT_STYLE["accent"], lw=1.5,
               label=f"mean={float(np.mean(probs)):.3f}")
    ax.set_xlabel("Probability"); ax.set_ylabel("Count"); ax.set_title("Distribution")
    ax.legend(facecolor=PLOT_STYLE["panel"], fontsize=8)

    plt.tight_layout()
    try:
        cfg.DRIVE_PLOTS.mkdir(parents=True, exist_ok=True)
        out = cfg.DRIVE_PLOTS / f"batch_test_{cat}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  💾 Plot saved: {out}")
    except Exception:
        pass
    _show_inline(fig); plt.close(fig)


def _plot_overall_summary(summary: dict, cfg: Config):
    """2-panel overall summary: confusion matrix + category accuracy bars."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    cats_with_data = [c for c in ["drone", "non_drone"] if c in summary]
    all_results    = [r for c in cats_with_data for r in summary[c]["results"]]

    fig = plt.figure(figsize=(12, 4), facecolor=PLOT_STYLE["bg"])
    fig.suptitle("Batch Test — Overall Summary", color=PLOT_STYLE["accent"],
                 fontsize=13, fontweight="bold")
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    _apply_dark_style(fig, axes)

    # Confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for r in all_results:
        cm[r["true"], r["predicted"]] += 1
    ax  = axes[0]
    cmap = LinearSegmentedColormap.from_list("", [PLOT_STYLE["panel"], PLOT_STYLE["accent"]])
    ax.imshow(cm, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f"{cm[i,j]}\n({pct:.0f}%)", ha="center", va="center",
                    color=PLOT_STYLE["text"], fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: non_drone", "pred: drone"], color=PLOT_STYLE["text"], fontsize=8)
    ax.set_yticklabels(["true: non_drone", "true: drone"], color=PLOT_STYLE["text"], fontsize=8)
    ax.set_title("Confusion matrix")

    # Accuracy per category
    ax       = axes[1]
    cat_names = list(cats_with_data)
    accs      = [summary[c]["accuracy"] * 100 for c in cat_names]
    bar_cols  = [PLOT_STYLE["ok"] if a >= 70 else PLOT_STYLE["warn"] if a >= 50
                 else PLOT_STYLE["err"] for a in accs]
    bars = ax.bar(cat_names, accs, color=bar_cols, alpha=0.8, width=0.4)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", color=PLOT_STYLE["text"], fontsize=10)
    ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)"); ax.set_title("Per-category accuracy")
    ax.axhline(50, color=PLOT_STYLE["muted"], lw=1, ls=":")

    plt.tight_layout()
    try:
        out = cfg.DRIVE_PLOTS / "batch_test_overall.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  💾 Overall plot saved: {out}")
    except Exception:
        pass
    _show_inline(fig); plt.close(fig)
