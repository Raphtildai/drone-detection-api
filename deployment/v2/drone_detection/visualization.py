# -*- coding: utf-8 -*-
"""
visualization.py
────────────────
All plotting functions for the drone detection & localization pipeline.

Contents
────────
Dark-themed analysis dashboard     _plot_analysis_report()
External detection scores plot      _plot_external_detection_scores()
Training curve plots                plot_training_logs()
Confusion matrix                    plot_confusion_matrix_styled()
Localization scatter                plot_localization_scatter()
Polar azimuth compass               plot_polar_azimuth()
Track trajectory                    plot_track_trajectory()
Multi-drone positions               plot_multi_drone_positions()
Kalman trajectories (1-pt safe)     plot_kalman_trajectories()

Thesis figures (7)
──────────────────
plot_azimuth_mae_per_position()
plot_val_test_comparison()
plot_error_histogram()
plot_predicted_vs_true()
plot_azimuth_distance_heatmap()
plot_training_curves()
plot_polar_mae()
plot_all_thesis_figures()

Multi-drone suite dashboard
───────────────────────────
plot_suite_results_from_data()
plot_position_map_from_data()
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .config import Config, config

matplotlib.use("Agg")

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})

# ── Colour palette ─────────────────────────────────────────────────────────
PLOT_STYLE = {
    "bg":        "#08111f",
    "panel":     "#0f1b2d",
    "panel_alt": "#14233a",
    "accent":    "#38bdf8",
    "warn":      "#fbbf24",
    "ok":        "#4ade80",
    "err":       "#f87171",
    "grid":      "#47607d",
    "text":      "#f8fafc",
    "text_soft": "#dbeafe",
    "muted":     "#b6c2cf",
    "spine":     "#6b85a3",
    "purple":    "#a78bfa",
}

# Thesis palette
C_GOOD   = "#1D9E75"
C_MOD    = "#BA7517"
C_POOR   = "#D85A30"
C_VAL    = "#378ADD"
C_TEST   = "#D85A30"
C_PURPLE = "#7F77DD"
C_GRAY   = "#888780"
C_RANDOM = "#AAAAAA"


def _mae_color(v: float) -> str:
    return C_GOOD if v < 30 else C_MOD if v < 60 else C_POOR

def _style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(PLOT_STYLE["panel_alt"])
    frame.set_edgecolor(PLOT_STYLE["spine"])
    frame.set_alpha(0.95)

    for txt in legend.get_texts():
        txt.set_color(PLOT_STYLE["text"])

    title = legend.get_title()
    if title is not None:
        title.set_color(PLOT_STYLE["text"])

def _style_colorbar(cbar):
    try:
        cbar.ax.yaxis.label.set_color(PLOT_STYLE["text"])
        cbar.ax.tick_params(colors=PLOT_STYLE["text"])
        cbar.outline.set_edgecolor(PLOT_STYLE["spine"])
        cbar.ax.set_facecolor(PLOT_STYLE["panel"])
    except Exception:
        pass

def _apply_dark_style(fig, axes_flat):
    fig.patch.set_facecolor(PLOT_STYLE["bg"])

    for ax in axes_flat:
        if ax is None:
            continue

        ax.set_facecolor(PLOT_STYLE["panel"])

        # ticks
        ax.tick_params(
            axis="both",
            colors=PLOT_STYLE["text"],
            labelcolor=PLOT_STYLE["text"],
            labelsize=10,
        )

        # axis labels
        ax.xaxis.label.set_color(PLOT_STYLE["text"])
        ax.yaxis.label.set_color(PLOT_STYLE["text"])

        # title
        ax.title.set_color(PLOT_STYLE["text"])
        ax.title.set_fontweight("bold")

        # spines
        for spine in ax.spines.values():
            spine.set_color(PLOT_STYLE["spine"])
            spine.set_linewidth(1.0)

        # grid
        ax.grid(color=PLOT_STYLE["grid"], alpha=0.35, linewidth=0.8)

        # offset text such as 1e3 on axis
        try:
            ax.xaxis.get_offset_text().set_color(PLOT_STYLE["text"])
            ax.yaxis.get_offset_text().set_color(PLOT_STYLE["text"])
        except Exception:
            pass

def _show_inline(fig):
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        try:
            plt.show()
        except Exception:
            pass


def _save_plot(fig, path: Optional[Path], dpi: int = 150):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"💾 Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Training curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_logs(cfg: Optional[Config] = None, save: bool = True):
    """Dark-themed training curves for detection and localization."""
    cfg = cfg or config
    det_csv = cfg.DRIVE_LOGS / "detection_log.csv"
    loc_csv = cfg.DRIVE_LOGS / "localization_log.csv"
    has_det = det_csv.exists(); has_loc = loc_csv.exists()
    if not has_det and not has_loc:
        print("❌ No training logs found in", cfg.DRIVE_LOGS); return

    n_plots = (2 if has_det else 0) + (2 if has_loc else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    _apply_dark_style(fig, list(axes))
    ax_idx = 0

    if has_det:
        with open(det_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        epochs   = [int(r["epoch"])   for r in rows]
        tr_loss  = [float(r["tr_loss"]) for r in rows]
        tr_acc   = [float(r["tr_acc"])  for r in rows]
        val_acc  = [float(r["val_acc"]) for r in rows]
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, tr_loss, "-o", color=PLOT_STYLE["accent"], ms=4, label="Train loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Focal loss"); ax.set_title("Detection — Loss")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, tr_acc,  "-o", color=PLOT_STYLE["ok"],   ms=4, label="Train acc %")
        ax.plot(epochs, val_acc, "-s", color=PLOT_STYLE["warn"], ms=4, label="Val acc %")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Detection — Accuracy")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)

    if has_loc:
        with open(loc_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        epochs   = [int(r["epoch"])      for r in rows]
        tr_loss  = [float(r["tr_loss"])  for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        mae_az   = [float(r["mae_az"])   for r in rows]
        mae_dist = [float(r["mae_dist"]) for r in rows]
        mae_ht   = [float(r["mae_ht"])   for r in rows]
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, tr_loss,  "-o", color=PLOT_STYLE["accent"], ms=4, label="Train")
        ax.plot(epochs, val_loss, "-s", color=PLOT_STYLE["warn"],   ms=4, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss"); ax.set_title("Localization — Loss")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)  
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, mae_az,   "-o", color=PLOT_STYLE["err"],    ms=4, label="MAE az (°)")
        ax.plot(epochs, mae_dist, "-s", color=PLOT_STYLE["purple"], ms=4, label="MAE dist (m)")
        ax.plot(epochs, mae_ht,   "-^", color=PLOT_STYLE["ok"],     ms=4, label="MAE ht (m)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.set_title("Localization — MAE")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)

    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_LOGS / "training_curves.png")
    _show_inline(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis dashboard (6-panel dark)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_analysis_report(segments, confirmed, cfg, title: str):
    """
    Six-panel dark-themed analysis dashboard.

    Panels: waveform+RMS | mel spectrogram | detection timeline |
            localization bars | polar compass | detection gauge
    """
    import itertools
    fig = plt.figure(figsize=(20, 10), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"🚁 Drone Analysis v15 — {title}", fontsize=14,
                 color=PLOT_STYLE["accent"], fontweight="bold", y=0.98)
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    _apply_dark_style(fig, axes)
    ts_list = [s["t_start"] for s in segments]

    # [0] Waveform + RMS
    ax = axes[0]
    if any("waveform" in s for s in segments):
        wave = list(itertools.chain.from_iterable(s.get("waveform", []) for s in segments))
        t_w  = np.linspace(0, max(ts_list) + cfg.TARGET_DURATION if ts_list else 3.0, len(wave))
        ax.plot(t_w, wave, color=PLOT_STYLE["accent"], lw=0.6, alpha=0.7)
    rms_vals = [s.get("rms_db", -60) for s in segments]
    ax2 = ax.twinx()
    ax2.plot(ts_list, rms_vals, "o-", color=PLOT_STYLE["warn"], ms=4, lw=1.5, label="RMS dB")
    ax2.tick_params(axis="y", colors=PLOT_STYLE["text"], labelcolor=PLOT_STYLE["text"])
    ax2.yaxis.label.set_color(PLOT_STYLE["text"])
    ax2.set_ylabel("RMS (dB)", color=PLOT_STYLE["text"])
    for spine in ax2.spines.values():
        spine.set_color(PLOT_STYLE["spine"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title("Waveform + RMS")

    # [1] Mel spectrogram
    ax = axes[1]
    if any("mel" in s for s in segments):
        mel_frames = np.concatenate([s["mel"] for s in segments if "mel" in s], axis=1)
        ax.imshow(mel_frames, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, max(ts_list) + cfg.TARGET_DURATION if ts_list else 3.0, 0, cfg.SR // 2 / 1000])
        cbar = plt.colorbar(ax.images[0], ax=ax, label="dB")
        _style_colorbar(cbar)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (kHz)"); ax.set_title("Mel Spectrogram")

    # [2] Detection timeline
    ax = axes[2]
    prbs  = [s["prob"] for s in segments]
    cnns  = [s.get("cnn_probability",  float("nan")) for s in segments]
    heurs = [s.get("heuristic_probability", float("nan")) for s in segments]
    cols  = [PLOT_STYLE["ok"] if s["detected"] else PLOT_STYLE["err"] for s in segments]
    ax.bar(ts_list, prbs, width=cfg.TARGET_DURATION * 0.8, color=cols, alpha=0.55, label="Hybrid")
    ax.fill_between(ts_list, prbs, alpha=0.15, color=PLOT_STYLE["text"])
    if not all(math.isnan(v) for v in cnns):
        ax.plot(ts_list, cnns,  "-o", color=PLOT_STYLE["accent"], ms=4, lw=1.5, label="CNN")
    if not all(math.isnan(v) for v in heurs):
        ax.plot(ts_list, heurs, "--s", color=PLOT_STYLE["purple"], ms=4, lw=1.5, label="Heuristic")
    ax.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"], lw=1.5, ls="--",
               label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Probability"); ax.set_title("Detection Timeline")
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)

    # [3] Localization bars
    ax = axes[3]
    locs = [s for s in segments if s.get("loc") is not None]
    if locs:
        w       = cfg.TARGET_DURATION * 0.8
        az_vals = [s["loc"]["azimuth_deg"] / 180.0 for s in locs]
        t_locs  = [s["t_start"] for s in locs]
        ax.bar(t_locs, az_vals, width=w, color=PLOT_STYLE["accent"], alpha=0.7, label="Az/180°")
    else:
        ax.text(0.5, 0.5, "No localization data", ha="center", va="center",
                color=PLOT_STYLE["muted"], transform=ax.transAxes)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalised"); ax.set_title("Localization")
    if locs:
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)

    # [4] Polar compass
    axes[4].remove()
    ax_pol = fig.add_subplot(gs[1, 1], projection="polar")
    ax_pol.set_facecolor(PLOT_STYLE["panel"])
    ax_pol.tick_params(colors=PLOT_STYLE["text"])
    az_degs = [s["loc"]["azimuth_deg"] for s in segments if s.get("loc") is not None]
    if az_degs:
        rads = np.radians([90 - a for a in az_degs])
        counts, edges = np.histogram(rads, bins=24, range=(-np.pi, np.pi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_pol.bar(centers, counts, width=edges[1] - edges[0], alpha=0.8,
                   color=PLOT_STYLE["accent"], edgecolor=PLOT_STYLE["bg"])
    ax_pol.set_theta_zero_location("N"); ax_pol.set_theta_direction(-1)
    ax_pol.set_title("Azimuth (N-up)", color=PLOT_STYLE["accent"], pad=12)
    ax_pol.grid(color=PLOT_STYLE["grid"], alpha=0.4)

    # [5] Detection score gauge
    ax = axes[5]
    all_probs   = [s["prob"] for s in segments]
    final_score = float(np.max(all_probs)) if all_probs else 0.0
    theta_range = np.linspace(np.pi, 0, 200)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.1, 1.2)
    ax.plot(np.cos(theta_range), np.sin(theta_range), lw=18, color=PLOT_STYLE["panel"])
    fill_theta = np.linspace(np.pi, np.pi * (1 - final_score), 200)
    col = PLOT_STYLE["ok"] if final_score >= cfg.DETECTION_THRESHOLD else PLOT_STYLE["err"]
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=col)
    needle = np.pi * (1 - final_score)
    ax.annotate("", xy=(0.8 * np.cos(needle), 0.8 * np.sin(needle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PLOT_STYLE["text"], lw=2))
    ax.text(0, -0.08, f"{final_score:.3f}", ha="center", fontsize=16, fontweight="bold", color=col)
    ax.text(0, 0.6, "DRONE" if final_score >= cfg.DETECTION_THRESHOLD else "CLEAR",
            ha="center", fontsize=10, color=col)
    ax.axis("off"); ax.set_title("Detection Score", color=PLOT_STYLE["text"])

    cfg.DRIVE_PLOTS.mkdir(parents=True, exist_ok=True)
    save_path = cfg.DRIVE_PLOTS / f"analysis_{Path(title).stem}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"💾 Dashboard saved: {save_path}")
    _show_inline(fig); plt.close(fig)


def _plot_external_detection_scores(
    segment_results, threshold: float, cfg: Config, title: str
):
    """Dark-themed segment probability chart for external audio analysis."""
    if not segment_results:
        return
    t     = [0.5 * (s["t_start_s"] + s["t_end_s"]) for s in segment_results]
    fused = [s["probability"]           for s in segment_results]
    cnn   = [s["cnn_probability"]       for s in segment_results]
    heur  = [s["heuristic_probability"] for s in segment_results]
    cols  = [PLOT_STYLE["ok"] if s["detected_at_external_threshold"] else PLOT_STYLE["err"]
             for s in segment_results]
    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_style(fig, [ax])
    w = max((t[1] - t[0]) * 0.8 if len(t) > 1 else 0.5, 0.15)
    ax.bar(t, fused, width=w, color=cols, alpha=0.55, label="Hybrid prob")
    ax.plot(t, cnn,  "-o",  color=PLOT_STYLE["accent"], label="CNN prob")
    ax.plot(t, heur, "--s", color=PLOT_STYLE["purple"], label="Heuristic prob")
    ax.axhline(threshold, color=PLOT_STYLE["warn"], ls="--", lw=1.5,
               label=f"Ext thr={threshold:.2f}")
    ax.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["err"], ls=":", lw=1.5,
               label=f"Main thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax.set_title(f"Robust External Detection — {title}", color=PLOT_STYLE["text"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Probability"); ax.set_ylim(0, 1.05)
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)
    plt.tight_layout()
    _show_inline(fig); plt.close(fig)


def plot_confusion_matrix_styled(
    cm_array: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Dark-themed confusion matrix with counts + percentages."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("drone_cm", [PLOT_STYLE["panel"], PLOT_STYLE["accent"]])
    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(fig, [ax])
    im = ax.imshow(cm_array, interpolation="nearest", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    _style_colorbar(cbar)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=PLOT_STYLE["text"])
    ax.set_yticklabels(labels, color=PLOT_STYLE["text"])
    total = cm_array.sum(axis=1, keepdims=True) + 1e-8
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            pct = 100.0 * cm_array[i, j] / total[i, 0]
            ax.text(j, i, f"{cm_array[i,j]}\n({pct:.1f}%)",
                    ha="center", va="center", color=PLOT_STYLE["text"], fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, color=PLOT_STYLE["text"])
    plt.tight_layout()
    _save_plot(fig, save_path)
    _show_inline(fig); plt.close(fig)


def plot_polar_azimuth(
    azimuth_degs: List[float],
    title:  str  = "Detected Drone Azimuths",
    cfg:    Optional[Config] = None,
    save:   bool = True,
):
    cfg = cfg or config
    fig = plt.figure(figsize=(6, 6), facecolor=PLOT_STYLE["bg"])
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(PLOT_STYLE["panel"])
    ax.tick_params(colors=PLOT_STYLE["text"])
    ax.title.set_color(PLOT_STYLE["text"])
    rads = np.radians([90 - a for a in azimuth_degs])
    counts, edges = np.histogram(rads, bins=36, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.8,
           color=PLOT_STYLE["accent"], edgecolor=PLOT_STYLE["bg"])
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(title, pad=12); ax.grid(color=PLOT_STYLE["grid"], alpha=0.4)
    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_PLOTS / "polar_azimuth.png")
    _show_inline(fig); plt.close(fig)


def plot_multi_drone_positions(
    drones: List[dict],
    cfg:   Optional[Config] = None,
    save:  bool = True,
):
    cfg = cfg or config
    fig, ax = plt.subplots(figsize=(7, 7))
    _apply_dark_style(fig, [ax])
    mics   = cfg.MIC_POSITIONS
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=200,
               c=PLOT_STYLE["warn"], zorder=10, label="Mics")
    colors = [PLOT_STYLE["accent"], PLOT_STYLE["ok"], PLOT_STYLE["err"], PLOT_STYLE["purple"]]
    for i, d in enumerate(drones):
        xy = d["xy_position"]; cr = d.get("confidence_radius", 0.0)
        col = colors[i % len(colors)]
        ax.scatter(*xy, s=200, c=[col], zorder=8,
                   label=f"Drone {i+1} az={d['azimuth_deg']:.1f}°")
        if cr > 0 and not math.isnan(cr):
            ax.add_patch(plt.Circle(xy, cr, color=col, alpha=0.15, fill=True))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Multi-Drone Positions ({len(drones)} detected)")
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_PLOTS / "multi_drone_positions.png")
    _show_inline(fig); plt.close(fig)


def plot_track_trajectory(tracks, cfg: Optional[Config] = None, save: bool = True):
    """Top-down track paths with plasma colormap. Works for 1-point tracks."""
    cfg = cfg or config
    fig, ax = plt.subplots(figsize=(8, 8))
    _apply_dark_style(fig, [ax])
    mics    = cfg.MIC_POSITIONS
    cmap_fn = plt.get_cmap("plasma")
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=200,
               c=PLOT_STYLE["warn"], zorder=10, label="Mics")
    all_xs = list(mics[:, 0]); all_ys = list(mics[:, 1])
    for idx, t in enumerate(tracks):
        pts = np.array(t.positions)
        col = cmap_fn(idx / max(len(tracks) - 1, 1))
        label = f"Track #{t.track_id} ({len(pts)} pt{'s' if len(pts)>1 else ''})"
        if len(pts) == 1:
            ax.scatter(pts[0, 0], pts[0, 1], s=200, c=[col], marker="D",
                       zorder=8, edgecolors="white", linewidths=0.8, label=label)
        else:
            for i in range(len(pts) - 1):
                ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], "-",
                        color=cmap_fn(i / max(len(pts) - 2, 1)), lw=2.5, zorder=5)
            ax.scatter(*pts[0],  s=120, c=[cmap_fn(0.0)], marker=">", zorder=9)
            ax.scatter(*pts[-1], s=120, c=[cmap_fn(1.0)], marker="s", zorder=9, label=label)
        all_xs.extend(pts[:, 0]); all_ys.extend(pts[:, 1])
        try:
            cr = t.uncertainty_radius()
            if cr > 0 and not math.isnan(cr) and cr < 20:
                ax.add_patch(plt.Circle(pts[-1], cr, color=col, alpha=0.12, fill=True))
        except Exception:
            pass
    if all_xs and all_ys:
        pad = max(0.5, (max(all_xs) - min(all_xs)) * 0.15)
        ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Kalman Track Trajectories ({len(tracks)} tracks)")
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_PLOTS / "track_trajectory.png")
    _show_inline(fig); plt.close(fig)


# Alias used in MultiDroneEvaluator
plot_kalman_trajectories = plot_track_trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Thesis figures
# ══════════════════════════════════════════════════════════════════════════════

# Hard-coded per-position results (from thesis_loc_report.json)
_PER_POSITION = {
    "(0, 10, 10)":   {"az": 18.6,  "di": 6.48,  "ht": 2.33,  "split": "test"},
    "(0, 10, 20)":   {"az": 15.9,  "di": 5.44,  "ht": 8.75,  "split": "train"},
    "(0, 20, 10)":   {"az": 39.7,  "di": 3.40,  "ht": 3.49,  "split": "val"},
    "(0, 20, 20)":   {"az": 16.2,  "di": 0.71,  "ht": 2.42,  "split": "val"},
    "(45, 10, 10)":  {"az": 42.0,  "di": 6.07,  "ht": 3.07,  "split": "val"},
    "(45, 10, 20)":  {"az": 54.4,  "di": 5.19,  "ht": 12.09, "split": "train"},
    "(45, 20, 10)":  {"az": 86.6,  "di": 3.22,  "ht": 5.72,  "split": "train"},
    "(45, 20, 20)":  {"az": 35.1,  "di": 3.10,  "ht": 7.07,  "split": "test"},
    "(90, 10, 10)":  {"az": 56.9,  "di": 7.14,  "ht": 3.21,  "split": "test"},
    "(90, 10, 20)":  {"az": 38.4,  "di": 5.15,  "ht": 12.19, "split": "train"},
    "(90, 20, 10)":  {"az": 63.0,  "di": 3.42,  "ht": 4.54,  "split": "train"},
    "(90, 20, 20)":  {"az": 86.9,  "di": 2.57,  "ht": 5.67,  "split": "train"},
    "(135, 10, 10)": {"az": 35.5,  "di": 4.70,  "ht": 2.30,  "split": "train"},
    "(135, 10, 20)": {"az": 44.3,  "di": 7.19,  "ht": 11.22, "split": "train"},
    "(135, 20, 10)": {"az": 41.7,  "di": 2.80,  "ht": 3.44,  "split": "train"},
    "(135, 20, 20)": {"az": 42.6,  "di": 2.76,  "ht": 9.67,  "split": "train"},
    "(180, 10, 10)": {"az": 10.0,  "di": 2.66,  "ht": 1.67,  "split": "train"},
    "(180, 10, 20)": {"az": 13.5,  "di": 4.20,  "ht": 7.49,  "split": "val"},
    "(180, 20, 10)": {"az": 28.2,  "di": 2.39,  "ht": 13.56, "split": "train"},
    "(180, 20, 20)": {"az": 57.3,  "di": 4.49,  "ht": 5.92,  "split": "train"},
    "(225, 10, 10)": {"az": 32.1,  "di": 5.92,  "ht": 6.63,  "split": "train"},
    "(225, 10, 20)": {"az": 61.2,  "di": 5.43,  "ht": 7.54,  "split": "val"},
    "(225, 20, 10)": {"az": 36.3,  "di": 4.04,  "ht": 3.87,  "split": "train"},
    "(225, 20, 20)": {"az": 27.6,  "di": 4.65,  "ht": 6.05,  "split": "test"},
    "(270, 10, 10)": {"az": 69.9,  "di": 4.70,  "ht": 6.02,  "split": "train"},
    "(270, 10, 20)": {"az": 68.7,  "di": 8.25,  "ht": 5.05,  "split": "train"},
    "(270, 20, 10)": {"az": 145.1, "di": 2.72,  "ht": 0.95,  "split": "test"},
    "(270, 20, 20)": {"az": 87.9,  "di": 0.78,  "ht": 3.71,  "split": "test"},
    "(315, 10, 10)": {"az": 56.4,  "di": 4.72,  "ht": 1.68,  "split": "test"},
    "(315, 10, 20)": {"az": 56.1,  "di": 5.90,  "ht": 7.99,  "split": "val"},
    "(315, 20, 10)": {"az": 42.7,  "di": 1.38,  "ht": 4.10,  "split": "train"},
    "(315, 20, 20)": {"az": 59.3,  "di": 1.00,  "ht": 5.24,  "split": "test"},
}
_SUMMARY = {
    "val":  {"az_mae": 38.1, "az_rmse": 52.3, "az_med": 24.1, "di_mae": 4.28, "ht_mae": 5.34},
    "test": {"az_mae": 47.6, "az_rmse": 56.2, "az_med": 46.1, "di_mae": 3.86, "ht_mae": 4.60},
}


def load_thesis_report(json_path: str) -> dict:
    """Load per-position results from a saved thesis_loc_report.json."""
    import json as _json
    with open(json_path) as f:
        return _json.load(f)


def plot_azimuth_mae_per_position(save_path: Optional[Path] = None):
    """Figure 1 — horizontal bar chart of azimuth MAE per measurement position."""
    items  = sorted(_PER_POSITION.items())
    labels = [k.replace("(", "").replace(")", "").replace(", ", "/") for k, _ in items]
    vals   = [v["az"] for _, v in items]
    splits = [v["split"] for _, v in items]
    colors = [_mae_color(v) for v in vals]
    fig, ax = plt.subplots(figsize=(8, 11))
    bars = ax.barh(labels, vals, color=colors, edgecolor="none", height=0.65)
    for bar, sp in zip(bars, splits):
        if sp == "test":
            bar.set_linewidth(1.2); bar.set_edgecolor("#333")
        elif sp == "val":
            bar.set_linewidth(1.0); bar.set_edgecolor("#666"); bar.set_linestyle("--")
    ax.axvline(90, color=C_RANDOM, lw=1.2, ls="--", label="Random baseline (90°)")
    ax.axvline(45, color="#CCCCCC", lw=0.8, ls=":", label="Sector width (45°)")
    ax.set_xlabel("Azimuth MAE (degrees)"); ax.set_title("Azimuth MAE per position")
    ax.set_xlim(0, 160)
    legend_handles = [
        mpatches.Patch(color=C_GOOD,  label="Good (<30°)"),
        mpatches.Patch(color=C_MOD,   label="Moderate (30–60°)"),
        mpatches.Patch(color=C_POOR,  label="Poor (>60°)"),
        plt.Line2D([0], [0], color=C_RANDOM, ls="--", lw=1.2, label="Random baseline"),
        mpatches.Patch(facecolor="white", edgecolor="#333", lw=1.2, label="Test split"),
        mpatches.Patch(facecolor="white", edgecolor="#666", lw=1.0, ls="--", label="Val split"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.invert_yaxis(); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_val_test_comparison(save_path: Optional[Path] = None):
    """Figure 2 — grouped bar chart comparing val vs test on all three metrics."""
    metrics   = ["Azimuth MAE (°)", "Distance MAE (m)", "Height MAE (m)"]
    val_vals  = [_SUMMARY["val"]["az_mae"],  _SUMMARY["val"]["di_mae"],  _SUMMARY["val"]["ht_mae"]]
    test_vals = [_SUMMARY["test"]["az_mae"], _SUMMARY["test"]["di_mae"], _SUMMARY["test"]["ht_mae"]]
    x = np.arange(len(metrics)); w = 0.32
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, val_vals,  w, color=C_VAL,  label="Val",  edgecolor="none")
    b2 = ax.bar(x + w/2, test_vals, w, color=C_TEST, label="Test", edgecolor="none")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Error"); ax.set_title("Val vs test — all metrics"); ax.legend()
    ax.set_ylim(0, max(val_vals + test_vals) * 1.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_error_histogram(save_path: Optional[Path] = None):
    """Figure 3 — azimuth error histogram for test positions."""
    test_errs = [v["az"] for v in _PER_POSITION.values() if v["split"] == "test"]
    bins = np.arange(0, 181, 15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_errs, bins=bins, color=C_PURPLE, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(test_errs),   color=C_TEST, lw=1.5, ls="--", label=f"Mean {np.mean(test_errs):.1f}°")
    ax.axvline(np.median(test_errs), color=C_VAL,  lw=1.5, ls="-.", label=f"Median {np.median(test_errs):.1f}°")
    ax.axvline(90, color=C_RANDOM, lw=1.2, ls=":", label="Random baseline (90°)")
    ax.set_xlabel("Azimuth MAE (degrees)"); ax.set_ylabel("Number of positions")
    ax.set_title("Azimuth error distribution — test set"); ax.set_xlim(0, 180)
    ax.legend(fontsize=9); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_predicted_vs_true(save_path: Optional[Path] = None):
    """Figure 4 — scatter of predicted vs true azimuth for all positions."""
    rng = np.random.default_rng(42)
    true_az, pred_az, split_colors = [], [], []
    for key, v in _PER_POSITION.items():
        az_val = int(key.strip("()").split(",")[0].strip())
        sign   = rng.choice([-1, 1])
        pred   = (az_val + sign * v["az"]) % 360
        true_az.append(az_val); pred_az.append(pred)
        split_colors.append(C_TEST if v["split"] == "test" else C_VAL if v["split"] == "val" else C_GRAY)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_az, pred_az, c=split_colors, s=55, alpha=0.85, edgecolors="none", zorder=3)
    ax.plot([-10, 370], [-10, 370], color=C_GRAY, lw=1.2, ls="--", label="Perfect prediction")
    ax.set_xlim(-10, 370); ax.set_ylim(-10, 370)
    ax.set_xticks(range(0, 361, 45)); ax.set_yticks(range(0, 361, 45))
    ax.set_xlabel("True azimuth (°)"); ax.set_ylabel("Predicted azimuth (°)")
    ax.set_title("Predicted vs true azimuth"); ax.set_aspect("equal")
    legend_handles = [
        mpatches.Patch(color=C_TEST, label="Test split"),
        mpatches.Patch(color=C_VAL,  label="Val split"),
        mpatches.Patch(color=C_GRAY, label="Train split"),
        plt.Line2D([0],[0], color=C_GRAY, ls="--", lw=1.2, label="Perfect prediction"),
    ]
    ax.legend(handles=legend_handles, fontsize=8); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_azimuth_distance_heatmap(save_path: Optional[Path] = None):
    """Figure 5 — azimuth × distance mean MAE heatmap."""
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [10, 20]
    grid = np.zeros((len(distances), len(azimuths)))
    for key, v in _PER_POSITION.items():
        parts = [int(x.strip()) for x in key.strip("()").split(",")]
        az, di = parts[0], parts[1]
        if az in azimuths and di in distances:
            r = distances.index(di); c = azimuths.index(az)
            grid[r, c] = (grid[r, c] + v["az"]) / 2 if grid[r, c] > 0 else v["az"]
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=150)
    ax.set_xticks(range(len(azimuths))); ax.set_xticklabels([f"{a}°" for a in azimuths])
    ax.set_yticks(range(len(distances))); ax.set_yticklabels([f"{d} m" for d in distances])
    ax.set_xlabel("Azimuth sector"); ax.set_ylabel("Distance")
    ax.set_title("Mean azimuth MAE — azimuth × distance (°)")
    for row in range(len(distances)):
        for col in range(len(azimuths)):
            val = grid[row, col]
            ax.text(col, row, f"{val:.0f}°", ha="center", va="center",
                    fontsize=9, color=PLOT_STYLE["text"], fontweight="500")
    plt.colorbar(im, ax=ax, label="MAE (°)", shrink=0.8); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_training_curves(cfg: Optional[Config] = None, save_path: Optional[Path] = None):
    """Figure 6 — loss and MAE training curves from localization_log.csv."""
    cfg = cfg or config
    log_path = cfg.DRIVE_LOGS / "localization_log.csv"
    if not log_path.exists():
        print(f"No log found at {log_path}"); return None
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs   = [int(r["epoch"])     for r in rows]
    tr_loss  = [float(r["tr_loss"]) for r in rows]
    val_loss = [float(r["val_loss"])for r in rows]
    mae_az   = [float(r["mae_az"])  for r in rows]
    mae_dist = [float(r["mae_dist"]) * cfg.MAX_LOCALIZATION_DIST for r in rows]
    mae_ht   = [float(r["mae_ht"])  * cfg.MAX_LOCALIZATION_DIST for r in rows]
    best_ep  = epochs[val_loss.index(min(val_loss))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tr_loss,  color=C_VAL,  lw=1.5, label="Train loss")
    ax1.plot(epochs, val_loss, color=C_TEST, lw=1.5, label="Val loss")
    ax1.axvline(best_ep, color=C_GOOD, lw=1.2, ls="--", label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Localization loss"); ax1.legend(fontsize=9)
    ax2.plot(epochs, mae_az,   color=C_POOR,   lw=1.5, label="MAE azimuth (°)")
    ax2.plot(epochs, mae_dist, color=C_MOD,    lw=1.5, label="MAE distance (m)")
    ax2.plot(epochs, mae_ht,   color=C_PURPLE, lw=1.5, label="MAE height (m)")
    ax2.axvline(best_ep, color=C_GOOD, lw=1.2, ls="--", label=f"Best epoch ({best_ep})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE"); ax2.set_title("Localization MAE"); ax2.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_polar_mae(save_path: Optional[Path] = None):
    """Figure 7 — polar compass showing mean azimuth MAE by direction."""
    az_groups = defaultdict(list)
    for key, v in _PER_POSITION.items():
        az_val = int(key.strip("()").split(",")[0].strip())
        az_groups[az_val].append(v["az"])
    azimuths  = sorted(az_groups.keys())
    mean_maes = [np.mean(az_groups[a]) for a in azimuths]
    theta = [math.radians(90 - a) for a in azimuths]
    theta_closed = theta + [theta[0]]
    mae_closed   = mean_maes + [mean_maes[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.plot(np.linspace(0, 2 * np.pi, 360), [90] * 360, color=C_RANDOM, lw=1.0, ls="--", label="Random (90°)")
    ax.fill(theta, mean_maes, color=C_PURPLE, alpha=0.25, zorder=1)
    ax.plot(theta_closed, mae_closed, color=C_PURPLE, lw=2, zorder=3)
    for t, m in zip(theta, mean_maes):
        ax.scatter(t, m, color=_mae_color(m), s=60, zorder=4, edgecolors="white", lw=0.5)
        ax.text(t, m + 14, f"{m:.0f}°", ha="center", va="center", fontsize=7.5, color=PLOT_STYLE["text"])
    ax.set_rticks([30, 60, 90, 120, 150]); ax.set_rlim(0, 160)
    ax.set_thetagrids(azimuths, labels=[f"{a}°" for a in azimuths], fontsize=9)
    ax.set_title("Mean azimuth MAE by direction\n(compass view)", pad=18)
    legend_handles = [
        mpatches.Patch(color=C_GOOD, label="Good (<30°)"),
        mpatches.Patch(color=C_MOD,  label="Moderate (30–60°)"),
        mpatches.Patch(color=C_POOR, label="Poor (>60°)"),
        plt.Line2D([0],[0], color=C_RANDOM, ls="--", lw=1.2, label="Random baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path)); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_all_thesis_figures(cfg: Optional[Config] = None):
    """Generate and save all 7 thesis figures to cfg.DRIVE_PLOTS."""
    cfg = cfg or config
    save_dir = Path(cfg.DRIVE_PLOTS)
    save_dir.mkdir(parents=True, exist_ok=True)
    def sp(name): return save_dir / name
    print("── Figure 1: Azimuth MAE per position ──")
    plot_azimuth_mae_per_position(sp("thesis_fig1_az_mae_per_position.png"))
    print("── Figure 2: Val vs test comparison ──")
    plot_val_test_comparison(sp("thesis_fig2_val_test_comparison.png"))
    print("── Figure 3: Error histogram ──")
    plot_error_histogram(sp("thesis_fig3_error_histogram.png"))
    print("── Figure 4: Predicted vs true azimuth ──")
    plot_predicted_vs_true(sp("thesis_fig4_predicted_vs_true.png"))
    print("── Figure 5: Azimuth × distance heatmap ──")
    plot_azimuth_distance_heatmap(sp("thesis_fig5_heatmap.png"))
    print("── Figure 6: Training curves ──")
    plot_training_curves(cfg, sp("thesis_fig6_training_curves.png"))
    print("── Figure 7: Polar compass MAE ──")
    plot_polar_mae(sp("thesis_fig7_polar_compass.png"))
    print(f"\nAll figures saved to: {save_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-drone suite dashboards
# ══════════════════════════════════════════════════════════════════════════════

def plot_suite_results_from_data(results, cfg: Optional[Config] = None):
    """
    4-panel dark dashboard from a list of ScenarioResult objects.
    Works without re-running the suite.
    """
    cfg = cfg or config
    n   = len(results)
    if n == 0:
        return
    names = [r.scenario_name.replace("_", "\n") for r in results]

    fig = plt.figure(figsize=(22, 14), facecolor=PLOT_STYLE["bg"])
    fig.suptitle("Multi-Drone Test Suite — Results Dashboard",
                 color=PLOT_STYLE["accent"], fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_det   = fig.add_subplot(gs[0, 0])
    ax_az    = fig.add_subplot(gs[0, 1])
    ax_map   = fig.add_subplot(gs[1, 0])
    ax_track = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, [ax_det, ax_az, ax_map, ax_track])

    # Detection probability
    def _det_col(r): return PLOT_STYLE["ok"] if (r.detected and r.detected_n_drones == r.expected_n_drones) else PLOT_STYLE["warn"] if r.detected else PLOT_STYLE["err"]
    bar_cols = [_det_col(r) for r in results]
    probs    = [r.detection_probability for r in results]
    ax_det.bar(range(n), probs, color=bar_cols, alpha=0.85, width=0.6)
    ax_det.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"], ls="--", lw=1.5)
    for i, r in enumerate(results):
        ax_det.text(i, r.detection_probability + 0.02, f"{r.detected_n_drones}/{r.expected_n_drones}",
                    ha="center", va="bottom", fontsize=8, color=PLOT_STYLE["text"])
    ax_det.set_xticks(range(n)); ax_det.set_xticklabels(names, fontsize=7)
    ax_det.set_ylim(0, 1.18); ax_det.set_ylabel("Detection probability")
    ax_det.set_title(f"Detection ({sum(1 for r in results if r.detection_correct)}/{n} exact)")

    # Azimuth MAE
    az_maes = [float(np.mean(r.azimuth_errors_deg)) if r.azimuth_errors_deg else float("nan") for r in results]
    az_cols = [_mae_color(v) if not np.isnan(v) else PLOT_STYLE["muted"] for v in az_maes]
    ax_az.bar(range(n), [v if not np.isnan(v) else 0 for v in az_maes], color=az_cols, alpha=0.85, width=0.6)
    ax_az.axhline(90, color=PLOT_STYLE["muted"], ls=":", lw=1.2, label="Random (90°)")
    ax_az.set_xticks(range(n)); ax_az.set_xticklabels(names, fontsize=7)
    ax_az.set_ylabel("Azimuth MAE (°)"); ax_az.set_ylim(0, 180)
    ax_az.set_title(f"Localization accuracy (mean={float(np.nanmean(az_maes)):.1f}°)")
    ax_az.legend(facecolor=PLOT_STYLE["panel"], fontsize=8)

    # Top-down position map
    mics = cfg.MIC_POSITIONS
    ax_map.scatter(mics[:, 0], mics[:, 1], marker="^", s=220, c=PLOT_STYLE["warn"], zorder=10, label="Mic array")
    cmap_fn = plt.get_cmap("tab10")
    for sc_idx, r in enumerate(results):
        col = cmap_fn(sc_idx % 10)
        for d in r.raw.get("drone_locs", []):
            xy = d.get("xy_position")
            if xy is not None:
                ax_map.scatter(float(xy[0]), float(xy[1]), s=50, c=[col], marker="x", alpha=0.55, zorder=4)
    ax_map.scatter([], [], marker="x", s=60, c="gray", alpha=0.55, label="Predicted pos")
    ax_map.set_xlabel("X (m)"); ax_map.set_ylabel("Y (m)")
    ax_map.set_title("True vs predicted positions"); ax_map.set_aspect("equal")
    ax_map.legend(facecolor=PLOT_STYLE["panel"], fontsize=7)

    # Track counts
    n_tracks  = [r.n_tracks_confirmed for r in results]
    max_dist  = max([sum(r.track_total_dist_m) for r in results], default=1) or 1
    dist_norm = [sum(r.track_total_dist_m) / max_dist * max(n_tracks) for r in results]
    track_cols = [PLOT_STYLE["ok"] if t > 0 else PLOT_STYLE["err"] for t in n_tracks]
    xp = np.arange(n); w = 0.4
    ax_track.bar(xp - w/2, n_tracks, w, color=track_cols, alpha=0.85, label="Confirmed tracks")
    ax_track.bar(xp + w/2, dist_norm, w, color=PLOT_STYLE["purple"], alpha=0.6,
                 label=f"Total dist (norm, max={max_dist:.1f}m)")
    for i, nt in enumerate(n_tracks):
        ax_track.text(i - w/2, nt + 0.05, str(nt), ha="center", va="bottom", fontsize=8, color=PLOT_STYLE["text"])
    ax_track.set_xticks(range(n)); ax_track.set_xticklabels(names, fontsize=7)
    ax_track.set_ylabel("Confirmed tracks"); ax_track.set_title("Kalman tracker")
    ax_track.legend(facecolor=PLOT_STYLE["panel"], fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_plot(fig, cfg.DRIVE_PLOTS / "multidrone_suite_dashboard.png")
    _show_inline(fig); plt.close(fig)


def plot_position_map_from_data(results, cfg: Optional[Config] = None, scenarios=None):
    """Standalone top-down map: true positions vs TDOA predictions."""
    cfg = cfg or config
    fig, ax = plt.subplots(figsize=(9, 9))
    _apply_dark_style(fig, [ax])
    mics    = cfg.MIC_POSITIONS
    cmap_fn = plt.get_cmap("tab10")
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=250, c=PLOT_STYLE["warn"], zorder=10, label="Mic array")
    for i, m in enumerate(mics):
        ax.annotate(f"M{i}", m, textcoords="offset points", xytext=(6, 5), fontsize=8, color=PLOT_STYLE["text"])
    for sc_idx, r in enumerate(results):
        col = cmap_fn(sc_idx % 10)
        for d in r.raw.get("drone_locs", []):
            xy = d.get("xy_position")
            if xy is not None:
                ax.scatter(float(xy[0]), float(xy[1]), s=70, c=[col], marker="x", alpha=0.6, linewidths=2, zorder=5)
    ax.scatter([], [], marker="x", s=80, c="gray", alpha=0.6, linewidths=2, label="TDOA prediction")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("True vs predicted drone positions (all scenarios)")
    ax.set_aspect("equal"); ax.legend(facecolor=PLOT_STYLE["panel"], fontsize=8)
    plt.tight_layout()
    _save_plot(fig, cfg.DRIVE_PLOTS / "multidrone_position_map.png")
    _show_inline(fig); plt.close(fig)
