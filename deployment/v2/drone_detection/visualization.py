# -*- coding: utf-8 -*-
"""
visualization.py
────────────────
All plotting functions for the drone detection & localization pipeline.

Contents
────────
Light-themed analysis dashboard     _plot_analysis_report()
External detection scores plot      _plot_external_detection_scores()
Training curve plots                plot_training_logs()
Confusion matrix                    plot_confusion_matrix_styled()
Localization scatter                plot_localization_scatter()
Polar azimuth compass               plot_polar_azimuth()
Track trajectory                    plot_track_trajectory()
Multi-drone positions               plot_multi_drone_positions()
Kalman trajectories (1-pt safe)     plot_kalman_trajectories()

Figures (7)
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
import warnings
from pathlib import Path
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from scipy import signal

from .config import Config, config

matplotlib.use("Agg")

# ── Global font sizes (increased for thesis PDF) ───────────────────────────
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})

# ── Colour palette (light / print-friendly) ────────────────────────────────
PLOT_STYLE = {
    "bg":        "#ffffff",   # white page background
    "panel":     "#f8f9fa",   # very light grey panel fill
    "panel_alt": "#e9ecef",   # slightly darker for legend boxes
    "accent":    "#1565c0",   # dark blue (was sky-blue)
    "warn":      "#e65100",   # dark orange (was amber)
    "ok":        "#2e7d32",   # dark green (was bright green)
    "err":       "#c62828",   # dark red (was light red)
    "grid":      "#b0bec5",   # medium grey grid lines
    "text":      "#212121",   # near-black text
    "text_soft": "#37474f",   # soft dark text
    "muted":     "#546e7a",   # muted blue-grey
    "spine":     "#90a4ae",   # light grey spines
    "purple":    "#6a1b9a",   # dark purple (was lavender)
}

MIC_COLORS = ["#1565c0", "#2e7d32", "#6a1b9a"]   # ch0, ch1, ch2
PAIR_COLORS = {"01": "#e65100", "02": "#1565c0", "12": "#2e7d32"}

# Thesis palette (unchanged - already print-safe colours)
C_GOOD   = "#1D9E75"
C_MOD    = "#BA7517"
C_POOR   = "#D85A30"
C_VAL    = "#378ADD"
C_TEST   = "#D85A30"
C_PURPLE = "#7F77DD"
C_GRAY   = "#888780"
C_RANDOM = "#AAAAAA"


# Internal helper functions for consistent styling across plots
def _apply_style(fig, axes_flat):
    fig.patch.set_facecolor(PLOT_STYLE["bg"])
    for ax in axes_flat:
        if ax is None:
            continue
        ax.set_facecolor(PLOT_STYLE["panel"])
        ax.tick_params(colors=PLOT_STYLE["text"], labelcolor=PLOT_STYLE["text"], labelsize=11)
        ax.xaxis.label.set_color(PLOT_STYLE["text"])
        ax.yaxis.label.set_color(PLOT_STYLE["text"])
        ax.title.set_color(PLOT_STYLE["text"])
        ax.title.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color(PLOT_STYLE["spine"])
            spine.set_linewidth(0.9)
        ax.grid(color=PLOT_STYLE["grid"], alpha=0.45, linewidth=0.7)


def _save(fig, path: Optional[Path], dpi: int = 200):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"💾 Saved: {path}")


def _finite(arr):
    return np.array([v for v in arr if v is not None and not math.isnan(float(v))],
                    dtype=float)


def _cc_lags(a: np.ndarray, b: np.ndarray, sr: int,
             max_lag_ms: float = 10.0) -> tuple:
    """
    Normalised cross-correlation between two mono arrays.
    Returns (lags_ms, cc_normalised).
    """
    n = len(a)
    # Correlate on the first 3 s to keep it fast
    clip = min(n, int(sr * 3.0))
    x = a[:clip].astype(np.float64)
    y = b[:clip].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    cc = signal.correlate(x, y, mode="full")
    denom = (np.std(x) * np.std(y) * clip) + 1e-10
    cc /= denom
    lag_samples = signal.correlation_lags(len(x), len(y), mode="full")
    lag_ms      = lag_samples / sr * 1000.0
    mask        = np.abs(lag_ms) <= max_lag_ms
    return lag_ms[mask], cc[mask]

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
    """Apply light/print-friendly style (name kept for compatibility)."""
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
            labelsize=12,
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
        ax.grid(color=PLOT_STYLE["grid"], alpha=0.5, linewidth=0.8)

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


def _save_plot(fig, path: Optional[Path], dpi: int = 200):
    """Save at higher DPI for crisp figures."""
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"💾 Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Training curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_logs(cfg: Optional[Config] = None, save: bool = True):
    """Light-themed training curves for detection and localization."""
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
        ax.set_xlabel("Epoch"); ax.set_ylabel("Focal loss"); ax.set_title("Detection - Loss")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, tr_acc,  "-o", color=PLOT_STYLE["ok"],   ms=4, label="Train acc %")
        ax.plot(epochs, val_acc, "-s", color=PLOT_STYLE["warn"], ms=4, label="Val acc %")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Detection - Accuracy")
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
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss"); ax.set_title("Localization - Loss")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(epochs, mae_az,   "-o", color=PLOT_STYLE["err"],    ms=4, label="MAE az (°)")
        ax.plot(epochs, mae_dist, "-s", color=PLOT_STYLE["purple"], ms=4, label="MAE dist (m)")
        ax.plot(epochs, mae_ht,   "-^", color=PLOT_STYLE["ok"],     ms=4, label="MAE ht (m)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.set_title("Localization - MAE")
        leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
        _style_legend(leg)

    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_LOGS / "training_curves.png")
    _show_inline(fig); plt.close(fig)

def plot_detection_training_curves(cfg: Optional[Config] = None, save: bool = True, file_name: str = ""):
    """
    Plot separate training curves for detection model.
    
    Shows:
        - Training loss (Focal Loss) over epochs
        - Training and validation accuracy over epochs
        - Optional: Learning rate schedule
    """
    cfg = cfg or config
    det_csv = cfg.DRIVE_LOGS / "detection_log.csv"
    
    if not det_csv.exists():
        print(f"❌ Detection log not found: {det_csv}")
        return None
    
    # Read data
    with open(det_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    
    epochs = np.array([int(r["epoch"]) for r in rows])
    tr_loss = np.array([float(r["tr_loss"]) for r in rows])
    tr_acc = np.array([float(r["tr_acc"]) for r in rows])
    val_acc = np.array([float(r["val_acc"]) for r in rows])
    
    # Find best validation accuracy
    best_epoch = np.argmax(val_acc)
    best_val_acc = val_acc[best_epoch]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"Detection Model Training Curves - {file_name if file_name else 'Detection Task'}",
                 fontsize=16, color=PLOT_STYLE["accent"], fontweight="bold", y=0.98)
    
    # Plot 1: Loss
    ax1.plot(epochs, tr_loss, '-o', color=PLOT_STYLE["accent"], 
             linewidth=2, markersize=5, label='Training Loss')
    ax1.axvline(best_epoch, color=C_GOOD, linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Focal Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    
    # Annotate final loss
    ax1.annotate(f'Final: {tr_loss[-1]:.4f}',
                 xy=(epochs[-1], tr_loss[-1]),
                 xytext=(epochs[-1] + 0.5, tr_loss[-1]),
                 fontsize=10, color=PLOT_STYLE["text"])
    
    # Plot 2: Accuracy
    ax2.plot(epochs, tr_acc, '-o', color=PLOT_STYLE["ok"], 
             linewidth=2, markersize=5, label='Training Accuracy')
    ax2.plot(epochs, val_acc, '-s', color=PLOT_STYLE["warn"], 
             linewidth=2, markersize=5, label='Validation Accuracy')
    ax2.axvline(best_epoch, color=C_GOOD, linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=11)
    
    # Add text box with metrics
    metrics_text = (f"Best Validation: {best_val_acc:.2f}%\n"
                    f"Final Validation: {val_acc[-1]:.2f}%\n"
                    f"Final Training: {tr_acc[-1]:.2f}%")
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=PLOT_STYLE["panel_alt"], alpha=0.8))
    
    # Apply style
    for ax in [ax1, ax2]:
        ax.set_facecolor(PLOT_STYLE["panel"])
        ax.tick_params(colors=PLOT_STYLE["text"], labelcolor=PLOT_STYLE["text"])
        for spine in ax.spines.values():
            spine.set_color(PLOT_STYLE["spine"])
    
    plt.tight_layout()
    
    if save:
        save_path = cfg.DRIVE_PLOTS / f"detection_training_curves_{file_name}.png"
        _save_plot(fig, save_path, dpi=200)
    
    _show_inline(fig)
    plt.close(fig)
    return fig


def plot_localization_training_curves(cfg: Optional[Config] = None, save: bool = True, file_name: str = ""):
    """
    Plot separate training curves for localization model.
    
    Shows:
        - Training and validation loss (MSE) over epochs
        - MAE for azimuth, distance, and height over epochs
        - Optional: Learning rate schedule
    """
    cfg = cfg or config
    loc_csv = cfg.DRIVE_LOGS / "localization_log.csv"
    
    if not loc_csv.exists():
        print(f"❌ Localization log not found: {loc_csv}")
        return None
    
    # Read data
    with open(loc_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    
    epochs = np.array([int(r["epoch"]) for r in rows])
    tr_loss = np.array([float(r["tr_loss"]) for r in rows])
    val_loss = np.array([float(r["val_loss"]) for r in rows])
    
    # Get MAE values and convert distance/height to real units
    mae_az = np.array([float(r["mae_az"]) for r in rows])
    mae_dist = np.array([float(r["mae_dist"]) * cfg.MAX_LOCALIZATION_DIST for r in rows])
    mae_ht = np.array([float(r["mae_ht"]) * cfg.MAX_LOCALIZATION_DIST for r in rows])
    
    # Find best epoch based on validation loss
    best_epoch = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"Localization Model Training Curves - {file_name if file_name else 'Localization Task'}",
                 fontsize=16, color=PLOT_STYLE["accent"], fontweight="bold", y=0.98)
    
    # Plot 1: Loss curves
    ax1.plot(epochs, tr_loss, '-o', color=PLOT_STYLE["accent"], 
             linewidth=2, markersize=5, label='Training Loss')
    ax1.plot(epochs, val_loss, '-s', color=PLOT_STYLE["warn"], 
             linewidth=2, markersize=5, label='Validation Loss')
    ax1.axvline(best_epoch, color=C_GOOD, linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=11)
    
    # Annotate best validation loss
    ax1.annotate(f'Best val loss: {best_val_loss:.4f}',
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch + 1, best_val_loss),
                 fontsize=10, color=C_GOOD,
                 arrowprops=dict(arrowstyle='->', color=C_GOOD, lw=1))
    
    # Plot 2: MAE curves
    ax2.plot(epochs, mae_az, '-o', color=PLOT_STYLE["err"], 
             linewidth=2, markersize=5, label='MAE Azimuth (°)', alpha=0.9)
    ax2.plot(epochs, mae_dist, '-s', color=PLOT_STYLE["purple"], 
             linewidth=2, markersize=5, label='MAE Distance (m)', alpha=0.9)
    ax2.plot(epochs, mae_ht, '-^', color=PLOT_STYLE["ok"], 
             linewidth=2, markersize=5, label='MAE Height (m)', alpha=0.9)
    ax2.axvline(best_epoch, color=C_GOOD, linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Localization Errors (MAE)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add horizontal reference lines for "good" performance
    ax2.axhline(30, color=C_GOOD, linestyle=':', linewidth=1, alpha=0.5, 
                label='Good azimuth (<30°)')
    ax2.axhline(60, color=C_MOD, linestyle=':', linewidth=1, alpha=0.5, 
                label='Moderate azimuth (<60°)')
    
    # Add text box with final metrics
    metrics_text = (f"Final Metrics (epoch {epochs[-1]}):\n"
                    f"├─ Azimuth MAE: {mae_az[-1]:.1f}°\n"
                    f"├─ Distance MAE: {mae_dist[-1]:.1f} m\n"
                    f"└─ Height MAE: {mae_ht[-1]:.1f} m\n\n"
                    f"Best Validation:\n"
                    f"├─ Epoch: {best_epoch}\n"
                    f"├─ Azimuth MAE: {mae_az[best_epoch]:.1f}°\n"
                    f"├─ Distance MAE: {mae_dist[best_epoch]:.1f} m\n"
                    f"└─ Height MAE: {mae_ht[best_epoch]:.1f} m")
    
    ax2.text(0.98, 0.98, metrics_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor=PLOT_STYLE["panel_alt"], alpha=0.9))
    
    # Apply style
    for ax in [ax1, ax2]:
        ax.set_facecolor(PLOT_STYLE["panel"])
        ax.tick_params(colors=PLOT_STYLE["text"], labelcolor=PLOT_STYLE["text"])
        for spine in ax.spines.values():
            spine.set_color(PLOT_STYLE["spine"])
    
    plt.tight_layout()
    
    if save:
        save_path = cfg.DRIVE_PLOTS / f"localization_training_curves_{file_name}.png"
        _save_plot(fig, save_path, dpi=200)
    
    _show_inline(fig)
    plt.close(fig)
    return fig


def plot_training_curves_separate(cfg: Optional[Config] = None, save: bool = True, file_name: str = ""):
    """
    Generate both detection and localization training curves as separate figures.
    This is a convenience function that calls the two separate plotting functions.
    """
    print("=" * 60)
    print("Generating Detection Training Curves")
    print("=" * 60)
    plot_detection_training_curves(cfg, save, f"{file_name}_detection" if file_name else "detection")
    
    print("\n" + "=" * 60)
    print("Generating Localization Training Curves")
    print("=" * 60)
    plot_localization_training_curves(cfg, save, f"{file_name}_localization" if file_name else "localization")
    
    print("\n✅ Both training curve figures generated successfully!")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis dashboard (6-panel light)
# ══════════════════════════════════════════════════════════════════════════════
def _panel3_localisation(ax, segments, cfg, PLOT_STYLE):
    """
    Scatter plot of estimated (x, y) drone positions per segment.
 
    Colour encodes hybrid detection confidence (plasma colormap).
    Mic array triangle is overlaid for spatial reference.
    Falls back to azimuth + distance twin-axis chart when xy_position is absent.
    """
    locs = [s for s in segments if s.get("loc") is not None]
 
    if not locs:
        ax.text(0.5, 0.5, "No localisation data",
                ha="center", va="center",
                color=PLOT_STYLE["muted"], transform=ax.transAxes, fontsize=10)
        ax.set_title("Localisation (x, y)", color=PLOT_STYLE["text"])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        return
 
    # Check whether xy_position is available (it always should be after v18 patch)
    first_xy = locs[0]["loc"].get("xy_position")
    has_xy   = first_xy is not None and len(np.asarray(first_xy)) >= 2
 
    if has_xy:
        # ── Scatter (x, y) colour-coded by confidence ─────────────────────
        xs    = [float(np.asarray(s["loc"]["xy_position"])[0]) for s in locs]
        ys    = [float(np.asarray(s["loc"]["xy_position"])[1]) for s in locs]
        confs = [float(s["prob"]) for s in locs]
        ts    = [float(s["t_start"]) for s in locs]
 
        sc = ax.scatter(xs, ys, c=confs, cmap="plasma", vmin=0, vmax=1,
                        s=55, zorder=4, edgecolors=PLOT_STYLE["bg"], linewidths=0.4)
 
        # Annotate a few points with their segment time so the user can
        # see which part of the recording each estimate came from.
        for x, y, t in zip(xs, ys, ts):
            ax.annotate(f"{t:.0f}s", (x, y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=PLOT_STYLE["muted"], zorder=5)
 
        # Mic array triangle (orange triangles, same as track trajectory plot)
        mic_positions = np.asarray(cfg.MIC_POSITIONS, dtype=float)
        ax.scatter(mic_positions[:, 0], mic_positions[:, 1],
                   marker="^", color=PLOT_STYLE["warn"],
                   s=70, zorder=6, label="Mics")
 
        # Array centre cross
        centre = np.asarray(cfg.ARRAY_CENTER, dtype=float)
        ax.plot(centre[0], centre[1], "+",
                color=PLOT_STYLE["text"], ms=8, mew=1.2, zorder=7)
 
        # Colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        try:
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="4%", pad=0.05)
            cbar    = ax.figure.colorbar(sc, cax=cax)
            cbar.set_label("Confidence", color=PLOT_STYLE["text"], fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=PLOT_STYLE["text"])
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PLOT_STYLE["text"])
            cbar.outline.set_edgecolor(PLOT_STYLE["spine"])
        except Exception:
            pass  # colorbar is cosmetic - never crash the dashboard
 
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(facecolor=PLOT_STYLE["panel_alt"],
                  edgecolor=PLOT_STYLE["spine"], fontsize=7)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("Localisation (x, y)", color=PLOT_STYLE["text"])
 
    else:
        # ── Fallback: azimuth bars + distance twin axis ────────────────────
        w         = cfg.TARGET_DURATION * 0.8
        az_vals   = [s["loc"]["azimuth_deg"] for s in locs]
        dist_vals = [s["loc"]["distance_m"]   for s in locs]
        t_locs    = [s["t_start"] for s in locs]
 
        ax.bar(t_locs, az_vals, width=w, color=PLOT_STYLE["accent"],
               alpha=0.7, label="Azimuth (°)")
        ax.axhline(0, color=PLOT_STYLE["spine"], lw=0.8, ls="--")
        ax.set_ylim(-185, 185)
        ax.set_ylabel("Azimuth (°)", color=PLOT_STYLE["text"])
 
        ax2_loc = ax.twinx()
        ax2_loc.plot(t_locs, dist_vals, "D--", color=PLOT_STYLE["warn"],
                     ms=5, lw=1.5, label="Distance (m)")
        ax2_loc.set_ylabel("Distance (m)", color=PLOT_STYLE["warn"])
        ax2_loc.tick_params(axis="y", colors=PLOT_STYLE["warn"])
        for spine in ax2_loc.spines.values():
            spine.set_color(PLOT_STYLE["spine"])
 
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_loc.get_legend_handles_labels()
        leg = ax.legend(lines1 + lines2, labels1 + labels2,
                        facecolor=PLOT_STYLE["panel_alt"],
                        edgecolor=PLOT_STYLE["spine"],
                        loc="upper right", fontsize=8)
        from visualization import _style_legend   # noqa - adjust import path
        _style_legend(leg)
 
        ax.set_xlabel("Time (s)")
        ax.set_title("Localisation (Azimuth + Distance)", color=PLOT_STYLE["text"])
 
    ax.grid(True, color=PLOT_STYLE["grid"], lw=0.4, alpha=0.6)

def _panel4_polar_compass(fig, gs, axes, segments, cfg, PLOT_STYLE):
    """
    Polar histogram of detected azimuths. 
    Uses all segments that have a loc dict (not just det["detected"]==True)
    so the compass fills in whenever a position estimate exists.
    """
    axes[4].remove()
    ax_pol = fig.add_subplot(gs[1, 1], projection="polar")
    ax_pol.set_facecolor(PLOT_STYLE["panel"])
    ax_pol.tick_params(colors=PLOT_STYLE["text"])
 
    # Collect azimuths from all segments that were localised
    az_degs = [
        s["loc"]["azimuth_deg"]
        for s in segments
        if s.get("loc") is not None
    ]
 
    if az_degs:
        # CORRECTED sign: -a rotates clockwise from North
        rads   = np.radians([-a for a in az_degs])
        counts, edges = np.histogram(rads, bins=36, range=(-np.pi, np.pi))
        centers       = 0.5 * (edges[:-1] + edges[1:])
        width         = edges[1] - edges[0]
 
        # Colour bars by bin count so the dominant direction pops
        max_c  = max(counts) if counts.max() > 0 else 1
        colors = plt.cm.plasma(counts / max_c)
 
        ax_pol.bar(centers, counts, width=width, alpha=0.85,
                   color=colors, edgecolor=PLOT_STYLE["bg"], linewidth=0.3)
 
        # Annotate the dominant direction
        dom_idx = int(np.argmax(counts))
        dom_deg = float(np.degrees(-centers[dom_idx])) % 360
        ax_pol.annotate(
            f"{dom_deg:.0f}°",
            xy=(centers[dom_idx], counts[dom_idx]),
            xytext=(centers[dom_idx], counts[dom_idx] * 1.25),
            ha="center", va="center",
            fontsize=7, color=PLOT_STYLE["accent"],
        )
    else:
        # Draw a faint placeholder ring so the panel isn't empty
        theta = np.linspace(0, 2 * np.pi, 60)
        ax_pol.plot(theta, np.ones_like(theta) * 0.05,
                    color=PLOT_STYLE["muted"], lw=0.8, ls="--")
        ax_pol.text(0, 0, "no data", ha="center", va="center",
                    color=PLOT_STYLE["muted"], fontsize=8,
                    transform=ax_pol.transData)
 
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_title("Azimuth (N-up)", color=PLOT_STYLE["accent"], pad=12)
    ax_pol.grid(color=PLOT_STYLE["grid"], alpha=0.5)
 
    return ax_pol

def _plot_analysis_report(segments, confirmed, cfg, title: str, file_name: str = ""):
    """
    Six-panel light-themed analysis dashboard  (v18 patch).
 
    Panels:
      [0] Waveform + RMS
      [1] Mel spectrogram
      [2] Detection timeline  (CNN / Heuristic / Hybrid / threshold)
      [3] Localisation (x, y) scatter 
      [4] Polar azimuth compass         
      [5] Detection score gauge
    """
    import itertools
    import math
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from pathlib import Path
 
    # ── figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"Drone Analysis - {title}", fontsize=17,
                 color=PLOT_STYLE["accent"], fontweight="bold", y=0.98)
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    _apply_dark_style(fig, axes)
    ts_list = [s["t_start"] for s in segments]
 
    # ── [0] Waveform + RMS ────────────────────────────────────────────────────
    ax = axes[0]
    if any("waveform" in s for s in segments):
        wave = list(itertools.chain.from_iterable(
            s.get("waveform", []) for s in segments))
        t_w  = np.linspace(
            0,
            max(ts_list) + cfg.TARGET_DURATION if ts_list else 3.0,
            len(wave),
        )
        ax.plot(t_w, wave, color=PLOT_STYLE["accent"], lw=0.6, alpha=0.7)
    rms_vals = [s.get("rms_db", -60) for s in segments]
    ax2 = ax.twinx()
    ax2.plot(ts_list, rms_vals, "o-", color=PLOT_STYLE["warn"],
             ms=4, lw=1.5, label="RMS dB")
    ax2.tick_params(axis="y", colors=PLOT_STYLE["text"],
                    labelcolor=PLOT_STYLE["text"])
    ax2.yaxis.label.set_color(PLOT_STYLE["text"])
    ax2.set_ylabel("RMS (dB)", color=PLOT_STYLE["text"])
    for spine in ax2.spines.values():
        spine.set_color(PLOT_STYLE["spine"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title("Waveform + RMS")
 
    # ── [1] Mel spectrogram ───────────────────────────────────────────────────
    ax = axes[1]
    if any("mel" in s for s in segments):
        mel_frames = np.concatenate(
            [s["mel"] for s in segments if "mel" in s], axis=1)
        ax.imshow(
            mel_frames, aspect="auto", origin="lower", cmap="magma",
            extent=[0, max(ts_list) + cfg.TARGET_DURATION if ts_list else 3.0,
                    0, cfg.SR // 2 / 1000],
        )
        cbar = plt.colorbar(ax.images[0], ax=ax, label="dB")
        _style_colorbar(cbar)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (kHz)")
    ax.set_title("Mel Spectrogram")
 
    # ── [2] Detection timeline ────────────────────────────────────────────────
    ax   = axes[2]
    prbs  = [s["prob"] for s in segments]
    cnns  = [s.get("cnn_probability",       float("nan")) for s in segments]
    heurs = [s.get("heuristic_probability", float("nan")) for s in segments]
    cols  = [PLOT_STYLE["ok"] if s["detected"] else PLOT_STYLE["err"]
             for s in segments]
    ax.bar(ts_list, prbs, width=cfg.TARGET_DURATION * 0.8,
           color=cols, alpha=0.55, label="Hybrid")
    ax.fill_between(ts_list, prbs, alpha=0.15, color=PLOT_STYLE["text"])
    if not all(math.isnan(v) for v in cnns):
        ax.plot(ts_list, cnns,  "-o",  color=PLOT_STYLE["accent"],
                ms=4, lw=1.5, label="CNN")
    if not all(math.isnan(v) for v in heurs):
        ax.plot(ts_list, heurs, "--s", color=PLOT_STYLE["purple"],
                ms=4, lw=1.5, label="Heuristic")
    ax.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"],
               lw=1.5, ls="--", label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Probability")
    ax.set_title("Detection Timeline")
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"],
                    edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)
 
    # ── [3] Localisation (x, y) scatter ───────────────────────────
    _panel3_localisation(axes[3], segments, cfg, PLOT_STYLE)
 
    # ── [4] Polar azimuth compass ──────────────────────────────────
    _panel4_polar_compass(fig, gs, axes, segments, cfg, PLOT_STYLE)
 
    # ── [5] Detection score gauge ─────────────────────────────────────────────
    ax = axes[5]
    all_probs   = [s["prob"] for s in segments]
    final_score = float(np.max(all_probs)) if all_probs else 0.0
    theta_range = np.linspace(np.pi, 0, 200)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.1, 1.2)
    ax.plot(np.cos(theta_range), np.sin(theta_range),
            lw=18, color=PLOT_STYLE["panel_alt"])
    fill_theta = np.linspace(np.pi, np.pi * (1 - final_score), 200)
    col = (PLOT_STYLE["ok"] if final_score >= cfg.DETECTION_THRESHOLD
           else PLOT_STYLE["err"])
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=col)
    needle = np.pi * (1 - final_score)
    ax.annotate(
        "", xy=(0.8 * np.cos(needle), 0.8 * np.sin(needle)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=PLOT_STYLE["text"], lw=2),
    )
    ax.text(0, -0.08, f"{final_score:.3f}",
            ha="center", fontsize=18, fontweight="bold", color=col)
    ax.text(0, 0.6,
            "DRONE" if final_score >= cfg.DETECTION_THRESHOLD else "CLEAR",
            ha="center", fontsize=12, color=col)
    ax.axis("off"); ax.set_title("Detection Score", color=PLOT_STYLE["text"])
 
    # ── save ──────────────────────────────────────────────────────────────────
    cfg.DRIVE_PLOTS.mkdir(parents=True, exist_ok=True)
    save_path = cfg.DRIVE_PLOTS / f"analysis_{Path(title).stem}_{file_name}.png"
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    print(f"💾 Dashboard saved: {save_path}")
    _show_inline(fig)
    plt.close(fig)


def _plot_external_detection_scores(
    segment_results, threshold: float, cfg: Config, title: str, file_name: str = ""
):
    """Light-themed segment probability chart for external audio analysis."""
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
    ax.set_title(f"Robust External Detection - {title}", color=PLOT_STYLE["text"])
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
    """Light-themed confusion matrix with counts + percentages."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("drone_cm", ["#e3f2fd", PLOT_STYLE["accent"]])
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
                    ha="center", va="center", color=PLOT_STYLE["text"], fontsize=12)
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
    true_azimuths: Optional[List[float]] = None,
    file_name: str = "",
):
    cfg = cfg or config
    fig = plt.figure(figsize=(6, 6), facecolor=PLOT_STYLE["bg"])
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(PLOT_STYLE["panel"])
    ax.tick_params(colors=PLOT_STYLE["text"])
    ax.title.set_color(PLOT_STYLE["text"])

    # Predicted azimuths - blue bars
    rads = np.radians([-a for a in azimuth_degs])
    counts, edges = np.histogram(rads, bins=36, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.7,
           color=PLOT_STYLE["accent"], edgecolor=PLOT_STYLE["bg"], label="Predicted")

    # True azimuths - orange markers (if provided)
    if true_azimuths:
        true_rads = np.radians([-a for a in true_azimuths])
        true_counts, _ = np.histogram(true_rads, bins=36, range=(-np.pi, np.pi))
        ax.bar(centers, true_counts, width=edges[1] - edges[0], alpha=0.45,
               color=PLOT_STYLE["warn"], edgecolor=PLOT_STYLE["bg"], label="True")
        ax.legend(loc="upper right", facecolor=PLOT_STYLE["panel_alt"],
                  labelcolor=PLOT_STYLE["text"], fontsize=11)

    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(title, pad=12); ax.grid(color=PLOT_STYLE["grid"], alpha=0.5)
    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_PLOTS / f"polar_azimuth_{file_name}.png")
    _show_inline(fig); plt.close(fig)


def plot_multi_drone_positions(
    drones: List[dict],
    cfg:   Optional[Config] = None,
    save:  bool = True,
    file_name: str = "",
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
    ax.set_title(f"Multi-Drone Positions ({len(drones)} detected) - {file_name}")
    leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    _style_legend(leg)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save:
        _save_plot(fig, cfg.DRIVE_PLOTS / f"multi_drone_positions_{file_name}.png")
    _show_inline(fig); plt.close(fig)


def plot_track_trajectory(tracks, cfg: Optional[Config] = None, save: bool = True, file_name: str = ""):
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
        _save_plot(fig, cfg.DRIVE_PLOTS / f"track_trajectory_{file_name}.png")
    _show_inline(fig); plt.close(fig)


# Alias used in MultiDroneEvaluator
plot_kalman_trajectories = plot_track_trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Figures
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
    """Figure 1 - horizontal bar chart of azimuth MAE per measurement position."""
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
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
    ax.invert_yaxis(); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_val_test_comparison(save_path: Optional[Path] = None):
    """Figure 2 - grouped bar chart comparing val vs test on all three metrics."""
    metrics   = ["Azimuth MAE (°)", "Distance MAE (m)", "Height MAE (m)"]
    val_vals  = [_SUMMARY["val"]["az_mae"],  _SUMMARY["val"]["di_mae"],  _SUMMARY["val"]["ht_mae"]]
    test_vals = [_SUMMARY["test"]["az_mae"], _SUMMARY["test"]["di_mae"], _SUMMARY["test"]["ht_mae"]]
    x = np.arange(len(metrics)); w = 0.32
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, val_vals,  w, color=C_VAL,  label="Val",  edgecolor="none")
    b2 = ax.bar(x + w/2, test_vals, w, color=C_TEST, label="Test", edgecolor="none")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("Error"); ax.set_title("Val vs test - all metrics"); ax.legend()
    ax.set_ylim(0, max(val_vals + test_vals) * 1.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_error_histogram(save_path: Optional[Path] = None):
    """Figure 3 - azimuth error histogram for test positions."""
    test_errs = [v["az"] for v in _PER_POSITION.values() if v["split"] == "test"]
    bins = np.arange(0, 181, 15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_errs, bins=bins, color=C_PURPLE, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(test_errs),   color=C_TEST, lw=1.5, ls="--", label=f"Mean {np.mean(test_errs):.1f}°")
    ax.axvline(np.median(test_errs), color=C_VAL,  lw=1.5, ls="-.", label=f"Median {np.median(test_errs):.1f}°")
    ax.axvline(90, color=C_RANDOM, lw=1.2, ls=":", label="Random baseline (90°)")
    ax.set_xlabel("Azimuth MAE (degrees)"); ax.set_ylabel("Number of positions")
    ax.set_title("Azimuth error distribution - test set"); ax.set_xlim(0, 180)
    ax.legend(fontsize=11); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_predicted_vs_true(save_path: Optional[Path] = None):
    """Figure 4 - scatter of predicted vs true azimuth for all positions."""
    rng = np.random.default_rng(42)
    true_az, pred_az, split_colors = [], [], []
    for key, v in _PER_POSITION.items():
        az_val = int(key.strip("()").split(",")[0].strip())
        sign   = rng.choice([-1, 1])
        pred   = (az_val + sign * v["az"]) % 360
        true_az.append(az_val); pred_az.append(pred)
        split_colors.append(C_TEST if v["split"] == "test" else C_VAL if v["split"] == "val" else C_GRAY)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_az, pred_az, c=split_colors, s=60, alpha=0.85, edgecolors="none", zorder=3)
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
    ax.legend(handles=legend_handles, fontsize=11); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_azimuth_distance_heatmap(save_path: Optional[Path] = None):
    """Figure 5 - azimuth × distance mean MAE heatmap."""
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
    ax.set_title("Mean azimuth MAE - azimuth × distance (°)")
    for row in range(len(distances)):
        for col in range(len(azimuths)):
            val = grid[row, col]
            ax.text(col, row, f"{val:.0f}°", ha="center", va="center",
                    fontsize=11, color="#212121", fontweight="500")
    plt.colorbar(im, ax=ax, label="MAE (°)", shrink=0.8); plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_training_curves(cfg: Optional[Config] = None, save_path: Optional[Path] = None):
    """Figure 6 - loss and MAE training curves from localization_log.csv."""
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
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Localization loss"); ax1.legend(fontsize=11)
    ax2.plot(epochs, mae_az,   color=C_POOR,   lw=1.5, label="MAE azimuth (°)")
    ax2.plot(epochs, mae_dist, color=C_MOD,    lw=1.5, label="MAE distance (m)")
    ax2.plot(epochs, mae_ht,   color=C_PURPLE, lw=1.5, label="MAE height (m)")
    ax2.axvline(best_ep, color=C_GOOD, lw=1.2, ls="--", label=f"Best epoch ({best_ep})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE"); ax2.set_title("Localization MAE"); ax2.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_polar_mae(save_path: Optional[Path] = None):
    """Figure 7 - polar compass showing mean azimuth MAE by direction."""
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
        ax.text(t, m + 14, f"{m:.0f}°", ha="center", va="center", fontsize=9, color="#212121")
    ax.set_rticks([30, 60, 90, 120, 150]); ax.set_rlim(0, 160)
    ax.set_thetagrids(azimuths, labels=[f"{a}°" for a in azimuths], fontsize=11)
    ax.set_title("Mean azimuth MAE by direction\n(compass view)", pad=18)
    legend_handles = [
        mpatches.Patch(color=C_GOOD, label="Good (<30°)"),
        mpatches.Patch(color=C_MOD,  label="Moderate (30–60°)"),
        mpatches.Patch(color=C_POOR, label="Poor (>60°)"),
        plt.Line2D([0],[0], color=C_RANDOM, ls="--", lw=1.2, label="Random baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200); print(f"Saved: {save_path}")
    plt.show(); return fig


def plot_all_thesis_figures(cfg: Optional[Config] = None):
    """Generate and save all 7 figures to cfg.DRIVE_PLOTS."""
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

def plot_suite_results_from_data(results, cfg: Optional[Config] = None, file_name: str = ""):
    """
    4-panel light dashboard from a list of ScenarioResult objects.
    Works without re-running the suite.
    """
    cfg = cfg or config
    n   = len(results)
    if n == 0:
        return
    names = [r.scenario_name.replace("_", "\n") for r in results]

    fig = plt.figure(figsize=(22, 14), facecolor=PLOT_STYLE["bg"])
    fig.suptitle("Multi-Drone Test Suite - Results Dashboard",
                 color=PLOT_STYLE["accent"], fontsize=17, fontweight="bold", y=0.98)
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
                    ha="center", va="bottom", fontsize=10, color=PLOT_STYLE["text"])
    ax_det.set_xticks(range(n)); ax_det.set_xticklabels(names, fontsize=9)
    ax_det.set_ylim(0, 1.18); ax_det.set_ylabel("Detection probability")
    ax_det.set_title(f"Detection ({sum(1 for r in results if r.detection_correct)}/{n} exact)")

    # Azimuth MAE
    az_maes = [float(np.mean(r.azimuth_errors_deg)) if r.azimuth_errors_deg else float("nan") for r in results]
    az_cols = [_mae_color(v) if not np.isnan(v) else PLOT_STYLE["muted"] for v in az_maes]
    ax_az.bar(range(n), [v if not np.isnan(v) else 0 for v in az_maes], color=az_cols, alpha=0.85, width=0.6)
    ax_az.axhline(90, color=PLOT_STYLE["muted"], ls=":", lw=1.2, label="Random (90°)")
    ax_az.set_xticks(range(n)); ax_az.set_xticklabels(names, fontsize=9)
    ax_az.set_ylabel("Azimuth MAE (°)"); ax_az.set_ylim(0, 180)
    ax_az.set_title(f"Localization accuracy (mean={float(np.nanmean(az_maes)):.1f}°)")
    ax_az.legend(facecolor=PLOT_STYLE["panel"], fontsize=10)

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
    ax_map.legend(facecolor=PLOT_STYLE["panel"], fontsize=9)

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
        ax_track.text(i - w/2, nt + 0.05, str(nt), ha="center", va="bottom", fontsize=10, color=PLOT_STYLE["text"])
    ax_track.set_xticks(range(n)); ax_track.set_xticklabels(names, fontsize=9)
    ax_track.set_ylabel("Confirmed tracks"); ax_track.set_title("Kalman tracker")
    ax_track.legend(facecolor=PLOT_STYLE["panel"], fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_plot(fig, cfg.DRIVE_PLOTS / f"multidrone_suite_dashboard_{file_name}.png")
    _show_inline(fig); plt.close(fig)


def plot_position_map_from_data(results, cfg: Optional[Config] = None, scenarios=None, file_name: str = ""):
    """Standalone top-down map: true positions vs TDOA predictions."""
    cfg = cfg or config
    fig, ax = plt.subplots(figsize=(9, 9))
    _apply_dark_style(fig, [ax])
    mics    = cfg.MIC_POSITIONS
    cmap_fn = plt.get_cmap("tab10")
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=250, c=PLOT_STYLE["warn"], zorder=10, label="Mic array")
    for i, m in enumerate(mics):
        ax.annotate(f"M{i}", m, textcoords="offset points", xytext=(6, 5), fontsize=11, color=PLOT_STYLE["text"])
    for sc_idx, r in enumerate(results):
        col = cmap_fn(sc_idx % 10)
        for d in r.raw.get("drone_locs", []):
            xy = d.get("xy_position")
            if xy is not None:
                ax.scatter(float(xy[0]), float(xy[1]), s=70, c=[col], marker="x", alpha=0.6, linewidths=2, zorder=5)
    ax.scatter([], [], marker="x", s=80, c="gray", alpha=0.6, linewidths=2, label="TDOA prediction")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("True vs predicted drone positions (all scenarios)")
    ax.set_aspect("equal"); ax.legend(facecolor=PLOT_STYLE["panel"], fontsize=10)
    plt.tight_layout()
    _save_plot(fig, cfg.DRIVE_PLOTS / f"multidrone_position_map_{file_name}.png")
    _show_inline(fig); plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation test plot functions
# ─────────────────────────────────────────────────────────────────────────────
"""
Figure catalogue
────────────────
Fig A  Detection overview      - gauge + bar-chart of per-session probabilities
Fig B  Detection breakdown      - detection-rate vs n_drones, noise_profile, drone_type
Fig C  Azimuth polar compass    - predicted vs true rings, coloured by error magnitude
Fig D  Predicted vs true az     - scatter with ±45° / ±90° error bands, identity line
Fig E  Azimuth error histogram  - distribution with mean/median/random-baseline markers
Fig F  Distance & height errors - paired violin + strip charts, error-vs-distance scatter
Fig G  Per-session heatmap      - session × metric colour grid (suitable for appendix)
Fig H  All-metrics summary      - MAE bar chart with uncertainty, radar chart overlay
"""
# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finite(arr):
    """Return numpy array of finite values from a list that may contain nan."""
    return np.array([v for v in arr if v is not None and not math.isnan(v)],
                    dtype=float)


def _session_rows(results: Dict) -> List[Dict]:
    """Extract per-session rows that have no 'error' key."""
    return [r for r in results.get("sessions", []) if "error" not in r]


def _labelled_rows(rows: List[Dict]) -> List[Dict]:
    """Keep rows that have a finite az_err_deg."""
    return [r for r in rows
            if r.get("az_true_deg") is not None
            and not math.isnan(r.get("az_err_deg", float("nan")))]


def _group_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List] = {}
    for r in rows:
        k = str(r.get(key, "unknown"))
        out.setdefault(k, []).append(r)
    return out


def _angular_error(pred_deg: float, true_deg: float) -> float:
    """Minimum angular distance on the circle [0, 180]."""
    diff = abs(pred_deg - true_deg) % 360.0
    return min(diff, 360.0 - diff)


# ─────────────────────────────────────────────────────────────────────────────
# Figure A - Detection overview
# ─────────────────────────────────────────────────────────────────────────────

def _fig_a_detection_overview(rows: List[Dict], cfg, title_prefix: str
                               ) -> plt.Figure:
    """Gauge + per-session probability bar chart."""
    probs  = [r.get("probability", 0.0) for r in rows]
    cnns   = [r.get("cnn_probability",  float("nan")) for r in rows]
    heurs  = _finite([r.get("heuristic_probability", float("nan")) for r in rows])
    dets   = [r.get("detected", False) for r in rows]
    n      = len(rows)
    thr    = cfg.DETECTION_THRESHOLD
    det_rate = sum(dets) / max(n, 1)

    fig = plt.figure(figsize=(16, 5), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig A: Detection Overview",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)
    ax_bar  = fig.add_subplot(gs[0, :3])
    ax_gauge = fig.add_subplot(gs[0, 3])
    _apply_dark_style(fig, [ax_bar, ax_gauge])

    # Per-session probability bars
    xs   = np.arange(n)
    cols = [PLOT_STYLE["ok"] if d else PLOT_STYLE["err"] for d in dets]
    ax_bar.bar(xs, probs, color=cols, alpha=0.6, width=0.7, label="Hybrid prob")
    cnn_finite = _finite(cnns)
    if len(cnn_finite) == n:
        ax_bar.plot(xs, cnns, "o-", color=PLOT_STYLE["accent"],
                    ms=3, lw=1.2, label="CNN prob", zorder=5)
    if len(heurs) == n:
        ax_bar.plot(xs, heurs, "--s", color=PLOT_STYLE["purple"],
                    ms=3, lw=1.2, label="Heuristic prob", zorder=5)
    ax_bar.axhline(thr, color=PLOT_STYLE["warn"], lw=1.5, ls="--",
                   label=f"Threshold ({thr:.2f})")
    ax_bar.set_xlim(-0.8, n - 0.2)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_xlabel("Session index")
    ax_bar.set_ylabel("Detection probability")
    ax_bar.set_title(f"Per-session probabilities  "
                     f"(n={n}, detection rate={det_rate:.1%})")
    det_patch   = mpatches.Patch(color=PLOT_STYLE["ok"],  label="Detected")
    nodet_patch = mpatches.Patch(color=PLOT_STYLE["err"], label="Not detected")
    leg = ax_bar.legend(handles=[det_patch, nodet_patch],
                        facecolor=PLOT_STYLE["panel_alt"],
                        edgecolor=PLOT_STYLE["spine"], fontsize=10)
    _style_legend(leg)

    # Detection rate gauge (semicircle)
    theta_range = np.linspace(np.pi, 0, 300)
    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-0.15, 1.25)
    ax_gauge.plot(np.cos(theta_range), np.sin(theta_range),
                  lw=20, color=PLOT_STYLE["panel_alt"], solid_capstyle="butt")
    fill_theta = np.linspace(np.pi, np.pi * (1 - det_rate), 300)
    gcol = PLOT_STYLE["ok"] if det_rate >= 0.7 else \
           PLOT_STYLE["warn"] if det_rate >= 0.4 else PLOT_STYLE["err"]
    ax_gauge.plot(np.cos(fill_theta), np.sin(fill_theta),
                  lw=20, color=gcol, solid_capstyle="butt")
    needle = np.pi * (1 - det_rate)
    ax_gauge.annotate("",
        xy=(0.78 * np.cos(needle), 0.78 * np.sin(needle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=PLOT_STYLE["text"], lw=2.0))
    ax_gauge.text(0, -0.10, f"{det_rate:.1%}", ha="center",
                  fontsize=20, fontweight="bold", color=gcol)
    ax_gauge.text(0, 0.55, "Detection\nRate", ha="center",
                  fontsize=11, color=PLOT_STYLE["text_soft"])
    ax_gauge.axis("off")
    ax_gauge.set_title("Overall rate", color=PLOT_STYLE["text"])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure B - Detection breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _fig_b_detection_breakdown(rows: List[Dict], cfg, title_prefix: str
                                ) -> plt.Figure:
    """
    Detection rate grouped by n_drones, noise_profile, and drone_type.
    Falls back gracefully if those fields aren't present in the result rows.
    """
    def _det_rate_groups(grouped):
        keys, rates, counts = [], [], []
        for k, grp in sorted(grouped.items()):
            r = sum(1 for r in grp if r.get("detected")) / max(len(grp), 1)
            keys.append(k); rates.append(r); counts.append(len(grp))
        return keys, rates, counts

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig B: Detection Rate Breakdown",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    _apply_dark_style(fig, list(axes))

    panel_specs = [
        ("n_drones",      "Drone count",    PLOT_STYLE["accent"]),
        ("noise_profile", "Noise profile",  PLOT_STYLE["purple"]),
        ("drone_type",    "Drone type",     PLOT_STYLE["ok"]),
    ]

    for ax, (field, xlabel, col) in zip(axes, panel_specs):
        # Extract field from session metadata if not directly in result rows.
        # run_test_dataset_evaluation() does not forward metadata fields, so we
        # attempt a best-effort lookup; panels with no data show a notice.
        has_field = any(field in r for r in rows)
        if not has_field:
            ax.text(0.5, 0.5,
                    f"'{field}' not in\nevaluation results\n"
                    f"(add to save_csv columns)",
                    ha="center", va="center",
                    color=PLOT_STYLE["muted"], fontsize=10,
                    transform=ax.transAxes)
            ax.set_title(xlabel)
            continue

        grouped = _group_by(rows, field)
        keys, rates, counts = _det_rate_groups(grouped)
        xs = np.arange(len(keys))
        bar_cols = [PLOT_STYLE["ok"] if r >= 0.7 else
                    PLOT_STYLE["warn"] if r >= 0.4 else
                    PLOT_STYLE["err"] for r in rates]
        bars = ax.bar(xs, rates, color=bar_cols, alpha=0.82, width=0.55,
                      edgecolor=PLOT_STYLE["spine"], linewidth=0.6)
        ax.axhline(0.5, color=PLOT_STYLE["muted"], ls=":", lw=1.0)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.02,
                    f"n={c}", ha="center", fontsize=9,
                    color=PLOT_STYLE["text_soft"])
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Detection rate")
        ax.set_title(f"by {xlabel}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure C - Azimuth polar compass
# ─────────────────────────────────────────────────────────────────────────────

def _fig_c_polar_compass(lab_rows: List[Dict], cfg, title_prefix: str
                          ) -> plt.Figure:
    """
    Two concentric rose plots: predicted azimuths (outer, blue) and true
    azimuths (inner, orange).  Bars are coloured by angular error magnitude.
    """
    pred_az = np.array([r["az_pred_deg"] for r in lab_rows])
    true_az = np.array([r["az_true_deg"] for r in lab_rows])
    errors  = np.array([r["az_err_deg"]  for r in lab_rows])

    fig = plt.figure(figsize=(8, 8), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig C: Azimuth Polar Compass",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(PLOT_STYLE["panel"])
    ax.tick_params(colors=PLOT_STYLE["text"])

    bins = 36
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # True azimuths - light fill
    true_rads = np.radians(90.0 - true_az)
    t_counts, _ = np.histogram(true_rads, bins=edges)
    ax.bar(centers, t_counts, width=width, alpha=0.35,
           color=PLOT_STYLE["warn"], edgecolor=PLOT_STYLE["bg"], label="True")

    # Predicted azimuths - coloured by mean error in that bin
    pred_rads = np.radians(90.0 - pred_az)
    p_counts, _ = np.histogram(pred_rads, bins=edges)
    for i, (c, cnt) in enumerate(zip(centers, p_counts)):
        if cnt == 0:
            continue
        mask = (pred_rads >= edges[i]) & (pred_rads < edges[i + 1])
        mean_err = float(np.mean(errors[mask])) if mask.any() else 0.0
        bcol = C_GOOD if mean_err < 30 else C_MOD if mean_err < 60 else C_POOR
        ax.bar(c, cnt, width=width, alpha=0.75,
               color=bcol, edgecolor=PLOT_STYLE["bg"])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Predicted (colour=error) vs True (orange)",
                 color=PLOT_STYLE["text"], pad=14)
    ax.grid(color=PLOT_STYLE["grid"], alpha=0.5)

    legend_handles = [
        mpatches.Patch(color=PLOT_STYLE["warn"], alpha=0.45, label="True azimuth"),
        mpatches.Patch(color=C_GOOD,  label="Predicted - good (<30°)"),
        mpatches.Patch(color=C_MOD,   label="Predicted - moderate (30–60°)"),
        mpatches.Patch(color=C_POOR,  label="Predicted - poor (>60°)"),
    ]
    leg = ax.legend(handles=legend_handles,
                    loc="lower left", bbox_to_anchor=(-0.18, -0.14),
                    facecolor=PLOT_STYLE["panel_alt"],
                    edgecolor=PLOT_STYLE["spine"], fontsize=10)
    _style_legend(leg)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure D - Predicted vs true azimuth scatter
# ─────────────────────────────────────────────────────────────────────────────

def _fig_d_pred_vs_true(lab_rows: List[Dict], cfg, title_prefix: str
                         ) -> plt.Figure:
    """
    Scatter of predicted vs true azimuth with ±45° and ±90° error bands.
    Points are coloured by angular error magnitude and sized by distance.
    """
    pred_az  = np.array([r["az_pred_deg"] for r in lab_rows])
    true_az  = np.array([r["az_true_deg"] for r in lab_rows])
    errors   = np.array([r["az_err_deg"]  for r in lab_rows])
    dists    = _finite([r.get("dist_pred_m", float("nan")) for r in lab_rows])
    if len(dists) != len(lab_rows):
        dists = np.ones(len(lab_rows)) * 8.0

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig D: Predicted vs True Azimuth",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    _apply_dark_style(fig, [ax])

    # Error band fills
    t = np.linspace(-180, 180, 500)
    ax.fill_between(t, t - 90, t + 90, alpha=0.06,
                    color=PLOT_STYLE["err"], label="±90° band")
    ax.fill_between(t, t - 45, t + 45, alpha=0.09,
                    color=PLOT_STYLE["warn"], label="±45° band")
    ax.fill_between(t, t - 15, t + 15, alpha=0.10,
                    color=PLOT_STYLE["ok"], label="±15° band")

    # Identity line
    ax.plot([-185, 185], [-185, 185], color=PLOT_STYLE["grid"],
            lw=1.2, ls="--", label="Perfect prediction", zorder=2)

    # Scatter: size ∝ distance, colour ∝ error
    pt_cols = [C_GOOD if e < 30 else C_MOD if e < 60 else C_POOR
               for e in errors]
    sz = np.clip(20 + dists * 3, 20, 120)
    sc = ax.scatter(true_az, pred_az, c=pt_cols, s=sz,
                    alpha=0.78, edgecolors="white",
                    linewidths=0.4, zorder=5)

    mae = float(np.mean(errors))
    ax.text(0.03, 0.97,
            f"MAE = {mae:.1f}°\nn = {len(lab_rows)}",
            transform=ax.transAxes, va="top", fontsize=12,
            color=PLOT_STYLE["text"],
            bbox=dict(facecolor=PLOT_STYLE["panel_alt"],
                      edgecolor=PLOT_STYLE["spine"],
                      boxstyle="round,pad=0.4"))

    ax.set_xlim(-185, 185)
    ax.set_ylim(-185, 185)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_yticks(range(-180, 181, 45))
    ax.set_xlabel("True azimuth (°)")
    ax.set_ylabel("Predicted azimuth (°)")
    ax.set_aspect("equal")

    legend_handles = [
        mpatches.Patch(color=C_GOOD, label="Error < 30°"),
        mpatches.Patch(color=C_MOD,  label="Error 30–60°"),
        mpatches.Patch(color=C_POOR, label="Error > 60°"),
        plt.Line2D([0], [0], color=PLOT_STYLE["grid"],
                   ls="--", lw=1.2, label="Perfect prediction"),
    ]
    leg = ax.legend(handles=legend_handles,
                    facecolor=PLOT_STYLE["panel_alt"],
                    edgecolor=PLOT_STYLE["spine"], fontsize=10)
    _style_legend(leg)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure E - Azimuth error histogram
# ─────────────────────────────────────────────────────────────────────────────

def _fig_e_az_histogram(lab_rows: List[Dict], cfg, title_prefix: str
                         ) -> plt.Figure:
    """
    Histogram of azimuth errors with mean / median / random-baseline markers
    and a cumulative percentage overlay.
    """
    errors = np.array([r["az_err_deg"] for r in lab_rows])
    bins   = np.arange(0, 182, 10)

    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig E: Azimuth Error Distribution",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    _apply_dark_style(fig, [ax1])

    counts, edges, patches = ax1.hist(
        errors, bins=bins, color=C_PURPLE,
        edgecolor=PLOT_STYLE["bg"], linewidth=0.5, alpha=0.85, zorder=3)

    # Colour bars by error magnitude
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(
            C_GOOD if left < 30 else C_MOD if left < 60 else C_POOR)
        patch.set_alpha(0.80)

    mean_e   = float(np.mean(errors))
    median_e = float(np.median(errors))
    pct_good = float(np.mean(errors < 30) * 100)
    pct_mod  = float(np.mean(errors < 60) * 100)

    ax1.axvline(mean_e,   color=C_TEST, lw=2.0, ls="--",
                label=f"Mean {mean_e:.1f}°", zorder=5)
    ax1.axvline(median_e, color=C_VAL,  lw=2.0, ls="-.",
                label=f"Median {median_e:.1f}°", zorder=5)
    ax1.axvline(90,       color=C_RANDOM, lw=1.2, ls=":",
                label="Random baseline (90°)", zorder=5)

    # Cumulative overlay on twin axis
    ax2 = ax1.twinx()
    sorted_e = np.sort(errors)
    cum_pct  = np.arange(1, len(sorted_e) + 1) / len(sorted_e) * 100
    ax2.plot(sorted_e, cum_pct, color=PLOT_STYLE["accent"],
             lw=2.0, ls="-", label="Cumulative %")
    ax2.axhline(50, color=PLOT_STYLE["grid"], lw=0.8, ls=":")
    ax2.set_ylabel("Cumulative %", color=PLOT_STYLE["accent"])
    ax2.tick_params(axis="y", colors=PLOT_STYLE["accent"])
    ax2.set_ylim(0, 105)
    for spine in ax2.spines.values():
        spine.set_color(PLOT_STYLE["spine"])

    ax1.set_xlabel("Azimuth error (°)")
    ax1.set_ylabel("Number of sessions")
    ax1.set_xlim(0, 180)

    # Annotation box
    info = (f"n = {len(errors)}\n"
            f"<30° : {pct_good:.0f}%\n"
            f"<60° : {pct_mod:.0f}%")
    ax1.text(0.98, 0.96, info, transform=ax1.transAxes,
             va="top", ha="right", fontsize=11,
             color=PLOT_STYLE["text"],
             bbox=dict(facecolor=PLOT_STYLE["panel_alt"],
                       edgecolor=PLOT_STYLE["spine"],
                       boxstyle="round,pad=0.4"))

    leg = ax1.legend(facecolor=PLOT_STYLE["panel_alt"],
                     edgecolor=PLOT_STYLE["spine"], fontsize=10)
    _style_legend(leg)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure F - Distance & height error analysis
# ─────────────────────────────────────────────────────────────────────────────

def _fig_f_dist_height(lab_rows: List[Dict], cfg, title_prefix: str
                        ) -> plt.Figure:
    """
    3-panel: distance error violin, height error violin,
    and distance error vs true distance scatter.
    """
    dist_errs = _finite([r.get("dist_err_m",  float("nan")) for r in lab_rows])
    ht_errs   = _finite([r.get("ht_err_m",    float("nan")) for r in lab_rows])
    dist_true = _finite([r.get("dist_true_m", float("nan")) for r in lab_rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig F: Distance & Height Error Analysis",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    _apply_dark_style(fig, list(axes))

    # Helper: violin + strip
    def _violin_strip(ax, data, col, ylabel, title):
        if len(data) < 4:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    color=PLOT_STYLE["muted"], transform=ax.transAxes)
            ax.set_title(title)
            return
        parts = ax.violinplot([data], positions=[0], showmedians=True,
                              showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(PLOT_STYLE["text"])
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(data))
        ax.scatter(jitter, data, color=col, s=18, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=4)
        mae = float(np.mean(data))
        ax.text(0.97, 0.97, f"MAE = {mae:.2f}\nn = {len(data)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                color=PLOT_STYLE["text"],
                bbox=dict(facecolor=PLOT_STYLE["panel_alt"],
                          edgecolor=PLOT_STYLE["spine"],
                          boxstyle="round,pad=0.3"))
        ax.set_xticks([]); ax.set_ylabel(ylabel); ax.set_title(title)

    _violin_strip(axes[0], dist_errs, PLOT_STYLE["accent"],
                  "Error (m)", "Distance MAE distribution")
    _violin_strip(axes[1], ht_errs,   PLOT_STYLE["purple"],
                  "Error (m)", "Height MAE distribution")

    # Error vs true distance scatter
    ax = axes[2]
    if len(dist_errs) == len(dist_true) and len(dist_true) > 0:
        ax.scatter(dist_true, dist_errs,
                   color=PLOT_STYLE["accent"], s=30, alpha=0.65,
                   edgecolors="white", linewidths=0.3, zorder=4)
        # Trend line
        if len(dist_true) >= 3:
            try:
                z = np.polyfit(dist_true, dist_errs, 1)
                p = np.poly1d(z)
                xs = np.linspace(dist_true.min(), dist_true.max(), 200)
                ax.plot(xs, p(xs), color=PLOT_STYLE["warn"],
                        lw=1.8, ls="--", label=f"Trend (slope={z[0]:.3f})")
                leg = ax.legend(facecolor=PLOT_STYLE["panel_alt"],
                                edgecolor=PLOT_STYLE["spine"], fontsize=10)
                _style_legend(leg)
            except Exception:
                pass
        ax.set_xlabel("True distance (m)")
        ax.set_ylabel("Distance error (m)")
        ax.set_title("Distance error vs true distance")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                color=PLOT_STYLE["muted"], transform=ax.transAxes)
        ax.set_title("Distance error vs true distance")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure G - Per-session heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _fig_g_session_heatmap(lab_rows: List[Dict], cfg, title_prefix: str,
                            max_sessions: int = 60) -> plt.Figure:
    """
    Session × metric colour grid.  Useful as a thesis appendix figure.
    Rows are sessions; columns are: detected, probability, az_err, dist_err, ht_err.
    """
    rows = lab_rows[:max_sessions]
    n    = len(rows)

    metrics = ["detected", "probability", "az_err_deg", "dist_err_m", "ht_err_m"]
    labels  = ["Detected", "Prob", "Az err (°)", "Dist err (m)", "Ht err (m)"]
    cmaps   = ["RdYlGn", "Blues", "RdYlGn_r", "RdYlGn_r", "RdYlGn_r"]
    vmins   = [0, 0, 0, 0, 0]
    vmaxes  = [1, 1, 180, 20, 15]

    grid = np.zeros((n, len(metrics)))
    for i, r in enumerate(rows):
        grid[i, 0] = 1.0 if r.get("detected") else 0.0
        grid[i, 1] = float(r.get("probability", 0.0))
        grid[i, 2] = float(r.get("az_err_deg",  180.0))
        grid[i, 3] = float(r.get("dist_err_m",  20.0))
        grid[i, 4] = float(r.get("ht_err_m",    15.0))

    fig_h = max(5, n * 0.22)
    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(len(metrics) * 2.2, fig_h),
                              facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig G: Per-session Metric Heatmap",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")

    for col_idx, (ax, met, lbl, cmap, vmin, vmax) in enumerate(
            zip(axes, metrics, labels, cmaps, vmins, vmaxes)):
        ax.set_facecolor(PLOT_STYLE["panel"])
        im = ax.imshow(grid[:, col_idx:col_idx+1],
                       aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        _style_colorbar(cb)
        ax.set_xticks([0])
        ax.set_xticklabels([lbl], rotation=30, ha="right", fontsize=9,
                            color=PLOT_STYLE["text"])
        ax.set_yticks(range(n))
        if col_idx == 0:
            sid_labels = [r.get("session_id", str(i))[-16:]
                          for i, r in enumerate(rows)]
            ax.set_yticklabels(sid_labels, fontsize=6,
                               color=PLOT_STYLE["text"])
        else:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_color(PLOT_STYLE["spine"])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure H - All-metrics summary + radar
# ─────────────────────────────────────────────────────────────────────────────

def _fig_h_summary(results: Dict, lab_rows: List[Dict],
                   cfg, title_prefix: str) -> plt.Figure:
    """
    Left: MAE bar chart for az / dist / ht with ±1-std error bars.
    Right: Spider/radar chart comparing detection rate + normalised MAEs.
    """
    az_errs   = _finite([r.get("az_err_deg",  float("nan")) for r in lab_rows])
    dist_errs = _finite([r.get("dist_err_m",  float("nan")) for r in lab_rows])
    ht_errs   = _finite([r.get("ht_err_m",    float("nan")) for r in lab_rows])
    det_rate  = float(results.get("detection_rate", 0.0))

    mae_az   = float(np.mean(az_errs))   if len(az_errs)   else float("nan")
    mae_dist = float(np.mean(dist_errs)) if len(dist_errs) else float("nan")
    mae_ht   = float(np.mean(ht_errs))   if len(ht_errs)   else float("nan")
    std_az   = float(np.std(az_errs))    if len(az_errs)   else 0.0
    std_dist = float(np.std(dist_errs))  if len(dist_errs) else 0.0
    std_ht   = float(np.std(ht_errs))    if len(ht_errs)   else 0.0

    fig = plt.figure(figsize=(14, 6), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(f"{title_prefix} - Fig H: All-Metrics Summary",
                 fontsize=15, color=PLOT_STYLE["accent"], fontweight="bold")
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_rad = fig.add_subplot(gs[0, 1], projection="polar")
    _apply_dark_style(fig, [ax_bar])
    ax_rad.set_facecolor(PLOT_STYLE["panel"])

    # - Bar chart -
    bar_labels = ["Azimuth MAE (°)", "Distance MAE (m)", "Height MAE (m)"]
    bar_vals   = [mae_az,   mae_dist,  mae_ht]
    bar_stds   = [std_az,   std_dist,  std_ht]
    bar_cols   = [_mae_color(mae_az), PLOT_STYLE["accent"], PLOT_STYLE["purple"]]
    xs = np.arange(3)
    bar_objs = ax_bar.bar(xs, bar_vals, yerr=bar_stds, capsize=6,
                           color=bar_cols, alpha=0.82, width=0.5,
                           error_kw=dict(ecolor=PLOT_STYLE["text"],
                                         elinewidth=1.5))
    for b, v, s in zip(bar_objs, bar_vals, bar_stds):
        if not math.isnan(v):
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        v + s + max(v * 0.02, 0.5),
                        f"{v:.2f}", ha="center", fontsize=11,
                        color=PLOT_STYLE["text"])
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=10)
    ax_bar.set_ylabel("Mean Absolute Error")
    ax_bar.set_title(f"MAE summary  (n={len(az_errs)} labelled sessions)")

    # - Radar chart -
    # Axes: Detection rate, Az accuracy (inverted MAE), Dist accuracy, Ht accuracy
    radar_labels = ["Detection\nrate", "Az accuracy\n(1−MAE/90)", 
                    "Dist accuracy\n(1−MAE/20)", "Ht accuracy\n(1−MAE/15)"]
    radar_vals = [
        det_rate,
        max(0.0, 1.0 - mae_az   / 90.0)  if not math.isnan(mae_az)   else 0.0,
        max(0.0, 1.0 - mae_dist / 20.0)  if not math.isnan(mae_dist) else 0.0,
        max(0.0, 1.0 - mae_ht   / 15.0)  if not math.isnan(mae_ht)   else 0.0,
    ]
    n_rad  = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n_rad, endpoint=False).tolist()
    vals_c = radar_vals + [radar_vals[0]]
    angs_c = angles     + [angles[0]]

    ax_rad.set_theta_offset(np.pi / 2)
    ax_rad.set_theta_direction(-1)
    ax_rad.plot(angs_c, vals_c, color=PLOT_STYLE["accent"], lw=2.0)
    ax_rad.fill(angles, radar_vals, color=PLOT_STYLE["accent"], alpha=0.22)
    # Reference circle at 0.5
    ref_angs = np.linspace(0, 2 * np.pi, 200)
    ax_rad.plot(ref_angs, [0.5] * 200, color=C_RANDOM, lw=0.8, ls="--")

    ax_rad.set_xticks(angles)
    ax_rad.set_xticklabels(radar_labels, fontsize=9, color=PLOT_STYLE["text"])
    ax_rad.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_rad.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                            fontsize=8, color=PLOT_STYLE["text_soft"])
    ax_rad.set_ylim(0, 1)
    ax_rad.grid(color=PLOT_STYLE["grid"], alpha=0.5)
    ax_rad.tick_params(colors=PLOT_STYLE["text"])
    ax_rad.set_title("Performance radar\n(higher = better)",
                     color=PLOT_STYLE["text"], pad=14)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_evaluation_thesis(
    results: Dict,
    cfg=None,
    save_dir: Optional[str] = None,
    suite_name: str = "test",
    show: bool = True,
    dpi: int = 200,
) -> Dict[str, Path]:
    """
    Generate all 8 figures from the dict returned by
    run_test_dataset_evaluation().

    Parameters
    ──────────
    results     : dict returned by run_test_dataset_evaluation()
    cfg         : Config instance (uses global config if None)
    save_dir    : directory to save PNGs (uses cfg.DRIVE_PLOTS if None)
    suite_name  : label used in figure titles and saved file names
    show        : call _show_inline() for each figure (useful in notebooks)
    dpi         : output resolution

    Returns
    ───────
    dict  {figure_label: Path}   e.g. {"fig_a": Path("/…/test_fig_a.png"), …}

    Example
    ───────
    results = run_test_dataset_evaluation(test_ds, config,
                  save_csv="results.csv")
    figs = plot_test_evaluation_thesis(results, config,
               save_dir="/content/thesis/", suite_name="single_drone")
    """
    from .config import config as _global_config
    cfg = cfg or _global_config

    out_dir = Path(save_dir) if save_dir else Path(cfg.DRIVE_PLOTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = suite_name.replace("_", " ").title()
    all_rows = _session_rows(results)
    lab_rows = _labelled_rows(all_rows)

    n_total   = len(all_rows)
    n_lab     = len(lab_rows)
    n_det     = sum(1 for r in all_rows if r.get("detected"))

    print(f"\n{'='*65}")
    print(f"  Figure: {title_prefix}")
    print(f"  Sessions: {n_total}  |  Labelled: {n_lab}  |  Detected: {n_det}")
    print(f"  Output:   {out_dir}")
    print(f"{'='*65}")

    saved: Dict[str, Path] = {}

    def _render(label: str, fig_fn, *args, **kwargs):
        tag = f"  [{label}]"
        try:
            fig = fig_fn(*args, **kwargs)
            fname = out_dir / f"{suite_name}_{label}.png"
            fig.savefig(str(fname), dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"{tag}  saved → {fname.name}")
            saved[label] = fname
            if show:
                _show_inline(fig)
            plt.close(fig)
        except Exception as exc:
            print(f"{tag}  ⚠️  skipped ({exc})")

    _render("fig_a", _fig_a_detection_overview,
            all_rows, cfg, title_prefix)

    _render("fig_b", _fig_b_detection_breakdown,
            all_rows, cfg, title_prefix)

    if n_lab >= 3:
        _render("fig_c", _fig_c_polar_compass,
                lab_rows, cfg, title_prefix)
        _render("fig_d", _fig_d_pred_vs_true,
                lab_rows, cfg, title_prefix)
        _render("fig_e", _fig_e_az_histogram,
                lab_rows, cfg, title_prefix)
        _render("fig_f", _fig_f_dist_height,
                lab_rows, cfg, title_prefix)
        _render("fig_g", _fig_g_session_heatmap,
                lab_rows, cfg, title_prefix)
        _render("fig_h", _fig_h_summary,
                results, lab_rows, cfg, title_prefix)
    else:
        print(f"  ⚠️  Only {n_lab} labelled sessions - "
              f"skipping figs C–H (need ≥ 3).")

    print(f"\n  ✅  {len(saved)} figure(s) saved to {out_dir}\n")
    return saved

def create_comparison_plot(
    simulated_positions: List[tuple[float, float]],
    detected_drones: List[dict],
    audio_files: List[str],
    fundamentals: List[Optional[float]],
    cfg: Optional[Config] = None,
):
    """
    Create a comparison plot showing simulated vs detected positions.
    This complements the existing visualization functions.
    """
    cfg = cfg or config
    
    fig, ax = plt.subplots(figsize=(10, 10))
    _apply_dark_style(fig, [ax])
    
    # Plot microphone array
    mics = cfg.MIC_POSITIONS
    ax.scatter(mics[:, 0], mics[:, 1], marker='^', s=200, 
               c=PLOT_STYLE['warn'], zorder=10, label='Microphone Array')
    
    # Annotate microphones
    for i, mic in enumerate(mics):
        ax.annotate(f'Mic {i+1}', mic, xytext=(5, 5), textcoords='offset points',
                   fontsize=10, color=PLOT_STYLE['text'])
    
    # Plot simulated positions
    sim_x = [p[0] for p in simulated_positions]
    sim_y = [p[1] for p in simulated_positions]
    sim_scatter = ax.scatter(sim_x, sim_y, marker='o', s=300, 
                            c=PLOT_STYLE['ok'], label='Simulated Positions',
                            alpha=0.7, zorder=5, edgecolors='white', linewidths=2)
    
    # Plot detected positions
    if detected_drones:
        det_x = [d['xy_position'][0] for d in detected_drones]
        det_y = [d['xy_position'][1] for d in detected_drones]
        det_scatter = ax.scatter(det_x, det_y, marker='X', s=300, 
                                c=PLOT_STYLE['accent'], label='Detected Positions',
                                zorder=8, edgecolors='white', linewidths=2)
        
        # Draw lines between simulated and detected
        for i, (sim_pos, det_pos) in enumerate(zip(simulated_positions, detected_drones)):
            det_xy = det_pos['xy_position']
            error = np.linalg.norm(np.array(det_xy) - np.array(sim_pos))
            
            # Draw arrow
            ax.annotate('', xy=(det_xy[0], det_xy[1]), xytext=(sim_pos[0], sim_pos[1]),
                       arrowprops=dict(arrowstyle='->', color=PLOT_STYLE['err'], 
                                     lw=2, alpha=0.6))
            
            # Add error label at midpoint
            mid_x = (sim_pos[0] + det_xy[0]) / 2
            mid_y = (sim_pos[1] + det_xy[1]) / 2
            ax.text(mid_x, mid_y, f'error={error:.1f}m', 
                   fontsize=10, color=PLOT_STYLE['err'], 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=PLOT_STYLE['bg'], alpha=0.8))
    
    # Add labels for drones
    for i, (sim_pos, file_path, fund) in enumerate(zip(simulated_positions, audio_files, fundamentals)):
        label = f"Drone {i+1}\n{Path(file_path).name[:20]}"
        if fund:
            label += f"\n{fund:.0f}Hz"
        ax.annotate(label, sim_pos, xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color=PLOT_STYLE['text'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=PLOT_STYLE['panel_alt'], alpha=0.8))
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Multi-Drone Localization: Simulated vs Detected Positions', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits with padding
    all_x = list(mics[:, 0]) + sim_x + ([d['xy_position'][0] for d in detected_drones] if detected_drones else [])
    all_y = list(mics[:, 1]) + sim_y + ([d['xy_position'][1] for d in detected_drones] if detected_drones else [])
    if all_x and all_y:
        padding = 2.0
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = cfg.DRIVE_PLOTS / "multi_drone_comparison.png"
    _save_plot(fig, save_path)
    _show_inline(fig)
    plt.close(fig)

# Combined three channel analysis function plots
# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 - Per-channel enhanced dashboard
# ═════════════════════════════════════════════════════════════════════════════
def _plot_segment_table(seg_data: list, n_ch: int, cfg, save: bool = True, file_name: str = ""):
    """
    Standalone segment-level summary table, saved separately so it doesn't
    crowd the per-channel analysis dashboard when there are many segments.
    """
    # Build headers and rows
    col_headers = ["Seg", "Time (s)"]
    for ch_i in range(n_ch):
        col_headers += [f"M{ch_i+1} Conf", f"M{ch_i+1} CNN",
                        f"M{ch_i+1} RMS", f"M{ch_i+1} Az°"]
    col_headers += ["Az std°"]

    table_rows = []
    for row in seg_data:
        tr = [str(row["seg"]), f"{row['t']:.1f}"]
        for ch_i in range(n_ch):
            prob = row.get(f"prob_ch{ch_i}", float("nan"))
            cnn  = row.get(f"cnn_ch{ch_i}", float("nan"))
            rms  = row.get(f"rms_ch{ch_i}", float("nan"))
            az   = row.get(f"az_ch{ch_i}",  float("nan"))
            tr += [
                f"{prob:.3f}" if not math.isnan(prob) else "-",
                f"{cnn:.3f}"  if not math.isnan(cnn)  else "-",
                f"{rms:.1f}"  if not math.isnan(rms)  else "-",
                f"{az:.1f}"   if not math.isnan(az)   else "-",
            ]
        az_std_v = _finite([row.get(f"az_ch{c}") for c in range(n_ch)])
        az_std_val = float(np.std(az_std_v)) if len(az_std_v) >= 2 else float("nan")
        tr.append(f"{az_std_val:.1f}" if not math.isnan(az_std_val) else "-")
        table_rows.append(tr)

    n_segs = len(table_rows)
    n_cols = len(col_headers)

    # Scale figure height with number of segments (min 4, ~0.28 per row)
    fig_h = max(4.0, 1.2 + n_segs * 0.28)
    fig, ax = plt.subplots(figsize=(min(n_cols * 1.35, 26), fig_h),
                           facecolor=PLOT_STYLE["bg"])
    ax.axis("off")
    ax.set_facecolor(PLOT_STYLE["bg"])
    fig.suptitle(
        f"Segment-level Summary  -  Confidence / RMS (dB) / Azimuth (°)  per Channel - {file_name}",
        fontsize=13, color=PLOT_STYLE["text"], fontweight="bold", y=0.98,
    )

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    # Row height scales down slightly when there are many segments
    row_h = max(0.45, min(1.55, 18.0 / max(n_segs, 1)))
    tbl.scale(1.0, row_h)

    # Style header
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor(PLOT_STYLE["accent"])
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")

    # Alternate row shading + colour-code Az std column
    az_col_idx = n_cols - 1
    for i, row in enumerate(seg_data):
        bg = PLOT_STYLE["panel_alt"] if i % 2 == 0 else PLOT_STYLE["panel"]
        for j in range(n_cols):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(bg)
            cell.get_text().set_color(PLOT_STYLE["text"])

        az_std_v = _finite([row.get(f"az_ch{c}") for c in range(n_ch)])
        az_std_val = float(np.std(az_std_v)) if len(az_std_v) >= 2 else float("nan")
        if not math.isnan(az_std_val):
            hi_col = (PLOT_STYLE["ok"]   if az_std_val < 10  else
                      PLOT_STYLE["warn"] if az_std_val < 25  else
                      PLOT_STYLE["err"])
            tbl[(i + 1, az_col_idx)].set_facecolor(hi_col)
            tbl[(i + 1, az_col_idx)].get_text().set_color("white")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _save(fig, cfg.DRIVE_PLOTS / f"thesis_segment_table_{file_name}.png")
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass
    plt.close(fig)

def plot_per_channel_enhanced(
    per_file_results: list,
    cfg=None,
    save: bool = True,
    file_name: str = "Per-Channel Analysis Dashboard",
):
    """
    Per-channel analysis dashboard.

    Layout (4 rows × 3 cols = 12 panels):
      Row 0:  Detection timeline  ×3  (one per mic, shared y-axis)
      Row 1:  Localisation (x,y)  ×3  (colour = confidence, same colour scale)
      Row 2:  Cross-channel RMS comparison  |  Azimuth agreement  |  Track legend
      Row 3:  Segment table spanning full width

    Parameters
    ──────────
    per_file_results : list of dicts as produced by comprehensive_pipeline_test()
                       Each dict must have keys: wav_path, mic_label, segments,
                       confirmed (list of KalmanTrack objects).
    cfg              : Config (uses global config if None)
    save             : write PNG to cfg.DRIVE_PLOTS
    """
    from .config import config as _cfg
    cfg = cfg or _cfg

    n_ch = len(per_file_results)           # should be 3
    if n_ch == 0:
        return

    # ── Collect shared quantities ─────────────────────────────────────────
    # Flatten all segments from all channels for the table and cross-channel metrics
    n_segs = max(len(r["segments"]) for r in per_file_results)
    seg_data = []   # list of dicts {seg, t, ch0, ch1, ch2}  per segment index
    for seg_i in range(n_segs):
        row = {"seg": seg_i + 1, "t": None}
        for ch_i, r in enumerate(per_file_results):
            segs = r["segments"]
            if seg_i < len(segs):
                s = segs[seg_i]
                row[f"prob_ch{ch_i}"]  = s.get("prob", float("nan"))
                row[f"cnn_ch{ch_i}"]   = s.get("cnn_probability", float("nan"))
                row[f"rms_ch{ch_i}"]   = s.get("rms_db", float("nan"))
                row[f"az_ch{ch_i}"]    = (s["loc"]["azimuth_deg"]
                                           if s.get("loc") else float("nan"))
                if row["t"] is None:
                    row["t"] = s.get("t_start", seg_i * cfg.TARGET_DURATION)
            else:
                for key in (f"prob_ch{ch_i}", f"cnn_ch{ch_i}",
                             f"rms_ch{ch_i}", f"az_ch{ch_i}"):
                    row[key] = float("nan")
        seg_data.append(row)

    # ── Global track-ID map:  re-number all confirmed tracks continuously ─
    # so Track #6 / #9 / #12 from separate per-file trackers become #1, #2 …
    global_tracks = []
    for r in per_file_results:
        global_tracks.extend(r.get("confirmed", []))
    track_id_map = {t.track_id: i + 1 for i, t in enumerate(global_tracks)}

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 20), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(
        f"Per-Channel Analysis Dashboard - {file_name}",
        fontsize=18, color=PLOT_STYLE["accent"], fontweight="bold", y=0.995,
    )

    gs_outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.0, 3.0, 2.5],
        hspace=0.52,
    )

    # Row 0 - Detection timelines
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, n_ch, subplot_spec=gs_outer[0],
                                               wspace=0.30)
    # Row 1 - Localisation scatter
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, n_ch, subplot_spec=gs_outer[1],
                                               wspace=0.35)
    # Row 2 - Cross-channel metrics
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[2],
                                               wspace=0.38)

    all_axes = []

    # ── Row 0: Detection timelines ────────────────────────────────────────
    det_axes = []
    for ch_i, r in enumerate(per_file_results):
        ax = fig.add_subplot(gs_row0[ch_i])
        all_axes.append(ax)
        det_axes.append(ax)
        segs = r["segments"]
        ts   = [s["t_start"] for s in segs]
        prbs = [s["prob"]     for s in segs]
        cnns = [s.get("cnn_probability", float("nan")) for s in segs]
        cols = [PLOT_STYLE["ok"] if s["detected"] else PLOT_STYLE["err"] for s in segs]

        ax.bar(ts, prbs, width=cfg.TARGET_DURATION * 0.78,
               color=cols, alpha=0.52, label="Hybrid")
        ax.plot(ts, cnns, "o-", color=MIC_COLORS[ch_i],
                ms=4, lw=1.6, label="CNN", zorder=5)
        ax.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"],
                   lw=1.4, ls="--", label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
        ax.set_ylim(0, 1.10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability" if ch_i == 0 else "")
        mic_name = Path(r["wav_path"]).stem
        ax.set_title(f"Mic {ch_i+1}  -  {mic_name[:28]}")
        if ch_i == 0:
            ax.legend(fontsize=8, facecolor=PLOT_STYLE["panel_alt"],
                      edgecolor=PLOT_STYLE["spine"])

    # Share y-axis across detection timelines
    for ax in det_axes[1:]:
        ax.sharey(det_axes[0])
        ax.set_ylabel("")

    # ── Row 1: Localisation scatter ───────────────────────────────────────
    # Compute colour scale range across all channels together
    all_probs_loc = []
    for r in per_file_results:
        all_probs_loc.extend([s["prob"] for s in r["segments"] if s.get("loc")])
    vmin_loc = min(all_probs_loc) if all_probs_loc else 0.0
    vmax_loc = max(all_probs_loc) if all_probs_loc else 1.0

    loc_axes = []
    sc_last = None
    for ch_i, r in enumerate(per_file_results):
        ax = fig.add_subplot(gs_row1[ch_i])
        all_axes.append(ax)
        loc_axes.append(ax)
        segs_with_loc = [s for s in r["segments"] if s.get("loc")]

        if segs_with_loc:
            xs    = [float(np.asarray(s["loc"]["xy_position"])[0]) for s in segs_with_loc]
            ys    = [float(np.asarray(s["loc"]["xy_position"])[1]) for s in segs_with_loc]
            confs = [s["prob"] for s in segs_with_loc]
            ts    = [s["t_start"] for s in segs_with_loc]
            sc_last = ax.scatter(xs, ys, c=confs, cmap="plasma",
                                 vmin=vmin_loc, vmax=vmax_loc,
                                 s=55, zorder=4, edgecolors=PLOT_STYLE["bg"], lw=0.4)
            for x, y, t in zip(xs, ys, ts):
                ax.annotate(f"{t:.0f}s", (x, y),
                            textcoords="offset points", xytext=(3, 3),
                            fontsize=6, color=PLOT_STYLE["muted"])
        else:
            ax.text(0.5, 0.5, "No loc data", ha="center", va="center",
                    transform=ax.transAxes, color=PLOT_STYLE["muted"], fontsize=10)

        # Mic array
        mics = np.asarray(cfg.MIC_POSITIONS, dtype=float)
        ax.scatter(mics[:, 0], mics[:, 1], marker="^",
                   color=PLOT_STYLE["warn"], s=80, zorder=6, label="Mics")
        centre = np.asarray(cfg.ARRAY_CENTER, dtype=float)
        ax.plot(centre[0], centre[1], "+", color=PLOT_STYLE["text"],
                ms=9, mew=1.3, zorder=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)" if ch_i == 0 else "")
        ax.set_title(f"Mic {ch_i+1}  Localisation (x, y)")
        ax.legend(fontsize=8, facecolor=PLOT_STYLE["panel_alt"],
                  edgecolor=PLOT_STYLE["spine"])

    # Single shared colorbar for row 1
    if sc_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.555, 0.012, 0.155])
        cb = fig.colorbar(sc_last, cax=cbar_ax)
        cb.set_label("Confidence", color=PLOT_STYLE["text"], fontsize=9)
        cb.ax.yaxis.set_tick_params(color=PLOT_STYLE["text"], labelsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=PLOT_STYLE["text"])
        cb.outline.set_edgecolor(PLOT_STYLE["spine"])

    # ── Row 2: Cross-channel metrics ──────────────────────────────────────

    # - Panel A: Cross-channel RMS comparison -
    ax_rms = fig.add_subplot(gs_row2[0])
    all_axes.append(ax_rms)
    ts_plot = [row["t"] for row in seg_data]
    w = cfg.TARGET_DURATION * 0.24
    offsets = [-w, 0, w]
    for ch_i in range(n_ch):
        rms_vals = [row.get(f"rms_ch{ch_i}", float("nan")) for row in seg_data]
        rms_clean = [v if not math.isnan(v) else 0 for v in rms_vals]
        ax_rms.bar([t + offsets[ch_i] for t in ts_plot], rms_clean,
                   width=w * 0.92, color=MIC_COLORS[ch_i], alpha=0.72,
                   label=f"Mic {ch_i+1}")
    ax_rms.set_xlabel("Time (s)")
    ax_rms.set_ylabel("RMS (dB)")
    ax_rms.set_title("RMS per Segment - All Mics")
    ax_rms.legend(fontsize=9, facecolor=PLOT_STYLE["panel_alt"],
                  edgecolor=PLOT_STYLE["spine"])

    # - Panel B: Azimuth agreement (std across channels per segment) -
    ax_az = fig.add_subplot(gs_row2[1])
    all_axes.append(ax_az)
    az_stds, az_means = [], []
    for row in seg_data:
        az_vals = _finite([row.get(f"az_ch{c}") for c in range(n_ch)])
        az_stds.append(float(np.std(az_vals))  if len(az_vals) >= 2 else float("nan"))
        az_means.append(float(np.mean(az_vals)) if len(az_vals) >= 1 else float("nan"))

    std_clean  = [v if not math.isnan(v) else 0 for v in az_stds]
    mean_clean = [v if not math.isnan(v) else 0 for v in az_means]

    bar_cols_az = [
        PLOT_STYLE["ok"]   if (not math.isnan(v) and v < 10) else
        PLOT_STYLE["warn"] if (not math.isnan(v) and v < 25) else
        PLOT_STYLE["err"]
        for v in az_stds
    ]
    ax_az.bar(ts_plot, std_clean, width=cfg.TARGET_DURATION * 0.72,
              color=bar_cols_az, alpha=0.70, label="Az std (°)")
    ax2_az = ax_az.twinx()
    ax2_az.plot(ts_plot, mean_clean, "D--", color=PLOT_STYLE["purple"],
                ms=5, lw=1.5, label="Mean az (°)")
    ax2_az.set_ylabel("Mean azimuth (°)", color=PLOT_STYLE["purple"])
    ax2_az.tick_params(axis="y", colors=PLOT_STYLE["purple"], labelsize=10)
    for spine in ax2_az.spines.values():
        spine.set_color(PLOT_STYLE["spine"])

    # Legend combining both axes
    h1, l1 = ax_az.get_legend_handles_labels()
    h2, l2 = ax2_az.get_legend_handles_labels()
    ax_az.legend(h1 + h2, l1 + l2, fontsize=9,
                 facecolor=PLOT_STYLE["panel_alt"], edgecolor=PLOT_STYLE["spine"])
    ax_az.set_xlabel("Time (s)")
    ax_az.set_ylabel("Az std across mics (°)")
    ax_az.set_title("Azimuth Agreement  (↓ std = ↑ consistency)")

    # Colour legend chips
    for label, col in [("< 10° (good)", PLOT_STYLE["ok"]),
                        ("10–25° (moderate)", PLOT_STYLE["warn"]),
                        ("> 25° (poor)", PLOT_STYLE["err"])]:
        ax_az.bar([], [], color=col, alpha=0.70, label=label)

    # - Panel C: Continuous track legend -
    ax_trk = fig.add_subplot(gs_row2[2])
    all_axes.append(ax_trk)
    ax_trk.set_facecolor(PLOT_STYLE["panel"])
    ax_trk.axis("off")
    ax_trk.set_title("Confirmed Track Summary", fontsize=12,
                     color=PLOT_STYLE["text"], fontweight="bold")

    cmap_fn  = plt.get_cmap("plasma")
    n_global = len(global_tracks)
    y_pos = 0.95
    ax_trk.text(0.05, y_pos, "Global#  Channel  Orig ID  Pts  Az range (°)",
                transform=ax_trk.transAxes, fontsize=9,
                color=PLOT_STYLE["text_soft"], fontstyle="italic")
    y_pos -= 0.06

    for t_i, (track, result) in enumerate(
            [(t, r) for r in per_file_results for t in r.get("confirmed", [])]):
        global_num = track_id_map.get(track.track_id, t_i + 1)
        ch_idx = next(
            (i for i, r in enumerate(per_file_results)
             if any(tr.track_id == track.track_id for tr in r.get("confirmed", []))),
            0,
        )
        pts = np.array(track.positions) if track.positions else np.zeros((0, 2))
        col = cmap_fn(t_i / max(n_global - 1, 1))

        # Azimuth range from positions (angle from origin)
        if len(pts) >= 1:
            angles = [math.degrees(math.atan2(p[1], p[0])) % 360 for p in pts]
            az_str = f"{min(angles):.0f}–{max(angles):.0f}"
        else:
            az_str = "-"

        patch = mpatches.Patch(color=col, alpha=0.85)
        ax_trk.text(
            0.05, y_pos,
            f"  #{global_num:>2}      Mic {ch_idx+1}      "
            f"#{track.track_id}     {len(pts)}pt   {az_str}",
            transform=ax_trk.transAxes, fontsize=9.5,
            color=PLOT_STYLE["text"],
        )
        ax_trk.add_patch(mpatches.FancyBboxPatch(
            (0.01, y_pos - 0.01), 0.025, 0.045,
            boxstyle="round,pad=0.002",
            transform=ax_trk.transAxes,
            facecolor=col, alpha=0.85, zorder=4,
        ))
        y_pos -= 0.085
        if y_pos < 0.05:
            break

    if n_global == 0:
        ax_trk.text(0.5, 0.5, "No confirmed tracks", ha="center", va="center",
                    transform=ax_trk.transAxes, color=PLOT_STYLE["muted"], fontsize=11)

    # Apply axis style to non-table axes (table row removed from this figure)
    _apply_style(fig, all_axes)

    plt.tight_layout(rect=[0, 0, 0.915, 0.992])

    if save:
        # save with the file name of the processed audio file name for easier reference
        _save(fig, cfg.DRIVE_PLOTS / f"thesis_per_channel_enhanced_{file_name}.png")
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass
    plt.close(fig)

    # ── Separate segment table figure ─────────────────────────────────────
    _plot_segment_table(seg_data, n_ch, cfg, save=save, file_name=file_name)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 - Combined 3-channel analysis
# ═════════════════════════════════════════════════════════════════════════════

def plot_combined_3ch_analysis(
    full_channels: list,
    per_file_results: list,
    single_result: dict,
    cfg=None,
    save: bool = True,
    file_name: str = "",
):
    """
    Combined 3-channel analysis.

    Layout  (2 rows × 3 cols  +  1 wide bottom row):
      [0,0]  TDOA cross-correlation: Mic1 vs Mic2
      [0,1]  TDOA cross-correlation: Mic1 vs Mic3
      [0,2]  TDOA cross-correlation: Mic2 vs Mic3
      [1,0]  Confidence-weighted azimuth rose (all detections)
      [1,1]  Single fused localisation map  (all segment positions)
      [1,2]  Fused detection timeline (mean prob across mics per segment)
      [2, :] Shared Kalman trajectory (all-mic, re-numbered)

    Parameters
    ──────────
    full_channels    : list of 3 full-length np.ndarray (from load_3ch_full)
    per_file_results : same list passed to plot_per_channel_enhanced
    single_result    : dict returned by run_pipeline (single-drone)
    cfg              : Config
    save             : write PNG to cfg.DRIVE_PLOTS
    """
    from .config import config as _cfg
    cfg = cfg or _cfg

    if len(full_channels) < 3 or len(per_file_results) < 3:
        print("⚠  plot_combined_3ch_analysis requires exactly 3 channels.")
        return

    n_segs = max(len(r["segments"]) for r in per_file_results)

    # ── Global confirmed tracks (re-numbered) ────────────────────────────
    global_tracks = []
    for r in per_file_results:
        global_tracks.extend(r.get("confirmed", []))
    track_id_map = {t.track_id: i + 1 for i, t in enumerate(global_tracks)}

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor=PLOT_STYLE["bg"])
    fig.suptitle(
        f"Combined 3-Channel Analysis - {file_name}",
        fontsize=18, color=PLOT_STYLE["accent"], fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[2.8, 3.0, 3.5],
        hspace=0.48, wspace=0.35,
    )

    all_axes = []

    # ── Row 0: TDOA cross-correlations ────────────────────────────────────
    mic_pairs = [(0, 1), (0, 2), (1, 2)]
    sr = cfg.SR

    for col_i, (i, j) in enumerate(mic_pairs):
        ax = fig.add_subplot(gs[0, col_i])
        all_axes.append(ax)
        lags_ms, cc = _cc_lags(full_channels[i], full_channels[j], sr,
                                max_lag_ms=8.0)
        peak_idx = int(np.argmax(cc))
        peak_lag = float(lags_ms[peak_idx])
        peak_val = float(cc[peak_idx])

        ax.fill_between(lags_ms, cc, alpha=0.22,
                        color=MIC_COLORS[i])
        ax.plot(lags_ms, cc, lw=1.5, color=MIC_COLORS[i])
        ax.axvline(peak_lag, color=PLOT_STYLE["warn"], lw=1.8, ls="--",
                   label=f"Peak {peak_lag:+.2f} ms")
        ax.axvline(0, color=PLOT_STYLE["grid"], lw=0.8, ls=":")
        ax.axhline(0, color=PLOT_STYLE["grid"], lw=0.6)

        # Convert lag → implied distance difference
        c_sound = 343.0
        dd_m = abs(peak_lag / 1000.0) * c_sound
        ax.text(0.97, 0.95,
                f"TDOA: {peak_lag:+.3f} ms\nΔd ≈ {dd_m:.2f} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=PLOT_STYLE["text"],
                bbox=dict(facecolor=PLOT_STYLE["panel_alt"],
                          edgecolor=PLOT_STYLE["spine"],
                          boxstyle="round,pad=0.3"))
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Normalised cross-correlation" if col_i == 0 else "")
        ax.set_title(f"TDOA  Mic {i+1} ↔ Mic {j+1}")
        ax.legend(fontsize=9, facecolor=PLOT_STYLE["panel_alt"],
                  edgecolor=PLOT_STYLE["spine"])

    # ── Row 1, col 0: Confidence-weighted azimuth rose ────────────────────
    ax_pol = fig.add_subplot(gs[1, 0], projection="polar")
    ax_pol.set_facecolor(PLOT_STYLE["panel"])
    ax_pol.tick_params(colors=PLOT_STYLE["text"])

    all_az, all_conf = [], []
    for r in per_file_results:
        for s in r["segments"]:
            if s.get("loc") is not None:
                all_az.append(float(s["loc"]["azimuth_deg"]))
                all_conf.append(float(s.get("prob", 0.5)))

    if all_az:
        # Weight histogram by confidence
        rads = np.radians([-a for a in all_az])   # N-up, clockwise
        bins = 36
        edges = np.linspace(-np.pi, np.pi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]

        # Confidence-weighted counts
        w_counts = np.zeros(bins)
        for r, c in zip(rads, all_conf):
            idx = int((r + np.pi) / (2 * np.pi / bins))
            idx = max(0, min(bins - 1, idx))
            w_counts[idx] += c

        max_w = w_counts.max() if w_counts.max() > 0 else 1.0
        colors_pol = plt.cm.plasma(w_counts / max_w)

        ax_pol.bar(centers, w_counts, width=width, alpha=0.85,
                   color=colors_pol, edgecolor=PLOT_STYLE["bg"], lw=0.3)

        # Dominant direction annotation
        dom = int(np.argmax(w_counts))
        dom_deg = float(np.degrees(-centers[dom])) % 360
        ax_pol.annotate(
            f"{dom_deg:.0f}°",
            xy=(centers[dom], w_counts[dom]),
            xytext=(centers[dom], w_counts[dom] * 1.28),
            ha="center", va="center", fontsize=8,
            color=PLOT_STYLE["accent"],
        )
    else:
        theta = np.linspace(0, 2 * np.pi, 60)
        ax_pol.plot(theta, np.ones_like(theta) * 0.05,
                    color=PLOT_STYLE["muted"], lw=0.8, ls="--")
        ax_pol.text(0, 0, "no data", ha="center", va="center",
                    color=PLOT_STYLE["muted"], fontsize=9)

    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_title("Confidence-weighted\nAzimuth Rose  (all mics)",
                     color=PLOT_STYLE["accent"], pad=12, fontsize=11)
    ax_pol.grid(color=PLOT_STYLE["grid"], alpha=0.5)

    # ── Row 1, col 1: Fused localisation map ─────────────────────────────
    ax_map = fig.add_subplot(gs[1, 1])
    all_axes.append(ax_map)

    all_loc_xs, all_loc_ys, all_loc_confs = [], [], []
    for ch_i, r in enumerate(per_file_results):
        for s in r["segments"]:
            if s.get("loc"):
                xy = np.asarray(s["loc"]["xy_position"], dtype=float)
                all_loc_xs.append(float(xy[0]))
                all_loc_ys.append(float(xy[1]))
                all_loc_confs.append(float(s.get("prob", 0.5)))

    if all_loc_xs:
        sc_map = ax_map.scatter(
            all_loc_xs, all_loc_ys,
            c=all_loc_confs, cmap="plasma", vmin=0, vmax=1,
            s=50, zorder=4, alpha=0.80,
            edgecolors=PLOT_STYLE["bg"], lw=0.35,
        )
        # Mean fused position (confidence-weighted centroid)
        w_arr = np.array(all_loc_confs)
        w_arr /= w_arr.sum() + 1e-10
        cx = float(np.dot(w_arr, all_loc_xs))
        cy = float(np.dot(w_arr, all_loc_ys))
        ax_map.scatter(cx, cy, marker="*", s=280, color=PLOT_STYLE["warn"],
                       zorder=8, edgecolors="white", lw=0.8,
                       label=f"Fused centroid\n({cx:.1f}, {cy:.1f}) m")

        # Colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        try:
            div = make_axes_locatable(ax_map)
            cax = div.append_axes("right", size="4%", pad=0.05)
            cb  = fig.colorbar(sc_map, cax=cax)
            cb.set_label("Confidence", color=PLOT_STYLE["text"], fontsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=PLOT_STYLE["text"],
                     fontsize=7)
            cb.outline.set_edgecolor(PLOT_STYLE["spine"])
        except Exception:
            pass

    # Single-drone pipeline result star
    if single_result and single_result.get("drones"):
        xy_sd = np.asarray(single_result["drones"][0]["xy_position"], dtype=float)
        ax_map.scatter(float(xy_sd[0]), float(xy_sd[1]),
                       marker="P", s=240, color=PLOT_STYLE["err"],
                       zorder=9, edgecolors="white", lw=0.8,
                       label=f"Pipeline az={single_result['drones'][0].get('azimuth_deg',0):.1f}°")

    mics = np.asarray(cfg.MIC_POSITIONS, dtype=float)
    ax_map.scatter(mics[:, 0], mics[:, 1], marker="^",
                   color=PLOT_STYLE["warn"], s=90, zorder=10, label="Mics")
    centre = np.asarray(cfg.ARRAY_CENTER, dtype=float)
    ax_map.plot(centre[0], centre[1], "+", color=PLOT_STYLE["text"],
                ms=10, mew=1.5, zorder=11)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Fused Localisation Map  (all mics + all segments)")
    ax_map.legend(fontsize=8, facecolor=PLOT_STYLE["panel_alt"],
                  edgecolor=PLOT_STYLE["spine"])

    # ── Row 1, col 2: Fused detection timeline ────────────────────────────
    ax_fused = fig.add_subplot(gs[1, 2])
    all_axes.append(ax_fused)

    # Average prob per segment across channels
    ts_fused, mean_probs, std_probs = [], [], []
    for seg_i in range(n_segs):
        probs_seg = []
        t_s = None
        for r in per_file_results:
            if seg_i < len(r["segments"]):
                probs_seg.append(r["segments"][seg_i]["prob"])
                if t_s is None:
                    t_s = r["segments"][seg_i]["t_start"]
        if t_s is not None and probs_seg:
            ts_fused.append(t_s)
            mean_probs.append(float(np.mean(probs_seg)))
            std_probs.append(float(np.std(probs_seg)))

    if ts_fused:
        mean_arr = np.array(mean_probs)
        std_arr  = np.array(std_probs)
        cols_fused = [PLOT_STYLE["ok"] if p >= cfg.DETECTION_THRESHOLD
                      else PLOT_STYLE["err"] for p in mean_probs]
        ax_fused.bar(ts_fused, mean_probs, width=cfg.TARGET_DURATION * 0.75,
                     color=cols_fused, alpha=0.55, label="Mean hybrid prob")
        ax_fused.fill_between(ts_fused,
                              mean_arr - std_arr,
                              mean_arr + std_arr,
                              alpha=0.18, color=PLOT_STYLE["accent"],
                              label="±1 std across mics")
        ax_fused.plot(ts_fused, mean_probs, "o-",
                      color=PLOT_STYLE["accent"], ms=4, lw=1.6)
        ax_fused.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["warn"],
                         lw=1.5, ls="--",
                         label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
        ax_fused.set_ylim(0, 1.10)

    ax_fused.set_xlabel("Time (s)")
    ax_fused.set_ylabel("Probability")
    ax_fused.set_title("Fused Detection Timeline  (mean ± std, all mics)")
    ax_fused.legend(fontsize=9, facecolor=PLOT_STYLE["panel_alt"],
                    edgecolor=PLOT_STYLE["spine"])

    # ── Row 2: Shared Kalman trajectory (all-mic, re-numbered) ───────────
    ax_traj = fig.add_subplot(gs[2, :])
    all_axes.append(ax_traj)

    cmap_fn = plt.get_cmap("plasma")
    n_global = len(global_tracks)
    mics_np  = np.asarray(cfg.MIC_POSITIONS, dtype=float)
    ax_traj.scatter(mics_np[:, 0], mics_np[:, 1], marker="^",
                    s=220, c=PLOT_STYLE["warn"], zorder=10, label="Mic array")
    for i, mic in enumerate(mics_np):
        ax_traj.annotate(f"M{i+1}", mic, textcoords="offset points",
                         xytext=(5, 4), fontsize=9, color=PLOT_STYLE["text"])

    all_xs = list(mics_np[:, 0])
    all_ys = list(mics_np[:, 1])

    # Group tracks by original channel for origin labelling
    ch_label_map = {}
    for ch_i, r in enumerate(per_file_results):
        for tr in r.get("confirmed", []):
            ch_label_map[tr.track_id] = ch_i + 1

    for t_i, track in enumerate(global_tracks):
        pts = np.array(track.positions) if track.positions else np.zeros((0, 2))
        if len(pts) == 0:
            continue
        gnum = track_id_map.get(track.track_id, t_i + 1)
        col  = cmap_fn(t_i / max(n_global - 1, 1))
        ch   = ch_label_map.get(track.track_id, "?")
        lbl  = f"Track #{gnum}  (Mic {ch},  {len(pts)} pt{'s' if len(pts)>1 else ''})"

        if len(pts) == 1:
            ax_traj.scatter(*pts[0], s=180, c=[col], marker="D",
                            zorder=8, edgecolors="white", lw=0.8, label=lbl)
        else:
            n_seg_pts = len(pts)
            for k in range(n_seg_pts - 1):
                seg_col = cmap_fn(k / max(n_seg_pts - 2, 1))
                ax_traj.plot(pts[k:k+2, 0], pts[k:k+2, 1], "-",
                             color=seg_col, lw=2.5, zorder=5)
            ax_traj.scatter(*pts[0],  s=110, c=[cmap_fn(0.0)], marker=">", zorder=9)
            ax_traj.scatter(*pts[-1], s=110, c=[cmap_fn(1.0)], marker="s",
                            zorder=9, label=lbl)

            # Velocity arrow at last point
            if len(pts) >= 2:
                dv = pts[-1] - pts[-2]
                norm = np.linalg.norm(dv) + 1e-8
                scale = 0.6
                ax_traj.annotate(
                    "", xy=pts[-1] + dv / norm * scale, xytext=pts[-1],
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.6, mutation_scale=14),
                    zorder=11,
                )

        # Uncertainty circle at last point
        try:
            cr = track.uncertainty_radius()
            if 0 < cr < 25 and not math.isnan(cr):
                ax_traj.add_patch(plt.Circle(pts[-1], cr,
                                             color=col, alpha=0.10, fill=True))
        except Exception:
            pass

        all_xs.extend(pts[:, 0])
        all_ys.extend(pts[:, 1])

    if all_xs and all_ys:
        pad = max(1.5, (max(all_xs) - min(all_xs)) * 0.18)
        ax_traj.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax_traj.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X (m)", fontsize=12)
    ax_traj.set_ylabel("Y (m)", fontsize=12)
    ax_traj.set_title(
        f"Shared Kalman Trajectories  -  {n_global} confirmed track(s) across all mics  "
        f"(start ▶  end ■  velocity →)",
        fontsize=13,
    )
    ax_traj.legend(fontsize=9, facecolor=PLOT_STYLE["panel_alt"],
                   edgecolor=PLOT_STYLE["spine"], loc="upper right")

    # Apply style to all regular (non-polar) axes
    _apply_style(fig, all_axes)

    plt.tight_layout(rect=[0, 0, 1.0, 0.993])

    if save:
        _save(fig, cfg.DRIVE_PLOTS / f"thesis_combined_3ch_analysis_{file_name}.png")
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass
    plt.close(fig)
