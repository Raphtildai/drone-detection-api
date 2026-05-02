# -*- coding: utf-8 -*-
"""
dunakeszi_pipeline_analysis.py
══════════════════════════════════════════════════════════════════════════════
End-to-end Dunakeszi 2025-10-20 dataset analysis for Google Colab.

Covers the full three-stage pipeline:
  Stage 1  dunakeszi_ground_truth_fixed.py   → ground truth JSON/CSV
  Stage 2  dunakeszi_segment_extractor_fixed.py → per-segment WAVs + label JSONs
  Stage 3  prepare_dunakeszi_for_pipeline.py  → pipeline-ready format
  Stage 4  drone_detection inference          → detect + localize + evaluate

Quick start (Colab cells at the bottom of this file).
══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════════════════════
# §0  Imports
# ══════════════════════════════════════════════════════════════════════════════

import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# §1  Config patch  — must be applied before importing drone_detection
# ══════════════════════════════════════════════════════════════════════════════

def patch_config_for_dunakeszi(cfg, max_dist: float = 100.0):
    """
    Apply the required config overrides for the Dunakeszi BK-6-E array.

    Parameters
    ──────────
    cfg      : your drone_detection.config singleton
    max_dist : MAX_LOCALIZATION_DIST in metres.
               100 m covers all orbit radii (5, 10, 30, 60 m) plus margin.
               Set to 350 m for the long-range show_10 segment.

    Must be called BEFORE load_detection_model() / load_localization_model().
    """
    # BK-6-E array geometry (baseline ≈ 2500 mm, gp2 config)
    try:
        cfg.set_array_geometry("gp2")
        print("  ✅ cfg.set_array_geometry('gp2')  — BK-6-E baseline applied")
    except AttributeError:
        # Fallback: set MIC_POSITIONS directly from mic_array_geometry.json
        # BK-6-E channels 9,10,11 (E, H, B directions); approximate mic offsets
        # for the gp2 config at 2500mm baseline.
        print("  ⚠️  set_array_geometry() not found — setting MIC_POSITIONS manually")
        cfg.MIC_POSITIONS = np.array([
            [ 0.000,  0.000],
            [ 0.025,  0.000],
            [ 0.012,  0.022],
        ], dtype=np.float32)

    cfg.MAX_LOCALIZATION_DIST = float(max_dist)
    print(f"  ✅ cfg.MAX_LOCALIZATION_DIST = {max_dist:.1f} m")

    # Array centre is the BK-6-E GPS position in local XY
    # From mic_array_geometry.json: BK-6-E = (+9.6, -0.4) m from origin
    cfg.ARRAY_CENTER = [9.6, -0.4]
    print(f"  ✅ cfg.ARRAY_CENTER = {cfg.ARRAY_CENTER}  (BK-6-E local XY)")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# §2  Ground truth helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth_csv(gt_csv: str) -> List[Dict[str, Any]]:
    """
    Load ground_truth_segments.csv into a list of dicts.
    Parses JSON-encoded list columns (drones, start_coord, quality_flags).
    """
    rows = []
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("drones", "start_coord", "end_coord", "quality_flags"):
                if row.get(col):
                    try:
                        row[col] = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            for col in ("id", "n_drones", "altitude_m", "duration_s",
                        "within_session_offset_s", "local_start_s",
                        "smpte_start_s", "onset_from_rec_s"):
                if row.get(col) not in (None, ""):
                    try:
                        row[col] = float(row[col])
                    except ValueError:
                        pass
            for col in ("mems_available", "bk_available"):
                row[col] = str(row.get(col, "")).strip().lower() == "true"
            rows.append(row)
    return rows


def load_manifest(manifest_json: str) -> List[Dict[str, Any]]:
    """Load the extractor manifest.json output."""
    with open(manifest_json) as f:
        return json.load(f)


def load_pipeline_labels(labels_csv: str) -> List[Dict[str, Any]]:
    """Load the pipeline-ready labels.csv produced by prepare_dunakeszi."""
    rows = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("azimuth_deg", "distance_m", "height_m",
                        "original_bearing_deg", "n_drones",
                        "radius_m", "speed_mps", "duration_s"):
                v = row.get(col)
                if v in (None, ""):
                    row[col] = None          # always None, never ""
                else:
                    try:
                        row[col] = float(v)
                    except (ValueError, TypeError):
                        row[col] = None
            row["has_position"] = str(row.get("has_position", "")).strip().lower() == "true"
            rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# §3  Core inference loop over pipeline-ready segments
# ══════════════════════════════════════════════════════════════════════════════

def run_dunakeszi_inference(
    pipeline_dir: str,
    cfg,
    gt_csv:       Optional[str]  = None,
    splits:       Optional[List[str]] = None,
    multi_drone:  bool = False,
    verbose:      bool = True,
) -> List[Dict[str, Any]]:
    """
    Run detect() + localize() on every segment in pipeline_dir.

    Parameters
    ──────────
    pipeline_dir : output of prepare_dunakeszi_for_pipeline.py
                   contains *_ch0/1/2.wav + *_label.json + labels.csv
    cfg          : patched Config (call patch_config_for_dunakeszi first)
    gt_csv       : path to ground_truth_segments.csv for enrichment
    splits       : filter to ["train"] / ["val"] / ["test"] or None = all
    multi_drone  : use localize_multi_drone() when n_drones > 1
    verbose      : print per-segment results

    Returns
    ───────
    List of result dicts, one per segment, with keys:
      session_id, segment_id, session, maneuver_type, n_drones, split,
      detected, probability, cnn_probability, heuristic_probability,
      pred_azimuth_deg, pred_distance_m, pred_height_m,
      gt_azimuth_deg, gt_distance_m, gt_height_m,
      az_error_deg, dist_error_m, ht_error_m,
      rms_db, quality_flags, has_position
    """
    from drone_detection.inference import (
        detect, localize, load_detection_model, load_localization_model,
    )
    from drone_detection.multidrone import localize_multi_drone
    from drone_detection.audio_processing import AudioProcessor

    pdir = Path(pipeline_dir)
    labels_csv = pdir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found in {pdir}")

    labels = load_pipeline_labels(str(labels_csv))
    if splits:
        labels = [r for r in labels if r.get("split") in splits]

    # Optionally enrich with quality_flags from GT CSV
    gt_by_session: Dict[str, dict] = {}
    if gt_csv and Path(gt_csv).exists():
        gt_rows = load_ground_truth_csv(gt_csv)
        # Key by segment_id (int)
        gt_by_id = {int(r["id"]): r for r in gt_rows}
    else:
        gt_by_id = {}

    # Load models
    print("🔄 Loading models …")
    load_detection_model(cfg)
    can_localize = False
    try:
        load_localization_model(cfg)
        can_localize = True
    except FileNotFoundError:
        print("  ⚠️  No localization checkpoint — detection-only mode.")

    ap = AudioProcessor(cfg)
    results = []
    n = len(labels)

    print(f"\n{'─'*70}")
    print(f"  Running inference on {n} segment(s) from {pdir.name}")
    if splits:
        print(f"  Splits: {splits}")
    print(f"{'─'*70}")

    for i, row in enumerate(labels):
        sid    = row["session_id"]
        seg_id = row.get("segment_id")
        try:
            seg_id = int(float(seg_id)) if seg_id not in (None, "") else None
        except (ValueError, TypeError):
            seg_id = None

        ch_paths = [pdir / row[f"wav_ch{k}"] for k in range(3)]
        missing  = [p for p in ch_paths if not p.exists()]
        if missing:
            print(f"  ⚠️  {sid}: missing WAV(s) — skipped")
            continue

        # Load audio
        channels = [ap.pad_or_truncate(ap.load(str(p), mono=True)) for p in ch_paths]

        # RMS
        rms_db = float(20 * math.log10(
            float(np.sqrt(np.mean(np.asarray(channels[0]) ** 2))) + 1e-8
        ))

        # Detection
        det = detect(channels, cfg)

        # Localization
        pred_az = pred_dist = pred_ht = None
        if det["detected"] and can_localize:
            try:
                if multi_drone and row.get("n_drones", 1) > 1:
                    drones = localize_multi_drone(channels, cfg)
                    if drones:
                        pred_az   = drones[0]["azimuth_deg"]
                        pred_dist = drones[0]["distance_m"]
                        pred_ht   = None  # multi-drone doesn't return height
                else:
                    loc       = localize(channels, cfg)
                    pred_az   = loc["azimuth_deg"]
                    pred_dist = loc["distance_m"]
                    pred_ht   = loc.get("height_m")
            except Exception as exc:
                print(f"    ⚠️  localize() failed for {sid}: {exc}")

        # Ground truth
        gt_az   = row.get("azimuth_deg")
        gt_dist = row.get("distance_m")
        gt_ht   = row.get("height_m")
        has_pos = row.get("has_position", False)

        # Errors — only computed when GT distance > 0.5 m.
        # Hover/survey segments sit at d=0 (drone overhead); azimuth is
        # undefined there so errors would be meaningless noise.
        try:
            _gt_dist_val = float(gt_dist) if gt_dist not in (None, "") else 0.0
            _gt_ht_val   = float(gt_ht)   if gt_ht   not in (None, "") else None
        except (ValueError, TypeError):
            _gt_dist_val = 0.0
            _gt_ht_val   = None

        has_meaningful_pos = has_pos and _gt_dist_val > 0.5

        def _az_err(pred, gt):
            if pred is None or gt is None:
                return None
            diff = (pred - gt + 180) % 360 - 180
            return round(abs(diff), 2)

        az_err   = _az_err(pred_az, gt_az) if has_meaningful_pos else None
        dist_err = (round(abs(pred_dist - _gt_dist_val), 2)
                    if has_meaningful_pos and pred_dist is not None else None)
        ht_err   = (round(abs(pred_ht - _gt_ht_val), 2)
                    if has_meaningful_pos and pred_ht is not None
                    and _gt_ht_val is not None else None)

        # Quality flags from ground truth
        qflags = []
        if seg_id is not None and seg_id in gt_by_id:
            raw_flags = gt_by_id[seg_id].get("quality_flags", [])
            qflags = raw_flags if isinstance(raw_flags, list) else []

        result = {
            # identity
            "session_id":            sid,
            "segment_id":            seg_id,
            "session":               row.get("session", ""),
            "maneuver_type":         row.get("maneuver_type", ""),
            "flight_phase":          row.get("flight_phase", ""),
            "n_drones":              int(row.get("n_drones") or 1),
            "split":                 row.get("split", ""),
            "radius_m":              row.get("radius_m"),
            "speed_mps":             row.get("speed_mps"),
            "duration_s":            row.get("duration_s"),
            # detection
            "detected":              det["detected"],
            "probability":           round(det["probability"], 4),
            "cnn_probability":       round(det.get("cnn_probability", float("nan")), 4),
            "heuristic_probability": round(det.get("heuristic_probability", float("nan")), 4),
            # predicted localization
            "pred_azimuth_deg":      round(pred_az,   2) if pred_az   is not None else None,
            "pred_distance_m":       round(pred_dist, 2) if pred_dist is not None else None,
            "pred_height_m":         round(pred_ht,   2) if pred_ht   is not None else None,
            # ground truth  (coerce to float — guards against stray "" from CSV)
            "gt_azimuth_deg":        round(float(gt_az),   2) if gt_az   not in (None, "") else None,
            "gt_distance_m":         round(float(gt_dist), 2) if gt_dist not in (None, "") else None,
            "gt_height_m":           round(float(gt_ht),   2) if gt_ht   not in (None, "") else None,
            "has_position":          has_pos,
            # errors
            "az_error_deg":          az_err,
            "dist_error_m":          dist_err,
            "ht_error_m":            ht_err,
            # audio
            "rms_db":                round(rms_db, 2),
            # metadata
            "quality_flags":         qflags,
        }
        results.append(result)

        if verbose:
            icon = "🚁" if det["detected"] else "🌳"
            gt_str = (f"GT az={gt_az:.1f}° d={_gt_dist_val:.1f}m" if has_meaningful_pos and gt_az is not None else ("d=0 (hover)" if has_pos else "no-GT-pos"))
            pred_str = (f"pred az={pred_az:.1f}° d={pred_dist:.1f}m" if pred_az is not None else "no-pred")
            err_str  = (f"err az={az_err:.1f}° d={dist_err:.1f}m" if az_err is not None else "")
            print(
                f"  [{i+1:>3}/{n}] {icon} {sid:<42s} "
                f"p={det['probability']:.3f}  {gt_str}  {pred_str}  {err_str}"
            )

    _print_inference_summary(results, cfg)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §4  Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def _print_inference_summary(results: List[dict], cfg):
    n     = len(results)
    n_det = sum(r["detected"] for r in results)
    loc_results = [r for r in results
                   if r["detected"] and r["az_error_deg"] is not None]

    print(f"\n{'═'*70}")
    print(f"  DUNAKESZI INFERENCE SUMMARY")
    print(f"{'═'*70}")
    print(f"  Segments run          : {n}")
    print(f"  Detected (≥ thr)      : {n_det}  ({100*n_det/max(n,1):.1f}%)")
    print(f"  Detection threshold   : {cfg.DETECTION_THRESHOLD:.3f}")
    print(f"  MAX_LOCALIZATION_DIST : {cfg.MAX_LOCALIZATION_DIST:.1f} m")

    if loc_results:
        az_errs   = [r["az_error_deg"]  for r in loc_results]
        dist_errs = [r["dist_error_m"]  for r in loc_results if r["dist_error_m"] is not None]
        ht_errs   = [r["ht_error_m"]    for r in loc_results if r["ht_error_m"]   is not None]
        print(f"\n  Localization (on {len(loc_results)} detected+positioned segments):")
        print(f"    Azimuth MAE   : {np.mean(az_errs):.2f}°  "
              f"(median {np.median(az_errs):.2f}°, max {max(az_errs):.1f}°)")
        if dist_errs:
            print(f"    Distance MAE  : {np.mean(dist_errs):.2f} m  "
                  f"(median {np.median(dist_errs):.2f} m)")
        if ht_errs:
            print(f"    Height MAE    : {np.mean(ht_errs):.2f} m  "
                  f"(median {np.median(ht_errs):.2f} m)")

    # Break down by session
    from collections import defaultdict
    by_session = defaultdict(list)
    for r in results:
        by_session[r["session"]].append(r)

    print(f"\n  Per-session breakdown:")
    for sess in sorted(by_session):
        rs     = by_session[sess]
        n_s    = len(rs)
        n_d    = sum(r["detected"] for r in rs)
        n_loc  = sum(1 for r in rs if r["az_error_deg"] is not None)
        az_m   = np.mean([r["az_error_deg"] for r in rs if r["az_error_deg"] is not None]) if n_loc else float("nan")
        nd_str = rs[0].get("n_drones", "?")
        print(f"    {sess:<12s}  n_drones={nd_str}  {n_d}/{n_s} det  "
              f"az_mae={az_m:.1f}°" if not math.isnan(az_m) else
              f"    {sess:<12s}  n_drones={nd_str}  {n_d}/{n_s} det  az_mae=—")

    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# §5  Plots
# ══════════════════════════════════════════════════════════════════════════════

_S = {
    "bg":     "#ffffff", "panel":  "#f8f9fa", "alt":    "#e9ecef",
    "accent": "#1565c0", "warn":   "#e65100", "ok":     "#2e7d32",
    "err":    "#c62828", "grid":   "#b0bec5", "text":   "#212121",
    "muted":  "#546e7a", "spine":  "#90a4ae", "purple": "#6a1b9a",
}
_CMAP = plt.get_cmap("plasma")
_SESSION_COLORS = {
    "show_1": "#1565c0", "show_2": "#0288d1", "show_5": "#00838f",
    "show_7": "#2e7d32", "show_8": "#558b2f", "show_9": "#f57f17",
    "show_11": "#e65100", "show_12": "#c62828", "show_13": "#6a1b9a",
    "show_14": "#880e4f", "show_15": "#4a148c", "show_10": "#37474f",
}

def _style(fig, axes):
    fig.patch.set_facecolor(_S["bg"])
    for ax in ([axes] if not hasattr(axes, "__iter__") else axes):
        if ax is None: continue
        ax.set_facecolor(_S["panel"])
        ax.tick_params(colors=_S["text"], labelcolor=_S["text"])
        ax.xaxis.label.set_color(_S["text"]); ax.yaxis.label.set_color(_S["text"])
        ax.title.set_color(_S["text"]); ax.title.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_color(_S["spine"]); sp.set_linewidth(0.8)
        ax.grid(color=_S["grid"], alpha=0.4, linewidth=0.7)

def _leg(ax):
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(_S["alt"])
        leg.get_frame().set_edgecolor(_S["spine"])
        for t in leg.get_texts(): t.set_color(_S["text"])

def _save(fig, path, dpi=180):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"  💾 {path}")

def _show(fig):
    try:
        from IPython.display import display; display(fig)
    except Exception:
        pass


# ── Plot 1: Detection probability per segment ─────────────────────────────────

def plot_detection_timeline(results: List[dict], cfg, save_path=None):
    """Bar chart of fused/CNN/heuristic probability per segment, coloured by detection."""
    fig, ax = plt.subplots(figsize=(max(14, len(results) * 0.45), 5))
    _style(fig, [ax])

    xs    = list(range(len(results)))
    fused = [r["probability"]           for r in results]
    cnn   = [r["cnn_probability"]       for r in results]
    heur  = [r["heuristic_probability"] for r in results]
    cols  = [_S["ok"] if r["detected"] else _S["err"] for r in results]

    ax.bar(xs, fused, color=cols, alpha=0.45, label="Fused", width=0.75)
    ax.plot(xs, fused, "o-",  color=_S["accent"], ms=4, lw=1.8, label="Fused")
    ax.plot(xs, cnn,   "s--", color=_S["purple"], ms=3, lw=1.2, label="CNN")
    ax.plot(xs, heur,  "^:",  color=_S["warn"],   ms=3, lw=1.2, label="Heuristic")
    ax.axhline(cfg.DETECTION_THRESHOLD, color=_S["err"], ls="--", lw=1.5,
               label=f"Threshold ({cfg.DETECTION_THRESHOLD:.2f})")

    labels = [f"{r['session_id'].split('_')[1][:3]}\n{r['maneuver_type'][:4]}"
              for r in results]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Segment"); ax.set_ylabel("Probability")
    ax.set_title("Detection probability per segment (Dunakeszi)")
    ax.legend(fontsize=9, loc="upper right"); _leg(ax)
    plt.tight_layout()
    _save(fig, save_path); _show(fig); plt.close(fig)


# ── Plot 2: Azimuth error per segment ─────────────────────────────────────────

def plot_azimuth_errors(results: List[dict], cfg, save_path=None):
    """Horizontal bar chart of azimuth MAE per segment, coloured by error magnitude."""
    loc_res = [r for r in results if r["az_error_deg"] is not None]
    if not loc_res:
        print("  ⚠️  No localization results to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(loc_res) * 0.38)))
    _style(fig, [ax])

    labels = [r["session_id"] for r in loc_res]
    errs   = [r["az_error_deg"] for r in loc_res]
    cols   = ["#2e7d32" if e < 30 else "#ba7517" if e < 60 else "#c62828" for e in errs]

    ax.barh(labels, errs, color=cols, alpha=0.85, height=0.65)
    ax.axvline(90, color=_S["muted"], ls="--", lw=1.2, label="Random baseline (90°)")
    ax.axvline(np.mean(errs), color=_S["accent"], ls="-.", lw=1.5,
               label=f"Mean {np.mean(errs):.1f}°")

    legend_handles = [
        mpatches.Patch(color="#2e7d32", label="Good (<30°)"),
        mpatches.Patch(color="#ba7517", label="Moderate (30–60°)"),
        mpatches.Patch(color="#c62828", label="Poor (>60°)"),
        plt.Line2D([0],[0], color=_S["muted"], ls="--", lw=1.2, label="Random (90°)"),
        plt.Line2D([0],[0], color=_S["accent"], ls="-.", lw=1.5,
                   label=f"Mean {np.mean(errs):.1f}°"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax.set_xlabel("Azimuth error (°)"); ax.set_title("Per-segment azimuth error (Dunakeszi)")
    ax.set_xlim(0, max(errs) * 1.12)
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, save_path); _show(fig); plt.close(fig)


# ── Plot 3: Predicted vs ground-truth azimuth ─────────────────────────────────

def plot_predicted_vs_gt(results: List[dict], save_path=None):
    """Scatter of predicted azimuth vs GT azimuth, coloured by session."""
    loc_res = [r for r in results
               if r["pred_azimuth_deg"] is not None and r["gt_azimuth_deg"] is not None]
    if not loc_res:
        print("  ⚠️  No paired pred/GT azimuth data.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    _style(fig, [ax])

    seen_sessions = {}
    for r in loc_res:
        sess = r["session"]
        col  = _SESSION_COLORS.get(sess, _S["muted"])
        label = sess if sess not in seen_sessions else "_nolegend_"
        seen_sessions[sess] = True
        ax.scatter(r["gt_azimuth_deg"], r["pred_azimuth_deg"],
                   c=[col], s=60, alpha=0.85, edgecolors="white",
                   linewidths=0.5, label=label)

    lim = (-185, 185)
    ax.plot(lim, lim, color=_S["muted"], ls="--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks(range(-180, 181, 45)); ax.set_yticks(range(-180, 181, 45))
    ax.set_xlabel("Ground-truth azimuth (°)")
    ax.set_ylabel("Predicted azimuth (°)")
    ax.set_title("Predicted vs GT azimuth — Dunakeszi")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left"); _leg(ax)
    plt.tight_layout()
    _save(fig, save_path); _show(fig); plt.close(fig)


# ── Plot 4: Error vs distance ──────────────────────────────────────────────────

def plot_error_vs_distance(results: List[dict], save_path=None):
    """Scatter of azimuth error vs GT distance, sized by n_drones."""
    loc_res = [r for r in results
               if r["az_error_deg"] is not None and r["gt_distance_m"] is not None]
    if not loc_res:
        print("  ⚠️  No data for error-vs-distance plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    _style(fig, [ax])

    dists = np.array([r["gt_distance_m"] for r in loc_res])
    errs  = np.array([r["az_error_deg"]  for r in loc_res])
    sizes = np.array([30 + r["n_drones"] * 20 for r in loc_res])
    cols  = [_SESSION_COLORS.get(r["session"], _S["muted"]) for r in loc_res]

    ax.scatter(dists, errs, s=sizes, c=cols, alpha=0.80,
               edgecolors="white", linewidths=0.5)
    ax.axhline(90, color=_S["muted"], ls="--", lw=1.2, label="Random baseline")
    ax.axhline(30, color="#2e7d32",  ls=":",  lw=1.0, label="Good threshold (30°)")

    # Trend line — only when distances have meaningful spread (not all zero)
    dist_range = dists.max() - dists.min() if len(dists) >= 4 else 0
    if len(dists) >= 4 and dist_range > 1.0:
        try:
            z = np.polyfit(dists, errs, 1)
            xl = np.linspace(dists.min(), dists.max(), 200)
            ax.plot(xl, np.polyval(z, xl), color=_S["accent"], lw=1.5, ls="-.",
                    label=f"Trend (slope {z[0]:+.2f}°/m)")
        except np.linalg.LinAlgError:
            pass  # degenerate data — skip trend line silently

    ax.set_xlabel("GT distance from array (m)")
    ax.set_ylabel("Azimuth error (°)")
    ax.set_title("Azimuth error vs distance (Dunakeszi)\n(marker size ∝ n_drones)")
    ax.legend(fontsize=9); _leg(ax)
    plt.tight_layout()
    _save(fig, save_path); _show(fig); plt.close(fig)


# ── Plot 5: Top-down position map ─────────────────────────────────────────────

def plot_position_map(results: List[dict], cfg, save_path=None):
    """
    Top-down map showing:
      - Mic array positions (BK-6-E, BK-6-W from mic_array_geometry.json)
      - GT positions (filled circles, coloured by session)
      - Predicted positions (crosses, matched colour)
      - Error lines connecting GT → prediction
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    _style(fig, [ax])

    # Mic array footprint (global origin + BK-6-E and BK-6-W positions)
    mic_positions = np.array([
        [9.6, -0.4],    # BK-6-E
        [-9.6, 0.4],    # BK-6-W
        [-0.85, -9.6],  # MEMS-S
        [0.69, 10.03],  # MEMS-N
    ])
    mic_labels = ["BK-6-E", "BK-6-W", "MEMS-S", "MEMS-N"]
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1],
               marker="^", s=220, c=_S["warn"], zorder=10)
    for m, lbl in zip(mic_positions, mic_labels):
        ax.annotate(lbl, m, textcoords="offset points", xytext=(6, 5),
                    fontsize=9, color=_S["text"])

    # GT and predicted positions
    seen = {}
    for r in results:
        if r["gt_azimuth_deg"] is None or r["gt_distance_m"] is None:
            continue
        sess = r["session"]
        col  = _SESSION_COLORS.get(sess, _S["muted"])

        # GT position: convert pipeline az + GT distance to XY
        # Pipeline azimuth is math angle (atan2(y,x)), 0°=East
        az_rad = math.radians(r["gt_azimuth_deg"])
        gx = r["gt_distance_m"] * math.cos(az_rad) + cfg.ARRAY_CENTER[0]
        gy = r["gt_distance_m"] * math.sin(az_rad) + cfg.ARRAY_CENTER[1]

        label = sess if sess not in seen else "_nolegend_"
        seen[sess] = True
        ax.scatter(gx, gy, s=80, c=[col], alpha=0.85, zorder=6,
                   edgecolors="white", linewidths=0.5, label=label)

        # Predicted position
        if r["pred_azimuth_deg"] is not None and r["pred_distance_m"] is not None:
            paz_rad = math.radians(r["pred_azimuth_deg"])
            px = r["pred_distance_m"] * math.cos(paz_rad) + cfg.ARRAY_CENTER[0]
            py = r["pred_distance_m"] * math.sin(paz_rad) + cfg.ARRAY_CENTER[1]
            ax.scatter(px, py, s=55, c=[col], marker="x", alpha=0.70,
                       zorder=7, linewidths=1.5)
            ax.plot([gx, px], [gy, py], "-", color=col, lw=0.8, alpha=0.45, zorder=5)

    ax.scatter([], [], marker="o", s=70, c="gray", label="GT position")
    ax.scatter([], [], marker="x", s=60, c="gray", linewidths=1.5, label="Predicted")
    ax.set_xlabel("X (m, East)"); ax.set_ylabel("Y (m, North)")
    ax.set_title("GT vs predicted positions — Dunakeszi (pipeline math angle convention)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left"); _leg(ax)
    plt.tight_layout()
    _save(fig, save_path); _show(fig); plt.close(fig)


# ── Plot 6: Summary dashboard ─────────────────────────────────────────────────

def plot_summary_dashboard(results: List[dict], cfg, save_path=None):
    """
    4-panel summary: detection timeline | azimuth error bars |
                     pred-vs-GT scatter | error-vs-distance
    """
    fig = plt.figure(figsize=(20, 14), facecolor=_S["bg"])
    fig.suptitle("Dunakeszi Pipeline Analysis — Summary Dashboard",
                 fontsize=16, color=_S["accent"], fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30)

    # ── [0] Detection timeline ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _style(fig, [ax0])
    xs    = list(range(len(results)))
    fused = [r["probability"] for r in results]
    cols  = [_S["ok"] if r["detected"] else _S["err"] for r in results]
    ax0.bar(xs, fused, color=cols, alpha=0.45, width=0.7)
    ax0.plot(xs, fused, "o-", color=_S["accent"], ms=3, lw=1.5, label="Fused")
    ax0.plot(xs, [r["cnn_probability"] for r in results], "s--",
             color=_S["purple"], ms=2, lw=1.0, label="CNN")
    ax0.axhline(cfg.DETECTION_THRESHOLD, color=_S["err"], ls="--", lw=1.2,
                label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax0.set_ylim(0, 1.08); ax0.set_xlabel("Segment index")
    ax0.set_ylabel("Probability"); ax0.set_title("Detection timeline")
    ax0.legend(fontsize=8); _leg(ax0)

    # ── [1] Azimuth error bars ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _style(fig, [ax1])
    loc_res = [r for r in results if r["az_error_deg"] is not None]
    if loc_res:
        by_sess = {}
        for r in loc_res:
            by_sess.setdefault(r["session"], []).append(r["az_error_deg"])
        sessions  = sorted(by_sess)
        mean_errs = [np.mean(by_sess[s]) for s in sessions]
        sess_cols = [_SESSION_COLORS.get(s, _S["muted"]) for s in sessions]
        ax1.barh(sessions, mean_errs, color=sess_cols, alpha=0.85, height=0.6)
        ax1.axvline(90, color=_S["muted"], ls="--", lw=1.2, label="Random")
        ax1.axvline(np.mean(mean_errs), color=_S["accent"], ls="-.", lw=1.5,
                    label=f"Overall {np.mean(mean_errs):.1f}°")
        ax1.set_xlabel("Mean azimuth MAE (°)")
        ax1.set_title("Per-session azimuth MAE")
        ax1.invert_yaxis()
        ax1.legend(fontsize=8); _leg(ax1)
    else:
        ax1.text(0.5, 0.5, "No localization results", ha="center", va="center",
                 color=_S["muted"], transform=ax1.transAxes)

    # ── [2] Predicted vs GT azimuth ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(fig, [ax2])
    pv = [r for r in results
          if r["pred_azimuth_deg"] is not None and r["gt_azimuth_deg"] is not None]
    if pv:
        gts  = [r["gt_azimuth_deg"]   for r in pv]
        preds= [r["pred_azimuth_deg"] for r in pv]
        ccols= [_SESSION_COLORS.get(r["session"], _S["muted"]) for r in pv]
        ax2.scatter(gts, preds, c=ccols, s=50, alpha=0.85,
                    edgecolors="white", linewidths=0.4)
        lim = (-185, 185)
        ax2.plot(lim, lim, color=_S["muted"], ls="--", lw=1.0)
        ax2.set_xlim(lim); ax2.set_ylim(lim); ax2.set_aspect("equal")
        ax2.set_xlabel("GT azimuth (°)"); ax2.set_ylabel("Predicted azimuth (°)")
        ax2.set_title("Predicted vs GT azimuth")

    # ── [3] Error vs distance ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(fig, [ax3])
    ev = [r for r in results
          if r["az_error_deg"] is not None and r["gt_distance_m"] is not None]
    if ev:
        dists = [r["gt_distance_m"] for r in ev]
        errs  = [r["az_error_deg"]  for r in ev]
        ecols = [_SESSION_COLORS.get(r["session"], _S["muted"]) for r in ev]
        sizes = [25 + r["n_drones"] * 15 for r in ev]
        ax3.scatter(dists, errs, c=ecols, s=sizes, alpha=0.80,
                    edgecolors="white", linewidths=0.4)
        ax3.axhline(90, color=_S["muted"], ls="--", lw=1.0, label="Random")
        ax3.axhline(30, color="#2e7d32",  ls=":",  lw=0.9, label="Good (30°)")
        ax3.set_xlabel("GT distance (m)"); ax3.set_ylabel("Azimuth error (°)")
        ax3.set_title("Error vs GT distance (size ∝ n_drones)")
        ax3.legend(fontsize=8); _leg(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, save_path); _show(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# §6  CSV export of results
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: List[dict], path: str):
    """Save full per-segment inference results to CSV."""
    fields = [
        "session_id", "segment_id", "session", "maneuver_type", "flight_phase",
        "n_drones", "split", "radius_m", "speed_mps", "duration_s",
        "detected", "probability", "cnn_probability", "heuristic_probability",
        "pred_azimuth_deg", "pred_distance_m", "pred_height_m",
        "gt_azimuth_deg", "gt_distance_m", "gt_height_m", "has_position",
        "az_error_deg", "dist_error_m", "ht_error_m",
        "rms_db",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  💾 Results CSV: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §7  High-level entry point
# ══════════════════════════════════════════════════════════════════════════════

def analyse_dunakeszi(
    pipeline_dir:  str,
    cfg,
    gt_csv:        Optional[str]       = None,
    splits:        Optional[List[str]] = None,
    multi_drone:   bool                = False,
    max_dist:      float               = 100.0,
    show_plots:    bool                = True,
    save_plots:    bool                = True,
    save_csv:      bool                = True,
    plots_dir:     Optional[str]       = None,
) -> Dict[str, Any]:
    """
    Full Dunakeszi analysis: config patch → inference → plots → CSV.

    Parameters
    ──────────
    pipeline_dir : output of prepare_dunakeszi_for_pipeline.py
    cfg          : drone_detection.config
    gt_csv       : path to ground_truth_segments.csv (optional, for quality_flags)
    splits       : ["train"] / ["val"] / ["test"] / None = all
    multi_drone  : use localize_multi_drone() for n_drones > 1 segments
    max_dist     : MAX_LOCALIZATION_DIST in metres (default 100)
    show_plots   : display inline (Colab)
    save_plots   : save PNGs to plots_dir
    save_csv     : save results CSV
    plots_dir    : where to save plots; defaults to cfg.DRIVE_PLOTS

    Returns
    ───────
    dict with keys: results, n_detected, n_segments, plots_dir
    """
    print("═" * 70)
    print("  Dunakeszi Pipeline Analysis")
    print("═" * 70)

    # ── 1. Config patch ───────────────────────────────────────────────────────
    print("\n§1  Patching config for Dunakeszi BK-6-E array …")
    patch_config_for_dunakeszi(cfg, max_dist=max_dist)

    # ── 2. Inference ──────────────────────────────────────────────────────────
    print("\n§2  Running inference …")
    results = run_dunakeszi_inference(
        pipeline_dir=pipeline_dir,
        cfg=cfg,
        gt_csv=gt_csv,
        splits=splits,
        multi_drone=multi_drone,
        verbose=True,
    )

    if not results:
        print("  ⚠️  No results produced.")
        return {"results": [], "n_detected": 0, "n_segments": 0}

    # ── 3. CSV export ─────────────────────────────────────────────────────────
    if save_csv:
        csv_out = str(Path(pipeline_dir) / "dunakeszi_inference_results.csv")
        save_results_csv(results, csv_out)

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    pdir = Path(plots_dir or str(cfg.DRIVE_PLOTS))

    def _p(name): return str(pdir / name) if save_plots else None

    if show_plots or save_plots:
        print("\n§3  Generating plots …")
        plot_summary_dashboard(results, cfg, _p("dunakeszi_dashboard.png"))
        plot_detection_timeline(results, cfg, _p("dunakeszi_detection_timeline.png"))
        plot_azimuth_errors(results, cfg, _p("dunakeszi_azimuth_errors.png"))
        plot_predicted_vs_gt(results, _p("dunakeszi_pred_vs_gt.png"))
        plot_error_vs_distance(results, _p("dunakeszi_error_vs_distance.png"))
        plot_position_map(results, cfg, _p("dunakeszi_position_map.png"))

    return {
        "results":     results,
        "n_detected":  sum(r["detected"] for r in results),
        "n_segments":  len(results),
        "plots_dir":   str(pdir),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §8  Colab quick-start cells
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Cell 1: Mount Drive & install deps ───────────────────────────────────────
#
# from google.colab import drive
# drive.mount("/content/drive")
# !pip install -q librosa soundfile scipy torch torchvision
#
# ── Cell 2: Rebuild ground truth (only needed once) ──────────────────────────
#
# !python dunakeszi_ground_truth_fixed.py \
#     --out_dir /content/drive/MyDrive/dunakeszi/ground_truth
#
# ── Cell 3: Extract segments from polywav files ───────────────────────────────
#
# !python dunakeszi_segment_extractor_fixed.py \
#     --segments /content/drive/MyDrive/dunakeszi/ground_truth/ground_truth_segments.json \
#     --wav-dir  /content/drive/MyDrive/dunakeszi/polywav_J/ \
#     --array    BK-6-E \
#     --clip-duration 3 \
#     --clip-position 0.5 \
#     --output-dir /content/drive/MyDrive/dunakeszi/extracted_J/
#
# ── Cell 4: Convert to pipeline format ───────────────────────────────────────
#
# !python prepare_dunakeszi_for_pipeline.py \
#     --input  /content/drive/MyDrive/dunakeszi/extracted_J/ \
#     --output /content/drive/MyDrive/dunakeszi/pipeline_ready/ \
#     --max-dist 100.0
#
# ── Cell 5: Run analysis (this file) ─────────────────────────────────────────
#
# import sys
# sys.path.insert(0, "/content/drive/MyDrive/drone_detection_project")
# from drone_detection import config
# from dunakeszi_pipeline_analysis import analyse_dunakeszi
#
# out = analyse_dunakeszi(
#     pipeline_dir = "/content/drive/MyDrive/dunakeszi/pipeline_ready/",
#     cfg          = config,
#     gt_csv       = "/content/drive/MyDrive/dunakeszi/ground_truth/ground_truth_segments.csv",
#     splits       = None,          # None = all  |  ["train"] / ["test"] etc.
#     multi_drone  = False,         # True to use localize_multi_drone() for multi-drone shows
#     max_dist     = 100.0,         # set 350.0 for show_10 long-range
#     show_plots   = True,
#     save_plots   = True,
#     save_csv     = True,
# )
# print(f"\n{out['n_detected']}/{out['n_segments']} segments detected.")
#
# ── Cell 6: Inspect results as a DataFrame ───────────────────────────────────
#
# import pandas as pd
# df = pd.DataFrame(out["results"])
# df[["session_id","session","maneuver_type","n_drones","split","detected",
#     "probability","gt_azimuth_deg","pred_azimuth_deg","az_error_deg",
#     "gt_distance_m","pred_distance_m","dist_error_m"]].to_string(index=False)
#
# ── Cell 7: Per-session summary ───────────────────────────────────────────────
#
# summary = (
#     df[df["az_error_deg"].notna()]
#     .groupby("session")
#     .agg(
#         n_segs      = ("session_id", "count"),
#         n_detected  = ("detected",   "sum"),
#         az_mae      = ("az_error_deg",  "mean"),
#         dist_mae    = ("dist_error_m",  "mean"),
#         n_drones    = ("n_drones",      "first"),
#     )
#     .round(2)
# )
# print(summary.to_string())
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Dunakeszi pipeline analysis (ground truth evaluation)"
    )
    ap.add_argument("--pipeline-dir",  required=True,
                    help="Directory produced by prepare_dunakeszi_for_pipeline.py")
    ap.add_argument("--gt-csv",        default=None,
                    help="Path to ground_truth_segments.csv (optional)")
    ap.add_argument("--splits",        nargs="+", default=None,
                    choices=["train", "val", "test"])
    ap.add_argument("--multi-drone",   action="store_true")
    ap.add_argument("--max-dist",      type=float, default=100.0)
    ap.add_argument("--no-plots",      action="store_true")
    ap.add_argument("--no-save",       action="store_true")
    ap.add_argument("--plots-dir",     default=None)
    args = ap.parse_args()

    from drone_detection import config
    out = analyse_dunakeszi(
        pipeline_dir = args.pipeline_dir,
        cfg          = config,
        gt_csv       = args.gt_csv,
        splits       = args.splits,
        multi_drone  = args.multi_drone,
        max_dist     = args.max_dist,
        show_plots   = not args.no_plots,
        save_plots   = not args.no_save,
        save_csv     = not args.no_save,
        plots_dir    = args.plots_dir,
    )
    print(f"\nDone. {out['n_detected']}/{out['n_segments']} segments detected.")