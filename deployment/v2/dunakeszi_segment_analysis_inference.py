# -*- coding: utf-8 -*-
"""
drone_segment_analysis.py
──────────────────────────────────────────────────────────────────────────────
Real-world drone segment analysis for Google Colab.

What this notebook does
───────────────────────
Given a folder of real drone audio segments (single-channel WAV/MP3/FLAC),
it runs the full detection + localization + tracking pipeline and produces:

  1. Per-segment table  — timestamp, CNN prob, heuristic prob, fused prob,
                          azimuth (°), distance (m), height (m), detected flag
  2. Detection timeline — fused / CNN / heuristic probability over time
  3. Drone path map     — top-down XY positions coloured by time,
                          with confidence-radius rings and Kalman track overlay
  4. Polar azimuth plot — histogram of detected bearings (N-up compass)
  5. Height profile     — estimated height over time for detected segments
  6. Summary printout   — total detections, track count, mean azimuth/distance

Usage (Colab)
─────────────
  # Mount Drive, install deps, then:
  from drone_segment_analysis import run_segment_analysis

  results = run_segment_analysis(
      segment_dir   = "/content/drive/MyDrive/drone_segments",
      cfg           = config,          # your Config instance
      file_ext      = [".wav"],        # or [".wav", ".mp3", ".flac"]
      sort_by_name  = True,            # keep temporal order from filenames
      n_mics        = 1,               # 1 = mono, 3 = true 3-mic array
      mic_files     = None,            # set to list of 3 dirs for 3-mic mode
      show_plots    = True,
      save_plots    = True,            # saves to cfg.DRIVE_PLOTS
      save_csv      = True,            # saves results table to cfg.DRIVE_LOGS
  )

3-mic mode
───────────
  Set n_mics=3 and mic_files to a list of 3 directories (one per mic),
  with filenames matching across dirs.  This enables localize() and
  localize_multi_drone() and produces accurate XY positions.

  mic_files = [
      "/content/drive/MyDrive/drone_segments/mic1",
      "/content/drive/MyDrive/drone_segments/mic2",
      "/content/drive/MyDrive/drone_segments/mic3",
  ]
  results = run_segment_analysis(cfg=config, n_mics=3, mic_files=mic_files)

1-mic / mono mode
──────────────────
  Set n_mics=1 (default).  Detection uses the hybrid CNN + heuristic pipeline.
  Localization falls back to azimuth-only heuristic (no XY triangulation).
  The drone path map shows concentric rings at estimated distance.
"""

# ══════════════════════════════════════════════════════════════════════════════
# §0  Imports
# ══════════════════════════════════════════════════════════════════════════════

import csv
import math
import time
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Suppress noisy librosa / torch warnings ───────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# §1  Helpers
# ══════════════════════════════════════════════════════════════════════════════

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

_STYLE = {
    "bg":       "#ffffff",
    "panel":    "#f8f9fa",
    "alt":      "#e9ecef",
    "accent":   "#1565c0",
    "warn":     "#e65100",
    "ok":       "#2e7d32",
    "err":      "#c62828",
    "grid":     "#b0bec5",
    "text":     "#212121",
    "muted":    "#546e7a",
    "spine":    "#90a4ae",
    "purple":   "#6a1b9a",
}

_TRACK_CMAP = plt.get_cmap("plasma")


def _style_ax(fig, axes):
    fig.patch.set_facecolor(_STYLE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        if ax is None:
            continue
        ax.set_facecolor(_STYLE["panel"])
        ax.tick_params(colors=_STYLE["text"], labelcolor=_STYLE["text"])
        ax.xaxis.label.set_color(_STYLE["text"])
        ax.yaxis.label.set_color(_STYLE["text"])
        ax.title.set_color(_STYLE["text"])
        ax.title.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_color(_STYLE["spine"])
            sp.set_linewidth(0.8)
        ax.grid(color=_STYLE["grid"], alpha=0.45, linewidth=0.7)


def _style_legend(leg):
    if leg is None:
        return
    leg.get_frame().set_facecolor(_STYLE["alt"])
    leg.get_frame().set_edgecolor(_STYLE["spine"])
    for t in leg.get_texts():
        t.set_color(_STYLE["text"])


def _save(fig, path: Optional[Path], dpi: int = 180):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"  💾 Saved: {path}")


def _show(fig):
    try:
        from IPython.display import display
        display(fig)
    except Exception:
        pass


def _collect_files(
    segment_dir: str,
    file_ext: List[str],
    sort_by_name: bool,
) -> List[Path]:
    """Return sorted list of audio files from segment_dir."""
    d = Path(segment_dir)
    if not d.exists():
        raise FileNotFoundError(f"segment_dir not found: {d}")
    exts = {e.lower() for e in file_ext}
    files = [f for f in d.iterdir() if f.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(
            f"No audio files with extensions {exts} found in {d}"
        )
    return sorted(files, key=lambda f: f.name) if sort_by_name else files


def _match_mic_files(
    mic_dirs: List[str],
    file_ext: List[str],
    sort_by_name: bool,
) -> List[List[Path]]:
    """
    For 3-mic mode: match filenames across 3 directories.
    Returns a list of [mic1_path, mic2_path, mic3_path] triplets.
    """
    assert len(mic_dirs) == 3, "mic_files must contain exactly 3 directories."
    exts = {e.lower() for e in file_ext}
    dirs = [Path(d) for d in mic_dirs]
    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"mic dir not found: {d}")

    # Use filenames from mic1 as the reference set
    ref_files = sorted([f for f in dirs[0].iterdir() if f.suffix.lower() in exts],
                       key=lambda f: f.name) if sort_by_name else \
                [f for f in dirs[0].iterdir() if f.suffix.lower() in exts]

    triplets = []
    for ref in ref_files:
        group = [ref]
        ok = True
        for d in dirs[1:]:
            match = d / ref.name
            if not match.exists():
                print(f"  ⚠️  No matching file for {ref.name} in {d} — skipping segment.")
                ok = False
                break
            group.append(match)
        if ok:
            triplets.append(group)
    if not triplets:
        raise FileNotFoundError("No matching file triplets found across mic directories.")
    return triplets


# ══════════════════════════════════════════════════════════════════════════════
# §2  Core analysis loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_mono_segment(audio: np.ndarray, cfg, ap, can_localize: bool) -> Dict[str, Any]:
    """Detection + fallback localization for a single mono channel."""
    from drone_detection.inference import detect, localize

    det = detect([audio, audio, audio], cfg)
    loc = None

    if det["detected"] and can_localize:
        try:
            loc = localize([audio, audio, audio], cfg)
        except Exception as exc:
            print(f"    ⚠️  localize() failed: {exc}")

    return {
        "detected":              det["detected"],
        "probability":           det["probability"],
        "cnn_probability":       det.get("cnn_probability", float("nan")),
        "heuristic_probability": det.get("heuristic_probability", float("nan")),
        "loc":                   loc,
        "drones":                [loc] if loc else [],
    }


def _run_3mic_segment(triplet: List[Path], cfg, ap, can_localize: bool,
                      multi_drone: bool) -> Dict[str, Any]:
    """Detection + 3-mic localization for a matched file triplet."""
    from drone_detection.inference import detect, localize, load_3ch
    from drone_detection.multidrone import localize_multi_drone

    channels = load_3ch([str(p) for p in triplet], cfg)
    det      = detect(channels, cfg)
    loc      = None
    drones   = []

    if det["detected"] and can_localize:
        try:
            if multi_drone:
                drones = localize_multi_drone(channels, cfg)
                loc    = drones[0] if drones else None
            else:
                loc    = localize(channels, cfg)
                drones = [loc]
        except Exception as exc:
            print(f"    ⚠️  localize() failed: {exc}")

    return {
        "detected":              det["detected"],
        "probability":           det["probability"],
        "cnn_probability":       det.get("cnn_probability", float("nan")),
        "heuristic_probability": det.get("heuristic_probability", float("nan")),
        "loc":                   loc,
        "drones":                drones,
    }


def _process_segments(
    files_or_triplets,
    cfg,
    n_mics: int,
    multi_drone: bool,
    can_localize: bool,
) -> List[Dict[str, Any]]:
    """Main loop: iterate segments, detect + localize, build result rows."""
    from drone_detection.audio_processing import AudioProcessor
    from drone_detection.tracking import KalmanTracker

    ap      = AudioProcessor(cfg)
    tracker = KalmanTracker(cfg)
    base_ts = time.time()
    rows    = []

    n = len(files_or_triplets)
    print(f"\n{'─'*60}")
    print(f"  Processing {n} segment{'s' if n != 1 else ''} "
          f"({'3-mic' if n_mics == 3 else 'mono'} mode)")
    print(f"{'─'*60}")

    for i, item in enumerate(files_or_triplets):
        label = item[0].name if n_mics == 3 else item.name

        # Load audio for metadata (use first / only channel)
        src_path = item[0] if n_mics == 3 else item
        audio    = ap.pad_or_truncate(ap.load(str(src_path), mono=True))
        rms_db   = float(20 * math.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-8))
        mel      = ap.mel(audio)

        # Infer segment timestamp from filename if possible
        t_s = _infer_timestamp(label, i, cfg)

        # Run pipeline
        if n_mics == 3:
            res = _run_3mic_segment(item, cfg, ap, can_localize, multi_drone)
        else:
            res = _run_mono_segment(audio, cfg, ap, can_localize)

        # Kalman tracking step
        positions = [d["xy_position"] for d in res["drones"] if d and "xy_position" in d]
        tracks    = tracker.step(
            [np.asarray(p, dtype=np.float32) for p in positions],
            base_ts + t_s,
        )

        # Build result row
        loc = res["loc"]
        row = {
            "seg":                   i + 1,
            "filename":              label,
            "t_start":               t_s,
            "detected":              res["detected"],
            "probability":           round(res["probability"], 4),
            "cnn_probability":       round(res["cnn_probability"], 4),
            "heuristic_probability": round(res["heuristic_probability"], 4),
            "rms_db":                round(rms_db, 2),
            "azimuth_deg":           round(loc["azimuth_deg"], 2)  if loc else None,
            "distance_m":            round(loc["distance_m"],  2)  if loc else None,
            "height_m":              round(loc.get("height_m", float("nan")), 2) if loc else None,
            "xy_position":           loc["xy_position"] if loc else None,
            "n_drones":              len(res["drones"]),
            "all_drones":            res["drones"],
            "confidence_radius":     loc.get("confidence_radius", float("nan")) if loc else None,
            "mel":                   mel,
            "tracks":                tracks,
            "loc":                   loc,
        }
        rows.append(row)

        icon = "🚁" if res["detected"] else "🌳"
        az   = f"az={loc['azimuth_deg']:.1f}°" if loc else "no-loc"
        dist = f"d={loc['distance_m']:.1f}m"   if loc else ""
        print(f"  [{i+1:>3}/{n}] {icon}  {label:<35s}  "
              f"conf={res['probability']:.3f}  rms={rms_db:.1f}dB  {az} {dist}")

    confirmed = tracker.all_confirmed()
    n_det     = sum(r["detected"] for r in rows)
    print(f"\n  ✅ {n_det}/{n} segments detected  |  "
          f"{len(confirmed)} confirmed Kalman track(s)")

    return rows, confirmed


def _infer_timestamp(filename: str, fallback_idx: int, cfg) -> float:
    """
    Try to parse a timestamp from the filename.
    Supports patterns like: seg_000.wav, 000123ms.wav, t0.500.wav
    Falls back to fallback_idx * TARGET_DURATION.
    """
    import re
    name = Path(filename).stem
    # Pattern: leading digits (e.g. 001, 000500)
    m = re.match(r"^(\d+)", name)
    if m:
        val = int(m.group(1))
        # Heuristic: if value looks like ms (> 9999), convert
        if val > 9999:
            return val / 1000.0
        return float(val) * cfg.TARGET_DURATION
    # Pattern: t<float> (e.g. t0.500)
    m = re.search(r"t(\d+\.?\d*)", name)
    if m:
        return float(m.group(1))
    return fallback_idx * cfg.TARGET_DURATION


# ══════════════════════════════════════════════════════════════════════════════
# §3  Plots
# ══════════════════════════════════════════════════════════════════════════════

def _plot_detection_timeline(rows: List[dict], cfg, save_path: Optional[Path]):
    """Panel: fused / CNN / heuristic probability over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    _style_ax(fig, [ax])

    ts    = [r["t_start"]               for r in rows]
    fused = [r["probability"]           for r in rows]
    cnn   = [r["cnn_probability"]       for r in rows]
    heur  = [r["heuristic_probability"] for r in rows]
    cols  = [_STYLE["ok"] if r["detected"] else _STYLE["err"] for r in rows]

    w = max((ts[1] - ts[0]) * 0.75 if len(ts) > 1 else cfg.TARGET_DURATION * 0.75,
            cfg.TARGET_DURATION * 0.2)

    ax.bar(ts, fused, width=w, color=cols, alpha=0.45, label="Fused prob")
    ax.plot(ts, fused, "o-",  color=_STYLE["accent"], ms=5, lw=1.8, label="Fused")
    ax.plot(ts, cnn,   "s--", color=_STYLE["purple"], ms=4, lw=1.4, label="CNN")
    ax.plot(ts, heur,  "^:", color=_STYLE["warn"],   ms=4, lw=1.2, label="Heuristic")
    ax.axhline(cfg.DETECTION_THRESHOLD, color=_STYLE["err"], ls="--", lw=1.5,
               label=f"Threshold ({cfg.DETECTION_THRESHOLD:.2f})")

    ax.set_xlim(left=min(ts) - w, right=max(ts) + w)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Segment time (s)")
    ax.set_ylabel("Detection probability")
    ax.set_title("Detection timeline — fused / CNN / heuristic probabilities")
    _style_legend(ax.legend(loc="upper right", fontsize=10))

    plt.tight_layout()
    _save(fig, save_path)
    _show(fig)
    plt.close(fig)


def _plot_drone_path_map(rows: List[dict], confirmed_tracks: list,
                         cfg, save_path: Optional[Path]):
    """
    Top-down XY position map showing:
      - Mic array triangle
      - Per-segment drone positions coloured by time
      - Confidence-radius rings
      - Kalman track trajectories
      - Multi-drone positions (all drones per segment)
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    _style_ax(fig, [ax])

    mics = cfg.MIC_POSITIONS
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=220,
               c=_STYLE["warn"], zorder=10, label="Mic array")
    for i, m in enumerate(mics):
        ax.annotate(f"M{i}", m, textcoords="offset points", xytext=(6, 5),
                    fontsize=10, color=_STYLE["text"])

    # Collect all located segments (including multi-drone)
    all_locs = []
    for r in rows:
        for d in r.get("all_drones", []):
            if d and "xy_position" in d:
                all_locs.append((r["t_start"], d))

    if all_locs:
        t_vals = [t for t, _ in all_locs]
        t_min, t_max = min(t_vals), max(t_vals)
        t_range = max(t_max - t_min, 1e-3)

        for t, d in all_locs:
            xy = np.asarray(d["xy_position"], dtype=float)
            cr = d.get("confidence_radius", float("nan"))
            norm_t = (t - t_min) / t_range
            col = _TRACK_CMAP(norm_t)

            ax.scatter(*xy, s=90, color=col, zorder=6, alpha=0.85,
                       edgecolors="white", linewidths=0.6)

            if cr and not math.isnan(cr) and 0 < cr < cfg.MAX_LOCALIZATION_DIST:
                ax.add_patch(plt.Circle(xy, cr, color=col, alpha=0.10, fill=True))

        # Colorbar for time
        sm = cm.ScalarMappable(cmap=_TRACK_CMAP,
                               norm=plt.Normalize(vmin=t_min, vmax=t_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label("Segment time (s)", color=_STYLE["text"])
        cbar.ax.yaxis.set_tick_params(color=_STYLE["text"])
        cbar.ax.tick_params(colors=_STYLE["text"])

    # Kalman track trajectories
    track_colors = [_STYLE["accent"], _STYLE["ok"], _STYLE["err"], _STYLE["purple"]]
    for ti, track in enumerate(confirmed_tracks):
        pts = np.array(track.positions)
        col = track_colors[ti % len(track_colors)]
        if len(pts) >= 2:
            for k in range(len(pts) - 1):
                alpha = 0.4 + 0.5 * k / max(len(pts) - 2, 1)
                ax.plot(pts[k:k+2, 0], pts[k:k+2, 1], "-",
                        color=col, lw=2.5, alpha=alpha, zorder=7)
            ax.scatter(*pts[0],  marker=">", s=100, c=[col], zorder=9)
            ax.scatter(*pts[-1], marker="s", s=100, c=[col], zorder=9,
                       label=f"Track #{track.track_id} "
                             f"({len(pts)} pts, {track.total_distance():.1f} m)")
        else:
            ax.scatter(*pts[0], marker="D", s=140, c=[col], zorder=9,
                       label=f"Track #{track.track_id} (1 pt)")

        # Uncertainty ellipse at last position
        try:
            ur = track.uncertainty_radius()
            if ur and not math.isnan(ur) and ur < 20:
                ax.add_patch(plt.Circle(pts[-1], ur, color=col,
                                        alpha=0.12, fill=True, zorder=5))
        except Exception:
            pass

    n_det = sum(r["detected"] for r in rows)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Drone position map — {n_det}/{len(rows)} segments detected  |  "
        f"{len(confirmed_tracks)} confirmed track(s)"
    )
    _style_legend(ax.legend(loc="upper left", fontsize=10))
    ax.set_aspect("equal")
    plt.tight_layout()
    _save(fig, save_path)
    _show(fig)
    plt.close(fig)


def _plot_polar_azimuth(rows: List[dict], save_path: Optional[Path]):
    """Polar compass histogram of detected azimuths."""
    az_list = [r["azimuth_deg"] for r in rows
               if r["detected"] and r["azimuth_deg"] is not None]
    if not az_list:
        print("  ⚠️  No detected azimuths to plot.")
        return

    fig = plt.figure(figsize=(6, 6), facecolor=_STYLE["bg"])
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(_STYLE["panel"])
    ax.tick_params(colors=_STYLE["text"])

    # Convert azimuth (clockwise from North) → polar angle (CCW from East)
    rads = np.radians([90 - a for a in az_list])
    counts, edges = np.histogram(rads, bins=36, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=edges[1] - edges[0],
           color=_STYLE["accent"], alpha=0.75, edgecolor=_STYLE["bg"])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        f"Azimuth distribution\n({len(az_list)} detections)",
        pad=14, color=_STYLE["text"], fontweight="bold"
    )
    ax.grid(color=_STYLE["grid"], alpha=0.5)

    plt.tight_layout()
    _save(fig, save_path)
    _show(fig)
    plt.close(fig)


def _plot_height_profile(rows: List[dict], cfg, save_path: Optional[Path]):
    """Height over time for located segments."""
    loc_rows = [r for r in rows if r["detected"] and r["height_m"] is not None
                and not math.isnan(r["height_m"])]
    if not loc_rows:
        print("  ⚠️  No height data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    _style_ax(fig, [ax])

    ts  = [r["t_start"] for r in loc_rows]
    hts = [r["height_m"] for r in loc_rows]
    dists = [r["distance_m"] for r in loc_rows if r["distance_m"] is not None]

    ax.plot(ts, hts, "o-", color=_STYLE["accent"], ms=6, lw=2, label="Height (m)")
    ax.fill_between(ts, hts, alpha=0.12, color=_STYLE["accent"])

    if dists and len(dists) == len(ts):
        ax2 = ax.twinx()
        ax2.plot(ts, dists, "s--", color=_STYLE["warn"], ms=5, lw=1.5,
                 label="Distance (m)")
        ax2.set_ylabel("Distance (m)", color=_STYLE["text"])
        ax2.tick_params(colors=_STYLE["text"], labelcolor=_STYLE["text"])
        for sp in ax2.spines.values():
            sp.set_color(_STYLE["spine"])
        _style_legend(ax2.legend(loc="upper right", fontsize=10))

    ax.set_xlabel("Segment time (s)")
    ax.set_ylabel("Estimated height (m)")
    ax.set_title("Drone height & distance profile over time")
    _style_legend(ax.legend(loc="upper left", fontsize=10))

    plt.tight_layout()
    _save(fig, save_path)
    _show(fig)
    plt.close(fig)


def _plot_summary_dashboard(rows: List[dict], confirmed_tracks: list,
                             cfg, save_path: Optional[Path]):
    """
    4-panel summary dashboard:
      [0] Detection timeline  [1] Drone path map
      [2] Polar azimuth       [3] Height / distance profile
    """
    fig = plt.figure(figsize=(18, 12), facecolor=_STYLE["bg"])
    fig.suptitle("Drone Segment Analysis — Summary Dashboard",
                 fontsize=16, color=_STYLE["accent"], fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── [0] Detection timeline ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _style_ax(fig, [ax0])
    ts    = [r["t_start"]     for r in rows]
    fused = [r["probability"] for r in rows]
    cnn   = [r["cnn_probability"] for r in rows]
    heur  = [r["heuristic_probability"] for r in rows]
    cols  = [_STYLE["ok"] if r["detected"] else _STYLE["err"] for r in rows]
    w     = max((ts[1] - ts[0]) * 0.75 if len(ts) > 1 else 0.4, 0.2)
    ax0.bar(ts, fused, width=w, color=cols, alpha=0.40)
    ax0.plot(ts, fused, "o-",  color=_STYLE["accent"], ms=4, lw=1.8, label="Fused")
    ax0.plot(ts, cnn,   "s--", color=_STYLE["purple"], ms=3, lw=1.2, label="CNN")
    ax0.plot(ts, heur,  "^:",  color=_STYLE["warn"],   ms=3, lw=1.2, label="Heuristic")
    ax0.axhline(cfg.DETECTION_THRESHOLD, color=_STYLE["err"], ls="--", lw=1.2,
                label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax0.set_ylim(0, 1.08)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Probability")
    ax0.set_title("Detection timeline")
    _style_legend(ax0.legend(fontsize=9, loc="upper right"))

    # ── [1] Drone path map ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _style_ax(fig, [ax1])
    mics = cfg.MIC_POSITIONS
    ax1.scatter(mics[:, 0], mics[:, 1], marker="^", s=200,
                c=_STYLE["warn"], zorder=10, label="Mics")
    for i, m in enumerate(mics):
        ax1.annotate(f"M{i}", m, textcoords="offset points", xytext=(5, 4),
                     fontsize=9, color=_STYLE["text"])

    all_locs = []
    for r in rows:
        for d in r.get("all_drones", []):
            if d and "xy_position" in d:
                all_locs.append((r["t_start"], d))

    if all_locs:
        t_vals = [t for t, _ in all_locs]
        t_min, t_max = min(t_vals), max(t_vals)
        t_range = max(t_max - t_min, 1e-3)
        for t, d in all_locs:
            xy     = np.asarray(d["xy_position"], dtype=float)
            norm_t = (t - t_min) / t_range
            col    = _TRACK_CMAP(norm_t)
            ax1.scatter(*xy, s=60, color=col, zorder=6, alpha=0.85,
                        edgecolors="white", linewidths=0.5)
            cr = d.get("confidence_radius", float("nan"))
            if cr and not math.isnan(cr) and 0 < cr < cfg.MAX_LOCALIZATION_DIST:
                ax1.add_patch(plt.Circle(xy, cr, color=col, alpha=0.08, fill=True))

    track_colors = [_STYLE["accent"], _STYLE["ok"], _STYLE["err"], _STYLE["purple"]]
    for ti, track in enumerate(confirmed_tracks):
        pts = np.array(track.positions)
        col = track_colors[ti % len(track_colors)]
        if len(pts) >= 2:
            ax1.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=2.2, alpha=0.8, zorder=7)
        ax1.scatter(*pts[-1], marker="s", s=80, c=[col], zorder=9,
                    label=f"Track #{track.track_id}")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"Drone path map ({len(confirmed_tracks)} track(s))")
    ax1.set_aspect("equal")
    _style_legend(ax1.legend(fontsize=9, loc="upper left"))

    # ── [2] Polar azimuth ─────────────────────────────────────────────────────
    ax2_tmp = fig.add_subplot(gs[1, 0])
    ax2_tmp.remove()
    ax2 = fig.add_subplot(gs[1, 0], projection="polar")
    ax2.set_facecolor(_STYLE["panel"])
    ax2.tick_params(colors=_STYLE["text"])
    az_list = [r["azimuth_deg"] for r in rows
               if r["detected"] and r["azimuth_deg"] is not None]
    if az_list:
        rads = np.radians([90 - a for a in az_list])
        counts, edges = np.histogram(rads, bins=24, range=(-np.pi, np.pi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax2.bar(centers, counts, width=edges[1] - edges[0],
                color=_STYLE["accent"], alpha=0.75, edgecolor=_STYLE["bg"])
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_title(f"Azimuth (N-up)\n{len(az_list)} detections",
                  pad=12, color=_STYLE["text"], fontweight="bold")
    ax2.grid(color=_STYLE["grid"], alpha=0.5)

    # ── [3] Height / distance profile ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(fig, [ax3])
    loc_rows = [r for r in rows
                if r["detected"] and r["height_m"] is not None
                and not (isinstance(r["height_m"], float) and math.isnan(r["height_m"]))]
    if loc_rows:
        ts3  = [r["t_start"]  for r in loc_rows]
        hts  = [r["height_m"] for r in loc_rows]
        dsts = [r["distance_m"] for r in loc_rows if r["distance_m"] is not None]
        ax3.plot(ts3, hts, "o-", color=_STYLE["accent"], ms=5, lw=1.8, label="Height (m)")
        ax3.fill_between(ts3, hts, alpha=0.12, color=_STYLE["accent"])
        if len(dsts) == len(ts3):
            ax3b = ax3.twinx()
            ax3b.plot(ts3, dsts, "s--", color=_STYLE["warn"], ms=4, lw=1.4,
                      label="Distance (m)")
            ax3b.set_ylabel("Distance (m)", color=_STYLE["text"])
            ax3b.tick_params(colors=_STYLE["text"], labelcolor=_STYLE["text"])
            for sp in ax3b.spines.values():
                sp.set_color(_STYLE["spine"])
            _style_legend(ax3b.legend(loc="upper right", fontsize=9))
    else:
        ax3.text(0.5, 0.5, "No localization data", ha="center", va="center",
                 color=_STYLE["muted"], transform=ax3.transAxes, fontsize=12)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Height (m)")
    ax3.set_title("Height & distance profile")
    _style_legend(ax3.legend(fontsize=9, loc="upper left"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, save_path)
    _show(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# §4  CSV export
# ══════════════════════════════════════════════════════════════════════════════

def _save_csv(rows: List[dict], path: Path):
    """Save per-segment results as a CSV (excludes large array fields)."""
    fields = [
        "seg", "filename", "t_start", "detected",
        "probability", "cnn_probability", "heuristic_probability",
        "rms_db", "n_drones",
        "azimuth_deg", "distance_m", "height_m", "confidence_radius",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"  💾 CSV saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §5  Summary printout
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(rows: List[dict], confirmed_tracks: list, cfg):
    n      = len(rows)
    n_det  = sum(r["detected"] for r in rows)
    det_rows = [r for r in rows if r["detected"]]

    az_vals   = [r["azimuth_deg"] for r in det_rows if r["azimuth_deg"] is not None]
    dist_vals = [r["distance_m"]  for r in det_rows if r["distance_m"]  is not None]
    ht_vals   = [r["height_m"]    for r in det_rows
                 if r["height_m"] is not None
                 and not (isinstance(r["height_m"], float) and math.isnan(r["height_m"]))]

    print("\n" + "═" * 64)
    print("  DRONE SEGMENT ANALYSIS — SUMMARY")
    print("═" * 64)
    print(f"  Total segments     : {n}")
    print(f"  Detected           : {n_det}  ({100*n_det/max(n,1):.1f}%)")
    print(f"  Confirmed tracks   : {len(confirmed_tracks)}")
    print(f"  Detection threshold: {cfg.DETECTION_THRESHOLD:.3f}")

    if az_vals:
        print(f"\n  Azimuth (detected segments)")
        print(f"    mean   : {np.mean(az_vals):.1f}°")
        print(f"    median : {np.median(az_vals):.1f}°")
        print(f"    range  : {min(az_vals):.1f}° – {max(az_vals):.1f}°")

    if dist_vals:
        print(f"\n  Distance (detected segments)")
        print(f"    mean   : {np.mean(dist_vals):.2f} m")
        print(f"    median : {np.median(dist_vals):.2f} m")
        print(f"    range  : {min(dist_vals):.2f} – {max(dist_vals):.2f} m")

    if ht_vals:
        print(f"\n  Height (detected segments)")
        print(f"    mean   : {np.mean(ht_vals):.2f} m")
        print(f"    median : {np.median(ht_vals):.2f} m")

    if confirmed_tracks:
        print(f"\n  Kalman tracks")
        for t in confirmed_tracks:
            print(f"    Track #{t.track_id}: {len(t.positions)} pts  |  "
                  f"total dist = {t.total_distance():.2f} m  |  "
                  f"active = {t.active}")

    print("═" * 64 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# §6  Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_segment_analysis(
    segment_dir:   Optional[str]        = None,
    cfg                                 = None,
    file_ext:      List[str]            = None,
    sort_by_name:  bool                 = True,
    n_mics:        int                  = 1,
    mic_files:     Optional[List[str]]  = None,
    multi_drone:   bool                 = False,
    show_plots:    bool                 = True,
    save_plots:    bool                 = True,
    save_csv:      bool                 = True,
    plot_individual: bool               = True,
) -> Dict[str, Any]:
    """
    Run the full detection + localization + tracking pipeline over a folder
    of drone audio segments and generate analysis plots.

    Parameters
    ──────────
    segment_dir    : folder containing audio files (ignored in 3-mic mode)
    cfg            : Config instance (defaults to drone_detection.config)
    file_ext       : list of accepted extensions, e.g. [".wav", ".mp3"]
    sort_by_name   : sort files lexicographically (keeps temporal order)
    n_mics         : 1 = mono/single-channel, 3 = 3-mic array mode
    mic_files      : for n_mics=3, list of 3 directories (one per mic)
    multi_drone    : use localize_multi_drone() instead of localize()
    show_plots     : display plots inline (Colab)
    save_plots     : save plots to cfg.DRIVE_PLOTS
    save_csv       : save results table to cfg.DRIVE_LOGS
    plot_individual: also produce individual timeline/map/polar/height plots
                     in addition to the summary dashboard

    Returns
    ───────
    dict with keys:
      rows             — list of per-segment result dicts
      confirmed_tracks — list of confirmed KalmanTrack objects
      n_detected       — int
      n_segments       — int
    """
    # ── Defaults ──────────────────────────────────────────────────────────────
    if cfg is None:
        from drone_detection.config import config as _cfg
        cfg = _cfg

    file_ext = file_ext or [".wav", ".mp3", ".flac"]

    # ── Load models ───────────────────────────────────────────────────────────
    print("🔄 Loading models...")
    from drone_detection.inference import load_detection_model, load_localization_model
    load_detection_model(cfg)
    can_localize = False
    try:
        load_localization_model(cfg)
        can_localize = True
    except FileNotFoundError:
        print("  ⚠️  No localization model found — detection-only mode.")

    # ── Collect files ─────────────────────────────────────────────────────────
    if n_mics == 3:
        if mic_files is None:
            raise ValueError("mic_files must be provided when n_mics=3.")
        files_or_triplets = _match_mic_files(mic_files, file_ext, sort_by_name)
        print(f"\n📂 3-mic mode: {len(files_or_triplets)} matched segment triplets")
    else:
        if segment_dir is None:
            raise ValueError("segment_dir must be provided when n_mics=1.")
        files_or_triplets = _collect_files(segment_dir, file_ext, sort_by_name)
        print(f"\n📂 Mono mode: {len(files_or_triplets)} audio files in {segment_dir}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    rows, confirmed_tracks = _process_segments(
        files_or_triplets, cfg, n_mics, multi_drone, can_localize
    )

    # ── Summary printout ──────────────────────────────────────────────────────
    _print_summary(rows, confirmed_tracks, cfg)

    # ── CSV export ────────────────────────────────────────────────────────────
    if save_csv:
        csv_path = cfg.DRIVE_LOGS / "segment_analysis_results.csv"
        _save_csv(rows, csv_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = cfg.DRIVE_PLOTS if save_plots else None

    # Always generate the summary dashboard
    print("\n🖼️  Generating summary dashboard...")
    _plot_summary_dashboard(
        rows, confirmed_tracks, cfg,
        save_path=plots_dir / "segment_analysis_dashboard.png" if plots_dir else None,
    )

    if plot_individual:
        print("🖼️  Generating individual plots...")
        _plot_detection_timeline(
            rows, cfg,
            save_path=plots_dir / "segment_timeline.png" if plots_dir else None,
        )
        _plot_drone_path_map(
            rows, confirmed_tracks, cfg,
            save_path=plots_dir / "segment_path_map.png" if plots_dir else None,
        )
        _plot_polar_azimuth(
            rows,
            save_path=plots_dir / "segment_polar_azimuth.png" if plots_dir else None,
        )
        _plot_height_profile(
            rows, cfg,
            save_path=plots_dir / "segment_height_profile.png" if plots_dir else None,
        )

    return {
        "rows":             rows,
        "confirmed_tracks": confirmed_tracks,
        "n_detected":       sum(r["detected"] for r in rows),
        "n_segments":       len(rows),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §7  Colab quick-start cells  (copy-paste into a notebook)
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Cell 1: Mount Drive & install ────────────────────────────────────────────
#
# from google.colab import drive
# drive.mount("/content/drive")
# !pip install -q librosa soundfile pydub torch torchvision scipy
#
# ── Cell 2: Import your package ──────────────────────────────────────────────
#
# import sys
# sys.path.insert(0, "/content/drive/MyDrive/drone_detection_project")
# from drone_detection import config
# from drone_segment_analysis import run_segment_analysis
#
# ── Cell 3a: Mono / single-mic analysis ──────────────────────────────────────
#
# results = run_segment_analysis(
#     segment_dir  = "/content/drive/MyDrive/drone_segments",
#     cfg          = config,
#     file_ext     = [".wav"],
#     sort_by_name = True,
#     n_mics       = 1,
#     show_plots   = True,
#     save_plots   = True,
#     save_csv     = True,
# )
#
# ── Cell 3b: 3-mic array analysis with multi-drone localization ───────────────
#
# results = run_segment_analysis(
#     cfg          = config,
#     file_ext     = [".wav"],
#     n_mics       = 3,
#     mic_files    = [
#         "/content/drive/MyDrive/drone_segments/mic1",
#         "/content/drive/MyDrive/drone_segments/mic2",
#         "/content/drive/MyDrive/drone_segments/mic3",
#     ],
#     multi_drone  = True,          # enable multi-drone TDOA localization
#     show_plots   = True,
#     save_plots   = True,
#     save_csv     = True,
# )
#
# ── Cell 4: Inspect results ───────────────────────────────────────────────────
#
# import pandas as pd
# df = pd.DataFrame([
#     {k: r[k] for k in ["seg","filename","t_start","detected",
#                         "probability","azimuth_deg","distance_m","height_m"]}
#     for r in results["rows"]
# ])
# df.head(20)
#
# ── Cell 5: Inspect Kalman tracks ─────────────────────────────────────────────
#
# for track in results["confirmed_tracks"]:
#     print(track.to_dict())
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Stand-alone test with a synthetic signal so the file runs without a
    # real dataset.  Remove or replace with your own segment_dir in Colab.
    import argparse

    parser = argparse.ArgumentParser(description="Drone segment analysis")
    parser.add_argument("--segment_dir", type=str, required=True,
                        help="Directory of audio segments")
    parser.add_argument("--n_mics",      type=int, default=1,
                        choices=[1, 3])
    parser.add_argument("--mic1",        type=str, default=None)
    parser.add_argument("--mic2",        type=str, default=None)
    parser.add_argument("--mic3",        type=str, default=None)
    parser.add_argument("--multi_drone", action="store_true")
    parser.add_argument("--no_save",     action="store_true")
    args = parser.parse_args()

    mic_files = [args.mic1, args.mic2, args.mic3] if args.n_mics == 3 else None

    results = run_segment_analysis(
        segment_dir = args.segment_dir,
        n_mics      = args.n_mics,
        mic_files   = mic_files,
        multi_drone = args.multi_drone,
        show_plots  = True,
        save_plots  = not args.no_save,
        save_csv    = not args.no_save,
    )
    print(f"\nDone. {results['n_detected']}/{results['n_segments']} detected.")