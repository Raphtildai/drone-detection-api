# -*- coding: utf-8 -*-
"""
mems_inference.py
─────────────────
Adapter for running detection (and limited analysis) on single-channel
MEMS recordings like the Dunakeszi dataset.

WHY THIS FILE EXISTS
────────────────────
The pipeline's localize() and run_pipeline() functions expect exactly three
microphone channels with known spatial positions so they can compute TDOA
(Time Difference of Arrival) to estimate azimuth/distance.  Real-world
single-channel MEMS recordings contain only one channel — the spatial
information simply does not exist in the audio.

This adapter provides:
  analyse_mems_file()       — full detection analysis with dashboard
  batch_analyse_mems()      — run over a folder of MEMS WAVs
  analyse_mems_with_meta()  — same, but reads the companion _meta.json
                              to annotate results with ground-truth labels
  build_mems_report()       — print + return a summary table

WHAT WORKS vs. WHAT DOESN'T
────────────────────────────
  ✅ Detection (CNN + heuristic hybrid)
  ✅ BPF / spectral analysis (dominant frequency, harmonics, SNR estimate)
  ✅ Mel spectrogram + waveform dashboards
  ✅ Segment-level detection timeline
  ✅ Flight-phase / clip metadata from _meta.json
  ❌ Azimuth localisation — requires 3-mic array; produces NaN
  ❌ Distance / height estimation — same reason
  ❌ Kalman tracking — no spatial positions to track

USAGE
─────
    from drone_detection import config
    from mems_inference import analyse_mems_file, batch_analyse_mems

    # Single file
    result = analyse_mems_file(
        "251020VITEMOROM1AT01A_0227000_0230000.wav",
        config,
        meta_path="251020VITEMOROM1AT01A_0227000_0230000_meta.json",
    )

    # Folder of MEMS clips
    report = batch_analyse_mems("path/to/mems_clips/", config)
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

matplotlib.use("Agg")

# ── lazy pipeline imports (avoids hard dependency at import time) ──────────────

def _cfg_default():
    from .config import config
    return config

def _get_ap(cfg):
    from .audio_processing import AudioProcessor
    return AudioProcessor(cfg)

def _get_detect(cfg):
    from .inference import load_detection_model, detect
    load_detection_model(cfg)
    return detect

def _get_heuristic():
    from .inference import heuristic_detect
    return heuristic_detect

def _plot_style():
    try:
        from .visualization import PLOT_STYLE
        return PLOT_STYLE
    except Exception:
        return {
            "bg": "#08111f", "panel": "#0f1b2d", "panel_alt": "#14233a",
            "accent": "#38bdf8", "warn": "#fbbf24", "ok": "#4ade80",
            "err": "#f87171", "grid": "#47607d", "text": "#f8fafc",
            "muted": "#b6c2cf", "spine": "#6b85a3", "purple": "#a78bfa",
        }

def _apply_dark(fig, axes):
    try:
        from .visualization import _apply_dark_style
        _apply_dark_style(fig, axes)
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


# ══════════════════════════════════════════════════════════════════════════════
# Meta-JSON helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_meta(meta_path: str | Path) -> dict:
    """
    Load a Dunakeszi-format _meta.json file.
    Returns an empty dict if the file is missing or malformed.
    """
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _meta_summary(meta: dict) -> str:
    """One-line human-readable summary from a meta dict."""
    clip  = meta.get("clip", {})
    sig   = meta.get("signal_metrics", {})
    parts = []
    if clip.get("flight_phase"):
        parts.append(f"phase={clip['flight_phase']}")
    if clip.get("duration_s"):
        parts.append(f"dur={clip['duration_s']:.1f}s")
    if sig.get("rms_dbfs") is not None:
        parts.append(f"rms={sig['rms_dbfs']:.1f}dBFS")
    if sig.get("snr_db") is not None:
        parts.append(f"snr={sig['snr_db']:.1f}dB")
    if sig.get("dominant_freq_hz") is not None:
        parts.append(f"f0≈{sig['dominant_freq_hz']:.1f}Hz")
    return "  ".join(parts) if parts else "(no meta)"


# ══════════════════════════════════════════════════════════════════════════════
# Core spectral helpers
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_dominant_freq(y: np.ndarray, sr: int,
                             f_min: float = 50.0,
                             f_max: float = 700.0) -> float:
    """Return the most prominent spectral peak in [f_min, f_max] Hz."""
    try:
        import librosa
        S      = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
        Sm     = S.mean(axis=1)
        freqs  = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask   = (freqs >= f_min) & (freqs <= f_max)
        return float(freqs[mask][int(np.argmax(Sm[mask]))])
    except Exception:
        return float("nan")


def _bpf_energy_ratio(y: np.ndarray, sr: int,
                       bpf_hz: float, bw_hz: float = 20.0,
                       n_harmonics: int = 4) -> float:
    """Fraction of signal power in BPF band + harmonics."""
    try:
        import scipy.signal
        y  = y.astype(np.float64)
        nyq = sr / 2.0
        total = float(np.mean(y ** 2)) + 1e-10
        bpf   = 0.0
        for k in range(1, n_harmonics + 1):
            fc = bpf_hz * k
            if fc + bw_hz >= nyq:
                break
            lo = max(fc - bw_hz, 1.0)
            hi = min(fc + bw_hz, nyq - 1.0)
            sos = scipy.signal.butter(4, [lo / nyq, hi / nyq],
                                      btype="band", output="sos")
            band = scipy.signal.sosfilt(sos, y)
            bpf += float(np.mean(band ** 2))
        return float(np.clip(bpf / total, 0.0, 1.0))
    except Exception:
        return float("nan")


def _segment_snr(y: np.ndarray, sr: int,
                 bpf_hz: float, bw_hz: float = 20.0) -> float:
    """Rough in-band SNR estimate: power in BPF band vs. power outside it."""
    try:
        import scipy.signal
        y   = y.astype(np.float64)
        nyq = sr / 2.0
        lo  = max(bpf_hz - bw_hz, 1.0)
        hi  = min(bpf_hz + bw_hz, nyq - 1.0)
        sos_in  = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype="band", output="sos")
        sos_out = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype="bandstop", output="sos")
        p_in  = float(np.mean(scipy.signal.sosfilt(sos_in,  y) ** 2)) + 1e-10
        p_out = float(np.mean(scipy.signal.sosfilt(sos_out, y) ** 2)) + 1e-10
        return float(10 * np.log10(p_in / p_out))
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _plot_mems_dashboard(
    segments: List[dict],
    title: str,
    cfg,
    save_path: Optional[Path] = None,
):
    """
    5-panel dark dashboard for a single-channel MEMS analysis:

    [0] Waveform + RMS energy over time
    [1] Mel spectrogram (concatenated)
    [2] Detection timeline (CNN / heuristic / hybrid probabilities)
    [3] BPF energy ratio per segment
    [4] Detection score gauge (peak probability)
    """
    S = _plot_style()
    fig = plt.figure(figsize=(20, 9), facecolor=S["bg"])
    fig.suptitle(f"🚁 MEMS Analysis — {title}",
                 fontsize=13, color=S["accent"], fontweight="bold", y=0.98)
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
    ax0  = fig.add_subplot(gs[0, 0])   # waveform
    ax1  = fig.add_subplot(gs[0, 1])   # mel
    ax2  = fig.add_subplot(gs[0, 2])   # detection timeline
    ax3  = fig.add_subplot(gs[1, 0])   # BPF energy ratio
    ax4  = fig.add_subplot(gs[1, 1])   # dominant freq
    ax5  = fig.add_subplot(gs[1, 2])   # gauge

    _apply_dark(fig, [ax0, ax1, ax2, ax3, ax4, ax5])

    ts   = [s["t_start"] for s in segments]
    dur  = cfg.TARGET_DURATION

    # ── [0] Waveform + RMS ────────────────────────────────────────────────────
    wave_all = np.concatenate([s["waveform"] for s in segments])
    t_wave   = np.linspace(0, ts[-1] + dur if ts else dur, len(wave_all))
    ax0.plot(t_wave, wave_all, color=S["accent"], lw=0.5, alpha=0.7)
    ax0_r = ax0.twinx()
    ax0_r.plot(ts, [s["rms_db"] for s in segments],
               "o-", color=S["warn"], ms=4, lw=1.5, label="RMS dB")
    ax0_r.tick_params(colors=S["text"], labelcolor=S["text"])
    ax0_r.yaxis.label.set_color(S["text"])
    ax0_r.set_ylabel("RMS (dB)", color=S["text"])
    for sp in ax0_r.spines.values():
        sp.set_color(S["spine"])
    ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Amplitude")
    ax0.set_title("Waveform + RMS")

    # ── [1] Mel spectrogram ───────────────────────────────────────────────────
    mels = [s["mel"] for s in segments if s.get("mel") is not None]
    if mels:
        mel_cat = np.concatenate(mels, axis=1)
        extent  = [0, ts[-1] + dur if ts else dur, 0, cfg.SR // 2 / 1000]
        img = ax1.imshow(mel_cat, aspect="auto", origin="lower",
                         cmap="magma", extent=extent)
        cbar = plt.colorbar(img, ax=ax1, label="dB")
        try:
            from .visualization import _style_colorbar
            _style_colorbar(cbar)
        except Exception:
            pass
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Freq (kHz)")
    ax1.set_title("Mel Spectrogram")

    # ── [2] Detection timeline ────────────────────────────────────────────────
    probs = [s["prob"]                           for s in segments]
    cnns  = [s.get("cnn_prob",   float("nan"))   for s in segments]
    heurs = [s.get("heur_prob",  float("nan"))   for s in segments]
    cols  = [S["ok"] if s["detected"] else S["err"] for s in segments]
    ax2.bar(ts, probs, width=dur * 0.8, color=cols, alpha=0.55, label="Hybrid")
    ax2.fill_between(ts, probs, alpha=0.12, color=S["text"])
    if not all(math.isnan(v) for v in cnns):
        ax2.plot(ts, cnns,  "-o", color=S["accent"], ms=4, lw=1.5, label="CNN")
    if not all(math.isnan(v) for v in heurs):
        ax2.plot(ts, heurs, "--s", color=S["purple"], ms=4, lw=1.5, label="Heuristic")
    ax2.axhline(cfg.DETECTION_THRESHOLD, color=S["warn"], lw=1.5, ls="--",
                label=f"Thr={cfg.DETECTION_THRESHOLD:.2f}")
    ax2.set_xlim(left=0); ax2.set_ylim(0, 1.08)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Probability")
    ax2.set_title("Detection Timeline")
    leg = ax2.legend(facecolor=S["panel_alt"], edgecolor=S["spine"])
    try:
        from .visualization import _style_legend
        _style_legend(leg)
    except Exception:
        pass

    # ── [3] BPF energy ratio per segment ─────────────────────────────────────
    bpf_vals = [s.get("bpf_ratio", float("nan")) for s in segments]
    valid_bpf = [v for v in bpf_vals if not math.isnan(v)]
    bpf_cols  = []
    for v in bpf_vals:
        if math.isnan(v):
            bpf_cols.append(S["muted"])
        elif v >= 0.30:
            bpf_cols.append(S["ok"])
        elif v >= 0.10:
            bpf_cols.append(S["warn"])
        else:
            bpf_cols.append(S["err"])
    ax3.bar(ts, [0 if math.isnan(v) else v for v in bpf_vals],
            width=dur * 0.8, color=bpf_cols, alpha=0.85)
    if valid_bpf:
        ax3.axhline(np.mean(valid_bpf), color=S["accent"], lw=1.2, ls="--",
                    label=f"Mean={np.mean(valid_bpf):.2f}")
        ax3.legend(facecolor=S["panel_alt"], fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("BPF energy ratio")
    ax3.set_title("BPF Energy Ratio (drone signature)")

    # ── [4] Dominant frequency per segment ───────────────────────────────────
    dom_freqs = [s.get("dom_freq_hz", float("nan")) for s in segments]
    valid_df  = [(t, f) for t, f in zip(ts, dom_freqs) if not math.isnan(f)]
    if valid_df:
        t_df, f_df = zip(*valid_df)
        ax4.scatter(t_df, f_df, color=S["accent"], s=55, zorder=4)
        ax4.plot(t_df, f_df, "-", color=S["accent"], lw=1.2, alpha=0.6)
        # Shade known drone BPF bands
        for label, (lo, _, hi, _) in [
            ("mavic_pro",   (160, 209, 340, 4)),
            ("mavic_mini",  (260, 360, 620, 4)),
        ]:
            ax4.axhspan(lo, hi, alpha=0.08, color=S["ok"], label=label)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Frequency (Hz)")
    ax4.set_title("Dominant Frequency (50–700 Hz)")
    ax4.set_ylim(0, 750)
    if valid_df:
        ax4.legend(facecolor=S["panel_alt"], fontsize=7)

    # ── [5] Detection score gauge ─────────────────────────────────────────────
    peak_prob  = float(max((s["prob"] for s in segments), default=0.0))
    theta_rng  = np.linspace(np.pi, 0, 200)
    ax5.set_xlim(-1.2, 1.2); ax5.set_ylim(-0.1, 1.2)
    ax5.plot(np.cos(theta_rng), np.sin(theta_rng), lw=18, color=S["panel"])
    fill_theta = np.linspace(np.pi, np.pi * (1 - peak_prob), 200)
    col = S["ok"] if peak_prob >= cfg.DETECTION_THRESHOLD else S["err"]
    ax5.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=col)
    needle = np.pi * (1 - peak_prob)
    ax5.annotate("", xy=(0.8 * np.cos(needle), 0.8 * np.sin(needle)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=S["text"], lw=2))
    verdict = "DRONE" if peak_prob >= cfg.DETECTION_THRESHOLD else "CLEAR"
    ax5.text(0, -0.08, f"{peak_prob:.3f}", ha="center",
             fontsize=16, fontweight="bold", color=col)
    ax5.text(0, 0.55, verdict, ha="center", fontsize=11, color=col)
    ax5.text(0, 0.30, "(detection only —\nlocalisation N/A\nfor single-channel)",
             ha="center", fontsize=7, color=S["muted"], linespacing=1.4)
    ax5.axis("off"); ax5.set_title("Detection Score", color=S["text"])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Dashboard saved: {save_path}")

    _show_inline(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def analyse_mems_file(
    audio_path: str,
    cfg=None,
    meta_path: Optional[str] = None,
    n_segments: int = 10,
    bpf_hz: Optional[float] = None,
    show_plot: bool = True,
    save_plot: bool = True,
) -> dict:
    """
    Analyse a single-channel MEMS WAV for drone detection.

    Parameters
    ──────────
    audio_path  : path to the .wav file
    cfg         : Config instance (uses module singleton if None)
    meta_path   : optional path to companion _meta.json
    n_segments  : number of non-overlapping analysis windows
    bpf_hz      : known blade-pass frequency in Hz; estimated if None
    show_plot   : display the 5-panel dashboard inline (Colab / Jupyter)
    save_plot   : also save as PNG to cfg.DRIVE_PLOTS

    Returns
    ───────
    dict with keys:
        detected, probability, n_detected_segments,
        segments (list of per-segment dicts),
        dom_freq_hz, bpf_ratio_mean, meta
    """
    cfg = cfg or _cfg_default()
    ap  = _get_ap(cfg)
    detect = _get_detect(cfg)
    heuristic_detect = _get_heuristic()

    # ── Load audio ────────────────────────────────────────────────────────────
    y_full  = ap.load(str(audio_path), mono=True)
    total_s = len(y_full) / cfg.SR
    seg_n   = int(cfg.TARGET_DURATION * cfg.SR)
    hop     = seg_n  # non-overlapping windows

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta = {}
    if meta_path is None:
        # Try to find a companion _meta.json automatically
        auto_meta = Path(str(audio_path)).with_name(
            Path(audio_path).stem + "_meta.json"
        )
        if auto_meta.exists():
            meta_path = str(auto_meta)
    if meta_path:
        meta = load_meta(meta_path)

    stem = Path(audio_path).stem
    print(f"\n🎵 MEMS: {Path(audio_path).name}  ({total_s:.1f}s)")
    if meta:
        print(f"   Meta: {_meta_summary(meta)}")
    print(f"   ⚠️  Single-channel recording — localization is not available.\n"
          f"   Running detection-only analysis over {n_segments} segments.")

    # Use dominant freq from meta if available and bpf_hz not provided
    if bpf_hz is None:
        bpf_hz = (meta.get("signal_metrics", {}).get("dominant_freq_hz")
                  or _estimate_dominant_freq(y_full, cfg.SR))
        print(f"   BPF estimate: {bpf_hz:.1f} Hz")

    segments = []

    for seg_i in range(n_segments):
        start = min(seg_i * hop, max(0, len(y_full) - seg_n))
        audio = y_full[start : start + seg_n]
        if len(audio) < seg_n:
            audio = np.pad(audio, (0, seg_n - len(audio)))

        t_s    = start / cfg.SR
        mel_fr = ap.mel(ap.pad_or_truncate(audio))
        rms_db = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-8))

        # Detection: replicate across 3 channels (all-same — no TDOA signal)
        det = detect([audio, audio, audio], cfg)
        # Also run heuristic standalone for richer features
        heur = heuristic_detect(audio, cfg)

        dom_f = _estimate_dominant_freq(audio, cfg.SR)
        bpf_r = _bpf_energy_ratio(audio, cfg.SR, bpf_hz)
        snr_e = _segment_snr(audio, cfg.SR, bpf_hz)

        segments.append({
            "seg":       seg_i + 1,
            "t_start":   t_s,
            "detected":  det["detected"],
            "prob":      det["probability"],
            "cnn_prob":  det.get("cnn_probability",       float("nan")),
            "heur_prob": det.get("heuristic_probability", float("nan")),
            "rms_db":    rms_db,
            "dom_freq_hz": dom_f,
            "bpf_ratio": bpf_r,
            "snr_est_db": snr_e,
            "mel":       mel_fr,
            "waveform":  audio,
            # Localization fields are explicitly None — single channel
            "azimuth_deg": None,
            "distance_m":  None,
            "height_m":    None,
        })

        icon = "🚁" if det["detected"] else "🌳"
        print(f"  Seg {seg_i+1:3d}  {icon}  "
              f"conf={det['probability']:.3f}  rms={rms_db:.1f}dB  "
              f"f0={dom_f:.0f}Hz  bpf={bpf_r:.2f}")

    n_det    = sum(s["detected"] for s in segments)
    peak_p   = max(s["prob"] for s in segments)
    bpf_vals = [s["bpf_ratio"] for s in segments if not math.isnan(s["bpf_ratio"])]
    bpf_mean = float(np.mean(bpf_vals)) if bpf_vals else float("nan")
    dom_freq = float(np.median([s["dom_freq_hz"] for s in segments
                                if not math.isnan(s["dom_freq_hz"])])) \
               if any(not math.isnan(s["dom_freq_hz"]) for s in segments) \
               else float("nan")

    print(f"\n  📊 {n_det}/{n_segments} detected  |  "
          f"peak_prob={peak_p:.3f}  bpf_mean={bpf_mean:.2f}  dom_f={dom_freq:.0f}Hz")
    print(f"  ℹ️  Azimuth/distance/height: N/A (single-channel recording)")

    if show_plot or save_plot:
        save_path = None
        if save_plot:
            try:
                out_dir = cfg.DRIVE_PLOTS
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / f"mems_{stem}.png"
            except Exception:
                pass
        _plot_mems_dashboard(segments, stem, cfg, save_path=save_path)

    return {
        "file":               str(audio_path),
        "detected":           n_det > 0,
        "probability":        peak_p,
        "n_detected_segments": n_det,
        "n_segments":         n_segments,
        "total_duration_s":   total_s,
        "dom_freq_hz":        dom_freq,
        "bpf_hz_used":        bpf_hz,
        "bpf_ratio_mean":     bpf_mean,
        "segments":           segments,
        "meta":               meta,
        # Always None — make the limitation explicit in the return value
        "azimuth_deg":  None,
        "distance_m":   None,
        "height_m":     None,
        "localization_available": False,
    }


def analyse_mems_with_meta(
    audio_path: str,
    meta_path: str,
    cfg=None,
    **kwargs,
) -> dict:
    """Convenience wrapper that always loads the companion meta JSON."""
    return analyse_mems_file(audio_path, cfg=cfg, meta_path=meta_path, **kwargs)


def batch_analyse_mems(
    folder: str,
    cfg=None,
    pattern: str = "*.wav",
    bpf_hz: Optional[float] = None,
    show_plots: bool = False,
    save_plots: bool = True,
    max_files: Optional[int] = None,
) -> dict:
    """
    Run analyse_mems_file() over every WAV in a folder and print a summary.

    Parameters
    ──────────
    folder      : directory containing MEMS WAV files
    cfg         : Config instance
    pattern     : glob pattern for audio files (default "*.wav")
    bpf_hz      : shared BPF Hz if known; estimated per-file if None
    show_plots  : display dashboards inline (slow for large batches)
    save_plots  : save PNG to cfg.DRIVE_PLOTS per file
    max_files   : cap number of files processed

    Returns
    ───────
    dict with keys:
        results (List[dict]), n_detected, detection_rate,
        mean_bpf_ratio, mean_dom_freq_hz
    """
    cfg   = cfg or _cfg_default()
    root  = Path(folder)
    wavs  = sorted(root.glob(pattern))
    if max_files is not None:
        wavs = wavs[:max_files]

    if not wavs:
        print(f"⚠️  No files matching '{pattern}' in {folder}")
        return {"results": [], "n_detected": 0, "detection_rate": 0.0}

    print(f"\n📂 Batch MEMS analysis: {len(wavs)} files in {root.name}/")
    print("   ⚠️  Localization is not available for single-channel recordings.\n")

    results      = []
    n_detected   = 0
    bpf_ratios   = []
    dom_freqs    = []

    for i, wav in enumerate(wavs, start=1):
        print(f"\n[{i}/{len(wavs)}] {wav.name}")
        try:
            r = analyse_mems_file(
                str(wav), cfg=cfg, bpf_hz=bpf_hz,
                show_plot=show_plots, save_plot=save_plots,
            )
            results.append(r)
            if r["detected"]:
                n_detected += 1
            if not math.isnan(r["bpf_ratio_mean"]):
                bpf_ratios.append(r["bpf_ratio_mean"])
            if r["dom_freq_hz"] and not math.isnan(r["dom_freq_hz"]):
                dom_freqs.append(r["dom_freq_hz"])
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            results.append({"file": str(wav), "error": str(e), "detected": False})

    det_rate = n_detected / max(len(wavs), 1)
    mean_bpf = float(np.mean(bpf_ratios)) if bpf_ratios else float("nan")
    mean_f0  = float(np.mean(dom_freqs))  if dom_freqs  else float("nan")

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY — {len(wavs)} files")
    print(f"  Detection rate  : {det_rate:.1%}  ({n_detected}/{len(wavs)})")
    if not math.isnan(mean_bpf):
        print(f"  Mean BPF ratio  : {mean_bpf:.3f}")
    if not math.isnan(mean_f0):
        print(f"  Mean dom. freq  : {mean_f0:.1f} Hz")
    print(f"  Localisation    : ❌ N/A — single-channel MEMS recording")
    print(f"{'='*60}\n")

    return {
        "results":         results,
        "n_detected":      n_detected,
        "detection_rate":  det_rate,
        "mean_bpf_ratio":  mean_bpf,
        "mean_dom_freq_hz": mean_f0,
        "localization_available": False,
    }


def build_mems_report(results: List[dict]) -> None:
    """
    Print a compact per-file table from a list of analyse_mems_file() outputs.

    Columns: filename | detected | prob | dom_freq | bpf_ratio | duration
    """
    header = (f"{'File':40s}  {'Det':3s}  {'Prob':5s}  "
              f"{'f0(Hz)':7s}  {'BPF':5s}  {'Dur(s)':6s}")
    print(header)
    print("-" * len(header))
    for r in results:
        name = Path(r.get("file", "?")).name[:40]
        det  = "YES" if r.get("detected") else "no "
        prob = f"{r.get('probability', float('nan')):5.3f}"
        f0   = (f"{r['dom_freq_hz']:7.1f}"
                if r.get("dom_freq_hz") and not math.isnan(r["dom_freq_hz"])
                else "      -")
        bpf  = (f"{r['bpf_ratio_mean']:5.3f}"
                if r.get("bpf_ratio_mean") is not None
                   and not math.isnan(r.get("bpf_ratio_mean", float("nan")))
                else "    -")
        dur  = (f"{r['total_duration_s']:6.1f}"
                if r.get("total_duration_s") is not None else "     -")
        print(f"{name:40s}  {det}  {prob}  {f0}  {bpf}  {dur}")
