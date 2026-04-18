# -*- coding: utf-8 -*-
"""
dataset_visualizer.py
─────────────────────
Visualise synthetic dataset samples and key metrics BEFORE training, so
you can inspect the data quality and distribution at a glance.

Public API
──────────
plot_dataset_overview(cfg, n_samples)
    A single 3×3 dashboard figure covering:
      1. Mel spectrogram (single drone, indoor)
      2. Mel spectrogram (multi-drone, outdoor)
      3. Mel spectrogram (low-SNR far-field)
      4. IPD waveform across 3 mic pairs
      5. BPF spectral fingerprint by drone type
      6. Noise floor comparison (indoor vs outdoor)
      7. Spatial distribution polar plot
      8. Distance/height histogram
      9. Per-drone-type BPF energy ratio box plot

plot_sample_session(cfg, scenario, array_name)
    One session in detail: waveform, mel, IPD, spectrum + BPF markers.

plot_position_coverage(cfg, n_samples)
    2-D scatter of sampled positions around the array, coloured by distance.

plot_noise_profiles(cfg)
    Side-by-side PSD of real indoor / outdoor noise models.

Usage
─────
from drone_detection.dataset_visualizer import (
    plot_dataset_overview,
    plot_sample_session,
    plot_position_coverage,
    plot_noise_profiles,
)

# Full dashboard — call this before training
plot_dataset_overview(config, n_samples=200)

# Detailed single session
plot_sample_session(config, scenario="outdoor_fly")

# Position coverage
plot_position_coverage(config, n_samples=300)
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── lazy imports ──────────────────────────────────────────────────────────────

def _cfg(cfg=None):
    if cfg is not None:
        return cfg
    from .config import config
    return config

def _ap(cfg):
    from .audio_processing import AudioProcessor
    return AudioProcessor(cfg)

def _synth(mics, src_xy, **kw):
    from .audio_processing import synthesise_drone
    return synthesise_drone(mics, src_xy, **kw)

def _make_noise(n, sr, profile, amplitude):
    from .audio_processing import _make_noise as _mn
    return _mn(n, sr, profile=profile, amplitude=amplitude)

def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")   # safe for Colab + headless
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    return plt, gridspec

def _wrap(a):
    return float((a + 180.0) % 360.0 - 180.0)

# ── colour palette (colour-blind safe) ────────────────────────────────────────
_DRONE_COLOURS = {
    "mavic_pro":   "#1976D2",
    "mavic_2_pro": "#43A047",
    "mavic_mini":  "#FB8C00",
    "generic_quad":"#8E24AA",
    "hexarotor":   "#E53935",
}
_NOISE_COLOURS = {"indoor": "#00ACC1", "outdoor": "#F4511E"}
_MIC_COLOURS   = ["#E53935", "#43A047", "#1976D2"]   # ch0, ch1, ch2


# ── position sampler (re-used across plots) ───────────────────────────────────

def _sample_positions(cfg, rng, n=200,
                      dist_range=(1.0, 20.0), height_range=(0.5, 15.0)):
    cx, cy = cfg.ARRAY_CENTER
    azs, dists, hts, dtypes = [], [], [], []
    drone_pool = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]
    weights    = [0.30, 0.30, 0.25, 0.15]
    for _ in range(n):
        az = float(rng.uniform(-180, 180))
        d  = float(np.exp(rng.uniform(np.log(dist_range[0]),
                                      np.log(dist_range[1]))))
        h  = float(rng.uniform(*height_range))
        dt = str(rng.choice(drone_pool, p=weights))
        azs.append(az); dists.append(d); hts.append(h); dtypes.append(dt)
    return np.array(azs), np.array(dists), np.array(hts), dtypes


# ── synthesise one clean example per scenario ─────────────────────────────────

def _make_example(cfg, ap, scenario: str, seed: int = 0):
    """
    Return (channels, az_deg, dist_m, ht_m, drone_type, noise_profile).
    channels is a list of 3 float32 arrays at cfg.SR.
    """
    from .synthetic_dataset_generator import SCENARIO_SPECS
    rng  = np.random.default_rng(seed)
    spec = SCENARIO_SPECS.get(scenario, SCENARIO_SPECS["indoor_hover"])
    cfg.set_array_geometry(spec.get("array", "uavirbase"))
    cx, cy = cfg.ARRAY_CENTER

    dtype = str(rng.choice(spec["drone_types"]))
    nl    = float(rng.uniform(*spec["noise_range"]))
    npf   = spec["noise_profile"]
    if npf == "mixed":
        npf = str(rng.choice(["indoor", "outdoor"]))

    dr    = spec["dist_range"]
    hr    = spec["height_range"]
    dist  = float(np.exp(rng.uniform(np.log(dr[0]), np.log(dr[1]))))
    ht    = float(rng.uniform(*hr))
    az    = float(rng.uniform(-180, 180))
    az_r  = math.radians(az)
    src   = np.array([cx + dist * math.cos(az_r),
                      cy + dist * math.sin(az_r)], dtype=np.float32)

    chs = _synth(cfg.MIC_POSITIONS, src,
                 noise_level=nl, drone_type=dtype,
                 noise_profile=npf, cfg=cfg)
    return chs, _wrap(az), dist, ht, dtype, npf


# ══════════════════════════════════════════════════════════════════════════════
# 1. Full dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset_overview(
    cfg=None,
    n_samples: int = 300,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    3×3 dashboard showing the full synthetic dataset at a glance.

    Panels
    ──────
    Row 1 — Sample spectrograms
      [0] Indoor hover mel (Mavic Pro)
      [1] Multi-drone outdoor mel
      [2] Low-SNR outdoor mel (Mavic Mini)

    Row 2 — Signal characteristics
      [3] IPD (TDOA delay) waveforms across 3 mic pairs
      [4] BPF spectral fingerprints by drone type
      [5] Noise floor PSD: indoor vs outdoor

    Row 3 — Position/label statistics
      [6] Polar scatter of sampled positions
      [7] Distance and height histograms
      [8] BPF energy ratio by drone type (box plot)
    """
    cfg = _cfg(cfg)
    ap  = _ap(cfg)
    plt, gs_mod = _import_mpl()
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0F1117")
    GS  = gs_mod.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    _style_ax = lambda ax: (
        ax.set_facecolor("#1A1D27"),
        [s.set_color("#3A3D4D") for s in ax.spines.values()],
        ax.tick_params(colors="#9CA3AF", labelsize=8),
        ax.xaxis.label.set_color("#D1D5DB"),
        ax.yaxis.label.set_color("#D1D5DB"),
        ax.title.set_color("#F9FAFB"),
    )

    # ── Panel 0: Indoor hover mel ─────────────────────────────────────────
    ax0 = fig.add_subplot(GS[0, 0])
    chs_in, az_in, di_in, ht_in, dt_in, _ = _make_example(
        cfg, ap, "indoor_hover", seed=seed)
    mel_in = ap.mel(ap.pad_or_truncate(chs_in[0]))
    ax0.imshow(mel_in, aspect="auto", origin="lower",
               cmap="inferno", interpolation="nearest",
               extent=[0, cfg.TARGET_DURATION, 0, cfg.SR // 2 / 1000])
    ax0.set_title(f"Indoor hover  ({dt_in.replace('_',' ')})", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Time (s)", fontsize=8); ax0.set_ylabel("Freq (kHz)", fontsize=8)
    ax0.text(0.02, 0.94, f"az={az_in:.0f}°  d={di_in:.1f}m  h={ht_in:.1f}m",
             transform=ax0.transAxes, fontsize=7, color="#A5B4FC",
             verticalalignment="top")
    _style_ax(ax0)

    # ── Panel 1: Multi-drone mel ──────────────────────────────────────────
    ax1 = fig.add_subplot(GS[0, 1])
    from .synthetic_dataset_generator import _session as _gen_session
    cfg.set_array_geometry("uavirbase")
    cx, cy = cfg.ARRAY_CENTER
    srcs_m = []
    for a in [40.0, -110.0]:
        r = math.radians(a)
        srcs_m.append((np.array([cx + 5*math.cos(r), cy + 5*math.sin(r)],
                                dtype=np.float32),
                       random.choice(["mavic_pro", "mavic_mini"])))
    import tempfile, shutil, soundfile as sf
    _tmp = Path(tempfile.mkdtemp())
    try:
        _session(cfg, ap, cfg.MIC_POSITIONS, srcs_m, 0.05, "outdoor", _tmp, "multi_ex")
        y_m, _ = sf.read(str(_tmp / "multi_ex_ch0.wav"), dtype="float32")
    finally:
        shutil.rmtree(_tmp)
    mel_m = ap.mel(ap.pad_or_truncate(y_m))
    ax1.imshow(mel_m, aspect="auto", origin="lower",
               cmap="inferno", interpolation="nearest",
               extent=[0, cfg.TARGET_DURATION, 0, cfg.SR // 2 / 1000])
    ax1.set_title("Multi-drone outdoor (2 drones)", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=8); ax1.set_ylabel("Freq (kHz)", fontsize=8)
    _style_ax(ax1)

    # ── Panel 2: Low-SNR mel ──────────────────────────────────────────────
    ax2 = fig.add_subplot(GS[0, 2])
    chs_lo, az_lo, di_lo, ht_lo, dt_lo, _ = _make_example(
        cfg, ap, "low_snr", seed=seed + 7)
    mel_lo = ap.mel(ap.pad_or_truncate(chs_lo[0]))
    ax2.imshow(mel_lo, aspect="auto", origin="lower",
               cmap="inferno", interpolation="nearest",
               extent=[0, cfg.TARGET_DURATION, 0, cfg.SR // 2 / 1000])
    ax2.set_title(f"Low-SNR outdoor  ({dt_lo.replace('_',' ')})",
                  fontsize=9, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=8); ax2.set_ylabel("Freq (kHz)", fontsize=8)
    ax2.text(0.02, 0.94, f"az={az_lo:.0f}°  d={di_lo:.1f}m",
             transform=ax2.transAxes, fontsize=7, color="#FCA5A5",
             verticalalignment="top")
    _style_ax(ax2)

    # ── Panel 3: IPD waveforms ────────────────────────────────────────────
    ax3 = fig.add_subplot(GS[1, 0])
    chs_ipd, _, _, _, dtype_ipd, _ = _make_example(
        cfg, ap, "indoor_hover", seed=seed + 1)
    n_show = int(cfg.SR * 0.04)   # 40 ms
    t_ms   = np.arange(n_show) / cfg.SR * 1000
    pairs  = [(0, 1, "ch0-ch1"), (0, 2, "ch0-ch2"), (1, 2, "ch1-ch2")]
    for k, (i, j, lbl) in enumerate(pairs):
        yi = ap.pad_or_truncate(chs_ipd[i])[:n_show]
        yj = ap.pad_or_truncate(chs_ipd[j])[:n_show]
        ax3.plot(t_ms, yi, color=_MIC_COLOURS[k], lw=0.8,
                 alpha=0.85, label=f"ch{i}")
        ax3.plot(t_ms, yj, color=_MIC_COLOURS[k], lw=0.8,
                 alpha=0.40, linestyle="--")
    ax3.set_title("IPD: mic-pair waveforms (40 ms)", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Time (ms)", fontsize=8); ax3.set_ylabel("Amplitude", fontsize=8)
    ax3.legend(fontsize=7, framealpha=0.2, labelcolor="white")
    _style_ax(ax3)

    # ── Panel 4: BPF spectral fingerprints ───────────────────────────────
    ax4 = fig.add_subplot(GS[1, 1])
    from .config import DRONE_BPF_PROFILES
    import librosa
    freq_ax = librosa.fft_frequencies(sr=cfg.SR, n_fft=2048)
    for dtype, colour in _DRONE_COLOURS.items():
        if dtype not in DRONE_BPF_PROFILES:
            continue
        f_lo, f_mid, f_hi, n_harm = DRONE_BPF_PROFILES[dtype]
        chs_fp, _, _, _, _, _ = _make_example(
            cfg, ap, "indoor_hover", seed=seed + hash(dtype) % 100)
        S = np.abs(librosa.stft(
            ap.pad_or_truncate(
                _synth(cfg.MIC_POSITIONS,
                       np.array([cfg.ARRAY_CENTER[0]+3, cfg.ARRAY_CENTER[1]]),
                       fundamental=f_mid, drone_type=dtype,
                       noise_level=0.01, noise_profile="indoor",
                       cfg=cfg)[0]
            ).astype(np.float32), n_fft=2048, hop_length=512
        )).mean(axis=1)
        mask = freq_ax <= 1200
        S_norm = S[mask] / (S[mask].max() + 1e-8)
        ax4.plot(freq_ax[mask], S_norm, color=colour, lw=1.2, alpha=0.85,
                 label=dtype.replace("_", " "))
        ax4.axvline(f_mid, color=colour, lw=0.6, linestyle=":", alpha=0.7)
    ax4.set_title("BPF spectral fingerprints by drone type", fontsize=9, fontweight="bold")
    ax4.set_xlabel("Frequency (Hz)", fontsize=8); ax4.set_ylabel("Norm. power", fontsize=8)
    ax4.legend(fontsize=6.5, framealpha=0.2, labelcolor="white", ncol=2)
    _style_ax(ax4)

    # ── Panel 5: Noise floor PSD ──────────────────────────────────────────
    ax5 = fig.add_subplot(GS[1, 2])
    import librosa
    n_noise = int(cfg.SR * 4)
    freq_n  = librosa.fft_frequencies(sr=cfg.SR, n_fft=2048)
    for profile, colour in _NOISE_COLOURS.items():
        n_sig = _make_noise(n_noise, cfg.SR, profile=profile, amplitude=0.03)
        S_n   = np.abs(librosa.stft(n_sig.astype(np.float32),
                                     n_fft=2048, hop_length=512)).mean(axis=1)
        S_db  = 20 * np.log10(S_n + 1e-10)
        ax5.plot(freq_n, S_db, color=colour, lw=1.0, alpha=0.85, label=profile)
    ax5.set_title("Synthetic noise floor PSD", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Frequency (Hz)", fontsize=8)
    ax5.set_ylabel("Power (dB)", fontsize=8)
    ax5.set_xscale("log"); ax5.set_xlim(20, cfg.SR // 2)
    ax5.legend(fontsize=8, framealpha=0.2, labelcolor="white")
    _style_ax(ax5)

    # ── Panel 6: Polar position scatter ──────────────────────────────────
    ax6 = fig.add_subplot(GS[2, 0], polar=True)
    ax6.set_facecolor("#1A1D27")
    azs, dists, hts, dtypes_s = _sample_positions(cfg, rng, n=n_samples)
    az_rad = np.radians(azs)
    sc = ax6.scatter(az_rad, dists, c=dists, cmap="plasma",
                     alpha=0.5, s=12, linewidths=0)
    ax6.set_theta_zero_location("N")
    ax6.set_theta_direction(-1)
    ax6.tick_params(colors="#9CA3AF", labelsize=7)
    ax6.set_title("Position distribution (polar)", fontsize=9,
                  fontweight="bold", color="#F9FAFB", pad=14)
    cb = plt.colorbar(sc, ax=ax6, pad=0.12, shrink=0.7)
    cb.set_label("Distance (m)", color="#D1D5DB", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#9CA3AF", labelsize=6)

    # ── Panel 7: Distance + height histograms ─────────────────────────────
    ax7 = fig.add_subplot(GS[2, 1])
    _style_ax(ax7)
    ax7b = ax7.twinx()
    bins = np.linspace(0, 22, 25)
    ax7.hist(dists, bins=bins, color="#60A5FA", alpha=0.65, label="Distance (m)", density=True)
    ax7b.hist(hts, bins=np.linspace(0, 16, 20), color="#F472B6",
              alpha=0.55, label="Height (m)", density=True)
    ax7.set_xlabel("Distance / height (m)", fontsize=8)
    ax7.set_ylabel("Density", fontsize=8, color="#60A5FA")
    ax7b.set_ylabel("Density", fontsize=8, color="#F472B6")
    ax7b.tick_params(colors="#9CA3AF", labelsize=7)
    ax7.set_title("Distance & height distributions", fontsize=9, fontweight="bold")
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7b.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, framealpha=0.2, labelcolor="white")
    _style_ax(ax7)

    # ── Panel 8: BPF energy ratio box plot ───────────────────────────────
    ax8 = fig.add_subplot(GS[2, 2])
    _style_ax(ax8)
    from .config import DRONE_BPF_ENERGY_RATIOS, DRONE_BPF_PROFILES
    from .audio_processing import AudioProcessor as AP
    bpf_data: Dict[str, List[float]] = {}
    n_bpf = 30
    for dt_b, colour in _DRONE_COLOURS.items():
        if dt_b not in DRONE_BPF_PROFILES:
            continue
        ratios = []
        for s in range(n_bpf):
            chs_b = _synth(cfg.MIC_POSITIONS,
                           np.array([cfg.ARRAY_CENTER[0] + 4, cfg.ARRAY_CENTER[1]]),
                           drone_type=dt_b, noise_level=0.04,
                           noise_profile="mixed", cfg=cfg)
            y_b   = ap.pad_or_truncate(chs_b[0])
            f_lo, f_mid, f_hi, _ = DRONE_BPF_PROFILES[dt_b]
            ratios.append(float(ap.compute_bpf_energy_ratio(y_b, f_mid)))
        bpf_data[dt_b] = ratios
    labels_b = [k.replace("_", "\n") for k in bpf_data]
    bp = ax8.boxplot(list(bpf_data.values()), labels=labels_b, patch_artist=True,
                     medianprops=dict(color="white", lw=1.5),
                     whiskerprops=dict(color="#9CA3AF"),
                     capprops=dict(color="#9CA3AF"),
                     flierprops=dict(marker=".", color="#9CA3AF", markersize=3))
    for patch, (dt_b, col) in zip(bp["boxes"], _DRONE_COLOURS.items()):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    # Overlay measured means from config
    for xi, dt_b in enumerate(bpf_data, start=1):
        if dt_b in DRONE_BPF_ENERGY_RATIOS:
            ax8.axhline(DRONE_BPF_ENERGY_RATIOS[dt_b], xmin=(xi-1.4)/(len(bpf_data)),
                        xmax=(xi-0.6)/(len(bpf_data)),
                        color="white", lw=1.2, linestyle="--", alpha=0.8)
    ax8.set_title("BPF energy ratio by drone type\n(dashed = measured Q2 mean)",
                  fontsize=9, fontweight="bold")
    ax8.set_ylabel("BPF energy ratio", fontsize=8)
    ax8.tick_params(axis="x", labelsize=7)
    _style_ax(ax8)

    fig.suptitle("Synthetic dataset overview — pre-training data inspection",
                 fontsize=13, fontweight="bold", color="#F9FAFB", y=0.99)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"💾 Saved: {save_path}")
    plt.tight_layout()
    plt.show()
    return fig


# ── helper also used by panel 1 ───────────────────────────────────────────────

def _session(cfg, ap, mics, drone_positions, noise_level, noise_profile,
             out_dir, session_id):
    import soundfile as sf
    sr  = cfg.SR
    n   = int(sr * cfg.TARGET_DURATION)
    mix = [np.zeros(n, dtype=np.float32) for _ in range(len(mics))]
    for src_xy, dtype in drone_positions:
        chs = _synth(mics, src_xy, noise_level=noise_level,
                     drone_type=dtype, noise_profile=noise_profile, cfg=cfg)
        for i, ch in enumerate(chs):
            mix[i] = np.clip(mix[i] + ap.pad_or_truncate(ch), -1.0, 1.0).astype(np.float32)
    peak = max(float(np.max(np.abs(c))) for c in mix) + 1e-8
    if peak > 0.9:
        mix = [(c / peak * 0.9).astype(np.float32) for c in mix]
    for i, ch in enumerate(mix):
        sf.write(str(out_dir / f"{session_id}_ch{i}.wav"), ch, sr)
    return mix


# ══════════════════════════════════════════════════════════════════════════════
# 2. Detailed single-session view
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_session(
    cfg=None,
    scenario: str = "indoor_hover",
    array_name: str = "uavirbase",
    seed: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Detailed 2×3 figure for one synthesised session.

    Panels
    ──────
    [0] Waveform — all 3 channels overlaid (zoom to first 80 ms)
    [1] Mel spectrogram (ch0, log-mel)
    [2] Cross-correlation (GCC-PHAT) for each mic pair
    [3] Magnitude spectrum + BPF harmonic markers
    [4] TDOA delay estimates (cross-correlation peak positions)
    [5] 2-D bird's-eye of array geometry + drone position
    """
    cfg = _cfg(cfg)
    cfg.set_array_geometry(array_name)
    ap  = _ap(cfg)
    plt, gs_mod = _import_mpl()
    import librosa

    chs, az, dist, ht, dtype, npf = _make_example(cfg, ap, scenario, seed)
    n_show = int(cfg.SR * 0.08)   # 80 ms
    t_ms   = np.arange(n_show) / cfg.SR * 1000
    padded = [ap.pad_or_truncate(c) for c in chs]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0F1117")
    _s = lambda ax: (
        ax.set_facecolor("#1A1D27"),
        [sp.set_color("#3A3D4D") for sp in ax.spines.values()],
        ax.tick_params(colors="#9CA3AF", labelsize=8),
        ax.xaxis.label.set_color("#D1D5DB"),
        ax.yaxis.label.set_color("#D1D5DB"),
        ax.title.set_color("#F9FAFB"),
    )

    # ── [0] Waveforms ──────────────────────────────────────────────────────
    ax = axes[0, 0]; _s(ax)
    for i, (ch, col) in enumerate(zip(padded, _MIC_COLOURS)):
        ax.plot(t_ms, ch[:n_show], color=col, lw=0.8, alpha=0.85, label=f"ch{i}")
    ax.set_xlabel("Time (ms)", fontsize=8); ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title(f"Waveforms (80 ms)  —  {scenario}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")
    ax.text(0.98, 0.96, f"az={az:.0f}°  d={dist:.1f}m  h={ht:.1f}m",
            transform=ax.transAxes, fontsize=7, color="#A5B4FC",
            ha="right", va="top")

    # ── [1] Mel spectrogram ────────────────────────────────────────────────
    ax = axes[0, 1]; _s(ax)
    mel = ap.mel(padded[0])
    im  = ax.imshow(mel, aspect="auto", origin="lower", cmap="inferno",
                    interpolation="nearest",
                    extent=[0, cfg.TARGET_DURATION, 0, cfg.SR // 2 / 1000])
    ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Freq (kHz)", fontsize=8)
    ax.set_title(f"Log-mel spectrogram (ch0)  —  {dtype.replace('_',' ')}",
                 fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9).ax.tick_params(labelsize=6)

    # ── [2] GCC-PHAT cross-correlations ────────────────────────────────────
    ax = axes[0, 2]; _s(ax)
    pairs = [(0, 1, "ch0–ch1"), (0, 2, "ch0–ch2"), (1, 2, "ch1–ch2")]
    n_cc  = 2048
    max_delay_smp = int(cfg.SR * 0.003)   # ±3 ms
    lags_ms = np.arange(-max_delay_smp, max_delay_smp + 1) / cfg.SR * 1000
    for (i, j, lbl), col in zip(pairs, _MIC_COLOURS):
        yi = padded[i][:n_cc].astype(np.float64)
        yj = padded[j][:n_cc].astype(np.float64)
        Ri = np.fft.rfft(yi, n=n_cc * 2)
        Rj = np.fft.rfft(yj, n=n_cc * 2)
        G  = Ri * np.conj(Rj)
        G /= (np.abs(G) + 1e-10)
        cc = np.fft.irfft(G)
        cc = np.roll(cc, max_delay_smp)[:2 * max_delay_smp + 1]
        ax.plot(lags_ms, cc / (np.max(np.abs(cc)) + 1e-8),
                color=col, lw=1.0, alpha=0.85, label=lbl)
    ax.axvline(0, color="#6B7280", lw=0.5, linestyle="--")
    ax.set_xlabel("Lag (ms)", fontsize=8); ax.set_ylabel("GCC-PHAT", fontsize=8)
    ax.set_title("Cross-correlation (GCC-PHAT)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

    # ── [3] Magnitude spectrum + BPF markers ───────────────────────────────
    ax = axes[1, 0]; _s(ax)
    from .config import DRONE_BPF_PROFILES
    freqs = librosa.fft_frequencies(sr=cfg.SR, n_fft=2048)
    S_db  = 20 * np.log10(
        np.abs(librosa.stft(padded[0].astype(np.float32),
                            n_fft=2048, hop_length=512)).mean(axis=1) + 1e-10
    )
    ax.plot(freqs, S_db, color="#60A5FA", lw=0.9, alpha=0.85)
    if dtype in DRONE_BPF_PROFILES:
        f_lo, f_mid, f_hi, n_h = DRONE_BPF_PROFILES[dtype]
        for k in range(1, n_h + 1):
            fk = f_mid * k
            if fk < cfg.SR // 2:
                ax.axvline(fk, color="#F59E0B", lw=0.8, linestyle="--", alpha=0.8,
                           label=f"BPF h{k}" if k == 1 else "")
        ax.axvspan(f_lo, f_hi, alpha=0.08, color="#F59E0B")
    ax.set_xlim(0, min(4000, cfg.SR // 2))
    ax.set_xlabel("Frequency (Hz)", fontsize=8); ax.set_ylabel("Power (dB)", fontsize=8)
    ax.set_title(f"Magnitude spectrum + BPF harmonics", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

    # ── [4] TDOA delay bar chart ───────────────────────────────────────────
    ax = axes[1, 1]; _s(ax)
    from .utils import compute_ipd_features
    ipd = compute_ipd_features(padded, cfg)   # (3,) TDOA in seconds
    ipd_ms = ipd * 1000
    pair_lbls = ["ch0–ch1", "ch0–ch2", "ch1–ch2"]
    bars = ax.bar(pair_lbls, ipd_ms, color=_MIC_COLOURS, alpha=0.75, width=0.5)
    ax.axhline(0, color="#6B7280", lw=0.7)
    for bar, val in zip(bars, ipd_ms):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.005 * np.sign(val + 1e-8),
                f"{val:.3f} ms", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, color="white")
    ax.set_ylabel("TDOA delay (ms)", fontsize=8)
    ax.set_title("IPD: TDOA delays (mic pairs)", fontsize=9, fontweight="bold")

    # ── [5] Bird's-eye array + drone position ─────────────────────────────
    ax = axes[1, 2]; _s(ax); ax.set_aspect("equal")
    mics = cfg.MIC_POSITIONS
    cx, cy = cfg.ARRAY_CENTER
    az_r  = math.radians(az)
    src_x = cx + dist * math.cos(az_r)
    src_y = cy + dist * math.sin(az_r)

    # Draw mics
    for mi, (mx, my) in enumerate(mics):
        ax.scatter(mx, my, s=120, color=_MIC_COLOURS[mi], zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"M{mi}", (mx, my), xytext=(0.08, 0.08),
                    textcoords="offset fontsize", fontsize=7, color="white")

    # Draw array centre
    ax.scatter(cx, cy, s=60, color="#6B7280", marker="+", zorder=4)

    # Draw drone
    ax.scatter(src_x, src_y, s=200, color="#F59E0B", marker="*",
               zorder=6, edgecolors="white", linewidths=0.5, label=f"Drone ({dtype.replace('_',' ')})")

    # Draw azimuth ray
    ax.annotate("", xy=(src_x, src_y), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#F59E0B",
                                lw=1.2, connectionstyle="arc3,rad=0"))

    # Label
    ax.text(src_x + 0.15, src_y, f"az={az:.0f}°\nd={dist:.1f}m\nh={ht:.1f}m",
            fontsize=7, color="#FCD34D")

    # Baseline indicator
    baseline = float(np.linalg.norm(mics[0] - mics[1]))
    ax.text(0.02, 0.02, f"Baseline: {baseline*100:.0f} cm",
            transform=ax.transAxes, fontsize=7, color="#9CA3AF")

    ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_title("Array geometry + drone position", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="white", loc="upper right")

    fig.suptitle(
        f"Session detail — scenario: {scenario}  |  "
        f"drone: {dtype.replace('_',' ')}  |  noise: {npf}",
        fontsize=11, fontweight="bold", color="#F9FAFB", y=1.01,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"💾 Saved: {save_path}")
    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Position coverage scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_position_coverage(
    cfg=None,
    n_samples: int = 400,
    array_names: Optional[List[str]] = None,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> None:
    """
    2×N grid of bird's-eye and polar scatter plots, one column per array.

    Shows how well the sampled positions cover the reachable space for
    each array geometry — useful for checking that the synthetic dataset
    has no large angular or distance gaps.
    """
    cfg = _cfg(cfg)
    if array_names is None:
        array_names = ["uavirbase", "gp1", "gp2"]
    plt, _ = _import_mpl()
    rng = np.random.default_rng(seed)

    n_arr = len(array_names)
    fig, axes = plt.subplots(2, n_arr, figsize=(6 * n_arr, 10))
    fig.patch.set_facecolor("#0F1117")
    if n_arr == 1:
        axes = axes[:, np.newaxis]

    _s = lambda ax: (
        ax.set_facecolor("#1A1D27"),
        [sp.set_color("#3A3D4D") for sp in ax.spines.values()],
        ax.tick_params(colors="#9CA3AF", labelsize=8),
        ax.xaxis.label.set_color("#D1D5DB"),
        ax.yaxis.label.set_color("#D1D5DB"),
        ax.title.set_color("#F9FAFB"),
    )

    for col, arr in enumerate(array_names):
        cfg.set_array_geometry(arr)
        mics = cfg.MIC_POSITIONS
        cx, cy = cfg.ARRAY_CENTER
        azs, dists, hts, dtypes_s = _sample_positions(cfg, rng, n=n_samples)

        # ── Row 0: Cartesian scatter ───────────────────────────────────────
        ax0 = axes[0, col]; _s(ax0); ax0.set_aspect("equal")
        xs = cx + dists * np.cos(np.radians(azs))
        ys = cy + dists * np.sin(np.radians(azs))

        sc0 = ax0.scatter(xs, ys, c=dists, cmap="plasma",
                          alpha=0.4, s=10, linewidths=0)
        for mi, (mx, my) in enumerate(mics):
            ax0.scatter(mx, my, s=120, color=_MIC_COLOURS[mi],
                        zorder=5, edgecolors="white", linewidths=0.8)
            ax0.annotate(f"M{mi}", (mx, my), xytext=(0.1, 0.1),
                         textcoords="offset fontsize", fontsize=7, color="white")
        ax0.scatter(cx, cy, s=60, color="#6B7280", marker="+", zorder=4)
        plt.colorbar(sc0, ax=ax0, pad=0.02, shrink=0.8, label="Distance (m)")
        baseline = float(np.linalg.norm(mics[0] - mics[1]))
        ax0.set_title(f"{arr.upper()}  (baseline {baseline*100:.0f} cm)\n"
                      f"Cartesian position coverage",
                      fontsize=9, fontweight="bold")
        ax0.set_xlabel("X (m)", fontsize=8); ax0.set_ylabel("Y (m)", fontsize=8)

        # ── Row 1: Polar scatter ───────────────────────────────────────────
        ax1 = fig.add_subplot(2, n_arr, n_arr + col + 1, polar=True)
        ax1.set_facecolor("#1A1D27")
        ax1.tick_params(colors="#9CA3AF", labelsize=7)
        sc1 = ax1.scatter(np.radians(azs), dists, c=hts, cmap="viridis",
                          alpha=0.45, s=10, linewidths=0)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        plt.colorbar(sc1, ax=ax1, pad=0.14, shrink=0.7, label="Height (m)")
        ax1.set_title(f"Polar coverage  (colour = height)",
                      fontsize=9, fontweight="bold", color="#F9FAFB", pad=14)

    fig.suptitle("Synthetic dataset position coverage by array geometry",
                 fontsize=12, fontweight="bold", color="#F9FAFB", y=1.01)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"💾 Saved: {save_path}")
    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Noise profile comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_noise_profiles(
    cfg=None,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side PSD comparison of the synthetic indoor and outdoor noise
    models versus the real measurement floor profiles from config.

    Left:  indoor — complex broadband + HVAC peaks at 627/1637/4363 Hz
    Right: outdoor — Brownian f⁻² slope
    """
    cfg = _cfg(cfg)
    ap  = _ap(cfg)
    plt, _ = _import_mpl()
    import librosa
    from .config import NOISE_FLOOR_INDOOR, NOISE_FLOOR_OUTDOOR

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    _s = lambda ax: (
        ax.set_facecolor("#1A1D27"),
        [sp.set_color("#3A3D4D") for sp in ax.spines.values()],
        ax.tick_params(colors="#9CA3AF", labelsize=8),
        ax.xaxis.label.set_color("#D1D5DB"),
        ax.yaxis.label.set_color("#D1D5DB"),
        ax.title.set_color("#F9FAFB"),
    )

    n_sig = int(cfg.SR * 6)
    freqs = librosa.fft_frequencies(sr=cfg.SR, n_fft=4096)

    # ── Indoor ───────────────────────────────────────────────────────────
    _s(ax_l)
    for amp, lbl, col in [(0.01, "low (0.01)", "#93C5FD"),
                           (0.03, "med (0.03)", "#60A5FA"),
                           (0.08, "high (0.08)", "#3B82F6")]:
        n_i = _make_noise(n_sig, cfg.SR, profile="indoor", amplitude=amp)
        S_i = np.abs(librosa.stft(n_i.astype(np.float32), n_fft=4096,
                                   hop_length=1024)).mean(axis=1)
        ax_l.plot(freqs, 20 * np.log10(S_i + 1e-12),
                  color=col, lw=0.9, alpha=0.85, label=f"synth {lbl}")
    # Mark real HVAC peaks
    for fk, prom in zip(NOISE_FLOOR_INDOOR["peaks_hz"],
                        NOISE_FLOOR_INDOOR["peak_prominences_db"]):
        if fk < cfg.SR // 2:
            ax_l.axvline(fk, color="#FCA5A5", lw=0.7, linestyle=":", alpha=0.7)
            ax_l.text(fk + 20, ax_l.get_ylim()[1] * 0.8 if ax_l.get_ylim()[1] != 0 else -10,
                      f"{fk}Hz", fontsize=6, color="#FCA5A5", rotation=90)
    ax_l.axhline(NOISE_FLOOR_INDOOR["median_db"], color="#FCA5A5",
                  lw=1.0, linestyle="--", alpha=0.7,
                  label=f"measured floor {NOISE_FLOOR_INDOOR['median_db']} dB")
    ax_l.set_xscale("log"); ax_l.set_xlim(20, cfg.SR // 2)
    ax_l.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_l.set_ylabel("Power (dB)", fontsize=9)
    ax_l.set_title("Indoor noise — synthetic vs measured\n(PannoniaFS pre-flight silence)",
                   fontsize=10, fontweight="bold")
    ax_l.legend(fontsize=7.5, framealpha=0.25, labelcolor="white")

    # ── Outdoor ───────────────────────────────────────────────────────────
    _s(ax_r)
    for amp, lbl, col in [(0.03, "low (0.03)", "#86EFAC"),
                           (0.08, "med (0.08)", "#4ADE80"),
                           (0.15, "high (0.15)", "#16A34A")]:
        n_o = _make_noise(n_sig, cfg.SR, profile="outdoor", amplitude=amp)
        S_o = np.abs(librosa.stft(n_o.astype(np.float32), n_fft=4096,
                                   hop_length=1024)).mean(axis=1)
        ax_r.plot(freqs, 20 * np.log10(S_o + 1e-12),
                  color=col, lw=0.9, alpha=0.85, label=f"synth {lbl}")
    # Overlay theoretical f^-2 Brownian slope
    f_ref = freqs[freqs > 20]
    P_brownian = -40 * np.log10(f_ref / 100) - 60   # rough reference
    ax_r.plot(f_ref, P_brownian, color="#FCD34D", lw=1.2, linestyle="--",
              alpha=0.7, label="Brownian f⁻²  (ref)")
    ax_r.axhline(NOISE_FLOOR_OUTDOOR["median_db"], color="#FCD34D",
                  lw=1.0, linestyle=":", alpha=0.7,
                  label=f"measured floor {NOISE_FLOOR_OUTDOOR['median_db']} dB")
    ax_r.set_xscale("log"); ax_r.set_xlim(20, cfg.SR // 2)
    ax_r.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_r.set_ylabel("Power (dB)", fontsize=9)
    ax_r.set_title("Outdoor noise — synthetic vs measured\n(Dunakeszi Brüel array)",
                   fontsize=10, fontweight="bold")
    ax_r.legend(fontsize=7.5, framealpha=0.25, labelcolor="white")

    fig.suptitle("Synthetic noise profiles — real measurement ground truth overlay",
                 fontsize=12, fontweight="bold", color="#F9FAFB", y=1.02)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"💾 Saved: {save_path}")
    plt.tight_layout()
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convenience: run all four plots in one call
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(
    cfg=None,
    save_dir: Optional[str] = None,
    n_samples: int = 300,
    seed: int = 42,
) -> None:
    """
    Run all four visualisations in sequence.  If save_dir is given, each
    figure is saved as a PNG inside that directory.

    Typical pre-training call:
        from drone_detection.dataset_visualizer import plot_all
        plot_all(config, save_dir="/content/drive/MyDrive/drone_v15/figures/")
    """
    cfg = _cfg(cfg)
    sd  = Path(save_dir) if save_dir else None

    plot_dataset_overview(
        cfg, n_samples=n_samples, seed=seed,
        save_path=str(sd / "dataset_overview.png") if sd else None)

    for sc in ["indoor_hover", "outdoor_fly", "low_snr", "multi_drone_2"]:
        plot_sample_session(
            cfg, scenario=sc, seed=seed,
            save_path=str(sd / f"session_{sc}.png") if sd else None)

    plot_position_coverage(
        cfg, n_samples=n_samples, seed=seed,
        save_path=str(sd / "position_coverage.png") if sd else None)

    plot_noise_profiles(
        cfg,
        save_path=str(sd / "noise_profiles.png") if sd else None)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Test-ZIP visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_test_zip_overview(
    zip_path: str,
    cfg=None,
    n_sample_sessions: int = 6,
    save_path: Optional[str] = None,
    seed: int = 0,
) -> None:
    """
    Comprehensive visual inspection of a generated test-dataset ZIP.

    Figures produced
    ─────────────────
    Figure 1 — Label statistics (4 panels)
      [0] Azimuth distribution — rose diagram (polar histogram)
      [1] Distance vs height — scatter coloured by drone type
      [2] Scenario / n_drones breakdown — stacked bar
      [3] BPF energy ratio distribution — violin per drone type

    Figure 2 — Sample sessions grid  (n_sample_sessions columns)
      Row 0: Mel spectrogram (ch0) for each sampled session
      Row 1: Magnitude spectrum + BPF harmonic markers
      Row 2: GCC-PHAT cross-correlation (ch0–ch1 pair)

    Figure 3 — Spatial coverage
      Left:  Cartesian scatter coloured by scenario
      Right: Polar scatter coloured by distance

    All figures are shown inline and optionally saved to save_path
    (base filename — suffixes _labels, _sessions, _spatial are added).

    Parameters
    ──────────
    zip_path          : path to the generated test ZIP
    cfg               : Config instance
    n_sample_sessions : number of random sessions to show in Fig 2
    save_path         : base path for saving; e.g. "/content/test_inspect"
                        → /content/test_inspect_labels.png etc.
    seed              : random seed for session sampling

    Example
    ───────
    from drone_detection.dataset_visualizer import plot_test_zip_overview

    # After generating:
    generate_single_drone_dataset(config, "/content/test.zip", n_sessions=50)

    # Inspect before evaluating:
    plot_test_zip_overview("/content/test.zip", config)
    """
    import csv as _csv
    import io
    import json
    import shutil
    import tempfile
    import zipfile as _zf

    cfg = _cfg(cfg)
    ap  = _ap(cfg)
    plt, gs_mod = _import_mpl()
    import librosa
    import soundfile as sf

    rng = np.random.default_rng(seed)

    # ── 1. Read labels.csv and sample session WAVs from the ZIP ──────────
    if not _zf.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP: {zip_path}")

    rows: List[Dict] = []
    wav_names: List[str] = []
    with _zf.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        wav_names = sorted([n for n in all_names if n.endswith(".wav")])
        if "labels.csv" in all_names:
            with zf.open("labels.csv") as f:
                rows = list(_csv.DictReader(io.TextIOWrapper(f)))

    if not rows:
        print("⚠️  No labels.csv found — label statistics plots skipped.")
    if not wav_names:
        raise ValueError("No WAV files found in the ZIP.")

    n_sessions = len(wav_names) // 3
    print(f"📦 {Path(zip_path).name}: {n_sessions} sessions, {len(rows)} labelled")

    # ── Extract a temp copy for audio loading ─────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="viz_zip_"))
    try:
        with _zf.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        # ── FIGURE 1: Label statistics ─────────────────────────────────────
        if rows:
            fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
            fig1.patch.set_facecolor("#0F1117")

            _s = lambda ax: (
                ax.set_facecolor("#1A1D27"),
                [sp.set_color("#3A3D4D") for sp in ax.spines.values()],
                ax.tick_params(colors="#9CA3AF", labelsize=8),
                ax.xaxis.label.set_color("#D1D5DB"),
                ax.yaxis.label.set_color("#D1D5DB"),
                ax.title.set_color("#F9FAFB"),
            )

            azs   = [float(r["azimuth_deg"]) for r in rows if "azimuth_deg" in r]
            dists = [float(r["distance_m"])  for r in rows if "distance_m"  in r]
            hts   = [float(r["height_m"])    for r in rows if "height_m"    in r]

            # [0] Azimuth rose diagram
            ax = fig1.axes[0] if False else fig1.add_subplot(2, 2, 1, polar=True)
            ax.set_facecolor("#1A1D27")
            ax.tick_params(colors="#9CA3AF", labelsize=7)
            if azs:
                az_rad = np.radians(azs)
                counts, edges = np.histogram(az_rad, bins=36,
                                             range=(-np.pi, np.pi))
                centres = 0.5 * (edges[:-1] + edges[1:])
                bars_r  = ax.bar(centres, counts,
                                 width=edges[1] - edges[0],
                                 color="#60A5FA", alpha=0.75,
                                 edgecolor="#1A1D27", linewidth=0.4)
                # Colour by count
                max_c = max(counts) + 1
                for bar, cnt in zip(bars_r, counts):
                    bar.set_facecolor(plt.cm.plasma(cnt / max_c))
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title("Azimuth distribution", fontsize=9,
                         fontweight="bold", color="#F9FAFB", pad=14)

            # Re-reference axes for flat panels
            ax0 = fig1.add_subplot(2, 2, 2)

            # [1] Distance vs height scatter
            _s(ax0)
            drone_types_u = sorted(set(
                r.get("drone_type", r.get("drone_types", "?")).split("|")[0]
                for r in rows
            ))
            cmap_d = plt.cm.get_cmap("tab10", len(drone_types_u))
            dt_idx = {dt: i for i, dt in enumerate(drone_types_u)}
            for r, d, h in zip(rows, dists, hts):
                dt_key = r.get("drone_type",
                               r.get("drone_types", "?")).split("|")[0]
                col = cmap_d(dt_idx.get(dt_key, 0))
                ax0.scatter(d, h, color=col, alpha=0.5, s=18, linewidths=0)
            handles = [
                plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=cmap_d(dt_idx[dt]),
                           markersize=6, label=dt.replace("_"," "))
                for dt in drone_types_u
            ]
            ax0.legend(handles=handles, fontsize=7, framealpha=0.2,
                       labelcolor="white", loc="upper right")
            ax0.set_xlabel("Distance (m)", fontsize=8)
            ax0.set_ylabel("Height (m)", fontsize=8)
            ax0.set_title("Distance vs height (colour = drone type)",
                          fontsize=9, fontweight="bold")

            # [2] Scenario / n_drones breakdown
            ax2 = fig1.add_subplot(2, 2, 3); _s(ax2)
            from collections import Counter
            if "scenario" in rows[0]:
                sc_counts = Counter(r["scenario"] for r in rows)
                scs = sorted(sc_counts, key=lambda k: -sc_counts[k])
                cmap_sc = plt.cm.get_cmap("tab20", len(scs))
                # Split by n_drones if present
                if "n_drones" in rows[0]:
                    nd_vals  = sorted(set(int(r["n_drones"]) for r in rows))
                    bottom   = np.zeros(len(scs))
                    for nd in nd_vals:
                        heights_b = [sum(1 for r in rows
                                         if r["scenario"] == sc
                                         and int(r["n_drones"]) == nd)
                                     for sc in scs]
                        ax2.bar(range(len(scs)), heights_b, bottom=bottom,
                                label=f"{nd} drone{'s' if nd>1 else ''}",
                                alpha=0.8, edgecolor="#1A1D27", linewidth=0.3)
                        bottom += np.array(heights_b, dtype=float)
                    ax2.legend(fontsize=7, framealpha=0.2, labelcolor="white")
                else:
                    ax2.bar(range(len(scs)),
                            [sc_counts[s] for s in scs],
                            color=[cmap_sc(i) for i in range(len(scs))],
                            alpha=0.8, edgecolor="#1A1D27", linewidth=0.3)
                ax2.set_xticks(range(len(scs)))
                ax2.set_xticklabels(
                    [s.replace("_","\n") for s in scs],
                    fontsize=6.5)
            else:
                nd_counts = Counter(int(r.get("n_drones", 1)) for r in rows)
                ax2.bar([str(k) for k in sorted(nd_counts)],
                        [nd_counts[k] for k in sorted(nd_counts)],
                        color="#818CF8", alpha=0.75)
                ax2.set_xlabel("Number of drones", fontsize=8)
            ax2.set_ylabel("Session count", fontsize=8)
            ax2.set_title("Scenario / drone-count breakdown",
                          fontsize=9, fontweight="bold")

            # [3] BPF energy ratio violin / strip
            ax3 = fig1.add_subplot(2, 2, 4); _s(ax3)
            bpf_by_type: Dict[str, List[float]] = {}
            for r in rows:
                dt_key = r.get("drone_type",
                               r.get("drone_types", "?")).split("|")[0]
                bpf_val = r.get("bpf_ratio", r.get("bpf_energy_ratio"))
                if bpf_val not in (None, "", "nan"):
                    bpf_by_type.setdefault(dt_key, []).append(float(bpf_val))
            if bpf_by_type:
                labels_v = list(bpf_by_type.keys())
                data_v   = [bpf_by_type[l] for l in labels_v]
                vp = ax3.violinplot(data_v, positions=range(len(labels_v)),
                                    showmedians=True, showextrema=True)
                for i, (body, lbl) in enumerate(zip(vp["bodies"], labels_v)):
                    col = _DRONE_COLOURS.get(lbl, "#9CA3AF")
                    body.set_facecolor(col); body.set_alpha(0.55)
                vp["cmedians"].set_color("white")
                vp["cbars"].set_color("#6B7280")
                vp["cmins"].set_color("#6B7280")
                vp["cmaxes"].set_color("#6B7280")
                ax3.set_xticks(range(len(labels_v)))
                ax3.set_xticklabels(
                    [l.replace("_","\n") for l in labels_v], fontsize=7)
                ax3.set_ylabel("BPF energy ratio", fontsize=8)
            else:
                # Fall back: noise level distribution
                nls = [float(r["noise_level"])
                       for r in rows if "noise_level" in r]
                if nls:
                    ax3.hist(nls, bins=20, color="#A78BFA", alpha=0.75,
                             edgecolor="#1A1D27", linewidth=0.3)
                    ax3.set_xlabel("Noise level", fontsize=8)
                    ax3.set_ylabel("Count", fontsize=8)
            ax3.set_title("BPF energy ratio or noise level",
                          fontsize=9, fontweight="bold")

            fig1.suptitle(
                f"Test ZIP — label statistics: {Path(zip_path).name}  "
                f"({n_sessions} sessions, {len(rows)} labelled)",
                fontsize=11, fontweight="bold", color="#F9FAFB", y=1.0)

            if save_path:
                p = str(save_path) + "_labels.png"
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                fig1.savefig(p, dpi=150, bbox_inches="tight",
                             facecolor=fig1.get_facecolor())
                print(f"💾 Saved: {p}")
            plt.tight_layout()
            plt.show()

        # ── FIGURE 2: Sample session audio panels ─────────────────────────
        # Find session stems that have all 3 channels
        ch0_files = sorted([p for p in tmp.rglob("*_ch0.wav")])
        if not ch0_files:
            # flat naming: session_0000_ch0.wav etc
            ch0_files = sorted(tmp.glob("*_ch0.wav"))
        if not ch0_files:
            print("⚠️  Could not locate *_ch0.wav files for audio panels.")
        else:
            n_show = min(n_sample_sessions, len(ch0_files))
            indices = rng.choice(len(ch0_files), n_show, replace=False)
            sampled = [ch0_files[i] for i in sorted(indices)]

            fig2, axes2 = plt.subplots(3, n_show,
                                       figsize=(max(4*n_show, 12), 10))
            fig2.patch.set_facecolor("#0F1117")
            if n_show == 1:
                axes2 = axes2[:, np.newaxis]

            for col_i, ch0_path in enumerate(sampled):
                stem = ch0_path.name.replace("_ch0.wav", "")
                ch0  = ch0_path
                ch1  = ch0_path.parent / f"{stem}_ch1.wav"
                ch2  = ch0_path.parent / f"{stem}_ch2.wav"

                # Load channel 0
                y0, sr_file = sf.read(str(ch0), dtype="float32")
                if y0.ndim > 1:
                    y0 = y0[:, 0]
                y0 = ap.pad_or_truncate(y0)

                # Load label if present
                lbl_path = ch0_path.parent / f"{stem}_label.json"
                lbl_text = ""
                if lbl_path.exists():
                    try:
                        d = json.loads(lbl_path.read_text())
                        lbl_text = (f"az={d.get('azimuth_deg',0):.0f}°  "
                                    f"d={d.get('distance_m',0):.1f}m  "
                                    f"n={d.get('n_drones',1)}")
                    except Exception:
                        pass
                # Also check from rows dict
                if not lbl_text and rows:
                    match = next((r for r in rows if r.get("session_id","") == stem), None)
                    if match:
                        lbl_text = (f"az={match.get('azimuth_deg','?')}°  "
                                    f"d={match.get('distance_m','?')}m  "
                                    f"n={match.get('n_drones','?')}")

                # ── Row 0: Mel spectrogram ─────────────────────────────────
                ax_mel = axes2[0, col_i]
                ax_mel.set_facecolor("#1A1D27")
                [sp.set_color("#3A3D4D") for sp in ax_mel.spines.values()]
                ax_mel.tick_params(colors="#9CA3AF", labelsize=6)
                mel = ap.mel(y0)
                ax_mel.imshow(mel, aspect="auto", origin="lower",
                              cmap="inferno", interpolation="nearest",
                              extent=[0, cfg.TARGET_DURATION,
                                      0, cfg.SR // 2 / 1000])
                ax_mel.set_title(f"{stem[:20]}", fontsize=7, color="#F9FAFB",
                                 fontweight="bold")
                if lbl_text:
                    ax_mel.text(0.02, 0.97, lbl_text, transform=ax_mel.transAxes,
                                fontsize=5.5, color="#A5B4FC",
                                verticalalignment="top")
                if col_i == 0:
                    ax_mel.set_ylabel("Freq (kHz)", fontsize=7,
                                      color="#D1D5DB")
                ax_mel.set_xlabel("Time (s)", fontsize=6, color="#D1D5DB")

                # ── Row 1: Magnitude spectrum + BPF markers ────────────────
                ax_sp = axes2[1, col_i]
                ax_sp.set_facecolor("#1A1D27")
                [sp.set_color("#3A3D4D") for sp in ax_sp.spines.values()]
                ax_sp.tick_params(colors="#9CA3AF", labelsize=6)
                freqs_sp = librosa.fft_frequencies(sr=cfg.SR, n_fft=2048)
                S_sp = np.abs(librosa.stft(y0, n_fft=2048,
                                            hop_length=512)).mean(axis=1)
                ax_sp.plot(freqs_sp,
                           20*np.log10(S_sp + 1e-12),
                           color="#60A5FA", lw=0.7, alpha=0.85)
                # Detect dominant BPF from label
                if rows:
                    match = next((r for r in rows
                                  if r.get("session_id","") == stem), None)
                    if match:
                        dt_key = match.get("drone_type",
                                  match.get("drone_types","")).split("|")[0]
                        from .config import DRONE_BPF_PROFILES
                        if dt_key in DRONE_BPF_PROFILES:
                            f_lo, f_mid, f_hi, n_h = DRONE_BPF_PROFILES[dt_key]
                            for k in range(1, n_h + 1):
                                fk = f_mid * k
                                if fk < cfg.SR // 2:
                                    ax_sp.axvline(fk, color="#F59E0B",
                                                  lw=0.6, linestyle=":",
                                                  alpha=0.7)
                ax_sp.set_xlim(0, min(3000, cfg.SR // 2))
                ax_sp.set_ylim(bottom=None)
                if col_i == 0:
                    ax_sp.set_ylabel("Power (dB)", fontsize=7, color="#D1D5DB")
                ax_sp.set_xlabel("Hz", fontsize=6, color="#D1D5DB")

                # ── Row 2: GCC-PHAT ch0–ch1 ───────────────────────────────
                ax_gcc = axes2[2, col_i]
                ax_gcc.set_facecolor("#1A1D27")
                [sp.set_color("#3A3D4D") for sp in ax_gcc.spines.values()]
                ax_gcc.tick_params(colors="#9CA3AF", labelsize=6)

                if ch1.exists():
                    y1, _ = sf.read(str(ch1), dtype="float32")
                    if y1.ndim > 1:
                        y1 = y1[:, 0]
                    y1 = ap.pad_or_truncate(y1)
                    n_cc = 2048
                    max_dl = int(cfg.SR * 0.003)
                    lags_ms = np.arange(-max_dl, max_dl+1) / cfg.SR * 1000
                    Ri = np.fft.rfft(y0[:n_cc].astype(np.float64), n=n_cc*2)
                    Rj = np.fft.rfft(y1[:n_cc].astype(np.float64), n=n_cc*2)
                    G  = Ri * np.conj(Rj)
                    G /= (np.abs(G) + 1e-10)
                    cc = np.fft.irfft(G)
                    cc = np.roll(cc, max_dl)[:2*max_dl+1]
                    cc_n = cc / (np.max(np.abs(cc)) + 1e-8)
                    ax_gcc.plot(lags_ms, cc_n, color="#E53935", lw=0.8)
                    # Mark peak
                    pk = np.argmax(np.abs(cc_n))
                    ax_gcc.axvline(lags_ms[pk], color="#FCD34D",
                                   lw=0.8, linestyle="--", alpha=0.85)
                    ax_gcc.text(lags_ms[pk], 0.92,
                                f"{lags_ms[pk]:.2f}ms",
                                fontsize=5.5, color="#FCD34D",
                                ha="center", transform=ax_gcc.get_xaxis_transform())
                    ax_gcc.axvline(0, color="#6B7280", lw=0.4, linestyle="--")
                else:
                    ax_gcc.text(0.5, 0.5, "ch1 missing",
                                ha="center", va="center",
                                color="#9CA3AF", transform=ax_gcc.transAxes,
                                fontsize=7)
                if col_i == 0:
                    ax_gcc.set_ylabel("GCC-PHAT", fontsize=7, color="#D1D5DB")
                ax_gcc.set_xlabel("Lag (ms)", fontsize=6, color="#D1D5DB")

            # Row labels
            for row_i, lbl in enumerate(["Mel spectrogram (ch0)",
                                          "Spectrum + BPF (ch0)",
                                          "GCC-PHAT ch0–ch1"]):
                axes2[row_i, 0].set_ylabel(
                    f"{lbl}\n{axes2[row_i,0].get_ylabel()}",
                    fontsize=7, color="#D1D5DB")

            fig2.suptitle(
                f"Test ZIP — {n_show} sample sessions: {Path(zip_path).name}",
                fontsize=11, fontweight="bold", color="#F9FAFB", y=1.0)

            if save_path:
                p = str(save_path) + "_sessions.png"
                fig2.savefig(p, dpi=150, bbox_inches="tight",
                             facecolor=fig2.get_facecolor())
                print(f"💾 Saved: {p}")
            plt.tight_layout()
            plt.show()

        # ── FIGURE 3: Spatial coverage ────────────────────────────────────
        if rows and azs:
            fig3, (ax_cart, ax_pol_wrap) = plt.subplots(1, 2, figsize=(13, 6))
            fig3.patch.set_facecolor("#0F1117")

            # Cartesian scatter coloured by scenario
            ax_cart.set_facecolor("#1A1D27")
            [sp.set_color("#3A3D4D") for sp in ax_cart.spines.values()]
            ax_cart.tick_params(colors="#9CA3AF", labelsize=8)
            ax_cart.xaxis.label.set_color("#D1D5DB")
            ax_cart.yaxis.label.set_color("#D1D5DB")
            ax_cart.title.set_color("#F9FAFB")

            cfg.set_array_geometry(
                rows[0].get("array", "uavirbase") if rows else "uavirbase"
            )
            cx, cy = cfg.ARRAY_CENTER
            mics   = cfg.MIC_POSITIONS

            # Unique scenarios
            scenarios_u = sorted(set(r.get("scenario","default") for r in rows))
            cmap_sc     = plt.cm.get_cmap("tab10", max(len(scenarios_u), 1))
            sc_idx      = {s: i for i, s in enumerate(scenarios_u)}

            for r, d, az_v in zip(rows, dists, azs):
                az_r2 = math.radians(az_v)
                x = cx + d * math.cos(az_r2)
                y = cy + d * math.sin(az_r2)
                col = cmap_sc(sc_idx.get(r.get("scenario","default"), 0))
                n_d = int(r.get("n_drones", 1))
                mk  = "o" if n_d == 1 else ("s" if n_d == 2 else "^")
                ax_cart.scatter(x, y, color=col, alpha=0.45, s=18,
                                marker=mk, linewidths=0)

            # Draw mics
            for mi, (mx, my) in enumerate(mics):
                ax_cart.scatter(mx, my, s=150, color=_MIC_COLOURS[mi],
                                zorder=5, edgecolors="white", linewidths=0.8)
                ax_cart.annotate(f"M{mi}", (mx, my),
                                 xytext=(0.1, 0.1),
                                 textcoords="offset fontsize",
                                 fontsize=7, color="white")

            # Legend for scenarios
            sc_handles = [
                plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=cmap_sc(sc_idx[s]),
                           markersize=6, alpha=0.8,
                           label=s.replace("_"," "))
                for s in scenarios_u
            ]
            nd_handles = [
                plt.Line2D([0],[0], marker="o", color="gray",
                           markersize=6, label="1 drone"),
                plt.Line2D([0],[0], marker="s", color="gray",
                           markersize=6, label="2 drones"),
                plt.Line2D([0],[0], marker="^", color="gray",
                           markersize=6, label="3 drones"),
            ]
            ax_cart.legend(handles=sc_handles + nd_handles,
                           fontsize=6.5, framealpha=0.2,
                           labelcolor="white", ncol=2, loc="upper right")
            ax_cart.set_aspect("equal")
            ax_cart.set_xlabel("X (m)", fontsize=8)
            ax_cart.set_ylabel("Y (m)", fontsize=8)
            ax_cart.set_title("Spatial coverage — Cartesian\n(colour=scenario, shape=n_drones)",
                              fontsize=9, fontweight="bold")

            # Polar scatter coloured by distance
            ax_pol = fig3.add_subplot(1, 2, 2, polar=True)
            ax_pol.set_facecolor("#1A1D27")
            ax_pol.tick_params(colors="#9CA3AF", labelsize=7)
            sc_pol = ax_pol.scatter(
                np.radians(azs), dists,
                c=dists, cmap="plasma", alpha=0.5, s=12, linewidths=0)
            ax_pol.set_theta_zero_location("N")
            ax_pol.set_theta_direction(-1)
            cb_pol = plt.colorbar(sc_pol, ax=ax_pol, pad=0.13, shrink=0.75)
            cb_pol.set_label("Distance (m)", color="#D1D5DB", fontsize=7)
            cb_pol.ax.yaxis.set_tick_params(labelsize=6)
            ax_pol.set_title("Spatial coverage — polar\n(colour=distance)",
                             fontsize=9, fontweight="bold",
                             color="#F9FAFB", pad=14)

            fig3.suptitle(
                f"Test ZIP — spatial coverage: {Path(zip_path).name}",
                fontsize=11, fontweight="bold", color="#F9FAFB", y=1.0)

            if save_path:
                p = str(save_path) + "_spatial.png"
                fig3.savefig(p, dpi=150, bbox_inches="tight",
                             facecolor=fig3.get_facecolor())
                print(f"💾 Saved: {p}")
            plt.tight_layout()
            plt.show()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Quick single-session waveform inspector (from a ZIP session_id)
# ══════════════════════════════════════════════════════════════════════════════

def plot_test_session(
    zip_path: str,
    session_id: str,
    cfg=None,
    save_path: Optional[str] = None,
) -> None:
    """
    Load one specific session from a test ZIP and show a detailed
    6-panel view (identical layout to plot_sample_session).

    Useful for inspecting a session that gave a surprising result after
    running run_test_dataset_evaluation().

    Parameters
    ──────────
    zip_path   : path to the test ZIP
    session_id : the session ID shown in the evaluation table (e.g. "single_0003")
    cfg        : Config instance

    Example
    ───────
    # After evaluation revealed a bad prediction:
    plot_test_session("/content/test_single.zip", "single_0047", config)
    """
    import json
    import shutil
    import soundfile as sf
    import tempfile
    import zipfile as _zf

    cfg = _cfg(cfg)
    ap  = _ap(cfg)
    plt, gs_mod = _import_mpl()
    import librosa

    tmp = Path(tempfile.mkdtemp(prefix="viz_sess_"))
    try:
        with _zf.ZipFile(zip_path, "r") as zf:
            matching = [n for n in zf.namelist()
                        if session_id in n and
                        (n.endswith(".wav") or n.endswith(".json"))]
            if not matching:
                raise ValueError(
                    f"Session '{session_id}' not found in {zip_path}.\n"
                    f"First entries: {zf.namelist()[:8]}"
                )
            for m in matching:
                zf.extract(m, tmp)

        # Locate channel files
        ch_files = {i: tmp / f"{session_id}_ch{i}.wav" for i in range(3)}
        # Handle sub-folder layout
        if not ch_files[0].exists():
            found = list(tmp.rglob(f"{session_id}_ch0.wav"))
            if found:
                d = found[0].parent
                ch_files = {i: d / f"{session_id}_ch{i}.wav" for i in range(3)}
        if not ch_files[0].exists():
            raise FileNotFoundError(
                f"ch0 WAV not found for session '{session_id}' after extraction."
            )

        padded = []
        for i in range(3):
            if ch_files[i].exists():
                y, _ = sf.read(str(ch_files[i]), dtype="float32")
                if y.ndim > 1:
                    y = y[:, 0]
                padded.append(ap.pad_or_truncate(y))
            else:
                padded.append(np.zeros(int(cfg.SR * cfg.TARGET_DURATION),
                                       dtype=np.float32))

        # Load label
        lbl_path = tmp / f"{session_id}_label.json"
        if not lbl_path.exists():
            found_lbl = list(tmp.rglob(f"{session_id}_label.json"))
            if found_lbl:
                lbl_path = found_lbl[0]
        label = {}
        if lbl_path.exists():
            label = json.loads(lbl_path.read_text())

        az    = float(label.get("azimuth_deg", 0))
        dist  = float(label.get("distance_m",  0))
        ht    = float(label.get("height_m",    0))
        dtype = label.get("drone_types", label.get("drone_type", "unknown"))
        if isinstance(dtype, list):
            dtype = "|".join(dtype)
        npf   = label.get("noise_profile", "?")
        n_d   = int(label.get("n_drones", 1))

        # ── Delegate to the shared 6-panel layout ────────────────────────
        # Re-implement inline so we don't need a real scenario spec
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor("#0F1117")

        _s = lambda ax: (
            ax.set_facecolor("#1A1D27"),
            [sp.set_color("#3A3D4D") for sp in ax.spines.values()],
            ax.tick_params(colors="#9CA3AF", labelsize=8),
            ax.xaxis.label.set_color("#D1D5DB"),
            ax.yaxis.label.set_color("#D1D5DB"),
            ax.title.set_color("#F9FAFB"),
        )

        n_show = int(cfg.SR * 0.08)
        t_ms   = np.arange(n_show) / cfg.SR * 1000

        # [0] Waveforms
        ax = axes[0, 0]; _s(ax)
        for i, (ch, col) in enumerate(zip(padded, _MIC_COLOURS)):
            ax.plot(t_ms, ch[:n_show], color=col, lw=0.8, alpha=0.85,
                    label=f"ch{i}")
        ax.set_xlabel("Time (ms)", fontsize=8); ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title(f"Waveforms (80 ms) — {session_id}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")
        ax.text(0.98, 0.96,
                f"az={az:.0f}°  d={dist:.1f}m  h={ht:.1f}m  n_drones={n_d}",
                transform=ax.transAxes, fontsize=7, color="#A5B4FC",
                ha="right", va="top")

        # [1] Mel spectrogram
        ax = axes[0, 1]; _s(ax)
        mel = ap.mel(padded[0])
        im  = ax.imshow(mel, aspect="auto", origin="lower", cmap="inferno",
                        interpolation="nearest",
                        extent=[0, cfg.TARGET_DURATION,
                                0, cfg.SR // 2 / 1000])
        ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Freq (kHz)", fontsize=8)
        ax.set_title(f"Log-mel (ch0)  —  {dtype.replace('_',' ')}",
                     fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9).ax.tick_params(labelsize=6)

        # [2] GCC-PHAT all pairs
        ax = axes[0, 2]; _s(ax)
        pairs = [(0,1,"ch0–ch1"), (0,2,"ch0–ch2"), (1,2,"ch1–ch2")]
        n_cc  = 2048; max_dl = int(cfg.SR * 0.003)
        lags_ms = np.arange(-max_dl, max_dl+1) / cfg.SR * 1000
        for (i, j, lbl), col in zip(pairs, _MIC_COLOURS):
            yi = padded[i][:n_cc].astype(np.float64)
            yj = padded[j][:n_cc].astype(np.float64)
            Ri = np.fft.rfft(yi, n=n_cc*2); Rj = np.fft.rfft(yj, n=n_cc*2)
            G  = Ri * np.conj(Rj); G /= (np.abs(G) + 1e-10)
            cc = np.fft.irfft(G)
            cc = np.roll(cc, max_dl)[:2*max_dl+1]
            ax.plot(lags_ms, cc/(np.max(np.abs(cc))+1e-8),
                    color=col, lw=1.0, alpha=0.85, label=lbl)
        ax.axvline(0, color="#6B7280", lw=0.5, linestyle="--")
        ax.set_xlabel("Lag (ms)", fontsize=8); ax.set_ylabel("GCC-PHAT", fontsize=8)
        ax.set_title("GCC-PHAT (all mic pairs)", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

        # [3] Magnitude spectrum + BPF
        ax = axes[1, 0]; _s(ax)
        freqs_sp = librosa.fft_frequencies(sr=cfg.SR, n_fft=2048)
        S_sp = np.abs(librosa.stft(padded[0], n_fft=2048,
                                    hop_length=512)).mean(axis=1)
        ax.plot(freqs_sp, 20*np.log10(S_sp+1e-12),
                color="#60A5FA", lw=0.9, alpha=0.85)
        from .config import DRONE_BPF_PROFILES
        dt_key = dtype.split("|")[0]
        if dt_key in DRONE_BPF_PROFILES:
            f_lo, f_mid, f_hi, n_h = DRONE_BPF_PROFILES[dt_key]
            for k in range(1, n_h+1):
                fk = f_mid * k
                if fk < cfg.SR // 2:
                    ax.axvline(fk, color="#F59E0B", lw=0.8,
                               linestyle="--", alpha=0.8,
                               label=f"BPF h{k}" if k==1 else "")
            ax.axvspan(f_lo, f_hi, alpha=0.08, color="#F59E0B")
        ax.set_xlim(0, min(4000, cfg.SR//2))
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Power (dB)", fontsize=8)
        ax.set_title("Magnitude spectrum + BPF harmonics",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

        # [4] TDOA bar chart
        ax = axes[1, 1]; _s(ax)
        from .utils import compute_ipd_features
        ipd = compute_ipd_features(padded, cfg) * 1000   # → ms
        pair_lbls = ["ch0–ch1", "ch0–ch2", "ch1–ch2"]
        brs = ax.bar(pair_lbls, ipd, color=_MIC_COLOURS, alpha=0.75, width=0.5)
        ax.axhline(0, color="#6B7280", lw=0.7)
        for bar, val in zip(brs, ipd):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.005*np.sign(val+1e-8),
                    f"{val:.3f} ms", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, color="white")
        ax.set_ylabel("TDOA delay (ms)", fontsize=8)
        ax.set_title("IPD: TDOA delays", fontsize=9, fontweight="bold")

        # [5] Bird's-eye
        ax = axes[1, 2]; _s(ax); ax.set_aspect("equal")
        cfg.set_array_geometry(label.get("array", "uavirbase"))
        mics   = cfg.MIC_POSITIONS
        cx, cy = cfg.ARRAY_CENTER
        az_r   = math.radians(az)
        src_x  = cx + dist * math.cos(az_r)
        src_y  = cy + dist * math.sin(az_r)
        for mi, (mx, my) in enumerate(mics):
            ax.scatter(mx, my, s=120, color=_MIC_COLOURS[mi],
                       zorder=5, edgecolors="white", linewidths=0.8)
            ax.annotate(f"M{mi}", (mx, my),
                        xytext=(0.08, 0.08), textcoords="offset fontsize",
                        fontsize=7, color="white")
        ax.scatter(src_x, src_y, s=200, color="#F59E0B", marker="*",
                   zorder=6, edgecolors="white", linewidths=0.5,
                   label=f"{dtype.replace('_',' ')}")
        ax.annotate("", xy=(src_x, src_y), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.2))
        if n_d > 1:
            # Plot other drones if present in label
            for d_info in label.get("drones", [])[1:]:
                az2 = math.radians(float(d_info.get("azimuth_deg", 0)))
                d2  = float(d_info.get("distance_m", 1))
                x2  = cx + d2 * math.cos(az2)
                y2  = cy + d2 * math.sin(az2)
                ax.scatter(x2, y2, s=150, color="#A78BFA", marker="*",
                           zorder=6, edgecolors="white", linewidths=0.5)
        ax.text(src_x + 0.15, src_y,
                f"az={az:.0f}°\nd={dist:.1f}m\nh={ht:.1f}m",
                fontsize=7, color="#FCD34D")
        ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_title("Array geometry + drone position",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.2, labelcolor="white")

        fig.suptitle(
            f"Session detail — {session_id}  |  "
            f"drone: {dtype.replace('_',' ')}  |  noise: {npf}  |  "
            f"n_drones: {n_d}",
            fontsize=11, fontweight="bold", color="#F9FAFB", y=1.01,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"💾 Saved: {save_path}")
        plt.tight_layout()
        plt.show()
        return fig

    finally:
        shutil.rmtree(tmp, ignore_errors=True)