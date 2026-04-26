# -*- coding: utf-8 -*-
"""
multichannel_inference.py
─────────────────────────
Adapter for the Dunakeszi 12-channel recording rig:

  Channels  0–4  : DPA stand A  (5 professional capsules, linear rail ~20 cm)
  Channels  5–9  : DPA stand B  (5 professional capsules, linear rail ~20 cm)
  Channels 10–11 : XMOS MEMS   (2 low-cost capsules, circular PCB, USB)

Clock domains
─────────────
  DPA A + DPA B  — synchronised on the same recorder clock.
  XMOS MEMS      — independent USB clock.  NOT synchronised with DPA.

  ⚠ TDOA across the clock boundary (any DPA ch vs any MEMS ch) is INVALID.
  This module enforces that boundary automatically.

What this file provides
───────────────────────
  analyse_multichannel_file(wav_path, cfg, ...)
      Run the full pipeline on a single 12-channel (or N-channel) WAV:
        1. Validate + split channels by clock domain.
        2. Detect on the best available DPA channel (highest SNR).
        3. If detected → run localize_multi_drone() on the chosen DPA triplet.
        4. Run mems_inference.analyse_mems_file() on each MEMS channel.
        5. Return a combined result dict.

  build_dpa_triplet(channels_array, strategy='best_snr')
      Select 3 channels from the 10 DPA channels to form a non-collinear
      microphone triplet suitable for 2D TDOA localization.

  STRATEGIES
  ──────────
  'best_snr'       — pick the 3 channels with highest per-channel SNR
                     (best for noisy outdoor conditions)
  'max_baseline'   — pick ch 0, ch 4, ch 9 (stand A outer + stand B centre)
                     maximises TDOA sensitivity at longer ranges
  'equilateral'    — pick ch 2 (A centre), ch 5 (B first), ch 7 (B centre)
                     closest to equilateral geometry given typical stand spacing
  'custom'         — caller provides channel indices directly

Usage
─────
    from multichannel_inference import analyse_multichannel_file
    from drone_detection import config

    result = analyse_multichannel_file(
        "dunakeszi_12ch.wav",
        config,
        dpa_strategy="max_baseline",
        show_plots=True,
    )
    print(result["dpa"]["detected"], result["dpa"]["drones"])
    print(result["mems"][0]["distance_m"])

    python multichannel_inference.py 251020VITEMOROM1AT01U.wav
    # → Strategy auto-selected: stand_a

    python multichannel_inference.py 251020VITEMOROM1AT01U.wav --stand-separation-m 1.35
    # → Strategy auto-selected: max_baseline
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# ── lazy pipeline imports ─────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
# Ensure drone_detection is importable whether this file is run as a script
# or imported as part of a package.
_sys.path.insert(0, str(_Path(__file__).parent))

def _get_cfg(cfg=None):
    if cfg is not None:
        return cfg
    from drone_detection.config import config
    return config

def _get_multidrone():
    from drone_detection.multidrone import localize_multi_drone
    return localize_multi_drone

def _get_mems():
    from drone_detection.mems_inference import analyse_mems_file
    return analyse_mems_file

def _get_detect(cfg):
    from drone_detection.inference import load_detection_model, detect
    load_detection_model(cfg)
    return detect

def _get_heuristic():
    from drone_detection.inference import heuristic_detect
    return heuristic_detect

def _get_ap(cfg):
    from drone_detection.audio_processing import AudioProcessor
    return AudioProcessor(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Channel layout constants
# ══════════════════════════════════════════════════════════════════════════════

# Default channel assignments for the Dunakeszi 12-ch rig.
# Override by passing channel_map= to analyse_multichannel_file().
DEFAULT_CHANNEL_MAP = {
    "dpa_a":  list(range(0, 5)),   # stand A channels 0-4
    "dpa_b":  list(range(5, 10)),  # stand B channels 5-9
    "mems":   [10, 11],            # XMOS MEMS channels
}

# Stand A geometry is KNOWN from the photo:
#   5 DPA capsules on a calibrated rail, ~5 cm spacing, y = 0.
#
# Stand B geometry is PARTIALLY known:
#   Same 5-mic linear rail, same 5 cm spacing.
#   UNKNOWN: the distance between stand A and stand B (y-separation).
#
# SAFE DEFAULT: STAND_SEPARATION_M = None
#   → single-stand-only mode until you measure or calibrate the distance.
#
# Update options:
#   set_stand_separation(metres)           # call from Python
#   --stand-separation-m X                # CLI flag
#   calibrate_stand_separation(wav, cfg)  # auto from impulse recording

STAND_A_SPACING_M  = 0.05   # known from photo (calibrated rail)
STAND_B_SPACING_M  = 0.05   # same rail model
STAND_SEPARATION_M: Optional[float] = None   # unknown until measured


def _build_positions(separation_m: Optional[float]) -> dict:
    """Build mic positions. Stand B y = separation_m (NaN if unknown)."""
    pos = {}
    for i in range(5):
        pos[i] = np.array([i * STAND_A_SPACING_M, 0.0])
    y_b = separation_m if separation_m is not None else float("nan")
    for i in range(5):
        pos[5 + i] = np.array([i * STAND_B_SPACING_M, y_b])
    return pos


def set_stand_separation(metres: float) -> None:
    """
    Call once you measure the stand separation.

    Example
    -------
        from multichannel_inference import set_stand_separation
        set_stand_separation(1.35)
    """
    global STAND_SEPARATION_M, DPA_MIC_POSITIONS_M
    STAND_SEPARATION_M = float(metres)
    DPA_MIC_POSITIONS_M = _build_positions(STAND_SEPARATION_M)
    print(f"  Stand separation set to {metres:.3f} m — cross-stand triplets enabled.")


def get_positions() -> dict:
    """Return current position dict."""
    return _build_positions(STAND_SEPARATION_M)


# Initial positions — stand B is NaN until measured
DPA_MIC_POSITIONS_M = get_positions()

# Triplet presets.
# "single_stand" strategies use only stand A or B mics — no cross-stand TDOA.
# "cross_stand" strategies require STAND_SEPARATION_M to be set first.
DPA_TRIPLET_PRESETS = {
    # ── Single-stand only (always available, no separation needed) ────────────
    # Stand A: outermost + centre → triangle with known 10 cm baselines
    "stand_a":        (0, 2, 4),
    # Stand B: same geometry
    "stand_b":        (5, 7, 9),
    # ── Cross-stand (requires set_stand_separation() first) ───────────────────
    # Max baseline: wide triangle across both stands → best at long range
    "max_baseline":   (0, 4, 9),
    # Equilateral-ish across stands
    "equilateral":    (2, 5, 7),
    # Cross-stand general
    "cross_stand":    (0, 4, 7),
}

# Strategies that require the stand separation to be known
_CROSS_STAND_STRATEGIES = {"max_baseline", "equilateral", "cross_stand"}

# Single-stand strategies that work without knowing the separation
_SINGLE_STAND_STRATEGIES = {"stand_a", "stand_b", "best_snr", "custom"}


def _guard_cross_stand(strategy: str) -> None:
    """Raise if a cross-stand strategy is requested without known separation."""
    if strategy in _CROSS_STAND_STRATEGIES and STAND_SEPARATION_M is None:
        raise RuntimeError(
            f"Strategy '{strategy}' requires the stand separation distance.\n"
            "  Option 1: measure it, then call set_stand_separation(metres)\n"
            "  Option 2: auto-calibrate: calibrate_stand_separation(clap_wav, dist_m)\n"
            "  Option 3: use single-stand strategy 'stand_a' or 'stand_b'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Geometry auto-calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_stand_separation(
    impulse_wav:     str | Path,
    known_dist_m:    float,
    cfg              = None,
    impulse_channel: int = 0,
    ref_channel:     int = 4,
    verbose:         bool = True,
) -> float:
    """
    Estimate stand separation from a calibration impulse recording.

    HOW TO DO THE CALIBRATION SHOT
    ───────────────────────────────
    1. Place a known sound source (hand-clap, starter pistol, balloon pop)
       at a measured distance from the array — e.g. 3.0 m in front of stand A.
    2. Record normally with your 12-channel rig.
    3. Call this function with the WAV path and the known_dist_m.

    The function:
      a) Picks two mics from stand A (impulse_channel=0, ref_channel=4)
         to estimate the direction of the calibration source via GCC-PHAT.
      b) Then picks one mic from stand A and one from stand B and uses the
         known source direction + distance to back-calculate the stand
         separation.

    Parameters
    ──────────
    impulse_wav    : path to the calibration recording
    known_dist_m   : distance from source to stand A centre (in metres)
    impulse_channel: channel to treat as stand A reference (default 0)
    ref_channel    : second stand A channel for direction estimation (default 4)
    verbose        : print progress

    Returns
    ───────
    Estimated stand separation in metres.
    Also calls set_stand_separation() automatically.
    """
    import scipy.signal as sp_signal

    cfg_obj = _get_cfg(cfg)
    sr      = cfg_obj.SR
    c       = cfg_obj.SPEED_OF_SOUND

    if verbose:
        print(f"  Calibration: loading channels {impulse_channel}, {ref_channel}, 5 "
              f"from {Path(impulse_wav).name}")

    chs, _ = load_wav_channels(
        impulse_wav,
        [impulse_channel, ref_channel, 5],   # A0, A4, B0
        sr,
    )
    y_a0, y_a4, y_b0 = chs

    # ── Step 1: GCC-PHAT between A0 and A4 → TDOA_AA ─────────────────────────
    # Known physical distance between ch 0 and ch 4 = 4 × 5 cm = 20 cm
    baseline_aa = 4 * STAND_A_SPACING_M  # 0.20 m

    def gcc_phat_tdoa(x: np.ndarray, y: np.ndarray) -> float:
        n   = len(x) + len(y) - 1
        N   = int(2 ** np.ceil(np.log2(n)))
        X   = np.fft.rfft(x, n=N)
        Y   = np.fft.rfft(y, n=N)
        GCC = X * np.conj(Y)
        denom = np.abs(GCC) + 1e-10
        GCC  = GCC / denom
        corr = np.fft.irfft(GCC, n=N)
        lag  = np.argmax(np.abs(corr))
        if lag > N // 2:
            lag -= N
        return lag / sr

    tdoa_aa = gcc_phat_tdoa(y_a0, y_a4)
    if verbose:
        print(f"  TDOA A0→A4 : {tdoa_aa*1e6:.1f} µs  "
              f"(max physical = {baseline_aa/c*1e6:.1f} µs)")

    # ── Step 2: Estimate angle of calibration source from stand A ─────────────
    # TDOA_AA = baseline_AA * cos(theta) / c  → theta = arccos(TDOA_AA * c / baseline_AA)
    cos_theta = float(np.clip(tdoa_aa * c / baseline_aa, -1.0, 1.0))
    theta_rad = math.acos(abs(cos_theta))   # angle from stand A axis
    if verbose:
        print(f"  Source angle from stand A axis: {math.degrees(theta_rad):.1f}°")

    # ── Step 3: GCC-PHAT between A0 and B0 → TDOA_AB ─────────────────────────
    tdoa_ab = gcc_phat_tdoa(y_a0, y_b0)
    if verbose:
        print(f"  TDOA A0→B0 : {tdoa_ab*1e6:.1f} µs")

    # ── Step 4: Back-calculate stand separation ───────────────────────────────
    # Source is at (known_dist_m * sin(theta), known_dist_m * cos(theta))
    # relative to stand A centre (x = 0.10 m, y = 0).
    # A0 is at (0, 0), B0 is at (0, d) where d = stand separation.
    #
    # dist_source_to_A0 = sqrt((src_x)^2 + (src_y)^2)
    # dist_source_to_B0 = sqrt((src_x)^2 + (src_y - d)^2)
    # TDOA_AB = (dist_A0 - dist_B0) / c  →  solve for d

    src_x = known_dist_m * math.sin(theta_rad)
    src_y = known_dist_m * math.cos(theta_rad)
    dist_a0 = math.sqrt(src_x**2 + src_y**2)

    # dist_b0 = dist_a0 - tdoa_ab * c
    dist_b0 = dist_a0 - tdoa_ab * c
    if dist_b0 <= 0:
        raise ValueError(
            f"Invalid calibration: dist_b0={dist_b0:.3f} m (negative). "
            f"Check that the source faces both stands and the recording has "
            f"a clear impulse."
        )

    # d = src_y - sqrt(dist_b0^2 - src_x^2)
    inner = dist_b0**2 - src_x**2
    if inner < 0:
        raise ValueError(
            f"Geometry inconsistency: sqrt({inner:.6f}). "
            f"Verify known_dist_m={known_dist_m} and stand positions."
        )
    sep_m = src_y - math.sqrt(inner)

    if sep_m <= 0.05:
        raise ValueError(
            f"Calibrated separation {sep_m:.3f} m is unrealistically small. "
            f"Check known_dist_m and that the impulse is not behind the array."
        )

    if verbose:
        print(f"  Estimated stand separation: {sep_m:.3f} m")

    set_stand_separation(sep_m)
    return sep_m


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_wav_channels(
    path: str | Path,
    channel_indices: List[int],
    target_sr: int,
) -> Tuple[List[np.ndarray], int]:
    """
    Load specific channels from a multi-channel WAV.
    Returns (list_of_mono_arrays, actual_sr).
    """
    path = Path(path)
    info = sf.info(str(path))
    n_ch = info.channels
    src_sr = info.samplerate

    for idx in channel_indices:
        if idx >= n_ch:
            raise ValueError(
                f"Channel {idx} requested but WAV only has {n_ch} channels "
                f"({path.name})"
            )

    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    # data: (n_samples, n_channels)

    channels = []
    for idx in channel_indices:
        ch = data[:, idx]
        if src_sr != target_sr:
            from math import gcd
            from scipy import signal as sp_signal
            g = gcd(src_sr, target_sr)
            ch = sp_signal.resample_poly(ch, target_sr // g, src_sr // g).astype(np.float32)
        channels.append(ch)

    return channels, target_sr


def channel_snr(y: np.ndarray) -> float:
    """Rough SNR estimate: 90th percentile RMS / 10th percentile RMS in dB."""
    frame = max(int(0.1 * len(y) / 10), 64)
    n_fr  = len(y) // frame
    if n_fr < 4:
        return float(np.mean(y ** 2))
    frames = y[:n_fr * frame].reshape(n_fr, frame)
    rms    = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    noise  = float(np.percentile(rms, 10)) + 1e-12
    sig    = float(np.percentile(rms, 90)) + 1e-12
    return 20 * math.log10(sig / noise)


def build_dpa_triplet(
    dpa_channel_indices: List[int],
    all_channels: List[np.ndarray],
    strategy: str = "stand_a",
    custom: Optional[Tuple[int, int, int]] = None,
    positions: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[np.ndarray], Tuple[int, int, int], Dict]:
    """
    Select 3 DPA channels for TDOA localization.

    Parameters
    ──────────
    dpa_channel_indices : global channel indices for the 10 DPA channels
                          (e.g. list(range(0, 10)) for a 12-ch WAV)
    all_channels        : loaded audio arrays (indexed globally)
    strategy            : 'max_baseline' | 'equilateral' | 'cross_stand' |
                          'stand_a_short' | 'stand_b_short' |
                          'best_snr' | 'custom'
    custom              : 3-tuple of global channel indices (for strategy='custom')
    positions           : dict mapping global channel index → np.array([x, y])
                          defaults to DPA_MIC_POSITIONS_M

    Returns
    ───────
    (channels_3, global_indices_3, info_dict)
    """
    pos = positions or DPA_MIC_POSITIONS_M

    # Guard: cross-stand strategies need the separation to be known
    _guard_cross_stand(strategy)

    if strategy == "custom":
        if custom is None or len(custom) != 3:
            raise ValueError("strategy='custom' requires custom=(i, j, k)")
        idx3 = custom
    elif strategy == "best_snr":
        # Rank DPA channels by SNR, pick top 3 that are non-collinear
        snrs = {i: channel_snr(all_channels[i]) for i in dpa_channel_indices}
        ranked = sorted(dpa_channel_indices, key=lambda i: -snrs[i])
        idx3 = _pick_noncollinear(ranked, pos)
        if idx3 is None:
            # Fall back to max_baseline if geometry is degenerate
            idx3 = _global_from_preset("max_baseline", dpa_channel_indices)
    else:
        idx3 = _global_from_preset(strategy, dpa_channel_indices)

    c0, c1, c2 = idx3
    info = {
        "strategy":      strategy,
        "global_indices": idx3,
        "positions_m":   {c: pos.get(c, pos.get(c - dpa_channel_indices[0]))
                          for c in idx3},
    }
    return [all_channels[c0], all_channels[c1], all_channels[c2]], idx3, info


def _global_from_preset(
    preset: str,
    dpa_indices: List[int],
) -> Tuple[int, int, int]:
    """Map preset local indices to global channel indices."""
    local = DPA_TRIPLET_PRESETS[preset]
    base  = dpa_indices[0]
    return (base + local[0], base + local[1], base + local[2])


def _pick_noncollinear(
    ranked: List[int],
    pos: Dict[int, np.ndarray],
    min_area: float = 1e-4,  # m²
) -> Optional[Tuple[int, int, int]]:
    """
    From ranked channel list, pick the first 3 that form a non-degenerate
    triangle (area > min_area).
    """
    from itertools import combinations
    for a, b, c in combinations(ranked[:6], 3):
        pa = pos.get(a)
        pb = pos.get(b)
        pc = pos.get(c)
        if pa is None or pb is None or pc is None:
            continue
        area = 0.5 * abs(
            (pb[0] - pa[0]) * (pc[1] - pa[1]) -
            (pc[0] - pa[0]) * (pb[1] - pa[1])
        )
        if area > min_area:
            return (a, b, c)
    return None


def _make_cfg_for_triplet(
    base_cfg,
    global_indices: Tuple[int, int, int],
    positions: Dict[int, np.ndarray],
) -> object:
    """
    Return a lightweight cfg-like object with MIC_POSITIONS set to the
    3 positions of the chosen triplet.  All other cfg fields are inherited.
    """
    import copy
    cfg2 = copy.copy(base_cfg)

    c0, c1, c2 = global_indices
    p = positions
    base_idx = min(p.keys())
    m0 = p.get(c0, p.get(c0 - base_idx))
    m1 = p.get(c1, p.get(c1 - base_idx))
    m2 = p.get(c2, p.get(c2 - base_idx))

    cfg2.MIC_POSITIONS = np.array([
        [m0[0], m0[1], 0.0],
        [m1[0], m1[1], 0.0],
        [m2[0], m2[1], 0.0],
    ], dtype=np.float32)

    cx = float(cfg2.MIC_POSITIONS[:, 0].mean())
    cy = float(cfg2.MIC_POSITIONS[:, 1].mean())
    cfg2.ARRAY_CENTER = (cx, cy)

    return cfg2


# ══════════════════════════════════════════════════════════════════════════════
# Main public API
# ══════════════════════════════════════════════════════════════════════════════

def analyse_multichannel_file(
    wav_path:      str | Path,
    cfg            = None,
    channel_map:   Optional[Dict[str, List[int]]] = None,
    dpa_strategy:  str = "stand_a",
    custom_triplet: Optional[Tuple[int, int, int]] = None,
    mic_positions:  Optional[Dict[int, np.ndarray]] = None,
    max_drones:     int = 3,
    show_plots:     bool = False,
    save_plots:     bool = True,
    meta_path:      Optional[str] = None,
) -> Dict:
    """
    Full pipeline analysis of a 12-channel Dunakeszi recording.

    Parameters
    ──────────
    wav_path      : path to multi-channel WAV
    cfg           : Config instance (defaults to global config)
    channel_map   : override default channel assignment, e.g.
                    {"dpa_a": [0,1,2,3,4], "dpa_b": [5,6,7,8,9], "mems": [10,11]}
    dpa_strategy  : how to pick the 3-mic TDOA triplet from DPA channels
                    ('max_baseline' | 'equilateral' | 'cross_stand' |
                     'stand_a_short' | 'stand_b_short' | 'best_snr' | 'custom')
    custom_triplet: (i, j, k) global channel indices when dpa_strategy='custom'
    mic_positions : {global_ch_idx: np.array([x_m, y_m])} — override defaults
    max_drones    : maximum simultaneous drones for multidrone pipeline
    show_plots    : display MEMS dashboards inline
    save_plots    : save MEMS dashboard PNGs
    meta_path     : companion _meta.json path for MEMS cross-check

    Returns
    ───────
    dict with keys:
        "dpa"   : DPA detection + multi-drone localization result
        "mems"  : list of mems_inference results (one per MEMS channel)
        "rig"   : recording rig info (channel map, triplet used, positions)
        "clock_warning": reminder that DPA/MEMS TDOAs cannot be mixed
    """
    cfg        = _get_cfg(cfg)
    wav_path   = Path(wav_path)
    ch_map     = channel_map or DEFAULT_CHANNEL_MAP
    pos        = mic_positions or DPA_MIC_POSITIONS_M
    sr         = cfg.SR

    dpa_all_indices = ch_map.get("dpa_a", []) + ch_map.get("dpa_b", [])
    mems_indices    = ch_map.get("mems", [])
    all_indices     = dpa_all_indices + mems_indices

    print(f"\n{'='*64}")
    print(f"  MULTICHANNEL ANALYSIS  {wav_path.name}")
    print(f"  DPA channels  : {dpa_all_indices}")
    print(f"  MEMS channels : {mems_indices}")
    print(f"  ⚠  Clock domains: DPA sync=yes, MEMS↔DPA sync=NO")
    print(f"{'='*64}\n")

    # ── 1. Load all relevant channels ─────────────────────────────────────────
    print("  Loading channels…")
    all_channels: Dict[int, np.ndarray] = {}
    loaded, actual_sr = load_wav_channels(wav_path, all_indices, sr)
    for ch_idx, audio in zip(all_indices, loaded):
        all_channels[ch_idx] = audio

    # ── 2. Select DPA triplet ─────────────────────────────────────────────────
    print(f"  Building DPA triplet (strategy='{dpa_strategy}')…")
    dpa_triplet_channels, triplet_global_idx, triplet_info = build_dpa_triplet(
        dpa_all_indices, all_channels,
        strategy=dpa_strategy,
        custom=custom_triplet,
        positions=pos,
    )
    print(f"  → Triplet: global channels {triplet_global_idx}")

    # ── 3. DPA detection ──────────────────────────────────────────────────────
    print("  Running DPA detection…")
    detect_fn = _get_detect(cfg)
    heuristic_fn = _get_heuristic()
    ap = _get_ap(cfg)

    # Use the centre channel of the triplet for detection
    ch_detect = dpa_triplet_channels[1]  # middle of 3
    det = detect_fn([ch_detect], cfg)
    final_prob = det["probability"]
    detected   = det["detected"]
    cnn_prob   = det["cnn_probability"]
    heur_prob  = det["heuristic_probability"]
    print(f"  DPA detection: prob={final_prob:.3f}  detected={detected}")

    # ── 4. DPA localization ────────────────────────────────────────────────────
    drones = []
    if detected:
        print("  Running multi-drone localization…")
        localize_fn = _get_multidrone()
        cfg_triplet = _make_cfg_for_triplet(cfg, triplet_global_idx, pos)
        try:
            drones = localize_fn(dpa_triplet_channels, cfg_triplet, max_drones)
            for i, d in enumerate(drones, 1):
                print(f"  Drone {i}: az={d['azimuth_deg']:.1f}°  "
                      f"dist={d['distance_m']:.1f}m  "
                      f"conf_r={d.get('confidence_radius', float('nan')):.2f}m")
        except Exception as e:
            print(f"  ⚠ Localization error: {e}")

    dpa_result = {
        "detected":        detected,
        "probability":     final_prob,
        "cnn_probability": float(cnn_prob),
        "heuristic_probability": float(heur_prob),
        "drones_found":    len(drones),
        "drones":          drones,
        "triplet_channels": triplet_global_idx,
        "triplet_strategy": dpa_strategy,
    }

    # ── 5. MEMS analysis (independent clock — no TDOA mixing) ─────────────────
    print("  Running MEMS analysis (independent clock domain)…")
    mems_results = []
    analyse_mems = _get_mems()
    for mems_ch_idx in mems_indices:
        print(f"  MEMS channel {mems_ch_idx}…")
        # Write this channel to a temp WAV for mems_inference
        import tempfile, os
        tmp = Path(tempfile.mktemp(suffix=f"_ch{mems_ch_idx}.wav"))
        try:
            sf.write(str(tmp), all_channels[mems_ch_idx], actual_sr)
            r = analyse_mems(
                str(tmp), cfg=cfg,
                meta_path=meta_path,
                show_plot=show_plots,
                save_plot=save_plots,
            )
            r["source_channel"] = mems_ch_idx
            mems_results.append(r)
        except Exception as e:
            print(f"  ⚠ MEMS ch {mems_ch_idx} error: {e}")
            mems_results.append({"source_channel": mems_ch_idx, "error": str(e)})
        finally:
            tmp.unlink(missing_ok=True)

    # ── 6. Build combined result ───────────────────────────────────────────────
    return {
        "file":          str(wav_path),
        "dpa":           dpa_result,
        "mems":          mems_results,
        "rig": {
            "channel_map":        ch_map,
            "dpa_triplet":        triplet_global_idx,
            "triplet_strategy":   dpa_strategy,
            "triplet_info":       triplet_info,
            "mic_positions_m":    {k: v.tolist() for k, v in pos.items()},
        },
        "clock_warning": (
            "DPA channels 0-9 are synchronised. "
            "MEMS channels 10-11 are on a SEPARATE USB clock. "
            "TDOA estimation across clock domains is INVALID and not performed."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: analyse all DPA triplet combinations (sensitivity study)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_all_triplets(
    wav_path: str | Path,
    cfg=None,
    channel_map: Optional[Dict[str, List[int]]] = None,
    mic_positions: Optional[Dict[int, np.ndarray]] = None,
    max_drones: int = 3,
) -> Dict[str, List]:
    """
    Run localize_multi_drone() for every preset triplet strategy and return
    all results for comparison. Useful for understanding which array geometry
    gives the most stable localization for your specific stand placement.

    Returns
    ───────
    dict mapping strategy_name → list of drone dicts
    """
    cfg    = _get_cfg(cfg)
    ch_map = channel_map or DEFAULT_CHANNEL_MAP
    pos    = mic_positions or DPA_MIC_POSITIONS_M
    sr     = cfg.SR

    dpa_all = ch_map.get("dpa_a", []) + ch_map.get("dpa_b", [])
    loaded, _ = load_wav_channels(wav_path, dpa_all, sr)
    all_channels = {idx: aud for idx, aud in zip(dpa_all, loaded)}

    localize_fn = _get_multidrone()
    results: Dict[str, List] = {}

    for strategy, local_idx in DPA_TRIPLET_PRESETS.items():
        base = dpa_all[0]
        g0, g1, g2 = base + local_idx[0], base + local_idx[1], base + local_idx[2]
        triplet_chs = [all_channels[g0], all_channels[g1], all_channels[g2]]
        cfg2 = _make_cfg_for_triplet(cfg, (g0, g1, g2), pos)
        try:
            drones = localize_fn(triplet_chs, cfg2, max_drones)
            results[strategy] = drones
            print(f"  {strategy:20s}: {len(drones)} drone(s) found  "
                  f"{[(round(d['azimuth_deg'],1), round(d['distance_m'],1)) for d in drones]}")
        except Exception as e:
            results[strategy] = []
            print(f"  {strategy:20s}: ERROR {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Standalone usage
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(
        description="Analyse a 12-channel Dunakeszi WAV file"
    )
    ap.add_argument("wav", help="Path to 12-channel WAV")
    ap.add_argument("--strategy", default=None,
                    choices=list(DPA_TRIPLET_PRESETS.keys()) + ["best_snr", "custom"],
                    help="DPA triplet selection strategy (default: stand_a when no "
                         "separation is known, max_baseline once separation is set)")
    ap.add_argument("--custom-triplet", default=None,
                    help="Three comma-separated global channel indices for custom strategy")
    ap.add_argument("--meta", default=None,
                    help="Companion _meta.json for MEMS cross-check")
    ap.add_argument("--all-triplets", action="store_true",
                    help="Run all preset triplets and compare results")
    ap.add_argument("--stand-separation-m", type=float, default=None,
                    help="Distance between stand A and stand B rails (metres). "
                         "Required for cross-stand strategies. "
                         "If omitted, single-stand mode is used automatically.")
    ap.add_argument("--calibrate-from", default=None,
                    help="WAV file of a calibration impulse (clap/balloon pop) "
                         "to auto-estimate stand separation. "
                         "Use with --known-dist-m.")
    ap.add_argument("--known-dist-m", type=float, default=3.0,
                    help="Distance of calibration source from stand A (metres, default 3.0)")
    args = ap.parse_args()

    # Apply stand separation if provided
    if args.stand_separation_m is not None:
        set_stand_separation(args.stand_separation_m)
    elif args.calibrate_from:
        print(f"  Auto-calibrating stand separation from {args.calibrate_from}…")
        try:
            sep = calibrate_stand_separation(
                args.calibrate_from, args.known_dist_m
            )
            print(f"  Calibration result: {sep:.3f} m")
        except Exception as e:
            print(f"  ⚠ Calibration failed: {e}")
            print("  Falling back to single-stand mode.")
    else:
        print("  ℹ  No stand separation provided — using single-stand mode (stand_a).")
        print("     To enable cross-stand strategies:")
        print("       --stand-separation-m X       (if you measure it)")
        print("       --calibrate-from CLAP.wav    (auto from impulse recording)")

    # Auto-select strategy: use cross-stand only if separation is known
    if args.strategy is None:
        if STAND_SEPARATION_M is not None:
            args.strategy = "max_baseline"
            print(f"  Strategy auto-selected: max_baseline (separation={STAND_SEPARATION_M:.3f} m known)")
        else:
            args.strategy = "stand_a"
            print("  Strategy auto-selected: stand_a (no separation known)")

    custom_t = None
    if args.custom_triplet:
        parts = [int(x.strip()) for x in args.custom_triplet.split(",")]
        if len(parts) != 3:
            print("--custom-triplet expects exactly 3 comma-separated ints")
            sys.exit(1)
        custom_t = tuple(parts)

    # Import config here when run standalone
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from drone_detection import config
    except ImportError:
        print("⚠ Could not import drone_detection.config — using dummy config")
        config = type("Cfg", (), {
            "SR": 22050, "DETECTION_THRESHOLD": 0.945,
            "MAX_DRONES": 3, "MAX_LOCALIZATION_DIST": 25.0,
            "SPEED_OF_SOUND": 343.0,
        })()

    if args.all_triplets:
        print(f"\n=== Comparing all DPA triplet strategies: {args.wav} ===\n")
        analyse_all_triplets(args.wav, cfg=config)
    else:
        result = analyse_multichannel_file(
            args.wav, cfg=config,
            dpa_strategy=args.strategy,
            custom_triplet=custom_t,
            meta_path=args.meta,
        )
        print(json.dumps({
            "detected":    result["dpa"]["detected"],
            "probability": result["dpa"]["probability"],
            "drones":      [
                {"az": d["azimuth_deg"], "dist": d["distance_m"]}
                for d in result["dpa"]["drones"]
            ],
            "mems_distances": [
                r.get("distance_m") for r in result["mems"]
            ],
            "clock_warning": result["clock_warning"],
        }, indent=2))