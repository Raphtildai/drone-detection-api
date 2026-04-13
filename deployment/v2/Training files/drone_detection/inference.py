# -*- coding: utf-8 -*-
"""
drone_detection/inference.py
─────────────────────────────
Inference pipeline:
  - load_detection_model / load_localization_model / reload_models
  - heuristic_detect()          — per-frame entropy + comb score
  - detect()                    — CNN + heuristic hybrid
  - localize()                  — single-drone localisation
  - localize_multi_drone()      — multi-drone TDOA (v2 Cartesian solver)
  - run_pipeline()              — detect + localize + track in one call
  - analyse_audio_file()        — segment-level analysis + 6-panel dashboard
  - analyse_external_audio_robust()
  - load_3ch(), _mel_tensor(), _ipd_tensor()
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.optimize
import scipy.signal
import torch
from tqdm.auto import tqdm

from drone_detection.audio import AudioProcessor, load_audio_any, synthesise_drone
from drone_detection.config import config as _default_cfg
from drone_detection.models import DetectionCNN, make_localization_model
from drone_detection.utils import (
    angular_error_deg,
    azimuth_deg_to_xy,
    bandpass,
    classify_detection_score,
    gcc_phat_peaks,
    safe_prob_average,
    sigmoid,
    wrap_angle_deg,
    xy_to_azimuth_deg,
)

# ── Module-level model cache (load once per session) ─────────────────────────
_det_model = None
_loc_model = None


def load_detection_model(cfg=None) -> DetectionCNN:
    global _det_model
    if _det_model is not None:
        return _det_model
    cfg  = cfg or _default_cfg
    ckpt = cfg.DRIVE_MODELS / "best_detection.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No detection checkpoint at {ckpt}. Run train_detection() first.")
    dev  = torch.device(cfg.DEVICE)
    data = torch.load(ckpt, map_location=dev)
    m    = DetectionCNN().to(dev)
    m.load_state_dict(data["model_state"]); m.eval()
    cfg.DETECTION_THRESHOLD = float(data.get("best_threshold", cfg.DETECTION_THRESHOLD))
    _det_model = m
    print(f"✅ Detection model loaded (ep={data.get('epoch','?')}, "
          f"f1={data.get('best_val_f1','?')}, thr={cfg.DETECTION_THRESHOLD:.3f})")
    return m


def load_localization_model(cfg=None):
    global _loc_model
    if _loc_model is not None:
        return _loc_model
    cfg  = cfg or _default_cfg
    ckpt = cfg.DRIVE_MODELS / "best_localization.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No localization checkpoint at {ckpt}. Run train_localization() first.")
    dev  = torch.device(cfg.DEVICE)
    data = torch.load(ckpt, map_location=dev)
    m    = make_localization_model(cfg).to(dev)
    m.load_state_dict(data["model_state"]); m.eval()
    _loc_model = m
    print(f"✅ Localization model loaded (ep={data.get('epoch','?')}, "
          f"val_loss={data.get('best_val_loss','?'):.5f})")
    return m


def reload_models(cfg=None) -> None:
    global _det_model, _loc_model
    _det_model = _loc_model = None
    load_detection_model(cfg); load_localization_model(cfg)


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _mel_tensor(channels: list, cfg=None) -> torch.Tensor:
    cfg  = cfg or _default_cfg
    ap   = AudioProcessor(cfg)
    mels = [ap.mel(ap.pad_or_truncate(c)) for c in channels]
    return torch.tensor(np.stack(mels, axis=0), dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)


def _ipd_tensor(channels: list, cfg=None) -> torch.Tensor:
    from drone_detection.audio import compute_ipd_features
    cfg = cfg or _default_cfg
    ipd = compute_ipd_features(channels, cfg)
    return torch.tensor(ipd, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)


def load_3ch(paths: list, cfg=None) -> list:
    cfg = cfg or _default_cfg
    ap  = AudioProcessor(cfg)
    return [ap.pad_or_truncate(ap.load(p)) for p in paths]


# ═══════════════════════════════════════════════════════════════════════════════
# Heuristic detection
# ═══════════════════════════════════════════════════════════════════════════════

def _per_frame_entropy(S: np.ndarray, min_energy_fraction: float = 0.01) -> np.ndarray:
    """
    Normalised spectral entropy per STFT frame.
    Frames below min_energy_fraction × max_frame_energy are set to NaN so
    callers can use np.nanmedian to ignore silent frames.
    """
    S = np.asarray(S, np.float64) + 1e-10
    frame_energy = S.sum(axis=0)
    energy_gate  = frame_energy.max() * min_energy_fraction
    active       = frame_energy >= energy_gate
    P      = S / frame_energy[np.newaxis, :]
    H      = -np.sum(P * np.log2(P + 1e-12), axis=0)
    H_norm = (H / math.log2(S.shape[0])).astype(np.float32)
    H_norm[~active] = np.nan
    return H_norm


def _harmonic_comb_score(y: np.ndarray, sr: int, f0_min: float = 80.0,
                          f0_max: float = 350.0, n_harmonics: int = 6) -> float:
    """Score how well the spectrum matches a drone harmonic comb."""
    import librosa as _lib
    S       = np.abs(_lib.stft(y, n_fft=2048, hop_length=512))
    S_mean  = S.mean(axis=1)
    freqs   = _lib.fft_frequencies(sr=sr, n_fft=2048)
    best    = 0.0
    for f0 in np.linspace(f0_min, f0_max, 30):
        harmonic_energy = 0.0
        for k in range(1, n_harmonics + 1):
            target = f0 * k
            if target > sr / 2: break
            idx = int(np.argmin(np.abs(freqs - target)))
            w   = max(1, int(20 / (freqs[1] - freqs[0])))
            harmonic_energy += float(S_mean[max(0,idx-w):idx+w+1].max())
        score = harmonic_energy / (float(S_mean.sum()) + 1e-10) / n_harmonics * n_harmonics
        if score > best: best = score
    return float(min(best, 1.0))


def _impulsiveness(y: np.ndarray) -> float:
    rms  = float(np.sqrt(np.mean(y**2)) + 1e-8)
    peak = float(np.max(np.abs(y)) + 1e-8)
    return float(np.clip(peak / rms, 1.0, 10.0))


def heuristic_detect(audio: np.ndarray, cfg=None) -> dict:
    """
    Heuristic drone detector based on:
      - Per-frame spectral entropy (absolute-energy gated)
      - Crest factor (high → impulsive → not drone)
      - Harmonic comb score
      - F0 stability + voiced ratio

    Correctly vetoes car horns, sirens, and near-silent clips.
    """
    import librosa as _lib
    import warnings

    cfg = cfg or _default_cfg
    y   = np.asarray(audio, np.float32)
    if len(y) == 0:
        return {"probability": 0.0, "label": "non_drone", "features": {}}

    rms    = float(np.sqrt(np.mean(y**2)) + 1e-8)
    rms_db = 20.0 * math.log10(rms + 1e-8)

    try:
        S = np.abs(_lib.stft(y, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH))
        frame_ent  = _per_frame_entropy(S, min_energy_fraction=0.01)
        voiced_ent = frame_ent[~np.isnan(frame_ent)]
        if len(voiced_ent) == 0:
            return {"probability": 0.0, "label": "non_drone",
                    "features": {"rms_db": rms_db, "veto": "silent_clip"}}
        median_ent = float(np.median(voiced_ent))
        p10_ent    = float(np.percentile(voiced_ent, 10))
        centroid   = float(np.mean(_lib.feature.spectral_centroid(S=S, sr=cfg.SR)))
        rolloff    = float(np.mean(_lib.feature.spectral_rolloff(S=S, sr=cfg.SR, roll_percent=0.85)))
        bandwidth  = float(np.mean(_lib.feature.spectral_bandwidth(S=S, sr=cfg.SR)))
    except Exception:
        centroid = rolloff = bandwidth = 0.0
        median_ent = p10_ent = 0.5

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            f0, _, _ = _lib.pyin(y, fmin=50.0, fmax=500.0, sr=cfg.SR,
                                  hop_length=cfg.HOP_LENGTH, fill_na=0.0)
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        voiced_ratio = float(np.mean(f0 > 0.0))
        voiced_f0    = f0[f0 > 0.0]
        f0_med       = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std       = float(np.std(voiced_f0))    if len(voiced_f0) > 0 else 0.0
    except Exception:
        voiced_ratio = f0_med = f0_std = 0.0

    try:
        comb = _harmonic_comb_score(y, cfg.SR) if len(y) >= cfg.SR else 0.0
    except Exception:
        comb = 0.0

    cf = _impulsiveness(y)

    # ── Veto 1: low active-frame entropy → narrow-band tonal content ─────────
    if median_ent < 0.38:
        prob = float(np.clip(median_ent / 0.38 * 0.25, 0.0, 0.25))
        return {"probability": prob, "label": classify_detection_score(prob, cfg),
                "features": {"rms_db": rms_db, "median_frame_entropy": median_ent,
                              "p10_entropy": p10_ent, "crest_factor": cf,
                              "centroid_hz": centroid, "veto": "low_frame_entropy"}}

    # ── Veto 2: high crest factor + moderately narrow ─────────────────────────
    if cf > 8.5 and median_ent < 0.52:
        return {"probability": 0.10, "label": classify_detection_score(0.10, cfg),
                "features": {"rms_db": rms_db, "median_frame_entropy": median_ent,
                              "crest_factor": cf, "veto": "high_crest_factor"}}

    # ── Normal scoring ────────────────────────────────────────────────────────
    score = (
        0.14 * float(np.clip((rms_db + 45.0) / 25.0, 0.0, 1.0))
        + 0.16 * (1.0 if 80 <= f0_med <= 350 else 0.3 if 50 <= f0_med < 80 else 0.0)
        + 0.10 * float(np.clip(voiced_ratio / 0.50, 0.0, 1.0))
        + 0.08 * (1.0 - float(np.clip(f0_std / 80.0, 0.0, 1.0)) if f0_med > 0 else 0.2)
        + 0.10 * (1.0 if 120 <= centroid <= 4000 else 0.35)
        + 0.07 * (1.0 if 300 <= rolloff  <= 8000 else 0.4)
        + 0.07 * (1.0 if 150 <= bandwidth <= 4000 else 0.5)
        + 0.16 * float(np.clip((median_ent - 0.35) / 0.35, 0.0, 1.0))
        + 0.12 * float(np.clip(comb * 3.0, 0.0, 1.0))
    )

    prob = sigmoid(8.0 * (score - 0.50))
    return {
        "probability": float(prob),
        "label":       classify_detection_score(prob, cfg),
        "features": {
            "rms_db": rms_db, "median_frame_entropy": median_ent, "p10_entropy": p10_ent,
            "crest_factor": cf, "comb_score": comb, "centroid_hz": centroid,
            "rolloff_hz": rolloff, "bandwidth_hz": bandwidth,
            "voiced_ratio": voiced_ratio, "f0_median_hz": f0_med, "f0_std_hz": f0_std,
        },
    }


# ── RMS gate threshold ────────────────────────────────────────────────────────
_RMS_FLOOR_DB = -45.0


def detect(channels: list, cfg=None, use_hybrid: bool = True) -> dict:
    """
    Hybrid CNN + heuristic drone detector.

    Parameters
    ----------
    channels     : list of at least 1 float32 array (mono waveforms)
    cfg          : Config object
    use_hybrid   : if False, use raw CNN probability only

    Returns
    -------
    dict with keys: detected, probability, label,
                    cnn_probability, heuristic_probability, heuristic_features
    """
    cfg = cfg or _default_cfg
    ap  = AudioProcessor(cfg)
    y0  = ap.pad_or_truncate(np.asarray(channels[0], np.float32))
    rms_db = float(20 * math.log10(float(np.sqrt(np.mean(y0**2))) + 1e-8))

    # Near-silent gate
    if rms_db < _RMS_FLOOR_DB:
        return {"detected": False, "probability": 0.0, "label": "non_drone",
                "cnn_probability": 0.0, "heuristic_probability": 0.0,
                "heuristic_features": {"veto": "below_rms_floor", "rms_db": rms_db}}

    m    = load_detection_model(cfg)
    mel0 = ap.mel(y0)
    m0   = torch.tensor(np.stack([mel0, mel0, mel0], axis=0), dtype=torch.float32
                        ).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        cnn_prob = float(torch.softmax(m(m0), dim=1)[0, 1].item())

    heur           = heuristic_detect(y0, cfg)
    heuristic_prob = float(heur["probability"])

    if use_hybrid:
        fused = 0.80 * cnn_prob + 0.20 * heuristic_prob
        if cnn_prob > 0.45 and heuristic_prob > 0.45:
            fused = min(1.0, fused + 0.06)
        if cnn_prob < 0.55 and heur["features"].get("veto"):
            fused = min(fused, 0.40)
    else:
        fused = cnn_prob

    label = classify_detection_score(fused, cfg)
    return {
        "detected":              bool(fused >= cfg.DETECTION_THRESHOLD),
        "probability":           float(fused),
        "label":                 label,
        "cnn_probability":       float(cnn_prob),
        "heuristic_probability": float(heuristic_prob),
        "heuristic_features":    heur["features"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-drone localisation
# ═══════════════════════════════════════════════════════════════════════════════

def localize(channels: list, cfg=None) -> dict:
    """Run the single-drone CNN localiser and return azimuth, distance, height."""
    cfg = cfg or _default_cfg
    m   = load_localization_model(cfg)
    mel = _mel_tensor(channels, cfg); ipd = _ipd_tensor(channels, cfg)
    with torch.no_grad():
        pred = m(mel, ipd)[0].cpu().numpy()
    sin_az, cos_az, dist_raw, ht_raw = pred
    az_deg = wrap_angle_deg(float(np.degrees(np.arctan2(sin_az, cos_az))))
    dist_m = float(abs(dist_raw) * cfg.MAX_LOCALIZATION_DIST)
    ht_m   = float(abs(ht_raw)   * cfg.MAX_LOCALIZATION_DIST)
    xy     = azimuth_deg_to_xy(az_deg, dist_m, cfg.ARRAY_CENTER)
    return {"azimuth_deg": az_deg, "distance_m": dist_m, "height_m": ht_m, "xy_position": xy}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-drone localisation  (v2 — Cartesian solver, no hard boundary clip)
# ═══════════════════════════════════════════════════════════════════════════════

# ── v2 constants ──────────────────────────────────────────────────────────────
_MD_MIN_DIST_M      = 0.30    # reject solutions closer than this to array
_MD_TDOA_DEDUP_S    = 0.029e-3  # 29 µs  (was 0.05 µs — 580× too tight)
_MD_RESIDUAL_THR    = 1e-8    # tighter than original 1e-6
_MD_POS_DEDUP_M     = 0.50
_MD_BARRIER_MIN_R   = 0.25
_MD_BARRIER_WEIGHT  = 1e-6    # repel from array centre
_MD_BARRIER_OUT     = 5e-8    # soft repel from max_dist boundary


def _tdoa_residual_cartesian(
    xy, tdoas: np.ndarray, mics: np.ndarray, c: float,
    cx: float, cy: float, max_dist: float,
) -> float:
    """TDOA residual in Cartesian (x,y) with inner + outer soft barriers."""
    pos = np.asarray(xy, dtype=float)
    d   = np.linalg.norm(mics - pos[None, :], axis=1)
    res = (((d[1]-d[0])/c - tdoas[0])**2 +
           ((d[2]-d[0])/c - tdoas[1])**2 +
           ((d[2]-d[1])/c - tdoas[2])**2)
    r = float(np.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2))
    if r < _MD_BARRIER_MIN_R:
        res += _MD_BARRIER_WEIGHT / max((r - 0.01)**2, 1e-12)
    if r > max_dist:
        res += _MD_BARRIER_OUT * (r - max_dist)**2
    return float(res)


def _nelder_mead_cartesian(
    tdoas: np.ndarray, mics: np.ndarray, c: float, max_dist: float
) -> tuple[np.ndarray, float]:
    """Unconstrained Cartesian Nelder-Mead with multi-radius grid seeding."""
    cx, cy    = mics.mean(axis=0)
    grid_radii = [0.5, 1.0, 2.0, 3.5, 6.0, 10.0, 15.0, 20.0]
    best_e     = float("inf"); best_xy = np.array([cx + 2.0, cy])
    for r in grid_radii:
        for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
            xy = np.array([cx + r*np.cos(a), cy + r*np.sin(a)])
            e  = _tdoa_residual_cartesian(xy, tdoas, mics, c, cx, cy, max_dist)
            if e < best_e: best_e = e; best_xy = xy.copy()
    res = scipy.optimize.minimize(
        lambda xy: _tdoa_residual_cartesian(xy, tdoas, mics, c, cx, cy, max_dist),
        x0=best_xy, method="Nelder-Mead",
        options={"xatol":1e-7,"fatol":1e-16,"maxiter":30000,"adaptive":True},
    )
    pos = np.array(res.x, dtype=np.float32)
    err = _tdoa_residual_cartesian(pos, tdoas, mics, c, cx, cy, max_dist)
    return pos, err


def localize_multi_drone(channels: list, cfg=None, max_drones: int = None) -> list:
    """
    Multi-drone TDOA localisation (v2 Cartesian solver).

    Fixes vs original:
      - TDOA dedup window: 29 µs (was 0.05 µs — 580× too tight)
      - Cartesian solver: no hard distance boundary clip
      - Soft outer barrier instead of hard max_dist clipping
      - Minimum solution distance: 0.30 m (rejects r ≈ 0 degenerate solutions)
      - Residual acceptance: 1e-8 (tighter than original 1e-6)
      - Position dedup: 0.50 m (was 0.15 m — too tight for noisy estimates)
    """
    cfg        = cfg or _default_cfg
    max_drones = max_drones or cfg.MAX_DRONES
    mics       = cfg.MIC_POSITIONS; c = cfg.SPEED_OF_SOUND; sr = cfg.SR
    max_tau    = (np.max([np.linalg.norm(mics[i]-mics[j])
                          for i in range(3) for j in range(i+1,3)]) / c * 1.5)
    chs_bp  = [bandpass(ch, sr, 200, 5000) for ch in channels]
    n_peaks = max_drones + 2
    peaks12 = gcc_phat_peaks(chs_bp[1], chs_bp[0], sr, max_tau, n_peaks)
    peaks13 = gcc_phat_peaks(chs_bp[2], chs_bp[0], sr, max_tau, n_peaks)
    peaks23 = gcc_phat_peaks(chs_bp[2], chs_bp[1], sr, max_tau, n_peaks)

    DEDUP      = _MD_TDOA_DEDUP_S
    candidates = []
    for tau12, s12 in peaks12:
        for tau13, s13 in peaks13:
            tau23_pred = tau13 - tau12
            best23     = min(peaks23, key=lambda x: abs(x[0] - tau23_pred))
            if abs(best23[0] - tau23_pred) < DEDUP * 50:
                candidates.append((tau12, tau13, tau13-tau12, s12+s13+best23[1]))
    candidates.sort(key=lambda x: -x[3])

    drones: list   = []; seen_pos: list = []; seen_tds: list = []
    cx, cy = cfg.ARRAY_CENTER

    for tau12, tau13, tau23, _ in candidates:
        if len(drones) >= max_drones: break
        if any(abs(tau12-st[0]) < DEDUP and abs(tau13-st[1]) < DEDUP for st in seen_tds):
            continue

        tdoas = np.array([tau12, tau13, tau23])
        pos, err = _nelder_mead_cartesian(tdoas, mics, c, cfg.MAX_LOCALIZATION_DIST)

        dist_from_centre = float(np.sqrt((pos[0]-cx)**2 + (pos[1]-cy)**2))
        if dist_from_centre < _MD_MIN_DIST_M: continue
        if err > _MD_RESIDUAL_THR:             continue
        if any(np.linalg.norm(pos - sp) < _MD_POS_DEDUP_M for sp in seen_pos): continue

        # Confidence radius from finite-difference Hessian
        try:
            eps  = max(dist_from_centre * 0.05, 0.10)
            e00  = err
            e_xp = _tdoa_residual_cartesian(pos+[eps,0], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_xm = _tdoa_residual_cartesian(pos-[eps,0], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_yp = _tdoa_residual_cartesian(pos+[0,eps], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_ym = _tdoa_residual_cartesian(pos-[0,eps], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            hxx  = max((e_xp + e_xm - 2*e00) / eps**2, 1e-10)
            hyy  = max((e_yp + e_ym - 2*e00) / eps**2, 1e-10)
            cr   = float(np.clip(np.sqrt(0.5/hxx + 0.5/hyy), 0.05, cfg.MAX_LOCALIZATION_DIST))
        except Exception:
            cr = float("nan")

        az_deg = math.degrees(math.atan2(pos[1]-cy, pos[0]-cx))
        drones.append({"xy_position": pos, "azimuth_deg": az_deg,
                       "distance_m": dist_from_centre, "tdoa_residual": err, "confidence_radius": cr})
        seen_pos.append(pos.copy()); seen_tds.append((tau12, tau13))

    return drones


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    wav_paths: list, cfg=None, tracker=None,
    multi_drone: bool = False, timestamp: float = None,
) -> dict:
    """Detect → localise → track in a single call."""
    cfg      = cfg or _default_cfg
    ts       = timestamp or time.time()
    channels = load_3ch(wav_paths, cfg)
    det      = detect(channels, cfg)
    result   = {"detected": det["detected"], "probability": det["probability"],
                "cnn_probability": det.get("cnn_probability", float("nan")),
                "heuristic_probability": det.get("heuristic_probability", float("nan")),
                "drones": [], "tracks": []}
    if not det["detected"]:
        if tracker: result["tracks"] = tracker.step([], ts)
        return result
    if multi_drone:
        drone_locs = localize_multi_drone(channels, cfg)
        result["drones"] = drone_locs
        positions  = [d["xy_position"] for d in drone_locs]
    else:
        loc = localize(channels, cfg)
        result["drones"] = [loc]; positions = [loc["xy_position"]]
    if tracker and positions:
        result["tracks"] = tracker.step(positions, ts)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Segment-level analysis helpers (used by analyse_audio_file)
# ═══════════════════════════════════════════════════════════════════════════════

def synthesise_3ch(
    audio: np.ndarray, sr: int, drone_pos, mic_positions: np.ndarray
) -> list:
    """Apply TDOA delays to convert mono audio to 3-channel simulation."""
    from drone_detection.utils import _fractional_delay
    c   = 343.0; src = np.array(drone_pos, dtype=float); n = len(audio)
    d   = np.linalg.norm(mic_positions - src[None, :], axis=1)
    rel = (d - d.min()) / c * sr
    out = []
    for i in range(len(mic_positions)):
        delayed = _fractional_delay(audio.copy(), rel[i])
        out.append(delayed + (0.003 * np.random.randn(n)).astype(np.float32))
    return out


def analyse_audio_file(
    audio_path: str, cfg=None, drone_pos=None, n_segments: int = 10,
    threshold_override: float = None, show_plot: bool = True,
    use_synthesis: bool = False,
) -> dict:
    """
    Segment-level analysis of an audio file with the 6-panel dark dashboard.

    Parameters
    ----------
    use_synthesis : False (default) — direct mono detection (recommended);
                    True  — legacy 3-mic TDOA simulation path.
    """
    from drone_detection.visualization import _plot_analysis_report
    from drone_detection.tracking import KalmanTracker

    cfg = cfg or _default_cfg
    if threshold_override is not None:
        old_thr = cfg.DETECTION_THRESHOLD; cfg.DETECTION_THRESHOLD = threshold_override

    ap      = AudioProcessor(cfg)
    y_full  = ap.load(audio_path, mono=True)
    total   = len(y_full) / cfg.SR
    seg_s   = int(cfg.TARGET_DURATION * cfg.SR)
    hop     = max(seg_s, int((len(y_full)-seg_s) / max(n_segments-1, 1)))
    dp      = drone_pos or [1.0, 0.8]
    load_detection_model(cfg)
    try:
        load_localization_model(cfg); can_localize = True
    except FileNotFoundError:
        can_localize = False

    tracker  = KalmanTracker(cfg); segments = []; base_ts = time.time()
    mode_str = "synthesis" if use_synthesis else "direct mono"
    print(f"\n🎵 {Path(audio_path).name}  ({total:.1f}s)  |  {n_segments} segments  |  mode={mode_str}")

    for seg_i in range(n_segments):
        start = min(seg_i * hop, max(0, len(y_full)-seg_s))
        audio = y_full[start:start+seg_s]
        if len(audio) < seg_s: audio = np.pad(audio, (0, seg_s-len(audio)))
        t_s     = start / cfg.SR
        mel_f   = ap.mel(ap.pad_or_truncate(audio))
        rms_db  = float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8))

        import soundfile as sf
        if use_synthesis:
            chs = synthesise_3ch(audio, cfg.SR, dp, cfg.MIC_POSITIONS)
            det = detect(chs, cfg)
            tmp_paths = []
            for ch in chs:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tf.name, ch, cfg.SR); tmp_paths.append(tf.name)
            try:
                res = run_pipeline(tmp_paths, cfg, tracker=tracker, timestamp=base_ts+t_s)
                res["cnn_probability"]       = det.get("cnn_probability", float("nan"))
                res["heuristic_probability"] = det.get("heuristic_probability", float("nan"))
            finally:
                for p in tmp_paths: os.unlink(p)
            drone_loc = res["drones"][0] if res["drones"] else None
        else:
            det       = detect([audio, audio, audio], cfg)
            drone_loc = None
            if det["detected"] and can_localize:
                drone_loc = {"azimuth_deg": 0.0, "distance_m": 0.0, "height_m": 0.0,
                             "xy_position": np.array(cfg.ARRAY_CENTER)}
            positions = [drone_loc["xy_position"]] if drone_loc else []
            tracks    = tracker.step(positions, base_ts+t_s)
            res = {"detected": det["detected"], "probability": det["probability"],
                   "cnn_probability": det.get("cnn_probability", float("nan")),
                   "heuristic_probability": det.get("heuristic_probability", float("nan")),
                   "drones": [drone_loc] if drone_loc else [], "tracks": tracks}

        segments.append({
            "seg": seg_i+1, "t_start": t_s,
            "detected": res["detected"], "prob": res["probability"],
            "cnn_probability":       res.get("cnn_probability", float("nan")),
            "heuristic_probability": res.get("heuristic_probability", float("nan")),
            "xy":  res["drones"][0]["xy_position"] if res["drones"] else None,
            "loc": res["drones"][0] if res["drones"] else None,
            "mel": mel_f, "rms_db": rms_db,
        })
        icon = "🚁" if res["detected"] else "🌳"
        print(f"  Seg {seg_i+1:3d}  {icon}  conf={res['probability']:.3f}  rms={rms_db:.1f}dB")

    confirmed = tracker.all_confirmed()
    n_det = sum(s["detected"] for s in segments)
    print(f"\n  📊 {n_det}/{n_segments} detected  |  {len(confirmed)} confirmed track(s)")
    if show_plot:
        _plot_analysis_report(segments, confirmed, cfg, Path(audio_path).name)
    if threshold_override is not None:
        cfg.DETECTION_THRESHOLD = old_thr
    return {"segments": segments, "tracker": tracker, "tracks": confirmed,
            "detected": n_det > 0,
            "probability": max((s["prob"] for s in segments), default=0.0),
            "duration_sec": total}


def analyse_external_audio_robust(
    audio_path: str, cfg=None, segment_sec: float = None, overlap: float = None,
    threshold: float = None, min_pos_segments: int = None, agg_mode: str = None,
    topk: int = None, show_plot: bool = True,
) -> dict:
    """
    Robust external audio analysis using overlapping segments and
    configurable score aggregation.
    """
    from drone_detection.visualization import PLOT_STYLE, _apply_dark_style
    import matplotlib.pyplot as plt

    cfg              = cfg or _default_cfg
    segment_sec      = segment_sec or cfg.EXTERNAL_SEGMENT_SEC
    overlap          = cfg.EXTERNAL_SEGMENT_OVERLAP if overlap is None else overlap
    threshold        = threshold if threshold is not None else cfg.EXTERNAL_INFER_THRESHOLD
    min_pos_segments = min_pos_segments or cfg.EXTERNAL_MIN_POS_SEGMENTS
    agg_mode         = agg_mode or cfg.EXTERNAL_AGG_MODE
    topk             = topk or cfg.EXTERNAL_TOPK

    ap      = AudioProcessor(cfg); y = ap.load(audio_path, mono=True)
    total_s = len(y) / cfg.SR
    seg_len = max(1, int(segment_sec * cfg.SR))
    hop_len = max(1, int(seg_len * (1.0 - overlap)))
    starts  = list(range(0, len(y)-seg_len+1, hop_len)) if len(y) > seg_len else [0]
    if starts and starts[-1] != len(y)-seg_len: starts.append(len(y)-seg_len)

    segment_results = []
    for i, start in enumerate(starts):
        seg = ap.pad_or_truncate(y[start:start+seg_len])
        res = detect([seg, seg, seg], cfg, use_hybrid=True)
        t0  = start / cfg.SR; t1 = (start + seg_len) / cfg.SR
        segment_results.append({
            "segment_index": i+1, "t_start_s": float(t0), "t_end_s": float(t1),
            "probability": float(res["probability"]),
            "cnn_probability": float(res["cnn_probability"]),
            "heuristic_probability": float(res["heuristic_probability"]),
            "label": res["label"],
            "detected_at_main_threshold":     bool(res["probability"] >= cfg.DETECTION_THRESHOLD),
            "detected_at_external_threshold": bool(res["probability"] >= threshold),
        })

    probs       = [s["probability"] for s in segment_results]
    probs_sorted = sorted(probs, reverse=True)
    if agg_mode == "max":          clip_score = float(max(probs)) if probs else 0.0
    elif agg_mode == "mean_topk":  clip_score = safe_prob_average(probs_sorted[:max(1,topk)])
    elif agg_mode == "vote":       clip_score = float(sum(p>=threshold for p in probs)/max(len(probs),1))
    else:                          clip_score = safe_prob_average(probs_sorted[:max(1,topk)])

    pos_count     = sum(s["detected_at_external_threshold"] for s in segment_results)
    clip_detected = clip_score >= threshold or pos_count >= min_pos_segments
    clip_label    = ("drone" if clip_detected and clip_score >= threshold
                     else "possible_drone" if pos_count >= 1 or clip_score >= cfg.DETECTION_THRESHOLD_WEAK
                     else "non_drone")

    print(f"\n🎧 {Path(audio_path).name} | duration={total_s:.2f}s | {len(segment_results)} segments")
    for s in segment_results:
        icon = "🚁" if s["detected_at_external_threshold"] else "🌳"
        print(f"  Seg {s['segment_index']:3d} {icon} prob={s['probability']:.3f} "
              f"cnn={s['cnn_probability']:.3f} heur={s['heuristic_probability']:.3f}")
    print(f"\n📊 Clip score: {clip_score:.3f}  |  Positive: {pos_count}/{len(segment_results)}  |  Label: {clip_label}")

    if show_plot:
        t     = [0.5*(s["t_start_s"]+s["t_end_s"]) for s in segment_results]
        fused = [s["probability"] for s in segment_results]
        cnn   = [s["cnn_probability"] for s in segment_results]
        heur  = [s["heuristic_probability"] for s in segment_results]
        cols  = [PLOT_STYLE["ok"] if s["detected_at_external_threshold"] else PLOT_STYLE["err"]
                 for s in segment_results]
        fig, ax = plt.subplots(figsize=(12, 5))
        _apply_dark_style(fig, [ax])
        width = max((t[1]-t[0])*0.8 if len(t)>1 else 0.5, 0.15)
        ax.bar(t, fused, width=width, color=cols, alpha=0.55, label="Hybrid prob")
        ax.plot(t, cnn,  "-o", color=PLOT_STYLE["accent"], label="CNN prob")
        ax.plot(t, heur, "--s", color=PLOT_STYLE["purple"], label="Heuristic prob")
        ax.axhline(threshold, color=PLOT_STYLE["warn"], ls="--", lw=1.5, label=f"Ext thr={threshold:.2f}")
        ax.axhline(cfg.DETECTION_THRESHOLD, color=PLOT_STYLE["err"], ls=":", lw=1.5)
        ax.set_title(f"Robust External Detection — {Path(audio_path).name}", color=PLOT_STYLE["accent"])
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Probability"); ax.set_ylim(0, 1.05)
        ax.legend(facecolor=PLOT_STYLE["panel"]); plt.tight_layout(); plt.show()

    return {"file": str(audio_path), "duration_s": float(total_s),
            "segment_results": segment_results, "clip_score": float(clip_score),
            "positive_segments": int(pos_count), "segments_total": int(len(segment_results)),
            "clip_detected": bool(clip_detected), "clip_label": clip_label,
            "aggregation_mode": agg_mode, "external_threshold": float(threshold)}