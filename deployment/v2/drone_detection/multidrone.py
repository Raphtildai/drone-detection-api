# -*- coding: utf-8 -*-
"""
multidrone.py
─────────────
Multi-drone acoustic localization using GCC-PHAT TDOA estimation
followed by Cartesian Nelder-Mead optimisation.

This module integrates ALL fixes from multidrone_localization_patch.py (v1)
and multidrone_localization_patch_v2.py (v2) directly — no monkey-patching
at runtime is needed.

Key improvements over the original localize_multi_drone()
──────────────────────────────────────────────────────────
v1 fixes
  - TDOA dedup window: 29 µs (was 0.05 µs — 580× too tight)
  - Minimum solution distance 0.30 m (rejects degenerate r≈0 solutions)
  - Grid search starts at 0.5 m (never from zero)
  - Soft inner penalty repels solver from array centre

v2 fixes
  - Solver works in Cartesian (x,y) instead of polar (r,θ) — no hard
    boundary clipping that caused distance to saturate at MAX_LOCALIZATION_DIST
  - Soft outer barrier replaces hard clip for r > max_dist
  - Hessian-based confidence radius computed at the Cartesian optimum
  - Residual acceptance threshold tightened to 1e-8

Public API
──────────
localize_multi_drone(channels, cfg, max_drones=None)
    Drop-in replacement for the original function; uses v2 solver.
"""

import math
from typing import List, Optional

import numpy as np
import scipy.optimize

from .config import Config, config
from .utils import bandpass, gcc_phat_peaks


# ══════════════════════════════════════════════════════════════════════════════
# Tunable constants
# ══════════════════════════════════════════════════════════════════════════════

# Minimum accepted solution distance from the array centre (metres).
# Solutions closer than this are degenerate (r ≈ 0 satisfies any TDOA).
_MIN_SOLUTION_DIST_M = 0.30

# TDOA dedup window (seconds).
# Physical limit for the UaVirBASE 3-mic selection (3.44 m baseline @ 343 m/s ≈ 10.0 ms).
# We use 5% of that as the dedup window: 0.5 ms.
# (The original 29 µs was designed for a 200 mm array and is 17× too tight
#  for the 3.44 m baseline used in this thesis.)
_TDOA_DEDUP_S = 0.5e-3          # 500 µs = 5% of 10 ms physical limit

# Residual acceptance threshold.  Only solutions with TDOA residual below
# this are accepted.  Tightened from 1e-6 to 1e-8 in v2.
_RESIDUAL_THRESHOLD = 1e-8

# Position deduplication: two solutions within this distance are the same drone.
_POS_DEDUP_M = 0.50             # v1 raised from 0.15 m

# Minimum GCC-PHAT strength a candidate needs, relative to the strongest
# candidate in this call, to be accepted as a distinct drone. Without this
# gate, every prior filter here is purely geometric (TDOA self-consistency,
# residual, dedup, min distance) — none of them check whether a peak is
# actually strong evidence of a real source vs. reverberation/multipath
# noise. PHAT-normalized correlation specifically sharpens the true peak
# while also amplifying noise-floor peaks relative to plain GCC, so on a
# real single-drone recording it's common to get 2-3 "geometrically valid"
# but spurious extra peaks — exactly enough to fill out whatever max_drones
# was requested, regardless of how many drones are actually present.
# 0.6 is a starting point, not a validated constant — tune against known
# single- vs multi-drone recordings if it still over- or under-reports.
_MIN_RELATIVE_STRENGTH = 0.6

# Soft barrier strengths
_BARRIER_WEIGHT_IN  = 1e-6      # repel from r < _MIN_SOLUTION_DIST_M
_BARRIER_WEIGHT_OUT = 5e-8      # soft repel from r > max_dist (v2 — no hard clip)


# ══════════════════════════════════════════════════════════════════════════════
# TDOA residual (Cartesian, with barriers)
# ══════════════════════════════════════════════════════════════════════════════

def _tdoa_residual_cartesian(
    xy:       np.ndarray,
    tdoas:    np.ndarray,
    mics:     np.ndarray,
    c:        float,
    cx:       float,
    cy:       float,
    max_dist: float,
) -> float:
    """
    Sum-of-squared TDOA residuals in Cartesian (x, y) coordinates.

    Includes two soft barrier penalties:
    - Inner barrier: repels solver from r < _MIN_SOLUTION_DIST_M
    - Outer barrier: soft penalty for r > max_dist (v2: no hard clip)
    """
    pos = np.asarray(xy, dtype=float)
    d   = np.linalg.norm(mics - pos[None, :], axis=1)
    residual = (
        ((d[1] - d[0]) / c - tdoas[0]) ** 2
        + ((d[2] - d[0]) / c - tdoas[1]) ** 2
        + ((d[2] - d[1]) / c - tdoas[2]) ** 2
    )
    r = float(np.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2))
    if r < _MIN_SOLUTION_DIST_M:
        residual += _BARRIER_WEIGHT_IN / max((r - 0.01) ** 2, 1e-12)
    if r > max_dist:
        over = r - max_dist
        residual += _BARRIER_WEIGHT_OUT * over ** 2
    return float(residual)


# ══════════════════════════════════════════════════════════════════════════════
# Nelder-Mead localizer (v2 Cartesian solver)
# ══════════════════════════════════════════════════════════════════════════════

_GRID_RADII = [0.5, 1.0, 2.0, 3.5, 6.0, 10.0, 15.0, 20.0]


def _nelder_mead_localize(
    tdoas:    np.ndarray,
    mics:     np.ndarray,
    c:        float,
    max_dist: float,
) -> tuple:
    """
    Two-phase localizer:
    1. Coarse grid search to find a good basin.
    2. Unconstrained Nelder-Mead in Cartesian (x, y) with soft barriers.

    Returns (pos_xy: np.ndarray, residual: float).
    """
    cx, cy    = mics.mean(axis=0)
    best_e    = float("inf")
    best_xy   = np.array([cx + 2.0, cy], dtype=float)

    for r in _GRID_RADII:
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            xy = np.array([cx + r * np.cos(a), cy + r * np.sin(a)], dtype=float)
            e  = _tdoa_residual_cartesian(xy, tdoas, mics, c, cx, cy, max_dist)
            if e < best_e:
                best_e, best_xy = e, xy.copy()

    res = scipy.optimize.minimize(
        lambda xy: _tdoa_residual_cartesian(xy, tdoas, mics, c, cx, cy, max_dist),
        x0      = best_xy,
        method  = "Nelder-Mead",
        options = {"xatol": 1e-7, "fatol": 1e-16, "maxiter": 30000, "adaptive": True},
    )
    pos = np.array(res.x, dtype=np.float32)
    err = _tdoa_residual_cartesian(pos, tdoas, mics, c, cx, cy, max_dist)
    return pos, err


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def localize_multi_drone(
    channels:   List[np.ndarray],
    cfg:        Optional[Config] = None,
    max_drones: Optional[int]    = None,
) -> List[dict]:
    """
    Localise up to max_drones drones from a 3-channel microphone array.

    Algorithm
    ─────────
    1. Bandpass filter all channels (200–5000 Hz).
    2. Compute GCC-PHAT peaks for each mic pair to get TDOA candidates.
    3. Enumerate TDOA triplets satisfying the self-consistency constraint.
    4. For each candidate: run Cartesian Nelder-Mead to find (x, y).
    5. Filter degenerate / duplicate / high-residual solutions.
    6. Compute a Hessian-based confidence radius at each solution.

    Parameters
    ──────────
    channels   : list of 3 float32 arrays (one per mic)
    max_drones : maximum drones to report (defaults to cfg.MAX_DRONES)

    Returns
    ───────
    List of dicts, each containing:
        xy_position, azimuth_deg, distance_m,
        tdoa_residual, confidence_radius
    """
    cfg        = cfg or config
    max_drones = max_drones or cfg.MAX_DRONES
    mics       = cfg.MIC_POSITIONS
    c          = cfg.SPEED_OF_SOUND
    sr         = cfg.SR
    cx, cy     = cfg.ARRAY_CENTER

    # Maximum physical TDOA for this array
    max_tau = (
        np.max([
            np.linalg.norm(mics[i] - mics[j])
            for i in range(3)
            for j in range(i + 1, 3)
        ]) / c * 1.5
    )

    chs_bp   = [bandpass(ch, sr, 200, 5000) for ch in channels]
    n_peaks  = max_drones + 2   # ask for extra; dedup handles the rest
    peaks12  = gcc_phat_peaks(chs_bp[1], chs_bp[0], sr, max_tau, n_peaks)
    peaks13  = gcc_phat_peaks(chs_bp[2], chs_bp[0], sr, max_tau, n_peaks)
    peaks23  = gcc_phat_peaks(chs_bp[2], chs_bp[1], sr, max_tau, n_peaks)

    # Build TDOA candidate set
    DEDUP      = _TDOA_DEDUP_S
    candidates = []
    for tau12, s12 in peaks12:
        for tau13, s13 in peaks13:
            tau23_pred = tau13 - tau12
            best23     = min(peaks23, key=lambda x: abs(x[0] - tau23_pred))
            consistency = abs(best23[0] - tau23_pred)
            if consistency < DEDUP * 50:     # TDOA self-consistency gate
                strength = s12 + s13 + best23[1]
                candidates.append((tau12, tau13, tau13 - tau12, strength, consistency))

    candidates.sort(key=lambda x: -x[3])    # highest GCC strength first
    min_strength = candidates[0][3] * _MIN_RELATIVE_STRENGTH if candidates else 0.0

    drones:   List[dict]          = []
    seen_pos: List[np.ndarray]    = []
    seen_tds: List[tuple]         = []

    for tau12, tau13, tau23, strength, consistency in candidates:
        if len(drones) >= max_drones:
            break

        # Candidates are sorted strongest-first, so once one falls below the
        # relative-strength gate, every remaining candidate does too.
        if strength < min_strength:
            break

        # Coarse TDOA deduplication
        if any(
            abs(tau12 - st[0]) < DEDUP and abs(tau13 - st[1]) < DEDUP
            for st in seen_tds
        ):
            continue

        tdoas = np.array([tau12, tau13, tau23])
        pos, err = _nelder_mead_localize(tdoas, mics, c, cfg.MAX_LOCALIZATION_DIST)
        dist_from_centre = float(np.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2))

        # Filter bad solutions
        if dist_from_centre < _MIN_SOLUTION_DIST_M:
            continue
        if err > _RESIDUAL_THRESHOLD:
            continue
        if any(np.linalg.norm(pos - sp) < _POS_DEDUP_M for sp in seen_pos):
            continue

        # Hessian-based confidence radius (finite differences at Cartesian optimum)
        try:
            eps  = max(dist_from_centre * 0.05, 0.10)
            e00  = _tdoa_residual_cartesian(pos,           tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_xp = _tdoa_residual_cartesian(pos + [eps, 0], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_xm = _tdoa_residual_cartesian(pos - [eps, 0], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_yp = _tdoa_residual_cartesian(pos + [0, eps], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            e_ym = _tdoa_residual_cartesian(pos - [0, eps], tdoas, mics, c, cx, cy, cfg.MAX_LOCALIZATION_DIST)
            hxx  = max((e_xp + e_xm - 2 * e00) / eps ** 2, 1e-10)
            hyy  = max((e_yp + e_ym - 2 * e00) / eps ** 2, 1e-10)
            cr   = float(np.clip(
                np.sqrt(0.5 / hxx + 0.5 / hyy), 0.05, cfg.MAX_LOCALIZATION_DIST
            ))
        except Exception:
            cr = float("nan")

        az_deg = math.degrees(math.atan2(pos[1] - cy, pos[0] - cx))

        drones.append({
            "xy_position":       pos,
            "azimuth_deg":       az_deg,
            "distance_m":        dist_from_centre,
            "tdoa_residual":     err,
            "confidence_radius": cr,
        })
        seen_pos.append(pos.copy())
        seen_tds.append((tau12, tau13))

    return drones