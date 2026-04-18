# -*- coding: utf-8 -*-
"""
synthetic_dataset_generator.py
───────────────────────────────
Generate labelled synthetic test datasets as ZIP files that can be loaded
directly by inference_test_loader.load_test_dataset_zip().

Public API
──────────
generate_single_drone_dataset   one drone per session, random positions
generate_multi_drone_dataset    2-3 drones per session (multi-drone test)
generate_hover_grid_dataset     systematic az x dist x height x drone grid
generate_scenario_dataset       named real-world scenarios
generate_all_test_suites        convenience: all four suites in one call
describe_test_zip               print summary of a generated ZIP

ZIP layout (generic_triplet format, auto-detected by the loader)
────────────────────────────────────────────────────────────────
<session_id>_ch0.wav
<session_id>_ch1.wav
<session_id>_ch2.wav
<session_id>_label.json  {azimuth_deg, distance_m, height_m, n_drones,
                          drone_types, scenario, drones:[...], ...}
labels.csv               all sessions in one table for quick analysis

Usage
─────
from drone_detection.synthetic_dataset_generator import (
    generate_single_drone_dataset,
    generate_multi_drone_dataset,
    generate_hover_grid_dataset,
    generate_scenario_dataset,
    generate_all_test_suites,
    describe_test_zip,
)

# Single-drone test set — 30 sessions
generate_single_drone_dataset(config, "/content/test_single.zip", n_sessions=30)

# Multi-drone stress test — 20 sessions with 2-3 drones each
generate_multi_drone_dataset(config, "/content/test_multi.zip", n_sessions=20)

# Full systematic grid (az x dist x height x drone_type)
generate_hover_grid_dataset(config, "/content/test_grid.zip")

# Specific acoustic scenarios
generate_scenario_dataset(config, "/content/test_scenarios.zip",
    scenarios=["indoor_hover", "outdoor_fly", "low_snr",
               "multi_drone_2", "multi_drone_3"])

# All four suites in one call, then evaluate each
paths = generate_all_test_suites(config, output_dir="/content/suites/")
for name, path in paths.items():
    test_ds = load_test_dataset_zip(path, config)
    results = run_test_dataset_evaluation(test_ds, config,
                  save_csv=f"/content/{name}_results.csv")
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


# ── lazy imports ──────────────────────────────────────────────────────────────

def _cfg(cfg=None):
    if cfg is not None:
        return cfg
    from .config import config
    return config

def _ap(cfg):
    from .audio_processing import AudioProcessor
    return AudioProcessor(cfg)

def _synth(mic_positions, src_xy, **kw):
    from .audio_processing import synthesise_drone
    return synthesise_drone(mic_positions, src_xy, **kw)

def _wrap(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


# ── Scenario catalogue ────────────────────────────────────────────────────────

SCENARIO_SPECS: Dict[str, Dict] = {
    "indoor_hover": {
        "desc":         "Indoor studio, low altitude 1-4 m",
        "n_sessions":   20, "n_drones": 1,
        "array":        "uavirbase",
        "noise_profile":"indoor", "noise_range": (0.01, 0.03),
        "dist_range":   (1.0,  8.0), "height_range": (1.0,  4.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini"],
    },
    "indoor_moving": {
        "desc":         "Indoor, larger distances and more noise",
        "n_sessions":   20, "n_drones": 1,
        "array":        "uavirbase",
        "noise_profile":"indoor", "noise_range": (0.03, 0.06),
        "dist_range":   (2.0, 15.0), "height_range": (1.0,  8.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro"],
    },
    "outdoor_fly": {
        "desc":         "Outdoor flight, moderate wind",
        "n_sessions":   20, "n_drones": 1,
        "array":        "uavirbase",
        "noise_profile":"outdoor", "noise_range": (0.06, 0.10),
        "dist_range":   (5.0, 25.0), "height_range": (5.0, 20.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"],
    },
    "outdoor_far": {
        "desc":         "Outdoor far-field, high wind, low SNR",
        "n_sessions":   15, "n_drones": 1,
        "array":        "uavirbase",
        "noise_profile":"outdoor", "noise_range": (0.10, 0.18),
        "dist_range":  (15.0, 25.0), "height_range": (10.0, 20.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro"],
    },
    "low_snr": {
        "desc":         "Challenging low-SNR conditions",
        "n_sessions":   15, "n_drones": 1,
        "array":        "uavirbase",
        "noise_profile":"outdoor", "noise_range": (0.12, 0.20),
        "dist_range":  (10.0, 25.0), "height_range": (5.0, 20.0),
        "drone_types":  ["mavic_mini", "generic_quad"],
    },
    "multi_drone_2": {
        "desc":         "Two drones simultaneously",
        "n_sessions":   20, "n_drones": 2,
        "array":        "uavirbase",
        "noise_profile":"mixed", "noise_range": (0.03, 0.07),
        "dist_range":   (2.0, 15.0), "height_range": (1.0, 10.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini"],
        "min_sep_deg":  45.0,
    },
    "multi_drone_3": {
        "desc":         "Three drones simultaneously (stress test)",
        "n_sessions":   15, "n_drones": 3,
        "array":        "uavirbase",
        "noise_profile":"mixed", "noise_range": (0.04, 0.08),
        "dist_range":   (2.0, 12.0), "height_range": (1.0,  8.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"],
        "min_sep_deg":  40.0,
    },
    "gp1_array": {
        "desc":         "PannoniaFS GP1 array, 2165 mm baseline",
        "n_sessions":   20, "n_drones": 1,
        "array":        "gp1",
        "noise_profile":"indoor", "noise_range": (0.02, 0.05),
        "dist_range":   (1.0, 10.0), "height_range": (1.0, 5.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini"],
    },
    "gp2_array": {
        "desc":         "PannoniaFS GP2 array, 2500 mm baseline",
        "n_sessions":   20, "n_drones": 1,
        "array":        "gp2",
        "noise_profile":"indoor", "noise_range": (0.02, 0.05),
        "dist_range":   (1.0, 10.0), "height_range": (1.0, 5.0),
        "drone_types":  ["mavic_pro", "mavic_2_pro", "mavic_mini"],
    },
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def _random_pos(cfg, rng, az_deg=None, dist_m=None, height_m=None,
                min_dist=1.0, max_dist=None, min_ht=0.5, max_ht=20.0):
    if max_dist is None:
        max_dist = cfg.MAX_LOCALIZATION_DIST
    cx, cy = cfg.ARRAY_CENTER
    if az_deg is None:
        az_deg = float(rng.uniform(-180.0, 180.0))
    if dist_m is None:
        dist_m = float(np.exp(rng.uniform(np.log(max(min_dist, 0.1)),
                                           np.log(max_dist))))
    if height_m is None:
        height_m = float(rng.uniform(min_ht, max_ht))
    r   = math.radians(az_deg)
    src = np.array([cx + dist_m * math.cos(r),
                    cy + dist_m * math.sin(r)], dtype=np.float32)
    return src, _wrap(az_deg), float(dist_m), float(height_m)


def _session(cfg, ap, mics, drone_positions, noise_level, noise_profile,
             out_dir, session_id):
    """Synthesise one session: mix all drones, write 3 mono WAVs."""
    sr = cfg.SR
    n  = int(sr * cfg.TARGET_DURATION)
    mix = [np.zeros(n, dtype=np.float32) for _ in range(len(mics))]
    for src_xy, dtype in drone_positions:
        chs = _synth(mics, src_xy, noise_level=noise_level,
                     drone_type=dtype, noise_profile=noise_profile, cfg=cfg)
        for i, ch in enumerate(chs):
            mix[i] = np.clip(mix[i] + ap.pad_or_truncate(ch), -1.0, 1.0).astype(np.float32)
    peak = max(float(np.max(np.abs(c))) for c in mix) + 1e-8
    if peak > 0.9:
        mix = [(c / peak * 0.9).astype(np.float32) for c in mix]
    paths = []
    for i, ch in enumerate(mix):
        p = str(out_dir / f"{session_id}_ch{i}.wav")
        sf.write(p, ch, sr)
        paths.append(p)
    return paths


def _label(out_dir, sid, az, dist, ht, n_drones, drone_types,
           scenario, extra=None):
    data = {"azimuth_deg": round(az, 4), "distance_m": round(dist, 4),
            "height_m": round(ht, 4), "n_drones": n_drones,
            "drone_types": drone_types, "scenario": scenario}
    if extra:
        data.update(extra)
    (out_dir / f"{sid}_label.json").write_text(json.dumps(data, indent=2))


def _pack(src_dir, zip_path, csv_rows):
    if csv_rows:
        with open(src_dir / "labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
    zip_path = str(zip_path)
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(src_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(src_dir))
    mb = os.path.getsize(zip_path) / 1e6
    print(f"  Wrote: {zip_path}  ({mb:.1f} MB, {len(csv_rows)} sessions)")
    return zip_path


def _angular_sep(a1, a2):
    d = abs(_wrap(a1 - a2))
    return min(d, 360 - d)


# ══════════════════════════════════════════════════════════════════════════════
# Generator 1 — single drone
# ══════════════════════════════════════════════════════════════════════════════

def generate_single_drone_dataset(
    cfg=None,
    output_zip: str = "test_single_drone.zip",
    n_sessions: int = 50,
    drone_types: Optional[List[str]] = None,
    array_name: str = "uavirbase",
    noise_profile: str = "mixed",
    noise_level_range: Tuple[float, float] = (0.02, 0.10),
    dist_range_m: Tuple[float, float] = (1.0, 20.0),
    height_range_m: Tuple[float, float] = (0.5, 15.0),
    seed: int = 42,
) -> str:
    """
    Generate a single-drone test ZIP with random positions.

    Each session has 3 WAV files + label JSON.
    The ZIP is in generic_triplet format, loaded automatically by
    load_test_dataset_zip().

    Parameters
    ──────────
    n_sessions          : number of sessions
    drone_types         : types to sample from; default = 4 real-world DJI types
    array_name          : "uavirbase" | "gp1" | "gp2"
    noise_profile       : "indoor" | "outdoor" | "mixed"
    noise_level_range   : (min, max) noise amplitude
    dist_range_m        : (min, max) source distance from array centre
    height_range_m      : (min, max) source height
    """
    cfg = _cfg(cfg)
    cfg.set_array_geometry(array_name)
    ap  = _ap(cfg)
    rng = np.random.default_rng(seed)

    if drone_types is None:
        drone_types = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]

    tmp = Path(tempfile.mkdtemp(prefix="synth_single_"))
    rows: List[Dict] = []

    print(f"🔬 Single-drone dataset: {n_sessions} sessions  "
          f"(array={array_name}, noise={noise_profile})")

    for i in tqdm(range(n_sessions)):
        sid   = f"single_{i:04d}"
        dtype = str(rng.choice(drone_types))
        nl    = float(rng.uniform(*noise_level_range))
        npf   = noise_profile if noise_profile != "mixed" else str(
            rng.choice(["indoor", "outdoor"]))
        src, az, dist, ht = _random_pos(
            cfg, rng,
            min_dist=dist_range_m[0], max_dist=dist_range_m[1],
            min_ht=height_range_m[0], max_ht=height_range_m[1])

        _session(cfg, ap, cfg.MIC_POSITIONS, [(src, dtype)],
                 nl, npf, tmp, sid)
        _label(tmp, sid, az, dist, ht, 1, [dtype], "single_drone",
               {"noise_level": round(nl, 4), "noise_profile": npf, "array": array_name})
        rows.append({"session_id": sid, "azimuth_deg": round(az, 2),
                     "distance_m": round(dist, 2), "height_m": round(ht, 2),
                     "n_drones": 1, "drone_type": dtype,
                     "noise_level": round(nl, 4), "noise_profile": npf,
                     "array": array_name})

    result = _pack(tmp, output_zip, rows)
    shutil.rmtree(tmp)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Generator 2 — multi-drone
# ══════════════════════════════════════════════════════════════════════════════

def generate_multi_drone_dataset(
    cfg=None,
    output_zip: str = "test_multi_drone.zip",
    n_sessions: int = 30,
    n_drones_range: Tuple[int, int] = (2, 3),
    drone_types: Optional[List[str]] = None,
    array_name: str = "uavirbase",
    noise_profile: str = "mixed",
    noise_level_range: Tuple[float, float] = (0.03, 0.08),
    dist_range_m: Tuple[float, float] = (1.0, 15.0),
    height_range_m: Tuple[float, float] = (1.0, 10.0),
    min_separation_deg: float = 30.0,
    seed: int = 123,
) -> str:
    """
    Generate a multi-drone test ZIP with 2-3 drones per session.

    Label JSON records:
      azimuth_deg/distance_m/height_m  — primary drone (closest)
      drones                           — list of all drone positions
      n_drones                         — total drone count

    The 3-WAV input is suitable for both single-drone localize() and
    multi-drone localize_multi_drone() evaluation.

    Parameters
    ──────────
    n_drones_range       : (min, max) drones per session (inclusive)
    min_separation_deg   : minimum angular separation between drones
    """
    cfg = _cfg(cfg)
    cfg.set_array_geometry(array_name)
    ap  = _ap(cfg)
    rng = np.random.default_rng(seed)

    if drone_types is None:
        drone_types = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]

    tmp = Path(tempfile.mkdtemp(prefix="synth_multi_"))
    rows: List[Dict] = []

    print(f"🔬 Multi-drone dataset: {n_sessions} sessions  "
          f"({n_drones_range[0]}-{n_drones_range[1]} drones, array={array_name})")

    for i in tqdm(range(n_sessions)):
        sid  = f"multi_{i:04d}"
        n_d  = int(rng.integers(n_drones_range[0], n_drones_range[1] + 1))
        nl   = float(rng.uniform(*noise_level_range))
        npf  = noise_profile if noise_profile != "mixed" else str(
            rng.choice(["indoor", "outdoor"]))

        # Sample positions with angular separation constraint
        drone_list = []   # [(src_xy, az, dist, ht, dtype)]
        for _ in range(n_d):
            for _ in range(200):
                src, az, dist, ht = _random_pos(
                    cfg, rng,
                    min_dist=dist_range_m[0], max_dist=dist_range_m[1],
                    min_ht=height_range_m[0], max_ht=height_range_m[1])
                if all(_angular_sep(az, d[1]) >= min_separation_deg
                       for d in drone_list):
                    dtype = str(rng.choice(drone_types))
                    drone_list.append((src, az, dist, ht, dtype))
                    break

        if len(drone_list) < n_drones_range[0]:
            continue  # could not satisfy separation — skip

        drone_positions = [(d[0], d[4]) for d in drone_list]
        _session(cfg, ap, cfg.MIC_POSITIONS, drone_positions, nl, npf, tmp, sid)

        primary   = min(drone_list, key=lambda d: d[2])
        _, az_p, dist_p, ht_p, dtype_p = primary
        all_d = [{"azimuth_deg": round(d[1], 2), "distance_m": round(d[2], 2),
                  "height_m": round(d[3], 2), "drone_type": d[4]}
                 for d in drone_list]

        _label(tmp, sid, az_p, dist_p, ht_p, len(drone_list),
               [d[4] for d in drone_list],
               f"multi_drone_{len(drone_list)}",
               {"drones": all_d, "noise_level": round(nl, 4),
                "noise_profile": npf, "array": array_name})
        rows.append({"session_id": sid, "azimuth_deg": round(az_p, 2),
                     "distance_m": round(dist_p, 2), "height_m": round(ht_p, 2),
                     "n_drones": len(drone_list),
                     "drone_types": "|".join(d[4] for d in drone_list),
                     "noise_level": round(nl, 4), "noise_profile": npf,
                     "array": array_name})

    result = _pack(tmp, output_zip, rows)
    shutil.rmtree(tmp)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Generator 3 — systematic hover grid
# ══════════════════════════════════════════════════════════════════════════════

def generate_hover_grid_dataset(
    cfg=None,
    output_zip: str = "test_hover_grid.zip",
    azimuths_deg: Optional[List[float]] = None,
    distances_m: Optional[List[float]] = None,
    heights_m: Optional[List[float]] = None,
    drone_types: Optional[List[str]] = None,
    array_name: str = "uavirbase",
    noise_profile: str = "indoor",
    noise_level: float = 0.03,
    n_repeats: int = 1,
    seed: int = 7,
) -> str:
    """
    Generate a systematic grid covering every combination of
    azimuth x distance x height x drone_type.

    This matches the UaVirBASE measurement protocol and is the best
    dataset for per-position accuracy breakdown in the thesis.

    Defaults produce 8 x 4 x 4 x 4 x 1 = 512 sessions.
    With n_repeats=2: 1024 sessions for variance estimation.

    Parameters
    ──────────
    azimuths_deg  : default = 8 cardinal directions
    distances_m   : default = [2, 5, 10, 20] m
    heights_m     : default = [1, 2, 4, 8] m
    drone_types   : default = all 4 real-world types
    n_repeats     : repeats per grid point (noise varies per repeat)
    """
    cfg = _cfg(cfg)
    cfg.set_array_geometry(array_name)
    ap  = _ap(cfg)
    rng = np.random.default_rng(seed)

    if azimuths_deg is None:
        azimuths_deg = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]
    if distances_m  is None:
        distances_m  = [2.0, 5.0, 10.0, 20.0]
    if heights_m    is None:
        heights_m    = [1.0, 2.0, 4.0, 8.0]
    if drone_types  is None:
        drone_types  = ["mavic_pro", "mavic_2_pro", "mavic_mini", "generic_quad"]

    total = (len(azimuths_deg) * len(distances_m) * len(heights_m)
             * len(drone_types) * n_repeats)
    tmp  = Path(tempfile.mkdtemp(prefix="synth_grid_"))
    rows: List[Dict] = []
    cx, cy = cfg.ARRAY_CENTER

    print(f"🔬 Hover grid: {total} sessions  "
          f"({len(azimuths_deg)} az x {len(distances_m)} dist x "
          f"{len(heights_m)} ht x {len(drone_types)} types x {n_repeats} rep)")

    with tqdm(total=total) as pbar:
        for az in azimuths_deg:
            for dist in distances_m:
                for ht in heights_m:
                    for dtype in drone_types:
                        for rep in range(n_repeats):
                            sid = (f"grid"
                                   f"_az{int(az):+04d}"
                                   f"_d{int(dist):02d}m"
                                   f"_h{int(ht):02d}m"
                                   f"_{dtype[:4]}_r{rep}")
                            az_r = math.radians(az)
                            src  = np.array(
                                [cx + dist * math.cos(az_r),
                                 cy + dist * math.sin(az_r)],
                                dtype=np.float32)
                            nl = noise_level * float(rng.uniform(0.8, 1.2))
                            _session(cfg, ap, cfg.MIC_POSITIONS, [(src, dtype)],
                                     nl, noise_profile, tmp, sid)
                            _label(tmp, sid, _wrap(az), dist, ht,
                                   1, [dtype], "hover_grid",
                                   {"repeat": rep, "array": array_name,
                                    "noise_level": round(nl, 4)})
                            rows.append({
                                "session_id": sid,
                                "azimuth_deg": round(_wrap(az), 2),
                                "distance_m": dist, "height_m": ht,
                                "n_drones": 1, "drone_type": dtype,
                                "repeat": rep,
                                "noise_level": round(nl, 4),
                                "array": array_name,
                            })
                            pbar.update(1)

    result = _pack(tmp, output_zip, rows)
    shutil.rmtree(tmp)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Generator 4 — named scenarios
# ══════════════════════════════════════════════════════════════════════════════

def generate_scenario_dataset(
    cfg=None,
    output_zip: str = "test_scenarios.zip",
    scenarios: Optional[List[str]] = None,
    array_name: str = "uavirbase",
    seed: int = 999,
) -> str:
    """
    Generate a dataset covering named acoustic scenarios.

    Available scenarios
    ───────────────────
    indoor_hover    quiet studio, 1-4 m altitude
    indoor_moving   indoor flight, higher noise
    outdoor_fly     outdoor moderate wind
    outdoor_far     outdoor far-field, low SNR
    low_snr         very challenging
    multi_drone_2   two drones simultaneously
    multi_drone_3   three drones (stress test)
    gp1_array       PannoniaFS GP1 geometry
    gp2_array       PannoniaFS GP2 geometry

    Each session label includes a 'scenario' field so you can slice
    the evaluation results by scenario type in post-processing:

        import pandas as pd
        df = pd.read_csv("results.csv")
        print(df.groupby("scenario")["az_err_deg"].mean())
    """
    cfg = _cfg(cfg)
    ap  = _ap(cfg)
    rng = np.random.default_rng(seed)

    if scenarios is None:
        scenarios = list(SCENARIO_SPECS.keys())
    unknown = [s for s in scenarios if s not in SCENARIO_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown scenarios: {unknown}\n"
            f"Available: {list(SCENARIO_SPECS.keys())}"
        )

    tmp  = Path(tempfile.mkdtemp(prefix="synth_scenarios_"))
    rows: List[Dict] = []
    total = sum(SCENARIO_SPECS[s]["n_sessions"] for s in scenarios)
    print(f"🔬 Scenario dataset: {len(scenarios)} scenarios, {total} sessions")

    for sc_name in scenarios:
        spec  = SCENARIO_SPECS[sc_name]
        n_d   = spec["n_drones"]
        arr   = spec.get("array", array_name)
        npf   = spec["noise_profile"]
        nlr   = spec["noise_range"]
        dr    = spec["dist_range"]
        hr    = spec["height_range"]
        dts   = spec["drone_types"]
        sep   = spec.get("min_sep_deg", 30.0)
        cfg.set_array_geometry(arr)

        print(f"  {sc_name:20s} — {spec['desc']}")
        for i in tqdm(range(spec["n_sessions"]), desc=f"  {sc_name}", leave=False):
            sid  = f"{sc_name}_{i:04d}"
            nl   = float(rng.uniform(*nlr))
            npf2 = npf if npf != "mixed" else str(rng.choice(["indoor","outdoor"]))

            drone_list = []
            for _ in range(n_d):
                for _ in range(200):
                    src, az, dist, ht = _random_pos(
                        cfg, rng, min_dist=dr[0], max_dist=dr[1],
                        min_ht=hr[0], max_ht=hr[1])
                    if all(_angular_sep(az, d[1]) >= sep for d in drone_list):
                        drone_list.append((src, az, dist, ht, str(rng.choice(dts))))
                        break
            if not drone_list:
                continue

            drone_positions = [(d[0], d[4]) for d in drone_list]
            _session(cfg, ap, cfg.MIC_POSITIONS, drone_positions, nl, npf2, tmp, sid)

            primary = min(drone_list, key=lambda d: d[2])
            _, az_p, dist_p, ht_p, dtype_p = primary
            all_d = [{"azimuth_deg": round(d[1], 2), "distance_m": round(d[2], 2),
                      "height_m": round(d[3], 2), "drone_type": d[4]}
                     for d in drone_list]

            _label(tmp, sid, az_p, dist_p, ht_p, len(drone_list),
                   [d[4] for d in drone_list], sc_name,
                   {"drones": all_d, "noise_level": round(nl, 4),
                    "noise_profile": npf2, "array": arr,
                    "description": spec["desc"]})
            rows.append({"session_id": sid, "scenario": sc_name,
                         "azimuth_deg": round(az_p, 2),
                         "distance_m": round(dist_p, 2),
                         "height_m": round(ht_p, 2),
                         "n_drones": len(drone_list),
                         "drone_types": "|".join(d[4] for d in drone_list),
                         "noise_level": round(nl, 4),
                         "noise_profile": npf2, "array": arr})

    result = _pack(tmp, output_zip, rows)
    shutil.rmtree(tmp)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Generator 5 — all suites in one call
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_test_suites(
    cfg=None,
    output_dir: str = ".",
    prefix: str = "test",
    array_name: str = "uavirbase",
    seed: int = 42,
    single_n: int = 50,
    multi_n: int = 30,
    grid_repeats: int = 1,
    scenarios: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate all four test suites and return {name: zip_path}.

    Suites
    ──────
    single      random single-drone positions
    multi       2-3 drone sessions
    grid        systematic az x dist x height x drone_type grid
    scenarios   named acoustic scenario collection

    Example
    ───────
    paths = generate_all_test_suites(config, "/content/suites/")

    from drone_detection.inference_test_loader import (
        load_test_dataset_zip, run_test_dataset_evaluation)

    for name, path in paths.items():
        ds = load_test_dataset_zip(path, config)
        run_test_dataset_evaluation(ds, config,
            save_csv=f"/content/{name}_results.csv")
    """
    cfg = _cfg(cfg)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    print("=" * 60)
    print("  Generating all test suites")
    print("=" * 60)

    paths["single"] = generate_single_drone_dataset(
        cfg, str(out / f"{prefix}_single.zip"),
        n_sessions=single_n, array_name=array_name, seed=seed)

    paths["multi"] = generate_multi_drone_dataset(
        cfg, str(out / f"{prefix}_multi.zip"),
        n_sessions=multi_n, array_name=array_name, seed=seed + 1)

    paths["grid"] = generate_hover_grid_dataset(
        cfg, str(out / f"{prefix}_grid.zip"),
        array_name=array_name, n_repeats=grid_repeats, seed=seed + 2)

    paths["scenarios"] = generate_scenario_dataset(
        cfg, str(out / f"{prefix}_scenarios.zip"),
        scenarios=scenarios, array_name=array_name, seed=seed + 3)

    print("\n" + "=" * 60)
    print("  All suites complete")
    for name, path in paths.items():
        mb = os.path.getsize(path) / 1e6
        print(f"  {name:12s}: {path}  ({mb:.1f} MB)")
    print("=" * 60)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic — describe a generated ZIP without loading audio
# ══════════════════════════════════════════════════════════════════════════════

def describe_test_zip(zip_path: str) -> Dict:
    """
    Print a human-readable summary of a test ZIP.
    Reads labels.csv only — does not load any WAV files.

    Useful for quickly checking what was generated before running evaluation.
    """
    zip_path = str(zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP: {zip_path}")

    rows: List[Dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "labels.csv" in names:
            import io
            with zf.open("labels.csv") as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f)))

    n_wav      = sum(1 for n in names if n.endswith(".wav"))
    n_sessions = n_wav // 3

    print(f"\n📦 {Path(zip_path).name}")
    print(f"   Total files  : {len(names)}")
    print(f"   Sessions     : {n_sessions}  ({n_wav} WAV files, 3 per session)")
    if rows:
        print(f"   Labelled     : {len(rows)}")
        if "scenario" in rows[0]:
            sc = Counter(r["scenario"] for r in rows)
            print("   Scenarios    :")
            for s, c in sorted(sc.items()):
                print(f"     {s:28s}: {c}")
        if "n_drones" in rows[0]:
            nd = Counter(int(r["n_drones"]) for r in rows)
            print(f"   Drone counts : {dict(sorted(nd.items()))}")
        if "azimuth_deg" in rows[0]:
            azs  = [float(r["azimuth_deg"]) for r in rows]
            dsts = [float(r["distance_m"])  for r in rows]
            hts  = [float(r["height_m"])    for r in rows]
            print(f"   Azimuth      : {min(azs):.1f}° … {max(azs):.1f}°  "
                  f"(mean {np.mean(azs):.1f}°)")
            print(f"   Distance     : {min(dsts):.1f} … {max(dsts):.1f} m  "
                  f"(mean {np.mean(dsts):.1f} m)")
            print(f"   Height       : {min(hts):.1f} … {max(hts):.1f} m  "
                  f"(mean {np.mean(hts):.1f} m)")
        if "array" in rows[0]:
            print(f"   Arrays       : {dict(Counter(r['array'] for r in rows))}")
        if "drone_type" in rows[0] or "drone_types" in rows[0]:
            key = "drone_type" if "drone_type" in rows[0] else "drone_types"
            dt  = Counter(r[key].split("|")[0] for r in rows)
            print(f"   Primary drone: {dict(sorted(dt.items()))}")

    return {"zip_path": zip_path, "n_entries": len(names),
            "n_sessions": n_sessions, "n_labelled": len(rows), "rows": rows}