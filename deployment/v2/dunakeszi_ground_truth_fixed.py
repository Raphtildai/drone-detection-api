# -*- coding: utf-8 -*-
"""
dunakeszi_ground_truth.py
─────────────────────────
Builds the master ground-truth tables for the Dunakeszi 2025-10-20 dataset.

Outputs (written to --out_dir):
  ground_truth_sessions.json   — one record per show/session
  ground_truth_segments.json   — one record per maneuver segment
  ground_truth_segments.csv    — same, flat CSV for quick inspection
  mic_array_geometry.json      — surveyed GPS + local XY positions of all arrays
  drone_registry.json          — drone IDs and which shows they appear in
  file_index.json              — reverse lookup: filename → segment ID list
  scorpio_channel_map.json     — Scorpio/Pix270i channel→microphone serial mapping

These files are the authoritative index used by the extractor when it later
processes any individual audio or GPX file.  No audio is read here — this is
purely metadata/ground-truth construction.

Sources used (in priority order)
─────────────────────────────────
  1. notes_2025-10-20_16-33-21_volk_janos.md   — wall-clock triggers, drone IDs,
                                                  log file names, mic GPS positions
  2. Jegyzőkönyv PDF (protocol)                 — maneuver table, durations, channel map
  3. Multidrónos akusztikus mérés PDF           — planning sheet, show→maneuver mapping
  4. File-structure screenshots                  — actual filenames on the NAS

Timing model
────────────
  Wall-clock times are LOCAL (CEST = UTC+2).
  Drone log timestamps are UTC → add UTC_TO_LOCAL_S = 7200 to convert to CEST.

  Recording reference: show_1 triggered at local 13:36:00.
  SMPTE ToD in the PolyWav/MOV starts at 00:00:00 (midnight) = Time of Day mode.
  → local_start_s   = HH*3600 + MM*60  (seconds since midnight, local CEST)
  → smpte_start_s   = local_start_s    (SMPTE ToD == local wall-clock)
  → onset_from_rec_s = local_start_s - RECORDING_REF_LOCAL_S
                       where RECORDING_REF_LOCAL_S = 13*3600+36*60 = 48 960 s

Coordinate system
─────────────────
  Local XY (metres) centred on the measurement origin (0,0,0).
  X = East, Y = North, Z = Up.
  Origin GPS: midpoint of BK-6-E and BK-6-W mic arrays.

PolyWav files
─────────────
  17 files: 251020VITEMOROM1AT01.wav (no letter, ~first chunk)
            251020VITEMOROM1AT01A.wav … 251020VITEMOROM1AT01P.wav
  Each is exactly 4 GB, 192 kHz, 14 channels, 32-bit float.
  4 GB / (14ch × 4bytes × 192 000 Hz) ≈ 399.5 s ≈ 6 min 39 s per chunk.
  All chunks are sample-accurately joinable (continuous recording, >3 hours).

MEMS files
──────────
  12 files: Audio 01_02.wav, Audio 02_01.wav … Audio 11_01.wav, Audio 12_02.wav
  (_02 suffix on file 01 and 12 indicates a retake; all others are _01.)
  Each is exactly 133.1 MB.  Format MUST be verified from the WAV header —
  call verify_mems_format(path_to_any_mems_file) once you have local access.
  Best guess: 48 kHz, 4 ch, 24-bit integer → 242 s per file → 48.5 min total.
  The MEMS starts at 14:06 local.  Coverage ends ~14:54 (24-bit) or earlier.
  ⚠  This means MEMS only captured show_5 (14:08) and show_7 (14:24) in full,
     show_8 (14:41) partially, and no later shows.  Verify after reading headers.

Video file
──────────
  1-251020VITEMOROM1V-001.mov  (79 GB, DNxHD 36 Mbit/s, 1920×1080 30 FPS)
  Audio: 12 ch, 24-bit, 48 kHz = Scorpio channels 3–14 downsampled.
  SMPTE ToD timecode 24h 30FPS ND, same time base as PolyWav.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Timing constants
# ══════════════════════════════════════════════════════════════════════════════

UTC_TO_LOCAL_S: int = 7200          # CEST = UTC + 2 h

# show_1 wall-clock trigger (Volk notes, authoritative)
RECORDING_REF_LOCAL_S: int = 13 * 3600 + 36 * 60   # 48 960 s


def local_hhmm_to_s(hhmm: str) -> int:
    """'HH:MM' → seconds since midnight (local CEST)."""
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    return h * 3600 + m * 60


def s_to_hhmm(s: int) -> str:
    """Seconds since midnight → 'HH:MM:SS' string."""
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def onset_from_rec(hhmm: str) -> int:
    """Seconds from recording start (show_1 trigger) for a local HH:MM."""
    return local_hhmm_to_s(hhmm) - RECORDING_REF_LOCAL_S


def smpte_s(hhmm: str) -> int:
    """SMPTE ToD seconds = local wall-clock seconds since midnight."""
    return local_hhmm_to_s(hhmm)


# ══════════════════════════════════════════════════════════════════════════════
# Scorpio / Pix270i channel map
# (Source: Jegyzőkönyv page 5 — channel table)
# ══════════════════════════════════════════════════════════════════════════════

# Direction codes: E=Ear(horizontal), H=Head(vertical), B=Back, J=Jaw, F=Front, L=Low
SCORPIO_CHANNEL_MAP: Dict[int, Dict[str, Any]] = {
    1:  {"label": "Mix L",      "array": None,     "direction": None, "serial": None,    "useful": False,
         "note": "Monitor mix left — not a raw mic signal"},
    2:  {"label": "Mix R",      "array": None,     "direction": None, "serial": None,    "useful": False,
         "note": "Monitor mix right — not a raw mic signal"},
    3:  {"label": "W–E",        "array": "BK-6-W", "direction": "E",  "serial": "2105962", "useful": True},
    4:  {"label": "W–H",        "array": "BK-6-W", "direction": "H",  "serial": "2113214", "useful": True},
    5:  {"label": "W–B",        "array": "BK-6-W", "direction": "B",  "serial": "2113215", "useful": True},
    6:  {"label": "W–J",        "array": "BK-6-W", "direction": "J",  "serial": "2113212", "useful": True},
    7:  {"label": "W–F",        "array": "BK-6-W", "direction": "F",  "serial": "2105972", "useful": True},
    8:  {"label": "W–L",        "array": "BK-6-W", "direction": "L",  "serial": "2105965", "useful": True},
    9:  {"label": "E–E",        "array": "BK-6-E", "direction": "E",  "serial": "2113216", "useful": True},
    10: {"label": "E–H",        "array": "BK-6-E", "direction": "H",  "serial": "2113217", "useful": True},
    11: {"label": "E–B",        "array": "BK-6-E", "direction": "B",  "serial": "2113213", "useful": True},
    12: {"label": "E–J",        "array": "BK-6-E", "direction": "J",  "serial": "2105973", "useful": True},
    13: {"label": "E–F",        "array": "BK-6-E", "direction": "F",  "serial": "2105966", "useful": True},
    14: {"label": "E–L",        "array": "BK-6-E", "direction": "L",  "serial": "2105963", "useful": True},
}

# Pix270i (48 kHz video recorder) = Scorpio channels 3–14 in the same order
# (Pix channels 1–12 = Scorpio ch 3–14)
PIX270I_CHANNEL_MAP: Dict[int, Dict[str, Any]] = {
    pix_ch: {**SCORPIO_CHANNEL_MAP[scorpio_ch], "scorpio_ch": scorpio_ch}
    for pix_ch, scorpio_ch in enumerate(range(3, 15), start=1)
}

# Subsets used for TDOA within each 3D array (first 3 channels only per array)
TDOA_CHANNELS = {
    "BK-6-W": [3, 4, 5],   # W-E, W-H, W-B
    "BK-6-E": [9, 10, 11], # E-E, E-H, E-B
}


# ══════════════════════════════════════════════════════════════════════════════
# Microphone array GPS positions
# (Source: Volk notes — surveyed with drone hover)
# ══════════════════════════════════════════════════════════════════════════════

MIC_GPS = {
    "BK-6-E":  {"lat": 47.6086296, "lon": 19.1470983},   # drone 132 hover
    "BK-6-W":  {"lat": 47.6086368, "lon": 19.1468423},   # drone 78 hover
    "MEMS-S":  {"lat": 47.6085469, "lon": 19.1469590},   # drone 132 hover
    "MEMS-N":  {"lat": 47.6087234, "lon": 19.1469795},   # drone 78 hover
}

# Measurement origin = midpoint of the two BK arrays
ORIGIN_LAT = (MIC_GPS["BK-6-E"]["lat"] + MIC_GPS["BK-6-W"]["lat"]) / 2
ORIGIN_LON = (MIC_GPS["BK-6-E"]["lon"] + MIC_GPS["BK-6-W"]["lon"]) / 2

R_EARTH = 6_371_000.0   # metres (WGS-84 mean radius for flat-Earth approx)


def gps_to_xy(lat: float, lon: float) -> Tuple[float, float]:
    """
    GPS → local XY metres relative to ORIGIN.
    Flat-Earth, valid within the ~300 m measurement area.
    X = East, Y = North.
    """
    cos_lat = math.cos(math.radians(ORIGIN_LAT))
    dx = (lon - ORIGIN_LON) * math.pi / 180.0 * R_EARTH * cos_lat
    dy = (lat - ORIGIN_LAT) * math.pi / 180.0 * R_EARTH
    return round(dx, 2), round(dy, 2)


def build_mic_geometry() -> dict:
    arrays = {}
    for name, gps in MIC_GPS.items():
        x, y = gps_to_xy(gps["lat"], gps["lon"])
        arrays[name] = {
            "gps":               gps,
            "local_xy_m":        {"x": x, "y": y},
            "type":              "BK_3D_6ch" if name.startswith("BK") else "MEMS_2D_4ch",
            "scorpio_channels":  TDOA_CHANNELS.get(name),          # TDOA subset
            "all_scorpio_channels": {
                "BK-6-E": [9, 10, 11, 12, 13, 14],
                "BK-6-W": [3,  4,  5,  6,  7,  8],
            }.get(name),
            "config_geometry":   "gp2" if name.startswith("BK") else None,
        }

    bke = arrays["BK-6-E"]["local_xy_m"]
    bkw = arrays["BK-6-W"]["local_xy_m"]
    dist_ew = round(math.hypot(bke["x"] - bkw["x"], bke["y"] - bkw["y"]), 2)
    mems_s = arrays["MEMS-S"]["local_xy_m"]
    mems_n = arrays["MEMS-N"]["local_xy_m"]
    dist_ns = round(math.hypot(mems_s["x"] - mems_n["x"], mems_s["y"] - mems_n["y"]), 2)

    return {
        "origin":    {"lat": ORIGIN_LAT, "lon": ORIGIN_LON},
        "arrays":    arrays,
        "inter_array_distances_m": {
            "BK-6-E_to_BK-6-W": dist_ew,
            "MEMS-S_to_MEMS-N": dist_ns,
        },
        "notes": {
            "survey_method":    "Drone GPS hover at each array centroid",
            "surveyed_by":      "Volk János (drone132 for BK-6-E+MEMS-S, drone78 for BK-6-W+MEMS-N)",
            "coordinate_frame": "Local XY: X=East Y=North Z=Up; origin=midpoint of BK-6-E and BK-6-W",
            "measurement_area_orientation": "NE-SW (előre-hátra); deviates slightly from mic array axis",
            "timezone":         "CEST (UTC+2)",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Drone registry
# (Source: Volk notes — authoritative for which drone flew which show)
# ══════════════════════════════════════════════════════════════════════════════

# Log file IDs per show per drone  (format: YYYYMMDD-NNN)
# ⚠ drone132 log "20251020-002" appears for BOTH show_13 and show_14 in Volk notes —
#   this may mean the same flight log covers both shows, or it is a note error.
#   Flagged with quality flag "log_id_shared_with_show_13".
DRONE_LOGS: Dict[str, Dict[str, str]] = {
    "show_1":  {"drone78":  "20251020-001"},
    "show_2":  {"drone78":  "20251020-002"},
    "show_5":  {"drone78":  "20251020-003"},
    "show_7":  {"drone78":  "20251020-004"},
    "show_8":  {"drone88":  "20251020-001"},
    "show_9":  {"drone130": "20251020-001"},
    "show_11": {"drone130": "20251020-002", "drone65":  "20251020-001"},
    "show_12": {"drone65":  "20251020-002", "drone130": "20251020-003", "drone132": "20251020-001"},
    "show_13": {"drone65":  "20251020-003", "drone78":  "20251020-005",
                "drone130": "20251020-004", "drone132": "20251020-002"},
    "show_14": {"drone65":  "20251020-004", "drone78":  "20251020-006",
                "drone88":  "20251020-002", "drone130": "20251020-005",
                "drone132": "20251020-002"},   # ⚠ same log ID as show_13!
    "show_15": {"drone65":  "20251020-005", "drone78":  "20251020-007",
                "drone88":  "20251020-003", "drone130": "20251020-006",
                "drone132": "20251020-003"},
    "show_10": {"drone130": "20251020-007"},
}

# Which shows each drone flew (derived from DRONE_LOGS above)
DRONE_SHOWS: Dict[str, List[str]] = {}
for _show, _drones in DRONE_LOGS.items():
    for _drone in _drones:
        DRONE_SHOWS.setdefault(_drone, []).append(_show)


def build_drone_registry() -> dict:
    registry = {}
    for drone_id in sorted(DRONE_SHOWS):
        shows = DRONE_SHOWS[drone_id]
        registry[drone_id] = {
            "shows":          shows,
            "n_shows":        len(shows),
            "logs":           {s: DRONE_LOGS[s][drone_id] for s in shows if drone_id in DRONE_LOGS.get(s, {})},
            "gpx_path_show":  f"DRON-GPX/bySHOW/",
            "gpx_path_drone": f"DRON-GPX/byDRONE/{drone_id}/",
            "telemetry_path": f"Drón telemetria/collmot_dron_meres_20251020/{drone_id}/",
        }
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# Session (show) definitions
# (wall_clock from Volk notes; flight_time_min from Jegyzőkönyv)
# ══════════════════════════════════════════════════════════════════════════════

SESSIONS: List[Dict[str, Any]] = [
    # ── Survey + hover series ─────────────────────────────────────────────────
    {
        "session_id":      "show_1",
        "show_number":     1,
        "wall_clock":      "13:36",   # Volk notes (authoritative trigger time)
        "flight_time_min": 8.43,
        "maneuver_ids":    [0, 1, 2, 3, 4],
        "n_drones":        1,
        "drones":          ["drone78"],
        "description":     "Range-corner survey + altitude hover series 10–40 m",
        "mems_recording":  False,
        "notes":           "MEMS not recording; 13:44 gain increased to 60 dB",
        "gpx_folder":      "DRON-GPX/bySHOW/Show01/",
    },
    {
        "session_id":      "show_2",
        "show_number":     2,
        "wall_clock":      "13:50",   # Volk notes
        "flight_time_min": 9.70,
        "maneuver_ids":    [5, 6, 7, 8, 10, 11],
        "n_drones":        1,
        "drones":          ["drone78"],
        "description":     "Yaw rotations + linear/diagonal transits at 20 m",
        "mems_recording":  False,
        "notes":           "Yaw rotations (m5, m6) not executed; MEMS not recording",
        "gpx_folder":      "DRON-GPX/bySHOW/Show02/",
    },
    # ── Diagonal transits ────────────────────────────────────────────────────
    {
        "session_id":      "show_5",
        "show_number":     5,
        "wall_clock":      "14:08",   # Volk notes; matches Jegyzőkönyv 14:08-14:15
        "flight_time_min": 6.82,
        "maneuver_ids":    [13, 14],
        "n_drones":        1,
        "drones":          ["drone78"],
        "description":     "Square diagonal transits at 60 m altitude (4 and 8 m/s)",
        "mems_recording":  True,
        "mems_starts_at":  "14:06",   # MEMS started 2 min before show_5 trigger
        "notes":           "MEMS recording started at 14:06 (pre-show_5); show_5 drone log timestamps may be unreliable (saved as tmp log)",
        "gpx_folder":      "DRON-GPX/bySHOW/Show05/",
    },
    {
        "session_id":      "show_7",
        "show_number":     7,
        "wall_clock":      "14:24",   # Volk notes; matches Jegyzőkönyv ~14:24-14:32
        "flight_time_min": 8.32,
        "maneuver_ids":    [16, 17],
        "n_drones":        1,
        "drones":          ["drone78"],
        "description":     "Square diagonal transits at 120 m altitude (4 and 8 m/s)",
        "mems_recording":  True,
        "gpx_folder":      "DRON-GPX/bySHOW/Show07/",
    },
    # ── Circular orbits (single drone) ───────────────────────────────────────
    {
        "session_id":      "show_8",
        "show_number":     8,
        "wall_clock":      "14:41",   # Volk notes; matches Jegyzőkönyv ~14:41-14:51
        "flight_time_min": 9.63,
        "maneuver_ids":    [18, 19, 20, 21],
        "n_drones":        1,
        "drones":          ["drone88"],
        "description":     "Single-drone circular orbits r=5,30,60 m + figure-8 at 20 m",
        "mems_recording":  True,
        "notes":           "drone88 (not drone78); m18 r=5m — uncertain if always started from r=5m",
        "gpx_folder":      "DRON-GPX/bySHOW/Show08/",
    },
    # ── 3D cube diagonals ────────────────────────────────────────────────────
    {
        "session_id":      "show_9",
        "show_number":     9,
        "wall_clock":      "14:57",   # Volk notes (Jegyzőkönyv says ~15:02 = flight start not trigger)
        "flight_time_min": 6.78,
        "maneuver_ids":    [22, 25, 23, 24],   # actual execution order per Volk notes
        "n_drones":        1,
        "drones":          ["drone130"],
        "description":     "3D cube diagonal transits (all 4 space diagonals at 4 m/s)",
        "mems_recording":  True,
        "notes":           "drone130 could not descend to z=0 for lower vertices; used ~z=5m instead. Execution order: 22→25→23→24",
        "gpx_folder":      "DRON-GPX/bySHOW/Show09/",
    },
    # ── Multi-drone circular orbits ──────────────────────────────────────────
    {
        "session_id":      "show_11",
        "show_number":     11,
        "wall_clock":      "15:13",   # Volk notes; matches Jegyzőkönyv 15:13-15:19
        "flight_time_min": 6.23,
        "maneuver_ids":    [27, 28, 29],
        "n_drones":        2,
        "drones":          ["drone65", "drone130"],
        "description":     "2-drone circular orbits r=5, 30, 60 m at 20 m altitude",
        "mems_recording":  True,
        "gpx_folder":      "DRON-GPX/bySHOW/Show11/",
    },
    {
        "session_id":      "show_12",
        "show_number":     12,
        "wall_clock":      "15:27",   # Volk notes; matches Jegyzőkönyv 15:27-15:33
        "flight_time_min": 6.23,
        "maneuver_ids":    [30, 31, 32],
        "n_drones":        3,
        "drones":          ["drone65", "drone130", "drone132"],
        "description":     "3-drone circular orbits r=10, 30, 60 m at 20 m altitude",
        "mems_recording":  True,
        "notes":           "r=5m too narrow → adjusted to r=10m (may have been adjusted from show_12 onward)",
        "gpx_folder":      "DRON-GPX/bySHOW/Show12/",
    },
    {
        "session_id":      "show_13",
        "show_number":     13,
        "wall_clock":      "15:42",   # Volk notes; matches Jegyzőkönyv 15:42-15:49
        "flight_time_min": 6.23,
        "maneuver_ids":    [33, 34, 35],
        "n_drones":        4,
        "drones":          ["drone65", "drone78", "drone130", "drone132"],
        "description":     "4-drone circular orbits r=10, 30, 60 m at 20 m altitude",
        "mems_recording":  True,
        "gpx_folder":      "DRON-GPX/bySHOW/Show13/",
    },
    {
        "session_id":      "show_14",
        "show_number":     14,
        "wall_clock":      "15:57",   # Volk notes; matches Jegyzőkönyv ~15:57-16:03
        "flight_time_min": 6.23,
        "maneuver_ids":    [36, 37, 38],
        "n_drones":        5,
        "drones":          ["drone65", "drone78", "drone88", "drone130", "drone132"],
        "description":     "5-drone circular orbits r=10, 30, 60 m at 20 m altitude",
        "mems_recording":  True,
        "notes":           "drone132 log 20251020-002 same ID as show_13 — may be same log file or note error",
        "gpx_folder":      "DRON-GPX/bySHOW/Show14/",
    },
    # ── V-formation ──────────────────────────────────────────────────────────
    {
        "session_id":      "show_15",
        "show_number":     15,
        "wall_clock":      "16:11",   # Volk notes
        "flight_time_min": 3.45,
        "maneuver_ids":    [39, 40],
        "n_drones":        5,
        "drones":          ["drone65", "drone78", "drone88", "drone130", "drone132"],
        "description":     "5-drone V-formation transits at 4 and 8 m/s",
        "mems_recording":  True,
        "notes":           "V formation not always maintained; Jegyzőkönyv lists ~16:25 which is the end time",
        "gpx_folder":      "DRON-GPX/bySHOW/Show15/",
    },
    # ── Long-range detection test ────────────────────────────────────────────
    {
        "session_id":      "show_10",
        "show_number":     10,
        "wall_clock":      "16:25",   # Volk notes (manual flight, no show file)
        "flight_time_min": None,      # manual flight, duration approximate
        "maneuver_ids":    [41],
        "n_drones":        1,
        "drones":          ["drone130"],
        "description":     "Max-range detection test: manual NW→SE approach at 20 m, 300 m radius",
        "mems_recording":  True,
        "notes":           "Flown manually (no show file); audible ~200m on BK-6-E. 300m NW then 300m SE of origin.",
        "gpx_folder":      "DRON-GPX/bySHOW/Show10/",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Maneuver segment definitions
# (Source: Jegyzőkönyv maneuver table, corrected by Volk notes)
# ══════════════════════════════════════════════════════════════════════════════

# Transition buffer between maneuvers within a show (seconds)
_T = 3

MANEUVER_SEGMENTS: List[Dict[str, Any]] = [

    # ── show_1: survey + hovering ─────────────────────────────────────────────
    {"id": 0,  "session": "show_1",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "survey",     "flight_phase": "survey",
     "description": "Range-corner survey at 10 m",
     "altitude_m": 10, "speed_mps": None, "radius_m": None,
     "start_coord": None, "end_coord": None, "duration_s": 120,
     
     "quality_flags": ["pre_measurement"]},

    {"id": 1,  "session": "show_1",  "within_session_offset_s": 123,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover",
     "description": "Hover at 10 m, 1 min",
     "altitude_m": 10, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 10], "end_coord": [0, 0, 10], "duration_s": 60,
      "quality_flags": []},

    {"id": 2,  "session": "show_1",  "within_session_offset_s": 186,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover",
     "description": "Hover at 20 m, 1 min",
     "altitude_m": 20, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 20], "end_coord": [0, 0, 20], "duration_s": 60,
      "quality_flags": []},

    {"id": 3,  "session": "show_1",  "within_session_offset_s": 249,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover",
     "description": "Hover at 30 m, 1 min",
     "altitude_m": 30, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 30], "end_coord": [0, 0, 30], "duration_s": 60,
      "quality_flags": []},

    {"id": 4,  "session": "show_1",  "within_session_offset_s": 312,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover",
     "description": "Hover at 40 m, 1 min",
     "altitude_m": 40, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 40], "end_coord": [0, 0, 40], "duration_s": 60,
     
     "quality_flags": ["gain_increased_to_60dB_at_t376s_from_rec_start"]},

    # ── show_2: yaw rotations + linear + diagonal transits ───────────────────
    {"id": 5,  "session": "show_2",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover_yaw_ccw",
     "description": "CCW yaw rotation at 20 m, 1 min (NOT executed)",
     "altitude_m": 20, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 20], "end_coord": [0, 0, 20], "duration_s": 60,
     
     "quality_flags": ["yaw_not_executed", "effectively_hover"]},

    {"id": 6,  "session": "show_2",  "within_session_offset_s": 63,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "hover",      "flight_phase": "hover_yaw_cw",
     "description": "CW yaw rotation at 20 m, 1 min (NOT executed)",
     "altitude_m": 20, "speed_mps": 0.0, "radius_m": None,
     "start_coord": [0, 0, 20], "end_coord": [0, 0, 20], "duration_s": 60,
     
     "quality_flags": ["yaw_not_executed", "effectively_hover"]},

    {"id": 7,  "session": "show_2",  "within_session_offset_s": 126,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "transit",    "flight_phase": "transit_linear",
     "description": "Linear transit at 20 m, 4 m/s (back-and-forth along Y axis)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, 0, 20], "end_coord": [60, 0, 20], "duration_s": 30,
      "quality_flags": []},

    {"id": 8,  "session": "show_2",  "within_session_offset_s": 159,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "transit",    "flight_phase": "transit_linear",
     "description": "Linear transit at 20 m, 8 m/s (back-and-forth along Y axis)",
     "altitude_m": 20, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-60, 0, 20], "end_coord": [60, 0, 20], "duration_s": 15,
      "quality_flags": []},

    # maneuver 9 (2 m/s diagonal) was planned but skipped ("hagyjuk el")
    {"id": 10, "session": "show_2",  "within_session_offset_s": 177,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 20 m, 4 m/s × 2 (SW↔NE)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, -60, 20], "end_coord": [60, 60, 20], "duration_s": 66,
      "quality_flags": []},

    {"id": 11, "session": "show_2",  "within_session_offset_s": 246,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 20 m, 8 m/s × 2 (SW↔NE)",
     "altitude_m": 20, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-60, -60, 20], "end_coord": [60, 60, 20], "duration_s": 33,
      "quality_flags": []},

    # ── show_5: 60 m diagonals ───────────────────────────────────────────────
    # maneuver 12 (2 m/s, show_4) was skipped. maneuver 13 is first in show_5.
    {"id": 13, "session": "show_5",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 60 m, 4 m/s × 2 (SW↔NE)",
     "altitude_m": 60, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, -60, 60], "end_coord": [60, 60, 60], "duration_s": 66,
     
     "quality_flags": ["mems_starts_120s_before_show_trigger", "show5_log_timestamps_unreliable"]},

    {"id": 14, "session": "show_5",  "within_session_offset_s": 69,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 60 m, 8 m/s × 2 (SW↔NE)",
     "altitude_m": 60, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-60, -60, 60], "end_coord": [60, 60, 60], "duration_s": 33,
     
     "quality_flags": ["show5_log_timestamps_unreliable"]},

    # ── show_7: 120 m diagonals ──────────────────────────────────────────────
    # maneuver 15 (2 m/s, show_6) was skipped.
    {"id": 16, "session": "show_7",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 120 m, 4 m/s × 2 (SW↔NE)",
     "altitude_m": 120, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, -60, 120], "end_coord": [60, 60, 120], "duration_s": 66,
      "quality_flags": []},

    {"id": 17, "session": "show_7",  "within_session_offset_s": 69,
     "n_drones": 1, "drones": ["drone78"],
     "maneuver_type": "diagonal",   "flight_phase": "transit_diagonal",
     "description": "Square diagonal at 120 m, 8 m/s × 2 (SW↔NE)",
     "altitude_m": 120, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-60, -60, 120], "end_coord": [60, 60, 120], "duration_s": 33,
      "quality_flags": []},

    # ── show_8: single-drone circular orbits + figure-8 ─────────────────────
    {"id": 18, "session": "show_8",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone88"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "Circle r=5 m at 20 m altitude, 4 m/s × 2 laps",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 5.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 16,
     
     "quality_flags": ["uncertain_start_radius", "short_segment"]},

    {"id": 19, "session": "show_8",  "within_session_offset_s": 19,
     "n_drones": 1, "drones": ["drone88"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "Circle r=30 m at 20 m altitude, 4 m/s × 2 laps",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 30.0,
     "start_coord": [-15, 0, 20], "end_coord": None, "duration_s": 94,
      "quality_flags": []},

    {"id": 20, "session": "show_8",  "within_session_offset_s": 116,
     "n_drones": 1, "drones": ["drone88"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "Circle r=60 m at 20 m altitude, 4 m/s × 2 laps",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 60.0,
     "start_coord": [-30, 0, 20], "end_coord": None, "duration_s": 188,
      "quality_flags": []},

    {"id": 21, "session": "show_8",  "within_session_offset_s": 307,
     "n_drones": 1, "drones": ["drone88"],
     "maneuver_type": "figure8",    "flight_phase": "figure8",
     "description": "Figure-8 (lying-8) at 20 m altitude, 4 m/s × 2 laps (~120 m span)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": None,
     "start_coord": None, "end_coord": None, "duration_s": 63,
     
     "quality_flags": ["approx_120m_total_span"]},

    # ── show_9: 3D cube diagonals ────────────────────────────────────────────
    # Execution order from Volk notes: 22→25→23→24
    # ⚠ FIX: mems_available corrected to False — show_9 starts at 14:57, but
    #   MEMS recording ends at ~14:54 (12 files × 242.3 s = 48.5 min from 14:06).
    {"id": 22, "session": "show_9",  "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone130"],
     "maneuver_type": "diagonal_3d", "flight_phase": "transit_diagonal_3d",
     "description": "Cube diagonal A: (−60,−60,~5) → (+60,+60,120) at 4 m/s",
     "altitude_m": 60, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, -60, 5], "end_coord": [60, 60, 120], "duration_s": 69,
     
     "quality_flags": ["start_z_approx_5m_not_0", "execution_order_1_of_4", "mems_ended_before_show"]},

    {"id": 25, "session": "show_9",  "within_session_offset_s": 72,
     "n_drones": 1, "drones": ["drone130"],
     "maneuver_type": "diagonal_3d", "flight_phase": "transit_diagonal_3d",
     "description": "Cube diagonal D: (+60,+60,~5) → (−60,−60,120) at 4 m/s",
     "altitude_m": 60, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [60, 60, 5], "end_coord": [-60, -60, 120], "duration_s": 69,
     
     "quality_flags": ["start_z_approx_5m_not_0", "execution_order_2_of_4", "mems_ended_before_show"]},

    {"id": 23, "session": "show_9",  "within_session_offset_s": 144,
     "n_drones": 1, "drones": ["drone130"],
     "maneuver_type": "diagonal_3d", "flight_phase": "transit_diagonal_3d",
     "description": "Cube diagonal B: (−60,+60,~5) → (+60,−60,120) at 4 m/s",
     "altitude_m": 60, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, 60, 5], "end_coord": [60, -60, 120], "duration_s": 69,
     
     "quality_flags": ["start_z_approx_5m_not_0", "execution_order_3_of_4", "mems_ended_before_show"]},

    {"id": 24, "session": "show_9",  "within_session_offset_s": 216,
     "n_drones": 1, "drones": ["drone130"],
     "maneuver_type": "diagonal_3d", "flight_phase": "transit_diagonal_3d",
     "description": "Cube diagonal C: (+60,−60,~5) → (−60,+60,120) at 4 m/s",
     "altitude_m": 60, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [60, -60, 5], "end_coord": [-60, 60, 120], "duration_s": 69,
     
     "quality_flags": ["start_z_approx_5m_not_0", "execution_order_4_of_4", "mems_ended_before_show"]},

    # ── show_11: 2-drone circular orbits ─────────────────────────────────────
    # ⚠ FIX: show_11 starts at 15:13 — well after MEMS recording ended at ~14:54.
    #   mems_available corrected to False for all show_11–show_15 and show_10 segments.
    {"id": 27, "session": "show_11", "within_session_offset_s": 0,
     "n_drones": 2, "drones": ["drone65", "drone130"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "2-drone circle r=5 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 5.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 8,
     
     "quality_flags": ["short_segment", "mems_ended_before_show"]},

    {"id": 28, "session": "show_11", "within_session_offset_s": 11,
     "n_drones": 2, "drones": ["drone65", "drone130"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "2-drone circle r=30 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 30.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 47,
      "quality_flags": ["mems_ended_before_show"]},

    {"id": 29, "session": "show_11", "within_session_offset_s": 61,
     "n_drones": 2, "drones": ["drone65", "drone130"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "2-drone circle r=60 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 60.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 188,
      "quality_flags": ["mems_ended_before_show"]},

    # ── show_12: 3-drone circular orbits ─────────────────────────────────────
    # r=5m adjusted to r=10m (Jegyzőkönyv note: "5m-es sugarú kör túl szűk, 10m-re módosítva")
    {"id": 30, "session": "show_12", "within_session_offset_s": 0,
     "n_drones": 3, "drones": ["drone65", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "3-drone circle r=10 m at 20 m altitude, 4 m/s (r=5m was too narrow → 10m)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 10.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 8,
     
     "quality_flags": ["radius_adjusted_from_5m_to_10m", "short_segment", "mems_ended_before_show"]},

    {"id": 31, "session": "show_12", "within_session_offset_s": 11,
     "n_drones": 3, "drones": ["drone65", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "3-drone circle r=30 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 30.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 47,
      "quality_flags": ["mems_ended_before_show"]},

    {"id": 32, "session": "show_12", "within_session_offset_s": 61,
     "n_drones": 3, "drones": ["drone65", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "3-drone circle r=60 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 60.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 188,
      "quality_flags": ["mems_ended_before_show"]},

    # ── show_13: 4-drone circular orbits ─────────────────────────────────────
    {"id": 33, "session": "show_13", "within_session_offset_s": 0,
     "n_drones": 4, "drones": ["drone65", "drone78", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "4-drone circle r=10 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 10.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 8,
     
     "quality_flags": ["short_segment", "mems_ended_before_show"]},

    {"id": 34, "session": "show_13", "within_session_offset_s": 11,
     "n_drones": 4, "drones": ["drone65", "drone78", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "4-drone circle r=30 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 30.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 47,
      "quality_flags": ["mems_ended_before_show"]},

    {"id": 35, "session": "show_13", "within_session_offset_s": 61,
     "n_drones": 4, "drones": ["drone65", "drone78", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "4-drone circle r=60 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 60.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 188,
      "quality_flags": ["mems_ended_before_show"]},

    # ── show_14: 5-drone circular orbits ─────────────────────────────────────
    {"id": 36, "session": "show_14", "within_session_offset_s": 0,
     "n_drones": 5, "drones": ["drone65", "drone78", "drone88", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "5-drone circle r=10 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 10.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 8,
     
     "quality_flags": ["short_segment", "drone132_log_id_shared_with_show_13", "mems_ended_before_show"]},

    {"id": 37, "session": "show_14", "within_session_offset_s": 11,
     "n_drones": 5, "drones": ["drone65", "drone78", "drone88", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "5-drone circle r=30 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 30.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 47,
     
     "quality_flags": ["drone132_log_id_shared_with_show_13", "mems_ended_before_show"]},

    {"id": 38, "session": "show_14", "within_session_offset_s": 61,
     "n_drones": 5, "drones": ["drone65", "drone78", "drone88", "drone130", "drone132"],
     "maneuver_type": "circle",     "flight_phase": "orbit",
     "description": "5-drone circle r=60 m at 20 m altitude, 4 m/s (drones evenly spaced)",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": 60.0,
     "start_coord": [-2.5, 0, 20], "end_coord": None, "duration_s": 188,
     
     "quality_flags": ["drone132_log_id_shared_with_show_13", "mems_ended_before_show"]},

    # ── show_15: 5-drone V-formation transits ────────────────────────────────
    {"id": 39, "session": "show_15", "within_session_offset_s": 0,
     "n_drones": 5, "drones": ["drone65", "drone78", "drone88", "drone130", "drone132"],
     "maneuver_type": "formation",  "flight_phase": "formation_v",
     "description": "V-formation transit at 20 m altitude, 4 m/s",
     "altitude_m": 20, "speed_mps": 4.0, "radius_m": None,
     "start_coord": [-60, 0, 20], "end_coord": [60, 0, 20], "duration_s": 60,
     
     "quality_flags": ["v_formation_not_always_maintained", "mems_ended_before_show"]},

    {"id": 40, "session": "show_15", "within_session_offset_s": 63,
     "n_drones": 5, "drones": ["drone65", "drone78", "drone88", "drone130", "drone132"],
     "maneuver_type": "formation",  "flight_phase": "formation_v",
     "description": "V-formation transit at 20 m altitude, 8 m/s",
     "altitude_m": 20, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-60, 0, 20], "end_coord": [60, 0, 20], "duration_s": 30,
     
     "quality_flags": ["v_formation_not_always_maintained", "mems_ended_before_show"]},

    # ── show_10: long-range detection test ───────────────────────────────────
    {"id": 41, "session": "show_10", "within_session_offset_s": 0,
     "n_drones": 1, "drones": ["drone130"],
     "maneuver_type": "long_range", "flight_phase": "long_range",
     "description": "Max-range manual approach: 300m NW → origin → 300m SE at 20m, 8 m/s",
     "altitude_m": 20, "speed_mps": 8.0, "radius_m": None,
     "start_coord": [-212.0, -212.0, 20],  # 300m * cos(45°) ≈ 212m
     "end_coord":   [212.0,  212.0,  20],
     "duration_s":  188,
     
     "quality_flags": ["manual_flight", "no_show_file", "audible_to_200m_on_BK6E", "mems_ended_before_show"]},
]


# ══════════════════════════════════════════════════════════════════════════════
# MEMS format — verify at runtime from WAV header
# ══════════════════════════════════════════════════════════════════════════════

# Default assumed format — VERIFY from actual WAV header before trusting the
# file index.  Call verify_mems_format(path) once you have local file access.
MEMS_ASSUMED_FORMAT = {
    "sample_rate_hz":  48_000,
    "channels":        4,
    "bit_depth":       24,
    "bytes_per_sample": 3,       # 24-bit packed integer
    "duration_s":      242.3,    # per 133.1 MB file at these settings
    "verified":        False,    # ← SET TO True AFTER verify_mems_format()
    "note": (
        "133.1 MB / (4ch × 3 bytes × 48000 Hz) = 242 s. "
        "If your analysis shows a different format, re-run build_ground_truth() "
        "with the corrected MEMS_ASSUMED_FORMAT values."
    ),
}

MEMS_START_LOCAL_S = local_hhmm_to_s("14:06")  # Volk notes: MEMS recording start

# Actual MEMS filenames (from screenshot — note _02 suffix on first and last)
MEMS_FILES = [
    "Audio 01_02.wav",   # take 2 (retake of first recording)
    "Audio 02_01.wav",
    "Audio 03_01.wav",
    "Audio 04_01.wav",
    "Audio 05_01.wav",
    "Audio 06_01.wav",
    "Audio 07_01.wav",
    "Audio 08_01.wav",
    "Audio 09_01.wav",
    "Audio 10_01.wav",
    "Audio 11_01.wav",
    "Audio 12_02.wav",   # take 2 (retake of last recording)
]


def verify_mems_format(wav_path: str) -> dict:
    """
    Read the WAV header from any MEMS file and return the true format dict.
    Updates MEMS_ASSUMED_FORMAT in place.

    Usage (once you have local file access):
        fmt = verify_mems_format("/path/to/Audio 01_02.wav")
        # then re-run build_ground_truth() to get correct file_index timings
    """
    with open(wav_path, "rb") as f:
        riff = f.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {wav_path}")
        # Find the fmt chunk
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                raise ValueError("No fmt chunk found")
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                break
            f.seek(chunk_size, 1)

    audio_fmt  = struct.unpack_from("<H", fmt_data, 0)[0]
    channels   = struct.unpack_from("<H", fmt_data, 2)[0]
    sample_rate= struct.unpack_from("<I", fmt_data, 4)[0]
    bit_depth  = struct.unpack_from("<H", fmt_data, 14)[0]
    bytes_per_sample = (bit_depth + 7) // 8

    file_size = Path(wav_path).stat().st_size
    # Subtract WAV header overhead (~44 bytes typical, but be safe)
    data_bytes = file_size - 44
    frames = data_bytes // (channels * bytes_per_sample)
    duration_s = frames / sample_rate

    result = {
        "sample_rate_hz":   sample_rate,
        "channels":         channels,
        "bit_depth":        bit_depth,
        "bytes_per_sample": bytes_per_sample,
        "duration_s":       round(duration_s, 2),
        "audio_format":     "PCM" if audio_fmt == 1 else "float" if audio_fmt == 3 else f"fmt_{audio_fmt}",
        "verified":         True,
        "source_file":      str(wav_path),
    }
    MEMS_ASSUMED_FORMAT.update(result)
    print(f"  ✅ MEMS format verified: {channels}ch {bit_depth}-bit {sample_rate}Hz "
          f"→ {duration_s:.1f}s per file, total {12*duration_s/60:.1f} min")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PolyWav file list
# (from screenshot: 17 files — unlabeled first + A through P)
# ══════════════════════════════════════════════════════════════════════════════

# Each file is exactly 4 GB, 192 kHz, 14 ch, float32
_PW_SIZE_BYTES  = 4 * 1024 ** 3
_PW_BYTES_PER_FRAME = 14 * 4   # 14 channels × float32
_PW_FRAMES = _PW_SIZE_BYTES // _PW_BYTES_PER_FRAME
PW_CHUNK_DUR_S = _PW_FRAMES / 192_000   # 399.46 s ≈ 6 min 39 s

# FIX: the recording ran 13:36–16:28 (≈172 min = 10 328 s), requiring 26 chunks (slots 0-25).
# The original list was hardcoded to 17 files (unlabeled + A–P, covering only to ~15:29).
# Shows 13, 14, 15, 10 (15:42 onward) require files Q through Y — now included.
# The extractor scans the WAV directory at runtime, so it will find any file present;
# this list is used only for file_index.json cross-reference and consistency checks.
POLYWAV_FILES: List[str] = (
    ["251020VITEMOROM1AT01.wav"]                                       # slot 0 — unlabeled
    + [f"251020VITEMOROM1AT01{L}.wav" for L in "ABCDEFGHIJKLMNOPQRSTUVWXY"]  # slots 1-25 (A–Y)
)
assert len(POLYWAV_FILES) == 26, "Expected 26 PolyWav chunks (unlabeled + A–Y)"


# ══════════════════════════════════════════════════════════════════════════════
# Enrichment
# ══════════════════════════════════════════════════════════════════════════════

def _enrich_sessions(sessions: list) -> list:
    out = []
    for s in sessions:
        s = dict(s)
        local_s = local_hhmm_to_s(s["wall_clock"])
        s["local_start_s"]    = local_s
        s["smpte_start_s"]    = local_s
        s["onset_from_rec_s"] = local_s - RECORDING_REF_LOCAL_S
        s["utc_start_s"]      = local_s - UTC_TO_LOCAL_S
        s["local_start_hms"]  = s_to_hhmm(local_s)
        s["utc_start_hms"]    = s_to_hhmm(s["utc_start_s"])
        out.append(s)
    return out


def _assign_split(seg: dict) -> str:
    """Train/val/test split assignment based on task difficulty."""
    if seg["flight_phase"] == "long_range":   return "test"
    if seg["n_drones"] == 5:                  return "test"
    if seg["n_drones"] == 4:                  return "val"
    if seg["maneuver_type"] in ("hover", "survey"): return "val"
    return "train"


def _enrich_segments(segments: list, sessions_by_id: dict) -> list:
    out = []
    for seg in segments:
        seg = dict(seg)
        sess = sessions_by_id[seg["session"]]

        # Absolute timing (must come FIRST)
        local_start = sess["local_start_s"] + seg["within_session_offset_s"]
        seg["local_start_s"]    = local_start
        seg["local_start_hms"]  = s_to_hhmm(local_start)
        seg["smpte_start_s"]    = local_start
        seg["utc_start_s"]      = local_start - UTC_TO_LOCAL_S
        seg["utc_start_hms"]    = s_to_hhmm(seg["utc_start_s"])
        seg["onset_from_rec_s"] = local_start - RECORDING_REF_LOCAL_S
        seg["local_end_s"]      = local_start + seg["duration_s"]
        seg["local_end_hms"]    = s_to_hhmm(seg["local_end_s"])

        # MEMS availability calculation (now onset_from_rec_s exists)
        mems_start_rec_s = MEMS_START_LOCAL_S - RECORDING_REF_LOCAL_S
        mems_duration_total = len(MEMS_FILES) * MEMS_ASSUMED_FORMAT["duration_s"]
        mems_end_rec_s = mems_start_rec_s + mems_duration_total
        
        seg_start = seg["onset_from_rec_s"]
        seg_end = seg_start + seg["duration_s"]
        
        seg["mems_available"] = (
            seg_start < mems_end_rec_s and 
            seg_end > mems_start_rec_s and
            seg_start >= 0
        )
        
        # BK-6 arrays recorded continuously for all shows, but the range-read
        # streamer refuses onsets within the first second of the recording
        # (that offset falls inside the WAV header, not audio) — see the
        # onset_s < 1.0 guard in dunakeszi_nextcloud.stream_segment_from_nextcloud.
        seg["bk_available"] = seg_start >= 1.0

        # Azimuth and distance at onset (from start_coord XY)
        # Coordinate frame: X=East, Y=North, Z=Up (mic_array_geometry.json)
        # Azimuth convention: standard geographic bearing measured clockwise from North.
        #   bearing = atan2(X, Y)  (NOT atan2(Y, X) which gives the math/Cartesian angle)
        # Examples:
        #   drone due North  → (x=0,  y=+60) → atan2(0,  60) =   0°
        #   drone due East   → (x=+60,y=0  ) → atan2(60,  0) =  90°
        #   drone due South  → (x=0,  y=-60) → atan2(0, -60) = 180°
        #   drone due West   → (x=-60,y=0  ) → atan2(-60,0)  = -90° (= 270°)
        # FIX: original used atan2(y, x) giving math angle, not bearing from North.
        sc = seg.get("start_coord")
        if sc and sc[0] is not None and sc[1] is not None:
            x, y = sc[0], sc[1]
            seg["azimuth_deg_onset"]   = round(math.degrees(math.atan2(x, y)), 1)
            seg["distance_xy_m_onset"] = round(math.hypot(x, y), 1)
            seg["distance_3d_m_onset"] = round(math.hypot(x, y, sc[2] if len(sc) > 2 else 0), 1)
        else:
            seg["azimuth_deg_onset"]   = None
            seg["distance_xy_m_onset"] = None
            seg["distance_3d_m_onset"] = None

        # Task labels
        seg["split"] = _assign_split(seg)

        # Data source pointers
        seg["gpx_folder"] = sess.get("gpx_folder")
        seg["log_files"]  = {
            d: DRONE_LOGS.get(seg["session"], {}).get(d)
            for d in seg["drones"]
        }

        out.append(seg)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# File index  (reverse lookup: filename → segment IDs)
# ══════════════════════════════════════════════════════════════════════════════
# Add MEMS file duration verification before building index
def verify_mems_files(mems_directory):
    """Actually read all MEMS files to get true start times"""
    files = sorted(Path(mems_directory).glob("Audio*.wav"))
    actual_starts = []
    for f in files:
        # Read creation time or use filename patterns
        # Your filenames like "0071000" suggest seconds from some reference
        match = re.search(r'(\d{7})', str(f))
        if match:
            actual_starts.append(int(match.group(1)) / 1000)  # Convert to seconds
    return actual_starts

def build_file_index(segments: list) -> dict:
    """
    For every data file on the NAS, list which maneuver segment IDs it contains.

    Lookup keys:
      polywav_chunks  : "251020VITEMOROM1AT01.wav", "251020VITEMOROM1AT01A.wav" …
      mems_audio      : "Audio 01_02.wav", "Audio 02_01.wav" …
      gpx_logs        : "drone78/log_20251020-001", "drone130/log_20251020-003" …
    """
    # ── PolyWav chunks ─────────────────────────────────────────────────────
    polywav_index: Dict[str, List[int]] = {}
    for i, fname in enumerate(POLYWAV_FILES):
        chunk_start = i * PW_CHUNK_DUR_S           # seconds from rec start
        chunk_end   = chunk_start + PW_CHUNK_DUR_S
        overlap = [
            seg["id"] for seg in segments
            if seg["onset_from_rec_s"] < chunk_end
            and seg["onset_from_rec_s"] + seg["duration_s"] > chunk_start
        ]
        if overlap:
            polywav_index[fname] = overlap

    # ── MEMS audio files ────────────────────────────────────────────────────
    mems_start_rec_s = MEMS_START_LOCAL_S - RECORDING_REF_LOCAL_S
    mems_dur = MEMS_ASSUMED_FORMAT["duration_s"]

    mems_index: Dict[str, List[int]] = {}
    for i, fname in enumerate(MEMS_FILES):
        chunk_start = mems_start_rec_s+ i * mems_dur
        chunk_end   = chunk_start + mems_dur
        overlap = [
            seg["id"] for seg in segments
            if seg.get("mems_available")
            and seg["onset_from_rec_s"] < chunk_end
            and seg["onset_from_rec_s"] + seg["duration_s"] > chunk_start
        ]
        if overlap:
            mems_index[fname] = overlap

    # ── GPX / telemetry logs ────────────────────────────────────────────────
    gpx_index: Dict[str, List[int]] = {}
    for seg in segments:
        show_id = seg["session"]
        for drone in seg["drones"]:
            log_id = DRONE_LOGS.get(show_id, {}).get(drone)
            if log_id:
                key = f"{drone}/log_{log_id}"
                gpx_index.setdefault(key, [])
                if seg["id"] not in gpx_index[key]:
                    gpx_index[key].append(seg["id"])

    return {
        "polywav_chunks": polywav_index,
        "mems_audio":     mems_index,
        "gpx_logs":       gpx_index,
        "_meta": {
            "polywav_chunk_dur_s":        round(PW_CHUNK_DUR_S, 2),
            "polywav_fs_hz":              192_000,
            "polywav_channels":           14,
            "polywav_format":             "float32",
            "polywav_n_chunks":           len(POLYWAV_FILES),
            "mems_chunk_dur_s":           mems_dur,
            "mems_format_verified":       MEMS_ASSUMED_FORMAT["verified"],
            "mems_assumed_fs_hz":         MEMS_ASSUMED_FORMAT["sample_rate_hz"],
            "mems_assumed_channels":      MEMS_ASSUMED_FORMAT["channels"],
            "mems_assumed_bit_depth":     MEMS_ASSUMED_FORMAT["bit_depth"],
            "mems_start_local_s":         MEMS_START_LOCAL_S,
            "mems_start_local_hms":       s_to_hhmm(MEMS_START_LOCAL_S),
            "recording_ref_local_s":      RECORDING_REF_LOCAL_S,
            "recording_ref_local_hms":    s_to_hhmm(RECORDING_REF_LOCAL_S),
            "WARNING_mems":               (
                "MEMS format NOT verified from file header. "
                "Call verify_mems_format('/path/to/Audio 01_02.wav') then "
                "rebuild with build_ground_truth() for accurate timings."
                if not MEMS_ASSUMED_FORMAT["verified"] else "Format verified OK"
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

_CSV_COLS = [
    "id", "session", "split", "n_drones", "drones",
    "maneuver_type", "flight_phase", "description",
    "altitude_m", "speed_mps", "radius_m",
    "start_coord", "end_coord", "duration_s",
    "within_session_offset_s",
    "local_start_s", "local_start_hms", "local_end_hms",
    "smpte_start_s", "utc_start_s", "utc_start_hms",
    "onset_from_rec_s",
    "azimuth_deg_onset", "distance_xy_m_onset", "distance_3d_m_onset",
    "mems_available", "bk_available", "quality_flags", "gpx_folder",  
]


def segments_to_csv(segments: list, path: Path):
    rows = []
    for seg in segments:
        row = {}
        for c in _CSV_COLS:
            v = seg.get(c)
            row[c] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(row)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ CSV  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════════

def build_ground_truth(out_dir: str = "ground_truth") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("🏗️  Building Dunakeszi 2025-10-20 ground-truth tables …\n")

    # 1. Mic geometry
    mics = build_mic_geometry()
    for name, arr in mics["arrays"].items():
        xy = arr["local_xy_m"]
        print(f"  {name:8s}  GPS({arr['gps']['lat']:.7f}, {arr['gps']['lon']:.7f})  "
              f"XY({xy['x']:+.1f}, {xy['y']:+.1f}) m")
    print(f"  BK-6-E ↔ BK-6-W: {mics['inter_array_distances_m']['BK-6-E_to_BK-6-W']} m")
    print(f"  MEMS-S ↔ MEMS-N:  {mics['inter_array_distances_m']['MEMS-S_to_MEMS-N']} m\n")
    (out / "mic_array_geometry.json").write_text(json.dumps(mics, indent=2))
    print(f"  ✅ mic_array_geometry.json")

    # 2. Scorpio channel map
    ch_map = {
        "scorpio": SCORPIO_CHANNEL_MAP,
        "pix270i": PIX270I_CHANNEL_MAP,
        "tdoa_channels": TDOA_CHANNELS,
        "notes": {
            "scorpio_useful_channels": "3–14 (ch 1–2 are mix bus, not raw mic)",
            "pix270i_channels": "1–12 correspond to Scorpio ch 3–14",
            "tdoa_subset": "First 3 channels per array (E, H, B directions)",
        },
    }
    (out / "scorpio_channel_map.json").write_text(json.dumps(ch_map, indent=2))
    print(f"  ✅ scorpio_channel_map.json")

    # 3. Drone registry
    drones = build_drone_registry()
    for did, info in sorted(drones.items()):
        print(f"  {did}: {info['n_shows']} shows ({', '.join(info['shows'])})")
    (out / "drone_registry.json").write_text(json.dumps(drones, indent=2))
    print(f"  ✅ drone_registry.json\n")

    # 4. Sessions
    sessions_enriched = _enrich_sessions(SESSIONS)
    sessions_by_id = {s["session_id"]: s for s in sessions_enriched}
    (out / "ground_truth_sessions.json").write_text(json.dumps(sessions_enriched, indent=2))
    print(f"  ✅ ground_truth_sessions.json  ({len(sessions_enriched)} sessions)")

    # 5. Segments
    segments_enriched = _enrich_segments(MANEUVER_SEGMENTS, sessions_by_id)
    (out / "ground_truth_segments.json").write_text(json.dumps(segments_enriched, indent=2))
    print(f"  ✅ ground_truth_segments.json  ({len(segments_enriched)} segments)")
    segments_to_csv(segments_enriched, out / "ground_truth_segments.csv")

    mems_segments = [s for s in segments_enriched if s.get("mems_available")]
    print("\n📊 MEMS segments by show:")
    for seg in mems_segments:
        print(f"  {seg['session']}: seg {seg['id']} - {seg['description']} "
            f"(start: {seg['local_start_hms']}, dur: {seg['duration_s']}s)")

    # 6. File index
    file_index = build_file_index(segments_enriched)
    (out / "file_index.json").write_text(json.dumps(file_index, indent=2))
    n_pw   = len(file_index["polywav_chunks"])
    n_mems = len(file_index["mems_audio"])
    n_gpx  = len(file_index["gpx_logs"])
    print(f"  ✅ file_index.json")
    print(f"     PolyWav chunks with segments : {n_pw} / {len(POLYWAV_FILES)}")
    print(f"     MEMS files with segments     : {n_mems} / {len(MEMS_FILES)}"
          + (" ⚠ format not verified" if not MEMS_ASSUMED_FORMAT["verified"] else ""))
    print(f"     GPX log keys                 : {n_gpx}")

    # 7. Consistency checks
    # Verify that segments identified as mems_available actually appear in file_index
    mems_segments = [s for s in segments_enriched if s.get("mems_available", False)]
    mems_file_segments = set()
    for overlaps in file_index["mems_audio"].values():
        mems_file_segments.update(overlaps)

    missing = [s["id"] for s in mems_segments if s["id"] not in mems_file_segments]
    if missing:
        print(f"⚠️ WARNING: {len(missing)} segments marked mems_available but not in any MEMS file!")

    # 8 Summary
    print(f"\n{'═'*60}")
    print(f"  GROUND TRUTH SUMMARY — Dunakeszi 2025-10-20")
    print(f"{'═'*60}")
    print(f"  Sessions  : {len(sessions_enriched)}")
    print(f"  Segments  : {len(segments_enriched)}")
    total_s = sum(s["duration_s"] for s in segments_enriched)
    print(f"  Drone audio: {total_s:.0f}s = {total_s/60:.1f} min")
    for split in ("train", "val", "test"):
        segs = [s for s in segments_enriched if s["split"] == split]
        dur  = sum(s["duration_s"] for s in segs)
        print(f"    {split:5s}: {len(segs):2d} segments, {dur:.0f}s ({dur/60:.1f} min)")
    mems_segs = [s for s in segments_enriched if s.get("mems_available")]
    print(f"  MEMS available: {len(mems_segs)} segments")
    print(f"  PolyWav: {len(POLYWAV_FILES)} × 4 GB chunks "
          f"({PW_CHUNK_DUR_S:.1f}s each, 192kHz 14ch float32)")
    print(f"  MEMS   : {len(MEMS_FILES)} × 133.1 MB files "
          f"({MEMS_ASSUMED_FORMAT['duration_s']:.1f}s each ASSUMED — verify header)")
    print(f"{'═'*60}\n")

    return {
        "sessions":       sessions_enriched,
        "segments":       segments_enriched,
        "mic_geometry":   mics,
        "drone_registry": drones,
        "file_index":     file_index,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Runtime helpers  (import these in your extractor)
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_dir: str) -> dict:
    """Load pre-built ground truth from the output directory."""
    d = Path(gt_dir)
    return {
        "sessions":       json.loads((d / "ground_truth_sessions.json").read_text()),
        "segments":       json.loads((d / "ground_truth_segments.json").read_text()),
        "mic_geometry":   json.loads((d / "mic_array_geometry.json").read_text()),
        "drone_registry": json.loads((d / "drone_registry.json").read_text()),
        "file_index":     json.loads((d / "file_index.json").read_text()),
        "channel_map":    json.loads((d / "scorpio_channel_map.json").read_text()),
    }


def segments_for_file(filename: str, gt: dict) -> List[dict]:
    """
    Given any raw data filename, return the list of ground-truth segment dicts
    that overlap it.

    Examples:
        gt = load_ground_truth("ground_truth/")

        # PolyWav chunk (unlabeled or lettered):
        segs = segments_for_file("251020VITEMOROM1AT01C.wav", gt)

        # MEMS audio file:
        segs = segments_for_file("Audio 05_01.wav", gt)

        # GPX / telemetry log (drone/log_YYYYMMDD-NNN key):
        segs = segments_for_file("drone130/log_20251020-003", gt)
    """
    idx = gt["file_index"]
    seg_ids: set = set()
    for index_key in ("polywav_chunks", "mems_audio", "gpx_logs"):
        seg_ids.update(idx.get(index_key, {}).get(filename, []))
    if not seg_ids:
        return []
    seg_by_id = {s["id"]: s for s in gt["segments"]}
    return [seg_by_id[i] for i in sorted(seg_ids) if i in seg_by_id]


def timing_for_segment(seg_id: int, gt: dict) -> dict:
    """Return all timing fields for a single segment ID."""
    for s in gt["segments"]:
        if s["id"] == seg_id:
            return {k: s[k] for k in (
                "id", "session", "description",
                "local_start_s", "local_start_hms", "local_end_hms",
                "smpte_start_s", "utc_start_s", "utc_start_hms",
                "onset_from_rec_s", "duration_s",
            )}
    return {}


def polywav_seek_sample(seg_id: int, gt: dict) -> dict:
    """
    Return the sample offset within the correct PolyWav chunk for a segment.
    Useful for ffmpeg/soundfile seek-and-extract.

        import soundfile as sf
        seek = polywav_seek_sample(22, gt)
        data, fs = sf.read(
            seek["chunk_file"],
            start=seek["chunk_sample_offset"],
            stop=seek["chunk_sample_offset"] + seek["duration_samples"],
        )
    """
    seg = timing_for_segment(seg_id, gt)
    if not seg:
        return {}
    onset = seg["onset_from_rec_s"]
    fs = 192_000
    chunk_i = int(onset // PW_CHUNK_DUR_S)
    chunk_i = min(chunk_i, len(POLYWAV_FILES) - 1)
    offset_in_chunk = onset - chunk_i * PW_CHUNK_DUR_S
    return {
        "chunk_file":          POLYWAV_FILES[chunk_i],
        "chunk_index":         chunk_i,
        "chunk_sample_offset": int(offset_in_chunk * fs),
        "duration_samples":    int(seg["duration_s"] * fs),
        "sample_rate":         fs,
        "channels":            14,
        "useful_channels":     list(range(3, 15)),  # Scorpio ch 3–14 (1-indexed)
        "onset_from_rec_s":    onset,
    }


def mems_seek_sample(seg_id: int, gt: dict) -> Optional[dict]:
    """
    Return the sample offset within the correct MEMS file for a segment.
    Returns None if MEMS was not recording during this segment.

    ⚠  Accuracy depends on MEMS_ASSUMED_FORMAT being correct.
       Call verify_mems_format() first.
    """
    seg = timing_for_segment(seg_id, gt)
    if not seg:
        return None
    # Check mems_available on the full segment
    full_seg = next((s for s in gt["segments"] if s["id"] == seg_id), None)
    if not full_seg or not full_seg.get("mems_available"):
        return None

    mems_dur = MEMS_ASSUMED_FORMAT["duration_s"]
    mems_onset_from_rec = MEMS_START_LOCAL_S - RECORDING_REF_LOCAL_S
    seg_onset = seg["onset_from_rec_s"]
    offset_from_mems_start = seg_onset - mems_onset_from_rec
    if offset_from_mems_start < 0 or offset_from_mems_start > len(MEMS_FILES) * mems_dur:
        return None

    file_i = int(offset_from_mems_start // mems_dur)
    file_i = min(file_i, len(MEMS_FILES) - 1)
    offset_in_file = offset_from_mems_start - file_i * mems_dur
    fs = MEMS_ASSUMED_FORMAT["sample_rate_hz"]

    return {
        "mems_file":            MEMS_FILES[file_i],
        "file_index":           file_i,
        "file_sample_offset":   int(offset_in_file * fs),
        "duration_samples":     int(seg["duration_s"] * fs),
        "sample_rate":          fs,
        "channels":             MEMS_ASSUMED_FORMAT["channels"],
        "format_verified":      MEMS_ASSUMED_FORMAT["verified"],
        "onset_from_mems_start_s": offset_from_mems_start,
        "WARNING": (
            None if MEMS_ASSUMED_FORMAT["verified"]
            else "Offsets assume unverified format. Call verify_mems_format() first."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Dunakeszi ground-truth tables")
    p.add_argument("--out_dir", default="ground_truth",
                   help="Output directory (default: ./ground_truth)")
    p.add_argument("--verify_mems", metavar="WAV_PATH",
                   help="Path to any MEMS WAV file — verify format before building")
    args = p.parse_args()

    if args.verify_mems:
        print(f"🔍 Verifying MEMS format from: {args.verify_mems}")
        verify_mems_format(args.verify_mems)
        print()

    build_ground_truth(args.out_dir)

    # # Usage
    # python dunakeszi_ground_truth_fixed.py --out_dir ground_truth/
    # # Then in your extractor, import load_ground_truth and segments_for_file to access the data.
    # python dunakeszi_segment_extractor_fixed.py \
    # --segments ground_truth/ground_truth_segments.json \
    # --wav-dir /path/containing/just/J_file/ \
    # --array BK-6-E \
    # --clip-position 0.5