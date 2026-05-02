#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dunakeszi_segment_extractor_fixed.py
──────────────────────────────────────
Builds on the previous fixed version.  Additional bugs fixed:

  BUG 5 — args scoping: extract_from_wav_dir() referenced the global `args`
           object (defined only in main()) for args.clip_duration and
           args.clip_position.  This accidentally worked when run as __main__
           but is a latent scoping bug.  All parameters are now passed
           explicitly as function arguments.

  BUG 6 — Full-segment mode (clip_duration=0) was advertised but broken:
             • clip_audio() correctly returned full audio for clip_dur_s=0,
               but the resample → WAV write path still used TARGET_SAMPLES
               (66 150 frames = 3 s) as an implicit assumption.
             • make_label_json() was passed args.clip_duration but the
               trajectory was still computed against a 3-second window
               when seg_dur_native <= n_want_native, making clip_start_s=0
               with the wrong duration propagated to the label.
             • The manifest fields wav_ch0/1/2 and the channel-loop were
               fine, but downstream consumers would see "clip_dur_s: 3.0"
               in the label JSON even for a 45-second hover.
           Fix: actual_clip_dur_s is now computed from the clipped audio
           shape and propagated to make_label_json() and the manifest.

  BUG 7 — clip_start_s was computed correctly for long segments but then
           immediately overwritten unconditionally:
               clip_start_s = clip_start_sample / NATIVE_SR   # set inside if
               ...
               clip_start_s = clip_start_sample / NATIVE_SR   # overwrite!
           The second assignment always referenced clip_start_sample which
           was only defined inside the if-branch, causing a NameError when
           seg_dur_native <= n_want_native (short segments / full-seg mode).
           Removed the duplicate assignment.
"""

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed.  Run: pip install librosa")
    sys.exit(1)

# ── Audio constants ────────────────────────────────────────────────────────────

NATIVE_SR    = 192_000
TARGET_SR    = 22_050
# TARGET_DUR_S / TARGET_SAMPLES are only used as defaults when clip_duration>0
DEFAULT_CLIP_DUR_S = 3.0

# Exact 4 GB polywav → duration per file
_BYTES_PER_FRAME = 14 * 4          # 14 channels × float32
CHUNK_DUR_S      = (4 * 1024**3) / (_BYTES_PER_FRAME * NATIVE_SR)  # ≈ 399.4613 s

# ── FIX 1: channel mappings for both arrays ────────────────────────────────────
ARRAY_CHANNELS = {
    "BK-6-E": [8, 9, 10],
    "BK-6-W": [2, 3, 4],
}
DEFAULT_ARRAY = "BK-6-E"

REC_REF_LOCAL_S = 13 * 3600 + 36 * 60   # 48 960 s

TRAJECTORY_SAMPLE_HZ = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory computation  (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_position(seg: dict, t_within_seg: float) -> Tuple[float, float, float]:
    mtype = seg.get("maneuver_type", "hover")
    sc    = seg.get("start_coord") or [0.0, 0.0, seg.get("altitude_m", 0.0)]
    ec    = seg.get("end_coord")
    dur   = float(seg.get("duration_s", 1.0))
    alt   = float(seg.get("altitude_m") or (sc[2] if len(sc) > 2 else 0.0))

    t = max(0.0, min(t_within_seg, dur))

    if mtype in ("hover", "survey"):
        return (float(sc[0]), float(sc[1]), float(sc[2] if len(sc) > 2 else alt))

    if mtype in ("transit", "diagonal", "diagonal_3d", "formation", "long_range"):
        if ec is None:
            ec = sc
        frac = t / dur if dur > 0 else 0.0
        x = sc[0] + frac * (ec[0] - sc[0])
        y = sc[1] + frac * (ec[1] - sc[1])
        z_s = sc[2] if len(sc) > 2 else alt
        z_e = ec[2] if len(ec) > 2 else alt
        z = z_s + frac * (z_e - z_s)
        return (x, y, z)

    if mtype == "circle":
        r = float(seg.get("radius_m") or math.hypot(sc[0], sc[1]) or 30.0)
        v = float(seg.get("speed_mps") or 4.0)
        if r < 1e-3:
            return (float(sc[0]), float(sc[1]), alt)
        omega  = v / r
        theta0 = math.atan2(sc[0], sc[1])
        theta  = theta0 + omega * t
        return (r * math.sin(theta), r * math.cos(theta), alt)

    if mtype == "figure8":
        r = float(seg.get("radius_m") or math.hypot(sc[0], sc[1]) or 30.0)
        v = float(seg.get("speed_mps") or 4.0)
        if r < 1e-3:
            return (float(sc[0]), float(sc[1]), alt)
        T_circle = 2 * math.pi * r / v
        t_mod = t % (2 * T_circle)
        if t_mod < T_circle:
            cx, cy = r, 0.0
            theta  = 2 * math.pi * t_mod / T_circle
            x = cx + r * math.cos(math.pi + theta)
            y = cy + r * math.sin(math.pi + theta)
        else:
            cx, cy = -r, 0.0
            theta  = 2 * math.pi * (t_mod - T_circle) / T_circle
            x = cx + r * math.cos(-theta)
            y = cy + r * math.sin(-theta)
        return (x, y, alt)

    return (float(sc[0]), float(sc[1]), float(sc[2] if len(sc) > 2 else alt))


def _pos_to_bearing(x: float, y: float, z: float) -> dict:
    az  = math.degrees(math.atan2(x, y))
    dxy = math.hypot(x, y)
    d3d = math.hypot(x, y, z)
    return {
        "azimuth_deg":   round(az, 2),
        "distance_xy_m": round(dxy, 2),
        "distance_3d_m": round(d3d, 2),
    }


def compute_clip_trajectory(
    seg: dict,
    clip_start_s: float,
    clip_dur_s: float,
    sample_hz: float = TRAJECTORY_SAMPLE_HZ,
) -> dict:
    """
    Compute analytic trajectory over the extracted clip window.
    clip_dur_s=0 means the full segment was kept; the trajectory spans
    the entire segment duration in that case.
    """
    actual_dur = clip_dur_s if clip_dur_s > 0 else float(seg.get("duration_s", 1.0))
    n_samples  = max(1, int(actual_dur * sample_hz))
    samples: List[dict] = []

    for i in range(n_samples):
        t_in_clip = i / sample_hz
        t_in_seg  = clip_start_s + t_in_clip
        x, y, z   = interpolate_position(seg, t_in_seg)
        bearing   = _pos_to_bearing(x, y, z)
        samples.append({
            "t_in_clip_s": round(t_in_clip, 3),
            "t_in_seg_s":  round(t_in_seg, 3),
            "x_m":         round(x, 2),
            "y_m":         round(y, 2),
            "z_m":         round(z, 2),
            **bearing,
        })

    mid_idx = len(samples) // 2
    s_start = samples[0]
    s_mid   = samples[mid_idx]
    s_end   = samples[-1]

    az_start = s_start["azimuth_deg"]
    az_end   = s_end["azimuth_deg"]
    az_diff  = ((az_end - az_start) + 180) % 360 - 180

    summary = {
        "azimuth_at_start_deg":    az_start,
        "azimuth_at_mid_deg":      s_mid["azimuth_deg"],
        "azimuth_at_end_deg":      az_end,
        "distance_xy_at_start_m":  s_start["distance_xy_m"],
        "distance_xy_at_mid_m":    s_mid["distance_xy_m"],
        "distance_xy_at_end_m":    s_end["distance_xy_m"],
        "distance_3d_at_mid_m":    s_mid["distance_3d_m"],
        "altitude_at_mid_m":       s_mid["z_m"],
        "azimuth_swept_deg":       round(az_diff, 2),
        "is_approaching":          s_end["distance_xy_m"] < s_mid["distance_xy_m"],
    }

    return {
        "trajectory_source":   "analytic",
        "sample_hz":           sample_hz,
        "n_drones":            int(seg.get("n_drones", 1)),
        "clip_start_s_in_seg": round(clip_start_s, 3),
        "clip_dur_s":          actual_dur,
        "samples":             samples,
        "summary":             summary,
    }


# ── File-slot resolver ─────────────────────────────────────────────────────────

_SUFFIX_RE = re.compile(
    r"251020VITEMOROM1AT01([A-Z]*)\.wav$", re.IGNORECASE
)

def wav_slot(path: Path) -> Optional[int]:
    m = _SUFFIX_RE.match(path.name)
    if m is None:
        return None
    suffix = m.group(1).upper()
    if suffix == "":
        return 0
    if len(suffix) == 1:
        return ord(suffix) - ord('A') + 1
    if len(suffix) == 2:
        return 26 + (ord(suffix[0]) - ord('A')) * 26 + (ord(suffix[1]) - ord('A') + 1)
    return None


def slot_time_range(slot: int) -> Tuple[float, float]:
    return slot * CHUNK_DUR_S, (slot + 1) * CHUNK_DUR_S


def slot_local_hms(slot: int) -> str:
    s = int(slot * CHUNK_DUR_S) + REC_REF_LOCAL_S
    return f"{s//3600:02d}:{(s%3600)//60:02d}"


def find_best_window_by_distance(seg: dict, clip_dur_s: float) -> float:
    dur = float(seg["duration_s"])
    if dur <= clip_dur_s:
        return 0.0
    step     = 0.1
    best_t   = 0.0
    best_dist = float("inf")
    t = 0.0
    while t <= dur - clip_dur_s:
        mid_t     = t + clip_dur_s / 2
        x, y, z   = interpolate_position(seg, mid_t)
        dist      = math.hypot(x, y)
        if dist < best_dist:
            best_dist = dist
            best_t    = t
        t += step
    return best_t


def find_best_window_by_energy(audio: np.ndarray, clip_dur_s: float) -> int:
    n_total = audio.shape[1]
    n_win   = int(clip_dur_s * NATIVE_SR)
    if n_total <= n_win:
        return 0
    step       = int(0.1 * NATIVE_SR)
    best_idx   = 0
    best_energy = -1.0
    for i in range(0, n_total - n_win, step):
        energy = float(np.mean(audio[:, i:i+n_win] ** 2))
        if energy > best_energy:
            best_energy = energy
            best_idx    = i
    return best_idx


# ── Audio I/O ──────────────────────────────────────────────────────────────────

def read_polywav_window(
    wav_path: Path,
    start_sample: int,
    n_frames: int,
    channels: List[int],
) -> np.ndarray:
    info          = sf.info(str(wav_path))
    actual_start  = max(0, min(start_sample, info.frames - 1))
    actual_frames = min(n_frames, max(0, info.frames - actual_start))

    if actual_frames > 0:
        data, _ = sf.read(str(wav_path), start=actual_start,
                          frames=actual_frames, dtype="float32", always_2d=True)
        out = data[:, channels].T.astype(np.float32)
    else:
        out = np.zeros((len(channels), 0), dtype=np.float32)

    if actual_frames < n_frames:
        pad = np.zeros((len(channels), n_frames - actual_frames), dtype=np.float32)
        out = np.concatenate([out, pad], axis=1)
    return out


def extract_overlap(
    seg: dict,
    wav_path: Path,
    slot: int,
    channels: List[int],
) -> Tuple[np.ndarray, int]:
    slot_start, slot_end = slot_time_range(slot)
    seg_onset = float(seg["onset_from_rec_s"])
    seg_end   = seg_onset + float(seg["duration_s"])

    ovlp_start = max(seg_onset, slot_start)
    ovlp_end   = min(seg_end,   slot_end)
    if ovlp_end <= ovlp_start:
        return np.zeros((len(channels), 0), dtype=np.float32), 0

    within_file_s = ovlp_start - slot_start
    start_sample  = int(within_file_s * NATIVE_SR)
    n_frames      = int((ovlp_end - ovlp_start) * NATIVE_SR)

    audio          = read_polywav_window(wav_path, start_sample, n_frames, channels)
    offset_samples = int((ovlp_start - seg_onset) * NATIVE_SR)
    return audio, offset_samples


def assemble_segment(
    seg: dict,
    available: Dict[int, Path],
    channels: List[int],
) -> Optional[np.ndarray]:
    seg_onset   = float(seg["onset_from_rec_s"])
    seg_dur     = float(seg["duration_s"])
    need_native = int(seg_dur * NATIVE_SR)

    start_slot = int(seg_onset            / CHUNK_DUR_S)
    end_slot   = int((seg_onset + seg_dur) / CHUNK_DUR_S)

    n_ch = len(channels)
    buf  = np.zeros((n_ch, need_native), dtype=np.float32)
    any_data = False

    for slot in range(start_slot, end_slot + 1):
        if slot not in available:
            continue
        chunk, offset = extract_overlap(seg, available[slot], slot, channels)
        if chunk.shape[1] == 0:
            continue
        end_idx = min(offset + chunk.shape[1], need_native)
        src_len = end_idx - offset
        if src_len > 0:
            buf[:, offset:end_idx] = chunk[:, :src_len]
            any_data = True

    return buf if any_data else None


# ── FIX 2 + 5 + 6: clip_audio now purely a utility; clip_duration flows through ──

def clip_audio(
    audio_native: np.ndarray,
    clip_position: float,
    clip_dur_s: float,
) -> Tuple[np.ndarray, int]:
    """
    Slice ``audio_native`` to the requested clip.

    Parameters
    ──────────
    clip_position : 0.0 = start, 0.5 = centre, 1.0 = end
    clip_dur_s    : desired clip length in seconds.
                    0 (or negative) → return the FULL segment unchanged.

    Returns
    ───────
    (clipped_audio, start_sample_in_native)
        start_sample_in_native lets the caller compute clip_start_s exactly.
    """
    if clip_dur_s <= 0:
        # Full-segment mode: no clipping
        return audio_native, 0

    n_total  = audio_native.shape[1]
    n_want   = int(clip_dur_s * NATIVE_SR)

    if n_total <= n_want:
        return audio_native, 0

    max_start  = n_total - n_want
    start_samp = int(clip_position * max_start)
    return audio_native[:, start_samp : start_samp + n_want], start_samp


def resample_audio(audio_native: np.ndarray) -> np.ndarray:
    out = []
    for ch in audio_native:
        r = librosa.resample(ch, orig_sr=NATIVE_SR, target_sr=TARGET_SR)
        out.append(r.astype(np.float32))
    return np.stack(out, axis=0)


# ── Label writer ───────────────────────────────────────────────────────────────

def make_label_json(
    seg: dict,
    array: str,
    clip_start_s: float = 0.0,
    clip_dur_s: float = DEFAULT_CLIP_DUR_S,
) -> dict:
    trajectory = compute_clip_trajectory(seg, clip_start_s, clip_dur_s=clip_dur_s)
    summary    = trajectory["summary"]

    return {
        "drone": {
            "azimuth_deg":               summary["azimuth_at_mid_deg"],
            "distance_m":                summary["distance_xy_at_mid_m"],
            "height_m":                  summary["altitude_at_mid_m"],
            "azimuth_at_clip_start_deg": summary["azimuth_at_start_deg"],
            "azimuth_at_clip_end_deg":   summary["azimuth_at_end_deg"],
            "azimuth_swept_deg":         summary["azimuth_swept_deg"],
            "is_approaching":            summary["is_approaching"],
        },
        "trajectory":    trajectory,
        "segment_id":    int(seg["id"]),
        "session":       str(seg["session"]),
        "maneuver_type": str(seg["maneuver_type"]),
        "flight_phase":  str(seg["flight_phase"]) if seg.get("flight_phase") else None,
        "n_drones":      int(seg.get("n_drones", 1)),
        "drones":        list(seg.get("drones", [])),
        "split":         str(seg["split"]),
        "speed_mps":     float(seg["speed_mps"])  if seg.get("speed_mps")  is not None else None,
        "radius_m":      float(seg["radius_m"])   if seg.get("radius_m")   is not None else None,
        "duration_s":    float(seg["duration_s"]),
        "clip_start_s_in_seg": round(clip_start_s, 3),
        "clip_dur_s":    clip_dur_s if clip_dur_s > 0 else float(seg["duration_s"]),
        "array":         array,
        "onset_azimuth_deg":   float(seg["azimuth_deg_onset"])   if seg.get("azimuth_deg_onset")   is not None else None,
        "onset_distance_xy_m": float(seg["distance_xy_m_onset"]) if seg.get("distance_xy_m_onset") is not None else None,
        "onset_distance_3d_m": float(seg["distance_3d_m_onset"]) if seg.get("distance_3d_m_onset") is not None else None,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

DRONE_FREQ_MIN         = 30
DRONE_FREQ_MAX         = 300
ENERGY_RATIO_THRESHOLD = 0.03


def validate_extracted_segment(audio_path: Path) -> dict:
    try:
        audio, sr = sf.read(str(audio_path))
        # Use first 3 s for validation even if clip is longer
        max_samples = int(3.0 * sr)
        audio_v     = audio[:max_samples] if len(audio) > max_samples else audio

        fft   = np.abs(np.fft.rfft(audio_v))
        freqs = np.fft.rfftfreq(len(audio_v), 1 / sr)

        drone_mask = (freqs >= DRONE_FREQ_MIN) & (freqs <= DRONE_FREQ_MAX)

        if np.any(drone_mask):
            peak_idx     = np.argmax(fft[drone_mask])
            dom_freq     = freqs[drone_mask][peak_idx]
            drone_energy = np.sum(fft[drone_mask] ** 2)
        else:
            dom_freq     = float("nan")
            drone_energy = 0.0

        total_energy = np.sum(fft ** 2)
        energy_ratio = drone_energy / total_energy if total_energy > 0 else 0.0
        rms_db       = 20 * np.log10(np.sqrt(np.mean(audio_v ** 2)) + 1e-8)

        is_valid = bool(
            not math.isnan(dom_freq)
            and DRONE_FREQ_MIN <= dom_freq <= DRONE_FREQ_MAX
            and rms_db > -40
            and energy_ratio > ENERGY_RATIO_THRESHOLD
        )

        return {
            "valid":        is_valid,
            "dom_freq_hz":  float(dom_freq) if not math.isnan(dom_freq) else None,
            "rms_db":       float(rms_db),
            "energy_ratio": float(energy_ratio),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_from_wav_dir(
    segments_json:   Path,
    wav_dir:         Path,
    output_dir:      Path,
    array:           str   = DEFAULT_ARRAY,
    clip_position:   float = 0.5,
    clip_duration:   float = DEFAULT_CLIP_DUR_S,   # ← BUG 5 FIX: explicit arg, not global args
    splits:          Optional[List[str]] = None,
    skip_unlabelled: bool  = False,
    validate:        bool  = True,
    verbose:         bool  = True,
) -> List[dict]:

    polywav_channels = ARRAY_CHANNELS[array]

    with open(segments_json) as f:
        all_segments: List[dict] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    available: Dict[int, Path] = {}
    for wp in sorted(wav_dir.glob("*.wav")):
        slot = wav_slot(wp)
        if slot is None:
            if verbose:
                print(f"  [skip] unrecognised filename: {wp.name}")
            continue
        available[slot] = wp

    if not available:
        print(f"ERROR: no recognised polywav files in {wav_dir}")
        sys.exit(1)

    full_seg_mode = (clip_duration <= 0)

    if verbose:
        print(f"\nArray          : {array}  (polywav columns {polywav_channels})")
        if full_seg_mode:
            print(f"Clip duration  : FULL SEGMENT (clip_duration=0)")
        else:
            print(f"Clip duration  : {clip_duration:.2f} s")
            print(f"Clip position  : {clip_position:.2f}  (0=start, 0.5=centre, 1=end)")
        print(f"\nFound {len(available)} polywav file(s):")
        for slot in sorted(available):
            s, e = slot_time_range(slot)
            print(f"  slot {slot:2d}  ({s:7.0f}–{e:7.0f} s / local {slot_local_hms(slot)})  {available[slot].name}")

    covered_slots = set(available.keys())

    candidates = []
    for seg in all_segments:
        if splits and seg.get("split") not in splits:
            continue
        if skip_unlabelled and seg.get("azimuth_deg_onset") is None:
            continue
        onset  = float(seg["onset_from_rec_s"])
        dur    = float(seg["duration_s"])
        ss     = int(onset        / CHUNK_DUR_S)
        es     = int((onset + dur) / CHUNK_DUR_S)
        needed = set(range(ss, es + 1))
        if needed & covered_slots:
            candidates.append(seg)

    if verbose:
        print(f"\n{len(candidates)} segment(s) overlap the provided file(s)"
              + (f"  [splits={splits}]" if splits else ""))

    if not candidates:
        print("  No WAV file(s) cover any segments in the requested splits.")
        return []

    manifest:    List[dict] = []
    skipped      = 0
    val_results  = []

    for seg in candidates:
        sid        = seg["id"]
        split      = seg.get("split", "unknown")
        maneuver   = seg.get("maneuver_type", "unknown")
        session_id = f"seg_{sid:03d}_{maneuver}_{split}"
        flags      = list(seg.get("quality_flags", []))
        onset      = float(seg["onset_from_rec_s"])
        dur        = float(seg["duration_s"])

        if verbose:
            print(f"  → {session_id}  onset={onset:.1f}s  dur={dur:.1f}s", end="  ")

        ss     = int(onset        / CHUNK_DUR_S)
        es     = int((onset + dur) / CHUNK_DUR_S)
        needed = set(range(ss, es + 1))
        missing = needed - covered_slots
        if missing:
            flags.append("partial_coverage")
            if verbose:
                print(f"[partial: slots {sorted(missing)} absent]", end="  ")

        # Assemble full segment at native SR
        audio_native = assemble_segment(seg, available, polywav_channels)
        if audio_native is None:
            if verbose:
                print("SKIP (no audio)")
            skipped += 1
            continue

        # ── BUG 6 + 7 FIX: clip_audio returns (audio, start_sample) ──────────
        if full_seg_mode:
            # Keep everything; skip the distance/energy window search entirely
            audio_clipped   = audio_native
            clip_start_samp = 0
            clip_start_s    = 0.0
        else:
            seg_dur_native = audio_native.shape[1]
            n_want_native  = int(clip_duration * NATIVE_SR)

            if seg_dur_native > n_want_native:
                # Find best window: closest approach ± energy refinement
                approx_start_s    = find_best_window_by_distance(seg, clip_duration)
                approx_start_samp = int(approx_start_s * NATIVE_SR)

                search_radius = int(2 * NATIVE_SR)
                sub_start     = max(0, approx_start_samp - search_radius)
                sub_end       = min(audio_native.shape[1], approx_start_samp + search_radius)
                sub_audio     = audio_native[:, sub_start:sub_end]

                best_local      = find_best_window_by_energy(sub_audio, clip_duration)
                clip_start_samp = sub_start + best_local
            else:
                clip_start_samp = 0

            clip_start_s  = clip_start_samp / NATIVE_SR
            audio_clipped, _ = clip_audio(audio_native, clip_position, clip_duration)
            # Override with the distance+energy–selected start (more accurate)
            n_want = int(clip_duration * NATIVE_SR)
            audio_clipped = audio_native[:, clip_start_samp : clip_start_samp + n_want]
            # Pad if edge of recording
            if audio_clipped.shape[1] < n_want:
                pad = np.zeros((audio_clipped.shape[0], n_want - audio_clipped.shape[1]),
                               dtype=np.float32)
                audio_clipped = np.concatenate([audio_clipped, pad], axis=1)

        # ── actual_clip_dur_s: truth, not assumption ───────────────────────────
        actual_clip_dur_s = audio_clipped.shape[1] / NATIVE_SR

        # Resample to model SR
        audio_model = resample_audio(audio_clipped)

        rms_vals = [float(np.sqrt(np.mean(ch ** 2))) for ch in audio_model]
        max_rms  = max(rms_vals)

        if max_rms < 1e-7:
            if verbose:
                print("SKIP (silent — file does not cover this segment)")
            skipped += 1
            continue

        # Write WAV files (one per channel)
        for i, ch in enumerate(audio_model):
            sf.write(
                str(output_dir / f"{session_id}_ch{i}.wav"),
                ch, TARGET_SR, subtype="FLOAT",
            )

        # Write label JSON — clip_dur_s now reflects actual extracted length
        label = make_label_json(seg, array, clip_start_s, actual_clip_dur_s)
        with open(output_dir / f"{session_id}_label.json", "w") as f:
            json.dump(label, f, indent=2)

        rms_db = round(20 * np.log10(max_rms + 1e-10), 1)

        validation = None
        if validate:
            ch0_path   = output_dir / f"{session_id}_ch0.wav"
            validation = validate_extracted_segment(ch0_path)
            val_results.append(validation)
            status_icon = "✅" if validation.get("valid") else "⚠️"
            freq_str    = f"{validation.get('dom_freq_hz', 0):.1f}Hz" if validation.get("dom_freq_hz") else "?"
            ratio_str   = f"ratio={validation.get('energy_ratio', 0):.3f}"
            if verbose:
                print(f"✓  {actual_clip_dur_s:.1f}s  rms={rms_db} dB  {status_icon} {freq_str} {ratio_str}")
            if not validation.get("valid"):
                flags.append("validation_warning")
        else:
            if verbose:
                print(f"✓  {actual_clip_dur_s:.1f}s  rms={rms_db} dB")

        traj_summary = label["trajectory"]["summary"]

        entry = {
            "session_id":       str(session_id),
            "segment_id":       int(sid),
            "session":          str(seg["session"]),
            "split":            str(split),
            "maneuver_type":    str(maneuver),
            "flight_phase":     str(seg.get("flight_phase")) if seg.get("flight_phase") else None,
            "n_drones":         int(seg.get("n_drones", 1)),
            "drones":           list(seg.get("drones", [])),
            "onset_from_rec_s": float(onset),
            "duration_s":       float(dur),
            # ── clip window ────────────────────────────────────────────────
            "clip_start_s":     round(clip_start_s, 3),
            "clip_dur_s":       round(actual_clip_dur_s, 3),    # BUG 6 FIX
            "clip_position":    float(clip_position) if not full_seg_mode else None,
            # ── trajectory summary ─────────────────────────────────────────
            "azimuth_at_clip_start_deg": traj_summary["azimuth_at_start_deg"],
            "azimuth_at_clip_mid_deg":   traj_summary["azimuth_at_mid_deg"],
            "azimuth_at_clip_end_deg":   traj_summary["azimuth_at_end_deg"],
            "azimuth_swept_deg":         traj_summary["azimuth_swept_deg"],
            "distance_xy_at_mid_m":      traj_summary["distance_xy_at_mid_m"],
            "distance_3d_at_mid_m":      traj_summary["distance_3d_at_mid_m"],
            "altitude_at_mid_m":         traj_summary["altitude_at_mid_m"],
            "is_approaching":            traj_summary["is_approaching"],
            "trajectory_source":         label["trajectory"]["trajectory_source"],
            # ── legacy onset snapshot ──────────────────────────────────────
            "onset_azimuth_deg":   float(seg["azimuth_deg_onset"])   if seg.get("azimuth_deg_onset")   is not None else None,
            "onset_distance_xy_m": float(seg["distance_xy_m_onset"]) if seg.get("distance_xy_m_onset") is not None else None,
            "onset_distance_3d_m": float(seg["distance_3d_m_onset"]) if seg.get("distance_3d_m_onset") is not None else None,
            # ── segment properties ─────────────────────────────────────────
            "altitude_m":       float(seg["altitude_m"])  if seg.get("altitude_m")  is not None else None,
            "speed_mps":        float(seg["speed_mps"])   if seg.get("speed_mps")   is not None else None,
            "radius_m":         float(seg["radius_m"])    if seg.get("radius_m")    is not None else None,
            # ── audio quality ──────────────────────────────────────────────
            "rms_ch0":          float(rms_vals[0]),
            "rms_ch1":          float(rms_vals[1]),
            "rms_ch2":          float(rms_vals[2]),
            "rms_max_db":       float(rms_db),
            "array":            array,
            "polywav_channels": [int(c) for c in polywav_channels],
            "quality_flags":    list(flags),
            "source_files":     [available[s].name for s in sorted(needed & covered_slots)],
            "wav_ch0":          f"{session_id}_ch0.wav",
            "wav_ch1":          f"{session_id}_ch1.wav",
            "wav_ch2":          f"{session_id}_ch2.wav",
            "label_json":       f"{session_id}_label.json",
            "validation":       validation if validate else None,
        }
        manifest.append(entry)

    if verbose:
        print(f"\n✅ Extracted {len(manifest)} segments  ({skipped} skipped)")
        if validate and val_results:
            n_valid = sum(1 for v in val_results if v.get("valid"))
            print(f"   Validation (relaxed threshold={ENERGY_RATIO_THRESHOLD}): "
                  f"{n_valid}/{len(val_results)} passed")
            if n_valid < len(val_results):
                print(f"   ⚠️  {len(val_results)-n_valid} flagged — but NOT dropped (ground truth).")

    return manifest


# ── Output writers ─────────────────────────────────────────────────────────────

def write_labels_csv(manifest: List[dict], output_dir: Path):
    import csv
    fieldnames = [
        "session_id",
        "azimuth_at_clip_start_deg", "azimuth_at_clip_mid_deg", "azimuth_at_clip_end_deg",
        "azimuth_swept_deg", "distance_xy_at_mid_m", "altitude_at_mid_m",
        "is_approaching", "trajectory_source",
        "clip_start_s", "clip_dur_s", "clip_position",
        "split", "maneuver_type", "n_drones", "altitude_m", "speed_mps", "radius_m",
        "array",
        "onset_azimuth_deg", "onset_distance_xy_m",
        "validation_valid", "validation_dom_freq_hz", "validation_energy_ratio",
    ]
    with open(output_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in manifest:
            val = r.get("validation") or {}
            row = {k: r.get(k, "") for k in fieldnames}
            row["validation_valid"]        = val.get("valid", "")
            row["validation_dom_freq_hz"]  = val.get("dom_freq_hz", "")
            row["validation_energy_ratio"] = val.get("energy_ratio", "")
            w.writerow(row)
    print("📋 labels.csv written")


def write_manifest(manifest: List[dict], output_dir: Path):
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 manifest.json written  ({len(manifest)} segments)")


def create_zip(output_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(output_dir.iterdir()):
            if fp.is_file():
                zf.write(fp, arcname=fp.name)
    mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"📦 ZIP → {zip_path}  ({mb:.1f} MB)")


def print_summary(manifest: List[dict]):
    from collections import Counter
    print("\n" + "═" * 65)
    print("  EXTRACTION SUMMARY")
    print("═" * 65)
    print(f"  Total extracted : {len(manifest)} segments")
    for sp, n in sorted(Counter(r["split"] for r in manifest).items()):
        print(f"    {sp:8s}: {n}")

    print(f"\n  Maneuver types:")
    for m, n in sorted(Counter(r["maneuver_type"] for r in manifest).items(), key=lambda x: -x[1]):
        print(f"    {m:20s}: {n}")

    clip_durs = [r["clip_dur_s"] for r in manifest]
    print(f"\n  Clip duration (s): min={min(clip_durs):.1f}  max={max(clip_durs):.1f}  "
          f"mean={sum(clip_durs)/len(clip_durs):.1f}")

    approaching = sum(1 for r in manifest if r.get("is_approaching"))
    print(f"\n  Trajectory (analytic, 10 Hz):")
    print(f"    Approaching at clip end  : {approaching}/{len(manifest)}")
    swept = [abs(r["azimuth_swept_deg"]) for r in manifest if r.get("azimuth_swept_deg") is not None]
    if swept:
        print(f"    Azimuth swept (|°|)      : min={min(swept):.1f}  max={max(swept):.1f}  "
              f"mean={sum(swept)/len(swept):.1f}")

    print(f"\n  Source files:")
    for f in sorted({f for r in manifest for f in r["source_files"]}):
        n = sum(1 for r in manifest if f in r["source_files"])
        print(f"    {f}  ({n} seg{'s' if n!=1 else ''})")

    rms_dbs = [r["rms_max_db"] for r in manifest]
    print(f"\n  RMS range            : {min(rms_dbs):.1f} … {max(rms_dbs):.1f} dB")
    validated = [r for r in manifest if r.get("validation") is not None]
    if validated:
        n_valid = sum(1 for r in validated if r["validation"].get("valid"))
        print(f"  Validation passed    : {n_valid}/{len(validated)}  "
              f"(threshold={ENERGY_RATIO_THRESHOLD}; flagged but not dropped)")
    print("═" * 65)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Dunakeszi ground-truth segments (fixed v2)"
    )
    ap.add_argument("--segments",    required=True,
                    help="Path to ground_truth_segments.json")
    ap.add_argument("--wav-dir",     required=True,
                    help="Directory containing polywav .wav file(s)")
    ap.add_argument("--output-dir",  default="dunakeszi_test_segments",
                    help="Output directory (default: dunakeszi_test_segments)")
    ap.add_argument("--array",       choices=list(ARRAY_CHANNELS), default=DEFAULT_ARRAY,
                    help=f"Which mic array to extract (default: {DEFAULT_ARRAY}). "
                         f"BK-6-E = East (Scorpio ch 9-11), BK-6-W = West (ch 3-5)")
    ap.add_argument("--clip-position", type=float, default=0.5,
                    help="Where in the segment to take the clip: "
                         "0.0=start, 0.5=centre (default), 1.0=end.  "
                         "Ignored when --clip-duration 0 (full segment mode).")
    ap.add_argument("--clip-duration", type=float, default=DEFAULT_CLIP_DUR_S,
                    help=f"Duration of extracted clip in seconds (default: {DEFAULT_CLIP_DUR_S}). "
                         "Set to 0 to keep the FULL segment (no clipping). "
                         "In full-segment mode --clip-position is ignored.")
    ap.add_argument("--splits",      nargs="+", choices=["train", "val", "test"],
                    default=None, help="Restrict to specific splits")
    ap.add_argument("--no-zip",      action="store_true")
    ap.add_argument("--skip-unlabelled", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    if not 0.0 <= args.clip_position <= 1.0:
        print("ERROR: --clip-position must be in [0, 1]")
        sys.exit(1)

    segments_json = Path(args.segments)
    wav_dir       = Path(args.wav_dir)
    output_dir    = Path(args.output_dir)

    for p, name in [(segments_json, "segments JSON"), (wav_dir, "wav directory")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}"); sys.exit(1)

    full_seg_mode    = (args.clip_duration <= 0)
    polywav_channels = ARRAY_CHANNELS[args.array]

    print("═" * 65)
    print("  Dunakeszi Ground-Truth Segment Extractor (fixed v2)")
    print("═" * 65)
    print(f"  Segments JSON  : {segments_json}")
    print(f"  WAV directory  : {wav_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Array          : {args.array}  (polywav cols {polywav_channels})")
    if full_seg_mode:
        print(f"  Clip mode      : FULL SEGMENT (clip_duration=0, no clipping)")
    else:
        print(f"  Clip duration  : {args.clip_duration} s")
        print(f"  Clip position  : {args.clip_position}  (0=start, 0.5=centre, 1=end)")
    print(f"  Splits         : {args.splits or 'all'}")
    print(f"  Resample       : {NATIVE_SR} Hz → {TARGET_SR} Hz")
    print(f"  Trajectory     : analytic 10 Hz over full clip duration")
    print(f"  Validate audio : {'NO' if args.no_validate else f'YES  (energy_ratio > {ENERGY_RATIO_THRESHOLD}, flagged only)'}")

    if args.dry_run:
        with open(segments_json) as f:
            all_segs = json.load(f)
        available = {wav_slot(wp): wp for wp in sorted(wav_dir.glob("*.wav"))
                     if wav_slot(wp) is not None}
        print(f"\nDRY RUN — {len(available)} file(s), slots {sorted(available.keys())}")
        for seg in all_segs:
            if args.splits and seg.get("split") not in args.splits:
                continue
            onset = float(seg["onset_from_rec_s"])
            dur   = float(seg["duration_s"])
            ss    = int(onset        / CHUNK_DUR_S)
            es    = int((onset + dur) / CHUNK_DUR_S)
            slots = set(range(ss, es + 1))
            has   = bool(slots & set(available.keys()))
            traj_note = ""
            if has:
                mid_t   = dur / 2 if full_seg_mode else (
                    args.clip_position * max(0, dur - args.clip_duration)
                    + (args.clip_duration if args.clip_duration > 0 else dur) / 2
                )
                x, y, z = interpolate_position(seg, mid_t)
                bearing  = _pos_to_bearing(x, y, z)
                traj_note = (f"  az={bearing['azimuth_deg']:.0f}°"
                             f"  d={bearing['distance_xy_m']:.0f}m"
                             f"  h={z:.0f}m")
            print(f"  seg_{seg['id']:03d}_{seg['maneuver_type']:15s} "
                  f"slots={sorted(slots)}  dur={dur:.0f}s  "
                  f"{'✓ extractable' if has else '-- not covered'}"
                  f"{traj_note}")
        return

    # ── BUG 5 FIX: pass clip_duration explicitly, never rely on global args ───
    manifest = extract_from_wav_dir(
        segments_json   = segments_json,
        wav_dir         = wav_dir,
        output_dir      = output_dir,
        array           = args.array,
        clip_position   = args.clip_position,
        clip_duration   = args.clip_duration,   # ← explicit
        splits          = args.splits,
        skip_unlabelled = args.skip_unlabelled,
        validate        = not args.no_validate,
        verbose         = True,
    )

    if not manifest:
        print("\nNothing extracted.  Use --dry-run to preview coverage.")
        sys.exit(0)

    write_labels_csv(manifest, output_dir)
    write_manifest(manifest, output_dir)
    print_summary(manifest)

    if not args.no_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        create_zip(output_dir, zip_path)


if __name__ == "__main__":
    main()

    # ── Recommended usage ─────────────────────────────────────────────────────
    #
    # Full segment mode (no clipping — drone airborne duration preserved):
    #   python dunakeszi_segment_extractor_fixed.py --segments new_ground_truth/ground_truth_segments.json --wav-dir wavs/ --array BK-6-E --clip-duration 0
    #
    # Standard 3-second clip from centre (original behaviour, now correct):
    #   python dunakeszi_segment_extractor_fixed.py --segments new_ground_truth/ground_truth_segments.json --wav-dir wavs/ --array BK-6-E --clip-duration 3 --clip-position 0.5
    #
    # Custom 10-second clip from closest-approach window:
    #   python dunakeszi_segment_extractor_fixed.py --segments new_ground_truth/ground_truth_segments.json --wav-dir wavs/ --array BK-6-E --clip-duration 10 --clip-position 0.5
    #
    # Dry run — preview coverage and mid-clip trajectory:
    #   python dunakeszi_segment_extractor_fixed.py --segments new_ground_truth/ground_truth_segments.json --wav-dir wavs/ --dry-run