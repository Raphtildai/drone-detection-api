#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dunakeszi_segment_extractor_fixed.py
──────────────────────────────────────
Fixed version of the Dunakeszi ground-truth segment extractor.

Bugs fixed vs original:
  BUG 1 — POLYWAV_CHANNELS was [2,3,4] (BK-6-W / West array, Scorpio ch 3-5).
           Changed to [8,9,10] = BK-6-E / East array (Scorpio ch 9-11),
           matching the scorpio_channel_map.json TDOA subset and the labels
           already used in validate_extracted_segments.py ("BK-6-E").
           Add --array {BK-6-W,BK-6-E} CLI flag so callers can choose.

  BUG 2 — The 3-second clip was always taken from the start (onset) of each
           segment.  For transit / circle / formation manoeuvres the drone can
           be 60–300 m away at onset, producing near-silence.
           Fix: sample from the segment CENTRE by default.  Add --clip-offset
           {start,center,end,<float 0-1>} to make it explicit.

  BUG 3 — Validation threshold energy_ratio > 0.30 is far too strict for
           outdoor recordings where wind / 1/f noise dominates.  Real drone
           recordings in open fields typically show energy ratios of 0.03–0.15
           in the 30–300 Hz band.  Changed to > 0.03.  Validation is only a
           quality flag now — it never silently drops a segment that has audio.
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed.  Run: pip install librosa")
    sys.exit(1)

# ── Audio constants ────────────────────────────────────────────────────────────

NATIVE_SR      = 192_000
TARGET_SR      = 22_050
TARGET_DUR_S   = 3.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DUR_S)   # 66 150

# Exact 4 GB polywav → duration per file
_BYTES_PER_FRAME = 14 * 4          # 14 channels × float32
CHUNK_DUR_S      = (4 * 1024**3) / (_BYTES_PER_FRAME * NATIVE_SR)  # ≈ 399.4613 s

# ── FIX 1: channel mappings for both arrays ────────────────────────────────────
# Scorpio channel numbers are 1-indexed; polywav columns are 0-indexed.
# Scorpio ch N  →  polywav column N-1.
#
#   BK-6-W (West):  Scorpio ch 3,4,5   → polywav columns [2,3,4]
#   BK-6-E (East):  Scorpio ch 9,10,11 → polywav columns [8,9,10]  ← TDOA subset
#
# The original code used [2,3,4] (West) but the rest of the pipeline
# (validate_extracted_segments.py labels, scorpio_channel_map tdoa_channels)
# refers to BK-6-E as the primary array.  Default changed to East.
ARRAY_CHANNELS = {
    "BK-6-E": [8, 9, 10],   # East array — DEFAULT (was bug: West used instead)
    "BK-6-W": [2, 3, 4],    # West array
}
DEFAULT_ARRAY = "BK-6-E"

# Recording reference (onset_from_rec_s = 0 corresponds to this local time)
REC_REF_LOCAL_S = 13 * 3600 + 36 * 60   # 48 960 s


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


# ── Audio I/O ──────────────────────────────────────────────────────────────────

def read_polywav_window(
    wav_path: Path,
    start_sample: int,
    n_frames: int,
    channels: List[int],
) -> np.ndarray:
    info = sf.info(str(wav_path))
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

    audio = read_polywav_window(wav_path, start_sample, n_frames, channels)
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

    start_slot = int(seg_onset          / CHUNK_DUR_S)
    end_slot   = int((seg_onset+seg_dur) / CHUNK_DUR_S)

    n_ch = len(channels)
    buf = np.zeros((n_ch, need_native), dtype=np.float32)
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


# ── FIX 2: configurable clip position ─────────────────────────────────────────
# Original always clipped from sample 0 (segment onset).
# Many long segments (transits, circles) have the drone far away at onset.
# Default changed to 0.5 = centre of segment, where the drone is closest on average.

def clip_audio(audio_native: np.ndarray, clip_position: float = 0.5) -> np.ndarray:
    """
    Select a TARGET_DUR_S window from audio_native.

    clip_position : float in [0, 1]
        0.0 = start of segment (original behaviour)
        0.5 = centre (new default — best for transits / circles)
        1.0 = end of segment
    """
    n_total = audio_native.shape[1]
    n_want  = int(TARGET_DUR_S * NATIVE_SR)

    if n_total <= n_want:
        # Shorter than 3 s — just pad
        return audio_native

    # Start sample for the clip window
    max_start = n_total - n_want
    start     = int(clip_position * max_start)
    return audio_native[:, start : start + n_want]


def resample_and_pad(audio_native: np.ndarray) -> np.ndarray:
    """(N_ch, n) @ NATIVE_SR → (N_ch, TARGET_SAMPLES) @ TARGET_SR, float32."""
    out = []
    for ch in audio_native:
        r = librosa.resample(ch, orig_sr=NATIVE_SR, target_sr=TARGET_SR)
        if len(r) >= TARGET_SAMPLES:
            r = r[:TARGET_SAMPLES]
        else:
            r = np.pad(r, (0, TARGET_SAMPLES - len(r)))
        out.append(r.astype(np.float32))
    return np.stack(out, axis=0)


# ── Label writer ───────────────────────────────────────────────────────────────

def make_label_json(seg: dict, array: str) -> dict:
    az   = seg.get("azimuth_deg_onset")   or 0.0
    dist = seg.get("distance_xy_m_onset") or 0.0
    ht   = seg.get("altitude_m") or seg.get("distance_3d_m_onset") or 0.0

    return {
        "drone": {
            "azimuth":  float(az),
            "distance": float(dist),
            "height":   float(ht),
        },
        "segment_id":    int(seg["id"]),
        "session":       str(seg["session"]),
        "maneuver_type": str(seg["maneuver_type"]),
        "flight_phase":  str(seg["flight_phase"]) if seg.get("flight_phase") else None,
        "n_drones":      int(seg.get("n_drones", 1)),
        "split":         str(seg["split"]),
        "speed_mps":     float(seg["speed_mps"])  if seg.get("speed_mps")  is not None else None,
        "radius_m":      float(seg["radius_m"])   if seg.get("radius_m")   is not None else None,
        "duration_s":    float(seg["duration_s"]),
        "array":         array,
    }


# ── FIX 3: validation — relaxed threshold, never drops segments ────────────────
# Original: energy_ratio > 0.30 — far too strict for outdoor open-field recordings.
# Fixed:    energy_ratio > 0.03 — realistic for broadband ambient + drone.
# Also: validation result is a quality flag only; it NEVER causes a segment skip.
# The segment is ground-truth data — the drone IS there by definition.

DRONE_FREQ_MIN = 30    # Hz
DRONE_FREQ_MAX = 300   # Hz
ENERGY_RATIO_THRESHOLD = 0.03   # was 0.30 — relaxed for outdoor conditions


def validate_extracted_segment(audio_path: Path) -> dict:
    try:
        audio, sr = sf.read(str(audio_path))

        fft   = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        drone_mask = (freqs >= DRONE_FREQ_MIN) & (freqs <= DRONE_FREQ_MAX)

        if np.any(drone_mask):
            peak_idx  = np.argmax(fft[drone_mask])
            dom_freq  = freqs[drone_mask][peak_idx]
            drone_energy = np.sum(fft[drone_mask] ** 2)
        else:
            dom_freq     = np.nan
            drone_energy = 0.0

        total_energy = np.sum(fft ** 2)
        energy_ratio = drone_energy / total_energy if total_energy > 0 else 0.0
        rms_db       = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-8)

        is_valid = bool(
            not np.isnan(dom_freq)
            and DRONE_FREQ_MIN <= dom_freq <= DRONE_FREQ_MAX
            and rms_db > -40
            and energy_ratio > ENERGY_RATIO_THRESHOLD   # relaxed from 0.30 → 0.03
        )

        return {
            "valid":        is_valid,
            "dom_freq_hz":  float(dom_freq) if not np.isnan(dom_freq) else None,
            "rms_db":       float(rms_db),
            "energy_ratio": float(energy_ratio),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_from_wav_dir(
    segments_json:  Path,
    wav_dir:        Path,
    output_dir:     Path,
    array:          str  = DEFAULT_ARRAY,
    clip_position:  float = 0.5,
    splits:         Optional[List[str]] = None,
    skip_unlabelled: bool = False,
    validate:       bool = True,
    verbose:        bool = True,
) -> List[dict]:

    polywav_channels = ARRAY_CHANNELS[array]

    with open(segments_json) as f:
        all_segments: List[dict] = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover WAV files
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

    if verbose:
        print(f"\nArray          : {array}  (polywav columns {polywav_channels})")
        print(f"Clip position  : {clip_position:.2f}  (0=start, 0.5=centre, 1=end)")
        print(f"\nFound {len(available)} polywav file(s):")
        for slot in sorted(available):
            s, e = slot_time_range(slot)
            print(f"  slot {slot:2d}  ({s:7.0f}–{e:7.0f} s / local {slot_local_hms(slot)})  {available[slot].name}")

    covered_slots = set(available.keys())

    # Filter segments
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

    # Extract
    manifest: List[dict] = []
    skipped   = 0
    val_results = []

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

        # Partial coverage check
        ss     = int(onset       / CHUNK_DUR_S)
        es     = int((onset+dur) / CHUNK_DUR_S)
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

        # FIX 2: clip from configurable position (default = centre)
        audio_clipped = clip_audio(audio_native, clip_position)

        # Resample to model SR
        audio_model = resample_and_pad(audio_clipped)

        rms_vals = [float(np.sqrt(np.mean(ch ** 2))) for ch in audio_model]
        max_rms  = max(rms_vals)

        if max_rms < 1e-7:
            if verbose:
                print("SKIP (silent — file does not cover this segment)")
            skipped += 1
            continue

        # Write WAV files
        for i, ch in enumerate(audio_model):
            sf.write(str(output_dir / f"{session_id}_ch{i}.wav"),
                     ch, TARGET_SR, subtype="FLOAT")

        with open(output_dir / f"{session_id}_label.json", "w") as f:
            json.dump(make_label_json(seg, array), f, indent=2)

        rms_db = round(20 * np.log10(max_rms + 1e-10), 1)

        # FIX 3: validate but NEVER skip — just record result as a flag
        validation = None
        if validate:
            ch0_path = output_dir / f"{session_id}_ch0.wav"
            validation = validate_extracted_segment(ch0_path)
            val_results.append(validation)

            status_icon = "✅" if validation.get("valid") else "⚠️"
            freq_str    = f"{validation.get('dom_freq_hz', 0):.1f}Hz" if validation.get('dom_freq_hz') else "?"
            ratio_str   = f"ratio={validation.get('energy_ratio', 0):.3f}"
            if verbose:
                print(f"✓  rms={rms_db} dB  {status_icon} {freq_str} {ratio_str}")

            if not validation.get("valid"):
                flags.append("validation_warning")
                # NOTE: we do NOT skip — ground truth says drone is present
        else:
            if verbose:
                print(f"✓  rms={rms_db} dB")

        # Clip timing metadata
        seg_dur_native = audio_native.shape[1]
        clip_start_s   = (clip_position * max(0, seg_dur_native - int(TARGET_DUR_S * NATIVE_SR))) / NATIVE_SR
        clip_start_s   = round(clip_start_s, 3)

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
            "clip_start_s":     float(clip_start_s),    # offset into segment where 3s clip starts
            "clip_position":    float(clip_position),
            "altitude_m":       float(seg["altitude_m"])          if seg.get("altitude_m")          is not None else None,
            "speed_mps":        float(seg["speed_mps"])           if seg.get("speed_mps")            is not None else None,
            "radius_m":         float(seg["radius_m"])            if seg.get("radius_m")             is not None else None,
            "azimuth_deg":      float(seg["azimuth_deg_onset"])   if seg.get("azimuth_deg_onset")    is not None else None,
            "distance_xy_m":    float(seg["distance_xy_m_onset"]) if seg.get("distance_xy_m_onset")  is not None else None,
            "distance_3d_m":    float(seg["distance_3d_m_onset"]) if seg.get("distance_3d_m_onset")  is not None else None,
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
        "session_id", "azimuth_deg", "distance_m", "height_m",
        "split", "maneuver_type", "n_drones", "altitude_m", "speed_mps",
        "clip_start_s", "clip_position", "array",
        "validation_valid", "validation_dom_freq_hz", "validation_energy_ratio",
    ]
    with open(output_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in manifest:
            val = r.get("validation") or {}
            w.writerow({
                "session_id":               r["session_id"],
                "azimuth_deg":              r["azimuth_deg"]   if r["azimuth_deg"]   is not None else "",
                "distance_m":               r["distance_xy_m"] if r["distance_xy_m"] is not None else "",
                "height_m":                 r["altitude_m"]    if r["altitude_m"]    is not None else "",
                "split":                    r["split"],
                "maneuver_type":            r["maneuver_type"],
                "n_drones":                 r["n_drones"],
                "altitude_m":               r["altitude_m"]    if r["altitude_m"]    is not None else "",
                "speed_mps":                r["speed_mps"]     if r["speed_mps"]     is not None else "",
                "clip_start_s":             r.get("clip_start_s", ""),
                "clip_position":            r.get("clip_position", ""),
                "array":                    r.get("array", ""),
                "validation_valid":         val.get("valid", ""),
                "validation_dom_freq_hz":   val.get("dom_freq_hz", ""),
                "validation_energy_ratio":  val.get("energy_ratio", ""),
            })
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
    print(f"\n  Source files:")
    for f in sorted({f for r in manifest for f in r["source_files"]}):
        n = sum(1 for r in manifest if f in r["source_files"])
        print(f"    {f}  ({n} seg{'s' if n!=1 else ''})")
    labelled = sum(1 for r in manifest if r["azimuth_deg"] is not None)
    print(f"\n  Labelled (az+dist+ht): {labelled}/{len(manifest)}")
    rms_dbs = [r["rms_max_db"] for r in manifest]
    print(f"  RMS range            : {min(rms_dbs):.1f} … {max(rms_dbs):.1f} dB")
    validated = [r for r in manifest if r.get("validation") is not None]
    if validated:
        n_valid = sum(1 for r in validated if r["validation"].get("valid"))
        print(f"  Validation passed    : {n_valid}/{len(validated)}  "
              f"(threshold={ENERGY_RATIO_THRESHOLD}; flagged but not dropped)")
    print("═" * 65)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract Dunakeszi ground-truth segments (fixed version)"
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
                    help="Where in the segment to take the 3-second clip: "
                         "0.0=start, 0.5=centre (default), 1.0=end.  "
                         "For transits/circles 0.5 picks the closest-approach window.")
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

    polywav_channels = ARRAY_CHANNELS[args.array]

    print("═" * 65)
    print("  Dunakeszi Ground-Truth Segment Extractor (fixed)")
    print("═" * 65)
    print(f"  Segments JSON  : {segments_json}")
    print(f"  WAV directory  : {wav_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Array          : {args.array}  (polywav cols {polywav_channels})")
    print(f"  Clip position  : {args.clip_position}  (0=start, 0.5=centre, 1=end)")
    print(f"  Splits         : {args.splits or 'all'}")
    print(f"  Resample       : {NATIVE_SR} Hz → {TARGET_SR} Hz")
    print(f"  Output length  : {TARGET_DUR_S} s  ({TARGET_SAMPLES} samples)")
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
            print(f"  seg_{seg['id']:03d}_{seg['maneuver_type']:15s} "
                  f"slots={sorted(slots)}  dur={dur:.0f}s  "
                  f"{'✓ extractable' if has else '-- not covered'}")
        return

    manifest = extract_from_wav_dir(
        segments_json  = segments_json,
        wav_dir        = wav_dir,
        output_dir     = output_dir,
        array          = args.array,
        clip_position  = args.clip_position,
        splits         = args.splits,
        skip_unlabelled= args.skip_unlabelled,
        validate       = not args.no_validate,
        verbose        = True,
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

    # Recommended usage:
    # python dunakeszi_segment_extractor_fixed.py \
    #   --segments ground_truth/ground_truth_segments.json \
    #   --wav-dir wavs/ \
    #   --array BK-6-E \
    #   --clip-position 0.5
    #
    # To compare both arrays:
    # python dunakeszi_segment_extractor_fixed.py --segments ground_truth/ground_truth_segments.json --wav-dir wavs/ --array BK-6-E --output-dir segs_E --clip-position 0.5
    # python dunakeszi_segment_extractor_fixed.py --segments ground_truth/ground_truth_segments.json --wav-dir wavs/ ... --array BK-6-W --output-dir segs_W --clip-position 0.5
    #
    # For hover segments where onset == closest approach, use --clip-position 0.0
    # For transit/circle segments, keep the default --clip-position 0.5

    # For 251020VITEMOROM1AT01J.wav file
    # python dunakeszi_segment_extractor_fixed.py --segments new_ground_truth/ground_truth_segments.json  --wav-dir wavs/  --array BK-6-E  --clip-position 0.5  --output-dir extracted_P/