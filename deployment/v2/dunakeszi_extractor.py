# -*- coding: utf-8 -*-
"""
dunakeszi_extractor.py
──────────────────────
Extraction pipeline for the Dunakeszi multi-drone field recording dataset.

What this does
──────────────
1. Parses the maneuver schedule (ground truth) from the protocol document —
   encoded directly from the Jegyzőkönyv (2025-10-20).
2. Reads large PolyWav files in streaming chunks — never loads the full file.
3. For each maneuver segment:
     • Extracts the relevant time window (with configurable padding)
     • Selects the correct channel subset for the BK-6-E array (ch 9–14,
       0-indexed 8–13) which maps to the 6-element 3D Brüel East array —
       the primary 3-channel array for the existing model.
     • Down-samples from 192 kHz → 22 050 Hz (cfg.SR)
     • Writes a 3-channel WAV (mic subset E, H, B) → clean_drone_segments/
     • Writes a companion _meta.json with full ground-truth labels
4. Outputs a dataset_manifest.json that lists every segment and its labels.
5. Copies non-drone windows (external noise, between maneuvers) →
   background_pool/ for use as non-drone training data.

Output layout (drop-in compatible with the existing pipeline)
──────────────────────────────────────────────────────────────
<out_root>/
  clean_drone_segments/          ← labelled drone clips (3-ch WAV + meta JSON)
  background_pool/               ← non-drone background clips
  dataset_manifest.json          ← full index of all extracted segments
  extraction_report.txt          ← human-readable summary

Ground-truth labels per segment
────────────────────────────────
Each _meta.json contains:
  clip/
    maneuver_id        : int   — row number from the protocol table
    flight_phase       : str   — "hover"|"transit_linear"|"transit_diagonal"|
                                  "orbit"|"formation_v"|"long_range"
    n_drones           : int   — number of active drones (1,2,3,4,5)
    altitude_m         : float — nominal altitude from protocol
    radius_m           : float|null — orbit radius if circular
    speed_mps          : float|null — nominal speed
    maneuver_type      : str   — coarse label for maneuver classification
    start_coord        : [x,y,z]|null
    end_coord          : [x,y,z]|null
    duration_s         : float
  signal_metrics/      — computed from extracted audio
  array/
    name               : "BK-6-E"
    channel_names      : ["E-E","E-H","E-B"]
    scorpio_channels   : [9,10,11]
    geometry           : "gp2"    ← 2.5 m equilateral triangle
  detection/
    detected_start_s, detected_end_s
  labels/              — all four model task labels in one place

Usage
─────
    python dunakeszi_extractor.py \\
        --wav_dir  wavs/ \\
        --out_dir  wavs_result/

    # Dry run first — see the plan without writing anything
    python dunakeszi_extractor.py --wav_dir wavs/ --out_dir wavs_result/ --dry_run

    # Full extraction
    python dunakeszi_extractor.py --wav_dir wavs/ --out_dir wavs_result/ 

    # Or programmatic:
    from dunakeszi_extractor import DunakesziExtractor
    ext = DunakesziExtractor(wav_dir="...", out_dir="...")
    ext.run()

    # After extraction, plug into the pipeline:
    from dunakeszi_extractor import import_dunakeszi_dataset
    cfg = import_dunakeszi_dataset("/path/to/output/")

Notes on the recording
───────────────────────
• Recording start: ~13:37 local (SMPTE ToD 24h 30fps ND)
• MEMS mics started at 14:06 — BK arrays recorded the full session
• PolyWav is split into 4 GB chunks (sample-point aligned, continuity guaranteed)
• Channel numbering (1-based Scorpio) → 0-based numpy index:
    ch1=Mix_L (skip), ch2=Mix_R (skip)
    ch3=W-E  ch4=W-H  ch5=W-B  ch6=W-J  ch7=W-F  ch8=W-L
    ch9=E-E  ch10=E-H ch11=E-B ch12=E-J ch13=E-F ch14=E-L
• For 3-mic TDOA matching the GP2 geometry: channels 9,10,11 (E-E,E-H,E-B)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import soundfile as sf
    _SF_OK = True
except ImportError:
    _SF_OK = False

try:
    import librosa
    _LIB_OK = True
except ImportError:
    _LIB_OK = False

try:
    import scipy.signal
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Timing reference
# ══════════════════════════════════════════════════════════════════════════════

# The PolyWav SMPTE ToD starts at 00:00:00 (midnight).
# Measurement started at ~13:37 local time.
# Set this to the exact SMPTE second of the first maneuver trigger once known.
RECORDING_START_SMPTE_S: float = 13 * 3600 + 37 * 60   # 13:37:00 = 48 620 s
_APPROX_TIMING = True   # Set False once confirmed with exact SMPTE frame


def _wall_to_s(hhmm: str, offset_s: float = 0.0) -> float:
    """Convert 'HH:MM' wall-clock string to seconds from RECORDING_START."""
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    return (h * 3600 + m * 60 + offset_s) - RECORDING_START_SMPTE_S


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth maneuver catalogue
# Transcribed from Jegyzőkönyv 2025-10-20, pages 3–4
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Maneuver:
    id:            int
    n_drones:      int
    description:   str
    maneuver_type: str    # coarse: hover|transit|circle|diagonal|formation|long_range
    flight_phase:  str    # fine:   hover|transit_linear|transit_diagonal|orbit|formation_v|long_range
    altitude_m:    float
    speed_mps:     Optional[float]
    radius_m:      Optional[float]
    start_coord:   Optional[List[float]]   # [x, y, z] metres in measurement frame
    end_coord:     Optional[List[float]]
    duration_s:    float
    show_file:     Optional[str]
    wall_clock:    Optional[str]           # approx "HH:MM"
    notes:         str = ""

    def to_dict(self) -> dict:
        return asdict(self)


MANEUVERS: List[Maneuver] = [
    # ── Altitude hover series ─────────────────────────────────────────────────
    Maneuver(1, 1, "Hover 10 m, 1 min",  "hover","hover", 10, 0.0, None, [0,0,10],  [0,0,10],  60,  "show_2","13:53"),
    Maneuver(2, 1, "Hover 20 m, 1 min",  "hover","hover", 20, 0.0, None, [0,0,20],  [0,0,20],  60,  "show_2","13:54"),
    Maneuver(3, 1, "Hover 30 m, 1 min",  "hover","hover", 30, 0.0, None, [0,0,30],  [0,0,30],  60,  "show_2","13:55"),
    Maneuver(4, 1, "Hover 40 m, 1 min",  "hover","hover", 40, 0.0, None, [0,0,40],  [0,0,40],  60,  "show_2","13:56", notes="60 dB gain from 13:44"),
    # ── Yaw rotations ─────────────────────────────────────────────────────────
    Maneuver(5, 1, "CCW yaw 20 m, 1 min","hover","hover", 20, 0.0, None, [0,0,20],  [0,0,20],  60,  "show_2","13:57", notes="CCW; not always executed"),
    Maneuver(6, 1, "CW yaw 20 m, 1 min", "hover","hover", 20, 0.0, None, [0,0,20],  [0,0,20],  60,  "show_2","13:58", notes="CW; not always executed"),
    # ── Linear transits ───────────────────────────────────────────────────────
    Maneuver(7, 1, "Linear transit 20 m, 4 m/s",  "transit","transit_linear", 20, 4.0, None, [-60,0,20],[60,0,20], 30, "show_2","13:59"),
    Maneuver(8, 1, "Linear transit 20 m, 8 m/s",  "transit","transit_linear", 20, 8.0, None, [-60,0,20],[60,0,20], 15, "show_2","14:00"),
    # ── Diagonal transits — 20 m ──────────────────────────────────────────────
    Maneuver(10,1, "Diagonal 20 m, 4 m/s",  "diagonal","transit_diagonal",20,4.0,None,[-60,-60,20],[60,60,20],66, "show_2","14:01"),
    Maneuver(11,1, "Diagonal 20 m, 8 m/s",  "diagonal","transit_diagonal",20,8.0,None,[-60,-60,20],[60,60,20],33, "show_2","14:02"),
    # ── Diagonal transits — 60 m ──────────────────────────────────────────────
    Maneuver(13,1, "Diagonal 60 m, 4 m/s",  "diagonal","transit_diagonal",60,4.0,None,[-60,-60,60],[60,60,60],66, "show_5","14:08", notes="MEMS starts 14:06"),
    Maneuver(14,1, "Diagonal 60 m, 8 m/s",  "diagonal","transit_diagonal",60,8.0,None,[-60,-60,60],[60,60,60],33, "show_5","14:09"),
    # ── Diagonal transits — 120 m ─────────────────────────────────────────────
    Maneuver(16,1, "Diagonal 120 m, 4 m/s", "diagonal","transit_diagonal",120,4.0,None,[-60,-60,120],[60,60,120],66,"show_7","14:24"),
    Maneuver(17,1, "Diagonal 120 m, 8 m/s", "diagonal","transit_diagonal",120,8.0,None,[-60,-60,120],[60,60,120],33,"show_7","14:25"),
    # ── Single-drone circles ──────────────────────────────────────────────────
    Maneuver(18,1, "Circle r=5 m, 20 m, 4 m/s",   "circle","orbit",20,4.0,  5.0,[-2.5,0,20],None, 16, "show_8","14:41"),
    Maneuver(19,1, "Circle r=30 m, 20 m, 4 m/s",  "circle","orbit",20,4.0, 30.0,[-15,0,20], None, 94, "show_8","14:43"),
    Maneuver(20,1, "Circle r=60 m, 20 m, 4 m/s",  "circle","orbit",20,4.0, 60.0,[-30,0,20], None,188, "show_8","14:45"),
    Maneuver(21,1, "Figure-8 at 20 m, 4 m/s",     "circle","orbit",20,4.0, None, None,       None, 63, "show_8","14:49"),
    # ── 3D cube diagonals ─────────────────────────────────────────────────────
    Maneuver(22,1, "Cube diag A, 4 m/s","diagonal","transit_diagonal",60,4.0,None,[-60,-60,0],  [60,60,120], 69,"show_9","15:02"),
    Maneuver(23,1, "Cube diag B, 4 m/s","diagonal","transit_diagonal",60,4.0,None,[-60, 60,0],  [60,-60,120],69,"show_9","15:03"),
    Maneuver(24,1, "Cube diag C, 4 m/s","diagonal","transit_diagonal",60,4.0,None,[ 60,-60,0],  [-60,60,120],69,"show_9","15:04"),
    Maneuver(25,1, "Cube diag D, 4 m/s","diagonal","transit_diagonal",60,4.0,None,[ 60, 60,0],  [-60,-60,120],69,"show_9","15:05"),
    # ── 2-drone orbits ────────────────────────────────────────────────────────
    Maneuver(27,2,"2× circle r=5 m",  "circle","orbit",20,4.0, 5.0, [-2.5,0,20],None,  8,"show_11","15:13"),
    Maneuver(28,2,"2× circle r=30 m", "circle","orbit",20,4.0,30.0, [-2.5,0,20],None, 47,"show_11","15:14"),
    Maneuver(29,2,"2× circle r=60 m", "circle","orbit",20,4.0,60.0, [-2.5,0,20],None,188,"show_11","15:15"),
    # ── 3-drone orbits ────────────────────────────────────────────────────────
    Maneuver(30,3,"3× circle r=10 m", "circle","orbit",20,4.0,10.0,[-2.5,0,20],None,  8,"show_12","15:27",notes="5 m too tight → 10 m"),
    Maneuver(31,3,"3× circle r=30 m", "circle","orbit",20,4.0,30.0,[-2.5,0,20],None, 47,"show_12","15:28"),
    Maneuver(32,3,"3× circle r=60 m", "circle","orbit",20,4.0,60.0,[-2.5,0,20],None,188,"show_12","15:29"),
    # ── 4-drone orbits ────────────────────────────────────────────────────────
    Maneuver(33,4,"4× circle r=10 m", "circle","orbit",20,4.0,10.0,[-2.5,0,20],None,  8,"show_13","15:42"),
    Maneuver(34,4,"4× circle r=30 m", "circle","orbit",20,4.0,30.0,[-2.5,0,20],None, 47,"show_13","15:43"),
    Maneuver(35,4,"4× circle r=60 m", "circle","orbit",20,4.0,60.0,[-2.5,0,20],None,188,"show_13","15:44"),
    # ── 5-drone orbits ────────────────────────────────────────────────────────
    Maneuver(36,5,"5× circle r=10 m", "circle","orbit",20,4.0,10.0,[-2.5,0,20],None,  8,"show_14","15:57"),
    Maneuver(37,5,"5× circle r=30 m", "circle","orbit",20,4.0,30.0,[-2.5,0,20],None, 47,"show_14","15:58"),
    Maneuver(38,5,"5× circle r=60 m", "circle","orbit",20,4.0,60.0,[-2.5,0,20],None,188,"show_14","15:59"),
    # ── V-formation transits ──────────────────────────────────────────────────
    Maneuver(39,5,"V-formation 4 m/s","formation","formation_v",20,4.0,None,[-60,0,20],[60,0,20],60,"show_15","16:25",notes="Shape not always maintained"),
    Maneuver(40,5,"V-formation 8 m/s","formation","formation_v",20,8.0,None,[-60,0,20],[60,0,20],30,"show_15","16:26"),
    # ── Long-range detection ──────────────────────────────────────────────────
    Maneuver(41,1,"Max-range approach 300 m","long_range","long_range",20,8.0,None,
             [-300*0.707,-300*0.707,20],[300*0.707,300*0.707,20],188,None,"16:27",
             notes="Audible ~200 m on BK-6-E; NW→SE pass"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Channel mapping
# ══════════════════════════════════════════════════════════════════════════════

CHANNEL_MAP = {
    # East BK-6-E (0-based numpy indices)
    "E-E": 8,  "E-H": 9,  "E-B": 10, "E-J": 11, "E-F": 12, "E-L": 13,
    # West BK-6-W
    "W-E": 2,  "W-H": 3,  "W-B": 4,  "W-J": 5,  "W-F": 6,  "W-L": 7,
}

# 3-mic subset for TDOA — matches GP2 equilateral triangle geometry in config
ARRAY_3CH = {
    "BK-6-E": ["E-E", "E-H", "E-B"],   # East array (primary)
    "BK-6-W": ["W-E", "W-H", "W-B"],   # West array (cross-validation)
}

NATIVE_SR      = 192_000
TARGET_SR      = 22_050
TOTAL_CHANNELS = 14


# ══════════════════════════════════════════════════════════════════════════════
# Signal metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signal_metrics(channels: List[np.ndarray], sr: int) -> Dict[str, Any]:
    y = channels[0].astype(np.float64)
    rms      = float(np.sqrt(np.mean(y ** 2)))
    rms_dbfs = float(20 * np.log10(rms + 1e-10))

    dom_freq = None
    snr_db   = None

    if _LIB_OK:
        try:
            S     = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            mask  = (freqs >= 50) & (freqs <= 700)
            dom_freq = round(float(freqs[mask][int(np.argmax(S.mean(axis=1)[mask]))]), 1)
        except Exception:
            pass

    if _SCIPY_OK and dom_freq:
        try:
            nyq = sr / 2.0
            bw  = 20.0
            lo  = max(dom_freq - bw, 1.0) / nyq
            hi  = min(dom_freq + bw, nyq - 1.0) / nyq
            sos_in  = scipy.signal.butter(4, [lo, hi], btype="band",     output="sos")
            sos_out = scipy.signal.butter(4, [lo, hi], btype="bandstop", output="sos")
            p_in  = float(np.mean(scipy.signal.sosfilt(sos_in,  y) ** 2)) + 1e-12
            p_out = float(np.mean(scipy.signal.sosfilt(sos_out, y) ** 2)) + 1e-12
            snr_db = round(float(10 * np.log10(p_in / p_out)), 2)
        except Exception:
            pass

    return {
        "rms_dbfs":         round(rms_dbfs, 2),
        "snr_db":           snr_db,
        "dominant_freq_hz": dom_freq,
        "n_samples":        len(y),
        "sample_rate":      sr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PolyWav streaming reader — never loads the full file into memory
# ══════════════════════════════════════════════════════════════════════════════

class PolyWavReader:
    """
    Streams specific channels from one or more 4 GB PolyWav chunks.
    Chunks must be in chronological order and sample-point aligned.
    """

    def __init__(
        self,
        wav_paths:  List[Path],
        ch_indices: List[int],
        native_sr:  int = NATIVE_SR,
        target_sr:  int = TARGET_SR,
    ):
        if not _SF_OK:
            raise RuntimeError("soundfile required: pip install soundfile")

        self.wav_paths  = sorted(wav_paths)
        self.ch_indices = ch_indices
        self.native_sr  = native_sr
        self.target_sr  = target_sr
        self._do_resample = (native_sr != target_sr)

        # Build cumulative frame index per chunk
        self._chunk_frames: List[int] = []
        self._cum_frames:   List[int] = []
        total = 0
        for p in self.wav_paths:
            info = sf.info(str(p))
            if info.samplerate != native_sr:
                raise ValueError(
                    f"Chunk {p.name}: expected {native_sr} Hz, got {info.samplerate} Hz"
                )
            self._chunk_frames.append(info.frames)
            total += info.frames
            self._cum_frames.append(total)

        self.total_frames    = total
        self.total_duration_s = total / native_sr

    def read_window(
        self,
        start_s: float,
        end_s:   float,
        pad_s:   float = 0.0,
    ) -> List[np.ndarray]:
        """
        Return one float32 array per requested channel for [start_s, end_s].
        pad_s extends the window symmetrically (clamped to recording bounds).
        """
        t0 = max(start_s - pad_s, 0.0)
        t1 = min(end_s   + pad_s, self.total_duration_s)

        f0 = int(t0 * self.native_sr)
        f1 = int(t1 * self.native_sr)
        n  = f1 - f0

        if n <= 0:
            return [np.zeros(1, dtype=np.float32) for _ in self.ch_indices]

        # Allocate buffer only for requested channels + position
        raw = np.zeros((n, TOTAL_CHANNELS), dtype=np.float32)

        prev_cum = 0
        for path, chunk_n, cum in zip(self.wav_paths, self._chunk_frames, self._cum_frames):
            chunk_start = prev_cum
            chunk_end   = cum

            win_s = max(f0, chunk_start)
            win_e = min(f1, chunk_end)
            if win_s >= win_e:
                prev_cum = cum
                continue

            read_start = win_s - chunk_start
            read_n     = win_e - win_s
            buf_offset = win_s - f0

            with sf.SoundFile(str(path)) as sff:
                sff.seek(read_start)
                block = sff.read(read_n, dtype="float32", always_2d=True)
                actual = block.shape[0]
                raw[buf_offset : buf_offset + actual, :] = block[:actual, :]

            prev_cum = cum

        channels = [raw[:, ci].copy() for ci in self.ch_indices]

        if self._do_resample:
            if _LIB_OK:
                channels = [
                    librosa.resample(ch, orig_sr=self.native_sr, target_sr=self.target_sr)
                    for ch in channels
                ]
            elif _SCIPY_OK:
                n_out = int(round(len(channels[0]) * self.target_sr / self.native_sr))
                channels = [
                    scipy.signal.resample(ch, n_out).astype(np.float32)
                    for ch in channels
                ]
            else:
                warnings.warn("No resampling library found — returning at native SR")

        return channels


# ══════════════════════════════════════════════════════════════════════════════
# Main extractor
# ══════════════════════════════════════════════════════════════════════════════

class DunakesziExtractor:
    """
    Full extraction pipeline for the Dunakeszi dataset.

    Parameters
    ──────────
    wav_dir      : directory containing Scorpio PolyWav .wav chunks
    out_dir      : output root
    pad_s        : context padding around each maneuver (seconds)
    sr           : target sample rate (must match cfg.SR)
    dry_run      : print plan without writing files
    arrays       : "east" | "west" | "both"
    bg_window_s  : length of background (non-drone) clips in seconds
    """

    def __init__(
        self,
        wav_dir:     str,
        out_dir:     str,
        pad_s:       float = 2.0,
        sr:          int   = TARGET_SR,
        dry_run:     bool  = False,
        arrays:      str   = "east",
        bg_window_s: float = 3.0,
    ):
        self.wav_dir     = Path(wav_dir)
        self.out_dir     = Path(out_dir)
        self.pad_s       = pad_s
        self.sr          = sr
        self.dry_run     = dry_run
        self.bg_window_s = bg_window_s

        self.wav_chunks = sorted(self.wav_dir.glob("*.wav"))
        if not self.wav_chunks:
            raise FileNotFoundError(f"No .wav files found in {self.wav_dir}")

        # Build readers for each array
        active_arrays = list(ARRAY_3CH.keys()) if arrays == "both" else (
            ["BK-6-E"] if arrays == "east" else ["BK-6-W"]
        )
        self.readers: Dict[str, PolyWavReader] = {
            name: PolyWavReader(
                self.wav_chunks,
                ch_indices=[CHANNEL_MAP[c] for c in ARRAY_3CH[name]],
                native_sr=NATIVE_SR, target_sr=sr,
            )
            for name in active_arrays
        }

        self.total_duration_s = next(iter(self.readers.values())).total_duration_s

        print(
            f"📂 PolyWav: {len(self.wav_chunks)} chunk(s), "
            f"{self.total_duration_s/3600:.2f} h\n"
            f"   Target SR: {sr} Hz  |  Pad: {pad_s}s  |  Arrays: {', '.join(active_arrays)}"
        )
        if _APPROX_TIMING:
            print(
                "   ⚠️  TIMING IS APPROXIMATE — refine RECORDING_START_SMPTE_S "
                "with the exact SMPTE frame of the first trigger."
            )

    # ── Onset calculation ─────────────────────────────────────────────────────

    def _onset(self, m: Maneuver) -> float:
        if m.wall_clock:
            return _wall_to_s(m.wall_clock)
        # Fallback: cumulative durations with ~5 s inter-maneuver gaps
        total = 0.0
        for prior in MANEUVERS:
            if prior.id >= m.id:
                break
            total += prior.duration_s + 5.0
        return total

    # ── Output path helpers ───────────────────────────────────────────────────

    def _stem(self, m: Maneuver, array_name: str) -> str:
        return (
            f"dunakeszi_m{m.id:03d}_"
            f"n{m.n_drones}_"
            f"{m.maneuver_type}_"
            f"alt{int(m.altitude_m)}m_"
            f"{array_name.replace('-','')}"
        )

    # ── Split assignment ──────────────────────────────────────────────────────

    @staticmethod
    def _split(m: Maneuver) -> str:
        """
        Deterministic train/val/test assignment.
          test  — long-range + all 5-drone maneuvers
          val   — 4-drone + single-drone hover (altitude series)
          train — everything else
        """
        if m.flight_phase == "long_range":              return "test"
        if m.n_drones == 5:                             return "test"
        if m.n_drones == 4:                             return "val"
        if m.maneuver_type == "hover" and m.n_drones == 1: return "val"
        return "train"

    # ── Single maneuver extraction ────────────────────────────────────────────

    def _extract_one(
        self,
        m:          Maneuver,
        reader:     PolyWavReader,
        array_name: str,
        ch_names:   List[str],
    ) -> Optional[dict]:

        onset   = self._onset(m)
        start_s = max(onset, 0.0)
        end_s   = onset + m.duration_s

        if end_s > self.total_duration_s:
            print(f"  ⚠️  m{m.id:03d} extends past recording end ({end_s:.0f}s) — skipping")
            return None

        stem     = self._stem(m, array_name)
        seg_dir  = self.out_dir / "clean_drone_segments"
        wav_out  = seg_dir / f"{stem}.wav"
        meta_out = seg_dir / f"{stem}_meta.json"

        if not self.dry_run:
            seg_dir.mkdir(parents=True, exist_ok=True)

            channels = reader.read_window(start_s, end_s, pad_s=self.pad_s)

            # Write 3-channel interleaved WAV
            audio_np = np.stack(channels, axis=1)
            sf.write(str(wav_out), audio_np, self.sr, subtype="PCM_16")

            # Compute signal metrics
            sig = _compute_signal_metrics(channels, self.sr)

            # Azimuth and distance at onset (if coordinates available)
            az_onset   = None
            dist_onset = None
            if m.start_coord:
                x, y = m.start_coord[0], m.start_coord[1]
                az_onset   = round(math.degrees(math.atan2(y, x)), 1)
                dist_onset = round(math.sqrt(x**2 + y**2), 1)

            meta = {
                "clip": {
                    "maneuver_id":   m.id,
                    "flight_phase":  m.flight_phase,
                    "maneuver_type": m.maneuver_type,
                    "n_drones":      m.n_drones,
                    "altitude_m":    m.altitude_m,
                    "speed_mps":     m.speed_mps,
                    "radius_m":      m.radius_m,
                    "start_coord":   m.start_coord,
                    "end_coord":     m.end_coord,
                    "duration_s":    m.duration_s,
                    "pad_s":         self.pad_s,
                    "total_clip_s":  round(m.duration_s + 2 * self.pad_s, 2),
                    "description":   m.description,
                    "notes":         m.notes,
                    "show_file":     m.show_file,
                    "onset_s_from_recording_start": round(onset, 2),
                    "timing_approx": _APPROX_TIMING,
                },
                "signal_metrics": sig,
                "array": {
                    "name":             array_name,
                    "channel_names":    ch_names,
                    "scorpio_channels": [CHANNEL_MAP[c] + 1 for c in ch_names],
                    "geometry":         "gp2",
                    "mic_spacing_m":    2.5,
                    "native_sr":        NATIVE_SR,
                    "target_sr":        self.sr,
                    "n_channels_extracted": len(ch_names),
                },
                "detection": {
                    "detected_start_s": round(self.pad_s, 2),
                    "detected_end_s":   round(self.pad_s + m.duration_s, 2),
                },
                # ── All four model task labels ────────────────────────────────
                "labels": {
                    # Task 1 — binary detection
                    "drone_present":    True,
                    # Task 2 — count estimation
                    "n_drones":         m.n_drones,
                    # Task 3 — maneuver / movement classification
                    "maneuver_type":    m.maneuver_type,
                    "flight_phase":     m.flight_phase,
                    # Task 4 — spatial localisation
                    "altitude_m":       m.altitude_m,
                    "start_coord":      m.start_coord,
                    "end_coord":        m.end_coord,
                    "azimuth_deg_onset":    az_onset,
                    "distance_m_onset":     dist_onset,
                    "radius_m":         m.radius_m,
                    "speed_mps":        m.speed_mps,
                },
            }

            with open(str(meta_out), "w") as f:
                json.dump(meta, f, indent=2)

        return {
            "stem":          stem,
            "wav":           str(wav_out.relative_to(self.out_dir)),
            "meta":          str(meta_out.relative_to(self.out_dir)),
            "maneuver_id":   m.id,
            "n_drones":      m.n_drones,
            "maneuver_type": m.maneuver_type,
            "flight_phase":  m.flight_phase,
            "altitude_m":    m.altitude_m,
            "duration_s":    m.duration_s,
            "array":         array_name,
            "split":         self._split(m),
            "drone_present": True,
        }

    # ── Background clip extraction ────────────────────────────────────────────

    def _extract_backgrounds(
        self,
        reader:     PolyWavReader,
        array_name: str,
    ) -> List[dict]:
        """Extract non-drone windows from gaps between maneuvers."""
        bg_dir = self.out_dir / "background_pool"
        if not self.dry_run:
            bg_dir.mkdir(parents=True, exist_ok=True)

        onsets = sorted(
            [(self._onset(m), m.duration_s) for m in MANEUVERS],
            key=lambda x: x[0],
        )

        entries = []
        bg_idx  = 0

        for i in range(len(onsets) - 1):
            gap_start = onsets[i][0]  + onsets[i][1]   + 1.5
            gap_end   = onsets[i+1][0] - 1.5
            gap_dur   = gap_end - gap_start

            if gap_dur < self.bg_window_s + 2.0:
                continue

            # Extract up to 3 clips per gap
            n_clips = min(3, int(gap_dur / self.bg_window_s))
            for ci in range(n_clips):
                t0 = gap_start + ci * self.bg_window_s
                t1 = t0 + self.bg_window_s
                if t1 > self.total_duration_s:
                    break

                stem    = f"dunakeszi_bg_{bg_idx:04d}_{array_name.replace('-','')}"
                wav_out = bg_dir / f"{stem}.wav"
                bg_idx += 1

                if not self.dry_run:
                    channels = reader.read_window(t0, t1, pad_s=0.0)
                    audio_np = np.stack(channels, axis=1)
                    sf.write(str(wav_out), audio_np, self.sr, subtype="PCM_16")

                entries.append({
                    "stem":          stem,
                    "wav":           str(wav_out.relative_to(self.out_dir)),
                    "label":         "non_drone",
                    "drone_present": False,
                    "n_drones":      0,
                    "array":         array_name,
                    "split":         "train",
                })

        return entries

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run full extraction. Returns manifest dict."""
        mode = "DRY RUN" if self.dry_run else "WRITING FILES"
        print(f"\n{'='*60}\n  Dunakeszi Extractor — {mode}\n{'='*60}\n")

        manifest = {"drone_segments": [], "background_segments": []}

        for array_name, reader in self.readers.items():
            ch_names = ARRAY_3CH[array_name]
            print(f"── Array: {array_name}  channels: {', '.join(ch_names)}\n")

            for m in MANEUVERS:
                split = self._split(m)
                print(f"  m{m.id:03d}  {m.n_drones}×  {m.description:<48s} "
                      f"{m.duration_s:>5.0f}s  [{split}]")
                if not self.dry_run:
                    entry = self._extract_one(m, reader, array_name, ch_names)
                    if entry:
                        manifest["drone_segments"].append(entry)
                else:
                    manifest["drone_segments"].append({
                        "maneuver_id": m.id, "n_drones": m.n_drones,
                        "maneuver_type": m.maneuver_type, "split": split,
                    })

            print(f"\n  Extracting background clips …")
            bg = self._extract_backgrounds(reader, array_name)
            manifest["background_segments"].extend(bg)
            print(f"  → {len(bg)} background clips")

        if not self.dry_run:
            mpath = self.out_dir / "dataset_manifest.json"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(str(mpath), "w") as f:
                json.dump(manifest, f, indent=2)
            self._write_report(manifest)

        self._print_summary(manifest)
        return manifest

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_summary(self, manifest: dict):
        drone = manifest["drone_segments"]
        bg    = manifest["background_segments"]
        print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
        print(f"  Drone segments   : {len(drone)}")
        print(f"  Background clips : {len(bg)}\n")
        for split in ("train", "val", "test"):
            nd = sum(1 for x in drone if x.get("split") == split)
            nb = sum(1 for x in bg    if x.get("split") == split)
            print(f"    {split:5s}: {nd:3d} drone  +  {nb:3d} bg")
        counts = {}
        for x in drone:
            counts[x.get("n_drones", "?")] = counts.get(x.get("n_drones","?"), 0) + 1
        print(f"\n  Drone count distribution:")
        for k, v in sorted(counts.items()):
            print(f"    {k} drone(s): {v} clips")
        print(f"{'='*60}\n")

    def _write_report(self, manifest: dict):
        drone = manifest["drone_segments"]
        bg    = manifest["background_segments"]
        lines = [
            "Dunakeszi Extraction Report",
            "=" * 50,
            f"Drone segments   : {len(drone)}",
            f"Background clips : {len(bg)}",
            "",
            "Split breakdown:",
        ]
        for split in ("train", "val", "test"):
            nd = sum(1 for x in drone if x.get("split") == split)
            lines.append(f"  {split}: {nd} drone")
        lines += [
            "",
            "Timing: " + (
                "APPROXIMATE — set RECORDING_START_SMPTE_S from exact SMPTE frame"
                if _APPROX_TIMING else "Confirmed"
            ),
            "",
            "Channel map (0-based index):",
        ]
        for name, idx in sorted(CHANNEL_MAP.items()):
            lines.append(f"  {name}: index {idx} (Scorpio ch{idx+1})")
        (self.out_dir / "extraction_report.txt").write_text("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_config_for_dunakeszi(base_cfg=None):
    """
    Return a Config tuned for the Dunakeszi dataset:
      - Array geometry GP2 (2.5 m equilateral triangle, East BK-6-E)
      - MAX_LOCALIZATION_DIST = 300 m (long-range test)
    """
    try:
        from drone_detection.config import Config
    except ImportError:
        raise ImportError("drone_detection package not on PYTHONPATH")

    cfg = base_cfg or Config()
    cfg.set_array_geometry("gp2")
    cfg.MAX_LOCALIZATION_DIST = 300.0
    cfg.CUSTOM_DATASET_ENABLED = True
    print("✅ Config: array=gp2, MAX_LOCALIZATION_DIST=300 m")
    return cfg


def import_dunakeszi_dataset(out_dir: str, cfg=None):
    """
    Import an already-extracted Dunakeszi dataset into the pipeline.
    Sets cfg.CUSTOM_DATASET_ROOT and calls import_custom_builder_dataset().
    """
    cfg = build_config_for_dunakeszi(cfg)
    cfg.CUSTOM_DATASET_ROOT = str(out_dir)
    try:
        from drone_detection import import_custom_builder_dataset
        import_custom_builder_dataset(out_dir, cfg)
        print(f"✅ Dataset imported from {out_dir}")
    except ImportError:
        print("⚠️  drone_detection not importable — set cfg.CUSTOM_DATASET_ROOT "
              "and call import_custom_builder_dataset(out_dir, cfg) manually")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser(
        description="Extract Dunakeszi multi-drone dataset from PolyWav files"
    )
    p.add_argument("--wav_dir",     required=True, help="Scorpio PolyWav chunk directory")
    p.add_argument("--out_dir",     required=True, help="Output root directory")
    p.add_argument("--pad_s",       type=float, default=2.0,  help="Padding in seconds (default 2.0)")
    p.add_argument("--sr",          type=int,   default=TARGET_SR, help=f"Target SR (default {TARGET_SR})")
    p.add_argument("--arrays",      choices=["east","west","both"], default="east")
    p.add_argument("--bg_window_s", type=float, default=3.0,  help="Background clip length (default 3.0)")
    p.add_argument("--dry_run",     action="store_true")
    args = p.parse_args()

    DunakesziExtractor(
        wav_dir     = args.wav_dir,
        out_dir     = args.out_dir,
        pad_s       = args.pad_s,
        sr          = args.sr,
        dry_run     = args.dry_run,
        arrays      = args.arrays,
        bg_window_s = args.bg_window_s,
    ).run()


if __name__ == "__main__":
    _cli()