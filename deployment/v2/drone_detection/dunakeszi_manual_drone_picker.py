#!/usr/bin/env python3
# manual_drone_section_picker.py
"""
Dunakeszi enhanced manual drone section picker
====================================================
Auto-detects drone-active segments using two independent detectors:

  1. BPF detector  — tracks blade-pass fundamental energy (default ~82 Hz)
                     and its harmonics via short-time Welch windows.
  2. RMS detector  — flags frames whose broadband energy exceeds
                     (noise_floor + threshold_db).

Detection logic: --logic or  (default) → either detector fires → candidate
                 --logic and           → both must fire simultaneously

For each detected candidate the tool:
  • prints a summary (duration, SNR, BPF energy, GPS bbox)
  • offers playback of the whole candidate
  • lets you accept as-is (press Enter) OR enter trimmed A:B ranges
  • saves WAV + rich sidecar _meta.json per clip

Outputs
-------
  {output}/
    clean_drone_sections/
      MEMS/    {stem}_{start_ms}_{end_ms}.wav + _meta.json
      BRUEL/   …
    manifests/
      manual_clips_manifest.json
      manual_clips_manifest.csv
    detection/
      {stem}_detections.json   ← raw detector output per file

Usage
-----
  python dunakeszi_manual_drone_picker.py \\
    --source  Dunakeszi_Data \\
    --output  Dunakeszi_Data \\
    --gpx     Dunakeszi_Data/output/gpx_combined.csv \\
    --rpm     Dunakeszi_Data/output/rotor_rpm_ch0.csv \\
    --flight-takeoff 2025-10-20T12:57:21 \\
    --flight-landing 2025-10-20T13:04:09 \\
    --logic or \\
    --bpf-hz 82 \\
    --rms-threshold-db 12 \\
    --min-segment-sec 2.0 \\
    --merge-gap-sec 1.5 \\
    --detection-sr 4000

    python dunakeszi_manual_drone_picker.py --source  Dunakeszi_Data --output  Dunakeszi_Data --gpx Dunakeszi_Data/output/gpx_combined.csv --rpm Dunakeszi_Data/output/rotor_rpm_ch0.csv --flight-takeoff 2025-10-20T12:57:21 --flight-landing 2025-10-20T13:04:09 --logic or --bpf-hz 82 --rms-threshold-db 12 --min-segment-sec 2.0 --merge-gap-sec 1.5 --detection-sr 4000

Dependencies: numpy scipy soundfile pandas
Optional:     librosa (better resampling), sounddevice / ffplay (playback)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import json
import math
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats
from scipy.io import wavfile

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False


# =============================================================================
#  DEFAULTS
# =============================================================================

DEFAULT_TAKEOFF     = datetime(2025, 10, 20, 12, 57, 21, tzinfo=timezone.utc)
DEFAULT_LANDING     = datetime(2025, 10, 20, 13,  4,  9, tzinfo=timezone.utc)
DEFAULT_SR          = 22050     # final clip save SR
DETECTION_SR        = 4000      # SR used for detection (fast)
AUDIO_EXTS          = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff",
                       ".m4a", ".w64")
MEMS_TIMESTAMPS_FILE = "mems_file_starts.csv"

# Brüel file start times (absolute UTC of sample 0)
BRUEL_FILE_STARTS: Dict[str, datetime] = {
    "251020VITEMOROM1AT01I": datetime(2025, 10, 20, 12, 50, 34, tzinfo=timezone.utc),
    "251020VITEMOROM1AT01J": datetime(2025, 10, 20, 12, 59, 28, tzinfo=timezone.utc),
}

# Flight phase margins
TAKEOFF_RAMP_SEC = 30
LANDING_RAMP_SEC = 30


# =============================================================================
#  I/O UTILITIES
# =============================================================================

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def safe_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in name).strip("_") or "clip"


def classify_source(path: Path) -> str:
    u = path.stem.upper()
    for k in BRUEL_FILE_STARTS:
        if k.upper() in u:
            return "BRUEL"
    return "MEMS"


def file_start_utc(path: Path, mems_table: Optional[pd.DataFrame],
                   takeoff_utc: datetime) -> datetime:
    stem = path.stem
    for key, dt in BRUEL_FILE_STARTS.items():
        if key.upper() in stem.upper():
            return dt
    if mems_table is not None and not mems_table.empty:
        match = mems_table[
            mems_table["filename"].str.contains(stem, case=False, na=False)
        ]
        if not match.empty:
            try:
                return pd.to_datetime(
                    match.iloc[0]["start_utc"], utc=True
                ).to_pydatetime()
            except Exception:
                pass
    return takeoff_utc


# =============================================================================
#  AUDIO LOADING  (chunked for large files)
# =============================================================================

def _resample_scipy(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from math import gcd
    g = gcd(src_sr, dst_sr)
    return signal.resample_poly(y, dst_sr // g, src_sr // g).astype(np.float32)


def load_audio_full(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load entire file, mix to mono, resample to target_sr."""
    if LIBROSA_OK:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return y.astype(np.float32), target_sr

    # soundfile path
    info = sf.info(str(path))
    src_sr = info.samplerate
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if src_sr != target_sr:
        y = _resample_scipy(y, src_sr, target_sr)
    return y.astype(np.float32), target_sr


def load_audio_segment(path: Path, start_s: float, end_s: float,
                        target_sr: int) -> Tuple[np.ndarray, int]:
    """Load a specific time region, mono, resampled."""
    if LIBROSA_OK:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=target_sr, mono=True,
                                 offset=start_s, duration=end_s - start_s)
        return y.astype(np.float32), target_sr

    info = sf.info(str(path))
    src_sr = info.samplerate
    s0 = int(start_s * src_sr)
    s1 = int(end_s   * src_sr)
    data, _ = sf.read(str(path), start=s0, stop=s1,
                      dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if src_sr != target_sr:
        y = _resample_scipy(y, src_sr, target_sr)
    return y.astype(np.float32), target_sr


def get_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


# =============================================================================
#  DETECTION ENGINE
# =============================================================================

class DroneDetector:
    """
    Runs two independent detectors on a mono downsampled signal and returns
    a list of (start_s, end_s) candidate segments.

    BPF detector:
      Short-time Welch windows; sums spectral energy at expected BPF
      harmonics (within ±bpf_tol_hz). Fires when harmonic sum exceeds
      bpf_threshold × median.

    RMS detector:
      Per-frame broadband RMS in dB. Fires when frame exceeds
      (10th-percentile noise floor + rms_threshold_db).
    """

    def __init__(
        self,
        sr:               int   = DETECTION_SR,
        bpf_hz:           float = 82.0,
        n_harmonics:      int   = 8,
        bpf_tol_hz:       float = 4.0,
        bpf_threshold:    float = 3.0,   # ×median to flag a frame
        rms_threshold_db: float = 12.0,  # dB above noise floor
        frame_sec:        float = 0.5,
        min_segment_sec:  float = 2.0,
        merge_gap_sec:    float = 1.5,
        logic:            str   = "or",  # "or" | "and"
    ):
        self.sr               = sr
        self.bpf_hz           = bpf_hz
        self.n_harmonics      = n_harmonics
        self.bpf_tol_hz       = bpf_tol_hz
        self.bpf_threshold    = bpf_threshold
        self.rms_threshold_db = rms_threshold_db
        self.frame_sec        = frame_sec
        self.min_segment_sec  = min_segment_sec
        self.merge_gap_sec    = merge_gap_sec
        self.logic            = logic.lower().strip()

        self.frame_samples = max(int(frame_sec * sr), 64)
        # Welch window: 4× frame for better freq resolution
        self.nperseg = min(self.frame_samples * 4, sr)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bpf_energy_series(self, y: np.ndarray) -> np.ndarray:
        """Return per-frame BPF harmonic energy array."""
        n_fr = len(y) // self.frame_samples
        if n_fr == 0:
            return np.array([])
        energies = np.zeros(n_fr, dtype=np.float64)
        hop = self.frame_samples

        # harmonic bin mask (reusable)
        # compute once after first welch call
        f_ref = None
        mask  = None

        for i in range(n_fr):
            chunk = y[i * hop: (i + 1) * hop]
            # pad to nperseg if needed
            if len(chunk) < self.nperseg:
                chunk = np.pad(chunk, (0, self.nperseg - len(chunk)))
            f, psd = signal.welch(chunk, fs=self.sr, nperseg=self.nperseg)
            if f_ref is None:
                f_ref = f
                mask = np.zeros(len(f), dtype=bool)
                for h in range(1, self.n_harmonics + 1):
                    fh = self.bpf_hz * h
                    if fh > self.sr / 2:
                        break
                    mask |= (np.abs(f - fh) <= self.bpf_tol_hz)
            energies[i] = float(psd[mask].sum()) if mask.any() else 0.0

        return energies

    def _rms_series(self, y: np.ndarray) -> np.ndarray:
        """Return per-frame RMS in dBFS."""
        n_fr = len(y) // self.frame_samples
        if n_fr == 0:
            return np.array([])
        frames = y[:n_fr * self.frame_samples].reshape(n_fr, self.frame_samples)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        return 20 * np.log10(rms + 1e-12)

    @staticmethod
    def _frames_to_segments(mask: np.ndarray, frame_sec: float,
                             min_seg: float, merge_gap: float
                             ) -> List[Tuple[float, float]]:
        """Convert a boolean per-frame mask into merged (start_s, end_s) pairs."""
        segs = []
        in_seg = False
        s_start = 0
        for i, active in enumerate(mask):
            if active and not in_seg:
                s_start = i; in_seg = True
            elif not active and in_seg:
                segs.append((s_start * frame_sec, i * frame_sec))
                in_seg = False
        if in_seg:
            segs.append((s_start * frame_sec, len(mask) * frame_sec))

        # merge close gaps
        merged = []
        for seg in segs:
            if merged and seg[0] - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(list(seg))

        # filter short
        return [(s, e) for s, e in merged if e - s >= min_seg]

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, y: np.ndarray) -> Dict:
        """
        Run detection on a mono float32 array.
        Returns dict with:
          segments   : list of (start_s, end_s)
          bpf_energy : per-frame array
          rms_db     : per-frame array
          bpf_mask   : per-frame bool
          rms_mask   : per-frame bool
          combined   : per-frame bool (logic applied)
        """
        bpf_energy = self._bpf_energy_series(y)
        rms_db     = self._rms_series(y)

        n_fr = min(len(bpf_energy), len(rms_db))
        if n_fr == 0:
            return {"segments": [], "bpf_energy": bpf_energy,
                    "rms_db": rms_db, "bpf_mask": np.array([]),
                    "rms_mask": np.array([]), "combined": np.array([])}

        bpf_energy = bpf_energy[:n_fr]
        rms_db     = rms_db[:n_fr]

        # BPF mask: above threshold × median
        bpf_med = np.median(bpf_energy) + 1e-20
        bpf_mask = bpf_energy > (self.bpf_threshold * bpf_med)

        # RMS mask: above (noise_floor + offset)
        noise_floor = float(np.percentile(rms_db, 10))
        rms_mask = rms_db > (noise_floor + self.rms_threshold_db)

        if self.logic == "and":
            combined = bpf_mask & rms_mask
        else:
            combined = bpf_mask | rms_mask

        segs = self._frames_to_segments(
            combined, self.frame_sec, self.min_segment_sec, self.merge_gap_sec
        )

        return {
            "segments":   segs,
            "bpf_energy": bpf_energy,
            "rms_db":     rms_db,
            "bpf_mask":   bpf_mask,
            "rms_mask":   rms_mask,
            "combined":   combined,
            "noise_floor_db": round(noise_floor, 2),
        }


# =============================================================================
#  METADATA HELPERS  (same as v2)
# =============================================================================

def compute_signal_metrics(y: np.ndarray, sr: int) -> Dict:
    m: Dict = {}
    if y is None or len(y) == 0:
        return m
    rms  = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    m["rms"]           = round(rms, 6)
    m["peak"]          = round(peak, 6)
    m["rms_dbfs"]      = round(20 * math.log10(rms  + 1e-12), 2)
    m["peak_dbfs"]     = round(20 * math.log10(peak + 1e-12), 2)
    m["dc_offset"]     = round(float(np.mean(y)), 6)
    m["crest_factor"]  = round(peak / (rms + 1e-9), 3)
    m["kurtosis"]      = round(float(stats.kurtosis(y)), 3)
    frame = max(int(0.1 * sr), 64)
    n_fr  = len(y) // frame
    if n_fr >= 4:
        frames = y[:n_fr * frame].reshape(n_fr, frame)
        fr_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        fr_db  = 20 * np.log10(fr_rms + 1e-12)
        noise  = float(np.percentile(fr_rms, 10)) + 1e-12
        sig    = float(np.percentile(fr_rms, 90)) + 1e-12
        m["snr_db"]          = round(20 * math.log10(sig / noise), 2)
        m["noise_floor_dbfs"]= round(20 * math.log10(noise), 2)
        m["rms_db_mean"]     = round(float(fr_db.mean()), 2)
        m["rms_db_std"]      = round(float(fr_db.std()),  3)
        m["silence_pct"]     = round(float(np.mean(fr_db < -50) * 100), 2)
        m["clip_pct"]        = round(float(np.mean(np.abs(y) > 0.99) * 100), 3)
    nperseg = min(4096, max(256, len(y) // 8))
    try:
        f_psd, psd = signal.welch(y, fs=sr, nperseg=nperseg)
        denom = np.sum(psd) + 1e-10
        m["spectral_centroid_hz"] = round(float(np.sum(f_psd * psd) / denom), 1)
        peaks, props = signal.find_peaks(psd, height=psd.max() * 0.05, distance=3)
        if len(peaks):
            top = peaks[np.argmax(props["peak_heights"])]
            m["dominant_freq_hz"] = round(float(f_psd[top]), 2)
        idx = (f_psd > 50) & (f_psd < sr / 2 * 0.9)
        if idx.sum() > 5:
            slope, *_ = np.polyfit(
                np.log10(f_psd[idx] + 1e-6),
                np.log10(psd[idx]   + 1e-10), 1)
            m["noise_slope"]  = round(float(slope), 3)
            m["noise_colour"] = ("pink" if -1.5 < slope < -0.5 else
                                 "brownian" if slope < -1.5 else "white")
    except Exception:
        pass
    return m


def infer_flight_phase(clip_start: datetime, clip_end: datetime,
                        takeoff: datetime, landing: datetime) -> str:
    mid = clip_start + (clip_end - clip_start) / 2
    if mid < takeoff:                                        return "pre-flight"
    if mid > landing:                                        return "post-flight"
    if mid <= takeoff + timedelta(seconds=TAKEOFF_RAMP_SEC): return "takeoff"
    if mid >= landing - timedelta(seconds=LANDING_RAMP_SEC): return "landing"
    return "cruise"


def gpx_in_window(gpx_df: Optional[pd.DataFrame],
                   t0: datetime, t1: datetime) -> List[Dict]:
    if gpx_df is None or gpx_df.empty or "time" not in gpx_df.columns:
        return []
    ts0, ts1 = pd.Timestamp(t0, tz="UTC"), pd.Timestamp(t1, tz="UTC")
    sub = gpx_df[(gpx_df["time"] >= ts0) & (gpx_df["time"] <= ts1)].copy()
    if sub.empty:
        return []
    if "time" in sub.columns:
        sub["time"] = sub["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    keep = [c for c in ("time","lat","lon","elevation","speed","source")
            if c in sub.columns]
    return sub[keep].replace({float("nan"): None}).to_dict(orient="records")


def rpm_in_window(rpm_df: Optional[pd.DataFrame],
                   t0: datetime, t1: datetime, takeoff: datetime) -> Dict:
    if rpm_df is None or rpm_df.empty or "time_s" not in rpm_df.columns:
        return {}
    s0 = (t0 - takeoff).total_seconds()
    s1 = (t1 - takeoff).total_seconds()
    sub = rpm_df[(rpm_df["time_s"] >= s0) & (rpm_df["time_s"] <= s1)]
    if sub.empty:
        return {}
    out: Dict = {}
    for col in ("rpm", "rpm_smooth", "bpf_hz", "bpf_hz_smooth"):
        if col in sub.columns:
            vals = sub[col].dropna()
            if not vals.empty:
                out[f"{col}_mean"] = round(float(vals.mean()), 1)
                out[f"{col}_min"]  = round(float(vals.min()),  1)
                out[f"{col}_max"]  = round(float(vals.max()),  1)
    return out


# =============================================================================
#  PLAYBACK
# =============================================================================

def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(y)) + 1e-8)
    return np.clip(y * (peak / m), -1.0, 1.0).astype(np.float32)


def try_play(y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        print("  (nothing to play)"); return
    if SOUNDDEVICE_OK:
        try:
            sd.stop(); sd.play(y, sr, blocking=True); return
        except Exception as e:
            print(f"  sounddevice: {e}")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(str(tmp), y, sr)
        if sys.platform == "darwin":
            subprocess.run(["afplay", str(tmp)], check=False)
        elif sys.platform.startswith("win"):
            import os; os.startfile(str(tmp)); input("  Enter when done…")  # type: ignore
        else:
            for cmd in (["ffplay","-nodisp","-autoexit",str(tmp)],
                        ["mpv","--no-video",str(tmp)],
                        ["paplay",str(tmp)],
                        ["aplay", str(tmp)]):
                try:
                    subprocess.run(cmd, check=False,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue
    finally:
        tmp.unlink(missing_ok=True)


# =============================================================================
#  RANGE PARSING
# =============================================================================

def parse_ranges(text: str, total_sec: float) -> List[Tuple[float, float]]:
    out = []
    for p in [x.strip() for x in text.split(",") if x.strip()]:
        if ":" not in p:
            raise ValueError(f"Use start:end — got '{p}'")
        a, b = p.split(":", 1)
        s = max(0.0, float(a.strip()))
        e = min(total_sec, float(b.strip()))
        if e <= s:
            raise ValueError(f"end must be > start in '{p}'")
        out.append((s, e))
    return out


def print_help():
    print("""
  Commands during segment review:
    Enter / y    accept segment as-is
    A:B          trim to this range within segment
    A:B, C:D     save multiple sub-clips
    p            play segment
    p A B        play sub-region A..B seconds (relative to segment start)
    n / s        skip this segment
    q            quit and save manifest
    h            this help
""")


# =============================================================================
#  SAVE CLIP + SIDECAR
# =============================================================================

def save_clip_with_meta(
    path:         Path,
    clip_y:       np.ndarray,
    save_sr:      int,
    start_s:      float,
    end_s:        float,
    source_type:  str,
    clips_dir:    Path,
    f_start_utc:  datetime,
    takeoff_utc:  datetime,
    landing_utc:  datetime,
    gpx_df:       Optional[pd.DataFrame],
    rpm_df:       Optional[pd.DataFrame],
    det_summary:  Optional[Dict] = None,
) -> Dict:
    """Save .wav + _meta.json. Returns manifest row dict."""
    clip_start_utc = f_start_utc + timedelta(seconds=start_s)
    clip_end_utc   = f_start_utc + timedelta(seconds=end_s)

    sig_metrics = compute_signal_metrics(clip_y, save_sr)
    gpx_points  = gpx_in_window(gpx_df, clip_start_utc, clip_end_utc)
    rpm_stats   = rpm_in_window(rpm_df, clip_start_utc, clip_end_utc, takeoff_utc)
    phase       = infer_flight_phase(clip_start_utc, clip_end_utc,
                                      takeoff_utc, landing_utc)

    out_subdir = clips_dir / source_type
    ensure_dir(out_subdir)
    slug      = safe_slug(path.stem)
    fstem     = f"{slug}_{int(start_s*1000):07d}_{int(end_s*1000):07d}"
    wav_path  = out_subdir / f"{fstem}.wav"
    meta_path = out_subdir / f"{fstem}_meta.json"

    sf.write(str(wav_path), normalize_peak(clip_y), save_sr)

    meta = {
        "clip": {
            "source_file":  str(path),
            "source_type":  source_type,
            "output_wav":   str(wav_path),
            "start_s":      round(start_s, 4),
            "end_s":        round(end_s,   4),
            "duration_s":   round(end_s - start_s, 4),
            "sample_rate":  save_sr,
            "start_utc":    clip_start_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "end_utc":      clip_end_utc.strftime(  "%Y-%m-%dT%H:%M:%S.%fZ"),
            "flight_phase": phase,
        },
        "signal_metrics": sig_metrics,
        "rpm":  rpm_stats,
        "gpx":  {"n_points": len(gpx_points), "trackpoints": gpx_points},
        "detection": det_summary or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    row = {
        "source_file":      str(path),
        "source_type":      source_type,
        "output_wav":       str(wav_path),
        "output_meta_json": str(meta_path),
        "start_s":          round(start_s, 4),
        "end_s":            round(end_s,   4),
        "duration_s":       round(end_s - start_s, 4),
        "start_utc":        clip_start_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "end_utc":          clip_end_utc.strftime(  "%Y-%m-%dT%H:%M:%S.%fZ"),
        "flight_phase":     phase,
        "gpx_n_points":     len(gpx_points),
        **{f"sig_{k}": v for k, v in sig_metrics.items()},
        **{f"rpm_{k}": v for k, v in rpm_stats.items()},
    }
    return row


# =============================================================================
#  PER-FILE PROCESSING
# =============================================================================

def process_file(
    path:         Path,
    detector:     DroneDetector,
    clips_dir:    Path,
    det_dir:      Path,
    save_sr:      int,
    gpx_df:       Optional[pd.DataFrame],
    rpm_df:       Optional[pd.DataFrame],
    mems_table:   Optional[pd.DataFrame],
    takeoff_utc:  datetime,
    landing_utc:  datetime,
) -> Tuple[List[Dict], bool]:
    """
    Run detector, then interactive trim loop for each candidate.
    Returns (manifest_rows, quit_requested).
    """
    source_type = classify_source(path)
    f_start     = file_start_utc(path, mems_table, takeoff_utc)
    total_sec   = get_duration(path)

    print(f"\n{'═'*72}")
    print(f"  File   : {path.name}  [{source_type}]")
    print(f"  Length : {total_sec:.1f}s  ({total_sec/60:.1f} min)")
    print(f"  T₀ UTC : {f_start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  Detecting drone activity (sr={detector.sr} Hz, logic={detector.logic})…")

    # ── Step 1: load at detection SR ─────────────────────────────────────────
    try:
        y_det, _ = load_audio_full(path, detector.sr)
    except Exception as e:
        print(f"  ⚠  Load failed: {e}"); return [], False

    # ── Step 2: detect ────────────────────────────────────────────────────────
    det = detector.detect(y_det)
    segs = det["segments"]

    print(f"  ✔  {len(segs)} segment(s) detected  "
          f"(noise floor {det.get('noise_floor_db','?')} dBFS)")

    # save raw detection JSON
    det_json = det_dir / f"{safe_slug(path.stem)}_detections.json"
    det_json.write_text(json.dumps(
        {
            "file": str(path),
            "total_sec": total_sec,
            "logic": detector.logic,
            "noise_floor_db": det.get("noise_floor_db"),
            "segments": [{"start_s": s, "end_s": e, "duration_s": round(e-s,3)}
                         for s, e in segs],
        }, indent=2
    ))

    del y_det; gc.collect()

    if not segs:
        print("  — No drone activity detected, skipping file.")
        return [], False

    # ── Step 3: interactive review ────────────────────────────────────────────
    print_help()
    manifest_rows: List[Dict] = []

    for seg_idx, (seg_s, seg_e) in enumerate(segs, 1):
        seg_dur = seg_e - seg_s
        seg_start_utc = f_start + timedelta(seconds=seg_s)
        seg_end_utc   = f_start + timedelta(seconds=seg_e)
        phase = infer_flight_phase(seg_start_utc, seg_end_utc,
                                    takeoff_utc, landing_utc)
        gpx_pts = gpx_in_window(gpx_df, seg_start_utc, seg_end_utc)

        # quick RMS on detection-SR signal for display
        try:
            y_seg_det, _ = load_audio_segment(path, seg_s, seg_e, detector.sr)
            frame = max(int(0.1 * detector.sr), 64)
            nf    = len(y_seg_det) // frame
            if nf:
                fr   = y_seg_det[:nf*frame].reshape(nf, frame)
                frms = np.sqrt(np.mean(fr.astype(np.float64)**2, axis=1))
                noise_r = float(np.percentile(frms, 10)) + 1e-12
                sig_r   = float(np.percentile(frms, 90)) + 1e-12
                snr_preview = round(20*math.log10(sig_r/noise_r), 1)
            else:
                snr_preview = "?"
            del y_seg_det; gc.collect()
        except Exception:
            snr_preview = "?"

        print(f"\n  ┌─ Segment {seg_idx}/{len(segs)} ─────────────────────────────────")
        print(f"  │  Range  : {seg_s:.2f}s → {seg_e:.2f}s  ({seg_dur:.1f}s)")
        print(f"  │  UTC    : {seg_start_utc.strftime('%H:%M:%S')} → "
              f"{seg_end_utc.strftime('%H:%M:%S')}")
        print(f"  │  Phase  : {phase}")
        print(f"  │  SNR    : {snr_preview} dB")
        print(f"  │  GPX pts: {len(gpx_pts)}")
        print(f"  └{'─'*52}")

        y_seg_full = None   # load at save_sr only if user needs it

        while True:
            cmd = input(
                f"  ▶  [Enter]=accept  A:B=trim  p=play  n=skip  q=quit  h=help : "
            ).strip()

            if cmd.lower() == "h":
                print_help(); continue
            if cmd.lower() in ("n", "s"):
                print("  → Skipped."); break
            if cmd.lower() == "q":
                return manifest_rows, True   # quit flag

            # ── playback ──────────────────────────────────────────────────────
            if cmd.lower().startswith("p"):
                parts = cmd.split()
                # load at save_sr for playback
                try:
                    y_play, _ = load_audio_segment(path, seg_s, seg_e, save_sr)
                except Exception as e:
                    print(f"  ⚠  Load failed: {e}"); continue
                if len(parts) == 1:
                    print(f"  ♪ Playing segment ({seg_dur:.1f}s)…")
                    try_play(y_play, save_sr)
                elif len(parts) == 3:
                    try:
                        ps = max(0.0, float(parts[1]))
                        pe = min(seg_dur, float(parts[2]))
                        if pe <= ps: print("  end > start"); continue
                        print(f"  ♪ Playing {ps:.2f}s → {pe:.2f}s…")
                        try_play(y_play[int(ps*save_sr):int(pe*save_sr)], save_sr)
                    except ValueError:
                        print("  Use: p 10 16")
                else:
                    print("  Use: p  OR  p A B")
                del y_play; gc.collect()
                continue

            # ── accept as-is (Enter or 'y') ───────────────────────────────────
            if cmd in ("", "y"):
                try:
                    y_clip, _ = load_audio_segment(path, seg_s, seg_e, save_sr)
                except Exception as e:
                    print(f"  ⚠  Load failed: {e}"); break
                det_summary = {
                    "detector_logic": detector.logic,
                    "detected_start_s": seg_s,
                    "detected_end_s":   seg_e,
                    "trim_applied":     False,
                }
                row = save_clip_with_meta(
                    path, y_clip, save_sr, seg_s, seg_e,
                    source_type, clips_dir, f_start,
                    takeoff_utc, landing_utc, gpx_df, rpm_df, det_summary
                )
                manifest_rows.append(row)
                print(f"  ✔  Saved  phase={row['flight_phase']}  "
                      f"GPX={row['gpx_n_points']}  "
                      f"SNR={row.get('sig_snr_db','?')}dB")
                del y_clip; gc.collect()
                break

            # ── trim ranges ───────────────────────────────────────────────────
            try:
                ranges = parse_ranges(cmd, seg_dur)
            except Exception as e:
                print(f"  ⚠  {e}"); continue

            saved_now = 0
            for sub_s, sub_e in ranges:
                abs_s = seg_s + sub_s
                abs_e = seg_s + sub_e
                try:
                    y_clip, _ = load_audio_segment(path, abs_s, abs_e, save_sr)
                except Exception as e:
                    print(f"  ⚠  Load failed: {e}"); continue
                det_summary = {
                    "detector_logic":   detector.logic,
                    "detected_start_s": seg_s,
                    "detected_end_s":   seg_e,
                    "trim_applied":     True,
                    "trim_start_s":     abs_s,
                    "trim_end_s":       abs_e,
                }
                row = save_clip_with_meta(
                    path, y_clip, save_sr, abs_s, abs_e,
                    source_type, clips_dir, f_start,
                    takeoff_utc, landing_utc, gpx_df, rpm_df, det_summary
                )
                manifest_rows.append(row)
                saved_now += 1
                print(f"  ✔  Saved  phase={row['flight_phase']}  "
                      f"GPX={row['gpx_n_points']}  "
                      f"SNR={row.get('sig_snr_db','?')}dB")
                del y_clip; gc.collect()

            if saved_now:
                break   # move to next segment

    return manifest_rows, False


# =============================================================================
#  MANIFEST WRITER
# =============================================================================

def write_manifests(all_rows: List[Dict], manifest_dir: Path):
    mj = manifest_dir / "manual_clips_manifest.json"
    mc = manifest_dir / "manual_clips_manifest.csv"
    mj.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    if all_rows:
        keys = sorted({k for r in all_rows for k in r if not k.startswith("_")})
        with open(mc, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(all_rows)
    print(f"\n  📄  Manifest JSON : {mj}")
    print(f"  📄  Manifest CSV  : {mc}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Dunakeszi drone section picker — auto-detect + trim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source",            required=True)
    ap.add_argument("--output",            required=True)
    ap.add_argument("--gpx",               default=None)
    ap.add_argument("--rpm",               default=None)
    ap.add_argument("--mems-timestamps",   default=None)
    ap.add_argument("--flight-takeoff",
                    default=DEFAULT_TAKEOFF.strftime("%Y-%m-%dT%H:%M:%S"))
    ap.add_argument("--flight-landing",
                    default=DEFAULT_LANDING.strftime("%Y-%m-%dT%H:%M:%S"))
    ap.add_argument("--sr",           type=int,   default=DEFAULT_SR,
                    help="Sample rate for saved clips (default 22050)")
    ap.add_argument("--detection-sr", type=int,   default=DETECTION_SR,
                    help="Sample rate for detection pass (default 4000)")
    ap.add_argument("--logic",        default="or",  choices=["or","and"],
                    help="Detector combination logic (default: or)")
    ap.add_argument("--bpf-hz",       type=float, default=82.0,
                    help="Blade-pass fundamental frequency Hz (default 82)")
    ap.add_argument("--bpf-harmonics",type=int,   default=8,
                    help="Number of harmonics to check (default 8)")
    ap.add_argument("--bpf-threshold",type=float, default=3.0,
                    help="BPF energy × median threshold (default 3.0)")
    ap.add_argument("--rms-threshold-db", type=float, default=12.0,
                    help="dB above noise floor for RMS detector (default 12)")
    ap.add_argument("--min-segment-sec",  type=float, default=2.0,
                    help="Discard segments shorter than this (default 2.0)")
    ap.add_argument("--merge-gap-sec",    type=float, default=1.5,
                    help="Merge segments separated by less than this (default 1.5)")
    ap.add_argument("--non-recursive",    action="store_true")
    args = ap.parse_args()

    def _parse_dt(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ValueError(f"Cannot parse: {s!r}")

    takeoff_utc = _parse_dt(args.flight_takeoff)
    landing_utc = _parse_dt(args.flight_landing)
    dur_s = (landing_utc - takeoff_utc).total_seconds()

    print("=" * 72)
    print("  DUNAKESZI — Drone Section Picker  (auto-detect + trim)")
    print(f"  Takeoff : {takeoff_utc}  →  Landing : {landing_utc}  "
          f"({dur_s:.0f}s)")
    print(f"  Logic   : {args.logic.upper()}  |  "
          f"BPF {args.bpf_hz}Hz ×{args.bpf_threshold}  |  "
          f"RMS +{args.rms_threshold_db}dB")
    print("=" * 72)

    # ── load aux data ─────────────────────────────────────────────────────────
    gpx_df = None
    if args.gpx and Path(args.gpx).is_file():
        try:
            gpx_df = pd.read_csv(args.gpx, parse_dates=["time"])
            gpx_df["time"] = pd.to_datetime(gpx_df["time"], utc=True, errors="coerce")
            print(f"  GPX  : {len(gpx_df)} trackpoints")
        except Exception as e:
            print(f"  ⚠  GPX: {e}")

    rpm_df = None
    if args.rpm and Path(args.rpm).is_file():
        try:
            rpm_df = pd.read_csv(args.rpm)
            print(f"  RPM  : {len(rpm_df)} rows")
        except Exception as e:
            print(f"  ⚠  RPM: {e}")

    mems_table = None
    ts_path = (Path(args.mems_timestamps) if args.mems_timestamps
               else Path(args.source) / MEMS_TIMESTAMPS_FILE)
    if ts_path.is_file():
        try:
            mems_table = pd.read_csv(ts_path)
            print(f"  MEMS timestamps: {len(mems_table)} entries")
        except Exception as e:
            print(f"  ⚠  MEMS-TS: {e}")

    # ── find files ────────────────────────────────────────────────────────────
    source = Path(args.source)
    glob_fn = source.rglob if not args.non_recursive else source.glob
    files = sorted(p for p in glob_fn("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"\n  ⚠  No audio files in {source}"); return
    print(f"\n  Found {len(files)} audio file(s).\n")

    # ── output dirs ───────────────────────────────────────────────────────────
    output      = Path(args.output)
    clips_dir   = output / "clean_drone_sections"
    manifest_dir = output / "manifests"
    det_dir     = output / "detection"
    for d in (clips_dir, manifest_dir, det_dir):
        ensure_dir(d)

    # ── detector ──────────────────────────────────────────────────────────────
    detector = DroneDetector(
        sr               = args.detection_sr,
        bpf_hz           = args.bpf_hz,
        n_harmonics      = args.bpf_harmonics,
        bpf_threshold    = args.bpf_threshold,
        rms_threshold_db = args.rms_threshold_db,
        min_segment_sec  = args.min_segment_sec,
        merge_gap_sec    = args.merge_gap_sec,
        logic            = args.logic,
    )

    # ── main loop ─────────────────────────────────────────────────────────────
    all_rows: List[Dict] = []
    for idx, path in enumerate(files, 1):
        print(f"\n  [{idx}/{len(files)}]", flush=True)
        rows, quit_req = process_file(
            path, detector, clips_dir, det_dir, args.sr,
            gpx_df, rpm_df, mems_table, takeoff_utc, landing_utc,
        )
        all_rows.extend(rows)
        if quit_req:
            print("  Quit requested — saving manifest and exiting.")
            break

    # ── final manifest ────────────────────────────────────────────────────────
    write_manifests(all_rows, manifest_dir)

    print("\n" + "=" * 72)
    print("  DONE")
    print(f"  Files processed : {min(idx, len(files))}")
    print(f"  Clips saved     : {len(all_rows)}")
    print(f"  Output          : {output}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()