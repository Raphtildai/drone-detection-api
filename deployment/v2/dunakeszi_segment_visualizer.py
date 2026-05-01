#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dunakeszi_segment_visualizer.py
────────────────────────────────
Acoustic evidence visualizer for extracted Dunakeszi segments.

Generates a self-contained interactive HTML report with:
  • Waveform (all 3 channels)
  • STFT spectrogram (0–500 Hz, showing harmonic comb structure)
  • FFT spectrum with harmonic markers and training BPF range overlay
  • BPF energy ratio bar with threshold annotations
  • GCC-PHAT cross-correlation (spatial evidence of TDOA)
  • Harmonic score (comb structure detector)
  • Per-segment verdict: DRONE CONFIRMED / PROBABLE / WEAK

Usage
─────
    # From a folder of extracted segments:
    python dunakeszi_segment_visualizer.py --input-dir extracted_P/

    # From a ZIP:
    python dunakeszi_segment_visualizer.py --input-dir extracted_P.zip

    # Combined folders:
    python dunakeszi_segment_visualizer.py --input-dir extracted_J/ extracted_P/

    # Output:
    # drone_evidence_report.html  — open in any browser, no server needed
"""

import argparse
import json
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed.  pip install soundfile"); sys.exit(1)

try:
    import scipy.signal
    import scipy.fft
except ImportError:
    print("ERROR: scipy not installed.  pip install scipy"); sys.exit(1)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not installed — mel spectrogram disabled")


# ── Acoustic analysis ─────────────────────────────────────────────────────────

def compute_fft_spectrum(y: np.ndarray, sr: int, max_freq: float = 500.0):
    """Return (freqs, magnitudes_db) up to max_freq Hz."""
    N = len(y)
    fft_mag = np.abs(scipy.fft.rfft(y * np.hanning(N)))
    freqs   = scipy.fft.rfftfreq(N, 1/sr)
    mask    = freqs <= max_freq
    mag_db  = 20 * np.log10(fft_mag[mask] + 1e-10)
    # Normalise so max = 0 dB
    mag_db  -= mag_db.max()
    return freqs[mask].tolist(), mag_db.tolist()


def find_dominant_freq(y: np.ndarray, sr: int,
                       fmin: float = 20.0, fmax: float = 400.0) -> float:
    N      = len(y)
    fft_m  = np.abs(scipy.fft.rfft(y * np.hanning(N)))
    freqs  = scipy.fft.rfftfreq(N, 1/sr)
    mask   = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float('nan')
    idx = np.argmax(fft_m[mask])
    return float(freqs[mask][idx])


def compute_harmonic_score(y: np.ndarray, sr: int,
                            f0: float, n_harmonics: int = 5,
                            bw_hz: float = 8.0) -> float:
    """
    Comb filter score: fraction of total power concentrated at
    f0, 2f0, 3f0 … n*f0 (±bw_hz each).
    Returns 0–1. >0.30 is strong harmonic structure (drone comb).
    """
    if f0 <= 0 or np.isnan(f0):
        return 0.0
    nyq   = sr / 2.0
    total = float(np.mean(y**2)) + 1e-12
    comb  = 0.0
    for k in range(1, n_harmonics + 1):
        fc = f0 * k
        if fc + bw_hz >= nyq:
            break
        lo  = max(fc - bw_hz, 1.0)
        hi  = min(fc + bw_hz, nyq - 1.0)
        sos = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype='band', output='sos')
        band = scipy.signal.sosfilt(sos, y.astype(np.float64))
        comb += float(np.mean(band**2))
    return float(np.clip(comb / total, 0.0, 1.0))


def compute_energy_ratio(y: np.ndarray, sr: int,
                          fmin: float = 25.0, fmax: float = 350.0) -> float:
    """Fraction of total spectral power between fmin and fmax Hz."""
    fft_p = np.abs(scipy.fft.rfft(y))**2
    freqs = scipy.fft.rfftfreq(len(y), 1/sr)
    mask  = (freqs >= fmin) & (freqs <= fmax)
    total = np.sum(fft_p) + 1e-12
    return float(np.clip(np.sum(fft_p[mask]) / total, 0.0, 1.0))


def gcc_phat(x: np.ndarray, y: np.ndarray, sr: int,
             max_lag_s: float = 0.015) -> tuple:
    """
    GCC-PHAT cross-correlation.
    Returns (lags_ms, gcc_normalised, peak_lag_ms).
    """
    N   = len(x) + len(y) - 1
    Xf  = scipy.fft.rfft(x, n=N)
    Yf  = scipy.fft.rfft(y, n=N)
    R   = Xf * np.conj(Yf)
    R  /= (np.abs(R) + 1e-10)
    gcc = np.real(scipy.fft.irfft(R, n=N))
    gcc = np.fft.fftshift(gcc)

    lags_s = (np.arange(N) - N//2) / sr
    mask   = np.abs(lags_s) <= max_lag_s

    lags_ms  = (lags_s[mask] * 1000).tolist()
    gcc_norm = gcc[mask]
    gcc_norm = (gcc_norm / (np.max(np.abs(gcc_norm)) + 1e-10)).tolist()

    peak_idx   = np.argmax(np.abs(gcc[mask]))
    peak_lag_ms = float(lags_s[mask][peak_idx] * 1000)

    return lags_ms, gcc_norm, peak_lag_ms


def compute_stft_spectrogram(y: np.ndarray, sr: int,
                              max_freq: float = 500.0,
                              n_fft: int = 2048,
                              hop: int = 256) -> dict:
    """Returns dict with times, freqs (≤max_freq), and log-magnitude matrix."""
    f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag  = 20 * np.log10(np.abs(Zxx) + 1e-10)
    mask = f <= max_freq
    mag_norm = mag[mask]
    vmin = np.percentile(mag_norm, 5)
    vmax = np.percentile(mag_norm, 99)
    mag_scaled = np.clip((mag_norm - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return {
        "times": t.tolist(),
        "freqs": f[mask].tolist(),
        "mag":   [row.tolist() for row in mag_scaled],
    }


def downsample_waveform(y: np.ndarray, target_pts: int = 1500) -> list:
    """Downsample for display without aliasing."""
    if len(y) <= target_pts:
        return y.tolist()
    step = len(y) // target_pts
    # Take max absolute value in each window
    pts = [float(y[i:i+step][np.argmax(np.abs(y[i:i+step]))]) 
           for i in range(0, len(y)-step, step)]
    return pts


def compute_harmonic_magnitudes(y: np.ndarray, sr: int,
                                  f0: float, n: int = 6) -> list:
    """Return [freq_hz, magnitude_db] for f0 * k, k=1..n."""
    fft_m  = np.abs(scipy.fft.rfft(y * np.hanning(len(y))))
    freqs  = scipy.fft.rfftfreq(len(y), 1/sr)
    result = []
    for k in range(1, n+1):
        fc = f0 * k
        if fc >= sr/2:
            break
        idx = np.argmin(np.abs(freqs - fc))
        mag = 20 * np.log10(fft_m[idx] + 1e-10)
        result.append({"hz": round(fc, 1), "db": round(float(mag), 1)})
    return result


def drone_verdict(rms_db: float, dom_freq: float,
                  energy_ratio: float, harmonic_score: float,
                  n_drones: int) -> dict:
    """
    Traffic-light verdict based on acoustic evidence.
    Returns {"label": str, "confidence": float, "reasons": [str]}.
    """
    score   = 0.0
    reasons = []

    if rms_db > -35:
        score += 0.2
        reasons.append(f"RMS {rms_db:.1f} dB (audible signal)")
    else:
        reasons.append(f"⚠ RMS {rms_db:.1f} dB (weak signal)")

    if 20 <= dom_freq <= 400:
        score += 0.25
        reasons.append(f"Dominant peak at {dom_freq:.1f} Hz (drone range 20–400 Hz)")
    else:
        reasons.append(f"⚠ Peak at {dom_freq:.1f} Hz (outside 20–400 Hz)")

    if energy_ratio > 0.30:
        score += 0.30
        reasons.append(f"Energy ratio {energy_ratio:.3f} > 0.30 (strong drone band concentration)")
    elif energy_ratio > 0.10:
        score += 0.20
        reasons.append(f"Energy ratio {energy_ratio:.3f} > 0.10 (moderate drone band concentration)")
    elif energy_ratio > 0.03:
        score += 0.10
        reasons.append(f"Energy ratio {energy_ratio:.3f} > 0.03 (minimal drone signal above noise)")
    else:
        reasons.append(f"⚠ Energy ratio {energy_ratio:.3f} < 0.03 (no drone band concentration)")

    if harmonic_score > 0.30:
        score += 0.25
        reasons.append(f"Harmonic score {harmonic_score:.3f} (strong comb structure — definitive drone signature)")
    elif harmonic_score > 0.10:
        score += 0.15
        reasons.append(f"Harmonic score {harmonic_score:.3f} (moderate harmonic structure)")
    else:
        reasons.append(f"⚠ Harmonic score {harmonic_score:.3f} (weak harmonic structure)")

    if n_drones > 1:
        reasons.append(f"Multi-drone: {n_drones} drones → beat frequencies may shift dominant peak")

    if score >= 0.70:
        label = "DRONE CONFIRMED"
    elif score >= 0.45:
        label = "DRONE PROBABLE"
    else:
        label = "WEAK / UNCERTAIN"

    return {"label": label, "confidence": round(score, 3), "reasons": reasons}


# ── Segment loader ────────────────────────────────────────────────────────────

def load_segment(seg_dir: Path, stem: str) -> Optional[dict]:
    """Load one segment and compute all acoustic evidence metrics."""
    ch_paths = [seg_dir / f"{stem}_ch{i}.wav" for i in range(3)]
    lbl_path = seg_dir / f"{stem}_label.json"

    if not all(p.exists() for p in ch_paths):
        return None

    channels = []
    for p in ch_paths:
        y, sr = sf.read(str(p))
        channels.append(y.astype(np.float32))

    y0 = channels[0]

    label = {}
    if lbl_path.exists():
        label = json.loads(lbl_path.read_text())

    # Flatten nested drone key if present (extractor format)
    if "drone" in label and isinstance(label["drone"], dict):
        drone = label["drone"]
        label.setdefault("azimuth_deg",  drone.get("azimuth"))
        label.setdefault("distance_m",   drone.get("distance"))
        label.setdefault("height_m",     drone.get("height"))

    # Core metrics
    dom_freq      = find_dominant_freq(y0, sr)
    energy_ratio  = compute_energy_ratio(y0, sr)
    harmonic_score = compute_harmonic_score(y0, sr, dom_freq) if not np.isnan(dom_freq) else 0.0
    rms_db        = float(20 * np.log10(np.sqrt(np.mean(y0**2)) + 1e-10))
    rms_per_ch    = [float(20 * np.log10(np.sqrt(np.mean(ch**2)) + 1e-10)) for ch in channels]

    # Spectral data
    freqs_fft, mag_fft = compute_fft_spectrum(y0, sr, max_freq=500)
    harmonics          = compute_harmonic_magnitudes(y0, sr, dom_freq) if not np.isnan(dom_freq) else []
    spectrogram        = compute_stft_spectrogram(y0, sr, max_freq=500)

    # TDOA evidence
    _, gcc01, lag01 = gcc_phat(channels[0], channels[1], sr)
    _, gcc02, lag02 = gcc_phat(channels[0], channels[2], sr)
    # For display only use a subset of points
    step    = max(1, len(gcc01)//400)
    lags_ms = [i/sr*1000 for i in range(-len(gcc01)//2, len(gcc01)//2, step)][:400]
    gcc01_d = gcc01[::step][:400]
    gcc02_d = gcc02[::step][:400]

    # Waveform display
    waveforms = [downsample_waveform(ch, 1500) for ch in channels]

    # Verdict
    n_drones = label.get("n_drones", 1)
    verdict  = drone_verdict(rms_db, dom_freq, energy_ratio, harmonic_score, n_drones)

    return {
        "stem":           stem,
        "sr":             int(sr),
        "duration_s":     float(len(y0) / sr),
        "label":          label,
        "rms_db":         round(rms_db, 2),
        "rms_per_ch":     [round(r, 2) for r in rms_per_ch],
        "dom_freq_hz":    round(dom_freq, 2) if not np.isnan(dom_freq) else None,
        "energy_ratio":   round(energy_ratio, 4),
        "harmonic_score": round(harmonic_score, 4),
        "harmonics":      harmonics,
        "verdict":        verdict,
        "waveforms":      waveforms,
        "fft_freqs":      freqs_fft,
        "fft_mag":        mag_fft,
        "spectrogram":    spectrogram,
        "gcc_lags_ms":    lags_ms,
        "gcc_01":         gcc01_d,
        "gcc_02":         gcc02_d,
        "gcc_lag01_ms":   round(lag01, 4),
        "gcc_lag02_ms":   round(lag02, 4),
    }


def collect_segments(input_dirs: List[Path]) -> tuple:
    """Returns (seg_dir, stems) for all *_ch0.wav files found."""
    results = []
    for d in input_dirs:
        for ch0 in sorted(d.glob("*_ch0.wav")):
            stem = ch0.stem.replace("_ch0", "")
            results.append((d, stem))
    return results


# ── HTML generator ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drone Acoustic Evidence Report — Dunakeszi 2025-10-20</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:        #0a0c10;
    --surface:   #111419;
    --surface2:  #181c24;
    --border:    #242830;
    --accent:    #00d4aa;
    --accent2:   #ff6b35;
    --accent3:   #7b61ff;
    --warn:      #ffb800;
    --danger:    #ff4455;
    --confirmed: #00d4aa;
    --probable:  #7b61ff;
    --weak:      #ff6b35;
    --text:      #e8eaf0;
    --text2:     #8891a4;
    --text3:     #525a6b;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --radius:    8px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans);
         font-size: 14px; line-height: 1.5; }

  /* ── Header ── */
  .header {
    border-bottom: 1px solid var(--border);
    padding: 24px 32px 20px;
    display: flex; align-items: flex-end; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    background: linear-gradient(180deg, #0a0c10 80%, transparent);
  }
  .header-title { font-family: var(--mono); font-size: 11px;
                  font-weight: 600; letter-spacing: 0.12em;
                  color: var(--accent); text-transform: uppercase; }
  .header-sub { font-size: 22px; font-weight: 500; color: var(--text);
                margin-top: 4px; letter-spacing: -0.02em; }
  .header-meta { font-family: var(--mono); font-size: 10px;
                 color: var(--text3); text-align: right; line-height: 1.8; }

  /* ── Summary bar ── */
  .summary-bar {
    display: flex; gap: 0; border-bottom: 1px solid var(--border);
    padding: 0 32px;
  }
  .summary-stat {
    padding: 16px 28px 16px 0; margin-right: 28px;
    border-right: 1px solid var(--border);
  }
  .summary-stat:last-child { border-right: none; }
  .summary-stat .val { font-family: var(--mono); font-size: 24px;
                       font-weight: 700; color: var(--accent); }
  .summary-stat .lbl { font-size: 11px; color: var(--text3);
                       text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

  /* ── Segment cards ── */
  .segments { padding: 24px 32px; display: flex; flex-direction: column; gap: 32px; }

  .seg-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }

  .seg-header {
    padding: 16px 20px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .seg-id { font-family: var(--mono); font-size: 13px; font-weight: 600;
            color: var(--text); }
  .verdict-badge {
    font-family: var(--mono); font-size: 10px; font-weight: 700;
    padding: 4px 10px; border-radius: 4px; letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .verdict-CONFIRMED { background: rgba(0,212,170,0.15); color: var(--confirmed);
                       border: 1px solid rgba(0,212,170,0.3); }
  .verdict-PROBABLE  { background: rgba(123,97,255,0.15); color: var(--probable);
                       border: 1px solid rgba(123,97,255,0.3); }
  .verdict-WEAK      { background: rgba(255,107,53,0.15);  color: var(--weak);
                       border: 1px solid rgba(255,107,53,0.3); }

  .seg-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  .pill { font-family: var(--mono); font-size: 10px; padding: 3px 8px;
          border-radius: 3px; background: var(--surface2); color: var(--text2);
          border: 1px solid var(--border); }
  .pill.hi { color: var(--accent); border-color: rgba(0,212,170,0.3); }
  .pill.warn { color: var(--warn); border-color: rgba(255,184,0,0.3); }

  /* ── Metrics row ── */
  .metrics-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    border-bottom: 1px solid var(--border);
  }
  .metric-cell {
    padding: 14px 20px; border-right: 1px solid var(--border);
  }
  .metric-cell:last-child { border-right: none; }
  .metric-val { font-family: var(--mono); font-size: 20px; font-weight: 600; }
  .metric-lbl { font-size: 10px; color: var(--text3); text-transform: uppercase;
                letter-spacing: 0.08em; margin-top: 3px; }
  .metric-sub { font-size: 10px; color: var(--text3); margin-top: 1px; font-family: var(--mono); }
  .col-green  { color: var(--confirmed); }
  .col-purple { color: var(--probable); }
  .col-orange { color: var(--accent2); }
  .col-yellow { color: var(--warn); }

  /* ── Plots grid ── */
  .plots-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 1px; background: var(--border);
  }
  .plot-cell {
    background: var(--surface); padding: 16px 18px;
  }
  .plot-cell.wide { grid-column: span 2; }
  .plot-cell.full { grid-column: span 3; }

  .plot-title {
    font-family: var(--mono); font-size: 10px; font-weight: 600;
    color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
  .plot-title .dot { width: 6px; height: 6px; border-radius: 50%; }

  canvas { display: block; width: 100%; }

  /* ── Reasons ── */
  .reasons { padding: 14px 20px; border-top: 1px solid var(--border);
             display: flex; flex-wrap: wrap; gap: 8px; }
  .reason {
    font-family: var(--mono); font-size: 10px; padding: 4px 10px;
    border-radius: 4px; background: var(--surface2);
    color: var(--text2); border: 1px solid var(--border);
  }
  .reason.good { color: var(--confirmed); border-color: rgba(0,212,170,0.2);
                 background: rgba(0,212,170,0.06); }
  .reason.bad  { color: var(--warn); border-color: rgba(255,184,0,0.2);
                 background: rgba(255,184,0,0.06); }

  /* ── Training gap banner ── */
  .gap-banner {
    margin: 0 32px 24px; padding: 14px 20px;
    background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.25);
    border-radius: 8px; display: flex; align-items: flex-start; gap: 12px;
  }
  .gap-banner .icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
  .gap-banner .text { font-size: 12px; line-height: 1.6; color: var(--text2); }
  .gap-banner strong { color: var(--accent2); }

  /* ── Domain gap chart ── */
  .gap-chart { padding: 24px 32px 8px; }
  .gap-chart-title {
    font-family: var(--mono); font-size: 10px; color: var(--text3);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;
  }
  .freq-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .freq-label { font-family: var(--mono); font-size: 11px; color: var(--text2);
                min-width: 180px; }
  .freq-track { flex: 1; height: 18px; background: var(--surface2);
                border-radius: 3px; position: relative; overflow: visible; }
  .freq-fill { height: 100%; border-radius: 3px; position: absolute; top: 0; }
  .freq-peak { position: absolute; top: -3px; width: 2px; height: 24px;
               border-radius: 1px; }

  .scrolltop { position: fixed; bottom: 24px; right: 24px;
               background: var(--surface); border: 1px solid var(--border);
               color: var(--text2); padding: 8px 12px; border-radius: 6px;
               font-family: var(--mono); font-size: 11px; cursor: pointer;
               transition: all 0.15s; }
  .scrolltop:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="header-title">Acoustic Evidence Report</div>
    <div class="header-sub">Dunakeszi 2025-10-20 · Drone Signal Analysis</div>
  </div>
  <div class="header-meta" id="header-meta"></div>
</div>

<div id="summary-bar" class="summary-bar"></div>

<div class="gap-chart">
  <div class="gap-chart-title">Training distribution vs measured frequencies — domain gap visualisation</div>
  <div id="freq-chart"></div>
</div>

<div class="gap-banner">
  <div class="icon">⚠</div>
  <div class="text">
    <strong>Domain gap detected.</strong>
    The drone acoustic model was trained on BPF fundamentals of <strong>80–360 Hz</strong>
    (Mavic Pro/2/Mini, generic quad) mapped to mel bands 2–8.
    All extracted Dunakeszi segments show dominant energy at <strong>34–62 Hz</strong> (mel band 1–2) —
    altitude-dependent blade slowdown pushes the fundamental below the training distribution.
    The CNN scores 0.03–0.04 (near-random) because it never saw this frequency regime.
    Heuristic scores 0.73–0.83 correctly confirm drone presence from harmonic structure alone.
    <strong>Fix:</strong> add <code>dji_high_altitude</code> BPF profile (30–90 Hz) and retrain.
  </div>
</div>

<div class="segments" id="segments"></div>

<button class="scrolltop" onclick="window.scrollTo(0,0)">↑ top</button>

<script>
// ═══════════════════════════════════════════════════════════
//  Embedded data
// ═══════════════════════════════════════════════════════════
const SEGMENTS = __SEGMENTS_DATA__;

// ═══════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);

function lerp(a, b, t) { return a + (b-a)*t; }

function cssVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
}

const COLORS = {
  accent:   '#00d4aa',
  accent2:  '#ff6b35',
  accent3:  '#7b61ff',
  warn:     '#ffb800',
  text2:    '#8891a4',
  text3:    '#525a6b',
  border:   '#242830',
  surface2: '#181c24',
  ch1:      '#00d4aa',
  ch2:      '#7b61ff',
  ch3:      '#ff6b35',
};

// ═══════════════════════════════════════════════════════════
//  Summary bar
// ═══════════════════════════════════════════════════════════
function buildSummary() {
  const n = SEGMENTS.length;
  const confirmed = SEGMENTS.filter(s => s.verdict.label.includes('CONFIRMED')).length;
  const probable  = SEGMENTS.filter(s => s.verdict.label.includes('PROBABLE')).length;
  const avgRatio  = (SEGMENTS.reduce((a,s) => a+s.energy_ratio, 0)/n).toFixed(3);
  const avgHarm   = (SEGMENTS.reduce((a,s) => a+s.harmonic_score, 0)/n).toFixed(3);
  const maxDrones = Math.max(...SEGMENTS.map(s => s.label.n_drones || 1));

  $('summary-bar').innerHTML = `
    <div class="summary-stat"><div class="val">${n}</div><div class="lbl">Segments</div></div>
    <div class="summary-stat"><div class="val" style="color:var(--confirmed)">${confirmed}</div><div class="lbl">Confirmed</div></div>
    <div class="summary-stat"><div class="val" style="color:var(--probable)">${probable}</div><div class="lbl">Probable</div></div>
    <div class="summary-stat"><div class="val">${avgRatio}</div><div class="lbl">Avg Energy Ratio</div></div>
    <div class="summary-stat"><div class="val">${avgHarm}</div><div class="lbl">Avg Harmonic Score</div></div>
    <div class="summary-stat"><div class="val" style="color:var(--accent3)">${maxDrones}</div><div class="lbl">Max Drones</div></div>
  `;

  const now = new Date().toISOString().slice(0,19).replace('T',' ');
  $('header-meta').innerHTML =
    `Generated ${now}<br>BK-6-E array · 22 050 Hz · 3.0 s clips`;
}

// ═══════════════════════════════════════════════════════════
//  Frequency domain gap chart
// ═══════════════════════════════════════════════════════════
function buildFreqChart() {
  const profiles = [
    { label: 'Mavic Mini (training)',       lo: 260, hi: 620, peak: 360,  col: '#525a6b' },
    { label: 'Mavic Pro/2 (training)',      lo: 160, hi: 340, peak: 200,  col: '#525a6b' },
    { label: 'DJI Phantom / generic quad',  lo:  80, hi: 350, peak: 110,  col: '#525a6b' },
    { label: 'hexarotor (training)',         lo:  60, hi: 195, peak:  75,  col: '#7b61ff' },
    ...SEGMENTS.map(s => ({
      label: `${s.stem.replace(/_/g,' ')} (measured)`,
      lo: (s.dom_freq_hz||0)*0.85, hi: (s.dom_freq_hz||0)*4.5,
      peak: s.dom_freq_hz,
      col: s.verdict.label.includes('CONFIRMED') ? '#00d4aa' : '#ff6b35',
    }))
  ];

  const maxFreq = 700;
  const html = profiles.map(p => {
    const pct   = v => `${Math.min(100, v/maxFreq*100).toFixed(1)}%`;
    const w     = Math.max(0, p.hi - p.lo) / maxFreq * 100;
    return `
      <div class="freq-bar">
        <div class="freq-label">${p.label}</div>
        <div class="freq-track">
          <div class="freq-fill" style="left:${pct(p.lo)};width:${w.toFixed(1)}%;background:${p.col}22;border:1px solid ${p.col}55;"></div>
          ${p.peak ? `<div class="freq-peak" style="left:calc(${pct(p.peak)} - 1px);background:${p.col};"></div>` : ''}
        </div>
        <span style="font-family:var(--mono);font-size:10px;color:${p.col};min-width:60px">
          ${p.peak ? Math.round(p.peak)+' Hz' : ''}
        </span>
      </div>`;
  }).join('');
  $('freq-chart').innerHTML = html + `
    <div style="font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:8px;display:flex;gap:16px;">
      <span>0 Hz</span><span style="margin-left:${100/7*1}%">100</span>
      <span style="margin-left:${100/7*1}%">200</span>
      <span style="margin-left:${100/7*1}%">300</span>
      <span style="margin-left:${100/7*1}%">400</span>
      <span style="margin-left:${100/7*1}%">500</span>
      <span style="margin-left:${100/7*1}%">600</span>
      <span>700 Hz</span>
    </div>`;
}

// ═══════════════════════════════════════════════════════════
//  Canvas drawing helpers
// ═══════════════════════════════════════════════════════════
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || canvas.offsetWidth || 400;
  const h = canvas.offsetHeight || 120;
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function drawWaveform(canvas, channels) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = '#111419';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();

  const chColors = [COLORS.ch1, COLORS.ch2, COLORS.ch3];
  channels.forEach((ch, ci) => {
    ctx.strokeStyle = chColors[ci];
    ctx.globalAlpha = ci === 0 ? 1.0 : 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ch.forEach((v, i) => {
      const x = (i / (ch.length-1)) * w;
      const y = h/2 - v * (h/2 - 4) * 0.9;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  // RMS indicator line (for ch0)
  const rms = Math.sqrt(channels[0].reduce((a,v)=>a+v*v,0)/channels[0].length);
  const rmsY = h/2 - rms*(h/2-4)*0.9;
  ctx.strokeStyle = COLORS.warn + '80';
  ctx.lineWidth = 1;
  ctx.setLineDash([3,4]);
  ctx.beginPath(); ctx.moveTo(0, rmsY); ctx.lineTo(w, rmsY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, h-rmsY); ctx.lineTo(w, h-rmsY); ctx.stroke();
  ctx.setLineDash([]);
}

function drawSpectrum(canvas, freqs, mags, dom_freq, harmonics) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = '#111419';
  ctx.fillRect(0, 0, w, h);

  const maxFreq = freqs[freqs.length-1];
  const minMag  = Math.min(...mags);
  const maxMag  = Math.max(...mags);
  const range   = maxMag - minMag || 1;

  // Training BPF zone (80–360 Hz)
  const bpfLoX = (80  / maxFreq) * w;
  const bpfHiX = (360 / maxFreq) * w;
  ctx.fillStyle = '#ff6b3508';
  ctx.fillRect(bpfLoX, 0, bpfHiX-bpfLoX, h);
  ctx.strokeStyle = '#ff6b3530';
  ctx.lineWidth = 1;
  ctx.setLineDash([2,3]);
  ctx.beginPath(); ctx.moveTo(bpfLoX, 0); ctx.lineTo(bpfLoX, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bpfHiX, 0); ctx.lineTo(bpfHiX, h); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ff6b3560';
  ctx.font = '9px JetBrains Mono';
  ctx.fillText('training range', bpfLoX+4, 12);

  // Spectrum line
  ctx.strokeStyle = COLORS.accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  freqs.forEach((f, i) => {
    const x = (f / maxFreq) * w;
    const y = h - 4 - ((mags[i] - minMag) / range) * (h - 8);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Harmonic markers
  if (harmonics && harmonics.length > 0) {
    harmonics.forEach((hm, k) => {
      const x = (hm.hz / maxFreq) * w;
      if (x < 0 || x > w) return;
      ctx.strokeStyle = k === 0 ? '#ffb800' : '#ffb80060';
      ctx.lineWidth = k === 0 ? 2 : 1;
      ctx.setLineDash(k === 0 ? [] : [2,3]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.setLineDash([]);
      if (k < 4) {
        ctx.fillStyle = k === 0 ? COLORS.warn : COLORS.warn+'80';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`${k===0?'f₀':''}${hm.hz}`, x+2, h-4-k*10);
      }
    });
  }

  // Freq axis ticks
  ctx.fillStyle = COLORS.text3;
  ctx.font = '9px JetBrains Mono';
  [50,100,150,200,250,300,350,400,450,500].forEach(f => {
    const x = (f / maxFreq) * w;
    ctx.fillText(f, x-8, h-2);
  });
}

function drawSpectrogram(canvas, spectData) {
  if (!spectData || !spectData.mag || spectData.mag.length === 0) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const nFreq = spectData.mag.length;
  const nTime = spectData.times.length;
  if (nTime === 0 || nFreq === 0) return;

  const cellW = w / nTime;
  const cellH = h / nFreq;

  for (let fi = 0; fi < nFreq; fi++) {
    for (let ti = 0; ti < nTime; ti++) {
      const v = spectData.mag[fi][ti] || 0;
      const r = Math.round(lerp(10,  0,   v) + lerp(0, 0,   v));
      const g = Math.round(lerp(18,  212, v));
      const b = Math.round(lerp(24,  170, v));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(
        Math.round(ti * cellW), Math.round((nFreq-1-fi) * cellH),
        Math.ceil(cellW)+1, Math.ceil(cellH)+1
      );
    }
  }

  // Freq axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '9px JetBrains Mono';
  const maxF = spectData.freqs[spectData.freqs.length-1];
  [100, 200, 300, 400].forEach(f => {
    if (f > maxF) return;
    const y = h - (f / maxF) * h;
    ctx.fillText(f+'Hz', 4, y+3);
  });
}

function drawGCC(canvas, lags, gcc01, gcc02, lag01, lag02) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = '#111419';
  ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  // Zero lag
  ctx.beginPath(); ctx.moveTo(w/2,0); ctx.lineTo(w/2,h); ctx.stroke();

  const drawLine = (data, color, alpha=1) => {
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((v,i) => {
      const x = (i/(data.length-1))*w;
      const y = h/2 - v*(h/2-4);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  };

  drawLine(gcc01, COLORS.ch2, 1);
  drawLine(gcc02, COLORS.ch3, 0.7);

  // Peak markers
  const peakLagToX = lag => {
    const maxLag = lags[lags.length-1];
    return ((lag/1000 + maxLag/1000) / (2*maxLag/1000)) * w;
  };
  [
    {lag: lag01, col: COLORS.ch2, label: `ch0↔1: ${lag01.toFixed(3)}ms`},
    {lag: lag02, col: COLORS.ch3, label: `ch0↔2: ${lag02.toFixed(3)}ms`},
  ].forEach(({lag, col, label}) => {
    const x = peakLagToX(lag);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = col;
    ctx.font = '9px JetBrains Mono';
    ctx.fillText(label, x+3, 14);
  });

  // Lag axis
  ctx.fillStyle = COLORS.text3;
  ctx.font = '9px JetBrains Mono';
  if (lags.length > 0) {
    const maxLag = Math.abs(lags[lags.length-1]).toFixed(1);
    ctx.fillText(`-${maxLag}ms`, 4, h-3);
    ctx.fillText(`+${maxLag}ms`, w-40, h-3);
    ctx.fillText('0', w/2-4, h-3);
  }
}

function drawEnergyBar(canvas, ratio, harmScore) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = '#111419';
  ctx.fillRect(0,0,w,h);

  const barH = 22;
  const gap  = 14;
  const labelW = 140;
  const barW = w - labelW - 16;

  const drawBar = (y, label, value, maxVal, color, thresholds) => {
    ctx.fillStyle = COLORS.text2;
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(label, 0, y+15);

    const trackX = labelW;
    ctx.fillStyle = COLORS.surface2;
    ctx.fillRect(trackX, y, barW, barH);

    const fillW = Math.min(1, value/maxVal) * barW;
    ctx.fillStyle = color;
    ctx.fillRect(trackX, y, fillW, barH);

    // Value text
    ctx.fillStyle = '#fff';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(value.toFixed(4), trackX + fillW + 6, y+15);

    // Threshold lines
    thresholds.forEach(({v, label:tl, col}) => {
      const tx = trackX + Math.min(1, v/maxVal)*barW;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3,2]);
      ctx.beginPath(); ctx.moveTo(tx,y-2); ctx.lineTo(tx,y+barH+2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.font = '8px JetBrains Mono';
      ctx.fillText(tl, tx+2, y-4);
    });
  };

  drawBar(8,  'Energy Ratio',   ratio,    1.0, COLORS.accent+'cc', [
    {v:0.03, label:'0.03 (relaxed)', col:'#ffb800'},
    {v:0.30, label:'0.30 (original)', col:'#ff4455'},
  ]);
  drawBar(8+barH+gap, 'Harmonic Score', harmScore, 1.0, COLORS.accent3+'cc', [
    {v:0.10, label:'0.10', col:'#ffb800'},
    {v:0.30, label:'0.30', col:'#00d4aa'},
  ]);
}

// ═══════════════════════════════════════════════════════════
//  Build segment cards
// ═══════════════════════════════════════════════════════════
function verdictClass(label) {
  if (label.includes('CONFIRMED')) return 'CONFIRMED';
  if (label.includes('PROBABLE'))  return 'PROBABLE';
  return 'WEAK';
}

function buildSegmentCard(seg, idx) {
  const vc = verdictClass(seg.verdict.label);
  const lbl = seg.label;
  const n_drones = lbl.n_drones || 1;
  const az   = lbl.azimuth_deg != null ? lbl.azimuth_deg.toFixed(1)+'°' : '—';
  const dist = lbl.distance_m  != null ? lbl.distance_m.toFixed(0)+'m'  : '—';
  const ht   = lbl.height_m   != null ? lbl.height_m.toFixed(0)+'m'    : '—';
  const maneuver = (lbl.maneuver_type||'').replace(/_/g,' ');

  const rmsCol   = seg.rms_db > -30 ? 'col-green' : seg.rms_db > -40 ? 'col-yellow' : '';
  const ratioCol = seg.energy_ratio > 0.30 ? 'col-green' : seg.energy_ratio > 0.10 ? 'col-purple' : 'col-orange';
  const harmCol  = seg.harmonic_score > 0.30 ? 'col-green' : seg.harmonic_score > 0.10 ? 'col-purple' : 'col-orange';
  const freqCol  = (seg.dom_freq_hz||0) < 80 ? 'col-orange' : 'col-green';

  const reasonsHtml = (seg.verdict.reasons||[]).map(r =>
    `<span class="reason ${r.startsWith('⚠') ? 'bad' : 'good'}">${r}</span>`
  ).join('');

  const div = document.createElement('div');
  div.className = 'seg-card';
  div.innerHTML = `
    <div class="seg-header">
      <div class="seg-id">${seg.stem}</div>
      <span class="verdict-badge verdict-${vc}">${seg.verdict.label}</span>
      <div class="seg-pills">
        <span class="pill">${maneuver}</span>
        <span class="pill ${n_drones > 1 ? 'hi' : ''}">${n_drones} drone${n_drones>1?'s':''}</span>
        <span class="pill">${lbl.session||'—'}</span>
        <span class="pill">${lbl.split||'—'}</span>
        <span class="pill">az ${az}</span>
        <span class="pill">dist ${dist}</span>
        <span class="pill">ht ${ht}</span>
        ${lbl.radius_m ? `<span class="pill">r=${lbl.radius_m}m</span>` : ''}
      </div>
    </div>
    <div class="metrics-row">
      <div class="metric-cell">
        <div class="metric-val ${rmsCol}">${seg.rms_db} dB</div>
        <div class="metric-lbl">RMS level</div>
        <div class="metric-sub">ch ${seg.rms_per_ch.map(r=>r.toFixed(1)).join(' / ')} dB</div>
      </div>
      <div class="metric-cell">
        <div class="metric-val ${freqCol}">${seg.dom_freq_hz != null ? seg.dom_freq_hz.toFixed(1)+' Hz' : '—'}</div>
        <div class="metric-lbl">Dominant frequency</div>
        <div class="metric-sub">training range 80–360 Hz</div>
      </div>
      <div class="metric-cell">
        <div class="metric-val ${ratioCol}">${seg.energy_ratio.toFixed(4)}</div>
        <div class="metric-lbl">Energy ratio</div>
        <div class="metric-sub">25–350 Hz / total</div>
      </div>
      <div class="metric-cell">
        <div class="metric-val ${harmCol}">${seg.harmonic_score.toFixed(4)}</div>
        <div class="metric-lbl">Harmonic score</div>
        <div class="metric-sub">comb structure 0–1</div>
      </div>
      <div class="metric-cell">
        <div class="metric-val col-purple">${seg.verdict.confidence.toFixed(2)}</div>
        <div class="metric-lbl">Confidence</div>
        <div class="metric-sub">composite 0–1</div>
      </div>
    </div>
    <div class="plots-grid">
      <div class="plot-cell wide" style="height:130px;">
        <div class="plot-title"><span class="dot" style="background:#00d4aa"></span>Waveform — ch0 (E), ch1 (H), ch2 (B)</div>
        <canvas id="wav-${idx}" style="height:90px;"></canvas>
      </div>
      <div class="plot-cell" style="height:130px;">
        <div class="plot-title"><span class="dot" style="background:#7b61ff"></span>GCC-PHAT cross-correlation (TDOA evidence)</div>
        <canvas id="gcc-${idx}" style="height:90px;"></canvas>
      </div>
      <div class="plot-cell full" style="height:200px;">
        <div class="plot-title"><span class="dot" style="background:#ffb800"></span>FFT spectrum — harmonic comb + training BPF range overlay</div>
        <canvas id="fft-${idx}" style="height:160px;"></canvas>
      </div>
      <div class="plot-cell wide" style="height:200px;">
        <div class="plot-title"><span class="dot" style="background:#00d4aa"></span>STFT spectrogram 0–500 Hz (harmonic lines = drone signature)</div>
        <canvas id="spect-${idx}" style="height:160px;"></canvas>
      </div>
      <div class="plot-cell" style="height:200px;">
        <div class="plot-title"><span class="dot" style="background:#ff6b35"></span>BPF energy ratio + harmonic score</div>
        <canvas id="bar-${idx}" style="height:160px;"></canvas>
      </div>
    </div>
    <div class="reasons">${reasonsHtml}</div>
  `;
  return div;
}

// ═══════════════════════════════════════════════════════════
//  Main render
// ═══════════════════════════════════════════════════════════
function render() {
  buildSummary();
  buildFreqChart();

  const container = $('segments');
  SEGMENTS.forEach((seg, i) => {
    const card = buildSegmentCard(seg, i);
    container.appendChild(card);
  });

  // Draw canvases after DOM is built
  requestAnimationFrame(() => {
    SEGMENTS.forEach((seg, i) => {
      const wavC  = $(`wav-${i}`);
      const fftC  = $(`fft-${i}`);
      const spectC= $(`spect-${i}`);
      const gccC  = $(`gcc-${i}`);
      const barC  = $(`bar-${i}`);

      if (wavC)  drawWaveform(wavC,  seg.waveforms);
      if (fftC)  drawSpectrum(fftC,  seg.fft_freqs, seg.fft_mag,
                               seg.dom_freq_hz, seg.harmonics);
      if (spectC) drawSpectrogram(spectC, seg.spectrogram);
      if (gccC)  drawGCC(gccC, seg.gcc_lags_ms, seg.gcc_01, seg.gcc_02,
                          seg.gcc_lag01_ms, seg.gcc_lag02_ms);
      if (barC)  drawEnergyBar(barC, seg.energy_ratio, seg.harmonic_score);
    });
  });
}

render();
</script>
</body>
</html>
"""


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(seg_dirs: List[Path], output_path: Path):
    """Compute all metrics and embed into the HTML template."""
    pairs = collect_segments(seg_dirs)
    if not pairs:
        print("❌  No segments found (looking for *_ch0.wav files)")
        sys.exit(1)

    print(f"\nAnalysing {len(pairs)} segment(s)…\n")
    segments_data = []

    for seg_dir, stem in pairs:
        print(f"  {stem}… ", end="", flush=True)
        data = load_segment(seg_dir, stem)
        if data is None:
            print("⚠ missing files, skipped")
            continue
        segments_data.append(data)
        print(f"dom={data['dom_freq_hz']}Hz  ratio={data['energy_ratio']:.3f}  "
              f"harm={data['harmonic_score']:.3f}  → {data['verdict']['label']}")

    if not segments_data:
        print("❌  No segments could be analysed.")
        sys.exit(1)

    # Embed into HTML
    json_data = json.dumps(segments_data, separators=(',', ':'))
    html = HTML_TEMPLATE.replace('__SEGMENTS_DATA__', json_data)

    output_path.write_text(html, encoding='utf-8')
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✅  Report written → {output_path}  ({size_kb:.0f} KB)")
    print(f"   Open in any browser — no server required.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate acoustic evidence report for extracted Dunakeszi segments"
    )
    ap.add_argument(
        "--input-dir", nargs="+", required=True,
        help="One or more directories (or .zip files) containing extracted segments"
    )
    ap.add_argument(
        "--output", default="drone_evidence_report.html",
        help="Output HTML file (default: drone_evidence_report.html)"
    )
    args = ap.parse_args()

    tmp_dirs = []
    seg_dirs = []

    try:
        for inp in args.input_dir:
            p = Path(inp)
            if p.suffix.lower() == '.zip':
                tmp = Path(tempfile.mkdtemp())
                tmp_dirs.append(tmp)
                print(f"📦 Extracting {p.name} → {tmp}")
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(tmp)
                seg_dirs.append(tmp)
            elif p.is_dir():
                seg_dirs.append(p)
            else:
                print(f"❌  Not a directory or zip: {p}")
                sys.exit(1)

        generate_report(seg_dirs, Path(args.output))

    finally:
        for tmp in tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()