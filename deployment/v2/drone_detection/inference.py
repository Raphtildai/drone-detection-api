# -*- coding: utf-8 -*-
"""
inference.py
────────────────────
All inference functions: detection, localisation, and higher-level analysis.

Public API
──────────
load_detection_model()          load best_detection.pth into a cached singleton
load_localization_model()       load best_localization.pth into a cached singleton
reload_models()                 force-reload both singletons
heuristic_detect()              signal-processing features → drone probability
detect()                        hybrid CNN + heuristic detection
localize()                      single-drone acoustic localisation
run_pipeline()                  load → detect → localize → track
load_3ch()                      load a list of WAV paths into 3-channel arrays
analyse_audio_file()            segment-level analysis with dashboard
analyse_external_audio_robust() sliding-window analysis for long files
comprehensive_pipeline_test()   end-to-end Colab-friendly inference test
"""

import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import scipy.signal
import torch

from .config import Config, config
from .audio_processing import AudioProcessor
from .models import DetectionCNN, LocalizationCNN, make_localization_model
from .utils import (
    azimuth_deg_to_xy,
    classify_detection_score,
    compute_ipd_features,
    safe_prob_average,
    sigmoid,
    wrap_angle_deg,
)

_det_model = None
_loc_model = None

# ── tuneable constants ────────────────────────────────────────────────────────
# CNN probability threshold below which localisation is skipped entirely.
# Lower than the full fused detection gate so more segments get positions.
LOC_CNN_THR: float = 0.30
 
# Degenerate multi-drone geometry: if all returned drone positions are within
# this radius of each other (metres), the result is treated as a single drone
# and the multi-drone map is suppressed.
MULTI_DRONE_MIN_SPREAD_M: float = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _slice_channel(channel: np.ndarray, start: int, seg_n: int) -> np.ndarray:
    """
    Extract a fixed-length segment from a FULL-LENGTH channel array.
    """
    ch = np.asarray(channel, dtype=np.float32)
    end = start + seg_n
    if len(ch) >= end:
        return ch[start:end]
    tail = ch[start:] if start < len(ch) else np.array([], dtype=np.float32)
    return np.pad(tail, (0, seg_n - len(tail)))
 
 
def _is_degenerate_multi(drones: list, threshold_m: float = MULTI_DRONE_MIN_SPREAD_M) -> bool:
    """
    Return True when all reported drone positions are clustered within
    threshold_m of each other (i.e. the multi-drone pipeline found no
    spatial separation between 'drones').
    """
    if len(drones) <= 1:
        return False
    xys = [np.asarray(d["xy_position"], dtype=float) for d in drones]
    for i in range(len(xys)):
        for j in range(i + 1, len(xys)):
            if np.linalg.norm(xys[i] - xys[j]) > threshold_m:
                return False
    return True
 
 
def _build_localisation_panel_data(segments: list) -> dict:
    """
     5  extract per-segment (x, y) positions for plotting.
 
    Returns a dict with keys:
      t_starts  - list of segment start times (s)
      xs        - list of x positions (m), None if no localisation
      ys        - list of y positions (m), None if no localisation
      azimuths  - list of azimuth_deg values, None if no localisation
    """
    t_starts, xs, ys, azimuths = [], [], [], []
    for seg in segments:
        t_starts.append(seg["t_start"])
        loc = seg.get("loc")
        if loc is not None:
            xy = np.asarray(loc["xy_position"], dtype=float)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
            azimuths.append(float(loc["azimuth_deg"]))
        else:
            xs.append(None)
            ys.append(None)
            azimuths.append(None)
    return {"t_starts": t_starts, "xs": xs, "ys": ys, "azimuths": azimuths}
 
# ─── Model loaders ────────────────────────────────────────────────────────────

def load_detection_model(cfg: Optional[Config] = None) -> DetectionCNN:
    """Load the best detection checkpoint into a cached singleton."""
    global _det_model
    if _det_model is not None:
        return _det_model
    cfg  = cfg or config
    ckpt = cfg.DRIVE_MODELS / "best_detection.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No detection checkpoint at {ckpt}. Run train_detection() first."
        )
    dev  = torch.device(cfg.DEVICE)
    data = torch.load(ckpt, map_location=dev)
    m    = DetectionCNN().to(dev)
    m.load_state_dict(data["model_state"])
    m.eval()
    cfg.DETECTION_THRESHOLD = float(
        data.get("best_threshold", cfg.DETECTION_THRESHOLD)
    )
    _det_model = m
    print(
        f"✅ Detection model loaded  (ep={data.get('epoch','?')}, "
        f"f1={data.get('best_val_f1','?')}, thr={cfg.DETECTION_THRESHOLD:.3f})"
    )
    return m


def load_localization_model(cfg: Optional[Config] = None):
    """
    Load the best localisation checkpoint into a cached singleton.

    v17  uses make_localization_model(cfg) instead of
    LocalizationCNN(cfg.N_MELS) so ipd_in_dim (3 or 4) is read from
    cfg.BPF_ENERGY_RATIO_AS_FEATURE and matches the saved checkpoint.
    """
    global _loc_model
    if _loc_model is not None:
        return _loc_model
    cfg  = cfg or config
    ckpt = cfg.DRIVE_MODELS / "best_localization.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No localization checkpoint at {ckpt}. Run train_localization() first."
        )
    dev  = torch.device(cfg.DEVICE)
    data = torch.load(ckpt, map_location=dev)
    # v17  make_localization_model reads ipd_in_dim from config
    m = make_localization_model(cfg).to(dev)
    m.load_state_dict(data["model_state"])
    m.eval()
    _loc_model = m
    print(
        f"✅ Localization model loaded  (ep={data.get('epoch','?')}, "
        f"val_loss={data.get('best_val_loss','?'):.5f})"
    )
    return m


def reload_models(cfg: Optional[Config] = None):
    """Force-reload both model singletons from disk."""
    global _det_model, _loc_model
    _det_model = None
    _loc_model = None
    load_detection_model(cfg)
    load_localization_model(cfg)


# ─── Heuristic detector ───────────────────────────────────────────────────────

def _per_frame_entropy(S: np.ndarray, min_energy_fraction: float = 0.01) -> np.ndarray:
    S            = np.asarray(S, dtype=np.float64) + 1e-10
    frame_energy = S.sum(axis=0)
    energy_gate  = frame_energy.max() * min_energy_fraction
    active       = frame_energy >= energy_gate
    P            = S / frame_energy[np.newaxis, :]
    H            = -np.sum(P * np.log2(P + 1e-12), axis=0)
    H_norm       = (H / math.log2(S.shape[0])).astype(np.float32)
    H_norm[~active] = np.nan
    return H_norm


def _harmonic_comb_score(
    y: np.ndarray, sr: int,
    f0_min: float = 80.0, f0_max: float = 350.0, n_harmonics: int = 6,
) -> float:
    S      = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_mean = S.mean(axis=1)
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=2048)
    best   = 0.0
    for f0 in np.linspace(f0_min, f0_max, 30):
        h_energy = 0.0
        for k in range(1, n_harmonics + 1):
            target = f0 * k
            if target > sr / 2:
                break
            idx = int(np.argmin(np.abs(freqs - target)))
            w   = max(1, int(20 / (freqs[1] - freqs[0])))
            h_energy += float(S_mean[max(0, idx - w) : idx + w + 1].max())
        total = float(S_mean.sum()) + 1e-10
        score = h_energy / (total * n_harmonics) * n_harmonics
        if score > best:
            best = score
    return float(min(best, 1.0))


def _impulsiveness(y: np.ndarray) -> float:
    rms  = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    peak = float(np.max(np.abs(y)) + 1e-8)
    return float(np.clip(peak / rms, 1.0, 10.0))


def heuristic_detect(audio: np.ndarray, cfg: Optional[Config] = None) -> dict:
    cfg = cfg or config
    y   = np.asarray(audio, np.float32)
    if len(y) == 0:
        return {"probability": 0.0, "label": "non_drone", "features": {}}
    rms    = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    rms_db = 20.0 * math.log10(rms + 1e-8)
    try:
        S          = np.abs(librosa.stft(y, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH))
        frame_ent  = _per_frame_entropy(S, min_energy_fraction=0.01)
        voiced_ent = frame_ent[~np.isnan(frame_ent)]
        if len(voiced_ent) == 0:
            return {"probability": 0.0, "label": "non_drone",
                    "features": {"rms_db": rms_db, "veto": "silent_clip"}}
        median_ent = float(np.median(voiced_ent))
        p10_ent    = float(np.percentile(voiced_ent, 10))
        centroid   = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=cfg.SR)))
        rolloff    = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=cfg.SR, roll_percent=0.85)))
        bandwidth  = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=cfg.SR)))
    except Exception:
        centroid = rolloff = bandwidth = 0.0
        median_ent = p10_ent = 0.5
    try:
        f0, _, _ = librosa.pyin(y, fmin=50.0, fmax=500.0, sr=cfg.SR,
                                hop_length=cfg.HOP_LENGTH, fill_na=0.0)
        f0           = np.nan_to_num(f0, nan=0.0).astype(np.float32)
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
    if median_ent < 0.38:
        prob = float(np.clip(median_ent / 0.38 * 0.25, 0.0, 0.25))
        return {"probability": prob, "label": classify_detection_score(prob, cfg),
                "features": {"rms_db": rms_db, "median_frame_entropy": median_ent,
                              "p10_entropy": p10_ent, "crest_factor": cf,
                              "centroid_hz": centroid, "veto": "low_frame_entropy"}}
    if cf > 8.5 and median_ent < 0.52:
        return {"probability": 0.10, "label": classify_detection_score(0.10, cfg),
                "features": {"rms_db": rms_db, "median_frame_entropy": median_ent,
                              "crest_factor": cf, "veto": "high_crest_factor"}}
    energy_score    = float(np.clip((rms_db + 45.0) / 25.0, 0.0, 1.0))
    f0_score        = (1.0  if 80.0  <= f0_med <= 350.0 else
                       0.3  if 50.0  <= f0_med <   80.0 else 0.0)
    voiced_score    = float(np.clip(voiced_ratio / 0.50, 0.0, 1.0))
    stability_score = (1.0 - float(np.clip(f0_std / 80.0, 0.0, 1.0))
                       if f0_med > 0 else 0.2)
    centroid_score  = 1.0 if 120.0  <= centroid  <= 4000.0 else 0.35
    rolloff_score   = 1.0 if 300.0  <= rolloff   <= 8000.0 else 0.40
    bandwidth_score = 1.0 if 150.0  <= bandwidth <= 4000.0 else 0.50
    entropy_score   = float(np.clip((median_ent - 0.35) / 0.35, 0.0, 1.0))
    comb_score      = float(np.clip(comb * 3.0, 0.0, 1.0))
    score = (0.14*energy_score + 0.16*f0_score + 0.10*voiced_score
             + 0.08*stability_score + 0.10*centroid_score + 0.07*rolloff_score
             + 0.07*bandwidth_score + 0.16*entropy_score + 0.12*comb_score)
    prob  = sigmoid(8.0 * (score - 0.50))
    return {"probability": float(prob), "label": classify_detection_score(prob, cfg),
            "features": {"rms_db": rms_db, "median_frame_entropy": median_ent,
                         "p10_entropy": p10_ent, "crest_factor": cf,
                         "comb_score": comb, "centroid_hz": centroid,
                         "rolloff_hz": rolloff, "bandwidth_hz": bandwidth,
                         "voiced_ratio": voiced_ratio, "f0_median_hz": f0_med,
                         "f0_std_hz": f0_std}}


# ─── Detection + localisation ─────────────────────────────────────────────────

_RMS_FLOOR_DB = -45.0


def detect(
    channels: List[np.ndarray],
    cfg: Optional[Config] = None,
    use_hybrid: bool = True,
) -> dict:
    cfg = cfg or config
    ap  = AudioProcessor(cfg)

    y0  = ap.pad_or_truncate(np.asarray(channels[0], dtype=np.float32))

    rms_db = float(20 * math.log10(float(np.sqrt(np.mean(y0 ** 2))) + 1e-8))

    # ── Early RMS veto ─────────────────────────────────────────────
    if rms_db < _RMS_FLOOR_DB:
        return {
            "detected": False,
            "probability": 0.0,
            "label": "non_drone",
            "cnn_probability": 0.0,
            "heuristic_probability": 0.0,
            "heuristic_features": {"veto": "below_rms_floor", "rms_db": rms_db},
        }

    # ── CNN ────────────────────────────────────────────────────────
    m    = load_detection_model(cfg)
    feat = ap.feature_stack(y0)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        cnn_prob = float(torch.softmax(m(x), dim=1)[0, 1].item())

    # ── Heuristic ──────────────────────────────────────────────────
    heur = heuristic_detect(y0, cfg)
    heuristic_prob = float(heur["probability"])
    veto = heur["features"].get("veto")

    # ── Threshold roles (NEW) ──────────────────────────────────────
    TH_HIGH = cfg.DETECTION_THRESHOLD       # e.g. 0.60
    TH_MID  = 0.60                          # agreement threshold
    TH_LOW  = 0.40                          # weak confidence

    TH_CNN_SURE = 0.90                      # CNN alone is trusted above this

    if use_hybrid:
        fused_prob = 0.80 * cnn_prob + 0.20 * heuristic_prob

        # Boost when both reasonably confident
        if cnn_prob > TH_MID and heuristic_prob > TH_MID:
            fused_prob = min(1.0, fused_prob + 0.05)

        # Heuristic veto is skipped when the CNN is already very confident
        # (>= TH_CNN_SURE). The model has seen the spectrogram directly and
        # outweighs hand-crafted features in edge cases like low-entropy hover.
        if veto and cnn_prob < TH_CNN_SURE:
            if cnn_prob < TH_LOW:
                fused_prob *= 0.5   # strong suppression
            elif cnn_prob < TH_MID:
                fused_prob *= 0.75  # moderate suppression
            else:
                fused_prob *= 0.90  # light penalty only

    else:
        fused_prob = cnn_prob

    # ── Final decision ───────────────────────────────────────────────
    # When the CNN is highly confident on its own, use it directly so a low
    # heuristic score cannot pull fused_prob below TH_HIGH.
    if cnn_prob >= TH_CNN_SURE:
        detected = cnn_prob >= TH_HIGH
    else:
        detected = fused_prob >= TH_HIGH

    label = classify_detection_score(fused_prob, cfg)

    # print({
    #     "cnn": cnn_prob,
    #     "heur": heuristic_prob,
    #     "fused_after": fused_prob,
    #     "detected": detected,
    #     "veto": veto
    # })

    return {
        "detected": bool(detected),
        "probability": float(fused_prob),
        "label": label,
        "cnn_probability": float(cnn_prob),
        "heuristic_probability": float(heuristic_prob),
        "heuristic_features": heur["features"],
    }

def _estimate_bpf_hz(y: np.ndarray, sr: int) -> float:
    """Quick spectral peak search 50-700 Hz for BPF estimation."""
    try:
        S     = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
        Sm    = S.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask  = (freqs >= 50) & (freqs <= 700)
        return float(freqs[mask][int(np.argmax(Sm[mask]))])
    except Exception:
        return 200.0


def localize(channels: List[np.ndarray], cfg: Optional[Config] = None) -> dict:
    """
    Single-drone acoustic localisation.

    v17: computes BPF energy ratio as 4th IPD scalar when
    cfg.BPF_ENERGY_RATIO_AS_FEATURE is True, matching training.
    """
    cfg = cfg or config
    ap  = AudioProcessor(cfg)
    m   = load_localization_model(cfg)

    padded = [ap.pad_or_truncate(c) for c in channels]
    mels   = [ap.mel(c) for c in padded]
    mel_t  = torch.tensor(np.stack(mels, axis=0),
                          dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    ipd_raw = compute_ipd_features(padded, cfg)   # (3,)
    if getattr(cfg, "BPF_ENERGY_RATIO_AS_FEATURE", False):
        try:
            bpf_hz = _estimate_bpf_hz(padded[0], cfg.SR)
            ratio  = ap.compute_bpf_energy_ratio(padded[0], bpf_hz)
        except Exception:
            ratio = 0.0
        ipd_raw = np.append(ipd_raw, float(ratio)).astype(np.float32)

    ipd_t = torch.tensor(ipd_raw, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        pred = m(mel_t, ipd_t)[0].cpu().numpy()

    sin_az, cos_az, dist_raw, ht_raw = pred
    az_deg = wrap_angle_deg(float(np.degrees(np.arctan2(sin_az, cos_az))))
    dist_m = float(abs(dist_raw) * cfg.MAX_LOCALIZATION_DIST)
    ht_m   = float(abs(ht_raw)   * cfg.MAX_LOCALIZATION_DIST)
    xy     = azimuth_deg_to_xy(az_deg, dist_m, cfg.ARRAY_CENTER)
    return {"azimuth_deg": az_deg, "distance_m": dist_m,
            "height_m": ht_m, "xy_position": xy}


# ─── Pipeline wrapper ─────────────────────────────────────────────────────────

def load_3ch(paths: List, cfg: Optional[Config] = None) -> List[np.ndarray]:
    """Load a list of 3 WAV paths into padded float32 arrays."""
    ap = AudioProcessor(cfg or config)
    return [ap.pad_or_truncate(ap.load(p)) for p in paths]

def load_3ch_full(paths: List, cfg=None) -> List[np.ndarray]:
    """
    Load each WAV file in full (mono, float32) without any truncation.
 
    Unlike load_3ch(), this does NOT call pad_or_truncate(), so the returned
    arrays have the natural length of each file.  The three arrays may
    therefore have different lengths if the input files have different
    durations — this is intentional and handled by _slice_channel().
 
    Used by comprehensive_pipeline_test() for per-segment slicing.
    run_pipeline() continues to use load_3ch() (truncated, one-segment).
    """
    from .config import config as _config
    from .audio_processing import AudioProcessor
    cfg = cfg or _config
    ap = AudioProcessor(cfg)
    return [ap.load(str(p), mono=True).astype(np.float32) for p in paths]


def run_pipeline(
    wav_paths: List,
    cfg: Optional[Config] = None,
    tracker=None,
    multi_drone: bool = False,
    timestamp: Optional[float] = None,
) -> dict:
    from .multidrone import localize_multi_drone
    cfg = cfg or config
    ts  = timestamp or time.time()
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
        positions = [d["xy_position"] for d in drone_locs]
    else:
        loc = localize(channels, cfg)
        result["drones"] = [loc]
        positions = [loc["xy_position"]]
    if tracker and positions:
        result["tracks"] = tracker.step(positions, ts)
    return result


# ─── Segment-level analysis ───────────────────────────────────────────────────

def _synthesise_3ch(audio, sr, drone_pos, mic_positions):
    from .utils import _fractional_delay
    c = 343.0; src = np.array(drone_pos, dtype=float); n = len(audio)
    dists = np.linalg.norm(mic_positions - src[None, :], axis=1)
    rel   = (dists - dists.min()) / c * sr
    out   = []
    for i in range(len(mic_positions)):
        delayed = _fractional_delay(audio.copy(), rel[i])
        out.append(delayed + (0.003 * np.random.randn(n)).astype(np.float32))
    return out


def analyse_audio_file(
    audio_path: str,
    cfg: Optional[Config] = None,
    drone_pos=None,
    n_segments: int = 10,
    threshold_override: Optional[float] = None,
    show_plot: bool = True,
    use_synthesis: bool = False,
) -> dict:
    from .tracking import KalmanTracker
    from .visualization import _plot_analysis_report
    cfg = cfg or config
    if threshold_override is not None:
        old_thr = cfg.DETECTION_THRESHOLD
        cfg.DETECTION_THRESHOLD = threshold_override
    ap       = AudioProcessor(cfg)
    y_full   = ap.load(audio_path, mono=True)
    total    = len(y_full) / cfg.SR
    seg_n    = int(cfg.TARGET_DURATION * cfg.SR)
    hop      = max(seg_n, int((len(y_full) - seg_n) / max(n_segments - 1, 1)))
    dp       = drone_pos or [1.0, 0.8]
    load_detection_model(cfg)
    try:    load_localization_model(cfg); can_localize = True
    except FileNotFoundError: can_localize = False
    tracker  = KalmanTracker(cfg)
    segments = []
    base_ts  = time.time()
    mode_str = "synthesis" if use_synthesis else "direct mono"
    print(f"\n🎵 {Path(audio_path).name}  ({total:.1f}s)  |  {n_segments} segments  |  {mode_str}")
    for seg_i in range(n_segments):
        start = min(seg_i * hop, max(0, len(y_full) - seg_n))
        audio = y_full[start : start + seg_n]
        if len(audio) < seg_n:
            audio = np.pad(audio, (0, seg_n - len(audio)))
        t_s    = start / cfg.SR
        mel_fr = ap.mel(ap.pad_or_truncate(audio))  # keep for plotting only
        rms_db = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-8))
        if use_synthesis:
            chs = _synthesise_3ch(audio, cfg.SR, dp, cfg.MIC_POSITIONS)
            det = detect(chs, cfg)
            tmp = []
            for ch in chs:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tf.name, ch, cfg.SR); tmp.append(tf.name)
            try:
                res = run_pipeline(tmp, cfg, tracker=tracker, timestamp=base_ts + t_s)
                res["cnn_probability"]       = det.get("cnn_probability", float("nan"))
                res["heuristic_probability"] = det.get("heuristic_probability", float("nan"))
            finally:
                for p in tmp: os.unlink(p)
            drone_loc = res["drones"][0] if res["drones"] else None
        else:
            # analyse_audio_file operates on a single mono file — detection
            # only uses channel 0 so passing [audio]*3 is fine for detection,
            # but localization requires real inter-channel delays and is
            # intentionally disabled here. Use use_synthesis=True or
            # run_pipeline() with 3 real microphone files for valid localization.
            det       = detect([audio, audio, audio], cfg)
            drone_loc = None
            # localization skipped — single-channel input has no spatial ITD
            positions = [drone_loc["xy_position"]] if drone_loc else []
            tracks    = tracker.step(positions, base_ts + t_s)
            res = {"detected": det["detected"], "probability": det["probability"],
                   "cnn_probability": det.get("cnn_probability", float("nan")),
                   "heuristic_probability": det.get("heuristic_probability", float("nan")),
                   "drones": [drone_loc] if drone_loc else [], "tracks": tracks}
        segments.append({"seg": seg_i+1, "t_start": t_s,
                         "detected": res["detected"], "prob": res["probability"],
                         "cnn_probability": res.get("cnn_probability", float("nan")),
                         "heuristic_probability": res.get("heuristic_probability", float("nan")),
                         "xy": res["drones"][0]["xy_position"] if res["drones"] else None,
                         "loc": res["drones"][0] if res["drones"] else None,
                         "mel": mel_fr, "rms_db": rms_db})
        icon = "🚁" if res["detected"] else "🌳"
        print(f"  Seg {seg_i+1:3d}  {icon}  conf={res['probability']:.3f}  rms={rms_db:.1f}dB")
    confirmed = tracker.all_confirmed()
    n_det = sum(s["detected"] for s in segments)
    print(f"\n  📊 {n_det}/{n_segments} detected  |  {len(confirmed)} confirmed track(s)")
    if show_plot:
        _plot_analysis_report(segments, confirmed, cfg, Path(audio_path).name)
        if confirmed:
            from .visualization import plot_track_trajectory
            plot_track_trajectory(confirmed, cfg, save=True)
    if threshold_override is not None:
        cfg.DETECTION_THRESHOLD = old_thr
    return {"segments": segments, "tracker": tracker, "tracks": confirmed,
            "detected": n_det > 0,
            "probability": max((s["prob"] for s in segments), default=0.0),
            "duration_sec": total}


def analyse_external_audio_robust(
    audio_path: str,
    cfg: Optional[Config] = None,
    segment_sec: Optional[float] = None,
    overlap: Optional[float] = None,
    threshold: Optional[float] = None,
    min_pos_segments: Optional[int] = None,
    agg_mode: Optional[str] = None,
    topk: Optional[int] = None,
    show_plot: bool = True,
) -> dict:
    segment_sec      = segment_sec      or (cfg or config).EXTERNAL_SEGMENT_SEC
    overlap          = overlap          if overlap is not None else (cfg or config).EXTERNAL_SEGMENT_OVERLAP
    threshold        = threshold        if threshold is not None else (cfg or config).EXTERNAL_INFER_THRESHOLD
    min_pos_segments = min_pos_segments or (cfg or config).EXTERNAL_MIN_POS_SEGMENTS
    agg_mode         = agg_mode         or (cfg or config).EXTERNAL_AGG_MODE
    topk             = topk             or (cfg or config).EXTERNAL_TOPK
    cfg              = cfg or config
    ap      = AudioProcessor(cfg)
    y       = ap.load(audio_path, mono=True)
    total_s = len(y) / cfg.SR
    seg_len = max(1, int(segment_sec * cfg.SR))
    hop_len = max(1, int(seg_len * (1.0 - overlap)))
    starts  = list(range(0, len(y) - seg_len + 1, hop_len)) if len(y) > seg_len else [0]
    if starts and starts[-1] != len(y) - seg_len:
        starts.append(len(y) - seg_len)
    segment_results = []
    for i, start in enumerate(starts):
        seg = ap.pad_or_truncate(y[start : start + seg_len])
        res = detect([seg, seg, seg], cfg, use_hybrid=True)
        t0  = start / cfg.SR; t1 = (start + seg_len) / cfg.SR
        segment_results.append({"segment_index": i+1, "t_start_s": float(t0),
                                 "t_end_s": float(t1),
                                 "probability": float(res["probability"]),
                                 "cnn_probability": float(res["cnn_probability"]),
                                 "heuristic_probability": float(res["heuristic_probability"]),
                                 "label": res["label"],
                                 "detected_at_main_threshold": bool(res["probability"] >= cfg.DETECTION_THRESHOLD),
                                 "detected_at_external_threshold": bool(res["probability"] >= threshold)})
    probs        = [s["probability"] for s in segment_results]
    probs_sorted = sorted(probs, reverse=True)
    if agg_mode == "max":
        clip_score = float(max(probs)) if probs else 0.0
    elif agg_mode == "vote":
        clip_score = float(sum(p >= threshold for p in probs) / max(len(probs), 1))
    else:
        clip_score = safe_prob_average(probs_sorted[:max(1, topk)], default=0.0)
    pos_count     = sum(s["detected_at_external_threshold"] for s in segment_results)
    clip_detected = clip_score >= threshold or pos_count >= min_pos_segments
    clip_label    = ("drone" if clip_detected and clip_score >= threshold else
                     "possible_drone" if pos_count >= 1 or clip_score >= cfg.DETECTION_THRESHOLD_WEAK
                     else "non_drone")
    print(f"\n🎧 Robust External Audio Analysis")
    print(f"🎵 File: {Path(audio_path).name} | duration={total_s:.2f}s | {len(segment_results)} segments")
    for s in segment_results:
        icon = "🚁" if s["detected_at_external_threshold"] else "🌳"
        print(f"  Seg {s['segment_index']:3d} {icon} prob={s['probability']:.3f}  "
              f"cnn={s['cnn_probability']:.3f}  heur={s['heuristic_probability']:.3f}  "
              f"[{s['t_start_s']:.2f}-{s['t_end_s']:.2f}s]")
    print(f"\n📊 Clip score: {clip_score:.3f}  |  Positive: {pos_count}/{len(segment_results)}  |  Label: {clip_label}")
    if show_plot:
        try:
            from .visualization import _plot_external_detection_scores
            _plot_external_detection_scores(segment_results, threshold, cfg, Path(audio_path).name)
        except Exception:
            pass
    return {"file": str(audio_path), "duration_s": float(total_s),
            "segment_results": segment_results, "clip_score": float(clip_score),
            "positive_segments": int(pos_count), "segments_total": int(len(segment_results)),
            "clip_detected": bool(clip_detected), "clip_label": clip_label,
            "aggregation_mode": agg_mode, "external_threshold": float(threshold)}


# ─── Comprehensive pipeline test (unchanged from v15) ────────────────────────

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _upload_test_audio_triplet_colab(exts: Optional[List[str]] = None) -> List[str]:
    if not _in_colab():
        raise RuntimeError(
            "No wav_paths were provided. Interactive upload is only available in Google Colab."
        )
    from google.colab import files
    allowed_exts = tuple(exts or [".wav", ".mp3", ".flac", ".ogg", ".m4a"])
    print("📁 Please upload exactly 3 audio files for Mic1, Mic2, and Mic3.")
    uploaded = files.upload()
    picked = sorted([name for name in uploaded.keys() if name.lower().endswith(allowed_exts)])
    if len(picked) != 3:
        raise ValueError(f"Expected exactly 3 audio files, but found {len(picked)}: {picked}")
    print("✅ Uploaded files:")
    for i, p in enumerate(picked, start=1):
        print(f"   Mic {i}: {p}")
    return picked


def _validate_test_audio_paths(wav_paths: List) -> List[str]:
    if not isinstance(wav_paths, (list, tuple)):
        raise TypeError("wav_paths must be a list or tuple of 3 file paths.")
    if len(wav_paths) != 3:
        raise ValueError(f"Expected exactly 3 audio paths, got {len(wav_paths)}.")
    norm = []
    for i, p in enumerate(wav_paths, start=1):
        ps = str(p)
        if not Path(ps).exists():
            raise FileNotFoundError(f"Mic {i} file does not exist: {ps}")
        norm.append(ps)
    return norm


def _summarize_loaded_channels(channels: List[np.ndarray]) -> List[dict]:
    summary = []
    for i, ch in enumerate(channels, start=1):
        ch = np.asarray(ch)
        rms    = float(np.sqrt(np.mean(ch ** 2)) + 1e-8) if len(ch) else 0.0
        rms_db = float(20.0 * np.log10(rms + 1e-8))
        summary.append({"channel": i, "shape": tuple(ch.shape), "dtype": str(ch.dtype),
                         "min": float(np.min(ch)) if len(ch) else 0.0,
                         "max": float(np.max(ch)) if len(ch) else 0.0,
                         "rms_db": rms_db})
    return summary


def _print_pipeline_test_summary(wav_paths, channel_summary, single_result,
                                  multi_result=None):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PIPELINE TEST SUMMARY")
    print("=" * 80)
    print("\n🎵 Input files")
    for i, p in enumerate(wav_paths, start=1):
        print(f"  Mic {i}: {p}")
    print("\n🔊 Channel statistics")
    for s in channel_summary:
        print(f"  Ch{s['channel']}: shape={s['shape']}  dtype={s['dtype']}  "
              f"min={s['min']:.5f}  max={s['max']:.5f}  rms={s['rms_db']:.2f} dB")
    print("\n🚁 Single-drone pipeline")
    print(f"  detected               : {single_result['detected']}")
    print(f"  probability            : {single_result['probability']:.4f}")
    print(f"  cnn_probability        : {single_result['cnn_probability']:.4f}")
    print(f"  heuristic_probability  : {single_result['heuristic_probability']:.4f}")
    print(f"  drones_found           : {len(single_result['drones'])}")
    print(f"  tracks_returned        : {len(single_result['tracks'])}")
    if single_result["drones"]:
        for i, d in enumerate(single_result["drones"], start=1):
            xy = d.get("xy_position", None)
            xy_str = (f"({float(xy[0]):.3f}, {float(xy[1]):.3f})"
                      if isinstance(xy, np.ndarray) and len(xy) >= 2 else str(xy))
            print(f"    Drone {i}: az={d.get('azimuth_deg', float('nan')):.2f}°  "
                  f"dist={d.get('distance_m', float('nan')):.2f}m  "
                  f"ht={d.get('height_m', float('nan')):.2f}m  xy={xy_str}")
    if multi_result is not None:
        print("\n🚁🚁 Multi-drone pipeline")
        print(f"  detected               : {multi_result['detected']}")
        print(f"  probability            : {multi_result['probability']:.4f}")
        print(f"  cnn_probability        : {multi_result['cnn_probability']:.4f}")
        print(f"  heuristic_probability  : {multi_result['heuristic_probability']:.4f}")
        print(f"  drones_found           : {len(multi_result['drones'])}")
        print(f"  tracks_returned        : {len(multi_result['tracks'])}")
        for i, d in enumerate(multi_result["drones"], start=1):
            xy = d.get("xy_position", None)
            xy_str = (f"({float(xy[0]):.3f}, {float(xy[1]):.3f})"
                      if isinstance(xy, np.ndarray) and len(xy) >= 2 else str(xy))
            print(f"    Drone {i}: az={d.get('azimuth_deg', float('nan')):.2f}°  "
                  f"dist={d.get('distance_m', float('nan')):.2f}m  xy={xy_str}")


def comprehensive_pipeline_test(
    wav_paths=None,
    cfg=None,
    run_multi_drone: bool = True,
    use_tracker: bool = True,
    timestamp=None,
    auto_upload_if_missing: bool = True,
    show_plots: bool = True,
    save_plots: bool = True,
) -> dict:
    """
    End-to-end pipeline test.
 
    Step-level notes:
       1  _slice_channel >= boundary
       2  localize() gated on LOC_CNN_THR not full hybrid threshold
       3  degenerate multi-drone suppression
       4  file_tracker min_hits=2
       5  localisation panel uses (x, y) not Az/180°
    """
    from .config import config as _config
    from .audio_processing import AudioProcessor
    from .tracking import KalmanTracker
    from .inference import (
        LOC_CNN_THR,
        MULTI_DRONE_MIN_SPREAD_M,
        _is_degenerate_multi,
        _build_localisation_panel_data,
        load_detection_model,
        load_localization_model,
        load_3ch,          # still used by run_pipeline
        detect,
        localize,
        run_pipeline,
        _upload_test_audio_triplet_colab,
        _validate_test_audio_paths,
        _summarize_loaded_channels,
        _print_pipeline_test_summary,
    )
    from .visualization import (
        _plot_analysis_report,
        plot_polar_azimuth,
        plot_multi_drone_positions,
        plot_track_trajectory,
    )
 
    cfg = cfg or _config
 
    if wav_paths is None:
        if not auto_upload_if_missing:
            raise ValueError("wav_paths is None and auto_upload_if_missing=False.")
        wav_paths = _upload_test_audio_triplet_colab()
    wav_paths = _validate_test_audio_paths(wav_paths)
 
    print("🔄 Loading models...")
    load_detection_model(cfg)
    load_localization_model(cfg)
 
    # ──  6  load FULL-LENGTH arrays for per-segment slicing ────────────
    print("\n🔊 Loading 3-channel audio (full length)...")
    full_channels = load_3ch_full(wav_paths, cfg)
 
    # Channel statistics use the full arrays
    channel_summary = _summarize_loaded_channels(full_channels)
    for s in channel_summary:
        print(f"  Ch{s['channel']}: shape={s['shape']}  rms={s['rms_db']:.2f} dB  "
              f"duration={s['shape'][0]/cfg.SR:.1f}s")
 
    # ── Trackers ──────────────────────────────────────────────────────────────
    tracker_single = tracker_multi = None
    if use_tracker:
        tracker_single = KalmanTracker(cfg)
        tracker_multi  = KalmanTracker(cfg) if run_multi_drone else None
 
    per_file_results   = []
    all_detected_azimuths = []
    seg_n = int(cfg.TARGET_DURATION * cfg.SR)
 
    for file_idx, wav_path in enumerate(wav_paths):
        mic_label = f"Mic{file_idx + 1}_{Path(wav_path).stem}"
        print(f"\n📊 Analysing {Path(wav_path).name} (channel {file_idx + 1})...")
 
        ap     = AudioProcessor(cfg)
        y_full = full_channels[file_idx]          #  6  use pre-loaded full array
        total_s = len(y_full) / cfg.SR
        n_segs  = max(1, int(math.ceil(total_s / cfg.TARGET_DURATION)))
 
        try:
            load_localization_model(cfg)
            can_localize = True
        except FileNotFoundError:
            can_localize = False
 
        #  4  min_hits=2
        file_tracker = KalmanTracker(cfg)
        file_tracker.cfg.KF_MIN_HITS = max(2, cfg.KF_MIN_HITS)
 
        base_ts  = time.time()
        segments = []
 
        for seg_i in range(n_segs):
            start = min(seg_i * seg_n, max(0, len(y_full) - seg_n))
 
            #  6  slice from the FULL channel arrays, not the
            # one-segment truncated buffers that load_3ch() would return.
            # ch_slice[i] uses full_channels[i] (the i-th file's full audio).
            # This preserves real inter-channel delays for all segments.
            ch_slice = [
                _slice_channel(full_channels[i], start, seg_n)
                for i in range(len(full_channels))
            ]
 
            audio  = ch_slice[file_idx]   # reference mono = this file's channel
            t_s    = start / cfg.SR
            mel_fr = ap.mel(ap.pad_or_truncate(audio))
            rms_db = float(20 * math.log10(
                math.sqrt(float(np.mean(audio ** 2))) + 1e-8))
 
            det = detect(ch_slice, cfg)
 
            #  2  localize when CNN is moderately confident
            drone_loc = None
            if can_localize and det["cnn_probability"] >= LOC_CNN_THR:
                try:
                    drone_loc = localize(ch_slice, cfg)
                except Exception as exc:
                    print(f"    ⚠  localize() failed at seg {seg_i+1}: {exc}")
                    drone_loc = {
                        "azimuth_deg": 0.0,
                        "distance_m":  0.0,
                        "height_m":    0.0,
                        "xy_position": np.array(cfg.ARRAY_CENTER, dtype=np.float32),
                    }
 
            positions = [drone_loc["xy_position"]] if drone_loc else []
            tracks    = file_tracker.step(positions, base_ts + t_s)
 
            segments.append({
                "seg":                   seg_i + 1,
                "t_start":               t_s,
                "detected":              det["detected"],
                "prob":                  det["probability"],
                "cnn_probability":       det.get("cnn_probability",       float("nan")),
                "heuristic_probability": det.get("heuristic_probability", float("nan")),
                "xy":  drone_loc["xy_position"] if drone_loc else None,
                "loc": drone_loc,
                "mel": mel_fr,
                "rms_db": rms_db,
                "waveform": audio.tolist(),
            })
 
            icon = "🚁" if det["detected"] else "🌳"
            print(f"  Seg {seg_i + 1:3d}  {icon}  "
                  f"conf={det['probability']:.3f}  "
                  f"cnn={det['cnn_probability']:.3f}  "
                  f"rms={rms_db:.1f} dB"
                  + (f"  az={drone_loc['azimuth_deg']:.1f}°" if drone_loc else ""))
 
            if drone_loc is not None:
                all_detected_azimuths.append(drone_loc["azimuth_deg"])
 
        confirmed = file_tracker.all_confirmed()
        n_det     = sum(s["detected"] for s in segments)
        n_loc     = sum(1 for s in segments if s["loc"] is not None)
        print(f"\n  📊 {n_det}/{n_segs} segments detected  |  "
              f"{len(confirmed)} confirmed track(s)  |  "
              f"{n_loc} segments localised")
 
        #  5: attach (x, y) per segment for the visualisation scatter panel
        loc_panel = _build_localisation_panel_data(segments)
        for seg, x, y, az in zip(segments, loc_panel["xs"],
                                  loc_panel["ys"], loc_panel["azimuths"]):
            seg["loc_x"]     = x
            seg["loc_y"]     = y
            seg["loc_az_deg"] = az
 
        per_file_results.append({
            "wav_path":       wav_path,
            "mic_label":      mic_label,
            "segments":       segments,
            "confirmed":      confirmed,
            "n_detected":     n_det,
            "loc_panel_data": loc_panel,
        })
 
        if show_plots:
            print(f"\n🖼️  Generating analysis dashboard for {Path(wav_path).name}...")
            _plot_analysis_report(segments, confirmed, cfg, mic_label)
 
        if show_plots and confirmed:
            print(f"  📍 Plotting Kalman tracks for {Path(wav_path).name}...")
            plot_track_trajectory(confirmed, cfg, save=save_plots)
 
    # ── Full-pipeline runs (run_pipeline uses load_3ch internally — unchanged) ─
    print("\n🚁 Running single-drone pipeline...")
    single_result = run_pipeline(
        wav_paths=wav_paths, cfg=cfg,
        tracker=tracker_single, multi_drone=False,
        timestamp=timestamp,
    )
 
    multi_result = None
    if run_multi_drone:
        print("\n🚁🚁 Running multi-drone pipeline...")
        multi_result = run_pipeline(
            wav_paths=wav_paths, cfg=cfg,
            tracker=tracker_multi, multi_drone=True,
            timestamp=timestamp,
        )
 
        #  3  suppress degenerate multi-drone map
        if multi_result and multi_result.get("drones"):
            if _is_degenerate_multi(multi_result["drones"]):
                print(
                    f"\n  ⚠  Multi-drone result suppressed: all "
                    f"{len(multi_result['drones'])} positions within "
                    f"{MULTI_DRONE_MIN_SPREAD_M} m — likely one drone."
                )
                multi_result["_degenerate"] = True
 
    # ── Cross-file summary plots ──────────────────────────────────────────────
    if show_plots:
        if all_detected_azimuths:
            print(f"\n🧭 Polar azimuth summary ({len(all_detected_azimuths)} detections)...")
            plot_polar_azimuth(
                all_detected_azimuths,
                title="Detected Azimuths — All Mic Channels",
                cfg=cfg, save=save_plots,
            )
 
        if (multi_result
                and multi_result.get("drones")
                and not multi_result.get("_degenerate")):
            print("\n🗺️  Multi-drone position map...")
            plot_multi_drone_positions(multi_result["drones"], cfg=cfg, save=save_plots)
        elif single_result and single_result.get("drones"):
            print("\n🗺️  Single-drone position map...")
            plot_multi_drone_positions(single_result["drones"], cfg=cfg, save=save_plots)
 
        if tracker_single is not None:
            all_single_tracks = tracker_single.all_confirmed()
            if all_single_tracks:
                print("\n📍 Single-drone pipeline Kalman trajectories...")
                plot_track_trajectory(all_single_tracks, cfg=cfg, save=save_plots)

    # ── Thesis figures ────────────────────────────────────────────────
    if show_plots and len(per_file_results) == 3:
        print("\n📐 Per-channel enhanced dashboard (thesis)...")
        from .visualization import (
            plot_per_channel_enhanced,
            plot_combined_3ch_analysis,
        )
        plot_per_channel_enhanced(
            per_file_results=per_file_results,
            cfg=cfg,
            save=save_plots,
        )
        print("\n🔬 Combined 3-channel analysis (thesis)...")
        plot_combined_3ch_analysis(
            full_channels    = full_channels,
            per_file_results = per_file_results,
            single_result    = single_result,
            cfg              = cfg,
            save             = save_plots,
        )
        
    _print_pipeline_test_summary(wav_paths, channel_summary, single_result, multi_result)
 
    return {
        "wav_paths":        wav_paths,
        "channels_loaded":  channel_summary,
        "single_result":    single_result,
        "multi_result":     multi_result,
        "per_file_results": per_file_results,
        "all_azimuths":     all_detected_azimuths,
    }

# If main analyze external audio file with robust method by default, but allow full pipeline test with --run_multi_drone and --use_tracker flags.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run comprehensive pipeline test or robust external audio analysis.")
    parser.add_argument("--wav_paths", nargs=3, help="Paths to 3 audio files for Mic1, Mic2, and Mic3.")
    parser.add_argument("--run_multi_drone", action="store_true", help="Whether to run the multi-drone pipeline.")
    parser.add_argument("--use_tracker", action="store_true", help="Whether to use Kalman tracking in the pipelines.")
    parser.add_argument("--show_plots", action="store_true", help="Whether to display analysis plots.")
    args = parser.parse_args()
    if args.run_multi_drone or args.use_tracker:
        comprehensive_pipeline_test(
            wav_paths=args.wav_paths,
            run_multi_drone=args.run_multi_drone,
            use_tracker=args.use_tracker,
            show_plots=args.show_plots,
        )
    else:
        # If not running the full pipeline test, run the robust external audio analysis on the first provided file or example file.
        if args.wav_paths is None:
            example_dir = Path(__file__).parent / "example_data"
            example_files = sorted(example_dir.glob("*.wav"))
            if not example_files:
                raise FileNotFoundError(f"No example WAV files found in {example_dir}. Please provide --wav_paths.")
            audio_path = str(example_files[0])
            print(f"No wav_paths provided. Using example file: {audio_path}")
        else:
            audio_path = args.wav_paths[0]
        analyse_external_audio_robust(
            audio_path=audio_path,
            cfg=None,
            segment_sec=None,
            overlap=None,
            threshold=None,
            min_pos_segments=None,
            agg_mode=None,
            topk=None,
            show_plot=args.show_plots,
        )
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Comprehensive pipeline test for drone detection and localization.")
#     parser.add_argument("--wav_paths", nargs=3, help="Paths to 3 audio files for Mic1, Mic2, and Mic3.")
#     parser.add_argument("--run_multi_drone", action="store_true", help="Whether to run the multi-drone pipeline.")
#     parser.add_argument("--use_tracker", action="store_true", help="Whether to use Kalman tracking in the pipelines.")
#     parser.add_argument("--show_plots", action="store_true", help="Whether to display analysis plots.")
#     args = parser.parse_args()

#     if args.wav_paths is None:
#         if _in_colab():
#             print("No wav_paths provided. Launching interactive upload in Colab...")
#             wav_paths = _upload_test_audio_triplet_colab()
#         else:
#             print("No wav_paths provided. Using example files from the 'example_data' directory...")
#             example_dir = Path(__file__).parent / "example_data"
#             wav_paths = sorted(example_dir.glob("*.wav"))[:3]
#             if len(wav_paths) < 3:
#                 raise FileNotFoundError(f"Expected at least 3 WAV files in {example_dir}, but found {len(wav_paths)}.")
#             wav_paths = [str(p) for p in wav_paths]
#             print(f"Using example files: {wav_paths}")
#     else:
#         wav_paths = _validate_test_audio_paths(args.wav_paths)

#     comprehensive_pipeline_test(
#         wav_paths=wav_paths,
#         run_multi_drone=args.run_multi_drone,
#         use_tracker=args.use_tracker,
#         show_plots=args.show_plots,
#     )