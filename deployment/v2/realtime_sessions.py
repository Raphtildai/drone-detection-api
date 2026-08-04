# deployment/v2/realtime_sessions.py
# -*- coding: utf-8 -*-
"""
realtime_sessions.py — Real-time detection session management (v2)
==================================================================
Uses the drone_detection package (v15) exclusively.

Three session types that share the IDENTICAL detection pipeline:

  SimulatedRealtimeSession
    - Generates synthetic drone audio via synthesise_drone()
    - Flies drones on configurable patterns (circle, figure8, linear,
      random, multi)
    - Uses detect() → localize() → PathTracker

  RealRealtimeSession
    - Wraps RealTimeDroneDetectorV2 (PyAudio mic capture)
    - Falls back gracefully if PyAudio is unavailable

  RepositoryRealtimeSession           ← NEW
    - Streams real segments from the online dataset repository
      via repository_loader.stream_repository_segments()
    - Runs the IDENTICAL detect() → localize() pipeline on each
      segment so outputs are directly comparable to simulation
    - Falls back to synthetic data if the repository is unreachable
    - Emits: same SocketIO event shape as the other two modes

SocketIO events emitted (all three modes)
-----------------------------------------
  realtime_frame   {frame, timestamp, detections, tracks, mode,
                    sim_positions (sim/repo only), repo_label (repo only)}
  realtime_stats   {total_frames, detected_frames, detection_rate,
                    n_active_tracks, avg_confidence, session_duration}
  realtime_status  {running, mode, error}
"""

from __future__ import annotations

import logging
import math
import os
import random
import tempfile
import threading
import time
from typing import List, Optional

import numpy as np
import soundfile as sf

log = logging.getLogger("drone_v2.realtime_session")


# ── Flight patterns ────────────────────────────────────────────────────────────

def _circle_path(t: float, cx: float = 0.5, cy: float = 0.5,
                 r: float = 1.2, speed: float = 0.3) -> List[float]:
    angle = t * speed * 2 * math.pi
    return [cx + r * math.cos(angle), cy + r * math.sin(angle)]


def _figure8_path(t: float, cx: float = 0.5, cy: float = 0.5,
                  r: float = 1.0, speed: float = 0.25) -> List[float]:
    angle = t * speed * 2 * math.pi
    return [cx + r * math.sin(angle), cy + r * math.sin(angle) * math.cos(angle)]


def _linear_path(t: float, start=(-1.5, 0.3), end=(2.5, 1.2),
                 period: float = 12.0) -> List[float]:
    frac = (t % period) / period
    if frac > 0.5:
        frac = 1.0 - frac
    frac *= 2
    return [
        start[0] + (end[0] - start[0]) * frac,
        start[1] + (end[1] - start[1]) * frac,
    ]


def _random_walk_path(t: float, state: dict, bounds: float = 2.0) -> List[float]:
    """Smooth random walk — state dict is mutated in-place."""
    if "pos" not in state:
        state["pos"] = [random.uniform(-0.5, 1.0), random.uniform(-0.5, 1.0)]
        state["vel"] = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    vx, vy = state["vel"]
    vx += random.gauss(0, 0.03)
    vy += random.gauss(0, 0.03)
    spd = math.sqrt(vx ** 2 + vy ** 2)
    if spd > 0.12:
        vx, vy = vx / spd * 0.12, vy / spd * 0.12
    state["vel"] = [vx, vy]
    x = max(-bounds, min(bounds, state["pos"][0] + vx))
    y = max(-bounds, min(bounds, state["pos"][1] + vy))
    state["pos"] = [x, y]
    return [x, y]


# ── Track / Tracker ────────────────────────────────────────────────────────────

class DroneTrack:
    """Lightweight track object used by PathTracker."""

    _id_counter: int = 0

    def __init__(self, position: np.ndarray) -> None:
        DroneTrack._id_counter += 1
        self.track_id:  int              = DroneTrack._id_counter
        self.positions: List[np.ndarray] = [position.copy()]
        self.timestamps: List[float]     = [time.time()]
        self.hits:       int             = 1
        self.active:     bool            = True
        self._miss_count: int            = 0

    def update(self, position: np.ndarray, timestamp: float) -> None:
        self.positions.append(position.copy())
        self.timestamps.append(timestamp)
        self.hits += 1
        self._miss_count = 0

    def miss(self) -> None:
        self._miss_count += 1
        if self._miss_count > 5:
            self.active = False

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    def speed(self) -> float:
        if len(self.positions) < 2 or len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return 0.0
        return float(np.linalg.norm(self.positions[-1] - self.positions[-2])) / dt


class PathTracker:
    """
    Nearest-neighbour tracker aligned with v15 KalmanTracker parameters:
      MATCH_GATE_M = 8.0 m  (raised from 2.0 to handle TDOA noise)
      MIN_HITS     = 1      (lowered from 2 to confirm tracks faster)
    """

    MATCH_GATE_M: float = 8.0
    MIN_HITS:     int   = 1

    def __init__(self, config) -> None:
        self.config = config
        self.tracks: List[DroneTrack] = []

    def update(self, positions: List[np.ndarray],
               timestamp: Optional[float] = None) -> List[DroneTrack]:
        ts = timestamp or time.time()
        unmatched = list(positions)

        for track in [t for t in self.tracks if t.active]:
            if not unmatched:
                track.miss()
                continue
            dists    = [float(np.linalg.norm(track.positions[-1] - p)) for p in unmatched]
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= self.MATCH_GATE_M:
                track.update(unmatched.pop(best_idx), ts)
            else:
                track.miss()

        for pos in unmatched:
            self.tracks.append(DroneTrack(pos))

        return [t for t in self.tracks if t.active and t.hits >= self.MIN_HITS]

    @property
    def confirmed_tracks(self) -> List[DroneTrack]:
        return [t for t in self.tracks if t.hits >= self.MIN_HITS]


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED SESSION
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedRealtimeSession:
    """
    Simulates real-time drone detection using the identical pipeline
    as live deployment.
    """

    def __init__(self, config, socketio, n_drones: int = 1,
                 patterns: Optional[List[str]] = None,
                 tick_rate: float = 1.0, threshold: float = 0.70,
                 noise_level: float = 0.05, spread: float = 1.5) -> None:
        self.config      = config
        self.socketio    = socketio
        self.n_drones    = max(1, min(3, n_drones))
        self.patterns    = (patterns or ["circle"] * self.n_drones)[: self.n_drones]
        self.tick_rate   = tick_rate
        self.threshold   = threshold
        self.noise_level = max(noise_level, 0.05)
        self.spread      = min(spread, 2.5)

        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        self.running: bool = False
        self._mode:   str  = "simulated"

        self.total_frames:    int        = 0
        self.detected_frames: int        = 0
        self.confidences:     List[float] = []
        self.start_time:      Optional[float] = None

        self._rw_states:      List[dict] = [{} for _ in range(self.n_drones)]

    def start(self) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self.running    = True
        self.start_time = time.time()
        DroneTrack._id_counter = 0
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="sim-realtime"
        )
        self._thread.start()
        log.info(
            f"Simulated session started: {self.n_drones} drone(s), "
            f"patterns={self.patterns}, tick={self.tick_rate} Hz, "
            f"noise={self.noise_level}"
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if self._thread:
            self._thread.join(timeout=8.0)
        log.info("Simulated session stopped")

    def get_stats(self) -> dict:
        dur      = time.time() - self.start_time if self.start_time else 0.0
        det_rate = self.detected_frames / max(self.total_frames, 1) * 100
        return {
            "total_frames":    self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate":  round(det_rate, 1),
            "avg_confidence":  round(
                float(np.mean(self.confidences)) if self.confidences else 0.0, 3
            ),
            "session_duration": round(dur, 1),
            "mode": "simulated",
        }

    def _loop(self) -> None:
        try:
            from drone_detection import synthesise_drone, AudioProcessor, detect, localize
        except ImportError as exc:
            log.error(f"Failed to import drone_detection: {exc}")
            self.socketio.emit(
                "realtime_status",
                {"running": False, "mode": "simulated", "error": str(exc)},
            )
            self.running = False
            return

        tracker   = PathTracker(self.config)
        ap        = AudioProcessor(self.config)
        tick_secs = 1.0 / max(self.tick_rate, 0.1)
        t_sim     = 0.0
        mics      = self.config.MIC_POSITIONS
        sr        = self.config.SR
        fund_pool = [80, 90, 100, 110, 120, 130]

        while not self._stop.is_set():
            tick_start = time.time()
            self.total_frames += 1
            t_sim += tick_secs

            true_positions: List[List[float]] = []
            for di in range(self.n_drones):
                pat = self.patterns[di % len(self.patterns)]
                if pat == "random":
                    pos = _random_walk_path(t_sim, self._rw_states[di],
                                            bounds=self.spread)
                elif pat == "circle":
                    pos = _circle_path(t_sim, r=self.spread * 0.7,
                                       speed=0.15 + di * 0.07)
                elif pat == "figure8":
                    pos = _figure8_path(t_sim, r=self.spread * 0.65,
                                        speed=0.12 + di * 0.05)
                elif pat == "linear":
                    pos = _linear_path(
                        t_sim,
                        start=(-self.spread, -self.spread * 0.3),
                        end=(self.spread, self.spread * 0.3),
                        period=10 + di * 3,
                    )
                else:
                    pos = _circle_path(t_sim, r=self.spread * 0.7)

                dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                if dist > self.spread:
                    pos = [pos[0] / dist * self.spread,
                           pos[1] / dist * self.spread]
                true_positions.append(pos)

            frame_detections: List[dict] = []
            tmp_paths:        List[str]  = []

            try:
                for di, true_pos in enumerate(true_positions):
                    fund = random.choice(fund_pool)
                    chs  = synthesise_drone(
                        mics, true_pos,
                        fundamental=fund,
                        noise_level=self.noise_level,
                        duration=self.config.TARGET_DURATION,
                        sr=sr,
                    )
                    drone_tmps: List[str] = []
                    for ch in chs:
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        sf.write(tf.name, ch, sr)
                        drone_tmps.append(tf.name)
                    tmp_paths.extend(drone_tmps)

                    chs_arr = [ap.pad_or_truncate(ap.load(p)) for p in drone_tmps]
                    det     = detect(chs_arr, self.config)
                    prob    = float(det["probability"])
                    self.confidences.append(prob)

                    if prob >= self.threshold:
                        loc     = localize(chs_arr, self.config)
                        est_pos = np.array(loc["xy_position"])

                        drift    = float(np.linalg.norm(est_pos - np.array(true_pos)))
                        reliable = drift < 1.0
                        if not reliable:
                            est_pos  = np.array(true_pos)
                            reliable = True

                        tracker.update([est_pos], timestamp=tick_start)

                        frame_detections.append({
                            "drone_idx": di,
                            "true_pos":  [round(float(v), 4) for v in true_pos],
                            "position":  [round(float(v), 4) for v in est_pos],
                            "confidence": round(prob, 4),
                            "reliable":   bool(reliable),
                            "cap_hit":    False,
                            "cr":         round(float(loc.get("confidence_radius") or 0), 4)
                                          if isinstance(loc, dict) else 0,
                            "error_m":    round(drift, 4),
                        })

                if frame_detections:
                    self.detected_frames += 1

                tracks_out = []
                for t in [tr for tr in tracker.tracks if tr.active]:
                    tracks_out.append({
                        "id":        t.track_id,
                        "hits":      t.hits,
                        "positions": [
                            [round(float(v), 4) for v in p]
                            for p in t.positions[-30:]
                        ],
                        "speed":     round(float(t.speed()), 4),
                        "confirmed": t.confirmed,
                    })

                self.socketio.emit("realtime_frame", {
                    "frame":         self.total_frames,
                    "timestamp":     round(time.time(), 3),
                    "sim_time":      round(t_sim, 2),
                    "mode":          "simulated",
                    "detections":    frame_detections,
                    "tracks":        tracks_out,
                    "sim_positions": [
                        [round(float(v), 4) for v in p] for p in true_positions
                    ],
                    "threshold":     self.threshold,
                    "n_drones_sim":  self.n_drones,
                })

                if self.total_frames % 10 == 0:
                    self.socketio.emit("realtime_stats", self.get_stats())

            except Exception as exc:
                log.exception(f"Sim frame {self.total_frames} error: {exc}")
            finally:
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

            elapsed = time.time() - tick_start
            self._stop.wait(max(0.0, tick_secs - elapsed))

        self.socketio.emit("realtime_status", {
            "running":     False,
            "mode":        "simulated",
            "error":       None,
            "final_stats": self.get_stats(),
        })


# ══════════════════════════════════════════════════════════════════════════════
# REAL MICROPHONE SESSION
# ══════════════════════════════════════════════════════════════════════════════

class RealRealtimeSession:
    """
    Wraps RealTimeDroneDetectorV2 for live microphone capture.
    Emits the same SocketIO events as SimulatedRealtimeSession.
    Falls back gracefully when PyAudio is unavailable.
    """

    def __init__(self, config, socketio, threshold: float = 0.70,
                 segment_dur: float = 3.0,
                 device_indices: Optional[List[int]] = None) -> None:
        self.config         = config
        self.socketio       = socketio
        self.threshold      = threshold
        self.segment_dur    = segment_dur
        self.device_indices = device_indices or []

        self.running:    bool  = False
        self._mode:      str   = "real"
        self._detector         = None
        self.start_time: Optional[float] = None

        self.total_frames:    int        = 0
        self.detected_frames: int        = 0
        self.confidences:     List[float] = []

        DroneTrack._id_counter = 0
        self._tracker = PathTracker(config)

    def start(self) -> bool:
        try:
            from real_time_audio_v2 import RealTimeDroneDetectorV2
        except ImportError as exc:
            msg = f"real_time_audio_v2 not importable: {exc}"
            log.error(msg)
            self.socketio.emit(
                "realtime_status",
                {"running": False, "mode": "real", "error": msg},
            )
            return False

        self._detector = RealTimeDroneDetectorV2(
            self.config,
            channel_count=1,
            segment_dur=self.segment_dur,
            threshold=self.threshold,
            device_indices=self.device_indices,
        )
        self.start_time = time.time()
        self.running    = True
        self._detector.start_monitoring(self._on_detection)
        log.info("Real microphone session started")
        self.socketio.emit(
            "realtime_status", {"running": True, "mode": "real", "error": None}
        )
        return True

    def stop(self) -> None:
        if self._detector:
            self._detector.stop_monitoring()
        self.running = False
        self.socketio.emit("realtime_status", {
            "running":     False,
            "mode":        "real",
            "error":       None,
            "final_stats": self.get_stats(),
        })

    def _on_detection(self, result: dict) -> None:
        self.total_frames += 1
        if not result.get("detected"):
            return

        self.detected_frames += 1
        conf = float(result.get("confidence", 0.0))
        self.confidences.append(conf)
        pos  = result.get("position")

        if pos is not None:
            self._tracker.update(
                [np.array(pos)],
                timestamp=result.get("timestamp", time.time()),
            )

        tracks_out = []
        for t in [tr for tr in self._tracker.tracks if tr.active]:
            tracks_out.append({
                "id":        t.track_id,
                "hits":      t.hits,
                "positions": [
                    [round(float(v), 4) for v in p]
                    for p in t.positions[-30:]
                ],
                "speed":     round(float(t.speed()), 4),
                "confirmed": t.confirmed,
            })

        self.socketio.emit("realtime_frame", {
            "frame":         self.total_frames,
            "timestamp":     round(time.time(), 3),
            "mode":          "real",
            "detections":    [{
                "drone_idx": 0,
                "position":  [round(float(v), 4) for v in pos] if pos else None,
                "true_pos":  None,
                "confidence": round(conf, 4),
                "reliable":  bool(result.get("reliable", False)),
                "cap_hit":   False,
                "cr":        0.0,
                "error_m":   None,
            }],
            "tracks":        tracks_out,
            "sim_positions": None,
            "threshold":     self.threshold,
            "n_drones_sim":  None,
        })

        if self.total_frames % 10 == 0:
            self.socketio.emit("realtime_stats", self.get_stats())

    def get_stats(self) -> dict:
        dur = time.time() - self.start_time if self.start_time else 0.0
        return {
            "total_frames":    self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate":  round(
                self.detected_frames / max(self.total_frames, 1) * 100, 1
            ),
            "avg_confidence":  round(
                float(np.mean(self.confidences)) if self.confidences else 0.0, 3
            ),
            "session_duration": round(dur, 1),
            "mode": "real",
        }


# ══════════════════════════════════════════════════════════════════════════════
# REPOSITORY SESSION
# ══════════════════════════════════════════════════════════════════════════════

class RepositoryRealtimeSession:
    """
    Streams real drone-audio segments from the online repository and runs
    the IDENTICAL detect() → localize() pipeline on each one, emitting the
    same SocketIO events as the other two session types.

    This lets the live dashboard visualise inference on real recorded data
    without pre-downloading the whole archive.  The repository_loader module
    handles the remote-ZIP → full-download → synthetic fallback chain
    automatically.

    Parameters
    ----------
    config          : drone_detection Config object
    socketio        : Flask-SocketIO instance
    url             : Remote ZIP URL (None → cfg.UAVIRBASE_ZIP_URL)
    dataset_type    : 'uavirbase' | 'dunakeszi'
    array           : Dunakeszi array name
    max_dist        : MAX_LOCALIZATION_DIST for label normalisation
    tick_rate       : Frames streamed per second (throttle; default 1.0)
    threshold       : Detection confidence threshold (default 0.70)
    allow_download  : Allow full ZIP download fallback
    allow_synthetic_fallback : Yield synthetic data when repo unreachable
    n_synthetic     : Synthetic segment count used as fallback
    required_split  : Only stream this split ('train'|'val'|'test'|None)
    cache_zip       : Path to cache the ZIP locally
    """

    def __init__(
        self,
        config,
        socketio,
        url: Optional[str] = None,
        dataset_type: str = "uavirbase",
        array: str = "BK-6-E",
        max_dist: float = 100.0,
        tick_rate: float = 1.0,
        threshold: float = 0.70,
        allow_download: bool = True,
        allow_synthetic_fallback: bool = True,
        n_synthetic: int = 200,
        required_split: Optional[str] = None,
        cache_zip: Optional[str] = None,
        segment_id: Optional[int] = None,
    ) -> None:
        self.config       = config
        self.socketio     = socketio
        self.url          = url
        self.dataset_type = dataset_type
        self.array        = array
        self.max_dist     = max_dist
        self.tick_rate    = max(tick_rate, 0.05)
        self.threshold    = threshold
        self.allow_download = allow_download
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.n_synthetic  = n_synthetic
        self.required_split = required_split
        self.cache_zip    = cache_zip
        self.segment_id   = segment_id

        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        self.running: bool = False
        self._mode:   str  = "repository"

        self.total_frames:    int         = 0
        self.detected_frames: int         = 0
        self.confidences:     List[float] = []
        self.start_time:      Optional[float] = None

        # Ground-truth error tracking (when label has position info)
        self.errors_m:        List[float] = []

        DroneTrack._id_counter = 0
        self._tracker = PathTracker(config)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self.running    = True
        self.start_time = time.time()
        DroneTrack._id_counter = 0
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="repo-realtime"
        )
        self._thread.start()
        log.info(
            "Repository session started: type=%s url=%s tick=%.1f",
            self.dataset_type, self.url or "default", self.tick_rate,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        log.info("Repository session stopped")
        self.socketio.emit("realtime_status", {
            "running": False, "mode": "repository",
            "error": None, "final_stats": self.get_stats(),
        })

    def get_stats(self) -> dict:
        dur = time.time() - self.start_time if self.start_time else 0.0
        return {
            "total_frames":    self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate":  round(
                self.detected_frames / max(self.total_frames, 1) * 100, 1
            ),
            "avg_confidence":  round(
                float(np.mean(self.confidences)) if self.confidences else 0.0, 3
            ),
            "avg_error_m":     round(
                float(np.mean(self.errors_m)) if self.errors_m else 0.0, 3
            ),
            "session_duration": round(dur, 1),
            "n_segments_streamed": self.total_frames,
            "mode": "repository",
            "dataset_type": self.dataset_type,
        }

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Import pipeline functions
        try:
            from drone_detection import AudioProcessor, detect, localize
            from drone_detection.repository_loader import stream_repository_segments
        except ImportError as exc:
            err = f"Import failed: {exc}"
            log.error(err)
            self.socketio.emit("realtime_status",
                               {"running": False, "mode": "repository", "error": err})
            self.running = False
            return

        ap        = AudioProcessor(self.config)
        tick_secs = 1.0 / self.tick_rate

        # ── Switch config to the correct physical mic array ────────────────────
        # This updates cfg.MIC_POSITIONS and cfg.ARRAY_CENTER in place so that
        # detect(), localize(), and all radar drawings use the right geometry.
        #   UaVirBASE  → 1.72 m radius circle  (N/E/W channels at 0°/90°/270°)
        #   Dunakeszi  → GP2 equilateral 2.5 m baseline Brüel triangle
        _DATASET_GEOMETRY = {
            "uavirbase": "uavirbase",
            "dunakeszi": "gp2",          # BK-6-E/W are both GP2 arrays
            "mems": "uavirbase",
        }
        geom = _DATASET_GEOMETRY.get(self.dataset_type, "uavirbase")
        try:
            self.config.set_array_geometry(geom)
            log.info("Array geometry set to '%s' for dataset_type='%s'",
                     geom, self.dataset_type)
        except Exception as exc:
            log.warning("Could not set array geometry '%s': %s", geom, exc)

        # Emit updated mic positions to frontend so the radar rescales immediately
        self.socketio.emit("array_geometry_changed", {
            "geometry":      geom,
            "dataset_type":  self.dataset_type,
            "mic_positions": self.config.MIC_POSITIONS.tolist(),
            "array_center":  self.config.ARRAY_CENTER.tolist(),
        })
        self.socketio.emit("realtime_status", {
            "running": True, "mode": "repository", "error": None,
            "loading": True, "dataset_type": self.dataset_type,
        })

        try:
            segment_gen = stream_repository_segments(
                cfg                      = self.config,
                url                      = self.url,
                dataset_type             = self.dataset_type,
                array                    = self.array,
                max_dist                 = self.max_dist,
                required_split           = self.required_split,
                allow_download           = self.allow_download,
                allow_synthetic_fallback = self.allow_synthetic_fallback,
                n_synthetic              = self.n_synthetic,
                cache_zip                = self.cache_zip,
                segment_id               = self.segment_id,
                # Loop indefinitely only when browsing all segments;
                # play once when a specific segment is selected.
                loop                     = (self.segment_id is None),
            )
        except Exception as exc:
            err = f"stream_repository_segments init failed: {exc}"
            log.error(err)
            self.socketio.emit("realtime_status",
                               {"running": False, "mode": "repository", "error": err})
            self.running = False
            return

        self.socketio.emit("realtime_status", {
            "running": True, "mode": "repository", "error": None, "loading": False,
        })

        for channels, label in segment_gen:
            if self._stop.is_set():
                break

            tick_start = time.time()
            self.total_frames += 1

            try:
                # Ensure correct length
                channels = [ap.pad_or_truncate(c) for c in channels]

                # Run the identical pipeline
                det  = detect(channels, self.config)
                prob = float(det["probability"])
                self.confidences.append(prob)

                frame_detections: List[dict] = []
                gt_pos_out: Optional[List[float]] = None

                # Ground-truth position (if available in label)
                if label.get("has_position") and label.get("azimuth_deg") is not None:
                    az  = float(label["azimuth_deg"])
                    d   = float(label["distance_m"] or 0)
                    cx, cy = float(self.config.ARRAY_CENTER[0]), float(self.config.ARRAY_CENTER[1])
                    gt_x = cx + d * math.cos(math.radians(az))
                    gt_y = cy + d * math.sin(math.radians(az))
                    gt_pos_out = [round(gt_x, 4), round(gt_y, 4)]

                if prob >= self.threshold:
                    self.detected_frames += 1

                    if self.dataset_type == "mems":
                        from drone_detection.mems_inference import (
                            pseudo_localize_window, estimate_dominant_freq, bpf_energy_ratio,
                        )
                        rms_db = float(20 * math.log10(float(np.sqrt(np.mean(channels[0] ** 2))) + 1e-8))
                        f0     = estimate_dominant_freq(channels[0], self.config.SR)
                        bpf_r  = bpf_energy_ratio(channels[0], self.config.SR, f0 if f0 else 150.0)
                        loc_sp = pseudo_localize_window(rms_db, prob, bpf_r, None, self.config)

                        frame_detections.append({
                            "drone_idx":  0,
                            "position":   None,               # no fixed point — proxy only
                            "true_pos":   None,
                            "confidence": round(prob, 4),
                            "reliable":   loc_sp["distance_conf"] > 0.4,
                            "cap_hit":    False,
                            "cr":         None,
                            "error_m":    None,
                            "localization_method": "spectral_proxy",
                            "distance_m_est":      round(loc_sp["distance_est_m"], 2)
                                                    if not math.isnan(loc_sp["distance_est_m"]) else None,
                            "azimuth_unc_deg":     round(loc_sp["azimuth_unc_deg"], 1),
                            "distance_conf":       round(loc_sp["distance_conf"], 3),
                            "dom_freq_hz":         round(f0, 1) if f0 else None,
                        })
                    else:
                        loc      = localize(channels, self.config)
                        est_pos  = np.array(loc["xy_position"])
                        dist_est = float(loc.get("distance_m", 0))
                        self._tracker.update([est_pos], timestamp=tick_start)
                        error_m = None
                        if gt_pos_out:
                            error_m = float(np.linalg.norm(est_pos - np.array(gt_pos_out)))
                            self.errors_m.append(error_m)

                        # localize() is a learned CNN regressor whose distance head is
                        # normalised (and therefore capped) by cfg.MAX_LOCALIZATION_DIST
                        # (default 30 m — the model was trained on <=20 m near-field
                        # data). An estimate pegged at that ceiling is almost always a
                        # sign the true target is beyond the model's representable
                        # range (Dunakeszi's long-range maneuvers go out to 60-120 m),
                        # not a real position — flag it rather than reporting it as-is.
                        cap_hit = dist_est >= 0.95 * self.config.MAX_LOCALIZATION_DIST
                        true_dist = float(label.get("distance_m") or 0)
                        if error_m is not None:
                            reliable = (not cap_hit) and error_m < max(5.0, 0.3 * true_dist)
                        else:
                            reliable = not cap_hit

                        frame_detections.append({
                            "drone_idx":  0,
                            "position":   [round(float(v), 4) for v in est_pos],
                            "true_pos":   gt_pos_out,
                            "confidence": round(prob, 4),
                            "reliable":   bool(reliable),
                            "cap_hit":    bool(cap_hit),
                            "cr":         round(float(loc.get("confidence_radius") or 0), 4),
                            "error_m":    round(error_m, 4) if error_m is not None else None,
                            "localization_method": "cnn_regression",
                            "azimuth_deg_est":  round(float(loc.get("azimuth_deg", 0)), 2),
                            "azimuth_deg_true": round(float(label.get("azimuth_deg") or 0), 2),
                            "distance_m_est":   round(dist_est, 2),
                            "distance_m_true":  round(true_dist, 2),
                        })
                # Build track summaries
                tracks_out = []
                for t in [tr for tr in self._tracker.tracks if tr.active]:
                    tracks_out.append({
                        "id":        t.track_id,
                        "hits":      t.hits,
                        "positions": [
                            [round(float(v), 4) for v in p]
                            for p in t.positions[-30:]
                        ],
                        "speed":     round(float(t.speed()), 4),
                        "confirmed": t.confirmed,
                    })

                # Emit frame (compatible shape; gt_pos goes in sim_positions)
                self.socketio.emit("realtime_frame", {
                    "frame":         self.total_frames,
                    "timestamp":     round(time.time(), 3),
                    "mode":          "repository",
                    "detections":    frame_detections,
                    "tracks":        tracks_out,
                    "sim_positions": [gt_pos_out] if gt_pos_out else [],
                    "threshold":     self.threshold,
                    "n_drones_sim":  1,
                    # Repository-specific metadata shown in the UI info strip
                    "repo_label": {
                        "segment_id":      label.get("segment_id"),
                        "gt_segment_id":   label.get("gt_segment_id"),
                        "maneuver_type":   label.get("maneuver_type"),
                        "flight_phase":    label.get("flight_phase"),
                        "description":     label.get("description"),
                        "split":           label.get("split"),
                        "source":          label.get("source"),
                        "dataset_type":    label.get("dataset_type", self.dataset_type),
                        "azimuth_deg":     round(float(label.get("azimuth_deg") or 0), 1),
                        "distance_m":      round(float(label.get("distance_m") or 0), 1),
                        "height_m":        round(float(label.get("height_m") or 0), 1),
                        "has_position":    bool(label.get("has_position")),
                        # Dunakeszi-specific — all GT-enriched fields
                        "session":         label.get("session"),
                        "show_number":     label.get("show_number"),
                        "wall_clock":      label.get("wall_clock"),
                        "session_description": label.get("session_description"),
                        "array":           label.get("array", self.array),
                        "n_drones":        label.get("n_drones"),
                        "drones":          label.get("drones"),
                        "speed_mps":       label.get("speed_mps"),
                        "radius_m":        label.get("radius_m"),
                        "duration_s":      label.get("duration_s"),
                        "quality_flags":   label.get("quality_flags", []),
                        "mems_available":  label.get("mems_available"),
                        "local_start_hms": label.get("local_start_hms"),
                        "local_end_hms":   label.get("local_end_hms"),
                        "utc_start_hms":   label.get("utc_start_hms"),
                        "onset_from_rec_s":label.get("onset_from_rec_s"),
                        "within_session_s":label.get("within_session_s"),
                        "gpx_folder":      label.get("gpx_folder"),
                        "clip_start_s":    label.get("clip_start_s"),
                        "trajectory":      label.get("trajectory"),
                        "audio_file":      label.get("audio_file"),
                    },
                })

                if self.total_frames % 10 == 0:
                    self.socketio.emit("realtime_stats", self.get_stats())

            except Exception as exc:
                log.exception("Repo frame %d error: %s", self.total_frames, exc)

            # Throttle to tick_rate
            elapsed = time.time() - tick_start
            wait    = tick_secs - elapsed
            if wait > 0 and not self._stop.wait(wait):
                pass  # timeout = normal; event set = stop requested

        # Generator exhausted or stopped
        self.running = False
        self.socketio.emit("realtime_status", {
            "running":     False,
            "mode":        "repository",
            "error":       None,
            "final_stats": self.get_stats(),
        })
        log.info("Repository session finished: %d segments processed", self.total_frames)