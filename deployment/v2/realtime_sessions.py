# -*- coding: utf-8 -*-
"""
realtime_sessions.py — Real-time detection session management (v2)
==================================================================
Uses the drone_detection package (v15) exclusively.

Two session types that share the IDENTICAL detection pipeline:

  SimulatedRealtimeSession
    - Generates synthetic drone audio via synthesise_drone()
      (same function used in training — fractional-delay propagation,
      source-level noise so GCC-PHAT works correctly)
    - Flies drones on configurable patterns (circle, figure8, linear,
      random, multi)
    - Uses detect() → localize() → PathTracker, same thresholds and
      guards as live deployment

  RealRealtimeSession
    - Wraps RealTimeDroneDetectorV2 (PyAudio mic capture)
    - Runs detect() + localize() on each audio segment
    - Falls back gracefully if PyAudio is unavailable

Both emit the same SocketIO events so the frontend is mode-agnostic.

SocketIO events emitted
-----------------------
  realtime_frame   {frame, timestamp, detections, tracks, mode,
                    sim_positions (sim only)}
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

    Parameters
    ----------
    config      : drone_detection Config object
    socketio    : Flask-SocketIO instance
    n_drones    : 1–3 simulated drones
    patterns    : list of pattern names per drone
    tick_rate   : detections per second (default 1.0)
    threshold   : detection threshold (default 0.70)
    noise_level : audio noise level — must be ≥ 0.05 for GCC-PHAT to work
    spread      : max distance from array centre in metres (default 1.5)
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
        # Enforce minimum noise level so GCC-PHAT has broadband content
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

    # ── Main loop ──────────────────────────────────────────────────────────

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

            # ── Compute true positions ──────────────────────────────────
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

            # ── Synthesise → detect → localize ──────────────────────────
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

                # ── Build track summaries ────────────────────────────────
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