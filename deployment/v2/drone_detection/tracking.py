# -*- coding: utf-8 -*-
"""
tracking.py
───────────
Kalman filter-based multi-object tracker for drone localization.

Classes
───────
KalmanTrack   — single tracked object with state (x, y, vx, vy)
KalmanTracker — multi-object tracker using greedy nearest-neighbour matching
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Config, config


class KalmanTrack:
    """
    Constant-velocity Kalman filter for one tracked drone.

    State vector: [x, y, vx, vy]  (positions in metres, velocities in m/frame)

    Parameters
    ──────────
    xy  : initial (x, y) position
    dt  : nominal frame interval in seconds
    cfg : pipeline configuration
    """

    _id_counter: int = 0

    def __init__(
        self,
        xy:  np.ndarray,
        dt:  float = 1.0,
        cfg: Optional[Config] = None,
    ):
        KalmanTrack._id_counter += 1
        self.track_id = KalmanTrack._id_counter
        cfg           = cfg or config

        sigma_q = cfg.KF_PROCESS_NOISE
        sigma_r = cfg.KF_MEASURE_NOISE

        # State covariance — position initialised from measurement noise
        self.P = np.diag([sigma_r ** 2, sigma_r ** 2, 1.0, 1.0]).astype(np.float64)
        # State mean
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float64)
        # Measurement noise
        self.R = np.diag([sigma_r ** 2, sigma_r ** 2]).astype(np.float64)
        # Observation matrix (we observe only x, y)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        self.cfg        = cfg
        self.age        = 0          # frames since last update
        self.hits       = 1          # total number of successful updates
        self.positions  = [xy.copy()]
        self.timestamps = [time.time()]
        self.active     = True

    # ──────────────────────────────────────────────────────────────────────────
    # Kalman predict / update
    # ──────────────────────────────────────────────────────────────────────────

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete-time process noise covariance."""
        s  = self.cfg.KF_PROCESS_NOISE
        dt2 = dt ** 2; dt3 = dt ** 3; dt4 = dt ** 4
        return np.array(
            [[dt4 / 4, 0, dt3 / 2, 0],
             [0, dt4 / 4, 0, dt3 / 2],
             [dt3 / 2, 0, dt2, 0],
             [0, dt3 / 2, 0, dt2]],
            dtype=np.float64,
        ) * s ** 2

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Propagate state forward by dt seconds; returns predicted (x, y)."""
        F = self._F(dt); Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        return self.x[:2].copy()

    def update(self, xy: np.ndarray, timestamp: Optional[float] = None):
        """Fuse a new measurement into the filter state."""
        z = xy.astype(np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.positions.append(xy.copy())
        self.timestamps.append(timestamp or time.time())
        self.age  = 0
        self.hits += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────────

    def predicted_xy(self) -> np.ndarray:
        return self.x[:2].copy()

    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def uncertainty_radius(self) -> float:
        """Approximate 1-σ position uncertainty in metres."""
        return float(np.sqrt(np.trace(self.P[:2, :2])))

    def total_distance(self) -> float:
        """Total path length in metres."""
        pts = np.array(self.positions)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0

    def to_dict(self) -> dict:
        return {
            "track_id":      self.track_id,
            "positions":     [p.tolist() for p in self.positions],
            "timestamps":    self.timestamps,
            "velocity_mps":  self.velocity().tolist(),
            "total_dist_m":  self.total_distance(),
            "active":        self.active,
        }


class KalmanTracker:
    """
    Multi-object Kalman tracker using greedy nearest-neighbour data association.

    Tracks are confirmed when they reach cfg.KF_MIN_HITS updates and
    deactivated when they coast for more than cfg.KF_MAX_COAST frames.

    v2 defaults (from multidrone patch):
        KF_MATCH_GATE = 8.0 m  (was 2.0 — too tight for TDOA accuracy)
        KF_MIN_HITS   = 1      (was 2 — lower so single detections confirm)
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg    = cfg or config
        self.tracks: List[KalmanTrack] = []
        self.frame  = 0
        self.dt     = 1.0

    def step(
        self,
        detections: List[np.ndarray],
        timestamp:  Optional[float] = None,
    ) -> List[KalmanTrack]:
        """
        Run one tracker step.

        1. Predict all active tracks forward.
        2. Greedy match detections to tracks by Euclidean distance.
        3. Update matched tracks, spawn new ones for unmatched detections.
        4. Deactivate tracks that have coasted too long.

        Returns the list of currently active confirmed tracks.
        """
        ts = timestamp or time.time()

        # Update dt from the most recent track timestamp
        if self.tracks:
            prev_ts = (
                self.tracks[0].timestamps[-1]
                if self.tracks[0].timestamps else ts
            )
            self.dt = max(0.01, ts - prev_ts)

        # Predict
        predicted = {}
        for t in self.tracks:
            if t.active:
                predicted[t.track_id] = t.predict(self.dt)

        # Match
        active      = [t for t in self.tracks if t.active]
        unmatched   = list(range(len(detections)))
        for track in sorted(active, key=lambda t: -t.hits):
            if not unmatched:
                break
            pred = predicted[track.track_id]
            dists = sorted(
                [(i, float(np.linalg.norm(pred - detections[i]))) for i in unmatched],
                key=lambda d: d[1],
            )
            best_i, best_d = dists[0]
            if best_d <= self.cfg.KF_MATCH_GATE:
                track.update(detections[best_i], ts)
                unmatched.remove(best_i)

        # Spawn new tracks for unmatched detections
        for i in unmatched:
            self.tracks.append(KalmanTrack(detections[i], self.dt, self.cfg))

        # Deactivate coasted tracks
        for t in self.tracks:
            if t.age > self.cfg.KF_MAX_COAST:
                t.active = False

        self.frame += 1
        return [t for t in self.tracks if t.active and t.hits >= self.cfg.KF_MIN_HITS]

    def all_confirmed(self) -> List[KalmanTrack]:
        """Return all tracks (active or inactive) that reached KF_MIN_HITS."""
        return [t for t in self.tracks if t.hits >= self.cfg.KF_MIN_HITS]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([t.to_dict() for t in self.tracks], f, indent=2)
        print(f"💾 Tracks saved: {path}")
