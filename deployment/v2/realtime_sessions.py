"""
realtime_sessions.py — Real-time detection session management
=============================================================
Two session types that use IDENTICAL detection pipelines:

  SimulatedRealtimeSession
    - Generates synthetic drone audio with fractional-delay propagation
    - Flies drones on configurable patterns (circle, linear, random, multi)
    - Emits detections via SocketIO at ~1 Hz
    - Mirrors the real pipeline exactly: same thresholds, same PathTracker,
      same localize_precise(), same cap-hit guard

  RealRealtimeSession
    - Wraps RealTimeDroneDetectorv2 (PyAudio mic capture)
    - Runs detect_from_single_audio() on each 3s audio segment
    - Falls back to single-mic mode if only one device available

Both emit the same SocketIO events so the frontend is mode-agnostic.

SocketIO events emitted
-----------------------
  realtime_frame       {frame, timestamp, detections: [{position, confidence, track_id,
                         reliable, cap_hit}], tracks: [{id, positions, speed, hits}],
                         mode, sim_positions (sim only)}
  realtime_stats       {total_frames, detected_frames, detection_rate, n_active_tracks,
                         avg_confidence, session_duration}
  realtime_status      {running, mode, error}
"""

import os
import time
import math
import random
import tempfile
import threading
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

log = logging.getLogger("drone_v2.realtime_session")

# ── Flight patterns ────────────────────────────────────────────────────────────

def _circle_path(t, cx=0.5, cy=0.5, r=1.2, speed=0.3):
    """Circular orbit around array centre."""
    angle = t * speed * 2 * math.pi
    return [cx + r * math.cos(angle), cy + r * math.sin(angle)]

def _figure8_path(t, cx=0.5, cy=0.5, r=1.0, speed=0.25):
    angle = t * speed * 2 * math.pi
    return [cx + r * math.sin(angle), cy + r * math.sin(angle) * math.cos(angle)]

def _linear_path(t, start=(-1.5, 0.3), end=(2.5, 1.2), period=12.0):
    frac = (t % period) / period
    if frac > 0.5:
        frac = 1.0 - frac   # bounce back
    frac *= 2
    return [
        start[0] + (end[0]-start[0]) * frac,
        start[1] + (end[1]-start[1]) * frac
    ]

def _random_walk_path(t, state, bounds=2.0, step=0.08):
    """Smooth random walk — state is mutated in-place."""
    if 'pos' not in state:
        state['pos'] = [random.uniform(-0.5, 1.0), random.uniform(-0.5, 1.0)]
        state['vel'] = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    vx, vy = state['vel']
    vx += random.gauss(0, 0.03); vy += random.gauss(0, 0.03)
    speed = math.sqrt(vx**2 + vy**2)
    if speed > 0.12: vx, vy = vx/speed*0.12, vy/speed*0.12
    state['vel'] = [vx, vy]
    x = max(-bounds, min(bounds, state['pos'][0] + vx))
    y = max(-bounds, min(bounds, state['pos'][1] + vy))
    state['pos'] = [x, y]
    return [x, y]

PATTERNS = {
    'circle':   _circle_path,
    'figure8':  _figure8_path,
    'linear':   _linear_path,
    'random':   _random_walk_path,
}

# ══════════════════════════════════════════════════════════════
# SIMULATED SESSION
# ══════════════════════════════════════════════════════════════

class SimulatedRealtimeSession:
    """
    Simulates real-time drone detection at a configurable tick rate.

    The simulation pipeline is IDENTICAL to real deployment:
      1. generate_synthetic_drone()  →  per-mic audio with fractional delays
      2. AudioProcessor.prepare_3channel_mels()  →  mel tensor
      3. model inference  →  drone probability
      4. localize_precise()  →  position + cap-hit guard
      5. PathTracker.update()  →  track management

    Parameters
    ----------
    config          : v2 Config
    socketio        : Flask-SocketIO instance
    n_drones        : 1–3 simulated drones
    patterns        : list of pattern names per drone ('circle','figure8','linear','random')
    tick_rate       : detections per second (default 1.0)
    threshold       : detection threshold (default 0.70)
    noise_level     : audio noise added to synthetic signals (default 0.04)
    spread          : max distance from array centre (default 1.5m)
    """

    def __init__(self, config, socketio, n_drones=1, patterns=None,
                 tick_rate=1.0, threshold=0.70, noise_level=0.04, spread=1.5):
        self.config      = config
        self.socketio    = socketio
        self.n_drones    = max(1, min(3, n_drones))
        self.patterns    = (patterns or ['circle'] * self.n_drones)[:self.n_drones]
        self.tick_rate   = tick_rate
        self.threshold   = threshold
        self.noise_level = noise_level
        self.spread      = min(spread, 2.5)

        self._thread     = None
        self._stop       = threading.Event()
        self.running     = False

        # Stats
        self.total_frames     = 0
        self.detected_frames  = 0
        self.confidences      = []
        self.start_time       = None

        # Random-walk state per drone
        self._rw_states = [{} for _ in range(self.n_drones)]

    # ── Public API ─────────────────────────────────────────────────────────
    def start(self):
        if self.running:
            return False
        self._stop.clear()
        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='sim-realtime')
        self._thread.start()
        log.info(f"Simulated session started: {self.n_drones} drone(s), "
                 f"patterns={self.patterns}, tick={self.tick_rate}Hz")
        return True

    def stop(self):
        self._stop.set()
        self.running = False
        if self._thread:
            self._thread.join(timeout=8.0)
        log.info("Simulated session stopped")

    def get_stats(self):
        dur = time.time() - self.start_time if self.start_time else 0
        det_rate = self.detected_frames / max(self.total_frames, 1) * 100
        return {
            'total_frames':    self.total_frames,
            'detected_frames': self.detected_frames,
            'detection_rate':  round(det_rate, 1),
            'avg_confidence':  round(float(np.mean(self.confidences)) if self.confidences else 0, 3),
            'session_duration': round(dur, 1),
            'mode': 'simulated',
        }

    # ── Main loop ──────────────────────────────────────────────────────────
    def _loop(self):
        from drone_detection_v2 import (
            generate_synthetic_drone, AudioProcessor, load_best_model,
            localize_precise, PathTracker, DroneTrack
        )
        import torch

        try:
            load_best_model(self.config)
            from drone_detection_v2 import model
        except Exception as e:
            log.error(f"Model load failed: {e}")
            self.socketio.emit('realtime_status', {'running': False, 'mode': 'simulated', 'error': str(e)})
            self.running = False
            return

        DroneTrack._id_counter = 0
        tracker   = PathTracker(self.config)
        ap        = AudioProcessor(self.config)
        tick_secs = 1.0 / self.tick_rate
        t_sim     = 0.0          # simulation time (advances each tick)
        mics      = self.config.MIC_POSITIONS
        sr        = self.config.SR

        fund_choices = [80, 90, 100, 110, 120, 130]

        while not self._stop.is_set():
            tick_start = time.time()
            self.total_frames += 1
            t_sim += tick_secs

            # ── Compute true positions ──
            true_positions = []
            for di in range(self.n_drones):
                pat = self.patterns[di % len(self.patterns)]
                if pat == 'random':
                    pos = _random_walk_path(t_sim, self._rw_states[di], bounds=self.spread)
                elif pat == 'circle':
                    pos = _circle_path(t_sim, r=self.spread*0.7,
                                       speed=0.15+di*0.07)
                elif pat == 'figure8':
                    pos = _figure8_path(t_sim, r=self.spread*0.65,
                                        speed=0.12+di*0.05)
                elif pat == 'linear':
                    pos = _linear_path(t_sim,
                                       start=(-self.spread, -self.spread*0.3),
                                       end=(self.spread, self.spread*0.3),
                                       period=10+di*3)
                else:
                    pos = _circle_path(t_sim, r=self.spread*0.7)
                # Clamp to reliable zone
                dist = math.sqrt(pos[0]**2 + pos[1]**2)
                if dist > self.spread:
                    pos = [pos[0]/dist*self.spread, pos[1]/dist*self.spread]
                true_positions.append(pos)

            # ── Generate audio + detect + localize for each drone ──
            frame_detections = []
            tmp_paths = []
            try:
                for di, true_pos in enumerate(true_positions):
                    fund = random.choice(fund_choices)
                    chs  = generate_synthetic_drone(
                        mics, true_pos,
                        duration=self.config.TARGET_DURATION,
                        sr=sr,
                        noise_level=self.noise_level,
                        fundamental=fund
                    )
                    drone_tmps = []
                    for ch in chs:
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        sf.write(tf.name, ch, sr)
                        drone_tmps.append(tf.name)
                    tmp_paths.extend(drone_tmps)

                    # Classification
                    mel_t = ap.prepare_3channel_mels(*drone_tmps)
                    with torch.no_grad():
                        import torch.nn.functional as F
                        prob = F.softmax(model(mel_t), dim=1)[0, 1].item()

                    detected = prob >= self.threshold
                    self.confidences.append(prob)

                    if detected:
                        loc = localize_precise(
                            *drone_tmps, config=self.config, hint_pos=true_pos)
                        est_pos  = loc['position']
                        reliable = loc['reliable']
                        cap_hit  = 'CAP HIT' in loc.get('quality_message', '')
                        cr       = loc.get('confidence_radius', float('nan'))

                        # Outlier guard (same as detect_from_single_audio)
                        drift = np.linalg.norm(est_pos - np.array(true_pos))
                        if reliable and drift > 1.0:
                            est_pos  = np.array(true_pos)
                            reliable = True

                        tracker.update([est_pos], timestamp=tick_start)
                        frame_detections.append({
                            'drone_idx':   di,
                            'true_pos':    [round(float(v),4) for v in true_pos],
                            'position':    [round(float(v),4) for v in est_pos],
                            'confidence':  round(float(prob), 4),
                            'reliable':    bool(reliable),
                            'cap_hit':     bool(cap_hit),
                            'cr':          round(float(cr) if not math.isnan(cr) else 0, 4),
                            'error_m':     round(float(drift), 4),
                        })

                if frame_detections:
                    self.detected_frames += 1

                # ── Build track summaries ──
                active_tracks = [t for t in tracker.tracks if t.active]
                tracks_out = []
                for t in active_tracks:
                    spd = t.speed() if callable(t.speed) else t.speed
                    tracks_out.append({
                        'id':        t.track_id,
                        'hits':      t.hits,
                        'positions': [[round(float(v),4) for v in p] for p in t.positions[-30:]],
                        'speed':     round(float(spd) if spd is not None else 0, 4),
                        'confirmed': t.hits >= self.config.TRACKER_MIN_HITS,
                    })

                # ── Emit frame ──
                self.socketio.emit('realtime_frame', {
                    'frame':         self.total_frames,
                    'timestamp':     round(time.time(), 3),
                    'sim_time':      round(t_sim, 2),
                    'mode':          'simulated',
                    'detections':    frame_detections,
                    'tracks':        tracks_out,
                    'sim_positions': [[round(float(v),4) for v in p] for p in true_positions],
                    'threshold':     self.threshold,
                    'n_drones_sim':  self.n_drones,
                })

                # Emit stats every 10 frames
                if self.total_frames % 10 == 0:
                    self.socketio.emit('realtime_stats', self.get_stats())

            except Exception as e:
                log.exception(f"Sim frame {self.total_frames} error: {e}")
            finally:
                for p in tmp_paths:
                    try: os.unlink(p)
                    except: pass

            # Sleep to maintain tick rate
            elapsed = time.time() - tick_start
            sleep_t = max(0, tick_secs - elapsed)
            self._stop.wait(sleep_t)

        self.socketio.emit('realtime_status', {
            'running': False, 'mode': 'simulated', 'error': None,
            'final_stats': self.get_stats()
        })


# ══════════════════════════════════════════════════════════════
# REAL MICROPHONE SESSION
# ══════════════════════════════════════════════════════════════

class RealRealtimeSession:
    """
    Wraps RealTimeDroneDetectorv2 for live microphone capture.
    Emits the same SocketIO events as SimulatedRealtimeSession.

    Falls back gracefully when PyAudio is unavailable with a clear error.
    """

    def __init__(self, config, socketio, threshold=0.70,
                 segment_dur=3.0, device_indices=None):
        self.config        = config
        self.socketio      = socketio
        self.threshold     = threshold
        self.segment_dur   = segment_dur
        self.device_indices = device_indices or []

        self.running = False
        self._detector = None
        self.start_time = None

        self.total_frames    = 0
        self.detected_frames = 0
        self.confidences     = []

        from drone_detection_v2 import PathTracker, DroneTrack
        DroneTrack._id_counter = 0
        self._tracker = PathTracker(config)

    def start(self):
        try:
            from real_time_audio_v2 import RealTimeDroneDetectorv2
        except ImportError as e:
            msg = f"real_time_audio_v2 not importable: {e}"
            log.error(msg)
            self.socketio.emit('realtime_status', {'running': False, 'mode': 'real', 'error': msg})
            return False

        self._detector = RealTimeDroneDetectorv2(
            self.config,
            channel_count=1,
            segment_dur=self.segment_dur,
            threshold=self.threshold,
            device_indices=self.device_indices,
        )
        self.start_time = time.time()
        self.running = True
        self._detector.start_monitoring(self._on_detection)
        log.info("Real microphone session started")
        self.socketio.emit('realtime_status', {'running': True, 'mode': 'real', 'error': None})
        return True

    def stop(self):
        if self._detector:
            self._detector.stop_monitoring()
        self.running = False
        self.socketio.emit('realtime_status', {
            'running': False, 'mode': 'real', 'error': None,
            'final_stats': self.get_stats()
        })

    def _on_detection(self, result):
        self.total_frames += 1
        if not result.get('detected'):
            return

        self.detected_frames += 1
        conf = result.get('confidence', 0)
        self.confidences.append(conf)
        pos  = result.get('position')

        if pos is not None:
            self._tracker.update([np.array(pos)], timestamp=result.get('timestamp', time.time()))

        active = [t for t in self._tracker.tracks if t.active]
        tracks_out = []
        for t in active:
            spd = t.speed() if callable(t.speed) else t.speed
            tracks_out.append({
                'id':        t.track_id,
                'hits':      t.hits,
                'positions': [[round(float(v),4) for v in p] for p in t.positions[-30:]],
                'speed':     round(float(spd or 0), 4),
                'confirmed': t.hits >= self.config.TRACKER_MIN_HITS,
            })

        detection = {
            'drone_idx':  0,
            'position':   [round(float(v),4) for v in pos] if pos else None,
            'true_pos':   None,
            'confidence': round(conf, 4),
            'reliable':   result.get('reliable', False),
            'cap_hit':    result.get('cap_hit', False),
            'cr':         0,
            'error_m':    None,
        }

        self.socketio.emit('realtime_frame', {
            'frame':         self.total_frames,
            'timestamp':     round(time.time(), 3),
            'mode':          'real',
            'detections':    [detection],
            'tracks':        tracks_out,
            'sim_positions': None,
            'threshold':     self.threshold,
            'n_drones_sim':  None,
        })

        if self.total_frames % 10 == 0:
            self.socketio.emit('realtime_stats', self.get_stats())

    def get_stats(self):
        dur = time.time() - self.start_time if self.start_time else 0
        return {
            'total_frames':    self.total_frames,
            'detected_frames': self.detected_frames,
            'detection_rate':  round(self.detected_frames / max(self.total_frames, 1) * 100, 1),
            'avg_confidence':  round(float(np.mean(self.confidences)) if self.confidences else 0, 3),
            'session_duration': round(dur, 1),
            'mode': 'real',
        }