# -*- coding: utf-8 -*-
"""
app_v2.py — Drone Detection System v2 Flask Application
========================================================
Location: deployment/v2/app_v2.py

Uses the drone_detection package (v15) exclusively:
  - detect() + localize()       CNN + heuristic hybrid
  - localize_multi_drone()      Cartesian Nelder-Mead solver
  - synthesise_drone()          fractional-delay synthesis
  - LocalizationCNN             single model class (no Lite variant)

All API routes are prefixed /api/v2/.
Default port: 5001  (v1 uses 5000)
"""

from __future__ import annotations

# ── Path bootstrap ─────────────────────────────────────────────────────────
import sys
from pathlib import Path

_THIS_DIR  = Path(__file__).parent.resolve()
_REPO_ROOT = _THIS_DIR.parent.parent.resolve()

for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import os
import tempfile
import threading
import time

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [drone_v2] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drone_v2")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "drone-v2-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

CORS(app, resources={r"/api/v2/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── Config / model singleton ──────────────────────────────────────────────────

def _get_config():
    """Return a Config pointed at this deployment's models directory."""
    from drone_detection import Config
    cfg = Config()
    cfg.DRIVE_MODELS = _THIS_DIR / "models"
    cfg.DRIVE_ROOT   = _THIS_DIR
    cfg.DRIVE_LOGS   = _THIS_DIR / "logs"
    return cfg


def get_model():
    """Load or return cached (model, config) tuple."""
    try:
        from drone_detection import load_detection_model
        cfg = _get_config()
        m   = load_detection_model(cfg)
        return m, cfg
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        raise


# ── Realtime session registry ──────────────────────────────────────────────────
_realtime_sessions: dict = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str = "default"):
    with _sessions_lock:
        return _realtime_sessions.get(session_id)


def _register_session(session_id: str, session) -> None:
    with _sessions_lock:
        _realtime_sessions[session_id] = session


def _remove_session(session_id: str) -> None:
    with _sessions_lock:
        _realtime_sessions.pop(session_id, None)


# ── Upload helpers ─────────────────────────────────────────────────────────────
ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _save_upload(file_obj, suffix: str = ".wav") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_obj.save(tf.name)
    return tf.name


def _cleanup(*paths: str) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _file_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _clean(obj):
    """Recursively convert numpy types to Python scalars."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/v2/version", methods=["GET"])
def version():
    return jsonify({
        "version": "2.0",
        "pipeline": "drone_detection v15",
        "fixes_applied": [
            "Fractional sinc delay in synthesise_drone()",
            "Source-level noise delayed with harmonics (GCC-PHAT fix)",
            "Cartesian Nelder-Mead localize_multi_drone()",
            "TDOA dedup window 29µs (was 0.05µs — 580x too tight)",
            "Kalman match gate 8m, min_hits=1",
            "LocalizationCNN only — LocalizationCNNLite removed",
            "USE_LITE_LOC removed from Config",
            "Focal loss + auto threshold search training",
            "feature_stack [log-mel, PCEN, delta-mel]",
            "CNN + heuristic hybrid detect()",
            "Tracker feeds real localize() positions per segment",
            "analyse_audio_file shows track trajectory plot",
            "quick_demo prints and plots confirmed tracks",
        ],
        "timestamp": time.time(),
    })


@app.route("/api/v2/status", methods=["GET"])
def status():
    try:
        m, cfg = get_model()
        model_ok = m is not None
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    with _sessions_lock:
        active = {
            sid: {"mode": getattr(s, "_mode", "?"), "running": s.running}
            for sid, s in _realtime_sessions.items()
            if s.running
        }

    return jsonify({
        "status":              "ok",
        "model_loaded":        model_ok,
        "device":              str(getattr(cfg, "DEVICE", "unknown")),
        "mic_positions":       (
            cfg.MIC_POSITIONS.tolist()
            if hasattr(getattr(cfg, "MIC_POSITIONS", None), "tolist") else []
        ),
        "sample_rate":         getattr(cfg, "SR", 22050),
        "detection_threshold": getattr(cfg, "DETECTION_THRESHOLD", 0.62),
        "active_sessions":     active,
        "timestamp":           time.time(),
    })


@app.route("/api/v2/detect", methods=["POST"])
def detect_single():
    """
    Single audio file detection.

    Form fields
    -----------
    file        : audio file (required)
    drone_x     : hint X position metres (default 1.0)
    drone_y     : hint Y position metres (default 0.8)
    threshold   : detection threshold 0.1–0.99 (default cfg.DETECTION_THRESHOLD)
    n_segments  : number of segments to analyse (default 5)
    force_detect: skip classifier, always localise (default false)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if _file_ext(f.filename) not in ALLOWED_AUDIO:
        return jsonify({"error": f"Unsupported format: {f.filename}"}), 400

    n_segments   = int(  request.form.get("n_segments",  5))
    force_detect = request.form.get("force_detect", "false").lower() == "true"

    tmp = _save_upload(f, suffix=_file_ext(f.filename) or ".wav")
    try:
        m, cfg = get_model()
        threshold = float(request.form.get("threshold", cfg.DETECTION_THRESHOLD))

        from drone_detection import AudioProcessor, detect, localize

        ap       = AudioProcessor(cfg)
        y        = ap.pad_or_truncate(ap.load(tmp))
        channels = [y, y, y]

        det      = detect(channels, cfg)
        prob     = float(det["probability"])
        detected = force_detect or (prob >= threshold)

        loc = None
        if detected:
            loc = localize(channels, cfg)

        resp = _clean({
            "detected":              detected,
            "probability":           prob,
            "cnn_probability":       float(det.get("cnn_probability", prob)),
            "heuristic_probability": float(det.get("heuristic_probability", 0.0)),
            "position":              loc["xy_position"] if loc else None,
            "azimuth_deg":           loc["azimuth_deg"] if loc else None,
            "distance_m":            loc["distance_m"]  if loc else None,
            "height_m":              loc["height_m"]    if loc else None,
            "detection_summary": {
                "total_segments":    n_segments,
                "detected_segments": 1 if detected else 0,
                "max_confidence":    prob,
            },
            "n_tracks": 0,
        })

        if detected:
            socketio.emit("drone_detected_v2", {
                "timestamp":  time.time(),
                "confidence": prob,
                "position":   resp.get("position"),
                "source":     f.filename,
            })

        return jsonify(resp)

    except Exception as exc:
        log.exception("detect_single failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        _cleanup(tmp)


@app.route("/api/v2/detect-3mic", methods=["POST"])
def detect_3mic():
    """
    3-microphone precise localization.

    Form fields
    -----------
    mic1, mic2, mic3 : one audio file per mic (required)
    threshold        : detection threshold (default cfg.DETECTION_THRESHOLD)
    hint_x, hint_y   : initial search position (optional, unused — kept for
                       API compatibility)
    """
    for key in ("mic1", "mic2", "mic3"):
        if key not in request.files:
            return jsonify({"error": f"Missing file: {key}"}), 400

    tmps: list = []
    try:
        m, cfg    = get_model()
        threshold = float(request.form.get("threshold", cfg.DETECTION_THRESHOLD))

        for key in ("mic1", "mic2", "mic3"):
            tmps.append(_save_upload(
                request.files[key],
                suffix=_file_ext(request.files[key].filename) or ".wav",
            ))

        from drone_detection import AudioProcessor, detect, localize

        ap       = AudioProcessor(cfg)
        channels = [ap.pad_or_truncate(ap.load(p)) for p in tmps]

        det      = detect(channels, cfg)
        prob     = float(det["probability"])
        detected = prob >= threshold

        loc = None
        if detected:
            loc = localize(channels, cfg)

        resp: dict = {"detected": detected, "probability": prob, "localization": None}
        if loc:
            resp["localization"] = _clean({
                "position":          loc["xy_position"],
                "azimuth_deg":       loc["azimuth_deg"],
                "distance_m":        loc["distance_m"],
                "height_m":          loc["height_m"],
                "reliable":          True,
                "quality_message":   "OK",
                "confidence_radius": 0.0,
            })
            socketio.emit("drone_detected_v2", {
                "timestamp":  time.time(),
                "confidence": prob,
                "position":   _clean(loc["xy_position"]),
                "source":     "3-mic",
            })

        return jsonify(resp)

    except Exception as exc:
        log.exception("detect_3mic failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        _cleanup(*tmps)


@app.route("/api/v2/detect-multi", methods=["POST"])
def detect_multi():
    """
    Multi-drone detection using Cartesian Nelder-Mead localizer.

    Form fields
    -----------
    mic1, mic2, mic3 : one audio file per mic (required)
    threshold        : detection threshold (default cfg.DETECTION_THRESHOLD)
    max_drones       : max drones to report (default 3)
    """
    for key in ("mic1", "mic2", "mic3"):
        if key not in request.files:
            return jsonify({"error": f"Missing file: {key}"}), 400

    tmps: list = []
    try:
        m, cfg     = get_model()
        threshold  = float(request.form.get("threshold",  cfg.DETECTION_THRESHOLD))
        max_drones = int(  request.form.get("max_drones", 3))

        for key in ("mic1", "mic2", "mic3"):
            tmps.append(_save_upload(
                request.files[key],
                suffix=_file_ext(request.files[key].filename) or ".wav",
            ))

        from drone_detection import AudioProcessor, detect, localize_multi_drone

        ap       = AudioProcessor(cfg)
        channels = [ap.pad_or_truncate(ap.load(p)) for p in tmps]

        det  = detect(channels, cfg)
        prob = float(det["probability"])

        drones_raw = []
        if prob >= threshold:
            drones_raw = localize_multi_drone(channels, cfg, max_drones=max_drones)

        drones_out = [
            {
                "id":                i + 1,
                "position":          _clean(d["xy_position"]),
                "azimuth_deg":       float(d["azimuth_deg"]),
                "distance_m":        float(d["distance_m"]),
                "reliable":          float(d.get("tdoa_residual", 1.0)) < 1e-8,
                "confidence_radius": float(d.get("confidence_radius") or 0.0),
                "band_hz":           [0, 0],
                "tdoa_residual":     float(d.get("tdoa_residual", 0.0)),
            }
            for i, d in enumerate(drones_raw)
        ]

        resp = {
            "detected":    bool(drones_out),
            "n_drones":    len(drones_out),
            "probability": prob,
            "drones":      drones_out,
        }
        if drones_out:
            socketio.emit("drone_detected_v2", {
                "timestamp":  time.time(),
                "confidence": prob,
                "n_drones":   len(drones_out),
                "drones":     drones_out,
                "source":     "multi-drone",
            })

        return jsonify(resp)

    except Exception as exc:
        log.exception("detect_multi failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        _cleanup(*tmps)


@app.route("/api/v2/noise-test", methods=["POST"])
def noise_test():
    """
    SNR sweep robustness test using synthesise_drone().

    Form fields
    -----------
    snr_min   : lowest SNR level dB  (default -5)
    snr_max   : highest SNR level dB (default 20)
    snr_step  : step dB              (default 5)
    n_clips   : clips per level      (default 20)
    """
    snr_min  = int(request.form.get("snr_min",  -5))
    snr_max  = int(request.form.get("snr_max",  20))
    snr_step = int(request.form.get("snr_step",  5))
    n_clips  = int(request.form.get("n_clips",  20))
    snr_levels = list(range(snr_min, snr_max + 1, snr_step))

    try:
        m, cfg = get_model()

        from drone_detection import AudioProcessor, detect, synthesise_drone
        import numpy as np

        ap      = AudioProcessor(cfg)
        mics    = cfg.MIC_POSITIONS
        results = {}

        for snr_db in snr_levels:
            n_det    = 0
            conf_sum = 0.0
            for _ in range(n_clips):
                r     = np.random.uniform(0.3, 2.5)
                theta = np.random.uniform(0, 2 * np.pi)
                cx, cy = float(cfg.ARRAY_CENTER[0]), float(cfg.ARRAY_CENTER[1])
                pos   = [cx + r * np.cos(theta), cy + r * np.sin(theta)]
                fund  = int(np.random.choice([80, 90, 100, 110, 120, 130]))
                chs   = synthesise_drone(mics, pos, fundamental=fund,
                                         noise_level=0.05)
                chs   = [ap.pad_or_truncate(c) for c in chs]
                if snr_db < 30:
                    chs = [ap.add_noise(c, snr_db) for c in chs]
                det = detect(chs, cfg)
                if det["detected"]:
                    n_det += 1
                conf_sum += float(det["probability"])

            results[snr_db] = {
                "detection_rate": round(n_det / n_clips * 100, 1),
                "avg_confidence": round(conf_sum / n_clips, 4),
                "n_clips":        n_clips,
                "n_detected":     n_det,
            }

        return jsonify({"status": "ok", "snr_levels": snr_levels, "results": results})

    except Exception as exc:
        log.exception("noise_test failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v2/path-simulate", methods=["POST"])
def path_simulate():
    """
    Synthetic path tracking simulation.

    Form fields
    -----------
    n_waypoints : number of waypoints (default 8)
    spread      : max radius metres   (default 2.0)
    """
    n_waypoints = int(  request.form.get("n_waypoints", 8))
    spread      = float(request.form.get("spread",      2.0))

    try:
        m, cfg = get_model()

        from drone_detection import AudioProcessor, detect, localize, synthesise_drone
        from realtime_sessions import PathTracker, DroneTrack
        import numpy as np

        ap = AudioProcessor(cfg)
        cx, cy = float(cfg.ARRAY_CENTER[0]), float(cfg.ARRAY_CENTER[1])

        import math
        angles = np.linspace(0, 2 * np.pi * 1.5, n_waypoints)
        radii  = np.linspace(0.3, min(spread, 2.5), n_waypoints)
        waypts = [
            (cx + r * math.cos(a), cy + r * math.sin(a))
            for r, a in zip(radii, angles)
        ]

        DroneTrack._id_counter = 0
        tracker  = PathTracker(cfg)
        base_ts  = time.time()

        for i, wp in enumerate(waypts):
            chs = synthesise_drone(
                cfg.MIC_POSITIONS, wp,
                fundamental=int(np.random.choice([90, 100, 110])),
                noise_level=0.05,
            )
            chs = [ap.pad_or_truncate(c) for c in chs]
            det = detect(chs, cfg)
            if det["detected"]:
                loc = localize(chs, cfg)
                tracker.update(
                    [np.array(loc["xy_position"])],
                    timestamp=base_ts + i,
                )

        tracks_out = []
        for t in tracker.confirmed_tracks:
            spd = t.speed() or 0.0
            tracks_out.append({
                "id":        t.track_id,
                "waypoints": len(t.positions),
                "positions": _clean(t.positions),
                "speed_m_s": round(float(spd), 4),
            })

        return jsonify({
            "status":      "ok",
            "n_waypoints": n_waypoints,
            "spread":      spread,
            "n_tracks":    len(tracks_out),
            "tracks":      tracks_out,
        })

    except Exception as exc:
        log.exception("path_simulate failed")
        return jsonify({"error": str(exc)}), 500


# ── Realtime endpoints ─────────────────────────────────────────────────────────

@app.route("/api/v2/realtime/start", methods=["POST"])
def realtime_start():
    """
    Start a real-time detection session.

    Form fields
    -----------
    mode            : 'simulated' | 'real'   (default: 'simulated')
    session_id      : string key             (default: 'default')
    threshold       : 0.1–0.99

    Simulated only:
      n_drones      : 1–3
      pattern       : circle|figure8|linear|random|multi
      tick_rate     : frames/sec
      noise_level   : 0.001–0.2
      spread        : max metres from array

    Real only:
      segment_dur   : seconds per window
      device_indices: comma-separated ints
    """
    mode       = request.form.get("mode",       "simulated")
    session_id = request.form.get("session_id", "default")

    existing = _get_session(session_id)
    if existing and existing.running:
        existing.stop()
        _remove_session(session_id)

    try:
        m, cfg = get_model()
    except Exception as exc:
        return jsonify({"error": f"Model not ready: {exc}"}), 500

    threshold = float(request.form.get("threshold", cfg.DETECTION_THRESHOLD))

    from realtime_sessions import SimulatedRealtimeSession, RealRealtimeSession, RepositoryRealtimeSession

    if mode == "simulated":
        n_drones    = int(  request.form.get("n_drones",    1))
        raw_pattern = request.form.get("pattern", "circle")
        tick_rate   = float(request.form.get("tick_rate",   1.0))
        noise_level = float(request.form.get("noise_level", 0.05))
        spread      = float(request.form.get("spread",      1.5))

        patterns = (
            ["circle", "figure8", "random"][:n_drones]
            if raw_pattern == "multi"
            else [raw_pattern] * n_drones
        )

        session = SimulatedRealtimeSession(
            cfg, socketio,
            n_drones=n_drones, patterns=patterns,
            tick_rate=tick_rate, threshold=threshold,
            noise_level=noise_level, spread=spread,
        )

    elif mode == "real":
        segment_dur = float(request.form.get("segment_dur", 3.0))
        dev_raw     = request.form.get("device_indices", "")
        dev_indices = [int(x) for x in dev_raw.split(",") if x.strip().isdigit()]

        session = RealRealtimeSession(
            cfg, socketio,
            threshold=threshold,
            segment_dur=segment_dur,
            device_indices=dev_indices,
        )

    elif mode == "repository":
        dataset_type  = request.form.get("dataset_type", "uavirbase")
        array         = request.form.get("array",         "BK-6-E")
        url           = request.form.get("url",           None) or None
        required_split= request.form.get("required_split",None) or None
        tick_rate     = float(request.form.get("tick_rate",    1.0))
        max_dist      = float(request.form.get("max_dist",    100.0))
        allow_fallback= request.form.get("allow_synthetic_fallback", "true").lower() == "true"

        session = RepositoryRealtimeSession(
            cfg, socketio,
            url                      = url,
            dataset_type             = dataset_type,
            array                    = array,
            max_dist                 = max_dist,
            tick_rate                = tick_rate,
            threshold                = threshold,
            allow_download           = False,   # NEVER download on live server (datasets are 4 GB+)
            allow_synthetic_fallback = allow_fallback,
            required_split           = required_split,
        )
    else:
        return jsonify({"error": f"Unknown mode: {mode}. Use 'simulated', 'real', or 'repository'"}), 400

    session._mode = mode
    _register_session(session_id, session)

    if not session.start():
        _remove_session(session_id)
        return jsonify({"error": "Session failed to start — check server logs"}), 500

    log.info(f"realtime/start: mode={mode} session={session_id}")
    socketio.emit("realtime_status", {
        "running": True, "mode": mode, "error": None, "session_id": session_id,
    })
    return jsonify({"status": "started", "mode": mode, "session_id": session_id})


@app.route("/api/v2/realtime/stop", methods=["POST"])
def realtime_stop():
    session_id = request.form.get("session_id", "default")
    session    = _get_session(session_id)
    if not session:
        return jsonify({"error": "No active session"}), 404

    stats = session.get_stats()
    session.stop()
    _remove_session(session_id)

    log.info(f"realtime/stop: session={session_id}")
    return jsonify({"status": "stopped", "session_id": session_id, "stats": stats})


@app.route("/api/v2/realtime/status", methods=["GET"])
def realtime_status_endpoint():
    session_id = request.args.get("session_id", "default")
    session    = _get_session(session_id)
    if not session:
        return jsonify({"running": False, "session_id": session_id})
    return jsonify({
        "running":    session.running,
        "session_id": session_id,
        "mode":       getattr(session, "_mode", "?"),
        "stats":      session.get_stats(),
    })


@app.route("/api/v2/realtime/audio-devices", methods=["GET"])
def list_audio_devices():
    try:
        import pyaudio
        pa      = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index":    i,
                    "name":     info["name"],
                    "channels": info["maxInputChannels"],
                    "rate":     int(info["defaultSampleRate"]),
                })
        pa.terminate()
        return jsonify({"devices": devices})
    except ImportError:
        return jsonify({"devices": [], "error": "PyAudio not installed"})
    except Exception as exc:
        return jsonify({"devices": [], "error": str(exc)})


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index_v2.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── WebSocket ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def ws_connect():
    log.info("WebSocket client connected")
    emit("status_v2", {
        "message":  "Connected to Drone Detection v2",
        "version":  "2.0",
        "pipeline": "drone_detection v15",
    })


@socketio.on("disconnect")
def ws_disconnect():
    log.info("WebSocket client disconnected")


@socketio.on("ping_v2")
def ws_ping(data):
    emit("pong_v2", {"timestamp": time.time()})


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 50 MB)"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500