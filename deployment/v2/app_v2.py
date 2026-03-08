"""
app_v2.py — Drone Detection System v2 Flask Application
========================================================
Location: drone-detection-api/deployment/v2/app_v2.py

All API routes are prefixed /api/v2/.
WebSocket namespace: /v2
Default port: 5001  (v1 uses 5000)
"""

# ── Path bootstrap ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

_THIS_DIR  = Path(__file__).parent.resolve()
_REPO_ROOT = _THIS_DIR.parent.parent.resolve()

for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import time
import json
import tempfile
import threading
import logging

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

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

# ── Model singleton ───────────────────────────────────────────────────────────

def _get_local_config():
    from drone_detection_v2 import Config
    cfg = Config()
    cfg.DRIVE_MODELS = _THIS_DIR / "models"
    cfg.DRIVE_ROOT   = _THIS_DIR
    cfg.DRIVE_LOGS   = _THIS_DIR / "logs"
    return cfg

def get_v2_model():
    try:
        from drone_detection_v2 import load_best_model
        _v2_config = _get_local_config()
        _v2_model  = load_best_model(_v2_config)
        return _v2_model, _v2_config
    except Exception as e:
        log.error(f"Model load failed: {e}")
        raise

# ── Realtime session registry ─────────────────────────────────────────────────
_realtime_sessions = {}
_sessions_lock     = threading.Lock()

def _get_session(session_id='default'):
    with _sessions_lock:
        return _realtime_sessions.get(session_id)

def _register_session(session_id, session):
    with _sessions_lock:
        _realtime_sessions[session_id] = session

def _remove_session(session_id):
    with _sessions_lock:
        _realtime_sessions.pop(session_id, None)

# ── Upload helpers ────────────────────────────────────────────────────────────
ALLOWED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def _save_upload(file_obj, suffix=".wav") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_obj.save(tf.name)
    return tf.name

def _cleanup(*paths):
    for p in paths:
        try: os.unlink(p)
        except: pass

def _file_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Existing REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/v2/version", methods=["GET"])
def version():
    return jsonify({
        "version": "4.1",
        "fixes_applied": [
            "FIX-1: fractional sinc delay synthesis",
            "FIX-2: cap-hit guard + residual_threshold=1e-8",
            "FIX-3: NaN-safe GCC-PHAT",
            "FIX-4/5: error chart always populated",
            "FIX-6: segment hop clamped for short files",
            "FIX-7: NaN band skip in multi-drone",
            "FIX-8: synthetic fallback for noise test",
            "FIX-9: real-time simulated + live mic sessions",
        ],
        "timestamp": time.time(),
    })


@app.route("/api/v2/status", methods=["GET"])
def status():
    try:
        m, cfg = get_v2_model()
        model_ok = m is not None
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    with _sessions_lock:
        active = {sid: {'mode': getattr(s, '_mode', '?'), 'running': s.running}
                  for sid, s in _realtime_sessions.items() if s.running}

    return jsonify({
        "status":          "ok",
        "model_loaded":    model_ok,
        "device":          str(getattr(cfg, "DEVICE", "unknown")),
        "mic_positions":   getattr(cfg, "MIC_POSITIONS", []).tolist()
                           if hasattr(getattr(cfg, "MIC_POSITIONS", None), "tolist") else [],
        "sample_rate":     getattr(cfg, "SR", 22050),
        "active_sessions": active,
        "timestamp":       time.time(),
    })


@app.route("/api/v2/detect", methods=["POST"])
def detect_single():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if _file_ext(f.filename) not in ALLOWED_AUDIO:
        return jsonify({"error": f"Unsupported format: {f.filename}"}), 400

    drone_x      = float(request.form.get("drone_x",    1.0))
    drone_y      = float(request.form.get("drone_y",    0.8))
    threshold    = float(request.form.get("threshold",  0.70))
    n_segments   = int(  request.form.get("n_segments", 5))
    force_detect = request.form.get("force_detect", "false").lower() == "true"

    tmp = _save_upload(f, suffix=_file_ext(f.filename) or ".wav")
    try:
        _, cfg = get_v2_model()
        from drone_detection_v2 import detect_from_single_audio
        result = detect_from_single_audio(tmp, cfg, drone_pos=[drone_x, drone_y],
            threshold=threshold, n_segments=n_segments,
            force_detect=force_detect, show_plot=False)

        def _clean(obj):
            if hasattr(obj, "tolist"): return obj.tolist()
            if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [_clean(x) for x in obj]
            return obj

        resp = _clean({
            "detected":    result["detected"],
            "probability": result["probability"],
            "position":    result["position"],
            "summary":     result["detection_summary"],
            "n_tracks":    len(result.get("active_tracks", [])),
        })
        if result["detected"]:
            socketio.emit("drone_detected_v2", {"timestamp": time.time(),
                "confidence": result["probability"], "position": resp["position"],
                "source": f.filename})
        return jsonify(resp)
    except Exception as e:
        log.exception("detect_single failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(tmp)


@app.route("/api/v2/detect-3mic", methods=["POST"])
def detect_3mic():
    for key in ("mic1", "mic2", "mic3"):
        if key not in request.files:
            return jsonify({"error": f"Missing file: {key}"}), 400

    threshold = float(request.form.get("threshold", 0.70))
    hint_x    = request.form.get("hint_x")
    hint_y    = request.form.get("hint_y")
    hint_pos  = [float(hint_x), float(hint_y)] if hint_x and hint_y else None
    tmps = []
    try:
        _, cfg = get_v2_model()
        for key in ("mic1", "mic2", "mic3"):
            tmps.append(_save_upload(request.files[key],
                        suffix=_file_ext(request.files[key].filename) or ".wav"))

        from drone_detection_v2 import AudioProcessor, localize_precise
        import torch, torch.nn.functional as F

        ap  = AudioProcessor(cfg)
        mel = ap.prepare_3channel_mels(*tmps)
        from drone_detection_v2 import model as m
        with torch.no_grad():
            prob = F.softmax(m(mel), dim=1)[0, 1].item()

        detected = prob >= threshold
        loc      = localize_precise(*tmps, config=cfg, hint_pos=hint_pos) if detected else None

        def _s(v): return v.tolist() if hasattr(v, "tolist") else v

        resp = {"detected": detected, "probability": prob, "localization": None}
        if loc:
            resp["localization"] = {
                "position":          _s(loc["position"]),
                "reliable":          loc["reliable"],
                "quality_message":   loc["quality_message"],
                "confidence_radius": float(loc["confidence_radius"]),
                "measured_tdoas_ms": [float(t*1000) for t in loc["measured_tdoas"]],
                "estimated_tdoas_ms":[float(t*1000) for t in loc["estimated_tdoas"]],
            }
            socketio.emit("drone_detected_v2", {"timestamp": time.time(),
                "confidence": prob, "position": _s(loc["position"]), "source": "3-mic"})
        return jsonify(resp)
    except Exception as e:
        log.exception("detect_3mic failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(*tmps)


@app.route("/api/v2/detect-multi", methods=["POST"])
def detect_multi():
    for key in ("mic1", "mic2", "mic3"):
        if key not in request.files:
            return jsonify({"error": f"Missing file: {key}"}), 400

    threshold  = float(request.form.get("threshold",  0.70))
    max_drones = int(  request.form.get("max_drones", 3))
    tmps = []
    try:
        _, cfg = get_v2_model()
        for key in ("mic1", "mic2", "mic3"):
            tmps.append(_save_upload(request.files[key],
                        suffix=_file_ext(request.files[key].filename) or ".wav"))

        from drone_detection_v2 import detect_and_localize_multi_drone
        result = detect_and_localize_multi_drone(*tmps, config=cfg,
            threshold=threshold, max_drones=max_drones)

        def _s(v): return v.tolist() if hasattr(v, "tolist") else v

        drones_out = [{"id": d["id"], "position": _s(d["position"]),
                       "reliable": d.get("error", 1) <= 1e-8,
                       "confidence_radius": float(d["confidence_radius"]),
                       "band_hz": list(d["band"]),
                       "tdoa_strength": float(d["tdoa_strength"])}
                      for d in result.get("drones", [])]

        resp = {"detected": result["detected"], "n_drones": result["n_drones"],
                "probability": float(result["probability"]), "drones": drones_out}
        if result["detected"] and drones_out:
            socketio.emit("drone_detected_v2", {"timestamp": time.time(),
                "confidence": float(result["probability"]), "n_drones": result["n_drones"],
                "drones": drones_out, "source": "multi-drone"})
        return jsonify(resp)
    except Exception as e:
        log.exception("detect_multi failed")
        return jsonify({"error": str(e)}), 500
    finally:
        _cleanup(*tmps)


@app.route("/api/v2/noise-test", methods=["POST"])
def noise_test():
    snr_min  = int(request.form.get("snr_min",  -5))
    snr_max  = int(request.form.get("snr_max",  20))
    snr_step = int(request.form.get("snr_step",  5))
    n_clips  = int(request.form.get("n_clips",  20))
    snr_levels = list(range(snr_min, snr_max + 1, snr_step))
    try:
        _, cfg = get_v2_model()
        from drone_detection_v2 import run_noise_robustness_test
        results = run_noise_robustness_test(cfg, snr_levels=snr_levels, n_clips=n_clips)

        def _s(obj):
            if hasattr(obj, "tolist"): return obj.tolist()
            if isinstance(obj, dict):  return {k: _s(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_s(x) for x in obj]
            if isinstance(obj, float): return round(obj, 4)
            return obj

        return jsonify({"status": "ok", "snr_levels": snr_levels, "results": _s(results)})
    except Exception as e:
        log.exception("noise_test failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v2/path-simulate", methods=["POST"])
def path_simulate():
    n_waypoints = int(  request.form.get("n_waypoints", 8))
    spread      = float(request.form.get("spread",      2.0))
    try:
        _, cfg = get_v2_model()
        from drone_detection_v2 import simulate_path_tracking_from_dataset
        tracker = simulate_path_tracking_from_dataset(cfg, n_positions=n_waypoints, spread=spread)

        def _s(v): return v.tolist() if hasattr(v, "tolist") else v

        tracks_out = []
        for t in tracker.tracks:
            if t.hits >= cfg.TRACKER_MIN_HITS:
                speed = t.speed() if callable(t.speed) else t.speed
                tracks_out.append({
                    "id": t.track_id, "waypoints": len(t.positions),
                    "positions": [_s(p) for p in t.positions],
                    "speed_m_s": float(speed) if speed is not None else None,
                })

        return jsonify({"status": "ok", "n_waypoints": n_waypoints, "spread": spread,
                        "n_tracks": len(tracks_out), "tracks": tracks_out})
    except Exception as e:
        log.exception("path_simulate failed")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# REALTIME endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/v2/realtime/start", methods=["POST"])
def realtime_start():
    """
    Start a real-time detection session.

    Form fields
    -----------
    mode            : 'simulated' | 'real'   (default: 'simulated')
    session_id      : string key             (default: 'default')
    threshold       : 0.1–0.99              (default: 0.70)

    Simulated only:
      n_drones      : 1–3                   (default: 1)
      pattern       : circle|figure8|linear|random|multi   (default: circle)
      tick_rate     : frames/sec            (default: 1.0)
      noise_level   : 0.001–0.2             (default: 0.04)
      spread        : max metres from array (default: 1.5)

    Real only:
      segment_dur   : seconds per window    (default: 3.0)
      device_indices: comma-separated ints  (default: '')
    """
    mode       = request.form.get("mode",       "simulated")
    session_id = request.form.get("session_id", "default")
    threshold  = float(request.form.get("threshold", 0.70))

    existing = _get_session(session_id)
    if existing and existing.running:
        existing.stop()
        _remove_session(session_id)

    try:
        _, cfg = get_v2_model()
    except Exception as e:
        return jsonify({"error": f"Model not ready: {e}"}), 500

    from realtime_sessions import SimulatedRealtimeSession, RealRealtimeSession

    if mode == "simulated":
        n_drones    = int(  request.form.get("n_drones",    1))
        raw_pattern = request.form.get("pattern", "circle")
        tick_rate   = float(request.form.get("tick_rate",   1.0))
        noise_level = float(request.form.get("noise_level", 0.04))
        spread      = float(request.form.get("spread",      1.5))

        if raw_pattern == "multi":
            patterns = ['circle', 'figure8', 'random'][:n_drones]
        else:
            patterns = [raw_pattern] * n_drones

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

        session = RealRealtimeSession(cfg, socketio, threshold=threshold,
            segment_dur=segment_dur, device_indices=dev_indices)
    else:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    session._mode = mode
    _register_session(session_id, session)

    if not session.start():
        _remove_session(session_id)
        return jsonify({"error": "Session failed to start — check server logs"}), 500

    log.info(f"realtime/start: mode={mode} session={session_id}")
    socketio.emit('realtime_status', {'running': True, 'mode': mode,
                                      'error': None, 'session_id': session_id})
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
    return jsonify({"running": session.running, "session_id": session_id,
                    "mode": getattr(session, '_mode', '?'),
                    "stats": session.get_stats()})


@app.route("/api/v2/realtime/audio-devices", methods=["GET"])
def list_audio_devices():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({'index': i, 'name': info['name'],
                                 'channels': info['maxInputChannels'],
                                 'rate': int(info['defaultSampleRate'])})
        pa.terminate()
        return jsonify({"devices": devices})
    except ImportError:
        return jsonify({"devices": [], "error": "PyAudio not installed"})
    except Exception as e:
        return jsonify({"devices": [], "error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Frontend + WebSocket
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/v2")
@app.route("/v2/")
def index_v2():
    return render_template("index_v2.html")

@app.route("/static/v2/<path:filename>")
def static_v2(filename):
    return send_from_directory("static", filename)

@socketio.on("connect", namespace="/v2")
def ws_connect():
    log.info("WebSocket client connected (v2 namespace)")
    emit("status_v2", {"message": "Connected to Drone Detection v2", "version": "4.1"})

@socketio.on("disconnect", namespace="/v2")
def ws_disconnect():
    log.info("WebSocket client disconnected (v2 namespace)")

@socketio.on("ping_v2", namespace="/v2")
def ws_ping(data):
    emit("pong_v2", {"timestamp": time.time()})

@app.errorhandler(413)
def too_large(e): return jsonify({"error": "File too large (max 50 MB)"}), 413

@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error", "detail": str(e)}), 500