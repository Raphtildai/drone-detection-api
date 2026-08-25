# app_v2.py
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

# ── Thread-pool limits (must run before numpy/scipy is imported) ───────────
# Shared cPanel hosting caps total processes/threads per account
# (RLIMIT_NPROC). OpenBLAS sizes its thread pool to the host's CPU count
# (e.g. 12) by default; with several Passenger workers each spawning that
# many threads, the account exhausts its process quota and pthread_create()
# fails with "Resource temporarily unavailable".
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# ── Force UTF-8 stdout/stderr ───────────────────────────────────────────────
# Passenger's stdout on this host defaults to ASCII. Code throughout this
# codebase prints emoji status markers (✅/❌); under ASCII encoding those
# print() calls raise UnicodeEncodeError, which callers can mistake for the
# operation itself having failed (e.g. a model that loaded fine gets
# reported as "load failed" because the success print() blew up).
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Path bootstrap ─────────────────────────────────────────────────────────
from pathlib import Path
import hashlib
import csv
import re
from urllib.parse import urlparse, parse_qs, urljoin

_THIS_DIR  = Path(__file__).parent.resolve()
_REPO_ROOT = _THIS_DIR.parent.parent.resolve()

for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load deployment/v2/.env into os.environ *before* drone_detection is
# imported below — Config() reads NEXTCLOUD_BASE_URL / NEXTCLOUD_SHARE_TOKEN
# from os.environ at import time (the `config` singleton in
# drone_detection/config.py is constructed on import). No-op if the file
# doesn't exist (e.g. cPanel, which injects env vars directly instead).
from dotenv import load_dotenv
load_dotenv(_THIS_DIR / ".env")

import json
import logging
import tempfile
import threading
import time

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from drone_detection.repository_loader import (
    list_dunakeszi_segments_rich,
    stream_segment_by_gt_id,
    get_dunakeszi_file_browser,
)

import history_store

# NOTE: _get_config() is defined below; _repo_cfg is initialised after it.

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

_RESULTS_LOG = _THIS_DIR / "logs" / "detection_results.jsonl"
_RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
_results_log_lock = threading.Lock()

def _append_result_log(record: dict) -> None:
    """Append one JSON line to the private results log. Never raises."""
    try:
        with _results_log_lock:
            with open(_RESULTS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        log.exception("Failed to append result log")

# ── Config / model singleton ──────────────────────────────────────────────────

def _parse_nextcloud_share_url(url: str):
    """
    Parse a Nextcloud public-share URL like:
        https://nc.example.com/index.php/s/<TOKEN>?dir=/some/path
    Returns (base_url, token, dir_path) or None if it isn't a share link.
    """
    try:
        parsed = urlparse(url)
        m = re.search(r"/s/([A-Za-z0-9]+)", parsed.path)
        if not m:
            return None
        token    = m.group(1)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        dir_path = parse_qs(parsed.query).get("dir", [None])[0]
        return base_url, token, dir_path
    except Exception:
        return None
    
def _get_config():
    """Return a Config pointed at this deployment's models directory."""
    from drone_detection import Config
    cfg = Config()
    cfg.DRIVE_MODELS = _THIS_DIR / "models"
    cfg.DRIVE_ROOT   = _THIS_DIR
    cfg.DRIVE_LOGS   = _THIS_DIR / "logs"
    return cfg


# Singleton config used by the repository endpoints.  Built once after
# _get_config() is defined so DRIVE_MODELS points at deployment/v2/models/.
_repo_cfg = _get_config()


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


def _assert_public_url(url: str) -> None:
    """
    Reject URLs that resolve to private/loopback/link-local addresses.

    This app fetches user-supplied URLs server-side (Single File "paste a
    link" input). Without this check, a request for e.g.
    http://169.254.169.254/computeMetadata/v1/instance/service-accounts/
    default/token would let anyone steal this Cloud Run service's
    credentials — a classic SSRF-to-metadata-server attack. Every hostname
    the URL resolves to must be a public address before we let `requests`
    touch it.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must be http:// or https://")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    try:
        addrs = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc

    for family, _, _, _, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise ValueError(
                f"URL resolves to a non-public address ({ip}) — not allowed"
            )


def _save_from_url(url: str, max_bytes: int) -> str:
    """
    Download a user-supplied URL to a temp file for detection, same
    contract as _save_upload(). Streams with a hard byte cap so a huge or
    slow-loris remote file can't exhaust memory/time — mirrors the
    MAX_CONTENT_LENGTH cap already enforced on direct uploads.
    """
    import requests

    ext = _file_ext(url.split("?")[0].split("#")[0]) or ".wav"
    if ext not in ALLOWED_AUDIO:
        ext = ".wav"  # content-type below decides whether this actually works

    # Validate + follow redirects manually (max 5 hops) rather than
    # requests' allow_redirects=True, which would only validate the
    # original URL — a public URL that 302s to an internal address would
    # otherwise sail straight past _assert_public_url().
    next_url = url
    for _ in range(5):
        _assert_public_url(next_url)
        resp = requests.get(next_url, stream=True, timeout=30, allow_redirects=False)
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise ValueError("Redirect response missing Location header")
            next_url = urljoin(next_url, location)
            continue
        break
    else:
        raise ValueError("Too many redirects")

    resp.raise_for_status()

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    written = 0
    try:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            written += len(chunk)
            if written > max_bytes:
                raise ValueError(f"Remote file exceeds {max_bytes // (1024*1024)}MB limit")
            tf.write(chunk)
    except Exception:
        tf.close()
        os.unlink(tf.name)
        raise
    tf.close()
    return tf.name


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


@app.route("/api/v2/history", methods=["GET"])
def history_list():
    """
    List saved detection history (Single File / 3-Mic / Multi-Drone),
    most recent first.

    Query params
    ------------
    type  : filter to one detection type ("single" | "3mic" | "multi")
    limit : max entries to return (default 50, capped at 200)
    """
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    detection_type = request.args.get("type") or None
    try:
        items = history_store.list_history(limit=limit, detection_type=detection_type)
        return jsonify({"items": items})
    except Exception as exc:
        log.exception("history_list failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v2/history/<record_id>", methods=["GET"])
def history_get(record_id):
    """Full record for one history entry, for replaying it in the UI."""
    try:
        record = history_store.get_detection(record_id)
    except Exception as exc:
        log.exception("history_get failed")
        return jsonify({"error": str(exc)}), 500
    if record is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(record)


@app.route("/api/v2/detect", methods=["POST"])
def detect_single():
    """
    Single audio file detection.

    Slides a fixed-length window (cfg.TARGET_DURATION, no overlap) across the
    whole uploaded file and returns a timeline of per-window results, same
    shape as the repository batch-scan endpoint — so long recordings get
    more than just a single verdict from their first few seconds.

    Form fields
    -----------
    file        : audio file — required unless `url` is given
    url         : direct link to an audio/video file, fetched server-side —
                  alternative to `file` (not both). Must resolve to a public
                  address; private/loopback/link-local targets are rejected.
    drone_x     : hint X position metres (default 1.0)
    drone_y     : hint Y position metres (default 0.8)
    threshold   : detection threshold 0.1–0.99 (default cfg.DETECTION_THRESHOLD)
    n_segments  : max number of windows to scan (default 20, capped at 150)
    force_detect: skip classifier, always localise (default false)
    """
    url = request.form.get("url", "").strip()
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        if _file_ext(f.filename) not in ALLOWED_AUDIO:
            return jsonify({"error": f"Unsupported format: {f.filename}"}), 400
        source_label = f.filename
        get_tmp = lambda: _save_upload(f, suffix=_file_ext(f.filename) or ".wav")
    elif url:
        source_label = url
        get_tmp = lambda: _save_from_url(url, app.config["MAX_CONTENT_LENGTH"])
    else:
        return jsonify({"error": "No file uploaded and no url given"}), 400

    MAX_WINDOWS  = 150
    max_windows  = min(int(request.form.get("n_segments", 20)), MAX_WINDOWS)
    force_detect = request.form.get("force_detect", "false").lower() == "true"

    try:
        tmp = get_tmp()
    except Exception as exc:
        return jsonify({"error": f"Could not fetch audio: {exc}"}), 400

    try:
        m, cfg = get_model()
        threshold = float(request.form.get("threshold", cfg.DETECTION_THRESHOLD))

        from drone_detection import AudioProcessor, detect, localize

        ap = AudioProcessor(cfg)
        y_full = ap.load(tmp)
        total_duration = len(y_full) / cfg.SR

        window_s = cfg.TARGET_DURATION
        hop_s    = window_s
        win_n    = int(round(window_s * cfg.SR))
        hop_n    = int(round(hop_s * cfg.SR))

        n_windows_available = max(int(len(y_full) // hop_n), 1)
        n_windows = min(n_windows_available, max_windows)
        truncated = n_windows_available > max_windows

        results = []
        for i in range(n_windows):
            off0, off1 = i * hop_n, i * hop_n + win_n
            seg = ap.pad_or_truncate(y_full[off0:off1])
            channels = [seg, seg, seg]

            det      = detect(channels, cfg)
            prob     = float(det["probability"])
            detected = force_detect or (prob >= threshold)
            loc = localize(channels, cfg) if detected else None

            results.append({
                "t_start": round(i * hop_s, 3),
                "t_end":   round(i * hop_s + window_s, 3),
                "detected": detected,
                "probability": prob,
                "cnn_probability": float(det.get("cnn_probability", prob)),
                "heuristic_probability": float(det.get("heuristic_probability", 0.0)),
                "position": loc["xy_position"] if loc else None,
                "azimuth_deg": loc["azimuth_deg"] if loc else None,
                "distance_m": loc["distance_m"] if loc else None,
                "height_m": loc["height_m"] if loc else None,
            })

        detected_count = sum(1 for r in results if r["detected"])
        peak = max(results, key=lambda r: r["probability"])
        avg_probability = float(sum(r["probability"] for r in results) / len(results))

        resp = _clean({
            "detected":              detected_count > 0,
            "probability":           peak["probability"],
            "cnn_probability":       peak["cnn_probability"],
            "heuristic_probability": peak["heuristic_probability"],
            "position":              peak["position"],
            "azimuth_deg":           peak["azimuth_deg"],
            "distance_m":            peak["distance_m"],
            "height_m":              peak["height_m"],
            "duration_s":            round(total_duration, 3),
            "window_s":              window_s,
            "hop_s":                 hop_s,
            "n_windows":             len(results),
            "truncated":             truncated,
            "detected_count":        detected_count,
            "detection_rate":        round(100.0 * detected_count / len(results), 1),
            "avg_probability":       round(avg_probability, 3),
            "detection_summary": {
                "total_segments":    len(results),
                "detected_segments": detected_count,
                "max_confidence":    peak["probability"],
            },
            "n_tracks": 0,
            "results": results,
        })

        _append_result_log({
            "timestamp":      time.time(),
            "filename_hash":  hashlib.sha256(source_label.encode()).hexdigest()[:12],
            "detected":       resp["detected"],
            "probability":    resp["probability"],
            "position":       resp.get("position"),
        })
        history_store.log_detection("single", {
            "source":      source_label,
            "detected":    resp["detected"],
            "probability": resp["probability"],
            "position":    resp.get("position"),
        }, resp)

        if resp["detected"]:
            socketio.emit("drone_detected_v2", {
                "timestamp":  time.time(),
                "confidence": resp["probability"],
                "position":   resp.get("position"),
                "source":     source_label,
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

        source_label = ", ".join(request.files[k].filename for k in ("mic1", "mic2", "mic3"))
        history_store.log_detection("3mic", {
            "source":      source_label,
            "detected":    resp["detected"],
            "probability": resp["probability"],
            "position":    resp.get("localization", {}).get("position") if resp.get("localization") else None,
        }, resp)

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

        source_label = ", ".join(request.files[k].filename for k in ("mic1", "mic2", "mic3"))
        history_store.log_detection("multi", {
            "source":      source_label,
            "detected":    resp["detected"],
            "probability": resp["probability"],
            "position":    drones_out[0]["position"] if drones_out else None,
        }, resp)

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
        dataset_type   = request.form.get("dataset_type", "uavirbase")
        array          = request.form.get("array",         "BK-6-E")
        raw_url        = request.form.get("url",           None) or None
        required_split = request.form.get("required_split",None) or None
        tick_rate      = float(request.form.get("tick_rate",    1.0))
        max_dist       = float(request.form.get("max_dist",    100.0))
        allow_fallback = request.form.get("allow_synthetic_fallback", "true").lower() == "true"

        # Use the shared singleton so Nextcloud credentials set here persist
        # across requests and match /api/v2/repository/* endpoints.
        cfg = _repo_cfg
        threshold = float(request.form.get("threshold", cfg.DETECTION_THRESHOLD))

        remotezip_url = raw_url
        if raw_url:
            nc = _parse_nextcloud_share_url(raw_url)
            if nc:
                base_url, token, dir_path = nc
                nc_ready = cfg.reload_nextcloud_env(base_url=base_url, share_token=token)
                if dir_path:
                    if dataset_type == "mems":
                        cfg.NEXTCLOUD_MEMS_PATH = dir_path
                    else:
                        cfg.NEXTCLOUD_POLYWAV_PATH = dir_path
                log.info("Parsed Nextcloud share URL: base=%s dir=%s dataset=%s nc_ready=%s",
                        base_url, dir_path, dataset_type, nc_ready)
                remotezip_url = None  # it's a share link, not a raw downloadable zip
            else:
                cfg.reload_nextcloud_env()  # pick up env vars set since startup
        else:
            cfg.reload_nextcloud_env()

        session = RepositoryRealtimeSession(
            cfg, socketio,
            url                      = remotezip_url,
            dataset_type             = dataset_type,
            array                    = array,
            max_dist                 = max_dist,
            tick_rate                = tick_rate,
            threshold                = threshold,
            allow_download           = False,
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
    
@app.route("/api/v2/repository/segments", methods=["GET"])
def repo_segments():
    split   = request.args.get("split")   or None
    session = request.args.get("session") or None
    # Refresh Nextcloud credentials from env in case they were set after startup
    _repo_cfg.reload_nextcloud_env()
    return jsonify(list_dunakeszi_segments_rich(_repo_cfg, split, session))


@app.route("/api/v2/repository/start-segment", methods=["POST"])
def repo_start_segment():
    seg_id     = int(  request.form.get("segment_id", 0))
    array      = request.form.get("array",           "BK-6-E")
    session_id = request.form.get("session_id",      "default")
    tick_rate  = float(request.form.get("tick_rate",  1.0))
    threshold  = float(request.form.get("threshold",  0.70))

    nc_url   = request.form.get("nextcloud_url")   or None
    nc_token = request.form.get("nextcloud_token") or None
    nc_ready = _repo_cfg.reload_nextcloud_env(base_url=nc_url, share_token=nc_token)
    log.info("repo_start_segment: seg=%d nc_ready=%s", seg_id, nc_ready)

    existing = _get_session(session_id)
    if existing and existing.running:
        existing.stop()
        _remove_session(session_id)

    # Keep this only to return metadata in the immediate JSON response —
    # the actual streaming no longer depends on it.
    preview = stream_segment_by_gt_id(seg_id, _repo_cfg, array=array)
    if preview is None:
        return jsonify({"error": f"Segment {seg_id} not available"}), 404
    _, label = preview

    from realtime_sessions import RepositoryRealtimeSession

    session = RepositoryRealtimeSession(
        _repo_cfg, socketio,
        dataset_type             = "dunakeszi",
        array                    = array,
        tick_rate                = tick_rate,
        threshold                = threshold,
        allow_download           = False,
        allow_synthetic_fallback = False,
        segment_id               = seg_id,   # ← this is what actually plays a single segment
    )
    session._mode = "repository"
    _register_session(session_id, session)

    if not session.start():
        _remove_session(session_id)
        return jsonify({"error": "Session failed to start — check server logs"}), 500

    log.info("repo_start_segment: seg=%d session=%s", seg_id, session_id)
    socketio.emit("realtime_status", {
        "running": True, "mode": "repository", "error": None, "session_id": session_id,
    })
    return jsonify({"ok": True, "segment_id": seg_id, "label": label, "session_id": session_id})


@app.route("/api/v2/repository/files/<file_type>", methods=["GET"])
def repo_files(file_type):
    _repo_cfg.reload_nextcloud_env()
    return jsonify(get_dunakeszi_file_browser(_repo_cfg, file_type))


@app.route("/api/v2/repository/files-list", methods=["GET"])
def repo_files_list():
    """
    List available audio filenames for the manual-window picker.

    Query params
    ------------
    dataset_type : "polywav" | "mems"   (default "polywav")
    url          : optional Nextcloud share URL — configures credentials
                   on the fly, same parsing as /api/v2/realtime/start
    """
    dataset_type = request.args.get("dataset_type", "polywav")
    raw_url = request.args.get("url") or None

    if raw_url:
        nc = _parse_nextcloud_share_url(raw_url)
        if nc:
            base_url, token, dir_path = nc
            _repo_cfg.reload_nextcloud_env(base_url=base_url, share_token=token)
            if dir_path:
                if dataset_type == "mems":
                    _repo_cfg.NEXTCLOUD_MEMS_PATH = dir_path
                else:
                    _repo_cfg.NEXTCLOUD_POLYWAV_PATH = dir_path
            log.info("files-list: parsed share URL, base=%s dir=%s dataset=%s",
                      base_url, dir_path, dataset_type)
        else:
            _repo_cfg.reload_nextcloud_env()
    else:
        _repo_cfg.reload_nextcloud_env()

    if not (_repo_cfg.NEXTCLOUD_BASE_URL and _repo_cfg.NEXTCLOUD_SHARE_TOKEN):
        return jsonify({
            "files": [],
            "error": "Nextcloud not configured — paste a share URL and try again",
        })

    from drone_detection.dunakeszi_nextcloud import list_polywav_files, list_mems_files
    try:
        files = list_mems_files(_repo_cfg) if dataset_type == "mems" else list_polywav_files(_repo_cfg)
        return jsonify({
            "files": sorted(f["name"] for f in files),
            "dataset_type": dataset_type,
        })
    except Exception as exc:
        log.exception("files-list failed")
        return jsonify({"files": [], "error": str(exc)})


@app.route("/api/v2/repository/manual-window", methods=["POST"])
def repo_manual_window():
    """
    Run detection (+localization for polywav) on an explicit file + time window.

    Form fields
    -----------
    filename     : e.g. "251020VITEMOROM1AT01G.wav" or "Audio 03_01.wav"
    start_s      : offset in seconds within that file
    duration_s   : window length (default 3.0, capped at 30.0)
    array        : "BK-6-E" | "BK-6-W"   (polywav only)
    dataset_type : "polywav" | "mems"    (default "polywav")
    url          : optional Nextcloud share URL to configure credentials
    threshold    : override detection threshold
    """
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    start_s      = float(request.form.get("start_s", 0))
    duration_s   = min(float(request.form.get("duration_s", 3.0)), 30.0)
    array        = request.form.get("array", "BK-6-E")
    dataset_type = request.form.get("dataset_type", "polywav")
    raw_url      = request.form.get("url") or None

    if raw_url:
        nc = _parse_nextcloud_share_url(raw_url)
        if nc:
            base_url, token, dir_path = nc
            _repo_cfg.reload_nextcloud_env(base_url=base_url, share_token=token)
            if dir_path:
                if dataset_type == "mems":
                    _repo_cfg.NEXTCLOUD_MEMS_PATH = dir_path
                else:
                    _repo_cfg.NEXTCLOUD_POLYWAV_PATH = dir_path
        else:
            _repo_cfg.reload_nextcloud_env()
    else:
        _repo_cfg.reload_nextcloud_env()

    threshold = float(request.form.get("threshold", _repo_cfg.DETECTION_THRESHOLD))

    from drone_detection import AudioProcessor, detect, localize
    ap = AudioProcessor(_repo_cfg)

    # ── Switch config to the correct physical mic array before localizing ──
    # _repo_cfg is a shared singleton (see _repo_cfg = _get_config() above),
    # reused by RepositoryRealtimeSession too. It does NOT reset itself
    # between requests, so we must set geometry explicitly on every call
    # rather than trust whatever the last request happened to leave behind.
    #   polywav (BK-6-E/BK-6-W) → GP2 equilateral 2.5 m Brüel triangle
    #   mems                    → not localized (single channel), left as-is
    _MANUAL_GEOMETRY = {"polywav": "gp2", "mems": "uavirbase"}
    geom = _MANUAL_GEOMETRY.get(dataset_type, "gp2")
    try:
        _repo_cfg.set_array_geometry(geom)
    except Exception as exc:
        log.warning("manual_window: could not set array geometry '%s': %s", geom, exc)

    try:
        if dataset_type == "mems":
            from drone_detection.dunakeszi_nextcloud import read_mems_window
            mems_path = f"{_repo_cfg.NEXTCLOUD_MEMS_PATH}/{filename}"
            audio_mc, native_sr = read_mems_window(_repo_cfg, mems_path, start_s, duration_s)
            mono = audio_mc.mean(axis=0).astype("float32")
            if native_sr != _repo_cfg.SR:
                import librosa
                mono = librosa.resample(mono, orig_sr=native_sr, target_sr=_repo_cfg.SR)
            y = ap.pad_or_truncate(mono)
            out_channels = [y, y, y]
        else:
            from drone_detection.dunakeszi_nextcloud import (
                read_polywav_window, BK6E_CHANNELS, BK6W_CHANNELS,
            )
            ch_map = {"BK-6-E": BK6E_CHANNELS, "BK-6-W": BK6W_CHANNELS}
            channel_indices = ch_map.get(array, BK6E_CHANNELS)
            pw_path = f"{_repo_cfg.NEXTCLOUD_POLYWAV_PATH}/{filename}"
            audio_raw, native_sr = read_polywav_window(
                _repo_cfg, pw_path, start_s, duration_s, channel_indices,
            )
            out_channels = []
            for ch_row in audio_raw:
                if native_sr != _repo_cfg.SR:
                    import librosa
                    ch_row = librosa.resample(ch_row, orig_sr=native_sr, target_sr=_repo_cfg.SR)
                out_channels.append(ap.pad_or_truncate(ch_row))
            while len(out_channels) < 3:
                out_channels.append(out_channels[-1].copy())
            out_channels = out_channels[:3]
    except Exception as exc:
        log.exception("manual_window read failed")
        return jsonify({"error": f"Failed to read window: {exc}"}), 500

    det      = detect(out_channels, _repo_cfg)
    prob     = float(det["probability"])
    detected = prob >= threshold
    # MEMS is single-channel — no real TDOA, so skip localize()
    loc = localize(out_channels, _repo_cfg) if (detected and dataset_type != "mems") else None

    return jsonify({
        "detected": detected,
        "probability": prob,
        "cnn_probability": float(det.get("cnn_probability", prob)),
        "heuristic_probability": float(det.get("heuristic_probability", 0.0)),
        "position": loc["xy_position"].tolist() if loc else None,
        "azimuth_deg": loc["azimuth_deg"] if loc else None,
        "distance_m": loc["distance_m"] if loc else None,
        "height_m": loc["height_m"] if loc else None,
        "filename": filename, "start_s": start_s,
        "duration_s": duration_s, "array": array,
        "dataset_type": dataset_type,
    })

@app.route("/api/v2/repository/batch-scan", methods=["POST"])
def repo_batch_scan():
    """
    Slide a detection window across a stretch of one file and return a
    timeline of results.

    Reads happen in fixed-size chunks (_CHUNK_S seconds each) rather than one
    contiguous HTTP range-read for the whole scan span — the original
    single-read approach made peak memory scale with scan_duration_s, which
    was hitting Cloud Run's container memory limit (visible client-side as
    an empty response / "Unexpected end of JSON input") on anything much
    past ~60s. Chunking keeps peak memory roughly constant regardless of how
    long the scan is, which is also what makes full_file=true (scan to the
    end of the recording, ignoring scan_duration_s) practical.

    Form fields
    -----------
    filename        : e.g. "251020VITEMOROM1AT01P.wav"
    start_s         : start of the scan range (default 0)
    scan_duration_s : total range to scan, seconds (default 60, capped at 600)
                       — ignored if full_file=true
    full_file       : "true" to scan from start_s to end of file, chunk by
                       chunk, stopping when a read comes back short (EOF)
                       instead of at a fixed duration
    window_s        : detection window length (default 3.0)
    hop_s           : step between window starts (default = window_s → no overlap)
    array           : "BK-6-E" | "BK-6-W"   (polywav only)
    dataset_type    : "polywav" | "mems"    (default "polywav")
    url             : optional Nextcloud share URL
    threshold       : detection threshold override
    """
    import numpy as np
    import librosa

    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    start_s      = float(request.form.get("start_s", 0))
    window_s     = float(request.form.get("window_s", 3.0))
    hop_s        = float(request.form.get("hop_s", window_s))
    array        = request.form.get("array", "BK-6-E")
    dataset_type = request.form.get("dataset_type", "polywav")
    raw_url      = request.form.get("url") or None
    full_file    = request.form.get("full_file", "false").lower() == "true"
    request_id   = request.form.get("request_id") or None

    if window_s <= 0 or hop_s <= 0:
        return jsonify({"error": "window_s and hop_s must both be > 0"}), 400

    if full_file:
        MAX_WINDOWS = 3000   # generous safety cap, not an expected target
        scan_duration_s = None
    else:
        MAX_WINDOWS = 150
        scan_duration_s = min(float(request.form.get("scan_duration_s", 60.0)), 600.0)

    if raw_url:
        nc = _parse_nextcloud_share_url(raw_url)
        if nc:
            base_url, token, dir_path = nc
            _repo_cfg.reload_nextcloud_env(base_url=base_url, share_token=token)
            if dir_path:
                if dataset_type == "mems":
                    _repo_cfg.NEXTCLOUD_MEMS_PATH = dir_path
                else:
                    _repo_cfg.NEXTCLOUD_POLYWAV_PATH = dir_path
        else:
            _repo_cfg.reload_nextcloud_env()
    else:
        _repo_cfg.reload_nextcloud_env()

    threshold = float(request.form.get("threshold", _repo_cfg.DETECTION_THRESHOLD))

    from drone_detection import AudioProcessor, detect, localize
    ap = AudioProcessor(_repo_cfg)

    # Same geometry fix as the manual-window endpoint — _repo_cfg is a shared
    # singleton and does not reset itself between requests.
    _GEOMETRY = {"polywav": "gp2", "mems": "uavirbase"}
    geom = _GEOMETRY.get(dataset_type, "gp2")
    try:
        _repo_cfg.set_array_geometry(geom)
    except Exception as exc:
        log.warning("batch_scan: could not set array geometry '%s': %s", geom, exc)

    if dataset_type == "mems":
        from drone_detection.dunakeszi_nextcloud import read_mems_window
        mems_path = f"{_repo_cfg.NEXTCLOUD_MEMS_PATH}/{filename}"
        channel_indices = None
    else:
        from drone_detection.dunakeszi_nextcloud import (
            read_polywav_window, BK6E_CHANNELS, BK6W_CHANNELS,
        )
        ch_map = {"BK-6-E": BK6E_CHANNELS, "BK-6-W": BK6W_CHANNELS}
        channel_indices = ch_map.get(array, BK6E_CHANNELS)
        pw_path = f"{_repo_cfg.NEXTCLOUD_POLYWAV_PATH}/{filename}"

    # Ground-truth position lookup. Real GPX telemetry (the drone's actual
    # recorded flight) is tried first and is authoritative when available;
    # the idealized flight-plan interpolation is only a fallback for moments
    # the real GPX doesn't cover (network hiccup, or genuinely outside every
    # session's recorded span). Neither is guaranteed for a given window —
    # only the cataloged maneuvers have ground truth at all (~24% of the
    # full recording) — a None result there is expected, not a bug.
    try:
        from dunakeszi_ground_truth_fixed import ground_truth_xy_at, ground_truth_xy_at_telemetry
    except ImportError:
        ground_truth_xy_at = None
        ground_truth_xy_at_telemetry = None

    # Chunk size for each HTTP range-read — bounds peak memory regardless of
    # scan_duration_s / full_file. Padded by window_s so windows near the
    # tail of a chunk still have a full window's worth of data available.
    _CHUNK_S = 30.0

    # Wall-clock safety net: some recordings run 6-12 minutes, and a Cloud
    # Run request timeout kills the connection mid-response with no chance
    # to send a body (the "Unexpected end of JSON input" failure mode this
    # whole endpoint was rewritten to avoid). Stopping ourselves comfortably
    # before that deadline means the client always gets valid JSON — a
    # truncated result with an honest reason, never a dead connection.
    # Keep _MAX_SCAN_SECONDS well under the Cloud Run --timeout you deploy
    # with (e.g. --timeout 900 gives 300s of buffer for read/serialize/network).
    _MAX_SCAN_SECONDS = 600
    _scan_deadline = time.time() + _MAX_SCAN_SECONDS
    time_budget_exceeded = False

    results = []
    chunk_start = start_s
    stop = False
    try:
        while not stop and len(results) < MAX_WINDOWS:
            if time.time() >= _scan_deadline:
                time_budget_exceeded = True
                break
            if scan_duration_s is not None and chunk_start >= start_s + scan_duration_s:
                break
            chunk_dur = _CHUNK_S
            if scan_duration_s is not None:
                chunk_dur = min(chunk_dur, start_s + scan_duration_s - chunk_start)
            read_dur = chunk_dur + window_s   # pad tail for the last full window

            if dataset_type == "mems":
                audio_mc, native_sr = read_mems_window(_repo_cfg, mems_path, chunk_start, read_dur)
                full_mono     = audio_mc.mean(axis=0).astype("float32")
                full_channels = None
                n_available   = len(full_mono)
            else:
                full_channels, native_sr = read_polywav_window(
                    _repo_cfg, pw_path, chunk_start, read_dur, channel_indices,
                )   # shape (3, n_samples_native)
                full_mono   = None
                n_available = full_channels.shape[1]

            expected_available = int(round(read_dur * native_sr))
            if n_available < expected_available:
                stop = True   # server returned less than asked — end of file

            hop_native = int(round(hop_s    * native_sr))
            win_native = int(round(window_s * native_sr))

            i = 0
            while True:
                off0, off1 = i * hop_native, i * hop_native + win_native
                if off1 > n_available:
                    break   # ran out of data within this chunk
                t_start = chunk_start + off0 / native_sr

                if dataset_type == "mems":
                    seg = full_mono[off0:off1]
                    if native_sr != _repo_cfg.SR:
                        seg = librosa.resample(seg, orig_sr=native_sr, target_sr=_repo_cfg.SR)
                    y = ap.pad_or_truncate(seg)
                    out_channels = [y, y, y]
                else:
                    out_channels = []
                    for ch_row in full_channels:
                        seg = ch_row[off0:off1]
                        if native_sr != _repo_cfg.SR:
                            seg = librosa.resample(seg, orig_sr=native_sr, target_sr=_repo_cfg.SR)
                        out_channels.append(ap.pad_or_truncate(seg))
                    while len(out_channels) < 3:
                        out_channels.append(out_channels[-1].copy())
                    out_channels = out_channels[:3]

                det      = detect(out_channels, _repo_cfg)
                prob     = float(det["probability"])
                detected = prob >= threshold
                loc = localize(out_channels, _repo_cfg) if (detected and dataset_type != "mems") else None

                gt_match = None
                if ground_truth_xy_at_telemetry:
                    try:
                        gt_match = ground_truth_xy_at_telemetry(filename, t_start, dataset_type, _repo_cfg)
                    except Exception:
                        log.exception("ground_truth_xy_at_telemetry failed for %s@%.2fs", filename, t_start)
                if gt_match is None and ground_truth_xy_at:
                    gt_match = ground_truth_xy_at(filename, t_start, dataset_type)

                window_result = {
                    "t_start": round(t_start, 3),
                    "t_end":   round(t_start + window_s, 3),
                    "detected": detected,
                    "probability": prob,
                    "cnn_probability": float(det.get("cnn_probability", prob)),
                    "heuristic_probability": float(det.get("heuristic_probability", 0.0)),
                    "position": loc["xy_position"].tolist() if loc else None,
                    "azimuth_deg": loc["azimuth_deg"] if loc else None,
                    "distance_m": loc["distance_m"] if loc else None,
                    "true_position":  gt_match["position"] if gt_match else None,
                    "maneuver_type":  gt_match.get("maneuver_type") if gt_match else None,
                    "true_position_source": gt_match.get("source", "planned_maneuver") if gt_match else None,
                }
                results.append(window_result)

                if request_id:
                    # Live progress — lets the UI plot each window as it's
                    # processed instead of waiting for the whole scan to
                    # finish, same idea as the Repository panel's streaming.
                    # The HTTP response at the end is still the authoritative
                    # complete result (used for History); this is purely
                    # progressive UI, safe to miss/reorder if a socket drops.
                    socketio.emit("batch_scan_progress", {
                        "request_id": request_id,
                        "index":      len(results) - 1,
                        "window":     window_result,
                    })

                i += 1
                if len(results) >= MAX_WINDOWS:
                    stop = True
                    break
                if time.time() >= _scan_deadline:
                    time_budget_exceeded = True
                    stop = True
                    break

            chunk_start += chunk_dur
    except Exception as exc:
        log.exception("batch_scan failed")
        if not results:
            return jsonify({"error": f"Scan failed: {exc}"}), 500
        # Partial results are still useful — surface the failure via
        # `truncated`/history rather than discarding everything scanned so far.
        log.warning("batch_scan: returning %d partial windows after failure", len(results))

    actual_scan_duration = (results[-1]["t_end"] - start_s) if results else 0.0
    detected_count = sum(1 for r in results if r["detected"])
    resp = {
        "filename": filename, "dataset_type": dataset_type, "array": array,
        "start_s": start_s,
        "scan_duration_s": scan_duration_s if scan_duration_s is not None else round(actual_scan_duration, 3),
        "full_file": full_file,
        "window_s": window_s, "hop_s": hop_s, "threshold": threshold,
        "n_windows": len(results),
        "truncated": len(results) >= MAX_WINDOWS or time_budget_exceeded,
        "time_budget_exceeded": time_budget_exceeded,
        "detected_count": detected_count,
        "detection_rate": round(100.0 * detected_count / len(results), 1) if results else 0.0,
        "avg_probability": round(float(np.mean([r["probability"] for r in results])), 3) if results else 0.0,
        "windows_with_ground_truth": sum(1 for r in results if r.get("true_position")),
        "windows_with_telemetry_ground_truth": sum(
            1 for r in results if r.get("true_position_source") == "gpx_telemetry"
        ),
        "results": results,
    }

    peak = max(results, key=lambda r: r["probability"]) if results else None
    history_store.log_detection("batch", {
        "source":      f"{filename} ({dataset_type}/{array})",
        "detected":    detected_count > 0,
        "probability": peak["probability"] if peak else 0.0,
        "position":    peak["position"] if peak else None,
    }, resp)

    return jsonify(resp)


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index_v2.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


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