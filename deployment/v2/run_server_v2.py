#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_server_v2.py — Drone Detection System v2 entry point
=========================================================
Location: deployment/v2/run_server_v2.py

Directory layout expected:
    drone-detection-api/               ← REPO_ROOT
    ├── drone_detection/               ← v15 package (drone_detection/__init__.py)
    └── deployment/
        └── v2/                        ← THIS directory (port 5001)
            ├── run_server_v2.py
            ├── app_v2.py
            ├── realtime_sessions.py
            ├── real_time_audio_v2.py
            ├── requirements_v2.txt
            ├── models/
            │   ├── best_detection.pth
            │   └── best_localization.pth
            ├── logs/
            └── templates/
                └── index_v2.html

Usage:
    python run_server_v2.py
    python run_server_v2.py --port 5001 --host 0.0.0.0 --no-debug
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

# ── Resolve paths before any app import ───────────────────────────────────────
THIS_DIR  = Path(__file__).parent.resolve()   # deployment/v2/
REPO_ROOT = THIS_DIR.parent.parent.resolve()  # drone-detection-api/

for p in (str(THIS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Pre-flight checks ──────────────────────────────────────────────────────────

def _check_module(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec:
        print(f"   ✅ {name}  →  {spec.origin}")
        return True
    print(f"   ❌ {name} — NOT FOUND")
    return False


def _preflight() -> None:
    print(f"\n📂 deployment dir : {THIS_DIR}")
    print(f"📂 repo root      : {REPO_ROOT}")

    print("\n🔍 Module check:")
    ok = _check_module("drone_detection")
    _check_module("realtime_sessions")
    _check_module("real_time_audio_v2")

    # Optional — PyAudio only needed for live mic mode
    try:
        import pyaudio  # noqa: F401
        print("   ✅ pyaudio  (live mic mode available)")
    except ImportError:
        print("   ℹ️  pyaudio not installed — live mic mode disabled "
              "(simulated mode still works)")

    if not ok:
        print()
        print("  Fix options:")
        print()
        print("  Option A — run from repo root:")
        print(f"    cd {REPO_ROOT}")
        print(f"    python deployment/v2/run_server_v2.py")
        print()
        print("  Option B — set PYTHONPATH:")
        print(f"    export PYTHONPATH={REPO_ROOT}:$PYTHONPATH")
        print(f"    python run_server_v2.py")
        print()
        print(f"  Expected package: {REPO_ROOT / 'drone_detection'}/__init__.py")
        sys.exit(1)

    print("\n🔍 Directory check:")
    for d in ("templates", "static", "models", "logs"):
        path = THIS_DIR / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created: {path}")
        else:
            print(f"   ✅ {path}")

    # Model files
    print("\n🔍 Model check:")
    model_candidates = {
        "best_detection.pth": [
            THIS_DIR   / "models" / "best_detection.pth",
            REPO_ROOT  / "models" / "best_detection.pth",
            Path("/content/drive/MyDrive/drone_v15/models/best_detection.pth"),
        ],
        "best_localization.pth": [
            THIS_DIR   / "models" / "best_localization.pth",
            REPO_ROOT  / "models" / "best_localization.pth",
            Path("/content/drive/MyDrive/drone_v15/models/best_localization.pth"),
        ],
    }
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        for key in model_candidates:
            model_candidates[key].insert(0, Path(env_path) / key)

    for model_name, candidates in model_candidates.items():
        found = next((p for p in candidates if p.exists()), None)
        if found:
            print(f"   ✅ {model_name}  →  {found}")
        else:
            print(f"   ⚠️  {model_name} not found")
            print(f"      Copy it to: {THIS_DIR / 'models' / model_name}")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drone Detection v2 server")
    p.add_argument("--host",     default="0.0.0.0",
                   help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",     default=5001, type=int,
                   help="Port (default: 5001)")
    p.add_argument("--no-debug", action="store_true",
                   help="Disable debug mode / reloader")
    return p.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging(debug: bool) -> Path:
    log_file = THIS_DIR / "logs" / "app_v2.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return log_file


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args     = _parse_args()
    debug    = not args.no_debug
    log_file = _setup_logging(debug)

    print()
    print("🚁 Drone Detection System v2  (pipeline: drone_detection v15)")
    print("=" * 60)
    print(f"   Host     : {args.host}")
    print(f"   Port     : {args.port}")
    print(f"   Debug    : {'ON' if debug else 'OFF'}")
    print(f"   Log file : {log_file}")

    _preflight()

    print()
    try:
        from app_v2 import app, socketio
    except ImportError as exc:
        print(f"❌ Failed to import app_v2: {exc}")
        print("   Run:  pip install -r requirements_v2.txt")
        return 1

    print("✅ app_v2 loaded successfully")
    print()
    print(f"🌐 Dashboard  : http://localhost:{args.port}/")
    print()
    print("API endpoints:")
    base = f"http://localhost:{args.port}/api/v2"
    for method, path in [
        ("GET ",  "/status"),
        ("GET ",  "/version"),
        ("POST",  "/detect"),
        ("POST",  "/detect-3mic"),
        ("POST",  "/detect-multi"),
        ("POST",  "/noise-test"),
        ("POST",  "/path-simulate"),
        ("POST",  "/realtime/start"),
        ("POST",  "/realtime/stop"),
        ("GET ",  "/realtime/status"),
        ("GET ",  "/realtime/audio-devices"),
    ]:
        print(f"   {method}  {base}{path}")
    print()
    print("WebSocket events: drone_detected_v2, realtime_frame,")
    print("                  realtime_stats, realtime_status")
    print()
    print("⏹  Ctrl+C to stop\n")

    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=debug,
            use_reloader=debug,
            log_output=True,
        )
    except OSError as exc:
        if "Address already in use" in str(exc):
            print(f"\n❌ Port {args.port} is already in use.")
            print(f"   Try: python run_server_v2.py --port {args.port + 1}")
        else:
            print(f"\n❌ OS error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        sys.exit(0)