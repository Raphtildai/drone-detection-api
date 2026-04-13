#!/usr/bin/env python3
"""
run_server_v2.py — Drone Detection System v2 (pipeline aligned with v15)
=========================================================================
Location: drone-detection-api/deployment/v2/run_server_v2.py

Usage:
    python run_server_v2.py
    python run_server_v2.py --port 5001
    python run_server_v2.py --host 0.0.0.0 --no-debug

Directory layout this script expects:
    drone-detection-api/               ← REPO_ROOT
    ├── drone_detection_v2.py          ← core ML module (v15 compatible)
    └── deployment/
        └── v2/                        ← THIS directory (port 5001)
            ├── run_server_v2.py       ← this file
            ├── app_v2.py
            ├── real_time_audio_v2.py
            ├── realtime_sessions.py
            ├── requirements_v2.txt
            ├── models/best_model.pth
            └── templates/index_v2.html
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

# ── Resolve paths BEFORE any app import ───────────────────────────────────────
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
    print(f"\n🔍 Module check:")

    ok = _check_module("drone_detection_v2")
    # v15 patch modules are optional but desirable
    for optional in ("multidrone_localization_patch_v2",):
        found = _check_module(optional)
        if not found:
            print(f"   ℹ️  {optional} not found — optional, basic fallback will be used")

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
        print(f"  Expected file: {REPO_ROOT / 'drone_detection_v2.py'}")
        sys.exit(1)

    print()
    print("🔍 Directory check:")
    for d in ("templates", "static", "models", "logs"):
        path = THIS_DIR / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created: {path}")
        else:
            print(f"   ✅ {path}")

    model_candidates = [
        THIS_DIR / "models" / "best_model.pth",
        REPO_ROOT / "models" / "best_model.pth",
        Path("/content/drive/MyDrive/drone_project/models/best_model.pth"),
    ]
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        model_candidates.insert(0, Path(env_path))

    print("\n🔍 Model check:")
    found_model = False
    for mp in model_candidates:
        if mp.exists():
            print(f"   ✅ {mp}")
            found_model = True
            break
    if not found_model:
        print("   ⚠️  best_model.pth not found.")
        print(f"      Copy it to: {THIS_DIR / 'models' / 'best_model.pth'}")
        print("      Or set:     export MODEL_PATH=/path/to/best_model.pth")


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
    print("🚁 Drone Detection System v2  (pipeline: drone_detection_v15)")
    print("=" * 55)
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
    print("WebSocket: realtime_frame, realtime_stats, realtime_status")
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