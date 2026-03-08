#!/usr/bin/env python3
"""
run_server_v2.py — Drone Detection System v2
=============================================
Location: drone-detection-api/deployment/v2/run_server_v2.py

Usage:
    python run_server_v2.py
    python run_server_v2.py --port 5001
    python run_server_v2.py --host 0.0.0.0 --no-debug

Directory layout this script expects:
    drone-detection-api/               ← REPO_ROOT
    ├── drone_detection_v2.py          ← core ML module (imported by app_v2.py)
    ├── drone_detection_v2_fixes.py    ← FIX-1 to FIX-8 patches
    └── deployment/
        ├── v1/                        ← original deployment (port 5000, untouched)
        └── v2/                        ← THIS directory (port 5001)
            ├── run_server_v2.py       ← this file
            ├── app_v2.py
            ├── real_time_audio_v2.py
            ├── requirements_v2.txt
            ├── models/best_model.pth
            └── templates/index_v2.html
"""

import os
import sys
import argparse
import logging
import importlib.util
from pathlib import Path

# ── Resolve paths BEFORE any app import ──────────────────────────────────────
#
#   __file__  = .../drone-detection-api/deployment/v2/run_server_v2.py
#   THIS_DIR  = .../drone-detection-api/deployment/v2/
#   REPO_ROOT = .../drone-detection-api/
#
THIS_DIR  = Path(__file__).parent.resolve()   # deployment/v2/
REPO_ROOT = THIS_DIR.parent.parent.resolve()  # drone-detection-api/

for p in (str(THIS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Pre-flight: verify the core ML module is importable ──────────────────────
def _check_module(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec:
        print(f"   ✅ {name}  →  {spec.origin}")
        return True
    print(f"   ❌ {name} — NOT FOUND")
    return False


def _preflight():
    print(f"\n📂 deployment dir : {THIS_DIR}")
    print(f"📂 repo root      : {REPO_ROOT}")
    print(f"\n🔍 Module check:")

    ok = _check_module("drone_detection_v2")
    _check_module("drone_detection_v2_fixes")   # optional but nice to know

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
    for d in ("templates", "static", "models"):
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

    found_model = False
    print("\n🔍 Model check:")
    for mp in model_candidates:
        if mp.exists():
            print(f"   ✅ {mp}")
            found_model = True
            break
    if not found_model:
        print(f"   ⚠️  best_model.pth not found.")
        print(f"      Copy it to: {THIS_DIR / 'models' / 'best_model.pth'}")
        print(f"      Or set:     export MODEL_PATH=/path/to/best_model.pth")


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Drone Detection v2 server")
    p.add_argument("--host",     default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port",     default=5001, type=int, help="Port (default: 5001)")
    p.add_argument("--no-debug", action="store_true", help="Disable debug mode / reloader")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(debug: bool):
    log_file = THIS_DIR / "app_v2.log"
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    debug = not args.no_debug
    log_file = setup_logging(debug)

    print()
    print("🚁 Drone Detection System v2")
    print("=" * 44)
    print(f"   Host     : {args.host}")
    print(f"   Port     : {args.port}")
    print(f"   Debug    : {'ON' if debug else 'OFF'}")
    print(f"   Log file : {log_file}")

    _preflight()

    print()
    try:
        from app_v2 import app, socketio
    except ImportError as e:
        print(f"❌ Failed to import app_v2: {e}")
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
    ]:
        print(f"   {method}  {base}{path}")
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
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Port {args.port} is already in use.")
            print(f"   Try: python run_server_v2.py --port {args.port + 1}")
        else:
            print(f"\n❌ OS error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        sys.exit(0)