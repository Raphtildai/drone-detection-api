# Drone Detection System — v2 Deployment

Independent deployment of the v2 ML pipeline.  
Lives in `deployment/v2/` alongside the v3 deployment in `deployment/v1/`.  
**Both can run simultaneously on different ports.**

---

## Quick start

```bash
# From the deployment/v2/ directory:
pip install -r requirements_v2.txt

# Make sure drone_detection_v2.py and drone_detection_v2_fixes.py
# are importable (either copy them here or add repo root to PYTHONPATH):
export PYTHONPATH=/path/to/repo/root:$PYTHONPATH

# Copy your trained model:
cp /content/drive/MyDrive/drone_project/models/best_model.pth models/

# Start the server (default port 5001 — v3 uses 5000):
python run_server_v2.py

# Or with options:
python run_server_v2.py --port 5001 --host 0.0.0.0 --no-debug
```

Open **http://localhost:5001/v2** in your browser.

---

## File structure

```
deployment/
├── v1/                         ← existing v3 deployment (unchanged)
│   ├── app.py
│   ├── real_time_api.py
│   ├── run_server.py
│   └── ...
│
└── v2/                         ← THIS directory
    ├── app_v2.py               Flask application (all endpoints under /api/v2/)
    ├── run_server_v2.py        Entry point  (port 5001 by default)
    ├── real_time_audio_v2.py   Mic capture + real-time detection
    ├── requirements_v2.txt     Python dependencies
    ├── models/                 Place best_model.pth here
    ├── templates/
    │   └── index_v2.html       Dashboard UI
    └── static/
        ├── css/
        └── js/
```

The v2 app imports `drone_detection_v2` and `drone_detection_v2_fixes` from
the repository root (or wherever they live on `sys.path`).  It never imports
from the v3 modules (`app.py`, `real_time_api.py`, `real_time_audio.py`).

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v2/status` | Health check + model info |
| GET | `/api/v2/version` | Version + active fixes list |
| POST | `/api/v2/detect` | Single file → detection + localization |
| POST | `/api/v2/detect-3mic` | 3 mic files → precise localization |
| POST | `/api/v2/detect-multi` | 3 mic files → multi-drone |
| POST | `/api/v2/noise-test` | SNR sweep (synthetic fallback built-in) |
| POST | `/api/v2/path-simulate` | Synthetic path tracking demo |

WebSocket namespace: `/v2`  
Real-time event: `drone_detected_v2`

---

## Running both versions simultaneously

```bash
# Terminal 1 — v3 (original)
cd deployment/v1
python run_server.py          # port 5000

# Terminal 2 — v2 (improved)
cd deployment/v2
python run_server_v2.py       # port 5001
```

Both share the same model file but have completely independent Flask apps,
SocketIO instances, and state.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `drone-v2-dev-secret` | Flask session secret |
| `MODEL_PATH` | `models/best_model.pth` | Override model location |
| `PYTHONPATH` | — | Add repo root for imports |