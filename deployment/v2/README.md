# Drone Detection System — v2 Deployment
### Pipeline: `drone_detection_v15`

Independent deployment of the v15 ML pipeline.
Lives in `deployment/v2/` alongside the original deployment in `deployment/v1/`.
**Both can run simultaneously on different ports.**

---

## What changed in this update (v15 alignment)

| Area | Before (v2 original) | After (v15 aligned) |
|------|----------------------|---------------------|
| **Detection** | CNN softmax only | CNN + heuristic hybrid (`detect()`) |
| **Features** | log-mel (3-channel repeat) | `[log-mel, PCEN, delta-mel]` via `feature_stack()` |
| **Training loss** | CrossEntropyLoss | FocalLoss (γ=2, α=0.6, label smoothing 0.02) |
| **Threshold** | Fixed 0.70 | Auto-searched on val set; stored in checkpoint |
| **Multi-drone localizer** | `(r,θ)` Nelder-Mead (hard clip at 25 m) | Cartesian `(x,y)` Nelder-Mead + soft barrier |
| **TDOA dedup window** | 0.05 µs (580× too tight) | 29 µs (physical resolution) |
| **Kalman match gate** | 2.0 m | 8.0 m (accounts for TDOA noise) |
| **Kalman min_hits** | 2 | 1 (confirms tracks faster) |
| **Path tracker** | `PathTracker` (placeholder) | `PathTracker` using `DroneTrack` (v15 API) |
| **Realtime session** | Uses `localize_precise()` | Uses `detect()` + `localize()` hybrid |

---

## Quick start

```bash
# From the deployment/v2/ directory:
pip install -r requirements_v2.txt

# Make drone_detection_v2.py (v15) importable:
export PYTHONPATH=/path/to/drone-detection-api:$PYTHONPATH

# Place your trained model:
cp /content/drive/MyDrive/drone_project/models/best_model.pth models/

# (Optional) Copy v15 patch modules for best multi-drone accuracy:
cp /path/to/multidrone_localization_patch_v2.py .

# Start the server (default port 5001):
python run_server_v2.py

# With options:
python run_server_v2.py --port 5001 --host 0.0.0.0 --no-debug
```

Open **http://localhost:5001/v2** in your browser.

---

## File structure

```
deployment/
├── v1/                              ← original deployment (unchanged, port 5000)
│   ├── app.py
│   ├── run_server.py
│   └── ...
│
└── v2/                              ← THIS directory (port 5001)
    ├── app_v2.py                    Flask app (all routes under /api/v2/)
    ├── run_server_v2.py             Entry point
    ├── real_time_audio_v2.py        Mic capture + real-time detection
    ├── realtime_sessions.py         SimulatedRealtimeSession + RealRealtimeSession
    ├── requirements_v2.txt          Python dependencies
    ├── models/                      Place best_detection.pth / best_model.pth here
    ├── logs/                        app_v2.log written here automatically
    ├── templates/
    │   └── index_v2.html            Dashboard UI
    └── static/
```

The v2 app imports from `drone_detection_v2` (the v15 module) at the repo root.
It never imports from the v1 modules.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v2/status`  | Health check + model info + detection threshold |
| `GET`  | `/api/v2/version` | Version + all active fixes list |
| `POST` | `/api/v2/detect`  | Single file → CNN+heuristic detection + localization |
| `POST` | `/api/v2/detect-3mic` | 3 mic files → precise localization |
| `POST` | `/api/v2/detect-multi` | 3 mic files → multi-drone (Cartesian solver) |
| `POST` | `/api/v2/noise-test` | SNR sweep (synthetic clips, no real audio needed) |
| `POST` | `/api/v2/path-simulate` | Synthetic spiral path tracking demo |
| `POST` | `/api/v2/realtime/start` | Start real-time session (simulated or live mic) |
| `POST` | `/api/v2/realtime/stop`  | Stop real-time session |
| `GET`  | `/api/v2/realtime/status` | Session status + stats |
| `GET`  | `/api/v2/realtime/audio-devices` | List PyAudio input devices |

### WebSocket events (connect to root namespace)

| Event | Payload |
|-------|---------|
| `drone_detected_v2` | `{timestamp, confidence, position, source}` |
| `realtime_frame` | `{frame, timestamp, detections, tracks, mode, sim_positions}` |
| `realtime_stats` | `{total_frames, detected_frames, detection_rate, avg_confidence, session_duration}` |
| `realtime_status` | `{running, mode, error}` |

### Example: single file detection

```bash
curl -X POST http://localhost:5001/api/v2/detect \
  -F "file=@drone_audio.wav" \
  -F "threshold=0.62" \
  -F "n_segments=5"
```

Response:
```json
{
  "detected": true,
  "probability": 0.951,
  "cnn_probability": 0.963,
  "heuristic_probability": 0.887,
  "position": [1.06, 0.85],
  "azimuth_deg": 38.7,
  "distance_m": 1.34,
  "height_m": 0.0,
  "detection_summary": {
    "total_segments": 5,
    "detected_segments": 1,
    "max_confidence": 0.951
  },
  "n_tracks": 0
}
```

---

## Detection threshold

The v15 training pipeline searches for the best F1-maximising threshold on the
validation set and stores it in the checkpoint. When the model is loaded via
`load_detection_model(cfg)`, `cfg.DETECTION_THRESHOLD` is automatically updated
to this value. The `/api/v2/status` endpoint reports the active threshold.

You can still override it per-request with the `threshold` form field.

---

## Multi-drone localizer — v15 fixes

The Cartesian Nelder-Mead solver (`localize_multi_drone_v2`) fixes three bugs
in the original `localize_multi_drone()`:

1. **Distance saturation** — old `(r, θ)` reparametrisation hard-clipped
   `r` at `MAX_LOCALIZATION_DIST` (25 m), causing all far solutions to
   saturate. The new solver works in `(x, y)` with a soft outer barrier,
   so the optimizer converges freely at any distance.

2. **Degenerate `r ≈ 0` solutions** — TDOA residual is trivially zero at
   the array centre. Fixed by a soft inner penalty + `MIN_SOLUTION_DIST = 0.30 m`.

3. **TDOA dedup** — old window was 0.05 µs (580× narrower than the physical
   resolution of ≈ 0.58 ms for a 20 cm baseline). New window: 29 µs.

To use the v15 solver, copy `multidrone_localization_patch_v2.py` from the
training repo to the same directory as `app_v2.py`. The app will auto-detect
and import it. If it is absent, the app falls back to the original
`localize_multi_drone()` from `drone_detection_v2.py`.

---

## Running both versions simultaneously

```bash
# Terminal 1 — v1 (original)
cd deployment/v1
python run_server.py          # → http://localhost:5000

# Terminal 2 — v2 (v15 pipeline)
cd deployment/v2
python run_server_v2.py       # → http://localhost:5001/v2
```

Both share the same model file but have completely independent Flask apps,
SocketIO instances, and state.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `drone-v2-dev-secret` | Flask session secret |
| `MODEL_PATH` | `models/best_detection.pth` | Override model file path |
| `PYTHONPATH` | — | Must include repo root for imports |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: drone_detection_v2` | Add repo root to `PYTHONPATH` |
| `Address already in use` | `python run_server_v2.py --port 5002` |
| Model not loaded warning | Copy `best_detection.pth` (or `best_model.pth`) to `deployment/v2/models/` |
| `No input devices found` | Install `portaudio19-dev` then `pip install pyaudio` |
| Multi-drone always returns 3 drones | Copy `multidrone_localization_patch_v2.py` to `deployment/v2/` |
| Detection threshold too high/low | Check `/api/v2/status` → `detection_threshold`; pass `threshold=0.62` explicitly |