# Drone Detection System — v2 Deployment
### Pipeline: `drone_detection` (v15)

Independent deployment of the v15 ML pipeline.
Lives in `deployment/v2/` alongside the original deployment in `deployment/v1/`.
**Both can run simultaneously on different ports.**

---

## Quick start

```bash
# From the deployment/v2/ directory:
pip install -r requirements_v2.txt

# Make the drone_detection package importable:
export PYTHONPATH=/path/to/drone-detection-api:$PYTHONPATH

# Place your trained models:
cp /content/drive/MyDrive/drone_v15/models/best_detection.pth    models/
cp /content/drive/MyDrive/drone_v15/models/best_localization.pth models/

# Start the server (default port 5001):
python run_server_v2.py

# With options:
python run_server_v2.py --port 5001 --host 0.0.0.0 --no-debug
```

Open **http://localhost:5001** in your browser.

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
    ├── real_time_audio_v2.py        PyAudio mic capture + real-time detection
    ├── realtime_sessions.py         SimulatedRealtimeSession + RealRealtimeSession
    ├── requirements_v2.txt          Python dependencies
    ├── models/
    │   ├── best_detection.pth       Detection model checkpoint
    │   └── best_localization.pth    Localization model checkpoint
    ├── logs/                        app_v2.log written here automatically
    ├── templates/
    │   └── index_v2.html            Dashboard UI
    └── static/
```

The v2 app imports exclusively from the `drone_detection` package (v15) at the
repo root. It has no dependency on v1 modules, `drone_detection_v2_fixes.py`,
or any external patch files — all v15 fixes are integrated into the package.

---

## What changed from the original v2

| Area | Original v2 | Current v2 |
|------|-------------|------------|
| Core imports | `drone_detection_v2` + `drone_detection_v2_fixes` | `drone_detection` (v15 package, single import) |
| Patch files | `multidrone_localization_patch_v2.py` (separate) | Integrated into `drone_detection.multidrone` |
| Localization model | `LocalizationCNNLite` (depthwise-separable) | `LocalizationCNN` (matches checkpoint architecture) |
| `USE_LITE_LOC` config flag | Present, drove model selection | Removed |
| TDOA synthesis | Integer sample delay | Fractional sinc delay via `_fractional_delay()` |
| Noise in synthesis | Per-mic independent noise | Source-level noise — delayed with signal so GCC-PHAT works |
| Multi-drone solver | Polar `(r,θ)` Nelder-Mead (saturated at 25 m) | Cartesian `(x,y)` Nelder-Mead + soft outer barrier |
| TDOA dedup window | 0.05 µs (580× too tight) | 29 µs (5 % of physical resolution) |
| WebSocket namespace | `/v2` | Root namespace (consistent with frontend) |
| Model file name | `best_model.pth` | `best_detection.pth` + `best_localization.pth` |
| Tracker positions | Hardcoded `ARRAY_CENTER` | Real `localize()` output per segment |
| `noise_level` minimum | 0.04 (GCC-PHAT could fail) | Enforced ≥ 0.05 |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v2/status` | Health check + model info + active threshold |
| `GET`  | `/api/v2/version` | Version string + full fixes list |
| `POST` | `/api/v2/detect` | Single file → CNN+heuristic detection + localization |
| `POST` | `/api/v2/detect-3mic` | 3 mic files → precise localization |
| `POST` | `/api/v2/detect-multi` | 3 mic files → multi-drone (Cartesian solver) |
| `POST` | `/api/v2/noise-test` | SNR sweep using `synthesise_drone()` |
| `POST` | `/api/v2/path-simulate` | Synthetic spiral path tracking demo |
| `POST` | `/api/v2/realtime/start` | Start real-time session (simulated or live mic) |
| `POST` | `/api/v2/realtime/stop` | Stop real-time session |
| `GET`  | `/api/v2/realtime/status` | Session status + running stats |
| `GET`  | `/api/v2/realtime/audio-devices` | List PyAudio input devices |

### WebSocket events (root namespace)

| Event | Direction | Payload |
|-------|-----------|---------|
| `drone_detected_v2` | server → client | `{timestamp, confidence, position, source}` |
| `realtime_frame` | server → client | `{frame, timestamp, detections, tracks, mode, sim_positions}` |
| `realtime_stats` | server → client | `{total_frames, detected_frames, detection_rate, avg_confidence, session_duration}` |
| `realtime_status` | server → client | `{running, mode, error}` |

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

## Real-time modes

### Simulated mode
Uses `synthesise_drone()` from the `drone_detection` package — the identical
function used during training, including fractional-delay propagation and
source-level noise. The pipeline is:

```
synthesise_drone() → detect() → localize() → PathTracker
```

Same thresholds, same model, same guards as live deployment.
`noise_level` is enforced ≥ 0.05 to ensure GCC-PHAT finds a broadband peak.

### Live mic mode
Requires PyAudio + PortAudio. Captures 3-second windows (50 % overlap) from
the selected input device and runs the same `detect() + localize()` pipeline.
Falls back gracefully if PyAudio is unavailable — simulated mode still works.

```bash
# Install PyAudio (Linux):
apt-get install portaudio19-dev
pip install pyaudio

# Install PyAudio (macOS):
brew install portaudio
pip install pyaudio
```

---

## Model loading

The v15 training pipeline stores the F1-maximising threshold in the checkpoint.
`load_detection_model(cfg)` automatically restores it to `cfg.DETECTION_THRESHOLD`.
The `/api/v2/status` endpoint reports the active value. You can override it
per-request with the `threshold` form field.

`LocalizationCNN` is the only localization model class — `LocalizationCNNLite`
and `USE_LITE_LOC` have been removed as the checkpoint architecture matches
`LocalizationCNN` exactly.

---

## Running both versions simultaneously

```bash
# Terminal 1 — v1 (original)
cd deployment/v1
python run_server.py          # → http://localhost:5000

# Terminal 2 — v2 (v15 pipeline)
cd deployment/v2
python run_server_v2.py       # → http://localhost:5001
```

Both share the same model files but have completely independent Flask apps,
SocketIO instances, and runtime state.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `drone-v2-dev-secret` | Flask session secret |
| `MODEL_PATH` | — | Directory containing `best_detection.pth` and `best_localization.pth` |
| `PYTHONPATH` | — | Must include repo root so `drone_detection` is importable |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: drone_detection` | Add repo root to `PYTHONPATH` |
| `Address already in use` | `python run_server_v2.py --port 5002` |
| `best_detection.pth not found` | Copy from `drone_v15/models/` to `deployment/v2/models/` |
| `RuntimeError: Error(s) in loading state_dict` | Old checkpoint with wrong architecture — retrain or check model class |
| `No input devices found` | Install `portaudio19-dev` then `pip install pyaudio` |
| Detection threshold too high/low | Check `/api/v2/status` → `detection_threshold`; pass `threshold=0.62` per-request |
| TDOA always near zero in simulate | `noise_level` too low — enforced to ≥ 0.05 automatically |