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

## Dashboard UI

The dashboard (`templates/index_v2.html`) is a single-page app: a left sidebar
switches between panels, a right-hand **Live Events** feed shows detections
from any panel as they happen (pushed over WebSocket), and a header dot shows
the live connection state. A 🌙/☀️ toggle switches the whole UI between dark
and light themes (persisted in `localStorage`).

The sidebar is grouped into four sections:

### Detection Modes — single-shot analysis on uploaded audio

| Panel | What it does |
|-------|---------------|
| 🎵 **Single File** | Upload one audio file, set an assumed drone (x, y), threshold and max windows → slides a detection window across the **whole file** (not just the first few seconds), showing a verdict banner, confidence ring, and — for files with more than one window — a clickable detection-probability timeline plus a Detection Map of every detected window's position. |
| 📡 **3-Mic Array** | Upload one file per mic (fixed 3-mic baseline geometry) → precise TDOA localization. Shows a measured-vs-estimated TDOA table per mic pair, a reliability badge, and the confidence radius. |
| 🚁🚁 **Multi-Drone** | Upload 3 mic files, set max drones to report → the Cartesian Nelder-Mead solver separates multiple simultaneous drones and plots each on the map with its own confidence radius and frequency band. |
| 📂 **File + Window** | Point at a Nextcloud share (Prof. Mics / polywav, or MEMS — leave the share URL blank to use the server's configured default), pick one specific remote file from a dropdown, and an exact start time + duration → runs detection and localization on just that window. Built for spot-checking a specific moment in a specific recording. |
| 📊 **Batch Scan** | Same file/array picker as above, but slides a detection window across a whole span of the file (capped at 600 s / 150 windows) in one read, and renders a clickable detection-probability timeline plus a Detection Map showing every detected window's position as a chronological trajectory. |

### Analysis — synthetic robustness checks

| Panel | What it does |
|-------|---------------|
| 🔊 **Noise SNR** | Runs an SNR sweep (configurable dB range/step, clips per level) through `synthesise_drone()` to show how detection rate degrades with noise. |
| 🛤️ **Path Simulate** | Generates a synthetic multi-waypoint flight path and localizes each waypoint, exercising the `PathTracker` / Kalman filter and flagging points outside the array's reliable 2.5 m zone. |

### Live Operations — continuous streaming sessions with a radar view

| Panel | What it does |
|-------|---------------|
| 🔴 **Real-Time** | A radar-style live view with two modes: **Simulated** (runs the exact production pipeline — `synthesise_drone() → detect() → localize() → PathTracker` — against configurable drone count, flight pattern, noise, and spread) or **Live Mic** (captures from a real PyAudio input device). Shows a live frame counter, active/confirmed tracks, detection log, and running stats (detection rate, avg confidence). |
| 🗄️ **Repository** | Streams **real recorded drone audio** live from a remote dataset — UaVirBASE, Dunakeszi, or MEMS — via HTTP range-requests against a Nextcloud share (no files are downloaded or stored server-side; falls back to a local cache or synthetic BPF-profile audio if the repository is unreachable). Overlays ground-truth position/azimuth against the model's estimate when a segment is labelled, plots a rolling localisation-error histogram, and — for Dunakeszi — includes a **Segment Browser** to filter by split/show and play one specific labelled maneuver on demand. |

### System

| Panel | What it does |
|-------|---------------|
| ⚙️ **Status** | Live health check (model loaded, device, sample rate, active sessions) plus a built-in API reference card. |

A **"v2 Fixes Active"** checklist is always visible at the bottom of the
sidebar, listing the pipeline correctness fixes currently in effect
(fractional sinc delay, cap-hit guard, NaN-safe GCC-PHAT, etc. — see
[What changed from the original v2](#what-changed-from-the-original-v2)).

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
    ├── run_server_v2.py             Local dev entry point (socketio.run())
    ├── passenger_wsgi.py            cPanel/Passenger entry point (plain WSGI, no SocketIO)
    ├── real_time_audio_v2.py        PyAudio mic capture + real-time detection
    ├── realtime_sessions.py         Simulated / Live-mic / Repository session classes
    ├── requirements_v2.txt          Python dependencies (local dev)
    ├── requirements-prod.txt        Python dependencies for shared/CPU-only hosting
    ├── drone_detection/
    │   ├── config.py                Config singleton (thresholds, mic geometry, model paths)
    │   ├── repository_loader.py     Dataset-agnostic router: UaVirBASE / Dunakeszi / MEMS
    │   ├── dunakeszi_nextcloud.py   Nextcloud HTTP range-read streaming, WAV/polywav decode
    │   └── mems_inference.py        Single-channel MEMS spectral-proxy localisation
    ├── dunakeszi_ground_truth_fixed.py  Ground-truth maneuver/segment metadata for Dunakeszi
    ├── dunakeszi_pipeline_ready_B/  Local cache of pre-paired Dunakeszi audio+label triplets
    ├── wavs/                        Local polywav cache used as a fallback when Nextcloud
    │                                is unreachable or unconfigured
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
| `POST` | `/api/v2/realtime/start` | Start real-time session (`mode`: `simulated` \| `real` \| `repository`) |
| `POST` | `/api/v2/realtime/stop` | Stop real-time session |
| `GET`  | `/api/v2/realtime/status` | Session status + running stats |
| `GET`  | `/api/v2/realtime/audio-devices` | List PyAudio input devices |
| `GET`  | `/api/v2/repository/segments` | Dunakeszi segment browser: filterable list of labelled maneuvers |
| `POST` | `/api/v2/repository/start-segment` | Start a repository session streaming one specific labelled segment |
| `GET`  | `/api/v2/repository/files/<file_type>` | Dunakeszi file browser (raw listing) |
| `GET`  | `/api/v2/repository/files-list` | Filename list for the File+Window / Batch Scan pickers (`dataset_type`: `polywav` \| `mems`) |
| `POST` | `/api/v2/repository/manual-window` | Detect + localize an explicit file + time window |
| `POST` | `/api/v2/repository/batch-scan` | Slide a detection window across a span of one file → timeline of results |

### WebSocket events (root namespace)

| Event | Direction | Payload |
|-------|-----------|---------|
| `drone_detected_v2` | server → client | `{timestamp, confidence, position, source}` |
| `realtime_frame` | server → client | `{frame, timestamp, detections, tracks, mode, sim_positions, repo_label}` |
| `realtime_stats` | server → client | `{total_frames, detected_frames, detection_rate, avg_confidence, avg_error_m, session_duration}` |
| `realtime_status` | server → client | `{running, mode, error, loading, final_stats}` |
| `array_geometry_changed` | server → client | `{geometry, mic_positions}` — fired when a repository session switches mic array layout (e.g. UaVirBASE ring vs. Dunakeszi GP1/GP2 triangle) |

### Example: single file detection

```bash
curl -X POST http://localhost:5001/api/v2/detect \
  -F "file=@drone_audio.wav" \
  -F "threshold=0.62" \
  -F "n_segments=20"
```

`n_segments` caps how many 3-second windows are scanned (max 150) — the file
is scanned end-to-end regardless of length, not just its first few seconds.
Top-level `detected`/`probability`/`position` reflect the peak-confidence
window; `results` holds the full per-window timeline.

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
  "duration_s": 12.0,
  "window_s": 3.0,
  "hop_s": 3.0,
  "n_windows": 4,
  "truncated": false,
  "detected_count": 1,
  "detection_rate": 25.0,
  "avg_probability": 0.312,
  "detection_summary": {
    "total_segments": 4,
    "detected_segments": 1,
    "max_confidence": 0.951
  },
  "n_tracks": 0,
  "results": [
    {"t_start": 0.0, "t_end": 3.0, "detected": false, "probability": 0.04, "cnn_probability": 0.03, "heuristic_probability": 0.11, "position": null, "azimuth_deg": null, "distance_m": null, "height_m": null},
    {"t_start": 3.0, "t_end": 6.0, "detected": true,  "probability": 0.951, "cnn_probability": 0.963, "heuristic_probability": 0.887, "position": [1.06, 0.85], "azimuth_deg": 38.7, "distance_m": 1.34, "height_m": 0.0}
  ]
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

### Repository mode
Streams real recorded drone audio from a public dataset instead of synthetic
or live-mic audio — `mode=repository`, `dataset_type` one of `uavirbase` |
`dunakeszi` | `mems`. No audio is ever downloaded to disk; it's read via
HTTP range-requests against a Nextcloud public share (`remotezip` /
byte-range reads), with graceful fallback at each step:

```
local pipeline-ready cache  →  Nextcloud HTTP range-read  →  synthetic
(if present)                   (real recorded audio)         BPF-profile audio
                                                               (repo unreachable)
```

Each dataset has its own mic array geometry, switched automatically and
broadcast to the UI via `array_geometry_changed`:

| `dataset_type` | Array | Channels |
|---|---|---|
| `uavirbase` | 8-mic ring, 1.72 m radius | ch1 (N), ch2 (E), ch4 (W) |
| `dunakeszi` | GP1/GP2 equilateral triangle (2.165 m / 2.5 m baseline) | `BK-6-E` or `BK-6-W` channel group, selected per request |
| `mems` | Single channel | Spectral-proxy azimuth/distance only — no true TDOA, so tracks aren't drawn |

When a segment has ground-truth position/azimuth metadata (Dunakeszi), the
estimate is compared against it live and the running localisation error is
histogrammed in the UI. Note that the localization model's distance output
is normalised to `cfg.MAX_LOCALIZATION_DIST` (30 m, trained on ≤20 m
near-field data) — Dunakeszi's long-range outdoor maneuvers can exceed that
by 3–4×, in which case the estimate saturates at the cap and is flagged
`cap_hit` / not `reliable` rather than reported as a real position.

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
| `NEXTCLOUD_BASE_URL` | — | Nextcloud host for Repository mode (e.g. `https://nc.example.com`) — can also be set at runtime by pasting a share URL into the UI |
| `NEXTCLOUD_SHARE_TOKEN` | — | Public share token for the above; paired with the share URL's `?dir=` path |

Priority is **explicit share URL pasted into the UI > these env vars > whatever
was set on a previous request** (`Config.reload_nextcloud_env()`). This means
every Nextcloud share URL field in the UI — Repository, File + Window, Batch
Scan — is optional: leave it blank and the server falls back to these env
vars, so people who don't have the actual share link can still browse and
test against whatever dataset the server operator configured by default.

---

## Deploying to shared cPanel hosting (Truehost)

This app is built for local/VPS use (`socketio.run()`, PyTorch, ~7GB of
CUDA-bundled deps if installed naively) — a few adjustments make it work on
a shared cPanel "Setup Python App" (Passenger) plan too.

Two files exist specifically for this:

- **`passenger_wsgi.py`** — the entry point Passenger requires, physically
  present at the app root, exposing `application = app_v2.app`. cPanel's
  "Application startup file" / "Application Entry point" fields (set to
  `app_v2.py` / `app`) only re-derive this file's wiring at the moment you
  click **Save** — if the file is later missing (deleted, or absent on a
  fresh `git clone`/`pull`), Passenger falls back to its own generic
  placeholder page ("It works! Python vX.Y.Z") instead of regenerating it.
  So this file has to stay committed and present, not hand-managed per
  deploy. It bypasses `socketio.run()` entirely, which is fine — Single
  File and Batch Scan are both plain HTTP POST endpoints. **The Real-Time
  (WebSocket) panel will not work this way**, but shared cPanel hosting
  generally doesn't support persistent WebSocket connections regardless.

`requirements-prod.txt` exists for the one thing that *does* need changing:
same runtime deps as `requirements_v2.txt` minus test-only packages, with
`torch`/`torchaudio` pulled from PyTorch's CPU-only wheel index (embedded as
an `--extra-index-url` line in the file itself) instead of PyPI's default
CUDA-bundled build — several GB each, which will blow a shared disk quota
and can time out mid-install.

### Procedure

1. **Package only the runtime files** — not this whole directory, which also
   holds tens of GB of research data (`wavs/`, `dataset_builders/`,
   `dunakeszi_data/`, raw multi-GB `.wav` recordings, etc.). Upload just:
   ```
   app_v2.py  passenger_wsgi.py  requirements-prod.txt  run_server_v2.py
   real_time_audio_v2.py  realtime_sessions.py  repository_loader.py
   drone_detection/  templates/  static/  models/  logs/
   ```
2. **Upload/clone** into the app root cPanel assigns (or point Setup Python
   App at wherever you `git clone`d the repo — the app root just needs to be
   `deployment/v2/` inside your checkout).
3. **cPanel → Setup Python App → Create Application**
   - Application root: your `deployment/v2/` path
   - Application URL: whichever subdomain/path you want it under
   - Python version: the highest available (torch needs a recent one)
   - Application startup file: `app_v2.py` · Entry point: `app`
4. **Configuration files** (same page) — remove `requirements_v2.txt` from
   the list if present, add `requirements-prod.txt`, then **Run Pip Install**.
   If CPU wheels for the exact pinned `torch`/`torchaudio` version aren't
   available for the selected Python version, drop the version pins on just
   those two lines in `requirements-prod.txt` and reinstall — any recent 2.x
   CPU build reads the same `.pth` checkpoints fine.
5. **Restart** the app, then check `https://your-app-url/api/v2/status` →
   `"model_loaded": true`.
6. **Environment variables** (same page) — set `NEXTCLOUD_BASE_URL` /
   `NEXTCLOUD_SHARE_TOKEN` if you want Batch Scan / File + Window to work
   against your Nextcloud dataset by default (see above).

If you `git pull` onto a cPanel app root that was set up before you had this
repo layout, you may hit `error: The following untracked working tree files
would be overwritten by merge: deployment/v2/passenger_wsgi.py` — that's
cPanel's own placeholder stub sitting untracked at that path (not your
data), from before this repo's `passenger_wsgi.py` existed there. Safe to
remove and re-pull, since the pull immediately replaces it with the real one:
```bash
rm deployment/v2/passenger_wsgi.py
git pull
```
If the site ever shows cPanel's generic **"It works! Python vX.Y.Z"** page
instead of the app, that means `passenger_wsgi.py` is missing or broken at
the app root — confirm it's present and unmodified, then hit **Restart** on
the Setup Python App page (or `touch tmp/restart.txt` if you're doing it
over SSH).

### Operating notes

- Upload size is already capped server-side — `MAX_CONTENT_LENGTH = 50MB` in
  `app_v2.py` rejects oversized files before any processing starts.
- The real resource risk on shared hosting is **CPU time, not upload size**:
  Single File's sliding-window scan runs one CNN inference pass per window
  (up to `n_segments`, capped at 150). Shared plans enforce a per-request
  timeout you can't configure — scanning near the 150-window cap on a large
  file risks the request getting killed mid-scan on constrained CPU. Keep
  the UI's "Max Windows" default (20 ≈ 60s of audio) for production rather
  than pushing it toward 150.

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
| `files-list` / `manual-window` / `batch-scan` returns "Nextcloud not configured" | Paste a Nextcloud share URL into that panel's field, or set `NEXTCLOUD_BASE_URL` + `NEXTCLOUD_SHARE_TOKEN` |
| Repository detection shows huge position error but "Reliable ✓ Yes" | Check `cap_hit` in the response — the localization model's distance output is capped at `cfg.MAX_LOCALIZATION_DIST` (30 m); targets beyond that (e.g. Dunakeszi's long-range maneuvers) can't be represented and the estimate saturates |