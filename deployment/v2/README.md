# 🚁 Drone Detection & Localization System v2

A deep learning system for detecting drone presence in audio recordings and estimating drone position using a 3-microphone array. Includes a Flask web API, real-time WebSocket monitoring, and an interactive browser dashboard.

> **v2** fixes 8 critical localization bugs from v1 — near-field position errors drop from ~14 m to < 0.5 m.

---

## ✨ Features

- **AI-powered detection** — CNN on mel spectrograms, 98.83% validation accuracy
- **3-mic TDOA localization** — GCC-PHAT + Nelder-Mead solver, reliable within 2.5 m of array
- **Multi-drone detection** — band-separated GCC-PHAT across 6 frequency bands
- **Real-time monitoring** — WebSocket push via Socket.IO
- **Noise robustness testing** — SNR sweep from −5 to +20 dB with synthetic fallback
- **Interactive dashboard** — dark-themed browser UI with live event feed
- **Independent from v1** — runs on port 5001 alongside the original deployment

---

## 📁 Project Structure

```
drone-detection-api/
├── drone_detection_v2.py          # Core ML pipeline (train + detect + localize)
│
├── deployment/
│   ├── v1/                        # Original v1 deployment (port 5000)
│   │   ├── app.py
│   │   ├── real_time_api.py
│   │   └── run_server.py
│   │
│   └── v2/                        # This deployment (port 5001)
│       ├── app_v2.py              # Flask app — all routes under /api/v2/
│       ├── run_server_v2.py       # Entry point
│       ├── real_time_audio_v2.py  # Microphone capture + real-time detection
│       ├── requirements_v2.txt    # Python dependencies
│       ├── models/                # Place best_model.pth here
│       └── templates/
│           └── index_v2.html      # Dashboard UI
│
└── notebooks/
    └── drone_detection_v2_test_suite.ipynb  # Colab training + testing notebook
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- A trained model file (`best_model.pth`) — see [Training](#training)

### 1. Clone the repo

```bash
git clone https://github.com/Raphtildai/drone-detection-api.git
cd drone-detection-api
```

### 2. Install dependencies

```bash
cd deployment/v2
pip install -r requirements_v2.txt
```

### 3. Make the v2 source importable

```bash
# Linux / macOS
export PYTHONPATH=/path/to/drone-detection-api:$PYTHONPATH

# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\drone-detection-api;" + $env:PYTHONPATH
```

### 4. Place your trained model

```bash
cp /content/drive/MyDrive/drone_project/models/best_model.pth deployment/v2/models/
```

### 5. Start the server

```bash
python run_server_v2.py
```

Open **http://localhost:5001/v2** in your browser.

---

## 🌐 API Endpoints

All endpoints are prefixed with `/api/v2/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v2/status` | Health check + model info |
| `GET` | `/api/v2/version` | Version + active fixes list |
| `POST` | `/api/v2/detect` | Single audio file → detection + localization |
| `POST` | `/api/v2/detect-3mic` | 3 mic files → precise GCC-PHAT localization |
| `POST` | `/api/v2/detect-multi` | 3 mic files → multi-drone band-separated detection |
| `POST` | `/api/v2/noise-test` | SNR sweep (generates synthetic clips if needed) |
| `POST` | `/api/v2/path-simulate` | Synthetic path tracking demo |

### Example: single file detection

```bash
curl -X POST http://localhost:5001/api/v2/detect \
  -F "file=@drone_audio.wav" \
  -F "threshold=0.70" \
  -F "n_segments=5"
```

**Response:**
```json
{
  "detected": true,
  "probability": 0.952,
  "position": [1.06, 0.85],
  "summary": { "total_segments": 5, "detected_segments": 3, "max_confidence": 0.952 },
  "n_tracks": 1
}
```

### Example: 3-microphone localization

```bash
curl -X POST http://localhost:5001/api/v2/detect-3mic \
  -F "mic1=@channel1.wav" \
  -F "mic2=@channel2.wav" \
  -F "mic3=@channel3.wav" \
  -F "threshold=0.70"
```

### POST field reference

**`/api/v2/detect`**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file (.wav .mp3 .flac .ogg .m4a) |
| `drone_x` | float | `1.0` | Hint X position in metres |
| `drone_y` | float | `0.8` | Hint Y position in metres |
| `threshold` | float | `0.70` | Detection probability threshold (0.1–0.99) |
| `n_segments` | int | `5` | Number of 3-second segments to analyse |
| `force_detect` | bool | `false` | Skip classifier, always run localization |

**`/api/v2/detect-3mic` and `/api/v2/detect-multi`**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mic1` / `mic2` / `mic3` | file | required | One audio file per microphone |
| `threshold` | float | `0.70` | Detection probability threshold |
| `hint_x`, `hint_y` | float | optional | Initial search position for solver |
| `max_drones` | int | `3` | Max drones to report (multi mode only) |

**`/api/v2/noise-test`**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `snr_min` | int | `-5` | Lowest SNR level in dB |
| `snr_max` | int | `20` | Highest SNR level in dB |
| `snr_step` | int | `5` | Step between levels |
| `n_clips` | int | `20` | Clips per level — synthesised automatically if no real clips exist |

### WebSocket events

Connect to the `/v2` Socket.IO namespace to receive real-time push events.

```javascript
const socket = io("http://localhost:5001", { path: "/socket.io" });

socket.on("drone_detected_v2", (data) => {
  console.log(data);
  // { timestamp, confidence, position, n_drones, source }
});
```

---

## 🖥️ Dashboard

The browser UI at `http://localhost:5001/v2` has six modes:

| Tab | Description |
|-----|-------------|
| 🎵 Single File | Upload audio, adjust threshold, view position on mini-map |
| 📡 3-Mic Array | Upload 3 mic files, view localization with `reliable` / `cap_hit` flags |
| 🚁🚁 Multi-Drone | Band-separated detection with per-drone frequency and confidence radius |
| 🔊 Noise SNR | SNR sweep with colour-coded accuracy bar chart |
| 🛤️ Path Simulate | Synthetic waypoint path tracking with track results |
| ⚙️ Status | System health, model info, and inline API reference |

A live event feed on the right side shows every `drone_detected_v2` WebSocket event in real time.

---

## 🔧 v2 Bug Fixes

Eight precision fixes are applied on top of the v1 baseline:

| Fix | Function | Problem fixed |
|-----|----------|---------------|
| FIX-1 | `generate_synthetic_drone()` | Integer delay rounding caused ~45 µs TDOA error → optimizer escaped to wrong hyperboloid branch at 15 m. Now uses 8-tap fractional sinc filter. |
| FIX-2 | `localize_precise()` | Wrong-branch solutions had tiny residuals, passing quality gate. Added `cap_hit` guard: any solution within 5 cm of the 15 m ceiling is flagged unreliable. |
| FIX-3 | `gcc_phat()` | Silent frequency bands produced NaN, corrupting TDOA estimates. Now returns zeros immediately when signal energy < 1e-10. |
| FIX-4 | `detect_from_single_audio()` | Error chart was always blank — estimate was logged inside the drift guard. Now logged before it. |
| FIX-5 | `simulate_path_tracking_from_dataset()` | Same blank-chart issue as FIX-4, in the path tracking loop. |
| FIX-6 | `detect_from_single_audio()` | Short files (< segment duration) had segments starting past EOF, returning silence and 0 confidence. Hop is now clamped. |
| FIX-7 | `detect_and_localize_multi_drone()` | NaN band strengths sorted to top of candidates. Silent/NaN bands are now skipped. |
| FIX-8 | `run_noise_robustness_test()` | "No test drone clips" error after Colab disconnect. Now synthesises clips using FIX-1 delays if the test directory is empty. |

---

## 🏋️ Training

Training runs in Google Colab using the provided notebook.

```python
# In Colab — paste drone_detection_v2.py source, then:
config = Config()
main(config, num_epochs=20)
```

**Dataset:** GitHub DroneAudioDataset + optional Freesound scraping  
**Model:** 3-layer CNN (32→64→128 filters) + AdaptiveAvgPool + 2-class head  
**Input:** 3-channel mel spectrogram, 64 mel bands × 259 time frames  
**Training results:**

| Metric | Value |
|--------|-------|
| Best validation accuracy | 98.83% (epoch 15) |
| Final test accuracy | 98.0% |
| Test set | 600 samples (400 non-drone, 200 drone) |
| Drone precision / recall | 0.98 / 0.96 |

The trained model is saved to `/content/drive/MyDrive/drone_project/models/best_model.pth`. Copy it to `deployment/v2/models/` before starting the server.

---

## 📐 Microphone Array

Default configuration — equilateral triangle, 20 cm spacing:

```
Mic 1: [0.00, 0.00]  m  (origin / reference)
Mic 2: [0.20, 0.00]  m  (20 cm right)
Mic 3: [0.10, 0.17]  m  (apex of equilateral triangle)
```

**Reliable localization zone: ≤ 2.5 m from array centre.**  
Beyond this range, TDOA differences fall below ~0.6 ms — smaller than GCC-PHAT resolution at 22 050 Hz.

For multi-drone detection, drones need:
- Distinct fundamental motor frequencies (≥ 20 Hz apart)
- Physical separation > 0.3 m

---

## ⚙️ Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `SECRET_KEY` | `drone-v2-dev-secret` | Flask session secret — change in production |
| `MODEL_PATH` | `models/best_model.pth` | Override model file path |
| `PYTHONPATH` | — | Must include repo root for imports |

Server options:

```bash
python run_server_v2.py --port 5001       # change port
python run_server_v2.py --host 0.0.0.0   # bind all interfaces
python run_server_v2.py --no-debug       # disable reloader (production)
```

---

## 🔄 Running v1 and v2 Together

Both deployments share the same model file but have completely independent Flask apps and state:

```bash
# Terminal 1 — v1 (original)
cd deployment/v1
python run_server.py        # → http://localhost:5000

# Terminal 2 — v2 (improved)
cd deployment/v2
python run_server_v2.py     # → http://localhost:5001/v2
```

---

## 🎤 Real-time Monitoring (Optional)

For live microphone capture, install PyAudio:

```bash
# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin && pipwin install pyaudio
```

If PyAudio is not installed the server starts normally — only the real-time mic capture is disabled. All file-upload and simulation endpoints continue to work.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: drone_detection_v2` | Add repo root to `PYTHONPATH` (see Step 3 of Quick Start) |
| `Address already in use` | Use a different port: `python run_server_v2.py --port 5002` |
| Model not loaded warning | Copy `best_model.pth` to `deployment/v2/models/` |
| `No test drone clips` | This is fixed in v2 (FIX-8) — synthetic clips are generated automatically |
| Position always returns 15 m | Ensure `drone_detection_v2_fixes.py` patches are applied (FIX-1 + FIX-2) |
| WebSocket not connecting | Check that `flask-socketio` and `python-socketio` are the same major version |
| PyAudio install fails | Install `portaudio` system library first (see Real-time Monitoring above) |

---

## 📚 Tech Stack

- **PyTorch** — CNN model training and inference
- **Flask + Flask-SocketIO** — REST API and WebSocket server
- **NumPy / SciPy** — GCC-PHAT, Nelder-Mead localization, fractional sinc delay
- **Librosa / SoundFile** — audio loading and feature extraction
- **Leaflet + OpenStreetMap** — geographical map in the v1 monitoring UI
- **Plotly** — localization visualization

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📞 Support

- **Issues:** [github.com/Raphtildai/drone-detection-api/issues](https://github.com/Raphtildai/drone-detection-api/issues)
- **Email:** raphael@tildaitech.co.ke

> This project is for research and educational purposes. Always comply with local regulations regarding drone detection and airspace monitoring.