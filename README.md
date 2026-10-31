# Integrated Drone Detection, Localization and Path Tracking using Deep Learning

Source code for the MSc thesis (TDK paper) of **Kipchirchir Raphael**,
Eötvös Loránd University, Faculty of Informatics, Department of Numerical Analysis.
Supervisor: Dr. Bognár Gergő.

The system detects drones passively from their sound with a 2D CNN on mel-spectrograms.
For confident detections it localizes them with GCC-PHAT time differences and a
localization CNN, then tracks them with a Kalman filter. Training combines the public
DroneAudioDataset with physics-informed synthetic multi-channel samples, and the system is
evaluated on UaVirBASE and on outdoor field recordings.

## Repository layout

```
deployment/v2/
├── drone_detection/        The thesis pipeline (Python package)
├── dataset_builders/       Detection-dataset construction (clip extraction, purity segmentation, mixing)
├── experiments/            Additional experiments for the paper (STFT sensitivity study)
├── Training files/         Stand-alone Colab training script (v15, predecessor of the package)
├── dunakeszi_*.py          Field-recording ground truth, segment extraction and preparation
├── dunakeszi_data/         Field-recording analysis and benchmarks
├── models/                 Trained checkpoints (best_detection.pth, best_localization.pth)
├── logs/                   Training logs behind the reported results
├── test_dataset/           Synthetic test-suite results
└── app_v2.py, templates/   Web dashboard and API (see "Dashboard" below)
```

## Thesis → code

| Thesis section | Code |
|---|---|
| System Architecture | `drone_detection/inference.py` (`run_pipeline`), `drone_detection/orchestration.py` |
| Detection Dataset: Real Dataset | `drone_detection/datasets.py` (`DroneAudioDatasetManager`), `drone_detection/dataset_builder.py`, `dataset_builders/drone_urls.txt`, `none_drone_url.txt` |
| Purity-Aware Segmentation | `dataset_builders/custom_drone_dataset_builder.py` |
| Augmentation with Background Mixing | `drone_detection/orchestration.py` (`generate_mixed_drone_training_audio`), `dataset_builders/sort_urbansound8k.py` |
| Localization Dataset: UaVirBASE | `drone_detection/datasets.py` (`UaVirBASEDatasetManager`), `dataset_builders/test_dataset_downloader.py`, `uavirbase_patch.py` |
| Physics-Informed Synthetic Augmentation | `drone_detection/audio_processing.py` (`synthesise_drone`), `drone_detection/datasets.py` (`SyntheticLocDatasetV2`, `MelCacheManager.inject_synthetic`) |
| Audio Preprocessing: Feature Extraction | `drone_detection/audio_processing.py` (`mel_power`, `pcen`, `feature_stack`), parameters in `drone_detection/config.py` |
| CNN Detection Architecture | `drone_detection/models.py` (`DetectionCNN`) |
| Loss Function, Training Procedure | `drone_detection/models.py` (`FocalLoss`), `drone_detection/training.py` (`WarmupCosineScheduler`, `DetectionTrainer`, `LocalizationTrainer`) |
| Inference-Time Hybrid Detection | `drone_detection/inference.py` (`heuristic_detect`, `detect`) |
| Microphone Array Geometry | `drone_detection/config.py` (`set_array_geometry`) |
| Single-Drone Localization | `drone_detection/utils.py` (`gcc_phat`), `drone_detection/models.py` (localization CNN), `drone_detection/inference.py` (`localize`) |
| Multi-Drone Localization | `drone_detection/multidrone.py`, `drone_detection/multi_drone_audio_demo.py` |
| Kalman tracking | `drone_detection/tracking.py` |
| Single-Channel (MEMS) Detection and Spectral Proxy | `drone_detection/mems_inference.py` |
| Synthetic Test Suite, Scenario Dataset | `drone_detection/synthetic_dataset_generator.py`, `drone_detection/batch_testing.py`, `drone_detection/inference_test_loader.py`, results in `test_dataset/*_results.csv` |
| Real-World 3-Channel Validation (hover, circle) | `dunakeszi_ground_truth.py`, `dunakeszi_segment_extractor.py`, `prepare_dunakeszi_for_pipeline.py`, `drone_detection/dunakeszi_segment_analysis_inference.py`, `drone_detection/run_dunakeszi_local.py` |
| Figures | `drone_detection/visualization.py`, `visualization_dark.py`, `dataset_visualizer.py`, `build_training_datasets_and_figures.py`, `dataset_builders/generate_thesis_figures.py` |
| Training results (122 detection / 50 localization epochs) | `logs/detection_log.csv`, `logs/localization_log.csv` |

## Setup

```bash
cd deployment/v2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

All commands below run from `deployment/v2`.

## Training

The notebooks `drone_detection/pipeline_colab.ipynb` (Colab) and `drone_detection/pipeline.ipynb`
(local) run the full pipeline. The same steps from Python:

```python
from drone_detection import train_detection, train_localization

train_detection()      # downloads/builds the detection data, builds the feature cache, trains
train_localization()   # UaVirBASE + synthetic multi-channel samples
```

Checkpoints are written to `models/`, and per-epoch logs to `logs/`. All hyperparameters are in
`drone_detection/config.py`.

## Paper experiments

`experiments/stft_sensitivity.py` retrains the detection CNN while varying one STFT front-end
parameter at a time (FFT size, hop, mel bands, maximum frequency, feature channels):

```bash
python experiments/stft_sensitivity.py --prepare
python experiments/stft_sensitivity.py --epochs 30 --seeds 42 43 44
```

## Dashboard (not part of the evaluated system)

`app_v2.py` is a Flask/Socket.IO web dashboard built on top of the pipeline for demonstration:
single-file and 3-microphone detection, simulated and live streams, repository playback, batch
scans with ground-truth metrics, and a detection history. The thesis does not describe it.

```bash
python run_server_v2.py        # http://localhost:5001
```

Deployment to Google Cloud Run (`Dockerfile`) and other details are in
[`deployment/v2/README.md`](deployment/v2/README.md).

## Data

- **DroneAudioDataset** (Al-Emadi et al.) and **UaVirBASE** are public and downloaded by the code.
- The **Dunakeszi outdoor field recordings** (multi-drone measurement of 20 October 2025, including
  GPX telemetry and MEMS recordings) were provided by Prof. János Volk and the Nanosensors Lab,
  HUN-REN Centre for Energy Research, and are not included in this repository. With access, set
  `NEXTCLOUD_BASE_URL` and `NEXTCLOUD_SHARE_TOKEN`, or point `DUNAKESZI_LOCAL_PATH` /
  `DUNAKESZI_LOCAL_POLYWAV_DIR` at a local copy.
- Optional: `FREESOUND_API_KEY` for downloading additional background clips.
