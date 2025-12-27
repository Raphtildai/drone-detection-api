# 🚁 Drone Audio Detection & Localization System

A full end-to-end **drone detection and localization system** based on audio signals.
This project supports **training, evaluation, inference, long-audio analysis, and interactive testing**, including **3-microphone localization using TDOA (GCC-PHAT)**.

---

## 📌 Features

### ✅ Drone Detection

* CNN-based classifier trained on **mel-spectrograms**
* Binary classification: **Drone vs Non-Drone**
* Supports:

  * Mono audio
  * 3-channel recordings
  * 3 separate microphone WAV files

### 📍 Drone Localization

* Uses **3 microphones** with known geometry
* Implements **GCC-PHAT** for TDOA estimation
* Grid-search + refinement for `(x, y)` localization
* Outputs estimated drone position in meters

### 🎧 Long Audio Analysis

* Automatically segments long recordings
* Detects drones appearing at **different time intervals**
* Returns:

  * Per-segment confidence
  * Best detection window
  * Summary statistics

### 🧪 Synthetic Data Augmentation

* Generates synthetic 3-channel drone sounds
* Injects synthetic data into training mel-cache
* Improves robustness and class balance

### 📊 Training & Monitoring

* Balanced dataset using **WeightedRandomSampler**
* Automatic checkpointing & resume
* TensorBoard integration (local + Colab)
* Confusion matrix & classification report

### 🧑‍💻 Interactive Testing

* Upload audio files via **IPyWidgets**
* Live playback + detection results
* Optional long-audio analysis toggle

---

## 🗂️ Project Structure

```
model_training_and_testing/
├── raw/                    # Raw downloaded dataset
├── processed/              # Train/val/test WAV splits
│   ├── train/
│   ├── val/
│   └── test/
├── mel_cache/              # Cached mel-spectrograms (.npy)
│   ├── train/
│   ├── val/
│   └── test/
├── drone_project_local/    # Models, logs, TensorBoard
│   ├── models/
│   ├── logs/
│   ├── tensorboard/
│   └── backup/
```

---

## ⚙️ Configuration

All configuration is centralized in the `Config` class:

```python
config = Config()
```

### Key Parameters

* **Sample Rate**: `22050 Hz`
* **Segment Duration**: `3.0 seconds`
* **Mel Bands**: `64`
* **Batch Size**: `32`
* **Synthetic Samples**: `2000`
* **Device**: CUDA (if available) or CPU

### Microphone Geometry

```python
MIC_POSITIONS = [
    [0.0, 0.0],
    [0.2, 0.0],
    [0.1, 0.2 * sqrt(3)]
]
```

---

## 📥 Dataset

The dataset is automatically downloaded from:

**DroneAudioDataset**
[https://github.com/saraalemadi/DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset)

Classes are normalized to:

* `drone`
* `non_drone`

Dataset is automatically:

* Downloaded
* Extracted
* Balanced
* Split (70% / 15% / 15%)

---

## 🏗️ Training Pipeline

### 1️⃣ Prepare Dataset

```python
dataset_mgr.prepare_dataset()
```

### 2️⃣ Generate Synthetic Data

```python
inject_synthetic_3ch_data()
```

### 3️⃣ Create Mel Cache

```python
mel_cache_mgr.create_mel_cache()
```

### 4️⃣ Train Model

```python
main(num_epochs=100)
```

Features:

* Resume training from checkpoint
* Early stopping
* AMP (mixed precision)
* Automatic best-model saving

---

## 🧠 Model Architecture

**SimpleDroneDetector**

* 3-channel CNN
* Conv → BatchNorm → ReLU → Pooling
* Adaptive pooling for size robustness
* Fully connected classifier

Output:

* Class 0: `non_drone`
* Class 1: `drone`

---

## 📍 Localization Method

1. **GCC-PHAT** computes TDOA:

   * Mic2 − Mic1
   * Mic3 − Mic1
2. **Grid search** over candidate `(x, y)`
3. **Refinement step** for accuracy
4. Outputs estimated position and error

---

## 🔍 Inference Usage

### Detect + Localize (Short Audio)

```python
detect_and_localize_if_drone(
    "mic1.wav", "mic2.wav", "mic3.wav",
    threshold=0.75
)
```

### Single 3-Channel File

```python
detect_and_localize_if_drone("recording_3channel.wav")
```

---

## ⏱️ Long Audio Detection

### Automatic Segmented Analysis

```python
quick_test_long_audio("long_recording.wav", threshold=0.7)
```

Returns:

* Segment-wise probabilities
* Best detection window
* Summary statistics

---

## 🧪 Interactive Testing (Notebook)

```python
test_my_file_enhanced()
```

Features:

* File upload
* Audio playback
* Long-audio toggle
* Live results

---

## 📊 TensorBoard

Automatically starts during training.

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir {config.DRIVE_TBOARD}
```

Tracks:

* Loss
* Accuracy
* Learning rate

---

## 🧬 Synthetic Test File Generation

```python
generate_test_files()
```

Creates:

* Clean drone
* Indoor noise
* Outdoor noise
* 3-channel recordings

---

## ✅ Checkpoints

* `best_model.pth` → best validation accuracy
* `final_model.pth` → last training state
* Automatically detected & loaded for inference

---

## 🚀 Entry Point

```bash
python main.py
```

Or in notebook:

```python
main(num_epochs=100)
```

---

## 🛡️ Notes & Assumptions

* Localization assumes **synchronized microphones**
* Best results with **3-channel audio**
* Long-audio localization currently uses best segment
* Grid search prioritizes robustness over speed

---

## 📜 License & Usage

This project is intended for:

* Research
* Education
* Prototyping
* Non-commercial use (dataset dependent)

---