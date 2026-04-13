# Drone Detection System v2 — Performance Report
### Pipeline: `drone_detection_v15`  |  Generated from training logs and thesis evaluation

---

## 1. Detection Model (CNN + Heuristic Hybrid)

### Training results (epoch 15, best checkpoint)

| Metric | Value |
|--------|-------|
| Best validation accuracy | **98.83 %** |
| Final test accuracy | **98.0 %** |
| Test set size | 600 samples (400 non-drone, 200 drone) |
| Drone precision | 0.98 |
| Drone recall | 0.96 |
| F1 score | 0.97 |
| Auto-selected threshold | **0.62** (F1-maximising on val set) |

### v15 training improvements vs v2-original

| Change | Impact |
|--------|--------|
| Focal loss (γ=2, α=0.6) replaces CrossEntropy | Reduces penalty from easy negatives; improves minority-class recall |
| Label smoothing 0.02 | Reduces overconfidence on ambiguous clips |
| CosineAnnealingLR with 3-epoch warmup | Smoother convergence, avoids early spikes |
| `feature_stack()`: [log-mel, PCEN, delta-mel] | PCEN compresses dynamic range; delta captures temporal change |
| Focal loss threshold search on val set | Threshold 0.62 vs fixed 0.70 → +3 % recall on edge cases |
| Mixed drone + background augmentation (1 200 clips) | +4 % accuracy on challenging non-drone backgrounds |
| Grouped train/val/test split (by recording session) | Prevents data leakage between splits |

### Noise robustness (SNR sweep)

| SNR (dB) | Detection rate | Avg confidence |
|----------|---------------|----------------|
| -5       | 42 %          | 0.51           |
| 0        | 61 %          | 0.63           |
| 5        | 78 %          | 0.74           |
| 10       | 89 %          | 0.83           |
| 15       | 94 %          | 0.89           |
| 20       | 97 %          | 0.93           |

**Tolerable noise floor: ≥ 0 dB SNR (≥ 61 % detection rate)**

---

## 2. Localization Model (3-mic TDOA + Nelder-Mead)

### UaVirBASE evaluation (real held-out sessions)

| Split | n | Az MAE | Az RMSE | Az Median | Dist MAE | Ht MAE |
|-------|---|--------|---------|-----------|----------|--------|
| Val (real) | — | **38.1°** | 52.3° | 24.1° | 4.28 m | 5.34 m |
| Test (real) | — | **47.6°** | 56.2° | 46.1° | 3.86 m | 4.60 m |

### Per-azimuth sector performance (test set)

| Direction | Az MAE | Dist MAE | Notes |
|-----------|--------|----------|-------|
| 0° (North) | 18.6° | 6.48 m | Best — symmetric array geometry |
| 45° | 42.0° | 6.07 m | Moderate |
| 90° (East) | 56.9° | 7.14 m | TDOA ambiguity at perpendicular |
| 135° | 35.5° | 4.70 m | Moderate |
| 180° (South) | 10.0° | 2.66 m | Best — mirror of 0° |
| 225° | 32.1° | 5.92 m | Moderate |
| 270° (West) | 69.9° | 4.70 m | Worst — symmetric ambiguity |
| 315° | 56.4° | 4.72 m | Poor |

**Random baseline: 90° MAE.  Overall mean: 47.6° vs 90° random — 47 % better than chance.**

### Reliable localization zone
- Within **2.5 m** of array centre: reliable (TDOA resolution adequate at 22 050 Hz)
- Beyond 2.5 m: TDOA difference < 0.6 ms → estimates become unreliable

---

## 3. Multi-Drone Detection — v15 Cartesian Solver vs Original

### Bug fixes applied (multidrone_localization_patch_v2.py)

| Bug | Original behaviour | v15 fix | Impact |
|-----|--------------------|---------|--------|
| TDOA dedup window | 0.05 µs (580× too tight) | 29 µs (5 % of physical resolution) | Eliminates spurious duplicate candidates |
| Distance saturation | Always saturates at 25 m | Cartesian `(x,y)` solver + soft barrier | Correct distances 0.3–25 m |
| `r ≈ 0` degenerate solutions | Optimizer collapses to array centre | Soft inner penalty + `MIN_SOLUTION_DIST = 0.3 m` | Azimuth MAE drops from ~85° to ~40°–50° |
| Confidence radius overflow | Always capped at 20.0 m | Finite-diff Hessian, clamped [0.05, 25] m | Meaningful uncertainty estimate |

### Synthetic multi-drone test suite results (8 scenarios)

| Scenario | Drones | Det. prob | Detected | Az MAE | Dist MAE | Tracks |
|----------|--------|-----------|----------|--------|----------|--------|
| single_near [2.0, 0.5] | 1 | 0.95 | ✅ | 18° | 0.4 m | 1 |
| single_far [8.0, 5.0] | 1 | 0.87 | ✅ | 35° | 1.2 m | 1 |
| two_drones_separated | 2 | 0.93 | ✅ | 42° | 1.8 m | 2 |
| two_drones_close | 2 | 0.88 | ⚠️ (1/2) | 55° | 2.1 m | 1 |
| three_drones_triangle | 3 | 0.91 | ⚠️ (2/3) | 51° | 1.9 m | 2 |
| two_drones_opposite | 2 | 0.90 | ✅ | 38° | 1.5 m | 2 |
| single_very_near | 1 | 0.98 | ✅ | 12° | 0.2 m | 1 |
| two_drones_noisy | 2 | 0.79 | ⚠️ (1/2) | 68° | 3.2 m | 1 |

**Summary: 8/8 detected at least 1 drone; 5/8 exact count; overall Az MAE ≈ 40°**

### Multi-drone limitations

- Drones need **distinct fundamental motor frequencies (≥ 20 Hz apart)** for reliable separation
- Physical separation > **0.3 m** required (below this, TDOA ambiguity makes separation impossible)
- At noise_level > 0.10, the second drone is frequently missed
- All multi-drone evaluation is synthetic — no simultaneous real multi-drone recordings were used in training

---

## 4. Real-Time Session

### Simulated mode (synthetic audio, identical pipeline)

| Parameter | Default | Notes |
|-----------|---------|-------|
| Tick rate | 1.0 Hz | Adjustable 0.2–3.0 Hz |
| Detection rate | ~90 % | At noise_level=0.04, within 2.5 m |
| Avg latency per frame | ~200 ms | On CPU; ~60 ms on GPU |
| Kalman match gate | 8.0 m | Raised from 2.0 m to handle TDOA noise |
| Track confirmation | 1 hit | Lowered from 2 to confirm tracks faster |

### Live mic mode

- Requires PyAudio + PortAudio installed on the server
- 3-second sliding window with 50 % overlap
- Same `detect() + localize()` pipeline as simulated mode
- Falls back gracefully if PyAudio is unavailable (file-upload endpoints still work)

---

## 5. Thesis Acknowledgements

The following limitations are noted for thesis transparency:

1. Dataset contains only 1 drone type recorded on a single day at a single outdoor location
2. All real localization positions are at exactly 10 m or 20 m distance — no near-field (< 10 m) real data
3. Only 8 discrete azimuths (multiples of 45°) — no real continuous azimuth ground truth
4. Multi-drone evaluation is entirely synthetic
5. Synthetic acoustic model approximates but does not fully capture real drone acoustics
   (missing airframe resonance, rotor–rotor interaction, turbulent inflow noise)
6. Microphone array geometry assumed from config — no independent calibration verification
7. Kalman tracker parameters are heuristic, not fitted to real trajectories

---

## 6. Deployment Fixes Active in This Release

| # | Fix | Function | Problem solved |
|---|-----|----------|---------------|
| 1 | Fractional sinc delay | `synthesise_drone()` | Integer-delay rounding caused ~45 µs TDOA error |
| 2 | Cap-hit guard | `localize()` | Wrong-branch solutions at 15 m boundary |
| 3 | NaN-safe GCC-PHAT | `gcc_phat()` | Silent bands produced NaN TDOA |
| 4/5 | Error chart populated | `detect()` | Error chart was always blank |
| 6 | Segment hop clamp | `detect()` | Short files returned silence + 0 confidence |
| 7 | NaN band skip | `localize_multi_drone()` | NaN band strengths sorted to top |
| 8 | Synthetic SNR fallback | `noise_test` endpoint | "No test clips" error after disconnect |
| 9 | Real-time sessions | `realtime_sessions.py` | Both simulated + live mic modes |
| 10 | CNN+heuristic hybrid | `detect()` (v15) | Heuristic veto for tonal non-drone audio |
| 11 | Cartesian Nelder-Mead | `localize_multi_drone_v2()` | Distance saturation at 25 m |
| 12 | TDOA dedup 29 µs | `localize_multi_drone_v2()` | 0.05 µs was 580× too tight |
| 13 | Kalman gate 8 m | `PathTracker` | 2 m gate too tight for TDOA noise |
| 14 | Focal loss + threshold search | Training pipeline | Better F1; threshold auto-calibrated |
| 15 | feature_stack [mel, PCEN, Δmel] | `AudioProcessor` (v15) | Richer features, better generalisation |