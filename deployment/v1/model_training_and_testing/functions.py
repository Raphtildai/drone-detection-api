# functions.py

import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf

from config import Config
import config
from model import SimpleDroneDetector
from synthetic import generate_synthetic_drone
from audio_utils import load_mono_audio, save_temp_wav

# ==================== DRONE DETECTION SYSTEM ====================

class DroneDetectionSystemFunctions:
    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        self.device = None

    # -------------------- MODEL LOADING --------------------
    def load_best_model(self):
        """Load the best model from training, considering environment"""
        if self.model is not None:
            return self.model

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Find the latest checkpoint depending on environment
        model_path = self.get_latest_checkpoint()
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError("No checkpoint found! Train the model first.")
        print(f"Loading best model: {model_path.name}")

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = SimpleDroneDetector(in_channels=3).to(self.device)

        # Handle old/new checkpoint formats
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            epoch = ckpt.get('epoch', 'unknown')
            best_val_acc = ckpt.get('best_val_acc', 'unknown')
            print(f"✅ Model loaded (Epoch: {epoch}, Best Val Acc: {best_val_acc})")
        else:
            self.model.load_state_dict(ckpt)
            print("✅ Model loaded (old format)")

        self.model.eval()
        return self.model

    # -------------------- ENVIRONMENT-AWARE CHECKPOINT --------------------
    def get_latest_checkpoint(self):
        """Check environment and return the latest checkpoint"""
        # Check Google Colab / Drive
        if 'COLAB_GPU' in os.environ or Path('/content/drive/MyDrive/models').exists():
            drive_path = Path('/content/drive/MyDrive/models')
            checkpoint = self._latest_checkpoint_in_dir(drive_path)
            if checkpoint:
                print("✅ Found checkpoint in Google Drive")
                return checkpoint

        # Check local drive folder
        if self.config.MODELS_DIR.exists():
            checkpoint = self._latest_checkpoint_in_dir(self.config.MODELS_DIR)
            if checkpoint:
                print("✅ Found checkpoint in local MODELS_DIR folder")
                return checkpoint

        # Fallback: current working directory
        cwd = Path('.')
        checkpoint = self._latest_checkpoint_in_dir(cwd)
        if checkpoint:
            print("✅ Found checkpoint in current working directory")
            return checkpoint

        # No checkpoint found
        return None

    def _latest_checkpoint_in_dir(self, directory: Path):
        """Return latest .pth or .pt checkpoint in a directory"""
        files = list(directory.glob("*.pth")) + list(directory.glob("*.pt"))
        if not files:
            return None
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]

    # -------------------- AUDIO PREPROCESSING --------------------
    def preprocess_audio_file(self, wav_path):
        y, _ = librosa.load(str(wav_path), sr=self.config.SR, mono=True)
        target_len = int(self.config.SR * self.config.TARGET_DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y, sr=self.config.SR, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH, n_mels=self.config.N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.astype(np.float32)

    # -------------------- GCC-PHAT FUNCTION --------------------    
    def gcc_phat(self, sig, refsig, fs=None, max_tau=None, interp=16):
        """
        Compute GCC-PHAT between sig and refsig.
        Returns estimated TDOA (seconds) and cross-correlation array (lags, cc).
        """
        if fs is None:
            fs = self.config.SR

        sig = sig.flatten()
        refsig = refsig.flatten()

        n = sig.size + refsig.size

        # FFT
        SIG = np.fft.rfft(sig, n=n)
        REF = np.fft.rfft(refsig, n=n)
        R = SIG * np.conj(REF)
        denom = np.abs(R)
        denom[denom == 0] = 1e-8
        R /= denom

        # Interpolated IFFT
        cc = np.fft.irfft(R, n=interp*n)

        max_shift = int(interp*n/2)
        if max_tau is not None:
            max_shift = min(max_shift, int(interp*fs*max_tau))

        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

        # Compute lags
        lags = np.arange(-max_shift, max_shift+1) / float(interp * fs)

        # Find peak
        shift = np.argmax(np.abs(cc)) - max_shift
        tau = shift / float(interp * fs)

        return tau, lags, cc

    # -------------------- TDOA ESTIMATION HELPERS --------------------
    def estimate_tdoas_from_three_wavs(self, wav1_path, wav2_path, wav3_path):
        """
        Returns TDOAs relative to mic1: tau12 (mic2 - mic1), tau13 (mic3 - mic1)
        """
        # load with same sr
        y1, sr = librosa.load(str(wav1_path), sr=self.config.SR, mono=True)
        y2, _  = librosa.load(str(wav2_path), sr=self.config.SR, mono=True)
        y3, _  = librosa.load(str(wav3_path), sr=self.config.SR, mono=True)
        # ensure equal length by trimming to target duration
        target_len = int(self.config.SR * self.config.TARGET_DURATION)
        def fixlen(y):
            if len(y) < target_len:
                return np.pad(y, (0, target_len - len(y)))
            else:
                return y[:target_len]
        y1 = fixlen(y1); y2 = fixlen(y2); y3 = fixlen(y3)
        max_tau = 0.01  # 10 ms max expected TDOA for small mic spacing
        tau12, _, _ = self.gcc_phat(y2, y1, fs=sr, max_tau=max_tau)
        tau13, _, _ = self.gcc_phat(y3, y1, fs=sr, max_tau=max_tau)
        return np.array([tau12, tau13])
    
    # -------------------- TDOA ESTIMATION --------------------
    def tdoa_error_for_position(self, pos, measured_tdoas, mic_positions=None, c=None):
        """
        pos: (2,) candidate (x,y) - can be list or numpy array
        measured_tdoas: [tau12, tau13] (seconds) (relative to mic1)
        returns squared error
        """
        if mic_positions is None: # Mic Positions
            mic_positions = self.config.MIC_POSITIONS
        if c is None: # Speed of sound
            c = self.config.SPEED_OF_SOUND

        # Ensure pos is numpy array
        pos = np.array(pos)

        # compute theoretical TDOAs relative to mic1
        dists = np.linalg.norm(mic_positions - pos[None, :], axis=1)  # shape (3,)
        # arrival times
        times = dists / c
        # relative to mic1:
        tau12_theo = times[1] - times[0]
        tau13_theo = times[2] - times[0]
        err = (tau12_theo - measured_tdoas[0])**2 + (tau13_theo - measured_tdoas[1])**2
        return err

    # -------------------- TDOA ESTIMATION FROM WAVS --------------------
    def localize_from_three_wavs(self, wav1_path, wav2_path, wav3_path, mic_positions=None, grid_radius=5.0, grid_res=0.02):
        """
        Main localization function (grid search + refinement).
        - wav paths are 3 files recorded simultaneously (or single multi-channel wav split into channels)
        - mic_positions shape (3,2)
        Returns (x,y) in meters relative to mic coordinate system.
        """
        if mic_positions is None:
            mic_positions = self.config.MIC_POSITIONS
        measured_tdoas = self.estimate_tdoas_from_three_wavs(wav1_path, wav2_path, wav3_path)
        # coarse grid search within bounding box around mics
        # compute bounding box: min/max mic coords +- margin
        xs = mic_positions[:,0]; ys = mic_positions[:,1]
        xmin, xmax = xs.min() - grid_radius, xs.max() + grid_radius
        ymin, ymax = ys.min() - grid_radius, ys.max() + grid_radius
        # coarse grid
        gx = np.arange(xmin, xmax, grid_res)
        gy = np.arange(ymin, ymax, grid_res)
        best = None
        best_err = 1e12
        for xx in gx:
            # vectorized across gy for speed
            pts = np.stack([np.full_like(gy, xx), gy], axis=1)
            # compute distances & times
            dists = np.linalg.norm(mic_positions[None,:,:] - pts[:,None,:], axis=2)  # (G,3)
            times = dists / self.config.SPEED_OF_SOUND
            tau12_theo = times[:,1] - times[:,0]
            tau13_theo = times[:,2] - times[:,0]
            err = (tau12_theo - measured_tdoas[0])**2 + (tau13_theo - measured_tdoas[1])**2
            idx = np.argmin(err)
            if err[idx] < best_err:
                best_err = err[idx]
                best = pts[idx]
        # local refinement via simple Nelder-Mead-like pattern search
        pos = best.copy()
        step = grid_res
        for _ in range(20):
            improved = False
            candidates = [pos,
                        pos + np.array([ step, 0.0]),
                        pos + np.array([-step, 0.0]),
                        pos + np.array([0.0,  step]),
                        pos + np.array([0.0, -step])]
            for cand in candidates:
                e = self.tdoa_error_for_position(cand, measured_tdoas, mic_positions)
                if e < best_err:
                    best_err = e
                    pos = cand
                    improved = True
            if not improved:
                step *= 0.5
            if step < 1e-4:
                break
        return pos, best_err
    
    # -------------------- LOCALIZATION --------------------
    def localize_now(self, wav1, wav2=None, wav3=None):
        if wav2 is None:
            data, sr = sf.read(wav1)
            if data.ndim == 1:
                data = data[:, None]
            data = data.T
            if data.shape[0] < 3:
                data = np.tile(data[0], (3, 1))
            ch1, ch2, ch3 = data[0], data[1], data[2]

            # Create temp files
            f1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f3 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

            try:
                sf.write(f1.name, ch1, sr)
                sf.write(f2.name, ch2, sr)
                sf.write(f3.name, ch3, sr)

                # Close the file handles so Windows allows deletion later
                f1.close()
                f2.close()
                f3.close()

                pos, err = self.localize_from_three_wavs(f1.name, f2.name, f3.name)

            finally:
                # Safe deletion
                os.unlink(f1.name)
                os.unlink(f2.name)
                os.unlink(f3.name)

        else:
            pos, err = self.localize_from_three_wavs(wav1, wav2, wav3)

        print(f"📍 Localization: x = {pos[0]:.3f} m, y = {pos[1]:.3f} m (error={err:.2e})")
        return pos

    # -------------------- DETECTION --------------------
    def detect_and_localize_if_drone(self, wav1, wav2=None, wav3=None, threshold=0.75):
        self.load_best_model()
        with torch.no_grad():
            if wav2 is not None:
                mels = [
                    self.preprocess_audio_file(wav1),
                    self.preprocess_audio_file(wav2),
                    self.preprocess_audio_file(wav3)
                ]
            else:
                data, _ = sf.read(wav1)
                data = data.T if data.ndim == 2 else data[:, None].T
                mels = []
                for i in range(min(3, data.shape[0])):
                    y = data[i]
                    target_len = int(self.config.SR * self.config.TARGET_DURATION)
                    if len(y) < target_len:
                        y = np.pad(y, (0, target_len - len(y)))
                    else:
                        y = y[:target_len]
                    mel = librosa.feature.melspectrogram(
                        y=y, sr=self.config.SR, n_fft=self.config.N_FFT,
                        hop_length=self.config.HOP_LENGTH, n_mels=self.config.N_MELS
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
                    mels.append(mel_db.astype(np.float32))
                while len(mels) < 3:
                    mels.append(mels[0])

            mel_tensor = torch.tensor(np.stack(mels), dtype=torch.float32).unsqueeze(0)
            mel_tensor = mel_tensor.to(self.device)

            logits = self.model(mel_tensor)
            prob_drone = torch.softmax(logits, dim=1)[0, 1].item()
            print(f"🎯 Drone probability: {prob_drone:.3f}")

            if prob_drone >= threshold:
                print("🚁 DRONE DETECTED → Running localization...")
                pos = self.localize_now(wav1, wav2=wav2, wav3=wav3)
                return {"detected": True, "probability": prob_drone, "position": pos}
            else:
                print("🌳 No drone detected.")
                return {"detected": False, "probability": prob_drone}

    # -------------------- LONG AUDIO ANALYSIS --------------------
    def analyze_long_audio(self, wav_path, analysis_segments=10, threshold=0.75):
        print(f"🔍 Analyzing long audio file: {wav_path}")
        y, sr = load_mono_audio(wav_path)
        duration = len(y) / sr
        print(f"⏱️ Audio duration: {duration:.1f} seconds")

        segment_duration = self.config.TARGET_DURATION
        hop_duration = max(1.0, (duration - segment_duration) / analysis_segments)

        segments, detections = [], []

        for i in range(analysis_segments):
            start_time = i * hop_duration
            start_sample = int(start_time * sr)
            end_sample = start_sample + int(segment_duration * sr)
            if end_sample > len(y):
                break

            temp_path = save_temp_wav(y[start_sample:end_sample], sr)
            try:
                result = self.detect_and_localize_if_drone(temp_path, temp_path, temp_path, threshold=threshold)
                segment_info = {
                    "segment": i + 1,
                    "start_time": start_time,
                    "end_time": start_time + segment_duration,
                    "probability": result["probability"],
                    "detected": result["detected"]
                }
                segments.append(segment_info)
                detections.append(result["probability"])
                os.unlink(temp_path)
            except Exception as e:
                print(f"Segment {i+1} Error: {e}")
                os.unlink(temp_path)

        if detections:
            max_prob = max(detections)
            avg_prob = np.mean(detections)
            detection_count = sum(1 for seg in segments if seg["detected"])
            overall_detected = max_prob >= threshold
            detected_segments = [seg for seg in segments if seg["detected"]]
            best_segment = max(detected_segments, key=lambda x: x["probability"]) if detected_segments else None

            return {
                "detected": overall_detected,
                "probability": max_prob,
                "segments": segments,
                "best_segment": best_segment,
                "detection_summary": {
                    "total_segments": len(segments),
                    "detected_segments": detection_count,
                    "max_confidence": max_prob,
                    "average_confidence": avg_prob
                }
            }
        return {"detected": False, "probability": 0.0, "segments": []}

    # -------------------- CHECK LONG AUDIO --------------------
    def detect_and_localize_if_drone_enhanced(self, wav1, wav2=None, wav3=None, threshold=0.75, analyze_long=False):
        if analyze_long and wav2 is None:
            y, sr = load_mono_audio(wav1)
            if len(y) / sr > self.config.TARGET_DURATION * 2:
                return self.analyze_long_audio(wav1, analysis_segments=10, threshold=threshold)
        return self.detect_and_localize_if_drone(wav1, wav2, wav3, threshold=threshold)

    # -------------------- TEST FILE GENERATION --------------------
    def generate_test_files(self):
        mic_pos = self.config.MIC_POSITIONS
        print("🎵 Generating synthetic test data...")
        chs = generate_synthetic_drone(mic_pos, [0.85, 0.30])
        for idx, f in enumerate(["mic1.wav", "mic2.wav", "mic3.wav"]):
            sf.write(f, chs[idx], self.config.SR)
        sf.write("recording_3channel.wav", np.column_stack([chs[0], chs[1], chs[2]]), self.config.SR)
        print("✅ All test files generated!")

