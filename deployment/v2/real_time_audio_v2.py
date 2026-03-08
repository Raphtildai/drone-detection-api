"""
real_time_audio_v2.py — Real-time drone detection for v2 deployment
=====================================================================
Standalone module — does NOT import from real_time_audio.py (v3).

Uses PyAudio for microphone capture.  If PyAudio is unavailable the class
degrades gracefully: is_monitoring stays False and start_monitoring() logs
a warning instead of crashing.

v2 improvements over v3:
  - Uses detect_from_single_audio() with fractional-delay synthesis
  - Reports cap_hit flag from localize_precise()
  - Callback receives full result dict, not just position
"""

import os
import time
import queue
import logging
import threading
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

log = logging.getLogger("drone_v2.realtime")

try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False
    log.warning("PyAudio not installed — real-time microphone capture disabled. "
                "Install with: pip install pyaudio")


class RealTimeDroneDetectorv2:
    """
    Continuously captures audio from 1 or 3 microphones and runs v2 detection.

    Parameters
    ----------
    config          : v2 Config object
    channel_count   : 1 (single mic) or 3 (3-mic array for localization)
    segment_dur     : seconds per detection window (default 3.0)
    threshold       : drone classification threshold (default 0.70)
    device_indices  : list of PyAudio device indices for 3-mic mode
    """

    CHUNK       = 4096
    FORMAT      = pyaudio.paFloat32 if _PYAUDIO_OK else None
    CHANNELS    = 1

    def __init__(self, config, channel_count=1, segment_dur=3.0,
                 threshold=0.70, device_indices=None):
        self.config        = config
        self.channel_count = channel_count
        self.segment_dur   = segment_dur
        self.threshold     = threshold
        self.device_indices= device_indices or []
        self.is_monitoring = False
        self._thread       = None
        self._stop_event   = threading.Event()
        self._audio_queue  = queue.Queue(maxsize=50)
        self._pa           = None

    # ── Public API ─────────────────────────────────────────────────────────
    def start_monitoring(self, callback):
        """
        Start background monitoring thread.
        callback(result: dict) is called on every detection event.
        result keys: detected, confidence, position, reliable, cap_hit,
                     timestamp, segment_index
        """
        if not _PYAUDIO_OK:
            log.warning("PyAudio unavailable — cannot start real-time monitoring")
            return

        if self.is_monitoring:
            log.warning("Monitoring already active")
            return

        self._stop_event.clear()
        self.is_monitoring = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback,),
            daemon=True,
            name="drone-v2-monitor",
        )
        self._thread.start()
        log.info(f"v2 real-time monitoring started  "
                 f"(channels={self.channel_count}, seg={self.segment_dur}s, "
                 f"threshold={self.threshold})")

    def stop_monitoring(self):
        """Stop monitoring and release audio resources."""
        self._stop_event.set()
        self.is_monitoring = False
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        log.info("v2 real-time monitoring stopped")

    # ── Internal ────────────────────────────────────────────────────────────
    def _monitor_loop(self, callback):
        sr           = self.config.SR
        seg_samples  = int(self.segment_dur * sr)
        buffer       = np.zeros(seg_samples, dtype=np.float32)
        filled       = 0
        seg_idx      = 0

        try:
            self._pa = pyaudio.PyAudio()
            stream = self._pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=sr,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            log.info("Audio stream opened")

            while not self._stop_event.is_set():
                raw = stream.read(self.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32)

                space = seg_samples - filled
                take  = min(len(chunk), space)
                buffer[filled:filled + take] = chunk[:take]
                filled += take

                if filled >= seg_samples:
                    # Process this segment
                    self._process_segment(buffer.copy(), sr, seg_idx, callback)
                    seg_idx += 1
                    # Slide: 50% overlap
                    half = seg_samples // 2
                    buffer[:half] = buffer[half:seg_samples]
                    filled = half

            stream.stop_stream()
            stream.close()

        except Exception as e:
            log.exception(f"Monitor loop error: {e}")
        finally:
            self.is_monitoring = False

    def _process_segment(self, audio_seg, sr, seg_idx, callback):
        """Write segment to temp WAV, run v2 detection, fire callback."""
        tmp = None
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tf.name, audio_seg, sr)
            tmp = tf.name

            from drone_detection_v2 import detect_from_single_audio
            result = detect_from_single_audio(
                tmp, self.config,
                threshold=self.threshold,
                n_segments=1,
                show_plot=False,
            )

            if result["detected"]:
                pos      = result.get("position")
                segs     = result.get("segments", [{}])
                loc_info = segs[0] if segs else {}
                callback({
                    "detected":      True,
                    "confidence":    result["probability"],
                    "position":      pos.tolist() if hasattr(pos, "tolist") else pos,
                    "reliable":      loc_info.get("reliable", False),
                    "cap_hit":       "CAP HIT" in loc_info.get("quality_message", ""),
                    "timestamp":     time.time(),
                    "segment_index": seg_idx,
                })

        except Exception as e:
            log.debug(f"Segment {seg_idx} processing error: {e}")
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass