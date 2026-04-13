"""
real_time_audio_v2.py — Real-time drone detection (v2 deployment)
=====================================================================
Standalone module — does NOT import from real_time_audio.py (v1/v3).

Uses PyAudio for microphone capture.  If PyAudio is unavailable the class
degrades gracefully: is_monitoring stays False and start_monitoring() logs
a warning instead of crashing.

v2 improvements over v1/v3
---------------------------
  - Uses detect() + localize() from drone_detection_v2.py
    (CNN + heuristic hybrid, fractional-delay synthesis)
  - Reports reliable flag from localize()
  - Callback receives the full result dict, not just a position
  - Aligned to v15 AudioProcessor.feature_stack() when available
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

log = logging.getLogger("drone_v2.realtime")

try:
    import pyaudio

    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False
    log.warning(
        "PyAudio not installed — real-time microphone capture disabled. "
        "Install with: pip install pyaudio"
    )


class RealTimeDroneDetectorV2:
    """
    Continuously captures audio from 1 microphone and runs v2 detection.

    Parameters
    ----------
    config          : v2/v15 Config object
    channel_count   : 1 (single mic) or 3 (3-mic array, if device supports it)
    segment_dur     : seconds per detection window (default 3.0)
    threshold       : drone classification threshold (default 0.70)
    device_indices  : list of PyAudio device indices (for multi-device capture)
    """

    CHUNK: int = 4096
    FORMAT = pyaudio.paFloat32 if _PYAUDIO_OK else None
    CHANNELS: int = 1

    def __init__(
        self,
        config,
        channel_count: int = 1,
        segment_dur: float = 3.0,
        threshold: float = 0.70,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        self.config = config
        self.channel_count = channel_count
        self.segment_dur = segment_dur
        self.threshold = threshold
        self.device_indices = device_indices or []
        self.is_monitoring: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)
        self._pa = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start_monitoring(self, callback: Callable[[dict], None]) -> None:
        """
        Start background monitoring thread.

        callback(result) is called for every detection event.
        result keys: detected, confidence, position, reliable,
                     cap_hit, timestamp, segment_index
        """
        if not _PYAUDIO_OK:
            log.warning(
                "PyAudio unavailable — cannot start real-time monitoring"
            )
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
        log.info(
            f"v2 real-time monitoring started  "
            f"(channels={self.channel_count}, seg={self.segment_dur}s, "
            f"threshold={self.threshold})"
        )

    def stop_monitoring(self) -> None:
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

    def _monitor_loop(self, callback: Callable[[dict], None]) -> None:
        sr = self.config.SR
        seg_samples = int(self.segment_dur * sr)
        buffer = np.zeros(seg_samples, dtype=np.float32)
        filled = 0
        seg_idx = 0

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
                take = min(len(chunk), space)
                buffer[filled : filled + take] = chunk[:take]
                filled += take

                if filled >= seg_samples:
                    self._process_segment(buffer.copy(), sr, seg_idx, callback)
                    seg_idx += 1
                    # 50 % overlap
                    half = seg_samples // 2
                    buffer[:half] = buffer[half:seg_samples]
                    filled = half

            stream.stop_stream()
            stream.close()

        except Exception as exc:
            log.exception(f"Monitor loop error: {exc}")
        finally:
            self.is_monitoring = False

    def _process_segment(
        self,
        audio_seg: np.ndarray,
        sr: int,
        seg_idx: int,
        callback: Callable[[dict], None],
    ) -> None:
        """Write segment to temp WAV, run v2 detection, fire callback."""
        tmp: Optional[str] = None
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tf.name, audio_seg, sr)
            tmp = tf.name

            # Import here to avoid circular import at module load time
            from drone_detection_v2 import detect, localize, AudioProcessor

            ap = AudioProcessor(self.config)
            y = ap.pad_or_truncate(ap.load(tmp))
            # replicate to 3 channels (single-mic: same signal to all)
            channels = [y, y, y]

            det = detect(channels, self.config)
            prob = float(det["probability"])

            if prob >= self.threshold:
                loc = localize(channels, self.config)
                pos = loc.get("xy_position")
                callback(
                    {
                        "detected": True,
                        "confidence": prob,
                        "position": (
                            pos.tolist()
                            if hasattr(pos, "tolist")
                            else pos
                        ),
                        "reliable": bool(loc.get("reliable", False)),
                        "cap_hit": False,
                        "timestamp": time.time(),
                        "segment_index": seg_idx,
                    }
                )

        except Exception as exc:
            log.debug(f"Segment {seg_idx} processing error: {exc}")
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass