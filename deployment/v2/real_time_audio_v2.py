# -*- coding: utf-8 -*-
"""
real_time_audio_v2.py — Real-time drone detection via microphone (v2)
======================================================================
Uses the drone_detection package (v15) exclusively.
Does NOT import from v1 modules.

If PyAudio is unavailable the class degrades gracefully:
is_monitoring stays False and start_monitoring() logs a warning.
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
    Continuously captures audio from a microphone and runs v15 detection.

    Parameters
    ----------
    config         : drone_detection Config object
    channel_count  : 1 (single mic) or 3 (3-mic device)
    segment_dur    : seconds per detection window (default 3.0)
    threshold      : drone classification threshold (default 0.70)
    device_indices : list of PyAudio device indices
    """

    CHUNK: int = 4096

    def __init__(
        self,
        config,
        channel_count:  int  = 1,
        segment_dur:    float = 3.0,
        threshold:      float = 0.70,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        self.config        = config
        self.channel_count = channel_count
        self.segment_dur   = segment_dur
        self.threshold     = threshold
        self.device_indices = device_indices or []

        self.is_monitoring: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pa         = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start_monitoring(self, callback: Callable[[dict], None]) -> None:
        """
        Start background monitoring thread.

        callback(result) is called for every window processed.
        result keys: detected, confidence, position, reliable, timestamp,
                     segment_index
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
        log.info(
            f"v2 real-time monitoring started "
            f"(seg={self.segment_dur}s, threshold={self.threshold})"
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
        sr          = self.config.SR
        seg_samples = int(self.segment_dur * sr)
        buffer      = np.zeros(seg_samples, dtype=np.float32)
        filled      = 0
        seg_idx     = 0

        try:
            self._pa = pyaudio.PyAudio()
            stream   = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            log.info("Audio stream opened")

            while not self._stop_event.is_set():
                raw   = stream.read(self.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32)

                space = seg_samples - filled
                take  = min(len(chunk), space)
                buffer[filled : filled + take] = chunk[:take]
                filled += take

                if filled >= seg_samples:
                    self._process_segment(buffer.copy(), sr, seg_idx, callback)
                    seg_idx += 1
                    # 50 % overlap
                    half             = seg_samples // 2
                    buffer[:half]    = buffer[half:seg_samples]
                    filled           = half

            stream.stop_stream()
            stream.close()

        except Exception as exc:
            log.exception(f"Monitor loop error: {exc}")
        finally:
            self.is_monitoring = False

    def _process_segment(
        self,
        audio_seg: np.ndarray,
        sr:        int,
        seg_idx:   int,
        callback:  Callable[[dict], None],
    ) -> None:
        """Write segment to temp WAV, run v15 detection, fire callback."""
        tmp: Optional[str] = None
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tf.name, audio_seg, sr)
            tmp = tf.name

            from drone_detection import detect, localize, AudioProcessor

            ap       = AudioProcessor(self.config)
            y        = ap.pad_or_truncate(ap.load(tmp))
            channels = [y, y, y]   # single mic replicated to 3 channels

            det  = detect(channels, self.config)
            prob = float(det["probability"])

            if prob >= self.threshold:
                loc = localize(channels, self.config)
                pos = loc.get("xy_position")
                callback({
                    "detected":      True,
                    "confidence":    prob,
                    "position":      pos.tolist() if hasattr(pos, "tolist") else pos,
                    "reliable":      True,
                    "timestamp":     time.time(),
                    "segment_index": seg_idx,
                })
            else:
                callback({
                    "detected":      False,
                    "confidence":    prob,
                    "position":      None,
                    "reliable":      False,
                    "timestamp":     time.time(),
                    "segment_index": seg_idx,
                })

        except Exception as exc:
            log.debug(f"Segment {seg_idx} error: {exc}")
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass