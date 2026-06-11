#!/usr/bin/env python3
"""
diagnose_polywav.py
───────────────────
Quick diagnostic for a local BRUEL polywav file.
Checks: header offset, channel RMS levels, NaN distribution,
        and whether the target channels contain plausible audio.

Usage:
    python diagnose_polywav.py /path/to/251020VITEMOROM1AT01P.wav

Optional args:
    --onset 268.68       start time in seconds (default: 268.68)
    --dur   30.0         window duration in seconds (default: 30.0)
    --channels 8 9 10    channel indices to inspect (default: 8 9 10)
    --offset 80          force data offset in bytes (default: auto-detect)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

POLYWAV_SR       = 192_000
POLYWAV_CHANNELS = 14
POLYWAV_BPS      = 4   # bytes per sample
POLYWAV_BPF      = POLYWAV_CHANNELS * POLYWAV_BPS   # 56 bytes per frame
KNOWN_OFFSET     = 80  # BRUEL 14-ch float32 WAV


def detect_offset(path: Path) -> int:
    with open(path, "rb") as f:
        hdr = f.read(512)
    if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
        print(f"  ⚠️  Not a standard RIFF/WAVE file — header probe returned non-RIFF bytes")
        print(f"      Using known BRUEL polywav offset: {KNOWN_OFFSET} bytes")
        return KNOWN_OFFSET
    pos = 12
    while pos + 8 <= len(hdr):
        chunk_id   = hdr[pos:pos+4]
        chunk_size = struct.unpack_from("<I", hdr, pos+4)[0]
        if chunk_id == b"data":
            offset = pos + 8
            print(f"  ✅ RIFF parse found 'data' chunk at byte {offset}")
            return offset
        pos += 8 + chunk_size + (chunk_size % 2)
    print(f"  ⚠️  'data' chunk not found in first 512 bytes — using {KNOWN_OFFSET}")
    return KNOWN_OFFSET


def read_window(path: Path, offset: int, onset_s: float, dur_s: float):
    frame_start = int(onset_s * POLYWAV_SR)
    n_frames    = int(dur_s   * POLYWAV_SR)
    byte_start  = offset + frame_start * POLYWAV_BPF
    byte_end    = byte_start + n_frames * POLYWAV_BPF
    file_size   = path.stat().st_size

    print(f"\n  File size : {file_size:,} bytes  ({file_size/1e9:.2f} GB)")
    print(f"  Byte range: [{byte_start:,}, {byte_end:,})  "
          f"({(byte_end-byte_start)/1e6:.1f} MB)")

    if byte_start >= file_size:
        print(f"  ❌ byte_start {byte_start:,} is beyond file size {file_size:,}")
        sys.exit(1)
    actual_end = min(byte_end, file_size)
    if actual_end < byte_end:
        print(f"  ⚠️  Window truncated: only {actual_end - byte_start:,} bytes available "
              f"(expected {byte_end - byte_start:,})")

    with open(path, "rb") as f:
        f.seek(byte_start)
        raw = f.read(actual_end - byte_start)

    usable = (len(raw) // POLYWAV_BPF) * POLYWAV_BPF
    arr = np.frombuffer(raw[:usable], dtype=np.float32).reshape(-1, POLYWAV_CHANNELS)
    print(f"  Decoded   : {arr.shape[0]:,} frames × {POLYWAV_CHANNELS} channels")
    return arr


def report(arr: np.ndarray, channels: list[int]):
    total   = arr.size
    n_nan   = int(np.sum(np.isnan(arr)))
    n_inf   = int(np.sum(np.isinf(arr)))
    n_bad   = n_nan + n_inf
    pct_bad = 100.0 * n_bad / total if total else 0

    print(f"\n  NaN/Inf summary:")
    print(f"    Total samples : {total:,}")
    print(f"    NaN           : {n_nan:,}")
    print(f"    Inf           : {n_inf:,}")
    print(f"    Bad total     : {n_bad:,}  ({pct_bad:.2f}%)")

    # Clean copy for stats
    clean = np.where(np.isfinite(arr), arr, 0.0)

    print(f"\n  Per-channel RMS (all {POLYWAV_CHANNELS} channels, NaN→0):")
    for ch in range(POLYWAV_CHANNELS):
        col     = clean[:, ch]
        rms     = float(np.sqrt(np.mean(col**2)))
        peak    = float(np.max(np.abs(col)))
        n_ch_bad= int(np.sum(~np.isfinite(arr[:, ch])))
        marker  = " ◀ TARGET" if ch in channels else ""
        print(f"    ch{ch:02d}: RMS={rms:.6f}  peak={peak:.4f}  "
              f"bad={n_ch_bad:,}{marker}")

    print(f"\n  Target channels {channels} detail:")
    for ch in channels:
        col  = clean[:, ch]
        rms  = float(np.sqrt(np.mean(col**2)))
        rms_db = 20*np.log10(rms + 1e-12)
        dc   = float(np.mean(col))
        peak = float(np.max(np.abs(col)))
        # Rough spectral check: energy in drone BPF band (50–500 Hz at 192kHz)
        # Use simple variance of a decimated signal as a proxy
        decimate = 64   # 192000/64 = 3000 Hz effective SR
        dec = col[::decimate]
        bpf_proxy = float(np.var(dec))
        print(f"    ch{ch:02d}: RMS={rms:.6f} ({rms_db:.1f} dBFS)  "
              f"DC={dc:.6f}  peak={peak:.4f}  "
              f"low-freq var={bpf_proxy:.2e}")
        if rms < 1e-6:
            print(f"         ⚠️  Nearly silent — channel may be wrong or file corrupt")
        elif rms_db > -3:
            print(f"         ⚠️  Clipping likely (RMS > -3 dBFS)")
        else:
            print(f"         ✅ Signal present")

    print(f"\n  Diagnosis:")
    if pct_bad > 20:
        print(f"    ❌ Very high NaN/Inf rate ({pct_bad:.1f}%) — data_offset={arr.shape} "
              f"is almost certainly wrong")
    elif pct_bad > 5:
        print(f"    ⚠️  Elevated NaN/Inf rate ({pct_bad:.1f}%) — offset may be slightly off "
              f"or file has corrupt regions")
    else:
        print(f"    ✅ NaN/Inf rate acceptable ({pct_bad:.1f}%)")

    target_rms = [float(np.sqrt(np.mean(clean[:, ch]**2))) for ch in channels]
    if all(r < 1e-6 for r in target_rms):
        print(f"    ❌ All target channels are silent — wrong channels or wrong file")
    elif any(r < 1e-6 for r in target_rms):
        print(f"    ⚠️  Some target channels are silent — check channel mapping")
    else:
        avg_rms_db = 20 * np.log10(np.mean(target_rms) + 1e-12)
        print(f"    ✅ Target channels have signal (avg RMS: {avg_rms_db:.1f} dBFS)")
        if avg_rms_db < -40:
            print(f"    ⚠️  Signal is very quiet ({avg_rms_db:.1f} dBFS) — "
                  f"drone may be far away or SNR is low")
        if avg_rms_db > -3:
            print(f"    ⚠️  Signal may be clipping")


def main():
    ap = argparse.ArgumentParser(description="Diagnose a BRUEL polywav window")
    ap.add_argument("file",               help="Path to .wav file")
    ap.add_argument("--onset",   type=float, default=268.68,
                    help="Onset in seconds (default: 268.68)")
    ap.add_argument("--dur",     type=float, default=30.0,
                    help="Duration in seconds (default: 30.0)")
    ap.add_argument("--channels",type=int, nargs="+", default=[8, 9, 10],
                    help="Channel indices to inspect (default: 8 9 10)")
    ap.add_argument("--offset",  type=int, default=None,
                    help="Force data offset in bytes (default: auto-detect)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  File   : {path.name}")
    print(f"  Onset  : {args.onset}s")
    print(f"  Dur    : {args.dur}s")
    print(f"  Channels: {args.channels}")
    print(f"{'='*60}")

    print(f"\n  Probing WAV header...")
    offset = args.offset if args.offset is not None else detect_offset(path)
    print(f"  Using data offset: {offset} bytes")

    arr = read_window(path, offset, args.onset, args.dur)
    report(arr, args.channels)

    # Also check a few other candidate offsets if NaN rate is high
    clean = np.where(np.isfinite(arr), arr, 0.0)
    target_rms = [float(np.sqrt(np.mean(clean[:, ch]**2))) for ch in args.channels]
    if all(r < 1e-4 for r in target_rms) or (np.sum(~np.isfinite(arr)) / arr.size) > 0.05:
        print(f"\n  Trying alternative offsets to find valid PCM alignment...")
        for alt_offset in [44, 60, 68, 80, 92, 100]:
            if alt_offset == offset:
                continue
            try:
                arr2  = read_window(path, alt_offset, args.onset, args.dur)
                clean2 = np.where(np.isfinite(arr2), arr2, 0.0)
                rms2   = [float(np.sqrt(np.mean(clean2[:, ch]**2))) for ch in args.channels]
                nan2   = np.sum(~np.isfinite(arr2)) / arr2.size * 100
                print(f"    offset={alt_offset:3d}: "
                      f"ch{args.channels} RMS={[f'{r:.5f}' for r in rms2]}  "
                      f"NaN={nan2:.1f}%")
            except Exception as e:
                print(f"    offset={alt_offset:3d}: error — {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

    # python diagnose_polywav.py /home/tildai/Desktop/Development/drone-detection-api/deployment/v2/wavs/251020VITEMOROM1AT01P.wav --onset 268.68 --dur 30 --channels 8 9 10