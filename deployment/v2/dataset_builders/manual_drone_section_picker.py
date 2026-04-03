#!/usr/bin/env python3
# manual_drone_section_picker.py
"""
Interactive manual clipper for reviewing audio files and saving only clean drone sections.

What it does:
- loops through audio files in a folder
- optionally plays each full file or selected regions
- shows duration
- lets you enter one or more start:end ranges in seconds
- saves clipped clean drone sections to an output folder
- writes a JSON + CSV manifest

Example:
    python manual_drone_section_picker.py \
        --source output_v3/clean_drone_segments \
        --output output_v3/manual_review

Range input examples:
    12.5:18.2
    3.0:6.5, 10.0:14.2
Commands:
    p             play full file
    p 10 16       play region 10s to 16s
    s             skip file
    q             quit
    h             help
"""

import csv
import json
import sys
import warnings
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False


AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff", ".m4a")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name]
    return "".join(keep).strip("_") or "output"


def load_audio_any(path: Path, sr: int) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        if PYDUB_OK:
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            try:
                AudioSegment.from_file(str(path)).export(str(tmp), format="wav")
                y, _ = librosa.load(str(tmp), sr=sr, mono=True)
                return y.astype(np.float32)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
        raise


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y)) + 1e-8)
    return np.clip(y * (peak / m), -1.0, 1.0).astype(np.float32)


def try_play_audio(y: np.ndarray, sr: int):
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        print("Nothing to play.")
        return

    if SOUNDDEVICE_OK:
        try:
            sd.stop()
            sd.play(y, sr, blocking=True)
            return
        except Exception as e:
            print(f"sounddevice playback failed: {e}")

    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(str(tmp), y, sr)
        if sys.platform.startswith("darwin"):
            subprocess.run(["afplay", str(tmp)], check=False)
        elif sys.platform.startswith("win"):
            import os
            os.startfile(str(tmp))  # type: ignore[attr-defined]
            input("Press Enter after playback finishes...")
        else:
            tried = False
            for cmd in (["ffplay", "-nodisp", "-autoexit", str(tmp)],
                        ["paplay", str(tmp)],
                        ["aplay", str(tmp)],
                        ["mpv", "--no-video", str(tmp)]):
                try:
                    subprocess.run(cmd, check=False,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    tried = True
                    break
                except Exception:
                    continue
            if not tried:
                print("No playback backend found. Install sounddevice, ffplay, mpv, paplay, or aplay.")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def parse_ranges(text: str, total_sec: float) -> List[Tuple[float, float]]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = []
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Invalid range '{p}'. Use start:end")
        a, b = p.split(":", 1)
        s = max(0.0, float(a.strip()))
        e = min(total_sec, float(b.strip()))
        if e <= s:
            raise ValueError(f"End must be greater than start in '{p}'")
        out.append((s, e))
    return out


def save_clip(y: np.ndarray, sr: int, out_path: Path):
    ensure_dir(out_path.parent)
    y = normalize_peak(y)
    sf.write(str(out_path), y, sr)


def print_help():
    print("\nCommands:")
    print("  p             play full file")
    print("  p A B         play region A to B seconds")
    print("  h             show help")
    print("  s             skip file")
    print("  q             quit")
    print("  A:B           save one region")
    print("  A:B, C:D      save multiple regions")
    print()


def process_files(source: Path, output: Path, sr: int, recursive: bool):
    ensure_dir(output)
    clips_dir = output / "clean_drone_sections"
    manifests_dir = output / "manifests"
    ensure_dir(clips_dir)
    ensure_dir(manifests_dir)

    files = [p for p in (source.rglob("*") if recursive else source.glob("*"))
             if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    files = sorted(files)

    if not files:
        print(f"No audio files found in {source}")
        return

    manifest_rows = []
    reviewed = 0

    print(f"Found {len(files)} audio files.")
    print_help()

    for idx, path in enumerate(files, 1):
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(files)}] {path.name}")
        try:
            y = load_audio_any(path, sr)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        total_sec = len(y) / sr
        print(f"Duration: {total_sec:.2f} sec")

        reviewed += 1

        while True:
            cmd = input(
                "Enter command or ranges to save "
                "(p / p start end / s / q / start:end[,start:end]): "
            ).strip()

            if not cmd:
                continue

            if cmd.lower() == "h":
                print_help()
                continue

            if cmd.lower() == "s":
                print("Skipped.")
                break

            if cmd.lower() == "q":
                print("Stopping review.")
                manifest_json = manifests_dir / "manual_clips_manifest.json"
                manifest_csv = manifests_dir / "manual_clips_manifest.csv"
                manifest_json.write_text(json.dumps(manifest_rows, indent=2))
                if manifest_rows:
                    keys = sorted({k for row in manifest_rows for k in row.keys()})
                    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=keys)
                        w.writeheader()
                        w.writerows(manifest_rows)
                return

            if cmd.lower().startswith("p"):
                parts = cmd.split()
                if len(parts) == 1:
                    print("Playing full file...")
                    try_play_audio(y, sr)
                elif len(parts) == 3:
                    try:
                        s = max(0.0, float(parts[1]))
                        e = min(total_sec, float(parts[2]))
                        if e <= s:
                            print("End must be greater than start.")
                            continue
                        ys = y[int(s * sr):int(e * sr)]
                        print(f"Playing {s:.2f}s to {e:.2f}s...")
                        try_play_audio(ys, sr)
                    except ValueError:
                        print("Invalid play command. Use: p 10 16")
                else:
                    print("Invalid play command. Use: p or p 10 16")
                continue

            try:
                ranges = parse_ranges(cmd, total_sec)
            except Exception as e:
                print(f"Invalid input: {e}")
                continue

            saved_now = 0
            for clip_i, (s, e) in enumerate(ranges, 1):
                ys = y[int(s * sr):int(e * sr)]
                if len(ys) == 0:
                    continue
                out_name = (
                    f"{safe_slug(path.stem)}"
                    f"_{int(s*1000):07d}_{int(e*1000):07d}.wav"
                )
                out_path = clips_dir / out_name
                save_clip(ys, sr, out_path)
                manifest_rows.append({
                    "source_file": str(path),
                    "output_file": str(out_path),
                    "start_s": round(float(s), 4),
                    "end_s": round(float(e), 4),
                    "duration_s": round(float(e - s), 4),
                })
                saved_now += 1
                print(f"Saved: {out_path.name}")

            print(f"Saved {saved_now} clip(s) from this file.")
            break

    manifest_json = manifests_dir / "manual_clips_manifest.json"
    manifest_csv = manifests_dir / "manual_clips_manifest.csv"
    manifest_json.write_text(json.dumps(manifest_rows, indent=2))
    if manifest_rows:
        keys = sorted({k for row in manifest_rows for k in row.keys()})
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(manifest_rows)

    print("\nDone.")
    print(f"Files reviewed: {reviewed}")
    print(f"Clips saved   : {len(manifest_rows)}")
    print(f"Output dir    : {clips_dir}")


def main():
    ap = argparse.ArgumentParser(description="Manual drone section picker")
    ap.add_argument("--source", required=True, help="Folder with audio files to review")
    ap.add_argument("--output", required=True, help="Folder to save clipped sections")
    ap.add_argument("--sr", type=int, default=22050, help="Playback/load sample rate")
    ap.add_argument("--non-recursive", action="store_true", help="Only scan top-level folder")
    args = ap.parse_args()

    process_files(
        source=Path(args.source),
        output=Path(args.output),
        sr=args.sr,
        recursive=not args.non_recursive,
    )


if __name__ == "__main__":
    main()
