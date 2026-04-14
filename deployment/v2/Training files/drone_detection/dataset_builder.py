# -*- coding: utf-8 -*-
"""
dataset_builder.py
──────────────────
Dataset download, extraction, and preparation pipeline.

Contents
────────
AudioWebScraper              — multi-source audio scraper (Freesound, BBC, xeno-canto, etc.)
_incorporate_scraped_audio() — move scraped clips into the processed detection splits
DroneAudioDatasetManager     — download + process the GitHub DroneAudioDataset
_convert_to_wav()            — convert any audio format to WAV via pydub / copy

These utilities are consumed by notebook.train_detection() and the top-level
main() entry-point.  The AudioWebScraper class is intentionally kept separate
from audio.py so that the heavy web-scraping logic does not bloat the core
audio processing module.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from pydub import AudioSegment
    _PYDUB_OK = True
except ImportError:
    _PYDUB_OK = False

try:
    import yt_dlp as ytdlp
    _YTDLP_OK = True
except ImportError:
    _YTDLP_OK = False

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

from drone_detection.config import AUDIO_EXTS, config as _default_cfg
from drone_detection.audio import AudioProcessor, load_audio_any


# ══════════════════════════════════════════════════════════════════════════════
# Format conversion helper
# ══════════════════════════════════════════════════════════════════════════════

def _convert_to_wav(src: Path, dst: Path) -> None:
    """
    Copy src to dst as a WAV file.

    If src is already a WAV, performs a plain file copy.
    Otherwise, uses pydub to decode and re-encode as WAV.
    Raises RuntimeError if pydub is unavailable and the format is not WAV.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".wav":
        shutil.copy2(str(src), str(dst))
        return

    if not _PYDUB_OK:
        raise RuntimeError(
            f"pydub is required to convert {src.suffix!r} → WAV, "
            "but it is not installed."
        )
    AudioSegment.from_file(str(src)).export(str(dst), format="wav")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-source audio web scraper
# ══════════════════════════════════════════════════════════════════════════════

class AudioWebScraper:
    """
    Download drone and non-drone audio from multiple free public sources:

      • Freesound.org   (optional API key — set cfg.FREESOUND_API_KEY)
      • FreeSound.io    (no key required)
      • BBC Sound Effects (no key required)
      • xeno-canto      (no key required, bird/ambient sounds)
      • SoundBible      (no key required)
      • YouTube via yt-dlp (opt-in, cfg.SCRAPE_YTDLP_ENABLED = True)

    Downloaded files are stored under cfg.RAW_DIR / "scraped_audio" / {label}
    where label is either "drone" or "non_drone".
    """

    AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")

    # Freesound.org search queries
    _FS_QUERIES: Dict[str, List[str]] = {
        "drone":     ["drone flying", "quadcopter", "uav sound", "drone propeller"],
        "non_drone": ["wind", "car passing", "crowd noise", "bird chirping",
                      "engine", "airplane", "construction"],
    }

    # BBC Sound Effects category keywords
    _BBC_DRONE = ["drone", "buzz", "propeller", "rotor"]
    _BBC_NON   = ["wind", "crowd", "traffic", "rain", "birds", "urban"]

    # SoundBible sound IDs for non-drone backgrounds
    _SOUNDBIBLE_IDS = ["1575", "1480", "1350", "1288", "1192"]

    def __init__(self, cfg=None):
        self.cfg      = cfg or _default_cfg
        self.api_key  = getattr(self.cfg, "FREESOUND_API_KEY", "")
        self.out_root = self.cfg.RAW_DIR / "scraped_audio"

        if _REQUESTS_OK:
            import requests
            self.sess = requests.Session()
            self.sess.headers.update({"User-Agent": "DroneDetectionResearch/1.0"})
        else:
            self.sess = None

    # ── Public entry-point ─────────────────────────────────────────────────

    def download(self, force: bool = False) -> None:
        """
        Run all scrapers and download into cfg.RAW_DIR / "scraped_audio".

        Parameters
        ──────────
        force : if True, re-download even if files already exist
        """
        if not _REQUESTS_OK:
            print("⚠️  requests not installed — audio scraping skipped.")
            return

        for label in ("drone", "non_drone"):
            (self.out_root / label).mkdir(parents=True, exist_ok=True)

        existing = self._count()
        if existing > 0 and not force:
            print(f"✅ Scraped audio already present ({existing} files) — skipping.")
            return

        print("🌐 Multi-source audio scraping …")
        self._scrape_freesound()
        self._scrape_freesound_io()
        self._scrape_bbc()
        self._scrape_xeno_canto()
        self._scrape_soundbible()

        if getattr(self.cfg, "SCRAPE_YTDLP_ENABLED", False) and _YTDLP_OK:
            self._scrape_youtube()

        print(f"✅ Scraping complete — {self._count()} files collected.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _count(self) -> int:
        total = 0
        for label in ("drone", "non_drone"):
            d = self.out_root / label
            if d.exists():
                for ext in self.AUDIO_EXTS:
                    total += len(list(d.glob(f"*{ext}")))
        return total

    def _get(self, url: str, timeout: int = 10) -> Optional[bytes]:
        """GET a URL and return bytes, or None on failure."""
        if self.sess is None:
            return None
        try:
            resp = self.sess.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception:
            return None

    # ── Freesound.org ──────────────────────────────────────────────────────

    def _scrape_freesound(self) -> None:
        if not self.api_key:
            print("  ℹ️  FREESOUND_API_KEY not set — skipping Freesound.org")
            return

        print("  🔎 Freesound.org …")
        base = "https://freesound.org/apiv2/search/text/"

        for label, terms in self._FS_QUERIES.items():
            dst = self.out_root / label
            for term in terms:
                try:
                    resp = self.sess.get(
                        base,
                        params={
                            "query":     term,
                            "filter":    f"duration:[{self.cfg.SCRAPE_MIN_DURATION} TO "
                                         f"{self.cfg.SCRAPE_MAX_DURATION}]",
                            "fields":    "id,name,previews,duration",
                            "page_size": self.cfg.SCRAPE_MAX_PER_QUERY,
                            "token":     self.api_key,
                        },
                        timeout=10,
                    )
                    data = resp.json()
                except Exception as exc:
                    print(f"    ⚠️  FS ({term}): {exc}")
                    continue

                for sound in data.get("results", []):
                    dur = sound.get("duration", 0)
                    if not (self.cfg.SCRAPE_MIN_DURATION <= dur <= self.cfg.SCRAPE_MAX_DURATION):
                        continue
                    fp = dst / f"fs_{sound['id']}.mp3"
                    if fp.exists():
                        continue
                    content = self._get(sound["previews"]["preview-hq-mp3"])
                    if content and len(content) > 5000:
                        fp.write_bytes(content)

    # ── FreeSound.io (key-free) ────────────────────────────────────────────

    def _scrape_freesound_io(self) -> None:
        import re as _re
        print("  🔎 FreeSound.io (key-free) …")
        base = "https://freesound.io/api/sounds/search"
        queries: Dict[str, List[str]] = {
            "drone":     ["drone", "uav", "quadcopter"],
            "non_drone": ["wind", "crowd", "ambient"],
        }
        for label, terms in queries.items():
            dst = self.out_root / label
            for term in terms:
                try:
                    resp  = self.sess.get(base, params={"query": term, "limit": 20}, timeout=8)
                    data  = resp.json()
                    items = data.get("results", data if isinstance(data, list) else [])
                    for item in items:
                        url = item.get("download_url") or item.get("url", "")
                        if not url or not any(url.endswith(e) for e in self.AUDIO_EXTS):
                            continue
                        fid = _re.sub(r"[^\w]", "_", url[-20:])
                        ext = Path(url).suffix or ".mp3"
                        fp  = dst / f"fsio_{fid}{ext}"
                        if fp.exists():
                            continue
                        content = self._get(url)
                        if content and len(content) > 5000:
                            fp.write_bytes(content)
                except Exception as exc:
                    print(f"    ⚠️  FSio ({term}): {exc}")

    # ── BBC Sound Effects ──────────────────────────────────────────────────

    def _scrape_bbc(self) -> None:
        print("  🔎 BBC Sound Effects …")
        base = "https://sound-effects.bbcrewind.co.uk/api/search"
        qs   = {"drone": self._BBC_DRONE, "non_drone": self._BBC_NON}
        for label, terms in qs.items():
            dst = self.out_root / label
            for term in terms:
                try:
                    resp   = self.sess.get(base, params={"q": term, "limit": 15}, timeout=8)
                    data   = resp.json()
                    sounds = data.get("results", data.get("sounds", []))
                    for s in sounds:
                        sid = s.get("id", s.get("assetId", ""))
                        if not sid:
                            continue
                        fp = dst / f"bbc_{sid}.wav"
                        if fp.exists():
                            continue
                        url     = f"https://sound-effects-media.bbcrewind.co.uk/zip/{sid}.wav"
                        content = self._get(url, timeout=12)
                        if content and len(content) > 10_000:
                            fp.write_bytes(content)
                except Exception as exc:
                    print(f"    ⚠️  BBC ({term}): {exc}")

    # ── xeno-canto ─────────────────────────────────────────────────────────

    def _scrape_xeno_canto(self) -> None:
        print("  🔎 xeno-canto …")
        dst = self.out_root / "non_drone"
        for term in ("wind", "rain", "stream", "ambient"):
            try:
                data = self.sess.get(
                    "https://xeno-canto.org/api/2/recordings",
                    params={"query": term, "page": 1},
                    timeout=8,
                ).json()
                for rec in data.get("recordings", [])[:10]:
                    url = "https:" + rec.get("file", "")
                    if url == "https:":
                        continue
                    fp = dst / f"xc_{rec.get('id', '')}.mp3"
                    if fp.exists():
                        continue
                    content = self._get(url, timeout=12)
                    if content and len(content) > 5000:
                        fp.write_bytes(content)
            except Exception as exc:
                print(f"    ⚠️  XC ({term}): {exc}")

    # ── SoundBible ─────────────────────────────────────────────────────────

    def _scrape_soundbible(self) -> None:
        print("  🔎 SoundBible …")
        dst = self.out_root / "non_drone"
        for sid in self._SOUNDBIBLE_IDS:
            fp = dst / f"sb_{sid}.mp3"
            if fp.exists():
                continue
            content = self._get(
                f"https://soundbible.com/grab.php?id={sid}&type=mp3", timeout=10
            )
            if content and len(content) > 5000:
                fp.write_bytes(content)

    # ── YouTube via yt-dlp ─────────────────────────────────────────────────

    def _scrape_youtube(self) -> None:
        if not _YTDLP_OK:
            print("  ⚠️  yt-dlp not installed.")
            return
        print("  🔎 yt-dlp (YouTube) …")
        dst  = self.out_root / "drone"
        opts = {
            "format":    "bestaudio/best",
            "outtmpl":   str(dst / "yt_%(id)s.%(ext)s"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
            ],
            "download_sections": {"*": "00:00:00-00:00:30"},
            "quiet":       True,
            "no_warnings": True,
        }
        for url in getattr(self.cfg, "SCRAPE_YTDLP_URLS", []):
            try:
                with ytdlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
            except Exception as exc:
                print(f"    ⚠️  yt-dlp ({url}): {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Incorporate scraped audio into the processed detection splits
# ══════════════════════════════════════════════════════════════════════════════

def _incorporate_scraped_audio(cfg=None, force: bool = False) -> None:
    """
    Move scraped audio clips into the processed detection train/val splits as WAVs.

    Layout after this call
    ──────────────────────
    cfg.PROCESSED_DIR / detection / {train,val} / {drone,non_drone} / *.wav

    The test split is intentionally left untouched so that it remains clean
    and representative of real-world (non-scraped) recordings.

    Parameters
    ──────────
    cfg   : pipeline Config object; defaults to the module-level singleton
    force : if True, overwrite already-converted WAV files
    """
    cfg     = cfg or _default_cfg
    scraped = cfg.RAW_DIR / "scraped_audio"

    if not scraped.exists():
        print("ℹ️  No scraped audio directory found — skipping incorporation.")
        return

    ap = AudioProcessor(cfg)

    for label in ("drone", "non_drone"):
        src_dir = scraped / label
        if not src_dir.exists():
            continue

        # Collect every audio file regardless of extension
        files: List[Path] = []
        for ext in AudioWebScraper.AUDIO_EXTS:
            files.extend(src_dir.glob(f"*{ext}"))

        if not files:
            continue

        random.shuffle(files)

        # 85 % → train, 15 % → val  (test stays clean)
        split_idx  = int(len(files) * 0.85)
        splits     = [("train", files[:split_idx]), ("val", files[split_idx:])]

        added = skipped = failed = 0
        for split_name, flist in splits:
            dst_dir = cfg.PROCESSED_DIR / "detection" / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src in flist:
                dst = dst_dir / f"{src.stem}.wav"
                if dst.exists() and not force:
                    skipped += 1
                    continue
                try:
                    # Load → resample → pad/truncate → write
                    y = ap.pad_or_truncate(load_audio_any(src, cfg.SR))
                    sf.write(str(dst), y, cfg.SR)
                    added += 1
                except Exception as exc:
                    failed += 1
                    print(f"   ⚠️  Skipping {src.name}: {exc}")

        print(
            f"   ✅ {label}: {added} added, {skipped} skipped, {failed} failed"
            f"  ({len(files)} source files)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DroneAudioDataset (GitHub) download + processing
# ══════════════════════════════════════════════════════════════════════════════

class DroneAudioDatasetManager:
    """
    Download and prepare the binary-label DroneAudioDataset from GitHub.

    Source
    ──────
    https://github.com/saraalemadi/DroneAudioDataset

    The repository contains a Binary_Drone_Audio/ folder with subfolders
    whose names map to "drone" or "non_drone":
        yes_drone / Drone  →  drone
        unknown / noDrone  →  non_drone

    After processing, WAV files are written to:
        cfg.PROCESSED_DIR / detection / {train,val,test} / {drone,non_drone} / *.wav
    with a 70 / 15 / 15 train / val / test split.
    """

    _CLASS_MAP = {
        "yes_drone": "drone",
        "Drone":     "drone",
        "unknown":   "non_drone",
        "noDrone":   "non_drone",
    }

    def __init__(self, cfg=None):
        self.cfg = cfg or _default_cfg
        self.ap  = AudioProcessor(cfg)

    # ── Public entry-point ─────────────────────────────────────────────────

    def prepare(self) -> bool:
        """
        Ensure the processed detection dataset is populated.

        Returns True if the dataset is ready (either already existed or was
        freshly built), False on unrecoverable failure.
        """
        proc = self.cfg.PROCESSED_DIR / "detection"
        counts = {
            f"{split}_{lbl}": len(list((proc / split / lbl).glob("*.wav")))
            if (proc / split / lbl).exists() else 0
            for split in ("train", "val", "test")
            for lbl   in ("drone", "non_drone")
        }
        ready = all([
            counts["train_drone"]     > 20,
            counts["train_non_drone"] > 20,
            counts["val_drone"]       >  0,
            counts["val_non_drone"]   >  0,
            counts["test_drone"]      >  0,
            counts["test_non_drone"]  >  0,
        ])
        if ready:
            total = sum(counts.values())
            print(f"✅ Detection dataset ready ({total} files)")
            return True

        print("⚠️  Detection dataset incomplete — rebuilding …")
        if proc.exists():
            shutil.rmtree(proc)

        raw_dir = self.cfg.DRONEDS_RAW
        self._download(raw_dir)
        return self._process(raw_dir, proc)

    # ── Download ───────────────────────────────────────────────────────────

    def _download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "drone_dataset.zip"

        if not archive.exists():
            print("📥 Downloading DroneAudioDataset …")
            try:
                urllib.request.urlretrieve(self.cfg.DRONEDS_ZIP_URL, str(archive))
            except Exception as exc:
                raise RuntimeError(f"Download failed: {exc}") from exc

        # Only extract if Binary_Drone_Audio is not already present
        already_extracted = any(
            d.name == "Binary_Drone_Audio"
            for d in dest.rglob("*")
            if d.is_dir()
        )
        if not already_extracted:
            print("📦 Extracting …")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(str(dest))

    # ── Process ────────────────────────────────────────────────────────────

    def _process(self, raw_dir: Path, proc_dir: Path) -> bool:
        binary_dir = next(
            (d for d in raw_dir.rglob("Binary_Drone_Audio") if d.is_dir()), None
        )
        if binary_dir is None:
            print("❌ Binary_Drone_Audio directory not found inside extracted archive.")
            return False

        all_files: Dict[str, List[Path]] = {"drone": [], "non_drone": []}
        for cls_dir in binary_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            label = self._CLASS_MAP.get(cls_dir.name, "non_drone")
            all_files[label].extend(
                [f for f in cls_dir.glob("*.*") if f.is_file()]
            )

        drone_files     = all_files["drone"]
        non_drone_files = all_files["non_drone"]
        print(
            f"📊 Raw — Drone: {len(drone_files)}, "
            f"Non-drone: {len(non_drone_files)}"
        )

        # Balance: cap non-drone at 2× drone count
        if len(non_drone_files) > len(drone_files) * 2:
            non_drone_files = random.sample(
                non_drone_files, len(drone_files) * 2
            )

        for label, files in (("drone", drone_files), ("non_drone", non_drone_files)):
            self._create_splits(label, files, proc_dir)

        total = len(list(proc_dir.rglob("*.wav")))
        print(f"✅ Detection dataset processed ({total} files)")
        return total > 0

    def _create_splits(
        self, label: str, files: List[Path], proc_dir: Path
    ) -> None:
        random.shuffle(files)
        n = len(files)
        splits = {
            "train": files[: int(n * 0.70)],
            "val":   files[int(n * 0.70) : int(n * 0.85)],
            "test":  files[int(n * 0.85) :],
        }
        for split_name, flist in splits.items():
            dst = proc_dir / split_name / label
            dst.mkdir(parents=True, exist_ok=True)
            for src in flist:
                out = dst / f"{src.stem}.wav"
                if out.exists():
                    continue
                try:
                    _convert_to_wav(src, out)
                except Exception as exc:
                    # Fallback: load with librosa and write
                    if _LIBROSA_OK:
                        try:
                            y, _ = librosa.load(str(src), sr=self.cfg.SR, mono=True)
                            sf.write(str(out), y.astype("float32"), self.cfg.SR)
                        except Exception as exc2:
                            print(f"   ⚠️  {src.name}: {exc2}")
                    else:
                        print(f"   ⚠️  {src.name}: {exc}")