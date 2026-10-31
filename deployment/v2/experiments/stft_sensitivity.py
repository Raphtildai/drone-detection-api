"""
STFT front-end sensitivity study for the detection CNN (paper: Table
tab:stft-ablation, Paper/stft_setup.tex).

Each configuration retrains the detection CNN with the thesis training
procedure (drone_detection.training.DetectionTrainer: AdamW 3e-4, weight
decay 1e-3, warmup-cosine, focal loss, gradient clip 1.0) on the same
processed WAVs and split, changing exactly one front-end parameter. The
deployed checkpoint is never touched: models, logs and results go to
<out>/<config>/seed_<n>/.

Run from deployment/v2 (locally) or on Colab with the repo checked out:

    # once: build the processed detection WAVs (same steps as train_detection)
    python experiments/stft_sensitivity.py --prepare

    # full study (9 configs x 3 seeds); finished runs are skipped on re-run
    python experiments/stft_sensitivity.py --epochs 30 --seeds 42 43 44 \
        --field-dir dunakeszi_test_segments_B

    # one config only, or just rebuild the summary table
    python experiments/stft_sensitivity.py --configs default nfft512
    python experiments/stft_sensitivity.py --summary-only

On Colab use %run instead of !python: config.py detects Colab through
sys.modules, and a !python subprocess would use local paths instead.
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drone_detection import config as base_config
from drone_detection.audio_processing import AudioProcessor
from drone_detection.datasets import (
    DroneAudioDatasetManager, MelCacheManager, get_det_dataloaders,
    report_detection_split_counts,
)
from drone_detection.orchestration import (
    generate_mixed_drone_training_audio, import_custom_builder_dataset,
)
from drone_detection.training import (
    DetectionTrainer, collect_val_probs, evaluate_binary_metrics,
)
from drone_detection.utils import _set_seed, compute_delta_2d, safe_standardize


# ── Configurations ────────────────────────────────────────────────────────────

# The thesis front end (config.py and AudioProcessor.mel_power).
DEFAULTS = dict(N_FFT=1024, HOP_LENGTH=256, N_MELS=64, FMIN=20, FMAX=8000,
                FRONTEND="mel", FEATURES="stack")

# name -> (overrides of DEFAULTS, row label used in the LaTeX table)
CONFIGS = {
    "default":   ({},                                r"Default: $N{=}1024$, $H{=}256$, 64 mel, 3-ch"),
    "nfft512":   ({"N_FFT": 512, "HOP_LENGTH": 128},  r"$N{=}512$, $H{=}128$"),
    "nfft2048":  ({"N_FFT": 2048, "HOP_LENGTH": 512}, r"$N{=}2048$, $H{=}512$"),
    "hop512":    ({"HOP_LENGTH": 512},                r"$H{=}512$ (50\% overlap)"),
    "mel32":     ({"N_MELS": 32},                     r"32 mel bands"),
    "mel128":    ({"N_MELS": 128},                    r"128 mel bands"),
    "fmax4k":    ({"FMAX": 4000},                     r"$f_\mathrm{max} = 4$~kHz"),
    "logmel1ch": ({"FEATURES": "logmel"},             r"Log-mel only (1-ch)"),
    "linear":    ({"FRONTEND": "linear"},             r"Linear STFT magnitude (no mel)"),
}


# ── Front-end patch ───────────────────────────────────────────────────────────
# AudioProcessor hard-codes fmin/fmax and the 3-channel stack. These versions
# read them from cfg instead; with DEFAULTS they compute exactly what
# AudioProcessor.mel_power / feature_stack compute.

def _mel_power(self, y):
    c = self.cfg
    if c.FRONTEND == "linear":
        S = np.abs(librosa.stft(y, n_fft=c.N_FFT, hop_length=c.HOP_LENGTH)) ** 2
        f = librosa.fft_frequencies(sr=c.SR, n_fft=c.N_FFT)
        return S[(f >= c.FMIN) & (f <= c.FMAX)].astype(np.float32)
    return librosa.feature.melspectrogram(
        y=y, sr=c.SR, n_fft=c.N_FFT, hop_length=c.HOP_LENGTH,
        n_mels=c.N_MELS, fmin=c.FMIN, fmax=c.FMAX).astype(np.float32)


def _feature_stack(self, y):
    m = safe_standardize(self.mel(y))
    if self.cfg.FEATURES == "logmel":
        # the CNN expects 3 input channels: repeat log-mel instead of adding PCEN/delta
        return np.stack([m, m, m], axis=0).astype(np.float32)
    p = safe_standardize(self.pcen(y))
    d = safe_standardize(compute_delta_2d(m))
    return np.stack([m, p, d], axis=0).astype(np.float32)


AudioProcessor.mel_power = _mel_power
AudioProcessor.feature_stack = _feature_stack


def make_cfg(name):
    cfg = copy.deepcopy(base_config)
    for k, v in {**DEFAULTS, **CONFIGS[name][0]}.items():
        setattr(cfg, k, v)
    # caches are large and rebuildable -> local disk; results -> DRIVE_ROOT
    cfg.MEL_CACHE_DIR = cfg.LOCAL_BASE / "stft_ablation_cache" / name
    cfg.SEED = 42
    return cfg


# ── Steps ─────────────────────────────────────────────────────────────────────

def prepare(cfg):
    """The data steps of orchestration.train_detection, without training."""
    cfg.ensure_dirs()
    if getattr(cfg, "ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD", True):
        DroneAudioDatasetManager(cfg).prepare()
    if getattr(cfg, "ALLOW_EXTERNAL_AUDIO_DOWNLOADS", False):
        from drone_detection.dataset_builder import AudioWebScraper, _incorporate_scraped_audio
        AudioWebScraper(cfg).download(force=False)
        _incorporate_scraped_audio(cfg, force=False)
    if getattr(cfg, "CUSTOM_DATASET_ROOT", ""):
        import_custom_builder_dataset(
            cfg, cfg.CUSTOM_DATASET_ROOT, force=False,
            include_background_pool_as_non_drone=cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE)
    generate_mixed_drone_training_audio(cfg, force=False)
    report_detection_split_counts(cfg)


def build_cache(cfg):
    cfg.SEED = 42  # synthetic samples use SEED+1: keep them identical across configs
    mcm = MelCacheManager(cfg)
    mcm.build(force=False)
    mcm.inject_synthetic(force=False)
    return mcm.count()


def field_windows(ap, path, win_s=3.0, hop_s=1.5):
    y = ap.load(path)
    n, h = int(ap.cfg.SR * win_s), int(ap.cfg.SR * hop_s)
    starts = range(0, max(len(y) - n, 0) + 1, h)
    return [ap.feature_stack(ap.pad_or_truncate(y[s:s + n])) for s in starts]


def eval_field(model, ap, files, thr, dev):
    """All field segments contain a drone, so this is the window-level recall."""
    if not files:
        return {}
    probs = []
    model.eval()
    with torch.no_grad():
        for f in files:
            X = torch.tensor(np.stack(field_windows(ap, f)), dtype=torch.float32).to(dev)
            probs.extend(torch.softmax(model(X), dim=1)[:, 1].cpu().numpy().tolist())
    probs = np.asarray(probs)
    return {"field_det_rate": float((probs >= thr).mean()),
            "field_mean_prob": float(probs.mean()), "field_n_windows": int(len(probs))}


def time_cpu(model, ap, reps=30):
    y = np.random.randn(int(ap.cfg.SR * ap.cfg.TARGET_DURATION)).astype(np.float32) * 0.1
    ap.feature_stack(y)
    t0 = time.perf_counter()
    for _ in range(reps):
        x = ap.feature_stack(y)
    feat_ms = 1000 * (time.perf_counter() - t0) / reps
    m = copy.deepcopy(model).cpu().eval()
    X = torch.tensor(x[None], dtype=torch.float32)
    with torch.no_grad():
        m(X)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(X)
    return {"feature_ms": feat_ms, "cnn_cpu_ms": 1000 * (time.perf_counter() - t0) / reps,
            "input_shape": list(x.shape)}


def run_one(name, seed, epochs, out_root, field_files, threshold):
    run_dir = out_root / name / f"seed_{seed}"
    res_path = run_dir / "results.json"
    if res_path.exists():
        print(f"⏭️  {name} seed {seed}: already done")
        return json.loads(res_path.read_text())

    cfg = make_cfg(name)
    counts = build_cache(cfg)
    cfg.SEED = seed
    cfg.DRIVE_MODELS = run_dir / "models"
    cfg.DRIVE_LOGS = run_dir / "logs"
    for p in cfg.DRIVE_MODELS.glob("*.pth"):
        p.unlink()  # an interrupted run left checkpoints behind: start it over
    if (cfg.DRIVE_LOGS / "detection_log.csv").exists():
        (cfg.DRIVE_LOGS / "detection_log.csv").unlink()

    print(f"\n{'=' * 70}\n  {name}  seed={seed}  {CONFIGS[name][0] or 'defaults'}\n{'=' * 70}")
    _set_seed(seed)
    loaders = get_det_dataloaders(cfg)
    trainer = DetectionTrainer(cfg)  # model weights initialised after seeding
    trainer._set_loaders(*loaders)
    trainer.run(epochs=epochs, resume=False)

    # run() ends with the best-validation checkpoint loaded
    ck = torch.load(cfg.DRIVE_MODELS / "best_detection.pth", map_location=trainer.dev)
    trainer.model.load_state_dict(ck["model_state"])
    val_thr = float(ck.get("best_threshold", cfg.DETECTION_THRESHOLD))
    y_te, p_te = collect_val_probs(trainer.model, loaders[2], trainer.dev)
    ap = AudioProcessor(cfg)
    res = {
        "config": name, "seed": seed, "epochs": epochs, "best_epoch": ck.get("epoch"),
        "params": {**DEFAULTS, **CONFIGS[name][0]},
        "cache_counts": counts,
        "threshold": threshold, "val_selected_threshold": val_thr,
        "test": evaluate_binary_metrics(y_te, p_te, threshold),
        **eval_field(trainer.model, ap, field_files, threshold, trainer.dev),
        **time_cpu(trainer.model, ap),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "test_predictions.npz", y_true=y_te, y_prob=p_te, threshold=threshold)
    res_path.write_text(json.dumps(res, indent=2))
    return res


def summarise(out_root):
    rows, tex = [], []
    for name, (_, label) in CONFIGS.items():
        runs = [json.loads(p.read_text()) for p in sorted((out_root / name).glob("seed_*/results.json"))]
        if not runs:
            continue

        def ms(get):
            v = np.array([get(r) for r in runs if get(r) is not None], dtype=float)
            return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))

        f1 = ms(lambda r: r["test"]["f1"])
        acc = ms(lambda r: r["test"]["accuracy"])
        fld = ms(lambda r: r.get("field_det_rate"))
        ft = ms(lambda r: r["feature_ms"])
        rows.append({"config": name, "n_seeds": len(runs), "threshold": runs[0]["threshold"],
                     "test_f1_mean": f1[0], "test_f1_std": f1[1],
                     "test_acc_mean": acc[0], "test_acc_std": acc[1],
                     "field_det_rate_mean": fld[0], "field_det_rate_std": fld[1],
                     "feature_ms_mean": ft[0], "input_shape": runs[0]["input_shape"]})

        def pm(m, s, scale=1.0, d=3):
            return "--" if np.isnan(m) else f"{m * scale:.{d}f} $\\pm$ {s * scale:.{d}f}"
        tex.append(f"    {label:<45} & {pm(*f1)} & {pm(*acc, 100, 1)}\\% & "
                   f"{pm(*fld, 100, 1)}\\% & {ft[0]:.1f} \\\\")

    if not rows:
        print("No finished runs yet.")
        return
    with open(out_root / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_root / "stft_ablation_rows.tex").write_text("\n".join(tex) + "\n")
    print("\n" + "\n".join(tex))
    print(f"\n✅ {out_root / 'summary.csv'}\n✅ {out_root / 'stft_ablation_rows.tex'} "
          "(paste between the \\hline lines of tab:stft-ablation)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=base_config.NUM_EPOCHS)
    ap.add_argument("--threshold", type=float, default=0.945,
                    help="decision threshold for test and field metrics (thesis: tau = 0.945)")
    ap.add_argument("--field-dir", nargs="*", default=[], help="folders of drone-present field WAVs")
    ap.add_argument("--field-glob", default="*_ch0.wav", help="which files in --field-dir to score")
    ap.add_argument("--out", default=None, help="results folder (default: <DRIVE_ROOT>/stft_ablation)")
    ap.add_argument("--prepare", action="store_true", help="build processed WAVs, then exit")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    out_root = Path(a.out) if a.out else base_config.DRIVE_ROOT / "stft_ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    if a.prepare:
        prepare(make_cfg("default"))
        return
    if not a.summary_only:
        field_files = sorted(f for d in a.field_dir for f in Path(d).glob(a.field_glob))
        print(f"Field files: {len(field_files)}")
        for name in a.configs:
            for seed in a.seeds:
                run_one(name, seed, a.epochs, out_root, field_files, a.threshold)
    summarise(out_root)


if __name__ == "__main__":
    main()
