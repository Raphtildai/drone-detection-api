# -*- coding: utf-8 -*-
"""
drone_detection/training.py
────────────────────────────
Training managers and entry-point functions:
  - TrainingLogger
  - WarmupCosineScheduler
  - collect_val_probs / find_best_threshold / evaluate_binary_metrics
  - DetectionTrainer
  - LocalizationTrainer
  - train_localization_v2()  ← position-grouped + physics-aware synth + early-stopping
  - train_detection()
  - train_localization()
  - train_all()
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from drone_detection.config import config as _default_cfg
from drone_detection.datasets import (
    DatasetManifest,
    LocalizationDataset,
    MelCacheManager,
    SyntheticLocDataset,
    SyntheticLocDatasetV2,
    UaVirBASEDatasetManager,
    inject_synthetic_det_data,
    generate_mixed_drone_training_audio,
    get_det_dataloaders,
    report_detection_split_counts,
    audit_localization_labels,
    DroneAudioDatasetManager,
)
from drone_detection.models import (
    DetectionCNN,
    FocalLoss,
    localization_loss,
    make_localization_model,
)
from drone_detection.utils import set_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Training logger (CSV-backed, resumable)
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """Append-safe CSV logger that survives Colab disconnects."""

    def __init__(self, path: Path, columns: list) -> None:
        self.path    = path
        self.columns = columns
        self.rows: list[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    parsed: dict = {}
                    for k, v in row.items():
                        try:
                            parsed[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    self.rows.append(parsed)
            print(f"   📋 Resumed log from {path.name} ({len(self.rows)} rows)")
        else:
            print(f"   📋 New log: {path}")

    def log(self, **kwargs) -> None:
        self.rows.append(kwargs); self._flush()

    def _flush(self) -> None:
        tmp = self.path.parent / (self.path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader(); writer.writerows(self.rows)
        tmp.replace(self.path)

    def to_lists(self) -> Dict[str, list]:
        out = {c: [] for c in self.columns}
        for row in self.rows:
            for c in self.columns:
                out[c].append(row.get(c, float("nan")))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# LR schedule
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup → cosine annealing to min_lr_factor × base_lr."""

    def __init__(
        self, optimizer, warmup_epochs: int, total_epochs: int,
        min_lr_factor: float = 0.05, last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs  = max(total_epochs, self.warmup_epochs + 1)
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            scale = (ep + 1) / self.warmup_epochs
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale    = self.min_lr_factor + 0.5 * (1 - self.min_lr_factor) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════════
# Detection metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collect_val_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            ys.extend(y.numpy()); ps.extend(probs)
    return np.asarray(ys), np.asarray(ps)


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0
) -> Tuple[float, float]:
    """Grid-search F-beta threshold on validation probabilities."""
    best_t = 0.5; best_score = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= t).astype(np.int64)
        tp   = np.sum((pred == 1) & (y_true == 1))
        fp   = np.sum((pred == 1) & (y_true == 0))
        fn   = np.sum((pred == 0) & (y_true == 1))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        fb   = 0.0 if prec == 0 and rec == 0 else (
            (1 + beta**2) * prec * rec / max(beta**2 * prec + rec, 1e-8)
        )
        if fb > best_score: best_score = fb; best_t = float(t)
    return best_t, best_score


def evaluate_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((pred==1)&(y_true==1))); fp = int(np.sum((pred==1)&(y_true==0)))
    tn = int(np.sum((pred==0)&(y_true==0))); fn = int(np.sum((pred==0)&(y_true==1)))
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)
    acc  = (tp+tn)/max(tp+tn+fp+fn,1)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auroc = auprc = float("nan")
    return {"threshold":float(threshold),"accuracy":float(acc),"precision":float(prec),
            "recall":float(rec),"f1":float(f1),"auroc":auroc,"auprc":auprc,
            "tp":tp,"fp":fp,"tn":tn,"fn":fn}


def print_detection_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> None:
    m    = evaluate_binary_metrics(y_true, y_prob, threshold)
    pred = (y_prob >= threshold).astype(np.int64)
    print("\n=== Detection validation report ===")
    for k in ["threshold","accuracy","precision","recall","f1","auroc","auprc"]:
        print(f"  {k:10s}: {m[k]:.4f}")
    print(f"  TP/FP/TN/FN = {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    print(classification_report(y_true, pred, target_names=["non_drone","drone"], digits=4))
    print(confusion_matrix(y_true, pred))


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionTrainer
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionTrainer:
    """
    Trains DetectionCNN with:
      - FocalLoss (γ=2, α=0.6, label_smoothing=0.02)
      - CosineAnnealingLR
      - Auto-threshold search on validation F1
      - Gradient clipping (max_norm=2)
      - AMP support
      - CSV training log
      - Checkpoint resume
    """

    def __init__(self, cfg=None) -> None:
        self.cfg   = cfg or _default_cfg
        self.dev   = torch.device(self.cfg.DEVICE)
        self.model = DetectionCNN().to(self.dev)
        self._loaders = None

    def _set_loaders(self, tr_l, va_l, te_l) -> None:
        self._loaders = (tr_l, va_l, te_l)

    def run(self, epochs: int = None, resume: bool = True) -> None:
        epochs = epochs or self.cfg.NUM_EPOCHS
        set_seed(self.cfg.SEED)

        if self._loaders is not None:
            tr_l, va_l, te_l = self._loaders
        else:
            tr_l, va_l, te_l = get_det_dataloaders(self.cfg)

        opt    = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        crit   = FocalLoss(gamma=2.0, alpha=0.6, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda")
        logger = TrainingLogger(
            self.cfg.DRIVE_LOGS / "detection_log.csv",
            columns=["epoch","tr_loss","tr_acc","val_acc","val_f1","val_thr","val_auprc"],
        )
        logged_epochs = {int(r["epoch"]) for r in logger.rows}
        start_epoch   = 1; best_f1 = 0.0; best_thr = self.cfg.DETECTION_THRESHOLD
        ckpt_path     = self.cfg.DRIVE_MODELS / "best_detection.pth"
        latest_path   = self.cfg.DRIVE_MODELS / "latest_detection.pth"

        if resume and latest_path.exists():
            ck = torch.load(latest_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            best_f1     = float(ck.get("best_val_f1", 0.0))
            best_thr    = float(ck.get("best_threshold", self.cfg.DETECTION_THRESHOLD))
            print(f"▶️  Resuming detection from epoch {start_epoch} (best f1: {best_f1:.4f}, thr: {best_thr:.3f})")

        for ep in range(start_epoch, start_epoch + epochs):
            self.model.train(); loss_sum = correct = total = 0
            pbar = tqdm(tr_l, desc=f"Det train ep {ep}", leave=False)
            for X, y in pbar:
                X = X.to(self.dev, non_blocking=True); y = y.to(self.dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda"):
                    out = self.model(X); loss = crit(out, y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                loss_sum += loss.item() * X.size(0)
                correct  += (out.argmax(1) == y).sum().item(); total += X.size(0)
                pbar.set_postfix({"loss": f"{loss_sum/max(total,1):.4f}", "acc": f"{100*correct/max(total,1):.1f}%"})

            tr_loss = loss_sum / max(total, 1); tr_acc = 100 * correct / max(total, 1)
            y_val, p_val = collect_val_probs(self.model, va_l, self.dev)
            thr, _       = find_best_threshold(y_val, p_val)
            val_metrics  = evaluate_binary_metrics(y_val, p_val, thr)
            val_acc = 100.0 * val_metrics["accuracy"]; val_f1 = val_metrics["f1"]
            sched.step()
            print(f"  Det Ep {ep:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1f}%  val_acc={val_acc:.1f}%  val_f1={val_f1:.4f} thr={thr:.3f}")

            self._save(latest_path, ep, val_metrics)
            if ep not in logged_epochs:
                logger.log(epoch=ep, tr_loss=round(tr_loss,6), tr_acc=round(tr_acc,3),
                           val_acc=round(val_acc,3), val_f1=round(val_f1,6),
                           val_thr=round(thr,4),
                           val_auprc=round(val_metrics["auprc"],6) if not math.isnan(val_metrics["auprc"]) else "nan")
                logged_epochs.add(ep)

            if val_f1 > best_f1:
                best_f1 = val_f1; best_thr = thr
                self._save(ckpt_path, ep, val_metrics)
                print("   ✨ New best!")
                self.cfg.DETECTION_THRESHOLD = float(best_thr)

        print("\n🎯 Final detection test:")
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            test_thr = float(ck.get("best_threshold", self.cfg.DETECTION_THRESHOLD))
        else:
            test_thr = self.cfg.DETECTION_THRESHOLD
        y_test, p_test = collect_val_probs(self.model, te_l, self.dev)
        print_detection_report(y_test, p_test, test_thr)
        self.cfg.DETECTION_THRESHOLD = float(test_thr)

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "epoch":           epoch,
            "best_val_acc":    float(metrics.get("accuracy", 0.0) * 100.0),
            "best_val_f1":     float(metrics.get("f1", 0.0)),
            "best_threshold":  float(metrics.get("threshold", self.cfg.DETECTION_THRESHOLD)),
            "val_metrics":     metrics,
        }, path)
        print(f"💾 Saved {path.name} (ep={epoch}, f1={metrics.get('f1',0):.4f}, thr={metrics.get('threshold',0):.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# LocalizationTrainer
# ═══════════════════════════════════════════════════════════════════════════════

class LocalizationTrainer:
    """Standard localisation trainer (used by train_localization() / train_all())."""

    def __init__(self, cfg=None) -> None:
        self.cfg   = cfg or _default_cfg
        self.dev   = torch.device(self.cfg.DEVICE)
        self.model = make_localization_model(self.cfg).to(self.dev)

    def run(
        self, data_root: Path, epochs: int = None,
        use_synthetic_fallback: bool = True, resume: bool = True,
    ) -> None:
        epochs = epochs or self.cfg.NUM_EPOCHS
        set_seed(self.cfg.SEED)
        bs = self.cfg.BATCH_SIZE
        if getattr(self.cfg, "USE_LITE_LOC", False): bs = max(8, bs // 2)

        try:
            tr_real  = LocalizationDataset(data_root, "train", augment=True, cfg=self.cfg)
            va       = LocalizationDataset(data_root, "val",   augment=False, cfg=self.cfg)
            te       = LocalizationDataset(data_root, "test",  augment=False, cfg=self.cfg)
            tr_synth = SyntheticLocDataset(self.cfg, n_samples=100, augment=True)
            tr       = ConcatDataset([tr_real, tr_synth])
            print(f"   📊 Train: {len(tr_real)} real + {len(tr_synth)} synthetic = {len(tr)} total")
        except RuntimeError as e:
            if not use_synthetic_fallback: print(f"❌ {e}"); return
            print("⚠️  Real data unavailable — synthetic fallback.")
            tr = SyntheticLocDataset(self.cfg, 500, True)
            va = SyntheticLocDataset(self.cfg, 50,  False)
            te = SyntheticLocDataset(self.cfg, 50,  False)

        tr_l = DataLoader(tr, batch_size=bs, shuffle=True,  drop_last=True, num_workers=0)
        va_l = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)
        te_l = DataLoader(te, batch_size=bs, shuffle=False, num_workers=0)

        opt    = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=4, factor=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda")
        logger = TrainingLogger(
            self.cfg.DRIVE_LOGS / "localization_log.csv",
            columns=["epoch","tr_loss","val_loss","mae_az","mae_dist","mae_ht"],
        )
        logged_epochs = {int(r["epoch"]) for r in logger.rows}
        start_epoch   = 1; best_val = 1e9
        ckpt_path     = self.cfg.DRIVE_MODELS / "best_localization.pth"
        latest_path   = self.cfg.DRIVE_MODELS / "latest_localization.pth"

        if resume and latest_path.exists():
            ck = torch.load(latest_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            if ckpt_path.exists():
                best_val = torch.load(ckpt_path, map_location=self.dev).get("best_val_loss", 1e9)
            print(f"▶️  Resuming localization from epoch {start_epoch} (best val_loss: {best_val:.5f})")

        for ep in range(start_epoch, start_epoch + epochs):
            self.model.train(); loss_sum = n = 0
            for mel, ipd, lbl in tqdm(tr_l, desc=f"Loc train ep {ep}", leave=False):
                mel = mel.to(self.dev); ipd = ipd.to(self.dev); lbl = lbl.to(self.dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda"):
                    pred = self.model(mel, ipd); loss = localization_loss(pred, lbl)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                loss_sum += loss.item() * mel.size(0); n += mel.size(0)

            tr_loss = loss_sum / max(n, 1)
            val_loss, mae_az, mae_dist, mae_ht = self._eval(va_l)
            sched.step(val_loss)
            print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f} val={val_loss:.5f} az={mae_az:.2f}° dist={mae_dist:.3f}m ht={mae_ht:.3f}m")
            self._save(latest_path, ep, val_loss)
            if ep not in logged_epochs:
                logger.log(epoch=ep, tr_loss=round(tr_loss,6), val_loss=round(val_loss,6),
                           mae_az=round(mae_az,4), mae_dist=round(mae_dist,4), mae_ht=round(mae_ht,4))
                logged_epochs.add(ep)
            if val_loss < best_val:
                best_val = val_loss; self._save(ckpt_path, ep, val_loss); print("   ✨ New best!")

        print("\n🎯 Final localization test:"); self._eval(te_l, verbose=True)

    def _eval(self, loader, verbose=False) -> Tuple[float, float, float, float]:
        from drone_detection.utils import angular_error_deg
        self.model.eval(); loss_sum = n = 0
        az_errs: list[float] = []; dist_errs: list[float] = []; ht_errs: list[float] = []
        with torch.no_grad():
            for mel, ipd, lbl in tqdm(loader, desc="Loc eval", leave=False):
                mel = mel.to(self.dev); ipd = ipd.to(self.dev); lbl = lbl.to(self.dev)
                pred = self.model(mel, ipd)
                loss_sum += localization_loss(pred, lbl).item() * mel.size(0); n += mel.size(0)
                p = pred.cpu().numpy(); t = lbl.cpu().numpy()
                az_errs.extend(angular_error_deg(np.degrees(np.arctan2(p[:,0], p[:,1])),
                                                  np.degrees(np.arctan2(t[:,0], t[:,1]))).tolist())
                dist_errs.extend(np.abs(p[:,2] - t[:,2]).tolist())
                ht_errs.extend(  np.abs(p[:,3] - t[:,3]).tolist())
        val_loss  = loss_sum / max(n, 1)
        mae_az    = float(np.mean(az_errs))
        mae_dist  = float(np.mean(dist_errs)) * self.cfg.MAX_LOCALIZATION_DIST
        mae_ht    = float(np.mean(ht_errs))   * self.cfg.MAX_LOCALIZATION_DIST
        if verbose:
            print(f"  Val loss={val_loss:.5f}  MAE az={mae_az:.2f}°  dist={mae_dist:.3f}m  ht={mae_ht:.3f}m")
        return val_loss, mae_az, mae_dist, mae_ht

    def _save(self, path: Path, epoch: int, metric: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "epoch": epoch, "best_val_loss": metric}, path)
        print(f"💾 Saved {path.name} (ep={epoch}, val_loss={metric:.5f})")


# ═══════════════════════════════════════════════════════════════════════════════
# train_localization_v2 — position-grouped + physics-aware synth + early-stopping
# ═══════════════════════════════════════════════════════════════════════════════

def train_localization_v2(
    cfg=None,
    epochs: int               = None,
    resume: bool              = True,
    reset_best: bool          = False,
    n_synthetic_train: int    = 2000,
    n_synthetic_val: int      = 200,
    grid_fraction: float      = 0.55,
    az_jitter_deg: float      = 18.0,
    dist_jitter_m: float      = 3.0,
    ht_jitter_m: float        = 2.5,
    force_redownload: bool    = False,
    save_manifest: bool       = True,
    early_stop_patience: int  = 15,
    warmup_epochs: int        = 3,
    min_lr_factor: float      = 0.05,
) -> DatasetManifest:
    """
    Full localisation training pipeline:
      - Position-grouped real data (val/test positions never seen in train)
      - SyntheticLocDatasetV2 (physics-aware, grid-conditioned)
      - WarmupCosine LR schedule
      - Early stopping
      - num_workers=0 (no multiprocessing assertion errors in Colab)
      - DatasetManifest saved to Drive logs

    Returns the populated DatasetManifest for thesis reporting.
    """
    import shutil
    cfg = cfg or _default_cfg
    cfg.ensure_dirs()
    set_seed(cfg.SEED)
    total_epochs = epochs or cfg.NUM_EPOCHS

    print("=" * 70)
    print("  STAGE 2 — Localization v2  (position-grouped + physics synthesis)")
    print("=" * 70)

    manifest = DatasetManifest(
        epochs=total_epochs, batch_size=cfg.BATCH_SIZE, learning_rate=cfg.LR, cfg_seed=cfg.SEED,
        synthetic_n_train=n_synthetic_train, synthetic_n_val=n_synthetic_val,
        synthetic_grid_fraction=grid_fraction, synthetic_az_jitter_deg=az_jitter_deg,
        synthetic_dist_jitter_m=dist_jitter_m, synthetic_ht_jitter_m=ht_jitter_m,
    )

    proc = cfg.PROCESSED_DIR / "localization"
    um   = UaVirBASEDatasetManager(cfg)
    if force_redownload and proc.exists():
        shutil.rmtree(proc)
    try:
        um.prepare()
    except Exception as e:
        print(f"⚠️  UaVirBASE prepare failed ({e}) → synthetic-only fallback")

    # Populate manifest with real data stats
    real_counts:  Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    real_positions: set = set()
    for split in ["train", "val", "test"]:
        lfiles = list((proc/split).glob("*_label.json")) if (proc/split).exists() else []
        real_counts[split] = len(lfiles)
        for lf in lfiles:
            try:
                d = json.loads(lf.read_text())
                real_positions.add((round(d["azimuth_deg"]), round(d["distance_m"]), round(d["height_m"])))
            except Exception: pass

    manifest.real_drone_sessions   = sum(real_counts.values())
    manifest.real_ambient_sessions = 4
    manifest.real_total_sessions   = manifest.real_drone_sessions + manifest.real_ambient_sessions
    manifest.real_unique_positions = len(real_positions)
    manifest.real_az_values_deg    = sorted({p[0] for p in real_positions})
    manifest.real_dist_values_m    = sorted({p[1] for p in real_positions})
    manifest.real_ht_values_m      = sorted({p[2] for p in real_positions})
    manifest.real_train_sessions   = real_counts["train"]
    manifest.real_val_sessions     = real_counts["val"]
    manifest.real_test_sessions    = real_counts["test"]

    # Build datasets
    synth_train = SyntheticLocDatasetV2(cfg, n_samples=n_synthetic_train, grid_fraction=grid_fraction,
                                         augment=True, az_jitter_deg=az_jitter_deg, dist_jitter_m=dist_jitter_m,
                                         ht_jitter_m=ht_jitter_m, seed=manifest.synthetic_rng_seed)
    synth_val   = SyntheticLocDatasetV2(cfg, n_samples=n_synthetic_val, grid_fraction=grid_fraction,
                                         augment=False, seed=manifest.synthetic_rng_seed + 1)
    try:
        real_train = LocalizationDataset(proc, "train", augment=True,  cfg=cfg)
        real_val   = LocalizationDataset(proc, "val",   augment=False, cfg=cfg)
        real_test  = LocalizationDataset(proc, "test",  augment=False, cfg=cfg)
        print(f"\n   Real data — train: {len(real_train)}  val: {len(real_val)}  test: {len(real_test)}")
        tr_ds = ConcatDataset([real_train, synth_train])
        va_ds = ConcatDataset([real_val,   synth_val])
        te_ds = real_test
    except RuntimeError as e:
        print(f"⚠️  Real data unavailable ({e}) — synthetic only")
        tr_ds = synth_train; va_ds = synth_val
        te_ds = SyntheticLocDatasetV2(cfg, n_samples=200, augment=False, seed=manifest.synthetic_rng_seed+2)

    print(f"   Train total : {len(tr_ds)}  (real + {n_synthetic_train} synthetic)")
    print(f"   Val total   : {len(va_ds)}  (real + {n_synthetic_val} synthetic)")
    print(f"   Test        : {len(te_ds)}  (real only — thesis primary metric)")

    bs = cfg.BATCH_SIZE
    if getattr(cfg, "USE_LITE_LOC", False): bs = max(8, bs // 2)
    lkw = dict(batch_size=bs, num_workers=0, pin_memory=False)
    tr_l = DataLoader(tr_ds, shuffle=True,  drop_last=True,  **lkw)
    va_l = DataLoader(va_ds, shuffle=False, drop_last=False, **lkw)
    te_l = DataLoader(te_ds, shuffle=False, drop_last=False, **lkw)

    trainer = LocalizationTrainer(cfg)
    dev     = trainer.dev

    if reset_best:
        ckpt = cfg.DRIVE_MODELS / "best_localization.pth"
        if ckpt.exists():
            ck = torch.load(ckpt, map_location=cfg.DEVICE); ck["best_val_loss"] = 1e9
            torch.save(ck, ckpt); print("🔄 best_val_loss reset to 1e9")

    opt    = torch.optim.AdamW(trainer.model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP and cfg.DEVICE == "cuda")
    logger = TrainingLogger(cfg.DRIVE_LOGS/"localization_log.csv",
                            columns=["epoch","tr_loss","val_loss","mae_az","mae_dist","mae_ht"])
    logged_epochs = {int(r["epoch"]) for r in logger.rows}
    start_epoch   = 1; best_val = 1e9
    ckpt_path     = cfg.DRIVE_MODELS / "best_localization.pth"
    latest_path   = cfg.DRIVE_MODELS / "latest_localization.pth"

    if resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=dev)
        trainer.model.load_state_dict(ck["model_state"])
        start_epoch = ck.get("epoch", 1) + 1
        if ckpt_path.exists():
            best_val = torch.load(ckpt_path, map_location=dev).get("best_val_loss", 1e9)
        print(f"▶️  Resuming localization from epoch {start_epoch} (best val_loss: {best_val:.5f})")

    sched = WarmupCosineScheduler(opt, warmup_epochs=warmup_epochs, total_epochs=total_epochs,
                                  min_lr_factor=min_lr_factor, last_epoch=-1)
    no_improve = 0
    ep = start_epoch

    for ep in range(start_epoch, start_epoch + total_epochs):
        trainer.model.train(); loss_sum = n_items = 0
        for mel, ipd, lbl in tqdm(tr_l, desc=f"Loc train ep {ep}", leave=False):
            mel = mel.to(dev); ipd = ipd.to(dev); lbl = lbl.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP and cfg.DEVICE == "cuda"):
                pred = trainer.model(mel, ipd); loss = localization_loss(pred, lbl)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * mel.size(0); n_items += mel.size(0)

        tr_loss = loss_sum / max(n_items, 1)
        val_loss, mae_az, mae_dist, mae_ht = trainer._eval(va_l)
        sched.step()
        current_lr = opt.param_groups[0]["lr"]
        print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f} val={val_loss:.5f} "
              f"az={mae_az:.2f}° dist={mae_dist:.3f}m ht={mae_ht:.3f}m lr={current_lr:.2e}")
        trainer._save(latest_path, ep, val_loss)
        if ep not in logged_epochs:
            logger.log(epoch=ep, tr_loss=round(tr_loss,6), val_loss=round(val_loss,6),
                       mae_az=round(mae_az,4), mae_dist=round(mae_dist,4), mae_ht=round(mae_ht,4))
            logged_epochs.add(ep)
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            trainer._save(ckpt_path, ep, val_loss); print("   ✨ New best!")
        else:
            no_improve += 1
            print(f"   (no improvement for {no_improve}/{early_stop_patience} epochs)")
        if no_improve >= early_stop_patience:
            print(f"\n🛑 Early stopping at epoch {ep}.")
            break

    print("\n🎯 Final localization test (real-only test set):")
    trainer._eval(te_l, verbose=True)
    manifest.epochs = ep
    if save_manifest:
        manifest.save(cfg.DRIVE_LOGS / "dataset_manifest.json")
        manifest.print_thesis_summary()
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Public training entry points
# ═══════════════════════════════════════════════════════════════════════════════

def train_detection(
    cfg=None,
    epochs: int               = None,
    resume: bool              = True,
    force_rebuild_cache: bool = False,
    force_regen_mixed_audio: bool = False,
    custom_builder_root: str  = None,
    use_custom_builder: bool  = None,
    import_custom_backgrounds_as_non_drone: bool = None,
    download_builtin_detection_dataset: bool     = None,
    download_external_audio: bool                = None,
) -> None:
    """Run Stage 1: detection model training with the full data pipeline."""
    from drone_detection.audio import AudioWebScraper, incorporate_scraped_audio

    cfg = cfg or _default_cfg
    cfg.ensure_dirs()
    set_seed(cfg.SEED)

    if custom_builder_root is not None:
        cfg.CUSTOM_DATASET_ROOT = str(custom_builder_root)
    if use_custom_builder is None:
        use_custom_builder = bool(cfg.CUSTOM_DATASET_ROOT)
    if import_custom_backgrounds_as_non_drone is not None:
        cfg.CUSTOM_DATASET_COPY_BACKGROUNDS_AS_NON_DRONE = bool(import_custom_backgrounds_as_non_drone)
    if download_builtin_detection_dataset is None:
        download_builtin_detection_dataset = bool(cfg.ALLOW_BUILTIN_DETECTION_DATASET_DOWNLOAD)
    if download_external_audio is None:
        download_external_audio = bool(cfg.ALLOW_EXTERNAL_AUDIO_DOWNLOADS)

    print("=" * 70)
    print("  STAGE 1 — Detection Model")
    print("=" * 70)

    if download_builtin_detection_dataset:
        DroneAudioDatasetManager(cfg).prepare()
    else:
        print("ℹ️  Skipping built-in DroneAudioDataset download.")
        for split in ["train", "val", "test"]:
            for label in ["drone", "non_drone"]:
                (cfg.PROCESSED_DIR/"detection"/split/label).mkdir(parents=True, exist_ok=True)

    if download_external_audio:
        AudioWebScraper(cfg).download(force=False)
        incorporate_scraped_audio(cfg, force=False)
    else:
        print("ℹ️  Skipping external audio scraping.")

    if use_custom_builder and cfg.CUSTOM_DATASET_ROOT:
        from drone_detection.custom_builder import import_custom_builder_dataset
        import_custom_builder_dataset(cfg, cfg.CUSTOM_DATASET_ROOT)

    generate_mixed_drone_training_audio(cfg, force=force_regen_mixed_audio)
    report_detection_split_counts(cfg)
    mcm = MelCacheManager(cfg); mcm.build(force=force_rebuild_cache)
    inject_synthetic_det_data(cfg, force=False)
    counts = mcm.count()
    print("\n📊 MEL CACHE CLASS BALANCE")
    for key, n in counts.items(): print(f"  {key:28s}  {n:6d}")
    print()
    try:
        tr_l, va_l, te_l = get_det_dataloaders(cfg)
    except RuntimeError as e:
        print(f"❌ Could not build dataloaders: {e}"); return
    tr = DetectionTrainer(cfg); tr._set_loaders(tr_l, va_l, te_l); tr.run(epochs=epochs, resume=resume)


def train_localization(
    cfg=None,
    epochs: int = None,
    use_synthetic_fallback: bool = True,
    resume: bool = True,
    reset_best: bool = False,
) -> None:
    """Run Stage 2: localisation model training (standard pipeline)."""
    cfg = cfg or _default_cfg; cfg.ensure_dirs(); set_seed(cfg.SEED)
    print("=" * 65); print("  STAGE 2 — Localization Model (UaVirBASE)"); print("=" * 65)
    try:
        UaVirBASEDatasetManager(cfg).prepare()
    except Exception as e:
        print(f"⚠️  UaVirBASE prepare failed: {e}")
    if reset_best:
        ckpt = cfg.DRIVE_MODELS/"best_localization.pth"
        if ckpt.exists():
            ck = torch.load(ckpt, map_location=cfg.DEVICE); ck["best_val_loss"] = 1e9
            torch.save(ck, ckpt); print("🔄 best_val_loss reset to 1e9")
    tr = LocalizationTrainer(cfg)
    tr.run(cfg.PROCESSED_DIR/"localization", epochs=epochs,
           use_synthetic_fallback=use_synthetic_fallback, resume=resume)


def train_all(
    cfg=None,
    det_epochs: int  = 5,
    loc_epochs: int  = 5,
    use_synthetic_loc: bool = True,
    resume: bool     = True,
    force_rebuild_cache: bool = False,
    **detection_kwargs,
) -> None:
    """Run both training stages end-to-end."""
    train_detection(cfg, det_epochs, resume=resume,
                    force_rebuild_cache=force_rebuild_cache, **detection_kwargs)
    train_localization(cfg, loc_epochs, use_synthetic_loc, resume=resume)
    print("\n✅ Both models trained.")