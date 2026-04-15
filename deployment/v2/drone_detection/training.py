# -*- coding: utf-8 -*-
"""
training.py
───────────
Training managers for detection and localization models.

Classes
───────
TrainingLogger        — CSV-backed epoch logger with resume support
WarmupCosineScheduler — linear warmup + cosine annealing LR schedule
DetectionTrainer      — v15 detection training (FocalLoss + threshold search)
LocalizationTrainer   — v15 localization training
train_localization_v2 — full pipeline with position-grouped data + early stop

Metric helpers
──────────────
collect_val_probs()    — gather (y_true, y_prob) from a DataLoader
find_best_threshold()  — maximise F-beta on validation probabilities
evaluate_binary_metrics() — precision/recall/F1/AUROC/AUPRC
print_detection_report()  — pretty-print full classification report
"""

import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from .config import Config, config
from .models import (
    DetectionCNN,
    FocalLoss,
    localization_loss,
    make_localization_model,
)
from .utils import _set_seed, angular_error_deg


# ══════════════════════════════════════════════════════════════════════════════
# Training logger
# ══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Append-only CSV logger with automatic resume.

    Usage
    ─────
    logger = TrainingLogger(path, columns=["epoch","tr_loss","val_acc"])
    logger.log(epoch=1, tr_loss=0.45, val_acc=91.2)
    """

    def __init__(self, path: Path, columns: List[str]):
        self.path    = path
        self.columns = columns
        self.rows: List[Dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)

        # Verify write access early
        test_file = path.parent / ".write_test"
        try:
            test_file.write_text("ok"); test_file.unlink()
        except OSError as e:
            raise RuntimeError(f"Log directory {path.parent} not writable: {e}")

        if path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = {}
                    for k, v in row.items():
                        try:
                            parsed[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                        except (ValueError, AttributeError):
                            parsed[k] = v
                    self.rows.append(parsed)
            print(f"   📋 Resumed log from {path.name} ({len(self.rows)} existing rows)")
        else:
            print(f"   📋 New log: {path}")

    def log(self, **kwargs):
        self.rows.append(kwargs)
        self._flush()

    def _flush(self):
        tmp = self.path.parent / (self.path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        tmp.replace(self.path)

    def to_lists(self) -> Dict[str, List]:
        out = {c: [] for c in self.columns}
        for row in self.rows:
            for c in self.columns:
                out[c].append(row.get(c, float("nan")))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# LR scheduler
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to
    `min_lr_factor * base_lr`.

    Parameters
    ──────────
    warmup_epochs : epochs to ramp from 0 → base_lr
    total_epochs  : total training epochs (warmup + cosine phase)
    min_lr_factor : cosine floor as a fraction of base_lr
    """

    def __init__(
        self,
        optimizer:      torch.optim.Optimizer,
        warmup_epochs:  int,
        total_epochs:   int,
        min_lr_factor:  float = 0.05,
        last_epoch:     int   = -1,
    ):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs  = max(total_epochs, self.warmup_epochs + 1)
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            scale = (ep + 1) / self.warmup_epochs
        else:
            progress = (ep - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = self.min_lr_factor + 0.5 * (1 - self.min_lr_factor) * (
                1 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def collect_val_probs(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (y_true, y_prob_drone)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x     = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(probs.tolist())
    return np.asarray(ys), np.asarray(ps)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta:   float = 1.0,
) -> Tuple[float, float]:
    """
    Grid-search the threshold that maximises F-beta on the given set.

    Returns
    ───────
    best_threshold, best_fbeta_score
    """
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= t).astype(np.int64)
        tp   = np.sum((pred == 1) & (y_true == 1))
        fp   = np.sum((pred == 1) & (y_true == 0))
        fn   = np.sum((pred == 0) & (y_true == 1))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        fbeta = (
            (1 + beta ** 2) * prec * rec / max(beta ** 2 * prec + rec, 1e-8)
            if (prec + rec) > 0 else 0.0
        )
        if fbeta > best_score:
            best_score, best_t = fbeta, float(t)
    return best_t, best_score


def evaluate_binary_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float,
) -> Dict:
    """Return a dict of precision, recall, F1, accuracy, AUROC, AUPRC, TP/FP/TN/FN."""
    pred = (y_prob >= threshold).astype(np.int64)
    tp   = int(np.sum((pred == 1) & (y_true == 1)))
    fp   = int(np.sum((pred == 1) & (y_true == 0)))
    tn   = int(np.sum((pred == 0) & (y_true == 0)))
    fn   = int(np.sum((pred == 0) & (y_true == 1)))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auroc = auprc = float("nan")
    return {
        "threshold": float(threshold), "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "auroc": auroc, "auprc": auprc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def print_detection_report(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float,
):
    """Pretty-print detection metrics to stdout."""
    m    = evaluate_binary_metrics(y_true, y_prob, threshold)
    pred = (y_prob >= threshold).astype(np.int64)
    print("\n=== Detection validation report ===")
    print(f"Threshold : {m['threshold']:.3f}")
    print(f"Accuracy  : {m['accuracy']:.4f}")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1        : {m['f1']:.4f}")
    print(f"AUROC     : {m['auroc']:.4f}")
    print(f"AUPRC     : {m['auprc']:.4f}")
    print(f"TP/FP/TN/FN = {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    print(classification_report(y_true, pred, target_names=["non_drone", "drone"], digits=4))
    print(confusion_matrix(y_true, pred))


# ══════════════════════════════════════════════════════════════════════════════
# Detection trainer (v15)
# ══════════════════════════════════════════════════════════════════════════════

class DetectionTrainer:
    """
    Manages the full detection training loop.

    v15 improvements over v13
    ─────────────────────────
    - FocalLoss(γ=2, α=0.6, label_smoothing=0.02) replaces CrossEntropyLoss
    - CosineAnnealingLR replaces ReduceLROnPlateau
    - Threshold auto-searched on val set each epoch (stored in checkpoint)
    - Checkpoint keys: best_val_f1, best_threshold, val_metrics
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg      = cfg or config
        self.dev      = torch.device(self.cfg.DEVICE)
        self.model    = DetectionCNN().to(self.dev)
        self._loaders = None

    def _set_loaders(self, tr_l, va_l, te_l):
        self._loaders = (tr_l, va_l, te_l)

    def run(self, epochs: Optional[int] = None, resume: bool = True):
        epochs = epochs or self.cfg.NUM_EPOCHS
        _set_seed(self.cfg.SEED)

        if self._loaders is not None:
            tr_l, va_l, te_l = self._loaders
        else:
            from .datasets import DetectionDataset
            from torch.utils.data import WeightedRandomSampler
            data_root = self.cfg.PROCESSED_DIR / "detection"
            try:
                tr = DetectionDataset(data_root, "train", augment=True,  cfg=self.cfg)
                va = DetectionDataset(data_root, "val",   augment=False, cfg=self.cfg)
                te = DetectionDataset(data_root, "test",  augment=False, cfg=self.cfg)
            except RuntimeError as e:
                print(f"❌ {e}"); return
            lbs     = np.array(tr.labels)
            cnt     = np.bincount(lbs); cnt[cnt == 0] = 1
            weights = (1.0 / cnt)[lbs]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            def _col(batch): xs, ys = zip(*batch); return torch.stack(xs), torch.stack(ys)
            tr_l = DataLoader(tr, batch_size=self.cfg.BATCH_SIZE, sampler=sampler, collate_fn=_col)
            va_l = DataLoader(va, batch_size=self.cfg.BATCH_SIZE, shuffle=False,   collate_fn=_col)
            te_l = DataLoader(te, batch_size=self.cfg.BATCH_SIZE, shuffle=False,   collate_fn=_col)

        opt    = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        crit   = FocalLoss(gamma=2.0, alpha=0.6, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda")
        logger = TrainingLogger(
            self.cfg.DRIVE_LOGS / "detection_log.csv",
            columns=["epoch", "tr_loss", "tr_acc", "val_acc", "val_f1", "val_thr", "val_auprc"],
        )
        logged_epochs = {int(r["epoch"]) for r in logger.rows}
        start_epoch   = 1
        best_f1       = 0.0
        best_thr      = self.cfg.DETECTION_THRESHOLD
        ckpt_path     = self.cfg.DRIVE_MODELS / "best_detection.pth"
        latest_path   = self.cfg.DRIVE_MODELS / "latest_detection.pth"

        if resume and latest_path.exists():
            ck = torch.load(latest_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            best_f1     = float(ck.get("best_val_f1", 0.0))
            best_thr    = float(ck.get("best_threshold", self.cfg.DETECTION_THRESHOLD))
            print(f"▶️  Resuming detection from epoch {start_epoch} "
                  f"(best f1: {best_f1:.4f}, thr: {best_thr:.3f})")
        elif resume and ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            best_f1     = float(ck.get("best_val_f1", 0.0))
            best_thr    = float(ck.get("best_threshold", self.cfg.DETECTION_THRESHOLD))

        for ep in range(start_epoch, start_epoch + epochs):
            self.model.train()
            loss_sum = correct = total = 0
            pbar = tqdm(tr_l, desc=f"Det train ep {ep}", leave=False)
            for X, y in pbar:
                X = X.to(self.dev, non_blocking=True)
                y = y.to(self.dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda"):
                    out  = self.model(X)
                    loss = crit(out, y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                loss_sum += loss.item() * X.size(0)
                correct  += (out.argmax(1) == y).sum().item()
                total    += X.size(0)
                pbar.set_postfix({
                    "loss": f"{loss_sum/max(total,1):.4f}",
                    "acc":  f"{100*correct/max(total,1):.1f}%",
                })
            tr_loss = loss_sum / max(total, 1)
            tr_acc  = 100 * correct / max(total, 1)
            sched.step()

            y_val, p_val = collect_val_probs(self.model, va_l, self.dev)
            thr, _       = find_best_threshold(y_val, p_val, beta=1.0)
            val_metrics  = evaluate_binary_metrics(y_val, p_val, thr)
            val_acc      = 100.0 * val_metrics["accuracy"]
            val_f1       = val_metrics["f1"]
            current_lr   = opt.param_groups[0]["lr"]

            print(f"  Det Ep {ep:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1f}%  "
                  f"val_acc={val_acc:.1f}%  val_f1={val_f1:.4f}  thr={thr:.3f}  lr={current_lr:.2e}")
            self._save(latest_path, ep, val_metrics)

            if ep not in logged_epochs:
                logger.log(
                    epoch=ep, tr_loss=round(tr_loss, 6), tr_acc=round(tr_acc, 3),
                    val_acc=round(val_acc, 3), val_f1=round(val_f1, 6),
                    val_thr=round(thr, 4),
                    val_auprc=round(val_metrics["auprc"], 6)
                    if not math.isnan(val_metrics["auprc"]) else "nan",
                )
                logged_epochs.add(ep)

            if val_f1 > best_f1:
                best_f1, best_thr = val_f1, thr
                self._save(ckpt_path, ep, val_metrics)
                self.cfg.DETECTION_THRESHOLD = float(best_thr)
                print("   ✨ New best!")

        # Final test evaluation using best checkpoint
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

    def _save(self, path: Path, epoch: int, metrics: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "epoch":           epoch,
            "best_val_acc":    float(metrics.get("accuracy", 0.0) * 100.0),
            "best_val_f1":     float(metrics.get("f1", 0.0)),
            "best_threshold":  float(metrics.get("threshold", self.cfg.DETECTION_THRESHOLD)),
            "val_metrics":     metrics,
        }, path)
        print(f"💾 Saved {path.name} "
              f"(ep={epoch}, f1={metrics.get('f1',0):.4f}, thr={metrics.get('threshold',0):.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# Localization trainer
# ══════════════════════════════════════════════════════════════════════════════

class LocalizationTrainer:
    """Manages the localization model training loop."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg   = cfg or config
        self.dev   = torch.device(self.cfg.DEVICE)
        self.model = make_localization_model(self.cfg).to(self.dev)

    def run(
        self,
        data_root:               Path,
        epochs:                  Optional[int] = None,
        use_synthetic_fallback:  bool = True,
        resume:                  bool = True,
    ):
        from .datasets import LocalizationDataset, SyntheticLocDataset
        epochs = epochs or self.cfg.NUM_EPOCHS
        _set_seed(self.cfg.SEED)
        bs = self.cfg.BATCH_SIZE
        if getattr(self.cfg, "USE_LITE_LOC", False):
            bs = max(8, bs // 2)
        try:
            tr_real  = LocalizationDataset(data_root, "train", augment=True,  cfg=self.cfg)
            va       = LocalizationDataset(data_root, "val",   augment=False, cfg=self.cfg)
            te       = LocalizationDataset(data_root, "test",  augment=False, cfg=self.cfg)
            tr_synth = SyntheticLocDataset(self.cfg, n_samples=100, augment=True)
            tr       = ConcatDataset([tr_real, tr_synth])
            print(f"   📊 Train: {len(tr_real)} real + {len(tr_synth)} synthetic = {len(tr)} total")
        except RuntimeError as e:
            if not use_synthetic_fallback:
                print(f"❌ {e}"); return
            print(f"⚠️  Real data unavailable ({e}) — synthetic only.")
            tr = SyntheticLocDataset(self.cfg, n_samples=500, augment=True)
            va = SyntheticLocDataset(self.cfg, n_samples=50,  augment=False)
            te = SyntheticLocDataset(self.cfg, n_samples=50,  augment=False)

        tr_l = DataLoader(tr, batch_size=bs, shuffle=True,  drop_last=True,  num_workers=0)
        va_l = DataLoader(va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)
        te_l = DataLoader(te, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)

        opt    = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=4, factor=0.5)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda")
        logger = TrainingLogger(
            self.cfg.DRIVE_LOGS / "localization_log.csv",
            columns=["epoch", "tr_loss", "val_loss", "mae_az", "mae_dist", "mae_ht"],
        )
        logged_epochs = {int(r["epoch"]) for r in logger.rows}
        start_epoch   = 1
        best_val      = 1e9
        ckpt_path     = self.cfg.DRIVE_MODELS / "best_localization.pth"
        latest_path   = self.cfg.DRIVE_MODELS / "latest_localization.pth"

        if resume and latest_path.exists():
            ck = torch.load(latest_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            if ckpt_path.exists():
                best_val = torch.load(ckpt_path, map_location=self.dev).get("best_val_loss", 1e9)
            print(f"▶️  Resuming localization from epoch {start_epoch} (best: {best_val:.5f})")
        elif resume and ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch", 1) + 1
            best_val    = ck.get("best_val_loss", 1e9)

        for ep in range(start_epoch, start_epoch + epochs):
            self.model.train()
            loss_sum = n_items = 0
            for mel, ipd, lbl in tqdm(tr_l, desc=f"Loc train ep {ep}", leave=False):
                mel = mel.to(self.dev); ipd = ipd.to(self.dev); lbl = lbl.to(self.dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE == "cuda"):
                    pred = self.model(mel, ipd)
                    loss = localization_loss(pred, lbl)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                scaler.step(opt); scaler.update()
                loss_sum += loss.item() * mel.size(0)
                n_items  += mel.size(0)

            tr_loss = loss_sum / max(n_items, 1)
            val_loss, mae_az, mae_dist, mae_ht = self._eval(va_l)
            sched.step(val_loss)
            current_lr = opt.param_groups[0]["lr"]
            print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"az={mae_az:.2f}°  dist={mae_dist:.3f}m  ht={mae_ht:.3f}m  lr={current_lr:.2e}")
            self._save(latest_path, ep, val_loss)
            if ep not in logged_epochs:
                logger.log(epoch=ep, tr_loss=round(tr_loss,6), val_loss=round(val_loss,6),
                           mae_az=round(mae_az,4), mae_dist=round(mae_dist,4), mae_ht=round(mae_ht,4))
                logged_epochs.add(ep)
            if val_loss < best_val:
                best_val = val_loss
                self._save(ckpt_path, ep, val_loss)
                print("   ✨ New best!")

        print("\n🎯 Final localization test:")
        self._eval(te_l, verbose=True)

    def _eval(self, loader: DataLoader, verbose: bool = False) -> Tuple[float, float, float, float]:
        self.model.eval()
        loss_sum = n_items = 0
        az_errs = []; dist_errs = []; ht_errs = []
        with torch.no_grad():
            for mel, ipd, lbl in tqdm(loader, desc="Loc eval", leave=False):
                mel = mel.to(self.dev); ipd = ipd.to(self.dev); lbl = lbl.to(self.dev)
                pred      = self.model(mel, ipd)
                loss_sum += localization_loss(pred, lbl).item() * mel.size(0)
                n_items  += mel.size(0)
                p = pred.cpu().numpy(); t = lbl.cpu().numpy()
                p_az = np.degrees(np.arctan2(p[:, 0], p[:, 1]))
                t_az = np.degrees(np.arctan2(t[:, 0], t[:, 1]))
                az_errs.extend(angular_error_deg(p_az, t_az).tolist())
                dist_errs.extend(np.abs(p[:, 2] - t[:, 2]).tolist())
                ht_errs.extend(np.abs(p[:, 3] - t[:, 3]).tolist())
        val_loss = loss_sum / max(n_items, 1)
        mae_az   = float(np.mean(az_errs))
        mae_dist = float(np.mean(dist_errs)) * self.cfg.MAX_LOCALIZATION_DIST
        mae_ht   = float(np.mean(ht_errs))   * self.cfg.MAX_LOCALIZATION_DIST
        if verbose:
            print(f"  Val loss={val_loss:.5f}  MAE az={mae_az:.2f}°  "
                  f"dist={mae_dist:.3f}m  ht={mae_ht:.3f}m")
        return val_loss, mae_az, mae_dist, mae_ht

    def _save(self, path: Path, epoch: int, metric: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(),
                    "epoch": epoch, "best_val_loss": metric}, path)
        print(f"💾 Saved {path.name} (ep={epoch}, val_loss={metric:.5f})")


# ══════════════════════════════════════════════════════════════════════════════
# High-level training pipeline  (v2 — position-grouped + early stopping)
# ══════════════════════════════════════════════════════════════════════════════

def train_localization_v2(
    cfg:                  Optional[Config] = None,
    epochs:               Optional[int]    = None,
    resume:               bool  = True,
    reset_best:           bool  = False,
    n_synthetic_train:    int   = 2000,
    n_synthetic_val:      int   = 200,
    grid_fraction:        float = 0.55,
    az_jitter_deg:        float = 18.0,
    dist_jitter_m:        float = 3.0,
    ht_jitter_m:          float = 2.5,
    force_redownload:     bool  = False,
    save_manifest:        bool  = True,
    early_stop_patience:  int   = 15,
    warmup_epochs:        int   = 3,
    min_lr_factor:        float = 0.05,
):
    """
    Full localization training pipeline with:
    - Position-grouped real data (loc patch v1)
    - Grid-conditioned physics-aware synthetic augmentation
    - Linear warmup + cosine LR schedule
    - Early stopping

    Parameters
    ──────────
    early_stop_patience : stop after this many epochs with no val improvement
    warmup_epochs       : LR ramp-up length
    min_lr_factor       : cosine schedule floor as fraction of base LR
    """
    from .datasets import (
        LocalizationDataset, SyntheticLocDatasetV2,
        UaVirBASEDatasetManager,
    )
    from dataclasses import dataclass, field, asdict
    import json as _json

    cfg = cfg or config
    cfg.ensure_dirs()
    _set_seed(cfg.SEED)
    total_epochs = epochs or cfg.NUM_EPOCHS

    print("=" * 70)
    print("  STAGE 2 — Localization v2  (position-grouped + physics synthesis)")
    print("=" * 70)

    proc = cfg.PROCESSED_DIR / "localization"
    um   = UaVirBASEDatasetManager(cfg)
    if force_redownload and proc.exists():
        shutil.rmtree(proc)
    try:
        um.prepare()
    except Exception as e:
        print(f"⚠️  UaVirBASE prepare failed ({e}) → synthetic-only fallback")

    synth_train = SyntheticLocDatasetV2(
        cfg, n_samples=n_synthetic_train, grid_fraction=grid_fraction,
        augment=True, az_jitter_deg=az_jitter_deg,
        dist_jitter_m=dist_jitter_m, ht_jitter_m=ht_jitter_m,
    )
    synth_val = SyntheticLocDatasetV2(
        cfg, n_samples=n_synthetic_val, grid_fraction=grid_fraction, augment=False,
    )
    try:
        real_train = LocalizationDataset(proc, "train", augment=True,  cfg=cfg)
        real_val   = LocalizationDataset(proc, "val",   augment=False, cfg=cfg)
        real_test  = LocalizationDataset(proc, "test",  augment=False, cfg=cfg)
        tr_ds = ConcatDataset([real_train, synth_train])
        va_ds = ConcatDataset([real_val,   synth_val])
        te_ds = real_test
        print(f"   Real: train={len(real_train)}  val={len(real_val)}  test={len(real_test)}")
    except RuntimeError as e:
        print(f"⚠️  Real data unavailable ({e}) — synthetic only")
        tr_ds = synth_train; va_ds = synth_val
        te_ds = SyntheticLocDatasetV2(cfg, n_samples=200, augment=False, seed=9999)

    print(f"   Train total : {len(tr_ds)}  |  Val total : {len(va_ds)}  |  Test: {len(te_ds)}")

    bs = cfg.BATCH_SIZE
    if getattr(cfg, "USE_LITE_LOC", False):
        bs = max(8, bs // 2)
    loader_kw = dict(batch_size=bs, num_workers=0, pin_memory=False)
    tr_l = DataLoader(tr_ds, shuffle=True,  drop_last=True,  **loader_kw)
    va_l = DataLoader(va_ds, shuffle=False, drop_last=False, **loader_kw)
    te_l = DataLoader(te_ds, shuffle=False, drop_last=False, **loader_kw)

    trainer = LocalizationTrainer(cfg)
    dev     = trainer.dev

    if reset_best:
        ckpt = cfg.DRIVE_MODELS / "best_localization.pth"
        if ckpt.exists():
            ck = torch.load(ckpt, map_location=cfg.DEVICE)
            ck["best_val_loss"] = 1e9
            torch.save(ck, ckpt)
            print("🔄 best_val_loss reset to 1e9")

    opt    = torch.optim.AdamW(trainer.model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP and cfg.DEVICE == "cuda")
    sched  = WarmupCosineScheduler(opt, warmup_epochs=warmup_epochs,
                                   total_epochs=total_epochs, min_lr_factor=min_lr_factor)
    logger = TrainingLogger(
        cfg.DRIVE_LOGS / "localization_log.csv",
        columns=["epoch", "tr_loss", "val_loss", "mae_az", "mae_dist", "mae_ht"],
    )
    logged_epochs = {int(r["epoch"]) for r in logger.rows}
    start_epoch   = 1
    best_val      = 1e9
    ckpt_path     = cfg.DRIVE_MODELS / "best_localization.pth"
    latest_path   = cfg.DRIVE_MODELS / "latest_localization.pth"

    if resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=dev)
        trainer.model.load_state_dict(ck["model_state"])
        start_epoch = ck.get("epoch", 1) + 1
        if ckpt_path.exists():
            best_val = torch.load(ckpt_path, map_location=dev).get("best_val_loss", 1e9)
        print(f"▶️  Resuming localization from epoch {start_epoch} (best: {best_val:.5f})")
    elif resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        trainer.model.load_state_dict(ck["model_state"])
        start_epoch = ck.get("epoch", 1) + 1
        best_val    = ck.get("best_val_loss", 1e9)

    epochs_no_improve = 0
    ep = start_epoch

    for ep in range(start_epoch, start_epoch + total_epochs):
        trainer.model.train()
        loss_sum = n_items = 0
        for mel, ipd, lbl in tqdm(tr_l, desc=f"Loc train ep {ep}", leave=False):
            mel = mel.to(dev); ipd = ipd.to(dev); lbl = lbl.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP and cfg.DEVICE == "cuda"):
                pred = trainer.model(mel, ipd)
                loss = localization_loss(pred, lbl)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * mel.size(0)
            n_items  += mel.size(0)

        tr_loss = loss_sum / max(n_items, 1)
        val_loss, mae_az, mae_dist, mae_ht = trainer._eval(va_l)
        sched.step()
        current_lr = opt.param_groups[0]["lr"]
        print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f}  val={val_loss:.5f}  "
              f"az={mae_az:.2f}°  dist={mae_dist:.3f}m  ht={mae_ht:.3f}m  lr={current_lr:.2e}")
        trainer._save(latest_path, ep, val_loss)

        if ep not in logged_epochs:
            logger.log(epoch=ep, tr_loss=round(tr_loss,6), val_loss=round(val_loss,6),
                       mae_az=round(mae_az,4), mae_dist=round(mae_dist,4), mae_ht=round(mae_ht,4))
            logged_epochs.add(ep)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            trainer._save(ckpt_path, ep, val_loss)
            print("   ✨ New best!")
        else:
            epochs_no_improve += 1
            print(f"   (no improvement for {epochs_no_improve}/{early_stop_patience} epochs)")

        if epochs_no_improve >= early_stop_patience:
            print(f"\n🛑 Early stopping at epoch {ep} "
                  f"({early_stop_patience} consecutive epochs without improvement).")
            break

    print("\n🎯 Final localization test (real-only test set):")
    trainer._eval(te_l, verbose=True)
    return {"epochs_trained": ep, "best_val_loss": best_val}


import shutil  # noqa: E402 (needed in _write_synthetic)
