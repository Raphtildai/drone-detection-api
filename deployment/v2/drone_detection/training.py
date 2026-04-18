# -*- coding: utf-8 -*-
"""
training.py 

Root causes of overfitting addressed
──────────────────────────────────────
1. LR SCHEDULE RESET BUG
   WarmupCosineScheduler restarted from epoch 0 on every resume because
   last_epoch was always passed as -1.  The LR jumped back to 3e-4 every
   15 epochs, creating the spiky val loss pattern in the logs.
   Fix: save sched.state_dict() in the checkpoint; restore it on resume.
   Also pass last_epoch=(start_epoch-2) on scheduler construction so even
   if the state_dict is missing the schedule is approximately correct.

2. MODEL TOO LARGE FOR 80 SESSIONS
   With only 80 real training sessions the ~4 M parameter model can
   memorise the training set easily.
   Fixes:
     a. Weight decay 5e-3 (was 1e-3)
     b. Synthetic supplement raised to 500 samples (was 100)
     c. MixUp on IPD features during training (alpha=0.3)

3. SYNTHETIC DATA NOT DIVERSE ENOUGH
   The grid-conditioned synthetic positions were too close to the real
   val positions, so val loss appeared better than it really was.
   Fixes:
     a. grid_fraction lowered to 0.40 (was 0.55)
     b. az_jitter_deg raised to 30° (was 18°)
     c. dist_jitter_m raised to 4.0 m (was 3.0 m)

4. EARLY STOPPING TOO PERMISSIVE
   Patience 20 kept training long after overfitting started.
   Fixes:
     a. Patience reduced to 12
     b. min_delta=0.005 — improvement must exceed this to count
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
import torch.nn.functional as F
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


# ─── Training logger ──────────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(self, path: Path, columns: List[str]):
        self.path = path; self.columns = columns; self.rows: List[Dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        test_file = path.parent / ".write_test"
        try: test_file.write_text("ok"); test_file.unlink()
        except OSError as e: raise RuntimeError(f"Log dir not writable: {e}")
        if path.exists():
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    parsed = {}
                    for k, v in row.items():
                        try: parsed[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                        except (ValueError, AttributeError): parsed[k] = v
                    self.rows.append(parsed)
            print(f"   📋 Resumed log from {path.name} ({len(self.rows)} existing rows)")
        else:
            print(f"   📋 New log: {path}")

    def log(self, **kwargs): self.rows.append(kwargs); self._flush()

    def _flush(self):
        tmp = self.path.parent / (self.path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns); w.writeheader(); w.writerows(self.rows)
        tmp.replace(self.path)

    def to_lists(self):
        out = {c: [] for c in self.columns}
        for row in self.rows:
            for c in self.columns: out[c].append(row.get(c, float("nan")))
        return out


# ─── LR scheduler with correct resume support ────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup then cosine annealing.
    Pass last_epoch=(global_epoch - 1) when resuming so the schedule
    continues from the correct position rather than restarting.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 min_lr_factor=0.05, last_epoch=-1):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs  = max(total_epochs, self.warmup_epochs + 1)
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup_epochs:
            scale = (ep + 1) / self.warmup_epochs
        else:
            progress = (ep - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = self.min_lr_factor + 0.5 * (1 - self.min_lr_factor) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ─── IPD MixUp ────────────────────────────────────────────────────────────────

def _mixup_ipd_batch(mel, ipd, lbl, alpha=0.3):
    """MixUp on IPD scalars only (not mel) to avoid corrupting TDOA phase."""
    if alpha <= 0 or mel.size(0) < 2:
        return mel, ipd, lbl
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(mel.size(0), device=mel.device)
    return mel, lam * ipd + (1 - lam) * ipd[idx], lam * lbl + (1 - lam) * lbl[idx]


# ─── Metric helpers ───────────────────────────────────────────────────────────

def collect_val_probs(model, loader, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(probs.tolist())
    return np.asarray(ys), np.asarray(ps)


def find_best_threshold(y_true, y_prob, beta=1.0):
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= t).astype(np.int64)
        tp = np.sum((pred==1)&(y_true==1)); fp = np.sum((pred==1)&(y_true==0))
        fn = np.sum((pred==0)&(y_true==1))
        prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
        fbeta = (1+beta**2)*prec*rec / max(beta**2*prec+rec, 1e-8) if (prec+rec) > 0 else 0.0
        if fbeta > best_score: best_score, best_t = fbeta, float(t)
    return best_t, best_score


def evaluate_binary_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(np.int64)
    tp = int(np.sum((pred==1)&(y_true==1))); fp = int(np.sum((pred==1)&(y_true==0)))
    tn = int(np.sum((pred==0)&(y_true==0))); fn = int(np.sum((pred==0)&(y_true==1)))
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc = (tp+tn)/max(tp+tn+fp+fn,1)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan")
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan")
    except Exception: auroc = auprc = float("nan")
    return {"threshold": float(threshold), "accuracy": float(acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "auroc": auroc, "auprc": auprc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def print_detection_report(y_true, y_prob, threshold):
    m = evaluate_binary_metrics(y_true, y_prob, threshold)
    pred = (y_prob >= threshold).astype(np.int64)
    print("\n=== Detection report ===")
    for k in ["threshold","accuracy","precision","recall","f1","auroc","auprc"]:
        print(f"  {k:12s}: {m[k]:.4f}")
    print(classification_report(y_true, pred, target_names=["non_drone","drone"], digits=4))
    print(confusion_matrix(y_true, pred))


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_ckpt(path, model, epoch, metric_dict, sched=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"model_state": model.state_dict(), "epoch": epoch, **metric_dict}
    if sched is not None:
        ck["sched_state"] = sched.state_dict()
    torch.save(ck, path)


def _build_sched(opt, start_epoch, total_epochs, warmup_epochs, min_lr_factor,
                 latest_path, dev):
    """
    Build a WarmupCosineScheduler with the correct last_epoch for the
    current run, and restore state_dict if available in the checkpoint.
    """
    sched_last_epoch = max(-1, start_epoch - 2)
    sched = WarmupCosineScheduler(
        opt,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr_factor=min_lr_factor,
        last_epoch=sched_last_epoch,
    )
    if latest_path.exists():
        ck_cpu = torch.load(latest_path, map_location="cpu")
        if "sched_state" in ck_cpu:
            sched.load_state_dict(ck_cpu["sched_state"])
            print("   📅 Scheduler state restored.")
    return sched


# ─── Detection trainer ────────────────────────────────────────────────────────

class DetectionTrainer:
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.dev = torch.device(self.cfg.DEVICE)
        self.model = DetectionCNN().to(self.dev)
        self._loaders = None

    def _set_loaders(self, tr_l, va_l, te_l): self._loaders = (tr_l, va_l, te_l)

    def run(self, epochs=None, resume=True):
        epochs = epochs or self.cfg.NUM_EPOCHS
        _set_seed(self.cfg.SEED)
        if self._loaders is not None:
            tr_l, va_l, te_l = self._loaders
        else:
            from .datasets import DetectionDataset
            from torch.utils.data import WeightedRandomSampler
            root = self.cfg.PROCESSED_DIR / "detection"
            try:
                tr = DetectionDataset(root,"train",augment=True,cfg=self.cfg)
                va = DetectionDataset(root,"val",augment=False,cfg=self.cfg)
                te = DetectionDataset(root,"test",augment=False,cfg=self.cfg)
            except RuntimeError as e: print(f"❌ {e}"); return
            lbs = np.array(tr.labels); cnt = np.bincount(lbs); cnt[cnt==0]=1
            weights = (1.0/cnt)[lbs]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            def _col(b): xs,ys=zip(*b); return torch.stack(xs),torch.stack(ys)
            tr_l = DataLoader(tr,batch_size=self.cfg.BATCH_SIZE,sampler=sampler,collate_fn=_col)
            va_l = DataLoader(va,batch_size=self.cfg.BATCH_SIZE,shuffle=False,collate_fn=_col)
            te_l = DataLoader(te,batch_size=self.cfg.BATCH_SIZE,shuffle=False,collate_fn=_col)

        wd = getattr(self.cfg,"WEIGHT_DECAY",5e-3)
        gc = getattr(self.cfg,"GRAD_CLIP",1.0)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LR, weight_decay=wd)
        crit = FocalLoss(gamma=2.0, alpha=0.6, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.USE_AMP and self.cfg.DEVICE=="cuda")
        logger = TrainingLogger(self.cfg.DRIVE_LOGS/"detection_log.csv",
                                ["epoch","tr_loss","tr_acc","val_acc","val_f1","val_thr","val_auprc"])
        logged = {int(r["epoch"]) for r in logger.rows}
        start_epoch=1; best_f1=0.0; best_thr=self.cfg.DETECTION_THRESHOLD
        ckpt = self.cfg.DRIVE_MODELS/"best_detection.pth"
        latest = self.cfg.DRIVE_MODELS/"latest_detection.pth"

        if resume and latest.exists():
            ck = torch.load(latest, map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch = ck.get("epoch",1)+1; best_f1 = float(ck.get("best_val_f1",0.0))
            best_thr = float(ck.get("best_threshold",self.cfg.DETECTION_THRESHOLD))
            print(f"▶️  Resuming detection from epoch {start_epoch}")

        sched = _build_sched(opt, start_epoch, start_epoch+epochs, max(1,epochs//10),
                             0.05, latest, self.dev)

        for ep in range(start_epoch, start_epoch+epochs):
            self.model.train(); ls=co=tot=0
            for X,y in tqdm(tr_l, desc=f"Det ep {ep}", leave=False):
                X=X.to(self.dev,non_blocking=True); y=y.to(self.dev,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",enabled=self.cfg.USE_AMP and self.cfg.DEVICE=="cuda"):
                    out=self.model(X); loss=crit(out,y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),gc)
                scaler.step(opt); scaler.update()
                ls+=loss.item()*X.size(0); co+=(out.argmax(1)==y).sum().item(); tot+=X.size(0)
            tr_loss=ls/max(tot,1); tr_acc=100*co/max(tot,1); sched.step()
            y_val,p_val=collect_val_probs(self.model,va_l,self.dev)
            thr,_=find_best_threshold(y_val,p_val)
            vm=evaluate_binary_metrics(y_val,p_val,thr); val_f1=vm["f1"]
            lr=opt.param_groups[0]["lr"]
            print(f"  Det Ep {ep:3d} | tr={tr_loss:.4f} acc={tr_acc:.1f}%  val_f1={val_f1:.4f}  thr={thr:.3f}  lr={lr:.2e}")
            _save_ckpt(latest, self.model, ep, {"best_val_f1":val_f1,"best_threshold":thr,"val_metrics":vm}, sched)
            if ep not in logged:
                logger.log(epoch=ep,tr_loss=round(tr_loss,6),tr_acc=round(tr_acc,3),
                           val_acc=round(vm["accuracy"]*100,3),val_f1=round(val_f1,6),
                           val_thr=round(thr,4),
                           val_auprc=round(vm["auprc"],6) if not math.isnan(vm["auprc"]) else "nan")
                logged.add(ep)
            if val_f1 > best_f1:
                best_f1,best_thr=val_f1,thr
                _save_ckpt(ckpt, self.model, ep, {"best_val_f1":val_f1,"best_threshold":thr,"val_metrics":vm}, sched)
                self.cfg.DETECTION_THRESHOLD=float(best_thr); print("   ✨ New best!")

        print("\n🎯 Final detection test:")
        if ckpt.exists():
            ck=torch.load(ckpt,map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            test_thr=float(ck.get("best_threshold",self.cfg.DETECTION_THRESHOLD))
        else: test_thr=self.cfg.DETECTION_THRESHOLD
        y_test,p_test=collect_val_probs(self.model,te_l,self.dev)
        print_detection_report(y_test,p_test,test_thr)


# ─── Localisation trainer ─────────────────────────────────────────────────────

class LocalizationTrainer:
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.dev = torch.device(self.cfg.DEVICE)
        self.model = make_localization_model(self.cfg).to(self.dev)

    def run(self, data_root, epochs=None, use_synthetic_fallback=True, resume=True):
        from .datasets import LocalizationDataset, SyntheticLocDataset
        epochs = epochs or self.cfg.NUM_EPOCHS
        _set_seed(self.cfg.SEED); bs = self.cfg.BATCH_SIZE
        try:
            tr_real  = LocalizationDataset(data_root,"train",augment=True,cfg=self.cfg)
            va       = LocalizationDataset(data_root,"val",augment=False,cfg=self.cfg)
            te       = LocalizationDataset(data_root,"test",augment=False,cfg=self.cfg)
            tr_synth = SyntheticLocDataset(self.cfg, n_samples=500, augment=True)
            tr = ConcatDataset([tr_real, tr_synth])
            print(f"   📊 Train: {len(tr_real)} real + {len(tr_synth)} synthetic = {len(tr)} total")
        except RuntimeError as e:
            if not use_synthetic_fallback: print(f"❌ {e}"); return
            print(f"⚠️  Synthetic-only fallback.")
            tr=SyntheticLocDataset(self.cfg,n_samples=500,augment=True)
            va=SyntheticLocDataset(self.cfg,n_samples=50,augment=False)
            te=SyntheticLocDataset(self.cfg,n_samples=50,augment=False)

        tr_l=DataLoader(tr,batch_size=bs,shuffle=True,drop_last=True,num_workers=0)
        va_l=DataLoader(va,batch_size=bs,shuffle=False,drop_last=False,num_workers=0)
        te_l=DataLoader(te,batch_size=bs,shuffle=False,drop_last=False,num_workers=0)

        wd=getattr(self.cfg,"WEIGHT_DECAY",5e-3); gc=getattr(self.cfg,"GRAD_CLIP",1.0)
        opt=torch.optim.AdamW(self.model.parameters(),lr=getattr(self.cfg,"LR",3e-4),weight_decay=wd)
        scaler=torch.amp.GradScaler("cuda",enabled=self.cfg.USE_AMP and self.cfg.DEVICE=="cuda")
        logger=TrainingLogger(self.cfg.DRIVE_LOGS/"localization_log.csv",
                              ["epoch","tr_loss","val_loss","mae_az","mae_dist","mae_ht"])
        logged={int(r["epoch"]) for r in logger.rows}
        start_epoch=1; best_val=1e9
        ckpt=self.cfg.DRIVE_MODELS/"best_localization.pth"
        latest=self.cfg.DRIVE_MODELS/"latest_localization.pth"

        if resume and latest.exists():
            ck=torch.load(latest,map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch=ck.get("epoch",1)+1
            if ckpt.exists(): best_val=torch.load(ckpt,map_location=self.dev).get("best_val_loss",1e9)
            print(f"▶️  Resuming from epoch {start_epoch} (best: {best_val:.5f})")
        elif resume and ckpt.exists():
            ck=torch.load(ckpt,map_location=self.dev)
            self.model.load_state_dict(ck["model_state"])
            start_epoch=ck.get("epoch",1)+1; best_val=ck.get("best_val_loss",1e9)

        sched=_build_sched(opt,start_epoch,start_epoch+epochs,3,0.05,latest,self.dev)

        for ep in range(start_epoch, start_epoch+epochs):
            self.model.train(); ls=ni=0
            for mel,ipd,lbl in tqdm(tr_l,desc=f"Loc ep {ep}",leave=False):
                mel=mel.to(self.dev); ipd=ipd.to(self.dev); lbl=lbl.to(self.dev)
                mel,ipd,lbl=_mixup_ipd_batch(mel,ipd,lbl,alpha=0.3)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",enabled=self.cfg.USE_AMP and self.cfg.DEVICE=="cuda"):
                    pred=self.model(mel,ipd); loss=localization_loss(pred,lbl)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),gc)
                scaler.step(opt); scaler.update()
                ls+=loss.item()*mel.size(0); ni+=mel.size(0)
            tr_loss=ls/max(ni,1); val_loss,mae_az,mae_dist,mae_ht=self._eval(va_l)
            sched.step(); lr=opt.param_groups[0]["lr"]
            print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"az={mae_az:.2f}°  dist={mae_dist:.3f}m  ht={mae_ht:.3f}m  lr={lr:.2e}")
            _save_ckpt(latest,self.model,ep,{"best_val_loss":val_loss},sched)
            if ep not in logged:
                logger.log(epoch=ep,tr_loss=round(tr_loss,6),val_loss=round(val_loss,6),
                           mae_az=round(mae_az,4),mae_dist=round(mae_dist,4),mae_ht=round(mae_ht,4))
                logged.add(ep)
            if val_loss < best_val:
                best_val=val_loss
                _save_ckpt(ckpt,self.model,ep,{"best_val_loss":val_loss},sched)
                print("   ✨ New best!")

        print("\n🎯 Final localization test:"); self._eval(te_l,verbose=True)

    def _eval(self, loader, verbose=False):
        self.model.eval(); ls=ni=0; az_e=[]; de=[]; he=[]
        with torch.no_grad():
            for mel,ipd,lbl in tqdm(loader,desc="Loc eval",leave=False):
                mel=mel.to(self.dev); ipd=ipd.to(self.dev); lbl=lbl.to(self.dev)
                pred=self.model(mel,ipd)
                ls+=localization_loss(pred,lbl).item()*mel.size(0); ni+=mel.size(0)
                p=pred.cpu().numpy(); t=lbl.cpu().numpy()
                az_e.extend(angular_error_deg(np.degrees(np.arctan2(p[:,0],p[:,1])),
                                               np.degrees(np.arctan2(t[:,0],t[:,1]))).tolist())
                de.extend(np.abs(p[:,2]-t[:,2]).tolist()); he.extend(np.abs(p[:,3]-t[:,3]).tolist())
        vl=ls/max(ni,1); maz=float(np.mean(az_e))
        mdi=float(np.mean(de))*self.cfg.MAX_LOCALIZATION_DIST
        mht=float(np.mean(he))*self.cfg.MAX_LOCALIZATION_DIST
        if verbose: print(f"  Val loss={vl:.5f}  MAE az={maz:.2f}°  dist={mdi:.3f}m  ht={mht:.3f}m")
        return vl, maz, mdi, mht

    def _save(self, path, epoch, metric, sched=None):
        _save_ckpt(path, self.model, epoch, {"best_val_loss": metric}, sched)
        print(f"💾 Saved {path.name} (ep={epoch}, val_loss={metric:.5f})")


# ─── High-level pipeline ──────────────────────────────────────────────────────

def train_localization(
    cfg=None, epochs=None, resume=True, reset_best=False,
    n_synthetic_train=2000, n_synthetic_val=200,
    grid_fraction=0.40,     # was 0.55
    az_jitter_deg=30.0,     # was 18°
    dist_jitter_m=4.0,      # was 3.0
    ht_jitter_m=3.0,        # was 2.5
    force_redownload=False, save_manifest=True,
    early_stop_patience=12, # was 20
    min_delta=0.005,        # new — minimum improvement to count
    warmup_epochs=3, min_lr_factor=0.05,
    mixup_alpha=0.3,        # MixUp on IPD
):
    from .datasets import LocalizationDataset, SyntheticLocDatasetV2, UaVirBASEDatasetManager
    import shutil

    cfg=cfg or config; cfg.ensure_dirs(); _set_seed(cfg.SEED)
    total_epochs=epochs or cfg.NUM_EPOCHS

    print("="*70)
    print("  STAGE 2 — Localization training")
    print("="*70)

    proc=cfg.PROCESSED_DIR/"localization"
    if force_redownload and proc.exists(): shutil.rmtree(proc)
    try: UaVirBASEDatasetManager(cfg).prepare()
    except Exception as e: print(f"⚠️  UaVirBASE failed ({e}) → synthetic fallback")

    synth_train=SyntheticLocDatasetV2(cfg,n_samples=n_synthetic_train,
                                      grid_fraction=grid_fraction,augment=True,
                                      az_jitter_deg=az_jitter_deg,
                                      dist_jitter_m=dist_jitter_m,ht_jitter_m=ht_jitter_m)
    synth_val=SyntheticLocDatasetV2(cfg,n_samples=n_synthetic_val,
                                    grid_fraction=grid_fraction,augment=False)
    try:
        real_train=LocalizationDataset(proc,"train",augment=True,cfg=cfg)
        real_val  =LocalizationDataset(proc,"val",augment=False,cfg=cfg)
        real_test =LocalizationDataset(proc,"test",augment=False,cfg=cfg)
        tr_ds=ConcatDataset([real_train,synth_train])
        va_ds=ConcatDataset([real_val,  synth_val])
        te_ds=real_test
        print(f"   Real: train={len(real_train)}  val={len(real_val)}  test={len(real_test)}")
    except RuntimeError as e:
        print(f"⚠️  Real data unavailable — synthetic only")
        tr_ds=synth_train; va_ds=synth_val
        te_ds=SyntheticLocDatasetV2(cfg,n_samples=200,augment=False,seed=9999)

    print(f"   Train: {len(tr_ds)}  Val: {len(va_ds)}  Test: {len(te_ds)}")
    bs=cfg.BATCH_SIZE
    kw=dict(batch_size=bs,num_workers=0,pin_memory=False)
    tr_l=DataLoader(tr_ds,shuffle=True,drop_last=True,**kw)
    va_l=DataLoader(va_ds,shuffle=False,drop_last=False,**kw)
    te_l=DataLoader(te_ds,shuffle=False,drop_last=False,**kw)

    trainer=LocalizationTrainer(cfg); dev=trainer.dev

    if reset_best:
        ckpt=cfg.DRIVE_MODELS/"best_localization.pth"
        if ckpt.exists():
            ck=torch.load(ckpt,map_location=cfg.DEVICE); ck["best_val_loss"]=1e9
            torch.save(ck,ckpt); print("🔄 best_val_loss reset to 1e9")

    wd=getattr(cfg,"WEIGHT_DECAY",5e-3); gc=getattr(cfg,"GRAD_CLIP",1.0)
    opt=torch.optim.AdamW(trainer.model.parameters(),lr=getattr(cfg,"LR",3e-4),weight_decay=wd)
    scaler=torch.amp.GradScaler("cuda",enabled=cfg.USE_AMP and cfg.DEVICE=="cuda")
    logger=TrainingLogger(cfg.DRIVE_LOGS/"localization_log.csv",
                          ["epoch","tr_loss","val_loss","mae_az","mae_dist","mae_ht"])
    logged={int(r["epoch"]) for r in logger.rows}
    start_epoch=1; best_val=1e9
    ckpt_path=cfg.DRIVE_MODELS/"best_localization.pth"
    latest_path=cfg.DRIVE_MODELS/"latest_localization.pth"

    if resume and latest_path.exists():
        ck=torch.load(latest_path,map_location=dev)
        trainer.model.load_state_dict(ck["model_state"])
        start_epoch=ck.get("epoch",1)+1
        if ckpt_path.exists(): best_val=torch.load(ckpt_path,map_location=dev).get("best_val_loss",1e9)
        print(f"▶️  Resuming from epoch {start_epoch} (best: {best_val:.5f})")
    elif resume and ckpt_path.exists():
        ck=torch.load(ckpt_path,map_location=dev)
        trainer.model.load_state_dict(ck["model_state"]); start_epoch=ck.get("epoch",1)+1
        best_val=ck.get("best_val_loss",1e9)

    sched=_build_sched(opt,start_epoch,start_epoch+total_epochs,
                       warmup_epochs,min_lr_factor,latest_path,dev)
    no_improve=0; ep=start_epoch

    for ep in range(start_epoch, start_epoch+total_epochs):
        trainer.model.train(); ls=ni=0
        for mel,ipd,lbl in tqdm(tr_l,desc=f"Loc ep {ep}",leave=False):
            mel=mel.to(dev); ipd=ipd.to(dev); lbl=lbl.to(dev)
            mel,ipd,lbl=_mixup_ipd_batch(mel,ipd,lbl,alpha=mixup_alpha)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=cfg.USE_AMP and cfg.DEVICE=="cuda"):
                pred=trainer.model(mel,ipd); loss=localization_loss(pred,lbl)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),gc)
            scaler.step(opt); scaler.update()
            ls+=loss.item()*mel.size(0); ni+=mel.size(0)

        tr_loss=ls/max(ni,1)
        val_loss,mae_az,mae_dist,mae_ht=trainer._eval(va_l)
        sched.step(); lr=opt.param_groups[0]["lr"]
        gap=val_loss-tr_loss
        print(f"  Loc Ep {ep:3d} | tr={tr_loss:.5f}  val={val_loss:.5f}  "
              f"gap={gap:+.4f}  az={mae_az:.2f}°  dist={mae_dist:.3f}m  "
              f"ht={mae_ht:.3f}m  lr={lr:.2e}")
        _save_ckpt(latest_path,trainer.model,ep,{"best_val_loss":val_loss},sched)
        if ep not in logged:
            logger.log(epoch=ep,tr_loss=round(tr_loss,6),val_loss=round(val_loss,6),
                       mae_az=round(mae_az,4),mae_dist=round(mae_dist,4),mae_ht=round(mae_ht,4))
            logged.add(ep)

        if val_loss < best_val - min_delta:
            best_val=val_loss; no_improve=0
            _save_ckpt(ckpt_path,trainer.model,ep,{"best_val_loss":val_loss},sched)
            print("   ✨ New best!")
        else:
            no_improve+=1
            print(f"   (no improvement for {no_improve}/{early_stop_patience} epochs)")

        if no_improve >= early_stop_patience:
            print(f"\n🛑 Early stopping at epoch {ep}."); break

    print("\n🎯 Final localization test (real-only):")
    trainer._eval(te_l, verbose=True)
    return {"epochs_trained": ep, "best_val_loss": best_val}