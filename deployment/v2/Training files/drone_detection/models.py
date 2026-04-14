# -*- coding: utf-8 -*-
"""
models.py
─────────
PyTorch model definitions for the drone detection & localization pipeline.

Models
──────
DetectionCNN          — frequency-attention CNN, binary drone/non-drone
LocalizationCNN       — full-capacity multi-mic CNN+IPD localizer
LocalizationCNNLite   — depthwise-separable lightweight variant (< 4 GB VRAM)

Loss functions
──────────────
FocalLoss             — focal loss with optional alpha balancing (detection)
localization_loss     — mixed MSE + SmoothL1 for (sin, cos, dist, ht)

Utilities
─────────
make_localization_model()  — auto-selects full vs lite based on cfg.USE_LITE_LOC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Detection model
# ══════════════════════════════════════════════════════════════════════════════

class DetectionCNN(nn.Module):
    """
    Binary drone / non-drone classifier.

    Architecture
    ────────────
    4× ConvBNReLU + MaxPool encoder → frequency-axis soft-attention pooling
    → GlobalAveragePool → 256-d MLP head.

    Input:  (B, 3, N_MELS, T)  — 3-channel feature stack (log-mel/PCEN/delta)
    Output: (B, 2)             — logits for [non_drone, drone]
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
        )
        # Soft attention over the frequency axis
        self.freq_attn = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(64,   1, kernel_size=(1, 1)), nn.Softmax(dim=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        attn = self.freq_attn(feat)                        # (B,1,F',T')
        feat = (feat * attn).sum(dim=2, keepdim=True)      # frequency pooling
        feat = self.gap(feat).flatten(1)                   # (B,256)
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════════════════
# Localization models
# ══════════════════════════════════════════════════════════════════════════════

class LocalizationCNN(nn.Module):
    """
    Full-capacity multi-microphone acoustic localizer.

    Input
    ─────
    mel : (B, 3, N_MELS, T)  — stacked mel-spectrograms (one per mic)
    ipd : (B, 3)             — IPD features for the 3 mic pairs

    Output
    ──────
    (B, 4) — [sin(az), cos(az), dist/MAX_DIST, ht/MAX_DIST]
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.mel_enc = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.ipd_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        fused = 256 * 4 * 4 + 32
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),  nn.ReLU(),
            nn.Linear(128, 4),
        )

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, mel: torch.Tensor, ipd: torch.Tensor) -> torch.Tensor:
        mel_feat = self.mel_enc(mel).flatten(1)
        ipd_feat = self.ipd_fc(ipd)
        return self.head(torch.cat([mel_feat, ipd_feat], dim=1))


class LocalizationCNNLite(nn.Module):
    """
    Resource-constrained localizer using depthwise-separable convolutions.
    ~4× fewer parameters than LocalizationCNN.
    Recommended for GPU VRAM < 4 GB (Colab free tier).

    Same input / output signature as LocalizationCNN.
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.mel_enc = nn.Sequential(
            self._ds_block(3,   16),
            self._ds_block(16,  32),
            self._ds_block(32,  64),
            self._ds_block(64,  128),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.ipd_fc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        fused = 128 * 2 * 2 + 16
        self.head = nn.Sequential(
            nn.Linear(fused, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    @staticmethod
    def _ds_block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cin,  3, padding=1, groups=cin, bias=False),  # depthwise
            nn.Conv2d(cin, cout, 1, bias=False),                          # pointwise
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, mel: torch.Tensor, ipd: torch.Tensor) -> torch.Tensor:
        mel_feat = self.mel_enc(mel).flatten(1)
        ipd_feat = self.ipd_fc(ipd)
        return self.head(torch.cat([mel_feat, ipd_feat], dim=1))


def make_localization_model(cfg) -> nn.Module:
    """
    Factory that selects the full or lite localization model based on
    cfg.USE_LITE_LOC (auto-set from GPU VRAM, can be overridden).
    """
    if getattr(cfg, "USE_LITE_LOC", False):
        print("🔧 Using LocalizationCNNLite (resource-constrained mode)")
        return LocalizationCNNLite(cfg.N_MELS)
    print("🔧 Using LocalizationCNN (full-capacity mode)")
    return LocalizationCNN(cfg.N_MELS)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary classification (Lin et al., 2017).

    Parameters
    ──────────
    gamma          : focusing exponent (default 2.0)
    alpha          : positive-class weight in (0,1); None = no weighting
    label_smoothing: label smoothing factor applied before focal modulation
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma          = gamma
        self.alpha          = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(
            logits, targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at    = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal = focal * at
        return focal.mean()


def localization_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mixed regression loss for the 4-dimensional localization output.

    Components
    ──────────
    2 × MSE for (sin(az), cos(az))  — upweighted for angular accuracy
    1 × SmoothL1 for normalised distance
    0.7 × SmoothL1 for normalised height
    """
    loss_sin  = F.mse_loss(pred[:, 0], target[:, 0])
    loss_cos  = F.mse_loss(pred[:, 1], target[:, 1])
    loss_dist = F.smooth_l1_loss(pred[:, 2], target[:, 2])
    loss_ht   = F.smooth_l1_loss(pred[:, 3], target[:, 3])
    return 2.0 * (loss_sin + loss_cos) + 1.0 * loss_dist + 0.7 * loss_ht
