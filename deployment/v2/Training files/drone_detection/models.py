# -*- coding: utf-8 -*-
"""
drone_detection/models.py
─────────────────────────
PyTorch model definitions:
  - DetectionCNN        (frequency-axis attention pooling)
  - LocalizationCNNLite (MobileNet-style depthwise-separable, < 4 GB VRAM)
  - LocalizationCNN     (full-capacity)
  - make_localization_model()
  - FocalLoss           (γ=2, α=0.6, label_smoothing=0.02)
  - localization_loss()
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Detection ─────────────────────────────────────────────────────────────────

class DetectionCNN(nn.Module):
    """
    CNN + frequency-axis soft-attention for domain-robust detection.

    Input  : (B, 3, N_MELS, T) — 3-channel v15 feature stack
    Output : (B, 2)             — logits [non_drone, drone]
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.freq_attn = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(64,  1,  kernel_size=(1, 1)), nn.Softmax(dim=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Dropout(0.4),
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
        attn = self.freq_attn(feat)
        feat = (feat * attn).sum(dim=2, keepdim=True)
        feat = self.gap(feat).flatten(1)
        return self.classifier(feat)


# ── Localisation ──────────────────────────────────────────────────────────────

class LocalizationCNNLite(nn.Module):
    """
    MobileNet-style (depthwise-separable) localisation model.
    ~4× fewer parameters than LocalizationCNN — for < 4 GB VRAM / Colab Free.

    Input  : mel (B, 3, N_MELS, T) + ipd (B, 3)
    Output : (B, 4) — [sin_az, cos_az, dist_norm, ht_norm]
    """

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()

        def _ds_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.mel_enc = nn.Sequential(
            _ds_block(3,   16),
            _ds_block(16,  32),
            _ds_block(32,  64),
            _ds_block(64, 128),
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

    def forward(self, mel: torch.Tensor, ipd: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.mel_enc(mel).flatten(1), self.ipd_fc(ipd)], dim=1))


class LocalizationCNN(nn.Module):
    """
    Full-capacity localisation model for high-VRAM environments.

    Input  : mel (B, 3, N_MELS, T) + ipd (B, 3)
    Output : (B, 4) — [sin_az, cos_az, dist_norm, ht_norm]
    """

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()
        self.mel_enc = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.ipd_fc = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        fused = 256 * 4 * 4 + 32
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),   nn.ReLU(),
            nn.Linear(128, 4),
        )

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, mel: torch.Tensor, ipd: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.mel_enc(mel).flatten(1), self.ipd_fc(ipd)], dim=1))


def make_localization_model(cfg) -> nn.Module:
    """
    Factory: selects Lite vs Full model based on cfg.USE_LITE_LOC or GPU VRAM.
    """
    if getattr(cfg, "USE_LITE_LOC", False):
        print("🔧 Using LocalizationCNNLite (resource-constrained mode)")
        return LocalizationCNNLite(cfg.N_MELS)
    print("🔧 Using LocalizationCNN (full-capacity mode)")
    return LocalizationCNN(cfg.N_MELS)


# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    γ=2, α=0.6, label_smoothing=0.02 are the v15 defaults.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce    = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at    = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal = focal * at
        return focal.mean()


def localization_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Combined loss for the localisation head output [sin_az, cos_az, dist, ht].
    Az error is weighted 2× to prioritise angular accuracy.
    """
    loss_sin  = F.mse_loss(pred[:, 0], target[:, 0])
    loss_cos  = F.mse_loss(pred[:, 1], target[:, 1])
    loss_dist = F.smooth_l1_loss(pred[:, 2], target[:, 2])
    loss_ht   = F.smooth_l1_loss(pred[:, 3], target[:, 3])
    return 2.0 * (loss_sin + loss_cos) + 1.0 * loss_dist + 0.7 * loss_ht