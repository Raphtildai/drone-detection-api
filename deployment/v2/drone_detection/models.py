# -*- coding: utf-8 -*-
"""
models.py
─────────
PyTorch model definitions for the drone detection & localization pipeline.

Includes:
  LocalizationCNN
    - IPD branch capacity: 3→32→32 replaced by 3(+1)→64→128→64 (deeper, wider)
    - BPF energy ratio accepted as optional 4th scalar input when
      cfg.BPF_ENERGY_RATIO_AS_FEATURE is True (IPD tensor shape becomes (4,))
    - Dropout added to IPD branch for regularisation

  localization_loss
    - Azimuth components now use cosine-distance loss:
        L_az = 1 - (pred_sin*true_sin + pred_cos*true_cos)
      This is the geometrically correct loss for circular quantities and
      avoids the MSE(sin,cos) bias that inflated azimuth MAE.
    - Focal weighting on distance/height when target > 0.5 (hard examples)

  make_localization_model — updated to always return full-capacity model
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

    Input:  (B, 3, N_MELS, T)
    Output: (B, 2)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
        )
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
        attn = self.freq_attn(feat)
        feat = (feat * attn).sum(dim=2, keepdim=True)
        feat = self.gap(feat).flatten(1)
        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════════════════
# Localization model  (stronger IPD branch + BPF ratio input)
# ══════════════════════════════════════════════════════════════════════════════

class LocalizationCNN(nn.Module):
    """
    Full-capacity multi-microphone acoustic localizer.

    Input
    ─────
    mel : (B, 3, N_MELS, T)  — stacked mel-spectrograms (one per mic)
    ipd : (B, 3) or (B, 4)   — IPD features + optional BPF energy ratio

    Output
    ──────
    (B, 4) — [sin(az), cos(az), dist/MAX_DIST, ht/MAX_DIST]

    Internal architecture changes:
    - IPD branch: Linear(in→64) → BN → ReLU → Dropout(0.2)
                  Linear(64→128) → BN → ReLU → Dropout(0.2)
                  Linear(128→64) → BN → ReLU
      vs old:     Linear(3→32) → ReLU → Linear(32→32) → ReLU
      This gives the TDOA/IPD information a much larger capacity, matching
      its importance for azimuth estimation.
    - ipd_in_dim: 3 (standard) or 4 (with BPF energy ratio as 4th scalar).
      Pass ipd_in_dim=4 when cfg.BPF_ENERGY_RATIO_AS_FEATURE is True.
    - BatchNorm on each IPD FC layer for stable training.
    - Head dropout increased from 0.3 to 0.4 to match the wider branch.
    """

    def __init__(self, n_mels: int = 64, ipd_in_dim: int = 3):
        super().__init__()
        self._ipd_in_dim = ipd_in_dim

        self.mel_enc = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64,  128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Deeper, wider IPD branch 
        self.ipd_fc = nn.Sequential(
            nn.Linear(ipd_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        fused = 256 * 4 * 4 + 64   # mel flat + IPD output
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.2),
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


def make_localization_model(cfg) -> nn.Module:
    """
    Instantiate LocalizationCNN with the correct IPD input dimension.
    Uses 4-dim IPD when cfg.BPF_ENERGY_RATIO_AS_FEATURE is True.
    """
    ipd_in_dim = 4 if getattr(cfg, "BPF_ENERGY_RATIO_AS_FEATURE", False) else 3
    print(
        f"🔧 Using LocalizationCNN (full-capacity mode, "
        f"ipd_in_dim={ipd_in_dim})"
    )
    return LocalizationCNN(n_mels=cfg.N_MELS, ipd_in_dim=ipd_in_dim)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary classification (Lin et al., 2017).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def localization_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mixed regression loss for the 4-dimensional localization output.

    Azimuth via cosine-distance loss
    ───────────────────────────────────────
    Old:  2 × MSE(sin_az) + 2 × MSE(cos_az)
    New:  2 × (1 − cos_similarity(pred_az_vec, true_az_vec))
              = 2 × (1 − (pred_s·true_s + pred_c·true_c))

    This is the correct circular loss for azimuth angles.  MSE on (sin, cos)
    is biased: an error of 90° gives MSE = 1.0 but cosine distance = 1.0 too,
    however at small angles MSE penalises proportionally to sin² while cosine
    distance penalises proportionally to the actual angular deviation — which
    is what we care about.

    The cosine dot product is bounded to [−1, 1] so we clamp before subtracting
    to avoid numerical issues with fp16 training.

    Distance / height
    ─────────────────
    SmoothL1 (Huber).
    Added: focal up-weighting for hard examples (dist or ht > 0.5 normalised),
    which correspond to drones at longer range where SNR is lower.
    """
    # ── Azimuth: cosine-distance loss  ──────────────────────────────
    dot    = (pred[:, 0] * target[:, 0] + pred[:, 1] * target[:, 1]).clamp(-1.0, 1.0)
    loss_az = (1.0 - dot).mean()

    # ── Distance and height: SmoothL1 with hard-example focal weight ─────
    loss_dist_raw = F.smooth_l1_loss(pred[:, 2], target[:, 2], reduction="none")
    loss_ht_raw   = F.smooth_l1_loss(pred[:, 3], target[:, 3], reduction="none")

    # Weight hard examples (far drones) more strongly
    # focal_w ∈ [1.0, 2.0] — scales up loss for targets > 0.5 (normalised)
    focal_w_dist = 1.0 + target[:, 2].clamp(0.0, 1.0)
    focal_w_ht   = 1.0 + target[:, 3].clamp(0.0, 1.0)
    loss_dist = (loss_dist_raw * focal_w_dist).mean()
    loss_ht   = (loss_ht_raw   * focal_w_ht  ).mean()

    # Overall: azimuth upweighted 2×, height 0.7× to reflect importance for performance
    return 2.0 * loss_az + 1.0 * loss_dist + 0.7 * loss_ht