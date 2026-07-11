"""
FoodNet — Loss Functions
=========================
Food-251 is more imbalanced than the spec's nominal "100-600/class" suggests
(measured ~34 to ~656, ~19:1). Three options: plain CE + label smoothing
(no per-class weighting, curbs over-confidence on 251 fine-grained classes);
weighted CE (per-class weight from data_handler.compute_class_weights —
sqrt-tempered/effective, not raw inverse frequency, so the 34-image class
doesn't get a ~20x gradient multiplier); FocalLoss (down-weights easy
examples, useful if the smallest classes stay weak — no stock PyTorch
multi-class focal loss exists, hence the custom class below).

Correct class frequency in exactly one place: if a WeightedRandomSampler is
active in the DataLoader, pass class_weights=None to build_criterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Focal Loss

class FocalLoss(nn.Module):
    """Focal Loss: FL = -alpha_t (1 - p_t)^gamma log(p_t). Down-weights easy/
    well-classified examples so the smallest food classes dominate the
    gradient. gamma=2 is the standard starting point; raise to 3 if the
    rarest classes stay weak. Pass alpha=None when a weighted sampler is active."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


#  SimCLR NT-Xent contrastive loss (for the SSL task) 

class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled cross-entropy (SimCLR). Given 2N
    L2-normalised projections (two views of N images, stacked), each sample's
    positive is its counterpart view; the other 2N-2 samples are negatives —
    the contrastive objective that pretrains the backbone without labels.
    temperature is the softmax temperature (config.SSL_TEMPERATURE, e.g. 0.5)."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                  # (2N, D)
        z = F.normalize(z, dim=1)

        sim = (z @ z.t()) / self.temperature            # (2N, 2N) cosine sims
        mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))            # mask self-similarity

        targets = torch.arange(2 * n, device=z.device)
        targets = (targets + n) % (2 * n)                # positive = partner view
        return F.cross_entropy(sim, targets)


# Factory

def build_criterion(loss_type: str, class_weights: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.1) -> nn.Module:
    """Return a configured supervised loss by name: "ce" | "weighted_ce" |
    "focal". class_weights is compute_class_weights(train_df).to(device), or
    None when a WeightedRandomSampler is active; only applied for
    weighted_ce/focal — "ce" is always unweighted (the baseline
    imbalance_grid compares weighted_ce/focal against). gamma is the focal
    focusing parameter (ignored otherwise); label_smoothing applies to the
    CE-based losses."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=None, label_smoothing=label_smoothing)
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Choose: 'ce', 'weighted_ce', 'focal'.")
