"""
FoodNet — Loss Functions
=========================
The Food-251 training set is more imbalanced than the spec's nominal
"100–600 / class" suggests: the measured per-class counts run from ~34 (class
162) to ~656, i.e. roughly **19:1** max-to-min. Three options:

  1. Plain CE + label smoothing — ``nn.CrossEntropyLoss(label_smoothing=...)``.
     With 251 fine-grained classes, smoothing curbs over-confidence and
     improves generalisation. No per-class weighting.

  2. Weighted CE — the same ``nn.CrossEntropyLoss``, but with per-class
     ``weight`` (use the sqrt-tempered or effective-number weights, NOT raw
     inverse frequency, so the 34-image class does not get a ~20x gradient
     multiplier — see data_handler.compute_class_weights).

  3. ``FocalLoss`` — down-weights easy examples; useful if the smallest classes
     stay weak after a few epochs. With a ~19:1 tail this is a strong choice.
     No stock PyTorch loss implements multi-class focal loss, hence the
     custom class below.

⚠️  Choose ONE place to correct class frequency: if a ``WeightedRandomSampler``
    is active in the DataLoader, pass ``class_weights=None`` to build_criterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Focal Loss

class FocalLoss(nn.Module):
    """
    Focal Loss :  ``FL = -alpha_t (1 - p_t)^gamma log(p_t)``.

    Down-weights easy/well-classified examples so the smallest food classes
    dominate the gradient. ``gamma=2`` is the standard starting point; raise to
    3 if the rarest classes stay weak. Pass ``alpha=None`` when a weighted
    sampler is already active.
    """

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
    """
    Normalised Temperature-scaled cross-entropy (SimCLR).

    Given ``2N`` L2-normalised projections (two views of N images, stacked), each
    sample's positive is its counterpart view; all other ``2N-2`` samples are
    negatives. This is the contrastive objective that pretrains the custom
    backbone WITHOUT labels.

    Args:
        temperature : softmax temperature (config.SSL_TEMPERATURE, e.g. 0.5).
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: (N, D) projection-head outputs for the two views.
        n = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                  # (2N, D)
        z = F.normalize(z, dim=1)

        sim = (z @ z.t()) / self.temperature            # (2N, 2N) cosine sims
        # Mask self-similarity on the diagonal.
        mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive index for row i is its partner view.
        targets = torch.arange(2 * n, device=z.device)
        targets = (targets + n) % (2 * n)
        return F.cross_entropy(sim, targets)


#  Factory 

def build_criterion(loss_type: str, class_weights: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.1) -> nn.Module:
    """
    Return a configured supervised loss by name.

    Args:
        loss_type       : 'ce' | 'weighted_ce' | 'focal'.
        class_weights   : compute_class_weights(train_df).to(device), or None
                          when a WeightedRandomSampler is active. Only applied
                          for 'weighted_ce'/'focal' -- 'ce' is always
                          unweighted (that is the axis imbalance_grid compares
                          against weighted_ce/focal).
        gamma           : focal focusing parameter (ignored otherwise).
        label_smoothing : smoothing eps for the CE-based losses.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=None, label_smoothing=label_smoothing)
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Choose: 'ce', 'weighted_ce', 'focal'.")
