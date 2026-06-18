"""
FoodNet — Custom Model Definitions
===================================
All architectures here are built from scratch (NO pretrained weights) and
every one is verified to stay < 10 M parameters.
Run python -m codes.model to print the live parameter-budget table.

Two custom CNNs, both sharing the BaseModel interface so the rest of the
pipeline (train / evaluate / SSL / tuning) is architecture-agnostic:

  ┌──────────────────┬───────────┬──────────────────────────────────────────────┐
  │ Model            │ Params*   │ Role                                         │
  ├──────────────────┼───────────┼──────────────────────────────────────────────┤
  │ FoodNet          │ < 7.642  M   │ PROPOSED model — residual DWS + SE           │
  │ FoodNetLite      │ ~0.45 M   │ Lightweight baseline for comparison/ablation │
  └──────────────────┴───────────┴──────────────────────────────────────────────┘
  * measured at width_mult=1.0, num_classes=251, 224x224 input (see __main__).

What changed in this corrected version (and WHY it matters for accuracy)
------------------------------------------------------------------------
  1. Weight initialisation (_init_weights). The original relied on
     PyTorch defaults; a 20+-layer from-scratch net then converges slowly and
     unstably. Kaiming-fan_out on convs + zeroed BN/linear biases is the single
     biggest stability win here.
  2. Residual connections (minimal: only where in_ch == out_ch and
     stride == 1). For a deep from-scratch stack, identity shortcuts give the
     gradient a clean path and let the deep stage-3/4 blocks actually train.
     We add them ONLY on the within-stage blocks so they stay parameter-free
     (no 1x1 projection), keeping us comfortably under the 10 M cap.
  3. AMP-safe Squeeze-Excitation The SSL pipeline pretrains under autocast;
     the original SE sigmoid ran in fp16 and saturated. The gate now computes
     in fp32 and casts back, and uses the standard reduction r=16.
  4. Two-sided embedding for SSL The final block omits its trailing ReLU so
     forward_features is not clamped to the non-negative orthant — important
     for the cosine/linear classifiers in the SSL read-out.
  5. Configurable head dropout The original hardcoded Dropout(0.5) on pooled
     features over-regularised the rare classes (down to 34 images here).

Shared interface (identical contract across all models)
-------------------------------------------------------
  - forward(x)            — classification logits (B, num_classes)
  - forward_features(x)   — penultimate embedding (B, feature_dim)
                                ← REQUIRED by the SSL task.
  - feature_dim           — embedding width, for SSL heads / probes
  - freeze_backbone() / unfreeze_backbone()
  - get_trainable_params() / model_info()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


#  Shared classification head 

def make_head(in_features: int, num_classes: int = 251, dropout: float = 0.3,
              pooled_dropout: float = 0.3) -> nn.Sequential:
    """
    Lightweight regularised classification head.

        Dropout(pooled_dropout) → Linear(in_features, 512) → ReLU
        → Dropout(dropout) → Linear(512, num_classes)

    The bulk of the parameter budget stays in the convolutional backbone, where
    it does the most good. Both dropout rates are configurable; defaults are a
    gentle 0.3 rather than the original 0.5 on pooled features, because the
    rarest classes here have as few as 34 images and heavy head dropout starves
    an already data-poor signal. Raise ``pooled_dropout`` toward 0.5 only if you
    observe train/val accuracy diverging (true over-fitting).
    """
    return nn.Sequential(
        nn.Dropout(pooled_dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )


#  Abstract base class 

class BaseModel(ABC, nn.Module):
    """
    Shared interface for every FoodNet architecture.

    Subclasses MUST implement ``build()`` (construct ``self.backbone`` and
    ``self.head`` and set ``self.feature_dim``) and ``forward_features()``.
    """

    NAME: str = "base"

    def __init__(self, num_classes: int = 251, dropout: float = 0.3, width_mult: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.width_mult = width_mult           # global channel scaler (tuner can shrink the net)
        self.feature_dim: int = 0              # MUST be set inside _build()
        self.build()
        if self.feature_dim <= 0:              # fail fast if a subclass forgot the contract
            raise RuntimeError(f"{self.NAME}.build() must set self.feature_dim > 0")
        self.init_weights()

    @abstractmethod
    def build(self) -> None:
        """Construct ``self.backbone`` + ``self.head`` and set ``self.feature_dim``."""
        ...

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate embedding of shape ``(B, feature_dim)``."""
        ...

    def init_weights(self) -> None:
        """
        Proper from-scratch initialisation. Without this, a deep
        depthwise-separable net converges slowly and unstably — one of the
        largest silent accuracy losses when training from scratch.

          * Conv2d    → Kaiming-normal (fan_out, ReLU).
          * BatchNorm → weight 1, bias 0.
          * Linear    → normal(0, 0.01), bias 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification logits ``(B, num_classes)`` for the supervised task."""
        feats = self.forward_features(x)
        return self.head(feats)

    #  Transfer / fine-tuning helpers 

    def freeze_backbone(self) -> None:
        """Freeze every backbone parameter; the head stays trainable (linear-probe mode)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the whole backbone for end-to-end training."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_trainable_params(self) -> list:
        """Trainable parameters as a list (optimizer-ready, safely re-iterable)."""
        return [p for p in self.parameters() if p.requires_grad]

    def model_info(self) -> dict:
        """Parameter-count dict for the comparison tables; flags the < 10 M budget."""
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "name": self.NAME,
            "total_params_M": round(total / 1e6, 3),
            "frozen_params_M": round(frozen / 1e6, 3),
            "trainable_params_M": round((total - frozen) / 1e6, 3),
            "feature_dim": self.feature_dim,
            "under_10M": total < 10_000_000,   # Our constraint
        }


#  Building blocks 

def conv_batchnorm_activation(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    """Standard Conv → BatchNorm → ReLU block (bias-free: BN already supplies a shift)."""
    if p is None:
        p = k // 2                              # 'same' padding for odd kernels
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation channel attention (AMP-safe).

    'Squeeze' = global-average-pool each channel to a scalar; 'excite' = a tiny
    bottleneck MLP learns a per-channel gate in [0, 1] that rescales the feature
    map. The gate is computed in fp32 (autocast disabled) so the sigmoid does not
    saturate under mixed precision during SSL pretraining. ``r`` is the reduction
    ratio (larger r ⇒ cheaper, weaker); r=16 is the standard SE setting.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)       # (B,C,H,W) → (B,C,1,1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)               # squeeze
        # Compute the gate in fp32 so the sigmoid doesn't saturate under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = self.fc2(self.act(self.fc1(s.float())))
            w = torch.sigmoid(s)                      # excite: per-channel gate in [0,1]
        w = w.to(x.dtype).view(b, c, 1, 1)
        return x * w                                  # channel-wise recalibration


class DepthwiseSeparable(nn.Module):
    """
    Depthwise-separable convolution (MobileNet-style), optionally with SE and a
    residual identity shortcut.

    A 3x3 depthwise conv (one filter per channel) followed by a 1x1 pointwise
    conv (channel mixing). This costs roughly ``1/out_ch + 1/9`` of a full 3x3
    conv while keeping comparable representational power.

    Residual: when ``in_ch == out_ch`` and ``stride == 1`` we add the input back
    as a parameter-free identity shortcut. We deliberately do NOT add a 1x1
    projection shortcut on the channel-changing blocks, to keep extra parameters
    at zero and stay under the 10 M cap — the within-stage blocks (the deep
    stage-3/4 stacks) are exactly where a deep from-scratch net needs the
    gradient highway most.

    ``final_act=False`` omits the trailing ReLU; used by the last block before
    GAP so ``forward_features`` yields a two-sided embedding for the SSL read-out.
    The residual add happens BEFORE the final activation (post-activation ResNet
    ordering), so the SE recalibration is part of the residual branch.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = False,
                 final_act: bool = True) -> None:
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == 1)
        self.final_act = final_act

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        # Pointwise WITHOUT activation here; activation is applied after the
        # (optional) residual add, so the skip path is folded in pre-ReLU.
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True) if final_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.pointwise(self.depthwise(x)))
        if self.use_residual:
            out = out + x                  # parameter-free identity shortcut
        return self.act(out)


#  FoodNet  (PROPOSED — residual depthwise-separable + SE) 

class FoodNet(BaseModel):
    """
    Proposed custom CNN: a residual depthwise-separable network with SE
    attention, built from scratch and kept under the 10 M budget.

    Stage layout (channels scale with "width_mult" ; SE on every DWS block;
    identity residuals on the same-channel within-stage blocks)::

        Stem:   Conv3x3 s2 (3→32) → Conv3x3 (32→64)        224 → 112
        Stage1: DWS (64→128) + DWS-res          + pool      112 → 56
        Stage2: DWS (128→256) + DWS-res         + pool      56  → 28
        Stage3: DWS (256→512) + DWS-res ×4      + pool       28  → 14
        Stage4: DWS (512→1024) + DWS-res ×4     + pool       14  → 7
        Head:   GAP → (B, 1024) → make_head → 251 logits

    The deep stage-3/4 stacks concentrate capacity at the semantically rich,
    low-resolution end of the network, and the identity residuals there give the
    gradient a clean path so those blocks actually contribute. feature_dim=1024.
    """

    NAME = "foodnet"

    def build(self) -> None:
        w = self.width_mult

        def c(ch: int) -> int:                 # width-scaled channel count
            return max(8, int(round(ch * w)))

        stem_out = c(64)
        feat = c(1024)
        self.backbone = nn.Sequential(
            conv_batchnorm_activation(3, c(32), k=3, s=2),                         # 224 → 112
            conv_batchnorm_activation(c(32), stem_out, k=3, s=1),

            DepthwiseSeparable(stem_out, c(128), use_se=True),       # stage 1 (channel change)
            DepthwiseSeparable(c(128), c(128), use_se=True),         #         (residual)
            nn.MaxPool2d(2),                                         # 112 → 56

            DepthwiseSeparable(c(128), c(256), use_se=True),         # stage 2 (channel change)
            DepthwiseSeparable(c(256), c(256), use_se=True),         #         (residual)
            nn.MaxPool2d(2),                                         # 56 → 28

            DepthwiseSeparable(c(256), c(512), use_se=True),         # stage 3 (channel change)
            DepthwiseSeparable(c(512), c(512), use_se=True),         #         (residual ×4)
            DepthwiseSeparable(c(512), c(512), use_se=True),
            DepthwiseSeparable(c(512), c(512), use_se=True),
            DepthwiseSeparable(c(512), c(512), use_se=True),
            nn.MaxPool2d(2),                                         # 28 → 14

            DepthwiseSeparable(c(512), feat, use_se=True),           # stage 4 (channel change)
            DepthwiseSeparable(feat, feat, use_se=True),             #         (residual ×4)
            DepthwiseSeparable(feat, feat, use_se=True),
            DepthwiseSeparable(feat, feat, use_se=True),
            DepthwiseSeparable(feat, feat, use_se=True, final_act=False),  # two-sided embedding
            nn.MaxPool2d(2),                                         # 14 → 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # GAP → (B, feat, 1, 1)
        self.flatten = nn.Flatten()
        self.feature_dim = feat
        self.head = make_head(self.feature_dim, self.num_classes, self.dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                   # (B, feat, 7, 7)
        x = self.pool(x)                       # (B, feat, 1, 1)
        return self.flatten(x)                 # (B, feat)


#  Food251NetLite  (lightweight baseline, ~0.45 M) 

class FoodNetLite(BaseModel):
    """
    Lightweight baseline (~0.45 M params). Shallower, narrower depthwise-separable
    network. Trains fast — ideal as the ablation lower bound and for quick
    hyper-parameter sweeps before committing to the full Food251Net.
    """

    NAME = "foodnet_lite"

    def build(self) -> None:
        w = self.width_mult

        def c(ch: int) -> int:
            return max(8, int(round(ch * w)))

        stem_out = c(32)
        self.backbone = nn.Sequential(
            conv_batchnorm_activation(3, c(16), k=3, s=2),         # 224 → 112
            conv_batchnorm_activation(c(16), stem_out, k=3, s=1),

            DepthwiseSeparable(stem_out, c(64)),     # 112
            nn.MaxPool2d(2),                         # 112 → 56

            DepthwiseSeparable(c(64), c(128)),       # 56
            nn.MaxPool2d(2),                         # 56 → 28

            DepthwiseSeparable(c(128), c(256)),      # 28 (channel change)
            DepthwiseSeparable(c(256), c(256)),      #    (residual)
            nn.MaxPool2d(2),                         # 28 → 14

            DepthwiseSeparable(c(256), c(256), final_act=False),  # 14, two-sided (residual)
            nn.MaxPool2d(2),                         # 14 → 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.feature_dim = c(256)
        self.head = make_head(self.feature_dim, self.num_classes, self.dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.flatten(x)


#  Registry & factory 

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "foodnet":      FoodNet,
    "foodnet_lite": FoodNetLite,
}


def build_model(name: str, num_classes: int = 251, dropout: float = 0.3,
                width_mult: float = 1.0, freeze: bool = False) -> BaseModel:
    """
    Preferred factory. Instantiates a custom model from scratch (no pretrained
    weights) and ENFORCES the < 10 M parameter budget.

    Raises:
        ValueError if the resulting model is ≥ 10 M params (hard exam limit).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes=num_classes, dropout=dropout, width_mult=width_mult)
    if freeze:
        model.freeze_backbone()

    info = model.model_info()
    if not info["under_10M"]:
        raise ValueError(
            f"{name} (width_mult={width_mult}) has {info['total_params_M']} M params (≥ 10 M). "
            "Lower width_mult or pick a smaller architecture to satisfy the < 10 M constraint."
        )
    return model


def create_model(num_classes: int = 251, pretrained: bool = False, model_name: str = "food251net") -> BaseModel:
    """
    Legacy alias kept for compatibility with the demo / older scripts.

    ``pretrained`` is ignored on purpose — the exam forbids pretrained weights.
    """
    if pretrained:
        import warnings
        warnings.warn(
            "pretrained weights are NOT used in this project (forbidden by the spec); "
            "ignoring pretrained=True.",
            UserWarning,
            stacklevel=2,
        )
    return build_model(name=model_name, num_classes=num_classes)


#  Self-check / parameter-budget report 

if __name__ == "__main__":
    print(f"{'model':<18}{'total(M)':>10}{'feat_dim':>10}{'<10M?':>8}")
    print("-" * 46)
    x = torch.randn(2, 3, 224, 224)
    for key in MODEL_REGISTRY:
        m = build_model(key)
        info = m.model_info()
        logits = m(x)
        feats = m.forward_features(x)
        assert logits.shape == (2, 251), logits.shape
        assert feats.shape[1] == info["feature_dim"], feats.shape
        print(f"{info['name']:<18}{info['total_params_M']:>10}{info['feature_dim']:>10}{str(info['under_10M']):>8}")
