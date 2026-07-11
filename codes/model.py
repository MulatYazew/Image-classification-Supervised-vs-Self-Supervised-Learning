"""
FoodNet — Custom Model Definitions
===================================
Two custom CNNs, built from scratch (no pretrained weights), each verified
< 10M params. Run `python -m codes.model` to print the live parameter table.
Both share the BaseModel interface so train/evaluate/SSL/tuning stay
architecture-agnostic.

  Model      Params*   Conv2d   Role
  FoodNet30  ~7.6M     30       Residual DWS + SE backbone
  FoodNet46  ~4.4M     46       PROPOSED — MBConv + SE + DropPath
  * at width_mult=1.0, num_classes=251, 224x224 input (see __main__)

Stability choices that matter for a from-scratch net: Kaiming-fan_out conv
init + zeroed BN/linear biases (init_weights); parameter-free identity
residuals on same-shape blocks; SE gate computed in fp32 so it doesn't
saturate under AMP; final block skips its trailing ReLU so forward_features
gives a two-sided embedding for the SSL/linear read-out; configurable head
dropout (default 0.3, not 0.5 — the rarest classes have only ~34 images).

Shared interface: forward(x) -> logits (B, num_classes); forward_features(x)
-> penultimate embedding (B, feature_dim), required by the SSL task;
feature_dim; freeze_backbone()/unfreeze_backbone(); get_trainable_params();
model_info().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shared classification head

def make_head(in_features: int, num_classes: int = 251, dropout: float = 0.3,
              pooled_dropout: float = 0.3) -> nn.Sequential:
    """Dropout -> Linear(in_features, 512) -> ReLU -> Dropout -> Linear(512, num_classes).
    Most of the parameter budget stays in the conv backbone. Defaults are a
    gentle 0.3 (not the original 0.5) since the rarest classes have as few as
    34 images and heavy head dropout starves an already data-poor signal —
    raise pooled_dropout toward 0.5 only if train/val accuracy diverges."""
    return nn.Sequential(
        nn.Dropout(pooled_dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )


# Abstract base class

class BaseModel(ABC, nn.Module):
    """Shared interface for every FoodNet architecture. Subclasses must
    implement build() (construct self.backbone/self.head, set
    self.feature_dim) and forward_features()."""

    NAME: str = "base"

    backbone: nn.Module                        # set inside build()
    head: nn.Module                            # set inside build()

    def __init__(self, num_classes: int = 251, dropout: float = 0.3, width_mult: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.width_mult = width_mult           # global channel scaler (tuner can shrink the net)
        self.feature_dim: int = 0
        self.build()
        if self.feature_dim <= 0:              # fail fast if a subclass forgot the contract
            raise RuntimeError(f"{self.NAME}.build() must set self.feature_dim > 0")
        self.init_weights()

    @abstractmethod
    def build(self) -> None:
        """Construct self.backbone + self.head and set self.feature_dim."""
        ...

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate embedding of shape (B, feature_dim)."""
        ...

    def init_weights(self) -> None:
        """Kaiming-normal (fan_out, ReLU) convs, BN weight=1/bias=0,
        Linear normal(0, 0.01)/bias=0 — without this a deep from-scratch net
        converges slowly and unstably."""
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
        """Classification logits (B, num_classes)."""
        feats = self.forward_features(x)
        return self.head(feats)

    # Transfer / fine-tuning helpers

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
        """Parameter-count dict for comparison tables; flags the < 10M budget."""
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "name": self.NAME,
            "total_params_M": round(total / 1e6, 3),
            "frozen_params_M": round(frozen / 1e6, 3),
            "trainable_params_M": round((total - frozen) / 1e6, 3),
            "feature_dim": self.feature_dim,
            "under_10M": total < 10_000_000,
        }


# Building blocks

def conv_batchnorm_activation(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU (bias-free: BN already supplies a shift)."""
    if p is None:
        p = k // 2                              # 'same' padding for odd kernels
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention (AMP-safe): global-average-pool
    each channel, then a bottleneck MLP learns a per-channel gate that
    rescales the feature map. The gate runs in fp32 (autocast disabled) so it
    doesn't saturate under mixed precision during SSL pretraining.

    Two call conventions: FoodNet30's DepthwiseSeparable uses a standard SE
    (reduction=16, sigmoid gate); FoodNet46's MBConvBlock passes
    reduction_channels explicitly, computed from the block's pre-expansion
    input channels rather than the expanded gate width, with a hardsigmoid
    gate (hard_gate=True).
    """

    def __init__(self, channels: int, reduction: int = 16,
                 reduction_channels: int | None = None, hard_gate: bool = False) -> None:
        super().__init__()
        hidden = reduction_channels if reduction_channels is not None else max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)           # (B,C,H,W) -> (B,C,1,1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.hard_gate = hard_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)                   # squeeze
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = self.fc2(self.act(self.fc1(s.float())))
            w = F.hardsigmoid(s) if self.hard_gate else torch.sigmoid(s)   # excite
        w = w.to(x.dtype).view(b, c, 1, 1)
        return x * w                                  # channel-wise recalibration


class DepthwiseSeparable(nn.Module):
    """Depthwise-separable conv (MobileNet-style) with optional SE and a
    residual identity shortcut. 3x3 depthwise + 1x1 pointwise costs roughly
    1/out_ch + 1/9 of a full 3x3 conv at comparable representational power.

    Residual added only when in_ch == out_ch and stride == 1 (parameter-free
    identity — no 1x1 projection on channel-changing blocks, to stay under
    the 10M cap) — exactly the within-stage blocks where a deep from-scratch
    net needs the gradient highway most.

    final_act=False omits the trailing ReLU (used on the last block before
    GAP, so forward_features yields a two-sided embedding for SSL). Residual
    add happens before the final activation (post-activation ResNet ordering).
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
        # pointwise has no activation here — applied after the residual add
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


# FoodNet building blocks (inverted-residual / MBConv style)

class DropPath(nn.Module):
    """Stochastic depth: during training, drop the entire residual branch
    with probability drop_prob for a random subset of the batch (full
    residual at test time). Better than standard Dropout for residual nets
    since it drops the structural path, not individual activations."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        noise = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)   # per-sample, broadcast over C/H/W
        noise.bernoulli_(survival)
        return x.div(survival) * noise


class MBConvBlock(nn.Module):
    """Inverted Residual Block (MBConv): Expand-1x1 -> DW-3x3 -> SE -> Project-1x1.

    Expanding channels before the depthwise conv (vs. DepthwiseSeparable's
    plain DWS) gives the depthwise filter more channels to mix at modest
    extra cost — the "inverted bottleneck" behind MobileNetV2/V3. Uses
    Hard-Swish (better than ReLU on fine-grained vision), no activation after
    the final projection (pre-activation residual), and DropPath on the
    residual branch for regularisation.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 6,
                 se_ratio: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        self.has_residual = (stride == 1 and in_ch == out_ch)
        exp_ch = max(in_ch, int(in_ch * expand))
        se_hidden = max(8, in_ch // se_ratio)

        layers: list[nn.Module] = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                nn.Hardswish(inplace=True),
            ]
        layers += [
            nn.Conv2d(exp_ch, exp_ch, 3, stride=stride, padding=1,
                      groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.Hardswish(inplace=True),
            SqueezeExcite(exp_ch, reduction_channels=se_hidden, hard_gate=True),
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.drop_path: nn.Module = (
            DropPath(drop_path)
            if drop_path > 0.0 and self.has_residual
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.has_residual:
            out = self.drop_path(out) + x
        return out


# FoodNet30 (residual depthwise-separable + SE, 30 Conv2d layers)

class FoodNet30(BaseModel):
    """30-conv residual depthwise-separable network with SE attention.

        Stem:   Conv3x3 s2 (3->32) -> Conv3x3 (32->64)      224 -> 112  [2 conv]
        Stage1: DWS (64->128) + DWS-res         + pool      112 -> 56   [4 conv]
        Stage2: DWS (128->256) + DWS-res        + pool      56  -> 28   [4 conv]
        Stage3: DWS (256->512) + DWS-res x4     + pool      28  -> 14   [10 conv]
        Stage4: DWS (512->1024) + DWS-res x4    + pool      14  -> 7    [10 conv]
        Head:   GAP -> (B, 1024) -> make_head -> 251 logits

    feature_dim=1024.
    """

    NAME = "foodnet30"

    def build(self) -> None:
        w = self.width_mult

        def c(ch: int) -> int:                 # width-scaled channel count
            return max(8, round(ch * w))

        stem_out = c(64)
        feat = c(1024)
        self.backbone = nn.Sequential(
            conv_batchnorm_activation(3, c(32), k=3, s=2),                         # 224 -> 112
            conv_batchnorm_activation(c(32), stem_out, k=3, s=1),

            DepthwiseSeparable(stem_out, c(128), use_se=True),       # stage 1 (channel change)
            DepthwiseSeparable(c(128), c(128), use_se=True),         #         (residual)
            nn.MaxPool2d(2),                                         # 112 -> 56

            DepthwiseSeparable(c(128), c(256), use_se=True),         # stage 2 (channel change)
            DepthwiseSeparable(c(256), c(256), use_se=True),         #         (residual)
            nn.MaxPool2d(2),                                         # 56 -> 28

            DepthwiseSeparable(c(256), c(512), use_se=True),         # stage 3 (channel change)
            DepthwiseSeparable(c(512), c(512), use_se=True),         #         (residual x4)
            DepthwiseSeparable(c(512), c(512), use_se=True),
            DepthwiseSeparable(c(512), c(512), use_se=True),
            DepthwiseSeparable(c(512), c(512), use_se=True),
            nn.MaxPool2d(2),                                         # 28 -> 14

            DepthwiseSeparable(c(512), feat, use_se=True),           # stage 4 (channel change)
            DepthwiseSeparable(feat, feat, use_se=True),             #         (residual x4)
            DepthwiseSeparable(feat, feat, use_se=True),
            DepthwiseSeparable(feat, feat, use_se=True),
            DepthwiseSeparable(feat, feat, use_se=True, final_act=False),  # two-sided embedding
            nn.MaxPool2d(2),                                         # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # GAP -> (B, feat, 1, 1)
        self.flatten = nn.Flatten()
        self.feature_dim = feat
        self.head = make_head(self.feature_dim, self.num_classes, self.dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                   # (B, feat, 7, 7)
        x = self.pool(x)                       # (B, feat, 1, 1)
        return self.flatten(x)                 # (B, feat)


# FoodNet46 (PROPOSED — MBConv inverted-residual + SE + DropPath, 46 Conv2d layers)

class FoodNet46(BaseModel):
    """46-conv MBConv inverted-residual network with SE attention and DropPath
    stochastic depth, tuned for Apple Silicon.

    Each block: Expand-1x1 -> DW-3x3 -> SE -> Project-1x1, Hard-Swish
    activation, DropPath rate ramping linearly 0 -> 0.2 across blocks,
    EfficientNet-style SE (se_hidden = in_ch // 4, tied to pre-expansion
    channels). Reduced from a 73-layer draft by trimming stage 3-7 block
    counts (24 -> 15 MBConv blocks) and the neck embedding (960 -> 768),
    cutting ~35% of forward-pass cost while keeping the full 7-stage hierarchy.

        Stem:    Conv3x3/s=2, 3->32, BN, H-Swish            224->112  [ 1 conv]
        Stage 1: MBConv(t=1, 32->24,   n=1, s=1, SE)            112  [ 2 conv]
        Stage 2: MBConv(t=4, 24->40,   n=2, s=2, SE)             56  [ 6 conv]
        Stage 3: MBConv(t=4, 40->80,   n=2, s=2, SE)             28  [ 6 conv]
        Stage 4: MBConv(t=4, 80->112,  n=3, s=2, SE)             14  [ 9 conv]
        Stage 5: MBConv(t=6, 112->112, n=2, s=1, SE)             14  [ 6 conv]
        Stage 6: MBConv(t=6, 112->192, n=3, s=2, SE)              7  [ 9 conv]
        Stage 7: MBConv(t=6, 192->192, n=2, s=1, SE)              7  [ 6 conv]
        Neck:    Conv1x1, ->768, BN, H-Swish                      7  [ 1 conv]
        Head:    GAP -> head -> 251 logits

    feature_dim=768, ~4.4M params at width_mult=1.0.
    """

    NAME = "foodnet46"

    # (expand_ratio, out_channels, num_blocks, stride)
    STAGES = (
        (1,  24, 1, 1),
        (4,  40, 2, 2),
        (4,  80, 2, 2),
        (4, 112, 3, 2),
        (6, 112, 2, 1),
        (6, 192, 3, 2),
        (6, 192, 2, 1),
    )
    NECK_CH       = 768
    MAX_DROP_PATH = 0.2

    def build(self) -> None:
        w = self.width_mult

        def c(ch: int) -> int:
            return max(8, round(ch * w))

        total_blocks = sum(n for _, _, n, _ in self.STAGES)
        dp_rates = [
            self.MAX_DROP_PATH * i / max(total_blocks - 1, 1)
            for i in range(total_blocks)
        ]

        layers: list[nn.Module] = [
            nn.Conv2d(3, c(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.Hardswish(inplace=True),
        ]

        in_ch = c(32)
        block_idx = 0
        for expand, out_ch_base, num_blocks, stride in self.STAGES:
            out_ch = c(out_ch_base)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(MBConvBlock(
                    in_ch, out_ch,
                    stride=s,
                    expand=expand,
                    se_ratio=4,
                    drop_path=dp_rates[block_idx],
                ))
                in_ch = out_ch
                block_idx += 1

        # neck: 1x1 conv to a larger embedding (more capacity for 251 classes)
        neck_ch = c(self.NECK_CH)
        layers += [
            nn.Conv2d(in_ch, neck_ch, 1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.Hardswish(inplace=True),
        ]

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.feature_dim = neck_ch
        self.head = make_head(self.feature_dim, self.num_classes, self.dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.flatten(x)


# Registry & factory

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "foodnet30": FoodNet30,
    "foodnet46": FoodNet46,
}


def build_model(name: str, num_classes: int = 251, dropout: float = 0.3,
                width_mult: float = 1.0, freeze: bool = False) -> BaseModel:
    """Preferred factory: instantiates a from-scratch model and enforces the
    < 10M parameter budget, raising ValueError if it's exceeded."""
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


# Self-check / parameter-budget report

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
        print(f"{info['name']:<18}{info['total_params_M']:>10}{info['feature_dim']:>10}{info['under_10M']!s:>8}")
