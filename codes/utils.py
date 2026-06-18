"""
FoodNet Utilities
==================
Seed management, device detection, parameter counting, and small helpers.
Apple-Silicon (MPS) aware, with CUDA / CPU fallbacks.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """
    Resolve the compute device.

    ``prefer`` (or config.DEVICE when None) may be:
      * "auto"  → pick the best available: CUDA → MPS (Apple Silicon) → CPU.
      * "cuda" / "mps" / "cpu" → use that backend if available, else warn and
        fall back to auto-detection (so a config pinned to "cuda" still runs on
        an M-series Mac instead of crashing).

    AMP (GradScaler) is CUDA-only and engages automatically there; on MPS/CPU the
    trainer runs full precision.
    """
    if prefer is None:
        try:
            from . import config as _cfg
            prefer = getattr(_cfg, "DEVICE", "auto")
        except Exception:
            prefer = "auto"

    def _auto() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    prefer = str(prefer).lower()
    if prefer in ("auto", "", "none"):
        return _auto()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # Requested backend not available → fall back gracefully.
    dev = _auto()
    import warnings
    warnings.warn(f"Requested device '{prefer}' unavailable; using '{dev.type}'.", stacklevel=2)
    return dev


def create_directories(*paths) -> None:
    """Create directories if they don't already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total (or trainable) parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def assert_param_budget(model: torch.nn.Module, limit: int = 10_000_000) -> int:
    """
    Raise if the model exceeds the exam's parameter cap. Returns the total count.
    Call right after building a model to fail fast on a budget violation.
    """
    total = count_parameters(model, trainable_only=False)
    if total >= limit:
        raise ValueError(f"Model has {total/1e6:.3f} M params (≥ {limit/1e6:.0f} M cap).")
    return total


def format_time(seconds: float) -> str:
    """Seconds → HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
