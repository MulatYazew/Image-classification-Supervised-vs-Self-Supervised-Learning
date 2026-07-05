"""
FoodNet Utilities
==================
Seed management, device detection, parameter counting, and small helpers.
Apple-Silicon (MPS) aware, with CUDA / CPU fallbacks.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

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


def amp_enabled(use_amp: bool, device: torch.device) -> bool:
    """
    True if autocast should engage.

    CUDA: always follows ``use_amp`` (autocast is a clear win there).

    MPS: gated by config.AMP_MPS_ENABLED, default False. MEASURED on a
    MacBook Air M4 (foodnet46, batch 64, 20-batch/3-repeat harness): FP32
    ran at 0.658 s/batch vs 0.748-0.749 s/batch for BOTH float16 and
    bfloat16 autocast -- i.e. autocast was ~14% SLOWER, not faster, on this
    backend/model, reproduced across two independent FP32 re-checks. The
    infrastructure (this function + amp_dtype_for) is kept because it's the
    correct plumbing on CUDA and costs nothing when disabled, but it must
    NOT be forced on for MPS by default given it measurably regresses this
    workload -- flip config.AMP_MPS_ENABLED=True to opt in if a future torch
    version or different model changes this.

    CPU never autocasts.
    """
    if device.type == "cuda":
        return use_amp
    if device.type == "mps":
        try:
            from . import config as _cfg
        except ImportError:
            return False
        return use_amp and getattr(_cfg, "AMP_MPS_ENABLED", False)
    return False


def amp_dtype_for(device: torch.device) -> torch.dtype:
    """
    Autocast dtype per backend: float16 on CUDA (paired with a CUDA-only
    GradScaler for overflow protection). On MPS, config.AMP_MPS_DTYPE picks
    float16 (default) or bfloat16 -- set from a 1-epoch stability sanity check
    on this project's from-scratch BatchNorm models (see config.py). Unused
    (returns float32) when autocast is disabled/CPU.
    """
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        # Mirrors get_device()'s defensive fallback above: every caller in
        # this package imports via proper relative/package imports (codes.*),
        # so this relative import always resolves in practice -- the
        # try/except only guards against this module being executed outside
        # the codes package (e.g. copied elsewhere without its __init__.py).
        try:
            from . import config as _cfg
        except ImportError:
            return torch.float16
        return torch.bfloat16 if getattr(_cfg, "AMP_MPS_DTYPE", "float16") == "bfloat16" else torch.float16
    return torch.float32


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


def stage_done(*expected_paths: str | Path) -> bool:
    """Return True if every expected output already exists on disk.

    Use this to guard an expensive pipeline stage: call it with the file(s)
    that stage is supposed to produce, and skip recomputation (loading the
    existing outputs instead) when it returns True.
    """
    return all(Path(p).exists() for p in expected_paths)


def is_fresh(target: str | Path, *dep_paths: str | Path) -> bool:
    """
    True if ``target`` exists and is at least as new as every existing path in
    ``dep_paths`` (by mtime). Dependencies that don't exist are ignored (a
    missing upstream file can't make a target stale). Used to decide whether a
    derived artifact (e.g. the cleaned-manifest CSV) needs rebuilding after its
    inputs (e.g. the outlier-review CSVs) changed.
    """
    target = Path(target)
    if not target.exists():
        return False
    target_mtime = target.stat().st_mtime
    deps = [Path(p) for p in dep_paths if Path(p).exists()]
    return all(target_mtime >= p.stat().st_mtime for p in deps)


class LocalEarlyStopper:
    """
    Per-candidate early-stop tracker for a hyperparameter-SEARCH training loop
    (NOT the same thing as Trainer's own PATIENCE, which governs the Phase C
    full retrain of the winning config).

    Instantiate ONE fresh tracker per candidate config so state never leaks
    between candidates, call ``update(metric)`` after every epoch with a
    "higher is better" value, and stop the loop as soon as it returns True.
    ``best`` always holds the best value seen so far, so callers can rank a
    candidate that stopped early by its best (not final) epoch.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.epochs_since_improve = 0

    def update(self, metric: float) -> bool:
        """Record this epoch's metric; return True if the loop should stop now."""
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.epochs_since_improve = 0
        else:
            self.epochs_since_improve += 1
        return self.patience > 0 and self.epochs_since_improve >= self.patience


def format_time(seconds: float) -> str:
    """Seconds → HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
