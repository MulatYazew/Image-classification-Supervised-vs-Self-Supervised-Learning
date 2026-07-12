"""
FoodNet Utilities
==================
Seed management, device detection, parameter counting, and small helpers.
Apple-Silicon (MPS) aware, with CUDA / CPU fallbacks.
"""

from __future__ import annotations

import json
import os
import random
import time
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


def make_amp_context(use_amp: bool, device: torch.device) -> tuple[bool, torch.dtype, torch.amp.GradScaler]:
    """
    One-call AMP setup, returning ``(autocast_enabled, autocast_dtype, scaler)``.

    ``autocast_enabled``/``autocast_dtype`` are exactly ``amp_enabled(use_amp,
    device)``/``amp_dtype_for(device)``. The ``GradScaler`` stays CUDA-only
    regardless of the MPS autocast decision: MPS doesn't need/support the same
    overflow-scaling machinery, only CUDA's fp16 autocast does. Consolidates
    the 4-line block that was previously repeated identically in
    ``Trainer.__init__``/``lr_finder`` (train.py), ``probe_supervised``
    (hyperparameter_tuning.py), and ``pretrain_simclr``/``pretrain_rotation``
    (self_supervised.py).
    """
    cuda_amp = use_amp and device.type == "cuda"
    return amp_enabled(use_amp, device), amp_dtype_for(device), torch.amp.GradScaler("cuda", enabled=cuda_amp)


#  Device-aware NUM_WORKERS

NUM_WORKERS_CANDIDATES = (4, 6, 8, 12)   # CUDA benchmark sweep (see select_num_workers)


def _num_workers_cache_path(results_dir: str | Path | None) -> Path:
    """Resolve the cache file path, falling back to config.RESULTS_DIR (or
    a bare "results" dir) when the caller doesn't pass one explicitly."""
    if results_dir is None:
        try:
            from . import config as _cfg
            results_dir = getattr(_cfg, "RESULTS_DIR", "results")
        except ImportError:
            results_dir = "results"
    return Path(results_dir) / "num_workers_benchmark.json"


def load_num_workers_benchmark(device: torch.device, results_dir: str | Path | None = None) -> dict | None:
    """
    This machine's cached CUDA ``select_num_workers`` benchmark entry --
    ``{"num_workers": int, "sec_per_batch": {"4": ..., ...}, "candidates": [...]}``
    -- or ``None`` for non-CUDA devices, or before the first benchmark has run.

    Shared by ``select_num_workers`` (cache-hit check) and ``config.py``
    (to override ``BENCHMARKED_SEC_PER_BATCH`` with THIS machine's own
    measurement instead of the Mac-only default when running on CUDA).
    """
    if device.type != "cuda":
        return None
    cache_path = _num_workers_cache_path(results_dir)
    if not cache_path.exists():
        return None
    try:
        cache = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cache.get(torch.cuda.get_device_name(0))


def _benchmark_num_workers_cuda(
    device: torch.device,
    candidates: tuple[int, ...] = NUM_WORKERS_CANDIDATES,
    n_batches: int = 20,
    n_repeats: int = 3,
    batch_size: int = 64,
    model_name: str = "foodnet46",
) -> dict[int, float]:
    """
    Median sec/batch per candidate ``num_workers``, measured on the REAL
    training pipeline -- the same harness (20-batch/3-repeat, foodnet46,
    batch 64, worker_init_fn + persistent_workers via
    ``data_handler.loader_kwargs``) used to hand-pick NUM_WORKERS=4 on the
    MacBook (see config.py's MPS comment), just automated here for CUDA.
    A fresh DataLoader is built per candidate so persistent_workers actually
    restarts a clean worker pool each time.
    """
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from . import config as _cfg
    from . import data_handler as dh
    from .model import build_model

    df = dh.build_dataframe(_cfg.TRAIN_CSV)
    needed = batch_size * n_batches
    if len(df) > needed:
        df = df.sample(n=needed, random_state=_cfg.SEED).reset_index(drop=True)
    dataset = dh.FoodDataset(df, _cfg.IMAGE_DIR, augment=True, image_size=_cfg.INPUT_SIZE)

    model = build_model(model_name, num_classes=_cfg.NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    sec_per_batch: dict[int, float] = {}
    for n in candidates:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=n, **dh.loader_kwargs(n))
        repeat_times = []
        for _ in range(n_repeats):
            model.train()
            it = iter(loader)
            t0 = time.time()
            for _ in range(n_batches):
                try:
                    images, labels = next(it)
                except StopIteration:
                    it = iter(loader)
                    images, labels = next(it)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            repeat_times.append((time.time() - t0) / n_batches)
            del it
        sec_per_batch[n] = float(np.median(repeat_times))
        del loader

    del model, optimizer
    torch.cuda.empty_cache()
    return sec_per_batch


def _pick_preferred_num_workers(sec_per_batch: dict[int, float],
                                preferred: tuple[int, ...] = (6, 8),
                                tolerance: float = 0.15) -> int:
    """
    Prefer the faster of the ``preferred`` candidates (6 or 8) unless some
    OTHER candidate beats it by more than ``tolerance`` (fractional sec/batch
    improvement) -- mirrors the ">15%" rule used to hand-pick NUM_WORKERS on
    the Mac (see config.py).
    """
    fastest_n = min(sec_per_batch, key=lambda n: sec_per_batch[n])
    preferred_present = {n: t for n, t in sec_per_batch.items() if n in preferred}
    if not preferred_present or fastest_n in preferred:
        return fastest_n
    best_preferred_n = min(preferred_present, key=lambda n: preferred_present[n])
    fastest_t, best_preferred_t = sec_per_batch[fastest_n], preferred_present[best_preferred_n]
    improvement = (best_preferred_t - fastest_t) / best_preferred_t
    return fastest_n if improvement > tolerance else best_preferred_n


def select_num_workers(
    device: torch.device,
    *,
    force_rebenchmark: bool = False,
    results_dir: str | Path | None = None,
    cpu_fallback: int = 4,
) -> int:
    """
    Device-aware DataLoader ``num_workers``, resolved automatically right
    after ``get_device()`` -- no manual benchmarking step on either machine.

      * MPS  -> 0 immediately. cv2's own thread pool fights the DataLoader's
        worker PROCESSES on Apple Silicon (see data_handler.py's
        worker_init_fn comment), and data loading isn't the bottleneck there
        anyway -- the MacBook Air M4 sweep in config.py measured pure data
        loading at only ~13% of total batch time, so this needs no benchmark.
      * CUDA -> benchmarks ``NUM_WORKERS_CANDIDATES`` (4, 6, 8, 12) on THIS
        machine's real training pipeline the first time it runs (see
        ``_benchmark_num_workers_cuda``), preferring 6 or 8 unless another
        candidate wins by more than 15% (``_pick_preferred_num_workers``).
        The result is cached to ``<results_dir>/num_workers_benchmark.json``
        keyed by ``torch.cuda.get_device_name(0)``, so every run AFTER the
        first on a given GPU just loads the cached value. Pass
        ``force_rebenchmark=True`` to ignore the cache (e.g. after a driver
        or hardware change).
      * CPU  -> ``min(cpu_fallback, os.cpu_count())``, no benchmark (data
        loading is rarely the bottleneck when the CPU is also doing the
        compute; extra workers would just compete with the main process for
        cores).

    Every call prints the resolved decision, mirroring config.py's existing
    "documented decision" comment style.
    """
    if device.type == "mps":
        print("[num_workers] device=mps -> 0 (cv2/DataLoader-worker thread "
              "contention on Apple Silicon; data loading isn't the bottleneck "
              "on MPS -- see data_handler.py / config.py)")
        return 0

    if device.type == "cpu":
        n = min(cpu_fallback, os.cpu_count() or 1)
        print(f"[num_workers] device=cpu -> {n} (min({cpu_fallback}, os.cpu_count()), no benchmark)")
        return n

    # CUDA: cache-or-benchmark, keyed by GPU name so different machines (or a
    # swapped card) don't share a stale decision.
    gpu_name = torch.cuda.get_device_name(0)
    cache_path = _num_workers_cache_path(results_dir)

    if not force_rebenchmark:
        cached = load_num_workers_benchmark(device, results_dir=results_dir)
        if cached is not None:
            n = int(cached["num_workers"])
            print(f"[num_workers] device=cuda ({gpu_name}) -> {n} "
                  f"(cached benchmark ← {cache_path})")
            return n

    print(f"[num_workers] device=cuda ({gpu_name}) -> benchmarking "
          f"{NUM_WORKERS_CANDIDATES} (first run on this GPU; result will be "
          f"cached to {cache_path}) ...")
    sec_per_batch = _benchmark_num_workers_cuda(device, NUM_WORKERS_CANDIDATES)
    chosen = _pick_preferred_num_workers(sec_per_batch)
    fastest_n = min(sec_per_batch, key=lambda n: sec_per_batch[n])
    reason = "fastest candidate" if chosen == fastest_n else \
        "preferred 6/8 (fastest candidate didn't win by >15%)"

    table = "  ".join(f"{n}:{t:.3f}s/batch" for n, t in sorted(sec_per_batch.items()))
    print(f"[num_workers] sweep -> {table}")
    print(f"[num_workers] chosen={chosen} ({reason})")

    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    cache[gpu_name] = {
        "num_workers": chosen,
        "sec_per_batch": {str(n): t for n, t in sec_per_batch.items()},
        "candidates": list(NUM_WORKERS_CANDIDATES),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))
    print(f"[num_workers] cached decision → {cache_path}")
    return chosen


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
