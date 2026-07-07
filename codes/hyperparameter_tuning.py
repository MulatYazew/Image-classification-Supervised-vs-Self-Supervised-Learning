"""
FoodNet — Hyperparameter Tuning
================================
Implements Task 5 of the exam ("tune the hyperparameters of the models to
achieve better performance"), sized for a single Apple Silicon Mac (no
multi-GPU cluster), while *respecting and recording* the constraints you
care about:

    * the < 10 M parameter cap,
    * peak accelerator memory (MB) — CUDA gets a true peak, MPS a live
      allocation reading, CPU reports 0 (see peak_mem_mb),
    * wall-clock time per trial (s).

Design (kept affordable on limited, single-device compute)
------------------------------------------------------------
  * A reproducible GRID is defined in "default_sl_grid" / "default_ssl_grid".
    Every trial is only a short "probe_epochs" run (not full training), and
    "config.TUNE_STRATEGY" controls how much of the grid is actually probed:
    "grid" (exhaustive — only realistic for small grids), "random" (sample
    N configs), or "successive_halving" (start every config cheap, keep the
    top half, double the budget, repeat). Within each probe, a candidate that
    clearly plateaus stops early (config.TUNE_EARLY_STOP_PATIENCE) instead of
    burning the rest of its budget. The winning config is then trained to
    convergence separately (train.Trainer / run_supervised_pipeline).
  * Selection metric = config.TUNE_SELECTION_METRIC, "f1_macro" by default
    (NOT accuracy — see grid_search_over_configs for why). Ties (within
    "tie_tol") are broken by lower time then lower memory, so among
    equally-good configs we prefer the cheaper one.
  * Every trial logs the metric, params(M), peak-mem(MB), time(s) and
    epochs_run to a tidy list of dicts you can dump to CSV and drop straight
    into the report table.

The module is deliberately framework-light: it depends only on "build_model"
and a couple of tiny callbacks, so it works for both the SL and SSL tasks.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from .model import build_model, BaseModel
from .loss_function import build_criterion
from .data_handler import compute_class_weights, check_single_imbalance_correction

from .utils import get_device, LocalEarlyStopper, make_amp_context

# Progress bars for the probe loops. Falls back to a no-op shim if tqdm isn't
# installed, so the module never hard-depends on it.
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):   # type: ignore
        return iterable if iterable is not None else _NullBar()

    class _NullBar:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass



#  Result container 

@dataclass
class TrialResult:
    """One row of the tuning table — everything the report needs per config."""
    config: dict
    val_accuracy: float
    val_f1_macro: float
    params_M: float
    peak_mem_MB: float
    time_s: float
    under_10M: bool
    probe_epochs: int = 0
    epochs_run: int = 0            # actual epochs this candidate trained before stopping
                                    # (== probe_epochs unless the search-loop early stop fired)
    data_subset: str = "full"     # "full" | "capped" — was this trial run on a documented subset?
    extra: dict = field(default_factory=dict)   # e.g. per-class F1 if computed

    def row(self) -> dict:
        """Flatten to a single dict (config keys + metrics) for CSV/DataFrame."""
        r = {f"cfg.{k}": v for k, v in self.config.items()}
        r.update(
            val_accuracy=round(self.val_accuracy, 4),
            val_f1_macro=round(self.val_f1_macro, 4),
            params_M=self.params_M,
            peak_mem_MB=round(self.peak_mem_MB, 1),
            time_s=round(self.time_s, 1),
            under_10M=self.under_10M,
            probe_epochs=self.probe_epochs,
            epochs_run=self.epochs_run,
            data_subset=self.data_subset,
        )
        r.update(self.extra)
        return r


#  Default search grids 

def default_sl_grid() -> dict[str, list]:
    """
    Supervised-learning grid. Keys map to ``build_model`` / optimiser knobs.

    Chosen to cover the highest-leverage axes (LR, optimiser, weight decay,
    dropout, model width) without exploding the combinatorics. ``width_mult``
    lets the search trade accuracy against the param budget directly.
    """
    return {
        "model_name":      ["foodnet46"],          # redesigned MBConv model (proposed)
        "lr":              [3e-4, 1e-3],          # the single most important knob
        "optimizer":       ["adamw", "sgd"],       # AdamW vs SGD+momentum
        "weight_decay":    [1e-4, 5e-4],          # regularisation strength
        "dropout":         [0.2, 0.3],            # head dropout (over-fitting guard)
        "width_mult":      [1.0],                 # 1.0 = 7.6 M; lower to shrink
        "label_smoothing": [0.0, 0.1],            # helps with 251 fine-grained classes
    }


def default_ssl_grid() -> dict[str, list]:
    """
    Self-supervised grid (applies to whichever pretext method you tune).

    The method itself (simclr or rotation) is NOT a grid axis here, because the
    two pretext tasks consume differently shaped inputs (SimCLR needs augmented
    pairs, rotation needs single images) and therefore different DataLoaders.
    Instead you run tune_ssl once per method and compare the winners — exactly
    the SL-vs-SSL-vs-method workflow the report asks for.

    temperature only affects SimCLR; it is ignored for rotation. classifier
    selects the downstream traditional read-out and is tuned too.
    """
    return {
        "model_name":     ["foodnet46"],
        "lr":             [5e-4, 1e-3],
        "temperature":    [0.1, 0.5],            # NT-Xent sharpness (SimCLR only)
        "weight_decay":   [1e-4],
        "projection_dim": [128],
        "classifier":     ["logreg", "linear_svm", "knn"],
        "width_mult":     [1.0, 0.75],
    }


def imbalance_grid() -> dict[str, list]:
    """
    Its OWN small grid for the imbalance-handling axis: loss type × per-class
    weight scheme. Tuned as a dedicated phase AFTER LR/weight-decay/dropout are
    fixed (see the project's priority order), with the winner chosen by
    tail-class F1, not just the aggregate — a scheme that quietly zeroes out
    the rarest classes should not win on macro-F1 alone.
    """
    return {
        "loss_type":           ["ce", "weighted_ce", "focal"],
        "class_weight_scheme": ["sqrt_inv", "effective"],
    }


def augmentation_grid() -> dict[str, list]:
    """
    Augmentation-strength / mix-method axis. Tuned AFTER the model is
    numerically stable (LR + imbalance axes fixed) — strong augmentation on
    top of a badly-tuned LR just adds noise to the ranking signal.
    """
    return {
        "augmentation_intensity": [0.3, 0.5, 0.8],
        "mix_method":             ["none", "mixup", "cutmix"],
    }


def grid_with_overrides(winner_config: dict, varying: dict[str, list]) -> dict[str, list]:
    """
    Freeze every key of ``winner_config`` (as a single-value list) except the
    keys in ``varying``, which keep their multiple candidate values.

    This is what makes the "priority order" tuning strategy possible: tune LR
    first, fix it, tune weight-decay/dropout, fix those, tune the imbalance
    axis, etc — each phase is just ``grid_search(grid_with_overrides(prev_best,
    next_axis), ...)``.
    """
    grid = {k: [v] for k, v in winner_config.items()}
    grid.update(varying)
    return grid


def iter_grid(grid: dict[str, list]) -> Iterable[dict]:
    """Yield every combination of the grid as a config dict (Cartesian product)."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def sample_random_configs(grid: dict[str, list], n: int, seed: int = 42) -> list[dict]:
    """
    Sample ``n`` configs uniformly at random (without replacement) from the
    grid's Cartesian product, instead of enumerating it exhaustively.

    For the same compute budget, random search finds better configs than grid
    search once you're combining more than ~2 axes, because grid search wastes
    trials on unimportant axis combinations (Bergstra & Bengio, 2012). Returns
    the full product if ``n`` exceeds its size.
    """
    all_configs = list(iter_grid(grid))
    if n >= len(all_configs):
        return all_configs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_configs), size=n, replace=False)
    return [all_configs[i] for i in idx]


#  Optimiser factory 

def make_optimizer(model: nn.Module, cfg: dict):
    """Build the optimiser named in ``cfg`` (AdamW or SGD+momentum)."""
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.get("optimizer", "adamw")
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 1e-4)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        # Nesterov momentum is the standard from-scratch CNN choice.
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'.")


#  Memory / time instrumentation 

def reset_peak_mem(device: torch.device) -> None:
    """Reset the peak-memory counter (CUDA); clear cache on MPS for a clean baseline."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # torch.mps has no resettable peak stat; empty the cache so the
        # current-allocated reading in peak_mem_mb starts from a clean baseline.
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def peak_mem_mb(device: torch.device) -> float:
    """
    Accelerator memory in MB for the tuning table.

      * CUDA → true peak via max_memory_allocated.
      * MPS  → current allocated memory (torch.mps exposes no peak counter, so
               this is the live allocation after the probe — fine for comparing
               configs against each other on the Mac).
      * CPU  → 0 (nothing to measure).
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            torch.mps.synchronize()
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return 0.0
    return 0.0


#  Short SL probe (train a few epochs, return val accuracy) 

def probe_supervised(
    cfg: dict,
    train_loader,
    val_loader,
    device: torch.device,
    probe_epochs: int = 5,
    num_classes: int = 251,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    use_weighted_sampler: bool = False,
    selection_metric: str = "f1_macro",
    early_stop_patience: int = 0,
    early_stop_check_batches: int | None = None,
) -> tuple[float, BaseModel, float, int]:
    """
    Train ``model_name`` for up to ``probe_epochs`` under ``cfg`` and return
    ``(val_accuracy, model, val_f1_macro, epochs_run)``. This is a *probe*, not
    full training: it ranks configs cheaply. The winner is later trained to
    convergence elsewhere.

    ``cfg`` may set ``loss_type`` ("ce"|"weighted_ce"|"focal") and
    ``class_weight_scheme`` ("sqrt_inv"|"inv"|"effective") to tune the
    imbalance-handling axis (see ``imbalance_grid``); both default to the
    library's plain-CE behaviour when absent. Class weights are computed from
    ``train_loader.dataset.df`` and skipped entirely when
    ``use_weighted_sampler=True``, enforced via
    ``data_handler.check_single_imbalance_correction`` so a sampler-based run
    can never accidentally double-correct.

    Two DIFFERENT validation subsets are in play here, both distinct from the
    REAL validation set used by Phase C / the final reported metrics:
      * ``val_loader`` itself is already ``tune_val_loader`` in the notebook
        -- a documented, capped Phase A/B subset (config.
        TUNE_SUBSET_IMAGES_PER_CLASS), never the full real val set.
      * ``early_stop_patience`` > 0 additionally evaluates a cheap PROXY
        signal every epoch (not just once at the end) — only the first
        ``early_stop_check_batches`` (config.TUNE_EARLY_STOP_CHECK_BATCHES)
        batches of ``val_loader``, not a full pass — and stops this candidate
        once that proxy hasn't improved for ``early_stop_patience`` epochs.
        This is a search-loop-local mechanism, independent of ``Trainer``'s
        own ``PATIENCE`` (which only governs the Phase C full retrain).
    The returned ``val_accuracy``/``val_f1_macro`` (used to RANK/SELECT
    configs) always come from ONE full ``val_loader`` pass at the end of the
    loop (whichever epoch it stopped/finished at) — NOT from the cheap
    per-epoch proxy above, so config ranking is never based on a 10-batch
    sample. This trades a small amount of accuracy (the reported metric is
    from the END of the loop, not necessarily the exact best epoch a
    full-precision per-epoch check would have found) for cutting N-1 of N
    full validation passes per candidate. 0 (default) disables
    early_stop_patience entirely and keeps the original
    single-evaluation-at-the-end behaviour, unaffected by this trade-off.
    """
    model = build_model(
        cfg["model_name"], num_classes=num_classes,
        dropout=cfg.get("dropout", 0.3), width_mult=cfg.get("width_mult", 1.0),
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=probe_epochs, eta_min=1e-6)

    class_weights = None
    if not use_weighted_sampler:
        train_df = getattr(train_loader.dataset, "df", None)
        if train_df is not None:
            class_weights = compute_class_weights(
                train_df, num_classes=num_classes,
                scheme=cfg.get("class_weight_scheme", "sqrt_inv"),
            ).to(device)
    check_single_imbalance_correction(use_weighted_sampler, class_weights)
    criterion = build_criterion(
        cfg.get("loss_type", "ce"), class_weights=class_weights,
        gamma=cfg.get("focal_gamma", 2.0), label_smoothing=cfg.get("label_smoothing", 0.0),
    ).to(device)

    # Autocast now also engages on MPS (measured speedup on this M4 Mac -- see
    # config.AMP_MPS_DTYPE); GradScaler stays CUDA-only since MPS doesn't
    # need/support the same overflow-scaling machinery.
    amp_on, amp_dtype, scaler = make_amp_context(use_amp, device)

    def _validate(max_batches: int | None = None) -> tuple[float, float]:
        # Validation accuracy AND macro-F1 (the selection metric — see
        # config.TUNE_SELECTION_METRIC). Accuracy alone can look fine at epoch 5
        # while the ~19:1 imbalance means tail classes are being ignored entirely.
        # ``max_batches`` caps this to a cheap proxy pass (the per-epoch
        # early-stop CHECK only); None (default) is a full pass, used for the
        # metric actually returned/ranked on — see probe_supervised's docstring.
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        loader = val_loader if max_batches is None else itertools.islice(val_loader, max_batches)
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="probe val", leave=False, unit="batch",
                                       total=max_batches):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                    preds = model(images).argmax(1)
                correct += (preds.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        acc = correct / max(total, 1)
        f1_macro = f1_score(np.concatenate(all_labels), np.concatenate(all_preds),
                            average="macro", zero_division=0)
        model.train()
        return acc, f1_macro

    stopper = LocalEarlyStopper(early_stop_patience) if early_stop_patience > 0 else None
    epochs_run = 0

    for epoch in range(1, probe_epochs + 1):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"probe ep {epoch}/{probe_epochs}",
                   leave=False, unit="batch")
        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # Live training stats on the bar.
            run_loss += loss.item() * images.size(0)
            run_correct += (outputs.argmax(1) == labels).sum().item()
            run_n += labels.size(0)
            if _HAS_TQDM:
                bar.set_postfix(loss=f"{run_loss / max(run_n, 1):.3f}",
                                acc=f"{run_correct / max(run_n, 1):.3f}")
        scheduler.step()
        epochs_run = epoch

        if stopper is not None:
            # Cheap proxy check ONLY -- capped to early_stop_check_batches,
            # never a full validation pass (see docstring).
            check_batches = early_stop_check_batches
            if check_batches is None:
                try:
                    from . import config as _cfg
                    check_batches = getattr(_cfg, "TUNE_EARLY_STOP_CHECK_BATCHES", 10)
                except ImportError:
                    check_batches = 10
            proxy_acc, proxy_f1_macro = _validate(max_batches=check_batches)
            current = proxy_f1_macro if selection_metric == "f1_macro" else proxy_acc
            stop_now = stopper.update(current)
            if stop_now:
                print(f"[tune] search early-stop at epoch {epoch}/{probe_epochs} "
                      f"(no {selection_metric} improvement for {early_stop_patience} epochs, "
                      f"proxy check over {check_batches} batches)")
                break

    # Reported/ranked metric: ALWAYS one full val_loader pass at the END of
    # the loop -- not the cheap per-epoch proxy above -- so config
    # ranking is never based on a small batch sample.
    best_val_acc, best_val_f1_macro = _validate(max_batches=None)

    return best_val_acc, model, best_val_f1_macro, epochs_run


#  Main grid-search driver 

def _metric_of(result: TrialResult, selection_metric: str) -> float:
    """Look up the configured ranking metric on a TrialResult."""
    return result.val_f1_macro if selection_metric == "f1_macro" else result.val_accuracy


def grid_search(
    grid: dict[str, list],
    probe_fn: Callable[..., tuple[float, BaseModel, float, int]],
    device: torch.device,
    tie_tol: float = 0.005,
    verbose: bool = True,
    selection_metric: str = "f1_macro",
    **probe_kwargs,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Exhaustively evaluate ``grid`` with ``probe_fn`` and pick the best config.

    Selection rule (in priority order, matching the brief):
        1. HARD constraint: keep only configs with < 10 M params.
        2. Maximise ``selection_metric`` — macro-F1 by default (NOT accuracy),
           because with ~19:1 class imbalance a config can look fine on
           accuracy while ignoring the tail classes entirely.
        3. Break near-ties (within ``tie_tol`` of the metric) by lower time,
           then by lower peak memory — prefer the cheaper of two equally-good nets.

    Args:
        probe_fn : a callable like ``probe_supervised`` returning
                   (val_acc, model, val_f1_macro).
        probe_kwargs : forwarded to ``probe_fn`` (loaders, epochs, etc.).

    Returns:
        (best_result, all_results) — ``all_results`` is ready to write to CSV
        for the report's tuning table.
    """
    configs = list(iter_grid(grid))
    return grid_search_over_configs(
        configs, probe_fn, device, tie_tol=tie_tol, verbose=verbose,
        selection_metric=selection_metric, **probe_kwargs
    )


def grid_search_over_configs(
    configs: list[dict],
    probe_fn: Callable[..., tuple[float, BaseModel, float, int]],
    device: torch.device,
    tie_tol: float = 0.005,
    verbose: bool = True,
    selection_metric: str = "f1_macro",
    data_subset: str = "full",
    **probe_kwargs,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Core driver shared by grid_search (SL) and tune_ssl (SSL): evaluate an
    explicit list of config dicts with probe_fn and pick the best under the same
    <10M / selection-metric / cheap-tie-break rule. Taking a config LIST (not a
    grid) lets the SSL path drop redundant rotation-vs-temperature duplicates
    first, and lets successive-halving re-probe a shrinking config list.

    ``selection_metric``: "f1_macro" (default, imbalance-safe) or "accuracy".
    ``data_subset``: recorded on each TrialResult ("full" | "capped") so the
    tuning-table CSV documents whether a trial ran on the full dataset or a
    documented ``config.TUNE_SUBSET_IMAGES_PER_CLASS`` cap (Phase A/B).
    """
    if selection_metric not in ("f1_macro", "accuracy"):
        raise ValueError(f"Unknown selection_metric '{selection_metric}'. Choose: f1_macro, accuracy.")

    results: list[TrialResult] = []
    if verbose:
        print(f"[tune] {len(configs)} configurations to probe "
              f"(selection_metric={selection_metric}, data_subset={data_subset}).\n")

    config_bar = tqdm(total=len(configs), desc="configs", unit="cfg") if _HAS_TQDM else None
    for i, cfg in enumerate(configs, 1):
        # Pre-check the param budget WITHOUT training, so doomed configs cost ~0.
        probe_model = build_model(
            cfg["model_name"],
            num_classes=probe_kwargs.get("num_classes", 251),
            dropout=cfg.get("dropout", 0.3),
            width_mult=cfg.get("width_mult", 1.0),
        )
        info = probe_model.model_info()
        del probe_model
        if not info["under_10M"]:
            if verbose:
                print(f"[tune] ({i}/{len(configs)}) SKIP {cfg} — "
                      f"{info['total_params_M']} M ≥ 10 M")
            if config_bar is not None:
                config_bar.update(1)
            continue

        reset_peak_mem(device)
        t0 = time.time()
        val_acc, _, val_f1_macro, epochs_run = probe_fn(
            cfg, device=device, selection_metric=selection_metric, **probe_kwargs)
        elapsed = time.time() - t0
        peak = peak_mem_mb(device)

        res = TrialResult(
            config=cfg, val_accuracy=val_acc, val_f1_macro=val_f1_macro,
            params_M=info["total_params_M"], peak_mem_MB=peak, time_s=elapsed,
            under_10M=True, probe_epochs=probe_kwargs.get("probe_epochs", 0),
            epochs_run=epochs_run, data_subset=data_subset,
        )
        results.append(res)
        if config_bar is not None:
            best_so_far = max(_metric_of(r, selection_metric) for r in results)
            config_bar.set_postfix(last=f"{_metric_of(res, selection_metric):.3f}",
                                   best=f"{best_so_far:.3f}")
            config_bar.update(1)
        if verbose:
            print(f"[tune] ({i}/{len(configs)}) acc={val_acc:.4f} f1_macro={val_f1_macro:.4f} "
                  f"params={info['total_params_M']}M mem={peak:.0f}MB "
                  f"time={elapsed:.0f}s :: {cfg}")

    if config_bar is not None:
        config_bar.close()

    if not results:
        raise RuntimeError("No valid (<10M) configurations were probed.")

    # Sort: selection_metric desc, then time asc, then memory asc (cheap tie-break).
    results_sorted = sorted(results, key=lambda r: (-_metric_of(r, selection_metric), r.time_s, r.peak_mem_MB))
    best = results_sorted[0]
    # Among configs within tie_tol of the best metric, pick the cheapest.
    best_metric = _metric_of(best, selection_metric)
    near = [r for r in results_sorted if best_metric - _metric_of(r, selection_metric) <= tie_tol]
    best = min(near, key=lambda r: (r.time_s, r.peak_mem_MB))

    if verbose:
        print("\n[tune] BEST CONFIG")
        print(f"       {best.config}")
        print(f"       val_acc={best.val_accuracy:.4f} | val_f1_macro={best.val_f1_macro:.4f} | "
              f"params={best.params_M}M | mem={best.peak_mem_MB:.0f}MB | time={best.time_s:.0f}s")
    return best, results


#  Successive halving

def estimate_successive_halving_epochs(n_configs: int, initial_epochs: int,
                                       reduction_factor: int = 2, min_configs: int = 1) -> int:
    """
    Total epoch-equivalents ``successive_halving_search`` will actually run --
    the exact same round-by-round arithmetic it uses internally (keep the top
    ``1/reduction_factor`` survivors, double their epoch budget, repeat until
    ``min_configs`` remain), computed WITHOUT running anything.

    This is what makes "TUNE_STRATEGY=successive_halving, N_RANDOM_CONFIGS=20,
    PROBE_EPOCHS=5" NOT mean 20*5=100 epochs: e.g. n_configs=20,
    initial_epochs=5, reduction_factor=2 comes to 460 (100 + 100 + 100 + 80 +
    80 across 5 rounds), not 100. Doesn't model <10M-param rejections or
    search-loop early stopping (config.TUNE_EARLY_STOP_PATIENCE) -- both can
    only REDUCE the real total below this, so treat it as an upper bound.
    """
    remaining = n_configs
    epochs = initial_epochs
    total = 0
    while True:
        total += remaining * epochs
        keep_n = max(min_configs, remaining // reduction_factor)
        if keep_n >= remaining or remaining <= min_configs:
            break
        remaining = keep_n
        epochs *= reduction_factor
    return total


def print_tuning_budget(strategy: str, n_configs: int, probe_epochs: int,
                        reduction_factor: int = 2, min_configs: int = 1,
                        batches_per_epoch: int | None = None) -> int:
    """
    Print a one-line "[tune-budget]" summary BEFORE a search starts: strategy,
    total configs, total epoch-equivalents, and an estimated wall-clock time
    (using config.BENCHMARKED_SEC_PER_BATCH and ``batches_per_epoch``, when
    given). Returns the epoch-equivalent count so callers can reuse it.

    Exists so "I didn't realize probe_epochs=5 meant hours of compute" can't
    happen silently again -- every tune_supervised/tune_ssl call prints this
    regardless of strategy.
    """
    if strategy == "successive_halving":
        total_epochs = estimate_successive_halving_epochs(
            n_configs, probe_epochs, reduction_factor, min_configs)
        detail = "successive-halving (budget doubles each round, see estimate_successive_halving_epochs)"
    else:
        total_epochs = n_configs * probe_epochs
        detail = strategy

    msg = (f"[tune-budget] strategy={strategy} | configs={n_configs} | "
          f"probe_epochs={probe_epochs} | ~{total_epochs} total epoch-equivalents ({detail})")
    try:
        from . import config as _cfg
    except ImportError:
        _cfg = None
    sec_per_batch = getattr(_cfg, "BENCHMARKED_SEC_PER_BATCH", None) if _cfg else None
    if sec_per_batch and batches_per_epoch:
        from .utils import format_time
        est_seconds = total_epochs * batches_per_epoch * sec_per_batch
        msg += f" | est. wall-clock ~{format_time(est_seconds)} @ {sec_per_batch:.2f}s/batch"
    print(msg)
    return total_epochs


def successive_halving_search(
    configs: list[dict],
    probe_fn: Callable[..., tuple[float, BaseModel, float, int]],
    device: torch.device,
    initial_epochs: int,
    selection_metric: str = "f1_macro",
    reduction_factor: int = 2,
    min_configs: int = 1,
    verbose: bool = True,
    **probe_kwargs,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Successive-halving search: start ALL configs at ``initial_epochs``, keep
    the top ``1/reduction_factor`` by ``selection_metric``, double their
    epoch budget, and repeat until ``min_configs`` remain. The assumption
    (documented in the project's tuning plan): a config that is clearly worse
    at a short probe budget almost never overtakes a good config after many
    more epochs, so most of the compute goes to the configs worth it.

    Returns ``(best, results_from_the_final_round)``. Call ``results_to_csv``
    on the returned ``results`` for the report's tuning table (it only
    reflects the LAST round's survivors at their final budget; combine with
    the earlier rounds' printed output if you want the full elimination history).
    """
    if verbose:
        train_loader = probe_kwargs.get("train_loader")
        batches_per_epoch = len(train_loader) if train_loader is not None else None
        print_tuning_budget("successive_halving", len(configs), initial_epochs,
                           reduction_factor, min_configs, batches_per_epoch)

    remaining = list(configs)
    epochs = initial_epochs
    round_num = 1
    all_results: list[TrialResult] = []
    while True:
        if verbose:
            print(f"\n[successive-halving] round {round_num}: "
                  f"{len(remaining)} configs @ {epochs} probe epochs")
        _, results = grid_search_over_configs(
            remaining, probe_fn, device, verbose=verbose,
            selection_metric=selection_metric, probe_epochs=epochs, **probe_kwargs,
        )
        all_results = results
        results_sorted = sorted(results, key=lambda r: -_metric_of(r, selection_metric))
        keep_n = max(min_configs, len(results_sorted) // reduction_factor)
        if keep_n >= len(remaining) or len(remaining) <= min_configs:
            break
        remaining = [r.config for r in results_sorted[:keep_n]]
        epochs *= reduction_factor
        round_num += 1

    best = max(all_results, key=lambda r: _metric_of(r, selection_metric))
    if verbose:
        print(f"\n[successive-halving] done after {round_num} round(s). "
              f"BEST: {best.config} | val_f1_macro={best.val_f1_macro:.4f} "
              f"val_acc={best.val_accuracy:.4f}")
    return best, all_results


#  CSV export helper 

def results_to_csv(results: list[TrialResult], path: str) -> None:
    """Write all trial rows to ``path`` — drop the file straight into the report."""
    import csv
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [r.row() for r in results]
    keys: list[str] = []
    for row in rows:                               # union of keys, order-stable
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tune] Wrote {len(rows)} rows → {path}")


def top_n_table(results: list[TrialResult], n: int = 5, selection_metric: str = "f1_macro") -> pd.DataFrame:
    """
    Top-``n`` configs ranked by ``selection_metric`` — the exact table the
    report's "hyperparameter tuning" section needs (config + val_acc +
    val_f1_macro + cost columns for the top candidates).
    """
    rows = [r.row() for r in sorted(results, key=lambda r: -_metric_of(r, selection_metric))[:n]]
    return pd.DataFrame(rows)


def write_tuning_summary(best: TrialResult, results: list[TrialResult], path: str,
                         n: int = 5, selection_metric: str = "f1_macro") -> None:
    """
    Write a short markdown summary — top-``n`` configs table plus one line on
    why the winner was chosen — ready to paste into the report or drop as a
    figure/table source. Complements ``results_to_csv`` (the full trial log).
    """
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = top_n_table(results, n=n, selection_metric=selection_metric)
    try:
        table_str = table.to_markdown(index=False)   # needs the optional 'tabulate' package
    except ImportError:
        table_str = table.to_string(index=False)
    lines = [
        f"# Hyperparameter search — top {n} configs (ranked by {selection_metric})",
        "",
        table_str,
        "",
        f"**Winner:** `{best.config}`",
        f"- val_accuracy = {best.val_accuracy:.4f}, val_f1_macro = {best.val_f1_macro:.4f}",
        f"- params = {best.params_M} M, peak_mem = {best.peak_mem_MB:.0f} MB, time = {best.time_s:.0f} s",
        f"- Chosen by maximising {selection_metric}; ties within tolerance broken by lower "
        "time then lower memory (see grid_search_over_configs).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[tune] Wrote tuning summary → {path}")


#  Convenience: end-to-end SL tuning

def tune_supervised(
    train_loader,
    val_loader,
    device: torch.device | None = None,
    grid: dict[str, list] | None = None,
    probe_epochs: int = 5,
    num_classes: int = 251,
    csv_path: str | None = None,
    strategy: str = "grid",
    selection_metric: str = "f1_macro",
    n_random_configs: int = 20,
    use_weighted_sampler: bool = False,
    data_subset: str = "full",
    seed: int = 42,
    early_stop_patience: int = 0,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    One-call SL hyperparameter sweep. Returns ``(best, all_results)`` and, if
    ``csv_path`` is given, also writes the full table for the report.

    ``strategy``:
      * "grid"                — exhaustive Cartesian product of ``grid``.
      * "random"               — sample ``n_random_configs`` from the same
                                 product (config.TUNE_N_RANDOM_CONFIGS);
                                 preferred once ``grid`` combines >~2 axes.
      * "successive_halving"   — start every sampled config at
                                 ``probe_epochs``, keep the top half, double
                                 the budget, repeat (config.TUNE_STRATEGY).

    ``early_stop_patience`` (config.TUNE_EARLY_STOP_PATIENCE) is forwarded to
    ``probe_supervised``: each candidate stops as soon as it plateaus for that
    many epochs, instead of always burning the full ``probe_epochs`` budget.
    0 disables it (original behaviour).

    ``device`` defaults to ``utils.get_device()`` when omitted, so the same call
    resolves to MPS on the Mac and CUDA on your friend's PC with no edits.
    """
    device = device or get_device()
    grid = grid or default_sl_grid()
    probe_kwargs = dict(
        train_loader=train_loader, val_loader=val_loader,
        num_classes=num_classes, use_weighted_sampler=use_weighted_sampler,
        early_stop_patience=early_stop_patience,
    )
    batches_per_epoch = len(train_loader)

    if strategy == "grid":
        n_configs = len(list(iter_grid(grid)))
        print_tuning_budget("grid", n_configs, probe_epochs, batches_per_epoch=batches_per_epoch)
        best, results = grid_search(
            grid, probe_supervised, device, selection_metric=selection_metric,
            data_subset=data_subset, probe_epochs=probe_epochs, **probe_kwargs,
        )
    elif strategy == "random":
        configs = sample_random_configs(grid, n_random_configs, seed=seed)
        print_tuning_budget("random", len(configs), probe_epochs, batches_per_epoch=batches_per_epoch)
        best, results = grid_search_over_configs(
            configs, probe_supervised, device, selection_metric=selection_metric,
            data_subset=data_subset, probe_epochs=probe_epochs, **probe_kwargs,
        )
    elif strategy == "successive_halving":
        # print_tuning_budget is called INSIDE successive_halving_search
        # (needs to see initial_epochs/reduction_factor there, not here).
        configs = sample_random_configs(grid, n_random_configs, seed=seed)
        best, results = successive_halving_search(
            configs, probe_supervised, device, initial_epochs=probe_epochs,
            selection_metric=selection_metric, data_subset=data_subset, **probe_kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: grid, random, successive_halving.")

    if csv_path:
        results_to_csv(results, csv_path)
    return best, results




#  SSL probe + tuner (run once per pretext method) 

def probe_ssl(
    cfg: dict,
    ssl_loader,
    train_feat_loader,
    val_feat_loader,
    device: torch.device,
    method: str = "simclr",
    probe_epochs: int = 10,
    num_classes: int = 251,
    use_amp: bool = True,
    selection_metric: str = "f1_macro",
    early_stop_patience: int = 0,
) -> tuple[float, BaseModel, float, int]:
    """
    One SSL trial under cfg: short pretrain (no labels) -> freeze -> extract
    features -> fit the chosen traditional classifier -> return its validation
    accuracy AND macro-F1. Mirrors probe_supervised's (val_acc, model,
    val_f1_macro, epochs_run) contract so the SAME search driver works for SSL.
    method is fixed per call (simclr or rotation). The traditional classifier
    already applies class_weight="balanced" for logreg/linear_svm (see
    self_supervised.fit_traditional_classifier) so this probe reflects the
    same imbalance handling as the final SSL run.

    ``early_stop_patience`` > 0 refits the downstream classifier and evaluates
    ``selection_metric`` after EVERY pretraining epoch, stopping as soon as it
    plateaus for that many epochs (see ``self_supervised.pretrain_simclr`` /
    ``pretrain_rotation``'s ``eval_fn`` contract). This is more expensive per
    epoch than the original single-fit-at-the-end probe (a classifier is
    fit every epoch instead of once), but bounded by ``probe_epochs`` and only
    engaged during the search — the final SSL retrain (``run_ssl_pipeline``)
    never passes ``early_stop_patience`` and is unaffected. 0 (default)
    disables it and keeps the original cheap behaviour.
    """
    from .self_supervised import (
        pretrain_simclr, pretrain_rotation, extract_features,
        fit_traditional_classifier,
    )
    from sklearn.metrics import accuracy_score, f1_score as _f1_score

    backbone = build_model(
        cfg["model_name"], num_classes=num_classes,
        dropout=cfg.get("dropout", 0.3), width_mult=cfg.get("width_mult", 1.0),
    ).to(device)

    best = {"acc": 0.0, "f1_macro": 0.0, "metric": None}

    def eval_fn() -> float:
        Xtr, ytr = extract_features(backbone, train_feat_loader, device)
        Xva, yva = extract_features(backbone, val_feat_loader, device)
        clf = fit_traditional_classifier(Xtr, ytr, classifier=cfg.get("classifier", "logreg"))
        pred = clf.predict(Xva)
        acc = accuracy_score(yva, pred)
        f1_macro = _f1_score(yva, pred, average="macro", zero_division=0)
        current = f1_macro if selection_metric == "f1_macro" else acc
        if best["metric"] is None or current > best["metric"]:
            best.update(acc=acc, f1_macro=f1_macro, metric=current)
        return current

    use_search_eval = early_stop_patience > 0
    if method == "simclr":
        hist = pretrain_simclr(
            backbone, ssl_loader, device,
            epochs=probe_epochs, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
            temperature=cfg.get("temperature", 0.5),
            projection_dim=cfg.get("projection_dim", 128), use_amp=use_amp,
            early_stop_patience=early_stop_patience, eval_fn=eval_fn if use_search_eval else None,
        )
    elif method == "rotation":
        hist = pretrain_rotation(
            backbone, ssl_loader, device,
            epochs=probe_epochs, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
            use_amp=use_amp,
            early_stop_patience=early_stop_patience, eval_fn=eval_fn if use_search_eval else None,
        )
    else:
        raise ValueError(f"Unknown SSL method '{method}'. Choose: simclr, rotation.")

    epochs_run = len(hist.get("ssl_loss", []))
    if use_search_eval and best["metric"] is not None:
        val_acc, val_f1_macro = best["acc"], best["f1_macro"]
    else:
        # Original cheap behaviour: fit the classifier once, at the very end.
        Xtr, ytr = extract_features(backbone, train_feat_loader, device)
        Xva, yva = extract_features(backbone, val_feat_loader, device)
        clf = fit_traditional_classifier(Xtr, ytr, classifier=cfg.get("classifier", "logreg"))
        val_pred = clf.predict(Xva)
        val_acc = accuracy_score(yva, val_pred)
        val_f1_macro = _f1_score(yva, val_pred, average="macro", zero_division=0)
    return val_acc, backbone, val_f1_macro, epochs_run


def deduplicate_ssl_configs(configs: list[dict], method: str) -> list[dict]:
    """
    For rotation, temperature is irrelevant, so the grid's temperature axis
    creates identical duplicate configs. Collapse them to one per unique config.
    SimCLR configs are returned unchanged.
    """
    if method != "rotation":
        return configs
    seen, kept = set(), []
    for cfg in configs:
        c = dict(cfg); c.pop("temperature", None)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key); kept.append(cfg)
    return kept


def tune_ssl(
    ssl_loader,
    train_feat_loader,
    val_feat_loader,
    method: str = "simclr",
    device: torch.device | None = None,
    grid: dict[str, list] | None = None,
    probe_epochs: int = 10,
    num_classes: int = 251,
    csv_path: str | None = None,
    selection_metric: str = "f1_macro",
    early_stop_patience: int = 0,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Tune ONE SSL pretext method (simclr or rotation). Returns (best, all_results).

    Run it once per method with the matching loader (SimCLR: augmented-pair
    loader; rotation: single-image loader), then compare the two winners'
    val_f1_macro (default) or val_accuracy to choose the better paradigm — the
    traditional classifier's ``class_weight="balanced"`` (self_supervised.py)
    already accounts for imbalance, but ranking by macro-F1 still avoids a
    scheme that wins on accuracy while starving the tail classes.

    ``early_stop_patience`` (config.SSL_TUNE_EARLY_STOP_PATIENCE) is forwarded
    to ``probe_ssl``; 0 disables it (original behaviour).

    device defaults to utils.get_device() (MPS on the Mac, CUDA on a CUDA PC).
    """
    device = device or get_device()
    grid = grid or default_ssl_grid()
    configs = deduplicate_ssl_configs(list(iter_grid(grid)), method)
    # No wall-clock estimate here: config.BENCHMARKED_SEC_PER_BATCH was
    # measured on the supervised probe pipeline, and SSL's per-batch cost
    # (two augmented views + NT-Xent, or 4x rotation expansion) isn't the
    # same shape -- reporting a number from the wrong pipeline would be
    # actively misleading, so this only prints the epoch-equivalent count.
    print_tuning_budget(f"grid[ssl:{method}]", len(configs), probe_epochs)

    best, results = grid_search_over_configs(
        configs, probe_ssl, device, selection_metric=selection_metric,
        ssl_loader=ssl_loader,
        train_feat_loader=train_feat_loader,
        val_feat_loader=val_feat_loader,
        method=method, probe_epochs=probe_epochs, num_classes=num_classes,
        early_stop_patience=early_stop_patience,
    )
    if csv_path:
        results_to_csv(results, csv_path)
    print(f"[tune-ssl:{method}] BEST val_acc={best.val_accuracy:.4f} "
          f"val_f1_macro={best.val_f1_macro:.4f} :: {best.config}")
    return best, results


def compare_ssl_methods(simclr_best: TrialResult, rotation_best: TrialResult,
                        selection_metric: str = "f1_macro") -> dict:
    """
    Compare the two tuned SSL winners and name the better pretext method
    (by macro-F1 default, to stay imbalance-safe like the rest of the search).
    Returns {"winner", "simclr_acc", "rotation_acc", "simclr_f1_macro", "rotation_f1_macro"}.
    """
    sm, rm = _metric_of(simclr_best, selection_metric), _metric_of(rotation_best, selection_metric)
    winner = "simclr" if sm >= rm else "rotation"
    print(f"[ssl-compare:{selection_metric}] simclr={sm:.4f} | rotation={rm:.4f} -> winner: {winner}")
    return {
        "winner": winner,
        "simclr_acc": simclr_best.val_accuracy, "rotation_acc": rotation_best.val_accuracy,
        "simclr_f1_macro": simclr_best.val_f1_macro, "rotation_f1_macro": rotation_best.val_f1_macro,
    }