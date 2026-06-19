"""
FoodNet — Hyperparameter Tuning
================================
Implements Task 5 of the exam ("tune the hyperparameters of the models to
achieve better performance"). The search optimises **validation accuracy
first**, while *respecting and recording* the constraints you care about:

    * the < 10 M parameter cap,
    * peak GPU memory (MB),
    * wall-clock time per trial (s).

Design (matched to a strong multi-GPU machine)
----------------------------------------------
  * A reproducible GRID is defined in "default_sl_grid" / "default_ssl_grid".
    Because we have ample compute we enumerate the grid exhaustively, but each
    trial is trained for only "probe_epochs" (a short, early-stopped probe) so
    the sweep stays affordable; the winning config is then trained to
    convergence by your normal training script.
  * Selection metric = validation accuracy. Ties (within "tie_tol") are broken
    by lower time then lower memory, so among equally-accurate configs we prefer
    the cheaper one — exactly the trade-off the brief asks you to document.
  * Every trial logs accuracy, params(M), peak-mem(MB) and time(s) to a tidy
    list of dicts you can dump to CSV and drop straight into the report table.

The module is deliberately framework-light: it depends only on "build_model"
and a couple of tiny callbacks, so it works for both the SL and SSL tasks.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model, BaseModel

from utils import get_device

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
    params_M: float
    peak_mem_MB: float
    time_s: float
    under_10M: bool
    extra: dict = field(default_factory=dict)   # e.g. per-class F1 if computed

    def row(self) -> dict:
        """Flatten to a single dict (config keys + metrics) for CSV/DataFrame."""
        r = {f"cfg.{k}": v for k, v in self.config.items()}
        r.update(
            val_accuracy=round(self.val_accuracy, 4),
            params_M=self.params_M,
            peak_mem_MB=round(self.peak_mem_MB, 1),
            time_s=round(self.time_s, 1),
            under_10M=self.under_10M,
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
        "model_name":      ["foodnet_v2"],        # redesigned MBConv model (proposed)
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
        "model_name":     ["foodnet_v2"],
        "lr":             [5e-4, 1e-3],
        "temperature":    [0.1, 0.5],            # NT-Xent sharpness (SimCLR only)
        "weight_decay":   [1e-4],
        "projection_dim": [128],
        "classifier":     ["logreg", "linear_svm", "knn"],
        "width_mult":     [1.0, 0.75],
    }


def iter_grid(grid: dict[str, list]) -> Iterable[dict]:
    """Yield every combination of the grid as a config dict (Cartesian product)."""
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


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
) -> tuple[float, BaseModel]:
    """
    Train ``model_name`` for a few epochs under ``cfg`` and return
    ``(val_accuracy, model)``. This is a *probe*, not full training: it ranks
    configs cheaply. The winner is later trained to convergence elsewhere.
    """
    model = build_model(
        cfg["model_name"], num_classes=num_classes,
        dropout=cfg.get("dropout", 0.3), width_mult=cfg.get("width_mult", 1.0),
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=probe_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    # AMP (fp16 + GradScaler) only helps on CUDA, so it stays gated to CUDA; but
    # key the calls off device.type so they're valid on MPS/CPU too (your friend's
    # CUDA PC gets mixed precision; your Mac runs full precision — no crash).
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)

    for epoch in range(1, probe_epochs + 1):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"probe ep {epoch}/{probe_epochs}",
                   leave=False, unit="batch")
        for images, labels in bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_on):
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

    # Validation accuracy (the selection metric).
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="probe val", leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_on):
                preds = model(images).argmax(1)
            correct += (preds.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)
    return correct / max(total, 1), model


#  Main grid-search driver 

def grid_search(
    grid: dict[str, list],
    probe_fn: Callable[..., tuple[float, BaseModel]],
    device: torch.device,
    tie_tol: float = 0.005,
    verbose: bool = True,
    **probe_kwargs,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Exhaustively evaluate ``grid`` with ``probe_fn`` and pick the best config.

    Selection rule (in priority order, matching the brief):
        1. HARD constraint: keep only configs with < 10 M params.
        2. Maximise validation accuracy.
        3. Break near-ties (within ``tie_tol`` accuracy) by lower time, then by
           lower peak memory — i.e. prefer the cheaper of two equally-good nets.

    Args:
        probe_fn : a callable like ``probe_supervised`` returning (val_acc, model).
        probe_kwargs : forwarded to ``probe_fn`` (loaders, epochs, etc.).

    Returns:
        (best_result, all_results) — ``all_results`` is ready to write to CSV
        for the report's tuning table.
    """
    configs = list(iter_grid(grid))
    return grid_search_over_configs(
        configs, probe_fn, device, tie_tol=tie_tol, verbose=verbose, **probe_kwargs
    )


def grid_search_over_configs(
    configs: list[dict],
    probe_fn: Callable[..., tuple[float, BaseModel]],
    device: torch.device,
    tie_tol: float = 0.005,
    verbose: bool = True,
    **probe_kwargs,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Core driver shared by grid_search (SL) and tune_ssl (SSL): evaluate an
    explicit list of config dicts with probe_fn and pick the best under the same
    <10M / accuracy / cheap-tie-break rule. Taking a config LIST (not a grid)
    lets the SSL path drop redundant rotation-vs-temperature duplicates first.
    """
    results: list[TrialResult] = []
    if verbose:
        print(f"[tune] {len(configs)} configurations to probe.\n")

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
        val_acc, _ = probe_fn(cfg, device=device, **probe_kwargs)
        elapsed = time.time() - t0
        peak = peak_mem_mb(device)

        res = TrialResult(
            config=cfg, val_accuracy=val_acc, params_M=info["total_params_M"],
            peak_mem_MB=peak, time_s=elapsed, under_10M=True,
        )
        results.append(res)
        if config_bar is not None:
            best_so_far = max(r.val_accuracy for r in results)
            config_bar.set_postfix(last=f"{val_acc:.3f}", best=f"{best_so_far:.3f}")
            config_bar.update(1)
        if verbose:
            print(f"[tune] ({i}/{len(configs)}) acc={val_acc:.4f} "
                  f"params={info['total_params_M']}M mem={peak:.0f}MB "
                  f"time={elapsed:.0f}s :: {cfg}")

    if config_bar is not None:
        config_bar.close()

    if not results:
        raise RuntimeError("No valid (<10M) configurations were probed.")

    # Sort: accuracy desc, then time asc, then memory asc (cheap tie-break).
    results_sorted = sorted(results, key=lambda r: (-r.val_accuracy, r.time_s, r.peak_mem_MB))
    best = results_sorted[0]
    # Among configs within tie_tol of the best accuracy, pick the cheapest.
    near = [r for r in results_sorted if best.val_accuracy - r.val_accuracy <= tie_tol]
    best = min(near, key=lambda r: (r.time_s, r.peak_mem_MB))

    if verbose:
        print("\n[tune] BEST CONFIG")
        print(f"       {best.config}")
        print(f"       val_acc={best.val_accuracy:.4f} | params={best.params_M}M | "
              f"mem={best.peak_mem_MB:.0f}MB | time={best.time_s:.0f}s")
    return best, results


#  CSV export helper 

def results_to_csv(results: list[TrialResult], path: str) -> None:
    """Write all trial rows to ``path`` — drop the file straight into the report."""
    import csv
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


#  Convenience: end-to-end SL tuning 

def tune_supervised(
    train_loader,
    val_loader,
    device: torch.device | None = None,
    grid: dict[str, list] | None = None,
    probe_epochs: int = 5,
    num_classes: int = 251,
    csv_path: str | None = None,
) -> tuple[TrialResult, list[TrialResult]]:
    """
    One-call SL hyperparameter sweep. Returns ``(best, all_results)`` and, if
    ``csv_path`` is given, also writes the full table for the report.

    ``device`` defaults to ``utils.get_device()`` when omitted, so the same call
    resolves to MPS on the Mac and CUDA on your friend's PC with no edits.
    """
    device = device or get_device()
    grid = grid or default_sl_grid()
    best, results = grid_search(
        grid, probe_supervised, device,
        train_loader=train_loader, val_loader=val_loader,
        probe_epochs=probe_epochs, num_classes=num_classes,
    )
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
) -> tuple[float, BaseModel]:
    """
    One SSL trial under cfg: short pretrain (no labels) -> freeze -> extract
    features -> fit the chosen traditional classifier -> return its validation
    accuracy. Mirrors probe_supervised's (val_acc, model) contract so the SAME
    search driver works for SSL. method is fixed per call (simclr or rotation).
    """
    from .self_supervised import (
        pretrain_simclr, pretrain_rotation, extract_features,
        fit_traditional_classifier,
    )
    from sklearn.metrics import accuracy_score

    backbone = build_model(
        cfg["model_name"], num_classes=num_classes,
        dropout=cfg.get("dropout", 0.3), width_mult=cfg.get("width_mult", 1.0),
    ).to(device)

    if method == "simclr":
        pretrain_simclr(
            backbone, ssl_loader, device,
            epochs=probe_epochs, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
            temperature=cfg.get("temperature", 0.5),
            projection_dim=cfg.get("projection_dim", 128), use_amp=use_amp,
        )
    elif method == "rotation":
        pretrain_rotation(
            backbone, ssl_loader, device,
            epochs=probe_epochs, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
            use_amp=use_amp,
        )
    else:
        raise ValueError(f"Unknown SSL method '{method}'. Choose: simclr, rotation.")

    Xtr, ytr = extract_features(backbone, train_feat_loader, device)
    Xva, yva = extract_features(backbone, val_feat_loader, device)
    clf = fit_traditional_classifier(Xtr, ytr, classifier=cfg.get("classifier", "logreg"))
    val_acc = accuracy_score(yva, clf.predict(Xva))
    return val_acc, backbone


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
) -> tuple[TrialResult, list[TrialResult]]:
    """
    Tune ONE SSL pretext method (simclr or rotation). Returns (best, all_results).

    Run it once per method with the matching loader (SimCLR: augmented-pair
    loader; rotation: single-image loader), then compare the two winners'
    val_accuracy to choose the better paradigm.

    device defaults to utils.get_device() (MPS on the Mac, CUDA on a CUDA PC).
    """
    device = device or get_device()
    grid = grid or default_ssl_grid()
    configs = deduplicate_ssl_configs(list(iter_grid(grid)), method)
    print(f"[tune-ssl:{method}] {len(configs)} configs to probe.")

    best, results = grid_search_over_configs(
        configs, probe_ssl, device,
        ssl_loader=ssl_loader,
        train_feat_loader=train_feat_loader,
        val_feat_loader=val_feat_loader,
        method=method, probe_epochs=probe_epochs, num_classes=num_classes,
    )
    if csv_path:
        results_to_csv(results, csv_path)
    print(f"[tune-ssl:{method}] BEST val_acc={best.val_accuracy:.4f} :: {best.config}")
    return best, results


def compare_ssl_methods(simclr_best: TrialResult, rotation_best: TrialResult) -> dict:
    """
    Compare the two tuned SSL winners and name the better pretext method.
    Returns {"winner": "simclr"|"rotation", "simclr_acc":.., "rotation_acc":..}.
    """
    sa, ra = simclr_best.val_accuracy, rotation_best.val_accuracy
    winner = "simclr" if sa >= ra else "rotation"
    print(f"[ssl-compare] simclr={sa:.4f} | rotation={ra:.4f} -> winner: {winner}")
    return {"winner": winner, "simclr_acc": sa, "rotation_acc": ra}