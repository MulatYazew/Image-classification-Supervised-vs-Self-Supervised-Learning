"""
FoodNet Supervised Trainer
===========================
Training loop for the SUPERVISED (SL) task:
  - configurable loss (CE / weighted-CE / focal, via codes.loss_function.build_criterion)
  - AdamW + CosineAnnealingLR
  - gradient clipping (stabilises from-scratch training)
  - early stopping on validation loss
  - best-model checkpointing
  - MPS / CUDA / CPU compatible

It also exposes a ``grid_search`` helper so the report's hyper-parameter tuning
section can sweep a small grid and keep the best configuration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

# Progress bars. Falls back to a no-op shim if tqdm isn't installed, so the
# trainer never hard-depends on it.
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):   # type: ignore
        return iterable if iterable is not None else iter(())


class Trainer:
    """
    Manages the supervised training loop for a custom Food-251 model.

    Args:
        model         : a codes.model BaseModel instance.
        device        : torch.device.
        criterion     : loss module (from codes.loss_function.build_criterion). If None,
                        falls back to plain CrossEntropyLoss with class_weights.
        learning_rate : initial LR.
        weight_decay  : AdamW weight decay.
        class_weights : optional 1-D tensor used only for the fallback criterion.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        if criterion is None:
            w = class_weights.to(device) if class_weights is not None else None
            criterion = nn.CrossEntropyLoss(weight=w)
        self.criterion = criterion.to(device)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Mixed precision: ~2x faster and ~half the memory on CUDA.
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
        }
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    #  Single epoch 

    def run_epoch(self, loader, training: bool, desc: str | None = None) -> tuple[float, float]:
        self.model.train(training)
        total_loss, correct, n = 0.0, 0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        bar = tqdm(loader, desc=desc or ("train" if training else "val"),
                   leave=False, unit="batch")
        with ctx:
            for images, labels in bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                if training:
                    # GradScaler path (CUDA fp16): scale → unscale → clip → step.
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                n += labels.size(0)
                if _HAS_TQDM:
                    bar.set_postfix(loss=f"{total_loss / max(n, 1):.3f}",
                                    acc=f"{correct / max(n, 1):.3f}")
        return total_loss / max(n, 1), correct / max(n, 1)

    #  Full loop 

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 15,
        model_save_dir: str = "models",
        run_name: str = "supervised",
        warmup_frozen_epochs: int = 0,
        lr_warmup_epochs: int = 5,
    ) -> dict[str, list[float]]:
        """
        Train with linear LR warmup → cosine annealing, and early stopping.

        ``lr_warmup_epochs`` ramps the LR from 0.1× up to the initial value
        over the first N epochs, then cosine-decays for the remainder. This is
        critical for stable from-scratch training: cold-starting at full LR
        causes chaotic early updates that push the model into poor basins.

        ``warmup_frozen_epochs`` > 0 trains only the head for that many epochs
        (backbone frozen) before unfreezing — useful when starting from
        SSL-pretrained backbone weights.
        """
        save_dir = Path(model_save_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Build scheduler: linear warm-up → cosine decay.
        # SequentialLR chains two schedulers at ``milestones=[lr_warmup_epochs]``.
        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, lr_warmup_epochs),
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, num_epochs - lr_warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[lr_warmup_epochs],
        )

        # Optional frozen warm-up (e.g. linear-probe phase on SSL weights).
        if warmup_frozen_epochs > 0 and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            if warmup_frozen_epochs > 0 and epoch == warmup_frozen_epochs + 1 \
               and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
                print(f"  → backbone unfrozen at epoch {epoch}")

            train_loss, train_acc = self.run_epoch(
                train_loader, training=True, desc=f"epoch {epoch}/{num_epochs} train")
            val_loss, val_acc = self.run_epoch(
                val_loader, training=False, desc=f"epoch {epoch}/{num_epochs} val")
            scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            for k, v in zip(
                ("train_loss", "val_loss", "train_acc", "val_acc", "lr"),
                (train_loss, val_loss, train_acc, val_acc, lr),
            ):
                self.history[k].append(v)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f} acc {val_acc:.4f} | LR {lr:.2e}"
            )

            if val_loss < self._best_val_loss - 1e-4:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                ckpt = save_dir / "best_model.pth"
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ Best model saved → {ckpt}")
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min.")
        return self.history

    #  Evaluation / inference 

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """
        Run one no-grad pass over ``loader`` and return average loss and accuracy.
        Convenience wrapper around ``run_epoch`` for a quick val/test score
        without going through the full training loop.
        """
        loss, acc = self.run_epoch(loader, training=False)
        return {"loss": loss, "accuracy": acc}

    @torch.no_grad()
    def predict(self, loader, return_probs: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict over ``loader``. Returns ``(predictions, labels)`` as numpy arrays,
        or ``(probabilities, labels)`` when ``return_probs=True`` (softmax over the
        251 classes). Labels are returned too so the output lines up for metrics.
        """
        self.model.eval()
        preds_or_probs, labels_all = [], []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
            if return_probs:
                out = torch.softmax(logits.float(), dim=1)
                preds_or_probs.append(out.cpu().numpy())
            else:
                preds_or_probs.append(logits.argmax(1).cpu().numpy())
            labels_all.append(labels.numpy())
        return np.concatenate(preds_or_probs), np.concatenate(labels_all)

    #  Checkpoint save / resume 

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save a FULL checkpoint (model + optimizer + scaler + history + counters),
        so training can be resumed exactly. ``train`` saves only model weights for
        the best epoch; use this when you want to pause and continue later.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": self.history,
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str | Path, weights_only: bool = False) -> None:
        """
        Load a checkpoint. With ``weights_only=True`` only the model weights are
        restored (e.g. to load a best_model.pth for evaluation). Otherwise the
        optimizer, scaler, history and early-stopping state are restored too, so
        ``train`` can be called again to resume.
        """
        ckpt = torch.load(path, map_location=self.device)
        if weights_only or "model" not in ckpt:
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state)
            print(f"Model weights loaded ← {path}")
            return
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.history = ckpt.get("history", self.history)
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self._patience_counter = ckpt.get("patience_counter", 0)
        print(f"Full state restored ← {path} (resume-ready)")



#  Learning-rate finder 

@torch.no_grad()
def _set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def lr_finder(
    model: nn.Module,
    train_loader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iters: int = 100,
    weight_decay: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """
    Leslie Smith style LR range test. Trains for ``num_iters`` mini-batches while
    exponentially increasing the learning rate from ``start_lr`` to ``end_lr``,
    recording the loss at each step. Plot loss vs lr (log scale) and pick a
    learning rate roughly one order of magnitude below where the loss is steepest
    / just before it explodes — a fast, principled alternative to grid-searching
    the LR when you are not running a full hyperparameter sweep.

    Returns ``(lrs, losses)``. Does NOT mutate the passed model's final weights in
    a meaningful way for training (it runs a short transient), but for safety run
    it on a fresh model or rebuild afterwards.

    Example:
        lrs, losses = lr_finder(M.build_model("foodnet"), train_loader, device)
        import matplotlib.pyplot as plt
        plt.plot(lrs, losses); plt.xscale("log"); plt.xlabel("lr"); plt.ylabel("loss")
    """
    model = model.to(device).train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    # Geometric LR schedule across iterations.
    mult = (end_lr / start_lr) ** (1.0 / max(num_iters - 1, 1))
    lr = start_lr
    lrs, losses = [], []
    best = float("inf")
    it = 0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    done = False
    while not done:
        for images, labels in train_loader:
            if it >= num_iters:
                done = True
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item()
            lrs.append(lr)
            losses.append(loss_val)
            best = min(best, loss_val)
            # Stop early if the loss diverges badly (4x the best seen).
            if loss_val > 4 * best:
                done = True
                break
            lr *= mult
            it += 1

    print(f"[lr_finder] ran {len(lrs)} iters | lr {lrs[0]:.1e} → {lrs[-1]:.1e}")
    return lrs, losses