"""
FoodNet Supervised Trainer
===========================
Training loop for the SUPERVISED (SL) task:
  - configurable loss (CE / weighted-CE / focal, via codes.loss.build_criterion)
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

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """
    Manages the supervised training loop for a custom Food-251 model.

    Args:
        model         : a codes.model BaseModel instance.
        device        : torch.device.
        criterion     : loss module (from codes.loss.build_criterion). If None,
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

        # Mixed precision: ~2x faster and ~half the memory on CUDA. GradScaler is
        # only meaningful for fp16 on CUDA; on MPS/CPU we run autocast without
        # scaling (or disabled), so behaviour is unchanged there.
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
        }
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    #  Single epoch 

    def run_epoch(self, loader, training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss, correct, n = 0.0, 0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
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
        return total_loss / max(n, 1), correct / max(n, 1)

    #  Full loop 

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 60,
        patience: int = 8,
        model_save_dir: str = "models",
        run_name: str = "supervised",
        warmup_frozen_epochs: int = 0,
    ) -> dict[str, list[float]]:
        """
        Train with cosine LR annealing and early stopping.

        ``warmup_frozen_epochs`` > 0 trains only the head for that many epochs
        (backbone frozen) before unfreezing — useful when starting from
        SSL-pretrained backbone weights.
        """
        save_dir = Path(model_save_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # Optional frozen warm-up (e.g. linear-probe phase on SSL weights).
        if warmup_frozen_epochs > 0 and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            if warmup_frozen_epochs > 0 and epoch == warmup_frozen_epochs + 1 \
               and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
                print(f"  → backbone unfrozen at epoch {epoch}")

            train_loss, train_acc = self.run_epoch(train_loader, training=True)
            val_loss, val_acc = self.run_epoch(val_loader, training=False)
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

