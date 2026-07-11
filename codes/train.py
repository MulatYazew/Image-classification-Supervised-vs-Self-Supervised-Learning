"""
FoodNet Supervised Trainer
===========================
Training loop for the supervised (SL) task: configurable loss (CE/weighted-CE
/focal via loss_function.build_criterion), AdamW + CosineAnnealingLR,
gradient clipping, early stopping on validation loss, best-model
checkpointing, MPS/CUDA/CPU compatible.

Also exposes lr_finder and log_run helpers for the report's tuning and
ablation sections.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from .utils import make_amp_context

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:  # pragma: no cover
    HAS_TQDM = False

    def tqdm(iterable=None, *args, **kwargs):   # type: ignore
        return iterable if iterable is not None else iter(())


class Trainer:
    """Supervised training loop for a custom Food-251 model.

    mix_method ("none"|"mixup"|"cutmix", config.MIX_METHOD) is CutMix or
    MixUp sample-mixing regularisation — CutMix tends to help more on
    fine-grained, texture-heavy classes since it preserves local texture
    patches instead of globally blending pixels. class_weights is only used
    when criterion is None (fallback plain CrossEntropyLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        use_amp: bool = True,
        mix_method: str = "none",
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        if criterion is None:
            w = class_weights.to(device) if class_weights is not None else None
            criterion = nn.CrossEntropyLoss(weight=w)
        self.criterion = criterion.to(device)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Autocast also engages on MPS (measured speedup on this M4); GradScaler
        # stays CUDA-only since MPS doesn't need the same overflow-scaling machinery
        self.use_amp, self.amp_dtype, self.scaler = make_amp_context(use_amp, device)

        # mixup_alpha > 0 with default mix_method="none" still enables MixUp,
        # so existing call sites that only pass mixup_alpha keep working
        if mix_method == "none" and mixup_alpha > 0.0:
            mix_method = "mixup"
        if mix_method not in ("none", "mixup", "cutmix"):
            raise ValueError(f"Unknown mix_method '{mix_method}'. Choose: none, mixup, cutmix.")
        self.mix_method = mix_method
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
        }
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self.scheduler: LRScheduler | None = None
        self._epoch = 0

    # Single epoch

    def mixup_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply MixUp to a batch. Returns (mixed_images, y_a, y_b, lam)."""
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[idx]
        return mixed, labels, labels[idx], lam

    def cutmix_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply CutMix: paste a random box from a shuffled copy of the batch
        into each image, mixing labels by the actual (not sampled) box area.
        Same (mixed_images, y_a, y_b, lam) contract as mixup_batch."""
        lam_sampled = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        h, w = images.shape[2], images.shape[3]
        cut_ratio = (1.0 - lam_sampled) ** 0.5
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cy, cx = int(np.random.randint(h)), int(np.random.randint(w))
        y1, y2 = int(np.clip(cy - cut_h // 2, 0, h)), int(np.clip(cy + cut_h // 2, 0, h))
        x1, x2 = int(np.clip(cx - cut_w // 2, 0, w)), int(np.clip(cx + cut_w // 2, 0, w))

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))   # actual pasted area, may differ from lam_sampled
        return mixed, labels, labels[idx], lam

    def run_epoch(self, loader, training: bool, desc: str | None = None) -> tuple[float, float]:
        self.model.train(training)
        total_loss, correct, n = 0.0, 0, 0
        use_mix = training and self.mix_method != "none"
        ctx = torch.enable_grad() if training else torch.no_grad()
        bar = tqdm(loader, desc=desc or ("train" if training else "val"),
                   leave=False, unit="batch")
        with ctx:
            for images, labels in bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                if use_mix:
                    mix_fn = self.mixup_batch if self.mix_method == "mixup" else self.cutmix_batch
                    images, y_a, y_b, lam = mix_fn(images, labels)
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(images)
                        loss = lam * self.criterion(outputs, y_a) + \
                               (1.0 - lam) * self.criterion(outputs, y_b)
                    # accuracy: credit the dominant label when lam >= 0.5
                    pred = outputs.argmax(1)
                    hits = (lam * (pred == y_a).float() +
                            (1.0 - lam) * (pred == y_b).float()).sum().item()
                else:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    hits = (outputs.argmax(1) == labels).sum().item()

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                correct += hits
                n += labels.size(0)
                if HAS_TQDM:
                    bar.set_postfix(loss=f"{total_loss / max(n, 1):.3f}",
                                    acc=f"{correct / max(n, 1):.3f}")
        return total_loss / max(n, 1), correct / max(n, 1)

    # Full loop

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 60,
        patience: int = 8,
        model_save_dir: str = "models",
        run_name: str = "supervised",
        warmup_frozen_epochs: int = 0,
        warmup_epochs: int = 5,
        log_per_class_every: int = 0,
        class_names: list[str] | None = None,
        tail_classes: set[int] | None = None,
        results_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train with linear LR warmup -> cosine annealing and early stopping.

        warmup_epochs ramps LR from 0.1x base_lr to base_lr over the first N
        epochs before cosine-annealing to eta_min, preventing a large initial
        LR from destabilising BN statistics — especially harmful for deep
        from-scratch nets.

        warmup_frozen_epochs > 0 trains only the head for that many epochs
        (backbone frozen) before unfreezing — useful when starting from
        SSL-pretrained backbone weights.

        log_per_class_every > 0 runs a full per-class val F1 report
        (evaluate.Evaluator) every N epochs and at the end, printing a
        tail-vs-head F1 summary (and writing the per-class table to
        results_dir if given) — the only way to see whether the ~34-image
        tail classes are actually improving, since aggregate val_acc/loss
        average over all 251 classes. 0 disables it.

        resume_from restores model/optimizer/scheduler/scaler/history from a
        checkpoint (last_checkpoint.pth is written every epoch) and continues
        right after the saved epoch, with num_epochs/warmup_epochs unchanged
        so the cosine schedule still matches.
        """
        save_dir = Path(model_save_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        warmup_epochs = max(1, warmup_epochs)
        cosine_epochs = max(1, num_epochs - warmup_epochs)
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs, eta_min=1e-6)
        scheduler = SequentialLR(self.optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_epochs])
        self.scheduler = scheduler

        if warmup_frozen_epochs > 0 and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        start_epoch = 1
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self._epoch + 1
            print(f"Resuming training at epoch {start_epoch}/{num_epochs}")
            # match the backbone-freeze state an uninterrupted run would have reached
            if warmup_frozen_epochs > 0 and start_epoch > warmup_frozen_epochs \
               and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

        t0 = time.time()
        for epoch in range(start_epoch, num_epochs + 1):
            self._epoch = epoch
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
                strict=True,
            ):
                self.history[k].append(v)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f} acc {val_acc:.4f} | LR {lr:.2e}"
            )

            if log_per_class_every > 0 and (epoch % log_per_class_every == 0 or epoch == num_epochs):
                from .evaluate import Evaluator   # local import: avoids a hard sklearn/seaborn dep for other callers
                num_classes = getattr(self.model, "num_classes", None)
                preds, labels_arr = self.predict(val_loader)
                evaluator = Evaluator(num_classes=num_classes or int(labels_arr.max()) + 1,
                                      class_names=class_names)
                per_class_df = evaluator.per_class_metrics(labels_arr, preds)
                tail_head = evaluator.head_vs_tail_summary(per_class_df, tail_classes=tail_classes)
                print(
                    f"  [per-class] tail F1 {tail_head['tail_f1_mean']:.3f} "
                    f"(n={tail_head['n_tail_classes']}) | "
                    f"head F1 {tail_head['head_f1_mean']:.3f} (n={tail_head['n_head_classes']})"
                )
                if results_dir is not None:
                    out_path = Path(results_dir) / f"per_class_val_{run_name}_epoch{epoch}.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    per_class_df.to_csv(out_path, index=False)

            if val_loss < self._best_val_loss - 1e-4:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                ckpt = save_dir / "best_model.pth"
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ Best model saved → {ckpt}")
            else:
                self._patience_counter += 1

            # resumable snapshot overwritten every epoch: rerun with
            # train(..., resume_from=save_dir/"last_checkpoint.pth") after a crash
            self.save_checkpoint(save_dir / "last_checkpoint.pth")

            if self._patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min.")
        return self.history

    # Evaluation / inference

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """One no-grad pass over loader; returns {"loss", "accuracy"} —
        a quick val/test score without the full training loop."""
        loss, acc = self.run_epoch(loader, training=False)
        return {"loss": loss, "accuracy": acc}

    @torch.no_grad()
    def predict(self, loader, return_probs: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict over loader. Returns (predictions, labels), or
        (probabilities, labels) when return_probs=True (softmax over 251 classes)."""
        self.model.eval()
        preds_or_probs, labels_all = [], []
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(images)
            if return_probs:
                out = torch.softmax(logits.float(), dim=1)
                preds_or_probs.append(out.cpu().numpy())
            else:
                preds_or_probs.append(logits.argmax(1).cpu().numpy())
            labels_all.append(labels.numpy())
        return np.concatenate(preds_or_probs), np.concatenate(labels_all)

    # Checkpoint save / resume

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a full checkpoint (model/optimizer/scaler/scheduler/epoch/
        history/counters) for exact resume. train() saves best_model.pth
        (weights only for the best epoch) separately; this is what
        resume_from= reads back in."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": self._epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "history": self.history,
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
        }, path)
        print(f"Checkpoint saved → {path} (epoch {self._epoch})")

    def load_checkpoint(self, path: str | Path, weights_only: bool = False) -> None:
        """Load a checkpoint. weights_only=True restores only model weights
        (e.g. loading best_model.pth for evaluation); otherwise optimizer,
        scaler, scheduler, epoch, history and early-stopping state are also
        restored, so train(..., resume_from=path) continues after the saved epoch."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if weights_only or "model" not in ckpt:
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state)
            print(f"Model weights loaded ← {path}")
            return
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.history = ckpt.get("history", self.history)
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self._patience_counter = ckpt.get("patience_counter", 0)
        self._epoch = ckpt.get("epoch", 0)
        print(f"Full state restored ← {path} (epoch {self._epoch}, resume-ready)")


# Learning-rate finder

@torch.no_grad()
def _set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def lr_finder(
    model: nn.Module,
    train_loader,
    device: torch.device,
    criterion: nn.Module | None = None,
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iters: int = 100,
    weight_decay: float = 1e-4,
) -> tuple[list[float], list[float]]:
    """Leslie Smith style LR range test: trains num_iters mini-batches while
    exponentially raising the LR from start_lr to end_lr, recording loss at
    each step. Plot loss vs. lr (log scale) and pick a value about one order
    of magnitude below where loss is steepest / about to explode — a fast
    alternative to a full LR sweep. Returns (lrs, losses); runs a short
    transient so use a fresh model or rebuild afterwards.

    Example:
        lrs, losses = lr_finder(build_model("foodnet46"), train_loader, device)
        plt.plot(lrs, losses); plt.xscale("log")
    """
    model = model.to(device).train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    mult = (end_lr / start_lr) ** (1.0 / max(num_iters - 1, 1))   # geometric LR step
    lr = start_lr
    lrs, losses = [], []
    best = float("inf")
    it = 0

    use_amp, amp_dtype, scaler = make_amp_context(True, device)

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
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item()
            lrs.append(lr)
            losses.append(loss_val)
            best = min(best, loss_val)
            if loss_val > 4 * best:   # stop early once loss diverges badly
                done = True
                break
            lr *= mult
            it += 1

    print(f"[lr_finder] ran {len(lrs)} iters | lr {lrs[0]:.1e} → {lrs[-1]:.1e}")
    return lrs, losses


# Experiment logging (augmentation ablation / final-run tracking)

def log_run(run_name: str, config: dict, metrics: dict, csv_path: str | Path) -> None:
    """Append one full training run's config + final metrics as a CSV row —
    e.g. augmentation policy vs. final val accuracy/macro-F1 for the report's
    ablation table. Distinct from hyperparameter_tuning.results_to_csv, which
    logs cheap HPO probes; this logs full/confirmation runs."""
    import csv as _csv
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run_name": run_name, **config, **metrics}
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[log_run] appended '{run_name}' → {csv_path}")
