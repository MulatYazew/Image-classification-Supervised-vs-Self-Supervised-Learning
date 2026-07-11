"""
FoodNet — Self-Supervised Learning (SSL) Task
==============================================
The exam's second paradigm: pretrain the same custom backbone (codes.model)
without labels, extract features, and classify them with a traditional
classifier.

Pipeline: (1) pretrain the backbone via a pretext task — "simclr" (contrastive
NT-Xent on two augmented views, default) or "rotation" (predict a 4-way
{0,90,180,270} rotation); (2) freeze the backbone and extract penultimate
features (forward_features); (3) fit a traditional classifier (logreg/linear
SVM/kNN) on train features, score on val (=test) features. SL and SSL are
then compared on identical splits/metrics, so the only variable is whether
the backbone saw labels.

Efficiency: AMP autocast + GradScaler on CUDA (~2x faster, half memory); MPS/
CPU run full precision. SimCLR LR scales linearly with batch size (SimCLR
rule — contrastive loss is batch-size sensitive). The traditional classifier
defaults to fast SAGA logistic regression (one-vs-rest LinearSVC over 251
classes on high-dim features is slow).
"""

from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .loss_function import NTXentLoss
from .model import BaseModel
from .utils import get_device, LocalEarlyStopper, make_amp_context


# Projection head (SimCLR)

class ProjectionHead(nn.Module):
    """2-layer MLP mapping backbone features to contrastive space. SimCLR
    contrasts in this projection space, not the feature space — the head is
    used only during pretraining and discarded afterwards, so the downstream
    classifier sees the richer pre-projection embedding (forward_features)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),         # stabilises contrastive training
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# SimCLR contrastive pretraining

def pretrain_simclr(backbone: BaseModel, ssl_loader, device: torch.device,epochs: int = 100,lr: float = 1e-3,
                    weight_decay: float = 1e-4,temperature: float = 0.5, projection_dim: int = 128,
                    batch_size_ref: int = 256, use_amp: bool = True, save_path: str | Path | None = None,
                    early_stop_patience: int = 0,
                    eval_fn: Callable[[], float] | None = None) -> dict[str, list[float]]:
    """Contrastively pretrain backbone.forward_features with NT-Xent.
    ssl_loader yields (view1, view2) pairs with no labels (two independent
    augmentations of the same image). The projection head is discarded after
    pretraining; only backbone weights are kept.

    lr is the base LR for a 256-sample batch; the effective LR scales
    linearly with the real batch size (SimCLR rule — contrastive loss quality
    depends on the number of negatives, ~batch size). early_stop_patience > 0
    stops once eval_fn hasn't improved for that many epochs (search-loop
    early stop, see hyperparameter_tuning.probe_ssl); eval_fn is a zero-arg
    "higher is better" callable, only invoked when early_stop_patience > 0 so
    a normal run_ssl_pipeline call pays zero overhead for it.
    """
    backbone = backbone.to(device)
    proj = ProjectionHead(backbone.feature_dim, hidden_dim=512, out_dim=projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)

    try:
        sample_view1, _ = next(iter(ssl_loader))
        real_bs = sample_view1.size(0)
    except Exception:
        real_bs = batch_size_ref
    scaled_lr = lr * real_bs / batch_size_ref
    print(f"[SimCLR] base_lr={lr:.2e}  batch={real_bs}  → scaled_lr={scaled_lr:.2e}")

    optimizer = AdamW(list(backbone.parameters()) + list(proj.parameters()),
                      lr=scaled_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # autocast also engages on MPS (measured speedup on this M4); GradScaler stays CUDA-only
    amp_on, amp_dtype, scaler = make_amp_context(use_amp, device)

    history: dict[str, list[float]] = {"ssl_loss": []}
    stopper = LocalEarlyStopper(early_stop_patience) if early_stop_patience > 0 and eval_fn is not None else None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        backbone.train(); proj.train()
        running, n = 0.0, 0
        for view1, view2 in ssl_loader:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                z1 = proj(backbone.forward_features(view1))   # project both views...
                z2 = proj(backbone.forward_features(view2))
                loss = criterion(z1, z2)                       # ...and contrast them
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(proj.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * view1.size(0)
            n += view1.size(0)
        scheduler.step()
        epoch_loss = running / max(n, 1)
        history["ssl_loss"].append(epoch_loss)
        print(f"[SimCLR] Epoch {epoch:3d}/{epochs} | NT-Xent {epoch_loss:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        if stopper is not None and eval_fn is not None:
            metric = eval_fn()
            history.setdefault("probe_metric", []).append(metric)
            if stopper.update(metric):
                print(f"[SimCLR] Search early-stop at epoch {epoch}/{epochs} "
                      f"(no probe-metric improvement for {early_stop_patience} epochs).")
                break

    print(f"\n[SimCLR] Pretraining finished in {(time.time() - t0)/60:.1f} min "
          f"({len(history['ssl_loss'])} epoch(s) run).")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), save_path)
        print(f"[SimCLR] Backbone weights saved → {save_path}")
    return history


# Rotation-prediction pretraining (alternative pretext)

def pretrain_rotation(
    backbone: BaseModel,
    feature_loader,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    max_rot_batch: int = 256,
    save_path: str | Path | None = None,
    early_stop_patience: int = 0,
    eval_fn: Callable[[], float] | None = None,
) -> dict[str, list[float]]:
    """Rotation-prediction pretext (Gidaris et al., 2018): each image is
    rotated by one of {0,90,180,270} degrees and the backbone + a 4-way head
    predicts which. Labels in feature_loader are ignored; the rotation index
    is the self-supervised target. Source batch is capped at max_rot_batch
    before the 4x rotation expansion to avoid OOM.

    early_stop_patience/eval_fn follow the same search-loop early-stop
    contract as pretrain_simclr (both default disabled).
    """
    backbone = backbone.to(device)
    rot_head = nn.Linear(backbone.feature_dim, 4).to(device)   # 4-way rotation classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(list(backbone.parameters()) + list(rot_head.parameters()),
                      lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    amp_on, amp_dtype, scaler = make_amp_context(use_amp, device)

    history: dict[str, list[float]] = {"ssl_loss": [], "ssl_acc": []}
    stopper = LocalEarlyStopper(early_stop_patience) if early_stop_patience > 0 and eval_fn is not None else None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        backbone.train(); rot_head.train()
        running, correct, n = 0.0, 0, 0
        for images, _ in feature_loader:                       # labels ignored
            images = images.to(device, non_blocking=True)
            cap = max(1, max_rot_batch // 4)   # keep the 4x expansion within the memory cap
            if images.size(0) > cap:
                images = images[:cap]
            batch, targets = [], []
            for k in range(4):
                batch.append(torch.rot90(images, k, dims=(2, 3)))
                targets.append(torch.full((images.size(0),), k, dtype=torch.long, device=device))
            x = torch.cat(batch, dim=0)
            y = torch.cat(targets, dim=0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                logits = rot_head(backbone.forward_features(x))
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(rot_head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        scheduler.step()
        history["ssl_loss"].append(running / max(n, 1))
        history["ssl_acc"].append(correct / max(n, 1))
        print(f"[Rotation] Epoch {epoch:3d}/{epochs} | loss {history['ssl_loss'][-1]:.4f} | "
              f"rot-acc {history['ssl_acc'][-1]:.4f}")

        if stopper is not None and eval_fn is not None:
            metric = eval_fn()
            history.setdefault("probe_metric", []).append(metric)
            if stopper.update(metric):
                print(f"[Rotation] Search early-stop at epoch {epoch}/{epochs} "
                      f"(no probe-metric improvement for {early_stop_patience} epochs).")
                break

    print(f"\n[Rotation] Pretraining finished in {(time.time() - t0)/60:.1f} min "
          f"({len(history['ssl_loss'])} epoch(s) run).")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), save_path)
        print(f"[Rotation] Backbone weights saved → {save_path}")
    return history


# Feature extraction

@torch.no_grad()
def extract_features(
    backbone: BaseModel,
    loader,
    device: torch.device,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen backbone over loader (yielding (image, label)) and
    return (features, labels) — features are forward_features' penultimate
    embeddings. L2-normalising puts every embedding on the unit sphere, which
    is what cosine-based/linear classifiers expect and what SimCLR trained
    the space to be."""
    backbone = backbone.to(device).eval()
    feats_list, labels_list = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        f = backbone.forward_features(images)
        if l2_normalize:
            f = F.normalize(f, dim=1)
        feats_list.append(f.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(feats_list), np.concatenate(labels_list)


# Traditional classifier on frozen features

def fit_traditional_classifier(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    classifier: str = "logreg",
    seed: int = 42,
):
    """Fit a traditional classifier (logreg: multinomial SAGA | linear_svm |
    knn) on SSL features. Returns a fitted sklearn Pipeline (StandardScaler
    -> estimator); standardising first speeds up and stabilises the linear
    solvers on 251-class, high-dimensional features.

    The pretext task itself needs no imbalance correction (NT-Xent ignores
    labels), but this read-out classifier is fit on labels and inherits the
    same ~19:1 imbalance as the supervised task, so logreg/linear_svm get
    class_weight="balanced". KNN has no class_weight — weights="distance" is
    its closest lever, avoiding a dense majority-class neighbourhood
    dominating a tied vote.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if classifier == "logreg":
        # SAGA handles 251-class softmax (multinomial) far better than
        # one-vs-rest; capped iterations keep it fast. Recent scikit-learn
        # defaults saga to multinomial automatically (no multi_class arg needed).
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            solver="saga", max_iter=200, C=1.0, n_jobs=-1, random_state=seed,
            class_weight="balanced",
        )
    elif classifier == "linear_svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=seed, class_weight="balanced")
    elif classifier == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=20, weights="distance", n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier '{classifier}'. Choose: logreg, linear_svm, knn.")

    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(train_feats, train_labels)
    return pipe


def run_ssl_pipeline(
    backbone: BaseModel,
    ssl_loader,
    train_feat_loader,
    val_feat_loader,
    device: torch.device | None = None,
    method: str = "simclr",
    classifier: str = "logreg",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    projection_dim: int = 128,
    use_amp: bool = True,
    save_path: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """End-to-end SSL task: pretrain -> freeze -> extract features -> fit &
    score a traditional classifier on train/val(=test) splits. device
    defaults to utils.get_device(). seed forwards to
    fit_traditional_classifier (config.SEED).

    Returns a dict with the SSL training history, fitted classifier,
    extracted feature arrays, and train/val predictions+labels — hand val_*
    to evaluate.Evaluator for the same metrics used in the SL task (apples-
    to-apples comparison). train_predictions is also returned so the
    downstream classifier's own train-vs-val accuracy/macro-F1 can be
    measured (whether the logreg/linear_svm/knn head over/underfits the
    frozen features — a question the NT-Xent pretext loss says nothing about).
    """
    device = device or get_device()
    if method == "simclr":
        hist = pretrain_simclr(backbone, ssl_loader, device, epochs=epochs, lr=lr,
                               weight_decay=weight_decay, temperature=temperature,
                               projection_dim=projection_dim, use_amp=use_amp,
                               save_path=save_path)
    elif method == "rotation":
        hist = pretrain_rotation(backbone, ssl_loader, device, epochs=epochs, lr=lr,
                                 weight_decay=weight_decay, use_amp=use_amp,
                                 save_path=save_path)
    else:
        raise ValueError(f"Unknown SSL method '{method}'. Choose: simclr, rotation.")

    print("\n[SSL] Extracting frozen-backbone features …")
    Xtr, ytr = extract_features(backbone, train_feat_loader, device)
    Xva, yva = extract_features(backbone, val_feat_loader, device)
    print(f"[SSL] Train features {Xtr.shape} | Val features {Xva.shape}")

    print(f"[SSL] Fitting traditional classifier: {classifier}")
    clf = fit_traditional_classifier(Xtr, ytr, classifier=classifier, seed=seed)
    train_pred = clf.predict(Xtr)
    val_pred = clf.predict(Xva)

    return {
        "ssl_history": hist,
        "classifier": clf,
        "train_features": Xtr,
        "train_labels": ytr,
        "val_features": Xva,
        "val_labels": yva,
        "train_predictions": train_pred,
        "val_predictions": val_pred,
    }
