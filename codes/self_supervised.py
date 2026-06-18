"""
FoodNet — Self-Supervised Learning (SSL) Task
==============================================
Implements the SECOND paradigm required by the exam:

    "compare the performance with CNNs trained in Self-Supervised Learning
     (even on the same dataset ignoring the labels), extracting the features
     and classifying them with a traditional classifier."

Pipeline
--------
    1. Pretrain the SAME custom backbone (codes.model) WITHOUT labels via a
       pretext task:
         * "simclr"   — contrastive NT-Xent on two augmented views (default).
         * "rotation" — predict the 4-way rotation {0,90,180,270} of an image.
    2. Freeze the backbone and extract penultimate features (forward_features).
    3. Fit a TRADITIONAL classifier (logistic regression / linear SVM / kNN) on
       the train features and evaluate on the validation (= test) features.

The SL and SSL results are then compared in the report on identical splits and
metrics, so the only thing that changes between them is *how the backbone was
trained* (with vs without labels).

Efficiency notes
-------------------------------------------------------------------------
  * Mixed-precision (AMP) autocast + GradScaler — ~2x faster, half the memory,
    on any modern GPU. Enabled automatically when CUDA is available.
  * SimCLR LR is scaled linearly with batch size (lr = base_lr * B / 256), the
    standard SimCLR rule — contrastive learning is very batch-size sensitive.
  * The traditional classifier defaults to a fast SAGA logistic regression;
    a one-vs-rest LinearSVC over 251 classes on high-dim features is slow, so
    we standardise features and cap iterations sensibly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss_function import NTXentLoss


#  Projection head (SimCLR) 

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head mapping backbone features → contrastive space.

    SimCLR contrasts in the projection space, not the feature space: the head
    is used ONLY during pretraining and discarded afterwards, so the downstream
    classifier sees the richer pre-projection embedding (``forward_features``).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),         # BN here stabilises contrastive training
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#  SimCLR contrastive pretraining 

def pretrain_simclr(backbone: nn.Module, ssl_loader, device: torch.device,epochs: int = 100,lr: float = 1e-3, 
                    weight_decay: float = 1e-4,temperature: float = 0.5, projection_dim: int = 128, 
                    batch_size_ref: int = 256, use_amp: bool = True, save_path: str | Path | None = None,) -> dict[str, list[float]]:
    """
    Contrastively pretrain ``backbone.forward_features`` with NT-Xent.

    ``ssl_loader`` must yield pairs ``(view1, view2)`` with NO labels
    (two independent augmentations of the same image). The projection head is
    discarded after pretraining; only the backbone weights are kept.

    Args:
        lr            : base LR for a 256-sample batch. The effective LR is
                        scaled linearly with the real batch size (SimCLR rule),
                        because contrastive loss quality depends strongly on the
                        number of negatives (≈ batch size).
        batch_size_ref: reference batch size for the linear LR scaling.
        use_amp       : enable mixed-precision autocast (faster / less memory).
    """
    backbone = backbone.to(device)
    proj = ProjectionHead(backbone.feature_dim, hidden_dim=512, out_dim=projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)

    # Linear LR scaling: infer the real batch size from the first batch.
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
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=amp_on)

    history: dict[str, list[float]] = {"ssl_loss": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        backbone.train(); proj.train()
        running, n = 0.0, 0
        for view1, view2 in ssl_loader:
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_on):
                z1 = proj(backbone.forward_features(view1))   # project both views …
                z2 = proj(backbone.forward_features(view2))
                loss = criterion(z1, z2)                       # … and contrast them
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                         # unscale before clipping
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

    print(f"\n[SimCLR] Pretraining finished in {(time.time() - t0)/60:.1f} min.")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), save_path)
        print(f"[SimCLR] Backbone weights saved → {save_path}")
    return history


#  Rotation-prediction pretraining (alternative pretext) 

def pretrain_rotation(
    backbone: nn.Module,
    feature_loader,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    max_rot_batch: int = 256,
    save_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """
    Rotation-prediction pretext (Gidaris et al., 2018).

    Each image is rotated by one of {0°, 90°, 180°, 270°} and the backbone + a
    4-way head must predict which. Labels in ``feature_loader`` are ignored —
    the rotation index is the self-supervised target.

    NOTE: building all 4 rotations stacks a 4x-sized tensor in memory. To avoid
    OOM with large input batches we cap the effective rotation batch at
    ``max_rot_batch`` by trimming the source batch before expansion.
    """
    backbone = backbone.to(device)
    rot_head = nn.Linear(backbone.feature_dim, 4).to(device)   # 4-way rotation classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(list(backbone.parameters()) + list(rot_head.parameters()),
                      lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    amp_on = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=amp_on)

    history: dict[str, list[float]] = {"ssl_loss": [], "ssl_acc": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        backbone.train(); rot_head.train()
        running, correct, n = 0.0, 0, 0
        for images, _ in feature_loader:                       # labels ignored
            images = images.to(device, non_blocking=True)
            # Trim so the 4x expansion stays within the memory cap.
            cap = max(1, max_rot_batch // 4)
            if images.size(0) > cap:
                images = images[:cap]
            # Build the 4 rotations and their target indices.
            batch, targets = [], []
            for k in range(4):
                batch.append(torch.rot90(images, k, dims=(2, 3)))
                targets.append(torch.full((images.size(0),), k, dtype=torch.long, device=device))
            x = torch.cat(batch, dim=0)
            y = torch.cat(targets, dim=0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_on):
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

    print(f"\n[Rotation] Pretraining finished in {(time.time() - t0)/60:.1f} min.")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), save_path)
        print(f"[Rotation] Backbone weights saved → {save_path}")
    return history


#  Feature extraction 

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader,
    device: torch.device,
    l2_normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the FROZEN backbone over ``loader`` and return ``(features, labels)``.

    ``loader`` yields ``(image, label)``. Features are the penultimate
    embeddings from ``forward_features``. L2-normalising them puts every
    embedding on the unit sphere, which is what cosine-based / linear
    classifiers expect and what SimCLR trained the space to be.
    """
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


#  Traditional classifier on frozen features 

def fit_traditional_classifier(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    classifier: str = "logreg",
    seed: int = 42,
):
    """
    Fit a traditional classifier on the SSL features.

    Args:
        classifier : 'logreg' (multinomial SAGA) | 'linear_svm' | 'knn'.

    Returns:
        A fitted scikit-learn ``Pipeline`` (StandardScaler → estimator) exposing
        ``.predict``. Standardising first markedly speeds up and stabilises the
        linear solvers on 251-class, high-dimensional features.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if classifier == "logreg":
        # SAGA solver handles 251 classes with softmax (multinomial) far better
        # than one-vs-rest; capped iterations keep it fast (efficiency grade).
        # Note: recent scikit-learn defaults saga to multinomial automatically,
        # so we don't pass the (now-removed in some versions) multi_class arg.
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            solver="saga", max_iter=200, C=1.0, n_jobs=-1, random_state=seed,
        )
    elif classifier == "linear_svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=seed)
    elif classifier == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=20, weights="distance", n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier '{classifier}'. Choose: logreg, linear_svm, knn.")

    pipe = make_pipeline(StandardScaler(), clf)   # scale → classify
    pipe.fit(train_feats, train_labels)
    return pipe


def run_ssl_pipeline(
    backbone: nn.Module,
    ssl_loader,
    train_feat_loader,
    val_feat_loader,
    device: torch.device,
    method: str = "simclr",
    classifier: str = "logreg",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    projection_dim: int = 128,
    use_amp: bool = True,
    save_path: str | Path | None = None,
) -> dict:
    """
    End-to-end SSL task: pretrain → freeze → extract features → fit & score a
    traditional classifier on train/val (= test) splits.

    Returns a dict with the SSL training history, the fitted classifier, the
    extracted feature arrays, and validation predictions/labels (hand these to
    codes.evaluate.Evaluator for the SAME metrics used in the SL task, so the
    SL-vs-SSL comparison is apples-to-apples).
    """
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
    clf = fit_traditional_classifier(Xtr, ytr, classifier=classifier, seed=42)
    val_pred = clf.predict(Xva)

    return {
        "ssl_history": hist,
        "classifier": clf,
        "train_features": Xtr,
        "train_labels": ytr,
        "val_features": Xva,
        "val_labels": yva,
        "val_predictions": val_pred,
    }