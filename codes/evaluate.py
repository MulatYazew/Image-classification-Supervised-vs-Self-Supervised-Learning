"""
FoodNet Evaluator
=================
Computes the exam-required metrics — accuracy, precision, recall, F1-score —
plus confusion matrices and a per-class report. Works for BOTH paradigms:

  * Supervised : pass a trained model + DataLoader to "evaluate".
  * Self-sup.  : pass the SSL pipeline's "val_predictions" / "val_labels"
                 straight to "metrics_from_predictions" (no model needed).

A "compare_paradigms" helper assembles the SL-vs-SSL table for the report.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


class Evaluator:
    """
    Metrics and visualisations for Food-251 models.

    Args:
        num_classes : 251.
        class_names : list of names (length num_classes); auto-filled if None.
        device      : inference device.
    """

    def __init__(self, num_classes: int = 251, class_names: list[str] | None = None,
                 device: torch.device = torch.device("cpu")) -> None:
        self.num_classes = num_classes
        self.device = device
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    #  Inference 

    @torch.no_grad()
    def predict(self, model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
        """Run ``model`` over ``loader``; return (predictions, true_labels)."""
        model = model.to(self.device).eval()
        preds, labels = [], []
        for images, lbls in loader:
            images = images.to(self.device)
            preds.extend(model(images).argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
        return np.array(preds), np.array(labels)

    #  Metrics 

    @staticmethod
    def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Accuracy + macro & weighted precision/recall/F1.

        Macro treats every one of the 251 classes equally (so the small classes
        count as much as the big ones); weighted accounts for support.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Per-class precision/recall/F1/support — NOT just the macro average.

        With ~19:1 class imbalance, macro-F1 can look acceptable while the
        smallest (~34-image) classes sit at zero recall; this table is what
        actually shows whether those tail classes are improving across a
        training run, rather than the head classes inflating the average.
        """
        labels = np.arange(self.num_classes)
        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        support = pd.Series(y_true).value_counts().reindex(labels, fill_value=0).to_numpy()
        return pd.DataFrame({
            "label": labels,
            "class_name": [self.class_names[i] for i in labels],
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    @staticmethod
    def head_vs_tail_summary(per_class_df: pd.DataFrame, tail_classes: set[int] | None = None,
                             tail_frac: float = 0.2) -> dict:
        """
        Split the per-class table into head/tail groups and report mean
        recall/F1 for each — the single number that tells you whether an
        imbalance fix (re-weighting, tail-aware augmentation, oversampling)
        is actually helping the data-poor classes rather than just the
        macro average moving because the head classes got better.

        ``tail_classes`` should be ``data_handler.compute_tail_classes(...)``
        so the definition of "tail" matches the one used for augmentation;
        if omitted, the smallest ``tail_frac`` of classes BY SUPPORT in this
        table are used instead (only valid when the table covers a
        representative validation split).
        """
        if tail_classes is None:
            n_tail = max(1, round(len(per_class_df) * tail_frac))
            tail_classes = set(
                per_class_df.sort_values("support", kind="mergesort")["label"].head(n_tail)
            )
        is_tail = per_class_df["label"].isin(tail_classes)
        tail_df, head_df = per_class_df[is_tail], per_class_df[~is_tail]
        return {
            "n_tail_classes": int(is_tail.sum()),
            "n_head_classes": int((~is_tail).sum()),
            "tail_recall_mean": float(tail_df["recall"].mean()) if len(tail_df) else float("nan"),
            "tail_f1_mean": float(tail_df["f1"].mean()) if len(tail_df) else float("nan"),
            "head_recall_mean": float(head_df["recall"].mean()) if len(head_df) else float("nan"),
            "head_f1_mean": float(head_df["f1"].mean()) if len(head_df) else float("nan"),
        }

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray, max_classes: int = 30) -> None:
        """Print the sklearn classification report (truncated for 251 classes)."""
        names = self.class_names if self.num_classes <= max_classes else None
        print(classification_report(y_true, y_pred, target_names=names, zero_division=0))

    @staticmethod
    def top_k_accuracy(model: nn.Module, loader, device, k: int = 5) -> float:
        """Top-k accuracy — a fairer headline metric for 251 fine-grained classes."""
        model = model.to(device).eval()
        correct, n = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                topk = model(images).topk(k, dim=1).indices.cpu()
                correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
                n += len(labels)
        return correct / max(n, 1)

    #  Confusion matrix 

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(12, 10), normalize=True,
                              annotate=False, save_path: str | None = None) -> None:
        """
        Confusion-matrix heatmap. For 251 classes the cells are tiny, so the
        default omits annotations and normalises by true-label counts to reveal
        systematic confusions between similar dishes.
        """
        cm = confusion_matrix(y_true, y_pred)
        fmt = "d"
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            fmt = ".2f"
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=annotate, fmt=fmt, cmap="viridis",
                    cbar_kws={"label": "Rate" if normalize else "Count"})
        plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()

    #  Full pipeline 

    def evaluate(self, model: nn.Module, loader) -> dict:
        """Predict → metrics → confusion matrix (supervised path)."""
        y_pred, y_true = self.predict(model, loader)
        return {
            "predictions": y_pred,
            "true_labels": y_true,
            "metrics": self.metrics_from_predictions(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }


#  SL vs SSL comparison 

def compare_paradigms(sl_metrics: dict, ssl_metrics: dict,
                      sl_label: str = "Supervised (SL)",
                      ssl_label: str = "Self-Supervised (SSL)") -> pd.DataFrame:
    """
    Build the headline SL-vs-SSL comparison table for the report.

    Args:
        sl_metrics  : Evaluator metrics dict from the supervised model.
        ssl_metrics : Evaluator metrics dict from the SSL + traditional-classifier
                      pipeline (via metrics_from_predictions on val_predictions).
        sl_label    : column header for the supervised row values. Pass something
                      unambiguous (e.g. "Supervised (best CNN: foodnet46)").
        ssl_label   : column header for the SSL row values. Pass something that
                      names it as the FROZEN-FEATURE + TRADITIONAL-CLASSIFIER
                      result (e.g. "Self-Supervised downstream classifier (SimCLR
                      features + logreg)"), never just "SimCLR" -- that would read
                      as the pretext task's own accuracy, which doesn't exist
                      (NT-Xent has no classification accuracy).
    """

    keys = ["accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro"]
    rows = []
    for k in keys:
        sl = sl_metrics.get(k, float("nan"))
        ssl = ssl_metrics.get(k, float("nan"))
        rows.append({
            "metric": k,
            sl_label: round(sl, 4),
            ssl_label: round(ssl, 4),
            "Δ (SL − SSL)": round(sl - ssl, 4),
        })
    return pd.DataFrame(rows)


def plot_training_curves(history: dict, save_path: str | None = None) -> None:
    """Plot train/val loss and accuracy curves from a Trainer history dict."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_ssl_curves(history: dict, method: str = "", save_path: str | None = None) -> None:
    """
    Plot an SSL pretraining history dict (``self_supervised.pretrain_simclr`` /
    ``pretrain_rotation``): the pretext ``ssl_loss`` per epoch, plus
    ``ssl_acc`` (rotation's 4-way prediction accuracy) when present.

    Unlike ``plot_training_curves`` (which expects a supervised ``Trainer``
    history's train/val loss+accuracy KEYS), SSL pretraining has no labelled
    validation split during the pretext task itself — there is one loss curve
    (SimCLR NT-Xent, or rotation cross-entropy + its own accuracy), not a
    train-vs-val pair, hence a dedicated plot rather than reusing
    ``plot_training_curves`` on a differently-shaped dict.
    """
    has_acc = bool(history.get("ssl_acc"))
    _fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14, 5) if has_acc else (7, 5))
    ax1 = axes[0] if has_acc else axes
    ax1.plot(history["ssl_loss"], label="ssl_loss")
    ax1.set_title(f"{method} pretext loss".strip() or "SSL pretext loss")
    ax1.set_xlabel("epoch"); ax1.legend()
    if has_acc:
        ax2 = axes[1]
        ax2.plot(history["ssl_acc"], label="ssl_acc", color="tab:orange")
        ax2.set_title(f"{method} pretext accuracy".strip() or "SSL pretext accuracy")
        ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_ssl_downstream_performance(train_metrics: dict, val_metrics: dict, method: str = "",
                                    classifier: str = "", save_path: str | None = None) -> pd.DataFrame:
    """
    Bar chart + summary table for the SSL DOWNSTREAM (read-out) classifier's
    own train-vs-val accuracy and macro-F1.

    This is deliberately separate from ``plot_ssl_curves`` (the pretext-task
    NT-Xent/rotation loss, which has no train/val split of its own): the
    traditional classifier (logreg/linear_svm/knn) is a single fit on frozen
    features, not an iterative training run, so there is no per-epoch curve --
    just one train score and one val score per metric. The gap between them is
    what shows whether the read-out classifier over/underfits the frozen
    features, independent of how well the pretext task converged.

    Args:
        train_metrics : ``Evaluator.metrics_from_predictions`` output scored on
                        the classifier's OWN training features/labels.
        val_metrics   : same, scored on the val (= test) features/labels.
        method        : pretext method name, e.g. "simclr" (plot title only).
        classifier    : classifier name, e.g. "logreg" (plot title only).

    Returns:
        The summary DataFrame (accuracy/f1_macro train vs val vs gap) -- also
        useful to ``display()`` or save as CSV alongside the plot.
    """
    keys = ["accuracy", "f1_macro"]
    df = pd.DataFrame([
        {
            "metric": k,
            "train": train_metrics.get(k, float("nan")),
            "val": val_metrics.get(k, float("nan")),
            "gap (train − val)": train_metrics.get(k, float("nan")) - val_metrics.get(k, float("nan")),
        }
        for k in keys
    ])

    x = np.arange(len(df)); width = 0.35
    _fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, df["train"], width, label="train")
    ax.bar(x + width / 2, df["val"], width, label="val")
    ax.set_xticks(x); ax.set_xticklabels(df["metric"])
    ax.set_ylabel("score")
    title = " ".join(p for p in (method, classifier) if p) or "SSL downstream classifier"
    ax.set_title(f"{title}: downstream classifier train vs val")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return df
