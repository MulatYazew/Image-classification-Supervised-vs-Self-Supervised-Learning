"""
FoodNet Streamlit demo — model loading & inference utilities.

Thin wrapper around codes.model / codes.data_handler / codes.self_supervised
so the app reuses the EXACT training-time architecture registry and
preprocessing instead of duplicating that logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from codes import config as C
from codes import data_handler as dh
from codes import model as M


class ArtifactError(RuntimeError):
    """A required checkpoint/results file is missing or unreadable — caught
    at the call site so the app shows a clean st.error() instead of a raw traceback."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise ArtifactError(
            f"Missing {what}: {path}\n\n"
            f"Run the training notebook "
            f"(notebooks/FoodNet_Supervised_Self_Supervised.ipynb) first to produce it."
        )
    return path


def load_class_names(num_classes: int = 251) -> dict[int, str]:
    """id -> human-readable food name from dataset/class_list.txt, via the
    same data_handler.load_class_names the training notebook uses. Falls
    back to 'class_<id>' for any missing id (including a missing file)."""
    return dh.load_class_names(num_classes=num_classes, class_list_path=C.CLASS_LIST_PATH)


def load_supervised_model(device: torch.device):
    """Load the overall-best supervised architecture (from
    results/sl_model_comparison.csv) + its checkpoint. Returns
    (model, info_dict); raises ArtifactError if anything is missing."""
    comparison_path = _require(
        C.RESULTS_DIR / "sl_model_comparison.csv", "supervised model comparison table")
    comparison = pd.read_csv(comparison_path)
    best_rows = comparison[comparison["is_overall_best"]]
    if best_rows.empty:
        raise ArtifactError(f"{comparison_path} has no row with is_overall_best=True.")
    best_row = best_rows.iloc[0]
    arch = str(best_row["architecture"])

    hparams_path = _require(C.RESULTS_DIR / f"{arch}_best_hparams.json", f"{arch} hyperparameters")
    cfg = json.loads(hparams_path.read_text())["config"]

    ckpt_path = _require(C.MODELS_DIR / arch / "best_model.pth", f"{arch} checkpoint")
    model = M.build_model(
        arch, num_classes=C.NUM_CLASSES,
        dropout=cfg.get("dropout", C.DROPOUT), width_mult=cfg.get("width_mult", C.WIDTH_MULT),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    info = {
        "architecture": arch,
        "paradigm": "Supervised",
        "params_M": model.model_info()["total_params_M"],
        "metrics": best_row.drop(labels=["architecture", "is_overall_best"]).to_dict(),
    }
    return model, info


def load_ssl_model(device: torch.device):
    """Load the winning SSL method's backbone + downstream classifier (from
    results/ssl_best_hparams.json + models/ssl_best/*). Returns (backbone,
    classifier, info_dict); raises ArtifactError if anything is missing."""
    hparams_path = _require(C.RESULTS_DIR / "ssl_best_hparams.json", "SSL hyperparameters")
    hparams = json.loads(hparams_path.read_text())
    cfg = hparams["config"]
    method = hparams["method"]

    backbone_path = _require(C.MODELS_DIR / "ssl_best" / "backbone.pth", "SSL backbone checkpoint")
    clf_path = _require(C.MODELS_DIR / "ssl_best" / "classifier.joblib", "SSL downstream classifier")

    backbone = M.build_model(
        cfg["model_name"], num_classes=C.NUM_CLASSES, width_mult=cfg.get("width_mult", C.WIDTH_MULT))
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    backbone.to(device).eval()
    classifier = joblib.load(clf_path)

    info = {
        "architecture": cfg["model_name"],
        "paradigm": f"Self-Supervised ({method})",
        "params_M": backbone.model_info()["total_params_M"],
        "metrics": {
            "selection_metric": hparams.get("selection_metric"),
            "selection_metric_value": hparams.get("selection_metric_value"),
        },
    }
    return backbone, classifier, info


def load_sl_vs_ssl_comparison() -> pd.DataFrame | None:
    """results/sl_vs_ssl_comparison.csv, if the notebook has produced it."""
    path = C.RESULTS_DIR / "sl_vs_ssl_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Exact training-time preprocessing (data_handler.get_transforms,
    augment=False): resize to config.INPUT_SIZE + ImageNet normalisation.
    Returns a (1, 3, H, W) tensor ready for model input."""
    rgb = np.array(image.convert("RGB"))
    tf = dh.get_transforms(image_size=C.INPUT_SIZE, augment=False)
    tensor = tf(image=rgb)["image"]
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_supervised(model, tensor: torch.Tensor, device: torch.device,
                       class_names: dict[int, str], k: int = 5) -> list[tuple[str, float]]:
    """Top-k (class_name, confidence) from the supervised model's softmax."""
    logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = np.argsort(-probs)[:k]
    return [(class_names.get(int(i), f"class_{i}"), float(probs[i])) for i in top_idx]


@torch.no_grad()
def predict_self_supervised(backbone, classifier, tensor: torch.Tensor, device: torch.device,
                            class_names: dict[int, str], k: int = 5) -> list[tuple[str, float]]:
    """Top-k (class_name, confidence) from the frozen SSL backbone's features
    + traditional classifier. L2-normalises features exactly like
    extract_features (the classifier was fit on normalised features). Falls
    back to a single top-1 prediction with confidence 1.0 when the fitted
    classifier has no predict_proba (e.g. classifier="linear_svm")."""
    feats = backbone.forward_features(tensor.to(device))
    feats = F.normalize(feats, dim=1).cpu().numpy()

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(feats)[0]
        order = np.argsort(-probs)[:k]
        classes = classifier.classes_
        return [(class_names.get(int(classes[i]), f"class_{classes[i]}"), float(probs[i])) for i in order]

    pred = int(classifier.predict(feats)[0])
    return [(class_names.get(pred, f"class_{pred}"), 1.0)]
