"""
FoodNet Data Handler
=====================
Dataset, DataLoaders, augmentation pipelines, class-weight helpers, and the
stratified train/validation split — all food-dataset aware.

Key spec-driven decisions: 251 classes always (build_dataframe never drops a
class; optional sub-sampling only caps images per class). Validation is our
test set, carved from train via stratified_split so all 251 classes appear
in both. Inputs are resized to a common square and kept RGB (food colour is
discriminative). Moderate imbalance (100-600/class) is handled via weighted
CE/focal loss (loss_function.py). SSLPairDataset returns two augmented views
per image with no label, for SimCLR-style contrastive pretraining.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

# cv2's internal thread pool contends with DataLoader worker processes for
# cores (measured cause of num_workers>0 hangs/slowdowns on Apple Silicon).
# Must be set here (main process, import time) AND in worker_init_fn below,
# since macOS spawns (not forks) workers, re-running this module fresh.
cv2.setNumThreads(0)

# ImageNet stats are a reasonable normalisation even when training from scratch
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def worker_init_fn(_worker_id: int) -> None:
    """Pass to every DataLoader with num_workers > 0 — required: measured
    that the module-level cv2.setNumThreads(0) does not persist into spawned
    worker processes on macOS without this (workers came back at 10 threads
    each without it, 1 with it)."""
    cv2.setNumThreads(0)


def loader_kwargs(num_workers: int, prefetch_factor: int = 2) -> dict:
    """Extra DataLoader kwargs for num_workers — persistent_workers/
    prefetch_factor/worker_init_fn only when num_workers > 0 (PyTorch errors
    if they're passed alongside num_workers=0). Spread into every
    DataLoader(ds, num_workers=n, **loader_kwargs(n))."""
    if num_workers <= 0:
        return {}
    return {
        "worker_init_fn": worker_init_fn,
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


# Manifest construction & label maps

def build_dataframe(
    csv_path: str | Path,
    image_col_candidates: tuple[str, ...] = ("image_id", "img_id", "image", "filename", "id", "img_name", "image_name"),
    label_col_candidates: tuple[str, ...] = ("label", "class", "class_id", "category", "target"),
) -> pd.DataFrame:
    """Load a labels CSV, normalise to columns image_id (str) and label (int);
    raises if either can't be found. String labels are factorised to
    contiguous ints 0..K-1, with the mapping stored on df.attrs."""
    df = pd.read_csv(csv_path)
    rename: dict[str, str] = {}
    for col in df.columns:
        low = col.lower()
        if low in image_col_candidates:
            rename[col] = "image_id"
        elif low in label_col_candidates:
            rename[col] = "label"
    df = df.rename(columns=rename)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Cannot find image_id/label columns. Found: {df.columns.tolist()}")

    df["image_id"] = df["image_id"].astype(str)

    if not np.issubdtype(df["label"].dtype, np.integer):
        codes, uniques = pd.factorize(df["label"], sort=True)
        df["label"] = codes.astype(int)
        df.attrs["label_names"] = {i: str(name) for i, name in enumerate(uniques)}
    else:
        df["label"] = df["label"].astype(int)
        df.attrs["label_names"] = {int(c): str(c) for c in sorted(df["label"].unique())}

    return df.reset_index(drop=True)


def load_class_names(
    df: pd.DataFrame | None = None,
    num_classes: int = 251,
    class_list_path: str | Path | None = None,
) -> dict[int, str]:
    """Return {label_id: name}, falling back to 'class_<i>' for missing ids.
    df.attrs["label_names"] only holds real names when the CSV's label column
    was originally strings — this dataset's labels are already numeric IDs,
    so pass class_list_path (config.CLASS_LIST_PATH, "<id> <name>" per line)
    to fill in real food names; where both sources cover an id, this wins.
    """
    names: dict[int, str] = {}
    if df is not None:
        names.update(df.attrs.get("label_names", {}))
    if class_list_path is not None:
        path = Path(class_list_path)
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                idx_str, _, name = line.partition(" ")
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if 0 <= idx < num_classes:
                    names[idx] = name
    return {i: names.get(i, f"class_{i}") for i in range(num_classes)}


def cap_images_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    """Optionally cap images per class (documented compute-cost reduction);
    the set of classes itself is unchanged. Returns df unchanged if
    max_per_class is None."""
    if max_per_class is None:
        return df.reset_index(drop=True)
    capped = pd.concat([
        g.sample(n=min(len(g), max_per_class), random_state=seed)
        for _, g in df.groupby("label", group_keys=False)
    ])
    return capped.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split — validation is our test set (the
    official test split has no ground truth). Stratifying by label guarantees
    every one of the 251 classes appears in both partitions, with at least
    one validation image even for the smallest classes."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, group in df.groupby("label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_split)) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    return train_df, val_df


# Augmentation pipelines

def get_transforms(image_size: int = 224, augment: bool = True, intensity: float = 0.5) -> A.Compose:
    """Standard pipeline for majority classes (training) and val/inference.

    intensity in [0, 1] scales every magnitude/probability via
    scale = intensity / 0.5, so intensity=0.5 reproduces the original
    hand-tuned pipeline, 0 degrades towards near-identity, 1 is visibly more
    aggressive — lets config.AUGMENTATION_INTENSITY drive ablation runs.

    Food-specific choices: RandomResizedCrop (varying plate framing),
    HorizontalFlip (no preferred orientation), Rotate ±20° (casual phone
    photos), BrightnessContrast (lighting varies), ColorJitter with hue
    capped tight even at intensity=1 (colour is a strong food cue — a wide
    hue jitter could turn a tomato blue; no grayscale, unlike the SSL
    pipeline below which can afford to destroy colour), CoarseDropout
    (garnish/utensils/hand occlusion), RandomShadow (uneven lighting).
    """
    if not augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    scale = float(np.clip(intensity, 0.0, 1.0)) / 0.5   # 1.0 at the default intensity=0.5

    def p(base: float) -> float:
        return float(np.clip(base * scale, 0.0, 1.0))
    crop_lo = float(np.clip(1.0 - 0.3 * scale, 0.3, 1.0))
    rotate_limit = max(1, round(15 * scale))
    affine_rotate = 10 * scale
    affine_translate = float(np.clip(0.05 * scale, 0.0, 0.3))
    affine_scale = float(np.clip(0.1 * scale, 0.0, 0.4))
    hue_jitter = min(0.06, 0.03 * scale)                             # capped regardless of scale
    n_holes = max(1, round(2 * scale))
    hole_frac = float(np.clip(0.10 * scale, 0.03, 0.25))

    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(crop_lo, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=rotate_limit, border_mode=0, p=p(0.5)),
        A.RandomBrightnessContrast(brightness_limit=0.15 * scale, contrast_limit=0.15 * scale, p=p(0.3)),
        A.ColorJitter(brightness=0.15 * scale, contrast=0.15 * scale, saturation=0.1 * scale,
                     hue=hue_jitter, p=p(0.3)),
        A.Affine(translate_percent=affine_translate,
                 scale=(1.0 - affine_scale, 1.0 + affine_scale),
                 rotate=(-affine_rotate, affine_rotate), p=p(0.4)),
        A.CoarseDropout(num_holes_range=(1, n_holes),
                        hole_height_range=(0.03, hole_frac),
                        hole_width_range=(0.03, hole_frac),
                        fill=0, p=p(0.25)),
        A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_limit=(1, 2),
                       shadow_intensity_range=(0.3, 0.6), p=p(0.2)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_ssl_transforms(image_size: int = 224) -> A.Compose:
    """Strong augmentation for SimCLR contrastive pretraining — two
    independent draws of this pipeline on the same image form a positive
    pair. Standard SimCLR recipe (heavy crop + colour distortion + grayscale
    + blur) forces the backbone to learn structure invariant to appearance,
    without labels."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.2, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# Datasets

def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FoodDataset(Dataset):
    """Supervised Food-251 dataset (columns image_id, label).

    Imbalance is primarily handled via weighted CE/focal loss. Optionally, a
    second, independent correction can be enabled here: tail_classes get
    augmented at intensity * tail_boost for more diverse synthetic variation
    per epoch — this reweights pixels, not gradients, so it's safe to combine
    with either loss-weight or sampler correction (no double-correction risk).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str | Path,
        augment: bool = True,
        image_size: int = 224,
        intensity: float = 0.5,
        tail_classes: set[int] | None = None,
        tail_boost: float = 1.4,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tail_classes = tail_classes or set()
        self.tf = get_transforms(image_size, augment=augment, intensity=intensity)
        # only build a second pipeline when tail-aware augmentation is requested
        self.tf_tail = (
            get_transforms(image_size, augment=augment, intensity=min(1.0, intensity * tail_boost))
            if augment and self.tail_classes else self.tf
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = read_rgb(self.images_dir / row["image_id"])
        tf = self.tf_tail if int(row["label"]) in self.tail_classes else self.tf
        image = tf(image=image)["image"]
        return image, torch.tensor(int(row["label"]), dtype=torch.long)


class SSLPairDataset(Dataset):
    """Self-supervised dataset: returns two augmented views of each image and
    no label, for a SimCLR NT-Xent loss. Manifest labels are deliberately
    ignored — the "ignore the labels" SSL setting required by the exam."""

    def __init__(self, dataframe: pd.DataFrame, images_dir: str | Path, image_size: int = 224) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.ssl_tf = get_ssl_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = read_rgb(self.images_dir / row["image_id"])
        view1 = self.ssl_tf(image=image)["image"]
        view2 = self.ssl_tf(image=image)["image"]
        return view1, view2


class FeatureExtractionDataset(FoodDataset):
    """Deterministic (no-augmentation) FoodDataset alias for extracting
    frozen-backbone features in the SSL -> traditional-classifier pipeline —
    named for intent rather than passing an augment flag at call sites."""

    def __init__(self, dataframe: pd.DataFrame, images_dir: str | Path, image_size: int = 224) -> None:
        super().__init__(dataframe, images_dir, augment=False, image_size=image_size)


# Class-weight / sampler / minority helpers

def compute_tail_classes(dataframe: pd.DataFrame, num_classes: int = 251,
                         tail_frac: float = 0.2) -> set[int]:
    """Class ids in the smallest tail_frac fraction of the per-class image
    count (e.g. the ~34-image classes). Shared definition used by both
    FoodDataset (tail-aware augmentation) and evaluate.py (head-vs-tail
    metric breakdown, via config.TAIL_CLASS_FRACTION)."""
    counts = dataframe["label"].value_counts().reindex(range(num_classes), fill_value=0)
    n_tail = max(1, round(num_classes * tail_frac))
    return set(counts.sort_values(kind="mergesort").index[:n_tail].tolist())


def _label_counts(dataframe: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Per-class image counts as a dense (num_classes,) tensor, vectorised
    rather than looped — shared by compute_class_weights and build_weighted_sampler."""
    counts = dataframe["label"].value_counts().reindex(range(num_classes), fill_value=0)
    return torch.tensor(counts.to_numpy(), dtype=torch.float)


def check_single_imbalance_correction(use_weighted_sampler: bool,
                                      class_weights: torch.Tensor | None) -> None:
    """Raise if both a weighted sampler and loss class-weights are active at
    once — stacking both over-corrects the same imbalance (oversampling rare
    classes AND up-weighting their loss), destabilising the rarest ~34-image
    classes. Enforces the "pick exactly one" convention as an invariant."""
    if use_weighted_sampler and class_weights is not None:
        raise ValueError(
            "Both a WeightedRandomSampler and non-None class_weights are active — "
            "pick exactly one imbalance-correction path (config.USE_WEIGHTED_SAMPLER "
            "XOR config.CLASS_WEIGHT_SCHEME), not both."
        )


def compute_class_weights(dataframe: pd.DataFrame, num_classes: int = 251,
                          scheme: str = "sqrt_inv", beta: float = 0.999,
                          clip: float = 10.0) -> torch.Tensor:
    """Per-class loss weights for the real Food-251 imbalance (measured
    ~34-656 images/class, ~19:1, worse than the spec's nominal 100-600).
    Raw inverse frequency would give the 34-image class a ~20x gradient
    multiplier, amplifying its label noise — so three milder schemes are
    offered (default sqrt_inv): "inv" (classic inverse frequency, for
    reference/ablation), "sqrt_inv" (weights ~ 1/sqrt(count), standard for
    ~10-20:1 imbalance), "effective" (class-balanced weights, Cui et al. 2019,
    beta near 1). All schemes are mean-normalised to ~1 and clipped at clip.
    Use this OR a weighted sampler, never both.
    """
    counts = _label_counts(dataframe, num_classes).clamp(min=1.0)

    if scheme == "inv":
        w = counts.sum() / (num_classes * counts)
    elif scheme == "sqrt_inv":
        w = 1.0 / counts.sqrt()
    elif scheme == "effective":
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / eff
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose: inv, sqrt_inv, effective.")

    w = w / w.mean()                 # normalise so mean weight ≈ 1
    w = w.clamp(max=clip)            # bound the rarest-class multiplier
    return w


def build_weighted_sampler(dataframe: pd.DataFrame, num_classes: int = 251) -> WeightedRandomSampler:
    """WeightedRandomSampler for balanced mini-batches (use instead of loss
    weights). Uses raw inverse-frequency per-sample weights (unlike the
    sqrt-tempered loss weights) so rare classes are oversampled with
    replacement until every class is equally likely per batch. Combining with
    weighted loss double-corrects — pick exactly one (config.USE_WEIGHTED_SAMPLER)."""
    counts = _label_counts(dataframe, num_classes).clamp(min=1.0)
    inv_freq = 1.0 / counts
    labels = torch.from_numpy(dataframe["label"].to_numpy().astype("int64"))
    sample_weights = inv_freq[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
