"""
FoodNet Data Handler
=====================
Dataset, DataLoaders, augmentation pipelines, class-weight helpers, and the
stratified train/validation split — all *food-dataset aware*.

Key spec-driven decisions
-------------------------
  * 251 classes, always.** ``build_dataframe`` never drops a class. Optional
    sub-sampling caps images-PER-CLASS only (documented, computational reason).
  * Validation = our test set, carved from train  ``stratified_split`` holds
    out a per-class fraction so all 251 classes appear in validation.
  * Uncontrolled input size → resize to a common square.
  * RGB kept (food colour is highly discriminative — no greyscale).
  * Moderate imbalance (100–600 / class) handled by weighted CE / focal loss
    (see loss_function.py). A single uniform augmentation pipeline is applied
    across all classes.
  * SSL views SSLPairDataset returns two augmented views per image with
    NO label — exactly what SimCLR-style contrastive pretraining needs.
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

# ImageNet statistics are a reasonable normalisation even when training from
# scratch; they centre RGB inputs sensibly and match the demo / outlier code.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


#  Manifest construction & label maps 

def build_dataframe(
    csv_path: str | Path,
    image_col_candidates: tuple[str, ...] = ("image_id", "img_id", "image", "filename", "id", "img_name", "image_name"),
    label_col_candidates: tuple[str, ...] = ("label", "class", "class_id", "category", "target"),
) -> pd.DataFrame:
    """
    Load a labels CSV and normalise it to columns ``image_id`` (str) and
    ``label`` (int). Raises if either column cannot be found.

    String labels (e.g. food-name folders) are factorised to contiguous integer
    ids 0..K-1; the mapping is stored on the returned frame as ``df.attrs``.
    """
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

    # Map string labels → contiguous ints if needed.
    if not np.issubdtype(df["label"].dtype, np.integer):
        codes, uniques = pd.factorize(df["label"], sort=True)
        df["label"] = codes.astype(int)
        df.attrs["label_names"] = {i: str(name) for i, name in enumerate(uniques)}
    else:
        df["label"] = df["label"].astype(int)
        df.attrs["label_names"] = {int(c): str(c) for c in sorted(df["label"].unique())}

    return df.reset_index(drop=True)


def load_class_names(df: pd.DataFrame, num_classes: int = 251) -> dict[int, str]:
    """Return {label_id: name}. Falls back to 'class_<i>' for any missing id."""
    names = dict(df.attrs.get("label_names", {}))
    return {i: names.get(i, f"class_{i}") for i in range(num_classes)}


def cap_images_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    """
    Optionally cap images PER CLASS (documented computational-cost reduction).

    The number of classes is preserved — every class that exists keeps at least
    its available images up to ``max_per_class``. Returns ``df`` unchanged when
    ``max_per_class`` is None.
    """
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
    """
    Stratified train/validation split — the validation set is our test set and
    is carved out of the training data (the official test split has no GT).

    Stratifying by ``label`` guarantees every one of the 251 classes is present
    in BOTH partitions. Classes with very few samples still contribute at least
    one validation image.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, group in df.groupby("label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_split))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    return train_df, val_df


#  Augmentation pipelines 

def get_transforms(image_size: int = 224, augment: bool = True, intensity: float = 0.5) -> A.Compose:
    """
    Standard pipeline for majority classes (training) and val/inference.

    ``intensity`` in [0, 1] scales every magnitude/probability below via
    ``scale = intensity / 0.5``, so ``intensity=0.5`` reproduces the original
    hand-tuned pipeline exactly, ``intensity=0`` degrades towards a near
    identity transform, and ``intensity=1`` is a visibly more aggressive
    policy. This lets a single config value (``config.AUGMENTATION_INTENSITY``)
    drive the report's augmentation-ablation runs.

    Food-specific choices:
      - RandomResizedCrop  : plates are shot at varying distances / framings.
      - HorizontalFlip     : food has no preferred left/right orientation.
      - Rotate (±20°)      : casual phone photos are rarely perfectly level.
      - BrightnessContrast : restaurant vs daylight vs flash lighting varies.
      - ColorJitter (mild) : white-balance differs across cameras — but hue is
                             capped tight (even at intensity=1) because colour
                             is a strong food cue (e.g. garlic bread vs.
                             focaccia); a wide hue jitter could turn a tomato
                             blue. Grayscale is deliberately NOT used here for
                             the same reason (unlike the SSL pipeline below,
                             which can afford to destroy colour because it
                             only needs structural invariance).
      - CoarseDropout      : simulates garnish, utensils, or a hand partially
                             covering the dish.
      - RandomShadow       : simulates uneven restaurant/daylight lighting.
      - Normalize          : ImageNet stats centre the RGB inputs.
    """
    if not augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    scale = float(np.clip(intensity, 0.0, 1.0)) / 0.5   # 1.0 at the default intensity=0.5
    p = lambda base: float(np.clip(base * scale, 0.0, 1.0))          # noqa: E731
    crop_lo = float(np.clip(1.0 - 0.3 * scale, 0.3, 1.0))
    rotate_limit = max(1, int(round(15 * scale)))
    affine_rotate = 10 * scale
    affine_translate = float(np.clip(0.05 * scale, 0.0, 0.3))
    affine_scale = float(np.clip(0.1 * scale, 0.0, 0.4))
    hue_jitter = min(0.06, 0.03 * scale)                             # tightly capped regardless of scale
    n_holes = max(1, int(round(2 * scale)))
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
    """
    Strong augmentation for SELF-SUPERVISED contrastive pretraining (SimCLR).

    Two independent draws of this pipeline on the same image form a positive
    pair. The heavy crop + colour distortion + grayscale + blur is the standard
    SimCLR recipe — it forces the backbone to learn food structure invariant to
    appearance nuisances, without using any labels.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.2, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


#  Datasets 

def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FoodDataset(Dataset):
    """
    Supervised Food-251 dataset.

    Class imbalance is primarily handled via weighted CE / focal loss (see
    loss_function.py). Optionally, a SECOND, complementary correction can be
    enabled here: classes in ``tail_classes`` are augmented at a higher
    ``intensity`` (``intensity * tail_boost``) so the data-poor classes see
    more diverse synthetic variation per epoch. This is independent of the
    loss-weight / sampler correction (which reweights gradients, not pixels),
    so it is safe to combine with either — unlike sampler-vs-loss-weights,
    there is no double-correction risk here.

    Args:
        dataframe    : columns 'image_id' and 'label'.
        images_dir   : directory of raw images.
        augment      : apply training augmentations when True.
        image_size   : common resize target (uncontrolled inputs → square).
        intensity    : base augmentation intensity (config.AUGMENTATION_INTENSITY).
        tail_classes : optional set of class ids to augment more aggressively
                       (see ``compute_tail_classes``); None/empty = uniform
                       augmentation across all classes (original behaviour).
        tail_boost   : multiplier applied to ``intensity`` for tail classes.
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
        # Only build a second pipeline when tail-aware augmentation is actually
        # requested; otherwise every sample uses the single uniform pipeline.
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
    """
    Self-supervised dataset: returns TWO augmented views of each image and NO
    label. Feed the pair to a SimCLR NT-Xent loss for contrastive pretraining.

    The labels in the manifest are deliberately ignored here — this is the
    "ignore the labels" SSL setting required by the exam.
    """

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


class FeatureExtractionDataset(Dataset):
    """
    Deterministic (no-augmentation) dataset used to extract frozen-backbone
    features for the SSL → traditional-classifier pipeline. Returns
    ``(image_tensor, label)`` so the traditional classifier can be fit/scored.
    """

    def __init__(self, dataframe: pd.DataFrame, images_dir: str | Path, image_size: int = 224) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tf = get_transforms(image_size, augment=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = read_rgb(self.images_dir / row["image_id"])
        image = self.tf(image=image)["image"]
        return image, torch.tensor(int(row["label"]), dtype=torch.long)


#  Class-weight / sampler / minority helpers

def compute_tail_classes(dataframe: pd.DataFrame, num_classes: int = 251,
                         tail_frac: float = 0.2) -> set[int]:
    """
    Return the class ids in the smallest ``tail_frac`` fraction of the
    per-class image-count distribution (e.g. the ~34-image classes).

    Two consumers share this single definition of "tail class" so the report
    stays internally consistent:
      * ``FoodDataset`` (tail-aware augmentation boost, see above), and
      * ``evaluate.py`` (head-vs-tail metric breakdown), via
        ``config.TAIL_CLASS_FRACTION``.
    """
    counts = dataframe["label"].value_counts().reindex(range(num_classes), fill_value=0)
    n_tail = max(1, int(round(num_classes * tail_frac)))
    return set(counts.sort_values(kind="mergesort").index[:n_tail].tolist())


def check_single_imbalance_correction(use_weighted_sampler: bool,
                                      class_weights: torch.Tensor | None) -> None:
    """
    Raise if BOTH a weighted sampler and non-None loss class-weights are
    active at once. Stacking the two over-corrects the same imbalance twice
    (oversampling rare classes AND up-weighting their loss), which can
    destabilise training on the rarest ~34-image classes. Call this once
    after building the sampler/criterion for a run — cheap, and turns the
    "pick exactly one" convention documented on ``compute_class_weights`` /
    ``build_weighted_sampler`` into an enforced invariant rather than a
    comment that can silently rot.
    """
    if use_weighted_sampler and class_weights is not None:
        raise ValueError(
            "Both a WeightedRandomSampler and non-None class_weights are active — "
            "pick exactly one imbalance-correction path (config.USE_WEIGHTED_SAMPLER "
            "XOR config.CLASS_WEIGHT_SCHEME), not both."
        )


def compute_class_weights(dataframe: pd.DataFrame, num_classes: int = 251,
                          scheme: str = "sqrt_inv", beta: float = 0.999,
                          clip: float = 10.0) -> torch.Tensor:
    """
    Per-class loss weights for the REAL Food-251 imbalance.

    The actual training distribution is far more skewed than the spec's nominal
    "100–600": measured counts run from ~34 (class 162) to ~656, i.e. roughly
    19:1, not 6:1. Raw inverse frequency (``N / (K * count_c)``) then hands
    the 34-image class a ~20x gradient multiplier, which amplifies label noise on
    exactly the classes with the least reliable signal and destabilises training.

    Three milder schemes are offered (default ``sqrt_inv``):

      * ``inv``       : classic inverse frequency (kept for reference / ablation).
      * ``sqrt_inv``  : weights ∝ 1/sqrt(count_c) — tempers the tail (the standard
                        practical choice for ~10–20:1 imbalance).
      * ``effective`` : class-balanced weights ∝ (1 - beta) / (1 - beta^count_c)
                        (Cui et al., 2019), with ``beta`` near 1.

    All schemes are mean-normalised to ≈1 (so the overall loss scale matches an
    unweighted run) and clipped at ``clip`` to bound the largest multiplier.
    Pass the result to CrossEntropy / Focal — and use EITHER this OR a weighted
    sampler, never both (see config.USE_WEIGHTED_SAMPLER / loss.py).
    """
    counts = torch.zeros(num_classes)
    for label in dataframe["label"]:
        counts[int(label)] += 1
    counts = counts.clamp(min=1.0)

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
    """
    WeightedRandomSampler for balanced mini-batches (use INSTEAD of loss weights).

    Unlike the loss weights (which are sqrt-tempered to avoid over-amplifying the
    34-image class), the sampler uses RAW inverse-frequency per-sample weights:
    its job is to make every class equally likely to appear in a batch, so the
    rarest classes are oversampled with replacement. Combining this with weighted
    loss would double-correct, so pick exactly one (config.USE_WEIGHTED_SAMPLER).
    """
    counts = torch.zeros(num_classes)
    for label in dataframe["label"]:
        counts[int(label)] += 1
    counts = counts.clamp(min=1.0)
    inv_freq = 1.0 / counts                       # raw inverse frequency per class
    sample_weights = torch.tensor(
        [inv_freq[int(lbl)] for lbl in dataframe["label"]],
        dtype=torch.float,
    )
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
