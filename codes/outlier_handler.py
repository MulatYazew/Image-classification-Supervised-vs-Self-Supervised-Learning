"""
FoodNet - outlier_handler 
========================================
Core 3-stage outlier detection for food image datasets (251 classes).
Trimmed down to: Stage 1 (integrity), Stage 2 (pixel stats), Stage 3
(per-class embedding outliers), plus essential visualisations.


Usage
-----
    from outlier_handler import run_outlier_pipeline, visualize_flagged_images, apply_review_decisions

    df, stats_df, flagged2, feats, ids, global_scores, per_class_scores, flagged3 = run_outlier_pipeline(
        csv_path="train_labels.csv", img_dir="train_set/", device="mps", out_dir="results"
    )

    visualize_flagged_images(flagged2, "train_set/", title="Stage 2 review")
    visualize_flagged_images(flagged3, "train_set/", title="Stage 3 review")

    # Manually inspect results/review_stage2_*.csv and review_stage3_*.csv,
    # delete rows you want to KEEP, save as confirmed_remove_stage{2,3}.csv,
    # then:
    final_df = apply_review_decisions(df, ["confirmed_remove_stage2.csv",
                                            "confirmed_remove_stage3.csv"],
                                       output_csv="train_labels_clean.csv")
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from typing import Optional, List

#  colours for plots
BG, PANEL, ACCENT, FLAG, LINE, SPINE = "#1a1a2e", "#12122a", "#7b7baa", "#ff4444", "#ffcc00", "#444466"

#  thresholds 
FOOD_STATS_BOUNDS = {
    "saturation_mean":  (4.0,  248.0),
    "edge_density":     (0.005, 0.80),
    "grey_fraction":    (0.0,   0.93),
    "brightness_range": (1.5,   220.0),
}
EXTREME_SINGLE = {
    "saturation_mean":  (1.5,   252.0),
    "edge_density":     (0.002, 0.90),
    "grey_fraction":    (0.0,   0.97),
    "brightness_range": (0.5,   None),
}
FOOD_SAFE_SATURATION_MIN    = 20.0
FOOD_SAFE_EDGE_MAX          = 0.75
FOOD_SAFE_FAILURES_REQUIRED = 4
N_FAILS_DEFAULT  = 3
CLASS_ZSCORE_THR = 4.5
MIN_CLASS_SIZE   = 20



# STAGE 1 – Integrity & Blank Audit

# First line of defence: automatically and unconditionally removes images
# that are definitively unusable, requiring no human review.  Six failure
# modes are checked in priority order:
#   1. File missing from disk entirely.
#   2. File too small in bytes  → likely truncated / partially downloaded.
#   3. PIL cannot decode the file  → corrupt JPEG/PNG header.
#   4. Spatial dimensions below the minimum  → too small to carry useful
#      texture information for a 251-class classifier.
#   5. Near-black mean pixel value  → underexposed, lens-capped, or
#      all-black placeholder images.
#   6. Near-white mean with very low HSV saturation  → blank white cards,
#      overexposed shots, or scanner artefacts with no food content.
#   7. Extremely low pixel standard deviation  → single-colour / solid-fill
#      images that passed the brightness checks but carry no content.
# Survivors are returned as `clean_df`; removed images are logged to
# `removed_stage1_integrity.csv` for traceability.

def image_integrity_audit(
    df: pd.DataFrame,
    img_dir: str,
    out_dir: str = ".",
    min_bytes: int = 1_500,
    min_size_px: int = 64,
    min_std: float = 6.0,
    black_thresh: float = 8.0,
    white_thresh: float = 252.0,
    white_min_sat: float = 3.0,
) -> tuple:
    """Scan every image in *df* for hard integrity failures and remove them
    unconditionally (no manual review required).

    Checks performed in order:
      - **missing**        : file does not exist on disk.
      - **truncated**      : file size < *min_bytes* (likely incomplete download).
      - **corrupt**        : PIL raises an exception while opening/decoding.
      - **too_small**      : either spatial dimension < *min_size_px* pixels.
      - **near_black**     : RGB mean < *black_thresh* (underexposed / capped lens).
      - **near_white_blank**: RGB mean > *white_thresh* AND HSV saturation mean
                             < *white_min_sat* (blank/overexposed with no colour).
      - **low_contrast**   : pixel standard deviation < *min_std* (solid-fill or
                             near-uniform images that slipped through the above).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset manifest with columns ``image_id`` and ``label``.
    img_dir : str
        Root directory under which image files are stored.
    out_dir : str
        Directory where the audit CSV is written.
    min_bytes : int
        Minimum acceptable file size in bytes (default 1 500).
    min_size_px : int
        Minimum acceptable width **and** height in pixels (default 64).
    min_std : float
        Minimum acceptable pixel standard deviation across all channels
        (default 6.0).
    black_thresh : float
        RGB mean below this value → flagged as near-black (default 8.0).
    white_thresh : float
        RGB mean above this value triggers the saturation sub-check
        (default 252.0).
    white_min_sat : float
        HSV saturation mean below this value (when mean > *white_thresh*)
        → flagged as near-white blank (default 3.0).

    Returns
    -------
    clean_df : pd.DataFrame
        Subset of *df* with all integrity failures removed.
    issues_df : pd.DataFrame
        Records of every removed image with columns
        ``image_id``, ``label``, ``reason``.
    """
    issues = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Integrity audit"):
        path, reason = os.path.join(img_dir, str(row["image_id"])), None

        # file presence 
        if not os.path.exists(path):
            reason = "missing"

        # minimum file size (truncation guard) 
        elif os.path.getsize(path) < min_bytes:
            reason = f"truncated(bytes={os.path.getsize(path)})"

        else:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size

                # minimum spatial resolution 
                if w < min_size_px or h < min_size_px:
                    reason = f"too_small({w}x{h})"
                else:
                    arr = np.array(img, dtype=np.float32)
                    mean, std = arr.mean(), arr.std()

                    # near-black (underexposed / black frame) 
                    if mean < black_thresh:
                        reason = f"near_black(mean={mean:.1f})"

                    # near-white blank (overexposed / no content)
                    elif mean > white_thresh:
                        sat = np.array(img.convert("HSV"), dtype=np.float32)[:, :, 1].mean()
                        if sat < white_min_sat:
                            reason = f"near_white_blank(mean={mean:.1f},sat={sat:.1f})"

                    # near-uniform / solid-fill image 
                    elif std < min_std:
                        reason = f"low_contrast(mean={mean:.0f},std={std:.1f})"

            # corrupt / unreadable file (PIL decode error) 
            except Exception as ex:
                reason = f"corrupt:{ex}"

        if reason:
            issues.append({"image_id": row["image_id"], "label": row["label"], "reason": reason})

    # Build a DataFrame of removed images; keep a clean copy of survivors.
    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(columns=["image_id", "label", "reason"])
    clean_df = df[~df["image_id"].isin(issues_df["image_id"])].reset_index(drop=True)

    # Report a breakdown of removal reasons and persist the audit log.
    print(f"\n[Stage 1] Auto-removed {len(issues_df):,} / {len(df):,} images")
    if not issues_df.empty:
        print(issues_df["reason"].str.split("(").str[0].value_counts().to_string())
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "removed_stage1_integrity.csv")
        issues_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    return clean_df, issues_df



# STAGE 2 – Non-Food Pixel-Statistics Detector

# Second line of defence: lightweight, GPU-free analysis of raw pixel
# properties for every surviving image.  No deep model is needed here;
# classical image statistics are sufficient to catch most non-food images
# (e.g. text cards, pure-white product shots, monochrome diagrams, solid
# colour patches, synthetically generated placeholders).
#
# Five complementary criteria are combined with OR logic so that a single
# very strong violation (EXTREME_SINGLE) can flag an image on its own,
# while weaker violations must accumulate (n_fails_required / food-safe
# guard) before an image is flagged:
#   (a) Standard bounds  – 4 pixel-stat features each with a [lo, hi] range
#       tuned for food imagery; images outside the range accumulate failures.
#   (b) Food-safe guard  – images that are clearly colourful AND textured
#       (likely real food) need *more* simultaneous failures before being
#       flagged, reducing false positives on vivid dishes.
#   (c) Extreme single violation – very far out-of-range values on any one
#       feature are strong enough to flag the image alone.
#   (d) Per-class z-score – detects images that are statistical outliers
#       *within their own food class*, catching label errors and
#       cross-class contamination that global thresholds miss.
#   (e) Unique-colour count – images with fewer than 50 distinct colours
#       after a 64×64 resize are treated as synthetic or solid-fill.
#
# Flagged images are written to `review_stage2_nonfood_pixelstats.csv`
# for human review rather than automatic deletion.



def pixel_stats(path: str) -> Optional[dict]:
    """Compute a compact set of pixel-level statistics for a single image.

    All statistics are designed to be cheap to compute (no GPU, no deep
    model) while being informative enough to distinguish typical food
    photographs from non-food or degenerate images.

    Statistics returned
    -------------------
    mean_r / mean_g / mean_b : float
        Per-channel mean pixel intensity in [0, 255].  Strong colour
        dominance (e.g. very blue or very green) can signal non-food.
    overall_mean : float
        Mean intensity across all three channels combined.
    overall_std : float
        Standard deviation across all channels; very low values indicate
        near-uniform images.
    saturation_mean : float
        Mean HSV saturation in [0, 255].  Food images typically have
        moderate-to-high saturation; near-zero values suggest greyscale
        or washed-out content.
    grey_fraction : float
        Fraction of pixels where the per-pixel max–min channel spread is
        < 15; high values (close to 1.0) indicate a predominantly grey or
        desaturated image.
    edge_density : float
        Fraction of pixels with gradient magnitude > 20 (computed from
        finite differences on the greyscale channel).  Very low values
        indicate smooth/featureless images; very high values suggest
        noise or pure-text content.
    brightness_range : float
        Difference between the highest and lowest per-channel mean; a proxy
        for colour cast strength.
    n_unique_colors : int
        Number of distinct (R, G, B) triples in a 64×64 downsampled copy of
        the image.  Fewer than ~50 strongly suggests a synthetic, solid-fill,
        or heavily compressed placeholder.
    aspect : float
        Width-to-height ratio; extreme values can indicate banners or strips
        accidentally included in the dataset.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    dict or None
        Dictionary of computed statistics, or ``None`` if the file cannot
        be opened (corrupt / missing).
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        #  Per-channel RGB means 
        mean_r, mean_g, mean_b = (float(arr[:, :, c].mean()) for c in range(3))

        #  HSV saturation mean (colour richness) 
        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        saturation_mean = float(hsv[:, :, 1].mean())

        #  Grey fraction: proportion of near-achromatic pixels 
        # A pixel is considered "grey" when its max–min channel spread < 15.
        max_ch, min_ch = arr.max(axis=2), arr.min(axis=2)
        grey_fraction = float((max_ch - min_ch < 15).mean())

        #  Edge density: fraction of pixels with strong local gradient 
        # Finite-difference gradient on the luminance channel; pixels with
        # magnitude > 20 are counted as "edges".
        grey = np.array(img.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        edge_density = float((np.sqrt(gx**2 + gy**2) > 20).mean())

        #  Brightness range: spread between brightest and dimmest channel 
        brightness_range = float(max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b))

        #  Unique colour count at 64×64 resolution (synthetic-image proxy) 
        n_unique_colors = len(set(img.resize((64, 64), Image.BILINEAR).getdata()))

        return dict(mean_r=mean_r, mean_g=mean_g, mean_b=mean_b,
                     overall_mean=float(arr.mean()), overall_std=float(arr.std()),
                     saturation_mean=saturation_mean, grey_fraction=grey_fraction,
                     edge_density=edge_density, brightness_range=brightness_range,
                     n_unique_colors=n_unique_colors, aspect=img.width / img.height)

    except Exception:
        # Return None for any image that cannot be decoded; the caller skips it.
        return None


def detect_nonfood_pixel_outliers(
    df: pd.DataFrame,
    img_dir: str,
    out_dir: str = ".",
    n_fails_required: int = N_FAILS_DEFAULT,
    class_zscore_thr: float = CLASS_ZSCORE_THR,
    min_class_size: int = MIN_CLASS_SIZE,
) -> tuple:
    """
    Apply five complementary pixel-statistic criteria to flag images
    that are likely non-food, mislabelled, or otherwise unsuitable for
    training a 251-class food classifier.

    Unlike Stage 1 (which auto-deletes), flagged images here are written
    to a CSV for human review; no image is removed automatically.

    Flagging criteria (applied with OR logic — any one can flag an image):

    (a) **Standard bounds** (``FOOD_STATS_BOUNDS``):
        Four features — ``saturation_mean``, ``edge_density``,
        ``grey_fraction``, ``brightness_range`` — each have a [low, high]
        range calibrated for food photos.  Each out-of-range feature
        increments the image's failure counter.

    (b) **Food-safe guard**:
        Images that appear genuinely colourful (``saturation_mean`` ≥
        ``FOOD_SAFE_SATURATION_MIN``) *and* richly textured (``edge_density``
        ≤ ``FOOD_SAFE_EDGE_MAX``) are treated as likely food and require
        ``FOOD_SAFE_FAILURES_REQUIRED`` simultaneous failures before being
        flagged — a stricter threshold that reduces false positives on vivid,
        high-detail dishes.

    (c) **Extreme single violation** (``EXTREME_SINGLE``):
        A feature value so far outside its expected range that it alone is
        sufficient evidence; the image is flagged regardless of how other
        features look.

    (d) **Per-class z-score**:
        For each class with at least *min_class_size* images, compute a
        z-score for every feature relative to that class's mean and std.
        Images exceeding *class_zscore_thr* standard deviations on any
        feature are flagged — catching intra-class outliers (e.g. a photo
        of a plate with no food in a class that is otherwise consistent).
        Classes too small for reliable z-scores are skipped.

    (e) **Low unique-colour count**:
        Images with fewer than 50 distinct colours at 64×64 resolution are
        likely synthetic fills, solid-colour patches, or heavily artefacted;
        they are flagged regardless of other criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 1–cleaned manifest with ``image_id`` and ``label`` columns.
    img_dir : str
        Root directory for image files.
    out_dir : str
        Directory where the review CSV is saved.
    n_fails_required : int
        Minimum number of standard-bound failures (criterion a) needed to
        flag an image for non-food-safe images (default ``N_FAILS_DEFAULT``).
    class_zscore_thr : float
        Z-score threshold for per-class outlier detection (default
        ``CLASS_ZSCORE_THR = 4.5``).
    min_class_size : int
        Minimum number of images a class must have for per-class z-scores
        to be computed (default ``MIN_CLASS_SIZE = 20``).

    Returns
    -------
    stats_df : pd.DataFrame
        Full table of pixel statistics for every successfully processed image.
    flagged_df : pd.DataFrame
        Subset of *stats_df* for flagged images, augmented with columns
        ``fail_reasons`` (semicolon-separated list of triggered criteria)
        and ``n_fails`` (count of standard-bound failures), sorted by
        ``n_fails`` descending so the most suspicious images appear first.
    """
    #  Collect pixel statistics for every image 
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Pixel stats"):
        stats = pixel_stats(os.path.join(img_dir, str(row["image_id"])))
        if stats is None:
            continue  # skip images that could not be read (already caught in Stage 1)
        stats["image_id"], stats["label"] = row["image_id"], row["label"]
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df, stats_df.copy()

    # Initialise per-image failure tracking: boolean mask and reason lists.
    fail_mask = pd.Series(False, index=stats_df.index)
    fail_reasons = {i: [] for i in stats_df.index}

    #  Standard bounds: accumulate failures per out-of-range feature ────
    # Each feature is compared against its [lo, hi] range from FOOD_STATS_BOUNDS.
    # Violations are recorded as human-readable strings, e.g.
    # "saturation_mean=2.100[4.0, 248.0]".
    for stat, (lo, hi) in FOOD_STATS_BOUNDS.items():
        col = stats_df[stat]
        out = col < lo
        if hi is not None:
            out = out | (col > hi)
        for idx in stats_df.index[out]:
            fail_reasons[idx].append(f"{stat}={stats_df.at[idx, stat]:.3f}[{lo},{hi}]")

    # Tally total standard-bound failures per image.
    n_fails = pd.Series({i: len(v) for i, v in fail_reasons.items()}, index=stats_df.index)

    #  Food-safe guard: raise the bar for clearly food-like images 
    # An image is "food-safe" if it has both notable colour (high saturation)
    # and rich texture (moderate edge density) — hallmarks of real food photos.
    # Such images require more simultaneous failures before being flagged, to
    # avoid discarding legitimate but unusual food images.
    is_food_safe = (stats_df["saturation_mean"] >= FOOD_SAFE_SATURATION_MIN) & \
                   (stats_df["edge_density"] <= FOOD_SAFE_EDGE_MAX)
    required = np.where(is_food_safe, FOOD_SAFE_FAILURES_REQUIRED, n_fails_required)
    fail_mask |= (n_fails >= required)

    #  Extreme single violation: one very bad stat is enough to flag 
    # EXTREME_SINGLE uses tighter-than-normal bounds; exceeding them on even one
    # feature is treated as conclusive evidence of a non-food or degenerate image.
    for stat, (lo, hi) in EXTREME_SINGLE.items():
        col = stats_df[stat]
        extreme = pd.Series(False, index=stats_df.index)
        if lo is not None:
            extreme |= col < lo
        if hi is not None:
            extreme |= col > hi
        for idx in stats_df.index[extreme]:
            fail_reasons[idx].append(f"EXTREME:{stat}={stats_df.at[idx, stat]:.3f}")
        fail_mask |= extreme

    #  Per-class z-score: intra-class statistical outliers 
    # For classes with enough samples, compute each image's z-score relative to
    # its class's distribution for each feature.  A high z-score indicates the
    # image looks unlike its peers — useful for catching label errors and
    # cross-class contamination that global thresholds miss.
    class_sizes = stats_df.groupby("label").size()
    for stat in FOOD_STATS_BOUNDS:
        cstats = stats_df.groupby("label")[stat].agg(["mean", "std"])
        for idx, r in stats_df.iterrows():
            cls = r["label"]
            # Skip classes too small for meaningful statistics.
            if class_sizes.get(cls, 0) < min_class_size:
                continue
            cmean, cstd = cstats.at[cls, "mean"], cstats.at[cls, "std"]
            # Skip degenerate classes where all values are identical (std ≈ 0).
            if pd.isna(cstd) or cstd < 1e-6:
                continue
            z = abs(r[stat] - cmean) / cstd
            if z > class_zscore_thr:
                fail_reasons[idx].append(f"class_z:{stat}={z:.1f}sigma")
                fail_mask[idx] = True

    # Low unique-colour count: synthetic or solid-fill images 
    # After downsampling to 64×64, genuine photographs have hundreds of
    # distinct colours.  Fewer than 50 strongly suggests a generated,
    # solid-fill, or heavily quantised placeholder.
    synthetic = stats_df["n_unique_colors"] < 50
    for idx in stats_df.index[synthetic]:
        fail_reasons[idx].append(f"low_unique_colors={stats_df.at[idx, 'n_unique_colors']}")
    fail_mask |= synthetic

    # Assemble the flagged subset and sort by severity 
    flagged_df = stats_df[fail_mask].copy()
    flagged_df["fail_reasons"] = flagged_df.index.map(lambda i: "; ".join(fail_reasons[i]))
    flagged_df["n_fails"] = n_fails[fail_mask].values
    # Sort most suspicious (most failures) first to prioritise human review.
    flagged_df = flagged_df.sort_values("n_fails", ascending=False).reset_index(drop=True)

    print(f"\n[Stage 2] Flagged {len(flagged_df):,} / {len(stats_df):,} images")
    if not flagged_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "review_stage2_nonfood_pixelstats.csv")
        flagged_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    return stats_df, flagged_df


# STAGE 3 – Per-Class Embedding Outliers  (ResNet-50 + IsolationForest)

# Third and most powerful line of defence: deep semantic outlier detection
# using ImageNet-pretrained ResNet-50 features.  Unlike Stages 1–2, which
# rely on raw pixel properties, Stage 3 operates in the model's learned
# feature space, catching subtle semantic mismatches that pixel statistics
# cannot see (e.g. a food image that looks photometrically normal but
# belongs to the wrong class, or a highly realistic non-food image).
#
# Pipeline overview:
#   1. FlatFoodDataset wraps the manifest and applies standard ImageNet
#      preprocessing so every image is fed to ResNet-50 identically.
#   2. ResNet-50 (IMAGENET1K_V2 weights) is used as a frozen feature
#      extractor; its final FC layer is replaced with nn.Identity() to
#      obtain raw 2048-d global-average-pool embeddings.
#   3. Features are standardised (zero mean, unit variance) then compressed
#      to 128 PCA dimensions — enough to preserve semantic structure while
#      dramatically speeding up the IsolationForest fits.
#   4. A *global* IsolationForest is fit on all images at once; its decision
#      scores identify images that are anomalous across the entire dataset.
#   5. A *per-class* IsolationForest is fit independently for each class
#      (with ≥ min_class_size images); its scores identify images that are
#      anomalous within their own class — catching within-class semantic
#      drift and label errors invisible to the global model.
#   6. Two flagging thresholds are applied with OR logic:
#      - EXTREME global outlier (score < −0.20): flagged unconditionally.
#      - MODERATE: score below *global_thr* AND below *per_class_thr*
#        simultaneously — requiring both models to agree reduces false
#        positives from images that look unusual globally but fit their
#        class well.
#
# Flagged images are exported to `review_stage3_nonfood_embedding.csv`
# (sorted by global anomaly score, most anomalous first) for human review.



class FlatFoodDataset(Dataset):
    """PyTorch Dataset that wraps the flat image manifest and applies
    standard ImageNet preprocessing for ResNet-50 feature extraction.

    Each item returned is a triple ``(tensor, label_int, image_id_str)``
    so that the DataLoader simultaneously delivers model-ready tensors and
    the metadata needed to map features back to the original manifest rows.

    If an image file cannot be opened (e.g. it was corrupt but somehow
    survived Stage 1), a black 224×224 placeholder is used so that the
    DataLoader batch is never interrupted.  Such placeholder features will
    appear as extreme outliers in the IsolationForest and will be flagged
    automatically.

    Class attribute
    ---------------
    TFM : torchvision.transforms.Compose
        Standard ImageNet preprocessing: resize shortest side to 256 px,
        centre-crop to 224×224, convert to tensor, normalise with
        ImageNet channel means and standard deviations.
    """

    # Standard ImageNet preprocessing pipeline for ResNet-50 input.
    TFM = transforms.Compose([
        transforms.Resize(256),                                    # scale shortest side to 256 px
        transforms.CenterCrop(224),                                # crop to 224×224 (ResNet input size)
        transforms.ToTensor(),                                     # [0,255] uint8 → [0,1] float32 tensor
        transforms.Normalize([0.485, 0.456, 0.406],               # subtract ImageNet channel means
                             [0.229, 0.224, 0.225]),               # divide by ImageNet channel stds
    ])

    def __init__(self, df, img_dir):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Manifest with ``image_id`` and ``label`` columns.
        img_dir : str
            Root directory containing the image files.
        """
        self.df, self.img_dir = df.reset_index(drop=True), img_dir

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """Load, preprocess, and return the i-th image with its metadata.

        Parameters
        ----------
        i : int
            Integer index into the manifest.

        Returns
        -------
        tensor : torch.Tensor
            Preprocessed image tensor of shape (3, 224, 224).
        label : int
            Integer class label for the image.
        image_id : str
            Filename / identifier used to match back to the manifest.
        """
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback: all-black image ensures the batch shape is preserved;
            # the resulting near-zero embedding will be flagged as anomalous.
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.TFM(img), int(row["label"]), str(row["image_id"])


def detect_nonfood_embedding_outliers(
    df: pd.DataFrame,
    img_dir: str,
    out_dir: str = ".",
    device: str = "mps",          # set "mps" or "mps" if available
    batch_size: int = 64,
    num_workers: int = 0,
    global_thr: float = -0.12,
    per_class_thr: float = -0.15,
    min_class_size: int = 10,
) -> tuple:
    """Extract ResNet-50 embeddings for all images and flag those that are
    semantic outliers under a global and/or per-class IsolationForest model.

    Two complementary anomaly scores are computed:

    * **Global score** — fit on the entire dataset; images anomalous across
      all 251 classes (e.g. non-food content or completely wrong domain) will
      receive very negative scores.

    * **Per-class score** — fit independently for each class; images that
      look unusual *within their own class* (e.g. a different dish mislabelled,
      or an atypical view) will receive very negative per-class scores even
      if they look acceptable globally.

    Flagging logic (OR):
      - ``global_score < −0.20``  → extreme global outlier, flagged alone.
      - ``global_score < global_thr AND per_class_score < per_class_thr``
        → moderate outlier confirmed by both models.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 1–cleaned manifest with ``image_id`` and ``label`` columns.
    img_dir : str
        Root directory for image files.
    out_dir : str
        Directory where the review CSV is saved.
    device : str
        PyTorch device string (``"mps"``, ``"mps"``, or ``"cpu"``).
    batch_size : int
        Number of images per DataLoader batch during feature extraction.
    num_workers : int
        DataLoader worker processes (0 = main process only; safe default).
    global_thr : float
        IsolationForest decision-function threshold for the global model;
        scores below this are considered moderately anomalous (default −0.12).
    per_class_thr : float
        IsolationForest decision-function threshold for per-class models;
        scores below this (in combination with *global_thr*) flag an image
        (default −0.15).
    min_class_size : int
        Minimum images a class must have for a per-class model to be fit;
        smaller classes are skipped and their per-class scores left as +∞
        (default 10).

    Returns
    -------
    feats_pca : np.ndarray, shape (N, 128)
        PCA-compressed, standardised embeddings for all N images.
    all_ids : list of str
        Image IDs in the same order as *feats_pca* rows.
    global_scores : np.ndarray, shape (N,)
        Global IsolationForest decision scores (more negative = more anomalous).
    per_class_scores : np.ndarray, shape (N,)
        Per-class IsolationForest decision scores (+∞ for skipped classes).
    outlier_df : pd.DataFrame
        Flagged images from *df* augmented with ``anomaly_score_global`` and
        ``anomaly_score_per_class``, sorted by ``anomaly_score_global`` ascending.
    """
    #  Load frozen ResNet-50 backbone
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()   # replace classification head with identity to get raw pool features
    backbone.eval().to(device)

    # Stream all images through the backbone to collect embeddings 
    dl = DataLoader(FlatFoodDataset(df, img_dir), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=(device == "mps"))

    feats_list, all_ids, all_labels = [], [], []
    with torch.no_grad():   # disable gradient computation for inference efficiency
        for imgs, labels, ids in tqdm(dl, desc="[Stage 3] Feature extraction"):
            feats_list.append(backbone(imgs.to(device)).cpu().numpy())
            all_ids.extend(ids)
            all_labels.extend(labels.tolist())
    feats = np.vstack(feats_list)        
    labels_arr = np.array(all_labels)

    # Standardise and compress features with PCA 
    # StandardScaler removes per-feature mean/variance differences that would
    # otherwise bias the IsolationForest.  PCA to 128 components retains the
    # bulk of semantic variance while making the IF fits much faster.
    print("[Stage 3] Fitting global IsolationForest...")
    feats_scaled = StandardScaler().fit_transform(feats)
    feats_pca = PCA(n_components=min(128, feats_scaled.shape[1]), random_state=42).fit_transform(feats_scaled)

    #  Global IsolationForest — dataset-wide anomaly scoring 
    # contamination="auto" lets sklearn estimate the contamination fraction;
    # n_jobs=-1 uses all available CPU cores for parallel tree building.
    clf_global = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    clf_global.fit(feats_pca)
    global_scores = clf_global.decision_function(feats_pca)   # negative = anomalous

    #  Per-class IsolationForest — intra-class anomaly scoring 
    # A separate IF is fit for each class, allowing the model to learn what
    # "normal" looks like within a specific food category.  n_estimators is
    # scaled to class size (capped at 200) so small classes don't overfit.
    print("[Stage 3] Fitting per-class IsolationForest models...")
    per_class_scores = np.full(len(all_ids), np.inf)   # default +∞ for skipped/small classes
    for lbl in tqdm(np.unique(labels_arr), desc="  per-class IF"):
        mask = labels_arr == lbl
        if mask.sum() < min_class_size:
            continue   # too few images for a meaningful model
        n_est = max(50, min(200, mask.sum()))   # scale tree count to available samples
        clf_c = IsolationForest(n_estimators=n_est, contamination="auto", random_state=42, n_jobs=-1)
        clf_c.fit(feats_pca[mask])
        per_class_scores[mask] = clf_c.decision_function(feats_pca[mask])

    # Combine global and per-class scores to produce final flags 
    # Two conditions are OR'd:
    #   • extreme_global: score < −0.20  → clear dataset-wide outlier; flag alone.
    #   • moderate: both global AND per-class below their respective thresholds
    #     → requires agreement from both models, reducing false positives.
    extreme_global = global_scores < -0.20
    moderate = (global_scores < global_thr) & (per_class_scores < per_class_thr)
    outlier_mask = extreme_global | moderate

    # Build the output DataFrame with anomaly scores attached 
    outlier_ids = [all_ids[i] for i in np.where(outlier_mask)[0]]
    outlier_df = df[df["image_id"].astype(str).isin(outlier_ids)].copy()
    g_map = {all_ids[i]: float(global_scores[i]) for i in np.where(outlier_mask)[0]}
    c_map = {all_ids[i]: float(per_class_scores[i]) for i in np.where(outlier_mask)[0]}
    outlier_df["anomaly_score_global"] = outlier_df["image_id"].astype(str).map(g_map)
    outlier_df["anomaly_score_per_class"] = outlier_df["image_id"].astype(str).map(c_map)
    # Sort most anomalous first so reviewers see the clearest outliers at the top.
    outlier_df = outlier_df.sort_values("anomaly_score_global").reset_index(drop=True)

    print(f"\n[Stage 3] Flagged {len(outlier_df):,} images "
          f"(global thr={global_thr}, per-class thr={per_class_thr})")
    if not outlier_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "review_stage3_nonfood_embedding.csv")
        outlier_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    return feats_pca, all_ids, global_scores, per_class_scores, outlier_df


# VISUALISATION HELPERS

# Three plot functions support human review of pipeline outputs:
#   • load_image_safe      – fault-tolerant single-image loader for plotting.
#   • visualize_flagged_images    – compact grid of flagged images with
#                                   colour-coded confidence borders.
#   • visualize_outlier_decisions – per-class scatter plots of pixel stats
#                                   with flagged images marked as red X.
#   • plot_anomaly_score_distribution – histogram of IsolationForest scores
#                                   with the flagging threshold indicated.
# All plots use the dark theme defined by BG / PANEL / ACCENT / FLAG / LINE /
# SPINE at the top of the file.



def load_image_safe(path: str, size: int = 300) -> Optional[np.ndarray]:
    """Load a single image from disk and resize it to fit within *size* pixels
    on its longest dimension, returning a uint8 RGB numpy array.

    The thumbnail operation preserves the original aspect ratio.  Returns
    ``None`` instead of raising if the file cannot be read, so callers can
    render a placeholder rather than crashing during visualisation.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.
    size : int
        Maximum pixel dimension (width or height) of the returned array
        (default 300).

    Returns
    -------
    np.ndarray or None
        RGB array of shape (H, W, 3) with dtype uint8, or ``None`` on error.
    """
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)   # in-place resize preserving aspect ratio
        return np.array(img)
    except Exception:
        return None   # return None so the caller can display a "load error" placeholder


def visualize_flagged_images(flagged_df, img_dir, title="Flagged images",
                              max_images=48, cols=8, save_path=None) -> None:
    """Render a dark-themed grid of up to *max_images* flagged images with
    colour-coded spine borders indicating detection confidence.

    Border colours encode confidence level, making the most suspicious
    images immediately identifiable at a glance:
      - **Red**   (``FLAG``)  : high confidence — 3+ standard-bound failures
                                OR an EXTREME single-stat violation.
      - **Yellow** (``LINE``) : moderate confidence — exactly 2 failures.
      - **Blue**  (``#44aaff``): low confidence — 0–1 failure (flagged by
                                 z-score, unique-colour count, or embedding).

    Each cell also displays the class label and the first 50 characters of
    the most informative reason string (``fail_reasons`` for Stage 2,
    ``anomaly_score_global`` for Stage 3).

    Parameters
    ----------
    flagged_df : pd.DataFrame
        Output of ``detect_nonfood_pixel_outliers`` or
        ``detect_nonfood_embedding_outliers``; must contain ``image_id``
        and ``label`` columns.
    img_dir : str
        Root directory for image files.
    title : str
        Figure super-title displayed above the grid.
    max_images : int
        Maximum number of images rendered (first *max_images* rows of
        *flagged_df* are used, so sorting by severity before calling is
        recommended; default 48).
    cols : int
        Number of grid columns (default 8).
    save_path : str or None
        If provided, the figure is saved to this path at 120 dpi instead of
        being displayed interactively.
    """
    if flagged_df.empty:
        print("Nothing to plot - dataframe is empty.")
        return

    subset = flagged_df.head(max_images)
    n = len(subset)
    rows = max(1, (n + cols - 1) // cols)   # ceiling division for row count
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), facecolor=BG)
    fig.suptitle(f"{title} ({n} of {len(flagged_df)} shown)", fontsize=12, color="white", fontweight="bold", y=1.01)
    axes = np.array(axes).reshape(-1)   # flatten to 1-D for uniform indexing

    for i, (_, row) in enumerate(subset.iterrows()):
        ax = axes[i]
        ax.set_facecolor(PANEL)

        # Load image; display a text placeholder if loading fails.
        arr = load_image_safe(os.path.join(img_dir, str(row["image_id"])))
        if arr is not None:
            ax.imshow(arr)
        else:
            ax.text(0.5, 0.5, "load\nerror", ha="center", va="center", fontsize=6, color="salmon", transform=ax.transAxes)

        # Colour-code the subplot border by confidence level.
        n_fails = row.get("n_fails", 1)
        reasons = str(row.get("fail_reasons", ""))
        border = FLAG if (n_fails >= 3 or "EXTREME" in reasons) else (LINE if n_fails == 2 else "#44aaff")
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(2)

        # Show the most informative reason string as the subplot title.
        reason = ""
        for col in ("fail_reasons", "anomaly_score_global"):
            if col in row.index and pd.notna(row[col]):
                reason = str(row[col])[:50]
                break
        ax.set_title(f"cls {row['label']}\n{reason}", fontsize=5, color="white", pad=2)
        ax.axis("off")

    # Hide unused subplot cells (when n < rows × cols).
    for j in range(n, len(axes)):
        axes[j].axis("off"); axes[j].set_facecolor(BG)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        print(f"  -> saved grid: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_outlier_decisions(stats_df, flagged_df, label, save_path=None) -> None:
    """Plot per-class scatter charts for each pixel-stat feature, overlaying
    flagged images as red X markers to support false-positive identification.

    One subplot is produced per feature in ``FOOD_STATS_BOUNDS``.  Each subplot
    shows a jittered strip chart of feature values for all images in the class,
    with the configured [lo, hi] bounds drawn as dashed yellow lines.
    Flagged images are overlaid as large red X markers; a red X sitting well
    inside the normal cluster of the same feature is a strong indicator of a
    false positive driven by a *different* feature or detection criterion.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Full pixel-statistics table returned by
        ``detect_nonfood_pixel_outliers`` (all images, not just flagged ones).
    flagged_df : pd.DataFrame
        Flagged-image subset, also returned by
        ``detect_nonfood_pixel_outliers``.
    label : int or str
        The class label to visualise; only rows with this label are plotted.
    save_path : str or None
        If provided, the figure is saved here instead of being shown
        interactively.
    """
    class_stats = stats_df[stats_df["label"] == label]
    class_flagged = flagged_df[flagged_df["label"] == label] if not flagged_df.empty else pd.DataFrame()
    if class_stats.empty:
        print(f"No stats for label={label}.")
        return

    # Only plot features present in both FOOD_STATS_BOUNDS and the DataFrame columns.
    stat_cols = [c for c in FOOD_STATS_BOUNDS if c in class_stats.columns]
    fig, axes = plt.subplots(1, len(stat_cols), figsize=(len(stat_cols) * 4, 5), facecolor=BG)
    if len(stat_cols) == 1:
        axes = [axes]   # ensure axes is always iterable

    for ax, col in zip(axes, stat_cols):
        ax.set_facecolor(PANEL)
        lo, hi = FOOD_STATS_BOUNDS.get(col, (None, None))

        # Background scatter: all images in the class (jittered x-axis for readability).
        ax.scatter(np.random.uniform(-0.2, 0.2, len(class_stats)), class_stats[col],
                   color=ACCENT, alpha=0.4, s=10, label=f"all ({len(class_stats)})")

        # Overlay: flagged images as red X markers at higher z-order.
        if not class_flagged.empty:
            ax.scatter(np.random.uniform(-0.2, 0.2, len(class_flagged)), class_flagged[col],
                       color=FLAG, s=50, marker="X", zorder=5, label=f"flagged ({len(class_flagged)})")

        # Draw the configured [lo, hi] threshold lines for reference.
        if lo is not None:
            ax.axhline(lo, color=LINE, ls="--", lw=1, alpha=0.8)
        if hi is not None:
            ax.axhline(hi, color=LINE, ls="--", lw=1, alpha=0.8, label="threshold")

        ax.set_title(col, color="white", fontsize=10)
        ax.set_xticks([]); ax.tick_params(colors="white"); ax.spines[:].set_color(SPINE)
        ax.legend(fontsize=7, labelcolor="white", facecolor=BG, edgecolor=SPINE)

    fig.suptitle(f"Class {label} - ({len(class_flagged)} flagged / {len(class_stats)} total)",
                  color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()


def plot_anomaly_score_distribution(scores, score_thr=-0.12, title="Anomaly score distribution", save_path=None) -> None:
    """Plot a histogram of IsolationForest decision scores with the flagging
    threshold marked as a vertical dashed line.

    Useful for calibrating *global_thr* and *per_class_thr*: the natural
    gap or shoulder in the score distribution often indicates a better
    threshold than the default, and the count of flagged images is shown
    directly on the plot to aid decision-making.

    IsolationForest scores are normalised so that 0 is the theoretical
    boundary between inliers and outliers; more negative values indicate
    greater anomaly.  In practice, the bulk of clean food images cluster
    slightly above 0, while genuine outliers form a left tail.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of IsolationForest decision-function values, one per image.
    score_thr : float
        Threshold below which images are considered flagged; drawn as a
        vertical dashed red line (default −0.12).
    title : str
        Title displayed above the histogram.
    save_path : str or None
        If provided, the figure is saved here instead of being shown
        interactively.
    """
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
    ax.set_facecolor(PANEL)

    # Draw the score histogram with 200 bins for fine-grained resolution.
    ax.hist(scores, bins=200, color="#5555bb", alpha=0.8, edgecolor="none")

    # Mark the flagging threshold and annotate the count of images below it.
    ax.axvline(score_thr, color=FLAG, lw=2, ls="--", label=f"threshold = {score_thr}")
    flagged = (scores < score_thr).sum()
    ax.text(score_thr, ax.get_ylim()[1] * 0.85, f"{flagged:,} flagged", color=FLAG, ha="right", fontsize=10)

    ax.set_xlabel("Anomaly score (more negative = more anomalous)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title(title, color="white", fontweight="bold")
    ax.tick_params(colors="white"); ax.spines[:].set_color(SPINE)
    ax.legend(labelcolor="white", facecolor=BG, edgecolor=SPINE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close()


# Apply Review Decisions

# After human review of the Stage 2 and Stage 3 CSVs, this function applies
# the confirmed removal decisions to produce the final cleaned manifest.
# The expected workflow is:
#   1. Open `review_stage2_nonfood_pixelstats.csv` in a spreadsheet viewer.
#   2. Delete rows for images you want to **keep** (false positives).
#   3. Save the remaining rows as `confirmed_remove_stage2.csv`.
#   4. Repeat for the Stage 3 CSV → `confirmed_remove_stage3.csv`.
#   5. Call `apply_review_decisions(df, [...csv paths...])`.
# The function is robust to missing or malformed CSVs — it warns and skips
# them rather than crashing, so partial decisions are also supported.


def apply_review_decisions(df: pd.DataFrame, remove_csv_paths: List[str], output_csv: str = "train_labels_clean.csv") -> pd.DataFrame:
    """Remove reviewer-confirmed outliers from the dataset manifest and write
    the cleaned result to a new CSV.

    This is the final step of the human-in-the-loop review workflow.  Each
    CSV in *remove_csv_paths* should contain an ``image_id`` column listing
    the images to be removed.  Rows not present in any removal CSV are kept.

    Missing files or files lacking an ``image_id`` column are skipped with a
    warning so that partial review sessions (e.g. only Stage 2 completed) do
    not block progress.

    Parameters
    ----------
    df : pd.DataFrame
        The Stage 1–cleaned manifest (returned by ``run_outlier_pipeline``
        or ``image_integrity_audit``).
    remove_csv_paths : list of str
        Paths to one or more CSVs of confirmed-to-remove images.  Typically
        ``["confirmed_remove_stage2.csv", "confirmed_remove_stage3.csv"]``.
    output_csv : str
        Path for the final cleaned manifest CSV (default
        ``"train_labels_clean.csv"``).

    Returns
    -------
    final_df : pd.DataFrame
        Cleaned manifest with confirmed outliers removed and a fresh
        contiguous integer index.
    """
    # Collect all image IDs that should be removed across all supplied CSVs.
    ids_to_remove = set()
    for path in remove_csv_paths:
        if not os.path.exists(path):
            print(f"  Warning: '{path}' not found - skipped.")
            continue
        tmp = pd.read_csv(path)
        if "image_id" not in tmp.columns:
            print(f"  Warning: '{path}' has no 'image_id' column - skipped.")
            continue
        ids_to_remove.update(tmp["image_id"].astype(str).tolist())

    # Filter the manifest, resetting the index for downstream compatibility.
    before = len(df)
    final_df = df[~df["image_id"].astype(str).isin(ids_to_remove)].reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)

    print(f"\n[apply_review_decisions] Removed {before - len(final_df):,} confirmed outliers.")
    print(f"  Final dataset: {len(final_df):,} images, {final_df['label'].nunique()} classes")
    print(f"  -> saved: {output_csv}")
    return final_df


# Per-Class Audit Summary Report

# Ties the 3 stages together into one per-class trail: how many images each
# class started with, how many were auto-removed at Stage 1, how many are
# flagged (not yet removed) at Stages 2/3, and — once reviewer decisions are
# applied — the final surviving count. This is what confirms the audit did
# not gut any of the ~34-image tail classes below a usable minimum, and it
# is the CSV the report's "Data Preprocessing" section can cite directly.


def audit_summary_report(
    raw_df: pd.DataFrame,
    stage1_df: pd.DataFrame,
    flagged2_df: pd.DataFrame,
    flagged3_df: pd.DataFrame,
    final_df: Optional[pd.DataFrame] = None,
    num_classes: int = 251,
    out_dir: str = ".",
    min_remaining: int = 15,
) -> pd.DataFrame:
    """Build and save the per-class outlier-audit trail across all 3 stages.

    Parameters
    ----------
    raw_df : pd.DataFrame
        The original, unfiltered manifest (before Stage 1).
    stage1_df : pd.DataFrame
        Manifest after Stage 1 auto-removal (``image_integrity_audit`` output).
    flagged2_df, flagged3_df : pd.DataFrame
        Stage 2 / Stage 3 flagged-for-review subsets (NOT yet removed).
    final_df : pd.DataFrame or None
        The reviewer-confirmed final manifest (``apply_review_decisions``
        output / ``train_labels_clean.csv``), if review has been completed.
        When omitted, a WORST-CASE final count is estimated by assuming every
        flagged image (Stage 2 or Stage 3, whichever flags more per class) is
        eventually removed — a conservative lower bound to sanity-check
        BEFORE committing to manual review decisions.
    num_classes : int
        Total number of classes (251, fixed by the spec).
    out_dir : str
        Directory the summary CSV is written to.
    min_remaining : int
        Classes at or below this many surviving images are printed as a
        warning (default 15 — half the smallest known class size of ~34).

    Returns
    -------
    pd.DataFrame
        One row per class with columns ``label``, ``raw_count``,
        ``stage1_removed``, ``stage1_remaining``, ``stage2_flagged_for_review``,
        ``stage3_flagged_for_review``, plus either ``final_count`` (if
        *final_df* given) or ``final_count_worst_case``, and
        ``pct_of_raw_remaining``.
    """
    idx = range(num_classes)
    raw_counts = raw_df["label"].value_counts().reindex(idx, fill_value=0)
    stage1_counts = stage1_df["label"].value_counts().reindex(idx, fill_value=0)
    stage2_flags = (flagged2_df["label"].value_counts().reindex(idx, fill_value=0)
                    if flagged2_df is not None and not flagged2_df.empty
                    else pd.Series(0, index=idx))
    stage3_flags = (flagged3_df["label"].value_counts().reindex(idx, fill_value=0)
                    if flagged3_df is not None and not flagged3_df.empty
                    else pd.Series(0, index=idx))

    report = pd.DataFrame({
        "label": list(idx),
        "raw_count": raw_counts.values,
        "stage1_removed": (raw_counts - stage1_counts).values,
        "stage1_remaining": stage1_counts.values,
        "stage2_flagged_for_review": stage2_flags.values,
        "stage3_flagged_for_review": stage3_flags.values,
    })

    if final_df is not None:
        final_counts = final_df["label"].value_counts().reindex(idx, fill_value=0)
        report["final_count"] = final_counts.values
        count_col = "final_count"
    else:
        # Worst case: assume every flagged image is eventually confirmed removed.
        max_flagged = np.maximum(stage2_flags.values, stage3_flags.values)
        report["final_count_worst_case"] = np.clip(stage1_counts.values - max_flagged, 0, None)
        count_col = "final_count_worst_case"

    report["pct_of_raw_remaining"] = (
        report[count_col] / report["raw_count"].replace(0, np.nan) * 100
    ).round(1)

    gutted = report[report[count_col] < min_remaining].sort_values(count_col)
    tag = " (worst-case estimate — review not yet applied)" if count_col.endswith("worst_case") else ""
    print(f"\n[audit_summary_report] {len(gutted)} / {num_classes} classes below "
          f"{min_remaining} images after cleaning{tag}")
    if not gutted.empty:
        print(gutted[["label", "raw_count", count_col, "pct_of_raw_remaining"]].to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "outlier_summary_by_class.csv")
    report.to_csv(path, index=False)
    print(f"  -> saved: {path}")
    return report



# Master Pipeline

# Orchestrates all three stages in sequence, handling column normalisation,
# stage skipping, and summary reporting.  This is the single entry point
# intended for typical usage; individual stage functions can also be called
# directly for more granular control.

def run_outlier_pipeline(
    csv_path: str,
    img_dir: str,
    out_dir: str = ".",
    device: str = "mps",
    global_thr: float = -0.12,
    per_class_thr: float = -0.15,
    n_fails_required: int = N_FAILS_DEFAULT,
    skip_stage3: bool = False,
) -> tuple:
    """Run the full 3-stage outlier detection pipeline on a food image dataset.

    Stages executed in order:

    1. **Integrity audit** (``image_integrity_audit``) — unconditional removal
       of missing, corrupt, undersized, near-black, near-white-blank, and
       low-contrast images.  Survivors form the working manifest for Stages 2–3.

    2. **Pixel-statistics detection** (``detect_nonfood_pixel_outliers``) —
       lightweight CPU-only analysis of colour, texture, and uniqueness
       statistics; flagged images are written to CSV for human review.

    3. **Embedding-based detection** (``detect_nonfood_embedding_outliers``) —
       ResNet-50 feature extraction followed by global and per-class
       IsolationForest anomaly scoring; flagged images are written to a
       separate CSV.  Can be skipped by setting *skip_stage3=True* (e.g.
       when no GPU is available or for a quick first pass).

    The function normalises common column-name variants (e.g. ``filename``,
    ``class``) to the canonical ``image_id`` / ``label`` names expected by all
    downstream functions, and raises a ``ValueError`` if neither can be found.

    Parameters
    ----------
    csv_path : str
        Path to the training manifest CSV (must have image-ID and label columns).
    img_dir : str
        Root directory where image files are stored.
    out_dir : str
        Directory for all output CSVs (created if absent).
    device : str
        PyTorch device for Stage 3 feature extraction (``"mps"``, ``"mps"``,
        or ``"cpu"``).
    global_thr : float
        Global IsolationForest threshold for Stage 3 (default −0.12).
    per_class_thr : float
        Per-class IsolationForest threshold for Stage 3 (default −0.15).
    n_fails_required : int
        Minimum pixel-stat bound failures for Stage 2 flagging
        (default ``N_FAILS_DEFAULT = 3``).
    skip_stage3 : bool
        If ``True``, skip the embedding stage entirely and return empty
        arrays/DataFrames for its outputs (default ``False``).

    Returns
    -------
    df : pd.DataFrame
        Stage 1–cleaned manifest (images surviving integrity checks).
    stats_df : pd.DataFrame
        Full pixel-statistics table from Stage 2.
    flagged2 : pd.DataFrame
        Stage 2 flagged images for human review.
    feats : np.ndarray
        PCA-compressed ResNet-50 embeddings; empty array if Stage 3 skipped.
    image_ids : list of str
        Image IDs corresponding to *feats* rows; empty list if Stage 3 skipped.
    global_scores : np.ndarray
        Global IsolationForest scores from Stage 3; empty array if skipped.
    per_class_scores : np.ndarray
        Per-class IsolationForest scores from Stage 3; empty array if skipped.
    flagged3 : pd.DataFrame
        Stage 3 flagged images for human review; empty DataFrame if skipped.
    """
    #  Load and normalise the manifest CSV 
    # Accept a variety of common column names for image ID and class label to
    # reduce friction with different dataset conventions.
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        if col.lower() in ("image_id", "img_id", "img_name", "filename", "file_name", "image"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "class_id", "category", "target"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Cannot find image_id/label columns. Found: {df.columns.tolist()}")

    print(f"\nLoaded: {len(df):,} images - {df['label'].nunique()} classes")
    print("-" * 60)

    #  Stage 1: integrity & blank audit (auto-removes bad images) 
    df, _ = image_integrity_audit(df, img_dir, out_dir=out_dir)

    #  Stage 2: pixel-statistics outlier detection (flags for review) 
    stats_df, flagged2 = detect_nonfood_pixel_outliers(df, img_dir, out_dir=out_dir, n_fails_required=n_fails_required)

    #  Stage 3: embedding-based outlier detection (optional, GPU-accelerated)
    if skip_stage3:
        print("\n[Stage 3] Skipped (skip_stage3=True).")
        feats, image_ids, global_scores, per_class_scores, flagged3 = np.array([]), [], np.array([]), np.array([]), pd.DataFrame()
    else:
        feats, image_ids, global_scores, per_class_scores, flagged3 = detect_nonfood_embedding_outliers(
            df, img_dir, out_dir=out_dir, device=device, global_thr=global_thr, per_class_thr=per_class_thr
        )

    #  Summary report 
    print(f"\n{'-' * 60}\nPIPELINE COMPLETE\n{'-' * 60}")
    print(f"  After Stage 1      : {len(df):,}")
    print(f"  Stage 2 for review : {len(flagged2):,}")
    if not skip_stage3:
        print(f"  Stage 3 for review : {len(flagged3):,}")
    print(f"{'-' * 60}\n")

    return df, stats_df, flagged2, feats, image_ids, global_scores, per_class_scores, flagged3