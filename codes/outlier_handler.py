"""
outlier_handler.py
==================
Outlier detection pipeline for a flat-folder food image dataset (251 classes).

Goal
----
Detect ONLY:
  1. Blank / near-black / near-white images (no real content at all).
  2. Non-food images (stock watermarks, cartoons, screenshots, solid
     backgrounds, wrong-domain photos that slipped through scraping).

Mislabelled-but-valid food images and label-noise are NOT the target
here — Cleanlab / embedding stages have been removed accordingly.

Expected CSV schema
-------------------
  image_id  : filename of the image  e.g. "00001.jpg"
  label     : integer class index    e.g. 0-250

Expected directory layout
--------------------------
  train_set/
      00001.jpg
      00002.jpg
      ...   ← all images in ONE flat folder, no sub-folders

Pipeline stages
---------------
  Stage 1 – File integrity + blank/background audit   (CPU, ~3 min)  → auto-remove
             Catches: missing files, truncated downloads,
             corrupt JPEGs, pure-black, pure-white, near-uniform
             single-colour backgrounds (no food content visible).

  Stage 2 – Non-food domain detector                  (CPU, ~5 min)  → review CSV
             Catches: watermarked stock photos, cartoon illustrations,
             recipe screenshots / text-heavy images, solid-colour
             backgrounds with thin borders that pass Stage 1.
             Uses pixel statistics that deviate sharply from ALL food
             classes combined, not just the per-class mean.

  Stage 3 – Embedding-space non-food detector         (GPU/MPS, ~20 min) → review CSV
             Extracts ResNet-50 features and flags images whose embedding
             sits far from the *global* food-image manifold (one global
             IsolationForest, not per-class), making it sensitive to
             images that simply do not look like food at all.

Usage
-----
  from outlier_handler import run_outlier_pipeline, plot_stage2_outlier_context

  df, stats_df, flagged_df, feats, image_ids, embed_outlier_df = run_outlier_pipeline(
      csv_path    = "train_labels.csv",
      img_dir     = "train_set/",
      num_classes = 251,
      device      = "mps",   # "cuda" on Linux/Windows, "mps" on Apple Silicon
  )

  # Visualise Stage 2 (pixel-stat) outliers for a given class
  plot_stage2_outlier_context(stats_df, flagged_df, label=12)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# STAGE 1 – File integrity + blank / background audit
# Catches: missing files, truncated downloads, corrupt JPEGs,
#          pure-black, pure-white, near-uniform single-colour images
#          (no food content visible at all)
# Decision: AUTO-REMOVE — every flagged image is either unreadable
#           or carries zero usable visual information
# ══════════════════════════════════════════════════════════════════════

def image_integrity_audit(
    df:         pd.DataFrame,
    img_dir:    str,
    min_bytes:  int   = 1_500,
    min_std:    float = 8.0,
    black_thresh: float = 15.0,
    white_thresh: float = 240.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan every image in the dataframe and flag integrity failures and
    blank / near-uniform backgrounds.

    Checks performed
    ----------------
    1. missing       – file does not exist on disk at all
    2. truncated     – file exists but is smaller than min_bytes
                       (partial download / write interrupted mid-save)
    3. corrupt       – PIL cannot decode the file at all
    4. near-black    – mean pixel value across all channels < black_thresh
                       (blank black background, lens-cap shot, etc.)
    5. near-white    – mean pixel value across all channels > white_thresh
                       (overexposed, blank white background, etc.)
    6. low-contrast  – pixel std < min_std even though mean is mid-range
                       (solid-colour background with no discernible content,
                        slightly off-white/off-grey uniform images)

    Why auto-remove is safe here
    ----------------------------
    Every flagged image is either unreadable or contains no real food
    content. No valid food photo will be a near-uniform field of colour
    or have a pixel std below 8.0 at 8-bit scale.

    Parameters
    ----------
    df            : DataFrame with columns ['image_id', 'label']
    img_dir       : path to the flat image folder  (train_set/)
    min_bytes     : minimum acceptable file size in bytes
    min_std       : minimum pixel standard deviation (low = near-uniform)
    black_thresh  : mean brightness below this → flagged as near-black
    white_thresh  : mean brightness above this → flagged as near-white

    Returns
    -------
    clean_df   : df with all flagged rows removed, index reset
    issues_df  : flagged rows with an extra 'reason' column
                 → also saved to removed_stage1_integrity.csv
    """
    issues = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Integrity + blank audit"):
        path   = os.path.join(img_dir, str(row["image_id"]))
        reason = None

        # Check 1: file exists
        if not os.path.exists(path):
            reason = "missing"

        # Check 2: file is not truncated
        elif os.path.getsize(path) < min_bytes:
            reason = f"truncated(bytes={os.path.getsize(path)})"

        else:
            try:
                arr  = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
                mean = arr.mean()
                std  = arr.std()

                # Check 3: near-black background (no content)
                if mean < black_thresh:
                    reason = f"near_black(mean={mean:.1f})"

                # Check 4: near-white / overexposed background (no content)
                elif mean > white_thresh:
                    reason = f"near_white(mean={mean:.1f})"

                # Check 5: low contrast / near-uniform colour — solid background
                elif std < min_std:
                    reason = f"low_contrast(mean={mean:.0f}, std={std:.1f})"

            except Exception as ex:
                reason = f"corrupt: {ex}"

        if reason:
            issues.append({
                "image_id": row["image_id"],
                "label":    row["label"],
                "reason":   reason,
            })

    # Build output dataframes
    issues_df = (
        pd.DataFrame(issues)
        if issues
        else pd.DataFrame(columns=["image_id", "label", "reason"])
    )
    clean_df = (
        df[~df["image_id"].isin(issues_df["image_id"])]
        .reset_index(drop=True)
    )

    # Console report
    print(f"\n[Stage 1] Removed {len(issues_df):,} / {len(df):,} images")
    if not issues_df.empty:
        summary = issues_df["reason"].str.split("(").str[0].value_counts()
        print(summary.to_string())
        issues_df.to_csv("../results/removed_stage1_integrity.csv", index=False)
        print("  → saved: removed_stage1_integrity.csv")

    return clean_df, issues_df


# ══════════════════════════════════════════════════════════════════════
# STAGE 2 – Non-food domain detector (pixel statistics)
# Catches: watermarked stock photos, cartoon illustrations,
#          recipe screenshots / text-heavy images, solid-colour
#          backgrounds with thin content strips that passed Stage 1
# Decision: REVIEW CSV — do NOT auto-remove; inspect before deleting
# ══════════════════════════════════════════════════════════════════════

# Reference statistics derived from a broad sample of real food photos.
# An image whose stats fall far outside these global ranges is very
# unlikely to be a genuine food photograph.
#
# Heuristics used:
#   saturation_mean  : food is colourful; near-zero saturation = greyscale
#                      watermark background or B&W image
#   r_g_ratio        : food images have warm colours (r ≈ g); extreme ratios
#                      suggest unnatural cartoon palettes or heavy overlays
#   edge_density     : food has moderate texture; very high edge density
#                      signals text-heavy screenshots; very low signals
#                      plain background panels
#   brightness_range : range of per-channel means; near-zero = greyscale/mono
#   grey_fraction    : fraction of pixels where R≈G≈B (within ±10); high
#                      fraction signals greyscale background or watermark

FOOD_STATS_BOUNDS = {
    # (min_acceptable, max_acceptable)
    "saturation_mean":  (10.0,  220.0),   # HSV saturation 0-255 scale
    "edge_density":     (0.02,  0.45),    # fraction of pixels that are edges
    "grey_fraction":    (0.0,   0.65),    # fraction of near-grey pixels
    "brightness_range": (5.0,   200.0),   # max(mean_r,g,b) - min(mean_r,g,b)
}


def pixel_stats(path: str) -> dict | None:
    """Compute non-food detection statistics for a single image."""
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)          # (H, W, 3)

        mean_r, mean_g, mean_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()

        # HSV saturation (PIL)
        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        saturation_mean = hsv[:, :, 1].mean()

        # Grey fraction: pixels where all channels within 10 of each other
        max_ch = arr.max(axis=2)
        min_ch = arr.min(axis=2)
        grey_fraction = float((max_ch - min_ch < 10).mean())

        # Edge density via simple gradient magnitude
        grey = np.array(img.convert("L"), dtype=np.float32)
        gx   = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy   = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        grad = np.sqrt(gx**2 + gy**2)
        edge_density = float((grad > 20).mean())

        brightness_range = float(max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b))

        return {
            "mean_r":           float(mean_r),
            "mean_g":           float(mean_g),
            "mean_b":           float(mean_b),
            "saturation_mean":  float(saturation_mean),
            "grey_fraction":    grey_fraction,
            "edge_density":     edge_density,
            "brightness_range": brightness_range,
            "aspect":           img.width / img.height,
        }
    except Exception:
        return None


def detect_nonfood_pixel_outliers(
    df:      pd.DataFrame,
    img_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag images whose pixel statistics fall outside the expected range
    for any real food photograph.

    This function uses GLOBAL absolute
    thresholds derived from the visual properties shared by all food
    images. An image that fails these thresholds is probably not food
    at all — not just an unusual instance of its class.

    Statistics and what they catch
    --------------------------------
    saturation_mean  : near zero → greyscale backgrounds / watermark overlays
    edge_density     : very high → text-heavy screenshots; very low → plain
                       colour panels or blank background squares
    grey_fraction    : high → greyscale or mono image (B&W photo / overlay)
    brightness_range : near zero → perfectly greyscale or monochrome image

    Parameters
    ----------
    df       : DataFrame with columns ['image_id', 'label']
    img_dir  : path to the flat image folder

    Returns
    -------
    stats_df   : per-image statistics table
    flagged_df : rows that fail at least one global bound
                 → also saved to review_stage2_nonfood_pixelstats.csv
    """
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Non-food pixel stats"):
        path  = os.path.join(img_dir, str(row["image_id"]))
        stats = pixel_stats(path)
        if stats is None:
            continue
        stats["image_id"] = row["image_id"]
        stats["label"]    = row["label"]
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df, stats_df.copy()

    # Flag images that violate at least one global food-image bound
    fail_mask = pd.Series(False, index=stats_df.index)
    fail_reasons = [""] * len(stats_df)

    for stat, (lo, hi) in FOOD_STATS_BOUNDS.items():
        if stat not in stats_df.columns:
            continue
        out_of_range = (stats_df[stat] < lo) | (stats_df[stat] > hi)
        for idx in stats_df[out_of_range].index:
            val = stats_df.at[idx, stat]
            fail_reasons[stats_df.index.get_loc(idx)] += (
                f"{stat}={val:.2f}[{lo},{hi}]; "
            )
        fail_mask |= out_of_range

    flagged_df = stats_df[fail_mask].copy()
    flagged_df.insert(
        flagged_df.columns.get_loc("image_id") + 1,
        "fail_reasons",
        [r.rstrip("; ") for r in np.array(fail_reasons)[fail_mask.values]],
    )

    # Console report
    print(f"\n[Stage 2] Flagged {len(flagged_df):,} / {len(stats_df):,} images "
          f"as potential non-food / background-only")
    if not flagged_df.empty:
        flagged_df.to_csv("../results/review_stage2_nonfood_pixelstats.csv", index=False)
        print("  → saved: review_stage2_nonfood_pixelstats.csv")
        print("  ⚠  Inspect before removing — unusual-but-valid food shots may appear here")

    return stats_df, flagged_df


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 – Embedding-space non-food detector
# Catches: images that do not look like food at all in ResNet-50 feature
#          space — wrong-domain photos (people, text, objects) that pass
#          pixel statistics because they have normal colour/edge properties
# Decision: REVIEW CSV — inspect before removing
#
# KEY DIFFERENCE from original Stage 3
# ------------------------------------
# The original ran a PER-CLASS IsolationForest to find unusual instances
# within each food class. That catches rare food plating, not non-food.
#
# This version fits ONE GLOBAL IsolationForest on the features of ALL
# images together. An image far from the global food manifold is genuinely
# non-food; an unusual close-up of ramen is still near the food manifold.
# ══════════════════════════════════════════════════════════════════════

class FlatFoodDataset(Dataset):
    """
    Minimal Dataset for feature extraction (Stage 3).
    Reads images from a flat folder via (image_id, label) rows.
    Returns (tensor, int_label, image_id_string).
    """
    TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, img_dir: str) -> None:
        self.df      = df.reset_index(drop=True)
        self.img_dir = img_dir
        classes      = sorted(df["label"].unique())
        self.c2i     = {c: i for i, c in enumerate(classes)}
        self.targets = [self.c2i[lbl] for lbl in df["label"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row  = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.TFM(img), self.targets[i], str(row["image_id"])


def detect_nonfood_embedding_outliers(
    df:                pd.DataFrame,
    img_dir:           str,
    contamination:     float = 0.05,
    batch_size:        int   = 64,
    num_workers:       int   = 0,
    device:            str   = "mps",
) -> tuple[np.ndarray, list, pd.DataFrame]:
    """
    Extract ResNet-50 features for every image, then run a SINGLE GLOBAL
    IsolationForest to flag images that sit far from the food-image
    manifold in feature space.

    Why global (not per-class)
    --------------------------
    A non-food image (a person, a piece of text, a random object) will
    be an outlier in the GLOBAL food distribution. Running per-class
    forests would instead flag rare-but-valid food shots within each
    class cluster — exactly what we do NOT want here.

    contamination=0.05 means the model flags the ~5 % of images it
    considers most anomalous. Adjust downward (e.g. 0.02) if you see
    too many valid food images in the output CSV.

    Parameters
    ----------
    df             : DataFrame with columns ['image_id', 'label']
    img_dir        : path to the flat image folder
    contamination  : IsolationForest contamination fraction
    batch_size     : inference batch size
    num_workers    : DataLoader workers (0 = safe on Mac)
    device         : 'mps', 'cuda', or 'cpu'

    Returns
    -------
    feats          : (N, 2048) feature matrix
    image_ids      : list of image_id strings in the same row-order as feats
    outlier_df     : flagged rows
                     → also saved to review_stage3_nonfood_embedding.csv
    """
    # Build backbone (classifier head removed)
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    ds = FlatFoodDataset(df, img_dir)
    dl = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = device in ("cuda",),
    )

    all_feats, all_ids = [], []

    with torch.no_grad():
        for imgs, _, ids in tqdm(dl, desc="[Stage 3] Feature extraction"):
            feats_batch = backbone(imgs.to(device)).cpu().numpy()
            all_feats.append(feats_batch)
            all_ids.extend(ids)

    feats = np.vstack(all_feats)   # (N, 2048)

    # Single global IsolationForest — flags non-food outliers
    print("[Stage 3] Running global IsolationForest for non-food detection...")
    clf = IsolationForest(
        contamination = contamination,
        random_state  = 42,
        n_jobs        = -1,
    )
    preds = clf.fit_predict(feats)   # -1 = outlier, 1 = inlier

    outlier_indices = np.where(preds == -1)[0].tolist()
    outlier_ids     = [all_ids[i] for i in outlier_indices]
    # Include anomaly score (more negative = more anomalous)
    scores          = clf.decision_function(feats)

    outlier_df = df[df["image_id"].isin(outlier_ids)].copy()
    score_map  = {all_ids[i]: float(scores[i]) for i in outlier_indices}
    outlier_df["anomaly_score"] = outlier_df["image_id"].map(score_map)
    outlier_df = outlier_df.sort_values("anomaly_score")   # most anomalous first

    # Save features for optional downstream reuse
    np.save("../results/embeddings_feats.npy", feats)
    with open("../results/embeddings_image_ids.txt", "w") as f:
        f.write("\n".join(all_ids))

    # Console report
    print(f"\n[Stage 3] Flagged {len(outlier_df):,} global embedding outliers "
          f"out of {len(df):,}  (contamination={contamination})")
    if not outlier_df.empty:
        outlier_df.to_csv("../results/review_stage3_nonfood_embedding.csv", index=False)
        print("  → saved: review_stage3_nonfood_embedding.csv")
        print("  ⚠  Inspect before removing — unusual-but-valid food shots may appear here")

    return feats, all_ids, outlier_df


# ══════════════════════════════════════════════════════════════════════
# Utility – plot any flagged dataframe as an image grid
# ══════════════════════════════════════════════════════════════════════

def plot_flagged_images(
    flagged_df:  pd.DataFrame,
    img_dir:     str,
    title:       str   = "Flagged images",
    max_images:  int   = 20,
    cols:        int   = 5,
    figsize_per: tuple = (3, 3),
) -> None:
    """
    Display a grid of flagged images for visual inspection.

    Works with the output of any stage. An optional 'reason',
    'fail_reasons', or 'anomaly_score' column is shown below each
    image if present.

    Parameters
    ----------
    flagged_df  : DataFrame with at least ['image_id', 'label']
    img_dir     : path to the flat image folder
    title       : overall figure title
    max_images  : how many images to show (capped for readability)
    cols        : grid columns
    figsize_per : (width, height) per subplot cell in inches
    """
    if flagged_df.empty:
        print("Nothing to plot — dataframe is empty.")
        return

    subset = flagged_df.head(max_images)
    n      = len(subset)
    rows   = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per[0] * cols, figsize_per[1] * rows),
    )
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(subset.iterrows()):
        ax   = axes[i]
        path = os.path.join(img_dir, str(row["image_id"]))

        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception as ex:
            ax.text(
                0.5, 0.5, f"Cannot open\n{ex}",
                ha="center", va="center", fontsize=7,
                transform=ax.transAxes, wrap=True,
            )

        # Show the most informative extra column if present
        extra = ""
        for col in ("reason", "fail_reasons", "anomaly_score"):
            if col in row.index:
                val   = str(row[col])
                extra = f"\n{col}: {val[:35]}"
                break

        ax.set_title(f"label={row['label']}{extra}", fontsize=7)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title}  ({n} shown)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# Utility – visualise Stage 2 pixel-stat outliers in class context
# ══════════════════════════════════════════════════════════════════════

def plot_stage2_outlier_context(
    stats_df:   pd.DataFrame,
    flagged_df: pd.DataFrame,
    label:      int,
    cols:       list = ("saturation_mean", "edge_density",
                        "grey_fraction", "brightness_range"),
) -> None:
    """
    For ONE class label, show a strip plot of the non-food detection
    pixel statistics for all images in that class, with Stage-2-flagged
    images overlaid as red points.

    Reading the plot
    ----------------
    • Grey points  = images of this class that passed Stage 2.
    • Red X points = images flagged as potential non-food by Stage 2.
    • A red point far outside the grey cloud on saturation or
      brightness_range is very likely a greyscale overlay or watermark.
    • A red point with very high edge_density is likely a screenshot.
    • A red point with very high grey_fraction is likely a B&W image.

    Parameters
    ----------
    stats_df   : first return value of detect_nonfood_pixel_outliers()
    flagged_df : second return value of detect_nonfood_pixel_outliers()
    label      : which class to inspect
    cols       : which statistic columns to plot
    """
    class_stats   = stats_df[stats_df["label"] == label]
    class_flagged = flagged_df[flagged_df["label"] == label]

    if class_stats.empty:
        print(f"No Stage 2 stats found for label={label}.")
        return

    plot_cols = [c for c in cols if c in class_stats.columns]
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(4 * len(plot_cols), 4))
    if len(plot_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, plot_cols):
        sns.stripplot(
            x=[label] * len(class_stats), y=class_stats[col],
            ax=ax, color="lightgrey", alpha=0.6, jitter=True, size=4,
        )
        if not class_flagged.empty:
            sns.stripplot(
                x=[label] * len(class_flagged), y=class_flagged[col],
                ax=ax, color="red", jitter=True, size=6, marker="X",
            )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_xticks([])

    fig.suptitle(
        f"Stage 2 — class {label}: grey = passed, red X = non-food flagged "
        f"({len(class_flagged)} flagged / {len(class_stats)} total)",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# Utility – apply human review decisions → final clean CSV
# ══════════════════════════════════════════════════════════════════════

def apply_review_decisions(
    df:               pd.DataFrame,
    remove_csv_paths: list,
    output_csv:       str = "train_labels_clean.csv",
) -> pd.DataFrame:
    """
    After manually reviewing the stage 2/3 CSVs, create trimmed
    versions containing only the rows you want removed, then pass
    their paths here to produce the final clean DataFrame.

    Workflow
    --------
    1. Open review_stage2_nonfood_pixelstats.csv in a spreadsheet viewer.
    2. Delete rows you want to KEEP (valid food shots).
    3. Save the remaining rows as  review_stage2_confirmed_remove.csv.
    4. Repeat for stage 3 (review_stage3_nonfood_embedding.csv).
    5. Call this function with the two confirmed-remove paths.

    Parameters
    ----------
    df               : DataFrame after Stage 1 auto-removal
    remove_csv_paths : list of CSV file paths — each must have 'image_id'
    output_csv       : path for the final saved CSV

    Returns
    -------
    final_df : df with confirmed non-food outliers removed
    """
    ids_to_remove: set = set()

    for path in remove_csv_paths:
        if not os.path.exists(path):
            print(f"  Warning: '{path}' not found — skipped.")
            continue
        tmp = pd.read_csv(path)
        if "image_id" not in tmp.columns:
            print(f"  Warning: '{path}' has no 'image_id' column — skipped.")
            continue
        ids_to_remove.update(tmp["image_id"].astype(str).tolist())

    before   = len(df)
    final_df = df[~df["image_id"].astype(str).isin(ids_to_remove)].reset_index(drop=True)

    final_df.to_csv(output_csv, index=False)
    print(f"\n[apply_review_decisions] Removed {before - len(final_df):,} confirmed non-food outliers.")
    print(f"  Final dataset : {len(final_df):,} images, "
          f"{final_df['label'].nunique()} classes")
    print(f"  → saved: {output_csv}")
    return final_df


# ══════════════════════════════════════════════════════════════════════
# Master pipeline
# ══════════════════════════════════════════════════════════════════════

def run_outlier_pipeline(
    csv_path:    str,
    img_dir:     str,
    num_classes: int   = 251,
    device:      str   = "mps",
    contamination: float = 0.05,
) -> tuple:
    """
    Run Stages 1, 2, and 3 in order to detect non-food images and
    blank/black/white backgrounds.

    Stage 1  → auto-removes blank/black/white/corrupt/missing images.
    Stages 2, 3 → write review CSVs only; nothing is removed until
    you call apply_review_decisions() after human inspection.

    Parameters
    ----------
    csv_path      : path to train_labels.csv  (columns: image_id, label)
    img_dir       : path to the flat train_set/ folder
    num_classes   : number of classes  (251 for food-251) — kept for
                    compatibility; not used in the new global Stage 3
    device        : 'mps' (Apple Silicon Mac), 'cuda', or 'cpu'
    contamination : IsolationForest contamination for Stage 3 (default 0.05)

    Returns
    -------
    df              : DataFrame after Stage 1 auto-removal
    stats_df        : per-image pixel stats (Stage 2) — for plot_stage2_outlier_context()
    flagged_df      : Stage 2 flagged rows  — for plot_stage2_outlier_context()
    feats           : (N, 2048) feature matrix — for optional downstream use
    image_ids       : list of image_id strings, same order as feats
    embed_outlier_df: Stage 3 flagged rows
    """
    # Load and normalise column names
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        if col.lower() in ("image_id", "img_id", "img_name",
                           "filename", "file_name", "image"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "class_id",
                              "category", "target"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Cannot find image_id / label columns.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    print(f"\nLoaded: {len(df):,} images — {df['label'].nunique()} classes")
    print("─" * 60)

    # Stage 1: auto-remove corrupt / blank / black / white images
    df, _ = image_integrity_audit(df, img_dir)

    # Stage 2: non-food pixel stats → review CSV
    stats_df, flagged_df = detect_nonfood_pixel_outliers(df, img_dir)

    # Stage 3: global embedding outliers → review CSV
    feats, image_ids, embed_outlier_df = detect_nonfood_embedding_outliers(
        df,
        img_dir,
        contamination = contamination,
        device        = device,
    )

    # Summary
    orig_len = len(pd.read_csv(csv_path))
    print(f"\n{'─' * 60}")
    print("PIPELINE COMPLETE  (non-food + blank/background detection)")
    print(f"{'─' * 60}")
    print(f"  Original images     : {orig_len:,}")
    print(f"  After Stage 1       : {len(df):,}  (auto-removed blank/black/white/corrupt)")
    print(f"  Stage 2 review CSV  : review_stage2_nonfood_pixelstats.csv")
    print(f"  Stage 3 review CSV  : review_stage3_nonfood_embedding.csv")
    print()
    print("Visualise Stage 2 flagged images in context:")
    print("  plot_stage2_outlier_context(stats_df, flagged_df, label=<class_id>)")
    print("  plot_flagged_images(flagged_df, img_dir='train_set/')")
    print("  plot_flagged_images(embed_outlier_df, img_dir='train_set/', title='Stage 3')")
    print()
    print("Next step — after reviewing the CSVs:")
    print("  apply_review_decisions(")
    print("      df,")
    print("      remove_csv_paths=[")
    print("          'review_stage2_confirmed_remove.csv',")
    print("          'review_stage3_confirmed_remove.csv',")
    print("      ],")
    print("      output_csv='train_labels_clean.csv',")
    print("  )")
    print(f"{'─' * 60}\n")

    return df, stats_df, flagged_df, feats, image_ids, embed_outlier_df