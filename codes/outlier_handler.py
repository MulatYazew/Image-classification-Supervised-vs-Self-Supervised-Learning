"""
FoodNet - outlier_handler
========================================
3-stage outlier detection for food image datasets (251 classes): Stage 1
(integrity), Stage 2 (pixel stats), Stage 3 (per-class embedding outliers),
plus review visualisations.

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

To rerun without rescanning: check outlier_stage_outputs_exist(out_dir) and,
if True, call load_cached_outlier_outputs() instead of run_outlier_pipeline.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

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

# Dark-theme plot colours
BG, PANEL, ACCENT, FLAG, LINE, SPINE = "#1a1a2e", "#12122a", "#7b7baa", "#ff4444", "#ffcc00", "#444466"

# Pixel-statistic thresholds tuned for food imagery
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


# STAGE 1 - Integrity & Blank Audit
# Unconditional auto-removal of unusable images (no human review needed):
# missing/truncated/corrupt files, undersized, near-black, near-white-blank,
# or near-uniform low-contrast images.

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
    """Scan every image in df for hard integrity failures and auto-remove them.

    Checks (in order): missing, truncated (< min_bytes), corrupt (PIL decode
    error), too_small (< min_size_px), near_black (mean < black_thresh),
    near_white_blank (mean > white_thresh and low saturation), low_contrast
    (std < min_std). Returns (clean_df, issues_df) and writes the removal log.
    """
    issues = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Integrity audit"):
        path, reason = os.path.join(img_dir, str(row["image_id"])), None

        if not os.path.exists(path):
            reason = "missing"
        elif os.path.getsize(path) < min_bytes:
            reason = f"truncated(bytes={os.path.getsize(path)})"
        else:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if w < min_size_px or h < min_size_px:
                    reason = f"too_small({w}x{h})"
                else:
                    arr = np.array(img, dtype=np.float32)
                    mean, std = arr.mean(), arr.std()
                    if mean < black_thresh:
                        reason = f"near_black(mean={mean:.1f})"
                    elif mean > white_thresh:
                        sat = np.array(img.convert("HSV"), dtype=np.float32)[:, :, 1].mean()
                        if sat < white_min_sat:
                            reason = f"near_white_blank(mean={mean:.1f},sat={sat:.1f})"
                    elif std < min_std:
                        reason = f"low_contrast(mean={mean:.0f},std={std:.1f})"
            except Exception as ex:
                reason = f"corrupt:{ex}"

        if reason:
            issues.append({"image_id": row["image_id"], "label": row["label"], "reason": reason})

    issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(columns=["image_id", "label", "reason"])
    clean_df = df[~df["image_id"].isin(issues_df["image_id"])].reset_index(drop=True)

    print(f"\n[Stage 1] Auto-removed {len(issues_df):,} / {len(df):,} images")
    if not issues_df.empty:
        print(issues_df["reason"].str.split("(").str[0].value_counts().to_string())
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "removed_stage1_integrity.csv")
        issues_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    return clean_df, issues_df


# STAGE 2 - Non-Food Pixel-Statistics Detector
# GPU-free pixel-statistic checks flag likely non-food images (text cards,
# blank shots, solid colours) for human review; nothing is auto-deleted here.

def pixel_stats(path: str) -> dict | None:
    """Compute cheap pixel-level statistics used by Stage 2's flagging rules.

    Returns per-channel means, overall mean/std, HSV saturation mean, grey
    fraction (near-achromatic pixel share), edge density (gradient-based),
    brightness range, unique-colour count (64x64 downsample), and aspect
    ratio. Returns None if the file can't be decoded.
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        mean_r, mean_g, mean_b = (float(arr[:, :, c].mean()) for c in range(3))

        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        saturation_mean = float(hsv[:, :, 1].mean())

        # grey = max-min channel spread < 15
        max_ch, min_ch = arr.max(axis=2), arr.min(axis=2)
        grey_fraction = float((max_ch - min_ch < 15).mean())

        # edge = finite-difference gradient magnitude > 20 on luminance
        grey = np.array(img.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(grey, axis=1, prepend=grey[:, :1]))
        gy = np.abs(np.diff(grey, axis=0, prepend=grey[:1, :]))
        edge_density = float((np.sqrt(gx**2 + gy**2) > 20).mean())

        brightness_range = float(max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b))
        n_unique_colors = len(set(img.resize((64, 64), Image.BILINEAR).getdata()))

        return dict(mean_r=mean_r, mean_g=mean_g, mean_b=mean_b,
                     overall_mean=float(arr.mean()), overall_std=float(arr.std()),
                     saturation_mean=saturation_mean, grey_fraction=grey_fraction,
                     edge_density=edge_density, brightness_range=brightness_range,
                     n_unique_colors=n_unique_colors, aspect=img.width / img.height)

    except Exception:
        return None


def detect_nonfood_pixel_outliers(
    df: pd.DataFrame,
    img_dir: str,
    out_dir: str = ".",
    n_fails_required: int = N_FAILS_DEFAULT,
    class_zscore_thr: float = CLASS_ZSCORE_THR,
    min_class_size: int = MIN_CLASS_SIZE,
) -> tuple:
    """Flag likely non-food/mislabelled images from pixel statistics (OR logic):
    (a) standard bounds (FOOD_STATS_BOUNDS) accumulate failures; (b) food-safe
    images (colourful + textured) need more simultaneous failures
    (FOOD_SAFE_FAILURES_REQUIRED); (c) an EXTREME_SINGLE violation flags alone;
    (d) per-class z-score outliers (class_zscore_thr, min class size
    min_class_size); (e) < 50 unique colours at 64x64. Writes flagged images
    to CSV for review; returns (stats_df, flagged_df).
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Pixel stats"):
        stats = pixel_stats(os.path.join(img_dir, str(row["image_id"])))
        if stats is None:
            continue
        stats["image_id"], stats["label"] = row["image_id"], row["label"]
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df, stats_df.copy()

    fail_mask = pd.Series(False, index=stats_df.index)
    fail_reasons = {i: [] for i in stats_df.index}

    # (a) standard bounds
    for stat, (lo, hi) in FOOD_STATS_BOUNDS.items():
        col = stats_df[stat]
        out = col < lo
        if hi is not None:
            out = out | (col > hi)
        for idx in stats_df.index[out]:
            fail_reasons[idx].append(f"{stat}={stats_df.at[idx, stat]:.3f}[{lo},{hi}]")

    n_fails = pd.Series({i: len(v) for i, v in fail_reasons.items()}, index=stats_df.index)

    # (b) food-safe guard: colourful + textured images need more failures
    is_food_safe = (stats_df["saturation_mean"] >= FOOD_SAFE_SATURATION_MIN) & \
                   (stats_df["edge_density"] <= FOOD_SAFE_EDGE_MAX)
    required = np.where(is_food_safe, FOOD_SAFE_FAILURES_REQUIRED, n_fails_required)
    fail_mask |= (n_fails >= required)

    # (c) extreme single violation
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

    # (d) per-class z-score, vectorised (groupby/transform) — ~118K images x 4
    # features would be too slow as a per-row Python loop
    row_class_size = stats_df["label"].map(stats_df.groupby("label").size())
    for stat in FOOD_STATS_BOUNDS:
        class_mean = stats_df.groupby("label")[stat].transform("mean")
        class_std = stats_df.groupby("label")[stat].transform("std")
        valid = (row_class_size >= min_class_size) & class_std.notna() & (class_std >= 1e-6)
        z = (stats_df[stat] - class_mean).abs() / class_std
        flagged = valid & (z > class_zscore_thr)
        for idx in stats_df.index[flagged]:
            fail_reasons[idx].append(f"class_z:{stat}={z.at[idx]:.1f}sigma")
        fail_mask |= flagged

    # (e) low unique-colour count -> likely synthetic/solid-fill
    synthetic = stats_df["n_unique_colors"] < 50
    for idx in stats_df.index[synthetic]:
        fail_reasons[idx].append(f"low_unique_colors={stats_df.at[idx, 'n_unique_colors']}")
    fail_mask |= synthetic

    flagged_df = stats_df[fail_mask].copy()
    flagged_df["fail_reasons"] = flagged_df.index.map(lambda i: "; ".join(fail_reasons[i]))
    flagged_df["n_fails"] = n_fails[fail_mask].values
    flagged_df = flagged_df.sort_values("n_fails", ascending=False).reset_index(drop=True)

    print(f"\n[Stage 2] Flagged {len(flagged_df):,} / {len(stats_df):,} images")
    if not flagged_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "review_stage2_nonfood_pixelstats.csv")
        flagged_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    # full stats table persisted so cached reruns skip rescanning every image
    os.makedirs(out_dir, exist_ok=True)
    full_stats_path = os.path.join(out_dir, "stage2_pixel_stats_full.csv")
    stats_df.to_csv(full_stats_path, index=False)
    print(f"  -> saved full pixel-stats table (all images): {full_stats_path}")

    return stats_df, flagged_df


# STAGE 3 - Per-Class Embedding Outliers (ResNet-50 + IsolationForest)
# Deep semantic outlier detection in ResNet-50 feature space, catching
# mismatches pixel statistics can't see. Global + per-class IsolationForest
# scores are combined (OR): extreme global outlier, or moderate outlier
# confirmed by both models.

class FlatFoodDataset(Dataset):
    """Wraps the manifest with standard ImageNet preprocessing for ResNet-50.

    Returns (tensor, label_int, image_id_str). Unreadable images fall back to
    a black 224x224 placeholder (never breaks the batch), which then reads as
    an extreme outlier and gets flagged automatically.
    """

    TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    def __init__(self, df, img_dir):
        self.df, self.img_dir = df.reset_index(drop=True), img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, str(row["image_id"]))
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.TFM(img), int(row["label"]), str(row["image_id"])


def detect_nonfood_embedding_outliers(
    df: pd.DataFrame,
    img_dir: str,
    out_dir: str = ".",
    device: str = "mps",
    batch_size: int = 64,
    num_workers: int = 0,
    global_thr: float = -0.12,
    per_class_thr: float = -0.15,
    min_class_size: int = 10,
) -> tuple:
    """Extract ResNet-50 embeddings and flag semantic outliers via IsolationForest.

    Global score (fit on all classes) catches dataset-wide anomalies;
    per-class score (fit per class, min_class_size floor) catches images
    unusual within their own class. Flagged if global_score < -0.20, or
    global_score < global_thr AND per_class_score < per_class_thr. Returns
    (feats_pca, all_ids, global_scores, per_class_scores, outlier_df).
    """
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    backbone.fc = nn.Identity()   # drop classification head -> raw pooled features
    backbone.eval().to(device)

    # pin_memory only helps host->device copies on CUDA
    dl = DataLoader(FlatFoodDataset(df, img_dir), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=(str(device) == "cuda"))

    feats_list, all_ids, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels, ids in tqdm(dl, desc="[Stage 3] Feature extraction"):
            feats_list.append(backbone(imgs.to(device)).cpu().numpy())
            all_ids.extend(ids)
            all_labels.extend(labels.tolist())
    feats = np.vstack(feats_list)
    labels_arr = np.array(all_labels)

    print("[Stage 3] Fitting global IsolationForest...")
    feats_scaled = StandardScaler().fit_transform(feats)
    feats_pca = PCA(n_components=min(128, feats_scaled.shape[1]), random_state=42).fit_transform(feats_scaled)

    clf_global = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    clf_global.fit(feats_pca)
    global_scores = clf_global.decision_function(feats_pca)   # negative = anomalous

    print("[Stage 3] Fitting per-class IsolationForest models...")
    per_class_scores = np.full(len(all_ids), np.inf)   # +inf for skipped/small classes
    for lbl in tqdm(np.unique(labels_arr), desc="  per-class IF"):
        mask = labels_arr == lbl
        if mask.sum() < min_class_size:
            continue
        n_est = max(50, min(200, mask.sum()))   # scale tree count to class size
        clf_c = IsolationForest(n_estimators=n_est, contamination="auto", random_state=42, n_jobs=-1)
        clf_c.fit(feats_pca[mask])
        per_class_scores[mask] = clf_c.decision_function(feats_pca[mask])

    extreme_global = global_scores < -0.20
    moderate = (global_scores < global_thr) & (per_class_scores < per_class_thr)
    outlier_mask = extreme_global | moderate

    outlier_ids = [all_ids[i] for i in np.where(outlier_mask)[0]]
    outlier_df = df[df["image_id"].astype(str).isin(outlier_ids)].copy()
    g_map = {all_ids[i]: float(global_scores[i]) for i in np.where(outlier_mask)[0]}
    c_map = {all_ids[i]: float(per_class_scores[i]) for i in np.where(outlier_mask)[0]}
    outlier_df["anomaly_score_global"] = outlier_df["image_id"].astype(str).map(g_map)
    outlier_df["anomaly_score_per_class"] = outlier_df["image_id"].astype(str).map(c_map)
    outlier_df = outlier_df.sort_values("anomaly_score_global").reset_index(drop=True)

    print(f"\n[Stage 3] Flagged {len(outlier_df):,} images "
          f"(global thr={global_thr}, per-class thr={per_class_thr})")
    if not outlier_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "review_stage3_nonfood_embedding.csv")
        outlier_df.to_csv(path, index=False)
        print(f"  -> saved: {path}")

    # embeddings/scores for ALL images persisted so cached reruns skip the
    # expensive ResNet-50 forward pass
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "stage3_embeddings.npz")
    np.savez_compressed(npz_path, feats=feats_pca, ids=np.array(all_ids),
                        global_scores=global_scores, per_class_scores=per_class_scores)
    print(f"  -> saved embeddings/scores: {npz_path}")

    return feats_pca, all_ids, global_scores, per_class_scores, outlier_df


# VISUALISATION HELPERS
# load_image_safe (fault-tolerant loader), visualize_flagged_images (review
# grid), plot_before_after_class_counts, visualize_outlier_decisions
# (per-class scatter), plot_anomaly_score_distribution. Dark theme via
# BG/PANEL/ACCENT/FLAG/LINE/SPINE.

def load_image_safe(path: str, size: int = 300) -> np.ndarray | None:
    """Load an image as an RGB uint8 array, thumbnailed to size; None on error."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def visualize_flagged_images(flagged_df, img_dir, title="Flagged images",
                              max_images=48, cols=8, save_path=None) -> None:
    """Grid of up to max_images flagged images, border colour = confidence:
    red = high (3+ failures, EXTREME violation, or Stage 1 removal), yellow =
    moderate (2 failures), blue = low (0-1). Expects flagged_df from Stage
    1/2/3 with image_id + label columns.
    """
    if flagged_df.empty:
        print("Nothing to plot - dataframe is empty.")
        return

    subset = flagged_df.head(max_images)
    n = len(subset)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), facecolor=BG)
    fig.suptitle(f"{title} ({n} of {len(flagged_df)} shown)", fontsize=12, color="white", fontweight="bold", y=1.01)
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(subset.iterrows()):
        ax = axes[i]
        ax.set_facecolor(PANEL)

        arr = load_image_safe(os.path.join(img_dir, str(row["image_id"])))
        if arr is not None:
            ax.imshow(arr)
        else:
            ax.text(0.5, 0.5, "load\nerror", ha="center", va="center", fontsize=6, color="salmon", transform=ax.transAxes)

        # Stage 1 rows (a "reason" col, no "n_fails") are unconditional
        # removals, not confidence-scored — always render as most severe
        if "n_fails" not in row.index and "reason" in row.index:
            border = FLAG
        else:
            n_fails = row.get("n_fails", 1)
            reasons = str(row.get("fail_reasons", ""))
            border = FLAG if (n_fails >= 3 or "EXTREME" in reasons) else (LINE if n_fails == 2 else "#44aaff")
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(2)

        reason = ""
        for col in ("fail_reasons", "reason", "anomaly_score_global"):
            if col in row.index and pd.notna(row[col]):
                reason = str(row[col])[:50]
                break
        ax.set_title(f"cls {row['label']}\n{reason}", fontsize=5, color="white", pad=2)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off"); axes[j].set_facecolor(BG)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        print(f"  -> saved grid: {save_path}")
    plt.show()
    plt.close()


def plot_before_after_class_counts(raw_df, final_df, num_classes: int = 251,
                                   title: str = "Images per class — before vs. after outlier removal",
                                   save_path=None) -> None:
    """Grouped bar chart of per-class image counts before vs. after cleaning,
    sorted by raw count — shows how much each class shrank."""
    idx = range(num_classes)
    raw_counts = raw_df["label"].value_counts().reindex(idx, fill_value=0)
    final_counts = final_df["label"].value_counts().reindex(idx, fill_value=0)
    order = raw_counts.sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    x = np.arange(num_classes)
    ax.bar(x, raw_counts.loc[order].values, width=1.0, color=ACCENT, alpha=0.6, label="before (raw)")
    ax.bar(x, final_counts.loc[order].values, width=1.0, color=LINE, alpha=0.9, label="after (cleaned)")

    ax.set_xlabel("class rank (sorted by raw count)", color="white")
    ax.set_ylabel("image count", color="white")
    ax.set_title(title, color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color(SPINE)
    ax.legend(labelcolor="white", facecolor=BG, edgecolor=SPINE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        print(f"  -> saved: {save_path}")
    plt.show()
    plt.close()


def visualize_outlier_decisions(stats_df, flagged_df, label, save_path=None) -> None:
    """Per-class scatter of each pixel-stat feature (jittered strip chart)
    with threshold lines and flagged images as red X markers — an X inside
    the normal cluster suggests a false positive from a different feature."""
    class_stats = stats_df[stats_df["label"] == label]
    class_flagged = flagged_df[flagged_df["label"] == label] if not flagged_df.empty else pd.DataFrame()
    if class_stats.empty:
        print(f"No stats for label={label}.")
        return

    stat_cols = [c for c in FOOD_STATS_BOUNDS if c in class_stats.columns]
    fig, axes = plt.subplots(1, len(stat_cols), figsize=(len(stat_cols) * 4, 5), facecolor=BG)
    if len(stat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, stat_cols, strict=True):
        ax.set_facecolor(PANEL)
        lo, hi = FOOD_STATS_BOUNDS.get(col, (None, None))

        ax.scatter(np.random.uniform(-0.2, 0.2, len(class_stats)), class_stats[col],
                   color=ACCENT, alpha=0.4, s=10, label=f"all ({len(class_stats)})")

        if not class_flagged.empty:
            ax.scatter(np.random.uniform(-0.2, 0.2, len(class_flagged)), class_flagged[col],
                       color=FLAG, s=50, marker="X", zorder=5, label=f"flagged ({len(class_flagged)})")

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
    plt.show()
    plt.close()


def plot_anomaly_score_distribution(scores, score_thr=-0.12, title="Anomaly score distribution", save_path=None) -> None:
    """Histogram of IsolationForest scores with the flagging threshold marked —
    helps calibrate global_thr/per_class_thr from the score distribution's shoulder."""
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.hist(scores, bins=200, color="#5555bb", alpha=0.8, edgecolor="none")

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
    plt.show()
    plt.close()


# Apply Review Decisions
# After manually reviewing the Stage 2/3 CSVs (delete rows to KEEP, save the
# rest as confirmed_remove_stage{2,3}.csv), apply the confirmed removals here.

def apply_review_decisions(df: pd.DataFrame, remove_csv_paths: list[str], output_csv: str = "train_labels_clean.csv") -> pd.DataFrame:
    """Remove reviewer-confirmed outliers (image_id column in each CSV in
    remove_csv_paths) from df and write the cleaned manifest to output_csv.
    Missing files or files without an image_id column are skipped with a
    warning, so partial review sessions still work.
    """
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

    before = len(df)
    final_df = df[~df["image_id"].astype(str).isin(ids_to_remove)].reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)

    print(f"\n[apply_review_decisions] Removed {before - len(final_df):,} confirmed outliers.")
    print(f"  Final dataset: {len(final_df):,} images, {final_df['label'].nunique()} classes")
    print(f"  -> saved: {output_csv}")
    return final_df


# Per-Class Audit Summary Report
# Ties the 3 stages into one per-class trail (raw -> stage1 removed ->
# flagged -> final) so no tail class gets silently gutted below a usable size.

def audit_summary_report(
    raw_df: pd.DataFrame,
    stage1_df: pd.DataFrame,
    flagged2_df: pd.DataFrame,
    flagged3_df: pd.DataFrame,
    final_df: pd.DataFrame | None = None,
    num_classes: int = 251,
    out_dir: str = ".",
    min_remaining: int = 15,
) -> pd.DataFrame:
    """Build the per-class outlier-audit trail across all 3 stages.

    If final_df is omitted, final count is a worst-case estimate assuming
    every flagged image (Stage 2 or 3, whichever flags more) gets removed —
    a conservative check to run before committing to manual review decisions.
    Classes at or below min_remaining images are printed as a warning.
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
# Runs all 3 stages in sequence with column normalisation and summary
# reporting; the single entry point for typical usage.

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
    """Run the full 3-stage outlier pipeline: integrity audit -> pixel-stat
    detection -> embedding-based detection (skip with skip_stage3=True, e.g.
    no GPU). Normalises common column-name variants to image_id/label,
    raising ValueError if neither can be found. Returns (df, stats_df,
    flagged2, feats, image_ids, global_scores, per_class_scores, flagged3).
    """
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

    df, _ = image_integrity_audit(df, img_dir, out_dir=out_dir)

    stats_df, flagged2 = detect_nonfood_pixel_outliers(df, img_dir, out_dir=out_dir, n_fails_required=n_fails_required)

    if skip_stage3:
        print("\n[Stage 3] Skipped (skip_stage3=True).")
        feats, image_ids, global_scores, per_class_scores, flagged3 = np.array([]), [], np.array([]), np.array([]), pd.DataFrame()
    else:
        feats, image_ids, global_scores, per_class_scores, flagged3 = detect_nonfood_embedding_outliers(
            df, img_dir, out_dir=out_dir, device=device, global_thr=global_thr, per_class_thr=per_class_thr
        )

    print(f"\n{'-' * 60}\nPIPELINE COMPLETE\n{'-' * 60}")
    print(f"  After Stage 1      : {len(df):,}")
    print(f"  Stage 2 for review : {len(flagged2):,}")
    if not skip_stage3:
        print(f"  Stage 3 for review : {len(flagged3):,}")
    print(f"{'-' * 60}\n")

    return df, stats_df, flagged2, feats, image_ids, global_scores, per_class_scores, flagged3


# Idempotency helpers (skip the 3-stage scan on repeat notebook runs)
# Stage 1+2 open every image and Stage 3 runs a ResNet-50 pass over the whole
# dataset — expensive to redo. The notebook checks outlier_stage_outputs_exist()
# itself and decides whether to call run_outlier_pipeline or reload via
# load_cached_outlier_outputs.

def outlier_stage_outputs_exist(out_dir: str, skip_stage3: bool = False) -> bool:
    """True if the outputs written unconditionally on every run (not the
    flagged-review CSVs, which only appear when something is flagged) exist
    under out_dir — the reliable "did this stage complete" signal.
    """
    out_path = Path(out_dir)
    expected = [out_path / "stage2_pixel_stats_full.csv"]
    if not skip_stage3:
        expected.append(out_path / "stage3_embeddings.npz")
    return all(p.exists() for p in expected)


def load_cached_outlier_outputs(csv_path: str, out_dir: str, skip_stage3: bool) -> tuple:
    """Reconstruct run_outlier_pipeline's 8-tuple return from cached files on
    disk (see outlier_stage_outputs_exist)."""
    out_path = Path(out_dir)
    raw_df = pd.read_csv(csv_path)
    col_map = {}
    for col in raw_df.columns:
        if col.lower() in ("image_id", "img_id", "img_name", "filename", "file_name", "image"):
            col_map[col] = "image_id"
        elif col.lower() in ("label", "class", "class_id", "category", "target"):
            col_map[col] = "label"
    raw_df = raw_df.rename(columns=col_map)

    # Stage 1 only saves the removed-images log, so df_clean is cheaply
    # reconstructed as raw_df minus those ids, without reopening image files
    s1_path = out_path / "removed_stage1_integrity.csv"
    if s1_path.exists():
        removed_ids = set(pd.read_csv(s1_path)["image_id"].astype(str))
        df_clean = raw_df[~raw_df["image_id"].astype(str).isin(removed_ids)].reset_index(drop=True)
    else:
        df_clean = raw_df.reset_index(drop=True)

    stats_df = pd.read_csv(out_path / "stage2_pixel_stats_full.csv")
    flagged2_path = out_path / "review_stage2_nonfood_pixelstats.csv"
    flagged2 = pd.read_csv(flagged2_path) if flagged2_path.exists() else stats_df.iloc[0:0].copy()

    if skip_stage3:
        feats, image_ids = np.array([]), []
        global_scores, per_class_scores = np.array([]), np.array([])
        flagged3 = pd.DataFrame()
    else:
        npz = np.load(out_path / "stage3_embeddings.npz", allow_pickle=True)
        feats = npz["feats"]
        image_ids = npz["ids"].tolist()
        global_scores = npz["global_scores"]
        per_class_scores = npz["per_class_scores"]
        flagged3_path = out_path / "review_stage3_nonfood_embedding.csv"
        flagged3 = pd.read_csv(flagged3_path) if flagged3_path.exists() else pd.DataFrame()

    return df_clean, stats_df, flagged2, feats, image_ids, global_scores, per_class_scores, flagged3
