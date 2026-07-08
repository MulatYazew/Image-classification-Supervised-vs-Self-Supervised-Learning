"""
FoodNet Configuration
=======================
All hyperparameters for the Food-251 recognition project are centralised here.

This project follows the exam specification (ML for Modelling — Supervised Learning):

  * Task ........ image classification on a 251-class food-recognition dataset,
                  solved as BOTH Supervised Learning (SL) and Self-Supervised
                  Learning (SSL → feature extraction → traditional classifier).
  * Backbone .... a *custom* CNN with < 10 M parameters (NO pretrained weights).
  * Classes ..... exactly 251 (this number is fixed by the spec and must never
                  be reduced — only the number of images per class may be cut
                  for documented computational reasons).
  * Splits ...... the public test set has no ground truth, so the *validation*
                  set is our test set, and it is carved out of the training set.

The defaults below are tuned for Apple Silicon (MPS) but fall back to CUDA/CPU.
"""

from pathlib import Path

import utils as _utils

#  Reproducibility 
SEED = 42

#  Device 
# Priority handled in utils.get_device(): MPS (Apple Silicon) → CUDA → CPU.
DEVICE = "auto"   # auto → CUDA if available, else MPS (Apple Silicon), else CPU

#  Paths 
# codes/ is one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "dataset"
IMAGE_DIR    = DATA_DIR / "train_set"            # raw training images
TEST_IMAGE_DIR = DATA_DIR / "test_set"
TRAIN_CSV    = DATA_DIR / "train_labels.csv"     # manifest: image_id,label of training data
TEST_CSV     = DATA_DIR / "test_labels.csv"      # manifest: image_id, label of the test data
CLEAN_CSV    = PROJECT_ROOT / "results" / "train_labels_clean.csv"  # post-outlier manifest
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

#  Dataset facts (fixed by the exam specification) 
NUM_CLASSES = 251          # MUST remain 251 — never reduce the number of classes
INPUT_SIZE  = 224          # common square size; input images have uncontrolled size
CHANNELS    = 3            # RGB (food colour is highly discriminative — do NOT greyscale)

# The official test split has no public ground truth. Our "test" = validation,
# and the validation set is extracted from the training set.
VAL_SPLIT  = 0.15          # fraction of TRAIN held out for validation (= our test)
# Stratified by class so every one of the 251 classes appears in validation.
STRATIFY   = True

# Optional documented sub-sampling for computational reasons.
# The number of CLASSES stays 251; only images-per-class may be capped.
# Set to an int (e.g. 250) to cap, or None to use every available image.
MAX_IMAGES_PER_CLASS = None

#  Supervised-Learning training 
BATCH_SIZE    = 64
NUM_EPOCHS    = 80
LEARNING_RATE = 5e-4       # v2: lowered from 1e-3; more stable for deep from-scratch nets
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTHING = 0.1      # mild smoothing helps with 251 fine-grained classes

# NUM_WORKERS is now resolved automatically per-device (utils.select_num_workers),
# called right here at import time -- right after get_device(), before any
# DataLoader is built -- so there is no manual benchmarking step on either
# machine:
#   * MPS  -> 0 immediately. Measured on this machine (MacBook Air M4, 10
#     cores): with cv2.setNumThreads(0) applied in every worker
#     (data_handler.worker_init_fn -- without it, cv2's own thread pool
#     fights the DataLoader's worker processes and workers>0 is actually
#     SLOWER/prone to hang), sweeping num_workers over the real training
#     pipeline (foodnet46, batch 64, 20-batch/3-repeat harness) gave:
#     0->1.09 s/batch, 2->1.05, 4->0.97, 6->0.96, 8->0.95. Gains plateau past
#     4 (this pipeline is compute-bound on MPS -- pure data loading measured
#     at only ~0.14 s/batch of the ~1.09 s/batch total, so overlapping it
#     with compute can save at most ~13%, which is what was observed) -- not
#     worth a benchmark, so MPS short-circuits straight to 0.
#   * CUDA -> auto-benchmarked candidates [4, 6, 8, 12] on THIS machine's real
#     training pipeline the FIRST time it runs there, then cached to
#     results/num_workers_benchmark.json keyed by the GPU name (every run
#     after that just loads the cached value). Prefers 6 or 8 unless another
#     candidate wins by >15%. Pass force_rebenchmark=True to
#     utils.select_num_workers to ignore the cache (e.g. after a driver/
#     hardware change).
#   * CPU  -> min(NUM_WORKERS_FALLBACK, os.cpu_count()), no benchmark.
NUM_WORKERS_FALLBACK = 4   # historical MPS-benchmarked value; also the CPU cap
_DEVICE_RESOLVED = _utils.get_device(DEVICE)
NUM_WORKERS = _utils.select_num_workers(
    _DEVICE_RESOLVED, results_dir=RESULTS_DIR, cpu_fallback=NUM_WORKERS_FALLBACK,
)

#  Idempotency (Apple Silicon / long-pipeline reruns)
# When False (default), notebook cells that already have their expected
# output(s) on disk (cleaned manifests, tuning results, checkpoints, ...)
# print "SKIPPING" and reload them instead of recomputing. Set True to ignore
# every cache and force a full clean rerun; or just delete the specific
# output file(s) you want to regenerate for a more targeted rerun.
FORCE_RECOMPUTE_DATA = False   # outlier pipeline / cleaned manifest / audit report
FORCE_RETRAIN        = False   # supervised + SSL tuning/training checkpoints

#  Custom model
# No pretrained backbones are allowed. codes.model.MODEL_REGISTRY holds both
# custom architectures ("foodnet30": 30-layer residual DWS+SE, "foodnet46":
# 46-layer MBConv, proposed) — every option is verified < 10 M parameters.
# The notebook trains and compares BOTH (run_supervised_pipeline per
# architecture), so there is no single "active" architecture to select here.
DROPOUT = 0.3
WIDTH_MULT = 1.0           # global channel multiplier; lower to shrink the model

#  Loss / imbalance handling 
# 100–600 images per class → moderate imbalance. Options: "ce" | "weighted_ce" | "focal".
LOSS_TYPE   = "weighted_ce"
FOCAL_GAMMA = 2.0
USE_WEIGHTED_SAMPLER = False   # pick ONE correction point (sampler XOR loss weights)
# Per-class weight scheme for compute_class_weights: "inv" | "sqrt_inv" | "effective".
CLASS_WEIGHT_SCHEME = "sqrt_inv"

#  Hyperparameter tuning
# TUNE_EARLY_STOP_PATIENCE / SSL_TUNE_EARLY_STOP_PATIENCE let a clearly-
# plateaued candidate stop before burning its full probe budget. This is a
# separate, local-to-the-search-loop mechanism from Trainer's own early
# stopping (PATIENCE below), which only applies to the Phase C full retrain.
TUNE_EARLY_STOP_PATIENCE     = 5
SSL_TUNE_EARLY_STOP_PATIENCE = 5
# probe_supervised's per-epoch early-stop CHECK (only engages when
# TUNE_EARLY_STOP_PATIENCE > 0) samples only this many batches of the
# Phase A/B tuning validation loader (tune_val_loader -- already a separate,
# smaller, documented subset from the REAL validation set; see
# TUNE_SUBSET_IMAGES_PER_CLASS below) as a cheap proxy signal for "has this
# candidate plateaued", instead of a full validation pass every epoch. The
# metric actually used to RANK/SELECT configs is still a full tune_val_loader
# pass, computed ONCE at the end of each candidate's loop (see
# probe_supervised's docstring) -- this only cuts the redundant PER-EPOCH cost.
TUNE_EARLY_STOP_CHECK_BATCHES = 10

# TUNE_MODE picks which block below actually drives TUNE_STRATEGY /
# TUNE_N_RANDOM_CONFIGS / TUNE_PROBE_EPOCHS / SSL_TUNE_PROBE_EPOCHS. This
# exists because "TUNE_STRATEGY=successive_halving, TUNE_N_RANDOM_CONFIGS=20,
# TUNE_PROBE_EPOCHS=5" does NOT mean 20 configs x 5 epochs -- successive
# halving keeps doubling the survivors' epoch budget each round, which comes
# to ~460 epoch-equivalents for ONE grid phase alone (see
# hyperparameter_tuning.estimate_successive_halving_epochs, and the
# "[tune-budget]" line every tune_supervised/tune_ssl call prints before it
# starts). Re-running a notebook cell with TUNE_MODE="full" left in place can
# silently kick off a multi-hour search -- "fast_dev" is the default so that
# doesn't happen by accident; switch to "full" deliberately for the report run.
TUNE_MODE = "fast_dev"   # "fast_dev" | "full"

if TUNE_MODE == "full":
    TUNE_STRATEGY         = "successive_halving"
    TUNE_N_RANDOM_CONFIGS = 20     # sampled configs when TUNE_STRATEGY != "grid"
    TUNE_PROBE_EPOCHS     = 5
    SSL_TUNE_PROBE_EPOCHS = 10     # tune_ssl always does an exhaustive grid (no
                                    # random/successive-halving option there yet)
elif TUNE_MODE == "fast_dev":
    TUNE_STRATEGY         = "random"
    TUNE_N_RANDOM_CONFIGS = 10
    TUNE_PROBE_EPOCHS     = 5
    SSL_TUNE_PROBE_EPOCHS = 3
else:
    raise ValueError(f"Unknown TUNE_MODE '{TUNE_MODE}'. Choose: fast_dev, full.")

# Rank configs by macro-F1, not accuracy: with ~19:1 class imbalance a config
# can look fine on accuracy at epoch 5 while ignoring the tail classes.
TUNE_SELECTION_METRIC = "f1_macro"    # "f1_macro" | "accuracy"
# Optional documented cap on the SEARCH data only (Phase A/B); Phase C's final
# confirmation run always uses the full dataset regardless of this value.
TUNE_SUBSET_IMAGES_PER_CLASS = 100

# Measured sec/batch for the tuning-time estimate printed by tune_supervised /
# tune_ssl ("[tune-budget]" line) -- foodnet46, batch 64, NUM_WORKERS=4, FP32
# (MPS autocast measured slower here, see AMP_MPS_ENABLED above), 20-batch/
# 3-repeat harness. This is the MacBook Air M4 (MPS) default; treated as an
# estimate, not a guarantee.
BENCHMARKED_SEC_PER_BATCH = 0.66

# On CUDA, override the Mac-only default above with THIS machine's own
# measurement from the NUM_WORKERS auto-benchmark (same harness: foodnet46,
# batch 64, 20-batch/3-repeat, just at the num_workers value actually chosen
# for this GPU) -- so the "[tune-budget]" wall-clock ETA reflects real
# Legion-class hardware instead of the Mac's number. Silently keeps the Mac
# default if the benchmark cache doesn't have an entry for some reason (e.g.
# force_rebenchmark hasn't run yet); it will pick up the real number on the
# very next run, once NUM_WORKERS above has cached it.
if _DEVICE_RESOLVED.type == "cuda":
    _cuda_benchmark = _utils.load_num_workers_benchmark(_DEVICE_RESOLVED, results_dir=RESULTS_DIR)
    if _cuda_benchmark is not None:
        _measured = _cuda_benchmark.get("sec_per_batch", {}).get(str(NUM_WORKERS))
        if _measured is not None:
            BENCHMARKED_SEC_PER_BATCH = _measured

#  Self-Supervised Learning (pretext) 
# SSL pretrains the SAME custom backbone on the images WITHOUT labels, then we
# freeze it, extract features, and fit a traditional classifier on top.
# Choices: "simclr" (contrastive) | "rotation" (4-way rotation prediction)
SSL_METHOD          = "simclr"
SSL_EPOCHS          = 100
SSL_BATCH_SIZE      = 64
SSL_LEARNING_RATE   = 1e-3
SSL_WEIGHT_DECAY    = 1e-4
SSL_TEMPERATURE     = 0.5       # SimCLR NT-Xent temperature
SSL_PROJECTION_DIM  = 128       # projection-head output dimension
# Traditional classifier fitted on frozen SSL features:
# "logreg" (logistic regression) | "linear_svm" | "knn"
SSL_CLASSIFIER      = "logreg"

#  Augmentation
# Scales augmentation magnitude/probability continuously (0=off, 1=aggressive);
# 0.5 reproduces the original hand-tuned pipeline exactly. Wired into
# data_handler.get_transforms so a single knob drives the report's
# augmentation-ablation runs instead of hand-editing the pipeline each time.
AUGMENTATION_INTENSITY = 0.5
# Class-aware augmentation: the smallest TAIL_CLASS_FRACTION of classes (by
# image count) get AUGMENTATION_INTENSITY * TAIL_AUG_BOOST instead of the
# uniform intensity, so data-poor classes see more diverse synthetic variation
# per epoch. Same fraction is reused by evaluate.py to report tail-vs-head
# metrics separately (see codes.data_handler.compute_tail_classes).
TAIL_CLASS_FRACTION = 0.2
TAIL_AUG_BOOST       = 1.4
USE_TAIL_AWARE_AUGMENTATION = True

# Sample-mixing regulariser applied inside Trainer.run_epoch. CutMix tends to
# help more than MixUp on fine-grained, texture-heavy classes (e.g. garlic
# bread vs. focaccia) because it preserves local texture patches instead of
# globally blending pixel values.
MIX_METHOD   = "mixup"    # "none" | "mixup" | "cutmix"
MIXUP_ALPHA  = 0.2        # Beta(0.2, 0.2) — strong regularisation for 251-class food task
CUTMIX_ALPHA = 1.0        # Beta(1, 1) — uniform box-size sampling, the standard CutMix default

WARMUP_EPOCHS = 5     # Linear LR ramp before cosine annealing; prevents early BN instability

#  Per-class metric logging
# Epochs between full per-class val F1 logging during training (0 = only at
# the very end). Cheap relative to an epoch, but not free at 251 classes, so
# it is not computed every epoch by default.
LOG_PER_CLASS_EVERY = 10

#  Outlier-handling report
# If any class has fewer than this many images left after all 3 audit stages,
# the summary report (outlier_handler.audit_summary_report) flags it instead
# of silently accepting a gutted class.
OUTLIER_MIN_CLASS_REMAINING = 15

#  Early stopping
PATIENCE  = 12             # epochs without val-loss improvement before stopping Phase C training
MIN_DELTA = 1e-4

#  Mixed precision (AMP)
# train.py / hyperparameter_tuning.py / self_supervised.py were previously all
# hardcoded to CUDA-only autocast; the plumbing to extend it to MPS now
# exists (codes.utils.amp_enabled / amp_dtype_for), but MPS autocast is OFF
# by default here because it MEASURED SLOWER, not faster, on this machine
# (MacBook Air M4): FP32 ran at 0.658 s/batch on foodnet46 (batch 64) vs
# 0.748-0.749 s/batch for autocast in EITHER float16 or bfloat16 -- ~14%
# slower, reproduced across repeated runs. GradScaler stays CUDA-only
# regardless (MPS doesn't need/support the same overflow-scaling machinery).
# Set AMP_MPS_ENABLED=True to opt back in if a future torch/MPS version (or a
# different, larger model) changes this -- re-run the benchmark first.
AMP_MPS_ENABLED = False
AMP_MPS_DTYPE   = "float16"    # only consulted when AMP_MPS_ENABLED=True; float16 and
                                # bfloat16 measured identically (~0.748s/batch either way)