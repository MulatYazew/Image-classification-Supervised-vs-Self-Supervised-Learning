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

# num_workers > 0 can hang on macOS with some DataLoader configs; keep low.
NUM_WORKERS = 0

#  Custom model 
# No pretrained backbones are allowed. Choose among the custom architectures
# defined in model.py — every option is verified < 10 M parameters.
# Choices: "food251net" (proposed) | "food251net_lite" (baseline)
MODEL_ARCHITECTURE = "foodnet46"
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
# Short probe budget per configuration during the search (the winner is then
# trained to convergence with NUM_EPOCHS / SSL_EPOCHS).
TUNE_PROBE_EPOCHS     = 5
SSL_TUNE_PROBE_EPOCHS = 10
# Search strategy: "grid" (exhaustive), "random" (sample N configs — better
# than grid once >~2 axes are combined, for the same compute budget), or
# "successive_halving" (start many configs at a short budget, keep the top
# half, double their budget, repeat — cheap-search-then-confirm in one loop).
TUNE_STRATEGY         = "successive_halving"
TUNE_N_RANDOM_CONFIGS = 20     # sampled configs when TUNE_STRATEGY != "grid"
# Rank configs by macro-F1, not accuracy: with ~19:1 class imbalance a config
# can look fine on accuracy at epoch 5 while ignoring the tail classes.
TUNE_SELECTION_METRIC = "f1_macro"    # "f1_macro" | "accuracy"
# Optional documented cap on the SEARCH data only (Phase A/B); Phase C's final
# confirmation run always uses the full dataset regardless of this value.
TUNE_SUBSET_IMAGES_PER_CLASS = 100

#  Self-Supervised Learning (pretext) 
# SSL pretrains the SAME custom backbone on the images WITHOUT labels, then we
# freeze it, extract features, and fit a traditional classifier on top.
# Choices: "simclr" (contrastive) | "rotation" (4-way rotation prediction)
SSL_METHOD          = "simclr"
SSL_EPOCHS          = 100
SSL_BATCH_SIZE      = 128
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
PATIENCE  = 12             # v2: increased from 8; improved model needs more time to settle
MIN_DELTA = 1e-4