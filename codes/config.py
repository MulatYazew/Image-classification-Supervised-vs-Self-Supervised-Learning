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
MODEL_ARCHITECTURE = "foodnet"
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
AUGMENTATION_INTENSITY = 0.5
USE_MIXUP    = True
MIXUP_ALPHA  = 0.2    # Beta(0.2, 0.2) — strong regularisation for 251-class food task
WARMUP_EPOCHS = 5     # Linear LR ramp before cosine annealing; prevents early BN instability
AUGMENTATION_INTENSITY = 0.5
USE_MIXUP    = True
MIXUP_ALPHA  = 0.2    # Beta(0.2, 0.2) — strong regularisation for 251-class food task
WARMUP_EPOCHS = 5     # Linear LR ramp before cosine annealing; prevents early BN instability
#  Early stopping 
PATIENCE  = 12             # v2: increased from 8; improved model needs more time to settle
MIN_DELTA = 1e-4