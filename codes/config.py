"""
FoodNet Configuration
=======================
Centralised hyperparameters for the Food-251 recognition project (ML for
Modelling exam): custom CNN (< 10M params, no pretrained weights), solved as
both Supervised and Self-Supervised Learning. Defaults target Apple Silicon
(MPS), falling back to CUDA/CPU.
"""

from pathlib import Path

from . import utils as _utils

# Reproducibility
SEED = 42

# Device: resolved via utils.get_device() -> CUDA > MPS > CPU
DEVICE = "auto"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "dataset"
IMAGE_DIR    = DATA_DIR / "train_set"
TEST_IMAGE_DIR = DATA_DIR / "test_set"
TRAIN_CSV    = DATA_DIR / "train_labels.csv"
TEST_CSV     = DATA_DIR / "test_labels.csv"
CLASS_LIST_PATH = DATA_DIR / "class_list.txt"          # "<id> <name>" per line
CLEAN_CSV    = PROJECT_ROOT / "results" / "train_labels_clean.csv"  # post-outlier manifest
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Dataset facts (fixed by the exam spec)
NUM_CLASSES = 251          # must stay 251; only images-per-class may be capped
INPUT_SIZE  = 224
CHANNELS    = 3             # RGB — food colour is discriminative, do not greyscale

# Public test split has no labels, so validation (carved from train) is our test set
VAL_SPLIT  = 0.15
STRATIFY   = True           # stratified so every class appears in validation

MAX_IMAGES_PER_CLASS = None  # optional per-class cap for compute reasons; None = use all

# Supervised-learning training
BATCH_SIZE    = 64
NUM_EPOCHS    = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTHING = 0.1        # helps with 251 fine-grained classes

# NUM_WORKERS resolved per-device below: MPS->0 (benchmarked slower otherwise),
# CUDA->auto-benchmarked and cached in results/num_workers_benchmark.json, CPU->fallback.
NUM_WORKERS_FALLBACK = 4
_DEVICE_RESOLVED = _utils.get_device(DEVICE)
NUM_WORKERS = _utils.select_num_workers(
    _DEVICE_RESOLVED, results_dir=RESULTS_DIR, cpu_fallback=NUM_WORKERS_FALLBACK,
)

# Idempotency: skip and reload cached outputs instead of recomputing when False
FORCE_RECOMPUTE_DATA = False   # outlier pipeline / cleaned manifest / audit report
FORCE_RETRAIN        = False   # supervised + SSL tuning/training checkpoints

# Custom model — both architectures in codes.model.MODEL_REGISTRY are trained
# and compared (foodnet30, foodnet46), so no single "active" one is selected here
DROPOUT = 0.3
WIDTH_MULT = 1.0             # global channel multiplier; lower to shrink the model

# Loss / imbalance handling — 100-600 images/class, moderate imbalance
LOSS_TYPE   = "weighted_ce"  # "ce" | "weighted_ce" | "focal"
FOCAL_GAMMA = 2.0
USE_WEIGHTED_SAMPLER = False   # pick one correction: sampler XOR loss weights
CLASS_WEIGHT_SCHEME = "sqrt_inv"   # "inv" | "sqrt_inv" | "effective"

# Hyperparameter tuning — early-stop patience local to the search loop,
# separate from Trainer's own early stopping (PATIENCE below, Phase C only)
TUNE_EARLY_STOP_PATIENCE     = 10
SSL_TUNE_EARLY_STOP_PATIENCE = 10
TUNE_EARLY_STOP_CHECK_BATCHES = 10   # batches sampled for the cheap per-epoch plateau check

# TUNE_MODE selects the block below; keep "fast_dev" as default so re-running a
# notebook cell can't accidentally kick off a multi-hour "full" search
TUNE_MODE = "fast_dev"   # "fast_dev" | "full"

if TUNE_MODE == "full":
    TUNE_STRATEGY         = "successive_halving"
    TUNE_N_RANDOM_CONFIGS = 20
    TUNE_PROBE_EPOCHS     = 5
    SSL_TUNE_PROBE_EPOCHS = 10   # tune_ssl always does an exhaustive grid
elif TUNE_MODE == "fast_dev":
    TUNE_STRATEGY         = "random"
    TUNE_N_RANDOM_CONFIGS = 10
    TUNE_PROBE_EPOCHS     = 4
    SSL_TUNE_PROBE_EPOCHS = 3
else:
    raise ValueError(f"Unknown TUNE_MODE '{TUNE_MODE}'. Choose: fast_dev, full.")

TUNE_SELECTION_METRIC = "f1_macro"   # macro-F1, not accuracy — ~19:1 imbalance
TUNE_SUBSET_IMAGES_PER_CLASS = 100   # cap applies to search (Phase A/B) only

# Baseline sec/batch for the tuning ETA (foodnet46, batch 64, MPS FP32);
# overridden below with a real measurement when running on CUDA
BENCHMARKED_SEC_PER_BATCH = 0.66

if _DEVICE_RESOLVED.type == "cuda":
    _cuda_benchmark = _utils.load_num_workers_benchmark(_DEVICE_RESOLVED, results_dir=RESULTS_DIR)
    if _cuda_benchmark is not None:
        _measured = _cuda_benchmark.get("sec_per_batch", {}).get(str(NUM_WORKERS))
        if _measured is not None:
            BENCHMARKED_SEC_PER_BATCH = _measured

# Self-Supervised Learning (pretext) — same backbone, no labels, then frozen
# features feed a traditional classifier
SSL_METHOD          = "simclr"   # "simclr" | "rotation"
SSL_EPOCHS          = 100
SSL_BATCH_SIZE      = 64
SSL_LEARNING_RATE   = 1e-3
SSL_WEIGHT_DECAY    = 1e-4
SSL_TEMPERATURE     = 0.5        # SimCLR NT-Xent temperature
SSL_PROJECTION_DIM  = 128
SSL_CLASSIFIER      = "logreg"   # "logreg" | "linear_svm" | "knn"

# Augmentation — continuous intensity knob (0=off, 1=aggressive), 0.5 = original pipeline
AUGMENTATION_INTENSITY = 0.5
# Tail classes get AUGMENTATION_INTENSITY * TAIL_AUG_BOOST; also used by
# evaluate.py to report tail-vs-head metrics separately
TAIL_CLASS_FRACTION = 0.2
TAIL_AUG_BOOST       = 1.4
USE_TAIL_AWARE_AUGMENTATION = True

# Sample-mixing regulariser (Trainer.run_epoch) — CutMix preserves texture
# patches, useful for fine-grained classes (e.g. garlic bread vs. focaccia)
MIX_METHOD   = "mixup"    # "none" | "mixup" | "cutmix"
MIXUP_ALPHA  = 0.2        # Beta(0.2, 0.2)
CUTMIX_ALPHA = 1.0        # Beta(1, 1)

WARMUP_EPOCHS = 5     # linear LR ramp before cosine annealing, prevents early BN instability

LOG_PER_CLASS_EVERY = 10   # epochs between full per-class val F1 logging (0 = end only)

# If a class drops below this after outlier auditing, the summary report flags it
OUTLIER_MIN_CLASS_REMAINING = 15

# Early stopping (Phase C)
PATIENCE  = 12
MIN_DELTA = 1e-4

# Mixed precision (AMP) — off on MPS by default: measured ~14% slower than FP32
# on this machine (MacBook Air M4); GradScaler stays CUDA-only regardless
AMP_MPS_ENABLED = False
AMP_MPS_DTYPE   = "float16"   # only used when AMP_MPS_ENABLED=True
