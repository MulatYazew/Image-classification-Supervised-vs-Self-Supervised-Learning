
# FoodNet — Supervised vs Self-Supervised Learning

Custom-CNN image classification on the **Food-251** dataset (251 classes, 100–600
images/class, uncontrolled input size), solved under **two paradigms** as required
by the exam specification:

1. **Supervised Learning (SL)** — a custom CNN (**< 10 M parameters, no pretrained
   weights**) trained end-to-end with labels.
2. **Self-Supervised Learning (SSL)** — the *same* backbone pretrained on the images
   **ignoring the labels** (SimCLR contrastive / rotation pretext), then frozen,
   feature-extracted, and classified with a **traditional classifier** (logistic
   regression / linear SVM / kNN).

The official test split has no public ground truth, so the **validation set is the
test set** and is **stratified-split out of the training data** (all 251 classes
present in both partitions).

## Project layout

```
251_food_classification_model/
├── codes/
│   ├── config.py               # central, dataset-aware hyperparameters
│   ├── utils.py                # seeds, device, param-budget guard
│   ├── data_handler.py         # manifest, stratified split, augmentation, datasets
│   ├── model.py                # custom CNNs (all verified < 10M params) + registry
│   ├── loss.py                 # CE / weighted-CE / focal + NT-Xent (SimCLR)
│   ├── train.py                # supervised Trainer + grid_search
│   ├── self_supervised.py      # SimCLR/rotation pretrain → features → trad. classifier
│   ├── evaluate.py             # accuracy/precision/recall/F1, confusion, SL-vs-SSL
│   ├── gradcam.py              # Grad-CAM explanations for the custom CNNs
│   ├── hyperparamete_tuning.py # 
│   └── outlier_handler.py   # 3-stage food-image outlier audit
├── notebooks/
│   ├── Food_Supervised_vs_SelfSupervised.ipynb   # main pipeline (both tasks)
├── dataset/                 # put train_labels.csv + train_set/ here
├── models/                  # checkpoints (created at runtime)
└── results/                 # metrics, figures, cleaned manifest
```

## Expected data format

`dataset/train_labels.csv` with at least an image column (`image_id` / `filename` /
`image` / `id`) and a label column (`label` / `class` / `category` / `target`).
Images live in `dataset/train_set/`. String labels are auto-mapped to integer ids.

## Custom models (all < 10 M parameters)

| Model              | Params | Feature dim | Role                                  |
|--------------------|--------|-------------|---------------------------------------|
| `foodnet`          | ~4.1 M | 1024        | Proposed — depthwise-separable CNN    |
| `foodnet_lite`     | ~0.45 M| 256         | Lightweight baseline / fast sweeps    |

Verify any time with:

```bash
python -m codes.model
```

## Running

Open `notebooks/Food251_Supervised_vs_SelfSupervised.ipynb` and run top to bottom.
It performs: preprocessing → stratified split → custom CNN build (with budget check)
→ **Task A (SL)** training & evaluation → **Task B (SSL)** pretraining, feature
extraction & traditional classifier → SL-vs-SSL comparison → hyperparameter-tuning
hook → Grad-CAM.

## Notes
- No pretrained weights are used anywhere (forbidden by the spec).
- The number of classes is always **251**; only images-per-class may be capped, for
  documented computational reasons (`config.MAX_IMAGES_PER_CLASS`).
- Pick **one** class-frequency correction: weighted sampler *or* loss weights.

## Plagiarism statement
This project was developed by the author(s). No form of plagiarism — including the
use of ChatGPT or similar generative tools to produce the submitted material — was
used. All external sources are properly cited in the report.