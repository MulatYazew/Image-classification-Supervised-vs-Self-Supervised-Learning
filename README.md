# FoodNet — Supervised vs Self-Supervised Learning

Custom-CNN image classification on the Food-251 dataset (251 classes,
uncontrolled input size, roughly 34 to 656 images per class), solved under two
paradigms as required by the exam specification:

1. Supervised Learning (SL) — a custom CNN (under 10 M parameters, no pretrained
   weights) trained end-to-end with labels.
2. Self-Supervised Learning (SSL) — the same backbone pretrained on the images
   ignoring the labels (SimCLR contrastive or rotation pretext), then frozen,
   feature-extracted, and classified with a traditional classifier (logistic
   regression, linear SVM, or kNN).

The official test split has no public ground truth, so the validation set is the
test set and is stratified-split out of the training data (all 251 classes
present in both partitions).

## Project layout

```
foodnet/
├── codes/
│   ├── config.py                 # central, dataset-aware hyperparameters
│   ├── utils.py                  # seeds, device selection, param-budget guard
│   ├── data_handler.py           # manifest, stratified split, augmentation, datasets
│   ├── model.py                  # custom CNNs (all verified under 10M params) + registry
│   ├── loss_function.py          # CE / weighted-CE / focal + NT-Xent (SimCLR)
│   ├── train.py                  # supervised Trainer with early stopping
│   ├── self_supervised.py        # SimCLR/rotation pretrain, features, traditional classifier
│   ├── hyperparameter_tuning.py  # grid search for the SL and SSL tasks
│   ├── evaluate.py               # accuracy/precision/recall/F1, confusion, SL-vs-SSL
│   ├── gradcam.py                # Grad-CAM explanations for the custom CNN
│   └── outlier_handler.py        # 3-stage food-image outlier audit
├── FoodNet_Supervised_SelfSupervised.ipynb   # main pipeline (both tasks)
├── dataset/                      # put train_labels.csv + train_set/ here
├── models/                       # checkpoints (created at runtime)
└── results/                      # metrics, figures, tuning tables
```

## Expected data format

dataset/train_labels.csv with at least an image column (image_id / filename /
image / id) and a label column (label / class / category / target).
Images live in dataset/train_set/. String labels are auto-mapped to integer ids.

## Custom models (all under 10 M parameters)

| Model          | Params  | Feature dim | Role                                             |
|----------------|---------|-------------|--------------------------------------------------|
| foodnet        | 7.64 M  | 1024        | Proposed — residual depthwise-separable CNN + SE |
| foodnet_lite   | 0.45 M  | 256         | Lightweight baseline / fast sweeps               |

Verify any time with:

```bash
python -m codes.model
```

## Running

Open FoodNet_Supervised_SelfSupervised.ipynb and run top to bottom.
It performs: preprocessing and outlier audit, stratified split, custom CNN build
with budget check, then Task A (SL) where the model is tuned with a full grid,
the best configuration is selected by validation accuracy and retrained to
convergence, then Task B (SSL) where SimCLR and rotation are tuned separately and
the better pretext method is selected and retrained, then the SL-vs-SSL
comparison, and finally Grad-CAM explanations.

## Notes
- No pretrained weights are used anywhere (forbidden by the spec).
- The number of classes is always 251; only images-per-class may be capped, for
  documented computational reasons (config.MAX_IMAGES_PER_CLASS).
- Pick one class-frequency correction: weighted sampler or loss weights, never both.
- The device is chosen automatically (CUDA, then Apple Silicon MPS, then CPU);
  mixed precision is enabled only on CUDA.

## Plagiarism statement
This project was developed by the author or authors named in the report. No form
of plagiarism, including the use of generative tools to produce the submitted
material, was used. All external sources are properly cited in the report.