---
title: FoodNet Demo
emoji: 🍜
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
---

# FoodNet Demo

Interactive demo for the Food-251 classification project: compares a **Supervised**
custom CNN against a **Self-Supervised** (SimCLR/rotation pretext + frozen features +
traditional classifier) model trained on the same architecture and data split.

## Running locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app expects the training notebook
(`notebooks/FoodNet_Supervised_Self_Supervised.ipynb`) to have already produced:

- `results/sl_model_comparison.csv` + `models/<architecture>/best_model.pth` +
  `results/<architecture>_best_hparams.json` (the winning supervised architecture)
- `models/ssl_best/backbone.pth` + `models/ssl_best/classifier.joblib` +
  `results/ssl_best_hparams.json` (the winning self-supervised method)

If any of these are missing, the app shows a clear error for that model instead of
crashing, and still serves whichever paradigm IS available.

## Deploying to Hugging Face Spaces

Push this `app/` directory as the root of a Streamlit Space (the YAML front-matter
above is required by Spaces). The Space also needs the `codes/` package, `dataset/class_list.txt`,
and the `models/`/`results/` artifacts listed above alongside it.
