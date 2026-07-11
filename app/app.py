"""
FoodNet Demo — Supervised vs Self-Supervised Food-251 classification.

Loads the overall-best supervised model and the winning self-supervised
model (backbone + traditional classifier) produced by the training notebook,
and serves single-image or batch predictions with a side-by-side comparison view.
"""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

import model_utils as mu

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="FoodNet Demo", page_icon="🍜", layout="wide")


# Cached resource loading (runs once per server process)

@st.cache_resource
def get_device() -> torch.device:
    from codes import utils as U
    return U.get_device()


@st.cache_resource
def get_supervised():
    try:
        model, info = mu.load_supervised_model(get_device())
        return model, info, None
    except mu.ArtifactError as e:
        return None, None, str(e)


@st.cache_resource
def get_self_supervised():
    try:
        backbone, classifier, info = mu.load_ssl_model(get_device())
        return backbone, classifier, info, None
    except mu.ArtifactError as e:
        return None, None, None, str(e)


@st.cache_resource
def get_class_names():
    return mu.load_class_names()


# Background styling

def inject_background(image_path: Path) -> None:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(12, 8, 8, 0.80), rgba(12, 8, 8, 0.80)),
                url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(12, 8, 8, 0.85);
        }}
        div[data-testid="stFileUploader"], div[data-testid="stExpander"] {{
            background-color: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


bg_path = APP_DIR / "assets" / "background.jpg"
if bg_path.exists():
    inject_background(bg_path)


# Load everything

device = get_device()
sl_model, sl_info, sl_error = get_supervised()
ssl_backbone, ssl_classifier, ssl_info, ssl_error = get_self_supervised()
class_names = get_class_names()

st.title("🍜 FoodNet — Supervised vs Self-Supervised")
st.caption("Food-251 image classification with a custom CNN (< 10M params, no pretrained weights).")

if sl_error:
    st.error(f"Supervised model unavailable: {sl_error}")
if ssl_error:
    st.error(f"Self-supervised model unavailable: {ssl_error}")

# Sidebar
st.sidebar.header("Settings")
mode_options = []
if sl_model is not None:
    mode_options.append("Supervised")
if ssl_backbone is not None:
    mode_options.append("Self-Supervised")
if sl_model is not None and ssl_backbone is not None:
    mode_options.append("Both side-by-side")

if not mode_options:
    st.stop()

mode = st.sidebar.radio("Prediction mode", mode_options, index=len(mode_options) - 1)
top_k = st.sidebar.slider("Top-K predictions", min_value=1, max_value=10, value=5)

# Model card
with st.expander("Model card", expanded=False):
    comparison_df = mu.load_sl_vs_ssl_comparison()
    cols = st.columns(2)
    if sl_info:
        with cols[0]:
            st.subheader("Supervised")
            st.write(f"**Architecture:** {sl_info['architecture']}")
            st.write(f"**Parameters:** {sl_info['params_M']:.2f} M")
            st.write(f"**Paradigm:** {sl_info['paradigm']}")
            st.write("**Validation metrics:**")
            st.json({k: v for k, v in sl_info["metrics"].items()})
    if ssl_info:
        with cols[1]:
            st.subheader("Self-Supervised")
            st.write(f"**Architecture:** {ssl_info['architecture']}")
            st.write(f"**Parameters:** {ssl_info['params_M']:.2f} M")
            st.write(f"**Paradigm:** {ssl_info['paradigm']}")
            st.write("**Validation metrics:**")
            st.json({k: v for k, v in ssl_info["metrics"].items()})
    if comparison_df is not None:
        st.write("**Full supervised-vs-self-supervised comparison (from the training notebook):**")
        st.dataframe(comparison_df, width="stretch")


# Main panel: upload + predict

st.subheader("Upload food photo(s)")
uploaded_files = st.file_uploader(
    "Single image or a batch — a list of length 1 is just a single upload.",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to see predictions.")
    st.stop()

summary_rows = []

for uploaded_file in uploaded_files:
    st.markdown("---")
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not open {uploaded_file.name}: {e}")
        continue

    tensor = mu.preprocess_image(image)

    if mode == "Both side-by-side":
        thumb_col, sl_col, ssl_col = st.columns([1, 2, 2])
    else:
        thumb_col, pred_col = st.columns([1, 2])

    with thumb_col:
        st.image(image, caption=uploaded_file.name, width="stretch")

    row = {"filename": uploaded_file.name}

    def render_predictions(container, title, preds):
        with container:
            st.markdown(f"**{title}**")
            for name, conf in preds:
                st.write(f"{name} — {conf * 100:.1f}%")
                st.progress(min(max(conf, 0.0), 1.0))

    if mode in ("Supervised", "Both side-by-side") and sl_model is not None:
        sl_preds = mu.predict_supervised(sl_model, tensor, device, class_names, k=top_k)
        target = sl_col if mode == "Both side-by-side" else pred_col
        render_predictions(target, "Supervised", sl_preds)
        row["supervised_predicted_class"] = sl_preds[0][0]
        row["supervised_confidence"] = sl_preds[0][1]

    if mode in ("Self-Supervised", "Both side-by-side") and ssl_backbone is not None:
        ssl_preds = mu.predict_self_supervised(ssl_backbone, ssl_classifier, tensor, device, class_names, k=top_k)
        target = ssl_col if mode == "Both side-by-side" else pred_col
        render_predictions(target, "Self-Supervised", ssl_preds)
        row["self_supervised_predicted_class"] = ssl_preds[0][0]
        row["self_supervised_confidence"] = ssl_preds[0][1]

    summary_rows.append(row)

# Batch summary + CSV download (also shown for a single upload)
st.markdown("---")
st.subheader("Summary")
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width="stretch")
st.download_button(
    "Download predictions as CSV",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="foodnet_predictions.csv",
    mime="text/csv",
)
