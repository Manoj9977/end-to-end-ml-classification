
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

from model.common import (
    MODEL_DIR,
    MODEL_FILES,
    TARGET,
    compute_metrics,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
)

st.title("Breast Cancer Classification")
st.subheader("End-to-End Machine Learning Model Evaluation")

st.caption(
    "Benign = 0 | Malignant = 1 | "
    "Five pre-trained classification models are evaluated."
)


# ============================================================
# 2. LOAD MODEL AND CONFIGURATION FILES
# ============================================================

@st.cache_resource
def load_model(filename):
    """Load a previously trained machine learning model."""

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


@st.cache_data
def load_json_file(filename):
    """Load JSON configuration or evaluation data."""

    file_path = MODEL_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    return json.loads(
        file_path.read_text(encoding="utf-8")
    )


try:
    features = load_json_file("feature_columns.json")
    class_names = load_json_file("class_names.json")
    saved_metrics = load_json_file("metrics.json")

except (FileNotFoundError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load project artifacts: {exc}")
    st.stop()


# ============================================================
# 3. SIDEBAR OPTIONS
# ============================================================

st.sidebar.title("Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    options=list(MODEL_FILES.keys()),
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv"],
    help="Upload test_data.csv containing 30 features and target.",
)

st.sidebar.divider()

st.sidebar.info(
    "Select a trained model and upload a CSV file "
    "to evaluate its classification performance."
)


# ============================================================
# 4. DISPLAY TRAINING RESULTS
# ============================================================

with st.expander(
    "View Model Evaluation Results from Training",
    expanded=False,
):

    training_results = pd.DataFrame(
        saved_metrics
    ).T

    st.dataframe(
        training_results.round(4),
        use_container_width=True,
    )


# ============================================================
# 5. UPLOAD AND READ CSV DATASET
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload test_data.csv using the sidebar "
        "to start model evaluation."
    )

    st.stop()


try:

    df = pd.read_csv(uploaded_file)

except Exception as exc:

    st.error(f"Unable to read uploaded CSV file: {exc}")
    st.stop()


if df.empty:

    st.error("The uploaded CSV file is empty.")
    st.stop()


# ============================================================
# 6. VALIDATE DATASET COLUMNS
# ============================================================

required_columns = features + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


# Select the required columns in the expected order.

df = df[required_columns].copy()


# ============================================================
# 7. DATA PREPROCESSING AND VALIDATION
# ============================================================

df = df.apply(
    pd.to_numeric,
    errors="coerce",
)

if df.isna().any().any():

    st.error(
        "The uploaded dataset contains missing or "
        "non-numeric values. Please clean the dataset."
    )

    st.stop()


x = df[features]
y = df[TARGET]


# Validate target values.

if not set(y.unique()).issubset({0, 1}):

    st.error(
        "Invalid target values. Expected 0 (Benign) "
        "or 1 (Malignant)."
    )

    st.stop()


if y.nunique() < 2:

    st.error(
        "Both target classes must be present "
        "to calculate AUC."
    )

    st.stop()


y = y.astype(int)


# ============================================================
# 8. DISPLAY DATASET INFORMATION
# ============================================================

st.subheader("Uploaded Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    len(df),
)

col2.metric(
    "Total Features",
    len(features),
)

col3.metric(
    "Target Classes",
    y.nunique(),
)

with st.expander("Preview Uploaded Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )


# ============================================================
# 9. EVALUATE SELECTED MODEL
# ============================================================

try:

    selected_model = load_model(
        MODEL_FILES[model_name]
    )

    predictions, metrics = compute_metrics(
        selected_model,
        x,
        y,
    )

except Exception as exc:

    st.error(
        f"Error during model evaluation: {exc}"
    )

    st.stop()


st.divider()

st.subheader(
    f"Evaluation Results: {model_name}"
)


# ============================================================
# 10. DISPLAY PERFORMANCE METRICS
# ============================================================

metric_columns = st.columns(6)

for column, (metric_name, value) in zip(
    metric_columns,
    metrics.items(),
):

    column.metric(
        label=metric_name,
        value=f"{value:.4f}",
    )


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

st.divider()

st.subheader("Confusion Matrix and Classification Report")

left, right = st.columns(2)


with left:

    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

with right:

    st.markdown("### Classification Report")

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).T

    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )


# ============================================================
# 13. COMPARE ALL FIVE TRAINED MODELS
# ============================================================

st.divider()

st.subheader(
    "Comparison of All Five Models on Uploaded Dataset"
)

comparison_results = {}

try:

    for name, filename in MODEL_FILES.items():

        model = load_model(filename)

        _, model_metrics = compute_metrics(
            model,
            x,
            y,
        )

        comparison_results[name] = model_metrics


    comparison_df = pd.DataFrame(
        comparison_results
    ).T

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
    )

except Exception as exc:

    st.error(
        f"Unable to compare all models: {exc}"
    )

    st.stop()


# ============================================================
# 14. FINAL STATUS
# ============================================================

st.success(
    "Model evaluation and comparison completed successfully!"
)

st.caption(
    "Note: This application evaluates predictions against "
    "known target labels. It is intended for academic purposes "
    "and is not a clinically validated diagnostic system."
)
