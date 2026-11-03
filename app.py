
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

from model.common import MODEL_DIR, MODEL_FILES, TARGET, compute_metrics


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification - Model Evaluation")

st.caption(
    "Target: 0 = Benign, 1 = Malignant. "
    "Pre-trained models are loaded from disk. "
    "No retraining happens on uploaded data."
)


# --------------------------------------------------
# 2. Load trained models
# --------------------------------------------------

@st.cache_resource
def load_model(filename):
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    return joblib.load(model_path)


# --------------------------------------------------
# 3. Load feature names and class names
# --------------------------------------------------

try:
    features = json.loads(
        (MODEL_DIR / "feature_columns.json").read_text(
            encoding="utf-8"
        )
    )

    class_names = json.loads(
        (MODEL_DIR / "class_names.json").read_text(
            encoding="utf-8"
        )
    )

except FileNotFoundError as exc:
    st.error(f"Required configuration file not found: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Sidebar options
# --------------------------------------------------

st.sidebar.header("Options")

model_name = st.sidebar.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV (features + target)",
    type=["csv"]
)


# --------------------------------------------------
# 5. Display saved training results
# --------------------------------------------------

with st.expander(
    "Test-set results of all models recorded during training"
):
    try:
        saved = json.loads(
            (MODEL_DIR / "metrics.json").read_text(
                encoding="utf-8"
            )
        )

        st.dataframe(
            pd.DataFrame(saved).T.round(4),
            use_container_width=True
        )

    except FileNotFoundError:
        st.warning("Training metrics file not found.")


# --------------------------------------------------
# 6. Upload CSV file
# --------------------------------------------------

if uploaded_file is None:
    st.info(
        "Upload test_data.csv in the sidebar to evaluate "
        "the selected model."
    )
    st.stop()


# --------------------------------------------------
# 7. Read and validate uploaded CSV
# --------------------------------------------------

try:
    df = pd.read_csv(uploaded_file)

except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()


if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()


required_columns = features + [TARGET]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


df = df[required_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

if df.isna().any().any():
    st.error(
        "The CSV contains non-numeric or missing values. "
        "Please check the uploaded data."
    )
    st.stop()


# --------------------------------------------------
# 8. Separate input features and target
# --------------------------------------------------

x = df[features]
y = df[TARGET]

if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
    st.error(
        "Target must contain both classes: "
        "0 (Benign) and 1 (Malignant)."
    )
    st.stop()

y = y.astype(int)


# --------------------------------------------------
# 9. Evaluate selected model
# --------------------------------------------------

try:
    selected_model = load_model(MODEL_FILES[model_name])

    predictions, metrics = compute_metrics(
        selected_model,
        x,
        y
    )

except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()


st.subheader(f"Evaluation Results: {model_name}")

metric_columns = st.columns(6)

for column, (metric_name, value) in zip(
    metric_columns,
    metrics.items()
):
    column.metric(
        metric_name,
        f"{value:.4f}"
    )


# --------------------------------------------------
# 10. Confusion Matrix and Classification Report
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.markdown("### Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        confusion_matrix(y, predictions, labels=[0, 1]),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)


with right:

    st.markdown("### Classification Report")

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )


# --------------------------------------------------
# 11. Compare all five models
# --------------------------------------------------

st.subheader(
    "Comparison of All Five Models on Uploaded Test Data"
)

comparison_results = {}

for name, filename in MODEL_FILES.items():

    model = load_model(filename)

    _, model_metrics = compute_metrics(
        model,
        x,
        y
    )

    comparison_results[name] = model_metrics


comparison_df = pd.DataFrame(
    comparison_results
).T

st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)

st.success(
    "Model evaluation and comparison completed successfully!"
)
