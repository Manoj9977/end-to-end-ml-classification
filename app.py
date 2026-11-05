
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from model.common import (
    MODEL_DIR,
    MODEL_FILES,
    TARGET,
    compute_metrics,
)


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide",
)

st.title("🎗️ Breast Cancer Wisconsin Classification")

st.markdown(
    """
    **End-to-End Machine Learning Classification Project**

    This application trains and evaluates five machine learning
    classification algorithms using the Breast Cancer Wisconsin
    Diagnostic dataset.

    It provides model performance metrics, confusion matrices,
    ROC curves, Precision-Recall curves and model comparisons.
    """
)

st.divider()


# ============================================================
# 2. MODEL INFORMATION
# ============================================================

MODEL_DESCRIPTIONS = {
    "Logistic Regression":
        "Linear classification algorithm that estimates class probabilities.",

    "Decision Tree":
        "Tree-based algorithm that makes predictions using decision rules.",

    "K-Nearest Neighbours":
        "Distance-based algorithm that classifies observations using nearby samples.",

    "Naive Bayes (Gaussian)":
        "Probability-based classification algorithm using Bayes' theorem.",

    "Support Vector Machine":
        "Classification algorithm that identifies an optimal separating boundary.",
}


# ============================================================
# 3. LOAD MODEL AND JSON ARTIFACTS
# ============================================================

@st.cache_resource
def load_model(filename):

    model_path = Path(MODEL_DIR) / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


@st.cache_data
def load_json_file(filename):

    file_path = Path(MODEL_DIR) / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:

    features = load_json_file("feature_columns.json")
    class_names = load_json_file("class_names.json")
    saved_metrics = load_json_file("metrics.json")

except (FileNotFoundError, json.JSONDecodeError) as error:

    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


if not isinstance(features, list) or not features:

    st.error("Invalid feature_columns.json file.")
    st.stop()


if not isinstance(class_names, list) or len(class_names) != 2:

    st.error("class_names.json must contain two class names.")
    st.stop()


class_names = [str(name) for name in class_names]


# ============================================================
# 4. SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.title("Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    options=list(MODEL_FILES.keys()),
)

st.sidebar.markdown("---")

st.sidebar.subheader("Selected Model")

st.sidebar.write(
    MODEL_DESCRIPTIONS[selected_model_name]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic"
)

st.sidebar.caption(
    "Benign = 0 | Malignant = 1"
)


# ============================================================
# 5. SAVED TRAINING RESULTS
# ============================================================

st.header("1. Saved Model Training Results")

with st.expander(
    "View Training Results of All Five Models",
    expanded=False,
):

    try:

        if isinstance(saved_metrics, dict):

            saved_metrics_df = pd.DataFrame.from_dict(
                saved_metrics,
                orient="index",
            )

            saved_metrics_df.index.name = "Model"

            saved_metrics_df = saved_metrics_df.reset_index()

        elif isinstance(saved_metrics, list):

            saved_metrics_df = pd.DataFrame(saved_metrics)

        else:

            saved_metrics_df = pd.DataFrame()

        if not saved_metrics_df.empty:

            st.dataframe(
                saved_metrics_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.warning("No saved training results available.")

    except Exception as error:

        st.error(
            f"Unable to display training results: {error}"
        )


st.divider()


# ============================================================
# 6. UPLOAD TEST DATASET
# ============================================================

st.header("2. Upload Test Dataset")

st.markdown(
    """
    Upload the test dataset generated by `train_models.py`.
    The CSV must contain all required input features and the target column.
    """
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
)


if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to view model evaluation "
        "and visualization results."
    )

    st.stop()


# ============================================================
# 7. READ AND VALIDATE DATASET
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read uploaded dataset: {error}"
    )

    st.stop()


if df.empty:

    st.error("Uploaded dataset is empty.")
    st.stop()


if df.columns.duplicated().any():

    st.error("Dataset contains duplicate column names.")
    st.stop()


# Remove spaces around column names
df.columns = df.columns.astype(str).str.strip()

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


# Select required columns
df = df[required_columns].copy()


# Convert values into numeric format
try:

    df[required_columns] = df[required_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

except (ValueError, TypeError):

    st.error(
        "Dataset contains non-numeric values in the "
        "feature or target columns."
    )

    st.stop()


# Validate missing values
if df.isnull().values.any():

    st.error(
        "Dataset contains missing values. "
        "Please upload a cleaned dataset."
    )

    st.stop()


# Validate infinite values
if not np.isfinite(
    df[required_columns].to_numpy(dtype=float)
).all():

    st.error(
        "Dataset contains infinite values."
    )

    st.stop()


# Validate target column
if not df[TARGET].isin([0, 1]).all():

    st.error(
        "Target values must be 0 (Benign) or 1 (Malignant)."
    )

    st.stop()


x = df[features]

y = df[TARGET].astype(int)


if y.nunique() != 2:

    st.error(
        "Both target classes must be present in the test dataset."
    )

    st.stop()


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

st.header("3. Dataset Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Samples",
        len(df),
    )

with col2:

    st.metric(
        "Total Features",
        len(features),
    )

with col3:

    st.metric(
        "Number of Classes",
        y.nunique(),
    )

with col4:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum()),
    )


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
)


# ============================================================
# 9. TARGET CLASS DISTRIBUTION
# ============================================================

st.subheader("Target Class Distribution")

class_distribution = (
    y.map({
        0: class_names[0],
        1: class_names[1],
    })
    .value_counts()
    .reindex(class_names, fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 4))

sns.barplot(
    x=class_distribution.index,
    y=class_distribution.values,
    ax=ax,
)

ax.set_xlabel("Diagnosis")
ax.set_ylabel("Number of Samples")
ax.set_title("Benign vs Malignant Class Distribution")

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)
plt.close(fig)


st.divider()


# ============================================================
# 10. EVALUATE ALL FIVE MODELS
# ============================================================

st.header("4. Model Evaluation")

st.markdown(
    """
    All five trained models are evaluated using the uploaded
    test dataset. The results are used to generate individual
    and combined visualizations.
    """
)

model_results = {}
comparison_results = []

progress_bar = st.progress(0)

status_text = st.empty()

total_models = len(MODEL_FILES)

for index, (model_name, model_filename) in enumerate(
    MODEL_FILES.items()
):

    try:

        status_text.write(
            f"Evaluating {model_name}..."
        )

        model = load_model(model_filename)

        # Predictions
        y_pred = model.predict(x)

        # Probability estimates required for ROC and PR curves
        if not hasattr(model, "predict_proba"):

            raise ValueError(
                "Model does not support predict_proba()."
            )

        y_prob = model.predict_proba(x)[:, 1]

        # Metrics
        metrics = compute_metrics(
            model,
            x,
            y,
        )

        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(
            y,
            y_prob,
            pos_label=1,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        # Precision-Recall curve data
        precision, recall, pr_thresholds = (
            precision_recall_curve(
                y,
                y_prob,
                pos_label=1,
            )
        )

        # Confusion matrix
        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1],
        )

        # Save results for visualization
        model_results[model_name] = {

            "model": model,

            "predictions": y_pred,

            "probabilities": y_prob,

            "metrics": metrics,

            "fpr": fpr,

            "tpr": tpr,

            "roc_auc": roc_auc,

            "precision": precision,

            "recall": recall,

            "confusion_matrix": cm,

        }

        comparison_results.append({
            "Model": model_name,
            **metrics,
        })

    except Exception as error:

        st.error(
            f"Error evaluating {model_name}: {error}"
        )

    progress_bar.progress(
        (index + 1) / total_models
    )


status_text.success(
    "Model evaluation completed."
)

progress_bar.empty()


if len(model_results) != total_models:

    st.error(
        "One or more models could not be evaluated. "
        "Please verify your saved model artifacts."
    )

    st.stop()


# ============================================================
# 11. SELECTED MODEL PERFORMANCE METRICS
# ============================================================

st.header("5. Selected Model Performance Metrics")

selected_results = model_results[selected_model_name]

selected_metrics = selected_results["metrics"]

st.subheader(selected_model_name)

st.caption(
    MODEL_DESCRIPTIONS[selected_model_name]
)

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_names):

    with metric_columns[index % 3]:

        st.metric(
            metric_name,
            f"{selected_metrics[metric_name]:.4f}",
        )


st.divider()


# ============================================================
# 12. SELECTED MODEL CONFUSION MATRIX
# ============================================================

st.header("6. Confusion Matrix")

selected_cm = selected_results["confusion_matrix"]

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    selected_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    ax=ax,
)

ax.set_xlabel("Predicted Diagnosis")
ax.set_ylabel("Actual Diagnosis")

ax.set_title(
    f"{selected_model_name} - Confusion Matrix"
)

st.pyplot(fig)
plt.close(fig)


# ============================================================
# 13. SELECTED MODEL CLASSIFICATION REPORT
# ============================================================

st.header("7. Classification Report")

report = classification_report(
    y,
    selected_results["predictions"],
    labels=[0, 1],
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True,
)


st.divider()


# ============================================================
# 14. INDIVIDUAL ROC AND PRECISION-RECALL GRAPHS
#     FOR ALL FIVE MODELS
# ============================================================

st.header("8. Individual Model Visualizations")

st.markdown(
    """
    Select each model tab below to view its individual
    ROC curve and Precision-Recall curve.
    """
)

model_tabs = st.tabs(
    list(model_results.keys())
)


for tab, model_name in zip(
    model_tabs,
    model_results.keys(),
):

    with tab:

        results = model_results[model_name]

        st.subheader(model_name)

        st.caption(
            MODEL_DESCRIPTIONS[model_name]
        )

        # --------------------------------------------
        # Individual model ROC curve
        # --------------------------------------------

        st.markdown("### ROC Curve")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            results["fpr"],
            results["tpr"],
            linewidth=2,
            label=(
                f"{model_name} "
                f"(AUC = {results['roc_auc']:.4f})"
            ),
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_title(
            f"{model_name} - ROC Curve"
        )

        ax.legend(loc="lower right")
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------
        # Individual model Precision-Recall curve
        # --------------------------------------------

        st.markdown("### Precision-Recall Curve")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            results["recall"],
            results["precision"],
            linewidth=2,
            label=f"{model_name}",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        ax.set_title(
            f"{model_name} - Precision-Recall Curve"
        )

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------
        # Individual model metric summary
        # --------------------------------------------

        st.markdown("### Performance Summary")

        summary_columns = st.columns(3)

        for i, metric_name in enumerate(metric_names):

            with summary_columns[i % 3]:

                st.metric(
                    metric_name,
                    f"{results['metrics'][metric_name]:.4f}",
                )


st.divider()


# ============================================================
# 15. COMBINED ROC CURVE COMPARISON
# ============================================================

st.header("9. Combined ROC Curve Comparison")

st.markdown(
    """
    This graph compares the ROC performance of all five
    classification algorithms on the same test dataset.
    """
)

fig, ax = plt.subplots(figsize=(9, 6))

for model_name, results in model_results.items():

    ax.plot(
        results["fpr"],
        results["tpr"],
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {results['roc_auc']:.4f})"
        ),
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title(
    "ROC Curve Comparison - All Five Models"
)

ax.legend(loc="lower right")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


st.divider()


# ============================================================
# 16. COMBINED PRECISION-RECALL CURVE COMPARISON
# ============================================================

st.header("10. Combined Precision-Recall Curve Comparison")

st.markdown(
    """
    This graph compares the Precision-Recall performance
    of all five classification algorithms.
    """
)

fig, ax = plt.subplots(figsize=(9, 6))

for model_name, results in model_results.items():

    ax.plot(
        results["recall"],
        results["precision"],
        linewidth=2,
        label=model_name,
    )

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "Precision-Recall Curve Comparison - All Five Models"
)

ax.legend(loc="best")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


st.divider()


# ============================================================
# 17. COMPARISON OF ALL FIVE MODELS
# ============================================================

st.header("11. Comparison of All Five ML Models")

comparison_df = pd.DataFrame(
    comparison_results
)

st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True,
)


# Identify best accuracy
best_model_index = comparison_df[
    "Accuracy"
].astype(float).idxmax()

best_model_name = comparison_df.loc[
    best_model_index,
    "Model",
]

best_accuracy = comparison_df.loc[
    best_model_index,
    "Accuracy",
]

st.success(
    f"Highest accuracy on the uploaded dataset: "
    f"{best_model_name} ({best_accuracy:.4f})"
)


# ============================================================
# 18. APPLICATION FEATURE CHECKLIST
# ============================================================

st.header("12. Application Feature Verification")

feature_list = [
    "Five machine learning classification algorithms",
    "CSV dataset upload",
    "Dataset validation",
    "Dataset information and preview",
    "Target class distribution visualization",
    "Accuracy, AUC, Precision, Recall, F1 and MCC",
    "Confusion matrix heatmap",
    "Classification report",
    "Logistic Regression ROC and Precision-Recall curves",
    "Decision Tree ROC and Precision-Recall curves",
    "K-Nearest Neighbours ROC and Precision-Recall curves",
    "Naive Bayes ROC and Precision-Recall curves",
    "Support Vector Machine ROC and Precision-Recall curves",
    "Combined ROC curve comparison",
    "Combined Precision-Recall curve comparison",
    "Comparison of all five models",
    "Saved model training results",
]

for feature in feature_list:

    st.markdown(f"✅ {feature}")


# ============================================================
# 19. SCREENSHOT GUIDE
# ============================================================

st.header("13. Final Submission Screenshot Guide")

with st.expander(
    "View Screenshot Requirements",
    expanded=False,
):

    st.markdown(
        """
        Capture the following screenshots from the running application:

        1. `01_application_home.png`
           - Application title and model selection.

        2. `02_dataset_upload.png`
           - Uploaded test dataset and dataset preview.

        3. `03_dataset_information.png`
           - Dataset statistics and class distribution.

        4. `04_model_metrics.png`
           - Selected model evaluation metrics.

        5. `05_confusion_matrix_report.png`
           - Confusion matrix and classification report.

        6. `06_logistic_regression_visualization.png`
           - Logistic Regression ROC and Precision-Recall curves.

        7. `07_decision_tree_visualization.png`
           - Decision Tree ROC and Precision-Recall curves.

        8. `08_knn_visualization.png`
           - KNN ROC and Precision-Recall curves.

        9. `09_naive_bayes_visualization.png`
           - Naive Bayes ROC and Precision-Recall curves.

        10. `10_svm_visualization.png`
            - SVM ROC and Precision-Recall curves.

        11. `11_combined_roc_comparison.png`
            - ROC comparison of all five models.

        12. `12_combined_precision_recall.png`
            - Precision-Recall comparison of all five models.

        13. `13_model_comparison.png`
            - Performance comparison table.

        14. `14_saved_training_results.png`
            - Saved training metrics.
        """
    )


# ============================================================
# 20. FINAL APPLICATION STATUS
# ============================================================

st.success(
    "All five machine learning models were evaluated successfully. "
    "Individual and combined visualizations are available."
)

st.caption(
    """
    Academic demonstration only. This application is not intended
    for clinical diagnosis or medical decision-making.
    """
)
