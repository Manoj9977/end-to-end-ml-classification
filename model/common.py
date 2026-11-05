
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "model_artifacts"

TARGET = "target"


# ============================================================
# 2. MODEL FILE MAPPING
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# ============================================================
# 3. EVALUATION METRIC NAMES
# ============================================================

METRIC_NAMES = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]


# ============================================================
# 4. COMPUTE MODEL EVALUATION METRICS
# ============================================================

def compute_metrics(model, x, y):
    """
    Calculate classification evaluation metrics.

    Returns:
        dict: Accuracy, AUC, Precision, Recall, F1 and MCC.

    Note:
        This function consistently returns a dictionary.
        It does not return (predictions, metrics).
    """

    # Generate class predictions
    predictions = model.predict(x)

    # Generate positive-class probabilities
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The selected model does not support predict_proba()."
        )

    probabilities = model.predict_proba(x)

    # Locate the probability column for positive class 1
    model_classes = list(model.classes_)

    if 1 not in model_classes:
        raise ValueError(
            "Positive class 1 is missing from the trained model."
        )

    positive_class_index = model_classes.index(1)

    positive_probabilities = probabilities[:, positive_class_index]

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y, predictions),

        "AUC": roc_auc_score(
            y,
            positive_probabilities,
        ),

        "Precision": precision_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions,
        ),
    }

    return metrics
