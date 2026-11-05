
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "model_artifacts"

TARGET = "target"


# ==================================================
# 2. MODEL FILE MAPPING
# ==================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# ==================================================
# 3. EVALUATION METRIC NAMES
# ==================================================

METRIC_NAMES = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]


# ==================================================
# 4. COMPUTE MODEL EVALUATION METRICS
# ==================================================

def compute_metrics(model, x, y):

    # Generate predictions
    predictions = model.predict(x)

    # Generate positive-class probabilities
    if not hasattr(model, "predict_proba"):

        raise ValueError(
            "The model must support predict_proba() "
            "to calculate AUC."
        )

    probabilities = model.predict_proba(x)[:, 1]

    # Calculate evaluation metrics
    metrics = {

        "Accuracy": accuracy_score(
            y,
            predictions,
        ),

        "AUC": roc_auc_score(
            y,
            probabilities,
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions,
        ),

    }

    # Return one dictionary containing all six metrics
    return metrics
