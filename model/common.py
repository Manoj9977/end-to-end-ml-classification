
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
# PROJECT CONFIGURATION
# ==================================================

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Directory to store trained models and evaluation metrics
MODEL_DIR = ROOT / "model_artifacts"

# Target column used in saved test data
TARGET = "target"


# ==================================================
# MODEL FILE NAMES
# ==================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# ==================================================
# EVALUATION METRIC NAMES
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
# COMPUTE CLASSIFICATION METRICS
# ==================================================

def compute_metrics(model, x, y):
    """
    Calculate classification evaluation metrics.

    Class 0 = Benign
    Class 1 = Malignant

    Returns:
        predictions: Predicted class labels.
        metrics: Dictionary containing six evaluation metrics.
    """

    # Generate class predictions
    predictions = model.predict(x)

    # Generate probability predictions
    probabilities = model.predict_proba(x)

    # Find the probability column corresponding to malignant (class 1)
    positive_class_index = list(model.classes_).index(1)

    malignant_probabilities = probabilities[:, positive_class_index]

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y, predictions),

        "AUC": roc_auc_score(
            y,
            malignant_probabilities
        ),

        "Precision": precision_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "F1": f1_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        ),
    }

    return predictions, metrics
