from pathlib import Path
from sklearn import naive_bayes
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

MODEL_FILES = {
           "Logistic Regression": "logistic_regression.joblib",
           "Decision Tree": "decision_tree.joblib",
           "K-Nearest Neighbors": "knn.joblib",
           "Naive bayes (Gaussian)": naive_bayes.joblib,
           "Support Vector Machine": "SVM.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def compute_metrics(model, x, y):
    """Return (predictions, metrics dict). Class 1 (malignant) is positive."""
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return pred, {
       "Accuracy" : accuracy_score(y, pred),
       "AUC": roc_auc_score(y, prob),
       "Precision": recall_score(y, pred, zero_division=0),
       "F1": f1_score(y, pred, zero_division=0),
      "MCC": matthews_corrcoef(y, pred)
    }
    
