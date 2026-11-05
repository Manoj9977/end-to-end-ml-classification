
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. PROJECT ROOT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python import path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# 2. IMPORT COMMON MODULE
# ==================================================

from model.common import (
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    ROOT,
    TARGET,
    compute_metrics,
)


# ==================================================
# 3. CONFIGURATION
# ==================================================

DATA_FILE = (
    ROOT / "model" /
    "breast+cancer+wisconsin+diagnostic.csv"
)

TARGET_COL = "Diagnosis"

RANDOM_STATE = 42
TEST_SIZE = 0.20



# ==================================================
# 2. CREATE MODEL DIRECTORY
# ==================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. LOAD DATASET
# ==================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")

print("Original columns:", df.columns.tolist())

# Handle target column capitalization differences
target_matches = [
    column for column in df.columns
    if column.strip().lower() == TARGET_COL.lower()
]

if not target_matches:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found."
    )

actual_target_col = target_matches[0]


# ==================================================
# 4. DATA PREPROCESSING
# ==================================================

# Remove unnecessary ID columns
df = df.drop(
    columns=[
        column for column in df.columns
        if column.strip().lower() in [
            "id",
            "unnamed: 32"
        ]
    ],
    errors="ignore"
)

# Remove rows containing missing values
df = df.dropna().copy()

# Encode target variable
# B = 0 (Benign)
# M = 1 (Malignant)

df[actual_target_col] = (
    df[actual_target_col]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "B": 0,
        "M": 1
    })
)

if df[actual_target_col].isna().any():
    raise ValueError(
        "Target column contains invalid values. "
        "Expected only B and M."
    )

# Separate features and target
y = df[actual_target_col].astype(int)

x = df.drop(
    columns=[actual_target_col]
)

# Convert feature columns to numeric values
x = x.apply(pd.to_numeric, errors="raise")

if not x.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all():
    raise ValueError("All features must be numeric.")

if x.isna().any().any():
    raise ValueError("Feature columns contain missing values.")

class_names = [
    "Benign",
    "Malignant"
]

print("\nPreprocessing completed!")

print("Number of rows:", x.shape[0])
print("Number of features:", x.shape[1])
print("\nTarget distribution:")
print(y.value_counts().sort_index())

assert x.shape[0] >= 500, (
    "Dataset must contain at least 500 rows."
)

assert x.shape[1] >= 12, (
    "Dataset must contain at least 12 features."
)

assert set(y.unique()) == {0, 1}, (
    "Both target classes must be present."
)


# ==================================================
# 5. TRAIN-TEST SPLIT
# ==================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain-test split completed!")

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# ==================================================
# 6. DEFINE CLASSIFICATION MODELS
# ==================================================

models = {

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "K-Nearest Neighbours": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "Naive Bayes (Gaussian)": GaussianNB(),

    "Support Vector Machine": CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            SVC(
                random_state=RANDOM_STATE
            )
        ),
        ensemble=False,
        cv=5
    ),

}



# ==================================================
# 7. TRAIN AND EVALUATE MODELS
# ==================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training model:", name)
    print("=" * 60)

    # Train model
    model.fit(
        x_train,
        y_train,
    )

    # Calculate evaluation metrics
    # Returns a dictionary with six metrics
    metrics = compute_metrics(
        model,
        x_test,
        y_test,
    )

    # Store model evaluation results
    results[name] = metrics

    # Save trained model
    model_filename = MODEL_FILES[name]

    model_path = MODEL_DIR / model_filename

    joblib.dump(
        model,
        model_path,
    )

    print("\nModel saved:", model_path)

    print("\nEvaluation metrics:")

    for metric in METRIC_NAMES:

        print(
            f"{metric}: {metrics[metric]:.4f}"
        )



# ==================================================
# 8. SAVE FEATURE COLUMNS AND CLASS NAMES
# ==================================================

feature_columns = list(x.columns)

(MODEL_DIR / "feature_columns.json").write_text(
    json.dumps(feature_columns, indent=2),
    encoding="utf-8"
)

(MODEL_DIR / "class_names.json").write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8"
)


# ==================================================
# 9. SAVE EVALUATION METRICS
# ==================================================

(MODEL_DIR / "metrics.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)


# ==================================================
# 10. SAVE TEST DATA
# ==================================================

test = x_test.copy()

test[TARGET] = y_test

test.to_csv(
    ROOT / "test_data.csv",
    index=False
)


# ==================================================
# 11. GENERATE MODEL COMPARISON TABLE
# ==================================================

lines = [
    "| Model | " + " | ".join(METRIC_NAMES) + " |",
    "|---|" + "---|" * len(METRIC_NAMES)
]

for name, metrics in results.items():

    metric_values = [
        f"{metrics[metric]:.4f}"
        for metric in METRIC_NAMES
    ]

    lines.append(
        f"| {name} | "
        + " | ".join(metric_values)
        + " |"
    )

results_table = "\n".join(lines)

(ROOT / "results_table.md").write_text(
    results_table,
    encoding="utf-8"
)


# ==================================================
# 12. FINAL OUTPUT
# ==================================================

print("\n" + "=" * 60)

print("MODEL EVALUATION RESULTS")

print("=" * 60)

print(results_table)

print("\nTraining completed successfully!")

print("\nGenerated files:")

print("1. Trained models:", MODEL_DIR)

print("2. Feature columns:", MODEL_DIR / "feature_columns.json")

print("3. Class names:", MODEL_DIR / "class_names.json")

print("4. Evaluation metrics:", MODEL_DIR / "metrics.json")

print("5. Test dataset:", ROOT / "test_data.csv")

print("6. Model comparison:", ROOT / "results_table.md")

print("\n" + "=" * 60)