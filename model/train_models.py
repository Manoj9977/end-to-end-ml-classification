import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, logisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from common import METRICS_NAMES, MODEL_DIR, MODEL_FILES, ROOT, TARGET, compute_metrics


## ---------------CONFIG------------------
DATA_FILE = ROOT / "D:OCT_31st/breast+cancer+wisconsin+diagnostic.csv"  # your downloaded dataset
TARGET_COL = "diagnosis"               # M = malignant, B = benign
# -------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
print("Columns found:", df.columns.tolist())
df = df.drop(columns=["id", "Id", "ID", "Unnamed: 32"], errors="ignore").dropna()

df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper().map({"B": 0, "M": 1})
if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains values other than 'M' and 'B'")

y = df[TARGET_COL].astype(int)
x = df.drop(columns=[TARGET_COL]).astype(float)
class_names = ["Benign", "Malignant"]

print(f"Rows: {x.shape[0]} | Features: {x.shape[1]}")
assert x.shape[0] >= 500, "Need at least 500 instances"
assert x.shape[0] >= 12, "Need at least 12 features"

x_train, x_test, y_train, y_test = train_test_split(
                       x, y, test_size=0.2, stratify=y, random_state=42)

models = {
       "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
       "Decision Tree": DecisionTreeClassifier(random_state=42),
       "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
       "Naive Bayes (Gaussian)": GaussianNB(),
       "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
   }
results = {}
for name, model in models.items():
        model.fit(x_train, y_train)
        _, results[name] = compute_metrics(model, x_test, y_test)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])

(MODEL_DIR / "feature_columns.json").write_text(json.dumps(list(x.columns)))
(MODEL_DIR / "class_names.json").write_text(json.dumps(class_names))
(MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))


test = x.loc[y_test.index].copy()
test[TARGET] = y_test
test.to_csv(ROOT / "test_data.csv", index=False)

lines = ["|Model | " + " | ".join(METRICS_NAMES) + " |",
         "|---|" + "---|" * len(METRICS_NAMES)]

for name, r in results. items():
           lines.append(f"| {name} | " + " | ".join(f"{r[m]:.4f}" for m in METRICS_NAMES) + " |")
(ROOT / "results_table.md").write_text("\n".join)(lines)

print("\n".join(lines))
print("Saved: model/, test_data.csv, results_table.md")

