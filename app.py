import json
from ntpath import join
from shlex import join
from pickle import TRUE
from tkinter import TRUE

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from common import MODEL_DIR, MODEL_FILES, TARGET, compute_metrics

@st.cache_resource
def load_model(filename):
     return joblib.load(MODEL_DIR/filename)

st.setpage_config(page_title="Breast Cancer Classification", layout="wide")
st.title("Breast Cancer Classification - ModelEvaluation")
st.caption("Target:0 = Benign, 1 = Malignant. Models are pre-trained and loaded"
            "from disk; no retraining happens on uploaded data.")

features = json.loads((MODEL_DIR / "feature_columns.json").read_text())
class_names = json.loads((MODEL_DIR / "class_names.json").read_text())

st.sidebar.header("Options")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES))
file = st.sidebar.file_uploader("Upload test CSV (features + 'target')", type="CSV")

with st.expander("Test-set results of all models recorded during training"):
     saved = join.loads((MODEL_DIR / "metrics.json").read_text())
     st.dataframe(pd.DataFrame(saved).T.round(4), Use_container_width=TRUE)
if file is None:
   st.info("Upload test_data.csv in the sidebar to evaluate the selected model.")
   st.stop()

try:
    df = pd.read_csv(file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = df[features + [TARGET]].apply(pd.to_numeric, errors='coerce')
if df.isna().any().any():
    st.error("The CSV contains non-numberic or missing values.")
    st.stop

x,y = df[features], df[TARGET].astype(int)
if set(y.unique()) - {0,1} or y.nunique()<2:
    st.error("Target must contain both classes: 0 (Benign) and 1 (Malignant).")
    st.stop()

pred,metrics = compute_metrics(load_model(MODEL_FILES[model_name]), x, y)


st.subheader(f"Results: {model_name}")
for col, (k, v) in zip(st.columns(6), metrics.items()):
    col.metric(k, f"{v:4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(confusion_matrix(y, pred, labels=[0,1]), annot=True, fmt="d",
                   cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pred, labels=[0,1], target_names=class_names,
                            output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

st.subheader("Comparision of all 5 models on the uploaded test data")
rows = {n: compute_metrics(load_model(f), x,y)[1] for n, f in MODEL_FILES.items()}
st.dataframe(pd.DataFrame(rows).T.round(4), use_container_width=True)
