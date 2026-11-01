"""
Streamlit front-end: evaluate pre-trained classifiers on an uploaded test CSV.

The app NEVER retrains anything - it only loads the .joblib files in model/
and scores the uploaded rows.
"""
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

# ----------------------------------------------------------------------------
# CONFIG - must match model/train_models.py
# ----------------------------------------------------------------------------
DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic)"
DATASET_BLURB = "569 patients, 30 cell-nucleus measurements"
TARGET = "diagnosis"
CLASS_NAMES = {0: "Benign", 1: "Malignant"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_NAMES = list(MODEL_FILES)
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# must be the first Streamlit call
st.set_page_config(page_title="Classifier Lab", page_icon="🧪", layout="wide")


@st.cache_resource
def load_model(model_name):
    """Load a saved, already-trained pipeline from model/ (no retraining)."""
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def compute_metrics(clf, X, y_true):
    """The six required metrics. Binary: positive class = 1. Multi-class:
    weighted averages and one-vs-rest AUC."""
    y_pred = clf.predict(X)
    binary = len(clf.classes_) == 2
    avg = "binary" if binary else "weighted"
    try:
        proba = clf.predict_proba(X)
        auc = (roc_auc_score(y_true, proba[:, 1]) if binary
               else roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                  labels=clf.classes_))
    except ValueError:                      # e.g. uploaded file has only one class
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# the saved pipelines remember the feature names they were trained on
FEATURES = list(load_model(MODEL_NAMES[0]).feature_names_in_)


def prepare_target(series):
    """Accept either numeric labels (0/1) or class names ('Benign'/'Malignant')."""
    if not pd.api.types.is_numeric_dtype(series):
        reverse = {v.lower(): k for k, v in CLASS_NAMES.items()}
        mapped = series.astype(str).str.strip().str.lower().map(reverse)
        if mapped.isna().any():
            raise ValueError(
                f"Column '{TARGET}' contains values that are not in {list(CLASS_NAMES.values())}."
            )
        return mapped.astype(int)
    return series.astype(int)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧪 Classifier Lab")
st.markdown(
    f"Compare five pre-trained classifiers on **{DATASET_NAME}** — "
    f"{DATASET_BLURB}, predicting **{CLASS_NAMES[0]}** vs **{CLASS_NAMES[1]}**."
)

# ----------------------------------------------------------------------------
# Sidebar: upload + model choice
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help=f"Must contain all {len(FEATURES)} feature columns and the '{TARGET}' column.")
    use_sample = st.checkbox("No file handy? Use the bundled test_data.csv", value=uploaded is None)
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", MODEL_NAMES)
    st.caption("Models are loaded from disk — nothing is retrained.")

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = f"uploaded file `{uploaded.name}`"
elif use_sample and SAMPLE_CSV.exists():
    data = pd.read_csv(SAMPLE_CSV)
    source = "bundled `test_data.csv`"
else:
    st.info("👈 Upload a test CSV (or tick the sample-data box) to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

try:
    y_true = prepare_target(data[TARGET])
except ValueError as err:
    st.error(str(err))
    st.stop()

X = data[FEATURES]
if X.isna().any().any():
    st.error("The feature columns contain missing values. Please clean the file and re-upload.")
    st.stop()

st.success(f"Loaded {len(data)} rows from {source}.")

clf = load_model(model_name)
scores = compute_metrics(clf, X, y_true)
y_pred = clf.predict(X)

tab_eval, tab_compare, tab_preds, tab_data = st.tabs(
    ["📊 Evaluation", "🏁 Compare all models", "🔍 Predictions", "🗂 Data preview"]
)

# ----------------------------------------------------------------------------
# Tab 1: metrics + confusion matrix + classification report
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader(f"{model_name}")
    cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(cols, METRIC_NAMES):
        value = scores[metric]
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

    left, right = st.columns(2)
    labels = sorted(CLASS_NAMES)
    display_names = [CLASS_NAMES[i] for i in labels]

    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=display_names, yticklabels=display_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=display_names,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

        if len(labels) == 2 and len(np.unique(y_true)) == 2:
            st.markdown("**ROC curve**")
            fig, ax = plt.subplots(figsize=(4.5, 3.4))
            RocCurveDisplay.from_estimator(clf, X, y_true, ax=ax, name=model_name)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 2: all five models on the same uploaded data
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("All five models on this data")
    rows = []
    for name in MODEL_NAMES:
        rows.append({"ML Model Name": name, **compute_metrics(load_model(name), X, y_true)})
    comparison = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(
        comparison,
        column_config={m: st.column_config.NumberColumn(format="%.4f") for m in METRIC_NAMES},
    )

    best = comparison["MCC"].idxmax()
    st.markdown(f"Highest **MCC** on this data: **{best}** ({comparison.loc[best, 'MCC']:.4f})")

    metric_pick = st.radio("Chart metric", METRIC_NAMES, horizontal=True, index=5)
    st.bar_chart(comparison[metric_pick])

# ----------------------------------------------------------------------------
# Tab 3: row-level predictions
# ----------------------------------------------------------------------------
with tab_preds:
    st.subheader("Row-level predictions")
    result = data.copy()
    result["actual"] = y_true.map(CLASS_NAMES)
    result["predicted"] = pd.Series(y_pred, index=data.index).map(CLASS_NAMES)
    if hasattr(clf, "predict_proba") and len(labels) == 2:
        result[f"P({CLASS_NAMES[1]})"] = clf.predict_proba(X)[:, 1].round(4)
    result["correct"] = result["actual"] == result["predicted"]

    only_wrong = st.checkbox("Show only misclassified rows")
    shown = result[~result["correct"]] if only_wrong else result
    front = ["actual", "predicted", "correct"] + [c for c in shown.columns if c.startswith("P(")]
    st.dataframe(shown[front + FEATURES])
    st.caption(f"{int((~result['correct']).sum())} misclassified out of {len(result)} rows.")
    st.download_button("⬇️ Download predictions CSV", result.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 4: data preview
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Uploaded data")
    st.write(f"{data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(50))
    st.markdown("**Class balance**")
    balance = y_true.map(CLASS_NAMES).value_counts()
    st.bar_chart(balance)
