"""
Train the five classifiers, evaluate them on a held-out test split, and save
the fitted models + test_data.csv.

Run from the project root:   python model/train_models.py

Outputs
-------
model/<name>.joblib   one saved, fitted pipeline per classifier
test_data.csv         held-out rows (features + target) -> upload this in the app
(printed)             the comparison table that goes into the README
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# scikit-learn >= 1.9 warns that SVC(probability=True) will change in a future
# release. The version is pinned in requirements.txt, so the warning is noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

MODEL_DIR = Path(__file__).parent          # this script lives inside model/
TEST_CSV = MODEL_DIR.parent / "test_data.csv"

# ----------------------------------------------------------------------------
# CONFIG - edit these if you switch to your own dataset
# (if you change TARGET_COLUMN / CLASS_NAMES, change them in app.py as well)
# ----------------------------------------------------------------------------
TARGET_COLUMN = "diagnosis"
TEST_SIZE = 0.20
SEED = 2026


def load_data():
    """Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn (UCI origin).

    To use your own dataset, replace the body with e.g.
        return pd.read_csv("path/to/your.csv")
    and make sure the target column is called TARGET_COLUMN with 0/1 (or 0..k) labels.
    """
    df = load_breast_cancer(as_frame=True).frame
    df[TARGET_COLUMN] = 1 - df["target"]            # 1 = Malignant, 0 = Benign
    df = df.drop(columns=["target"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def build_models():
    """Five classifiers. Scaling lives inside a Pipeline for the distance /
    margin / gradient based models (LR, KNN, SVM)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ]),
    }


def evaluate(model, X, y):
    """Six required metrics. Binary: positive class = 1. Multi-class: weighted
    averages and one-vs-rest AUC."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    binary = len(model.classes_) == 2
    avg = "binary" if binary else "weighted"
    auc = (roc_auc_score(y, proba[:, 1]) if binary
           else roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"   # also used by app.py


def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
    print(f"Dataset: {len(df)} rows, {X.shape[1]} features, classes {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # test_data.csv = features + target, exactly what the app expects as upload
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / file_name(name))
        rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})

    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nSaved 5 models to {MODEL_DIR} and test data to {TEST_CSV}")


if __name__ == "__main__":
    main()
