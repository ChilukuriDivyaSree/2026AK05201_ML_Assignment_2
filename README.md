# Breast Cancer Diagnosis — Classification Models & Streamlit App

M.Tech (AIML) · Machine Learning · Assignment 2

## a. Problem statement

Predict whether a breast mass is **Malignant** or **Benign** from numeric measurements of cell nuclei taken from a digitised fine-needle-aspirate image. This is a binary classification problem. Five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine — are trained on the same data, compared with six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and exposed through an interactive Streamlit web app that evaluates the **saved** models on an uploaded test CSV without retraining.

In this task a missed malignant case (false negative) is more costly than a false alarm, so **Recall** for the Malignant class is read alongside MCC, which summarises all four confusion-matrix cells in one number.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository (also distributed with scikit-learn as `load_breast_cancer`) |
| Instances | 569 (assignment minimum: 500) |
| Features | 30 numeric (assignment minimum: 12) |
| Target | `diagnosis` — `1` = Malignant (212 rows, 37.3 %), `0` = Benign (357 rows, 62.7 %) |
| Missing values | None |

The 30 features are three statistics — **mean**, **standard error** and **worst** (mean of the three largest values) — of ten measurements computed for each nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension (e.g. `mean_radius`, `radius_error`, `worst_radius`).

**Split and preprocessing**

- Stratified 80 / 20 train–test split (`random_state = 2026`): 455 training rows, **114 test rows** (72 Benign, 42 Malignant). The 114 test rows are saved as `test_data.csv`.
- `StandardScaler` is fitted on the training data only and is part of the saved scikit-learn `Pipeline` for Logistic Regression, KNN and SVM (distance / margin based models). Decision Tree and Naive Bayes use raw features.
- Metrics treat **Malignant (1)** as the positive class for Precision, Recall and F1.

## c. GitHub repository link

**<https://github.com/ChilukuriDivyaSree/2026AK05201_ML_Assignment_2>**

Live Streamlit app: **<https://2026ak05201-ml-assignment-2.streamlit.app/>**

```
project-folder/
│-- app.py                                 Streamlit app (upload CSV, pick model, view metrics / confusion matrix)
│-- requirements.txt
│-- README.md
│-- test_data.csv                          held-out test set (features + target) — upload this in the app
│-- model/
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- support_vector_machine.joblib      saved trained models (scaler included in the pipelines)
    │-- train_models.py                    source code: loads data, trains the 5 models, evaluates, saves everything
```

**Run locally** (from the project root)

```bash
pip install -r requirements.txt
python model/train_models.py     # trains, evaluates, prints the metrics table, saves models + test_data.csv
streamlit run app.py
```

## d. Models used

Hyper-parameters (all other settings are scikit-learn defaults):

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler → `LogisticRegression(max_iter=1000)` (L2, C = 1) |
| Decision Tree | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)` |
| KNN | StandardScaler → `KNeighborsClassifier(n_neighbors=7)` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | StandardScaler → `SVC(kernel="rbf", C=1.0, probability=True)` |

### Comparison table (114-row held-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9987 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9737 | 0.9785 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| KNN | 0.9737 | 0.9974 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |
| Naive Bayes | 0.9474 | 0.9927 | 0.9286 | 0.9286 | 0.9286 | 0.8869 |
| Support Vector Machine | **0.9825** | **1.0000** | 0.9545 | **1.0000** | **0.9767** | **0.9633** |

Confusion-matrix counts on the same test set (positive = Malignant):

| Model | TN | FP | FN | TP | Total errors |
|---|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 2 | 40 | 3 |
| Decision Tree | 71 | 1 | 2 | 40 | 3 |
| KNN | 70 | 2 | 1 | 41 | 3 |
| Naive Bayes | 69 | 3 | 3 | 39 | 6 |
| Support Vector Machine | 70 | 2 | 0 | 42 | 2 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9737 and AUC 0.9987 with only 3 errors out of 114 (1 FP, 2 FN). The very high AUC shows the two classes are almost linearly separable once the features are standardised, so a simple linear model ranks malignant cases very well. It is also fast and interpretable. |
| Decision Tree | Same confusion counts as Logistic Regression (71/1/2/40), so identical accuracy, precision, recall, F1 and MCC, but made on different rows. Its AUC is the lowest of all models (0.9785) because a depth-5 tree outputs coarse leaf-proportion probabilities, so it ranks cases less finely than the smooth models. |
| KNN | Same accuracy (0.9737) and the best MCC of the three tied models (0.9439). Its errors lean the other way (2 FP, 1 FN), giving higher recall (0.9762) but lower precision (0.9535). It relies on feature scaling and on k (k = 7 here), and its AUC (0.9974) is high. |
| Naive Bayes | Weakest on this test set: accuracy 0.9474, MCC 0.8869, 6 errors. The independence assumption is clearly violated here: mean radius, perimeter and area are almost perfectly correlated (r ≈ 0.99), so the same evidence is counted several times. Its AUC is still high (0.9927), so it ranks cases well but its default decision threshold is poorly placed. |
| Support Vector Machine | Best on this test set: accuracy 0.9825, AUC 1.0000, recall 1.0000 (no malignant case missed), F1 0.9767 and MCC 0.9633, with only 2 errors (both false positives). The RBF kernel fits the mild non-linearity in the data after scaling. |
| **Overall, winner for this dataset?** | **Support Vector Machine** — highest on 5 of 6 metrics (all except precision) and the only model with zero missed malignant cases, which is the more costly error in screening. Caveat: the test set has only 114 rows, and SVM beats Logistic Regression, Decision Tree and KNN by a single row (2 errors vs 3), so SVM, Logistic Regression and KNN should be regarded as closely matched. Naive Bayes is clearly the weakest. |

**Live Streamlit app:** **<https://2026ak05201-ml-assignment-2.streamlit.app/>**
## Streamlit app features

1. **Dataset upload** — upload a CSV containing all 30 feature columns and the `diagnosis` column (class names `Benign` / `Malignant` are also accepted). A bundled `test_data.csv` option and a download button are provided for quick demos.
2. **Model selection dropdown** — Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
4. **Confusion matrix and classification report** — plus an ROC curve.
5. **Compare all models** tab — all five models scored on the uploaded data with a metric bar chart.
6. **Predictions** tab — row-level predictions with probabilities, a misclassified-only filter and CSV download.

The app loads the `.joblib` files from `model/` and only calls `predict` / `predict_proba`; no model is retrained on uploaded data. The metrics shown for `test_data.csv` match the table above exactly.
