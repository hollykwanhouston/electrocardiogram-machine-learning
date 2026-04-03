# svm_signals.py
# SVM (RBF) on SignalsDatasets
# Install dependencies: pip install pandas scikit-learn numpy

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# ── Load dataset ──────────────────────────────────────────────────────────────
train_signal = pd.read_csv("train_signal.csv")

# ── Select feature columns (columns 2–6002, i.e. index 1–6001 inclusive) ─────
# R's train_signal[, 2:6002] is 1-indexed and inclusive on both ends
train_signal = train_signal.iloc[:, 1:6002]
train_signal["Type"] = train_signal["Type"].astype("category")

# ── Train / test split (65% train, 35% test) ──────────────────────────────────
# Single set.seed(50) mirrors the R script
rng = np.random.default_rng(50)
s   = int(np.floor(0.65 * len(train_signal)))
idx = rng.choice(len(train_signal), size=s, replace=False)

mask      = np.zeros(len(train_signal), dtype=bool)
mask[idx] = True

train = train_signal[mask].reset_index(drop=True)
test  = train_signal[~mask].reset_index(drop=True)

X_train = train.drop(columns=["Type"])
y_train = train["Type"]
X_test  = test.drop(columns=["Type"])
y_test  = test["Type"]

# ── Initial SVM fit with RBF kernel (no scaling) ──────────────────────────────
svmfit = SVC(kernel="rbf")
svmfit.fit(X_train, y_train)
print("Initial SVM model:")
print(f"  Support vectors per class : {svmfit.n_support_}")
print(f"  Total support vectors     : {sum(svmfit.n_support_)}\n")

# ── Hyperparameter tuning ─────────────────────────────────────────────────────
# R: gamma = seq(1/2^nrow(train_feat), 1, 0.01)
# Note: 2^nrow(train_feat) underflows to 0 for large datasets, so gamma_start
# is clamped to a practical minimum of 1e-6 to avoid division-by-zero issues.
n_rows      = len(train_signal)
gamma_start = max(1 / (2 ** n_rows), 1e-6)
gamma_range = np.arange(gamma_start, 1.0, 0.01)

# R: cost = 2^seq(-6, 4, 2) → 2^[-6, -4, -2, 0, 2, 4] → [0.016, 0.0625, 0.25, 1, 4, 16]
cost_range = 2.0 ** np.arange(-6, 5, 2)

param_grid = {
    "gamma": gamma_range,
    "C":     cost_range,
}

grid_search = GridSearchCV(
    SVC(kernel="rbf"),
    param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print("\nHyperparameter tuning results:")
print(f"  Best parameters : {grid_search.best_params_}")
print(f"  Best CV score   : {grid_search.best_score_:.4f}\n")

# ── Final model with paper's chosen parameters (cost=4, gamma=0.5) ────────────
svmfit = SVC(kernel="rbf", C=4, gamma=0.5)
svmfit.fit(X_train, y_train)
print("Tuned SVM model:")
print(f"  Support vectors per class : {svmfit.n_support_}")
print(f"  Total support vectors     : {sum(svmfit.n_support_)}\n")

# ── Predictions and confusion matrix ──────────────────────────────────────────
svm_pred = svmfit.predict(X_test)

print("Confusion matrix:")
cm = confusion_matrix(y_test, svm_pred, labels=svmfit.classes_)
cm_df = pd.DataFrame(
    cm,
    index   = [f"true_{c}" for c in svmfit.classes_],
    columns = [f"pred_{c}" for c in svmfit.classes_],
)
print(cm_df)

print("\nClassification report:")
print(classification_report(y_test, svm_pred))
