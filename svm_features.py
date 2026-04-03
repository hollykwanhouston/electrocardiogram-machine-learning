# svm_features.py
# SVM (RBF) on FeaturesDatasets
# Install dependencies: pip install pandas scikit-learn numpy

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# ── Load dataset ──────────────────────────────────────────────────────────────
train_feat = pd.read_csv("train_feat.csv")

# ── Select feature columns (columns 2–190, i.e. index 1–189 inclusive) ───────
# R's train_feat[, 2:190] is 1-indexed and inclusive on both ends
train_feat = train_feat.iloc[:, 1:190]
train_feat["Type"] = train_feat["Type"].astype("category")

# ── Train / test split (65% train, 35% test) ──────────────────────────────────
# Two separate seeds mirror the R script's set.seed(50) / set.seed(20) pattern
rng      = np.random.default_rng(50)
s        = int(np.floor(0.65 * len(train_feat)))
rng2     = np.random.default_rng(20)
idx      = rng2.choice(len(train_feat), size=s, replace=False)
mask     = np.zeros(len(train_feat), dtype=bool)
mask[idx] = True

train = train_feat[mask].reset_index(drop=True)
test  = train_feat[~mask].reset_index(drop=True)

X_train = train.drop(columns=["Type"])
y_train = train["Type"]
X_test  = test.drop(columns=["Type"])
y_test  = test["Type"]

# ── Initial SVM fit with RBF kernel (no scaling) ──────────────────────────────
svmfit = SVC(kernel="rbf", gamma="scale")
svmfit.fit(X_train, y_train)
print("Initial SVM model:")
print(f"  Support vectors per class : {svmfit.n_support_}")
print(f"  Total support vectors     : {sum(svmfit.n_support_)}\n")

# ── Hyperparameter tuning — gamma in 2^(-1:1), cost in 2^(2:4) ───────────────
# Mirrors R: gamma = 2^(-1:1) → [0.5, 1, 2]  |  cost = 2^(2:4) → [4, 8, 16]
param_grid = {
    "gamma": [2**i for i in range(-1, 2)],   # [0.5, 1, 2]
    "C":     [2**i for i in range(2, 5)],    # [4, 8, 16]
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

# ── Final model with paper's chosen parameters (cost=8, gamma=0.5) ────────────
svmfit = SVC(kernel="rbf", C=8, gamma=0.5)
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
