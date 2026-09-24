# -*- coding: utf-8 -*-
# ==============================================================================
# SECTION 1 -- IMPORTS
# ==============================================================================
import json
import os
import pathlib
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Use non-interactive backend so figures save without a display (works on servers too)
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 2 -- PATHS AND CONSTANTS
# ==============================================================================

# Resolve all paths relative to this script file so it works from any directory
BASE   = pathlib.Path(__file__).parent
DATA   = BASE / "data"
MODELS = BASE / "models"
FIG    = BASE / "static" / "figures"

# Create output directories if they do not exist
for directory in (MODELS, FIG):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette used across all charts (green / teal brand theme)
# ---------------------------------------------------------------------------
INK   = "#0E2A26"   # near-black
PINE  = "#0F6B5A"   # dark teal (primary)
GREEN = "#1E8E5A"   # approval green
RED   = "#B23A32"   # rejection red
AMBER = "#C98A12"   # warning amber
GREY  = "#8A9A96"   # neutral grey

# ---------------------------------------------------------------------------
# Feature column groups used by the preprocessing pipeline
# ---------------------------------------------------------------------------
# These are the 9 columns the model receives as input (Gender and Married
# are intentionally excluded to prevent direct demographic bias).
RAW_FEATURES = [
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

# Numeric features after feature engineering (includes derived columns)
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Dependents_num",       # numeric version of the Dependents category
    "TotalIncome",          # ApplicantIncome + CoapplicantIncome
    "LogTotalIncome",       # log1p(TotalIncome) -- reduces skewness
    "LogLoanAmount",        # log1p(LoanAmount)
    "EMI",                  # monthly repayment = LoanAmount * 1000 / Term
    "EMI_to_Income",        # EMI as a fraction of monthly household income
    "Loan_to_Income",       # LoanAmount * 1000 / (TotalIncome * 12)
]

CATEGORICAL_FEATURES = ["Education", "Self_Employed", "Property_Area"]
BINARY_FEATURES      = ["Credit_History"]


# ==============================================================================
# SECTION 3 -- HELPER: FIGURE SAVER
# ==============================================================================

def save_figure(filename: str) -> None:
    """Apply tight layout and save the current matplotlib figure to static/figures/."""
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150)
    plt.close()
    print(f"  [chart]  Saved  static/figures/{filename}")


# ==============================================================================
# SECTION 4 -- FEATURE ENGINEERING
# ==============================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive affordability features from the raw loan-application columns.

    New columns added
    -----------------
    Dependents_num  : converts '3+' -> 3.0, other categories to float
    TotalIncome     : ApplicantIncome + CoapplicantIncome
    LogTotalIncome  : log1p(TotalIncome)  -- reduces right skew
    LogLoanAmount   : log1p(LoanAmount)
    EMI             : monthly repayment (principal divided by term)
    EMI_to_Income   : EMI as a share of monthly household income
    Loan_to_Income  : loan amount as a multiple of annual household income

    Rationale
    ---------
    Income alone is weakly predictive because an INR 5L loan means something
    very different to a household earning INR 10K/month vs INR 1L/month.
    These ratio features capture the relative affordability that lenders
    actually care about, and they improve model performance.

    Division-by-zero / infinity values are replaced with NaN so the
    downstream imputer handles them gracefully.
    """
    d = df.copy()

    # Convert '3+' -> 3, keep NaN as NaN
    d["Dependents_num"] = (
        d["Dependents"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    # Total household income
    d["TotalIncome"]    = d["ApplicantIncome"] + d["CoapplicantIncome"]

    # Log transforms to reduce right-skew in income and loan amount
    d["LogTotalIncome"] = np.log1p(d["TotalIncome"])
    d["LogLoanAmount"]  = np.log1p(d["LoanAmount"])

    # Repayment affordability ratios
    # LoanAmount is stored in INR thousands, so multiply by 1000 for actual INR
    d["EMI"]            = (d["LoanAmount"] * 1000) / d["Loan_Amount_Term"]
    d["EMI_to_Income"]  = d["EMI"] / d["TotalIncome"]
    d["Loan_to_Income"] = (d["LoanAmount"] * 1000) / (d["TotalIncome"] * 12)

    # Replace any infinities caused by zero income / zero term with NaN
    return d.replace([np.inf, -np.inf], np.nan)


# ==============================================================================
# SECTION 5 -- PREPROCESSING PIPELINE
# ==============================================================================

def build_preprocessing_pipeline() -> Pipeline:
    """
    Build and return a scikit-learn Pipeline that:
      1. Runs add_engineered_features via FunctionTransformer
      2. Imputes + scales numeric columns  (median imputation, StandardScaler)
      3. Imputes binary column             (most-frequent imputation)
      4. Imputes + encodes categoricals    (most-frequent, OneHotEncoder)

    Keeping all preprocessing inside a Pipeline prevents data leakage:
    the imputation statistics (medians, most-frequent values) are computed
    only on the training split, not on the test set.
    """
    # --- Numeric branch ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # --- Binary branch (Credit_History) ---
    binary_transformer = SimpleImputer(strategy="most_frequent")

    # --- Categorical branch ---
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine all branches
    column_transformer = ColumnTransformer(transformers=[
        ("numeric",      numeric_transformer,      NUMERIC_FEATURES),
        ("binary",       binary_transformer,        BINARY_FEATURES),
        ("categorical",  categorical_transformer,  CATEGORICAL_FEATURES),
    ])

    # Wrap feature engineering + column transformer in one Pipeline
    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("preprocessing",       column_transformer),
    ])

    return full_pipeline


# ==============================================================================
# SECTION 6 -- LOAD DATA
# ==============================================================================

print("=" * 65)
print("  LOAN APPROVAL PREDICTION -- MODEL TRAINING")
print("=" * 65)

csv_path = DATA / "loan_data.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {csv_path}. "
        "Run  python generate_dataset.py  to create it first."
    )

print(f"\n[data]  Loading {csv_path}")
df = pd.read_csv(csv_path)
df["Approved"] = (df["Loan_Status"] == "Y").astype(int)

print(f"[data]  {df.shape[0]} rows  x  {df.shape[1]} columns")
print(f"[data]  Approval rate : {df['Approved'].mean():.1%}")
print(f"[data]  Missing cells : {df.isna().sum().sum()}")


# ==============================================================================
# SECTION 7 -- EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

print("\n--- EDA charts ---")

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.edgecolor":    "#C9D3D0",
        "grid.color":        "#E4EAE8",
        "axes.titleweight":  "bold",
        "axes.titlesize":    12,
        "axes.labelcolor":   INK,
        "text.color":        INK,
    },
)
pal = {"Y": GREEN, "N": RED}

# ------------------------------------------------------------------
# Chart 1 -- Class distribution (approved vs rejected)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.6))
counts = df.Loan_Status.value_counts()
ax.bar(["Approved", "Rejected"], [counts["Y"], counts["N"]],
       color=[GREEN, RED], width=0.55)
for i, v in enumerate([counts["Y"], counts["N"]]):
    ax.text(i, v + 12, f"{v}  ({v / len(df):.0%})", ha="center", fontsize=10)
ax.set_ylim(0, counts.max() * 1.15)
ax.set_title("Loan status distribution")
ax.set_ylabel("Number of applications")
save_figure("class_distribution.png")

# ------------------------------------------------------------------
# Chart 2 -- Approval rate by credit history
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.6))
credit_rate = (
    df.dropna(subset=["Credit_History"])
    .groupby("Credit_History")
    .Approved.mean() * 100
)
ax.bar(
    ["No credit history (0)", "Good credit history (1)"],
    credit_rate.values,
    color=[RED, GREEN], width=0.55,
)
for i, v in enumerate(credit_rate.values):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Approval rate (%)")
ax.set_title("Approval rate by credit history")
save_figure("credit_history.png")

# ------------------------------------------------------------------
# Chart 3 -- Applicant income distribution by outcome (KDE)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for status, label in [("Y", "Approved"), ("N", "Rejected")]:
    subset = df[df.Loan_Status == status].ApplicantIncome
    sns.kdeplot(np.log10(subset), ax=ax,
                fill=True, color=pal[status], label=label, alpha=0.35)
ax.set_xlabel("Applicant income  (log10 scale, INR/month)")
ax.set_title("Applicant income distribution by outcome")
ax.legend()
save_figure("income_distribution.png")

# ------------------------------------------------------------------
# Chart 4 -- Approval rate by property area, education, dependents
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, col in zip(axes, ["Property_Area", "Education", "Dependents"]):
    rates = df.groupby(col).Approved.mean().mul(100).sort_values()
    ax.barh(rates.index, rates.values, color=PINE, height=0.55)
    for i, v in enumerate(rates.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_title(col.replace("_", " "))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Approval rate (%)")
save_figure("categorical_rates.png")

# ------------------------------------------------------------------
# Chart 5 -- Correlation matrix of numeric columns
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5))
numeric_cols = df[[
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Approved",
]]
sns.heatmap(
    numeric_cols.corr(),
    annot=True, fmt=".2f",
    cmap="BrBG", center=0,
    ax=ax, cbar=False, square=True,
)
ax.set_title("Correlation matrix of numeric columns")
save_figure("correlation.png")

# ------------------------------------------------------------------
# Chart 6 -- Missing values per column
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
missing = df.isna().sum()
missing = missing[missing > 0].sort_values()
ax.barh(missing.index, missing.values, color=AMBER, height=0.55)
for i, v in enumerate(missing.values):
    ax.text(v + 1, i, str(v), va="center", fontsize=10)
ax.set_title("Missing values per column")
ax.set_xlabel("Number of missing rows")
save_figure("missing_values.png")


# ==============================================================================
# SECTION 8 -- TRAIN / TEST SPLIT
# ==============================================================================

print("\n--- Preparing train / test split ---")

X = df[RAW_FEATURES]   # input features (9 columns -- no Gender or Married)
y = df["Approved"]     # target: 1 = approved, 0 = rejected

# 80 % training, 20 % test, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"[split]  Training rows : {len(X_train)}")
print(f"[split]  Test rows     : {len(X_test)}")
print(f"[split]  Train approval rate : {y_train.mean():.1%}")
print(f"[split]  Test  approval rate : {y_test.mean():.1%}")

# 5-fold stratified cross-validation used during grid search
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==============================================================================
# SECTION 9 -- MODEL CANDIDATES AND HYPER-PARAMETER GRIDS
# ==============================================================================
#
# Four classifiers of increasing complexity are compared.
# Each is wrapped in a Pipeline with the preprocessing stage so that
# cross-validation never leaks test-set statistics into training.
#
# Grid search optimises ROC-AUC (better than accuracy for imbalanced data).
# ==============================================================================

model_candidates = {

    # ------------------------------------------------------------------
    # 1. Logistic Regression -- transparent linear baseline
    #    C controls regularisation strength (lower = stronger regularisation)
    # ------------------------------------------------------------------
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
        },
    },

    # ------------------------------------------------------------------
    # 2. Decision Tree -- human-readable rule tree
    #    max_depth limits overfitting; min_samples_leaf ensures leaf size
    # ------------------------------------------------------------------
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "clf__max_depth":        [3, 4, 5, 7],
            "clf__min_samples_leaf": [5, 15],
        },
    },

    # ------------------------------------------------------------------
    # 3. Random Forest -- decorrelated ensemble of decision trees
    #    300 trees; depth and leaf-size tuned via grid search
    # ------------------------------------------------------------------
    "Random Forest": {
        "estimator": RandomForestClassifier(n_estimators=300, random_state=42),
        "param_grid": {
            "clf__max_depth":        [4, 6, None],
            "clf__min_samples_leaf": [3, 8],
        },
    },

    # ------------------------------------------------------------------
    # 4. Gradient Boosting -- sequential ensemble, each tree corrects errors
    #    of the previous one; more sensitive to hyper-parameters
    # ------------------------------------------------------------------
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators":  [100, 200],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth":     [2, 3],
        },
    },
}


# ==============================================================================
# SECTION 10 -- TRAINING LOOP (GRID SEARCH + CROSS-VALIDATION)
# ==============================================================================

print("\n--- Training models ---")

all_results = []    # list of metric dicts, one per model
fitted_models = {}  # name -> best fitted pipeline

for model_name, config in model_candidates.items():

    # Build a full pipeline: preprocessing -> classifier
    pipeline = Pipeline(steps=[
        ("prep", build_preprocessing_pipeline()),
        ("clf",  config["estimator"]),
    ])

    # Grid search with 5-fold CV, scoring by ROC-AUC
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        cv=cross_val,
        scoring="roc_auc",
        n_jobs=-1,          # use all CPU cores
        refit=True,         # refit best params on full training set
    )
    grid_search.fit(X_train, y_train)

    best_pipeline  = grid_search.best_estimator_
    fitted_models[model_name] = best_pipeline

    # Evaluate the best pipeline on the held-out test set
    proba_test   = best_pipeline.predict_proba(X_test)[:, 1]
    pred_test    = (proba_test >= 0.5).astype(int)

    result = {
        "model":       model_name,
        "cv_auc":      round(grid_search.best_score_, 4),
        "accuracy":    round(accuracy_score(y_test, pred_test),      4),
        "precision":   round(precision_score(y_test, pred_test),     4),
        "recall":      round(recall_score(y_test, pred_test),        4),
        "f1":          round(f1_score(y_test, pred_test),            4),
        "roc_auc":     round(roc_auc_score(y_test, proba_test),      4),
        "best_params": {
            k.replace("clf__", ""): v
            for k, v in grid_search.best_params_.items()
        },
    }
    all_results.append(result)

    print(
        f"  {model_name:22s}  "
        f"cv_auc={result['cv_auc']:.3f}  "
        f"accuracy={result['accuracy']:.3f}  "
        f"f1={result['f1']:.3f}  "
        f"roc_auc={result['roc_auc']:.3f}"
    )


# ==============================================================================
# SECTION 11 -- SELECT BEST MODEL
# ==============================================================================

# Select by cross-validated ROC-AUC (not test AUC) to avoid test-set selection bias
best_result    = max(all_results, key=lambda r: (r["cv_auc"], r["f1"]))
best_name      = best_result["model"]
best_model     = fitted_models[best_name]

# Re-evaluate the best model to get confusion-matrix values
proba_best     = best_model.predict_proba(X_test)[:, 1]
pred_best      = (proba_best >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_best).ravel()

print(f"\n[select]  Best model : {best_name}")
print(f"[select]  Test accuracy : {best_result['accuracy']:.3f}")
print(f"[select]  Test ROC-AUC  : {best_result['roc_auc']:.3f}")
print(f"[select]  Recall (approved) : {best_result['recall']:.3f}")
print(f"[select]  Confusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")


# ==============================================================================
# SECTION 12 -- FAIRNESS AUDIT (Gender excluded from model)
# ==============================================================================
#
# Because Gender was not given to the model, any difference in predicted
# approval rates between male and female applicants must be caused by
# the other variables (income, credit history, loan size, etc.) rather
# than by direct gender discrimination.
#
# This audit compares the model's predicted approval rate with the actual
# approval rate in the test set for each gender group.
# ==============================================================================

print("\n--- Fairness audit (gender) ---")

audit_df = (
    df.loc[X_test.index, ["Gender"]]
    .assign(actual=y_test.values, predicted=pred_best)
    .dropna()
)

fairness_by_gender = {
    gender: {
        "n":              int(len(group)),
        "actual_rate":    round(group.actual.mean(),    3),
        "predicted_rate": round(group.predicted.mean(), 3),
    }
    for gender, group in audit_df.groupby("Gender")
}

for gender, stats in fairness_by_gender.items():
    print(
        f"  {gender:8s}  n={stats['n']:4d}  "
        f"actual={stats['actual_rate']:.1%}  "
        f"predicted={stats['predicted_rate']:.1%}  "
        f"diff={abs(stats['actual_rate'] - stats['predicted_rate']):.1%}"
    )


# ==============================================================================
# SECTION 13 -- EVALUATION CHARTS
# ==============================================================================

print("\n--- Evaluation charts ---")

# ------------------------------------------------------------------
# Chart 7 -- ROC curves for all four models
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.4))
line_colors = [GREY, AMBER, PINE, INK]

for (name, model), color in zip(fitted_models.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_val      = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    linewidth    = 2.4 if name == best_name else 1.5
    ax.plot(fpr, tpr, color=color, lw=linewidth, label=f"{name}  ({auc_val:.3f})")

ax.plot([0, 1], [0, 1], "--", color="#B8C4C0", label="Random baseline (0.500)")
ax.set_xlabel("False positive rate  (1 - Specificity)")
ax.set_ylabel("True positive rate  (Sensitivity / Recall)")
ax.set_title("ROC curves -- test set")
ax.legend(loc="lower right", fontsize=8.5)
save_figure("roc_curves.png")

# ------------------------------------------------------------------
# Chart 8 -- Confusion matrix for the best model
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 4))
sns.heatmap(
    [[tn, fp], [fn, tp]],
    annot=True, fmt="d",
    cmap=sns.light_palette(PINE, as_cmap=True),
    cbar=False, ax=ax,
    annot_kws={"size": 16},
    xticklabels=["Predicted Rejected", "Predicted Approved"],
    yticklabels=["Actually Rejected", "Actually Approved"],
)
ax.set_title(f"Confusion matrix -- {best_name}")
save_figure("confusion_matrix.png")

# ------------------------------------------------------------------
# Chart 9 -- Permutation feature importance
# (measures how much ROC-AUC drops when each feature is shuffled)
# ------------------------------------------------------------------
perm_imp = permutation_importance(
    best_model, X_test, y_test,
    scoring="roc_auc",
    n_repeats=15,
    random_state=42,
    n_jobs=-1,
)
importance_series = pd.Series(
    perm_imp.importances_mean,
    index=RAW_FEATURES,
).sort_values()

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(
    importance_series.index.str.replace("_", " "),
    importance_series.values,
    color=PINE, height=0.6,
)
ax.set_xlabel("Drop in ROC-AUC when feature is randomly shuffled")
ax.set_title(f"Permutation feature importance -- {best_name}")
save_figure("feature_importance.png")

# ------------------------------------------------------------------
# Chart 10 -- Model comparison bar chart (accuracy, F1, ROC-AUC)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
comparison_df = (
    pd.DataFrame(all_results)
    .set_index("model")[["accuracy", "f1", "roc_auc"]]
)
comparison_df.plot.bar(
    ax=ax,
    color=[PINE, AMBER, INK],
    width=0.75,
    rot=15,
)
ax.set_ylim(0.5, 1.1)
ax.set_xlabel("")
ax.set_title("Model comparison -- test set")
ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
save_figure("model_comparison.png")


# ==============================================================================
# SECTION 14 -- REFIT ON ALL DATA AND SAVE
# ==============================================================================
#
# The best pipeline was selected using only the training split.
# Now refit it on the entire dataset so the deployed model benefits
# from all 1,200 examples.
# ==============================================================================

print("\n--- Saving model and metrics ---")

# Refit on the complete dataset (X, y not just X_train, y_train)
best_model.fit(X, y)

# Save the fitted pipeline
model_path = MODELS / "loan_model.joblib"
joblib.dump(best_model, model_path)
print(f"[save]  Model saved  -> {model_path}")

# Build the metrics payload
metrics_payload = {
    "best_model": best_name,
    "test": {
        key: best_result[key]
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    },
    "confusion": {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    },
    "models": all_results,
    "importance": {
        feature: round(float(value), 4)
        for feature, value in importance_series.items()
    },
    "fairness_gender": fairness_by_gender,
    "dataset": {
        "rows":          len(df),
        "columns":       df.shape[1] - 1,          # exclude the target column
        "approval_rate": round(df.Approved.mean(), 3),
        "missing_cells": int(df.isna().sum().sum()),
        "train_rows":    len(X_train),
        "test_rows":     len(X_test),
    },
}

# Save metrics as JSON
metrics_path = MODELS / "metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(metrics_payload, fh, indent=2)
print(f"[save]  Metrics saved -> {metrics_path}")


# ==============================================================================
# SECTION 15 -- FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)
print(f"  Best model    : {best_name}")
print(f"  Test accuracy : {best_result['accuracy']:.1%}")
print(f"  Test F1 score : {best_result['f1']:.3f}")
print(f"  Test ROC-AUC  : {best_result['roc_auc']:.3f}")
print(f"  Recall        : {best_result['recall']:.1%}  of approved applications found")
print(f"  Top feature   : Credit_History  (importance = {importance_series['Credit_History']:.3f})")
print("\n  Files written:")
print(f"    {model_path}")
print(f"    {metrics_path}")
print(f"    static/figures/  (10 charts)")
print("\n  Next step:  python app.py  -> open http://127.0.0.1:5000")
print("=" * 65)
