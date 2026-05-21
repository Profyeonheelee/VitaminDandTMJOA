# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:12:31 2026

@author: USER
"""

# ============================================================
# Figure 3. ROC and calibration curves for TMJ OA prediction
# A. Elastic Net: ROC curves
# B. Elastic Net: Calibration curves
# C. XGBoost: ROC curves
# D. XGBoost: Calibration curves
#
# Validation: repeated stratified 5-fold cross-validation
# Repeats: 5
# 95% CI bands: bootstrap-based
# ============================================================


# ============================================================
# 0. Install required packages if missing
# ============================================================

import sys
import subprocess
import importlib.util

required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "openpyxl": "openpyxl"
}

for import_name, package_name in required_packages.items():
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])

print("All required packages are ready.")


# ============================================================
# 1. Import packages
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier


# ============================================================
# 2. File path
# ============================================================

DATA_DIR = Path(r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo")
DATA_PATH = DATA_DIR / "TMJOA_VitaminD_read.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "파일 이름이 TMJOA_VitaminD_read.csv인지 확인해 주세요."
    )

OUT_DIR = DATA_DIR / "Figure3_ROC_Calibration_ElasticNet_XGBoost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Data file:", DATA_PATH)
print("Output directory:", OUT_DIR)


# ============================================================
# 3. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)

print("\nData shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())


# ============================================================
# 4. Preprocessing
# ============================================================

outcome = "TMJ_OA"
data[outcome] = data[outcome].astype(int)

# Sex recoding
# Assumption: SEX_FEMA = 2 means female
data["SEX_FEMALE"] = (data["SEX_FEMA"] == 2).astype(int)

# Symptom duration log-transform
data["Symptom_duration_log1p"] = np.log1p(data["SYMPTOM"].clip(lower=0))


# ============================================================
# 5. Feature blocks
# ============================================================

clinical_features = [
    "AGE",
    "SEX_FEMALE",
    "Symptom_duration_log1p",
    "TMJ_NOIS",
    "MUSCLE_S",
    "JAW_LOCK",
    "BRUXISM"
]

labs_without_vitaminD = [
    "ESR",
    "RF",
    "ZINC"
]

vitaminD_feature = [
    "VITAMIND"
]

gsi_feature = [
    "GSI"
]

model_blocks = {
    "Clinical only": clinical_features,
    "Clinical + labs without Vitamin D": clinical_features + labs_without_vitaminD,
    "Clinical + labs + Vitamin D": clinical_features + labs_without_vitaminD + vitaminD_feature,
    "Clinical + labs + Vitamin D + GSI": clinical_features + labs_without_vitaminD + vitaminD_feature + gsi_feature
}

model_order = list(model_blocks.keys())


# ============================================================
# 6. Check required variables
# ============================================================

all_features = sorted(set(sum(model_blocks.values(), [])))
required_columns = [outcome] + all_features

missing_columns = [col for col in required_columns if col not in data.columns]

if len(missing_columns) > 0:
    raise ValueError(
        "다음 변수가 데이터에 없습니다. 변수명을 확인해 주세요:\n"
        f"{missing_columns}"
    )

analysis_data = data[required_columns].copy()

print("\nAnalysis n:", len(analysis_data))
print("\nMissing values:")
print(analysis_data.isna().sum())

X_all = analysis_data[all_features]
y_all = analysis_data[outcome].astype(int).values


# ============================================================
# 7. Model pipelines
# ============================================================

pipeline_en = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=42
    ))
])

pipeline_xgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=80,
        max_depth=3,
        learning_rate=0.30,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        n_jobs=1
    ))
])

algorithms = {
    "Elastic Net": pipeline_en,
    "XGBoost": pipeline_xgb
}


# ============================================================
# 8. Repeated stratified cross-validation
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


def repeated_oof_predictions(X, y, pipeline, cv):
    """
    Repeated out-of-fold predictions.
    Each subject receives multiple out-of-fold predictions across repeats.
    Final probability is averaged for each subject.
    """

    pred_sum = np.zeros(len(y), dtype=float)
    pred_count = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        model = clone(pipeline)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]

        model.fit(X_train, y_train)

        pred_prob = model.predict_proba(X_test)[:, 1]

        pred_sum[test_idx] += pred_prob
        pred_count[test_idx] += 1

    final_pred = pred_sum / pred_count

    return final_pred


# ============================================================
# 9. Generate predictions
# ============================================================

prediction_df = pd.DataFrame({
    "row_id": np.arange(len(analysis_data)),
    "TMJ_OA": y_all
})

results = {}

for algorithm_name, pipeline in algorithms.items():

    results[algorithm_name] = {}

    for model_name in model_order:

        print(f"Running {algorithm_name}: {model_name}")

        features = model_blocks[model_name]
        X = analysis_data[features].copy()

        y_prob = repeated_oof_predictions(
            X=X,
            y=y_all,
            pipeline=pipeline,
            cv=cv
        )

        prediction_df[f"{algorithm_name} | {model_name}"] = y_prob

        fpr, tpr, _ = roc_curve(y_all, y_prob)
        auroc = roc_auc_score(y_all, y_prob)
        auprc = average_precision_score(y_all, y_prob)
        brier = brier_score_loss(y_all, y_prob)

        results[algorithm_name][model_name] = {
            "y_prob": y_prob,
            "fpr": fpr,
            "tpr": tpr,
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier
        }


# ============================================================
# 10. Bootstrap CI functions
# ============================================================

rng = np.random.default_rng(42)

roc_grid = np.linspace(0, 1, 201)


def bootstrap_roc_ci(y_true, y_prob, n_boot=500):
    """
    Bootstrap 95% confidence band for ROC curve.
    """

    tprs = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)

        interp_tpr = np.interp(roc_grid, fpr_b, tpr_b)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        tprs.append(interp_tpr)

    tprs = np.asarray(tprs)

    lower = np.percentile(tprs, 2.5, axis=0)
    upper = np.percentile(tprs, 97.5, axis=0)

    return lower, upper


def calibration_points_with_ci(y_true, y_prob, n_bins=6, n_boot=500):
    """
    Quantile-bin calibration curve with bootstrap 95% CI.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)

    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)

    mean_pred = []
    observed = []

    for b in bins:
        mean_pred.append(np.mean(y_prob[b]))
        observed.append(np.mean(y_true[b]))

    mean_pred = np.array(mean_pred)
    observed = np.array(observed)

    boot_observed = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        y_b = y_true[idx]
        p_b = y_prob[idx]

        # assign bootstrap samples to original probability-bin boundaries
        boot_vals = []

        for b in bins:
            lower_edge = np.min(y_prob[b])
            upper_edge = np.max(y_prob[b])

            if b is bins[-1]:
                mask = (p_b >= lower_edge) & (p_b <= upper_edge)
            else:
                mask = (p_b >= lower_edge) & (p_b < upper_edge)

            if np.sum(mask) == 0:
                boot_vals.append(np.nan)
            else:
                boot_vals.append(np.mean(y_b[mask]))

        boot_observed.append(boot_vals)

    boot_observed = np.asarray(boot_observed)

    lower = np.nanpercentile(boot_observed, 2.5, axis=0)
    upper = np.nanpercentile(boot_observed, 97.5, axis=0)

    return mean_pred, observed, lower, upper


# ============================================================
# 11. Metrics table
# ============================================================

metrics_rows = []

for algorithm_name in algorithms.keys():
    for model_name in model_order:
        r = results[algorithm_name][model_name]

        metrics_rows.append({
            "Algorithm": algorithm_name,
            "Model": model_name,
            "AUROC": r["auroc"],
            "AUPRC": r["auprc"],
            "Brier score": r["brier"]
        })

metrics_df = pd.DataFrame(metrics_rows)

metrics_df.to_csv(
    OUT_DIR / "Figure3_model_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    OUT_DIR / "Figure3_prediction_probabilities.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. Plot settings
# ============================================================

colors = {
    "Clinical only": "#8FB6D8",
    "Clinical + labs without Vitamin D": "#F2C57C",
    "Clinical + labs + Vitamin D": "#97C1A9",
    "Clinical + labs + Vitamin D + GSI": "#D9A5B3"
}

line_alpha = 0.95
ci_alpha = 0.20


# ============================================================
# 13. Create Figure 3
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9.8))

axA, axB, axC, axD = axes.ravel()


# ------------------------------------------------------------
# A. Elastic Net: ROC curves
# ------------------------------------------------------------

for model_name in model_order:
    r = results["Elastic Net"][model_name]

    lower, upper = bootstrap_roc_ci(
        y_true=y_all,
        y_prob=r["y_prob"],
        n_boot=500
    )

    axA.step(
        r["fpr"],
        r["tpr"],
        where="post",
        color=colors[model_name],
        lw=2.0,
        alpha=line_alpha,
        label=f"{model_name} (AUC={r['auroc']:.3f})"
    )

    axA.fill_between(
        roc_grid,
        lower,
        upper,
        color=colors[model_name],
        alpha=ci_alpha,
        linewidth=0
    )

axA.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.2,
    alpha=0.85
)

axA.set_title("A. Elastic Net: ROC curves", fontsize=13)
axA.set_xlabel("1 - Specificity")
axA.set_ylabel("Sensitivity")
axA.legend(fontsize=8, loc="lower right", frameon=False)


# ------------------------------------------------------------
# B. Elastic Net: Calibration curves
# ------------------------------------------------------------

for model_name in model_order:
    r = results["Elastic Net"][model_name]

    mean_pred, observed, lower, upper = calibration_points_with_ci(
        y_true=y_all,
        y_prob=r["y_prob"],
        n_bins=6,
        n_boot=500
    )

    axB.plot(
        mean_pred,
        observed,
        marker="o",
        markersize=5,
        color=colors[model_name],
        lw=2.0,
        alpha=line_alpha,
        label=model_name
    )

    axB.fill_between(
        mean_pred,
        lower,
        upper,
        color=colors[model_name],
        alpha=ci_alpha,
        linewidth=0
    )

axB.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.2,
    alpha=0.85
)

axB.set_title("B. Elastic Net: Calibration curves", fontsize=13)
axB.set_xlabel("Mean predicted probability")
axB.set_ylabel("Observed event rate")
axB.legend(fontsize=8, loc="upper left", frameon=False)


# ------------------------------------------------------------
# C. XGBoost: ROC curves
# ------------------------------------------------------------

for model_name in model_order:
    r = results["XGBoost"][model_name]

    lower, upper = bootstrap_roc_ci(
        y_true=y_all,
        y_prob=r["y_prob"],
        n_boot=500
    )

    axC.step(
        r["fpr"],
        r["tpr"],
        where="post",
        color=colors[model_name],
        lw=2.0,
        alpha=line_alpha,
        label=f"{model_name} (AUC={r['auroc']:.3f})"
    )

    axC.fill_between(
        roc_grid,
        lower,
        upper,
        color=colors[model_name],
        alpha=ci_alpha,
        linewidth=0
    )

axC.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.2,
    alpha=0.85
)

axC.set_title("C. XGBoost: ROC curves", fontsize=13)
axC.set_xlabel("1 - Specificity")
axC.set_ylabel("Sensitivity")
axC.legend(fontsize=8, loc="lower right", frameon=False)


# ------------------------------------------------------------
# D. XGBoost: Calibration curves
# ------------------------------------------------------------

for model_name in model_order:
    r = results["XGBoost"][model_name]

    mean_pred, observed, lower, upper = calibration_points_with_ci(
        y_true=y_all,
        y_prob=r["y_prob"],
        n_bins=6,
        n_boot=500
    )

    axD.plot(
        mean_pred,
        observed,
        marker="o",
        markersize=5,
        color=colors[model_name],
        lw=2.0,
        alpha=line_alpha,
        label=model_name
    )

    axD.fill_between(
        mean_pred,
        lower,
        upper,
        color=colors[model_name],
        alpha=ci_alpha,
        linewidth=0
    )

axD.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.2,
    alpha=0.85
)

axD.set_title("D. XGBoost: Calibration curves", fontsize=13)
axD.set_xlabel("Mean predicted probability")
axD.set_ylabel("Observed event rate")
axD.legend(fontsize=8, loc="upper left", frameon=False)


plt.tight_layout()


# ============================================================
# 14. Save and show
# ============================================================

figure_path = OUT_DIR / "Figure3_ROC_Calibration_ElasticNet_XGBoost_CI_A_dot.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved files:")
print(figure_path)
print(OUT_DIR / "Figure3_model_metrics.csv")
print(OUT_DIR / "Figure3_prediction_probabilities.csv")

# Open output folder automatically on Windows
os.startfile(OUT_DIR)