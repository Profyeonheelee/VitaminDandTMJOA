# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:14:24 2026

@author: USER
"""

# ============================================================
# Figure 3. XGBoost model performance for TMJ OA prediction
# Validation: stratified 80:20 train-test split
# This code is aligned with New Table 4.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier


# ============================================================
# 1. File path
# ============================================================

DATA_DIR = Path(r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo")
DATA_PATH = DATA_DIR / "TMJOA_VitaminD_read.csv"

OUT_DIR = DATA_DIR / "Figure3_model_performance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

print("Data file:", DATA_PATH)
print("Output directory:", OUT_DIR)


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)

print("\nData shape:", data.shape)


# ============================================================
# 3. Preprocessing
# ============================================================

outcome = "TMJ_OA"
data[outcome] = data[outcome].astype(int)

# Sex recoding: SEX_FEMA = 2 means female
data["SEX_FEMALE"] = (data["SEX_FEMA"] == 2).astype(int)

# Symptom duration log-transform
data["Symptom_duration_log1p"] = np.log1p(data["SYMPTOM"].clip(lower=0))


# ============================================================
# 4. Feature blocks
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
    "Clinical + labs w/o Vitamin D": clinical_features + labs_without_vitaminD,
    "Clinical + labs + Vitamin D": clinical_features + labs_without_vitaminD + vitaminD_feature,
    "Clinical + labs + Vitamin D + GSI": clinical_features + labs_without_vitaminD + vitaminD_feature + gsi_feature
}

model_order = list(model_blocks.keys())


# ============================================================
# 5. Check variables
# ============================================================

all_features = sorted(set(sum(model_blocks.values(), [])))
required_columns = [outcome] + all_features

missing_columns = [col for col in required_columns if col not in data.columns]

if len(missing_columns) > 0:
    raise ValueError(f"Missing columns: {missing_columns}")

analysis_data = data[required_columns].copy()

X_all = analysis_data[all_features]
y_all = analysis_data[outcome].astype(int)


# ============================================================
# 6. Stratified 80:20 train-test split
# ============================================================

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    stratify=y_all,
    random_state=42
)

print("\nTrain n:", len(y_train))
print("Test n:", len(y_test))

print("\nTest outcome distribution:")
print(y_test.value_counts())


# ============================================================
# 7. XGBoost pipeline
# ============================================================

xgb_pipeline = Pipeline([
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


# ============================================================
# 8. Helper functions
# ============================================================

def sensitivity_specificity(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return sensitivity, specificity


def net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return nb


def treat_all_net_benefit(y_true, threshold):
    prevalence = np.mean(y_true)
    nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
    return nb


# ============================================================
# 9. Fit models and calculate predictions
# ============================================================

results = {}
prediction_rows = []

for model_name in model_order:

    print(f"\nRunning XGBoost model: {model_name}")

    feature_list = model_blocks[model_name]

    X_train = X_train_all[feature_list]
    X_test = X_test_all[feature_list]

    xgb_pipeline.fit(X_train, y_train)

    y_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=6,
        strategy="quantile"
    )

    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    sensitivity, specificity = sensitivity_specificity(y_true, y_prob)

    thresholds = np.arange(0.05, 0.81, 0.01)

    nb_model = np.array([
        net_benefit(y_true, y_prob, t)
        for t in thresholds
    ])

    nb_treat_all = np.array([
        treat_all_net_benefit(y_true, t)
        for t in thresholds
    ])

    nb_treat_none = np.zeros_like(thresholds)

    results[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auroc": auroc,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "auprc": auprc,
        "brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "thresholds": thresholds,
        "nb_model": nb_model,
        "nb_treat_all": nb_treat_all,
        "nb_treat_none": nb_treat_none,
        "y_true": y_true,
        "y_prob": y_prob
    }

    temp_pred = pd.DataFrame({
        "Model": model_name,
        "y_true": y_true,
        "y_pred_prob": y_prob
    })

    prediction_rows.append(temp_pred)


prediction_df = pd.concat(prediction_rows, ignore_index=True)


# ============================================================
# 10. Metrics table
# ============================================================

metrics_rows = []

for model_name in model_order:
    metrics_rows.append({
        "Model": model_name,
        "AUROC": results[model_name]["auroc"],
        "AUPRC": results[model_name]["auprc"],
        "Sensitivity": results[model_name]["sensitivity"],
        "Specificity": results[model_name]["specificity"],
        "Brier score": results[model_name]["brier"]
    })

metrics_df = pd.DataFrame(metrics_rows)

print("\nModel metrics:")
print(metrics_df.round(3).to_string(index=False))


# ============================================================
# 11. Plot Figure 3
# ============================================================

colors = {
    "Clinical only": "#4C72B0",
    "Clinical + labs w/o Vitamin D": "#DD715B",
    "Clinical + labs + Vitamin D": "#55A868",
    "Clinical + labs + Vitamin D + GSI": "#8172B2"
}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax1, ax2, ax3, ax4 = axes.ravel()


# ------------------------------------------------------------
# A. ROC curves
# ------------------------------------------------------------

for model_name in model_order:
    ax1.plot(
        results[model_name]["fpr"],
        results[model_name]["tpr"],
        lw=2.2,
        color=colors[model_name],
        label=f"{model_name} (AUROC={results[model_name]['auroc']:.3f})"
    )

ax1.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.4,
    alpha=0.8
)

ax1.set_title("A. ROC curves", fontsize=12)
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate")
ax1.legend(fontsize=8, loc="lower right", frameon=False)


# ------------------------------------------------------------
# B. Calibration curves
# ------------------------------------------------------------

for model_name in model_order:
    ax2.plot(
        results[model_name]["mean_pred"],
        results[model_name]["frac_pos"],
        marker="o",
        lw=2.2,
        color=colors[model_name],
        label=model_name
    )

ax2.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    lw=1.4,
    alpha=0.8
)

ax2.set_title("B. Calibration curves", fontsize=12)
ax2.set_xlabel("Predicted probability")
ax2.set_ylabel("Observed probability")
ax2.legend(fontsize=8, loc="upper left", frameon=False)


# ------------------------------------------------------------
# C. Performance metrics
# ------------------------------------------------------------

metric_order = [
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity",
    "Brier score"
]

y_positions = np.arange(len(metric_order))

for model_name in model_order:

    values = [
        results[model_name]["auroc"],
        results[model_name]["auprc"],
        results[model_name]["sensitivity"],
        results[model_name]["specificity"],
        results[model_name]["brier"]
    ]

    ax3.plot(
        values,
        y_positions,
        marker="o",
        lw=2.2,
        color=colors[model_name],
        label=model_name
    )

ax3.set_yticks(y_positions)
ax3.set_yticklabels(metric_order)
ax3.set_xlim(0, 1)
ax3.set_title("C. Performance metrics", fontsize=12)
ax3.set_xlabel("Performance value")
ax3.legend(fontsize=8, loc="lower right", frameon=False)


# ------------------------------------------------------------
# D. Decision curve analysis
# ------------------------------------------------------------

for model_name in model_order:
    ax4.plot(
        results[model_name]["thresholds"],
        results[model_name]["nb_model"],
        lw=2.2,
        color=colors[model_name],
        label=model_name
    )

reference_model = model_order[0]

ax4.plot(
    results[reference_model]["thresholds"],
    results[reference_model]["nb_treat_all"],
    linestyle="--",
    color="gray",
    lw=1.6,
    label="Treat all"
)

ax4.plot(
    results[reference_model]["thresholds"],
    results[reference_model]["nb_treat_none"],
    linestyle="--",
    color="black",
    lw=1.6,
    label="Treat none"
)

ax4.set_title("D. Decision curve analysis", fontsize=12)
ax4.set_xlabel("Threshold probability")
ax4.set_ylabel("Net benefit")
ax4.set_xlim(0.05, 0.80)
ax4.legend(fontsize=8, loc="upper right", frameon=False)


plt.tight_layout()


# ============================================================
# 12. Save files and show results
# ============================================================

figure_path = OUT_DIR / "Figure3_ROC_calibration_DCA_XGBoost_80_20.png"
metrics_path = OUT_DIR / "Figure3_model_metrics_XGBoost_80_20.csv"
prediction_path = OUT_DIR / "Figure3_prediction_probabilities_XGBoost_80_20.csv"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)

plt.show()

print("\nSaved files:")
print(figure_path)
print(metrics_path)
print(prediction_path)