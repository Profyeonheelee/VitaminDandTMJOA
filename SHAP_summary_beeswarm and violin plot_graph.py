# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:56:02 2026

@author: USER
"""

# ============================================================
# Figure 4. SHAP-based interpretation of the final XGBoost model
# for TMJ osteoarthritis prediction
#
# A. SHAP summary beeswarm plot
# B. SHAP value distribution violin plot
#
# Final model:
# Clinical + labs + Vitamin D + GSI
# Algorithm:
# XGBoost
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
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
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

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shap


# ============================================================
# 2. File path
# ============================================================

DATA_DIR = Path(r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo")

DATA_PATH = DATA_DIR / "TMJOA_VitaminD_read.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "파일명이 TMJOA_VitaminD_read.csv인지 확인해 주세요."
    )

OUT_DIR = DATA_DIR / "Figure4_SHAP"
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

# Outcome
outcome = "TMJ_OA"
data[outcome] = data[outcome].astype(int)

# Sex recoding
# Original coding assumption:
# SEX_FEMA = 2 -> Female
# SEX_FEMA = 1 -> Male
data["Female_sex"] = (data["SEX_FEMA"] == 2).astype(int)

# Symptom duration log-transform
data["Symptom_duration_log1p"] = np.log1p(data["SYMPTOM"].clip(lower=0))


# ============================================================
# 5. Final model feature set
# ============================================================

# Raw variable names used in the data
feature_cols_raw = [
    "VITAMIND",
    "GSI",
    "RF",
    "ESR",
    "ZINC",
    "AGE",
    "Symptom_duration_log1p",
    "MUSCLE_S",
    "TMJ_NOIS",
    "Female_sex",
    "JAW_LOCK",
    "BRUXISM"
]

# Display names for figure
feature_name_map = {
    "VITAMIND": "Vitamin D",
    "GSI": "Global Severity Index",
    "RF": "Rheumatoid factor",
    "ESR": "ESR",
    "ZINC": "Zinc",
    "AGE": "Age",
    "Symptom_duration_log1p": "Symptom duration (log1p)",
    "MUSCLE_S": "Muscle stiffness",
    "TMJ_NOIS": "TMJ noise",
    "Female_sex": "Female sex",
    "JAW_LOCK": "Jaw locking",
    "BRUXISM": "Bruxism"
}

required_columns = [outcome] + feature_cols_raw

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if len(missing_columns) > 0:
    raise ValueError(
        "다음 변수가 데이터에 없습니다. 변수명을 확인해 주세요:\n"
        f"{missing_columns}"
    )

analysis_data = data[required_columns].copy()

print("\nAnalysis n before imputation:", len(analysis_data))
print("\nMissing values:")
print(analysis_data.isna().sum())


# ============================================================
# 6. Prepare X and y
# ============================================================

X_raw = analysis_data[feature_cols_raw].copy()
y = analysis_data[outcome].astype(int).values

# Rename columns for display
X_display = X_raw.rename(columns=feature_name_map)

# Median imputation
imputer = SimpleImputer(strategy="median")

X_imputed_array = imputer.fit_transform(X_display)

X_imputed = pd.DataFrame(
    X_imputed_array,
    columns=X_display.columns,
    index=X_display.index
)

print("\nFinal model features:")
print(X_imputed.columns.tolist())


# ============================================================
# 7. Fit final XGBoost model
# ============================================================

final_xgb = XGBClassifier(
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
)

final_xgb.fit(X_imputed, y)

print("\nFinal XGBoost model fitted.")


# ============================================================
# 8. Calculate SHAP values
# ============================================================

explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_imputed)

# For binary classification, some SHAP versions return a list
if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

print("\nSHAP values shape:", shap_values.shape)


# ============================================================
# 9. Feature importance table: mean absolute SHAP
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_imputed.columns,
    "Mean absolute SHAP": mean_abs_shap
}).sort_values(
    "Mean absolute SHAP",
    ascending=False
).reset_index(drop=True)

ordered_features = importance_df["Feature"].tolist()

# Reorder X and SHAP values
feature_indices = [
    list(X_imputed.columns).index(feature)
    for feature in ordered_features
]

X_ordered = X_imputed[ordered_features]
shap_values_ordered = shap_values[:, feature_indices]

print("\nSHAP feature importance:")
print(importance_df)


# ============================================================
# 10. Save SHAP data tables
# ============================================================

model_input_path = OUT_DIR / "Figure4_model_input_data.xlsx"
shap_values_path = OUT_DIR / "Figure4_SHAP_values.xlsx"
importance_path = OUT_DIR / "Figure4_SHAP_feature_importance.xlsx"

X_imputed.to_excel(model_input_path, index=False)

pd.DataFrame(
    shap_values,
    columns=X_imputed.columns
).to_excel(shap_values_path, index=False)

importance_df.to_excel(importance_path, index=False)

print("\nSaved SHAP data tables:")
print(model_input_path)
print(shap_values_path)
print(importance_path)


# ============================================================
# 11. Figure 4A. SHAP summary beeswarm plot
# ============================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_ordered,
    X_ordered,
    show=False,
    plot_size=None,
    cmap=plt.get_cmap("cool")
)

plt.xlabel(
    "SHAP value (impact on model output)",
    fontsize=12
)

plt.tight_layout()

figure4a_path = OUT_DIR / "Figure4A_SHAP_summary_beeswarm.png"

plt.savefig(
    figure4a_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved Figure 4A:")
print(figure4a_path)


# ============================================================
# 12. Figure 4B. SHAP violin distribution plot
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Data in ordered feature importance order
violin_data = [
    shap_values_ordered[:, i]
    for i in range(shap_values_ordered.shape[1])
]

positions = np.arange(len(ordered_features), 0, -1)

parts = ax.violinplot(
    violin_data,
    positions=positions,
    vert=False,
    widths=0.85,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

# Violin style
for pc in parts["bodies"]:
    pc.set_facecolor("#8FB6D8")
    pc.set_edgecolor("#4F81BD")
    pc.set_alpha(0.70)
    pc.set_linewidth(1.2)

# Median SHAP line for each feature
for pos, values in zip(positions, violin_data):
    median_value = np.median(values)
    ax.vlines(
        x=median_value,
        ymin=pos - 0.18,
        ymax=pos + 0.18,
        color="#C0392B",
        linewidth=1.5
    )

# Zero reference line
ax.axvline(
    x=0,
    color="gray",
    linewidth=1.2
)

# Axes
ax.set_yticks(positions)
ax.set_yticklabels(ordered_features, fontsize=11)

ax.set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=12
)

ax.set_ylabel("")

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

ax.grid(
    axis="y",
    linestyle=":",
    alpha=0.20
)

plt.tight_layout()

figure4b_path = OUT_DIR / "Figure4B_SHAP_violin_distribution.png"

plt.savefig(
    figure4b_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved Figure 4B:")
print(figure4b_path)


# ============================================================
# 13. Final message
# ============================================================

print("\nAll Figure 4 files saved successfully.")
print("\nOutput directory:")
print(OUT_DIR)