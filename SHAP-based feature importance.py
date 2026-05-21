# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:14:51 2026

@author: USER
"""

# ============================================================
# Table 5. SHAP-based feature importance and adjusted logistic
# regression estimates for TMJ OA prediction
# ============================================================

import sys
import subprocess
import importlib.util

# ============================================================
# 0. Install required packages if missing
# ============================================================

required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "statsmodels": "statsmodels",
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

# ============================================================
# 1. Import packages
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm


# ============================================================
# 2. File paths
# ============================================================

DATA_DIR = Path(r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo")

RAW_DATA_PATH = DATA_DIR / "TMJOA_VitaminD_read.csv"
SHAP_IMPORTANCE_PATH = DATA_DIR / "Figure4_SHAP_feature_importance.xlsx"
SHAP_VALUES_PATH = DATA_DIR / "Figure4_SHAP_values.xlsx"

OUT_DIR = DATA_DIR / "Table5_SHAP_logistic_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

if not SHAP_IMPORTANCE_PATH.exists() and not SHAP_VALUES_PATH.exists():
    raise FileNotFoundError(
        "Neither Figure4_SHAP_feature_importance.xlsx nor "
        "Figure4_SHAP_values.xlsx was found."
    )

print("Raw data:", RAW_DATA_PATH)
print("Output directory:", OUT_DIR)


# ============================================================
# 3. Helper functions
# ============================================================

def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def p_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def normalize_name(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
    )


# ============================================================
# 4. Load raw data
# ============================================================

df = pd.read_csv(RAW_DATA_PATH)

print("\nRaw data shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 5. Create analysis variables for adjusted logistic regression
# ============================================================

# Outcome
df["TMJ_OA"] = pd.to_numeric(df["TMJ_OA"], errors="coerce")

# Sex recoding
# Assumption: SEX_FEMA = 2 indicates female
df["Female_sex"] = (df["SEX_FEMA"] == 2).astype(int)

# Symptom duration log-transform
df["Symptom_duration_log1p"] = np.log1p(
    pd.to_numeric(df["SYMPTOM"], errors="coerce").clip(lower=0)
)

# Continuous variables scaled for clinical interpretability
df["Age_10"] = pd.to_numeric(df["AGE"], errors="coerce") / 10
df["VitaminD_10"] = pd.to_numeric(df["VITAMIND"], errors="coerce") / 10
df["ESR_10"] = pd.to_numeric(df["ESR"], errors="coerce") / 10
df["RF_10"] = pd.to_numeric(df["RF"], errors="coerce") / 10
df["Zinc_10"] = pd.to_numeric(df["ZINC"], errors="coerce") / 10
df["GSI_10"] = pd.to_numeric(df["GSI"], errors="coerce") / 10

# Binary clinical variables
binary_cols = ["TMJ_NOIS", "MUSCLE_S", "JAW_LOCK", "BRUXISM"]

for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Logistic regression variables
regression_variables = {
    "Vitamin D": "VitaminD_10",
    "Global Severity Index": "GSI_10",
    "Rheumatoid factor": "RF_10",
    "ESR": "ESR_10",
    "Zinc": "Zinc_10",
    "Age": "Age_10",
    "Symptom duration (log1p)": "Symptom_duration_log1p",
    "Muscle stiffness": "MUSCLE_S",
    "TMJ noise": "TMJ_NOIS",
    "Female sex": "Female_sex",
    "Jaw locking": "JAW_LOCK",
    "Bruxism": "BRUXISM"
}

model_cols = ["TMJ_OA"] + list(regression_variables.values())

analysis_df = df[model_cols].dropna().copy()

print("\nLogistic regression analysis n:", len(analysis_df))

y = analysis_df["TMJ_OA"].astype(int)
X = analysis_df[list(regression_variables.values())]
X = sm.add_constant(X)

logit_model = sm.Logit(y, X).fit(disp=False)

# Extract logistic regression results
logistic_rows = []

for display_name, var_name in regression_variables.items():
    beta = logit_model.params[var_name]
    se = logit_model.bse[var_name]
    p = logit_model.pvalues[var_name]

    or_value = np.exp(beta)
    ci_low = np.exp(beta - 1.96 * se)
    ci_high = np.exp(beta + 1.96 * se)

    logistic_rows.append({
        "Feature": display_name,
        "β": beta,
        "Adjusted OR": or_value,
        "CI low": ci_low,
        "CI high": ci_high,
        "p-value raw": p
    })

logistic_df = pd.DataFrame(logistic_rows)


# ============================================================
# 6. Load SHAP feature importance
# ============================================================

def load_shap_importance_from_importance_file(path):
    shap_df = pd.read_excel(path)

    print("\nSHAP importance file columns:")
    print(shap_df.columns.tolist())

    # Try to detect feature and importance columns
    cols = list(shap_df.columns)

    feature_col_candidates = [
        c for c in cols
        if "feature" in str(c).lower() or "variable" in str(c).lower()
    ]

    importance_col_candidates = [
        c for c in cols
        if (
            ("shap" in str(c).lower() and "mean" in str(c).lower())
            or "importance" in str(c).lower()
            or "mean_abs" in str(c).lower()
            or "mean absolute" in str(c).lower()
        )
    ]

    feature_col = feature_col_candidates[0] if feature_col_candidates else cols[0]

    if importance_col_candidates:
        importance_col = importance_col_candidates[0]
    else:
        # Use the first numeric column that is not the feature column
        numeric_cols = [
            c for c in cols
            if c != feature_col and pd.api.types.is_numeric_dtype(shap_df[c])
        ]
        if len(numeric_cols) == 0:
            raise ValueError("Could not identify SHAP importance column.")
        importance_col = numeric_cols[0]

    shap_imp = shap_df[[feature_col, importance_col]].copy()
    shap_imp.columns = ["Raw feature", "Mean absolute SHAP"]

    shap_imp["Mean absolute SHAP"] = pd.to_numeric(
        shap_imp["Mean absolute SHAP"],
        errors="coerce"
    )

    shap_imp = shap_imp.dropna(subset=["Mean absolute SHAP"])
    return shap_imp


def load_shap_importance_from_values_file(path):
    shap_values = pd.read_excel(path)

    print("\nSHAP values file columns:")
    print(shap_values.columns.tolist())

    exclude_keywords = [
        "row", "id", "index", "true", "actual", "pred", "prob",
        "base", "expected", "outcome", "tmj_oa", "y_"
    ]

    feature_cols = []

    for c in shap_values.columns:
        c_norm = normalize_name(c)
        if any(k in c_norm for k in exclude_keywords):
            continue
        if pd.api.types.is_numeric_dtype(shap_values[c]):
            feature_cols.append(c)

    if len(feature_cols) == 0:
        raise ValueError("No numeric SHAP feature columns found.")

    shap_imp = pd.DataFrame({
        "Raw feature": feature_cols,
        "Mean absolute SHAP": [
            np.nanmean(np.abs(shap_values[c].values))
            for c in feature_cols
        ]
    })

    return shap_imp


if SHAP_IMPORTANCE_PATH.exists():
    shap_importance = load_shap_importance_from_importance_file(
        SHAP_IMPORTANCE_PATH
    )
else:
    shap_importance = load_shap_importance_from_values_file(
        SHAP_VALUES_PATH
    )


# ============================================================
# 7. Map SHAP feature names to manuscript feature names
# ============================================================

feature_name_map = {
    normalize_name("Vitamin D"): "Vitamin D",
    normalize_name("VITAMIND"): "Vitamin D",
    normalize_name("VitaminD_10"): "Vitamin D",

    normalize_name("GSI"): "Global Severity Index",
    normalize_name("GSI_10"): "Global Severity Index",
    normalize_name("Global Severity Index"): "Global Severity Index",

    normalize_name("RF"): "Rheumatoid factor",
    normalize_name("RF_10"): "Rheumatoid factor",
    normalize_name("Rheumatoid factor"): "Rheumatoid factor",

    normalize_name("ESR"): "ESR",
    normalize_name("ESR_10"): "ESR",

    normalize_name("ZINC"): "Zinc",
    normalize_name("Zinc"): "Zinc",
    normalize_name("Zinc_10"): "Zinc",

    normalize_name("AGE"): "Age",
    normalize_name("Age"): "Age",
    normalize_name("Age_10"): "Age",

    normalize_name("SYMPTOM"): "Symptom duration (log1p)",
    normalize_name("Symptom_duration_log1p"): "Symptom duration (log1p)",
    normalize_name("Symptom duration log1p"): "Symptom duration (log1p)",

    normalize_name("MUSCLE_S"): "Muscle stiffness",
    normalize_name("Muscle stiffness"): "Muscle stiffness",

    normalize_name("TMJ_NOIS"): "TMJ noise",
    normalize_name("TMJ noise"): "TMJ noise",

    normalize_name("SEX_FEMALE"): "Female sex",
    normalize_name("Female sex"): "Female sex",
    normalize_name("SEX_FEMA"): "Female sex",

    normalize_name("JAW_LOCK"): "Jaw locking",
    normalize_name("Jaw locking"): "Jaw locking",

    normalize_name("BRUXISM"): "Bruxism",
    normalize_name("Bruxism"): "Bruxism"
}

shap_importance["Feature"] = shap_importance["Raw feature"].apply(
    lambda x: feature_name_map.get(normalize_name(x), str(x))
)

# If multiple raw columns map to the same manuscript feature, sum their SHAP importance
shap_importance_clean = (
    shap_importance
    .groupby("Feature", as_index=False)["Mean absolute SHAP"]
    .sum()
    .sort_values("Mean absolute SHAP", ascending=False)
    .reset_index(drop=True)
)

shap_importance_clean["Rank"] = np.arange(1, len(shap_importance_clean) + 1)

print("\nCleaned SHAP importance:")
print(shap_importance_clean)


# ============================================================
# 8. Merge SHAP importance with logistic regression results
# ============================================================

table5 = shap_importance_clean.merge(
    logistic_df,
    on="Feature",
    how="left"
)

# Keep only features included in logistic regression table
table5 = table5[table5["Feature"].isin(logistic_df["Feature"])].copy()

# Re-rank after filtering
table5 = table5.sort_values("Mean absolute SHAP", ascending=False).reset_index(drop=True)
table5["Rank"] = np.arange(1, len(table5) + 1)


# ============================================================
# 9. Format final Table 5
# ============================================================

table5_formatted = pd.DataFrame({
    "Rank": table5["Rank"],
    "Feature": table5["Feature"],
    "Mean absolute SHAP": table5["Mean absolute SHAP"].map(lambda x: f"{x:.3f}"),
    "βᵃ": table5["β"].map(lambda x: fmt_num(x, 3)),
    "Adjusted ORᵇ": table5["Adjusted OR"].map(lambda x: fmt_num(x, 3)),
    "95% CI": table5.apply(
        lambda r: f"{r['CI low']:.3f}–{r['CI high']:.3f}"
        if pd.notna(r["CI low"]) else "NA",
        axis=1
    ),
    "p-value": table5["p-value raw"].map(
        lambda p: f"{format_p_value(p)}{p_stars(p)}"
    )
})

# Desired manuscript order is SHAP order
table5_formatted = table5_formatted[
    [
        "Rank",
        "Feature",
        "Mean absolute SHAP",
        "βᵃ",
        "Adjusted ORᵇ",
        "95% CI",
        "p-value"
    ]
]


# ============================================================
# 10. Save outputs
# ============================================================

table5_csv_path = OUT_DIR / "Table5_SHAP_logistic_regression.csv"
table5_xlsx_path = OUT_DIR / "Table5_SHAP_logistic_regression.xlsx"
shap_clean_path = OUT_DIR / "Table5_cleaned_SHAP_importance.csv"
logistic_raw_path = OUT_DIR / "Table5_logistic_regression_raw.csv"

table5_formatted.to_csv(
    table5_csv_path,
    index=False,
    encoding="utf-8-sig"
)

shap_importance_clean.to_csv(
    shap_clean_path,
    index=False,
    encoding="utf-8-sig"
)

logistic_df.to_csv(
    logistic_raw_path,
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(table5_xlsx_path, engine="openpyxl") as writer:
    table5_formatted.to_excel(writer, sheet_name="Table5", index=False)
    shap_importance_clean.to_excel(writer, sheet_name="SHAP_importance", index=False)
    logistic_df.to_excel(writer, sheet_name="Logistic_regression", index=False)

print("\nFinal Table 5:")
print(table5_formatted.to_string(index=False))

print("\nSaved files:")
print(table5_csv_path)
print(table5_xlsx_path)
print(shap_clean_path)
print(logistic_raw_path)