# ============================================================
# Table 4. Machine-learning model performance for TMJ OA prediction
# Train:test split = 8:2
# XGBoost learning_rate = 0.30
#
# Output columns:
# Algorithm
# Model
# AUROC (95% CI)
# p-value of ΔAUROC
# AUPRC
# Brier score
# Calibration intercept
# Calibration slope
# Sensitivity
# Specificity
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
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
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

print("All required packages are ready.")


# ============================================================
# 1. Import packages
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from xgboost import XGBClassifier
import statsmodels.api as sm


# ============================================================
# 2. File path
# ============================================================

DATA_DIR = Path(r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo")

candidate_files = [
    DATA_DIR / "TMJOA_VitaminD_read.csv",
    DATA_DIR / "TMJOA_VitaminD_read.xlsx",
    DATA_DIR / "TMJOA_VitaminD_read.txt"
]

DATA_PATH = None

for file in candidate_files:
    if file.exists():
        DATA_PATH = file
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "TMJOA_VitaminD_read 파일을 찾을 수 없습니다. "
        "파일 이름과 확장자(csv, xlsx, txt)를 확인해 주세요."
    )

OUT_DIR = DATA_DIR / "Table4_model_performance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Data file:", DATA_PATH)
print("Output directory:", OUT_DIR)


# ============================================================
# 3. Load data
# ============================================================

if DATA_PATH.suffix.lower() == ".csv":
    data = pd.read_csv(DATA_PATH)

elif DATA_PATH.suffix.lower() == ".xlsx":
    data = pd.read_excel(DATA_PATH)

elif DATA_PATH.suffix.lower() == ".txt":
    data = pd.read_csv(DATA_PATH, sep="\t")

else:
    raise ValueError("지원되지 않는 파일 형식입니다.")


print("\nData shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())


# ============================================================
# 4. Preprocessing
# ============================================================

# Sex recoding
# Original coding assumption:
# SEX_FEMA = 2 → Female
# SEX_FEMA = 1 → Male
data["SEX_FEMALE"] = (data["SEX_FEMA"] == 2).astype(int)

# Symptom duration log-transform
data["Symptom_duration_log1p"] = np.log1p(data["SYMPTOM"].clip(lower=0))

# Outcome
outcome = "TMJ_OA"
data[outcome] = data[outcome].astype(int)


# ============================================================
# 5. Feature blocks
# ============================================================

features_clinical = [
    "AGE",
    "SEX_FEMALE",
    "Symptom_duration_log1p",
    "TMJ_NOIS",
    "MUSCLE_S",
    "JAW_LOCK",
    "BRUXISM"
]

features_labs = [
    "ESR",
    "RF",
    "ZINC"
]

features_vitD = [
    "VITAMIND"
]

features_gsi = [
    "GSI"
]

blocks = {
    "Clinical only": features_clinical,
    "Clinical + labs without Vitamin D": features_clinical + features_labs,
    "Clinical + labs + Vitamin D": features_clinical + features_labs + features_vitD,
    "Clinical + labs + Vitamin D + GSI": features_clinical + features_labs + features_vitD + features_gsi
}

model_order = list(blocks.keys())


# ============================================================
# 6. Check variables
# ============================================================

all_features = sorted(set(sum(blocks.values(), [])))
required_columns = [outcome] + all_features

missing_features = [
    col for col in required_columns
    if col not in data.columns
]

if len(missing_features) > 0:
    raise ValueError(
        "다음 변수가 데이터에 없습니다. 변수명을 확인해 주세요:\n"
        f"{missing_features}"
    )

analysis_data = data[required_columns].copy()

print("\nAnalysis variables:")
print(required_columns)

print("\nOriginal analysis n:", len(analysis_data))

print("\nMissing values:")
print(analysis_data.isna().sum())


# ============================================================
# 7. Define X and y
# ============================================================

X_all = analysis_data[all_features]
y_all = analysis_data[outcome]


# ============================================================
# 8. Train:test split = 8:2
# ============================================================

row_id_all = np.arange(len(analysis_data))

X_train_all, X_test_all, y_train, y_test, row_train, row_test = train_test_split(
    X_all,
    y_all,
    row_id_all,
    test_size=0.20,
    stratify=y_all,
    random_state=42
)

print("\nTrain n:", len(y_train))
print("Test n:", len(y_test))

print("\nTrain outcome distribution:")
print(y_train.value_counts())

print("\nTest outcome distribution:")
print(y_test.value_counts())


# ============================================================
# 9. DeLong's test functions
# ============================================================

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1

        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T

    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def delong_roc_test(y_true, pred_ref, pred_new):
    y_true = np.asarray(y_true).astype(int)
    pred_ref = np.asarray(pred_ref)
    pred_new = np.asarray(pred_new)

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    preds_sorted = np.vstack((pred_ref, pred_new))[:, order]

    label_1_count = int(y_sorted.sum())

    aucs, cov = fast_delong(preds_sorted, label_1_count)

    auc_ref = aucs[0]
    auc_new = aucs[1]
    delta_auc = auc_new - auc_ref

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    if var <= 0:
        p_value = np.nan
    else:
        z = abs(delta_auc) / np.sqrt(var)
        p_value = 2 * st.norm.sf(z)

    return auc_ref, auc_new, delta_auc, p_value


# ============================================================
# 10. Performance metric functions
# ============================================================

def calibration_intercept_slope(y_true, y_prob):
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    X_cal = sm.add_constant(logit_p)

    cal_model = sm.GLM(
        y_true,
        X_cal,
        family=sm.families.Binomial()
    ).fit()

    intercept = cal_model.params[0]
    slope = cal_model.params[1]

    return intercept, slope


def calculate_metrics(y_true, y_prob, threshold=0.50):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    y_pred_bin = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    cal_intercept, cal_slope = calibration_intercept_slope(
        y_true,
        y_prob
    )

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier score": brier,
        "Calibration intercept": cal_intercept,
        "Calibration slope": cal_slope,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        aucs.append(roc_auc_score(y_b, p_b))

    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)

    return lower, upper


# ============================================================
# 11. Model pipelines
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

models = {
    "Elastic Net": pipeline_en,
    "XGBoost": pipeline_xgb
}


# ============================================================
# 12. Main model loop
# ============================================================

results = []
prediction_rows = []

for algorithm_name, pipeline in models.items():

    previous_pred_prob = None

    for block_name in model_order:

        feature_list = blocks[block_name]

        X_train = X_train_all[feature_list]
        X_test = X_test_all[feature_list]

        pipeline.fit(X_train, y_train)

        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(
            y_test.values,
            y_pred_prob
        )

        auc_low, auc_high = bootstrap_auc_ci(
            y_test.values,
            y_pred_prob,
            n_boot=2000,
            seed=42
        )

        auroc_ci_text = (
            f"{metrics['AUROC']:.3f} "
            f"({auc_low:.3f}–{auc_high:.3f})"
        )

        if previous_pred_prob is None:
            p_delta = "Reference"

        else:
            _, _, delta_auc_value, p_value = delong_roc_test(
                y_test.values,
                previous_pred_prob,
                y_pred_prob
            )

            if np.isnan(p_value):
                p_delta = "NA"
            elif p_value < 0.001:
                p_delta = "<0.001"
            else:
                p_delta = f"{p_value:.3f}"

        results.append({
            "Algorithm": algorithm_name,
            "Model": block_name,
            "AUROC (95% CI)": auroc_ci_text,
            "p-value of ΔAUROC": p_delta,
            "AUPRC": round(metrics["AUPRC"], 3),
            "Brier score": round(metrics["Brier score"], 3),
            "Calibration intercept": round(metrics["Calibration intercept"], 3),
            "Calibration slope": round(metrics["Calibration slope"], 3),
            "Sensitivity": round(metrics["Sensitivity"], 3),
            "Specificity": round(metrics["Specificity"], 3)
        })

        temp_pred = pd.DataFrame({
            "Algorithm": algorithm_name,
            "Model": block_name,
            "Original row index": row_test,
            "y_true": y_test.values,
            "y_pred_prob": y_pred_prob
        })

        prediction_rows.append(temp_pred)

        previous_pred_prob = y_pred_prob


# ============================================================
# 13. Create final Table 4
# ============================================================

table_4 = pd.DataFrame(results)

table_4 = table_4[
    [
        "Algorithm",
        "Model",
        "AUROC (95% CI)",
        "p-value of ΔAUROC",
        "AUPRC",
        "Brier score",
        "Calibration intercept",
        "Calibration slope",
        "Sensitivity",
        "Specificity"
    ]
]

predictions = pd.concat(prediction_rows, ignore_index=True)


# ============================================================
# 14. Save output files
# ============================================================

table_4_csv_path = OUT_DIR / "Table4_model_performance_train_test_8_2.csv"
table_4_xlsx_path = OUT_DIR / "Table4_model_performance_train_test_8_2.xlsx"
prediction_path = OUT_DIR / "Table4_prediction_probabilities_train_test_8_2.csv"

table_4.to_csv(
    table_4_csv_path,
    index=False,
    encoding="utf-8-sig"
)

predictions.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(table_4_xlsx_path, engine="openpyxl") as writer:
    table_4.to_excel(
        writer,
        sheet_name="Table4",
        index=False
    )

    predictions.to_excel(
        writer,
        sheet_name="Prediction_probabilities",
        index=False
    )


# ============================================================
# 15. Display results
# ============================================================

print("\nFinal Table 4:")
print(table_4.to_string(index=False))

try:
    display(table_4)
except NameError:
    pass

print("\nSaved files:")
print(table_4_csv_path)
print(table_4_xlsx_path)
print(prediction_path)