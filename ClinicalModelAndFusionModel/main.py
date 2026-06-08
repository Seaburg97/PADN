#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clinical ordinal regression and DL-clinical stacking fusion."""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


CLINICAL_COLS = [
    "Age",
    "Male",
    "mFS_score",
    "SEBES_score",
    "Acute_hydrocephalus",
    "GCS_score",
    "WFNS_score",
    "Hunt-Hess_score",
    "Posterior_circulation",
    "Size",
    "Hypertension",
    "Clipping",
]

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "data" / "features"
PADN_OUTPUT_DIR = (BASE_DIR / ".." / "PADN" / "outputs" / "dl_models").resolve()
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_FILES = [
    FEATURE_DIR / "featuresaq.csv",
    FEATURE_DIR / "featuresay.csv",
    FEATURE_DIR / "featuresfy.csv",
    FEATURE_DIR / "featurestl.csv",
    FEATURE_DIR / "featuresyjs.csv",
]

TEST_SETS = {
    "Test-Combined": {
        "feature_files": [
            FEATURE_DIR / "featuresth.csv",
            FEATURE_DIR / "featuresay2.csv",
            FEATURE_DIR / "featuresefy.csv",
        ],
        "dl_predictions": PADN_OUTPUT_DIR / "Test-Combined_predictions_best_kappa.csv",
    },
    "th": {
        "feature_files": [
            FEATURE_DIR / "featuresth.csv",
        ],
        "dl_predictions": PADN_OUTPUT_DIR / "external_th_predictions_best_kappa.csv",
    },
    "ay2": {
        "feature_files": [
            FEATURE_DIR / "featuresay2.csv",
        ],
        "dl_predictions": PADN_OUTPUT_DIR / "external_ay2_predictions_best_kappa.csv",
    },
    "efy": {
        "feature_files": [
            FEATURE_DIR / "featuresefy.csv",
        ],
        "dl_predictions": PADN_OUTPUT_DIR / "external_efy_predictions_best_kappa.csv",
    },
}

DL_TRAIN_PRED = PADN_OUTPUT_DIR / "train_predictions_best_kappa.csv"
DL_VAL_PRED = PADN_OUTPUT_DIR / "val_predictions_best_kappa.csv"
BOOTSTRAP_N = 1000
RANDOM_SEED = 42
RFE_MIN_FEATURES = 1
RFE_MANUAL_N_FEATURES = 5
VIF_THRESHOLD = 5.0
R_EXECUTABLE = os.environ.get("RSCRIPT", "")


def resolve_rscript():
    if R_EXECUTABLE:
        return R_EXECUTABLE
    path_rscript = shutil.which("Rscript")
    if path_rscript:
        return path_rscript
    local_rscript = BASE_DIR.parent.parent / ".mamba_envs" / "r-base-only" / "bin" / "Rscript"
    if local_rscript.exists():
        return str(local_rscript)
    return "Rscript"


def metric_kappa(y_true, pred):
    return cohen_kappa_score(
        np.asarray(y_true).astype(int),
        np.asarray(pred).astype(int),
        weights="quadratic",
        labels=[0, 1, 2, 3, 4, 5, 6],
    )


def metric_mae(y_true, pred_label):
    return mean_absolute_error(np.asarray(y_true).astype(float), np.asarray(pred_label).astype(float))


def metric_auc(y_binary, prob):
    y_binary = np.asarray(y_binary).astype(int)
    prob = np.asarray(prob).astype(float)
    if len(np.unique(y_binary)) < 2:
        return np.nan
    return roc_auc_score(y_binary, prob)


def threshold_probs_to_pred_label(threshold_probs, threshold=0.5):
    threshold_probs = np.asarray(threshold_probs, dtype=float)
    if threshold_probs.ndim == 1:
        threshold_probs = threshold_probs.reshape(1, -1)
    return (threshold_probs > threshold).sum(axis=1).astype(int)


def compute_midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(float)
    pred_b = np.asarray(pred_b).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return diff, np.nan
    z = abs(diff) / np.sqrt(var)
    p_value = 2 * stats.norm.sf(z)
    return diff, p_value


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric_fn, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            diff = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
        except Exception:
            continue
        if np.isfinite(diff):
            diffs.append(diff)

    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return observed, np.nan, np.nan, np.nan
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    if observed == 0:
        p_value = 1.0
    else:
        opposite = np.mean(np.sign(diffs) != np.sign(observed))
        p_value = min(1.0, 2.0 * opposite)
    return observed, ci_low, ci_high, p_value


def read_feature_csv(path):
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError(f"{path} is missing patient_id")
    if "mRS" not in df.columns:
        raise ValueError(f"{path} is missing mRS")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["mRS"] = pd.to_numeric(df["mRS"], errors="coerce")
    return df


def load_train_data():
    dfs = [read_feature_csv(path) for path in TRAIN_FILES]
    train = pd.concat(dfs, ignore_index=True)
    train = train.dropna(subset=["mRS"]).copy()
    train["mRS"] = train["mRS"].astype(int)
    train["y_binary"] = (train["mRS"] >= 3).astype(int)
    missing_cols = [c for c in CLINICAL_COLS if c not in train.columns]
    if missing_cols:
        raise ValueError(f"Training data is missing clinical columns: {missing_cols}")
    train = train.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    return train


def load_test_data(feature_files):
    dfs = [read_feature_csv(path) for path in feature_files]
    test = pd.concat(dfs, ignore_index=True)
    test = test.dropna(subset=["mRS"]).copy()
    test["mRS"] = test["mRS"].astype(int)
    test["y_binary"] = (test["mRS"] >= 3).astype(int)
    missing_cols = [c for c in CLINICAL_COLS if c not in test.columns]
    if missing_cols:
        raise ValueError(f"Test data is missing clinical columns: {missing_cols}")
    test = test.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
    return test


def prepare_xy(df, features):
    X = df[features].copy()
    return X


def _fit_ordered_model_frame(X_frame, y_train):
    X_frame = X_frame.astype(float).copy()
    if X_frame.isna().any().any():
        missing = X_frame.columns[X_frame.isna().any()].tolist()
        raise ValueError(f"Ordinal regression input has missing values without imputation: {missing}")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_frame),
        columns=X_frame.columns,
        index=X_frame.index,
    )
    model = OrderedModel(np.asarray(y_train).astype(int), X_scaled, distr="logit")
    last_exc = None
    for method in ("bfgs", "lbfgs", "newton"):
        try:
            result = model.fit(method=method, disp=False, maxiter=1000)
            return {
                "model": model,
                "result": result,
                "scaler": scaler,
                "X_scaled": X_scaled,
                "method": method,
            }
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc
    raise RuntimeError(f"Ordinal regression fitting failed: {last_exc}")


def iterative_vif_filter(X_train, threshold, out_dir):
    remaining = list(X_train.columns)
    history = []
    removed = []
    final_vif_df = pd.DataFrame(columns=["feature", "vif"])

    while len(remaining) > 1:
        X_sub = X_train[remaining].astype(float).copy()
        if X_sub.isna().any().any():
            missing = X_sub.columns[X_sub.isna().any()].tolist()
            raise ValueError(f"VIF input has missing values without imputation: {missing}")
        X_values = StandardScaler().fit_transform(X_sub)
        vif_values = []
        for i in range(X_values.shape[1]):
            try:
                vif_values.append(float(variance_inflation_factor(X_values, i)))
            except Exception:
                vif_values.append(np.inf)
        vif_df = pd.DataFrame({"feature": remaining, "vif": vif_values}).sort_values("vif", ascending=False)
        history.append(vif_df)
        max_vif = float(vif_df.iloc[0]["vif"])
        if not np.isfinite(max_vif) or max_vif <= threshold:
            final_vif_df = vif_df.sort_values("vif", ascending=False).reset_index(drop=True)
            break
        drop_feature = str(vif_df.iloc[0]["feature"])
        removed.append({"dropped_feature": drop_feature, "dropped_vif": max_vif, "drop_reason": "iterative_vif"})
        remaining.remove(drop_feature)
        final_vif_df = vif_df.sort_values("vif", ascending=False).reset_index(drop=True)

    if final_vif_df.empty and remaining:
        X_sub = X_train[remaining].astype(float).copy()
        if X_sub.isna().any().any():
            missing = X_sub.columns[X_sub.isna().any()].tolist()
            raise ValueError(f"VIF input has missing values without imputation: {missing}")
        X_values = StandardScaler().fit_transform(X_sub)
        vif_values = []
        for i in range(X_values.shape[1]):
            try:
                vif_values.append(float(variance_inflation_factor(X_values, i)))
            except Exception:
                vif_values.append(np.inf)
        final_vif_df = pd.DataFrame({"feature": remaining, "vif": vif_values}).sort_values("vif", ascending=False).reset_index(drop=True)

    vif_history_df = pd.concat(history, ignore_index=True) if history else pd.DataFrame(columns=["feature", "vif"])
    vif_history_df.to_csv(out_dir / "vif_iterative_history.csv", index=False, encoding="utf-8-sig")
    final_vif_df.to_csv(out_dir / "vif_final_values.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(removed).to_csv(out_dir / "vif_removed_features.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(final_vif_df))))
    plot_df = final_vif_df.sort_values("vif", ascending=True)
    ax.barh(plot_df["feature"], plot_df["vif"], color="#8c564b")
    ax.axvline(threshold, color="#d62728", linestyle="--", label=f"threshold={threshold:.1f}")
    ax.set_xlabel("VIF")
    ax.set_title("Iterative VIF filtering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "vif_filtering.png", dpi=300)
    plt.close(fig)

    return remaining, final_vif_df


def write_ordinal_lasso_r_script(script_path):
    script = r'''
suppressPackageStartupMessages(library(ordinalNet))

args <- commandArgs(trailingOnly = TRUE)
input_path <- args[[1]]
output_dir <- args[[2]]
alpha <- as.numeric(args[[3]])
n_folds <- as.integer(args[[4]])
seed <- as.integer(args[[5]])

set.seed(seed)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(input_path, check.names = FALSE)
y <- ordered(df[["mRS"]], levels = sort(unique(df[["mRS"]])))
feature_names <- setdiff(colnames(df), "mRS")
x <- as.matrix(df[, feature_names, drop = FALSE])

fold_ids <- sample(rep(seq_len(n_folds), length.out = nrow(x)))
lambda_template <- ordinalNet(
  x = x,
  y = y,
  family = "cumulative",
  link = "logit",
  alpha = alpha,
  standardize = TRUE,
  nLambda = 20
)
lambda_vals <- lambda_template$lambdaVals
path_df <- data.frame(lambda_index = seq_len(nrow(lambda_template$coefs)), lambda_template$coefs, check.names = FALSE)
write.csv(data.frame(lambda_index = seq_along(lambda_vals), lambda = lambda_vals), file.path(output_dir, "ordinal_lasso_lambda_path.csv"), row.names = FALSE)
write.csv(path_df, file.path(output_dir, "ordinal_lasso_coefficient_path.csv"), row.names = FALSE)

loss_mat <- matrix(NA_real_, nrow = n_folds, ncol = length(lambda_vals))
misclass_mat <- matrix(NA_real_, nrow = n_folds, ncol = length(lambda_vals))
brier_mat <- matrix(NA_real_, nrow = n_folds, ncol = length(lambda_vals))
devpct_mat <- matrix(NA_real_, nrow = n_folds, ncol = length(lambda_vals))

for (fold_idx in seq_len(n_folds)) {
  test_idx <- which(fold_ids == fold_idx)
  train_idx <- setdiff(seq_len(nrow(x)), test_idx)
  x_train <- x[train_idx, , drop = FALSE]
  y_train <- y[train_idx]
  x_test <- x[test_idx, , drop = FALSE]
  y_test <- y[test_idx]
  fold_fit <- ordinalNet(
    x = x_train,
    y = y_train,
    family = "cumulative",
    link = "logit",
    alpha = alpha,
    standardize = TRUE,
    lambdaVals = lambda_vals
  )
  pred_prob_list <- lapply(seq_along(lambda_vals), function(j) {
    predict(fold_fit, newx = x_test, whichLambda = j, type = "response")
  })
  pred_class_list <- lapply(seq_along(lambda_vals), function(j) {
    predict(fold_fit, newx = x_test, whichLambda = j, type = "class")
  })
  y_num <- as.integer(y_test) - 1L
  for (j in seq_along(lambda_vals)) {
    p <- as.matrix(pred_prob_list[[j]])
    eps <- 1e-15
    p <- pmax(pmin(p, 1 - eps), eps)
    idx <- cbind(seq_along(y_num), y_num + 1L)
    loss_mat[fold_idx, j] <- -mean(log(p[idx]))
    pred_class <- as.integer(pred_class_list[[j]]) - 1L
    misclass_mat[fold_idx, j] <- mean(pred_class != y_num)
    onehot <- matrix(0, nrow = length(y_num), ncol = ncol(p))
    onehot[idx] <- 1
    brier_mat[fold_idx, j] <- mean(rowSums((p - onehot)^2))
    devpct_mat[fold_idx, j] <- NA_real_
  }
}

cv_df <- data.frame(
  lambda = lambda_vals,
  loglik_loss = colMeans(loss_mat, na.rm = TRUE),
  loglik_loss_sd = apply(loss_mat, 2, sd, na.rm = TRUE),
  misclass = colMeans(misclass_mat, na.rm = TRUE),
  brier = colMeans(brier_mat, na.rm = TRUE),
  dev_pct = colMeans(devpct_mat, na.rm = TRUE)
)
write.csv(cv_df, file.path(output_dir, "ordinal_lasso_cv_curve.csv"), row.names = FALSE)

best_idx <- which.min(cv_df$loglik_loss)
best_loss <- cv_df$loglik_loss[best_idx]
loss_se <- cv_df$loglik_loss_sd[best_idx] / sqrt(n_folds)
one_se_cutoff <- best_loss + loss_se
eligible <- which(cv_df$loglik_loss <= one_se_cutoff)
lambda_1se <- max(cv_df$lambda[eligible])
lambda_min <- cv_df$lambda[best_idx]

fit <- ordinalNet(
  x = x,
  y = y,
  family = "cumulative",
  link = "logit",
  alpha = alpha,
  standardize = TRUE,
  lambdaVals = lambda_min,
  keepTrainingData = TRUE
)

coef_mat <- as.matrix(coef(fit))
coef_df <- data.frame(term = rownames(coef_mat), coef = as.numeric(coef_mat[, 1]))
feature_coef <- coef_df[coef_df$term %in% feature_names, , drop = FALSE]
feature_coef$abs_coef <- abs(feature_coef$coef)
feature_coef$selected <- as.integer(feature_coef$abs_coef > 1e-8)
feature_coef <- feature_coef[order(-feature_coef$abs_coef, feature_coef$term), ]
write.csv(feature_coef, file.path(output_dir, "ordinal_lasso_coefficients.csv"), row.names = FALSE)

selected <- feature_coef[feature_coef$selected == 1, "term"]
if (length(selected) == 0) {
  selected <- feature_coef[order(-feature_coef$abs_coef), "term"][1]
}
write.csv(data.frame(selected_feature = selected), file.path(output_dir, "ordinal_lasso_selected_features.csv"), row.names = FALSE)

summary_df <- data.frame(
  alpha = alpha,
  n_folds = n_folds,
  lambda_min = lambda_min,
  lambda_1se = lambda_1se,
  best_loglik_loss = best_loss,
  best_loglik_loss_se = loss_se,
  one_se_cutoff = one_se_cutoff,
  selected_n = length(selected),
  selected_features = paste(selected, collapse = ";")
)
write.csv(summary_df, file.path(output_dir, "ordinal_lasso_summary.csv"), row.names = FALSE)
'''
    script_path.write_text(script, encoding="utf-8")


def fit_ordinal_lasso_selection(X_train, y_train, out_dir):
    r_executable = resolve_rscript()
    r_executable_path = Path(r_executable)
    if r_executable_path.name != r_executable and not r_executable_path.exists():
        raise FileNotFoundError(f"Rscript was not found: {r_executable}")

    lasso_dir = out_dir / "ordinal_lasso_r"
    lasso_dir.mkdir(parents=True, exist_ok=True)
    input_path = lasso_dir / "ordinal_lasso_input.csv"
    script_path = lasso_dir / "run_ordinal_lasso.R"

    input_df = X_train.copy()
    input_df["mRS"] = np.asarray(y_train).astype(int)
    input_df.to_csv(input_path, index=False, encoding="utf-8-sig")
    write_ordinal_lasso_r_script(script_path)

    cmd = [
        str(r_executable),
        str(script_path),
        str(input_path),
        str(lasso_dir),
        str(ORDINAL_LASSO_ALPHA),
        str(ORDINAL_LASSO_N_FOLDS),
        str(RANDOM_SEED),
    ]
    result = subprocess.run(cmd, cwd=str(out_dir), text=True, capture_output=True, check=False)
    (lasso_dir / "ordinal_lasso_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (lasso_dir / "ordinal_lasso_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ordinalNet failed. See {lasso_dir / 'ordinal_lasso_stderr.txt'}")

    coef_df = pd.read_csv(lasso_dir / "ordinal_lasso_coefficients.csv")
    summary_df = pd.read_csv(lasso_dir / "ordinal_lasso_summary.csv")
    selected_df = pd.read_csv(lasso_dir / "ordinal_lasso_selected_features.csv")
    cv_df = pd.read_csv(lasso_dir / "ordinal_lasso_cv_curve.csv")
    path_df = pd.read_csv(lasso_dir / "ordinal_lasso_coefficient_path.csv")
    lambda_path_df = pd.read_csv(lasso_dir / "ordinal_lasso_lambda_path.csv")

    coef_df.to_csv(out_dir / "ordinal_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "ordinal_lasso_summary.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(out_dir / "ordinal_lasso_selected_features.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(out_dir / "ordinal_lasso_cv_curve.csv", index=False, encoding="utf-8-sig")
    path_df.to_csv(out_dir / "ordinal_lasso_coefficient_path.csv", index=False, encoding="utf-8-sig")
    lambda_path_df.to_csv(out_dir / "ordinal_lasso_lambda_path.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log(cv_df["lambda"]), cv_df["loglik_loss"], marker="o", color="#1f77b4")
    lambda_min = float(summary_df["lambda_min"].iloc[0])
    lambda_1se = float(summary_df["lambda_1se"].iloc[0])
    ax.axvline(np.log(lambda_min), color="#7f7f7f", linestyle=":", label="lambda.min")
    ax.axvline(np.log(lambda_1se), color="#d62728", linestyle="--", label="lambda.1se")
    ax.set_xlabel("log(lambda)")
    ax.set_ylabel("CV log-likelihood loss")
    ax.set_title("Ordinal LASSO CV curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ordinal_lasso_cv_curve.png", dpi=300)
    plt.close(fig)

    lambda_vals = lambda_path_df["lambda"].astype(float).to_numpy()
    path_long = path_df.melt(id_vars="lambda_index", var_name="term", value_name="coef")
    path_long["lambda"] = path_long["lambda_index"].astype(int).map(
        lambda i: lambda_vals[i - 1] if 1 <= i <= len(lambda_vals) else np.nan
    )
    path_long = path_long.dropna(subset=["lambda"])
    path_long["abs_coef"] = path_long["coef"].abs()
    terms = [t for t in path_long["term"].unique() if not str(t).startswith("(Intercept)")]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(terms))))
    for term in terms:
        term_df = path_long[path_long["term"] == term].sort_values("lambda", ascending=False)
        ax.plot(np.log(term_df["lambda"]), term_df["coef"], linewidth=1.0, label=term)
    ax.axvline(np.log(lambda_min), color="#d62728", linestyle="--", label="lambda.min")
    ax.axvline(np.log(lambda_1se), color="#1f77b4", linestyle=":", label="lambda.1se")
    ax.set_xlabel("log(lambda)")
    ax.set_ylabel("Coefficient")
    ax.set_title("Ordinal LASSO coefficient path")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ordinal_lasso_path.png", dpi=300)
    plt.close(fig)

    plot_df = coef_df.sort_values("abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["term"], plot_df["coef"], color=np.where(plot_df["selected"] == 1, "#2ca02c", "#bdbdbd"))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient at lambda.min")
    ax.set_title("Ordinal LASSO selected coefficients")
    fig.tight_layout()
    fig.savefig(out_dir / "ordinal_lasso_coefficients.png", dpi=300)
    plt.close(fig)

    selected = selected_df["selected_feature"].astype(str).tolist()
    return {
        "selected_features": selected,
        "summary": summary_df,
        "coefficients": coef_df,
        "path": path_df,
        "cv_curve": cv_df,
    }


class OrdinalModelRFEEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=RANDOM_SEED, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        X = pd.DataFrame(np.asarray(X, dtype=float))
        X.columns = [f"x{i}" for i in range(X.shape[1])]
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array(sorted(np.unique(y)))
        self.n_features_in_ = X.shape[1]
        fit_pack = _fit_ordered_model_frame(X, y)
        self.fit_pack_ = fit_pack
        result = fit_pack["result"]
        params = pd.Series(np.asarray(result.params, dtype=float), index=result.model.exog_names)
        self.coef_ = np.abs(params.reindex(X.columns).fillna(0.0).to_numpy(dtype=float)).reshape(1, -1)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(np.asarray(X, dtype=float))
        X.columns = [f"x{i}" for i in range(X.shape[1])]
        fit_pack = self.fit_pack_
        if X.isna().any().any():
            missing = X.columns[X.isna().any()].tolist()
            raise ValueError(f"RFE prediction input has missing values without imputation: {missing}")
        X_scaled = pd.DataFrame(
            fit_pack["scaler"].transform(X),
            columns=X.columns,
            index=X.index,
        )
        probs = fit_pack["result"].model.predict(fit_pack["result"].params, exog=X_scaled)
        return np.asarray(probs, dtype=float)

    def predict(self, X):
        probs = self.predict_proba(X)
        threshold_probs = np.column_stack([probs[:, k + 1 :].sum(axis=1) for k in range(probs.shape[1] - 1)])
        return threshold_probs_to_pred_label(threshold_probs)

    def score(self, X, y):
        return cohen_kappa_score(
            np.asarray(y).astype(int),
            self.predict(X),
            weights="quadratic",
            labels=[0, 1, 2, 3, 4, 5, 6],
        )


def fit_rfe(X_train, y_train, out_dir, candidate_features=None):
    if candidate_features is None:
        candidate_features = list(X_train.columns)
    if len(candidate_features) == 1:
        pd.DataFrame(
            {
                "feature": candidate_features,
                "ranking": [1],
                "selected": [True],
            }
        ).to_csv(out_dir / "rfe_ranking.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"n_features": [1], "mean_test_score": [np.nan], "std_test_score": [np.nan]}).to_csv(
            out_dir / "rfe_curve.csv", index=False, encoding="utf-8-sig"
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(candidate_features, [1.0], color="#2ca02c")
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Selected")
        ax.set_title("RFE curve")
        fig.tight_layout()
        fig.savefig(out_dir / "rfe_curve.png", dpi=300)
        plt.close(fig)
        return {
            "model": None,
            "selected_features": candidate_features,
        }

    X_sub = X_train[candidate_features]
    if X_sub.isna().any().any():
        missing = X_sub.columns[X_sub.isna().any()].tolist()
        raise ValueError(f"RFE input has missing values without imputation: {missing}")
    X_values = X_sub.astype(float).to_numpy()
    base_estimator = OrdinalModelRFEEstimator(random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(
        estimator=base_estimator,
        step=1,
        cv=cv,
        scoring=None,
        min_features_to_select=RFE_MIN_FEATURES,
        n_jobs=1,
        importance_getter="coef_",
    )
    rfecv.fit(X_values, y_train)

    grid = pd.DataFrame(rfecv.cv_results_)
    grid.to_csv(out_dir / "rfe_curve.csv", index=False, encoding="utf-8-sig")

    if "n_features" in grid.columns:
        n_features = grid["n_features"].to_numpy()
    else:
        n_features = np.arange(RFE_MIN_FEATURES, len(candidate_features) + 1)

    best_idx = int(np.argmax(grid["mean_test_score"]))
    best_mean = float(grid["mean_test_score"].iloc[best_idx])
    best_std = float(grid["std_test_score"].iloc[best_idx])
    cv_splits = getattr(cv, "n_splits", 5)
    best_se = best_std / np.sqrt(float(cv_splits))
    if "n_features" in grid.columns:
        n_feature_series = grid["n_features"]
    else:
        n_feature_series = pd.Series(np.arange(RFE_MIN_FEATURES, len(candidate_features) + 1), index=grid.index)
    n_features_best = int(n_feature_series.iloc[best_idx])
    if RFE_MANUAL_N_FEATURES is not None:
        n_features_elbow = int(RFE_MANUAL_N_FEATURES)
        if n_features_elbow < RFE_MIN_FEATURES or n_features_elbow > len(candidate_features):
            raise ValueError(
                f"RFE_MANUAL_N_FEATURES={n_features_elbow} is outside the valid range "
                f"[{RFE_MIN_FEATURES}, {len(candidate_features)}]"
            )
    else:
        n_features_elbow = n_features_best

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_features, grid["mean_test_score"], marker="o", color="#2ca02c")
    ax.fill_between(
        n_features,
        grid["mean_test_score"] - grid["std_test_score"] / np.sqrt(float(cv_splits)),
        grid["mean_test_score"] + grid["std_test_score"] / np.sqrt(float(cv_splits)),
        color="#2ca02c",
        alpha=0.2,
        label="mean +/- SE",
    )
    selected_score_row = grid.loc[n_feature_series == n_features_elbow]
    selected_mean = float(selected_score_row["mean_test_score"].iloc[0]) if len(selected_score_row) else np.nan
    ax.axvline(n_features_best, color="#7f7f7f", linestyle=":", label=f"best={n_features_best}")
    ax.axvline(n_features_elbow, color="#d62728", linestyle="--", label=f"selected={n_features_elbow}")
    if np.isfinite(selected_mean):
        ax.scatter([n_features_elbow], [selected_mean], color="#d62728", zorder=3)
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("Quadratic weighted Kappa")
    ax.set_title("Ordinal RFE curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rfe_curve.png", dpi=300)
    plt.close(fig)

    final_rfe = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features_elbow,
        step=1,
    )
    final_rfe.fit(X_values, y_train)

    selected = [f for f, keep in zip(candidate_features, final_rfe.support_) if keep]
    ranking_df = pd.DataFrame({
        "feature": candidate_features,
        "ranking": final_rfe.ranking_,
        "selected": final_rfe.support_,
    }).sort_values(["ranking", "feature"])
    ranking_df.to_csv(out_dir / "rfe_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{
            "best_mean_test_score": best_mean,
            "best_std_test_score": best_std,
            "best_se_test_score": best_se,
            "best_n_features": n_features_best,
            "manual_n_features": RFE_MANUAL_N_FEATURES,
            "selected_n_elbow": n_features_elbow,
            "selected_features_elbow": ";".join(selected),
            "selected_mean_test_score": selected_mean,
            "selection_rule": (
                "manual elbow feature count from RFECV curve"
                if RFE_MANUAL_N_FEATURES is not None
                else "feature count at the maximum mean CV quadratic weighted Kappa on the RFE curve"
            ),
        }]
    ).to_csv(out_dir / "rfe_elbow_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "model": final_rfe,
        "selected_features": selected,
        "rfecv": rfecv,
    }


def choose_final_features(lasso_selected, rfe_selected, vif_features):
    if rfe_selected:
        return list(rfe_selected)
    if vif_features:
        return list(vif_features)
    if lasso_selected:
        return list(lasso_selected)
    raise ValueError("No clinical features are available")


def fit_ordinal_model(X_train, y_train, features):
    X_model = X_train[features].astype(float).copy()
    if X_model.isna().any().any():
        missing = X_model.columns[X_model.isna().any()].tolist()
        raise ValueError(f"Clinical model input has missing values without imputation: {missing}")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_model),
        columns=features,
        index=X_train.index,
    )
    model = OrderedModel(y_train, X_scaled, distr="logit")
    result = model.fit(method="bfgs", disp=False, maxiter=1000)
    return {"scaler": scaler, "model": model, "result": result, "features": features}


def predict_ordinal_probs(fit_pack, df):
    features = fit_pack["features"]
    scaler = fit_pack["scaler"]
    result = fit_pack["result"]
    X = df[features].astype(float).copy()
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Clinical prediction input has missing values without imputation: {missing}")
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=features,
        index=df.index,
    )
    probs = result.model.predict(result.params, exog=X_scaled)
    probs = np.asarray(probs, dtype=float)
    return probs


def ordinal_prob_to_outputs(class_probs):
    class_probs = np.asarray(class_probs, dtype=float)
    if class_probs.ndim == 1:
        class_probs = class_probs.reshape(1, -1)
    threshold_probs = np.column_stack(
        [class_probs[:, k + 1 :].sum(axis=1) for k in range(class_probs.shape[1] - 1)]
    )
    pred_label = threshold_probs_to_pred_label(threshold_probs)
    return {
        "class_probs": class_probs,
        "threshold_probs": threshold_probs,
        "pred_label": pred_label,
        "prob_poor_outcome": threshold_probs[:, 2],
        "binary_pred": (pred_label >= 3).astype(int),
    }


def build_prediction_frame(df, class_probs, prefix):
    outputs = ordinal_prob_to_outputs(class_probs)
    out = pd.DataFrame({
        "patient_id": df["patient_id"].astype(str).values,
        "true_label": df["mRS"].astype(int).values,
        f"{prefix}_pred_label": outputs["pred_label"],
        f"{prefix}_prob_poor_outcome": outputs["prob_poor_outcome"],
    })
    out[f"{prefix}_binary_true"] = (out["true_label"] >= 3).astype(int)
    out[f"{prefix}_binary_pred"] = outputs["binary_pred"]
    for i in range(class_probs.shape[1] - 1):
        out[f"{prefix}_prob_mrs_gt_{i}"] = outputs["threshold_probs"][:, i]
    return out, outputs


def merge_by_patient(df_left, df_right):
    merged = df_left.merge(df_right, on="patient_id", how="inner", suffixes=("_left", "_right"))
    if len(merged) == 0:
        raise ValueError("The two data sources have no matched patients")
    return merged


def fit_stackers(meta_df):
    stackers = {}
    for k in range(6):
        target = (meta_df["true_label"] > k).astype(int).values
        x = meta_df[[f"dl_prob_mrs_gt_{k}", f"clinical_prob_mrs_gt_{k}"]].astype(float).values
        clf = LogisticRegression(
            C=0.1,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        clf.fit(x, target)
        stackers[k] = clf
    return stackers


def predict_stack(stackers, dl_probs, clinical_probs):
    if dl_probs.shape != clinical_probs.shape:
        raise ValueError("DL and clinical probability matrices have different shapes")
    fused_threshold_probs = []
    for k in range(6):
        x = np.column_stack([dl_probs[:, k], clinical_probs[:, k]])
        p = stackers[k].predict_proba(x)[:, 1]
        fused_threshold_probs.append(p)
    fused_threshold_probs = np.column_stack(fused_threshold_probs)
    pred_label = threshold_probs_to_pred_label(fused_threshold_probs)
    return {
        "threshold_probs": fused_threshold_probs,
        "pred_label": pred_label,
        "prob_poor_outcome": fused_threshold_probs[:, 2],
        "binary_pred": (pred_label >= 3).astype(int),
    }


def save_stacking_feature_importance(out_dir, stackers):
    feature_names = ["DL threshold probability", "Clinical threshold probability"]
    rows = []
    coef_matrix = []
    for k in range(6):
        coef = stackers[k].coef_.reshape(-1)
        coef_matrix.append(coef)
        for feature_name, value in zip(feature_names, coef):
            rows.append({
                "threshold": f"mRS>{k}",
                "feature": feature_name,
                "coefficient": float(value),
                "abs_coefficient": float(abs(value)),
            })

    importance_df = pd.DataFrame(rows)
    importance_df.to_csv(out_dir / "stacking_feature_importance.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame(
        coef_matrix,
        index=[f"mRS>{k}" for k in range(6)],
        columns=feature_names,
    )
    fig_width = max(8.0, 0.8 * len(feature_names))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    im = ax.imshow(coef_df.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(coef_df.index)
    ax.set_title("Stacking Meta-model Coefficients")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Ordinal threshold")
    fig.colorbar(im, ax=ax, label="Coefficient")
    fig.tight_layout()
    fig.savefig(out_dir / "stacking_feature_importance_heatmap.png", dpi=300)
    plt.close(fig)

    mean_abs = importance_df.groupby("feature", as_index=False)["abs_coefficient"].mean()
    mean_abs = mean_abs.sort_values("abs_coefficient", ascending=True)
    fig_height = max(4.0, 0.35 * len(mean_abs))
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.barh(mean_abs["feature"], mean_abs["abs_coefficient"], color="#4e79a7")
    ax.set_xlabel("Mean absolute coefficient across thresholds")
    ax.set_title("Stacking Feature Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "stacking_feature_importance_bar.png", dpi=300)
    plt.close(fig)


def save_clinical_model_feature_importance(out_dir, fit_pack):
    features = fit_pack["features"]
    result = fit_pack["result"]
    coef_series = pd.Series(np.asarray(result.params, dtype=float), index=result.model.exog_names)
    feature_coefs = coef_series.reindex(features).fillna(0.0)
    importance_df = pd.DataFrame({
        "feature": features,
        "coefficient": feature_coefs.to_numpy(dtype=float),
    })
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False)
    importance_df.to_csv(out_dir / "clinical_model_feature_importance.csv", index=False, encoding="utf-8-sig")

    plot_df = importance_df.sort_values("abs_coefficient", ascending=True)
    fig_height = max(4.0, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    colors = ["#4e79a7" if value >= 0 else "#e15759" for value in plot_df["coefficient"]]
    ax.barh(plot_df["feature"], plot_df["abs_coefficient"], color=colors)
    ax.set_xlabel("Absolute standardized coefficient")
    ax.set_title("Clinical Ordinal Model Feature Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "clinical_model_feature_importance_bar.png", dpi=300)
    plt.close(fig)


def save_feature_selection_outputs(out_dir, lasso_pack, rfe_pack, final_features):
    pd.DataFrame({"selected_feature": final_features}).to_csv(
        out_dir / "selected_clinical_features.csv", index=False, encoding="utf-8-sig"
    )
    summary = pd.DataFrame(
        [
            {
                "ordinal_lasso_selected_n": len(lasso_pack["selected_features"]),
                "final_selected_n": len(final_features),
                "final_selected_features": ";".join(final_features),
                "ordinal_lasso_selected_features": ";".join(lasso_pack["selected_features"]),
                "rfe_selected_n": len(rfe_pack["selected_features"]),
                "rfe_selected_features": ";".join(rfe_pack["selected_features"]),
                "ordinal_lasso_alpha": ORDINAL_LASSO_ALPHA,
                "ordinal_lasso_lambda_rule": "lambda.min",
                "vif_threshold": VIF_THRESHOLD,
                "selection_method": "VIF + ordinalNet ordinal LASSO + RFE",
            }
        ]
    )
    summary.to_csv(out_dir / "feature_selection_summary.csv", index=False, encoding="utf-8-sig")


def save_trained_models(out_dir, ordinal_pack, stackers, final_features):
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    clinical_model = {
        "model_type": "statsmodels_ordered_logit",
        "features": list(final_features),
        "scaler": ordinal_pack["scaler"],
        "model": ordinal_pack["model"],
        "result": ordinal_pack["result"],
    }
    fusion_model = {
        "model_type": "threshold_level_logistic_stacking",
        "thresholds": [f"mRS>{k}" for k in range(6)],
        "input_features": ["dl_threshold_probability", "clinical_threshold_probability"],
        "stackers": stackers,
    }
    pipeline = {
        "clinical_model": clinical_model,
        "fusion_model": fusion_model,
        "clinical_columns": list(CLINICAL_COLS),
        "selected_features": list(final_features),
        "vif_threshold": VIF_THRESHOLD,
        "vif_filtering": "iterative",
        "ordinal_lasso_alpha": ORDINAL_LASSO_ALPHA,
        "ordinal_lasso_n_folds": ORDINAL_LASSO_N_FOLDS,
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(clinical_model, model_dir / "clinical_ordinal_model.joblib")
    joblib.dump(fusion_model, model_dir / "fusion_stacking_model.joblib")
    joblib.dump(pipeline, model_dir / "clinical_fusion_pipeline.joblib")

    pd.DataFrame(
        [
            {
                "file": "clinical_ordinal_model.joblib",
                "description": "Clinical ordered-logit model with scaler and selected features",
            },
            {
                "file": "fusion_stacking_model.joblib",
                "description": "Six threshold-level logistic stacking models",
            },
            {
                "file": "clinical_fusion_pipeline.joblib",
                "description": "Complete clinical and fusion inference package",
            },
        ]
    ).to_csv(model_dir / "model_files.csv", index=False, encoding="utf-8-sig")

    return model_dir


def summarize_dataset_characteristics(df, dataset_name):
    row = {"dataset": dataset_name, "n": int(len(df))}
    numeric_cols = [c for c in CLINICAL_COLS if c in df.columns]
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna().astype(float)
        if len(valid) == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_sd"] = np.nan
            row[f"{col}_median"] = np.nan
            row[f"{col}_q1"] = np.nan
            row[f"{col}_q3"] = np.nan
            continue
        row[f"{col}_mean"] = float(valid.mean())
        row[f"{col}_sd"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
        row[f"{col}_median"] = float(valid.median())
        row[f"{col}_q1"] = float(valid.quantile(0.25))
        row[f"{col}_q3"] = float(valid.quantile(0.75))
        if col in {"Male", "Acute_hydrocephalus", "Posterior_circulation", "Hypertension", "Clipping"}:
            row[f"{col}_n"] = int((valid > 0).sum())
            row[f"{col}_pct"] = float((valid > 0).mean() * 100.0)
    row["mrs_median"] = float(pd.to_numeric(df["mRS"], errors="coerce").median())
    row["mrs_q1"] = float(pd.to_numeric(df["mRS"], errors="coerce").quantile(0.25))
    row["mrs_q3"] = float(pd.to_numeric(df["mRS"], errors="coerce").quantile(0.75))
    row["poor_outcome_n"] = int((pd.to_numeric(df["mRS"], errors="coerce") >= 3).sum())
    row["poor_outcome_pct"] = float((pd.to_numeric(df["mRS"], errors="coerce") >= 3).mean() * 100.0)
    return row


def save_paper_tables(
    out_dir,
    train_df,
    test_dfs,
    vif_df,
    lasso_pack,
    rfe_pack,
    per_model_rows,
    compare_rows,
):
    dataset_rows = [summarize_dataset_characteristics(train_df, "Training")]
    for name, df in test_dfs.items():
        dataset_rows.append(summarize_dataset_characteristics(df, name))
    pd.DataFrame(dataset_rows).to_csv(out_dir / "paper_table_dataset_characteristics.csv", index=False, encoding="utf-8-sig")

    vif_df.to_csv(out_dir / "paper_table_vif_final.csv", index=False, encoding="utf-8-sig")
    lasso_pack["coefficients"].to_csv(out_dir / "paper_table_ordinal_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
    lasso_pack["summary"].to_csv(out_dir / "paper_table_ordinal_lasso_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"selected_feature": rfe_pack["selected_features"]}).to_csv(
        out_dir / "paper_table_rfe_selected_features.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(per_model_rows).to_csv(out_dir / "paper_table_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(compare_rows).to_csv(out_dir / "paper_table_fused_vs_dl.csv", index=False, encoding="utf-8-sig")


def load_dl_predictions(path):
    df = pd.read_csv(path)
    required = {
        "patient_id",
        "true_label",
        "pred_label",
        "prob_poor_outcome",
        "prob_mrs_gt_0",
        "prob_mrs_gt_1",
        "prob_mrs_gt_2",
        "prob_mrs_gt_3",
        "prob_mrs_gt_4",
        "prob_mrs_gt_5",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["true_label"] = df["true_label"].astype(int)
    threshold_probs = np.column_stack([df[f"prob_mrs_gt_{k}"].astype(float).values for k in range(6)])
    pred_label = threshold_probs_to_pred_label(threshold_probs)
    df["pred_label"] = pred_label
    df["binary_pred"] = (pred_label >= 3).astype(int)
    df["prob_poor_outcome"] = threshold_probs[:, 2]
    return df


def evaluate_pair(y_true, dl_pred, fused_pred, dataset_name, out_dir):
    rows = []
    rows.append(
        {
            "dataset": dataset_name,
            "model": "DL",
            "kappa": metric_kappa(y_true, dl_pred["pred_label"]),
            "mae": metric_mae(y_true, dl_pred["pred_label"]),
            "auc": metric_auc((np.asarray(y_true) >= 3).astype(int), dl_pred["prob_poor_outcome"]),
        }
    )
    rows.append(
        {
            "dataset": dataset_name,
            "model": "Fused",
            "kappa": metric_kappa(y_true, fused_pred["pred_label"]),
            "mae": metric_mae(y_true, fused_pred["pred_label"]),
            "auc": metric_auc((np.asarray(y_true) >= 3).astype(int), fused_pred["prob_poor_outcome"]),
        }
    )
    paired = [
        {
            "dataset": dataset_name,
            "metric": "Kappa",
            "dl": rows[0]["kappa"],
            "fused": rows[1]["kappa"],
            "diff_fused_minus_dl": rows[1]["kappa"] - rows[0]["kappa"],
            "ci95_low": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_kappa, BOOTSTRAP_N, RANDOM_SEED
            )[1],
            "ci95_high": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_kappa, BOOTSTRAP_N, RANDOM_SEED
            )[2],
            "p_value": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_kappa, BOOTSTRAP_N, RANDOM_SEED
            )[3],
            "test_method": f"paired bootstrap, n={BOOTSTRAP_N}",
        },
        {
            "dataset": dataset_name,
            "metric": "MAE",
            "dl": rows[0]["mae"],
            "fused": rows[1]["mae"],
            "diff_fused_minus_dl": rows[1]["mae"] - rows[0]["mae"],
            "ci95_low": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_mae, BOOTSTRAP_N, RANDOM_SEED + 1
            )[1],
            "ci95_high": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_mae, BOOTSTRAP_N, RANDOM_SEED + 1
            )[2],
            "p_value": paired_bootstrap_diff(
                y_true, fused_pred["pred_label"], dl_pred["pred_label"], metric_mae, BOOTSTRAP_N, RANDOM_SEED + 1
            )[3],
            "test_method": f"paired bootstrap, n={BOOTSTRAP_N}",
        },
        {
            "dataset": dataset_name,
            "metric": "AUC",
            "dl": rows[0]["auc"],
            "fused": rows[1]["auc"],
            "diff_fused_minus_dl": rows[1]["auc"] - rows[0]["auc"],
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": delong_roc_test(
                (np.asarray(y_true) >= 3).astype(int),
                fused_pred["prob_poor_outcome"],
                dl_pred["prob_poor_outcome"],
            )[1],
            "test_method": "DeLong test",
        },
    ]
    pd.DataFrame(rows).to_csv(
        out_dir / f"{dataset_name}_metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    return rows, paired


def append_model_comparisons(compare_rows, dataset_name, y_true, per_model, model_predictions):
    comparison_pairs = [
        ("DL", "Fused", "Fusion vs DL", "diff_fused_minus_dl"),
        ("DL", "Clinical", "Clinical vs DL", "diff_clinical_minus_dl"),
        ("Clinical", "Fused", "Fusion vs Clinical", "diff_fused_minus_clinical"),
    ]
    metric_specs = [
        ("Kappa", "kappa", "pred_label", metric_kappa, "paired bootstrap", RANDOM_SEED),
        ("MAE", "mae", "pred_label", metric_mae, "paired bootstrap", RANDOM_SEED + 1),
        ("AUC", "auc", "prob_poor_outcome", None, "DeLong test", RANDOM_SEED + 2),
    ]
    y_binary = (np.asarray(y_true) >= 3).astype(int)

    for model_a, model_b, comparison_name, legacy_diff_col in comparison_pairs:
        if model_a not in model_predictions or model_b not in model_predictions:
            continue
        if model_a not in set(per_model["model"]) or model_b not in set(per_model["model"]):
            continue
        pred_a = model_predictions[model_a]
        pred_b = model_predictions[model_b]
        row_a = per_model.loc[per_model["model"] == model_a].iloc[0]
        row_b = per_model.loc[per_model["model"] == model_b].iloc[0]

        for metric_name, metric_col, pred_key, metric_fn, test_label, seed in metric_specs:
            value_a = row_a[metric_col]
            value_b = row_b[metric_col]
            diff = value_b - value_a
            ci_low = np.nan
            ci_high = np.nan
            p_value = np.nan

            if metric_name in {"Kappa", "MAE"}:
                boot = paired_bootstrap_diff(
                    y_true,
                    pred_b[pred_key],
                    pred_a[pred_key],
                    metric_fn,
                    BOOTSTRAP_N,
                    seed,
                )
                ci_low, ci_high, p_value = boot[1], boot[2], boot[3]
                test_method = f"{test_label}, n={BOOTSTRAP_N}"
            else:
                _, p_value = delong_roc_test(
                    y_binary,
                    pred_b[pred_key],
                    pred_a[pred_key],
                )
                test_method = test_label

            row = {
                "dataset": dataset_name,
                "comparison": comparison_name,
                "metric": metric_name,
                "model_a": model_a,
                "model_b": model_b,
                "value_model_a": value_a,
                "value_model_b": value_b,
                "diff_model_b_minus_model_a": diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": p_value,
                "test_method": test_method,
                "dl": row_a[metric_col] if model_a == "DL" else row_b[metric_col] if model_b == "DL" else np.nan,
                "clinical": row_a[metric_col] if model_a == "Clinical" else row_b[metric_col] if model_b == "Clinical" else np.nan,
                "fused": row_a[metric_col] if model_a == "Fused" else row_b[metric_col] if model_b == "Fused" else np.nan,
                "diff_fused_minus_dl": diff if legacy_diff_col == "diff_fused_minus_dl" else np.nan,
                "diff_clinical_minus_dl": diff if legacy_diff_col == "diff_clinical_minus_dl" else np.nan,
                "diff_fused_minus_clinical": diff if legacy_diff_col == "diff_fused_minus_clinical" else np.nan,
            }
            compare_rows.append(row)


def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train_data()
    dl_train = load_dl_predictions(DL_TRAIN_PRED)
    dl_val = load_dl_predictions(DL_VAL_PRED)

    dl_train_ids = set(dl_train["patient_id"].astype(str))
    dl_val_ids = set(dl_val["patient_id"].astype(str))
    clinical_train = train[train["patient_id"].astype(str).isin(dl_train_ids)].copy().reset_index(drop=True)
    clinical_val = train[train["patient_id"].astype(str).isin(dl_val_ids)].copy().reset_index(drop=True)
    if clinical_train.empty:
        raise ValueError("DL_TRAIN_PRED has no matched patients in the clinical training CSV files")
    if clinical_val.empty:
        raise ValueError("DL_VAL_PRED has no matched patients in the clinical training CSV files")

    split_audit = pd.DataFrame(
        [
            {
                "source": "all_development_csv",
                "n": len(train),
            },
            {
                "source": "clinical_base_train_matched_to_dl_train",
                "n": len(clinical_train),
            },
            {
                "source": "stacking_calibration_matched_to_dl_val",
                "n": len(clinical_val),
            },
        ]
    )
    split_audit.to_csv(OUTPUT_DIR / "development_split_audit.csv", index=False, encoding="utf-8-sig")
    print(
        f"[Info] Clinical base-model cases: {len(clinical_train)}; "
        f"stacking calibration cases: {len(clinical_val)}."
    )

    train_y = clinical_train["mRS"].values

    X_train = prepare_xy(clinical_train, CLINICAL_COLS)
    X_train = X_train.astype(float)

    vif_features, vif_df = iterative_vif_filter(
        X_train,
        VIF_THRESHOLD,
        OUTPUT_DIR,
    )
    X_vif = X_train[vif_features].copy()

    lasso_pack = fit_ordinal_lasso_selection(X_vif, train_y, OUTPUT_DIR)
    rfe_pack = fit_rfe(X_vif, train_y, OUTPUT_DIR, candidate_features=lasso_pack["selected_features"])
    final_features = choose_final_features(lasso_pack["selected_features"], rfe_pack["selected_features"], vif_features)
    save_feature_selection_outputs(OUTPUT_DIR, lasso_pack, rfe_pack, final_features)
    pd.DataFrame({
        "stacking_input": ["dl_threshold_probability", "clinical_threshold_probability"],
    }).to_csv(OUTPUT_DIR / "stacking_meta_model_inputs.csv", index=False, encoding="utf-8-sig")

    ordinal_pack = fit_ordinal_model(clinical_train, train_y, final_features)
    save_clinical_model_feature_importance(OUTPUT_DIR, ordinal_pack)
    train_class_probs = predict_ordinal_probs(ordinal_pack, clinical_train)
    train_pred_df, train_outputs = build_prediction_frame(clinical_train, train_class_probs, "clinical")
    train_pred_df.to_csv(OUTPUT_DIR / "clinical_train_predictions.csv", index=False, encoding="utf-8-sig")

    val_class_probs = predict_ordinal_probs(ordinal_pack, clinical_val)
    val_pred_df, val_outputs = build_prediction_frame(clinical_val, val_class_probs, "clinical")
    val_pred_df.to_csv(OUTPUT_DIR / "clinical_val_predictions_for_stacking.csv", index=False, encoding="utf-8-sig")

    val_stack_source = merge_by_patient(
        dl_val[["patient_id", "true_label"] + [f"prob_mrs_gt_{k}" for k in range(6)]].rename(
            columns={f"prob_mrs_gt_{k}": f"dl_prob_mrs_gt_{k}" for k in range(6)}
        ),
        val_pred_df[["patient_id", "true_label"] + [f"clinical_prob_mrs_gt_{k}" for k in range(6)]].rename(
            columns={f"clinical_prob_mrs_gt_{k}": f"clinical_prob_mrs_gt_{k}" for k in range(6)}
        ),
    )
    val_stack_source["true_label"] = val_stack_source["true_label_right"].astype(int)
    val_stack_source["label_mismatch"] = (
        val_stack_source["true_label_left"].astype(int) != val_stack_source["true_label_right"].astype(int)
    )
    mismatch_rate = val_stack_source["label_mismatch"].mean()
    val_stack_source.to_csv(
        OUTPUT_DIR / "val_stacking_alignment_audit.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[Info] Validation DL-label mismatch rate: {mismatch_rate:.3f}. CSV mRS labels are used for stacking.")
    train_stackers = fit_stackers(
        val_stack_source[
            ["true_label"]
            + [f"dl_prob_mrs_gt_{k}" for k in range(6)]
            + [f"clinical_prob_mrs_gt_{k}" for k in range(6)]
        ]
    )
    save_stacking_feature_importance(OUTPUT_DIR, train_stackers)
    model_dir = save_trained_models(OUTPUT_DIR, ordinal_pack, train_stackers, final_features)

    clinical_metric_rows = []
    per_model_rows = []
    compare_rows = []
    loaded_test_dfs = {}

    for dataset_name, cfg in TEST_SETS.items():
        test = load_test_data(cfg["feature_files"])
        loaded_test_dfs[dataset_name] = test
        test_X = test[CLINICAL_COLS].copy().astype(float)
        test_class_probs = predict_ordinal_probs(ordinal_pack, test)
        clinical_pred_df, clinical_outputs = build_prediction_frame(test, test_class_probs, "clinical")
        dl_df = load_dl_predictions(cfg["dl_predictions"])
        merged = merge_by_patient(
            dl_df,
            clinical_pred_df,
        )
        y_true = merged["true_label_right"].astype(int).values

        dl_probs = np.column_stack([merged[f"prob_mrs_gt_{k}"].astype(float).values for k in range(6)])
        clinical_probs = np.column_stack([merged[f"clinical_prob_mrs_gt_{k}"].astype(float).values for k in range(6)])

        fused_outputs = predict_stack(train_stackers, dl_probs, clinical_probs)

        clinical_model_only = {
            "pred_label": merged["clinical_pred_label"].astype(int).values,
            "prob_poor_outcome": merged["clinical_prob_poor_outcome"].astype(float).values,
        }
        dl_model_only = {
            "pred_label": merged["pred_label"].astype(int).values,
            "prob_poor_outcome": merged["prob_poor_outcome"].astype(float).values,
        }
        fused_pred = {
            "pred_label": fused_outputs["pred_label"],
            "prob_poor_outcome": fused_outputs["prob_poor_outcome"],
        }

        per_model = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "model": "DL",
                    "kappa": metric_kappa(y_true, dl_model_only["pred_label"]),
                    "mae": metric_mae(y_true, dl_model_only["pred_label"]),
                    "auc": metric_auc((y_true >= 3).astype(int), dl_model_only["prob_poor_outcome"]),
                },
                {
                    "dataset": dataset_name,
                    "model": "Clinical",
                    "kappa": metric_kappa(y_true, clinical_model_only["pred_label"]),
                    "mae": metric_mae(y_true, clinical_model_only["pred_label"]),
                    "auc": metric_auc((y_true >= 3).astype(int), clinical_model_only["prob_poor_outcome"]),
                },
                {
                    "dataset": dataset_name,
                    "model": "Fused",
                    "kappa": metric_kappa(y_true, fused_pred["pred_label"]),
                    "mae": metric_mae(y_true, fused_pred["pred_label"]),
                    "auc": metric_auc((y_true >= 3).astype(int), fused_pred["prob_poor_outcome"]),
                },
            ]
        )
        per_model.to_csv(OUTPUT_DIR / f"{dataset_name}_model_metrics.csv", index=False, encoding="utf-8-sig")
        per_model_rows.extend(per_model.to_dict(orient="records"))

        append_model_comparisons(
            compare_rows,
            dataset_name,
            y_true,
            per_model,
            {
                "DL": dl_model_only,
                "Clinical": clinical_model_only,
                "Fused": fused_pred,
            },
        )

        out_df = pd.DataFrame({
            "patient_id": merged["patient_id"].values,
            "true_label": y_true,
            "dl_pred_label": dl_model_only["pred_label"],
            "dl_prob_poor_outcome": dl_model_only["prob_poor_outcome"],
            "clinical_pred_label": clinical_model_only["pred_label"],
            "clinical_prob_poor_outcome": clinical_model_only["prob_poor_outcome"],
            "fused_pred_label": fused_pred["pred_label"],
            "fused_prob_poor_outcome": fused_pred["prob_poor_outcome"],
        })
        for k in range(6):
            out_df[f"dl_prob_mrs_gt_{k}"] = dl_probs[:, k]
            out_df[f"clinical_prob_mrs_gt_{k}"] = clinical_probs[:, k]
            out_df[f"fused_prob_mrs_gt_{k}"] = fused_outputs["threshold_probs"][:, k]
        out_df.to_csv(OUTPUT_DIR / f"{dataset_name}_stacking_predictions.csv", index=False, encoding="utf-8-sig")

        clinical_metric_rows.append(
            {
                "dataset": dataset_name,
                "model": "Clinical",
                "kappa": metric_kappa(y_true, clinical_model_only["pred_label"]),
                "mae": metric_mae(y_true, clinical_model_only["pred_label"]),
                "auc": metric_auc((y_true >= 3).astype(int), clinical_model_only["prob_poor_outcome"]),
            }
        )

    comparison_df = pd.DataFrame(compare_rows)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUTPUT_DIR / "fused_vs_dl_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clinical_metric_rows).to_csv(OUTPUT_DIR / "clinical_model_metrics.csv", index=False, encoding="utf-8-sig")
    save_paper_tables(
        OUTPUT_DIR,
        train,
        loaded_test_dfs,
        vif_df,
        lasso_pack,
        rfe_pack,
        per_model_rows,
        compare_rows,
    )

    pd.DataFrame({
        "patient_id": clinical_train["patient_id"],
        "true_label": clinical_train["mRS"],
        "clinical_pred_label": train_pred_df["clinical_pred_label"],
        "clinical_prob_poor_outcome": train_pred_df["clinical_prob_poor_outcome"],
    }).to_csv(OUTPUT_DIR / "clinical_train_predictions_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [{
            "vif_threshold": VIF_THRESHOLD,
            "vif_filtering": "iterative",
            "vif_selected_n": len(vif_features),
            "vif_selected_features": ";".join(vif_features),
            "initial_clinical_n": len(CLINICAL_COLS),
        }]
    ).to_csv(OUTPUT_DIR / "vif_selection_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Model files saved to: {model_dir}")


def main():
    run_pipeline()


# Ordinal LASSO uses the R ordinalNet package; alpha=1 is LASSO.
ORDINAL_LASSO_ALPHA = 1.0
ORDINAL_LASSO_N_FOLDS = 5


if __name__ == "__main__":
    main()
