#!/usr/bin/env python3
"""
Estimate regional and global clinical-prior parameters from feature CSV files.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")


REGIONS_10 = [
    "ACA_Left",
    "ACA_Right",
    "MCA_Left",
    "MCA_Right",
    "PCA_Left",
    "PCA_Right",
    "Brainstem_Left",
    "Brainstem_Right",
    "Cerebellum_Left",
    "Cerebellum_Right",
]

REGIONS_5_MAPPING = {
    "ACA": ["ACA_Left", "ACA_Right"],
    "MCA": ["MCA_Left", "MCA_Right"],
    "PCA": ["PCA_Left", "PCA_Right"],
    "Brainstem": ["Brainstem_Left", "Brainstem_Right"],
    "Cerebellum": ["Cerebellum_Left", "Cerebellum_Right"],
}

GLOBAL_PRIOR_FEATURES = [
    ("pre_total_hemorrhage_volume", "Preoperative total hemorrhage volume"),
    ("post_total_hemorrhage_volume", "Postoperative total hemorrhage volume"),
]


def bootstrap_ci(y_true, y_score, metric_func, n_bootstrap=1000, alpha=0.05):
    scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        try:
            scores.append(metric_func(y_true[indices], y_score[indices]))
        except Exception:
            continue

    scores = np.array(scores)
    return (
        float(np.percentile(scores, alpha / 2 * 100)),
        float(np.percentile(scores, (1 - alpha / 2) * 100)),
    )


def calculate_metrics(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    or_value = (tp + 0.5) * (tn + 0.5) / ((fp + 0.5) * (fn + 0.5))

    auc_ci = bootstrap_ci(y_true, y_score, roc_auc_score)

    def sensitivity_at_threshold(y_t, y_s):
        y_p = (y_s >= best_threshold).astype(int)
        _, _, fn_i, tp_i = confusion_matrix(y_t, y_p).ravel()
        return tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0

    def specificity_at_threshold(y_t, y_s):
        y_p = (y_s >= best_threshold).astype(int)
        tn_i, fp_i, _, _ = confusion_matrix(y_t, y_p).ravel()
        return tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0

    sens_ci = bootstrap_ci(y_true, y_score, sensitivity_at_threshold)
    spec_ci = bootstrap_ci(y_true, y_score, specificity_at_threshold)

    return {
        "auc": float(auc),
        "auc_ci_lower": auc_ci[0],
        "auc_ci_upper": auc_ci[1],
        "threshold": float(best_threshold),
        "sensitivity": float(sensitivity),
        "sens_ci_lower": sens_ci[0],
        "sens_ci_upper": sens_ci[1],
        "specificity": float(specificity),
        "spec_ci_lower": spec_ci[0],
        "spec_ci_upper": spec_ci[1],
        "or_value": float(or_value),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def load_training_data(csv_files):
    print("Loading feature CSV files")
    data_frames = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
        df_features = pd.read_csv(csv_path, encoding="utf-8-sig")
        data_frames.append(df_features)
        missing_mrs = df_features["mRS"].isna().sum() if "mRS" in df_features.columns else "missing"
        print(f"  {csv_path.name}: rows={len(df_features)}, missing mRS={missing_mrs}")

    df_all = pd.concat(data_frames, ignore_index=True)
    if "mRS" not in df_all.columns:
        raise ValueError("Input feature CSV files must include an 'mRS' column.")

    df_all["poor_outcome"] = (df_all["mRS"] >= 3).astype(int)
    print(f"Total rows={len(df_all)}, missing mRS={df_all['mRS'].isna().sum()}\n")
    return df_all


def evaluate_single_region(df_all, col_name, region_name):
    y = df_all["poor_outcome"].values
    x = df_all[col_name].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0 or len(np.unique(y_clean)) < 2:
        print(f"{region_name}: insufficient data")
        return None

    metrics = calculate_metrics(y_clean, x_clean)
    result = {
        "region": region_name,
        "n_samples": len(x_clean),
        "mean_volume": float(x_clean.mean()),
        "std_volume": float(x_clean.std()),
        **metrics,
    }
    print(
        f"{region_name}: AUC={metrics['auc']:.3f}, "
        f"threshold={metrics['threshold']:.2f} mL, OR={metrics['or_value']:.2f}"
    )
    return result


def evaluate_continuous_feature(df_all, col_name, display_name):
    y = df_all["poor_outcome"].values
    x = df_all[col_name].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0 or len(np.unique(y_clean)) < 2:
        print(f"{display_name}: insufficient data")
        return None

    metrics = calculate_metrics(y_clean, x_clean)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_clean.reshape(-1, 1), y_clean)
    or_continuous = float(np.exp(model.coef_[0][0]))
    beta = float(np.log(or_continuous))

    binary_pred = (x_clean >= metrics["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_clean, binary_pred).ravel()
    or_binary = (tp + 0.5) * (tn + 0.5) / ((fp + 0.5) * (fn + 0.5))

    result = {
        "feature": col_name,
        "display_name": display_name,
        "n_samples": len(x_clean),
        "mean": float(x_clean.mean()),
        "std": float(x_clean.std()),
        "auc": metrics["auc"],
        "auc_ci_lower": metrics["auc_ci_lower"],
        "auc_ci_upper": metrics["auc_ci_upper"],
        "threshold_T": metrics["threshold"],
        "sensitivity": metrics["sensitivity"],
        "sens_ci_lower": metrics["sens_ci_lower"],
        "sens_ci_upper": metrics["sens_ci_upper"],
        "specificity": metrics["specificity"],
        "spec_ci_lower": metrics["spec_ci_lower"],
        "spec_ci_upper": metrics["spec_ci_upper"],
        "or_binary": float(or_binary),
        "or_continuous": or_continuous,
        "beta": beta,
    }

    print(
        f"{display_name}: AUC={result['auc']:.3f}, "
        f"threshold={result['threshold_T']:.3f}, beta={result['beta']:.4f}"
    )
    return result


def build_global_prior_outputs(df_all):
    df_all = df_all.copy()
    if "poor_outcome" not in df_all.columns:
        df_all["poor_outcome"] = (df_all["mRS"] >= 3).astype(int)

    results = []
    for col_name, display_name in GLOBAL_PRIOR_FEATURES:
        if col_name not in df_all.columns:
            print(f"Warning: missing column {col_name}")
            continue
        result = evaluate_continuous_feature(df_all, col_name, display_name)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


def build_phase_outputs(df_all, phase):
    volume_prefix = f"{phase}_SAH"
    print("\n" + "=" * 80)
    print(f"Phase: {phase}")
    print("=" * 80)

    results_10 = []
    for region in REGIONS_10:
        col_name = f"{volume_prefix}_{region}_volume"
        if col_name not in df_all.columns:
            print(f"Warning: missing column {col_name}")
            continue
        result = evaluate_single_region(df_all, col_name, region)
        if result is not None:
            results_10.append(result)

    cistern_col = f"{volume_prefix}_Cistern_volume"
    if cistern_col in df_all.columns:
        result = evaluate_single_region(df_all, cistern_col, "Cistern")
        if result is not None:
            results_10.append(result)
    else:
        print(f"Warning: missing column {cistern_col}")

    df_10_regions = pd.DataFrame(results_10)

    df_phase = df_all.copy()
    results_6 = []
    for region_name, sub_regions in REGIONS_5_MAPPING.items():
        cols = [f"{volume_prefix}_{sub_region}_volume" for sub_region in sub_regions]
        missing_cols = [col for col in cols if col not in df_phase.columns]
        if missing_cols:
            print(f"Warning: missing columns for {region_name}: {missing_cols}")
            continue
        merged_col = f"{phase}_{region_name}_volume"
        df_phase[merged_col] = df_phase[cols].sum(axis=1)
        result = evaluate_single_region(df_phase, merged_col, region_name)
        if result is not None:
            results_6.append(result)

    if cistern_col in df_phase.columns:
        df_phase[f"{phase}_Cistern_volume"] = df_phase[cistern_col]
        result = evaluate_single_region(df_phase, f"{phase}_Cistern_volume", "Cistern")
        if result is not None:
            results_6.append(result)

    ivh_col = f"{phase}_IVH_volume"
    if ivh_col in df_phase.columns:
        result = evaluate_single_region(df_phase, ivh_col, "IVH")
        if result is not None:
            results_6.append(result)
    else:
        print(f"Warning: missing column {ivh_col}")

    df_6_regions = pd.DataFrame(results_6)
    if df_6_regions.empty:
        return df_10_regions, df_6_regions, pd.DataFrame()

    region_volumes = {
        region_name: df_phase[f"{phase}_{region_name}_volume"].values
        for region_name in ["ACA", "MCA", "PCA", "Brainstem", "Cerebellum", "Cistern"]
        if f"{phase}_{region_name}_volume" in df_phase.columns
    }
    if ivh_col in df_phase.columns:
        region_volumes["IVH"] = df_phase[ivh_col].values

    y = df_phase["poor_outcome"].values
    continuous_or_values = {}
    for region_name, x_raw in region_volumes.items():
        mask = ~(np.isnan(x_raw) | np.isnan(y))
        x_clean = x_raw[mask].reshape(-1, 1)
        y_clean = y[mask]
        if len(x_clean) == 0 or len(np.unique(y_clean)) < 2:
            continue
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_clean, y_clean)
        continuous_or_values[region_name] = float(np.exp(model.coef_[0][0]))

    df_6_regions["or_value_continuous"] = df_6_regions["region"].map(continuous_or_values)
    df_6_regions["beta"] = np.log(df_6_regions["or_value_continuous"])

    df_prior = df_6_regions[
        [
            "region",
            "beta",
            "threshold",
            "std_volume",
            "or_value_continuous",
            "auc",
            "sensitivity",
            "specificity",
            "auc_ci_lower",
            "auc_ci_upper",
            "sens_ci_lower",
            "sens_ci_upper",
            "spec_ci_lower",
            "spec_ci_upper",
        ]
    ].copy()
    df_prior.columns = [
        "region",
        "beta",
        "threshold_T",
        "std_SD",
        "or_continuous",
        "auc",
        "sensitivity",
        "specificity",
        "auc_ci_lower",
        "auc_ci_upper",
        "sens_ci_lower",
        "sens_ci_upper",
        "spec_ci_lower",
        "spec_ci_upper",
    ]
    return df_10_regions, df_6_regions, df_prior


def save_phase_outputs(output_dir, phase, df_10_regions, df_6_regions, df_prior):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_10 = output_dir / f"region_metrics_10_{phase}.csv"
    output_6 = output_dir / f"region_metrics_6_full_{phase}.csv"
    output_prior = output_dir / f"region_prior_params_{phase}.csv"

    df_10_regions.to_csv(output_10, index=False)
    df_6_regions.to_csv(output_6, index=False)
    df_prior.to_csv(output_prior, index=False)

    print(f"Saved: {output_10}")
    print(f"Saved: {output_6}")
    print(f"Saved: {output_prior}")


def save_global_prior_outputs(output_dir, df_global):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_global = output_dir / "global_prior_params.csv"
    df_global.to_csv(output_global, index=False)
    print(f"Saved: {output_global}")


def main(csv_files, output_dir):
    df_all = load_training_data(csv_files)
    for phase in ["pre", "post"]:
        df_10_regions, df_6_regions, df_prior = build_phase_outputs(df_all, phase)
        save_phase_outputs(output_dir, phase, df_10_regions, df_6_regions, df_prior)

    df_global = build_global_prior_outputs(df_all)
    save_global_prior_outputs(output_dir, df_global)
    print("Done.")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    CSV_FILES = [
        BASE_DIR / "outputs/features.csv",
    ]
    OUTPUT_DIR = BASE_DIR / "outputs/prior_params"

    main(CSV_FILES, OUTPUT_DIR)
