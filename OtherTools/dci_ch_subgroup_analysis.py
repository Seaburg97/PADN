from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def metric_kappa(y_true, pred_label):
    return safe_float(
        cohen_kappa_score(
            np.asarray(y_true).astype(int),
            np.asarray(pred_label).astype(int),
            labels=list(range(7)),
            weights="quadratic",
        )
    )


def metric_mae(y_true, pred_label):
    return safe_float(mean_absolute_error(np.asarray(y_true).astype(float), np.asarray(pred_label).astype(float)))


def metric_auc(y_binary, prob):
    y_binary = np.asarray(y_binary).astype(int)
    if len(np.unique(y_binary)) < 2:
        return np.nan
    return safe_float(roc_auc_score(y_binary, np.asarray(prob).astype(float)))


def independent_bootstrap_p_value(y_a, pred_a, y_b, pred_b, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_a = np.asarray(y_a)
    pred_a = np.asarray(pred_a)
    y_b = np.asarray(y_b)
    pred_b = np.asarray(pred_b)
    observed = metric_fn(y_a, pred_a) - metric_fn(y_b, pred_b)

    diffs = []
    for _ in range(n_boot):
        idx_a = rng.integers(0, len(y_a), size=len(y_a))
        idx_b = rng.integers(0, len(y_b), size=len(y_b))
        try:
            diff = metric_fn(y_a[idx_a], pred_a[idx_a]) - metric_fn(y_b[idx_b], pred_b[idx_b])
        except Exception:
            continue
        if np.isfinite(diff):
            diffs.append(diff)

    if len(diffs) == 0:
        return np.nan
    diffs = np.asarray(diffs, dtype=float)
    if observed == 0:
        return 1.0
    opposite = np.mean(np.sign(diffs) != np.sign(observed))
    return safe_float(min(1.0, 2.0 * opposite))


def load_label_table(label_files):
    rows = []
    for label_file in label_files:
        label_file = Path(label_file)
        frame = pd.read_csv(label_file, usecols=["patient_id", "mRS", "DCI", "CH"])
        frame = frame.copy()
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["label_source"] = label_file.name
        rows.append(frame)

    labels = pd.concat(rows, ignore_index=True)
    duplicated = labels["patient_id"].duplicated(keep=False)
    if duplicated.any():
        conflict_cols = ["mRS", "DCI", "CH"]
        conflict_ids = []
        for patient_id, group in labels.loc[duplicated].groupby("patient_id"):
            if group[conflict_cols].drop_duplicates().shape[0] > 1:
                conflict_ids.append(patient_id)
        if conflict_ids:
            raise ValueError(
                "The label tables contain duplicated patient_id values with inconsistent mRS/DCI/CH values. "
                f"Use more specific label files for this prediction file. Examples: {conflict_ids[:10]}"
            )
        labels = labels.drop_duplicates("patient_id", keep="first")

    return labels


def load_predictions_with_labels(prediction_file, label_files):
    prediction_file = Path(prediction_file)
    predictions = pd.read_csv(prediction_file)
    predictions = predictions.copy()
    predictions["patient_id"] = predictions["patient_id"].astype(str)

    required_columns = ["patient_id", "true_label"]
    for spec in MODEL_SPECS:
        required_columns.extend([spec["pred_col"], spec["prob_col"]])
    missing = [col for col in required_columns if col not in predictions.columns]
    if missing:
        raise ValueError(f"{prediction_file} is missing columns: {missing}")

    labels = load_label_table(label_files)
    merged = predictions.merge(labels, on="patient_id", how="left", validate="one_to_one")
    missing_label = merged["DCI"].isna() | merged["CH"].isna()
    if missing_label.any():
        missing_ids = merged.loc[missing_label, "patient_id"].head(10).tolist()
        print(
            f"[Info] {prediction_file.name}: {int(missing_label.sum())} patients have no matched DCI/CH labels "
            f"and will be excluded. Examples: {missing_ids}"
        )
        merged = merged.loc[~missing_label].copy()

    for column in ["true_label", "mRS", "DCI", "CH"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    label_mismatch = (merged["true_label"].round().astype(int) != merged["mRS"].round().astype(int)).sum()
    if label_mismatch:
        print(
            f"[Info] {prediction_file.name}: {int(label_mismatch)} true_label values differ from label-table mRS. "
            "Metrics use prediction-file true_label values; DCI/CH are used only for subgrouping."
        )

    return merged


def evaluate_model_subset(frame, dataset_name, model_name, pred_col, prob_col, subgroup_column, subgroup_value):
    subgroup_label = f"{subgroup_column}_{'positive' if subgroup_value == 1 else 'negative'}"
    subset = frame.loc[frame[subgroup_column] == subgroup_value].copy()
    if subset.empty:
        return None

    y_true = subset["true_label"].round().astype(int).to_numpy()
    pred_label = subset[pred_col].round().astype(int).to_numpy()
    prob = subset[prob_col].astype(float).to_numpy()
    y_binary = (y_true >= 3).astype(int)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "subgroup": subgroup_label,
        "subgroup_column": subgroup_column,
        "subgroup_value": subgroup_value,
        "n": len(subset),
        "positive_outcome_n": int(y_binary.sum()),
        "negative_outcome_n": int((1 - y_binary).sum()),
        "kappa": metric_kappa(y_true, pred_label),
        "mae": metric_mae(y_true, pred_label),
        "auc": metric_auc(y_binary, prob),
    }


def build_case_detail(frame, dataset_name):
    case_rows = []
    for spec in MODEL_SPECS:
        for subgroup_column in ["DCI", "CH"]:
            for subgroup_value in [1, 0]:
                subset = frame.loc[frame[subgroup_column] == subgroup_value].copy()
                if subset.empty:
                    continue
                detail = pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "model": spec["name"],
                        "subgroup": f"{subgroup_column}_{'positive' if subgroup_value == 1 else 'negative'}",
                        "subgroup_column": subgroup_column,
                        "subgroup_value": subgroup_value,
                        "patient_id": subset["patient_id"].astype(str).values,
                        "label_source": subset["label_source"].values,
                        "true_label": subset["true_label"].round().astype(int).values,
                        "pred_label": subset[spec["pred_col"]].round().astype(int).values,
                        "prob_poor_outcome": subset[spec["prob_col"]].astype(float).values,
                        "binary_true": (subset["true_label"].astype(float).values >= 3).astype(int),
                        "binary_pred": (subset[spec["pred_col"]].astype(float).values >= 3).astype(int),
                        "mRS": subset["mRS"].values,
                        "DCI": subset["DCI"].values,
                        "CH": subset["CH"].values,
                    }
                )
                detail["signed_error"] = detail["pred_label"] - detail["true_label"]
                detail["abs_error"] = detail["signed_error"].abs()
                case_rows.append(detail)

    return pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()


def evaluate_dataset(dataset_name, prediction_file, label_files):
    frame = load_predictions_with_labels(prediction_file, label_files)
    rows = []
    for spec in MODEL_SPECS:
        for subgroup_column in ["DCI", "CH"]:
            for subgroup_value in [1, 0]:
                row = evaluate_model_subset(
                    frame=frame,
                    dataset_name=dataset_name,
                    model_name=spec["name"],
                    pred_col=spec["pred_col"],
                    prob_col=spec["prob_col"],
                    subgroup_column=subgroup_column,
                    subgroup_value=subgroup_value,
                )
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows), build_case_detail(frame, dataset_name), frame


def build_prediction_dict(frame):
    predictions = {}
    for spec in MODEL_SPECS:
        predictions[spec["name"]] = {
            "pred_label": frame[spec["pred_col"]].round().astype(int).to_numpy(),
            "prob_poor_outcome": frame[spec["prob_col"]].astype(float).to_numpy(),
        }
    return predictions


def append_group_comparisons(compare_rows, dataset_name, frame):
    metric_specs = [
        ("Kappa", "kappa", "pred_label", metric_kappa, "independent bootstrap", RANDOM_SEED),
        ("MAE", "mae", "pred_label", metric_mae, "independent bootstrap", RANDOM_SEED + 1),
        ("AUC", "auc", "prob_poor_outcome", metric_auc, "independent bootstrap", RANDOM_SEED + 2),
    ]
    metric_specs[0] = ("Kappa", "kappa", "pred_label", metric_kappa, "independent bootstrap", RANDOM_SEED)
    metric_specs[1] = ("MAE", "mae", "pred_label", metric_mae, "independent bootstrap", RANDOM_SEED + 1)

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        for subgroup_column in ["DCI", "CH"]:
            positive = frame.loc[frame[subgroup_column] == 1].copy()
            negative = frame.loc[frame[subgroup_column] == 0].copy()
            if positive.empty or negative.empty:
                continue

            y_pos = positive["true_label"].round().astype(int).to_numpy()
            y_neg = negative["true_label"].round().astype(int).to_numpy()
            y_pos_binary = (y_pos >= 3).astype(int)
            y_neg_binary = (y_neg >= 3).astype(int)
            pred_pos = positive[spec["pred_col"]].round().astype(int).to_numpy()
            pred_neg = negative[spec["pred_col"]].round().astype(int).to_numpy()
            prob_pos = positive[spec["prob_col"]].astype(float).to_numpy()
            prob_neg = negative[spec["prob_col"]].astype(float).to_numpy()

            values = {
                "kappa": (metric_kappa(y_pos, pred_pos), metric_kappa(y_neg, pred_neg)),
                "mae": (metric_mae(y_pos, pred_pos), metric_mae(y_neg, pred_neg)),
                "auc": (metric_auc(y_pos_binary, prob_pos), metric_auc(y_neg_binary, prob_neg)),
            }

            for metric_name, metric_col, pred_key, metric_fn, test_label, seed in metric_specs:
                if metric_col == "auc":
                    p_value = independent_bootstrap_p_value(
                        y_pos_binary,
                        prob_pos,
                        y_neg_binary,
                        prob_neg,
                        metric_fn,
                        BOOTSTRAP_N,
                        seed,
                    )
                else:
                    p_value = independent_bootstrap_p_value(
                        y_pos,
                        pred_pos,
                        y_neg,
                        pred_neg,
                        metric_fn,
                        BOOTSTRAP_N,
                        seed,
                    )

                compare_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "subgroup_column": subgroup_column,
                        "comparison": f"{subgroup_column}_positive vs {subgroup_column}_negative",
                        "metric": metric_name,
                        "positive_n": len(positive),
                        "negative_n": len(negative),
                        "positive_value": values[metric_col][0],
                        "negative_value": values[metric_col][1],
                        "p_value": p_value,
                        "test_method": f"{test_label}, n={BOOTSTRAP_N}",
                    }
                )


def build_model_comparison_table(frames_by_dataset):
    rows = []
    for dataset_name, frame in frames_by_dataset.items():
        append_group_comparisons(rows, dataset_name, frame)
    return pd.DataFrame(rows)


def write_outputs(summary, case_detail, model_comparison, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summary.sort_values(
        ["dataset", "subgroup_column", "subgroup_value", "model"],
        ascending=[True, True, False, True],
    )

    summary_path = output_dir / "best_kappa_dci_ch_subgroup_metrics.csv"
    case_path = output_dir / "best_kappa_dci_ch_case_details.csv"
    comparison_path = output_dir / "best_kappa_dci_ch_model_comparison_p_values.csv"
    xlsx_path = output_dir / "best_kappa_dci_ch_subgroup_metrics.xlsx"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    case_detail.to_csv(case_path, index=False, encoding="utf-8-sig")
    model_comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        model_comparison.to_excel(writer, sheet_name="model_p_values", index=False)
        case_detail.to_excel(writer, sheet_name="case_detail", index=False)

    print(f"Metric table saved: {summary_path}")
    print(f"Model-comparison p-value table saved: {comparison_path}")
    print(f"Case details saved: {case_path}")
    print(f"Excel summary saved: {xlsx_path}")


def main():
    all_summary = []
    all_cases = []
    frames_by_dataset = {}
    for dataset_name, cfg in DATASETS.items():
        prediction_file = Path(cfg["prediction_file"])
        if not prediction_file.exists():
            print(f"[Skipping] {dataset_name}: prediction file does not exist: {prediction_file}")
            continue
        summary, cases, frame = evaluate_dataset(dataset_name, prediction_file, cfg["label_files"])
        all_summary.append(summary)
        all_cases.append(cases)
        frames_by_dataset[dataset_name] = frame
        print(f"[Done] {dataset_name}: {len(summary)} subgroup-model metric rows")

    if not all_summary:
        raise RuntimeError("No metrics were generated. Check the DATASETS paths.")

    write_outputs(
        summary=pd.concat(all_summary, ignore_index=True),
        case_detail=pd.concat(all_cases, ignore_index=True),
        model_comparison=build_model_comparison_table(frames_by_dataset),
        output_dir=OUTPUT_DIR,
    )


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "data" / "features"
STACKING_DIR = BASE_DIR / ".." / "ClinicalModelAndFusionModel" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs" / "dci_ch_subgroup_best_kappa"

MODEL_SPECS = [
    {"name": "DL", "pred_col": "dl_pred_label", "prob_col": "dl_prob_poor_outcome"},
    {"name": "Clinical", "pred_col": "clinical_pred_label", "prob_col": "clinical_prob_poor_outcome"},
    {"name": "Fused", "pred_col": "fused_pred_label", "prob_col": "fused_prob_poor_outcome"},
]

BOOTSTRAP_N = 1000
RANDOM_SEED = 42

DATASETS = {
    "Test-Combined": {
        "prediction_file": STACKING_DIR / "Test-Combined_stacking_predictions.csv",
        "label_files": [
            CSV_DIR / "featuresth.csv",
            CSV_DIR / "featuresay2.csv",
            CSV_DIR / "featuresefy.csv",
        ],
    },
}


if __name__ == "__main__":
    main()
