from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DIR_A = BASE_DIR / "data" / "ablation" / "model_a"
DIR_B = BASE_DIR / "data" / "ablation" / "model_b"

NAME_A = 'Model_A'
NAME_B = 'Model_B'

OUTPUT_CSV = BASE_DIR / "outputs" / "ablation_model_comparison.csv"

BOOTSTRAP_N = 1000
RANDOM_SEED = 42

FILES_TO_COMPARE = []


def compute_midrank(x):
    sorted_x = np.sort(x)
    order = np.argsort(x)
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

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
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
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    if observed == 0:
        p_value = 1.0
    else:
        opposite = np.mean(np.sign(diffs) != np.sign(observed))
        p_value = min(1.0, 2.0 * opposite)

    return observed, ci_low, ci_high, p_value


def metric_kappa(y_true, pred):
    return cohen_kappa_score(y_true.astype(int), pred.astype(int), weights='quadratic')


def metric_mae(y_true, pred_label):
    return mean_absolute_error(y_true.astype(float), pred_label.astype(float))


def metric_auc(y_binary, prob):
    if len(np.unique(y_binary)) < 2:
        return np.nan
    return roc_auc_score(y_binary.astype(int), prob.astype(float))


def threshold_probs_to_pred_label(threshold_probs, threshold=0.5):
    threshold_probs = np.asarray(threshold_probs, dtype=float)
    if threshold_probs.ndim == 1:
        threshold_probs = threshold_probs.reshape(1, -1)
    return (threshold_probs > threshold).sum(axis=1).astype(int)


def read_prediction_csv(path):
    df = pd.read_csv(path)
    required = {
        'patient_id',
        'true_label',
        'pred_label',
        'prob_poor_outcome',
        'prob_mrs_gt_0',
        'prob_mrs_gt_1',
        'prob_mrs_gt_2',
        'prob_mrs_gt_3',
        'prob_mrs_gt_4',
        'prob_mrs_gt_5',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    df = df.copy()
    df['patient_id'] = df['patient_id'].astype(str)
    df['true_label'] = df['true_label'].astype(int)
    threshold_probs = np.column_stack([df[f'prob_mrs_gt_{k}'].astype(float).values for k in range(6)])
    df['pred_label'] = threshold_probs_to_pred_label(threshold_probs)
    df['prob_poor_outcome'] = threshold_probs[:, 2]
    return df


def compare_one_file(filename):
    path_a = DIR_A / filename
    path_b = DIR_B / filename
    df_a = read_prediction_csv(path_a)
    df_b = read_prediction_csv(path_b)

    if df_a['patient_id'].duplicated().any():
        raise ValueError(f'{path_a} has duplicated patient_id values')
    if df_b['patient_id'].duplicated().any():
        raise ValueError(f'{path_b} has duplicated patient_id values')

    merged = df_a.merge(
        df_b,
        on='patient_id',
        suffixes=('_a', '_b'),
        how='inner',
    )
    if len(merged) == 0:
        raise ValueError(f'{filename} has no matched patients between the two directories')

    label_mismatch = (merged['true_label_a'] != merged['true_label_b']).sum()
    if label_mismatch:
        print(f'[Info] {filename}: {label_mismatch} matched patients have inconsistent true_label values. Each model uses its own labels for metric calculation.')

    matched = merged[merged['true_label_a'] == merged['true_label_b']].copy()
    if len(matched) == 0:
        print(f'[Warning] {filename}: no patients have matching true_label values; p-values cannot be computed.')

    y_a = df_a['true_label'].to_numpy()
    y_b = df_b['true_label'].to_numpy()
    y_binary_a = (y_a >= 3).astype(int)
    y_binary_b = (y_b >= 3).astype(int)

    pred_a = df_a['pred_label'].to_numpy()
    pred_b = df_b['pred_label'].to_numpy()
    prob_a = df_a['prob_poor_outcome'].to_numpy()
    prob_b = df_b['prob_poor_outcome'].to_numpy()

    kappa_a = metric_kappa(y_a, pred_a)
    kappa_b = metric_kappa(y_b, pred_b)
    mae_a = metric_mae(y_a, pred_a)
    mae_b = metric_mae(y_b, pred_b)
    auc_a = metric_auc(y_binary_a, prob_a)
    auc_b = metric_auc(y_binary_b, prob_b)

    kappa_diff = kappa_a - kappa_b
    mae_diff = mae_a - mae_b
    auc_diff = auc_a - auc_b

    if len(matched) >= 2 and len(np.unique(matched['true_label_a'])) >= 2:
        y_test = matched['true_label_a'].to_numpy()
        y_test_binary = (y_test >= 3).astype(int)

        pred_test_a = matched['pred_label_a'].to_numpy()
        pred_test_b = matched['pred_label_b'].to_numpy()
        prob_test_a = matched['prob_poor_outcome_a'].to_numpy()
        prob_test_b = matched['prob_poor_outcome_b'].to_numpy()

        _, kappa_ci_l, kappa_ci_h, kappa_p = paired_bootstrap_diff(
            y_test, pred_test_a, pred_test_b, metric_kappa, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED
        )
        _, mae_ci_l, mae_ci_h, mae_p = paired_bootstrap_diff(
            y_test, pred_test_a, pred_test_b, metric_mae, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED + 1
        )
        _, auc_p = delong_roc_test(y_test_binary, prob_test_a, prob_test_b)
    else:
        kappa_ci_l = kappa_ci_h = kappa_p = np.nan
        mae_ci_l = mae_ci_h = mae_p = np.nan
        auc_p = np.nan

    return [
        {
            'file': filename,
            'n_a': len(df_a),
            'n_b': len(df_b),
            'n_paired': len(merged),
            'n_label_mismatch': int(label_mismatch),
            'n_label_agree': int(len(matched)),
            'metric': 'Quadratic weighted Kappa',
            f'{NAME_A}': kappa_a,
            f'{NAME_B}': kappa_b,
            'diff_A_minus_B': kappa_diff,
            'ci95_low': kappa_ci_l,
            'ci95_high': kappa_ci_h,
            'p_value': kappa_p,
            'test_method': f'paired bootstrap, n={BOOTSTRAP_N}, on true_label-matched subset',
        },
        {
            'file': filename,
            'n_a': len(df_a),
            'n_b': len(df_b),
            'n_paired': len(merged),
            'n_label_mismatch': int(label_mismatch),
            'n_label_agree': int(len(matched)),
            'metric': 'MAE',
            f'{NAME_A}': mae_a,
            f'{NAME_B}': mae_b,
            'diff_A_minus_B': mae_diff,
            'ci95_low': mae_ci_l,
            'ci95_high': mae_ci_h,
            'p_value': mae_p,
            'test_method': f'paired bootstrap, n={BOOTSTRAP_N}, on true_label-matched subset',
        },
        {
            'file': filename,
            'n_a': len(df_a),
            'n_b': len(df_b),
            'n_paired': len(merged),
            'n_label_mismatch': int(label_mismatch),
            'n_label_agree': int(len(matched)),
            'metric': 'Binary AUC(mRS>=3)',
            f'{NAME_A}': auc_a,
            f'{NAME_B}': auc_b,
            'diff_A_minus_B': auc_diff,
            'ci95_low': np.nan,
            'ci95_high': np.nan,
            'p_value': auc_p,
            'test_method': 'DeLong on true_label-matched subset',
        },
    ]


def main():
    if FILES_TO_COMPARE:
        filenames = FILES_TO_COMPARE
    else:
        files_a = {p.name for p in DIR_A.glob('*_predictions_best_kappa.csv')}
        files_b = {p.name for p in DIR_B.glob('*_predictions_best_kappa.csv')}
        filenames = sorted(files_a & files_b)

    if not filenames:
        raise FileNotFoundError('No shared *_predictions_best_kappa.csv files were found in the two directories')

    all_rows = []
    for filename in filenames:
        print(f'Comparing: {filename}')
        all_rows.extend(compare_one_file(filename))

    out = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 50)
    print('\nResults:')
    print(out.to_string(index=False))
    print(f'\nSaved: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
