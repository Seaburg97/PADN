import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def format_p_value(p_value):
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def metric_kappa(y_true, pred_label):
    try:
        value = cohen_kappa_score(
            np.asarray(y_true).astype(int),
            np.asarray(pred_label).astype(int),
            labels=list(range(7)),
            weights="quadratic",
        )
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def metric_mae(y_true, pred_label):
    try:
        value = mean_absolute_error(np.asarray(y_true).astype(float), np.asarray(pred_label).astype(float))
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def metric_auc(y_binary, prob):
    y_binary = np.asarray(y_binary).astype(int)
    if len(np.unique(y_binary)) < 2:
        return np.nan
    try:
        value = roc_auc_score(y_binary, np.asarray(prob).astype(float))
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def calculate_metric(frame, metric):
    y = frame["true_label"].to_numpy()
    pred = frame["pred_label"].to_numpy()
    prob = frame["prob_poor_outcome"].to_numpy()
    if metric == "Kappa":
        return metric_kappa(y, pred)
    if metric == "MAE":
        return metric_mae(y, pred)
    return metric_auc((y >= 3).astype(int), prob)


def bootstrap_ci(frame, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    values = []
    n = len(frame)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = calculate_metric(frame.iloc[idx], metric)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return np.percentile(np.asarray(values), [2.5, 97.5])


def paired_bootstrap_p_value(y_true, pred_a, pred_b, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)

    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            diff = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
        except Exception:
            continue
        if np.isfinite(diff):
            diffs.append(diff)
    if not diffs:
        return np.nan
    diffs = np.asarray(diffs, dtype=float)
    if observed == 0:
        return 1.0
    return min(1.0, 2.0 * np.mean(np.sign(diffs) != np.sign(observed)))


def subgroup_model_metric(frame, metric, pred_column):
    y = frame["true_label"].to_numpy()
    pred = frame[pred_column].to_numpy()
    if metric == "Kappa":
        return metric_kappa(y, pred)
    if metric == "MAE":
        return metric_mae(y, pred)
    return metric_auc((y >= 3).astype(int), pred)


def calculate_positive_model_comparisons(case_df, subgroup_column):
    metric_specs = {
        "Kappa": ("pred_label", metric_kappa),
        "AUC": ("prob_poor_outcome", lambda y, pred: metric_auc((np.asarray(y) >= 3).astype(int), pred)),
    }
    comparison_specs = [
        ("Fused", "DL", "Fusion-PADN"),
    ]
    positive_df = case_df[
        (case_df["subgroup_column"] == subgroup_column) & (case_df["subgroup_value"] == 1)
    ].copy()
    model_frames = {
        model: positive_df[positive_df["model"] == model].sort_values("patient_id")
        for model in ["DL", "Clinical", "Fused"]
    }

    rows = []
    for model_a, model_b, comparison_label in comparison_specs:
        frame_a = model_frames[model_a]
        frame_b = model_frames[model_b]
        if frame_a.empty or frame_b.empty or frame_a["patient_id"].tolist() != frame_b["patient_id"].tolist():
            continue
        y_true = frame_a["true_label"].to_numpy()
        for metric_idx, (metric, (pred_column, metric_fn)) in enumerate(metric_specs.items()):
            rows.append(
                {
                    "comparison": comparison_label,
                    "metric": metric,
                    "value_a": subgroup_model_metric(frame_a, metric, pred_column),
                    "value_b": subgroup_model_metric(frame_b, metric, pred_column),
                    "p_value": paired_bootstrap_p_value(
                        y_true,
                        frame_a[pred_column].to_numpy(),
                        frame_b[pred_column].to_numpy(),
                        metric_fn,
                        n_boot=BOOTSTRAP_N,
                        seed=RANDOM_SEED + metric_idx * 53 + len(rows) * 11,
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_positive_model_comparison_box(fig, comparison_df, group_name):
    lines = []
    for comparison in ["Fusion-PADN"]:
        parts = []
        for metric in ["Kappa", "AUC"]:
            row = comparison_df[
                (comparison_df["comparison"] == comparison) & (comparison_df["metric"] == metric)
            ]
            if row.empty:
                display_metric = "QWK" if metric == "Kappa" else metric
                parts.append(f"{display_metric} p=NA")
            else:
                display_metric = "QWK" if metric == "Kappa" else metric
                parts.append(f"{display_metric} {format_p_value(row['p_value'].iloc[0])}")
        lines.append(f"{comparison}: " + ", ".join(parts))

    fig.text(
        0.5,
        -0.075,
        f"{group_name} positive model comparisons: " + " | ".join(lines),
        ha="center",
        va="top",
        fontsize=9,
        color="#2E3440",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#6F7782",
            "linewidth": 0.9,
            "alpha": 0.96,
        },
    )


def add_p_bracket(ax, x1, x2, y, text, height):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#2E3440", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + height * 1.35, text, ha="center", va="bottom", fontsize=9, color="#2E3440")


def plot_one_subgroup(summary_df, case_df, subgroup_column, output_png, output_pdf):
    plot_df = summary_df[summary_df["subgroup_column"] == subgroup_column].copy()
    if plot_df.empty:
        raise ValueError(f"No results found for subgroup_column={subgroup_column}.")

    group_name = "DCI" if subgroup_column == "DCI" else "CH"
    model_order = ["DL", "Clinical", "Fused"]
    model_labels = {"DL": "PADN", "Clinical": "Clinical", "Fused": "Fusion"}
    metric_order = ["Kappa", "AUC"]
    colors = {"positive": "#E07A70", "negative": "#5B8CC0"}
    edge_colors = {"positive": "#A8443D", "negative": "#2F5F8F"}

    metric_ylims = {
        "Kappa": (0.0, 1.12),
        "AUC": (0.5, 1.08),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.3), constrained_layout=True)
    bar_width = 0.34
    x = np.arange(len(model_order), dtype=float)

    for metric_idx, (ax, metric) in enumerate(zip(axes, metric_order)):
        metric_df = plot_df[plot_df["metric"] == metric].copy()
        display_metric = "QWK" if metric == "Kappa" else metric
        pos_values, neg_values, pos_err, neg_err, p_values = [], [], [], [], []

        for model_idx, model in enumerate(model_order):
            row = metric_df[metric_df["model"] == model].iloc[0]
            pos_value = float(row["positive_value"])
            neg_value = float(row["negative_value"])
            pos_values.append(pos_value)
            neg_values.append(neg_value)
            p_values.append(row["p_value"])

            positive_cases = case_df[
                (case_df["model"] == model)
                & (case_df["subgroup_column"] == subgroup_column)
                & (case_df["subgroup_value"] == 1)
            ]
            negative_cases = case_df[
                (case_df["model"] == model)
                & (case_df["subgroup_column"] == subgroup_column)
                & (case_df["subgroup_value"] == 0)
            ]
            pos_low, pos_high = bootstrap_ci(
                positive_cases,
                metric,
                n_boot=BOOTSTRAP_N,
                seed=RANDOM_SEED + metric_idx * 101 + model_idx * 17,
            )
            neg_low, neg_high = bootstrap_ci(
                negative_cases,
                metric,
                n_boot=BOOTSTRAP_N,
                seed=RANDOM_SEED + metric_idx * 101 + model_idx * 17 + 7,
            )
            pos_err.append([pos_value - pos_low, pos_high - pos_value])
            neg_err.append([neg_value - neg_low, neg_high - neg_value])

        pos_values = np.asarray(pos_values)
        neg_values = np.asarray(neg_values)
        pos_err = np.asarray(pos_err).T
        neg_err = np.asarray(neg_err).T

        pos_bars = ax.bar(
            x - bar_width / 2,
            pos_values,
            bar_width,
            yerr=pos_err,
            capsize=3.5,
            color=colors["positive"],
            edgecolor=edge_colors["positive"],
            linewidth=0.9,
            alpha=0.88,
            error_kw={"elinewidth": 1.05, "capthick": 1.05, "ecolor": "#333842"},
            label=f"{group_name} positive",
        )
        neg_bars = ax.bar(
            x + bar_width / 2,
            neg_values,
            bar_width,
            yerr=neg_err,
            capsize=3.5,
            color=colors["negative"],
            edgecolor=edge_colors["negative"],
            linewidth=0.9,
            alpha=0.88,
            error_kw={"elinewidth": 1.05, "capthick": 1.05, "ecolor": "#333842"},
            label=f"{group_name} negative",
        )

        y_min, y_max = metric_ylims[metric]
        ax.set_ylim(y_min, y_max)
        ax.set_title(display_metric, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([model_labels[model] for model in model_order])
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C8CDD3")
        ax.spines["bottom"].set_color("#C8CDD3")
        ax.tick_params(axis="both", colors="#30343B")

        y_range = y_max - y_min
        label_offset = y_range * 0.020
        for bars, values, errors in [(pos_bars, pos_values, pos_err), (neg_bars, neg_values, neg_err)]:
            for bar_idx, (bar, value) in enumerate(zip(bars, values)):
                label_y = max(y_min + y_range * 0.015, value - errors[0, bar_idx] - label_offset)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.3f}",
                    ha="center",
                    va="top",
                    fontsize=8.0,
                    color="#20242A",
                    clip_on=False,
                )

        for idx, p_value in enumerate(p_values):
            top = max(pos_values[idx] + pos_err[1, idx], neg_values[idx] + neg_err[1, idx])
            bracket_y = min(top + y_range * 0.045, y_max - y_range * 0.11)
            add_p_bracket(
                ax,
                x[idx] - bar_width / 2,
                x[idx] + bar_width / 2,
                bracket_y,
                format_p_value(p_value),
                y_range * 0.022,
            )

        if metric == "Kappa":
            ax.set_ylabel("Metric value")

    sample_row = plot_df.iloc[0]
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors["positive"], label=f"{group_name} positive"),
            plt.Rectangle((0, 0), 1, 1, color=colors["negative"], label=f"{group_name} negative"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.suptitle(
        f"{group_name} subgroup performance comparison  |  positive n={int(sample_row['positive_n'])}, negative n={int(sample_row['negative_n'])}",
        fontsize=16,
        fontweight="bold",
        y=1.14,
    )
    add_positive_model_comparison_box(
        fig,
        calculate_positive_model_comparisons(case_df, subgroup_column),
        group_name,
    )

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG saved: {output_png}")
    print(f"PDF saved: {output_pdf}")


def plot_group_comparison(input_csv, case_detail_csv, output_dir):
    summary_df = pd.read_csv(input_csv)
    case_df = pd.read_csv(case_detail_csv)
    required = {
        "model",
        "subgroup_column",
        "metric",
        "positive_value",
        "negative_value",
        "p_value",
    }
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {sorted(missing)}")

    output_dir = Path(output_dir)
    plot_one_subgroup(
        summary_df,
        case_df,
        "DCI",
        output_dir / "best_kappa_dci_subgroup_model_comparison_p_values.png",
        output_dir / "best_kappa_dci_subgroup_model_comparison_p_values.pdf",
    )
    plot_one_subgroup(
        summary_df,
        case_df,
        "CH",
        output_dir / "best_kappa_ch_subgroup_model_comparison_p_values.png",
        output_dir / "best_kappa_ch_subgroup_model_comparison_p_values.pdf",
    )


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "dci_ch_subgroup_best_kappa"
INPUT_CSV = OUTPUT_DIR / "best_kappa_dci_ch_model_comparison_p_values.csv"
CASE_DETAIL_CSV = OUTPUT_DIR / "best_kappa_dci_ch_case_details.csv"
BOOTSTRAP_N = 1000
RANDOM_SEED = 42


if __name__ == "__main__":
    plot_group_comparison(INPUT_CSV, CASE_DETAIL_CSV, OUTPUT_DIR)
