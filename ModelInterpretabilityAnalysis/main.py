from __future__ import annotations

import json
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PADN_DIR = (SCRIPT_DIR / ".." / "PADN").resolve()
if str(PADN_DIR) not in sys.path:
    sys.path.insert(0, str(PADN_DIR))
REGISTRATION_OUTPUT_DIR = (SCRIPT_DIR / ".." / "RegistrationAndSkullStripping" / "result_v3").resolve()
MODEL_FILE = PADN_DIR / "Main.py"
RESULT_DIR = PADN_DIR / "outputs" / "dl_models"
CSV_DIR = SCRIPT_DIR / "data" / "features"
CT_DATA_DIR = SCRIPT_DIR / "data" / "registered_ct"

REGION_NAMES = [
    "ACA",
    "MCA",
    "PCA",
    "Brainstem",
    "Cerebellum",
    "Cistern",
    "IVH",
]
REGION_DISPLAY_NAMES = {
    "ACA": "ACA",
    "MCA": "MCA",
    "PCA": "PCA",
    "Brainstem": "Brainstem",
    "Cerebellum": "Cerebellum",
    "Cistern": "Cistern",
    "IVH": "Ventricle",
}

DATASET_SPECS = {
    "efy": {
        "feature_csv": CSV_DIR / "featuresefy.csv",
        "pred_csv": RESULT_DIR / "external_efy_predictions_best_kappa.csv",
        "contrib_csv": RESULT_DIR / "external_efy_contributions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "efy",
        "flat_folder_mode": True,
    },
    "ay2": {
        "feature_csv": CSV_DIR / "featuresay2.csv",
        "pred_csv": RESULT_DIR / "external_ay2_predictions_best_kappa.csv",
        "contrib_csv": RESULT_DIR / "external_ay2_contributions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "ay2",
        "flat_folder_mode": True,
    },
    "th": {
        "feature_csv": CSV_DIR / "featuresth.csv",
        "pred_csv": RESULT_DIR / "external_th_predictions_best_kappa.csv",
        "contrib_csv": RESULT_DIR / "external_th_contributions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "th",
        "flat_folder_mode": True,
    },
}


DATASET_NAME = "efy"
CHECKPOINT_NAME = "best_kappa.pth"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
RUN_ALL_CASES = True
CASE_IDS: list[str] = []
GRADCAM_TARGET_MODE = "predicted"
AXIAL_SLICE_COUNT = 30
SAVE_NIFTI = False
SAVE_AXIAL_PNGS = True
SKIP_EXISTING_GRADCAM = True


def _set_plot_font():
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _load_model_module():
    spec = importlib.util.spec_from_file_location("padn_main", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model file: {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_model(config: dict, device: torch.device):
    module = _load_model_module()
    dual_channel_predictor = module.DualChannelPredictor

    model = dual_channel_predictor(
        dropout=float(config.get("dropout", 0.6)),
        use_region_attention=bool(config.get("use_region_attention", True)),
        region_masks=None,
        model_mode=config.get("model_mode", "prepost_prior_attention"),
    )

    ckpt_path = RESULT_DIR / CHECKPOINT_NAME
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected if k != "region_masks"]
    if missing:
        print(f"[Warning] Missing parameters: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected parameters: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return module, model


def _load_config():
    config_path = RESULT_DIR / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg.setdefault("dropout", 0.6)
    cfg.setdefault("use_region_attention", True)
    cfg.setdefault("model_mode", "prepost_prior_attention")
    cfg.setdefault("target_shape", (182, 218, 182))
    cfg.setdefault("patient_template_root", str(REGISTRATION_OUTPUT_DIR))
    return cfg


def _ensure_dirs():
    dirs = [
        OUTPUT_ROOT,
        OUTPUT_ROOT / "case_level",
        OUTPUT_ROOT / "group_level",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _threshold_count_pred_label(df: pd.DataFrame) -> np.ndarray:
    required = [f"prob_mrs_gt_{i}" for i in range(6)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing ordinal threshold probability columns: {missing}")
    threshold_probs = df[required].astype(float).to_numpy()
    return (threshold_probs > 0.5).sum(axis=1).astype(int)


def _required_logit_columns():
    cols = [f"image_logit_{i}" for i in range(6)]
    cols += [f"pre_prior_logit_{i}" for i in range(6)]
    cols += [f"post_prior_logit_{i}" for i in range(6)]
    return cols


def _load_data():
    spec = DATASET_SPECS[DATASET_NAME]
    pred_df = pd.read_csv(spec["pred_csv"])
    contrib_df = pd.read_csv(spec["contrib_csv"])
    feature_df = pd.read_csv(spec["feature_csv"])
    pred_cols_to_drop = [col for col in contrib_df.columns if col in pred_df.columns and col != "patient_id"]
    contrib_df = contrib_df.drop(columns=pred_cols_to_drop)
    pred_with_contrib = pred_df.merge(contrib_df, on="patient_id", how="left")
    merged = pred_with_contrib.merge(feature_df, on="patient_id", how="inner", suffixes=("", "_feat"))

    if merged.empty:
        raise ValueError("Prediction and feature CSV files have no matched patients")
    missing_logits = [col for col in _required_logit_columns() if col not in merged.columns]
    if missing_logits:
        raise ValueError(f"Contribution CSV is missing branch logit columns: {missing_logits}")

    merged["true_label"] = merged["true_label"].astype(int)
    merged["pred_label"] = _threshold_count_pred_label(merged)
    merged["prob_poor_outcome"] = merged["prob_mrs_gt_2"].astype(float)
    merged["binary_pred"] = (merged["pred_label"] >= 3).astype(int)
    merged["y"] = merged["true_label"].astype(int)
    merged["ct_data_dir"] = str(spec["ct_dir"])
    merged["flat_folder_mode"] = bool(spec["flat_folder_mode"])
    merged["center"] = merged.get("center", "")
    return merged


def _select_cases(df: pd.DataFrame) -> pd.DataFrame:
    if RUN_ALL_CASES or not CASE_IDS:
        picked = df.copy()
    else:
        picked = df[df["patient_id"].astype(str).isin([str(x) for x in CASE_IDS])].copy()
    if picked.empty:
        raise ValueError(f"CASE_IDS did not match any patients: {CASE_IDS}")
    picked["gap"] = (picked["pred_label"] - picked["true_label"]).abs()
    picked["correct"] = picked["pred_label"] == picked["true_label"]
    return picked


def _build_single_case_df(row: pd.Series) -> pd.DataFrame:
    case_df = pd.DataFrame([row.to_dict()])
    case_df["y"] = case_df["true_label"].astype(int)
    case_df["ct_data_dir"] = row["ct_data_dir"]
    case_df["flat_folder_mode"] = bool(row["flat_folder_mode"])
    return case_df


def _load_single_case(module, row: pd.Series):
    case_df = _build_single_case_df(row)
    cfg = _load_config()
    dataset = module.CTDataset(
        case_df,
        ct_data_dir=str(row["ct_data_dir"]),
        target_shape=tuple(cfg["target_shape"]),
        patient_template_root=cfg.get("patient_template_root"),
        use_region_attention=bool(cfg.get("use_region_attention", True)),
    )
    pre_ct, post_ct, label, volumes, pre_region_masks, post_region_masks = dataset[0]
    return (
        pre_ct.unsqueeze(0),
        post_ct.unsqueeze(0),
        label.unsqueeze(0),
        volumes.unsqueeze(0),
        pre_region_masks.unsqueeze(0),
        post_region_masks.unsqueeze(0),
    )


def _target_index_from_row(row: pd.Series) -> int:
    mode = GRADCAM_TARGET_MODE.lower()
    if mode == "poor":
        return 2
    if mode == "true":
        return int(min(int(row["true_label"]), 5))
    return int(min(int(row["pred_label"]), 5))


def _compute_gradcam(
    model,
    target_layer,
    pre_ct,
    post_ct,
    volumes,
    pre_region_masks,
    post_region_masks,
    target_index: int,
):
    activations = {}
    gradients = {}

    def forward_hook(_module, _input, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(
            pre_ct,
            post_ct,
            volumes=volumes,
            pre_region_masks=pre_region_masks,
            post_region_masks=post_region_masks,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs.ndim != 2:
            raise RuntimeError(f"Unexpected model output shape: {tuple(outputs.shape)}")
        score = outputs[:, target_index].sum()
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=pre_ct.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    finally:
        h1.remove()
        h2.remove()


def _normalize_volume(vol):
    v = vol.astype(np.float32)
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


def _orthogonal_indices(cam: np.ndarray):
    z, y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return int(z), int(y), int(x)


def _overlay_ax(ax, image_slice, cam_slice, title):
    ax.imshow(image_slice, cmap="gray")
    ax.imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")


def _rot_axial(volume: np.ndarray, z_idx: int):
    return np.rot90(volume[:, :, z_idx], k=1)


def _save_axial_slice_series(case_dir: Path, phase: str, ct_volume: np.ndarray, cam_volume: np.ndarray):
    out_dir = case_dir / "axial_slices" / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    # Axial/transverse slices are taken along the third image axis.
    slice_indices = np.linspace(0, ct_volume.shape[2] - 1, AXIAL_SLICE_COUNT, dtype=int)
    for order, z_idx in enumerate(slice_indices, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ct_slice = _rot_axial(ct_volume, z_idx)
        cam_slice = _rot_axial(cam_volume, z_idx)

        axes[0].imshow(ct_slice, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        _overlay_ax(axes[1], ct_slice, cam_slice, "Input + Grad-CAM")

        fig.suptitle(f"{phase.upper()} | axial slice {z_idx}", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / f"{order:02d}_slice_{z_idx:03d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _save_axial_overview(case_dir: Path, pre_img: np.ndarray, pre_cam: np.ndarray, post_img: np.ndarray, post_cam: np.ndarray, row: pd.Series):
    out = case_dir / "gradcam_3d.png"
    slice_indices = np.linspace(0, pre_img.shape[2] - 1, 6, dtype=int)
    fig, axes = plt.subplots(4, len(slice_indices), figsize=(18, 10))

    for col, z_idx in enumerate(slice_indices):
        pre_ct_slice = _rot_axial(pre_img, z_idx)
        pre_cam_slice = _rot_axial(pre_cam, z_idx)
        post_ct_slice = _rot_axial(post_img, z_idx)
        post_cam_slice = _rot_axial(post_cam, z_idx)

        axes[0, col].imshow(pre_ct_slice, cmap="gray")
        axes[0, col].set_title(f"Pre input z={z_idx}")
        axes[0, col].axis("off")

        _overlay_ax(axes[1, col], pre_ct_slice, pre_cam_slice, "Pre overlay")

        axes[2, col].imshow(post_ct_slice, cmap="gray")
        axes[2, col].set_title(f"Post input z={z_idx}")
        axes[2, col].axis("off")

        _overlay_ax(axes[3, col], post_ct_slice, post_cam_slice, "Post overlay")

    fig.suptitle(
        f"Patient {row['patient_id']} | true={int(row['true_label'])} | pred={int(row['pred_label'])}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _save_gradcam_case(row: pd.Series, model, module):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    overview_path = case_dir / "gradcam_3d.png"
    pre_slice_dir = case_dir / "axial_slices" / "pre"
    post_slice_dir = case_dir / "axial_slices" / "post"
    if (
        SKIP_EXISTING_GRADCAM
        and overview_path.exists()
        and (not SAVE_AXIAL_PNGS or (
            pre_slice_dir.exists()
            and post_slice_dir.exists()
            and len(list(pre_slice_dir.glob("*.png"))) >= AXIAL_SLICE_COUNT
            and len(list(post_slice_dir.glob("*.png"))) >= AXIAL_SLICE_COUNT
        ))
    ):
        return overview_path

    pre_ct, post_ct, _label, volumes, pre_region_masks, post_region_masks = _load_single_case(module, row)
    device = next(model.parameters()).device
    pre_ct = pre_ct.to(device)
    post_ct = post_ct.to(device)
    volumes = volumes.to(device)
    pre_region_masks = pre_region_masks.to(device)
    post_region_masks = post_region_masks.to(device)

    target_index = _target_index_from_row(row)
    pre_layer = model.dual_stream.pre_branch[-1].conv2
    post_layer = model.dual_stream.post_branch[-1].conv2
    pre_cam = _compute_gradcam(
        model,
        pre_layer,
        pre_ct,
        post_ct,
        volumes,
        pre_region_masks,
        post_region_masks,
        target_index,
    )
    post_cam = _compute_gradcam(
        model,
        post_layer,
        pre_ct,
        post_ct,
        volumes,
        pre_region_masks,
        post_region_masks,
        target_index,
    )

    pre_img = pre_ct[0, 0].detach().cpu().numpy().astype(np.float32)
    post_img = post_ct[0, 0].detach().cpu().numpy().astype(np.float32)
    pre_img = _normalize_volume(pre_img)
    post_img = _normalize_volume(post_img)

    out = _save_axial_overview(case_dir, pre_img, pre_cam, post_img, post_cam, row)

    if SAVE_NIFTI:
        nib.save(nib.Nifti1Image(pre_img.astype(np.float32), affine=np.eye(4)), case_dir / "pre_ct_input.nii.gz")
        nib.save(nib.Nifti1Image(post_img.astype(np.float32), affine=np.eye(4)), case_dir / "post_ct_input.nii.gz")
        nib.save(nib.Nifti1Image(pre_cam.astype(np.float32), affine=np.eye(4)), case_dir / "pre_gradcam.nii.gz")
        nib.save(nib.Nifti1Image(post_cam.astype(np.float32), affine=np.eye(4)), case_dir / "post_gradcam.nii.gz")

    if SAVE_AXIAL_PNGS:
        _save_axial_slice_series(case_dir, "pre", pre_img, pre_cam)
        _save_axial_slice_series(case_dir, "post", post_img, post_cam)
    return out


def _bar_pair_figure(row: pd.Series, prefix: str, ylabel: str, title: str, out_path: Path):
    values_pre = np.array([row[f"pre_{prefix}_{region}"] for region in REGION_NAMES], dtype=float)
    values_post = np.array([row[f"post_{prefix}_{region}"] for region in REGION_NAMES], dtype=float)

    x = np.arange(len(REGION_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - width / 2, values_pre, width, label="Pre-op", color="#f28e2b")
    ax.bar(x + width / 2, values_post, width, label="Post-op", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_prior_scores(row: pd.Series):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    _bar_pair_figure(
        row,
        prefix="prior_branch",
        ylabel="Prior score",
        title="Prior Scores",
        out_path=case_dir / "prior_scores.png",
    )


def _save_attention_weights(row: pd.Series):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    _bar_pair_figure(
        row,
        prefix="prior_attention_attention_weight",
        ylabel="Attention weight",
        title="Attention Weights",
        out_path=case_dir / "attention_weights.png",
    )


def _save_phase_prior_attention(row: pd.Series, phase: str):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    prior = np.array([row[f"{phase}_prior_branch_{region}"] for region in REGION_NAMES], dtype=float)
    attention = np.array(
        [row[f"{phase}_prior_attention_attention_weight_{region}"] for region in REGION_NAMES],
        dtype=float,
    )

    x = np.arange(len(REGION_NAMES))
    fig, ax_prior = plt.subplots(figsize=(10.5, 4.8))
    bars = ax_prior.bar(x, prior, color="#4e79a7", alpha=0.82, label="Prior score")
    ax_prior.set_xticks(x)
    ax_prior.set_xticklabels(REGION_NAMES, rotation=25, ha="right")
    ax_prior.set_ylabel("Prior score")
    ax_prior.set_ylim(0, max(0.1, float(prior.max()) * 1.22))
    ax_prior.set_title(f"{phase.upper()} Regional Prior Scores with Attention Weights")

    ax_attention = ax_prior.twinx()
    ax_attention.plot(
        x,
        attention,
        color="#f28e2b",
        marker="o",
        linewidth=2.0,
        label="Attention weight",
    )
    ax_attention.set_ylabel("Attention weight")
    ax_attention.set_ylim(0, max(0.1, float(attention.max()) * 1.25))

    for bar, value in zip(bars, prior):
        if value > 0:
            ax_prior.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for i, value in enumerate(attention):
        ax_attention.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color="#a34f00")

    handles_1, labels_1 = ax_prior.get_legend_handles_labels()
    handles_2, labels_2 = ax_attention.get_legend_handles_labels()
    ax_prior.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper right")

    plt.tight_layout()
    out_path = case_dir / f"{phase}_prior_attention.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_pre_prior_attention(row: pd.Series):
    return _save_phase_prior_attention(row, "pre")


def _save_post_prior_attention(row: pd.Series):
    return _save_phase_prior_attention(row, "post")


def _save_logit_contribution(row: pd.Series):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    image = np.array([row[f"image_logit_{i}"] for i in range(6)], dtype=float)
    pre = np.array([row[f"pre_prior_logit_{i}"] for i in range(6)], dtype=float)
    post = np.array([row[f"post_prior_logit_{i}"] for i in range(6)], dtype=float)
    final = image + pre + post
    image_probs = _sigmoid_np(image)
    full_probs = _sigmoid_np(final)

    x = np.arange(6)
    width = 0.22

    image_pred = _pred_label_from_logits(image)
    full_pred = _pred_label_from_logits(final)
    true_label = int(row["true_label"])

    fig1, ax = plt.subplots(figsize=(28.0, 14.0))
    ax.bar(x - width, image, width, label="Image", color="#4e79a7")
    ax.bar(x, pre, width, label="Pre-prior", color="#f28e2b")
    ax.bar(x + width, post, width, label="Post-prior", color="#59a14f")
    ax.plot(x, final, color="black", marker="o", linewidth=3.2, markersize=14, label="Final (PADN model)")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Threshold {i}" for i in range(6)], fontsize=38, rotation=18, ha="right")
    ax.set_ylabel("Logit", fontsize=46, labelpad=24)
    ax.tick_params(axis="y", labelsize=38)
    ax.tick_params(axis="x", pad=16)
    fig1.suptitle("Logit Decomposition", fontsize=60, y=0.93)
    legend1 = fig1.legend(
        handles=[
            Patch(facecolor="#4e79a7", edgecolor="none", label="Image"),
            Patch(facecolor="#f28e2b", edgecolor="none", label="Pre-prior"),
            Patch(facecolor="#59a14f", edgecolor="none", label="Post-prior"),
            plt.Line2D([0], [0], color="black", lw=4, marker="o", markersize=12, label="Final (PADN model)"),
        ],
        loc="center left",
        bbox_to_anchor=(0.715, 0.50),
        ncol=1,
        frameon=False,
        fontsize=40,
        handlelength=2.8,
        labelspacing=1.0,
        borderpad=0.8,
    )
    fig1.subplots_adjust(left=0.13, right=0.72, bottom=0.19, top=0.86)
    out_path = case_dir / "logit_contribution.png"
    fig1.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(28.0, 14.0))
    ax.plot(
        x,
        image_probs,
        color="#4e79a7",
        marker="o",
        linewidth=3.6,
        markersize=14,
        label=f"Image (pred={image_pred})",
    )
    ax.plot(
        x,
        full_probs,
        color="#444444",
        marker="o",
        linewidth=3.6,
        markersize=14,
        label=f"Final PADN model (pred={full_pred})",
    )
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=2.2, label="Decision threshold = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P(mRS>{i})" for i in range(6)], fontsize=38, rotation=18, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel("Probability", fontsize=46, labelpad=24)
    ax.tick_params(axis="y", labelsize=38)
    ax.tick_params(axis="x", pad=16)
    fig2.suptitle("Image threshold probability VS Final PADN model threshold probability", fontsize=46, y=0.93)
    legend2 = fig2.legend(
        handles=[
            plt.Line2D([0], [0], color="#4e79a7", lw=4, marker="o", markersize=12, label=f"Image (pred={image_pred})"),
            plt.Line2D([0], [0], color="#444444", lw=4, marker="o", markersize=12, label=f"Final PADN model (pred={full_pred})"),
            plt.Line2D([0], [0], color="#999999", lw=3, linestyle="--", label="Decision threshold = 0.5"),
        ],
        loc="center left",
        bbox_to_anchor=(0.715, 0.50),
        ncol=1,
        frameon=False,
        fontsize=40,
        handlelength=2.8,
        labelspacing=1.0,
        borderpad=0.8,
    )
    fig2.subplots_adjust(left=0.13, right=0.72, bottom=0.19, top=0.86)
    out_path2 = case_dir / "image_vs_final_probabilities.png"
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    return out_path


def _save_global_logit_contribution_distribution(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        for threshold in range(6):
            image_logit = float(row[f"image_logit_{threshold}"])
            pre_logit = float(row[f"pre_prior_logit_{threshold}"])
            post_logit = float(row[f"post_prior_logit_{threshold}"])
            abs_values = np.abs([image_logit, pre_logit, post_logit])
            denom = float(abs_values.sum())
            if denom <= 1e-12:
                ratios = [0.0, 0.0, 0.0]
            else:
                ratios = (abs_values / denom).tolist()

            for branch, logit, abs_logit, ratio in [
                ("Image", image_logit, abs_values[0], ratios[0]),
                ("Pre-prior", pre_logit, abs_values[1], ratios[1]),
                ("Post-prior", post_logit, abs_values[2], ratios[2]),
            ]:
                records.append(
                    {
                        "patient_id": patient_id,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "threshold": threshold,
                        "branch": branch,
                        "logit": logit,
                        "abs_logit": float(abs_logit),
                        "absolute_contribution_ratio": float(ratio),
                        "absolute_contribution_percent": float(ratio * 100.0),
                    }
                )

    contrib_df = pd.DataFrame(records)
    detail_path = out_dir / "global_logit_contribution_distribution.csv"
    contrib_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    branch_order = ["Image", "Pre-prior", "Post-prior"]
    colors = {
        "Image": "#4e79a7",
        "Pre-prior": "#f28e2b",
        "Post-prior": "#59a14f",
    }
    stats = (
        contrib_df.groupby("branch")["absolute_contribution_percent"]
        .agg(["mean", "median", "std", "min", "max"])
        .reindex(branch_order)
        .reset_index()
    )
    stats_path = out_dir / "global_logit_contribution_summary.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    box_data = [
        contrib_df.loc[
            contrib_df["branch"] == branch, "absolute_contribution_percent"
        ].to_numpy()
        for branch in branch_order
    ]
    positions = np.arange(1, len(branch_order) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    violin = ax.violinplot(
        box_data,
        positions=positions,
        widths=0.76,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, branch in zip(violin["bodies"], branch_order):
        body.set_facecolor(colors[branch])
        body.set_edgecolor(colors[branch])
        body.set_alpha(0.30)
        body.set_linewidth(1.5)

    box = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.28,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"linewidth": 1.6},
        capprops={"linewidth": 1.6},
        boxprops={"linewidth": 1.6},
    )
    for patch, branch in zip(box["boxes"], branch_order):
        patch.set_facecolor(colors[branch])
        patch.set_alpha(0.88)

    means = stats["mean"].to_numpy()
    medians = stats["median"].to_numpy()
    ax.scatter(
        positions,
        medians,
        s=72,
        color="black",
        zorder=5,
        label="Median",
    )
    for x_pos, mean_value in zip(positions, means):
        ax.text(
            x_pos,
            103.5,
            f"{mean_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(branch_order, fontsize=18)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Absolute logit contribution (%)", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#d9d9d9", linewidth=1.0, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Global Logit Contribution Distribution", fontsize=24, pad=18)

    fig.tight_layout()
    out_path = out_dir / "global_logit_contribution_distribution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _class_probabilities_from_row(row: pd.Series):
    cumulative = np.array([row[f"prob_mrs_gt_{i}"] for i in range(6)], dtype=float)
    return _class_probabilities_from_cumulative(cumulative)


def _class_probabilities_from_cumulative(cumulative: np.ndarray):
    probs = np.zeros(7, dtype=float)
    probs[0] = 1.0 - cumulative[0]
    for i in range(1, 6):
        probs[i] = cumulative[i - 1] - cumulative[i]
    probs[6] = cumulative[5]
    return np.clip(probs, 0.0, 1.0)


def _sigmoid_np(logits: np.ndarray):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def _logits_from_row(row: pd.Series, prefix: str):
    return np.array([row[f"{prefix}_{i}"] for i in range(6)], dtype=float)


def _image_logits_from_row(row: pd.Series):
    return _logits_from_row(row, "image_logit")


def _full_logits_from_row(row: pd.Series):
    image = _image_logits_from_row(row)
    pre = _logits_from_row(row, "pre_prior_logit")
    post = _logits_from_row(row, "post_prior_logit")
    return image + pre + post


def _class_probabilities_from_logits(logits: np.ndarray):
    cumulative = _sigmoid_np(logits)
    return _class_probabilities_from_cumulative(cumulative)


def _pred_label_from_logits(logits: np.ndarray):
    return int((_sigmoid_np(logits) > 0.5).sum())


def _save_image_vs_full_probabilities(row: pd.Series):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    image_logits = _image_logits_from_row(row)
    full_logits = _full_logits_from_row(row)
    image_probs = _sigmoid_np(image_logits)
    full_probs = _sigmoid_np(full_logits)
    image_pred = _pred_label_from_logits(image_logits)
    full_pred = _pred_label_from_logits(full_logits)

    x = np.arange(6)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width / 2, image_probs, width, color="#4e79a7", label=f"Image-only pred={image_pred}")
    ax.bar(x + width / 2, full_probs, width, color="#f28e2b", label=f"Image+Prior pred={full_pred}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P(mRS>{i})" for i in range(6)])
    ax.set_ylim(0, max(1.0, float(max(image_probs.max(), full_probs.max())) * 1.16))
    ax.set_ylabel("Threshold probability")
    ax.set_title(f"Image-only vs Image+Prior Threshold Probabilities | true={int(row['true_label'])}")
    ax.legend(frameon=False, loc="upper right")

    for i, p in enumerate(image_probs):
        ax.text(i - width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7)
    for i, p in enumerate(full_probs):
        ax.text(i + width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = case_dir / "image_vs_full_probabilities.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_image_vs_full_summary(df: pd.DataFrame):
    rows = []
    for _, row in df.iterrows():
        image_logits = _image_logits_from_row(row)
        full_logits = _full_logits_from_row(row)
        image_threshold_probs = _sigmoid_np(image_logits)
        full_threshold_probs = _sigmoid_np(full_logits)
        image_class_probs = _class_probabilities_from_cumulative(image_threshold_probs)
        full_class_probs = _class_probabilities_from_cumulative(full_threshold_probs)
        true_label = int(row["true_label"])
        image_pred = _pred_label_from_logits(image_logits)
        full_pred = _pred_label_from_logits(full_logits)
        image_abs_error = abs(image_pred - true_label)
        full_abs_error = abs(full_pred - true_label)
        image_correct = image_pred == true_label
        full_correct = full_pred == true_label
        if image_correct and full_correct:
            effect_category = "Both correct"
        elif (not image_correct) and full_correct:
            effect_category = "Corrected"
        elif image_correct and (not full_correct):
            effect_category = "Harmed"
        else:
            effect_category = "Both wrong"

        out = {
            "patient_id": row["patient_id"],
            "true_label": true_label,
            "image_pred_label": image_pred,
            "full_pred_label": full_pred,
            "image_correct": image_correct,
            "full_correct": full_correct,
            "prior_effect_category": effect_category,
            "image_abs_error": image_abs_error,
            "full_abs_error": full_abs_error,
            "delta_abs_error": float(full_abs_error - image_abs_error),
            "image_prob_poor_outcome": float(image_threshold_probs[2]),
            "full_prob_poor_outcome": float(full_threshold_probs[2]),
            "delta_full_minus_image_poor": float(full_threshold_probs[2] - image_threshold_probs[2]),
        }
        for i in range(7):
            out[f"image_prob_mrs_{i}"] = float(image_class_probs[i])
            out[f"full_prob_mrs_{i}"] = float(full_class_probs[i])
        rows.append(out)
    return pd.DataFrame(rows)


def _save_prediction_error_change(summary_df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = summary_df["delta_abs_error"].to_numpy(dtype=float)
    bins = np.arange(delta.min() - 0.5, delta.max() + 1.5, 1.0) if len(delta) else np.array([-0.5, 0.5])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(delta, bins=bins, color="#4e79a7", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Delta absolute error = |Image+Prior - True| - |Image-only - True|")
    ax.set_ylabel("Case count")
    ax.set_title("Prediction Error Change After Adding Prior")
    ax.text(
        0.98,
        0.95,
        f"Mean = {delta.mean():.3f}\nMedian = {np.median(delta):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    plt.tight_layout()
    out_path = out_dir / "prediction_error_change.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_prediction_transition_matrix(summary_df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.crosstab(
        summary_df["image_pred_label"],
        summary_df["full_pred_label"],
    ).reindex(index=range(7), columns=range(7), fill_value=0)
    matrix.to_csv(out_dir / "image_to_full_prediction_transition_matrix.csv", encoding="utf-8-sig")

    values = matrix.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(values, cmap="Blues", aspect="equal")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([f"mRS {i}" for i in range(7)])
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels([f"mRS {i}" for i in range(7)])
    ax.set_xlabel("Image+Prior prediction")
    ax.set_ylabel("Image-only prediction")
    ax.set_title("Prediction Shift Matrix: Image-only -> Image+Prior")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Case count")

    threshold = values.max() / 2 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, str(values[i, j]), ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "image_to_full_prediction_transition_matrix.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_prior_effect_categories(summary_df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["Both correct", "Corrected", "Harmed", "Both wrong"]
    counts = summary_df["prior_effect_category"].value_counts().reindex(order, fill_value=0)
    percent = counts / max(int(counts.sum()), 1) * 100.0
    table = pd.DataFrame({"category": order, "count": counts.to_numpy(), "percent": percent.to_numpy()})
    table.to_csv(out_dir / "prior_effect_categories.csv", index=False, encoding="utf-8-sig")

    colors = ["#4e79a7", "#59a14f", "#e15759", "#9c755f"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(order))
    bars = ax.bar(x, counts.to_numpy(), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("Prior Effect on Exact mRS Prediction")
    ax.set_ylim(0, max(1, int(counts.max())) * 1.22)
    for bar, count_value, pct in zip(bars, counts.to_numpy(), percent.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count_value)}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path = out_dir / "prior_effect_categories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_attention_frequency(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase in ["pre", "post"]:
        att_cols = [f"{phase}_prior_attention_attention_weight_{region}" for region in REGION_NAMES]
        if not all(col in df.columns for col in att_cols):
            continue
        top_regions = []
        for _, row in df.iterrows():
            att_values = np.array([row[col] for col in att_cols], dtype=float)
            if np.all(np.isnan(att_values)):
                continue
            top_regions.append(REGION_NAMES[int(np.nanargmax(att_values))])
        counts = pd.Series(top_regions).value_counts().reindex(REGION_NAMES, fill_value=0)
        rows.append({"phase": phase, **counts.to_dict()})

    if not rows:
        return None

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "attention_top_region_frequency.csv", index=False, encoding="utf-8-sig")
    plot_df = table.set_index("phase")

    x = np.arange(len(REGION_NAMES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    pre_vals = plot_df.loc["pre", REGION_NAMES].to_numpy(dtype=float) if "pre" in plot_df.index else np.zeros(len(REGION_NAMES))
    post_vals = plot_df.loc["post", REGION_NAMES].to_numpy(dtype=float) if "post" in plot_df.index else np.zeros(len(REGION_NAMES))
    ax.bar(x - width / 2, pre_vals, width, label="Pre-op", color="#f28e2b")
    ax.bar(x + width / 2, post_vals, width, label="Post-op", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, rotation=25, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("Most-attended Region Frequency by Phase")
    ax.legend(frameon=False)
    for idx, value in enumerate(pre_vals):
        ax.text(idx - width / 2, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(post_vals):
        ax.text(idx + width / 2, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "attention_top_region_frequency.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_prior_frequency(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase in ["pre", "post"]:
        prior_cols = [f"{phase}_prior_branch_{region}" for region in REGION_NAMES]
        if not all(col in df.columns for col in prior_cols):
            continue
        top_regions = []
        for _, row in df.iterrows():
            prior_values = np.array([row[col] for col in prior_cols], dtype=float)
            if np.all(np.isnan(prior_values)):
                continue
            top_regions.append(REGION_NAMES[int(np.nanargmax(prior_values))])
        counts = pd.Series(top_regions).value_counts().reindex(REGION_NAMES, fill_value=0)
        rows.append({"phase": phase, **counts.to_dict()})

    if not rows:
        return None

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "prior_top_region_frequency.csv", index=False, encoding="utf-8-sig")
    plot_df = table.set_index("phase")

    x = np.arange(len(REGION_NAMES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    pre_vals = plot_df.loc["pre", REGION_NAMES].to_numpy(dtype=float) if "pre" in plot_df.index else np.zeros(len(REGION_NAMES))
    post_vals = plot_df.loc["post", REGION_NAMES].to_numpy(dtype=float) if "post" in plot_df.index else np.zeros(len(REGION_NAMES))
    ax.bar(x - width / 2, pre_vals, width, label="Pre-op", color="#f28e2b")
    ax.bar(x + width / 2, post_vals, width, label="Post-op", color="#4e79a7")
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, rotation=25, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("Most-prior Region Frequency by Phase")
    ax.legend(frameon=False)
    for idx, value in enumerate(pre_vals):
        ax.text(idx - width / 2, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(post_vals):
        ax.text(idx + width / 2, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "prior_top_region_frequency.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_attention_prior_correlation(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for ax, phase, color in zip(axes, ["pre", "post"], ["#f28e2b", "#4e79a7"]):
        prior_cols = [f"{phase}_prior_branch_{region}" for region in REGION_NAMES]
        att_cols = [f"{phase}_prior_attention_attention_weight_{region}" for region in REGION_NAMES]
        prior_rows = []
        att_rows = []
        if not all(col in df.columns for col in prior_cols + att_cols):
            ax.set_title(f"{phase.upper()} Prior vs Attention")
            ax.set_xlabel("Prior score")
            continue
        for _, row in df.iterrows():
            prior_values = np.array([row[col] for col in prior_cols], dtype=float)
            att_values = np.array([row[col] for col in att_cols], dtype=float)
            if np.all(np.isnan(prior_values)) or np.all(np.isnan(att_values)):
                continue
            prior_rows.extend(prior_values.tolist())
            att_rows.extend(att_values.tolist())

        prior_arr = np.array(prior_rows, dtype=float)
        att_arr = np.array(att_rows, dtype=float)
        ax.scatter(prior_arr, att_arr, s=14, alpha=0.35, color=color, edgecolors="none")
        if len(prior_arr) >= 2 and np.std(prior_arr) > 0 and np.std(att_arr) > 0:
            corr = float(np.corrcoef(prior_arr, att_arr)[0, 1])
        else:
            corr = float("nan")
        if len(prior_arr) >= 2:
            slope, intercept = np.polyfit(prior_arr, att_arr, 1)
            xs = np.linspace(prior_arr.min(), prior_arr.max(), 100)
            ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2)
        else:
            corr = float("nan")
        ax.set_title(f"{phase.upper()} Prior vs Attention")
        ax.set_xlabel("Prior score")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"r = {corr:.3f}" if np.isfinite(corr) else "r = N/A",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axes[0].set_ylabel("Attention weight")
    plt.tight_layout()
    out_path = out_dir / "attention_prior_correlation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_image_vs_full_scatter(summary_df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    colors = np.where(summary_df["true_label"].to_numpy(dtype=int) >= 3, "#d62728", "#4e79a7")
    ax.scatter(
        summary_df["image_prob_poor_outcome"],
        summary_df["full_prob_poor_outcome"],
        c=colors,
        alpha=0.72,
        s=22,
        edgecolors="none",
    )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Image-only P(mRS > 2)")
    ax.set_ylabel("Image+Prior P(mRS > 2)")
    ax.set_title("Image-only vs Image+Prior Poor-outcome Probability")
    ax.text(0.02, 0.96, "Blue: true mRS 0-2\nRed: true mRS 3-6", va="top", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "image_vs_full_poor_outcome_scatter.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_class_probabilities(row: pd.Series):
    case_dir = OUTPUT_ROOT / "case_level" / f"{row['patient_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    probs = _class_probabilities_from_row(row)
    x = np.arange(7)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    colors = ["#4e79a7"] * 7
    colors[int(row["pred_label"])] = "#f28e2b"
    ax.bar(x, probs, color=colors)
    ax.axvline(int(row["true_label"]), color="black", linestyle="--", linewidth=1.2, label="True label")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mRS {i}" for i in range(7)])
    ax.set_ylim(0, max(1.0, probs.max() * 1.15))
    ax.set_ylabel("Class probability")
    ax.set_title("mRS Class Probabilities")
    ax.legend(frameon=False)

    for i, p in enumerate(probs):
        ax.text(i, p, f"{p:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(case_dir / "class_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_case_summary(case_rows: pd.DataFrame, selected_out: Path):
    cols = ["patient_id", "true_label", "pred_label", "prob_poor_outcome"]
    extra = [c for c in ["gap", "correct"] if c in case_rows.columns]
    case_rows[cols + extra].to_csv(selected_out, index=False, encoding="utf-8-sig")


def _values_by_regions(row: pd.Series, prefix: str, phase: str):
    return np.array([row[f"{phase}_{prefix}_{region}"] for region in REGION_NAMES], dtype=float)


def _heatmap_text_color(value: float, vmin: float, vmax: float):
    if not np.isfinite(value) or vmax <= vmin:
        return "black"
    normalized = (value - vmin) / (vmax - vmin)
    return "black" if normalized >= 0.42 else "white"


def _save_heatmap(matrix: np.ndarray, row_labels: list[str], title: str, cbar_label: str, out_path: Path):
    height = max(4.8, 0.86 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(12.0, height))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(REGION_NAMES)))
    ax.set_xticklabels(
        [REGION_DISPLAY_NAMES.get(name, name) for name in REGION_NAMES],
        rotation=30,
        ha="right",
        fontsize=19,
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=19)
    ax.tick_params(axis="both", labelsize=19)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label(cbar_label, fontsize=19)
    cbar.ax.tick_params(labelsize=18)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=16,
                fontweight="semibold",
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _group_matrix(df: pd.DataFrame, group_col: str, prefix: str, phase: str):
    labels = []
    rows = []
    for label in sorted(df[group_col].dropna().astype(int).unique()):
        sub = df[df[group_col].astype(int) == label]
        values = np.vstack([_values_by_regions(row, prefix, phase) for _, row in sub.iterrows()])
        labels.append(f"mRS={label}")
        rows.append(values.mean(axis=0))
    return np.vstack(rows), labels


def _save_group_region_figures(df: pd.DataFrame, prefix: str, ylabel: str, name: str):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    for group_col in ["true_label", "pred_label"]:
        for phase in ["pre", "post"]:
            matrix, labels = _group_matrix(df, group_col, prefix, phase)
            _save_heatmap(
                matrix,
                labels,
                title="",
                cbar_label=ylabel,
                out_path=out_dir / f"{name}_{phase}_by_{group_col}.png",
            )


def _save_combined_prior_attention_heatmap(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("pre", "prior_branch", "PRE Prior score"),
        ("post", "prior_branch", "POST Prior score"),
        ("pre", "prior_attention_attention_weight", "PRE Attention weight"),
        ("post", "prior_attention_attention_weight", "POST Attention weight"),
    ]

    matrices = []
    labels = None
    for phase, prefix, _title in panels:
        matrix, row_labels = _group_matrix(df, "true_label", prefix, phase)
        matrices.append(matrix)
        labels = row_labels

    fig, axes = plt.subplots(2, 2, figsize=(18.5, 11.5), sharex=True, sharey=True)
    xticklabels = [REGION_DISPLAY_NAMES.get(name, name) for name in REGION_NAMES]
    for ax, matrix, (_phase, _prefix, panel_title) in zip(axes.flat, matrices, panels):
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_title(panel_title, fontsize=21)
        ax.set_xticks(np.arange(len(REGION_NAMES)))
        ax.set_xticklabels(xticklabels, rotation=30, ha="right", fontsize=18)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=18)
        ax.tick_params(axis="both", labelsize=18)
        cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.025)
        cbar.set_label(panel_title.split(" ", 1)[1], fontsize=18)
        cbar.ax.tick_params(labelsize=17)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=15,
                    fontweight="semibold",
                )

    plt.tight_layout()
    out_path = out_dir / "combined_prior_attention_by_true_label.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_group_logit_contribution(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    for group_col in ["true_label", "pred_label"]:
        labels = []
        image_rows = []
        pre_rows = []
        post_rows = []
        final_rows = []
        for label in sorted(df[group_col].dropna().astype(int).unique()):
            sub = df[df[group_col].astype(int) == label]
            image = sub[[f"image_logit_{i}" for i in range(6)]].mean(axis=0).to_numpy(dtype=float)
            pre = sub[[f"pre_prior_logit_{i}" for i in range(6)]].mean(axis=0).to_numpy(dtype=float)
            post = sub[[f"post_prior_logit_{i}" for i in range(6)]].mean(axis=0).to_numpy(dtype=float)
            labels.append(f"{group_col}={label} (n={len(sub)})")
            image_rows.append(image)
            pre_rows.append(pre)
            post_rows.append(post)
            final_rows.append(image + pre + post)

        xticklabels = [f"T{i}" for i in range(6)]
        for matrix, tag, title in [
            (np.vstack(image_rows), "image", "Image Logits"),
            (np.vstack(pre_rows), "pre_prior", "Pre-prior Logits"),
            (np.vstack(post_rows), "post_prior", "Post-prior Logits"),
            (np.vstack(final_rows), "final", "Final Logits"),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.55 * len(labels) + 1.6)))
            im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
            ax.set_xticks(np.arange(len(xticklabels)))
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_title(f"{title} by {group_col}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
            cbar.set_label("Logit")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
            plt.tight_layout()
            fig.savefig(out_dir / f"logit_{tag}_by_{group_col}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def _save_group_class_probabilities(df: pd.DataFrame):
    out_dir = OUTPUT_ROOT / "group_level"
    for group_col in ["true_label", "pred_label"]:
        labels = []
        rows = []
        for label in sorted(df[group_col].dropna().astype(int).unique()):
            sub = df[df[group_col].astype(int) == label]
            matrix = np.vstack([_class_probabilities_from_row(row) for _, row in sub.iterrows()])
            labels.append(f"{group_col}={label} (n={len(sub)})")
            rows.append(matrix.mean(axis=0))

        matrix = np.vstack(rows)
        fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.55 * len(labels) + 1.6)))
        im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(1.0, float(matrix.max())))
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([f"mRS {i}" for i in range(7)])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"Mean mRS Class Probabilities by {group_col}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
        cbar.set_label("Class probability")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / f"class_probabilities_by_{group_col}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_group_level_figures(df: pd.DataFrame):
    _save_group_region_figures(
        df,
        prefix="prior_branch",
        ylabel="Prior score",
        name="prior_scores",
    )
    _save_group_region_figures(
        df,
        prefix="prior_attention_attention_weight",
        ylabel="Attention weight",
        name="attention_weights",
    )
    _save_combined_prior_attention_heatmap(df)
    _save_prior_frequency(df)
    _save_attention_frequency(df)


def main():
    _set_plot_font()
    _ensure_dirs()

    if DATASET_NAME not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module, model = _load_model(cfg, device)
    df = _load_data()
    selected = _select_cases(df)

    selected_out = OUTPUT_ROOT / "selected_cases.csv"
    _save_case_summary(selected, selected_out)
    image_vs_full_summary = _build_image_vs_full_summary(selected)
    image_vs_full_summary.to_csv(
        OUTPUT_ROOT / "image_vs_full_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_group_level_figures(selected)

    case_outputs = []
    total = len(selected)
    for idx, (_, row) in enumerate(selected.iterrows(), start=1):
        print(f"[{idx}/{total}] Processing patient {row['patient_id']} ...", flush=True)
        case_outputs.append(
            {
                "patient_id": str(row["patient_id"]),
                "gradcam": str(_save_gradcam_case(row, model, module)),
                "logit_contribution": str(_save_logit_contribution(row)),
                "image_vs_full_probabilities": str(_save_image_vs_full_probabilities(row)),
            }
        )

    pd.DataFrame(case_outputs).to_csv(
        OUTPUT_ROOT / "case_outputs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Interpretability outputs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
