"""Generate four Grad-CAM axial series for every selected patient."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PADN_DIR = (SCRIPT_DIR / ".." / "PADN").resolve()
if str(PADN_DIR) not in sys.path:
    sys.path.insert(0, str(PADN_DIR))

MODEL_FILE = PADN_DIR / "Main.py"
RESULT_DIR = PADN_DIR / "outputs" / "dl_models"
REGISTRATION_OUTPUT_DIR = (
    SCRIPT_DIR / ".." / "RegistrationAndSkullStripping" / "result_v3"
).resolve()
CSV_DIR = SCRIPT_DIR / "data" / "features"
CT_DATA_DIR = SCRIPT_DIR / "data" / "registered_ct"

CAM_TYPES = (
    "pre_image",
    "post_image",
    "pre_prior_guided",
    "post_prior_guided",
)

CAM_TITLES = {
    "pre_image": "Pre-op image",
    "post_image": "Post-op image",
    "pre_prior_guided": "Pre-op prior-guided",
    "post_prior_guided": "Post-op prior-guided",
}

DATASET_SPECS = {
    "efy": {
        "feature_csv": CSV_DIR / "featuresefy.csv",
        "pred_csv": RESULT_DIR / "external_efy_predictions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "efy",
    },
    "ay2": {
        "feature_csv": CSV_DIR / "featuresay2.csv",
        "pred_csv": RESULT_DIR / "external_ay2_predictions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "ay2",
    },
    "th": {
        "feature_csv": CSV_DIR / "featuresth.csv",
        "pred_csv": RESULT_DIR / "external_th_predictions_best_kappa.csv",
        "ct_dir": CT_DATA_DIR / "th",
    },
}


def load_model_module(model_file: Path):
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    spec = importlib.util.spec_from_file_location("padn_main", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(result_dir: Path) -> dict:
    config_path = result_dir / "config.json"
    config = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    config.setdefault("dropout", 0.6)
    config.setdefault("use_region_attention", True)
    config.setdefault("model_mode", "phase_prior_guided_change")
    config.setdefault("target_shape", (182, 218, 182))
    config.setdefault("patient_template_root", str(REGISTRATION_OUTPUT_DIR))
    return config


def load_model(model_file: Path, checkpoint_file: Path, config: dict, device: torch.device):
    module = load_model_module(model_file)
    model = module.DualChannelPredictor(
        dropout=float(config["dropout"]),
        use_region_attention=bool(config["use_region_attention"]),
        region_masks=None,
        model_mode=config["model_mode"],
    )
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return module, model


def threshold_count_prediction(frame: pd.DataFrame) -> np.ndarray:
    columns = [f"prob_mrs_gt_{index}" for index in range(6)]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    return (frame[columns].astype(float).to_numpy() > 0.5).sum(axis=1).astype(int)


def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    spec = DATASET_SPECS[dataset_name]
    for key in ("feature_csv", "pred_csv"):
        if not spec[key].is_file():
            raise FileNotFoundError(f"Input file not found: {spec[key]}")

    predictions = pd.read_csv(spec["pred_csv"], dtype={"patient_id": str})
    features = pd.read_csv(spec["feature_csv"], dtype={"patient_id": str})
    duplicates = [
        column for column in predictions.columns
        if column in features.columns and column != "patient_id"
    ]
    features = features.drop(columns=duplicates)
    merged = predictions.merge(features, on="patient_id", how="inner")
    if merged.empty:
        raise ValueError("Prediction and feature CSV files have no matched patients.")

    merged["pred_label"] = threshold_count_prediction(merged)
    if "true_label" not in merged.columns:
        if "mRS" not in merged.columns:
            raise ValueError("Input data must contain true_label or mRS.")
        merged["true_label"] = merged["mRS"]
    merged["true_label"] = merged["true_label"].astype(int)
    merged["y"] = merged["true_label"]
    merged["ct_data_dir"] = str(spec["ct_dir"])
    merged["flat_folder_mode"] = True
    merged["center"] = dataset_name
    return merged


def select_cases(frame: pd.DataFrame) -> pd.DataFrame:
    if RUN_ALL_CASES or not CASE_IDS:
        selected = frame.copy()
    else:
        wanted = {str(patient_id) for patient_id in CASE_IDS}
        selected = frame[frame["patient_id"].astype(str).isin(wanted)].copy()
    if selected.empty:
        raise ValueError(f"CASE_IDS did not match any patients: {CASE_IDS}")
    return selected


def load_single_case(module, row: pd.Series, config: dict):
    case_frame = pd.DataFrame([row.to_dict()])
    case_frame["y"] = case_frame["true_label"].astype(int)
    case_frame["ct_data_dir"] = row["ct_data_dir"]
    case_frame["flat_folder_mode"] = True
    dataset = module.CTDataset(
        case_frame,
        ct_data_dir=str(row["ct_data_dir"]),
        target_shape=tuple(config["target_shape"]),
        patient_template_root=config["patient_template_root"],
        use_region_attention=bool(config["use_region_attention"]),
    )
    values = dataset[0]
    if len(values) != 6:
        raise RuntimeError(f"Expected six CTDataset outputs, received {len(values)}.")
    return tuple(value.unsqueeze(0) for value in values)


def target_index(row: pd.Series) -> int:
    mode = GRADCAM_TARGET_MODE.lower()
    if mode == "poor":
        return 2
    if mode == "true":
        return min(int(row["true_label"]), 5)
    if mode == "predicted":
        return min(int(row["pred_label"]), 5)
    raise ValueError("GRADCAM_TARGET_MODE must be predicted, true, or poor.")


def compute_gradcam(
    model,
    target_layer,
    pre_ct: torch.Tensor,
    post_ct: torch.Tensor,
    volumes: torch.Tensor,
    pre_region_masks: torch.Tensor,
    post_region_masks: torch.Tensor,
    output_index: int,
    disable_all_priors: bool,
) -> np.ndarray:
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(
            pre_ct,
            post_ct,
            volumes=volumes,
            pre_region_masks=pre_region_masks,
            post_region_masks=post_region_masks,
            disable_all_priors=disable_all_priors,
        )
        logits = output[0] if isinstance(output, tuple) else output
        if logits.ndim != 2 or logits.shape[1] != 6:
            raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
        logits[:, output_index].sum().backward()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM did not capture activation and gradient tensors.")
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=pre_ct.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        maximum = float(cam.max())
        if maximum > 0:
            cam /= maximum
        return cam
    finally:
        handle.remove()


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    minimum = float(volume.min())
    maximum = float(volume.max())
    if maximum <= minimum:
        return np.zeros_like(volume)
    return (volume - minimum) / (maximum - minimum)


def axial_slice_indices(depth: int, count: int) -> list[int]:
    if depth < count:
        raise ValueError(f"Volume depth ({depth}) is smaller than slice count ({count}).")
    return np.linspace(0, depth - 1, count, dtype=int).tolist()


def save_axial_series(image: np.ndarray, cam: np.ndarray, output_dir: Path, title: str) -> None:
    if image.shape != cam.shape:
        raise ValueError(f"Image and CAM shapes differ: {image.shape} vs {cam.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()

    indices = axial_slice_indices(image.shape[2], AXIAL_SLICE_COUNT)
    for order, slice_index in enumerate(indices, start=1):
        image_slice = np.rot90(image[:, :, slice_index], k=1)
        cam_slice = np.rot90(cam[:, :, slice_index], k=1)
        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(image_slice, cmap="gray")
        axes[1].imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[1].set_title("Input + Grad-CAM")
        axes[1].axis("off")
        figure.suptitle(f"{title} | axial slice {slice_index}", fontsize=10)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{order:02d}_slice_{slice_index:03d}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)


def patient_is_complete(patient_dir: Path) -> bool:
    return all(
        len(list((patient_dir / cam_type).glob("*.png"))) == AXIAL_SLICE_COUNT
        for cam_type in CAM_TYPES
    )


def process_patient(row: pd.Series, model, module, config: dict, device: torch.device) -> None:
    patient_id = str(row["patient_id"])
    patient_dir = OUTPUT_ROOT / patient_id
    if SKIP_EXISTING_GRADCAM and patient_is_complete(patient_dir):
        print(f"Skipping completed patient {patient_id}.")
        return

    pre_ct, post_ct, _label, volumes, pre_masks, post_masks = load_single_case(
        module, row, config
    )
    pre_ct = pre_ct.to(device)
    post_ct = post_ct.to(device)
    volumes = volumes.to(device)
    pre_masks = pre_masks.to(device)
    post_masks = post_masks.to(device)

    if not hasattr(model, "get_gradcam_target_layers"):
        raise RuntimeError("The model does not expose four Grad-CAM target layers.")
    layers = model.get_gradcam_target_layers()
    missing_layers = [cam_type for cam_type in CAM_TYPES if cam_type not in layers]
    if missing_layers:
        raise RuntimeError(f"The model is missing Grad-CAM layers: {missing_layers}")

    output_index = target_index(row)
    cams = {}
    for cam_type in CAM_TYPES:
        cams[cam_type] = compute_gradcam(
            model,
            layers[cam_type],
            pre_ct,
            post_ct,
            volumes,
            pre_masks,
            post_masks,
            output_index,
            disable_all_priors=cam_type in {"pre_image", "post_image"},
        )

    pre_image = normalize_volume(pre_ct[0, 0].detach().cpu().numpy())
    post_image = normalize_volume(post_ct[0, 0].detach().cpu().numpy())
    for cam_type in CAM_TYPES:
        image = pre_image if cam_type.startswith("pre_") else post_image
        save_axial_series(image, cams[cam_type], patient_dir / cam_type, CAM_TITLES[cam_type])
    print(f"Patient {patient_id}: saved {4 * AXIAL_SLICE_COUNT} PNG files.")


def main() -> None:
    config = load_config(RESULT_DIR)
    device = torch.device(DEVICE)
    module, model = load_model(
        MODEL_FILE, RESULT_DIR / CHECKPOINT_NAME, config, device
    )
    patients = select_cases(load_data(DATASET_NAME))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(patients)} patients on {device}.")
    for index, (_, row) in enumerate(patients.iterrows(), start=1):
        print(f"[{index}/{len(patients)}] Patient {row['patient_id']}")
        process_patient(row, model, module, config, device)
    print(f"Grad-CAM output: {OUTPUT_ROOT}")


# Edit these settings, then run this file directly.
DATASET_NAME = "efy"
CHECKPOINT_NAME = "best_kappa.pth"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
RUN_ALL_CASES = True
CASE_IDS: list[str] = []
GRADCAM_TARGET_MODE = "predicted"  # predicted, true, or poor
AXIAL_SLICE_COUNT = 30
SKIP_EXISTING_GRADCAM = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
