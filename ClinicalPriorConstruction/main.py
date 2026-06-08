#!/usr/bin/env python3
"""
Extract clinical prior features from registered CT images, segmentation masks,
and patient-specific vascular-territory atlases.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


VASCULAR_TERRITORIES = {
    0: "Background",
    1: "ACA_Left",
    2: "ACA_Right",
    3: "MCA_Left",
    4: "MCA_Right",
    5: "PCA_Left",
    6: "PCA_Right",
    7: "Brainstem_Left",
    8: "Brainstem_Right",
    9: "Cerebellum_Left",
    10: "Cerebellum_Right",
    11: "Cistern",
}


def compute_mask_stats(ct_array, mask, voxel_vol_cm3):
    voxel_count = int(np.sum(mask))
    volume = voxel_count * voxel_vol_cm3
    features = {"volume": volume, "mean_hu": 0.0, "std_hu": 0.0, "entropy": 0.0}

    if voxel_count > 0 and ct_array is not None:
        hu_vals = ct_array[mask].astype(np.float32)
        features["mean_hu"] = float(np.mean(hu_vals))
        features["std_hu"] = float(np.std(hu_vals))
        hist, _ = np.histogram(hu_vals, bins=32)
        hist = hist[hist > 0].astype(np.float32)
        prob = hist / hist.sum()
        features["entropy"] = float(-np.sum(prob * np.log2(prob + 1e-10)))

    return features


def extract_sah_region_features(ct_array, seg_array, atlas_array, voxel_vol_cm3):
    sah_mask = seg_array == 2
    results = {}

    for region_id, region_name in VASCULAR_TERRITORIES.items():
        if region_id == 0:
            continue

        overlap = sah_mask & (atlas_array == region_id)
        voxel_count = int(np.sum(overlap))
        features = {
            "volume": voxel_count * voxel_vol_cm3,
            "mean_hu": 0.0,
            "std_hu": 0.0,
            "entropy": 0.0,
        }

        if voxel_count > 0 and ct_array is not None:
            hu_vals = ct_array[overlap].astype(np.float32)
            features["mean_hu"] = float(np.mean(hu_vals))
            features["std_hu"] = float(np.std(hu_vals))
            hist, _ = np.histogram(hu_vals, bins=32)
            hist = hist[hist > 0].astype(np.float32)
            prob = hist / hist.sum()
            features["entropy"] = float(-np.sum(prob * np.log2(prob + 1e-10)))

        results[region_name] = features

    return results


def compute_global_volume(region_features):
    return sum(features["volume"] for features in region_features.values())


def compute_global_mean_hu(region_features):
    total_vol = 0.0
    weighted_hu = 0.0
    for features in region_features.values():
        if features["volume"] > 0:
            weighted_hu += features["mean_hu"] * features["volume"]
            total_vol += features["volume"]
    return weighted_hu / total_vol if total_vol > 0 else 0.0


def analyze_patient(
    patient_id,
    pre_ct_path,
    pre_seg_path,
    post_ct_path,
    post_seg_path,
    pre_atlas_path,
    post_atlas_path,
):
    logger.info("Analyzing patient: %s", patient_id)
    result = {
        "patient_id": patient_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        pre_atlas = sitk.ReadImage(str(pre_atlas_path))
        post_atlas = sitk.ReadImage(str(post_atlas_path))
        pre_atlas_array = sitk.GetArrayFromImage(pre_atlas)
        post_atlas_array = sitk.GetArrayFromImage(post_atlas)
        spacing = pre_atlas.GetSpacing()
        voxel_vol_cm3 = spacing[0] * spacing[1] * spacing[2] * 0.001

        pre_seg = sitk.ReadImage(str(pre_seg_path))
        post_seg = sitk.ReadImage(str(post_seg_path))
        pre_seg_arr = sitk.GetArrayFromImage(pre_seg)
        post_seg_arr = sitk.GetArrayFromImage(post_seg)

        pre_ct_arr = None
        if pre_ct_path and Path(pre_ct_path).exists():
            pre_ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pre_ct_path))).astype(np.float32)

        post_ct_arr = None
        if post_ct_path and Path(post_ct_path).exists():
            post_ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(post_ct_path))).astype(np.float32)

        region_names = [name for name in VASCULAR_TERRITORIES.values() if name != "Background"]

        pre_sah = extract_sah_region_features(pre_ct_arr, pre_seg_arr, pre_atlas_array, voxel_vol_cm3)
        post_sah = extract_sah_region_features(post_ct_arr, post_seg_arr, post_atlas_array, voxel_vol_cm3)

        for region_name in region_names:
            for feature_key in ["volume", "mean_hu"]:
                result[f"pre_SAH_{region_name}_{feature_key}"] = pre_sah[region_name][feature_key]
                result[f"post_SAH_{region_name}_{feature_key}"] = post_sah[region_name][feature_key]

        pre_ivh_stats = compute_mask_stats(pre_ct_arr, pre_seg_arr == 1, voxel_vol_cm3)
        post_ivh_stats = compute_mask_stats(post_ct_arr, post_seg_arr == 1, voxel_vol_cm3)
        pre_vent_stats = compute_mask_stats(pre_ct_arr, pre_seg_arr == 3, voxel_vol_cm3)
        post_vent_stats = compute_mask_stats(post_ct_arr, post_seg_arr == 3, voxel_vol_cm3)

        pre_complete_vent_stats = compute_mask_stats(
            pre_ct_arr, (pre_seg_arr == 1) | (pre_seg_arr == 3), voxel_vol_cm3
        )
        post_complete_vent_stats = compute_mask_stats(
            post_ct_arr, (post_seg_arr == 1) | (post_seg_arr == 3), voxel_vol_cm3
        )

        for feature_key in ["volume", "mean_hu", "std_hu", "entropy"]:
            result[f"pre_IVH_{feature_key}"] = pre_ivh_stats[feature_key]
            result[f"post_IVH_{feature_key}"] = post_ivh_stats[feature_key]
            result[f"pre_Ventricle_{feature_key}"] = pre_vent_stats[feature_key]
            result[f"post_Ventricle_{feature_key}"] = post_vent_stats[feature_key]
            result[f"pre_CompleteVent_{feature_key}"] = pre_complete_vent_stats[feature_key]
            result[f"post_CompleteVent_{feature_key}"] = post_complete_vent_stats[feature_key]

        pre_ivh_vol = pre_ivh_stats["volume"]
        post_ivh_vol = post_ivh_stats["volume"]
        pre_sah_vol = compute_global_volume(pre_sah)
        post_sah_vol = compute_global_volume(post_sah)
        pre_vent_vol = pre_vent_stats["volume"]
        post_vent_vol = post_vent_stats["volume"]
        pre_complete_vent_vol = pre_complete_vent_stats["volume"]
        post_complete_vent_vol = post_complete_vent_stats["volume"]

        pre_sah_stats = compute_mask_stats(pre_ct_arr, pre_seg_arr == 2, voxel_vol_cm3)
        post_sah_stats = compute_mask_stats(post_ct_arr, post_seg_arr == 2, voxel_vol_cm3)

        result["IVH_clearance_rate"] = (
            (pre_ivh_vol - post_ivh_vol) / pre_ivh_vol * 100 if pre_ivh_vol > 0 else 0.0
        )
        result["SAH_clearance_rate"] = (
            (pre_sah_vol - post_sah_vol) / pre_sah_vol * 100 if pre_sah_vol > 0 else 0.0
        )
        result["Ventricle_volume_change_abs"] = post_vent_vol - pre_vent_vol
        result["Ventricle_volume_change_pct"] = (
            (post_vent_vol - pre_vent_vol) / pre_vent_vol * 100 if pre_vent_vol > 0 else 0.0
        )

        pre_total_hemorrhage = pre_ivh_vol + pre_sah_vol
        post_total_hemorrhage = post_ivh_vol + post_sah_vol

        result["pre_IVH_occupation_ratio"] = (
            pre_ivh_vol / pre_complete_vent_vol if pre_complete_vent_vol > 0 else 0.0
        )
        result["post_IVH_occupation_ratio"] = (
            post_ivh_vol / post_complete_vent_vol if post_complete_vent_vol > 0 else 0.0
        )
        result["pre_total_hemorrhage_volume"] = pre_total_hemorrhage
        result["post_total_hemorrhage_volume"] = post_total_hemorrhage
        result["total_ventricle_expansion_index"] = (
            (post_complete_vent_vol - pre_complete_vent_vol) / pre_complete_vent_vol
            if pre_complete_vent_vol > 0
            else 0.0
        )
        result["total_hemorrhage_clearance_rate"] = (
            (pre_total_hemorrhage - post_total_hemorrhage) / pre_total_hemorrhage * 100
            if pre_total_hemorrhage > 0
            else 0.0
        )

        ivh_cleared = pre_ivh_vol - post_ivh_vol
        vent_change = post_vent_vol - pre_vent_vol
        result["IVH_clearance_ventricle_recovery_ratio"] = (
            ivh_cleared / abs(vent_change) if abs(vent_change) > 0 else 0.0
        )

        pre_ivh_hu = pre_ivh_stats["mean_hu"]
        post_ivh_hu = post_ivh_stats["mean_hu"]
        pre_sah_hu = compute_global_mean_hu(pre_sah)
        post_sah_hu = compute_global_mean_hu(post_sah)
        result["IVH_density_evolution_rate"] = (
            (post_ivh_hu - pre_ivh_hu) / abs(pre_ivh_hu) if abs(pre_ivh_hu) > 0 else 0.0
        )
        result["SAH_density_evolution_rate"] = (
            (post_sah_hu - pre_sah_hu) / abs(pre_sah_hu) if abs(pre_sah_hu) > 0 else 0.0
        )

        result["IVH_heterogeneity_change"] = post_ivh_stats["std_hu"] - pre_ivh_stats["std_hu"]
        result["SAH_heterogeneity_change"] = post_sah_stats["std_hu"] - pre_sah_stats["std_hu"]

        pre_complete_vent_hu = pre_complete_vent_stats["mean_hu"]
        post_complete_vent_hu = post_complete_vent_stats["mean_hu"]
        result["ventricular_density_reduction"] = pre_complete_vent_hu - post_complete_vent_hu

        ivh_clearance = result["IVH_clearance_rate"] / 100.0
        sah_clearance = result["SAH_clearance_rate"] / 100.0
        vent_recovery = max(0.0, -result["total_ventricle_expansion_index"])
        result["surgical_efficacy_score"] = (
            0.5 * ivh_clearance + 0.3 * sah_clearance + 0.2 * vent_recovery
        )

        vent_hu_change = post_complete_vent_hu - pre_complete_vent_hu
        result["ventricle_shape_recovery"] = (
            -result["total_ventricle_expansion_index"] * 0.7
            + (vent_hu_change / (abs(pre_complete_vent_hu) + 1e-6)) * 0.3
        )

        result["pre_IVH_burden"] = (
            pre_ivh_vol / pre_complete_vent_vol if pre_complete_vent_vol > 0 else 0.0
        )
        result["post_IVH_burden"] = (
            post_ivh_vol / post_complete_vent_vol if post_complete_vent_vol > 0 else 0.0
        )
        result["IVH_occupation_reduction"] = (
            result["pre_IVH_occupation_ratio"] - result["post_IVH_occupation_ratio"]
        )
        result["complete_ventricular_volume_change_abs"] = (
            post_complete_vent_vol - pre_complete_vent_vol
        )
        result["complete_ventricular_volume_change_pct"] = (
            (post_complete_vent_vol - pre_complete_vent_vol) / pre_complete_vent_vol * 100
            if pre_complete_vent_vol > 0
            else 0.0
        )
        result["IVH_texture_simplification"] = pre_ivh_stats["entropy"] - post_ivh_stats["entropy"]
        result["SAH_texture_simplification"] = pre_sah_stats["entropy"] - post_sah_stats["entropy"]

        hemorrhage_relief = pre_total_hemorrhage - post_total_hemorrhage
        vent_relief = max(0.0, pre_vent_vol - post_vent_vol)
        result["space_occupying_relief_ratio"] = hemorrhage_relief / (vent_relief + 1e-6)

        logger.info("Feature extraction completed for %s: %d features", patient_id, len(result))
        return result

    except Exception as exc:
        logger.error("Failed to analyze patient %s: %s", patient_id, exc)
        return None


def _scan_flat_folder(folder):
    file_dict = {}
    for file_path in Path(folder).glob("*.nii.gz"):
        name = file_path.name
        if name.endswith("_seg.nii.gz"):
            patient_id = name.replace("_seg.nii.gz", "")
            file_dict.setdefault(patient_id, {})["seg"] = file_path
        else:
            patient_id = name.replace(".nii.gz", "")
            file_dict.setdefault(patient_id, {})["img"] = file_path

    pairs = []
    for patient_id, files in file_dict.items():
        if patient_id.endswith("-1") or "img" not in files or "seg" not in files:
            continue

        post_id = f"{patient_id}-1"
        post_files = file_dict.get(post_id, {})
        if "img" in post_files and "seg" in post_files:
            pairs.append(
                {
                    "patient_id": patient_id,
                    "pre_img": files["img"],
                    "pre_seg": files["seg"],
                    "post_img": post_files["img"],
                    "post_seg": post_files["seg"],
                }
            )

    return pairs


def normalize_data_roots(config):
    data_roots = config.get("data_roots")
    if data_roots is None:
        data_root = config.get("data_root")
        if not data_root:
            raise ValueError("Set either 'data_roots' or 'data_root'.")
        data_roots = [data_root]
    elif isinstance(data_roots, (str, Path)):
        data_roots = [data_roots]

    normalized = []
    for root in data_roots:
        root_path = Path(root)
        if root_path.exists():
            normalized.append(root_path)
        else:
            logger.warning("Data root does not exist, skipping: %s", root_path)

    if not normalized:
        raise ValueError("No valid data root was found.")
    return normalized


def collect_patient_pairs(data_roots):
    all_pairs = []
    seen_patient_ids = set()

    for data_root in data_roots:
        current_pairs = _scan_flat_folder(data_root)
        logger.info("Found %d patient pairs in %s", len(current_pairs), data_root)
        for pair in current_pairs:
            patient_id = pair["patient_id"]
            if patient_id in seen_patient_ids:
                logger.warning("Duplicate patient ID skipped: %s", patient_id)
                continue
            seen_patient_ids.add(patient_id)
            all_pairs.append(pair)

    logger.info("Found %d unique patient pairs", len(all_pairs))
    return all_pairs


def attach_individualized_atlas_paths(patient_pairs, registration_result_root):
    registration_result_root = Path(registration_result_root)
    updated_pairs = []

    for pair in patient_pairs:
        patient_id = pair["patient_id"]
        registration_dir = registration_result_root / patient_id
        pre_atlas = registration_dir / "individualized_annotation_in_preop_mni_affine.nii.gz"
        post_atlas = registration_dir / "individualized_annotation_in_postop_mni_affine.nii.gz"

        if not pre_atlas.exists() or not post_atlas.exists():
            logger.warning(
                "Missing individualized atlas for %s: pre=%s, post=%s",
                patient_id,
                pre_atlas.exists(),
                post_atlas.exists(),
            )
            continue

        pair = dict(pair)
        pair["pre_atlas"] = pre_atlas
        pair["post_atlas"] = post_atlas
        updated_pairs.append(pair)

    logger.info("Found individualized atlases for %d patients", len(updated_pairs))
    return updated_pairs


def resolve_output_csv_path(config):
    output_dir = Path(config["output_dir"])
    output_csv_name = config.get("output_csv_name", "features.csv")
    return output_dir / output_csv_name


def normalize_centers_config(config):
    centers = config.get("centers")
    if not centers:
        single_config = dict(config)
        single_config.setdefault("name", "default")
        return [single_config]

    shared_keys = ["registration_result_root", "output_dir", "resume", "max_patients"]
    normalized = []
    for center_config in centers:
        merged = {key: config[key] for key in shared_keys if key in config}
        merged.update(center_config)
        if "name" not in merged:
            raise ValueError("Each center config must include 'name'.")
        if "output_csv_name" not in merged:
            raise ValueError(f"Center {merged['name']} is missing 'output_csv_name'.")
        normalized.append(merged)

    return normalized


def run(config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = resolve_output_csv_path(config)

    patient_pairs = collect_patient_pairs(normalize_data_roots(config))
    patient_pairs = attach_individualized_atlas_paths(
        patient_pairs,
        config["registration_result_root"],
    )
    if not patient_pairs:
        logger.error("No usable patient pairs found.")
        return

    if config.get("max_patients"):
        patient_pairs = patient_pairs[: config["max_patients"]]

    processed = set()
    all_results = []
    if config.get("resume") and output_csv.exists():
        existing = pd.read_csv(output_csv, encoding="utf-8-sig")
        if "patient_id" in existing.columns:
            processed = set(str(value) for value in existing["patient_id"])
            all_results = existing.to_dict("records")
            logger.info("Resume enabled: %d patients already processed", len(processed))

    remaining = [pair for pair in patient_pairs if pair["patient_id"] not in processed]
    logger.info("Patients remaining: %d", len(remaining))

    for pair in tqdm(remaining, desc="Feature extraction"):
        result = analyze_patient(
            patient_id=pair["patient_id"],
            pre_ct_path=pair["pre_img"],
            pre_seg_path=pair["pre_seg"],
            post_ct_path=pair["post_img"],
            post_seg_path=pair["post_seg"],
            pre_atlas_path=pair["pre_atlas"],
            post_atlas_path=pair["post_atlas"],
        )
        if result is not None:
            all_results.append(result)
            pd.DataFrame(all_results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info("Saved %d patients and %d columns to %s", len(df), len(df.columns), output_csv)
    else:
        logger.warning("No patient was processed successfully.")


def run_all_centers(config):
    center_configs = normalize_centers_config(config)
    selected_centers = config.get("selected_centers")
    if selected_centers:
        selected_centers = set(selected_centers)
        center_configs = [cfg for cfg in center_configs if cfg["name"] in selected_centers]
        if not center_configs:
            raise ValueError(f"No center matched selected_centers: {sorted(selected_centers)}")

    for center_config in center_configs:
        logger.info("Processing center: %s", center_config["name"])
        run(center_config)


if __name__ == "__main__":
    CONFIG = {
        "registration_result_root": "/path/to/RegistrationAndSkullStripping/result_v3",
        "data_roots": [
            "/path/to/registered_ct_and_segmentation_folder",
        ],
        "output_dir": "./outputs",
        "output_csv_name": "features.csv",
        "resume": True,
        "max_patients": None,
    }

    run(CONFIG)
