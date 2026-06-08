#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click prediction entry point for PADN, clinical model, and fusion model.

Edit CONFIG at the end of this file, then run:

    python run_full_prediction.py
"""

from __future__ import annotations

import os
import shutil
import sys
import importlib.util
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PADN_DIR = BASE_DIR / "PADN"
REG_DIR = BASE_DIR / "RegistrationAndSkullStripping"
SEG_DIR = BASE_DIR / "AutoSegmentation"
PRIOR_DIR = BASE_DIR / "ClinicalPriorConstruction"
CLINICAL_DIR = BASE_DIR / "ClinicalModelAndFusionModel"


def add_module_path(path):
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_file(module_name, file_path):
    file_path = Path(file_path)
    add_module_path(file_path.parent)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def threshold_probs_to_pred_label(threshold_probs, threshold=0.5):
    threshold_probs = np.asarray(threshold_probs, dtype=float)
    if threshold_probs.ndim == 1:
        threshold_probs = threshold_probs.reshape(1, -1)
    return (threshold_probs > threshold).sum(axis=1).astype(int)


def ordinal_class_probs_to_threshold_probs(class_probs):
    class_probs = np.asarray(class_probs, dtype=float)
    if class_probs.ndim == 1:
        class_probs = class_probs.reshape(1, -1)
    return np.column_stack(
        [class_probs[:, k + 1 :].sum(axis=1) for k in range(class_probs.shape[1] - 1)]
    )


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Required file not found:\n" + "\n".join(missing))


def require_columns(df, cols, name):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def check_conda_env():
    expected = "/home/yinpengzhan/miniconda3/envs/env/bin/python"
    if Path(sys.executable) != Path(expected):
        print(f"Warning: current Python is {sys.executable}")
        print(f"Expected Conda env python: {expected}")


def run_registration_from_dicom(config):
    if not config.get("run_registration", False):
        return

    reg_main = load_module_from_file("registration_pipeline_main", REG_DIR / "main.py")

    input_root = Path(config["input_dicom_root"])
    output_root = Path(config["registration_result_root"])
    greedy_bin = Path(config["greedy_bin"])
    dcm2niix_bin = Path(config["dcm2niix_bin"])
    hdbet_bin = Path(config["hdbet_bin"])
    mni_template = Path(config["mni_template"])
    mni_annotation = Path(config["mni_annotation"])

    require_files([greedy_bin, dcm2niix_bin, hdbet_bin, mni_template, mni_annotation])
    if not input_root.exists():
        raise FileNotFoundError(f"Input DICOM root not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    postop_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and path.name.endswith("-1"))
    patients = []
    for postop_dir in postop_dirs:
        patient_id = postop_dir.name[:-2]
        preop_dir = input_root / patient_id
        if preop_dir.is_dir():
            patients.append((patient_id, preop_dir, postop_dir))

    print(f"Registration patients: {len(patients)}")
    registered_files = []
    for patient_id, preop_dir, postop_dir in patients:
        out_dir = output_root / patient_id
        registered_files.extend(
            reg_main.process_patient(
                patient_id=patient_id,
                preop_dicom_dir=preop_dir,
                postop_dicom_dir=postop_dir,
                out_dir=out_dir,
                mni_template=mni_template,
                mni_annotation=mni_annotation,
                greedy_bin=greedy_bin,
                dcm2niix_bin=dcm2niix_bin,
            )
        )

    if config.get("run_skull_strip", True):
        for nii_path in registered_files:
            reg_main.skull_strip_and_clip(nii_path, hdbet_bin, str(Path(nii_path).parent))


def _copy_or_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_flat_registered_ct(config):
    flat_dir = Path(config["flat_registered_ct_dir"])
    flat_dir.mkdir(parents=True, exist_ok=True)

    supplied_flat = Path(config.get("registered_ct_dir", ""))
    if supplied_flat.exists():
        for src in sorted(supplied_flat.glob("*.nii.gz")):
            _copy_or_link(src.resolve(), flat_dir / src.name)

    registration_root = Path(config["registration_result_root"])
    if registration_root.exists():
        for patient_dir in sorted(path for path in registration_root.iterdir() if path.is_dir()):
            patient_id = patient_dir.name
            candidates = {
                f"{patient_id}.nii.gz": [
                    patient_dir / "preop_in_mni_space_final.nii.gz",
                    patient_dir / "preop_in_mni_space.nii.gz",
                ],
                f"{patient_id}-1.nii.gz": [
                    patient_dir / "postop_in_mni_space_final.nii.gz",
                    patient_dir / "postop_in_mni_space.nii.gz",
                ],
            }
            for out_name, paths in candidates.items():
                src = next((path for path in paths if path.exists()), None)
                if src is not None:
                    _copy_or_link(src.resolve(), flat_dir / out_name)

    image_count = len(
        [
            path
            for path in flat_dir.glob("*.nii.gz")
            if not path.name.endswith("_seg.nii.gz") and not path.name.endswith("_segmentation.nii.gz")
        ]
    )
    if image_count == 0:
        raise FileNotFoundError(f"No registered NIfTI files found in {flat_dir}")
    print(f"Flat registered CT folder: {flat_dir} ({image_count} images)")
    return flat_dir


def run_auto_segmentation(config, flat_ct_dir):
    if not config.get("run_segmentation", True):
        return

    add_module_path(SEG_DIR)
    from batch_inference import BatchOutputInference

    inferencer = BatchOutputInference(
        model_path=config["segmentation_model_path"],
        device=config["device"],
        base_channels=config["segmentation_base_channels"],
        target_size=tuple(config["segmentation_target_size"]),
    )
    inferencer.infer_all_directories(
        base_dirs=[str(flat_ct_dir)],
        threshold=config["segmentation_threshold"],
        postprocess=config["segmentation_postprocess"],
        min_size=config["segmentation_min_size"],
        output_dir=None,
        summary_path=str(Path(config["output_dir"]) / "segmentation_summary.json"),
    )


def run_feature_extraction(config, flat_ct_dir):
    if not config.get("run_feature_extraction", True):
        return Path(config["feature_csv"])

    prior_main = load_module_from_file("clinical_prior_main", PRIOR_DIR / "main.py")

    prior_main.run(
        {
            "registration_result_root": config["registration_result_root"],
            "data_roots": [str(flat_ct_dir)],
            "output_dir": str(Path(config["feature_csv"]).parent),
            "output_csv_name": Path(config["feature_csv"]).name,
            "resume": config.get("feature_resume", True),
            "max_patients": config.get("max_patients"),
        }
    )
    return Path(config["feature_csv"])


def load_clinical_table(path):
    if path is None or str(path).strip() == "":
        return None
    if not Path(path).exists():
        print(f"Clinical CSV not found, PADN-only prediction will be generated: {path}")
        return None
    df = pd.read_csv(path)
    require_columns(df, ["patient_id"], "Clinical CSV")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    return df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)


def predict_clinical_threshold_probs(clinical_df, clinical_model_path):
    model_pack = joblib.load(clinical_model_path)
    features = list(model_pack["features"])
    require_columns(clinical_df, features, "Clinical CSV")

    x = clinical_df[features].astype(float).copy()
    if x.isna().any().any():
        missing = x.columns[x.isna().any()].tolist()
        raise ValueError(f"Clinical CSV has missing values in model features: {missing}")

    x_scaled = pd.DataFrame(
        model_pack["scaler"].transform(x),
        columns=features,
        index=clinical_df.index,
    )
    class_probs = model_pack["result"].model.predict(model_pack["result"].params, exog=x_scaled)
    return ordinal_class_probs_to_threshold_probs(class_probs)


def build_padn_input_df(feature_df, clinical_df=None):
    feature_df = feature_df.copy()
    feature_df["patient_id"] = feature_df["patient_id"].astype(str)
    if clinical_df is None:
        merged = feature_df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
        merged["y"] = 0
        merged["center"] = ""
        merged["flat_folder_mode"] = True
        return merged

    clinical_ids = clinical_df[["patient_id"]].copy()
    clinical_ids["patient_id"] = clinical_ids["patient_id"].astype(str)
    merged = clinical_ids.merge(feature_df, on="patient_id", how="inner")
    if merged.empty:
        raise ValueError("No matched patient_id between clinical CSV and image feature CSV")
    merged["y"] = 0
    merged["center"] = ""
    merged["flat_folder_mode"] = True
    return merged


def predict_padn_threshold_probs(padn_df, flat_ct_dir, config):
    add_module_path(PADN_DIR)
    import torch
    from torch.utils.data import DataLoader
    from torch.amp import autocast
    import Main as padn_main

    checkpoint_path = Path(config["padn_checkpoint_path"])
    require_files([checkpoint_path])

    device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    dataset_df = padn_df.copy()
    dataset_df.attrs["flat_folder_mode"] = True
    dataset = padn_main.CTDataset(
        dataset_df,
        ct_data_dir=str(flat_ct_dir),
        transform=None,
        target_shape=tuple(config["padn_target_shape"]),
        patient_template_root=config["registration_result_root"],
        use_region_attention=config["padn_use_region_attention"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["padn_batch_size"],
        shuffle=False,
        num_workers=config["padn_num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = padn_main.DualChannelPredictor(
        dropout=config["padn_dropout"],
        use_region_attention=config["padn_use_region_attention"],
        model_mode=config["padn_model_mode"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    padn_main.load_model_state_compat(model, checkpoint)
    model.eval()

    all_probs = []
    amp_context = autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        for pre_ct, post_ct, _labels, volumes, pre_masks, post_masks in loader:
            pre_ct = pre_ct.to(device)
            post_ct = post_ct.to(device)
            volumes = volumes.to(device)
            pre_masks = pre_masks.to(device)
            post_masks = post_masks.to(device)
            with amp_context:
                outputs, _weights = model(
                    pre_ct,
                    post_ct,
                    volumes,
                    pre_region_masks=pre_masks,
                    post_region_masks=post_masks,
                    enable_region_prior=not config["padn_disable_all_priors"],
                    disable_all_priors=config["padn_disable_all_priors"],
                )
            probs, _preds = padn_main.decode_ordinal_predictions(outputs)
            all_probs.append(probs.cpu().numpy())

    if not all_probs:
        raise RuntimeError("PADN inference produced no prediction")
    return np.vstack(all_probs)


def predict_fusion_threshold_probs(padn_probs, clinical_probs, fusion_model_path):
    fusion_pack = joblib.load(fusion_model_path)
    stackers = fusion_pack["stackers"]
    if padn_probs.shape != clinical_probs.shape:
        raise ValueError(f"PADN and clinical probability shapes differ: {padn_probs.shape} vs {clinical_probs.shape}")

    fused = []
    for k in range(6):
        x = np.column_stack([padn_probs[:, k], clinical_probs[:, k]])
        fused.append(stackers[k].predict_proba(x)[:, 1])
    return np.column_stack(fused)


def build_final_prediction_table(
    patient_ids,
    padn_threshold_probs,
    clinical_threshold_probs,
    fused_threshold_probs,
):
    patient_ids = pd.Series(patient_ids).astype(str).reset_index(drop=True)
    outputs = {
        "padn": np.asarray(padn_threshold_probs, dtype=float),
    }
    if clinical_threshold_probs is not None:
        outputs["clinical"] = np.asarray(clinical_threshold_probs, dtype=float)
    if fused_threshold_probs is not None:
        outputs["fusion"] = np.asarray(fused_threshold_probs, dtype=float)

    out = pd.DataFrame({"patient_id": patient_ids})
    for prefix, probs in outputs.items():
        out[f"{prefix}_pred_mRS"] = threshold_probs_to_pred_label(probs)
        out[f"{prefix}_prob_poor_outcome"] = probs[:, 2]
        for k in range(6):
            out[f"{prefix}_prob_mRS_gt_{k}"] = probs[:, k]
    return out


def run(config):
    check_conda_env()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_registration_from_dicom(config)
    flat_ct_dir = prepare_flat_registered_ct(config)
    run_auto_segmentation(config, flat_ct_dir)
    feature_csv = run_feature_extraction(config, flat_ct_dir)

    clinical_df = load_clinical_table(config.get("clinical_csv"))
    feature_df = pd.read_csv(feature_csv)
    require_columns(feature_df, ["patient_id"], "Feature CSV")

    padn_df = build_padn_input_df(feature_df, clinical_df)
    patient_ids = padn_df["patient_id"].astype(str).reset_index(drop=True)

    padn_probs = predict_padn_threshold_probs(padn_df, flat_ct_dir, config)
    clinical_probs = None
    fusion_probs = None
    if clinical_df is not None:
        clinical_for_matched = pd.DataFrame({"patient_id": patient_ids}).merge(
            clinical_df,
            on="patient_id",
            how="left",
        )
        clinical_probs = predict_clinical_threshold_probs(clinical_for_matched, config["clinical_model_path"])
        fusion_probs = predict_fusion_threshold_probs(padn_probs, clinical_probs, config["fusion_model_path"])
    else:
        print("No clinical CSV was provided. Skipping clinical and fusion models.")

    final_df = build_final_prediction_table(patient_ids, padn_probs, clinical_probs, fusion_probs)
    output_csv = output_dir / "PADN_clinical_fusion_predictions.csv"
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved final predictions: {output_csv}")
    return final_df


if __name__ == "__main__":
    CONFIG = {
        "output_dir": str(BASE_DIR / "outputs" / "full_prediction"),
        # Leave this as "" if clinical data is not uploaded. The script will
        # output PADN predictions only.
        "clinical_csv": "",

        # The default workflow starts from unregistered DICOM folders and
        # creates individualized atlases during registration.
        "registered_ct_dir": "",
        "flat_registered_ct_dir": str(BASE_DIR / "outputs" / "full_prediction" / "registered_ct_flat"),

        # Set run_registration=True for raw DICOM folders:
        # input_dicom_root/<patient_id>/ and input_dicom_root/<patient_id>-1/
        "run_registration": True,
        "input_dicom_root": "/path/to/input_dicom_root",
        "registration_result_root": str(REG_DIR / "result_v3"),
        "greedy_bin": str(REG_DIR / "tools/itksnap-4.4.0-20250909-Linux-x86_64/bin/greedy"),
        "dcm2niix_bin": "/home/yinpengzhan/miniconda3/envs/env/fsl/pkgs/dcm2niix-1.0.20250506-h84d6215_0/bin/dcm2niix",
        "hdbet_bin": "/home/yinpengzhan/miniconda3/envs/env/bin/hd-bet",
        "mni_template": str(REG_DIR / "template_with_skull_MNI_space_1mm.nii.gz"),
        "mni_annotation": str(REG_DIR / "mni_vascular_territories.nii.gz"),
        "run_skull_strip": True,

        "run_segmentation": True,
        "segmentation_model_path": str(SEG_DIR / "best.pth"),
        "segmentation_base_channels": 32,
        "segmentation_target_size": (182, 218, 182),
        "segmentation_threshold": 0.5,
        "segmentation_postprocess": False,
        "segmentation_min_size": 0,

        "run_feature_extraction": True,
        "feature_csv": str(BASE_DIR / "outputs" / "full_prediction" / "features.csv"),
        "feature_resume": True,
        "max_patients": None,

        "padn_checkpoint_path": str(PADN_DIR / "best_kappa.pth"),
        "padn_target_shape": (182, 218, 182),
        "padn_batch_size": 1,
        "padn_num_workers": 0,
        "padn_dropout": 0.6,
        "padn_use_region_attention": True,
        "padn_model_mode": "prepost_prior_attention",
        "padn_disable_all_priors": True,

        "clinical_model_path": str(CLINICAL_DIR / "clinical_ordinal_model.joblib"),
        "fusion_model_path": str(CLINICAL_DIR / "fusion_stacking_model.joblib"),
        "device": "cuda",
    }

    run(CONFIG)
