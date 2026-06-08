#!/usr/bin/env python3
"""
CT registration pipeline.

The pipeline registers preoperative and postoperative CT scans to an MNI-space
template, maps an MNI annotation into each registered CT grid, and optionally
applies CT windowing plus HD-BET skull stripping.
"""

import glob
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def run_cmd(cmd, desc=""):
    print(f"\n{'=' * 60}")
    if desc:
        print(f"[{desc}]")
    print("CMD:", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[:800])
    if result.returncode != 0:
        print("STDERR:", result.stderr[:800])
        raise RuntimeError(f"Command failed with return code {result.returncode}: {desc}")
    return result


def convert_dicom_to_nifti(dicom_dir, output_dir, prefix, dcm2niix_bin):
    os.makedirs(output_dir, exist_ok=True)
    run_cmd(
        [dcm2niix_bin, "-z", "y", "-o", output_dir, "-f", prefix, dicom_dir],
        f"DICOM to NIfTI: {prefix}",
    )
    nii_files = sorted(glob.glob(os.path.join(output_dir, f"{prefix}*.nii.gz")))
    if not nii_files:
        raise FileNotFoundError(f"dcm2niix did not create: {output_dir}/{prefix}*.nii.gz")
    if len(nii_files) > 1:
        print(f"  Warning: multiple NIfTI files found; using: {nii_files[0]}")
    return nii_files[0]


def affine_registration(moving_nii, fixed_nii, out_mat, greedy_bin):
    run_cmd(
        [
            greedy_bin,
            "-d",
            "3",
            "-a",
            "-m",
            "NCC",
            "2x2x2",
            "-i",
            fixed_nii,
            moving_nii,
            "-o",
            out_mat,
            "-ia-image-centers",
            "-n",
            "100x50x10",
        ],
        f"Affine registration: {Path(moving_nii).name} to MNI",
    )


def reslice_image(moving_nii, fixed_nii, out_nii, transforms, greedy_bin, label=False):
    cmd = [greedy_bin, "-d", "3", "-rf", fixed_nii]
    if label:
        cmd += ["-ri", "LABEL", "0.2vox"]
    cmd += ["-rm", moving_nii, out_nii]
    cmd += ["-r"] + transforms
    run_cmd(cmd, f"Reslice: {Path(moving_nii).name} to {Path(out_nii).name}")


def deformable_registration(
    moving_nii, fixed_nii, affine_mat, out_warp, out_inv_warp, greedy_bin
):
    run_cmd(
        [
            greedy_bin,
            "-d",
            "3",
            "-m",
            "NCC",
            "2x2x2",
            "-i",
            fixed_nii,
            moving_nii,
            "-it",
            affine_mat,
            "-o",
            out_warp,
            "-oinv",
            out_inv_warp,
            "-n",
            "100x50x10",
        ],
        f"Deformable registration: {Path(moving_nii).name} to MNI",
    )


def clip_ct_image(nii_path, out_path, min_hu=0, max_hu=100):
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    data = np.clip(data, min_hu, max_hu)
    clipped_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(clipped_img, out_path)
    return out_path


def process_patient(
    patient_id,
    preop_dicom_dir,
    postop_dicom_dir,
    out_dir,
    mni_template,
    mni_annotation,
    greedy_bin,
    dcm2niix_bin,
):
    print(f"\n{'#' * 70}")
    print(f"# Patient: {patient_id}")
    print(f"{'#' * 70}")

    os.makedirs(out_dir, exist_ok=True)
    nifti_dir = os.path.join(out_dir, "nifti")

    preop_nii = convert_dicom_to_nifti(preop_dicom_dir, nifti_dir, "preop", dcm2niix_bin)
    postop_nii = convert_dicom_to_nifti(
        postop_dicom_dir, nifti_dir, "postop", dcm2niix_bin
    )

    preop_affine_mat = os.path.join(out_dir, "preop_to_mni_affine.mat")
    preop_mni_nii = os.path.join(out_dir, "preop_in_mni_space.nii.gz")

    if not os.path.exists(preop_mni_nii):
        affine_registration(preop_nii, mni_template, preop_affine_mat, greedy_bin)
        reslice_image(preop_nii, mni_template, preop_mni_nii, [preop_affine_mat], greedy_bin)
    else:
        print(f"Skipping existing file: {preop_mni_nii}")

    preop_warp = os.path.join(out_dir, "preop_to_mni_warp.nii.gz")
    preop_inv_warp = os.path.join(out_dir, "preop_to_mni_inv_warp.nii.gz")

    if not os.path.exists(preop_warp):
        deformable_registration(
            preop_nii, mni_template, preop_affine_mat, preop_warp, preop_inv_warp, greedy_bin
        )
    else:
        print(f"Skipping existing file: {preop_warp}")

    annotation_preop_mni = os.path.join(
        out_dir, "individualized_annotation_in_preop_mni_affine.nii.gz"
    )

    if not os.path.exists(annotation_preop_mni):
        reslice_image(
            mni_annotation,
            preop_mni_nii,
            annotation_preop_mni,
            transforms=[preop_inv_warp],
            greedy_bin=greedy_bin,
            label=True,
        )
    else:
        print(f"Skipping existing file: {annotation_preop_mni}")

    postop_affine_mat = os.path.join(out_dir, "postop_to_mni_affine.mat")
    postop_mni_nii = os.path.join(out_dir, "postop_in_mni_space.nii.gz")

    if not os.path.exists(postop_mni_nii):
        affine_registration(postop_nii, mni_template, postop_affine_mat, greedy_bin)
        reslice_image(
            postop_nii, mni_template, postop_mni_nii, [postop_affine_mat], greedy_bin
        )
    else:
        print(f"Skipping existing file: {postop_mni_nii}")

    postop_warp = os.path.join(out_dir, "postop_to_mni_warp.nii.gz")
    postop_inv_warp = os.path.join(out_dir, "postop_to_mni_inv_warp.nii.gz")

    if not os.path.exists(postop_warp):
        deformable_registration(
            postop_nii,
            mni_template,
            postop_affine_mat,
            postop_warp,
            postop_inv_warp,
            greedy_bin,
        )
    else:
        print(f"Skipping existing file: {postop_warp}")

    annotation_postop_mni = os.path.join(
        out_dir, "individualized_annotation_in_postop_mni_affine.nii.gz"
    )

    if not os.path.exists(annotation_postop_mni):
        reslice_image(
            mni_annotation,
            postop_mni_nii,
            annotation_postop_mni,
            transforms=[postop_inv_warp],
            greedy_bin=greedy_bin,
            label=True,
        )
    else:
        print(f"Skipping existing file: {annotation_postop_mni}")

    print(f"\nPatient completed: {patient_id}")
    print(f"  Preoperative CT in MNI space: {preop_mni_nii}")
    print(f"  Preoperative annotation: {annotation_preop_mni}")
    print(f"  Postoperative CT in MNI space: {postop_mni_nii}")
    print(f"  Postoperative annotation: {annotation_postop_mni}")

    return [preop_mni_nii, postop_mni_nii]


def skull_strip_and_clip(nii_path, hdbet_bin, out_dir):
    stem = Path(nii_path).name.replace(".nii.gz", "")
    window_out = os.path.join(out_dir, f"{stem}_window.nii.gz")
    bet_out = os.path.join(out_dir, f"{stem}_bet.nii.gz")
    final_path = nii_path.replace(".nii.gz", "_final.nii.gz")

    if not os.path.exists(window_out):
        clip_ct_image(nii_path, window_out)
        print(f"  CT windowing completed: {window_out}")
    else:
        print(f"  Skipping existing CT window file: {window_out}")

    if not os.path.exists(bet_out):
        print(f"\n[HD-BET] {stem}")
        env = os.environ.copy()
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            lib_path = os.path.join(conda_prefix, "lib")
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

        result = subprocess.run(
            [hdbet_bin, "-i", window_out, "-o", bet_out],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"  HD-BET failed: {result.stderr[:400]}")
            return None
    else:
        print(f"  Skipping existing HD-BET file: {bet_out}")

    possible_outputs = [
        bet_out,
        os.path.join(out_dir, f"{stem}.nii.gz"),
        os.path.join(out_dir, f"{stem}_bet.nii.gz"),
    ]
    hdbet_output = next((p for p in possible_outputs if os.path.exists(p)), None)
    if hdbet_output is None:
        print(f"  Warning: HD-BET did not create an output image: {bet_out}")
        return None

    img = nib.load(hdbet_output)
    nib.save(img, final_path)
    print(f"  Skull stripping and CT windowing completed: {final_path}")
    return final_path


def main():
    base_dir = Path(__file__).resolve().parent

    GREEDY_BIN = base_dir / "tools/itksnap-4.4.0-20250909-Linux-x86_64/bin/greedy"
    DCM2NIIX_BIN = Path(
        "/home/yinpengzhan/miniconda3/envs/env/fsl/pkgs/"
        "dcm2niix-1.0.20250506-h84d6215_0/bin/dcm2niix"
    )
    HDBET_BIN = Path("/home/yinpengzhan/miniconda3/envs/env/bin/hd-bet")
    MNI_TEMPLATE = base_dir / "template_with_skull_MNI_space_1mm.nii.gz"
    MNI_ANNOTATION = base_dir / "mni_vascular_territories.nii.gz"
    INPUT_ROOT = Path("/path/to/input_dicom_root")
    OUTPUT_ROOT = base_dir / "result_v3"

    for path in [GREEDY_BIN, DCM2NIIX_BIN, HDBET_BIN, MNI_TEMPLATE, MNI_ANNOTATION]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    # A postoperative folder is expected to have the same ID as the preoperative
    # folder plus a "-1" suffix.
    all_entries = os.listdir(INPUT_ROOT)
    postop_ids = sorted([entry for entry in all_entries if entry.endswith("-1")])

    patients = []
    for postop_id in postop_ids:
        base_id = postop_id[:-2]
        preop_dir = INPUT_ROOT / base_id
        postop_dir = INPUT_ROOT / postop_id
        if preop_dir.is_dir() and postop_dir.is_dir():
            patients.append((base_id, preop_dir, postop_dir))
        else:
            print(f"Warning: preoperative folder not found, skipping {postop_id}: {preop_dir}")

    print(f"Found {len(patients)} patients:")
    for base_id, preop_dir, postop_dir in patients:
        print(f"  {base_id}: preop={preop_dir} postop={postop_dir}")

    errors = []
    all_registered_nii = []
    for base_id, preop_dir, postop_dir in patients:
        out_dir = OUTPUT_ROOT / base_id
        try:
            registered = process_patient(
                patient_id=base_id,
                preop_dicom_dir=preop_dir,
                postop_dicom_dir=postop_dir,
                out_dir=out_dir,
                mni_template=MNI_TEMPLATE,
                mni_annotation=MNI_ANNOTATION,
                greedy_bin=GREEDY_BIN,
                dcm2niix_bin=DCM2NIIX_BIN,
            )
            all_registered_nii.extend(registered)
        except Exception as exc:
            print(f"\nPatient failed: {base_id}: {exc}")
            errors.append((base_id, str(exc)))

    print(f"\n{'=' * 70}")
    print(f"Registration completed. Success: {len(patients) - len(errors)}, failed: {len(errors)}")

    print(f"\n{'=' * 70}")
    print(f"Starting skull stripping and CT windowing for {len(all_registered_nii)} files")
    for nii_path in all_registered_nii:
        if not os.path.exists(nii_path):
            print(f"  Skipping missing file: {nii_path}")
            continue
        out_dir = str(Path(nii_path).parent)
        try:
            final_path = skull_strip_and_clip(nii_path, HDBET_BIN, out_dir)
            if final_path is None:
                raise RuntimeError("HD-BET did not create the final skull-stripped image")
        except Exception as exc:
            print(f"  Skull stripping failed for {nii_path}: {exc}")

    print(f"\n{'=' * 70}")
    print("All processing completed.")
    if errors:
        for patient_id, error in errors:
            print(f"  [FAILED] {patient_id}: {error}")


if __name__ == "__main__":
    main()
