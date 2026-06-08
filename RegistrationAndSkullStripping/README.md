# CT Registration and Skull Stripping Pipeline

This folder contains a CT registration pipeline for paired preoperative and postoperative DICOM studies. The script converts DICOM images to NIfTI, registers each CT scan to an MNI-space template with Greedy, maps an MNI vascular-territory annotation into the registered CT grid, and optionally applies CT windowing plus HD-BET skull stripping.

## Contents

- `main.py`: main registration pipeline.
- `template_with_skull_MNI_space_1mm.nii.gz`: fixed MNI-space CT template.
- `mni_vascular_territories.nii.gz`: MNI-space vascular-territory annotation.
- `tools/itksnap-4.4.0-20250909-Linux-x86_64/`: bundled ITK-SNAP tools, including `greedy`.

## Requirements

- Linux x86_64.
- Python 3.10 or newer.
- Python packages: `numpy`, `nibabel`.
- `dcm2niix` for DICOM-to-NIfTI conversion.
- `HD-BET` for skull stripping.
- Greedy registration, bundled in this folder under `tools/`.

The script was originally run in a Conda environment. Before running, update the paths at the end of `main.py` if your `dcm2niix`, `hd-bet`, input folder, or output folder are different.

## Input Layout

The input root should contain paired folders. A postoperative folder is expected to use the same patient ID as the preoperative folder plus a `-1` suffix.

Example:

```text
input_dicom_root/
  210927019L/
  210927019L-1/
  220511223L/
  220511223L-1/
```

## Configuration

Edit the variables in `main()` near the end of `main.py`:

```python
DCM2NIIX_BIN = Path("/path/to/dcm2niix")
HDBET_BIN = Path("/path/to/hd-bet")
INPUT_ROOT = Path("/path/to/input_dicom_root")
OUTPUT_ROOT = base_dir / "result_v3"
```

The template, annotation, and bundled Greedy binary are resolved relative to the script folder.

## Run

```bash
python main.py
```

## Outputs

For each patient, the output folder contains:

- `preop_in_mni_space.nii.gz`
- `postop_in_mni_space.nii.gz`
- `individualized_annotation_in_preop_mni_affine.nii.gz`
- `individualized_annotation_in_postop_mni_affine.nii.gz`
- affine matrices and deformation fields
- optional `_window`, `_bet`, and `_final` images after CT windowing and HD-BET

## Notes

This repository does not include patient DICOM data. Keep clinical imaging data outside the repository unless it has been explicitly approved for public release.
