# Auto Segmentation

This folder contains a 3D CT segmentation inference pipeline. It loads a trained 3D U-Net checkpoint, segments `.nii.gz` CT volumes, and writes single-label segmentation maps with a `_seg.nii.gz` suffix.

## Contents

- `batch_inference.py`: batch inference entry point.
- `model.py`: 3D U-Net model definition and label-map postprocessing.
- `best.pth`: trained model checkpoint.

## Requirements

- Python 3.10 or newer.
- PyTorch.
- NumPy.
- SciPy.
- SimpleITK.
- tqdm.

Install the Python packages in your preferred environment before running the script.

## Input

The input folders should contain 3D NIfTI files:

```text
input_nifti_folder/
  case_001.nii.gz
  case_002.nii.gz
```

Files that already end with `_seg.nii.gz` or `_segmentation.nii.gz` are ignored.

## Configuration

Edit the options at the end of `batch_inference.py`:

```python
INPUT_DIRS = [
    "/path/to/input_nifti_folder",
]

MODELS = [
    {
        "name": "best",
        "model_path": str(BASE_DIR / "best.pth"),
        "output_suffix": None,
    },
]

OUTPUT_SEG_DIR = None
SUMMARY_DIR = str(BASE_DIR / "reports")
```

Set `OUTPUT_SEG_DIR` to `None` to write segmentations next to each input image. Set it to a folder path to collect outputs in one location.

## Run

```bash
python batch_inference.py
```

## Outputs

For each input image, the script writes:

- `<case_id>_seg.nii.gz`: predicted label map.
- `<input_folder>_inference_report.json`: per-folder inference report.
- `reports/<model_name>_summary.json`: summary report.

The output labels are:

- `1`: IVH.
- `2`: SAH.
- `3`: Ventricle.

## Missing Dependencies Checked

The original inference script imported model code from `train.py`, but that file was not present in this upload folder. The required model definition and postprocessing code have been moved into `model.py`, so the upload folder is self-contained apart from the Python packages listed above.
