# PADN Prediction Code

This repository contains the code used for image preprocessing, individualized vascular territory mapping, PADN prediction, clinical prediction, and PADN-clinical fusion prediction.

Each subfolder contains its own `README.md` with module-specific usage notes. This top-level README only describes the overall workflow and how the folders fit together.

## Model Files and Registration Tools

Large trained model files, registration tools, templates, and related resources are available from Google Drive:

<https://drive.google.com/drive/folders/1Nlues7rmcTzgHkqkHtFooKLE3Kc7l32g>

After downloading, place the files in the corresponding module folders as described in each subfolder README.

Typical model/resource files include:

- `PADN/best_kappa.pth`
- `AutoSegmentation/best.pth`
- `ClinicalModelAndFusionModel/clinical_ordinal_model.joblib`
- `ClinicalModelAndFusionModel/fusion_stacking_model.joblib`
- registration tools, templates, and atlas files under `RegistrationAndSkullStripping/`

## Folder Overview

- `RegistrationAndSkullStripping`: DICOM-to-NIfTI conversion, MNI-space registration, skull stripping, deformable registration, and individualized vascular territory atlas generation.
- `AutoSegmentation`: automatic segmentation of IVH, SAH, and ventricles from CT volumes.
- `ClinicalPriorConstruction`: extraction of regional hemorrhage burden and image-derived prior features using CT images, segmentation masks, and individualized vascular territory masks.
- `PADN`: paired preoperative/postoperative CT deep learning prognostic model.
- `ClinicalModelAndFusionModel`: clinical ordinal regression model and PADN-clinical stacking fusion model.
- `ModelInterpretabilityAnalysis`: model interpretation and visualization scripts.
- `OtherTools`: auxiliary statistical analysis and plotting scripts.
- `run_full_prediction.py`: top-level prediction entry point that links preprocessing, segmentation, feature extraction, PADN prediction, and optional clinical/fusion prediction.



Main Python dependencies:

- Python 3.10 or newer
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- statsmodels
- joblib
- NiBabel
- SimpleITK
- matplotlib
- seaborn
- tqdm


External tools used by the image preprocessing workflow:

- `dcm2niix` for DICOM-to-NIfTI conversion
- Greedy registration package for affine and deformable registration
- HD-BET for skull stripping
- ITK-SNAP for atlas editing and visual quality control

## Input Data

The default workflow assumes raw, unregistered DICOM input.

Expected DICOM folder layout:

```text
input_dicom_root/
  P001/
  P001-1/
  P002/
  P002-1/
```

Where:

- `P001/` is the preoperative CT DICOM folder.
- `P001-1/` is the early postoperative CT DICOM folder for the same patient.

The preprocessing pipeline generates registered NIfTI files and individualized vascular territory masks for each patient. PADN and the image-prior feature extraction step use these patient-specific masks.

## Optional Clinical Data

Clinical data are optional.

If clinical data are provided, the CSV must contain `patient_id` and the variables required by the saved clinical model. For the provided model, the selected variables are:

```text
patient_id
Clipping
Hunt-Hess_score
mFS_score
SEBES_score
Age
```

If no clinical data are uploaded, leave `CONFIG["clinical_csv"]` as an empty string `""` in `run_full_prediction.py`. The script will skip the clinical model and fusion model, and will output PADN predictions only.

## Running the Full Pipeline

Edit the `CONFIG` block at the end of `run_full_prediction.py`.

Example with clinical data:

```python
"clinical_csv": "/path/to/clinical_general_info.csv",
"input_dicom_root": "/path/to/input_dicom_root",
"run_registration": True,
"run_segmentation": True,
"run_feature_extraction": True,
"device": "cuda",
```

Example without clinical data:

```python
"clinical_csv": "",
"input_dicom_root": "/path/to/input_dicom_root",
"run_registration": True,
"run_segmentation": True,
"run_feature_extraction": True,
"device": "cuda",
```

Run:

```bash
python run_full_prediction.py
```

The script performs the following steps:

1. Convert preoperative and postoperative DICOM scans to NIfTI.
2. Register each CT scan to the MNI CT template.
3. Generate individualized vascular territory masks for each patient.
4. Run automatic segmentation.
5. Extract image-prior features.
6. Run PADN prediction.
7. If clinical data are provided, run the clinical model.
8. If clinical data are provided, run the PADN-clinical fusion model.

Final predictions are saved to:

```text
outputs/full_prediction/PADN_clinical_fusion_predictions.csv
```

Main output columns:

- `patient_id`
- `padn_pred_mRS`
- `padn_prob_poor_outcome`
- `clinical_pred_mRS`, generated only when clinical data are provided
- `clinical_prob_poor_outcome`, generated only when clinical data are provided
- `fusion_pred_mRS`, generated only when clinical data are provided
- `fusion_prob_poor_outcome`, generated only when clinical data are provided
- `*_prob_mRS_gt_0` to `*_prob_mRS_gt_5`

`prob_poor_outcome` corresponds to the probability of `mRS > 2`.

## Referenced Methods and Third-Party Code

This codebase uses and cites external methods, tools, and resources for image preprocessing and registration:

- The CT template used for common anatomical space registration is based on the publicly available high-resolution CT brain template described by Muschelli [1].
- The image registration workflow uses cross-correlation-based affine and deformable registration concepts from symmetric diffeomorphic registration literature [2], implemented here with the Greedy registration tool.
- Skull stripping is performed with HD-BET, which is based on the deep learning brain extraction method described by Isensee et al. [3].

The trained model files and the registration tools/resources required to run this code are provided in the Google Drive folder listed above.

## Notes

- Patient IDs must be consistent across DICOM folders, clinical CSV rows, and generated registration outputs.
- The postoperative DICOM folder must use the `<patient_id>-1` naming convention.
- PADN requires patient-specific atlas files generated by the registration pipeline:

```text
RegistrationAndSkullStripping/result_v3/<patient_id>/
  individualized_annotation_in_preop_mni_affine.nii.gz
  individualized_annotation_in_postop_mni_affine.nii.gz
```

- If segmentations have already been generated, set `run_segmentation=False`.
- If image-prior features have already been generated, set `run_feature_extraction=False` and point `feature_csv` to the existing feature CSV.
- This repository does not include patient data. Before uploading or sharing imaging and clinical data, ensure that all records have been de-identified and approved for sharing.

## References

1. Muschelli J. A Publicly Available, High Resolution, Unbiased CT Brain Template. In: Lesot MJ, Vieira S, Reformat MZ, et al., eds. *Information Processing and Management of Uncertainty in Knowledge-Based Systems*. Springer International Publishing; 2020:358-366. doi:10.1007/978-3-030-50153-2_27
2. Avants BB, Epstein CL, Grossman M, Gee JC. Symmetric Diffeomorphic Image Registration with Cross-Correlation: Evaluating Automated Labeling of Elderly and Neurodegenerative Brain. *Med Image Anal*. 2008;12(1):26-41. doi:10.1016/j.media.2007.06.004
3. Isensee F, Schell M, Pflueger I, et al. Automated brain extraction of multisequence MRI using artificial neural networks. *Hum Brain Mapp*. 2019;40(17):4952-4964. doi:10.1002/hbm.24750
