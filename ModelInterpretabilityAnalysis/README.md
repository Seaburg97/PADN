# Model Interpretability Analysis

This folder contains case-level and group-level interpretability analysis for the PADN model.

The script reads PADN prediction and contribution CSV files, loads the trained PADN checkpoint, generates Grad-CAM visualizations, and summarizes prior-branch and attention behavior.

## Upstream Requirements

Run the upstream modules before this analysis:

1. `RegistrationAndSkullStripping`
   - Produces individualized patient atlases.
   - Default path used by this script:

```text
../RegistrationAndSkullStripping/result_v3/
```

Each patient folder should contain:

```text
individualized_annotation_in_preop_mni_affine.nii.gz
individualized_annotation_in_postop_mni_affine.nii.gz
```

2. `PADN`
   - Produces the trained checkpoint, prediction CSV files, and contribution CSV files.
   - Default path used by this script:

```text
../PADN/outputs/dl_models/
```

Required PADN files:

```text
best_kappa.pth
config.json
external_efy_predictions_best_kappa.csv
external_efy_contributions_best_kappa.csv
external_ay2_predictions_best_kappa.csv
external_ay2_contributions_best_kappa.csv
external_th_predictions_best_kappa.csv
external_th_contributions_best_kappa.csv
```

The contribution CSV files must include:

```text
image_logit_0 ... image_logit_5
pre_prior_logit_0 ... pre_prior_logit_5
post_prior_logit_0 ... post_prior_logit_5
pre_prior_branch_ACA ... pre_prior_branch_IVH
post_prior_branch_ACA ... post_prior_branch_IVH
pre_prior_attention_attention_weight_ACA ... pre_prior_attention_attention_weight_IVH
post_prior_attention_attention_weight_ACA ... post_prior_attention_attention_weight_IVH
```

## Local Data Layout

Clinical and hemorrhage-volume feature CSV files should be placed in:

```text
data/features/
```

Expected files:

```text
featuresefy.csv
featuresay2.csv
featuresth.csv
```

Registered CT files should be placed in:

```text
data/registered_ct/
```

Expected dataset folders:

```text
data/registered_ct/efy/
data/registered_ct/ay2/
data/registered_ct/th/
```

For each patient, the script expects:

```text
<patient_id>.nii.gz
<patient_id>-1.nii.gz
```

## Dependencies

Python packages:

```text
matplotlib
numpy
pandas
nibabel
torch
scipy
```

The script imports the PADN model and dataset definitions from:

```text
../PADN/Main.py
```

## Usage

Edit the options at the top of `main.py` if needed:

```text
DATASET_NAME
CHECKPOINT_NAME
RUN_ALL_CASES
CASE_IDS
GRADCAM_TARGET_MODE
SAVE_NIFTI
SAVE_AXIAL_PNGS
```

Then run:

```bash
python main.py
```

Outputs are saved to:

```text
outputs/
```

## Main Outputs

```text
selected_cases.csv
image_vs_full_summary.csv
case_outputs.csv
case_level/<patient_id>/gradcam_3d.png
case_level/<patient_id>/prior_scores.png
case_level/<patient_id>/attention_weights.png
group_level/*.png
```
