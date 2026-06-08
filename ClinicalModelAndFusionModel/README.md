# Clinical Model and Fusion Model

This folder contains the clinical ordinal regression model and the DL-clinical stacking fusion pipeline.

The script trains a clinical ordinal model from tabular clinical features, reads PADN prediction CSV files, fits threshold-level stacking models, and evaluates DL-only, clinical-only, and fused predictions.

## Required Inputs

Place clinical feature CSV files in:

```text
data/features/
```

Expected training feature files:

```text
featuresaq.csv
featuresay.csv
featuresfy.csv
featurestl.csv
featuresyjs.csv
```

Expected external test feature files:

```text
featuresth.csv
featuresay2.csv
featuresefy.csv
```

Each feature CSV must contain:

```text
patient_id
mRS
Age
Male
mFS_score
SEBES_score
Acute_hydrocephalus
GCS_score
WFNS_score
Hunt-Hess_score
Posterior_circulation
Size
Hypertension
Clipping
```

## PADN Outputs

By default, this script reads PADN prediction files from:

```text
../PADN/outputs/dl_models/
```

Required PADN files:

```text
train_predictions_best_kappa.csv
val_predictions_best_kappa.csv
Test-Combined_predictions_best_kappa.csv
external_th_predictions_best_kappa.csv
external_ay2_predictions_best_kappa.csv
external_efy_predictions_best_kappa.csv
```

Each PADN prediction CSV must contain:

```text
patient_id
true_label
pred_label
prob_poor_outcome
prob_mrs_gt_0
prob_mrs_gt_1
prob_mrs_gt_2
prob_mrs_gt_3
prob_mrs_gt_4
prob_mrs_gt_5
```

## Dependencies

Python packages:

```text
matplotlib
numpy
pandas
scipy
scikit-learn
statsmodels
```

R is also required for ordinal LASSO:

```text
ordinalNet
```

The script first checks the `RSCRIPT` environment variable, then `Rscript` in `PATH`, then a local `.mamba_envs/r-base-only/bin/Rscript` under the project root.

## Usage

Edit the path and feature-selection options in `main.py` if needed, then run:

```bash
python main.py
```

Results are saved to:

```text
outputs/
```

## Outputs

Main outputs include:

```text
selected_clinical_features.csv
clinical_train_predictions.csv
clinical_val_predictions_for_stacking.csv
stacking_feature_importance.csv
clinical_model_feature_importance.csv
*_model_metrics.csv
*_stacking_predictions.csv
model_comparison.csv
paper_table_*.csv
```
