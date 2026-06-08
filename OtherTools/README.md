# Other Tools

This folder contains auxiliary analysis and plotting scripts used after the main model pipelines.

All scripts use relative paths. Put input files under `data/` and read generated outputs from the upstream folders when noted below.

## Scripts

### `ablation_evaluation.py`

Compares two sets of PADN prediction CSV files with paired statistics.

Inputs:

```text
data/ablation/model_a/*_predictions_best_kappa.csv
data/ablation/model_b/*_predictions_best_kappa.csv
```

Output:

```text
outputs/ablation_model_comparison.csv
```

### `clinical_variable_summary.py`

Builds Excel summary tables for clinical variables across development and external test cohorts.

Inputs:

```text
data/features/featuresaq.csv
data/features/featuresay.csv
data/features/featuresfy.csv
data/features/featurestl.csv
data/features/featuresyjs.csv
data/features/featuresefy.csv
data/features/featuresay2.csv
data/features/featuresth.csv
```

Outputs:

```text
outputs/clinical_variable_summary.xlsx
outputs/clinical_variable_summary_by_development_center.xlsx
```

### `dci_ch_subgroup_analysis.py`

Evaluates DL, clinical, and fusion predictions in DCI and CH subgroups.

Inputs:

```text
../ClinicalModelAndFusionModel/outputs/Test-Combined_stacking_predictions.csv
data/features/featuresth.csv
data/features/featuresay2.csv
data/features/featuresefy.csv
```

Outputs:

```text
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_subgroup_metrics.csv
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_case_details.csv
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_model_comparison_p_values.csv
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_subgroup_metrics.xlsx
```

### `dci_ch_subgroup_plot.py`

Plots DCI and CH subgroup comparison figures from the outputs of `dci_ch_subgroup_analysis.py`.

Inputs:

```text
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_model_comparison_p_values.csv
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_ch_case_details.csv
```

Outputs:

```text
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_subgroup_model_comparison_p_values.png
outputs/dci_ch_subgroup_best_kappa/best_kappa_dci_subgroup_model_comparison_p_values.pdf
outputs/dci_ch_subgroup_best_kappa/best_kappa_ch_subgroup_model_comparison_p_values.png
outputs/dci_ch_subgroup_best_kappa/best_kappa_ch_subgroup_model_comparison_p_values.pdf
```

### `model_performance_bootstrap_plot.py`

Creates bootstrap model-performance tables and summary figures from the clinical/fusion model outputs.

Inputs:

```text
../ClinicalModelAndFusionModel/outputs/*_stacking_predictions.csv
../ClinicalModelAndFusionModel/outputs/model_comparison.csv
```

Outputs:

```text
outputs/paper_table_model_performance_grouped_1000boot.csv
outputs/paper_table_model_performance_grouped_1000boot.xlsx
outputs/model_performance_bootstrap_plot.png
outputs/model_performance_bootstrap_plot.pdf
outputs/model_performance_kappa_mae_auc_comparison.png
outputs/model_performance_kappa_mae_auc_comparison.pdf
outputs/model_performance_integrated_core_metrics.png
outputs/model_performance_integrated_core_metrics.pdf
```

## Dependencies

Python packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
```

## Usage

Run each script directly:

```bash
python ablation_evaluation.py
python clinical_variable_summary.py
python dci_ch_subgroup_analysis.py
python dci_ch_subgroup_plot.py
python model_performance_bootstrap_plot.py
```

No data files are included in this folder.
