import re
from pathlib import Path

file_path = "/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/orchestration.py"

with open(file_path, "r") as f:
    content = f.read()

# Replace the directory creation blocks
creation_block_old = """    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)"""

creation_block_new = """    plots_dir = results_dir / "plots"
    data_dir = results_dir / "data"
    metrics_dir = results_dir / "metrics"
    models_dir = results_dir / "models"
    null_dir = results_dir / "null"
    reports_dir = results_dir / "reports"
    importance_dir = results_dir / "importance"
    
    for d in [results_dir, plots_dir, data_dir, metrics_dir, models_dir, null_dir, reports_dir, importance_dir]:
        ensure_dir(d)"""

content = content.replace(creation_block_old, creation_block_new)

mapper = {
    r'results_dir \/ "included_subjects\.tsv"': 'reports_dir / "included_subjects.tsv"',
    r'results_dir \/ "excluded_subjects\.tsv"': 'reports_dir / "excluded_subjects.tsv"',
    r'results_dir \/ "reproducibility_info\.json"': 'reports_dir / "reproducibility_info.json"',
    
    r'results_dir \/ f"best_params_\{model_name\}\.jsonl"': 'models_dir / f"best_params_{model_name}.jsonl"',
    r'results_dir \/ f"best_params_\{model_type\}\.tsv"': 'models_dir / f"best_params_{model_type}.tsv"',
    
    r'results_dir \/ "loso_predictions\.tsv"': 'data_dir / "loso_predictions.tsv"',
    r'results_dir \/ "cv_predictions\.tsv"': 'data_dir / "cv_predictions.tsv"',
    r'results_dir \/ "loso_indices\.tsv"': 'data_dir / "loso_indices.tsv"',
    r'results_dir \/ "cv_indices\.tsv"': 'data_dir / "cv_indices.tsv"',
    r'results_dir \/ "baseline_predictions\.tsv"': 'data_dir / "baseline_predictions.tsv"',
    r'results_dir \/ "prediction_intervals\.tsv"': 'data_dir / "prediction_intervals.tsv"',
    
    r'results_dir \/ "per_subject_correlations\.tsv"': 'metrics_dir / "per_subject_correlations.tsv"',
    r'results_dir \/ "per_subject_errors\.tsv"': 'metrics_dir / "per_subject_errors.tsv"',
    r'results_dir \/ "pooled_metrics\.json"': 'metrics_dir / "pooled_metrics.json"',
    r'results_dir \/ "per_subject_metrics\.tsv"': 'metrics_dir / "per_subject_metrics.tsv"',
    r'results_dir \/ "calibration_data\.json"': 'metrics_dir / "calibration_data.json"',
    r'results_dir \/ "model_comparison\.tsv"': 'metrics_dir / "model_comparison.tsv"',
    r'results_dir \/ "model_comparison_summary\.json"': 'metrics_dir / "model_comparison_summary.json"',
    r'results_dir \/ "incremental_validity\.tsv"': 'metrics_dir / "incremental_validity.tsv"',
    r'results_dir \/ "incremental_validity_summary\.json"': 'metrics_dir / "incremental_validity_summary.json"',
    r'results_dir \/ "per_subject_uncertainty\.tsv"': 'metrics_dir / "per_subject_uncertainty.tsv"',
    r'results_dir \/ "uncertainty_metrics\.json"': 'metrics_dir / "uncertainty_metrics.json"',

    r'results_dir \/ f"loso_null_\{model_name\}\.npz"': 'null_dir / f"loso_null_{model_name}.npz"',
    r'results_dir \/ f"loso_null_\{model_type\}\.npz"': 'null_dir / f"loso_null_{model_type}.npz"',
    r'results_dir \/ f"cv_null_\{model_name\}\.npz"': 'null_dir / f"cv_null_{model_name}.npz"',
    r'results_dir \/ f"cv_null_\{model_type\}\.npz"': 'null_dir / f"cv_null_{model_type}.npz"',
    r'results_dir \/ f"classification_null_\{model_type\}\.npz"': 'null_dir / f"classification_null_{model_type}.npz"',
    r'results_dir \/ "within_subject_null\.npz"': 'null_dir / "within_subject_null.npz"',

    r'results_dir \/ "permutation_importance\.tsv"': 'importance_dir / "permutation_importance.tsv"',
    r'results_dir \/ "permutation_importance_by_group_band\.tsv"': 'importance_dir / "permutation_importance_by_group_band.tsv"',
    r'results_dir \/ "permutation_importance_by_group_band_roi\.tsv"': 'importance_dir / "permutation_importance_by_group_band_roi.tsv"',
    r'results_dir \/ "shap_importance\.tsv"': 'importance_dir / "shap_importance.tsv"',
    r'results_dir \/ "shap_importance_by_group_band\.tsv"': 'importance_dir / "shap_importance_by_group_band.tsv"',
    r'results_dir \/ "shap_importance_by_group_band_roi\.tsv"': 'importance_dir / "shap_importance_by_group_band_roi.tsv"',
}

for old, new in mapper.items():
    content = re.sub(old, new, content)

with open(file_path, "w") as f:
    f.write(content)
print("Updated orchestration.py")
