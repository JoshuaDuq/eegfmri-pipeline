#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from utils import load_config, get_log_function, PipelinePaths
from utils.io_utils import extract_vas_ratings, get_events_paths, sanitize_for_json
from utils.image_utils import load_signature_weights, voxel_volume
from utils.stats_utils import safe_spearman, cohens_d
from analysis.signature_scoring import (
    determine_scale_factor,
    compute_signature_response,
    compute_validation_metrics,
    score_single_trial_beta,
)
from analysis.glm import convert_bold_to_percent_signal
from analysis.lss import (
    create_lss_events_for_trial,
    extract_trial_info,
    fit_lss_glm_for_trial,
    extract_target_trial_beta,
)


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)


###################################################################
# Condition-Level Scoring
###################################################################


def score_conditions_for_subject(config: Dict, inventory: Dict, subject: str, signature_name: str,
                                 harmonized_root: Path, scores_root: Path, qc_dir: Path,
                                 weights_img: nib.Nifti1Image, mask: np.ndarray) -> bool:
    temp_labels = config['glm']['temp_labels']
    temp_mapping = config['glm'].get('temp_celsius_mapping', {})

    sig_metadata = config.get('signature_metadata', {}).get(signature_name, {})
    apply_scale_factor = sig_metadata.get('requires_beta_scale_factor', True)

    events_paths = get_events_paths(inventory)

    subject_output_dir = PipelinePaths.ensure_dir(scores_root / subject)
    qc_records = []

    score_col = f'{signature_name}_score'

    harmonization_metadata_path = harmonized_root / subject / "harmonization_metadata.json"
    harmonization_metadata = {}
    if harmonization_metadata_path.exists():
        with open(harmonization_metadata_path, 'r') as f:
            harmonization_metadata = json.load(f)

    results = []
    scoring_metadata = {
        'subject': subject,
        'n_temperatures': len(temp_labels),
        'temperatures': {}
    }

    for temp in temp_labels:
        try:
            temp_celsius = temp_mapping.get(temp, np.nan)
            beta_path = harmonized_root / subject / f"beta_{temp}_on{signature_name.upper()}grid.nii.gz"

            if not beta_path.exists():
                continue

            beta_img = nib.load(str(beta_path))

            original_beta_volume = None
            if harmonization_metadata and 'temperatures' in harmonization_metadata:
                temp_meta = harmonization_metadata['temperatures'].get(temp, {})
                original_beta_volume = temp_meta.get('original_voxel_volume_mm3')

            sig_result = compute_signature_response(
                beta_img, weights_img, mask,
                apply_scale_factor=apply_scale_factor,
                original_beta_volume_mm3=original_beta_volume
            )

            if sig_result is None:
                continue

            br_score = sig_result['br_score']

            if not sig_result['br_is_finite']:
                continue

            vas_stats = extract_vas_ratings(events_paths, temp)

            result_row = {
                'subject': subject,
                'temp_label': temp,
                'temp_celsius': temp_celsius,
                score_col: br_score,
                'mean_vas': vas_stats['mean_vas'],
                'std_vas': vas_stats['std_vas'],
                'median_vas': vas_stats['median_vas'],
                'n_trials': vas_stats['n_trials']
            }
            results.append(result_row)

            scoring_metadata['temperatures'][temp] = {
                'temp_celsius': temp_celsius,
                f'{signature_name}_score': br_score,
                f'{signature_name}_score_raw': sig_result['br_score_raw'],
                'scale_factor_applied': sig_result['scale_factor_applied'],
                'scale_factor_value': sig_result['scale_factor_value'],
                'vas_stats': vas_stats,
                'beta_stats': {
                    'mean': sig_result['beta_mean'],
                    'std': sig_result['beta_std'],
                    'min': sig_result['beta_min'],
                    'max': sig_result['beta_max'],
                    'n_voxels': sig_result['n_voxels'],
                    'n_finite': sig_result['n_beta_finite'],
                    'n_nonzero': sig_result['n_beta_nonzero']
                },
                'n_beta_nan': sig_result['n_beta_nan'],
                'n_beta_inf': sig_result['n_beta_inf'],
                'pct_voxels_invalid': sig_result['pct_voxels_invalid'],
                'quality_check_passed': sig_result['quality_check_passed']
            }
            qc_records.append({
                'subject': subject,
                'signature': signature_name,
                'temperature': temp,
                'temp_celsius': temp_celsius,
                'n_voxels': sig_result['n_voxels'],
                'n_beta_finite': sig_result['n_beta_finite'],
                'n_beta_nonzero': sig_result['n_beta_nonzero'],
                'n_beta_nan': sig_result['n_beta_nan'],
                'n_beta_inf': sig_result['n_beta_inf'],
                'pct_voxels_invalid': sig_result['pct_voxels_invalid'],
                'quality_check_passed': sig_result['quality_check_passed'],
                'beta_mean': sig_result['beta_mean'],
                'beta_std': sig_result['beta_std'],
                'br_score': br_score
            })

        except Exception as e:
            log(f"{temp}: {e}", "ERROR")
            continue

    if not results:
        return False

    results_df = pd.DataFrame(results)

    output_path = subject_output_dir / f"level_{signature_name}.tsv"
    results_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')

    if len(results_df) >= 2:
        validation_metrics = compute_validation_metrics(results_df, score_col=score_col)
        scoring_metadata['validation'] = validation_metrics


    metadata_path = subject_output_dir / "scoring_metadata.json"
    sanitized_metadata = sanitize_for_json(scoring_metadata)
    metadata_path.write_text(json.dumps(sanitized_metadata, indent=2))

    if qc_records and qc_dir is not None:
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / f"{subject}_grid_check.tsv"
        pd.DataFrame(qc_records).to_csv(qc_path, sep='\t', index=False)

    return True


###################################################################
# Trial-Level Scoring (LSS)
###################################################################

def compute_discrimination_metrics(trial_df: pd.DataFrame, score_col: str = 'br_score', pain_threshold: float = 100.0) -> Dict[str, float]:
    metrics = {
        'n_trials_total': int(len(trial_df)),
        'n_subjects': int(trial_df['subject'].nunique()) if 'subject' in trial_df.columns else np.nan,
        'spearman_temp_score': float('nan'),
        'spearman_temp_score_p': float('nan'),
        'spearman_temp_score_n': 0,
        'spearman_vas_score': float('nan'),
        'spearman_vas_score_p': float('nan'),
        'spearman_vas_score_n': 0,
        'pain_auc': float('nan'),
        'pain_auc_n': 0,
        'pain_auc_label_positive': 1,
        'mean_score_pain': float('nan'),
        'mean_score_nonpain': float('nan'),
        'cohens_d_pain': float('nan'),
        'pain_rate': float('nan')
    }

    df = trial_df.copy()

    if 'pain_binary' not in df.columns or df['pain_binary'].isna().all():
        if 'vas_rating' in df.columns:
            df['pain_binary'] = (df['vas_rating'] > pain_threshold).astype(float)
        else:
            df['pain_binary'] = np.nan

    if 'temp_celsius' in df.columns and score_col in df.columns:
        temp_series = pd.to_numeric(df['temp_celsius'], errors='coerce')
        spearman_temp = safe_spearman(temp_series, df[score_col])
        metrics['spearman_temp_score'] = spearman_temp['rho']
        metrics['spearman_temp_score_p'] = spearman_temp['p']
        metrics['spearman_temp_score_n'] = spearman_temp['n']

    if 'vas_rating' in df.columns and score_col in df.columns:
        vas_series = pd.to_numeric(df['vas_rating'], errors='coerce')
        spearman_vas = safe_spearman(vas_series, df[score_col])
        metrics['spearman_vas_score'] = spearman_vas['rho']
        metrics['spearman_vas_score_p'] = spearman_vas['p']
        metrics['spearman_vas_score_n'] = spearman_vas['n']

    if 'pain_binary' in df.columns and score_col in df.columns:
        labels = pd.to_numeric(df['pain_binary'], errors='coerce')
        scores = pd.to_numeric(df[score_col], errors='coerce')
        valid = labels.notna() & scores.notna()
        labels_valid = labels[valid]
        scores_valid = scores[valid]
        if len(labels_valid) >= 2 and labels_valid.nunique() > 1:
            metrics['pain_auc'] = float(roc_auc_score(labels_valid, scores_valid))
            metrics['pain_auc_n'] = int(len(labels_valid))
        if len(labels_valid) > 0:
            metrics['pain_rate'] = float(labels_valid.mean())

        pain_scores = scores_valid[labels_valid == 1]
        nopain_scores = scores_valid[labels_valid == 0]
        if len(pain_scores) > 0:
            metrics['mean_score_pain'] = float(np.mean(pain_scores))
        if len(nopain_scores) > 0:
            metrics['mean_score_nonpain'] = float(np.mean(nopain_scores))
        if len(pain_scores) > 1 and len(nopain_scores) > 1:
            metrics['cohens_d_pain'] = float(cohens_d(pain_scores.values, nopain_scores.values))

    return metrics


def score_trials_for_subject(config: Dict, inventory: Dict, subject: str, signature_name: str,
                            signature_weights: nib.Nifti1Image, signature_voxel_volume: float, apply_scale_factor: bool,
                            work_dir: Path, scores_root: Path, qc_dir: Path, condition_threshold: float = 30.0) -> bool:

    temp_mapping = config['glm'].get('temp_celsius_mapping', {})
    tr = config['glm']['tr']
    hrf_model = config['glm']['hrf']['model']
    high_pass_sec = config['glm']['high_pass_sec']
    high_pass_hz = 1.0 / high_pass_sec
    pain_threshold = config.get('behavior', {}).get('vas_pain_threshold', 100.0)

    onset_offset = config['glm'].get('stim_onset_offset_sec', 0.0)
    plateau_duration = config['glm'].get('stim_dur_sec', None)

    subject_output_dir = PipelinePaths.ensure_dir(scores_root / subject)
    score_col = f'{signature_name}_score'

    all_trial_results = []
    qc_records = []
    global_trial_idx = 0

    for run_key in sorted(inventory['runs'].keys()):
        run_data = inventory['runs'][run_key]
        run_num = run_data['run_number']

        if not run_data['complete']:
            continue

        try:
            bold_path = Path(run_data['files']['bold']['path'])
            mask_path = Path(run_data['files']['mask']['path'])
            events_path = Path(run_data['files']['events']['path'])

            confounds_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
            if not confounds_path.exists():
                log("    Confounds not found", "ERROR")
                continue

            confounds = pd.read_csv(confounds_path, sep='\t')

            trial_info = extract_trial_info(
                events_path,
                run_num,
                global_trial_idx,
                onset_offset_sec=onset_offset,
                plateau_duration_sec=plateau_duration,
            )

            if trial_info.empty:
                continue

            n_trials = len(trial_info)

            bold_img = nib.load(str(bold_path))
            mask_img = nib.load(str(mask_path))

            original_bold_volume = voxel_volume(bold_img)

            convert_to_psc = config["glm"].get("convert_to_percent_signal", False)
            if convert_to_psc:
                psc_window_sec = config["glm"].get("percent_signal_baseline_sec", 120.0)
                psc_min_baseline = config["glm"].get("percent_signal_min_baseline", 1e-3)
                bold_img = convert_bold_to_percent_signal(bold_img, tr, baseline_window_sec=psc_window_sec, min_baseline=psc_min_baseline)

            trial_scores = {}
            condition_numbers_raw = []
            condition_numbers_z = []

            for trial_idx in range(n_trials):
                trial_row = trial_info.iloc[trial_idx]
                trial_reg = trial_row['trial_regressor']

                nuisance_events = config['glm'].get('nuisance_events', [])
                lss_events = create_lss_events_for_trial(events_path, trial_idx, trial_info, nuisance_events)

                glm, cond_raw, cond_z = fit_lss_glm_for_trial(bold_img, mask_img, lss_events, confounds, tr, hrf_model, high_pass_hz)

                condition_numbers_raw.append(cond_raw)
                condition_numbers_z.append(cond_z)

                beta_masked = extract_target_trial_beta(glm, signature_weights)

                sig_score = score_single_trial_beta(
                    beta_masked,
                    signature_weights,
                    signature_voxel_volume,
                    apply_scale_factor=apply_scale_factor,
                    original_beta_volume_mm3=original_bold_volume
                )
                trial_scores[trial_reg] = sig_score

            mean_cond_raw = float(np.mean(condition_numbers_raw))
            max_cond_raw = float(np.max(condition_numbers_raw))
            mean_cond_z = float(np.mean(condition_numbers_z))
            max_cond_z = float(np.max(condition_numbers_z))
            n_flagged = int(sum(1 for c in condition_numbers_z if c >= condition_threshold))


            nuisance_events = config['glm'].get('nuisance_events', [])
            first_lss_events = create_lss_events_for_trial(events_path, 0, trial_info, nuisance_events)
            temp_glm, _, _ = fit_lss_glm_for_trial(bold_img, mask_img, first_lss_events, confounds, tr, hrf_model, high_pass_hz)
            n_regressors = temp_glm.design_matrices_[0].shape[1]

            qc_records.append({
                'subject': subject,
                'run': run_num,
                'n_trials_modelled': int(n_trials),
                'n_design_columns': int(n_regressors),
                'condition_number_mean': mean_cond_z,
                'condition_number_max': max_cond_z,
                'condition_number_mean_raw': mean_cond_raw,
                'condition_number_max_raw': max_cond_raw,
                'condition_threshold': condition_threshold,
                'n_trials_flagged': int(n_flagged),
            })

            for _, row in trial_info.iterrows():
                trial_reg = row['trial_regressor']
                if trial_reg not in trial_scores:
                    continue

                trial_result = {
                    'subject': subject,
                    'run': run_num,
                    'trial_regressor': trial_reg,
                    'trial_idx_global': int(row['trial_idx_global']),
                    'trial_idx_run': int(row['trial_idx_run']),
                    'trial_type': row['trial_type'],
                    'temp_celsius': temp_mapping.get(row['trial_type'], np.nan),
                    'vas_rating': np.nan,
                    'pain_binary': row.get('pain_binary', np.nan),
                    score_col: trial_scores[trial_reg],
                    'n_trials_removed_run': 0
                }

                if 'vas_0_200' in row.index:
                    trial_result['vas_rating'] = row['vas_0_200']
                elif 'rating' in row.index:
                    trial_result['vas_rating'] = row['rating']

                if ('pain_binary' not in row.index or pd.isna(trial_result['pain_binary'])) and not pd.isna(trial_result['vas_rating']):
                    trial_result['pain_binary'] = int(trial_result['vas_rating'] > pain_threshold)

                all_trial_results.append(trial_result)

            global_trial_idx += n_trials

        except Exception as e:
            log(f"    Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue

    if qc_records:
        qc_df = pd.DataFrame(qc_records)
        qc_path = qc_dir / f"{subject}_trial_glm_qc.tsv"
        qc_df.to_csv(qc_path, sep='\t', index=False, float_format='%.6f')


    if not all_trial_results:
        return False

    trial_df = pd.DataFrame(all_trial_results)
    trial_output_path = subject_output_dir / f"trial_{signature_name}.tsv"
    trial_df.to_csv(trial_output_path, sep='\t', index=False, float_format='%.6f')

    discrim_metrics = compute_discrimination_metrics(trial_df, score_col=score_col, pain_threshold=pain_threshold)

    discrim_output_path = subject_output_dir / f"{signature_name}_discrimination_metrics.json"
    sanitized_discrim = sanitize_for_json(discrim_metrics)
    discrim_output_path.write_text(json.dumps(sanitized_discrim, indent=2))

    return True


###################################################################
# Main
###################################################################

def main():
    parser = argparse.ArgumentParser(description='Score signatures: condition-level and/or trial-level (LSS)')
    parser.add_argument('--config', default='utils/config.yaml', help='Configuration file')
    parser.add_argument('--subject', default=None, help='Process specific subject')
    parser.add_argument('--work-dir', default=None, help='Working directory (default: config outputs/work)')
    parser.add_argument('--output-dir', default=None, help='Base directory for signature outputs (default: config outputs root)')
    parser.add_argument('--qc-dir', default=None, help='QC directory (default: config outputs/qc)')
    parser.add_argument('--signatures', nargs='+', default=None, help='Signatures to score (e.g., nps siips1)')
    parser.add_argument('--skip-conditions', action='store_true', help='Skip condition-level scoring')
    parser.add_argument('--skip-trials', action='store_true', help='Skip trial-level LSS scoring')
    parser.add_argument('--condition-threshold', type=float, default=30.0, help='LSS condition number threshold')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        log(f"Config: {e}", "ERROR")
        return 1

    if args.signatures:
        signatures_to_process = args.signatures
    else:
        signatures_to_process = config.get('enabled_signatures', ['nps'])

    signature_data = {}
    for sig_name in signatures_to_process:
        weights_key = f"{sig_name}_weights_path"
        if weights_key not in config.get('resources', {}):
            continue

        log(sig_name.upper())
        weights_path = Path(config['resources'][weights_key])
        try:
            weights_img, mask = load_signature_weights(weights_path, sig_name)
            voxel_sizes = weights_img.header.get_zooms()[:3]
            signature_data[sig_name] = {'weights_img': weights_img, 'mask': mask, 'voxel_volume': float(np.prod(voxel_sizes))}
        except Exception as e:
            log(f"{sig_name}: {e}", "ERROR")
            return 1

    if not signature_data:
        return 1

    paths = PipelinePaths.from_config(config, work_dir=args.work_dir, output_dir=args.output_dir, qc_dir=args.qc_dir)
    paths.ensure_core_roots()

    index_dir = paths.index_dir
    if not index_dir.exists():
        return 1

    qc_stage_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("04_score_signatures"))

    subjects = [args.subject] if args.subject else config['subjects']
    all_success = True
    for subject in subjects:
        log(subject)
        inventory_path = index_dir / f"{subject}_files.json"
        if not inventory_path.exists():
            all_success = False
            continue

        try:
            with open(inventory_path, 'r') as f:
                inventory = json.load(f)

            for sig_name, sig_info in signature_data.items():
                sig_metadata = config.get('signature_metadata', {}).get(sig_name, {})
                apply_scale = sig_metadata.get('requires_beta_scale_factor', True)

                harmonized_root = PipelinePaths.ensure_dir(paths.harmonized_dir(sig_name))
                scores_root = PipelinePaths.ensure_dir(paths.signature_scores_dir(sig_name))
                qc_sig_dir = PipelinePaths.ensure_dir(qc_stage_dir / sig_name)

                if not args.skip_conditions:
                    log("Conditions")
                    if not score_conditions_for_subject(config, inventory, subject, sig_name, harmonized_root, scores_root, qc_sig_dir,
                                                       sig_info['weights_img'], sig_info['mask']):
                        all_success = False

                if not args.skip_trials:
                    log("Trials")
                    work_dir = paths.work_root
                    if not score_trials_for_subject(config, inventory, subject, sig_name, sig_info['weights_img'], sig_info['voxel_volume'],
                                                   apply_scale, work_dir, scores_root, qc_sig_dir, args.condition_threshold):
                        all_success = False

        except Exception as e:
            log(f"{subject}: {e}", "ERROR")
            all_success = False

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

