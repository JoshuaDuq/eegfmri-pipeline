#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix

from utils import load_config, get_log_function, PipelinePaths
from utils.io_utils import load_inventory, get_bold_n_volumes, get_bold_info, sanitize_for_json
from analysis.glm import extract_confounds, create_design_matrix, compute_design_correlations, validate_design


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)


###################################################################
# Build Confounds
###################################################################



def process_confounds(config: Dict, inventory: Dict, work_dir: Path) -> Tuple[int, int]:
    subject = inventory['subject']
    motion_24_cols = config['confounds']['motion_24_params']
    motion_outlier_prefix = config['confounds']['motion_outlier_prefix']
    subject_dir = PipelinePaths.ensure_dir(work_dir / "firstlevel" / subject)
    n_success, n_total = 0, 0

    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        n_total += 1

        if not run_data['complete']:
            continue

        log(f"Run {run_num}")
        try:
            bold_path = Path(run_data['files']['bold']['path'])
            confounds_path = Path(run_data['files']['confounds']['path'])
            n_volumes = get_bold_n_volumes(bold_path)
            confounds_clean, metadata = extract_confounds(confounds_path, motion_24_cols, motion_outlier_prefix, n_volumes)
            output_tsv = subject_dir / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
            output_tsv.parent.mkdir(parents=True, exist_ok=True)
            confounds_clean.to_csv(output_tsv, sep='\t', index=False, float_format='%.10f')

            metadata.update({
                'subject': subject,
                'run': run_num,
                'bold_path': str(bold_path),
                'confounds_path': str(confounds_path),
                'output_path': str(output_tsv)
            })
            metadata_path = subject_dir / f"run-{run_num:02d}_confounds_summary.json"
            sanitized_metadata = sanitize_for_json(metadata)
            metadata_path.write_text(json.dumps(sanitized_metadata, indent=2))
            n_success += 1
        except Exception as e:
            log(f"Failed: {e}", "ERROR")

    if n_success > 0:
        summary_files = sorted(subject_dir.glob("run-*_confounds_summary.json"))
        if summary_files:
            all_meta = [json.loads(sf.read_text()) for sf in summary_files]
            outlier_counts = [m['motion_outliers_found'] for m in all_meta]
            summary = {
                'subject': subject,
                'n_runs': len(all_meta),
                'total_volumes': sum(m['output_rows'] for m in all_meta),
                'outlier_stats': {
                    'min': min(outlier_counts),
                    'max': max(outlier_counts),
                    'mean': np.mean(outlier_counts),
                    'total': sum(outlier_counts)
                } if outlier_counts else {}
            }
            sanitized_summary = sanitize_for_json(summary)
            (subject_dir / f"{subject}_confounds_summary.json").write_text(json.dumps(sanitized_summary, indent=2))

    return n_success, n_total


###################################################################
# Build Design Matrices
###################################################################


def load_clean_confounds(subject: str, run_num: int, work_dir: Path) -> Optional[pd.DataFrame]:
    subject_id = subject if subject.startswith("sub-") else f"sub-{subject}"
    path = work_dir / "firstlevel" / subject_id / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
    return pd.read_csv(path, sep="\t") if path.exists() else None


def process_design_matrices(config: Dict, inventory: Dict, work_dir: Path, qc_dir: Path) -> Tuple[int, int]:
    subject = inventory['subject']
    n_success = 0
    n_total = 0

    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        n_total += 1

        if not run_data['complete']:
            continue

        log(f"Run {run_num}")
        events_path = Path(run_data['files']['events']['path'])
        bold_path = Path(run_data['files']['bold']['path'])

        try:
            events = pd.read_csv(events_path, sep='\t')

            n_volumes, bold_tr = get_bold_info(bold_path)
            config_tr = config['glm']['tr']
            tr = config_tr


            temp_labels = config['glm']['temp_labels']
            nuisance_events = config['glm']['nuisance_events']
            hrf_model = config['glm']['hrf']['model']
            high_pass_sec = config['glm']['high_pass_sec']

            if high_pass_sec is None or high_pass_sec <= 0:
                raise ValueError('high_pass_sec must be positive in config')
            high_pass_hz = 1.0 / float(high_pass_sec)

            confounds_df = load_clean_confounds(subject, run_num, work_dir)
            confound_names = []
            if confounds_df is not None:
                if len(confounds_df) != n_volumes:
                    confounds_df = None
                else:
                    confound_names = list(confounds_df.columns)
            else:
                confound_names = []

            design_matrix = create_design_matrix(events, n_volumes, tr, temp_labels, nuisance_events, hrf_model, high_pass_hz, confounds_df)

            validation = validate_design(design_matrix, temp_labels, nuisance_events, events)
            if 'temp_regressors_present_in_events' not in validation:
                if 'trial_type' in events:
                    temps_in_events = [label for label in temp_labels if label in set(events['trial_type'])]
                else:
                    temps_in_events = []
                validation['temp_regressors_present_in_events'] = temps_in_events

            if validation['warnings']:
                for warning in validation['warnings']:
                    log(warning)

            if not validation['valid']:
                log("Design validation failed")
                continue


            task_summary, corr_matrix, task_confound_summary = compute_design_correlations(design_matrix, temp_labels, confound_names)


            metadata = {
                'subject': subject,
                'run': run_num,
                'n_volumes': n_volumes,
                'tr': tr,
                'n_events': len(events),
                'hrf_model': hrf_model,
                'design_shape': list(design_matrix.shape),
                'temp_labels': temp_labels,
                'nuisance_events': nuisance_events,
                'confound_columns': confound_names,
                'validation': validation,
                'task_correlation_summary': {
                    'n_pairs': len(task_summary),
                    'n_high_corr': int(task_summary['flag_high'].sum()) if not task_summary.empty else 0,
                    'max_abs_corr': float(task_summary['abs_correlation'].max()) if not task_summary.empty else None,
                    'mean_abs_corr': float(task_summary['abs_correlation'].mean()) if not task_summary.empty else None,
                },
                'task_confound_summary': {
                    'n_tasks_tested': len(task_confound_summary),
                    'n_high_corr': int(task_confound_summary['flag_high'].sum()) if not task_confound_summary.empty else 0,
                    'max_abs_corr': float(task_confound_summary['abs_correlation'].max()) if not task_confound_summary.empty else None,
                },
            }

            subject_dir = PipelinePaths.ensure_dir(work_dir / "firstlevel" / subject)
            design_preview_path = subject_dir / f"run-{run_num:02d}_design_preview.tsv"
            design_preview_path.parent.mkdir(parents=True, exist_ok=True)
            design_matrix.to_csv(design_preview_path, sep='\t', index=False, float_format='%.6f')

            metadata_path = subject_dir / f"run-{run_num:02d}_design_metadata.json"
            sanitized_metadata = sanitize_for_json(metadata)
            metadata_path.write_text(json.dumps(sanitized_metadata, indent=2))

            if not task_summary.empty:
                corr_path = qc_dir / f"{subject}_design_corr_run-{run_num:02d}.tsv"
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                task_summary.to_csv(corr_path, sep='\t', index=False, float_format='%.6f')

            if task_confound_summary is not None and not task_confound_summary.empty:
                conf_corr_path = qc_dir / f"{subject}_design_task_confound_corr_run-{run_num:02d}.tsv"
                conf_corr_path.parent.mkdir(parents=True, exist_ok=True)
                task_confound_summary.to_csv(conf_corr_path, sep='\t', index=False, float_format='%.6f')

            corr_matrix_path = subject_dir / f"run-{run_num:02d}_design_corr_matrix.tsv"
            corr_matrix.to_csv(corr_matrix_path, sep='\t', float_format='%.6f')

            n_success += 1
        except Exception as e:
            log(f"Failed: {e}", "ERROR")

    return n_success, n_total


###################################################################
# Main
###################################################################

def main():
    parser = argparse.ArgumentParser(description='Prepare GLM inputs: confounds and design matrices')
    parser.add_argument('--config', default='utils/config.yaml', help='Configuration file')
    parser.add_argument('--subject', default=None, help='Process specific subject')
    parser.add_argument('--work-dir', default='work', help='Working directory')
    parser.add_argument('--qc-dir', default='qc', help='QC directory')
    parser.add_argument('--skip-confounds', action='store_true', help='Skip building confounds (if already done)')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        log(f"Config: {e}", "ERROR")
        return 1

    subjects = [args.subject] if args.subject else config['subjects']
    paths = PipelinePaths.from_config(config, work_dir=args.work_dir, qc_dir=args.qc_dir)
    paths.ensure_core_roots()
    work_dir = paths.work_root
    index_dir = paths.index_dir

    if not index_dir.exists():
        log(f"Index directory not found: {index_dir}", "ERROR")
        return 1

    qc_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("02_prepare_glm_inputs"))
    all_success = True

    for subject in subjects:
        log(subject)
        inventory_path = index_dir / f"{subject}_files.json"
        if not inventory_path.exists():
            log(f"Inventory not found: {inventory_path}", "ERROR")
            all_success = False
            continue

        try:
            inventory = load_inventory(inventory_path)

            if not args.skip_confounds:
                log("Confounds")
                n_success, n_total = process_confounds(config, inventory, work_dir)
                if n_success != n_total:
                    all_success = False

            log("Design matrices")
            n_success, n_total = process_design_matrices(config, inventory, work_dir, qc_dir)
            if n_success != n_total:
                all_success = False
        except Exception as e:
            log(f"{subject}: {e}", "ERROR")
            all_success = False

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

