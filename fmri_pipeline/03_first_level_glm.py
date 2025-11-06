#!/usr/bin/env python3

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import apply_mask
from nilearn.image import resample_to_img
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", message="The following unexpected columns in events data will be ignored", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", message="Matrix is singular at working precision, regularizing", category=UserWarning, module="nilearn")

from utils import load_config, get_log_function, PipelinePaths
from utils.io_utils import load_inventory, load_confounds, sanitize_for_json
from utils.image_utils import (
    build_analysis_mask,
    find_run_betas_for_temperature,
    find_variance_maps,
    load_signature_weights,
    resample_beta_to_signature_grid,
    validate_grid_match,
)
from analysis.glm import (
    prepare_events,
    convert_bold_to_percent_signal,
    fit_glm,
    compute_regressor_snr,
    select_task_columns,
    compute_condition_number,
    summarize_glm_design,
    extract_beta_maps,
    combine_betas_fixed_effects,
    validate_output,
)


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)


###################################################################
# Main Processing
###################################################################

def fit_glms_for_subject(config: Dict, inventory: Dict, work_dir: Path, qc_dir: Path, compute_snr: bool = False) -> Tuple[int, int]:
    subject = inventory["subject"]
    n_success = 0
    n_total = 0

    for run_key in sorted(inventory["runs"].keys()):
        run_data = inventory["runs"][run_key]
        run_num = run_data["run_number"]
        n_total += 1

        if not run_data["complete"]:
            continue
        
        log(f"Run {run_num}")

        try:
            bold_path = Path(run_data["files"]["bold"]["path"])
            mask_path = Path(run_data["files"]["mask"]["path"])
            subject_id = subject if subject.startswith("sub-") else f"sub-{subject}"
            subject_dir = PipelinePaths.ensure_dir(work_dir / "firstlevel" / subject_id)
            confounds_path = subject_dir / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
            if not confounds_path.exists():
                continue

            confounds = load_confounds(confounds_path)

            events_path_str = run_data["files"].get("events", {}).get("path")
            if not events_path_str:
                raise FileNotFoundError("Events path missing from run inventory")
            events_path = Path(events_path_str)
            if not events_path.exists():
                raise FileNotFoundError(f"Events file not found: {events_path}")

            temp_labels = config["glm"]["temp_labels"]
            nuisance_events = config["glm"]["nuisance_events"]
            onset_offset = config["glm"].get("stim_onset_offset_sec", 0.0)
            plateau_duration = config["glm"].get("stim_dur_sec", None)
            events = prepare_events(events_path, temp_labels, nuisance_events, onset_offset_sec=onset_offset, plateau_duration_sec=plateau_duration)

            tr = config["glm"]["tr"]
            hrf_model = config["glm"]["hrf"]["model"]
            high_pass_sec = config["glm"]["high_pass_sec"]
            if high_pass_sec <= 0:
                raise ValueError("high_pass_sec must be positive in configuration")
            high_pass_hz = 1.0 / float(high_pass_sec)

            convert_to_psc = config["glm"].get("convert_to_percent_signal", False)
            psc_window_sec = config["glm"].get("percent_signal_baseline_sec", 120.0)
            psc_min_baseline = config["glm"].get("percent_signal_min_baseline", 1e-3)

            glm, diagnostics = fit_glm(bold_path, mask_path, events, confounds, tr, hrf_model, high_pass_hz,
                                      convert_to_psc, psc_window_sec, psc_min_baseline)

            design_qc = summarize_glm_design(glm, list(temp_labels), subject, run_num, subject_dir, qc_dir, list(confounds.columns))
            diagnostics["design_qc_glm"] = design_qc

            beta_outputs = extract_beta_maps(glm, temp_labels, subject_dir, run_num)
            if len(beta_outputs) == 0:
                continue

            diagnostics["beta_maps"] = {k: str(v["beta"]) for k, v in beta_outputs.items()}
            diagnostics["beta_variance_maps"] = {k: str(v["variance"]) for k, v in beta_outputs.items()}
            diagnostics["design_columns_for_temperature"] = {k: v["design_column"] for k, v in beta_outputs.items()}
            diagnostics["subject"] = subject
            diagnostics["run"] = run_num

            diag_path = subject_dir / f"run-{run_num:02d}_modeldiag.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            sanitized_diagnostics = sanitize_for_json(diagnostics)
            diag_path.write_text(json.dumps(sanitized_diagnostics, indent=2))

            if compute_snr:
                snr_df = compute_regressor_snr(glm, bold_path, mask_path, tuple(confounds.columns))
                if not snr_df.empty:
                    snr_path = qc_dir / f"{subject}_run-{run_num:02d}_regressor_snr.tsv"
                    snr_path.parent.mkdir(parents=True, exist_ok=True)
                    snr_df.to_csv(snr_path, sep="\t", index=False, float_format="%.6f")

            n_success += 1
        except Exception as exc:
            log(f"{run_num}: {exc}", "ERROR")

    return n_success, n_total


def combine_runs_for_subject(config: Dict, inventory: Dict, work_dir: Path, firstlevel_root: Path, variance_method: str = 'uniform') -> Tuple[int, int]:
    subject = inventory['subject']
    temp_labels, n_runs = config['glm']['temp_labels'], len(inventory['runs'])
    subject_output_dir = PipelinePaths.ensure_dir(firstlevel_root / subject)

    analysis_mask_img = build_analysis_mask(inventory)
    analysis_mask_data = analysis_mask_img.get_fdata() > 0
    analysis_mask_path = subject_output_dir / "analysis_mask.nii.gz"
    nib.save(analysis_mask_img, str(analysis_mask_path))

    n_success = 0
    combination_summary = {
        'subject': subject,
        'n_runs': n_runs,
        'variance_method': variance_method,
        'analysis_mask_path': str(analysis_mask_path),
        'analysis_mask_voxels': int(np.sum(analysis_mask_data)),
        'temperatures': {}
    }

    for temp in temp_labels:
        try:
            subject_dir = work_dir / "firstlevel" / subject
            beta_paths = find_run_betas_for_temperature(subject_dir, temp, list(range(1, n_runs + 1)))

            if not beta_paths:
                continue

            beta_paths, variance_paths = find_variance_maps(beta_paths)

            combined_beta, combined_var, n_runs_img, used_mask = combine_betas_fixed_effects(
                beta_paths, variance_paths=variance_paths, variance_method=variance_method, analysis_mask_img=analysis_mask_img)
            mask_voxels_used = int(np.sum(used_mask))

            validation = validate_output(combined_beta, used_mask)

            beta_path = subject_output_dir / f"beta_{temp}.nii.gz"
            var_path = subject_output_dir / f"beta_{temp}_variance.nii.gz"
            nruns_path = subject_output_dir / f"beta_{temp}_n_runs.nii.gz"

            nib.save(combined_beta, str(beta_path))
            nib.save(combined_var, str(var_path))
            nib.save(n_runs_img, str(nruns_path))

            combination_summary['temperatures'][temp] = {
                'n_runs_found': len(beta_paths),
                'validation': validation,
                'success': True,
                'output_path': str(beta_path),
                'variance_path': str(var_path),
                'n_runs_path': str(nruns_path),
                'weights_used': variance_method,
                'weights_requested': variance_method,
                'variance_maps_found': len(variance_paths),
                'mask_voxels_used': mask_voxels_used,
            }

            n_success += 1

        except Exception as e:
            log(f"Failed: {e}", "ERROR")
            combination_summary['temperatures'][temp] = {
                'n_runs_found': len(beta_paths) if 'beta_paths' in locals() else 0,
                'success': False,
                'error': str(e),
                'weights_used': variance_method,
                'weights_requested': variance_method,
                'variance_maps_found': len(variance_paths) if 'variance_paths' in locals() else 0
            }
            continue

    summary_path = subject_output_dir / "combination_summary.json"
    sanitized_summary = sanitize_for_json(combination_summary)
    summary_path.write_text(json.dumps(sanitized_summary, indent=2))

    return n_success, len(temp_labels)


def harmonize_betas_for_subject(config: Dict, subject: str, firstlevel_root: Path, signatures_root: Path,
                               qc_root: Path, signature_weights: Dict[str, nib.Nifti1Image]) -> Tuple[int, int]:
    temp_labels = config['glm']['temp_labels']
    sig_metadata = config.get('signature_metadata', {})

    total_success = 0
    total_expected = len(temp_labels) * len(signature_weights)

    for sig_name, sig_img in signature_weights.items():
        subject_output_dir = PipelinePaths.ensure_dir(signatures_root / sig_name / "harmonized" / subject)
        qc_sig_dir = PipelinePaths.ensure_dir(qc_root / sig_name)

        qc_records = []
        harmonization_metadata = {
            'subject': subject,
            'signature': sig_name,
            f'{sig_name}_grid': {
                'shape': list(sig_img.shape),
                'voxel_sizes': [float(v) for v in sig_img.header.get_zooms()[:3]],
                'voxel_volume_mm3': float(np.prod(sig_img.header.get_zooms()[:3])),
                'affine': sig_img.affine.tolist()
            },
            'temperatures': {}
        }

        for temp in temp_labels:
            try:
                beta_path = firstlevel_root / subject / f"beta_{temp}.nii.gz"
                if not beta_path.exists():
                    continue
                beta_img = nib.load(str(beta_path))

                original_voxel_sizes = [float(v) for v in beta_img.header.get_zooms()[:3]]
                original_voxel_volume = float(np.prod(original_voxel_sizes))

                resampled_beta = resample_beta_to_signature_grid(beta_img, sig_img)
                val = validate_grid_match(resampled_beta, sig_img)
                if not val['shapes_match']:
                    continue

                min_overlap = sig_metadata.get(sig_name, {}).get('min_overlap_pct', 98.0)
                if val['overlap_pct'] < min_overlap:
                    harmonization_metadata['temperatures'][temp] = {
                        'validation': val,
                        'original_voxel_sizes': original_voxel_sizes,
                        'original_voxel_volume_mm3': original_voxel_volume,
                        'success': False
                    }
                    continue

                output_path = subject_output_dir / f"beta_{temp}_on{sig_name.upper()}grid.nii.gz"
                nib.save(resampled_beta, str(output_path))
                harmonization_metadata['temperatures'][temp] = {
                    'validation': val,
                    'original_voxel_sizes': original_voxel_sizes,
                    'original_voxel_volume_mm3': original_voxel_volume,
                    'success': True
                }
                qc_records.append({'subject': subject, 'signature': sig_name, 'temperature': temp, **val})
                total_success += 1
            except Exception as e:
                log(f"Failed: {e}", "ERROR")
                harmonization_metadata['temperatures'][temp] = {'success': False, 'error': str(e)}

        sanitized_harmonization = sanitize_for_json(harmonization_metadata)
        (subject_output_dir / "harmonization_metadata.json").write_text(json.dumps(sanitized_harmonization, indent=2))
        if qc_records:
            pd.DataFrame(qc_records).to_csv(qc_sig_dir / f"{subject}_{sig_name}_grid_check.tsv", sep='\t', index=False)

    return total_success, total_expected


###################################################################
# Main
###################################################################

def main():
    parser = argparse.ArgumentParser(description='First-level GLM: fit, combine runs, harmonize to signature grids')
    parser.add_argument("--config", default="utils/config.yaml", help="Configuration file")
    parser.add_argument("--subject", default=None, help="Process single subject")
    parser.add_argument("--work-dir", default="work", help="Working directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--qc-dir", default="qc", help="QC directory")
    parser.add_argument("--compute-snr", action="store_true", help="Compute regressor SNR")
    parser.add_argument("--variance-method", default='glm', choices=['glm', 'uniform'], help="Variance weighting method")
    parser.add_argument("--skip-fit", action="store_true", help="Skip GLM fitting (if already done)")
    parser.add_argument("--skip-combine", action="store_true", help="Skip combining runs (if already done)")
    parser.add_argument("--skip-harmonize", action="store_true", help="Skip harmonization (if already done)")
    parser.add_argument("--signatures", nargs='+', default=None, help="Signatures to harmonize (e.g., nps siips1)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:
        log(f"Config: {exc}", "ERROR")
        return 1

    subjects = [args.subject] if args.subject else config['subjects']
    paths = PipelinePaths.from_config(config, work_dir=args.work_dir, output_dir=args.output_dir, qc_dir=args.qc_dir)
    paths.ensure_core_roots()
    work_dir = paths.work_root
    index_dir = paths.index_dir

    if not index_dir.exists():
        return 1

    qc_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("03_first_level_glm"))
    all_success = True

    for subject in subjects:
        log(subject)
        inventory_path = index_dir / f"{subject}_files.json"
        if not inventory_path.exists():
            all_success = False
            continue

        try:
            inventory = load_inventory(inventory_path)

            if not args.skip_fit:
                log("Fit GLM")
                n_success, n_total = fit_glms_for_subject(config, inventory, work_dir, qc_dir, args.compute_snr)
                if n_success != n_total:
                    all_success = False

            firstlevel_root = PipelinePaths.ensure_dir(paths.firstlevel_root)

            if not args.skip_combine:
                log("Combine runs")
                n_success, n_total = combine_runs_for_subject(config, inventory, work_dir, firstlevel_root, args.variance_method)
                if n_success != n_total:
                    all_success = False

            if not args.skip_harmonize:
                log("Harmonize")
                if args.signatures:
                    signatures_to_process = args.signatures
                else:
                    signatures_to_process = config.get('enabled_signatures', ['nps'])

                if signatures_to_process:
                    signature_weights = {}
                    for sig_name in signatures_to_process:
                        weights_key = f"{sig_name}_weights_path"
                        if weights_key not in config.get('resources', {}):
                            log(f"{sig_name.upper()} weights path not found in config", "WARNING")
                            continue

                        weights_path = Path(config['resources'][weights_key])
                        try:
                            sig_img, sig_mask = load_signature_weights(weights_path, sig_name)
                            signature_weights[sig_name] = sig_img
                        except Exception as e:
                            log(f"{sig_name}: {e}", "ERROR")
                            all_success = False
                            continue

                    if signature_weights:
                        signatures_root = PipelinePaths.ensure_dir(paths.signatures_root)
                        n_success, n_total = harmonize_betas_for_subject(config, subject, firstlevel_root, signatures_root, qc_dir, signature_weights)
                        if n_success != n_total:
                            all_success = False

        except Exception as exc:
            log(f"{subject}: {exc}", "ERROR")
            all_success = False

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())

