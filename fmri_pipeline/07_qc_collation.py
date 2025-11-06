#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import pandas as pd
import numpy as np
import nibabel as nib
import yaml

try:
    import nilearn
    import scipy
    import sklearn
    import matplotlib
except ImportError as e:
    warnings.warn(f"Failed to import package: {e}")

from utils import load_config, get_log_function, PipelinePaths


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)


###################################################################
# Motion Metrics
###################################################################


def compute_fd_from_confounds(confounds_tsv_path: Path) -> Dict:
    if not confounds_tsv_path.exists():
        return {'error': 'File not found'}
    try:
        confounds = pd.read_csv(confounds_tsv_path, sep='\t')
        fd_col = next((col for col in ['framewise_displacement', 'fd', 'FD'] if col in confounds.columns), None)
        if fd_col is None:
            return {'error': 'FD column not found'}
        fd_values = confounds[fd_col].dropna().values
        if len(fd_values) == 0:
            return {'error': 'All FD NaN'}
        return {'mean_fd': float(np.mean(fd_values)), 'median_fd': float(np.median(fd_values)),
                'max_fd': float(np.max(fd_values)), 'std_fd': float(np.std(fd_values)), 'n_volumes': len(fd_values) + 1}
    except Exception as e:
        return {'error': str(e)}


def get_outliers_during_heat_fraction(work_dir: Path, subject: str, run_num: int,
                                     inventory: Dict, tr: Optional[float]) -> Optional[float]:
    try:
        confounds_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
        if not confounds_path.exists():
            return None

        confounds = pd.read_csv(confounds_path, sep='\t')
        outlier_cols = [col for col in confounds.columns if col.startswith('motion_outlier')]
        if not outlier_cols:
            return 0.0

        outlier_mask = confounds[outlier_cols].sum(axis=1) > 0
        if (n_outliers_total := outlier_mask.sum()) == 0:
            return 0.0

        run_key = f"run-{run_num:02d}"
        if run_key not in inventory['runs']:
            return None

        events_path = Path(inventory['runs'][run_key]['files']['events']['path'])
        if not events_path.exists():
            return None

        events = pd.read_csv(events_path, sep='\t')
        heat_events = events[events['trial_type'].str.contains('temp', case=False, na=False)]
        if len(heat_events) == 0 or not tr or tr <= 0:
            return None

        n_volumes = len(confounds)
        heat_mask = np.zeros(n_volumes, dtype=bool)
        for _, event in heat_events.iterrows():
            start_vol = max(0, int(event['onset'] / tr))
            end_vol = min(n_volumes, int((event['onset'] + event['duration']) / tr))
            heat_mask[start_vol:end_vol] = True

        return float((outlier_mask & heat_mask).sum() / n_outliers_total)
    except:
        return None


def summarize_motion_per_subject(config: Dict, inventory: Dict, work_dir: Path,
                                 tr: Optional[float]) -> Dict:
    subject = inventory['subject']
    
    all_fd_means = []
    all_outlier_counts = []
    all_outlier_fracs = []
    all_heat_outlier_fracs = []
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        
        # Load confounds summary
        confounds_summary_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_summary.json"
        if confounds_summary_path.exists():
            with open(confounds_summary_path, 'r') as f:
                summary = json.load(f)
            n_outliers = summary.get('motion_outliers_found', 0)
            n_volumes = summary.get('output_rows', 0)
        else:
            n_outliers = 0
            n_volumes = 0
        
        # Get FD stats
        confounds_raw_path = Path(run_data['files']['confounds']['path'])
        fd_stats = compute_fd_from_confounds(confounds_raw_path)
        
        # Get heat outlier fraction
        heat_outlier_frac = get_outliers_during_heat_fraction(work_dir, subject, run_num, inventory, tr)
        
        if fd_stats.get('mean_fd') is not None:
            all_fd_means.append(fd_stats['mean_fd'])
        
        all_outlier_counts.append(n_outliers)
        
        if n_volumes > 0:
            all_outlier_fracs.append(n_outliers / n_volumes)
        
        if heat_outlier_frac is not None:
            all_heat_outlier_fracs.append(heat_outlier_frac)
    
    return {
        'mean_fd_across_runs': float(np.mean(all_fd_means)) if all_fd_means else None,
        'median_fd_across_runs': float(np.median(all_fd_means)) if all_fd_means else None,
        'max_fd_across_runs': float(np.max([fd_stats.get('max_fd', 0) for run_data in inventory['runs'].values() if (fd_stats := compute_fd_from_confounds(Path(run_data['files']['confounds']['path']))).get('max_fd')])) if any((fd_stats := compute_fd_from_confounds(Path(run_data['files']['confounds']['path']))).get('max_fd') for run_data in inventory['runs'].values()) else None,
        'total_outliers': sum(all_outlier_counts),
        'mean_outlier_fraction': float(np.mean(all_outlier_fracs)) if all_outlier_fracs else None,
        'mean_heat_outlier_fraction': float(np.mean(all_heat_outlier_fracs)) if all_heat_outlier_fracs else None
    }


###################################################################
# NPS Grid Validation
###################################################################


def _resolve_nps_weights_path(config: Dict) -> Optional[Path]:
    if 'nps' in config and config['nps'].get('weights_path'):
        return Path(config['nps']['weights_path'])
    if 'resources' in config and config['resources'].get('nps_weights_path'):
        return Path(config['resources']['nps_weights_path'])
    return None


def validate_nps_grid_match(harmonization_metadata_path: Path, nps_weights_path: Path) -> Dict:
    validation = {'exact_match': False, 'n_nonzero_nps_voxels': 0, 'temperatures_validated': []}
    if not harmonization_metadata_path.exists() or not nps_weights_path.exists():
        validation['error'] = 'Files not found'
        return validation
    try:
        metadata = json.loads(harmonization_metadata_path.read_text())
        nps_data = nib.load(str(nps_weights_path)).get_fdata()
        validation['n_nonzero_nps_voxels'] = int(np.sum(nps_data != 0))
        all_match = True
        for temp, temp_data in metadata.get('temperatures', {}).items():
            if temp_data.get('success', False):
                validation['temperatures_validated'].append(temp)
            else:
                all_match = False
        validation['exact_match'] = all_match and len(validation['temperatures_validated']) > 0
    except Exception as e:
        validation['error'] = str(e)
    return validation


def check_nans_infs_in_betas(nps_ready_dir: Path, subject: str, temp_labels: List[str]) -> Dict:
    total_nans = total_infs = 0
    for temp in temp_labels:
        if (beta_path := nps_ready_dir / subject / f"beta_{temp}_onNPSgrid.nii.gz").exists():
            try:
                beta_data = nib.load(str(beta_path)).get_fdata()
                total_nans += int(np.sum(np.isnan(beta_data)))
                total_infs += int(np.sum(np.isinf(beta_data)))
            except:
                pass
    return {'total_nans': total_nans, 'total_infs': total_infs, 'has_issues': (total_nans > 0 or total_infs > 0)}


###################################################################
# GLM Diagnostics
###################################################################


def summarize_glm_diagnostics(work_dir: Path, subject: str, n_runs: int) -> Dict:
    all_dofs, all_r2s = [], []
    for run_num in range(1, n_runs + 1):
        if (diag_path := work_dir / "firstlevel" / subject / f"run-{run_num:02d}_modeldiag.json").exists():
            try:
                diag = json.loads(diag_path.read_text())
                if diag.get('dof') is not None:
                    all_dofs.append(diag['dof'])
                if diag.get('mean_r_squared') is not None:
                    all_r2s.append(diag['mean_r_squared'])
            except:
                pass
    return {'mean_dof': float(np.mean(all_dofs)) if all_dofs else None, 'min_dof': int(np.min(all_dofs)) if all_dofs else None,
            'mean_r_squared': float(np.mean(all_r2s)) if all_r2s else None}


def get_package_versions() -> Dict[str, str]:
    versions = {}
    for pkg_name in ['numpy', 'scipy', 'pandas', 'nibabel', 'nilearn', 'sklearn', 'matplotlib', 'yaml']:
        try:
            versions[pkg_name] = __import__(pkg_name).__version__
        except (ImportError, AttributeError):
            versions[pkg_name] = 'not installed'
    return versions


def get_system_info() -> Dict[str, str]:
    return {'platform': platform.platform(), 'python_version': platform.python_version(), 'system': platform.system()}


###################################################################
# Environment and Configuration
###################################################################


def compute_config_hash(config: Dict) -> Dict[str, str]:
    return {section: hashlib.sha256(json.dumps(config[section], sort_keys=True).encode()).hexdigest()[:16]
            for section in ['glm', 'confounds', 'nps', 'acquisition_params'] if section in config}


###################################################################
# QC Collation
###################################################################


def collate_qc_for_subject(config: Dict, subject: str, work_dir: Path, paths: PipelinePaths) -> Dict:
    qc_summary = {
        'subject': subject,
        'critical_flags': [],
        'warnings': []
    }
    
    # Load inventory
    inventory_path = work_dir / "index" / f"{subject}_files.json"
    if not inventory_path.exists():
        qc_summary['critical_flags'].append('File inventory missing')
        return qc_summary
    
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    tr = config.get('glm', {}).get('tr')
    qc_summary['motion'] = summarize_motion_per_subject(config, inventory, work_dir, tr)
    
    mean_fd = qc_summary['motion'].get('mean_fd_across_runs')
    if mean_fd is not None:
        fd_warn_thresh = config.get('qc', {}).get('motion_thresholds', {}).get('fd_mean_warn', 0.3)
        if mean_fd > fd_warn_thresh:
            qc_summary['warnings'].append(f'Mean FD ({mean_fd:.3f}) exceeds threshold ({fd_warn_thresh})')
    
    qc_summary['glm'] = summarize_glm_diagnostics(work_dir, subject, len(inventory['runs']))
    
    mean_dof = qc_summary['glm'].get('mean_dof')
    if mean_dof is not None and mean_dof < 50:
        qc_summary['warnings'].append(f'Low DOF ({mean_dof:.0f})')
    
    harmonization_path = Path(paths.harmonized_dir("nps", subject)) / "harmonization_metadata.json"
    nps_weights_path = _resolve_nps_weights_path(config)
    if nps_weights_path is None:
        qc_summary['critical_flags'].append('NPS weights path missing in config')
        grid_validation = {'exact_match': False}
    else:
        grid_validation = validate_nps_grid_match(harmonization_path, nps_weights_path)
    qc_summary['nps_grid'] = {
        'exact_match': grid_validation.get('exact_match', False),
        'n_nonzero_nps_voxels': grid_validation.get('n_nonzero_nps_voxels', 0),
        'n_temperatures_validated': len(grid_validation.get('temperatures_validated', []))
    }
    
    if not grid_validation.get('exact_match', False):
        qc_summary['critical_flags'].append('NPS grid mismatch')
    
    nps_ready_dir = Path(paths.harmonized_dir("nps"))
    temp_labels = config['glm']['temp_labels']
    
    nan_check = check_nans_infs_in_betas(nps_ready_dir, subject, temp_labels)
    qc_summary['data_integrity'] = nan_check
    
    if nan_check['total_nans'] > 0:
        qc_summary['critical_flags'].append(f'{nan_check["total_nans"]} NaN voxels in betas')
    if nan_check['total_infs'] > 0:
        qc_summary['critical_flags'].append(f'{nan_check["total_infs"]} Inf voxels in betas')
    
    # Status
    qc_summary['status'] = 'PASS' if len(qc_summary['critical_flags']) == 0 else 'FAIL'
    
    if qc_summary['status'] != 'PASS' and qc_summary['critical_flags']:
        log(f"{subject}: {qc_summary['critical_flags'][0]}", "ERROR")
    
    return qc_summary


###################################################################
# Output Functions
###################################################################


def create_summary_tsv(qc_summaries: List[Dict], output_path: Path):
    rows = [
        {'subject': qc['subject'], 'status': qc['status'],
         'n_critical_flags': len(qc['critical_flags']), 'n_warnings': len(qc['warnings']),
         'mean_fd': qc['motion'].get('mean_fd_across_runs'), 'max_fd': qc['motion'].get('max_fd_across_runs'),
         'total_outliers': qc['motion'].get('total_outliers'),
         'mean_outlier_fraction': qc['motion'].get('mean_outlier_fraction'),
         'mean_heat_outlier_fraction': qc['motion'].get('mean_heat_outlier_fraction'),
         'mean_dof': qc['glm'].get('mean_dof'), 'mean_r_squared': qc['glm'].get('mean_r_squared'),
         'nps_grid_exact_match': qc['nps_grid'].get('exact_match'),
         'n_nps_voxels_used': qc['nps_grid'].get('n_nonzero_nps_voxels'),
         'total_nans': qc['data_integrity'].get('total_nans'), 'total_infs': qc['data_integrity'].get('total_infs'),
         'critical_flags': '; '.join(qc['critical_flags']) if qc['critical_flags'] else '',
         'warnings': '; '.join(qc['warnings']) if qc['warnings'] else ''}
        for qc in qc_summaries
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep='\t', index=False, float_format='%.6f')


###################################################################
# Main
###################################################################


def main():
    parser = argparse.ArgumentParser(description='Consolidate QC from all pipeline steps')
    parser.add_argument('--config', default='utils/config.yaml', help='Config file')
    parser.add_argument('--subject', default=None, help='Specific subject')
    parser.add_argument('--work-dir', default='work', help='Working directory')
    parser.add_argument('--outputs-dir', default='outputs', help='Outputs directory')
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        log(f"Config: {e}", "ERROR")
        return 1
    
    # Determine subjects
    subjects = [args.subject] if args.subject else config['subjects']
    
    paths = PipelinePaths.from_config(
        config,
        work_dir=args.work_dir,
        output_dir=args.outputs_dir,
    )
    paths.ensure_core_roots()
    work_dir = paths.work_root
    qc_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("12_qc_collation"))
    
    qc_summaries = []
    all_pass = True
    
    for subject in subjects:
        log(subject)
        try:
            qc_summary = collate_qc_for_subject(config, subject, work_dir, paths)
            qc_summaries.append(qc_summary)
            
            if qc_summary['status'] != 'PASS':
                all_pass = False
                
        except Exception as e:
            log(f"{subject}: {e}", "ERROR")
            all_pass = False
    
    if qc_summaries:
        summary_path = qc_dir / "summary_qc.tsv"
        create_summary_tsv(qc_summaries, summary_path)
    
    env_path = qc_dir / "ENV.yaml"
    env_info = {'system': get_system_info(), 'packages': get_package_versions()}
    with open(env_path, 'w') as f:
        yaml.dump(env_info, f, default_flow_style=False, sort_keys=False)
    
    hash_path = qc_dir / "config_hash.json"
    hash_info = {
        'config_version': config.get('metadata', {}).get('config_version', 'unknown'),
        'pipeline_version': config.get('metadata', {}).get('pipeline_version', 'unknown'),
        'parameter_hashes': compute_config_hash(config)
    }
    with open(hash_path, 'w') as f:
        json.dump(hash_info, f, indent=2)
    
    n_pass = sum(1 for qc in qc_summaries if qc['status'] == 'PASS')
    n_fail = len(qc_summaries) - n_pass
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
