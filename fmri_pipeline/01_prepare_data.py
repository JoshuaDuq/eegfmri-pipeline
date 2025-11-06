#!/usr/bin/env python3

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd

from utils import get_subject_files, load_config, get_log_function, PipelinePaths
from utils.io_utils import normalize_subject_id, load_eeg_drop_log, save_inventory


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME)


###################################################################
# Split Events to Runs
###################################################################


def setup_temperature_mapping():
    return {
        44.3: 'temp44p3',
        45.3: 'temp45p3',
        46.3: 'temp46p3',
        47.3: 'temp47p3',
        48.3: 'temp48p3',
        49.3: 'temp49p3'
    }


def safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def process_run_events(df_run: pd.DataFrame, temp_mapping: Dict, include_feedback: bool = True) -> pd.DataFrame:
    run_id = df_run['run_id'].iloc[0] if 'run_id' in df_run.columns and len(df_run) > 0 else None
    all_events = []

    for idx, row in df_run.iterrows():
        trial_num = row['trial_number']
        run_id_value = row.get('run_id', run_id)
        source_idx = int(row.get('original_index', idx))

        stim_start = safe_float(row.get('stim_start_time'))
        stim_end = safe_float(row.get('stim_end_time'))
        if stim_start is None or stim_end is None or stim_end <= stim_start:
            log(f"Skipping trial {trial_num}: invalid stimulus timing", "WARNING")
            continue

        heat_event = {
            'onset': round(stim_start, 3),
            'duration': round(stim_end - stim_start, 3),
            'trial_type': temp_mapping.get(row['stimulus_temp'],
                                         f'temp{row["stimulus_temp"]:.1f}'.replace('.', 'p')),
            'temp_celsius': row['stimulus_temp'],
            'pain_binary': row['pain_binary_coded'],
            'vas_0_200': row['vas_final_coded_rating'],
            'block': run_id_value,
            'trial_index': trial_num,
            'original_index': source_idx,
            'stim_start_time': row['stim_start_time'],
            'stim_end_time': row['stim_end_time'],
            'vas_start_time': row['vas_start_time'],
            'vas_end_time': row['vas_end_time']
        }
        all_events.append(heat_event)

        if include_feedback:
            onset_candidates = [safe_float(row.get('pain_q_start_time')), safe_float(row.get('vas_start_time'))]
            onset_candidates = [val for val in onset_candidates if val is not None]
            feedback_onset = min(onset_candidates) if onset_candidates else None

            end_candidates = [safe_float(row.get('vas_end_time')), safe_float(row.get('pain_q_end_time'))]
            end_candidates = [val for val in end_candidates if val is not None]
            feedback_end = max(end_candidates) if end_candidates else None

            if feedback_onset is not None and feedback_end is not None and feedback_end > feedback_onset:
                feedback_event = {
                    'onset': round(feedback_onset, 3),
                    'duration': round(feedback_end - feedback_onset, 3),
                    'trial_type': 'feedback',
                    'temp_celsius': row['stimulus_temp'],
                    'pain_binary': row['pain_binary_coded'],
                    'vas_0_200': row['vas_final_coded_rating'],
                    'block': run_id_value,
                    'trial_index': trial_num,
                    'original_index': source_idx,
                    'stim_start_time': row['stim_start_time'],
                    'stim_end_time': row['stim_end_time'],
                    'vas_start_time': row['vas_start_time'],
                    'vas_end_time': row['vas_end_time']
                }
                all_events.append(feedback_event)

    events_out = pd.DataFrame(all_events)
    events_out = events_out.sort_values('onset').reset_index(drop=True)
    return events_out


def split_events_to_runs(subject: str, input_file: Path, output_dir: Path, include_feedback: bool = True, drop_log_path: Path = None) -> bool:
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        log(f"Load failed: {e}", "ERROR")
        return False

    if 'original_index' not in df.columns:
        df.insert(0, 'original_index', np.arange(len(df), dtype=int))

    run_col = 'run_id' if 'run_id' in df.columns else 'run'

    if drop_log_path:
        drop_log = load_eeg_drop_log(drop_log_path)
        if not drop_log.empty:
            n_before = len(df)

            mask = pd.Series(False, index=df.index)
            if 'original_index' in drop_log.columns and 'original_index' in df.columns:
                drop_idx = pd.to_numeric(drop_log['original_index'], errors='coerce').dropna().astype(int)
                if not drop_idx.empty:
                    mask |= df['original_index'].isin(drop_idx)

            drop_log_run_col = 'run_id' if 'run_id' in drop_log.columns else 'run'
            if not mask.any() and {drop_log_run_col, 'trial_number'}.issubset(drop_log.columns) and 'trial_number' in df.columns:
                drop_pairs = set(
                    (int(row[drop_log_run_col]), int(row['trial_number']))
                    for _, row in drop_log.iterrows()
                    if not pd.isna(row.get(drop_log_run_col)) and not pd.isna(row.get('trial_number'))
                )
                if drop_pairs:
                    run_trial_pairs = df.apply(
                        lambda r: (int(r[run_col]), int(r['trial_number'])), axis=1
                    )
                    mask |= run_trial_pairs.isin(drop_pairs)

            if mask.any():
                df = df.loc[~mask].reset_index(drop=True)
                pass

    temp_mapping = setup_temperature_mapping()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for run_num in sorted(df[run_col].unique()):
        df_run = df[df[run_col] == run_num].copy()

        if len(df_run) == 0:
            continue

        events_processed = process_run_events(df_run, temp_mapping, include_feedback)
        output_filename = f"{subject}_task-pain_run-{run_num:02d}_events.tsv"
        output_file = output_path / output_filename
        events_processed.to_csv(output_file, sep='\t', index=False, float_format='%.6f')

    return True


###################################################################
# Discover and Validate Inputs
###################################################################

def discover_subject_files(config: Dict, subject: str) -> Dict:
    runs = config["runs"]
    inventory = {
        "subject": subject,
        "task": config["task"],
        "space": config["space"],
        "runs": {},
        "anatomical": {},
        "summary": {"total_runs": len(runs), "runs_complete": 0, "missing_files": []}
    }

    try:
        anat = get_subject_files(config, subject, run=1, file_type="anat")
        inventory["anatomical"]["t1w_preproc"] = {"path": str(anat), "exists": anat.exists()}
    except Exception:
        pass

    for run in runs:
        run_key = f"run-{run:02d}"
        files = {}
        for file_type in ("bold", "mask", "confounds", "events"):
            try:
                path = get_subject_files(config, subject, run, file_type)
                files[file_type] = {"path": str(path), "exists": path.exists()}
            except Exception:
                files[file_type] = {"path": "", "exists": False}

        run_entry = {"run_number": run, "files": files, "complete": False}
        inventory["runs"][run_key] = run_entry

        if not all(files[ft]["exists"] for ft in ("bold", "mask", "confounds", "events")):
            inventory["summary"]["missing_files"].extend([fdata["path"] for ft, fdata in files.items() if not fdata["exists"]])
            continue

        try:
            img = nib.load(files["bold"]["path"])
            files["bold"].update({"shape": img.shape, "n_volumes": img.shape[3] if img.ndim == 4 else 1})
            conf_df = pd.read_csv(files["confounds"]["path"], sep="\t")
            files["confounds"].update({"n_rows": len(conf_df), "n_columns": conf_df.shape[1]})
            events_df = pd.read_csv(files["events"]["path"], sep="\t")
            files["events"].update({"n_events": len(events_df), "event_types": sorted(events_df["trial_type"].unique()) if "trial_type" in events_df else []})
            run_entry["complete"] = True
            inventory["summary"]["runs_complete"] += 1
        except Exception as exc:
            log(f"{run_key}: {exc}", "ERROR")

    return inventory


def validate_events(config: Dict, subject: str, inventory: Dict) -> pd.DataFrame:
    temp_labels = set(config["glm"]["temp_labels"])
    expected_raw_duration = 12.5
    nuisance_events = set(config["glm"]["nuisance_events"])

    rows = []
    for run_data in inventory["runs"].values():
        events_path = Path(run_data.get("files", {}).get("events", {}).get("path", ""))
        row = {
            "subject": subject,
            "run": run_data["run_number"],
            "events_file_exists": events_path.exists() if events_path else False,
            "n_total_events": 0,
            "n_heat_trials": 0,
            "heat_duration_mean": np.nan,
            "heat_duration_std": np.nan,
            "n_feedback": 0,
            "valid_heat_count": False,
            "valid_temp_labels": False,
            "valid_durations": False,
            "valid_nuisance": False,
            "errors": []
        }

        if not row["events_file_exists"]:
            row["errors"].append("Missing")
            rows.append(row)
            continue

        try:
            events = pd.read_csv(events_path, sep="\t")
        except Exception:
            row["errors"].append("Read error")
            rows.append(row)
            continue

        row["n_total_events"] = len(events)
        heat_events = events[events["trial_type"].isin(temp_labels)] if "trial_type" in events else pd.DataFrame()
        row["n_heat_trials"] = len(heat_events)

        expected_heat = config["glm"].get("expected_heat_trials_per_run")
        row["valid_heat_count"] = row["n_heat_trials"] > 0
        if expected_heat and row["n_heat_trials"] != expected_heat:
            row["errors"].append(f"Heat={row['n_heat_trials']} (exp {expected_heat})")
        elif row["n_heat_trials"] == 0:
            row["errors"].append("No heat trials")

        if not heat_events.empty:
            row["heat_duration_mean"] = float(heat_events["duration"].mean())
            row["heat_duration_std"] = float(heat_events["duration"].std())
            invalid_temps = set(heat_events["trial_type"].unique()) - temp_labels
            row["valid_temp_labels"] = not invalid_temps
            if invalid_temps:
                row["errors"].append(f"Invalid temps: {sorted(invalid_temps)}")
            duration_ok = abs(row["heat_duration_mean"] - expected_raw_duration) < 1.0 and row["heat_duration_std"] < 0.5
            row["valid_durations"] = duration_ok
            if not duration_ok:
                row["errors"].append(f"Heat duration {row['heat_duration_mean']:.2f}s (expected {expected_raw_duration}s)")

        counts = events.groupby("trial_type").size() if "trial_type" in events else pd.Series(dtype=int)
        row["n_feedback"] = int(counts.get("feedback", 0))
        expected = config["glm"].get("expected_trials_per_run", 11)
        if "feedback" in nuisance_events and row["n_feedback"] != expected:
            row["errors"].append(f"feedback count={row['n_feedback']} (expected {expected})")
        row["valid_nuisance"] = True
        rows.append(row)

    return pd.DataFrame(rows)


###################################################################
# Validation Functions
###################################################################


def validate_confounds(config: Dict, subject: str, inventory: Dict) -> pd.DataFrame:
    motion_24 = set(config['confounds']['motion_24_params'])
    motion_outlier_prefix = config['confounds']['motion_outlier_prefix']
    validation_rows = []

    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        row = {
            'subject': subject,
            'run': run_num,
            'confounds_file_exists': False,
            'n_rows': 0,
            'n_columns': 0,
            'n_motion_24': 0,
            'missing_motion_params': [],
            'n_motion_outliers': 0,
            'motion_outlier_columns': [],
            'n_bold_volumes': 0,
            'rows_match_volumes': False,
            'valid_motion_24': False,
            'errors': []
        }

        if not run_data['files']['confounds']['exists']:
            row['errors'].append('Confounds file missing')
            validation_rows.append(row)
            continue

        row['confounds_file_exists'] = True
        if run_data['files']['bold']['exists']:
            row['n_bold_volumes'] = run_data['files']['bold'].get('n_volumes', 0)

        try:
            confounds = pd.read_csv(run_data['files']['confounds']['path'], sep='\t')
            row['n_rows'], row['n_columns'] = len(confounds), len(confounds.columns)
            columns = set(confounds.columns)
            missing_motion = motion_24 - columns
            row['n_motion_24'] = 24 - len(missing_motion)
            row['missing_motion_params'] = list(missing_motion)
            row['valid_motion_24'] = len(missing_motion) == 0
            if missing_motion:
                row['errors'].append(f"Missing motion params: {missing_motion}")
            outlier_cols = [col for col in confounds.columns if col.startswith(motion_outlier_prefix)]
            row['n_motion_outliers'] = len(outlier_cols)
            row['motion_outlier_columns'] = outlier_cols
            if row['n_bold_volumes'] > 0:
                row['rows_match_volumes'] = row['n_rows'] == row['n_bold_volumes']
                if not row['rows_match_volumes']:
                    row['errors'].append(f"Confounds rows ({row['n_rows']}) != BOLD volumes ({row['n_bold_volumes']})")
        except Exception as e:
            row['errors'].append(f"Error reading confounds: {e}")
        validation_rows.append(row)

    df = pd.DataFrame(validation_rows)
    n_valid, n_total = df['valid_motion_24'].sum(), len(df)
    level = "INFO" if n_valid == n_total else "WARNING"
    log(f"Confounds validation: {n_valid}/{n_total} runs passed", level)
    total_outliers = int(df['n_motion_outliers'].sum())
    if total_outliers > 0:
        log(f"Motion outlier columns detected: {total_outliers}")
    return df




def save_validation_report(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    df.to_csv(output_path, sep='\t', index=False, float_format='%.3f')


###################################################################
# Main
###################################################################

def main():
    parser = argparse.ArgumentParser(description='Prepare data: split events and discover inputs')
    parser.add_argument('--config', default='utils/config.yaml', help='Configuration file')
    parser.add_argument('--subject', default=None, help='Process specific subject')
    parser.add_argument('--work-dir', default='work', help='Working directory')
    parser.add_argument('--qc-dir', default='qc', help='QC directory')
    parser.add_argument('--skip-split-events', action='store_true', help='Skip splitting events (if already done)')
    parser.add_argument('--no-feedback', action='store_true', help='Exclude feedback events')
    parser.add_argument('--drop-log', default=None, help='Path to EEG drop log')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        log(f"Config: {e}", "ERROR")
        return 1

    subjects = [args.subject] if args.subject else config['subjects']
    paths = PipelinePaths.from_config(config, work_dir=args.work_dir, qc_dir=args.qc_dir)
    paths.ensure_core_roots()
    bids_root = Path(config["bids_root"])
    eeg_derivatives_root = Path(config.get("eeg_derivatives_root", "")).resolve() if config.get("eeg_derivatives_root") else None

    all_success = True

    for subject in subjects:
        log(subject)
        if not args.skip_split_events:
            log("Split events")
            candidate_paths = [
                bids_root / subject / "func" / f"{subject}_task-thermalactive_events.tsv",
            ]
            if eeg_derivatives_root:
                candidate_paths.append(eeg_derivatives_root / subject / "eeg" / f"{subject}_task-thermalactive_events.tsv")
                candidate_paths.append(eeg_derivatives_root.parent / subject / "eeg" / f"{subject}_task-thermalactive_events.tsv")

            master_events = next((p for p in candidate_paths if p.exists()), None)
            if master_events is not None:
                output_dir = bids_root / subject / "func"
                drop_log_path = Path(args.drop_log) if args.drop_log else (
                    eeg_derivatives_root / subject / "eeg" / "features" / "dropped_trials.tsv" if eeg_derivatives_root else None
                )
                include_feedback = not args.no_feedback
                if not split_events_to_runs(subject, master_events, output_dir, include_feedback, drop_log_path):
                    all_success = False

        log("Discover files")
        try:
            inventory = discover_subject_files(config, subject)
            index_dir = PipelinePaths.ensure_dir(paths.index_dir)
            save_inventory(inventory, index_dir / f"{subject}_files.json")

            stage_qc_dir = PipelinePaths.ensure_dir(paths.qc_stage_dir("01_prepare_data"))
            events_validation = validate_events(config, subject, inventory)
            save_validation_report(events_validation, stage_qc_dir / f"{subject}_events_check.tsv")
            confounds_validation = validate_confounds(config, subject, inventory)
            save_validation_report(confounds_validation, stage_qc_dir / f"{subject}_confounds_check.tsv")

            if inventory['summary']['missing_files']:
                all_success = False

            events_pass = events_validation['valid_heat_count'].all()
            confounds_pass = confounds_validation['valid_motion_24'].all()

            if not events_pass or not confounds_pass:
                all_success = False
        except Exception as e:
            log(f"{subject}: {e}", "ERROR")
            all_success = False

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

