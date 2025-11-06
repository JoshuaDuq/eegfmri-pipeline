from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import logging

from .target_signatures import TargetSignature, SubjectDataset
from .feature_utils import select_direct_power_columns


###################################################################
# Subject Discovery
###################################################################

def discover_subjects(
    eeg_deriv_root: Path,
    fmri_outputs_root: Path,
    target: TargetSignature,
) -> List[str]:
    eeg_subjects = {
        path.parent.parent.parent.name
        for path in eeg_deriv_root.glob("sub-*/eeg/features/features_eeg_direct.tsv")
    }
    fmri_subjects: set[str] = set()
    for spec in target.trial_sources:
        base_dir = fmri_outputs_root.joinpath(*spec.subdir_parts)
        pattern = f"sub-*/{spec.filename}"
        for path in base_dir.glob(pattern):
            fmri_subjects.add(path.parent.name)
    return sorted(eeg_subjects & fmri_subjects)


###################################################################
# Event Alignment
###################################################################

def identify_missing_events(events: pd.DataFrame, ratings: Sequence[float]) -> List[int]:
    missing: List[int] = []
    ev_values = events["vas_final_coded_rating"].tolist()
    ratings_list = list(ratings)
    if len(ev_values) < len(ratings_list):
        raise ValueError(
            f"Events have fewer rows ({len(ev_values)}) than EEG features ({len(ratings_list)})."
        )

    rating_idx = 0
    for event_idx, event_rating in enumerate(ev_values):
        if rating_idx >= len(ratings_list):
            missing.extend(range(event_idx, len(ev_values)))
            break
        if np.isclose(event_rating, ratings_list[rating_idx], rtol=1e-5, atol=1e-4):
            rating_idx += 1
        else:
            missing.append(event_idx)

    if rating_idx != len(ratings_list):
        raise ValueError(
            "Could not perfectly align events to EEG features (matched "
            f"{rating_idx} of {len(ratings_list)} ratings)."
        )

    expected_missing = len(ev_values) - len(ratings_list)
    if len(missing) != expected_missing:
        raise ValueError(
            f"Alignment discrepancy: expected {expected_missing} missing events, found {len(missing)}."
        )
    return missing


###################################################################
# Dataset Loading
###################################################################

def load_subject_dataset(
    subject: str,
    eeg_deriv_root: Path,
    fmri_outputs_root: Path,
    bands: Sequence[str],
    target: TargetSignature,
    logger: logging.Logger,
) -> SubjectDataset:
    subject_dir = eeg_deriv_root / subject / "eeg"
    features_dir = subject_dir / "features"
    features_path = features_dir / "features_eeg_direct.tsv"
    target_path = features_dir / "target_vas_ratings.tsv"
    events_path = eeg_deriv_root.parent / subject / "eeg" / f"{subject}_task-thermalactive_events.tsv"
    trial_path, trial_score_column = target.resolve_trial_file(fmri_outputs_root, subject)

    for path in [features_path, target_path, events_path, trial_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {subject}: {path}")

    direct = pd.read_csv(features_path, sep="\t")
    direct_cols = select_direct_power_columns(direct.columns, bands)
    if not direct_cols:
        sample_cols = list(direct.columns[:10])
        raise ValueError(
            "No direct EEG power columns for bands %s in %s. Sample columns: %s"
            % (bands, features_path, sample_cols)
        )
    direct = direct.loc[:, direct_cols].reset_index(drop=True)

    conn_path = features_dir / "features_connectivity.tsv"
    if conn_path.exists():
        conn_df = pd.read_csv(conn_path, sep="\t")
        conn_cols = [col for col in conn_df.columns if any(band in col for band in bands)]
        if conn_cols:
            conn_df = conn_df.loc[:, conn_cols].reset_index(drop=True)
        else:
            conn_df = pd.DataFrame()
    else:
        conn_df = pd.DataFrame()

    feature_blocks = [direct]
    if not conn_df.empty:
        feature_blocks.append(conn_df)
    
    feature_df = pd.concat(feature_blocks, axis=1) if len(feature_blocks) > 1 else direct
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    target_df = pd.read_csv(target_path, sep="\t")
    target_values = pd.to_numeric(target_df.iloc[:, 0], errors="coerce")
    events_df = pd.read_csv(events_path, sep="\t")

    missing_indices = identify_missing_events(events_df, target_values.tolist())
    dropped_trials: List[Dict[str, float]] = []
    if missing_indices:
        run_col = "run_id" if "run_id" in events_df.columns else "run"
        dropped_view = events_df.iloc[missing_indices][[run_col, "trial_number", "stimulus_temp", "vas_final_coded_rating"]]
        dropped_trials = [
            {
                "run": int(row[run_col]),
                "trial_number": int(row["trial_number"]),
                "temp_celsius": float(row["stimulus_temp"]),
                "vas_rating": float(row["vas_final_coded_rating"]),
            }
            for _, row in dropped_view.iterrows()
        ]
        events_aligned = events_df.drop(index=missing_indices).reset_index(drop=True)
    else:
        events_aligned = events_df.reset_index(drop=True)

    trial_df = pd.read_csv(trial_path, sep="\t")
    if trial_score_column not in trial_df.columns:
        raise ValueError(
            f"Target column '{trial_score_column}' not found in {trial_path}. "
            f"Available columns: {list(trial_df.columns)}"
        )
    if target.score_column not in trial_df.columns:
        trial_df = trial_df.rename(columns={trial_score_column: target.score_column})

    if "trial_idx_run" in trial_df.columns:
        trial_df["trial_idx_run"] = pd.to_numeric(trial_df["trial_idx_run"], errors="coerce")
    elif "trial_regressor" in trial_df.columns:
        extracted = trial_df["trial_regressor"].astype(str).str.extract(r"(\d+)$", expand=False)
        trial_df["trial_idx_run"] = pd.to_numeric(extracted, errors="coerce")
    else:
        raise ValueError(
            f"fMRI trial file for {subject} is missing 'trial_idx_run' information: {trial_path}"
        )
    
    trial_run_col = "run_id" if "run_id" in trial_df.columns else "run"
    trial_df[trial_run_col] = pd.to_numeric(trial_df[trial_run_col], errors="coerce")

    if trial_df["trial_idx_run"].isna().any() or trial_df[trial_run_col].isna().any():
        raise ValueError(
            f"Could not determine run/trial indices for every row in {trial_path}. "
            f"Ensure the file contains integer '{trial_run_col}' and 'trial_idx_run' columns."
        )

    trial_df["trial_idx_run"] = trial_df["trial_idx_run"].astype(int)
    trial_df[trial_run_col] = trial_df[trial_run_col].astype(int)

    duplicate_trial_keys = trial_df.duplicated([trial_run_col, "trial_idx_run"], keep=False)
    if duplicate_trial_keys.any():
        dup_preview = trial_df.loc[duplicate_trial_keys, [trial_run_col, "trial_idx_run"]].head(5).to_dict("records")
        raise ValueError(
            f"Duplicate fMRI trial identifiers detected for {subject} in {trial_path}: {dup_preview}"
        )

    trial_lookup: Dict[Tuple[int, int], int] = {
        (getattr(row, trial_run_col), row.trial_idx_run): row.Index 
        for row in trial_df.reset_index(drop=False).itertuples()
    }

    trial_aligned_indices: List[int] = []
    events_to_keep: List[int] = []
    unmatched_events: List[Dict[str, float]] = []
    used_trial_indices: Set[int] = set()
    
    events_run_col = "run_id" if "run_id" in events_aligned.columns else "run"

    for idx, event_row in events_aligned.iterrows():
        event_trial_number = event_row.get("trial_number")
        if pd.isna(event_trial_number):
            unmatched_events.append(
                {
                    "idx": idx,
                    "run": int(event_row[events_run_col]) if not pd.isna(event_row.get(events_run_col)) else None,
                    "trial": None,
                    "temp": float(event_row["stimulus_temp"]),
                    "vas": float(event_row["vas_final_coded_rating"]),
                    "reason": "missing trial_number in EEG events",
                }
            )
            continue

        event_temp = float(event_row["stimulus_temp"])
        event_vas = float(event_row["vas_final_coded_rating"])
        event_run = int(event_row[events_run_col])
        event_trial_idx = int(event_trial_number) - 1
        if event_trial_idx < 0:
            unmatched_events.append(
                {
                    "idx": idx,
                    "run": event_run,
                    "trial": int(event_trial_number),
                    "temp": event_temp,
                    "vas": event_vas,
                    "reason": "negative derived trial_idx_run",
                }
            )
            continue

        key = (event_run, event_trial_idx)
        match_idx = trial_lookup.get(key)

        if match_idx is None:
            unmatched_events.append(
                {
                    "idx": idx,
                    "run": event_run,
                    "trial": int(event_trial_number),
                    "temp": event_temp,
                    "vas": event_vas,
                    "reason": "fMRI trial missing for run/trial_idx_run key",
                }
            )
            continue

        if match_idx in used_trial_indices:
            unmatched_events.append(
                {
                    "idx": idx,
                    "run": event_run,
                    "trial": int(event_trial_number),
                    "temp": event_temp,
                    "vas": event_vas,
                    "reason": "duplicate fMRI trial assignment encountered",
                }
            )
            continue

        trial_aligned_indices.append(match_idx)
        events_to_keep.append(idx)
        used_trial_indices.add(match_idx)

    if unmatched_events:
        logger.info(
            "  Dropping %d events without matching fMRI %s scores:",
            len(unmatched_events),
            target.short_name,
        )
        for um in unmatched_events[:5]:
            logger.info(
                "    Run %s, trial %s: temp=%.1f, VAS=%.1f (reason: %s)",
                um["run"],
                um["trial"],
                um["temp"],
                um["vas"],
                um.get("reason", "unspecified"),
            )
        if len(unmatched_events) > 5:
            logger.info("    ... and %d more", len(unmatched_events) - 5)

        if not events_to_keep:
            raise ValueError(
                f"No EEG events for {subject} could be aligned to fMRI trials; aborting due to missing keys."
            )

        events_aligned = events_aligned.loc[events_to_keep].reset_index(drop=True)
        feature_df = feature_df.loc[events_to_keep].reset_index(drop=True)

    trial_aligned = trial_df.loc[trial_aligned_indices].reset_index(drop=True)

    if not len(feature_df) == len(events_aligned) == len(trial_aligned):
        raise ValueError(
            f"Alignment mismatch for {subject}: features {len(feature_df)}, events {len(events_aligned)}, "
            f"trial scores {len(trial_aligned)}."
        )

    temp_events = events_aligned["stimulus_temp"].astype(float).to_numpy()
    temp_trial = trial_aligned["temp_celsius"].astype(float).to_numpy()
    vas_events = events_aligned["vas_final_coded_rating"].astype(float).to_numpy()
    vas_trial = trial_aligned["vas_rating"].astype(float).to_numpy()

    temp_mismatch = ~np.isclose(temp_events, temp_trial, rtol=1e-5, atol=1e-3)
    vas_mismatch = ~np.isclose(vas_events, vas_trial, rtol=1e-5, atol=1e-3)

    if np.any(temp_mismatch) or np.any(vas_mismatch):
        n_temp_mismatch = int(np.sum(temp_mismatch))
        n_vas_mismatch = int(np.sum(vas_mismatch))
        raise ValueError(
            f"Alignment validation FAILED for {subject}: temperature mismatches={n_temp_mismatch}, "
            f"VAS mismatches={n_vas_mismatch} (total trials={len(events_aligned)}). "
            "Check upstream preprocessing; EEG and fMRI metadata must be perfectly aligned."
        )
    logger.info("  Alignment validation passed: temperature and VAS ratings match perfectly.")

    metadata = pd.DataFrame(
        {
            "subject": subject,
            "run": events_aligned["run"].astype(int),
            "trial_idx_run": events_aligned["trial_number"].astype(int) - 1,
            "trial_idx_global": trial_aligned["trial_idx_global"].astype(int),
            "temp_celsius": trial_aligned["temp_celsius"].astype(float),
            "vas_rating": trial_aligned["vas_rating"].astype(float),
            "pain_binary": trial_aligned["pain_binary"].astype(int),
            target.score_column: trial_aligned[target.score_column].astype(float),
        }
    )

    data = pd.concat([metadata, feature_df.reset_index(drop=True)], axis=1)
    if "br_score" not in data.columns:
        data["br_score"] = data[target.score_column]
    feature_columns = list(feature_df.columns)

    logger.info(
        "Loaded %d aligned trials for %s (dropped %d trials).",
        len(data),
        subject,
        len(dropped_trials),
    )

    return SubjectDataset(
        subject=subject,
        data=data,
        feature_columns=feature_columns,
        dropped_trials=dropped_trials,
        target_column=target.score_column,
    )

