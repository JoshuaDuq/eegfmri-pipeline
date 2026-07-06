"""Trial/run filtering helpers for Study 2 source-stage analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.machine_learning.circular_shift import admissible_circular_shifts


def permutation_valid_source_runs_with_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    row_index_column = "__source_stage_row_index"
    if row_index_column in frame.columns:
        raise ValueError(f"Study 2 source-stage table uses reserved column '{row_index_column}'.")

    working = frame.reset_index(drop=True).copy()
    working[row_index_column] = np.arange(len(working), dtype=int)
    valid_frame = permutation_valid_source_runs(working)
    retained_row_indices = valid_frame.pop(row_index_column).to_numpy(dtype=int)
    return valid_frame, retained_row_indices


def permutation_valid_source_runs(frame: pd.DataFrame) -> pd.DataFrame:
    runs = pd.to_numeric(frame["run"], errors="coerce")
    trial_indices = pd.to_numeric(frame["trial_index"], errors="coerce")
    if runs.isna().any() or trial_indices.isna().any():
        raise ValueError("Source-stage run and trial_index columns must be finite.")

    valid_indices: list[int] = []
    working = frame.copy()
    working["run"] = runs.to_numpy(dtype=int)
    working["trial_index"] = trial_indices.to_numpy(dtype=int)
    for _run, run_frame in working.groupby("run", sort=True):
        ordered = run_frame.sort_values("trial_index")
        admissible = admissible_circular_shifts(
            ordered["trial_index"].to_numpy(dtype=int),
        )
        if admissible:
            valid_indices.extend(ordered.index.tolist())

    if not valid_indices:
        return working.iloc[0:0].copy()
    return working.loc[sorted(valid_indices)].reset_index(drop=True)


__all__ = [
    "permutation_valid_source_runs",
    "permutation_valid_source_runs_with_rows",
]
