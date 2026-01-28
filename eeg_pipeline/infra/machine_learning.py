from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.tsv import write_parquet, write_tsv

logger = get_logger(__name__)


###################################################################
# Best Parameters I/O
###################################################################

def prepare_best_params_path(
    base_path: Path,
    mode: str,
    run_id: Optional[str] = None,
) -> Path:
    """Prepare the path for best parameters output based on mode."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "run_scoped":
        resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        output_path = base_path.with_name(
            f"{base_path.stem}_{resolved_run_id}{base_path.suffix}"
        )
    else:
        if mode == "truncate":
            base_path.unlink(missing_ok=True)
        output_path = base_path

    logger.info(f"Best-params mode='{mode}'; resolved path: {output_path}")
    return output_path


###################################################################
# Predictions and Indices Export
###################################################################

def export_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups_ordered: List[str],
    test_indices: List[int],
    fold_ids: List[int],
    model_name: str,
    meta: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    extra_cols = {}
    if "run_id" in meta.columns:
        extra_cols["run_id"] = meta.loc[test_indices, "run_id"].tolist()
    if "block" in meta.columns:
        extra_cols["block"] = meta.loc[test_indices, "block"].tolist()

    pred_df = pd.DataFrame(
        {
            "subject_id": groups_ordered,
            "trial_id": meta.loc[test_indices, "trial_id"].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "fold": fold_ids,
            "model": model_name,
            **extra_cols,
        }
    )

    # Write both TSV (for plotting/back-compat) and parquet (for fast downstream use).
    write_tsv(pred_df, save_path.with_suffix(".tsv"))
    write_parquet(pred_df, save_path.with_suffix(".parquet"))
    return pred_df


def export_indices(
    groups_ordered: List[str],
    test_indices: List[int],
    fold_ids: List[int],
    meta: pd.DataFrame,
    save_path: Path,
    blocks_source: Optional[str] = None,
    add_heldout_subject_id: bool = False,
) -> None:
    extra = {}
    if blocks_source is not None:
        extra["blocks_source"] = blocks_source
    if "run_id" in meta.columns:
        extra["run_id"] = meta.loc[test_indices, "run_id"].tolist()
    if "block" in meta.columns:
        extra["block"] = meta.loc[test_indices, "block"].tolist()

    idx_df = pd.DataFrame(
        {
            "subject_id": groups_ordered,
            "trial_id": meta.loc[test_indices, "trial_id"].values,
            "fold": fold_ids,
            **extra,
        }
    )

    if add_heldout_subject_id:
        idx_df["heldout_subject_id"] = idx_df["subject_id"].astype(str)

    write_tsv(idx_df, save_path.with_suffix(".tsv"))
    write_parquet(idx_df, save_path.with_suffix(".parquet"))


__all__ = [
    "prepare_best_params_path",
    "export_predictions",
    "export_indices",
]
