#!/usr/bin/env python
"""Build the Study 2 source-stage input table from a completed Study 1 root."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.overrides import apply_runtime_overrides
from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.stages import _study1_capable_config
from studies.pain_study.study2.study1_context import load_study1_model_context
from studies.pain_study.study2.contributions import (
    compute_held_out_contribution_scores,
    contribution_band_members,
    standardize_contribution_scores,
)
from studies.pain_study.study2.source_stage_design import contribution_bands
from studies.pain_study.study2.source_stage_trials import (
    SOURCE_TRIAL_INDEX_COLUMNS,
    SOURCE_TRIAL_KEY_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-file", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--study2-config", required=True)
    parser.add_argument("--deriv-root", required=True)
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--study1-root-name", required=True)
    parser.add_argument("--study2-root-name", required=True)
    parser.add_argument("--max-condition-number", type=float, default=None)
    parser.add_argument("--target-retrained-valid-draws", type=int, default=None)
    return parser.parse_args()


def load_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not subjects:
        raise ValueError(f"Study 2 subject list is empty: {path}")
    return subjects


def build_config(args: argparse.Namespace) -> dict:
    set_overrides = [
        f"study2.source_modeling.anatomy.subjects_dir={args.subjects_dir}",
        'study2.source_modeling.anatomy.trans_path_template="{subjects_dir}/{subject}/bem/{subject}-trans.fif"',
        'study2.source_modeling.anatomy.bem_path_template="{subjects_dir}/{subject}/bem/{subject}-5120-5120-5120-bem-sol.fif"',
        f"study2.inputs.study1_root_name={args.study1_root_name}",
        f"study2.outputs.root_name={args.study2_root_name}",
    ]
    if args.max_condition_number is not None:
        set_overrides.append(
            f"study2.source_stage.max_condition_number={args.max_condition_number}"
        )
    if args.target_retrained_valid_draws is not None:
        set_overrides.append(
            "study2.permutations.target_retrained_valid_draws="
            f"{args.target_retrained_valid_draws}"
        )

    config = load_study2_config(args.study2_config)
    apply_runtime_overrides(
        config,
        task=args.task,
        deriv_root=args.deriv_root,
        set_overrides=set_overrides,
    )
    return config


def build_source_stage_frame(
    context,
    config,
    source_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bands = contribution_bands(config)
    raw_scores = compute_held_out_contribution_scores(
        context,
        bands=bands,
        band_members=contribution_band_members(bands),
    )
    score_columns = ("eta_combined", *(f"eta_{band}" for band in bands))
    standardized_scores, qc = standardize_contribution_scores(
        raw_scores,
        subject_column="subject_id",
        score_columns=score_columns,
    )

    frame = context.meta.copy().reset_index(drop=True)
    # The contribution scores are keyed by position in the model matrix, so that position
    # is the join key. The recording's own trial_id has to survive untouched: it is half
    # of the identity the source rows are matched on.
    frame["model_row"] = np.arange(len(frame), dtype=int)
    frame["trial_index_within_run"] = pd.to_numeric(
        frame["within_run_trial"],
        errors="raise",
    )
    standardized_columns = ["trial_id", *(f"{column}_z" for column in score_columns)]
    frame = frame.merge(
        standardized_scores[standardized_columns].rename(columns={"trial_id": "model_row"}),
        on="model_row",
        how="inner",
        validate="one_to_one",
    )
    return _join_source_rows(frame, source_trials), qc


def _join_source_rows(frame: pd.DataFrame, source_trials: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in SOURCE_TRIAL_INDEX_COLUMNS if column not in source_trials.columns
    ]
    if missing:
        raise ValueError(f"Study 2 source-trial index is missing columns: {missing}.")

    keys = list(SOURCE_TRIAL_KEY_COLUMNS)
    left = _normalized_trial_keys(frame, table_name="source-stage frame")
    right = _normalized_trial_keys(
        source_trials.loc[:, list(SOURCE_TRIAL_INDEX_COLUMNS)],
        table_name="source-trial index",
    )
    merged = left.merge(right, on=keys, how="left", validate="one_to_one")
    unmatched = merged["source_row"].isna()
    if unmatched.any():
        unresolved = merged.loc[unmatched, keys].to_dict("records")[:5]
        raise ValueError(
            f"{int(unmatched.sum())} model trials have no source row; first unresolved: "
            f"{unresolved}."
        )
    merged["source_row"] = merged["source_row"].astype(int)
    return merged


def _normalized_trial_keys(frame: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    missing = [column for column in SOURCE_TRIAL_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 {table_name} is missing trial identity columns: {missing}.")
    normalized = frame.copy()
    normalized["subject_id"] = normalized["subject_id"].astype(str)
    for column in ("run", "trial_id"):
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Study 2 {table_name} requires finite {column} values.")
        normalized[column] = values.astype(int)
    return normalized


def main() -> None:
    args = parse_args()
    subjects = load_subjects(Path(args.subjects_file))
    config = build_config(args)

    context = load_study1_model_context(
        subjects=subjects,
        task=args.task,
        config=_study1_capable_config(config),
    )
    source_trials_path = paths.source_trial_index_path(config)
    if not source_trials_path.exists():
        raise FileNotFoundError(
            "Study 2 source-stage input needs the source-trial index written by the "
            f"source-power stage: {source_trials_path}"
        )
    source_trials = pd.read_csv(source_trials_path, sep="\t")
    frame, contribution_qc = build_source_stage_frame(context, config, source_trials)

    output_path = paths.source_stage_frame_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)
    contribution_qc.to_csv(
        output_path.with_name("contribution_qc.tsv"),
        sep="\t",
        index=False,
    )
    print(f"Wrote {len(frame)} rows to {output_path}")
    print(frame.groupby("subject_id").size().to_string())


if __name__ == "__main__":
    main()
