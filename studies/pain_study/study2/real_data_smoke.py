"""Real-data smoke runner for the Study 2 source-stage pipeline.

This runner is intentionally explicit about its scope. It exercises Study 2
source-stage association and group/family inference on real Study 1 targets and
real plateau EEG feature rows. It does not claim source localization when
subject-level source-power maps are unavailable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.gates import (
    evaluate_study1_confirmatory_criteria,
    load_study1_confirmatory_row,
)
from studies.pain_study.study2.source_family import (
    compute_source_family_inference,
    summarize_source_family,
)
from studies.pain_study.study2.source_maps import (
    CohortSourceAssociationResult,
    compute_cohort_source_association_maps,
)


BANDS = ("alpha", "beta", "gamma")
DEFAULT_DERIVATIVES_ROOT = Path("/Volumes/KINGSTON/EEG_fMRI_data/derivatives")
FEATURE_RELATIVE_PATH = Path("eeg/features/power/features_power_plateau.csv")
PRIMARY_TARGETS_RELATIVE_PATH = Path(
    "group/multimodal/study1/targets/primary_targets.tsv"
)
STUDY1_REPORT_RELATIVE_PATH = Path("group/multimodal/study1/reports/study1_report.tsv")
DEFAULT_RANDOM_SEED = 20260601


@dataclass(frozen=True)
class RealDataSmokeResult:
    output_dir: Path
    subject_ids: tuple[str, ...]
    bands: tuple[str, ...]
    vertex_labels: tuple[str, ...]
    summary_path: Path
    family_summary_path: Path


def run_real_data_smoke(
    *,
    derivatives_root: str | Path = DEFAULT_DERIVATIVES_ROOT,
    output_dir: str | Path,
    force_gate_override: bool,
    n_permutations: int = 10,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_subjects: int | None = None,
    config_path: str | Path | None = None,
) -> RealDataSmokeResult:
    """Run the Study 2 smoke chain on available real Study 1 derivative rows."""
    if not isinstance(force_gate_override, bool):
        raise TypeError("Study 2 real-data smoke force_gate_override must be boolean.")

    derivatives_path = Path(derivatives_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    _require_directory(derivatives_path, name="Study 2 derivatives root")
    _validate_positive_integer(n_permutations, name="n_permutations")
    if max_subjects is not None:
        _validate_positive_integer(max_subjects, name="max_subjects")

    config = _load_smoke_config(config_path)
    report_path = derivatives_path / STUDY1_REPORT_RELATIVE_PATH
    targets_path = derivatives_path / PRIMARY_TARGETS_RELATIVE_PATH

    study1_row = load_study1_confirmatory_row(report_path, config=config)
    study1_gate_qc = evaluate_study1_confirmatory_criteria(study1_row, config)
    if not study1_gate_qc.confirmatory_criteria_met and not force_gate_override:
        unmet_criteria = ", ".join(study1_gate_qc.unmet_criteria)
        raise RuntimeError(f"unmet Study 1 criteria: {unmet_criteria}")

    targets = _load_primary_targets(targets_path)
    requested_subject_ids = _requested_subject_ids(targets, max_subjects=max_subjects)
    feature_tables = _load_feature_tables(
        derivatives_path,
        subject_ids=requested_subject_ids,
    )
    vertex_labels = _shared_channel_labels(feature_tables, config=config)
    stage_frame, source_power_by_band = _build_source_stage_frame(
        targets,
        feature_tables=feature_tables,
        subject_ids=requested_subject_ids,
        vertex_labels=vertex_labels,
    )
    band_results = _compute_band_source_maps(
        stage_frame,
        source_power_by_band=source_power_by_band,
        config=config,
    )
    valid_vertices = _common_valid_vertices(band_results)
    retained_vertex_labels = tuple(
        label for label, is_valid in zip(vertex_labels, valid_vertices, strict=True) if is_valid
    )
    if not retained_vertex_labels:
        raise ValueError("Study 2 real-data smoke found no finite shared vertices.")

    subject_ids = _common_source_valid_subject_ids(band_results)
    if len(subject_ids) < 2:
        raise ValueError(
            "Study 2 real-data smoke requires at least two source-valid subjects."
        )

    observed_maps_by_band = {
        band: result.fisher_z_maps[:, valid_vertices]
        for band, result in band_results.items()
    }
    null_maps_by_band = _sign_flip_null_maps(
        observed_maps_by_band,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    family_result = compute_source_family_inference(
        observed_maps_by_band=observed_maps_by_band,
        null_maps_by_band=null_maps_by_band,
        adjacency=_chain_adjacency(len(retained_vertex_labels)),
        cluster_forming_p=float(
            get_config_value(
                config,
                "study2.source_inference.primary_cluster_forming_p",
                0.01,
            )
        ),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    family_summary_path = output_path / "family_summary.tsv"
    summary_path = output_path / "run_summary.json"
    _write_outputs(
        output_path,
        stage_frame=stage_frame,
        band_results=band_results,
        observed_maps_by_band=observed_maps_by_band,
        valid_vertices=valid_vertices,
        retained_vertex_labels=retained_vertex_labels,
        family_result=family_result,
        family_summary_path=family_summary_path,
    )
    _write_summary(
        summary_path,
        derivatives_root=derivatives_path,
        output_dir=output_path,
        report_path=report_path,
        targets_path=targets_path,
        study1_gate_qc=study1_gate_qc,
        force_gate_override=force_gate_override,
        requested_subject_ids=requested_subject_ids,
        source_valid_subject_ids=subject_ids,
        vertex_labels=retained_vertex_labels,
        n_permutations=n_permutations,
        random_seed=random_seed,
        family_summary_path=family_summary_path,
    )
    return RealDataSmokeResult(
        output_dir=output_path,
        subject_ids=subject_ids,
        bands=BANDS,
        vertex_labels=retained_vertex_labels,
        summary_path=summary_path,
        family_summary_path=family_summary_path,
    )


def _load_smoke_config(config_path: str | Path | None) -> dict[str, Any]:
    default_path = Path(__file__).parent / "config" / "study2_smoketest.yaml"
    config = load_study2_config(config_path or default_path)
    gates = config["study2"]["confirmatory"]["study1_gates"]
    gates["min_level2_delta_r2"] = None
    gates["min_target_split_half_reliability"] = None
    gates["min_target_reliability_n_trials"] = None
    gates["require_positive_within_subject_delta_r2"] = False
    gates["require_artifact_censoring_robustness"] = False
    return config


def _load_primary_targets(targets_path: Path) -> pd.DataFrame:
    _require_file(targets_path, name="Study 1 primary targets")
    targets = pd.read_csv(targets_path, sep="\t")
    _require_columns(
        targets,
        (
            "subject_id",
            "block",
            "trial_index",
            "within_block_trial",
            "onset",
            "hrf_weighted_framewise_displacement",
            "hrf_weighted_std_dvars",
            "hrf_weighted_fp1_fp2_high_frequency_power",
            "residual_ecg_coupling",
            "stimulus_temp",
            "selected_surface",
        ),
        name="Study 1 primary targets",
    )
    prepared = targets.copy()
    prepared["subject_id"] = prepared["subject_id"].astype(str)
    prepared["trial_id"] = prepared.groupby("subject_id", sort=False).cumcount() + 1
    prepared["trial_index_within_block"] = prepared["within_block_trial"]
    return prepared


def _requested_subject_ids(
    targets: pd.DataFrame,
    *,
    max_subjects: int | None,
) -> tuple[str, ...]:
    subject_ids = tuple(sorted(targets["subject_id"].astype(str).unique()))
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]
    if len(subject_ids) < 2:
        raise ValueError(
            "Study 2 real-data smoke requires at least two subjects with Study 1 targets."
        )
    return subject_ids


def _load_feature_tables(
    derivatives_root: Path,
    *,
    subject_ids: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for subject_id in subject_ids:
        feature_path = derivatives_root / subject_id / FEATURE_RELATIVE_PATH
        _require_file(
            feature_path,
            name=f"Study 2 EEG feature file for {subject_id}",
        )
        tables[subject_id] = pd.read_csv(feature_path)
    return tables


def _shared_channel_labels(
    feature_tables: Mapping[str, pd.DataFrame],
    *,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    excluded_channels = {
        str(channel)
        for channel in get_config_value(
            config,
            "study2.source_modeling.rank_excluded_channels",
            [],
        )
    }
    first_order: list[str] = []
    shared_labels: set[str] | None = None
    for subject_id, table in feature_tables.items():
        for band in BANDS:
            labels = [
                label
                for label in _band_channel_labels(table, band=band)
                if label not in excluded_channels
            ]
            if not labels:
                raise ValueError(
                    f"Study 2 real-data smoke found no usable {band} channels "
                    f"for {subject_id}."
                )
            if not first_order:
                first_order = labels
            shared_labels = set(labels) if shared_labels is None else shared_labels & set(labels)

    retained = tuple(label for label in first_order if shared_labels and label in shared_labels)
    if not retained:
        raise ValueError(
            "Study 2 real-data smoke found no shared non-excluded channel labels."
        )
    return retained


def _band_channel_labels(table: pd.DataFrame, *, band: str) -> tuple[str, ...]:
    prefix = f"power_plateau_{band}_ch_"
    suffix = "_logratio"
    labels = [
        column.removeprefix(prefix).removesuffix(suffix)
        for column in table.columns
        if column.startswith(prefix) and column.endswith(suffix)
    ]
    return tuple(labels)


def _build_source_stage_frame(
    targets: pd.DataFrame,
    *,
    feature_tables: Mapping[str, pd.DataFrame],
    subject_ids: tuple[str, ...],
    vertex_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    stage_frames: list[pd.DataFrame] = []
    source_power_by_band: dict[str, dict[str, np.ndarray]] = {band: {} for band in BANDS}
    for subject_id in subject_ids:
        subject_targets = (
            targets.loc[targets["subject_id"] == subject_id]
            .copy()
            .reset_index(drop=True)
        )
        features = feature_tables[subject_id].reset_index(drop=True)
        _validate_trial_alignment(subject_id, subject_targets, features)

        stage_frame = subject_targets.copy()
        standardized_scores: dict[str, np.ndarray] = {}
        for band in BANDS:
            columns = tuple(_feature_column(band, label) for label in vertex_labels)
            _require_columns(features, columns, name=f"{subject_id} {band} features")
            source_power = features.loc[:, columns].to_numpy(dtype=float)
            _require_finite_matrix(source_power, name=f"{subject_id} {band} source power")
            source_power_by_band[band][subject_id] = source_power
            standardized_scores[band] = _zscore(
                np.mean(source_power, axis=1),
                name=f"{subject_id} {band} contribution score",
            )
            stage_frame[f"eta_{band}_z"] = standardized_scores[band]

        combined_score = np.mean(
            np.column_stack([standardized_scores[band] for band in BANDS]),
            axis=1,
        )
        stage_frame["eta_combined_z"] = _zscore(
            combined_score,
            name=f"{subject_id} combined contribution score",
        )
        stage_frames.append(stage_frame)

    return pd.concat(stage_frames, ignore_index=True), source_power_by_band


def _validate_trial_alignment(
    subject_id: str,
    targets: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    if len(targets) != len(features):
        raise ValueError(
            f"Study 2 real-data smoke row mismatch for {subject_id}: "
            f"{len(targets)} targets and {len(features)} feature rows."
        )
    _require_columns(features, ("trial_id",), name=f"{subject_id} features")
    feature_trial_ids = pd.to_numeric(features["trial_id"], errors="coerce").to_numpy()
    target_trial_ids = pd.to_numeric(targets["trial_id"], errors="coerce").to_numpy()
    if not np.array_equal(feature_trial_ids, target_trial_ids):
        raise ValueError(
            f"Study 2 real-data smoke trial_id alignment failed for {subject_id}."
        )


def _compute_band_source_maps(
    stage_frame: pd.DataFrame,
    *,
    source_power_by_band: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, CohortSourceAssociationResult]:
    return {
        band: compute_cohort_source_association_maps(
            stage_frame,
            source_power_by_band[band],
            band=band,
            config=config,
        )
        for band in BANDS
    }


def _common_valid_vertices(
    band_results: Mapping[str, CohortSourceAssociationResult],
) -> np.ndarray:
    first_result = band_results[BANDS[0]]
    valid_vertices = np.ones(first_result.fisher_z_maps.shape[1], dtype=bool)
    for band, result in band_results.items():
        if result.fisher_z_maps.shape[1] != len(valid_vertices):
            raise ValueError(
                "Study 2 real-data smoke band maps do not share a vertex count."
            )
        valid_vertices &= np.all(np.isfinite(result.fisher_z_maps), axis=0)
    return valid_vertices


def _common_source_valid_subject_ids(
    band_results: Mapping[str, CohortSourceAssociationResult],
) -> tuple[str, ...]:
    subject_ids = band_results[BANDS[0]].subject_ids
    for band, result in band_results.items():
        if result.subject_ids != subject_ids:
            raise ValueError(
                "Study 2 real-data smoke band maps do not share source-valid subjects."
            )
    return subject_ids


def _sign_flip_null_maps(
    observed_maps_by_band: Mapping[str, np.ndarray],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    null_maps: dict[str, np.ndarray] = {}
    for band, observed_maps in observed_maps_by_band.items():
        draws = []
        for _draw in range(n_permutations):
            signs = rng.choice(np.array([-1.0, 1.0]), size=observed_maps.shape[0])
            draws.append(observed_maps * signs[:, np.newaxis])
        null_maps[band] = np.stack(draws, axis=0)
    return null_maps


def _chain_adjacency(n_vertices: int) -> np.ndarray:
    _validate_positive_integer(n_vertices, name="n_vertices")
    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for index in range(n_vertices - 1):
        adjacency[index, index + 1] = True
        adjacency[index + 1, index] = True
    return adjacency


def _write_outputs(
    output_dir: Path,
    *,
    stage_frame: pd.DataFrame,
    band_results: Mapping[str, CohortSourceAssociationResult],
    observed_maps_by_band: Mapping[str, np.ndarray],
    valid_vertices: np.ndarray,
    retained_vertex_labels: tuple[str, ...],
    family_result: Any,
    family_summary_path: Path,
) -> None:
    stage_frame.to_csv(output_dir / "source_stage_input.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "vertex_index": np.arange(len(retained_vertex_labels), dtype=int),
            "channel_label": retained_vertex_labels,
        }
    ).to_csv(output_dir / "vertex_labels.tsv", sep="\t", index=False)

    for band, result in band_results.items():
        result.qc.to_csv(output_dir / f"source_stage_qc_{band}.tsv", sep="\t", index=False)
        np.save(output_dir / f"fisher_z_maps_{band}.npy", observed_maps_by_band[band])
        np.save(
            output_dir / f"partial_r_maps_{band}.npy",
            result.partial_r_maps[:, valid_vertices],
        )

    summarize_source_family(family_result).to_csv(
        family_summary_path,
        sep="\t",
        index=False,
    )


def _write_summary(
    summary_path: Path,
    *,
    derivatives_root: Path,
    output_dir: Path,
    report_path: Path,
    targets_path: Path,
    study1_gate_qc: Any,
    force_gate_override: bool,
    requested_subject_ids: tuple[str, ...],
    source_valid_subject_ids: tuple[str, ...],
    vertex_labels: tuple[str, ...],
    n_permutations: int,
    random_seed: int,
    family_summary_path: Path,
) -> None:
    payload = {
        "derivatives_root": str(derivatives_root),
        "output_dir": str(output_dir),
        "study1_report_path": str(report_path),
        "primary_targets_path": str(targets_path),
        "study1_confirmatory_criteria_met": study1_gate_qc.confirmatory_criteria_met,
        "study1_unmet_criteria": list(study1_gate_qc.unmet_criteria),
        "gate_override_requested": force_gate_override,
        "gate_override_applied": bool(
            force_gate_override and not study1_gate_qc.confirmatory_criteria_met
        ),
        "source_localization_input_status": (
            "no_precomputed_source_maps_found; "
            "plateau_eeg_feature_columns_used_as_test_vertices"
        ),
        "requested_subject_ids": list(requested_subject_ids),
        "source_valid_subject_ids": list(source_valid_subject_ids),
        "bands": list(BANDS),
        "n_vertices": len(vertex_labels),
        "vertex_labels": list(vertex_labels),
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "family_summary_path": str(family_summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _feature_column(band: str, channel_label: str) -> str:
    return f"power_plateau_{band}_ch_{channel_label}_logratio"


def _zscore(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Study 2 {name} must be 1D.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Study 2 {name} contains non-finite values.")
    centered = arr - np.mean(arr)
    scale = float(np.std(centered, ddof=0))
    if scale <= 0.0:
        raise ValueError(f"Study 2 {name} has zero variance.")
    return centered / scale


def _require_finite_matrix(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Study 2 {name} contains non-finite values.")


def _require_file(path: Path, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")


def _require_directory(path: Path, *, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {name}: {path}")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}.")


def _validate_positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Study 2 real-data smoke {name} must be a positive integer.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Study 2 source-stage real-data smoke analysis.",
    )
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=DEFAULT_DERIVATIVES_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--force-gate-override",
        action="store_true",
        help="Run despite failed Study 1 confirmatory gates and record the override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_real_data_smoke(
        derivatives_root=args.derivatives_root,
        output_dir=args.output_dir,
        force_gate_override=args.force_gate_override,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        max_subjects=args.max_subjects,
        config_path=args.config,
    )
    print(json.dumps({"summary_path": str(result.summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BANDS",
    "DEFAULT_DERIVATIVES_ROOT",
    "RealDataSmokeResult",
    "run_real_data_smoke",
]
