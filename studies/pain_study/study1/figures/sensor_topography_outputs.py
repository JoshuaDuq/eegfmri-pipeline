"""Atomic publication families for Study 1 sensor topographies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
from matplotlib.figure import Figure

from eeg_pipeline.infra.paths import find_clean_events_path
from eeg_pipeline.infra.tsv import write_parquet, write_tsv
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.cohort import (
    primary_targets_parquet_path,
    study1_feature_table_path,
)
from studies.pain_study.study1.figures.sensor_cluster_inference import (
    SensorClusterResult,
    compute_sensor_cluster_inference,
)
from studies.pain_study.study1.figures.sensor_topography_data import (
    SensorMontage,
    load_sensor_montage,
    load_sensor_power_data,
)
from studies.pain_study.study1.figures.sensor_topography_estimands import ParticipantEffects
from studies.pain_study.study1.figures.sensor_topography_plot import (
    SensorTopographyPlotSummary,
    build_sensor_topography_figure,
)
from studies.pain_study.study1.figures.validity_style import (
    save_publication_png,
    save_publication_svg,
)

FigureFamily = Literal["construct", "signature"]
EffectBuilder = Callable[[Any, Any], ParticipantEffects]
FIGURE_CONFIG_PATH = "study1.figures.sensor_topographies"

CONSTRUCT_EFFECT_COLUMNS = (
    "effect_value",
    "partial_r",
    "fisher_z",
    "inference_value",
    "n_trials",
    "n_runs",
    "subject_id",
    "estimand",
    "band",
    "channel",
    "channel_scope",
    "include_fp1_fp2",
    "design_rank",
    "design_parameters",
    "residual_degrees_of_freedom",
    "condition_number",
)
SIGNATURE_EFFECT_COLUMNS = (
    "effect_value",
    "partial_r",
    "fisher_z",
    "inference_value",
    "n_trials",
    "n_runs",
    "resolved_nuisance_columns",
    "omitted_constant_nuisance_columns",
    "retained_nuisance_columns",
    "design_parameters",
    "residual_degrees_of_freedom",
    "minimum_singular_value_ratio",
    "condition_number",
    "minimum_standardized_nuisance_norm",
    "maximum_standardized_nuisance_norm",
    "subject_id",
    "estimand",
    "band",
    "channel",
    "channel_scope",
    "include_fp1_fp2",
)
SUMMARY_COLUMNS = (
    "estimand",
    "band",
    "channel",
    "mean_inference_value",
    "display_value",
    "n_participants",
)
EXCLUSION_COLUMNS = ("subject_id", "reason", "estimand", "band", "channel")
SENSOR_COLUMNS = (
    "estimand",
    "band",
    "sensor",
    "sensor_index",
    "cohort_value",
    "t_statistic",
    "significant_corrected",
    "significant_cluster_ids",
)
CLUSTER_COLUMNS = (
    "estimand",
    "band",
    "cluster_id",
    "sign",
    "sensors",
    "extent",
    "mass",
    "corrected_p_value",
    "significant",
)
FAMILY_COLUMNS = (
    "participant_order",
    "n_participants",
    "n_maps",
    "n_sensors",
    "degrees_of_freedom",
    "inference_available",
    "inference_reason",
    "cluster_forming_p",
    "positive_threshold",
    "negative_threshold",
    "family_alpha",
    "requested_max_null_draws",
    "sampled_null_draws",
    "total_exact_patterns",
    "observed_included",
    "exact_enumeration",
    "seed",
    "permutation_method",
    "correction_method",
    "cluster_mass",
)


@dataclass(frozen=True)
class SensorTopographyPaths:
    """Exact core publication paths for one sensor-topography family."""

    svg: Path
    png: Path
    by_subject_tsv: Path
    by_subject_parquet: Path
    sensors_tsv: Path
    sensors_parquet: Path
    clusters_tsv: Path
    family_tsv: Path
    caption: Path
    manifest: Path

    @property
    def all_files(self) -> tuple[Path, ...]:
        return (
            self.svg,
            self.png,
            self.by_subject_tsv,
            self.by_subject_parquet,
            self.sensors_tsv,
            self.sensors_parquet,
            self.clusters_tsv,
            self.family_tsv,
            self.caption,
            self.manifest,
        )


@dataclass(frozen=True)
class ConstructSensorTopographyPaths(SensorTopographyPaths):
    """Core publication paths plus construct channel-sensitivity audits."""

    sensitivity_by_subject_tsv: Path
    sensitivity_by_subject_parquet: Path
    sensitivity_summary_tsv: Path
    sensitivity_summary_parquet: Path

    @property
    def all_files(self) -> tuple[Path, ...]:
        return (
            *super().all_files[:-1],
            self.sensitivity_by_subject_tsv,
            self.sensitivity_by_subject_parquet,
            self.sensitivity_summary_tsv,
            self.sensitivity_summary_parquet,
            self.manifest,
        )


def sensor_topography_paths(
    svg: Path,
    *,
    include_sensitivity: bool,
) -> SensorTopographyPaths:
    """Build an immutable exact family from its strict SVG path."""

    output = Path(svg)
    if output.suffix != ".svg":
        raise ValueError(f"Study 1 sensor-topography output must be an SVG: {output}.")
    stem = output.stem
    core = {
        "svg": output,
        "png": output.with_suffix(".png"),
        "by_subject_tsv": output.with_name(f"{stem}_by_subject.tsv"),
        "by_subject_parquet": output.with_name(f"{stem}_by_subject.parquet"),
        "sensors_tsv": output.with_name(f"{stem}_sensors.tsv"),
        "sensors_parquet": output.with_name(f"{stem}_sensors.parquet"),
        "clusters_tsv": output.with_name(f"{stem}_clusters.tsv"),
        "family_tsv": output.with_name(f"{stem}_family.tsv"),
        "caption": output.with_name(f"{stem}_caption.txt"),
        "manifest": output.with_name(f"{stem}_manifest.json"),
    }
    if not include_sensitivity:
        return SensorTopographyPaths(**core)
    return ConstructSensorTopographyPaths(
        **core,
        sensitivity_by_subject_tsv=output.with_name(f"{stem}_sensitivity_by_subject.tsv"),
        sensitivity_by_subject_parquet=output.with_name(f"{stem}_sensitivity_by_subject.parquet"),
        sensitivity_summary_tsv=output.with_name(f"{stem}_sensitivity_summary.tsv"),
        sensitivity_summary_parquet=output.with_name(f"{stem}_sensitivity_summary.parquet"),
    )


def write_sensor_topography_family(
    *,
    task: str,
    config: Any,
    output_path: Path,
    family: FigureFamily,
    effect_builder: EffectBuilder,
) -> SensorTopographyPaths:
    """Run the established analysis stages and publish one complete family."""

    data = load_sensor_power_data(task=task, config=config)
    effects = effect_builder(data, config)
    montage = load_sensor_montage(effects.sensor_order, config)
    inference = compute_sensor_cluster_inference(
        effects=effects,
        positions_xy=montage.positions_xy,
        config=config,
    )
    summary = SensorTopographyPlotSummary.from_cluster_result(
        inference,
        positions_xy=montage.positions_xy,
    )
    figure = build_sensor_topography_figure(summary, config)
    return publish_sensor_topography_family(
        figure=figure,
        effects=effects,
        inference=inference,
        montage=montage,
        config=config,
        source_paths=_source_paths(task=task, subjects=data.subjects, config=config),
        output_path=output_path,
        family=family,
    )


def publish_sensor_topography_family(
    *,
    figure: Figure,
    effects: ParticipantEffects,
    inference: SensorClusterResult,
    montage: SensorMontage,
    config: Any,
    source_paths: Sequence[Path],
    output_path: Path,
    family: FigureFamily,
) -> SensorTopographyPaths:
    """Stage, validate, and atomically promote one full publication family."""

    _validate_family_name(family)
    include_sensitivity = family == "construct"
    outputs = sensor_topography_paths(
        Path(output_path),
        include_sensitivity=include_sensitivity,
    )
    tables = _validated_tables(effects=effects, inference=inference, family=family)
    sources = _validated_source_paths(source_paths)
    _validate_montage(montage, inference)
    figure_config = _figure_config(config)
    outputs.svg.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        dir=outputs.svg.parent,
        prefix=f".{outputs.svg.stem}-",
    ) as temporary:
        staged = sensor_topography_paths(
            Path(temporary) / outputs.svg.name,
            include_sensitivity=include_sensitivity,
        )
        _write_staged_family(
            figure=figure,
            effects=effects,
            inference=inference,
            montage=montage,
            config=config,
            figure_config=figure_config,
            sources=sources,
            family=family,
            tables=tables,
            outputs=staged,
        )
        _validate_staged_family(staged)
        _promote_family(staged, outputs)
    return outputs


def _write_staged_family(
    *,
    figure: Figure,
    effects: ParticipantEffects,
    inference: SensorClusterResult,
    montage: SensorMontage,
    config: Any,
    figure_config: Mapping[str, object],
    sources: tuple[Path, ...],
    family: FigureFamily,
    tables: Mapping[str, pd.DataFrame],
    outputs: SensorTopographyPaths,
) -> None:
    dimensions = figure_config["dimensions_mm"]
    _save_png(
        figure,
        outputs.png,
        config,
        dimensions_mm=dimensions,
        dpi=int(figure_config["png_dpi"]),
    )
    _save_svg(figure, outputs.svg, config, dimensions_mm=dimensions)
    _write_table_pair(
        tables["by_subject"],
        outputs.by_subject_tsv,
        outputs.by_subject_parquet,
    )
    _write_table_pair(tables["sensors"], outputs.sensors_tsv, outputs.sensors_parquet)
    write_tsv(tables["clusters"], outputs.clusters_tsv)
    write_tsv(tables["family"], outputs.family_tsv)
    if family == "construct":
        if not isinstance(outputs, ConstructSensorTopographyPaths):
            raise TypeError("Construct publication requires construct output paths.")
        _write_table_pair(
            tables["sensitivity_by_subject"],
            outputs.sensitivity_by_subject_tsv,
            outputs.sensitivity_by_subject_parquet,
        )
        _write_table_pair(
            tables["sensitivity_summary"],
            outputs.sensitivity_summary_tsv,
            outputs.sensitivity_summary_parquet,
        )
    _write_text(outputs.caption, _caption(family, len(effects.participant_order)))
    _write_manifest(
        outputs.manifest,
        effects=effects,
        inference=inference,
        montage=montage,
        config=config,
        sources=sources,
        family=family,
        outputs=outputs,
    )


def _validated_tables(
    *,
    effects: ParticipantEffects,
    inference: SensorClusterResult,
    family: FigureFamily,
) -> dict[str, pd.DataFrame]:
    if not isinstance(effects, ParticipantEffects):
        raise TypeError("Sensor-topography publication requires ParticipantEffects.")
    if not isinstance(inference, SensorClusterResult):
        raise TypeError("Sensor-topography publication requires SensorClusterResult.")
    effect_columns = CONSTRUCT_EFFECT_COLUMNS if family == "construct" else SIGNATURE_EFFECT_COLUMNS
    _require_schema(effects.effects, effect_columns, "participant effects")
    _require_schema(effects.summary, SUMMARY_COLUMNS, "cohort summary")
    _require_schema(effects.exclusions, EXCLUSION_COLUMNS, "participant exclusions")
    if effects.participant_order != inference.participant_order:
        raise ValueError("Effect and inference participant orders differ.")
    if effects.map_order != inference.map_order:
        raise ValueError("Effect and inference map orders differ.")
    if effects.sensor_order != inference.sensor_order:
        raise ValueError("Effect and inference sensor orders differ.")
    if not effects.participant_order:
        raise ValueError("Sensor-topography publication requires included participants.")
    if not np.isfinite(effects.effects["inference_value"].to_numpy(dtype=float)).all():
        raise ValueError("Participant inference values must be finite before publication.")

    sensors = inference.sensor_frame()
    clusters = inference.cluster_frame()
    family_frame = inference.family_frame()
    _require_schema(sensors, SENSOR_COLUMNS, "sensor inference")
    _require_schema(clusters, CLUSTER_COLUMNS, "cluster inference")
    _require_schema(family_frame, FAMILY_COLUMNS, "family inference")
    tables = {
        "by_subject": effects.effects,
        "sensors": sensors,
        "clusters": clusters,
        "family": family_frame,
    }
    if family == "construct":
        if effects.sensitivity_effects is None or effects.sensitivity_summary is None:
            raise ValueError("Construct publication requires both channel-sensitivity tables.")
        _require_schema(
            effects.sensitivity_effects,
            CONSTRUCT_EFFECT_COLUMNS,
            "sensitivity participant effects",
        )
        _require_schema(
            effects.sensitivity_summary,
            SUMMARY_COLUMNS,
            "sensitivity cohort summary",
        )
        tables["sensitivity_by_subject"] = effects.sensitivity_effects
        tables["sensitivity_summary"] = effects.sensitivity_summary
    elif effects.sensitivity_effects is not None or effects.sensitivity_summary is not None:
        raise ValueError("Signature publication cannot contain construct sensitivity tables.")
    return tables


def _require_schema(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Sensor-topography {label} must be a pandas DataFrame.")
    observed = tuple(str(column) for column in frame.columns)
    if observed != columns:
        raise ValueError(
            f"Sensor-topography {label} has an invalid schema: "
            f"expected={list(columns)}, observed={list(observed)}."
        )


def _write_table_pair(frame: pd.DataFrame, tsv: Path, parquet: Path) -> None:
    write_tsv(frame, tsv)
    write_parquet(frame, parquet)


def _write_text(path: Path, content: str) -> None:
    _write_text_unchecked(path, content)


def _write_text_unchecked(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    effects: ParticipantEffects,
    inference: SensorClusterResult,
    montage: SensorMontage,
    config: Any,
    sources: tuple[Path, ...],
    family: FigureFamily,
    outputs: SensorTopographyPaths,
) -> None:
    output_files = tuple(output for output in outputs.all_files if output != path)
    payload = {
        "schema_version": 1,
        "figure_family": family,
        "config": _manifest_config(config, family),
        "source_sha256": {str(source): _sha256(source) for source in sources},
        "output_sha256": {output.name: _sha256(output) for output in output_files},
        "participant_count": len(effects.participant_order),
        "participant_order": list(effects.participant_order),
        "participant_exclusions": _records(effects.exclusions),
        "channel_order": list(effects.sensor_order),
        "montage": {
            "name": montage.name,
            "channels": list(montage.channels),
            "positions_3d": [list(position) for position in montage.positions_3d],
            "positions_xy": [list(position) for position in montage.positions_xy],
        },
        "adjacency": {
            "method": "Delaunay top-view triangulation",
            "coordinate_system": "configured montage x-y projection",
            "matrix": [list(row) for row in inference.adjacency],
            "connected": True,
        },
        "inference": {
            "available": inference.inference_available,
            "reason": inference.inference_reason,
            "permutation_method": "synchronized participant sign flip",
            "correction_method": "joint maximum cluster mass across all ten maps",
            "requested_sign_count": inference.requested_max_null_draws,
            "actual_sign_count": inference.sampled_null_draws,
            "total_exact_sign_count": inference.total_exact_patterns,
            "observed_label_included": inference.observed_included,
            "exact_enumeration": inference.exact_enumeration,
            "seed": inference.seed,
        },
        "software": _software_versions(),
    }
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _manifest_config(config: Any, family: FigureFamily) -> dict[str, object]:
    values: dict[str, object] = {
        "sensor_topographies": require_config_value(config, FIGURE_CONFIG_PATH),
        "montage": require_config_value(config, "eeg.montage"),
    }
    if family == "construct":
        values["construct_channels"] = require_config_value(
            config,
            "study1.figures.power_construct_validity.channels",
        )
        values["construct_rating_model"] = require_config_value(
            config,
            "study1.figures.power_construct_validity.rating_model",
        )
        values["temperatures"] = require_config_value(
            config,
            "study1.figures.validity.temperatures",
        )
    else:
        values["excluded_channels"] = require_config_value(
            config,
            "study1.feature_benchmark.excluded_channels",
        )
        values["target_nuisance_regression"] = require_config_value(
            config,
            "study1.targets.nuisance_regression",
        )
        values["signature_sample_requirements"] = require_config_value(
            config,
            "study1.feature_benchmark.circular_shift",
        )
    return _json_mapping(values)


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "mne": mne.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": importlib.metadata.version("pyarrow"),
        "scipy": scipy.__version__,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_json_mapping(record) for record in frame.to_dict(orient="records")]


def _json_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_value(value) for key, value in mapping.items()}


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Manifest value is not JSON serializable: {type(value).__name__}.")


def _caption(family: FigureFamily, participant_count: int) -> str:
    if family == "construct":
        estimands = (
            "Delivered-temperature dB-per-degree effects and subjective-intensity "
            "partial correlations beyond temperature"
        )
        qualification = ""
    else:
        estimands = "NPS and SIIPS1 partial correlations"
        qualification = " These are univariate associations, not model importance."
    return (
        f"{estimands} across two rows and five frequency bands (n = {participant_count}). "
        "Maps show unthresholded cohort effects at measured EEG sensors; dark rings mark "
        "sensors in clusters surviving the synchronized sign-flip joint family-wise error "
        "correction across both rows and all bands. Results are in sensor space and are not "
        f"source localization.{qualification}\n"
    )


def _source_paths(*, task: str, subjects: Sequence[str], config: Any) -> tuple[Path, ...]:
    paths: list[Path] = [primary_targets_parquet_path(config)]
    for subject in subjects:
        event_path = find_clean_events_path(subject, task, config=config)
        if event_path is None:
            raise FileNotFoundError(
                f"Study 1 clean-event provenance path is missing for {subject}, task-{task}."
            )
        paths.extend((Path(event_path), study1_feature_table_path(config, subject, "power")))
    return _validated_source_paths(paths)


def _validated_source_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = tuple(dict.fromkeys(Path(path).expanduser().resolve() for path in paths))
    if not resolved:
        raise ValueError("Sensor-topography provenance requires source files.")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sensor-topography provenance files are missing: {missing}.")
    return resolved


def _validate_montage(montage: SensorMontage, inference: SensorClusterResult) -> None:
    if not isinstance(montage, SensorMontage):
        raise TypeError("Sensor-topography publication requires SensorMontage.")
    if montage.channels != inference.sensor_order:
        raise ValueError("Montage and inference channel orders differ.")
    adjacency = np.asarray(inference.adjacency, dtype=bool)
    if adjacency.shape != (len(montage.channels), len(montage.channels)):
        raise ValueError("Inference adjacency does not match the montage channel count.")


def _figure_config(config: Any) -> Mapping[str, object]:
    value = require_config_value(config, FIGURE_CONFIG_PATH)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_PATH} must be a mapping.")
    return value


def _validate_family_name(family: str) -> None:
    if family not in {"construct", "signature"}:
        raise ValueError(f"Unsupported sensor-topography family: {family!r}.")


def _validate_staged_family(outputs: SensorTopographyPaths) -> None:
    files = outputs.all_files
    if files[-1] != outputs.manifest:
        raise ValueError("Sensor-topography manifest must be the final family member.")
    if len(files) != len(set(files)):
        raise ValueError("Sensor-topography staged family contains duplicate paths.")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise OSError(f"Sensor-topography staged family is incomplete: {missing}.")
    payload = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    expected = {path.name: _sha256(path) for path in files if path != outputs.manifest}
    if payload.get("output_sha256") != expected:
        raise ValueError("Sensor-topography staged output checksums are invalid.")


def _promote_family(
    staged: SensorTopographyPaths,
    destination: SensorTopographyPaths,
) -> None:
    if len(staged.all_files) != len(destination.all_files):
        raise ValueError("Staged and destination sensor-topography families differ.")
    backup_directory = staged.svg.parent / "backup"
    backup_directory.mkdir()
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for destination_path in destination.all_files:
            if destination_path.exists():
                backup = backup_directory / destination_path.name
                _replace_path(destination_path, backup)
                backups.append((destination_path, backup))
        for staged_path, destination_path in zip(
            staged.all_files[:-1],
            destination.all_files[:-1],
            strict=True,
        ):
            _replace_path(staged_path, destination_path)
            promoted.append(destination_path)
        _replace_path(staged.manifest, destination.manifest)
        promoted.append(destination.manifest)
    except OSError as promotion_error:
        try:
            _rollback_promotion(promoted, backups)
        except OSError as rollback_error:
            promotion_error.add_note(f"Publication-family rollback failed: {rollback_error}")
        raise


def _rollback_promotion(
    promoted: Sequence[Path],
    backups: Sequence[tuple[Path, Path]],
) -> None:
    for promoted_path in promoted:
        promoted_path.unlink(missing_ok=True)
    for destination_path, backup_path in backups:
        _replace_path(backup_path, destination_path)


def _replace_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _save_png(
    figure: Figure,
    path: Path,
    config: Any,
    **kwargs: object,
) -> Path:
    return save_publication_png(figure, path, config, **kwargs)


def _save_svg(
    figure: Figure,
    path: Path,
    config: Any,
    **kwargs: object,
) -> Path:
    return save_publication_svg(figure, path, config, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CLUSTER_COLUMNS",
    "CONSTRUCT_EFFECT_COLUMNS",
    "ConstructSensorTopographyPaths",
    "EXCLUSION_COLUMNS",
    "FAMILY_COLUMNS",
    "SENSOR_COLUMNS",
    "SIGNATURE_EFFECT_COLUMNS",
    "SUMMARY_COLUMNS",
    "SensorTopographyPaths",
    "publish_sensor_topography_family",
    "sensor_topography_paths",
    "write_sensor_topography_family",
]
