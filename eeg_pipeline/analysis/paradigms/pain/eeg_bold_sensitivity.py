"""Sensitivity analysis configuration and trial filtering for EEG-BOLD coupling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _require_sequence(value: Any, *, path: str) -> Tuple[Any, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    return tuple(value)


@dataclass(frozen=True)
class PainfulOnlySensitivityConfig:
    enabled: bool
    binary_column: Optional[str]
    painful_values: Tuple[str, ...]
    rating_column: Optional[str]
    min_rating: Optional[float]
    output_name: str


@dataclass(frozen=True)
class AlternativeFMRISensitivityConfig:
    enabled: bool
    name: str
    beta_dir_template: Optional[str]


@dataclass(frozen=True)
class DeltaTemperatureSensitivityConfig:
    enabled: bool
    output_name: str
    temperature_column: str
    model_terms: Tuple[str, ...]


@dataclass(frozen=True)
class TemperatureCategoricalSensitivityConfig:
    enabled: bool
    output_name: str
    temperature_column: str
    factor_column: str
    model_terms: Tuple[str, ...]
    max_levels: Optional[int]


@dataclass(frozen=True)
class ResidualizedCorrelationSensitivityConfig:
    enabled: bool
    output_name: str
    bootstrap_iterations: int
    permutation_iterations: int


@dataclass(frozen=True)
class PrimaryPermutationSensitivityConfig:
    enabled: bool
    output_name: str
    bootstrap_iterations: int
    permutation_iterations: int


@dataclass(frozen=True)
class SourceMethodItemConfig:
    name: str
    method: str
    bands: Tuple[str, ...]


@dataclass(frozen=True)
class SourceMethodSensitivityConfig:
    enabled: bool
    items: Tuple[SourceMethodItemConfig, ...]


@dataclass(frozen=True)
class AnatomicalSpecificityItemConfig:
    name: str
    rois: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class AnatomicalSpecificitySensitivityConfig:
    enabled: bool
    items: Tuple[AnatomicalSpecificityItemConfig, ...]


@dataclass(frozen=True)
class WithinBetweenSensitivityConfig:
    enabled: bool
    output_name: str


@dataclass(frozen=True)
class ArtifactModelItemConfig:
    name: str
    nuisance_overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ArtifactModelSensitivityConfig:
    enabled: bool
    items: Tuple[ArtifactModelItemConfig, ...]


@dataclass(frozen=True)
class CouplingSensitivityConfig:
    painful_only: PainfulOnlySensitivityConfig
    alternative_fmri: AlternativeFMRISensitivityConfig
    delta_temperature: DeltaTemperatureSensitivityConfig
    temperature_categorical: TemperatureCategoricalSensitivityConfig
    residualized_correlation: ResidualizedCorrelationSensitivityConfig
    primary_permutation: PrimaryPermutationSensitivityConfig
    source_methods: SourceMethodSensitivityConfig
    anatomical_specificity: AnatomicalSpecificitySensitivityConfig
    within_between: WithinBetweenSensitivityConfig
    artifact_models: ArtifactModelSensitivityConfig

    @classmethod
    def from_config(cls, config: Any) -> "CouplingSensitivityConfig":
        raw = _require_mapping(
            get_config_value(config, "eeg_bold_coupling.sensitivities", {}),
            path="eeg_bold_coupling.sensitivities",
        )
        painful_raw = _require_mapping(
            raw.get("painful_only", {}),
            path="eeg_bold_coupling.sensitivities.painful_only",
        )
        alternative_raw = _require_mapping(
            raw.get("alternative_fmri", {}),
            path="eeg_bold_coupling.sensitivities.alternative_fmri",
        )
        delta_raw = _require_mapping(
            raw.get("delta_temperature", {}),
            path="eeg_bold_coupling.sensitivities.delta_temperature",
        )
        categorical_raw = _require_mapping(
            raw.get("temperature_categorical", {}),
            path="eeg_bold_coupling.sensitivities.temperature_categorical",
        )
        residualized_raw = _require_mapping(
            raw.get("residualized_correlation", {}),
            path="eeg_bold_coupling.sensitivities.residualized_correlation",
        )
        primary_permutation_raw = _require_mapping(
            raw.get("primary_permutation", {}),
            path="eeg_bold_coupling.sensitivities.primary_permutation",
        )
        source_raw = _require_mapping(
            raw.get("source_methods", {}),
            path="eeg_bold_coupling.sensitivities.source_methods",
        )
        anatomical_raw = _require_mapping(
            raw.get("anatomical_specificity", {}),
            path="eeg_bold_coupling.sensitivities.anatomical_specificity",
        )
        within_between_raw = _require_mapping(
            raw.get("within_between", {}),
            path="eeg_bold_coupling.sensitivities.within_between",
        )
        artifact_raw = _require_mapping(
            raw.get("artifact_models", {}),
            path="eeg_bold_coupling.sensitivities.artifact_models",
        )
        painful_only = PainfulOnlySensitivityConfig(
            enabled=bool(painful_raw.get("enabled", False)),
            binary_column=str(painful_raw.get("binary_column", "")).strip() or None,
            painful_values=tuple(
                str(value).strip()
                for value in _require_sequence(
                    painful_raw.get("painful_values"),
                    path="eeg_bold_coupling.sensitivities.painful_only.painful_values",
                )
                if str(value).strip()
            ),
            rating_column=str(painful_raw.get("rating_column", "")).strip() or None,
            min_rating=(
                None
                if painful_raw.get("min_rating", None) in {None, ""}
                else float(painful_raw.get("min_rating"))
            ),
            output_name=str(
                painful_raw.get("output_name", "painful_only")
            ).strip(),
        )
        if painful_only.enabled:
            has_binary = painful_only.binary_column is not None and bool(painful_only.painful_values)
            has_rating = painful_only.rating_column is not None and painful_only.min_rating is not None
            if not (has_binary or has_rating):
                raise ValueError(
                    "Painful-only sensitivity requires either binary_column + painful_values or rating_column + min_rating."
                )
            if not painful_only.output_name:
                raise ValueError(
                    "Painful-only sensitivity output_name must not be blank."
                )

        alternative_fmri = AlternativeFMRISensitivityConfig(
            enabled=bool(alternative_raw.get("enabled", False)),
            name=str(alternative_raw.get("name", "alternative_fmri")).strip(),
            beta_dir_template=str(
                alternative_raw.get("beta_dir_template", "")
            ).strip()
            or None,
        )
        if alternative_fmri.enabled and not alternative_fmri.beta_dir_template:
            raise ValueError(
                "Alternative fMRI sensitivity requires a beta_dir_template."
            )
        if alternative_fmri.enabled and not alternative_fmri.name:
            raise ValueError(
                "Alternative fMRI sensitivity name must not be blank."
            )
        delta_temperature = DeltaTemperatureSensitivityConfig(
            enabled=bool(delta_raw.get("enabled", False)),
            output_name=str(delta_raw.get("output_name", "delta_temperature")).strip(),
            temperature_column=str(delta_raw.get("temperature_column", "temperature")).strip(),
            model_terms=tuple(
                str(value).strip()
                for value in _require_sequence(
                    delta_raw.get("model_terms"),
                    path="eeg_bold_coupling.sensitivities.delta_temperature.model_terms",
                )
                if str(value).strip()
            ),
        )
        if delta_temperature.enabled:
            if not delta_temperature.output_name:
                raise ValueError("Delta-temperature sensitivity output_name must not be blank.")
            if not delta_temperature.temperature_column:
                raise ValueError("Delta-temperature sensitivity temperature_column must not be blank.")
            if not delta_temperature.model_terms:
                raise ValueError("Delta-temperature sensitivity model_terms must not be empty.")

        temperature_categorical = TemperatureCategoricalSensitivityConfig(
            enabled=bool(categorical_raw.get("enabled", False)),
            output_name=str(categorical_raw.get("output_name", "temperature_categorical")).strip(),
            temperature_column=str(categorical_raw.get("temperature_column", "temperature")).strip(),
            factor_column=str(categorical_raw.get("factor_column", "temperature_factor")).strip(),
            model_terms=tuple(
                str(value).strip()
                for value in _require_sequence(
                    categorical_raw.get("model_terms"),
                    path="eeg_bold_coupling.sensitivities.temperature_categorical.model_terms",
                )
                if str(value).strip()
            ),
            max_levels=(
                None
                if categorical_raw.get("max_levels", None) in {None, ""}
                else int(categorical_raw.get("max_levels"))
            ),
        )
        if temperature_categorical.enabled:
            if not temperature_categorical.output_name:
                raise ValueError(
                    "Temperature-categorical sensitivity output_name must not be blank."
                )
            if not temperature_categorical.temperature_column:
                raise ValueError(
                    "Temperature-categorical sensitivity temperature_column must not be blank."
                )
            if not temperature_categorical.factor_column:
                raise ValueError(
                    "Temperature-categorical sensitivity factor_column must not be blank."
                )
            if not temperature_categorical.model_terms:
                raise ValueError(
                    "Temperature-categorical sensitivity model_terms must not be empty."
                )
        residualized_correlation = ResidualizedCorrelationSensitivityConfig(
            enabled=bool(residualized_raw.get("enabled", False)),
            output_name=str(
                residualized_raw.get(
                    "output_name",
                    "residualized_correlation",
                )
            ).strip(),
            bootstrap_iterations=int(
                residualized_raw.get("bootstrap_iterations", 5000)
            ),
            permutation_iterations=int(
                residualized_raw.get("permutation_iterations", 5000)
            ),
        )
        if residualized_correlation.enabled:
            if not residualized_correlation.output_name:
                raise ValueError(
                    "Residualized-correlation sensitivity output_name must not be blank."
                )
            if residualized_correlation.bootstrap_iterations <= 0:
                raise ValueError(
                    "Residualized-correlation sensitivity bootstrap_iterations must be positive."
                )
            if residualized_correlation.permutation_iterations <= 0:
                raise ValueError(
                    "Residualized-correlation sensitivity permutation_iterations must be positive."
                )
        primary_permutation = PrimaryPermutationSensitivityConfig(
            enabled=bool(primary_permutation_raw.get("enabled", False)),
            output_name=str(
                primary_permutation_raw.get(
                    "output_name",
                    "primary_permutation",
                )
            ).strip(),
            bootstrap_iterations=int(
                primary_permutation_raw.get("bootstrap_iterations", 5000)
            ),
            permutation_iterations=int(
                primary_permutation_raw.get("permutation_iterations", 5000)
            ),
        )
        if primary_permutation.enabled:
            if not primary_permutation.output_name:
                raise ValueError(
                    "Primary-permutation sensitivity output_name must not be blank."
                )
            if primary_permutation.bootstrap_iterations <= 0:
                raise ValueError(
                    "Primary-permutation sensitivity bootstrap_iterations must be positive."
                )
            if primary_permutation.permutation_iterations <= 0:
                raise ValueError(
                    "Primary-permutation sensitivity permutation_iterations must be positive."
                )

        items: list[SourceMethodItemConfig] = []
        items_raw = _require_sequence(
            source_raw.get("items"),
            path="eeg_bold_coupling.sensitivities.source_methods.items",
        )
        for item in items_raw:
            mapping = _require_mapping(
                item,
                path="eeg_bold_coupling.sensitivities.source_methods.items[*]",
            )
            name = str(mapping.get("name", "")).strip()
            method = str(mapping.get("method", "")).strip().lower()
            bands = tuple(
                str(value).strip()
                for value in _require_sequence(
                    mapping.get("bands"),
                    path="eeg_bold_coupling.sensitivities.source_methods.items[*].bands",
                )
                if str(value).strip()
            )
            if not name or not method or not bands:
                raise ValueError(
                    "Each source-method sensitivity item requires name, method, and bands."
                )
            if method not in {"lcmv", "eloreta", "dspm", "wmne"}:
                raise ValueError(
                    "Source-method sensitivity method must be one of "
                    "{'lcmv','eloreta','dspm','wmne'}."
                )
            items.append(SourceMethodItemConfig(name=name, method=method, bands=bands))
        source_methods = SourceMethodSensitivityConfig(
            enabled=bool(source_raw.get("enabled", False)),
            items=tuple(items),
        )
        if source_methods.enabled and not source_methods.items:
            raise ValueError(
                "Source-method sensitivity requires at least one item."
            )
        anatomical_items: list[AnatomicalSpecificityItemConfig] = []
        anatomical_items_raw = _require_sequence(
            anatomical_raw.get("items"),
            path="eeg_bold_coupling.sensitivities.anatomical_specificity.items",
        )
        for item in anatomical_items_raw:
            mapping = _require_mapping(
                item,
                path="eeg_bold_coupling.sensitivities.anatomical_specificity.items[*]",
            )
            name = str(mapping.get("name", "")).strip()
            if not name:
                raise ValueError(
                    "Each anatomical-specificity item requires a non-blank name."
                )
            roi_items_raw = _require_sequence(
                mapping.get("rois"),
                path=(
                    "eeg_bold_coupling.sensitivities.anatomical_specificity.items[*].rois"
                ),
            )
            if not roi_items_raw:
                raise ValueError(
                    f"Anatomical-specificity item {name!r} must define at least one ROI."
                )
            roi_items: list[Mapping[str, Any]] = []
            for roi_item in roi_items_raw:
                roi_mapping = _require_mapping(
                    roi_item,
                    path=(
                        "eeg_bold_coupling.sensitivities.anatomical_specificity.items[*].rois[*]"
                    ),
                )
                roi_items.append(dict(roi_mapping))
            anatomical_items.append(
                AnatomicalSpecificityItemConfig(
                    name=name,
                    rois=tuple(roi_items),
                )
            )
        anatomical_specificity = AnatomicalSpecificitySensitivityConfig(
            enabled=bool(anatomical_raw.get("enabled", False)),
            items=tuple(anatomical_items),
        )
        if anatomical_specificity.enabled and not anatomical_specificity.items:
            raise ValueError(
                "Anatomical-specificity sensitivity requires at least one item."
            )
        within_between = WithinBetweenSensitivityConfig(
            enabled=bool(within_between_raw.get("enabled", False)),
            output_name=str(
                within_between_raw.get("output_name", "within_between")
            ).strip(),
        )
        if within_between.enabled and not within_between.output_name:
            raise ValueError(
                "Within-between sensitivity output_name must not be blank."
            )
        artifact_items: list[ArtifactModelItemConfig] = []
        artifact_items_raw = _require_sequence(
            artifact_raw.get("items"),
            path="eeg_bold_coupling.sensitivities.artifact_models.items",
        )
        for item in artifact_items_raw:
            mapping = _require_mapping(
                item,
                path="eeg_bold_coupling.sensitivities.artifact_models.items[*]",
            )
            name = str(mapping.get("name", "")).strip()
            if not name:
                raise ValueError(
                    "Each artifact-model sensitivity item requires a non-blank name."
                )
            nuisance_overrides = _require_mapping(
                mapping.get("nuisance_overrides", {}),
                path=(
                    "eeg_bold_coupling.sensitivities.artifact_models.items[*]."
                    "nuisance_overrides"
                ),
            )
            if not nuisance_overrides:
                raise ValueError(
                    f"Artifact-model sensitivity {name!r} must define nuisance_overrides."
                )
            artifact_items.append(
                ArtifactModelItemConfig(
                    name=name,
                    nuisance_overrides=dict(nuisance_overrides),
                )
            )
        artifact_models = ArtifactModelSensitivityConfig(
            enabled=bool(artifact_raw.get("enabled", False)),
            items=tuple(artifact_items),
        )
        if artifact_models.enabled and not artifact_models.items:
            raise ValueError(
                "Artifact-model sensitivity requires at least one item."
            )
        return cls(
            painful_only=painful_only,
            alternative_fmri=alternative_fmri,
            delta_temperature=delta_temperature,
            temperature_categorical=temperature_categorical,
            residualized_correlation=residualized_correlation,
            primary_permutation=primary_permutation,
            source_methods=source_methods,
            anatomical_specificity=anatomical_specificity,
            within_between=within_between,
            artifact_models=artifact_models,
        )


def resolve_sensitivity_beta_dir(
    *,
    template: str,
    deriv_root: Path,
    subject: str,
    task: str,
    contrast_name: str,
) -> Path:
    subject_raw = str(subject).replace("sub-", "", 1)
    rendered = template.format(
        deriv_root=str(deriv_root),
        subject=subject_raw,
        subject_bids=f"sub-{subject_raw}",
        task=task,
        contrast_name=contrast_name,
    )
    path = Path(rendered).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Sensitivity beta directory does not exist: {path}"
        )
    return path


def filter_painful_trials(
    *,
    merged_table: pd.DataFrame,
    cfg: PainfulOnlySensitivityConfig,
) -> pd.DataFrame:
    if not cfg.enabled:
        return merged_table.copy()

    mask = pd.Series([False] * len(merged_table), index=merged_table.index)
    if cfg.binary_column is not None and cfg.painful_values:
        column = str(cfg.binary_column)
        if column not in merged_table.columns:
            raise ValueError(
                f"Painful-only binary column {column!r} is missing from merged trial table."
            )
        values = merged_table[column].astype(str).str.strip()
        mask |= values.isin(list(cfg.painful_values))
    if cfg.rating_column is not None and cfg.min_rating is not None:
        column = str(cfg.rating_column)
        if column not in merged_table.columns:
            raise ValueError(
                f"Painful-only rating column {column!r} is missing from merged trial table."
            )
        ratings = pd.to_numeric(merged_table[column], errors="coerce")
        mask |= ratings >= float(cfg.min_rating)

    out = merged_table.loc[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError("Painful-only sensitivity retained zero trials.")
    return out


__all__ = [
    "AlternativeFMRISensitivityConfig",
    "CouplingSensitivityConfig",
    "DeltaTemperatureSensitivityConfig",
    "PainfulOnlySensitivityConfig",
    "SourceMethodItemConfig",
    "SourceMethodSensitivityConfig",
    "TemperatureCategoricalSensitivityConfig",
    "filter_painful_trials",
    "resolve_sensitivity_beta_dir",
]
