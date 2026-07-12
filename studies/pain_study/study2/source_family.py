"""Family-level source inference across Study 2 bands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from studies.pain_study.study2.source_inference import (
    GroupSourceInferenceResult,
    compute_group_source_inference,
)
from studies.pain_study.study2.statistics import holm_adjusted_p_values
from studies.pain_study.study2.validation import finite_number


@dataclass(frozen=True)
class SourceFamilyBandResult:
    band: str
    inference: GroupSourceInferenceResult
    min_cluster_p_value: float
    holm_adjusted_p_value: float
    significant: bool


@dataclass(frozen=True)
class SourceFamilyInferenceResult:
    bands: tuple[str, ...]
    band_results: dict[str, SourceFamilyBandResult]
    alpha: float


def compute_source_family_inference(
    *,
    observed_maps_by_band: Mapping[str, np.ndarray],
    null_maps_by_band: Mapping[str, np.ndarray],
    adjacency: np.ndarray,
    cluster_forming_p: float,
    alpha: float = 0.05,
) -> SourceFamilyInferenceResult:
    """Run one-band cluster tests and Holm-correct the band-level p-values."""
    observed_maps = _band_mapping(observed_maps_by_band, name="observed maps")
    null_maps = _band_mapping(null_maps_by_band, name="null maps")
    _require_matching_band_keys(observed_maps, null_maps)
    alpha_value = _probability(alpha, name="source family alpha")

    p_values: dict[str, float] = {}
    inferences: dict[str, GroupSourceInferenceResult] = {}
    for band, observed in observed_maps.items():
        inference = compute_group_source_inference(
            observed_maps=observed,
            null_maps=null_maps[band],
            adjacency=adjacency,
            cluster_forming_p=cluster_forming_p,
        )
        inferences[band] = inference
        p_values[band] = _min_cluster_p_value(inference)

    adjusted_p_values = holm_adjusted_p_values(p_values)
    band_results = {
        band: SourceFamilyBandResult(
            band=band,
            inference=inferences[band],
            min_cluster_p_value=p_values[band],
            holm_adjusted_p_value=adjusted_p_values[band],
            significant=adjusted_p_values[band] <= alpha_value,
        )
        for band in observed_maps
    }
    return SourceFamilyInferenceResult(
        bands=tuple(observed_maps),
        band_results=band_results,
        alpha=alpha_value,
    )


def summarize_source_family(result: SourceFamilyInferenceResult) -> pd.DataFrame:
    """Flatten a source-family result into one row per band for reporting."""
    records: list[dict[str, object]] = []
    for band in result.bands:
        band_result = result.band_results[band]
        inference = band_result.inference
        records.append(
            {
                "band": band,
                "n_subjects": inference.n_subjects,
                "n_permutations": inference.n_permutations,
                "n_clusters": len(inference.clusters),
                "min_cluster_p_value": band_result.min_cluster_p_value,
                "holm_adjusted_p_value": band_result.holm_adjusted_p_value,
                "significant": band_result.significant,
            }
        )
    return pd.DataFrame.from_records(records)


def _band_mapping(
    values: Mapping[str, np.ndarray],
    *,
    name: str,
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"Study 2 {name} must be a mapping.")
    if not values:
        raise ValueError(f"Study 2 {name} must not be empty.")

    normalized: dict[str, np.ndarray] = {}
    for band, maps in values.items():
        band_name = _normalized_band(band)
        if band_name in normalized:
            raise ValueError(f"Study 2 {name} contains duplicate band '{band_name}'.")
        normalized[band_name] = np.asarray(maps, dtype=float)
    return normalized


def _require_matching_band_keys(
    observed_maps: Mapping[str, np.ndarray],
    null_maps: Mapping[str, np.ndarray],
) -> None:
    if set(observed_maps) != set(null_maps):
        raise ValueError(
            "Study 2 source family inference requires matching band keys for "
            "observed and null maps."
        )


def _min_cluster_p_value(inference: GroupSourceInferenceResult) -> float:
    if not inference.clusters:
        return 1.0
    return float(min(cluster.p_value for cluster in inference.clusters))


def _probability(value: object, *, name: str) -> float:
    probability = finite_number(value, name)
    if probability <= 0.0 or probability > 1.0:
        raise ValueError(f"Study 2 {name} must be in (0, 1].")
    return probability


def _normalized_band(band: object) -> str:
    band_name = str(band).strip().lower()
    if not band_name:
        raise ValueError("Study 2 source family band names must be non-empty.")
    return band_name


__all__ = [
    "SourceFamilyBandResult",
    "SourceFamilyInferenceResult",
    "compute_source_family_inference",
    "summarize_source_family",
]
