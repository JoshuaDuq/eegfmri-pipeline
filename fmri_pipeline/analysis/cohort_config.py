"""Strict YAML configuration for second-level inference reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_ALLOWED_FORMATS = frozenset({"png", "svg"})
_ALLOWED_HEIGHT_CONTROLS = frozenset({"fdr", "fpr", "bonferroni", "none"})


@dataclass(frozen=True)
class CohortReportConfig:
    """Rendering-only options for the second-level cohort report."""

    enabled: bool = True
    html_report: bool = True
    formats: Sequence[str] = field(default_factory=lambda: ("png",))
    embed_images: bool = True
    include_design_correlation: bool = True
    include_input_map_concordance: bool = True
    #: Refit the model without each participant and rethreshold. Costs one extra
    #: Nilearn fit per participant and no permutations, so it scales with cohort size.
    include_leave_one_out_influence: bool = True
    #: Residuals of the reported design, against the Gaussian errors the report's
    #: assumption note already claims.
    include_residual_diagnostics: bool = True
    #: Nilearn's interactive volume viewer, inlined. Self-contained and offline, and
    #: it lets a reader reach any cluster instead of the few a fixed mosaic cuts
    #: through; costs roughly 0.6 MB of HTML.
    include_interactive_viewer: bool = True
    include_true_discovery_proportion: bool = True
    include_unthresholded: bool = True
    cluster_table_max_rows: int = 10
    #: Label volume naming the structure each cluster peak falls in, plus an optional
    #: index-to-name table. Same convention as ``fmri_report``: supplied by the study
    #: rather than downloaded, so the report needs no network and stays pinned to a file
    #: the study controls. Second-level coordinates are MNI by construction, so unlike
    #: the subject report there is no per-contrast space gate to clear.
    atlas_labels_img: str | None = None
    atlas_labels_tsv: str | None = None
    #: Cortical mesh for the surface panel, e.g. "fsaverage" or "fsaverage5".
    #: Off by default: Nilearn fetches a mesh it does not find, and a report that
    #: reads a derivatives tree offline must not acquire a download as a side effect
    #: of drawing a figure. Naming one here is the study saying it has the data.
    surface_mesh: str | None = None

    def normalized(self) -> "CohortReportConfig":
        self.validate()
        return CohortReportConfig(
            enabled=self.enabled,
            html_report=self.html_report,
            formats=tuple(str(value).strip().lower() for value in self.formats),
            embed_images=self.embed_images,
            include_design_correlation=self.include_design_correlation,
            include_input_map_concordance=self.include_input_map_concordance,
            include_leave_one_out_influence=self.include_leave_one_out_influence,
            include_residual_diagnostics=self.include_residual_diagnostics,
            include_interactive_viewer=self.include_interactive_viewer,
            include_true_discovery_proportion=self.include_true_discovery_proportion,
            include_unthresholded=self.include_unthresholded,
            cluster_table_max_rows=self.cluster_table_max_rows,
            atlas_labels_img=self.atlas_labels_img,
            atlas_labels_tsv=self.atlas_labels_tsv,
            surface_mesh=self.surface_mesh,
        )

    def validate(self) -> None:
        for name in (
            "enabled",
            "html_report",
            "embed_images",
            "include_design_correlation",
            "include_input_map_concordance",
            "include_leave_one_out_influence",
            "include_residual_diagnostics",
            "include_interactive_viewer",
            "include_true_discovery_proportion",
            "include_unthresholded",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"fmri_group_level.report.{name} must be a YAML boolean")
        if isinstance(self.formats, str) or not isinstance(self.formats, (list, tuple)):
            raise TypeError("fmri_group_level.report.formats must be a YAML list")
        formats = tuple(str(value).strip().lower() for value in self.formats)
        if not formats or any(value not in _ALLOWED_FORMATS for value in formats):
            raise ValueError("fmri_group_level.report.formats must contain png and/or svg")
        if type(self.cluster_table_max_rows) is not int:
            raise TypeError("fmri_group_level.report.cluster_table_max_rows must be a YAML integer")
        if not 1 <= self.cluster_table_max_rows <= 50:
            raise ValueError("fmri_group_level.report.cluster_table_max_rows must be in [1, 50]")
        if self.surface_mesh is not None and type(self.surface_mesh) is not str:
            raise TypeError("fmri_group_level.report.surface_mesh must be a YAML string")
        if self.atlas_labels_tsv and not self.atlas_labels_img:
            raise ValueError(
                "fmri_group_level.report.atlas_labels_tsv names labels for an atlas, but "
                "atlas_labels_img is unset"
            )


@dataclass(frozen=True)
class CohortThresholdConfig:
    """Predeclared inferential display for the cohort statistic map."""

    height_control: str = "fdr"
    alpha: float = 0.05
    uncorrected_z_threshold: float = 3.09
    cluster_min_voxels: int = 0
    two_sided: bool = True
    min_distance_mm: float = 8.0

    def validate(self) -> None:
        if (
            type(self.height_control) is not str
            or self.height_control not in _ALLOWED_HEIGHT_CONTROLS
        ):
            raise ValueError(
                "fmri_group_level.threshold.height_control must be one of "
                f"{sorted(_ALLOWED_HEIGHT_CONTROLS)}"
            )
        if type(self.alpha) not in {int, float}:
            raise TypeError("fmri_group_level.threshold.alpha must be a YAML number")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("fmri_group_level.threshold.alpha must be in (0, 1)")
        if type(self.uncorrected_z_threshold) not in {int, float}:
            raise TypeError(
                "fmri_group_level.threshold.uncorrected_z_threshold must be a YAML number"
            )
        if not math.isfinite(self.uncorrected_z_threshold) or self.uncorrected_z_threshold <= 0:
            raise ValueError("fmri_group_level.threshold.uncorrected_z_threshold must be > 0")
        if type(self.cluster_min_voxels) is not int or self.cluster_min_voxels < 0:
            raise ValueError(
                "fmri_group_level.threshold.cluster_min_voxels must be an integer >= 0"
            )
        if type(self.two_sided) is not bool:
            raise TypeError("fmri_group_level.threshold.two_sided must be a YAML boolean")
        if type(self.min_distance_mm) not in {int, float}:
            raise TypeError("fmri_group_level.threshold.min_distance_mm must be a YAML number")
        if not math.isfinite(self.min_distance_mm) or self.min_distance_mm <= 0:
            raise ValueError("fmri_group_level.threshold.min_distance_mm must be finite and > 0")


def cohort_report_config_from_mapping(section: Mapping[str, Any]) -> CohortReportConfig:
    """Build report rendering config and reject misspelled YAML keys."""
    if not isinstance(section, Mapping):
        raise TypeError("fmri_group_level.report must be a YAML mapping")
    known = set(CohortReportConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"Unknown fmri_group_level.report key(s): {unknown}")
    return CohortReportConfig(**dict(section)).normalized()


def cohort_threshold_config_from_mapping(
    section: Mapping[str, Any],
) -> CohortThresholdConfig:
    """Build group threshold config and reject misspelled YAML keys."""
    if not isinstance(section, Mapping):
        raise TypeError("fmri_group_level.threshold must be a YAML mapping")
    known = set(CohortThresholdConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"Unknown fmri_group_level.threshold key(s): {unknown}")
    config = CohortThresholdConfig(**dict(section))
    config.validate()
    return config


__all__ = [
    "CohortReportConfig",
    "CohortThresholdConfig",
    "cohort_report_config_from_mapping",
    "cohort_threshold_config_from_mapping",
]
