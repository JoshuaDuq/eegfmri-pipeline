from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

_ALLOWED_FORMATS: Set[str] = {"png", "svg"}
_ALLOWED_SPACES: Set[str] = {"native", "mni", "both"}
_ALLOWED_THRESHOLD_MODES: Set[str] = {"z", "fdr", "none"}
_ALLOWED_RASTER_MODES: Set[str] = {"weighted", "task", "all"}
_ALLOWED_DESIGN_MATRIX_RUNS: Set[str] = {"none", "first", "all"}


def _normalize_str_list(values: Optional[Iterable[str]]) -> List[str]:
    if values is None:
        return []
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


#: Keys that moved off the plotting config, and where they went.
#:
#: Each of these caused statistics to be computed while living on a config named for
#: rendering: ``space`` including ``mni`` triggered a complete second GLM fit, and
#: ``include_effect_size`` / ``include_standard_error`` drove ``compute_contrast``
#: calls. Deprecated aliases are deliberately not offered -- an alias that accepts a
#: rendering flag which fits a GLM preserves exactly the confusion the split removes.
MOVED_KEYS = {
    "space": "fmri_stats.space",
    "include_effect_size": "fmri_stats.include_effect_size",
    "include_standard_error": "fmri_stats.include_standard_error",
    "include_signatures": "fmri_stats.include_signatures",
    "threshold_mode": "fmri_stats.threshold_mode",
    "z_threshold": "fmri_stats.z_threshold",
    "fdr_q": "fmri_stats.fdr_q",
    "cluster_min_voxels": "fmri_stats.cluster_min_voxels",
    "two_sided": "fmri_stats.two_sided",
}


@dataclass(frozen=True)
class FmriStatsConfig:
    """Settings that cause statistics to be computed.

    Separate from the report config because each of these costs a model fit or a
    contrast computation, and that cost should be visible where it is configured.
    """

    space: str = "native"
    include_effect_size: bool = True
    include_standard_error: bool = True
    include_signatures: bool = True
    threshold_mode: str = "z"
    z_threshold: float = 2.3
    fdr_q: float = 0.05
    cluster_min_voxels: int = 0
    two_sided: bool = True

    def validate(self) -> None:
        boolean_fields = (
            "include_effect_size",
            "include_standard_error",
            "include_signatures",
            "two_sided",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"fmri_stats.{name} must be a YAML boolean")
        if self.space not in _ALLOWED_SPACES:
            raise ValueError(
                f"fmri_stats.space must be one of {sorted(_ALLOWED_SPACES)}, " f"got '{self.space}'"
            )
        if self.threshold_mode not in _ALLOWED_THRESHOLD_MODES:
            raise ValueError(
                "fmri_stats.threshold_mode must be one of "
                f"{sorted(_ALLOWED_THRESHOLD_MODES)}, got '{self.threshold_mode}'"
            )
        if self.threshold_mode == "z" and self.z_threshold <= 0:
            raise ValueError("fmri_stats.z_threshold must be > 0")
        if self.threshold_mode == "fdr" and not 0 < self.fdr_q <= 1:
            raise ValueError("fmri_stats.fdr_q must be in (0, 1]")
        if self.cluster_min_voxels < 0:
            raise ValueError("fmri_stats.cluster_min_voxels must be >= 0")


def stats_config_from_mapping(section: dict) -> FmriStatsConfig:
    """Build the compute-triggering configuration and reject stale keys."""
    known = set(FmriStatsConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"Unknown fmri_stats key(s): {unknown}")
    config = FmriStatsConfig(**section)
    config.validate()
    return config


@dataclass(frozen=True)
class FmriReportConfig:
    """Settings that only decide how existing results are drawn.

    Nothing here may cause a GLM to be fit. That invariant is what lets a report be
    regenerated from a derivatives tree without the model, and it is enforced by a
    test that imports the report path and asserts the fitting modules stay absent.
    """

    enabled: bool = False
    html_report: bool = False
    formats: Sequence[str] = field(default_factory=lambda: ("png",))
    include_unthresholded: bool = True
    include_motion_qc: bool = True
    include_carpet_qc: bool = True
    include_tsnr_qc: bool = True
    include_design_qc: bool = True
    embed_images: bool = True

    #: Which design columns the event raster draws.
    #:
    #: ``weighted`` collapses to one lane for any single-regressor contrast, which is
    #: exactly when a reader most needs to see what the condition is interleaved
    #: against; ``task`` is the default for that reason. ``all`` adds columns with no
    #: onsets, and exists for a design whose task columns the role classifier does not
    #: recognise.
    raster_conditions: str = "task"

    #: How many per-run design matrices to draw.
    #:
    #: Six near-identical 41-column heatmaps cost 1.4 MB of a 9.9 MB report and are
    #: compared through the summary table, the variance-inflation panel and the
    #: correlation matrix -- all of which are already drawn across every run. ``first``
    #: keeps one so the design's shape is still on the page.
    design_matrix_runs: str = "first"

    #: Clusters rendered in the HTML cluster table before it is capped.
    #:
    #: The table used to list every surviving cluster. Measured on sub-0003 once
    #: inference runs on the z map rather than the effect map: 288 rows over 224
    #: clusters at |z| > 2.3, which no reader scans and which stops reading as a
    #: result. The TSV beside the table is always complete, so the cap moves the tail
    #: one click away rather than hiding it. The cohort report caps its own table for
    #: the same reason.
    cluster_table_max_rows: int = 20

    #: Draw the pooled residual standard deviation as its own volume panel.
    #:
    #: Off by default: it is the standard-error map again. Measured on sub-0003 across
    #: 79,540 in-mask voxels, Pearson r = 0.944 and the ratio between them has a
    #: coefficient of variation of 12%, which is what the algebra predicts -- with one
    #: design per run the standard error is the residual SD scaled by a quantity that
    #: does not vary voxelwise. Two 21-tile mosaics for one spatial pattern; the
    #: scalar summary is already a column of the model-fit table.
    include_residual_sd_map: bool = False

    #: Draw the regressor correlation matrix beside the variance-inflation panel.
    #:
    #: Off by default on a confound-expanded design: with motion24 the readable
    #: structure is that ``rot_z`` correlates with ``rot_z_power2``, which is a
    #: property of the expansion rather than of the data, and the actionable summary --
    #: per-regressor inflation with the contrast's own columns separated -- is the
    #: variance-inflation panel directly above it.
    include_regressor_correlation: bool = False

    #: Draw the glass brain beside the thresholded mosaic.
    #:
    #: A maximum-intensity projection catches a cluster that falls between two mosaic
    #: tiles, which is the one thing the mosaic cannot do.
    include_glass_brain: bool = True


    #: Label volume naming the structure each cluster peak falls in, and an optional
    #: index-to-name table for it.
    #:
    #: Supplied by the study rather than downloaded, so a report built from a
    #: derivatives tree needs no network and is pinned to a file the study controls.
    #: Read only for contrasts in MNI space; see
    #: :func:`fmri_pipeline.analysis.report.atlas.atlas_applies_to`.
    atlas_labels_img: Optional[str] = None
    atlas_labels_tsv: Optional[str] = None

    def validate(self) -> None:
        boolean_fields = (
            "enabled",
            "html_report",
            "include_unthresholded",
            "include_motion_qc",
            "include_carpet_qc",
            "include_tsnr_qc",
            "include_design_qc",
            "embed_images",
            "include_glass_brain",
            "include_residual_sd_map",
            "include_regressor_correlation",
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"fmri_report.{name} must be a YAML boolean")
        if type(self.cluster_table_max_rows) is not int:
            raise TypeError("fmri_report.cluster_table_max_rows must be a YAML integer")
        if not 1 <= self.cluster_table_max_rows <= 500:
            raise ValueError(
                "fmri_report.cluster_table_max_rows must be in [1, 500], got "
                f"{self.cluster_table_max_rows}"
            )
        if self.raster_conditions not in _ALLOWED_RASTER_MODES:
            raise ValueError(
                "fmri_report.raster_conditions must be one of "
                f"{sorted(_ALLOWED_RASTER_MODES)}, got '{self.raster_conditions}'"
            )
        if self.design_matrix_runs not in _ALLOWED_DESIGN_MATRIX_RUNS:
            raise ValueError(
                "fmri_report.design_matrix_runs must be one of "
                f"{sorted(_ALLOWED_DESIGN_MATRIX_RUNS)}, got '{self.design_matrix_runs}'"
            )
        if isinstance(self.formats, str) or not isinstance(self.formats, (list, tuple)):
            raise TypeError("fmri_report.formats must be a YAML list")
        if self.atlas_labels_tsv and not self.atlas_labels_img:
            raise ValueError(
                "atlas_labels_tsv names labels for an atlas, but atlas_labels_img is "
                "unset; there is nothing for the table to name."
            )
        formats = _normalize_str_list(self.formats)
        if not formats:
            raise ValueError("plot formats must include at least one of: png, svg")
        unknown_formats = sorted(set(formats) - _ALLOWED_FORMATS)
        if unknown_formats:
            raise ValueError(
                f"Unsupported plot format(s): {unknown_formats}. "
                f"Allowed: {sorted(_ALLOWED_FORMATS)}"
            )


def report_config_from_mapping(section: dict) -> FmriReportConfig:
    """Build a report config from a YAML mapping, rejecting unknown keys.

    Fails rather than silently ignoring a moved key: a study whose YAML still sets
    ``plotting.space`` would otherwise get native-only output with no indication
    that its setting had stopped being read.
    """
    stale = sorted(key for key in MOVED_KEYS if key in section)
    if stale:
        moved = ", ".join(f"'{key}' -> {MOVED_KEYS[key]}" for key in stale)
        raise ValueError(
            f"These plotting keys moved to the fmri_stats section because they cause "
            f"statistics to be computed: {moved}. Update the config; there are "
            f"deliberately no aliases."
        )
    known = set(FmriReportConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"Unknown fmri_report key(s): {unknown}")
    return FmriReportConfig(**section)
