from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

_ALLOWED_FORMATS: Set[str] = {"png", "svg"}
_ALLOWED_SPACES: Set[str] = {"native", "mni", "both"}
_ALLOWED_THRESHOLD_MODES: Set[str] = {"z", "fdr", "none"}


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
        )
        for name in boolean_fields:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"fmri_report.{name} must be a YAML boolean")
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
