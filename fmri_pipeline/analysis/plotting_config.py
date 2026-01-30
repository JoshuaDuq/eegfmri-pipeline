from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set


_ALLOWED_FORMATS: Set[str] = {"png", "svg"}
_ALLOWED_SPACES: Set[str] = {"native", "mni", "both"}
_ALLOWED_PLOT_TYPES: Set[str] = {"slices", "glass", "hist", "clusters"}
_ALLOWED_THRESHOLD_MODES: Set[str] = {"z", "fdr", "none"}
_ALLOWED_VMAX_MODES: Set[str] = {"per_space_robust", "shared_robust", "manual"}


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


@dataclass(frozen=True)
class FmriPlottingConfig:
    enabled: bool = False
    html_report: bool = False
    formats: Sequence[str] = field(default_factory=lambda: ("png",))
    space: str = "both"  # native|mni|both
    # Thresholding (applies to thresholded overlays / cluster table).
    threshold_mode: str = "z"  # z|fdr|none
    z_threshold: float = 2.3
    fdr_q: float = 0.05
    cluster_min_voxels: int = 0
    two_sided: bool = True
    # Scaling
    vmax_mode: str = "per_space_robust"  # per_space_robust|shared_robust|manual
    vmax_manual: Optional[float] = None
    # Content toggles
    include_unthresholded: bool = True
    plot_types: Sequence[str] = field(default_factory=lambda: ("slices", "glass", "hist", "clusters"))
    include_effect_size: bool = True
    include_standard_error: bool = True
    include_motion_qc: bool = True
    include_carpet_qc: bool = True
    include_tsnr_qc: bool = True
    include_design_qc: bool = True
    embed_images: bool = True
    include_signatures: bool = True

    def normalized(self) -> "FmriPlottingConfig":
        formats = _normalize_str_list(self.formats)
        plot_types = _normalize_str_list(self.plot_types)
        space = (self.space or "").strip().lower() or "both"
        threshold_mode = (self.threshold_mode or "").strip().lower() or "z"
        vmax_mode = (self.vmax_mode or "").strip().lower() or "per_space_robust"
        return FmriPlottingConfig(
            enabled=bool(self.enabled),
            html_report=bool(self.html_report),
            formats=tuple(formats) if formats else (),
            space=space,
            threshold_mode=threshold_mode,
            z_threshold=float(self.z_threshold),
            fdr_q=float(self.fdr_q),
            cluster_min_voxels=int(self.cluster_min_voxels),
            two_sided=bool(self.two_sided),
            vmax_mode=vmax_mode,
            vmax_manual=float(self.vmax_manual) if self.vmax_manual is not None else None,
            include_unthresholded=bool(self.include_unthresholded),
            plot_types=tuple(plot_types) if plot_types else (),
            include_effect_size=bool(self.include_effect_size),
            include_standard_error=bool(self.include_standard_error),
            include_motion_qc=bool(self.include_motion_qc),
            include_carpet_qc=bool(self.include_carpet_qc),
            include_tsnr_qc=bool(self.include_tsnr_qc),
            include_design_qc=bool(self.include_design_qc),
            embed_images=bool(self.embed_images),
            include_signatures=bool(self.include_signatures),
        )

    def validate(self) -> None:
        cfg = self.normalized()

        if not cfg.enabled:
            return

        if cfg.space not in _ALLOWED_SPACES:
            raise ValueError(f"plot space must be one of {sorted(_ALLOWED_SPACES)}, got '{cfg.space}'")

        if cfg.threshold_mode not in _ALLOWED_THRESHOLD_MODES:
            raise ValueError(
                f"threshold mode must be one of {sorted(_ALLOWED_THRESHOLD_MODES)}, got '{cfg.threshold_mode}'"
            )

        if not cfg.formats:
            raise ValueError("plot formats must include at least one of: png, svg")
        unknown_formats = sorted(set(cfg.formats) - _ALLOWED_FORMATS)
        if unknown_formats:
            raise ValueError(f"Unsupported plot format(s): {unknown_formats}. Allowed: {sorted(_ALLOWED_FORMATS)}")

        if cfg.threshold_mode == "z" and cfg.z_threshold <= 0:
            raise ValueError("plot z-threshold must be > 0")
        if cfg.threshold_mode == "fdr" and not (0 < cfg.fdr_q <= 1):
            raise ValueError("plot FDR q must be in (0, 1]")
        if cfg.cluster_min_voxels < 0:
            raise ValueError("cluster_min_voxels must be >= 0")

        if cfg.vmax_mode not in _ALLOWED_VMAX_MODES:
            raise ValueError(f"vmax_mode must be one of {sorted(_ALLOWED_VMAX_MODES)}, got '{cfg.vmax_mode}'")
        if cfg.vmax_mode == "manual" and (cfg.vmax_manual is None or cfg.vmax_manual <= 0):
            raise ValueError("vmax_manual must be provided and > 0 when vmax_mode=manual")

        if not cfg.plot_types:
            raise ValueError(f"plot types must include at least one of: {sorted(_ALLOWED_PLOT_TYPES)}")
        unknown_types = sorted(set(cfg.plot_types) - _ALLOWED_PLOT_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported plot type(s): {unknown_types}. Allowed: {sorted(_ALLOWED_PLOT_TYPES)}")


def build_fmri_plotting_config_from_args(
    *,
    enabled: Optional[bool] = None,
    html_report: Optional[bool] = None,
    formats: Optional[Sequence[str]] = None,
    space: Optional[str] = None,
    threshold_mode: Optional[str] = None,
    z_threshold: Optional[float] = None,
    fdr_q: Optional[float] = None,
    cluster_min_voxels: Optional[int] = None,
    two_sided: Optional[bool] = None,
    vmax_mode: Optional[str] = None,
    vmax_manual: Optional[float] = None,
    include_unthresholded: Optional[bool] = None,
    plot_types: Optional[Sequence[str]] = None,
    include_effect_size: Optional[bool] = None,
    include_standard_error: Optional[bool] = None,
    include_motion_qc: Optional[bool] = None,
    include_carpet_qc: Optional[bool] = None,
    include_tsnr_qc: Optional[bool] = None,
    include_design_qc: Optional[bool] = None,
    embed_images: Optional[bool] = None,
    include_signatures: Optional[bool] = None,
) -> FmriPlottingConfig:
    cfg = FmriPlottingConfig(
        enabled=bool(enabled) if enabled is not None else False,
        html_report=bool(html_report) if html_report is not None else False,
        formats=tuple(formats) if formats is not None else ("png",),
        space=str(space) if space is not None else "both",
        threshold_mode=str(threshold_mode) if threshold_mode is not None else "z",
        z_threshold=float(z_threshold) if z_threshold is not None else 2.3,
        fdr_q=float(fdr_q) if fdr_q is not None else 0.05,
        cluster_min_voxels=int(cluster_min_voxels) if cluster_min_voxels is not None else 0,
        two_sided=bool(two_sided) if two_sided is not None else True,
        vmax_mode=str(vmax_mode) if vmax_mode is not None else "per_space_robust",
        vmax_manual=float(vmax_manual) if vmax_manual is not None else None,
        include_unthresholded=bool(include_unthresholded) if include_unthresholded is not None else True,
        plot_types=tuple(plot_types) if plot_types is not None else ("slices", "glass", "hist", "clusters"),
        include_effect_size=bool(include_effect_size) if include_effect_size is not None else True,
        include_standard_error=bool(include_standard_error) if include_standard_error is not None else True,
        include_motion_qc=bool(include_motion_qc) if include_motion_qc is not None else True,
        include_carpet_qc=bool(include_carpet_qc) if include_carpet_qc is not None else True,
        include_tsnr_qc=bool(include_tsnr_qc) if include_tsnr_qc is not None else True,
        include_design_qc=bool(include_design_qc) if include_design_qc is not None else True,
        embed_images=bool(embed_images) if embed_images is not None else True,
        include_signatures=bool(include_signatures) if include_signatures is not None else True,
    ).normalized()
    cfg.validate()
    return cfg
