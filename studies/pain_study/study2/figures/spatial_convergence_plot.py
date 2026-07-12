"""Publication rendering for Study 2 multimodal spatial convergence."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study2.figures.primary_source_associations_plot import (
    CommonSourceSurfaces,
    HemisphereSurface,
    _validate_surfaces,
)
from studies.pain_study.study2.figures.spatial_convergence import SpatialConvergenceSummary
from studies.pain_study.study2.figures.style import (
    figure_size_inches,
    publication_style,
    study2_diverging_color_map,
)
from studies.pain_study.study2.validation import require_config_int

FIGURE_CONFIG_KEY = "study2.figures.spatial_convergence"
SPATIAL_BANDS = ("alpha", "beta", "gamma")
HEMISPHERES = ("left", "right")
BAND_COLORS = {
    "alpha": "#0072B2",
    "beta": "#CC79A7",
    "gamma": "#009E73",
}
INFERENCE_STATEMENT = (
    "BrainSMASH surrogate maps preserve spatial autocorrelation and carry the "
    "map-correspondence inference."
)


def build_spatial_convergence_figure(
    summary: SpatialConvergenceSummary,
    surfaces: CommonSourceSurfaces,
    config: Any,
) -> Figure:
    """Render fixed cortical-map and spatial-null panels without closing the figure."""

    figure_config = _figure_config(config)
    band_specs = _validate_summary(summary, figure_config)
    _validate_surfaces(surfaces, summary.vertices_manifest)
    fmri_limit, eeg_limit = _display_limits(summary)
    null_limits = _null_limits(summary)
    dimensions = figure_config["dimensions_mm"]
    font_family = str(figure_config["font_family"])
    color_map = study2_diverging_color_map()
    histogram_bins = require_config_int(config, f"{FIGURE_CONFIG_KEY}.null_histogram_bins")
    if histogram_bins < 1:
        raise ValueError("Study 2 spatial null histogram bin count must be positive.")

    with publication_style(font_family):
        figure = plt.figure(figsize=figure_size_inches(dimensions), facecolor="white")
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        row_grid = figure.add_gridspec(
            2,
            1,
            left=0.035,
            right=0.985,
            bottom=0.145,
            top=0.925,
            height_ratios=(1.22, 1.0),
            hspace=0.62,
        )
        _add_surface_row(
            figure,
            row_grid[0, 0],
            summary=summary,
            surfaces=surfaces,
            band_specs=band_specs,
            color_map=color_map,
            fmri_limit=fmri_limit,
            eeg_limit=eeg_limit,
        )
        _add_colorbars(
            figure,
            color_map=color_map,
            fmri_limit=fmri_limit,
            eeg_limit=eeg_limit,
        )
        _add_null_row(
            figure,
            row_grid[1, 0],
            summary=summary,
            band_specs=band_specs,
            histogram_bins=histogram_bins,
            null_limits=null_limits,
        )
        _add_panel_text(figure)
    return figure


def _add_surface_row(
    figure: Figure,
    grid_cell,
    *,
    summary: SpatialConvergenceSummary,
    surfaces: CommonSourceSurfaces,
    band_specs: tuple[Mapping[str, object], ...],
    color_map: LinearSegmentedColormap,
    fmri_limit: float,
    eeg_limit: float,
) -> None:
    surface_grid = grid_cell.subgridspec(1, 4, wspace=0.04)
    map_specs = (
        ("fmri", "NPS-L2 fMRI reference\nforward covariance", summary.fmri_map, fmri_limit),
        *(
            (
                band,
                f"{band_spec['label']}\n{band_spec['frequency_label']}",
                summary.band_results[band].eeg_map,
                eeg_limit,
            )
            for band, band_spec in zip(summary.bands, band_specs, strict=True)
        ),
    )
    for map_index, (map_name, title, values, limit) in enumerate(map_specs):
        map_grid = surface_grid[0, map_index].subgridspec(1, 2, wspace=-0.26)
        for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
            axis = figure.add_subplot(map_grid[0, hemisphere_index], projection="3d")
            axis.set_gid(f"surface-{map_name}-{hemisphere}-lateral")
            displayed_values = _hemisphere_display_values(
                values,
                summary.mask,
                surfaces,
                hemisphere=hemisphere,
            )
            _draw_surface(
                axis,
                surfaces.for_hemisphere(hemisphere),
                values=displayed_values,
                color_map=color_map,
                limit=limit,
            )
            axis.text2D(
                0.5,
                0.02,
                "L" if hemisphere == "left" else "R",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.0,
                color="#4A4A4A",
            )
        figure.text(
            0.035 + (map_index + 0.5) * 0.95 / 4.0,
            0.918,
            title,
            ha="center",
            va="top",
            fontsize=6.3,
            fontweight="bold",
        )


def _draw_surface(
    axis,
    surface: HemisphereSurface,
    *,
    values: np.ndarray,
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    from nilearn import plotting

    coordinates = surface.coordinates.copy()
    faces = surface.faces.copy()
    before_background = len(axis.collections)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"mpl_toolkits")
        plotting.plot_surf(
            surf_mesh=(coordinates, faces),
            bg_map=surface.sulcal_depth,
            hemi=surface.hemisphere,
            view="lateral",
            colorbar=False,
            axes=axis,
            figure=axis.figure,
        )
    surface_collections = axis.collections[before_background:]
    if len(surface_collections) != 1:
        raise RuntimeError("Study 2 surface rendering must create exactly one face collection.")

    valid_face_mask = np.isfinite(values[faces]).all(axis=1)
    valid_faces = faces[valid_face_mask]
    normalization = Normalize(vmin=-limit, vmax=limit, clip=True)
    face_colors = _anatomical_face_colors(surface.sulcal_depth, faces)
    if valid_faces.size:
        face_values = np.mean(values[valid_faces], axis=1)
        face_colors[valid_face_mask] = color_map(normalization(face_values))
    surface_collection = surface_collections[0]
    surface_collection.set_gid("unthresholded-map")
    surface_collection.set_facecolors(face_colors)
    surface_collection.set_edgecolors(face_colors)
    surface_collection.set_cmap(color_map)
    surface_collection.set_norm(normalization)
    surface_collection.set_clim(-limit, limit)
    surface_collection._study2_effect_face_mask = valid_face_mask.copy()
    axis._study2_display_values = values.copy()


def _anatomical_face_colors(sulcal_depth: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_values = np.mean(sulcal_depth[faces], axis=1)
    minimum = float(face_values.min())
    maximum = float(face_values.max())
    if minimum < 0.0 or maximum > 1.0:
        face_values = Normalize(vmin=minimum, vmax=maximum)(face_values)
    return plt.get_cmap("gray_r")(face_values)


def _hemisphere_display_values(
    values: np.ndarray,
    mask: np.ndarray,
    surfaces: CommonSourceSurfaces,
    *,
    hemisphere: str,
) -> np.ndarray:
    n_left = surfaces.left.vertex_numbers.size
    hemisphere_slice = slice(None, n_left) if hemisphere == "left" else slice(n_left, None)
    hemisphere_values = np.asarray(values[hemisphere_slice], dtype=float)
    hemisphere_mask = mask[hemisphere_slice]
    return np.where(hemisphere_mask, hemisphere_values, np.nan)


def _add_colorbars(
    figure: Figure,
    *,
    color_map: LinearSegmentedColormap,
    fmri_limit: float,
    eeg_limit: float,
) -> None:
    _add_colorbar(
        figure,
        bounds=(0.075, 0.545, 0.18, 0.018),
        gid="fmri-colorbar",
        label="NPS-L2 forward covariance (covariance units)",
        color_map=color_map,
        limit=fmri_limit,
    )
    _add_colorbar(
        figure,
        bounds=(0.405, 0.545, 0.43, 0.018),
        gid="eeg-colorbar",
        label="EEG partial correlation, r",
        color_map=color_map,
        limit=eeg_limit,
    )


def _add_colorbar(
    figure: Figure,
    *,
    bounds: tuple[float, float, float, float],
    gid: str,
    label: str,
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    color_axis = figure.add_axes(bounds)
    color_axis.set_gid(gid)
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=color_map),
        cax=color_axis,
        orientation="horizontal",
        ticks=(-limit, 0.0, limit),
    )
    colorbar.set_label(label, labelpad=1.5, fontsize=5.6)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=1.8, width=0.5, pad=1.2, labelsize=5.0)


def _add_null_row(
    figure: Figure,
    grid_cell,
    *,
    summary: SpatialConvergenceSummary,
    band_specs: tuple[Mapping[str, object], ...],
    histogram_bins: int,
    null_limits: tuple[float, float],
) -> None:
    null_grid = grid_cell.subgridspec(1, 3, wspace=0.23)
    for band_index, (band, band_spec) in enumerate(
        zip(summary.bands, band_specs, strict=True)
    ):
        axis = figure.add_subplot(null_grid[0, band_index])
        axis.set_gid(f"null-{band}")
        _draw_null_distribution(
            axis,
            summary=summary,
            band=band,
            band_label=str(band_spec["label"]),
            color=BAND_COLORS[band],
            histogram_bins=histogram_bins,
            null_limits=null_limits,
        )
        if band_index == 0:
            axis.set_ylabel("Density")
        axis.set_xlabel("Spatial correlation, r")


def _draw_null_distribution(
    axis: Axes,
    *,
    summary: SpatialConvergenceSummary,
    band: str,
    band_label: str,
    color: str,
    histogram_bins: int,
    null_limits: tuple[float, float],
) -> None:
    result = summary.band_results[band]
    heights, _edges, patches = axis.hist(
        result.surrogate_r,
        bins=histogram_bins,
        range=null_limits,
        density=True,
        color=color,
        alpha=0.34,
        edgecolor="white",
        linewidth=0.25,
    )
    for patch in patches:
        patch.set_gid("null-histogram")
    axis.axvline(0.0, color="#555555", linewidth=0.65, linestyle="--", gid="zero-reference")
    observed_height = float(heights.max()) * 1.08
    axis.plot(
        result.spatial_r,
        observed_height,
        marker="D",
        markersize=5.0,
        markerfacecolor=color,
        markeredgecolor="#1A1A1A" if result.holm_significant else "white",
        markeredgewidth=1.1 if result.holm_significant else 0.65,
        linestyle="none",
        gid="observed-correlation",
        zorder=4,
    )
    axis.set_xlim(*null_limits)
    axis.set_ylim(0.0, observed_height * 1.18)
    axis.set_title(
        f"r = {result.spatial_r:.3f}   plus-one two-sided p = {result.p_value:.4f}   "
        f"Holm p = {result.holm_adjusted_p_value:.4f}",
        fontsize=5.1,
        pad=3.0,
    )
    axis.text(
        0.02,
        0.94,
        band_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=color,
        fontsize=6.2,
        fontweight="bold",
    )
    if result.holm_significant:
        outline = Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=axis.transAxes,
            fill=False,
            edgecolor="#1A1A1A",
            linewidth=1.0,
            clip_on=False,
        )
        outline.set_gid("holm-significant-outline")
        axis.add_patch(outline)


def _add_panel_text(figure: Figure) -> None:
    figure.text(0.012, 0.965, "a", ha="left", va="top", fontsize=8.0, fontweight="bold")
    figure.text(0.012, 0.475, "b", ha="left", va="top", fontsize=8.0, fontweight="bold")
    figure.text(
        0.5,
        0.025,
        INFERENCE_STATEMENT,
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#333333",
    )


def _display_limits(summary: SpatialConvergenceSummary) -> tuple[float, float]:
    _validate_mask(summary)
    fmri_values = _masked_nonzero_values(
        summary.fmri_map,
        summary.mask,
        label="fMRI map",
        n_vertices=summary.n_vertices,
    )
    eeg_values = [
        _masked_nonzero_values(
            summary.band_results[band].eeg_map,
            summary.mask,
            label=f"{band} EEG map",
            n_vertices=summary.n_vertices,
        )
        for band in summary.bands
    ]
    fmri_limit = float(np.max(np.abs(fmri_values)))
    eeg_limit = float(max(np.max(np.abs(values)) for values in eeg_values))
    return fmri_limit, eeg_limit


def _masked_nonzero_values(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    n_vertices: int,
) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError(f"Study 2 spatial masked {label} must be real-valued.")
    if array.ndim != 1 or array.size != n_vertices:
        raise ValueError(f"Study 2 spatial {label} does not match the source vertex manifest.")
    masked_values = array[mask]
    if not np.isfinite(masked_values).all() or not np.any(masked_values != 0.0):
        raise ValueError(
            f"Study 2 spatial masked {label} must contain a finite nonzero effect."
        )
    return masked_values


def _validate_mask(summary: SpatialConvergenceSummary) -> None:
    if summary.mask.dtype != np.bool_:
        raise ValueError("Study 2 spatial figure mask must have boolean dtype.")
    if summary.mask.ndim != 1 or summary.mask.size != summary.n_vertices:
        raise ValueError("Study 2 spatial figure mask does not match the source vertex manifest.")
    if int(summary.mask.sum()) < 2:
        raise ValueError("Study 2 spatial figure mask must contain at least two vertices.")


def _null_limits(summary: SpatialConvergenceSummary) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.append(summary.band_results[band].surrogate_r, summary.band_results[band].spatial_r)
            for band in summary.bands
        ]
        + [np.asarray([0.0])]
    )
    if not np.isfinite(values).all():
        raise ValueError("Study 2 spatial null correlations must be finite.")
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        raise ValueError("Study 2 spatial null correlations must span a nonzero range.")
    padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


def _validate_summary(
    summary: SpatialConvergenceSummary,
    figure_config: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw_band_specs = figure_config["bands"]
    if not isinstance(raw_band_specs, list) or not all(
        isinstance(spec, Mapping) for spec in raw_band_specs
    ):
        raise ValueError("Study 2 spatial-convergence figure bands must be a list of mappings.")
    band_specs = tuple(raw_band_specs)
    configured_bands = tuple(str(spec["name"]) for spec in band_specs)
    if configured_bands != summary.bands:
        raise ValueError(
            "Study 2 spatial-convergence configured band order must exactly match the summary."
        )
    if summary.bands != SPATIAL_BANDS:
        raise ValueError("Study 2 spatial-convergence figure requires alpha, beta, and gamma.")
    return band_specs


def _figure_config(config: Any) -> Mapping[str, object]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


__all__ = ["build_spatial_convergence_figure"]
