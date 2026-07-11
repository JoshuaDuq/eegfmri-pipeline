"""Publication rendering for Study 1 whole-brain fMRI construct validity."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.fmri_construct_validity import (
    ESTIMANDS,
    FIGURE_CONFIG_KEY,
    FmriConstructValiditySummary,
)
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

PANEL_SPECS = {
    "temperature": {
        "label": "a  Delivered temperature",
        "unit": "Mean BOLD signal change (% per 1 °C)",
    },
    "rating": {
        "label": "b  Subjective intensity beyond temperature",
        "unit": "Mean BOLD signal change (% per 10 rating points)",
    },
}
SURFACE_VIEWS = (
    ("left", "lateral"),
    ("right", "lateral"),
    ("left", "medial"),
    ("right", "medial"),
)


def build_fmri_construct_validity_figure(
    summary: FmriConstructValiditySummary,
    config: Any,
) -> Figure:
    """Render fixed cortical and axial views for both whole-brain estimands."""

    from nilearn import datasets

    figure_config = _figure_config(config)
    _validate_summary(summary, figure_config)
    dimensions = figure_config["dimensions_mm"]
    slices = tuple(float(value) for value in figure_config["axial_slices_mm"])
    percentile = float(figure_config["display"]["robust_percentile"])
    mesh = datasets.load_fsaverage(str(figure_config["surface_mesh"]))
    sulcal = datasets.load_fsaverage_data(
        str(figure_config["surface_mesh"]),
        mesh_type="inflated",
        data_type="sulcal",
    )
    color_map = _diverging_color_map()

    with publication_style(config):
        figure = plt.figure(figsize=figure_size_inches(dimensions), facecolor="white")
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            1,
            2,
            left=0.025,
            right=0.985,
            bottom=0.245,
            top=0.84,
            wspace=0.075,
        )
        for panel_index, estimand in enumerate(ESTIMANDS):
            group = summary.group_maps[estimand]
            limit = symmetric_display_limit(group.mean_effect, percentile=percentile)
            panel_grid = grid[0, panel_index].subgridspec(
                3,
                2,
                height_ratios=(0.78, 0.78, 1.04),
                hspace=-0.03,
                wspace=-0.02,
            )
            _draw_surface_views(
                figure,
                panel_grid,
                effect_image=group.mean_effect,
                significance_mask=group.significance_mask,
                mesh=mesh,
                sulcal=sulcal,
                color_map=color_map,
                limit=limit,
            )
            volume_axis = figure.add_subplot(panel_grid[2, :])
            _draw_axial_views(
                volume_axis,
                effect_image=group.mean_effect,
                significance_mask=group.significance_mask,
                slices=slices,
                color_map=color_map,
                limit=limit,
            )
            title_x = 0.025 + panel_index * 0.52
            figure.text(
                title_x,
                0.935,
                str(PANEL_SPECS[estimand]["label"]),
                ha="left",
                va="top",
                fontsize=7.5,
                fontweight="bold",
            )
            _add_colorbar(
                figure,
                panel_index=panel_index,
                color_map=color_map,
                limit=limit,
                label=str(PANEL_SPECS[estimand]["unit"]),
            )

        readiness = (
            "Article-ready cohort"
            if summary.article_ready
            else "Preliminary cohort; spatial estimates are descriptive"
        )
        figure.text(
            0.5,
            0.985,
            f"Whole-brain fMRI construct validity · {readiness} (n={summary.n_subjects})",
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )
        slice_text = ", ".join(_format_coordinate(value) for value in slices)
        figure.text(
            0.5,
            0.185,
            f"Axial sections: z = {slice_text} mm",
            ha="center",
            va="center",
            fontsize=5.5,
            color="#333333",
        )
        alpha = float(figure_config["inference"]["alpha"])
        figure.text(
            0.5,
            0.008,
            (
                "Unthresholded participant-mean effects; dark outlines indicate "
                f"two-sided voxelwise max-T FWE p < {alpha:.2f}."
            ),
            ha="center",
            va="bottom",
            fontsize=5.3,
            color="#333333",
        )
    return figure


def symmetric_display_limit(image: nib.Nifti1Image, *, percentile: float) -> float:
    """Return a finite robust absolute display limit centered at zero."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("Display percentile must be within (0, 100].")
    values = np.asanyarray(image.dataobj, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    nonzero = finite[finite > 0]
    if nonzero.size == 0:
        raise ValueError("fMRI effect map contains no non-zero finite values.")
    limit = float(np.percentile(nonzero, percentile))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("fMRI effect-map display limit must be positive and finite.")
    return limit


def _draw_surface_views(
    figure: Figure,
    panel_grid,
    *,
    effect_image: nib.Nifti1Image,
    significance_mask: nib.Nifti1Image,
    mesh,
    sulcal,
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    from nilearn import plotting, surface

    for view_index, (hemisphere, view) in enumerate(SURFACE_VIEWS):
        axis = figure.add_subplot(
            panel_grid[view_index // 2, view_index % 2],
            projection="3d",
        )
        pial = mesh.pial.parts[hemisphere]
        white = mesh.white_matter.parts[hemisphere]
        inflated = mesh.inflated.parts[hemisphere]
        effect_texture = surface.vol_to_surf(
            effect_image,
            pial,
            inner_mesh=white,
            interpolation="linear",
        )
        significance_texture = surface.vol_to_surf(
            significance_mask,
            pial,
            inner_mesh=white,
            interpolation="nearest_most_frequent",
        )
        if not np.isfinite(effect_texture).all() or not np.isfinite(significance_texture).all():
            raise ValueError(f"Surface projection produced non-finite {hemisphere} values.")
        significant_vertices = (significance_texture >= 0.5).astype(np.int8)
        with warnings.catch_warnings():
            for message in (
                "divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul",
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=message,
                    category=RuntimeWarning,
                    module=r"mpl_toolkits\.mplot3d\.art3d",
                )
            plotting.plot_surf_stat_map(
                surf_mesh=inflated,
                stat_map=effect_texture,
                bg_map=sulcal.data.parts[hemisphere],
                hemi=hemisphere,
                view=view,
                cmap=color_map,
                colorbar=False,
                threshold=None,
                vmin=-limit,
                vmax=limit,
                symmetric_cbar=True,
                bg_on_data=True,
                axes=axis,
                figure=figure,
            )
            if significant_vertices.any():
                plotting.plot_surf_contours(
                    surf_mesh=inflated,
                    roi_map=significant_vertices,
                    levels=[1],
                    colors=["#202020"],
                    legend=False,
                    axes=axis,
                    figure=figure,
                )
        axis.set_title(
            f"{hemisphere[0].upper()}{hemisphere[1:]} {view}",
            fontsize=5.3,
            pad=-2.0,
        )


def _draw_axial_views(
    axis,
    *,
    effect_image: nib.Nifti1Image,
    significance_mask: nib.Nifti1Image,
    slices: tuple[float, ...],
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    from nilearn import datasets, plotting

    template = datasets.load_mni152_template(resolution=2)
    display = plotting.plot_stat_map(
        effect_image,
        bg_img=template,
        display_mode="z",
        cut_coords=slices,
        cmap=color_map,
        colorbar=False,
        symmetric_cbar=True,
        threshold=None,
        vmin=-limit,
        vmax=limit,
        annotate=False,
        draw_cross=False,
        black_bg=False,
        axes=axis,
    )
    axis.text(
        0.005,
        0.96,
        "L",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        fontweight="bold",
    )
    axis.text(
        0.995,
        0.96,
        "R",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        fontweight="bold",
    )
    if np.any(np.asanyarray(significance_mask.dataobj) > 0):
        display.add_contours(
            significance_mask,
            levels=[0.5],
            colors=["#202020"],
            linewidths=0.45,
        )


def _add_colorbar(
    figure: Figure,
    *,
    panel_index: int,
    color_map: LinearSegmentedColormap,
    limit: float,
    label: str,
) -> None:
    left = 0.075 + 0.505 * panel_index
    axis = figure.add_axes([left, 0.112, 0.37, 0.018])
    colorbar = mpl.colorbar.ColorbarBase(
        axis,
        cmap=color_map,
        norm=Normalize(vmin=-limit, vmax=limit),
        orientation="horizontal",
        ticks=(-limit, 0.0, limit),
    )
    colorbar.ax.set_xticklabels((_format_tick(-limit), "0", _format_tick(limit)))
    colorbar.ax.tick_params(length=2.0, pad=1.0, labelsize=5.2)
    colorbar.set_label(label, fontsize=5.6, labelpad=1.0)
    colorbar.outline.set_linewidth(0.4)


def _diverging_color_map() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "study1_fmri_diverging",
        ("#0072B2", "#F7F7F7", "#D55E00"),
        N=256,
    )


def _format_coordinate(value: float) -> str:
    number = f"{value:g}"
    return number.replace("-", "−")


def _format_tick(value: float) -> str:
    return f"{value:.2g}".replace("-", "−")


def _validate_summary(
    summary: FmriConstructValiditySummary,
    figure_config: Mapping[str, object],
) -> None:
    if set(summary.group_maps) != set(ESTIMANDS):
        raise ValueError("fMRI construct-validity figure requires both configured estimands.")
    if summary.n_subjects < 2 or len(summary.subjects) != summary.n_subjects:
        raise ValueError("fMRI construct-validity participant count is inconsistent.")
    if str(figure_config["surface_mesh"]) != "fsaverage5":
        raise ValueError("fMRI construct-validity rendering requires fsaverage5.")
    slices = tuple(float(value) for value in figure_config["axial_slices_mm"])
    if slices != (-12.0, 0.0, 12.0, 24.0, 36.0, 48.0):
        raise ValueError("fMRI construct-validity axial slices must match the fixed design.")
    for estimand in ESTIMANDS:
        group = summary.group_maps[estimand]
        if group.estimand != estimand or group.n_subjects != summary.n_subjects:
            raise ValueError(f"The {estimand} group map has inconsistent metadata.")
        if group.mean_effect.shape != group.significance_mask.shape:
            raise ValueError(f"The {estimand} effect and significance maps must share a grid.")
        if not np.allclose(
            group.mean_effect.affine,
            group.significance_mask.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"The {estimand} effect and significance maps must share an affine.")


def _figure_config(config: Any) -> Mapping[str, Any]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


__all__ = [
    "build_fmri_construct_validity_figure",
    "symmetric_display_limit",
]
