"""Publication rendering for Study 2 primary cortical associations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study2.figures.primary_source_associations import (
    PrimarySourceAssociations,
)
from studies.pain_study.study2.figures.style import figure_size_inches, publication_style
from studies.pain_study.study2.source_vertex_manifest import CommonSourceVertices
from studies.pain_study.study2.validation import require_config_string

FIGURE_CONFIG_KEY = "study2.figures.primary_source_associations"
SURFACE_VIEWS = (
    ("left", "lateral"),
    ("right", "lateral"),
    ("left", "medial"),
    ("right", "medial"),
)


@dataclass(frozen=True)
class HemisphereSurface:
    hemisphere: str
    vertex_numbers: np.ndarray
    coordinates: np.ndarray
    faces: np.ndarray
    sulcal_depth: np.ndarray


@dataclass(frozen=True)
class CommonSourceSurfaces:
    left: HemisphereSurface
    right: HemisphereSurface
    source_paths: tuple[Path, ...]

    def for_hemisphere(self, hemisphere: str) -> HemisphereSurface:
        if hemisphere == "left":
            return self.left
        if hemisphere == "right":
            return self.right
        raise ValueError(f"Unknown cortical hemisphere: {hemisphere!r}.")


def load_common_source_surfaces(
    config: Any,
    manifest: CommonSourceVertices,
) -> CommonSourceSurfaces:
    """Load the exact common-space mesh used by the source arrays."""

    figure_config = _figure_config(config)
    subjects_dir = Path(
        require_config_string(config, "study2.source_modeling.anatomy.subjects_dir")
    )
    subject = manifest.common_subject
    surface_name = str(figure_config["surface"])
    subject_dir = subjects_dir / subject / "surf"
    surface_paths = tuple(
        subject_dir / filename
        for filename in (
            f"lh.{surface_name}",
            "lh.sulc",
            f"rh.{surface_name}",
            "rh.sulc",
        )
    )
    missing = [path for path in surface_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Study 2 common cortical surface files are missing: {missing}.")

    import mne
    from nibabel.freesurfer.io import read_geometry, read_morph_data

    source_space = mne.setup_source_space(
        subject,
        spacing=manifest.spacing,
        subjects_dir=str(subjects_dir),
        add_dist=False,
        verbose=False,
    )
    if len(source_space) != 2:
        raise ValueError("Study 2 common surface source space must contain two hemispheres.")

    surfaces = []
    for hemisphere_index, (hemisphere, prefix, expected_vertices) in enumerate(
        (
            ("left", "lh", manifest.lh_vertices),
            ("right", "rh", manifest.rh_vertices),
        )
    ):
        coordinates, _full_faces = read_geometry(subject_dir / f"{prefix}.{surface_name}")
        sulcal_depth = read_morph_data(subject_dir / f"{prefix}.sulc")
        source = source_space[hemisphere_index]
        source_vertices = np.asarray(source["vertno"], dtype=np.int64)
        if not np.array_equal(source_vertices, expected_vertices):
            raise ValueError(
                f"Study 2 {hemisphere} source-space vertices do not match the saved manifest."
            )
        use_tris = np.asarray(source["use_tris"], dtype=np.int64)
        faces = _reindex_faces(use_tris, source_vertices, n_full_vertices=len(coordinates))
        surfaces.append(
            HemisphereSurface(
                hemisphere=hemisphere,
                vertex_numbers=source_vertices,
                coordinates=np.asarray(coordinates, dtype=float)[source_vertices],
                faces=faces,
                sulcal_depth=np.asarray(sulcal_depth, dtype=float)[source_vertices],
            )
        )
    geometry = CommonSourceSurfaces(
        left=surfaces[0],
        right=surfaces[1],
        source_paths=surface_paths,
    )
    _validate_surfaces(geometry, manifest)
    return geometry


def build_primary_source_associations_figure(
    summary: PrimarySourceAssociations,
    surfaces: CommonSourceSurfaces,
    config: Any,
) -> Figure:
    """Render the three-band unthresholded effects and corrected contours."""

    figure_config = _figure_config(config)
    _validate_summary(summary, figure_config)
    _validate_surfaces(surfaces, summary.vertices_manifest)
    dimensions = figure_config["dimensions_mm"]
    font_family = str(figure_config["font_family"])
    color_map = _association_color_map()

    with publication_style(font_family):
        figure = plt.figure(
            figsize=figure_size_inches(dimensions),
            facecolor="white",
        )
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        outer_grid = figure.add_gridspec(
            1,
            3,
            left=0.018,
            right=0.982,
            bottom=0.205,
            top=0.785,
            wspace=0.035,
        )
        for band_index, band_spec in enumerate(figure_config["bands"]):
            band = str(band_spec["name"])
            result = summary.band_results[band]
            band_grid = outer_grid[0, band_index].subgridspec(
                2,
                2,
                wspace=-0.18,
                hspace=-0.12,
            )
            for view_index, (hemisphere, view) in enumerate(SURFACE_VIEWS):
                axis = figure.add_subplot(
                    band_grid[view_index // 2, view_index % 2],
                    projection="3d",
                )
                axis.set_gid(f"surface-{band}-{hemisphere}-{view}")
                effect, corrected = _hemisphere_values(
                    result.effect_r,
                    result.corrected_vertex_mask,
                    surfaces,
                    hemisphere=hemisphere,
                )
                _draw_surface(
                    axis,
                    surfaces.for_hemisphere(hemisphere),
                    effect=effect,
                    corrected=corrected,
                    view=view,
                    color_map=color_map,
                    limit=summary.display_limit,
                )
                axis.set_title(
                    f"{hemisphere[0].upper()}{hemisphere[1:]} {view}",
                    fontsize=5.0,
                    pad=-3.0,
                )
            _add_band_header(
                figure,
                band_index=band_index,
                band_spec=band_spec,
                n_subjects=summary.n_subjects,
                holm_adjusted_p_value=result.holm_adjusted_p_value,
                significant=result.significant,
            )

        _add_shared_colorbar(
            figure,
            color_map=color_map,
            limit=summary.display_limit,
        )
        figure.text(
            0.5,
            0.975,
            "Cortical source association with held-out NPS prediction",
            ha="center",
            va="top",
            fontsize=7.6,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.018,
            (
                "Unthresholded Fisher-mean partial correlations; dark contours require "
                "target-retrained max-cluster control and Holm correction across bands."
            ),
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#333333",
        )
    return figure


def _draw_surface(
    axis,
    surface: HemisphereSurface,
    *,
    effect: np.ndarray,
    corrected: np.ndarray,
    view: str,
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    from nilearn import plotting

    mesh = (surface.coordinates, surface.faces)
    before_effect = len(axis.collections)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"mpl_toolkits")
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=effect,
            bg_map=surface.sulcal_depth,
            hemi=surface.hemisphere,
            view=view,
            cmap=color_map,
            colorbar=False,
            threshold=None,
            vmin=-limit,
            vmax=limit,
            symmetric_cbar=True,
            bg_on_data=True,
            axes=axis,
            figure=axis.figure,
        )
    for collection in axis.collections[before_effect:]:
        collection.set_gid("unthresholded-effect")

    if corrected.any():
        before_contour = len(axis.collections)
        plotting.plot_surf_contours(
            surf_mesh=mesh,
            roi_map=corrected.astype(np.int8),
            levels=[1],
            colors=["#202020"],
            legend=False,
            axes=axis,
            figure=axis.figure,
        )
        contour_collections = axis.collections[before_contour:]
        if contour_collections:
            for collection in contour_collections:
                collection.set_gid("corrected-contour")
                collection.set_linewidth(0.45)
        else:
            for collection in axis.collections[before_effect:]:
                collection._study2_corrected_contour = True
                collection.set_linewidth(0.45)


def _hemisphere_values(
    effect: np.ndarray,
    corrected: np.ndarray,
    surfaces: CommonSourceSurfaces,
    *,
    hemisphere: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_left = surfaces.left.vertex_numbers.size
    if hemisphere == "left":
        return effect[:n_left], corrected[:n_left]
    return effect[n_left:], corrected[n_left:]


def _add_band_header(
    figure: Figure,
    *,
    band_index: int,
    band_spec: Mapping[str, object],
    n_subjects: int,
    holm_adjusted_p_value: float,
    significant: bool,
) -> None:
    x_position = (band_index + 0.5) / 3.0
    status = (
        f"Holm q = {holm_adjusted_p_value:.3f}"
        if significant
        else "no family-corrected cluster"
    )
    figure.text(
        x_position,
        0.905,
        f"{band_spec['label']}\n{band_spec['frequency_label']}",
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
    )
    figure.text(
        x_position,
        0.842,
        f"n = {n_subjects} · {status}",
        ha="center",
        va="top",
        fontsize=5.4,
        color="#333333",
    )


def _add_shared_colorbar(
    figure: Figure,
    *,
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    color_axis = figure.add_axes((0.29, 0.105, 0.42, 0.018))
    color_axis.set_gid("shared-colorbar")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=color_map),
        cax=color_axis,
        orientation="horizontal",
        ticks=(-limit, 0.0, limit),
    )
    colorbar.set_label("Fisher mean partial correlation, r", labelpad=2.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2.0, width=0.5, pad=1.5)


def _association_color_map() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "study2_source_association",
        ("#2166AC", "#F7F7F7", "#D95F0E"),
        N=256,
    )


def _reindex_faces(
    faces: np.ndarray,
    vertices: np.ndarray,
    *,
    n_full_vertices: int,
) -> np.ndarray:
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise ValueError("Study 2 common source surface must contain triangular faces.")
    lookup = np.full(n_full_vertices, -1, dtype=np.int64)
    lookup[vertices] = np.arange(vertices.size)
    if np.any(faces < 0) or np.any(faces >= n_full_vertices):
        raise ValueError("Study 2 common source triangles contain invalid vertex numbers.")
    reindexed = lookup[faces]
    if np.any(reindexed < 0):
        raise ValueError("Study 2 common source triangles are not confined to source vertices.")
    return reindexed


def _validate_surfaces(
    surfaces: CommonSourceSurfaces,
    manifest: CommonSourceVertices,
) -> None:
    for surface, hemisphere, expected in (
        (surfaces.left, "left", manifest.lh_vertices),
        (surfaces.right, "right", manifest.rh_vertices),
    ):
        if surface.hemisphere != hemisphere:
            raise ValueError("Study 2 common surface hemisphere labels are invalid.")
        if not np.array_equal(surface.vertex_numbers, expected):
            raise ValueError(
                f"Study 2 {hemisphere} plotted vertices do not match the source manifest."
            )
        n_vertices = expected.size
        if surface.coordinates.shape != (n_vertices, 3):
            raise ValueError(f"Study 2 {hemisphere} surface coordinates have invalid shape.")
        if surface.sulcal_depth.shape != (n_vertices,):
            raise ValueError(f"Study 2 {hemisphere} sulcal data have invalid shape.")
        if surface.faces.ndim != 2 or surface.faces.shape[1] != 3 or surface.faces.size == 0:
            raise ValueError(f"Study 2 {hemisphere} surface faces have invalid shape.")
        if np.any(surface.faces < 0) or np.any(surface.faces >= n_vertices):
            raise ValueError(f"Study 2 {hemisphere} surface faces contain invalid indices.")
        if (
            not np.isfinite(surface.coordinates).all()
            or not np.isfinite(surface.sulcal_depth).all()
        ):
            raise ValueError(f"Study 2 {hemisphere} surface geometry contains non-finite values.")


def _validate_summary(
    summary: PrimarySourceAssociations,
    figure_config: Mapping[str, object],
) -> None:
    bands = tuple(str(spec["name"]) for spec in figure_config["bands"])
    if summary.bands != bands or bands != ("alpha", "beta", "gamma"):
        raise ValueError("Study 2 primary source figure requires alpha, beta, and gamma.")
    if not np.isfinite(summary.display_limit) or summary.display_limit <= 0.0:
        raise ValueError("Study 2 primary source display limit must be positive and finite.")


def _figure_config(config: Any) -> Mapping[str, object]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


__all__ = [
    "CommonSourceSurfaces",
    "HemisphereSurface",
    "build_primary_source_associations_figure",
    "load_common_source_surfaces",
]
