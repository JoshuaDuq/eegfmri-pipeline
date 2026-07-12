"""Strict artifact-contract tests for the Study 2 spatial-convergence figure."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize, to_hex
from matplotlib.text import Text
from PIL import Image

from studies.pain_study.study2 import paths
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.figures.spatial_convergence import (
    AUDIT_COLUMNS,
    SpatialConvergenceSummary,
    load_spatial_convergence,
)
from studies.pain_study.study2.figures.spatial_convergence_plot import (
    build_spatial_convergence_figure,
)
from studies.pain_study.study2.figures.primary_source_associations_plot import (
    CommonSourceSurfaces,
    HemisphereSurface,
)
from studies.pain_study.study2.figures.style import study2_diverging_color_map
from studies.pain_study.study2.source_vertex_manifest import (
    CommonSourceVertices,
    ensure_common_source_vertices,
)
from studies.pain_study.study2.spatial_comparison import compute_spatial_correspondence
from studies.tests.pipelines.test_study2_primary_source_associations_figure import (
    _synthetic_surfaces,
)

BANDS = ("alpha", "beta", "gamma")


def test_spatial_convergence_paths_use_study2_figure_directory(tmp_path: Path) -> None:
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    figure_directory = paths.figures_dir(config)

    assert (
        paths.spatial_convergence_figure_path(config),
        paths.spatial_convergence_png_path(config),
        paths.spatial_convergence_summary_path(config),
        paths.spatial_convergence_caption_path(config),
        paths.spatial_convergence_manifest_path(config),
    ) == (
        figure_directory / "spatial_convergence.svg",
        figure_directory / "spatial_convergence.png",
        figure_directory / "spatial_convergence_summary.tsv",
        figure_directory / "spatial_convergence_caption.txt",
        figure_directory / "spatial_convergence_manifest.json",
    )


def test_writer_creates_complete_spatial_convergence_publication_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study2.figures.plot_spatial_convergence as module

    config = _write_spatial_artifacts(tmp_path)
    surface_paths = tuple(tmp_path / f"surface-{index}" for index in range(4))
    for surface_path in surface_paths:
        surface_path.write_bytes(surface_path.name.encode("utf-8"))
    surfaces = replace(_synthetic_surfaces(), source_paths=surface_paths)
    monkeypatch.setattr(
        module,
        "load_common_source_surfaces",
        lambda *args: surfaces,
    )
    output = tmp_path / "article" / "spatial_convergence.svg"

    outputs = module.write_spatial_convergence(config=config, output_path=output)

    expected_names = {
        "spatial_convergence.svg",
        "spatial_convergence.png",
        "spatial_convergence_summary.tsv",
        "spatial_convergence_caption.txt",
        "spatial_convergence_manifest.json",
    }
    assert {path.name for path in outputs.all_files} == expected_names
    assert all(path.is_file() for path in outputs.all_files)
    assert {path.name for path in output.parent.iterdir()} == expected_names

    svg_root = ElementTree.parse(outputs.svg).getroot()
    assert float(svg_root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0,
        abs=0.01,
    )
    assert float(svg_root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        112.0,
        abs=0.01,
    )
    svg_text = outputs.svg.read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "<path" in svg_text
    assert "<dc:date" not in svg_text

    dimensions = config["study2"]["figures"]["spatial_convergence"]["dimensions_mm"]
    dpi = config["study2"]["figures"]["spatial_convergence"]["png_dpi"]
    expected_pixels = tuple(
        int(dimensions[dimension] / 25.4 * dpi) for dimension in ("width", "height")
    )
    with Image.open(outputs.png) as png:
        assert png.size == expected_pixels

    audit = pd.read_csv(outputs.summary, sep="\t")
    assert tuple(audit.columns) == AUDIT_COLUMNS
    assert tuple(audit["band"]) == BANDS

    caption = outputs.caption.read_text(encoding="utf-8")
    required_caption_phrases = (
        "resolution-matched, nuisance-residualized NPS-L2 fMRI forward covariance",
        "alpha, beta, and scanner-clean gamma EEG group source partial-r maps",
        "unthresholded masked maps",
        "BrainSMASH surrogate correlations preserve spatial autocorrelation",
        "observed Pearson r",
        "two-sided plus-one p",
        "Holm adjustment across the three bands",
        "coarse spatial correspondence",
        "not vertex independence or shared generators",
    )
    assert all(phrase in caption for phrase in required_caption_phrases)

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["bands"] == list(BANDS)
    assert manifest["n_vertices"] == 6
    assert manifest["n_surrogates"] == 3
    assert manifest["common_subject"] == "fsaverage"
    assert manifest["common_source_space_spacing"] == "oct6"
    assert manifest["family_alpha"] == 0.05
    assert manifest["figure_dimensions_mm"] == {"height": 112.0, "width": 183.0}
    assert manifest["analysis_parameters"] == {
        "correlation": "Pearson r",
        "multiple_comparison_adjustment": "Holm across bands",
        "null_method": "BrainSMASH Base",
        "p_value": "two-sided plus-one",
    }
    assert set(manifest["software"]) == {
        "matplotlib",
        "mne",
        "nibabel",
        "nilearn",
        "numpy",
        "pandas",
        "python",
        "scipy",
    }
    summary = load_spatial_convergence(config)
    expected_sources = (*summary.source_paths, *surface_paths)
    assert len(expected_sources) == len(set(expected_sources))
    assert set(manifest["source_sha256"]) == {
        str(path.resolve()) for path in expected_sources
    }
    assert manifest["source_sha256"] == {
        str(path.resolve()): _sha256(path) for path in expected_sources
    }
    expected_output_paths = tuple(
        path for path in outputs.all_files if path != outputs.manifest
    )
    assert manifest["output_sha256"] == {
        path.name: _sha256(path) for path in expected_output_paths
    }


def test_writer_fails_before_output_for_missing_scientific_inputs(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.plot_spatial_convergence import (
        write_spatial_convergence,
    )

    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    output = tmp_path / "article" / "spatial_convergence.svg"

    with pytest.raises(FileNotFoundError, match="source vertex array"):
        write_spatial_convergence(config=config, output_path=output)

    assert not output.parent.exists()


def test_writer_rejects_non_svg_output_before_reading_config(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.plot_spatial_convergence import (
        write_spatial_convergence,
    )

    with pytest.raises(ValueError, match="must be an SVG"):
        write_spatial_convergence(
            config={},
            output_path=tmp_path / "spatial_convergence.pdf",
        )


def test_spatial_convergence_cli_module_help_has_no_runtime_warning(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["MNE_DONTWRITE_HOME"] = "true"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "studies.pain_study.study2.figures.plot_spatial_convergence",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--study2-config" in result.stdout
    assert "--deriv-root" in result.stdout
    assert "--output" in result.stdout


def test_writer_is_deterministic_for_identical_spatial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study2.figures.plot_spatial_convergence as module

    first_config = _write_spatial_artifacts(tmp_path / "first")
    second_config = _write_spatial_artifacts(tmp_path / "second")
    surfaces_by_root = {}
    for name, config in (("first", first_config), ("second", second_config)):
        surface_paths = tuple(tmp_path / name / f"surface-{index}" for index in range(4))
        for surface_path in surface_paths:
            surface_path.write_bytes(surface_path.name.encode("utf-8"))
        surfaces_by_root[Path(config["paths"]["deriv_root"]).parent.name] = replace(
            _synthetic_surfaces(),
            source_paths=surface_paths,
        )

    monkeypatch.setattr(
        module,
        "load_common_source_surfaces",
        lambda config, _manifest: surfaces_by_root[
            Path(config["paths"]["deriv_root"]).parent.name
        ],
    )
    first_outputs = module.write_spatial_convergence(
        config=first_config,
        output_path=tmp_path / "first-article" / "spatial_convergence.svg",
    )
    second_outputs = module.write_spatial_convergence(
        config=second_config,
        output_path=tmp_path / "second-article" / "spatial_convergence.svg",
    )

    for first_path, second_path in zip(
        first_outputs.all_files[:-1],
        second_outputs.all_files[:-1],
        strict=True,
    ):
        assert first_path.read_bytes() == second_path.read_bytes()
    assert b"<dc:date" not in first_outputs.svg.read_bytes()
    with Image.open(first_outputs.png) as png:
        assert not any("date" in key.lower() for key in png.info)

    first_manifest = json.loads(first_outputs.manifest.read_text(encoding="utf-8"))
    second_manifest = json.loads(second_outputs.manifest.read_text(encoding="utf-8"))
    first_source_hashes = first_manifest.pop("source_sha256")
    second_source_hashes = second_manifest.pop("source_sha256")
    assert sorted(first_source_hashes.values()) == sorted(second_source_hashes.values())
    assert first_manifest == second_manifest
    assert first_manifest["output_sha256"] == second_manifest["output_sha256"]


def test_renderer_has_fixed_multimodal_structure(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    figure = build_spatial_convergence_figure(summary, _synthetic_surfaces(), config)

    try:
        assert plt.fignum_exists(figure.number)
        assert figure._suptitle is None
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 112.0 / 25.4))
        surface_axes = [
            axis for axis in figure.axes if str(axis.get_gid()).startswith("surface-")
        ]
        assert {axis.get_gid() for axis in surface_axes} == {
            f"surface-{map_name}-{hemisphere}-lateral"
            for map_name in ("fmri", *BANDS)
            for hemisphere in ("left", "right")
        }
        assert len(surface_axes) == 8

        null_axes = [axis for axis in figure.axes if str(axis.get_gid()).startswith("null-")]
        assert [axis.get_gid() for axis in null_axes] == [f"null-{band}" for band in BANDS]
        assert len(null_axes) == 3
        assert len({axis.get_xlim() for axis in null_axes}) == 1
        expected_band_colors = {
            "alpha": "#0072B2",
            "beta": "#CC79A7",
            "gamma": "#009E73",
        }
        for axis, band in zip(null_axes, BANDS, strict=True):
            result = summary.band_results[band]
            observed_lines = [
                line
                for line in axis.lines
                if line.get_gid() == "observed-correlation" and line.get_marker() == "D"
            ]
            assert len(observed_lines) == 1
            assert to_hex(observed_lines[0].get_markerfacecolor()).upper() == (
                expected_band_colors[band]
            )
            assert len([line for line in axis.lines if line.get_gid() == "zero-reference"]) == 1
            assert axis.get_title() == (
                f"r = {result.spatial_r:.3f}   plus-one two-sided p = "
                f"{result.p_value:.4f}   "
                f"Holm p = {result.holm_adjusted_p_value:.4f}"
            )
            histogram_patches = [
                patch for patch in axis.patches if patch.get_gid() == "null-histogram"
            ]
            assert len(histogram_patches) == 40
            histogram_area = sum(
                patch.get_width() * patch.get_height() for patch in histogram_patches
            )
            assert histogram_area == pytest.approx(1.0)
            expected_heights, expected_edges = np.histogram(
                result.surrogate_r,
                bins=40,
                range=axis.get_xlim(),
                density=True,
            )
            assert np.allclose(
                [patch.get_height() for patch in histogram_patches],
                expected_heights,
            )
            assert np.allclose(
                [patch.get_x() for patch in histogram_patches],
                expected_edges[:-1],
            )
        fmri_colorbar = [axis for axis in figure.axes if axis.get_gid() == "fmri-colorbar"]
        eeg_colorbar = [axis for axis in figure.axes if axis.get_gid() == "eeg-colorbar"]
        assert len(fmri_colorbar) == 1
        assert len(eeg_colorbar) == 1
        assert fmri_colorbar[0].get_xlabel() == "NPS-L2 forward covariance (covariance units)"
        assert eeg_colorbar[0].get_xlabel() == "EEG partial correlation, r"

        eeg_limits = {
            tuple(collection.get_clim())
            for axis in surface_axes
            if axis.get_gid().split("-")[1] in BANDS
            for collection in axis.collections
            if collection.get_gid() == "unthresholded-map"
        }
        assert eeg_limits == {(-0.3, 0.3)}
        fmri_limits = {
            tuple(collection.get_clim())
            for axis in surface_axes
            if axis.get_gid().split("-")[1] == "fmri"
            for collection in axis.collections
            if collection.get_gid() == "unthresholded-map"
        }
        assert fmri_limits == {(-3.0, 3.0)}
        displayed_maps = {
            axis.get_gid(): axis._study2_display_values for axis in surface_axes
        }
        assert np.array_equal(
            displayed_maps["surface-fmri-left-lateral"],
            summary.fmri_map[:3],
        )
        assert np.array_equal(
            displayed_maps["surface-fmri-right-lateral"],
            summary.fmri_map[3:],
        )
        for band in BANDS:
            eeg_map = summary.band_results[band].eeg_map
            assert np.array_equal(
                displayed_maps[f"surface-{band}-left-lateral"],
                eeg_map[:3],
            )
            assert np.array_equal(
                displayed_maps[f"surface-{band}-right-lateral"],
                eeg_map[3:],
            )

        panel_labels = [text for text in figure.texts if text.get_text() in {"a", "b"}]
        assert {text.get_text() for text in panel_labels} == {"a", "b"}
        assert all(text.get_fontsize() == 8.0 for text in panel_labels)
        assert all(text.get_fontweight() == "bold" for text in panel_labels)
        figure_text = " ".join(text.get_text() for text in figure.texts)
        assert (
            "BrainSMASH surrogate maps preserve spatial autocorrelation and carry the "
            "map-correspondence inference."
        ) in figure_text
    finally:
        plt.close(figure)


def test_renderer_rejects_all_zero_masked_fmri_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    invalid_summary = replace(summary, fmri_map=np.zeros(summary.n_vertices))

    with pytest.raises(ValueError, match="masked fMRI map must contain a finite nonzero effect"):
        build_spatial_convergence_figure(invalid_summary, _synthetic_surfaces(), config)


def test_renderer_rejects_complex_masked_fmri_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    fmri_map = summary.fmri_map.astype(np.complex128)
    fmri_map[0] += 1.0j
    invalid_summary = replace(summary, fmri_map=fmri_map)

    with pytest.raises(ValueError, match="masked fMRI map must be real-valued"):
        build_spatial_convergence_figure(invalid_summary, _synthetic_surfaces(), config)


def test_renderer_colors_only_fully_masked_faces_without_diluting_effects(
    tmp_path: Path,
) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary, surfaces = _two_face_mask_fixture(load_spatial_convergence(config))

    figure = build_spatial_convergence_figure(summary, surfaces, config)

    try:
        figure.canvas.draw()
        surface_axes = [
            axis for axis in figure.axes if str(axis.get_gid()).startswith("surface-")
        ]
        expected_face_values = {
            "fmri": (3.0, -3.0),
            "alpha": (0.3, -0.3),
            "beta": (0.2, -0.2),
            "gamma": (0.1, -0.1),
        }
        color_map = study2_diverging_color_map()
        expected_background_color = plt.get_cmap("gray_r")(0.5)
        for axis in surface_axes:
            _, map_name, hemisphere, _ = axis.get_gid().split("-")
            masked_map_collections = [
                collection
                for collection in axis.collections
                if collection.get_gid() == "unthresholded-map"
            ]
            assert len(masked_map_collections) == 1
            masked_map_collection = masked_map_collections[0]
            assert len(masked_map_collection.get_facecolors()) == 2
            assert np.array_equal(
                masked_map_collection._study2_effect_face_mask,
                np.array([True, False]),
            )

            expected_limit = 3.0 if map_name == "fmri" else 0.3
            assert tuple(masked_map_collection.get_clim()) == (
                -expected_limit,
                expected_limit,
            )
            hemisphere_index = 0 if hemisphere == "left" else 1
            expected_value = expected_face_values[map_name][hemisphere_index]
            expected_color = color_map(
                Normalize(-expected_limit, expected_limit)(expected_value)
            )
            rendered_colors = masked_map_collection.get_facecolors()
            assert sum(np.allclose(color, expected_color) for color in rendered_colors) == 1
            assert (
                sum(
                    np.allclose(color, expected_background_color)
                    for color in rendered_colors
                )
                == 1
            )
    finally:
        plt.close(figure)


def test_renderer_does_not_mutate_common_surface_coordinates(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    surfaces = _synthetic_surfaces()
    original_left = surfaces.left.coordinates.copy()
    original_right = surfaces.right.coordinates.copy()

    figure = build_spatial_convergence_figure(summary, surfaces, config)

    try:
        assert np.array_equal(surfaces.left.coordinates, original_left)
        assert np.array_equal(surfaces.right.coordinates, original_right)
    finally:
        plt.close(figure)


def test_renderer_rejects_all_zero_masked_eeg_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    band_results = dict(summary.band_results)
    band_results["beta"] = replace(
        band_results["beta"],
        eeg_map=np.zeros(summary.n_vertices),
    )
    invalid_summary = replace(summary, band_results=band_results)

    with pytest.raises(
        ValueError,
        match="masked beta EEG map must contain a finite nonzero effect",
    ):
        build_spatial_convergence_figure(invalid_summary, _synthetic_surfaces(), config)


def test_renderer_rejects_complex_masked_eeg_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    eeg_map = summary.band_results["beta"].eeg_map.astype(np.complex128)
    eeg_map[0] += 1.0j
    band_results = dict(summary.band_results)
    band_results["beta"] = replace(band_results["beta"], eeg_map=eeg_map)
    invalid_summary = replace(summary, band_results=band_results)

    with pytest.raises(ValueError, match="masked beta EEG map must be real-valued"):
        build_spatial_convergence_figure(invalid_summary, _synthetic_surfaces(), config)


def test_renderer_outlines_only_holm_significant_null_panel(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    band_results = dict(summary.band_results)
    band_results["alpha"] = replace(band_results["alpha"], holm_significant=True)
    significant_summary = replace(summary, band_results=band_results)

    figure = build_spatial_convergence_figure(
        significant_summary,
        _synthetic_surfaces(),
        config,
    )

    try:
        outlined_axes = {
            axis.get_gid()
            for axis in figure.axes
            if any(patch.get_gid() == "holm-significant-outline" for patch in axis.patches)
        }
        assert outlined_axes == {"null-alpha"}
        assert "*" not in " ".join(text.get_text() for text in figure.findobj(Text))
    finally:
        plt.close(figure)


def test_renderer_rejects_configured_band_order_mismatch(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary = load_spatial_convergence(config)
    bands = config["study2"]["figures"]["spatial_convergence"]["bands"]
    bands[0], bands[1] = bands[1], bands[0]

    with pytest.raises(ValueError, match="configured band order must exactly match the summary"):
        build_spatial_convergence_figure(summary, _synthetic_surfaces(), config)


def test_loader_reconciles_complete_spatial_family(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)

    summary = load_spatial_convergence(config)

    assert summary.bands == BANDS
    assert summary.n_vertices == 6
    assert summary.n_surrogates == 3
    assert summary.mask.sum() == 6
    assert tuple(summary.audit.columns) == AUDIT_COLUMNS


def test_loader_rejects_differing_band_fmri_map(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    beta_path = paths.spatial_fmri_map_path(config, band="beta")
    beta_map = np.load(beta_path, allow_pickle=False)
    beta_map[0] += 1.0
    _write_fmri_map_and_refresh_hash(config, band="beta", values=beta_map)

    with pytest.raises(ValueError, match="do not share one reference map"):
        load_spatial_convergence(config)


def test_loader_rejects_mismatched_saved_adjusted_p_value(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    summary_path = paths.spatial_correspondence_summary_path(config)
    summary = pd.read_csv(summary_path, sep="\t")
    summary.loc[summary["band"].eq("alpha"), "holm_adjusted_p_value"] = 0.25
    summary.to_csv(summary_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="does not match recomputed inference"):
        load_spatial_convergence(config)


def test_loader_rejects_map_with_wrong_vertex_count(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    np.save(paths.spatial_eeg_map_path(config, band="alpha"), np.ones(5))

    with pytest.raises(ValueError, match="source vertex manifest"):
        load_spatial_convergence(config)


@pytest.mark.parametrize("artifact", ["eeg", "fmri", "surrogate"])
def test_loader_rejects_complex_spatial_arrays(tmp_path: Path, artifact: str) -> None:
    config = _write_spatial_artifacts(tmp_path)
    artifact_paths = {
        "eeg": paths.spatial_eeg_map_path(config, band="alpha"),
        "fmri": paths.spatial_fmri_map_path(config, band="alpha"),
        "surrogate": paths.spatial_surrogate_maps_path(config, band="alpha"),
    }
    artifact_path = artifact_paths[artifact]
    values = np.load(artifact_path, allow_pickle=False).astype(np.complex128)
    np.save(artifact_path, values)
    if artifact in {"fmri", "surrogate"}:
        _refresh_band_artifact_hash(config, band="alpha", artifact=artifact)

    labels = {
        "eeg": "alpha EEG map",
        "fmri": "alpha fMRI map",
        "surrogate": "alpha surrogate maps",
    }
    with pytest.raises(ValueError, match=rf"Study 2 {labels[artifact]} must be real-valued\."):
        load_spatial_convergence(config)


@pytest.mark.parametrize("invalid_value", [-1.1, -1.0, 1.0, 1.1])
def test_loader_rejects_eeg_values_outside_correlation_bounds(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    config = _write_spatial_artifacts(tmp_path)
    eeg_path = paths.spatial_eeg_map_path(config, band="alpha")
    eeg_map = np.load(eeg_path, allow_pickle=False)
    eeg_map[0] = invalid_value
    np.save(eeg_path, eeg_map)

    with pytest.raises(
        ValueError, match=r"alpha EEG map values must lie strictly within \(-1, 1\)"
    ):
        load_spatial_convergence(config)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_loader_rejects_nonboolean_stored_mask(
    tmp_path: Path,
    dtype: type[np.generic],
) -> None:
    config = _write_spatial_artifacts(tmp_path)
    np.save(paths.spatial_mask_path(config), np.ones(6, dtype=dtype))

    with pytest.raises(ValueError, match="must have boolean dtype"):
        load_spatial_convergence(config)


def test_loader_rejects_metadata_naming_wrong_manifest(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    metadata_path = paths.spatial_surrogate_metadata_path(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_vertex_manifest_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong vertex manifest"):
        load_spatial_convergence(config)


def test_loader_rejects_fmri_mismatch_above_absolute_tolerance(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    gamma_path = paths.spatial_fmri_map_path(config, band="gamma")
    gamma_map = np.load(gamma_path, allow_pickle=False)
    gamma_map[0] += 1.1e-12
    _write_fmri_map_and_refresh_hash(config, band="gamma", values=gamma_map)

    with pytest.raises(ValueError, match="do not share one reference map"):
        load_spatial_convergence(config)


def test_loader_requires_every_surrogate_artifact(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    paths.spatial_surrogate_maps_path(config, band="gamma").unlink()

    with pytest.raises(FileNotFoundError, match="gamma surrogate maps artifact not found"):
        load_spatial_convergence(config)


def test_loader_rejects_modified_surrogates_with_unchanged_inference(tmp_path: Path) -> None:
    config = _write_spatial_artifacts(tmp_path)
    surrogate_path = paths.spatial_surrogate_maps_path(config, band="alpha")
    surrogate_maps = np.load(surrogate_path, allow_pickle=False)
    correspondence_inputs = {
        "eeg_map": np.load(
            paths.spatial_eeg_map_path(config, band="alpha"),
            allow_pickle=False,
        ),
        "fmri_map": np.load(
            paths.spatial_fmri_map_path(config, band="alpha"),
            allow_pickle=False,
        ),
        "mask": np.load(paths.spatial_mask_path(config), allow_pickle=False),
        "config": config,
    }
    original_result = compute_spatial_correspondence(
        surrogate_maps=surrogate_maps,
        **correspondence_inputs,
    )
    surrogate_maps[0] *= 2.0
    modified_result = compute_spatial_correspondence(
        surrogate_maps=surrogate_maps,
        **correspondence_inputs,
    )
    assert modified_result.p_value == original_result.p_value
    np.save(surrogate_path, surrogate_maps)

    with pytest.raises(ValueError, match="alpha spatial metadata names the wrong surrogate maps"):
        load_spatial_convergence(config)


def _two_face_mask_fixture(
    summary: SpatialConvergenceSummary,
) -> tuple[SpatialConvergenceSummary, CommonSourceSurfaces]:
    vertices = np.arange(4, dtype=np.int64)
    manifest = CommonSourceVertices(
        lh_vertices=vertices,
        rh_vertices=vertices,
        common_subject="fsaverage",
        spacing="oct6",
    )
    coordinates = np.array(
        [
            [-1.0, -1.0, -0.5],
            [1.0, 0.0, -0.8],
            [-0.5, 1.0, 0.7],
            [1.2, 0.6, 0.4],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    sulcal_depth = np.array([-1.0, 0.0, 1.0, 0.5])
    surfaces = CommonSourceSurfaces(
        left=HemisphereSurface(
            hemisphere="left",
            vertex_numbers=vertices.copy(),
            coordinates=coordinates.copy(),
            faces=faces.copy(),
            sulcal_depth=sulcal_depth.copy(),
        ),
        right=HemisphereSurface(
            hemisphere="right",
            vertex_numbers=vertices.copy(),
            coordinates=coordinates * np.array([-1.0, 1.0, 1.0]),
            faces=faces.copy(),
            sulcal_depth=sulcal_depth.copy(),
        ),
        source_paths=(),
    )
    mask = np.array([True, True, True, False, True, True, True, False])
    fmri_map = np.array([3.0, 3.0, 3.0, 300.0, -3.0, -3.0, -3.0, -400.0])
    band_values = {"alpha": 0.3, "beta": 0.2, "gamma": 0.1}
    band_results = {
        band: replace(
            summary.band_results[band],
            eeg_map=np.array([value, value, value, 0.99, -value, -value, -value, -0.99]),
        )
        for band, value in band_values.items()
    }
    return (
        replace(
            summary,
            fmri_map=fmri_map,
            mask=mask,
            band_results=band_results,
            vertices_manifest=manifest,
        ),
        surfaces,
    )


def _write_spatial_artifacts(tmp_path: Path) -> dict:
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 3
    ensure_common_source_vertices(
        array_path=paths.source_vertex_manifest_path(config),
        metadata_path=paths.source_vertex_metadata_path(config),
        vertices=(np.asarray([0, 2, 4]), np.asarray([1, 3, 5])),
        common_subject="fsaverage",
        spacing="oct6",
    )

    fmri_map = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    eeg_maps = {
        "alpha": fmri_map / 10.0,
        "beta": -fmri_map / 10.0,
        "gamma": np.asarray([0.1, -0.1, 0.2, -0.2, 0.3, -0.3]),
    }
    surrogate_maps = np.asarray(
        [
            [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
            [-2.0, -3.0, 1.0, 3.0, -1.0, 2.0],
            [1.0, 3.0, -2.0, 2.0, -3.0, -1.0],
        ]
    )
    paths.spatial_dir(config).mkdir(parents=True)
    np.save(paths.spatial_mask_path(config), np.ones(6, dtype=np.bool_))
    for band in BANDS:
        np.save(paths.spatial_eeg_map_path(config, band=band), eeg_maps[band])
        np.save(paths.spatial_fmri_map_path(config, band=band), fmri_map)
        np.save(paths.spatial_surrogate_maps_path(config, band=band), surrogate_maps)

    metadata = {
        "schema_version": 1,
        "method": "BrainSMASH Base",
        "n_surrogates": 3,
        "common_subject": "fsaverage",
        "common_source_space_spacing": "oct6",
        "source_vertex_manifest_sha256": _sha256(paths.source_vertex_manifest_path(config)),
        "analysis_mask_sha256": _sha256(paths.spatial_mask_path(config)),
        "bands": {
            band: {
                "seed": 42 + band_index,
                "masked_vertices": 6,
                "fmri_map_sha256": _sha256(paths.spatial_fmri_map_path(config, band=band)),
                "surrogate_maps_sha256": _sha256(
                    paths.spatial_surrogate_maps_path(config, band=band)
                ),
            }
            for band_index, band in enumerate(BANDS)
        },
    }
    paths.spatial_surrogate_metadata_path(config).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "band": BANDS,
            "spatial_r": [1.0, -1.0, -2.0 / 7.0],
            "p_value": [0.5, 0.5, 1.0],
            "meaningful": [True, True, True],
            "holm_adjusted_p_value": [1.0, 1.0, 1.0],
            "holm_significant": [False, False, False],
        }
    ).to_csv(paths.spatial_correspondence_summary_path(config), sep="\t", index=False)
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fmri_map_and_refresh_hash(
    config: dict,
    *,
    band: str,
    values: np.ndarray,
) -> None:
    fmri_path = paths.spatial_fmri_map_path(config, band=band)
    np.save(fmri_path, values)
    _refresh_band_artifact_hash(config, band=band, artifact="fmri")


def _refresh_band_artifact_hash(config: dict, *, band: str, artifact: str) -> None:
    artifact_paths = {
        "fmri": paths.spatial_fmri_map_path(config, band=band),
        "surrogate": paths.spatial_surrogate_maps_path(config, band=band),
    }
    metadata_keys = {
        "fmri": "fmri_map_sha256",
        "surrogate": "surrogate_maps_sha256",
    }
    metadata_path = paths.spatial_surrogate_metadata_path(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bands"][band][metadata_keys[artifact]] = _sha256(artifact_paths[artifact])
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
