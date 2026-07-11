from __future__ import annotations

from pathlib import Path
import json
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from studies.tests.pipelines.test_study2_primary_source_associations import _write_artifacts


def test_primary_source_figure_has_fixed_publication_structure(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )
    from studies.pain_study.study2.figures.primary_source_associations_plot import (
        build_primary_source_associations_figure,
    )

    config = _write_artifacts(tmp_path)
    summary = load_primary_source_associations(config)
    figure = build_primary_source_associations_figure(
        summary,
        _synthetic_surfaces(),
        config,
    )

    try:
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 100.0 / 25.4))
        surface_axes = [axis for axis in figure.axes if str(axis.get_gid()).startswith("surface-")]
        assert len(surface_axes) == 12
        assert {axis.get_gid().split("-")[1] for axis in surface_axes} == {
            "alpha",
            "beta",
            "gamma",
        }
        color_axes = [axis for axis in figure.axes if axis.get_gid() == "shared-colorbar"]
        assert len(color_axes) == 1
        assert color_axes[0].get_xlabel() == "Fisher mean partial correlation, r"
        text = " ".join(label.get_text() for label in figure.texts)
        assert "Alpha" in text
        assert "Beta" in text
        assert "Scanner-clean gamma" in text
        assert "Holm q = 0.030" in text
        assert text.count("no family-corrected cluster") == 2
    finally:
        plt.close(figure)


def test_primary_source_figure_keeps_unthresholded_maps_and_corrected_contours(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study2.figures.primary_source_associations import (
        load_primary_source_associations,
    )
    from studies.pain_study.study2.figures.primary_source_associations_plot import (
        build_primary_source_associations_figure,
    )

    config = _write_artifacts(tmp_path)
    summary = load_primary_source_associations(config)
    figure = build_primary_source_associations_figure(
        summary,
        _synthetic_surfaces(),
        config,
    )

    try:
        surface_axes = [axis for axis in figure.axes if str(axis.get_gid()).startswith("surface-")]
        for axis in surface_axes:
            assert any(
                collection.get_gid() == "unthresholded-effect" for collection in axis.collections
            )
        contour_axes = {
            axis.get_gid()
            for axis in surface_axes
            if any(
                collection.get_gid() == "corrected-contour"
                or getattr(collection, "_study2_corrected_contour", False)
                for collection in axis.collections
            )
        }
        assert contour_axes == {"surface-alpha-left-lateral", "surface-alpha-left-medial"}
    finally:
        plt.close(figure)


def test_primary_source_writer_creates_complete_publication_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study2.figures.plot_primary_source_associations as module

    config = _write_artifacts(tmp_path)
    monkeypatch.setattr(module, "load_common_source_surfaces", lambda *args: _synthetic_surfaces())
    output = tmp_path / "article" / "primary_source_associations.svg"

    paths = module.write_primary_source_associations(config=config, output_path=output)

    expected = {
        "primary_source_associations.svg",
        "primary_source_associations.png",
        "primary_source_associations_vertices.tsv",
        "primary_source_associations_clusters.tsv",
        "primary_source_associations_summary.tsv",
        "primary_source_associations_caption.txt",
        "primary_source_associations_manifest.json",
    }
    assert {path.name for path in paths.all_files} == expected
    assert all(path.is_file() for path in paths.all_files)
    root = ElementTree.parse(paths.svg).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0, abs=0.01
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        100.0, abs=0.01
    )
    assert paths.svg.read_text(encoding="utf-8").count("<text") > 0

    vertices = pd.read_csv(paths.vertices, sep="\t", keep_default_na=False)
    clusters = pd.read_csv(paths.clusters, sep="\t")
    summary = pd.read_csv(paths.summary, sep="\t")
    assert tuple(vertices.columns) == module.VERTEX_COLUMNS
    assert tuple(clusters.columns) == module.CLUSTER_COLUMNS
    assert tuple(summary.columns) == module.SUMMARY_COLUMNS
    caption = paths.caption.read_text(encoding="utf-8")
    assert "target-retrained maximum-cluster" in caption
    assert "Holm corrected across alpha, beta, and scanner-clean gamma" in caption
    assert "deep generators" in caption

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["figure_dimensions_mm"] == {"height": 100.0, "width": 183.0}
    assert set(manifest["output_sha256"]) == expected.difference(
        {"primary_source_associations_manifest.json"}
    )
    assert len(manifest["source_sha256"]) == 16


def test_primary_source_writer_fails_before_output_for_missing_inputs(tmp_path: Path) -> None:
    from studies.pain_study.study2.config import load_study2_config
    from studies.pain_study.study2.figures.plot_primary_source_associations import (
        write_primary_source_associations,
    )

    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    output = tmp_path / "article" / "primary_source_associations.svg"

    with pytest.raises(FileNotFoundError, match="source vertex array"):
        write_primary_source_associations(config=config, output_path=output)

    assert not output.parent.exists()


def test_primary_source_writer_requires_svg_output(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.plot_primary_source_associations import (
        write_primary_source_associations,
    )

    with pytest.raises(ValueError, match="must be an SVG"):
        write_primary_source_associations(
            config={},
            output_path=tmp_path / "primary_source_associations.pdf",
        )


def _synthetic_surfaces():
    from studies.pain_study.study2.figures.primary_source_associations_plot import (
        CommonSourceSurfaces,
        HemisphereSurface,
    )

    coordinates = np.array(
        [
            [-0.6, -0.8, -0.4],
            [-0.6, 0.8, -0.4],
            [-0.6, 0.0, 0.9],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    sulcal = np.array([-1.0, 0.0, 1.0], dtype=float)
    left = HemisphereSurface(
        hemisphere="left",
        vertex_numbers=np.array([0, 2, 4], dtype=int),
        coordinates=coordinates,
        faces=faces,
        sulcal_depth=sulcal,
    )
    right = HemisphereSurface(
        hemisphere="right",
        vertex_numbers=np.array([1, 3, 5], dtype=int),
        coordinates=coordinates * np.array([-1.0, 1.0, 1.0]),
        faces=faces,
        sulcal_depth=sulcal,
    )
    return CommonSourceSurfaces(left=left, right=right, source_paths=())
