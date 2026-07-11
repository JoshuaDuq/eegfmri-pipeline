from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.sensor_patterns import SensorPatternSummary

BANDS = (
    "alpha",
    "beta",
    "gamma_low_clean",
    "gamma_mid_clean",
    "gamma_high_clean",
)
CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")


def test_load_haufe_artifacts_validates_stage_outputs(tmp_path: Path) -> None:
    from studies.pain_study.study2.figures.haufe_forward_patterns import (
        load_haufe_forward_pattern_artifacts,
    )

    summary = sensor_summary()
    fold_path = tmp_path / "fold.tsv"
    aggregate_path = tmp_path / "aggregate.tsv"
    stability_path = tmp_path / "stability.tsv"
    summary.fold_patterns.to_csv(fold_path, sep="\t", index=False)
    summary.aggregate_patterns.to_csv(aggregate_path, sep="\t", index=False)
    summary.stability.to_csv(stability_path, sep="\t", index=False)

    loaded = load_haufe_forward_pattern_artifacts(
        fold_path=fold_path,
        aggregate_path=aggregate_path,
        stability_path=stability_path,
        config=load_study2_config(),
    )

    assert loaded.target == "NPS"
    assert loaded.bands == BANDS
    assert loaded.channels == tuple(sorted(CHANNELS))
    assert loaded.n_subjects == 4
    assert loaded.article_ready is False


def test_build_haufe_figure_has_fixed_scientific_structure() -> None:
    from studies.pain_study.study2.figures.haufe_forward_patterns_plot import (
        build_haufe_forward_patterns_figure,
    )

    figure = build_haufe_forward_patterns_figure(sensor_summary(), load_study2_config())

    try:
        assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 92.0 / 25.4))
        assert len(figure.axes) == 7
        assert [axis.get_title().split("\n")[0] for axis in figure.axes[:5]] == [
            "Alpha",
            "Beta",
            "Low gamma",
            "Mid gamma",
            "High gamma",
        ]
        assert figure.axes[5].get_title(loc="left") == "Descriptive fold stability"
        assert figure.axes[6].get_ylabel() == ("Normalized Haufe forward-pattern loading (a.u.)")
        assert any("Preliminary cohort" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_build_haufe_figure_uses_one_shared_symmetric_scale() -> None:
    from studies.pain_study.study2.figures.haufe_forward_patterns_plot import (
        build_haufe_forward_patterns_figure,
    )

    figure = build_haufe_forward_patterns_figure(sensor_summary(), load_study2_config())

    try:
        limits = [axis.images[0].get_clim() for axis in figure.axes[:5]]
        assert all(np.allclose(limit, limits[0]) for limit in limits)
        assert np.isclose(limits[0][0], -limits[0][1])
        assert all(len(axis.collections) >= 1 for axis in figure.axes[:5])
    finally:
        plt.close(figure)


def test_haufe_writer_creates_exact_editable_svg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import studies.pain_study.study2.figures.plot_haufe_forward_patterns as module

    monkeypatch.setattr(
        module,
        "load_haufe_forward_pattern_artifacts",
        lambda **kwargs: sensor_summary(),
    )
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path / "derivatives")}
    output = module.write_haufe_forward_patterns(
        config=config,
        output_path=tmp_path / "haufe_forward_patterns.svg",
    )

    root = ElementTree.parse(output).getroot()
    assert float(root.attrib["width"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        183.0, abs=0.01
    )
    assert float(root.attrib["height"].removesuffix("pt")) * 25.4 / 72.0 == pytest.approx(
        92.0, abs=0.01
    )
    assert output.read_text(encoding="utf-8").count("<text") > 0


def sensor_summary() -> SensorPatternSummary:
    aggregate_rows = []
    fold_rows = []
    stability_rows = []
    for band_index, band in enumerate(BANDS):
        values = np.sin(np.linspace(0.2, 2.7, len(CHANNELS)) + 0.3 * band_index)
        values /= np.linalg.norm(values)
        for channel, value in zip(CHANNELS, values, strict=True):
            aggregate_rows.append(
                {
                    "target": "NPS",
                    "band": band,
                    "channel": channel,
                    "median_normalized_pattern": value,
                    "n_folds": 4,
                }
            )
            for fold in range(4):
                fold_rows.append(
                    {
                        "target": "NPS",
                        "fold": fold,
                        "test_subject": f"sub-{fold:04d}",
                        "band": band,
                        "channel": channel,
                        "haufe_pattern": value,
                        "normalized_pattern": value,
                    }
                )
        for comparison in range(6):
            stability_rows.append(
                {
                    "target": "NPS",
                    "band": band,
                    "comparison_index": comparison,
                    "fold_a": comparison % 3,
                    "fold_b": comparison % 3 + 1,
                    "spatial_correlation": 0.35 + 0.08 * comparison,
                }
            )
    return SensorPatternSummary(
        fold_patterns=pd.DataFrame(fold_rows),
        aggregate_patterns=pd.DataFrame(aggregate_rows),
        stability=pd.DataFrame(stability_rows),
        target="NPS",
        bands=BANDS,
        channels=CHANNELS,
        n_subjects=4,
        article_ready=False,
    )
