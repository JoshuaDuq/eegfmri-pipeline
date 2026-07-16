from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.config.loader import load_study1_config

EXPECTED_SENSOR_TOPOGRAPHY_CONFIG = {
    "dimensions_mm": {"width": 183.0, "height": 86.0},
    "png_dpi": 600,
    "bands": [
        {"name": "alpha", "label": "Alpha", "frequency_hz": [8.0, 12.9]},
        {"name": "beta", "label": "Beta", "frequency_hz": [13.0, 30.0]},
        {
            "name": "gamma_low_clean",
            "label": "Low gamma",
            "frequency_hz": [30.1, 38.0],
        },
        {
            "name": "gamma_mid_clean",
            "label": "Mid gamma",
            "frequency_hz": [43.0, 56.0],
        },
        {
            "name": "gamma_high_clean",
            "label": "High gamma",
            "frequency_hz": [67.0, 77.0],
        },
    ],
    "signature_model": {
        "rank_tolerance": 1.0e-10,
        "max_condition_number": 100.0,
        "minimum_residual_standard_deviation": 1.0e-12,
    },
    "inference": {
        "cluster_forming_p": 0.01,
        "family_alpha": 0.05,
        "max_null_draws": 10000,
        "seed": 20260715,
    },
}


def test_sensor_topography_config_exposes_only_approved_settings() -> None:
    config = load_study1_config()

    assert config["study1"]["figures"]["sensor_topographies"] == (EXPECTED_SENSOR_TOPOGRAPHY_CONFIG)


def test_save_publication_png_is_byte_reproducible_and_atomic(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_style import save_publication_png

    output_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for output_path in output_paths:
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [1.0, 0.0])
        saved_path = save_publication_png(
            figure,
            output_path,
            _style_config(),
            dimensions_mm={"width": 25.4, "height": 25.4},
            dpi=600,
        )
        assert saved_path == output_path
        assert not plt.fignum_exists(figure.number)

    assert output_paths[0].read_bytes() == output_paths[1].read_bytes()
    assert sorted(tmp_path.iterdir()) == output_paths


def test_save_publication_png_preserves_destination_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.validity_style import save_publication_png

    output_path = tmp_path / "figure.png"
    original_bytes = b"existing publication"
    output_path.write_bytes(original_bytes)
    figure, _ = plt.subplots()

    def fail_save(path: Path, **kwargs: object) -> None:
        Path(path).write_bytes(b"incomplete publication")
        raise RuntimeError("render failed")

    monkeypatch.setattr(figure, "savefig", fail_save)

    with pytest.raises(RuntimeError, match="render failed"):
        save_publication_png(
            figure,
            output_path,
            _style_config(),
            dimensions_mm={"width": 25.4, "height": 25.4},
            dpi=300,
        )

    assert output_path.read_bytes() == original_bytes
    assert sorted(tmp_path.iterdir()) == [output_path]
    assert not plt.fignum_exists(figure.number)


def test_save_publication_png_rejects_non_png_suffix(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.validity_style import save_publication_png

    figure = plt.figure()
    with pytest.raises(ValueError, match=r"require an \.png path"):
        save_publication_png(
            figure,
            tmp_path / "figure.svg",
            _style_config(),
            dimensions_mm={"width": 25.4, "height": 25.4},
            dpi=600,
        )
    plt.close(figure)


@pytest.mark.parametrize("dpi", [True, False, 300.0, "600", None])
def test_save_publication_png_rejects_non_integer_dpi(
    tmp_path: Path,
    dpi: object,
) -> None:
    from studies.pain_study.study1.figures.validity_style import save_publication_png

    figure = plt.figure()
    with pytest.raises(TypeError, match="DPI must be an integer"):
        save_publication_png(
            figure,
            tmp_path / "figure.png",
            _style_config(),
            dimensions_mm={"width": 25.4, "height": 25.4},
            dpi=dpi,  # type: ignore[arg-type]
        )
    plt.close(figure)


@pytest.mark.parametrize("dpi", [-1, 0, 299])
def test_save_publication_png_rejects_dpi_below_print_resolution(
    tmp_path: Path,
    dpi: int,
) -> None:
    from studies.pain_study.study1.figures.validity_style import save_publication_png

    figure = plt.figure()
    with pytest.raises(ValueError, match="DPI must be at least 300"):
        save_publication_png(
            figure,
            tmp_path / "figure.png",
            _style_config(),
            dimensions_mm={"width": 25.4, "height": 25.4},
            dpi=dpi,
        )
    plt.close(figure)


def _style_config() -> ConfigDict:
    return ConfigDict(
        {
            "study1": {
                "figures": {
                    "validity": {
                        "font": {
                            "family": "DejaVu Sans",
                            "axis_label_pt": 7.0,
                            "tick_label_pt": 6.0,
                            "legend_pt": 6.0,
                        },
                        "style": {"axis_line_width_pt": 0.6},
                    }
                }
            }
        }
    )
