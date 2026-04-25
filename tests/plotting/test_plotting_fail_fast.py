from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from eeg_pipeline.plotting.config import PlotConfig
from eeg_pipeline.plotting.io.figures import save_fig


matplotlib.use("Agg", force=True)


def _plot_config(config: dict) -> PlotConfig:
    return PlotConfig.from_config({"plotting": config})


def test_plot_config_requires_known_figure_size() -> None:
    plot_cfg = _plot_config(
        {
            "figure_sizes": {"standard": [10.0, 8.0]},
            "defaults": {},
            "styling": {"colors": {"gray": "#555555"}},
        }
    )

    with pytest.raises(ValueError, match="Unknown figure size"):
        plot_cfg.get_figure_size("missing")


def test_plot_config_rejects_invalid_color_values() -> None:
    plot_cfg = _plot_config(
        {
            "figure_sizes": {"standard": [10.0, 8.0]},
            "defaults": {},
            "styling": {"colors": {"gray": "not-a-color"}},
        }
    )

    with pytest.raises(ValueError, match="Invalid color"):
        plot_cfg.get_color("gray")


def test_plot_config_requires_known_color_name() -> None:
    plot_cfg = _plot_config(
        {
            "figure_sizes": {"standard": [10.0, 8.0]},
            "defaults": {},
            "styling": {"colors": {"gray": "#555555"}},
        }
    )

    with pytest.raises(ValueError, match="Unknown color"):
        plot_cfg.get_color("missing")


def test_save_fig_propagates_original_save_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fig = plt.figure()

    def raise_permission_error(*args, **kwargs) -> None:
        raise PermissionError("cannot write figure")

    monkeypatch.setattr(fig, "savefig", raise_permission_error)

    with pytest.raises(PermissionError, match="cannot write figure"):
        save_fig(
            fig,
            tmp_path / "figure.png",
            formats=("png",),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.1,
            overwrite=True,
        )
