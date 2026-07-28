from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriPlottingConfig
from fmri_pipeline.analysis.reporting import generate_fmri_space_section


def _stat_img(seed: int = 0) -> nib.Nifti1Image:
    rng = np.random.default_rng(seed)
    return nib.Nifti1Image(
        rng.standard_normal((12, 12, 12)).astype(np.float32), np.eye(4)
    )


def _cfg(**kwargs) -> FmriPlottingConfig:
    base = dict(enabled=True, plot_types=("slices", "glass", "hist"), threshold_mode="z")
    base.update(kwargs)
    return FmriPlottingConfig(**base).normalized()


def _section(tmp_path: Path, **kwargs):
    params = dict(
        space="mni",
        stat_img=_stat_img(),
        out_base_dir=tmp_path,
        formats=("png",),
        z_threshold=2.3,
        include_unthresholded=False,
        plot_types=("hist",),
        cfg=_cfg(),
    )
    params.update(kwargs)
    return generate_fmri_space_section(**params)


def test_a_figure_yields_one_report_entry_per_figure_not_per_format(
    tmp_path: Path,
) -> None:
    section = _section(
        tmp_path, formats=("png", "svg"), cfg=_cfg(formats=("png", "svg"))
    )
    titles = [image.title for image in section.images]
    assert len(titles) == len(set(titles))


def test_both_requested_formats_are_still_written_to_disk(tmp_path: Path) -> None:
    _section(tmp_path, formats=("png", "svg"), cfg=_cfg(formats=("png", "svg")))
    out_dir = tmp_path / "plots" / "mni"
    assert (out_dir / "z_hist.png").exists()
    assert (out_dir / "z_hist.svg").exists()


def test_a_failing_panel_does_not_abort_the_section(tmp_path: Path) -> None:
    with patch(
        "fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic",
        side_effect=RuntimeError("boom"),
    ):
        section = _section(
            tmp_path, include_unthresholded=True, plot_types=("slices", "hist")
        )
    # The histogram survived even though the mosaic raised.
    assert any("histogram" in image.title.lower() for image in section.images)


def test_no_unthresholded_glass_brain_is_produced(tmp_path: Path) -> None:
    section = _section(tmp_path, include_unthresholded=True, plot_types=("glass",))
    titles = " ".join(image.title.lower() for image in section.images)
    assert "unthresholded" not in titles


def test_figures_are_closed_when_saving_fails(tmp_path: Path) -> None:
    plt.close("all")
    with patch("matplotlib.figure.Figure.savefig", side_effect=OSError("disk full")):
        _section(tmp_path)
    assert plt.get_fignums() == []


def test_one_sided_configuration_reaches_the_map_panels(tmp_path: Path) -> None:
    with patch(
        "fmri_pipeline.analysis.report.figures.stat_maps.stat_map_mosaic"
    ) as mock_mosaic:
        mock_mosaic.return_value = plt.figure()
        _section(tmp_path, plot_types=("slices",), cfg=_cfg(two_sided=False))
    assert mock_mosaic.call_args.kwargs["two_sided"] is False
    plt.close("all")


def test_regenerating_a_section_produces_byte_identical_output(tmp_path: Path) -> None:
    def render(directory: str) -> bytes:
        _section(
            tmp_path,
            out_base_dir=tmp_path / directory,
            formats=("svg",),
            cfg=_cfg(formats=("svg",)),
        )
        return (tmp_path / directory / "plots" / "mni" / "z_hist.svg").read_bytes()

    first = hashlib.sha256(render("a")).hexdigest()
    time.sleep(1.1)
    assert hashlib.sha256(render("b")).hexdigest() == first


def test_cluster_extent_is_not_described_as_inference(tmp_path: Path) -> None:
    section = _section(
        tmp_path, plot_types=("clusters",), cfg=_cfg(cluster_min_voxels=20)
    )
    caption = " ".join(table.caption for table in section.tables).lower()
    assert "not familywise-error-corrected" in caption
    assert "cluster-level significan" not in caption


def test_a_height_threshold_alone_carries_no_extent_claim(tmp_path: Path) -> None:
    section = _section(tmp_path, plot_types=("clusters",), cfg=_cfg())
    caption = " ".join(table.caption for table in section.tables).lower()
    assert "height threshold" in caption
    assert "extent filter" not in caption
