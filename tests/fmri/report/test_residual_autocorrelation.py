from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.report.figures import residual_autocorrelation


def _image(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))
    return path


def test_residual_acf_preserves_acquired_lags_across_a_retained_frame_gap(
    tmp_path: Path,
) -> None:
    mask = _image(
        tmp_path / "mask.nii.gz",
        np.ones((2, 1, 1), dtype=np.uint8),
    )
    residual = _image(
        tmp_path / "run-01_residual.nii.gz",
        np.array(
            [
                [[[1.0, 2.0, 4.0, 5.0, 6.0]]],
                [[[2.0, 4.0, 8.0, 10.0, 12.0]]],
            ]
        ),
    )

    runs = residual_autocorrelation.collect_residual_autocorrelation(
        run_labels=("run-01",),
        residual_paths=(residual,),
        retained_frame_indices=((0, 1, 3, 4, 5),),
        acquired_frame_counts=(6,),
        mask_path=mask,
        max_lag_frames=2,
    )

    assert len(runs) == 1
    run = runs[0]
    assert run.label == "run-01"
    assert run.lags_frames == (1, 2)
    assert run.valid_pairs == (3, 2)
    assert run.retained_frames == 5
    assert run.acquired_frames == 6
    assert run.voxel_count == 2
    assert run.quartiles[0] == pytest.approx((8.08 / 17.2,) * 3)
    assert run.quartiles[1] == pytest.approx((0.32 / 17.2,) * 3)


def _run(
    label: str,
    quartiles: tuple[tuple[float, float, float], ...],
) -> residual_autocorrelation.RunResidualAutocorrelation:
    return residual_autocorrelation.RunResidualAutocorrelation(
        label=label,
        lags_frames=(1, 2),
        valid_pairs=(9, 8),
        quartiles=quartiles,
        retained_frames=10,
        acquired_frames=11,
        voxel_count=100,
    )


def test_residual_acf_figure_draws_run_panels_with_median_and_iqr() -> None:
    runs = (
        _run("run-01", ((-0.1, 0.1, 0.3), (-0.2, 0.0, 0.2))),
        _run("run-02", ((0.0, 0.2, 0.4), (-0.3, -0.1, 0.1))),
    )

    figure = residual_autocorrelation.residual_autocorrelation_figure(
        runs,
        tr=2.0,
        title="Residual autocorrelation",
    )

    # One axis carrying one median line per run, and the widest run's IQR as a band.
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    median_line = next(line for line in axis.lines if line.get_label() == "run-01")
    np.testing.assert_allclose(median_line.get_xdata(), [2.0, 4.0])
    np.testing.assert_allclose(median_line.get_ydata(), [0.1, 0.0])
    assert axis.collections, "the interquartile band is missing"
    strip = " ".join(text.get_text() for text in figure.texts)
    # The per-panel pair counts moved into the strip when the panels merged.
    assert "retained pairs per voxel" in strip
    assert "no criterion" in strip.lower()
    plt.close(figure)


def test_residual_acf_tsv_contains_every_plotted_quantile(tmp_path: Path) -> None:
    runs = (_run("run-01", ((-0.1, 0.1, 0.3), (-0.2, 0.0, 0.2))),)

    path = residual_autocorrelation.write_residual_autocorrelation_tsv(
        runs,
        tr=2.0,
        path=tmp_path / "residual_autocorrelation.tsv",
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "Run\tLag (frames)\tLag (s)\tValid retained pairs per voxel\tACF Q1\tACF median\tACF Q3",
        "run-01\t1\t2.000000\t9\t-0.100000\t0.100000\t0.300000",
        "run-01\t2\t4.000000\t8\t-0.200000\t0.000000\t0.200000",
    ]


def test_every_run_is_drawn_on_one_shared_axis() -> None:
    # Six subplots showed that six runs behave the same and left the spread between
    # them -- the one thing a per-run panel is read for -- to be reconstructed by eye.
    # The variance-inflation panel already makes this argument against itself.
    runs = tuple(
        _run(f"run-{i + 1:02d}", ((-0.1 + 0.01 * i, 0.1, 0.3), (-0.2, 0.0, 0.2)))
        for i in range(6)
    )
    figure = residual_autocorrelation.residual_autocorrelation_figure(runs, tr=0.9)
    try:
        drawable = [
            axis
            for axis in figure.axes
            if axis.lines or axis.collections
        ]
        median_lines = [
            line for axis in drawable for line in axis.lines if line.get_label() != "_nolegend_"
        ]
    finally:
        plt.close(figure)

    assert len(drawable) == 1, f"{len(drawable)} axes carry data; expected one"
    assert len(median_lines) == 6, "one median line per run"


def test_each_run_line_is_separately_identifiable() -> None:
    runs = tuple(
        _run(f"run-{i + 1:02d}", ((-0.1, 0.1, 0.3), (-0.2, 0.0, 0.2))) for i in range(4)
    )
    figure = residual_autocorrelation.residual_autocorrelation_figure(runs, tr=0.9)
    try:
        legend = figure.axes[0].get_legend()
        labels = [entry.get_text() for entry in legend.get_texts()]
    finally:
        plt.close(figure)

    for run in runs:
        assert any(run.label in label for label in labels), f"{run.label} is not in the legend"
