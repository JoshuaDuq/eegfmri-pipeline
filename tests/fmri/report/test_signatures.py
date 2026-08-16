"""Contracts for the signature expression panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from fmri_pipeline.analysis.report.figures import signatures
from fmri_pipeline.analysis.report.figures.signatures import (
    SignaturePoint,
    signature_dot_plot,
)


def _points() -> list[SignaturePoint]:
    return [
        SignaturePoint(name="NPS", dot=12.0, cosine=0.31, pearson_r=0.28, n_voxels=9000),
        SignaturePoint(
            name="SIIPS", dot=-4.0, cosine=-0.12, pearson_r=-0.10, n_voxels=9000
        ),
        SignaturePoint(name="PINES", dot=0.5, cosine=0.02, pearson_r=0.01, n_voxels=9000),
    ]


def test_every_signature_gets_a_mark() -> None:
    figure = signature_dot_plot(_points())
    try:
        labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
        assert {"NPS", "SIIPS", "PINES"} <= set(labels)
    finally:
        plt.close(figure)


def test_zero_is_marked_because_sign_is_the_interpretation() -> None:
    figure = signature_dot_plot(_points())
    try:
        positions = [
            float(line.get_xdata()[0])
            for line in figure.axes[0].lines
            if len(line.get_xdata())
        ]
        assert any(abs(p) < 1e-9 for p in positions)
    finally:
        plt.close(figure)


def test_cosine_is_the_default_metric_not_the_dot_product() -> None:
    """The dot product carries the effect map's units, so it is not comparable."""
    figure = signature_dot_plot(_points())
    try:
        assert "cosine" in figure.axes[0].get_xlabel().lower()
    finally:
        plt.close(figure)


def test_the_dot_product_axis_warns_that_it_is_not_comparable() -> None:
    figure = signature_dot_plot(_points(), metric="dot")
    try:
        assert "not comparable" in figure.axes[0].get_xlabel().lower()
    finally:
        plt.close(figure)


def test_an_unknown_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="metric"):
        signature_dot_plot(_points(), metric="euclidean")


def test_a_signature_missing_the_metric_is_dropped_not_drawn_as_zero() -> None:
    """A missing similarity is not a similarity of zero."""
    points = _points() + [
        SignaturePoint(name="GONE", dot=1.0, cosine=None, pearson_r=None, n_voxels=10)
    ]
    figure = signature_dot_plot(points)
    try:
        labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
        assert "GONE" not in labels
    finally:
        plt.close(figure)


def test_a_dropped_signature_is_declared_rather_than_silently_omitted() -> None:
    points = _points() + [
        SignaturePoint(name="GONE", dot=1.0, cosine=None, pearson_r=None, n_voxels=10)
    ]
    figure = signature_dot_plot(points)
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "1 signature" in text
    finally:
        plt.close(figure)


def test_no_usable_signature_raises_rather_than_drawing_an_empty_panel() -> None:
    with pytest.raises(ValueError, match="finite"):
        signature_dot_plot([])


def test_the_panel_states_how_many_voxels_it_summarises() -> None:
    figure = signature_dot_plot(_points())
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "9,000" in text
    finally:
        plt.close(figure)


def test_a_varying_voxel_count_is_reported_as_a_range() -> None:
    """Signatures cover different territory; one number would misdescribe them."""
    points = [
        SignaturePoint(name="A", dot=1.0, cosine=0.2, pearson_r=0.2, n_voxels=8000),
        SignaturePoint(name="B", dot=1.0, cosine=0.3, pearson_r=0.3, n_voxels=9500),
    ]
    figure = signature_dot_plot(points)
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "8,000" in text and "9,500" in text
    finally:
        plt.close(figure)


# --- the report section reads what the analysis run wrote -------------------


def _tsv(directory, rows: str) -> Path:
    path = Path(directory) / "signature_expression.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "signature\tdot\tcosine\tpearson_r\tn_voxels\tweight_path\n"
    path.write_text(header + rows, encoding="utf-8")
    return path


def test_expression_is_read_from_the_tsv_the_analysis_run_wrote(tmp_path) -> None:
    """The report reads derivatives; it does not recompute signature expression."""
    from fmri_pipeline.analysis.report.figures.signatures import read_expression_tsv

    _tsv(tmp_path, "NPS\t12.0\t0.31\t0.28\t9000\t/w/nps.nii.gz\n")
    points = read_expression_tsv(tmp_path / "signature_expression.tsv")
    assert [p.name for p in points] == ["NPS"]
    assert points[0].cosine == pytest.approx(0.31)
    assert points[0].n_voxels == 9000


def test_a_blank_metric_reads_as_missing_not_as_zero(tmp_path) -> None:
    """The writer emits an empty field for an unavailable metric."""
    from fmri_pipeline.analysis.report.figures.signatures import read_expression_tsv

    _tsv(tmp_path, "NPS\t12.0\t\t\t9000\t/w/nps.nii.gz\n")
    point = read_expression_tsv(tmp_path / "signature_expression.tsv")[0]
    assert point.cosine is None
    assert point.pearson_r is None
    assert point.dot == pytest.approx(12.0)


def test_a_missing_tsv_yields_no_points_rather_than_raising(tmp_path) -> None:
    """No configured signatures is the stock configuration, not a fault."""
    from fmri_pipeline.analysis.report.figures.signatures import read_expression_tsv

    assert read_expression_tsv(tmp_path / "absent.tsv") == []


# --- the reader locates columns by name, not position ---------------------


def _write(tmp_path, header, *rows):
    path = tmp_path / "signature_expression.tsv"
    path.write_text(
        "\n".join(["\t".join(header)] + ["\t".join(row) for row in rows]) + "\n"
    )
    return path


def test_the_reader_survives_a_column_inserted_before_the_metrics(tmp_path) -> None:
    # The writer lives in the analysis package and this reader in the report package,
    # coupled only by the order of a TSV header. Parsed by position, an inserted
    # column made the panel plot a number that is not the one it names.
    path = _write(
        tmp_path,
        ["signature", "weight_path", "dot", "cosine", "pearson_r", "n_voxels"],
        ["nps", "/weights/nps.nii.gz", "12.5", "0.42", "0.40", "1000"],
    )
    points = signatures.read_expression_tsv(path)
    assert len(points) == 1
    assert points[0].name == "nps"
    assert points[0].cosine == pytest.approx(0.42)
    assert points[0].n_voxels == 1000


def test_the_reader_handles_reordered_columns(tmp_path) -> None:
    path = _write(
        tmp_path,
        ["n_voxels", "cosine", "signature", "pearson_r", "dot"],
        ["512", "-0.31", "siips", "-0.30", "-7.5"],
    )
    point = signatures.read_expression_tsv(path)[0]
    assert point.name == "siips"
    assert point.cosine == pytest.approx(-0.31)
    assert point.dot == pytest.approx(-7.5)
    assert point.n_voxels == 512


def test_a_table_missing_a_required_column_yields_nothing(tmp_path) -> None:
    # Better than guessing: a wrong value here is a wrong scientific claim, and the
    # panel names the metric it is drawing.
    path = _write(tmp_path, ["signature", "dot"], ["nps", "12.5"])
    assert signatures.read_expression_tsv(path) == []


def test_a_blank_metric_stays_unmeasured_rather_than_zero(tmp_path) -> None:
    path = _write(
        tmp_path,
        ["signature", "dot", "cosine", "pearson_r", "n_voxels"],
        ["nps", "12.5", "", "0.4", "100"],
    )
    assert signatures.read_expression_tsv(path)[0].cosine is None


def test_an_empty_table_yields_nothing(tmp_path) -> None:
    path = tmp_path / "signature_expression.tsv"
    path.write_text("")
    assert signatures.read_expression_tsv(path) == []


# --- a table when a plot would add nothing --------------------------------
#
# A lollipop chart is read for the ordering across many marks. Two marks on an axis
# are two numbers, and the report printed those same two numbers immediately beneath
# the figure -- the same values twice on one screen.


def test_the_table_carries_every_quantity() -> None:
    points = [
        signatures.SignaturePoint(
            name="NPS", dot=1.25, cosine=0.034, pearson_r=0.031, n_voxels=76_963
        ),
        signatures.SignaturePoint(
            name="SIIPS1", dot=9.4, cosine=0.158, pearson_r=None, n_voxels=252_312
        ),
    ]
    table, rows = signatures.signature_table(points)
    assert "Cosine" in table and "Dot product" in table and "Voxels" in table
    assert "+0.034" in table and "76,963" in table
    # An unmeasured correlation is not a correlation of zero.
    assert "n/a" in table
    assert len(rows) == 3


def test_the_plot_is_reserved_for_enough_signatures_to_order() -> None:
    assert signatures.MIN_SIGNATURES_FOR_A_PLOT > 2
