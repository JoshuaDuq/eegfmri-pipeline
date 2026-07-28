"""The two table builders every report section renders through."""

from __future__ import annotations

import pytest

from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    Metric,
    grid_table,
    metric_table,
)


class TestMetricTable:
    def test_renders_label_as_row_header_and_value_as_numeric_cell(self):
        rendered = metric_table([("EEG channels", 63)])
        assert "<th scope='row'>EEG channels</th>" in rendered
        assert '<td class="num">63</td>' in rendered

    def test_carries_the_shared_table_class(self):
        assert 'class="report-table"' in metric_table([("Runs", 6)])

    def test_marks_the_emphasised_row_without_markup_in_the_value(self):
        rendered = metric_table([Metric("Spearman-Brown corrected", "0.466", emphasis=True)])
        assert '<tr class="key">' in rendered
        assert "<strong>" not in rendered

    def test_plain_rows_are_not_emphasised(self):
        assert 'class="key"' not in metric_table([("Runs", 6)])

    def test_escapes_both_cells(self):
        rendered = metric_table([("<b>label</b>", "<i>value</i>")])
        assert "&lt;b&gt;label&lt;/b&gt;" in rendered
        assert "&lt;i&gt;value&lt;/i&gt;" in rendered

    def test_renders_a_missing_value_as_an_em_dash(self):
        assert "—" in metric_table([("Marker count", None)])

    def test_returns_empty_string_for_no_rows(self):
        assert metric_table([]) == ""


class TestGridTable:
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Beats"),
    )

    def test_renders_a_single_header_row_when_no_column_is_grouped(self):
        rendered = grid_table(self.columns, [["run-1", 520]])
        assert rendered.count("<tr>") == 2  # one header, one body
        assert "rowspan" not in rendered

    def test_aligns_declared_numeric_columns_in_header_and_body(self):
        rendered = grid_table(self.columns, [["run-1", 520]])
        assert '<th class="num">Beats</th>' in rendered
        assert '<td class="num">520</td>' in rendered

    def test_leaves_text_columns_unaligned(self):
        rendered = grid_table(self.columns, [["run-1", 520]])
        assert "<th>Run</th>" in rendered
        assert "<td>run-1</td>" in rendered

    def test_escapes_headers_and_cells(self):
        rendered = grid_table([Column("<b>H</b>", align=Align.TEXT)], [["<i>v</i>"]])
        assert "&lt;b&gt;H&lt;/b&gt;" in rendered
        assert "&lt;i&gt;v&lt;/i&gt;" in rendered

    def test_renders_a_missing_cell_as_an_em_dash(self):
        rendered = grid_table(self.columns, [["run-2", None]])
        assert "—" in rendered

    def test_returns_empty_string_for_no_rows(self):
        assert grid_table(self.columns, []) == ""

    def test_rejects_a_row_that_does_not_match_the_columns(self):
        with pytest.raises(ValueError, match="1 value"):
            grid_table(self.columns, [["run-1"]])

    def test_rejects_a_table_with_no_columns(self):
        with pytest.raises(ValueError, match="at least one column"):
            grid_table([], [])


class TestGroupedHeader:
    columns = (
        Column("Run", align=Align.TEXT),
        Column("exponent", group="Before ICA"),
        Column("offset (dB)", group="Before ICA"),
        Column("exponent", group="After ICA"),
        Column("offset (dB)", group="After ICA"),
        Column("Δ exponent"),
    )

    def test_spans_grouped_columns_across_the_top_header_row(self):
        rendered = grid_table(self.columns, [["run-1", "1.50", "16.0", "1.21", "8.3", "-0.30"]])
        assert "<th colspan='2' scope='colgroup'>Before ICA</th>" in rendered
        assert "<th colspan='2' scope='colgroup'>After ICA</th>" in rendered

    def test_ungrouped_columns_span_both_header_rows(self):
        rendered = grid_table(self.columns, [["run-1", "1.50", "16.0", "1.21", "8.3", "-0.30"]])
        assert "<th rowspan='2'>Run</th>" in rendered
        assert "<th rowspan='2' class=\"num\">Δ exponent</th>" in rendered

    def test_grouped_column_headers_land_in_the_second_header_row(self):
        rendered = grid_table(self.columns, [["run-1", "1.50", "16.0", "1.21", "8.3", "-0.30"]])
        header = rendered.split("</thead>")[0]
        assert header.count("<tr>") == 2
        assert header.count("exponent</th>") == 3  # two grouped, one spanning
