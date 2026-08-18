"""The two table shapes the subject report is built from.

Every table in this report is one of two things: a short list of named measurements,
or a grid with one row per run, per component, or per stage. Rendering them through
one pair of builders is what keeps a reader from having to work out, halfway down the
document, whether a table with different padding and a different run label came from
a different pipeline.

Alignment is declared per column rather than inferred. A "right-align everything but
the first column" rule reads the component ledger's ``Marked by`` column and the
spectra section's narrowband summaries as numbers, and both are prose.

Run labels are not rewritten here. A call site that has a recording id passes it
through :func:`~eeg_pipeline.preprocessing.report.style.run_label` itself, because a
naming convention that lives inside a rendering helper is a convention nobody can
find.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

#: Class every table in this report carries, and the hook :data:`REPORT_CSS` styles.
TABLE_CLASS = "report-table"

#: Class marking a cell whose content is a numeral, for right alignment and lining
#: figures. Applied to both the header and the body cells of a numeric column, so the
#: header sits over the digits it names.
NUMERIC_CLASS = "num"

#: Rendered in a cell that has no value.
#:
#: An em dash rather than an empty cell, so that "this run recorded nothing" and "this
#: row is short a column" do not look the same. The literal character rather than
#: ``&mdash;``: the report is UTF-8 throughout, and a literal survives being read back
#: out of the rendered document by anything that is not a browser.
MISSING = "—"


class Align(Enum):
    """Whether a column holds prose or numerals."""

    TEXT = "text"
    NUM = "num"


@dataclass(frozen=True)
class Metric:
    """One row of a :func:`metric_table`.

    ``emphasis`` marks the row the panel exists to report, which several sections
    already distinguished by wrapping both cells in ``<strong>``. Carrying it as a flag
    rather than as markup in the value keeps every cell escapable.

    ``indent`` marks a row that breaks down the one above it — the per-reason drop
    counts under a total. Previously spelled with two ``&nbsp;`` in the label, which
    puts layout in the data and collapses if the label is ever read by anything but a
    browser.
    """

    label: str
    value: object
    emphasis: bool = False
    indent: bool = False


#: Class marking the row a metric table exists to report.
EMPHASIS_CLASS = "key"

#: Class marking a row that breaks down the row above it.
INDENT_CLASS = "sub"


@dataclass(frozen=True)
class Column:
    """One column of a :func:`grid_table`.

    ``group`` names a spanning header placed above this column and its neighbours.
    Columns sharing a ``group`` must be adjacent; a column with no ``group`` spans
    both header rows instead.

    ``code`` renders the body cells in monospace. It marks a column of identifiers a
    reader matches against something outside the document — a config key, a component
    name, a stage name — rather than reads as prose.
    """

    header: str
    align: Align = Align.NUM
    group: str | None = None
    code: bool = False


def _cell(value: object, *, align: Align, tag: str = "td", code: bool = False) -> str:
    if value is None:
        text = MISSING
    else:
        text = html.escape(str(value))
        if code:
            text = f"<code>{text}</code>"
    if align is Align.NUM:
        return f'<{tag} class="{NUMERIC_CLASS}">{text}</{tag}>'
    return f"<{tag}>{text}</{tag}>"


def metric_table(rows: Iterable[Metric | tuple[str, object]]) -> str:
    """Render a two-column list of named measurements.

    The value column is numeric: this shape is used for scalars a reader compares
    down the column, and a stray prose value reads no worse right-aligned than a
    column of digits reads ragged.

    Plain ``(label, value)`` pairs are accepted for the common case; a
    :class:`Metric` is only needed to mark the emphasised row.
    """
    body = []
    for row in rows:
        metric = row if isinstance(row, Metric) else Metric(row[0], row[1])
        classes = [
            name
            for name, active in ((EMPHASIS_CLASS, metric.emphasis), (INDENT_CLASS, metric.indent))
            if active
        ]
        attribute = f' class="{" ".join(classes)}"' if classes else ""
        body.append(
            f"<tr{attribute}><th scope='row'>{html.escape(str(metric.label))}</th>"
            f"{_cell(metric.value, align=Align.NUM)}</tr>"
        )
    if not body:
        return ""
    return f'<table class="{TABLE_CLASS}"><tbody>{"".join(body)}</tbody></table>'


@dataclass(frozen=True)
class SpanningRow:
    """A row whose leading cells carry values and whose note spans the rest.

    A run that produced no measurement still earns a row, because a table that lists
    only the runs that succeeded silently shortens itself. What that row cannot do is
    put a sentence under a column headed "Beats", so the sentence spans the columns it
    has no values for.
    """

    lead: Sequence[object]
    note: str


def _header(columns: Sequence[Column]) -> str:
    """Render one header row, or two when any column declares a group."""
    if not any(column.group for column in columns):
        cells = "".join(_cell(column.header, align=column.align, tag="th") for column in columns)
        return f"<thead><tr>{cells}</tr></thead>"

    top: list[str] = []
    bottom: list[str] = []
    index = 0
    while index < len(columns):
        column = columns[index]
        if column.group is None:
            top.append(_spanning_header(column))
            index += 1
            continue
        span = index
        while span < len(columns) and columns[span].group == column.group:
            span += 1
        width = span - index
        top.append(f"<th colspan='{width}' scope='colgroup'>{html.escape(column.group)}</th>")
        bottom.extend(
            _cell(grouped.header, align=grouped.align, tag="th") for grouped in columns[index:span]
        )
        index = span
    return f"<thead><tr>{''.join(top)}</tr><tr>{''.join(bottom)}</tr></thead>"


def _spanning_row(row: SpanningRow, columns: Sequence[Column]) -> str:
    lead = list(row.lead)
    if len(lead) >= len(columns):
        raise ValueError(
            f"A spanning row must leave at least one column for its note: got "
            f"{len(lead)} lead value(s) for {len(columns)} column(s)."
        )
    cells = "".join(
        _cell(value, align=column.align, code=column.code) for value, column in zip(lead, columns)
    )
    span = len(columns) - len(lead)
    return f"<tr>{cells}<td colspan='{span}'>{html.escape(row.note)}</td></tr>"


def _spanning_header(column: Column) -> str:
    """Render a header for a column that has no group, spanning both header rows."""
    classes = f' class="{NUMERIC_CLASS}"' if column.align is Align.NUM else ""
    return f"<th rowspan='2'{classes}>{html.escape(column.header)}</th>"


def grid_table(
    columns: Sequence[Column],
    rows: Iterable[Sequence[object] | SpanningRow],
) -> str:
    """Render a headed grid with one row per run, component, or stage.

    A row shorter than ``columns`` is a programming error and raises: silently padding
    it would put a measurement under the wrong heading, which is the one failure this
    table cannot be allowed to render. A row that genuinely has no values for its later
    columns says so with a :class:`SpanningRow`.
    """
    if not columns:
        raise ValueError("A grid table needs at least one column.")
    body = []
    for row in rows:
        if isinstance(row, SpanningRow):
            body.append(_spanning_row(row, columns))
            continue
        values = list(row)
        if len(values) != len(columns):
            raise ValueError(
                f"Row has {len(values)} value(s) for {len(columns)} column(s): {values!r}"
            )
        cells = "".join(
            _cell(value, align=column.align, code=column.code)
            for value, column in zip(values, columns)
        )
        body.append(f"<tr>{cells}</tr>")
    if not body:
        return ""
    return f'<table class="{TABLE_CLASS}">{_header(columns)}<tbody>{"".join(body)}</tbody></table>'


__all__ = [
    "Align",
    "Column",
    "EMPHASIS_CLASS",
    "INDENT_CLASS",
    "MISSING",
    "Metric",
    "NUMERIC_CLASS",
    "SpanningRow",
    "TABLE_CLASS",
    "grid_table",
    "metric_table",
]
