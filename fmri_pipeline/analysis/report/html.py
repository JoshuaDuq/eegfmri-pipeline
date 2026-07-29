"""Document primitives for the subject report.

Blocks are data, not markup: a figure module returns a figure, the assembler
describes a document, and only :func:`render` knows HTML. That is what lets the
document's structure be tested without parsing markup, and what will let the group
report reuse the same primitives.

Styling follows the EEG report conventions -- tabular numerals so a column of values
lines up its decimal points, and no zebra striping, because the light-to-dark
neutral ramp is reserved for pipeline decisions rather than spent on rows that
decide nothing.
"""

from __future__ import annotations

import base64
import html as html_escape
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Figure:
    title: str
    path: Path
    caption: str = ""
    #: Dense figures are raster and stay raster; carried so the assembler does not
    #: have to re-derive it from the suffix.
    dense: bool = True


@dataclass(frozen=True)
class Table:
    title: str
    html: str = ""
    tsv_path: Optional[Path] = None
    caption: str = ""


@dataclass(frozen=True)
class KeyValues:
    title: str
    items: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Note:
    text: str


Block = Union[Figure, Table, KeyValues, Note]


@dataclass(frozen=True)
class Section:
    slug: str
    title: str
    blocks: Tuple[Block, ...] = ()
    #: Collapsed sections hold diagnostics: available, but not competing with the
    #: result for a reader's attention.
    collapsed: bool = False


@dataclass(frozen=True)
class Document:
    title: str
    subtitle: str = ""
    sections: Tuple[Section, ...] = ()


_CSS = """
:root { --fg:#111; --muted:#555; --bg:#fff; --border:#e6e6ea; }
* { box-sizing: border-box; }
body { font-family: Arial, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: var(--bg); color: var(--fg); margin: 0; line-height: 1.4; }
.layout { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 28px;
          max-width: 1240px; margin: 0 auto; padding: 24px; }
nav { position: sticky; top: 24px; align-self: start; font-size: 13px; }
nav ol { list-style: none; margin: 0; padding: 0; }
nav li { margin: 0 0 6px 0; }
.toc-link { color: var(--muted); text-decoration: none; }
.toc-link:hover { color: var(--fg); text-decoration: underline; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 17px; margin: 0 0 10px 0; }
.subhead { color: var(--muted); margin: 0 0 20px 0; font-size: 13px; }
section { border-top: 1px solid var(--border); padding-top: 18px; margin-bottom: 26px; }
.fig { border: 1px solid var(--border); border-radius: 8px; padding: 12px;
       margin: 0 0 14px 0; background: #fff; }
.fig-title { font-weight: 600; margin: 0 0 6px 0; font-size: 14px; }
.fig-cap { color: var(--muted); font-size: 12px; margin-top: 6px; }
img { width: 100%; height: auto; display: block; }
.missing { color: var(--muted); font-size: 12px; font-style: italic; padding: 18px;
           border: 1px dashed var(--border); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 12px;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--border); padding: 5px 8px; text-align: left; }
thead th { border-bottom: 1px solid #b8b8b8; font-weight: 600; }
.kvs { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 4px 14px;
       font-size: 13px; }
.k { color: var(--muted); }
.overflow { overflow-x: auto; }
details > summary { cursor: pointer; color: var(--muted); font-size: 13px;
                    margin-bottom: 10px; }
a { color: #0a58ca; }
@media (max-width: 820px) { .layout { grid-template-columns: 1fr; }
                            nav { position: static; } }
"""


def _esc(value: object) -> str:
    return html_escape.escape("" if value is None else str(value))


def _relpath(base_dir: Path, target: Path) -> str:
    try:
        return str(Path(target).relative_to(base_dir))
    except ValueError:
        return str(target)


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _image_source(path: Path, *, base_dir: Path, embed: bool) -> str:
    if not embed:
        return _relpath(base_dir, path)
    data = path.read_bytes()
    return f"data:{_mime(path)};base64,{base64.b64encode(data).decode('ascii')}"


def _render_figure(figure: Figure, *, base_dir: Path, embed: bool) -> str:
    parts = ['<div class="fig">', f'<div class="fig-title">{_esc(figure.title)}</div>']
    try:
        source = _image_source(figure.path, base_dir=base_dir, embed=embed)
        parts.append(
            f'<img src="{_esc(source)}" loading="lazy" alt="{_esc(figure.title)}" />'
        )
    except OSError as exc:
        # A missing panel is a gap in the document, not a reason to lose it.
        logger.warning("Figure %s could not be read (%s)", figure.path, exc)
        parts.append(
            '<div class="missing">This figure could not be rendered: '
            f"{_esc(figure.path.name)}</div>"
        )
    if figure.caption:
        parts.append(f'<div class="fig-cap">{_esc(figure.caption)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _render_table(table: Table, *, base_dir: Path) -> str:
    parts = ['<div class="fig">', f'<div class="fig-title">{_esc(table.title)}</div>']
    if table.tsv_path is not None:
        link = _esc(_relpath(base_dir, table.tsv_path))
        parts.append(f'<div class="fig-cap"><a href="{link}">Download TSV</a></div>')
    if table.html:
        # Wide tables scroll inside their own box rather than widening the page.
        parts.append(f'<div class="overflow">{table.html}</div>')
    if table.caption:
        parts.append(f'<div class="fig-cap">{_esc(table.caption)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _render_block(block: Block, *, base_dir: Path, embed: bool) -> str:
    if isinstance(block, Figure):
        return _render_figure(block, base_dir=base_dir, embed=embed)
    if isinstance(block, Table):
        return _render_table(block, base_dir=base_dir)
    if isinstance(block, KeyValues):
        rows = "".join(
            f'<div class="k">{_esc(k)}</div><div>{_esc(v)}</div>'
            for k, v in block.items
        )
        return (
            f'<div class="fig"><div class="fig-title">{_esc(block.title)}</div>'
            f'<div class="kvs">{rows}</div></div>'
        )
    return f'<p class="fig-cap">{_esc(block.text)}</p>'


def render(document: Document, *, base_dir: Path, embed: bool = True) -> str:
    """Render ``document`` as one self-contained HTML page."""
    base_dir = Path(base_dir)
    toc = "".join(
        f"<li><a class=\"toc-link\" href=\"#{_esc(s.slug)}\">{_esc(s.title)}</a></li>"
        for s in document.sections
    )

    body = []
    for section in document.sections:
        blocks = "".join(
            _render_block(b, base_dir=base_dir, embed=embed) for b in section.blocks
        )
        inner = (
            f"<details><summary>Show diagnostics</summary>{blocks}</details>"
            if section.collapsed
            else blocks
        )
        body.append(
            f'<section id="{_esc(section.slug)}">'
            f"<h2>{_esc(section.title)}</h2>{inner}</section>"
        )

    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8" />'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />'
        f"<title>{_esc(document.title)}</title><style>{_CSS}</style></head>"
        '<body><div class="layout">'
        f"<nav><ol>{toc}</ol></nav>"
        f"<main><h1>{_esc(document.title)}</h1>"
        f'<p class="subhead">{_esc(document.subtitle)}</p>'
        f"{''.join(body)}</main>"
        "</div></body></html>\n"
    )


__all__ = [
    "Block",
    "Document",
    "Figure",
    "KeyValues",
    "Note",
    "Section",
    "Table",
    "render",
]
