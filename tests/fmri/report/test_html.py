from __future__ import annotations

from pathlib import Path

import pytest

from fmri_pipeline.analysis.report.html import (
    Document,
    Figure,
    KeyValues,
    Note,
    Section,
    Table,
    render,
)


def _png(tmp_path: Path, name: str = "a.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def _doc(*sections: Section) -> Document:
    return Document(
        title="sub-01 · task-heat",
        subtitle="First-level report",
        sections=tuple(sections),
    )


def test_a_section_becomes_a_navigable_anchor(tmp_path: Path) -> None:
    html = render(_doc(Section(slug="qc", title="Quality control", blocks=())), base_dir=tmp_path)
    assert 'id="qc"' in html
    assert 'href="#qc"' in html


def test_the_table_of_contents_lists_every_section(tmp_path: Path) -> None:
    html = render(
        _doc(
            Section(slug="model", title="Model", blocks=()),
            Section(slug="results", title="Results", blocks=()),
        ),
        base_dir=tmp_path,
    )
    assert html.count('class="toc-link"') == 2


def test_a_figure_is_embedded_as_a_data_uri_when_requested(tmp_path: Path) -> None:
    section = Section(slug="s", title="S", blocks=(Figure(title="F", path=_png(tmp_path)),))
    html = render(_doc(section), base_dir=tmp_path, embed=True)
    assert "data:image/png;base64," in html


def test_a_figure_is_linked_relatively_when_not_embedded(tmp_path: Path) -> None:
    section = Section(slug="s", title="S", blocks=(Figure(title="F", path=_png(tmp_path)),))
    html = render(_doc(section), base_dir=tmp_path, embed=False)
    assert 'src="a.png"' in html
    assert "base64" not in html


def test_a_collapsed_section_renders_as_a_disclosure(tmp_path: Path) -> None:
    html = render(
        _doc(Section(slug="d", title="Diagnostics", blocks=(), collapsed=True)),
        base_dir=tmp_path,
    )
    assert "<details" in html and "<summary" in html


def test_titles_and_captions_are_escaped(tmp_path: Path) -> None:
    section = Section(slug="s", title="S", blocks=(Note(text="<script>alert(1)</script>"),))
    html = render(_doc(section), base_dir=tmp_path)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_key_values_render_as_labelled_pairs(tmp_path: Path) -> None:
    section = Section(
        slug="s",
        title="S",
        blocks=(KeyValues(title="Model", items=(("TR", "2.0 s"),)),),
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert "TR" in html and "2.0 s" in html


def test_a_table_offers_its_tsv_for_download(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("a\tb\n")
    section = Section(
        slug="s",
        title="S",
        blocks=(Table(title="Clusters", html="<table></table>", tsv_path=tsv),),
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert 'href="clusters.tsv"' in html


def test_a_table_above_the_report_directory_uses_a_portable_relative_link(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    tsv = tmp_path / "input_manifest.tsv"
    tsv.write_text("subject\nsub-01\n")
    section = Section(
        slug="s",
        title="S",
        blocks=(Table(title="Inputs", tsv_path=tsv),),
    )

    html = render(_doc(section), base_dir=report_dir)

    assert 'href="../input_manifest.tsv"' in html
    assert str(tmp_path) not in html


def test_a_missing_figure_file_fails_instead_of_silently_degrading(tmp_path: Path) -> None:
    section = Section(
        slug="s", title="S", blocks=(Figure(title="F", path=tmp_path / "absent.png"),)
    )
    with pytest.raises(FileNotFoundError):
        render(_doc(section), base_dir=tmp_path)


def test_the_document_is_self_contained_html(tmp_path: Path) -> None:
    html = render(_doc(), base_dir=tmp_path)
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert html.rstrip().endswith("</html>")


def test_a_wide_table_scrolls_inside_its_own_box(tmp_path: Path) -> None:
    # A wide cluster table must not widen the page body.
    section = Section(slug="s", title="S", blocks=(Table(title="T", html="<table></table>"),))
    html = render(_doc(section), base_dir=tmp_path)
    assert "overflow" in html


def test_numbers_use_tabular_figures_so_columns_align(tmp_path: Path) -> None:
    html = render(_doc(), base_dir=tmp_path)
    assert "tabular-nums" in html
