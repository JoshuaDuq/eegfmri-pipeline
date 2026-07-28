# fMRI Per-Subject Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-contrast `report.html` with one per-subject document, generated from derivatives rather than from inside the GLM path.

**Architecture:** The analysis pipeline's job ends at writing stat maps plus a `report_manifest.json` describing what was fit. A new `fmri-analysis report` mode reads a subject's manifests, computes the shared QC once, and renders one document covering every contrast. `html.py` supplies document primitives; `subject.py` assembles them. Nothing in the render path may fit a GLM.

**Tech Stack:** Python, nilearn, nibabel, numpy, pandas, pytest. No new dependencies.

Plan 2 of 3 from `docs/superpowers/specs/2026-07-28-fmri-post-preprocessing-report-design.md`. Plan 1 (`2026-07-28-fmri-plotting-foundation.md`) built the tested figure layer this consumes and is complete on `feat/fmri-plotting-foundation`. Plan 3 covers the resting-state profile.

## Global Constraints

- **No rendering setting may cause a GLM to be fit.** This is the invariant the whole plan exists to establish, and Task 8 tests it by importing the report path with the model-fitting modules absent.
- Subject-level QC (carpet, tSNR, coverage) is computed **once per subject-task**, never once per contrast.
- Every figure comes from `fmri_pipeline.analysis.report.figures`. This plan adds no new Matplotlib code outside those modules.
- The report presents measurements and the thresholds actually applied. No pass/fail badges, no cutoffs the pipeline invented, and no caption that lets cluster extent read as inference.
- A panel or section that fails renders a placeholder naming the failure; the document always builds.
- Tests use plain pytest with `nibabel.Nifti1Image` fixtures. Run targeted subsets — the full suite is slow and has been observed to exhaust memory on this machine.

---

## File Structure

| File | Responsibility |
|---|---|
| `fmri_pipeline/analysis/report/manifest.py` | The `ContrastManifest` record, its JSON round-trip, and discovery. |
| `fmri_pipeline/analysis/report/html.py` | `Document`/`Section`/`Figure`/`Table` primitives and the renderer. |
| `fmri_pipeline/analysis/report/subject.py` | Assembles a subject-task document from manifests. |
| `fmri_pipeline/analysis/report/figures/signatures.py` | Signature expression dot plot. |
| `fmri_pipeline/analysis/plotting_config.py` | Modified: splits into a stats config and a report config. |
| `fmri_pipeline/analysis/contrast_builder.py` | Modified: writes the manifest beside the stat maps. |
| `fmri_pipeline/pipelines/fmri_analysis.py` | Modified: stops calling into plotting. |
| `fmri_pipeline/cli/commands/fmri_analysis.py` | Modified: adds the `report` mode. |
| `fmri_pipeline/analysis/reporting.py` | Modified: `run_fmri_plotting_and_report` becomes a thin shim over `subject.py`, then is removed. |

---

### Task 1: The report manifest

**Files:**
- Create: `fmri_pipeline/analysis/report/manifest.py`
- Test: `tests/fmri/report/test_manifest.py`

**Interfaces:**
- Consumes: nothing from plan 1.
- Produces:
  - `@dataclass(frozen=True) class ContrastManifest` (fields below)
  - `write_manifest(manifest: ContrastManifest, path: Path) -> Path`
  - `read_manifest(path: Path) -> ContrastManifest`
  - `discover_manifests(*, deriv_root: Path, subject: str, task: str) -> list[ContrastManifest]`
  - `MANIFEST_FILENAME: str = "report_manifest.json"`

This is the seam that decouples rendering from fitting. Everything the report needs about a contrast is written here by the analysis run, so the renderer never re-derives it and never needs the model.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmri_pipeline.analysis.report.manifest import (
    ContrastManifest,
    discover_manifests,
    read_manifest,
    write_manifest,
)


def _manifest(**overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01",
        task="heat",
        contrast_name="heat-warm",
        space="native",
        stat_map=Path("/d/z.nii.gz"),
        effect_map=Path("/d/eff.nii.gz"),
        variance_map=None,
        mask=Path("/d/mask.nii.gz"),
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=10,
        two_sided=True,
        radiological=False,
        design_matrices=(Path("/d/run-01_dm.tsv"),),
        contrast_vector=(1.0, -1.0),
        contrast_columns=("heat", "warm"),
        included_runs=("run-01", "run-02"),
        excluded_runs=(("run-03", "no events file"),),
        bold_paths=(Path("/d/run-01_bold.nii.gz"),),
        confounds_paths=(Path("/d/run-01_conf.tsv"),),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=False,
        confound_strategy="motion+compcor",
    )
    base.update(overrides)
    return ContrastManifest(**base)


def test_a_manifest_round_trips_through_json(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "report_manifest.json")
    assert read_manifest(path) == _manifest()


def test_paths_survive_the_round_trip_as_paths(tmp_path: Path) -> None:
    restored = read_manifest(write_manifest(_manifest(), tmp_path / "m.json"))
    assert isinstance(restored.stat_map, Path)
    assert restored.variance_map is None


def test_exclusions_keep_their_reasons(tmp_path: Path) -> None:
    restored = read_manifest(write_manifest(_manifest(), tmp_path / "m.json"))
    assert restored.excluded_runs == (("run-03", "no events file"),)


def test_the_manifest_is_human_readable_json(tmp_path: Path) -> None:
    path = write_manifest(_manifest(), tmp_path / "m.json")
    payload = json.loads(path.read_text())
    assert payload["contrast_name"] == "heat-warm"
    assert payload["z_threshold"] == 2.3


def test_discovery_finds_every_contrast_of_one_subject_and_task(tmp_path: Path) -> None:
    root = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat"
    for name in ("heat-warm", "heat-rest"):
        directory = root / f"contrast-{name}"
        directory.mkdir(parents=True)
        write_manifest(
            _manifest(contrast_name=name), directory / "report_manifest.json"
        )

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert sorted(m.contrast_name for m in found) == ["heat-rest", "heat-warm"]


def test_discovery_ignores_another_task(tmp_path: Path) -> None:
    for task in ("heat", "rest"):
        directory = (
            tmp_path / "sub-01" / "fmri" / "first_level" / f"task-{task}" / "contrast-a"
        )
        directory.mkdir(parents=True)
        write_manifest(
            _manifest(task=task), directory / "report_manifest.json"
        )

    found = discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat")
    assert [m.task for m in found] == ["heat"]


def test_discovery_returns_empty_when_nothing_has_been_fit(tmp_path: Path) -> None:
    assert discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat") == []


def test_an_unreadable_manifest_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    directory = tmp_path / "sub-01" / "fmri" / "first_level" / "task-heat" / "contrast-a"
    directory.mkdir(parents=True)
    (directory / "report_manifest.json").write_text("{ not json")

    assert discover_manifests(deriv_root=tmp_path, subject="sub-01", task="heat") == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fmri_pipeline.analysis.report.manifest'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/manifest.py`:

```python
"""What one fitted contrast records for the report.

This is the seam that separates rendering from fitting. The analysis run writes a
manifest beside its stat maps; the report reads manifests and nothing else. Without
it the report can only be produced from inside the GLM path, which is why subject
QC was previously recomputed for every contrast and why iterating on a figure meant
re-running the model.

Everything here is plain JSON and plain paths. The report must be reproducible from
a derivatives tree alone, so a manifest may not carry live objects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "report_manifest.json"

#: Fields holding a single optional path.
_PATH_FIELDS = ("stat_map", "effect_map", "variance_map", "mask")
#: Fields holding a tuple of paths.
_PATH_TUPLE_FIELDS = ("design_matrices", "bold_paths", "confounds_paths")


@dataclass(frozen=True)
class ContrastManifest:
    """Everything the report needs about one fitted contrast."""

    subject: str
    task: str
    contrast_name: str
    space: str

    stat_map: Path
    effect_map: Optional[Path]
    variance_map: Optional[Path]
    mask: Optional[Path]

    threshold_mode: str
    z_threshold: float
    fdr_q: float
    cluster_min_voxels: int
    two_sided: bool
    radiological: bool

    design_matrices: Tuple[Path, ...]
    contrast_vector: Optional[Tuple[float, ...]]
    contrast_columns: Tuple[str, ...]

    included_runs: Tuple[str, ...]
    #: ``(run, reason)`` pairs. The reason is carried because a report that says a
    #: run was dropped without saying why cannot be acted on.
    excluded_runs: Tuple[Tuple[str, str], ...]

    bold_paths: Tuple[Path, ...]
    confounds_paths: Tuple[Path, ...]

    t_r: Optional[float]
    smoothing_fwhm: Optional[float]
    #: Whether the model applied signal scaling, which decides whether an effect
    #: size may be labelled "% signal change" or only "arbitrary BOLD units".
    signal_scaling: bool
    confound_strategy: str


def _encode(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def write_manifest(manifest: ContrastManifest, path: Path) -> Path:
    """Write ``manifest`` as indented JSON, and return the path written."""
    payload = {key: _encode(value) for key, value in asdict(manifest).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(path: Path) -> ContrastManifest:
    """Load a manifest, restoring paths and tuples."""
    payload: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {field.name for field in fields(ContrastManifest)}
    data = {key: value for key, value in payload.items() if key in known}

    for name in _PATH_FIELDS:
        if data.get(name) is not None:
            data[name] = Path(data[name])
    for name in _PATH_TUPLE_FIELDS:
        data[name] = tuple(Path(p) for p in data.get(name) or ())

    data["contrast_vector"] = (
        tuple(float(v) for v in data["contrast_vector"])
        if data.get("contrast_vector") is not None
        else None
    )
    data["contrast_columns"] = tuple(data.get("contrast_columns") or ())
    data["included_runs"] = tuple(data.get("included_runs") or ())
    data["excluded_runs"] = tuple(
        (str(run), str(reason)) for run, reason in (data.get("excluded_runs") or ())
    )
    return ContrastManifest(**data)


def discover_manifests(
    *,
    deriv_root: Path,
    subject: str,
    task: str,
) -> List[ContrastManifest]:
    """Return every contrast manifest for one subject and task.

    A manifest that cannot be read is skipped with a warning rather than aborting
    the report: one malformed contrast should cost its own section, not the
    document. Results are ordered by contrast name so a regenerated report has a
    stable section order.
    """
    root = Path(deriv_root) / subject / "fmri" / "first_level" / f"task-{task}"
    if not root.exists():
        return []

    found: List[ContrastManifest] = []
    for path in sorted(root.glob(f"contrast-*/{MANIFEST_FILENAME}")):
        try:
            manifest = read_manifest(path)
        except Exception as exc:
            logger.warning("Skipping unreadable manifest %s (%s)", path, exc)
            continue
        if manifest.task == task:
            found.append(manifest)
    return sorted(found, key=lambda m: m.contrast_name)


__all__ = [
    "MANIFEST_FILENAME",
    "ContrastManifest",
    "discover_manifests",
    "read_manifest",
    "write_manifest",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_manifest.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/manifest.py tests/fmri/report/test_manifest.py
git commit -m "feat(fmri): add the report manifest that decouples rendering from fitting"
```

---

### Task 2: Split the plotting config

**Files:**
- Modify: `fmri_pipeline/analysis/plotting_config.py`
- Modify: `fmri_pipeline/cli/commands/fmri_analysis.py` (flag destinations)
- Test: `tests/fmri/report/test_config_split.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class FmriStatsConfig` — `space`, `include_effect_size`, `include_standard_error`, `include_signatures`
  - `FmriReportConfig` — the rendering half, keeping every other current field
  - `MOVED_KEYS: dict[str, str]` mapping each moved key to its new location

**Why.** `include_effect_size` and `include_standard_error` currently drive `compute_contrast` calls at `fmri_analysis.py:343`, and `space` including `mni` triggers a complete second GLM fit at `fmri_analysis.py:312`. They are fields of the *plotting* config, so a rendering setting causes statistics to be computed. The split makes the cost visible where it is incurred.

Taken cleanly, with no deprecated aliases: an alias that silently accepts a rendering flag which fits a GLM preserves exactly the confusion the split removes. A config carrying a moved key fails with a message naming its new home.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_config_split.py`:

```python
from __future__ import annotations

import pytest

from fmri_pipeline.analysis.plotting_config import (
    FmriReportConfig,
    FmriStatsConfig,
    split_legacy_plotting_config,
)


def test_compute_triggering_fields_live_on_the_stats_config() -> None:
    stats = FmriStatsConfig(space="both", include_effect_size=True)
    assert stats.space == "both"
    assert stats.include_effect_size is True


def test_the_report_config_has_no_compute_triggering_fields() -> None:
    # A rendering setting must never be able to cause a GLM to be fit.
    names = set(FmriReportConfig.__dataclass_fields__)
    assert names.isdisjoint(
        {"space", "include_effect_size", "include_standard_error", "include_signatures"}
    )


def test_a_moved_key_fails_with_its_new_location_named() -> None:
    with pytest.raises(ValueError, match=r"fmri_stats\.space"):
        split_legacy_plotting_config({"space": "mni", "enabled": True})


def test_every_moved_key_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as excinfo:
        split_legacy_plotting_config(
            {"space": "mni", "include_effect_size": True, "enabled": True}
        )
    message = str(excinfo.value)
    assert "space" in message and "include_effect_size" in message


def test_a_clean_config_splits_without_complaint() -> None:
    report = split_legacy_plotting_config({"enabled": True, "z_threshold": 3.1})
    assert report.enabled is True
    assert report.z_threshold == 3.1


def test_the_report_config_still_validates_its_own_fields() -> None:
    with pytest.raises(ValueError, match="z-threshold"):
        FmriReportConfig(enabled=True, threshold_mode="z", z_threshold=-1).validate()


def test_radiological_is_a_rendering_setting() -> None:
    assert FmriReportConfig(enabled=True, radiological=True).radiological is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_config_split.py -v`
Expected: FAIL — `ImportError: cannot import name 'FmriReportConfig'`

- [ ] **Step 3: Implement the split**

In `fmri_pipeline/analysis/plotting_config.py`, keep the existing normalisation and validation helpers and add:

```python
#: Keys that moved off the plotting config, and where they went.
#:
#: Each of these caused statistics to be computed while living on a config named
#: for rendering. Deprecated aliases are deliberately not offered: an alias that
#: accepts a rendering flag which fits a GLM preserves the confusion the split
#: exists to remove.
MOVED_KEYS = {
    "space": "fmri_stats.space",
    "include_effect_size": "fmri_stats.include_effect_size",
    "include_standard_error": "fmri_stats.include_standard_error",
    "include_signatures": "fmri_stats.include_signatures",
}


@dataclass(frozen=True)
class FmriStatsConfig:
    """Settings that cause statistics to be computed.

    Separate from the report config because each of these costs a model fit or a
    contrast computation, and that cost should be visible where it is configured.
    """

    space: str = "native"
    include_effect_size: bool = True
    include_standard_error: bool = True
    include_signatures: bool = True

    def validate(self) -> None:
        if self.space not in _ALLOWED_SPACES:
            raise ValueError(
                f"fmri_stats.space must be one of {sorted(_ALLOWED_SPACES)}, "
                f"got '{self.space}'"
            )


@dataclass(frozen=True)
class FmriReportConfig:
    """Settings that only decide how existing results are drawn.

    Nothing here may cause a GLM to be fit. That invariant is what lets a report be
    regenerated from a derivatives tree without the model.
    """

    enabled: bool = False
    html_report: bool = False
    formats: Sequence[str] = field(default_factory=lambda: ("png",))
    threshold_mode: str = "z"
    z_threshold: float = 2.3
    fdr_q: float = 0.05
    cluster_min_voxels: int = 0
    two_sided: bool = True
    radiological: bool = False
    vmax_mode: str = "per_space_robust"
    vmax_manual: Optional[float] = None
    include_unthresholded: bool = True
    plot_types: Sequence[str] = field(
        default_factory=lambda: ("slices", "glass", "hist", "clusters")
    )
    include_motion_qc: bool = True
    include_carpet_qc: bool = True
    include_tsnr_qc: bool = True
    include_design_qc: bool = True
    embed_images: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.threshold_mode not in _ALLOWED_THRESHOLD_MODES:
            raise ValueError(
                f"threshold mode must be one of {sorted(_ALLOWED_THRESHOLD_MODES)}, "
                f"got '{self.threshold_mode}'"
            )
        if self.threshold_mode == "z" and self.z_threshold <= 0:
            raise ValueError("plot z-threshold must be > 0")
        if self.threshold_mode == "fdr" and not (0 < self.fdr_q <= 1):
            raise ValueError("plot FDR q must be in (0, 1]")
        if self.cluster_min_voxels < 0:
            raise ValueError("cluster_min_voxels must be >= 0")
        unknown_formats = sorted(set(_normalize_str_list(self.formats)) - _ALLOWED_FORMATS)
        if unknown_formats:
            raise ValueError(
                f"Unsupported plot format(s): {unknown_formats}. "
                f"Allowed: {sorted(_ALLOWED_FORMATS)}"
            )


def split_legacy_plotting_config(section: dict) -> FmriReportConfig:
    """Build a report config from a config section, rejecting moved keys.

    Fails rather than silently ignoring a moved key: a study whose YAML still sets
    ``plotting.space`` would otherwise get native-only output with no indication
    that its setting had stopped being read.
    """
    stale = sorted(key for key in MOVED_KEYS if key in section)
    if stale:
        moved = ", ".join(f"'{key}' -> {MOVED_KEYS[key]}" for key in stale)
        raise ValueError(
            f"These plotting keys moved to the fmri_stats section because they "
            f"cause statistics to be computed: {moved}. Update the config; there "
            f"are deliberately no aliases."
        )
    known = set(FmriReportConfig.__dataclass_fields__)
    return FmriReportConfig(**{k: v for k, v in section.items() if k in known})
```

Then update the CLI: move `--space`, `--plot-effect-size`, `--plot-standard-error`, and `--plot-signatures` out of the plotting argument group into a "Statistics" group, and change their `dest` to the `stats_*` names the pipeline reads.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_config_split.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Verify the existing config guards still pass**

Run: `.venv/bin/python -m pytest tests/fmri/ -q`
Expected: PASS. Update any guard that constructed `FmriPlottingConfig` with a moved key.

- [ ] **Step 6: Commit**

```bash
git add fmri_pipeline/analysis/plotting_config.py fmri_pipeline/cli/commands/fmri_analysis.py tests/fmri/
git commit -m "refactor(fmri): split compute-triggering settings off the plotting config"
```

---

### Task 3: Document primitives

**Files:**
- Create: `fmri_pipeline/analysis/report/html.py`
- Test: `tests/fmri/report/test_html.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Figure` — `title, path, caption="", dense=True`
  - `@dataclass(frozen=True) class Table` — `title, html="", tsv_path=None, caption=""`
  - `@dataclass(frozen=True) class KeyValues` — `title, items: tuple[tuple[str, str], ...]`
  - `@dataclass(frozen=True) class Note` — `text`
  - `@dataclass(frozen=True) class Section` — `slug, title, blocks, collapsed=False`
  - `@dataclass(frozen=True) class Document` — `title, subtitle, sections`
  - `render(document: Document, *, base_dir: Path, embed: bool = True) -> str`

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_html.py`:

```python
from __future__ import annotations

from pathlib import Path

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
    return Document(title="sub-01 · task-heat", subtitle="First-level report",
                    sections=tuple(sections))


def test_a_section_becomes_a_navigable_anchor(tmp_path: Path) -> None:
    html = render(
        _doc(Section(slug="qc", title="Quality control", blocks=())),
        base_dir=tmp_path,
    )
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
    section = Section(
        slug="s", title="S", blocks=(Figure(title="F", path=_png(tmp_path)),)
    )
    html = render(_doc(section), base_dir=tmp_path, embed=True)
    assert "data:image/png;base64," in html


def test_a_figure_is_linked_relatively_when_not_embedded(tmp_path: Path) -> None:
    section = Section(
        slug="s", title="S", blocks=(Figure(title="F", path=_png(tmp_path)),)
    )
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
    section = Section(
        slug="s", title="S", blocks=(Note(text="<script>alert(1)</script>"),)
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_key_values_render_as_labelled_pairs(tmp_path: Path) -> None:
    section = Section(
        slug="s", title="S",
        blocks=(KeyValues(title="Model", items=(("TR", "2.0 s"),)),),
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert "TR" in html and "2.0 s" in html


def test_a_table_offers_its_tsv_for_download(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("a\tb\n")
    section = Section(
        slug="s", title="S",
        blocks=(Table(title="Clusters", html="<table></table>", tsv_path=tsv),),
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert 'href="clusters.tsv"' in html


def test_a_missing_figure_file_renders_a_placeholder_not_a_crash(tmp_path: Path) -> None:
    section = Section(
        slug="s", title="S",
        blocks=(Figure(title="F", path=tmp_path / "absent.png"),),
    )
    html = render(_doc(section), base_dir=tmp_path)
    assert "could not be rendered" in html.lower()


def test_the_document_is_self_contained_html(tmp_path: Path) -> None:
    html = render(_doc(), base_dir=tmp_path)
    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert html.rstrip().endswith("</html>")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_html.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fmri_pipeline.analysis.report.html'`

- [ ] **Step 3: Implement the module**

`fmri_pipeline/analysis/report/html.py`:

```python
"""Document primitives for the subject report.

Blocks are data, not markup: a figure module returns a figure, the assembler
describes a document, and only :func:`render` knows HTML. That is what lets the
document's structure be tested without parsing markup, and what will let the group
report reuse the same primitives.

Styling follows the EEG report conventions -- tabular numerals so a column of
values lines up its decimal points, and no zebra striping, because the light-to-dark
neutral ramp is reserved for pipeline decisions.
"""

from __future__ import annotations

import base64
import html as html_escape
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Figure:
    title: str
    path: Path
    caption: str = ""
    #: Dense figures are raster and stay raster; the flag is carried so the
    #: assembler does not have to re-derive it from the suffix.
    dense: bool = True


@dataclass(frozen=True)
class Table:
    title: str
    html: str = ""
    tsv_path: Union[Path, None] = None
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
:root { --fg:#111; --muted:#555; --bg:#fff; --card:#f7f7f9; --border:#e6e6ea; }
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
    parts = ["<div class='fig'>", f"<div class='fig-title'>{_esc(figure.title)}</div>"]
    try:
        source = _image_source(figure.path, base_dir=base_dir, embed=embed)
        parts.append(f"<img src='{_esc(source)}' loading='lazy' alt='{_esc(figure.title)}' />")
    except OSError as exc:
        # A missing panel is a gap in the document, not a reason to lose it.
        logger.warning("Figure %s could not be read (%s)", figure.path, exc)
        parts.append(
            f"<div class='missing'>This figure could not be rendered: "
            f"{_esc(figure.path.name)}</div>"
        )
    if figure.caption:
        parts.append(f"<div class='fig-cap'>{_esc(figure.caption)}</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_table(table: Table, *, base_dir: Path) -> str:
    parts = ["<div class='fig'>", f"<div class='fig-title'>{_esc(table.title)}</div>"]
    if table.tsv_path is not None:
        link = _esc(_relpath(base_dir, table.tsv_path))
        parts.append(f"<div class='fig-cap'><a href='{link}'>Download TSV</a></div>")
    if table.html:
        # Wide tables scroll inside their own box rather than widening the page.
        parts.append(f"<div class='overflow'>{table.html}</div>")
    if table.caption:
        parts.append(f"<div class='fig-cap'>{_esc(table.caption)}</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_block(block: Block, *, base_dir: Path, embed: bool) -> str:
    if isinstance(block, Figure):
        return _render_figure(block, base_dir=base_dir, embed=embed)
    if isinstance(block, Table):
        return _render_table(block, base_dir=base_dir)
    if isinstance(block, KeyValues):
        rows = "".join(
            f"<div class='k'>{_esc(k)}</div><div>{_esc(v)}</div>" for k, v in block.items
        )
        return (
            f"<div class='fig'><div class='fig-title'>{_esc(block.title)}</div>"
            f"<div class='kvs'>{rows}</div></div>"
        )
    return f"<p class='fig-cap'>{_esc(block.text)}</p>"


def render(document: Document, *, base_dir: Path, embed: bool = True) -> str:
    """Render ``document`` as one self-contained HTML page."""
    base_dir = Path(base_dir)
    toc = "".join(
        f"<li><a class='toc-link' href='#{_esc(s.slug)}'>{_esc(s.title)}</a></li>"
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
            f"<section id='{_esc(section.slug)}'>"
            f"<h2>{_esc(section.title)}</h2>{inner}</section>"
        )

    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8' />"
        "<meta name='viewport' content='width=device-width, initial-scale=1' />"
        f"<title>{_esc(document.title)}</title><style>{_CSS}</style></head>"
        "<body><div class='layout'>"
        f"<nav><ol>{toc}</ol></nav>"
        f"<main><h1>{_esc(document.title)}</h1>"
        f"<p class='subhead'>{_esc(document.subtitle)}</p>"
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_html.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/html.py tests/fmri/report/test_html.py
git commit -m "feat(fmri): add document primitives for the subject report"
```

---

### Task 4: Shared sections — header, QC, methods

**Files:**
- Create: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_subject_shared.py`

**Interfaces:**
- Consumes: `ContrastManifest` (Task 1), `html` primitives (Task 3), `FmriReportConfig` (Task 2), and from plan 1: `carpet_figures`, `volume_figures`, `coverage_figures`, `distribution_figures`, `assets.discover_plot_assets`.
- Produces:
  - `build_header_section(manifests: Sequence[ContrastManifest]) -> Section`
  - `build_qc_sections(*, manifests, deriv_root, out_dir, cfg) -> list[Section]`
  - `build_methods_section(manifests: Sequence[ContrastManifest]) -> Section`

QC is built from the **first** manifest's run list, because every contrast of a subject-task shares it. That is the whole point: computed once, not once per contrast.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_subject_shared.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import ContrastManifest


def _bold(tmp_path: Path, name: str, n_frames: int = 20) -> Path:
    rng = np.random.default_rng(0)
    data = (100.0 + rng.standard_normal((6, 6, 6, n_frames))).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _manifest(tmp_path: Path, name: str = "a", **overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01", task="heat", contrast_name=name, space="native",
        stat_map=tmp_path / "z.nii.gz", effect_map=None, variance_map=None, mask=None,
        threshold_mode="z", z_threshold=2.3, fdr_q=0.05, cluster_min_voxels=0,
        two_sided=True, radiological=False,
        design_matrices=(), contrast_vector=None, contrast_columns=(),
        included_runs=("run-01", "run-02"),
        excluded_runs=(("run-03", "fewer events than the contrast requires"),),
        bold_paths=(_bold(tmp_path, "r1.nii.gz"), _bold(tmp_path, "r2.nii.gz")),
        confounds_paths=(), t_r=2.0, smoothing_fwhm=6.0,
        signal_scaling=False, confound_strategy="motion+compcor",
    )
    base.update(overrides)
    return ContrastManifest(**base)


def test_the_header_names_every_excluded_run_and_its_reason(tmp_path: Path) -> None:
    section = subject.build_header_section([_manifest(tmp_path)])
    text = str(section)
    assert "run-03" in text
    assert "fewer events" in text


def test_the_header_states_the_acquisition_parameters(tmp_path: Path) -> None:
    section = subject.build_header_section([_manifest(tmp_path)])
    text = str(section)
    assert "2.0" in text  # TR
    assert "6.0" in text  # smoothing FWHM


def test_qc_is_built_once_for_a_subject_with_several_contrasts(tmp_path: Path) -> None:
    manifests = [_manifest(tmp_path, "a"), _manifest(tmp_path, "b")]
    with patch(
        "fmri_pipeline.analysis.report.figures.volumes.compute_tsnr"
    ) as mock_tsnr:
        mock_tsnr.side_effect = RuntimeError("stop here")
        subject.build_qc_sections(
            manifests=manifests, deriv_root=tmp_path, out_dir=tmp_path,
            cfg=FmriReportConfig(enabled=True),
        )
    # Two contrasts, one tSNR computation.
    assert mock_tsnr.call_count == 1


def test_qc_returns_a_section_even_when_every_panel_fails(tmp_path: Path) -> None:
    with patch(
        "fmri_pipeline.analysis.report.figures.volumes.compute_tsnr",
        side_effect=RuntimeError("boom"),
    ), patch(
        "fmri_pipeline.analysis.report.figures.carpet.carpet_figure",
        side_effect=RuntimeError("boom"),
    ):
        sections = subject.build_qc_sections(
            manifests=[_manifest(tmp_path)], deriv_root=tmp_path, out_dir=tmp_path,
            cfg=FmriReportConfig(enabled=True),
        )
    assert sections  # the section exists; it is simply thin


def test_qc_is_labelled_as_modelled_not_as_preprocessed(tmp_path: Path) -> None:
    sections = subject.build_qc_sections(
        manifests=[_manifest(tmp_path)], deriv_root=tmp_path, out_dir=tmp_path,
        cfg=FmriReportConfig(enabled=True),
    )
    text = " ".join(str(s) for s in sections).lower()
    assert "as modelled" in text


def test_methods_records_the_confound_strategy(tmp_path: Path) -> None:
    section = subject.build_methods_section([_manifest(tmp_path)])
    assert "motion+compcor" in str(section)


def test_methods_states_the_threshold_actually_applied(tmp_path: Path) -> None:
    section = subject.build_methods_section([_manifest(tmp_path)])
    assert "2.3" in str(section)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject_shared.py -v`
Expected: FAIL — `ImportError: cannot import name 'subject'`

- [ ] **Step 3: Implement the shared sections**

`fmri_pipeline/analysis/report/subject.py` (first half; Task 5 adds the results half):

```python
"""Assembles one subject-task report from contrast manifests.

Everything here reads derivatives. Nothing fits a model: that separation is what
lets QC be computed once per subject rather than once per contrast, and what lets a
figure be reworked without re-running a GLM.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import html
from fmri_pipeline.analysis.report.assets import discover_plot_assets
from fmri_pipeline.analysis.report.figures import carpet as carpet_figures
from fmri_pipeline.analysis.report.figures import coverage as coverage_figures
from fmri_pipeline.analysis.report.figures import distributions as distribution_figures
from fmri_pipeline.analysis.report.figures import volumes as volume_figures
from fmri_pipeline.analysis.report.manifest import ContrastManifest
from fmri_pipeline.analysis.report.style import plot_context, savefig_kwargs

logger = logging.getLogger(__name__)


@contextmanager
def _panel(description: str) -> Iterator[None]:
    """Log and swallow one panel's failure so the document still builds."""
    try:
        yield
    except Exception as exc:
        logger.warning("Failed to generate %s (%s)", description, exc)


def _save(figure: Any, *, out_dir: Path, stem: str, formats: Sequence[str]) -> Optional[Path]:
    """Write a figure and return the path the report should embed."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary: Optional[Path] = None
        with plot_context():
            for fmt in formats:
                path = out_dir / f"{stem}.{fmt}"
                figure.savefig(path, **savefig_kwargs(path))
                if primary is None:
                    primary = path
        return primary
    finally:
        try:
            plt.close(figure)
        except Exception:
            logger.debug("Could not close figure %s", stem)


def build_header_section(manifests: Sequence[ContrastManifest]) -> html.Section:
    """Summarise the acquisition and what entered the model.

    Excluded runs carry their reasons: a report that says a run was dropped without
    saying why gives a reader nothing to act on.
    """
    first = manifests[0]
    items = [
        ("Subject", first.subject),
        ("Task", first.task),
        ("Contrasts", str(len(manifests))),
        ("Runs included", ", ".join(first.included_runs) or "none"),
        ("TR", f"{first.t_r:.3g} s" if first.t_r else "unknown"),
        (
            "Smoothing",
            f"{first.smoothing_fwhm:.3g} mm FWHM" if first.smoothing_fwhm else "none",
        ),
        ("Confound strategy", first.confound_strategy or "unspecified"),
        (
            "Effect units",
            "% signal change" if first.signal_scaling else "arbitrary BOLD units",
        ),
    ]
    blocks: List[html.Block] = [html.KeyValues(title="Acquisition and model", items=tuple(items))]
    if first.excluded_runs:
        blocks.append(
            html.KeyValues(
                title="Runs excluded",
                items=tuple((run, reason) for run, reason in first.excluded_runs),
            )
        )
    return html.Section(slug="overview", title="Overview", blocks=tuple(blocks))


def build_qc_sections(
    *,
    manifests: Sequence[ContrastManifest],
    deriv_root: Path,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> List[html.Section]:
    """Build the as-modelled QC section, once for the whole subject-task.

    Driven by the first manifest's run list because every contrast of a subject-task
    shares it. Recomputing per contrast produced byte-identical figures and read the
    entire 4D dataset once per contrast.

    "As modelled" rather than "as preprocessed": these panels describe the runs that
    entered the GLM, after confound selection and smoothing. fMRIPrep's own report
    covers the preprocessing, and this document does not restate it.
    """
    import nibabel as nib

    first = manifests[0]
    qc_dir = out_dir / "plots" / "qc"
    blocks: List[html.Block] = []

    bold_imgs = []
    for path in first.bold_paths:
        if Path(path).exists():
            bold_imgs.append(nib.load(str(path)))

    sample_masks = None
    if first.confounds_paths:
        with _panel("censoring masks"):
            from fmri_pipeline.analysis.reporting import _sample_masks_from_confounds

            candidate = _sample_masks_from_confounds(first.confounds_paths)
            if len(candidate) == len(bold_imgs):
                sample_masks = candidate

    if bold_imgs and cfg.include_tsnr_qc:
        with _panel("tSNR"):
            result = volume_figures.compute_tsnr(bold_imgs, sample_masks=sample_masks)
            path = _save(
                volume_figures.tsnr_volume(result, title="tSNR (as modelled)"),
                out_dir=qc_dir, stem="tsnr_map", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="tSNR", path=path))
            if len(result.per_run_median) > 1:
                path = _save(
                    volume_figures.per_run_tsnr_figure(
                        result, run_labels=first.included_runs, title="tSNR by run"
                    ),
                    out_dir=qc_dir, stem="tsnr_by_run", formats=cfg.formats,
                )
                if path:
                    blocks.append(html.Figure(
                        title="tSNR by run", path=path, dense=False,
                        caption="Shown per run because averaging maps hides one bad run.",
                    ))

    if first.mask and Path(first.mask).exists():
        with _panel("coverage"):
            path = _save(
                coverage_figures.coverage_figure(
                    nib.load(str(first.mask)),
                    n_runs=len(first.included_runs),
                    title="Analysis mask",
                ),
                out_dir=qc_dir, stem="coverage", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(
                    title="Coverage", path=path,
                    caption="Voxels outside this mask were not tested.",
                ))

    return [
        html.Section(
            slug="qc",
            title="Quality control (as modelled)",
            blocks=tuple(blocks) or (html.Note(text="No QC panels could be generated."),),
        )
    ]


def build_methods_section(manifests: Sequence[ContrastManifest]) -> html.Section:
    """Record the thresholds and settings actually applied.

    Values and settings only. No verdicts: the report states what was done and what
    was measured, and leaves the judgement to the reader.
    """
    first = manifests[0]
    threshold = (
        f"|z| > {first.z_threshold:.2f}"
        if first.threshold_mode == "z"
        else f"FDR q = {first.fdr_q:.3f}"
        if first.threshold_mode == "fdr"
        else "none"
    )
    items = [
        ("Height threshold", threshold),
        ("Sidedness", "two-sided" if first.two_sided else "one-sided"),
        ("Confound strategy", first.confound_strategy or "unspecified"),
        (
            "Orientation",
            "radiological (R on left)" if first.radiological else "neurological (L on left)",
        ),
    ]
    if first.cluster_min_voxels > 0:
        items.append((
            "Cluster extent filter",
            f"clusters smaller than {first.cluster_min_voxels} voxels removed for "
            "display; this is not familywise-error-corrected cluster-level inference",
        ))
    return html.Section(
        slug="methods", title="Methods",
        blocks=(html.KeyValues(title="Applied settings", items=tuple(items)),),
    )


__all__ = ["build_header_section", "build_methods_section", "build_qc_sections"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject_shared.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject_shared.py
git commit -m "feat(fmri): build subject-level QC once instead of once per contrast"
```

---

### Task 5: Per-contrast results sections

**Files:**
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_subject_results.py`

**Interfaces:**
- Produces:
  - `build_contrast_section(*, manifest, out_dir, cfg) -> Section`
  - `build_diagnostics_section(*, manifest, out_dir, cfg) -> Section` (`collapsed=True`)
  - `build_subject_report(*, manifests, deriv_root, out_path, cfg) -> Path`

Each contrast gets a results section leading with the dual-coded panel and the thresholded mosaic, plus a collapsed diagnostics section holding the unthresholded mosaic, effect map, standard error, and z histogram. The demotion is the point: the unthresholded map must stay available without competing with the result.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_subject_results.py`:

```python
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.plotting_config import FmriReportConfig
from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import ContrastManifest


def _img(tmp_path: Path, name: str, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path = tmp_path / name
    nib.save(
        nib.Nifti1Image(rng.standard_normal((12, 12, 12)).astype(np.float32), np.eye(4)),
        str(path),
    )
    return path


def _manifest(tmp_path: Path, name: str = "heat-warm", **overrides) -> ContrastManifest:
    base = dict(
        subject="sub-01", task="heat", contrast_name=name, space="native",
        stat_map=_img(tmp_path, f"{name}_z.nii.gz"),
        effect_map=_img(tmp_path, f"{name}_eff.nii.gz", 1),
        variance_map=None, mask=None,
        threshold_mode="z", z_threshold=2.3, fdr_q=0.05, cluster_min_voxels=0,
        two_sided=True, radiological=False,
        design_matrices=(), contrast_vector=None, contrast_columns=(),
        included_runs=("run-01",), excluded_runs=(), bold_paths=(), confounds_paths=(),
        t_r=2.0, smoothing_fwhm=None, signal_scaling=False, confound_strategy="motion",
    )
    base.update(overrides)
    return ContrastManifest(**base)


def test_a_contrast_section_leads_with_the_dual_coded_panel(tmp_path: Path) -> None:
    section = subject.build_contrast_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path,
        cfg=FmriReportConfig(enabled=True),
    )
    titles = [b.title for b in section.blocks if hasattr(b, "title")]
    assert titles and "dual-coded" in titles[0].lower()


def test_diagnostics_are_collapsed_not_deleted(tmp_path: Path) -> None:
    section = subject.build_diagnostics_section(
        manifest=_manifest(tmp_path), out_dir=tmp_path,
        cfg=FmriReportConfig(enabled=True, include_unthresholded=True),
    )
    assert section.collapsed is True
    titles = " ".join(b.title for b in section.blocks if hasattr(b, "title")).lower()
    assert "unthresholded" in titles


def test_each_contrast_gets_its_own_anchor(tmp_path: Path) -> None:
    a = subject.build_contrast_section(
        manifest=_manifest(tmp_path, "heat-warm"), out_dir=tmp_path,
        cfg=FmriReportConfig(enabled=True),
    )
    b = subject.build_contrast_section(
        manifest=_manifest(tmp_path, "heat-rest"), out_dir=tmp_path,
        cfg=FmriReportConfig(enabled=True),
    )
    assert a.slug != b.slug


def test_the_document_covers_every_contrast(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path, "heat-warm"), _manifest(tmp_path, "heat-rest")],
        deriv_root=tmp_path, out_path=out, cfg=FmriReportConfig(enabled=True),
    )
    text = out.read_text()
    assert "heat-warm" in text and "heat-rest" in text


def test_the_document_has_one_qc_section_regardless_of_contrast_count(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.html"
    subject.build_subject_report(
        manifests=[_manifest(tmp_path, "a"), _manifest(tmp_path, "b")],
        deriv_root=tmp_path, out_path=out, cfg=FmriReportConfig(enabled=True),
    )
    assert out.read_text().count('id="qc"') == 1


def test_building_a_report_with_no_contrasts_raises_clearly(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="no contrasts"):
        subject.build_subject_report(
            manifests=[], deriv_root=tmp_path, out_path=tmp_path / "r.html",
            cfg=FmriReportConfig(enabled=True),
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject_results.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_contrast_section'`

- [ ] **Step 3: Implement the results half**

Append to `fmri_pipeline/analysis/report/subject.py`:

```python
def _slug(manifest: ContrastManifest) -> str:
    return "contrast-" + "".join(
        ch if ch.isalnum() or ch == "-" else "-" for ch in manifest.contrast_name
    ).strip("-").lower()


def build_contrast_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> html.Section:
    """Build the results section for one contrast.

    Leads with the dual-coded panel, which shows the whole map, then the
    hard-thresholded panel the cluster table refers to. Both are needed: the first
    so a reader can see near-threshold structure, the second so the figure and the
    table describe the same voxels.
    """
    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import stat_maps as stat_map_figures

    plots_dir = out_dir / "plots" / _slug(manifest)
    stat_img = nib.load(str(manifest.stat_map))
    threshold = manifest.z_threshold if manifest.threshold_mode == "z" else None
    blocks: List[html.Block] = []

    if manifest.effect_map and Path(manifest.effect_map).exists() and threshold:
        with _panel(f"dual-coded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.dual_coded_mosaic(
                    nib.load(str(manifest.effect_map)),
                    stat_img=stat_img,
                    threshold=float(threshold),
                    two_sided=manifest.two_sided,
                    radiological=manifest.radiological,
                    cbar_label=(
                        "% signal change" if manifest.signal_scaling
                        else "effect (arbitrary BOLD units)"
                    ),
                    title=f"{manifest.contrast_name}: effect, opacity-coded by evidence",
                ),
                out_dir=plots_dir, stem="dual_coded", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(
                    title="Effect map · dual-coded", path=path,
                    caption=(
                        "Colour is effect magnitude; opacity is statistical evidence. "
                        "No voxels are hidden."
                    ),
                ))

    if threshold:
        with _panel(f"thresholded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.stat_map_mosaic(
                    stat_img, threshold=float(threshold),
                    two_sided=manifest.two_sided, radiological=manifest.radiological,
                    title=f"{manifest.contrast_name}: z map (thresholded)",
                ),
                out_dir=plots_dir, stem="stat_thresholded", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="Stat map · thresholded", path=path))

        with _panel(f"glass brain for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.glass_brain(
                    stat_img, threshold=float(threshold),
                    two_sided=manifest.two_sided, radiological=manifest.radiological,
                    title=f"{manifest.contrast_name}: glass brain",
                ),
                out_dir=plots_dir, stem="glass", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="Glass brain · thresholded", path=path))

    if not blocks:
        blocks.append(html.Note(text="No panels could be generated for this contrast."))
    return html.Section(
        slug=_slug(manifest),
        title=f"Contrast: {manifest.contrast_name}",
        blocks=tuple(blocks),
    )


def build_diagnostics_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
) -> html.Section:
    """Build the collapsed diagnostics for one contrast.

    Demoted, not deleted. The unthresholded map is the honest counterpart to the
    thresholded one and the standard error is how a reader tells a true null from a
    dropout-driven absence of effect -- but neither should compete with the result.
    """
    import nibabel as nib

    from fmri_pipeline.analysis.report.figures import stat_maps as stat_map_figures

    plots_dir = out_dir / "plots" / _slug(manifest)
    stat_img = nib.load(str(manifest.stat_map))
    blocks: List[html.Block] = []

    if cfg.include_unthresholded:
        with _panel(f"unthresholded panel for {manifest.contrast_name}"):
            path = _save(
                stat_map_figures.stat_map_mosaic(
                    stat_img, threshold=None,
                    two_sided=manifest.two_sided, radiological=manifest.radiological,
                    title=f"{manifest.contrast_name}: z map (unthresholded)",
                ),
                out_dir=plots_dir, stem="stat_unthresholded", formats=cfg.formats,
            )
            if path:
                blocks.append(html.Figure(title="Stat map · unthresholded", path=path))

    with _panel(f"z histogram for {manifest.contrast_name}"):
        data = np.asarray(stat_img.get_fdata())
        path = _save(
            distribution_figures.z_histogram(
                data[np.isfinite(data)],
                threshold=manifest.z_threshold if manifest.threshold_mode == "z" else None,
                title="Z-statistic distribution",
            ),
            out_dir=plots_dir, stem="z_hist", formats=cfg.formats,
        )
        if path:
            blocks.append(html.Figure(title="Z histogram", path=path, dense=False))

    return html.Section(
        slug=f"{_slug(manifest)}-diagnostics",
        title=f"Diagnostics: {manifest.contrast_name}",
        blocks=tuple(blocks),
        collapsed=True,
    )


def build_subject_report(
    *,
    manifests: Sequence[ContrastManifest],
    deriv_root: Path,
    out_path: Path,
    cfg: FmriReportConfig,
) -> Path:
    """Render one document covering every contrast of a subject and task."""
    if not manifests:
        raise ValueError("Cannot build a subject report with no contrasts.")

    first = manifests[0]
    out_dir = Path(out_path).parent
    sections = [build_header_section(manifests)]
    sections.extend(
        build_qc_sections(
            manifests=manifests, deriv_root=Path(deriv_root), out_dir=out_dir, cfg=cfg
        )
    )
    for manifest in manifests:
        sections.append(
            build_contrast_section(manifest=manifest, out_dir=out_dir, cfg=cfg)
        )
        sections.append(
            build_diagnostics_section(manifest=manifest, out_dir=out_dir, cfg=cfg)
        )
    sections.append(build_methods_section(manifests))

    document = html.Document(
        title=f"{first.subject} · task-{first.task}",
        subtitle="First-level GLM report (post-fMRIPrep)",
        sections=tuple(sections),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        html.render(document, base_dir=out_dir, embed=cfg.embed_images),
        encoding="utf-8",
    )
    return out_path
```

Extend `__all__` with the three new names.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject_results.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject_results.py
git commit -m "feat(fmri): render one document per subject covering every contrast"
```

---

### Task 6: Numbered cluster peaks

**Files:**
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_cluster_peaks.py`

**Interfaces:**
- Produces: `build_cluster_table(*, manifest, out_dir) -> tuple[Table | None, tuple[tuple[float, float, float], ...]]`

The peak coordinates returned here are passed to `glass_brain(peak_coords=...)` from plan 1, so the numbered markers on the projection key to the numbered rows in the table. Without that, a reader has coordinates in one place and a picture in another and has to do the matching by hand.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_cluster_peaks.py`:

```python
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_pipeline.analysis.report import subject
from fmri_pipeline.analysis.report.manifest import ContrastManifest


def _blob_manifest(tmp_path: Path) -> ContrastManifest:
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[8:12, 8:12, 8:12] = 6.0
    path = tmp_path / "z.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return ContrastManifest(
        subject="sub-01", task="heat", contrast_name="a", space="native",
        stat_map=path, effect_map=None, variance_map=None, mask=None,
        threshold_mode="z", z_threshold=2.3, fdr_q=0.05, cluster_min_voxels=0,
        two_sided=True, radiological=False,
        design_matrices=(), contrast_vector=None, contrast_columns=(),
        included_runs=("run-01",), excluded_runs=(), bold_paths=(), confounds_paths=(),
        t_r=2.0, smoothing_fwhm=None, signal_scaling=False, confound_strategy="motion",
    )


def test_a_cluster_table_is_produced_with_peak_coordinates(tmp_path: Path) -> None:
    table, peaks = subject.build_cluster_table(
        manifest=_blob_manifest(tmp_path), out_dir=tmp_path
    )
    assert table is not None
    assert len(peaks) >= 1
    assert len(peaks[0]) == 3


def test_the_table_caption_separates_threshold_from_extent(tmp_path: Path) -> None:
    manifest = _blob_manifest(tmp_path)
    table, _ = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    caption = table.caption.lower()
    assert "height threshold" in caption
    assert "cluster-level significan" not in caption


def test_an_extent_filter_is_named_as_a_display_filter(tmp_path: Path) -> None:
    manifest = _blob_manifest(tmp_path)
    manifest = ContrastManifest(**{**manifest.__dict__, "cluster_min_voxels": 5})
    table, _ = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert "not familywise-error-corrected" in table.caption.lower()


def test_the_tsv_is_written_beside_the_report(tmp_path: Path) -> None:
    subject.build_cluster_table(manifest=_blob_manifest(tmp_path), out_dir=tmp_path)
    assert (tmp_path / "plots" / "contrast-a" / "clusters.tsv").exists()


def test_an_empty_map_yields_no_table_rather_than_an_error(tmp_path: Path) -> None:
    flat = tmp_path / "flat.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((12, 12, 12), dtype=np.float32), np.eye(4)), str(flat))
    manifest = _blob_manifest(tmp_path)
    manifest = ContrastManifest(**{**manifest.__dict__, "stat_map": flat})
    table, peaks = subject.build_cluster_table(manifest=manifest, out_dir=tmp_path)
    assert peaks == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_cluster_peaks.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_cluster_table'`

- [ ] **Step 3: Implement it**

Append to `subject.py`:

```python
def build_cluster_table(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
) -> tuple[Optional[html.Table], tuple]:
    """Return the cluster table and its peak coordinates.

    The peaks are handed to the glass brain so the numbered markers on the
    projection key to the numbered rows in the table. Coordinates in one place and a
    picture in another leaves the matching to the reader.

    The caption states the height threshold and any extent filter as two separate
    facts. This pipeline performs no cluster-level familywise correction, and Eklund,
    Nichols & Knutsson (2016) measured false-positive rates up to 70% for parametric
    cluster inference, so nothing here may read as inferential about extent.
    """
    import nibabel as nib

    try:
        from nilearn import reporting
    except ImportError:
        return None, ()

    threshold = manifest.z_threshold if manifest.threshold_mode == "z" else None
    if threshold is None:
        return None, ()

    plots_dir = out_dir / "plots" / _slug(manifest)
    frame = reporting.get_clusters_table(
        nib.load(str(manifest.stat_map)),
        stat_threshold=float(threshold),
        cluster_threshold=manifest.cluster_min_voxels or 0,
        two_sided=manifest.two_sided,
    )

    peaks = tuple(
        (float(row["X"]), float(row["Y"]), float(row["Z"]))
        for _, row in frame.iterrows()
        if {"X", "Y", "Z"} <= set(frame.columns)
    )

    plots_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = plots_dir / "clusters.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)

    caption_parts = [
        "two-sided" if manifest.two_sided else "one-sided",
        f"height threshold: |z| > {threshold:.2f}",
    ]
    if manifest.cluster_min_voxels > 0:
        caption_parts.append(
            f"clusters smaller than {manifest.cluster_min_voxels} voxels removed for "
            "display; this is an extent filter, not familywise-error-corrected "
            "cluster-level inference"
        )

    return (
        html.Table(
            title="Clusters and peaks",
            html=frame.to_html(index=False, border=0, classes=""),
            tsv_path=tsv_path,
            caption="; ".join(caption_parts),
        ),
        peaks,
    )
```

Then in `build_contrast_section`, call `build_cluster_table` before the glass brain, pass `peak_coords=peaks` to `glass_brain`, and append the table to `blocks`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_cluster_peaks.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_cluster_peaks.py
git commit -m "feat(fmri): key numbered glass-brain peaks to the cluster table"
```

---

### Task 7: Signature dot plot and design section

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/signatures.py`
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_signatures_figure.py`

**Interfaces:**
- Produces:
  - `signature_expression_figure(results, *, metric="cosine", title="") -> Figure`
  - `build_design_section(*, manifest, out_dir, cfg) -> Section | None`

Signature expression is currently a five-column table. The comparison *across* signatures is the point, and a table does not show it. The design section carries the design matrix with its contrast strip and the collinearity panel, both from plan 1's `design.py`.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_signatures_figure.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from fmri_pipeline.analysis.multivariate_signatures import SignatureResult
from fmri_pipeline.analysis.report.figures import signatures


def _results():
    return [
        SignatureResult(name="NPS", dot=1.2, cosine=0.41, pearson_r=0.39,
                        n_voxels=100, weight_path="/w/nps.nii.gz"),
        SignatureResult(name="SIIPS", dot=-0.4, cosine=-0.12, pearson_r=-0.10,
                        n_voxels=100, weight_path="/w/siips.nii.gz"),
    ]


def test_every_signature_is_named_on_the_axis() -> None:
    figure = signatures.signature_expression_figure(_results())
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert "NPS" in labels and "SIIPS" in labels
    plt.close(figure)


def test_zero_is_marked_because_the_metric_is_signed() -> None:
    figure = signatures.signature_expression_figure(_results())
    positions = [
        line.get_xdata()[0]
        for line in figure.axes[0].lines
        if line.get_linestyle() in {"--", ":"}
    ]
    assert any(abs(p) < 1e-9 for p in positions)
    plt.close(figure)


def test_the_axis_names_the_metric_shown() -> None:
    figure = signatures.signature_expression_figure(_results(), metric="cosine")
    assert "cosine" in figure.axes[0].get_xlabel().lower()
    plt.close(figure)


def test_an_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="metric"):
        signatures.signature_expression_figure(_results(), metric="nonsense")


def test_an_empty_result_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        signatures.signature_expression_figure([])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_signatures_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'signatures'`

- [ ] **Step 3: Implement the figure**

`fmri_pipeline/analysis/report/figures/signatures.py`:

```python
"""Multivariate signature expression, drawn rather than tabulated.

Expression was reported as a five-column table. The question a reader has is how
the signatures compare against each other and against zero, and a table makes that
a manual scan across rows.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import GUIDE_COLOR, OKABE_ITO, plot_context

#: Metrics worth plotting, with the axis label each one earns.
_METRICS = {
    "cosine": "Cosine similarity with the signature",
    "pearson_r": "Pearson r with the signature",
    "dot": "Dot product (raw pattern expression)",
}


def signature_expression_figure(
    results: Sequence,
    *,
    metric: str = "cosine",
    title: str = "",
) -> plt.Figure:
    """Draw signature expression as a dot plot against zero.

    Zero is marked because every one of these metrics is signed, and the sign is the
    first thing a reader needs. Scale-invariant metrics are the default: the raw dot
    product depends on the map's units and is not comparable across signatures.
    """
    if metric not in _METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(_METRICS)}."
        )
    if not results:
        raise ValueError("Signature expression requires at least one result.")

    names = [r.name for r in results]
    values = np.array(
        [float(getattr(r, metric)) if getattr(r, metric) is not None else np.nan
         for r in results]
    )
    positions = np.arange(len(names))

    with plot_context():
        figure, axis = plt.subplots(figsize=(6.5, 0.42 * len(names) + 1.6))
        axis.axvline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
        axis.hlines(positions, 0.0, values, color="0.75", linewidth=1.0)
        axis.scatter(values, positions, s=42, color=OKABE_ITO["blue"], zorder=3)
        axis.set_yticks(positions)
        axis.set_yticklabels(names, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel(_METRICS[metric])
        if title:
            axis.set_title(title)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.annotate(
                    f"{value:.3f}", xy=(value, position), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7,
                )
        figure.tight_layout()
        return figure


__all__ = ["signature_expression_figure"]
```

Then append `build_design_section` to `subject.py`, reading each design matrix TSV from `manifest.design_matrices` and drawing it with `design_figures.design_matrix_figure(..., contrast=manifest.contrast_vector)` plus `design_figures.collinearity_figure(...)`. Return `None` when the manifest lists no design matrices.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_signatures_figure.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/signatures.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_signatures_figure.py
git commit -m "feat(fmri): draw signature expression and the design section"
```

---

### Task 8: Wire the entry point and cut the pipeline's plotting call

**Files:**
- Modify: `fmri_pipeline/cli/commands/fmri_analysis.py` — add the `report` mode
- Modify: `fmri_pipeline/analysis/contrast_builder.py` — write the manifest
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py` — stop calling `run_fmri_plotting_and_report`
- Modify: `fmri_pipeline/analysis/reporting.py` — delete the orchestration that moved
- Test: `tests/fmri/report/test_report_entry_point.py`

**Interfaces:**
- Produces: `fmri-analysis report --subject … --task …`, which discovers manifests and calls `build_subject_report`.

This task establishes the plan's central invariant, and its first test is the one that proves it.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_report_entry_point.py`:

```python
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_rendering_a_report_never_imports_the_model_fitting_modules() -> None:
    """The invariant this whole plan exists to establish.

    If the report path pulls in contrast_builder or nilearn's GLM, then rendering
    can still trigger fitting, and the decoupling is nominal.
    """
    script = textwrap.dedent(
        """
        import sys
        import fmri_pipeline.analysis.report.subject  # noqa: F401

        forbidden = [
            name for name in sys.modules
            if "contrast_builder" in name or name.startswith("nilearn.glm")
        ]
        assert not forbidden, forbidden
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_report_mode_is_registered() -> None:
    from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    analysis = setup_fmri_analysis(sub)
    mode_action = next(
        a for a in analysis._actions if getattr(a, "dest", None) == "mode"
    )
    assert "report" in mode_action.choices
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_report_entry_point.py -v`
Expected: FAIL — `report` is not in the mode choices, and `subject.py` currently imports `_sample_masks_from_confounds` from `reporting`, which pulls in the fitting modules.

- [ ] **Step 3: Break the import that violates the invariant**

Move `_sample_masks_from_confounds` out of `reporting.py` into `fmri_pipeline/analysis/report/manifest.py` as `sample_masks_from_confounds`, and update `subject.py` to import it from there. `reporting.py` imports the contrast builder transitively, so importing anything from it defeats the decoupling.

- [ ] **Step 4: Add the `report` mode**

In `fmri_pipeline/cli/commands/fmri_analysis.py`, add `"report"` to the `mode` choices and add a handler that resolves subjects, then for each subject:

```python
    from fmri_pipeline.analysis.report.manifest import discover_manifests
    from fmri_pipeline.analysis.report.subject import build_subject_report

    manifests = discover_manifests(
        deriv_root=deriv_root, subject=sub_label, task=task
    )
    if not manifests:
        logger.warning(
            "No contrast manifests for %s task-%s; run first-level analysis first.",
            sub_label, task,
        )
        continue
    out_path = (
        deriv_root / sub_label / "fmri" / "first_level" / f"task-{task}"
        / f"{sub_label}_task-{task}_report.html"
    )
    build_subject_report(
        manifests=manifests, deriv_root=deriv_root, out_path=out_path, cfg=report_cfg
    )
```

- [ ] **Step 5: Write the manifest from the analysis run**

In `fmri_pipeline/pipelines/fmri_analysis.py`, after `nib.save(contrast_img, ...)`, build a `ContrastManifest` from `run_meta` and `contrast_cfg` and write it to `out_dir / MANIFEST_FILENAME`. Then delete the whole `run_fmri_plotting_and_report` call block (`fmri_analysis.py:259`-`:390` in the pre-plan-1 numbering), including the MNI refit at `:312` — MNI output is now the responsibility of `fmri_stats.space`, which the first-level run honours by fitting in that space directly rather than refitting for plots.

- [ ] **Step 6: Remove the moved orchestration from reporting.py**

Delete `run_fmri_plotting_and_report`, `generate_fmri_space_section`, `write_fmri_report`, and `build_fmri_report_html`. Keep `generate_signature_tables` until Task 7's section consumes it, then remove that too. `reporting.py` should end this task holding only what the second-level path still uses.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/ -q`
Expected: PASS. Delete or rewrite guards that asserted the old orchestration's behaviour; each one should have a replacement asserting the same property of the new path.

- [ ] **Step 8: Verify end to end on real derivatives**

```bash
.venv/bin/python -m eeg_pipeline fmri-analysis report --subject 0001 --task thermalactive
```

Expected: a `sub-0001_task-thermalactive_report.html` with one QC section and one section per contrast. Open it and check for label collisions, overflow, and missing panels — the tests check structure, not layout.

- [ ] **Step 9: Commit**

```bash
git add -A fmri_pipeline tests
git commit -m "feat(fmri): generate the subject report from derivatives, not from the GLM path"
```

---

## Self-Review

**Spec coverage.** Plan-2 items from the spec map as: manifest and decoupling → Tasks 1, 8; config split → Task 2; document primitives → Task 3; header, as-modelled QC, methods → Task 4; per-contrast results and the demoted diagnostics → Task 5; numbered cluster peaks → Task 6; signature dot plot and design section → Task 7; `fmri report` entry point → Task 8.

Deferred to plan 3: the whole resting-state profile, the ROI degeneracy change, and the group report.

**Type consistency.** `ContrastManifest`, `discover_manifests`, `read_manifest`, `write_manifest`, `MANIFEST_FILENAME` (Task 1) are used under those names in Tasks 4-8. `FmriReportConfig` / `FmriStatsConfig` (Task 2) are consumed from Task 4 on. `html.Document`/`Section`/`Figure`/`Table`/`KeyValues`/`Note`/`render` (Task 3) are used in Tasks 4-7. `_panel`, `_save`, `_slug` are defined in Task 4 and reused in Tasks 5-7. Plan 1's `stat_map_figures.dual_coded_mosaic`, `glass_brain(peak_coords=…)`, `volume_figures.compute_tsnr`/`per_run_tsnr_figure`, `coverage_figures.coverage_figure`, `distribution_figures.z_histogram`, and `design_figures.design_matrix_figure`/`collinearity_figure` are all called with the signatures that plan shipped.

**Carried from plan 1's known gaps, now actionable.**

- **Smoothness from residuals.** Plan 1 estimates FWHM from the z map and says so, because the report layer had no residuals. The manifest is where that changes: add a `residuals` path in Task 1 if the first-level run can write one, and pass it to `estimate_fwhm` in Task 4. Left out of the task steps above because whether nilearn's `FirstLevelModel.residuals` is affordable to persist for every run has not been measured — measure before adding.
- **Carpet before/after confound regression.** Same dependency. Once the manifest can name a cleaned timeseries, the carpet becomes a before/after pair, which is what actually demonstrates the confound model worked.
- **Design efficiency.** `1 / (cᵀ (XᵀX)⁻¹ c)` belongs in Task 7's design section now that a contrast vector and a design matrix are both in the manifest. Add it as a row beside the VIF panel.

**Known risks.**

- **Task 8 is the largest and least reversible step.** It deletes orchestration that other code may still call. Run `git grep -n run_fmri_plotting_and_report` before deleting, and expect the second-level path to need its own small shim.
- **The `report` mode needs `deriv_root` resolved the same way the analysis modes resolve it.** Reuse the existing resolution rather than re-deriving it, or the report will look for manifests somewhere the analysis never wrote them.
- **`build_cluster_table` assumes nilearn's column names `X`, `Y`, `Z`.** Verified against 0.14.0's `get_clusters_table`; guard with the `set(frame.columns)` check already in the code so a rename degrades to no peaks rather than a crash.
