# Study 1 and Study 2 Publication Figure Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply one publication-quality visual contract to every current Study 1 and Study 2 figure and correct the layout, interpolation, and statistical-annotation defects found in the final-size render audit.

**Architecture:** Add a small cross-study style module containing only shared visual policy, then keep configuration and export ownership in the existing Study 1 and Study 2 wrappers. Make targeted rendering changes only in the standalone validity, Haufe, primary-source, spatial-convergence, and fMRI construct plot modules; scientific summaries and statistics remain untouched.

**Tech Stack:** Python 3.11+, Matplotlib, MNE-Python, Nilearn, NumPy, pytest, Ruff, deterministic SVG/PNG rendering.

---

### Task 1: Establish the shared publication contract

**Files:**
- Create: `studies/pain_study/figure_style.py`
- Modify: `studies/pain_study/study1/figures/validity_style.py`
- Modify: `studies/pain_study/study2/figures/style.py`
- Create: `studies/tests/pipelines/test_study_figure_style.py`

- [ ] **Step 1: Write failing tests for the common style API**

Create `studies/tests/pipelines/test_study_figure_style.py` with tests that require
the shared conversion, exact rc keys, no font fallback, and a validated outside
legend:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib import font_manager

from studies.pain_study.figure_style import (
    outside_top_legend,
    publication_rc_params,
    require_font_family,
)


def test_publication_rc_params_define_editable_neutral_artwork() -> None:
    params = publication_rc_params("Arial", svg_hash_salt="study-figures")

    assert params["font.family"] == "Arial"
    assert params["font.size"] == 6.0
    assert params["axes.labelsize"] == 7.0
    assert params["axes.titlesize"] == 7.0
    assert params["legend.frameon"] is False
    assert params["svg.fonttype"] == "none"
    assert params["pdf.fonttype"] == 42
    assert params["figure.facecolor"] == "white"
    assert params["text.color"] == "#1A1A1A"


def test_require_font_family_rejects_missing_font(monkeypatch) -> None:
    monkeypatch.setattr(font_manager, "findfont", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")))

    with pytest.raises(ValueError, match="Required figure font 'Missing' is unavailable"):
        require_font_family("Missing")


def test_outside_top_legend_requires_and_reserves_labeled_artists() -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Estimate")

    legend = outside_top_legend(figure, axis)
    figure.canvas.draw()

    assert axis.get_legend() is None
    assert figure.legends == [legend]
    assert legend.get_window_extent().y0 >= axis.get_window_extent().y1
    plt.close(figure)
```

- [ ] **Step 2: Run the new test and confirm the missing-module failure**

Run:

```bash
.venv/bin/python -m pytest -q studies/tests/pipelines/test_study_figure_style.py
```

Expected: collection fails because `studies.pain_study.figure_style` does not exist.

- [ ] **Step 3: Implement the common style module**

Create `studies/pain_study/figure_style.py` with constants and focused helpers:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib import font_manager
from matplotlib.figure import Figure

MILLIMETERS_PER_INCH = 25.4
TEXT_COLOR = "#1A1A1A"
MUTED_TEXT_COLOR = "#4A4A4A"
AXIS_COLOR = "#333333"
REFERENCE_COLOR = "#5C5C5C"
GRID_COLOR = "#D9D9D9"


def figure_size_inches(dimensions_mm: Mapping[str, float]) -> tuple[float, float]:
    return (
        float(dimensions_mm["width"]) / MILLIMETERS_PER_INCH,
        float(dimensions_mm["height"]) / MILLIMETERS_PER_INCH,
    )


def require_font_family(font_family: str) -> str:
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=font_family),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Required figure font '{font_family}' is unavailable."
        ) from exc
    return font_family


def publication_rc_params(
    font_family: str,
    *,
    svg_hash_salt: str,
) -> dict[str, Any]:
    require_font_family(font_family)
    return {
        "font.family": font_family,
        "font.size": 6.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 6.0,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": svg_hash_salt,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def outside_top_legend(figure: Figure, axis: Any):
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        raise ValueError("Publication figure legend requires labeled artists.")
    return figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
```

- [ ] **Step 4: Delegate both study style contexts to the common contract**

Import and re-export `figure_size_inches` in both existing style modules. In each
context manager, build the common mapping and update only study-configured font sizes
or the deterministic study hash salt. Remove duplicated font discovery and rc keys.
Keep both save functions and their closing behavior unchanged.

- [ ] **Step 5: Run the shared and existing style serialization tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study_figure_style.py \
  studies/tests/pipelines/test_study1_validity_figures.py::test_require_configured_font_rejects_missing_font \
  studies/tests/pipelines/test_study1_validity_figures.py::test_publication_svg_embeds_rasters_at_print_resolution \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py::test_haufe_writer_creates_exact_editable_svg
```

Expected: all selected tests pass and embedded rasters remain at least 599 dpi.

- [ ] **Step 6: Commit the shared visual contract**

```bash
git add studies/pain_study/figure_style.py \
  studies/pain_study/study1/figures/validity_style.py \
  studies/pain_study/study2/figures/style.py \
  studies/tests/pipelines/test_study_figure_style.py
git commit -m "refactor: unify study publication figure style"
```

### Task 2: Remove standalone legend collisions

**Files:**
- Modify: `studies/pain_study/study1/figures/dose_response.py`
- Modify: `studies/pain_study/study1/figures/coefficient_plot.py`
- Modify: `studies/tests/pipelines/test_study1_validity_figures.py`
- Modify: `studies/tests/pipelines/test_study1_behavioral_validity_figures.py`

- [ ] **Step 1: Add failing final-layout assertions**

In both existing structure tests, draw the figure and require:

```python
figure.canvas.draw()
assert axis.get_legend() is None
assert len(figure.legends) == 1
assert figure.legends[0].get_window_extent().y0 >= axis.get_window_extent().y1
```

- [ ] **Step 2: Run both tests and confirm they fail on axis-owned legends**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_validity_figures.py::test_build_dose_response_figure_draws_scientific_layers \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py::test_coefficient_figure_draws_scientific_layers
```

Expected: both fail because `axis.get_legend()` is not `None`.

- [ ] **Step 3: Use the shared outside legend after all artists are drawn**

Import `outside_top_legend` in both plot modules. Remove `axis.legend(...)` from the
axis-formatting helpers and call:

```python
outside_top_legend(figure, axis)
```

inside the existing constrained-layout context after axis formatting. Do not change
artist labels or data limits.

- [ ] **Step 4: Rerun both tests and the standalone writer tests**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py
```

Expected: all tests pass with exact 89 × 70 mm SVG output.

- [ ] **Step 5: Commit the layout correction**

```bash
git add studies/pain_study/study1/figures/dose_response.py \
  studies/pain_study/study1/figures/coefficient_plot.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py
git commit -m "fix: keep Study 1 legends outside data"
```

### Task 3: Correct and explain the Haufe topographies

**Files:**
- Modify: `studies/pain_study/study2/figures/haufe_forward_patterns_plot.py`
- Modify: `studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py`

- [ ] **Step 1: Add failing palette, boundary, hierarchy, and legend tests**

Extend the Haufe structure test to require:

```python
from matplotlib.colors import to_hex

assert len(figure.legends) == 1
assert [text.get_text() for text in figure.legends[0].get_texts()] == [
    "Fold-pair correlation",
    "Median",
]
assert {title.get_fontweight() for title in [axis.title for axis in figure.axes[:5]]} == {"bold"}
expected_endpoints = ("#2166ac", "#d95f0e")
for axis in figure.axes[:5]:
    image = axis.images[0]
    assert tuple(to_hex(image.get_cmap()(value)) for value in (0.0, 1.0)) == expected_endpoints
    assert getattr(image, "_study2_head_bounded", False)
```

- [ ] **Step 2: Run the structure test and confirm the current failures**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py::test_build_haufe_figure_has_fixed_scientific_structure
```

Expected: failure on missing legend, unequal title weight, old palette, or missing
head-boundary marker.

- [ ] **Step 3: Use head-bounded interpolation and the shared Study 2 map**

Import `study2_diverging_color_map`. In `plot_topomap`, replace the palette and
extrapolation arguments with:

```python
cmap=study2_diverging_color_map(),
extrapolate="head",
```

After creation, mark the tested semantic contract:

```python
image._study2_head_bounded = True
```

Set every band title to `fontweight="bold"`.

- [ ] **Step 4: Add the stability-symbol legend outside the axis**

Create two `Line2D` handles matching the gray fold-pair dots and the white diamond
with NPS-colored edge. Add one figure legend above the stability panel with two
columns, and update the footnote to:

```text
Scalp colors are interpolated within the head outline; sensor-level forward patterns are not cortical source localization
```

- [ ] **Step 5: Rerun the complete Haufe figure test file**

```bash
.venv/bin/python -m pytest -q studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py
```

Expected: all tests pass, including the six 600 dpi embedded image checks.

- [ ] **Step 6: Commit the topography correction**

```bash
git add studies/pain_study/study2/figures/haufe_forward_patterns_plot.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py
git commit -m "fix: clarify Study 2 Haufe topographies"
```

### Task 4: Make Study 2 statistical annotations exact and readable

**Files:**
- Modify: `studies/pain_study/study2/figures/primary_source_associations_plot.py`
- Modify: `studies/pain_study/study2/figures/spatial_convergence_plot.py`
- Modify: `studies/tests/pipelines/test_study2_primary_source_associations_figure.py`
- Modify: `studies/tests/pipelines/test_study2_spatial_convergence_figure.py`

- [ ] **Step 1: Add failing primary-source label assertions**

Replace the stale `Holm q` assertion with:

```python
assert "Holm-adjusted p = 0.030" in text
assert "Holm-adjusted p = 1.000" in text
assert "Holm q" not in text
assert {label.get_text() for label in figure.texts}.issuperset({"a", "b", "c"})
```

- [ ] **Step 2: Add failing spatial-null text assertions**

For each null axis, require a short title and two exact annotation lines:

```python
assert axis.get_title() in {"Alpha", "Beta", "Scanner-clean gamma"}
texts = {text.get_text() for text in axis.texts}
assert any(text.startswith("Observed r = ") for text in texts)
assert any("two-sided p = " in text and "Holm-adjusted p = " in text for text in texts)
```

- [ ] **Step 3: Run both structure tests and confirm wording failures**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py::test_primary_source_figure_has_fixed_publication_structure \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py::test_renderer_has_fixed_multimodal_structure
```

Expected: failures on stale `q`, omitted nonsignificant adjusted p-values, missing
panel letters, and the current long null title.

- [ ] **Step 4: Show every adjusted p-value and band panel letter**

In `_add_band_header`, remove the significance-dependent status and use:

```python
status = f"Holm-adjusted p = {holm_adjusted_p_value:.3f}"
```

Add lowercase `a`, `b`, and `c` above the three band columns. Retain the significant
argument only if it still controls an existing scientific artist; otherwise remove it
from the private helper signature and call.

- [ ] **Step 5: Split spatial-null statistics into hierarchy-managed text**

Set the axis title to the band label at the left. Add:

```python
axis.text(
    0.02,
    0.98,
    f"Observed r = {result.spatial_r:.3f}",
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontweight="bold",
)
axis.text(
    0.02,
    0.90,
    (
        f"Plus-one two-sided p = {result.p_value:.4f}\n"
        f"Holm-adjusted p = {result.holm_adjusted_p_value:.4f}"
    ),
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontsize=5.2,
)
```

Move the observed diamond high enough to avoid the text while retaining the same
x-coordinate and semantic gid.

- [ ] **Step 6: Rerun the two complete Study 2 figure files**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
```

Expected: all tests pass; contours, masks, shared limits, output families, and atomic
promotion tests remain unchanged.

- [ ] **Step 7: Commit the statistical communication cleanup**

```bash
git add studies/pain_study/study2/figures/primary_source_associations_plot.py \
  studies/pain_study/study2/figures/spatial_convergence_plot.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
git commit -m "fix: clarify Study 2 figure statistics"
```

### Task 5: Preserve anatomical background in the fMRI axial row

**Files:**
- Modify: `studies/pain_study/study1/figures/fmri_construct_validity_plot.py`
- Modify: `studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py`

- [ ] **Step 1: Add a failing plotting-call regression test**

Monkeypatch `nilearn.plotting.plot_stat_map` with a recorder, build the figure, and
assert both estimands pass an exact-zero threshold:

```python
assert len(recorded_calls) == 2
assert all(call["threshold"] == 0.0 for call in recorded_calls)
```

The recorder returns an object exposing `add_contours`, because the renderer must
continue to add the max-T significance outline.

- [ ] **Step 2: Run the new test and confirm `threshold=None` fails**

```bash
.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py::test_fmri_axial_views_leave_exact_zero_voxels_transparent
```

Expected: failure because the current call explicitly sets `threshold=None`.

- [ ] **Step 3: Mask only exact-zero axial effects**

Change `_draw_axial_views` to pass:

```python
threshold=0.0,
```

Keep the surface map threshold, robust display limits, color map, fixed slices, and
significance contours unchanged. Update the footnote to say `nonzero unthresholded`
effects so the rendering rule is explicit.

- [ ] **Step 4: Rerun the complete fMRI construct figure test file**

```bash
.venv/bin/python -m pytest -q studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py
```

Expected: all tests pass, including the exact output and reproducibility contract.

- [ ] **Step 5: Commit the anatomical-background correction**

```bash
git add studies/pain_study/study1/figures/fmri_construct_validity_plot.py \
  studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py
git commit -m "fix: preserve anatomy in fMRI figure overlays"
```

### Task 6: Render, inspect, and verify the full figure suite

**Files:**
- Modify only files from Tasks 1–5 if visual verification reveals a reproducible
  contract defect and a failing test is added first.

- [ ] **Step 1: Run the complete focused figure suite**

```bash
MPLCONFIGDIR=/tmp/eeg-fmri-mpl .venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study_figure_style.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_primary_prediction_figure.py \
  studies/tests/pipelines/test_study1_temporal_specificity_figure.py \
  studies/tests/pipelines/test_study1_spectral_specificity_figure.py \
  studies/tests/pipelines/test_study1_power_construct_validity_figure.py \
  studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_figure.py \
  studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Render deterministic before/after figures at final aspect ratios**

Use the existing synthetic summary helpers to write PNG previews for primary
prediction, temporal specificity, spectral specificity, power construct validity,
cohort PSD, scanner harmonics, Haufe patterns, fMRI construct validity, both
standalone validity families, primary source associations, and spatial convergence.
Inspect the retained images for legend/axis separation, clipped labels, head-boundary
containment, anatomical visibility, consistent type, exact probability text, panel
order, and color balance.

- [ ] **Step 3: Run focused lint and repository gates**

```bash
.venv/bin/ruff check \
  studies/pain_study/figure_style.py \
  studies/pain_study/study1/figures \
  studies/pain_study/study2/figures \
  studies/tests/pipelines/test_study_figure_style.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
make verify-architecture
make verify-maintainability
git diff --check
```

Expected: Ruff and both repository gates exit successfully with no whitespace errors.

- [ ] **Step 4: Review the implementation against the approved design**

Confirm each acceptance criterion in
`docs/superpowers/specs/2026-07-15-study1-study2-publication-figure-redesign-design.md`
has direct test or visual evidence. Confirm `git status --short` contains only intended
changes and no generated previews.

- [ ] **Step 5: Request final code review and address verified findings**

Invoke `superpowers:requesting-code-review` with the design, this plan, commit range,
and scientific invariants. For any blocking defect, add a failing regression test,
make the smallest correction, and repeat Steps 1 and 3 before completion.
