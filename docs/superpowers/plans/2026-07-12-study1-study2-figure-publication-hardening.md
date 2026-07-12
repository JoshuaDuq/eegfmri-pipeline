# Study 1 and Study 2 Figure Publication Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current Study 1 and Study 2 figures submission-ready by correcting embedded-image resolution, legend placement, visual consistency, panel navigation, and adjusted-p terminology without changing any analysis.

**Architecture:** Keep scientific computations separate from rendering. Shared style modules own SVG raster resolution and Study 2 color maps; plot modules own layout and explanatory labels; Study 2 statistical modules own adjusted-p names and artifact schemas. Tests inspect both in-memory figures and serialized SVG payloads.

**Tech Stack:** Python 3.11+, Matplotlib, MNE-Python, Nilearn, NumPy, pandas, pytest, SVG/XML, deterministic PNG-in-SVG serialization.

---

### Task 1: Correct Holm-adjusted p-value terminology at the source

**Files:**
- Modify: `studies/pain_study/study2/statistics.py`
- Modify: `studies/pain_study/study2/source_family.py`
- Modify: `studies/pain_study/study2/artifact_controls.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/pain_study/study2/figures/primary_source_associations.py`
- Test: `studies/tests/pipelines/test_study2_source_family.py`
- Test: `studies/tests/pipelines/test_study2_artifact_controls.py`
- Test: `studies/tests/pipelines/test_study2_primary_source_associations.py`
- Test: `studies/tests/pipelines/test_study2_stages.py`

- [ ] **Step 1: Rename the expected public API and artifact columns in tests**

Replace q-value assertions with adjusted-p assertions. The source-family tests must use:

```python
assert result.band_results["alpha"].holm_adjusted_p_value == pytest.approx(0.03)
assert result.band_results["beta"].holm_adjusted_p_value == 1.0
assert tuple(summary.columns) == (
    "band",
    "n_subjects",
    "n_permutations",
    "n_clusters",
    "min_cluster_p_value",
    "holm_adjusted_p_value",
    "significant",
)
```

The primary-source tests must require `band_holm_adjusted_p_value` in the cluster audit and
`holm_adjusted_p_value` in the band summary. Artifact-control and stage tests must require
`expression_adjusted_p_values`.

- [ ] **Step 2: Run the renamed tests and confirm they fail against the old API**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_source_family.py \
  studies/tests/pipelines/test_study2_artifact_controls.py \
  studies/tests/pipelines/test_study2_primary_source_associations.py \
  studies/tests/pipelines/test_study2_stages.py
```

Expected: failures naming missing `holm_adjusted_p_value` or stale `holm_q_value` /
`expression_q_values` columns.

- [ ] **Step 3: Rename the shared helper and source-family API**

In `statistics.py`, define and export only:

```python
def holm_adjusted_p_values(p_values: Mapping[str, object]) -> dict[str, float]:
    if not p_values:
        return {}

    parsed: list[tuple[str, float]] = []
    for name, value in p_values.items():
        p_value = finite_number(value, f"p-value {name}")
        if p_value < 0.0 or p_value > 1.0:
            raise ValueError(f"Study 2 p-value must be in [0, 1]: {name}.")
        parsed.append((str(name), p_value))

    n_tests = len(parsed)
    running_max = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p_value) in enumerate(sorted(parsed, key=lambda item: item[1])):
        running_max = max(running_max, (n_tests - rank) * p_value)
        adjusted[name] = float(min(running_max, 1.0))
    return adjusted
```

In `source_family.py`, use:

```python
@dataclass(frozen=True)
class SourceFamilyBandResult:
    band: str
    inference: GroupSourceInferenceResult
    min_cluster_p_value: float
    holm_adjusted_p_value: float
    significant: bool
```

Write `holm_adjusted_p_value` in `summarize_source_family`. Do not retain aliases for the old
function, field, or column.

- [ ] **Step 4: Rename artifact-control and reporting fields**

Change `ArtifactControlQC.expression_q_values` to
`ArtifactControlQC.expression_adjusted_p_values`, and update `stages.py` to serialize the exact
key `expression_adjusted_p_values`. Continue applying the same Holm calculation and alpha rule.

- [ ] **Step 5: Rename primary-source dataclasses and audit schemas**

Use these schema members in `primary_source_associations.py`:

```python
CLUSTER_COLUMNS = (
    "band",
    "cluster_id",
    "sign",
    "n_vertices",
    "cluster_mass",
    "max_cluster_p_value",
    "band_holm_adjusted_p_value",
    "corrected_contour",
)

SUMMARY_COLUMNS = (
    "band",
    "n_subjects",
    "n_vertices",
    "n_permutations",
    "n_clusters",
    "n_corrected_clusters",
    "min_cluster_p_value",
    "holm_adjusted_p_value",
    "significant",
    "cluster_forming_p",
    "cluster_threshold",
    "family_alpha",
    "display_limit",
)
```

Rename `PrimarySourceBand.holm_q_value` to `holm_adjusted_p_value`, update the joint contour
rule, and require regenerated saved source-family summaries with the new schema.

- [ ] **Step 6: Run the focused terminology tests**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 7: Commit the terminology correction**

```bash
git add studies/pain_study/study2/statistics.py \
  studies/pain_study/study2/source_family.py \
  studies/pain_study/study2/artifact_controls.py \
  studies/pain_study/study2/stages.py \
  studies/pain_study/study2/figures/primary_source_associations.py \
  studies/tests/pipelines/test_study2_source_family.py \
  studies/tests/pipelines/test_study2_artifact_controls.py \
  studies/tests/pipelines/test_study2_primary_source_associations.py \
  studies/tests/pipelines/test_study2_stages.py
git commit -m "fix: name Holm-adjusted p-values accurately"
```

### Task 2: Guarantee print-resolution scientific layers in SVG output

**Files:**
- Create: `studies/tests/figure_svg.py`
- Modify: `studies/pain_study/study1/figures/validity_style.py`
- Modify: `studies/pain_study/study2/figures/style.py`
- Test: `studies/tests/pipelines/test_study1_validity_figures.py`
- Test: `studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py`

- [ ] **Step 1: Add a dependency-free SVG embedded-raster inspector**

Create `studies/tests/figure_svg.py` with:

```python
from __future__ import annotations

import base64
import struct
from pathlib import Path
from xml.etree import ElementTree

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def embedded_raster_dpi(svg_path: Path) -> tuple[tuple[float, float], ...]:
    root = ElementTree.parse(svg_path).getroot()
    resolutions = []
    for image in root.findall(f".//{{{SVG_NAMESPACE}}}image"):
        payload = base64.b64decode(image.attrib[XLINK_HREF].split(",", 1)[1])
        if not payload.startswith(PNG_SIGNATURE):
            raise ValueError("SVG test helper requires embedded PNG images.")
        width_px, height_px = struct.unpack(">II", payload[16:24])
        width_pt = float(image.attrib["width"])
        height_pt = float(image.attrib["height"])
        resolutions.append(
            (width_px / (width_pt / 72.0), height_px / (height_pt / 72.0))
        )
    return tuple(resolutions)
```

- [ ] **Step 2: Add failing 600-dpi serialization tests**

In the Study 1 validity test, save a figure containing `axis.imshow(np.eye(2))`, inspect it with
`embedded_raster_dpi`, require at least one embedded image, and assert every x/y resolution is at
least 599.0 dpi.

In the Study 2 Haufe writer test, inspect the saved SVG the same way and assert all six embedded
images are at least 599.0 dpi.

- [ ] **Step 3: Run both serialization tests and confirm the measured 100-dpi failure**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_validity_figures.py::test_publication_svg_embeds_rasters_at_print_resolution \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py::test_haufe_writer_creates_exact_editable_svg
```

Expected: both tests fail because current embedded layers are approximately 100 dpi.

- [ ] **Step 4: Set one explicit embedded-raster resolution per style module**

Add this constant to both style modules:

```python
EMBEDDED_RASTER_DPI = 600
```

Pass `dpi=EMBEDDED_RASTER_DPI` to each `figure.savefig(..., format="svg")` call. Keep editable
text, deterministic hash salts, physical dimensions, and metadata unchanged.

- [ ] **Step 5: Rerun the serialization tests**

Run the command from Step 3.

Expected: both tests pass with effective resolutions within rounding tolerance of 600 dpi.

- [ ] **Step 6: Commit the export hardening**

```bash
git add studies/tests/figure_svg.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py \
  studies/pain_study/study1/figures/validity_style.py \
  studies/pain_study/study2/figures/style.py
git commit -m "fix: export embedded figure layers at print resolution"
```

### Task 3: Move standalone Study 1 legends out of the data region

**Files:**
- Modify: `studies/pain_study/study1/figures/validity_style.py`
- Modify: `studies/pain_study/study1/figures/dose_response.py`
- Modify: `studies/pain_study/study1/figures/coefficient_plot.py`
- Test: `studies/tests/pipelines/test_study1_validity_figures.py`
- Test: `studies/tests/pipelines/test_study1_behavioral_validity_figures.py`

- [ ] **Step 1: Add failing layout assertions for both standalone figure families**

After building each figure, draw its canvas and assert:

```python
assert axis.get_legend() is None
assert len(figure.legends) == 1
legend_bounds = figure.legends[0].get_window_extent()
axis_bounds = axis.get_window_extent()
assert legend_bounds.y0 >= axis_bounds.y1
```

The initial tests must fail because both legends currently belong to the axes.

- [ ] **Step 2: Run the two structure tests and confirm failure**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_validity_figures.py::test_build_dose_response_figure_draws_scientific_layers \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py::test_coefficient_figure_draws_scientific_layers
```

Expected: both fail on the outside-legend contract.

- [ ] **Step 3: Add one shared layout helper**

In `validity_style.py`, add:

```python
def add_outside_top_legend(figure: Figure, axis: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        raise ValueError("Publication figure legend requires at least one labeled artist.")
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
    )
```

Export the helper from `__all__`.

- [ ] **Step 4: Use the shared helper in both renderers**

Remove `axis.legend(...)` from `_format_axis` in `dose_response.py` and
`coefficient_plot.py`. Call `add_outside_top_legend(figure, axis)` after all labeled artists are
drawn. Retain `layout="constrained"` so Matplotlib reserves upper space.

- [ ] **Step 5: Rerun both structure tests**

Run the command from Step 2.

Expected: both pass; the figure legend is entirely above the axes.

- [ ] **Step 6: Commit the collision-free legend layout**

```bash
git add studies/pain_study/study1/figures/validity_style.py \
  studies/pain_study/study1/figures/dose_response.py \
  studies/pain_study/study1/figures/coefficient_plot.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py
git commit -m "fix: keep standalone figure legends outside data"
```

### Task 4: Unify and explain the Study 2 Haufe figure

**Files:**
- Modify: `studies/pain_study/study2/figures/style.py`
- Modify: `studies/pain_study/study2/figures/haufe_forward_patterns_plot.py`
- Modify: `studies/pain_study/study2/figures/primary_source_associations_plot.py`
- Test: `studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py`

- [ ] **Step 1: Add failing palette, hierarchy, and symbol-key tests**

Extend the Haufe structure test with:

```python
title_weights = {axis.title.get_fontweight() for axis in figure.axes[:5]}
assert len(title_weights) == 1
legend = figure.axes[5].get_legend()
assert legend is not None
assert [text.get_text() for text in legend.get_texts()] == ["Fold pairs", "Median"]
```

Compare each topomap color map to the exported Study 2 shared diverging map at 0.0, 0.5, and 1.0.

- [ ] **Step 2: Run the Haufe structure test and confirm failure**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py::test_build_haufe_figure_has_fixed_scientific_structure
```

Expected: failure because title weights differ, no stability legend exists, and `RdBu_r` is not
the shared palette.

- [ ] **Step 3: Centralize the accessible Study 2 diverging map**

In `style.py`, add and export:

```python
def study2_diverging_color_map() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "study2_diverging",
        ("#2166AC", "#F7F7F7", "#D95F0E"),
        N=256,
    )
```

Use this function from both Study 2 plot modules and delete the private cortical-map duplicate.

- [ ] **Step 4: Equalize titles and define stability symbols**

Pass the shared color map to all five `mne.viz.plot_topomap` calls. Set every band title to
`fontweight="bold"`. Add two `Line2D` handles to the stability axis:

```python
axis.legend(
    handles=(fold_pair_handle, median_handle),
    loc="lower right",
    bbox_to_anchor=(1.0, 1.01),
    ncol=2,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0,
)
```

The gray handle must match the fold-pair points; the outlined diamond must match the median.

- [ ] **Step 5: Rerun all Haufe figure tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the Study 2 visual-language cleanup**

```bash
git add studies/pain_study/study2/figures/style.py \
  studies/pain_study/study2/figures/haufe_forward_patterns_plot.py \
  studies/pain_study/study2/figures/primary_source_associations_plot.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py
git commit -m "fix: unify and explain Study 2 pattern figures"
```

### Task 5: Make the primary cortical figure fully transparent and citable by panel

**Files:**
- Modify: `studies/pain_study/study2/figures/primary_source_associations_plot.py`
- Modify: `studies/pain_study/study2/figures/plot_primary_source_associations.py`
- Modify: `studies/pain_study/study2/README.md`
- Modify: `docs/superpowers/specs/2026-07-11-study2-primary-source-associations-design.md`
- Modify: `docs/superpowers/plans/2026-07-11-study2-primary-source-associations.md`
- Test: `studies/tests/pipelines/test_study2_primary_source_associations_figure.py`

- [ ] **Step 1: Add failing visible-label assertions**

Require this cortical figure contract:

```python
assert color_axes[0].get_xlabel() == (
    "Mean partial correlation, r (Fisher-z averaged)"
)
figure_text = [label.get_text() for label in figure.texts]
for panel in ("a", "b", "c"):
    assert figure_text.count(panel) == 1
text = " ".join(figure_text)
assert text.count("Holm-adjusted p =") == 3
assert "Holm q" not in text
assert "no family-corrected cluster" not in text
```

Update writer/audit assertions to the adjusted-p columns from Task 1.

- [ ] **Step 2: Run the primary-source figure tests and confirm failure**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py
```

Expected: visible-label tests fail against the old conditional q-value wording and absent panel
letters.

- [ ] **Step 3: Show adjusted p-values for all bands**

Change `_add_band_header` to accept `holm_adjusted_p_value` and always draw:

```python
f"n = {n_subjects} · Holm-adjusted p = {holm_adjusted_p_value:.3f}"
```

Remove its unused `significant` argument. Keep corrected contours as the sole visual encoding of
the joint cluster/family rule.

- [ ] **Step 4: Add panel letters and clarify the displayed effect**

Add one lowercase bold panel letter at the upper left of each band column without changing the
band-header centers. Change the color-bar label to:

```text
Mean partial correlation, r (Fisher-z averaged)
```

Retain the existing main title, surface-view labels, shared symmetric range, and footer.

- [ ] **Step 5: Update caption and documentation terminology**

Use `≤` rather than `<=` in the generated caption. Replace Holm q-value terminology in the
current Study 2 source-figure README, design, and implementation plan with “Holm-adjusted
p-value”. State that every band header displays the adjusted p-value.

- [ ] **Step 6: Rerun primary-source analysis and figure tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_source_family.py \
  studies/tests/pipelines/test_study2_primary_source_associations.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the cortical-figure communication changes**

```bash
git add studies/pain_study/study2/figures/primary_source_associations_plot.py \
  studies/pain_study/study2/figures/plot_primary_source_associations.py \
  studies/pain_study/study2/README.md \
  docs/superpowers/specs/2026-07-11-study2-primary-source-associations-design.md \
  docs/superpowers/plans/2026-07-11-study2-primary-source-associations.md \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py
git commit -m "fix: clarify primary source figure inference"
```

### Task 6: Regenerate, inspect, and run repository gates

**Files:**
- Verify: all Study 1 and Study 2 figure modules and tests

- [ ] **Step 1: Run the complete focused figure suite into a retained directory**

```bash
rm -rf /tmp/eeg_fmri_figure_hardening
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  --basetemp=/tmp/eeg_fmri_figure_hardening -q \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_primary_prediction_figure.py \
  studies/tests/pipelines/test_study1_spectral_specificity_figure.py \
  studies/tests/pipelines/test_study1_temporal_specificity_figure.py \
  studies/tests/pipelines/test_study1_power_construct_validity_figure.py \
  studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_figure.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py \
  studies/tests/pipelines/test_study2_primary_source_associations.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py
```

Expected: all tests pass.

- [ ] **Step 2: Audit every generated SVG payload**

Use `studies.tests.figure_svg.embedded_raster_dpi` on each retained SVG. Assert that every
embedded image is at least 599 dpi, while figures containing no raster artists remain valid
editable SVGs.

- [ ] **Step 3: Rasterize and visually inspect representative outputs**

Convert the retained SVGs with macOS `sips` into `/tmp/eeg_fmri_figure_hardening_png`. Inspect at
the intended aspect ratio and verify:

- legends are above, not over, Study 1 standalone data;
- all text is legible at 89 mm or 183 mm width;
- no title, annotation, confidence interval, or panel label is clipped;
- Study 2 topographic and cortical palettes match;
- fold-pair and median symbols are unambiguous; and
- adjusted p-values and panel letters fit every cortical header.

- [ ] **Step 4: Run lint and architecture/maintainability gates**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/ruff check \
  studies/pain_study/study1/figures \
  studies/pain_study/study2 \
  studies/tests/figure_svg.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study2_artifact_controls.py \
  studies/tests/pipelines/test_study2_haufe_forward_pattern_figure.py \
  studies/tests/pipelines/test_study2_primary_source_associations.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_source_family.py \
  studies/tests/pipelines/test_study2_stages.py
make verify-architecture
make verify-maintainability
```

Expected: all commands exit successfully.

- [ ] **Step 5: Run the broader Study 2 and Study 1 reporting suites**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  studies/tests/pipelines/test_study2_artifact_controls.py \
  studies/tests/pipelines/test_study2_source_family.py \
  studies/tests/pipelines/test_study2_stages.py \
  studies/tests/pipelines/test_study2_runner.py
```

Expected: all tests pass.

- [ ] **Step 6: Confirm branch cleanliness and commit any verification-only correction**

```bash
git status --short
git log --oneline --decorate -8
```

Expected: no uncommitted implementation files. If visual verification required a scoped correction,
repeat its focused failing/passing test and commit it with an imperative `fix:` subject before
running this final status check again.
