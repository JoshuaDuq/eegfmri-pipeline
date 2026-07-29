# fMRI Report Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce three parallel implementations of the fMRI report to one in `main`, finish the per-subject report, and verify every panel against a real subject's derivatives.

**Architecture:** The `feat/fmri-subject-report` branch's decomposed `report/` package becomes the single implementation in `main`. Main's uncommitted working tree contributes a better `design.py`, half of `style.py`, and three correctness fixes harvested out of its monolithic `reporting.py`. The report is then completed by adding the signature panel, making `contrast_builder` write the manifest, and moving report generation out of the GLM path onto its own CLI mode.

**Tech Stack:** Python, matplotlib, nilearn 0.14.0, nibabel, numpy, pandas, pytest, git.

Implements `docs/superpowers/specs/2026-07-28-fmri-report-consolidation-design.md`, which is an addendum to `docs/superpowers/specs/2026-07-28-fmri-post-preprocessing-report-design.md`. Tasks 7–9 complete tasks 7–8 of `docs/superpowers/plans/2026-07-28-fmri-subject-report.md`; tasks 4–6 of that plan turn out to be already implemented in the uncommitted `subject.py` this plan lands.

## Global Constraints

- **All work happens in the main checkout at `/Users/joduq24/Desktop/EEG_fMRI_Pipeline` on branch `main`.** Do not create a git worktree. Do not use `EnterWorktree` or `isolation: "worktree"`.
- **Never run the full test suite.** It takes roughly nine minutes. Run targeted subsets: `.venv/bin/python -m pytest tests/fmri/report/ -q`.
- Use the project interpreter: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python`. `timeout` is not available on this machine; do not wrap commands in it.
- Signed maps (z, effect size) use `RdBu_r` with symmetric limits. Unsigned magnitude (tSNR, standard error) uses `cividis`. Categorical series use Okabe-Ito. No rainbow colormaps, no `cold_hot`.
- Every function in `report/figures/` returns a `matplotlib.figure.Figure` and takes no `Path` and no config object. Saving, formats, and configuration belong to `subject.py`.
- Every figure calls `annotate_provenance` with, at minimum, its sample size, the threshold applied, the colour limit, and the fraction of data that limit clipped.
- **No verdicts.** The report states measured values and the thresholds actually applied. No pass/fail badges, no cutoffs the pipeline invented, and no caption in which cluster extent can read as inference.
- A panel that fails renders a placeholder naming the exception; the document always builds. Figures are closed in `finally`.
- Rendering is deterministic: SVG saves pass `metadata={"Date": None}` and `svg.hashsalt` stays fixed.
- Commit after every task. Do not squash tasks together.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `fmri_pipeline/analysis/report/style.py` | Render conventions. Merged from both efforts. | 2 |
| `fmri_pipeline/analysis/report/figures/design.py` | Design matrix, regressor classes, VIF, correlation, contrast efficiency. Main's version adopted. | 3 |
| `fmri_pipeline/analysis/report/figures/volumes.py` | tSNR rendering. Gains drift removal. | 4 |
| `fmri_pipeline/analysis/report/subject.py` | Document assembly. Gains the glass-brain space guard, coordinate-space labels, and the signature section. | 5, 6, 7 |
| `fmri_pipeline/analysis/report/figures/signatures.py` | Signature expression dot plot. New. | 7 |
| `fmri_pipeline/analysis/contrast_builder.py` | Writes `report_manifest.json` beside the stat maps. | 8 |
| `fmri_pipeline/pipelines/fmri_analysis.py` | Stops calling `run_fmri_plotting_and_report`. | 9 |
| `fmri_pipeline/cli/commands/fmri_analysis.py` | Adds the `report` mode. | 9 |

---

### Task 1: Land the branch in main without losing uncommitted work

**Files:**
- Modify: repository state only. No source edits in this task.

**Interfaces:**
- Produces: a `main` branch containing the 13 commits of `feat/fmri-subject-report`, with main's complementary fMRI work and unrelated EEG work still present in the working tree, and main's three colliding files preserved in the scratchpad for tasks 2–6.

The working tree carries uncommitted changes to roughly twenty files, most of them unrelated EEG work. Four paths collide with the incoming branch and must be moved out of the way first: `reporting.py`, the private `report/` copy, `tests/fmri/report/`, and `tests/fmri/test_fmri_analysis_validity_guards.py`. Everything else — `contrast_builder.py`, `bold_discovery.py`, `pipelines/fmri_analysis.py`, `trial_signatures.py`, and all EEG files — is untouched by the branch and stays in the working tree throughout.

- [ ] **Step 1: Snapshot the entire working tree without modifying it**

`git stash create` writes a commit object recording the working tree and index, but neither moves `HEAD` nor touches any file. Tagging it makes it reachable so garbage collection cannot reclaim it.

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git tag wip/pre-fmri-consolidation "$(git stash create)" && git show --stat wip/pre-fmri-consolidation | head -30
```

Expected: a commit listing roughly twenty modified files. If `git stash create` prints nothing the working tree is clean, which contradicts the premise of this task — stop and re-read `git status` before continuing.

- [ ] **Step 2: Copy the three harvest sources into the scratchpad**

Tasks 2–6 read these. They must survive the merge.

```bash
mkdir -p /private/tmp/claude-501/-Users-joduq24-Desktop-EEG-fMRI-Pipeline/35cfdc40-5a2a-4d93-b154-4592a3f72af8/scratchpad/harvest && cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && cp fmri_pipeline/analysis/reporting.py fmri_pipeline/analysis/report/style.py fmri_pipeline/analysis/report/figures/design.py tests/fmri/report/test_design_figures.py tests/fmri/report/test_style.py /private/tmp/claude-501/-Users-joduq24-Desktop-EEG-fMRI-Pipeline/35cfdc40-5a2a-4d93-b154-4592a3f72af8/scratchpad/harvest/ && ls -la /private/tmp/claude-501/-Users-joduq24-Desktop-EEG-fMRI-Pipeline/35cfdc40-5a2a-4d93-b154-4592a3f72af8/scratchpad/harvest/
```

Expected: five files. `reporting.py` should be about 2874 lines; verify with `wc -l`.

- [ ] **Step 3: Commit the branch's uncommitted `subject.py`**

`subject.py` and `test_subject.py` are uncommitted in the worktree. They must be committed before the branch can be merged.

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.worktrees/fmri-plotting-foundation && git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject.py && git commit -m "feat(fmri): assemble the per-subject report document

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Remove the colliding paths from main's working tree**

`git stash push` with a pathspec stores only those paths and reverts them, leaving every other uncommitted change in place. The private `report/` copy and `tests/fmri/report/` are untracked, so they need `--include-untracked`.

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git stash push --include-untracked -m "main-fmri-report-restart" -- fmri_pipeline/analysis/reporting.py fmri_pipeline/analysis/report tests/fmri/report tests/fmri/test_fmri_analysis_validity_guards.py && git status --short | head -25
```

Expected: `fmri_pipeline/analysis/report/` and `tests/fmri/report/` are gone from `git status`; `eeg_pipeline/*`, `contrast_builder.py`, `bold_discovery.py`, `pipelines/fmri_analysis.py`, and `trial_signatures.py` are still listed as modified.

- [ ] **Step 5: Merge the branch**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git merge --no-ff feat/fmri-subject-report -m "merge: land the fMRI report package on main

Consolidates the plotting foundation and subject report onto main.
No further work happens in a worktree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Expected: a clean merge. If git reports a conflict, the pathspec in step 4 missed a file — resolve by taking the branch's version, and note which file for the record.

- [ ] **Step 6: Verify the merge and the preserved work**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ -q 2>&1 | tail -5 && git status --short | grep -c "^ M" && ls fmri_pipeline/analysis/report/figures/
```

Expected: tests pass at 165 or more; `git status` still shows the unrelated modified files; `figures/` lists `_display.py`, `carpet.py`, `coverage.py`, `design.py`, `distributions.py`, `stat_maps.py`, `volumes.py`.

- [ ] **Step 7: Retire the worktree**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git worktree remove .worktrees/fmri-plotting-foundation && git worktree list | grep -c fmri
```

Expected: `0`.

---

### Task 2: Merge the two style modules

**Files:**
- Modify: `fmri_pipeline/analysis/report/style.py`
- Modify: `tests/fmri/report/test_style.py`

**Interfaces:**
- Consumes: `OKABE_ITO` from `eeg_pipeline.preprocessing.report.style`.
- Produces, in addition to what the branch's module already exports:
  - `RADIOLOGICAL: bool = False`
  - `ORIENTATION_LABEL: str`
  - `SEQUENTIAL_DECISION_CMAP: str = "Greys"`
  - `HTML_FIGURE_DPI: int = 150`, `PRINT_FIGURE_DPI: int = 300`
  - `panel_label(ax, letter) -> None`
  - `colour_limit_note(limit: float, clipped: float) -> str`

The scratchpad copy at `harvest/style.py` is the source for everything added here. Do not replace the module — the branch's version carries the measured colormap rationale, `GUIDE_COLOR`, and `COLOR_LIMIT_PERCENTILE`, all of which stay.

Two conflicts are already resolved by the spec and are not open questions:

- **Percentile stays 98.0** (the branch's), not main's 99.0.
- **DPI splits.** `HTML_FIGURE_DPI = 150` for figures embedded in the report, `PRINT_FIGURE_DPI = 300` for files written for manuscript use. Main's 150 is a measurement of a rendered report at roughly 1180 px layout width; the branch's 300 embedded needless megabytes of base64 into every document.

The branch's raise-on-empty behaviour in the limit helpers stays. Do not adopt main's `Optional[float]` returns: a colour limit that cannot be computed is a broken panel, and the established policy is that the caller renders a placeholder naming the exception. Returning `None` pushes that decision into every call site.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fmri/report/test_style.py`:

```python
def test_orientation_is_a_stated_convention_not_an_inferred_one() -> None:
    """A left/right error is invisible in the image, so the convention must be named."""
    assert style.RADIOLOGICAL is False
    assert "neurological" in style.ORIENTATION_LABEL
    assert "L on viewer left" in style.ORIENTATION_LABEL


def test_pipeline_decisions_get_a_neutral_ramp_not_a_hue() -> None:
    """Retained-vs-censored is a pipeline decision, not a measured quantity."""
    assert style.SEQUENTIAL_DECISION_CMAP == "Greys"


def test_embedded_figures_are_lighter_than_print_figures() -> None:
    """300 dpi at a 1180 px layout width embeds resolution no reader sees."""
    assert style.HTML_FIGURE_DPI == 150
    assert style.PRINT_FIGURE_DPI == 300
    assert style.FMRI_RC["savefig.dpi"] == style.HTML_FIGURE_DPI


def test_the_robust_percentile_stays_conservative() -> None:
    assert style.COLOR_LIMIT_PERCENTILE == 98.0


def test_a_colour_limit_that_cannot_be_computed_raises() -> None:
    """A caller renders a placeholder; it does not silently draw an unscaled panel."""
    import numpy as np
    import pytest

    with pytest.raises(ValueError):
        style.robust_symmetric_limit(np.array([np.nan, np.inf]))


def test_the_colour_limit_note_states_what_was_clipped() -> None:
    note = style.colour_limit_note(3.5, 0.012)
    assert "3.5" in note
    assert "1.2" in note


def test_a_panel_letter_lands_outside_the_axes() -> None:
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    try:
        style.panel_label(ax, "A")
        texts = [t for t in ax.texts if t.get_text() == "A"]
        assert len(texts) == 1
        assert texts[0].get_fontweight() == "bold"
        assert texts[0].get_position()[1] > 1.0
    finally:
        plt.close(figure)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_style.py -q 2>&1 | tail -15
```

Expected: failures with `AttributeError: module ... has no attribute 'RADIOLOGICAL'` and similar.

- [ ] **Step 3: Add the merged content to `style.py`**

Add after the existing `MAGNITUDE_CMAP` definition:

```python
#: Pipeline decisions -- retained versus censored frames, ROI usable versus not --
#: get a neutral ramp. A hue would imply the quantity is measured rather than chosen.
SEQUENTIAL_DECISION_CMAP = "Greys"

#: False is the neurological convention: subject left on the viewer's left.
#:
#: Passed explicitly to every volume plotter and named in every figure's provenance
#: line. Nilearn's own default is the same, but a figure that relies on a library
#: default states nothing, and a left/right error is not visible in the image.
RADIOLOGICAL = False
ORIENTATION_LABEL = (
    "radiological (R on viewer left)"
    if RADIOLOGICAL
    else "neurological (L on viewer left)"
)

#: Figures embedded in the HTML report. The report lays out around 1180 px wide, so
#: 300 dpi produces resolution no reader sees while base64-encoding megabytes into
#: every document.
HTML_FIGURE_DPI = 150
#: Figures written to disk for manuscript use, where the resolution is wanted.
PRINT_FIGURE_DPI = 300
```

Change `FMRI_RC["savefig.dpi"]` from `300` to `HTML_FIGURE_DPI`, and `FMRI_RC["figure.dpi"]` to `HTML_FIGURE_DPI`.

Add these functions:

```python
def colour_limit_note(limit: float, clipped: float) -> str:
    """One-line description of a colour limit and how much it hid.

    A robust colour limit deliberately saturates the extreme values. Unstated, the
    figure silently claims it did not.
    """
    return f"colour limit ±{limit:.2f} · {clipped * 100:.2f}% clipped"


def panel_label(ax: Any, letter: str) -> None:
    """Put a bold panel letter above the top-left corner of an axes, journal style."""
    ax.text(
        -0.02,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
```

Add `"HTML_FIGURE_DPI"`, `"ORIENTATION_LABEL"`, `"PRINT_FIGURE_DPI"`, `"RADIOLOGICAL"`, `"SEQUENTIAL_DECISION_CMAP"`, `"colour_limit_note"`, and `"panel_label"` to `__all__`, keeping it sorted.

- [ ] **Step 4: Replace the duplicated orientation helper in `stat_maps.py`**

`stat_maps.py` defines a private `_orientation_label(radiological)` whose two strings now
duplicate `ORIENTATION_LABEL`. Move the function into `style.py` as a public helper and have
`stat_maps.py` import it, so the two wordings cannot drift apart:

```python
def orientation_label(radiological: bool = RADIOLOGICAL) -> str:
    """Name the convention a panel was actually drawn with.

    Takes an argument rather than only reading :data:`RADIOLOGICAL`, so a panel drawn
    against a non-default convention still describes itself truthfully.
    """
    return (
        "radiological (R on viewer left)"
        if radiological
        else "neurological (L on viewer left)"
    )
```

Define `ORIENTATION_LABEL = orientation_label()` immediately after it, delete
`_orientation_label` from `stat_maps.py`, and import `orientation_label` there instead. Add
`"orientation_label"` to `style.__all__`.

- [ ] **Step 5: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_style.py tests/fmri/report/test_stat_maps.py -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/style.py fmri_pipeline/analysis/report/figures/stat_maps.py tests/fmri/report/test_style.py && git commit -m "style(fmri): merge orientation constants and split embed from print dpi

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Adopt main's design figures

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/design.py` (replaced by the scratchpad copy)
- Modify: `tests/fmri/report/test_design.py` (merged with `harvest/test_design_figures.py`)
- Modify: `fmri_pipeline/analysis/report/subject.py` (call sites)

**Interfaces:**
- Produces:
  - `classify_regressors(columns: Sequence[str]) -> Tuple[List[str], List[RegressorGroup]]`
  - `variance_inflation_factors(design: np.ndarray) -> np.ndarray`
  - `contrast_efficiency(design: np.ndarray, contrast: np.ndarray) -> Optional[float]`
  - `summarize_design(design_matrix, ...) -> DesignSummary`
  - `design_matrix_figure(design_matrix, ...) -> Figure`
  - `regressor_correlation_figure(design_matrix, *, run_label="") -> Figure`
  - `variance_inflation_figure(design_matrix, *, run_label="") -> Figure`
- Replaces: the branch's `vif_from_design` and combined `collinearity_figure`.

Main's implementation is strictly richer. `classify_regressors` gives the task/confound/drift column grouping the parent spec asked for and the branch never implemented. `contrast_efficiency` (`1 / (cᵀ (XᵀX)⁻¹ c)`) is what tells a reader whether a contrast is estimable at all. Per-column display scaling is what keeps confound regressors on unrelated scales from rendering as uniform bands.

- [ ] **Step 1: Replace the module and read both test files**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && cp /private/tmp/claude-501/-Users-joduq24-Desktop-EEG-fMRI-Pipeline/35cfdc40-5a2a-4d93-b154-4592a3f72af8/scratchpad/harvest/design.py fmri_pipeline/analysis/report/figures/design.py && wc -l fmri_pipeline/analysis/report/figures/design.py
```

Expected: 465 lines. Then read both `tests/fmri/report/test_design.py` (the branch's, 116 lines) and `harvest/test_design_figures.py` (main's) in full before writing the merged file.

- [ ] **Step 2: Write the merged test file**

Replace `tests/fmri/report/test_design.py` with a file that keeps every assertion from both. The branch's tests target `vif_from_design` and `collinearity_figure`, which no longer exist: retarget them onto `variance_inflation_factors` and `variance_inflation_figure` / `regressor_correlation_figure` rather than deleting them. Add these tests, which neither file has:

```python
def test_a_perfectly_collinear_column_is_reported_as_such() -> None:
    """VIF is the figure that shows whether a contrast is estimable at all."""
    import numpy as np

    from fmri_pipeline.analysis.report.figures import design

    rng = np.random.default_rng(0)
    a = rng.standard_normal(50)
    X = np.column_stack([a, a * 2.0, rng.standard_normal(50), np.ones(50)])
    vif = design.variance_inflation_factors(X)
    assert np.isinf(vif[0]) or vif[0] > 1e6
    assert np.isinf(vif[1]) or vif[1] > 1e6
    assert vif[2] < 5.0


def test_an_inestimable_contrast_has_no_finite_efficiency() -> None:
    import numpy as np

    from fmri_pipeline.analysis.report.figures import design

    a = np.linspace(-1, 1, 40)
    X = np.column_stack([a, a, np.ones(40)])
    efficiency = design.contrast_efficiency(X, np.array([1.0, -1.0, 0.0]))
    assert efficiency is None or not np.isfinite(efficiency) or efficiency < 1e-8


def test_regressors_are_grouped_by_role_not_left_in_column_order() -> None:
    from fmri_pipeline.analysis.report.figures import design

    columns = ["trans_x", "heat", "drift_1", "warm", "constant", "a_comp_cor_00"]
    ordered, groups = design.classify_regressors(columns)
    assert ordered.index("heat") < ordered.index("trans_x")
    assert {g.name for g in groups} >= {"Task", "Confound"}
    assert sum(g.stop - g.start for g in groups) == len(columns)
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_design.py -q 2>&1 | tail -15
```

Expected: failures naming `RegressorGroup` attributes or import errors. Adjust the assertions to the actual `RegressorGroup` field names read in step 1 — do not change the module to fit a guessed field name.

- [ ] **Step 4: Update the call sites in `subject.py`**

`build_design_section` calls the branch's `collinearity_figure`. Replace that single call with two figures, and add the contrast efficiency to the section's key-values block:

```python
        with _panel(f"regressor correlation for {manifest.contrast_name}"):
            path = _save(
                design_figures.regressor_correlation_figure(
                    design_matrix, run_label=run_label
                ),
                out_dir=plots_dir,
                stem="design_correlation",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Regressor correlation",
                        path=path,
                        dense=False,
                        caption=(
                            "Correlation between design columns. Strong off-diagonal "
                            "structure means the contrast's regressors share variance."
                        ),
                    )
                )

        with _panel(f"variance inflation for {manifest.contrast_name}"):
            path = _save(
                design_figures.variance_inflation_figure(
                    design_matrix, run_label=run_label
                ),
                out_dir=plots_dir,
                stem="design_vif",
                formats=cfg.formats,
            )
            if path:
                blocks.append(
                    html.Figure(
                        title="Variance inflation",
                        path=path,
                        dense=False,
                        caption=(
                            "Variance inflation factor per regressor. Reported as a "
                            "measurement; no cutoff is applied."
                        ),
                    )
                )
```

- [ ] **Step 5: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/figures/design.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_design.py && git rm -q --ignore-unmatch tests/fmri/report/test_design_figures.py && git commit -m "feat(fmri): adopt regressor role grouping and contrast efficiency

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Harvest drift removal into tSNR

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/volumes.py`
- Test: `tests/fmri/report/test_volumes.py`

**Interfaces:**
- Produces: `detrended_temporal_sd(data: np.ndarray, mask: np.ndarray) -> np.ndarray`, used by the existing tSNR computation in this module.

The branch censors non-steady-state frames before computing temporal standard deviation, which the parent spec identified. It does not remove scanner drift, which inflates the same statistic for a different reason. Drift is not thermal or physiological noise and the GLM's cosine high-pass removes it, so leaving it in reports a tSNR lower than the one the model actually works with. **This corrects a reported numeric value, not an appearance.** Source: `harvest/reporting.py`, function `_detrended_temporal_sd`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_volumes.py`:

```python
def test_linear_drift_does_not_inflate_the_temporal_standard_deviation() -> None:
    """Drift is removed by the GLM's high-pass, so leaving it in under-reports tSNR."""
    import numpy as np

    from fmri_pipeline.analysis.report.figures import volumes

    rng = np.random.default_rng(0)
    n_frames = 60
    noise = rng.standard_normal((2, 2, 2, n_frames)) * 0.5
    drift = np.linspace(0.0, 20.0, n_frames)
    data = noise + drift
    mask = np.ones((2, 2, 2), dtype=bool)

    plain = np.std(data, axis=3)
    detrended = volumes.detrended_temporal_sd(data, mask)

    assert plain.mean() > 5.0, "the fixture must actually carry drift"
    assert detrended.mean() < 1.0
    assert np.allclose(detrended, 0.5, atol=0.2)


def test_detrending_degrades_gracefully_on_a_very_short_run() -> None:
    """Fewer frames than basis functions cannot be detrended; report the plain sd."""
    import numpy as np

    from fmri_pipeline.analysis.report.figures import volumes

    data = np.ones((2, 2, 2, 3), dtype=float)
    mask = np.ones((2, 2, 2), dtype=bool)
    result = volumes.detrended_temporal_sd(data, mask)
    assert result.shape == (2, 2, 2)
    assert np.all(np.isfinite(result))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_volumes.py -q 2>&1 | tail -10
```

Expected: `AttributeError: module ... has no attribute 'detrended_temporal_sd'`.

- [ ] **Step 3: Add the function**

```python
def detrended_temporal_sd(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Temporal standard deviation after removing low-order drift.

    Scanner drift is not thermal or physiological noise, and the GLM's cosine
    high-pass removes it, so leaving it in the temporal standard deviation reports a
    tSNR lower than the one the model actually works with. A cubic polynomial basis
    captures the drift the high-pass removes without needing the run's exact cutoff.
    """
    n_frames = data.shape[3]
    if n_frames < 4:
        # Fewer frames than basis functions: no drift estimate is possible, and
        # fitting one would consume the signal instead.
        return np.std(data, axis=3)

    time = np.linspace(-1.0, 1.0, n_frames, dtype=np.float64)
    basis = np.vstack([np.ones_like(time), time, time**2, time**3]).T

    series = data[mask].astype(np.float64).T  # (frames, voxels)
    if series.size == 0:
        return np.std(data, axis=3)

    beta, *_ = np.linalg.lstsq(basis, series, rcond=None)
    residual = series - basis @ beta

    out = np.zeros(data.shape[:3], dtype=np.float64)
    out[mask] = residual.std(axis=0)
    return out
```

Then find the existing temporal-standard-deviation computation in this module and route it through `detrended_temporal_sd`. Add `"detrended_temporal_sd"` to `__all__`.

- [ ] **Step 4: State the correction on the figure**

The tSNR figure's provenance line must say drift was removed, otherwise the reported value cannot be compared against a tSNR computed any other way. Add to the provenance list passed to `annotate_provenance`:

```python
"tSNR: cubic drift removed, non-steady-state frames excluded",
```

- [ ] **Step 5: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_volumes.py -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/figures/volumes.py tests/fmri/report/test_volumes.py && git commit -m "fix(fmri): remove scanner drift before computing tSNR

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Harvest the glass-brain space guard

**Files:**
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_subject.py`

**Interfaces:**
- Produces: `supports_glass_brain(space: str) -> bool` in `subject.py`.

`subject.py` draws the glass brain unconditionally. The projection is defined only against the MNI schematic, so a native-space map rendered on it is wrong rather than approximate — voxels are projected onto anatomy they do not correspond to. Source: `harvest/reporting.py`, function `_supports_glass_brain`.

The panel is omitted when the space does not support it, and **the omission is stated**. A silently missing panel reads as a rendering failure.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_subject.py`:

```python
def test_a_native_space_contrast_gets_no_glass_brain(tmp_path: Path) -> None:
    """The projection is defined only against the MNI schematic."""
    manifest = _manifest(tmp_path, space="native")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=FmriReportConfig(enabled=True)
    )
    titles = [getattr(b, "title", "") for b in section.blocks]
    assert not any("Glass brain" in t for t in titles)


def test_the_missing_glass_brain_is_explained_rather_than_silent(tmp_path: Path) -> None:
    """A panel that vanishes without a word reads as a rendering failure."""
    manifest = _manifest(tmp_path, space="native")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=FmriReportConfig(enabled=True)
    )
    text = " ".join(getattr(b, "text", "") for b in section.blocks)
    assert "glass brain" in text.lower()
    assert "mni" in text.lower()


def test_an_mni_contrast_still_gets_a_glass_brain(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, space="mni")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=FmriReportConfig(enabled=True)
    )
    titles = [getattr(b, "title", "") for b in section.blocks]
    assert any("Glass brain" in t for t in titles)
```

If `_manifest` in this file does not already accept a `space` override, extend its `**overrides` handling rather than duplicating the helper.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_subject.py -q -k glass 2>&1 | tail -10
```

Expected: the native-space test fails because a glass brain is drawn.

- [ ] **Step 3: Add the guard**

In `subject.py`:

```python
def supports_glass_brain(space: str) -> bool:
    """Whether a glass-brain projection is defined for ``space``.

    The projection is drawn against a fixed MNI schematic. A map in native or T1w
    space projected onto it lands on anatomy it does not correspond to, which is an
    error rather than an approximation.
    """
    return str(space or "").strip().lower() == "mni"
```

Wrap the existing glass-brain block at the `with _panel(f"glass brain for ...")` call site:

```python
        if supports_glass_brain(manifest.space):
            with _panel(f"glass brain for {manifest.contrast_name}"):
                ...  # existing body, unchanged
        else:
            blocks.append(
                html.Note(
                    text=(
                        f"No glass brain: the projection is defined only against the "
                        f"MNI schematic, and this contrast is in "
                        f"{manifest.space} space."
                    )
                )
            )
```

Add `"supports_glass_brain"` to `__all__`.

- [ ] **Step 4: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_subject.py -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject.py && git commit -m "fix(fmri): draw a glass brain only where the projection is defined

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Label cluster coordinates with their actual space

**Files:**
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_subject.py`

**Interfaces:**
- Produces: `coordinate_space_label(space: str) -> str` in `subject.py`.

The branch's cluster tables carry no coordinate-space label. A reader who sees an `X / Y / Z` column in an fMRI cluster table will read it as MNI, because that is the overwhelming convention. For a native-space contrast that is a silent misreport, and it is not recoverable from the figure. Source: `harvest/reporting.py`, functions `_label_cluster_coordinate_space` and `_coordinate_space_caption`.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_subject.py`:

```python
def test_native_space_cluster_coordinates_are_not_left_to_read_as_mni(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, space="native")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=FmriReportConfig(enabled=True)
    )
    captions = " ".join(getattr(b, "caption", "") for b in section.blocks)
    assert "native" in captions.lower()
    assert "not MNI" in captions


def test_mni_cluster_coordinates_say_so(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, space="mni")
    section = subject.build_contrast_section(
        manifest=manifest, out_dir=tmp_path, cfg=FmriReportConfig(enabled=True)
    )
    captions = " ".join(getattr(b, "caption", "") for b in section.blocks)
    assert "MNI" in captions
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_subject.py -q -k coordinate 2>&1 | tail -10
```

- [ ] **Step 3: Add the label**

```python
def coordinate_space_label(space: str) -> str:
    """Name the space a cluster table's coordinates are actually in.

    An unlabelled X/Y/Z column in an fMRI cluster table reads as MNI by convention.
    For a native-space contrast that is a silent misreport a reader cannot detect.
    """
    if str(space or "").strip().lower() == "mni":
        return "MNI152 (mm)"
    return f"{space} scanner-native (mm); not MNI"
```

Include it in the cluster table's caption where `build_cluster_table`'s result is turned into an `html.Table`, alongside the existing threshold and extent-filter text. Keep the existing separation of height threshold from extent filter — the extent filter must not read as inference.

Add `"coordinate_space_label"` to `__all__`.

- [ ] **Step 4: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_subject.py -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject.py && git commit -m "fix(fmri): name the coordinate space of every cluster table

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Signature expression dot plot

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/signatures.py`
- Create: `tests/fmri/report/test_signatures.py`
- Modify: `fmri_pipeline/analysis/report/subject.py`

**Interfaces:**
- Produces:
  - `signature_dot_plot(results: Sequence[SignaturePoint], *, metric: str = "cosine", title: str = "") -> Figure`
  - `@dataclass(frozen=True) class SignaturePoint` with fields `name: str`, `dot: float`, `cosine: Optional[float]`, `pearson_r: Optional[float]`, `n_voxels: int`
  - `build_signature_section(*, manifest, out_dir, cfg, results) -> Optional[Section]` in `subject.py`
- Consumes: `SignatureResult` from `fmri_pipeline.analysis.multivariate_signatures`, whose fields are `name`, `dot`, `cosine`, `pearson_r`, `n_voxels`, `weight_path`.

Signature expression is currently a five-column table. The comparison *across* signatures is the point, and a table does not show it. `SignaturePoint` exists so the figure module stays free of the analysis module's types, per the constraint that `figures/` takes arrays and plain records only.

Plot `cosine` by default rather than `dot`: the dot product scales with the effect map's units, so two subjects are not comparable on it, while cosine similarity is bounded and unitless. Draw a zero reference line, because sign is the whole interpretation.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_signatures.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from fmri_pipeline.analysis.report.figures.signatures import (
    SignaturePoint,
    signature_dot_plot,
)


def _points() -> list[SignaturePoint]:
    return [
        SignaturePoint(name="NPS", dot=12.0, cosine=0.31, pearson_r=0.28, n_voxels=9000),
        SignaturePoint(name="SIIPS", dot=-4.0, cosine=-0.12, pearson_r=-0.10, n_voxels=8800),
        SignaturePoint(name="PINES", dot=0.5, cosine=0.02, pearson_r=0.01, n_voxels=9100),
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
        positions = [line.get_xdata()[0] for line in figure.axes[0].lines]
        assert any(abs(float(p)) < 1e-9 for p in positions)
    finally:
        plt.close(figure)


def test_cosine_is_the_default_metric_not_the_dot_product() -> None:
    """The dot product carries the effect map's units, so it is not comparable."""
    figure = signature_dot_plot(_points())
    try:
        assert "cosine" in figure.axes[0].get_xlabel().lower()
    finally:
        plt.close(figure)


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


def test_no_usable_signature_raises_rather_than_drawing_an_empty_panel() -> None:
    with pytest.raises(ValueError):
        signature_dot_plot([])


def test_the_panel_states_how_many_voxels_it_summarises() -> None:
    figure = signature_dot_plot(_points())
    try:
        text = " ".join(t.get_text() for t in figure.texts)
        assert "9,000" in text or "8,800" in text or "voxel" in text.lower()
    finally:
        plt.close(figure)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_signatures.py -q 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

`fmri_pipeline/analysis/report/figures/signatures.py`:

```python
"""Signature expression across signatures, as a figure rather than a table.

Expression currently reaches the report as a five-column table. The comparison
across signatures is what a reader is actually doing, and a table makes them do it
by arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import (
    GUIDE_COLOR,
    OKABE_ITO,
    annotate_provenance,
    plot_context,
)

#: Metrics this panel can draw, and whether each is comparable across subjects.
_METRIC_LABELS = {
    "cosine": "Cosine similarity (unitless)",
    "pearson_r": "Pearson r",
    "dot": "Dot product (effect-map units; not comparable across subjects)",
}


@dataclass(frozen=True)
class SignaturePoint:
    """One signature's expression. Plain record, so `figures/` stays decoupled."""

    name: str
    dot: float
    cosine: Optional[float]
    pearson_r: Optional[float]
    n_voxels: int


def signature_dot_plot(
    results: Sequence[SignaturePoint],
    *,
    metric: str = "cosine",
    title: str = "",
) -> Any:
    """Draw each signature's expression on a shared axis.

    ``cosine`` is the default because the dot product scales with the effect map's
    units: two subjects cannot be compared on it, and a reader has no way to see
    that from the number. A signature whose metric is missing is dropped rather than
    drawn at zero -- an unmeasured similarity is not a similarity of zero.
    """
    if metric not in _METRIC_LABELS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(_METRIC_LABELS)}."
        )

    usable: List[SignaturePoint] = [
        point
        for point in results
        if getattr(point, metric) is not None
        and np.isfinite(float(getattr(point, metric)))
    ]
    if not usable:
        raise ValueError(f"No signature carried a finite {metric} to plot.")

    names = [point.name for point in usable]
    values = np.array([float(getattr(point, metric)) for point in usable])
    positions = np.arange(len(usable))

    with plot_context():
        figure, ax = plt.subplots(figsize=(6.4, 0.42 * len(usable) + 1.5))

        # Sign is the entire interpretation of a signature score, so zero is drawn
        # first and the marks are coloured by which side of it they fall on.
        ax.axvline(0.0, color=GUIDE_COLOR, linewidth=0.9, zorder=1)
        colours = [
            OKABE_ITO["vermillion"] if value >= 0 else OKABE_ITO["blue"]
            for value in values
        ]
        ax.hlines(
            positions, 0.0, values, color=colours, linewidth=1.4, alpha=0.65, zorder=2
        )
        ax.scatter(values, positions, s=46, c=colours, zorder=3)

        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(_METRIC_LABELS[metric])
        if title:
            ax.set_title(title)

        voxels = {point.n_voxels for point in usable}
        voxel_note = (
            f"n = {next(iter(voxels)):,} voxels"
            if len(voxels) == 1
            else f"n = {min(voxels):,}–{max(voxels):,} voxels"
        )
        dropped = len(results) - len(usable)
        provenance = [voxel_note, f"metric: {metric}"]
        if dropped:
            provenance.append(f"{dropped} signature(s) had no {metric}")
        annotate_provenance(figure, provenance)
        figure.tight_layout()
        return figure


__all__ = ["SignaturePoint", "signature_dot_plot"]
```

- [ ] **Step 4: Run the test**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_signatures.py -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 5: Add the section to `subject.py`**

```python
def build_signature_section(
    *,
    manifest: ContrastManifest,
    out_dir: Path,
    cfg: FmriReportConfig,
    results: Sequence[Any],
) -> Optional[html.Section]:
    """Signature expression as a dot plot plus the existing table.

    Returns ``None`` when no signatures are configured, which is the stock
    configuration rather than a misconfiguration.
    """
    if not results:
        return None

    from fmri_pipeline.analysis.report.figures import signatures as signature_figures

    points = [
        signature_figures.SignaturePoint(
            name=r.name,
            dot=float(r.dot),
            cosine=None if r.cosine is None else float(r.cosine),
            pearson_r=None if r.pearson_r is None else float(r.pearson_r),
            n_voxels=int(r.n_voxels),
        )
        for r in results
    ]

    plots_dir = out_dir / "plots"
    blocks: List[html.Block] = []
    with _panel(f"signature expression for {manifest.contrast_name}"):
        path = _save(
            signature_figures.signature_dot_plot(
                points, title=f"{manifest.contrast_name}: signature expression"
            ),
            out_dir=plots_dir,
            stem="signature_expression",
            formats=cfg.formats,
        )
        if path:
            blocks.append(
                html.Figure(
                    title="Signature expression",
                    path=path,
                    dense=False,
                    caption=(
                        "Cosine similarity between the unthresholded effect map and "
                        "each signature's weight map. Sign carries the "
                        "interpretation; no threshold is applied."
                    ),
                )
            )
    return html.Section(
        slug=f"signatures-{_slug(manifest)}",
        title=f"Signatures · {manifest.contrast_name}",
        blocks=tuple(blocks),
    )
```

Call it from `build_subject_report` after `build_contrast_section`, guarded by whether signature results are available for that manifest. Add `"build_signature_section"` to `__all__`.

- [ ] **Step 6: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/report/figures/signatures.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_signatures.py && git commit -m "feat(fmri): draw signature expression instead of tabulating it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Write the manifest from the analysis run

**Files:**
- Modify: `fmri_pipeline/analysis/contrast_builder.py`
- Test: `tests/fmri/report/test_manifest_writing.py`

**Interfaces:**
- Consumes: `ContrastManifest`, `write_manifest`, `MANIFEST_FILENAME` from `fmri_pipeline.analysis.report.manifest`.
- Produces: a `report_manifest.json` beside each contrast's stat maps.

This is the seam. Until the analysis run writes the manifest, the report can only be produced from inside the GLM path, and task 9's decoupling has nothing to read.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_manifest_writing.py`:

```python
from __future__ import annotations

from pathlib import Path

from fmri_pipeline.analysis.report.manifest import MANIFEST_FILENAME, read_manifest


def test_a_fitted_contrast_leaves_a_manifest_beside_its_stat_maps(
    tmp_path: Path,
) -> None:
    """Without this the report cannot be rebuilt without refitting."""
    from fmri_pipeline.analysis import contrast_builder

    contrast_dir = tmp_path / "contrast-heatgtwarm"
    contrast_dir.mkdir(parents=True)
    written = contrast_builder.write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01",
        task="heat",
        contrast_name="heat>warm",
        space="native",
        stat_map=contrast_dir / "z.nii.gz",
        effect_map=None,
        variance_map=None,
        mask=None,
        threshold_mode="z",
        z_threshold=2.3,
        fdr_q=0.05,
        cluster_min_voxels=0,
        two_sided=True,
        radiological=False,
        design_matrices=(),
        contrast_vector=None,
        contrast_columns=(),
        included_runs=("run-01",),
        excluded_runs=(("run-02", "mean FD 0.9 mm"),),
        bold_paths=(),
        confounds_paths=(),
        t_r=2.0,
        smoothing_fwhm=6.0,
        signal_scaling=True,
        confound_strategy="24HMP+aCompCor",
    )
    assert written.name == MANIFEST_FILENAME
    assert written.parent == contrast_dir

    restored = read_manifest(written)
    assert restored.subject == "sub-01"
    assert restored.excluded_runs == (("run-02", "mean FD 0.9 mm"),)
    assert restored.t_r == 2.0


def test_the_manifest_records_why_a_run_was_excluded(tmp_path: Path) -> None:
    """A report that says a run was dropped without saying why gives nothing to act on."""
    from fmri_pipeline.analysis import contrast_builder

    contrast_dir = tmp_path / "c"
    contrast_dir.mkdir()
    written = contrast_builder.write_report_manifest(
        contrast_dir=contrast_dir,
        subject="sub-01", task="heat", contrast_name="c", space="native",
        stat_map=contrast_dir / "z.nii.gz", effect_map=None, variance_map=None,
        mask=None, threshold_mode="z", z_threshold=2.3, fdr_q=0.05,
        cluster_min_voxels=0, two_sided=True, radiological=False,
        design_matrices=(), contrast_vector=None, contrast_columns=(),
        included_runs=(), excluded_runs=(("run-03", "no events"),),
        bold_paths=(), confounds_paths=(), t_r=None, smoothing_fwhm=None,
        signal_scaling=False, confound_strategy="none",
    )
    assert read_manifest(written).excluded_runs[0][1] == "no events"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_manifest_writing.py -q 2>&1 | tail -10
```

Expected: `AttributeError: module ... has no attribute 'write_report_manifest'`.

- [ ] **Step 3: Add the writer to `contrast_builder.py`**

```python
def write_report_manifest(*, contrast_dir: Path, **fields: Any) -> Path:
    """Record what was fit, beside what was fit.

    The report reads manifests and nothing else, which is what lets a document be
    regenerated from a derivatives tree without the model. Keyword-only and
    exhaustive by design: a field that silently defaults here becomes a field the
    report silently misreports.
    """
    from fmri_pipeline.analysis.report.manifest import (
        MANIFEST_FILENAME,
        ContrastManifest,
        write_manifest,
    )

    manifest = ContrastManifest(**fields)
    return write_manifest(manifest, Path(contrast_dir) / MANIFEST_FILENAME)
```

Then call it from the existing per-contrast write path, populating every field from the values already in scope there. Read the surrounding function before writing the call — do not invent field sources.

- [ ] **Step 4: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ -q 2>&1 | tail -8
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add fmri_pipeline/analysis/contrast_builder.py tests/fmri/report/test_manifest_writing.py && git commit -m "feat(fmri): write a report manifest beside every fitted contrast

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Wire the report mode and cut the pipeline's plotting call

**Files:**
- Modify: `fmri_pipeline/cli/commands/fmri_analysis.py`
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py`
- Modify: `fmri_pipeline/analysis/reporting.py`
- Test: `tests/fmri/report/test_report_entry_point.py`

**Interfaces:**
- Produces: `fmri-analysis report --subject … --task …`, which discovers manifests under the derivatives root and calls `build_subject_report`.

This establishes the invariant the whole design exists for: **no rendering setting may cause a GLM to be fit.** The first test is the one that proves it.

- [ ] **Step 1: Write the failing test**

`tests/fmri/report/test_report_entry_point.py`:

```python
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_rendering_a_report_never_imports_the_model_fitting_modules() -> None:
    """The invariant the design exists to establish.

    If importing the report path pulls in contrast_builder or nilearn's GLM, then
    rendering can still trigger fitting and the decoupling is nominal.
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
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_report_mode_is_registered() -> None:
    """``mode`` is a positional with a choices list; ``report`` joins it."""
    import argparse

    from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    setup_fmri_analysis(sub)
    args = parser.parse_args(
        ["fmri-analysis", "report", "--subject", "0001", "--task", "heat"]
    )
    assert args.mode == "report"
    # --subject uses action="append", so this is a list.
    assert args.subject == ["0001"]
    assert args.task == "heat"


def test_the_glm_path_no_longer_calls_into_plotting() -> None:
    """Subject QC was previously recomputed once per contrast because of this call."""
    from pathlib import Path

    source = Path("fmri_pipeline/pipelines/fmri_analysis.py").read_text(encoding="utf-8")
    assert "run_fmri_plotting_and_report" not in source
```

Adjust the argparse invocation in the second test to match this CLI's actual structure — read `setup_fmri_analysis` first and mirror how existing modes are registered.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/test_report_entry_point.py -q 2>&1 | tail -12
```

- [ ] **Step 3: Add the `report` mode**

`mode` is already a positional argument with a `choices` list at `fmri_pipeline/cli/commands/fmri_analysis.py:34`, and `add_common_subject_args`, `add_task_arg`, and `add_path_args` already supply `--subject`, `--task`, and `--deriv-root`. So registration is one edit:

```python
    parser.add_argument(
        "mode",
        choices=["first-level", "second-level", "beta-series", "lss", "rest", "report"],
        help=(
            "Operation to run (first-level | second-level | beta-series | lss | "
            "rest | report). 'report' renders from existing derivatives and never "
            "fits a model."
        ),
    )
```

Add one new flag, because the derivatives root is frequently a read-only or removable volume and a report must be renderable without writing to it:

```python
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help=(
            "Write reports here instead of beside the derivatives. Use when the "
            "derivatives root is read-only or on a removable volume."
        ),
    )
```

Then add the dispatch in `run_fmri_analysis`, before the branches that construct a pipeline:

```python
    if args.mode == "report":
        _run_report_mode(args, config)
        return
```

And the handler, in the same module:

```python
def _run_report_mode(args: argparse.Namespace, config: Any) -> None:
    """Render subject reports from derivatives. Fits nothing.

    Imported lazily and locally so that nothing on the report path pulls the GLM
    modules into ``sys.modules`` -- which is the invariant the first test asserts.
    """
    from fmri_pipeline.analysis.plotting_config import FmriReportConfig
    from fmri_pipeline.analysis.report.manifest import discover_manifests
    from fmri_pipeline.analysis.report.subject import build_subject_report

    deriv_root = Path(args.deriv_root) if args.deriv_root else Path(config["deriv_root"])
    task = resolve_task(args, config)
    subjects = resolve_subjects(args, config)
    if not subjects:
        raise SystemExit("No subjects selected; pass --subject, --group, or --all-subjects.")

    report_cfg = FmriReportConfig(enabled=True, html_report=True)
    for subject in subjects:
        sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
        manifests = discover_manifests(
            deriv_root=deriv_root, subject=sub_label, task=task
        )
        if not manifests:
            # No fitted contrast is a state of the derivatives tree, not a fault.
            # Reporting it and continuing keeps one unfitted subject from ending a
            # cohort run.
            logger.warning(
                "No report manifests for %s task-%s under %s; skipping.",
                sub_label, task, deriv_root,
            )
            continue
        report_root = Path(args.report_dir) if args.report_dir else deriv_root
        out_path = (
            report_root / sub_label / "fmri" / f"{sub_label}_task-{task}_report.html"
        )
        written = build_subject_report(
            manifests=manifests,
            deriv_root=deriv_root,
            out_path=out_path,
            cfg=report_cfg,
        )
        logger.info("Wrote %s", written)
```

Adjust `config["deriv_root"]` and `resolve_subjects`/`resolve_task` call shapes to match how the existing modes in this function read them — read those branches first rather than guessing.

- [ ] **Step 4: Remove the plotting call from the GLM path**

Delete the `run_fmri_plotting_and_report` call at `fmri_pipeline/pipelines/fmri_analysis.py:324` and the import beside it. Then remove `run_fmri_plotting_and_report` and the orchestration that moved to `subject.py` from `reporting.py`, keeping only what other modules still import. Check with:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && grep -rn "run_fmri_plotting_and_report\|generate_fmri_space_section\|write_fmri_report" --include="*.py" . | grep -v "\.venv\|analysis/reporting.py"
```

Every hit must be updated or deleted before this task is done.

- [ ] **Step 5: Run the tests**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ tests/fmri/test_fmri_analysis_validity_guards.py -q 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && git add -A fmri_pipeline tests/fmri && git commit -m "refactor(fmri): render reports from derivatives, not from the GLM path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Render a real subject and inspect every panel

**Files:**
- Create: `outputs/fmri_report_verification/` (rendered output; not committed)
- Modify: whichever figure modules the rendering exposes as defective

**Interfaces:**
- Consumes: everything built above.
- Produces: a written record of what each panel looked like, and fixes for what was wrong.

No figure this pipeline produces has ever been looked at against real BOLD data. `/Volumes/KINGSTON/EEG_fMRI_data/derivatives` holds fMRIPrep output for roughly fifteen subjects and not one analysis-stage report. Every quality claim in this plan is, until this task runs, a claim about synthetic `nibabel` fixtures.

**The derivatives tree is read-only input. Write nothing to `/Volumes/KINGSTON`.**

- [ ] **Step 1: Choose a subject with complete task derivatives**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && for s in /Volumes/KINGSTON/EEG_fMRI_data/derivatives/sub-*/fmri; do echo "== $s"; ls "$s" 2>/dev/null | head -5; done 2>/dev/null | head -60
```

Pick a subject with fitted contrasts and confounds present. Record which one and why in the task notes.

- [ ] **Step 2: Render the report**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && mkdir -p outputs/fmri_report_verification && .venv/bin/python -m eeg_pipeline.cli.main fmri-analysis report --subject <CHOSEN> --task <TASK> --deriv-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives --report-dir outputs/fmri_report_verification 2>&1 | tail -30
```

`--report-dir` is what keeps the drive read-only: manifests and stat maps are read from `--deriv-root`, and every byte written lands under `outputs/`.

If no fitted contrasts exist for any subject, fit one first with the smoketest config, directing its derivatives into `outputs/fmri_report_verification/derivatives` rather than onto the drive. Note in the task record that the figures were then checked against a locally fitted model rather than the existing derivatives.

- [ ] **Step 3: Inspect every panel**

Open each rendered figure with the Read tool, which displays images. Check, per panel:

- **Thresholded mosaic** — slice coordinates and L/R markers present; colour limit above the threshold; colorbar labelled `z`.
- **Dual-coded mosaic** — sub-threshold structure visible as faded rather than absent.
- **Glass brain** — present only for MNI; activation and deactivation visibly different.
- **Carpet** — GM/WM/CSF blocks banded and distinguishable; FD trace gapped at run boundaries rather than dipping to zero; run labels legible.
- **tSNR** — orientation correct; median annotated; values plausible (typical fMRIPrep-preprocessed tSNR is in the tens to low hundreds — a value near 1 or above 1000 means the drift correction or the mask is wrong).
- **Design matrix** — regressor labels legible; task/confound/drift groups visually separated.
- **VIF and correlation** — bars readable; no clipped labels.
- **Signature dot plot** — if signatures are configured.
- **Every provenance line** — sample size, threshold, colour limit, clipped fraction, orientation actually rendered and not overlapping the axes.

- [ ] **Step 4: Fix what is wrong, with a test for each fix**

For every defect found, write a failing test against a synthetic fixture that reproduces it, fix it, and confirm. Do not fix a rendering defect without a test — an unreproduced fix is a fix that regresses.

- [ ] **Step 5: Report findings**

Write up: the subject and task used, each panel's verdict, every defect found and its fix. Include the rendered report path so the user can open it.

- [ ] **Step 6: Commit the fixes**

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline && .venv/bin/python -m pytest tests/fmri/report/ -q 2>&1 | tail -5 && git add fmri_pipeline tests/fmri && git commit -m "fix(fmri): correct report figures found defective on real data

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**Recovering main's restart.** Everything removed in task 1 step 4 is in `git stash@{0}` under the message `main-fmri-report-restart`, and the whole pre-consolidation working tree is at tag `wip/pre-fmri-consolidation`. Read a file from either without disturbing the working tree:

```bash
git show 'stash@{0}:fmri_pipeline/analysis/reporting.py' | head -50
```

**Do not drop the stash or delete the tag** as part of this plan. Leave both for the user.

**If the merge in task 1 conflicts**, the pathspec missed a file. Resolve by taking the branch's version, note which file, and carry on — the harvest tasks are where main's content is reintroduced deliberately.
