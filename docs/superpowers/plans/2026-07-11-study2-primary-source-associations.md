# Study 2 Primary Cortical Source Associations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the publication-ready Study 2 cortical source-association figure, corrected-cluster overlays, complete audits, and exact fsaverage vertex provenance.

**Architecture:** Preserve common-space vertex identity at source extraction, then use a strict artifact reader to recompute and reconcile the existing source-family inference. Keep statistics, surface geometry, rendering, and output writing in separate focused modules; the figure displays unthresholded Fisher-mean partial correlations on three matched cortical maps and outlines only jointly cluster- and Holm-corrected vertices.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, MNE-Python, NiBabel, Nilearn, Matplotlib, Pytest, YAML.

---

### Task 1: Preserve common-space source vertices

**Files:**
- Create: `studies/pain_study/study2/source_vertex_manifest.py`
- Modify: `studies/pain_study/study2/source_power.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/pain_study/study2/paths.py`
- Test: `studies/tests/pipelines/test_study2_source_vertex_manifest.py`
- Test: `studies/tests/pipelines/test_study2_source_power.py`
- Test: `studies/tests/pipelines/test_study2_stages.py`

- [ ] **Step 1: Write failing vertex-provenance tests**

Add tests asserting that a morphed extraction returns exact left/right vertex arrays, that
the manifest writer creates `.npz` and `.json` artifacts, and that a second incompatible
vertex definition raises rather than overwriting the manifest.

```python
assert tuple(vertices.tolist() for vertices in result.vertices) == ([2, 5], [1, 4])
manifest = ensure_common_source_vertices(...)
np.testing.assert_array_equal(manifest.lh_vertices, [2, 5])
with pytest.raises(ValueError, match="does not match"):
    ensure_common_source_vertices(... incompatible vertices ...)
```

- [ ] **Step 2: Verify the provenance tests fail for the missing API**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study2_source_vertex_manifest.py studies/tests/pipelines/test_study2_source_power.py studies/tests/pipelines/test_study2_stages.py -q
```

Expected: collection or assertion failures because the manifest module, extraction vertex
field, and stage outputs do not yet exist.

- [ ] **Step 3: Implement strict vertex capture and manifest writing**

Create an immutable `CommonSourceVertices` record with validated non-empty, unique,
non-negative `lh_vertices` and `rh_vertices`. Add `MorphedSourcePowerExtraction`, capture
the exact `morphed.vertices` values while morphing scalar maps, validate identity across
trials, and have `run_source_power` call `ensure_common_source_vertices` for every
extraction. Add explicit helpers for:

```python
source_vertex_manifest_path(config)
source_vertex_metadata_path(config)
```

The writer creates both files atomically on the first extraction and validates exact
identity plus `common_subject` and `spacing` thereafter.

- [ ] **Step 4: Verify source provenance passes**

Run the Task 1 command and expect all selected tests to pass.

### Task 2: Build the strict statistical artifact reader

**Files:**
- Create: `studies/pain_study/study2/figures/primary_source_associations.py`
- Modify: `studies/pain_study/study2/paths.py`
- Test: `studies/tests/pipelines/test_study2_primary_source_associations.py`

- [ ] **Step 1: Write failing tests for the scientific estimand and validation contract**

Create synthetic alpha, beta, and gamma artifacts and assert:

```python
expected = np.tanh(np.mean(fisher_z_maps, axis=0))
np.testing.assert_allclose(summary.bands[0].effect_r, expected)
assert summary.display_limit == pytest.approx(max_abs_effect)
assert summary.bands[0].corrected_vertex_mask.tolist() == expected_mask
```

Add independent failing tests for inconsistent partial-r/Fisher-z arrays, different valid
participant order across bands, an invalid manifest dimension, a stale saved family
summary, and article cohorts below `source_stage.min_source_valid_subjects`.

- [ ] **Step 2: Verify reader tests fail because the module is absent**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study2_primary_source_associations.py -q
```

Expected: failure because `load_primary_source_associations` and its data structures are
not implemented.

- [ ] **Step 3: Implement reader, inference reconciliation, and audits**

Implement immutable `PrimarySourceBand` and `PrimarySourceAssociations` records. Read the
three bands in configured order, recover valid subject IDs from QC, validate every array,
recompute `compute_source_family_inference`, and compare its summary with the saved TSV.
For every cluster, define contour membership as:

```python
contour = cluster.p_value <= family_alpha and band_result.holm_q_value <= family_alpha
```

Build stable vertex-, cluster-, and band-summary tables. The displayed effect is exactly
`tanh(mean(fisher_z, axis=0))`, and the single display limit is the full pooled maximum
absolute effect.

- [ ] **Step 4: Verify reader tests pass**

Run the Task 2 command and expect all reader tests to pass.

### Task 3: Load exact surface geometry and render the fixed figure

**Files:**
- Create: `studies/pain_study/study2/figures/primary_source_associations_plot.py`
- Modify: `studies/pain_study/study2/config/study2_config.yaml`
- Test: `studies/tests/pipelines/test_study2_primary_source_associations_figure.py`

- [ ] **Step 1: Write failing renderer tests**

Use small synthetic bilateral triangular surfaces and a synthetic validated summary. Assert
the exact 183 mm figure width and configured height, three ordered band headers, four views
per band, one shared symmetric color bar, unthresholded mapped collections, corrected
contours only where the mask is true, and explicit no-cluster text where appropriate.

```python
assert np.allclose(figure.get_size_inches(), (183.0 / 25.4, 100.0 / 25.4))
assert len(surface_axes(figure)) == 12
assert figure.axes[-1].get_xlabel() == "Fisher mean partial correlation, r"
```

- [ ] **Step 2: Verify renderer tests fail because the renderer is absent**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study2_primary_source_associations_figure.py -q
```

Expected: failure because surface loading and the fixed renderer do not exist.

- [ ] **Step 3: Implement surface geometry and rendering**

Add a strict `load_common_source_surfaces` function that uses the configured fsaverage
inflated geometry, sulcal data, and MNE common source-space triangulation; it must validate
the source-space vertex numbers against the saved manifest. Render alpha, beta, and gamma
as matched four-view cortical columns using one balanced blue-white-orange scale centered
at zero. Show sulcal anatomy quietly beneath the complete unthresholded effect and draw
thin charcoal contours for corrected masks. Add a single external color bar and compact
band-level corrected status; no ROI annotations or over-data legend.

- [ ] **Step 4: Verify renderer tests pass and inspect a synthetic render**

Run the Task 3 command, save a temporary 600-dpi PNG from the synthetic fixture, and inspect
it for overlaps, clipping, cortical orientation, color balance, and contour legibility.

### Task 4: Write the complete figure family and provenance

**Files:**
- Create: `studies/pain_study/study2/figures/plot_primary_source_associations.py`
- Modify: `studies/pain_study/study2/figures/style.py`
- Modify: `studies/pain_study/study2/paths.py`
- Modify: `studies/pain_study/study2/figures/__init__.py`
- Test: `studies/tests/pipelines/test_study2_primary_source_associations_figure.py`

- [ ] **Step 1: Write failing end-to-end writer tests**

Assert that one call writes:

```python
{
    "primary_source_associations.svg",
    "primary_source_associations.png",
    "primary_source_associations_vertices.tsv",
    "primary_source_associations_clusters.tsv",
    "primary_source_associations_summary.tsv",
    "primary_source_associations_caption.txt",
    "primary_source_associations_manifest.json",
}
```

Parse the SVG to verify physical dimensions and editable text, validate exact audit schemas,
validate caption correction language, and verify manifest input/output hashes. Add tests
that a non-SVG override and missing upstream artifact fail before output is written.

- [ ] **Step 2: Verify writer tests fail for the missing writer**

Run the Task 3 test module and expect writer-specific failures.

- [ ] **Step 3: Implement atomic publication outputs**

Implement `write_primary_source_associations` and its CLI. Load all data and geometry before
creating outputs, write the SVG and 600-dpi PNG, write stable TSV audits and caption, then
write a JSON manifest containing resolved inputs, SHA-256 checksums, dimensions,
configuration, and package versions. Return an immutable paths record.

- [ ] **Step 4: Verify the full figure test modules pass**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study2_primary_source_associations.py studies/tests/pipelines/test_study2_primary_source_associations_figure.py -q
```

Expected: all tests pass without warnings.

### Task 5: Document, preflight, and verify the integrated analysis

**Files:**
- Modify: `studies/pain_study/study2/README.md`
- Modify: `studies/tests/config/test_study2_config_loader.py`
- Modify: `studies/tests/pipelines/test_study2_stages.py`

- [ ] **Step 1: Write failing configuration and path-contract assertions**

Assert the figure dimensions, band order, surface, font, and output helpers. Assert the
source-power stage writes and validates the common vertex manifest.

- [ ] **Step 2: Verify the new integration assertions fail**

Run the focused config and stage tests and confirm the expected missing-contract failures.

- [ ] **Step 3: Add concise execution and interpretation documentation**

Document the one-script command, required upstream stages, output family, Fisher-mean
estimand, target-retrained maximum-cluster correction, Holm correction, and cortical-source
interpretation boundary. Do not duplicate the full design specification.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study2_source_vertex_manifest.py studies/tests/pipelines/test_study2_source_power.py studies/tests/pipelines/test_study2_primary_source_associations.py studies/tests/pipelines/test_study2_primary_source_associations_figure.py -q
python -m pytest studies/tests -q
ruff check studies/pain_study/study2 studies/tests
make verify-structure
make verify-architecture
make verify-maintainability
```

Expected: all tests and repository gates pass.

- [ ] **Step 5: Run real-data preflight and final visual QA**

Invoke the writer with the configured derivative root. If upstream source-stage, null,
adjacency, family-summary, or vertex-manifest artifacts are absent, verify it fails before
writing any output and report the exact missing artifact. When artifacts exist, inspect the
SVG and 600-dpi PNG at full size and thumbnail size; verify matched views, readable status
text, no legend/data overlap, a centered shared scale, and correctly confined contours.
