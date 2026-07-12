# Study 2 Spatial-Convergence Figure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, publication-ready multimodal figure that displays the resolution-matched fMRI reference, the three EEG source maps, and BrainSMASH spatial-null inference.

**Architecture:** A focused summary loader validates and reconciles existing spatial-stage artifacts without changing the estimand. A separate renderer owns the fixed scientific composition, while a CLI writer owns deterministic output files, provenance, and atomic table/text writes. Shared surface and style utilities are reused from the existing Study 2 figure suite.

**Tech Stack:** Python 3.11+, NumPy, pandas, Matplotlib, Nilearn, MNE-Python surface geometry, pytest, SVG/XML.

---

### Task 1: Lock the spatial-convergence artifact contract

**Files:**
- Modify: `studies/tests/pipelines/test_study2_stages.py`
- Create: `studies/tests/pipelines/test_study2_spatial_convergence_figure.py`
- Modify: `studies/pain_study/study2/stages.py`
- Create: `studies/pain_study/study2/figures/spatial_convergence.py`

- [ ] **Step 1: Write the failing spatial-identity metadata test**

Extend the spatial-surrogate stage test to require `surrogate_metadata.json` to record the exact
common-source vertex-manifest SHA256, configured common subject and spacing, analysis-mask SHA256,
and per-band fMRI-map SHA256. These fields bind equal-sized spatial arrays to one explicit source
space and input family.

- [ ] **Step 2: Run the stage test and verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_stages.py::test_run_spatial_surrogates_records_source_identity
```

Expected: failure because the identity fields are absent.

- [ ] **Step 3: Implement strict spatial-identity metadata**

Require the common-source manifest and metadata as spatial-surrogate stage inputs. Validate the
manifest subject, spacing, and vertex count against the mask before generating surrogates. Write
the source-manifest, mask, and fMRI-map hashes with the existing method, draw-count, seed, and band
metadata. Do not accept or emit the old schema.

- [ ] **Step 4: Run the stage test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Write the failing strict-loader test**

Create deterministic artifacts for alpha, beta, and gamma with a six-vertex common-source manifest,
one shared fMRI map, one EEG map per band, a boolean analysis mask, three surrogate draws per band,
and an exact saved summary. Require:

```python
summary = load_spatial_convergence(config)

assert summary.bands == ("alpha", "beta", "gamma")
assert summary.n_vertices == 6
assert summary.n_surrogates == 3
assert summary.mask.sum() == 6
assert tuple(summary.audit.columns) == AUDIT_COLUMNS
```

- [ ] **Step 6: Run the loader test and verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py::test_loader_reconciles_complete_spatial_family
```

Expected: import failure because `spatial_convergence.py` does not exist.

- [ ] **Step 7: Implement the minimal validated summary loader**

Define:

```python
AUDIT_COLUMNS = (
    "band",
    "spatial_r",
    "p_value",
    "meaningful",
    "holm_adjusted_p_value",
    "holm_significant",
    "null_ci_low",
    "null_ci_high",
)

@dataclass(frozen=True)
class SpatialConvergenceBand:
    band: str
    eeg_map: np.ndarray
    surrogate_r: np.ndarray
    spatial_r: float
    p_value: float
    meaningful: bool
    holm_adjusted_p_value: float
    holm_significant: bool

@dataclass(frozen=True)
class SpatialConvergenceSummary:
    bands: tuple[str, ...]
    fmri_map: np.ndarray
    mask: np.ndarray
    band_results: dict[str, SpatialConvergenceBand]
    vertices_manifest: CommonSourceVertices
    audit: pd.DataFrame
    family_alpha: float
    source_paths: tuple[Path, ...]
```

Load all arrays with `allow_pickle=False`, validate shapes and finiteness, call the existing
`compute_spatial_correspondence()`, apply `holm_adjusted_p_values()`, and reconcile every saved
summary field. Calculate the 2.5th and 97.5th percentiles of `surrogate_r` only for the audit/display.
Require the spatial metadata's common subject, spacing, source-manifest hash, mask hash, and
per-band fMRI-map hashes to match the current artifacts exactly. Inspect the loaded mask before
conversion and require exact NumPy boolean dtype. Require the band-specific fMRI maps to match with
`rtol=0` and `atol=1e-12`.

- [ ] **Step 8: Run the loader test and verify GREEN**

Run the command from Step 6. Expected: pass.

- [ ] **Step 9: Add fail-fast reconciliation tests**

Add independent tests that require failure for:

- a band-specific fMRI map that differs from the shared reference;
- a saved adjusted p-value that differs from the recomputed family;
- a map whose vertex count differs from the manifest; and
- a stored integer or floating analysis mask;
- an equal-sized spatial family whose metadata names the wrong vertex manifest;
- a band-specific fMRI map that differs by more than `1e-12`;
- a missing surrogate artifact.

- [ ] **Step 10: Run the loader test module**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
```

Expected: all loader tests pass.

### Task 2: Render the fixed multimodal composition

**Files:**
- Modify: `studies/tests/pipelines/test_study2_spatial_convergence_figure.py`
- Create: `studies/pain_study/study2/figures/spatial_convergence_plot.py`
- Modify: `studies/pain_study/study2/figures/style.py`
- Modify: `studies/pain_study/study2/figures/primary_source_associations_plot.py`
- Modify: `studies/pain_study/study2/config/study2_config.yaml`
- Modify: `studies/tests/pipelines/test_study2_primary_source_associations_figure.py`

- [ ] **Step 1: Write the failing figure-structure test**

Build a summary and synthetic common surfaces. Require:

```python
figure = build_spatial_convergence_figure(summary, surfaces, config)
assert np.allclose(figure.get_size_inches(), (183 / 25.4, 112 / 25.4))
assert len([axis for axis in figure.axes if str(axis.get_gid()).startswith("surface-")]) == 8
assert len([axis for axis in figure.axes if str(axis.get_gid()).startswith("null-")]) == 3
```

Also require one fMRI colorbar, one shared EEG colorbar, identical EEG limits, exact adjusted-p
annotations, and observed-correlation diamonds.

- [ ] **Step 2: Run the structure test and verify RED**

Run the named structure test. Expected: import failure because the renderer does not exist.

- [ ] **Step 3: Add the fixed figure configuration**

Add `study2.figures.spatial_convergence` with:

```yaml
dimensions_mm:
  width: 183.0
  height: 112.0
bands:
  - {name: "alpha", label: "Alpha", frequency_label: "8.0–12.9 Hz"}
  - {name: "beta", label: "Beta", frequency_label: "13.0–30.0 Hz"}
  - {name: "gamma", label: "Scanner-clean gamma", frequency_label: "30.1–38, 43–56, 67–77 Hz"}
font_family: "Arial"
png_dpi: 600
null_histogram_bins: 40
```

- [ ] **Step 4: Extract one shared Study 2 diverging colormap**

Export `study2_diverging_color_map()` from `figures/style.py` and use it in the new renderer. Do
not duplicate the existing palette. Update `primary_source_associations_plot.py` to import the
shared function and retain its exact sampled colors; extend its test to protect that visual output.

- [ ] **Step 5: Implement the renderer**

Use a two-row GridSpec. The top row contains four columns, each with left- and right-lateral
surface views: fMRI, alpha, beta, gamma. Use neutral values outside the analysis mask and determine
limits only from masked vertices. Use separate fMRI and shared EEG horizontal colorbars.

The bottom row contains three aligned null axes. Draw a normalized histogram, a zero reference,
and a band-colored observed diamond. Use the same x limits across all three null axes. Report
`r`, raw p, and Holm-adjusted p for every band; use line weight, not a significance star, for
family-significant rows. Include the exact in-figure statement that BrainSMASH surrogate maps
preserve spatial autocorrelation and carry the map-correspondence inference.

- [ ] **Step 6: Run the structure test and verify GREEN**

Run the named structure test. Expected: pass without warnings.

- [ ] **Step 7: Add display-validation tests**

Require failure for an all-zero masked fMRI map, all-zero masked EEG maps, and a configured band
order that differs from the summary. Rerun the test module.

### Task 3: Write the complete publication family

**Files:**
- Modify: `studies/tests/pipelines/test_study2_spatial_convergence_figure.py`
- Create: `studies/pain_study/study2/figures/plot_spatial_convergence.py`
- Modify: `studies/pain_study/study2/paths.py`

- [ ] **Step 1: Write the failing writer test**

Patch only surface loading. Invoke `write_spatial_convergence()` and require exactly:

```text
spatial_convergence.svg
spatial_convergence.png
spatial_convergence_summary.tsv
spatial_convergence_caption.txt
spatial_convergence_manifest.json
```

Verify 183 x 112 mm SVG dimensions, editable text, 600-dpi PNG dimensions, audit schema, caption
language, source hashes, and output hashes.

- [ ] **Step 2: Run the writer test and verify RED**

Expected: import failure because the writer does not exist.

- [ ] **Step 3: Add output path helpers**

Add focused path functions for the five output files to `study2/paths.py` and export them.

- [ ] **Step 4: Implement the writer and CLI**

Follow `plot_primary_source_associations.py`: validate suffix and all inputs before output directory
creation, build the figure, save PNG then SVG, atomically write the audit/caption/manifest, and
include exact software and analysis provenance. The CLI accepts `--config`, `--study2-config`,
`--deriv-root`, and `--output`.

- [ ] **Step 5: Run the writer test and verify GREEN**

Run the writer test. Expected: pass.

- [ ] **Step 6: Add fail-before-output and CLI help tests**

Require a missing scientific artifact to fail before output-directory creation, reject non-SVG
output paths, and run module `--help` with RuntimeWarnings promoted to errors.

- [ ] **Step 7: Add a deterministic serialization test**

Write the complete figure family twice from identical synthetic artifacts into separate
directories. Require byte-identical SVG, PNG, summary TSV, and caption outputs; compare manifest
payloads after excluding absolute source paths. This proves deterministic rendering and stable
scientific content rather than merely deterministic in-memory artist construction.

### Task 4: Document and verify the figure

**Files:**
- Modify: `studies/pain_study/study2/README.md`
- Modify: `studies/pain_study/STUDY_RESULTS_README.md`

- [ ] **Step 1: Document generation and interpretation**

Add the exact CLI command and output family after Study 2 Section 7. State that the figure tests
coarse spatial correspondence, that BrainSMASH carries inference, and that the maps do not imply
vertex-level independence or shared generators.

- [ ] **Step 2: Run focused tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q \
  studies/tests/pipelines/test_study2_spatial_comparison.py \
  studies/tests/pipelines/test_study2_stages.py \
  studies/tests/pipelines/test_study2_primary_source_associations_figure.py \
  studies/tests/pipelines/test_study2_spatial_convergence_figure.py
```

- [ ] **Step 3: Run repository gates**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m ruff check \
  studies/pain_study/study2 studies/tests/pipelines/test_study2_spatial_convergence_figure.py
make verify-architecture
make verify-maintainability
```

- [ ] **Step 4: Perform visual QA**

Retain the synthetic SVG, rasterize it at final aspect ratio, and inspect every label, surface,
colorbar, null distribution, and footnote for clipping or ambiguous encoding.

- [ ] **Step 5: Run the broader Study 2 suite**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest -q studies/tests/pipelines/test_study2_*.py
```

Expected: all tests pass with no warnings or failures.
