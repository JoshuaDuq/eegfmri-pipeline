# Study 1 Whole-Brain fMRI Construct-Validity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one deterministic, publication-quality whole-brain figure of BOLD sensitivity to
delivered temperature and within-temperature subjective intensity, with participant-level
inference and complete reproducibility artifacts.

**Architecture:** Study-specific modules validate the retained cohort and construct two explicit
multi-run first-level designs from fMRIPrep BOLD data. Separate model and renderer modules produce
participant effects, max-T group inference, map audits, and a fixed surface-plus-volume SVG; one
standalone writer owns all filesystem outputs.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, NiBabel, Nilearn, Matplotlib, PyYAML, pytest.

---

### Task 1: Freeze the figure and inference configuration

**Files:**
- Modify: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Modify: `studies/tests/config/test_study1_config_loader.py`

- [ ] **Step 1: Add a failing exact-default test**

```python
def test_load_study1_config_includes_fmri_construct_validity_defaults() -> None:
    config = load_study1_config()
    assert config["study1"]["figures"]["fmri_construct_validity"] == {
        "dimensions_mm": {"width": 183.0, "height": 112.0},
        "axial_slices_mm": [-12.0, 0.0, 12.0, 24.0, 36.0, 48.0],
        "surface_mesh": "fsaverage5",
        "inference": {
            "n_permutations": 10000,
            "two_sided": True,
            "alpha": 0.05,
            "random_state": 20260711,
        },
        "display": {"robust_percentile": 99.5},
        "minimum_article_subjects": 30,
    }
```

- [ ] **Step 2: Run the test and confirm the key is absent**

Run:
`MNE_DONTWRITE_HOME=true python -m pytest studies/tests/config/test_study1_config_loader.py::test_load_study1_config_includes_fmri_construct_validity_defaults -q`

Expected: FAIL because `fmri_construct_validity` is not configured.

- [ ] **Step 3: Add the exact YAML configuration**

```yaml
fmri_construct_validity:
  dimensions_mm: {width: 183.0, height: 112.0}
  axial_slices_mm: [-12.0, 0.0, 12.0, 24.0, 36.0, 48.0]
  surface_mesh: "fsaverage5"
  inference:
    n_permutations: 10000
    two_sided: true
    alpha: 0.05
    random_state: 20260711
  display:
    robust_percentile: 99.5
  minimum_article_subjects: 30
```

- [ ] **Step 4: Rerun the focused configuration test**

Expected: PASS.

### Task 2: Validate fMRI runs and construct explicit event designs

**Files:**
- Create: `studies/pain_study/study1/figures/fmri_construct_data.py`
- Create: `studies/tests/pipelines/test_study1_fmri_construct_validity.py`

- [ ] **Step 1: Write failing tests for the scientific data contract**

Use compact synthetic event tables and assert:

```python
temperature = build_temperature_events(raw_events, retained_trials)
plateau = temperature.loc[temperature["trial_type"].eq("temperature_linear")]
assert np.allclose(plateau["modulation"], [-1.0, 0.0, 1.0])
assert set(temperature["trial_type"]) >= {
    "temperature_linear",
    "within_run_trial_order",
    "nuisance_fixation_rest",
}

rating = build_rating_events(raw_events, retained_trials)
rating_rows = rating.loc[rating["trial_type"].eq("rating_within_temperature")]
assert rating_rows.groupby("stimulus_temp")["modulation"].mean().abs().max() < 1e-12
assert np.allclose(rating_rows["modulation"], expected_rating_deviation / 10.0)
assert {f"temperature_{value:g}" for value in TEMPERATURES}.issubset(rating["trial_type"])
```

Add independent failure tests for duplicate retained keys, absent temperatures, non-constant
surface within run, invalid 0–200 protocol coding, unmatched raw/retained trials, empty residual
rating variation, and missing BOLD/mask/confound inputs.

- [ ] **Step 2: Run the tests and confirm import failure**

Run:
`MNE_DONTWRITE_HOME=true python -m pytest studies/tests/pipelines/test_study1_fmri_construct_validity.py -q`

Expected: FAIL because `fmri_construct_data` does not exist.

- [ ] **Step 3: Implement focused immutable inputs and event builders**

```python
@dataclass(frozen=True)
class FmriRunInput:
    subject_id: str
    run: int
    bold_path: Path
    mask_path: Path
    confounds_path: Path
    raw_events_path: Path
    raw_events: pd.DataFrame
    retained_trials: pd.DataFrame


@dataclass(frozen=True)
class FirstLevelRunDesign:
    subject_id: str
    run: int
    estimand: str
    target_column: str
    events: pd.DataFrame
    audit: Mapping[str, object]
```

Implement:

```python
def load_fmri_run_inputs(*, task: str, config: Any) -> tuple[FmriRunInput, ...]: ...
def build_temperature_events(raw_events: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame: ...
def build_rating_events(raw_events: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame: ...
def build_subject_designs(runs: Sequence[FmriRunInput]) -> tuple[FirstLevelRunDesign, ...]: ...
```

Resolve the cohort through `load_validity_trial_data`. Discover MNI fMRIPrep BOLD and matching
brain masks with public `bold_discovery` helpers. Resolve raw fMRI events under
`resolve_fmri_bids_root(config)` and confounds beside fMRIPrep BOLD. Join trials by integer run and
within-run trial number. Reuse `add_within_scale_intensity` for protocol scoring. Give every
continuous modulator and categorical temperature regressor its own event row.

- [ ] **Step 4: Run tests, Black, and focused Ruff**

Expected: all data-contract tests PASS and Ruff reports no errors.

### Task 3: Fit participant GLMs and preserve design audits

**Files:**
- Create: `studies/pain_study/study1/figures/fmri_construct_models.py`
- Modify: `studies/tests/pipelines/test_study1_fmri_construct_validity.py`

- [ ] **Step 1: Add failing synthetic-NIfTI recovery tests**

Create two small 4D runs with deterministic signals and assert:

```python
result = fit_subject_effects(
    subject_runs=synthetic_runs,
    designs=synthetic_designs,
    settings=first_level_settings(),
)
assert set(result.effect_images) == {"temperature", "rating"}
assert result.effect_images["temperature"].get_fdata()[signal_voxel] > 0
assert result.effect_images["rating"].get_fdata()[rating_voxel] > 0
assert set(result.design_audit["target_column"]) == {
    "temperature_linear",
    "rating_within_temperature",
}
```

Add failures for mixed TRs, mismatched geometry, missing motion24 confounds, rank deficiency,
condition number above `study1.targets.max_design_condition_number`, absent target columns, and
non-finite effect images.

- [ ] **Step 2: Run the new tests and confirm the model API is missing**

Expected: FAIL on import.

- [ ] **Step 3: Implement first-level settings and participant fitting**

```python
@dataclass(frozen=True)
class FirstLevelSettings:
    hrf_model: str
    drift_model: str | None
    high_pass_hz: float
    low_pass_hz: float | None
    smoothing_fwhm: float
    confounds_strategy: str
    max_condition_number: float


@dataclass(frozen=True)
class SubjectEffectResult:
    subject_id: str
    effect_images: Mapping[str, nib.Nifti1Image]
    design_audit: pd.DataFrame
```

Implement `first_level_settings(config)` and `fit_subject_effects(...)`. For each estimand, fit one
Nilearn multi-run `FirstLevelModel` using lists of BOLD images, explicit event tables, motion24
confounds, and sample masks. Call `validate_design_matrices` with the target column and configured
condition-number ceiling. Compute `effect_size` for the named target column. Record frames,
retained frames, regressors, rank, residual degrees of freedom, condition number, target
efficiency, TR, and source paths per run.

- [ ] **Step 4: Rerun the model tests and focused lint**

Expected: PASS.

### Task 4: Implement participant-level group inference and spatial audits

**Files:**
- Modify: `studies/pain_study/study1/figures/fmri_construct_models.py`
- Create: `studies/pain_study/study1/figures/fmri_construct_validity.py`
- Modify: `studies/tests/pipelines/test_study1_fmri_construct_validity.py`

- [ ] **Step 1: Add failing group-inference tests**

```python
settings = GroupInferenceSettings(
    n_permutations=64,
    two_sided=True,
    alpha=0.05,
    random_state=7,
)
group = run_group_inference(effect_images, estimand="temperature", settings=settings)
assert np.allclose(group.mean_effect.get_fdata()[signal_voxel], expected_mean)
assert np.isfinite(group.neg_log10_fwe_p.get_fdata()).all()
assert group.significance_mask.get_fdata().dtype == np.uint8
assert group.n_subjects == len(effect_images)
```

Test identical participant order across estimands, deterministic permutations, affine/shape
rejection, significance threshold `-log10(alpha)`, positive/negative connected components, peak
world coordinates, and no cluster-size filtering.

- [ ] **Step 2: Implement group types and inference**

```python
@dataclass(frozen=True)
class GroupInferenceSettings:
    n_permutations: int
    two_sided: bool
    alpha: float
    random_state: int


@dataclass(frozen=True)
class GroupMapResult:
    estimand: str
    mean_effect: nib.Nifti1Image
    neg_log10_fwe_p: nib.Nifti1Image
    significance_mask: nib.Nifti1Image
    peaks: pd.DataFrame
    n_subjects: int
```

Use an intercept-only `SecondLevelModel` for the group mean effect and
`non_parametric_inference(..., model_intercept=False, n_perm=..., two_sided_test=True,
random_state=...)` for voxelwise max-T FWE p values. Build connected components separately for
positive and negative significant effects with `scipy.ndimage.label`; report every component and
its maximum-absolute effect voxel.

- [ ] **Step 3: Implement end-to-end in-memory orchestration**

```python
@dataclass(frozen=True)
class FmriConstructValiditySummary:
    subjects: pd.DataFrame
    design_audit: pd.DataFrame
    subject_effects: Mapping[str, tuple[nib.Nifti1Image, ...]]
    group_maps: Mapping[str, GroupMapResult]
    article_ready: bool


def build_fmri_construct_validity_summary(
    *, task: str, config: Any
) -> FmriConstructValiditySummary: ...
```

Require the same ordered participants for both estimands and derive article readiness only from
the configured threshold.

- [ ] **Step 4: Run focused tests and Ruff**

Expected: PASS.

### Task 5: Render the fixed publication figure

**Files:**
- Create: `studies/pain_study/study1/figures/fmri_construct_validity_plot.py`
- Create: `studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py`

- [ ] **Step 1: Write failing structural rendering tests**

```python
figure = build_fmri_construct_validity_figure(synthetic_summary, load_study1_config())
assert np.allclose(figure.get_size_inches(), (183 / 25.4, 112 / 25.4))
assert panel_titles(figure) == [
    "Delivered temperature",
    "Subjective intensity beyond temperature",
]
assert rendered_slice_labels(figure) == ["z = -12", "z = 0", "z = 12", "z = 24", "z = 36", "z = 48"]
assert all_color_limits_are_symmetric(figure)
assert significance_contours_are_present(figure)
```

Also test fixed fsaverage5 view order, two correctly labelled color bars, no legend over data,
participant count/readiness annotation, missing-significance rejection, exact view geometry, and
no clipped SVG text.

- [ ] **Step 2: Implement rendering only**

```python
def build_fmri_construct_validity_figure(
    summary: FmriConstructValiditySummary,
    config: Any,
) -> Figure: ...
```

Load packaged fsaverage5 with `nilearn.datasets.load_fsaverage`. Project group effects and binary
significance masks with `surface.vol_to_surf`; draw lateral/medial views with
`plot_surf_stat_map` plus `plot_surf_contours`. Draw the six configured axial cuts with
`plot_stat_map` and add the binary mask as a contour. Use the existing Study 1 publication
context, a zero-centered color-vision-safe diverging map, and a 99.5th-percentile symmetric range
computed independently per estimand.

- [ ] **Step 3: Run figure tests and inspect a synthetic PNG**

Expected: tests PASS; labels, surfaces, axial slices, contours, and color bars are legible without
overlap or clipping.

### Task 6: Add the sole writer, provenance, and documentation

**Files:**
- Create: `studies/pain_study/study1/figures/plot_fmri_construct_validity.py`
- Modify: `studies/tests/pipelines/test_study1_fmri_construct_validity_figure.py`
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`

- [ ] **Step 1: Write a failing exact-output test**

Assert the writer produces the SVG, subject effect maps, four group maps, three TSV/parquet audit
pairs, and one provenance JSON. Parse the SVG to verify 183 × 112 mm and editable text. Parse the
JSON and verify source/output SHA-256 values, settings, software versions, and permutation seed.

- [ ] **Step 2: Implement writer and CLI**

```python
def write_fmri_construct_validity(
    *, task: str, config: Any, output_path: Path | None = None
) -> FmriConstructValidityPaths: ...


def main(argv: Sequence[str] | None = None) -> FmriConstructValidityPaths: ...
```

Support required `--config` and `--task`, plus optional `--study1-config`, `--deriv-root`, and
`--output`. Save participant maps beneath a deterministic `subject_maps/` directory. Save group
effect and p maps beside the SVG. Write audit tables with the shared TSV/parquet utilities and
write sorted, indented JSON only after every computation validates.

- [ ] **Step 3: Document purpose, command, outputs, and inference**

State that the effect maps are unthresholded, outlines are two-sided voxelwise max-T FWE p < .05,
rating is centered within participant and temperature, temperature is not adjusted for rating,
and participant is the group inferential unit.

- [ ] **Step 4: Run CLI help, focused tests, Black, and Ruff**

Expected: clean help without import warnings; all focused tests PASS.

### Task 7: Verify the complete implementation and real-data readiness

**Files:**
- Modify only if a failing test identifies a defect in files above.

- [ ] **Step 1: Run all Study 1 tests**

Run:
`MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/matplotlib-fmri-validity python -m pytest studies/tests -q`

Expected: zero failures.

- [ ] **Step 2: Run repository quality gates**

Run:

```bash
ruff check eeg_pipeline fmri_pipeline studies tests scripts
make verify-structure
make verify-architecture
make verify-maintainability
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run the production-data preflight**

```bash
python -m studies.pain_study.study1.figures.plot_fmri_construct_validity \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config studies/pain_study/study1/config/study1_config.yaml \
  --deriv-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives \
  --task thermalactive
```

If the current target table or BOLD derivatives are stale, record the first exact validation
error and verify no final SVG exists. Do not weaken validation or synthesize article results.

- [ ] **Step 4: Review the final diff against the written specification**

Confirm that every specified estimand, inference setting, fixed view, output artifact, failure
condition, documentation statement, and test is represented exactly once and has no competing
implementation path.
