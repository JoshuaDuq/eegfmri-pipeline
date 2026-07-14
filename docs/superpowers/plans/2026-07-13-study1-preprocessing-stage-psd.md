# Study 1 Preprocessing-Stage PSD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible cohort PSD artifacts for the stored raw 5,000-Hz, BrainVision-processed 1,000-Hz, and final MNE-processed 500-Hz Study 1 recordings.

**Architecture:** Add strict, format-specific source discovery behind one immutable run-source interface, then reuse a raw-object spectral estimator and the existing participant-first cohort summary. A separate stage writer validates every requested checkpoint before producing stage-labeled figures and audit tables, leaving the existing final-clean command unchanged.

**Tech Stack:** Python 3.11, MNE-Python, NumPy, pandas, Matplotlib, PyYAML configuration, pytest, Ruff, BrainVision files, ZIP archives, FIF files.

---

## File Map

- Create `studies/pain_study/study1/figures/preprocessing_psd_sources.py`: stage definitions, strict filename parsing, source discovery, header validation, and raw loaders.
- Modify `studies/pain_study/study1/figures/continuous_spectrum.py`: extract a raw-object estimator while preserving the final-FIF entry point.
- Create `studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py`: stage specifications and stage columns for audit tables.
- Create `studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density_plot.py`: stage-labeled rendering built from the existing cohort plot.
- Create `studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py`: all-stage validation, artifact writing, and CLI.
- Modify `studies/pain_study/study1/config/study1_figure_config.yaml`: common Welch duration and the three stage contracts.
- Modify `studies/tests/config/test_study1_config_loader.py`: configuration contract coverage.
- Modify `studies/tests/pipelines/test_study1_continuous_spectrum.py`: raw-object estimator coverage.
- Create `studies/tests/pipelines/test_study1_preprocessing_psd_sources.py`: archive, directory, and FIF discovery tests.
- Create `studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py`: specification, rendering, writer, table, and CLI tests.

### Task 1: Define and validate the three stage specifications

**Files:**
- Modify: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Create: `studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py`
- Modify: `studies/tests/config/test_study1_config_loader.py`
- Create: `studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py`

- [ ] **Step 1: Write failing configuration and derivation tests**

Add tests asserting the exact stage keys, expected sampling frequencies, shared
16.384-second segment duration, 0.5 overlap, and derived sample counts:

```python
@pytest.mark.parametrize(
    ("stage", "sampling_frequency_hz", "n_fft", "n_overlap"),
    (("raw", 5000.0, 81920, 40960),
     ("processed", 1000.0, 16384, 8192),
     ("mne", 500.0, 8192, 4096)),
)
def test_preprocessing_stage_psd_specification_uses_equal_time_windows(
    stage, sampling_frequency_hz, n_fft, n_overlap
):
    config = load_study1_config()
    specification = preprocessing_stage_psd_specification(config, stage)
    assert specification.stage.identifier == stage
    assert specification.spectrum.sampling_frequency_hz == sampling_frequency_hz
    assert specification.spectrum.n_fft == n_fft
    assert specification.spectrum.n_overlap == n_overlap
```

- [ ] **Step 2: Run the focused tests and confirm the missing API failure**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py -q`

Expected: FAIL because the stage configuration and module do not exist.

- [ ] **Step 3: Add the minimal YAML contract and immutable specifications**

Add:

```yaml
preprocessing_stage_power_spectral_density:
  segment_duration_s: 16.384
  overlap_fraction: 0.5
  stages:
    raw:
      label: "Original BrainVision"
      sampling_frequency_hz: 5000.0
    processed:
      label: "BrainVision processed"
      sampling_frequency_hz: 1000.0
    mne:
      label: "Final MNE processed"
      sampling_frequency_hz: 500.0
```

Implement `PreprocessingStage`, `PreprocessingStagePsdSpecification`, and
`preprocessing_stage_psd_specification(config, stage)` using
`require_config_value`. Require an exact stage key, positive finite duration,
overlap in `[0, 1)`, and integral sample counts; raise `ValueError` rather than
rounding a non-integral result.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/study1/config/study1_figure_config.yaml \
  studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py \
  studies/tests/config/test_study1_config_loader.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py
git commit -m "feat: define Study 1 PSD stages"
```

### Task 2: Discover strict raw, processed, and MNE run sources

**Files:**
- Create: `studies/pain_study/study1/figures/preprocessing_psd_sources.py`
- Create: `studies/tests/pipelines/test_study1_preprocessing_psd_sources.py`

- [ ] **Step 1: Write failing parsing and discovery tests**

Create synthetic BrainVision ZIP members and directory triplets. Cover:

```python
def test_discover_raw_sources_reads_complete_5000_hz_archive(tmp_path):
    archive = write_brainvision_archive(tmp_path, sampling_interval_us=200)
    sources = discover_raw_brainvision_runs(
        tmp_path, task="thermalactive", excluded_subjects=()
    )
    assert [(source.subject_id, source.run_id) for source in sources] == [
        ("sub-0001", "1")
    ]
    assert sources[0].source_path == f"{archive}::raw/{HEADER_NAME}"

def test_discover_processed_sources_requires_1000_hz(tmp_path):
    write_brainvision_triplet(tmp_path, sampling_interval_us=200)
    with pytest.raises(ValueError, match="expected 1000.0 Hz"):
        discover_processed_brainvision_runs(
            tmp_path, task="thermalactive", excluded_subjects=()
        )

def test_discovery_rejects_duplicate_subject_run(tmp_path):
    write_duplicate_triplets(tmp_path)
    with pytest.raises(ValueError, match="Duplicate EEG source for sub-0001 run 1"):
        discover_processed_brainvision_runs(
            tmp_path, task="thermalactive", excluded_subjects=()
        )
```

Also test missing `.eeg`/`.vmrk`, malformed thermal filenames, requested-subject
filtering, exclusion filtering, trash omission, no selected files, and reuse of
final-clean FIF discovery.

- [ ] **Step 2: Run the source tests and confirm import failures**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_psd_sources.py -q`

Expected: FAIL because `preprocessing_psd_sources` does not exist.

- [ ] **Step 3: Implement immutable sources and filename parsing**

Define a union of representations so invalid field combinations are
unrepresentable:

```python
@dataclass(frozen=True)
class BrainVisionArchiveRunSource:
    subject_id: str
    run_id: str
    source_path: str
    archive_path: Path
    header_member: str
    marker_member: str
    data_member: str


@dataclass(frozen=True)
class BrainVisionFileRunSource:
    subject_id: str
    run_id: str
    source_path: str
    header_path: Path


@dataclass(frozen=True)
class FifRunSource:
    subject_id: str
    run_id: str
    source_path: str
    path: Path


EegRunSource = (
    BrainVisionArchiveRunSource | BrainVisionFileRunSource | FifRunSource
)

THERMAL_RUN_PATTERN = re.compile(
    r"^ThermalPainEEGFMRI_run(?P<run>\d+)_sub(?P<subject>\d{4})_"
    r".+?(?P<processed>_scannerpulse_corrected)?\.vhdr$"
)
```

Each representation exposes a constant `representation` property for the run
audit.

- [ ] **Step 4: Implement strict discoverers and header validation**

Use `zipfile.ZipFile` for archive manifests and `Path.glob` for stored files.
Parse `SamplingInterval` from each header and convert microseconds to Hz. Validate
the expected suffix, all three BrainVision members, 5,000 or 1,000 Hz exactly,
unique `(subject_id, run_id)`, numbered participants, exclusions, and requested
subjects. Raise on every violated contract and never skip a malformed candidate.

- [ ] **Step 5: Run source tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_psd_sources.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/study1/figures/preprocessing_psd_sources.py \
  studies/tests/pipelines/test_study1_preprocessing_psd_sources.py
git commit -m "feat: discover Study 1 PSD sources"
```

### Task 3: Estimate spectra from a common raw-object boundary

**Files:**
- Modify: `studies/pain_study/study1/figures/continuous_spectrum.py`
- Modify: `studies/pain_study/study1/figures/preprocessing_psd_sources.py`
- Modify: `studies/tests/pipelines/test_study1_continuous_spectrum.py`
- Modify: `studies/tests/pipelines/test_study1_preprocessing_psd_sources.py`

- [ ] **Step 1: Write failing raw-object and loader tests**

Add a test that calls `estimate_raw_continuous_run_spectrum` with an explicit
identity and verifies the existing Welch arguments and audit values. Add loader
tests that monkeypatch `mne.io.read_raw_brainvision` and
`mne.io.read_raw_fif`, verify `set_channel_types(raw)` is called for BrainVision,
and ensure archive extraction contains exactly the selected `.vhdr`, `.vmrk`,
and `.eeg` members while the reader is active.

- [ ] **Step 2: Run tests and verify the missing raw-object API**

Run: `python -m pytest studies/tests/pipelines/test_study1_continuous_spectrum.py studies/tests/pipelines/test_study1_preprocessing_psd_sources.py -q`

Expected: FAIL because the raw-object estimator and loaders do not exist.

- [ ] **Step 3: Extract the raw-object estimator without changing final-FIF behavior**

Implement:

```python
def estimate_raw_continuous_run_spectrum(
    raw,
    *,
    subject_id: str,
    run_id: str,
    source_file: Path | str,
    specification: ContinuousSpectrumSpecification,
) -> ContinuousRunSpectrum:
    ...

def estimate_continuous_run_spectrum(path, specification):
    subject_id, run_id = parse_final_clean_filename(Path(path))
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    return estimate_raw_continuous_run_spectrum(
        raw,
        subject_id=subject_id,
        run_id=run_id,
        source_file=path,
        specification=specification,
    )
```

Change `ContinuousRunSpectrum.source_file` to `Path | str` so archive-member
provenance is retained exactly. Keep all existing validation in the shared
function.

- [ ] **Step 4: Implement source loading and immediate cleanup**

Add `estimate_source_spectrum(source, specification)`. Directory BrainVision
sources call `read_raw_brainvision`; archive sources extract one complete triplet
inside `TemporaryDirectory`; FIF sources call the existing final entry point.
Use the established `set_channel_types` function before BrainVision estimation,
then close raw objects in `finally` and allow reader, extraction, and validation
errors to surface.

- [ ] **Step 5: Run focused regression tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_continuous_spectrum.py studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_preprocessing_psd_sources.py -q`

Expected: PASS with unchanged scanner-harmonic behavior.

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/study1/figures/continuous_spectrum.py \
  studies/pain_study/study1/figures/preprocessing_psd_sources.py \
  studies/tests/pipelines/test_study1_continuous_spectrum.py \
  studies/tests/pipelines/test_study1_preprocessing_psd_sources.py
git commit -m "refactor: estimate spectra from loaded EEG"
```

### Task 4: Write stage-labeled figures and exact audit tables

**Files:**
- Modify: `studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py`
- Create: `studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density_plot.py`
- Create: `studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py`
- Modify: `studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py`

- [ ] **Step 1: Write failing stage summary, rendering, and writer tests**

Test that `label_stage_summary` prepends `stage` to all three tables without
mutating the original summary. Test that the stage figure includes the exact
checkpoint label and sampling rate. Test these exact output basenames:

```python
assert paths.svg.name == "cohort_power_spectral_density_raw.svg"
assert paths.run_tsv.name == "cohort_power_spectral_density_raw_by_run.tsv"
assert paths.participant_tsv.name == (
    "cohort_power_spectral_density_raw_by_subject.tsv"
)
assert paths.summary_tsv.name == "cohort_power_spectral_density_raw_summary.tsv"
```

Read TSV and Parquet outputs and compare complete frames after normalizing only
format-imposed dtypes. Verify deterministic SVG bytes and 183 × 92 mm dimensions.

- [ ] **Step 2: Run the focused writer tests and confirm failures**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py -q`

Expected: FAIL because stage labeling, rendering, and writing are incomplete.

- [ ] **Step 3: Implement non-mutating stage summary labeling**

Return a new `CohortPsdSummary` whose frames use
`frame.assign(stage=stage.identifier)` followed by an explicit column reorder.
Add `segment_duration_s` and `overlap_fraction` to the run audit from the stage
specification while retaining exact `n_fft` and `n_overlap` values.

- [ ] **Step 4: Implement the stage figure builder**

Call `build_cohort_psd_figure`, then add one concise upper-right axes annotation:

```python
axis.text(
    0.99,
    0.98,
    f"{stage.label} · {stage.sampling_frequency_hz:g} Hz",
    ha="right",
    va="top",
    transform=axis.transAxes,
    fontsize=annotation_size,
)
```

Do not duplicate the existing spectrum layers, legends, or scientific frequency
annotations.

- [ ] **Step 5: Implement the single-stage artifact writer**

Build a stage summary from already validated sources, render it, save the SVG
with the deterministic writer, and serialize all six tables. Define one
`PreprocessingStagePsdPaths` dataclass and one filename stem function so every
output name derives from `cohort_power_spectral_density_<stage>`.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py -q`

Expected: PASS and the existing figure tests remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py \
  studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density_plot.py \
  studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py
git commit -m "feat: write preprocessing-stage PSD reports"
```

### Task 5: Add the explicit multi-stage CLI and atomic prevalidation

**Files:**
- Modify: `studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py`
- Modify: `studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py`

- [ ] **Step 1: Write failing CLI and prevalidation tests**

Monkeypatch discovery and writing. Request `raw`, `processed`, and `mne`; make
the last discovery raise and assert no writer was called. Then make all discovery
succeed and assert each writer receives its matching validated sources. Test
duplicate stage arguments, an invalid stage, required Kingston root, explicit
derivative root, repeatable subject filters, output directory, printed SVG
paths, and warning-free module help.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py -q`

Expected: FAIL because orchestration and CLI parsing are incomplete.

- [ ] **Step 3: Implement prevalidation and CLI orchestration**

Parse repeatable required `--stage` choices (`raw`, `processed`, `mne`). Reject
duplicates. Resolve config, validate all stage specifications, and discover all
sources into a dictionary before calling any writer. Then write reports in the
user-provided stage order and print one SVG path per line. Require
`--kingston-root`; resolve the derivative root from config only when `mne` is
selected and no explicit derivative root is supplied. Default the output
directory through `validity_output_dir(config)` when `--output-dir` is omitted.

- [ ] **Step 4: Run CLI and full feature tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py studies/tests/pipelines/test_study1_cohort_power_spectral_density.py studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py
git commit -m "feat: orchestrate Study 1 stage PSDs"
```

### Task 6: Verify and generate the Kingston reporting artifacts

**Files:**
- Modify only if verification exposes a root-cause defect in files listed above.

- [ ] **Step 1: Format and lint changed code**

Run:

```bash
black studies/pain_study/study1/figures/preprocessing_psd_sources.py \
  studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density.py \
  studies/pain_study/study1/figures/preprocessing_stage_power_spectral_density_plot.py \
  studies/pain_study/study1/figures/plot_preprocessing_stage_power_spectral_density.py \
  studies/pain_study/study1/figures/continuous_spectrum.py \
  studies/tests/pipelines/test_study1_preprocessing_psd_sources.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py \
  studies/tests/pipelines/test_study1_continuous_spectrum.py
ruff check studies/pain_study/study1/figures studies/tests/pipelines
```

Expected: both commands exit 0.

- [ ] **Step 2: Run focused and structural verification**

Run:

```bash
python -m pytest studies/tests/pipelines/test_study1_continuous_spectrum.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py \
  studies/tests/pipelines/test_study1_cohort_power_spectral_density.py \
  studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py \
  studies/tests/pipelines/test_study1_preprocessing_psd_sources.py \
  studies/tests/pipelines/test_study1_preprocessing_stage_power_spectral_density.py -q
make verify-architecture
make verify-maintainability
```

Expected: all commands exit 0.

- [ ] **Step 3: Generate all three reports from Kingston**

Run the new module with:

```bash
python -m studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --task thermalactive \
  --kingston-root /Volumes/KINGSTON \
  --derivative-root /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
  --stage raw --stage processed --stage mne
```

Expected: three SVG paths and their exact TSV/Parquet families are written.

- [ ] **Step 4: Verify real-data contracts and visual output**

Confirm each run audit has exactly one stage and the expected sampling frequency,
all summary frequency axes are identical, TSV/Parquet frames agree, and every
figure opens without clipping, overlap, missing layers, or misleading labels.
Record participant/run counts and any scientifically meaningful differences
without changing thresholds or excluding data post hoc.

- [ ] **Step 5: Run final diff and status checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional changes.

- [ ] **Step 6: Commit any formatting-only changes**

```bash
git add studies/pain_study/study1 studies/tests/config studies/tests/pipelines
git commit -m "style: format preprocessing-stage PSD code"
```
