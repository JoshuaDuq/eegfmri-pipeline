# MNE Scanner-Harmonic Cohort Comb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically create a participant-equal cohort scanner-harmonic comb PNG and TSV after task-based full or epochs MNE preprocessing.

**Architecture:** A focused analysis module computes matched raw/epoch spectra and hierarchical participant summaries. A separate plotting module renders the single comb figure, while a preprocessing orchestration module discovers BIDS inputs and clean epochs, validates cohort completeness, and writes outputs. `PreprocessingPipeline` adds the QC as a final task-only step and records its output paths.

**Tech Stack:** Python 3.11+, MNE-Python, MNE-BIDS, NumPy, Pandas, Matplotlib, Pytest.

---

## File Structure

- Create `eeg_pipeline/analysis/qc/scanner_harmonic_comb.py`: spectral parameters, MNE-object spectrum estimation, participant aggregation, paired bootstrap summary, and TSV rows.
- Create `eeg_pipeline/plotting/scanner_harmonic_comb.py`: one publication-quality comb figure and PNG writer.
- Create `eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py`: strict BIDS/epoch discovery and end-to-end cohort QC writer.
- Modify `eeg_pipeline/pipelines/preprocessing.py`: add the final task-based QC step and run-metadata outputs.
- Modify `eeg_pipeline/utils/config/eeg_config.yaml`: add the minimal fixed QC settings.
- Create `tests/analysis/test_scanner_harmonic_comb.py`: numerical and validation tests.
- Create `tests/plotting/test_scanner_harmonic_comb_plot.py`: figure-content and export tests.
- Create `tests/preprocessing/test_scanner_harmonic_qc.py`: discovery/orchestration tests.
- Modify `tests/pipelines/test_pipeline_preprocessing.py`: step-selection and metadata integration tests.

### Task 1: Matched Spectrum and Participant Bootstrap Core

**Files:**
- Create: `eeg_pipeline/analysis/qc/scanner_harmonic_comb.py`
- Create: `tests/analysis/test_scanner_harmonic_comb.py`

- [ ] **Step 1: Write failing tests for matched frequency grids and epoch-wise PSDs**

```python
def test_raw_and_epochs_use_identical_frequency_grid() -> None:
    raw = _raw(sfreq=1000.0)
    epochs = _epochs(sfreq=500.0)
    parameters = ScannerCombParameters()

    raw_spectrum = compute_raw_comb_spectrum(raw, parameters)
    epoch_spectrum = compute_epoch_comb_spectrum(epochs, parameters)

    np.testing.assert_array_equal(raw_spectrum.frequencies_hz, epoch_spectrum.frequencies_hz)


def test_epoch_spectrum_is_computed_before_epoch_aggregation(monkeypatch) -> None:
    epochs = _epochs_with_distinct_edge_values()
    spectrum = compute_epoch_comb_spectrum(epochs, ScannerCombParameters())
    assert spectrum.power_db.shape == spectrum.frequencies_hz.shape
    assert np.isfinite(spectrum.power_db).all()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/analysis/test_scanner_harmonic_comb.py -q`

Expected: collection failure because `scanner_harmonic_comb` does not exist.

- [ ] **Step 3: Implement strict parameters and MNE-object spectrum functions**

```python
@dataclass(frozen=True)
class ScannerCombParameters:
    frequency_min_hz: float = 15.0
    frequency_max_hz: float = 90.0
    welch_duration_seconds: float = 4.0
    frequency_resolution_hz: float = 0.25
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class ParticipantSpectrum:
    participant: str
    frequencies_hz: np.ndarray
    power_db: np.ndarray


def compute_raw_comb_spectrum(
    raw: mne.io.BaseRaw,
    parameters: ScannerCombParameters,
) -> tuple[np.ndarray, np.ndarray]:
    return _compute_welch_spectrum(raw, parameters, include_epochs=False)


def compute_epoch_comb_spectrum(
    epochs: mne.BaseEpochs,
    parameters: ScannerCombParameters,
) -> tuple[np.ndarray, np.ndarray]:
    return _compute_welch_spectrum(epochs, parameters, include_epochs=True)
```

Use `compute_psd(method="welch", average="median")`, EEG picks excluding bads, `n_per_seg = welch_duration_seconds * sfreq`, and `n_fft = sfreq / frequency_resolution_hz`. Validate integer sample counts, finite positive linear power, non-empty channels, and non-empty epochs before converting the median to dB.

- [ ] **Step 4: Write failing participant-equal and paired-bootstrap tests**

```python
def test_cohort_summary_weights_participants_equally() -> None:
    input_spectra = [_participant("01", 1.0), _participant("02", 9.0)]
    final_spectra = [_participant("01", 0.5), _participant("02", 4.5)]
    summary = summarize_scanner_comb(input_spectra, final_spectra, _parameters(100))
    np.testing.assert_allclose(summary.input_median_db, np.median([[1.0], [9.0]], axis=0))


def test_bootstrap_is_deterministic_and_pairs_participants() -> None:
    first = summarize_scanner_comb(_inputs(), _finals(), _parameters(200))
    second = summarize_scanner_comb(_inputs(), _finals(), _parameters(200))
    np.testing.assert_array_equal(first.input_ci_low_db, second.input_ci_low_db)
    np.testing.assert_array_equal(first.final_ci_high_db, second.final_ci_high_db)
```

- [ ] **Step 5: Implement hierarchical participant and cohort summaries**

```python
@dataclass(frozen=True)
class ScannerCombSummary:
    participant_ids: tuple[str, ...]
    frequencies_hz: np.ndarray
    input_median_db: np.ndarray
    input_ci_low_db: np.ndarray
    input_ci_high_db: np.ndarray
    final_median_db: np.ndarray
    final_ci_low_db: np.ndarray
    final_ci_high_db: np.ndarray
    harmonic_frequencies_hz: tuple[float, ...]


def combine_participant_runs(
    participant: str,
    run_spectra: Sequence[tuple[np.ndarray, np.ndarray]],
) -> ParticipantSpectrum:
    _require_identical_frequency_grids(run_spectra)
    return ParticipantSpectrum(
        participant=participant,
        frequencies_hz=run_spectra[0][0],
        power_db=np.median(np.stack([item[1] for item in run_spectra]), axis=0),
    )
```

Pair stages by exact participant ID, require identical participants and frequency grids, resample participant indices once per bootstrap replicate for both stages, and choose the strongest input peak within each `DEFAULT_HARMONIC_WINDOWS` window using the established scanner-harmonic peak rule.

- [ ] **Step 6: Run analysis tests and commit**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/analysis/test_scanner_harmonic_comb.py tests/analysis/test_scanner_harmonics.py -q`

Expected: all tests pass.

Commit:

```bash
git add eeg_pipeline/analysis/qc/scanner_harmonic_comb.py tests/analysis/test_scanner_harmonic_comb.py
git commit -m "feat: compute MNE scanner harmonic comb spectra"
```

### Task 2: Publication Comb Figure and TSV Contract

**Files:**
- Create: `eeg_pipeline/plotting/scanner_harmonic_comb.py`
- Create: `tests/plotting/test_scanner_harmonic_comb_plot.py`

- [ ] **Step 1: Write failing figure and output tests**

```python
def test_comb_figure_contains_two_lines_four_windows_and_references() -> None:
    figure = build_scanner_harmonic_comb_figure(_summary(), task="thermalactive")
    axis = figure.axes[0]
    assert len(axis.lines) == 6
    assert len(axis.patches) == 4
    assert axis.get_xlim() == pytest.approx((15.0, 90.0))


def test_write_scanner_harmonic_comb_outputs_png_and_tsv(tmp_path: Path) -> None:
    png_path, tsv_path = write_scanner_harmonic_comb(
        _summary(), output_dir=tmp_path, task="thermalactive"
    )
    assert png_path.name == "task-thermalactive_desc-scannerharmoniccomb_qc.png"
    frame = pd.read_csv(tsv_path, sep="\t")
    assert frame.columns.tolist() == EXPECTED_COLUMNS
```

- [ ] **Step 2: Run plotting tests and verify RED**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/plotting/test_scanner_harmonic_comb_plot.py -q`

Expected: collection failure because the plotting module does not exist.

- [ ] **Step 3: Implement the one-panel figure and exact writer**

```python
def build_scanner_harmonic_comb_figure(
    summary: ScannerCombSummary,
    *,
    task: str,
) -> Figure:
    figure = Figure(figsize=(12.0, 5.5), layout="constrained", facecolor="white")
    axis = figure.subplots()
    _plot_confidence_spectrum(axis, summary, stage="input")
    _plot_confidence_spectrum(axis, summary, stage="final")
    _plot_harmonic_windows(axis, summary.harmonic_frequencies_hz)
    axis.set(xlim=(15.0, 90.0), xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)")
    return figure


def write_scanner_harmonic_comb(
    summary: ScannerCombSummary,
    *,
    output_dir: Path,
    task: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task-{task}_desc-scannerharmoniccomb_qc"
    _write_summary_tsv(summary, output_dir / f"{stem}.tsv")
    _save_figure(build_scanner_harmonic_comb_figure(summary, task=task), output_dir / f"{stem}.png")
    return output_dir / f"{stem}.png", output_dir / f"{stem}.tsv"
```

Use the native QC visual language: neutral input, blue final, translucent bootstrap bands, subtle neutral harmonic spans, thin dotted references, frameless legend, restrained grid, explicit participant count, and 300 DPI.

- [ ] **Step 4: Run plotting tests and commit**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/plotting/test_scanner_harmonic_comb_plot.py -q`

Expected: all tests pass.

Commit:

```bash
git add eeg_pipeline/plotting/scanner_harmonic_comb.py tests/plotting/test_scanner_harmonic_comb.py
git commit -m "feat: plot cohort scanner harmonic comb"
```

### Task 3: Strict Preprocessing QC Orchestration

**Files:**
- Create: `eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py`
- Create: `tests/preprocessing/test_scanner_harmonic_qc.py`

- [ ] **Step 1: Write failing discovery and end-to-end orchestration tests**

```python
def test_run_scanner_harmonic_qc_requires_every_subject_stage(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sub-0002.*clean epochs"):
        run_scanner_harmonic_qc(
            subjects=["0001", "0002"],
            task="thermalactive",
            bids_root=tmp_path / "bids",
            deriv_root=tmp_path / "derivatives",
            parameters=_parameters(),
        )


def test_run_scanner_harmonic_qc_writes_task_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "_load_input_spectra", lambda **kwargs: _input_spectra())
    monkeypatch.setattr(module, "_load_final_spectrum", lambda **kwargs: _final_spectrum())
    outputs = run_scanner_harmonic_qc(
        subjects=["0001"],
        task="thermalactive",
        bids_root=tmp_path / "bids",
        deriv_root=tmp_path / "derivatives",
        parameters=_parameters(),
    )
    assert outputs.png_path.parent.name == "qc"
    assert outputs.tsv_path.exists()
```

- [ ] **Step 2: Run orchestration tests and verify RED**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/preprocessing/test_scanner_harmonic_qc.py -q`

Expected: collection failure because the orchestration module does not exist.

- [ ] **Step 3: Implement exact BIDS and clean-epoch discovery**

```python
@dataclass(frozen=True)
class ScannerHarmonicQcOutputs:
    png_path: Path
    tsv_path: Path


def run_scanner_harmonic_qc(
    *,
    subjects: Sequence[str],
    task: str,
    bids_root: Path,
    deriv_root: Path,
    parameters: ScannerCombParameters,
) -> ScannerHarmonicQcOutputs:
    input_spectra = [_load_input_participant(subject, task, bids_root, parameters) for subject in subjects]
    final_spectra = [_load_final_participant(subject, task, deriv_root, parameters) for subject in subjects]
    summary = summarize_scanner_comb(input_spectra, final_spectra, parameters)
    png_path, tsv_path = write_scanner_harmonic_comb(
        summary,
        output_dir=deriv_root / "preprocessed" / "eeg" / "qc",
        task=task,
    )
    return ScannerHarmonicQcOutputs(png_path=png_path, tsv_path=tsv_path)
```

Discover visible run-level EEG data through MNE-BIDS matching, require at least one run per selected participant, read through `read_raw_bids`, and use `find_clean_epochs_path` for the final stage. Every selected participant must contribute both stages.

- [ ] **Step 4: Run orchestration tests and commit**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/preprocessing/test_scanner_harmonic_qc.py -q`

Expected: all tests pass.

Commit:

```bash
git add eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py tests/preprocessing/test_scanner_harmonic_qc.py
git commit -m "feat: orchestrate scanner harmonic cohort QC"
```

### Task 4: Pipeline Step, Configuration, and Metadata

**Files:**
- Modify: `eeg_pipeline/pipelines/preprocessing.py`
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml`
- Modify: `tests/pipelines/test_pipeline_preprocessing.py`

- [ ] **Step 1: Write failing step-selection and metadata tests**

```python
def test_task_full_mode_appends_scanner_harmonic_qc() -> None:
    pipeline = _pipeline()
    steps = pipeline._get_steps_for_run("full", task_is_rest=False)
    assert steps[-1] == "scanner-harmonic-qc"


def test_rest_and_incomplete_modes_do_not_run_scanner_harmonic_qc() -> None:
    pipeline = _pipeline()
    assert "scanner-harmonic-qc" not in pipeline._get_steps_for_run("full", task_is_rest=True)
    assert "scanner-harmonic-qc" not in pipeline._get_steps_for_run("ica", task_is_rest=False)


def test_batch_metadata_contains_scanner_harmonic_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "_run_scanner_harmonic_qc", lambda **kwargs: _outputs(tmp_path))
    pipeline.run_batch(["0001"], task="thermalactive", mode="epochs")
    assert written_metadata["outputs"]["scanner_harmonic_comb_png"].endswith(".png")
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/pipelines/test_pipeline_preprocessing.py -q`

Expected: failures because the run-specific step and QC method do not exist.

- [ ] **Step 3: Add config and strict parameter loader**

```yaml
  scanner_harmonic_qc:
    frequency_range_hz: [15.0, 90.0]
    welch_duration_seconds: 4.0
    frequency_resolution_hz: 0.25
    bootstrap_resamples: 10000
    confidence_level: 0.95
```

Map these exact keys plus `project.random_state` into `ScannerCombParameters`; reject missing, unknown, or invalid values.

- [ ] **Step 4: Integrate the final step and return metadata outputs**

```python
STEP_SCANNER_HARMONIC_QC = "scanner-harmonic-qc"


def _get_steps_for_run(self, mode: str, task_is_rest: bool) -> list[str]:
    steps = self._get_steps_for_mode(mode)
    if not task_is_rest and mode in {"full", "epochs"}:
        steps.append(STEP_SCANNER_HARMONIC_QC)
    return steps


def _run_scanner_harmonic_qc(self, subjects: list[str], task: str) -> dict[str, str]:
    outputs = run_scanner_harmonic_qc(
        subjects=subjects,
        task=task,
        bids_root=self.bids_root,
        deriv_root=self.deriv_root,
        parameters=self._scanner_comb_parameters(),
    )
    return {
        "scanner_harmonic_comb_png": str(outputs.png_path),
        "scanner_harmonic_comb_tsv": str(outputs.tsv_path),
    }
```

Make `_execute_steps` return an output dictionary. Pass it unchanged to `_write_run_metadata`; retain current behavior for every existing step.

- [ ] **Step 5: Run focused integration tests and commit**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/pipelines/test_pipeline_preprocessing.py tests/preprocessing/test_scanner_harmonic_qc.py tests/analysis/test_scanner_harmonic_comb.py tests/plotting/test_scanner_harmonic_comb_plot.py -q`

Expected: all tests pass.

Commit:

```bash
git add eeg_pipeline/pipelines/preprocessing.py eeg_pipeline/utils/config/eeg_config.yaml tests/pipelines/test_pipeline_preprocessing.py
git commit -m "feat: run scanner harmonic QC after MNE preprocessing"
```

### Task 5: Repository Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Format changed Python files**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m black eeg_pipeline/analysis/qc/scanner_harmonic_comb.py eeg_pipeline/plotting/scanner_harmonic_comb.py eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py eeg_pipeline/pipelines/preprocessing.py tests/analysis/test_scanner_harmonic_comb.py tests/plotting/test_scanner_harmonic_comb_plot.py tests/preprocessing/test_scanner_harmonic_qc.py tests/pipelines/test_pipeline_preprocessing.py`

- [ ] **Step 2: Run Ruff on changed Python files**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m ruff check eeg_pipeline/analysis/qc/scanner_harmonic_comb.py eeg_pipeline/plotting/scanner_harmonic_comb.py eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py eeg_pipeline/pipelines/preprocessing.py tests/analysis/test_scanner_harmonic_comb.py tests/plotting/test_scanner_harmonic_comb_plot.py tests/preprocessing/test_scanner_harmonic_qc.py tests/pipelines/test_pipeline_preprocessing.py`

Expected: no errors.

- [ ] **Step 3: Run focused and architecture verification**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/analysis/test_scanner_harmonics.py \
  tests/analysis/test_scanner_harmonic_comb.py \
  tests/plotting/test_scanner_harmonic_comb_plot.py \
  tests/preprocessing/test_scanner_harmonic_qc.py \
  tests/pipelines/test_pipeline_preprocessing.py -q
make verify-architecture
```

Expected: all tests and architecture checks pass.

- [ ] **Step 4: Inspect a rendered synthetic figure**

Generate the test summary PNG, inspect it for label overlap, confidence-band visibility, harmonic shading, and readable 15–90 Hz axes, then remove the temporary image.

- [ ] **Step 5: Commit final formatting or verification adjustments**

```bash
git add eeg_pipeline tests
git commit -m "test: verify MNE scanner harmonic comb QC"
```
