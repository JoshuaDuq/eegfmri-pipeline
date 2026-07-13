# Study 1 Cohort Power Spectral Density Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 1–90 Hz participant-first cohort PSD figure and complete audit-table family from Study 1 final-clean continuous EEG runs.

**Architecture:** Extract the existing final-clean run discovery and channel-median Welch boundary into a neutral continuous-spectrum module used by both scanner-harmonic and cohort analyses. Keep cohort aggregation, plotting, and artifact writing in separate focused modules, with strict validation and no interpolation, skipped runs, or fallback settings.

**Tech Stack:** Python 3.11, MNE-Python, NumPy, pandas, Matplotlib, PyYAML, pytest, Ruff.

---

## File map

- Create `studies/pain_study/study1/figures/continuous_spectrum.py`: final-clean discovery, filename parsing, `BAD`-annotation duration, and robust channel-median Welch estimation.
- Create `studies/pain_study/study1/figures/spectral_statistics.py`: shared participant-bootstrap specification and paired percentile bootstrap.
- Modify `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`: consume the shared continuous-spectrum boundary while preserving its API and outputs.
- Create `studies/pain_study/study1/figures/cohort_power_spectral_density.py`: configuration contract, participant-first aggregation, bootstrap summary, and audit tables.
- Create `studies/pain_study/study1/figures/cohort_power_spectral_density_plot.py`: pure publication figure rendering.
- Create `studies/pain_study/study1/figures/plot_cohort_power_spectral_density.py`: artifact writer and standalone CLI.
- Modify `studies/pain_study/study1/config/study1_figure_config.yaml`: dedicated PSD settings.
- Modify `studies/tests/config/test_study1_config_loader.py`: exact configuration-default contract.
- Create `studies/tests/pipelines/test_study1_continuous_spectrum.py`: shared discovery, validation, annotations, and Welch tests.
- Modify `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`: scanner regression against the shared boundary.
- Create `studies/tests/pipelines/test_study1_cohort_power_spectral_density.py`: cohort analysis tests.
- Create `studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py`: rendering, writer, schema, determinism, and CLI tests.

### Task 1: Lock the configuration contract

**Files:**
- Modify: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Modify: `studies/tests/config/test_study1_config_loader.py`

- [ ] **Step 1: Write the failing default-config test**

```python
def test_load_study1_config_includes_cohort_power_spectral_density_defaults() -> None:
    config = load_study1_config()

    assert config["study1"]["figures"]["cohort_power_spectral_density"] == {
        "dimensions_mm": {"width": 183.0, "height": 92.0},
        "frequency_range_hz": [1.0, 90.0],
        "n_fft": 8192,
        "n_overlap": 4096,
        "sampling_frequency_hz": 500.0,
        "colors": {
            "scanner_window": "#D55E00",
            "neural_band": "#0072B2",
        },
    }
    assert config["study1"]["figures"]["continuous_spectrum"] == {
        "excluded_subjects": ["sub-0006"],
    }
```

- [ ] **Step 2: Run the test and confirm the missing mapping fails**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py::test_load_study1_config_includes_cohort_power_spectral_density_defaults -v`

Expected: `KeyError: 'cohort_power_spectral_density'`.

- [ ] **Step 3: Add the exact YAML mapping**

```yaml
    cohort_power_spectral_density:
      dimensions_mm:
        width: 183.0
        height: 92.0
      frequency_range_hz: [1.0, 90.0]
      n_fft: 8192
      n_overlap: 4096
      sampling_frequency_hz: 500.0
      colors:
        scanner_window: "#D55E00"
        neural_band: "#0072B2"
    continuous_spectrum:
      excluded_subjects: ["sub-0006"]
```

Remove `excluded_subjects` from `scanner_harmonics`; update its existing exact
default-config assertion so exclusions have one shared source.

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py::test_load_study1_config_includes_cohort_power_spectral_density_defaults -v`

Expected: one passing test.

### Task 2: Extract the strict continuous-run spectrum boundary

**Files:**
- Create: `studies/pain_study/study1/figures/continuous_spectrum.py`
- Create: `studies/pain_study/study1/figures/spectral_statistics.py`
- Create: `studies/tests/pipelines/test_study1_continuous_spectrum.py`
- Modify: `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`
- Modify: `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`

- [ ] **Step 1: Write failing tests for discovery, filename parsing, and `BAD` interval union**

```python
def test_discover_final_clean_runs_filters_numbered_requested_subjects(tmp_path: Path) -> None:
    expected = _touch_run(tmp_path, "sub-0001", run=2)
    _touch_run(tmp_path, "sub-pilot", run=1)
    _touch_run(tmp_path, "sub-0002", run=1)

    assert discover_final_clean_runs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=("sub-0002",),
        requested_subjects=("sub-0001",),
    ) == (expected,)


def test_bad_annotation_duration_uses_clipped_interval_union() -> None:
    duration = bad_annotation_duration_s(
        onsets_s=np.asarray([-1.0, 2.0, 3.0, 8.0]),
        durations_s=np.asarray([2.0, 3.0, 2.0, 5.0]),
        descriptions=("BAD_edge", "BAD_motion", "BAD_motion", "stimulus"),
        recording_start_s=0.0,
        recording_duration_s=10.0,
    )

    assert duration == pytest.approx(4.0)
```

- [ ] **Step 2: Run those tests and confirm import failures**

Run: `python -m pytest studies/tests/pipelines/test_study1_continuous_spectrum.py -v`

Expected: collection fails because `continuous_spectrum` does not exist.

- [ ] **Step 3: Implement focused types and pure validation helpers**

```python
@dataclass(frozen=True)
class ContinuousSpectrumSpecification:
    frequency_range_hz: tuple[float, float]
    n_fft: int
    n_overlap: int
    sampling_frequency_hz: float


@dataclass(frozen=True)
class ContinuousRunSpectrum:
    subject_id: str
    run_id: str
    source_file: Path
    frequencies_hz: np.ndarray
    median_psd_v2_hz: np.ndarray
    n_channels: int
    sampling_frequency_hz: float
    n_samples: int
    recording_duration_s: float
    bad_annotation_duration_s: float
    analyzed_duration_s: float
```

Implement `discover_final_clean_runs`, `parse_final_clean_filename`,
`bad_annotation_duration_s`, and validation for frequency bounds, FFT size,
overlap, sampling rate, run length, PSD shape, and positive finite power.

- [ ] **Step 4: Add a failing Welch-boundary test**

```python
def test_estimate_continuous_run_spectrum_uses_linear_channel_median(monkeypatch, tmp_path):
    result = estimate_continuous_run_spectrum(_touch_run(tmp_path, "sub-0001", run=3), SPEC)

    assert result.median_psd_v2_hz == pytest.approx(EXPECTED_MEDIAN)
    assert raw.compute_kwargs == {
        "method": "welch",
        "fmin": 1.0,
        "fmax": 90.0,
        "n_fft": 8192,
        "n_per_seg": 8192,
        "n_overlap": 4096,
        "picks": "eeg",
        "reject_by_annotation": True,
        "verbose": False,
    }
```

- [ ] **Step 5: Implement MNE loading and Welch estimation**

Read with `mne.io.read_raw_fif(source_path, preload=False, verbose="ERROR")`, require the
configured sampling rate and at least `n_fft` samples, call `raw.compute_psd` with
the exact tested arguments, take the channel median in linear units, and calculate
recording, bad-annotation, and analyzed durations.

- [ ] **Step 6: Refactor scanner estimation to consume the shared result**

Import `discover_final_clean_runs` into the scanner module namespace and make
`estimate_run_spectrum` construct a `ContinuousSpectrumSpecification`, call
`estimate_continuous_run_spectrum`, convert its median to dB, and perform the
unchanged peak selection. Keep scanner dataclasses, public names, relative
normalization, audits, and rendering unchanged.

Move `ParticipantBootstrapSpecification` and `paired_participant_bootstrap` into
`spectral_statistics.py`, then import them into the scanner module namespace so
existing callers keep the same public API without wrapper functions.

- [ ] **Step 7: Run shared and scanner regression tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_continuous_spectrum.py studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_scanner_harmonic_figure.py -v`

Expected: all tests pass with no changed scanner artifact contract.

### Task 3: Implement participant-first cohort analysis

**Files:**
- Create: `studies/pain_study/study1/figures/cohort_power_spectral_density.py`
- Create: `studies/tests/pipelines/test_study1_cohort_power_spectral_density.py`

- [ ] **Step 1: Write failing specification and aggregation tests**

```python
def test_cohort_psd_specification_loads_fixed_settings() -> None:
    specification = cohort_psd_specification(load_study1_config())
    assert specification.spectrum.frequency_range_hz == (1.0, 90.0)
    assert specification.spectrum.n_fft == 8192
    assert specification.excluded_subjects == ("sub-0006",)


def test_build_cohort_psd_summary_aggregates_runs_in_linear_units() -> None:
    summary = build_cohort_psd_summary(
        (_run("sub-01", 1, [1.0, 9.0]), _run("sub-01", 2, [9.0, 1.0]),
         _run("sub-02", 1, [4.0, 4.0])),
        bootstrap=ParticipantBootstrapSpecification(iterations=200, confidence_level=0.95, seed=42),
    )
    subject = summary.participant_spectra.query("subject_id == 'sub-01'")
    assert subject["psd_db_uv2_hz"].to_numpy() == pytest.approx(
        10.0 * np.log10(np.asarray([5.0, 5.0]) * 1e12)
    )
```

- [ ] **Step 2: Run the tests and confirm the module import fails**

Run: `python -m pytest studies/tests/pipelines/test_study1_cohort_power_spectral_density.py -v`

Expected: collection fails because the cohort module does not exist.

- [ ] **Step 3: Implement the cohort contracts and summary**

```python
@dataclass(frozen=True)
class CohortPsdSpecification:
    spectrum: ContinuousSpectrumSpecification
    excluded_subjects: tuple[str, ...]


@dataclass(frozen=True)
class CohortPsdSummary:
    participant_spectra: pd.DataFrame
    cohort_spectrum: pd.DataFrame
    run_audit: pd.DataFrame

    @property
    def n_subjects(self) -> int:
        return int(self.participant_spectra["subject_id"].nunique())

    @property
    def n_runs(self) -> int:
        return int(len(self.run_audit))
```

Require non-empty runs and exact frequency equality. Pointwise-median runs in
linear power within participant, convert with `10 * log10(value * 1e12)`, then
call the shared paired participant bootstrap on the participant-by-frequency
matrix. Build deterministic long-form participant, cohort, and run-audit tables.

- [ ] **Step 4: Add failure tests for empty runs, frequency mismatch, and invalid power**

Each test must assert its precise `ValueError` message; do not introduce skipped
runs or interpolation.

- [ ] **Step 5: Run the cohort analysis tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_cohort_power_spectral_density.py -v`

Expected: all tests pass.

### Task 4: Render the scientific figure

**Files:**
- Create: `studies/pain_study/study1/figures/cohort_power_spectral_density_plot.py`
- Create: `studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py`

- [ ] **Step 1: Write failing structure and annotation tests**

```python
def test_build_cohort_psd_figure_has_publication_structure() -> None:
    figure = build_cohort_psd_figure(_summary(), load_study1_config())
    spectrum_axis, band_axis = figure.axes

    assert spectrum_axis.get_xlabel() == "Frequency (Hz)"
    assert spectrum_axis.get_ylabel() == "PSD (dB µV²/Hz)"
    assert spectrum_axis.get_xlim() == pytest.approx((1.0, 90.0))
    assert len(spectrum_axis.lines) == _summary().n_subjects + 1
    assert len(spectrum_axis.collections) >= 1
    assert [text.get_text() for text in band_axis.texts] == [
        "δ", "θ", "α", "β", "low γ", "mid γ", "high γ"
    ]
```

- [ ] **Step 2: Run the test and confirm the plot module is missing**

Run: `python -m pytest studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py::test_build_cohort_psd_figure_has_publication_structure -v`

Expected: import failure.

- [ ] **Step 3: Implement the pure renderer**

Use `publication_style`, `figure_size_inches`, `DEFAULT_HARMONIC_WINDOWS`, and
`SCANNER_CLEAN_GAMMA_RANGES_HZ`. Render pale scanner windows, bootstrap fill,
sorted participant traces, black cohort median, participant/run annotation, an
external legend, and a small share-x band strip with exact Study 1 ranges. Keep
all I/O outside this module.

- [ ] **Step 4: Add tests for legend placement, exact bands, and deterministic layout**

Assert the legend belongs to the figure, not either axes; the band strip contains
the seven exact intervals; and no top/right spines are visible.

- [ ] **Step 5: Run the renderer tests**

Run: `python -m pytest studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py -k 'build_cohort_psd' -v`

Expected: all selected tests pass.

### Task 5: Write the artifact family and CLI

**Files:**
- Create: `studies/pain_study/study1/figures/plot_cohort_power_spectral_density.py`
- Modify: `studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py`

- [ ] **Step 1: Write the failing writer test**

Construct a deterministic synthetic summary, replace the analysis builder, call
the writer twice into separate directories, and assert identical SVG bytes,
183 × 92 mm dimensions, embedded SVG text, exact seven filenames, and TSV/Parquet
column parity for run, participant, and cohort tables.

- [ ] **Step 2: Implement the writer contract**

```python
@dataclass(frozen=True)
class CohortPsdFigurePaths:
    svg: Path
    run_tsv: Path
    run_parquet: Path
    participant_tsv: Path
    participant_parquet: Path
    summary_tsv: Path
    summary_parquet: Path
```

Implement `_build_summary` from discovered final-clean runs and
`estimate_continuous_run_spectrum`. Implement `write_cohort_power_spectral_density`
with `save_publication_svg`, `write_tsv`, and `write_parquet`.

- [ ] **Step 3: Write the failing CLI test**

Call `main` with `--config`, `--task`, `--derivative-root`, and `--output` while
replacing config loading and the writer. Assert the returned paths and printed SVG
path. Run the real module with `--help` under `-W error::RuntimeWarning`.

- [ ] **Step 4: Implement the standalone CLI**

Use the established Study 1 figure CLI arguments and resolve the EEG derivative
root only when `--derivative-root` is absent. Do not catch analysis errors.

- [ ] **Step 5: Run the complete focused suite**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_continuous_spectrum.py studies/tests/pipelines/test_study1_cohort_power_spectral_density.py studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_scanner_harmonic_figure.py -v`

Expected: all tests pass.

### Task 6: Generate and inspect the Study 1 artifact

**Files:**
- Generated under the configured Study 1 supplementary validity output directory.

- [ ] **Step 1: Run formatting and static checks**

Run: `ruff check studies/pain_study/study1/figures studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_*spectrum*.py studies/tests/pipelines/test_study1_cohort_power_spectral_density*.py`

Run: `black --check studies/pain_study/study1/figures studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_*spectrum*.py studies/tests/pipelines/test_study1_cohort_power_spectral_density*.py`

Expected: both commands exit zero.

- [ ] **Step 2: Run repository gates**

Run: `make verify-architecture && make verify-maintainability`

Expected: both gates pass.

- [ ] **Step 3: Generate from available Study 1 final-clean data**

Run the new module with the repository pipeline config, Study 1 config, task
`thermalactive`, and the mounted EEG derivative root. Expected: one SVG and six
audit tables with no skipped or interpolated runs.

- [ ] **Step 4: Render the SVG to PNG and inspect it**

Verify participant traces are legible, the cohort and confidence interval are
dominant, the 1–90 Hz range and units are correct, annotations do not obscure the
spectrum, labels do not overlap, and all seven band labels are readable.

- [ ] **Step 5: Run the final focused test and cleanliness checks**

Run: `python -m pytest studies/tests/config/test_study1_config_loader.py studies/tests/pipelines/test_study1_continuous_spectrum.py studies/tests/pipelines/test_study1_cohort_power_spectral_density.py studies/tests/pipelines/test_study1_cohort_power_spectral_density_figure.py studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py studies/tests/pipelines/test_study1_scanner_harmonic_figure.py -q`

Run: `git diff --check && git status --short`

Expected: tests pass, diff check is silent, and status contains only intended
implementation and generated artifact changes.
