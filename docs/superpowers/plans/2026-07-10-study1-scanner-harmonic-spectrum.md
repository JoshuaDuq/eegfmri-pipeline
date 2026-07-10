# Study 1 Scanner-Harmonic Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one outcome-blind, article-quality supplementary figure that shows residual scanner-harmonic spectra and their agreement with MRI TR harmonics in final-clean Study 1 EEG.

**Architecture:** A standalone analysis module discovers numeric-subject final-clean FIF files, computes one robust Welch spectrum per run, aggregates runs within participants, and bootstraps participants as the cohort unit. A separate one-plot CLI renders a two-panel SVG and writes run- and participant-level audit tables; the general Study 1 report does not reread continuous EEG.

**Tech Stack:** Python 3.11+, MNE-Python, NumPy, SciPy, pandas, Matplotlib, Pytest, Ruff

---

## File structure

- Modify `studies/pain_study/study1/config/study1_figure_config.yaml`: add fixed scanner-spectrum acquisition, analysis, colors, and 183 × 82 mm dimensions.
- Modify `studies/pain_study/study1/figures/validity_style.py`: expose an atomic SVG writer that accepts explicit physical dimensions while preserving the existing validity writer.
- Create `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`: discovery, PSD/peak estimation, participant aggregation, bootstrap, audits, and rendering.
- Create `studies/pain_study/study1/figures/plot_scanner_harmonic_spectrum.py`: the only CLI/writer for the figure.
- Modify `studies/pain_study/study1/figures/__init__.py`: export the standalone writer and result paths.
- Create `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`: estimator and aggregation contracts.
- Create `studies/tests/pipelines/test_study1_scanner_harmonic_figure.py`: renderer, SVG, audit, and CLI contracts.
- Modify `studies/tests/config/test_study1_config_loader.py`: lock scanner-spectrum defaults.
- Modify `studies/pain_study/study1/README.md`: define the estimator and interpretation.
- Modify `studies/pain_study/study1/RUN_GUIDE.md`: document the standalone command and outputs.

### Task 1: Lock configuration and physical SVG output

**Files:**
- Modify: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Modify: `studies/pain_study/study1/figures/validity_style.py`
- Modify: `studies/tests/config/test_study1_config_loader.py`
- Test: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write failing configuration and explicit-size SVG tests**

Add assertions for this exact mapping:

```python
scanner = config["study1"]["figures"]["scanner_harmonics"]
assert scanner == {
    "dimensions_mm": {"width": 183.0, "height": 82.0},
    "frequency_range_hz": [15.0, 90.0],
    "n_fft": 8192,
    "n_overlap": 4096,
    "sampling_frequency_hz": 500.0,
    "peak_prominence_db": 1.0,
    "peak_distance_bins": 4,
    "volume_repetition_time_s": 0.9,
    "harmonic_orders": [18, 37, 55, 74],
    "excluded_subjects": ["sub-0006"],
    "colors": {"excluded": "#D55E00", "retained": "#0072B2"},
}
```

Add a physical-size test that saves a blank figure with:

```python
save_publication_svg(
    figure,
    output_path,
    config,
    dimensions_mm={"width": 183.0, "height": 82.0},
)
assert _svg_dimensions_mm(output_path) == pytest.approx((183.0, 82.0))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/matplotlib-study1-scanner \
  /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/config/test_study1_config_loader.py::test_load_study1_config_includes_scanner_harmonic_figure_defaults \
  studies/tests/pipelines/test_study1_validity_figures.py::test_publication_svg_respects_explicit_physical_dimensions -q
```

Expected: failures because the mapping and `save_publication_svg` do not exist.

- [ ] **Step 3: Add the fixed YAML mapping and generic atomic writer**

Append the mapping under `study1.figures`. Refactor SVG output without changing existing behavior:

```python
def save_publication_svg(
    figure: Figure,
    output_path: Path,
    config: Any,
    *,
    dimensions_mm: Mapping[str, float],
) -> Path:
    if output_path.suffix != ".svg":
        raise ValueError(f"Study 1 publication figures require an .svg path: {output_path}")
    width = float(dimensions_mm["width"]) / MILLIMETERS_PER_INCH
    height = float(dimensions_mm["height"]) / MILLIMETERS_PER_INCH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(width, height, forward=False)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".svg",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        with publication_style(config):
            figure.savefig(temporary_path, format="svg", metadata={"Date": None})
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)
    return output_path


def save_validity_svg(figure: Figure, output_path: Path, config: Any) -> Path:
    dimensions = require_config_value(config, "study1.figures.validity.dimensions_mm")
    return save_publication_svg(
        figure,
        output_path,
        config,
        dimensions_mm=dimensions,
    )
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the Task 1 command again. Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/study1/config/study1_figure_config.yaml \
  studies/pain_study/study1/figures/validity_style.py \
  studies/tests/config/test_study1_config_loader.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: configure Study 1 scanner spectrum figure"
```

### Task 2: Implement run discovery and spectral peak estimation

**Files:**
- Create: `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`
- Create: `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`

- [ ] **Step 1: Write failing discovery and peak tests**

Use temporary paths containing numeric participants, pilots, other tasks, and excluded subjects:

```python
paths = discover_final_clean_runs(
    tmp_path,
    task="thermalactive",
    excluded_subjects=("sub-0006",),
)
assert [path.name for path in paths] == [
    "sub-0001_task-thermalactive_run-1_proc-clean_raw.fif",
    "sub-0003_task-thermalactive_run-2_proc-clean_raw.fif",
]
```

Create a 0.061 Hz frequency grid and synthetic dB spectrum with peaks at 20, 41, 61, and 82 Hz:

```python
peaks = select_scanner_harmonic_peaks(frequencies, spectrum_db, specification)
assert [peak.window_name for peak in peaks] == [
    "scanner_18_23", "scanner_38_43", "scanner_56_67", "scanner_77_85"
]
assert [peak.peak_frequency_hz for peak in peaks] == pytest.approx([20, 41, 61, 82], abs=0.07)
assert all(peak.prominence_db >= 1.0 for peak in peaks)
```

Also test that no eligible peak raises `ValueError`, because errors must surface.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py \
  -k 'discover or peak' -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement focused domain types and pure functions**

Define immutable types and exact discovery:

```python
@dataclass(frozen=True)
class ScannerHarmonicSpecification:
    frequency_range_hz: tuple[float, float]
    n_fft: int
    n_overlap: int
    sampling_frequency_hz: float
    peak_prominence_db: float
    peak_distance_bins: int
    volume_repetition_time_s: float
    harmonic_orders: tuple[int, ...]
    excluded_subjects: tuple[str, ...]
    harmonic_windows: tuple[FrequencyWindow, ...] = DEFAULT_HARMONIC_WINDOWS


@dataclass(frozen=True)
class HarmonicPeak:
    window_name: str
    peak_frequency_hz: float
    prominence_db: float


def discover_final_clean_runs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[Path, ...]:
    subject_pattern = re.compile(r"^sub-\d+$")
    excluded = set(excluded_subjects)
    requested = set(requested_subjects)
    candidates = derivative_root.glob(
        f"sub-*/eeg/sub-*_task-{task}_run-*_proc-clean_raw.fif"
    )
    selected = []
    for path in candidates:
        subject = path.parts[-3]
        if subject_pattern.fullmatch(subject) is None or subject in excluded:
            continue
        if requested and subject not in requested:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"No numbered-participant final-clean FIF files found for task {task!r} in {derivative_root}."
        )
    return tuple(sorted(selected))
```

Add `scanner_harmonic_specification(config)` to read the exact YAML mapping, validate that harmonic
order count equals harmonic-window count, and return `ScannerHarmonicSpecification`. It must not
silently substitute missing configuration.

Implement peak selection using one `scipy.signal.find_peaks` call on the full spectrum, then the
greatest-prominence eligible peak per fixed window. Do not substitute window maxima when no peak
qualifies.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 2 command. Expected: all discovery and peak tests pass.

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/study1/figures/scanner_harmonic_spectrum.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py
git commit -m "feat: estimate Study 1 scanner harmonic spectra"
```

### Task 3: Aggregate participants and build paired bootstrap summaries

**Files:**
- Modify: `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`
- Modify: `studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py`

- [ ] **Step 1: Write failing aggregation tests with unequal run counts**

Construct two participant spectra with six runs for one participant and two for the other. Assert
that participant medians are computed first and the cohort median is not the eight-run pooled
median:

```python
summary = build_scanner_harmonic_summary(run_spectra, specification, bootstrap)
participant = summary.participant_spectra.pivot(
    index="subject_id", columns="frequency_hz", values="relative_psd_db"
)
assert participant.loc["sub-01"].to_numpy() == pytest.approx(expected_sub_01)
assert participant.loc["sub-02"].to_numpy() == pytest.approx(expected_sub_02)
assert summary.cohort_spectrum["median_relative_psd_db"].to_numpy() == pytest.approx(
    np.median(np.vstack([expected_sub_01, expected_sub_02]), axis=0)
)
```

Recreate the seeded participant resampling indices and assert the exact 2.5% and 97.5% quantiles
for both the spectrum and all peak offsets. This proves paired resampling across frequencies and
harmonic windows.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py \
  -k 'participant or bootstrap or audit' -q
```

Expected: missing summary/aggregation functions.

- [ ] **Step 3: Implement run estimation, aggregation, offsets, and audits**

Implement `estimate_run_spectrum` with `mne.io.read_raw_fif(path, preload=False)`, verify the exact
500 Hz sampling rate, compute Welch PSD over EEG picks, take the channel median in linear units,
and then convert to dB.

Use one shared bootstrap-index matrix for every cohort quantity:

```python
@dataclass(frozen=True)
class ParticipantBootstrapSpecification:
    iterations: int
    confidence_level: float
    seed: int


def validity_bootstrap_specification(config: Any) -> ParticipantBootstrapSpecification:
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    return ParticipantBootstrapSpecification(
        iterations=int(bootstrap["iterations"]),
        confidence_level=float(bootstrap["confidence_level"]),
        seed=int(bootstrap["seed"]),
    )


def paired_participant_bootstrap(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(iterations, values.shape[0]))
    estimates = np.median(values[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        np.median(values, axis=0),
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )
```

Build long-form participant spectra, run audit, participant audit, cohort spectrum, participant
offset, and cohort-offset tables in an immutable `ScannerHarmonicSummary`.

- [ ] **Step 4: Run the estimator test module and verify GREEN**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add studies/pain_study/study1/figures/scanner_harmonic_spectrum.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py
git commit -m "feat: summarize scanner spectra by participant"
```

### Task 4: Render and publish the standalone figure

**Files:**
- Modify: `studies/pain_study/study1/figures/scanner_harmonic_spectrum.py`
- Create: `studies/pain_study/study1/figures/plot_scanner_harmonic_spectrum.py`
- Modify: `studies/pain_study/study1/figures/__init__.py`
- Create: `studies/tests/pipelines/test_study1_scanner_harmonic_figure.py`

- [ ] **Step 1: Write failing renderer and writer tests**

Build a synthetic `ScannerHarmonicSummary` and assert:

```python
figure = build_scanner_harmonic_figure(summary, config)
assert len(figure.axes) == 2
assert figure.axes[0].get_xlabel() == "Frequency (Hz)"
assert figure.axes[0].get_ylabel() == "PSD relative to participant median (dB)"
assert figure.axes[1].get_ylabel() == "Peak offset from predicted\nTR harmonic (Hz)"
assert len(figure.axes[0].lines) >= summary.n_subjects + 4
assert len(figure.axes[1].collections) >= 5
```

Write twice and assert deterministic SVG bytes, editable text, exact `183mm × 82mm`, and the
presence of `scanner_harmonic_spectrum_by_run.tsv` and
`scanner_harmonic_spectrum_by_subject.tsv` with exact schemas.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/matplotlib-study1-scanner \
  /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_scanner_harmonic_figure.py -q
```

Expected: missing renderer and CLI writer.

- [ ] **Step 3: Implement the exact two-panel renderer**

Use `Figure` plus a 2.25:1 `GridSpec`; do not call pyplot stateful plotting. Render participant
spectra, cohort median and CI, four pale vermillion harmonic spans, three blue retained-gamma
segments, participant offsets, hollow cohort diamonds and CIs, zero line, and the one-bin band.
Add panel labels `a` and `b`, `n`/run annotation, a frame-free legend, and no figure title.

- [ ] **Step 4: Implement the one-plot writer and CLI**

Expose:

```python
@dataclass(frozen=True)
class ScannerHarmonicFigurePaths:
    svg: Path
    run_tsv: Path
    run_parquet: Path
    participant_tsv: Path
    participant_parquet: Path


def write_scanner_harmonic_spectrum(
    *,
    derivative_root: Path,
    task: str,
    config: Any,
    subjects: Sequence[str] = (),
    output_path: Path | None = None,
) -> ScannerHarmonicFigurePaths:
    specification = scanner_harmonic_specification(config)
    derivative_root = Path(derivative_root)
    run_paths = discover_final_clean_runs(
        derivative_root,
        task=task,
        excluded_subjects=specification.excluded_subjects,
        requested_subjects=subjects,
    )
    run_spectra = tuple(
        estimate_run_spectrum(path, specification) for path in run_paths
    )
    summary = build_scanner_harmonic_summary(
        run_spectra,
        specification,
        bootstrap=validity_bootstrap_specification(config),
    )
    resolved_output = output_path or (
        validity_output_dir(config) / "scanner_harmonic_spectrum.svg"
    )
    figure = build_scanner_harmonic_figure(summary, config)
    save_publication_svg(
        figure,
        resolved_output,
        config,
        dimensions_mm=require_config_value(
            config,
            "study1.figures.scanner_harmonics.dimensions_mm",
        ),
    )

    run_tsv = resolved_output.with_name("scanner_harmonic_spectrum_by_run.tsv")
    run_parquet = run_tsv.with_suffix(".parquet")
    participant_tsv = resolved_output.with_name(
        "scanner_harmonic_spectrum_by_subject.tsv"
    )
    participant_parquet = participant_tsv.with_suffix(".parquet")
    write_tsv(summary.run_audit, run_tsv)
    write_parquet(summary.run_audit, run_parquet)
    write_tsv(summary.participant_audit, participant_tsv)
    write_parquet(summary.participant_audit, participant_parquet)
    return ScannerHarmonicFigurePaths(
        svg=resolved_output,
        run_tsv=run_tsv,
        run_parquet=run_parquet,
        participant_tsv=participant_tsv,
        participant_parquet=participant_parquet,
    )
```

The CLI requires `--config` and `--task`, accepts `--study1-config`, `--derivative-root`,
`--subject` repeated, and `--output`. Without explicit paths it uses `resolve_eeg_deriv_root(config)`
and `validity_output_dir(config) / "scanner_harmonic_spectrum.svg"`.

- [ ] **Step 5: Run figure tests and verify GREEN**

Run the Task 4 test command. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/study1/figures/scanner_harmonic_spectrum.py \
  studies/pain_study/study1/figures/plot_scanner_harmonic_spectrum.py \
  studies/pain_study/study1/figures/__init__.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_figure.py
git commit -m "feat: render Study 1 scanner harmonic spectrum"
```

### Task 5: Document, run real data, and verify publication quality

**Files:**
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`

- [ ] **Step 1: Document the exact estimator and standalone command**

State the participant-first aggregation, constant spectrum centering, bootstrap unit, four harmonic
windows, three retained gamma windows, 0.9 s TR comparison, and why the 18–23 Hz peak qualifies beta
interpretation. Add the command:

```bash
"$PYTHON" -m studies.pain_study.study1.figures.plot_scanner_harmonic_spectrum \
  --config "$EEG_CONFIG" \
  --study1-config "$STUDY1_CONFIG" \
  --task thermalactive \
  --derivative-root "$DERIV_ROOT"
```

- [ ] **Step 2: Run lint, focused tests, pipeline tests, and repository gates**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m ruff check \
  studies/pain_study/study1/figures studies/tests/pipelines \
  studies/tests/config/test_study1_config_loader.py
env MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/matplotlib-study1-scanner \
  /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_scanner_harmonic_spectrum.py \
  studies/tests/pipelines/test_study1_scanner_harmonic_figure.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/config/test_study1_config_loader.py -q
env MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/matplotlib-study1-scanner \
  /Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  studies/tests/pipelines -q
make PYTHON=/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python verify-structure
make PYTHON=/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python verify-architecture
make PYTHON=/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python verify-maintainability
git diff --check
```

Expected: Ruff clean; all tests and gates pass; no whitespace errors.

- [ ] **Step 3: Generate the real 13-participant/77-run figure**

Run the standalone CLI against:

```text
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg
```

Expected audit: 13 numbered participants, 77 runs, and finite qualifying peaks in every window for
every run.

- [ ] **Step 4: Perform visual and numeric QA**

Render the SVG to a 400 dpi PNG and inspect it at both full size and 100% publication size. Confirm:

- participant traces remain visible but subordinate to the cohort median;
- all text is legible at 183 × 82 mm;
- no CI, annotation, or legend is clipped;
- harmonic spans do not obscure the spectra;
- retained-gamma segments are distinguishable without relying on hue alone;
- all participant median peak offsets lie within a scientifically interpretable range;
- the plotted cohort values equal the TSV audit values.

- [ ] **Step 5: Commit documentation and final verification state**

```bash
git add studies/pain_study/study1/README.md studies/pain_study/study1/RUN_GUIDE.md
git commit -m "docs: publish Study 1 scanner harmonic QC figure"
```
