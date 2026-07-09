# Study 1 Supplementary Validity Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate three standalone, Nature/Science-style supplementary SVG plots showing the behavioral, NPS, and SIIPS1 dose-response relationships in Study 1.

**Architecture:** A strict shared data boundary joins retained Study 1 target trials to clean events and computes equally weighted participant-temperature summaries with participant-cluster bootstrap intervals. One shared renderer implements the approved scientific grammar, while three small executable modules each own one outcome and one SVG filename. The existing report stage reuses the same validated trial object for tables and figures and records only supplementary SVG paths.

**Tech Stack:** Python 3.11+, pandas, NumPy, Matplotlib, PyYAML, pytest, Ruff, SVG

---

## File Structure

- Create `studies/pain_study/study1/config/study1_figure_config.yaml` for the fixed
  publication and bootstrap settings.
- Modify `studies/pain_study/study1/config/loader.py` to merge and validate the dedicated
  figure configuration.
- Modify `studies/pain_study/study1/config/__init__.py` to export the figure-config path.
- Modify `studies/tests/config/test_study1_config_loader.py` for strict figure-config tests.
- Create `studies/pain_study/study1/figures/__init__.py` for the public figure writers.
- Create `studies/pain_study/study1/figures/validity_data.py` for canonical trial loading,
  joining, aggregation, and participant-cluster bootstrap estimation.
- Create `studies/pain_study/study1/figures/validity_style.py` for font validation,
  publication dimensions, output paths, and atomic SVG saving.
- Create `studies/pain_study/study1/figures/dose_response.py` for the shared single-axis
  dose-response renderer.
- Create `studies/pain_study/study1/figures/plot_behavioral_dose_response.py` for the
  displayed-rating plot and its executable entry point.
- Create `studies/pain_study/study1/figures/plot_nps_dose_response.py` for the NPS plot and
  its executable entry point.
- Create `studies/pain_study/study1/figures/plot_siips1_dose_response.py` for the SIIPS1
  plot and its executable entry point.
- Create `studies/tests/pipelines/test_study1_validity_data.py` for data-contract and
  estimator tests.
- Create `studies/tests/pipelines/test_study1_validity_figures.py` for SVG, style, and
  plot-specific contract tests.
- Modify `studies/pain_study/study1/reporting.py` to reuse validated trial data, invoke the
  three writers, and add supplementary paths to the existing manifest.
- Modify Study 1 reporting tests under `studies/tests/pipelines/` and
  `tests/pipelines/test_study1_reporting.py` to use a small bootstrap budget and assert
  unchanged tables plus the new figure contract.
- Modify `studies/pain_study/study1/README.md` and
  `studies/pain_study/study1/RUN_GUIDE.md` to document the supplementary outputs.

### Task 1: Add and Validate the Dedicated Figure Configuration

**Files:**
- Create: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Modify: `studies/pain_study/study1/config/loader.py`
- Modify: `studies/pain_study/study1/config/__init__.py`
- Modify: `studies/tests/config/test_study1_config_loader.py`

- [ ] **Step 1: Write failing figure-configuration tests**

Add tests that require the dedicated defaults and reject malformed publication settings:

```python
def test_load_study1_config_includes_validity_figure_defaults() -> None:
    config = load_study1_config()
    validity = config["study1"]["figures"]["validity"]

    assert validity["temperatures"] == [44.3, 45.3, 46.3, 47.3, 48.3, 49.3]
    assert validity["dimensions_mm"] == {"width": 89.0, "height": 70.0}
    assert validity["font"]["family"] == "Arial"
    assert validity["bootstrap"] == {
        "iterations": 10000,
        "confidence_level": 0.95,
        "seed": 42,
        "max_invalid_fraction": 0.20,
    }
    assert validity["output_parts"] == [
        "reports",
        "figures",
        "supplementary",
        "validity",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperatures", [44.3, 44.3], "strictly increasing"),
        ("dimensions_mm.width", 0, "positive"),
        ("font.axis_label_pt", 8, "between 5 and 7"),
        ("bootstrap.iterations", 0, "positive integer"),
        ("bootstrap.confidence_level", 1.0, "between 0 and 1"),
        ("bootstrap.max_invalid_fraction", 1.0, r"in \[0, 1\)"),
        ("style.participant_alpha", 0.0, "in \(0, 1\]"),
        ("output_parts", ["reports", "../figures"], "path component"),
    ],
)
def test_load_study1_config_rejects_invalid_validity_figure_settings(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _complete_study1_yaml()
    _set_nested(config, f"study1.figures.validity.{field}", value)
    path = tmp_path / "invalid_figures.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_study1_config(path)
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/config/test_study1_config_loader.py \
  -q
```

Expected: the new defaults test fails with missing `study1.figures.validity`; malformed
figure settings are not yet validated.

- [ ] **Step 3: Add the YAML defaults and strict loader validation**

Create `study1_figure_config.yaml` with this complete mapping:

```yaml
study1:
  figures:
    validity:
      temperatures: [44.3, 45.3, 46.3, 47.3, 48.3, 49.3]
      dimensions_mm:
        width: 89.0
        height: 70.0
      font:
        family: "Arial"
        axis_label_pt: 7.0
        tick_label_pt: 6.0
        legend_pt: 6.0
        annotation_pt: 5.5
      style:
        participant_color: "#7F7F7F"
        participant_alpha: 0.22
        participant_line_width_pt: 0.45
        participant_marker_size_pt: 1.8
        cohort_line_width_pt: 1.2
        cohort_marker_size_pt: 3.0
        confidence_line_width_pt: 0.8
        axis_line_width_pt: 0.6
      colors:
        behavioral: "#222222"
        nps: "#0072B2"
        siips1: "#D55E00"
      bootstrap:
        iterations: 10000
        confidence_level: 0.95
        seed: 42
        max_invalid_fraction: 0.20
      output_parts:
        - "reports"
        - "figures"
        - "supplementary"
        - "validity"
```

In `loader.py`, load this file before the selected Study 1 YAML, merge the selected YAML
over the figure defaults with `_merge_non_null`, and call a new
`_validate_validity_figure_config`. The validator must require finite strictly increasing
temperatures, positive dimensions, 5–7 pt font sizes, valid `#RRGGBB` colours, positive
line and marker sizes, alpha in `(0, 1]`, a positive bootstrap count, confidence level in
`(0, 1)`, an integer seed, maximum invalid fraction in `[0, 1)`, and safe non-empty path
components without `/`, `\\`, `.`, or `..`.

Export the immutable default path:

```python
STUDY1_FIGURE_CONFIG_PATH = Path(__file__).with_name("study1_figure_config.yaml")
```

The merge order must be:

```python
figure_defaults = _load_yaml_mapping(STUDY1_FIGURE_CONFIG_PATH)
study_defaults = _load_yaml_mapping(resolved_path)
_merge_non_null(figure_defaults, study_defaults)
resolved = resolve_config_paths(figure_defaults, resolved_path)
```

- [ ] **Step 4: Run loader and package-data tests**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/config/test_study1_config_loader.py \
  tests/utils/test_cli_facade_hygiene.py \
  -q
```

Expected: PASS. `studies/pyproject.toml` already packages `config/*.yaml`, so no package-data
change is required.

- [ ] **Step 5: Commit the figure configuration contract**

```bash
git add studies/pain_study/study1/config/study1_figure_config.yaml \
  studies/pain_study/study1/config/loader.py \
  studies/pain_study/study1/config/__init__.py \
  studies/tests/config/test_study1_config_loader.py
git commit -m "feat: configure Study 1 validity figures"
```

### Task 2: Establish the Canonical Validity Trial Boundary

**Files:**
- Create: `studies/pain_study/study1/figures/__init__.py`
- Create: `studies/pain_study/study1/figures/validity_data.py`
- Create: `studies/tests/pipelines/test_study1_validity_data.py`
- Modify: `studies/pain_study/study1/reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_extended.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_full_picture.py`
- Modify: `tests/pipelines/test_study1_reporting.py`

- [ ] **Step 1: Write failing trial-loading and merge tests**

Build a two-subject, two-run fixture using the current target schema and clean-event schema.
Add these contract tests:

```python
def test_load_validity_trial_data_uses_only_retained_target_trials(tmp_path: Path) -> None:
    config = _validity_config(tmp_path)
    _write_targets(config, include_trials=(1, 2, 3))
    _write_events(config, include_trials=(1, 2, 3, 4))

    data = load_validity_trial_data(task="thermalactive", config=config)

    assert len(data.targets) == 6
    assert len(data.enriched_targets) == 6
    assert set(data.enriched_targets["trial_index"]) == {1, 2, 3}


def test_load_validity_trial_data_rejects_duplicate_target_keys(tmp_path: Path) -> None:
    config = _validity_config(tmp_path)
    _write_targets(config, duplicate_first_row=True)
    _write_events(config)

    with pytest.raises(ValueError, match="duplicate target trial keys"):
        load_validity_trial_data(task="thermalactive", config=config)


def test_load_validity_trial_data_rejects_duplicate_event_keys(tmp_path: Path) -> None:
    config = _validity_config(tmp_path)
    _write_targets(config)
    _write_events(config, duplicate_first_row=True)

    with pytest.raises(ValueError, match="duplicate clean-event trial keys"):
        load_validity_trial_data(task="thermalactive", config=config)


def test_load_validity_trial_data_rejects_unmatched_target_trial(tmp_path: Path) -> None:
    config = _validity_config(tmp_path)
    _write_targets(config)
    _write_events(config, omit_last_row=True)

    with pytest.raises(ValueError, match="without matching clean events"):
        load_validity_trial_data(task="thermalactive", config=config)


def test_load_validity_trial_data_rejects_temperature_disagreement(tmp_path: Path) -> None:
    config = _validity_config(tmp_path)
    _write_targets(config)
    _write_events(config, first_temperature=99.0)

    with pytest.raises(ValueError, match="stimulus temperature disagrees"):
        load_validity_trial_data(task="thermalactive", config=config)
```

Also cover a missing clean-event file, missing required columns, non-integer run/trial keys,
non-numeric values, non-finite values, and a target table containing a task other than the
requested task.

- [ ] **Step 2: Run the validity-data tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  -q
```

Expected: import failure because `validity_data.py` and `load_validity_trial_data` do not exist.

- [ ] **Step 3: Implement the immutable validated trial object**

Create the public boundary:

```python
@dataclass(frozen=True)
class ValidityTrialData:
    targets: pd.DataFrame
    clean_events: pd.DataFrame
    enriched_targets: pd.DataFrame


def load_validity_trial_data(*, task: str, config: Any) -> ValidityTrialData:
    targets = load_primary_target_table(config).copy()
    _require_single_task(targets, task)
    subjects = sorted(targets["subject_id"].astype(str).unique())
    clean_events = _load_clean_events(subjects=subjects, task=task, config=config)
    enriched_targets = _merge_targets_with_clean_events(targets, clean_events)
    return ValidityTrialData(
        targets=targets,
        clean_events=clean_events,
        enriched_targets=enriched_targets,
    )
```

Move the existing reporting implementations of `_load_article_clean_events`,
`_merge_targets_with_clean_events`, `_numeric_series`, and `_required_integer_series` into
focused helpers in `validity_data.py`. Before the pandas merge, materialize normalized
integer `_run_key` and `_within_run_trial_key` columns, require uniqueness on both sides,
and use `validate="one_to_one"` plus an indicator column. Require equality between target
and clean-event `stimulus_temp` for every matched row. Validate all plotting values
(`stimulus_temp`, `vas_final_coded_rating`, `NPS`, `SIIPS1`) as finite numeric values.

Keep the enriched event columns required by reporting:

```python
EVENT_COLUMNS = (
    "pain_binary_coded",
    "vas_final_coded_rating",
    "fp1_fp2_high_frequency_power",
    "stimulus_temp",
    "selected_surface",
    "residual_ecg_coupling",
)
```

- [ ] **Step 4: Refactor reporting to consume the shared object without changing tables**

Load `ValidityTrialData` once in `write_study1_report` and pass it to both table writers:

```python
trial_data = load_validity_trial_data(task=task, config=config)
_write_article_tables(
    frame=frame,
    trial_data=trial_data,
    config=config,
    report_root=report_root,
    report_path=tsv_path,
)
_write_full_picture_tables(
    frame=frame,
    trial_data=trial_data,
    config=config,
    report_root=report_root,
    report_path=tsv_path,
)
```

Delete the duplicate private loading and merge functions from `reporting.py`. Update test
fixtures to include a task-consistent target table and assert the existing article and
full-picture table frames are byte-for-byte equal before and after the refactor using fixed
expected frames.

- [ ] **Step 5: Run validity-data and reporting regression tests**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py \
  -q
```

Expected: PASS with unchanged report table schemas and values.

- [ ] **Step 6: Commit the canonical data boundary**

```bash
git add studies/pain_study/study1/figures/__init__.py \
  studies/pain_study/study1/figures/validity_data.py \
  studies/pain_study/study1/reporting.py \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py
git commit -m "refactor: share Study 1 validity trial data"
```

### Task 3: Implement Participant-Weighted Dose-Response Estimation

**Files:**
- Modify: `studies/pain_study/study1/figures/validity_data.py`
- Modify: `studies/tests/pipelines/test_study1_validity_data.py`

- [ ] **Step 1: Write failing aggregation and bootstrap tests**

Use unequal trial counts so a trial-weighted implementation produces the wrong answer:

```python
def test_build_dose_response_summary_weights_participants_equally() -> None:
    trials = pd.DataFrame(
        {
            "subject_id": ["sub-01"] * 4 + ["sub-02"] * 2,
            "stimulus_temp": [44.3, 44.3, 44.3, 49.3, 44.3, 49.3],
            "NPS": [0.0, 0.0, 0.0, 4.0, 2.0, 8.0],
        }
    )
    config = _figure_config(temperatures=[44.3, 49.3], iterations=20, seed=7)

    summary = build_dose_response_summary(trials, outcome="NPS", config=config)

    participant = summary.participant_means
    assert participant.loc[participant["stimulus_temp"] == 44.3, "value"].tolist() == [0.0, 2.0]
    cohort = summary.cohort_estimates.set_index("stimulus_temp")
    assert cohort.loc[44.3, "mean"] == pytest.approx(1.0)
    assert cohort.loc[49.3, "mean"] == pytest.approx(6.0)


def test_build_dose_response_summary_preserves_missing_cell_gap() -> None:
    trials = _three_subject_dose_response_trials().query(
        "not (subject_id == 'sub-03' and stimulus_temp == 49.3)"
    )
    summary = build_dose_response_summary(
        trials,
        outcome="NPS",
        config=_figure_config(temperatures=[44.3, 49.3]),
    )

    matrix = summary.participant_matrix
    assert np.isnan(matrix.loc["sub-03", 49.3])


def test_build_dose_response_summary_is_deterministic() -> None:
    config = _figure_config(temperatures=[44.3, 49.3], iterations=100, seed=19)
    first = build_dose_response_summary(_dose_response_trials(), "NPS", config)
    second = build_dose_response_summary(_dose_response_trials(), "NPS", config)

    pd.testing.assert_frame_equal(first.cohort_estimates, second.cohort_estimates)
```

Add tests for exact configured/observed temperature agreement, at least two participants per
temperature, no empty cells, valid outcome names, deterministic replacement of an invalid
bootstrap draw, failure when the configured invalid-draw budget is exceeded, and failure when
an accepted estimate or confidence bound is non-finite.

- [ ] **Step 2: Run the estimator tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  -q
```

Expected: failures because `DoseResponseSummary` and `build_dose_response_summary` are absent.

- [ ] **Step 3: Implement the explicit summary types and estimator**

Add:

```python
@dataclass(frozen=True)
class DoseResponseSummary:
    outcome: str
    temperatures: tuple[float, ...]
    participant_means: pd.DataFrame
    participant_matrix: pd.DataFrame
    cohort_estimates: pd.DataFrame


def build_dose_response_summary(
    trials: pd.DataFrame,
    outcome: str,
    config: Any,
) -> DoseResponseSummary:
    temperatures = _configured_temperatures(config)
    values = _validated_outcome_frame(trials, outcome, temperatures)
    participant_means = (
        values.groupby(["subject_id", "stimulus_temp"], sort=True, as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "value"})
    )
    matrix = participant_means.pivot(
        index="subject_id",
        columns="stimulus_temp",
        values="value",
    ).reindex(columns=temperatures)
    cohort = _participant_bootstrap_estimates(matrix, config)
    return DoseResponseSummary(
        outcome=outcome,
        temperatures=temperatures,
        participant_means=participant_means,
        participant_matrix=matrix,
        cohort_estimates=cohort,
    )
```

`_participant_bootstrap_estimates` must sample row indices with
`np.random.default_rng(seed).integers` and use the same sampled row indices for every
temperature. A draw with no finite selected value at any temperature is invalid. Resample
invalid draws until `iterations` valid draws have accumulated or until this explicit budget is
exhausted:

```python
max_attempts = math.ceil(iterations / (1.0 - max_invalid_fraction))
```

If the valid count is still short, raise a `ValueError` reporting valid, attempted, and invalid
draws. Compute `np.nanmean` only for accepted draws after validating at least two represented
participants per temperature, and reject any non-finite accepted estimate. Compute two-sided
percentile bounds with:

```python
tail = (1.0 - confidence_level) / 2.0
ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail], axis=0)
```

Return one row per configured temperature with `mean`, `ci_low`, `ci_high`, and
`n_subjects`.

- [ ] **Step 4: Run the complete estimator contract**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit the participant-weighted estimator**

```bash
git add studies/pain_study/study1/figures/validity_data.py \
  studies/tests/pipelines/test_study1_validity_data.py
git commit -m "feat: estimate Study 1 validity dose responses"
```

### Task 4: Build the Publication SVG Renderer

**Files:**
- Create: `studies/pain_study/study1/figures/validity_style.py`
- Create: `studies/pain_study/study1/figures/dose_response.py`
- Create: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write failing renderer and SVG tests**

Add tests for font failure, physical dimensions, editable text, scientific marks, and atomic
single-format output:

```python
def test_require_configured_font_rejects_missing_font(monkeypatch) -> None:
    def raise_missing(*args, **kwargs):
        raise ValueError("font missing")

    monkeypatch.setattr(font_manager, "findfont", raise_missing)
    with pytest.raises(ValueError, match="Required figure font 'Arial' is unavailable"):
        require_configured_font(_figure_config())


def test_build_dose_response_figure_draws_participants_estimate_and_ci() -> None:
    figure = build_dose_response_figure(
        summary=_summary_fixture(),
        specification=DoseResponseSpecification(
            ylabel="NPS expression (a.u.)",
            color_config_key="nps",
        ),
        config=_figure_config(),
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Temperature (°C)"
    assert axis.get_ylabel() == "NPS expression (a.u.)"
    gridlines = [*axis.get_xgridlines(), *axis.get_ygridlines()]
    assert not any(line.get_visible() for line in gridlines)
    assert len(axis.lines) >= 3
    assert len(axis.collections) >= 1


def test_save_validity_svg_writes_one_editable_89_mm_svg(tmp_path: Path) -> None:
    path = tmp_path / "nps_dose_response.svg"
    save_validity_svg(_figure_fixture(), path, _figure_config())

    root = ElementTree.parse(path).getroot()
    width_pt = float(root.attrib["width"].removesuffix("pt"))
    height_pt = float(root.attrib["height"].removesuffix("pt"))
    assert width_pt * 25.4 / 72.0 == pytest.approx(89.0, abs=0.01)
    assert height_pt * 25.4 / 72.0 == pytest.approx(70.0, abs=0.01)
    assert path.read_text(encoding="utf-8").count("<text") > 0
    assert sorted(item.suffix for item in tmp_path.iterdir()) == [".svg"]
```

Also test no title, no top/right spines, exact temperature ticks, outward tick direction,
data-driven signature y-limits containing all estimates and confidence bounds, cleanup of a
temporary file after a forced save error, and rejection of non-`.svg` destinations.

- [ ] **Step 2: Run the renderer tests and verify RED**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -q
```

Expected: import failures because the renderer and SVG writer do not exist.

- [ ] **Step 3: Implement publication style, output paths, and atomic SVG saving**

In `validity_style.py`, implement:

```python
MILLIMETERS_PER_INCH = 25.4


def validity_output_dir(config: Any) -> Path:
    parts = require_config_value(config, "study1.figures.validity.output_parts")
    return study1_output_root(config).joinpath(*parts)


def require_configured_font(config: Any) -> str:
    family = str(require_config_value(config, "study1.figures.validity.font.family"))
    try:
        font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Required figure font '{family}' is unavailable."
        ) from exc
    return family


def save_validity_svg(
    figure: Figure,
    output_path: Path,
    config: Any,
) -> Path:
    if output_path.suffix != ".svg":
        raise ValueError(f"Study 1 validity figures require an .svg path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".svg",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        figure.savefig(
            temporary_path,
            format="svg",
            metadata={"Date": None},
        )
        temporary_path.replace(output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)
    return output_path
```

Use `matplotlib.rc_context` with `svg.fonttype: "none"`, the validated Arial family,
configured 5–7 pt sizes, no grid, `axes.spines.top/right: False`, and configured 0.6 pt axes.
Build figures at `width_mm / 25.4` by `height_mm / 25.4` inches.

- [ ] **Step 4: Implement the shared single-axis dose-response renderer**

In `dose_response.py`, define explicit specifications without boolean flags:

```python
@dataclass(frozen=True)
class HorizontalReference:
    value: float
    label: str


@dataclass(frozen=True)
class DoseResponseSpecification:
    ylabel: str
    color_config_key: str
    y_limits: tuple[float, float] | None = None
    reference: HorizontalReference | None = None
```

`build_dose_response_figure` must:

1. draw each participant matrix row in configured gray with low alpha, small circular
   markers, and gaps at missing cells;
2. resolve the outcome colour from
   `study1.figures.validity.colors.<color_config_key>` and reject an unknown key;
3. draw vertical `ci_low`/`ci_high` whiskers with `Axes.errorbar`;
4. overlay the cohort mean line and markers in the outcome colour;
5. set only the exact configured temperature ticks;
6. use the fixed behavioral limits when provided, otherwise calculate a deterministic 5%
   vertical margin from participant values and confidence bounds;
7. add a light horizontal reference and black annotation only when a
   `HorizontalReference` object is present; and
8. omit figure and axis titles.

- [ ] **Step 5: Run renderer tests and inspect SVG structure**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -q
```

Expected: PASS. Generated test SVG contains editable `<text>` nodes and no sibling image
files.

- [ ] **Step 6: Commit the publication renderer**

```bash
git add studies/pain_study/study1/figures/validity_style.py \
  studies/pain_study/study1/figures/dose_response.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: render Study 1 validity SVGs"
```

### Task 5: Add the Behavioral Dose-Response Script

**Files:**
- Create: `studies/pain_study/study1/figures/plot_behavioral_dose_response.py`
- Modify: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write the failing behavioral-script tests**

```python
def test_write_behavioral_dose_response_creates_only_assigned_svg(tmp_path: Path) -> None:
    config, trial_data = _validity_fixture(tmp_path)

    output = write_behavioral_dose_response(trial_data=trial_data, config=config)

    assert output.name == "behavioral_dose_response.svg"
    assert output.parent.parts[-2:] == ("supplementary", "validity")
    assert sorted(output.parent.iterdir()) == [output]
    svg_text = " ".join(ElementTree.parse(output).getroot().itertext())
    assert "Displayed rating" in svg_text
    assert "Pain threshold" in svg_text
    assert "100" in svg_text
```

Also assert a 0–200 y-range, dark neutral cohort colour, and absence of NPS/SIIPS1 labels.

- [ ] **Step 2: Run the behavioral test and verify RED**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -k behavioral \
  -q
```

Expected: import failure because the behavioral module is absent.

- [ ] **Step 3: Implement the one-output behavioral writer and executable main**

Use one immutable specification:

```python
OUTPUT_FILENAME = "behavioral_dose_response.svg"
SPECIFICATION = DoseResponseSpecification(
    ylabel="Displayed rating",
    color_config_key="behavioral",
    y_limits=(0.0, 200.0),
    reference=HorizontalReference(value=100.0, label="Pain threshold"),
)


def write_behavioral_dose_response(
    *,
    trial_data: ValidityTrialData,
    config: Any,
) -> Path:
    summary = build_dose_response_summary(
        trial_data.enriched_targets,
        outcome="vas_final_coded_rating",
        config=config,
    )
    figure = build_dose_response_figure(summary, SPECIFICATION, config)
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )
```

Add this direct executable boundary:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 behavioral dose-response SVG."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    apply_study1_config_defaults(config, config_path=args.study1_config)
    trial_data = load_validity_trial_data(task=args.task, config=config)
    output_path = write_behavioral_dose_response(
        trial_data=trial_data,
        config=config,
    )
    print(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run behavioral tests**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -k behavioral \
  -q
```

Expected: PASS with exactly one SVG.

- [ ] **Step 5: Commit the behavioral plot**

```bash
git add studies/pain_study/study1/figures/plot_behavioral_dose_response.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: plot Study 1 behavioral validity"
```

### Task 6: Add the NPS Dose-Response Script

**Files:**
- Create: `studies/pain_study/study1/figures/plot_nps_dose_response.py`
- Modify: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write the failing NPS-script tests**

```python
def test_write_nps_dose_response_creates_only_assigned_svg(tmp_path: Path) -> None:
    config, trial_data = _validity_fixture(tmp_path)

    output = write_nps_dose_response(trial_data=trial_data, config=config)

    assert output.name == "nps_dose_response.svg"
    assert sorted(output.parent.iterdir()) == [output]
    svg = output.read_text(encoding="utf-8").lower()
    assert "nps expression (a.u.)" in svg
    assert "#0072b2" in svg
    assert "pain threshold" not in svg
    assert "siips1" not in svg
```

Also parse the y-axis limits from the returned figure in a unit-level test and prove they
contain every participant mean and confidence bound without forcing 0–200.

- [ ] **Step 2: Run the NPS test and verify RED**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -k nps \
  -q
```

Expected: import failure because the NPS module is absent.

- [ ] **Step 3: Implement the one-output NPS writer and executable main**

```python
OUTPUT_FILENAME = "nps_dose_response.svg"
SPECIFICATION = DoseResponseSpecification(
    ylabel="NPS expression (a.u.)",
    color_config_key="nps",
)


def write_nps_dose_response(
    *,
    trial_data: ValidityTrialData,
    config: Any,
) -> Path:
    summary = build_dose_response_summary(
        trial_data.targets,
        outcome="NPS",
        config=config,
    )
    figure = build_dose_response_figure(summary, SPECIFICATION, config)
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )
```

Add the plot-specific executable boundary:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 NPS dose-response SVG."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    apply_study1_config_defaults(config, config_path=args.study1_config)
    trial_data = load_validity_trial_data(task=args.task, config=config)
    output_path = write_nps_dose_response(
        trial_data=trial_data,
        config=config,
    )
    print(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run NPS and shared renderer tests**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -k "nps or renderer" \
  -q
```

Expected: PASS with exactly one NPS SVG for the NPS writer test.

- [ ] **Step 5: Commit the NPS plot**

```bash
git add studies/pain_study/study1/figures/plot_nps_dose_response.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: plot Study 1 NPS validity"
```

### Task 7: Add the SIIPS1 Dose-Response Script

**Files:**
- Create: `studies/pain_study/study1/figures/plot_siips1_dose_response.py`
- Modify: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write the failing SIIPS1-script tests**

```python
def test_write_siips1_dose_response_creates_only_assigned_svg(tmp_path: Path) -> None:
    config, trial_data = _validity_fixture(tmp_path)

    output = write_siips1_dose_response(trial_data=trial_data, config=config)

    assert output.name == "siips1_dose_response.svg"
    assert sorted(output.parent.iterdir()) == [output]
    svg = output.read_text(encoding="utf-8").lower()
    assert "siips1 expression (a.u.)" in svg
    assert "#d55e00" in svg
    assert "pain threshold" not in svg
    assert "nps expression" not in svg
```

Add an integration test that writes NPS and SIIPS1 into separate directories and proves their
SVG view boxes are equal while their y-axis tick labels are not copied from one another.

- [ ] **Step 2: Run the SIIPS1 test and verify RED**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -k siips1 \
  -q
```

Expected: import failure because the SIIPS1 module is absent.

- [ ] **Step 3: Implement the one-output SIIPS1 writer and executable main**

```python
OUTPUT_FILENAME = "siips1_dose_response.svg"
SPECIFICATION = DoseResponseSpecification(
    ylabel="SIIPS1 expression (a.u.)",
    color_config_key="siips1",
)


def write_siips1_dose_response(
    *,
    trial_data: ValidityTrialData,
    config: Any,
) -> Path:
    summary = build_dose_response_summary(
        trial_data.targets,
        outcome="SIIPS1",
        config=config,
    )
    figure = build_dose_response_figure(summary, SPECIFICATION, config)
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )
```

Add the plot-specific executable boundary:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Study 1 SIIPS1 dose-response SVG."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--study1-config", type=Path)
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    apply_study1_config_defaults(config, config_path=args.study1_config)
    trial_data = load_validity_trial_data(task=args.task, config=config)
    output_path = write_siips1_dose_response(
        trial_data=trial_data,
        config=config,
    )
    print(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all standalone figure tests**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -q
```

Expected: PASS. Calling each writer in an isolated output root creates one SVG and no raster
or second vector format.

- [ ] **Step 5: Commit the SIIPS1 plot**

```bash
git add studies/pain_study/study1/figures/plot_siips1_dose_response.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: plot Study 1 SIIPS1 validity"
```

### Task 8: Integrate the Three Supplementary SVGs into Reporting

**Files:**
- Modify: `studies/pain_study/study1/figures/__init__.py`
- Modify: `studies/pain_study/study1/reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_extended.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_full_picture.py`
- Modify: `tests/pipelines/test_study1_reporting.py`

- [ ] **Step 1: Write failing report-output and manifest tests**

Build each reporting test config from `load_study1_config()` so the dedicated YAML remains the
single source for dimensions, fonts, colours, and line styling. Override paths, root name, and
the low-cost test-specific fields only. Set `temperatures` to the exact observed levels in
each fixture: `[45.3, 49.3]` in the `studies/tests/pipelines/` reporting fixtures and
`[44.0, 46.0, 48.0, 50.0]` in `tests/pipelines/test_study1_reporting.py`:

```python
"figures": {
    "validity": {
        "temperatures": [45.3, 49.3],
        "bootstrap": {
            "iterations": 20,
            "confidence_level": 0.95,
            "seed": 42,
            "max_invalid_fraction": 0.20,
        },
    }
},
```

Then extend the full-picture test:

```python
figure_root = report_path.parent / "figures" / "supplementary" / "validity"
expected_figures = {
    "behavioral_dose_response": figure_root / "behavioral_dose_response.svg",
    "nps_dose_response": figure_root / "nps_dose_response.svg",
    "siips1_dose_response": figure_root / "siips1_dose_response.svg",
}
assert sorted(figure_root.iterdir()) == sorted(expected_figures.values())

manifest = json.loads(
    (report_path.parent / "full_picture" / "full_picture_manifest.json").read_text()
)
assert manifest["supplementary_figures"] == {
    name: str(path) for name, path in expected_figures.items()
}
```

Add a regression assertion that the article and full-picture TSV values are unchanged after
figure creation. Add a failure test that monkeypatches one writer to raise and asserts the
exception reaches the caller and the full-picture manifest is not rewritten.

- [ ] **Step 2: Run reporting tests and verify RED**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py \
  -q
```

Expected: failures because the report writes no validity SVGs or manifest figure entries.

- [ ] **Step 3: Export and invoke all three explicit writers**

Export named writers from `figures/__init__.py`. In `write_study1_report`, use the already
loaded `ValidityTrialData` and invoke each writer directly:

```python
supplementary_figures = {
    "behavioral_dose_response": write_behavioral_dose_response(
        trial_data=trial_data,
        config=config,
    ),
    "nps_dose_response": write_nps_dose_response(
        trial_data=trial_data,
        config=config,
    ),
    "siips1_dose_response": write_siips1_dose_response(
        trial_data=trial_data,
        config=config,
    ),
}
```

Pass this mapping to `_write_full_picture_tables` and serialize it under the exact
`supplementary_figures` manifest key. Do not catch writer exceptions. Do not add a loop that
discovers figure functions dynamically; the three approved outputs remain explicit and
searchable.

- [ ] **Step 4: Run reporting and figure integration tests**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py \
  -q
```

Expected: PASS with exactly three report SVGs and unchanged table values.

- [ ] **Step 5: Commit report integration**

```bash
git add studies/pain_study/study1/figures/__init__.py \
  studies/pain_study/study1/reporting.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py
git commit -m "feat: report Study 1 validity figures"
```

### Task 9: Document and Verify the Supplementary Figure Contract

**Files:**
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`

- [ ] **Step 1: Document scientific meaning and output classification**

In README section 6.1, state that the report writes three descriptive supplementary plots
using retained target trials, equal participant weighting, and 10,000 participant-cluster
bootstrap resamples. Explicitly state that the plots are not extra hypothesis tests and do
not replace the target QC table.

In the run guide report-output section, list exactly:

```text
$STUDY1_ROOT/reports/figures/supplementary/validity/behavioral_dose_response.svg
$STUDY1_ROOT/reports/figures/supplementary/validity/nps_dose_response.svg
$STUDY1_ROOT/reports/figures/supplementary/validity/siips1_dose_response.svg
```

Document the direct executable form for each module using `--config`, `--study1-config`, and
`--task`.

- [ ] **Step 2: Run focused tests, lint, and repository checks**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/config/test_study1_config_loader.py \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py \
  -q
.venv/bin/ruff check \
  studies/pain_study/study1/config \
  studies/pain_study/study1/figures \
  studies/pain_study/study1/reporting.py \
  studies/tests/config/test_study1_config_loader.py \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py
make verify-structure
make verify-architecture
```

Expected: all tests and checks pass with no Ruff findings.

- [ ] **Step 3: Generate deterministic QA figures at publication size**

Run the twelve-participant, six-temperature SVG integration fixture into a retained QA base:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_figures.py::test_qa_fixture_writes_three_publication_svgs \
  --basetemp=/tmp/study1-validity-qa \
  -q
rg --files /tmp/study1-validity-qa | rg '\.svg$' | sort
```

Expected: exactly three SVG paths named `behavioral_dose_response.svg`,
`nps_dose_response.svg`, and `siips1_dose_response.svg`.

- [ ] **Step 4: Render and visually inspect all three SVGs**

Render temporary previews:

```bash
mkdir -p /tmp/study1-validity-previews
qlmanage -t -s 1200 -o /tmp/study1-validity-previews \
  /tmp/study1-validity-qa/**/behavioral_dose_response.svg \
  /tmp/study1-validity-qa/**/nps_dose_response.svg \
  /tmp/study1-validity-qa/**/siips1_dose_response.svg
```

Inspect each PNG with the image-viewing tool at original detail. At 89 mm equivalent width,
confirm: readable 5–7 pt type, no clipping or overlap, visible participant variation,
confidence intervals distinguishable from participant traces, correct threshold annotation,
no grid or internal title, and visibly independent NPS/SIIPS1 y-scales. Any visual defect
requires a failing SVG/style test before correction.

- [ ] **Step 5: Run the full Study 1 pipeline test directory**

Run:

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_*.py \
  tests/pipelines/test_study1_reporting.py \
  tests/pipelines/test_study1_feature_benchmark_config.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit documentation and verification-driven adjustments**

```bash
git add studies/pain_study/study1/README.md \
  studies/pain_study/study1/RUN_GUIDE.md \
  studies/pain_study/study1/config \
  studies/pain_study/study1/figures \
  studies/pain_study/study1/reporting.py \
  studies/tests/config/test_study1_config_loader.py \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  tests/pipelines/test_study1_reporting.py
git commit -m "docs: document Study 1 validity figures"
```
