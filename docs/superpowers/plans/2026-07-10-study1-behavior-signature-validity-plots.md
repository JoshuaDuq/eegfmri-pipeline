# Study 1 Behavior–Signature Validity Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separate Nature/Science-quality NPS and SIIPS1 supplementary coefficient plots that quantify pain-report and within-scale intensity associations beyond categorical temperature.

**Architecture:** Extend the existing Study 1 validity-data boundary with strict protocol scoring, then place participant OLS estimation and paired participant bootstrap logic in a new scientific module. A separate renderer consumes only validated summaries, while two thin target-specific scripts each write exactly one deterministic SVG. The report stage registers both plots and writes auditable participant/cohort tables.

**Tech Stack:** Python 3.11+, NumPy, pandas, Matplotlib, PyYAML configuration, pytest, Ruff.

---

## File Map

**Create:**

- `studies/pain_study/study1/figures/behavioral_validity.py` — participant model contracts,
  OLS coefficients, non-estimability records, and paired participant bootstrap.
- `studies/pain_study/study1/figures/coefficient_plot.py` — target-agnostic horizontal
  coefficient renderer.
- `studies/pain_study/study1/figures/plot_nps_behavioral_validity.py` — NPS specification,
  CLI, and one-SVG writer.
- `studies/pain_study/study1/figures/plot_siips1_behavioral_validity.py` — SIIPS1
  specification, CLI, and one-SVG writer.
- `studies/tests/pipelines/test_study1_behavioral_validity.py` — estimator and bootstrap tests.
- `studies/tests/pipelines/test_study1_behavioral_validity_figures.py` — rendering and CLI tests.

**Modify:**

- `studies/pain_study/study1/figures/validity_data.py` — validate pain/rating coding and derive
  `within_scale_intensity` at the join boundary.
- `studies/pain_study/study1/figures/__init__.py` — export both new writers.
- `studies/pain_study/study1/reporting.py` — write both plots, add coefficient tables, and replace
  discontinuous-rating construct diagnostics with within-scale intensity diagnostics.
- `studies/tests/pipelines/test_study1_validity_data.py` — protocol-score tests.
- `studies/tests/pipelines/test_study1_validity_figures.py` — update shared trial fixtures with
  valid behavioral columns.
- `studies/tests/pipelines/test_study1_reporting.py` — update diagnostic schema assertions.
- `studies/tests/pipelines/test_study1_reporting_extended.py` — update diagnostic schema
  assertions.
- `studies/tests/pipelines/test_study1_reporting_full_picture.py` — assert new tables, SVGs, and
  manifest entries.
- `studies/pain_study/study1/README.md` — document estimands and supplementary outputs.
- `studies/pain_study/study1/RUN_GUIDE.md` — list the two new required report SVGs.

Do not modify the unrelated in-progress changes currently present in
`studies/pain_study/study1/targets.py` or `tests/pipelines/test_study1_targets.py`.

### Task 1: Enforce the behavioral protocol score at the validity-data boundary

**Files:**

- Modify: `studies/tests/pipelines/test_study1_validity_data.py`
- Modify: `studies/pain_study/study1/figures/validity_data.py`

- [ ] **Step 1: Write failing tests for valid scoring and inconsistent coding**

Add tests that exercise the public helper and the joined loader result:

```python
def test_add_within_scale_intensity_uses_protocol_scales() -> None:
    from studies.pain_study.study1.figures.validity_data import (
        add_within_scale_intensity,
    )

    trials = pd.DataFrame(
        {
            "pain_binary_coded": [0, 0, 1, 1],
            "vas_final_coded_rating": [0.0, 99.0, 100.0, 200.0],
        }
    )

    scored = add_within_scale_intensity(trials)

    assert scored["within_scale_intensity"].tolist() == [0.0, 99.0, 0.0, 100.0]
    assert "within_scale_intensity" not in trials.columns


@pytest.mark.parametrize(
    ("pain_report", "rating"),
    [(0, 100.0), (1, 99.0), (2, 150.0), (0.5, 50.0)],
)
def test_add_within_scale_intensity_rejects_inconsistent_protocol_codes(
    pain_report: float,
    rating: float,
) -> None:
    from studies.pain_study.study1.figures.validity_data import (
        add_within_scale_intensity,
    )

    trials = pd.DataFrame(
        {
            "pain_binary_coded": [pain_report],
            "vas_final_coded_rating": [rating],
        }
    )

    with pytest.raises(ValueError, match="pain/rating protocol coding"):
        add_within_scale_intensity(trials)
```

Extend the existing loader test with:

```python
assert trial_data.enriched_targets["within_scale_intensity"].tolist() == expected_scores
```

- [ ] **Step 2: Run the tests and verify the new API is absent**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  -q
```

Expected: FAIL because `add_within_scale_intensity` is not defined and the enriched table lacks
`within_scale_intensity`.

- [ ] **Step 3: Implement strict scoring and call it after the target/event join**

Add the public constant and helper near the behavioral validation functions:

```python
WITHIN_SCALE_INTENSITY_COLUMN = "within_scale_intensity"


def add_within_scale_intensity(trials: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        trials,
        ("pain_binary_coded", "vas_final_coded_rating"),
        table_name="behavioral validity trials",
    )
    pain_report = _finite_numeric_series(
        trials,
        "pain_binary_coded",
        table_name="behavioral validity trials",
    )
    if not np.allclose(pain_report, np.round(pain_report)):
        raise ValueError("Behavioral pain/rating protocol coding requires binary pain values.")
    pain_report = pain_report.round().astype(int)
    if not pain_report.isin((0, 1)).all():
        raise ValueError("Behavioral pain/rating protocol coding requires values 0 or 1.")

    ratings = _finite_numeric_series(
        trials,
        "vas_final_coded_rating",
        table_name="behavioral validity trials",
    )
    nonpain_invalid = (pain_report == 0) & ~ratings.between(0.0, 99.0, inclusive="both")
    pain_invalid = (pain_report == 1) & ~ratings.between(100.0, 200.0, inclusive="both")
    if (nonpain_invalid | pain_invalid).any():
        raise ValueError("Behavioral pain/rating protocol coding is inconsistent.")

    scored = trials.copy()
    scored[WITHIN_SCALE_INTENSITY_COLUMN] = np.where(
        pain_report.to_numpy(dtype=int) == 1,
        ratings.to_numpy(dtype=float) - 100.0,
        ratings.to_numpy(dtype=float),
    )
    return scored
```

In `load_validity_trial_data`, replace the direct merged assignment with:

```python
enriched_targets = add_within_scale_intensity(
    _merge_targets_with_clean_events(targets, clean_events)
)
```

Export the helper and constant through `__all__` if the module defines one.

- [ ] **Step 4: Run focused validity-data tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit the protocol-scoring boundary**

```bash
git add \
  studies/pain_study/study1/figures/validity_data.py \
  studies/tests/pipelines/test_study1_validity_data.py
git commit -m "feat: validate Study 1 within-scale intensity"
```

### Task 2: Estimate participant behavioral-validity coefficients

**Files:**

- Create: `studies/tests/pipelines/test_study1_behavioral_validity.py`
- Create: `studies/pain_study/study1/figures/behavioral_validity.py`

- [ ] **Step 1: Write failing tests for target specifications and known coefficients**

Create synthetic retained trials with three participants, all configured temperatures, repeated
pain/non-pain reports, and varying within-scale intensity. Test the public contracts:

```python
from studies.tests.test_support import DotConfig, validity_figure_test_config


def _config(*, min_subjects: int = 3) -> DotConfig:
    return DotConfig(
        {
            "study1": {
                "cohort": {"min_subjects": min_subjects},
                "figures": validity_figure_test_config((44.3, 45.3, 46.3)),
            }
        }
    )


def _synthetic_trials() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    pain_pattern = np.array([0.0, 0.0, 1.0, 1.0])
    intensity_pattern = np.array([20.0, 40.0, 40.0, 20.0])
    nuisance_pattern = np.array([1.0, -1.0, 1.0, -1.0])
    for subject_index, subject_id in enumerate(("sub-01", "sub-02", "sub-03")):
        subject_rows: list[dict[str, float | str]] = []
        for temperature_index, temperature in enumerate((44.3, 45.3, 46.3)):
            for pain_report, intensity, nuisance in zip(
                pain_pattern,
                intensity_pattern,
                nuisance_pattern,
                strict=True,
            ):
                subject_rows.append(
                    {
                        "subject_id": subject_id,
                        "stimulus_temp": temperature,
                        "pain_binary_coded": pain_report,
                        "within_scale_intensity": intensity,
                        "_nuisance": nuisance + 0.1 * temperature_index,
                    }
                )
        subject = pd.DataFrame(subject_rows)
        pain_z = (subject["pain_binary_coded"] - subject["pain_binary_coded"].mean()) / subject[
            "pain_binary_coded"
        ].std(ddof=1)
        intensity_z = (
            subject["within_scale_intensity"] - subject["within_scale_intensity"].mean()
        ) / subject["within_scale_intensity"].std(ddof=1)
        nuisance_z = (subject["_nuisance"] - subject["_nuisance"].mean()) / subject[
            "_nuisance"
        ].std(ddof=1)
        subject["NPS"] = 0.55 * pain_z + 0.25 * intensity_z + 0.40 * nuisance_z
        subject["SIIPS1"] = (
            0.30 * pain_z + 0.45 * intensity_z + 0.35 * subject["NPS"] + nuisance_z
        )
        subject["vas_final_coded_rating"] = np.where(
            subject["pain_binary_coded"] == 1.0,
            subject["within_scale_intensity"] + 100.0,
            subject["within_scale_intensity"],
        )
        rows.extend(subject.drop(columns="_nuisance").to_dict("records"))
    return pd.DataFrame(rows)


from studies.pain_study.study1.figures.behavioral_validity import (
    NPS_SPECIFICATION,
    SIIPS1_SPECIFICATION,
    build_behavioral_validity_summary,
)


def test_target_specifications_preserve_distinct_adjustment_models() -> None:
    assert NPS_SPECIFICATION.target == "NPS"
    assert NPS_SPECIFICATION.adjustment_columns == ()
    assert SIIPS1_SPECIFICATION.target == "SIIPS1"
    assert SIIPS1_SPECIFICATION.adjustment_columns == ("NPS",)


def test_nps_summary_recovers_participant_partial_coefficients() -> None:
    trials = _synthetic_trials()
    summary = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(),
    )

    estimable = summary.participant_models.query("estimable")
    assert estimable["subject_id"].tolist() == ["sub-01", "sub-02", "sub-03"]
    assert np.isfinite(
        estimable[["painful_report_beta", "within_scale_intensity_beta"]]
    ).all().all()
    assert summary.cohort_estimates["term"].tolist() == [
        "painful_report",
        "within_scale_intensity",
    ]
    assert summary.cohort_estimates["n_subjects"].tolist() == [3, 3]
```

In `_synthetic_trials`, generate each target from an explicit design matrix, then independently
solve that same matrix in the test with `np.linalg.lstsq`; compare the public participant
coefficients to the independently calculated values using `pytest.approx`.

- [ ] **Step 2: Write failing tests for non-estimability and the minimum cohort**

```python
def test_constant_pain_report_is_recorded_as_non_estimable() -> None:
    trials = _synthetic_trials()
    subject_mask = trials["subject_id"] == "sub-03"
    trials.loc[subject_mask, "pain_binary_coded"] = 1.0
    trials.loc[subject_mask, "vas_final_coded_rating"] = (
        trials.loc[subject_mask, "within_scale_intensity"] + 100.0
    )

    summary = build_behavioral_validity_summary(
        trials,
        specification=NPS_SPECIFICATION,
        config=_config(min_subjects=2),
    )

    excluded = summary.participant_models.query("not estimable").iloc[0]
    assert excluded["subject_id"] == "sub-03"
    assert excluded["non_estimability_reason"] == "constant_pain_binary_coded"


def test_summary_rejects_too_few_estimable_participants() -> None:
    with pytest.raises(ValueError, match="requires at least 3 estimable participants"):
        build_behavioral_validity_summary(
            _synthetic_trials().query("subject_id != 'sub-03'").copy(),
            specification=NPS_SPECIFICATION,
            config=_config(min_subjects=3),
        )
```

Also cover missing temperature levels, constant target, rank-deficient design, and non-positive
residual degrees of freedom with exact reason strings.

- [ ] **Step 3: Run the new test module and verify import failure**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_behavioral_validity.py \
  -q
```

Expected: FAIL because `behavioral_validity.py` does not exist.

- [ ] **Step 4: Implement the estimator and immutable summary contracts**

Create the module with these public types and entry point:

```python
@dataclass(frozen=True)
class BehavioralValiditySpecification:
    target: str
    color_config_key: str
    adjustment_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehavioralValiditySummary:
    target: str
    participant_models: pd.DataFrame
    cohort_estimates: pd.DataFrame


NPS_SPECIFICATION = BehavioralValiditySpecification(
    target="NPS",
    color_config_key="nps",
)
SIIPS1_SPECIFICATION = BehavioralValiditySpecification(
    target="SIIPS1",
    color_config_key="siips1",
    adjustment_columns=("NPS",),
)


def build_behavioral_validity_summary(
    trials: pd.DataFrame,
    *,
    specification: BehavioralValiditySpecification,
    config: Any,
) -> BehavioralValiditySummary:
    participant_models = _fit_participant_models(trials, specification, config)
    estimable = participant_models.loc[participant_models["estimable"]].copy()
    minimum = int(require_config_value(config, "study1.cohort.min_subjects"))
    if len(estimable) < minimum:
        raise ValueError(
            f"{specification.target} behavioral validity requires at least {minimum} "
            f"estimable participants; observed {len(estimable)}."
        )
    cohort = _participant_bootstrap(estimable, config)
    return BehavioralValiditySummary(
        target=specification.target,
        participant_models=participant_models,
        cohort_estimates=cohort,
    )
```

Build each participant design in this exact column order:

```python
design_columns = [
    np.ones(len(rows), dtype=float),
    *temperature_dummy_columns,
    *standardized_adjustment_columns,
    standardize(rows["pain_binary_coded"]),
    standardize(rows["within_scale_intensity"]),
]
```

Use sample standard deviations (`ddof=1`). Sort participants and temperatures before fitting.
Check the singular-value ratio against `1e-10`, then solve with `np.linalg.lstsq`. Store one row
per retained participant with columns:

```text
subject_id, target, estimable, non_estimability_reason,
painful_report_beta, within_scale_intensity_beta, n_trials, residual_degrees_of_freedom
```

The paired bootstrap samples complete coefficient rows and returns:

```text
target, term, mean, ci_low, ci_high, n_subjects
```

Use `study1.figures.validity.bootstrap.{iterations,confidence_level,seed,max_invalid_fraction}` and
the same attempt-budget calculation as the existing dose-response bootstrap.

- [ ] **Step 5: Run estimator tests and make only targeted corrections**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit the scientific estimator**

```bash
git add \
  studies/pain_study/study1/figures/behavioral_validity.py \
  studies/tests/pipelines/test_study1_behavioral_validity.py
git commit -m "feat: estimate Study 1 behavioral validity"
```

### Task 3: Render two standalone publication SVGs

**Files:**

- Create: `studies/tests/pipelines/test_study1_behavioral_validity_figures.py`
- Create: `studies/pain_study/study1/figures/coefficient_plot.py`
- Create: `studies/pain_study/study1/figures/plot_nps_behavioral_validity.py`
- Create: `studies/pain_study/study1/figures/plot_siips1_behavioral_validity.py`
- Modify: `studies/pain_study/study1/figures/__init__.py`
- Modify: `studies/tests/pipelines/test_study1_validity_figures.py`

- [ ] **Step 1: Write failing renderer tests**

Construct a `BehavioralValiditySummary` directly and assert the scientific layers:

```python
def test_coefficient_figure_draws_scientific_layers(tmp_path: Path) -> None:
    figure = build_behavioral_validity_figure(
        _summary("NPS"),
        color_config_key="nps",
        config=_config(tmp_path),
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Standardized partial coefficient (β)"
    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        "Within-scale intensity",
        "Painful report",
    ]
    assert axis.get_title() == ""
    assert axis.get_xlim()[0] == pytest.approx(-axis.get_xlim()[1])
    assert any(line.get_linestyle() == "--" for line in axis.lines)
    assert len(axis.collections) >= 4
    assert "n = 3" in " ".join(text.get_text() for text in axis.texts)
    plt.close(figure)
```

Parse a saved SVG and assert `89 × 70 mm`, editable `<text>`, the correct target colour, the beta
axis label, and zero-reference content.

- [ ] **Step 2: Write failing one-script-per-plot tests**

Parameterize both writers and CLI modules:

```python
@pytest.mark.parametrize(
    ("writer_name", "filename", "color"),
    [
        ("write_nps_behavioral_validity", "nps_behavioral_validity.svg", "#0072b2"),
        ("write_siips1_behavioral_validity", "siips1_behavioral_validity.svg", "#d55e00"),
    ],
)
def test_behavioral_validity_writer_creates_only_assigned_svg(
    tmp_path: Path,
    writer_name: str,
    filename: str,
    color: str,
) -> None:
    writer = getattr(figures, writer_name)
    output_path = writer(trial_data=_trial_data(), config=_config(tmp_path))

    assert output_path.name == filename
    assert sorted(output_path.parent.iterdir()) == [output_path]
    assert color in output_path.read_text(encoding="utf-8").lower()
```

Update `_trial_data()` in `test_study1_validity_figures.py` to include valid
`pain_binary_coded` and `within_scale_intensity` columns.

- [ ] **Step 3: Run the new figure tests and verify import failures**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  -q
```

Expected: FAIL because the renderer and target scripts do not exist.

- [ ] **Step 4: Implement the target-agnostic renderer**

Create:

```python
TERM_ROWS = (
    ("within_scale_intensity", "Within-scale intensity", 0.0),
    ("painful_report", "Painful report", 1.0),
)


def build_behavioral_validity_figure(
    summary: BehavioralValiditySummary,
    *,
    color_config_key: str,
    config: Any,
) -> Figure:
    colors = require_config_value(config, "study1.figures.validity.colors")
    if color_config_key not in colors:
        raise ValueError(f"Unknown Study 1 validity color key: {color_config_key!r}.")
    with publication_style(config):
        figure, axis = plt.subplots(
            figsize=configured_figure_size(config),
            layout="constrained",
        )
        _draw_zero_reference(axis)
        _draw_participant_coefficients(axis, summary, config)
        _draw_cohort_estimates(axis, summary, str(colors[color_config_key]), config)
        _format_coefficient_axis(axis, summary)
    return figure
```

Sort participant points by `subject_id` and assign deterministic offsets with
`np.linspace(-0.10, 0.10, n_subjects)`. Draw participant circles with the configured gray,
opacity, and size. Draw cohort diamonds and horizontal percentile-CI whiskers in the target
colour. Compute the symmetric x-limit from every finite participant coefficient and CI bound,
using `1.08 * max_abs` and a minimum half-range of `0.10`. Annotate `n = {n_subjects}` at the
right edge of each row in axes-x/data-y coordinates.

- [ ] **Step 5: Implement the two thin target scripts and exports**

Each module follows the existing dose-response CLI pattern. The NPS writer core is:

```python
OUTPUT_FILENAME = "nps_behavioral_validity.svg"


def write_nps_behavioral_validity(*, trial_data: ValidityTrialData, config: Any) -> Path:
    summary = build_behavioral_validity_summary(
        trial_data.enriched_targets,
        specification=NPS_SPECIFICATION,
        config=config,
    )
    figure = build_behavioral_validity_figure(
        summary,
        color_config_key=NPS_SPECIFICATION.color_config_key,
        config=config,
    )
    return save_validity_svg(
        figure,
        validity_output_dir(config) / OUTPUT_FILENAME,
        config,
    )
```

The SIIPS1 module is identical in structure but uses `SIIPS1_SPECIFICATION`,
`write_siips1_behavioral_validity`, and `siips1_behavioral_validity.svg`. Both `main()` functions
require `--config` and `--task`, accept optional `--study1-config`, load the canonical trial data,
write one SVG, print its path, and return it.

Export both writers from `figures/__init__.py`.

- [ ] **Step 6: Run all figure tests**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 7: Commit both standalone plots**

```bash
git add \
  studies/pain_study/study1/figures/__init__.py \
  studies/pain_study/study1/figures/coefficient_plot.py \
  studies/pain_study/study1/figures/plot_nps_behavioral_validity.py \
  studies/pain_study/study1/figures/plot_siips1_behavioral_validity.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_validity_figures.py
git commit -m "feat: render Study 1 behavioral validity SVGs"
```

### Task 4: Integrate plots and auditable tables into Study 1 reporting

**Files:**

- Modify: `studies/pain_study/study1/reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_extended.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_full_picture.py`

- [ ] **Step 1: Write failing report-integration assertions**

Extend the full-picture report test:

```python
assert (full_picture_root / "behavior_signature_validity_by_subject.tsv").exists()
assert (full_picture_root / "behavior_signature_validity_summary.tsv").exists()

expected_figures.update(
    {
        "nps_behavioral_validity": figure_root / "nps_behavioral_validity.svg",
        "siips1_behavioral_validity": figure_root / "siips1_behavioral_validity.svg",
    }
)
assert manifest["supplementary_figures"] == {
    name: str(path) for name, path in expected_figures.items()
}
```

Read the new tables and assert targets, term order, finite coefficients, common participant counts,
and explicit `estimable`/`non_estimability_reason` columns.

Replace old diagnostic assertions with:

```python
assert "within_scale_intensity_r" in target_qc.columns
assert "vas_rating_r" not in target_qc.columns
siips1 = target_qc.loc[target_qc["target"] == "SIIPS1"].iloc[0]
assert siips1["siips1_intensity_beyond_temperature_nps_r"] > 0.0
```

- [ ] **Step 2: Run reporting tests and verify missing outputs**

Run:

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  -q
```

Expected: FAIL because the report does not write the new SVGs/tables and still exposes the raw
displayed-rating diagnostic names.

- [ ] **Step 3: Add reusable report-table builders**

Import both specifications, `build_behavioral_validity_summary`, and the two writers. Add:

```python
def _behavioral_validity_summaries(
    trial_data: ValidityTrialData,
    config: Any,
) -> dict[str, BehavioralValiditySummary]:
    return {
        specification.target: build_behavioral_validity_summary(
            trial_data.enriched_targets,
            specification=specification,
            config=config,
        )
        for specification in (NPS_SPECIFICATION, SIIPS1_SPECIFICATION)
    }


def _behavioral_validity_tables(
    summaries: dict[str, BehavioralValiditySummary],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = pd.concat(
        [summary.participant_models for summary in summaries.values()],
        ignore_index=True,
    )
    cohort = pd.concat(
        [summary.cohort_estimates for summary in summaries.values()],
        ignore_index=True,
    )
    return participants, cohort
```

Build summaries once in `write_study1_report`. Pass them into `_write_full_picture_tables`, and
write both TSV/parquet stems through `_write_article_table`.

- [ ] **Step 4: Register both SVG writers**

Extend `supplementary_figures` in `write_study1_report`:

```python
"nps_behavioral_validity": write_nps_behavioral_validity(
    trial_data=trial_data,
    config=config,
),
"siips1_behavioral_validity": write_siips1_behavioral_validity(
    trial_data=trial_data,
    config=config,
),
```

Keep each writer responsible for only one SVG.

- [ ] **Step 5: Correct the construct-diagnostic schema**

In `_article_target_diagnostics`, replace the target association with raw
`vas_final_coded_rating` by `within_scale_intensity`. Rename:

```text
vas_rating_r -> within_scale_intensity_r
siips1_rating_beyond_temperature_nps_r
  -> siips1_intensity_beyond_temperature_nps_r
```

Apply the same explicit names in `_target_qc_metrics`, required-column validation, and affected
tests. Do not retain old aliases.

- [ ] **Step 6: Run reporting tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 7: Commit reporting integration**

```bash
git add \
  studies/pain_study/study1/reporting.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py
git commit -m "feat: publish Study 1 behavioral validity figures"
```

### Task 5: Document the article validity figure set

**Files:**

- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`

- [ ] **Step 1: Update the scientific methods text**

In README section 6.1, list all five validity SVGs and add the two participant models exactly as
specified in the design document. State that continuous behavioral construct diagnostics use the
protocol-defined 0–100 within-scale score, not the discontinuous 0–200 display code. State that
participant coefficients are equally weighted and uncertainty uses 10,000 participant-bootstrap
resamples.

- [ ] **Step 2: Update required report outputs**

Add these paths to the run guide output checklist:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/nps_behavioral_validity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/siips1_behavioral_validity.svg
```

Also list the two new full-picture TSVs.

- [ ] **Step 3: Run documentation and structure checks**

Run:

```bash
make verify-structure
```

Expected: PASS.

- [ ] **Step 4: Commit documentation**

```bash
git add \
  studies/pain_study/study1/README.md \
  studies/pain_study/study1/RUN_GUIDE.md
git commit -m "docs: describe Study 1 behavioral validity plots"
```

### Task 6: Verify scientific computation and publication rendering

**Files:**

- Modify only files implicated by verification failures; do not broaden scope.

- [ ] **Step 1: Run focused Study 1 validity and reporting tests**

```bash
.venv/bin/python -m pytest \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run lint on every touched Python file**

```bash
.venv/bin/python -m ruff check \
  studies/pain_study/study1/figures \
  studies/pain_study/study1/reporting.py \
  studies/tests/pipelines/test_study1_validity_data.py \
  studies/tests/pipelines/test_study1_validity_figures.py \
  studies/tests/pipelines/test_study1_behavioral_validity.py \
  studies/tests/pipelines/test_study1_behavioral_validity_figures.py \
  studies/tests/pipelines/test_study1_reporting.py \
  studies/tests/pipelines/test_study1_reporting_extended.py \
  studies/tests/pipelines/test_study1_reporting_full_picture.py
```

Expected: PASS with no diagnostics.

- [ ] **Step 3: Run repository architecture and maintainability gates**

```bash
make verify-architecture
make verify-maintainability
```

Expected: both PASS.

- [ ] **Step 4: Render deterministic synthetic SVGs for visual QA**

Use the tested synthetic retained-trial fixture to write both SVGs into
`/tmp/study1_behavioral_validity_qa`, then render them with macOS `sips`:

```bash
/usr/bin/sips -s format png \
  /tmp/study1_behavioral_validity_qa/nps_behavioral_validity.svg \
  --out /tmp/study1_behavioral_validity_qa/nps_behavioral_validity.png
/usr/bin/sips -s format png \
  /tmp/study1_behavioral_validity_qa/siips1_behavioral_validity.svg \
  --out /tmp/study1_behavioral_validity_qa/siips1_behavioral_validity.png
```

Inspect each raster preview
for clipping, label collisions, symmetric limits, visible zero line, participant/cohort hierarchy,
CI visibility, grayscale distinction, and colour accessibility. Do not add raster outputs to the
report or repository.

- [ ] **Step 5: Verify byte reproducibility and output isolation**

Write each SVG twice from identical input and assert byte equality. Assert that the destination
contains exactly:

```text
nps_behavioral_validity.svg
siips1_behavioral_validity.svg
```

- [ ] **Step 6: Run the complete Study 1 pipeline test directory**

```bash
.venv/bin/python -m pytest studies/tests/pipelines -q
```

Expected: PASS.

- [ ] **Step 7: Inspect the final diff and commit verification fixes**

```bash
git diff --check
git status --short
```

Confirm that unrelated pre-existing changes remain untouched. If verification required scoped
fixes, stage only those files and commit:

```bash
git commit -m "test: verify Study 1 behavioral validity plots"
```
