# fMRI subject report: run-level inference, MNI referability, print output

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the per-subject fMRI report a familywise correction that respects the actual between-run variance, publishable MNI coordinates, and submittable figures.

**Architecture:** Two new statistics (a run-level sign-flip null and leave-one-run-out influence) are computed in `analysis/run_level.py`, which holds the fitted model, and written to derivatives; the report package reads them and stays fit-free. Both go through nilearn's own `compute_contrast` path, so neither introduces a second pooling rule. The report's contrast table is re-sourced from the MNI maps that already exist in derivatives, while the drawn maps stay native. A `print` render profile re-renders figures at journal sizes without touching the screen report.

**Tech Stack:** Python 3.11+, nilearn 0.14.0, nibabel 5.4.2, numpy 1.26.4, matplotlib, pandas, pytest.

Spec: `docs/superpowers/specs/2026-08-01-fmri-subject-report-run-inference-and-print-design.md`

## Global Constraints

- **Interpreter is `.venv/bin/python`.** `python3` on this machine has no nibabel or nilearn.
- **Never run the full pytest suite** — it takes about 9 minutes. Every step below names a targeted subset. Run only that.
- **Work directly in the main checkout.** Do not create git worktrees.
- **No automated verdicts.** Panels present measurements. Never score a run, subject, or map against an invented threshold, and never emit pass/fail language. Published reference levels may be *drawn* and must be labelled as conventions, not criteria.
- **The report package must never fit a GLM.** `fmri_pipeline/analysis/report/**` may read maps and manifests only. An existing test asserts the fitting modules stay absent from the report import path; it must keep passing.
- **The pooling rule is nilearn's, verified as equal-weight:** `z ∝ Σᵢ eᵢ / sqrt(Σᵢ vᵢ)`. Do not substitute inverse-variance weighting anywhere.
- **The sign-flip p floor is `2 / (2**(n_runs - 1) + 1)`** — 0.061 for six runs. Never print a global p without it.
- **Two similarly named types, deliberately in different layers.** `run_level.SignFlipNull` (Task 2) is the analysis-side result and carries the full enumerated null. `inference.SignFlipSummary` (Task 5) is the report-side view and carries only the six scalars a panel needs. The report never imports the former — that would pull the analysis package into the report import path and break the no-fitting invariant.
- Figures are byte-reproducible: keep `savefig_kwargs` and the `svg.hashsalt` rcParam intact.

---

## File Structure

**Created:**
- `fmri_pipeline/analysis/report/figures/run_inference.py` — the two new contrast-section panels (run influence, sign-flip null).
- `fmri_pipeline/analysis/report/figures/offsets.py` — the global per-run offset panel.
- `fmri_pipeline/analysis/report/print_profile.py` — figure width registry and the print render context.
- `tests/fmri/test_pooling_rule.py` — pins nilearn's combination rule.
- `tests/fmri/test_sign_flip_null.py`, `tests/fmri/test_run_influence.py`
- `tests/fmri/report/test_run_inference.py`, `tests/fmri/report/test_offsets.py`, `tests/fmri/report/test_print_profile.py`

**Modified:**
- `fmri_pipeline/analysis/run_level.py` — add both statistics and their writers; correct the docstring.
- `fmri_pipeline/pipelines/fmri_analysis.py:333-386,597` — call and record them; correct the docstring.
- `fmri_pipeline/analysis/report/manifest.py` — new paired fields and scalars.
- `fmri_pipeline/analysis/report/inference.py` — sign-flip row in the threshold table.
- `fmri_pipeline/analysis/report/subject.py` — panel wiring, "At a glance", captions, MNI table.
- `fmri_pipeline/analysis/report/figures/timeseries.py:165`, `run_consistency.py:151` — peak ordering by `|z|`.
- `fmri_pipeline/analysis/report/figures/design.py` — split the contrast regressors out of the VIF panel.
- `fmri_pipeline/analysis/report/figures/_display.py:178-205` — cumulative-area cut placement.
- `fmri_pipeline/analysis/report/figures/_mosaic.py` — colorbar gap.
- `fmri_pipeline/analysis/report/figures/carpet.py`, `motion.py`, `tissue.py`
- `fmri_pipeline/analysis/plotting_config.py:7` — allow `pdf`.
- `fmri_pipeline/analysis/report/style.py` — print profile constants.
- `tests/fmri/test_run_level_contrasts.py` — docstring correction.

---

# PHASE 1A — New statistics, computed where the model lives

## Task 1: Pin the pooling rule and correct the claims built on it

The spec's defect 3. Three docstrings assert the combination is "driven by whichever run carries the most precision". nilearn implements equal weighting. Every later task depends on this being settled, so it goes first.

**Files:**
- Create: `tests/fmri/test_pooling_rule.py`
- Modify: `fmri_pipeline/analysis/run_level.py:1-20`, `fmri_pipeline/pipelines/fmri_analysis.py:343-354`, `tests/fmri/test_run_level_contrasts.py:1-8`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/fmri/test_pooling_rule.py::test_multirun_contrast_is_equal_weight` — the regression test every later task relies on for the rule's stability.

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/test_pooling_rule.py
"""Pins the rule nilearn uses to combine runs.

The rule is nilearn's, not ours. ``compute_fixed_effect_contrast`` sums ``Contrast``
objects and scales by 1/n; ``Contrast.__add__`` sums effects and variances, and
``__mul__ = __rmul__`` scales variance by the square of the scalar. The statistic is
therefore invariant to the 1/n and reduces to equal weight per run.

A nilearn release switching to precision weighting would silently invalidate the
sign-flip null, the leave-one-run-out deltas, and three corrected docstrings at once.
This test is what makes that a failure rather than a wrong number.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

SHAPE = (6, 6, 6)
N_FRAMES = 48
TR = 2.0
ONSETS = np.arange(4, 44, 10).astype(float)


def _two_run_model(seed: int = 0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for run in range(2):
        # Unequal noise between runs, so equal-weight and precision-weighted
        # combinations give visibly different answers.
        scale = 1.0 if run == 0 else 4.0
        data = 100 + (scale * rng.standard_normal(SHAPE + (N_FRAMES,))).astype(np.float32)
        for onset in ONSETS:
            frame = int(onset / TR)
            data[1:4, 1:4, 1:4, frame : frame + 3] += 6.0
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(
            pd.DataFrame({"onset": ONSETS, "duration": 4.0, "trial_type": "task"})
        )

    model = FirstLevelModel(
        t_r=TR, mask_img=mask, hrf_model="spm", drift_model=None,
        minimize_memory=False, standardize=False, signal_scaling=False,
    )
    model.fit(bolds, events=events)
    return model


@pytest.fixture(scope="module")
def model():
    return _two_run_model()


def test_multirun_contrast_is_equal_weight(model):
    """t = sum(effect) / sqrt(sum(variance)), not the precision-weighted combination."""
    from fmri_pipeline.analysis.run_level import compute_run_level_contrast

    stored = model.compute_contrast("task", output_type="stat")
    stored_t = model.masker_.transform(stored).ravel()

    per_run = compute_run_level_contrast(model, "task", run_labels=("run-01", "run-02"))
    effect = model.masker_.transform(per_run.effect)      # (2, V)
    variance = model.masker_.transform(per_run.variance)

    equal_weight = effect.sum(0) / np.sqrt(variance.sum(0))
    np.testing.assert_allclose(equal_weight, stored_t, rtol=1e-5, atol=1e-5)


def test_precision_weighting_is_not_the_rule(model):
    """The two rules must be distinguishable on this fixture.

    Without this, the test above would also pass under precision weighting whenever
    the runs happen to have equal variance, and would stop being a pin.
    """
    from fmri_pipeline.analysis.run_level import compute_run_level_contrast

    stored_t = model.masker_.transform(
        model.compute_contrast("task", output_type="stat")
    ).ravel()
    per_run = compute_run_level_contrast(model, "task", run_labels=("run-01", "run-02"))
    effect = model.masker_.transform(per_run.effect)
    variance = model.masker_.transform(per_run.variance)

    weights = 1.0 / variance
    precision_weighted = (weights * effect).sum(0) / np.sqrt(weights.sum(0))
    assert not np.allclose(precision_weighted, stored_t, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run the test to see where it stands**

Run: `.venv/bin/python -m pytest tests/fmri/test_pooling_rule.py -v`

Expected: both PASS. They characterise existing library behaviour rather than new code, so this is a characterisation test, not red-then-green. If `test_multirun_contrast_is_equal_weight` fails, stop — the entire spec rests on this rule and the plan needs revisiting before any further task.

- [ ] **Step 3: Correct the three false docstrings**

In `fmri_pipeline/analysis/run_level.py`, replace the second paragraph of the module docstring (currently beginning "A first-level contrast over six runs is a fixed-effects combination, and a fixed-effects combination is driven by whatever run carries the most precision."):

```python
"""Per-run estimates of a contrast, taken from the model that was already fitted.

A first-level contrast over several runs is a fixed-effects combination. nilearn
combines them with equal weight per run -- ``compute_fixed_effect_contrast`` sums
``Contrast`` objects and scales by 1/n, and ``Contrast.__mul__`` scales variance by
the square of the scalar, so the statistic reduces to ``sum(e) / sqrt(sum(v))``. A
noisy run therefore contributes its effect to the numerator at full strength while
inflating the denominator; the combination is *not* dominated by the most precise run.

The consequence either way is that an effect resting entirely on one run and an effect
present in all of them can produce the same map, the same z, and the same cluster
table. Nothing else in the report distinguishes them, and the distinction is usually
the first thing anyone asks about a single-subject result. See
``tests/fmri/test_pooling_rule.py``, which pins the rule.
"""
```

Apply the same correction to the docstring of `_run_level_maps` in
`fmri_pipeline/pipelines/fmri_analysis.py:343-348` and to the module docstring of
`tests/fmri/test_run_level_contrasts.py:1-8`. In both, the sentence "A fixed-effects
combination across runs is driven by whichever run carries the most precision" becomes
"nilearn combines runs with equal weight, so a noisy run contributes its effect at full
strength while inflating the variance".

- [ ] **Step 4: Confirm nothing regressed**

Run: `.venv/bin/python -m pytest tests/fmri/test_pooling_rule.py tests/fmri/test_run_level_contrasts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fmri/test_pooling_rule.py fmri_pipeline/analysis/run_level.py fmri_pipeline/pipelines/fmri_analysis.py tests/fmri/test_run_level_contrasts.py
git commit -m "test(fmri): pin nilearn's equal-weight run combination

Three docstrings claimed the combination is driven by whichever run carries
the most precision. compute_fixed_effect_contrast sums Contrast objects and
Contrast.__mul__ scales variance by the square of the scalar, so the rule is
equal weight. Verified against a stored map at r = 1.000000."
```

---

## Task 2: Run-level sign-flip null

**Files:**
- Modify: `fmri_pipeline/analysis/run_level.py`
- Test: `tests/fmri/test_sign_flip_null.py`

**Interfaces:**
- Consumes: `_contrast_vectors(flm, contrast_def) -> List[np.ndarray]` (existing, `run_level.py:47`), which already expands a contrast into one weight vector per run's own design columns.
- Produces:
  - `SignFlipNull` frozen dataclass with fields `null_max: Tuple[float, ...]`, `observed_max: float`, `fwe_height: float`, `fwe_survivors: int`, `global_p: float`, `p_floor: float`, `n_runs: int`, `n_patterns: int`, `applied_threshold: float`.
  - `compute_sign_flip_null(flm, contrast_def, *, alpha=0.05) -> Optional[SignFlipNull]`
  - `write_sign_flip_null(null, *, out_dir, stem, cfg_hash) -> Optional[Path]`

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/test_sign_flip_null.py
"""The run-level sign-flip null.

Sign-flipping run-level contributions and recombining them through the same pooling
path gives a familywise height that respects the actual between-run variance, which
neither Bonferroni nor an FDR against N(0, 1) does.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from fmri_pipeline.analysis import run_level

SHAPE = (6, 6, 6)
N_FRAMES = 48
TR = 2.0
ONSETS = np.arange(4, 44, 10).astype(float)


def _model(n_runs: int, *, effect: float, seed: int = 0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for _ in range(n_runs):
        data = 100 + rng.standard_normal(SHAPE + (N_FRAMES,)).astype(np.float32)
        if effect:
            for onset in ONSETS:
                frame = int(onset / TR)
                data[1:4, 1:4, 1:4, frame : frame + 3] += effect
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(
            pd.DataFrame({"onset": ONSETS, "duration": 4.0, "trial_type": "task"})
        )
    model = FirstLevelModel(
        t_r=TR, mask_img=mask, hrf_model="spm", drift_model=None,
        minimize_memory=False, standardize=False, signal_scaling=False,
    )
    model.fit(bolds, events=events)
    return model


def test_pattern_count_is_two_to_the_n_minus_one():
    """The +/-global pair is redundant for a two-sided max statistic."""
    model = _model(3, effect=6.0)
    null = run_level.compute_sign_flip_null(model, "task")
    assert null.n_patterns == 2 ** (3 - 1) == 4
    assert len(null.null_max) == 4


def test_identity_pattern_reproduces_the_observed_maximum():
    """The all-positive pattern is the real contrast, so its max is the observed max."""
    model = _model(3, effect=6.0)
    null = run_level.compute_sign_flip_null(model, "task")
    assert null.null_max[0] == pytest.approx(null.observed_max)


def test_p_floor_accounts_for_the_identity_tie():
    """The identity is always in the null set and always ties, so p >= 2/(n+1)."""
    model = _model(3, effect=6.0)
    null = run_level.compute_sign_flip_null(model, "task")
    assert null.p_floor == pytest.approx(2 / (2 ** (3 - 1) + 1))
    assert null.global_p >= null.p_floor


def test_strong_effect_puts_observed_at_the_top_of_the_null():
    model = _model(4, effect=10.0)
    null = run_level.compute_sign_flip_null(model, "task")
    assert null.observed_max >= max(null.null_max)
    assert null.global_p == pytest.approx(null.p_floor)


def test_single_run_model_yields_none():
    """One run has no sign pattern to flip; a degenerate null would be worse than none."""
    model = _model(1, effect=6.0)
    assert run_level.compute_sign_flip_null(model, "task") is None


def test_writer_emits_one_row_per_pattern(tmp_path):
    model = _model(3, effect=6.0)
    null = run_level.compute_sign_flip_null(model, "task")
    path = run_level.write_sign_flip_null(
        null, out_dir=tmp_path, stem="sub-01_task-x_contrast-c", cfg_hash="abc123"
    )
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == ["pattern", "signs", "max_abs_z"]
    assert len(frame) == null.n_patterns
    assert frame.loc[0, "signs"] == "+++"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/test_sign_flip_null.py -v`
Expected: FAIL — `AttributeError: module 'fmri_pipeline.analysis.run_level' has no attribute 'compute_sign_flip_null'`.

- [ ] **Step 3: Implement**

Append to `fmri_pipeline/analysis/run_level.py`:

```python
@dataclass(frozen=True)
class SignFlipNull:
    """A familywise height from sign-flipping run-level contributions.

    Exchangeability is over runs, which is the unit the design actually replicates.
    The null is exact and enumerated, not sampled: with ``n`` runs there are
    ``2**(n-1)`` distinct sign patterns for a two-sided maximum statistic, because a
    pattern and its global negation give identical ``|z|`` maps.
    """

    null_max: Tuple[float, ...]
    observed_max: float
    fwe_height: float
    fwe_survivors: int
    global_p: float
    p_floor: float
    n_runs: int
    n_patterns: int
    applied_threshold: float


def _sign_patterns(n_runs: int) -> np.ndarray:
    """Return the distinct sign patterns, identity first.

    The first run's sign is pinned to +1: a pattern and its global negation produce
    the same ``|z|`` map, so enumerating both would double the null with copies and
    halve the apparent resolution of the p-value for nothing.
    """
    import itertools

    patterns = [
        (1.0,) + rest
        for rest in itertools.product((1.0, -1.0), repeat=n_runs - 1)
    ]
    patterns.sort(key=lambda p: [s < 0 for s in p])  # identity (all +1) first
    return np.asarray(patterns, dtype=float)


def compute_sign_flip_null(
    flm: Any, contrast_def: Any, *, alpha: float = 0.05
) -> Optional[SignFlipNull]:
    """Enumerate the run sign-flip null for ``contrast_def``.

    Each pattern is evaluated by passing the per-run vectors ``s_i * c_i`` through the
    same ``compute_contrast`` call the pipeline already uses, so the pooling rule is
    nilearn's own and the identity pattern reproduces the stored map exactly rather
    than approximating it.

    ``None`` when the model holds a single run, or when the contrast cannot be
    expanded. Best-effort: this is a diagnostic, and the contrast it describes is
    already on disk by the time it runs.
    """
    designs = list(getattr(flm, "design_matrices_", []) or [])
    masker = getattr(flm, "masker_", None)
    if masker is None or len(designs) < 2:
        return None

    try:
        vectors = _contrast_vectors(flm, contrast_def)
    except Exception as exc:
        logger.warning("Could not expand the contrast for the sign-flip null (%s)", exc)
        return None

    n_runs = len(vectors)
    patterns = _sign_patterns(n_runs)

    maxima: List[float] = []
    observed: Optional[np.ndarray] = None
    for index, signs in enumerate(patterns):
        flipped = [s * v for s, v in zip(signs, vectors)]
        try:
            z_img = flm.compute_contrast(flipped, output_type="z_score")
        except Exception as exc:
            logger.warning("Sign-flip pattern %d failed (%s)", index, exc)
            return None
        z = np.asarray(masker.transform(z_img), dtype=float).ravel()
        z = z[np.isfinite(z)]
        if z.size == 0:
            return None
        maxima.append(float(np.abs(z).max()))
        if index == 0:
            observed = z

    if observed is None:
        return None

    null_max = np.asarray(maxima, dtype=float)
    observed_max = float(null_max[0])
    height = float(np.quantile(null_max, 1.0 - alpha))
    # The identity is a member of the null set and always ties the observed maximum,
    # so the +1 in the numerator is not a continuity correction here -- it is that tie.
    global_p = float((np.sum(null_max >= observed_max) + 1) / (null_max.size + 1))
    return SignFlipNull(
        null_max=tuple(float(v) for v in null_max),
        observed_max=observed_max,
        fwe_height=height,
        fwe_survivors=int(np.sum(np.abs(observed) >= height)),
        global_p=global_p,
        p_floor=float(2 / (null_max.size + 1)),
        n_runs=n_runs,
        n_patterns=int(null_max.size),
        applied_threshold=float(alpha),
    )


def write_sign_flip_null(
    null: SignFlipNull, *, out_dir: Path, stem: str, cfg_hash: str
) -> Optional[Path]:
    """Write the enumerated null, one row per sign pattern."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_desc-signflipnull_{cfg_hash}.tsv"

    patterns = _sign_patterns(null.n_runs)
    frame = pd.DataFrame(
        {
            "pattern": np.arange(1, null.n_patterns + 1),
            "signs": ["".join("+" if s > 0 else "-" for s in row) for row in patterns],
            "max_abs_z": null.null_max,
        }
    )
    try:
        frame.to_csv(path, sep="\t", index=False)
    except Exception as exc:
        logger.warning("Could not write %s (%s)", path.name, exc)
        return None
    return path
```

Add `SignFlipNull`, `compute_sign_flip_null`, `write_sign_flip_null` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fmri/test_sign_flip_null.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/run_level.py tests/fmri/test_sign_flip_null.py
git commit -m "feat(fmri): add a run-level sign-flip null

Enumerates all 2^(n-1) distinct sign patterns through nilearn's own
compute_contrast, so the identity pattern reproduces the stored map rather
than approximating it. Records the p floor of 2/(2^(n-1)+1) alongside the
p, since the identity always ties the observed maximum."
```

---

## Task 3: Leave-one-run-out influence

**Files:**
- Modify: `fmri_pipeline/analysis/run_level.py`
- Test: `tests/fmri/test_run_influence.py`

**Interfaces:**
- Consumes: `_contrast_vectors` (as Task 2).
- Produces:
  - `RunInfluence` frozen dataclass: `dropped_run: str`, `survivors: int`, `delta: int`, `max_abs_z: float`, `correlation: float`.
  - `compute_run_influence(flm, contrast_def, *, run_labels=(), threshold=2.3) -> Optional[Tuple[RunInfluence, ...]]`
  - `write_run_influence(rows, *, out_dir, stem, cfg_hash) -> Optional[Path]`

The exactness here rests on `compute_fixed_effect_contrast` (`nilearn/glm/contrasts.py:159`) skipping runs whose vector is all zeros — `if np.all(con_val == 0): continue` — and dividing by the count of *surviving* contrasts. A zero vector is a true drop, not a zero-effect contribution.

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/test_run_influence.py
"""Leave-one-run-out influence on the combined contrast.

A fixed-effects combination can rest on a single run. The forest plot shows each run's
estimate at the selected peaks; this measures what each run does to the whole map.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from fmri_pipeline.analysis import run_level

SHAPE = (6, 6, 6)
N_FRAMES = 48
TR = 2.0
ONSETS = np.arange(4, 44, 10).astype(float)


def _model(active_runs, n_runs=3, seed=0):
    from nilearn.glm.first_level import FirstLevelModel

    rng = np.random.default_rng(seed)
    mask = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    bolds, events = [], []
    for run in range(n_runs):
        data = 100 + rng.standard_normal(SHAPE + (N_FRAMES,)).astype(np.float32)
        if run in active_runs:
            for onset in ONSETS:
                frame = int(onset / TR)
                data[1:4, 1:4, 1:4, frame : frame + 3] += 12.0
        bolds.append(nib.Nifti1Image(data, np.eye(4)))
        events.append(
            pd.DataFrame({"onset": ONSETS, "duration": 4.0, "trial_type": "task"})
        )
    model = FirstLevelModel(
        t_r=TR, mask_img=mask, hrf_model="spm", drift_model=None,
        minimize_memory=False, standardize=False, signal_scaling=False,
    )
    model.fit(bolds, events=events)
    return model


def test_one_row_per_run():
    model = _model(active_runs={0, 1, 2})
    rows = run_level.compute_run_influence(
        model, "task", run_labels=("run-01", "run-02", "run-03")
    )
    assert [r.dropped_run for r in rows] == ["run-01", "run-02", "run-03"]


def test_dropping_the_only_active_run_reduces_survivors_most():
    """The effect lives in run-02 alone, so dropping it must cost the most."""
    model = _model(active_runs={1})
    rows = run_level.compute_run_influence(
        model, "task", run_labels=("run-01", "run-02", "run-03"), threshold=2.3
    )
    by_run = {r.dropped_run: r for r in rows}
    assert by_run["run-02"].delta < by_run["run-01"].delta
    assert by_run["run-02"].delta < by_run["run-03"].delta


def test_a_zero_vector_drops_the_run_rather_than_contributing_zero():
    """With two runs, dropping one must equal that run's own contrast exactly.

    This is the property the whole approach rests on. If nilearn ever stopped skipping
    null contrast vectors, a dropped run would instead contribute a zero effect and an
    inflated variance, and every delta here would be silently wrong.
    """
    model = _model(active_runs={0, 1}, n_runs=2)
    masker = model.masker_

    dropped_second = masker.transform(
        model.compute_contrast(
            [np.asarray(v) for v in run_level._contrast_vectors(model, "task")[:1]]
            + [np.zeros_like(run_level._contrast_vectors(model, "task")[1])],
            output_type="stat",
        )
    ).ravel()
    run_one_alone = masker.transform(
        model.compute_contrast(
            [
                run_level._contrast_vectors(model, "task")[0],
                np.zeros_like(run_level._contrast_vectors(model, "task")[1]),
            ],
            output_type="stat",
        )
    ).ravel()
    np.testing.assert_allclose(dropped_second, run_one_alone, rtol=1e-6)


def test_single_run_model_yields_none():
    model = _model(active_runs={0}, n_runs=1)
    assert run_level.compute_run_influence(model, "task", run_labels=("run-01",)) is None


def test_writer_columns(tmp_path):
    model = _model(active_runs={0, 1, 2})
    rows = run_level.compute_run_influence(
        model, "task", run_labels=("run-01", "run-02", "run-03")
    )
    path = run_level.write_run_influence(
        rows, out_dir=tmp_path, stem="sub-01_task-x_contrast-c", cfg_hash="abc123"
    )
    frame = pd.read_csv(path, sep="\t")
    assert list(frame.columns) == [
        "dropped_run", "survivors", "delta", "max_abs_z", "correlation"
    ]
    assert len(frame) == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/test_run_influence.py -v`
Expected: FAIL — `compute_run_influence` does not exist.

- [ ] **Step 3: Implement**

Append to `fmri_pipeline/analysis/run_level.py`:

```python
@dataclass(frozen=True)
class RunInfluence:
    """What dropping one run does to the combined map."""

    dropped_run: str
    survivors: int
    delta: int
    max_abs_z: float
    correlation: float


def compute_run_influence(
    flm: Any,
    contrast_def: Any,
    *,
    run_labels: Sequence[str] = (),
    threshold: float = 2.3,
) -> Optional[Tuple[RunInfluence, ...]]:
    """Recombine the contrast with each run dropped in turn.

    A dropped run is expressed as an all-zero contrast vector, which
    ``compute_fixed_effect_contrast`` skips outright while dividing by the count of
    surviving contrasts. That makes this exact rather than a second pooling rule: the
    all-runs case is the stored map, so the deltas reconcile with the cluster table.

    ``None`` for a single-run model, which has nothing to drop.
    """
    designs = list(getattr(flm, "design_matrices_", []) or [])
    masker = getattr(flm, "masker_", None)
    if masker is None or len(designs) < 2:
        return None

    try:
        vectors = _contrast_vectors(flm, contrast_def)
    except Exception as exc:
        logger.warning("Could not expand the contrast for run influence (%s)", exc)
        return None

    def _z(vecs: List[np.ndarray]) -> Optional[np.ndarray]:
        try:
            img = flm.compute_contrast(vecs, output_type="z_score")
        except Exception as exc:
            logger.warning("Could not recombine the contrast (%s)", exc)
            return None
        values = np.asarray(masker.transform(img), dtype=float).ravel()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    combined = _z(vectors)
    if combined is None:
        return None
    baseline = int(np.sum(np.abs(combined) > threshold))

    labels = [
        str(run_labels[i]) if i < len(run_labels) else f"run-{i + 1:02d}"
        for i in range(len(vectors))
    ]

    rows: List[RunInfluence] = []
    for index, label in enumerate(labels):
        held_out = [
            np.zeros_like(v) if i == index else v for i, v in enumerate(vectors)
        ]
        reduced = _z(held_out)
        if reduced is None:
            return None
        survivors = int(np.sum(np.abs(reduced) > threshold))
        if np.std(reduced) == 0 or np.std(combined) == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(reduced, combined)[0, 1])
        rows.append(
            RunInfluence(
                dropped_run=label,
                survivors=survivors,
                delta=survivors - baseline,
                max_abs_z=float(np.abs(reduced).max()),
                correlation=correlation,
            )
        )
    return tuple(rows)


def write_run_influence(
    rows: Sequence[RunInfluence], *, out_dir: Path, stem: str, cfg_hash: str
) -> Optional[Path]:
    """Write one row per dropped run."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_desc-runinfluence_{cfg_hash}.tsv"
    frame = pd.DataFrame(
        [
            {
                "dropped_run": r.dropped_run,
                "survivors": r.survivors,
                "delta": r.delta,
                "max_abs_z": r.max_abs_z,
                "correlation": r.correlation,
            }
            for r in rows
        ]
    )
    try:
        frame.to_csv(path, sep="\t", index=False)
    except Exception as exc:
        logger.warning("Could not write %s (%s)", path.name, exc)
        return None
    return path
```

Add `RunInfluence`, `compute_run_influence`, `write_run_influence` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/fmri/test_run_influence.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/run_level.py tests/fmri/test_run_influence.py
git commit -m "feat(fmri): add leave-one-run-out influence

Exact rather than a second pooling rule: compute_fixed_effect_contrast skips
all-zero contrast vectors and divides by the surviving count, so a zero vector
is a true drop and the all-runs case is the stored map."
```

---

## Task 4: Wire both statistics into the pipeline and manifest

**Files:**
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py:333-386,597-604`
- Modify: `fmri_pipeline/analysis/report/manifest.py:33-34,127-128,250-252,264-265,308-309,703-704,761-762`
- Test: `tests/fmri/test_fmri_manifest_records_what_was_fit.py`

**Interfaces:**
- Consumes: `compute_sign_flip_null`, `write_sign_flip_null`, `compute_run_influence`, `write_run_influence` (Tasks 2–3).
- Produces: manifest fields `sign_flip_null_tsv: Optional[Path]`, `run_influence_tsv: Optional[Path]`, and scalars `sign_flip_fwe_height`, `sign_flip_fwe_survivors`, `sign_flip_global_p`, `sign_flip_p_floor`, `sign_flip_n_patterns`, `sign_flip_n_runs` (all `Optional[float]`/`Optional[int]`). Consumed by Tasks 5, 10, 11 and 21.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/test_fmri_manifest_records_what_was_fit.py`:

```python
def test_manifest_records_run_inference_paths_and_scalars(tmp_path):
    """Both artefacts and the scalars the report reads without reopening a TSV."""
    from fmri_pipeline.analysis.report.manifest import (
        read_report_manifest,
        write_report_manifest,
    )

    tsv_a = tmp_path / "signflip.tsv"
    tsv_b = tmp_path / "influence.tsv"
    tsv_a.write_text("pattern\tsigns\tmax_abs_z\n1\t+++\t4.0\n")
    tsv_b.write_text("dropped_run\tsurvivors\tdelta\tmax_abs_z\tcorrelation\n")

    path = write_report_manifest(
        contrast_dir=tmp_path,
        subject="sub-01",
        task="x",
        contrast_name="c",
        sign_flip_null_tsv=tsv_a,
        run_influence_tsv=tsv_b,
        sign_flip_fwe_height=7.02,
        sign_flip_fwe_survivors=38,
        sign_flip_global_p=0.0606,
        sign_flip_p_floor=0.0606,
        sign_flip_n_patterns=32,
        sign_flip_n_runs=6,
        sign_flip_observed_max=8.87,
    )
    manifest = read_report_manifest(path)
    assert manifest.sign_flip_null_tsv == tsv_a
    assert manifest.run_influence_tsv == tsv_b
    assert manifest.sign_flip_fwe_height == 7.02
    assert manifest.sign_flip_p_floor == 0.0606
    assert manifest.sign_flip_n_runs == 6


def test_sign_flip_scalars_absent_when_no_null_was_computed(tmp_path):
    """A single-run contrast records nothing rather than zeros."""
    from fmri_pipeline.analysis.report.manifest import (
        read_report_manifest,
        write_report_manifest,
    )

    path = write_report_manifest(
        contrast_dir=tmp_path, subject="sub-01", task="x", contrast_name="c"
    )
    manifest = read_report_manifest(path)
    assert manifest.sign_flip_null_tsv is None
    assert manifest.sign_flip_fwe_height is None
    assert manifest.sign_flip_p_floor is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/test_fmri_manifest_records_what_was_fit.py -v -k run_inference or sign_flip`
Expected: FAIL — `write_report_manifest() got an unexpected keyword argument 'sign_flip_null_tsv'`.

- [ ] **Step 3: Extend the manifest**

In `fmri_pipeline/analysis/report/manifest.py`:

Add to the path-key tuple at line 33-34 (alongside `"run_effect_map"`, `"run_variance_map"`): `"sign_flip_null_tsv"`, `"run_influence_tsv"`.

Add to the `ContrastManifest` dataclass near line 127:

```python
    sign_flip_null_tsv: Optional[Path] = None
    run_influence_tsv: Optional[Path] = None
    sign_flip_fwe_height: Optional[float] = None
    sign_flip_fwe_survivors: Optional[int] = None
    sign_flip_global_p: Optional[float] = None
    sign_flip_p_floor: Optional[float] = None
    sign_flip_n_patterns: Optional[int] = None
    sign_flip_n_runs: Optional[int] = None
    sign_flip_observed_max: Optional[float] = None
```

Add the same keys to both serialisation dicts (lines 264-265 and 308-309), add matching
keyword parameters to `write_report_manifest` (line 703-704) defaulting to `None`, and
pass them through at line 761-762 with `Path(...) if ... else None` for the two paths
and plain coercion for the scalars.

Add a validation beside the existing paired check at line 250:

```python
    if (manifest.sign_flip_null_tsv is None) != (manifest.sign_flip_fwe_height is None):
        raise ValueError(
            "sign_flip_null_tsv and sign_flip_fwe_height must either both be present "
            "or both be absent: a height with no enumerated null cannot be checked, "
            "and a null with no height was not summarised."
        )
```

- [ ] **Step 4: Call both statistics from the pipeline**

In `fmri_pipeline/pipelines/fmri_analysis.py`, extend `_run_level_maps` to also compute
and write the two new artefacts, returning them. Change its return annotation to
`tuple[Optional[Path], Optional[Path], list[str], Optional[Path], Optional[Path], dict]`
and append before the existing `return`:

```python
        from fmri_pipeline.analysis.run_level import (
            compute_run_influence,
            compute_sign_flip_null,
            write_run_influence,
            write_sign_flip_null,
        )

        sign_flip_path: Optional[Path] = None
        influence_path: Optional[Path] = None
        scalars: dict = {}

        try:
            null = compute_sign_flip_null(flm, contrast_def)
        except Exception as exc:
            self.logger.warning("Could not compute the sign-flip null (%s)", exc)
            null = None
        if null is not None:
            sign_flip_path = write_sign_flip_null(
                null, out_dir=out_dir, stem=stem, cfg_hash=cfg_hash
            )
            if sign_flip_path is not None:
                scalars = {
                    "sign_flip_fwe_height": null.fwe_height,
                    "sign_flip_fwe_survivors": null.fwe_survivors,
                    "sign_flip_global_p": null.global_p,
                    "sign_flip_p_floor": null.p_floor,
                    "sign_flip_n_patterns": null.n_patterns,
                    "sign_flip_n_runs": null.n_runs,
                    "sign_flip_observed_max": null.observed_max,
                }
                self.logger.info(
                    "Sign-flip null over %d run(s): FWE height %.2f, %d voxel(s), "
                    "p = %.3f (floor %.3f)",
                    null.n_runs, null.fwe_height, null.fwe_survivors,
                    null.global_p, null.p_floor,
                )

        try:
            influence = compute_run_influence(
                flm, contrast_def, run_labels=labels,
                threshold=float(getattr(plotting_cfg, "z_threshold", 2.3)),
            )
        except Exception as exc:
            self.logger.warning("Could not compute run influence (%s)", exc)
            influence = None
        if influence:
            influence_path = write_run_influence(
                influence, out_dir=out_dir, stem=stem, cfg_hash=cfg_hash
            )
```

`_run_level_maps` does not currently receive `plotting_cfg`; add it as a keyword
parameter and pass it from the call site at line 597. Update that call site to unpack
six values and forward all of them into `write_report_manifest`.

- [ ] **Step 5: Run the targeted tests**

Run: `.venv/bin/python -m pytest tests/fmri/test_fmri_manifest_records_what_was_fit.py tests/fmri/test_fmri_first_level_manifest_wiring.py tests/fmri/report/test_manifest.py tests/fmri/report/test_manifest_writing.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add fmri_pipeline/pipelines/fmri_analysis.py fmri_pipeline/analysis/report/manifest.py tests/fmri/test_fmri_manifest_records_what_was_fit.py
git commit -m "feat(fmri): record the sign-flip null and run influence in the manifest"
```

---

# PHASE 1B — Reframing, no new computation

## Task 5: Sign-flip row in the threshold table, with the p floor

**Files:**
- Modify: `fmri_pipeline/analysis/report/inference.py:65-127,354-416`
- Modify: `fmri_pipeline/analysis/report/subject.py` (threshold table builder)
- Test: `tests/fmri/report/test_calibration.py`

**Interfaces:**
- Consumes: manifest scalars from Task 4.
- Produces: `ThresholdContext.sign_flip: Optional[SignFlipSummary]`, where `SignFlipSummary` is a frozen dataclass with `height: float`, `survivors: int`, `global_p: float`, `p_floor: float`, `n_runs: int`, `n_patterns: int`. Consumed by Task 11.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_calibration.py`:

```python
def test_threshold_context_carries_a_sign_flip_summary():
    import numpy as np
    from fmri_pipeline.analysis.report.inference import SignFlipSummary, threshold_context

    values = np.random.default_rng(0).standard_normal(5000)
    context = threshold_context(
        values, applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True,
        sign_flip=SignFlipSummary(
            height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
            n_runs=6, n_patterns=32, observed_max=8.87,
        ),
    )
    assert context.sign_flip.height == 7.02
    assert context.sign_flip.survivors == 38


def test_sign_flip_is_optional():
    """A single-run contrast has no null; the table must still build."""
    import numpy as np
    from fmri_pipeline.analysis.report.inference import threshold_context

    values = np.random.default_rng(0).standard_normal(5000)
    context = threshold_context(
        values, applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True
    )
    assert context.sign_flip is None


def test_p_floor_is_reachable_only_above_six_runs():
    """The identity always ties, so six runs cannot reach 0.05 at map level."""
    from fmri_pipeline.analysis.report.inference import sign_flip_p_floor

    assert sign_flip_p_floor(6) == pytest.approx(2 / 33)
    assert sign_flip_p_floor(6) > 0.05
    assert sign_flip_p_floor(7) < 0.05
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_calibration.py -v -k sign_flip or p_floor`
Expected: FAIL — `cannot import name 'SignFlipSummary'`.

- [ ] **Step 3: Implement**

In `fmri_pipeline/analysis/report/inference.py`:

```python
@dataclass(frozen=True)
class SignFlipSummary:
    """A familywise height from run exchangeability, and what it is worth.

    ``p_floor`` is carried beside ``global_p`` because the two are not independent:
    the identity sign pattern is always a member of the null and always ties the
    observed maximum, so ``global_p`` can never fall below ``p_floor``. Printed alone,
    a p of 0.061 from six runs reads as a near-miss when it is the smallest value the
    test can return.
    """

    height: float
    survivors: int
    global_p: float
    p_floor: float
    n_runs: int
    n_patterns: int
    observed_max: float

    @property
    def floor_limited(self) -> bool:
        """Whether no map-level p below 0.05 is reachable with this many runs."""
        return self.p_floor > 0.05


def sign_flip_p_floor(n_runs: int) -> float:
    """Smallest attainable global p for a run sign-flip test over ``n_runs`` runs."""
    if n_runs < 2:
        raise ValueError(f"A sign-flip null needs at least two runs, got {n_runs!r}.")
    return 2.0 / (2 ** (n_runs - 1) + 1)
```

Add `sign_flip: Optional[SignFlipSummary] = None` to `ThresholdContext`, add a
`sign_flip: Optional[SignFlipSummary] = None` keyword to `threshold_context`, and pass
it straight through to the returned `ThresholdContext`. Export both new names in
`__all__`.

- [ ] **Step 4: Add the row to the rendered table**

In `subject.py`'s thresholds panel builder, append a row after the Bonferroni row when
`context.sign_flip` is not `None`:

| Threshold | Rejection region | Voxels surviving |
|---|---|---|
| `Run sign-flip, FWE {alpha}` | `\|z\| > {height:.2f}` | `{survivors:,}` |

and extend the caption with, when `floor_limited`:

> Run sign-flip: {n_patterns} exact sign patterns over {n_runs} runs, exchangeable by
> run. Global p = {global_p:.3f}, which is the smallest value this test can return —
> the unflipped pattern is always a member of the null and always ties the observed
> maximum, so with {n_runs} runs no map-level p below {p_floor:.3f} is reachable. The
> height is unaffected by that floor.

and otherwise the same sentence ending at "…reachable." replaced by "the floor is
{p_floor:.3f}."

Also delete the now-false clause in the existing caption: "This pipeline applies no
cluster-level correction, so none of these heights is familywise-corrected for extent."
Replace with: "No cluster-extent correction is applied; the sign-flip height is
familywise-corrected over voxels, not over extent."

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_calibration.py tests/fmri/report/test_subject.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fmri_pipeline/analysis/report/inference.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_calibration.py
git commit -m "feat(report): add the sign-flip row and its p floor to the threshold table"
```

---

## Task 6: Order peaks by |z| so the strongest cluster is never hidden

Currently `timeseries.py:165` and `run_consistency.py:151` both take `list(peaks)[:max_peaks]` in the order supplied, which is signed. On sub-0001 the largest effect in the map — a 138,213 mm³ cluster peaking at z = −8.81 — appears in neither panel.

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/timeseries.py:145-170`
- Modify: `fmri_pipeline/analysis/report/figures/run_consistency.py:96-155`
- Test: `tests/fmri/report/test_timeseries.py`, `tests/fmri/report/test_run_consistency.py`

**Interfaces:**
- Consumes: the existing `peaks: Sequence[Tuple[str, Tuple[float, float, float]]]` argument, plus a new optional parallel `peak_stats: Optional[Sequence[float]]`.
- Produces: unchanged signatures apart from the new keyword; ordering is by `abs(stat)` descending when `peak_stats` is supplied, and unchanged otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_timeseries.py`:

```python
def test_peaks_are_ordered_by_absolute_stat():
    """A strong negative peak must outrank a weaker positive one."""
    from fmri_pipeline.analysis.report.figures.timeseries import order_peaks

    peaks = [("1", (0.0, 0.0, 0.0)), ("2", (1.0, 1.0, 1.0)), ("3", (2.0, 2.0, 2.0))]
    stats = [3.0, -8.8, 5.0]
    assert [p[0] for p in order_peaks(peaks, stats, max_peaks=3)] == ["2", "3", "1"]


def test_order_is_preserved_when_no_stats_are_supplied():
    from fmri_pipeline.analysis.report.figures.timeseries import order_peaks

    peaks = [("1", (0.0, 0.0, 0.0)), ("2", (1.0, 1.0, 1.0))]
    assert [p[0] for p in order_peaks(peaks, None, max_peaks=2)] == ["1", "2"]


def test_cap_is_applied_after_ordering():
    from fmri_pipeline.analysis.report.figures.timeseries import order_peaks

    peaks = [("1", (0.0, 0.0, 0.0)), ("2", (1.0, 1.0, 1.0)), ("3", (2.0, 2.0, 2.0))]
    stats = [1.0, -9.0, 2.0]
    assert [p[0] for p in order_peaks(peaks, stats, max_peaks=1)] == ["2"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_timeseries.py -v -k order_peaks`
Expected: FAIL — `cannot import name 'order_peaks'`.

- [ ] **Step 3: Implement**

Add to `fmri_pipeline/analysis/report/figures/timeseries.py`:

```python
def order_peaks(
    peaks: Sequence[Tuple[str, Tuple[float, float, float]]],
    peak_stats: Optional[Sequence[float]],
    *,
    max_peaks: int,
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Return the ``max_peaks`` strongest peaks, strongest first.

    Ordered by ``abs(stat)`` rather than by the signed value: a contrast's largest
    effect is frequently negative, and a signed order drops it off the end of every
    capped panel while keeping weaker positive peaks. Falls back to the supplied order
    when no statistics accompany the peaks, since an arbitrary reorder would be worse
    than the caller's own.
    """
    ordered = list(peaks)
    if peak_stats is not None and len(peak_stats) == len(ordered):
        ordered = [
            peak
            for _, peak in sorted(
                zip(peak_stats, ordered),
                key=lambda pair: abs(float(pair[0])),
                reverse=True,
            )
        ]
    return ordered[: max(int(max_peaks), 1)]
```

Import it in `run_consistency.py` and replace both `list(peaks)[: max(int(max_peaks), 1)]`
expressions (`timeseries.py:165`, `run_consistency.py:151`) with a call to it, threading
a new `peak_stats: Optional[Sequence[float]] = None` keyword through
`collect_peak_responses`, `collect_peak_estimates` and `peak_forest_figure`.

In `subject.py`, pass the cluster table's `Peak Stat` column as `peak_stats` at both
call sites.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_timeseries.py tests/fmri/report/test_run_consistency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/timeseries.py fmri_pipeline/analysis/report/figures/run_consistency.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_timeseries.py tests/fmri/report/test_run_consistency.py
git commit -m "fix(report): order peaks by |z| so the strongest cluster is shown

On sub-0001 the largest effect in the map is a 138,213 mm3 cluster peaking at
z = -8.81, and a signed order kept it out of both the response and forest panels."
```

---

## Task 7: Split the contrast regressors out of the VIF panel

The two regressors carrying the contrast sit at VIF ≈ 8–9 among 46 other bars. That is the number that costs the comparison its precision, and it is currently indistinguishable.

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/design.py`
- Test: `tests/fmri/report/test_design.py`

**Interfaces:**
- Consumes: the existing per-regressor VIF mapping and the contrast vector already available to the design section.
- Produces: `vif_figure(..., contrast_columns: Sequence[str] = ())` — draws a separate, labelled upper subplot containing only the contrast-weighted regressors, sharing the log y-axis with the main panel.

- [ ] **Step 1: Write the failing test**

Append to `tests/fmri/report/test_design.py`:

```python
def test_contrast_regressors_get_their_own_axis(tmp_path):
    """The regressors the contrast weights are drawn apart from the other 45."""
    import matplotlib
    matplotlib.use("Agg")
    from fmri_pipeline.analysis.report.figures import design

    vifs = {"run-01": {"cond_a": 8.6, "cond_b": 7.9, "rot_x": 155.0, "drift_1": 130.0}}
    figure = design.vif_figure(vifs, contrast_columns=("cond_a", "cond_b"))
    axes = figure.get_axes()
    assert len(axes) >= 2
    contrast_axis = axes[0]
    labels = [t.get_text() for t in contrast_axis.get_xticklabels()]
    assert set(labels) == {"cond_a", "cond_b"}


def test_no_separate_axis_without_contrast_columns(tmp_path):
    """Unchanged behaviour when the contrast's columns are unknown."""
    import matplotlib
    matplotlib.use("Agg")
    from fmri_pipeline.analysis.report.figures import design

    vifs = {"run-01": {"cond_a": 8.6, "rot_x": 155.0}}
    figure = design.vif_figure(vifs, contrast_columns=())
    assert len(figure.get_axes()) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v -k contrast_regressors or separate_axis`
Expected: FAIL — `vif_figure() got an unexpected keyword argument 'contrast_columns'`.

- [ ] **Step 3: Implement**

Add the `contrast_columns` keyword to `vif_figure`. When it is non-empty and matches at
least one regressor, build the figure with `gridspec` as two vertically stacked axes
sharing the log y-scale: the upper holds only the contrast columns with the existing
median-dot/range-bar marks, titled "Regressors the contrast weights"; the lower holds
the remainder, grouped as today. Keep the current single-axis path untouched when
`contrast_columns` is empty, so a manifest written before contrast columns were
recorded renders exactly as before.

In `subject.py`, pass `manifest.contrast_columns` to `vif_figure`.

Update the caption to name the split and keep it a measurement:

> Variance inflation per regressor, every run on one axis. The regressors this contrast
> weights are drawn separately above, since inflation on those is what costs the
> comparison its precision — inflation elsewhere costs it nothing. A regressor inflated
> in every run is a property of the design; one inflated in a single run is a property
> of that run. Reported as a measurement; no cutoff is applied.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/design.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_design.py
git commit -m "feat(report): draw the contrast's own regressors apart in the VIF panel"
```

---

## Task 8: Source the cluster table from the MNI map

The MNI effect, variance and z maps exist in derivatives at the same config hash. The report prints "coordinates: scanner-native (mm), not MNI; not atlas-referable" and omits anatomical labels. Native-space peak coordinates are not publishable.

**Files:**
- Modify: `fmri_pipeline/analysis/report/manifest.py` (record the MNI map paths)
- Modify: `fmri_pipeline/analysis/report/subject.py` (cluster table source)
- Modify: `fmri_pipeline/analysis/report/atlas.py:atlas_applies_to`
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py` (record the paths)
- Test: `tests/fmri/report/test_atlas.py`, `tests/fmri/report/test_subject.py`

**Interfaces:**
- Consumes: manifest fields.
- Produces: manifest field `mni_z_map: Optional[Path]` (plus `mni_effect_map`, `mni_variance_map`); `atlas_applies_to(space)` gains an `mni_map_available: bool` parameter so the gate moves rather than being removed.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_atlas.py
def test_atlas_applies_when_an_mni_map_is_available_for_a_native_fit():
    """The fit is native; the table is drawn from the MNI map beside it."""
    from fmri_pipeline.analysis.report.atlas import atlas_applies_to

    assert atlas_applies_to("native", mni_map_available=True) is True
    assert atlas_applies_to("native", mni_map_available=False) is False
    assert atlas_applies_to("MNI152NLin2009cAsym", mni_map_available=False) is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_atlas.py -v -k mni_map_available`
Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Implement**

Add `mni_effect_map`, `mni_variance_map`, `mni_z_map` to the manifest path keys, dataclass,
both serialisation dicts and `write_report_manifest`, following the `run_effect_map`
pattern exactly. Record them in `fmri_analysis.py` where the MNI maps are already written.

Give `atlas_applies_to` the new keyword:

```python
def atlas_applies_to(space: str, *, mni_map_available: bool = False) -> bool:
    """Whether atlas labels are defined for coordinates drawn from this contrast.

    An atlas is defined in MNI. A native-space *fit* is still atlas-referable when an
    MNI map of the same contrast exists beside it, because the table's coordinates come
    from that map rather than from the fitted volume -- which is why this takes the
    availability rather than inferring it from the space alone.
    """
    if _is_mni(space):
        return True
    return bool(mni_map_available)
```

In `subject.py`, build the clusters table from `manifest.mni_z_map` when it exists,
labelling the coordinate column `MNI (mm)`, and attach atlas labels through the existing
path. Keep the native table as an additional TSV download. Where the MNI map is absent,
behaviour is unchanged.

Replace the caption clause "coordinates: scanner-native (mm), not MNI; not
atlas-referable" with "coordinates: MNI152NLin2009cAsym (mm), from the MNI map written
beside this contrast; the maps above are drawn in the space the GLM was fitted in", and
drop "no anatomical labels: the atlas is defined in MNI and these coordinates are in
native space".

Delete the standalone paragraph "No glass brain for this contrast: the projection is
defined only against the MNI schematic, and these results are in native space" and gate
the glass brain on `mni_z_map` instead.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_atlas.py tests/fmri/report/test_subject.py tests/fmri/report/test_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/atlas.py fmri_pipeline/analysis/report/manifest.py fmri_pipeline/analysis/report/subject.py fmri_pipeline/pipelines/fmri_analysis.py tests/fmri/report/test_atlas.py
git commit -m "feat(report): draw the cluster table from the MNI map beside the fit

The MNI effect, variance and z maps already exist at the same config hash. Peak
coordinates become atlas-referable while the drawn maps stay in the fitted space."
```

---

## Task 9: Reframe "At a glance" and fix the calibration caption

**Files:**
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_subject.py`

**Interfaces:**
- Consumes: `ThresholdContext` (Task 5), manifest scalars (Task 4).
- Produces: no new API.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_subject.py
def test_at_a_glance_states_the_expected_count_beside_the_observed():
    """The raw survivor count alone reads as a result; the pair is the measurement."""
    from fmri_pipeline.analysis.report.subject import summary_rows

    rows = dict(
        summary_rows(
            contrast_name="c",
            applied_threshold=2.3,
            survivors=8463,
            n_voxels=50626,
            expected_under_fitted_null=8001.0,
            sign_flip_height=7.02,
            sign_flip_survivors=38,
        )
    )
    line = rows["c: voxels above the threshold"]
    assert "8,463" in line
    assert "8,001" in line
    assert "16.72%" in line
    assert rows["c: familywise (run sign-flip)"] == "|z| > 7.02 — 38 voxels"


def test_calibration_caption_does_not_blame_autocorrelation():
    from fmri_pipeline.analysis.report.subject import CALIBRATION_CAPTION

    assert "autocorrelation" not in CALIBRATION_CAPTION.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject.py -v -k at_a_glance or calibration_caption`
Expected: FAIL — `cannot import name 'summary_rows'` / `CALIBRATION_CAPTION`.

- [ ] **Step 3: Implement**

Extract the "At a glance" row construction into a module-level `summary_rows(...)`
returning an ordered sequence of `(label, value)` pairs, so it is testable without
rendering. The survivors row becomes:

```
8,463 of 50,626 (16.72%) — 8,001 expected under this map's own fitted null
```

and a new row is appended when the sign-flip scalars are present:

```
c: familywise (run sign-flip)    |z| > 7.02 — 38 voxels
```

Replace the calibration caption with a module constant that states the measured cause
rather than the assumed one:

```python
CALIBRATION_CAPTION = (
    "Where each threshold in the table above falls on the map's own distribution, "
    "with the fitted null beside the theoretical N(0, 1) the threshold assumes. "
    "Over-dispersion relative to N(0, 1) is a measurement, not an assumption: this "
    "panel states the fitted null's mean and width, the residual autocorrelation "
    "panel states what the residuals do, and the run-offset panel states what each "
    "run contributes to the centre. Nothing in a thresholded mosaic reveals any of "
    "the three."
)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_subject.py tests/fmri/report/test_subject_inference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_subject.py
git commit -m "fix(report): lead with observed against expected, and stop blaming autocorrelation

The map's own fitted null predicts 8,001 of the 8,463 survivors. The caption
attributed the over-dispersion to unmodelled autocorrelation, which this
subject's median residual ACF(1) of 0.05-0.07 does not support."
```

---

## Task 10: Phase 1 checkpoint — regenerate and inspect the report

- [ ] **Step 1: Regenerate against the real derivatives**

Run the report command that produced `outputs/fmri_report_redesign/`, writing to a new
directory `outputs/fmri_report_phase1/`.

- [ ] **Step 2: Confirm the numbers reconcile**

Check by hand:
- The threshold table's sign-flip row exists, and its survivor count matches the manifest scalar.
- The all-runs survivor count in the run-influence TSV equals the cluster table's count. They must be identical — that is the property Task 3's zero-vector approach buys.
- The cluster table's coordinates are MNI and carry atlas labels.
- The strongest peak in the response and forest panels is now the negative one.

- [ ] **Step 3: Commit any fixes, then stop for review**

Phase 1 is the scientific core. Do not begin Phase 2 until the regenerated report has been reviewed.

---

# PHASE 2 — New panels

## Task 11: Run influence and sign-flip panels

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/run_inference.py`
- Create: `tests/fmri/report/test_run_inference.py`
- Modify: `fmri_pipeline/analysis/report/subject.py`

**Interfaces:**
- Consumes: `manifest.run_influence_tsv`, `manifest.sign_flip_null_tsv`, and the sign-flip scalars (Task 4); `SignFlipSummary` (Task 5).
- Produces:
  - `run_influence_figure(rows: pd.DataFrame, *, baseline: int) -> plt.Figure`
  - `sign_flip_figure(null_max: Sequence[float], *, summary: SignFlipSummary) -> plt.Figure`

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/report/test_run_inference.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from fmri_pipeline.analysis.report.figures import run_inference
from fmri_pipeline.analysis.report.inference import SignFlipSummary


@pytest.fixture
def influence_rows():
    return pd.DataFrame(
        {
            "dropped_run": ["run-01", "run-02", "run-03"],
            "survivors": [7108, 8003, 10440],
            "delta": [-1367, -472, 1965],
            "max_abs_z": [8.18, 7.68, 8.47],
            "correlation": [0.941, 0.913, 0.953],
        }
    )


def test_influence_figure_draws_one_row_per_run(influence_rows):
    figure = run_inference.run_influence_figure(influence_rows, baseline=8475)
    axis = figure.get_axes()[0]
    assert [t.get_text() for t in axis.get_yticklabels()] == [
        "run-01", "run-02", "run-03"
    ]


def test_influence_figure_marks_the_baseline(influence_rows):
    """The all-runs count is the reference every delta is measured from."""
    figure = run_inference.run_influence_figure(influence_rows, baseline=8475)
    axis = figure.get_axes()[0]
    assert any(round(line.get_xdata()[0]) == 8475 for line in axis.get_lines())


def test_influence_figure_states_no_verdict(influence_rows):
    """Runs differing is a measurement. No run may be scored."""
    figure = run_inference.run_influence_figure(influence_rows, baseline=8475)
    text = " ".join(t.get_text().lower() for t in figure.findobj(match=lambda o: hasattr(o, "get_text")))
    for word in ("outlier", "fail", "exclude", "bad", "reject"):
        assert word not in text


def test_sign_flip_figure_marks_observed_and_height():
    summary = SignFlipSummary(
        height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
        n_runs=6, n_patterns=32, observed_max=8.87,
    )
    null_max = [8.87, 5.7, 5.9, 6.1, 5.4, 6.3, 5.8, 7.0]
    figure = run_inference.sign_flip_figure(null_max, summary=summary)
    axis = figure.get_axes()[0]
    positions = [round(line.get_xdata()[0], 2) for line in axis.get_lines()]
    assert 8.87 in positions
    assert 7.02 in positions


def test_sign_flip_figure_names_the_floor_when_it_binds():
    summary = SignFlipSummary(
        height=7.02, survivors=38, global_p=0.0606, p_floor=0.0606,
        n_runs=6, n_patterns=32, observed_max=8.87,
    )
    figure = run_inference.sign_flip_figure([8.87, 5.7, 6.1], summary=summary)
    text = " ".join(
        t.get_text() for t in figure.findobj(match=lambda o: hasattr(o, "get_text"))
    )
    assert "0.061" in text
    assert "floor" in text.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_run_inference.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `run_inference.py` following the conventions of `run_consistency.py`: use
`plot_context()`, draw the provenance strip via the shared helper, and keep every axis
label unit-bearing.

```python
"""What each run does to the combined map, and what run exchangeability is worth.

The forest panel beside this one shows each run's estimate at the selected peaks.
These two show the whole map: which run moves it, and how far a map like it can be
pushed by relabelling runs alone.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fmri_pipeline.analysis.report.style import plot_context, provenance_strip
from fmri_pipeline.analysis.report.inference import SignFlipSummary


def run_influence_figure(rows, *, baseline: int) -> plt.Figure:
    """Survivor count with each run dropped, against the all-runs count."""
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.0, 0.5 * len(rows) + 1.8))
        positions = np.arange(len(rows))[::-1]

        axis.axvline(baseline, color="0.35", lw=1.0, zorder=1)
        axis.annotate(
            "all runs", xy=(baseline, positions.max() + 0.5),
            xytext=(3, 0), textcoords="offset points",
            fontsize=7, color="0.35", va="bottom",
        )
        for y, (_, row) in zip(positions, rows.iterrows()):
            axis.plot([baseline, row["survivors"]], [y, y], color="#9ecae1", lw=2.5, zorder=2)
            axis.plot(row["survivors"], y, "o", color="#1f77b4", ms=7, zorder=3)
            axis.annotate(
                f'{int(row["delta"]):+,}  ·  r = {row["correlation"]:.3f}',
                xy=(row["survivors"], y), xytext=(9, 0), textcoords="offset points",
                fontsize=7, va="center", color="0.25",
            )

        axis.set_yticks(positions)
        axis.set_yticklabels(list(rows["dropped_run"]))
        axis.set_xlabel("Voxels above the applied threshold")
        axis.margins(x=0.22)

        provenance_strip(
            figure,
            f"{len(rows)} run(s)  ·  dot: survivors with that run dropped  ·  "
            "line: all runs  ·  r is the correlation of the reduced map with the "
            "combined one  ·  runs differing is a measurement, not a fault",
        )
    return figure


def sign_flip_figure(null_max: Sequence[float], *, summary: SignFlipSummary) -> plt.Figure:
    """The enumerated null of the maximum statistic, with the observed value on it."""
    with plot_context():
        figure, axis = plt.subplots(figsize=(7.0, 3.2))
        axis.hist(list(null_max), bins=min(24, max(6, len(null_max) // 2)),
                  color="0.82", edgecolor="0.55", lw=0.6)
        axis.axvline(summary.height, color="#1f77b4", lw=1.4,
                     label=f"FWE 5%: |z| > {summary.height:.2f}")
        axis.axvline(summary.observed_max, color="#d95f02", lw=1.4, ls="--",
                     label="observed max |z|")
        axis.set_xlabel("max |z| under a run sign flip")
        axis.set_ylabel("sign patterns")
        axis.legend(loc="upper right", fontsize=7)

        floor = (
            f"global p = {summary.global_p:.3f}, which is this test's floor: the "
            f"unflipped pattern is always in the null and always ties the observed "
            f"maximum, so with {summary.n_runs} runs no map-level p below "
            f"{summary.p_floor:.3f} is reachable"
            if summary.floor_limited
            else f"global p = {summary.global_p:.3f}  ·  floor {summary.p_floor:.3f}"
        )
        provenance_strip(
            figure,
            f"{summary.n_patterns} exact sign patterns over {summary.n_runs} runs, "
            f"exchangeable by run  ·  {summary.survivors:,} voxel(s) at the FWE "
            f"height  ·  {floor}  ·  the height is unaffected by the floor",
        )
    return figure
```

Check `style.py` for the actual name of the provenance-strip helper before importing it
— the constant near `style.py:148` marks where it lives; `provenance_strip` above is a
placeholder for whatever that helper is actually called, and every other figure module
already calls it.

Wire both into `subject.py`'s diagnostics section for the contrast, returning `None`
when the manifest carries no such TSV, following the `run_consistency` precedent at
`subject.py:1499`. Build `SignFlipSummary` from the manifest scalars — not by importing
`run_level`, which would pull the analysis package into the report import path.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_run_inference.py tests/fmri/report/test_subject.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/run_inference.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_run_inference.py
git commit -m "feat(report): add run-influence and sign-flip-null panels"
```

---

## Task 12: Global per-run offset panel

The −0.61 shift in the fitted null is a whole-brain per-run offset: on sub-0001 the six
runs sit at −0.083, −0.012, **+0.057**, −0.042, −0.068, −0.037 %SC. It is what
manufactures the 138,213 mm³ negative cluster and the CSF-over-GM survival ordering, and
nothing in the report currently measures it.

**Files:**
- Create: `fmri_pipeline/analysis/report/figures/offsets.py`
- Create: `tests/fmri/report/test_offsets.py`
- Modify: `fmri_pipeline/analysis/report/subject.py`

**Interfaces:**
- Consumes: `manifest.run_effect_map` (4D, run on the fourth axis), `manifest.effect_map`, `manifest.mask`.
- Produces:
  - `run_offsets(run_effect_img, combined_effect_img, mask_img) -> RunOffsets` — frozen dataclass with `per_run: Tuple[float, ...]`, `run_labels: Tuple[str, ...]`, `combined: float`.
  - `offset_figure(offsets: RunOffsets) -> plt.Figure`

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/report/test_offsets.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import nibabel as nib

from fmri_pipeline.analysis.report.figures import offsets


def _images(per_run_means):
    shape = (4, 4, 4)
    mask = nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4))
    data = np.stack(
        [np.full(shape, m, dtype=np.float32) for m in per_run_means], axis=-1
    )
    run_img = nib.Nifti1Image(data, np.eye(4))
    combined = nib.Nifti1Image(
        np.full(shape, float(np.mean(per_run_means)), dtype=np.float32), np.eye(4)
    )
    return run_img, combined, mask


def test_offsets_are_the_whole_mask_mean_per_run():
    run_img, combined, mask = _images([-0.083, 0.057, -0.037])
    result = offsets.run_offsets(run_img, combined, mask)
    np.testing.assert_allclose(result.per_run, [-0.083, 0.057, -0.037], atol=1e-6)


def test_combined_offset_is_reported_separately():
    run_img, combined, mask = _images([-0.083, 0.057, -0.037])
    result = offsets.run_offsets(run_img, combined, mask)
    assert result.combined == np.float32(np.mean([-0.083, 0.057, -0.037]))


def test_only_mask_voxels_count():
    """A background voxel must not drag the offset toward zero."""
    shape = (4, 4, 4)
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[:2] = 1
    data = np.zeros(shape + (1,), dtype=np.float32)
    data[:2, ..., 0] = 0.5
    result = offsets.run_offsets(
        nib.Nifti1Image(data, np.eye(4)),
        nib.Nifti1Image(data[..., 0], np.eye(4)),
        nib.Nifti1Image(mask_data, np.eye(4)),
    )
    np.testing.assert_allclose(result.per_run, [0.5], atol=1e-6)


def test_figure_marks_zero_and_the_combined_value():
    run_img, combined, mask = _images([-0.083, 0.057, -0.037])
    figure = offsets.offset_figure(offsets.run_offsets(run_img, combined, mask))
    axis = figure.get_axes()[0]
    positions = [round(float(line.get_xdata()[0]), 4) for line in axis.get_lines()]
    assert 0.0 in positions
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_offsets.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `offsets.py` with `RunOffsets`, `run_offsets` and `offset_figure`. The figure is
a horizontal dot plot, one row per run, x in % signal change, with a solid line at zero
and a dashed line at the combined value. Caption:

> The whole-mask mean of each run's own estimate of this contrast. A contrast that
> differences two conditions has no reason to carry a brain-wide offset, so a non-zero
> value is signal shared across the whole mask rather than anatomy — the fitted null's
> centre is this quantity, and a map centred away from zero produces large clusters of
> the offset's sign and shifts survival toward whichever tissue class the offset reaches
> most. Reported as a measurement; no run is scored against it.

Wire into `subject.py`'s diagnostics section, returning `None` when
`manifest.run_effect_map` is absent.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_offsets.py tests/fmri/report/test_subject.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/offsets.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_offsets.py
git commit -m "feat(report): measure the whole-brain per-run offset

The fitted null's -0.61 centre is a per-run brain-wide offset, not anatomy."
```

---

## Task 13: Design-confound measurements

`event_raster.png` shows the two conditions time-segregated within several runs, and
`design_summary.tsv` shows event counts of 8/3 and 3/8. Neither is quantified.

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/design.py`
- Test: `tests/fmri/report/test_design.py`

**Interfaces:**
- Consumes: the recorded design matrices and contrast vector.
- Produces: `contrast_confound_rows(designs, contrast_vector, columns) -> List[dict]` with keys `run`, `r_with_time`, `r_with_drift_max`, `events_min`, `events_max`, `balance`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_design.py
def test_contrast_confound_detects_time_on_task_correlation():
    """A contrast whose positive condition sits early correlates with time."""
    import numpy as np
    import pandas as pd
    from fmri_pipeline.analysis.report.figures.design import contrast_confound_rows

    n = 100
    early = np.zeros(n); early[:40] = 1.0
    late = np.zeros(n); late[60:] = 1.0
    design = pd.DataFrame({"cond_a": early, "cond_b": late, "drift_1": np.linspace(-1, 1, n)})
    rows = contrast_confound_rows(
        [design], np.array([1.0, -1.0, 0.0]), ["cond_a", "cond_b", "drift_1"]
    )
    assert rows[0]["r_with_time"] < -0.5


def test_balanced_interleaved_design_has_low_time_correlation():
    import numpy as np
    import pandas as pd
    from fmri_pipeline.analysis.report.figures.design import contrast_confound_rows

    n = 100
    a = np.zeros(n); a[::10] = 1.0
    b = np.zeros(n); b[5::10] = 1.0
    design = pd.DataFrame({"cond_a": a, "cond_b": b, "drift_1": np.linspace(-1, 1, n)})
    rows = contrast_confound_rows(
        [design], np.array([1.0, -1.0, 0.0]), ["cond_a", "cond_b", "drift_1"]
    )
    assert abs(rows[0]["r_with_time"]) < 0.2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v -k confound`
Expected: FAIL — `cannot import name 'contrast_confound_rows'`.

- [ ] **Step 3: Implement**

```python
def contrast_confound_rows(designs, contrast_vector, columns):
    """Per-run correlation of the contrast's own regressor with time and with drift.

    The contrast regressor is ``X @ c`` -- the single time series the comparison
    actually tests. Correlating it with a linear ramp measures time-on-task confounding;
    correlating it with each drift column measures how much of the comparison the
    high-pass basis can absorb. A raster shows both and quantifies neither.

    Reported as measurements. A blocked design is *supposed* to correlate with time, so
    no cutoff is applied to either quantity.
    """
    import numpy as np

    rows = []
    for index, design in enumerate(designs):
        values = design.reindex(columns=list(columns), fill_value=0.0).to_numpy(float)
        regressor = values @ np.asarray(contrast_vector, dtype=float)
        ramp = np.linspace(-1.0, 1.0, regressor.size)
        drift_columns = [c for c in design.columns if str(c).startswith("drift")]
        drift = design.loc[:, drift_columns].to_numpy(float) if drift_columns else None

        def _r(other):
            if np.std(regressor) == 0 or np.std(other) == 0:
                return float("nan")
            return float(np.corrcoef(regressor, other)[0, 1])

        rows.append(
            {
                "run": f"run-{index + 1:02d}",
                "r_with_time": _r(ramp),
                "r_with_drift_max": (
                    float("nan")
                    if drift is None
                    else max(abs(_r(drift[:, j])) for j in range(drift.shape[1]))
                ),
            }
        )
    return rows
```

Add the resulting columns to the existing design-summary table in `subject.py`, beside
the event counts already there, and extend that table's caption with:

> `r` with time is the correlation of the contrast's own regressor with a linear ramp
> across the run: a design whose conditions are ordered rather than interleaved
> confounds the comparison with time-on-task, and the high-pass basis then absorbs part
> of it. `r` with drift is the largest such correlation against any drift column. Both
> are measurements; a blocked design is expected to correlate with time.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_design.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/design.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_design.py
git commit -m "feat(report): quantify time-on-task and drift confounding of the contrast"
```

---

# PHASE 3 — Plot craft and print output

## Task 14: Place mosaic cuts by cumulative mask area

`mask_cut_coords` spaces cuts evenly across slices passing a 25%-of-max area floor, which
at the extremes admits tiles that are mostly neck (`z = -50`), eyeball (`y = +71`) and
brain edge (`x = ±55`).

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/_display.py:178-235`
- Test: `tests/fmri/report/test_display.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `mask_cut_coords(mask_img, direction, n_cuts, *, min_area_fraction=...)` unchanged in signature; placement changes to equal quantiles of cumulative in-plane area.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_display.py
def test_cuts_avoid_a_tapering_end():
    """A mask that tapers to a few voxels must spend no tile in the taper."""
    import numpy as np
    import nibabel as nib
    from fmri_pipeline.analysis.report.figures._display import mask_cut_coords

    data = np.zeros((10, 10, 40), dtype=np.uint8)
    data[2:8, 2:8, 5:35] = 1          # the body
    data[4:6, 4:6, 0:5] = 1           # a thin taper at the low end
    mask = nib.Nifti1Image(data, np.eye(4))

    cuts = mask_cut_coords(mask, "z", 7)
    assert min(cuts) >= 5, f"a cut landed in the taper: {cuts}"


def test_cuts_concentrate_where_there_is_most_brain():
    import numpy as np
    import nibabel as nib
    from fmri_pipeline.analysis.report.figures._display import mask_cut_coords

    data = np.zeros((10, 10, 30), dtype=np.uint8)
    data[1:9, 1:9, 20:28] = 1         # dense block, 8 slices
    data[4:6, 4:6, 0:20] = 1          # sparse column, 20 slices
    mask = nib.Nifti1Image(data, np.eye(4))

    cuts = mask_cut_coords(mask, "z", 6)
    assert sum(c >= 20 for c in cuts) >= 3


def test_cut_count_is_honoured():
    import numpy as np
    import nibabel as nib
    from fmri_pipeline.analysis.report.figures._display import mask_cut_coords

    data = np.zeros((10, 10, 20), dtype=np.uint8)
    data[2:8, 2:8, 2:18] = 1
    cuts = mask_cut_coords(nib.Nifti1Image(data, np.eye(4)), "z", 7)
    assert len(cuts) == 7
    assert len(set(cuts)) == 7
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_display.py -v -k tapering or concentrate`
Expected: FAIL — cuts land in the taper.

- [ ] **Step 3: Implement**

Replace the even-spacing body of `mask_cut_coords` with cumulative-area quantiles:

```python
    axis = _AXIS_OF[direction]
    mask = np.asanyarray(mask_img.dataobj).astype(bool)
    area = mask.sum(axis=tuple(i for i in range(3) if i != axis)).astype(float)
    if not area.any():
        raise ValueError("The mask has no voxels; no cut positions can be chosen.")

    # Quantiles of cumulative area rather than positions passing a floor. Even spacing
    # across a bounding box spends tiles wherever the mask happens to reach, which at
    # the extremes is neck, eye socket and brain edge; area quantiles spend them in
    # proportion to how much mask each slice actually holds, and a taper carrying
    # almost no area therefore receives almost no tiles.
    area[area < min_area_fraction * area.max()] = 0.0
    cumulative = np.cumsum(area)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("No slice carries enough mask area to place a cut.")

    targets = (np.arange(n_cuts) + 0.5) / n_cuts * total
    indices = np.unique(np.searchsorted(cumulative, targets))
    # searchsorted can collapse two targets onto one slice when the mask is thin; fill
    # from the remaining non-empty slices so the caller always gets n_cuts tiles.
    if indices.size < n_cuts:
        candidates = [i for i in np.flatnonzero(area) if i not in set(indices)]
        indices = np.array(sorted(set(indices).union(candidates[: n_cuts - indices.size])))
    indices = indices[:n_cuts]
```

Then map `indices` to world coordinates through the existing affine helper.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_display.py tests/fmri/report/test_mosaic.py tests/fmri/report/test_volumes.py tests/fmri/report/test_coverage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/_display.py tests/fmri/report/test_display.py
git commit -m "fix(report): place mosaic cuts by cumulative mask area

Even spacing across slices passing an area floor spent end tiles on neck,
eye socket and brain edge."
```

---

## Task 15: Make the carpets readable, and unify DVARS units

Both carpets z-score per voxel and clip at ±2.5, which renders structure as grey static.
DVARS is standardized in `motion_coupling.png` (~1.0) and raw in `carpet.png` (~25–30).

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/carpet.py`, `motion.py`
- Test: `tests/fmri/report/test_carpet.py`, `tests/fmri/report/test_motion.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `carpet_figure(..., scaling: str = "percent")` accepting `"percent"` or `"zscore"`; `standardize_dvars(values) -> np.ndarray` in `motion.py`, used by both panels.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_carpet.py
def test_percent_scaling_is_the_default_and_is_centred_on_zero():
    import numpy as np
    from fmri_pipeline.analysis.report.figures.carpet import scale_for_carpet

    rng = np.random.default_rng(0)
    series = 100.0 + rng.standard_normal((50, 200))
    scaled, limit = scale_for_carpet(series, scaling="percent")
    assert abs(float(np.nanmedian(scaled))) < 0.1
    assert limit > 0


def test_percent_scaling_preserves_relative_amplitude_across_voxels():
    """Per-voxel z-scoring destroys exactly the contrast a carpet exists to show."""
    import numpy as np
    from fmri_pipeline.analysis.report.figures.carpet import scale_for_carpet

    quiet = 100.0 + 0.1 * np.sin(np.linspace(0, 20, 200))
    loud = 100.0 + 3.0 * np.sin(np.linspace(0, 20, 200))
    scaled, _ = scale_for_carpet(np.vstack([quiet, loud]), scaling="percent")
    assert np.ptp(scaled[1]) > 5 * np.ptp(scaled[0])


def test_zscore_scaling_remains_available():
    import numpy as np
    from fmri_pipeline.analysis.report.figures.carpet import scale_for_carpet

    quiet = 100.0 + 0.1 * np.sin(np.linspace(0, 20, 200))
    loud = 100.0 + 3.0 * np.sin(np.linspace(0, 20, 200))
    scaled, _ = scale_for_carpet(np.vstack([quiet, loud]), scaling="zscore")
    assert np.ptp(scaled[1]) == pytest.approx(np.ptp(scaled[0]), rel=0.05)
```

```python
# append to tests/fmri/report/test_motion.py
def test_dvars_is_standardized_the_same_way_in_both_panels():
    import numpy as np
    from fmri_pipeline.analysis.report.figures.motion import standardize_dvars

    raw = np.array([25.0, 27.0, 30.0, 26.0])
    standardized = standardize_dvars(raw)
    assert abs(float(np.median(standardized)) - 1.0) < 0.05
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_carpet.py tests/fmri/report/test_motion.py -v -k scale_for_carpet or standardize_dvars`
Expected: FAIL — names do not exist.

- [ ] **Step 3: Implement**

```python
# carpet.py
def scale_for_carpet(series, *, scaling: str = "percent"):
    """Return the carpet's display values and its symmetric colour limit.

    ``percent`` expresses each voxel as its deviation from its own temporal mean, in
    per cent of that mean, and takes one colour limit for the whole panel. ``zscore``
    standardises each voxel independently.

    Percent is the default because a carpet exists to show that some voxels move more
    than others: per-voxel standardisation gives every row unit variance by
    construction, so a quiet voxel and a badly corrupted one are drawn identically and
    the panel becomes texture. The cost is that a single high-variance region can set
    the limit for everything, which the robust percentile below bounds.
    """
    import numpy as np

    values = np.asarray(series, dtype=float)
    if scaling == "zscore":
        centre = values.mean(axis=1, keepdims=True)
        spread = values.std(axis=1, keepdims=True)
        spread[spread == 0] = 1.0
        scaled = (values - centre) / spread
        return scaled, 2.5
    if scaling != "percent":
        raise ValueError(f"scaling must be 'percent' or 'zscore', got {scaling!r}.")

    centre = values.mean(axis=1, keepdims=True)
    safe = np.where(centre == 0, 1.0, centre)
    scaled = 100.0 * (values - centre) / safe
    limit = float(np.nanpercentile(np.abs(scaled), COLOR_LIMIT_PERCENTILE))
    return scaled, (limit if limit > 0 else 1.0)
```

```python
# motion.py
def standardize_dvars(values):
    """DVARS divided by its own median, so 1.0 is the run's typical frame.

    Both panels that show DVARS must use this. Raw DVARS is in image intensity units,
    which differ by scaling choice and by run, so a raw axis in one panel and a
    standardized axis in another invites the reader to compare two different
    quantities that share a name.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    centre = float(np.median(finite)) if finite.size else 0.0
    if centre == 0:
        return array
    return array / centre
```

Route the carpet's DVARS trace through `standardize_dvars`, label the axis
`std DVARS`, and default `carpet_figure(scaling="percent")`. Update both captions to
name the units.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_carpet.py tests/fmri/report/test_motion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/carpet.py fmri_pipeline/analysis/report/figures/motion.py tests/fmri/report/test_carpet.py tests/fmri/report/test_motion.py
git commit -m "fix(report): scale carpets in percent signal change and standardize DVARS everywhere"
```

---

## Task 16: Give the tissue panel a null to compare against

CSF survives at 21.6% against GM's 20.0%. Presented as three bars with no reference, the
ordering is invisible as an anomaly.

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/tissue.py`
- Test: `tests/fmri/report/test_tissue.py`

**Interfaces:**
- Consumes: the existing per-class z values and the applied threshold.
- Produces: `tissue_survival(values_by_class, *, threshold, two_sided, fitted_null=None) -> List[dict]` with keys `label`, `n`, `survivors`, `rate`, `expected_rate`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_tissue.py
def test_expected_rate_uses_the_fitted_null_when_supplied():
    import numpy as np
    from fmri_pipeline.analysis.report.figures.tissue import tissue_survival
    from fmri_pipeline.analysis.report.inference import EmpiricalNull

    rng = np.random.default_rng(0)
    values = {"GM": rng.standard_normal(20000) * 1.5 - 0.6}
    rows = tissue_survival(
        values, threshold=2.3, two_sided=True,
        fitted_null=EmpiricalNull(centre=-0.6, scale=1.5, n=20000),
    )
    # Under this null the observed and expected rates should nearly agree.
    assert abs(rows[0]["rate"] - rows[0]["expected_rate"]) < 2.0


def test_expected_rate_is_shared_across_classes():
    """The null does not know about tissue, so one rate applies to all three."""
    import numpy as np
    from fmri_pipeline.analysis.report.figures.tissue import tissue_survival
    from fmri_pipeline.analysis.report.inference import EmpiricalNull

    rng = np.random.default_rng(0)
    values = {
        "GM": rng.standard_normal(5000),
        "WM": rng.standard_normal(5000),
        "CSF": rng.standard_normal(5000),
    }
    rows = tissue_survival(
        values, threshold=2.3, two_sided=True,
        fitted_null=EmpiricalNull(centre=0.0, scale=1.0, n=15000),
    )
    assert len({round(r["expected_rate"], 6) for r in rows}) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_tissue.py -v -k expected_rate`
Expected: FAIL — `tissue_survival() got an unexpected keyword argument 'fitted_null'`.

- [ ] **Step 3: Implement**

Add the `fitted_null` keyword; compute `expected_rate` as the tail mass of the fitted
null beyond the threshold, times 100. Draw it as a horizontal reference line across all
three bars, labelled "expected under the map's fitted null". Extend the caption:

> The reference line is the survival rate the map's own fitted null predicts for any
> class, since a null knows nothing about tissue. A class above it survives more than
> chance under that null and a class below it survives less; how much grey-matter
> enrichment to expect depends on the contrast and on the segmentation's accuracy at
> this resolution, so the rates are reported without a criterion attached.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_tissue.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/tissue.py tests/fmri/report/test_tissue.py
git commit -m "feat(report): draw the fitted null's expected survival rate on the tissue panel"
```

---

## Task 17: Close the mosaic colorbar gap and thin the caption boilerplate

Two remaining Phase 3D items. Mosaics leave a wide dead band between the last tile and
the colorbar. Separately, "no criterion is applied" and near-variants appear about
twelve times across the report; the principle is right, but at that density it crowds
out the observed-against-expected comparisons, which are measurements and belong in the
foreground.

**Files:**
- Modify: `fmri_pipeline/analysis/report/figures/_mosaic.py:152-220`
- Modify: `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_mosaic.py`, `tests/fmri/report/test_subject.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `draw_colorbar(..., gap_fraction: float = 0.015)`; `SECTION_NOTE: dict[str, str]` in `subject.py` mapping section id to its single no-criterion statement.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_mosaic.py
def test_colorbar_sits_close_to_the_last_tile():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import nibabel as nib
    from fmri_pipeline.analysis.report.figures._mosaic import mosaic_figure

    data = np.zeros((12, 12, 12), dtype=np.float32)
    data[3:9, 3:9, 3:9] = 1.0
    image = nib.Nifti1Image(data, np.eye(4))
    figure = mosaic_figure(image, mask_img=image, n_cuts=5)

    tiles = [a for a in figure.get_axes() if a.get_label() != "colorbar"]
    bars = [a for a in figure.get_axes() if a.get_label() == "colorbar"]
    assert bars, "the mosaic drew no colorbar"
    rightmost = max(a.get_position().x1 for a in tiles)
    assert bars[0].get_position().x0 - rightmost < 0.06
```

```python
# append to tests/fmri/report/test_subject.py
def test_no_criterion_note_appears_once_per_section():
    """Repeated twelve times it stops being read; once per section it still is."""
    from fmri_pipeline.analysis.report.subject import SECTION_NOTE

    assert set(SECTION_NOTE) >= {"qc", "contrast", "design", "diagnostics"}
    for text in SECTION_NOTE.values():
        assert text.strip()


def test_panel_captions_do_not_each_repeat_the_note(report_html):
    """`report_html` is the existing fixture rendering a full report to a string."""
    assert report_html.lower().count("no criterion is applied") <= 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_mosaic.py tests/fmri/report/test_subject.py -v -k colorbar or criterion`
Expected: FAIL — the colorbar gap exceeds 0.06, and the note appears about twelve times.

- [ ] **Step 3: Implement**

In `_mosaic.py`'s `draw_colorbar`, compute the colorbar axes rectangle from the
rightmost tile's `x1` plus `gap_fraction` rather than from a fixed figure fraction, and
shrink the figure's right margin to match.

In `subject.py`, add:

```python
#: One statement of the no-criterion principle per section, rather than per panel.
#:
#: The principle is that this report measures and does not score. Twelve repetitions
#: of it stopped conveying that and started displacing the numbers -- in particular the
#: observed-against-expected pairs, which are measurements and are the reason the
#: principle is affordable in the first place.
SECTION_NOTE = {
    "qc": "Reference levels drawn in this section are published conventions. "
          "No run is scored against them here.",
    "contrast": "Counts are stated against the map's own fitted null wherever both "
                "are defined. No voxel is scored against a criterion.",
    "design": "Efficiency, condition number and variance inflation are comparable "
              "between designs for the same contrast and meaningless as absolute "
              "numbers, so no cutoff is applied to any of them.",
    "diagnostics": "Computed inside the fitted analysis mask and reported as "
                   "distributions across voxels. No criterion is applied.",
}
```

Render it once beneath each section heading and strip the per-panel repetitions,
keeping every caption's *substantive* content. Do not remove a clause that explains what
a panel measures — only the repeated no-criterion sentence itself.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_mosaic.py tests/fmri/report/test_subject.py tests/fmri/report/test_volumes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/figures/_mosaic.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_mosaic.py tests/fmri/report/test_subject.py
git commit -m "fix(report): close the mosaic colorbar gap, state the no-criterion note once per section"
```

---

## Task 18: Print profile

**Files:**
- Create: `fmri_pipeline/analysis/report/print_profile.py`
- Create: `tests/fmri/report/test_print_profile.py`
- Modify: `fmri_pipeline/analysis/plotting_config.py:7`, `fmri_pipeline/analysis/report/style.py`

**Interfaces:**
- Consumes: `FMRI_RC`, `PRINT_FIGURE_DPI`, `savefig_kwargs` from `style.py`.
- Produces:
  - `PRINT_WIDTHS_MM: dict[str, float]` — figure stem to target width, default 170.
  - `print_context(width_mm: float)` — a context manager layering print rc over `FMRI_RC`.
  - `rasterize_images(figure) -> int` — marks image artists rasterized, returns how many.

- [ ] **Step 1: Write the failing test**

```python
# tests/fmri/report/test_print_profile.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fmri_pipeline.analysis.report import print_profile


def test_pdf_is_an_allowed_format():
    from fmri_pipeline.analysis.plotting_config import FmriReportConfig

    cfg = FmriReportConfig(enabled=True, formats=("pdf",))
    cfg.validate()


def test_print_context_raises_the_dpi():
    from fmri_pipeline.analysis.report.style import PRINT_FIGURE_DPI

    with print_profile.print_context(170.0):
        assert plt.rcParams["savefig.dpi"] == PRINT_FIGURE_DPI


def test_rasterize_marks_images_but_not_text():
    figure, axis = plt.subplots()
    axis.imshow(np.random.default_rng(0).standard_normal((8, 8)))
    axis.set_title("kept as text")
    count = print_profile.rasterize_images(figure)
    assert count == 1
    assert axis.get_images()[0].get_rasterized() is True
    assert axis.title.get_rasterized() is False


def test_text_survives_into_the_pdf(tmp_path):
    """Rasterizing the image layer must not flatten the annotation with it."""
    figure, axis = plt.subplots()
    axis.imshow(np.random.default_rng(0).standard_normal((8, 8)))
    axis.set_title("SELECTABLE")
    print_profile.rasterize_images(figure)
    path = tmp_path / "f.pdf"
    figure.savefig(path)
    assert b"SELECTABLE" in path.read_bytes()


def test_default_width_is_double_column():
    assert print_profile.width_for("dual_coded") == 170.0
    assert print_profile.width_for("motion_by_run") == 85.0
    assert print_profile.width_for("an_unregistered_figure") == 170.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_print_profile.py -v`
Expected: FAIL — module does not exist; `FmriReportConfig` rejects `pdf`.

- [ ] **Step 3: Implement**

Add `"pdf"` to `_ALLOWED_FORMATS` in `plotting_config.py:7`. Extend `savefig_kwargs` in
`style.py` with a `.pdf` branch returning `{"bbox_inches": "tight", "metadata": {"CreationDate": None}}`
so PDFs are byte-reproducible like the other two.

Create `print_profile.py`:

```python
"""Rendering a figure for a manuscript rather than for the screen.

One figure specification, two render profiles. The screen report stays at
``HTML_FIGURE_DPI`` and its own type sizes; this module supplies the rc layer, the
physical width, and the rasterization policy a journal needs, without any figure
module knowing which profile it is being drawn under.
"""

MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 85.0
DOUBLE_COLUMN_MM = 170.0

#: Target print width per figure stem. Anything unlisted is double column, which is
#: the safe default: a figure authored wide and reduced stays legible, while one
#: authored narrow and enlarged does not.
PRINT_WIDTHS_MM: dict[str, float] = {
    "motion_by_run": SINGLE_COLUMN_MM,
    "motion_coupling": SINGLE_COLUMN_MM,
    "threshold_calibration": SINGLE_COLUMN_MM,
    "tissue_distribution": SINGLE_COLUMN_MM,
    "run_influence": SINGLE_COLUMN_MM,
    "sign_flip_null": SINGLE_COLUMN_MM,
    "run_offsets": SINGLE_COLUMN_MM,
}


def width_for(stem: str) -> float:
    return PRINT_WIDTHS_MM.get(str(stem), DOUBLE_COLUMN_MM)
```

with `print_context(width_mm)` layering `{"savefig.dpi": PRINT_FIGURE_DPI, "figure.dpi": PRINT_FIGURE_DPI, "font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6.5}`
over `FMRI_RC` via `plt.rc_context`, and:

```python
def rasterize_images(figure) -> int:
    """Mark image artists for rasterization, leaving everything else vector.

    A brain mosaic or a carpet has tens of thousands of cells; emitting them as vector
    paths produces a PDF a typesetter cannot open. Rasterizing the whole figure instead
    would flatten the axis labels and annotation with them, which is what makes a
    figure unusable at print size. Only ``AxesImage`` and ``QuadMesh`` are marked.
    """
    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage

    count = 0
    for artist in figure.findobj():
        if isinstance(artist, (AxesImage, QuadMesh)):
            artist.set_rasterized(True)
            count += 1
    return count
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_print_profile.py tests/fmri/report/test_configuration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/print_profile.py fmri_pipeline/analysis/plotting_config.py fmri_pipeline/analysis/report/style.py tests/fmri/report/test_print_profile.py
git commit -m "feat(report): add a print render profile with PDF output"
```

---

## Task 19: Emit print figures beside the screen report

**Files:**
- Modify: `fmri_pipeline/analysis/report/assets.py`, `fmri_pipeline/analysis/report/subject.py`
- Test: `tests/fmri/report/test_print_profile.py`

**Interfaces:**
- Consumes: `print_context`, `rasterize_images`, `width_for` (Task 18).
- Produces: `save_for_print(figure, *, out_dir, stem) -> Optional[Path]`, writing `plots/print/<stem>.pdf`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_print_profile.py
def test_print_figures_land_in_their_own_directory(tmp_path):
    import matplotlib.pyplot as plt
    from fmri_pipeline.analysis.report.assets import save_for_print

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    path = save_for_print(figure, out_dir=tmp_path, stem="motion_by_run")
    assert path == tmp_path / "print" / "motion_by_run.pdf"
    assert path.exists()


def test_print_figure_is_sized_in_millimetres(tmp_path):
    import matplotlib.pyplot as plt
    from fmri_pipeline.analysis.report.assets import save_for_print
    from fmri_pipeline.analysis.report.print_profile import MM_PER_INCH

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    save_for_print(figure, out_dir=tmp_path, stem="motion_by_run")
    assert figure.get_figwidth() == pytest.approx(85.0 / MM_PER_INCH, rel=1e-3)


def test_provenance_strip_is_absent_from_print_output(tmp_path):
    """A working-artifact device, not a manuscript one."""
    import matplotlib.pyplot as plt
    from fmri_pipeline.analysis.report.assets import save_for_print

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.text(0.0, -0.05, "n = 50,626 voxels", fontsize=6.5)
    path = save_for_print(figure, out_dir=tmp_path, stem="motion_by_run")
    assert b"50,626" not in path.read_bytes()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_print_profile.py -v -k print_figures or millimetres or provenance`
Expected: FAIL — `cannot import name 'save_for_print'`.

- [ ] **Step 3: Implement**

```python
def save_for_print(figure, *, out_dir: Path, stem: str) -> Optional[Path]:
    """Write ``figure`` as a manuscript PDF beside the screen assets.

    Resizes to the registered physical width preserving aspect, rasterizes only the
    image layer, and removes the provenance strip.

    The strip and the tight bounding box are removed together deliberately.
    ``savefig_kwargs`` passes ``bbox_inches="tight"`` precisely so the strip drawn
    below the canvas is included; dropping the strip while keeping the tight box would
    leave the figure cropped to a caption that is no longer there.
    """
    from fmri_pipeline.analysis.report.print_profile import (
        MM_PER_INCH, print_context, rasterize_images, width_for,
    )

    width_mm = width_for(stem)
    target = out_dir / "print" / f"{stem}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    for text in list(figure.texts):
        if text.get_position()[1] < 0:
            text.remove()

    width_in = width_mm / MM_PER_INCH
    aspect = figure.get_figheight() / figure.get_figwidth()
    figure.set_size_inches(width_in, width_in * aspect)
    rasterize_images(figure)

    try:
        with print_context(width_mm):
            figure.savefig(target, bbox_inches=None, metadata={"CreationDate": None})
    except Exception as exc:
        logger.warning("Could not write the print figure %s (%s)", target.name, exc)
        return None
    return target
```

Call it from `_save` in `subject.py` when `"pdf"` is among the configured formats.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_print_profile.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/assets.py fmri_pipeline/analysis/report/subject.py tests/fmri/report/test_print_profile.py
git commit -m "feat(report): write manuscript PDFs to plots/print"
```

---

## Task 20: Cohort-ready summary row

**Files:**
- Modify: `fmri_pipeline/analysis/report/manifest.py`
- Test: `tests/fmri/report/test_manifest_writing.py`

**Interfaces:**
- Consumes: every manifest field added above.
- Produces: `contrast_summary_row(manifest) -> dict[str, object]` with a fixed key set, and `write_contrast_summary(manifest, *, out_dir) -> Path` emitting `<stem>_desc-summary.tsv`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/fmri/report/test_manifest_writing.py
EXPECTED_SUMMARY_KEYS = [
    "subject", "task", "contrast_name", "n_runs",
    "applied_threshold", "applied_survivors", "n_voxels",
    "fitted_null_centre", "fitted_null_scale", "expected_under_fitted_null",
    "sign_flip_fwe_height", "sign_flip_fwe_survivors",
    "sign_flip_global_p", "sign_flip_p_floor",
    "loro_max_abs_delta", "loro_most_influential_run",
    "global_offset_mean_beta", "contrast_vif_max", "event_count_min_per_run",
]


def test_summary_row_has_a_fixed_key_set(tmp_path):
    """The cohort report stacks these rows; a varying key set makes that impossible."""
    from fmri_pipeline.analysis.report.manifest import (
        contrast_summary_row, read_report_manifest, write_report_manifest,
    )

    path = write_report_manifest(
        contrast_dir=tmp_path, subject="sub-01", task="x", contrast_name="c"
    )
    row = contrast_summary_row(read_report_manifest(path))
    assert list(row.keys()) == EXPECTED_SUMMARY_KEYS


def test_absent_measurements_are_none_not_zero(tmp_path):
    """A subject with no sign-flip null must not read as a height of zero."""
    from fmri_pipeline.analysis.report.manifest import (
        contrast_summary_row, read_report_manifest, write_report_manifest,
    )

    path = write_report_manifest(
        contrast_dir=tmp_path, subject="sub-01", task="x", contrast_name="c"
    )
    row = contrast_summary_row(read_report_manifest(path))
    assert row["sign_flip_fwe_height"] is None
    assert row["loro_max_abs_delta"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_manifest_writing.py -v -k summary_row or absent_measurements`
Expected: FAIL — `cannot import name 'contrast_summary_row'`.

- [ ] **Step 3: Implement**

Add `contrast_summary_row` returning the keys above in that exact order, reading the
influence TSV for `loro_max_abs_delta` and `loro_most_influential_run` (the row with the
largest `abs(delta)`), and `write_contrast_summary` writing a one-row TSV. Absent
measurements are `None`, never `0`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fmri/report/test_manifest_writing.py tests/fmri/report/test_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fmri_pipeline/analysis/report/manifest.py tests/fmri/report/test_manifest_writing.py
git commit -m "feat(report): emit a fixed-schema per-contrast summary row for cohort reporting"
```

---

## Task 21: Final verification

- [ ] **Step 1: Run the fMRI report suite**

Run: `.venv/bin/python -m pytest tests/fmri/report tests/fmri/test_pooling_rule.py tests/fmri/test_sign_flip_null.py tests/fmri/test_run_influence.py tests/fmri/test_run_level_contrasts.py -v`

Do not run the whole suite. Expected: all PASS.

- [ ] **Step 2: Confirm the report package still fits no GLM**

Run: `.venv/bin/python -m pytest tests/fmri/report -v -k import or invariant`
Expected: PASS.

- [ ] **Step 3: Regenerate the report and compare against Phase 1**

Regenerate to `outputs/fmri_report_final/`. Confirm the panels from Phases 2–3 appear,
the mosaics spend no tile on neck or eye socket, the carpets show structure, and
`plots/print/` holds PDFs whose text is selectable.

- [ ] **Step 4: Commit**

```bash
git commit --allow-empty -m "chore(fmri): verify the report end to end after phases 1-3"
```

---

## Notes for the implementer

**Why the sign-flip null exists.** On sub-0001 the applied |z| > 2.30 yields 8,463
voxels where the map's own fitted null predicts 8,001. Bonferroni and FDR both assume
N(0, 1), which the same page shows to be wrong — the fitted null is N(−0.61, 1.51²). The
sign-flip is the only correction on the page that respects the between-run variance
actually present.

**Why the p is nearly useless and the height is not.** The identity pattern is a member
of the null and always ties the observed maximum, so `p ≥ 2/(2^(n−1)+1)`. Six runs floor
at 0.061. The height has no such limitation. Never print the p bare.

**Do not "fix" the run-03 finding.** Dropping run-03 *raises* the survivor count by
1,965 because it is the only run with a positive whole-mask mean. That is a measurement
the report should state, not a defect to correct. The same applies to the per-run event
imbalance and the global offset — all three are analysis findings surfaced by reporting,
and none is a reporting bug.
