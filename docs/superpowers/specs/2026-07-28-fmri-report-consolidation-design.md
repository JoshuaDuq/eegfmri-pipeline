# fMRI report consolidation and subject-report completion

Date: 2026-07-28

Addendum to `2026-07-28-fmri-post-preprocessing-report-design.md`. That spec's problem
statement, figure inventory, colour policy, and correctness table stand unchanged. This
document covers only what that spec could not anticipate: the work was started three times
in parallel, and the three results must become one before anything further is built.

## Problem

Three implementations of the same design exist simultaneously.

| Where | State |
|---|---|
| `main` at HEAD (`401c26bf`) | Pre-work: the original monolithic `reporting.py`, plus the spec and plans as documentation. |
| `feat/fmri-subject-report` (worktree `.worktrees/fmri-plotting-foundation`) | 13 commits. Plan 1 complete; plan 2 tasks 1–3 complete; task 4 (`subject.py`, 792 lines) uncommitted. 14 test files, 165 tests passing in 29 s. |
| `main`'s working tree (uncommitted) | An independent in-place hardening of `reporting.py` (+1881/−612, now 2874 lines) that consumes a divergent private copy of `report/style.py` and `report/figures/design.py`. |

The two efforts fix substantially the same defects by opposite strategies: the branch
decomposes `reporting.py` into `report/figures/` as the spec directs, while main's working
tree keeps the monolith and repairs it in place. Neither is a superset of the other.

Left alone this produces two modules named `fmri_pipeline.analysis.report.style` with
different contents depending on which tree is checked out, and a `reporting.py` whose
structure depends on which effort last touched it.

## Decisions taken

**All work happens in the main checkout on the `main` branch.** No worktrees. The parallel
worktree is what produced the divergence, and the repository already carries roughly fifteen
stale worktrees from earlier features. The branch's commits are brought into main and the
worktree is retired.

**The branch's structure wins; main's better implementations win where they are better.**
Structure and implementation are decided separately, because the two efforts did not
distribute their quality evenly. Deciding the whole conflict on one axis would discard real
work in either direction.

**Nothing uncommitted is discarded without being captured first.** Main's working tree
carries unrelated in-flight EEG work across roughly fifteen files alongside the fMRI
changes. The consolidation begins with a `git stash create` snapshot captured under a tag,
which records the full working tree without modifying it.

**The resting-state profile is deferred.** It remains specified in the parent document and
unbuilt. This pass ends with the task profile complete and verified.

**Verification renders real data.** No figure this pipeline produces has ever been looked at
against real BOLD data — `/Volumes/KINGSTON/EEG_fMRI_data/derivatives` holds fMRIPrep output
and reports for roughly fifteen subjects, and no analysis-stage report among them. A
consolidation justified by figure quality that never renders a figure is not verified.

## Conflict resolution

Of the branch's 29 changed files, 26 are new — the `report/` package and its tests — and
land without conflict. Main's uncommitted fMRI work touches mostly disjoint files and is
retained in full:

| File | Retained addition |
|---|---|
| `pipelines/fmri_analysis.py` | `_write_skull_stripped_background`, `_discover_tissue_segmentation` |
| `utils/bold_discovery.py` | `prepare_confounds_for_first_level_model` |
| `analysis/contrast_builder.py` | `_coerce_optional_float` and surrounding changes |
| `analysis/trial_signatures.py` | net reduction, unrelated to reporting |

The skull-stripped background is a figure improvement in its own right: overlay panels drawn
against a non-stripped background carry skull and neck tissue that competes with the
overlay.

Three files genuinely collide.

| File | Resolution | Reason |
|---|---|---|
| `analysis/reporting.py` | Branch | Delegating and decomposed against a 2874-line monolith. Decomposition is the parent spec's central structural decision and the precondition for testing panels without rendering a report. |
| `report/style.py` | Merge | Neither is a superset. |
| `report/figures/design.py` | Main | Strictly richer; see below. |

### style.py merge

From the branch: the measured rejection of matplotlib 3.10's Crameri diverging maps
(midpoint CIE L\* — berlin 4.4, vanimo 7.1, managua 24.0, against RdBu_r's 97.1, all
unusable on a white canvas where the midpoint would become the heaviest ink on the page),
`GUIDE_COLOR`, and `COLOR_LIMIT_PERCENTILE`.

From main: `RADIOLOGICAL` and `ORIENTATION_LABEL` as module constants rather than a
per-module `_orientation_label` helper duplicated in `stat_maps.py`; `panel_label` for
journal-style panel letters; `SEQUENTIAL_DECISION_CMAP` for the deferred rest profile;
`colour_limit_note`; and limit helpers that accept a `mask` and nibabel images directly
rather than pre-extracted arrays.

Two conflicts inside the merge are resolved explicitly:

- **Percentile.** The branch uses 98.0, main 99.0. Take the branch's 98.0. It is the more
  conservative limit, and the clipped fraction is stated on every figure either way.
- **DPI.** The branch sets `savefig.dpi: 300`; main sets 150, on the ground that 300
  embedded base64 megabytes into every report at a layout width of roughly 1180 px. Take
  150 for figures embedded in HTML and retain 300 for figures written to disk for
  manuscript use. Main's figure is a measurement of a rendered report; the branch's is a
  default.

The branch's empty-input behaviour is kept: limit helpers raise on no finite values rather
than returning `None`. A colour limit that cannot be computed is a broken panel, and the
established failure policy is that the caller renders a placeholder naming the exception.
Returning `None` pushes the same decision into every call site.

### design.py

Main's implementation supersedes the branch's. It adds `classify_regressors`, which orders
columns by role and reports each role's span — the task/confound/drift grouping the parent
spec asked for and the branch does not implement; `contrast_efficiency`
(`1 / (cᵀ (XᵀX)⁻¹ c)`), which is what tells a reader whether a contrast is estimable at all;
`summarize_design`; and per-column display scaling, without which confound regressors on
unrelated scales render as uniform bands. It also separates `regressor_correlation_figure`
from `variance_inflation_figure`, where the branch draws one combined `collinearity_figure`.

The branch's `tests/fmri/report/test_design.py` and main's `test_design_figures.py` are
merged into one file covering the retained implementation.

### Fixes harvested from main's `reporting.py`

Three corrections exist only in main's monolith and are ported into the branch's structure.
Each gets a regression test.

- **`_detrended_temporal_sd` → `figures/volumes.py`.** The branch censors non-steady-state
  frames before computing temporal standard deviation, which the parent spec identified. It
  does not remove scanner drift, which inflates the same statistic for a different reason,
  so tSNR is still reported low. Main projects out a low-order polynomial basis first. This
  corrects a reported numeric value, not an appearance.
- **`_supports_glass_brain` → `subject.py`.** `subject.py:484` draws the glass brain
  unconditionally. The projection is defined only against the MNI schematic, so a
  native-space map rendered on it is wrong rather than approximate. Guarded by space, with
  the panel omitted and the omission stated when the space does not support it.
- **`_label_cluster_coordinate_space` and `_coordinate_space_caption`.** The branch's
  cluster tables carry no coordinate-space label, which lets native-space coordinates read
  as MNI. Ported.

## Remaining implementation

Plan 2 (`2026-07-28-fmri-subject-report.md`) tasks 4–8, unchanged by this addendum: shared
sections; per-contrast results sections; numbered cluster peaks; the signature dot plot and
design section; and wiring the entry point while cutting the pipeline's call into plotting.

Task 4 has an uncommitted 792-line `subject.py` and a 248-line test file to be brought in
with the rest of the branch and completed rather than restarted.

## Verification

Consolidation is verified by the existing suite: `tests/fmri/report/` passes in the
consolidated main checkout. The branch's 165 passing tests are the floor for coverage, not a
count to match — merging `design.py` replaces the branch's design tests with main's, so the
total may move in either direction. What must hold is that every behaviour asserted on
either side is still asserted after the merge, plus one new regression test per harvested
fix.

Figure quality is verified by rendering. One subject with complete task derivatives is
selected from `/Volumes/KINGSTON/EEG_fMRI_data/derivatives`, read-only, and a report is
rendered into the repository's `outputs/` directory. Every panel is inspected. The subject
used is named in the result, and any defect the rendering exposes is fixed before this pass
is called complete.

Per project convention, verification runs targeted subsets; the full suite takes roughly
nine minutes and is not run.

## Out of scope

- The resting-state profile, including the `_validate_roi_timeseries` degeneracy change.
  Specified in the parent document, deferred to a later pass.
- The cohort/group report.
- Retiring the fifteen unrelated stale worktrees.
- Any change to GLM estimation, contrast construction, or confound selection.
