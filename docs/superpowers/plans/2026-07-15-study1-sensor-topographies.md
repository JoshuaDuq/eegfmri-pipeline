# Study 1 Sensor Topographies Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build standalone, publication-grade Study 1 sensor topographies for temperature/intensity and NPS/SIIPS1 with joint ten-map cluster-permutation inference and complete audits.

**Architecture:** A strict shared loader reconstructs channel-level dB power, focused estimand modules produce participant maps, and an analysis-only cluster engine operates on complete tensors. A renderer consumes immutable summaries, while two thin writers atomically publish SVG/PNG/table/caption/manifest families.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, MNE-Python, Matplotlib, PyArrow, pytest.

---

## File map

- Create `studies/pain_study/study1/figures/sensor_topography_data.py`: strict trial-level channel-power reconstruction and montage resolution.
- Create `studies/pain_study/study1/figures/sensor_topography_estimands.py`: construct and signature participant estimands and exclusion audits.
- Create `studies/pain_study/study1/figures/sensor_cluster_inference.py`: Delaunay adjacency, complete-tensor validation, sign flips, clusters, and audit frames.
- Create `studies/pain_study/study1/figures/sensor_topography_plot.py`: shared 2-by-5 rendering only.
- Create `studies/pain_study/study1/figures/sensor_topography_outputs.py`: shared output paths, schemas, atomic staging, captions, and manifests.
- Create `studies/pain_study/study1/figures/plot_sensor_power_topographies.py`: construct figure CLI/writer.
- Create `studies/pain_study/study1/figures/plot_signature_power_topographies.py`: signature figure CLI/writer.
- Modify `studies/pain_study/study1/config/study1_figure_config.yaml`: explicit model, inference, layout, color, and PNG settings.
- Modify `studies/pain_study/study1/figures/validity_style.py`: deterministic PNG saving without changing SVG behavior.
- Create focused tests under `studies/tests/pipelines/` and update Study 1 README/run guide.

### Task 1: Configuration and publication PNG support

**Files:**
- Modify: `studies/pain_study/study1/config/study1_figure_config.yaml`
- Modify: `studies/pain_study/study1/figures/validity_style.py`
- Test: `studies/tests/pipelines/test_study1_sensor_topography_config.py`

- [ ] Write failing tests requiring `study1.figures.sensor_topographies` to expose exact bands, 183-by-86 mm dimensions, 600 DPI, signature-model tolerances, Delaunay inference values, and seed `20260715`.
- [ ] Write a failing deterministic-PNG test that rejects non-PNG suffixes and DPI below 300.
- [ ] Run `python -m pytest studies/tests/pipelines/test_study1_sensor_topography_config.py -q`; expect missing configuration/helper failures.
- [ ] Add only the approved YAML keys and `save_publication_png(figure, output_path, config, *, dimensions_mm, dpi)` using atomic temporary-file replacement. Do not alter existing SVG output.
- [ ] Rerun the focused test; expect pass.
- [ ] Commit with `feat: configure Study 1 sensor topographies`.

### Task 2: Strict channel-power data boundary

**Files:**
- Create: `studies/pain_study/study1/figures/sensor_topography_data.py`
- Test: `studies/tests/pipelines/test_study1_sensor_topography_data.py`

- [ ] Write failing tests for `reconstruct_channel_power`, proving that a channel log-ratio of `0.2` becomes `2.0` dB and that band/channel/trial ordering is deterministic.
- [ ] Add failing tests for duplicate trial IDs, missing baseline/log-ratio pairs, cross-band channel disagreement, nonfinite values, nonpositive baseline/reconstructed power, and trial-target misalignment.
- [ ] Add failing tests proving malformed power feature names, unexpected extra band columns, unsupported channel-power statistics, and otherwise invalid power columns are rejected rather than ignored by `NamingSchema` filtering.
- [ ] Add failing tests for `load_sensor_montage`: exact `preprocessing.montage`, analyzed EEG channels only, unique finite 3D and x-y positions, and input-order preservation.
- [ ] Run the focused file; expect import failure.
- [ ] Implement immutable `SensorPowerData` and small helpers reusing `NamingSchema`, `retained_event_trials`, `load_power_feature_tables`, and the prepared target table. Return tidy columns `subject_id`, `trial_id`, `band`, `channel`, `power_db` plus required event/target/nuisance fields.
- [ ] Run the focused file; expect pass.
- [ ] Commit with `feat: reconstruct Study 1 sensor power`.

### Task 3: Participant estimands and exclusions

**Files:**
- Create: `studies/pain_study/study1/figures/sensor_topography_estimands.py`
- Test: `studies/tests/pipelines/test_study1_sensor_topography_estimands.py`

- [ ] Write a hand-calculated temperature test for condition-equal dB-per-degree slopes at two sensors.
- [ ] Write a hand-calculated intensity partial-correlation test using the existing power-validity nuisance contract and Fisher-z output.
- [ ] Write NPS/SIIPS1 tests asserting ordered target-specific nuisance resolution, within-participant constant-column recording, standardization, rank/condition/residual checks, and mandatory NPS adjustment for SIIPS1.
- [ ] Prove correlation estimands retain both `partial_r` and `fisher_z`, that inference tensors use `fisher_z`, and that cohort display values use `tanh(mean_fisher_z)` rather than raw-r averaging or displayed z values.
- [ ] Write tests showing one failed cell excludes the whole participant from that figure family while malformed source data raises immediately.
- [ ] Write channel-scope tests: construct primary/complementary Fp1/Fp2 states and signature `feature_benchmark.excluded_channels` removal.
- [ ] Run the focused file; expect import failure.
- [ ] Implement `build_construct_effects` and `build_signature_effects`, returning `ParticipantEffects` with tidy effects, exclusions, sensitivity effects, map order, sensor order, and participant order. Keep input validation separate from participant non-estimability.
- [ ] Rerun the focused file; expect pass.
- [ ] Commit with `feat: estimate Study 1 sensor effects`.

### Task 4: Joint sensor-cluster inference

**Files:**
- Create: `studies/pain_study/study1/figures/sensor_cluster_inference.py`
- Test: `studies/tests/pipelines/test_study1_sensor_cluster_inference.py`

- [ ] Write failing tests constructing Delaunay adjacency from known x-y points, asserting triangle-edge connectivity, diagonal entries, channel order, and fatal duplicate/disconnected geometry.
- [ ] Write failing tests for positive and negative same-sign clusters and mass `sum(abs(t))`.
- [ ] Assert the configured two-sided cluster-forming probability is converted exactly to `scipy.stats.t.ppf(1 - p / 2, df)` and its negative, preventing a one-sided threshold implementation.
- [ ] Write an enumerable six-participant reference test proving synchronized signs across the entire map tensor, maximum mass across all ten maps, exact canonical pattern count, and exact corrected p-values.
- [ ] Write a sampled-path test proving unique non-observed canonical patterns, deterministic seed behavior, plus-one p-values, and audit counts.
- [ ] Write failure tests for incomplete tensors, invalid settings, and sign spaces unable to resolve family alpha.
- [ ] Run the focused file; expect import failure.
- [ ] Implement frozen `SensorClusterResult`, `SensorMapResult`, and `ClusterResult` records plus pure helpers for t statistics, thresholding, graph components, canonical signs, exact/sample selection, and family audit frames.
- [ ] Rerun the focused file; expect pass.
- [ ] Commit with `feat: add joint sensor cluster inference`.

### Task 5: Scientific topomap renderer

**Files:**
- Create: `studies/pain_study/study1/figures/sensor_topography_plot.py`
- Test: `studies/tests/pipelines/test_study1_sensor_topography_plot.py`

- [ ] Write a failing construct-render test requiring 10 topomap axes in row-major order, two row-specific colorbars, shared symmetric no-clipping limits, band labels, estimand labels, and actual participant count.
- [ ] Write a failing signature-render test requiring one shared symmetric partial-r scale across all ten maps.
- [ ] Monkeypatch `mne.viz.plot_topomap` and assert only corrected cluster sensors receive dark-ring masks; maps with no surviving cluster receive no mask.
- [ ] Run the focused file; expect import failure.
- [ ] Implement `build_sensor_topography_figure(summary, config)` with no artifact reads or statistics. Use actual sensor x-y positions, a zero-centered diverging map, no significance interpolation, and publication style.
- [ ] Rerun the focused file; expect pass.
- [ ] Commit with `feat: render Study 1 sensor topographies`.

### Task 6: Atomic publication families and standalone CLIs

**Files:**
- Create: `studies/pain_study/study1/figures/sensor_topography_outputs.py`
- Create: `studies/pain_study/study1/figures/plot_sensor_power_topographies.py`
- Create: `studies/pain_study/study1/figures/plot_signature_power_topographies.py`
- Test: `studies/tests/pipelines/test_study1_sensor_topography_writers.py`

- [ ] Write failing writer tests for exact output families: SVG, PNG, participant TSV/Parquet, sensor TSV/Parquet, cluster TSV, family TSV, caption, manifest, and construct sensitivity TSV/Parquet.
- [ ] Assert manifests contain source/output SHA-256 values, configuration, participant exclusions/order, sensor order, montage/adjacency, software versions, requested/actual sign counts, and observed-label inclusion.
- [ ] Add failure-injection tests for both staging and multi-file promotion. Promotion failure must trigger rollback and leave every prior destination file byte-for-byte unchanged with no partial new family.
- [ ] Add CLI tests for config, Study 1 config, derivative root, task, output path, and printed SVG path.
- [ ] Add one end-to-end CLI test per figure using synthetic prepared-target, clean-event, and prepared-feature roots; do not monkeypatch the loader, estimands, inference, renderer, or publisher.
- [ ] Run the focused file; expect import failure.
- [ ] Implement shared immutable path records and atomic promotion. Construct sensitivity must publish distinct participant-effect and cohort-summary TSV/Parquet pairs. Writers must orchestrate data, estimands, inference, rendering, schema validation, caption, and manifest without duplicating scientific logic. Manifest output checksums cover every other family member and explicitly exclude the manifest itself.
- [ ] Rerun the focused file; expect pass.
- [ ] Commit with `feat: publish Study 1 sensor topographies`.

### Task 7: Documentation and end-to-end verification

**Files:**
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`
- Modify: focused tests above only if verification reveals a genuine contract defect.

- [ ] Document both scientific estimands, sensor-space limitations, joint FWER procedure, prerequisites, exact standalone commands, outputs, and distinction from predictive importance/source localization.
- [ ] Run `python -m pytest studies/tests/pipelines/test_study1_sensor_topography_config.py studies/tests/pipelines/test_study1_sensor_topography_data.py studies/tests/pipelines/test_study1_sensor_topography_estimands.py studies/tests/pipelines/test_study1_sensor_cluster_inference.py studies/tests/pipelines/test_study1_sensor_topography_plot.py studies/tests/pipelines/test_study1_sensor_topography_writers.py -q`; expect all pass.
- [ ] Run existing neighboring tests: `python -m pytest studies/tests/pipelines/test_study1_power_construct_validity.py studies/tests/pipelines/test_study1_power_construct_validity_figure.py studies/tests/pipelines/test_study1_validity_figures.py -q`; expect all pass.
- [ ] Run `ruff check studies/pain_study/study1/figures studies/tests/pipelines/test_study1_sensor_topography_*.py`; expect clean.
- [ ] Run `make verify-architecture` and `make verify-maintainability`; expect success.
- [ ] Run `git diff --check`; expect no whitespace errors.
- [ ] Commit with `docs: document Study 1 sensor topographies`.

### Task 8: Final independent code review

**Files:** all files changed by Tasks 1-7.

- [ ] Invoke `superpowers:requesting-code-review` with the approved spec, this plan, commit range, and scientific invariants.
- [ ] Address only verified blocking findings with focused failing tests first.
- [ ] Repeat the focused and neighboring verification commands after any correction.
- [ ] Confirm `git status --short` contains no unintended changes and summarize actual test evidence.
