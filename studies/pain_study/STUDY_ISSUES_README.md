# Pain Study Issues Log

This file tracks participant-level and workflow-level issues that affect Study 1,
Study 2, or shared pain-study derivatives.

Use short entries. Keep enough information to justify later inclusion,
exclusion, repair, or rerun decisions. Do not use this log for long run notes,
full terminal output, or temporary debugging details.

Each issue should record:

```text
Date:
Subject(s):
Study/stage:
Issue:
Decision:
Follow-up:
```

## Issues

### 2026-07-24 - Cohort - Analyzer Pulse Artifact Correction Failed On 39% Of Runs

Study/stage: BrainVision Analyzer scanner artifact correction, upstream of all EEG
preprocessing and every pulse-artifact-dependent analysis.

Issue: Analyzer writes the R markers it uses to build the pulse-artifact template. In
35 of the 90 task runs on disk (39%) fewer than 0.5 `Pulse Artifact,R` markers per
scanner volume reached the exported files, so those runs were very likely never
pulse-corrected. Counting markers per volume rather than raw counts accounts for
recording length. The distribution is bimodal: 34 runs sit near zero markers per
volume against a healthy cluster at 0.75 to 1.0, with almost nothing between, so these
are outright failures rather than borderline cases. **No subject has all six runs
usable** (0 of 15, and 0 of 14 after excluding `sub-0006`). Affected runs per subject:

```text
0000:1/6  0001:2/6  0003:2/6  0004:4/6  0005:1/6  0006:5/6  0007:2/6  0008:1/6
0009:3/6  0010:1/6  0011:1/6  0012:3/6  0013:5/6  0014:1/6  0015:3/6
```

Analyzer's own log is close to reliable. For the 11 subjects confirmed present in the
2026-07 batch it raised `The time delay cannot be computed. Do you accept the default
value of 0.21 s?` for 28 of that batch's 29 broken runs, with no false positives. The
one miss in that batch is `sub-0012` run 2 (5 markers against 543 volumes). `sub-0005`
run 4 (69 markers, a partial failure) and `sub-0010` run 1 remain unclassified because
those subjects appear nowhere in the batch log. `sub-0014` and `sub-0015` were not in
that batch, so their 4 broken runs are expected to raise dialogs when processed.

The ECG channel is not the cause. In `sub-0015` it is healthy in all six runs (217 to
239 microvolts RMS) and MNE detects 474 to 530 R peaks at 57 to 64 bpm in every run,
including the three Analyzer could not mark. Marker-locked residual EEG amplitude is
about 18 times higher in the affected runs (11 to 12 microvolts against 0.6 to 0.7),
consistent with real uncorrected artifact rather than a measurement quirk.

Decision: Treat runs below the marker threshold as **not pulse-corrected**. Do not
assume Analyzer correction succeeded for any run without checking its marker count. No
exclusion decision is recorded yet, because the affected fraction is large enough that
exclusion would compromise the analytic sample; repair is the preferred route.

Follow-up: Investigate why Analyzer fails to mark R peaks on roughly 40% of runs when
the ECG channel is healthy, most plausibly a detection-parameter or channel-scaling
issue in its template step; fixing it there is cleaner than working around it. If
Analyzer cannot be made reliable, mark R peaks from the ECG channel outside Analyzer
and re-run pulse correction with those markers. Re-check `sub-0005` run 4 and
`sub-0010` run 1 against the batch records to classify them. Marker availability is
reported automatically: per subject and run in the `Scanner artifact correction
(Analyzer)` section of each subject report, and cohort-wide via
`eeg_pipeline/preprocessing/report/cohort_qc.py`, which accepts a subject filter so a
single processing batch can be summarized on its own.

### 2026-06-30 - `sub-0006` - Pilot Participant Exclusion

Study/stage: Study 1 and Study 2 eligibility.

Issue: `sub-0006` was a pilot participant.

Decision: Exclude `sub-0006` from Study 1 and Study 2 analyses.

Follow-up: Keep `sub-0006` out of the analytic sample unless a future written
analysis plan explicitly defines and justifies a separate pilot-participant
quality-control use case.

### 2026-06-22 - `sub-0008` - Resting-State Baseline Skipped

Study/stage: Ten-minute resting-state baseline recording.

Issue: The resting-state baseline recording was skipped because of technical
issues. No ten-minute baseline recording is available for `sub-0008`.

Decision: Treat the resting-state baseline as missing. Exclude `sub-0008` from
analyses that require this baseline recording.

Follow-up: Retain `sub-0008` for analyses that do not require the resting-state
baseline, provided the participant otherwise meets the relevant eligibility and
quality-control criteria.

### 2026-06-22 - `sub-0009` - Missing Pain Binary Responses

Study/stage: PsychoPy behavioral data and pain-binary-dependent analyses.

Issue: The participant initially did not understand that the pain yes/no question
required a keyboard response. PsychoPy therefore coded `pain_binary_coded` as `-1`
for run 1 trials 1, 3, 6, and 10, and run 2 trial 3.

Decision: Treat these five pain binary responses as missing. Do not infer a binary
response from the VAS rating or include these trials in analyses that require a valid
pain binary response.

Follow-up: Retain the trials for analyses that do not require `pain_binary_coded`,
provided they otherwise meet the relevant eligibility and quality-control criteria.

### 2026-06-14 - `sub-0003` - Run 3 Used Wrong Temperature/Surface Sequence

Study/stage: Study 1 and Study 2 trial eligibility.

Issue: Run 3 was acquired with the run 1 sequence of stimulus temperatures and
selected thermode surfaces instead of the intended run 3 sequence.

Decision: Exclude run 3 for `sub-0003` from analyses that depend on the
predefined run-specific temperature/surface order.

Follow-up: Keep the run-level exclusion documented when regenerating Study 1 or
Study 2 derivatives. Do not treat run 3 as a valid run 3 sequence unless a
future written analysis plan explicitly defines a separate sensitivity analysis
for duplicated sequence runs.

### 2026-06-09 - `sub-0000` - Invalid Study 2 BEM Surfaces

Study/stage: Study 2 source localization, BEM/trans generation.

Issue: MNE rejected the watershed BEM surfaces because the inner skull surface
was not completely inside the outer skull surface.

Decision: Exclude `sub-0000` from source-level Study 2 inference until the BEM is
repaired or regenerated and passes MNE surface nesting checks.

Follow-up: Repair or regenerate the BEM surfaces before including `sub-0000` in
confirmatory source localization. Do not bypass the MNE BEM validation.

### 2026-06-10 - `sub-0004` - Invalid Study 2 BEM Surfaces

Study/stage: Study 2 source localization, BEM/trans generation.

Issue: FreeSurfer `recon-all` completed, but MNE rejected the generated
watershed BEM surfaces because the inner skull surface was not completely inside
the outer skull surface. A controlled retry with MNE watershed `atlas=True`
reached the same MNE surface-nesting failure.

Decision: Exclude `sub-0004` from source-level Study 2 inference until the BEM
is repaired or regenerated and passes MNE validation.

Follow-up: Repair the subject-specific BEM surfaces before including `sub-0004`
in confirmatory source localization. Do not bypass the MNE BEM validation.

### 2026-05-16 - Restart Trigger Artifact In EEG Events

Study/stage: EEG events and downstream derivative generation.

Issue: PsychoPy restarted while EEG recording continued, leaving extra EEG task
triggers in the BIDS events file. These triggers could be mistaken for real task
trials if left unmarked.

Decision: Repair affected events with `scripts/fix_restart_trial_triggers.py`.
The repair keeps the valid behavior-matched trigger window and relabels extra
restart-related triggers as `BAD_restart/...` instead of deleting rows.

Follow-up: Keep pre-repair derivatives archived under:

```text
stale_before_restart_trigger_fix_20260516
```

If a similar restart issue occurs for another participant, relabel extra
restart-related triggers with `scripts/fix_restart_trial_triggers.py` and
regenerate downstream derivatives from the repaired events.

### 2026-05-16 - `sub-0002` - Incomplete MRI Experiment

Study/stage: Study 1 and Study 2 eligibility.

Issue: `sub-0002` wanted out of the MRI before the experiment was complete.

Decision: Exclude `sub-0002` from Study 1 and Study 2 analyses.

Follow-up: Keep `sub-0002` out of the analytic sample unless a future written
analysis plan explicitly defines and justifies a separate incomplete-session
quality-control use case.
