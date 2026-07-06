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
