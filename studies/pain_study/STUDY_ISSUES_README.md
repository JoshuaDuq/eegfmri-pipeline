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
