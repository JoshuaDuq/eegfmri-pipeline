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

### 2026-09-09 - `sub-0016` - Thermode Trigger Never Recorded, `Stim_on` Substituted

Study/stage: EEG acquisition; FASTR 1 kHz EEG BIDS conversion (`bids_output/eeg`) and
every thermode-locked Study 1 analysis reading this participant.

Issue: All six `sub-0016` thermal runs carry 11 `Stim_on/S  1` markers and zero
`Trig_therm/T  1`. The gap is in the acquisition, not in processing: `Trig_therm` is
absent from all four generations of the participant's data, `original_untrimmed_5khz`
included, so nothing holds it. sub-0015 (2026-07-13) and sub-0018 (2026-08-06) both
carry 11 of each, which isolates the fault to the 2026-07-29 session -- the thermode's
trigger line was not feeding the amplifier that day. The paradigm's other markers
(`Iti_start`, `Painq_on`, `Rating_on`) each recorded 11, and the baseline carries
neither marker, which is normal.

The two are not the same event. `Stim_on` is PsychoPy's software marker, written when
the script commands the stimulus; `Trig_therm` is the thermode's hardware echo
confirming it fired. In runs holding both, `Trig_therm` follows `Stim_on` by 5-10 ms
(mean 9 ms) -- the device round-trip.

Decision: Convert with `--canonicalize-thermode-markers`, rewriting the 11 `Stim_on`
markers as `Trig_therm/T  1`, so all 126 thermal runs carry 11 thermode events. The flag
is a no-op on the other 120 runs, which already hold exactly 11 canonical markers.
sub-0016's thermode onsets are therefore ~9 ms early against the rest of the cohort.
That bias is accepted rather than corrected: shifting the onsets by the cohort mean
would put a modelled number in the delivered events table in place of a recorded one.

Follow-up: Treat sub-0016 as ~9 ms early wherever thermode-locked timing matters at that
scale -- ERP latency measures in particular. Add the offset at analysis time, or exclude
the participant, if a result turns on latencies finer than roughly 20 ms.

### 2026-09-08 - Cohort - EEG Trim Ends One TR After The Last Volume Marker

Study/stage: 5 kHz source trimming (`original_trimmed_5khz`) and EEG BIDS conversion.

Issue: The previous trim cut from the first `Volume,V  1` marker through that last
marker as the final sample. A volume marker is the start of a volume, so that dropped
the last TR — or the fraction of it that was recorded after EEG stopped. Dummy volumes
the scanner plays before the first marker are not in the saved BOLD NIfTI; when EEG has
fewer markers than 570 BOLD volumes, the missing volumes are at the end of the scan.

Decision: Re-trim from `original_untrimmed_5khz`. Start at the first `Volume` of the
first contiguous 0.9 s block (EEG t = 0 = BOLD volume 1). End one TR after that block's
last marker, clipped to the recording end. Keep a later scanner restart out of the file
(sub-0000 run 1 keeps its first 570-volume block). The previous last-marker-as-final-sample
files are in `data/source_data/_archive/original_trimmed_5khz_last_marker_end/`. How the
live files were cut is in each `sub-*/eeg/original_trimmed_5khz/README.md`.

Follow-up: `step0_trimmed_raw_5khz/` is a flat copy of the old trim and is stale until
rebuilt. Analyzer 1 kHz BIDS was produced from the old trim and still ends on the last
marker; it does not gain the recovered last-TR samples until that chain is re-run.

### 2026-08-03 - `sub-0000` Run 1 - Two Scanner Acquisitions In One Run

Study/stage: EEG BIDS conversion and every Study 1 analysis reading this run.

Issue: The scanner stopped and restarted inside `sub-0000` run 1 while the EEG kept
recording. The run held two acquisition blocks: 570 volumes from 0 to 512.1 s, a 63.5 s
scanner-off gap, then 117 volumes from 575.6 to 680.0 s. Only the first block has an
fMRI counterpart; `sub-0000_task-thermalactive_run-01_bold.nii.gz` is 570 volumes.
`trim_to_volume_bounds` crops between the first and last volume marker, so both blocks
survived conversion and the run reported 687 volume markers against the BOLD's 570 --
the only run in the cohort with more EEG volume markers than BOLD volumes. PsychoPy was
restarted in the second block, which is where the two `BAD_restart` triggers from the
2026-05-16 repair live.

A whole-run measurement of volume-locked residual is inflated by the second block and
was first read as a scanner artifact correction failure. Measured per block, the
volume-locked average is 6.0 microvolts peak-to-peak in block A against 28.1 in block B
(runs 2-6: 4.0 to 6.4). Restricted to the 11 stimulation windows (-5 to +15 s around
each `Trig_therm`), block A sits inside the range of this participant's other five runs
on every measure taken: volume-locked residual 3.21 microvolts median (others 2.69 to
3.20), residual comb prominence 0.23 dB median (others 0.04 to 0.50), 0.937 pulse
markers per volume and 1.53 microvolts of R-locked residual, both the best of the six,
and no missed-beat gap inside any window. All 11 windows fall inside block A on an
unbroken 0.9 s volume train, ending 19.2 s before the scanner stopped.

Decision: Crop the run to its first acquisition block rather than exclude it. Applied
with `scripts/crop_restart_scanner_block.py` to both `data/bids_output/eeg` and
`data/bids_output/eeg_linecleaned` at 513.0 s (the last block-A volume plus one
repetition time), leaving 570 volume markers that match the BOLD exactly. The `.eeg`
binary is truncated byte-wise, so retained samples are bit-identical. Pre-crop copies
are kept alongside each file with a `.precrop.bak` suffix.

Follow-up: This repair lives in the delivered BIDS root, not in the conversion, so a
regeneration of `sub-0000` from source will reintroduce both blocks -- re-run the crop
script after any reconversion. The current adaptive line-comb workflow refits this cropped
source directly; the obsolete line-cleaned derivative had instead inherited a model fitted
to the full recording and must not be reused. The EEG/fMRI alignment table
`data/derivatives_local/qc/eeg_fmri_alignment/run_qc.tsv` still carries a pre-trim row
for this run (688 markers, 757.032 s) and needs a cohort regeneration to catch up. No
Study 1 derivative had been built from this run, so nothing else needed regenerating.

### 2026-08-04 - `sub-0001` - Volume-Bound Trimming Verified

Study/stage: EEG BIDS conversion and every Study 1 analysis reading `sub-0001`.

Issue: A previous trim was suspected of using the wrong end boundary. The six current BIDS
runs were rechecked from their BrainVision sample counts and volume markers. Each starts at
0.0 s and ends one 0.9 s repetition time after its last retained volume marker. Their
retained marker counts are 541, 544, 531, 536, 510, and 512 for runs 1--6, respectively.
The corresponding durations are 486.9, 489.6, 477.9, 482.4, 459.0, and 460.8 s.

Decision: Include all six `sub-0001` runs. The current files have the intended trim
geometry and are the inputs to the fresh line-comb benchmark, tests, and removal.

Follow-up: Regenerate `data/derivatives_local/qc/eeg_fmri_alignment/run_qc.tsv`; its
`sub-0001` rows describe the earlier files and are each approximately one marker and a
fraction of a second longer. Archive the newer trimming script in this repository before
the next conversion. The tracked `scripts/trim_brainvision_to_volume_bounds.py` currently
implements the older boundary convention in which the last volume marker is the final
sample, so it cannot reproduce the verified one-TR-after-last-marker files by itself.

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
assume Analyzer correction succeeded for any run without checking its marker count. A
cohort-wide exclusion decision is not recorded because the affected fraction is large
enough to compromise the analytic sample; repair is preferred. The later 2026-08-03
entry records the scanner-restart crop that keeps `sub-0000` run 1 eligible.

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
