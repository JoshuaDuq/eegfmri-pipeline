# Cardiac gap-fill: beat recovery and BCG correction in Analyzer's marker gaps

Date: 2026-07-30

BrainVision Analyzer's pulse artifact correction is kept. This spec recovers the beats
Analyzer never marked and corrects the BCG only in those stretches, where nothing was
corrected at all. Analyzer's gradient correction and its correction at marked beats are
left untouched, because neither shows a measured deficiency.

## Problem

### Analyzer's correction is excellent where it has a beat

`data/bids_output` is built from `processed_trimmed_0-60s_30-115bpm_marker_template`,
confirmed numerically on sub-0009 run 1 (max abs difference 2.4e-5 µV, float32
round-trip; the other three trimmed exports differ by 100-150 µV). All 104 exports record
the same `.vhdr` history chain, `Raw Data/Scanner Artifact Correction/Pulse Artifact
Correction (Mark R peaks)/Pulse Artifact Correction (Correction)`.

R-locked residual RMS against a circular-shift null, sub-0009 run 1 (RR 898 ms):

| time from R mark | residual / null |
|---|---:|
| 0 ms | 0.05 |
| ±450 ms | 0.05 - 0.6 |
| beyond ±500 ms | ~1.0 |

±450 ms spans the whole cardiac cycle, so there is no uncorrected phase and no limited
correction window. Held-out variance reduction at Analyzer's own marks is 0.14% (run 1)
and -0.03% (run 3) against nulls of 0.24% and 0.36%: zero of 63 channels exceed the null.

### The residual is in the gaps, and the gaps are large

A physiological RR interval cannot be 11 s, so Analyzer's own marker times prove missing
beats without any detector being involved. Across all 104 exports, taking RR > 2 s as a
gap:

| | |
|---|---:|
| recordings with at least one RR > 2 s | 85 of 104 |
| total recording time inside gaps | 6,954 s |
| implied missing beats at each run's median RR | ~6,752 |
| largest single gap | 53.9 s |
| median fraction of a run inside gaps | 4.2% |
| worst run | 77% |

Whole runs fail: sub-0008 run 4 carries 44 marked beats at a median RR of 6.74 s;
sub-0012's baseline has 450 s inside gaps; sub-0004 run 3 has a single 43.6 s gap.

Inside those stretches the BCG is untouched. Splitting ECG-detectable beats into those
Analyzer also marked and those it did not, scored by held-out variance reduction in
1-20 Hz against a circular-shift null:

| sub-0009 | n | R-locked variance | null | channels above null | template p-p |
|---|---:|---:|---:|---:|---:|
| run 1, matched | 442 | 9.5% | 1.7% | 45/63 | 38 µV |
| run 1, unmarked | 109 | 37.3% | 0.7% | 63/63 | 106 µV |
| run 3, matched | 468 | 2.0% | 2.0% | 0/63 | 24 µV |
| run 3, unmarked | 104 | 12.6% | 0.9% | 55/63 | 74 µV |

The unmarked beats are genuine. Their QRS lock ratio against the ECG channel is 3.4,
equal to the matched beats' 3.39, and their EEG epoch SD is 12.7 µV against 11.3 µV for
matched beats, so they are not high-amplitude motion stretches Analyzer sensibly
rejected. (The run-1 matched row is elevated because epochs of 1000 ms at RR 898 ms
overlap their neighbours, so matched beats adjacent to a gap carry part of an uncorrected
neighbour. Run 3, with fewer gaps, shows the clean case: 0 of 63.)

### Off-the-shelf detectors are worse than Analyzer

Lock ratio is the ECG-locked average peak-to-peak over mean single-trial SD, so higher
means the marker set sits on the QRS rather than a T-wave or noise:

| sub-0000 run 1 | beats | implied bpm | QRS lock ratio |
|---|---:|---:|---:|
| Analyzer | 689 | 60.7 | 4.39 |
| `mne.preprocessing.find_ecg_events` | 1073 | 94.5 | 1.58 |
| NeuroKit2, `neurokit` method | 1043 | 91.9 | 1.44 |
| NeuroKit2, `pantompkins1985` | 1566 | 138.0 | 0.71 |

Analyzer is the best of the four. On sub-0009 run 1 the Python detectors return 64.6,
71.1 and 130.1 bpm against Analyzer's 56.5; only on sub-0005, with clean ECG, do all four
agree. The magnetohydrodynamic effect - aortic blood flow in the static field inducing a
Hall voltage that inflates the T-wave - defeats amplitude thresholding, and
Pan-Tompkins' slope-based T-wave rejection does not rescue it despite being designed for
it.

Analyzer does not detect badly. It detects **conservatively**: what it marks is
trustworthy, and it marks too little. That is what makes gap-filling the right shape of
fix.

### Why ICA cannot absorb this

`ica.cardiac_review.enabled` and `promote_exclusions` are already `true`, and cardiac
exclusion runs at scale: 176 components cohort-wide (111 "ECG artifact (MNE)", 65 "heart
beat (MNE-ICALabel)"), 5 to 22 per subject. The residual survives it. The artifact
appears in bursts confined to the gaps, so no stationary spatial filter isolates it, and
ICLabel's `heart` class was trained on non-fMRI EEG where cardiac contamination is small
and ECG-shaped rather than BCG.

Separately, CTPS promotion has never run on the data on disk: no component table contains
the `full-recording CTPS in N/M runs` string promotion writes.

### Head-to-head: Analyzer beats our OBS on both arms

The pulse-markers-only export
(`processed_scanner_artifact_with_pulse_markers_no_bcg_correction`, node
`Scanner Artifact Correction/Pulse Artifact Correction (Mark R peaks)`) makes the
comparison measurable for the first time. It is sample-aligned with the corrected export
(identical `DataPoints`; ECG bit-identical at 0.0 µV difference, confirming pulse
correction touches EEG only), 104 of 104 recordings, `BINARY` / `IEEE_FLOAT_32` /
VECTORIZED at 125 MB per run, and read natively by MNE.

Held-out R-locked variance at Analyzer's own marks, sub-0009 run 1, 482 beats:

| | R-locked variance left | channels above null | alpha retained | 1-20 Hz variance retained |
|---|---:|---:|---:|---:|
| uncorrected | 57.9% | 63/63 | 1.00 | 1.00 |
| Analyzer | 0.16% | 0/63 | 0.54 | 0.61 |
| ours, OBS rank 4 | 2.03% | 11/63 | 0.34 | 0.41 |
| ours, rank 8 | 1.90% | 11/63 | 0.32 | 0.30 |
| ours, rank 12 | 1.97% | 31/63 | 0.29 | 0.24 |
| ours, rank 24 | 4.19% | 44/63 | 0.18 | 0.18 |

Removal plateaus near 1.9-2.0% and degrades above rank 12, so rank tuning does not close
the gap to Analyzer's 0.16%. Analyzer leaves less artifact *and* retains more signal.
Replacing its correction is therefore not justified, and this spec keeps it.

Retained-power columns cannot be read as signal loss on their own: BCG is broadband
across 1-20 Hz and contributes power inside the alpha band, so removing it is expected to
reduce alpha-band power. The sham control below is what separates the two.

### Sham correction: the preservation control

Running the identical correction at circularly-shifted beat times, where no artifact
sits, measures what the procedure destroys irrespective of any artifact. sub-0009 run 1,
OBS rank 4:

| | alpha retained | 1-20 Hz variance retained |
|---|---:|---:|
| at true beats | 0.34 | 0.41 |
| at shifted beats, run 1 | 0.71 | 0.67 |
| at shifted beats, run 2 | 0.74 | 0.68 |

OBS at rank 4 removes ~28% of alpha and ~32% of 1-20 Hz variance where there is nothing
to remove. That is the procedural cost, and it is the preservation arm's primary
statistic.

### Why the referee is built first

Two estimators reported earlier were inflated by their own construction:

- Band artifact share via `band_power_db` with `excluded_hz` dropped whole bins, counting
  background power in those bins as artifact.
- R-locked amplitude as max peak-to-peak across 63 channels has a null of 5-8 µV on this
  data. In sub-0009 run 3 the observed value, 5.71 µV, was **below** its own null of
  6.99 µV.

No correction may be evaluated by an estimator that has not itself been tested against
ground truth.

## Scope

1. Validation harness (the referee), with ground-truth tests.
2. Beat recovery inside Analyzer's gaps, with per-run quality measurement.
3. BCG correction applied **only** inside those gaps.
4. Cohort report over all 90 task runs plus baselines.

Explicitly not in scope: gradient correction (no measured deficiency - the residual comb
was traced to the room, not the scanner); re-correcting beats Analyzer already handled;
any change to `eeg_pipeline/preprocessing/eeg_fmri/`, which is exploratory and is not
imported by the new code.

## Definition of success

Measured on the same runs, sample-aligned, both arms required:

- **Removal.** No R-locked residual above the circular-shift null at recovered beats,
  bringing gap stretches to the state Analyzer already achieves at its own marks
  (0.16% residual, 0 of 63 channels above null).
- **Preservation.** Measured by the **sham control**: the identical correction applied at
  circularly-shifted beat times, where no artifact sits, so everything it removes is
  signal loss. Reported alongside stimulus-evoked response preservation on the `S 1` /
  `S 2` / `S 3` markers, in gap and non-gap stretches separately.

Retained band power against the uncorrected data is **not** a preservation measure on its
own. BCG is broadband across 1-20 Hz and contributes power inside every band of interest,
so removing it necessarily reduces band power; an uncontrolled reading of that reduction
as signal loss is wrong. The sham is what makes the arm interpretable.

The preservation arm is not optional. Analyzer's residual sits ~20x *below* its null at
marked beats, the signature of subtracting an empirical beat-locked average: the artifact
goes and a slice of genuine EEG at the cardiac phase goes with it. A removal-only
criterion would score that over-subtraction as success, and ours as greater success -
the rank sweep shows exactly that failure, with removal flat from rank 4 to 12 while
procedural loss climbs from 0.34 to 0.29 alpha retained.

A consequence to state plainly: after this stage each run is a mixture of Analyzer's
correction and ours, in a proportion that varies by run from 0% to 77%. That is a
documented limitation, reported per run so it can enter any between-participant analysis
as a known covariate rather than as an unmodelled offset.

## Components

### 1. Referee - `eeg_pipeline/qc/artifact_metrics.py`

Pure functions over arrays. No file I/O and no MNE objects in the signatures, so the same
code scores Analyzer's output and ours.

- `epoch_stack(data, onsets_samples, sfreq, window)` -> mean-removed `(n_ch, n_epoch, n_time)`.
- `held_out_reduction(data, onsets, sfreq, window)` -> per-channel variance reduction from
  an odd-beat template applied to even beats, so averaging noise contributes ~0.
- `circular_shift_null(data, onsets, sfreq, window, n_surrogate, seed)` -> null from
  circularly shifting the whole beat train by a random constant. Random event times are
  **not** an acceptable null: they destroy periodicity as well as phase, so any
  quasi-periodic train scores against them.
- `spectral_preservation(before, after, sfreq, exclude_bands)` -> per-channel dB change
  outside the named bands.
- `evoked_preservation(before, after, event_onsets, sfreq, window)` -> per-channel
  correlation and peak-amplitude ratio of the stimulus-locked average.

Returned as frozen dataclasses carrying the observed statistic, its null, and the count of
channels exceeding it. No pass/fail verdict is computed or stored: the harness reports
measurements, thresholds live in the calling stage's configuration.

### 2. Beat recovery - `eeg_pipeline/preprocessing/bcg/detect.py`

`recover_beats(raw, ecg_channel, analyzer_beats, settings) -> BeatRecovery`.

QRS template matching seeded from Analyzer's own marked beats, not a general-purpose
detector. Analyzer's marks are the highest-quality marker set available and every
amplitude-threshold detector tested inflates the count on MHD-affected ECG. Matching on
QRS *shape* rather than amplitude is the defence against the inflated T-wave.

1. Build a QRS template by averaging the ECG around Analyzer's marked beats.
2. Normalised cross-correlation of that template through gap stretches only
   (RR > 2 x the run's median RR). Analyzer's own beats are never revisited.
3. Accept candidates on correlation against the template, subject to a refractory floor
   derived from the run's own RR distribution rather than a fixed bpm ceiling.
4. Re-estimate the template from the combined set and repeat once, so runs where Analyzer
   marked little are not limited by a thin initial template.

Fallback where Analyzer marked too few beats to form a template (sub-0008 run 4 has 44):
seed from another run of the same subject and session, since ECG electrode placement and
body position are fixed within a session. The harness measures whether the transfer
worked, via the lock ratio of the resulting set.

NeuroKit2 0.2.12 is retained as an independent cross-check, not the primary detector.
Disagreement is recorded as a measurement and never rejects a beat: on this cohort the
cross-check is the one more often wrong.

`BeatQuality` fields, all measurements rather than verdicts:

| field | meaning |
|---|---|
| `qrs_lock_ratio` | ECG-locked average p-p over mean single-trial SD, for recovered beats and for Analyzer's separately |
| `rr_median_s`, `rr_sd_s`, `rr_min_s`, `rr_max_s` | physiological plausibility |
| `implied_bpm` | rate implied by the combined set |
| `gap_seconds_before`, `gap_seconds_after` | recording time still inside gaps |
| `recovered_beats` | count added |
| `crosscheck_agreement` | fraction placed within 50 ms by NeuroKit2 |
| `refractory_violations` | detections closer than the run's floor |

A run whose ECG cannot support detection must be distinguishable from a detector failure.
The lock ratio of *Analyzer's own* beats is the diagnostic: where even those score poorly,
the ECG is the problem and no detector rescues it. Those runs are reported, not corrected.

### 3. Gap correction - `eeg_pipeline/preprocessing/bcg/correct.py`

Correction is computed on the **pulse-markers-only export**, where BCG is present at every
beat, and only the gap stretches are carried into the output. Because that export is
sample-aligned with Analyzer's, the result is Analyzer's correction everywhere with gap
stretches substituted from ours.

Learning the basis from all beats and applying it only in gaps is what the new export
buys. An earlier version of this design had to learn the basis from the gap beats alone -
the only uncorrected epochs available at the time - which for a short gap means a handful
of epochs and a noisy basis. Now the basis comes from every beat in the run (~500) while
the subtraction stays confined to the gaps.

Two candidate methods, benchmarked rather than assumed:

- `mne.preprocessing.apply_pca_obs` (Niazy et al. 2005), present in MNE 1.12.1 with
  signature `(raw, picks, *, qrs_times, n_components=4, n_jobs=None, copy=True)`.
- Sliding-window average artifact subtraction (Allen et al. 1998), which is the family
  Analyzer's own correction belongs to and which preserves signal better than our OBS
  does in the measurements above.

Selection is by the two-armed criterion on the benchmark runs, with the sham control
supplying the preservation arm. Rank is not tuned for removal alone: the sweep shows
removal flat to rank 12 while procedural loss rises monotonically.

The implementation must verify, not assume, that the chosen method modifies only the
neighbourhoods of the supplied beat times. A test asserts that samples outside those
epochs are bit-identical before and after.

Output mirrors `remove_line_comb.py`: a new BIDS tree, sidecars mirrored, and a float32
round-trip check on every written binary.

### 4. CLI - `studies/pain_study/scripts/correct_cardiac_gaps.py`

`benchmark` over named runs, `apply` over the cohort, `verify` re-scoring the written
output with the referee. Mirrors the subcommand shape of `remove_line_comb.py`.

Input is the **Analyzer export**, not the BIDS tree: seeding needs Analyzer's R marks, and
BIDS carries none - the conversion keeps only `S 1` / `S 2` / `S 3`, so all 90 BIDS runs
hold zero R markers while the 104 exports hold 48,333 between them.

Beat times are emitted as sample indices valid against BIDS without remapping, because the
two are the same samples (sub-0009 run 1 matches its export to 2.4e-5 µV over 512,100
samples). The first implementation step verifies that identity for every run rather than
assuming it, since the approach depends on it.

## Data flow

```
Analyzer export (R marks + ECG + corrected EEG)
   -> recover beats in gaps          -> beats + quality TSV
   -> apply_pca_obs at recovered beats -> new BIDS tree
   -> referee re-scores               -> removal + preservation TSVs
```

Ordering against the existing line-comb stage: cardiac gap-fill runs first, on the
Analyzer export, and the line-comb stage then runs on its output. The two are independent
in frequency (BCG below 30 Hz, comb above 28 Hz), so the order is a convention rather
than a constraint, but it is fixed so results are reproducible.

## Error handling

A run whose beat recovery is poor is reported, never silently corrected, and never aborts
the cohort run - a detector resolving nothing is a measurement, not a fault. Quality
fields are written for all runs including failures, so the cohort report is complete by
construction. Missing ECG channel, unreadable recording, and zero recovered beats are each
a row with a populated `status` column, not an exception.

## Testing

The referee is tested against ground truth, which is the point of it:

- Inject a synthetic BCG of known amplitude at known beat times into real EEG; the
  measured variance reduction must recover the injected fraction within tolerance.
- On data with no injected artifact the statistic must report ~0 and must not exceed its
  null.
- The naive peak-to-peak statistic must fail that second test, locking in the regression.
- A random-times null must be shown anticonservative against a quasi-periodic train where
  the circular-shift null is not.
- The sham control must report substantial removal for a correction applied at shifted
  beat times, since that is the behaviour it exists to expose. Measured at 0.71-0.74 alpha
  retained for OBS rank 4 on sub-0009 run 1.

Export integrity is asserted per run before any correction: identical `n_times` against
the corrected export, and an ECG channel matching it at 0.0 µV. Both are cheap and both
would invalidate the substitution step if they failed.

Beat recovery is tested on synthetic ECG with known R times plus an inflated T-wave
simulating the MHD effect: recovery must return exactly the R peaks with zero refractory
violations. A second test deletes a known stretch of markers from a real run and requires
recovery to find them, scored against the deleted ground truth.

Correction is tested for confinement (samples outside recovered-beat epochs bit-identical),
for round-trip fidelity of the written binary, and end-to-end on a run with injected BCG
in a synthetic gap.

Per the project's standing constraint, verification uses targeted subsets; the full suite
takes ~9 minutes and is not run per change.

## Dependencies

`neurokit2==0.2.12` added to `pyproject.toml`, installed and verified: it resolves under
the project's `numpy<2.0.0` pin, leaving numpy at 1.26.4 and MNE at 1.12.1. Version 0.2.13
pulls numpy 2.5.1 and would break the pin for no gain here.

## Risks

- Runs where Analyzer marked almost nothing (sub-0008 run 4, 77% of the run in gaps) are
  effectively corrected by us end-to-end. The mixture proportion is reported per run so
  this is visible rather than hidden.
- Where the ECG itself is poor, no detector helps. The lock ratio of Analyzer's own beats
  separates that case, and those runs are reported uncorrected.
- Procedural signal loss is substantial and unavoidable. The sham shows OBS at rank 4
  removing ~28% of alpha where no artifact exists. Analyzer preserves better (0.54 alpha
  retained against our 0.34) while also removing more artifact, so the gap stretches will
  carry a different, more destructive correction than the rest of the run. Benchmarking
  AAS against OBS is the mitigation; accepting a worse correction in the gaps than
  elsewhere is the fallback, and it is still far better than the 57.9% left uncorrected
  there today.
- Recovered beats concentrate in stretches that may differ systematically from the rest of
  the run (motion, arousal). Correcting only there could introduce a time-varying
  difference within a run, which the preservation arm measures in gap and non-gap
  stretches separately.
- Preservation varies by subject in ways not yet understood. On sub-0009 run 1 Analyzer
  retains 0.54 alpha against our 0.34, but on sub-0005 run 1 the two are level (0.634 vs
  0.638) while Analyzer still removes more artifact. The benchmark must cover several
  subjects before a method is chosen, not the single run that motivated the design.

## References

- Allen, Polizzi, Krakow, Fish & Lemieux (1998). Pulse artifact removal by average artifact
  subtraction. NeuroImage.
- Niazy, Beckmann, Iannetti, Brady & Smith (2005). Removal of FMRI environment artifacts
  from EEG data using optimal basis sets. NeuroImage. Basis of `apply_pca_obs`.
- Pan & Tompkins (1985). A real-time QRS detection algorithm. IEEE TBME.
- Makowski et al. (2021). NeuroKit2: a Python toolbox for neurophysiological signal
  processing. Behavior Research Methods.
