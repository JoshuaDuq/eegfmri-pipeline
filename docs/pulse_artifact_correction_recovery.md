# Pulse artifact correction: why it failed on a third of the runs, and the fix

**Status:** RESOLVED at source and verified. The total failures are gone; incomplete
coverage within runs remains open — see [After the re-export](#after-the-re-export-what-the-parameter-sweep-showed).
**Investigated:** 2026-07-26. **Resolved:** 2026-07-27. **Verified:** 2026-07-27.
**Dataset:** `task-thermalactive`, 15 subjects, 90 runs, plus baseline recordings.

---

## Conclusion

**The EEG acquisition started before the MRI and stopped after it.** Those leading and
trailing periods carry no `Volume/V` markers, so BrainVision Analyzer's MR Correction had no
gradient template for them and left the gradient artifact in place. Analyzer's Cardioballistic
(CB) Correction then builds its pulse template from `Search Pulse Template: Start = 0,
Length = 15 s` — which is exactly that uncorrected region. A template built on gradient
artifact matches no heartbeat, so:

```
EEG recorded outside the MRI window
  -> no volume markers there
  -> MR Correction cannot remove the gradient artifact there
  -> huge uncorrected transient at the start (and end) of every recording
  -> CB Correction builds its pulse template from that transient
  -> no R peak correlates with it
  -> R detection fails, so the R-to-artifact delay cannot be computed
  -> "The time delay cannot be computed. Do you accept the default value of 0.21 s?"
  -> no markers written, no pulse template subtracted
  -> the ballistocardiogram stays in the EEG at 20-36 uV
```

**The fix is to trim each recording to the MRI acquisition window** — drop the samples before
the first volume marker and after the last — then re-run the Analyzer chain. Done that way,
the 0.21 s prompt no longer appears.

No parameter in the CB Correction dialog fixes this. The cause is upstream of it, in data that
should not have been handed to the gradient correction in the first place.

---

## Why this was hard to find

The prompt names the *delay*, not the detection, and the delay is a separate computation used
only to place the correction template. That sent the investigation toward CB Correction's
detection parameters, which turned out to be innocent:

- Marker-derived heart rates across the cohort are **48–72 bpm**, and the dialog's default
  window of **45–80 bpm already contained all 15 subjects' true intervals** — 15 of 15.
  Detection failed anyway, so no rate value could have fixed it.
- Every published external QRS detector also failed (see below), because they were all being
  asked to work on a signal whose first 15 seconds were gradient artifact.

The measured "large transients" that kept appearing — sub-0000 run 1 at **14 mV** peak-to-peak
against a 2.5 mV typical — were the uncorrected gradient artifact itself. That was the
symptom pointing at the cause the whole time.

---

## The evidence, and why it still matters

Kept because it quantifies the harm, and because it is the yardstick for confirming the fix
worked.

### Scope

Counted two independent ways that agree exactly — the pulse-marker QC's error strings, and a
direct scan of `Pulse Artifact/R` annotations:

| | runs |
|---|---|
| No pulse markers at all | **33** |
| Sparse train | 8 |
| Full train | 49 |

**14 of 15 subjects had at least one affected run.** Only sub-0005 was clean. Worst: sub-0013
and sub-0006 at 5 of 6 runs, sub-0004 at 4 of 6. Baseline recordings were affected too
(sub-0004, sub-0008, sub-0013 carry the prompt in the Analyzer log).

### Harm

Measured within subject, so head position, field strength and impedance are fixed. Beat train
taken independently of the Analyzer markers; metric is peak-to-peak of the across-channel RMS
of the beat-locked average over 0–0.5 s post-R, measured **before** ICA so it reports what
Analyzer left rather than what MNE cleaned up after it.

| subject | marker-bearing runs | affected run |
|---|---|---|
| sub-0014 | 0.53, 0.90, 0.92, 1.06 µV (median 0.91) | r6 = **20.18 µV** — 22× |
| sub-0010 | 1.46, 9.59 µV (median 5.53) | r1 = **36.11 µV** — 6.5× |

20–36 µV is larger than the alpha rhythm and far larger than any ERP. Where the correction
did run, it works very well — under 1 µV.

### The log fingerprint

The Analyzer batch log contains 31 entries, all the same message from
`Pulse Artifact Correction (Mark R peaks)`:

> The time delay cannot be computed. Do you accept the default value of 0.21 s? Yes

Its 28 thermal entries match the zero-marker runs **28 for 28**. Grep any Analyzer log for
that message to enumerate affected recordings.

### The ECG was never the problem

Measured without any peak detector, so a bad detector could not flatter or damn a run.
Periodicity strength is the autocorrelation peak height of the QRS-band envelope at a cardiac
lag:

| | n | median periodicity | two independent rate estimates agree |
|---|---|---|---|
| Analyzer marked | 57 | 0.270 | 50/57 |
| **Analyzer marked nothing** | 33 | **0.293** | **30/33** |

The affected runs' ECG is *marginally better* than the runs Analyzer handled. Consistent with
the resolution: nothing was wrong with the heart signal, only with what the template was built
from.

### Uncorrected, not contaminated

Testing whether Analyzer subtracted something at the 0.21 s default spacing (4.76 Hz) found no
harmonic ladder at 9.52 or 14.29 Hz, and an elevated fundamental in only 2 of 4 runs — both
being runs whose true heart rate puts a harmonic near 4.76 Hz anyway. So the affected runs were
left alone rather than actively damaged, which is why trimming and re-running recovers them
rather than requiring them to be discarded.

---

## What to do now

1. **Trim every affected recording** to the MRI acquisition window — drop samples before the
   first `Volume/V` marker and after the last — and re-run the Analyzer chain (MR Correction,
   then CB Correction). Confirmed to remove the 0.21 s prompt.

2. **Consider trimming every recording, not only the failing ones.** The 49 runs that produced
   a marker train did so despite the same untrimmed leading period; their templates happened to
   land on usable data. Their corrections are not necessarily as good as they could be, and
   uniform treatment removes a source of between-run variability.

3. **Verify, and not by marker count alone.** After re-export and re-running the pipeline,
   check the cohort report's *"Runs the pulse correction never ran on"* table is empty **and**
   that `bcg_residual_uv` has dropped to roughly 1 µV. sub-0010 r2 has 409 markers and still
   9.59 µV of residual — marker presence does not guarantee a good correction.

   *Done 2026-07-27. The table is empty — all 90 runs carry a marker train. The residual
   check turned out to need a correction of its own; see
   [Measuring the residual on Analyzer's own markers under-reports](#measuring-the-residual-on-analyzers-own-markers-under-reports).*

4. **Until verified, do not run cross-run analyses.** Corrected and uncorrected runs were
   interleaved *within* subjects, so pooling runs within a participant pools two noise regimes
   differing by more than an order of magnitude, and any run-position effect in the design is
   confounded with whether the correction happened to work for that run.

### Prevention

- **Stop the EEG inside the MRI window, or trim before Analyzer.** The acquisition overlap is
  what produced the whole problem. Trimming to the first and last volume marker is mechanical
  and can be part of the standard import step.
- **Watch for the 0.21 s prompt.** It is the fingerprint. Any Analyzer batch that emits it has
  runs whose pulse artifact was never corrected.
- **Check the cohort report's Analyzer section** after each preprocessing run. It now surfaces
  this directly, which it did not when the problem was introduced.

### Runs that were affected

Use as the checklist for step 1.

```
0000/r5   0001/r3  0001/r4  0003/r2  0003/r6
0004/r2   0004/r3  0004/r5  0004/r6
0006/r2   0006/r3  0006/r4  0006/r5  0006/r6
0007/r3   0007/r5  0008/r4
0009/r2   0009/r3  0009/r5  0010/r1  0011/r3
0012/r1   0012/r2  0012/r5
0013/r1   0013/r2  0013/r4  0013/r5  0013/r6
0014/r6   0015/r2  0015/r4
```

Sparse train, partially uncorrected *within* the run:

```
0003/r1  0003/r4  0005/r4  0009/r1  0009/r6  0011/r1  0013/r3  0015/r6
```

Affected baselines: **sub-0004, sub-0008, sub-0013**.

---

## After the re-export: what the parameter sweep showed

Added 2026-07-27, after trimming and re-exporting all 104 recordings (90 thermal, 14
baseline). Four batches were produced, differing only in the Cardioballistic Correction
settings named below. Everything here is measured from the exported `.vmrk`/`.eeg`
triplets; no signal was re-processed.

**Coverage** below means: the fraction of a recording that lies inside the R-marker train
and not inside a gap, where a gap is an inter-marker interval longer than 1.75× that run's
own median. It is a within-run measure, so it is not perfectly comparable between batches
whose marker density differs a lot — the sub-0009 r3 result below was confirmed
independently before being relied on.

### The trimming worked, and the export is deterministic

- **All 90 thermal runs now carry an R-marker train.** The 33 runs with no markers at all
  are gone; the cohort report's *"Runs the pulse correction never ran on"* table is empty.
- Re-running the same settings reproduced the R count on **90 of 90** runs. The one
  apparent difference was a join error on a mislabelled file, not a real change.
- **But coverage within runs is incomplete.** At the original `Start = 0, Length = 15 s`
  setting: median 83.2%, mean 78.6%, **11 recordings below 50%**, 30 at or above 95%.

### The template search window: one decisive case

sub-0009 run 3 exists as an A/B pair, same trimmed source and same history path, ECG
bit-identical, differing only in the template search window:

| | `Length = 15 s` | `Length = 60 s` |
|---|---|---|
| R markers | 44 | **487** |
| coverage | 9.8% | **86.1%** |
| residual BCG, both epoched on the same 486-beat train | 19.76 µV | **0.52 µV** |

The 44 markers were not a sparse-but-correct subset. Against the ECG they sit at +313 ms
with an interquartile spread of ~90 ms, while the 487 sit at −222 ms with a 6 ms MAD, and
none of the 44 falls within 50 ms of a good marker. Analyzer was subtracting its template
at times unrelated to the heartbeat.

### Measuring the residual on Analyzer's own markers under-reports

`compute_cardiac_residual` takes its beat train from `detect_ecg_events`, which prefers
Analyzer's marker train. For a partially-corrected run that means the metric averages only
the beats the correction found, so the run grades itself on its successes:

| sub-0009 r3, `Length = 15 s` | residual |
|---|---|
| scored on its own 44 markers | 1.33 µV |
| scored on the true 486-beat train | **19.76 µV** |

The preference is sound for the ICA review it was written for — that comparison is either
side of *MNE's* exclusions — but not for grading Analyzer's own correction. Any residual
figure quoted for a run with incomplete coverage is a floor, not an estimate.

**Fixed 2026-07-27.** `CardiacResidual` now carries `beat_train_coverage`, the share of the
recording its beat train actually spans, and the sidecar records it as
`bcg_beat_train_coverage` beside `bcg_residual_uv`. Gaps use the same `MISSED_BEAT_FACTOR`
as the tachogram's dropout count, so the figure and the table cannot disagree about which
intervals were missed. The beat source is unchanged — the ECG-channel detector is
documented below as unreliable on this dataset, so replacing the marker train would trade a
known bias for an unknown one. What changed is that the qualifier now travels with the
number: sub-0009 r3 under the 15 s window reports **1.33 µV at 7.9% coverage**, and under
the 60 s window **0.52 µV at 84.6% coverage**. Read together those cannot be confused; read
alone the first is the cleaner-looking run.

Those two percentages differ slightly from the 9.8% and 86.1% quoted elsewhere in this
section: the batch scans used a 1.75× gap factor, while the pipeline reuses its own
`MISSED_BEAT_FACTOR` of 1.5 so that this measurement and the tachogram agree. The choice of
factor moves the number by a few points and changes no ordering.

### The template window is a high-variance lever, not a fix

`Mark Found Template` writes `TPULSE` `TSTART`/`TPEAK`/`TEND`, exactly one per recording —
so Analyzer selects a single segment rather than averaging over the window. Template width
is either **0.500 s** (77 recordings) or **1.041 s** (27); 1.041 s is exactly the configured
`Pulse Rate` of 1041.5 ms, so **changing the rate setting also moves the template width**.

At `Length = 15 s` the search is not boundary-limited: median `TPEAK` is 4.10 s and only 9
of 104 templates land in the last 3 s. Template *position* barely tracks coverage
(r = +0.20); template *quality* — the percentile of the chosen segment's peak within the
run's own ECG envelope — does better but still weakly (r = +0.39). Quality separates only
the extreme failures: 15 of 104 templates sit below the 90th percentile of their own
envelope, sub-0009 r3 worst at the 45.5th with a crest factor of 0.93, i.e. a segment
*quieter than the run's median*. Meanwhile **18 recordings with a ≥95th-percentile template
still sit below 75% coverage** — for those the template was never the problem.

Three windows, all 104 recordings:

| `Length` | median | mean | <50% | ≥95% | total R | median `TPEAK` |
|---|---|---|---|---|---|---|
| 15 s | 83.2% | 78.6% | 11 | 30 | 44,352 | 4.10 s |
| **60 s** | 87.4% | **82.0%** | **9** | 33 | **45,596** | 18.40 s |
| 400 s | **88.0%** | 80.7% | 10 | **34** | 44,836 | 43.96 s |
| best-of-three | 89.2% | 84.2% | 6 | 36 | — | — |

Longer is not better. Against 15 s, the 400 s window improved 26 recordings and worsened
22, median change +0.00 pp. Its damage is one-sided where it matters: of the 30 recordings
already at ≥95%, **none improved and five lost ground**, worst sub-0008/11h28 at
99.3% → 56.3%. The 60 s window's worst loss in that same group is 6.8 pp.

**Twelve recordings return identical coverage across all three windows** — the template
search is irrelevant to them. Six of sub-0012's seven recordings are in that list, along
with sub-0008 ×2, sub-0010, sub-0003 and sub-0013.

### The 45–80 bpm rate window was censoring real beats

Across 90 runs and 36,835 inter-marker intervals at the default rate window:

```
0.680-0.740 s      0 intervals
0.740-0.750 s     21
0.750-0.760 s    739      <- 0.750 s is the 80 bpm limit
0.760-0.770 s    278
```

A cliff to zero followed by a spike in the next 10 ms bin is not a heart-rate distribution.
It binds in **22 of 90 runs** across sub-0001, sub-0003, sub-0004, sub-0006, sub-0009 and
sub-0012; sub-0009 r6 has 30.1% of its intervals in that single bin. **sub-0003 and
sub-0012 have all seven recordings pinned** — and they are the same subjects no template
window could move.

Widening to **30–115 bpm** (template window held at 60 s) cleared it: intervals below
0.750 s went from **0 to ~4,990**, and sub-0003 was transformed.

| sub-0003 (true rate 80.0 bpm) | 45–80 | 30–115 |
|---|---|---|
| 11h40 | 49.3% | **99.9%** |
| 11h49 | 50.1% | **99.7%** |
| 11h30 | 52.3% | **98.8%** |
| 11h20 | 79.4% | **99.8%** |
| 11h11 | 76.3% | **99.3%** |

Its implied rate moved from a clipped 42.9–74.9 bpm to 79.7–89.2, matching the 80.0 bpm
measured independently.

**No T-wave doubling occurred.** The concern that a ceiling above twice a slow subject's
rate would let a T wave be taken as the next R peak did not materialise: sub-0013
47.6 → 47.6 bpm (R count 1.00×), sub-0014 57.3 → 57.3 (1.00×), sub-0010 52.4 → 52.4
(1.04×), sub-0005 56.4 → 59.5 (1.13×). The correlation trigger rejects them.

**But widening removes a prior the detector was using.** sub-0008 sits at 60.0 bpm, well
inside the old window, so the raised ceiling gave it nothing — 0.0% of its intervals fall
below 0.70 s in either batch — while the loosened search cost it tracking:

| | 45–80 | 30–115 |
|---|---|---|
| sub-0008/11h19 | 97.4% | 67.2% |
| sub-0008/11h38 | 99.8% | 71.0% |
| sub-0008/11h48 | 166 markers, 0.969 s median interval | 44 markers, **6.737 s** median |
| sub-0012/11h13 | 57.6% | 16.5% |
| sub-0012/10h49 | 63.3% | 23.0% |

Cohort effect of 45–80 → 30–115 at a fixed 60 s template window: median 87.4% → **92.8%**,
mean 82.0% → 83.5%, ≥95% coverage 33 → **44** recordings, total R markers 45,596 →
**48,333**; 32 improved, 10 worsened.

**The two failure modes run in opposite directions.** Too narrow at the top clips the fast
subjects; too wide removes the constraint the detector relies on. Which one bites depends
on where the window sits relative to that subject's own rate, which is what a single global
number cannot get right for everyone. 30–115 bpm is nonetheless a clear improvement on
45–80 and is the better choice if a single window is required.

### Per-subject rates and the window each implies

Rates from the autocorrelation of the QRS-band ECG envelope — no peak detector, so a bad
detector can neither flatter nor damn a subject. Validated against the 30 recordings whose
marker train is both complete and uncensored: **median error 0.24 bpm, worst 1.30 bpm,
r = 0.996**. The window scales the rate by the real beat-to-beat spread measured on those
same runs (1st–99th percentile of interval / run median = 0.796–1.237) plus 10% for
across-run drift.

| subject | rate | Min | Max | | subject | rate | Min | Max |
|---|---|---|---|---|---|---|---|---|
| sub-0000 | 58.8 | 40 | 85 | | sub-0009 | 67.4 | 45 | 95 |
| sub-0001 | 70.6 | 50 | 100 | | sub-0010 | 53.1 | 35 | 75 |
| sub-0003 | 80.0 | 55 | 115 | | sub-0011 | 63.8 | 45 | 90 |
| sub-0004 | 69.8 | 50 | 100 | | sub-0012 | 67.4 | 45 | 95 |
| sub-0005 | 56.1 | 40 | 80 | | sub-0013 | 47.6 | 30 | 70 |
| sub-0006 | 67.4 | 45 | 95 | | sub-0014 | 57.1 | 40 | 80 |
| sub-0007 | 58.5 | 40 | 85 | | sub-0015 | 66.7 | 45 | 95 |
| sub-0008 | 60.0 | 40 | 85 | | | | | |

For a new participant the rule is `Min = rate × 0.73`, `Max = rate × 1.38`, rounded outward
to 5 bpm, with the rate read off that participant's own baseline recording. That reproduces
every row above.

### Open

- **The per-subject rate window is untested.** It predicts that sub-0003 keeps its ~+50 pp
  (window 55–115) while sub-0008 keeps its 97–99% (window 40–85). One batch settles it.
- **`Correlation Trigger` (0.6) and `Amplitude Trigger` (0.4–1.2) remain undetermined.**
  Reconstructing them from outside Analyzer failed its own sanity check — 93% of the beats
  Analyzer *did* mark fell outside the reconstructed amplitude bounds — so no value is
  recommended. They are the remaining candidate for the 18 recordings that have a good
  template and a rate window that fits and are still poor.
- **sub-0012 needs individual attention.** Under the wide window its implied rate rose to
  85–91 bpm against a measured 67.4 while coverage collapsed, which is the detector locking
  onto wrong intervals rather than a window problem.
- **Naming defect in the audit exports.** In all four batch folders sub-0003's
  `11h30.23.962` recording is labelled `run1`, duplicating the real `run1` at
  `11h11.33.962`; there is no `run3`. Join these folders on the acquisition timestamp, not
  the run label, until it is fixed.

### Where the data is

Under `/Volumes/KINGSTON/EEG_fMRI_data/source_data/`:

| batch | folder |
|---|---|
| `Length = 15 s`, 45–80 bpm | `processed_trimmed_0-15s_marker_template/` |
| `Length = 60 s`, 45–80 bpm | `processed_trimmed_0-60s_marker_template/` |
| `Length = 400 s`, 45–80 bpm | `processed_trimmed_0-400s_marker_template/` |
| `Length = 60 s`, 30–115 bpm | `processed_trimmed_0-60s_30-115bpm_marker_template/` |

All four carry `Mark Found Template` output. Per-run trimmed originals are in
`sub-*/eeg/original_trimmed_5khz/`, untrimmed in `original_untrimmed_5khz/`.

---

## What was tried and did not work

Recorded so nobody repeats it.

**Replacing Analyzer's detector with an external one.** The 75 runs Analyzer marked were used
as ground truth, scored two ways — sensitivity and precision, matched within ±50 ms — with a
95% bar on both, set before any run was scored. Model selection on 8 runs, confirmed on 8 held
out:

| detector | sensitivity | precision |
|---|---|---|
| neurokit | 54.3% | 42.0% |
| elgendi2010 | 44.3% | 26.9% |
| pantompkins1985 | 1.3% | 17.0% |
| hamilton2002 | 1.2% | 15.5% |
| engzeemod2012 | 0.4% | 55.0% |
| kalidas2017 | 1.5% | 25.4% |
| rodrigues2021 | crashed on every run | — |

Best on the held-out set: **62.2% / 50.8%**. Nothing was written; a wrong beat train would
make Analyzer subtract at the wrong times, adding artifact rather than removing it. In
hindsight these detectors were being asked to find QRS complexes in a window filled with
uncorrected gradient artifact.

**Tuning the CB Correction rate window.** Ground truth showed the default already covered every
subject. A per-subject rate table was produced and is not reproduced here, because the
parameter was never the cause. The two scripts written for these attempts
(`recover_pulse_markers.py`, `pulse_rate_settings.py`) have been removed.

> **Qualified 2026-07-27.** This stands for the *original* failure — the 33 runs with no
> markers were caused by the untrimmed gradient artifact, and no rate value would have
> fixed them. It does not stand as a general statement about the parameter. Once the
> recordings were trimmed, the 80 bpm ceiling turned out to be censoring real beats in 22
> of 90 runs, and raising it recovered five of sub-0003's seven recordings from ~50% to
> ~99% coverage. The earlier check compared *mean* rates against the window and so could
> not see a ceiling clipping the fast tail of the interval distribution. See
> [The 45–80 bpm rate window was censoring real beats](#the-4580-bpm-rate-window-was-censoring-real-beats).

---

## What changed in this repository

Kept, because it would have surfaced this in minutes rather than days.

**Two crash fixes.** Both were a QC review stage treating "this detector resolved nothing" as
fatal, killing a 15-subject run ~45 minutes in and leaving nobody a report:

- `UnusableEcg` in `ica_cardiac_review.py` — `detect_ecg_events` raised on 0 R peaks.
- `UnusableEog` in `ica_ocular_report.py` — `create_eog_epochs` returned empty and `average()`
  raised.

Unresolvable runs are now excluded from the evidence and **named with their reason**; a subject
with no resolvable run has that recorded as the finding rather than raised.

**Beat source corrected.** `detect_ecg_events` now prefers the Analyzer marker train and falls
back to ECG-channel detection, recording which was used. `find_ecg_events` disagreed badly with
the markers — sub-0000 reported **8 bpm and 2 bpm** where the markers say 61 and 60. It now
reports 59–61 bpm, matching the pulse-marker QC to within 0.3 bpm.

**New QC surfacing.** The evidence for this whole problem previously existed only in two
sidecar TSVs that no report read:

- `compute_cardiac_residual` in `report/analyzer_qc.py` — per-run BCG residual, measured
  pre-ICA on the pass `measure_runs` already makes.
- Sidecar columns `pulse_marker_count`, `beat_source`, `bcg_residual_uv` (in-scanner only).
- A cohort worklist in `report/cohort/analyzer.py`: affected runs sorted worst-first, plus a
  per-participant table whose *Which* column collapses consecutive runs (`2-6` versus
  `2, 3, 5`). Audit file `*_desc-cohortuncorrected_qc.tsv`.
- No pass/fail on any run and no residual threshold: marker absence is definitional and stated
  flatly, while how large a residual matters depends on the downstream measurement.

---

## Sources

- [All you ever wanted to know about markers in BrainVision Analyzer 2](https://pressrelease.brainproducts.com/markers/)
- [Peripheral physiology using the BrainAmp ExG MR (2): ECG](https://pressrelease.brainproducts.com/ecg-fmri/) — confirms MR Correction must precede R-peak detection
- [Extend your BrainVision Analyzer 2 to its full potential with Solutions](https://pressrelease.brainproducts.com/analyzer-solutions/)
