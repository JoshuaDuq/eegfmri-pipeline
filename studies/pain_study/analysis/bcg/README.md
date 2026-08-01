# BCG engine: ballistocardiogram gap-fill

**Driven by:** [`../../scripts/cardiac_gaps/`](../../scripts/cardiac_gaps/) ·
**Tests:** [`tests/analysis/bcg/`](../../../../tests/analysis/bcg/) ·
**Findings:** [`docs/pulse_artifact_correction_recovery.md`](../../../../docs/pulse_artifact_correction_recovery.md)

## Why this folder exists

Analyzer marks heartbeats conservatively. What it marks sits on the QRS complex more
reliably than any general-purpose detector measured on this cohort — but it marks too
little. Across the 104 exports, 85 recordings contain at least one RR interval above 2 s,
totalling 6,954 s, with a largest single gap of 53.9 s. An 11 s interval is not a heart
rate, so **the gaps are provable from Analyzer's own markers, with no detector involved.**

That asymmetry — excellent where it marked, absent where it did not — is what every module
here is shaped around.

## The files

| File | What it contributes |
|---|---|
| `sources.py` | Pairs the two Analyzer exports and proves they describe the same samples. The pulse-markers-only export carries the uncorrected artifact and the R marks; the corrected export is what currently feeds BIDS. |
| `detect.py` | Finds the unmarked stretches from Analyzer's own RR intervals, then recovers the beats inside them by QRS template matching. |
| `correct.py` | Removes the artifact at a given set of beats **and nowhere else**. The same call serves the real correction and the sham control, which measures what the procedure destroys when run at beat times where no artifact sits. |
| `markers.py` | Reads and writes BrainVision marker files, so recovered beats can be handed back to Analyzer rather than corrected here. |
| `metrics.py` | The measurements, and the nulls that make them interpretable. |

## Three constraints that are easy to break

**Confinement is as important as removal.** Analyzer's correction at the beats it marked
measures 0.16% residual variance with 0 of 63 channels above null. Touching those stretches
can only make them worse. `correct.py` exists to *not* touch them.

**Handing markers back beats correcting ourselves.** At marked beats Analyzer gets 0.16%
R-locked residual against our 2.03%, and retains 0.54 alpha against our 0.34. So the better
use of a recovered beat is to write it into the marker file at the
`Pulse Artifact Correction (Mark R peaks)` node and let Analyzer do the work. `markers.py`
is not a trivial append: markers are numbered, `[Marker User Infos]` assigns properties *by
marker number*, so inserting one renumbers the rest and invalidates every reference unless
they are remapped.

**No statistic here is raw.** Averaging 500 epochs of ordinary EEG produces several
microvolts of peak-to-peak by itself — on this cohort a naive peak-to-peak read 5.71 µV
against its own null of 6.99 µV, i.e. below chance. Every measurement is held out or
controlled against a circular-shift null.

**No function returns a verdict.** Thresholds belong to the caller.
