# Ballistocardiogram and Analyzer correction evidence

Gradient and pulse-artifact correction were performed in BrainVision Analyzer before this
pipeline ran, so their quality is an input rather than something the MNE stages can fix.
These tables and figures report that input. They are measurements, not verdicts: no run is
excluded by anything here.

    eeg-pipeline bcg measure    # writes *_analyzer_qc.tsv, *_marker_agreement.tsv
    eeg-pipeline bcg plot       # writes *_analyzer_qc.png, *_marker_agreement.png

## The Analyzer correction table

The R-locked EEG amplitude either side of the pipeline's own ICA is given as before and
after: a large *before* value means residual pulse artifact reached this pipeline.

Two kinds of run are named rather than dropped, because a run missing from the table has
not been measured and found clean — it has not been measured:

- **Outside configured bounds.** These runs have pulse-marker measurements outside the
  bounds this QC was configured with. The measured values stay in the table and in the QC
  sidecar; the bounds are reference values recorded beside them, not a threshold any run
  was excluded by.
- **Detected from the ECG channel.** These runs carried no Analyzer R markers, so the R
  peaks used for the measurement came from automated detection on the ECG channel. That
  substitution affects the *measurement* only, not the correction Analyzer applied.

Marker span and marker coverage sit beside each other because they answer different
questions, and the difference between them is the finding. A run can span 99% of its
recording while marking beats in only a fifth of that span, and the span alone reads as
complete.

## The beat-marker agreement table

Two detectors, one heartbeat. Analyzer's R markers drove the pulse-artifact correction
that ran before this pipeline; the detected beats were found here from the ECG signal
itself. The correction can only be as good as the marker train it was given, so the
comparison says whether that train described the heartbeat the ECG recorded.

Beat sensitivity is matched beats divided by ECG-detected beats; marker precision is
matched beats divided by Analyzer markers. The first reveals missed markers and the second
reveals unsupported extra markers. Beat sensitivity is left blank when no beats were
detected, because then there is nothing to take a share of.

The last two columns say what a low share is made of. They give the signed distance from
each detected beat to the nearest marker — positive when the marker came first — as a
median and an interquartile range, over every beat rather than only the matched ones.

**A low matched share does not mean the markers are wrong.** In the bore it usually does
not: an ordinary QRS detector locks onto the magnetohydrodynamic deflection, which is
larger than the R wave and follows it by a few hundred milliseconds, so two perfectly good
trains match at 0%. The lag separates the two readings — a tight lag means a fixed detector
offset, a broad one means genuine disagreement. Read the lag, not the share.
