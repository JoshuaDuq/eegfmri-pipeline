# Residual scanner gradient

What survived the upstream gradient correction, measured two ways and written as tables
and figures rather than as a report section.

Gradient switching repeats once per scanner volume, so whatever survives upstream
correction appears as a comb of narrow lines at multiples of the volume rate. Each line
is compared with the background measured between it and its neighbours, which separates
periodic gradient residual from everything else occupying the same band. The volume rate
is measured from the markers in each run, not read from configuration.

## Running it

    eeg-pipeline gradient measure    # writes *_comb.tsv, *_locked.tsv, *_declined.tsv
    eeg-pipeline gradient plot       # writes *_comb.png, *_locked.png, *_cohort_comb.png

## What the comb excess is, and is not

Excess is the power at a comb line minus the background beside it, measured within each
channel separately: 0 dB means the peak-window maximum equals the background-window
median. It is not a calibrated statistical null. The median and the largest surviving
line are reported together because gradient residual is focal — it concentrates in the
sensors with the largest lead loops — so a median across the montage can sit near zero
while individual channels are unusable. Marker jitter smears the comb across neighbouring
bins, which lowers the measured excess without the residual itself having changed.

Runs that carry volume markers but no comb measurement are listed in the declined table.
They are listed because a run missing from the comb table has not been measured and found
clean — it has not been measured. Where the reason is upstream filtering, the residual
that survives it no longer repeats once per volume, so the volume-locked table is where
the evidence for those runs is.

## What the volume-locked measurement is, and is not

Observed locked RMS is the root mean square over channels and latencies of the
volume-locked average. It still contains finite-average noise. The odd–even split
estimates that noise floor, and signed excess power is observed RMS squared minus floor
squared. Only a positive excess supports a resolved floor-adjusted amplitude; a negative
value is reported as `unresolved` rather than clipped to zero.

Halves agree is the correlation between the odd-epoch and even-epoch averages the floor
was taken from, and it says which of two very different situations an unresolved row
describes. The split estimates noise only where the locked waveform *cancels* between the
halves, which requires that it be the same waveform in both. Near +1 it is, and the floor
is what it claims to be; near 0 there is no locked waveform for the halves to share. Near
−1 the halves are mirror images, so the waveform cancels in the average and doubles in
the difference: the reported floor is then the residual itself and the excess is negative
by construction. That is what a residual repeating over two volume periods rather than
one looks like, which is what removing every integer harmonic of the volume rate upstream
leaves behind — and it makes an unresolved row mean the opposite of an absent artifact.
No threshold is applied to this column.

It is an envelope, not the artifact waveform: the RMS across channels is non-negative, so
the trace carries magnitude over time and not polarity. A channel whose residual is large
but opposite in sign to its neighbours' raises this trace exactly as one that agrees with
them. Its peak-to-peak range describes the envelope's variation; it is not the
floor-adjusted residual amplitude.

Each volume epoch has its own mean removed before averaging. Gradient switching never
stops, so there is no artifact-free interval inside a volume period to baseline against;
what is removed is the level each channel sits at, which is not part of the volume-locked
waveform but does enter a peak-to-peak taken on a non-negative trace.

The after figure is not guaranteed to be the smaller of the two, and on some recordings
it is not. ICA is fitted to maximise independence over the whole recording, not to
minimise what repeats at the volume rate, so the resolved amplitude can rise across ICA.
That result is worth reading beside the comb table rather than on its own.

## Reading the cohort tables

Each row of the attenuation table is one participant measured twice, so the removed
column is a within-participant difference rather than a difference of two cohort
summaries. Reading it as the latter would carry the spread between participants, which is
the larger of the two and not what this comparison is about.

The timing table's residual is derived from the signed difference between observed locked
power and its odd–even averaging-floor estimate. A non-positive difference is reported as
`unresolved`, not zero. Observed RMS and the estimated floor remain beside it so the
censored measurement is auditable. The odd–even floor uses the same number of epochs as
the observed average, so their comparison does not fall merely because a participant was
scanned for longer.

Reported either side of the exclusions, and paired within the participant. The after
column alone cannot separate a recording whose correction removed a locked residual from
one that never had a measurable residual to remove, and those are different recordings:
only the second tells a reader nothing needs looking at.

Jitter is reported because it works against the comb panel. Irregular volume markers
smear the comb across neighbouring frequency bins, which lowers every measured excess
without the residual itself having changed — so a participant with poor timing can appear
to have the cleanest comb in the cohort.
