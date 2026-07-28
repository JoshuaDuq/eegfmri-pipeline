# Cohort Preprocessing Report Design

## Purpose

The preprocessing pipeline renders evidence one subject at a time. A subject report answers
"is this run usable"; it cannot answer "is this cohort one population, was the cleaning
applied consistently, and did the group keep any brain signal". Those are the questions that
decide whether a group analysis is viable, and they are answered by a document this pipeline
does not currently produce.

This spec defines a `cohort-report` command that aggregates the per-subject evidence into a
single MNE Report, and the per-subject sidecar that makes the aggregation possible without
re-reading a gigabyte of filtered raw per subject.

The report describes a cohort. It does not test hypotheses about one, and it does not grade
subjects. Every number is a measurement with a stated denominator.

## Scope

In scope: aggregation of preprocessing QC evidence already computed by the subject report
stage, across subjects, for EEG-only and EEG-fMRI acquisitions and for task and resting-state
paradigms.

Out of scope: changing any per-subject estimator; source modelling; any analysis of the
experimental effect. One per-subject estimator weakness is recorded below under "Known
limitations" rather than fixed here.

## Architecture

### The seam

`run_evidence.add_run_evidence_review()` already makes the single expensive pass over each
run — read, `ICA.apply`, measure — producing `RunSpectra`, `CombResidual`,
`VolumeLockedAverage`, `RunContinuity`, `MarkerAgreement` and `RrIntervals`. Today those
objects are plotted and discarded. This design serialises them at the moment they are already
in memory, so the cohort command reads tables instead of re-deriving them.

Two properties follow. The cohort figure is provably the same measurement as the subject
figure, because there is one computation and two readers. And the cohort command is cheap
enough to re-run freely.

### Per-subject QC sidecar

Written beside the subject report, following the derivative naming the report already uses.

| File | Contents |
| --- | --- |
| `sub-XXXX_desc-qcsubject.json` | Subject-level scalars, acquisition context, settings, per-stage package versions |
| `sub-XXXX_desc-qcruns.tsv` | One row per run: every run-level scalar, including the volume-locked corrected amplitude, the noise floor it was measured against, the repetition time, the volume count, and the continuity reductions |
| `sub-XXXX_desc-qcspectrum_curves.tsv` | `run, stage, freq_hz, median_db, max_db` |
| `sub-XXXX_desc-qcchannels.tsv` | `channel, x, y, z, n_runs_bad` — positions from the recording, never from a montage name |
| `sub-XXXX_desc-qccomb_curves.tsv` | `run, harmonic_index, harmonic_hz, before_excess_db_{median,max}, after_excess_db_{median,max}` |
| `sub-XXXX_desc-qcconditions.tsv` | `condition, n_total, n_kept` — trials presented and trials retained, for a task acquisition |

The conditions table carries both counts rather than a retained fraction, which is the one
place two numbers are stored where one derived quantity would do. The reason is that the
fraction is not the quantity: forty retained trials is a well-sampled condition or a badly
decimated one depending on how many were presented, and the presented count is what the
events section reads while the retained count is what the rejection section reads. Neither
is derivable from the other, and the fraction is derivable from both — so the fraction is
what goes unstored.

Curve tables hold curves and the run table holds scalars, so the volume-locked correction —
which is one number per run, not a waveform — lives in the run table beside the volume count
it was corrected for.

The volume-locked *waveform* is deliberately not carried, and there is no cohort panel for
it. Pooling waveforms across participants means pooling curves that each carry their own
noise floor at every point, which is precisely the confound the scalar was introduced to
remove; and with mixed repetition times there is no common time axis to pool them on. Where
in the volume period a residual sits is a subject-level diagnostic and stays on the subject
panel. Nothing derived is stored either: the uncorrected RMS is
`sqrt(corrected^2 + floor^2)` and detectability is their ratio, so storing either would store
one measurement twice and invite the copies to disagree.

Required columns are validated on read, and on write before anything reaches disk, because a
column silently absent reaches a cohort figure as a panel that quietly lost half its
participants. Columns required only of in-scanner acquisitions are validated against the
recorded context rather than unconditionally. The distinction is between a column and a
value: a column is required when the recorded context implies the measurement was attempted,
so its absence is a fault in the writer, while the value in it may still be missing because a
measurement that was attempted and did not resolve is an ordinary outcome. A missing value
shrinks a denominator the panel prints; a missing column would shrink it silently.

The sidecar stores what a cohort can legitimately aggregate, not everything measured.
`RunContinuity` holds a time-by-channel matrix that no cohort can meaningfully average, so
the sidecar reduces it at write time to per-run scalars (flagged-time fraction, median and
maximum relative dB, bad-span count). Reducing at write time rather than read time is what
prevents the cohort command from inventing an aggregation the subject stage never sanctioned.

### Acquisition context

Derived per subject from its own evidence and recorded in the sidecar, never configured:
volume markers present means in-scanner; an events file with more than one condition means
task; otherwise resting state. The cohort command reads the recorded context and never
re-derives it.

### Module layout

```
eeg_pipeline/preprocessing/report/cohort/
    sidecar.py       schema and IO for the per-subject sidecar
    record.py        turning one subject's measured evidence into that sidecar
    noise_floor.py   the odd-even split behind the corrected locked amplitude
    collect.py       subject discovery, sidecar loading, stratification
    aggregate.py     participant-first pooling, quantile gating  (pure numerics)
    composition.py   cohort composition and denominators
    multiplicity.py  outer-decile counts per metric family, with denominators
    at_a_glance.py   the subject panel's headline list, each value a distribution
    homogeneity.py   version, settings and drift agreement
    coverage.py      bad-channel distributions and frequency topography
    rejection.py     epoch retention, reasons, per-condition balance
    ica.py           decomposition quality and label composition
    analyzer.py      marker agreement and RR physiology
    gradient.py      harmonic comb and volume-locked residual
    spectra.py       cohort PSD and aperiodic parameters
    preservation.py  alpha, reliability, cleaning-versus-signal
    continuity.py    flagged-time heatmap
    events.py        trials per condition
    provenance.py    the document's account of how it was built
    report.py        section assembly, ordering, audit tables, cohort log
eeg_pipeline/cli/commands/cohort_report{,_parser,_orchestrator}.py
```

The sidecar is written by `PreprocessingPipeline._write_qc_sidecar`, at the end of the
report-review stage — the point at which `measure_runs` has just made its one expensive
pass and its results are still in memory. `measure_runs` additionally records the sensor
positions, the per-run bad channels and the acquisition date while each recording is open,
none of which is a measurement and all of which a cohort needs.

Two departures from the design above, both narrowing rather than widening it. The
per-trial `residual_ecg_coupling` is not aggregated: it lives in the events table rather
than in the run evidence, so carrying it would mean the cohort command reading a second
file per participant for one panel. And the drift panels of section 2 are not drawn — the
agreement tables state what differs, and a trend read off a few dozen points is the thing
this report declines to do everywhere else.

`aggregate.py` is pure numerics with no plotting and no MNE import. Every statistical claim
in the report is therefore verifiable without rendering a figure. This is the testability
seam of the whole feature.

Reuses `report/style.py`, `report/tables.py`, `report/aperiodic.py`, and wraps the existing
`report/cohort_qc.py` rather than duplicating its R-marker roll-up.

## Aggregation rules

### Unit of inference

The participant. Sessions collapse to the participant before any cohort statistic, so a
two-session subject is one participant. Each participant carries equal weight regardless of
how many runs it contributed.

### Run to subject

The rule depends on what kind of quantity it is, because a single rule is wrong for at least
one of them.

| Metric type | Rule | Reason |
| --- | --- | --- |
| Curves (spectra, comb) | Median across runs, equal weight | Each run is an independent estimate of the same quantity |
| Fractions and rates (flagged time, epochs retained, marker agreement) | Sum of numerators over sum of denominators | The fraction is defined over the whole session; an unweighted median lets a 30-second run outvote a 12-minute one |
| Counts (bad channels) | Union across runs, with the policy recorded | Matches the existing `bad_channel_sync_policy` field |

### Median rather than mean, in decibels

Power is pooled in decibels. Inter-participant EEG power is approximately log-normal, so
decibels are the scale on which the spread is symmetric and a quantile band describes the
cohort rather than its tail. That is the primary argument.

The median is additionally near-indifferent to that choice, which the mean is not. The median
commutes with a monotone transform, so `median(dB(x)) == dB(median(x))` — but *exactly only
at odd participant counts*. At even counts `numpy` averages the two central order statistics,
and an arithmetic mean does not commute with a logarithm: pooling `{1, 100}` linearly gives
50.5, or 17.0 dB, where pooling their decibels gives 10.0 dB. The deviation is bounded by the
gap between the two central participants and vanishes at odd counts; a mean, by contrast, is
scale-dependent at every count and by much more. Both facts are pinned in
`test_cohort_aggregate.py` so the caveat cannot be quietly dropped.

### Uncertainty, and what is deliberately not drawn

No confidence intervals appear in this report, and no bootstrap is performed.

A confidence interval on the cohort median answers how precisely the group's central value is
known. That is not the question a QC report asks. The QC question is where a subject sits
among the others, which the empirical distribution answers directly. Drawing a confidence
interval where a tolerance interval is meant is a conflation, and percentile-bootstrap
coverage below roughly twenty participants is poor besides.

Spread is therefore shown as empirical quantile bands, gated on whether the quantile is
supported by the data at all. Under Weibull plotting positions the p-th quantile lies inside
the observed order statistics only when `1/(n+1) <= p <= n/(n+1)`, so p = 0.25 requires n >= 3
and p = 0.10 requires n >= 9. Conservatively:

- `n < 5` — individual points and traces only; no median, no band, no summary language.
- `n >= 5` — median and interquartile band.
- `n >= 10` — additionally the 10th-to-90th band.

Both gates are configurable, and every panel states which regime produced it.

The bands are estimated with the same plotting-position convention the gate is argued
from — `numpy`'s `method="weibull"`, not its default `"linear"`, under which the p = 0.10
"quantile" of ten participants is an interpolation a tenth of the way from the smallest to
the second smallest and exists for as few as two. Gating on one convention while estimating
with another leaves the gate describing something other than the band it produced.

A participant contributing a missing value is refused by the pooling functions rather than
absorbed, because `numpy`'s median returns a missing value wherever one appears: a single
gap at one frequency would erase the cohort curve at that frequency while the denominator
printed beside the figure stayed unchanged, which is precisely what the denominator exists
to prevent. Deciding what to do about such a participant is policy rather than arithmetic,
so the caller drops it and the panel's own denominator shrinks visibly with it.

This diverges deliberately from `2026-07-14-native-cohort-scanner-spectrum-qc-design.md`,
which uses a participant bootstrap. That pipeline is unchanged; the divergence is recorded so
it is not read as an inconsistency.

### Paired comparisons

Every before-and-after-ICA comparison is paired within subject. Two side-by-side box plots
discard the pairing and inflate the apparent spread, so each such comparison is drawn as a
per-subject line from before to after, alongside the distribution of within-subject
differences, which is the estimand. This applies to comb excess, the volume-locked ratio,
alpha prominence, aperiodic exponent and offset, and the spectra.

### No inferential statistics

No p-values, no hypothesis tests, no correlation coefficients and no fitted lines anywhere,
including on the cleaning-versus-signal scatter and the drift panels. A trend read from
fifteen points is not a finding. Everything a reader might wish to test is in the audit TSVs.

### What may still be flagged

A distinction the report encodes explicitly. *Algebraic* violations may be stated as
violations, because they are definitional rather than empirical: more ICA components than the
data rank supports, a fraction outside `[0, 1]`, a negative count. *Empirical* thresholds may
not be, because the cutoff would be invented. The report flags the first kind only.

### Outliers

No deviation score and no cutoff. Robust z-scores presume approximate normality, which is
false for the bounded fractions and small counts this report is full of, and the median
absolute deviation is exactly zero whenever more than half the participants share a value.
Extremes surface through sorted tables and through every participant being individually
labelled on every distribution — by position, never by assertion.

The same tie problem that rules out the median absolute deviation also rules out a
threshold test against the tenth percentile, which the multiplicity table below needs. On a
bad-channel count where eight of twelve participants have none, that percentile *is* zero
and all eight read as extreme — ten of twelve flagged by a panel written to prevent exactly
that. Outer-decile membership is therefore taken by rank: the `ceil(0.1 n)` most extreme
participants at each end, bounded by construction, with a tied group at the boundary
admitted only if all of it fits inside that budget. Half a tied group would be a statement
about the order the participants were read in.

### Multiplicity

With roughly forty panels and a few dozen participants, some participant is extreme on
something by chance. A report that highlights extremes manufactures suspicion, so it carries a
standing note that extremeness on a single metric is expected and that convergence across
mechanistically related metrics is what warrants a look. This is backed by a table giving, per
participant, how many of the metrics in each family (channel-level, ICA-level, scanner-level,
physiology-level) place them in the outer decile, out of how many. A count with a denominator,
with no cutoff applied to the count. It renders only at `n >= 10`, below which an outer decile
is undefined.

### Denominators

Every aggregate returns its value together with `n_subjects`, `n_runs` and the contributing
participant identifiers. A panel cannot render a number without its own denominator, so the
document cannot claim `n = 20` in the header while a panel silently used fourteen.

### Frequency grids and filter ranges

Spectra are never interpolated onto a common grid, because interpolation smears peaks.
Identical bins are required after restriction to the common range, and a mismatch is an error
naming the offending participants. The cohort spectrum is additionally restricted to the range
every contributing participant's filter supports — `RunSpectra.fmax_reason` already records
this — and the limiting participant is named on the panel.

## Measurement corrections

Four quantities are not comparable across participants in the form the subject report uses
them. Each is corrected in the sidecar, at write time.

### Volume-locked residual is confounded by volume count

`compute_volume_locked_average` averages `n_volumes` epochs and reduces to an across-channel
RMS. The noise floor of an average over N epochs falls as `1/sqrt(N)`, so a participant with
three hundred volumes reports a smaller residual than one with a hundred for reasons that have
nothing to do with correction quality. Pooling the raw microvolt peak-to-peak across
participants would measure run length.

The floor is measured by splitting the epochs odd against even. Writing `s` for the locked
waveform:

```
A = mean over all epochs      ->  s          + noise, power sigma^2 / N
D = mean(odd) - mean(even)    ->  s cancels  + noise, power 4 sigma^2 / N
```

The locked waveform is identical in both halves, so it cancels exactly in `D`, which
therefore measures the noise alone at a known scale. Hence

```
mean(s^2) = mean(A^2) - mean(D^2) / 4
```

is exact, needs no random draws, no seed and no correction term, and costs one pass over
epochs the caller has already extracted in order to average them. Alternating rather than
splitting at the midpoint, so that slow drift falls equally on both halves.

Two earlier drafts of this section were wrong and are recorded because the errors are
instructive. The first proposed pooling the *ratio* of the locked average to a random-offset
surrogate, calling it N-corrected; it is not, because a surrogate average retains
`mean(s^2)/N` of the artifact's own power, making the ratio grow as `sqrt(N)` — the same
confound reversed. The second fixed that by subtracting, but reached the answer through eight
Monte-Carlo passes over the recording per stage per run, to estimate something the odd-even
split gives exactly and for free. The ratio survives only as a per-participant *detectability*
annotation, explicitly labelled N-dependent and never aggregated.

`test_cohort_noise_floor.py` pins this against synthetic data with a known injected artifact:
the raw figure falls by nearly half between 50 and 400 epochs while the corrected amplitude
recovers the injected value at both; a recording with no locked artifact corrects to zero
where the raw figure reports a plausible microvolt; and a hundredfold larger artifact leaves
the measured floor unchanged, confirming that the split cancels the signal rather than
absorbing it.

Because the estimator rides on the average `compute_volume_locked_average` already builds,
the correction lives in `scanner.py` beside it and costs the measuring pass nothing. It also
improves the subject panel, which until now printed a residual amplitude with no way for a
reader to tell how much of it was the averaging floor.

### The harmonic comb has no common frequency axis

Harmonics sit at `k / TR`. Where participants differ in repetition time, pooling by frequency
mixes different harmonics into one bin. The comb is therefore aggregated by harmonic index,
and plotted against frequency only when every contributing participant shares a repetition
time within tolerance. The volume-locked waveform is normalised to fraction of TR under the
same condition, which is why the sidecar carries both `time_s` and `time_fraction_tr`.

### Split-half reliability is not comparable across trial counts

Reliability grows with test length, so a participant with forty retained trials is not
comparable with one with a hundred and twenty. Every participant's reliability is projected by
Spearman-Brown to a common reference trial count — the cohort minimum, printed on the panel —
and both the observed and the projected value are reported.

### Alpha peak frequency is fabricated when there is no peak

`PosteriorAlpha.has_peak` is `prominence_db > 0`, which the argmax of excess-over-aperiodic
satisfies almost always, so a participant with no rhythm still receives a plausible-looking
peak frequency. A cohort histogram built from those is fabricated structure.

Peak frequency is therefore pooled only for participants whose peak clears the scatter of
the band it was found in. Two corrections were needed to make that test do what it says,
and both were found by running it on rhythm-free recordings:

*The prominence is a maximum, and has to be judged as one.* It is the largest excess over
the fitted background across every bin of the alpha band — of the order of seventy bins at
the resolution used here. The largest of seventy normal residuals exceeds two standard
deviations about eighty per cent of the time, so a criterion of `k = 2` admitted noise four
times in five. The threshold is therefore the Gumbel level a maximum over `n` bins reaches
with probability `rate` under the null, which scales with how wide a band was searched and
how finely it was resolved rather than being accidentally strict on a narrow band and
vacuous on a wide one.

*It has to be judged against the right scale.* The aperiodic fit residual is measured where
the line was fitted, so the fit has absorbed part of it; the excess is measured where that
line is extrapolated across the excluded alpha window, where it also carries extrapolation
error. On rhythm-free recordings the prominence runs at about 3.8 times the fit residual,
so multiples of the residual are calibrated against the wrong yardstick. The peak is
therefore scored against the median and robust scatter of the very bins it was the maximum
of, which puts numerator and denominator on one footing. Robust, so a real rhythm occupying
a few bins of seventy does not inflate the scale it is about to be measured against.

Together these take the false-positive rate on rhythm-free simulated recordings from 81% to
about 8%, while a rhythm at a tenth of the noise amplitude is still resolved in every case
and its frequency recovered. Participants below the threshold are counted as having no
resolvable peak and given no frequency. That count is itself a cohort measurement and is
reported.

Two further guards came out of running this on real recordings, and both concern *where*
the peak is rather than whether there is one.

*An argmax on the band edge is not a peak.* It is the highest point of whatever was inside
the window, and the spectrum may go on rising outside it. Resolvability therefore requires
an interior local maximum.

*A band with two comparable bumps has no peak frequency.* One participant's peak sat at
13.4 Hz before cleaning and 9.9 Hz after — an apparent 3.4 Hz shift in someone's alpha
rhythm. The band held two bumps 0.5 dB apart, against a spectral roughness of 0.59 dB, and
cleaning nudged which of them was taller. Nothing moved; the argmax changed its mind. The
report therefore records the runner-up and its gap, calls the frequency contested where the
gap is smaller than the spectrum's own roughness, and names any participant whose frequency
appears to have shifted while a contest was open at either end. Recording a bistable number
in the cohort's peak-frequency distribution would make that distribution read sharper than
the evidence supports.

The pairing this feeds is measured in `measure_runs`, on the continuous run either side of
the exclusions, because that is the only point in the pipeline where the same data exists
in both states — there is no pre-ICA epochs file, so the preservation section's own
epochs-based alpha has no counterpart and cannot answer whether cleaning cost a participant
their rhythm. One estimator produces both sides, so the pairing is a difference between
stages rather than a difference between estimators.

### An exponent shift means opposite things in and out of a bore

The aperiodic panel was written to say that a cohort whose exponent shifted in one
direction "has lost something broadband". Run against real in-scanner data, both
participants' exponents fell sharply — 1.52 to 0.92 and 1.57 to 1.29 — and the panel would
have sent the reader looking for signal loss that is not there.

The exponent is a slope, so it can only move if removal was uneven across frequency.
Measuring how uneven settles it. On these recordings ICA took 11.0 dB out of 2–6 Hz and
4.6 dB out of 30–45 Hz for one participant, 7.2 and 2.4 dB for the other: removal
concentrated at the bottom of the band, which is what removing a ballistocardiogram must
look like, and which flattens the slope by arithmetic. Nineteen of about twenty-eight
excluded components were labelled cardiac. The larger tilt belongs to the larger shift.

So the report carries the tilt beside the shift — power removed in a low band, in a high
band, and the difference — and reads the shift against the recorded acquisition context.
Inside a scanner a downward shift is the signature of a correction working; outside one,
where there is no dominant low-frequency artifact, the alarming reading is the ordinary
one. This is the same refusal to pool across context that the variance-removed panel
already makes, applied to the quantity whose interpretation depends on it most.

### Aperiodic fits inside a scanner

`DEFAULT_FIT_RANGE_HZ` of 2 to 45 Hz sits below the line-noise fundamental but inside the
gradient comb, whose search band starts at 15 Hz. The peak-residual trim removes some harmonic
teeth but not reliably — and it trims them against a line the comb has already tilted. For
in-scanner participants the harmonic bins and their immediate neighbourhoods are therefore
passed to `fit_aperiodic` as `excluded_windows`, which the function already accepts. The
volume rate is measured before the spectra rather than after, so the fit knows where the
comb is; measuring the marker train twice would be two chances to disagree about that.

Measured effect on the real two-participant run, at a 0.9 s repetition time putting 39
harmonics inside the fit range: the fit r² improves and one participant's pre-ICA exponent
moves by 0.06. The correction is small here because the teeth are narrow relative to the
344 bins in the range, and it grows as the repetition time lengthens and the comb closes
up — which is the case it exists for. It does *not* account for the exponent shift recorded
above; that was checked explicitly and the shift is 0.51 either way.

## Figures

Seven, and the count is a constraint rather than an outcome. Every figure in this report has
to show something the table beside it cannot; a panel that restates a column costs the
reader attention and buys nothing.

| Figure | What only it shows |
| --- | --- |
| Gradient comb, before and after | Which harmonics survived, per participant |
| Cohort spectra, before and after | Gross spectral outliers, line noise in context |
| Aperiodic exponent, paired | Whether the background shifted in one direction across the cohort |
| Alpha prominence, paired | Which participants lost their rhythm to cleaning |
| Cleaning against signal | The joint distribution, unavailable from either margin |
| Excluded components by detector | Proportions a table of totals cannot convey |
| Where the cap failed | That the failed electrodes are *neighbours* — one dead region reads identically to four scattered contacts in any ranked list |
| Quality by participant and run | A two-dimensional grid: reading down a column is a fact about the session design that no per-participant column and no per-run column can carry |

Two of these are conditional, because the thing they show is not always there. The
topography is drawn only where some electrode actually failed; with none, every sensor is
one colour under a scale spanning nothing, which reads as a failure too small to see rather
than as no failure at all. The trial-retention panel — a participant's conditions drawn
inside their own range — is drawn only where a participant has more than one condition,
since a single condition has no inside and the panel collapses to the retained column of
the table beside it.

Six candidates were built and cut, and the reasons generalise:

- **Posterior alpha spectrum.** The prominence panel quantifies the bump and the sensor
  spectrum already shows spectral shape. Cutting it removed a whole sidecar table with it.
- **Aperiodic offset, paired.** The offset falls whenever variance is removed, near enough
  by construction, so the panel restated the variance-removed column. The exponent says
  whether the background changed *shape*, which is the part that distinguishes artifact
  from signal having been taken out.
- **Components fitted against rank.** In the normal case it shows a null result, and in the
  abnormal case the prose already names the participant. Both numbers stay in the table.
- **Cohort volume-locked waveform.** Recorded above under the measurement corrections: it
  reintroduces the noise-floor confound the scalar exists to remove.
- **Marker agreement and heart rate, as strip plots.** Built while filling in the Analyzer
  section and cut on the same rule that cut the rest. Every quantity that section measures
  is one number per participant, and a strip plot of one number per participant is the
  sorted table drawn with dots — carrying strictly less, since the table also holds the
  worst run, the dropout rate and the run count. The ordering the figure existed to show is
  the order the table is already in.
- **Trial retention as a bare strip plot.** The same failure in the single-condition case,
  which is why the panel survives only where there is a spread to draw. A figure that is
  informative for one shape of design and a restatement for another has to test for the
  shape rather than always render.

The general rule these keep arriving at: *one scalar per participant is a table*. A figure
earns its place by carrying a second dimension — frequency, harmonic index, run position,
head position, a paired stage, or a second measurement to be joint with.

## Report contents

Sections carry the same names as the subject report so a reviewer moves between the two
documents without relearning the layout. A section is absent, not empty, when no participant
supports it, and each panel states its own denominator.

0. **Cohort composition.** The denominator table. Per participant: context, runs, channels,
   sampling rate, duration, pipeline version, acquisition date. Participants found but not
   aggregated, for want of a sidecar, are listed here rather than silently dropped.
1. **At a glance.** The subject report's headline list, each value as a distribution.
2. **Preprocessing homogeneity.** Package versions per stage across participants with
   disagreements surfaced; settings differences; and two separate drift panels — metrics
   against acquisition date, which indexes cap ageing and electrode wear, and metrics against
   pipeline version, which indexes processing change. Conflating those two axes was an error
   in an earlier draft. Neither panel carries a fitted line.
3. **Channel and region coverage.** Bad-channel counts per participant, and the fraction of
   participants for whom each electrode was bad, drawn as a topography with a per-electrode
   denominator. That figure separates a systematically failing electrode — a cap defect, a
   bridged site, a lead near the bore — from idiosyncratic ones, which no subject report can
   show. Electrodes not present in every participant carry their own denominator, and the
   topography is refused in favour of a table where montages differ.
4. **Epoch rejection** (task only). Retained fraction per participant and drop reasons, plus
   retained fraction *per condition* and the within-participant range across conditions.
   Differential rejection confounds a contrast rather than merely underpowering it, and a
   total-epoch count cannot show it.
5. **ICA decomposition quality.** Components fitted, excluded and dimensions retained;
   `samples_per_squared_component` against its configured reference; rank against channel
   count and condition number. Variance removed is stratified by acquisition context, because
   a value ordinary inside a bore is alarming outside one and a pooled median describes
   neither. Excluded-component label composition per participant comes from
   `_proc-ica_components.tsv`.
6. **Scanner artifact correction (Analyzer)** (in-scanner only). The existing R-marker
   roll-up; marker-agreement distributions; RR-interval physiology as median rate, SDNN and
   the fraction of physiologically implausible intervals; and the per-trial
   `residual_ecg_coupling` already present in the events table, which is a direct
   post-cleaning measure of what the pulse correction left behind.
7. **Residual scanner gradient** (in-scanner only). Cohort comb excess by harmonic index
   before and after ICA; per-participant attenuation; the floor-corrected volume-locked
   amplitude, paired before against after; and repetition-time and jitter consistency, since a timing
   outlier means the volume markers are wrong and the correction upstream of everything else
   in this section is invalid.
8. **Sensor spectra.** Cohort median before and after ICA, participant-first, stratified by
   context, with participant traces overlaid faintly so outliers are visible against the group.
   Line-noise and gradient harmonics marked. Aperiodic exponent and offset before against
   after, paired: a cohort whose exponent shifted systematically through cleaning has had
   broadband signal removed, and nothing else in the report would show it.
9. **Signal preservation.** Posterior alpha as cohort median spectrum, prominence before
   against after, and the peak-frequency distribution over participants with a resolvable
   peak. Split-half reliability, observed and projected to a common trial count, for task
   cohorts; first-half against second-half spectral reliability for resting-state cohorts.
   And a cleaning-versus-signal scatter, one point per participant, variance removed against
   alpha prominence retained, with no fitted line and no coefficient. Prominence is the only
   valid ordinate: absolute alpha power is mechanically coupled to variance removed, so a
   scatter against power would draw an arithmetic relationship and present it as an empirical
   one.
10. **Data quality over time.** Flagged-time fraction as a participant-by-run heatmap, which
    reveals that the last run is always the worst — a study-design finding rather than a
    subject one.
11. **Events** (task only). Trials per condition per participant, timing consistency, missing
    conditions.
12. **How this report was built.** Cohort provenance mirroring the per-subject build record.

A resting-state EEG-only cohort renders sections 0, 1, 2, 3, 5, 8, 9, 10 and 12, and nothing
else.

## Command and outputs

```
eeg-pipeline cohort-report --subjects 0014 0015 --task thermalactive
```

`requires_subjects=False`; the default is every participant with a sidecar. Outputs land in
`derivatives/preprocessed/eeg/group/`:

- `task-<task>_desc-cohort_report.html`
- one audit TSV per figure, holding exactly the plotted values
- `task-<task>_desc-cohort_log.json`, recording contributing participants, per-panel
  denominators, seeds, gate settings and per-stage package versions

## Failure modes

| Condition | Behaviour |
| --- | --- |
| Participant without a sidecar | Listed in composition as not aggregated; report proceeds |
| Incompatible frequency grids | Error naming the participants |
| Mixed pipeline versions | Rendered in homogeneity; not an error |
| `n == 1` | Builds, individuals only, no summary language |
| Metric no participant supports | Section absent, not empty |
| Differing montages | Topography refused, table rendered instead |

## Testing

`aggregate.py` is pure numerics, so every statistical claim is testable without rendering:
equal participant weight under unequal run counts; the rate-versus-curve pooling rules; gate
boundaries at n of 4, 5, 9 and 10; refusal of unsupported quantiles; median and dB transform
invariance; invariance of the floor-corrected locked amplitude to epoch count on synthetic
data with a known injected residual; Spearman-Brown projection; grid-mismatch errors.

Beyond that: sidecar round-trip; stratification on a synthetic mixed cohort; section-drop on a
resting-state cohort; and the real two-participant run on sub-0014 and sub-0015.

## Known limitations

`compute_split_half_reliability` uses a single odd-and-even split. The split-to-split variance
of that estimator can rival the between-participant variance the cohort panel exists to show.
The fix is a seeded mean over many random splits, which changes the per-subject estimator and
therefore belongs to a separate change; this design consumes whatever the subject stage
provides and projects it to a common trial count. Recorded so the cohort panel is read with
the right level of confidence.

Separately, sub-0014 records `worst_marker_agreement` of 0.0 while its pulse-marker QC reports
a marker fraction near 0.97 with a passing status. Those describe different things and may
both be correct, but the disagreement is unexplained. This report surfaces it; it does not
resolve it.
