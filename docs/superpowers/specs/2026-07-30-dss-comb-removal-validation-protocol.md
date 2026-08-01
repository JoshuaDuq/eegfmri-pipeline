# Validation protocol: subspace removal of the room/mains comb

**Frozen 2026-07-30, before the cohort sweep was run.** Every threshold and decision rule
below is declared here so that the sweep cannot become both the tuning set and the
validation set. A result that fails these criteria means the method is wrong for this
data, not that the criteria should move. Any later amendment must appear as a separate
commit stating what changed and why.

## What is being tested

Whether the room/mains comb can be removed by projecting out a spatial subspace, instead
of subtracting a sinusoid at every target bin. The existing `spectrum_fit` removal drives
target bins to roughly 15 dB below their local background because it fits the *total*
amplitude at each bin -- artifact plus neural -- and subtracts all of it. Subspace removal
should leave neural activity that sits at comb frequencies but outside the artifact's
spatial subspace, returning those bins to background rather than hollowing them out.

This is not the scanner. Earlier work on this branch established the comb as a
mains-synchronous room artifact independent of gradient correction, and the terminology
matters here: an environmental field coupling through lead geometry has different
spatial-stationarity properties than gradient-coupled pickup, and the estimator depends on
that stationarity.

## Why the protocol is frozen before the sweep

A pilot on five runs produced an in-sample comb-to-background contrast of 11.9x to 45.7x.
Scored on held-out windows the median fell to 8.87x, and on one run (sub-0000 run-1, the
run with the *highest* in-sample contrast) it collapsed to 1.57x. In-sample eigenvalues of
a generalised eigenvalue problem estimated from few windows are inflated by construction.
No number produced without held-out scoring is admissible evidence in this work.

The pilot also used interleaved (even/odd) window splits, which share local
nonstationarity between the fitting and scoring halves. The 8.87x figure is therefore
optimistic, and this protocol replaces interleaving with contiguous blocks.

## Statistical design

The dataset is 15 participants x 6 `thermalactive` runs = 90 runs, balanced.

**The statistical unit is the participant.** Runs within a participant share a session,
cap placement and room state, so they are not independent. All uncertainty is computed
across the 15 participants, never across the 90 runs.

Two separate levels of resampling, which must not be confused:

- **Within-run, for fitting the projection.** The run's windows are divided into three
  *contiguous* blocks. The subspace is estimated on two blocks and scored on the held-out
  third, rotating over all three. Contiguity is required so that warming drift, motion and
  other nonstationarity cannot leak between fitting and scoring.
- **Across participants, for choosing hyperparameters.** Any quantity shared across runs
  -- the rank-selection rule, the competence thresholds -- is chosen under
  leave-one-participant-out cross-validation over the 15 participants. A hyperparameter is
  never selected on a participant it is then evaluated on.

## Variants compared

Both under identical cross-validation, no exceptions:

1. **Pooled** -- one subspace estimated from all comb harmonics in 28-95 Hz.
2. **Frequency-group** -- separate subspaces for 28-45, 45-65 and 65-95 Hz.

The pilot found per-group estimates unstable, but instability of one variant is not
evidence that the other is superior. Both are measured the same way and the comparison is
decided by held-out removal and preservation, not by subspace geometry.

## Rank selection

For each run and variant, choose the **smallest** rank *k* that jointly satisfies all
three criteria below on held-out blocks. Not the rank with the best residual: the smallest
rank that suffices, because every removed dimension is a dimension unavailable to
subsequent analysis.

If no *k* satisfies all three, the run is **abstained** (see below).

## Acceptance criteria

All measured on held-out blocks. Thresholds declared before any sweep result exists.

**Target activity.** A target counts as *active* in a run if its pre-clean prominence
exceeds the robust null of the prominence spectrum by 3 sigma, using the existing
`robust_null` estimator rather than a new constant. Active and inactive targets are judged
separately.

1. **Artifact residual (active targets).** Two-sided, against the *pre-clean* local
   background so the projection cannot move a line and its own reference together:
   - 90th percentile of `|residual prominence|` <= 1.0 dB
   - absolute maximum `|residual prominence|` <= 3.0 dB

   Two-sided is the point. The old gate's `max_residual_prominence_db <= 1.0` was
   one-sided, so a residual of -17.4 dB passed a check nominally about clean lines. The
   old `min_median_suppression_db >= 10.0` is dropped entirely: when a line sits only
   2-10 dB above background, demanding 10 dB of suppression mandates over-removal. That
   criterion is what produced the troughs. Suppression is still reported, as a diagnostic.

2. **Inactive targets.** `|change|` <= 0.5 dB. Guards against gate dilution: a target that
   was never an artifact must not be quietly modified.

3. **Co-frequency probe preservation.** See the probe bank below. For probes whose
   principal angle to the estimated artifact subspace is >= 60 degrees, retained energy
   ratio >= 0.90. For smaller angles, retention is **reported, not gated** -- a neural
   source collinear with the artifact cannot be recovered by any linear projection, and
   pretending otherwise would be dishonest about an identifiability limit rather than a
   defect in the method.

4. **Off-target preservation.** Outside comb neighbourhoods: PSD and cross-spectral
   density change <= 0.2 dB, matching the existing `max_nonline_change_db`. Fourier
   coefficients at off-target bins unchanged to numerical tolerance (1e-6 relative).

   The earlier claim that off-target data is "bit-identical" is withdrawn and must not
   reappear. A spatial projection is memoryless mixing across channels, so it preserves
   bandlimitation and off-target Fourier coefficients survive -- but the correction signal
   is nonzero at every *time* sample. No time sample is bit-identical.

5. **Rank ceiling.** `rank_removed / effective_rank` <= 0.33, where effective rank is the
   numerical rank after referencing and bad-channel handling, not the nominal channel
   count. The pilot found 54 in one run and 64 in the others, so nominal count would
   understate the fraction.

   This is a conservative engineering ceiling, not a preservation guarantee, and is
   labelled as such. Rank fraction does not measure information loss: removing one
   component that carries the source of interest is worse than removing ten empty
   directions. Criterion 3 is the preservation gate; this one only bounds gross cost.

## Amendment 1 (2026-07-30, before the cohort sweep)

A smoke test on five runs showed the estimator specified above does not work, and the
correction changes the method rather than any threshold. Recorded here in full.

**The subspace was ranked by the wrong quantity.** Whitening the comb covariance by the
background covariance and taking the leading generalised eigenvector finds the direction
with the best comb-to-background *ratio*. That direction can carry almost none of the
comb's absolute energy: a tiny background with a modest comb yields a huge ratio. Measured
on sub-0008 run-3, projecting out the top 12 ratio directions moved the median line
prominence from 7.50 dB to 7.37 dB -- it removed nothing, while reporting a held-out
contrast of 49x.

The artifact subspace is instead the leading eigenspace of the **excess covariance**
`C_comb - C_control`. This is the spatial form of the principle already adopted for
measurement on this branch: the artifact is the excess over background, not the whole
content of the bin. Same run, same folds: 7.50 dB falls to 2.86 dB at rank 3.

**Integer rank cannot land on background.** The same run gives +2.86 dB at rank 3 and
-2.38 dB at rank 4. No integer rank lands inside the +/-1 dB criterion, so a binary
projection reintroduces the trough in spatial form -- the exact failure this work exists
to remove.

Each direction is therefore scaled rather than zeroed, by the Wiener-style gain
`sqrt(background_power / comb_power)` clipped to 1, estimated on the fitting blocks and
applied to the held-out block. A direction that is mostly artifact is strongly attenuated;
a direction where the comb does not exceed background is left alone. This removes the
excess and keeps the background by construction, and replaces rank with a continuous knob.

**Consequences for the criteria.** "Smallest rank satisfying all criteria" no longer
applies unchanged: the selected quantity is the number of directions given a gain below 1,
reported alongside the gains themselves. The rank ceiling of criterion 5 now bounds the
count of attenuated directions. All other criteria and thresholds stand as frozen.

**What the smoke test does not settle.** With gains applied, only 40% of held-out folds
land within +/-1 dB, and retention for the declared probes runs 0.33-0.73 before the angle
gate is applied. Both numbers are reported by the sweep rather than treated as a reason to
adjust the criteria. If preservation fails cohort-wide, the finding is that the comb and
the dominant neural topography overlap too much for linear spatial separation in this
recording setup, which is an answer, not a failure to report.

## Amendment 2 (2026-07-30): development and confirmation participants

The five participants used in the pilot influenced the estimator itself -- excess
covariance replaced ratio ranking, and per-line gains replaced pooled gains, after looking
at where sub-0008 failed. They are development data from this point on and cannot serve as
confirmation, under leave-one-participant-out or any other resampling. Cross-validation
does not undo a design choice made after seeing the data.

- **Development (burned):** sub-0000, sub-0004, sub-0008, sub-0012, sub-0015.
  All estimator iteration, threshold sanity checks and debugging happen here.
- **Confirmation (untouched):** sub-0001, sub-0003, sub-0005, sub-0006, sub-0007,
  sub-0009, sub-0010, sub-0011, sub-0013, sub-0014. 60 runs.

Confirmation runs once, on a frozen estimator, and its result is reported whatever it is.
If the estimator is changed after seeing confirmation data, the confirmation set is spent
and the claim reverts to development-only.

## Amendment 3 (2026-07-30): corrections to the pilot's claims and probe handling

**Poor transfer is not by itself proof of nonstationarity.** The pilot showed a subspace
fitted on two contiguous blocks failing on the third. Spatial overlap with the dominant
neural topography, estimator variance at 16 fitting windows, and frequency-specific
topographies remain live contributors. What the pilot does isolate is a time-ordering
effect: interleaved and contiguous splits at matched sample size differ sharply, which
rules out sample size alone as the explanation. The three estimator variants tried share
covariances, folds and data; their agreement is correlated, not independent confirmation.

**Probes must be injected before the estimator is fitted.** The pilot computed retention
as the projection applied to a topography, with the projection fitted on probe-free data.
That cannot see signal-dependent overfitting: an estimator that partly fits the probe will
remove it more aggressively than that calculation predicts. Probes are injected into the
recording first, the estimator is fitted on the injected data, and retention is recovered
by differencing against the same pipeline run without the probe -- the structure
`remove_line_comb.py` already uses via `recover_probe`.

**Next experiment, on development participants only.** Pooled spatial basis fitted per run,
with time-adaptive per-line gains estimated causally from the preceding window and applied
to the next. A fully time-local spatial basis is not attempted first: 64 dimensions cannot
be identified from a single 20 s window, whereas a per-line gain is a scalar ratio. The
basis becomes adaptive only if gain adaptation fails and basis stability is demonstrated
separately.

## Probe bank

One topography is not enough -- a single choice may happen to be favourable or
unfavourable, and one drawn orthogonal to the artifact subspace would rig the result.
Declared before the sweep:

- Topographies: leading spatial components from *both* sides of each comb neighbourhood,
  drawn from off-comb frequencies, plus intermediate mixtures spanning a range of overlap
  with the estimated artifact subspace.
- Waveforms: stationary sinusoids at comb frequencies, and amplitude-modulated gamma
  bursts (the broadband case, where loss is partly unavoidable).
- Injected across multiple participants and runs, not one.

Preservation is reported **as a function of principal angle** between the probe topography
and the artifact subspace. That curve is the honest statement of what the method can and
cannot recover, and it is expected to fall toward zero as the angle closes.

## Competence and abstention

A run is competent only if some rank satisfies criteria 1-4 jointly. Contrast ratio alone
is not competence: a high ratio with failed probe preservation means the filter is
removing signal along with artifact.

When no rank qualifies:

- The run is recorded as `unresolved` and left **uncorrected**, carrying the flag forward
  into the cohort report.
- It does **not** fall back to `spectrum_fit`. Mixing preprocessing regimes across
  participants introduces a confound that can correlate with outcome. `spectrum_fit`
  remains only as the benchmark comparator.
- It does **not** raise an error or mark the run unusable. This repository already treats
  an unresolved measurement as a first-class recorded value rather than a fault
  (`cohort/gradient.py`, `cohort/noise_floor.py`, `cohort/preservation.py`), and a stage
  that resolves nothing must not kill a cohort run. Whether to exclude an uncorrected run
  is an analysis-time decision made with the flag in hand.

The number of abstentions is itself a reported result. If it is large, subspace removal is
not the right method for this cohort, and that is a finding rather than a failure to
report.

## Downstream checks (Stage 2)

The sweep above measures the removal in isolation. It does not establish that the
corrected data behaves downstream, and narrowband correction can alter frequency-dependent
covariance and MNE's numerical rank estimate even though it leaves broadband rank full.
On representative corrected files, compare against uncorrected:

- estimated numerical rank
- number and split-half stability of ICA components
- cardiac component detection and exclusion counts (176 are currently excluded cohort-wide)
- reconstruction error
- final Study 1 features

Stage 2 requires written corrected files and runs after Stage 1 reports.

## Reporting

Per run: variant, selected rank, all criteria, competence, held-out contrast. Per
participant: aggregates. Cohort: distributions, abstention count, and pooled-versus-group
comparison, with uncertainty computed across the 15 participants.

No pass/fail verdict is synthesised from invented thresholds beyond those declared here,
and measurements are presented as measurements.
