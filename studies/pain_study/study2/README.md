# Study 2 - Source-Space Analysis of NPS-Predictive EEG Activity

## 1. Problem Statement and Predesignated Study 1 Cell

Study 1 tests whether plateau-window EEG spectral power predicts trial-wise fMRI pain-signature
expression beyond prespecified nuisance structure. Study 2 maps the cortical source-power patterns
associated with the prespecified NPS-predictive EEG component. Because multivariate decoder weights
are backward-model coefficients whose amplitude does not read directly as a neural contribution,
the analysis uses Haufe-transformed sensor patterns and source-space association maps
(Haufe et al., 2014).

The analysis target is restricted a priori to the Study 1 primary cell: NPS prediction by
ElasticNet from individual-channel spectral power of alpha, beta, and scanner-clean gamma,
residualized at Level 2. The primary analysis uses the out-of-sample EEG prediction of the NPS
residual as a single combined prediction-derived score and associates each band's source power with
that score. The primary source family comprises three full-plateau source maps, $A_\alpha$,
$A_\beta$, and $A_\gamma$. Consistent with Study 1, gamma remains in the primary family, but it is
defined from retained intervals of 30.1–38.0, 43.0–56.0, and 67.0–77.0 Hz. The empirically
contaminated scanner-harmonic windows of 38.0–43.0, 56.0–67.0, and 77.0–85.0 Hz are excluded before
model fitting and source-power extraction.

This restriction is a measurement decision, not a post hoc outcome filter. The scanner-harmonic QC
benchmark applied to valid production recordings found subject-consistent residual peaks at
approximately 41.138, 61.096, and 82.214 Hz after BrainVision Analyzer correction. Median peak
prominences were 16.407, 26.187, and 20.890 dB across 53 successful runs from 9 subjects. Because
event-locked QC showed late-window increases in both broad gamma and harmonic prominence, a
contiguous 30.1–80.0 Hz source map would not isolate neural gamma. Study 2 therefore estimates
$A_\gamma$ from the same scanner-clean gamma intervals as Study 1 and reports artifact diagnostics
explicitly because high-frequency EEG recorded during simultaneous fMRI remains sensitive to
muscular, oculomotor, and gradient contamination.

Because scalp EEG cannot resolve the deep insular, thalamic, and brainstem generators that dominate
the NPS, these maps characterize the *cortical EEG correlates* of the NPS-predictive component, not
the signature's generators. Source claims are framed accordingly throughout.

Prespecified Study 1 criteria are evaluated for the selected primary cell before Study 2 inference is
summarized. The criteria are met when the cell shows a predictive gain
$\Delta R^2_{\mathrm{LOSO}} \ge 0.02$, a confidence-interval lower bound $\ge 0.005$, a
Holm-corrected $p \le 0.05$, a staged nuisance-adjusted sensitivity gain $\ge 0.005$, a positive
within-subject gain, satisfied temporal negative controls, satisfied artifact-censoring robustness,
and a split-half reliability $\ge 0.4$ on at least 30 trials. Missing diagnostic fields do not pass
these criteria. The output records `confirmatory_criteria_met` and `unmet_criteria`; it does not
assign an automatic interpretation label to the Study 2 maps.

## 2. Frozen Model and Prediction-Derived Score

All analyses use out-of-sample data. The model class, feature family, frequency preset, preprocessing
recipe, outer folds, target transformation, and fold-specific hyperparameters are frozen as estimated
in Study 1 before any source-map construction. For each held-out subject, the frozen incremental
model generates one EEG prediction of the NPS residual per retained plateau trial, denoted
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$. This score is centered and scaled to unit
variance within the held-out subject; a zero-variance score does not meet the source-stage criteria.
The primary analysis uses this single combined score rather than a per-band decomposition, which
avoids the collinearity between spectral contributions of one decoder.

The combined score is the frozen linear predictor in the transformed residual-target space,
equivalently the sum of the band-specific contributions $\eta_\alpha$, $\eta_\beta$, and
$\eta_\gamma$. The band-specific contributions are retained for the secondary mutually adjusted
analysis (Section 10).

## 3. Sensor-Level Pattern Estimation

Sensor-level linear feature patterns are estimated with the Haufe transformation, which converts a
backward weight vector into a directly readable activation pattern (Haufe et al., 2014). For the
primary ElasticNet model, the pattern is computed within each outer fold from training-fold
statistics,

$$
A_{\mathrm{Haufe}} = \Sigma_{X,\mathrm{train}} \, W,
$$

where $\Sigma_{X,\mathrm{train}}$ is the training-fold feature covariance matrix and $W$ is the
fitted weight vector in the same space. Because Study 1 standardizes features to zero mean and unit
variance, $W$ is estimated in standardized units and $\Sigma_{X,\mathrm{train}}$ is the training-fold
correlation matrix.

## 4. Cortical Source Reconstruction

The primary inverse operator uses sLORETA (Pascual-Marqui, 2002), FreeSurfer subject-specific head
models, and boundary element method forward solutions. Fp1 and Fp2 are excluded from the forward
solutions, noise covariance matrices, and inverse operators before rank estimation, so that the
frontal artifact-proxy channels do not influence reconstruction. The primary noise covariance is
derived from the −5.0 to −0.01 s pre-stimulus baseline, matching Study 1.

A single inverse operator is estimated with the broadband noise covariance and applied identically
to voltage time series filtered into each modeled band. Alpha and beta are filtered as contiguous
bands. Gamma is filtered separately in the three retained scanner-clean intervals, converted to
native-space source-level Hilbert log-ratio power in each interval, morphed to fsaverage as scalar
log-ratio maps, and combined as a bandwidth-weighted log-ratio mean to preserve the single Study 2
contribution label $\gamma$ without reintroducing the excluded scanner windows. The primary source
time series uses the cortical surface-normal component from the loose-orientation inverse
(constraint 0.2), and instantaneous power envelopes are computed with the Hilbert transform before
surface morphing. Source-power construction mirrors the Study 1 individual-channel spectral-power
estimand by using total band-limited power without evoked-response subtraction. Hilbert power is
baseline-corrected as a log-ratio with the same primary baseline and averaged over the same active
plateau window (3.0 to 10.5 s). Regularization uses SNR = 3.0, corresponding to $\lambda^2 \approx
0.111$, oct6 source spacing, and depth weighting of 0.8.

Source quality control is completed before map inspection. A subject is excluded when FreeSurfer
reconstruction fails visual quality control, the boundary element model fails, measured electrode
positions are unavailable, fewer than 90% of retained EEG channels have valid locations, mean
coregistration error exceeds 5 mm or its maximum exceeds 10 mm, the forward solution contains
rank-deficient channels, or morphing to fsaverage fails.

The cortical point-spread full width at half maximum is computed per subject from each individual
forward and inverse operator and morphed to fsaverage. The cohort median point-spread FWHM and its
regional range are reported and bound the spatial resolution of the source maps; the median value
sets the fMRI smoothing kernel for the spatial-correspondence test (Section 7). This resolution
report replaces a full Monte Carlo calibration and adds no permutation cost.

## 5. Source-Stage Association Model

Association maps are computed directly in source space from a fixed log-ratio power tensor and an
explicit residualization design. The prediction-derived score and each band's source power are
residualized against the same design and standardized within subject. The primary source-stage
design aligns with the Study 1 Level 2 nuisance family and comprises task-run intercepts,
categorical stimulus temperature and thermode surface with the Study 1 reference coding, trial onset
time and the linear trial number within run, HRF-weighted framewise displacement, HRF-weighted
standardized DVARS, HRF-weighted Fp1/Fp2 high-frequency power, and residual ECG coupling. The primary
design includes no opposite-band contribution term; the cross-band adjustment is a secondary analysis
(Section 10).

After this processing, each vertex value is a standardized regression coefficient equivalent to the
within-subject partial correlation between processed source power and the processed prediction-derived
score. Subject-level maps retain the partial-correlation-scale values and their Fisher-z transform for
group inference. A parallel map is computed for the actual residual target to verify the directional
consistency of the contribution pattern with the target-associated pattern.

These maps reflect the cortical topography of the model's spectral-power inputs and may partly
recapitulate canonical band-power topography (e.g., posterior alpha, sensorimotor beta) rather than
pain-specific cortical structure. The reported estimand is therefore the cortical EEG correlate of
the NPS-predictive component; the true-target directional-consistency map and the artifact
diagnostics (Section 9) provide the corresponding specificity checks.

Circular-shift permutations use the six 11-trial task runs as exchangeability units, matching
Study 1. A subject is excluded from source analyses when fewer than three permutation-valid runs,
fewer than 25 clean plateau trials, or fewer than 15 residual degrees of freedom remain, or when the
design is not full rank after censoring. The source-stage design must have condition number ≤ 100
after centering and scaling non-intercept columns. The quality-control report includes retained
trial counts, valid-run counts, design rank, residual degrees of freedom, and condition number for
every subject before map inspection. With fewer than 30 valid subjects, source inference is reported
as a feasibility analysis.

## 6. Group-Level Source Inference

The primary permutation test uses a target-retrained null, which estimates the source-power
association expected when the EEG-to-fMRI relationship is broken. For each outer fold and permutation,
the Level 2 nuisance residual is circularly shifted relative to EEG within run using the Study 1
minimum-distance rule, and the permuted target is reconstructed as the unshifted nuisance prediction
plus the shifted residual. The ElasticNet model is refit on the permuted target, reusing the frozen
features, preprocessing statistics, Yeo-Johnson transformation, and hyperparameters, then applied to
the permuted held-out subject to regenerate the permuted prediction-derived score. Within-subject
standardization, source residualization, group aggregation, and cluster testing are then repeated.
Inference uses 1,000 valid draws, fixing a minimum cluster-tail probability of approximately 0.001
under the $(b+1)/(m+1)$ correction. A draw is invalid when the held-out score has zero variance or
the source-stage design is not full rank; invalid draws are resampled before outcome inspection.

Group inference uses cluster-based permutation testing (Maris & Oostenveld, 2007). Each subject
contributes, per band, one Fisher-z-transformed association map after morphing to fsaverage, with a
positive value indicating greater source power on trials with a larger NPS prediction. Vertex-wise
one-sample $t$ statistics are computed, and vertices exceeding an uncorrected cluster-forming
threshold of $p < 0.01$ are aggregated into contiguous sign-preserving clusters; a threshold of
$p < 0.001$ is checked as sensitivity. Family-wise error correction uses the two-sided maximum
statistic within each band, and the band-level p-values for $A_\alpha$, $A_\beta$, and $A_\gamma$ are
Holm-corrected as the primary source family. Subjects are equally weighted, and leave-one-subject-out
influence diagnostics evaluate cluster stability.

Inferential scope is bounded by available power. For roughly 30 valid source subjects and tens of
rated trials per subject, the analysis is calibrated to detect moderate within-subject partial
correlations rather than subtle associations; this limit is stated explicitly, and non-significant
effects are not interpreted as the absence of association.

## 7. Multimodal Spatial Comparison

Spatial correspondence between the EEG source-power association pattern and the pain-related fMRI
covariance is evaluated within the cortical analysis mask. The primary fMRI target is a forward
covariance pattern,

$$
A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}
\propto
\mathrm{Cov}(B_{\mathrm{fMRI}}^{\mathrm{resid}},\; r_{\mathrm{NPS-L2}}),
$$

where $B_{\mathrm{fMRI}}^{\mathrm{resid}}$ denotes trial-wise fMRI beta maps after voxelwise
residualization against the Level 2 nuisance family, and $r_{\mathrm{NPS-L2}}$ the Level 2
residualized NPS expression, all estimated on training-fold subjects within the LOSO framework. The
primary statistic, computed separately per band, is the Pearson spatial correlation between the EEG
source map and $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ across matched cortical vertices. It is tested
against variogram-matched BrainSMASH spatial surrogates with 5,000 maps per hemisphere on the
fsaverage surface and medial-wall exclusion, which controls spatial autocorrelation (Burt et al.,
2020). The surrogate p-values for $A_\alpha$, $A_\beta$, and $A_\gamma$ are Holm-corrected as a
distinct spatial-correspondence family. Confound specificity is assessed by reporting, for comparison,
the correlation of the EEG source map with a motion covariance pattern and a physiological-noise
covariance pattern built by the same procedure. The fMRI target is smoothed with the cohort-median
source point-spread FWHM (Section 4), matching the kernel to the empirical source resolution, and
the kernel effect is checked over a sensitivity range spanning the regional point-spread range.

Because the cortical point-spread of the EEG inverse is broad (typically centimeters), this
comparison is treated as coarse corroboration of large-scale spatial alignment, not as voxel-scale
localization. A weak or null spatial correlation is reported as a resolution limit rather than as
evidence against shared spatial structure.

## 8. Secondary Behavioral Convergence

A secondary analysis determines whether the derived source pattern converges with the Study 1 thermal
or pain intensity score. For each target subject, the group pattern is estimated from the remaining
subjects, restricted to the cortical mask, and L2-normalized with its sign preserved. The held-out
subject's source power is transformed by the same pipeline as the primary maps, and the behavioral
expression is

$$
\text{expression}_{s,i}
=
\sum_v
\widetilde{\text{sourcepower}}_{s,i}(v)
\cdot
\text{Pattern}_{-s}(v).
$$

The association between expression and the intensity score is estimated by ordinary least squares
within each subject, controlling for the Level 2 nuisance design, and the statistic is the group mean
of standardized coefficients, tested two-sided with 5,000 permutations that circularly shift
expression relative to rating within run. A subject is excluded when fewer than 25 rated trials or
fewer than three valid runs remain, or when within-subject rating variance is zero; missing ratings
are not imputed. A positive significant coefficient supports intensity convergence.

## 9. Artifact Diagnostics and Robustness Controls

For each sensor and source map, trial-wise expression is regressed against framewise displacement,
DVARS, cardiac phase, scanner-frequency residual power, and Fp1/Fp2 high-frequency power, within
subject and then at the group level with Holm correction. The artifact-control summary reports
`artifact_control_criteria_met`, `unmet_criteria`, and Holm-adjusted expression q-values. Criteria
are unmet when the absolute spatial correlation with an artifact template exceeds 0.80 at the sensor
level or 0.50 in source space, or when expression significantly covaries with an artifact metric.
The summary does not relabel maps as neural, exploratory, or contaminated.

The primary analyses are repeated after trial censoring with the Study 1 thresholds: a trial is
censored when framewise displacement exceeds 0.5 mm, the robust z of DVARS, Fp1/Fp2 power, or scanner
residual peaks exceeds 3, or cardiac phase-locking exceeds the 95th percentile of the within-subject
null. Robustness summaries report preserved significance, retained cluster sign, unthresholded
spatial correlation (threshold 0.50), Dice overlap of thresholded clusters (threshold 0.40), and
centroid displacement (maximum 15 mm). Subject-wise performance, convergence coefficients, and
expression distributions are reported with 95% BCa bootstrap confidence intervals from 10,000
resamples.

## 10. Secondary Band-Specificity Analysis

The combined-score primary maps establish that a band's source power tracks the overall
NPS-predictive component, but they cannot attribute prediction to a specific band. Band specificity is
therefore addressed by a prespecified secondary analysis rather than left to a sensitivity footnote.
The predictor is decomposed into the frozen alpha, beta, and gamma contribution scores, and each
band's source power is associated with its own contribution score while adjusting for the other
bands' contribution scores. A subject is excluded from a band-unique map when the maximum
adjacent-band variance inflation factor exceeds 5 or the contribution design condition number exceeds
100, preserving numerical stability of the decomposition. Band-unique inference is run only when
band-unique target-retrained null maps have been generated explicitly for the same mutually adjusted
estimand; it is not part of the default `all` sequence. When those nulls are supplied, the
band-unique maps use the same cluster test as the primary family, and their band-level p-values are
Holm-corrected as a distinct band-specificity family. A band that is significant in the primary
family but not in the band-unique family is reported as reflecting shared rather than band-specific
prediction structure.

## 11. Sensitivity Analyses

Two sensitivity analyses complement the primary family without altering its conclusion: source
reconstruction at SNR values of 1.0 and 5.0; and a noise covariance estimated on the immediate
pre-stimulus baseline (−0.2 to −0.01 s). Each reports whether the primary cluster retains its
significance and sign.

## References

Burt, J. B., Helmer, M., Shinn, M., Anticevic, A., & Murray, J. D. (2020). Generative modeling of
brain maps with spatial autocorrelation. NeuroImage, 220, 117038.
doi.org/10.1016/j.neuroimage.2020.117038

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.
(2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
NeuroImage, 87, 96-110. doi.org/10.1016/j.neuroimage.2013.10.067

Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data.
Journal of Neuroscience Methods, 164(1), 177-190. doi.org/10.1016/j.jneumeth.2007.03.024

Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography
(sLORETA). Technical details. Methods and Findings in Experimental and Clinical Pharmacology,
24(Suppl. D), 5-12.
