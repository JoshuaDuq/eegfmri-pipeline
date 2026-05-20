# Study 2 - Source Interpretation of NPS-Predictive EEG Activity

## 1. Problem Statement

Study 1 tests whether plateau-window EEG spectral power predicts trial-wise fMRI pain-signature
expression beyond prespecified nuisance structure. Study 2 maps the cortical source-power patterns
associated with the prespecified NPS-predictive EEG component. Because multivariate decoder weights
are backward-model coefficients, source interpretation uses Haufe-transformed sensor patterns and
source-space association maps (Haufe et al., 2014).

The source-interpretation target is restricted a priori to the Study 1 NPS ElasticNet alpha+beta
individual-channel spectral-power cell, following the thesis hierarchy. NPS is the Study 1 target
most closely tied to evoked
nociceptive processing, whereas SIIPS1 indexes a related but distinct
stimulus-intensity-independent pain-signature construct. SIIPS1, Ridge, Random Forest,
BandTemporalRegressor, Level 1, Level 3, ROI-level, and global-average analyses are sensitivity or
exploratory analyses in Study 2.

## 2. Objectives

The primary objective is to derive band-specific standardized source-power association maps from
the frozen out-of-sample transformed-space EEG contribution scores of the predesignated Study 1
NPS ElasticNet alpha+beta individual-channel spectral-power cell.

The Study 2 primary source family comprises two contribution-matched full-plateau source maps. The
alpha map, denoted $A_\alpha$, associates processed alpha source power with the standardized alpha
contribution score. The beta map, denoted $A_\beta$, associates processed beta source power with
the standardized beta contribution score.

Study 2 is a planned source-interpretation follow-up of the prespecified Study 1 NPS ElasticNet
alpha+beta cell.

Primary source-map evidence is evaluated separately for $A_\alpha$ and $A_\beta$. For each band,
the report distinguishes five quantities: source-cluster
inference under the target-retrained null, artifact diagnostics, directional consistency with the
true residual NPS source association map, behavioral criterion overlap, and fMRI spatial
correspondence.

## 3. Frozen Study 1 Model and Band Contributions

All interpretive analyses use out-of-sample data. The selected Study 1 model class, feature family,
frequency preset, preprocessing recipe, outer folds, target transformation, and fold-specific
hyperparameters are frozen before source-map construction.

For each held-out subject, the frozen primary incremental Study 1 model generates one EEG residual
prediction per retained plateau trial, denoted
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$. The nuisance-only prediction and the full
combined raw prediction are retained for performance reporting.

Because the alpha+beta feature preset combines two spectral bands, Study 2 uses the frozen linear
contribution scores from the same ElasticNet model. For each outer fold, the standardized linear
predictor is decomposed in the
transformed residual-target space, before inverse target transformation, into
$\eta_\alpha = X_\alpha W_\alpha$ and $\eta_\beta = X_\beta W_\beta$. Each contribution score is
then centered and scaled to unit variance within the held-out subject and band using that subject's
retained plateau trials. Zero-variance contribution scores are ineligible for source-stage
analysis.

The band contribution scores are model-contribution variables in transformed residual-target space.
Source maps therefore index associations with alpha or beta contributions from the frozen
ElasticNet decoder. Raw-scale
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values feed performance reporting and
full-prediction sensitivity maps.

Each band-specific source analysis includes the standardized opposite-band contribution score as a
prespecified nuisance regressor. Alpha maps adjust for $\eta_\beta$, and beta maps adjust for
$\eta_\alpha$.
Full-prediction band association maps using $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$
are reported beside the unique contribution maps to characterize shared alpha-beta prediction
structure.

Band-unique source maps require numerically stable alpha/beta contribution decomposition. Before
source mapping, the within-subject source-stage design containing
$\eta_\alpha$, $\eta_\beta$, and the fixed nuisance terms must have condition number ≤ 100 after
centering and scaling non-intercept columns, and the opposite-band contribution term must have
variance inflation factor ≤ 5 for every retained source subject. Subjects not meeting these
criteria are excluded from band-unique maps. If fewer than 30 source-valid subjects remain, the
band-unique alpha and beta maps are reported as feasibility analyses and the shared full-prediction
maps characterize shared prediction structure.

## 4. Sensor-Level Interpretation

Sensor-level linear feature patterns are interpreted with the Haufe transformation when the model is
linear (Haufe et al., 2014). For the primary ElasticNet model, the pattern is computed within each
outer fold using training-fold statistics.

$$A_{\mathrm{Haufe}} = \Sigma_{X,\mathrm{train}} \, W.$$

Here, $\Sigma_{X,\mathrm{train}}$ is the training-fold feature covariance matrix and $W$ is the
fitted weight vector in the same feature space. Because Study 1 standardizes features to zero mean
and unit variance within each fold, $W$ is estimated in standardized units and
$\Sigma_{X,\mathrm{train}}$ is the training-fold correlation matrix.

For nonlinear models, Random Forest and BandTemporalRegressor maps are reported as
prediction-covariance association maps.

$$A_{\mathrm{assoc}} \propto \mathrm{Cov}(X_{\mathrm{features}},\; \hat{y}_{\mathrm{test}}).$$

Source-space follow-up uses spectral power features. Source-space power is recomputed from
inverse-projected voltage time series. Complexity topographies remain sensor-level analyses.

## 5. Artifact Controls

Artifact diagnostics compare sensor-level maps with topographies from independent calibration data
or ICA components labeled as ocular, cardiac, scanner, or high-frequency frontal artifacts. Spatial
patterns concentrated on the anterior periphery (e.g., AF7, AF8, F7, F8) are reported as potential
residual EMG contamination.

For each sensor-level map, trial-wise map expression is computed by projecting each trial's feature
vector onto the map. Artifact association is tested by regressing this expression score against
framewise displacement, DVARS, cardiac phase, scanner-frequency residual power, and Fp1/Fp2
high-frequency artifact power. A map is reported as artifact-contaminated when its absolute spatial
correlation with any artifact template exceeds $r = 0.80$ or its expression significantly covaries
with any artifact metric after Holm correction.

The same artifact diagnostics are applied in source space. For each primary source map, trial-wise
source-pattern expression is computed from the processed source-power matrix before group
aggregation. Expression is regressed against framewise displacement, DVARS, cardiac phase,
scanner-frequency residual power, and Fp1/Fp2 high-frequency artifact power within subject, then
tested at the group level with Holm correction across artifact metrics and the two primary bands.

Artifact-prediction source association maps are generated by replacing the prediction-derived score
with each artifact metric after the same source-stage preprocessing. Artifact summaries include
source-pattern expression associations, unthresholded spatial correlations with artifact-prediction
source maps, and artifact-censoring effects on the Study 1 primary cell.

## 6. Exploratory Spatio-Temporal Mapping

The BandTemporalRegressor processes continuous epochs and supports time-resolved
prediction-covariance mapping. Its spatio-temporal maps are exploratory.

Pain-evoked high-frequency oscillations are largely induced and non-phase-locked. The
spatio-temporal map is computed on instantaneous power envelopes.

$$A_{\mathrm{assoc}}(t) \propto \mathrm{Cov}(E_{\mathrm{test}}(t),\; \hat{y}_{\mathrm{test}}).$$

This analysis produces maps with dimensions bands, channels, and time. Time-resolved analyses are
evaluated within predefined early, mid, and late plateau windows to control temporal multiplicity.

## 7. Source Modeling

Source-space analysis estimates cortical source-power association with the out-of-sample
band-specific contribution scores from the EEG-predicted NPS residual model. The source estimand
is defined at the level of band-limited voltage time series. Inverse modeling is applied to
band-passed voltage, and Hilbert power is computed from the reconstructed source time series. The
resulting maps are inverse-model-dependent standardized source-power association estimates.

The primary inverse operator uses sLORETA (Pascual-Marqui, 2002), FreeSurfer subject-specific head
models, and boundary element method forward solutions. Primary forward solutions, noise covariance
matrices, and inverse operators exclude Fp1 and Fp2 before rank estimation to prevent frontal
artifact-proxy channels from influencing source reconstruction.

The primary noise covariance is derived from the −5.0 to −0.01 s pre-stimulus baseline, matching
Study 1. Source localization is repeated with the immediate pre-stimulus baseline
(−0.2 to −0.01 s) and the early pre-cue baseline (−7.0 to −5.5 s).

A single inverse operator is estimated with the broadband noise covariance and applied identically
to voltage time series filtered into each canonical band. The primary source time series uses the
cortical surface-normal component from the loose-orientation inverse. Vector output and
orientation-norm pooled estimates are orientation-sensitivity analyses. Instantaneous power
envelopes are computed from the signed normal-orientation source time series with the Hilbert
transform. Positive and negative map values indicate the direction of association between source
power and the standardized prediction-derived score.

Primary source-power construction mirrors the Study 1 individual-channel spectral-power
estimand by using total band-limited power without evoked-response subtraction. Source-space
Hilbert power is baseline-corrected as a log-ratio using the same primary baseline
(−5.0 to −0.01 s) and averaged over the same active plateau window (3.0 to 10.5 s).
ERP-subtracted source power is a sensitivity analysis when the template is estimated within the
relevant Study 1 training fold and recorded as fold-owned provenance.
Absolute plateau log-power maps that omit this baseline alignment are sensitivity maps.

Source modeling quality-control checks are completed before source-map inspection. A subject is
excluded from source-stage analyses if FreeSurfer reconstruction fails visual quality control, the
boundary element model fails, subject-specific measured electrode positions are unavailable,
template-only electrode coordinates are used, fewer than 90% of retained EEG channels have valid
locations, mean electrode-coregistration error exceeds 5 mm, maximum electrode-coregistration error
exceeds 10 mm, the forward solution contains invalid or rank-deficient channels, or morphing to
fsaverage fails.

## 8. Source-Stage Association Model

Prediction-associated source-power maps are computed directly in source space from a fixed
baseline-corrected log-ratio power tensor and an explicit source-stage residualization design. For
each subject, band, trial, and vertex, Hilbert power is averaged in the Study 1 primary baseline and
active plateau windows. Plateau power is converted to baseline-corrected log-ratio power with the
prespecified subject-band numerical offset. The resulting source log-ratio power tensor is fixed
before permutation testing. Residualization and standardization are recomputed from the observed or
permutation-specific source-stage design before each association map. Vertices with zero
post-residualization variance are excluded for that subject.

The relevant prediction-derived score is residualized against the same source-stage design and
standardized within subject before association mapping. The primary source-stage design aligns with
the Study 1 Level 2 nuisance family and includes the following terms.

1. Task-block intercepts.
2. Categorical stimulus temperature with the Study 1 reference coding.
3. Selected thermode surface with the Study 1 reference coding.
4. Trial onset time and explicit linear trial number within task block
   (`trial_index_within_block`).
5. HRF-weighted framewise displacement.
6. HRF-weighted standardized DVARS.
7. HRF-weighted Fp1/Fp2 high-frequency artifact power.
8. Residual ECG coupling.
9. The opposite-band contribution score for band-specific maps.

This design is identical for every retained source subject except for the prespecified opposite-band
contribution term. Alpha maps use standardized $\eta_\beta$, and beta maps use standardized
$\eta_\alpha$. In each permuted analysis, the corresponding permutation-specific standardized
contribution score is used. The covariate set is fixed across subjects, task blocks, bands, and
permutations. Subjects with rank-deficient categorical encoding after source-stage censoring are
excluded from the primary source analysis. The continuous nonlinear temperature basis is retained
as a sensitivity analysis to match Study 1.

After this processing, each vertex value is a standardized regression coefficient equivalent to the
within-subject partial correlation between processed source power and the processed
prediction-derived score. Subject-level maps retain the raw partial-correlation-scale values and
Fisher-z-transformed values for group inference. The primary source map represents the portion of
the Study 1 transformed-space band contribution that remains
after additional removal of discrete stimulus intensity, selected thermode surface, task-block
structure, trial order, acquisition-noise structure, and cross-band prediction association.

Full-prediction band association maps are reported beside the mutually adjusted maps to summarize
shared alpha-beta prediction structure. A parallel unadjusted source-power association map using
the original $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values and task-block
intercept adjustment is reported as an additional sensitivity analysis.

Circular-shift source permutations use the six 11-trial task blocks as exchangeability units,
matching Study 1. Source-stage censoring uses the same variable-length retained-block rule: a
permutation-valid source block must retain at least 8 source-valid plateau trials and allow at
least four admissible nonzero circular shifts under the Study 1 retained-sequence shift definition.
Partial source blocks satisfying these criteria remain in source-map computation;
partial blocks not meeting these criteria are excluded. A subject is excluded from source-stage
analyses if fewer than three permutation-valid task blocks remain, fewer than 25 clean plateau
trials remain, any primary source-stage nuisance covariate is unavailable, or the fixed
source-stage design is not full rank after censoring. Inclusion also requires at least 15 residual
degrees of freedom after source-stage censoring, defined as retained source-valid trials minus the
rank of the fixed source-stage design.
If fewer than 30 subjects remain after source-stage QC, source inference is reported as a
feasibility analysis.
The source-stage quality-control report includes retained source-valid trial counts, retained
permutation-valid block counts, fixed-design rank, residual degrees of freedom, condition number,
and opposite-band VIF for every subject before source-map inspection.

## 9. Target and Error Association Maps

Parallel maps are computed for the actual held-out residual target and the full-model prediction
error, allowing contribution-associated source patterns to be compared with target-associated and
error-associated source patterns.

$$\mathrm{Assoc}(\text{processed source power},\; r_{\mathrm{NPS-L2}}).$$

$$
\mathrm{Assoc}(
\text{processed source power},\;
\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}} - r_{\mathrm{NPS-L2}}
).
$$

For each primary band, the true-target and prediction-error maps use the same band-specific
source-power tensor and the same source-stage nuisance design as the corresponding contribution
map, including the opposite-band contribution regressor.

Directional consistency with the true residual NPS map is summarized with three criteria: the
unthresholded Pearson spatial correlation between the primary contribution map and the band-matched
true-target map is ≥ 0.20 across the cortical analysis mask; the median true-target association
inside the primary FWE-corrected source cluster has the same sign as the primary cluster statistic;
and at least 60% of vertices in that cluster have true-target association values with the primary
cluster sign. The full-model prediction-error association map summarizes prediction-error spatial
structure. Prediction-error dominance is reported when its absolute spatial correlation with a
primary contribution-matched map equals or exceeds that map's true-target association correlation,
or when it produces a larger FWE-corrected cluster in the same broad anatomical region.

## 10. Group-Level Source Inference

Analyses with 20-29 source-valid subjects are reported as feasibility analyses. Analyses with
fewer than 20 source-valid subjects are reported as QC summaries.

Before outcome-map interpretation, a source-stage precision and calibration simulation is completed
using source-valid trial counts, task-block structure, baseline source-noise covariance, observed
post-source-QC block lengths, admissible source-shift counts, source-stage nuisance matrices,
cortical adjacency, and each subject's empirical point-spread and cross-talk summaries. Simulated
maps include null maps and embedded smooth clusters with standardized within-subject partial
correlations of $|r| \in \{0.05, 0.10, 0.15, 0.20\}$ after source-stage nuisance removal. For each
retained-sample scenario, 2,000 Monte Carlo datasets are generated and tested with the same
cluster-forming thresholds, target-retrained permutation count, subject weighting, morphing, and
robustness checks planned for the observed maps.

The simulation report includes null family-wise error rate for the nominal 0.05 cluster test,
95% bootstrap CI half-width for mean cluster expression at $|r| = 0.15$, and recovery rate for an
embedded $|r| = 0.15$ cluster whose spatial extent matches the median empirical point-spread FWHM.
These quantities define the precision and calibration context for the observed source maps.

Regularization uses SNR = 3.0, corresponding to $\lambda^2 \approx 0.111$, oct6 source spacing,
loose orientation constraint of 0.2, and depth weighting of 0.8. Before outcome-map inspection,
empirical point-spread and cross-talk functions are estimated from each subject's actual forward and
inverse operator and morphed to fsaverage. The median cortical point-spread FWHM and its regional
range are reported and used to choose the fMRI smoothing kernel for spatial comparison.
Sensitivity analyses use SNR values of 1.0 and 5.0 and report whether the primary cluster retains
significance and sign under both values.

Subject-specific source maps are morphed to fsaverage. The primary group aggregation uses
non-normalized maps and includes every source-valid subject. GFP normalization is retained as a
sensitivity analysis.

## 11. Statistical Inference

### 11.1 Permutation Strategy

Nonparametric permutation testing is performed on source-localized maps. The primary null is
target-retrained with frozen Study 1 model-selection outputs.

Observed and permuted contribution scores are generated from the original Study 1 prediction-valid
LOSO training folds. Source-stage censoring and source-valid subject restrictions are applied after
contribution-score generation.

Each permutation and outer fold proceeds as follows.

1. Circularly shift the Level 2 nuisance residual target relative to EEG within task block for
   training and held-out subjects, using the same minimum shift-distance rule as Study 1.
2. Reconstruct the permuted raw target as the unshifted nuisance prediction plus shifted residual.
3. Reuse the observed Study 1 feature set, preprocessing statistics, Yeo-Johnson transformation,
   fold-specific ElasticNet hyperparameters, and prediction-valid training sample.
4. Refit the ElasticNet coefficients and intercept on the permuted training target, then apply the
   model to the permuted held-out subject to generate permuted alpha and beta contribution scores.
5. Refit the within-subject contribution-score standardization to each permuted held-out
   contribution vector.
6. Repeat source-stage nuisance control, source-map computation, group aggregation, and cluster
   testing. For each permuted band map, the opposite-band nuisance term is the standardized
   opposite-band contribution score from the same permuted model.

This null estimates the source-power association expected after disrupting the fMRI target
relationship. Circularly shifting the final observed contribution scores relative to source power is
a secondary model-anatomy sensitivity null.

The executable permutation plan is fixed before source-map inspection. Band-specific source
log-ratio power tensors, permutation-invariant nuisance covariates, cortical adjacency, and
point-spread summaries are cached before permutation testing. Source-stage residualization matrices
are rebuilt for each observed and permuted band map whenever they depend on prediction-derived
contribution scores.

Inference uses 1,000 valid target-retrained permutation draws, giving a minimum attainable
cluster-tail probability of approximately 0.001 with the standard $(b+1)/(m+1)$ correction.
Permutation validity is evaluated at the draw level across both bands. A draw is invalid if either
band produces zero-variance held-out contribution scores for any retained source subject, if the
source-stage design with the permutation-specific opposite-band term is not full rank, or if the
design fails the same condition-number or VIF criteria used for observed band-unique maps.
Invalidity in either band discards the paired alpha/beta draw. Invalid draws are resampled before
outcome-map inspection. If either band has an uncorrected cluster p-value below 0.10 or a maximum
cluster statistic within 10% of the 95th-percentile null threshold, both bands are extended to
5,000 valid paired draws before final reporting.

A smaller full-selection sensitivity null repeats Study 1 inner GroupKFold hyperparameter
selection and source-map computation for at least 250 permutations. The planned null size is fixed
before outcome inspection.

### 11.2 Cluster-Based Inference

Group-level inference uses cluster-based permutation testing on the primary ElasticNet model
(Maris & Oostenveld, 2007). The primary source family comprises the two contribution-matched
full-plateau maps, $A_\alpha$ and $A_\beta$, both derived from the prespecified Study 1 NPS
ElasticNet alpha+beta individual-channel spectral-power cell.
Each subject contributes one Fisher-z-transformed source-power association map per band after
morphing to fsaverage. Positive values indicate greater processed source power on trials with larger
band-specific prediction contribution after nuisance and opposite-band adjustment. Vertex-wise
one-sample $t$ statistics test whether the mean association differs from zero.

Vertex-wise statistics exceeding an uncorrected cluster-forming threshold of $p < 0.01$ are
aggregated into contiguous sign-preserving clusters. Results are checked under thresholds of
$p < 0.001$ and $p < 0.05$ and report whether the primary cluster retains significance and sign.
Family-wise error correction uses a two-sided maximum statistic across positive and negative
clusters within each band. Band-level p-values for $A_\alpha$ and $A_\beta$ are Holm-corrected as
the primary source family.

Behavioral criterion-overlap p-values are Holm-corrected as a separate two-band internal
convergence family. Secondary summaries spanning plateau subwindows and alternative feature
families are Holm-corrected as one exploratory family.

Equal subject weighting is used for the primary analysis. Baseline-noise inverse-variance weighting
is evaluated as a sensitivity analysis. Leave-one-subject-out influence diagnostics are applied to
the source cluster. These diagnostics evaluate the maximum change in cluster mass after removing
one subject, whether the cluster remains directionally consistent, and whether the result depends
on one or two high-amplitude outlier maps.

### 11.3 Gamma-Band Interpretation

Gamma-band source maps are exploratory because high-frequency EEG during simultaneous fMRI is
artifact-sensitive.

## 12. Multimodal Spatial Comparison

Spatial correspondence between the EEG source-power association pattern and fMRI pain-related
covariance is evaluated within the cortical analysis mask.

The primary comparison uses a within-study fMRI forward covariance pattern.

$$
A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}
\propto
\mathrm{Cov}(B_{\mathrm{fMRI}}^{\mathrm{resid}},\; r_{\mathrm{NPS-L2}}).
$$

Here, $B_{\mathrm{fMRI}}^{\mathrm{resid}}$ denotes trial-wise fMRI beta maps after voxelwise
residualization against the same Level 2 nuisance family used in Study 1. The term
$r_{\mathrm{NPS-L2}}$ denotes fold-contained residualized NPS expression after removal of the Level
2 nuisance family. Both the fMRI covariance target and voxelwise nuisance coefficients are
estimated from training-fold subjects within the same LOSO framework. The final fMRI cortical
vector is the equal-weight mean of the LOSO training-fold fMRI covariance maps across held-out
source-valid subjects, after projection to the Study 2 cortical analysis mask.

Spatial specificity is tested separately for $A_\alpha$ and $A_\beta$ using the same fMRI cortical
vector. The EEG source map is compared with $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ and with control
maps derived from circular-shifted NPS scores, motion parameters, physiological noise regressors,
stimulus variables, and pre-stimulus EEG predictions. Subjective-rating maps are evaluated
separately as criterion-overlap maps. For each band, $A_{\mathrm{control}}$ is the prespecified
control map with the largest absolute EEG-control spatial correlation. The primary statistic is

$$
\Delta r =
|r(\mathrm{EEG}, A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}})|
- |r(\mathrm{EEG}, A_{\mathrm{control}})|.
$$

This statistic is tested using variogram-matched BrainSMASH spatial surrogates (Burt et al., 2020).
For each surrogate EEG map, the maximum absolute surrogate-control correlation is recomputed across
the full prespecified control-map set before calculating surrogate $\Delta r$. The report includes
$\Delta r$, surrogate p-value, and absolute EEG-fMRI spatial correlation.

The fMRI target is smoothed with the empirically estimated sLORETA point-spread FWHM derived from
the source-resolution analysis before outcome-map inspection. Sensitivity analyses span the
subject-level point-spread interquartile range and one narrower and one broader kernel. Spatial
similarity is evaluated with Pearson correlation across matched cortical vertices.

Variogram-matched spatial surrogates are used as the primary spatial null. BrainSMASH generates
5,000 surrogate maps vertex-wise on the fsaverage cortical surface using geodesic distance,
independent variograms per hemisphere, and exclusion of medial wall vertices. The identical cortical
mask is applied to EEG and fMRI maps. Spherical spin permutations are reported as a sensitivity
analysis using 10,000 rotations (Alexander-Bloch et al., 2018).

## 13. Internal Cross-Validated Criterion Overlap

Internal criterion overlap tests whether the derived pattern is associated with the prespecified
behavioral intensity criterion. The primary analysis uses the same two contribution-matched
full-plateau source-power association maps, $A_\alpha$ and $A_\beta$, with behavioral p-values
Holm-corrected across bands as a separate internal-convergence family.

For each target subject, the group-level pattern is estimated from all remaining subjects. The
held-out subject's source power is transformed with the same log-power, source-stage
residualization, and within-subject vertex-wise standardization pipeline used for source-map
construction. Behavioral expression is computed separately for alpha and beta maps.

$$
\text{expression}_{s,i}^{(\text{band})} =
\sum_v
\widetilde{\text{sourcepower}}_{s,i}^{(\text{band})}(v)
\cdot \text{Pattern}_{-s}^{(\text{band})}(v).
$$

Here, $\widetilde{\text{sourcepower}}$ denotes source power after the same log transformation,
nuisance residualization, and vertex-wise standardization used in the primary source-stage
association maps.

Before projection into the held-out subject, each leave-one-subject-out group pattern is restricted
to the cortical analysis mask and L2-normalized across vertices with its original sign preserved. A
zero-norm pattern is ineligible for criterion-overlap analysis.

The primary behavioral criterion-overlap test evaluates association with the Study 1 within-scale
thermal/pain intensity score beyond stimulus, acquisition, and session structure.
Painful-trial-only intensity association and a two-part sensitivity analysis separating binary
pain report from intensity conditional on pain are reported as criterion-specific sensitivity
analyses. A subject is excluded from criterion-overlap analyses if any retained source-valid
trial lacks a synchronized rating, fewer than 25 rated source-valid plateau trials remain, fewer
than three permutation-valid task blocks remain, or within-subject rating variance is zero after
source-stage censoring. Missing ratings are not imputed.

Within each held-out subject, the association between expression score and the within-scale
thermal/pain intensity score is estimated with ordinary least squares regression controlling for the
fixed Study 1 Level 2 nuisance design. The primary test statistic is the group-level mean of
within-subject standardized regression coefficients. Statistical testing is two-sided and uses
5,000 permutations. Within each subject and permutation, expression is circularly shifted relative
to rating within task block, coefficients are recomputed, and the group mean is aggregated.
Group-level intervals resample subjects. Within-subject intervals use task-block-level resampling.
Behavioral convergence is summarized by the same-band standardized coefficient, its bootstrap
interval, and Holm-corrected p-value. Positive significant coefficients support thermal/pain
intensity convergence; significant negative coefficients are reported separately.

Pain-specific convergence is evaluated with painful-trial-only intensity association and the
intensity component of the two-part model conditional on pain. If adjusted pain-class or
painful-trial rating variance is insufficient, the corresponding sensitivity analysis is reported
as unavailable.

The pain/non-pain classification contrast is included when pain class retains meaningful residual
variance beyond the nonlinear temperature basis. Meaningful residual variance is defined as at
least 10% of the unadjusted within-subject pain-class variance remaining after adjustment in at
least 30 source-valid subjects, with nonzero adjusted pain-class variance in every retained subject
for that contrast.

## 14. Reporting and Robustness Controls

Subject-wise prediction performance, criterion-overlap coefficients, artifact metrics, and
map-expression distributions are reported with 95% BCa bootstrap confidence intervals using
10,000 resamples. Group-level intervals resample subjects. Within-subject intervals use
task-block-level or circular block bootstrap resampling.

All primary analyses are repeated under prespecified robustness conditions. Artifact censoring
thresholds match Study 1. Trials are censored when framewise displacement exceeds 0.5 mm, DVARS
robust z exceeds 3, Fp1/Fp2 high-frequency artifact power robust z exceeds 3, cardiac phase-locking
exceeds the 95th percentile of the within-subject null, or scanner-frequency residual peaks exceed
robust z > 3.

Analyses excluding artifact-contaminated maps are sensitivity analyses. Analyses are also repeated
after equalizing trial counts. Robustness summaries report preserved statistical
significance, retained cluster sign, spatial correlation of unthresholded maps, Dice overlap of
thresholded clusters, and centroid displacement within the same broad anatomical region.

## References

Alexander-Bloch, A. F., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C., Shinohara, R. T.,
Vandekar, S. N., & Raznahan, A. (2018). On testing for spatial correspondence between maps of
human brain structure and function. NeuroImage, 178, 540-551.
doi.org/10.1016/j.neuroimage.2018.05.070

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
