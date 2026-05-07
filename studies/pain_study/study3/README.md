# Study 3 - Source Interpretation of NPS-Predictive EEG Activity

## 1. Problem Statement

Study 1 tests whether plateau-window EEG spectral power predicts trial-wise fMRI pain-signature
expression beyond prespecified nuisance structure. A successful prediction model alone does not
identify the cortical sources carrying the predictive information. Multivariate decoders are
backward models: their raw weights can reflect noise cancellation, feature covariance,
preprocessing choices, or artifact structure rather than the neural activity that gives rise to
pain-relevant signal (Haufe et al., 2014).

Study 3 addresses this interpretability problem by testing whether the predesignated EEG prediction
component from Study 1 is associated with spatially coherent cortical source-power patterns. These
patterns must remain interpretable after artifact, regularization, baseline, and permutation
controls. The analysis estimates source-power association patterns linked to the EEG-derived
residual prediction of NPS expression, without claiming causal localization of pain processing or
direct identification of cortical generators.

The source-interpretation target is restricted a priori to the Study 1 NPS ElasticNet alpha+beta
individual-channel spectral-power cell. This restriction follows the thesis hierarchy rather than
observed Study 1 performance ranking. NPS is the Study 1 target most closely tied to evoked
nociceptive processing, whereas SIIPS1 indexes a related but distinct
stimulus-intensity-independent pain-signature construct. SIIPS1, Ridge, Random Forest,
BandTemporalRegressor, Level 1, Level 3, ROI-level, and global-average analyses are reported only
as sensitivity or exploratory analyses in Study 3.

## 2. Objectives and Confirmatory Boundary

The primary objective is to derive band-specific standardized source-power association maps from
the frozen out-of-sample transformed-space EEG contribution scores of the predesignated Study 1 NPS
ElasticNet alpha+beta individual-channel spectral-power cell. The analysis is interpretive: it asks
where source-power fluctuations covary with the portion of the EEG signal used by the Study 1 model
for NPS prediction.

The Study 3 confirmatory family comprises two contribution-matched full-plateau source maps. The
alpha map, denoted $A_\alpha$, associates processed alpha source power with the standardized alpha
contribution score. The beta map, denoted $A_\beta$, associates processed beta source power with
the standardized beta contribution score.

Study 3 proceeds confirmatorily only if the predesignated Study 1 NPS ElasticNet alpha+beta
individual-channel spectral-power cell satisfies all Study 1 gates. These gates require significant
positive out-of-sample $\Delta R^2_{\text{LOSO}}$, passage of the Study 1 practical-effect gate,
Level 2 residualized-target $\Delta R^2_{\text{LOSO}} \geq 0.005$, positive
within-subject-centered diagnostic $\Delta R^2_{\text{LOSO}}$, successful temporal negative
controls, and artifact-censoring robustness for the same prediction cell. Holm-corrected Level 2
significance strengthens the convergence interpretation but is not required for confirmatory
Study 3 eligibility. If any required gate is not satisfied, all Study 3 source maps, multimodal
spatial comparisons, and behavioral associations are labeled exploratory.

This confirmatory chain is intentionally high-specificity and may yield an exploratory Study 3 even
when Study 1 provides scientifically useful prediction evidence. If Study 3 is downgraded, its
dissertation role is limited to feasibility, sensitivity, and hypothesis-generation about the source
structure of the predesignated EEG prediction component.

Confirmatory source-map success is evaluated separately for $A_\alpha$ and $A_\beta$. Evidence
cannot be assembled across bands. A band may support a candidate NPS-predictive EEG source-power
association pattern when that same band shows a primary source cluster surviving family-wise error
correction under the target-retrained null, no artifact contamination, valid source-stage inference,
and preserved primary robustness thresholds. Stronger NPS-convergent source interpretation requires
directional consistency with the band-matched true residual NPS source association map and no
dominant band-matched full-model prediction-error association. Stronger thermal/pain-intensity
convergence requires behavioral criterion-overlap support as defined in Section 13. Stronger
pain-relevance language additionally requires the pain-specific criterion defined in Section 13.
fMRI spatial correspondence is descriptive and does not serve as a confirmatory gate.

## 3. Frozen Study 1 Model and Band Contributions

All interpretive analyses use out-of-sample data. The selected Study 1 model class, feature family,
frequency preset, preprocessing recipe, outer folds, target transformation, and fold-specific
hyperparameters are frozen before source-map construction. Study 3 cannot influence Study 1 model
hyperparameters, residualization level, feature family, frequency band, or time window.

For each held-out subject, the frozen primary incremental Study 1 model generates one EEG residual
prediction per retained plateau trial, denoted
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$. The nuisance-only prediction and the full
combined raw prediction are retained only for performance reporting.

Because the alpha+beta feature preset combines two spectral bands, Study 3 uses the frozen linear
contribution scores from the same ElasticNet model rather than assigning the same scalar prediction
to both bands. For each outer fold, the standardized linear predictor is decomposed in the
transformed residual-target space, before inverse target transformation, into
$\eta_\alpha = X_\alpha W_\alpha$ and $\eta_\beta = X_\beta W_\beta$. Each contribution score is
then centered and scaled to unit variance within the held-out subject and band using that subject's
retained plateau trials. Zero-variance contribution scores are ineligible for source-stage
analysis.

The band contribution scores are model-contribution variables, not additive raw-scale NPS residual
predictions. Source maps are interpreted as associations with transformed-space alpha or beta
contributions from the frozen ElasticNet decoder. Raw-scale
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values are used for performance reporting and
full-prediction sensitivity maps, but they are not decomposed into raw-scale band contributions.

Each band-specific source analysis includes the standardized opposite-band contribution score as a
prespecified nuisance regressor. Alpha maps are interpreted as alpha-specific only if they survive
adjustment for $\eta_\beta$, and beta maps only if they survive adjustment for $\eta_\alpha$.
Full-prediction band association maps using $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$
are reported beside the unique contribution maps to characterize shared alpha-beta prediction
structure. These full-prediction maps are sensitivity analyses and do not support band-specific
claims.

Band-unique source maps are eligible for confirmatory interpretation only when the alpha/beta
contribution decomposition is numerically stable. Before source mapping, the within-subject
source-stage design containing $\eta_\alpha$, $\eta_\beta$, and the fixed nuisance terms must have
condition number ≤ 100 after centering and scaling non-intercept columns, and the opposite-band
contribution term must have variance inflation factor ≤ 5 for every retained source subject.
Subjects failing these gates are excluded from band-unique confirmatory maps. If fewer than
30 source-valid subjects remain, or if more than 20% of otherwise source-valid subjects fail the
collinearity gate, the band-unique alpha and beta maps are downgraded to exploratory and only the
shared full-prediction maps are reported descriptively.

## 4. Sensor-Level Interpretation

Sensor-level linear feature patterns are interpreted with the Haufe transformation when the model is
linear (Haufe et al., 2014). For the primary ElasticNet model, the pattern is computed within each
outer fold using training-fold statistics only.

$$A_{\mathrm{Haufe}} = \Sigma_{X,\mathrm{train}} \, W.$$

Here, $\Sigma_{X,\mathrm{train}}$ is the training-fold feature covariance matrix and $W$ is the
fitted weight vector in the same feature space. Because Study 1 standardizes features to zero mean
and unit variance within each fold, $W$ is estimated in standardized units and
$\Sigma_{X,\mathrm{train}}$ is the training-fold correlation matrix.

The Haufe transformation is used only where its assumptions apply. For nonlinear models,
$\mathrm{Cov}(X, \hat{y})$ does not carry formal Haufe guarantees. Random Forest and
BandTemporalRegressor maps are therefore reported as prediction-covariance association maps without
causal or forward-model interpretation.

$$A_{\mathrm{assoc}} \propto \mathrm{Cov}(X_{\mathrm{features}},\; \hat{y}_{\mathrm{test}}).$$

Only spectral power features enter source-space follow-up. Source-space power is recomputed from
inverse-projected voltage time series rather than from projected sensor-level power maps. Nonlinear
complexity metrics are not projected through the lead field because they are not electric-field
quantities. Complexity topographies remain sensor-level analyses.

## 5. Artifact Controls

Artifact control is treated as an interpretive gate rather than a secondary cleaning step.
Sensor-level maps are compared against artifact topographies derived from independent calibration
data or from ICA components labeled as ocular, cardiac, scanner, or high-frequency frontal
artifacts. Template correlations are treated as flags, not correction procedures. Maps are not
orthogonalized against artifact templates because orthogonalization can remove neural signal when
neural and artifact topographies overlap.

For each sensor-level map, trial-wise map expression is computed by projecting each trial's feature
vector onto the map. Artifact association is tested by regressing this expression score against
framewise displacement, DVARS, cardiac phase, scanner-frequency residual power, and Fp1/Fp2
high-frequency artifact power. A map is labeled artifact-contaminated when its absolute spatial
correlation exceeds $r = 0.80$ with any artifact template or when its expression significantly
covaries with any artifact metric after Holm correction. If a primary confirmatory map is
artifact-contaminated, the source claim is ineligible for confirmatory interpretation. Contaminated
maps remain descriptive sensitivity outputs.

The same artifact gate applies in source space. For each primary source map, trial-wise
source-pattern expression is computed from the processed source-power matrix before group
aggregation. Expression is regressed against framewise displacement, DVARS, cardiac phase,
scanner-frequency residual power, and Fp1/Fp2 high-frequency artifact power within subject, then
tested at the group level with Holm correction across artifact metrics and the two primary bands.

Artifact-prediction source association maps are also generated by replacing the prediction-derived
score with each artifact metric after the same source-stage preprocessing. A primary source map is
labeled artifact-contaminated if source-pattern expression significantly tracks any artifact metric
after correction, if its unthresholded spatial correlation with an artifact-prediction source map
exceeds $|r| = 0.50$, or if the predesignated Study 1 NPS ElasticNet alpha+beta individual-channel
spectral-power cell does not retain the same sign, $\Delta R^2_{\text{LOSO}} \geq 0.02$, and
Holm-corrected significance after artifact censoring.

## 6. Exploratory Spatio-Temporal Mapping

The BandTemporalRegressor processes continuous epochs and therefore allows time-resolved
prediction-covariance mapping. Because this model is nonlinear and exploratory in Study 1, its
spatio-temporal maps are also exploratory.

Pain-evoked high-frequency oscillations are largely induced and non-phase-locked. The
spatio-temporal map is therefore computed on instantaneous power envelopes rather than raw voltage
to avoid destructive phase cancellation.

$$A_{\mathrm{assoc}}(t) \propto \mathrm{Cov}(E_{\mathrm{test}}(t),\; \hat{y}_{\mathrm{test}}).$$

This analysis produces maps with dimensions bands, channels, and time. Time-resolved analyses are
evaluated within predefined early, mid, and late plateau windows to control temporal multiplicity.

## 7. Source Modeling

Source-space analysis estimates cortical source-power association with the out-of-sample
band-specific contribution scores from the EEG-predicted NPS residual model. The source estimand is
defined at the level of band-limited voltage time series. Inverse modeling is therefore applied to
band-passed voltage rather than to sensor-level power or covariance topographies. Power is a
nonlinear, nonnegative summary of voltage and does not represent an electric field from cortical
dipoles. The resulting maps are described as inverse-model-dependent standardized source-power
association estimates, not as source-localized Haufe weight maps or causal localizations of
cortical generators.

The primary inverse operator uses sLORETA (Pascual-Marqui, 2002), FreeSurfer subject-specific head
models, and boundary element method forward solutions. Confirmatory forward solutions, noise
covariance matrices, and inverse operators exclude Fp1 and Fp2 before rank estimation to prevent
frontal artifact-proxy channels from influencing source reconstruction.

The primary noise covariance is derived from the −5.0 to −0.01 s pre-stimulus baseline, matching
Study 1. Because this interval may contain cue-locked expectancy activity, source localization is
repeated using the immediate pre-stimulus baseline (−0.2 to −0.01 s) and the early pre-cue
baseline (−7.0 to −5.5 s). A result is baseline-stable only when the primary spatial pattern,
cluster polarity, and internal criterion-overlap inference are preserved under the immediate
baseline sensitivity.

A single inverse operator is estimated with the broadband noise covariance and applied identically
to voltage time series filtered into each canonical band. The primary source time series uses the
cortical surface-normal component from the loose-orientation inverse. Vector output and
orientation-norm pooled estimates are retained only as orientation-sensitivity analyses because
they change the source-power estimand. Instantaneous power envelopes are computed from the signed
normal-orientation source time series with the Hilbert transform. Because source power is
nonnegative, positive and negative source-association map values indicate the direction of
association between source power and the standardized prediction-derived score, not positive or
negative current flow.

Confirmatory source-power construction mirrors the Study 1 individual-channel spectral-power
estimand. For each LOSO held-out subject, the fold-specific condition-agnostic Study 1 ERP template
is subtracted in sensor space before band-pass filtering and inverse projection. Source-space
Hilbert power is then baseline-corrected as a log-ratio using the same primary baseline
(−5.0 to −0.01 s) and averaged over the same active plateau window (3.0 to 10.5 s). Absolute
plateau log-power maps that omit this ERP and baseline alignment are descriptive sensitivity maps
only and cannot replace the confirmatory source-power maps.

Source modeling quality-control checks are completed before source-map inspection. A subject is
excluded from source-stage analyses if FreeSurfer reconstruction fails visual quality control, the
boundary element model fails, subject-specific measured electrode positions are unavailable,
template-only electrode coordinates are used, fewer than 90% of retained EEG channels have valid
locations, mean electrode-coregistration error exceeds 5 mm, maximum electrode-coregistration error
exceeds 10 mm, the forward solution contains invalid or rank-deficient channels, or morphing to
fsaverage fails. These checks are fixed exclusion rules rather than source-model repair steps.

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
4. Trial onset time and linear trial number within task block.
5. HRF-weighted framewise displacement.
6. HRF-weighted standardized DVARS.
7. Fp1/Fp2 high-frequency artifact power.
8. Residual ECG coupling.
9. The opposite-band contribution score for band-specific maps.

This design is identical for every retained source subject except for the prespecified opposite-band
contribution term. Alpha maps use standardized $\eta_\beta$, and beta maps use standardized
$\eta_\alpha$. In each permuted analysis, the corresponding permutation-specific standardized
contribution score is used. No covariate class is added or dropped within subject, task block, band,
or permutation. If the fixed categorical encoding is not full rank for a subject after source-stage
censoring, that subject is excluded from the confirmatory source analysis. The continuous nonlinear
temperature basis is retained only as a sensitivity analysis to match Study 1.

After this processing, each vertex value is a standardized regression coefficient equivalent to the
within-subject partial correlation between processed source power and the processed
prediction-derived score. Subject-level maps retain the raw partial-correlation-scale values for
descriptive reporting and Fisher-z-transformed values for group inference. The primary source map
therefore represents the portion of the Study 1 transformed-space band contribution that remains
after additional removal of discrete stimulus intensity, selected thermode surface, task-block
structure, trial order, acquisition-noise structure, and cross-band prediction association.

The unique-contribution estimand is intentionally conservative, but it can suppress shared
alpha-beta pain-related variance. Reports therefore include full-prediction band association maps
with identical artifact and robustness summaries as descriptive sensitivity results. A parallel
unadjusted source-power association map using the original
$\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values, with only task-block intercepts
removed, is also reported as a sensitivity analysis.

Circular-shift source permutations use the six 11-trial task blocks as exchangeability units,
matching Study 1. Source-stage censoring uses the same variable-length retained-block rule: a
permutation-valid source block must retain at least 8 source-valid plateau trials and allow at
least four admissible nonzero circular shifts under the Study 1 retained-sequence shift definition.
Partial source blocks satisfying these criteria remain in confirmatory source-map computation;
partial blocks that fail these criteria are excluded. A subject is excluded from source-stage
analyses if fewer than three permutation-valid task blocks remain, fewer than 25 clean plateau
trials remain, any primary source-stage nuisance covariate is unavailable, or the fixed
source-stage design is not full rank after censoring. Full rank alone is not sufficient for
inclusion. A subject must also have at least 15 residual degrees of freedom after source-stage
censoring, defined as retained source-valid trials minus the rank of the fixed source-stage design.
If fewer than 30 subjects remain after this residual-degrees-of-freedom gate, confirmatory source
inference is downgraded to exploratory.
The source-stage quality-control report includes retained source-valid trial counts, retained
permutation-valid block counts, fixed-design rank, residual degrees of freedom, condition number,
and opposite-band VIF for every subject before source-map inspection.

## 9. Target and Error Association Maps

Each prediction-derived score is a deterministic function of EEG features. The primary map
therefore identifies source patterns associated with model output, not necessarily source patterns
associated with true residualized NPS expression. To distinguish contribution-associated source
patterns from true target association, parallel maps are computed using the actual held-out residual
target and the full-model prediction error.

$$\mathrm{Assoc}(\text{processed source power},\; r_{\mathrm{NPS-L2}}).$$

$$\mathrm{Assoc}(\text{processed source power},\; \hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}} - r_{\mathrm{NPS-L2}}).$$

For each primary band, the true-target and prediction-error maps use the same band-specific
source-power tensor and the same source-stage nuisance design as the corresponding contribution
map, including the opposite-band contribution regressor. This design makes the diagnostic maps
directly comparable with the band-unique primary map rather than with a less-adjusted source
association.

Directional consistency with the true residual NPS map is satisfied only when all three criteria
are met: the unthresholded Pearson spatial correlation between the primary contribution map and the
band-matched true-target map is ≥ 0.20 across the cortical analysis mask; the median true-target
association inside the primary FWE-corrected source cluster has the same sign as the primary cluster
statistic; and at least 60% of vertices in that cluster have true-target association values with the
primary cluster sign. Failure of this criterion does not erase the source-map result, but it
prevents the stronger NPS-convergent source interpretation for that band. The full-model
prediction-error association map is a diagnostic flag, not a band-specific error decomposition. It
is considered dominant if its absolute spatial correlation with a primary contribution-matched map
equals or exceeds that map's true-target association correlation, or if it produces a larger
FWE-corrected cluster in the same broad anatomical region. A dominant prediction-error association
does not erase source-map success, but it prevents stronger NPS-convergent interpretation.

## 10. Group-Level Source Inference

Confirmatory source inference requires at least 30 source-valid subjects after source-stage
inclusion checks. Analyses with 20-29 source-valid subjects are reported as feasibility-limited
exploratory analyses even if cluster tests are significant.

Before source-map inspection, a source-stage precision and calibration simulation is completed using
only source-valid trial counts, task-block structure, baseline source-noise covariance, observed
post-source-QC block lengths, admissible source-shift counts, source-stage nuisance matrices,
cortical adjacency, and each subject's empirical point-spread and cross-talk summaries. Simulated
maps include null maps and embedded smooth clusters with standardized within-subject partial
correlations of $|r| \in \{0.05, 0.10, 0.15, 0.20\}$ after source-stage nuisance removal. For each
retained-sample scenario, 2,000 Monte Carlo datasets are generated and tested with the same
cluster-forming thresholds, target-retrained permutation count, subject weighting, morphing, and
robustness checks planned for the observed maps.

The $|r| = 0.15$ recovery threshold defines the smallest source association considered
scientifically interpretable for this 64-channel EEG source-space analysis. Effects below this
level are treated as too small to separate reliably from inverse-solution point spread, morphing
error, residual artifact covariance, and block-level nuisance structure, even if a large sample or
liberal cluster-forming threshold makes them detectable. The embedded cluster width matches the
median empirical point-spread FWHM so the simulation tests recovery of a spatial pattern that the
actual inverse operator can resolve, not an unrealistically focal source. The lower
$|r| \in \{0.05, 0.10\}$ conditions calibrate false-positive behavior and sensitivity to negligible
effects; the $|r| \in \{0.15, 0.20\}$ conditions evaluate the boundary between a minimal
interpretable source association and a clearly recoverable one.

Confirmatory source interpretation requires acceptable null calibration and precision before
outcome-map inspection. The null family-wise error rate must fall between 0.025 and 0.075 for the
nominal 0.05 cluster test, the 95% bootstrap CI half-width for mean cluster expression at
$|r| = 0.15$ must be ≤ 0.10, and the simulation must show at least 80% recovery of an embedded
$|r| = 0.15$ cluster whose spatial extent matches the median empirical point-spread FWHM. Failure of
any simulation gate downgrades Study 3 source inference to exploratory, regardless of observed
cluster p-values.

Regularization uses SNR = 3.0, corresponding to $\lambda^2 \approx 0.111$, oct6 source spacing,
loose orientation constraint of 0.2, and depth weighting of 0.8. Before outcome-map inspection,
empirical point-spread and cross-talk functions are estimated from each subject's actual forward and
inverse operator and morphed to fsaverage. The median cortical point-spread FWHM and its regional
range are reported and used to choose the fMRI smoothing kernel for descriptive spatial comparison.
Sensitivity analyses use SNR values of 1.0 and 5.0. A source-space result is regularization-stable
only if the primary cluster retains significance and the same sign under both sensitivity values.

Subject-specific source maps are morphed to fsaverage and optionally normalized by Global Field
Power. Because map GFP is itself an outcome-associated derived metric, GFP-based map exclusion is
not used in the primary analysis. All subjects are included in the primary non-normalized group
aggregation. GFP normalization is retained strictly as a sensitivity analysis.

## 11. Statistical Validation

### 11.1 Permutation Strategy

Nonparametric permutation testing is performed on source-localized maps. Standard sign-flip
permutations are inappropriate because they violate exchangeability for this estimand. The
prediction-derived scores are deterministic functions of EEG features, so naive sign-flipping of
subject maps does not generate a valid null for feature-prediction association. Such permutations
could preserve spatial structure driven by the intrinsic covariance of human EEG even when the model
does not track the true fMRI target.

The primary confirmatory null is target-retrained with frozen Study 1 model-selection outputs after
the Study 1 gate has been evaluated with the full Study 1 permutation procedure. This source-stage
null tests map specificity for the selected source-interpretation model. It does not retest the
full source-selection procedure or the original Study 1 prediction-performance claim.

Observed and permuted contribution scores are generated from the original Study 1 prediction-valid
LOSO training folds. Source-stage censoring and source-valid subject restrictions are applied only
after contribution-score generation; they do not redefine the Study 1 training sample, target
transformation, preprocessing statistics, inner-validation results, or frozen hyperparameters.

Each permutation and outer fold proceeds as follows.

1. Circularly shift the Level 2 nuisance residual target relative to EEG within task block for
   training and held-out subjects, using the same minimum shift-distance rule as Study 1.
2. Reconstruct the permuted raw target as the unshifted nuisance prediction plus shifted residual.
3. Reuse the observed Study 1 feature set, preprocessing statistics, Yeo-Johnson transformation,
   fold-specific ElasticNet hyperparameters, and prediction-valid training sample.
4. Refit only the ElasticNet coefficients and intercept on the permuted training target, then apply
   the model to the permuted held-out subject to generate permuted alpha and beta contribution
   scores.
5. Refit the within-subject contribution-score standardization to each permuted held-out
   contribution vector.
6. Repeat source-stage nuisance control, source-map computation, group aggregation, and cluster
   testing. For each permuted band map, the opposite-band nuisance term is the standardized
   opposite-band contribution score from the same permuted model.

This null tests whether the observed standardized source-power association map exceeds the map
expected from the same EEG covariance structure and frozen model-selection procedure after the fMRI
target relationship has been broken. Circularly shifting the final observed contribution scores
relative to source power is retained only as a secondary model-anatomy sensitivity null.

The executable permutation plan is fixed before source-map inspection. Band-specific source
log-ratio power tensors, permutation-invariant nuisance covariates, cortical adjacency, and
point-spread summaries are cached before permutation testing. Source-stage residualization matrices
are rebuilt for each observed and permuted band map whenever they depend on prediction-derived
contribution scores.

Confirmatory inference uses 1,000 valid target-retrained permutation draws as the minimum executable
null. This gives a minimum attainable cluster-tail probability of approximately 0.001 with the
standard $(b+1)/(m+1)$ correction and is adequate for the two-band Holm family. Permutation validity
is evaluated at the draw level across both bands. A draw is invalid if either band produces
zero-variance held-out contribution scores for any retained source subject, if the source-stage
design with the permutation-specific opposite-band term is not full rank, or if the design fails the
same condition-number or VIF gates used for observed band-unique maps. Invalidity in either band
discards the entire permutation draw for both bands so the alpha and beta nulls remain paired.
Invalid draws are resampled before outcome-map inspection. If more than 20% of attempted draws are
invalid, or if 1,000 valid draws cannot be obtained within the prespecified compute budget,
source-space inference is downgraded to exploratory. If either band has an uncorrected cluster
p-value below 0.10 or a maximum cluster statistic within 10% of the 95th-percentile null threshold,
both bands are extended to 5,000 valid paired draws before final reporting.

A smaller full-selection sensitivity null repeats Study 1 inner GroupKFold hyperparameter selection
and source-map computation for at least 250 permutations before final interpretation. Failure to
complete this sensitivity null within the compute budget does not invalidate the selected-model
source null, but it prevents claims about full source-selection stability. If the cached
implementation cannot complete at least 1,000 target-retrained permutations within the prespecified
72 h compute budget on the available workstation or cluster allocation, source-space inference is
downgraded to exploratory. The null size is not reduced based on interim results.

### 11.2 Cluster-Based Inference

Group-level inference uses cluster-based permutation testing on the primary ElasticNet model only
(Maris & Oostenveld, 2007). The confirmatory family comprises the two contribution-matched
full-plateau maps, $A_\alpha$ and $A_\beta$, both derived from the gated Study 1 NPS ElasticNet
alpha+beta individual-channel spectral-power cell.
Each subject contributes one Fisher-z-transformed source-power association map per band after
morphing to fsaverage. Positive values indicate greater processed source power on trials with larger
band-specific prediction contribution after nuisance and opposite-band adjustment. Vertex-wise
one-sample $t$ statistics test whether the mean association differs from zero.

Vertex-wise statistics exceeding an uncorrected cluster-forming threshold of $p < 0.01$ are
aggregated into contiguous sign-preserving clusters. Results are checked under thresholds of
$p < 0.001$ and $p < 0.05$. Stability requires the primary cluster to retain significance and the
same sign. Family-wise error correction uses a two-sided maximum statistic across positive and
negative clusters within each band. Band-level p-values for $A_\alpha$ and $A_\beta$ are
Holm-corrected as the confirmatory source family.

Behavioral criterion-overlap p-values are Holm-corrected as a separate two-band internal
convergence family. fMRI spatial correspondence remains descriptive and does not share a
confirmatory family with source clusters or behavioral overlap. Secondary summaries spanning
plateau subwindows and alternative feature families are Holm-corrected as one exploratory family.

Equal subject weighting is used for the primary analysis. Baseline-noise inverse-variance weighting
is evaluated as a sensitivity analysis. Leave-one-subject-out influence diagnostics are applied to
the source cluster. These diagnostics evaluate the maximum change in cluster mass after removing
one subject, whether the cluster remains directionally consistent, and whether the result depends
on one or two high-amplitude outlier maps.

### 11.3 Gamma-Band Interpretation

Gamma-band source maps are exploratory regardless of statistical significance because gamma EEG
during simultaneous fMRI is vulnerable to non-neural artifacts. Gamma findings do not designate a
candidate pain-relevant EEG pattern.

## 12. Multimodal Spatial Comparison

Spatial correspondence between the EEG source-power association pattern and fMRI pain-related
covariance is evaluated as a descriptive within-cohort spatial-consistency analysis. It is not
treated as independent validation. The comparison is restricted to the cortical subnetwork because
the full NPS includes subcortical structures that EEG source modeling cannot recover reliably.

Comparing the EEG forward-pattern map to published NPS backward-model weights would create a
theoretical asymmetry (Haufe et al., 2014). The primary comparison therefore uses a within-study
fMRI forward covariance pattern.

$$A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}} \propto \mathrm{Cov}(B_{\mathrm{fMRI}}^{\mathrm{resid}},\; r_{\mathrm{NPS-L2}}).$$

Here, $B_{\mathrm{fMRI}}^{\mathrm{resid}}$ denotes trial-wise fMRI beta maps after voxelwise
residualization against the same Level 2 nuisance family used in Study 1. The term
$r_{\mathrm{NPS-L2}}$ denotes fold-contained residualized NPS expression after removal of the Level
2 nuisance family. Both the fMRI covariance target and voxelwise nuisance coefficients are
estimated exclusively from training-fold subjects within the same LOSO framework. The final
descriptive fMRI cortical vector is the equal-weight mean of the LOSO training-fold fMRI covariance
maps across held-out source-valid subjects, after projecting each fold map to the Study 3 cortical
analysis mask. No EEG source outcome map, behavioral-overlap result, or spatial-comparison result
enters this fMRI vector construction.

The EEG source association map and $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ are derived with respect
to the same or closely related NPS Level 2 target variable. Their spatial correlation can therefore
be inflated by shared target dependency. Spatial specificity tests evaluate whether the observed
correlation exceeds expectations from spatially autocorrelated noise under a shared-target
structure, but they cannot fully separate shared-target inflation from genuine multimodal
neurophysiological convergence. Independent-cohort replication with a held-out fMRI covariance
target is required for a strong convergence claim.

Specificity is tested as a descriptive superiority analysis. The fMRI spatial comparison is
performed separately for $A_\alpha$ and $A_\beta$ with the same fMRI cortical vector. The EEG source
map is compared with $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ and with control maps derived from
circular-shifted NPS scores, motion parameters, physiological noise regressors, stimulus variables,
and pre-stimulus EEG predictions. Subjective-rating maps are evaluated separately as
criterion-overlap maps. For each band, $A_{\mathrm{control}}$ is the prespecified control map with
the largest absolute EEG-control spatial correlation, making the descriptive superiority contrast
conservative. The primary descriptive statistic is

$$\Delta r = |r(\mathrm{EEG}, A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}})| - |r(\mathrm{EEG}, A_{\mathrm{control}})|.$$

This statistic is tested using variogram-matched BrainSMASH spatial surrogates (Burt et al., 2020).
For each surrogate EEG map, the maximum absolute surrogate-control correlation is recomputed across
the full prespecified control-map set before calculating surrogate $\Delta r$. This preserves the
max-control selection in the null and prevents anticonservative superiority claims. fMRI spatial
consistency may support a spatial-convergence interpretation, but it cannot change the internal
convergence criterion. In addition to $\Delta r$ significance, the absolute EEG-fMRI spatial
correlation must exceed $|r| \geq 0.15$ before spatial correspondence is discussed as
neurophysiologically meaningful.

Because inverse-solution point spread is not spatially stationary and varies by cortical location
and depth, the fMRI target is smoothed with the empirically estimated sLORETA point-spread FWHM
derived from the source-resolution analysis before outcome-map inspection. Sensitivity analyses
span the subject-level point-spread interquartile range and one narrower and one broader kernel.
Spatial similarity is evaluated with Pearson correlation across matched cortical vertices. No sign
is hard-coded as valid overlap because alpha or beta power and BOLD can be negatively correlated
depending on region and task.

Because spin permutations can behave poorly on restricted, irregular cortical masks,
variogram-matched spatial surrogates are used as the primary spatial null. BrainSMASH generates
5,000 surrogate maps vertex-wise on the fsaverage cortical surface using geodesic distance,
independent variograms per hemisphere, and exclusion of medial wall vertices. The identical cortical
mask is applied to EEG and fMRI maps. Spherical spin permutations are reported as a sensitivity
analysis using 10,000 rotations (Alexander-Bloch et al., 2018).

## 13. Internal Cross-Validated Criterion Overlap

Internal criterion overlap characterizes whether the derived pattern is associated with the
prespecified behavioral intensity criterion. It is not described as independent validation. The
primary analysis uses the same two contribution-matched full-plateau source-power association maps,
$A_\alpha$ and $A_\beta$, that define the confirmatory source family. Their behavioral p-values are
Holm-corrected across the two bands as a separate internal-convergence family. No vertex set or
cluster is selected from the source-space significance map.

For each target subject, the group-level pattern is estimated from all remaining subjects. The
held-out subject's source power is transformed with the same log-power, source-stage
residualization, and within-subject vertex-wise standardization pipeline used for source-map
construction. Behavioral expression is computed separately for alpha and beta maps.

$$\text{expression}_{s,i}^{(\text{band})} = \sum_v \widetilde{\text{sourcepower}}_{s,i}^{(\text{band})}(v) \cdot \text{Pattern}_{-s}^{(\text{band})}(v).$$

Here, $\widetilde{\text{sourcepower}}$ denotes source power after the same log transformation,
nuisance residualization, and vertex-wise standardization used in the primary source-stage
association maps.

Before projection into the held-out subject, each leave-one-subject-out group pattern is restricted
to the cortical analysis mask and L2-normalized across vertices with its original sign preserved. A
zero-norm pattern is ineligible for criterion-overlap analysis. No cluster mask, sign flip, or
outcome-informed vertex selection is applied.

The primary behavioral criterion-overlap test evaluates association with the Study 1 within-scale
thermal/pain intensity score beyond stimulus, acquisition, and session structure. This score is not
interpreted as a single linear pain continuum across non-painful and painful trials. The raw 0 to
200 displayed rating is described separately. Painful-trial-only intensity association and a
two-part sensitivity analysis separating binary pain report from intensity conditional on pain are
reported as criterion-specific sensitivity analyses. A subject is excluded from criterion-overlap
analyses if any retained source-valid trial lacks a synchronized rating, fewer than 25 rated
source-valid plateau trials remain, fewer than three permutation-valid task blocks remain, or
within-subject rating variance is zero after source-stage censoring. Missing ratings are not
imputed.

Within each held-out subject, the association between expression score and the within-scale
thermal/pain intensity score is estimated with ordinary least squares regression controlling for the
fixed Study 1 Level 2 nuisance design. The primary test statistic is the group-level mean of
within-subject standardized regression coefficients. Statistical testing is two-sided and uses
5,000 permutations. Within each subject and permutation, expression is circularly shifted relative
to rating within task block, coefficients are recomputed, and the group mean is aggregated.
Group-level intervals resample subjects. Within-subject intervals use task-block-level resampling.
A band satisfies the behavioral criterion-overlap component only when its same-band standardized
coefficient is positive and Holm-corrected significant. Significant negative associations are
reported but do not support behavioral convergence.

For a band that has already met the source-association criteria, within-scale support justifies the
label "candidate thermal/pain-intensity source-power association pattern." The stronger label
"candidate pain-relevant EEG source-power association pattern" additionally requires directionally
consistent positive support in at least one pain-specific sensitivity analysis:
painful-trial-only intensity association or the intensity component of the two-part model
conditional on pain. If these pain-specific sensitivities are ineligible because adjusted pain-class
or painful-trial rating variance is insufficient, the stronger pain-relevance label is not used.

The pain/non-pain classification contrast may be deterministically coupled to the temperature
manipulation. It is therefore dropped unless pain class exhibits meaningful residual variance
beyond the nonlinear temperature basis. Meaningful residual variance requires at least 10% of the
unadjusted within-subject pain-class variance to remain after adjustment in at least 30 source-valid
subjects, with nonzero adjusted pain-class variance in every subject retained for that contrast.
Absence of adjusted rating association does not prove that the EEG pattern is unrelated to pain, but
it prevents strong pain-relevance claims. The label "EEG pain signature" is reserved for future
independent-cohort replication.

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

Analyses excluding artifact-contaminated maps are descriptive sensitivity analyses and cannot alter
primary interpretation if the primary map is contaminated. Analyses are also repeated after
equalizing trial counts. A candidate pattern is considered robust only when prespecified
quantitative thresholds are met across sensitivity controls. These thresholds require preserved
statistical significance, retained cluster sign, spatial correlation of unthresholded maps of at
least 0.50, Dice overlap of at least 0.40 for thresholded clusters, and centroid displacement no
greater than 15 mm within the same broad anatomical region. Gamma-band maps remain exploratory
regardless of these robustness results.

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
