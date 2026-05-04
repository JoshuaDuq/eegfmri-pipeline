# Materials and Methods

## Research Objectives
This study interpreted the primary predictive model from Study 1 by deriving spatially resolved EEG source-power covariance patterns associated with its trial-wise predictions. The primary interpretive target was fixed before source-map inspection: the non-transductive ElasticNet model trained on Level 2 stimulus- and acquisition-controlled NPS expression from the alpha+beta spectral-power preset in Study 1. This hierarchy does not demote SIIPS1 as a Study 1 co-primary prediction target; it narrows the source-interpretation target to NPS because the source claim specifically concerns nociceptive fMRI signature expression, whereas SIIPS1 indexes a distinct stimulus-intensity-independent construct. SIIPS1, Level 1, Level 3, Ridge, Random Forest, and BandTemporalRegressor were evaluated only as sensitivity or exploratory analyses.

Study 3 was gated on Study 1: confirmatory source interpretation was permitted only if all of the following held for the Study 1 Level 2 NPS ElasticNet alpha+beta model: (a) positive mean LOSO $R^2$, (b) Holm-corrected permutation $p \leq 0.05$, (c) no pre-stimulus temporal negative control yielded $R^2$ exceeding 50% of the active-window $R^2$, and (d) no single HRF/timing, baseline, or artifact robustness analysis reversed the sign of the mean $R^2$ or caused loss of Holm-corrected significance. If any gate condition failed, all Study 3 maps and behavioral associations were reported as exploratory. The Haufe et al. (2014) forward-backward duality formally applies to the primary ElasticNet model. Random Forest and BandTemporalRegressor source-space maps are reported as prediction-covariance association maps without formal forward-model guarantees.

## Model Ingestion and Out-of-Sample Prediction
All interpretive analyses were performed strictly on out-of-sample data. Following the LOSO framework from Study 1, finalized predictive architectures were frozen before source analysis. For each subject, the held-out test-set EEG data ($X_{test}$) was fed through the frozen model to generate a single scalar prediction per trial ($\hat{y}_{test}$) representing predicted Level 2 NPS expression. No Study 3 result was used to select Study 1 model hyperparameters, residualization level, feature family, frequency band, or time window.

## Prediction-Associated Sensor-Level Topographies
In multivariate decoding, predictive algorithms function as backward models mapping brain data to a target variable. Raw backward-model weights can highlight noise-canceling sensors rather than true signal sources (Haufe et al., 2014). The formal Haufe transformation was used where its assumptions applied; empirical prediction-covariance maps were used otherwise.

For the primary linear model (ElasticNet), the Haufe transformation is:

$$A_{Haufe} = \Sigma_{X,train} \, W$$

where $\Sigma_{X,train}$ is the training-fold data covariance matrix and $W$ is the fitted weight vector, both defined in the same feature space. Because Study 1 standardized features to zero mean and unit variance within each fold, $W$ was estimated in standardized units and $\Sigma_{X,train}$ was accordingly the training-fold correlation matrix $R_{train}$ (i.e., the covariance of the standardized features). The pattern was computed within each outer fold using training-fold statistics only.

For nonlinear models (Random Forest, BandTemporalRegressor), $\mathrm{Cov}(X, \hat{y})$ does not carry formal Haufe guarantees. These maps were designated prediction-covariance association maps without claims of causal feature attribution:

$$A_{assoc} \propto \mathrm{Cov}(X_{features},\; \hat{y}_{test})$$

Only spectral power features were subjected to source-space follow-up; source-space power was recomputed from inverse-projected voltage time series rather than from projected sensor-level power maps. Projecting nonlinear complexity metrics through a lead-field matrix is physically invalid; sensor-level topographies for complexity features are reported separately.

## Artifact Sensitivity Analysis
Sensor-level maps were compared against artifact topographies derived from independent calibration data or from ICA components labeled as ocular, cardiac, scanner, or electromyographic artifacts. Template correlations were treated as flags, not correction procedures. Maps were not orthogonalized against artifact templates because orthogonalization can remove neural signal when neural and artifact topographies spatially overlap.

A map was labeled artifact-contaminated when its absolute spatial correlation exceeded $r = 0.8$ with any artifact template or when its expression covaried with framewise displacement, DVARS, cardiac phase, scanner-frequency residual power, or temporal/frontal high-frequency EMG power after Holm correction. Contaminated maps were excluded from confirmatory interpretation.

## Spatio-Temporal Prediction-Covariance Mapping for the Deep Network
The BandTemporalRegressor processes continuous epochs, enabling time-resolved interpretation. Because the covariance computation requires only the model's scalar prediction $\hat{y}$, it was extended across time. As a nonlinear model, the resulting maps are prediction-covariance association maps.

Because pain-evoked high-frequency oscillations are largely induced and non-phase-locked, the spatio-temporal map was computed on instantaneous power envelopes rather than raw voltage to avoid destructive phase cancellation:

$$A_{assoc}(t) \propto \mathrm{Cov}(E_{test}(t),\; \hat{y}_{test})$$

This produced a map of shape bands $\times$ channels $\times$ time. Because the deep network is exploratory (see Study 1), these temporal maps are interpreted as exploratory characterizations. Time-resolved analyses were evaluated within predefined windows (early, mid, and late plateau) to control temporal multiplicity.

## Anatomical Source Projection
Source-space analysis projected prediction-associated EEG variance into cortical source space while respecting linearity constraints of the electromagnetic forward model. Inverse modeling was applied to band-passed voltage time series rather than to sensor-level power or covariance topographies, because power is a nonlinear, nonnegative summary of voltage that does not represent an electric field from cortical dipoles. The resulting maps are spatially smoothed source-space covariance projections, not causal localization of cortical origins.

The sLORETA inverse operator (Pascual-Marqui, 2002) was used, with forward solutions derived from FreeSurfer subject-specific head models and boundary element method solutions. The primary noise covariance was derived from the pre-stimulus baseline ($-5.0$ to $-0.01$ s), matching Study 1. Because this interval may contain cue-locked expectancy activity, source localization was repeated using the immediate pre-stimulus window ($-0.2$ to $-0.01$ s) and the early pre-cue window ($-7.0$ to $-5.5$ s). A result was considered baseline-stable only when the primary spatial pattern, cluster polarity, and behavioral-validation inference were preserved under the immediate-baseline sensitivity.

The inverse operator is computed once from broadband data and applied identically to voltage time series filtered into each canonical band. Instantaneous power envelopes were computed from source time series via the Hilbert transform. Prediction-associated source-power covariance was then computed as $\mathrm{Cov}(\text{source power},\; \hat{y}_{test})$ directly in source space.

Regularization was configured with SNR $= 3.0$ ($\lambda^2 \approx 0.111$), oct6 source spacing, loose orientation constraint of 0.2, and depth weighting of 0.8. Sensitivity analyses used SNR values of 1.0 and 5.0. A source-space result was considered regularization-stable if the primary cluster retained significance and the same sign under both SNR $= 1.0$ and SNR $= 5.0$. Source-level gamma findings are reported as exploratory given the compound fragility of gamma source localization from scalp EEG during simultaneous fMRI.

Subject-specific source maps were morphed to fsaverage and normalized by Global Field Power before group aggregation to test spatial pattern consistency rather than absolute magnitude. Subjects whose raw covariance map GFP fell below the 5th percentile of the cohort distribution were excluded from GFP-normalized group aggregation to prevent amplification of noise-dominated maps to unit scale.

## Statistical Validation

### Permutation Strategy
Non-parametric permutation testing was performed on source-localized maps. Standard sign-flipping permutations were avoided because predictive algorithms deterministically map $X$ to $\hat{y} = f(X)$, yielding covariance patterns with consistent spatial structure driven by the intrinsic covariance of human EEG ($\Sigma_X$), regardless of whether the model tracked the true fMRI targets.

Instead, out-of-sample predictions ($\hat{y}_{test}$) were circularly shifted relative to EEG source-power trials within each run (minimum shift distance of 5 trials), approximately preserving temporal structure (Winkler et al., 2014). The minimum shift distance of 5 trials was selected to exceed the expected trial-wise autocorrelation length, which for thermal pain paradigms with inter-trial intervals of several seconds is typically 1–3 trials. Sensitivity analyses with shift distances of 10 and 20 trials were performed to verify null-distribution stability; the null was considered well-calibrated if the median and 95th percentile of the maximum cluster statistic changed by less than 10% across shift distances. For each of 5,000 permutations, subject-level covariance maps were recomputed and carried through the same group-level clustering procedure.

### Cluster-Based Inference
Group-level inference used cluster-based permutation testing (Maris & Oostenveld, 2007) on the primary ElasticNet model only. The confirmatory map was the alpha+beta preset across the full plateau ($3.0$–$10.5$ s), matching the gated Study 1 model. Each subject contributed one source-power covariance map after morphing to fsaverage, with positive values indicating greater source power on trials with larger predicted NPS expression. Vertex-wise one-sample $t$ statistics tested whether mean covariance differed from zero. Ridge was evaluated as a sensitivity analysis; Random Forest and BandTemporalRegressor maps were exploratory.

Vertex-wise statistics exceeding an uncorrected cluster-forming threshold (CFT) of $p < 0.01$ were aggregated into contiguous sign-preserving clusters. The primary CFT of $p < 0.01$ was selected as a balance between sensitivity and specificity; results were verified to be stable under CFTs of $p < 0.001$ and $p < 0.05$, where stability required the primary cluster to retain significance and the same sign. Family-wise error correction used a joint two-sided maximum statistic across positive and negative clusters in the confirmatory map. Clusters exceeding the 95th percentile of this null distribution were significant. Secondary summaries spanning alpha-only, beta-only, and plateau subwindows were Holm-corrected as one exploratory family.

Equal subject weighting was used for the primary analysis. Baseline-noise inverse-variance weighting was evaluated as a sensitivity analysis.

### Gamma-Band Interpretation
Gamma-band source maps were exploratory regardless of statistical significance, given the vulnerability of gamma EEG during simultaneous fMRI to non-neural artifacts. Gamma findings were not used to designate a candidate pain-relevant EEG pattern.

## Multimodal Spatial Comparison
Spatial correspondence between the EEG source-power covariance pattern and fMRI pain-related covariance was evaluated as a within-cohort spatial-consistency analysis, not as independent validation. The comparison was restricted to the cortical sub-network because the full NPS includes subcortical structures (thalamus, periaqueductal gray) inaccessible to EEG source modeling.

Comparing the EEG forward-pattern map to published NPS backward-model weights would create a theoretical asymmetry (Haufe et al., 2014). The primary comparison therefore used a within-study fMRI forward covariance pattern:

$$A_{fMRI}^{NPS-L2} \propto \mathrm{Cov}(B_{fMRI}^{resid},\; y_{NPS-L2})$$

where $B_{fMRI}^{resid}$ denotes trial-wise fMRI beta maps after voxelwise residualization against the same Level 2 nuisance family used in Study 1, and $y_{NPS-L2}$ is the observed Level 2 residualized NPS expression. Voxelwise nuisance coefficients were estimated without the held-out subject in cross-validated comparisons.

Because both the EEG source map and $A_{fMRI}^{NPS-L2}$ are covariance patterns with respect to the same (or a close approximation of) the NPS Level 2 target variable, their spatial correlation is partly mechanistically guaranteed by this shared dependency. The spatial specificity tests below evaluate whether the observed correlation exceeds what would be expected from spatially autocorrelated noise under a shared-target structure, but cannot fully separate shared-target inflation from genuine multimodal neurophysiological convergence. Independent-cohort replication with a held-out fMRI covariance target is required for a strong convergence claim.

Specificity was tested as a superiority claim. The EEG source map had to align more strongly with $A_{fMRI}^{NPS-L2}$ than with control maps derived from circular-shifted NPS scores, motion parameters, physiological noise regressors, stimulus variables, and pre-stimulus EEG predictions. Subjective-rating maps were evaluated separately as criterion-overlap maps. The primary statistic was $\Delta r = |r(\text{EEG}, A_{fMRI}^{NPS-L2})| - |r(\text{EEG}, A_{control})|$, tested under a spin-permutation spatial null. A pattern was not designated pain-relevant if any $\Delta r \leq 0$ or if the Holm-corrected test failed. In addition to $\Delta r$ significance, the absolute EEG-fMRI spatial correlation was required to exceed $|r| \geq 0.15$ to be considered neurophysiologically meaningful; correlations below this threshold were reported but not interpreted as evidence of spatial correspondence.

The fMRI target was smoothed to match the sLORETA point-spread function (approximately FWHM $\approx 12$ mm), estimated before outcome-map inspection from the forward/inverse operator alone. Spatial similarity was evaluated using Pearson correlation across matched cortical vertices. No sign was hard-coded as indicating valid overlap, given that alpha/beta power and BOLD can be negatively correlated depending on region and task.

Statistical significance used spherical spin permutations (Alexander-Bloch et al., 2018; 10,000 rotations). BrainSMASH variogram-matched surrogates were evaluated as a sensitivity analysis.

## Internal Cross-Validated Behavioral Validation
To determine whether the derived pattern captures pain-related variance beyond stimulus parameters, a LOSO behavioral validation was performed using a fixed scoring rule. This was internal cross-validated validation, not independent external validation. The primary validation used the alpha+beta full-plateau source-power covariance map matching the gated Study 1 model. Alpha-only and beta-only were tested as Holm-corrected secondary validations. No vertex set or cluster was selected from the source-space significance map.

For each target subject, the group-level pattern was estimated from all remaining subjects. Trial-wise expression was computed as:

$$\text{expression}_{s,i} = \sum_v \text{sourcepower}_{s,i}^{(\text{band})}(v) \cdot \text{Pattern}_{-s}^{(\text{band})}(v)$$

Within each held-out subject, the association between expression score and subjective pain rating was estimated via ordinary least squares regression controlling for the full Level 2 nuisance family. The primary test statistic was the group-level mean of within-subject standardized regression coefficients, tested against zero via a one-sample permutation test (5,000 permutations, sign-flipping within subjects). Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling.

Pain versus non-pain classification was reported as a secondary temperature-matched contrast when the nuisance-adjusted design was full rank. Temperature-matching paired pain and non-pain trials within the same temperature level; classification used the sign of the nuisance-adjusted expression difference, evaluated via balanced accuracy with permutation testing (5,000 permutations). The internal rating validation was required before the pattern could be provisionally designated a candidate pain-relevant EEG source-power covariance pattern. The label "EEG pain signature" was reserved for future independent-cohort replication.

## Subject-Level Reporting
Subject-wise prediction performance, behavioral-validation coefficients, artifact metrics, and map-expression distributions are reported with 95% BCa bootstrap confidence intervals (10,000 resamples). Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling or circular block bootstrap.

## Artifact and Robustness Controls
All primary analyses were repeated under prespecified robustness conditions. Artifact censoring thresholds were identical to Study 1: trials were censored when framewise displacement $> 0.5$ mm, DVARS robust $z > 3$, temporal/frontal high-frequency EMG power robust $z > 3$, cardiac phase-locking exceeded the 95th percentile of the within-subject null, or scanner-frequency residual peaks exceeded robust $z > 3$. Analyses were repeated after equalizing trial counts and excluding artifact-contaminated maps. A candidate pattern had to retain its sign and inferential status under these controls. Gamma-band maps remained exploratory regardless.
