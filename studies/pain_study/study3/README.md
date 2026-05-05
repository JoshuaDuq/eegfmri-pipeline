# Materials and Methods

## Research Objectives
This study (Study 2 of the manuscript, corresponding to Study 3 in the internal preprocessing pipeline) interpreted the predesignated source-interpretation model from Study 1 by deriving spatially resolved EEG source-power covariance patterns associated with the EEG residual component of its trial-wise predictions. The primary interpretive target was fixed a priori before any Study 1 performance outcomes were inspected: the EEG-predicted residual component from the primary incremental NPS ElasticNet alpha+beta nuisance-plus-EEG model, denoted $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$. This component excludes the nuisance-only prediction and is not the full combined raw-target prediction or the secondary residualized-target prediction model.

Because the alpha+beta preset contains two spectral bands, the confirmatory source family comprised two band-specific full-plateau maps, $A_\alpha$ and $A_\beta$, both covarying source power with the same $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values. Vertex-wise $t$-tests, clustering, Dice overlap, centroid displacement, fMRI spatial correlation, and behavioral expression were computed for alpha and beta independently, and all p-values were Holm-corrected across the two bands as a single family.

This hierarchy does not demote SIIPS1 as a Study 1 co-primary prediction target; it narrows the source-interpretation target to NPS because the source claim specifically concerns nociceptive fMRI signature expression, whereas SIIPS1 indexes a distinct stimulus-intensity-independent construct. SIIPS1, Level 1, Level 3, Ridge, Random Forest, and BandTemporalRegressor were evaluated only as sensitivity or exploratory analyses.

Study 2 proceeds confirmatorily only if all of the following hold for the predesignated NPS ElasticNet alpha+beta cell: (1) the primary Study 1 incremental analysis shows significant $\Delta R^2_{\text{LOSO}} > 0$; (2) the Level 2 residualized-target convergence analysis for the same cell shows positive out-of-sample prediction; (3) temporal negative controls fail to yield significant prediction; and (4) robustness criteria hold. If any gate condition fails, all Study 2 maps and behavioral associations are reported as exploratory. The Haufe transformation is applicable to the primary ElasticNet model as a linear decoder, although the resulting patterns remain regularization- and preprocessing-dependent. Random Forest and BandTemporalRegressor sensor-space maps are reported as prediction-covariance association maps without formal forward-model guarantees.

To formalize the confirmatory boundary, the following decision rules were applied:
- **Primary Analysis:** Band-specific source-power covariance with $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ from the primary incremental NPS ElasticNet alpha+beta model.
- **Study 1 Gate Passed:** Significant out-of-sample $\Delta R^2$ and positive Level 2 prediction; no pre-stimulus negative control predicting active-window targets; stable under robustness checks.
- **Study 2 Source-Map Success Requirement:** At least one primary band-specific source cluster survives family-wise error correction under the target-retrained null; no artifact contamination; quantitative robustness thresholds met.
- **Candidate Pain-Relevant Designation:** Requires Study 2 Source-Map Success, plus internal criterion-overlap validation with subjective ratings, directional consistency with the true $r_{\mathrm{NPS-L2}}$ source covariance map (positive spatial correlation $r \ge 0.30$ within the primary cluster), and no dominant association with prediction error. fMRI spatial correspondence is reported descriptively and is not a gate for this designation.
- **Downgrade to Exploratory:** Any failure in the Study 1 gate, artifact contamination of the primary map, or failure in quantitative cluster stability downgrades the interpretation to exploratory.

## Model Ingestion and Out-of-Sample Prediction
All interpretive analyses were performed strictly on out-of-sample data. Following the LOSO framework from Study 1, the selected model class, feature family, frequency preset, preprocessing recipe, outer folds, target transformation, and fold-specific hyperparameters were frozen before source analysis. For each subject, the held-out test-set EEG data ($X_{\mathrm{test}}$) was fed through the corresponding frozen primary incremental model to generate a single EEG residual prediction per trial ($\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$). The nuisance-only prediction and the full combined raw prediction were retained only for performance reporting. No Study 2 result was used to select Study 1 model hyperparameters, residualization level, feature family, frequency band, or time window.

## Prediction-Associated Sensor-Level Topographies
In multivariate decoding, predictive algorithms function as backward models mapping brain data to a target variable. Raw backward-model weights can highlight noise-canceling sensors rather than true signal sources (Haufe et al., 2014). Sensor-level linear feature patterns were interpreted using the Haufe transformation where applicable. Source-space analyses were based on out-of-sample prediction-associated source-power covariance maps and were therefore interpreted as covariance patterns associated with the EEG-predicted NPS residual component, not as direct cortical generators of the decoder weights.

For the primary linear model (ElasticNet), the Haufe transformation is:

$$A_{\mathrm{Haufe}} = \Sigma_{X,\mathrm{train}} \, W$$

where $\Sigma_{X,\mathrm{train}}$ is the training-fold data covariance matrix and $W$ is the fitted weight vector, both defined in the same feature space. Because Study 1 standardized features to zero mean and unit variance within each fold, $W$ was estimated in standardized units and $\Sigma_{X,\mathrm{train}}$ was accordingly the training-fold correlation matrix $R_{\mathrm{train}}$ (i.e., the covariance of the standardized features). The pattern was computed within each outer fold using training-fold statistics only.

For nonlinear models (Random Forest, BandTemporalRegressor), $\mathrm{Cov}(X, \hat{y})$ does not carry formal Haufe guarantees. These maps were designated prediction-covariance association maps without claims of causal feature attribution:

$$A_{\mathrm{assoc}} \propto \mathrm{Cov}(X_{\mathrm{features}},\; \hat{y}_{\mathrm{test}})$$

Only spectral power features were subjected to source-space follow-up; source-space power was recomputed from inverse-projected voltage time series rather than from projected sensor-level power maps. Projecting nonlinear complexity metrics through a lead-field matrix is physically invalid; sensor-level topographies for complexity features are reported separately.

## Artifact Sensitivity Analysis
Sensor-level maps were compared against artifact topographies derived from independent calibration data or from ICA components labeled as ocular, cardiac, scanner, or electromyographic artifacts. Template correlations were treated as flags, not correction procedures. Maps were not orthogonalized against artifact templates because orthogonalization can remove neural signal when neural and artifact topographies spatially overlap.

For each sensor-level map, trial-wise map expression was computed by projecting each trial's feature vector onto the map. Artifact association was then tested by regressing this expression score against framewise displacement, DVARS, cardiac phase, scanner-frequency residual power, or temporal/frontal high-frequency EMG power. A map was labeled artifact-contaminated when its absolute spatial correlation exceeded $r = 0.8$ with any artifact template or when its expression significantly covaried with any artifact metric after Holm correction. If a primary confirmatory map was identified as artifact-contaminated, the confirmatory source claim was considered failed and the interpretation was downgraded to exploratory. Contaminated maps were not post-hoc excluded to rescue the analysis.

## Spatio-Temporal Prediction-Covariance Mapping for the BandTemporalRegressor
The BandTemporalRegressor processes continuous epochs, enabling time-resolved interpretation. Because the covariance computation requires only the model's scalar output, it was extended across time. As a nonlinear model, the resulting maps are prediction-covariance association maps.

Because pain-evoked high-frequency oscillations are largely induced and non-phase-locked, the spatio-temporal map was computed on instantaneous power envelopes rather than raw voltage to avoid destructive phase cancellation:

$$A_{\mathrm{assoc}}(t) \propto \mathrm{Cov}(E_{\mathrm{test}}(t),\; \hat{y}_{\mathrm{test}})$$

This produced a map of shape bands $\times$ channels $\times$ time. Because the BandTemporalRegressor is exploratory (see Study 1), these temporal maps are interpreted as exploratory characterizations. Time-resolved analyses were evaluated within predefined windows (early, mid, and late plateau) to control temporal multiplicity.

## Anatomical Source Projection
Source-space analysis estimated cortical source-power covariance with the out-of-sample EEG-predicted NPS residual component. Inverse modeling was applied to band-passed voltage time series rather than to sensor-level power or covariance topographies, because power is a nonlinear, nonnegative summary of voltage that does not represent an electric field from cortical dipoles. The resulting maps are inverse-model-dependent source-power covariance estimates, not causal localizations of cortical generators, and should be explicitly distinguished from a source-localized Haufe weight map.

The sLORETA inverse operator (Pascual-Marqui, 2002) was used, with forward solutions derived from FreeSurfer subject-specific head models and boundary element method solutions. The primary noise covariance was derived from the pre-stimulus baseline ($-5.0$ to $-0.01$ s), matching Study 1. Because this interval may contain cue-locked expectancy activity, source localization was repeated using the immediate pre-stimulus window ($-0.2$ to $-0.01$ s) and the early pre-cue window ($-7.0$ to $-5.5$ s). A result was considered baseline-stable only when the primary spatial pattern, cluster polarity, and behavioral-validation inference were preserved under the immediate-baseline sensitivity.

A single inverse operator was estimated using the broadband noise covariance and applied identically to voltage time series filtered into each canonical band. Instantaneous power envelopes were computed from source time series via the Hilbert transform.

Prediction-associated source-power covariance was computed directly in source space after residualizing both source power and $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ against a fixed source-stage nuisance design. The primary source-stage design contained:

- run intercepts;
- continuous stimulus temperature in °C;
- linear trial number within run;
- HRF-weighted framewise displacement;
- HRF-weighted standardized DVARS.

This design was identical for every retained source subject; no covariate was added or dropped within subject, run, band, or permutation. This source-stage residualization means the primary source map interprets the portion of the Study 1 EEG residual prediction that remains after additional source-stage removal of stimulus, run, trial-order, and acquisition-noise structure.

A parallel unadjusted source-power covariance map using the original $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values, with only run intercepts removed, was reported as a sensitivity analysis and could reveal whether the primary adjustment removed most of the mapped signal. A subject failed source-stage inclusion if fewer than 25 clean plateau trials remained, fewer than two runs contributed trials, any primary source-stage nuisance covariate was unavailable, or the fixed source-stage design was not full rank after censoring. EMG and ECG metrics were retained strictly for artifact-sensitivity analyses and were never part of the primary source-stage nuisance design.

Because $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ is a deterministic function of EEG features, the primary map reflects which source patterns covary with model output rather than necessarily true residualized NPS expression. To explicitly distinguish prediction-associated covariance from true target covariance, parallel maps were computed using the actual held-out residual target and prediction error:

$$\mathrm{Cov}(\text{source power},\; r_{\mathrm{NPS-L2}})$$

$$\mathrm{Cov}(\text{source power},\; \hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}} - r_{\mathrm{NPS-L2}})$$

Both maps used the identical source-stage nuisance control. Consistency with the true target map is required for a candidate pain-relevant designation. The prediction-error covariance map was considered dominant if its absolute spatial correlation with the primary $\hat{r}_{\mathrm{EEG}}^{\mathrm{NPS}}$-covariance map exceeded the true-target covariance correlation, or if it produced a larger FWE-corrected cluster in the same broad anatomical region.

Confirmatory source inference required at least 20 source-valid subjects after the source-stage inclusion checks above; otherwise, all source-space analyses were reported as exploratory. Regularization was configured with SNR $= 3.0$ ($\lambda^2 \approx 0.111$), oct6 source spacing, loose orientation constraint of 0.2, and depth weighting of 0.8. Before outcome-map inspection, empirical point-spread and cross-talk functions were estimated from each subject's actual forward/inverse operator and morphed to fsaverage. The median cortical point-spread FWHM and its regional range were reported and used to choose the fMRI smoothing kernel for descriptive spatial comparison. Sensitivity analyses used SNR values of 1.0 and 5.0. A source-space result was considered regularization-stable if the primary cluster retained significance and the same sign under both SNR $= 1.0$ and SNR $= 5.0$. Source-level gamma findings are reported as exploratory given the compound fragility of gamma source localization from scalp EEG during simultaneous fMRI.

Subject-specific source maps were morphed to fsaverage and optionally normalized by Global Field Power before group aggregation. Because raw covariance-map GFP is itself an outcome-associated derived metric, GFP-based map exclusion was removed from the primary analysis to prevent circular inflation of group consistency. All subjects were included in the primary non-normalized group aggregation. GFP normalization was retained strictly as a sensitivity analysis.

## Statistical Validation

### Permutation Strategy
Non-parametric permutation testing was performed on source-localized maps. Standard sign-flipping permutations were avoided because they violate exchangeability: $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ is a deterministic function of $X$, so naïve sign-flipping of subject maps does not generate a proper null for feature-prediction association. Covariance patterns would have consistent spatial structure driven by the intrinsic covariance of human EEG ($\Sigma_X$) regardless of whether the model tracked the true fMRI targets.

The primary confirmatory null was target-retrained with frozen model-selection outputs. For each permutation and outer fold:

1. The Level 2 nuisance residual target was circularly shifted relative to EEG within run for
   both training and held-out subjects, using the same minimum shift-distance rule as Study 1.
2. The permuted raw target was reconstructed as the unshifted nuisance prediction plus the
   shifted residual.
3. The observed Study 1 feature set, preprocessing statistics, Yeo-Johnson transformation,
   and fold-specific ElasticNet hyperparameters were reused. Inner GroupKFold hyperparameter
   selection was not repeated.
4. Only the ElasticNet coefficients and intercept were refit on the permuted training target,
   then applied to the permuted held-out subject to generate permuted
   $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values.
5. The fixed source-stage nuisance control, source-map computation, group aggregation, and
   cluster testing were repeated.

This null tests whether the observed source-power covariance map exceeds the map expected from the same EEG covariance structure and frozen model-selection procedure when the fMRI target relationship is broken. Circularly shifting the final observed $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values relative to source power was retained only as a secondary model-anatomy sensitivity null, not as the confirmatory source-map test.

The executable permutation plan was fixed before source-map inspection. Band-specific source-power tensors, source-stage residualization matrices, cortical adjacency, and point-spread summaries were cached before permutation testing. Confirmatory inference used 1,000 target-retrained permutations per band as the minimum executable null; this gives a minimum attainable one-sided tail probability of approximately $0.001$ with the standard $(b+1)/(m+1)$ correction and is adequate for the two-band Holm family. If either band had an uncorrected cluster p-value below 0.10 or a maximum cluster statistic within 10% of the 95th-percentile null threshold, permutations for both bands were extended to 5,000 before final reporting. If the cached implementation could not complete at least 1,000 target-retrained permutations within the prespecified compute budget of 72 wall-clock hours on the available workstation or cluster allocation, source-space inference was downgraded to exploratory rather than reducing the null after outcome inspection.

### Cluster-Based Inference
Group-level inference used cluster-based permutation testing (Maris & Oostenveld, 2007) on the primary ElasticNet model only. The confirmatory family comprised the two band-specific full-plateau maps, $A_\alpha$ and $A_\beta$, both derived from the gated alpha+beta Study 1 model and the same $\hat{r}_{\mathrm{EEG},\mathrm{test}}^{\mathrm{NPS}}$ values. Each subject contributed one source-power covariance map per band after morphing to fsaverage, with positive values indicating greater source power on trials with larger EEG-predicted NPS residuals. Vertex-wise one-sample $t$ statistics tested whether mean covariance differed from zero. Ridge was evaluated as a sensitivity analysis; Random Forest and BandTemporalRegressor maps were exploratory.

Vertex-wise statistics exceeding an uncorrected cluster-forming threshold (CFT) of $p < 0.01$ were aggregated into contiguous sign-preserving clusters. The primary CFT of $p < 0.01$ was selected as a balance between sensitivity and specificity; results were verified to be stable under CFTs of $p < 0.001$ and $p < 0.05$, where stability required the primary cluster to retain significance and the same sign. Family-wise error correction used a two-sided maximum statistic across positive and negative clusters within each band. Band-level p-values for $A_\alpha$ and $A_\beta$ were Holm-corrected as the confirmatory source family. Secondary summaries spanning plateau subwindows and alternative feature families were Holm-corrected as one exploratory family.

Equal subject weighting was used for the primary analysis. Baseline-noise inverse-variance weighting was evaluated as a sensitivity analysis. Given the sample size, leave-one-subject-out influence diagnostics were applied to the source cluster: we evaluated the maximum change in cluster mass after removing one subject, checked whether the cluster remained directionally consistent, and ensured the result was not driven by only one or two high-amplitude outlier maps.

### Gamma-Band Interpretation
Gamma-band source maps were exploratory regardless of statistical significance, given the vulnerability of gamma EEG during simultaneous fMRI to non-neural artifacts. Gamma findings were not used to designate a candidate pain-relevant EEG pattern.

## Multimodal Spatial Comparison
Spatial correspondence between the EEG source-power covariance pattern and fMRI pain-related covariance was evaluated strictly as a descriptive within-cohort spatial-consistency analysis, not as independent validation. The comparison was restricted to the cortical sub-network because the full NPS includes subcortical structures (thalamus, periaqueductal gray) inaccessible to EEG source modeling.

Comparing the EEG forward-pattern map to published NPS backward-model weights would create a theoretical asymmetry (Haufe et al., 2014). The primary comparison therefore used a within-study fMRI forward covariance pattern:

$$A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}} \propto \mathrm{Cov}(B_{\mathrm{fMRI}}^{\mathrm{resid}},\; r_{\mathrm{NPS-L2}})$$

where $B_{\mathrm{fMRI}}^{\mathrm{resid}}$ denotes trial-wise fMRI beta maps after voxelwise residualization against the same Level 2 nuisance family used in Study 1, and $r_{\mathrm{NPS-L2}}$ denotes fold-contained residualized NPS expression after removal of the Level 2 nuisance family. Both the fMRI covariance target and voxelwise nuisance coefficients were estimated exclusively from training-fold subjects within the same LOSO framework to avoid held-out-subject leakage.

Because both the EEG source map and $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ are covariance patterns with respect to the same (or a close approximation of) the NPS Level 2 target variable, their spatial correlation is partly mechanistically guaranteed by this shared dependency. The spatial specificity tests below evaluate whether the observed correlation exceeds what would be expected from spatially autocorrelated noise under a shared-target structure, but cannot fully separate shared-target inflation from genuine multimodal neurophysiological convergence. Independent-cohort replication with a held-out fMRI covariance target is required for a strong convergence claim.

Specificity was tested as a descriptive superiority analysis. The fMRI spatial comparison was performed separately for $A_\alpha$ and $A_\beta$ with the single fMRI cortical vector. The EEG source map was compared with $A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}}$ and with control maps derived from circular-shifted NPS scores, motion parameters, physiological noise regressors, stimulus variables, and pre-stimulus EEG predictions. Subjective-rating maps were evaluated separately as criterion-overlap maps. The primary descriptive statistic was:

$$\Delta r = |r(\mathrm{EEG}, A_{\mathrm{fMRI}}^{\mathrm{NPS-L2}})| - |r(\mathrm{EEG}, A_{\mathrm{control}})|$$

This statistic was tested using variogram-matched BrainSMASH spatial surrogates. fMRI spatial consistency could support a spatial-convergence interpretation, but it could not upgrade, rescue, or block the candidate pain-relevant source-pattern designation. In addition to $\Delta r$ significance, the absolute EEG-fMRI spatial correlation had to exceed $|r| \geq 0.15$ before any spatial correspondence was discussed as neurophysiologically meaningful; correlations below this threshold were reported without a convergence interpretation.

Because the inverse-solution point spread is not spatially stationary and varies by cortical location and depth, the fMRI target was smoothed using the empirically estimated sLORETA point-spread FWHM derived from the actual source-resolution analysis before outcome-map inspection. Sensitivity analyses were conducted across the subject-level point-spread interquartile range and across one narrower and one broader kernel to ensure claims did not rely strongly on the exact smoothing value. Spatial similarity was evaluated using Pearson correlation across matched cortical vertices. No sign was hard-coded as indicating valid overlap, given that alpha/beta power and BOLD can be negatively correlated depending on region and task.

Because spin permutations can behave poorly on restricted, irregular cortical masks, variogram-matched spatial surrogates (BrainSMASH; 5,000 surrogates) were used as the primary spatial null. Surrogates were generated vertex-wise on the fsaverage cortical surface using geodesic distance, fitting independent variograms per hemisphere, excluding medial wall vertices. The identical cortical mask was applied to EEG and fMRI maps. Spherical spin permutations (Alexander-Bloch et al., 2018; 10,000 rotations) were reported alongside as a sensitivity analysis.

## Internal Cross-Validated Behavioral Validation
To characterize the criterion overlap between the derived pattern and subjective pain report, a LOSO internal cross-validated analysis was performed. This is described strictly as internal criterion-overlap validation, not independent validation. The primary validation used the same two band-specific full-plateau source-power covariance maps, $A_\alpha$ and $A_\beta$, that defined the confirmatory source family. Their behavioral p-values were Holm-corrected across the two bands. No vertex set or cluster was selected from the source-space significance map.

For each target subject, the group-level pattern was estimated from all remaining subjects. The behavioral validation expression was computed separately for alpha and beta maps:

$$\text{expression}_{s,i}^{(\text{band})} = \sum_v \text{sourcepower}_{s,i}^{(\text{band})}(v) \cdot \text{Pattern}_{-s}^{(\text{band})}(v)$$

The primary behavioral validation tests rating association beyond stimulus/acquisition/session structure; the unadjusted rating association is reported descriptively. Within each held-out subject, the association between expression score and subjective pain rating was estimated via ordinary least squares regression controlling for the fixed Study 1 Level 2 nuisance design. The primary test statistic was the group-level mean of within-subject standardized regression coefficients. Statistical testing used a permutation test (5,000 permutations) where, within each subject, expression was circularly shifted relative to rating within run, coefficients were recomputed, and the group mean was aggregated. Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling.

Because the pain/non-pain classification contrast may be deterministically coupled to the temperature manipulation, it was dropped unless pain class exhibited meaningful residual variance beyond temperature. Absence of adjusted rating association does not necessarily mean the EEG pattern is unrelated to pain, but it prevents strong pain-relevance claims. The internal rating criterion overlap was required before the pattern could be provisionally designated a candidate pain-relevant EEG source-power covariance pattern. The label "EEG pain signature" was reserved for future independent-cohort replication.

## Subject-Level Reporting
Subject-wise prediction performance, behavioral-validation coefficients, artifact metrics, and map-expression distributions are reported with 95% BCa bootstrap confidence intervals (10,000 resamples). Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling or circular block bootstrap.

## Artifact and Robustness Controls
All primary analyses were repeated under prespecified robustness conditions. Artifact censoring thresholds were identical to Study 1: trials were censored when framewise displacement $> 0.5$ mm, DVARS robust $z > 3$, temporal/frontal high-frequency EMG power robust $z > 3$, cardiac phase-locking exceeded the 95th percentile of the within-subject null, or scanner-frequency residual peaks exceeded robust $z > 3$. Exclusion of artifact-contaminated maps was reported only as a descriptive sensitivity analysis and could not restore confirmatory status if the primary map was contaminated. Analyses were also repeated after equalizing trial counts. A candidate pattern was considered robust only when prespecified quantitative thresholds were met across sensitivity controls: preserving statistical significance, retaining the same cluster sign, maintaining a spatial correlation of unthresholded maps $\ge 0.50$, maintaining a minimum Dice overlap of 0.40 for thresholded clusters, and exhibiting a maximum centroid displacement of no more than 15 mm within the same broad anatomical region. Gamma-band maps remained exploratory regardless.
