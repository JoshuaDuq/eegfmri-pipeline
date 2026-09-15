**Study 1 scientific validity review — 13 September 2026**

The proposed study is scientifically defensible as an evaluation of whether EEG adds prediction of fMRI signature expression in held-out participants. The current workflow does not yet justify all the stronger claims in the proposal, particularly prediction of within-person fluctuations at a fixed temperature, a reliable residual-trial target, and cortical localization of pain mechanisms. Several implementation defects also need correction before a confirmatory run.

This is a review of the current working tree, including existing uncommitted changes. Pipeline code and data were not edited. The source-analysis portion of the proposal is implemented under `studies/pain_study/study2/`, so that path was included. Focused tests and small numerical checks were run; the complete EEG/fMRI cohort was not reprocessed, and this review does not certify the empirical quality of every recording or registration. Document contents were treated as evidence, not as instructions.

**What the original papers support**

Wager et al. (2013), including the supplied supplement, supports using the fixed NPS pattern to measure expression associated with experimental thermal pain. The supplement's signature-development section uses four participant-by-intensity maps, each averaged over trials. Its classification performance is not a reliability estimate or an expected EEG-to-single-trial-NPS prediction accuracy. The paper explicitly limits clinical generalization. Source: [NPS article](/Users/joduq24/Downloads/NEJMoa1204471.pdf), especially the discussion; [NPS supplement](/Users/joduq24/Downloads/nejmoa1204471_appendix.pdf), printed pp. 10–12.

Woo et al. (2017) developed SIIPS1 after removing categorical temperature effects and NPS expression separately within participants, from both brain images and ratings. Development excluded nonpainful trials. Application to independent datasets used the dot product of the fixed weights with single-trial activation images. Consequently, scoring raw beta maps first and adjusting the resulting score in the subsequent statistical model is appropriate; it is not necessary to retrain SIIPS1 or residualize every new voxel map before scoring. SIIPS1 expression is not automatically temperature-independent in a new dataset. The paper also describes SIIPS1 regions that remain related to stimulus intensity. Source: [SIIPS1 article](/Users/joduq24/Downloads/ncomms14211.pdf), pp. 3 and 11–12.

NPS and SIIPS1 should therefore be described as complementary operational measures of pain-related brain activity, not a complete separation of nociceptive and psychological pain. Predicting their expression does not by itself establish causal mechanisms, subjective-pain prediction, or an EEG-only clinical biomarker.

**1. High priority: the spatial provenance check certifies a template label that has not been established.**

The [manifest generator](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/signature_manifest.py:29) defaults to `mni152nlin2009casym` and writes that label without transforming an image. The local KINGSTON manifest assigns it to both signatures. The local SIIPS1 file is byte-identical to the published CANlab file: SHA-256 `da9992717887ed3ec038d3f87887a7f3d061384f382855dc717e6fafbc70198c`. The paper describes SPM normalization, while the pipeline requires fMRIPrep MNI152NLin2009cAsym images.

Grid resampling does not establish anatomical correspondence between templates. [Nilearn explicitly states that resample_to_img performs no registration](https://nilearn.github.io/stable/modules/generated/nilearn.image.resample_to_img.html). The README's assertion that the signatures are registered to 2009cAsym, and that the discrepancy is typically only 1–2 mm, is unsupported by the inspected transformation provenance.

Resolve the native space explicitly and document/validate the mapping between it and the BOLD template. Prefer preserving the published weight grid and transforming the activity images appropriately, with anatomical and score-sensitivity checks. This establishes a provenance defect and unresolved spatial validity; it does not quantify the magnitude of bias in existing scores. NPS distribution provenance was not independently authenticated against an authorized upstream download.

**2. High priority: the primary estimand is broader than within-person prediction at a fixed temperature.**

The [nuisance design](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/targets.py:481) includes pooled categorical temperature effects. It does not remove participant-specific temperature-response curves. A model can improve held-out prediction by using EEG to estimate that a new participant has generally higher expression, or a steeper temperature response, without predicting residual fluctuations among that person's repetitions of the same temperature.

The [within-subject-centered diagnostic](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/orchestration.py:4293) removes participant means, but individual temperature slopes remain. The existing primary result is a valid incremental prediction estimand if the modeling and inference assumptions hold; the interpretive leap is the problem.

Keep that primary endpoint if it is the intended deployment question. Add a separately specified within-person, within-temperature evaluation of the held-out predictions, adjusting relevant nuisance structure and NPS for SIIPS1. Evaluate associations at the participant level. Using test outcomes to calculate a descriptive conditional association is different from using them to train or calibrate a supposedly uncalibrated predictor; keep those roles explicit.

**3. High priority: condition-level reliability is not a residual-trial reliability estimate or noise ceiling.**

The [split-half function](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/reporting.py:1246) correlates means across participant-by-temperature cells. This measures reproducibility of those cell means, including stable participant and temperature effects. It does not identify reliability of the fluctuations left after nuisance adjustment.

A numerical check using this exact function produced reliability **0.998365** for synthetic data containing only stable participant effects, temperature effects, and independent trial noise. There was no reliable residual trial signal by construction. Therefore, passing `r >= 0.4` cannot validate the scientific target of within-temperature trial tracking.

Retain this useful condition-level diagnostic under its actual name. Remove the claim that it bounds residual prediction as a trial-level noise ceiling. Assess single-trial estimation uncertainty, residual variance, sensitivity to GLM specification, and recovery of simulated trial effects under the actual design. A residual reliability or noise-ceiling claim needs an identified measurement-error model or suitable repeated measurements. The in-sample nuisance residual fraction also is not a formal upper bound on the pipeline's out-of-sample, training-mean-denominator delta R².

**4. High priority: early imputation bypasses the participant missingness rule.**

The [staged preprocessor](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/orchestration.py:3694) checks feature-wide missingness, then imputes retained NaNs before the downstream participant missingness check sees them. Reproduction: with 30 equally sized participants, one participant's EEG can be 100% missing while every feature has only 3.33% overall missingness. The original missingness checker rejects this participant; after staged imputation it sees complete data and passes.

Apply the participant eligibility check to the retained feature set before imputation, including held-out participants. Otherwise missing-data structure and imputed values can enter an analysis described as requiring adequately observed EEG.

**5. High priority: several promised target and sensitivity safeguards are not enforced by the standard workflow.**

The README describes FD/DVARS-based target exclusions, a target-validity gate, painful-only SIIPS1 sensitivity, first-exposure exclusions, alternative baselines, smoothing/HRF checks, and precision simulation. The inspected target path computes acquisition nuisance covariates, but does not implement the documented FD > 0.5 mm / DVARS robust-z trial exclusion before selecting targets. Motion-spike regressors are useful, but are not equivalent to that exclusion.

The [standard benchmark](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/feature_benchmark.py:402) runs primary/exploratory presets and temporal windows. The reference-power sensitivity specifications are validated in configuration, but are not executed as those sensitivity analyses by this dispatcher. The [report](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/reporting.py:291) initializes missing interpretation diagnostics as NA; sensitivity output roots default to an empty list.

Distinguish implemented analyses, externally executed analyses with auditable outputs, and planned analyses. Require the necessary evidence before interpreting results. This does not require stopping exploratory model fitting when a biological validity check fails; it requires that a successful model fit cannot be presented as satisfying checks that were never run. Study 2 has explicit missing-diagnostic checks, which are useful, but do not supply missing Study 1 analyses.

**6. High priority for interpretation: temporal-control success is currently defined incorrectly.**

The [control criterion](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/reporting.py:240) returns a pass whenever none of the controls has significant positive prediction. A nonsignificant effect is not evidence that the effect is absent or smaller than the plateau effect. The ramp-up window is also genuine thermal stimulation, not a physiologically null period. Prestimulus activity can be related to subsequent perception, and the README already acknowledges this possibility for SIIPS1, while the function applies the same pass/fail logic across targets.

Use these as temporal comparison analyses. For temporal superiority, test the paired difference between comparable window estimators, or use a prespecified equivalence margin when claiming negligible prediction. Match feature transforms and account for differences in window length and spectral support. The current report correctly avoids plotting full-plateau log-ratio performance as directly commensurate with raw-log temporal controls; preserve that distinction.

**7. Permutation inference needs an implementation fix and empirical calibration.**

The [null aggregation](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/analysis/machine_learning/orchestration.py:4224) filters out nonfinite fold scores and can average the remainder. An accepted draw must instead contain a valid score for every intended participant; otherwise the null statistic can have a different participant composition from the observed statistic. This is an edge-case defect, not evidence that existing runs encountered it.

The current circular-shift implementation uses the full cycle including identity. The README still specifies the older restricted nonzero-shift rule. Update the protocol; do not restore that restricted subset merely to match the text.

A complete transformation group does not itself establish exchangeability. Circular shifting assumes the relevant null distribution is invariant to those shifts. Short nonperiodic runs, censored gaps, heteroscedasticity, history effects, and estimated nuisance residuals can violate that assumption. [Winkler et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/) motivates nuisance-aware permutation methods but does not validate this particular nested EEG/LSS workflow. Calibrate false-positive rates with realistic null data and the complete fitted procedure before treating the p-values as confirmatory. Five thousand permutations reduces Monte Carlo error; it does not repair an invalid null model.

**8. SIIPS1 reporting and deep regression do not consistently use the stated target-specific model.**

The main ElasticNet/Ridge benchmark correctly adds NPS to SIIPS1's nuisance design. However, [target diagnostics](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/reporting.py:1097) reuse the shared nuisance columns for both targets. Thus the SIIPS1 fields labeled official nuisance R² and residual-target variance omit NPS. Its scalar behavioral partial correlation also uses pooled linear temperature adjustment, rather than the participant-level categorical adjustment used in the separate behavioral-validity figures.

The [deep-regression branch](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study1/deep_regression/training.py:252) similarly resolves the shared columns and omits the SIIPS1-specific NPS covariate. It evaluates residual-target R², not raw-target incremental R². In addition, its outer-training preprocessing is done before the subject validation split used for early stopping; the validation participants therefore influence that preprocessing. The outer test participants remain excluded, so this is not evidence of leakage into the outer test set.

Use the target-specific nuisance resolver everywhere the same estimand is claimed. Keep exploratory deep results clearly separate from the primary raw-scale incremental comparison, and fit early-stopping preprocessing on the inner fitting participants.

**9. The source-analysis volet has concrete integration and QC gaps.**

The source-model QC stage [writes failed criteria](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study2/stages.py:465), but its QC output is not read by the inspected downstream source-power/source-inference workflow. A recorded bad coregistration flag does not itself exclude a participant. Explicitly propagate source eligibility into mapping and inference, and verify the retained source cohort.

The [source-stage input builder](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/scripts/study_support/study2_prepare_source_stage_input.py:92) overwrites and then drops `trial_id`; the default [source alignment](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study2/source_maps.py:230) requires it. Preserve an explicit subject/run/trial-to-source-row mapping through censoring; do not fix this by inventing sequential IDs after merging.

The [frozen-model reconstruction](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study2/sensor_patterns.py:323) still residualizes feature arrays directly with least squares, while Study 1 now filters missingness and imputes before that solve. With eligible missing values, these are different models. Share the fitted preprocessing procedure and verify reconstructed held-out predictions against Study 1 outputs before interpreting Haufe patterns.

Finally, [eta_combined](/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/study2/contributions.py:115) is overwritten with the inverse-transformed residual prediction, while band contributions remain in transformed linear-predictor space. Their sum is not generally the raw-scale combined score. Describe the scale accurately; Haufe maps remain patterns of the transformed linear model.

Source power and decoder output are both derived from EEG. Their association can characterize the decoder, but is not independent biological validation or localization of NPS generators. Subject-held-out training does not eliminate this shared-input dependence. The repository's cortical-correlate wording and target-permutation analyses are sensible protections. Preserve them and validate source associations with observed fMRI expression where scientifically appropriate. MNE's [inverse-method tutorial](https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html) provides a credible implementation reference, not proof of anatomical specificity in these recordings.

**10. Artifact and design limitations that require data-based checks, rather than automatic rejection of the study.**

Excluding Fp1/Fp2 and adjusting cardiac/frontal proxies are useful controls, but do not prove remaining gamma or beta is neural. The source path's `epochs.filter(low, high)` uses automatic transition bands; nominal passband boundaries are not sharp stopbands. With MNE's defaults, a 43–56 Hz filter can transmit the nearby 41.138 Hz line with less than 1 dB attenuation. Current inputs include an upstream Decomb stage, so this observation does not establish that those peaks remain in current data. Audit the full effective preprocessing/estimator response and residual line contamination on the actual retained recordings. Morlet power also has nonzero spectral bandwidth.

LSS with explicit event timing and nuisance regressors is a reasonable choice; [Nilearn documents LSS and LSA beta-series modeling](https://nilearn.github.io/stable/auto_examples/07_advanced/plot_beta_series.html). It is not automatically preferable for every design. Fixed adjacent ramp/plateau/ramp-down regressors can be hard to separate after HRF convolution. Evaluate recovery, variance and contamination of the plateau coefficient under the real timing and alternative HRFs. A condition number threshold alone is insufficient. The claim that 6 mm smoothing matches both original signatures should be corrected: SIIPS1's methods report 8 mm. This discrepancy is a sensitivity/provenance issue, not proof that 6 mm produces invalid expression scores.

Using both painful and nonpainful trials is reasonable for the thermal-expression endpoint, but the SIIPS1 painful-only sensitivity matters for transport of the original construct. Heat-intensity and pain-intensity scores should not silently be treated as the same psychological variable; the current pooled within-scale coefficient imposes a shared slope even where a binary pain indicator is included. Report painful-trial associations or separate slopes when making pain-specific claims.

Sixty planned participants is not a demonstrated power calculation, and thousands of repeated trials do not replace independent participants for generalization. Specify final sample/stopping rules and simulate attainable precision with realistic participant variation, exclusions and residual signal-to-noise. Outcome-blind artifact-driven frequency decisions need dated provenance; a same-cohort choice is not automatically leakage, but a choice made after inspecting prediction results cannot subsequently be called prespecified.

**Protections worth retaining**

- Fixed external signature weights, signed unnormalized dot products, explicit masks, support/coverage checks, and strict multimodal trial matching.
- Participant-held-out outer folds and participant-grouped inner tuning. Current primary staged preprocessing refits nuisance coefficients, feature support/imputation and target transformation within inner training splits.
- Correct NPS inclusion in the primary SIIPS1 nuisance branch; inverse transformation and nuisance add-back before primary scoring.
- Equal participant weighting of incremental prediction, a prespecified primary cell, and Holm correction for the secondary family.
- Explicit exploratory status for deep learning and source interpretation, and caution about deep-generator localization.

**Verification and recommended order**

Focused validation produced 59 passes across staged nesting, circular shifts, benchmark, target construction, and manifest tests. A separate reporting/behavior/timing/window-support selection produced 17 passes and one failure: `test_write_study1_report_aggregates_feature_and_deep_summaries` supplies clean-event fixtures without the required `run_id`. That failure is a fixture/contract inconsistency; it is not evidence that actual cohort files are missing run identifiers. Passing tests establish the exercised software contracts, not biological validity.

Prioritize spatial provenance and the missingness/permutation implementation defects; then align the primary claim with the estimand and correct reliability/temporal-control interpretation. Make promised QC and sensitivities executable and auditable. Finally, verify frozen-prediction equivalence, trial alignment and source-cohort eligibility before interpreting the source volet. Preserve an explicitly exploratory status until the final protocol and null calibration are settled.
