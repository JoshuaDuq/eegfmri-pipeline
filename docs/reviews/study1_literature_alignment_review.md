# Study 1 Literature Alignment Review

Review date: 2026-05-14

Implementation updates: the alignment passes following this review made the core hard gates
executable: frozen signature manifest validation, signature support/mass-retention thresholds,
scoring-mask extent hashes, HRF-weighted Level 2 artifact covariates, LSS condition-number/design-
efficiency thresholds, explicit Study 1 task-block identity, and raw/unbaselined temporal negative
control configuration.

Scope: Study 1 in `studies/pain_study/study1/`, with related EEG/fMRI utilities in
`eeg_pipeline/` and `fmri_pipeline/`. This review evaluates whether the design and current
implementation are aligned with the pertinent pain-neuroimaging, simultaneous EEG-fMRI,
single-trial fMRI, EEG preprocessing, and predictive-modeling literature.

Local files reviewed:

- `studies/pain_study/study1/README.md`
- `studies/pain_study/study1/RUN_GUIDE.md`
- `studies/pain_study/study1/config/study1_config.yaml`
- `studies/pain_study/study1/feature_benchmark.py`
- `studies/pain_study/study1/feature_spec.py`
- `studies/pain_study/study1/prepare_features.py`
- `studies/pain_study/study1/targets.py`
- `studies/pain_study/study1/reporting.py`
- `studies/pain_study/study1/deep_regression/`
- `eeg_pipeline/analysis/machine_learning/`
- `fmri_pipeline/analysis/trial_signatures.py`
- `fmri_pipeline/analysis/multivariate_signatures.py`
- `fmri_pipeline/utils/bold_discovery.py`

## Executive Assessment

Study 1 is broadly aligned with the relevant literature as an EEG-to-fMRI validation bridge,
not as a standalone EEG pain biomarker validation study. The strongest choices are the use of
externally defined NPS/SIIPS1 targets, single-trial LSS fMRI modeling, subject-held-out prediction,
fold-contained preprocessing, regularized linear primary models, restricted permutation inference,
and conservative claim language.

The main scientific risk identified at review time was not the overall concept. It was the gap between
the written protocol's hard validity gates and what the executable pipeline enforced. The follow-up
alignment pass closed the highest-risk target-construction and LSS-design gaps.

Highest-priority issues addressed in the alignment passes:

1. Signature provenance and support checks now enforce the frozen manifest, checksums,
   support-retention thresholds, positive/negative weight-mass stability, and identical scoring-mask
   extent.
2. The Level 2 nuisance model now names explicit HRF-weighted FD/DVARS/Fp1-Fp2 covariates.
3. LSS estimability now enforces rank, residual degrees of freedom, condition number > 100, and
   target design efficiency < 0.1.
4. Interpretation diagnostics are consumed by `reporting.py`, but missing diagnostics do not make
   `analysis_validity_status` invalid. They set missing-diagnostic/source-entry statuses. That is
   acceptable only if the paper/report language never presents such outputs as fully source-localizable
   confirmatory evidence.
5. The protocol explicitly distinguishes task blocks from BIDS acquisition runs, but block discovery
   can fall back to run/session labels. That is weaker than the README when acquisition runs contain
   multiple task blocks.

## Alignment Rubric

- Aligned: decision is well supported by literature and visibly represented in code/config.
- Mostly aligned: scientifically defensible, with caveats or limited implementation evidence.
- Partly aligned: the protocol is defensible, but implementation or interpretation needs tightening.
- Not confirmed: literature alignment is plausible, but implementation evidence was not found.
- Risk: likely mismatch between stated decision and current executable behavior.

## Decision Matrix

### 1. Scientific Framing: EEG Features Predict fMRI Pain-Signature Expression

Status: Aligned, with appropriate claim limits.

The Study 1 framing is scientifically defensible. It avoids claiming that scalp EEG alone is a
validated pain biomarker and instead asks whether EEG plateau features predict held-out trial-wise
NPS/SIIPS1 expression beyond nuisance structure. That is a sensible bridge because NPS and SIIPS1
are externally developed fMRI multivariate signatures, while scalp EEG spectral features are spatially
ambiguous and artifact-sensitive.

The README's claim boundary is aligned with neuroimaging biomarker guidance: a positive Study 1
result supports prediction of fMRI signature expression in held-out subjects, not independent
clinical pain diagnosis or source localization. This is important because pain biomarkers remain
context-bound and require external validation across laboratories, populations, and task conditions.

Implementation alignment:

- The feature benchmark predicts `fmri_signature` targets and keeps the primary claim tied to the
  `NPS / alpha_beta / ElasticNet` cell.
- Reporting separates primary prediction from Study 2 source-entry interpretation.
- The report code creates source-entry statuses that can block confirmatory source interpretation.

Literature basis:

- Wager et al. introduced the NPS for heat-induced physical pain and individual-level fMRI prediction.
- Woo et al. introduced SIIPS1 to isolate cerebral pain contributions beyond nociceptive intensity.
- Pain biomarker reviews emphasize externally validated signatures, cautious claims, and interpretation
  limits.

### 2. Primary Target: NPS Dot-Product Expression

Status: Mostly aligned.

NPS is the best primary target for this protocol because the task is thermal pain and the NPS was
developed for evoked physical pain. Using dot-product expression is also aligned with CANlab-style
pattern expression: the model should apply fixed weights to new subject/trial maps without refitting,
z-scoring, thresholding, sign-flipping, or re-estimating the signature on Study 1 data.

The caveat is that NPS is sensitive to nociceptive stimulus intensity. Study 1 uses fixed group-level
temperatures rather than participant-specific calibration, so raw NPS prediction may partly reflect
temperature exposure and thermal sensitivity. The README handles this correctly by making Level 2
nuisance-controlled convergence necessary for stronger interpretation.

Implementation alignment:

- `targets.py` requires primary signature names exactly `NPS` and `SIIPS1`.
- `feature_benchmark.py` runs NPS and SIIPS1 across the confirmatory frequency/model grid.
- `multivariate_signatures.py` computes dot, cosine, and Pearson correlation, with dot used by config.

Implementation update:

- Signature support/provenance enforcement is now aligned with the README through manifest,
  checksum, support-retention, mass-stability, and scoring-mask extent checks.

### 3. Secondary Target: SIIPS1

Status: Aligned.

SIIPS1 is a strong secondary target because it was designed to predict pain above and beyond
nociceptive input. In this Study 1 design, SIIPS1 helps distinguish EEG tracking of fMRI signature
variance from simple temperature or nociceptive drive. Keeping it secondary rather than primary is
appropriate because the thermal stimulation design and primary bridge are closer to NPS.

Implementation alignment:

- `targets.py` enforces SIIPS1 as one of exactly two required target names.
- `feature_benchmark.py` treats SIIPS1 as part of the required confirmatory grid.

### 4. Fixed Temperature Protocol, No Individual Calibration

Status: Mostly aligned for the stated goal; risky for broader pain claims.

Using fixed temperatures preserves between-participant thermal-sensitivity variance, which is useful
when modeling fMRI signature expression and external validity across participants. The design is not
ideal if the primary target were subjective pain intensity matched across participants. The README is
clear that the target is NPS/SIIPS1 expression, not equal subjective pain.

The first trial is fixed at 49.3 C. This is experimentally understandable but creates a position,
novelty, expectancy, and high-temperature coupling. The README mitigates this through first-trial and
first-block sensitivities and by using block-aware nuisance/permutation logic.

Implementation alignment:

- The config includes temperature, block, onset, and trial-order nuisance fields.
- Reporting expects first-exposure robustness diagnostics.

Risk:

- If first-exposure diagnostics are missing, reporting flags them as missing but does not make the
  primary analysis invalid.

### 5. Plateau-Only EEG and fMRI Target Scoping

Status: Aligned.

Restricting both EEG and fMRI targets to the 3.0-10.5 s plateau is scientifically coherent. It avoids
mixing ramp-up, plateau, and return-to-baseline phases that may carry different sensory, motor, and
anticipatory processes. The stated EEG/fMRI timing mismatch tolerance is also appropriate for
single-trial multimodal modeling.

Implementation alignment:

- Study 1 target config scopes fMRI trial signatures to stimulation/plateau rows.
- Feature preparation asserts Study 1 metadata such as no IAF and no evoked subtraction.
- `trial_signatures.py` supports condition-scope columns and phase scoping.

### 6. Trial-Wise fMRI Modeling: LSS

Status: Mostly aligned.

Least-squares separate (LSS) is an appropriate single-trial fMRI estimator when adjacent events are
correlated and trial-wise estimates are needed. Pooling other trials into nuisance regressors, keeping
non-plateau events as nuisance events, using a canonical HRF, and using fMRIPrep-space BOLD inputs are
consistent with standard task-fMRI practice.

Implementation alignment:

- `trial_signatures.py` fits one model per trial for LSS and extracts the target contrast.
- Non-selected events can be modeled as nuisance events.
- fMRIPrep MNI-space inputs are required for signature scoring.
- Design matrices are checked for rank deficiency, residual degrees of freedom, condition number, and
  target design efficiency.

### 7. No Smoothing in Primary Signature Extraction

Status: Aligned.

The no-smoothing primary decision is defensible for fixed multivariate signatures. Spatial smoothing can
alter pattern expression, weight support, and positive/negative mass distribution. A 4 mm smoothing
sensitivity is an appropriate robustness analysis rather than a primary setting.

Implementation alignment:

- `study1_config.yaml` sets smoothing to none for targets.
- `trial_signatures.py` passes smoothing through the first-level model configuration.

Implementation update:

- Post-resampling support and mass-stability thresholds are now enforced during signature scoring.

### 8. Signature Resampling and Mask Handling

Status: Partly aligned.

The scoring helper makes several good scientific choices: it rejects continuous resampling of images
with non-finite voxels, requires finite image values inside the fixed signature mask, and fails fast
when grids cannot be aligned. These choices prevent silent trial-specific support changes.

The helper now computes retained nonzero support, retained positive/negative support,
positive/negative absolute weight-mass stability, and a scoring-mask extent hash in addition to
dot/cosine/Pearson outputs.

Implementation alignment:

- Good: finite-mask handling, no trial-specific shrinking, MNI-space requirement, path/name checks.
- Missing: manifest checksum validation, positive/negative support retention, mass-stability thresholds.

### 9. fMRI Confounds: Motion, WM/CSF, CompCor, Outliers

Status: Mostly aligned.

Using fMRIPrep derivatives, motion regressors, WM/CSF signals, CompCor-like components, FD, DVARS,
and outlier censoring is aligned with fMRI denoising literature. The FD > 0.5 mm and robust DVARS
thresholds are reasonable for identifying high-motion/high-artifact trials, especially in single-trial
estimation.

Implementation alignment:

- `trial_signatures.py` requires confounds unless explicitly disabled.
- `bold_discovery.py` validates generated design matrices.
- `study1_config.yaml` defines fMRI target confounds and Level 2 nuisance columns.

Implementation update:

- The config and README now use explicit HRF-weighted FD/DVARS/Fp1-Fp2 nuisance covariate names.

### 10. EEG Simultaneous-fMRI Artifact Correction

Status: Aligned, with known limitations.

Average artifact subtraction for gradient artifacts and BCG template subtraction are established
approaches for EEG recorded in MRI. The BrainVision Analyzer stage is compatible with common
simultaneous EEG-fMRI workflows. The added ECG channel is important because BCG correction and
physiological nuisance estimation depend on cardiac timing.

Implementation evidence is mostly protocol-level rather than in the Study 1 Python code, which is
expected because BrainVision Analyzer processing is upstream. Study 1 correctly treats absence of
artifact covariates as invalid rather than imputing them.

### 11. EEG Preprocessing: MNE-BIDS, 0.1-100 Hz, Notch, PyPREP, ICA/ICLabel, Autoreject

Status: Mostly aligned.

The planned preprocessing sequence is standard and defensible:

- MNE-BIDS gives BIDS-compatible organization.
- 0.1-100 Hz and 60 Hz notch are conventional for broad spectral work.
- PREP/PyPREP-style bad-channel detection and robust referencing are aligned with large-scale EEG
  preprocessing recommendations.
- Extended infomax ICA plus ICLabel is aligned with the ICLabel training assumptions when common
  average reference and appropriate filtering are used.
- Autoreject local cross-validation is an established approach for local channel/trial repair.

Caveat: applying ICLabel thresholds automatically in simultaneous EEG-fMRI data needs careful audit
because residual MR/BCG artifacts may not match ordinary EEG artifact distributions. The README's
extra non-fMRI ICLabel artifact audit is therefore appropriate.

Implementation alignment:

- Study 1 feature preparation enforces feature metadata constraints before benchmarking.
- Fp1/Fp2 are excluded from confirmatory features.

Not confirmed:

- The upstream preprocessing pipeline's actual BrainVision/PyPREP/ICA/autoreject parameters were not
  fully audited here because they sit outside the Study 1 package.

### 12. Primary EEG Features: Alpha + Beta Individual-Channel Log-Ratio Power

Status: Aligned.

Alpha and beta power are defensible primary EEG features for pain-related prediction because the pain
EEG literature frequently reports low-frequency/sensorimotor alpha-beta modulation and because these
bands are less artifact-prone than gamma in scalp EEG, especially inside the scanner.

Individual-channel features preserve spatial information without overclaiming source localization.
Using log-ratio baseline-corrected power is conventional for time-frequency EEG and supports stable
modeling. Avoiding evoked subtraction is defensible because the prediction target is single-trial
signature expression and evoked/induced separation could remove target-relevant variance.

Implementation alignment:

- `feature_benchmark.py` defines primary presets `alpha`, `beta`, and `alpha_beta`.
- The primary feature scope is active-window, individual-channel, log-ratio power.
- `prepare_features.py` enforces no IAF and no evoked subtraction for Study 1 features.

### 13. Gamma as Exploratory

Status: Strongly aligned.

Gamma should not be confirmatory in this simultaneous EEG-fMRI protocol. High-frequency EEG overlaps
with facial, jaw, ocular, temporal, cardiac, and scanner residual artifacts. The README correctly treats
gamma as exploratory and artifact-sensitive.

Implementation alignment:

- Gamma is listed as exploratory.
- Fp1/Fp2 high-frequency artifact proxy and artifact robustness are explicit in the protocol.

Risk:

- Artifact-censoring robustness is consumed in reports but not clearly generated by the core feature
  benchmark.

### 14. Fp1/Fp2 High-Frequency Artifact Proxy

Status: Mostly aligned.

Without independent facial EMG, a frontal high-frequency proxy is a reasonable artifact-control
strategy, not a clean EMG measurement. Excluding Fp1/Fp2 from confirmatory features while using them
as nuisance/artifact metrics is appropriate because it avoids directly feeding the artifact proxy into
the predictor.

The README is scientifically careful: it calls the proxy artifact control, not independent EMG. It also
requires no imputation when artifact metrics are missing.

Implementation alignment:

- `feature_benchmark.py` excludes Fp1/Fp2 from primary feature matrices.
- `study1_config.yaml` defines Fp1/Fp2 artifact-proxy settings and Level 2 nuisance columns.

Risk:

- The HRF-weighted version described in the README is not explicit in the config column names.
- The categorical artifact-censoring sensitivity appears to be a report diagnostic rather than a
  generated primary benchmark product.

### 15. Residualization Levels and Staged Residual Learning

Status: Aligned.

The staged residual-learning design is one of the strongest parts of Study 1. It preserves raw target
prediction while quantifying whether EEG improves over nuisance-only prediction. Fitting nuisance
models only on training folds avoids leakage. Rank checks for nuisance matrices are appropriate.

Level 1 raw expression, Level 2 stimulus/artifact/acquisition nuisance control, and Level 3 adding
subjective report are conceptually clear. Level 2 is the most important interpretation level because
fixed-temperature thermal tasks can otherwise confound target expression with stimulus intensity.

Implementation alignment:

- `target_residualization.py` fits fold-contained nuisance models and checks SVD rank.
- `orchestration.py` implements staged residual learning with nuisance prediction, residual target
  transformation, EEG residual modeling, inverse transform, and nuisance add-back.

Risk:

- The canonical Level 2 nuisance columns must correspond to the intended HRF-weighted and artifact
  covariates, not merely raw trial metadata.

### 16. LOSO Outer CV and Subject-Grouped Inner Validation

Status: Aligned.

Leave-one-subject-out outer validation is appropriate for estimating generalization to unseen
participants. Subject-grouped inner validation is critical because trial-level random folds would leak
participant-specific structure. This is aligned with machine-learning guidance for hierarchical and
structured data.

Implementation alignment:

- `orchestration.py` uses leave-one-group-out outer folds and group-aware inner validation.
- Fold-contained preprocessing, missingness handling, residualization, feature transforms, and model
  selection are represented in code.

### 17. ElasticNet Primary, Ridge Secondary, RF and Deep Models Exploratory

Status: Aligned.

ElasticNet is a defensible primary model because it handles correlated channel-frequency predictors,
supports shrinkage, and remains interpretable relative to nonlinear models. Ridge is a useful secondary
linear comparator. Random forests and deep regression are correctly exploratory because sample size and
trial dependence make high-capacity models risky.

Implementation alignment:

- `feature_benchmark.py` runs ElasticNet and Ridge across the required grid.
- Deep regression is configured separately and marked exploratory.

Caveat:

- Deep models can still be useful for feature-learning exploration, but they should not drive the
  primary scientific claim unless externally validated.

### 18. Permutation Inference: Full Refits, Circular Shifts Within Blocks

Status: Mostly aligned.

Permutation inference is appropriate because analytic p-values are hard to justify for nested,
fold-contained predictive pipelines. Full refitting under the null is the right confirmatory procedure.
Restricted circular shifts within task blocks are a thoughtful way to preserve local autocorrelation
and block structure.

The remaining risk is exchangeability. The first trial is fixed high temperature, and temperature
schedule/position may be structured. Circular shifts can preserve within-block sequence shape but may
still move values across special positions unless distance and validity rules are strict enough.
The README's nonzero-shift, minimum-distance, minimum-trial, and invalid-draw controls help.

Implementation alignment:

- `orchestration.py` implements `circular_shift_within_run`, invalid-draw resampling, and a maximum
  invalid permutation fraction.
- Reporting enforces 5,000 completed permutations for required primary outputs.

Risk:

- The code names the scheme `within_run`; the README says task blocks are the exchangeability units.
  If the block column is not explicit and validated, acquisition runs could be used as a weaker proxy.

### 19. Primary Gate and Multiple Testing

Status: Aligned.

The primary gate is prespecified and narrow: NPS, alpha+beta, ElasticNet, positive delta R2, and
one-sided permutation p <= .05. Holm correction across the confirmatory grid is conservative and
appropriate for the required non-primary cells. Keeping exploratory families separate is good.

Implementation alignment:

- `reporting.py` defines the primary gate constants.
- `reporting.py` applies Holm-adjusted primary prediction status.
- `feature_benchmark.py` enforces required primary targets, frequency presets, models, and permutations.

### 20. Study 2 Source-Entry Criteria

Status: Scientifically aligned; implementation partly report-level.

The source-entry thresholds are conservative and appropriate: positive primary prediction, meaningful
delta R2, lower confidence bound, Level 2 convergence, within-subject positive prediction, temporal
negative controls, artifact robustness, and adequate target reliability. This is the correct posture
because EEG scalp features cannot justify source-localized Study 2 claims unless the predictive bridge
is robust.

Implementation alignment:

- `reporting.py` computes source-entry status from the required fields when they exist.

Risk:

- Missing source-entry diagnostics result in `source_entry_not_evaluated`, not failure of primary
  analysis validity. This is fine only if the final report foregrounds missing diagnostics and does
  not claim confirmatory source entry.

## Priority Findings

### F1. Signature Manifest, Checksum, and Support-Retention Rules Are Not Fully Enforced

Severity: High

Status after alignment pass: addressed for Study 1 target preparation and signature scoring.

The README requires a frozen signature manifest with source publication/access record, checksum, image
geometry, and weight-support summaries. It also requires target failure if the common mask retains less
than 90% of original nonzero support, less than 90% of either positive or negative support, or changes
positive/negative total absolute weight mass by more than 10%.

Observed implementation:

- `targets.py` checks required names, path provenance, MNI-space requirement, 3D image shape, finite
  voxels, and nonzero weights.
- `multivariate_signatures.py` computes expression over finite weight support and returns `n_voxels`.
- `targets.py` validates equal NPS/SIIPS1 voxel counts in the assembled target table.

Original gap:

- No visible checksum validation.
- No visible frozen manifest load.
- No visible positive/negative support-retention or weight-mass stability computation.

Implemented action:

- Add an explicit signature-manifest validator before target extraction.
- Store and validate checksum, shape, affine, nonzero support, positive support, negative support,
  positive absolute weight mass, and negative absolute weight mass.
- Emit those fields into target QC outputs and fail target extraction when thresholds are violated.

### F2. HRF-Weighted Nuisance Columns Need Explicit Enforcement

Severity: High

Status after alignment pass: addressed in Study 1 config and nuisance-column validation.

The protocol says Level 2 controls HRF-weighted FD, DVARS, and Fp1/Fp2 artifact power sampled at each
trial's fMRI plateau regressor peak. The config names `framewise_displacement`, `std_dvars`, and
`fp1_fp2_high_frequency_power`, which read as raw trial-level metrics.

Why it matters:

- The target is an fMRI GLM beta/signature expression. Nuisance regressors should be temporally aligned
  with the hemodynamic target estimate.
- Raw trial-level artifact values may not adequately capture BOLD contamination at the target regressor
  timing.

Implemented action:

- Require explicit columns such as `hrf_weighted_framewise_displacement`,
  `hrf_weighted_std_dvars`, and `hrf_weighted_fp1_fp2_high_frequency_power`.
- Add entry-point validation that rejects raw columns when the Study 1 Level 2 config expects
  HRF-weighted covariates.

### F3. LSS Condition Number and Design Efficiency Thresholds Are Not Evident

Severity: Medium-high

Status after alignment pass: addressed in LSS design-matrix validation.

The README excludes target GLMs with condition number above 100 or design efficiency below 0.1.
The code validates rank deficiency and residual degrees of freedom, which is necessary but weaker.

Why it matters:

- A full-rank design can still be unstable.
- LSS estimates are vulnerable to neighboring-event collinearity.
- The explicit thresholds are part of the protocol's scientific validity gate.

Implemented action:

- Compute condition number and target-regressor design efficiency for each LSS model.
- Fail target extraction when thresholds are exceeded.

### F4. Interpretation Diagnostics Are Consumed but Not Generated by the Core Benchmark

Severity: Medium-high

Status after alignment pass: remaining follow-up.

The report expects fields for target reliability, precision, Level 2 convergence, within-subject
centered prediction, temporal negative controls, artifact robustness, HRF/timing robustness,
first-exposure robustness, baseline robustness, and smoothing robustness.

Observed implementation:

- `reporting.py` adds missing diagnostic fields as `NA`.
- Missing diagnostics are listed in `missing_interpretation_diagnostics`.
- Study 2 source entry becomes `source_entry_not_evaluated` when required diagnostics are missing.

Gap:

- `analysis_validity_status` is set to `analysis_valid` for primary benchmark rows once required
  primary outputs exist. That status does not mean all interpretation diagnostics passed.

Recommended action:

- Rename or split statuses to avoid ambiguity:
  - `primary_outputs_valid`
  - `interpretation_diagnostics_complete`
  - `source_entry_status`
- Generate the major diagnostics as first-class pipeline outputs, not optional metrics injected later.

### F5. Task Block Identity Should Be Explicit, Not Inferred from Run

Severity: Medium

Status after alignment pass: addressed in Study 1 target preparation.

The README says confirmatory trial-order and permutation rules use the six 11-trial task blocks even
when BIDS `run` labels differ. Current utilities can fall back to run/session labels when block is
missing.

Why it matters:

- Restricted permutations depend on the correct exchangeability unit.
- Acquisition runs and task blocks are not interchangeable if a run contains multiple blocks.

Implemented action:

- Require an explicit task-block column for Study 1 feature benchmarking and circular-shift permutation.
- Fail fast if only acquisition run is available in a dataset where run/block are not one-to-one.

### F6. Respiratory Physiology Is Not Mentioned as a Required Nuisance

Severity: Medium

The design includes ECG but not respiration. For fMRI, respiration can affect BOLD signal through motion,
CO2, and physiological noise. fMRIPrep confounds and CompCor help, but respiration-specific regressors
would improve physiological control.

Recommended action:

- If respiration was not recorded, state this limitation explicitly in the protocol and final report.
- If recorded, add respiratory nuisance derivation and validation alongside ECG.

### F7. Sample Size Is Reasonable for Feasibility, Not Strong Biomarker Claims

Severity: Medium

Planned N=60, expected analyzable about 54, and minimum N=30 are reasonable for a feasibility bridge
with LOSO prediction and precision simulation. They are not sufficient for strong biomarker claims
without external validation.

Recommended action:

- Keep the primary claim as out-of-sample prediction of fMRI signature expression.
- Treat N=30 as a fail-fast minimum, not as evidence that inference is adequately powered.
- Present precision intervals and invalid-permutation rates before interpretation.

## Literature Crosswalk

### Pain fMRI Signatures

- NPS use is aligned with Wager et al. 2013 because the task is heat-induced physical pain and the
  target is evoked nociceptive signature expression.
- SIIPS1 use is aligned with Woo et al. 2017 because it tests pain-related variance beyond nociceptive
  input and supports interpretation beyond temperature coding.
- The protocol's caution around biomarker language aligns with pain biomarker reviews emphasizing
  validation boundaries and context-specific interpretation.

### Single-Trial fMRI

- LSS is aligned with Mumford et al. 2012 for trial-wise beta estimation under collinearity.
- fMRIPrep use is aligned with reproducible preprocessing practice.
- Motion, DVARS, WM/CSF, CompCor, and outlier controls align with standard fMRI confound literature.

### EEG and Simultaneous EEG-fMRI

- Gradient and BCG artifact correction are required for simultaneous EEG-fMRI and are appropriately
  treated as upstream prerequisites.
- PREP/PyPREP, ICA/ICLabel, and autoreject are aligned with established EEG preprocessing literature.
- Gamma as exploratory is strongly supported by the high-frequency muscle artifact literature.

### Predictive Modeling and Inference

- LOSO and grouped inner validation align with guidance for hierarchical/structured data.
- Fold-contained residualization and transformations are necessary to prevent leakage.
- Restricted permutation inference and full model refits align with permutation-inference principles.
- Linear regularized primary models are better matched to expected sample size than nonlinear models.

## Final Judgment

The Study 1 design is scientifically coherent and mostly literature-aligned. Its strongest form is:

> Trial-wise plateau EEG alpha/beta power can be tested as a held-out-subject predictor of externally
> defined fMRI pain-signature expression, with interpretation limited by target reliability, nuisance
> convergence, artifact robustness, and temporal specificity.

After the alignment passes, the most important target-construction and LSS-design gates are
executable: signature manifest/support/scoring-mask validation, explicit HRF-weighted nuisance
covariates, LSS condition-number/design-efficiency enforcement, task-block identity, and explicit
raw/unbaselined temporal negative-control configuration. The remaining practical follow-up is to
generate the broader interpretation diagnostics consumed by `reporting.py` as first-class pipeline
outputs.

## Key Sources

- Wager TD et al. 2013. An fMRI-based neurologic signature of physical pain.
  https://pubmed.ncbi.nlm.nih.gov/23574118/
- Woo CW et al. 2017. Quantifying cerebral contributions to pain beyond nociception.
  https://www.nature.com/articles/ncomms14211
- CANlab NPS pattern information.
  https://sites.google.com/dartmouth.edu/canlab-brainpatterns/multivariate-brain-signatures/2013-nps
- van der Miesen MM, Lindquist MA, Wager TD. Neuroimaging-based biomarkers for pain.
  https://www.iasp-pain.org/papers/125819-neuroimaging-based-biomarkers-pain-state-field-and-current-directions
- Mumford JA et al. 2012. Deconvolving BOLD activation in event-related designs for multivoxel pattern
  classification analyses.
  https://pubmed.ncbi.nlm.nih.gov/21924359/
- Esteban O et al. 2019. fMRIPrep: a robust preprocessing pipeline for functional MRI.
  https://pubmed.ncbi.nlm.nih.gov/30532080/
- Power JD et al. 2012. Spurious but systematic correlations in functional connectivity MRI networks
  arise from subject motion.
  https://pubmed.ncbi.nlm.nih.gov/22019881/
- Behzadi Y et al. 2007. A component based noise correction method, CompCor.
  https://pubmed.ncbi.nlm.nih.gov/17560126/
- Bigdely-Shamlo N et al. 2015. The PREP pipeline.
  https://www.frontiersin.org/Article/10.3389/fninf.2015.00016/abstract
- Pion-Tonachini L et al. 2019. ICLabel.
  https://colab.ws/articles/10.1016/j.neuroimage.2019.05.026
- Jas M et al. 2017. Autoreject.
  https://pubmed.ncbi.nlm.nih.gov/28645840/
- Muthukumaraswamy SD. 2013. High-frequency brain activity and muscle artifacts in MEG/EEG.
  https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00138/full
- Winkler AM et al. 2014. Permutation inference for the general linear model.
  https://doi.org/10.1016/j.neuroimage.2014.01.060
- Roberts DR et al. 2017. Cross-validation strategies for data with temporal, spatial, hierarchical,
  or phylogenetic structure.
  https://doi.org/10.1111/ecog.02881
