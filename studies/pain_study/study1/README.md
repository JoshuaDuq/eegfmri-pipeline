# Study 1 - EEG Prediction of Trial-Wise fMRI Pain-Signature Expression

## 1. Problem Statement

Pain responses vary among individuals exposed to the same nociceptive stimulus (Coghill et al.,
2003, Kim et al., 2004). EEG can measure fast oscillatory dynamics during painful stimulation
(Kim et al., 2020, Mari et al., 2022), but scalp alpha, beta, and gamma power are spatially
ambiguous and can reflect nociception, aversion, movement, facial muscle activity, cardiac
contamination, scanner artifact, or expectancy.

Simultaneous fMRI provides spatially defined targets for EEG validation (Davis et al., 2020,
van der Miesen et al., 2019). The Neurologic Pain Signature (NPS, Wager et al., 2013) indexes
evoked nociceptive signature expression, whereas the Stimulus Intensity Independent Pain
Signature-1 (SIIPS1, Woo et al., 2017) targets pain-related variance less dependent on stimulus
intensity. NPS and SIIPS1 are predefined fMRI-derived targets analyzed alongside subjective ratings
and thermal intensity as related but distinct criteria.

This study tests whether plateau-window EEG spectral power predicts trial-wise NPS and SIIPS1
expression during simultaneous EEG-fMRI thermal stimulation. The primary estimand is
fold-contained, subject-held-out incremental prediction beyond prespecified stimulus, acquisition,
and physiological nuisance structure, not within-sample EEG-target association.

## 2. Objectives

The primary objective is to test whether plateau-window EEG spectral power predicts trial-wise NPS
or SIIPS1 expression beyond measured nuisance structure. Prediction uses leave-one-subject-out
(LOSO) cross-validation with subject-grouped inner validation. Target construction, feature
preprocessing, imputation, filtering, nuisance residualization, hyperparameter selection, and target
transformation are all estimated without held-out-subject information.

The secondary objective is to test whether significant raw-target prediction converges with a
stimulus- and acquisition-controlled residualized-target analysis. Unadjusted EEG-only prediction is
descriptive. Subjective-rating residualization is a construct-attenuation sensitivity analysis.

Study 3 source interpretation is tied a priori to the NPS ElasticNet alpha+beta individual-channel
spectral-power cell. SIIPS1 remains a co-primary prediction target, while ROI-level and
global-average feature resolutions remain spatial-resolution sensitivities.

## 3. Hypotheses and Confirmatory Estimands

The primary hypothesis is that EEG features add out-of-sample predictive value for trial-wise fMRI
pain-signature expression beyond a nuisance-only model.

$$\Delta R^2_{\text{LOSO}} = R^2_{\text{nuisance+EEG}} - R^2_{\text{nuisance-only}} > 0.$$

The primary confirmatory estimand is the subject-weighted mean incremental raw-target prediction:

$$\Delta R^2_{\text{LOSO}} = \frac{1}{S}\sum_{s=1}^{S} \Delta R_s^2.$$

Confirmatory EEG prediction requires positive $\Delta R^2_{\text{LOSO}}$ and Holm-corrected
permutation significance. Confirmatory Study 3 source interpretation additionally requires the
predesignated NPS ElasticNet alpha+beta individual-channel spectral-power cell to pass the
practical-effect gate:
$\Delta R^2_{\text{LOSO}} \geq 0.02$ with a one-sided 95% lower confidence bound above 0.005.
Effects below this practical-effect gate do not support confirmatory source mapping.

The confirmatory family comprises 2 targets (NPS, SIIPS1) × 2 linear models (ElasticNet, Ridge)
× 3 frequency presets (alpha, beta, alpha+beta), evaluated with the primary
$\Delta R^2_{\text{LOSO}}$ statistic and the individual-channel spectral-power feature matrix. The
secondary convergence family uses the same cells to test residualized-target Level 2
$\Delta R^2_{\text{LOSO}}$ with separate Holm correction. The downstream Study 3 Level 2 convergence
gate is a minimal-effect gate: the predesignated NPS ElasticNet alpha+beta individual-channel cell
must have Level 2 $\Delta R^2_{\text{LOSO}} \geq 0.005$. Because Study 3 source maps use
within-subject standardized contribution scores, the same cell must also have positive
within-subject-centered diagnostic $\Delta R^2_{\text{LOSO}}$. Holm-corrected Level 2 significance is
reported as stronger stimulus- and acquisition-controlled convergence but is not required for
Study 3 eligibility.
ROI-level and global-average feature matrices are spatial-resolution sensitivity analyses with
separate correction. Gamma, unadjusted Level 1 prediction, subjective-rating residualization, Random
Forest, deep regression, and alternative designs remain outside the confirmatory family.

## 4. Study Design and Data Scope

### 4.1 Participants

The planned sample comprises 60 healthy adults, balanced by sex (30 women and 30 men). Eligible
participants are 18-50 years old, right-handed according to the adapted Edinburgh Handedness
Inventory (laterality quotient > +40), and have normal or corrected-to-normal vision. Exclusion
criteria include chronic or persistent pain, diagnosed neurological or psychiatric conditions,
pregnancy or breastfeeding, medication use that alters attention or vigilance, MRI
contraindications, and hair or hairstyle characteristics incompatible with high-quality EEG
recording.

Participants complete one approximately 180 min session at the CERVO neuroimaging unit, including
EEG preparation, task familiarization, structural MRI, resting-state fMRI, and the simultaneous
EEG-fMRI thermal pain task.

Self-report instruments characterize demographic, psychological, sleep, handedness, and pain-related
individual differences: sociodemographic and hormonal status, PHQ-9, GAD-7, RU-SATED, adapted
Edinburgh Handedness Inventory, Gender Role Expectation of Pain questionnaire, and Pain
Catastrophizing Scale. These variables support exploratory moderation analyses. Moderators are
restricted to sex assigned at recruitment, hormonal-status variables, PCS total score, GREP
self-ratings, PHQ-9, GAD-7, and RU-SATED total score. Each moderator is tested in a separate
trial-level linear mixed model with random intercepts for participant. Outcomes are NPS expression,
SIIPS1 expression, and the within-scale thermal/pain intensity score. Fixed effects are the
out-of-sample EEG residual prediction, moderator, and interaction. Continuous moderators are
standardized within the retained sample. Categorical moderators use data-dictionary reference levels.
Moderator-interaction p-values are Holm-corrected within each outcome.

The target sample allows approximately 10% loss from motion, technical failure, or physiological
artifact, yielding an expected analyzable sample of approximately 54 participants. Confirmatory
analysis requires at least 30 analyzable subjects after EEG, fMRI, synchronization, and
artifact-quality exclusions.

If more than 10% of recruited participants are lost, the study is labeled attrition-limited before
outcome inspection. Confirmatory Study 1 interpretation then requires at least 30 analyzable
subjects, a retained-trial distribution compatible with the permutation plan, positive primary
$\Delta R^2_{\text{LOSO}}$ with Holm-corrected significance, and the Section 9.2 precision
simulation. The practical-effect gate applies to downstream Study 3 source interpretation for the
predesignated NPS ElasticNet alpha+beta individual-channel cell.

### 4.2 Thermal Pain Protocol

Thermal stimulation is delivered to the inner forearm contralateral to the response hand using an
MRI-compatible QST.Lab T11 thermode with five independently controlled contact surfaces over 9 cm².
Temperatures are not individually calibrated to subjective pain intensity. The common stimulus
protocol targets NPS and SIIPS1 expression while preserving between-participant thermal-sensitivity
variance. Temperature, ratings, and nuisance structure are modeled separately.

Before scanning, two practice trials verify task comprehension and maximum-temperature tolerability.
Participants unable to tolerate the maximum temperature do not proceed.

The MRI task comprises six blocks of 11 trials, for a total of 66 thermal trials. Six temperatures
from 44.3 to 49.3 °C are presented 11 times each. Trial order is generated by constrained
randomization so that consecutive stimuli are not delivered on the same thermode surface,
auto-transitions between identical temperatures are excluded, and ordered transitions between
distinct temperatures are balanced across the session. The first trial of the first block uses
49.3 °C as a protocol-fixed exposure to the highest planned intensity. This prevents later first
exposure to the maximum stimulus from being confounded with mid-session trial history, but it does
not remove novelty, threat, or scanner-acclimation effects from that trial. The six 11-trial task
blocks serve as the analytic within-session units for block nuisance terms, block-aware resampling,
and circular-shift permutation. If acquisition files use a separate BIDS `run` label, that label is
retained as acquisition metadata, whereas confirmatory trial-order and permutation rules use the
explicit task-block identifier.

Because fixed temperatures can evoke painful and non-painful percepts in different participants,
all quality-controlled thermal trials are retained for the primary fMRI-signature prediction
analysis. Binary pain reports and continuous ratings are criterion and sensitivity variables, not
eligibility filters for the primary target. Temperature enters the Level 2 nuisance design so that
residualized-target analyses test EEG prediction beyond the fixed stimulus-intensity structure.

The deterministic first high-temperature trial is modeled through trial onset, within-block trial
number, and task-block index. Sensitivity analyses repeat the primary incremental model after
excluding the first trial and, separately, after excluding the first block. A result remains eligible
for confirmatory interpretation only if the sign of $\Delta R^2_{\text{LOSO}}$ is unchanged, the
estimate changes by less than 0.02 in both sensitivity analyses, and the target-specific temporal
negative-control gate remains satisfied.

Trial-history sensitivity analyses augment the Level 2 nuisance design with previous-trial
temperature, signed temperature change, cumulative exposure count, and previous-trial within-scale
rating when available.

Each trial begins with a variable 15-20 s fixation interval while the thermode is held at 35.0 °C.
Thermal stimulation lasts 12.5 s, including a 3.0 s ramp-up, a 7.5 s plateau, and a 2.0 s return to
baseline. After a 4.5-8.5 s post-stimulus fixation interval, participants report whether the
stimulus was painful and then rate intensity on a visual analogue scale.
Non-painful trials use a heat-intensity scale from 0 to 99, whereas painful trials use a
pain-intensity scale from 100 to 200. Behavioral data are stored as the binary pain report, the raw
displayed rating, and a within-scale thermal/pain intensity score. The within-scale score retains
the 0 to 99 heat rating for non-painful trials and subtracts 100 from painful-trial ratings,
yielding a 0 to 100 pain-intensity score. The raw 0 to 200 displayed rating is descriptive and is
not used as a single linear covariate in Level 3 residualization. Responses are collected with an
MRI-compatible five-button Pyka response device, and visual stimuli are projected to an
MRI-compatible display viewed through a head-coil mirror.

Behaviorally implausible responses are excluded before EEG or fMRI outcome inspection. A trial is
implausible when the raw displayed rating is outside the logged response scale or equals 0
("no sensation") at a stimulus temperature ≥ 47.3 °C. A participant is excluded from confirmatory
analyses if at least 3 high-temperature trials meet the 0-rating rule or if more than 10% of
otherwise synchronized thermal trials are behaviorally implausible.

### 4.3 Acquisition Summary

EEG is acquired with a 64-channel BrainCap MR system and BrainAmp MR Plus amplifier at 5,000 Hz
before scanner-artifact correction and downsampling. Passive Ag/AgCl electrodes are positioned
according to the extended 10-20 system. Hardware recording filters are set to 0.1-100 Hz, and
electrode impedances are maintained below 20 kΩ, with a target below 10 kΩ for most channels. An
ECG channel records cardiac activity for artifact correction and physiological nuisance
quantification.
EEG and fMRI timing are synchronized through BrainVision volume markers.

Functional MRI is acquired on a Siemens MAGNETOM Prisma 3 T system using a multiband T2*-weighted
echo-planar sequence (TR = 900 ms, TE = 20 ms, multiband factor 3, 3 × 3 × 3 mm voxels,
54 axial slices). A T1-weighted MP-RAGE anatomical image (1 × 1 × 1 mm voxels), a 10 min
resting-state fMRI acquisition, and paired field maps support anatomical registration,
resting-state characterization, and distortion correction.

### 4.4 Modeling Scope

Analyses use simultaneous EEG-fMRI data from thermal plateau trials, including painful and
non-painful trials. fMRI inputs are normalized to MNI152NLin2009cAsym space.

EEG epochs span −7.0 to 15.0 s relative to stimulus onset. The active EEG prediction window is the
thermal plateau interval from 3.0 to 10.5 s post-onset, and fMRI trial targets are LSS plateau
estimates aligned to the same protocol-defined plateau phase.

The multimodal dataset links fMRI targets to clean EEG trials using unique subject, acquisition-run,
task-block, and trial-index identifiers. EEG trigger onsets and fMRI plateau onsets are compared
after applying the protocol offset from trigger to plateau start. Retained trials require residual
absolute mismatch ≤ 0.010 s.

## 5. EEG Preprocessing and Artifact Controls

### 5.1 Simultaneous EEG-fMRI Preprocessing

MRI-induced EEG artifacts are corrected in BrainVision Analyzer 2.0 (Brain Products GmbH). Gradient
artifacts are removed with sliding-window average artifact subtraction, and ballistocardiogram
artifacts are corrected with template subtraction aligned to the detected cardiac cycle.

Corrected EEG data are imported into the MNE-BIDS-Pipeline, downsampled to 500 Hz, band-pass
filtered from 0.1 to 100 Hz, and notch-filtered at 60 Hz. Bad channels are identified with PyPREP
(Bigdely-Shamlo et al., 2015) using deviation-based and correlation-based criteria. Detection is
repeated three times with independent random seeds. A channel is marked bad only when flagged in a
strict majority. Bad channels are interpolated with spherical splines.

Confirmatory analyses use a common-average reference excluding Fp1 and Fp2. A reference including
Fp1/Fp2 is reported as an artifact-sensitivity analysis.

ICA uses extended infomax with 0.99 variance explained and is fitted on 1.0 Hz high-pass-filtered
epochs. ICA spatial weights are then applied to the 0.1-100 Hz continuous analysis data. ICLabel
(Pion-Tonachini et al., 2019) is used for component classification. Components are rejected when
their predicted probability exceeds 0.8 for any non-brain category other than "other."

Because ICLabel was trained on standard non-fMRI EEG, retained components undergo a prespecified
artifact audit before target construction and model fitting. The audit is blind to NPS, SIIPS1,
ratings, and model outcomes, and uses only component spectra, cardiac phase-locking,
volume-repetition spectral peaks, and scalp topographies. Components are rejected for
volume-repetition spectral peak robust z > 3, cardiac phase-locking above the 95th percentile of the
within-subject circular-shift null, or a high-frequency topographic artifact rule. The topographic
rule requires 70-95 Hz component power robust z > 3 plus either at least 50% absolute IC topographic
mass on Fp1, Fp2, FT9, FT10, TP9, and TP10, or within-subject Pearson r ≥ 0.50 between the component
70-95 Hz plateau envelope and the Fp1/Fp2 artifact proxy. Audited components, rejection reasons, and
thresholds are logged before feature extraction.

The preprocessing sequence is fixed in this order.

1. Create preliminary epochs on a 1.0 Hz high-pass-filtered copy.
2. Fit ICA on these filtered epochs.
3. Apply ICA spatial weights to the 0.1-100 Hz continuous analysis data.
4. Extract final analysis epochs from −7.0 to 15.0 s relative to stimulus onset.
5. Apply autoreject (Jas et al., 2017) in local mode with candidate interpolation counts
   {4, 8, 16} for trial rejection.

Subject-specific electrode positions are digitized with EasyCap M1 channel labels and co-registered
to individual MRI. Confirmatory Study 3 source interpretation requires subject-specific digitization
passing quality control. Subject-level unsupervised EEG preprocessing is performed independently
within each subject before cross-validation.

### 5.2 Epochs, Baselines, and Frequency Bands

A −0.2 to 0.0 s pre-stimulus voltage baseline removes DC offset before ERP and amplitude-based
analyses. For time-frequency decompositions, the primary log-ratio baseline is −5.0 to −0.01 s to
support stable alpha-band estimates under Morlet cycle requirements. Because this interval may
include cue-locked expectancy activity, sensitivity baselines are −0.2 to −0.01 s and −7.0 to
−5.5 s. The latter is interpreted only if event logs confirm no cue onset in that window. Baseline
conclusions require unchanged sign and inferential status of primary alpha and beta effects across
the primary and immediate-baseline analyses.

Neural oscillations are operationalized as alpha (8.0–12.9 Hz), beta (13.0–30.0 Hz), and gamma
(30.1–80.0 Hz). Confirmatory frequency presets are alpha, beta, and alpha+beta. Gamma and
gamma-containing composites are exploratory because simultaneous EEG-fMRI gamma is vulnerable to
facial muscle activity, jaw tension, scanner residuals, and cardiac artifacts. The gamma band
excludes a ± 1.0 Hz notch around 60 Hz due to line-noise removal.

### 5.3 Fp1/Fp2 Frontal High-Frequency Artifact Proxy

Because independent facial EMG channels are not acquired, artifact control uses a prespecified
Fp1/Fp2 high-frequency proxy and the dedicated ECG channel. Fp1/Fp2 are scalp EEG electrodes, so
this proxy is artifact control, not independent physiological validation.

The Fp1/Fp2 proxy is computed from the gradient- and BCG-corrected continuous signal after
downsampling, band-pass filtering, and notch filtering, but before PyPREP interpolation, ICA,
autoreject, or epoch-level rejection. For each plateau trial, Fp1 and Fp2 are refiltered to 70–95 Hz
with the existing 60 Hz notch excluded. Hilbert power is averaged over 3.0–10.5 s, log-transformed
with the prespecified training-fold offset, averaged across Fp1 and Fp2, and standardized within
subject using the median and MAD of retained plateau trials.

The HRF-weighted Fp1/Fp2 nuisance regressor is the canonical-HRF convolution of this trial series
sampled at each trial's fMRI plateau regressor peak. The unweighted proxy supports categorical
artifact censoring and continuous artifact-effect reporting. Fp1/Fp2 high-frequency power and
channels are excluded from confirmatory predictive features.

A subject fails primary inclusion if Fp1, Fp2, or ECG is absent, unsynchronized with EEG/fMRI event
logs, flat or saturated for any retained acquisition run, marked bad by PyPREP in a strict majority
of repetitions, interpolated, or unable to yield valid artifact metrics for at least 90% of
otherwise retained plateau trials. Missing physiological artifact metrics are not imputed.

Artifact censoring thresholds are framewise displacement > 0.5 mm, Fp1/Fp2 high-frequency power
robust z > 3, DVARS robust z > 3, cardiac phase-locking above the 95th percentile of the
within-subject circular-shift null, and scanner-frequency residual peaks robust z > 3. Continuous
artifact-effect associations are reported alongside categorical censoring. Gamma effects are labeled
artifact-sensitive when removing threshold-exceeding trials changes the effect direction or
eliminates significance.

## 6. fMRI Signature Target Construction

Trial-wise fMRI effects are estimated with Least-Squares Separate (LSS) models restricted to thermal
plateau trials. For each eligible target trial, one GLM includes a target-trial plateau regressor
with onset at plateau start and duration equal to the plateau hold. Other eligible plateau trials in
the same acquisition run are modeled with the prespecified pooled `other_trials` nuisance regressor
(`lss_other_regressors: all`). Ramp-up, ramp-down, fixation, and response epochs are modeled as
non-plateau nuisance events when timing is available.

Primary LSS models use a canonical SPM hemodynamic response function, cosine drift model, and
0.008 Hz high-pass filter without spatial smoothing. The denoising design includes the 24-parameter
rigid-body motion expansion, white-matter and CSF signals, framewise displacement, CompCor
regressors, and fMRIPrep motion-outlier regressors. Trials with framewise displacement > 0.5 mm or
standardized DVARS robust z > 3 are ineligible as target trials, but their thermal-event timing
remains in the nuisance-event design when valid. Task blocks losing more than 20% of plateau target
trials fail confirmatory inclusion. A target GLM is ineligible if the target regressor is absent,
duplicate-labeled, nonestimable, or has variance inflation pushing the design condition number above
the prespecified threshold.

HRF and timing robustness are evaluated by repeating target construction with HRF temporal and
dispersion derivatives, with a finite-impulse-response model, and after shifting the EEG active
window by ± 2.0 s. HRF/timing stability requires stable primary incremental
$\Delta R^2_{\text{LOSO}}$ (|Δ(ΔR²)| < 0.02), retained significance, and positive Level 2
residualized-target convergence.

NPS and SIIPS1 maps are registered to MNI152NLin2009cAsym space. Signature assets are
`NPS/weights_NSF_grouppred_cvpcr.nii.gz` and
`SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz`, resolved relative to the signature-map root.
SIIPS1 provenance is anchored to the CANlab Neuroimaging_Pattern_Masks repository. NPS provenance is
anchored to Wager et al. (2013) and the authorized CANlab NPS distribution or access record.
CanlabCore pattern-mask conventions define scoring. A frozen signature manifest records each source
publication, source repository or access record, source commit or release when applicable, retrieval
date, local path, file size, SHA-256 checksum, image shape, affine, voxel size, nonzero voxel count,
positive/negative weight counts, and total positive/negative absolute weight mass. Missing files,
absent provenance fields, changed checksums after manifest freeze, or mismatched target names make
the affected signature ineligible.

When image grids differ, LSS beta maps are resampled to the signature-weight grid using continuous
interpolation. Published weight sign and scale are preserved. Weights are not normalized,
re-estimated, rescaled, thresholded, or sign-flipped using study data. For each signature, a fixed
common group scoring mask $V^{(k)}$ is defined before subject-level scoring so voxel count and
spatial extent are identical across subjects, runs, and trials.

A signature target fails validity checks if the common mask retains less than 90% of original
nonzero signature support, retains less than 90% of either positive- or negative-weight support, or
changes positive or negative total absolute weight mass by more than 10% after resampling. Reports
include retained positive and negative voxel counts, percentage support retained, and
positive/negative weight-distribution stability. Sensitivity analyses use the canonical signature
grid.

Primary LSS beta maps are not smoothed, z-scored, or trial-normalized before scoring. A fixed 4 mm
FWHM smoothing sensitivity repeats signature scoring after target-map construction. Signature
weights are not re-estimated from study data. Signature expression is:

$$y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v),$$

where $\beta_{s,i}(v)$ is the LSS-derived BOLD estimate for subject $s$, trial $i$, voxel $v$, and
$M_k(v)$ is the a priori weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$.

LSS diagnostics include retained trial counts, design efficiency and collinearity, and
temperature-stratified split-half reliability of NPS/SIIPS1 expression. Primary inclusion is based
on acquisition and design estimability. Subjects are excluded from confirmatory analyses if they
retain fewer than 25 plateau trials, have LSS design condition number above 100, or have LSS design
efficiency below 0.1. Subjects with 15-24 retained plateau trials are summarized in feasibility or
exploratory analyses. Reliability-informed sensitivity analyses use split-half reliability r ≥ 0.4
and ≥ 30 plateau trials.

Target reliability is reported before substantive interpretation because unreliable targets bound
attainable out-of-sample prediction.

## 7. EEG Feature Construction

Spectral power features are extracted with Morlet wavelets using frequency-adaptive cycle counts
($\text{cycles}=f/2.0$, bounded between 3.0 and 15.0) and decimation factor 4. To remove the
condition-agnostic evoked response, the grand-average ERP is subtracted from each trial before
decomposition. The primary ERP estimate is the fold-level grand average across all training trials,
pooling conditions, and is applied unchanged to held-out subject trials. Subject-specific ERP
subtraction is not used in the primary fold-isolated analysis. Condition-specific ERP subtraction is
evaluated as a stricter sensitivity analysis.

Spectral power is log-ratio baseline-corrected using the primary baseline and averaged within
3.0–10.5 s. Each retained trial contributes one power value per channel and band. The primary
confirmatory matrix uses individual channels. ROI-level and global-average matrices are
spatial-resolution sensitivities. All confirmatory feature matrices exclude Fp1 and Fp2. Matrices
retaining Fp1/Fp2 are exploratory artifact-sensitivity analyses.

The ROI feature matrix uses fixed, non-overlapping scalp groups resolved from normalized extended
10-20 channel names before model fitting. Channel normalization uppercases names and removes
reference suffixes only. The frontal ROI includes FPZ, AF3, AF4, AF7, AF8, AFZ, F1, F2, F3, F4, F5,
F6, F7, F8, F9, F10, and FZ. The frontocentral ROI includes FC1, FC2, FC3, FC4, FC5, FC6, and FCZ.
The central ROI includes C1, C2, C3, C4, C5, C6, and CZ. The centroparietal ROI includes CP1, CP2,
CP3, CP4, CP5, CP6, and CPZ. The parietal ROI includes P1, P2, P3, P4, P5, P6, P7, P8, P9, P10,
and PZ. The occipital ROI includes PO1, PO2, PO3, PO4, PO5, PO6, PO7, PO8, PO9, PO10, POZ, O1, O2,
O9, O10, OZ, CB1, CB2, I1, I2, and IZ. The left temporal ROI includes FT7, FT9, T7, T9, TP7, and
TP9. The right temporal ROI includes FT8, FT10, T8, T10, TP8, and TP10. A ROI is included only when
at least two listed channels are present after preprocessing. Fp1 and Fp2 are excluded from ROI
features. If any predefined ROI fails this requirement in a sensitivity fold, the ROI-resolution
analysis for that fold is ineligible for interpretation. ROI features are arithmetic means of
channel-level log-ratio power within each ROI, computed separately for each frequency preset.

Exploratory features include spectral peak frequency and bandwidth, aperiodic slope and offset
(specparam fixed model), event-related desynchronization/synchronization, band power ratios
(θ/β, θ/α, α/β, δ/α, δ/θ), hemispheric alpha asymmetry, nonlinear complexity measures, and
oscillatory burst statistics. Reports list feature family, target, model, frequency preset,
preprocessing variant, and correction family. Within each exploratory feature family, permutation
p-values are Holm-corrected across tested target-model-frequency cells.

## 8. Nuisance Structure and Residualization Levels

The study separates three estimands by residualization level.

**Level 1 - Raw expression.** Models predict full NPS/SIIPS1 dot-product expression, which can mix
stimulus intensity, condition, temporal structure, subjective experience, and neural pain processing.

**Level 2 - Stimulus- and acquisition-controlled expression.** Targets are residualized against an
intercept, stimulus temperature, task-block index, trial onset time, within-block trial number,
selected thermode surface, HRF-weighted framewise displacement, HRF-weighted standardized DVARS,
Fp1/Fp2 high-frequency artifact power, and residual ECG coupling from the dedicated ECG channel.
The executable Study 1 configuration encodes stimulus temperature and selected thermode surface as
categorical regressors. A continuous nonlinear temperature basis (centered temperature in degrees
Celsius and centered temperature squared) is retained only as a sensitivity analysis. Binary pain
condition is excluded because it is a deterministic or near-deterministic recoding of the thermal
manipulation.

Level 2 tests whether EEG predicts fMRI signature expression beyond prespecified stimulus and
acquisition structure. Because thermal intensity and trial structure are meaningful components of
evoked pain, this control can remove construct-relevant variance.

**Level 3 - Rating-residualized sensitivity.** The Level 2 design is augmented with the binary pain
report and within-scale thermal/pain intensity score, not the raw discontinuous 0 to 200 displayed
rating. This level is interpreted as a construct-attenuation sensitivity analysis.

The primary Level 2 design is fixed across LOSO folds. Before SVD fitting, rank is checked from the
centered and scaled training-fold nuisance matrix. The design is full rank only when every
non-intercept singular value satisfies σ_j / σ_max ≥ 10⁻¹⁰. Cells with rank-deficient Level 2
designs in any outer training fold are ineligible for confirmatory interpretation. The continuous
nonlinear temperature basis remains a sensitivity analysis.

Within each level, nuisance coefficients are estimated exclusively on training subjects using
SVD-based least squares.

$$\hat{\gamma} = \underset{\gamma}{\mathrm{argmin}} \, \| y_{\mathrm{train}} - Z_{\mathrm{train}}\gamma \|_2^2.$$

Residualized targets for training and test sets are computed by applying training-derived
coefficients.

$$y_{\mathrm{train}}^{\mathrm{resid}} = y_{\mathrm{train}} - Z_{\mathrm{train}}\hat{\gamma}, \qquad y_{\mathrm{test}}^{\mathrm{resid}} = y_{\mathrm{test}} - Z_{\mathrm{test}}\hat{\gamma}.$$

## 9. Predictive Modeling

### 9.1 Primary Incremental Model

The primary analysis predicts raw NPS/SIIPS1 expression with a nuisance-only model and a combined
nuisance-plus-EEG model. The nuisance-only model uses unpenalized ordinary least squares with the
same rank-stable nuisance design in every fold. The same pre-SVD rank tolerance used for Level 2
applies here. If the nuisance-only design becomes rank deficient, the affected cell is ineligible
for confirmatory interpretation.

The nuisance-plus-EEG estimator is staged residual learning, not a joint penalized regression with
nuisance and EEG terms in one objective. The nuisance component is unpenalized ordinary least
squares. The penalized EEG model learns only training-fold nuisance residuals. Here,
"nuisance+EEG" means the held-out nuisance prediction plus the inverse-transformed EEG residual
prediction. The Yeo-Johnson target transformation is part of the prespecified ElasticNet and Ridge
pipelines and is estimated only from the training residual target. Each outer fold proceeds as
follows.

1. Fit the nuisance model on untransformed raw training targets.
2. Compute training and held-out nuisance predictions,
   $\hat{y}_{Z,\mathrm{train}}$ and $\hat{y}_{Z,\mathrm{test}}$.
3. Compute training residuals,
   $r_{\mathrm{train}} = y_{\mathrm{train}} - \hat{y}_{Z,\mathrm{train}}$.
4. Fit the Yeo-Johnson transformation on $r_{\mathrm{train}}$ only.
5. Transform $r_{\mathrm{train}}$ and train the EEG model only to predict the transformed residual
   target from EEG features.
6. Apply the trained EEG model to held-out EEG features and inverse-transform the held-out EEG
   residual predictions back to raw residual units.

The held-out combined prediction is defined as follows.

$$\hat{y}_{\mathrm{test}} = \hat{y}_{Z,\mathrm{test}} + \hat{r}_{\mathrm{EEG},\mathrm{test}}.$$

### 9.2 Feature-Based Models

A nested LOSO framework is used. The primary confirmatory model is ElasticNet regression on
individual-channel spectral power. Ridge is a secondary confirmatory linear model that supports
Haufe-style forward-pattern sensitivity analyses in Study 3. Random Forest is an exploratory
nonlinear model.

Feature preprocessing is fold-contained: feature statistics, imputation medians, variance
thresholds, standardization means, and standard deviations are estimated from training subjects
only. Imputation is limited to isolated nonfatal missing feature values within otherwise valid
trials and features. Missing fMRI targets, EEG-fMRI synchronization, event identifiers, required
channels, artifact metrics, frequency-band extraction, or entire trials are exclusion or
analysis-failure events and are not imputed.

For imputation-eligible feature values, the missingness rate must be below 5% for each feature
within the outer training fold and below 10% for each retained subject. Values are imputed with the
corresponding training-cohort median for that feature. Exceeding either limit makes the affected
cell ineligible for confirmatory interpretation. Retained features are standardized to zero mean and
unit variance, with constant features removed.

The Yeo-Johnson transformation is applied in every confirmatory ElasticNet and Ridge cell, only to
the target component learned by the penalized EEG model. In the primary incremental analysis this is
$r_{\mathrm{train}}$. The nuisance-only prediction remains on the raw target scale. In secondary
residualized-target models, targets are residualized before fold-contained transformation.
Predictions are inverse-transformed before primary metrics are reported. Analyses without target
transformation are sensitivities.

The ElasticNet objective is defined as follows.

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2.$$

The Ridge objective is defined as follows.

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \frac{\alpha}{2}\|\beta\|_2^2.$$

Hyperparameters are tuned with 5-fold inner GroupKFold cross-validation restricted to training
subjects, using the same subject-weighted $R^2$ metric as the outer LOSO evaluation. ElasticNet uses
$\rho \in \{0.2, 0.5, 0.8\}$ and a fold-specific logarithmic $\alpha$ grid derived from the
training-fold $\alpha_{\max}$ for each $\rho$:
$\alpha_{\max} \times \{1.0, 0.1, 0.01, 0.001, 0.0001\}$. This ties regularization scale to
training-fold feature and target variance without held-out subjects. ElasticNet uses 10,000 maximum
iterations. Ridge uses $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Random Forest uses
500 estimators with max depths $\in \{5, 10, 20, \text{None}\}$, min samples split
$\in \{2, 5, 10\}$, and min samples leaf $\in \{1, 2, 4\}$.

Confirmatory inference requires at least 30 analyzable subjects and a preregistered precision
simulation showing a 95% CI half-width ≤ 0.10 for the primary $\Delta R^2$ statistic. The simulation
complements, but does not replace, the subject-count requirement. It uses observed subject count,
retained-trial counts, task-block structure, temperature sequence, nuisance matrix, and target
split-half reliability estimated without EEG prediction outcomes.

The simulation uses 10,000 Monte Carlo datasets per retained-sample scenario. Subject-level random
effects, within-block autocorrelation, and trial-wise residual variance are matched to the observed
target and nuisance structure. Variance components come from nuisance-only target residuals using a
random-intercept mixed model with participant and task-block effects. Within-block autocorrelation
is the median Fisher-z-transformed AR(1) coefficient across retained task blocks, back-transformed
for simulation. Simulated EEG residual-prediction effects span
$\Delta R^2 \in \{0.000, 0.005, 0.010, 0.020, 0.050\}$. For each Monte Carlo dataset, the
subject-weighted $\Delta R^2_{\text{LOSO}}$ confidence interval is computed with a percentile
bootstrap over subjects using 10,000 resamples. Confirmatory precision requires the 95th percentile
of simulated 95% CI half-widths to be ≤ 0.10.

Before downstream source-map implementation or inspection, an attrition and precision audit
estimates how many subjects are expected to survive both Study 1 prediction inclusion and
source-stage quality control. Downstream source analysis requires at least 30 source-valid subjects.

### 9.3 Exploratory Deep Regression Model

The BandTemporalRegressor is an exploratory deep regression lane for continuous, band-limited EEG
dynamics.

Continuous EEG is band-pass filtered into target frequency ranges, and instantaneous power is
extracted with the Hilbert transform. Tensors are cropped to 3.0–10.5 s, yielding input shape
N × B × C × T for trials, bands, channels, and time. They are standardized per channel and band
using training-cohort statistics.

$$\tilde{X}_{n,b,c,t} = \frac{X_{n,b,c,t} - \mu_{b,c}}{\sigma_{b,c}^{*}}.$$

The network learns band-specific spatial filters before temporal integration. The depthwise spatial
convolution uses a kernel spanning all channels with independent filter groups per band.

$$H^{(1)}_{n,b,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{c} W^{\mathrm{spat}}_{b,f,c}\tilde{X}_{n,b,c,t}\right)\right).$$

A temporal convolution with kernel length 15 and 8 filters integrates across bands and time.

$$H^{(2)}_{n,f^{\prime},t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{b,f,u} W^{\mathrm{temp}}_{f^{\prime},b,f,u} H^{(1)}_{n,b,f,t+u}\right)\right).$$

The latent representation is downsampled with average pooling (factor 8) and passed through a
dropout-regularized fully connected regression head (p = 0.25).

$$z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}(H^{(2)}_n)\right).$$

$$\hat{y}_n = W_2 \, \mathrm{Dropout}\left(\mathrm{ELU}\left(W_1 \, \mathrm{vec}(z_n) + b_1\right)\right) + b_2.$$

Optimization uses AdamW (learning rate $= 0.001$, weight decay $= 0.0001$) with mean squared error
loss over 25 epochs and batch size 32. Twenty percent of training subjects are reserved for early
stopping with patience 5. Standardization statistics are computed exclusively from the remaining
80% of training subjects. Inner validation subjects are standardized with these statistics without
contributing to their estimation.

## 10. Statistical Inference

Primary metrics are computed on the original target scale after inverse-transforming predictions
and, for the primary incremental analysis, adding back the held-out nuisance prediction. The
coefficient of determination uses the training-fold target mean as the zero-skill baseline.

$$R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{\mathrm{train},f})^2}.$$

Mean $R^2$ for the nuisance-plus-EEG model and pooled trial-wise $R^2$ are reported descriptively.
Subject-wise metrics are reported with 95% BCa bootstrap confidence intervals (10,000 resamples).
Group-level intervals resample subjects. Within-subject intervals use task-block-level resampling or
circular block bootstrap.

Reports decompose held-out predictions and targets into subject means and within-subject deviations.
The primary confirmatory claim is out-of-sample trial-level prediction in held-out subjects.
Within-person trial-tracking interpretation requires positive within-subject-centered diagnostic
$\Delta R^2_{\text{LOSO}}$ for the same target-model-frequency cell.

Primary inference uses nonparametric permutation testing with 5,000 permutations. The primary null
repeats the full observed-analysis training procedure, including fold-level preprocessing
statistics, imputation, constant-feature filtering, target transformation, and inner GroupKFold
hyperparameter selection. Frozen-hyperparameter permutations are computational sensitivities only.

Circular-shift permutations use the six 11-trial task blocks as exchangeability units, retaining
censored blocks when they still support a valid circular shift. Within each block, trials are
ordered by original trial index after censoring. A permutation-valid block must retain at least
8 plateau trials and allow at least four distinct nonzero circular shifts after excluding shifts
shorter than 5 original trial positions. Blocks failing these rules are excluded before confirmatory
model fitting. A subject fails confirmatory prediction analysis if fewer than three
permutation-valid task blocks or fewer than 25 retained plateau trials remain. The precision
simulation uses observed post-censoring block lengths and admissible-shift counts.

For censored blocks, admissible shifts are defined on the retained plateau-trial sequence, not on
the complete 11-trial sequence with imputed gaps. Let retained trials be ordered by their original
within-block trial indices $i_1,\ldots,i_n$. A candidate shift $s \in \{1,\ldots,n-1\}$ assigns the
residual from retained trial $i_{m-s \bmod n}$ to target trial $i_m$ for every retained trial in the
block. The candidate is admissible only when every source-to-target pair has a forward circular
distance of at least 5 positions on the original 11-trial index ring. Censored trials are absent
from both sides of the mapping, and no additional trials are dropped within an admissible shift.

Each outer fold and permutation proceeds as follows.

1. Fit the nuisance-only model on unpermuted training data.
2. Compute nuisance predictions and residuals for training and held-out subjects using
   training-derived nuisance coefficients.
3. Circularly shift the residual component relative to EEG within each task block separately for
   training and held-out subjects, with a minimum shift distance of 5 trials.
4. Reconstruct permuted raw targets as the unshifted nuisance prediction plus shifted residual.
5. Refit nuisance-only and nuisance-plus-EEG models on the permuted training target, including
   inner-fold hyperparameter selection for the EEG model, and score against the permuted held-out
   target.

The $R^2$ denominator uses the permuted training-target mean for that fold, matching the
observed-analysis zero-skill baseline. This approximately preserves within-block temporal
autocorrelation (Winkler et al., 2014). Sensitivity nulls include block-label shuffling,
acquisition-run-level permutations when acquisition runs contain multiple task blocks, removal of
subject and task-block means from both EEG features and targets before permutation, and within-block
random shuffling.

A permutation draw is invalid if any outer fold cannot be scored under the prespecified pipeline:
rank-deficient nuisance design, failed target transformation, zero target or prediction variance
needed for the metric, no retained EEG features after fold-contained filtering, failed inner
GroupKFold split, model non-convergence after the prespecified maximum iterations, or a retained
trial structure that violates the permutation-valid block rules. Invalid draws are resampled before
outcome inspection. Confirmatory inference requires 5,000 valid draws. If more than 20% of attempted
draws are invalid, or if 5,000 valid draws cannot be obtained within the prespecified compute
budget, the affected target-model-frequency cell is downgraded to exploratory. The null size is not
reduced based on interim results.

## 11. Validity and Sensitivity Analyses

### 11.1 Nuisance Prediction Reporting

Supplementary models predict each nuisance variable and behavioral report variable from the same EEG
features. Behavioral report variables are the binary pain report and within-scale thermal/pain
intensity score. Nuisance-prediction p-values are Holm-corrected across supplementary targets.
Reports distinguish nuisance prediction, behavioral-report prediction, and primary incremental
$\Delta R^2_{\text{LOSO}}$.

### 11.2 Temporal Negative Controls

Temporal negative controls use the same nuisance-only versus nuisance-plus-EEG
$\Delta R^2_{\text{LOSO}}$ framework. Models trained on EEG features from −5.0 to 0.0 s and
−0.2 to 0.0 s predict post-stimulus target expression.

Negative controls are evaluated separately for NPS and SIIPS1. A target-specific confirmatory cell
passes the temporal-specificity gate only when both pre-stimulus models are nonsignificant after
Holm correction across the two windows within that cell and show bounded evidence against a
meaningful pre-stimulus effect. For the same target, the one-sided 95% upper confidence bound for
pre-stimulus $\Delta R^2$ must be below 0.02 and below 25% of the observed active-window
$\Delta R^2$. Study 3 requires the predesignated NPS ElasticNet alpha+beta individual-channel
spectral-power cell to pass this same NPS-specific temporal gate.

Wrong-lag windows are ramp-up ($0.0$-$3.0$ s), late ramp-down ($10.5$-$15.0$ s), early-shifted
active ($1.0$-$8.5$ s), and late-shifted active ($5.0$-$12.5$ s). Wrong-lag p-values are
Holm-corrected across the four windows within each target-model-frequency cell. A cell retains
temporal-specificity eligibility only if no wrong-lag window exceeds active-window
$\Delta R^2$ and no Holm-corrected significant wrong-lag window has $\Delta R^2$ at least 75% of the
active-window $\Delta R^2$. The primary temporal-negative control analysis repeats full inner
GroupKFold hyperparameter selection for every pre-stimulus and wrong-lag window. Active-window
hyperparameters are reused only in a secondary frozen-model sensitivity analysis.

## References

Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline.
Standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16.
doi.org/10.3389/fninf.2015.00016

Coghill, R. C., McHaffie, J. G., & Yen, Y.-F. (2003). Neural correlates of interindividual
differences in the subjective experience of pain. Proceedings of the National Academy of Sciences
of the United States of America, 100(14), 8538-8542. doi.org/10.1073/pnas.1430684100

Davis, K. D., Aghaeepour, N., Ahn, A. H., Angst, M. S., Borsook, D., Brenton, A., Burczynski,
M. E., Crean, C., Edwards, R., Gaudilliere, B., Hergenroeder, G. W., Iadarola, M. J., Iyengar, S.,
Jiang, Y., Kong, J.-T., Mackey, S., Saab, C. Y., Sang, C. N., Scholz, J., ... Pelleymounter, M. A.
(2020). Discovery and validation of biomarkers to aid the development of safe and effective pain
therapeutics. Challenges and opportunities. Nature Reviews Neurology, 16(7), 381-400.
doi.org/10.1038/s41582-020-0362-2

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject. Automated
artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
doi.org/10.1016/j.neuroimage.2017.06.030

Kim, H., Neubert, J. K., San Miguel, A., Xu, K., Krishnaraju, R. K., Iadarola, M. J., Goldman, D.,
& Dionne, R. A. (2004). Genetic influence on variability in human acute experimental pain
sensitivity associated with gender, ethnicity and psychological temperament. Pain, 109(3), 488-496.
doi.org/10.1016/j.pain.2004.02.027

Kim, J. A., & Davis, K. D. (2021). Neural oscillations. Understanding a neural code of pain. The
Neuroscientist, 27(5), 544-570. doi.org/10.1177/1073858420958629

Mari, T., Henderson, J., Maden, M., Nevitt, S. J., Duarte, R., & Fallon, N. (2022). Systematic
review of the effectiveness of machine learning algorithms for classifying pain intensity,
phenotype or treatment outcomes using electroencephalogram data. The Journal of Pain, 23(3),
349-369. doi.org/10.1016/j.jpain.2021.07.011

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel. An automated
electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198,
181-197. doi.org/10.1016/j.neuroimage.2019.05.026

van der Miesen, M. M., Lindquist, M. A., & Wager, T. D. (2019). Neuroimaging-based biomarkers for
pain. State of the field and current directions. Pain Reports, 4(4), e751.
doi.org/10.1097/PR9.0000000000000751

Wager, T. D., Atlas, L. Y., Lindquist, M. A., Roy, M., Woo, C.-W., & Kross, E. (2013). An
fMRI-based neurologic signature of physical pain. New England Journal of Medicine, 368(15),
1388-1397. doi.org/10.1056/NEJMoa1204471

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation
inference for the general linear model. NeuroImage, 92, 381-397.
doi.org/10.1016/j.neuroimage.2014.01.060

Woo, C.-W., Schmidt, L., Krishnan, A., Jepma, M., Roy, M., Lindquist, M. A., Atlas, L. Y., &
Wager, T. D. (2017). Quantifying cerebral contributions to pain beyond nociception. Nature
Communications, 8, 14211. doi.org/10.1038/ncomms14211
