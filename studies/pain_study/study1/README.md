# Study 1 - EEG Prediction of Trial-Wise fMRI Pain-Signature Expression

## 1. Problem Statement

Pain responses vary among individuals exposed to the same nociceptive stimulus (Coghill et al.,
2003; Kim et al., 2004). EEG can measure fast oscillatory dynamics during painful stimulation
(Kim and Davis, 2021; Mari et al., 2022), but scalp alpha, beta, and gamma power are spatially
ambiguous and can reflect nociception, aversion, movement, facial muscle activity, cardiac
contamination, scanner artifact, or expectancy (Allen et al., 1998, 2000;
Muthukumaraswamy, 2013).

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

The primary objective is to test whether plateau-window EEG spectral power adds LOSO predictive
value for trial-wise NPS expression beyond measured nuisance structure. Secondary objectives apply
the same framework to SIIPS1 and to stimulus- and acquisition-controlled residualized targets.
Unadjusted EEG-only prediction is descriptive; subjective-rating residualization is a
construct-attenuation sensitivity analysis.

### 2.1 Thesis Logic and Claim Tiers

Study 1 has one primary prediction gate and one Study 2 claim-tier decision: the NPS ElasticNet
alpha+beta individual-channel spectral-power cell. This cell is primary because NPS is the target
most closely aligned with evoked nociceptive signature expression, alpha and beta are more
artifact-defensible than gamma in simultaneous EEG-fMRI, ElasticNet gives sparse linear
contribution scores for Study 2, and individual-channel features preserve spatial information for
source follow-up. Secondary cells describe the prediction landscape but cannot replace this bridge.

Results are interpreted in this order: target reliability, subject and trial retention, precision
and permutation feasibility, primary prediction, temporal and artifact specificity, then Study 2
source-interpretation claim tier. Failed interpretation diagnostics change claim labels; they do
not trigger model search or post hoc replacement of the primary cell.

### 2.2 Decision Schema

Analysis-validity gates and interpretation flags are kept distinct. The primary estimand is
unevaluable only under hard failures: invalid or insufficient cohort/trial structure, invalid
EEG-fMRI alignment, invalid signature targets or provenance, missing required nuisance columns,
rank-deficient required nuisance designs, leakage in fold-contained modeling, infeasible
permutation inference, or missing required primary benchmark outputs.

Once validity gates pass, primary prediction status follows the criteria in Section 3. Target
reliability, precision, Level 2 convergence, within-subject-centered prediction, temporal controls,
artifact robustness, HRF/timing robustness, first-exposure sensitivity, baseline sensitivity, and
smoothing sensitivity are interpretation flags whose status shapes claim language.

Study 2 source-entry criteria are evaluated separately for the primary cell. Passing all of them
supports confirmatory source interpretation; otherwise Study 2 enters the planned
exploratory/source-characterization tier.

## 3. Hypotheses and Confirmatory Estimands

The primary hypothesis is that EEG features add out-of-sample predictive value for trial-wise fMRI
pain-signature expression beyond a nuisance-only model.

$$\Delta R^2_{\text{LOSO}} = R^2_{\text{nuisance+EEG}} - R^2_{\text{nuisance-only}} > 0.$$

The primary confirmatory estimand is the subject-weighted mean incremental raw-target prediction:

$$\Delta R^2_{\text{LOSO}} = \frac{1}{S}\sum_{s=1}^{S} \Delta R_s^2.$$

After analysis-validity gates pass, the primary thesis cell passes the Study 1 prediction gate
when $\Delta R^2_{\text{LOSO}} > 0$ and its valid one-sided upper-tail permutation p-value is
≤ 0.05.

Confirmatory Study 2 source interpretation further requires the primary cell to satisfy every
source-interpretation criterion: $\Delta R^2_{\text{LOSO}} \geq 0.02$, one-sided 95% lower
confidence bound above 0.005, Level 2 $\Delta R^2_{\text{LOSO}} \geq 0.005$, positive
within-subject-centered diagnostic $\Delta R^2_{\text{LOSO}}$, temporally specific
negative-control behavior, artifact-censoring robustness, and interpretable target reliability.
Any failure routes Study 2 to the exploratory/source-characterization tier.

The required feature benchmark spans the full 2 targets (NPS, SIIPS1) × 2 linear models
(ElasticNet, Ridge) × 3 frequency presets (alpha, beta, alpha+beta) grid, evaluated with
$\Delta R^2_{\text{LOSO}}$ on the individual-channel spectral-power matrix. The primary thesis
gate is corrected as its own prespecified family; the remaining benchmark cells form the secondary
confirmatory prediction family with separate Holm correction (Holm, 1979). The secondary
convergence family applies the same grid to residualized-target Level 2
$\Delta R^2_{\text{LOSO}}$, again Holm-corrected within family. Secondary-family success can
strengthen or qualify interpretation but cannot replace the primary gate for Study 2. ROI-level and
global-average feature matrices are spatial-resolution sensitivities with separate correction.
Gamma, unadjusted Level 1 prediction, subjective-rating residualization, Random Forest, deep
regression, and alternative designs lie outside the confirmatory prediction family.

## 4. Study Design and Data Scope

### 4.1 Participants

The planned sample comprises 60 healthy adults, balanced by sex (30 women and 30 men). Eligible
participants are 18-50 years old, right-handed according to the adapted Edinburgh Handedness
Inventory (Oldfield, 1971; laterality quotient > +40), and have normal or corrected-to-normal
vision. Exclusion criteria include chronic or persistent pain, diagnosed neurological or
psychiatric conditions, pregnancy or breastfeeding, medication use that alters attention or
vigilance, MRI contraindications, and hair or hairstyle characteristics incompatible with
high-quality EEG recording.

Participants complete one approximately 180 min session at the CERVO neuroimaging unit, including
EEG preparation, task familiarization, structural MRI, resting-state fMRI, and the simultaneous
EEG-fMRI thermal pain task.

Self-report instruments characterize demographic, psychological, sleep, handedness, and pain-related
individual differences: sociodemographic and hormonal status, PHQ-9 (Kroenke et al., 2001),
GAD-7 (Spitzer et al., 2006), RU-SATED (Buysse, 2014), adapted Edinburgh Handedness Inventory
(Oldfield, 1971), Gender Role Expectation of Pain questionnaire (Robinson et al., 2001), and
Pain Catastrophizing Scale (Sullivan et al., 1995). Exploratory moderators are restricted to sex
assigned at recruitment, hormonal-status variables, PCS total score, GREP self-ratings, PHQ-9,
GAD-7, and RU-SATED total score. Moderator interactions with the out-of-sample EEG residual
prediction are tested in separate mixed models and Holm-corrected within outcome.

The target sample allows approximately 10% loss from motion, technical failure, or physiological
artifact, yielding an expected analyzable sample of approximately 54 participants. Confirmatory
analysis requires at least 30 analyzable subjects after EEG, fMRI, synchronization, and
artifact-quality exclusions.

Attrition above 10% triggers an attrition-limited label before outcome inspection and constrains
thesis-bridge claims unless the primary thesis gate, precision audit, target-reliability audit, and
Study 2 claim-tier criteria remain interpretable.

### 4.2 Thermal Pain Protocol

Thermal stimulation is delivered to the inner forearm contralateral to the response hand using an
MRI-compatible QST.Lab T11 thermode with five independently controlled contact surfaces over 9 cm².
Temperatures are not individually calibrated to subjective pain intensity. Temperature, ratings,
and nuisance structure are modeled separately to preserve between-participant thermal-sensitivity
variance.

Before scanning, two practice trials verify task comprehension and maximum-temperature tolerability.
Participants unable to tolerate the maximum temperature do not proceed.

The MRI task comprises six blocks of 11 trials, for a total of 66 thermal trials. Six temperatures
from 44.3 to 49.3 °C are presented 11 times each. Trial order is generated by constrained
randomization so that consecutive stimuli are not delivered on the same thermode surface,
auto-transitions between identical temperatures are excluded, and ordered transitions between
distinct temperatures are balanced across the session. The first trial of the first block delivers
49.3 °C as a protocol-fixed exposure to the highest planned intensity. The six 11-trial task blocks
serve as the analytic units for block nuisance terms, block-aware resampling, and circular-shift
permutation. If acquisition files use a separate BIDS `run` label (Gorgolewski et al., 2016), that
label is retained as acquisition metadata; confirmatory trial-order and permutation rules use the
explicit task-block identifier. The target table records the original event-level trial order from
`trial_number` when available and from `trial_index` otherwise. One of these columns is required.
Censored trials retain their original labels and are not renumbered.

Because fixed temperatures can evoke painful and non-painful percepts in different participants,
all quality-controlled thermal trials are retained for the primary fMRI-signature prediction
analysis. Binary pain reports and continuous ratings are criterion and sensitivity variables, not
eligibility filters for the primary target. Temperature enters the Level 2 nuisance design so that
residualized-target analyses test EEG prediction beyond the fixed stimulus-intensity structure.

The deterministic first high-temperature trial is modeled by trial onset, within-block trial
number, and task-block index. Sensitivity analyses repeat the primary incremental model after
excluding the first trial and, separately, after excluding the first block. Sign changes,
|Δ(ΔR²)| ≥ 0.02, or temporal-negative-control concerns are flagged for the affected
target-model-frequency cell.

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
yielding a 0 to 100 pain-intensity score. Responses are collected with an MRI-compatible
five-button Pyka response device, and visual stimuli are projected to an MRI-compatible display
viewed through a head-coil mirror.

Behaviorally implausible responses are excluded before EEG or fMRI outcome inspection. A trial is
implausible when the raw displayed rating is outside the logged response scale or equals 0
("no sensation") at a stimulus temperature ≥ 47.3 °C. Repeated high-temperature 0 ratings are
reported as a behavior-QC flag. A participant is excluded from confirmatory analyses only when more
than 10% of otherwise synchronized thermal trials are behaviorally implausible.

### 4.3 Acquisition Summary

EEG is acquired with a 64-channel BrainCap MR system and BrainAmp MR Plus amplifier at 5,000 Hz
before scanner-artifact correction and downsampling. Passive Ag/AgCl electrodes are positioned
according to the extended 10-20 system (Jasper, 1958). Hardware recording filters are set to
0.1-100 Hz, and electrode impedances are maintained below 20 kΩ, with a target below 10 kΩ for
most channels. An ECG channel records cardiac activity for artifact correction and physiological
nuisance quantification.
EEG and fMRI timing are synchronized through BrainVision volume markers.

Functional MRI is acquired on a Siemens MAGNETOM Prisma 3 T system using a multiband T2*-weighted
echo-planar sequence (TR = 900 ms, TE = 20 ms, multiband factor 3, 3 × 3 × 3 mm voxels,
54 axial slices). A T1-weighted MP-RAGE anatomical image (1 × 1 × 1 mm voxels), a 10 min
resting-state fMRI acquisition, and paired field maps support anatomical registration,
resting-state characterization, and distortion correction.

### 4.4 Modeling Scope

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
artifacts are removed with sliding-window average artifact subtraction (Allen et al., 2000), and
ballistocardiogram artifacts are corrected with template subtraction aligned to the detected cardiac
cycle (Allen et al., 1998).

Corrected EEG data are imported into the MNE-BIDS-Pipeline (Appelhoff et al., 2019; Jas et al.,
2018), downsampled to 500 Hz, band-pass filtered from 0.1 to 100 Hz, and notch-filtered at 60 Hz.
Bad channels are identified with PyPREP (Bigdely-Shamlo et al., 2015) using deviation-based and
correlation-based criteria. Detection is repeated three times with independent random seeds. A
channel is marked bad only when flagged in a strict majority. Per-run PyPREP outputs are retained
as provenance, and Study 1 uses the subject-level union of bad channels for shared ICA fitting,
final cross-run epochs, and downstream feature construction. Globally bad channels are excluded
from primary predictors rather than entered as interpolated synthetic channels; local transient
epoch artifacts are handled by the autoreject stage below.

Confirmatory analyses use a common-average reference excluding Fp1 and Fp2. A reference including
Fp1/Fp2 is reported as an artifact-sensitivity analysis.

ICA uses extended infomax (Lee et al., 1999) with 0.99 variance explained and is fitted on 1.0 Hz
high-pass-filtered epochs. ICA spatial weights are then applied to the 0.1-100 Hz continuous
analysis data. ICLabel (Pion-Tonachini et al., 2019) is used for component classification.
Components are rejected when their predicted probability exceeds 0.8 for any non-brain category
other than "other."

Because ICLabel was trained on standard non-fMRI EEG, retained components undergo a prespecified
artifact audit before target construction and model fitting. The audit is blind to NPS, SIIPS1,
ratings, and model outcomes. Components are rejected for volume-repetition spectral peak robust
z > 3, cardiac phase-locking above the 95th percentile of the within-subject circular-shift null,
or a high-frequency topographic artifact rule. The topographic rule requires 70-95 Hz component
power robust z > 3 plus either at least 50% absolute IC topographic mass on Fp1, Fp2, FT9, FT10,
TP9, and TP10, or within-subject Pearson r ≥ 0.50 between the component 70-95 Hz plateau envelope
and the Fp1/Fp2 artifact proxy. Audited components, rejection reasons, and thresholds are logged
before feature extraction.

The preprocessing sequence is fixed in this order.

1. Create preliminary epochs on a 1.0 Hz high-pass-filtered copy.
2. Fit ICA on these filtered epochs.
3. Apply ICA spatial weights to the 0.1-100 Hz continuous analysis data.
4. Extract final analysis epochs from −7.0 to 15.0 s relative to stimulus onset.
5. Apply autoreject (Jas et al., 2017) in local mode with candidate interpolation counts
   {4, 8, 16} for trial rejection.

Subject-specific electrode positions are digitized with EasyCap M1 channel labels and co-registered
to individual MRI. Confirmatory Study 2 source interpretation requires subject-specific digitization
passing quality control. Subject-level unsupervised EEG preprocessing is performed independently
within each subject before cross-validation.

### 5.2 Epochs, Baselines, and Frequency Bands

A −0.2 to 0.0 s pre-stimulus voltage baseline removes DC offset before ERP and amplitude-based
analyses. For time-frequency decompositions, the primary log-ratio baseline is −5.0 to −0.01 s,
chosen for stable alpha-band estimates under Morlet cycle requirements. Because this interval may
carry cue-locked expectancy activity, two sensitivity baselines, −0.2 to −0.01 s and −7.0 to
−5.5 s, are reported as robustness qualifiers; the latter is interpreted only when event logs
confirm no cue onset in that window. Sign changes or shifts in inferential status across baselines
limit baseline-robust interpretation.

Neural oscillations are operationalized as alpha (8.0–12.9 Hz), beta (13.0–30.0 Hz), and gamma
(30.1–80.0 Hz). Confirmatory frequency presets are alpha, beta, and alpha+beta. Gamma and
gamma-containing composites are exploratory because simultaneous EEG-fMRI gamma is vulnerable to
facial muscle activity, jaw tension, scanner residuals, and cardiac artifacts (Allen et al.,
1998, 2000; Muthukumaraswamy, 2013). The gamma band excludes a ± 1.0 Hz notch around 60 Hz due to
line-noise removal.

### 5.3 Fp1/Fp2 Frontal High-Frequency Artifact Proxy

Without independent facial EMG channels, artifact control relies on a prespecified Fp1/Fp2
high-frequency proxy and the dedicated ECG channel. Because Fp1/Fp2 are scalp EEG electrodes and
facial EMG can spread beyond them, the proxy serves as artifact control rather than independent
physiological validation, and anterior spatial patterns are interpreted accordingly.

The Fp1/Fp2 proxy is computed from the gradient- and BCG-corrected continuous signal after
downsampling, band-pass filtering, and notch filtering, but before PyPREP interpolation, ICA,
autoreject, or epoch-level rejection. For each plateau trial, Fp1 and Fp2 are refiltered to
70–95 Hz with the existing 60 Hz notch excluded. Hilbert power is averaged over 3.0–10.5 s and
across Fp1 and Fp2. The clean-events table stores this raw trial-level artifact-power covariate;
any fold-contained scaling needed for nuisance modeling occurs inside the modeling pipeline rather
than during preprocessing.

The HRF-weighted Fp1/Fp2 nuisance regressor is the canonical-HRF convolution of this trial series
sampled at each trial's fMRI plateau regressor peak. The unweighted proxy supports categorical
artifact censoring and continuous artifact-effect reporting. Fp1/Fp2 high-frequency power and
channels are excluded from confirmatory predictive features.

Required artifact covariates must be computable for at least 90% of otherwise retained plateau
trials. A subject is excluded from nuisance-controlled primary analysis when Fp1, Fp2, ECG, or event
synchronization is absent or invalid. Missing physiological artifact metrics are not imputed.

Artifact censoring thresholds use framewise displacement and DVARS metrics (Power et al., 2012):
framewise displacement > 0.5 mm, Fp1/Fp2 high-frequency power robust z > 3, DVARS robust z > 3,
cardiac phase-locking above the 95th percentile of the within-subject circular-shift null, and
scanner-frequency residual peaks robust z > 3. Continuous artifact-effect associations are
reported alongside categorical censoring. Gamma effects are labeled artifact-sensitive when
removing threshold-exceeding trials changes the effect direction or eliminates significance.

## 6. fMRI Signature Target Construction

Trial-wise fMRI effects are estimated with Least-Squares Separate (LSS) models (Mumford et al.,
2012) restricted to thermal plateau trials. For each eligible target trial, one GLM includes a
target-trial plateau regressor with onset at plateau start and duration equal to the plateau hold.
Other eligible plateau trials in the same acquisition run are modeled with the prespecified pooled
`other_trials` nuisance regressor (`lss_other_regressors: all`). Ramp-up, ramp-down, fixation, and
response epochs are modeled as non-plateau nuisance events when timing is available.

Primary LSS models use a canonical SPM hemodynamic response function (Friston et al., 1998), cosine
drift model, and 0.008 Hz high-pass filter without spatial smoothing. The denoising design includes
the 24-parameter rigid-body motion expansion, white-matter and CSF signals, framewise displacement,
CompCor regressors (Behzadi et al., 2007), and fMRIPrep motion-outlier regressors (Esteban et al.,
2019). Trials with framewise displacement > 0.5 mm or standardized DVARS robust z > 3 are ineligible
as target trials, but their thermal-event timing remains in the nuisance-event design when valid.
Target-trial loss is summarized by task block and used in the permutation-structure checks in
Section 10. A target GLM is ineligible if the target regressor is absent, duplicate-labeled,
nonestimable, or pushes the design condition number above 100.

HRF and timing robustness are assessed by repeating target construction with HRF temporal and
dispersion derivatives, with a finite-impulse-response model, and after shifting the EEG active
window by ± 2.0 s. Stability is summarized by three flags: stable primary incremental
$\Delta R^2_{\text{LOSO}}$ (|Δ(ΔR²)| < 0.02), retained significance, and positive Level 2
residualized-target convergence. Any failure limits HRF/timing-robust interpretation.

NPS and SIIPS1 maps are registered to MNI152NLin2009cAsym space (Fonov et al., 2009, 2011).
Signature assets are `NPS/weights_NSF_grouppred_cvpcr.nii.gz` and
`SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz`, resolved relative to the signature-map root.
SIIPS1 provenance is anchored to the CANlab Neuroimaging_Pattern_Masks repository. NPS provenance is
anchored to Wager et al. (2013) and the authorized CANlab NPS distribution or access record.
CanlabCore pattern-mask conventions define scoring (CANlab, n.d.). A frozen signature manifest
records source publication, source repository or access record, checksum, image geometry, and
weight support. Missing files, incomplete provenance, changed checksums after manifest freeze, or
mismatched target names make the affected signature ineligible.

When image grids differ, LSS beta maps are resampled to the signature-weight grid using continuous
interpolation. Published weight sign and scale are preserved. Weights are not normalized,
re-estimated, rescaled, thresholded, or sign-flipped using study data. For each signature and
target image, the scoring mask $V^{(k)}$ is the finite signature-weight grid intersected with the
corresponding fMRIPrep brain mask after nearest-neighbor resampling to the scoring grid. The target
table records a SHA-256 hash of the scoring-mask extent, including grid shape and affine, and fails
validity checks unless voxel count and scoring-mask extent are identical across retained subjects,
runs, and trials for the same signature.

A signature target fails validity checks if the scoring mask retains less than 90% of original
nonzero signature support, retains less than 90% of either positive- or negative-weight support, or
changes positive or negative total absolute weight mass by more than 10% after resampling. QC
summaries include retained positive and negative voxel counts, percentage support retained,
scoring-mask hash, and weight-distribution stability. Sensitivity analyses use the canonical
signature grid.

Primary LSS beta maps are not smoothed, z-scored, or trial-normalized before scoring. A fixed 4 mm
FWHM smoothing sensitivity repeats signature scoring after target-map construction. Signature
weights are not re-estimated from study data. Signature expression is:

$$y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v),$$

where $\beta_{s,i}(v)$ is the LSS-derived BOLD estimate for subject $s$, trial $i$, voxel $v$, and
$M_k(v)$ is the a priori weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$.

LSS diagnostics include retained trial counts, design efficiency and collinearity, and
temperature-stratified split-half reliability of NPS/SIIPS1 expression. For each target LSS GLM
with design matrix $X$ and unit contrast $c_{\text{target}}$ selecting the target regressor,
design efficiency is
$1/(c_{\text{target}}^\top (X^\top X)^+ c_{\text{target}})$ on the fitted Nilearn design matrix
(Abraham et al., 2014).
The 0.1 cutoff is a conservative numerical estimability gate for this study's LSS design, not a
literature-derived effect-size threshold. Subjects are excluded from confirmatory analyses if they
retain fewer than 25 plateau trials, have LSS design condition number above 100, or have LSS design
efficiency below 0.1. Subjects with 15-24 retained plateau trials are summarized in feasibility or
exploratory analyses.
Reliability-informed sensitivity analyses use split-half reliability r ≥ 0.4 and ≥ 30 plateau
trials.

Target reliability is computed separately for NPS and SIIPS1 from retained raw LSS signature
expression after acquisition and design-estimability exclusions and before EEG feature inspection.
A fixed random seed produces 1,000 split-half partitions stratified within subject and stimulus
temperature. Each subject-by-temperature cell contributes the mean expression from half A and
half B, and the reliability statistic is the median Spearman-Brown-corrected (Brown, 1910;
Spearman, 1910) Pearson correlation across valid split-half vectors. A split is valid when both
halves contain finite values for every included subject-by-temperature cell. The reliability
estimate is valid only when every included subject has at least 30 retained plateau trials for that
target after acquisition and design-estimability exclusions. A target is labeled
target-reliability-limited when split-half reliability falls below r = 0.4, the 30-trial floor is
not met, or the 1,000 valid stratified splits cannot be generated. For a reliability-limited
target, significant prediction is reported as out-of-sample prediction of the measured target only
and does not support strong pain-signature interpretation or Study 2 source entry.

## 7. EEG Feature Construction

Spectral power features are extracted with Morlet wavelets (Cohen, 2014) using frequency-adaptive
cycle counts ($\text{cycles}=f/2.0$, bounded between 3.0 and 15.0) and decimation factor 4. The
primary precomputed benchmark uses baseline-normalized total power without evoked-response
subtraction: a condition-agnostic ERP template estimated across all trials would leak held-out
subjects, while a fold-level template would require CV-owned signal extraction and fold-specific
feature matrices. ERP-subtracted power is eligible only as a separate sensitivity analysis when
extraction is fold-owned and provenance records the training-fold template.

Spectral power is log-ratio baseline-corrected using the primary baseline and averaged within
3.0–10.5 s. Each retained trial contributes one power value per channel and band. The primary
confirmatory matrix uses active-window individual-channel log-ratio columns only. ROI-level and
global-average matrices are spatial-resolution sensitivities. All confirmatory feature matrices
exclude Fp1 and Fp2. Matrices retaining Fp1/Fp2 are exploratory artifact-sensitivity analyses.

The ROI feature matrix uses fixed, non-overlapping scalp groups resolved from normalized extended
10-20 channel names before model fitting. Fp1 and Fp2 are excluded. A ROI is included only when at
least two listed channels remain after preprocessing. ROI features are arithmetic means of
channel-level log-ratio power within each ROI and frequency preset.

Exploratory feature families include spectral peaks, aperiodic slope and offset (specparam fixed
model; Donoghue et al., 2020), event-related desynchronization/synchronization, band-power ratios,
hemispheric alpha asymmetry, nonlinear complexity, and oscillatory burst statistics. Each family is
labeled and Holm-corrected across tested target-model-frequency cells.

## 8. Nuisance Structure and Residualization Levels

**Level 1 - Raw expression.** Models predict full NPS/SIIPS1 dot-product expression, which can mix
stimulus intensity, condition, temporal structure, subjective experience, and neural pain
processing.

**Level 2 - Stimulus- and acquisition-controlled expression.** Targets are residualized against an
intercept, stimulus temperature, task-block index, trial onset time, within-block trial number,
selected thermode surface, HRF-weighted framewise displacement, HRF-weighted standardized DVARS,
HRF-weighted Fp1/Fp2 high-frequency artifact power, and residual ECG coupling from the dedicated
ECG channel. Stimulus temperature and selected thermode surface enter as categorical regressors. A
continuous nonlinear temperature basis is retained only as a sensitivity analysis. Binary pain
condition is excluded because it recodes the thermal manipulation.

Level 2 tests whether EEG predicts fMRI signature expression beyond prespecified stimulus and
acquisition structure, while recognizing that this control can remove construct-relevant evoked-pain
variance.

**Level 3 - Rating-residualized sensitivity.** The Level 2 design is augmented with the binary pain
report and within-scale thermal/pain intensity score, not the raw discontinuous 0 to 200 displayed
rating. This level is interpreted as a construct-attenuation sensitivity analysis.

The primary Level 2 design is fixed across LOSO folds. Before SVD fitting, rank is checked from the
centered and scaled training-fold nuisance matrix. The design is full rank only when every
non-intercept singular value satisfies σ_j / σ_max ≥ 10⁻¹⁰. Cells with rank-deficient
Level 2 designs in any outer training fold are ineligible for confirmatory interpretation. The
continuous nonlinear temperature basis remains a sensitivity analysis.

Within each level, nuisance coefficients are estimated exclusively on training subjects using
SVD-based least squares.

## 9. Predictive Modeling

### 9.1 Primary Incremental Model

The benchmark predicts raw NPS and SIIPS1 expression with a nuisance-only model and a combined
nuisance-plus-EEG model; the primary thesis gate remains the prespecified NPS ElasticNet
alpha+beta individual-channel cell. The nuisance-only model uses unpenalized ordinary least squares
with the same rank-stable nuisance design in every fold. The same pre-SVD rank tolerance used for
Level 2 applies here. If the nuisance-only design becomes rank deficient, the affected cell is
ineligible for confirmatory interpretation.

The nuisance-plus-EEG estimator is staged residual learning rather than a joint penalized
regression. The nuisance component is unpenalized ordinary least squares fit on the raw target
scale. Within each fold, the nuisance model, residual-target transformation, feature
residualization, standardization, imputation, and EEG model are learned from training subjects
only. The EEG model is fit to the Yeo-Johnson-transformed training residuals from the raw-scale
nuisance model. Held-out EEG residual predictions are inverse-transformed back to the raw
residual scale and then added to the held-out raw-scale nuisance prediction before scoring.

### 9.2 Feature-Based Models

A nested LOSO framework is used. The primary confirmatory model is ElasticNet regression
(Zou and Hastie, 2005) on individual-channel spectral power. Ridge (Hoerl and Kennard, 1970) is a
secondary confirmatory linear model that supports Haufe-style forward-pattern sensitivity analyses
in Study 2 (Haufe et al., 2014). Random Forest is an exploratory nonlinear model
(Breiman, 2001).

Feature preprocessing is fold-contained: feature statistics, imputation medians, variance
thresholds, standardization means, and standard deviations are estimated from training subjects
only. Imputation is limited to isolated nonfatal missing feature values within otherwise valid
trials and features. Missing fMRI targets, EEG-fMRI synchronization, event identifiers, required
channels, artifact metrics, frequency-band extraction, or entire trials are exclusion or
analysis-failure events and are not imputed.

For imputation-eligible feature values, the missingness rate must be below 5% for each feature
within the outer training fold and below 10% for each retained subject after feature filtering.
Features exceeding the feature-level limit are removed within the training fold. Values that remain
imputation-eligible are imputed with the corresponding training-cohort median for that feature.
Exceeding the subject-level limit, or removing all features, makes the affected cell ineligible for
confirmatory interpretation. Retained features are standardized to zero mean and unit variance, with
constant features removed.

The Yeo-Johnson transformation (Yeo and Johnson, 2000) is applied in every confirmatory ElasticNet
and Ridge cell, only to the target component learned by the penalized EEG model. In the primary
incremental analysis this is $r_{\mathrm{train}}$. The nuisance-only prediction remains on the raw
target scale. In secondary residualized-target models, targets are residualized before
fold-contained transformation.
Predictions are inverse-transformed before primary metrics are reported. Analyses without target
transformation are sensitivities.

Hyperparameters are tuned with 5-fold inner GroupKFold cross-validation restricted to training
subjects. In the staged primary incremental model, inner selection uses subject-weighted $R^2$ in
the transformed residual-target space learned inside the outer training fold; final LOSO inference
uses raw-scale subject-weighted $\Delta R^2_{\text{LOSO}}$ after inverse transformation and
nuisance prediction add-back. ElasticNet uses $\rho \in \{0.2, 0.5, 0.8\}$ and the configured
$\alpha \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$ grid. ElasticNet uses 10,000 maximum iterations.
Ridge uses $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Random Forest uses 500 estimators with
max depths $\in \{5, 10, 20, \text{None}\}$, min samples split $\in \{2, 5, 10\}$, and min
samples leaf $\in \{1, 2, 4\}$.

Confirmatory inference requires at least 30 analyzable subjects. A preregistered precision
simulation quantifies how precisely the retained sample can estimate the primary $\Delta R^2$
statistic; it complements rather than replaces the subject-count requirement. Inputs are the
observed subject count, retained-trial counts, task-block structure, temperature sequence,
nuisance matrix, and target split-half reliability, all estimated without EEG prediction outcomes.

The simulation uses 10,000 Monte Carlo datasets per retained-sample scenario. Subject-level random
effects, within-block autocorrelation, and trial-wise residual variance are matched to nuisance-only
target residuals. Simulated EEG residual-prediction effects span
$\Delta R^2 \in \{0.000, 0.005, 0.010, 0.020, 0.050\}$. For each dataset, the subject-weighted
$\Delta R^2_{\text{LOSO}}$ confidence interval is computed with a percentile bootstrap over
subjects using 10,000 resamples. The precision flag passes when the 95th percentile of simulated
95% CI half-widths is ≤ 0.10; otherwise the result is labeled precision-limited.

### 9.3 Exploratory Deep Regression Model

The BandTemporalRegressor is exploratory. It uses band-limited Hilbert-power tensors cropped to
3.0–10.5 s, training-cohort standardization, band-specific spatial filtering, temporal
integration, dropout, AdamW optimization (Loshchilov and Hutter, 2019), and subject-held-out
validation. Its results do not enter the confirmatory family.

## 10. Statistical Inference

Primary metrics are computed on the original target scale after inverse-transforming predictions
and, for the primary incremental analysis, adding back the held-out nuisance prediction. The
coefficient of determination uses the training-fold target mean as the zero-skill baseline.

$$
R_f^2 =
1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}
{\sum_{i \in f}(y_i - \bar{y}_{\mathrm{train},f})^2}.
$$

Mean $R^2$ for the nuisance-plus-EEG model and pooled trial-wise $R^2$ are reported descriptively.
Subject-wise metrics use configured percentile bootstrap confidence intervals. Group-level
intervals resample subjects; within-subject intervals use task-block-level resampling or circular
block bootstrap.

Reports decompose held-out predictions and targets into subject means and within-subject
deviations. The primary Study 1 claim covers out-of-sample prediction of fMRI pain-signature
expression in held-out subjects during this simultaneous EEG-fMRI thermal protocol; it does not
extend to standalone EEG pain biomarkers, clinical pain detection, subjective pain decoding, or
causal neural-generator inference. Within-person trial-tracking interpretation additionally
requires positive within-subject-centered diagnostic $\Delta R^2_{\text{LOSO}}$ for the same
target-model-frequency cell.

Primary inference uses nonparametric permutation testing (Winkler et al., 2014) with 5,000 valid
permutations and a one-sided upper-tail p-value for positive $\Delta R^2_{\text{LOSO}}$. The
primary null repeats the full observed-analysis training procedure, including fold-level
preprocessing statistics, imputation, constant-feature filtering, target transformation, and inner
GroupKFold hyperparameter selection. Frozen-hyperparameter permutations are computational
sensitivities only.

For nuisance-controlled cells, the primary null is a fold-contained reduced-model residual
permutation. Within each outer fold, the nuisance-only model is fit on the original training
subjects, nuisance predictions are held fixed for the training and held-out rows in that fold, and
the fold's nuisance residuals are circular-shifted within subject and task block. Permuted targets
are reconstructed as nuisance prediction plus shifted residual, and the full nuisance-plus-EEG
pipeline is then refit. This tests whether EEG residual information adds prediction beyond the
prespecified nuisance structure while preserving the fold-specific nuisance-target relationship.

Circular shifts use the six 11-trial task blocks as exchangeability units, retaining censored blocks
when they still support a valid circular shift. Within each block, trials are ordered by the
original trial-order label after censoring; censored trials are not imputed. A permutation-valid
block must retain at least 8 plateau trials and allow at least four distinct nonzero circular
shifts after excluding shifts shorter than 5 original trial positions. Blocks failing these rules
are excluded before confirmatory model fitting. A subject fails confirmatory prediction analysis if
fewer than three permutation-valid task blocks or fewer than 25 retained plateau trials remain. Each
permutation refits the nuisance-only and nuisance-plus-EEG models, including inner-fold
hyperparameter selection.

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
outcome inspection. Confirmatory inference requires 5,000 valid draws. Each
target-model-frequency cell attempts at most 6,250 draws, corresponding to the prespecified maximum
invalid-draw fraction of 20%. If 5,000 valid draws are not obtained within those 6,250 attempts, the
affected target-model-frequency cell is downgraded to exploratory. The null size is not reduced
based on interim results.

Source-entry diagnostics are required for Study 2 claim-tier assignment. Missing target reliability,
reliability trial count, Level 2 convergence, within-subject-centered prediction,
temporal-negative-control, or artifact-robustness diagnostics set Study 2 source-entry status to
not evaluated. Missing precision, HRF/timing, first-exposure, baseline, or smoothing diagnostics
disable the corresponding robustness language.

## 11. Validity and Sensitivity Analyses

### 11.1 Nuisance Prediction Reporting

Supplementary models predict each nuisance variable and behavioral report variable from the same EEG
features. Behavioral report variables are the binary pain report and within-scale thermal/pain
intensity score. Nuisance-prediction p-values are Holm-corrected across supplementary targets and
reported separately from primary incremental $\Delta R^2_{\text{LOSO}}$.

### 11.2 Temporal Negative Controls

Temporal negative controls use the same nuisance-only versus nuisance-plus-EEG
$\Delta R^2_{\text{LOSO}}$ framework. Models trained on EEG features from −5.0 to 0.0 s and
−0.2 to 0.0 s predict post-stimulus target expression. These controls use raw log-power summaries
with no TFR baseline correction (`feature_baseline_window: null`), so the pre-stimulus windows are
not made circular by reusing the active-window −5.0 to −0.01 s baseline.

Negative controls are evaluated separately for NPS and SIIPS1. A target-model-frequency cell is
temporally specific when both pre-stimulus models are nonsignificant after Holm correction across
the two windows within that cell and the one-sided 95% upper confidence bound for pre-stimulus
$\Delta R^2$ falls below 0.02 and below 25% of the observed active-window $\Delta R^2$. Failure
of this flag limits temporally specific interpretation.

Wrong-lag windows are ramp-up ($0.0$-$3.0$ s), late ramp-down ($10.5$-$15.0$ s), early-shifted
active ($1.0$-$8.5$ s), and late-shifted active ($5.0$-$12.5$ s). Wrong-lag p-values are
Holm-corrected across the four windows within each target-model-frequency cell. A cell is labeled
wrong-lag robust when no wrong-lag window exceeds active-window $\Delta R^2$ and no Holm-corrected
significant wrong-lag window has $\Delta R^2$ at least 75% of the active-window
$\Delta R^2$. The primary temporal-negative control analysis repeats full inner GroupKFold
hyperparameter selection for every pre-stimulus and wrong-lag window. Active-window hyperparameters
are reused only in a secondary frozen-model sensitivity analysis.

## References

Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J.,
Gramfort, A., Thirion, B., & Varoquaux, G. (2014). Machine learning for neuroimaging
with scikit-learn. Frontiers in Neuroinformatics, 8, 14. doi.org/10.3389/fninf.2014.00014

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from
continuous EEG recorded during functional MRI. NeuroImage, 12(2), 230-239.
doi.org/10.1006/nimg.2000.0599

Allen, P. J., Polizzi, G., Krakow, K., Fish, D. R., & Lemieux, L. (1998). Identification of EEG
events in the MR scanner. The problem of pulse artifact and a method for its subtraction.
NeuroImage, 8(3), 229-239. doi.org/10.1006/nimg.1998.0361

Appelhoff, S., Sanderson, M., Brooks, T. L., van Vliet, M., Quentin, R., Holdgraf, C.,
Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C.,
Rockhill, A. P., Larson, E., Gramfort, A., & Jas, M. (2019). MNE-BIDS. Organizing
electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open
Source Software, 4(44), 1896. doi.org/10.21105/joss.01896

Behzadi, Y., Restom, K., Liau, J., & Liu, T. T. (2007). A component based noise correction method
(CompCor) for BOLD and perfusion based fMRI. NeuroImage, 37(1), 90-101.
doi.org/10.1016/j.neuroimage.2007.04.042

Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline.
Standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16.
doi.org/10.3389/fninf.2015.00016

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
doi.org/10.1023/A:1010933404324

Brown, W. (1910). Some experimental results in the correlation of mental abilities. British Journal
of Psychology, 3(3), 296-322. doi.org/10.1111/j.2044-8295.1910.tb00207.x

Buysse, D. J. (2014). Sleep health. Can we define it? Does it matter? Sleep, 37(1), 9-17.
doi.org/10.5665/sleep.3298

CANlab. (n.d.). CanlabCore documentation. Retrieved May 17, 2026, from
https://canlabcore.readthedocs.io/

Coghill, R. C., McHaffie, J. G., & Yen, Y.-F. (2003). Neural correlates of interindividual
differences in the subjective experience of pain. Proceedings of the National Academy of Sciences
of the United States of America, 100(14), 8538-8542. doi.org/10.1073/pnas.1430684100

Cohen, M. X. (2014). Analyzing neural time series data. Theory and practice. MIT Press.

Davis, K. D., Aghaeepour, N., Ahn, A. H., Angst, M. S., Borsook, D., Brenton, A., Burczynski,
M. E., Crean, C., Edwards, R., Gaudilliere, B., Hergenroeder, G. W., Iadarola, M. J., Iyengar, S.,
Jiang, Y., Kong, J.-T., Mackey, S., Saab, C. Y., Sang, C. N., Scholz, J., ... Pelleymounter, M. A.
(2020). Discovery and validation of biomarkers to aid the development of safe and effective pain
therapeutics. Challenges and opportunities. Nature Reviews Neurology, 16(7), 381-400.
doi.org/10.1038/s41582-020-0362-2

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T.,
Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing
neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23(12),
1655-1665. doi.org/10.1038/s41593-020-00744-x

Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A.,
Kent, J. D., Goncalves, M., DuPre, E., Snyder, M., Oya, H., Ghosh, S. S., Wright, J.,
Durnez, J., Poldrack, R. A., & Gorgolewski, K. J. (2019). fMRIPrep. A robust preprocessing
pipeline for functional MRI. Nature Methods, 16(1), 111-116. doi.org/10.1038/s41592-018-0235-4

Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L. (2009). Unbiased
nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage,
47(Suppl. 1), S102. doi.org/10.1016/S1053-8119(09)70884-5

Fonov, V. S., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C., & Collins, D. L. (2011).
Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54(1), 313-327.
doi.org/10.1016/j.neuroimage.2010.07.033

Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998).
Event-related fMRI. Characterizing differential responses. NeuroImage, 7(1), 30-40.
doi.org/10.1006/nimg.1997.0306

Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P.,
Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M.,
Keator, D., Li, X., Michael, Z., Maumet, C., Nichols, B. N., Nichols, T. E., Pellman, J.,
... Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and
describing outputs of neuroimaging experiments. Scientific Data, 3, 160044.
doi.org/10.1038/sdata.2016.44

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F.
(2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging.
NeuroImage, 87, 96-110. doi.org/10.1016/j.neuroimage.2013.10.067

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression. Biased estimation for nonorthogonal
problems. Technometrics, 12(1), 55-67. doi.org/10.1080/00401706.1970.10488634

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of
Statistics, 6(2), 65-70.

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject. Automated
artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
doi.org/10.1016/j.neuroimage.2017.06.030

Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A.
(2018). A reproducible MEG/EEG group study with the MNE software. Recommendations, quality
assessments, and good practices. Frontiers in Neuroscience, 12, 530.
doi.org/10.3389/fnins.2018.00530

Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation.
Electroencephalography and Clinical Neurophysiology, 10, 371-375.

Kim, H., Neubert, J. K., San Miguel, A., Xu, K., Krishnaraju, R. K., Iadarola, M. J., Goldman, D.,
& Dionne, R. A. (2004). Genetic influence on variability in human acute experimental pain
sensitivity associated with gender, ethnicity and psychological temperament. Pain, 109(3), 488-496.
doi.org/10.1016/j.pain.2004.02.027

Kim, J. A., & Davis, K. D. (2021). Neural oscillations. Understanding a neural code of pain. The
Neuroscientist, 27(5), 544-570. doi.org/10.1177/1073858420958629

Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9. Validity of a brief
depression severity measure. Journal of General Internal Medicine, 16(9), 606-613.
doi.org/10.1046/j.1525-1497.2001.016009606.x

Lee, T.-W., Girolami, M., & Sejnowski, T. J. (1999). Independent component analysis using an
extended infomax algorithm for mixed sub-Gaussian and super-Gaussian sources. Neural Computation,
11(2), 417-441. doi.org/10.1162/089976699300016719

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International
Conference on Learning Representations. https://openreview.net/forum?id=Bkg6RiCqY7

Mari, T., Henderson, J., Maden, M., Nevitt, S. J., Duarte, R., & Fallon, N. (2022). Systematic
review of the effectiveness of machine learning algorithms for classifying pain intensity,
phenotype or treatment outcomes using electroencephalogram data. The Journal of Pain, 23(3),
349-369. doi.org/10.1016/j.jpain.2021.07.011

Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012). Deconvolving BOLD activation
in event-related designs for multivoxel pattern classification analyses. NeuroImage, 59(3),
2636-2643. doi.org/10.1016/j.neuroimage.2011.08.076

Muthukumaraswamy, S. D. (2013). High-frequency brain activity and muscle artifacts in MEG/EEG.
A review and recommendations. Frontiers in Human Neuroscience, 7, 138.
doi.org/10.3389/fnhum.2013.00138

Oldfield, R. C. (1971). The assessment and analysis of handedness. The Edinburgh inventory.
Neuropsychologia, 9(1), 97-113. doi.org/10.1016/0028-3932(71)90067-4

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel. An automated
electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198,
181-197. doi.org/10.1016/j.neuroimage.2019.05.026

Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). Spurious
but systematic correlations in functional connectivity MRI networks arise from subject motion.
NeuroImage, 59(3), 2142-2154. doi.org/10.1016/j.neuroimage.2011.10.018

Robinson, M. E., Riley, J. L. III, Myers, C. D., Papas, R. K., Wise, E. A., Waxenberg, L. B.,
& Fillingim, R. B. (2001). Gender role expectations of pain. Relationship to sex differences in
pain. The Journal of Pain, 2(5), 251-257. doi.org/10.1054/jpai.2001.24551

Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 3(3),
271-295. doi.org/10.1111/j.2044-8295.1910.tb00206.x

Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). A brief measure for assessing
generalized anxiety disorder. The GAD-7. Archives of Internal Medicine, 166(10), 1092-1097.
doi.org/10.1001/archinte.166.10.1092

Sullivan, M. J. L., Bishop, S. R., & Pivik, J. (1995). The Pain Catastrophizing Scale.
Development and validation. Psychological Assessment, 7(4), 524-532.
doi.org/10.1037/1040-3590.7.4.524

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

Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or
symmetry. Biometrika, 87(4), 954-959. doi.org/10.1093/biomet/87.4.954

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of
the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320.
doi.org/10.1111/j.1467-9868.2005.00503.x
