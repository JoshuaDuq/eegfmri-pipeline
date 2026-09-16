# Study 1: Electroencephalographic Prediction of Trial-Wise Functional Magnetic Resonance Imaging Pain-Signature Expression

Operational execution instructions, cluster commands, smoke test procedures, and output schemas are documented in the Study 1 Run Guide: [RUN_GUIDE.md](RUN_GUIDE.md). This document provides the formal scientific protocol, mathematical estimands, preprocessing rationale, machine learning architecture, inferential framework, and methodological interpretation guidelines for Study 1.

## 1. Problem Statement

Pain perception exhibits substantial variability among individuals exposed to physically identical nociceptive stimuli (Coghill et al., 2003; Kim et al., 2004). Electroencephalography captures rapid oscillatory dynamics during thermal nociception (Kim and Davis, 2021; Mari et al., 2022), but scalp-recorded spectral power in alpha, beta, and gamma bands remains spatially ambiguous and susceptible to non-nociceptive contamination. Observed scalp power variations can reflect aversion, motor preparation, cranial or facial muscle contraction, ballistocardiographic artifact, magnetic resonance scanner interference, or cognitive expectation (Allen et al., 1998, 2000; Muthukumaraswamy, 2013).

Simultaneous functional magnetic resonance imaging provides anatomically resolved spatial targets to validate electrophysiological markers of nociception (Davis et al., 2020; van der Miesen et al., 2019). Multivariate brain signatures provide validated operational targets for pain-related neural activity. The Neurologic Pain Signature indexes stimulus-evoked nociceptive pain expression across distributed brain systems (Wager et al., 2013). The Stimulus Intensity Independent Pain Signature-1 captures pain-related variance that is less dependent on objective stimulus intensity (Woo et al., 2017). Neither signature represents an exhaustive or purely unifactorial biological separation of physical from psychological pain, and their expression does not establish clinical biomarkers without prospective empirical testing. They serve as complementary multivariate targets to evaluate whether electrophysiological oscillations track central pain-related representations.

## 2. Objectives and Prespecified Gates

The primary objective is to test whether plateau-window electroencephalographic spectral power across alpha, beta, and gamma frequency bands predicts trial-wise expression of the Neurologic Pain Signature during simultaneous thermal stimulation beyond measured stimulus, acquisition, and physiological nuisance structure. The primary estimand is out-of-sample incremental prediction in held-out participants. Secondary objectives evaluate the same staged incremental framework applied to the Stimulus Intensity Independent Pain Signature-1, with expression of the Neurologic Pain Signature incorporated into the secondary nuisance model to isolate incremental variance.

Target quality control metrics are evaluated before interpreting predictive models. These metrics are descriptive diagnostics documenting target behavior rather than automated algorithmic gates. The prespecified primary confirmatory gate is the prediction of Neurologic Pain Signature expression using ElasticNet regression trained on individual-channel spectral power across alpha, beta, and gamma bands. A secondary frequency audit evaluates models restricted to alpha, beta, gamma, and alpha+beta bands following the primary test. Models trained on delta, theta, delta+theta, and all-band feature matrices are evaluated as exploratory broad-band audits rather than confirmatory primary presets.

## 3. Hypotheses and Confirmatory Estimands

The primary confirmatory hypothesis posits that electroencephalographic oscillatory features provide out-of-sample predictive information for trial-wise pain-signature expression beyond a comprehensive nuisance-only baseline:

$$
\Delta R^2_{\text{LOSO}} = R^2_{\text{nuisance+EEG}} - R^2_{\text{nuisance-only}} > 0.
$$

The primary confirmatory estimand is the subject-weighted mean incremental prediction across held-out participants:

$$
\Delta R^2_{\text{LOSO}} = \frac{1}{S}\sum_{s=1}^{S} \Delta R_s^2,
$$

where $S$ denotes the total count of included participants and $\Delta R_s^2$ represents the difference in predictive accuracy for participant $s$. The primary confirmatory gate supports the hypothesis when the cohort-level subject-weighted increment is strictly positive and the associated one-sided upper-tail permutation probability value satisfies $p \le 0.05$. Effect sizes and ninety-five percent bootstrap confidence intervals accompany the primary permutation test.

The primary estimand addresses whether electrophysiological activity enhances out-of-sample prediction of signature expression in participants unobserved during model training. Because the Level 2 nuisance model removes pooled categorical temperature effects while preserving participant-specific temperature slopes, a model can satisfy this estimand by inferring that an unobserved participant exhibits elevated overall expression or a steeper temperature response curve, without necessarily predicting trial-by-trial fluctuations across repetitions of the same stimulus temperature. Evaluating trial-by-trial fluctuations at fixed stimulus temperatures represents a separate within-condition endpoint and must not be inferred from the primary pooled estimand. Neither estimand demonstrates causal neurobiological mechanisms, subjective pain prediction, or clinical biomarker utility.

The secondary spectral hypothesis posits that predictive information is distributed across canonical oscillatory bands, including gamma-band power. High-frequency scalp power requires cautious interpretation because scalp gamma can reflect cranial myogenic activity, cognitive salience, ocular micro-saccades, and residual acquisition artifacts in addition to cortical oscillations. Models incorporating gamma features support a neural signature interpretation only when predictive effects survive prespecified physiological nuisance regression, ocular channel exclusion, artifact diagnostics, and empirical sensitivity checks.

Secondary confirmatory analyses evaluate a factorial grid crossing the two signature targets with two regularized linear estimators (ElasticNet and Ridge regression) across five spectral presets (alpha, beta, gamma, alpha+beta, and alpha+beta+gamma) using the individual-channel feature matrix. Delta, theta, delta+theta, and all-band specifications provide exploratory low-frequency and broad-band comparisons. Holm correction controls family-wise error across the secondary confirmatory grid (Holm, 1979). Region-of-interest aggregations, global-average power, subjective rating adjustment, Random Forest regression, and deep neural network models are designated as exploratory analyses.

## 4. Study Design and Data Scope

### 4.1 Participants and Clinical Characterization

The planned cohort comprises sixty healthy adult participants, balanced equally between female and male participants (thirty women and thirty men). Eligible individuals are between eighteen and fifty years of age, right-handed according to the adapted Edinburgh Handedness Inventory with a laterality quotient exceeding positive forty (Oldfield, 1971), and present normal or corrected-to-normal vision. Exclusion criteria comprise current or past chronic pain disorders, diagnosed neurological or psychiatric conditions, pregnancy or active lactation, pharmaceutical treatments altering central vigilance or attention, contraindications to magnetic resonance imaging, and hair textures or styles incompatible with high-density scalp electroencephalography.

Participants complete an experimental protocol lasting approximately one hundred and eighty minutes at the CERVO neuroimaging facility. The session encompasses electroencephalographic sensor montage preparation, task familiarization, high-resolution anatomical structural imaging, resting-state functional acquisitions, and the simultaneous electroencephalography and functional magnetic resonance imaging thermal pain paradigm.

Self-report instruments characterize individual psychological, affective, sleep, and pain-related traits. These comprise demographic and hormonal status surveys, the Patient Health Questionnaire-9 (Kroenke et al., 2001), the Generalized Anxiety Disorder-7 scale (Spitzer et al., 2006), the RU-SATED sleep health questionnaire (Buysse, 2014), the adapted Edinburgh Handedness Inventory (Oldfield, 1971), the Gender Role Expectation of Pain questionnaire (Robinson et al., 2001), and the Pain Catastrophizing Scale (Sullivan et al., 1995). Exploratory moderation analyses investigate interaction effects with out-of-sample electrophysiological residual predictions, evaluated via linear mixed-effects models with Holm correction across candidate moderators. The primary confirmatory analysis requires at least thirty analyzable participants following electroencephalographic, functional imaging, synchronization, and artifact exclusions.

### 4.2 Thermal Stimulation Protocol and Behavioral Metrics

Contact thermal stimulation is administered to the volar surface of the forearm contralateral to the behavioral response hand using an MRI-compatible QST.Lab T11 thermode featuring five independently controlled stimulation zones spanning nine square centimeters. Fixed stimulus temperatures are administered across all participants without individual calibration to pain thresholds. Objective stimulus temperature, subjective ratings, and acquisition nuisance terms are modeled as separate covariates.

Prior to scanning, participants complete two practice trials to verify task comprehension and confirm tolerability of the maximal planned temperature. Individuals unable to tolerate the maximum stimulus level do not proceed to scanner testing.

The functional imaging task comprises six distinct task runs consisting of eleven trials each, yielding sixty-six total thermal trials per participant. Six target temperatures ranging from 44.3 to 49.3 degrees Celsius are administered eleven times across the session. Trial sequences are generated through constrained pseudo-randomization ensuring that consecutive stimuli never engage the same thermode contact surface, identical temperatures are never presented consecutively, and first-order transition frequencies between temperatures remain balanced across the session. The initial trial of the first task run administers 49.3 degrees Celsius as a protocol-fixed initial exposure. When raw neuroimaging files use separate acquisition run labels, those labels are preserved as metadata while confirmatory trial ordering and permutation units rely strictly on the synchronized task-run index. Event-level trial order is indexed from trial number when available or trial index otherwise, and censored trials retain their original ordinal identifiers without renumbering.

All quality-controlled thermal trials are retained for primary predictive modeling. Stimulus temperature enters the Level 2 nuisance regression model, while subjective binary reports and continuous ratings serve as behavioral validation criteria. Because the Stimulus Intensity Independent Pain Signature-1 was initially developed on painful trials, secondary sensitivity analyses repeat estimation restricted to painful trials whenever the dataset combines painful and non-painful trials.

The deterministic initial high-temperature exposure is accommodated in the nuisance model via trial onset time, within-run trial position, and task-run index. Sensitivity analyses repeat the primary model after excluding the initial exposure trial and, separately, after omitting the entire initial task run. Trial-history sensitivity models augment the nuisance design with the preceding stimulus temperature, signed temperature change between trials, cumulative stimulus count, and previous within-scale ratings.

Individual trials initiate with a variable fifteen to twenty second fixation period while the thermode maintains an adaptation temperature of 35.0 degrees Celsius. Thermal stimulation spans 12.5 seconds, comprising a three-second linear ramp-up, a 7.5-second plateau at the target temperature, and a two-second return to baseline. Following a post-stimulus fixation interval lasting between 4.5 and 8.5 seconds, participants indicate via a binary prompt whether the stimulus was perceived as painful, followed by a continuous rating on a visual analogue scale. Non-painful sensations are rated on a warmth-intensity scale spanning zero to ninety-nine, whereas painful sensations are rated on a pain-intensity scale spanning one hundred to two hundred. Behavioral data are logged as the binary report, the raw display rating, and a standardized within-scale intensity score ranging from zero to one hundred. For non-painful trials, the within-scale score preserves the raw warmth rating; for painful trials, one hundred is subtracted from the raw rating. Behavioral responses are recorded using an MRI-compatible Pyka five-button response unit, and visual stimuli are presented via an MRI-compatible projection system viewed through a head-coil mirror.

Trials displaying implausible behavioral responses are excluded prior to electrophysiological or hemodynamic modeling. A trial is classified as implausible if the raw rating lies outside valid scale bounds or equals zero (indicating no sensation) at stimulus temperatures of 47.3 degrees Celsius or greater. Participants displaying greater than ten percent implausible trials among synchronized thermal exposures are excluded from analysis.

### 4.3 Simultaneous Acquisition Parameters

Continuous electroencephalography is recorded using a sixty-four-channel BrainCap MR cap and BrainAmp MR Plus amplifiers sampled at 5,000 Hz prior to scanner artifact correction and downsampling. Passive silver/silver-chloride electrodes are positioned according to the extended ten-twenty system (Jasper, 1958). Hardware filters are set to 0.1–100 Hz, and electrode impedances are maintained below twenty kilo-ohms, with target values below ten kilo-ohms for the majority of channels. A dedicated electrocardiogram lead monitors cardiac activity for ballistocardiographic artifact correction and physiological covariate construction. Timing synchronization between recording systems is achieved through hardware volume markers transmitted by the magnetic resonance console.

Functional magnetic resonance imaging is conducted on a Siemens MAGNETOM Prisma 3 Tesla scanner using a multiband gradient-echo echo-planar imaging pulse sequence with a repetition time of 900 ms, echo time of 20 ms, multiband acceleration factor of three, isotropic three-millimeter voxels, and fifty-four axial slices. High-resolution anatomical images are obtained via a T1-weighted MP-RAGE sequence with one-millimeter isotropic resolution. A ten-minute resting-state functional acquisition and spin-echo field maps support anatomical registration, baseline functional connectivity characterization, and geometric distortion correction.

### 4.4 Multimodal Trial Synchronization and Modeling Scope

Analysis epochs span −7.0 to 15.0 seconds relative to stimulus onset. The active electroencephalographic prediction window comprises the 7.5-second thermal plateau interval spanning 3.0 to 10.5 seconds post-stimulus onset, which aligns with single-trial Least-Squares Separate hemodynamic estimates targeted to the identical plateau duration.

Multimodal data integration links functional magnetic resonance imaging target estimates to clean electroencephalographic trials using unified subject, task-run, and trial identifiers. Physical trigger onsets in the electrophysiological recording and nominal plateau onsets in the functional imaging protocol are compared after adjusting for the three-second ramp-up interval. Retained trials must exhibit an absolute temporal discrepancy not exceeding 0.010 seconds.

## 5. Electroencephalographic Preprocessing and Quality Control

### 5.1 Artifact Correction and Independent Component Analysis

Magnetic resonance scanner gradient artifacts are eliminated in BrainVision Analyzer 2.3 using continuous-mode sliding-window average artifact subtraction (Allen et al., 2000) locked to volume acquisition markers with zero millisecond offset. Subtraction templates are computed across twenty-one consecutive artifact intervals using the complete interval for baseline correction, alongside channel-specific bad interval replacement across all sixty-four channels. Following gradient subtraction, the continuous signal is downsampled to 1,000 Hz and filtered with a 100 Hz low-pass infinite impulse response filter (twenty-four decibels per octave attenuation).

Cardioballistic pulse artifacts are corrected via average artifact subtraction locked to electrocardiographic R-peaks (Allen et al., 1998). Pulse intervals are detected automatically within a physiologically permitted range of forty-five to eighty beats per minute using a 0.6 coherence threshold. Pulse templates are estimated across twenty-one cardiac cycles and subtracted from scalp channels prior to export into MNE-Python.

The preprocessing sequence proceeds in a fixed execution order without branching. Preliminary epochs are created from a 1.0 Hz high-pass-filtered copy of the continuous recording to optimize component decomposition. Independent component analysis is fitted on these filtered epochs, and the resulting unmixing weights are transferred to the 0.1–100 Hz continuous analysis data. Final analysis epochs spanning −7.0 to 15.0 s relative to stimulus onset are then extracted from the unmixed continuous recording. Artifact rejection and channel repair are finalized by applying autoreject (Jas et al., 2017) in local mode using candidate interpolation counts of four, eight, and sixteen channels.

Independent components are categorized using the ICLabel deep neural network classifier (Pion-Tonachini et al., 2019). Components are rejected when the predicted probability exceeds 0.8 for any non-brain category other than the unclassified category. Retained components undergo objective artifact screening blind to experimental outcomes and model performance. A component is rejected if it exhibits a volume-repetition spectral peak with robust z exceeding three, cardiac phase-locking exceeding the ninety-fifth percentile of a participant-level circular-shift null distribution, or meets the high-frequency topographic artifact criterion. The topographic criterion identifies muscular contamination by requiring 70–95 Hz component power with robust z exceeding three combined with either at least fifty percent absolute topographic weight concentrated on frontopolar and temporal sensors (Fp1, Fp2, FT9, FT10, TP9, and TP10) or a Pearson correlation of r >= 0.50 between the component high-frequency plateau envelope and the frontopolar artifact proxy.

Electrode locations are digitized with EasyCap M1 coordinates and coregistered with individual T1-weighted structural volumes. Electrophysiological preprocessing is executed strictly within each participant prior to cross-validation.

### 5.2 Epoch Extraction, Reference Baselines, and Spectral Bands

A prestimulus voltage baseline spanning −0.2 to 0.0 seconds removes direct-current voltage offsets prior to event-related potential analyses. For oscillatory time-frequency decompositions, the primary baseline correction uses a log-ratio transformation relative to −5.0 to −0.01 seconds, referencing plateau power to the oscillatory state immediately preceding stimulation. To evaluate whether prestimulus power influences prediction, reference-window sensitivity analyses recompute power features using alternative prestimulus baselines of −2.0 to −0.01 seconds and −0.2 to −0.01 seconds, as well as an unnormalized specification using active-window log power with prestimulus power modeled as an explicit nuisance regressor.

Electrophysiological activity is decomposed into canonical frequency bands comprising delta (1.0–3.9 Hz), theta (4.0–7.9 Hz), alpha (8.0–12.9 Hz), beta (13.0–30.0 Hz), and gamma (30.1–77.0 Hz). Gamma activity is operationalized across three sub-bands: low gamma (30.1–38.0 Hz), mid gamma (43.0–56.0 Hz), and high gamma (67.0–77.0 Hz). Intervening intervals (38.0–43.0 Hz, 56.0–67.0 Hz, and 77.0–85.0 Hz) are omitted to prevent residual gradient-switching and volume-repetition harmonics from contaminating gamma features.

Empirical evaluation of continuous recordings following template subtraction reveals consistent narrowband spectral peaks coinciding with harmonics of the 900 ms volume repetition time. Across final-clean continuous recordings, cohort median peak frequencies appear at 20.02, 41.14, 61.10, and 82.21 Hz, exhibiting offsets from exact theoretical harmonics (+0.020, +0.027, −0.015, and −0.008 Hz) that fall well within the 0.061 Hz resolution of the spectral estimate. In the scanner harmonic spectrum quality control analysis, every continuous task run exhibits identifiable spectral peaks across these harmonic windows with run-level prominences exceeding 7.54 dB. The defined gamma sub-bands circumvent these harmonic peaks.

The residual 20 Hz harmonic peak falls within the conventional 13.0 to 30.0 Hz beta band. Conventional beta power is retained as a prespecified feature family, but beta effects are interpreted alongside artifact diagnostics and continuous spectral evidence. Exploratory beta sub-band specifications evaluate sensitivity to the 20 Hz harmonic by dividing the band into lower (13.0–17.9 Hz) and upper (23.1–30.0 Hz) ranges, excluding the 18.0 to 23.0 Hz window without altering the prespecified primary beta model.

The primary confirmatory gate evaluates ElasticNet regression using the joint alpha, beta, and gamma preset. The secondary frequency audit comprises alpha, beta, gamma, and alpha+beta presets. Delta, theta, delta+theta, and all-band models provide exploratory low-frequency and broad-band characterizations, with the all-band preset integrating the identical gamma sub-bands.

### 5.3 Frontal High-Frequency and Physiological Artifact Proxies

To prevent facial and cranial muscular activity from biasing predictive models, artifact control relies on an objective frontopolar high-frequency power proxy and electrocardiographic monitoring. Confirmatory feature matrices exclude frontopolar channels Fp1 and Fp2 entirely.

The frontopolar artifact proxy is extracted from continuous recordings following downsampling and band-pass filtering, prior to spatial interpolation, independent component rejection, or epoch thresholding. For each thermal trial, signals from Fp1 and Fp2 are filtered between 70 and 95 Hz outside the 60 Hz notch band, and Hilbert analytic power is averaged across 3.0 to 10.5 seconds and across both channels. This trial-level metric is written to the precleaning artifact proxy table and merged into the clean event table by run and trial index, ensuring that epoch rejection removes trials without altering proxy values.

During target preparation, the trial proxy is rasterized onto the acquisition time grid over the clean-event onset and duration. For each event, Nilearn generates a canonical-HRF regressor; its signed weights, divided by their sum, summarize the proxy and the per-volume FD/DVARS series. These deterministic summaries are prepared before cross-validation; nuisance regression coefficients are fitted within training folds. This operation is an HRF-weighted summary, not sampling a convolved proxy at a plateau peak. Non-finite timing, nonpositive durations, or absent HRF support raise an error. The unweighted proxy supports artifact reporting and separately executed censoring sensitivity checks. Required physiological nuisance values must be finite for every retained primary trial.

Rasterization uses the actual BOLD frame timestamps, including the slice-timing reference, and assigns the proxy where onset ≤ frame time < onset + duration. Rounding onset and duration separately to volume indices can select a different interval and is not used.

Multimodal artifact censoring thresholds comprise framewise displacement exceeding 0.5 mm, standardized DVARS robust z exceeding three, frontopolar high-frequency power robust z exceeding three, cardiac phase-locking exceeding the ninety-fifth percentile of the circular-shift null distribution, and scanner-harmonic spectral peaks with robust z exceeding three. Censoring sensitivity analyses examine effect sizes, prediction increments, and inferential stability following trial removal.

## 6. Functional Magnetic Resonance Imaging Target Construction

### 6.1 Single-Trial Estimation, Spatial Provenance, and Dot-Product Scoring

Trial-wise functional activation patterns are estimated using Least-Squares Separate general linear models restricted to thermal plateau exposures (Mumford et al., 2012). For each eligible target trial, a first-level model specifies a target-trial plateau regressor beginning at plateau onset and spanning the 7.5-second plateau duration. Concurrently, all other eligible plateau trials in the same task run are modeled via a single pooled non-target nuisance regressor. Ramp-up, ramp-down, post-stimulus fixation, and behavioral rating intervals enter the model as non-plateau nuisance regressors when timing data are available.

Single-trial models employ the canonical SPM hemodynamic response function (Friston et al., 1998), a cosine drift high-pass filter with a 0.008 Hz cutoff, and a six-millimeter full-width at half-maximum Gaussian spatial smoothing filter. The six-millimeter filter aligns with the derivation of the Neurologic Pain Signature (Wager et al., 2013), whereas the Stimulus Intensity Independent Pain Signature-1 was derived using eight-millimeter smoothing (Woo et al., 2017). Consequently, an eight-millimeter smoothing analysis is evaluated as a sensitivity check for the secondary signature. First-level head motion control employs twenty-four rigid-body motion parameters (Friston et al., 1996) alongside automated motion-outlier spike regressors. Single-trial estimates are excluded if the target regressor is collinear, non-estimable, or if the design matrix condition number exceeds 3000.

Published Neurologic Pain Signature and Stimulus Intensity Independent Pain Signature-1 patterns were trained in SPM MNI152 space, whereas preprocessed functional volumes are aligned to the fMRIPrep MNI152NLin2009cAsym template (Fonov et al., 2009, 2011). Because resampling grid dimensions does not alter anatomical template discrepancies, signatures are validated against published checksums and scored without assuming identical coordinate spaces. The published weights are preserved in sign and magnitude without study-specific thresholding, normalization, or rescaling.

To guarantee that scoring masks are invariant across participants and unaffected by individual acquisition boundaries, a standard scoring mask is established a priori by intersecting the non-zero signature weight extent with the standard two-millimeter nonlinear asymmetrical MNI152 template brain mask. Mask materialization verifies that the scoring mask retains 0.998 of Neurologic Pain Signature non-zero voxels and 0.957 of Stimulus Intensity Independent Pain Signature-1 non-zero voxels. Valid signature targets must preserve at least ninety percent of original non-zero voxel support, at least ninety percent of signed positive and negative weight support, and alter total absolute weight mass by no more than ten percent.

Trial-wise signature expression is computed as the unnormalized spatial dot product:

$$
y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v),
$$

where $\beta_{s,i}(v)$ denotes the Least-Squares Separate activation coefficient for participant $s$, trial $i$, and voxel $v$, and $M_k(v)$ denotes the fixed signature weight map for target $k$. Single-trial beta maps are not normalized or standardized prior to scoring, and absolute dot-product magnitudes are evaluated as within-participant relative indices. Target design efficiency is evaluated as the inverse of the contrast variance on the fitted design matrix. Participants are retained for confirmatory analysis if they provide at least twenty-five plateau trials with design condition numbers below 3000 and design efficiencies exceeding 0.1.

Target reproducibility is evaluated using stratified split-half reliability across one thousand random partitions stratified within participant and stimulus temperature. Each participant-by-temperature cell contributes average expression across split halves, and reproducibility is computed as the median Spearman-Brown-corrected Pearson correlation across valid splits (Brown, 1910; Spearman, 1910). This metric assesses the reproducibility of condition cell means driven by between-participant differences and temperature effects. It does not reflect residual trial-level variance or an empirical noise ceiling, and synthetic simulations lacking trial-level signal confirm that stable condition means yield split-half correlations exceeding 0.99.

### 6.2 Target Quality Control Metrics and Validation Figures

Whole-brain construct-validity GLMs use the slice-timing reference derived from each BOLD sidecar, matching signature-target estimation. The multi-run fit requires a common reference across a participant's runs and records it in the design audit; inconsistent references raise an error.

Quality control metrics are written to the target metrics table prior to evaluating electrophysiological predictors. For both signatures, the table summarizes retained trial counts, subject counts, bivariate correlations with stimulus temperature and behavioral ratings, and condition-level split-half reproducibility.

Target validity is documented across five standalone supplementary figures. Behavioral, Neurologic Pain Signature, and Stimulus Intensity Independent Pain Signature-1 dose-response plots display participant temperature trajectories behind equally weighted cohort means accompanied by ninety-five percent bootstrap confidence intervals. Complementary behavioral validity plots evaluate whether signature expression tracks reported pain beyond delivered temperature. Standardized partial regression slopes are estimated for the binary pain report and the continuous within-scale score while controlling categorical temperature, with the secondary signature model additionally adjusting for Neurologic Pain Signature expression. Standardized coefficients, non-estimability flags, and cohort summaries are recorded in the behavioral signature validity tables.

Electrophysiological and functional imaging manipulation checks are documented across twelve standalone publication figures. The cohort power spectral density figure displays continuous electroencephalographic spectra from 1 to 90 Hz with participant trajectories, cohort medians, ninety-five percent bootstrap confidence intervals, and marked harmonic exclusion windows alongside canonical frequency bands. Checkpoint spectral figures provide descriptive comparisons across raw, processed, and MNE-BIDS stages. The scanner harmonic spectrum figure characterizes continuous Welch spectra, harmonic peak offsets from volume-repetition multiples, and observed run-level prominences.

The power construct validity figure evaluates whether global oscillatory power tracks physical temperature and subjective intensity. Panel a depicts participant-centered temperature trajectories for alpha, beta, and gamma sub-bands across vertically aligned rows sharing a common decibel scale, with simultaneous ninety-five percent studentized bootstrap bands across thirty band-temperature cells. Panel b presents an aligned forest plot of participant-level partial correlations between band power and subjective intensity scores adjusting for temperature, task run, contact surface, within-run trial position, cardiac coupling, and frontopolar artifact power. The band power epoch evolution figure illustrates time-resolved global power from −5.0 to 14.5 seconds for alpha, beta, and gamma sub-bands, displaying individual trajectories and equally weighted cohort means with ninety-five percent bootstrap confidence intervals. The band time-frequency figure collection presents participant and cohort time-frequency representations (1 Hz resolution, 7-cycle Hanning windows) depicting temperature regression slopes in decibels per degree Celsius.

Sensor-space distributions are characterized by two ten-map inferential figures resolving alpha, beta, low gamma, mid gamma, and high gamma. The sensor power topographies figure presents temperature regression slopes and subjective intensity partial correlations across electrodes, with significant electrode clusters identified via Delaunay triangulation, cluster mass summation, and sign-permutation correction across all ten maps. The signature power topographies figure presents partial correlations between electrode-level power and signature expression after target-specific nuisance residualization. Whole-brain functional validity is documented in the functional magnetic resonance imaging construct validity figure, depicting unthresholded group-mean blood-oxygen-level-dependent responses to temperature and subjective intensity, with family-wise error control at p < 0.05 determined by ten thousand participant sign permutations.

Predictive performance is visualized in the primary prediction estimation figure, which depicts paired nuisance-only and nuisance-plus-EEG held-out predictive accuracies for the ElasticNet alpha+beta+gamma model across individual participants alongside cohort means and bootstrap intervals. The spectral specificity figure evaluates held-out prediction increments across alpha, beta, gamma, alpha+beta, and alpha+beta+gamma presets. Finally, the temporal specificity figure contrasts predictive accuracy across prestimulus, ramp-up, and plateau windows using the raw log-power feature transform.

### 6.3 Methodological Reporting Scope and Sensitivity Safeguards

The primary report records multiple empirical quality control metrics and diagnostic evaluations without deriving automatic interpretation verdicts.

Primary result cells must belong to the prespecified target, spectral-preset, and estimator grid. Required statistics must be finite, counts must be nonnegative integers, and probability values and exclusion fractions must lie in [0, 1]. Within each reported Holm family, a partly missing or invalid probability column raises an error instead of silently reducing the number of tests. An optional probability column absent for the entire family remains unreported.

For primary prediction performance, the report tabulates the subject-weighted out-of-sample incremental coefficient of determination and the associated one-sided permutation probability value for the single prespecified primary cell. A family-wise corrected probability value is included for database schema consistency, equaling the unadjusted probability value because the confirmatory gate evaluates a single prespecified model.

Target quality control metrics document the fidelity of the evoked neural signatures before predictive modeling. The report tabulates linear correlations of signature expression with stimulus temperature, raw pain report, and within-scale intensity, alongside standardized within-participant intensity regression slopes obtained from models that adjust categorical temperature and, for secondary signatures, primary signature expression. Condition-level reproducibility is characterized using stratified split-half correlations across participant-by-temperature cells with Spearman-Brown correction.

This condition-level reproducibility statistic measures the stability with which condition cell means are estimated across partitions. Because it is computed on cell means, its magnitude reflects stable between-participant baselines and stimulus-intensity differences rather than single-trial fluctuation reliability. Synthetic data comprising solely participant intercepts, temperature effects, and trial noise yield reproducibility values exceeding 0.99 in the absence of any reproducible single-trial residual signal. Consequently, this metric does not represent a trial-level reliability estimate or an empirical noise ceiling, and no threshold on it qualifies or disqualifies residual target validity.

Residual target attainability is quantified descriptively as the in-sample residual variance fraction remaining after subtracting the nuisance model coefficient of determination. This descriptive figure reflects in-sample linear variance explained by nuisance regressors and does not represent an analytical upper bound on the out-of-sample incremental prediction metric evaluated against the training-fold mean.

Temporal comparison analyses evaluate out-of-sample incremental prediction across prestimulus baseline intervals and pre-plateau ramp-up stimulation. These evaluations serve as empirical temporal comparisons rather than strict negative controls, because ramp-up represents genuine nociceptive energy and prestimulus state can genuinely modulate subsequent pain perception. A nonsignificant effect in a comparison window does not establish that prediction is absent or lower than plateau prediction; formal assertions of temporal superiority require testing paired differences between comparable estimators or prespecified equivalence intervals with matched spectral support.

The standard benchmark workflow executes the primary and secondary spectral presets, exploratory broad-band presets, within-condition centered evaluations, and temporal comparison windows. Conversely, sensitivity analyses examining artifact censoring thresholds, alternative hemodynamic response functions, shifted active windows, alternative pre-stimulus baselines, spatial smoothing filters, first-trial or first-run exclusions, and painful-trials-only sample scopes represent separate workflows whose derived outputs must be supplied explicitly through configured sensitivity output paths. When external sensitivity directories are omitted, corresponding report fields remain unpopulated, and results cannot be characterized as having satisfied unexecuted sensitivity checks. The primary artifact control within the automated execution remains the prespecified nuisance design combining electrode exclusion with hemodynamic-response-function-weighted movement, signal variance, high-frequency, and cardiac covariates. Formal precision simulations evaluating sample-size stopping rules remain planned methodological extensions.

### 6.4 Within-Person and Within-Condition Evaluation

The primary estimand evaluates out-of-sample prediction across a pooled cohort. Models can score positively on this metric by capturing between-participant baseline offsets or individual temperature-response slopes without tracking trial-to-trial variance among identical stimulus temperatures. Removing participant means addresses baseline offsets but preserves individual temperature slopes.

To isolate trial-level tracking, a within-condition centered metric is computed as a descriptive secondary endpoint. Target values, nuisance predictions, and combined predictions are centered within each participant-by-temperature cell. Temperature cells presented only once for a given participant are omitted because they center to zero and provide no variance. Predictive performance is evaluated per participant and averaged with equal weighting across participants, with trial and participant counts reported alongside. For the Stimulus Intensity Independent Pain Signature-1, the nuisance model incorporates Neurologic Pain Signature expression. This analysis provides an empirical description of out-of-sample predictions rather than a re-estimated model.

## 7. Electrophysiological Feature Extraction

Spectral power features are extracted using Morlet wavelet decomposition (Cohen, 2014). Wavelet cycles scale with frequency ($f / 2.0$, bounded between 3.0 and 15.0 cycles) with a temporal decimation factor of four. The precomputed benchmark extracts baseline-normalized total oscillatory power without event-related potential subtraction, avoiding cross-participant leakage or fold-dependent signal decomposition. Subtracting event-related templates is evaluated exclusively in sensitivity analyses where extraction is restricted strictly to training folds.

Single-trial spectral power is baseline-corrected via log-ratio transformation relative to the −5.0 to −0.01 second reference window and averaged across 3.0 to 10.5 seconds post-stimulus onset. Each retained trial yields one power estimate per channel and frequency band. Confirmatory feature matrices contain individual-channel log-ratio features across prespecified frequency bands, omitting frontopolar channels Fp1 and Fp2. Models incorporating gamma features employ the low, mid, and high gamma sub-bands. Region-of-interest and global-average matrices provide spatial-resolution sensitivities. Reference-window sensitivity analyses repeat extraction using −2.0 to −0.01 second and −0.2 to −0.01 second baselines, and an unnormalized specification extracts active-window raw log power with prestimulus power included as a model covariate.

Scalp regions of interest are defined from standardized ten-twenty channel layouts excluding Fp1 and Fp2, requiring at least two valid channels per region. Region features represent arithmetic averages of channel log-ratio power within each region and band. Exploratory electrophysiological features comprise spectral peak parameters, aperiodic exponent and offset (Donoghue et al., 2020), event-related synchronization and desynchronization, band power ratios, frontal alpha asymmetry, nonlinear signal complexity, and oscillatory burst characteristics. Exploratory families are evaluated with Holm correction across tested cells.

## 8. Nuisance Structure and Covariate Residualization

The primary benchmark evaluates prediction of raw signature expression using staged nuisance regression. The Level 2 nuisance model residualizes target expression against an intercept, categorical stimulus temperature, categorical thermode contact surface, task-run index, trial onset time, within-run trial position, hemodynamic-response-function-weighted framewise displacement, hemodynamic-response-function-weighted standardized DVARS, hemodynamic-response-function-weighted frontopolar high-frequency power, and residual electrocardiographic coupling. Modeling stimulus temperature and contact surface as categorical variables prevents linear assumptions from inflating residual variance. For the Stimulus Intensity Independent Pain Signature-1, Neurologic Pain Signature expression is included as an additional nuisance covariate.

The nuisance design matrix is checked for collinearity in each training fold using singular value decomposition on centered and scaled predictors. The design is classified as full rank when all non-intercept singular values satisfy the ratio $\sigma_j / \sigma_{\max} \ge 10^{-10}$. Any cross-validation fold exhibiting rank deficiency renders the cell ineligible for confirmatory inference. Within the staged estimator, nuisance regression parameters are estimated strictly on training-fold participants via singular-value-decomposition least squares.

Behavioral rating-residualized sensitivity analyses augment the nuisance model with binary pain reports and within-scale intensity ratings. These analyses quantify construct attenuation rather than primary incremental prediction.

## 9. Machine Learning Architecture and Model Estimation

### 9.1 Staged Incremental Residual Learning

Predictive modeling employs staged residual learning rather than joint regularized regression. In each cross-validation fold, the nuisance model is estimated via ordinary least squares on raw target scores using training participants only. Feature filtering, missingness evaluation, imputation, standardization, and target power transformations are learned exclusively from training participants.

The regularized machine learning estimator is fitted to Yeo-Johnson-transformed training residuals derived from the nuisance model (Yeo and Johnson, 2000). Held-out electrophysiological residual predictions are inverse-transformed back to the original target scale and added to held-out nuisance predictions before computing predictive accuracy.

### 9.2 Linear Regularized Models and Fold-Contained Preprocessing

Cross-validation uses a nested leave-one-subject-out architecture. The primary gate model is ElasticNet regression (Zou and Hastie, 2005) trained on individual-channel alpha+beta+gamma features. Ridge regression (Hoerl and Kennard, 1970) provides regularized linear sensitivity comparisons and supports Haufe forward-model reconstructions (Haufe et al., 2014). Random Forest regression (Breiman, 2001) provides an exploratory nonlinear benchmark.

Feature preprocessing is strictly contained within folds. Feature missingness must not exceed five percent across the training fold, and participant missingness must not exceed ten percent across retained features. Features exceeding the threshold are discarded within the fold; isolated missing values below the threshold are imputed using training-fold feature medians. Reaching participant missingness limits or discarding all features marks the fold ineligible. Retained features are standardized to zero mean and unit variance, with zero-variance features removed.

The Yeo-Johnson power transformation is applied to target residuals in confirmatory ElasticNet and Ridge models to stabilize variance and normalize error distributions. Predictions are inverse-transformed prior to performance scoring. Hyperparameters are tuned through five-fold inner GroupKFold cross-validation on training participants. Inner selection maximizes subject-weighted predictive accuracy on transformed residual targets. ElasticNet tunes the mixing parameter across values of 0.2, 0.5, and 0.8, and the penalty parameter across 0.001, 0.01, 0.1, 1.0, and 10.0, with a convergence limit of 10,000 iterations. Ridge tunes the penalty parameter across 0.01, 0.1, 1.0, 10.0, and 100.0. Random Forest evaluates five hundred estimators across tree depths of five, ten, twenty, and unconstrained depth, with minimum split samples of two, five, and ten, and minimum leaf samples of one, two, and four.

### 9.3 Exploratory Deep Regression Architecture

The BandTemporalRegressor deep neural network architecture is evaluated as an exploratory model. The model ingests band-limited Hilbert amplitude-envelope tensors cropped to 3.0 to 10.5 seconds post-stimulus onset, applying band-specific spatial convolutions, temporal integration layers, dropout regularization, and AdamW optimization (Loshchilov and Hutter, 2019). MNE's `apply_hilbert(envelope=True)` returns the absolute analytic signal; these tensors are amplitudes, not squared power or the baseline-normalized spectral features used by the primary benchmark.

The deep learning model is evaluated on residual target accuracy rather than raw incremental predictive accuracy, and its results are not directly comparable to the primary gate. Preprocessing standardization, target scaling, and nuisance regression parameters are fitted strictly on inner training participants, with the validation split used for early stopping excluded from preprocessing estimation to prevent information leakage. Summary logs document evaluation scales, preprocessing partitions, and trial counts.

Each varying band/channel amplitude series is scaled by its training standard deviation, including sub-microvolt variation; an absolute voltage threshold must not bypass standardization. Clean epochs must fully cover the requested analysis interval within half a sample. Fractional run or trial identifiers are rejected, and prediction metadata preserves the trial identifiers used for alignment after censoring.

Across participants, the cropped deep-model tensors must share the same sample times relative to stimulus onset. Matching array dimensions alone is insufficient: differing sampling rates or time origins raise an error. Epoch padding outside the configured analysis window may differ when the retained sample times agree.

## 10. Statistical Inference and Permutation Framework

Predictive accuracy is evaluated on the original target scale following inverse transformation and nuisance prediction addition. The out-of-sample coefficient of determination employs the training-fold target mean as the zero-skill reference:

$$
R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{\mathrm{train},f})^2},
$$

where $y_i$ denotes the observed signature target for trial $i$ in test fold $f$, $\hat{y}_i$ denotes the combined prediction, and $\bar{y}_{\mathrm{train},f}$ represents the mean target value across training participants in fold $f$. Participant-level R² is at most one and has no finite lower bound; scores are not clipped before equal-participant averaging. Ninety-five percent confidence intervals are derived from ten thousand participant-bootstrap resamples.

Nonparametric permutation testing evaluates the statistical significance of incremental prediction (Winkler et al., 2014) across five thousand valid permutation draws. Permutations follow a reduced-model residual scheme within outer cross-validation folds. The nuisance model is estimated on training participants, nuisance predictions are held constant for training and test trials, and nuisance residuals are circularly shifted within participant and task run. Permuted targets are reconstructed by adding shifted residuals to fixed nuisance predictions, followed by refitting the entire machine learning pipeline, including preprocessing and hyperparameter tuning.

Circular permutations treat task runs as independent exchangeability units. To preserve circular structure, valid runs must retain at least eight plateau trials, and permutation shifts are drawn uniformly from the complete cyclic group, including the zero-shift identity element. Retaining identity shifts preserves mathematical group properties necessary for exact permutation tests. Participants must provide at least three valid runs and twenty-five total trials to enter permutation testing. Draws with missing or non-finite fold scores are rejected within the configured invalid-draw budget; fitting and validation exceptions propagate as errors. Valid permutation draws must yield finite scores across all outer folds to ensure identical participant composition between observed and permuted test statistics. Permutation tests assess exchangeability under circular shifts within run, and false-positive rates must be interpreted alongside empirical exchangeability assumptions.

The complete cyclic group does not establish exchangeability of these recordings. Censored gaps, nonperiodic run boundaries, changing residual variance, and estimated nuisance effects can invalidate shift invariance. Confirmatory interpretation remains conditional on calibration using realistic null simulations of the complete nested procedure and the retained trial timings. The software tests exercise permutation construction and scoring contracts; they do not establish false-positive control for the study data.

## 11. Empirical Target Validation and Methodological Sensitivities

### 11.1 Behavioral Target Validity Diagnostics

Behavioral measures provide target validation benchmarks rather than independent predictive endpoints. General linear models evaluate associations of Neurologic Pain Signature and Stimulus Intensity Independent Pain Signature-1 expression with binary pain perception and within-scale intensity ratings. Models adjust for categorical temperature and, for the secondary signature, Neurologic Pain Signature expression.

The standardized intensity slope models warmth ratings on non-painful trials and pain ratings on painful trials along a continuous within-scale continuum, with binary pain status absorbing baseline level offsets. Pain-specific conclusions require secondary sensitivity analyses restricted to painful trials.

### 11.2 Temporal Specificity and Stimulus Comparison Windows

Temporal specificity is examined by training identical ElasticNet alpha+beta+gamma models on electrophysiological activity preceding or initiating stimulation. Prestimulus windows span −5.0 to −0.01 seconds and −0.2 to −0.01 seconds, using raw log-power features without baseline division. The wrong-lag comparison window spans the 0.0 to 3.0 second thermal ramp-up. Plateau sensitivity analyses divide the thermal plateau into early (3.0–5.5 s), mid (5.5–8.0 s), and late (8.0–10.5 s) intervals.

These analyses provide temporal comparisons rather than negative controls. The ramp-up window delivers nociceptive heat, and prestimulus oscillations can influence subsequent sensory processing. Reporting includes incremental predictive accuracies, bootstrap confidence intervals, and Holm-adjusted probability values without binary pass or fail labels. Demonstrating temporal superiority requires testing paired differences between estimators with equivalent spectral support.

### 11.3 Reference-Window and Baseline Normalization Sensitivities

The primary baseline correction evaluates plateau power relative to the preceding oscillatory context. Reference-window sensitivities repeat feature extraction and modeling using −2.0 to −0.01 second and −0.2 to −0.01 second reference windows. An unnormalized active power sensitivity evaluates active 3.0 to 10.5 second raw log power while including −5.0 to −0.01 second reference power as a linear covariate, testing whether predictive value depends on the ratio transform itself or persists when active power and prestimulus state are modeled independently.

## References

Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., Gramfort, A., Thirion, B., & Varoquaux, G. (2014). Machine learning for neuroimaging with scikit-learn. Frontiers in Neuroinformatics, 8, 14. https://doi.org/10.3389/fninf.2014.00014

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from continuous EEG recorded during functional MRI. NeuroImage, 12(2), 230-239. https://doi.org/10.1006/nimg.2000.0599

Allen, P. J., Polizzi, G., Krakow, K., Fish, D. R., & Lemieux, L. (1998). Identification of EEG events in the MR scanner: The problem of pulse artifact and a method for its subtraction. NeuroImage, 8(3), 229-239. https://doi.org/10.1006/nimg.1998.0361

Appelhoff, S., Sanderson, M., Brooks, T. L., van Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A. P., Larson, E., Gramfort, A., & Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software, 4(44), 1896. https://doi.org/10.21105/joss.01896

Behzadi, Y., Restom, K., Liau, J., & Liu, T. T. (2007). A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage, 37(1), 90-101. https://doi.org/10.1016/j.neuroimage.2007.04.042

Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: Standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16. https://doi.org/10.3389/fninf.2015.00016

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Brown, W. (1910). Some experimental results in the correlation of mental abilities. British Journal of Psychology, 3(3), 296-322. https://doi.org/10.1111/j.2044-8295.1910.tb00207.x

Buysse, D. J. (2014). Sleep health: Can we define it? Does it matter? Sleep, 37(1), 9-17. https://doi.org/10.5665/sleep.3298

CANlab. (n.d.). CanlabCore documentation. Retrieved May 17, 2026, from https://canlabcore.readthedocs.io/

Coghill, R. C., McHaffie, J. G., & Yen, Y.-F. (2003). Neural correlates of interindividual differences in the subjective experience of pain. Proceedings of the National Academy of Sciences of the United States of America, 100(14), 8538-8542. https://doi.org/10.1073/pnas.1430684100

Cohen, M. X. (2014). Analyzing neural time series data: Theory and practice. MIT Press.

Davis, K. D., Aghaeepour, N., Ahn, A. H., Angst, M. S., Borsook, D., Brenton, A., Burczynski, M. E., Crean, C., Edwards, R., Gaudilliere, B., Hergenroeder, G. W., Iadarola, M. J., Iyengar, S., Jiang, Y., Kong, J.-T., Mackey, S., Saab, C. Y., Sang, C. N., Scholz, J., ... Pelleymounter, M. A. (2020). Discovery and validation of biomarkers to aid the development of safe and effective pain therapeutics: Challenges and opportunities. Nature Reviews Neurology, 16(7), 381-400. https://doi.org/10.1038/s41582-020-0362-2

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23(12), 1655-1665. https://doi.org/10.1038/s41593-020-00744-x

Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A., Kent, J. D., Goncalves, M., DuPre, E., Snyder, M., Oya, H., Ghosh, S. S., Wright, J., Durnez, J., Poldrack, R. A., & Gorgolewski, K. J. (2019). fMRIPrep: A robust preprocessing pipeline for functional MRI. Nature Methods, 16(1), 111-116. https://doi.org/10.1038/s41592-018-0235-4

Fonov, V. S., Evans, A. C., McKinstry, R. C., Almli, C. R., & Collins, D. L. (2009). Unbiased nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage, 47(Suppl. 1), S102. https://doi.org/10.1016/S1053-8119(09)70884-5

Fonov, V. S., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C., & Collins, D. L. (2011). Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, 54(1), 313-327. https://doi.org/10.1016/j.neuroimage.2010.07.033

Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998). Event-related fMRI: Characterizing differential responses. NeuroImage, 7(1), 30-40. https://doi.org/10.1006/nimg.1997.0306

Friston, K. J., Williams, S., Howard, R., Frackowiak, R. S., & Turner, R. (1996). Movement-related effects in fMRI time-series. Magnetic Resonance in Medicine, 35(3), 346-355. https://doi.org/10.1002/mrm.1910350312

Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P., Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M., Keator, D., Li, X., Michael, Z., Maumet, C., Nichols, B. N., Nichols, T. E., Pellman, J., ... Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific Data, 3, 160044. https://doi.org/10.1038/sdata.2016.44

Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. NeuroImage, 87, 96-110. https://doi.org/10.1016/j.neuroimage.2013.10.067

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429. https://doi.org/10.1016/j.neuroimage.2017.06.030

Jas, M., Larson, E., Engemann, D. A., Leppäkangas, J., Taulu, S., Hämäläinen, M., & Gramfort, A. (2018). A reproducible MEG/EEG group study with the MNE software: Recommendations, quality assessments, and good practices. Frontiers in Neuroscience, 12, 530. https://doi.org/10.3389/fnins.2018.00530

Jasper, H. H. (1958). The ten-twenty electrode system of the International Federation. Electroencephalography and Clinical Neurophysiology, 10, 371-375.

Kim, H., Neubert, J. K., San Miguel, A., Xu, K., Krishnaraju, R. K., Iadarola, M. J., Goldman, D., & Dionne, R. A. (2004). Genetic influence on variability in human acute experimental pain sensitivity associated with gender, ethnicity and psychological temperament. Pain, 109(3), 488-496. https://doi.org/10.1016/j.pain.2004.02.027

Kim, J. A., & Davis, K. D. (2021). Neural oscillations: Understanding a neural code of pain. The Neuroscientist, 27(5), 544-570. https://doi.org/10.1177/1073858420958629

Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: Validity of a brief depression severity measure. Journal of General Internal Medicine, 16(9), 606-613. https://doi.org/10.1046/j.1525-1497.2001.016009606.x

Lee, T.-W., Girolami, M., & Sejnowski, T. J. (1999). Independent component analysis using an extended infomax algorithm for mixed sub-Gaussian and super-Gaussian sources. Neural Computation, 11(2), 417-441. https://doi.org/10.1162/089976699300016719

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations. https://openreview.net/forum?id=Bkg6RiCqY7

Mari, T., Henderson, J., Maden, M., Nevitt, S. J., Duarte, R., & Fallon, N. (2022). Systematic review of the effectiveness of machine learning algorithms for classifying pain intensity, phenotype or treatment outcomes using electroencephalogram data. The Journal of Pain, 23(3), 349-369. https://doi.org/10.1016/j.jpain.2021.07.011

Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012). Deconvolving BOLD activation in event-related designs for multivoxel pattern classification analyses. NeuroImage, 59(3), 2636-2643. https://doi.org/10.1016/j.neuroimage.2011.08.076

Muthukumaraswamy, S. D. (2013). High-frequency brain activity and muscle artifacts in MEG/EEG: A review and recommendations. Frontiers in Human Neuroscience, 7, 138. https://doi.org/10.3389/fnhum.2013.00138

Oldfield, R. C. (1971). The assessment and analysis of handedness: The Edinburgh inventory. Neuropsychologia, 9(1), 97-113. https://doi.org/10.1016/0028-3932(71)90067-4

Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. NeuroImage, 198, 181-197. https://doi.org/10.1016/j.neuroimage.2019.05.026

Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage, 59(3), 2142-2154. https://doi.org/10.1016/j.neuroimage.2011.10.018

Robinson, M. E., Riley, J. L. III, Myers, C. D., Papas, R. K., Wise, E. A., Waxenberg, L. B., & Fillingim, R. B. (2001). Gender role expectations of pain: Relationship to sex differences in pain. The Journal of Pain, 2(5), 251-257. https://doi.org/10.1054/jpai.2001.24551

Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 3(3), 271-295. https://doi.org/10.1111/j.2044-8295.1910.tb00206.x

Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). A brief measure for assessing generalized anxiety disorder: The GAD-7. Archives of Internal Medicine, 166(10), 1092-1097. https://doi.org/10.1001/archinte.166.10.1092

Sullivan, M. J. L., Bishop, S. R., & Pivik, J. (1995). The Pain Catastrophizing Scale: Development and validation. Psychological Assessment, 7(4), 524-532. https://doi.org/10.1037/1040-3590.7.4.524

van der Miesen, M. M., Lindquist, M. A., & Wager, T. D. (2019). Neuroimaging-based biomarkers for pain: State of the field and current directions. Pain Reports, 4(4), e751. https://doi.org/10.1097/PR9.0000000000000751

Wager, T. D., Atlas, L. Y., Lindquist, M. A., Roy, M., Woo, C.-W., & Kross, E. (2013). An fMRI-based neurologic signature of physical pain. New England Journal of Medicine, 368(15), 1388-1397. https://doi.org/10.1056/NEJMoa1204471

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. NeuroImage, 92, 381-397. https://doi.org/10.1016/j.neuroimage.2014.01.060

Woo, C.-W., Schmidt, L., Krishnan, A., Jepma, M., Roy, M., Lindquist, M. A., Atlas, L. Y., & Wager, T. D. (2017). Quantifying cerebral contributions to pain beyond nociception. Nature Communications, 8, 14211. https://doi.org/10.1038/ncomms14211

Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. Biometrika, 87(4), 954-959. https://doi.org/10.1093/biomet/87.4.954

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
