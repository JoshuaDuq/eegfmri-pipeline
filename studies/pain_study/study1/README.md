# Materials and Methods

## Research Objectives
This study tested whether trial-wise expression of established fMRI pain signatures can be predicted above the nuisance-only baseline from concurrent EEG-derived features. The predictive targets were the Neurologic Pain Signature (NPS; Wager et al., 2013) and the Stimulus Intensity Independent Pain Signature-1 (SIIPS1; Woo et al., 2017), modeled via a LOSO trial-wise prediction with subject-grouped validation. The primary confirmatory estimand was the incremental raw-target prediction beyond nuisance ($\Delta R^2$), comparing a nuisance-only model to a combined nuisance-plus-EEG model. Residualized-target EEG prediction (Level 2) was treated as a secondary convergence analysis. The predictive target, feature scaling, imputation, dimensional filtering, nuisance residualization, hyperparameter selection, and target transformation were fit without using held-out-subject information. Unadjusted EEG-only raw prediction was retained as a descriptive analysis; subjective-rating residualization (Level 3) was treated as a construct-attenuation sensitivity analysis.

The NPS was developed as a brain-based marker of nociceptive processing (Wager et al., 2013), and its prediction may partly reflect stimulus intensity. The SIIPS1 captures stimulus-intensity-independent pain-related variance (Woo et al., 2017), making it conceptually closer to the residualized objective. SIIPS1 was nevertheless evaluated under the same residualization levels to verify independence in this dataset. All analyses were restricted to the plateau phase of thermal stimulation, including pain and non-pain thermal trials, utilizing fMRI inputs normalized to MNI152NLin2009cAsym space.

## Data Preconditions and Frequency Definitions
Analyses relied on task-evoked EEG epochs from $-7.0$ to $15.0$ s relative to stimulus onset. A $-0.2$ to $0.0$ s pre-stimulus baseline was applied to epoch-level voltage to remove DC offset prior to ERP and amplitude-based analyses. For time-frequency decompositions, the primary log-ratio baseline correction used the $-5.0$ to $-0.01$ s pre-stimulus window, selected to provide stable power estimates given that Morlet wavelet decomposition at alpha frequencies requires several cycles for adequate spectral resolution. Because this window may contain cue-locked expectancy activity, sensitivity analyses were prespecified using an immediate pre-stimulus baseline ($-0.2$ to $-0.01$ s) and an early pre-stimulus baseline ($-7.0$ to $-5.5$ s, interpreted only when the protocol event-timing log confirmed no cue onset fell within the window). This immediate baseline is recognized as a leakage/expectancy sensitivity analysis rather than an equally reliable low-frequency spectral baseline due to cycle constraints. Baseline conclusions were considered stable only if the sign and inferential status of primary alpha and beta effects were unchanged across the primary and immediate-baseline analyses. All baseline windows were applied identically across conditions.

The active stimulation window for predictive modeling was $3.0$–$10.5$ s post-onset. Neural oscillations were operationalized into standard frequency bands: alpha ($8.0$–$12.9$ Hz), beta ($13.0$–$30.0$ Hz), and gamma ($30.1$–$80.0$ Hz). Composite alpha-to-gamma features were also constructed. Gamma-band EEG during simultaneous fMRI is vulnerable to facial muscle activity, jaw tension, scanner gradient residuals, and cardiac artifacts, all of which can be elicited by painful stimulation; gamma-band analyses were therefore exploratory and were not used to support confirmatory claims. Gamma robustness was evaluated using temporal/frontal high-frequency electromyographic power, framewise displacement, DVARS, cardiac phase-locking, and residual spectral peaks at the volume-repetition frequency. Prespecified artifact thresholds were: framewise displacement $> 0.5$ mm, temporal/frontal high-frequency power robust $z > 3$, DVARS robust $z > 3$, cardiac phase-locking above the 95th percentile of the within-subject circular-shift null, and scanner-frequency residual peaks robust $z > 3$. Continuous artifact-effect associations were reported alongside categorical censoring to provide empirical justification for artifact sensitivity. Gamma effects were reported as artifact-sensitive when removing threshold-exceeding trials changed the effect direction or eliminated significance. The gamma band excludes a $\pm 1.0$ Hz notch around 60 Hz due to line-noise removal.

Primary analyses requiring physiological artifact control required synchronized, independent facial EMG channels and a dedicated ECG channel that were not part of the predictive EEG feature space. A subject failed primary inclusion if either channel type was absent, unsynchronized with the EEG/fMRI event logs, flat or saturated for any retained run, or unable to yield valid artifact metrics for at least 90% of otherwise retained plateau trials. These metrics were not imputed from scalp EEG channels and were not replaced by alternate proxies after outcome inspection.

## Simultaneous EEG-fMRI Preprocessing
Simultaneous EEG-fMRI acquisition introduces structured artifacts requiring correction prior to feature extraction (Bullock et al., 2021). Preprocessing proceeded in two stages.

In the first stage, MRI-induced artifacts were corrected using BrainVision Analyzer 2.0 (Brain Products GmbH). Gradient artifacts were removed via average artifact subtraction using a sliding-window template. Ballistocardiogram (BCG) artifacts were corrected via template subtraction aligned to the detected cardiac cycle.

In the second stage, the corrected data were imported into the MNE-BIDS-Pipeline. Continuous data were downsampled to 500 Hz, band-pass filtered (0.1–100 Hz), and notch-filtered at 60 Hz.

Bad channels were identified using PyPREP (Bigdely-Shamlo et al., 2015) with deviation-based and correlation-based rejection criteria. Detection was repeated three times with independent random seeds; a channel was marked bad only if flagged in a strict majority. Bad channels were interpolated via spherical spline interpolation, and data were re-referenced to the common average.

Independent component analysis was performed using the extended infomax algorithm (0.99 variance explained). ICA was fitted on 1.0 Hz high-pass filtered epochs to improve decomposition stability. Component classification used ICLabel (Pion-Tonachini et al., 2019); components were rejected if their predicted probability exceeded 0.8 for any non-brain category other than "other." Because ICLabel was trained on standard (non-fMRI) EEG, retained components were additionally inspected for spectral peaks at the volume-repetition frequency, cardiac phase-locking, and spatial topographies consistent with known artifact patterns.

Preprocessing was performed in the following sequence:

1. Preliminary epochs were created on a 1.0 Hz high-pass-filtered copy.
2. ICA was fitted on these filtered epochs.
3. ICA spatial weights were applied to the 0.1–100 Hz continuous analysis data.
4. Final analysis epochs were extracted from $-7.0$ to $15.0$ s relative to stimulus onset.
5. Autoreject (Jas et al., 2017) was applied in local mode with candidate interpolation counts
   of $\{4, 8, 16\}$ for trial rejection.

Electrode locations were digitized using the EasyCap M1 montage and co-registered to individual MRI. Subject-level unsupervised EEG preprocessing (bad-channel detection, ICA component classification, autoreject thresholds) was performed independently within each subject before cross-validation and is therefore not target-leaking, but not strictly non-transductive with respect to the held-out EEG distribution.

## fMRI Signature Target Construction
Trial-by-trial fMRI effect estimates were extracted using the Least-Squares Separate (LSS) approach, restricted to thermal plateau trials. The LSS event regressor onset was defined at the beginning of the thermal plateau, with duration equal to the plateau hold time. Ramp-up, ramp-down, and non-stimulation trial types were modeled as separate nuisance regressors. This design aligns the fMRI event with the protocol-defined plateau phase used for EEG feature extraction, though the resulting LSS beta is an HRF-convolved event-amplitude estimate rather than a millisecond-scale neural measure.

The primary LSS models used a canonical SPM hemodynamic response function, a cosine drift model, and a $0.008$ Hz high-pass filter without spatial smoothing. The fMRI denoising design included the 24-parameter rigid-body motion expansion, white-matter and CSF signals, framewise displacement, CompCor regressors, and motion-outlier regressors from fMRIPrep. Trials exceeding framewise displacement $> 0.5$ mm or standardized DVARS robust $z > 3$ were excluded; runs losing more than 20% of plateau trials failed inclusion. HRF/timing robustness was evaluated by repeating target construction with HRF temporal and dispersion derivatives and with a finite-impulse-response model, and by shifting the EEG active window by $\pm 2.0$ s. Primary HRF/timing robustness was assessed on the primary $\Delta R^2_{\text{LOSO}}$ statistic for the relevant confirmatory cell. A result was considered HRF/timing-stable only if the primary incremental $\Delta R^2_{\text{LOSO}}$ remained stable ($|\Delta(\Delta R^2)| < 0.02$) and retained significance, and the Level 2 residualized-target convergence remained positive.

The NPS weight map (Wager et al., 2013) and the SIIPS1 map (Woo et al., 2017) were registered to MNI152NLin2009cAsym space. When image grids differed, LSS beta maps were resampled to the signature-weight grid using continuous interpolation. For each signature, a fixed common group scoring mask $V^{(k)}$ was defined a priori before subject-level scoring to ensure the contributing voxel count and spatial extent were strictly identical across all subjects, runs, and trials. A signature target failed validity checks and was not analyzed if the common mask retained less than 90% of the original nonzero signature support, retained less than 90% of either the positive- or negative-weight support, or changed either positive or negative total absolute weight mass by more than 10% after resampling. To ensure valid handling of negative weights, we explicitly report the number of positive and negative voxels retained, the percentage of original signature support retained, and the stability of positive/negative weight distributions after resampling. Sensitivity analyses were conducted using the canonical signature grid. LSS beta maps were not smoothed, z-scored, or trial-normalized before scoring; signature weights were not re-estimated from study data. The signature-expression metric was computed as:

$$y_{s,i}^{(k)} = \sum_{v \in V^{(k)}} \beta_{s,i}(v) \, M_k(v)$$

where $\beta_{s,i}(v)$ is the LSS-derived BOLD estimate for subject $s$, trial $i$, voxel $v$, and $M_k(v)$ is the a priori weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$.

To construct the multimodal dataset, fMRI targets were matched to clean EEG trials using unique subject, run/block, and trial-index identifiers. EEG trigger onsets and fMRI plateau onsets were compared after applying the protocol offset from trigger to plateau start; residual absolute mismatch was required to be $\leq 0.010$ s for every retained trial.

To ensure trial-wise fMRI targets were usable, LSS diagnostics were prespecified. These included reporting the final number of trials per subject after censoring, LSS design efficiency and collinearity, and split-half reliability of NPS/SIIPS1 expression (calculated using a temperature-stratified split). Primary inclusion was based only on acquisition and design estimability: subjects were excluded if they retained fewer than 15 plateau trials, had an LSS design condition number exceeding 100, or had LSS design efficiency below 0.1. Split-half reliability was reported as a target-specific diagnostic, not as a primary exclusion rule, because excluding low-reliability targets can select a predictability-enriched sample. Reliability-informed sensitivity analyses were prespecified for subjects with split-half reliability $r \ge 0.4$ and for subjects with $\ge 25$ plateau trials to evaluate whether results depended on target precision or trial count.

## Hierarchical Nuisance Residualization
To characterize whether EEG prediction reflects pain-relevant fMRI variance or confound-correlated variance, fMRI signature expression was evaluated at three prespecified residualization levels representing distinct estimands.

**Level 1 — Raw expression.** No residualization. Models predict the full NPS/SIIPS1 dot-product expression, reflecting stimulus intensity, condition, temporal structure, subjective experience, and neural pain processing in an unknown mixture.

**Level 2 — Stimulus- and acquisition-controlled expression.** Level 2 targets were residualized against an intercept, continuous stimulus temperature in °C, block/run index, trial onset time, within-run trial number, HRF-weighted framewise displacement, HRF-weighted standardized DVARS, peripheral low-gamma power (derived strictly from independent facial EMG channels), and residual ECG coupling (derived from a dedicated external ECG channel independent of the predictive EEG feature space). Thermode surface was excluded from the primary Level 2 model to keep the estimand fixed across folds and was evaluated only as a prespecified sensitivity analysis when all thermode-surface levels were represented in every training fold. Binary pain condition was excluded because it is a deterministic or near-deterministic recoding of the thermal manipulation and can render the estimand unidentified. This level asks whether EEG predicts signature variance beyond measured stimulus level, session structure, trial order, and acquisition noise.

To ensure the Level 2 estimand remains strictly identical across all LOSO folds, a single rank-stable nuisance model was prespecified and applied uniformly without fold-specific fallback mechanisms. The primary Level 2 design utilized continuous temperature to ensure rank stability. A categorical temperature model was evaluated as a sensitivity analysis when all levels were represented in all training folds.

**Level 3 — Rating-residualized sensitivity.** The Level 2 design was augmented with subjective pain rating. Because ratings are a criterion measure of pain experience rather than a pure nuisance variable, this level may remove construct-relevant variance. Level 3 was interpreted only as a sensitivity analysis.

Within each level, nuisance coefficients were estimated exclusively on training subjects via SVD-based least squares:

$$\hat{\gamma} = \underset{\gamma}{\mathrm{argmin}} \; \| y_{\mathrm{train}} - Z_{\mathrm{train}}\gamma \|_2^2$$

Residualized targets for both training and test sets were computed by applying training-derived coefficients:

$$y_{\mathrm{train}}^{\mathrm{resid}} = y_{\mathrm{train}} - Z_{\mathrm{train}}\hat{\gamma}, \qquad y_{\mathrm{test}}^{\mathrm{resid}} = y_{\mathrm{test}} - Z_{\mathrm{test}}\hat{\gamma}$$

To strictly evaluate whether EEG provides incremental predictive value beyond nuisance structure, the primary incremental prediction analysis predicted raw NPS/SIIPS1 using a nuisance-only model versus a combined nuisance + EEG feature model. The nuisance-only model used unpenalized ordinary least squares regression. The primary nuisance-only model used the same rank-stable nuisance design in every fold, with continuous temperature and only globally estimable categorical variables. If the nuisance-only design became rank-deficient despite this, SVD-based least squares was used, but the estimand remained fixed.

For the combined model, core nuisance regressors were not penalized. In each outer fold:

1. The nuisance model was fitted on untransformed raw training targets.
2. Training and held-out nuisance predictions were computed as
   $\hat{y}_{Z,\mathrm{train}}$ and $\hat{y}_{Z,\mathrm{test}}$.
3. Training residuals were computed as
   $r_{\mathrm{train}} = y_{\mathrm{train}} - \hat{y}_{Z,\mathrm{train}}$.
4. The EEG model was trained only to predict this residual component from EEG features.
5. When target transformation was enabled, the Yeo-Johnson transformation was fitted on
   $r_{\mathrm{train}}$ only. EEG-model predictions were inverse-transformed back to raw
   residual units before being added to the nuisance prediction.

The combined held-out prediction was therefore:

$$\hat{y}_{\mathrm{test}} = \hat{y}_{Z,\mathrm{test}} + \hat{r}_{\mathrm{EEG},\mathrm{test}}$$

Both models used the exact same outer folds and scoring metrics. The primary test statistic was out-of-sample $\Delta R^2_{\text{LOSO}}$, computed subject-wise and then averaged:

$$\Delta R^2_{\text{LOSO}} = R^2_{\text{nuisance+EEG}} - R^2_{\text{nuisance-only}}$$

Confirmatory EEG prediction required $\Delta R^2_{\text{LOSO}} > 0$ with Holm-corrected permutation significance. As a secondary residualized-target convergence analysis, EEG features ($X$) and fMRI targets ($y$) were residualized against the Level 2 nuisance family within the training fold prior to predictive modeling. If unadjusted prediction (Level 1) was significant but the primary incremental analysis was not, EEG features were interpreted as tracking stimulus or session structure. If prediction survived Level 2 but not Level 3, prediction was interpreted as overlapping with subjective pain-report variance.

## EEG Feature Extraction
Spectral power features were extracted using Morlet wavelet decomposition with frequency-adaptive cycle counts (cycles $= f / 2.0$, bounded between 3.0 and 15.0) and decimation factor 4. To remove the condition-agnostic evoked response, the grand-average ERP was subtracted from each trial prior to decomposition. The primary ERP estimate was the fold-level grand average across all training trials, pooling conditions. The training-fold ERP was applied unchanged to held-out subject trials to prevent leakage. Subject-specific ERP subtraction was not used in the primary fold-isolated analysis. This procedure preserves condition-related evoked differences in the residual signal rather than isolating purely induced activity. Condition-specific ERP subtraction was evaluated as a stricter sensitivity analysis.

Spectral power was log-ratio baseline-corrected ($-5.0$ to $-0.01$ s) and averaged within the active window ($3.0$–$10.5$ s) to yield one power value per trial, channel, and frequency band. Power features were computed at three spatial resolutions: individual channels, predefined regions of interest, and global average.

An exploratory feature set encompassed spectral peak frequency and bandwidth, aperiodic slope and offset (specparam; fixed model), event-related desynchronization/synchronization, band power ratios ($\theta/\beta$, $\theta/\alpha$, $\alpha/\beta$, $\delta/\alpha$, $\delta/\theta$), hemispheric alpha asymmetry, nonlinear complexity measures (permutation entropy, sample entropy, Lempel-Ziv complexity), and oscillatory burst statistics.

## Feature-Based Machine Learning Architecture
A nested LOSO cross-validation framework ensured true out-of-sample generalization. The primary confirmatory model was ElasticNet regression on spectral power features. Ridge regression was included as a secondary confirmatory model because, as a linear model, it satisfies the Haufe et al. (2014) forward-model assumptions required for Study 3 source-space interpretation, providing a linearity-matched sensitivity check on ElasticNet. Random Forest was exploratory due to its sensitivity to correlated predictors in high-dimensional, low-subject neuroimaging settings and because its nonlinearity precludes formal Haufe transformation.

Preprocessing was applied independently within each outer training fold. Feature statistics (medians, means, standard deviations, variance thresholds) were estimated from training subjects only. Missing trials were imputed using training-cohort medians. Features were standardized to zero mean and unit variance; constant features were removed.

The Yeo-Johnson power transformation was applied only to the target component actually learned by the penalized EEG model. In the primary incremental analysis, this component was the training-fold nuisance residual $r_{\mathrm{train}}$; the nuisance-only prediction remained on the raw target scale. In secondary residualized-target models, targets were residualized first, and then the transformation was fitted and applied to the training residual targets within the fold. Predictions were inverse-transformed before reporting primary metrics. Both the numerator and denominator of the out-of-sample $R^2$ were computed on the original (untransformed) target scale after inverse-transforming predictions and, for the primary incremental analysis, after adding back the held-out nuisance prediction. This ensured that the training-fold mean baseline ($\bar{y}_{\mathrm{train},f}$), nuisance-only predictions, combined predictions, and prediction residuals were all in commensurate units. Sensitivity analyses without target transformation were reported to evaluate transformation stability. The ElasticNet objective was:

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2$$

The Ridge objective was:

$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \frac{\alpha}{2}\|\beta\|_2^2$$

Hyperparameters were tuned via 5-fold inner GroupKFold cross-validation restricted to training subjects, using the same subject-weighted $R^2$ metric as the outer LOSO evaluation to ensure the inner CV optimized for subject-level generalization rather than raw trial count. Confirmatory inference required at least 20 analyzable subjects and a preregistered precision simulation showing a 95% CI half-width $\leq 0.10$ for the primary $\Delta R^2$ statistic; the simulation could not substitute for the hard minimum subject count. This precision simulation must reflect the observed trial structure, within-subject autocorrelation, expected target reliability, and nuisance structure. For ElasticNet, $\alpha \in \{0.001, 0.01, 0.1, 1, 10\}$ and $\rho \in \{0.2, 0.5, 0.8\}$ with 10,000 maximum iterations. For Ridge, $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Random Forest used 500 estimators with max depths $\in \{5, 10, 20, \text{None}\}$, min samples split $\in \{2, 5, 10\}$, and min samples leaf $\in \{1, 2, 4\}$.

## Deep Regression Neural Architecture (BandTemporalRegressor)
An exploratory deep regression lane (BandTemporalRegressor) learned directly from continuous, band-limited EEG dynamics. The deep network is designated permanently as an exploratory architecture, independent of sample size.

Continuous EEG was band-pass filtered into target frequency ranges, and instantaneous power was extracted via the Hilbert transform. Resulting tensors were cropped to $3.0$–$10.5$ s, yielding input shape $N \times B \times C \times T$ (trials $\times$ bands $\times$ channels $\times$ time), standardized per channel and band using training-cohort statistics:

$$\tilde{X}_{n,b,c,t} = \frac{X_{n,b,c,t} - \mu_{b,c}}{\sigma_{b,c}^{*}}$$

A custom convolutional neural network was structured to learn band-specific spatial filters before temporal integration. The depthwise spatial convolution used a kernel spanning all channels with independent filter groups per band:

$$H^{(1)}_{n,b,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{c} W^{\mathrm{spat}}_{b,f,c}\tilde{X}_{n,b,c,t}\right)\right)$$

A temporal convolution (kernel length 15, 8 filters) then integrated across bands and time:

$$H^{(2)}_{n,f^{\prime},t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{b,f,u} W^{\mathrm{temp}}_{f^{\prime},b,f,u} H^{(1)}_{n,b,f,t+u}\right)\right)$$

The latent representation was downsampled via average pooling (factor 8) and passed through a dropout-regularized ($p = 0.25$) fully connected regression head:

$$z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}(H^{(2)}_n)\right)$$

$$\hat{y}_n = W_2 \, \mathrm{Dropout}\left(\mathrm{ELU}\left(W_1 \, \mathrm{vec}(z_n) + b_1\right)\right) + b_2$$

Optimization used AdamW (learning rate $= 0.001$, weight decay $= 0.0001$) with mean squared error loss over 25 epochs (batch size 32). Twenty percent of training subjects were reserved for early stopping (patience $= 5$ epochs). Standardization statistics ($\mu_{b,c}$, $\sigma_{b,c}^{*}$) were computed exclusively from the remaining 80% of training subjects; inner validation subjects were standardized using these statistics without contributing to their estimation.

## Evaluation Metrics and Statistical Inference
Primary metrics were computed on the active target scale after inverse-transforming predictions. The coefficient of determination used the training-fold target mean as the zero-skill baseline:

$$R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{\mathrm{train},f})^2}$$

The primary confirmatory statistic was the subject-weighted mean $\Delta R^2_{\text{LOSO}}$, defined explicitly as $\Delta R^2_{\text{LOSO}} = \frac{1}{S}\sum_{s=1}^{S} \Delta R_s^2$. Mean $R^2$ for the nuisance+EEG model and pooled trial-wise $R^2$ were reported descriptively. A confirmatory cell was positive only when $\Delta R^2_{\text{LOSO}} > 0$ and its permutation $p$-value survived Holm correction. Subject-wise metrics are reported with 95% BCa bootstrap confidence intervals (10,000 resamples). Group-level intervals resampled subjects; within-subject intervals used run/block-level resampling or circular block bootstrap.

Statistical inference for the primary incremental prediction analysis used nonparametric permutation testing (5,000 permutations). To test whether EEG adds predictive value beyond nuisance structure without destroying legitimate nuisance relationships, the following sequence was used for each outer fold and permutation:

1. The nuisance-only model was fitted on the unpermuted training data.
2. Nuisance predictions and residuals were computed for both training and held-out subjects
   using the training-derived nuisance coefficients.
3. The residual component was circularly shifted relative to EEG within each run separately
   for training and held-out subjects (minimum shift distance of 5 trials).
4. Permuted raw targets were reconstructed for both training and held-out subjects as the
   unshifted nuisance prediction plus the shifted residual.
5. Nuisance-only and nuisance+EEG models were refit on the permuted training target and
   scored against the permuted held-out target.

The $R^2$ denominator used the permuted training-target mean for that fold, matching the observed-analysis zero-skill baseline. This approximately preserves within-run temporal autocorrelation (Winkler et al., 2014). To address potential run-level confounding (e.g., sensitization, scanner drift), stricter sensitivity nulls were prespecified, including block-label shuffling and run-level permutations. Additionally, a strict sensitivity test evaluated whether positive prediction survived when subject and run means were explicitly removed from both EEG features and targets prior to permutation. Within-block random shuffling was evaluated as a sensitivity analysis.

The Holm-corrected confirmatory family comprised 2 targets (NPS, SIIPS1) $\times$ 2 models (ElasticNet, Ridge) $\times$ 3 frequency presets (alpha, beta, alpha+beta), all evaluated using the primary $\Delta R^2$ statistic. The secondary convergence family comprised the same 2 targets $\times$ 2 models $\times$ 3 frequency presets, evaluated using residualized-target Level 2 $R^2$, and Holm-corrected separately from the primary $\Delta R^2$ family. The Study 2 (Source Interpretation) gate is not selected from the Study 1 significant cells; it is tied strictly to the predesignated NPS ElasticNet alpha+beta cell regardless of whether other Study 1 cells perform better. Gamma, unadjusted (Level 1), subjective-rating residualization (Level 3), Random Forest, deep regression, and alternative designs were reported outside this family.

## Nuisance Prediction Reporting
Supplementary models were trained to predict each nuisance variable and subjective pain ratings from the same EEG features. Permutation p-values for nuisance-prediction models were Holm-corrected across the full set of supplementary targets. Strong prediction of nuisance variables indicated caution that the model may track task structure or acquisition noise. Strong prediction of ratings was interpreted as construct overlap with pain report. The primary incremental $\Delta R^2$ framework provides the primary evidence: EEG features must add predictive value for fMRI pain-signature expression beyond measured nuisance variables before any pain-relevant interpretation is made.

## Temporal Negative Controls
To assess temporal specificity, temporal negative controls used the same nuisance-only versus nuisance+EEG $\Delta R^2$ framework. Models were trained using EEG features from the pre-stimulus window ($-5.0$ to $0.0$ s) and the immediate pre-stimulus window ($-0.2$ to $0.0$ s) to predict post-stimulus target expression. The negative controls were evaluated for both NPS and SIIPS1 in Study 1, with the NPS negative control failure serving as a requirement for the Study 2 gate. If either pre-stimulus model yielded significant $\Delta R^2$, the model was interpreted as capturing expectancy, block structure, or subject-level state rather than nociceptive processing. Wrong-lag windows were prespecified as ramp-up ($0.0$–$3.0$ s), late ramp-down ($10.5$–$15.0$ s), early-shifted active ($1.0$–$8.5$ s), and late-shifted active ($5.0$–$12.5$ s), evaluated without reselecting model hyperparameters.
