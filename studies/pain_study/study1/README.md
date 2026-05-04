# Materials and Methods

## Research Objectives
The primary objective of this study was to determine whether the continuous, trial-wise expression of established functional magnetic resonance imaging (fMRI) pain signatures can be directly predicted from concurrent band-limited electroencephalography (EEG) dynamics. Specifically, the predictive targets were the Neurologic Pain Signature (NPS) and the Stimulus Intensity Independent Pain Signature-1 (SIIPS1), modeled via a subject-level, trial-resolved multimodal architecture. 

As a secondary sensitivity objective, we assessed whether EEG-derived predictive accuracy remained robust after controlling for known behavioral and temporal confounds. This was achieved by testing the models against fMRI signatures that were strictly residualized against subjective pain ratings, stimulus temperatures, trial timing, and block effects using fold-contained nuisance regression. The analysis was strictly restricted to the plateau phase of thermal stimulation periods (contrasting pain versus non-pain), utilizing fMRI inputs normalized to the MNI152NLin2009cAsym standard space.

## Data Preconditions and Frequency Definitions
Analyses relied on task-evoked EEG epochs defined from -7.0 to 15.0 seconds relative to stimulus onset. Baseline correction was applied using the pre-stimulus window of -0.2 to 0.0 seconds. For time-frequency decompositions, a log-ratio baseline correction was independently applied utilizing a pre-stimulus window from -5.0 to -0.01 seconds. The active stimulation window utilized for predictive modeling was restricted to 3.0 to 10.5 seconds post-onset.

Neural oscillations were operationalized into standard frequency bands: alpha (8.0–12.9 Hz), beta (13.0–30.0 Hz), and gamma (30.1–80.0 Hz). Composite features spanning the full alpha-to-gamma spectrum were also constructed by concatenating these distinct ranges.

## fMRI Signature Target Construction
Trial-by-trial fMRI effect estimates were extracted using the Least-Squares Separate (LSS) generalized linear modeling approach. The LSS models utilized a canonical SPM hemodynamic response function (HRF), a cosine drift model, and a high-pass filter of 0.008 Hz, without additional spatial smoothing. Let $\beta_{s,i}(v)$ denote the LSS-derived blood-oxygen-level-dependent (BOLD) effect estimate for subject $s$, trial $i$, and voxel $v$. Let $M_k(v)$ denote the a priori spatial weight map for target $k \in \{\text{NPS}, \text{SIIPS1}\}$. The primary signature-expression metric was computed as the unscaled dot product across all voxels $V$:

$$y_{s,i}^{(k)} = \sum_{v \in V} \beta_{s,i}(v)M_k(v)$$

To construct the multimodal dataset, the resultant fMRI targets $y_{s,i}^{(k)}$ were matched to clean EEG trials using exact trial indices or rounded temporal onsets. In instances of duplicate event keys, target values were aggregated via the arithmetic mean.

## Feature-Based Machine Learning Architecture
We employed a rigorously nested, Leave-One-Subject-Out (LOSO) cross-validation framework to ensure true out-of-sample generalization. The primary confirmatory feature family consisted of predefined spectral power metrics, while exploratory models assessed alternative feature representations, including aperiodic components, event-related desynchronization, spectral asymmetries, and signal complexity.

To prevent cross-subject data leakage and mitigate the injection of physiological scaling outliers (e.g., variances in skull thickness), a strict sequence of preprocessing steps was applied independently within each outer training fold. Let $X$ represent the trial-by-feature matrix. Raw features were first standardized within-subject as $X_{s} = (X_s - \mu_s) / \sigma_s$. Missing trials were subsequently imputed using the median of the training cohort. Global standardization to unit variance was then applied across the training fold. Finally, to prevent bias against naturally low-amplitude high-frequency oscillations, a variance threshold was applied to remove features with a variance of exactly 0.0.

Processed fMRI targets were normalized using the Yeo-Johnson power transformation, denoted as $\tilde{y} = \mathrm{YJ}(y)$. Predictive mapping was performed using three distinct algorithms: ElasticNet regression, Ridge regression, and a bootstrap Random Forest regressor (500 estimators). The optimization objectives for the linear models were defined as follows:

For ElasticNet:
$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2$$

For Ridge regression:
$$\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \frac{\alpha}{2}\|\beta\|_2^2$$

Hyperparameters (e.g., regularization strength $\alpha$, L1 ratio $\rho$, tree depth, and sample split criteria) were dynamically tuned via a 5-fold inner group cross-validation loop restricted to the training subjects. 

## Deep Regression Neural Architecture
In parallel to feature engineering, a deep regression lane was implemented to learn directly from continuous, band-limited EEG dynamics. To accurately capture oscillatory power without inducing temporal edge artifacts or phase dependency, continuous EEG recordings were first band-pass filtered into the target frequency ranges. Instantaneous power was extracted via the analytic signal envelope using the Hilbert transform, and the resulting tensors were subsequently cropped to the active stimulation window. This yielded a four-dimensional input tensor encompassing trials, frequency bands, spatial channels, and time points.

Within each LOSO fold, the input tensors were standardized using the channel- and band-specific statistics of the training cohort:

$$\tilde{X}_{n,b,c,t} = \frac{X_{n,b,c,t} - \mu_{b,c}}{\sigma_{b,c}^{*}}$$

where $\mu_{b,c}$ is the mean and $\sigma_{b,c}^{*}$ is the lower-bounded standard deviation. Target variables were similarly z-scored utilizing the training subset statistics.

A custom PyTorch-based convolutional neural network was designed to enforce neurophysiological constraints. The architecture sequentially isolated spatial topographies prior to temporal integration. The depthwise spatial convolution extracted band-specific spatial maps:

$$H^{(1)}_{n,b,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{c} W^{spat}_{b,f,c}\tilde{X}_{n,b,c,t}\right)\right)$$

This was followed by a temporal convolution to integrate power dynamics over time:

$$H^{(2)}_{n,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{b,u} W^{temp}_{f,b,u} H^{(1)}_{n,b,f,t+u}\right)\right)$$

The latent temporal representation was downsampled via average pooling and passed through a dropout-regularized (probability 0.25) fully connected regression head:

$$z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}(H^{(2)}_n)\right)$$
$$\hat{y}_n = W_2 \, \mathrm{Dropout}\left(\mathrm{ELU}\left(W_1 \, \mathrm{vec}(z_n) + b_1\right)\right) + b_2$$

The network was optimized using the AdamW algorithm (learning rate = 0.001, weight decay = 0.0001) minimizing Mean Squared Error over 25 epochs. A 20% validation split of the training subjects was reserved for patience-based early stopping to prevent overfitting.

## Evaluation Metrics and Statistical Inference
Model performance was evaluated on the strictly held-out test subject during each LOSO iteration. To ensure an unbiased metric of generalizability that precludes target leakage, the coefficient of determination ($R^2$) was calculated using the target mean of the training fold ($\bar{y}_{train,f}$) as the zero-skill baseline:

$$R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{train,f})^2}$$

Statistical inference was established via nonparametric permutation testing (5,000 permutations). To preserve the block-wise temporal dependency structure while dismantling trial-wise predictability, target labels were permuted within-subject. Crucially, permutations were executed prior to the outer cross-validation splits, ensuring that nested hyperparameter tuning was completely unbiased against the null targets. P-values for the primary confirmatory models were corrected for multiple comparisons using the Holm step-down procedure. Absolute computational reproducibility was guaranteed by executing all random processes under a fixed global seed.
