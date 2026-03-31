EEG Feature Extraction
======================

**Module:** ``eeg_pipeline.analysis.features``

.. seealso::

   :doc:`preprocessing`
      Produces the clean epochs consumed by feature extraction.

   :doc:`../../user_guide/output_formats`
      Parquet layout, feature directory structure, and CSV export options.

   :doc:`../../user_guide/configuration`
      Full ``feature_engineering`` key reference.

   :doc:`../../user_guide/cli/features`
      CLI flags for feature category selection, spatial transforms, and IAF mode.

EEG feature extraction pipeline. Supports both **task-based** (event-related, trial-level)
and **resting-state** paradigms:

- **Task mode** (default): each trial (epoch) produces one row in the feature matrix;
  time windows are relative to event onset.
- **Rest mode** (``preprocessing.task_is_rest: true``): fixed-length segments replace
  trials; features are averaged across segments per subject.

See :doc:`../../glossary` for definitions of :term:`TFR`, :term:`wPLI`, :term:`AEC`,
:term:`PAC`, :term:`ITPC`, :term:`ERDS`, :term:`IAF`, and :term:`CSD`.

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 1

   * - Inputs
     - ``*_proc-clean_epo.fif`` (clean epochs from preprocessing)
   * - Outputs
     - ``features/<category>/features_<category>.parquet`` + metadata JSON
   * - CLI
     - ``eeg-pipeline features compute [--categories ...]``
   * - Config
     - ``feature_engineering`` section of ``eeg_config.yaml``

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Meaning
   * - :math:`e \in \{1,\dots,N_\text{trials}\}`
     - Trial (epoch) index
   * - :math:`c \in \{1,\dots,N_\text{ch}\}`
     - Channel index
   * - :math:`r`
     - Region-of-interest (ROI) index
   * - :math:`f`
     - Frequency (Hz)
   * - :math:`t`
     - Time (s)
   * - :math:`x_{e,c}(t)`
     - Time-domain EEG signal
   * - :math:`P_{e,c}(f,t)`
     - Time–frequency power (Morlet TFR)
   * - :math:`\mathrm{PSD}_{e,c}(f)`
     - Power spectral density (µV²/Hz)
   * - :math:`\mathcal{H}(\cdot)`
     - Hilbert transform
   * - :math:`B = [f_\text{min}^B, f_\text{max}^B]`
     - Frequency band with bounds
   * - :math:`T_\text{seg}`
     - Time-window segment (baseline, active, …)
   * - :math:`M_\text{seg}(t) \in \{0,1\}`
     - Boolean mask for segment :math:`T_\text{seg}`
   * - :math:`\Delta f`
     - Frequency bin width
   * - :math:`\varepsilon`
     - Small positive constant to prevent division by zero

Feature Categories
------------------

Available categories in priority order:

.. code-block:: text

   power, spectral, aperiodic, erp, erds, ratios, asymmetry, microstates,
   connectivity, directedconnectivity, itpc, pac, sourcelocalization,
   complexity, bursts, quality

Column Naming Schema
--------------------

Most feature columns follow:

.. code-block:: text

   {domain}_{segment}_{band}_{scope}_{statistic}

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Values
   * - ``domain``
     - ``power``, ``erp``, ``itpc``, ``erds``, ``conn``, ``comp``, …
   * - ``segment``
     - ``baseline``, ``active``, or named time windows; ``seg_N`` for rest segments
   * - ``band``
     - ``alpha``, ``theta``, ``beta``, ``broadband``, …
   * - ``scope``
     - ``ch`` (channel), ``roi``, ``global``, ``chpair``
   * - ``statistic``
     - ``logratio``, ``mean``, ``peak_freq``, ``db``, …

**Exceptions:** source-space features use ``src_*`` prefix. Connectivity and
microstate metrics encode method or state names in the suffix.

Spatial Aggregation Modes
--------------------------

Controlled by ``feature_engineering.spatial_modes``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Retained columns
   * - ``channels``
     - ``*_ch_*``, ``*_chpair_*``
   * - ``roi``
     - ``*_roi_*``, names matching ROI labels
   * - ``global``
     - ``*_global_*``, columns ending with ``_global``

Default when not set: ``["roi", "global"]``.

Time Windows
------------

For a segment with mask :math:`M_\text{seg}(t)`, the masked mean is:

.. math::

   \bar{x}_{e,c}^{(\text{seg})} =
   \frac{\sum_t M_\text{seg}(t)\, x_{e,c}(t)}{\sum_t M_\text{seg}(t)}.

Frequency Bands and IAF
-----------------------

Standard bands: ``delta``, ``theta``, ``alpha``, ``beta``, ``gamma``.

When ``feature_engineering.bands.use_iaf = true``, the :term:`IAF` is estimated from baseline PSD by fitting a log–log linear
regression to remove the aperiodic 1/f trend, then finding the peak
or power-weighted centroid in the search range (default 7–13 Hz). Bands are
adjusted:

.. math::

   \begin{aligned}
   \text{alpha} &= [\hat{f}_\alpha - w_\alpha,\; \hat{f}_\alpha + w_\alpha],\\
   \text{theta} &= [\max(3,\; \hat{f}_\alpha - 6),\; \max(4,\; f_\text{min}^\alpha)],\\
   \text{beta}  &= [\max(13,\; f_\text{max}^\alpha),\; f_\text{max}^\beta].
   \end{aligned}

In cross-validation, ``compute_iaf_for_fold`` restricts IAF estimation to
training trials only to prevent spectral band definitions from leaking
test-trial information.

Spatial Transform (:term:`CSD`)
--------------------------------

Applied before band filtering. Configured globally via
``feature_engineering.spatial_transform ∈ {none, csd, laplacian}``.

.. math::

   x^\text{CSD}(t) =
   \text{compute\_current\_source\_density}\!\bigl(x(t);\; \lambda^2,\; \text{stiffness}\bigr).

A failed transform raises a ``RuntimeError``; it is never silently skipped.

Feature Definitions
-------------------

8.1 Power (Oscillatory Power)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from :term:`TFR` power :math:`P_{e,c}(f,t)`:

**Band-integrated power:**

.. math::

   P_{e,c}^{B,\text{seg}} =
   \frac{\sum_{f \in B} \bar{P}_{e,c}^{B,\text{seg}}(f)\,\Delta f}{\sum_{f \in B} \Delta f}.

**Baseline-normalized log-ratio:**

.. math::

   \text{logratio}_{e,c}^B =
   \log_{10}\!\left(
     \frac{\max(P_{e,c}^{B,\text{active}},\, \varepsilon)}
          {\max(P_{e,c}^{B,\text{baseline}},\, \varepsilon)}
   \right), \quad \varepsilon = 10^{-20}.

**dB scaling:** :math:`\mathrm{dB}_{e,c}^B = 10 \cdot \text{logratio}_{e,c}^B`.

8.2 Spectral Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~

**Center frequency (spectral CoG):**

.. math::

   f_\text{cog} =
   \frac{\sum_{f \in B} f\,\mathrm{PSD}(f)\,\Delta f}
        {\sum_{f \in B} \mathrm{PSD}(f)\,\Delta f}.

**Bandwidth (power-weighted standard deviation):**

.. math::

   \sigma_B =
   \sqrt{
     \frac{\sum_{f \in B} (f - f_\text{cog})^2\,\mathrm{PSD}(f)\,\Delta f}
          {\sum_{f \in B} \mathrm{PSD}(f)\,\Delta f}
   }.

**Normalized spectral entropy** (with :math:`p(f) = \mathrm{PSD}(f)\,\Delta f \,/\, \sum_{f \in B} \mathrm{PSD}(f)\,\Delta f`):

.. math::

   H_B = -\frac{\sum_{f \in B} p(f)\ln p(f)}{\ln N_B}.

**Broadband spectral edge** :math:`f_\text{edge,95}` is the smallest :math:`f` such that cumulative PSD reaches 95%.

8.3 Aperiodic (1/f) Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Iterative aperiodic fit** in :math:`[f_\text{min}, f_\text{max}]` (e.g. 2–40 Hz):
fit model → compute residuals → remove large-positive-residual bins (oscillatory peaks) → repeat.

*Fixed-slope (linear):*

.. math::

   y(f) = \text{offset} + \text{slope} \cdot \log_{10} f.

*Knee model:*

.. math::

   y(f) = \text{offset} - \log_{10}\!\bigl(\text{knee} + f^\text{exponent}\bigr).

**Outputs per segment:** ``slope``, ``offset``, ``exponent``, ``knee``, ``r2``, ``rms``;
aperiodic-corrected band powers and theta/beta ratio.

8.4 ERP (Evoked Potentials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each ERP component window :math:`T_\text{comp}` and channel/ROI:

**Component mean:**

.. math::

   \text{mean}_{e,c}^\text{comp} =
   \frac{1}{|T_\text{comp}|} \sum_{t \in T_\text{comp}} \tilde{x}_{e,c}(t).

**Peak amplitude and latency:**

.. math::

   t^* = \arg\max_{t \in T_\text{comp}} s\bigl(\tilde{x}_{e,c}(t)\bigr), \qquad
   \text{peak}_{e,c}^\text{comp} = \tilde{x}_{e,c}(t^*).

**Area under the curve:** trapezoidal sum over contiguous valid intervals.

8.5 ERDS (Event-Related Desynchronization / Synchronization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using precomputed band envelopes :math:`|\mathcal{H}(x_{e,c}^B(t))|`:

**ERDS percentage:**

.. math::

   \text{ERDS\%}_{e,c}^B =
   100 \cdot
   \frac{P^{B,\text{active}}_{e,c} - P^{B,\text{baseline}}_{e,c}}
        {P^{B,\text{baseline}}_{e,c}}.

**ERDS in dB:**

.. math::

   \text{ERDS}_\text{dB} =
   10 \log_{10}\!\left(\frac{P^{B,\text{active}}_{e,c}}{P^{B,\text{baseline}}_{e,c}}\right).

**Laterality-aware pain markers:** when ``feature_engineering.erds.laterality_columns``
is configured and the events table contains a stimulus-side column, contralateral-hemisphere
somatosensory ERD features and onset/rebound latencies are computed.

8.6 Ratios (Band Power Ratios)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For numerator band :math:`B_\text{num}` and denominator band :math:`B_\text{den}`:

.. math::

   \text{power\_ratio}_e =
   \frac{P^{B_\text{num}}_e}{P^{B_\text{den}}_e}, \qquad
   \text{log\_ratio}_e =
   \ln\!\bigl(P^{B_\text{num}}_e + \varepsilon\bigr) -
   \ln\!\bigl(P^{B_\text{den}}_e + \varepsilon\bigr).

8.7 Asymmetry (Hemispheric Indices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a left–right electrode pair :math:`(L, R)` and band :math:`B`:

.. math::

   \text{index} =
   \frac{P^B_R - P^B_L}{P^B_R + P^B_L}, \qquad
   \text{logdiff} = \ln P^B_R - \ln P^B_L.

8.8 Connectivity (Undirected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`S_{ij}(f)` be the cross-spectrum and :math:`S_{ii}(f)`, :math:`S_{jj}(f)` auto-spectra.

**Weighted PLI (wPLI):**

.. math::

   \text{wPLI}_{ij} =
   \frac{\bigl|\mathbb{E}[\mathrm{Im}(X_i X_j^*)]\bigr|}
        {\mathbb{E}[|\mathrm{Im}(X_i X_j^*)|]}.

**Imaginary Coherence:**

.. math::

   \text{imCoh}_{ij} =
   \mathrm{Im}\!\left(\frac{S_{ij}}{\sqrt{S_{ii}\, S_{jj}}}\right).

**PLV:** :math:`\text{PLV}_{ij} = \bigl|\mathbb{E}[e^{i\Delta\varphi_{ij}}]\bigr|`.

**AEC:** :math:`z_{ij} = \mathrm{atanh}\!\bigl(\mathrm{clip}(\mathrm{corr}(A_i, A_j), -0.9999, 0.9999)\bigr)`.

**Dynamic connectivity:** sliding-window wPLI/AEC with optional K-means state clustering.
In ``trial_ml_safe`` mode, clustering is restricted to training-fold windows.

**Connectivity granularity:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Granularity
     - Pooling
     - CV safety
   * - ``trial``
     - One matrix per epoch
     - CV-safe by default
   * - ``condition``
     - Pool epochs sharing a condition label
     - Use ``train_mask`` in ``trial_ml_safe`` mode
   * - ``subject``
     - Pool all epochs
     - Cross-trial

8.9 Directed Connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Phase Slope Index (PSI):**

.. math::

   \text{PSI}_{ij} =
   \mathrm{Im}\!\left(\sum_f C_{ij}^*(f)\, C_{ij}(f + \Delta f)\right).

**DTF** and **PDC** from MVAR transfer and coefficient matrices.

8.10 ITPC and Phase Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit phasor: :math:`u_e(f,t) = Z_e(f,t) / (|Z_e(f,t)| + \varepsilon)`.

**ITPC:**

.. math::

   \text{ITPC}(f,t) =
   \left|\frac{1}{|\mathcal{T}|}\sum_{e \in \mathcal{T}} u_e(f,t)\right|.

Averaging modes: ``global``, ``fold_global`` (default; CV-safe), ``loo``, ``condition``.

8.11 PAC (Phase–Amplitude Coupling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mean vector length (MVL):**

.. math::

   \text{MVL} = \frac{\left|\sum_t A(t)\,u(t)\right|}{\sum_t A(t)}.

**Surrogate-based z-score** from trial-shuffled and/or circularly time-shifted surrogates.
Harmonic overlap guards reject invalid band combinations.

8.12 Source Localization
~~~~~~~~~~~~~~~~~~~~~~~~~

**LCMV Beamformer:**

.. math::

   w_v = \bigl(C + \text{reg}\cdot\mathrm{tr}(C)\,I\bigr)^{-1} l_v \bigl(l_v^\top \bigl(C + \text{reg}\cdot\mathrm{tr}(C)\,I\bigr)^{-1} l_v\bigr)^{-1}.

**eLORETA:**

.. math::

   \lambda^2 = \frac{1}{\mathrm{SNR}^2}, \qquad \hat{J}(t) = (A^\top A + \lambda^2 R)^{-1} A^\top x(t).

**fMRI constraint system:** when enabled, source space is restricted to suprathreshold
fMRI activation voxels. Voxels are thresholded (z-score or FDR), clustered
(minimum ``cluster_min_voxels`` = 50 voxels by default; override with
``cluster_min_volume_mm3`` for a volume-based threshold), and mapped to
``aparc+aseg`` labels for cross-subject harmonization.

**Output spaces:**

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Space
     - Column prefix
     - Description
   * - ``cluster``
     - ``..._fmri_cluster_*``
     - Subject-specific constrained clusters
   * - ``atlas``
     - ``..._atlas_*``
     - Voxels mapped to ``aparc+aseg`` label averages
   * - ``dual``
     - both
     - Emit cluster and atlas families together (default)

**Source condition contrasts:** difference of mean ROI band power between condition A and B.

.. note::

   ``feature_engineering.sourcelocalization.fmri.time_windows`` is explicitly unsupported
   and will raise ``ValueError`` if set.

8.13 Complexity
~~~~~~~~~~~~~~~~

**Lempel–Ziv Complexity (LZC):**

.. math::

   \text{LZC} = \frac{c}{n / \log_2 n}, \quad n = \text{sequence length}.

**Permutation Entropy (PE):**

.. math::

   \text{PE} = -\frac{\sum_\pi p(\pi)\log_2 p(\pi)}{\log_2(m!)}.

**Sample Entropy (SampEn):**

.. math::

   \text{SampEn}(m, r) = -\log\frac{A}{B}.

**Multiscale Entropy (MSE):** coarse-grain by averaging non-overlapping blocks
of length :math:`s`, then compute SampEn at each scale :math:`s`.

8.14 Bursts (Transient Oscillations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threshold-based burst detection on band envelopes using percentile, z-score, or MAD
thresholds estimated from baseline. Outputs per segment/band: burst count, rate, mean
duration, mean amplitude, occupancy fraction.

8.15 Microstates
~~~~~~~~~~~~~~~~~

**Global Field Power:**

.. math::

   \text{GFP}_e(t) =
   \sqrt{\frac{1}{N_\text{ch}} \sum_c \bigl(x_{e,c}(t) - \bar{x}_e(t)\bigr)^2}.

Scalp maps at GFP peaks are normalized and clustered into :math:`K` microstate templates.
**Per-state statistics:**

.. math::

   \text{coverage}_k = \frac{N_k}{N_t}, \qquad \text{mean duration (ms)}, \qquad \text{occurrence rate (Hz)}.

In ``trial_ml_safe`` mode, template clustering is restricted to training trials.

8.16 Quality Metrics
~~~~~~~~~~~~~~~~~~~~~

Per segment and channel: variance, peak-to-peak, finite fraction, SNR (dB),
muscle artifact index (high-frequency power fraction).

Change Scores
-------------

When ``feature_engineering.compute_change_scores = true``, change-score columns are
appended for paired baseline/active variants:

.. math::

   X_\Delta = X^\text{active} - X^\text{baseline} \quad \text{(difference; default)}.

Also available: ``percent``, ``log_ratio``, ``ratio``. Custom pairings via
``--change-scores-window-pairs baseline:plateau baseline:active``.

Normalization
-------------

All schemes estimate parameters from a reference set that **must** be the training
subset in cross-validation contexts.

**Z-score:**

.. math::

   z_i = \frac{x_i - \mu^\text{ref}}{\max(\sigma^\text{ref},\, \varepsilon)}.

**Robust (median/MAD):**

.. math::

   z_i^\text{robust} =
   \frac{x_i - m^\text{ref}}
        {\max\!\bigl(\mathrm{MAD}(x^\text{ref})_\text{normal},\, \varepsilon\bigr)}.

**Min–max** to :math:`[a, b]`; **rank-based** (0–1 normalized ranks);
**log** (:math:`\ln` or :math:`\log_{10}`).

Reference modes: ``all`` (default), ``condition``, ``run``.

Cross-Validation Hygiene
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - Leakage risk
     - Safeguard
   * - IAF estimation
     - Test-trial spectra influence band definitions
     - ``compute_iaf_for_fold`` uses ``train_mask`` only
   * - Evoked subtraction
     - Condition averages include test trials
     - ``ValueError`` raised unless ``train_mask`` provided
   * - ITPC
     - Trial-average phase involves test trials
     - ``fold_global`` mode restricts to training trials
   * - PAC surrogates
     - Surrogate distribution on all trials
     - Surrogates computed from training trials only
   * - Microstate templates
     - Clustering sees test-trial scalp maps
     - K-means restricted to training trials
   * - Connectivity (condition)
     - Condition average includes test trials
     - Training-only aggregation enforced

**Analysis modes:**

- ``trial_ml_safe`` — all cross-trial computations require a valid ``train_mask``; any violation raises an error.
- ``group_stats`` — aggregations over all available trials; for group-level descriptive statistics only.

Resting-State Restrictions
---------------------------

The following families require event-locked epochs and are automatically
skipped when ``task_is_rest = true``:

- ``erp`` — requires event-onset-aligned component windows
- ``erds`` — requires a baseline window relative to event onset
- ``itpc`` — inter-trial phase clustering requires repeated trial markers
- ``pac`` — surrogate scheme assumes repeated trial structure

All other families run normally. Analysis mode may be ``group_stats`` or
``trial_ml_safe``; the latter treats fixed-length segments as the trial unit.
Evoked subtraction (``subtract_evoked = true``) raises an error in rest mode.

Output Files
------------

.. code-block:: text

   derivatives/<study>/sub-<id>/<task>/eeg/features/<family>/
   ├── features_<family>.parquet
   └── metadata/
       └── features_<family>.json
