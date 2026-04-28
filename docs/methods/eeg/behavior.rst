Behavioral Statistics
=====================

.. raw:: html

   <p class="hero-lede">
     Dependency-resolved DAG of trial-level analyses linking EEG features to
     behavioral variables. Paradigm-agnostic: supports continuous, binary, and
     categorical predictors. FDR correction applied throughout.
   </p>

.. grid:: 2
   :gutter: 2
   :class-container: meta-cards

   .. grid-item-card:: Inputs

      Aligned events (typically ``*_proc-clean_events.tsv`` when available) ·
      feature Parquet tables · ``behavior_config.yaml``

   .. grid-item-card:: Outputs

      Per-analysis TSV results · correlation matrices ·
      condition comparison tables

   .. grid-item-card:: CLI

      ``eeg-pipeline behavior [compute | visualize]``

   .. grid-item-card:: Config

      ``behavior_config.yaml`` · ``feature_engineering.analysis_mode``

.. seealso::

   :doc:`features`
      EEG feature tables consumed as inputs to behavioral correlations.

   :doc:`machine_learning`
      Trial-level predictive modeling using the same behavioral targets.

   :doc:`../../user_guide/data_layout`
      ``events.tsv`` column requirements for predictor and outcome aliases.

   :doc:`../../user_guide/cli/behavior`
      CLI flags for behavioral analysis modes.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`i \in \{1,\dots,N_\text{trials}\}`
     - Trial index within a subject
   * - :math:`g`
     - Grouping unit (run / block) for non-i.i.d. safeguards
   * - :math:`s`
     - Subject index (group-level aggregation)
   * - :math:`f \in \{1,\dots,F\}`
     - EEG-derived feature index
   * - :math:`y_i`
     - Behavioral outcome for trial :math:`i` (e.g. subjective rating)
   * - :math:`P_i`
     - Predictor value for trial :math:`i` (e.g. stimulus intensity)
   * - :math:`x_{i,f}`
     - EEG feature :math:`f` for trial :math:`i`
   * - :math:`G_i \in \{1,\dots,N_\text{groups}\}`
     - Group label (run / block / cluster)
   * - :math:`n_\text{perm}`
     - Number of permutations

Trials are **not i.i.d.**: they are clustered within runs/blocks and subjects.
All permutation-based inference respects this structure via grouped label shuffling.

Formula Map
-----------

Use this map to jump from an analysis question to the corresponding estimand or
test statistic.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Analysis
     - Question
     - Primary quantity
   * - Predictor residual
     - What behavioral variance remains after the predictor?
     - :math:`y_i - \hat{y}_i`
   * - Correlations
     - Which EEG features track behavioral targets?
     - Pearson/Spearman :math:`r`, partial :math:`r`, permutation :math:`p`
   * - Regression
     - Does a feature explain outcome variance beyond covariates?
     - :math:`\Delta R^2`, HC3 standard errors, Freedman-Lane permutation
   * - Reliability
     - Are features stable across repeated measurements?
     - ICC(3,1)
   * - Condition comparison
     - Do feature values differ between conditions?
     - Welch :math:`t`, Hedges :math:`g`, paired :math:`d_z`
   * - Temporal statistics
     - When in time-frequency space does the feature-behavior link appear?
     - Correlation-to-:math:`t`, cluster mass
   * - Group-level correlation
     - What is the equal-subject aggregate association?
     - Fisher-:math:`z` averaged :math:`r_\text{group}`
   * - Multiple comparisons
     - Which discoveries survive family control?
     - BH and hierarchical FDR

Pipeline DAG
------------

Stages run in dependency order. ``cluster`` operates directly on epoch data
and is parallel to (not dependent on) the temporal-power stages.
``hierarchical_fdr_summary`` and ``export`` are leaf stages with no
downstream consumers within the per-subject DAG.

.. code-block:: text

   load
   └── trial_table
       ├── predictor_residual          [continuous predictor only]
       ├── correlate_design
       │   └── correlate_effect_sizes
       │       ├── correlate_pvalues
       │       └── correlate_primary_selection
       │           └── correlate_fdr
       ├── regression
       ├── icc
       ├── condition_column
       └── temporal_tfr
           └── temporal_stats
   cluster                              [requires epochs; parallel to temporal_*]
   hierarchical_fdr_summary             [validation summary across analyses]
   export

Computation groups (``--computations``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Stages enabled
   * - ``trial_table``
     - ``trial_table``
   * - ``predictor_residual``
     - ``predictor_residual``
   * - ``regression``
     - ``regression``
   * - ``icc``
     - ``icc``
   * - ``correlations``
     - full ``correlate_*`` chain
   * - ``condition``
     - ``condition_column``
   * - ``temporal``
     - ``temporal_tfr``, ``temporal_stats``
   * - ``cluster``
     - ``cluster``
   * - ``validation``
     - ``hierarchical_fdr_summary`` (cross-analysis FDR summary; on by default)
   * - ``multilevel_correlations``
     - group-level multilevel correlations (outside subject DAG)

Statistical Safeguards
----------------------

Non-i.i.d. Trial Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trials within a subject are clustered within runs/blocks and are not exchangeable.
The following stages enforce grouped permutation unless ``allow_iid_trials = true``
is explicitly set: ``correlate_*``, ``regression``, ``condition_column``.

.. note::

   Grouped permutation labels must be complete and non-missing for every
   analyzed trial — partially missing run/block labels raise instead of being
   silently excluded from the permutation sample. Permutation scheme values
   are validated strictly; unsupported values raise instead of falling back
   to ``shuffle``.

Group-Level Permutation Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group-level multilevel correlation tables report permutation-null quantiles as
``r_null_q_2_5`` and ``r_null_q_97_5``. These columns summarize the null
distribution used for permutation inference; they are not confidence intervals
for the observed group-level correlation estimate.

Predictor Type Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``behavior_analysis.predictor_type`` key declares the nature of the predictor:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Definition
   * - ``continuous``
     - Ordered numeric scale with ≥ 5 distinct levels. Enables
       ``predictor_residual`` and spline/outcome_hat control.
   * - ``binary``
     - Two-level factor. Disables curve-fitting analyses.
   * - ``categorical``
     - Unordered multi-level factor. Same restrictions as binary.

Predictor Control Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Strategy
     - Description
     - Requires
   * - ``linear``
     - Add predictor as a linear covariate
     - Any predictor type
   * - ``outcome_hat``
     - Use :math:`\hat{y} = f(P)` (fitted outcome) as covariate
     - ``continuous`` only
   * - ``spline``
     - Restricted cubic spline of predictor as covariate
     - ``continuous`` only

.. note::

   Requested controlled estimands fail fast when the requested control
   cannot actually be applied. ``spline`` requires
   ``behavior_analysis.predictor_type = continuous`` and a successfully
   identified nonlinear spline basis; ``outcome_hat`` requires the
   precomputed ``outcome_hat_from_predictor`` column. The pipeline does not
   silently degrade to linear or no adjustment.

Stage Definitions
-----------------

Stage 1 — Load and Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reads aligned behavioral events, predictor series, covariates, and trial-wise EEG
feature tables into ``BehaviorContext``. Writes a JSON QC summary (trial counts,
missingness fractions, outcome/predictor distributions, analysis configuration fields).

Stage 2 — Trial Table
~~~~~~~~~~~~~~~~~~~~~~~

Constructs the canonical trial-level DataFrame by joining aligned behavioral events
with all named feature tables on the canonical ``trial_id`` column. One row per trial;
one column per behavioral or EEG-feature variable.

.. note::

   Row-order-only alignment is not considered valid scientific evidence of
   correspondence. By default, behavior trial tables require explicit
   ``trial_id``-based feature/event alignment before feature columns are
   combined with clean events.

.. warning::

   The implementation still exposes an explicit
   ``behavior_analysis.trial_table.disallow_positional_alignment = false``
   override that permits positional row-order alignment when feature tables do
   not carry canonical ``trial_id`` metadata. Analyses produced under that
   override should be treated as methodologically unsafe until key-based
   alignment is restored.

Stage 3 — Predictor Residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Residualizes the outcome on the predictor to isolate variance not explained by
stimulus intensity. Requires ``predictor_type = continuous`` (≥ 5 unique
predictor values).

.. math::

   \text{predictor\_residual}_i = y_i - \hat{y}_i, \qquad \hat{y}_i = f(P_i).

Model selection: spline OLS candidates ``outcome ~ bs(predictor, df=d, degree=3)``
for configurable degrees of freedom, with the lowest-AIC spline selected.
Set ``behavior_analysis.predictor_residual.method: "poly"`` to use the
polynomial model instead.
Optional cross-fit residuals (``GroupKFold``) via ``--predictor-residual-crossfit``.

.. note::

   Correlation target selection only promotes ``predictor_residual_cv`` when
   that column contains finite residual values; if cross-fitting is skipped
   and the column is all missing, the standard ``predictor_residual`` target
   remains primary. Predictor-residual construction itself is strict: fit
   failures surface as errors instead of silently dropping the residual
   columns and reverting to the raw outcome target.

Stage 4 — Correlations
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Canonical behavior-column overrides are strict. If
   ``behavior_analysis.outcome_column`` or
   ``behavior_analysis.predictor_column`` is set, that named column must
   exist and be numeric — the pipeline does not fall back to
   ``event_columns.*`` aliases. Likewise, an explicit
   ``correlations.target_column`` must resolve to a valid numeric trial-table
   column, and every entry in an explicit ``correlations.targets`` list must
   resolve to one.

Correlation Types
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Description
   * - ``raw``
     - Simple Pearson or Spearman :math:`r` with no covariate control
   * - ``partial_cov``
     - Partial :math:`r` controlling for user-specified covariates
   * - ``partial_predictor``
     - Partial :math:`r` controlling for the predictor variable
   * - ``partial_cov_predictor``
     - Partial :math:`r` controlling for both covariates and predictor
   * - ``run_mean``
     - Correlations on run-aggregated means

Partial Correlation
^^^^^^^^^^^^^^^^^^^

Test statistic with :math:`k` covariates:

.. math::

   t = r\sqrt{\frac{n - k - 2}{1 - r^2}}, \qquad
   p = 2\, P\!\left(|T_{n-k-2}| \ge |t|\right).

Permutation P-Value
^^^^^^^^^^^^^^^^^^^

Phipson-Smyth correction:

.. math::

   p_\text{perm} = \frac{N_{\text{extreme}} + 1}{n_\text{perm} + 1}.

Stability and Bayes Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``--loso-stability``
     - Recompute correlations on :math:`N-1` subjects; report mean LOSO
       :math:`r` and SD across folds.
   * - ``--compute-bayes-factors``
     - JZS :math:`\mathrm{BF}_{10}` approximation alongside classical
       p-values.

Stage 5 — Regression
~~~~~~~~~~~~~~~~~~~~~~

Per-feature OLS with optional predictor interaction:

.. math::

   y = Z\gamma + \beta_f x_f + \beta_\text{int}(x_f \cdot P) + \varepsilon \quad \text{(full)}.

Incremental Explained Variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \Delta R^2 = R^2_\text{full} - R^2_\text{reduced}.

HC3 Standard Errors
^^^^^^^^^^^^^^^^^^^

Heteroskedasticity-consistent covariance:

.. math::

   \widehat{\mathrm{Cov}}_\text{HC3} =
   (X^\top X)^{-1} X^\top \mathrm{diag}(w)\, X (X^\top X)^{-1}, \qquad
   w_i = \frac{e_i^2}{(1 - h_i)^2}.

Feature-term permutation uses Freedman–Lane residual permutation.

Stage 6 — ICC Reliability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Intra-class correlation ICC(3,1) for test–retest reliability:

.. math::

   \mathrm{ICC}(3,1) =
   \frac{MS_\text{rows} - MS_\text{error}}
        {MS_\text{rows} + (k-1)\, MS_\text{error}}.

Stage 7 — Condition Comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-Group Welch Test
^^^^^^^^^^^^^^^^^^^^

.. math::

   t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}.

Effect Sizes
^^^^^^^^^^^^

.. math::

   d = \frac{\bar{x}_1 - \bar{x}_2}{s_\text{pooled}}, \qquad
   g = d\!\left(1 - \frac{3}{4\,df - 1}\right), \qquad
   d_z = \frac{\bar{d}}{s_d} \text{ (paired)}.

Multi-Group Comparisons
^^^^^^^^^^^^^^^^^^^^^^^

For 3+ levels, the pipeline uses pairwise Mann-Whitney U (unpaired) or Wilcoxon
signed-rank (paired). Omnibus tests are not performed.

When ``primary_unit = run_mean``, the pipeline first aggregates to run×condition
cells and drops cells below ``behavior_analysis.condition.min_trials_per_condition``
before running paired condition statistics.

.. note::

   Run-level condition inference is strict about aggregation keys: the
   configured run column must exist, and both the run column and condition
   column must be fully labeled before run×condition aggregation begins. An
   explicit ``condition.compare_column`` must resolve to an existing
   trial-table column — the stage does not substitute a fallback.

.. note::

   For ROI power correlations, an explicit
   ``correlations.power_segment_preference`` must match actual segment
   columns for the analyzed band; the pipeline does not widen back to all
   segments when the requested segment is absent. Permutation testing for
   ROI power correlations propagates grouped trial-label structure into the
   ROI permutation p-values instead of defaulting to an i.i.d. shuffle null.

Stage 8 — Temporal Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Power-bin statistic averaged within time window :math:`w`:

.. math::

   b_{i,f,w} = \mathrm{mean}_{t \in w}\, P_{i,f,t}.

Correlation-to-:math:`t` transform for cluster forming:

.. math::

   t = r\sqrt{\frac{\mathrm{dof}}{1 - r^2}}.

Temporal multiple-comparison correction: ``fdr``, ``bonferroni``, ``cluster``, or ``none``.

.. note::

   When no explicit ``temporal.target_column`` is set, temporal target
   resolution follows the canonical outcome resolver, so
   ``behavior_analysis.outcome_column`` takes precedence over
   ``event_columns.outcome``. Under ``correction_method = cluster``,
   cluster-correction failures surface as errors instead of degrading to
   uncorrected output. With ``split_by_condition = true`` a valid condition
   column is required, and any explicit ``selected_bands`` entry must match
   a configured band name — both raise on mismatch.

ERDS Trial Metrics
^^^^^^^^^^^^^^^^^^

.. math::

   \mathrm{ERDS\%} =
   100 \cdot \frac{P_\text{active} - P_\text{base}}{P_\text{base}}, \qquad
   \mathrm{ERDS}_z =
   \frac{P_\text{active} - P_\text{base}}{\sigma_\text{base}}.

Stage 9 — Cluster Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

Cluster-mass permutation test over time–frequency maps:

.. math::

   M_c = \sum_{i \in c} |t_i|, \qquad
   p_c = \frac{\#\{M_\text{max}^\text{perm} \ge M_c\} + 1}{n_\text{perm} + 1}.

.. note::

   Cluster-stage configuration is strict. An explicit
   ``cluster.condition_column`` must exist as an aligned-events column — no
   fallback to ``event_columns.condition`` or ``event_columns.binary_outcome``.
   ``cluster.condition_values``, when set, must contain exactly two values;
   when empty, the observed binary contrast is inferred from the resolved
   condition column instead of being assumed as ``0`` vs ``1``. Cluster
   tests require complete condition labels and do not silently drop
   unlabeled trials.

Group-Level Analysis
--------------------

Group-level computations run outside the per-subject DAG via
``BehaviorPipeline.run_group_level(...)``. The current implementation exposes
**multilevel correlations only** — the repository does not ship a separate
behavioral ``MixedLM`` stage.

For each feature, the pipeline computes a within-subject correlation estimate
:math:`r_s` (optionally after within-subject covariate adjustment), aggregates
subjects with equal weight via Fisher :math:`z`-averaging, and uses
subject-restricted or block-restricted trial permutations when configured:

.. math::

   r_\text{group} = \tanh\!\left(\mathrm{mean}_s\bigl[\mathrm{atanh}(r_s)\bigr]\right).

Multiple Comparison Correction
-------------------------------

Benjamini-Hochberg (BH)
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   q_{(i)} = \min_{j \ge i} \frac{m}{j}\, p_{(j)}.

Hierarchical FDR
~~~~~~~~~~~~~~~~

Families of hypotheses are gated by a Simes family-level test before
within-family corrections are applied:

.. math::

   p_\text{Simes} = \min_i \frac{m_f}{i}\, p_{(i,\text{family})}.

Within-family rejections are retained only when the family gate rejects at :math:`\alpha`.
