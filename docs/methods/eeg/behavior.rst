Behavioral Statistics
=====================

**Module:** ``eeg_pipeline.analysis.behavior``

.. seealso::

   :doc:`features`
      Produces the EEG feature tables consumed by the behavior pipeline.

   :doc:`machine_learning`
      Uses the same aligned events as inputs for predictive modeling.

   :doc:`../../user_guide/configuration`
      Full ``behavior_config.yaml`` key reference.

   :doc:`../../user_guide/cli/behavior`
      CLI flags for behavior compute and visualize modes.

Methods reference for the behavioral statistics pipeline. Analyses run as a
dependency-resolved DAG over aligned EEG and behavioral data. The pipeline targets
trial-level associations between behavioral variables and EEG-derived features, with
explicit non-i.i.d. safeguards and per-stage outputs.

The pipeline is **paradigm-agnostic**: supports any combination of continuous, binary,
or categorical predictors and any scalar outcome measure.
Multiple comparison correction uses :term:`FDR` throughout; all Fisher-z aggregations
are described in the :term:`Fisher-z` glossary entry.

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 1

   * - Inputs
     - ``*_proc-clean_events.tsv``, feature Parquet tables, ``behavior_config.yaml``
   * - Outputs
     - Per-analysis TSV results tables, correlation matrices, condition comparison tables
   * - CLI
     - ``eeg-pipeline behavior [compute | visualize]``
   * - Config
     - ``behavior_config.yaml``

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

Pipeline DAG
------------

Stages run in dependency order:

.. code-block:: text

   load
   └── trial_table
       ├── predictor_residual          [continuous predictor only]
       ├── correlate_design
       │   ├── correlate_effect_sizes
       │   │   ├── correlate_pvalues
       │   │   │   └── correlate_primary_selection
       │   │   │       └── correlate_fdr
       ├── regression
       ├── icc
       ├── condition_column
       ├── temporal_tfr
       │   └── temporal_stats
       │       └── cluster
       └── hierarchical_fdr_summary
           ├── report
           └── export

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
   * - ``multilevel_correlations``
     - group-level multilevel correlations (outside subject DAG)

Statistical Safeguards
----------------------

Non-i.i.d. Trial Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trials within a subject are clustered within runs/blocks and are not exchangeable.
The following stages enforce grouped permutation unless ``allow_iid_trials = true``
is explicitly set: ``correlate_*``, ``regression``, ``condition_column``.

Predictor Type Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``behavior_analysis.predictor_type`` key declares the nature of the predictor:

- **``continuous``** — ordered numeric scale with ≥ 5 distinct levels. Enables ``predictor_residual`` and spline/outcome_hat control.
- **``binary``** — two-level factor. Disables curve-fitting analyses.
- **``categorical``** — unordered multi-level factor. Same restrictions as binary.

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
   correspondence. The ``trial_id`` column is the only accepted join key.

Stage 3 — Predictor Residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requires:** ``predictor_type = continuous`` (≥ 5 unique predictor values).

Residualizes the outcome on the predictor to isolate variance not explained by
stimulus intensity:

.. math::

   \text{predictor\_residual}_i = y_i - \hat{y}_i, \qquad \hat{y}_i = f(P_i).

Model selection: spline OLS candidates ``outcome ~ bs(predictor, df=d, degree=3)``
for configurable degrees of freedom, lowest-AIC selection; fallback to polynomial.
Optional cross-fit residuals (``GroupKFold``) via ``--predictor-residual-crossfit``.

Stage 4 — Correlations
~~~~~~~~~~~~~~~~~~~~~~~~

**Correlation types:**

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

**Partial correlation** test statistic with :math:`k` covariates:

.. math::

   t = r\sqrt{\frac{n - k - 2}{1 - r^2}}, \qquad
   p = 2\, P\!\left(|T_{n-k-2}| \ge |t|\right).

**Permutation p-value** (Phipson–Smyth):

.. math::

   p_\text{perm} = \frac{N_{\text{extreme}} + 1}{n_\text{perm} + 1}.

**LOSO stability** (``--loso-stability``): recompute correlations on :math:`N-1`
subjects; report mean LOSO :math:`r` and SD across folds.

**Bayes factors** (``--compute-bayes-factors``): JZS :math:`\mathrm{BF}_{10}` approximation alongside classical p-values.

Stage 5 — Regression
~~~~~~~~~~~~~~~~~~~~~~

Per-feature OLS with optional predictor interaction:

.. math::

   y = Z\gamma + \beta_f x_f + \beta_\text{int}(x_f \cdot P) + \varepsilon \quad \text{(full)}.

**Incremental explained variance:**

.. math::

   \Delta R^2 = R^2_\text{full} - R^2_\text{reduced}.

**HC3 heteroskedasticity-consistent standard errors:**

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

**Two-group — Welch t-test:**

.. math::

   t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}.

**Effect sizes:**

.. math::

   d = \frac{\bar{x}_1 - \bar{x}_2}{s_\text{pooled}}, \qquad
   g = d\!\left(1 - \frac{3}{4\,df - 1}\right), \qquad
   d_z = \frac{\bar{d}}{s_d} \text{ (paired)}.

**Multi-group (3+ levels):** pairwise Mann–Whitney U (unpaired) or Wilcoxon
signed-rank (paired). Omnibus tests are not performed.

Stage 8 — Temporal Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Power-bin statistic averaged within time window :math:`w`:

.. math::

   b_{i,f,w} = \mathrm{mean}_{t \in w}\, P_{i,f,t}.

Correlation-to-:math:`t` transform for cluster forming:

.. math::

   t = r\sqrt{\frac{\mathrm{dof}}{1 - r^2}}.

Temporal multiple-comparison correction: ``fdr``, ``bonferroni``, ``cluster``, or ``none``.

**ERDS trial metrics:**

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

Group-Level Analysis
--------------------

Group-level computations run outside the per-subject DAG via
``BehaviorPipeline.run_group_level(...)``.

**Mixed effects models:** feature-wise ``MixedLM`` across subjects with subject
as a random effect, then hierarchical FDR across features.

**Multilevel correlations:** per-subject correlation estimates :math:`r_s` aggregated
via Fisher :math:`z`-averaging:

.. math::

   r_\text{group} = \tanh\!\left(\mathrm{mean}_s\bigl[\mathrm{atanh}(r_s)\bigr]\right).

Multiple Comparison Correction
-------------------------------

**Benjamini–Hochberg (BH):**

.. math::

   q_{(i)} = \min_{j \ge i} \frac{m}{j}\, p_{(j)}.

**Hierarchical FDR (family-gated):** families of hypotheses are gated by a Simes
family-level test before within-family corrections are applied:

.. math::

   p_\text{Simes} = \min_i \frac{m_f}{i}\, p_{(i,\text{family})}.

Within-family rejections are retained only when the family gate rejects at :math:`\alpha`.

.. seealso::

   :doc:`features`
      EEG feature tables consumed as inputs to behavioral correlations.

   :doc:`machine_learning`
      Trial-level predictive modeling using the same behavioral targets.

   :doc:`../../user_guide/data_layout`
      ``events.tsv`` column requirements for predictor and outcome aliases.

   :doc:`../../user_guide/cli/behavior`
      CLI flags for behavioral analysis modes.
