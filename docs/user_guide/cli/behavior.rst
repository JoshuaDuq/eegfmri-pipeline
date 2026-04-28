Behavioral Analysis
===================

Statistical analyses linking EEG features to behavioral outcomes at the
trial level. All joins between features and behavioral targets use the
canonical ``trial_id`` from ``proc-clean_events.tsv``.

.. code-block:: bash

   eeg-pipeline behavior [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``compute``
     - Run behavioral analysis stages and write numerical outputs.
   * - ``visualize``
     - Render standardized plots from previously computed results.

Computations
------------

Select individual stages with ``--computations``. Stages run in dependency
order regardless of the order you specify them.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Computation
     - What it produces
   * - ``trial_table``
     - Merges ``proc-clean_events.tsv`` with feature Parquet tables on
       ``trial_id``. Required by all subsequent stages.
   * - ``predictor_residual``
     - Residualizes the behavioral outcome on a continuous predictor using
       spline OLS (AIC-selected degrees of freedom) or polynomial regression.
       Isolates "outcome beyond predictor" (e.g., pain beyond intensity).
       Optional cross-fit residuals via GroupKFold.
   * - ``correlations``
     - Partial Spearman (or Pearson) correlations between each EEG feature
       and the behavioral outcome, controlling for the predictor variable.
       Includes permutation p-values (circular-shift scheme), effect sizes,
       LOSO stability, and optional Bayes factors (JZS BF10).
   * - ``regression``
     - Trial-wise OLS regression of EEG feature on outcome with run/block
       and predictor controls, incremental ΔR², and HC3 standard errors.
       Optional predictor × feature interaction term.
   * - ``icc``
     - ICC(3,1) intra-class correlation for run-level test–retest reliability
       of each EEG feature.
   * - ``condition``
     - Between-condition Welch t-test, Mann–Whitney U, or Wilcoxon
       signed-rank; reports Cohen's d, Hedges' g, and paired dz.
   * - ``temporal``
     - Time-resolved correlations between TFR or power/ITPC/ERDS and the
       behavioral outcome, with temporal cluster permutation correction.
   * - ``cluster``
     - Cluster permutation tests on epoch-level data.
   * - ``multilevel_correlations``
     - Cross-subject multilevel correlations with block-aware permutations.

Multiple comparison correction uses Benjamini–Hochberg FDR within analyses
and hierarchical Simes gating across analysis families when
``behavior_analysis.validation.enabled = true``.

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 38 42 20

   * - Option
     - Description
     - Default
   * - ``--computations``
     - Space-separated list of stages to run (see table above)
     - all enabled stages from config
   * - ``--predictor-control`` / ``--no-predictor-control``
     - Enable or disable predictor control in correlation analyses
     - config ``behavior_analysis.predictor_control_enabled``
   * - ``--predictor-column``
     - Explicit predictor column; empty config auto-resolves from
       ``event_columns.predictor`` aliases
     - config ``behavior_analysis.predictor_column``
   * - ``--outcome-column``
     - Explicit behavioral outcome column; empty config auto-resolves from
       ``event_columns.outcome`` aliases
     - config ``behavior_analysis.outcome_column``
   * - ``--n-perm``
     - Number of permutations for non-parametric inference
     - config ``behavior_analysis.statistics.n_permutations`` (1000)
   * - ``--robust-correlation``
     - Use a robust correlation estimator:
       ``percentage_bend``, ``winsorized``, or ``shepherd``
     - disabled (Spearman)
   * - ``--compute-bayes-factors``
     - Compute JZS BF₁₀ alongside classical p-values
     - disabled
   * - ``--loso-stability``
     - Compute leave-one-subject-out stability of feature–behavior correlations
     - config ``behavior_analysis.correlations.loso_stability``
   * - ``--predictor-residual-crossfit``
     - Use GroupKFold cross-fitting for predictor residualization (reduces
       overfitting of the nuisance model)
     - disabled
   * - ``--list-stages``
     - Print the resolved stage DAG and exit
     - —

Examples
--------

.. code-block:: bash

   # Full analysis suite for a single subject
   eeg-pipeline behavior compute --subject 0001

   # All subjects with selected stages
   eeg-pipeline behavior compute --all-subjects \
     --computations correlations condition temporal

   # Control for stimulus intensity; permutation testing
   eeg-pipeline behavior compute --subject 0001 \
     --computations correlations predictor_residual \
     --predictor-control --n-perm 5000

   # Bayes factors and robust correlations
   eeg-pipeline behavior compute --subject 0001 \
     --computations correlations \
     --robust-correlation percentage_bend \
     --compute-bayes-factors

   # LOSO stability (requires >= 3 subjects)
   eeg-pipeline behavior compute \
     --subject 0001 --subject 0002 --subject 0003 \
     --computations correlations --loso-stability

   # ICC reliability only
   eeg-pipeline behavior compute --subject 0001 --computations icc

   # Inspect the stage DAG without running
   eeg-pipeline behavior compute --subject 0001 --list-stages

   # Visualize from existing results
   eeg-pipeline behavior visualize --subject 0001

.. seealso::

   :doc:`../../methods/eeg/behavior`
      Full DAG, partial correlation formulas, permutation scheme, and FDR details.

   :doc:`../data_layout`
      ``events.tsv`` column requirements for predictor and outcome aliases.

   :doc:`index`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.
