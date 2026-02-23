package wizard

import (
	"fmt"
	"strings"
)

// Behavior pipeline advanced argument builder.

func (m Model) buildBehaviorAdvancedArgs() []string {
	var args []string
	appendBoolPair := func(enabled bool, onFlag, offFlag string) {
		if enabled {
			args = append(args, onFlag)
		} else {
			args = append(args, offFlag)
		}
	}
	appendFeatureSpec := func(flag, spec string) {
		spec = strings.TrimSpace(spec)
		if spec == "" {
			return
		}
		args = append(args, flag)
		args = append(args, splitCSVList(spec)...)
	}

	// General / statistics
	if m.correlationMethod != "spearman" {
		args = append(args, "--correlation-method", m.correlationMethod)
	}

	robustMethods := []string{"none", "percentage_bend", "winsorized", "shepherd"}
	if m.robustCorrelation > 0 && m.robustCorrelation < len(robustMethods) {
		args = append(args, "--robust-correlation", robustMethods[m.robustCorrelation])
	}

	if m.bootstrapSamples != 1000 {
		args = append(args, "--bootstrap", fmt.Sprintf("%d", m.bootstrapSamples))
	}

	if m.nPermutations != 1000 {
		args = append(args, "--n-perm", fmt.Sprintf("%d", m.nPermutations))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	if m.behaviorNJobs != -1 {
		args = append(args, "--n-jobs", fmt.Sprintf("%d", m.behaviorNJobs))
	}
	if m.behaviorMinSamples > 0 {
		args = append(args, "--min-samples", fmt.Sprintf("%d", m.behaviorMinSamples))
	}
	if col := strings.TrimSpace(m.behaviorOutcomeColumn); col != "" {
		args = append(args, "--outcome-column", col)
	}
	if col := strings.TrimSpace(m.behaviorPredictorColumn); col != "" {
		args = append(args, "--predictor-column", col)
	}

	appendBoolPair(m.controlTemperature, "--control-temperature", "--no-control-temperature")
	appendBoolPair(m.controlTrialOrder, "--control-trial-order", "--no-control-trial-order")

	// Run adjustment (subject-level; optional)
	appendBoolPair(m.runAdjustmentEnabled, "--run-adjustment", "--no-run-adjustment")
	if m.runAdjustmentEnabled {
		col := strings.TrimSpace(m.runAdjustmentColumn)
		if col != "" && col != "run_id" {
			args = append(args, "--run-adjustment-column", col)
		}
		appendBoolPair(
			m.runAdjustmentIncludeInCorrelations,
			"--run-adjustment-include-in-correlations",
			"--no-run-adjustment-include-in-correlations",
		)
		if m.runAdjustmentMaxDummies != 20 {
			args = append(args, "--run-adjustment-max-dummies", fmt.Sprintf("%d", m.runAdjustmentMaxDummies))
		}
	}

	if m.fdrAlpha != 0.05 {
		args = append(args, "--fdr-alpha", fmt.Sprintf("%.4f", m.fdrAlpha))
	}

	appendBoolPair(m.behaviorComputeChangeScores, "--compute-change-scores", "--no-compute-change-scores")
	appendBoolPair(m.behaviorComputeLosoStability, "--loso-stability", "--no-loso-stability")
	appendBoolPair(m.behaviorComputeBayesFactors, "--compute-bayes-factors", "--no-compute-bayes-factors")
	if m.behaviorValidateOnly {
		args = append(args, "--validate-only")
	}

	// Output options
	if !m.behaviorOverwrite {
		args = append(args, "--no-overwrite")
	}

	// Trial table / pain residual / diagnostics
	if m.isComputationSelected("trial_table") {
		formats := []string{"parquet", "tsv"}
		if m.trialTableFormat >= 0 && m.trialTableFormat < len(formats) && m.trialTableFormat != 0 {
			args = append(args, "--trial-table-format", formats[m.trialTableFormat])
		}
		appendBoolPair(m.trialTableAddLagFeatures, "--trial-table-add-lag-features", "--no-trial-table-add-lag-features")
		if m.trialOrderMaxMissingFraction != 0.1 {
			args = append(args, "--trial-order-max-missing-fraction", fmt.Sprintf("%.3f", m.trialOrderMaxMissingFraction))
		}

		appendBoolPair(m.featureSummariesEnabled, "--feature-summaries", "--no-feature-summaries")

		appendBoolPair(m.painResidualEnabled, "--pain-residual", "--no-pain-residual")
		if m.painResidualEnabled {
			methods := []string{"spline", "poly"}
			if m.painResidualMethod >= 0 && m.painResidualMethod < len(methods) && m.painResidualMethod != 0 {
				args = append(args, "--pain-residual-method", methods[m.painResidualMethod])
			}
			if m.painResidualMinSamples != 10 {
				args = append(args, "--pain-residual-min-samples", fmt.Sprintf("%d", m.painResidualMinSamples))
			}
			if m.painResidualPolyDegree != 2 {
				args = append(args, "--pain-residual-poly-degree", fmt.Sprintf("%d", m.painResidualPolyDegree))
			}
			if strings.TrimSpace(m.painResidualSplineDfCandidates) != "" && m.painResidualSplineDfCandidates != "3,4,5" {
				args = append(args, "--pain-residual-spline-df-candidates")
				args = append(args, splitCSVList(m.painResidualSplineDfCandidates)...)
			}
		}

		appendBoolPair(
			m.painResidualModelCompareEnabled,
			"--pain-residual-model-compare",
			"--no-pain-residual-model-compare",
		)
		if m.painResidualModelCompareMinSamples != 10 {
			args = append(args, "--pain-residual-model-compare-min-samples", fmt.Sprintf("%d", m.painResidualModelCompareMinSamples))
		}
		if strings.TrimSpace(m.painResidualModelComparePolyDegrees) != "" && m.painResidualModelComparePolyDegrees != "2,3" {
			args = append(args, "--pain-residual-model-compare-poly-degrees")
			args = append(args, splitCSVList(m.painResidualModelComparePolyDegrees)...)
		}
		appendBoolPair(
			m.painResidualBreakpointEnabled,
			"--pain-residual-breakpoint-test",
			"--no-pain-residual-breakpoint-test",
		)
		if m.painResidualBreakpointMinSamples != 12 {
			args = append(args, "--pain-residual-breakpoint-min-samples", fmt.Sprintf("%d", m.painResidualBreakpointMinSamples))
		}
		if m.painResidualBreakpointCandidates != 15 {
			args = append(args, "--pain-residual-breakpoint-candidates", fmt.Sprintf("%d", m.painResidualBreakpointCandidates))
		}
		if m.painResidualBreakpointQlow != 0.15 {
			args = append(args, "--pain-residual-breakpoint-quantile-low", fmt.Sprintf("%.3f", m.painResidualBreakpointQlow))
		}
		if m.painResidualBreakpointQhigh != 0.85 {
			args = append(args, "--pain-residual-breakpoint-quantile-high", fmt.Sprintf("%.3f", m.painResidualBreakpointQhigh))
		}

		if m.painResidualEnabled {
			appendBoolPair(m.painResidualCrossfitEnabled, "--pain-residual-crossfit", "--no-pain-residual-crossfit")
		}
		if m.painResidualEnabled && m.painResidualCrossfitEnabled {
			args = append(args, "--pain-residual-crossfit")
			if strings.TrimSpace(m.painResidualCrossfitGroupColumn) != "" {
				args = append(args, "--pain-residual-crossfit-group-column", strings.TrimSpace(m.painResidualCrossfitGroupColumn))
			}
			if m.painResidualCrossfitNSplits != 5 {
				args = append(args, "--pain-residual-crossfit-n-splits", fmt.Sprintf("%d", m.painResidualCrossfitNSplits))
			}
			cfMethods := []string{"spline", "poly"}
			if m.painResidualCrossfitMethod >= 0 && m.painResidualCrossfitMethod < len(cfMethods) && m.painResidualCrossfitMethod != 0 {
				args = append(args, "--pain-residual-crossfit-method", cfMethods[m.painResidualCrossfitMethod])
			}
			if m.painResidualCrossfitMethod == 0 && m.painResidualCrossfitSplineKnots != 5 {
				args = append(args, "--pain-residual-crossfit-spline-n-knots", fmt.Sprintf("%d", m.painResidualCrossfitSplineKnots))
			}
		}
	}

	// Feature QC (optional gating)
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		if m.featureQCEnabled {
			args = append(args, "--feature-qc-enabled")
			if m.featureQCMaxMissingPct != 0.2 {
				args = append(args, "--feature-qc-max-missing-pct", fmt.Sprintf("%.3f", m.featureQCMaxMissingPct))
			}
			if m.featureQCMinVariance != 1e-10 {
				args = append(args, "--feature-qc-min-variance", fmt.Sprintf("%.6e", m.featureQCMinVariance))
			}
			appendBoolPair(
				m.featureQCCheckWithinRunVariance,
				"--feature-qc-check-within-run-variance",
				"--no-feature-qc-check-within-run-variance",
			)
		} else {
			args = append(args, "--no-feature-qc-enabled")
		}
	}

	// Regression
	if m.isComputationSelected("regression") {
		outcomes := []string{"rating", "pain_residual", "temperature"}
		if m.regressionOutcome >= 0 && m.regressionOutcome < len(outcomes) && m.regressionOutcome != 0 {
			args = append(args, "--regression-outcome", outcomes[m.regressionOutcome])
		}
		appendBoolPair(
			m.regressionIncludeTemperature,
			"--regression-include-temperature",
			"--no-regression-include-temperature",
		)
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.regressionTempControl >= 0 && m.regressionTempControl < len(tempCtrl) && m.regressionTempControl != 0 {
			args = append(args, "--regression-temperature-control", tempCtrl[m.regressionTempControl])
		}
		if m.regressionTempControl == 2 {
			args = append(args, "--regression-temperature-spline-knots", fmt.Sprintf("%d", m.regressionTempSplineKnots))
			args = append(args, "--regression-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.regressionTempSplineQlow))
			args = append(args, "--regression-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.regressionTempSplineQhigh))
			if m.regressionTempSplineMinN != 12 {
				args = append(args, "--regression-temperature-spline-min-samples", fmt.Sprintf("%d", m.regressionTempSplineMinN))
			}
		}
		appendBoolPair(
			m.regressionIncludeTrialOrder,
			"--regression-include-trial-order",
			"--no-regression-include-trial-order",
		)
		appendBoolPair(
			m.regressionIncludePrev,
			"--regression-include-prev-terms",
			"--no-regression-include-prev-terms",
		)
		appendBoolPair(
			m.regressionIncludeRunBlock,
			"--regression-include-run-block",
			"--no-regression-include-run-block",
		)
		appendBoolPair(
			m.regressionIncludeInteraction,
			"--regression-include-interaction",
			"--no-regression-include-interaction",
		)
		appendBoolPair(
			m.regressionStandardize,
			"--regression-standardize",
			"--no-regression-standardize",
		)
		if m.regressionMinSamples != 15 {
			args = append(args, "--regression-min-samples", fmt.Sprintf("%d", m.regressionMinSamples))
		}
		if m.regressionPrimaryUnit == 1 {
			args = append(args, "--regression-primary-unit", "run_mean")
		}
		if m.regressionPermutations != 0 {
			args = append(args, "--regression-permutations", fmt.Sprintf("%d", m.regressionPermutations))
		}
		if m.regressionMaxFeatures != 0 {
			args = append(args, "--regression-max-features", fmt.Sprintf("%d", m.regressionMaxFeatures))
		}
	}

	// Models
	if m.isComputationSelected("models") {
		appendBoolPair(
			m.modelsIncludeTemperature,
			"--models-include-temperature",
			"--no-models-include-temperature",
		)
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.modelsTempControl >= 0 && m.modelsTempControl < len(tempCtrl) && m.modelsTempControl != 0 {
			args = append(args, "--models-temperature-control", tempCtrl[m.modelsTempControl])
		}
		if m.modelsTempControl == 2 {
			args = append(args, "--models-temperature-spline-knots", fmt.Sprintf("%d", m.modelsTempSplineKnots))
			args = append(args, "--models-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.modelsTempSplineQlow))
			args = append(args, "--models-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.modelsTempSplineQhigh))
			if m.modelsTempSplineMinN != 12 {
				args = append(args, "--models-temperature-spline-min-samples", fmt.Sprintf("%d", m.modelsTempSplineMinN))
			}
		}
		appendBoolPair(
			m.modelsIncludeTrialOrder,
			"--models-include-trial-order",
			"--no-models-include-trial-order",
		)
		appendBoolPair(
			m.modelsIncludePrev,
			"--models-include-prev-terms",
			"--no-models-include-prev-terms",
		)
		appendBoolPair(
			m.modelsIncludeRunBlock,
			"--models-include-run-block",
			"--no-models-include-run-block",
		)
		appendBoolPair(
			m.modelsIncludeInteraction,
			"--models-include-interaction",
			"--no-models-include-interaction",
		)
		appendBoolPair(
			m.modelsStandardize,
			"--models-standardize",
			"--no-models-standardize",
		)
		if m.modelsMinSamples != 20 {
			args = append(args, "--models-min-samples", fmt.Sprintf("%d", m.modelsMinSamples))
		}
		if m.modelsMaxFeatures != 100 {
			args = append(args, "--models-max-features", fmt.Sprintf("%d", m.modelsMaxFeatures))
		}
		out := []string{}
		if m.modelsOutcomeRating {
			out = append(out, "rating")
		}
		if m.modelsOutcomePainResidual {
			out = append(out, "pain_residual")
		}
		if m.modelsOutcomeTemperature {
			out = append(out, "temperature")
		}
		if m.modelsOutcomePainBinary {
			out = append(out, "pain_binary")
		}
		if len(out) > 0 && !(len(out) == 2 && out[0] == "rating" && out[1] == "pain_residual") {
			args = append(args, "--models-outcomes")
			args = append(args, out...)
		}
		fams := []string{}
		if m.modelsFamilyOLS {
			fams = append(fams, "ols_hc3")
		}
		if m.modelsFamilyRobust {
			fams = append(fams, "robust_rlm")
		}
		if m.modelsFamilyQuantile {
			fams = append(fams, "quantile_50")
		}
		if m.modelsFamilyLogit {
			fams = append(fams, "logit")
		}
		if len(fams) > 0 && len(fams) < 4 {
			args = append(args, "--models-families")
			args = append(args, fams...)
		}
		binOut := []string{"pain_binary", "rating_median"}
		if m.modelsBinaryOutcome >= 0 && m.modelsBinaryOutcome < len(binOut) && m.modelsBinaryOutcome != 0 {
			args = append(args, "--models-binary-outcome", binOut[m.modelsBinaryOutcome])
		}
		if m.modelsPrimaryUnit == 1 {
			args = append(args, "--models-primary-unit", "run_mean")
		}
		appendBoolPair(
			m.modelsForceTrialIIDAsymptotic,
			"--models-force-trial-iid-asymptotic",
			"--no-models-force-trial-iid-asymptotic",
		)
	}

	// Stability
	if m.isComputationSelected("stability") {
		if m.stabilityMethod == 1 {
			args = append(args, "--stability-method", "pearson")
		}
		outcome := []string{"auto", "rating", "pain_residual"}
		if m.stabilityOutcome > 0 && m.stabilityOutcome < len(outcome) {
			args = append(args, "--stability-outcome", outcome[m.stabilityOutcome])
		}
		groupCol := []string{"auto", "run", "block"}
		if m.stabilityGroupColumn > 0 && m.stabilityGroupColumn < len(groupCol) {
			args = append(args, "--stability-group-column", groupCol[m.stabilityGroupColumn])
		}
		appendBoolPair(
			m.stabilityPartialTemp,
			"--stability-partial-temperature",
			"--no-stability-partial-temperature",
		)
		if m.stabilityMaxFeatures != 50 {
			args = append(args, "--stability-max-features", fmt.Sprintf("%d", m.stabilityMaxFeatures))
		}
		if m.stabilityAlpha != 0.05 {
			args = append(args, "--stability-alpha", fmt.Sprintf("%.4f", m.stabilityAlpha))
		}
		if m.stabilityMinGroupN > 0 {
			args = append(args, "--stability-min-group-trials", fmt.Sprintf("%d", m.stabilityMinGroupN))
		}
	}

	// Consistency
	if m.isComputationSelected("consistency") {
		appendBoolPair(m.consistencyEnabled, "--consistency", "--no-consistency")
	}

	// Influence
	if m.isComputationSelected("influence") {
		out := []string{}
		if m.influenceOutcomeRating {
			out = append(out, "rating")
		}
		if m.influenceOutcomePainResidual {
			out = append(out, "pain_residual")
		}
		if m.influenceOutcomeTemperature {
			out = append(out, "temperature")
		}
		if len(out) > 0 && !(len(out) == 2 && out[0] == "rating" && out[1] == "pain_residual") {
			args = append(args, "--influence-outcomes")
			args = append(args, out...)
		}
		if m.influenceMaxFeatures != 20 {
			args = append(args, "--influence-max-features", fmt.Sprintf("%d", m.influenceMaxFeatures))
		}
		appendBoolPair(
			m.influenceIncludeTemperature,
			"--influence-include-temperature",
			"--no-influence-include-temperature",
		)
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.influenceTempControl >= 0 && m.influenceTempControl < len(tempCtrl) && m.influenceTempControl != 0 {
			args = append(args, "--influence-temperature-control", tempCtrl[m.influenceTempControl])
		}
		if m.influenceTempControl == 2 {
			args = append(args, "--influence-temperature-spline-knots", fmt.Sprintf("%d", m.influenceTempSplineKnots))
			args = append(args, "--influence-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.influenceTempSplineQlow))
			args = append(args, "--influence-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.influenceTempSplineQhigh))
			if m.influenceTempSplineMinN != 12 {
				args = append(args, "--influence-temperature-spline-min-samples", fmt.Sprintf("%d", m.influenceTempSplineMinN))
			}
		}
		appendBoolPair(
			m.influenceIncludeTrialOrder,
			"--influence-include-trial-order",
			"--no-influence-include-trial-order",
		)
		appendBoolPair(
			m.influenceIncludeRunBlock,
			"--influence-include-run-block",
			"--no-influence-include-run-block",
		)
		appendBoolPair(
			m.influenceIncludeInteraction,
			"--influence-include-interaction",
			"--no-influence-include-interaction",
		)
		appendBoolPair(
			m.influenceStandardize,
			"--influence-standardize",
			"--no-influence-standardize",
		)
		if m.influenceCooksThreshold > 0 {
			args = append(args, "--influence-cooks-threshold", fmt.Sprintf("%.6f", m.influenceCooksThreshold))
		}
		if m.influenceLeverageThreshold > 0 {
			args = append(args, "--influence-leverage-threshold", fmt.Sprintf("%.6f", m.influenceLeverageThreshold))
		}
	}

	// Correlations (trial-table)
	if m.isComputationSelected("correlations") {
		appendFeatureSpec("--correlations-features", m.correlationsFeaturesSpec)
		if strings.TrimSpace(m.correlationsTypesSpec) != "" && m.correlationsTypesSpec != "partial_cov_temp" {
			args = append(args, "--correlations-types")
			args = append(args, splitCSVList(m.correlationsTypesSpec)...)
		}
		if m.correlationsPrimaryUnit == 1 {
			args = append(args, "--correlations-primary-unit", "run_mean")
		}
		if m.correlationsMinRuns != 3 {
			args = append(args, "--correlations-min-runs", fmt.Sprintf("%d", m.correlationsMinRuns))
		}
		appendBoolPair(
			m.correlationsPreferPainResidual,
			"--correlations-prefer-pain-residual",
			"--no-correlations-prefer-pain-residual",
		)
		if m.correlationsPermutations > 0 {
			args = append(args, "--correlations-permutations", fmt.Sprintf("%d", m.correlationsPermutations))
		}
		appendBoolPair(
			m.correlationsPermutationPrimary,
			"--correlations-permutation-primary",
			"--no-correlations-permutation-primary",
		)
		appendBoolPair(
			m.correlationsUseCrossfitResidual,
			"--correlations-use-crossfit-pain-residual",
			"--no-correlations-use-crossfit-pain-residual",
		)
		// Always pass explicit target selection (possibly empty) so the backend
		// does not silently fall back to built-in defaults.
		args = append(args, "--correlations-target-column", strings.TrimSpace(m.correlationsTargetColumn))
	}

	// Multilevel correlations (group-level)
	if m.isComputationSelected("multilevel_correlations") {
		appendBoolPair(
			m.groupLevelBlockPermutation,
			"--group-level-block-permutation",
			"--no-group-level-block-permutation",
		)
		if col := strings.TrimSpace(m.groupLevelTarget); col != "" {
			args = append(args, "--group-level-target", col)
		}
		appendBoolPair(
			m.groupLevelControlTemperature,
			"--group-level-control-temperature",
			"--no-group-level-control-temperature",
		)
		appendBoolPair(
			m.groupLevelControlTrialOrder,
			"--group-level-control-trial-order",
			"--no-group-level-control-trial-order",
		)
		appendBoolPair(
			m.groupLevelControlRunEffects,
			"--group-level-control-run-effects",
			"--no-group-level-control-run-effects",
		)
		if m.groupLevelMaxRunDummies != 20 {
			args = append(args, "--group-level-max-run-dummies", fmt.Sprintf("%d", m.groupLevelMaxRunDummies))
		}
		appendBoolPair(
			m.groupLevelAllowParametricFallback,
			"--group-level-allow-parametric-fallback",
			"--no-group-level-allow-parametric-fallback",
		)
	}

	// Report
	if m.isComputationSelected("report") && m.reportTopN != 15 {
		args = append(args, "--report-top-n", fmt.Sprintf("%d", m.reportTopN))
	}

	// Pain sensitivity
	if m.isComputationSelected("pain_sensitivity") {
		appendFeatureSpec("--pain-sensitivity-features", m.painSensitivityFeaturesSpec)
		if m.painSensitivityMinTrials > 0 {
			args = append(args, "--pain-sensitivity-min-trials", fmt.Sprintf("%d", m.painSensitivityMinTrials))
		}
		if m.painSensitivityPrimaryUnit == 1 {
			args = append(args, "--pain-sensitivity-primary-unit", "run_mean")
		}
		if m.painSensitivityPermutations > 0 {
			args = append(args, "--pain-sensitivity-permutations", fmt.Sprintf("%d", m.painSensitivityPermutations))
		}
		appendBoolPair(
			m.painSensitivityPermutationPrimary,
			"--pain-sensitivity-permutation-primary",
			"--no-pain-sensitivity-permutation-primary",
		)
	}

	// Condition
	if m.isComputationSelected("condition") {
		appendFeatureSpec("--condition-features", m.conditionFeaturesSpec)
		if strings.TrimSpace(m.conditionCompareColumn) != "" {
			args = append(args, "--condition-compare-column", strings.TrimSpace(m.conditionCompareColumn))
		}
		if strings.TrimSpace(m.conditionCompareValues) != "" {
			args = append(args, "--condition-compare-values")
			args = append(args, splitCSVList(m.conditionCompareValues)...)
		}
		if strings.TrimSpace(m.conditionCompareLabels) != "" {
			args = append(args, "--condition-compare-labels")
			args = append(args, splitCSVList(m.conditionCompareLabels)...)
		}
		if strings.TrimSpace(m.conditionCompareWindows) != "" {
			args = append(args, "--condition-compare-windows")
			args = append(args, splitSpaceList(m.conditionCompareWindows)...)
		}
		if m.conditionMinTrials > 0 {
			args = append(args, "--condition-min-trials", fmt.Sprintf("%d", m.conditionMinTrials))
		}
		if m.conditionPrimaryUnit == 1 {
			args = append(args, "--condition-primary-unit", "run_mean")
		}
		if m.conditionWindowPrimaryUnit == 1 {
			args = append(args, "--condition-window-primary-unit", "run_mean")
		}
		if m.conditionWindowMinSamples != 10 {
			args = append(args, "--condition-window-min-samples", fmt.Sprintf("%d", m.conditionWindowMinSamples))
		}
		appendBoolPair(
			m.conditionPermutationPrimary,
			"--condition-permutation-primary",
			"--no-condition-permutation-primary",
		)
		appendBoolPair(m.conditionFailFast, "--condition-fail-fast", "--no-condition-fail-fast")
		if m.conditionEffectThreshold != 0.5 {
			args = append(args, "--condition-effect-threshold", fmt.Sprintf("%.4f", m.conditionEffectThreshold))
		}
		if m.conditionOverwrite {
			args = append(args, "--condition-overwrite")
		} else {
			args = append(args, "--no-condition-overwrite")
		}
	}

	// Temporal
	if m.isComputationSelected("temporal") {
		appendFeatureSpec("--temporal-features", m.temporalFeaturesSpec)
		if m.temporalResolutionMs != 50 {
			args = append(args, "--temporal-time-resolution-ms", fmt.Sprintf("%d", m.temporalResolutionMs))
		}
		if m.temporalCorrectionMethod == 1 {
			args = append(args, "--temporal-correction-method", "cluster")
		}
		if m.temporalTimeMinMs != -200 {
			args = append(args, "--temporal-time-min-ms", fmt.Sprintf("%d", m.temporalTimeMinMs))
		}
		if m.temporalTimeMaxMs != 1000 {
			args = append(args, "--temporal-time-max-ms", fmt.Sprintf("%d", m.temporalTimeMaxMs))
		}
		if m.temporalSmoothMs != 100 {
			args = append(args, "--temporal-smooth-window-ms", fmt.Sprintf("%d", m.temporalSmoothMs))
		}
		if strings.TrimSpace(m.temporalTargetColumn) != "" {
			args = append(args, "--temporal-target-column", strings.TrimSpace(m.temporalTargetColumn))
		}
		appendBoolPair(
			m.temporalSplitByCondition,
			"--temporal-split-by-condition",
			"--no-temporal-split-by-condition",
		)
		if strings.TrimSpace(m.temporalConditionColumn) != "" {
			args = append(args, "--temporal-condition-column", strings.TrimSpace(m.temporalConditionColumn))
		}
		if strings.TrimSpace(m.temporalConditionValues) != "" {
			args = append(args, "--temporal-condition-values")
			spec := strings.ReplaceAll(m.temporalConditionValues, ",", " ")
			args = append(args, splitSpaceList(spec)...)
		}
		appendBoolPair(
			m.temporalIncludeROIAverages,
			"--temporal-include-roi-averages",
			"--no-temporal-include-roi-averages",
		)
		appendBoolPair(
			m.temporalIncludeTFGrid,
			"--temporal-include-tf-grid",
			"--no-temporal-include-tf-grid",
		)
		// Temporal feature selection
		appendBoolPair(
			m.temporalFeaturePowerEnabled,
			"--temporal-feature-power",
			"--no-temporal-feature-power",
		)
		appendBoolPair(
			m.temporalFeatureITPCEnabled,
			"--temporal-feature-itpc",
			"--no-temporal-feature-itpc",
		)
		appendBoolPair(
			m.temporalFeatureERDSEnabled,
			"--temporal-feature-erds",
			"--no-temporal-feature-erds",
		)
		// ITPC-specific options (only if ITPC is selected in step 3)
		if m.featureFileSelected["itpc"] || m.temporalFeatureITPCEnabled {
			appendBoolPair(
				m.temporalITPCBaselineCorrection,
				"--temporal-itpc-baseline-correction",
				"--no-temporal-itpc-baseline-correction",
			)
			if m.temporalITPCBaselineMin != -0.5 {
				args = append(args, "--temporal-itpc-baseline-min", fmt.Sprintf("%.2f", m.temporalITPCBaselineMin))
			}
			if m.temporalITPCBaselineMax != -0.01 {
				args = append(args, "--temporal-itpc-baseline-max", fmt.Sprintf("%.2f", m.temporalITPCBaselineMax))
			}
		}
		// ERDS-specific options (only if ERDS is selected in step 3)
		if m.featureFileSelected["erds"] || m.temporalFeatureERDSEnabled {
			if m.temporalERDSBaselineMin != -0.5 {
				args = append(args, "--temporal-erds-baseline-min", fmt.Sprintf("%.2f", m.temporalERDSBaselineMin))
			}
			if m.temporalERDSBaselineMax != -0.1 {
				args = append(args, "--temporal-erds-baseline-max", fmt.Sprintf("%.2f", m.temporalERDSBaselineMax))
			}
			if m.temporalERDSMethod != 0 {
				methods := []string{"percent", "zscore"}
				args = append(args, "--temporal-erds-method", methods[m.temporalERDSMethod])
			}
		}
	}

	// Cluster-specific options
	if m.isComputationSelected("cluster") {
		appendFeatureSpec("--cluster-features", m.clusterFeaturesSpec)
		if m.clusterThreshold != 0.05 {
			args = append(args, "--cluster-threshold", fmt.Sprintf("%.3f", m.clusterThreshold))
		}
		if m.clusterMinSize != 2 {
			args = append(args, "--cluster-min-size", fmt.Sprintf("%d", m.clusterMinSize))
		}
		if m.clusterTail != 0 {
			args = append(args, "--cluster-tail", fmt.Sprintf("%d", m.clusterTail))
		}
		if strings.TrimSpace(m.clusterConditionColumn) != "" {
			args = append(args, "--cluster-condition-column", strings.TrimSpace(m.clusterConditionColumn))
		}
		if strings.TrimSpace(m.clusterConditionValues) != "" {
			args = append(args, "--cluster-condition-values")
			spec := strings.ReplaceAll(m.clusterConditionValues, ",", " ")
			args = append(args, splitSpaceList(spec)...)
		}
	}

	// Mediation-specific options
	if m.isComputationSelected("mediation") {
		appendFeatureSpec("--mediation-features", m.mediationFeaturesSpec)
		if m.mediationBootstrap != 1000 {
			args = append(args, "--mediation-bootstrap", fmt.Sprintf("%d", m.mediationBootstrap))
		}
		if m.mediationPermutations > 0 {
			args = append(args, "--mediation-permutations", fmt.Sprintf("%d", m.mediationPermutations))
		}
		appendBoolPair(
			m.mediationPermutationPrimary,
			"--mediation-permutation-primary",
			"--no-mediation-permutation-primary",
		)
		if m.mediationMinEffect != 0.05 {
			args = append(args, "--mediation-min-effect-size", fmt.Sprintf("%.4f", m.mediationMinEffect))
		}
		if m.mediationMaxMediatorsEnabled {
			if m.mediationMaxMediators != 20 {
				args = append(args, "--mediation-max-mediators", fmt.Sprintf("%d", m.mediationMaxMediators))
			}
		}
	}

	// Moderation-specific options
	if m.isComputationSelected("moderation") {
		appendFeatureSpec("--moderation-features", m.moderationFeaturesSpec)
		if m.moderationMinSamples != 15 {
			args = append(args, "--moderation-min-samples", fmt.Sprintf("%d", m.moderationMinSamples))
		}
		if m.moderationPermutations > 0 {
			args = append(args, "--moderation-permutations", fmt.Sprintf("%d", m.moderationPermutations))
		}
		appendBoolPair(
			m.moderationPermutationPrimary,
			"--moderation-permutation-primary",
			"--no-moderation-permutation-primary",
		)
		if m.moderationMaxFeaturesEnabled {
			if m.moderationMaxFeatures != 50 {
				args = append(args, "--moderation-max-features", fmt.Sprintf("%d", m.moderationMaxFeatures))
			}
		}
	}

	// Mixed effects-specific options
	if m.isComputationSelected("mixed_effects") {
		if m.mixedEffectsType == 1 {
			args = append(args, "--mixed-random-effects", "intercept_slope")
		}
		appendBoolPair(
			m.mixedIncludeTemperature,
			"--mixed-include-temperature",
			"--no-mixed-include-temperature",
		)
		if m.mixedMaxFeatures != 50 {
			args = append(args, "--mixed-max-features", fmt.Sprintf("%d", m.mixedMaxFeatures))
		}
	}

	// Output options
	appendBoolPair(m.alsoSaveCsv, "--also-save-csv", "--no-also-save-csv")

	// Behavior Statistics
	tempControls := []string{"none", "linear", "spline"}
	if m.behaviorStatsTempControl != 0 {
		args = append(args, "--stats-temp-control", tempControls[m.behaviorStatsTempControl%len(tempControls)])
	}
	appendBoolPair(
		m.behaviorStatsAllowIIDTrials,
		"--stats-allow-iid-trials",
		"--no-stats-allow-iid-trials",
	)
	appendBoolPair(
		m.behaviorStatsHierarchicalFDR,
		"--stats-hierarchical-fdr",
		"--no-stats-hierarchical-fdr",
	)
	appendBoolPair(
		m.behaviorStatsComputeReliability,
		"--stats-compute-reliability",
		"--no-stats-compute-reliability",
	)
	permSchemes := []string{"shuffle", "circular_shift"}
	if m.behaviorPermScheme != 0 {
		args = append(args, "--perm-scheme", permSchemes[m.behaviorPermScheme%len(permSchemes)])
	}
	if strings.TrimSpace(m.behaviorPermGroupColumnPreference) != "" {
		args = append(args, "--perm-group-column-preference", strings.TrimSpace(m.behaviorPermGroupColumnPreference))
	}
	appendBoolPair(
		m.behaviorExcludeNonTrialwiseFeatures,
		"--exclude-non-trialwise-features",
		"--no-exclude-non-trialwise-features",
	)

	// Global Statistics & Validation
	if m.globalNBootstrap != 1000 {
		args = append(args, "--global-n-bootstrap", fmt.Sprintf("%d", m.globalNBootstrap))
	}
	if m.clusterCorrectionEnabled {
		args = append(args, "--cluster-correction-enabled")
		if m.clusterCorrectionAlpha != 0.05 {
			args = append(args, "--cluster-correction-alpha", fmt.Sprintf("%.4f", m.clusterCorrectionAlpha))
		}
		if m.clusterCorrectionMinClusterSize != 2 {
			args = append(args, "--cluster-correction-min-cluster-size", fmt.Sprintf("%d", m.clusterCorrectionMinClusterSize))
		}
		tails := []string{"0", "1", "-1"}
		if m.clusterCorrectionTailGlobal != 0 {
			args = append(args, "--cluster-correction-tail", tails[m.clusterCorrectionTailGlobal%len(tails)])
		}
	} else {
		args = append(args, "--no-cluster-correction-enabled")
	}
	if m.validationMinEpochs != 5 {
		args = append(args, "--validation-min-epochs", fmt.Sprintf("%d", m.validationMinEpochs))
	}
	if m.validationMinChannels != 10 {
		args = append(args, "--validation-min-channels", fmt.Sprintf("%d", m.validationMinChannels))
	}
	if m.validationMaxAmplitudeUv != 500.0 {
		args = append(args, "--validation-max-amplitude-uv", fmt.Sprintf("%.1f", m.validationMaxAmplitudeUv))
	}

	// System / IO
	if strings.TrimSpace(m.ioTemperatureRange) != "" {
		args = append(args, "--temperature-range")
		args = append(args, splitCSVList(m.ioTemperatureRange)...)
	}
	if m.ioMaxMissingChannelsFraction != 0.1 {
		args = append(args, "--max-missing-channels-fraction", fmt.Sprintf("%.3f", m.ioMaxMissingChannelsFraction))
	}

	return args
}
