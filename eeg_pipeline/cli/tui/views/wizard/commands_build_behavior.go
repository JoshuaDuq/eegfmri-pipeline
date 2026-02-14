package wizard

import (
	"fmt"
	"strings"
)

// Behavior pipeline advanced argument builder.

func (m Model) buildBehaviorAdvancedArgs() []string {
	var args []string

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

	if !m.controlTemperature {
		args = append(args, "--no-control-temperature")
	}

	if !m.controlTrialOrder {
		args = append(args, "--no-control-trial-order")
	}

	// Run adjustment (subject-level; optional)
	if m.runAdjustmentEnabled {
		args = append(args, "--run-adjustment")
		col := strings.TrimSpace(m.runAdjustmentColumn)
		if col != "" && col != "run_id" {
			args = append(args, "--run-adjustment-column", col)
		}
		if !m.runAdjustmentIncludeInCorrelations {
			args = append(args, "--no-run-adjustment-include-in-correlations")
		}
		if m.runAdjustmentMaxDummies != 20 {
			args = append(args, "--run-adjustment-max-dummies", fmt.Sprintf("%d", m.runAdjustmentMaxDummies))
		}
	}

	if m.fdrAlpha != 0.05 {
		args = append(args, "--fdr-alpha", fmt.Sprintf("%.4f", m.fdrAlpha))
	}

	if !m.behaviorComputeChangeScores {
		args = append(args, "--no-compute-change-scores")
	}
	if !m.behaviorComputeLosoStability {
		args = append(args, "--no-loso-stability")
	}
	if m.behaviorComputeBayesFactors {
		args = append(args, "--compute-bayes-factors")
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
		if !m.trialTableAddLagFeatures {
			args = append(args, "--no-trial-table-add-lag-features")
		}
		if m.trialOrderMaxMissingFraction != 0.1 {
			args = append(args, "--trial-order-max-missing-fraction", fmt.Sprintf("%.3f", m.trialOrderMaxMissingFraction))
		}

		if !m.featureSummariesEnabled {
			args = append(args, "--no-feature-summaries")
		}

		if !m.painResidualEnabled {
			args = append(args, "--no-pain-residual")
		} else {
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

		if !m.painResidualModelCompareEnabled {
			args = append(args, "--no-pain-residual-model-compare")
		}
		if m.painResidualModelCompareMinSamples != 10 {
			args = append(args, "--pain-residual-model-compare-min-samples", fmt.Sprintf("%d", m.painResidualModelCompareMinSamples))
		}
		if strings.TrimSpace(m.painResidualModelComparePolyDegrees) != "" && m.painResidualModelComparePolyDegrees != "2,3" {
			args = append(args, "--pain-residual-model-compare-poly-degrees")
			args = append(args, splitCSVList(m.painResidualModelComparePolyDegrees)...)
		}
		if !m.painResidualBreakpointEnabled {
			args = append(args, "--no-pain-residual-breakpoint-test")
		}
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
			if !m.featureQCCheckWithinRunVariance {
				args = append(args, "--no-feature-qc-check-within-run-variance")
			}
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
		if !m.regressionIncludeTemperature {
			args = append(args, "--no-regression-include-temperature")
		}
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
		if !m.regressionIncludeTrialOrder {
			args = append(args, "--no-regression-include-trial-order")
		}
		if m.regressionIncludePrev {
			args = append(args, "--regression-include-prev-terms")
		}
		if !m.regressionIncludeRunBlock {
			args = append(args, "--no-regression-include-run-block")
		}
		if !m.regressionIncludeInteraction {
			args = append(args, "--no-regression-include-interaction")
		}
		if !m.regressionStandardize {
			args = append(args, "--no-regression-standardize")
		}
		if m.regressionMinSamples != 15 {
			args = append(args, "--regression-min-samples", fmt.Sprintf("%d", m.regressionMinSamples))
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
		if !m.modelsIncludeTemperature {
			args = append(args, "--no-models-include-temperature")
		}
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
		if !m.modelsIncludeTrialOrder {
			args = append(args, "--no-models-include-trial-order")
		}
		if m.modelsIncludePrev {
			args = append(args, "--models-include-prev-terms")
		}
		if !m.modelsIncludeRunBlock {
			args = append(args, "--no-models-include-run-block")
		}
		if !m.modelsIncludeInteraction {
			args = append(args, "--no-models-include-interaction")
		}
		if !m.modelsStandardize {
			args = append(args, "--no-models-standardize")
		}
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
		if !m.stabilityPartialTemp {
			args = append(args, "--no-stability-partial-temperature")
		}
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
	if m.isComputationSelected("consistency") && !m.consistencyEnabled {
		args = append(args, "--no-consistency")
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
		if !m.influenceIncludeTemperature {
			args = append(args, "--no-influence-include-temperature")
		}
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
		if !m.influenceIncludeTrialOrder {
			args = append(args, "--no-influence-include-trial-order")
		}
		if !m.influenceIncludeRunBlock {
			args = append(args, "--no-influence-include-run-block")
		}
		if m.influenceIncludeInteraction {
			args = append(args, "--influence-include-interaction")
		}
		if !m.influenceStandardize {
			args = append(args, "--no-influence-standardize")
		}
		if m.influenceCooksThreshold > 0 {
			args = append(args, "--influence-cooks-threshold", fmt.Sprintf("%.6f", m.influenceCooksThreshold))
		}
		if m.influenceLeverageThreshold > 0 {
			args = append(args, "--influence-leverage-threshold", fmt.Sprintf("%.6f", m.influenceLeverageThreshold))
		}
	}

	// Correlations (trial-table)
	if m.isComputationSelected("correlations") {
		targets := []string{}
		if m.correlationsTargetRating {
			targets = append(targets, "rating")
		}
		if m.correlationsTargetTemperature {
			targets = append(targets, "temperature")
		}
		if m.correlationsTargetPainResidual {
			targets = append(targets, "pain_residual")
		}
		defaultTargets := []string{"rating", "temperature", "pain_residual"}
		if len(targets) > 0 && !(len(targets) == len(defaultTargets) && strings.Join(targets, ",") == strings.Join(defaultTargets, ",")) {
			args = append(args, "--correlations-targets")
			args = append(args, targets...)
		}
		if !m.correlationsPreferPainResidual {
			args = append(args, "--no-correlations-prefer-pain-residual")
		}
		if strings.TrimSpace(m.correlationsTypesSpec) != "" && m.correlationsTypesSpec != "partial_cov_temp" {
			args = append(args, "--correlations-types")
			args = append(args, splitCSVList(m.correlationsTypesSpec)...)
		}
		if m.correlationsPrimaryUnit == 1 {
			args = append(args, "--correlations-primary-unit", "run_mean")
		}
		if m.correlationsPermutationPrimary {
			args = append(args, "--correlations-permutation-primary")
		}
		if m.correlationsUseCrossfitResidual {
			args = append(args, "--correlations-use-crossfit-pain-residual")
		}
		if strings.TrimSpace(m.correlationsTargetColumn) != "" {
			args = append(args, "--correlations-target-column", m.correlationsTargetColumn)
		}
	}

	// Multilevel correlations (group-level)
	if m.isComputationSelected("multilevel_correlations") && !m.groupLevelBlockPermutation {
		args = append(args, "--no-group-level-block-permutation")
	}

	// Report
	if m.isComputationSelected("report") && m.reportTopN != 15 {
		args = append(args, "--report-top-n", fmt.Sprintf("%d", m.reportTopN))
	}

	// Pain sensitivity
	if m.isComputationSelected("pain_sensitivity") {
		if m.painSensitivityMinTrials > 0 {
			args = append(args, "--pain-sensitivity-min-trials", fmt.Sprintf("%d", m.painSensitivityMinTrials))
		}
	}

	// Condition
	if m.isComputationSelected("condition") {
		if strings.TrimSpace(m.conditionCompareColumn) != "" {
			args = append(args, "--condition-compare-column", strings.TrimSpace(m.conditionCompareColumn))
		}
		if strings.TrimSpace(m.conditionCompareValues) != "" {
			args = append(args, "--condition-compare-values")
			args = append(args, splitCSVList(m.conditionCompareValues)...)
		}
		if strings.TrimSpace(m.conditionCompareWindows) != "" {
			args = append(args, "--condition-compare-windows")
			args = append(args, splitSpaceList(m.conditionCompareWindows)...)
		}
		if m.conditionMinTrials > 0 {
			args = append(args, "--condition-min-trials", fmt.Sprintf("%d", m.conditionMinTrials))
		}
		if m.conditionWindowPrimaryUnit == 1 {
			args = append(args, "--condition-window-primary-unit", "run_mean")
		}
		if m.conditionPermutationPrimary {
			args = append(args, "--condition-permutation-primary")
		}
		if !m.conditionFailFast {
			args = append(args, "--no-condition-fail-fast")
		}
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
		if m.temporalResolutionMs != 50 {
			args = append(args, "--temporal-time-resolution-ms", fmt.Sprintf("%d", m.temporalResolutionMs))
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
		if !m.temporalSplitByCondition {
			args = append(args, "--no-temporal-split-by-condition")
		}
		if strings.TrimSpace(m.temporalConditionColumn) != "" {
			args = append(args, "--temporal-condition-column", strings.TrimSpace(m.temporalConditionColumn))
		}
		if strings.TrimSpace(m.temporalConditionValues) != "" {
			args = append(args, "--temporal-condition-values")
			spec := strings.ReplaceAll(m.temporalConditionValues, ",", " ")
			args = append(args, splitSpaceList(spec)...)
		}
		if !m.temporalIncludeROIAverages {
			args = append(args, "--no-temporal-include-roi-averages")
		}
		if !m.temporalIncludeTFGrid {
			args = append(args, "--no-temporal-include-tf-grid")
		}
		// Temporal feature selection
		if !m.temporalFeaturePowerEnabled {
			args = append(args, "--no-temporal-feature-power")
		}
		if m.temporalFeatureITPCEnabled {
			args = append(args, "--temporal-feature-itpc")
		}
		if m.temporalFeatureERDSEnabled {
			args = append(args, "--temporal-feature-erds")
		}
		// ITPC-specific options (only if ITPC is selected in step 3)
		if m.featureFileSelected["itpc"] || m.temporalFeatureITPCEnabled {
			if !m.temporalITPCBaselineCorrection {
				args = append(args, "--no-temporal-itpc-baseline-correction")
			}
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
		if m.mediationBootstrap != 1000 {
			args = append(args, "--mediation-bootstrap", fmt.Sprintf("%d", m.mediationBootstrap))
		}
		if m.mediationPermutations > 0 {
			args = append(args, "--mediation-permutations", fmt.Sprintf("%d", m.mediationPermutations))
		}
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
		if m.moderationMinSamples != 15 {
			args = append(args, "--moderation-min-samples", fmt.Sprintf("%d", m.moderationMinSamples))
		}
		if m.moderationPermutations > 0 {
			args = append(args, "--moderation-permutations", fmt.Sprintf("%d", m.moderationPermutations))
		}
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
		if m.mixedMaxFeatures != 50 {
			args = append(args, "--mixed-max-features", fmt.Sprintf("%d", m.mixedMaxFeatures))
		}
	}

	// Output options
	if m.alsoSaveCsv {
		args = append(args, "--also-save-csv")
	}

	// Behavior Statistics
	tempControls := []string{"none", "linear", "spline"}
	if m.behaviorStatsTempControl != 0 {
		args = append(args, "--stats-temp-control", tempControls[m.behaviorStatsTempControl%len(tempControls)])
	}
	if m.behaviorStatsAllowIIDTrials {
		args = append(args, "--stats-allow-iid-trials")
	}
	if m.behaviorStatsHierarchicalFDR {
		args = append(args, "--stats-hierarchical-fdr")
	}
	if m.behaviorStatsComputeReliability {
		args = append(args, "--stats-compute-reliability")
	}
	permSchemes := []string{"shuffle", "circular_shift"}
	if m.behaviorPermScheme != 0 {
		args = append(args, "--perm-scheme", permSchemes[m.behaviorPermScheme%len(permSchemes)])
	}
	if strings.TrimSpace(m.behaviorPermGroupColumnPreference) != "" {
		args = append(args, "--perm-group-column-preference", strings.TrimSpace(m.behaviorPermGroupColumnPreference))
	}
	if m.behaviorExcludeNonTrialwiseFeatures {
		args = append(args, "--exclude-non-trialwise-features")
	}

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
		tails := []string{"two-tailed", "upper", "lower"}
		if m.clusterCorrectionTailGlobal != 0 {
			args = append(args, "--cluster-correction-tail", tails[m.clusterCorrectionTailGlobal%len(tails)])
		}
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
