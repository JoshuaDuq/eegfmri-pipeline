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
	predictorTypes := []string{"continuous", "binary", "categorical"}
	if m.predictorType > 0 && m.predictorType < len(predictorTypes) {
		args = append(args, "--predictor-type", predictorTypes[m.predictorType])
	}

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

	effectivePredictorControl := m.controlPredictor
	// "none" for stats predictor control implies disabling global predictor control.
	if m.behaviorStatsTempControl == 2 {
		effectivePredictorControl = false
	}
	appendBoolPair(effectivePredictorControl, "--predictor-control", "--no-predictor-control")
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
	appendBoolPair(m.behaviorComputeBayesFactors, "--compute-bayes-factors", "--no-compute-bayes-factors")
	if m.behaviorValidateOnly {
		args = append(args, "--validate-only")
	}

	// Output options
	if m.behaviorOverwrite {
		args = append(args, "--overwrite")
	} else {
		args = append(args, "--no-overwrite")
	}

	// Trial table / predictor residual
	if m.isComputationSelected("trial_table") {
		formats := []string{"parquet", "tsv"}
		if m.trialTableFormat >= 0 && m.trialTableFormat < len(formats) && m.trialTableFormat != 0 {
			args = append(args, "--trial-table-format", formats[m.trialTableFormat])
		}
		appendBoolPair(
			m.trialTableDisallowPositionalAlignment,
			"--trial-table-disallow-positional-alignment",
			"--no-trial-table-disallow-positional-alignment",
		)
		if m.trialOrderMaxMissingFraction != 0.1 {
			args = append(args, "--trial-order-max-missing-fraction", fmt.Sprintf("%.3f", m.trialOrderMaxMissingFraction))
		}
		appendBoolPair(m.featureSummariesEnabled, "--feature-summaries", "--no-feature-summaries")

		appendBoolPair(m.predictorResidualEnabled, "--predictor-residual", "--no-predictor-residual")
		if m.predictorResidualEnabled {
			methods := []string{"spline", "poly"}
			if m.predictorResidualMethod >= 0 && m.predictorResidualMethod < len(methods) && m.predictorResidualMethod != 0 {
				args = append(args, "--predictor-residual-method", methods[m.predictorResidualMethod])
			}
			if m.predictorResidualMinSamples != 10 {
				args = append(args, "--predictor-residual-min-samples", fmt.Sprintf("%d", m.predictorResidualMinSamples))
			}
			if m.predictorResidualPolyDegree != 2 {
				args = append(args, "--predictor-residual-poly-degree", fmt.Sprintf("%d", m.predictorResidualPolyDegree))
			}
			if strings.TrimSpace(m.predictorResidualSplineDfCandidates) != "" && m.predictorResidualSplineDfCandidates != "3,4,5" {
				args = append(args, "--predictor-residual-spline-df-candidates")
				args = append(args, splitCSVList(m.predictorResidualSplineDfCandidates)...)
			}
			appendBoolPair(m.predictorResidualCrossfitEnabled, "--predictor-residual-crossfit", "--no-predictor-residual-crossfit")
		}
		if m.predictorResidualEnabled && m.predictorResidualCrossfitEnabled {
			args = append(args, "--predictor-residual-crossfit")
			if strings.TrimSpace(m.predictorResidualCrossfitGroupColumn) != "" {
				args = append(args, "--predictor-residual-crossfit-group-column", strings.TrimSpace(m.predictorResidualCrossfitGroupColumn))
			}
			if m.predictorResidualCrossfitNSplits != 5 {
				args = append(args, "--predictor-residual-crossfit-n-splits", fmt.Sprintf("%d", m.predictorResidualCrossfitNSplits))
			}
			cfMethods := []string{"spline", "poly"}
			if m.predictorResidualCrossfitMethod >= 0 && m.predictorResidualCrossfitMethod < len(cfMethods) && m.predictorResidualCrossfitMethod != 0 {
				args = append(args, "--predictor-residual-crossfit-method", cfMethods[m.predictorResidualCrossfitMethod])
			}
			if m.predictorResidualCrossfitMethod == 0 && m.predictorResidualCrossfitSplineKnots != 5 {
				args = append(args, "--predictor-residual-crossfit-spline-n-knots", fmt.Sprintf("%d", m.predictorResidualCrossfitSplineKnots))
			}
		}
	}

	// Regression
	if m.isComputationSelected("regression") {
		outcomes := []string{"outcome", "predictor_residual", "predictor"}
		if m.regressionOutcome >= 0 && m.regressionOutcome < len(outcomes) && m.regressionOutcome != 0 {
			args = append(args, "--regression-outcome", outcomes[m.regressionOutcome])
		}
		appendBoolPair(
			m.regressionIncludePredictor,
			"--regression-include-predictor",
			"--no-regression-include-predictor",
		)
		tempCtrl := []string{"linear", "outcome_hat", "spline"}
		if m.regressionTempControl >= 0 && m.regressionTempControl < len(tempCtrl) && m.regressionTempControl != 0 {
			args = append(args, "--regression-predictor-control", tempCtrl[m.regressionTempControl])
		}
		if m.regressionTempControl == 2 {
			args = append(args, "--regression-predictor-spline-knots", fmt.Sprintf("%d", m.regressionTempSplineKnots))
			args = append(args, "--regression-predictor-spline-quantile-low", fmt.Sprintf("%.3f", m.regressionTempSplineQlow))
			args = append(args, "--regression-predictor-spline-quantile-high", fmt.Sprintf("%.3f", m.regressionTempSplineQhigh))
			if m.regressionTempSplineMinN != 12 {
				args = append(args, "--regression-predictor-spline-min-samples", fmt.Sprintf("%d", m.regressionTempSplineMinN))
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

	// Correlations (trial-table)
	if m.isComputationSelected("correlations") {
		appendFeatureSpec("--correlations-features", m.correlationsFeaturesSpec)
		if strings.TrimSpace(m.correlationsTypesSpec) != "" && m.correlationsTypesSpec != "partial_cov_predictor" {
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
			m.correlationsPreferPredictorResidual,
			"--correlations-prefer-predictor-residual",
			"--no-correlations-prefer-predictor-residual",
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
			"--correlations-use-crossfit-predictor-residual",
			"--no-correlations-use-crossfit-predictor-residual",
		)
		// Always pass explicit target selection (possibly empty) so the backend
		// does not silently fall back to built-in defaults.
		args = append(args, "--correlations-target-column", strings.TrimSpace(m.correlationsTargetColumn))
		if seg := strings.TrimSpace(m.correlationsPowerSegment); seg != "" {
			args = append(args, "--correlations-power-segment", seg)
		}
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
			m.groupLevelControlPredictor,
			"--group-level-control-predictor",
			"--no-group-level-control-predictor",
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
		if m.conditionMinTrials > 0 {
			args = append(args, "--condition-min-trials", fmt.Sprintf("%d", m.conditionMinTrials))
		}
		if m.conditionPrimaryUnit == 1 {
			args = append(args, "--condition-primary-unit", "run_mean")
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

	// Output options
	appendBoolPair(m.alsoSaveCsv, "--also-save-csv", "--no-also-save-csv")

	// Behavior Statistics
	tempControls := []string{"spline", "linear", "none"}
	if m.behaviorStatsTempControl >= 0 && m.behaviorStatsTempControl < len(tempControls) && m.behaviorStatsTempControl != 2 {
		args = append(args, "--stats-predictor-control", tempControls[m.behaviorStatsTempControl])
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
	if m.statisticsAlpha != 0.05 {
		args = append(args, "--statistics-alpha", fmt.Sprintf("%.4f", m.statisticsAlpha))
	}
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
	if strings.TrimSpace(m.behaviorFeatureRegistryFilesJSON) != "" {
		args = append(args, "--feature-registry-files-json", strings.TrimSpace(m.behaviorFeatureRegistryFilesJSON))
	}
	if strings.TrimSpace(m.behaviorFeatureRegistrySourceJSON) != "" {
		args = append(args, "--feature-registry-source-to-feature-type-json", strings.TrimSpace(m.behaviorFeatureRegistrySourceJSON))
	}
	if strings.TrimSpace(m.behaviorFeatureRegistryHierarchyJSON) != "" {
		args = append(args, "--feature-registry-type-hierarchy-json", strings.TrimSpace(m.behaviorFeatureRegistryHierarchyJSON))
	}
	if strings.TrimSpace(m.behaviorFeatureRegistryPatternsJSON) != "" {
		args = append(args, "--feature-registry-patterns-json", strings.TrimSpace(m.behaviorFeatureRegistryPatternsJSON))
	}
	if strings.TrimSpace(m.behaviorFeatureRegistryClassifiersJSON) != "" {
		args = append(args, "--feature-registry-classifiers-json", strings.TrimSpace(m.behaviorFeatureRegistryClassifiersJSON))
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
	if strings.TrimSpace(m.ioPredictorRange) != "" {
		args = append(args, "--predictor-range")
		args = append(args, splitCSVList(m.ioPredictorRange)...)
	}
	if m.ioMaxMissingChannelsFraction != 0.1 {
		args = append(args, "--max-missing-channels-fraction", fmt.Sprintf("%.3f", m.ioMaxMissingChannelsFraction))
	}

	return args
}
