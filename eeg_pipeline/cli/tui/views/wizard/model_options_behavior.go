package wizard

// Behavior advanced option builders.

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)
	options = append(options, optConfigSetOverrides)

	hasAnyComputation := len(m.SelectedComputations()) > 0
	hasSelectedComputation := func(keys ...string) bool {
		for _, key := range keys {
			if m.isComputationSelected(key) {
				return true
			}
		}
		return false
	}
	needsInferenceSettings := hasSelectedComputation(
		"correlations", "multilevel_correlations", "predictor_residual",
		"regression", "condition", "temporal", "cluster",
	)

	// ── Execution ─────────────────────────────────────────────────────────────
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			options = append(options, optRNGSeed, optBehaviorNJobs, optBehaviorMinSamples, optBehaviorValidateOnly)
		}
	}

	// ── Inference & Shared Settings ───────────────────────────────────────────
	if needsInferenceSettings {
		options = append(options, optBehaviorGroupStats)
		if m.behaviorGroupStatsExpanded {
			// Correlation method — only when correlations selected
			if hasSelectedComputation("correlations") {
				options = append(options, optBehaviorSubCorrelationSettings, optCorrMethod, optRobustCorrelation)
			}
			// Predictor variable type — gates curve-fitting analyses
			options = append(options, optPredictorType)
			// Canonical behavior columns used across analyses
			options = append(options, optBehaviorOutcomeColumn, optBehaviorPredictorColumn)
			if hasSelectedComputation("correlations") {
				options = append(options, optBootstrap)
			}
			// FDR alpha — any analysis with multiple comparisons
			if hasSelectedComputation("correlations", "condition", "temporal", "cluster", "regression") {
				options = append(options, optFDRAlpha)
			}
			// Global permutations
			if hasSelectedComputation("cluster", "temporal", "regression", "correlations") {
				options = append(options, optNPerm)
			}
			// Shared statistics settings
			options = append(options,
				optBehaviorStatsTempControl,
				optBehaviorStatsAllowIIDTrials,
				optBehaviorStatsHierarchicalFDR,
				optBehaviorStatsComputeReliability,
				optStatisticsAlpha,
				optBehaviorPermScheme,
				optBehaviorPermGroupColumnPreference,
				optBehaviorExcludeNonTrialwiseFeatures,
				optBehaviorSubFeatureRegistry,
				optBehaviorFeatureRegistryFilesJSON,
				optBehaviorFeatureRegistrySourceToTypeJSON,
				optBehaviorFeatureRegistryTypeHierarchyJSON,
				optBehaviorFeatureRegistryPatternsJSON,
				optBehaviorFeatureRegistryClassifiersJSON,
			)
			if hasSelectedComputation("regression", "correlations") {
				options = append(options, optBehaviorSubCovariates, optControlTemp, optControlOrder)
			}
			// Run adjustment
			if hasSelectedComputation("trial_table", "correlations") {
				options = append(options, optBehaviorSubRunAdjustment, optRunAdjustmentEnabled)
				if m.runAdjustmentEnabled {
					options = append(options,
						optRunAdjustmentColumn,
						optRunAdjustmentIncludeInCorrelations,
						optRunAdjustmentMaxDummies,
					)
				}
			}
			// Correlations extras (change scores, LOSO, Bayes)
			if m.isComputationSelected("correlations") {
				options = append(options,
					optBehaviorSubCorrelationsExtra,
					optComputeChangeScores,
					optComputeLosoStability,
					optComputeBayesFactors,
				)
			}
		}
	}

	// Trial table
	if m.isComputationSelected("trial_table") {
		options = append(options, optBehaviorGroupTrialTable)
		if m.behaviorGroupTrialTableExpanded {
			options = append(options,
				optTrialTableFormat,
				optBehaviorTrialTableDisallowPositionalAlignment,
				optTrialOrderMaxMissingFraction,
				optFeatureSummariesEnabled,
			)
		}
	}

	// Predictor Residual — only valid for continuous predictors.
	if m.isComputationSelected("predictor_residual") && m.predictorType == 0 {
		options = append(options, optBehaviorGroupPredictorResidual)
		if m.behaviorGroupPredictorResidualExpanded {
			options = append(options,
				optBehaviorSubFitting,
				optPredictorResidualEnabled,
				optPredictorResidualMethod,
				optPredictorResidualPolyDegree,
				optPredictorResidualSplineDfCandidates,
				optPredictorResidualMinSamples,
				optBehaviorSubCrossfit,
				optPredictorResidualCrossfitEnabled,
			)
			if m.predictorResidualCrossfitEnabled {
				options = append(options,
					optPredictorResidualCrossfitGroupColumn,
					optPredictorResidualCrossfitNSplits,
					optPredictorResidualCrossfitMethod,
					optPredictorResidualCrossfitSplineKnots,
				)
			}
		}
	}

	// Correlations
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
			if m.isComputationSelected("correlations") {
				options = append(options,
					optCorrelationsTargetColumn,
					optCorrelationsPowerSegment,
					optCorrelationsFeatures,
					optCorrelationsTypes,
					optCorrelationsPrimaryUnit,
					optCorrelationsMinRuns,
					optCorrelationsPermutations,
					optCorrelationsPermutationPrimary,
				)
				// Predictor-residual preference only meaningful for continuous predictors.
				if m.predictorType == 0 {
					options = append(options,
						optCorrelationsPreferPredictorResidual,
						optCorrelationsUseCrossfitPredictorResidual,
					)
				}
			}
			options = append(options, optBehaviorSubMultilevel, optCorrelationsMultilevel)
			if m.isComputationSelected("multilevel_correlations") {
				options = append(options,
					optGroupLevelBlockPermutation,
					optGroupLevelTarget,
					optGroupLevelControlPredictor,
					optGroupLevelControlTrialOrder,
					optGroupLevelControlRunEffects,
					optGroupLevelMaxRunDummies,
					optGroupLevelAllowParametricFallback,
				)
			}
		}
	}

	// Regression
	if m.isComputationSelected("regression") {
		options = append(options, optBehaviorGroupRegression)
		if m.behaviorGroupRegressionExpanded {
			options = append(options,
				optBehaviorSubOutcome,
				optRegressionOutcome,
				optRegressionIncludePredictor,
				optRegressionTempControl,
			)
			if m.predictorType == 0 && m.regressionTempControl == 2 {
				options = append(options,
					optRegressionTempSplineKnots,
					optRegressionTempSplineQlow,
					optRegressionTempSplineQhigh,
					optRegressionTempSplineMinN,
				)
			}
			options = append(options,
				optBehaviorSubCovariates,
				optRegressionIncludeTrialOrder,
				optRegressionIncludePrev,
				optRegressionIncludeRunBlock,
				optRegressionIncludeInteraction,
				optRegressionStandardize,
				optRegressionMinSamples,
				optBehaviorSubInference,
				optRegressionPrimaryUnit,
				optRegressionPermutations,
				optRegressionMaxFeatures,
			)
		}
	}

	// Condition
	if m.isComputationSelected("condition") {
		options = append(options, optBehaviorGroupCondition)
		if m.behaviorGroupConditionExpanded {
			options = append(options,
				optConditionCompareColumn,
				optConditionCompareValues,
				optConditionCompareLabels,
				optConditionFeatures,
				optConditionMinTrials,
				optConditionPrimaryUnit,
				optConditionPermutationPrimary,
				optConditionFailFast,
				optConditionEffectThreshold,
				optConditionOverwrite,
			)
		}
	}

	// Temporal
	if m.isComputationSelected("temporal") {
		options = append(options, optBehaviorGroupTemporal)
		if m.behaviorGroupTemporalExpanded {
			options = append(options,
				optBehaviorSubTimeWindow,
				optTemporalResolutionMs,
				optTemporalCorrectionMethod,
				optTemporalTimeMinMs,
				optTemporalTimeMaxMs,
				optTemporalSmoothMs,
				optTemporalTargetColumn,
				optBehaviorSubFeatures,
				optTemporalFeatures,
				optTemporalSplitByCondition,
				optTemporalConditionColumn,
				optTemporalConditionValues,
				optTemporalIncludeROIAverages,
				optTemporalIncludeTFGrid,
			)
			if m.featureFileSelected["itpc"] {
				options = append(options,
					optBehaviorSubITPC,
					optTemporalITPCBaselineCorrection,
					optTemporalITPCBaselineMin,
					optTemporalITPCBaselineMax,
				)
			}
			if m.featureFileSelected["erds"] {
				options = append(options,
					optBehaviorSubERDS,
					optTemporalERDSBaselineMin,
					optTemporalERDSBaselineMax,
					optTemporalERDSMethod,
				)
			}
		}
	}

	// Cluster
	if m.isComputationSelected("cluster") {
		options = append(options, optBehaviorGroupCluster)
		if m.behaviorGroupClusterExpanded {
			options = append(options,
				optClusterThreshold,
				optClusterMinSize,
				optClusterTail,
				optClusterFeatures,
				optClusterConditionColumn,
				optClusterConditionValues,
			)
		}
	}

	// Report
	if m.isComputationSelected("report") {
		options = append(options, optBehaviorGroupReport)
		if m.behaviorGroupReportExpanded {
			options = append(options, optReportTopN)
		}
	}

	// ── Output ────────────────────────────────────────────────────────────────
	if hasAnyComputation {
		options = append(options, optBehaviorGroupOutput)
		if m.behaviorGroupOutputExpanded {
			options = append(options, optAlsoSaveCsv, optBehaviorOverwrite)
		}
	}

	// ── Advanced (Global Validation + System IO) ───────────────────────────────
	if needsInferenceSettings {
		options = append(options, optBehaviorGroupAdvanced)
		if m.behaviorGroupAdvancedExpanded {
			options = append(options,
				optGlobalNBootstrap,
				optClusterCorrectionEnabled,
			)
			if m.clusterCorrectionEnabled {
				options = append(options,
					optClusterCorrectionAlpha,
					optClusterCorrectionMinClusterSize,
					optClusterCorrectionTail,
				)
			}
			options = append(options,
				optValidationMinEpochs,
				optValidationMinChannels,
				optValidationMaxAmplitudeUv,
				optIOPredictorRange,
				optIOMaxMissingChannelsFraction,
			)
		}
	}

	return options
}
