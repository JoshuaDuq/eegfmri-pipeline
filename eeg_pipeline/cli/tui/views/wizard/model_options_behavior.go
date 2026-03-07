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
	hasCorrelations := hasSelectedComputation("correlations")
	hasMultilevel := hasSelectedComputation("multilevel_correlations")
	hasICC := hasSelectedComputation("icc")
	hasPredictorResidual := hasSelectedComputation("predictor_residual")
	hasRegression := hasSelectedComputation("regression")
	hasCondition := hasSelectedComputation("condition")
	hasTemporal := hasSelectedComputation("temporal")
	hasCluster := hasSelectedComputation("cluster")

	needsSharedAnalysisSettings := hasCorrelations ||
		hasPredictorResidual ||
		hasRegression ||
		hasCondition ||
		hasTemporal ||
		hasCluster
	needsPredictorType := hasCorrelations || hasPredictorResidual || hasRegression
	needsOutcomePredictorColumns := hasCorrelations || hasPredictorResidual || hasRegression
	needsSharedInferenceControls := hasCorrelations || hasRegression || hasCondition || hasTemporal || hasCluster
	needsGlobalPermutationBudget := hasCorrelations || hasCondition || hasTemporal || hasCluster
	needsPermutationSettings := hasCorrelations || hasRegression || hasCondition || hasTemporal || hasCluster
	needsFeatureRegistrySettings := hasCorrelations || hasMultilevel || hasRegression || hasCondition || hasTemporal || hasCluster

	// ── Execution ─────────────────────────────────────────────────────────────
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			options = append(options, optRNGSeed, optBehaviorNJobs, optBehaviorMinSamples)
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
	if m.isComputationSelected("correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
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
	}

	if m.isComputationSelected("multilevel_correlations") {
		options = append(options, optBehaviorGroupGroupLevel)
		if m.behaviorGroupGroupLevelExpanded {
			options = append(options,
				optBehaviorSubMultilevel,
				optGroupLevelBlockPermutation,
				optGroupLevelTarget,
				optGroupLevelControlPredictor,
				optGroupLevelControlTrialOrder,
				optGroupLevelControlRunEffects,
				optGroupLevelMaxRunDummies,
			)
		}
	}

	if hasICC {
		options = append(options, optBehaviorGroupICC)
		if m.behaviorGroupICCExpanded {
			options = append(options, optICCUnitColumns)
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
				optRegressionPredictorControl,
			)
			if m.predictorType == 0 && m.regressionPredictorControl == 2 {
				options = append(options,
					optRegressionPredictorSplineKnots,
					optRegressionPredictorSplineQlow,
					optRegressionPredictorSplineQhigh,
					optRegressionPredictorSplineMinN,
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
	// ── Shared Analysis Settings ──────────────────────────────────────────────
	if needsSharedAnalysisSettings {
		options = append(options, optBehaviorGroupStats)
		if m.behaviorGroupStatsExpanded {
			if needsPredictorType || needsOutcomePredictorColumns {
				options = append(options, optBehaviorSubDataMapping)
			}
			if needsPredictorType {
				options = append(options, optPredictorType)
			}
			if needsOutcomePredictorColumns {
				options = append(options, optBehaviorOutcomeColumn, optBehaviorPredictorColumn)
			}

			if hasCorrelations || needsSharedInferenceControls {
				options = append(options, optBehaviorSubStatisticalInference)
			}
			if hasCorrelations {
				options = append(options, optBehaviorSubCorrelationSettings, optCorrMethod, optRobustCorrelation)
				options = append(options, optBootstrap, optBehaviorStatsPredictorControl, optBehaviorStatsComputeReliability)
			}
			if needsSharedInferenceControls {
				options = append(options,
					optFDRAlpha,
					optBehaviorStatsAllowIIDTrials,
					optBehaviorStatsHierarchicalFDR,
					optStatisticsAlpha,
				)
			}

			if needsPermutationSettings || needsGlobalPermutationBudget {
				options = append(options, optBehaviorSubPermutations)
			}
			if needsGlobalPermutationBudget {
				options = append(options, optNPerm)
			}
			if needsPermutationSettings {
				options = append(options, optBehaviorPermScheme, optBehaviorPermGroupColumnPreference)
			}

			if needsFeatureRegistrySettings {
				options = append(options,
					optBehaviorSubFeatureRegistry,
					optBehaviorExcludeNonTrialwiseFeatures,
					optBehaviorFeatureRegistryFilesJSON,
					optBehaviorFeatureRegistrySourceToTypeJSON,
					optBehaviorFeatureRegistryTypeHierarchyJSON,
					optBehaviorFeatureRegistryPatternsJSON,
					optBehaviorFeatureRegistryClassifiersJSON,
				)
			}
			if hasCorrelations || hasRegression {
				options = append(options, optBehaviorSubCovariates, optControlTemp, optControlOrder)
			}
			if hasCorrelations {
				options = append(options, optBehaviorSubRunAdjustment, optRunAdjustmentEnabled)
				if m.runAdjustmentEnabled {
					options = append(options,
						optRunAdjustmentColumn,
						optRunAdjustmentIncludeInCorrelations,
						optRunAdjustmentMaxDummies,
					)
				}
				options = append(options,
					optBehaviorSubCorrelationsExtra,
					optComputeChangeScores,
					optComputeLosoStability,
					optComputeBayesFactors,
				)
			}
		}
	}

	// ── Output ────────────────────────────────────────────────────────────────
	if hasAnyComputation {
		options = append(options, optBehaviorGroupOutput)
		if m.behaviorGroupOutputExpanded {
			options = append(options, optAlsoSaveCsv, optBehaviorOverwrite)
		}
	}

	return options
}
