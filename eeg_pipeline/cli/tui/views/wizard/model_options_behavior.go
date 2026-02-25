package wizard

// Behavior advanced option builders.

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

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
		"correlations", "multilevel_correlations", "predictor_sensitivity", "predictor_residual",
		"regression", "models", "stability", "influence",
		"condition", "temporal", "cluster", "mediation", "moderation", "mixed_effects",
	)

	// ── Execution ─────────────────────────────────────────────────────────────
	// RNG, parallelism, global thresholds — always relevant when any computation is selected.
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			options = append(options, optRNGSeed, optBehaviorNJobs, optBehaviorMinSamples, optBehaviorValidateOnly)
		}
	}

	// ── Inference & Shared Settings ───────────────────────────────────────────
	// Correlation method, FDR, permutations, covariates, run adjustment, feature QC.
	// Centralised so users configure shared inference settings once, not per-analysis.
	if needsInferenceSettings {
		options = append(options, optBehaviorGroupStats)
		if m.behaviorGroupStatsExpanded {
			// Correlation method — only when correlations/stability/predictor_sensitivity selected
			if hasSelectedComputation("correlations", "stability", "predictor_sensitivity") {
				options = append(options, optBehaviorSubCorrelationSettings, optCorrMethod, optRobustCorrelation)
			}
			// Predictor variable type — gates curve-fitting analyses
			options = append(options, optPredictorType)
			// Canonical behavior columns used across analyses
			options = append(options, optBehaviorOutcomeColumn, optBehaviorPredictorColumn)
			// Bootstrap — correlations, stability
			if hasSelectedComputation("correlations", "stability") {
				options = append(options, optBootstrap)
			}
			// FDR alpha — any analysis with multiple comparisons
			if hasSelectedComputation("correlations", "condition", "temporal", "cluster", "regression") {
				options = append(options, optFDRAlpha)
			}
			// Global permutations
			if hasSelectedComputation("cluster", "temporal", "regression", "correlations", "mediation", "moderation") {
				options = append(options, optNPerm)
			}
			// Shared statistics settings
			options = append(options,
				optBehaviorStatsTempControl,
				optBehaviorStatsAllowIIDTrials,
				optBehaviorStatsHierarchicalFDR,
				optBehaviorStatsComputeReliability,
				optBehaviorPermScheme,
				optBehaviorPermGroupColumnPreference,
				optBehaviorExcludeNonTrialwiseFeatures,
			)
			// Shared covariate controls
			if hasSelectedComputation("regression", "models", "influence", "correlations", "stability", "predictor_sensitivity") {
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
			// Feature QC
			if hasSelectedComputation("correlations", "multilevel_correlations") {
				options = append(options, optBehaviorSubFeatureQC, optFeatureQCEnabled)
				if m.featureQCEnabled {
					options = append(options,
						optFeatureQCMaxMissingPct,
						optFeatureQCMinVariance,
						optFeatureQCCheckWithinRunVariance,
					)
				}
			}
		}
	}

	// Trial table section - only show if trial_table computation is selected
	if m.isComputationSelected("trial_table") {
		options = append(options, optBehaviorGroupTrialTable)
		if m.behaviorGroupTrialTableExpanded {
			options = append(options,
				optTrialTableFormat,
				optTrialTableAddLagFeatures,
				optTrialOrderMaxMissingFraction,
				optFeatureSummariesEnabled,
			)
		}
	}

	// Predictor Residual — only valid for continuous predictors.
	// predictor_residual = outcome - f(predictor) requires a dose-response curve fit.
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
				optBehaviorSubDiagnostics,
				optPredictorResidualModelCompare,
				optPredictorResidualModelComparePolyDegrees,
				optPredictorResidualModelCompareMinSamples,
				optPredictorResidualBreakpoint,
				optPredictorResidualBreakpointCandidates,
				optPredictorResidualBreakpointMinSamples,
				optPredictorResidualBreakpointQlow,
				optPredictorResidualBreakpointQhigh,
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

	// Correlations section
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		options = append(options, optBehaviorGroupCorrelations)
		if m.behaviorGroupCorrelationsExpanded {
			if m.isComputationSelected("correlations") {
				options = append(options,
					optCorrelationsTargetColumn,
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

	// Regression section (includes model sensitivity options)
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
				optBehaviorSubModelFamilies,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
			)
		}
	}

	// Models section
	if m.isComputationSelected("models") {
		options = append(options, optBehaviorGroupModels)
		if m.behaviorGroupModelsExpanded {
			options = append(options,
				optBehaviorSubOutcomes,
				optModelsOutcomeRating,
				optModelsOutcomePredictor,
				optModelsOutcomeBinaryOutcome,
			)
			// predictor_residual outcome only exists for continuous predictors.
			if m.predictorType == 0 {
				options = append(options, optModelsOutcomePredictorResidual)
			}
			options = append(options,
				optBehaviorSubCovariates,
				optModelsIncludePredictor,
				optModelsTempControl,
			)
			// Spline sub-options only relevant for continuous predictors.
			if m.predictorType == 0 && m.modelsTempControl == 2 {
				options = append(options,
					optModelsTempSplineKnots,
					optModelsTempSplineQlow,
					optModelsTempSplineQhigh,
					optModelsTempSplineMinN,
				)
			}
			options = append(options,
				optModelsIncludeTrialOrder,
				optModelsIncludePrev,
				optModelsIncludeRunBlock,
				optModelsIncludeInteraction,
				optModelsStandardize,
				optModelsMinSamples,
				optBehaviorSubModelFamilies,
				optModelsFamilyOLS,
				optModelsFamilyRobust,
				optModelsFamilyQuantile,
				optModelsFamilyLogit,
				optModelsBinaryOutcome,
				optBehaviorSubInference,
				optModelsMaxFeatures,
				optModelsPrimaryUnit,
				optModelsForceTrialIIDAsymptotic,
			)
		}
	}

	// Predictor sensitivity section
	if m.isComputationSelected("predictor_sensitivity") {
		options = append(options, optBehaviorGroupPredictorSens)
		if m.behaviorGroupPredictorSensExpanded {
			options = append(
				options,
				optPredictorSensitivityMinTrials,
				optPredictorSensitivityPrimaryUnit,
				optPredictorSensitivityPermutations,
				optPredictorSensitivityPermutationPrimary,
				optPredictorSensitivityFeatures,
			)
		}
	}

	// Condition section
	if m.isComputationSelected("condition") {
		options = append(options, optBehaviorGroupCondition)
		if m.behaviorGroupConditionExpanded {
			options = append(options,
				optConditionCompareColumn,
				optConditionCompareValues,
				optConditionCompareLabels,
				optConditionCompareWindows,
				optConditionFeatures,
				optConditionMinTrials,
				optConditionPrimaryUnit,
				optConditionWindowPrimaryUnit,
				optConditionWindowMinSamples,
				optConditionPermutationPrimary,
				optConditionFailFast,
				optConditionEffectThreshold,
				optConditionOverwrite,
			)
		}
	}

	// Temporal section
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
			// Show ITPC-specific options when 'itpc' is selected in step 3 (feature selection)
			if m.featureFileSelected["itpc"] {
				options = append(options,
					optBehaviorSubITPC,
					optTemporalITPCBaselineCorrection,
					optTemporalITPCBaselineMin,
					optTemporalITPCBaselineMax,
				)
			}
			// Show ERDS-specific options when 'erds' is selected in step 3 (feature selection)
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

	// Cluster section
	if m.isComputationSelected("cluster") {
		options = append(options, optBehaviorGroupCluster)
		if m.behaviorGroupClusterExpanded {
			options = append(
				options,
				optClusterThreshold,
				optClusterMinSize,
				optClusterTail,
				optClusterFeatures,
				optClusterConditionColumn,
				optClusterConditionValues,
			)
		}
	}

	// ── Analyses (umbrella) ───────────────────────────────────────────────────
	// Stability, Consistency, Influence, Report, Mediation, Moderation, Mixed Effects.
	// Small analyses share one collapsible group to reduce visual noise.
	hasAnalysesGroup := hasSelectedComputation(
		"stability", "consistency", "influence", "report", "mediation", "moderation", "mixed_effects",
	)
	if hasAnalysesGroup {
		options = append(options, optBehaviorGroupAnalyses)
		if m.behaviorGroupAnalysesExpanded {
			if m.isComputationSelected("stability") {
				options = append(options,
					optBehaviorGroupStability,
				)
				if m.behaviorGroupStabilityExpanded {
					options = append(options,
						optStabilityMethod,
						optStabilityOutcome,
						optStabilityGroupColumn,
						optStabilityPartialTemp,
						optStabilityMinGroupTrials,
						optStabilityMaxFeatures,
						optStabilityAlpha,
					)
				}
			}
			if m.isComputationSelected("consistency") {
				options = append(options, optBehaviorGroupConsistency)
				if m.behaviorGroupConsistencyExpanded {
					options = append(options, optConsistencyEnabled)
				}
			}
			if m.isComputationSelected("influence") {
				options = append(options, optBehaviorGroupInfluence)
				if m.behaviorGroupInfluenceExpanded {
					options = append(options,
						optBehaviorSubOutcomes,
						optInfluenceOutcomeRating,
						optInfluenceOutcomePredictor,
						optBehaviorSubCovariates,
						optInfluenceIncludePredictor,
						optInfluenceTempControl,
					)
					// predictor_residual outcome and spline control: continuous only.
					if m.predictorType == 0 {
						options = append(options, optInfluenceOutcomePredictorResidual)
					}
					if m.predictorType == 0 && m.influenceTempControl == 2 {
						options = append(options,
							optInfluenceTempSplineKnots,
							optInfluenceTempSplineQlow,
							optInfluenceTempSplineQhigh,
							optInfluenceTempSplineMinN,
						)
					}
					options = append(options,
						optInfluenceIncludeTrialOrder,
						optInfluenceIncludeRunBlock,
						optInfluenceIncludeInteraction,
						optInfluenceStandardize,
						optBehaviorSubDiagnostics,
						optInfluenceMaxFeatures,
						optInfluenceCooksThreshold,
						optInfluenceLeverageThreshold,
					)
				}
			}
			if m.isComputationSelected("report") {
				options = append(options, optBehaviorGroupReport)
				if m.behaviorGroupReportExpanded {
					options = append(options, optReportTopN)
				}
			}
			if m.isComputationSelected("mediation") {
				options = append(options, optBehaviorGroupMediation)
				if m.behaviorGroupMediationExpanded {
					options = append(options,
						optMediationBootstrap,
						optMediationPermutations,
						optMediationPermutationPrimary,
						optMediationMinEffect,
						optMediationFeatures,
						optMediationMaxMediatorsEnabled,
						optMediationMaxMediators,
					)
				}
			}
			if m.isComputationSelected("moderation") {
				options = append(options, optBehaviorGroupModeration)
				if m.behaviorGroupModerationExpanded {
					options = append(options,
						optModerationMaxFeaturesEnabled,
						optModerationMaxFeatures,
						optModerationMinSamples,
						optModerationPermutations,
						optModerationPermutationPrimary,
						optModerationFeatures,
					)
				}
			}
			if m.isComputationSelected("mixed_effects") {
				options = append(options, optBehaviorGroupMixedEffects)
				if m.behaviorGroupMixedEffectsExpanded {
					options = append(options, optMixedEffectsType, optMixedIncludePredictor, optMixedMaxFeatures)
				}
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

	// ── Advanced (Global Validation + System IO) ───────────────────────────────
	// Rarely-touched settings merged into one group to reduce clutter.
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
