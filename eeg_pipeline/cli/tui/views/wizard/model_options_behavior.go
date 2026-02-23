package wizard

// Behavior advanced option builders.

func (m Model) getBehaviorOptions() []optionType {
	var options []optionType
	options = append(options, optUseDefaults)

	// Check if any computation is selected
	hasAnyComputation := len(m.SelectedComputations()) > 0
	hasSelectedComputation := func(keys ...string) bool {
		for _, key := range keys {
			if m.isComputationSelected(key) {
				return true
			}
		}
		return false
	}
	needsBehaviorStatsAndGlobalOptions := hasSelectedComputation(
		"correlations",
		"multilevel_correlations",
		"pain_sensitivity",
		"regression",
		"models",
		"stability",
		"influence",
		"condition",
		"temporal",
		"cluster",
		"mediation",
		"moderation",
		"mixed_effects",
	)

	// General section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupGeneral)
		if m.behaviorGroupGeneralExpanded {
			// RNG Seed and N Jobs are always relevant
			options = append(options, optRNGSeed, optBehaviorNJobs, optBehaviorMinSamples, optBehaviorValidateOnly)

			// Correlation method and robust correlation - only for correlations, stability, pain_sensitivity
			needsCorrelationMethod := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCorrelationMethod {
				options = append(options, optBehaviorSubCorrelationSettings, optCorrMethod, optRobustCorrelation)
			}

			// Bootstrap - relevant for correlations, stability
			needsBootstrap := m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability")
			if needsBootstrap {
				options = append(options, optBootstrap)
			}

			// FDR Alpha - relevant for correlations, condition, temporal, cluster, regression
			needsFDR := m.isComputationSelected("correlations") ||
				m.isComputationSelected("condition") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("cluster") ||
				m.isComputationSelected("regression")
			if needsFDR {
				options = append(options, optFDRAlpha)
			}

			// N Permutations - relevant for cluster, temporal, regression, correlations, mediation, moderation
			needsPermutations := m.isComputationSelected("cluster") ||
				m.isComputationSelected("temporal") ||
				m.isComputationSelected("regression") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("mediation") ||
				m.isComputationSelected("moderation")
			if needsPermutations {
				options = append(options, optNPerm)
			}

			// Covariate controls - relevant for regression, models, influence, correlations, stability, pain_sensitivity
			needsCovariates := m.isComputationSelected("regression") ||
				m.isComputationSelected("models") ||
				m.isComputationSelected("influence") ||
				m.isComputationSelected("correlations") ||
				m.isComputationSelected("stability") ||
				m.isComputationSelected("pain_sensitivity")
			if needsCovariates {
				options = append(options, optBehaviorSubCovariates, optControlTemp, optControlOrder)
			}

			// Run adjustment - relevant for trial_table, correlations
			needsRunAdjustment := m.isComputationSelected("trial_table") ||
				m.isComputationSelected("correlations")
			if needsRunAdjustment {
				options = append(options, optBehaviorSubRunAdjustment, optRunAdjustmentEnabled)
				if m.runAdjustmentEnabled {
					options = append(options,
						optRunAdjustmentColumn,
						optRunAdjustmentIncludeInCorrelations,
						optRunAdjustmentMaxDummies,
					)
				}
			}

			// Change scores, LOSO stability, Bayes factors - relevant for correlations
			if m.isComputationSelected("correlations") {
				options = append(options,
					optBehaviorSubCorrelationsExtra,
					optComputeChangeScores,
					optComputeLosoStability,
					optComputeBayesFactors,
				)
			}

			// Feature QC (optional gating) - relevant for correlations / multilevel correlations
			if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
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

	// Pain Residual section - only show if pain_residual computation is selected
	if m.isComputationSelected("pain_residual") {
		options = append(options, optBehaviorGroupPainResidual)
		if m.behaviorGroupPainResidualExpanded {
			options = append(options,
				optBehaviorSubFitting,
				optPainResidualEnabled,
				optPainResidualMethod,
				optPainResidualPolyDegree,
				optPainResidualSplineDfCandidates,
				optPainResidualMinSamples,
				optBehaviorSubDiagnostics,
				optPainResidualModelCompare,
				optPainResidualModelComparePolyDegrees,
				optPainResidualModelCompareMinSamples,
				optPainResidualBreakpoint,
				optPainResidualBreakpointCandidates,
				optPainResidualBreakpointMinSamples,
				optPainResidualBreakpointQlow,
				optPainResidualBreakpointQhigh,
				optBehaviorSubCrossfit,
				optPainResidualCrossfitEnabled,
			)
			if m.painResidualCrossfitEnabled {
				options = append(options,
					optPainResidualCrossfitGroupColumn,
					optPainResidualCrossfitNSplits,
					optPainResidualCrossfitMethod,
					optPainResidualCrossfitSplineKnots,
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
					optCorrelationsPreferPainResidual,
					optCorrelationsPermutations,
					optCorrelationsPermutationPrimary,
					optCorrelationsUseCrossfitPainResidual,
				)
			}
			options = append(options, optBehaviorSubMultilevel, optCorrelationsMultilevel)
			if m.isComputationSelected("multilevel_correlations") {
				options = append(options,
					optGroupLevelBlockPermutation,
					optGroupLevelTarget,
					optGroupLevelControlTemperature,
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
				optRegressionIncludeTemperature,
				optRegressionTempControl,
			)
			if m.regressionTempControl == 2 {
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
				optModelsOutcomePainResidual,
				optModelsOutcomeTemperature,
				optModelsOutcomePainBinary,
				optBehaviorSubCovariates,
				optModelsIncludeTemperature,
				optModelsTempControl,
			)
			if m.modelsTempControl == 2 {
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

	// Stability section
	if m.isComputationSelected("stability") {
		options = append(options, optBehaviorGroupStability)
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

	// Consistency section
	if m.isComputationSelected("consistency") {
		options = append(options, optBehaviorGroupConsistency)
		if m.behaviorGroupConsistencyExpanded {
			options = append(options, optConsistencyEnabled)
		}
	}

	// Influence section
	if m.isComputationSelected("influence") {
		options = append(options, optBehaviorGroupInfluence)
		if m.behaviorGroupInfluenceExpanded {
			options = append(options,
				optBehaviorSubOutcomes,
				optInfluenceOutcomeRating,
				optInfluenceOutcomePainResidual,
				optInfluenceOutcomeTemperature,
				optBehaviorSubCovariates,
				optInfluenceIncludeTemperature,
				optInfluenceTempControl,
			)
			if m.influenceTempControl == 2 {
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

	// Report section
	if m.isComputationSelected("report") {
		options = append(options, optBehaviorGroupReport)
		if m.behaviorGroupReportExpanded {
			options = append(options, optReportTopN)
		}
	}

	// Pain sensitivity section
	if m.isComputationSelected("pain_sensitivity") {
		options = append(options, optBehaviorGroupPainSens)
		if m.behaviorGroupPainSensExpanded {
			options = append(
				options,
				optPainSensitivityMinTrials,
				optPainSensitivityPrimaryUnit,
				optPainSensitivityPermutations,
				optPainSensitivityPermutationPrimary,
				optPainSensitivityFeatures,
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

	// Mediation section
	if m.isComputationSelected("mediation") {
		options = append(options, optBehaviorGroupMediation)
		if m.behaviorGroupMediationExpanded {
			options = append(options, optMediationBootstrap, optMediationPermutations, optMediationPermutationPrimary, optMediationMinEffect, optMediationFeatures, optMediationMaxMediatorsEnabled, optMediationMaxMediators)
		}
	}

	// Moderation section
	if m.isComputationSelected("moderation") {
		options = append(options, optBehaviorGroupModeration)
		if m.behaviorGroupModerationExpanded {
			options = append(options, optModerationMaxFeaturesEnabled, optModerationMaxFeatures, optModerationMinSamples, optModerationPermutations, optModerationPermutationPrimary, optModerationFeatures)
		}
	}

	// Mixed effects section
	if m.isComputationSelected("mixed_effects") {
		options = append(options, optBehaviorGroupMixedEffects)
		if m.behaviorGroupMixedEffectsExpanded {
			options = append(options, optMixedEffectsType, optMixedIncludeTemperature, optMixedMaxFeatures)
		}
	}

	// Output section - only show if at least one computation is selected
	if hasAnyComputation {
		options = append(options, optBehaviorGroupOutput)
		if m.behaviorGroupOutputExpanded {
			options = append(options, optAlsoSaveCsv, optBehaviorOverwrite)
		}
	}

	// Behavior Statistics
	if needsBehaviorStatsAndGlobalOptions {
		options = append(options, optBehaviorGroupStats)
		if m.behaviorGroupStatsExpanded {
			options = append(options,
				optBehaviorStatsTempControl,
				optBehaviorStatsAllowIIDTrials,
				optBehaviorStatsHierarchicalFDR,
				optBehaviorStatsComputeReliability,
				optBehaviorPermScheme,
				optBehaviorPermGroupColumnPreference,
				optBehaviorExcludeNonTrialwiseFeatures,
			)
		}
	}

	// Global Statistics & Validation
	if needsBehaviorStatsAndGlobalOptions {
		options = append(options, optBehaviorGroupGlobalValidation)
		if m.behaviorGroupGlobalValidationExpanded {
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
			)
		}
	}

	// System / IO
	if needsBehaviorStatsAndGlobalOptions {
		options = append(options, optBehaviorGroupSystemIO)
		if m.behaviorGroupSystemIOExpanded {
			options = append(options,
				optIOTemperatureRange,
				optIOMaxMissingChannelsFraction,
			)
		}
	}

	return options
}
