package wizard

import "strings"

// Machine-learning advanced option builders.

func (m Model) getMLOptions() []optionType {
	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	opts := []optionType{optUseDefaults, optConfigSetOverrides}

	// ── Data & Features ──────────────────────────────────────────
	opts = append(opts, optMLGroupData)
	if m.mlGroupDataExpanded {
		opts = append(opts,
			optMLTarget,
			optMLFeatureFamilies,
			optMLFeatureBands,
			optMLFeatureSegments,
			optMLFeatureScopes,
			optMLFeatureStats,
			optMLFeatureHarmonization,
			optMLCovariates,
		)
		if mode == "incremental_validity" {
			opts = append(opts, optMLBaselinePredictors)
		}
		if strings.EqualFold(strings.TrimSpace(m.mlTarget), "fmri_signature") {
			opts = append(opts, optMLFmriSigGroup)
			if m.mlFmriSigGroupExpanded {
				opts = append(opts,
					optMLFmriSigMethod,
					optMLFmriSigContrastName,
					optMLFmriSigSignature,
					optMLFmriSigMetric,
					optMLFmriSigNormalization,
					optMLFmriSigRoundDecimals,
				)
			}
		}
		opts = append(opts, optMLRequireTrialMlSafe)
	}

	// ── Model & Hyperparameters ───────────────────────────────────
	opts = append(opts, optMLGroupModel)
	if m.mlGroupModelExpanded {
		if mode == "classify" {
			opts = append(opts, optMLClassificationModel, optMLBinaryThresholdEnabled)
			if m.mlBinaryThresholdEnabled {
				opts = append(opts, optMLBinaryThreshold)
			}
			opts = append(opts, optVarianceThresholdGrid)

			switch m.mlClassificationModel {
			case MLClassificationSVM:
				opts = append(opts,
					optMLSvmKernel,
					optMLSvmCGrid,
					optMLSvmGammaGrid,
					optMLSvmClassWeight,
				)
			case MLClassificationLR:
				opts = append(opts, optMLLrPenalty, optMLLrCGrid)
				if m.mlLrPenalty == 2 {
					opts = append(opts, optMLLrL1RatioGrid)
				}
				opts = append(opts, optMLLrMaxIter, optMLLrClassWeight)
			case MLClassificationRF:
				opts = append(opts,
					optRfNEstimators,
					optRfMaxDepthGrid,
					optMLRfMinSamplesSplitGrid,
					optMLRfMinSamplesLeafGrid,
					optMLRfBootstrap,
					optMLRfClassWeight,
				)
			case MLClassificationEnsemble:
				opts = append(opts, optMLEnsembleCalibrate)
			case MLClassificationCNN:
				opts = append(opts, optMLGroupCNN)
				if m.mlGroupCNNExpanded {
					opts = append(opts,
						optMLCnnFilters1,
						optMLCnnFilters2,
						optMLCnnKernelSize1,
						optMLCnnKernelSize2,
						optMLCnnPoolSize,
						optMLCnnDenseUnits,
						optMLCnnDropoutConv,
						optMLCnnDropoutDense,
						optMLCnnBatchSize,
						optMLCnnEpochs,
						optMLCnnLearningRate,
						optMLCnnPatience,
						optMLCnnMinDelta,
						optMLCnnL2Lambda,
						optMLCnnRandomSeed,
					)
				}
			}
		} else if mode != "timegen" && mode != "" {
			if mode != "model_comparison" {
				opts = append(opts, optMLRegressionModel)
			}
			switch m.mlRegressionModel {
			case MLRegressionElasticNet:
				opts = append(opts, optElasticNetAlphaGrid, optElasticNetL1RatioGrid)
			case MLRegressionRidge:
				opts = append(opts, optRidgeAlphaGrid)
			case MLRegressionRF:
				opts = append(opts,
					optRfNEstimators,
					optRfMaxDepthGrid,
					optMLRfMinSamplesSplitGrid,
					optMLRfMinSamplesLeafGrid,
					optMLRfBootstrap,
				)
			}
			opts = append(opts, optVarianceThresholdGrid)
		}
	}

	// ── ML Preprocessing ─────────────────────────────────────────
	opts = append(opts, optMLGroupPreprocessing)
	if m.mlGroupPreprocessingExpanded {
		opts = append(opts,
			optMLImputer,
			optMLPowerTransformerMethod,
			optMLPowerTransformerStandardize,
			optMLDeconfound,
			optMLFeatureSelectionPercentile,
			optMLSpatialRegionsAllowed,
			optMLPCAEnabled,
		)
		if m.mlPCAEnabled {
			opts = append(opts,
				optMLPCANComponents,
				optMLPCAWhiten,
				optMLPCASvdSolver,
				optMLPCARngSeed,
			)
		}
		if mode == "classify" {
			opts = append(opts, optMLClassificationResampler)
			if m.mlClassificationResampler != 0 {
				opts = append(opts, optMLClassificationResamplerSeed)
			}
		}
	}

	// ── Training & CV ─────────────────────────────────────────────
	opts = append(opts, optMLGroupTraining)
	if m.mlGroupTrainingExpanded {
		opts = append(opts,
			optMLNPerm,
			optMLInnerSplits,
			optMLOuterJobs,
			optRNGSeed,
			optMLCvHygieneEnabled,
			optMLCvPermutationScheme,
			optMLCvMinValidPermFraction,
			optMLCvDefaultNBins,
			optMLEvalCIMethod,
			optMLEvalSubjectWeighting,
			optMLEvalBootstrapIterations,
			optMLDataCovariatesStrict,
			optMLDataMaxExcludedSubjectFraction,
			optMLTargetsStrictRegressionContinuous,
			optMLInterpretabilityGroupedOutputs,
		)
		if mode == "incremental_validity" {
			opts = append(opts, optMLIncrementalBaselineAlpha, optMLIncrementalRequireBaselinePredictors)
		}
		if mode == "uncertainty" {
			opts = append(opts, optMLUncertaintyAlpha)
		}
		if mode == "permutation" {
			opts = append(opts, optMLPermNRepeats)
		}
		if mode == "timegen" {
			opts = append(opts, optMLTimeGenMinSubjects, optMLTimeGenMinValidPermFraction)
		}
		if mode == "classify" {
			opts = append(opts, optMLClassMinSubjectsForAUC, optMLClassMaxFailedFoldFraction)
		}
	}

	// ── Output & Plots ────────────────────────────────────────────
	opts = append(opts, optMLGroupOutput)
	if m.mlGroupOutputExpanded {
		opts = append(opts,
			optMLPlotsEnabled,
			optMLPlotFormats,
			optMLPlotDPI,
			optMLPlotTopNFeatures,
			optMLPlotDiagnostics,
		)
	}

	return opts
}
