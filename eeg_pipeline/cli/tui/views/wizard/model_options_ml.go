package wizard

import "strings"

// Machine-learning advanced option builders.

func (m Model) getMLOptions() []optionType {
	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	opts := []optionType{
		optUseDefaults,
		optMLTarget,
		optMLFeatureFamilies,
		optMLFeatureBands,
		optMLFeatureSegments,
		optMLFeatureScopes,
		optMLFeatureStats,
		optMLFeatureHarmonization,
		optMLCovariates,
	}

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

	opts = append(
		opts,
		optMLRequireTrialMlSafe,
		optMLPlotsEnabled,
		optMLPlotFormats,
		optMLPlotDPI,
		optMLPlotTopNFeatures,
		optMLPlotDiagnostics,
	)

	if mode == "classify" {
		opts = append(opts, optMLClassificationModel, optMLBinaryThresholdEnabled)
		if m.mlBinaryThresholdEnabled {
			opts = append(opts, optMLBinaryThreshold)
		}
		opts = append(opts, optVarianceThresholdGrid)
	} else if mode != "timegen" && mode != "" {
		// Most non-classification stages use the regression model family (timegen is separate).
		if mode != "model_comparison" {
			opts = append(opts, optMLRegressionModel)
		}
		opts = append(
			opts,
			optElasticNetAlphaGrid,
			optElasticNetL1RatioGrid,
			optRidgeAlphaGrid,
			optRfNEstimators,
			optRfMaxDepthGrid,
			optVarianceThresholdGrid,
		)
	}

	// ML Preprocessing group
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
		if mode == "classify" {
			opts = append(opts, optMLClassificationResampler)
			if m.mlClassificationResampler != 0 {
				opts = append(opts, optMLClassificationResamplerSeed)
			}
		}
		if m.mlPCAEnabled {
			opts = append(opts,
				optMLPCANComponents,
				optMLPCAWhiten,
				optMLPCASvdSolver,
				optMLPCARngSeed,
			)
		}
	}

	// SVM hyperparameters (shown when SVM model is selected)
	if mode == "classify" && m.mlClassificationModel == MLClassificationSVM {
		opts = append(opts,
			optMLSvmKernel,
			optMLSvmCGrid,
			optMLSvmGammaGrid,
			optMLSvmClassWeight,
		)
	}

	// Logistic Regression hyperparameters
	if mode == "classify" && m.mlClassificationModel == MLClassificationLR {
		opts = append(opts,
			optMLLrPenalty,
			optMLLrCGrid,
		)
		if m.mlLrPenalty == 2 {
			opts = append(opts, optMLLrL1RatioGrid)
		}
		opts = append(opts,
			optMLLrMaxIter,
			optMLLrClassWeight,
		)
	}

	// Random Forest extras (shown alongside existing RF options)
	if (mode == "classify" && m.mlClassificationModel == MLClassificationRF) ||
		(mode != "classify" && mode != "timegen" && mode != "" && m.mlRegressionModel == MLRegressionRF) {
		opts = append(opts,
			optMLRfMinSamplesSplitGrid,
			optMLRfMinSamplesLeafGrid,
			optMLRfBootstrap,
		)
		if mode == "classify" {
			opts = append(opts, optMLRfClassWeight)
		}
	}

	// Ensemble extras
	if mode == "classify" && m.mlClassificationModel == MLClassificationEnsemble {
		opts = append(opts, optMLEnsembleCalibrate)
	}

	// CNN group (shown for classify mode with CNN model)
	if mode == "classify" && m.mlClassificationModel == MLClassificationCNN {
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

	opts = append(opts, optMLNPerm, optMLInnerSplits, optMLOuterJobs, optRNGSeed)

	// CV / Evaluation / Analysis options
	opts = append(opts,
		optMLCvHygieneEnabled,
		optMLCvPermutationScheme,
		optMLCvMinValidPermFraction,
		optMLCvDefaultNBins,
		optMLEvalCIMethod,
		optMLEvalBootstrapIterations,
		optMLDataCovariatesStrict,
		optMLDataMaxExcludedSubjectFraction,
		optMLTargetsStrictRegressionContinuous,
		optMLInterpretabilityGroupedOutputs,
	)

	if mode == "incremental_validity" {
		opts = append(opts, optMLIncrementalBaselineAlpha)
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

	return opts
}
