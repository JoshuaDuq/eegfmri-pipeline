package wizard

import (
	"fmt"
	"strings"
)

// ML pipeline advanced argument builder.

func (m Model) buildMLAdvancedArgs() []string {
	var args []string

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	if strings.TrimSpace(m.mlTarget) != "" {
		args = append(args, "--target", strings.TrimSpace(m.mlTarget))
	}

	if strings.EqualFold(strings.TrimSpace(m.mlTarget), "fmri_signature") {
		methods := []string{"beta-series", "lss"}
		args = append(args, "--fmri-signature-method", methods[m.mlFmriSigMethodIndex%len(methods)])

		contrast := strings.TrimSpace(m.mlFmriSigContrastName)
		if contrast != "" && contrast != "pain_vs_nonpain" {
			args = append(args, "--fmri-signature-contrast-name", contrast)
		}

		sigs := []string{"NPS", "SIIPS1"}
		args = append(args, "--fmri-signature-name", sigs[m.mlFmriSigSignatureIndex%len(sigs)])

		metrics := []string{"dot", "cosine", "pearson_r"}
		args = append(args, "--fmri-signature-metric", metrics[m.mlFmriSigMetricIndex%len(metrics)])

		norms := []string{
			"none",
			"zscore_within_run",
			"zscore_within_subject",
			"robust_zscore_within_run",
			"robust_zscore_within_subject",
		}
		if m.mlFmriSigNormalizationIndex%len(norms) != 0 {
			args = append(args, "--fmri-signature-normalization", norms[m.mlFmriSigNormalizationIndex%len(norms)])
		}
		if m.mlFmriSigRoundDecimals != 3 {
			args = append(args, "--fmri-signature-round-decimals", fmt.Sprintf("%d", m.mlFmriSigRoundDecimals))
		}
	}

	if mode == "classify" && m.mlBinaryThresholdEnabled {
		args = append(args, "--binary-threshold", fmt.Sprintf("%.6g", m.mlBinaryThreshold))
	}

	if strings.TrimSpace(m.mlFeatureFamiliesSpec) != "" {
		args = append(args, "--feature-families")
		args = append(args, splitLooseList(m.mlFeatureFamiliesSpec)...)
	}

	if strings.TrimSpace(m.mlFeatureBandsSpec) != "" {
		args = append(args, "--feature-bands")
		args = append(args, splitLooseList(m.mlFeatureBandsSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureSegmentsSpec) != "" {
		args = append(args, "--feature-segments")
		args = append(args, splitLooseList(m.mlFeatureSegmentsSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureScopesSpec) != "" {
		args = append(args, "--feature-scopes")
		args = append(args, splitLooseList(m.mlFeatureScopesSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureStatsSpec) != "" {
		args = append(args, "--feature-stats")
		args = append(args, splitLooseList(m.mlFeatureStatsSpec)...)
	}

	if v := m.mlFeatureHarmonization.CLIValue(); v != "" {
		args = append(args, "--feature-harmonization", v)
	}

	if strings.TrimSpace(m.mlCovariatesSpec) != "" {
		args = append(args, "--covariates")
		args = append(args, splitLooseList(m.mlCovariatesSpec)...)
	}

	if mode == "incremental_validity" && strings.TrimSpace(m.mlBaselinePredictorsSpec) != "" {
		args = append(args, "--baseline-predictors")
		args = append(args, splitLooseList(m.mlBaselinePredictorsSpec)...)
	}

	if mode == "classify" {
		if v := m.mlClassificationModel.CLIValue(); v != "" {
			args = append(args, "--classification-model", v)
		}
	}

	if m.mlRequireTrialMlSafe {
		args = append(args, "--require-trial-ml-safe")
	}

	plotsEnabled := m.mlPlotsEnabled
	plotFormatsSpec := strings.TrimSpace(m.mlPlotFormatsSpec)
	plotDPI := m.mlPlotDPI
	plotTopN := m.mlPlotTopNFeatures
	plotDiagnostics := m.mlPlotDiagnostics
	// Keep zero-value Model{} behavior consistent with default settings.
	if !plotsEnabled && plotFormatsSpec == "" && plotDPI == 0 && plotTopN == 0 && !plotDiagnostics {
		plotsEnabled = true
		plotDiagnostics = true
	}
	if plotFormatsSpec == "" {
		plotFormatsSpec = "png"
	}
	if plotDPI == 0 {
		plotDPI = 300
	}
	if plotTopN == 0 {
		plotTopN = 20
	}

	if !plotsEnabled {
		args = append(args, "--no-ml-plots")
	}
	plotFormats := splitLooseList(plotFormatsSpec)
	for i := range plotFormats {
		plotFormats[i] = strings.ToLower(strings.TrimSpace(plotFormats[i]))
	}
	if len(plotFormats) > 0 && !(len(plotFormats) == 1 && plotFormats[0] == "png") {
		args = append(args, "--ml-plot-formats")
		args = append(args, plotFormats...)
	}
	if plotDPI != 300 {
		args = append(args, "--ml-plot-dpi", fmt.Sprintf("%d", plotDPI))
	}
	if plotTopN != 20 {
		args = append(args, "--ml-plot-top-n-features", fmt.Sprintf("%d", plotTopN))
	}
	if !plotDiagnostics {
		args = append(args, "--ml-plot-no-diagnostics")
	}

	if mode != "classify" && mode != "timegen" && mode != "model_comparison" && m.mlRegressionModel != MLRegressionElasticNet {
		args = append(args, "--model", m.mlRegressionModel.CLIValue())
	}

	if mode == "uncertainty" && m.mlUncertaintyAlpha != 0.1 {
		args = append(args, "--uncertainty-alpha", fmt.Sprintf("%.6g", m.mlUncertaintyAlpha))
	}

	if mode == "permutation" && m.mlPermNRepeats != 10 {
		args = append(args, "--perm-n-repeats", fmt.Sprintf("%d", m.mlPermNRepeats))
	}

	if m.mlNPerm > 0 {
		args = append(args, "--n-perm", fmt.Sprintf("%d", m.mlNPerm))
	}

	if m.innerSplits != 3 {
		args = append(args, "--inner-splits", fmt.Sprintf("%d", m.innerSplits))
	}

	if m.outerJobs != 1 {
		args = append(args, "--outer-jobs", fmt.Sprintf("%d", m.outerJobs))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	// ElasticNet hyperparameters
	if strings.TrimSpace(m.elasticNetAlphaGrid) != "" && m.elasticNetAlphaGrid != "0.001,0.01,0.1,1,10" {
		args = append(args, "--elasticnet-alpha-grid")
		args = append(args, splitLooseList(m.elasticNetAlphaGrid)...)
	}
	if strings.TrimSpace(m.elasticNetL1RatioGrid) != "" && m.elasticNetL1RatioGrid != "0.2,0.5,0.8" {
		args = append(args, "--elasticnet-l1-ratio-grid")
		args = append(args, splitLooseList(m.elasticNetL1RatioGrid)...)
	}

	// Ridge hyperparameters
	if strings.TrimSpace(m.ridgeAlphaGrid) != "" && m.ridgeAlphaGrid != "0.01,0.1,1,10,100" {
		args = append(args, "--ridge-alpha-grid")
		args = append(args, splitLooseList(m.ridgeAlphaGrid)...)
	}

	// Random Forest hyperparameters
	if m.rfNEstimators != 500 {
		args = append(args, "--rf-n-estimators", fmt.Sprintf("%d", m.rfNEstimators))
	}
	if strings.TrimSpace(m.rfMaxDepthGrid) != "" && m.rfMaxDepthGrid != "5,10,20,null" {
		args = append(args, "--rf-max-depth-grid")
		args = append(args, splitLooseList(m.rfMaxDepthGrid)...)
	}

	if strings.TrimSpace(m.varianceThresholdGrid) != "" && m.varianceThresholdGrid != "0.0,0.01,0.1" {
		args = append(args, "--variance-threshold-grid")
		args = append(args, splitLooseList(m.varianceThresholdGrid)...)
	}

	// ML Preprocessing
	imputers := []string{"median", "mean", "most_frequent"}
	if m.mlImputer != 0 {
		args = append(args, "--imputer", imputers[m.mlImputer%len(imputers)])
	}
	ptMethods := []string{"yeo-johnson", "box-cox"}
	if m.mlPowerTransformerMethod != 0 {
		args = append(args, "--power-transformer-method", ptMethods[m.mlPowerTransformerMethod%len(ptMethods)])
	}
	if !m.mlPowerTransformerStandardize {
		args = append(args, "--no-power-transformer-standardize")
	}
	if m.mlPCAEnabled {
		args = append(args, "--pca-enabled")
		if m.mlPCANComponents != 0.95 {
			args = append(args, "--pca-n-components", fmt.Sprintf("%.6g", m.mlPCANComponents))
		}
		if m.mlPCAWhiten {
			args = append(args, "--pca-whiten")
		}
		svdSolvers := []string{"auto", "full", "randomized"}
		if m.mlPCASvdSolver != 0 {
			args = append(args, "--pca-svd-solver", svdSolvers[m.mlPCASvdSolver%len(svdSolvers)])
		}
		if m.mlPCARngSeed != 0 {
			args = append(args, "--pca-rng-seed", fmt.Sprintf("%d", m.mlPCARngSeed))
		}
	}

	if m.mlDeconfound {
		args = append(args, "--deconfound")
	}

	if m.mlFeatureSelectionPercentile != 100.0 && m.mlFeatureSelectionPercentile > 0.0 {
		args = append(args, "--feature-selection-percentile", fmt.Sprintf("%.6g", m.mlFeatureSelectionPercentile))
	}

	if m.mlSpatialRegionsAllowed != "" {
		args = append(args, "--spatial-regions-allowed")
		args = append(args, splitLooseList(m.mlSpatialRegionsAllowed)...)
	}

	if mode == "classify" {
		resamplers := []string{"none", "undersample", "smote"}
		if m.mlClassificationResampler != 0 {
			args = append(args, "--classification-resampler", resamplers[m.mlClassificationResampler%len(resamplers)])
			if m.mlClassificationResamplerSeed != 42 {
				args = append(args, "--classification-resampler-seed", fmt.Sprintf("%d", m.mlClassificationResamplerSeed))
			}
		}

		if m.mlClassificationModel == MLClassificationEnsemble && m.mlEnsembleCalibrate {
			args = append(args, "--ensemble-calibrate")
		}
	}

	// SVM hyperparameters
	if mode == "classify" && m.mlClassificationModel == MLClassificationSVM {
		kernels := []string{"rbf", "linear", "poly"}
		if m.mlSvmKernel != 0 {
			args = append(args, "--svm-kernel", kernels[m.mlSvmKernel%len(kernels)])
		}
		if m.mlSvmCGrid != "0.01,0.1,1,10,100" {
			args = append(args, "--svm-c-grid")
			args = append(args, splitLooseList(m.mlSvmCGrid)...)
		}
		if m.mlSvmGammaGrid != "scale,0.001,0.01,0.1" {
			args = append(args, "--svm-gamma-grid")
			args = append(args, splitLooseList(m.mlSvmGammaGrid)...)
		}
		classWeights := []string{"balanced", "none"}
		if m.mlSvmClassWeight != 0 {
			args = append(args, "--svm-class-weight", classWeights[m.mlSvmClassWeight%len(classWeights)])
		}
	}

	// Logistic Regression hyperparameters
	if mode == "classify" && m.mlClassificationModel == MLClassificationLR {
		penalties := []string{"l2", "l1", "elasticnet"}
		if m.mlLrPenalty != 0 {
			args = append(args, "--lr-penalty", penalties[m.mlLrPenalty%len(penalties)])
		}
		if m.mlLrCGrid != "0.01,0.1,1,10,100" {
			args = append(args, "--lr-c-grid")
			args = append(args, splitLooseList(m.mlLrCGrid)...)
		}
		if m.mlLrMaxIter != 1000 {
			args = append(args, "--lr-max-iter", fmt.Sprintf("%d", m.mlLrMaxIter))
		}
		classWeights := []string{"balanced", "none"}
		if m.mlLrClassWeight != 0 {
			args = append(args, "--lr-class-weight", classWeights[m.mlLrClassWeight%len(classWeights)])
		}
	}

	// Random Forest extras
	isRF := (mode == "classify" && m.mlClassificationModel == MLClassificationRF) ||
		(mode != "classify" && mode != "timegen" && mode != "" && m.mlRegressionModel == MLRegressionRF)
	if isRF {
		if m.mlRfMinSamplesSplitGrid != "2,5,10" {
			args = append(args, "--rf-min-samples-split-grid")
			args = append(args, splitLooseList(m.mlRfMinSamplesSplitGrid)...)
		}
		if m.mlRfMinSamplesLeafGrid != "1,2,4" {
			args = append(args, "--rf-min-samples-leaf-grid")
			args = append(args, splitLooseList(m.mlRfMinSamplesLeafGrid)...)
		}
		if !m.mlRfBootstrap {
			args = append(args, "--no-rf-bootstrap")
		}
		if mode == "classify" {
			rfWeights := []string{"balanced", "balanced_subsample", "none"}
			if m.mlRfClassWeight != 0 {
				args = append(args, "--rf-class-weight", rfWeights[m.mlRfClassWeight%len(rfWeights)])
			}
		}
	}

	// CNN hyperparameters
	if mode == "classify" && m.mlClassificationModel == MLClassificationCNN {
		if m.mlCnnFilters1 != 32 {
			args = append(args, "--cnn-filters1", fmt.Sprintf("%d", m.mlCnnFilters1))
		}
		if m.mlCnnFilters2 != 64 {
			args = append(args, "--cnn-filters2", fmt.Sprintf("%d", m.mlCnnFilters2))
		}
		if m.mlCnnKernelSize1 != 3 {
			args = append(args, "--cnn-kernel-size1", fmt.Sprintf("%d", m.mlCnnKernelSize1))
		}
		if m.mlCnnKernelSize2 != 3 {
			args = append(args, "--cnn-kernel-size2", fmt.Sprintf("%d", m.mlCnnKernelSize2))
		}
		if m.mlCnnPoolSize != 2 {
			args = append(args, "--cnn-pool-size", fmt.Sprintf("%d", m.mlCnnPoolSize))
		}
		if m.mlCnnDenseUnits != 128 {
			args = append(args, "--cnn-dense-units", fmt.Sprintf("%d", m.mlCnnDenseUnits))
		}
		if m.mlCnnDropoutConv != 0.25 {
			args = append(args, "--cnn-dropout-conv", fmt.Sprintf("%.6g", m.mlCnnDropoutConv))
		}
		if m.mlCnnDropoutDense != 0.5 {
			args = append(args, "--cnn-dropout-dense", fmt.Sprintf("%.6g", m.mlCnnDropoutDense))
		}
		if m.mlCnnBatchSize != 32 {
			args = append(args, "--cnn-batch-size", fmt.Sprintf("%d", m.mlCnnBatchSize))
		}
		if m.mlCnnEpochs != 100 {
			args = append(args, "--cnn-epochs", fmt.Sprintf("%d", m.mlCnnEpochs))
		}
		if m.mlCnnLearningRate != 0.001 {
			args = append(args, "--cnn-learning-rate", fmt.Sprintf("%.6g", m.mlCnnLearningRate))
		}
		if m.mlCnnPatience != 10 {
			args = append(args, "--cnn-patience", fmt.Sprintf("%d", m.mlCnnPatience))
		}
		if m.mlCnnMinDelta != 0.001 {
			args = append(args, "--cnn-min-delta", fmt.Sprintf("%.6g", m.mlCnnMinDelta))
		}
		if m.mlCnnL2Lambda != 0.01 {
			args = append(args, "--cnn-l2-lambda", fmt.Sprintf("%.6g", m.mlCnnL2Lambda))
		}
		if m.mlCnnRandomSeed != 42 {
			args = append(args, "--cnn-random-seed", fmt.Sprintf("%d", m.mlCnnRandomSeed))
		}
	}

	// CV / Evaluation / Analysis
	if !m.mlCvHygieneEnabled {
		args = append(args, "--no-cv-hygiene")
	}
	permSchemes := []string{"within_subject", "within_subject_within_block"}
	if m.mlCvPermutationScheme != 0 {
		args = append(args, "--cv-permutation-scheme", permSchemes[m.mlCvPermutationScheme%len(permSchemes)])
	}
	if m.mlCvMinValidPermFraction != 0.8 {
		args = append(args, "--cv-min-valid-perm-fraction", fmt.Sprintf("%.6g", m.mlCvMinValidPermFraction))
	}
	if m.mlCvDefaultNBins != 5 {
		args = append(args, "--cv-default-n-bins", fmt.Sprintf("%d", m.mlCvDefaultNBins))
	}
	ciMethods := []string{"bootstrap", "fixed_effects"}
	if m.mlEvalCIMethod != 0 {
		args = append(args, "--eval-ci-method", ciMethods[m.mlEvalCIMethod%len(ciMethods)])
	}
	if m.mlEvalBootstrapIterations != 1000 {
		args = append(args, "--eval-bootstrap-iterations", fmt.Sprintf("%d", m.mlEvalBootstrapIterations))
	}
	if m.mlDataCovariatesStrict {
		args = append(args, "--data-covariates-strict")
	}
	if m.mlDataMaxExcludedSubjectFraction != 0.2 {
		args = append(args, "--data-max-excluded-subject-fraction", fmt.Sprintf("%.6g", m.mlDataMaxExcludedSubjectFraction))
	}
	if !m.mlTargetsStrictRegressionCont {
		args = append(args, "--no-strict-regression-continuous")
	}
	if !m.mlInterpretabilityGroupedOutputs {
		args = append(args, "--no-interpretability-grouped-outputs")
	}
	if mode == "incremental_validity" && m.mlIncrementalBaselineAlpha != 0.05 {
		args = append(args, "--incremental-baseline-alpha", fmt.Sprintf("%.6g", m.mlIncrementalBaselineAlpha))
	}
	if mode == "timegen" {
		if m.mlTimeGenMinSubjects != 5 {
			args = append(args, "--timegen-min-subjects", fmt.Sprintf("%d", m.mlTimeGenMinSubjects))
		}
		if m.mlTimeGenMinValidPermFraction != 0.5 {
			args = append(args, "--timegen-min-valid-perm-fraction", fmt.Sprintf("%.6g", m.mlTimeGenMinValidPermFraction))
		}
	}
	if mode == "classify" {
		if m.mlClassMinSubjectsForAUC != 10 {
			args = append(args, "--class-min-subjects-for-auc", fmt.Sprintf("%d", m.mlClassMinSubjectsForAUC))
		}
		if m.mlClassMaxFailedFoldFraction != 0.2 {
			args = append(args, "--class-max-failed-fold-fraction", fmt.Sprintf("%.6g", m.mlClassMaxFailedFoldFraction))
		}
	}

	return args
}

// buildPreprocessingAdvancedArgs returns CLI args for preprocessing advanced options
