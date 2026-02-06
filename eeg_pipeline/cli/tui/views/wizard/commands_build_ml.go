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

	return args
}

// buildPreprocessingAdvancedArgs returns CLI args for preprocessing advanced options
