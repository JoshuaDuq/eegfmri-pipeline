// Machine learning pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderMLAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("machine learning")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter value, Enter to confirm, Esc to cancel") + "\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, Enter to confirm, Esc to cancel") + "\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space: select  Esc: close list") + "\n")
	} else {
		b.WriteString(infoStyle.Render("Space: edit/toggle  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidthWide
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	textFieldForOpt := func(opt optionType) (textField, bool) {
		switch opt {
		case optMLTarget:
			return textFieldMLTarget, true
		case optMLFmriSigContrastName:
			return textFieldMLFmriSigContrastName, true
		case optMLFeatureFamilies:
			return textFieldMLFeatureFamilies, true
		case optMLFeatureBands:
			return textFieldMLFeatureBands, true
		case optMLFeatureSegments:
			return textFieldMLFeatureSegments, true
		case optMLFeatureScopes:
			return textFieldMLFeatureScopes, true
		case optMLFeatureStats:
			return textFieldMLFeatureStats, true
		case optMLCovariates:
			return textFieldMLCovariates, true
		case optMLSpatialRegionsAllowed:
			return textFieldMLSpatialRegionsAllowed, true
		case optMLBaselinePredictors:
			return textFieldMLBaselinePredictors, true
		case optMLPlotFormats:
			return textFieldMLPlotFormats, true
		case optElasticNetAlphaGrid:
			return textFieldElasticNetAlphaGrid, true
		case optElasticNetL1RatioGrid:
			return textFieldElasticNetL1RatioGrid, true
		case optRidgeAlphaGrid:
			return textFieldRidgeAlphaGrid, true
		case optRfMaxDepthGrid:
			return textFieldRfMaxDepthGrid, true
		case optVarianceThresholdGrid:
			return textFieldVarianceThresholdGrid, true
		case optMLSvmCGrid:
			return textFieldMLSvmCGrid, true
		case optMLSvmGammaGrid:
			return textFieldMLSvmGammaGrid, true
		case optMLLrCGrid:
			return textFieldMLLrCGrid, true
		case optMLLrL1RatioGrid:
			return textFieldMLLrL1RatioGrid, true
		case optMLRfMinSamplesSplitGrid:
			return textFieldMLRfMinSamplesSplitGrid, true
		case optMLRfMinSamplesLeafGrid:
			return textFieldMLRfMinSamplesLeafGrid, true
		default:
			return textFieldNone, false
		}
	}

	renderTextOrDefault := func(raw string, emptyLabel string) string {
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			return emptyLabel
		}
		return trimmed
	}

	availableColumns := m.GetAvailableColumns()
	rows := m.getMLOptions()

	totalLines := len(rows)
	if m.expandedOption >= 0 {
		totalLines += len(m.getExpandedListItems())
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)
	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", startLine)) + "\n")
	}

	lineIdx := 0
	for i, opt := range rows {
		isFocused := i == m.advancedCursor

		label := ""
		value := ""
		hint := ""

		switch opt {
		case optUseDefaults:
			label, value, hint = "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"
		case optMLTarget:
			hint = "e.g., rating / temperature / pain_binary"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			label, value = "Target", renderTextOrDefault(m.mlTarget, "(stage default)")
		case optMLFmriSigGroup:
			if m.mlFmriSigGroupExpanded {
				label = "▾ fMRI Signature Target"
			} else {
				label = "▸ fMRI Signature Target"
			}
			value, hint = "", "Space to toggle"
		case optMLFmriSigMethod:
			methods := []string{"beta-series", "lss"}
			label, value, hint = "fMRI Sig Method", methods[m.mlFmriSigMethodIndex%len(methods)], "which fMRI target to load"
		case optMLFmriSigContrastName:
			label, value, hint = "fMRI Contrast Name", renderTextOrDefault(m.mlFmriSigContrastName, "(pain_vs_nonpain)"), "must match fmri/(beta_series|lss)/task-*/contrast-*"
		case optMLFmriSigSignature:
			sigs := []string{"NPS", "SIIPS1"}
			label, value, hint = "fMRI Signature", sigs[m.mlFmriSigSignatureIndex%len(sigs)], "signature used as target"
		case optMLFmriSigMetric:
			metrics := []string{"dot", "cosine", "pearson_r"}
			label, value, hint = "fMRI Sig Metric", metrics[m.mlFmriSigMetricIndex%len(metrics)], "target metric"
		case optMLFmriSigNormalization:
			norms := []string{
				"none",
				"zscore_within_run",
				"zscore_within_subject",
				"robust_zscore_within_run",
				"robust_zscore_within_subject",
			}
			label, value, hint = "Target Normalization", norms[m.mlFmriSigNormalizationIndex%len(norms)], "recommended: within_run"
		case optMLFmriSigRoundDecimals:
			label, value, hint = "Align Round Decimals", fmt.Sprintf("%d", m.mlFmriSigRoundDecimals), "onset/duration rounding for alignment"
		case optMLFeatureFamilies:
			label = "Feature Families"
			value = renderTextOrDefault(m.mlFeatureFamiliesSpec, "(config default)")
			hint = fmt.Sprintf("Space to select · %d options", len(m.mlFeatureFamiliesOptions())-1)
		case optMLFeatureBands:
			label = "Feature Bands"
			value = renderTextOrDefault(m.mlFeatureBandsSpec, "(none)")
			hint = fmt.Sprintf("Space to select · %d options", len(m.mlFeatureBandsOptions())-1)
		case optMLFeatureSegments:
			label = "Feature Segments"
			value = renderTextOrDefault(m.mlFeatureSegmentsSpec, "(none)")
			hint = fmt.Sprintf("Space to select · %d options", len(m.mlFeatureSegmentsOptions())-1)
		case optMLFeatureScopes:
			label = "Feature Scopes"
			value = renderTextOrDefault(m.mlFeatureScopesSpec, "(none)")
			hint = fmt.Sprintf("Space to select · %d options", len(m.mlFeatureScopesOptions())-1)
		case optMLFeatureStats:
			label = "Feature Stats"
			value = renderTextOrDefault(m.mlFeatureStatsSpec, "(none)")
			hint = fmt.Sprintf("Space to select · %d options", len(m.mlFeatureStatsOptions())-1)
		case optMLFeatureHarmonization:
			label, value, hint = "Feature Harmonization", m.mlFeatureHarmonization.Display(), "intersection vs union_impute"
		case optMLCovariates:
			label, value, hint = "Covariates", renderTextOrDefault(m.mlCovariatesSpec, "(none)"), "extra predictors from metadata (optional)"
		case optMLBaselinePredictors:
			label, value, hint = "Baseline Predictors", renderTextOrDefault(m.mlBaselinePredictorsSpec, "(config default)"), "used for incremental validity"
		case optMLRequireTrialMlSafe:
			label, value, hint = "Require trial_ml_safe", m.boolToOnOff(m.mlRequireTrialMlSafe), "fail-fast if feature pipeline isn't ML-safe"
		case optMLPlotsEnabled:
			label, value, hint = "ML Plots Enabled", m.boolToOnOff(m.mlPlotsEnabled), "generate ML result plots"
		case optMLPlotFormats:
			label, value, hint = "ML Plot Formats", renderTextOrDefault(m.mlPlotFormatsSpec, "(png)"), "space/comma-separated: png pdf svg"
		case optMLPlotDPI:
			label, value, hint = "ML Plot DPI", fmt.Sprintf("%d", m.mlPlotDPI), "render resolution"
		case optMLPlotTopNFeatures:
			label, value, hint = "Top N Features", fmt.Sprintf("%d", m.mlPlotTopNFeatures), "for SHAP/permutation plots"
		case optMLPlotDiagnostics:
			label, value, hint = "Include Diagnostics", m.boolToOnOff(m.mlPlotDiagnostics), "adds ROC/PR/calibration and residual panels"
		case optMLRegressionModel:
			label, value, hint = "Regression Model", m.mlRegressionModel.Display(), "elasticnet / ridge / rf"
		case optMLClassificationModel:
			label, value, hint = "Classification Model", m.mlClassificationModel.Display(), "svm / lr / rf / cnn"
		case optMLBinaryThresholdEnabled:
			label, value, hint = "Binary Threshold", m.boolToOnOff(m.mlBinaryThresholdEnabled), "enable fixed threshold for classification"
		case optMLBinaryThreshold:
			label, value, hint = "Threshold Value", fmt.Sprintf("%.6g", m.mlBinaryThreshold), "applies when threshold enabled"
		case optMLNPerm:
			label, value, hint = "Permutations", fmt.Sprintf("%d", m.mlNPerm), "0=disabled; 100+ for p-values"
		case optMLInnerSplits:
			label, value, hint = "Inner CV Splits", fmt.Sprintf("%d", m.innerSplits), "nested CV folds"
		case optMLOuterJobs:
			label, value, hint = "Outer Jobs", fmt.Sprintf("%d", m.outerJobs), "parallelism for outer CV"
		case optRNGSeed:
			label, value, hint = "RNG Seed", m.rngSeedDisplay(), "0=project default"
		case optMLUncertaintyAlpha:
			label, value, hint = "Uncertainty α", fmt.Sprintf("%.6g", m.mlUncertaintyAlpha), "0<α<1 (e.g., 0.1 = 90% PI)"
		case optMLPermNRepeats:
			label, value, hint = "Perm. Repeats", fmt.Sprintf("%d", m.mlPermNRepeats), "permutation importance repeats"
		case optElasticNetAlphaGrid:
			label, value, hint = "ElasticNet α Grid", renderTextOrDefault(m.elasticNetAlphaGrid, "(default)"), "values separated by space/comma"
		case optElasticNetL1RatioGrid:
			label, value, hint = "ElasticNet L1 Grid", renderTextOrDefault(m.elasticNetL1RatioGrid, "(default)"), "values separated by space/comma"
		case optRidgeAlphaGrid:
			label, value, hint = "Ridge α Grid", renderTextOrDefault(m.ridgeAlphaGrid, "(default)"), "values separated by space/comma"
		case optRfNEstimators:
			label, value, hint = "RF N Estimators", fmt.Sprintf("%d", m.rfNEstimators), "number of trees"
		case optRfMaxDepthGrid:
			label, value, hint = "RF Max Depth Grid", renderTextOrDefault(m.rfMaxDepthGrid, "(default)"), "use 'null' for None"
		case optVarianceThresholdGrid:
			label, value, hint = "Variance Threshold Grid", renderTextOrDefault(m.varianceThresholdGrid, "(default)"), "e.g. 0.0 or 0.0,0.01,0.1; use 0.0 only for small train folds"
		case optMLGroupPreprocessing:
			if m.mlGroupPreprocessingExpanded {
				label = "▾ ML Preprocessing"
			} else {
				label = "▸ ML Preprocessing"
			}
			value, hint = "", "imputer, scaler, PCA"
		case optMLImputer:
			imputers := []string{"median", "mean", "most_frequent"}
			label, value, hint = "Imputer", imputers[m.mlImputer%len(imputers)], "missing value strategy"
		case optMLPowerTransformerMethod:
			methods := []string{"yeo-johnson", "box-cox"}
			label, value, hint = "Power Transform", methods[m.mlPowerTransformerMethod%len(methods)], "feature normalization"
		case optMLPowerTransformerStandardize:
			label, value, hint = "PT Standardize", m.boolToOnOff(m.mlPowerTransformerStandardize), "standardize after transform"
		case optMLDeconfound:
			label, value, hint = "Deconfound", m.boolToOnOff(m.mlDeconfound), "regress covariates out of EEG features"
		case optMLFeatureSelectionPercentile:
			label, value, hint = "Feature Select %", fmt.Sprintf("%.6g", m.mlFeatureSelectionPercentile), "percent to keep (e.g. 10.0 = top 10%); 100.0 = all"
		case optMLSpatialRegionsAllowed:
			val := m.mlSpatialRegionsAllowed
			if val == "" {
				val = "<all regions>"
			}
			label, value, hint = "Spatial ROIs", val, "comma-separated regions to keep"
		case optMLClassificationResampler:
			resamplers := []string{"none", "undersample", "smote"}
			label, value, hint = "Resampler", resamplers[m.mlClassificationResampler%len(resamplers)], "balance classes during training"
		case optMLClassificationResamplerSeed:
			label, value, hint = "Resampler Seed", fmt.Sprintf("%d", m.mlClassificationResamplerSeed), "random state for sampling"
		case optMLPCAEnabled:
			label, value, hint = "PCA Enabled", m.boolToOnOff(m.mlPCAEnabled), "dimensionality reduction"
		case optMLPCANComponents:
			label, value, hint = "PCA N Components", fmt.Sprintf("%.6g", m.mlPCANComponents), "variance fraction (e.g. 0.95) or int"
		case optMLPCAWhiten:
			label, value, hint = "PCA Whiten", m.boolToOnOff(m.mlPCAWhiten), "decorrelate components"
		case optMLPCASvdSolver:
			solvers := []string{"auto", "full", "randomized"}
			label, value, hint = "PCA SVD Solver", solvers[m.mlPCASvdSolver%len(solvers)], "SVD algorithm"
		case optMLPCARngSeed:
			label, value, hint = "PCA RNG Seed", fmt.Sprintf("%d", m.mlPCARngSeed), "0=default"
		case optMLSvmKernel:
			kernels := []string{"rbf", "linear", "poly"}
			label, value, hint = "SVM Kernel", kernels[m.mlSvmKernel%len(kernels)], "kernel function"
		case optMLSvmCGrid:
			label, value, hint = "SVM C Grid", renderTextOrDefault(m.mlSvmCGrid, "(default)"), "regularization grid"
		case optMLSvmGammaGrid:
			label, value, hint = "SVM Gamma Grid", renderTextOrDefault(m.mlSvmGammaGrid, "(default)"), "use 'scale' for auto"
		case optMLSvmClassWeight:
			weights := []string{"balanced", "none"}
			label, value, hint = "SVM Class Weight", weights[m.mlSvmClassWeight%len(weights)], "class balancing"
		case optMLLrPenalty:
			penalties := []string{"l2", "l1", "elasticnet"}
			label, value, hint = "LR Penalty", penalties[m.mlLrPenalty%len(penalties)], "regularization type"
		case optMLLrCGrid:
			label, value, hint = "LR C Grid", renderTextOrDefault(m.mlLrCGrid, "(default)"), "inverse regularization"
		case optMLLrL1RatioGrid:
			label, value, hint = "LR L1 Ratio Grid", renderTextOrDefault(m.mlLrL1RatioGrid, "(default)"), "for elasticnet penalty"
		case optMLLrMaxIter:
			label, value, hint = "LR Max Iterations", fmt.Sprintf("%d", m.mlLrMaxIter), "solver convergence"
		case optMLLrClassWeight:
			weights := []string{"balanced", "none"}
			label, value, hint = "LR Class Weight", weights[m.mlLrClassWeight%len(weights)], "class balancing"
		case optMLRfMinSamplesSplitGrid:
			label, value, hint = "RF Min Samples Split", renderTextOrDefault(m.mlRfMinSamplesSplitGrid, "(default)"), "min samples to split node"
		case optMLRfMinSamplesLeafGrid:
			label, value, hint = "RF Min Samples Leaf", renderTextOrDefault(m.mlRfMinSamplesLeafGrid, "(default)"), "min samples per leaf"
		case optMLRfBootstrap:
			label, value, hint = "RF Bootstrap", m.boolToOnOff(m.mlRfBootstrap), "bootstrap sampling"
		case optMLRfClassWeight:
			weights := []string{"balanced", "balanced_subsample", "none"}
			label, value, hint = "RF Class Weight", weights[m.mlRfClassWeight%len(weights)], "class balancing"
		case optMLEnsembleCalibrate:
			label, value, hint = "Ensemble Calibrate", m.boolToOnOff(m.mlEnsembleCalibrate), "calibrate SVM/RF probabilities (soft voting)"
		case optMLGroupCNN:
			if m.mlGroupCNNExpanded {
				label = "▾ CNN Architecture"
			} else {
				label = "▸ CNN Architecture"
			}
			value, hint = "", "convolutional neural network params"
		case optMLCnnFilters1:
			label, value, hint = "Conv1 Filters", fmt.Sprintf("%d", m.mlCnnFilters1), "first conv layer"
		case optMLCnnFilters2:
			label, value, hint = "Conv2 Filters", fmt.Sprintf("%d", m.mlCnnFilters2), "second conv layer"
		case optMLCnnKernelSize1:
			label, value, hint = "Conv1 Kernel Size", fmt.Sprintf("%d", m.mlCnnKernelSize1), "first conv kernel"
		case optMLCnnKernelSize2:
			label, value, hint = "Conv2 Kernel Size", fmt.Sprintf("%d", m.mlCnnKernelSize2), "second conv kernel"
		case optMLCnnPoolSize:
			label, value, hint = "Pool Size", fmt.Sprintf("%d", m.mlCnnPoolSize), "max pooling"
		case optMLCnnDenseUnits:
			label, value, hint = "Dense Units", fmt.Sprintf("%d", m.mlCnnDenseUnits), "FC layer units"
		case optMLCnnDropoutConv:
			label, value, hint = "Conv Dropout", fmt.Sprintf("%.6g", m.mlCnnDropoutConv), "0-1"
		case optMLCnnDropoutDense:
			label, value, hint = "Dense Dropout", fmt.Sprintf("%.6g", m.mlCnnDropoutDense), "0-1"
		case optMLCnnBatchSize:
			label, value, hint = "Batch Size", fmt.Sprintf("%d", m.mlCnnBatchSize), "training batch"
		case optMLCnnEpochs:
			label, value, hint = "Epochs", fmt.Sprintf("%d", m.mlCnnEpochs), "training epochs"
		case optMLCnnLearningRate:
			label, value, hint = "Learning Rate", fmt.Sprintf("%.6g", m.mlCnnLearningRate), "optimizer LR"
		case optMLCnnPatience:
			label, value, hint = "Early Stop Patience", fmt.Sprintf("%d", m.mlCnnPatience), "epochs without improvement"
		case optMLCnnMinDelta:
			label, value, hint = "Early Stop Min Delta", fmt.Sprintf("%.6g", m.mlCnnMinDelta), "minimum improvement"
		case optMLCnnL2Lambda:
			label, value, hint = "L2 Lambda", fmt.Sprintf("%.6g", m.mlCnnL2Lambda), "weight decay"
		case optMLCnnRandomSeed:
			label, value, hint = "CNN Random Seed", fmt.Sprintf("%d", m.mlCnnRandomSeed), "reproducibility"
		case optMLCvHygieneEnabled:
			label, value, hint = "CV Hygiene", m.boolToOnOff(m.mlCvHygieneEnabled), "strict CV data leakage checks"
		case optMLCvPermutationScheme:
			schemes := []string{"within_subject", "within_subject_within_block"}
			label, value, hint = "Perm. Scheme", schemes[m.mlCvPermutationScheme%len(schemes)], "permutation test method"
		case optMLCvMinValidPermFraction:
			label, value, hint = "Min Valid Perm Frac", fmt.Sprintf("%.6g", m.mlCvMinValidPermFraction), "min fraction of valid permutations"
		case optMLCvDefaultNBins:
			label, value, hint = "Default N Bins", fmt.Sprintf("%d", m.mlCvDefaultNBins), "stratification bins"
		case optMLEvalCIMethod:
			methods := []string{"bootstrap", "fixed_effects"}
			label, value, hint = "CI Method", methods[m.mlEvalCIMethod%len(methods)], "confidence interval method"
		case optMLEvalBootstrapIterations:
			label, value, hint = "Bootstrap Iterations", fmt.Sprintf("%d", m.mlEvalBootstrapIterations), "for CI estimation"
		case optMLDataCovariatesStrict:
			label, value, hint = "Covariates Strict", m.boolToOnOff(m.mlDataCovariatesStrict), "error on missing covariates"
		case optMLDataMaxExcludedSubjectFraction:
			label, value, hint = "Max Excluded Subj Frac", fmt.Sprintf("%.6g", m.mlDataMaxExcludedSubjectFraction), "0-1"
		case optMLIncrementalBaselineAlpha:
			label, value, hint = "Baseline Alpha", fmt.Sprintf("%.6g", m.mlIncrementalBaselineAlpha), "incremental validity baseline"
		case optMLInterpretabilityGroupedOutputs:
			label, value, hint = "Grouped Outputs", m.boolToOnOff(m.mlInterpretabilityGroupedOutputs), "grouped importance tables"
		case optMLTargetsStrictRegressionContinuous:
			label, value, hint = "Strict Regression", m.boolToOnOff(m.mlTargetsStrictRegressionCont), "error on binary-like target"
		case optMLTimeGenMinSubjects:
			label, value, hint = "TG Min Subjects", fmt.Sprintf("%d", m.mlTimeGenMinSubjects), "temporal generalization"
		case optMLTimeGenMinValidPermFraction:
			label, value, hint = "TG Min Valid Perm", fmt.Sprintf("%.6g", m.mlTimeGenMinValidPermFraction), "0-1"
		case optMLClassMinSubjectsForAUC:
			label, value, hint = "Min Subj for AUC", fmt.Sprintf("%d", m.mlClassMinSubjectsForAUC), "AUC inference threshold"
		case optMLClassMaxFailedFoldFraction:
			label, value, hint = "Max Failed Fold Frac", fmt.Sprintf("%.6g", m.mlClassMaxFailedFoldFraction), "0-1"
		default:
			continue
		}

		if m.editingNumber && m.isCurrentlyEditing(opt) {
			value = m.numberBuffer + "█"
		}
		if m.editingText {
			if field, ok := textFieldForOpt(opt); ok && m.editingTextField == field {
				value = m.textBuffer + "█"
			}
		}

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}

		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}

		if lineIdx >= startLine && lineIdx < endLine {
			b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(value), hintStyle.Render(hint), labelWidth, m.contentWidth) + "\n")
		}
		lineIdx++

		if m.shouldRenderExpandedListAfterOption(opt) {
			items := m.getExpandedListItems()
			subIndent := "      "
			for j, item := range items {
				isSubFocused := j == m.subCursor
				isSelected := m.isExpandedItemSelected(j, item)
				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)
				itemStyle := lipgloss.NewStyle().Foreground(styles.Text)
				if isSubFocused {
					itemStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
				}
				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + " " + itemStyle.Render(item) + "\n")
				}
				lineIdx++
			}
		}
	}

	if showScrollIndicators && endLine < totalLines {
		remaining := totalLines - endLine
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", remaining)) + "\n")
	}

	return b.String()
}
