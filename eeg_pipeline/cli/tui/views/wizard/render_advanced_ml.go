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
		case optMLBaselinePredictors:
			return textFieldMLBaselinePredictors, true
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
			label, value, hint = "Feature Families", renderTextOrDefault(m.mlFeatureFamiliesSpec, "(config default)"), "e.g., power aperiodic connectivity"
		case optMLFeatureBands:
			label, value, hint = "Feature Bands", renderTextOrDefault(m.mlFeatureBandsSpec, "(none)"), "NamingSchema band, e.g., alpha beta"
		case optMLFeatureSegments:
			label, value, hint = "Feature Segments", renderTextOrDefault(m.mlFeatureSegmentsSpec, "(none)"), "NamingSchema segment, e.g., baseline active"
		case optMLFeatureScopes:
			label, value, hint = "Feature Scopes", renderTextOrDefault(m.mlFeatureScopesSpec, "(none)"), "NamingSchema scope: global roi ch chpair"
		case optMLFeatureStats:
			label, value, hint = "Feature Stats", renderTextOrDefault(m.mlFeatureStatsSpec, "(none)"), "NamingSchema stat, e.g. wpli aec"
		case optMLFeatureHarmonization:
			label, value, hint = "Feature Harmonization", m.mlFeatureHarmonization.Display(), "intersection vs union_impute"
		case optMLCovariates:
			label, value, hint = "Covariates", renderTextOrDefault(m.mlCovariatesSpec, "(none)"), "extra predictors from metadata (optional)"
		case optMLBaselinePredictors:
			label, value, hint = "Baseline Predictors", renderTextOrDefault(m.mlBaselinePredictorsSpec, "(config default)"), "used for incremental validity"
		case optMLRequireTrialMlSafe:
			label, value, hint = "Require trial_ml_safe", m.boolToOnOff(m.mlRequireTrialMlSafe), "fail-fast if feature pipeline isn't ML-safe"
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

		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(value), hintStyle.Render(hint), labelWidth, m.contentWidth) + "\n")

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
				b.WriteString(subIndent + checkbox + " " + itemStyle.Render(item) + "\n")
			}
		}
	}

	return b.String()
}
