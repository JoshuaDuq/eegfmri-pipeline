// Behavior pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderBehaviorAdvancedConfig() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Advanced configuration") + "\n\n")

	// Contextual help text (same as features pipeline)
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("behavior analysis")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter a value, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space to select item · ↑↓ to navigate · Esc to close list") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to toggle/expand · ↑↓ to navigate · Enter to proceed") + "\n\n")
	}

	labelWidth := defaultLabelWidthWide
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	options := m.getBehaviorOptions()
	availableColumns := m.GetAvailableColumns()

	getOptionDisplay := func(opt optionType) (string, string, string) {
		numberDisplay := m.numberBuffer + "█"
		textDisplay := m.textBuffer + "█"

		switch opt {
		case optUseDefaults:
			return "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"
		// Behavior section headers
		case optBehaviorGroupGeneral:
			label := "▸ General"
			if m.behaviorGroupGeneralExpanded {
				label = "▾ General"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupTrialTable:
			label := "▸ Trial Table"
			if m.behaviorGroupTrialTableExpanded {
				label = "▾ Trial Table"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupPainResidual:
			label := "▸ Pain Residual"
			if m.behaviorGroupPainResidualExpanded {
				label = "▾ Pain Residual"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCorrelations:
			label := "▸ Correlations"
			if m.behaviorGroupCorrelationsExpanded {
				label = "▾ Correlations"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupRegression:
			label := "▸ Regression"
			if m.behaviorGroupRegressionExpanded {
				label = "▾ Regression"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupModels:
			label := "▸ Models"
			if m.behaviorGroupModelsExpanded {
				label = "▾ Models"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupStability:
			label := "▸ Stability"
			if m.behaviorGroupStabilityExpanded {
				label = "▾ Stability"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupConsistency:
			label := "▸ Consistency"
			if m.behaviorGroupConsistencyExpanded {
				label = "▾ Consistency"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupInfluence:
			label := "▸ Influence"
			if m.behaviorGroupInfluenceExpanded {
				label = "▾ Influence"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupReport:
			label := "▸ Report"
			if m.behaviorGroupReportExpanded {
				label = "▾ Report"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCondition:
			label := "▸ Condition"
			if m.behaviorGroupConditionExpanded {
				label = "▾ Condition"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupTemporal:
			label := "▸ Temporal"
			if m.behaviorGroupTemporalExpanded {
				label = "▾ Temporal"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCluster:
			label := "▸ Cluster"
			if m.behaviorGroupClusterExpanded {
				label = "▾ Cluster"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupMediation:
			label := "▸ Mediation"
			if m.behaviorGroupMediationExpanded {
				label = "▾ Mediation"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupModeration:
			label := "▸ Moderation"
			if m.behaviorGroupModerationExpanded {
				label = "▾ Moderation"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupMixedEffects:
			label := "▸ Mixed Effects"
			if m.behaviorGroupMixedEffectsExpanded {
				label = "▾ Mixed Effects"
			}
			return label, "", "Space to toggle"
		case optCorrMethod:
			return "Correlation Method", m.correlationMethod, "spearman / pearson"
		case optRobustCorrelation:
			methods := []string{"none", "percentage_bend", "winsorized", "shepherd"}
			v := "none"
			if m.robustCorrelation >= 0 && m.robustCorrelation < len(methods) {
				v = methods[m.robustCorrelation]
			}
			return "Robust Correlation", v, "robust alternative for outliers"
		case optBootstrap:
			val := fmt.Sprintf("%d", m.bootstrapSamples)
			if m.editingNumber && m.isCurrentlyEditing(optBootstrap) {
				val = numberDisplay
			}
			return "Bootstrap Samples", val, "0=disabled"
		case optNPerm:
			val := fmt.Sprintf("%d", m.nPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optNPerm) {
				val = numberDisplay
			}
			return "Permutations", val, "cluster/global permutations"
		case optRNGSeed:
			val := m.rngSeedDisplay()
			if m.editingNumber && m.isCurrentlyEditing(optRNGSeed) {
				val = numberDisplay
			}
			return "RNG Seed", val, "0=project default"
		case optBehaviorNJobs:
			val := fmt.Sprintf("%d", m.behaviorNJobs)
			if m.editingNumber && m.isCurrentlyEditing(optBehaviorNJobs) {
				val = numberDisplay
			}
			return "N Jobs", val, "-1=all cores"
		case optControlTemp:
			return "Control Temperature", m.boolToOnOff(m.controlTemperature), "partial-correlation covariate"
		case optControlOrder:
			return "Control Trial Order", m.boolToOnOff(m.controlTrialOrder), "partial-correlation covariate"
		case optRunAdjustmentEnabled:
			return "Run Adjustment", m.boolToOnOff(m.runAdjustmentEnabled), "run-aware controls/aggregation"
		case optRunAdjustmentColumn:
			val := m.runAdjustmentColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: run_id)"
			}
			if m.editingText && m.editingTextField == textFieldRunAdjustmentColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if !m.runAdjustmentEnabled {
				hint = "run identifier column (enable Run Adjustment)"
			} else if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Run Column", val, hint
		case optRunAdjustmentIncludeInCorrelations:
			if !m.runAdjustmentEnabled {
				return "Run Dummies in Corr", "N/A", "enable Run Adjustment"
			}
			return "Run Dummies in Corr", m.boolToOnOff(m.runAdjustmentIncludeInCorrelations), "add to partial covariates"
		case optRunAdjustmentMaxDummies:
			if !m.runAdjustmentEnabled {
				return "Max Run Dummies", "N/A", "enable Run Adjustment"
			}
			val := fmt.Sprintf("%d", m.runAdjustmentMaxDummies)
			if m.editingNumber && m.isCurrentlyEditing(optRunAdjustmentMaxDummies) {
				val = numberDisplay
			}
			return "Max Run Dummies", val, "skip if > N levels"
		case optFDRAlpha:
			val := fmt.Sprintf("%.3f", m.fdrAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optFDRAlpha) {
				val = numberDisplay
			}
			return "FDR Alpha", val, "multiple comparison threshold"
		case optComputeChangeScores:
			return "Change Scores", m.boolToOnOff(m.behaviorComputeChangeScores), "Δ rating / Δ temperature"
		case optComputeLosoStability:
			return "LOSO Stability", m.boolToOnOff(m.behaviorComputeLosoStability), "leave-one-out stability"
		case optComputeBayesFactors:
			return "Bayes Factors", m.boolToOnOff(m.behaviorComputeBayesFactors), "optional BF reporting"
		case optFeatureQCEnabled:
			return "Feature QC", m.boolToOnOff(m.featureQCEnabled), "pre-filter features (optional gating)"
		case optFeatureQCMaxMissingPct:
			if !m.featureQCEnabled {
				return "QC Max Missing %", "N/A", "enable Feature QC"
			}
			val := fmt.Sprintf("%.2f", m.featureQCMaxMissingPct)
			if m.editingNumber && m.isCurrentlyEditing(optFeatureQCMaxMissingPct) {
				val = numberDisplay
			}
			return "QC Max Missing %", val, "fraction missing allowed"
		case optFeatureQCMinVariance:
			if !m.featureQCEnabled {
				return "QC Min Variance", "N/A", "enable Feature QC"
			}
			val := fmt.Sprintf("%.2e", m.featureQCMinVariance)
			if m.editingNumber && m.isCurrentlyEditing(optFeatureQCMinVariance) {
				val = numberDisplay
			}
			return "QC Min Variance", val, "drop near-constant features"
		case optFeatureQCCheckWithinRunVariance:
			if !m.featureQCEnabled {
				return "QC Within-Run Var", "N/A", "enable Feature QC"
			}
			return "QC Within-Run Var", m.boolToOnOff(m.featureQCCheckWithinRunVariance), "check per-run variance"

		// Trial table
		case optTrialTableFormat:
			v := "parquet"
			if m.trialTableFormat == 1 {
				v = "tsv"
			}
			return "Trial Table Format", v, "parquet recommended"
		case optTrialTableAddLagFeatures:
			return "Lag/Delta Columns", m.boolToOnOff(m.trialTableAddLagFeatures), "prev_* and delta_*"
		case optTrialOrderMaxMissingFraction:
			if !m.controlTrialOrder {
				return "Trial Order Max Missing", "N/A", "enable Control Trial Order"
			}
			val := fmt.Sprintf("%.2f", m.trialOrderMaxMissingFraction)
			if m.editingNumber && m.isCurrentlyEditing(optTrialOrderMaxMissingFraction) {
				val = numberDisplay
			}
			return "Trial Order Max Missing", val, "disable control if missing > threshold"

		// Pain residual
		case optPainResidualEnabled:
			return "Pain Residual", m.boolToOnOff(m.painResidualEnabled), "rating - f(temp)"
		case optPainResidualMethod:
			v := "spline"
			if m.painResidualMethod == 1 {
				v = "poly"
			}
			return "Residual Method", v, "spline preferred"
		case optPainResidualPolyDegree:
			val := fmt.Sprintf("%d", m.painResidualPolyDegree)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualPolyDegree) {
				val = numberDisplay
			}
			return "Poly Degree", val, "poly fallback degree"
		case optPainResidualSplineDfCandidates:
			val := m.painResidualSplineDfCandidates
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualSplineDfCandidates {
				val = textDisplay
			}
			return "Spline DF Candidates", val, "comma-separated (e.g., 3,4,5)"
		case optPainResidualModelCompare:
			return "Temp Model Compare", m.boolToOnOff(m.painResidualModelCompareEnabled), "non-gating diagnostics"
		case optPainResidualModelComparePolyDegrees:
			val := m.painResidualModelComparePolyDegrees
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualModelComparePolyDegrees {
				val = textDisplay
			}
			return "Model Compare Poly Deg", val, "comma-separated (e.g., 2,3)"
		case optPainResidualBreakpoint:
			return "Breakpoint Test", m.boolToOnOff(m.painResidualBreakpointEnabled), "single-hinge model"
		case optPainResidualBreakpointCandidates:
			val := fmt.Sprintf("%d", m.painResidualBreakpointCandidates)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointCandidates) {
				val = numberDisplay
			}
			return "Breakpoint Candidates", val, "grid size"
		case optPainResidualBreakpointQlow:
			val := fmt.Sprintf("%.2f", m.painResidualBreakpointQlow)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointQlow) {
				val = numberDisplay
			}
			return "Breakpoint Q Low", val, "quantile bound"
		case optPainResidualBreakpointQhigh:
			val := fmt.Sprintf("%.2f", m.painResidualBreakpointQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointQhigh) {
				val = numberDisplay
			}
			return "Breakpoint Q High", val, "quantile bound"
		case optPainResidualCrossfitEnabled:
			if !m.painResidualEnabled {
				return "Residual Crossfit", "N/A", "enable Pain Residual"
			}
			return "Residual Crossfit", m.boolToOnOff(m.painResidualCrossfitEnabled), "out-of-run temperature→rating"
		case optPainResidualCrossfitGroupColumn:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Group Col", "N/A", "enable Residual Crossfit"
			}
			val := m.painResidualCrossfitGroupColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: run column)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualCrossfitGroupColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			} else {
				hint = "Space to edit (blank=run column)"
			}
			return "Crossfit Group Col", val, hint
		case optPainResidualCrossfitNSplits:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Splits", "N/A", "enable Residual Crossfit"
			}
			val := fmt.Sprintf("%d", m.painResidualCrossfitNSplits)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualCrossfitNSplits) {
				val = numberDisplay
			}
			return "Crossfit Splits", val, "n_splits (>=2)"
		case optPainResidualCrossfitMethod:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Method", "N/A", "enable Residual Crossfit"
			}
			v := "spline"
			if m.painResidualCrossfitMethod == 1 {
				v = "poly"
			}
			return "Crossfit Method", v, "spline | poly"
		case optPainResidualCrossfitSplineKnots:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Knots", "N/A", "enable Residual Crossfit"
			}
			if m.painResidualCrossfitMethod == 1 {
				return "Crossfit Knots", "N/A", "poly method"
			}
			val := fmt.Sprintf("%d", m.painResidualCrossfitSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualCrossfitSplineKnots) {
				val = numberDisplay
			}
			return "Crossfit Knots", val, "spline knots (>=3)"

		// Regression
		case optRegressionOutcome:
			v := "rating"
			switch m.regressionOutcome {
			case 1:
				v = "pain_residual"
			case 2:
				v = "temperature"
			}
			return "Outcome", v, "dependent variable"
		case optRegressionIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.regressionIncludeTemperature), "add temperature covariate"
		case optRegressionTempControl:
			v := "linear"
			switch m.regressionTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optRegressionTempSplineKnots:
			val := fmt.Sprintf("%d", m.regressionTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optRegressionTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.regressionTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optRegressionTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.regressionTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optRegressionIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.regressionIncludeTrialOrder), "add trial_index covariate"
		case optRegressionIncludePrev:
			return "Prev/Delta Terms", m.boolToOnOff(m.regressionIncludePrev), "use prev_/delta_"
		case optRegressionIncludeRunBlock:
			return "Run/Block Dummies", m.boolToOnOff(m.regressionIncludeRunBlock), "categorical controls"
		case optRegressionIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.regressionIncludeInteraction), "moderation term"
		case optRegressionStandardize:
			return "Standardize", m.boolToOnOff(m.regressionStandardize), "z-score predictors"
		case optRegressionPermutations:
			val := fmt.Sprintf("%d", m.regressionPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionPermutations) {
				val = numberDisplay
			}
			return "Permutations", val, "Freedman–Lane (0=off)"
		case optRegressionMaxFeatures:
			val := fmt.Sprintf("%d", m.regressionMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"

		// Models
		case optModelsIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.modelsIncludeTemperature), "add temperature covariate"
		case optModelsTempControl:
			v := "linear"
			switch m.modelsTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optModelsTempSplineKnots:
			val := fmt.Sprintf("%d", m.modelsTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optModelsTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.modelsTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optModelsTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.modelsTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optModelsIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.modelsIncludeTrialOrder), "add trial_index covariate"
		case optModelsIncludePrev:
			return "Prev/Delta Terms", m.boolToOnOff(m.modelsIncludePrev), "use prev_/delta_"
		case optModelsIncludeRunBlock:
			return "Run/Block Dummies", m.boolToOnOff(m.modelsIncludeRunBlock), "categorical controls"
		case optModelsIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.modelsIncludeInteraction), "moderation term"
		case optModelsStandardize:
			return "Standardize", m.boolToOnOff(m.modelsStandardize), "z-score predictors"
		case optModelsMaxFeatures:
			val := fmt.Sprintf("%d", m.modelsMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optModelsMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"
		case optModelsOutcomeRating:
			return "Outcome: rating", m.boolToOnOff(m.modelsOutcomeRating), "include rating"
		case optModelsOutcomePainResidual:
			return "Outcome: pain_residual", m.boolToOnOff(m.modelsOutcomePainResidual), "include pain residual"
		case optModelsOutcomeTemperature:
			return "Outcome: temperature", m.boolToOnOff(m.modelsOutcomeTemperature), "include temperature"
		case optModelsOutcomePainBinary:
			return "Outcome: pain_binary", m.boolToOnOff(m.modelsOutcomePainBinary), "include binary outcome"
		case optModelsFamilyOLS:
			return "Family: OLS-HC3", m.boolToOnOff(m.modelsFamilyOLS), "ols_hc3"
		case optModelsFamilyRobust:
			return "Family: Robust", m.boolToOnOff(m.modelsFamilyRobust), "robust_rlm"
		case optModelsFamilyQuantile:
			return "Family: Quantile", m.boolToOnOff(m.modelsFamilyQuantile), "quantile_50"
		case optModelsFamilyLogit:
			return "Family: Logistic", m.boolToOnOff(m.modelsFamilyLogit), "logit"
		case optModelsBinaryOutcome:
			v := "pain_binary"
			if m.modelsBinaryOutcome == 1 {
				v = "rating_median"
			}
			return "Binary Outcome", v, "for logit models"

		// Stability
		case optStabilityMethod:
			v := "spearman"
			if m.stabilityMethod == 1 {
				v = "pearson"
			}
			return "Method", v, "within-group correlation"
		case optStabilityOutcome:
			v := "auto"
			switch m.stabilityOutcome {
			case 1:
				v = "rating"
			case 2:
				v = "pain_residual"
			}
			return "Outcome", v, "auto prefers pain_residual"
		case optStabilityGroupColumn:
			v := "(auto)"
			switch m.stabilityGroupColumn {
			case 1:
				v = "run"
			case 2:
				v = "block"
			}
			return "Group Column", v, "Space to select"
		case optStabilityPartialTemp:
			return "Partial Temperature", m.boolToOnOff(m.stabilityPartialTemp), "control temperature"
		case optStabilityMaxFeatures:
			val := fmt.Sprintf("%d", m.stabilityMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optStabilityMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"
		case optStabilityAlpha:
			val := fmt.Sprintf("%.3f", m.stabilityAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optStabilityAlpha) {
				val = numberDisplay
			}
			return "Alpha", val, "stability cutoff"

		// Consistency
		case optConsistencyEnabled:
			return "Consistency Summary", m.boolToOnOff(m.consistencyEnabled), "flag sign flips"

		// Influence
		case optInfluenceOutcomeRating:
			return "Outcome: rating", m.boolToOnOff(m.influenceOutcomeRating), "include rating"
		case optInfluenceOutcomePainResidual:
			return "Outcome: pain_residual", m.boolToOnOff(m.influenceOutcomePainResidual), "include residual"
		case optInfluenceOutcomeTemperature:
			return "Outcome: temperature", m.boolToOnOff(m.influenceOutcomeTemperature), "include temperature"
		case optInfluenceMaxFeatures:
			val := fmt.Sprintf("%d", m.influenceMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "top effects to inspect"
		case optInfluenceIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.influenceIncludeTemperature), "add covariate"
		case optInfluenceTempControl:
			v := "linear"
			switch m.influenceTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optInfluenceTempSplineKnots:
			val := fmt.Sprintf("%d", m.influenceTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optInfluenceTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.influenceTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optInfluenceTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.influenceTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optInfluenceIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.influenceIncludeTrialOrder), "add covariate"
		case optInfluenceIncludeRunBlock:
			return "Include Run/Block", m.boolToOnOff(m.influenceIncludeRunBlock), "categorical controls"
		case optInfluenceIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.influenceIncludeInteraction), "moderation term"
		case optInfluenceStandardize:
			return "Standardize", m.boolToOnOff(m.influenceStandardize), "z-score predictors"
		case optInfluenceCooksThreshold:
			val := "auto"
			if m.influenceCooksThreshold > 0 {
				val = fmt.Sprintf("%.4f", m.influenceCooksThreshold)
			}
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceCooksThreshold) {
				val = numberDisplay
			}
			return "Cook's Threshold", val, "0=auto heuristic"
		case optInfluenceLeverageThreshold:
			val := "auto"
			if m.influenceLeverageThreshold > 0 {
				val = fmt.Sprintf("%.4f", m.influenceLeverageThreshold)
			}
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceLeverageThreshold) {
				val = numberDisplay
			}
			return "Leverage Threshold", val, "0=auto heuristic"

		// Condition
		case optConditionCompareColumn:
			val := m.conditionCompareColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Compare Column", val, hint
		case optConditionCompareWindows:
			val := m.conditionCompareWindows
			if val == "" {
				val = "(select windows)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareWindows {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableWindows) > 0 {
				hint = fmt.Sprintf("Space to select · %d windows available", len(m.availableWindows))
			}
			return "Compare Windows", val, hint
		case optConditionCompareValues:
			if m.conditionCompareColumn == "" {
				return "Compare Values", "(select column first)", "requires column selection"
			}
			val := m.conditionCompareValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.conditionCompareColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.conditionCompareColumn)
			}
			return "Compare Values", val, hint
		case optConditionWindowPrimaryUnit:
			v := "trial"
			if m.conditionWindowPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Window Unit", v, "trial | run_mean"
		case optConditionPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.conditionPermutationPrimary), "within-run/block when available"
		case optConditionFailFast:
			return "Fail Fast", m.boolToOnOff(m.conditionFailFast), "error if split fails"
		case optConditionOverwrite:
			return "Overwrite", m.boolToOnOff(m.conditionOverwrite), "overwrite existing condition effects files"
		case optConditionEffectThreshold:
			val := fmt.Sprintf("%.3f", m.conditionEffectThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optConditionEffectThreshold) {
				val = numberDisplay
			}
			return "Effect Threshold", val, "Cohen's d cutoff"

		// Temporal
		case optTemporalResolutionMs:
			val := fmt.Sprintf("%d", m.temporalResolutionMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalResolutionMs) {
				val = numberDisplay
			}
			return "Time Resolution (ms)", val, "bin size"
		case optTemporalTimeMinMs:
			val := fmt.Sprintf("%d", m.temporalTimeMinMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTimeMinMs) {
				val = numberDisplay
			}
			return "Time Min (ms)", val, "window start"
		case optTemporalTimeMaxMs:
			val := fmt.Sprintf("%d", m.temporalTimeMaxMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTimeMaxMs) {
				val = numberDisplay
			}
			return "Time Max (ms)", val, "window end"
		case optTemporalSmoothMs:
			val := fmt.Sprintf("%d", m.temporalSmoothMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalSmoothMs) {
				val = numberDisplay
			}
			return "Smooth Window (ms)", val, "smoothing length"
		case optTemporalTargetColumn:
			val := m.temporalTargetColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: rating)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalTargetColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Target Column", val, hint
		case optTemporalSplitByCondition:
			return "Split by Condition", m.boolToOnOff(m.temporalSplitByCondition), "separate files per condition"
		case optTemporalConditionColumn:
			val := m.temporalConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalConditionColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Condition Column", val, hint
		case optTemporalConditionValues:
			if !m.temporalSplitByCondition {
				return "Condition Values", "N/A", "enable Split by Condition"
			}
			if m.temporalConditionColumn == "" {
				return "Condition Values", "(select column first)", "requires column selection"
			}
			val := m.temporalConditionValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalConditionValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.temporalConditionColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.temporalConditionColumn)
			}
			return "Condition Values", val, hint
		case optTemporalIncludeROIAverages:
			return "Include ROI Averages", m.boolToOnOff(m.temporalIncludeROIAverages), "add ROI-averaged rows to output"
		case optTemporalIncludeTFGrid:
			return "Include TF Grid", m.boolToOnOff(m.temporalIncludeTFGrid), "add individual frequency rows"

		// Temporal feature selection
		case optTemporalFeaturePower:
			return "Feature: Power", m.boolToOnOff(m.temporalFeaturePowerEnabled), "spectral power in bands"
		case optTemporalFeatureITPC:
			return "Feature: ITPC", m.boolToOnOff(m.temporalFeatureITPCEnabled), "inter-trial phase coherence"
		case optTemporalFeatureERDS:
			return "Feature: ERDS", m.boolToOnOff(m.temporalFeatureERDSEnabled), "event-related desync/sync"

		// ITPC-specific options
		case optTemporalITPCBaselineCorrection:
			return "ITPC Baseline Correction", m.boolToOnOff(m.temporalITPCBaselineCorrection), "subtract baseline ITPC"
		case optTemporalITPCBaselineMin:
			val := fmt.Sprintf("%.2f", m.temporalITPCBaselineMin)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalITPCBaselineMin) {
				val = numberDisplay
			}
			return "ITPC Baseline Start", val, "seconds"
		case optTemporalITPCBaselineMax:
			val := fmt.Sprintf("%.2f", m.temporalITPCBaselineMax)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalITPCBaselineMax) {
				val = numberDisplay
			}
			return "ITPC Baseline End", val, "seconds"

		// ERDS-specific options
		case optTemporalERDSBaselineMin:
			val := fmt.Sprintf("%.2f", m.temporalERDSBaselineMin)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalERDSBaselineMin) {
				val = numberDisplay
			}
			return "ERDS Baseline Start", val, "seconds"
		case optTemporalERDSBaselineMax:
			val := fmt.Sprintf("%.2f", m.temporalERDSBaselineMax)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalERDSBaselineMax) {
				val = numberDisplay
			}
			return "ERDS Baseline End", val, "seconds"
		case optTemporalERDSMethod:
			methods := []string{"percent", "zscore"}
			var v string
			if m.temporalERDSMethod >= 0 && m.temporalERDSMethod < len(methods) {
				v = methods[m.temporalERDSMethod]
			} else {
				v = "percent"
			}
			return "ERDS Method", v, "ERDS normalization"

		// TF Heatmap options
		case optTemporalTfHeatmapEnabled:
			return "TF Heatmap", m.boolToOnOff(m.tfHeatmapEnabled), "time-frequency correlation heatmap"
		case optTemporalTfHeatmapFreqs:
			val := m.tfHeatmapFreqsSpec
			if val == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldTfHeatmapFreqs {
				val = textDisplay
			}
			return "TF Freqs", val, "frequencies for heatmap"
		case optTemporalTfHeatmapTimeResMs:
			val := fmt.Sprintf("%d ms", m.tfHeatmapTimeResMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTfHeatmapTimeResMs) {
				val = numberDisplay
			}
			return "TF Time Res", val, "temporal resolution"

		// Report
		case optReportTopN:
			val := fmt.Sprintf("%d", m.reportTopN)
			if m.editingNumber && m.isCurrentlyEditing(optReportTopN) {
				val = numberDisplay
			}
			return "Top N Rows", val, "per TSV in report"

		// Correlations
		case optCorrelationsTargetRating:
			return "Target: rating", m.boolToOnOff(m.correlationsTargetRating), "include rating"
		case optCorrelationsTargetTemperature:
			return "Target: temperature", m.boolToOnOff(m.correlationsTargetTemperature), "include temperature"
		case optCorrelationsTargetPainResidual:
			return "Target: pain_residual", m.boolToOnOff(m.correlationsTargetPainResidual), "include residual"
		case optCorrelationsPreferPainResidual:
			return "Prefer pain_residual", m.boolToOnOff(m.correlationsPreferPainResidual), "auto target selection"
		case optCorrelationsTypes:
			val := m.correlationsTypesSpec
			if strings.TrimSpace(val) == "" {
				val = "(default: partial_cov_temp)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTypes {
				val = textDisplay
			}
			return "Correlation Types", val, "comma-separated: raw,partial_cov,partial_temp,partial_cov_temp,run_mean"
		case optCorrelationsUseCrossfitPainResidual:
			return "Use pain_residual_cv", m.boolToOnOff(m.correlationsUseCrossfitResidual), "requires residual crossfit"
		case optCorrelationsPrimaryUnit:
			v := "trial"
			if m.correlationsPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
		case optCorrelationsPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.correlationsPermutationPrimary), "within-run/block when available"
		case optCorrelationsTargetColumn:
			val := m.correlationsTargetColumn
			if strings.TrimSpace(val) == "" {
				val = "(not set)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTargetColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Custom Target Column", val, hint
		case optCorrelationsMultilevel:
			enabled := m.isComputationSelected("multilevel_correlations")
			val := "No"
			if enabled {
				val = "Yes"
			}
			return "Group Multilevel Correlations", val, "Space to toggle"
		case optGroupLevelBlockPermutation:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Block Permutation", "N/A", "enable Group Multilevel Correlations"
			}
			return "Block Permutation", m.boolToOnOff(m.groupLevelBlockPermutation), "block-restricted when available"

		// Cluster
		case optClusterThreshold:
			val := fmt.Sprintf("%.4f", m.clusterThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optClusterThreshold) {
				val = numberDisplay
			}
			return "Cluster Threshold", val, "forming threshold"
		case optClusterMinSize:
			val := fmt.Sprintf("%d", m.clusterMinSize)
			if m.editingNumber && m.isCurrentlyEditing(optClusterMinSize) {
				val = numberDisplay
			}
			return "Min Cluster Size", val, "minimum cluster size"
		case optClusterTail:
			v := "two-tailed"
			switch m.clusterTail {
			case 1:
				v = "upper"
			case -1:
				v = "lower"
			}
			return "Cluster Tail", v, "test direction"
		case optClusterConditionColumn:
			val := m.clusterConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldClusterConditionColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Cluster Column", val, hint
		case optClusterConditionValues:
			if m.clusterConditionColumn == "" {
				return "Cluster Values", "(select column first)", "requires column selection"
			}
			val := m.clusterConditionValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldClusterConditionValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.clusterConditionColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.clusterConditionColumn)
			}
			return "Cluster Values", val, hint

		// Mediation
		case optMediationBootstrap:
			val := fmt.Sprintf("%d", m.mediationBootstrap)
			if m.editingNumber && m.isCurrentlyEditing(optMediationBootstrap) {
				val = numberDisplay
			}
			return "Mediation Bootstrap", val, "bootstrap iterations"
		case optMediationPermutations:
			val := fmt.Sprintf("%d", m.mediationPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optMediationPermutations) {
				val = numberDisplay
			}
			return "Mediation Permutations", val, "0=disabled"
		case optMediationMinEffect:
			val := fmt.Sprintf("%.3f", m.mediationMinEffect)
			if m.editingNumber && m.isCurrentlyEditing(optMediationMinEffect) {
				val = numberDisplay
			}
			return "Min Effect Size", val, "minimum indirect effect"
		case optMediationMaxMediatorsEnabled:
			return "Limit Max Mediators", m.boolToOnOff(m.mediationMaxMediatorsEnabled), "enable mediator limit"
		case optMediationMaxMediators:
			if !m.mediationMaxMediatorsEnabled {
				return "Max Mediators", "N/A", "limit disabled"
			}
			val := fmt.Sprintf("%d", m.mediationMaxMediators)
			if m.editingNumber && m.isCurrentlyEditing(optMediationMaxMediators) {
				val = numberDisplay
			}
			return "Max Mediators", val, "max mediators tested"

		// Moderation
		case optModerationMaxFeaturesEnabled:
			return "Limit Max Features", m.boolToOnOff(m.moderationMaxFeaturesEnabled), "enable feature limit"
		case optModerationMaxFeatures:
			if !m.moderationMaxFeaturesEnabled {
				return "Max Features", "N/A", "limit disabled"
			}
			val := fmt.Sprintf("%d", m.moderationMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optModerationMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "max features for moderation"
		case optModerationPermutations:
			val := fmt.Sprintf("%d", m.moderationPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optModerationPermutations) {
				val = numberDisplay
			}
			return "Moderation Permutations", val, "0=disabled"

		// Mixed effects
		case optMixedEffectsType:
			v := "intercept"
			if m.mixedEffectsType == 1 {
				v = "intercept_slope"
			}
			return "Random Effects", v, "group-level only"
		case optMixedMaxFeatures:
			val := fmt.Sprintf("%d", m.mixedMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optMixedMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "max features to include"

		// Output section
		case optBehaviorGroupOutput:
			label := "▸ Output"
			if m.behaviorGroupOutputExpanded {
				label = "▾ Output"
			}
			return label, "", "Space to toggle"
		case optAlsoSaveCsv:
			return "Also Save CSV", m.boolToOnOff(m.alsoSaveCsv), "save tables as both TSV and CSV"
		case optBehaviorOverwrite:
			return "Overwrite Outputs", m.boolToOnOff(m.behaviorOverwrite), "if off, append timestamp to output folders"

		default:
			return "", "", ""
		}
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	totalLines := len(options)
	startIdx, endIdx, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	// Show scroll indicator for items above
	if showScrollIndicators && startIdx > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", startIdx)) + "\n")
	}

	for i := startIdx; i < endIdx; i++ {
		opt := options[i]
		isFocused := i == m.advancedCursor
		label, value, hint := getOptionDisplay(opt)

		// Check if this is a section header option
		isSectionHeader := opt >= optBehaviorGroupGeneral && opt <= optBehaviorGroupMixedEffects

		var labelStyle, valueStyle lipgloss.Style
		if isSectionHeader {
			// Section headers get special styling (like features pipeline)
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
			valueStyle = lipgloss.NewStyle()
		} else if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		if m.useDefaultAdvanced && i > 0 {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		} else if m.editingNumber && isFocused {
			// Highlight the editing field
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}

		if isSectionHeader {
			// Section headers don't have a colon after the label
			b.WriteString(cursor + labelStyle.Render(label) + "  " + hintStyle.Render(hint) + "\n")
		} else {
			b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value))
			b.WriteString("  " + hintStyle.Render(hint) + "\n")
		}

		// Render expanded column/value list after the relevant option
		if m.shouldRenderExpandedListAfterOption(opt) {
			items := m.getExpandedListItems()
			subIndent := "      " // 6 spaces for sub-items
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

	// Show scroll indicator for items below
	if showScrollIndicators && endIdx < len(options) {
		remaining := len(options) - endIdx
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", remaining)) + "\n")
	}

	return b.String()
}
