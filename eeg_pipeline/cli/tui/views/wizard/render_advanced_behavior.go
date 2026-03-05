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

	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("behavior analysis")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter value, Enter to confirm, Esc to cancel") + "\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, Enter to confirm, Esc to cancel") + "\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space: select  Esc: close list") + "\n")
	} else {
		b.WriteString(infoStyle.Render("Space: toggle/expand  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidthWide
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	options := m.getBehaviorOptions()
	availableColumns := m.GetAvailableColumns()

	getOptionDisplay := func(opt optionType) (string, string, string) {
		numberDisplay := m.numberBuffer + "█"
		textDisplay := m.textBuffer + "█"
		jsonDisplay := func(value string, field textField) string {
			if m.editingText && m.editingTextField == field {
				return textDisplay
			}
			trimmed := strings.TrimSpace(value)
			if trimmed == "" {
				return "(default)"
			}
			if len(trimmed) > 72 {
				return trimmed[:69] + "..."
			}
			return trimmed
		}

		switch opt {
		case optUseDefaults:
			return "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"
		case optConfigSetOverrides:
			val := strings.TrimSpace(m.configSetOverrides)
			if m.editingText && m.editingTextField == textFieldConfigSetOverrides {
				val = textDisplay
			} else if val == "" {
				val = "(none)"
			}
			return "Config Overrides", val, "Advanced/uncommon keys: key=value;key2=value2 (emits repeated --set)"
		// Behavior section headers
		case optBehaviorGroupGeneral:
			label := "▸ Execution"
			if m.behaviorGroupGeneralExpanded {
				label = "▾ Execution"
			}
			return label, "", "RNG · jobs · thresholds"
		case optBehaviorGroupTrialTable:
			label := "▸ Trial Table"
			if m.behaviorGroupTrialTableExpanded {
				label = "▾ Trial Table"
			}
			return label, "", "format · lag features · validation"
		case optBehaviorGroupPredictorResidual:
			label := "▸ Residual Modeling"
			if m.behaviorGroupPredictorResidualExpanded {
				label = "▾ Residual Modeling"
			}
			return label, "", "fitting method · diagnostics · crossfit"
		case optBehaviorGroupCorrelations:
			label := "▸ Correlations"
			if m.behaviorGroupCorrelationsExpanded {
				label = "▾ Correlations"
			}
			return label, "", "target · features · types · multilevel"
		case optBehaviorGroupRegression:
			label := "▸ Regression"
			if m.behaviorGroupRegressionExpanded {
				label = "▾ Regression"
			}
			return label, "", "outcome · covariates · model families"
		case optBehaviorGroupReport:
			label := "▸ Report"
			if m.behaviorGroupReportExpanded {
				label = "▾ Report"
			}
			return label, "", "top-N features to summarise"
		case optBehaviorGroupCondition:
			label := "▸ Condition"
			if m.behaviorGroupConditionExpanded {
				label = "▾ Condition"
			}
			return label, "", "compare column · windows · features"
		case optBehaviorGroupTemporal:
			label := "▸ Temporal"
			if m.behaviorGroupTemporalExpanded {
				label = "▾ Temporal"
			}
			return label, "", "time window · resolution · ITPC · ERDS"
		case optBehaviorGroupCluster:
			label := "▸ Cluster"
			if m.behaviorGroupClusterExpanded {
				label = "▾ Cluster"
			}
			return label, "", "threshold · min size · tail · features"
		case optBehaviorGroupStats:
			label := "▸ Inference & Shared Settings"
			if m.behaviorGroupStatsExpanded {
				label = "▾ Inference & Shared Settings"
			}
			return label, "", "corr method · FDR · perms · covariates"
		case optBehaviorGroupAdvanced:
			label := "▸ Advanced"
			if m.behaviorGroupAdvancedExpanded {
				label = "▾ Advanced"
			}
			return label, "", "global validation · system IO"
		// Behavior sub-section headers (non-collapsible visual separators)
		case optBehaviorSubCorrelationSettings:
			return "  ── Correlation Settings", "", ""
		case optBehaviorSubCovariates:
			return "  ── Covariates", "", ""
		case optBehaviorSubRunAdjustment:
			return "  ── Run Adjustment", "", ""
		case optBehaviorSubCorrelationsExtra:
			return "  ── Correlations Extra", "", ""
		case optBehaviorSubOutcome:
			return "  ── Outcome", "", ""
		case optBehaviorSubOutcomes:
			return "  ── Outcomes", "", ""
		case optBehaviorSubModelFamilies:
			return "  ── Model Families", "", ""
		case optBehaviorSubInference:
			return "  ── Inference", "", ""
		case optBehaviorSubDiagnostics:
			return "  ── Diagnostics", "", ""
		case optBehaviorSubFitting:
			return "  ── Fitting", "", ""
		case optBehaviorSubCrossfit:
			return "  ── Crossfit", "", ""
		case optBehaviorSubTimeWindow:
			return "  ── Time Window", "", ""
		case optBehaviorSubFeatures:
			return "  ── Features & Output", "", ""
		case optBehaviorSubITPC:
			return "  ── ITPC", "", ""
		case optBehaviorSubERDS:
			return "  ── ERDS", "", ""
		case optBehaviorSubMultilevel:
			return "  ── Group-Level", "", ""
		case optBehaviorSubFeatureRegistry:
			return "  ── Feature Registry", "", ""
		case optPredictorType:
			types := []string{"continuous", "binary", "categorical"}
			v := "continuous"
			if m.predictorType >= 0 && m.predictorType < len(types) {
				v = types[m.predictorType]
			}
			return "Predictor Type", v, "continuous | binary | categorical"
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
			return "Permutations", val, "0=disabled · shared across all analyses"
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
		case optBehaviorOutcomeColumn:
			val := m.behaviorOutcomeColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: auto)"
			}
			if m.editingText && m.editingTextField == textFieldBehaviorOutcomeColumn {
				val = textDisplay
			}
			hint := "Space to select · blank=auto"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Outcome Column", val, hint
		case optBehaviorPredictorColumn:
			val := m.behaviorPredictorColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: auto)"
			}
			if m.editingText && m.editingTextField == textFieldBehaviorPredictorColumn {
				val = textDisplay
			}
			hint := "Space to select · blank=auto"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Predictor Column", val, hint
		case optControlTemp:
			return "Control Predictor", m.boolToOnOff(m.controlPredictor), "partial-correlation covariate"
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
		case optBehaviorMinSamples:
			val := "unset"
			if m.behaviorMinSamples > 0 {
				val = fmt.Sprintf("%d", m.behaviorMinSamples)
			}
			if m.editingNumber && m.isCurrentlyEditing(optBehaviorMinSamples) {
				val = numberDisplay
			}
			return "Default Min Samples", val, "0=unset; behavior min_samples.default"
		case optFDRAlpha:
			val := fmt.Sprintf("%.3f", m.fdrAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optFDRAlpha) {
				val = numberDisplay
			}
			return "FDR Alpha", val, "Benjamini-Hochberg q threshold"
		case optComputeChangeScores:
			return "Change Scores", m.boolToOnOff(m.behaviorComputeChangeScores), "Δ outcome / Δ predictor"
		case optComputeLosoStability:
			return "LOSO Stability", m.boolToOnOff(m.behaviorComputeLosoStability), "leave-one-out stability"
		case optComputeBayesFactors:
			return "Bayes Factors", m.boolToOnOff(m.behaviorComputeBayesFactors), "optional BF reporting"
		case optBehaviorValidateOnly:
			return "Validate Only", m.boolToOnOff(m.behaviorValidateOnly), "load and validate without statistics"

		// Trial table
		case optTrialTableFormat:
			v := "parquet"
			if m.trialTableFormat == 1 {
				v = "tsv"
			}
			return "Trial Table Format", v, "parquet recommended"
		case optTrialTableAddLagFeatures:
			return "Lag/Delta Columns", m.boolToOnOff(m.trialTableAddLagFeatures), "prev_* and delta_*"
		case optBehaviorTrialTableDisallowPositionalAlignment:
			return "Disallow Positional Align", m.boolToOnOff(m.trialTableDisallowPositionalAlignment), "fail if fallback positional alignment is needed"
		case optTrialOrderMaxMissingFraction:
			if !m.controlTrialOrder {
				return "Trial Order Max Missing", "N/A", "enable Control Trial Order"
			}
			val := fmt.Sprintf("%.2f", m.trialOrderMaxMissingFraction)
			if m.editingNumber && m.isCurrentlyEditing(optTrialOrderMaxMissingFraction) {
				val = numberDisplay
			}
			return "Trial Order Max Missing", val, "disable control if missing > threshold"
		case optFeatureSummariesEnabled:
			return "Feature Summaries", m.boolToOnOff(m.featureSummariesEnabled), "per-feature descriptive stats in output"

		// Predictor residual
		case optPredictorResidualEnabled:
			return "Residual", m.boolToOnOff(m.predictorResidualEnabled), "outcome - f(predictor)"
		case optPredictorResidualMethod:
			v := "spline"
			if m.predictorResidualMethod == 1 {
				v = "poly"
			}
			return "Residual Method", v, "spline preferred"
		case optPredictorResidualPolyDegree:
			val := fmt.Sprintf("%d", m.predictorResidualPolyDegree)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualPolyDegree) {
				val = numberDisplay
			}
			return "Poly Degree", val, "poly fallback degree"
		case optPredictorResidualSplineDfCandidates:
			val := m.predictorResidualSplineDfCandidates
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPredictorResidualSplineDfCandidates {
				val = textDisplay
			}
			return "Spline DF Candidates", val, "comma-separated (e.g., 3,4,5)"
		case optPredictorResidualMinSamples:
			val := fmt.Sprintf("%d", m.predictorResidualMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualMinSamples) {
				val = numberDisplay
			}
			return "Residual Min Samples", val, "minimum rows for fit"
		case optPredictorResidualModelCompare:
			return "Predictor Model Compare", m.boolToOnOff(m.predictorResidualModelCompareEnabled), "non-gating diagnostics"
		case optPredictorResidualModelComparePolyDegrees:
			val := m.predictorResidualModelComparePolyDegrees
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPredictorResidualModelComparePolyDegrees {
				val = textDisplay
			}
			return "Model Compare Poly Deg", val, "comma-separated (e.g., 2,3)"
		case optPredictorResidualModelCompareMinSamples:
			val := fmt.Sprintf("%d", m.predictorResidualModelCompareMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualModelCompareMinSamples) {
				val = numberDisplay
			}
			return "Model Compare Min N", val, "minimum rows for comparison"
		case optPredictorResidualBreakpoint:
			return "Breakpoint Test", m.boolToOnOff(m.predictorResidualBreakpointEnabled), "single-hinge model"
		case optPredictorResidualBreakpointCandidates:
			val := fmt.Sprintf("%d", m.predictorResidualBreakpointCandidates)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualBreakpointCandidates) {
				val = numberDisplay
			}
			return "Breakpoint Candidates", val, "grid size"
		case optPredictorResidualBreakpointMinSamples:
			val := fmt.Sprintf("%d", m.predictorResidualBreakpointMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualBreakpointMinSamples) {
				val = numberDisplay
			}
			return "Breakpoint Min N", val, "minimum rows for hinge test"
		case optPredictorResidualBreakpointQlow:
			val := fmt.Sprintf("%.2f", m.predictorResidualBreakpointQlow)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualBreakpointQlow) {
				val = numberDisplay
			}
			return "Breakpoint Q Low", val, "quantile bound"
		case optPredictorResidualBreakpointQhigh:
			val := fmt.Sprintf("%.2f", m.predictorResidualBreakpointQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualBreakpointQhigh) {
				val = numberDisplay
			}
			return "Breakpoint Q High", val, "quantile bound"
		case optPredictorResidualCrossfitEnabled:
			if !m.predictorResidualEnabled {
				return "Residual Crossfit", "N/A", "enable Residual"
			}
			return "Residual Crossfit", m.boolToOnOff(m.predictorResidualCrossfitEnabled), "out-of-run predictor→outcome"
		case optPredictorResidualCrossfitGroupColumn:
			if !m.predictorResidualEnabled || !m.predictorResidualCrossfitEnabled {
				return "Crossfit Group Col", "N/A", "enable Residual Crossfit"
			}
			val := m.predictorResidualCrossfitGroupColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: run column)"
			}
			if m.editingText && m.editingTextField == textFieldPredictorResidualCrossfitGroupColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			} else {
				hint = "Space to edit (blank=run column)"
			}
			return "Crossfit Group Col", val, hint
		case optPredictorResidualCrossfitNSplits:
			if !m.predictorResidualEnabled || !m.predictorResidualCrossfitEnabled {
				return "Crossfit Splits", "N/A", "enable Residual Crossfit"
			}
			val := fmt.Sprintf("%d", m.predictorResidualCrossfitNSplits)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualCrossfitNSplits) {
				val = numberDisplay
			}
			return "Crossfit Splits", val, "n_splits (>=2)"
		case optPredictorResidualCrossfitMethod:
			if !m.predictorResidualEnabled || !m.predictorResidualCrossfitEnabled {
				return "Crossfit Method", "N/A", "enable Residual Crossfit"
			}
			v := "spline"
			if m.predictorResidualCrossfitMethod == 1 {
				v = "poly"
			}
			return "Crossfit Method", v, "spline | poly"
		case optPredictorResidualCrossfitSplineKnots:
			if !m.predictorResidualEnabled || !m.predictorResidualCrossfitEnabled {
				return "Crossfit Knots", "N/A", "enable Residual Crossfit"
			}
			if m.predictorResidualCrossfitMethod == 1 {
				return "Crossfit Knots", "N/A", "poly method"
			}
			val := fmt.Sprintf("%d", m.predictorResidualCrossfitSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorResidualCrossfitSplineKnots) {
				val = numberDisplay
			}
			return "Crossfit Knots", val, "spline knots (>=3)"

		// Regression
		case optRegressionOutcome:
			v := "outcome"
			switch m.regressionOutcome {
			case 1:
				v = "residual"
			case 2:
				v = "predictor"
			}
			return "Outcome", v, "dependent variable"
		case optRegressionIncludePredictor:
			return "Include Predictor", m.boolToOnOff(m.regressionIncludePredictor), "add predictor covariate"
		case optRegressionTempControl:
			v := "linear"
			if m.predictorType == 0 {
				switch m.regressionTempControl {
				case 1:
					v = "outcome_hat"
				case 2:
					v = "spline"
				}
			}
			hint := "linear | outcome_hat | spline"
			if m.predictorType != 0 {
				hint = "linear (continuous predictor required for outcome_hat/spline)"
			}
			return "Predictor Control", v, hint
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
		case optRegressionTempSplineMinN:
			val := fmt.Sprintf("%d", m.regressionTempSplineMinN)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineMinN) {
				val = numberDisplay
			}
			return "Spline Min Samples", val, "minimum rows for spline basis"
		case optRegressionIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.regressionIncludeTrialOrder), "add trial_index covariate"
		case optRegressionIncludePrev:
			return "Prev/Delta Terms", m.boolToOnOff(m.regressionIncludePrev), "use prev_/delta_"
		case optRegressionIncludeRunBlock:
			return "Run/Block Dummies", m.boolToOnOff(m.regressionIncludeRunBlock), "categorical controls"
		case optRegressionIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.regressionIncludeInteraction), "interaction term"
		case optRegressionStandardize:
			return "Standardize", m.boolToOnOff(m.regressionStandardize), "z-score predictors"
		case optRegressionMinSamples:
			val := fmt.Sprintf("%d", m.regressionMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionMinSamples) {
				val = numberDisplay
			}
			return "Min Samples", val, "minimum rows for regression"
		case optRegressionPrimaryUnit:
			v := "trial"
			if m.regressionPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
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
		case optConditionCompareLabels:
			val := m.conditionCompareLabels
			if val == "" {
				val = "(optional)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareLabels {
				val = textDisplay
			}
			return "Compare Labels", val, "optional labels aligned to compare values"
		case optConditionFeatures:
			val := m.conditionFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldConditionFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"
		case optConditionMinTrials:
			val := "unset"
			if m.conditionMinTrials > 0 {
				val = fmt.Sprintf("%d", m.conditionMinTrials)
			}
			if m.editingNumber && m.isCurrentlyEditing(optConditionMinTrials) {
				val = numberDisplay
			}
			return "Min Trials/Condition", val, "0=unset"
		case optConditionPrimaryUnit:
			v := "trial"
			if m.conditionPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
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
		case optTemporalCorrectionMethod:
			v := "fdr"
			if m.temporalCorrectionMethod == 1 {
				v = "cluster"
			}
			return "Correction", v, "fdr | cluster"
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
				if strings.TrimSpace(m.behaviorOutcomeColumn) != "" {
					val = fmt.Sprintf("(default: %s)", strings.TrimSpace(m.behaviorOutcomeColumn))
				} else {
					val = "(default: outcome)"
				}
			}
			if m.editingText && m.editingTextField == textFieldTemporalTargetColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(availableColumns))
			}
			return "Target Column", val, hint
		case optTemporalFeatures:
			val := m.temporalFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"
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
		case optCorrelationsTypes:
			val := m.correlationsTypesSpec
			if strings.TrimSpace(val) == "" {
				val = "(default: partial_cov_predictor)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTypes {
				val = textDisplay
			}
			return "Correlation Types", val, "comma-separated: raw,partial_cov,partial_predictor,partial_cov_predictor,run_mean"
		case optCorrelationsUseCrossfitPredictorResidual:
			return "Use residual_cv", m.boolToOnOff(m.correlationsUseCrossfitResidual), "requires residual crossfit"
		case optCorrelationsPrimaryUnit:
			v := "trial"
			if m.correlationsPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
		case optCorrelationsMinRuns:
			val := fmt.Sprintf("%d", m.correlationsMinRuns)
			if m.editingNumber && m.isCurrentlyEditing(optCorrelationsMinRuns) {
				val = numberDisplay
			}
			return "Min Runs", val, "minimum runs for run-mean stats"
		case optCorrelationsPreferPredictorResidual:
			return "Prefer Residual Target", m.boolToOnOff(m.correlationsPreferPredictorResidual), "prefer residual target ordering"
		case optCorrelationsPermutations:
			val := fmt.Sprintf("%d", m.correlationsPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optCorrelationsPermutations) {
				val = numberDisplay
			}
			return "Corr Permutations", val, "0=use global"
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
			return "Target Column", val, hint
		case optCorrelationsPowerSegment:
			val := strings.TrimSpace(m.correlationsPowerSegment)
			if val == "" {
				val = "(auto: all segments)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsPowerSegment {
				val = textDisplay
			}
			return "Power Segment", val, "optional segment filter for ROI power correlations"
		case optCorrelationsFeatures:
			val := m.correlationsFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"
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
		case optGroupLevelTarget:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Group Target", "N/A", "enable Group Multilevel Correlations"
			}
			v := strings.TrimSpace(m.groupLevelTarget)
			if v == "" {
				v = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldGroupLevelTarget {
				v = textDisplay
			}
			hint := "Space to select"
			if len(m.availableGroupLevelTargets()) == 0 {
				hint = "type column name (no discovered targets)"
			}
			return "Group Target", v, hint
		case optGroupLevelControlPredictor:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Ctrl Predictor", "N/A", "enable Group Multilevel Correlations"
			}
			return "Ctrl Predictor", m.boolToOnOff(m.groupLevelControlPredictor), "partial Spearman covariate"
		case optGroupLevelControlTrialOrder:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Ctrl Trial Order", "N/A", "enable Group Multilevel Correlations"
			}
			return "Ctrl Trial Order", m.boolToOnOff(m.groupLevelControlTrialOrder), "partial Spearman covariate"
		case optGroupLevelControlRunEffects:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Ctrl Run Effects", "N/A", "enable Group Multilevel Correlations"
			}
			return "Ctrl Run Effects", m.boolToOnOff(m.groupLevelControlRunEffects), "add run dummies if feasible"
		case optGroupLevelMaxRunDummies:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Max Run Dummies", "N/A", "enable Group Multilevel Correlations"
			}
			val := fmt.Sprintf("%d", m.groupLevelMaxRunDummies)
			if m.editingNumber && m.isCurrentlyEditing(optGroupLevelMaxRunDummies) {
				val = numberDisplay
			}
			return "Max Run Dummies", val, "skip run effects if too many levels"
		case optGroupLevelAllowParametricFallback:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Allow Parametric Fallback", "N/A", "enable Group Multilevel Correlations"
			}
			return "Allow Parametric Fallback", m.boolToOnOff(m.groupLevelAllowParametricFallback), "fallback when permutation is unavailable"

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
		case optClusterFeatures:
			val := m.clusterFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldClusterFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"
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

		// Output section
		case optBehaviorGroupOutput:
			label := "▸ Output"
			if m.behaviorGroupOutputExpanded {
				label = "▾ Output"
			}
			return label, "", "CSV export · overwrite policy"
		case optAlsoSaveCsv:
			return "Also Save CSV", m.boolToOnOff(m.alsoSaveCsv), "save tables as both TSV and CSV"
		case optBehaviorOverwrite:
			return "Overwrite Outputs", m.boolToOnOff(m.behaviorOverwrite), "if off, append timestamp to output folders"

		// Behavior Statistics
		case optBehaviorStatsTempControl:
			controls := []string{"spline", "linear", "none"}
			return "Stats Predictor Control", controls[m.behaviorStatsTempControl%len(controls)], "global predictor covariate for all analyses"
		case optBehaviorStatsAllowIIDTrials:
			return "Allow IID Trials", m.boolToOnOff(m.behaviorStatsAllowIIDTrials), "use asymptotic p-values when N_trials is large"
		case optBehaviorStatsHierarchicalFDR:
			return "Hierarchical FDR", m.boolToOnOff(m.behaviorStatsHierarchicalFDR), "family-wise correction across feature families"
		case optBehaviorStatsComputeReliability:
			return "Compute Reliability", m.boolToOnOff(m.behaviorStatsComputeReliability), "split-half ICC for each feature"
		case optStatisticsAlpha:
			val := fmt.Sprintf("%.4f", m.statisticsAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optStatisticsAlpha) {
				val = numberDisplay
			}
			return "Global Stats Alpha", val, "fallback alpha for shared stats helpers"
		case optBehaviorPermScheme:
			schemes := []string{"shuffle", "circular_shift"}
			return "Perm Scheme", schemes[m.behaviorPermScheme%len(schemes)], "shuffle=iid trials · circular_shift=time-series"
		case optBehaviorPermGroupColumnPreference:
			val := m.behaviorPermGroupColumnPreference
			if strings.TrimSpace(val) == "" {
				val = "(auto)"
			}
			if m.editingText && m.editingTextField == textFieldBehaviorPermGroupColumnPreference {
				val = textDisplay
			}
			return "Perm Group Column", val, "preferred grouping column"
		case optBehaviorExcludeNonTrialwiseFeatures:
			return "Exclude Non-Trialwise", m.boolToOnOff(m.behaviorExcludeNonTrialwiseFeatures), "skip features without trial-level resolution"
		case optBehaviorFeatureRegistryFilesJSON:
			return "Registry Files JSON", jsonDisplay(m.behaviorFeatureRegistryFilesJSON, textFieldBehaviorFeatureRegistryFilesJSON), "JSON object for behavior_analysis.feature_registry.files"
		case optBehaviorFeatureRegistrySourceToTypeJSON:
			return "Registry Source->Type JSON", jsonDisplay(m.behaviorFeatureRegistrySourceJSON, textFieldBehaviorFeatureRegistrySourceToTypeJSON), "JSON object for source_to_feature_type"
		case optBehaviorFeatureRegistryTypeHierarchyJSON:
			return "Registry Type Hierarchy JSON", jsonDisplay(m.behaviorFeatureRegistryHierarchyJSON, textFieldBehaviorFeatureRegistryTypeHierarchyJSON), "JSON object for feature_type_hierarchy"
		case optBehaviorFeatureRegistryPatternsJSON:
			return "Registry Patterns JSON", jsonDisplay(m.behaviorFeatureRegistryPatternsJSON, textFieldBehaviorFeatureRegistryPatternsJSON), "JSON object for feature_patterns"
		case optBehaviorFeatureRegistryClassifiersJSON:
			return "Registry Classifiers JSON", jsonDisplay(m.behaviorFeatureRegistryClassifiersJSON, textFieldBehaviorFeatureRegistryClassifiersJSON), "JSON array for feature_classifiers"

		// Global Statistics & Validation
		case optGlobalNBootstrap:
			val := fmt.Sprintf("%d", m.globalNBootstrap)
			if m.editingNumber && m.isCurrentlyEditing(optGlobalNBootstrap) {
				val = m.numberBuffer + "█"
			}
			return "Global N Bootstrap", val, "CI bootstrap iterations (0=disabled)"
		case optClusterCorrectionEnabled:
			return "Cluster Correction", m.boolToOnOff(m.clusterCorrectionEnabled), "spatial cluster correction across features"
		case optClusterCorrectionAlpha:
			val := fmt.Sprintf("%.4f", m.clusterCorrectionAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optClusterCorrectionAlpha) {
				val = m.numberBuffer + "█"
			}
			return "Cluster Corr Alpha", val, "cluster-forming p-value threshold"
		case optClusterCorrectionMinClusterSize:
			val := fmt.Sprintf("%d", m.clusterCorrectionMinClusterSize)
			if m.editingNumber && m.isCurrentlyEditing(optClusterCorrectionMinClusterSize) {
				val = m.numberBuffer + "█"
			}
			return "Cluster Min Size", val, "minimum cluster size"
		case optClusterCorrectionTail:
			tails := []string{"two-tailed", "upper", "lower"}
			return "Cluster Corr Tail", tails[m.clusterCorrectionTailGlobal%len(tails)], "tail direction"
		case optValidationMinEpochs:
			val := fmt.Sprintf("%d", m.validationMinEpochs)
			if m.editingNumber && m.isCurrentlyEditing(optValidationMinEpochs) {
				val = m.numberBuffer + "█"
			}
			return "Validation Min Epochs", val, "reject subjects with fewer epochs"
		case optValidationMinChannels:
			val := fmt.Sprintf("%d", m.validationMinChannels)
			if m.editingNumber && m.isCurrentlyEditing(optValidationMinChannels) {
				val = m.numberBuffer + "█"
			}
			return "Validation Min Channels", val, "reject subjects with fewer channels"
		case optValidationMaxAmplitudeUv:
			val := fmt.Sprintf("%.1f", m.validationMaxAmplitudeUv)
			if m.editingNumber && m.isCurrentlyEditing(optValidationMaxAmplitudeUv) {
				val = m.numberBuffer + "█"
			}
			return "Max Amplitude (µV)", val, "amplitude rejection threshold"

		// System / IO
		case optIOPredictorRange:
			val := m.ioPredictorRange
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldIOPredictorRange {
				val = textDisplay
			}
			return "Predictor Range", val, "e.g. 32.0,50.0"
		case optIOMaxMissingChannelsFraction:
			val := fmt.Sprintf("%.2f", m.ioMaxMissingChannelsFraction)
			if m.editingNumber && m.isCurrentlyEditing(optIOMaxMissingChannelsFraction) {
				val = m.numberBuffer + "█"
			}
			return "Max Missing Channels Frac", val, "skip trial if more channels are NaN"

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

		// Check if this is a collapsible group header or a non-collapsible sub-header
		isGroupHeader := opt >= optBehaviorGroupGeneral && opt <= optBehaviorGroupAdvanced
		isSubHeader := opt >= optBehaviorSubCorrelationSettings && opt <= optBehaviorSubFeatureRegistry
		isSectionHeader := isGroupHeader || isSubHeader

		var labelStyle, valueStyle lipgloss.Style
		if isGroupHeader {
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
			valueStyle = lipgloss.NewStyle()
		} else if isSubHeader {
			labelStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
			valueStyle = lipgloss.NewStyle()
		} else if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
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
			line := cursor + labelStyle.Render(label) + "  " + hintStyle.Render(hint)
			b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
		} else {
			styledLabel := labelStyle.Render(label + ":")
			styledValue := valueStyle.Render(value)
			styledHint := hintStyle.Render(hint)
			b.WriteString(styles.RenderConfigLine(cursor, styledLabel, styledValue, styledHint, labelWidth, m.contentWidth) + "\n")
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
