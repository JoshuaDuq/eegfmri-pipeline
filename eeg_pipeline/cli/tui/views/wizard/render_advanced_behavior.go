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

		switch opt {
		case optUseDefaults:
			return "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"
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
		case optBehaviorGroupPredictorSens:
			label := "▸ Predictor Sensitivity"
			if m.behaviorGroupPredictorSensExpanded {
				label = "▾ Predictor Sensitivity"
			}
			return label, "", "min trials · features · permutations"
		case optBehaviorGroupRegression:
			label := "▸ Regression"
			if m.behaviorGroupRegressionExpanded {
				label = "▾ Regression"
			}
			return label, "", "outcome · covariates · model families"
		case optBehaviorGroupModels:
			label := "▸ Models"
			if m.behaviorGroupModelsExpanded {
				label = "▾ Models"
			}
			return label, "", "outcomes · covariates · families · inference"
		case optBehaviorGroupStability:
			label := "▸ Stability"
			if m.behaviorGroupStabilityExpanded {
				label = "▾ Stability"
			}
			return label, "", "within-group correlation consistency"
		case optBehaviorGroupConsistency:
			label := "▸ Consistency"
			if m.behaviorGroupConsistencyExpanded {
				label = "▾ Consistency"
			}
			return label, "", "sign-flip detection across subjects"
		case optBehaviorGroupInfluence:
			label := "▸ Influence"
			if m.behaviorGroupInfluenceExpanded {
				label = "▾ Influence"
			}
			return label, "", "Cook's D · leverage · outlier detection"
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
		case optBehaviorGroupMediation:
			label := "▸ Mediation"
			if m.behaviorGroupMediationExpanded {
				label = "▾ Mediation"
			}
			return label, "", "bootstrap · permutations · mediator limit"
		case optBehaviorGroupModeration:
			label := "▸ Moderation"
			if m.behaviorGroupModerationExpanded {
				label = "▾ Moderation"
			}
			return label, "", "feature limit · min samples · permutations"
		case optBehaviorGroupMixedEffects:
			label := "▸ Mixed Effects"
			if m.behaviorGroupMixedEffectsExpanded {
				label = "▾ Mixed Effects"
			}
			return label, "", "random effects type · predictor · features"
		case optBehaviorGroupStats:
			label := "▸ Inference & Shared Settings"
			if m.behaviorGroupStatsExpanded {
				label = "▾ Inference & Shared Settings"
			}
			return label, "", "corr method · FDR · perms · covariates"
		case optBehaviorGroupAnalyses:
			label := "▸ Analyses"
			if m.behaviorGroupAnalysesExpanded {
				label = "▾ Analyses"
			}
			return label, "", "stability · influence · mediation · more"
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
		case optBehaviorSubFeatureQC:
			return "  ── Feature QC", "", ""
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
			return "Feature×Temp", m.boolToOnOff(m.regressionIncludeInteraction), "moderation term"
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

		// Models
		case optModelsIncludePredictor:
			return "Include Predictor", m.boolToOnOff(m.modelsIncludePredictor), "add predictor covariate"
		case optModelsTempControl:
			v := "linear"
			if m.predictorType == 0 {
				switch m.modelsTempControl {
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
		case optModelsTempSplineMinN:
			val := fmt.Sprintf("%d", m.modelsTempSplineMinN)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineMinN) {
				val = numberDisplay
			}
			return "Spline Min Samples", val, "minimum rows for spline basis"
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
		case optModelsMinSamples:
			val := fmt.Sprintf("%d", m.modelsMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optModelsMinSamples) {
				val = numberDisplay
			}
			return "Min Samples", val, "minimum rows per model fit"
		case optModelsMaxFeatures:
			val := fmt.Sprintf("%d", m.modelsMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optModelsMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"
		case optModelsOutcomeRating:
			return "Outcome: raw", m.boolToOnOff(m.modelsOutcomeValue), "include raw outcome"
		case optModelsOutcomePredictorResidual:
			return "Outcome: residual", m.boolToOnOff(m.modelsOutcomePredictorResidual), "include residualized outcome"
		case optModelsOutcomePredictor:
			return "Outcome: predictor", m.boolToOnOff(m.modelsOutcomePredictor), "include predictor"
		case optModelsOutcomeBinaryOutcome:
			return "Outcome: binary", m.boolToOnOff(m.modelsOutcomeBinaryOutcome), "include binary outcome"
		case optModelsFamilyOLS:
			return "Family: OLS-HC3", m.boolToOnOff(m.modelsFamilyOLS), "ols_hc3"
		case optModelsFamilyRobust:
			return "Family: Robust", m.boolToOnOff(m.modelsFamilyRobust), "robust_rlm"
		case optModelsFamilyQuantile:
			return "Family: Quantile", m.boolToOnOff(m.modelsFamilyQuantile), "quantile_50"
		case optModelsFamilyLogit:
			return "Family: Logistic", m.boolToOnOff(m.modelsFamilyLogit), "logit"
		case optModelsBinaryOutcome:
			v := "binary"
			if m.modelsBinaryOutcome == 1 {
				v = "outcome_median"
			}
			return "Binary Outcome", v, "for logit models"
		case optModelsPrimaryUnit:
			v := "trial"
			if m.modelsPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
		case optModelsForceTrialIIDAsymptotic:
			if m.modelsPrimaryUnit != 0 {
				return "Force Trial IID", "N/A", "only used for trial-level models"
			}
			return "Force Trial IID", m.boolToOnOff(m.modelsForceTrialIIDAsymptotic), "explicit asymptotic override"

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
				v = "outcome"
			case 2:
				v = "residual"
			}
			return "Outcome", v, "auto prefers residual outcome"
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
			return "Partial Predictor", m.boolToOnOff(m.stabilityPartialTemp), "control predictor"
		case optStabilityMinGroupTrials:
			val := "unset"
			if m.stabilityMinGroupN > 0 {
				val = fmt.Sprintf("%d", m.stabilityMinGroupN)
			}
			if m.editingNumber && m.isCurrentlyEditing(optStabilityMinGroupTrials) {
				val = numberDisplay
			}
			return "Min Group Trials", val, "0=unset"
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
			return "Alpha", val, "min |r| to flag as stable"

		// Consistency
		case optConsistencyEnabled:
			return "Consistency Summary", m.boolToOnOff(m.consistencyEnabled), "flag sign flips"

		// Influence
		case optInfluenceOutcomeRating:
			return "Outcome: raw", m.boolToOnOff(m.influenceOutcomeValue), "include raw outcome"
		case optInfluenceOutcomePredictorResidual:
			return "Outcome: residual", m.boolToOnOff(m.influenceOutcomePredictorResidual), "include residual"
		case optInfluenceOutcomePredictor:
			return "Outcome: predictor", m.boolToOnOff(m.influenceOutcomePredictor), "include predictor"
		case optInfluenceMaxFeatures:
			val := fmt.Sprintf("%d", m.influenceMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "top effects to inspect"
		case optInfluenceIncludePredictor:
			return "Include Predictor", m.boolToOnOff(m.influenceIncludePredictor), "add covariate"
		case optInfluenceTempControl:
			v := "linear"
			if m.predictorType == 0 {
				switch m.influenceTempControl {
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
		case optInfluenceTempSplineMinN:
			val := fmt.Sprintf("%d", m.influenceTempSplineMinN)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineMinN) {
				val = numberDisplay
			}
			return "Spline Min Samples", val, "minimum rows for spline basis"
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
		case optConditionWindowPrimaryUnit:
			v := "trial"
			if m.conditionWindowPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Window Unit", v, "trial | run_mean"
		case optConditionWindowMinSamples:
			val := fmt.Sprintf("%d", m.conditionWindowMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optConditionWindowMinSamples) {
				val = numberDisplay
			}
			return "Window Min Samples", val, "minimum rows for window comparison"
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
		case optPredictorSensitivityMinTrials:
			val := "unset"
			if m.predictorSensitivityMinTrials > 0 {
				val = fmt.Sprintf("%d", m.predictorSensitivityMinTrials)
			}
			if m.editingNumber && m.isCurrentlyEditing(optPredictorSensitivityMinTrials) {
				val = numberDisplay
			}
			return "Min Trials", val, "0=unset"
		case optPredictorSensitivityPrimaryUnit:
			v := "trial"
			if m.predictorSensitivityPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
		case optPredictorSensitivityPermutations:
			val := fmt.Sprintf("%d", m.predictorSensitivityPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optPredictorSensitivityPermutations) {
				val = numberDisplay
			}
			return "Permutations", val, "0=use global"
		case optPredictorSensitivityPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.predictorSensitivityPermutationPrimary), "perm_if_available | asymptotic"
		case optPredictorSensitivityFeatures:
			val := m.predictorSensitivityFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldPredictorSensitivityFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"

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
				val = "(default: partial_cov_temp)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTypes {
				val = textDisplay
			}
			return "Correlation Types", val, "comma-separated: raw,partial_cov,partial_temp,partial_cov_temp,run_mean"
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
		case optMediationPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.mediationPermutationPrimary), "perm_if_available | asymptotic"
		case optMediationMinEffect:
			val := fmt.Sprintf("%.3f", m.mediationMinEffect)
			if m.editingNumber && m.isCurrentlyEditing(optMediationMinEffect) {
				val = numberDisplay
			}
			return "Min Effect Size", val, "minimum indirect effect"
		case optMediationFeatures:
			val := m.mediationFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldMediationFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"
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
		case optModerationMinSamples:
			val := fmt.Sprintf("%d", m.moderationMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optModerationMinSamples) {
				val = numberDisplay
			}
			return "Min Samples", val, "minimum rows for moderation model"
		case optModerationPermutations:
			val := fmt.Sprintf("%d", m.moderationPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optModerationPermutations) {
				val = numberDisplay
			}
			return "Moderation Permutations", val, "0=disabled"
		case optModerationPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.moderationPermutationPrimary), "perm_if_available | asymptotic"
		case optModerationFeatures:
			val := m.moderationFeaturesSpec
			if strings.TrimSpace(val) == "" {
				val = "(all)"
			}
			if m.editingText && m.editingTextField == textFieldModerationFeatures {
				val = textDisplay
			}
			return "Feature Filters", val, "comma-separated feature families"

		// Mixed effects
		case optMixedEffectsType:
			v := "intercept"
			if m.mixedEffectsType == 1 {
				v = "intercept_slope"
			}
			return "Random Effects", v, "intercept=random intercept · intercept_slope=full random"
		case optMixedIncludePredictor:
			return "Include Predictor", m.boolToOnOff(m.mixedIncludePredictor), "add predictor covariate"
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
			return label, "", "CSV export · overwrite policy"
		case optAlsoSaveCsv:
			return "Also Save CSV", m.boolToOnOff(m.alsoSaveCsv), "save tables as both TSV and CSV"
		case optBehaviorOverwrite:
			return "Overwrite Outputs", m.boolToOnOff(m.behaviorOverwrite), "if off, append timestamp to output folders"

		// Behavior Statistics
		case optBehaviorStatsTempControl:
			controls := []string{"none", "linear", "spline"}
			return "Stats Predictor Control", controls[m.behaviorStatsTempControl%len(controls)], "global predictor covariate for all analyses"
		case optBehaviorStatsAllowIIDTrials:
			return "Allow IID Trials", m.boolToOnOff(m.behaviorStatsAllowIIDTrials), "use asymptotic p-values when N_trials is large"
		case optBehaviorStatsHierarchicalFDR:
			return "Hierarchical FDR", m.boolToOnOff(m.behaviorStatsHierarchicalFDR), "family-wise correction across feature families"
		case optBehaviorStatsComputeReliability:
			return "Compute Reliability", m.boolToOnOff(m.behaviorStatsComputeReliability), "split-half ICC for each feature"
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
		isSubHeader := opt >= optBehaviorSubCorrelationSettings && opt <= optBehaviorSubMultilevel
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
