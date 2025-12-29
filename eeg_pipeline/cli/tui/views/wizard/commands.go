package wizard

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
)

///////////////////////////////////////////////////////////////////
// Getters
///////////////////////////////////////////////////////////////////

func (m Model) SelectedCategories() []string {
	var result []string
	for i, sel := range m.selected {
		if sel && i < len(m.categories) {
			result = append(result, m.categories[i])
		}
	}
	sort.Strings(result)
	return result
}

// isCategorySelected checks if a specific category is currently selected
func (m Model) isCategorySelected(category string) bool {
	for i, sel := range m.selected {
		if sel && i < len(m.categories) && m.categories[i] == category {
			return true
		}
	}
	return false
}

func (m Model) SelectedComputations() []string {
	var result []string
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) {
			result = append(result, m.computations[i].Key)
		}
	}
	sort.Strings(result)
	return result
}

// isComputationSelected checks if a specific computation is currently selected
func (m Model) isComputationSelected(computation string) bool {
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) && m.computations[i].Key == computation {
			return true
		}
	}
	return false
}

func (m Model) SelectedSubjectIDs() []string {
	var result []string
	for id, sel := range m.subjectSelected {
		if sel {
			result = append(result, id)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) SelectedBands() []string {
	var result []string
	for i, sel := range m.bandSelected {
		if sel && i < len(m.bands) {
			result = append(result, m.bands[i].Key)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) SelectedSpatialModes() []string {
	var result []string
	for i, sel := range m.spatialSelected {
		if sel && i < len(spatialModes) {
			result = append(result, spatialModes[i].Key)
		}
	}
	return result
}

func (m Model) SelectedFeatureFiles() []string {
	var result []string
	for _, file := range m.featureFiles {
		if m.featureFileSelected[file.Key] {
			result = append(result, file.Key)
		}
	}
	return result
}

func (m Model) SelectedPlotIDs() []string {
	var result []string
	for i, plot := range m.plotItems {
		if m.plotSelected[i] && m.IsPlotCategorySelected(plot.Group) {
			result = append(result, plot.ID)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) SelectedPlotFormats() []string {
	var result []string
	for _, format := range m.plotFormats {
		if m.plotFormatSelected[format] {
			result = append(result, format)
		}
	}
	sort.Strings(result)
	return result
}

// selectedConnectivityMeasures returns the list of selected connectivity measures
func (m Model) selectedConnectivityMeasures() []string {
	var result []string
	for i, measure := range connectivityMeasures {
		if m.connectivityMeasures[i] {
			result = append(result, measure.Key)
		}
	}
	return result
}

///////////////////////////////////////////////////////////////////
// Subject Filtering
///////////////////////////////////////////////////////////////////

func (m Model) getFilteredSubjects() []types.SubjectStatus {
	var filtered []types.SubjectStatus
	filterLower := strings.ToLower(m.subjectFilter)

	for _, s := range m.subjects {
		if m.subjectFilter != "" && !strings.Contains(strings.ToLower(s.ID), filterLower) {
			continue
		}

		if m.showOnlyValid {
			valid := false
			if m.Pipeline == types.PipelinePlotting {
				valid, _ = m.validatePlottingSubject(s)
			} else {
				valid, _ = m.Pipeline.ValidateSubject(s)
			}
			if !valid {
				continue
			}
		}

		filtered = append(filtered, s)
	}

	if len(filtered) == 0 && m.subjectFilter == "" && !m.showOnlyValid {
		return m.subjects
	}

	return filtered
}

///////////////////////////////////////////////////////////////////
// Command Builder
///////////////////////////////////////////////////////////////////

func (m Model) BuildCommand() string {
	parts := []string{"eeg-pipeline", strings.ToLower(m.Pipeline.String())}

	// Positional mode (only for certain pipelines)
	needsMode := false
	switch m.Pipeline {
	case types.PipelinePreprocessing, types.PipelineFeatures, types.PipelineBehavior, types.PipelinePlotting:
		needsMode = true
	}

	if needsMode && len(m.modeOptions) > m.modeIndex {
		parts = append(parts, m.modeOptions[m.modeIndex])
	}

	if m.Pipeline == types.PipelinePlotting {
		selectedPlots := m.SelectedPlotIDs()
		if len(selectedPlots) > 0 && len(selectedPlots) < len(m.plotItems) {
			parts = append(parts, "--plots")
			parts = append(parts, selectedPlots...)
		}

		formats := m.SelectedPlotFormats()
		if len(formats) > 0 {
			parts = append(parts, "--formats")
			parts = append(parts, formats...)
		}

		if m.plotDpiIndex >= 0 && m.plotDpiIndex < len(m.plotDpiOptions) {
			parts = append(parts, "--dpi", fmt.Sprintf("%d", m.plotDpiOptions[m.plotDpiIndex]))
		}

		if m.plotSavefigDpiIndex >= 0 && m.plotSavefigDpiIndex < len(m.plotDpiOptions) {
			parts = append(parts, "--savefig-dpi", fmt.Sprintf("%d", m.plotDpiOptions[m.plotSavefigDpiIndex]))
		}

		if !m.plotSharedColorbar {
			parts = append(parts, "--no-shared-colorbar")
		}
	}

	if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		// Computations (analyses to run)
		comps := m.SelectedComputations()
		if len(comps) > 0 {
			parts = append(parts, "--computations")
			parts = append(parts, comps...)
		}

		// Feature files (consolidated feature selection)
		featureFiles := m.SelectedFeatureFiles()
		if len(featureFiles) > 0 && len(featureFiles) < len(m.featureFiles) {
			parts = append(parts, "--feature-files")
			parts = append(parts, featureFiles...)
		}
	} else if m.Pipeline == types.PipelineCombineFeatures {
		// Special handling for combine-features utility
		parts = []string{"eeg-pipeline", "utilities", "combine-features"}
		featureFiles := m.SelectedFeatureFiles()
		if len(featureFiles) > 0 && len(featureFiles) < len(m.featureFiles) {
			parts = append(parts, "--categories")
			parts = append(parts, featureFiles...)
		}
	} else if m.Pipeline == types.PipelineMergePsychoPyData || m.Pipeline == types.PipelineRawToBIDS {
		mode := "merge-behavior"
		if m.Pipeline == types.PipelineRawToBIDS {
			mode = "raw-to-bids"
		}
		parts = []string{"eeg-pipeline", "utilities", mode}
	} else if m.Pipeline != types.PipelinePlotting {
		// Features pipeline category selection
		cats := m.SelectedCategories()
		if len(cats) > 0 && len(cats) < len(m.categories) {
			parts = append(parts, "--categories")
			parts = append(parts, cats...)
		}
	} else {
		// Plotting pipeline handled above
	}

	if (m.Pipeline == types.PipelineFeatures || m.Pipeline == types.PipelineBehavior) && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}
	}

	// Path overrides for analysis pipelines
	needsPaths := false
	switch m.Pipeline {
	case types.PipelinePreprocessing, types.PipelineFeatures, types.PipelineBehavior,
		types.PipelineDecoding, types.PipelinePlotting, types.PipelineCombineFeatures:
		needsPaths = true
	}

	if needsPaths {
		if m.bidsRoot != "" {
			parts = append(parts, "--bids-root", expandUserPath(m.bidsRoot))
		}
		if m.derivRoot != "" {
			parts = append(parts, "--deriv-root", expandUserPath(m.derivRoot))
		}
	}
	if m.Pipeline == types.PipelineRawToBIDS || m.Pipeline == types.PipelineMergePsychoPyData {
		if m.sourceRoot != "" {
			parts = append(parts, "--source-root", expandUserPath(m.sourceRoot))
		}
		if m.bidsRoot != "" {
			parts = append(parts, "--bids-root", expandUserPath(m.bidsRoot))
		}
	}

	// Spatial modes (features pipeline)
	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		spatial := m.SelectedSpatialModes()
		if len(spatial) > 0 && len(spatial) < len(spatialModes) {
			parts = append(parts, "--spatial")
			parts = append(parts, spatial...)
		}

		// Time ranges
		for _, tr := range m.TimeRanges {
			tmin := tr.Tmin
			if tmin == "" {
				tmin = "none"
			}
			tmax := tr.Tmax
			if tmax == "" {
				tmax = "none"
			}
			parts = append(parts, "--time-range", tr.Name, tmin, tmax)
		}
	}

	// Advanced configuration options (only when not using defaults)
	if !m.useDefaultAdvanced {
		switch m.Pipeline {
		case types.PipelineFeatures:
			parts = append(parts, m.buildFeaturesAdvancedArgs()...)
		case types.PipelineBehavior:
			parts = append(parts, m.buildBehaviorAdvancedArgs()...)
		case types.PipelineDecoding:
			parts = append(parts, m.buildDecodingAdvancedArgs()...)
		case types.PipelinePreprocessing:
			parts = append(parts, m.buildPreprocessingAdvancedArgs()...)
		case types.PipelineRawToBIDS:
			parts = append(parts, m.buildRawToBidsAdvancedArgs()...)
		case types.PipelineMergePsychoPyData:
			parts = append(parts, m.buildMergeBehaviorAdvancedArgs()...)
		}
	}

	if m.task != "" {
		parts = append(parts, "--task", m.task)
	}

	subjs := m.SelectedSubjectIDs()
	if len(subjs) == 0 || len(subjs) == len(m.subjects) {
		parts = append(parts, "--all-subjects")
	} else if len(subjs) <= 10 {
		for _, s := range subjs {
			parts = append(parts, "--subject", s)
		}
	} else {
		parts = append(parts, "--all-subjects")
	}

	return strings.Join(parts, " ")
}

// buildFeaturesAdvancedArgs returns CLI args for features pipeline advanced options
func (m Model) buildFeaturesAdvancedArgs() []string {
	var args []string

	if m.useDefaultAdvanced {
		return args
	}

	// Connectivity options
	if m.isCategorySelected("connectivity") {
		measures := m.selectedConnectivityMeasures()
		if len(measures) > 0 {
			args = append(args, "--connectivity-measures")
			args = append(args, measures...)
		}
		if m.connOutputLevel == 1 {
			args = append(args, "--conn-output-level", "global_only")
		} else {
			args = append(args, "--conn-output-level", "full")
		}

		if m.connGraphMetrics {
			args = append(args, "--conn-graph-metrics")
		} else {
			args = append(args, "--no-conn-graph-metrics")
		}

		args = append(args, "--conn-graph-prop", fmt.Sprintf("%.2f", m.connGraphProp))
		args = append(args, "--conn-window-len", fmt.Sprintf("%.2f", m.connWindowLen))
		args = append(args, "--conn-window-step", fmt.Sprintf("%.2f", m.connWindowStep))

		aecModes := []string{"orth", "none", "sym"}
		if m.connAECMode < len(aecModes) {
			args = append(args, "--conn-aec-mode", aecModes[m.connAECMode])
		}
	}

	// PAC options
	if m.isCategorySelected("pac") {
		args = append(args, "--pac-phase-range", fmt.Sprintf("%.1f", m.pacPhaseMin), fmt.Sprintf("%.1f", m.pacPhaseMax))
		args = append(args, "--pac-amp-range", fmt.Sprintf("%.1f", m.pacAmpMin), fmt.Sprintf("%.1f", m.pacAmpMax))
		pacMethods := []string{"mvl", "kl", "tort", "ozkurt"}
		if m.pacMethod < len(pacMethods) {
			args = append(args, "--pac-method", pacMethods[m.pacMethod])
		}
		args = append(args, "--pac-min-epochs", fmt.Sprintf("%d", m.pacMinEpochs))
		if strings.TrimSpace(m.pacPairsSpec) != "" {
			args = append(args, "--pac-pairs")
			args = append(args, splitCSVList(m.pacPairsSpec)...)
		}
	}

	// Aperiodic options
	if m.isCategorySelected("aperiodic") {
		args = append(args, "--aperiodic-range", fmt.Sprintf("%.1f", m.aperiodicFmin), fmt.Sprintf("%.1f", m.aperiodicFmax))
		args = append(args, "--aperiodic-peak-z", fmt.Sprintf("%.2f", m.aperiodicPeakZ))
		args = append(args, "--aperiodic-min-r2", fmt.Sprintf("%.3f", m.aperiodicMinR2))
		args = append(args, "--aperiodic-min-points", fmt.Sprintf("%d", m.aperiodicMinPoints))
	}

	// Complexity options
	if m.isCategorySelected("complexity") {
		args = append(args, "--pe-order", fmt.Sprintf("%d", m.complexityPEOrder))
		args = append(args, "--pe-delay", fmt.Sprintf("%d", m.complexityPEDelay))
	}

	// ERP options
	if m.isCategorySelected("erp") {
		if m.erpBaselineCorrection {
			args = append(args, "--erp-baseline")
		} else {
			args = append(args, "--no-erp-baseline")
		}
		if m.erpAllowNoBaseline {
			args = append(args, "--erp-allow-no-baseline")
		} else {
			args = append(args, "--no-erp-allow-no-baseline")
		}
		if strings.TrimSpace(m.erpComponentsSpec) != "" {
			args = append(args, "--erp-components")
			args = append(args, splitCSVList(m.erpComponentsSpec)...)
		}
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		args = append(args, "--burst-threshold", fmt.Sprintf("%.2f", m.burstThresholdZ))
		args = append(args, "--burst-min-duration", fmt.Sprintf("%d", m.burstMinDuration))
		if strings.TrimSpace(m.burstBandsSpec) != "" {
			args = append(args, "--burst-bands")
			args = append(args, splitCSVList(m.burstBandsSpec)...)
		}
	}

	// Power options
	if m.isCategorySelected("power") {
		if m.powerRequireBaseline {
			args = append(args, "--power-require-baseline")
		} else {
			args = append(args, "--no-power-require-baseline")
		}
		modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
		if m.powerBaselineMode < len(modes) {
			args = append(args, "--power-baseline-mode", modes[m.powerBaselineMode])
		}
	}

	// Spectral options
	if m.isCategorySelected("spectral") {
		args = append(args, "--spectral-edge-percentile", fmt.Sprintf("%.2f", m.spectralEdgePercentile))
	}

	// Ratios options
	if m.isCategorySelected("ratios") && strings.TrimSpace(m.spectralRatioPairsSpec) != "" {
		args = append(args, "--ratio-pairs")
		args = append(args, splitCSVList(m.spectralRatioPairsSpec)...)
	}

	// Asymmetry options
	if m.isCategorySelected("asymmetry") && strings.TrimSpace(m.asymmetryChannelPairsSpec) != "" {
		args = append(args, "--asymmetry-channel-pairs")
		args = append(args, splitCSVList(m.asymmetryChannelPairsSpec)...)
	}

	// Generic & Validation
	if m.exportAllFeatures {
		args = append(args, "--export-all")
	}

	args = append(args, "--min-epochs", fmt.Sprintf("%d", m.minEpochsForFeatures))
	if m.failOnMissingWindows {
		args = append(args, "--fail-on-missing-windows")
	} else {
		args = append(args, "--no-fail-on-missing-windows")
	}
	if m.failOnMissingNamedWindow {
		args = append(args, "--fail-on-missing-named-window")
	} else {
		args = append(args, "--no-fail-on-missing-named-window")
	}

	return args
}

// buildBehaviorAdvancedArgs returns CLI args for behavior pipeline advanced options
func (m Model) buildBehaviorAdvancedArgs() []string {
	var args []string

	// General / statistics
	if m.correlationMethod != "spearman" {
		args = append(args, "--correlation-method", m.correlationMethod)
	}

	robustMethods := []string{"none", "percentage_bend", "winsorized", "shepherd"}
	if m.robustCorrelation > 0 && m.robustCorrelation < len(robustMethods) {
		args = append(args, "--robust-correlation", robustMethods[m.robustCorrelation])
	}

	if m.bootstrapSamples != 1000 {
		args = append(args, "--bootstrap", fmt.Sprintf("%d", m.bootstrapSamples))
	}

	if m.nPermutations != 1000 {
		args = append(args, "--n-perm", fmt.Sprintf("%d", m.nPermutations))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	if m.behaviorNJobs != -1 {
		args = append(args, "--n-jobs", fmt.Sprintf("%d", m.behaviorNJobs))
	}

	if m.behaviorMinSamples != 10 {
		args = append(args, "--min-samples", fmt.Sprintf("%d", m.behaviorMinSamples))
	}

	if !m.controlTemperature {
		args = append(args, "--no-control-temperature")
	}

	if !m.controlTrialOrder {
		args = append(args, "--no-control-trial-order")
	}
	if !m.trialTableOnly {
		args = append(args, "--no-trial-table-only")
	}

	if m.fdrAlpha != 0.05 {
		args = append(args, "--fdr-alpha", fmt.Sprintf("%.4f", m.fdrAlpha))
	}

	if !m.behaviorComputeChangeScores {
		args = append(args, "--no-compute-change-scores")
	}
	if !m.behaviorComputeLosoStability {
		args = append(args, "--no-loso-stability")
	}
	if m.behaviorComputeBayesFactors {
		args = append(args, "--compute-bayes-factors")
	}

	// Trial table / pain residual / diagnostics
	if m.isComputationSelected("trial_table") {
		formats := []string{"parquet", "tsv"}
		if m.trialTableFormat >= 0 && m.trialTableFormat < len(formats) && m.trialTableFormat != 0 {
			args = append(args, "--trial-table-format", formats[m.trialTableFormat])
		}
		if !m.trialTableIncludeFeatures {
			args = append(args, "--no-trial-table-include-features")
		}
		if !m.trialTableIncludeCovars {
			args = append(args, "--no-trial-table-include-covariates")
		}
		if !m.trialTableIncludeEvents {
			args = append(args, "--no-trial-table-include-events")
		}
		if !m.trialTableAddLagFeatures {
			args = append(args, "--no-trial-table-add-lag-features")
		}
		if strings.TrimSpace(m.trialTableExtraEventCols) != "" {
			args = append(args, "--trial-table-extra-event-columns")
			args = append(args, splitCSVList(m.trialTableExtraEventCols)...)
		}
		if !m.trialTableValidateEnabled {
			args = append(args, "--no-trial-table-validate")
		}
		if m.trialTableRatingMin != 0.0 {
			args = append(args, "--trial-table-rating-min", fmt.Sprintf("%.2f", m.trialTableRatingMin))
		}
		if m.trialTableRatingMax != 10.0 {
			args = append(args, "--trial-table-rating-max", fmt.Sprintf("%.2f", m.trialTableRatingMax))
		}
		if m.trialTableTempMin != 25.0 {
			args = append(args, "--trial-table-temperature-min", fmt.Sprintf("%.2f", m.trialTableTempMin))
		}
		if m.trialTableTempMax != 55.0 {
			args = append(args, "--trial-table-temperature-max", fmt.Sprintf("%.2f", m.trialTableTempMax))
		}
		if m.trialTableHighMissingFrac != 0.5 {
			args = append(args, "--trial-table-high-missing-frac", fmt.Sprintf("%.2f", m.trialTableHighMissingFrac))
		}
		if !m.featureSummariesEnabled {
			args = append(args, "--no-feature-summaries")
		}

		if !m.painResidualEnabled {
			args = append(args, "--no-pain-residual")
		} else {
			methods := []string{"spline", "poly"}
			if m.painResidualMethod >= 0 && m.painResidualMethod < len(methods) && m.painResidualMethod != 0 {
				args = append(args, "--pain-residual-method", methods[m.painResidualMethod])
			}
			if m.painResidualMinSamples != 10 {
				args = append(args, "--pain-residual-min-samples", fmt.Sprintf("%d", m.painResidualMinSamples))
			}
			if m.painResidualPolyDegree != 2 {
				args = append(args, "--pain-residual-poly-degree", fmt.Sprintf("%d", m.painResidualPolyDegree))
			}
		}

		if !m.painResidualModelCompareEnabled {
			args = append(args, "--no-pain-residual-model-compare")
		}
		if m.painResidualModelCompareMinSamples != 10 {
			args = append(args, "--pain-residual-model-compare-min-samples", fmt.Sprintf("%d", m.painResidualModelCompareMinSamples))
		}
		if !m.painResidualBreakpointEnabled {
			args = append(args, "--no-pain-residual-breakpoint-test")
		}
		if m.painResidualBreakpointMinSamples != 12 {
			args = append(args, "--pain-residual-breakpoint-min-samples", fmt.Sprintf("%d", m.painResidualBreakpointMinSamples))
		}
		if m.painResidualBreakpointCandidates != 15 {
			args = append(args, "--pain-residual-breakpoint-candidates", fmt.Sprintf("%d", m.painResidualBreakpointCandidates))
		}
		if m.painResidualBreakpointQlow != 0.15 {
			args = append(args, "--pain-residual-breakpoint-quantile-low", fmt.Sprintf("%.3f", m.painResidualBreakpointQlow))
		}
		if m.painResidualBreakpointQhigh != 0.85 {
			args = append(args, "--pain-residual-breakpoint-quantile-high", fmt.Sprintf("%.3f", m.painResidualBreakpointQhigh))
		}
	}

	// Confounds
	if m.isComputationSelected("confounds") {
		if m.confoundsAddAsCovariates {
			args = append(args, "--confounds-add-as-covariates")
		}
		if m.confoundsMaxCovariates != 3 {
			args = append(args, "--confounds-max-covariates", fmt.Sprintf("%d", m.confoundsMaxCovariates))
		}
		defaultPatterns := "^quality_.*_global_,^quality_.*_ch_"
		pat := strings.TrimSpace(m.confoundsQCColumnPatterns)
		if pat != "" && pat != defaultPatterns {
			args = append(args, "--confounds-qc-column-patterns")
			args = append(args, splitCSVList(pat)...)
		} else if pat == "" && defaultPatterns != "" {
			// Explicit clear
			args = append(args, "--confounds-qc-column-patterns", "none")
		}
	}

	// Regression
	if m.isComputationSelected("regression") {
		featSets := []string{"pain_summaries", "all"}
		if m.regressionFeatureSet >= 0 && m.regressionFeatureSet < len(featSets) && m.regressionFeatureSet != 0 {
			args = append(args, "--regression-feature-set", featSets[m.regressionFeatureSet])
		}
		outcomes := []string{"rating", "pain_residual", "temperature"}
		if m.regressionOutcome >= 0 && m.regressionOutcome < len(outcomes) && m.regressionOutcome != 0 {
			args = append(args, "--regression-outcome", outcomes[m.regressionOutcome])
		}
		if !m.regressionIncludeTemperature {
			args = append(args, "--no-regression-include-temperature")
		}
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.regressionTempControl >= 0 && m.regressionTempControl < len(tempCtrl) && m.regressionTempControl != 0 {
			args = append(args, "--regression-temperature-control", tempCtrl[m.regressionTempControl])
		}
		if m.regressionTempControl == 2 {
			args = append(args, "--regression-temperature-spline-knots", fmt.Sprintf("%d", m.regressionTempSplineKnots))
			args = append(args, "--regression-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.regressionTempSplineQlow))
			args = append(args, "--regression-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.regressionTempSplineQhigh))
			args = append(args, "--regression-temperature-spline-min-samples", fmt.Sprintf("%d", m.regressionTempSplineMinN))
		}
		if !m.regressionIncludeTrialOrder {
			args = append(args, "--no-regression-include-trial-order")
		}
		if m.regressionIncludePrev {
			args = append(args, "--regression-include-prev-terms")
		}
		if !m.regressionIncludeRunBlock {
			args = append(args, "--no-regression-include-run-block")
		}
		if !m.regressionIncludeInteraction {
			args = append(args, "--no-regression-include-interaction")
		}
		if !m.regressionStandardize {
			args = append(args, "--no-regression-standardize")
		}
		if m.regressionMinSamples != 15 {
			args = append(args, "--regression-min-samples", fmt.Sprintf("%d", m.regressionMinSamples))
		}
		if m.regressionPermutations != 0 {
			args = append(args, "--regression-permutations", fmt.Sprintf("%d", m.regressionPermutations))
		}
		if m.regressionMaxFeatures != 0 {
			args = append(args, "--regression-max-features", fmt.Sprintf("%d", m.regressionMaxFeatures))
		}
	}

	// Models
	if m.isComputationSelected("models") {
		featSets := []string{"pain_summaries", "all"}
		if m.modelsFeatureSet >= 0 && m.modelsFeatureSet < len(featSets) && m.modelsFeatureSet != 0 {
			args = append(args, "--models-feature-set", featSets[m.modelsFeatureSet])
		}
		if !m.modelsIncludeTemperature {
			args = append(args, "--no-models-include-temperature")
		}
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.modelsTempControl >= 0 && m.modelsTempControl < len(tempCtrl) && m.modelsTempControl != 0 {
			args = append(args, "--models-temperature-control", tempCtrl[m.modelsTempControl])
		}
		if m.modelsTempControl == 2 {
			args = append(args, "--models-temperature-spline-knots", fmt.Sprintf("%d", m.modelsTempSplineKnots))
			args = append(args, "--models-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.modelsTempSplineQlow))
			args = append(args, "--models-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.modelsTempSplineQhigh))
			args = append(args, "--models-temperature-spline-min-samples", fmt.Sprintf("%d", m.modelsTempSplineMinN))
		}
		if !m.modelsIncludeTrialOrder {
			args = append(args, "--no-models-include-trial-order")
		}
		if m.modelsIncludePrev {
			args = append(args, "--models-include-prev-terms")
		}
		if !m.modelsIncludeRunBlock {
			args = append(args, "--no-models-include-run-block")
		}
		if !m.modelsIncludeInteraction {
			args = append(args, "--no-models-include-interaction")
		}
		if !m.modelsStandardize {
			args = append(args, "--no-models-standardize")
		}
		if m.modelsMinSamples != 20 {
			args = append(args, "--models-min-samples", fmt.Sprintf("%d", m.modelsMinSamples))
		}
		if m.modelsMaxFeatures != 100 {
			args = append(args, "--models-max-features", fmt.Sprintf("%d", m.modelsMaxFeatures))
		}
		out := []string{}
		if m.modelsOutcomeRating {
			out = append(out, "rating")
		}
		if m.modelsOutcomePainResidual {
			out = append(out, "pain_residual")
		}
		if m.modelsOutcomeTemperature {
			out = append(out, "temperature")
		}
		if m.modelsOutcomePainBinary {
			out = append(out, "pain_binary")
		}
		if len(out) > 0 && !(len(out) == 2 && out[0] == "rating" && out[1] == "pain_residual") {
			args = append(args, "--models-outcomes")
			args = append(args, out...)
		}
		fams := []string{}
		if m.modelsFamilyOLS {
			fams = append(fams, "ols_hc3")
		}
		if m.modelsFamilyRobust {
			fams = append(fams, "robust_rlm")
		}
		if m.modelsFamilyQuantile {
			fams = append(fams, "quantile_50")
		}
		if m.modelsFamilyLogit {
			fams = append(fams, "logit")
		}
		if len(fams) > 0 && len(fams) < 4 {
			args = append(args, "--models-families")
			args = append(args, fams...)
		}
		binOut := []string{"pain_binary", "rating_median"}
		if m.modelsBinaryOutcome >= 0 && m.modelsBinaryOutcome < len(binOut) && m.modelsBinaryOutcome != 0 {
			args = append(args, "--models-binary-outcome", binOut[m.modelsBinaryOutcome])
		}
	}

	// Stability
	if m.isComputationSelected("stability") {
		featSets := []string{"pain_summaries", "all"}
		if m.stabilityFeatureSet >= 0 && m.stabilityFeatureSet < len(featSets) && m.stabilityFeatureSet != 0 {
			args = append(args, "--stability-feature-set", featSets[m.stabilityFeatureSet])
		}
		if m.stabilityMethod == 1 {
			args = append(args, "--stability-method", "pearson")
		}
		outcome := []string{"auto", "rating", "pain_residual"}
		if m.stabilityOutcome > 0 && m.stabilityOutcome < len(outcome) {
			args = append(args, "--stability-outcome", outcome[m.stabilityOutcome])
		}
		groupCol := []string{"auto", "run", "block"}
		if m.stabilityGroupColumn > 0 && m.stabilityGroupColumn < len(groupCol) {
			args = append(args, "--stability-group-column", groupCol[m.stabilityGroupColumn])
		}
		if !m.stabilityPartialTemp {
			args = append(args, "--no-stability-partial-temperature")
		}
		if m.stabilityMinGroupTrials != 8 {
			args = append(args, "--stability-min-group-trials", fmt.Sprintf("%d", m.stabilityMinGroupTrials))
		}
		if m.stabilityMaxFeatures != 50 {
			args = append(args, "--stability-max-features", fmt.Sprintf("%d", m.stabilityMaxFeatures))
		}
		if m.stabilityAlpha != 0.05 {
			args = append(args, "--stability-alpha", fmt.Sprintf("%.4f", m.stabilityAlpha))
		}
	}

	// Consistency
	if m.isComputationSelected("consistency") && !m.consistencyEnabled {
		args = append(args, "--no-consistency")
	}

	// Influence
	if m.isComputationSelected("influence") {
		featSets := []string{"pain_summaries", "all"}
		if m.influenceFeatureSet >= 0 && m.influenceFeatureSet < len(featSets) && m.influenceFeatureSet != 0 {
			args = append(args, "--influence-feature-set", featSets[m.influenceFeatureSet])
		}
		out := []string{}
		if m.influenceOutcomeRating {
			out = append(out, "rating")
		}
		if m.influenceOutcomePainResidual {
			out = append(out, "pain_residual")
		}
		if m.influenceOutcomeTemperature {
			out = append(out, "temperature")
		}
		if len(out) > 0 && !(len(out) == 2 && out[0] == "rating" && out[1] == "pain_residual") {
			args = append(args, "--influence-outcomes")
			args = append(args, out...)
		}
		if m.influenceMaxFeatures != 20 {
			args = append(args, "--influence-max-features", fmt.Sprintf("%d", m.influenceMaxFeatures))
		}
		if !m.influenceIncludeTemperature {
			args = append(args, "--no-influence-include-temperature")
		}
		tempCtrl := []string{"linear", "rating_hat", "spline"}
		if m.influenceTempControl >= 0 && m.influenceTempControl < len(tempCtrl) && m.influenceTempControl != 0 {
			args = append(args, "--influence-temperature-control", tempCtrl[m.influenceTempControl])
		}
		if m.influenceTempControl == 2 {
			args = append(args, "--influence-temperature-spline-knots", fmt.Sprintf("%d", m.influenceTempSplineKnots))
			args = append(args, "--influence-temperature-spline-quantile-low", fmt.Sprintf("%.3f", m.influenceTempSplineQlow))
			args = append(args, "--influence-temperature-spline-quantile-high", fmt.Sprintf("%.3f", m.influenceTempSplineQhigh))
			args = append(args, "--influence-temperature-spline-min-samples", fmt.Sprintf("%d", m.influenceTempSplineMinN))
		}
		if !m.influenceIncludeTrialOrder {
			args = append(args, "--no-influence-include-trial-order")
		}
		if !m.influenceIncludeRunBlock {
			args = append(args, "--no-influence-include-run-block")
		}
		if m.influenceIncludeInteraction {
			args = append(args, "--influence-include-interaction")
		}
		if !m.influenceStandardize {
			args = append(args, "--no-influence-standardize")
		}
		if m.influenceCooksThreshold > 0 {
			args = append(args, "--influence-cooks-threshold", fmt.Sprintf("%.6f", m.influenceCooksThreshold))
		}
		if m.influenceLeverageThreshold > 0 {
			args = append(args, "--influence-leverage-threshold", fmt.Sprintf("%.6f", m.influenceLeverageThreshold))
		}
	}

	// Correlations (trial-table)
	if m.isComputationSelected("correlations") {
		featSets := []string{"pain_summaries", "all"}
		if m.correlationsFeatureSet >= 0 && m.correlationsFeatureSet < len(featSets) && m.correlationsFeatureSet != 0 {
			args = append(args, "--correlations-feature-set", featSets[m.correlationsFeatureSet])
		}
		targets := []string{}
		if m.correlationsTargetRating {
			targets = append(targets, "rating")
		}
		if m.correlationsTargetTemperature {
			targets = append(targets, "temperature")
		}
		if m.correlationsTargetPainResidual {
			targets = append(targets, "pain_residual")
		}
		defaultTargets := []string{"rating", "temperature", "pain_residual"}
		if len(targets) > 0 && !(len(targets) == len(defaultTargets) && strings.Join(targets, ",") == strings.Join(defaultTargets, ",")) {
			args = append(args, "--correlations-targets")
			args = append(args, targets...)
		}
	}

	// Report
	if m.isComputationSelected("report") && m.reportTopN != 15 {
		args = append(args, "--report-top-n", fmt.Sprintf("%d", m.reportTopN))
	}

	// Pain sensitivity
	if m.isComputationSelected("pain_sensitivity") && m.painSensitivityMinTrials != 10 {
		args = append(args, "--pain-sensitivity-min-trials", fmt.Sprintf("%d", m.painSensitivityMinTrials))
	}
	if m.isComputationSelected("pain_sensitivity") {
		featSets := []string{"pain_summaries", "all"}
		if m.painSensitivityFeatureSet >= 0 && m.painSensitivityFeatureSet < len(featSets) && m.painSensitivityFeatureSet != 0 {
			args = append(args, "--pain-sensitivity-feature-set", featSets[m.painSensitivityFeatureSet])
		}
	}

	// Condition
	if m.isComputationSelected("condition") {
		if !m.conditionFailFast {
			args = append(args, "--no-condition-fail-fast")
		}
		if m.conditionEffectThreshold != 0.5 {
			args = append(args, "--condition-effect-threshold", fmt.Sprintf("%.4f", m.conditionEffectThreshold))
		}
		if m.conditionMinTrials != 10 {
			args = append(args, "--condition-min-trials", fmt.Sprintf("%d", m.conditionMinTrials))
		}
	}

	// Temporal
	if m.isComputationSelected("temporal") {
		if m.temporalResolutionMs != 50 {
			args = append(args, "--temporal-time-resolution-ms", fmt.Sprintf("%d", m.temporalResolutionMs))
		}
		if m.temporalTimeMinMs != -200 {
			args = append(args, "--temporal-time-min-ms", fmt.Sprintf("%d", m.temporalTimeMinMs))
		}
		if m.temporalTimeMaxMs != 1000 {
			args = append(args, "--temporal-time-max-ms", fmt.Sprintf("%d", m.temporalTimeMaxMs))
		}
		if m.temporalSmoothMs != 100 {
			args = append(args, "--temporal-smooth-window-ms", fmt.Sprintf("%d", m.temporalSmoothMs))
		}
	}

	// Cluster-specific options
	if m.isComputationSelected("cluster") {
		if m.clusterThreshold != 0.05 {
			args = append(args, "--cluster-threshold", fmt.Sprintf("%.3f", m.clusterThreshold))
		}
		if m.clusterMinSize != 2 {
			args = append(args, "--cluster-min-size", fmt.Sprintf("%d", m.clusterMinSize))
		}
		if m.clusterTail != 0 {
			args = append(args, "--cluster-tail", fmt.Sprintf("%d", m.clusterTail))
		}
	}

	// Mediation-specific options
	if m.isComputationSelected("mediation") {
		if m.mediationBootstrap != 1000 {
			args = append(args, "--mediation-bootstrap", fmt.Sprintf("%d", m.mediationBootstrap))
		}
		if m.mediationMinEffect != 0.05 {
			args = append(args, "--mediation-min-effect-size", fmt.Sprintf("%.4f", m.mediationMinEffect))
		}
		if m.mediationMaxMediators != 20 {
			args = append(args, "--mediation-max-mediators", fmt.Sprintf("%d", m.mediationMaxMediators))
		}
	}

	// Mixed effects-specific options
	if m.isComputationSelected("mixed_effects") {
		if m.mixedEffectsType == 1 {
			args = append(args, "--mixed-random-effects", "intercept_slope")
		}
		if m.mixedMaxFeatures != 50 {
			args = append(args, "--mixed-max-features", fmt.Sprintf("%d", m.mixedMaxFeatures))
		}
	}

	return args
}

func splitCSVList(raw string) []string {
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ';' || r == '\t' || r == '\n'
	})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

// buildDecodingAdvancedArgs returns CLI args for decoding pipeline advanced options
func (m Model) buildDecodingAdvancedArgs() []string {
	var args []string

	if m.decodingNPerm > 0 {
		args = append(args, "--n-perm", fmt.Sprintf("%d", m.decodingNPerm))
	}

	if m.innerSplits != 3 {
		args = append(args, "--inner-splits", fmt.Sprintf("%d", m.innerSplits))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	if m.skipTimeGen {
		args = append(args, "--skip-time-gen")
	}

	return args
}

// buildPreprocessingAdvancedArgs returns CLI args for preprocessing advanced options
func (m Model) buildPreprocessingAdvancedArgs() []string {
	var args []string

	if !m.prepUsePyprep {
		args = append(args, "--no-pyprep")
	}
	if !m.prepUseIcalabel {
		args = append(args, "--no-icalabel")
	}
	if m.prepNJobs != 1 {
		args = append(args, "--n-jobs", fmt.Sprintf("%d", m.prepNJobs))
	}

	// Filtering
	if m.prepResample != 500 {
		args = append(args, "--resample", fmt.Sprintf("%d", m.prepResample))
	}
	if m.prepLFreq != 0.1 {
		args = append(args, "--l-freq", fmt.Sprintf("%.1f", m.prepLFreq))
	}
	if m.prepHFreq != 100.0 {
		args = append(args, "--h-freq", fmt.Sprintf("%.1f", m.prepHFreq))
	}
	if m.prepNotch != 60 {
		args = append(args, "--notch", fmt.Sprintf("%d", m.prepNotch))
	}

	// ICA
	if m.prepICAMethod != 0 {
		icaMethodVal := []string{"fastica", "infomax", "picard"}[m.prepICAMethod]
		args = append(args, "--ica-method", icaMethodVal)
	}
	if m.prepICAComp != 0.99 {
		args = append(args, "--ica-components", fmt.Sprintf("%.2f", m.prepICAComp))
	}
	if m.prepProbThresh != 0.8 {
		args = append(args, "--prob-threshold", fmt.Sprintf("%.1f", m.prepProbThresh))
	}

	// Epoching
	if m.prepEpochsTmin != -5.0 {
		args = append(args, "--tmin", fmt.Sprintf("%.1f", m.prepEpochsTmin))
	}
	if m.prepEpochsTmax != 12.0 {
		args = append(args, "--tmax", fmt.Sprintf("%.1f", m.prepEpochsTmax))
	}

	return args
}

// buildRawToBidsAdvancedArgs returns CLI args for raw-to-bids advanced options
func (m Model) buildRawToBidsAdvancedArgs() []string {
	var args []string

	if m.rawMontage != "" && m.rawMontage != "easycap-M1" {
		args = append(args, "--montage", m.rawMontage)
	}
	if m.rawLineFreq != 60 {
		args = append(args, "--line-freq", fmt.Sprintf("%d", m.rawLineFreq))
	}
	if m.rawOverwrite {
		args = append(args, "--overwrite")
	}
	if m.rawZeroBaseOnsets {
		args = append(args, "--zero-base-onsets")
	}
	if m.rawTrimToFirstVolume {
		args = append(args, "--trim-to-first-volume")
	}
	if m.rawEventPrefixes != "" {
		for _, prefix := range splitListInput(m.rawEventPrefixes) {
			args = append(args, "--event-prefix", prefix)
		}
	}
	if m.rawKeepAnnotations {
		args = append(args, "--keep-all-annotations")
	}

	return args
}

// buildMergeBehaviorAdvancedArgs returns CLI args for merge-behavior advanced options
func (m Model) buildMergeBehaviorAdvancedArgs() []string {
	var args []string

	if m.mergeEventPrefixes != "" {
		for _, prefix := range splitListInput(m.mergeEventPrefixes) {
			args = append(args, "--event-prefix", prefix)
		}
	}
	if m.mergeEventTypes != "" {
		for _, eventType := range splitListInput(m.mergeEventTypes) {
			args = append(args, "--event-type", eventType)
		}
	}

	return args
}

func expandUserPath(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return value
	}
	if strings.HasPrefix(value, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			if value == "~" {
				return filepath.Clean(home)
			}
			if strings.HasPrefix(value, "~/") {
				return filepath.Clean(filepath.Join(home, value[2:]))
			}
		}
	}
	return filepath.Clean(value)
}

func splitListInput(value string) []string {
	parts := strings.FieldsFunc(value, func(r rune) bool {
		return r == ',' || r == ';'
	})
	var out []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}
