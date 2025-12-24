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
	if m.subjectFilter == "" {
		return m.subjects
	}

	var filtered []types.SubjectStatus
	filterLower := strings.ToLower(m.subjectFilter)
	for _, s := range m.subjects {
		if strings.Contains(strings.ToLower(s.ID), filterLower) {
			filtered = append(filtered, s)
		}
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
		if len(comps) > 0 && len(comps) < len(m.computations) {
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
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		args = append(args, "--burst-threshold", fmt.Sprintf("%.2f", m.burstThresholdZ))
		args = append(args, "--burst-min-duration", fmt.Sprintf("%d", m.burstMinDuration))
	}

	// Power options
	if m.isCategorySelected("power") {
		modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
		if m.powerBaselineMode < len(modes) {
			args = append(args, "--power-baseline-mode", modes[m.powerBaselineMode])
		}
	}

	// Spectral options
	if m.isCategorySelected("spectral") {
		args = append(args, "--spectral-edge-percentile", fmt.Sprintf("%.2f", m.spectralEdgePercentile))
	}

	// Generic & Validation
	if m.exportAllFeatures {
		args = append(args, "--export-all")
	}

	args = append(args, "--min-epochs", fmt.Sprintf("%d", m.minEpochsForFeatures))

	return args
}

// buildBehaviorAdvancedArgs returns CLI args for behavior pipeline advanced options
func (m Model) buildBehaviorAdvancedArgs() []string {
	var args []string

	if m.correlationMethod != "spearman" {
		args = append(args, "--correlation-method", m.correlationMethod)
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

	if !m.controlTemperature {
		args = append(args, "--no-control-temperature")
	}

	if !m.controlTrialOrder {
		args = append(args, "--no-control-trial-order")
	}

	if m.fdrAlpha != 0.05 {
		args = append(args, "--fdr-alpha", fmt.Sprintf("%.2f", m.fdrAlpha))
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
		if m.mediationMaxMediators != 20 {
			args = append(args, "--mediation-max-mediators", fmt.Sprintf("%d", m.mediationMaxMediators))
		}
	}

	// Mixed effects-specific options
	if m.isComputationSelected("mixed_effects") {
		if m.mixedMaxFeatures != 50 {
			args = append(args, "--mixed-max-features", fmt.Sprintf("%d", m.mixedMaxFeatures))
		}
	}

	// Condition-specific options
	if m.isComputationSelected("condition") {
		if m.conditionEffectThreshold != 0.5 {
			args = append(args, "--condition-effect-threshold", fmt.Sprintf("%.1f", m.conditionEffectThreshold))
		}
	}

	return args
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
