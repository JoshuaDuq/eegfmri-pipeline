package wizard

import (
	"fmt"
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
		if m.plotSelected[i] {
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

	if len(m.modeOptions) > 0 {
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
		}
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

	// Connectivity options
	if m.isCategorySelected("connectivity") {
		measures := m.selectedConnectivityMeasures()
		if len(measures) > 0 {
			args = append(args, "--connectivity-measures")
			args = append(args, measures...)
		}
		if m.connOutputLevel != 0 {
			val := "full"
			if m.connOutputLevel == 1 {
				val = "global_only"
			}
			args = append(args, "--conn-output-level", val)
		}
		if !m.connGraphMetrics {
			args = append(args, "--no-conn-graph-metrics")
		}
		if m.connAECMode != 0 {
			modes := []string{"orth", "none", "sym"}
			if m.connAECMode < len(modes) {
				args = append(args, "--conn-aec-mode", modes[m.connAECMode])
			}
		}
	}

	// PAC options
	if m.isCategorySelected("pac") {
		if m.pacPhaseMin != 4.0 || m.pacPhaseMax != 8.0 {
			args = append(args, "--pac-phase-range",
				fmt.Sprintf("%.1f", m.pacPhaseMin),
				fmt.Sprintf("%.1f", m.pacPhaseMax))
		}
		if m.pacAmpMin != 30.0 || m.pacAmpMax != 80.0 {
			args = append(args, "--pac-amp-range",
				fmt.Sprintf("%.1f", m.pacAmpMin),
				fmt.Sprintf("%.1f", m.pacAmpMax))
		}
	}

	// Aperiodic options
	if m.isCategorySelected("aperiodic") {
		if m.aperiodicFmin != 2.0 || m.aperiodicFmax != 40.0 {
			args = append(args, "--aperiodic-range",
				fmt.Sprintf("%.1f", m.aperiodicFmin),
				fmt.Sprintf("%.1f", m.aperiodicFmax))
		}
	}

	// Complexity options
	if m.isCategorySelected("complexity") {
		if m.complexityPEOrder != 3 {
			args = append(args, "--pe-order", fmt.Sprintf("%d", m.complexityPEOrder))
		}
	}

	// ERP options
	if m.isCategorySelected("erp") {
		if !m.erpBaselineCorrection {
			args = append(args, "--no-erp-baseline")
		}
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		if m.burstThresholdZ != 2.0 {
			args = append(args, "--burst-threshold", fmt.Sprintf("%.1f", m.burstThresholdZ))
		}
	}

	// Power options
	if m.isCategorySelected("power") {
		if m.powerBaselineMode != 0 {
			modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
			if m.powerBaselineMode < len(modes) {
				args = append(args, "--power-baseline-mode", modes[m.powerBaselineMode])
			}
		}
	}

	// Spectral options
	if m.isCategorySelected("spectral") {
		if m.spectralEdgePercentile != 0.95 {
			args = append(args, "--spectral-edge-percentile", fmt.Sprintf("%.2f", m.spectralEdgePercentile))
		}
	}

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
