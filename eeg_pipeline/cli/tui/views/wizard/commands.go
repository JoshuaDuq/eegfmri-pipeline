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

func (m Model) SelectedTFRROIs() []string {
	var result []string
	for _, roi := range m.tfrROIs {
		if m.tfrROISelected[roi] {
			result = append(result, roi)
		}
	}
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
	} else if m.Pipeline == types.PipelineMergeBehavior || m.Pipeline == types.PipelineRawToBIDS {
		mode := "merge-behavior"
		if m.Pipeline == types.PipelineRawToBIDS {
			mode = "raw-to-bids"
		}
		parts = []string{"eeg-pipeline", "utilities", mode}
	} else {
		// Features pipeline category selection
		cats := m.SelectedCategories()
		if len(cats) > 0 && len(cats) < len(m.categories) {
			parts = append(parts, "--categories")
			parts = append(parts, cats...)
		}
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

	// TFR-specific options
	if m.Pipeline == types.PipelineTFR {
		// Visualization type: 0=TFR, 1=Topomap
		if m.tfrVizType == 1 {
			// Topomap mode
			parts = append(parts, "--tfr-topomaps-only")
		} else {
			// TFR mode: add channel/ROI options
			// Bands
			bands := m.SelectedBands()
			if len(bands) > 0 && len(bands) < len(m.bands) {
				parts = append(parts, "--bands")
				parts = append(parts, bands...)
			}

			// Channel mode
			switch m.tfrChannelMode {
			case 0: // ROI
				parts = append(parts, "--tfr-roi")
				// Add selected ROIs
				selectedROIs := m.SelectedTFRROIs()
				if len(selectedROIs) > 0 && len(selectedROIs) < len(m.tfrROIs) {
					parts = append(parts, "--rois", strings.Join(selectedROIs, ","))
				}
			case 1: // Global/scalp-mean - default behavior
				// No specific flag needed, this is default
			case 2: // All channels
				parts = append(parts, "--all-channels")
			case 3: // Specific channels
				if m.tfrSpecificChans != "" {
					parts = append(parts, "--channels", m.tfrSpecificChans)
				}
			}

			// Time range (single range for TFR)
			if len(m.TimeRanges) > 0 {
				tr := m.TimeRanges[0]
				if tr.Tmin != "" {
					parts = append(parts, "--tmin", tr.Tmin)
				}
				if tr.Tmax != "" {
					parts = append(parts, "--tmax", tr.Tmax)
				}
			}
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

	// Connectivity options (only if connectivity category selected)
	if m.isCategorySelected("connectivity") {
		measures := m.selectedConnectivityMeasures()
		if len(measures) > 0 {
			args = append(args, "--connectivity-measures")
			args = append(args, measures...)
		}
	}

	// PAC options (only if pac category selected)
	if m.isCategorySelected("pac") {
		// Only add if not default values
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

	// Aperiodic options (only if aperiodic category selected)
	if m.isCategorySelected("aperiodic") {
		if m.aperiodicFmin != 2.0 || m.aperiodicFmax != 40.0 {
			args = append(args, "--aperiodic-range",
				fmt.Sprintf("%.1f", m.aperiodicFmin),
				fmt.Sprintf("%.1f", m.aperiodicFmax))
		}
	}

	// Complexity options (only if complexity category selected)
	if m.isCategorySelected("complexity") {
		if m.complexityPEOrder != 3 {
			args = append(args, "--pe-order", fmt.Sprintf("%d", m.complexityPEOrder))
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

	// Note: control_temperature, control_trial_order, fdr_alpha require config override
	// These could be passed as --config-override in future

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
