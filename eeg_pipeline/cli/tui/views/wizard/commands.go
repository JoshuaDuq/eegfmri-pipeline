package wizard

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
)

// File layout notes:
// - `commands.go`: selection getters, shared arg builders, root BuildCommand flow.
// - `commands_build_*.go`: pipeline-specific advanced argument builders.

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
			key := m.computations[i].Key

			switch key {
			case "predictor_residual":
				result = append(result, key, "predictor_models")
			case "validation":
				result = append(result, "consistency", "influence")
			case "regression":
				result = append(result, key)
				if m.modelsFamilyRobust || m.modelsFamilyQuantile || m.modelsFamilyLogit {
					result = append(result, "models")
				}
			default:
				result = append(result, key)
			}
		}
	}

	if len(result) > 0 {
		result = append(result, "trial_table")
	}

	seen := make(map[string]bool)
	unique := make([]string, 0, len(result))
	for _, r := range result {
		if !seen[r] {
			seen[r] = true
			unique = append(unique, r)
		}
	}

	sort.Strings(unique)
	return unique
}

// isComputationSelected checks if a specific computation is currently selected.
// Handles bundled computations: 'validation' includes 'consistency' and 'influence';
// 'predictor_residual' includes 'predictor_models'; 'regression' includes 'models' when multi-family enabled.
func (m Model) isComputationSelected(computation string) bool {
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) {
			key := m.computations[i].Key
			if key == computation {
				return true
			}
			if key == "validation" && (computation == "consistency" || computation == "influence") {
				return true
			}
			if key == "predictor_residual" && computation == "predictor_models" {
				return true
			}
			if key == "regression" && computation == "models" {
				// models is selected if regression is selected and multi-family is enabled
				if m.modelsFamilyRobust || m.modelsFamilyQuantile || m.modelsFamilyLogit {
					return true
				}
			}
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

func (m Model) GetFrequencyBandDefinitions() []string {
	if len(m.bands) == len(frequencyBands) {
		allMatch := true
		for i, band := range m.bands {
			def := frequencyBands[i]
			if band.Key != def.Key || band.LowHz != def.LowHz || band.HighHz != def.HighHz {
				allMatch = false
				break
			}
		}
		if allMatch {
			return nil
		}
	}

	var result []string
	for _, band := range m.bands {
		result = append(result, fmt.Sprintf("%s:%.1f:%.1f", band.Key, band.LowHz, band.HighHz))
	}
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

func (m Model) SelectedROIs() []string {
	var result []string
	for i := 0; i < len(m.rois); i++ {
		if m.roiSelected[i] {
			result = append(result, m.rois[i].Key)
		}
	}
	return result
}

func (m Model) GetROIDefinitions() []string {
	if len(m.rois) == len(defaultROIs) {
		allMatch := true
		for i, roi := range m.rois {
			def := defaultROIs[i]
			if roi.Key != def.Key || roi.Channels != def.Channels {
				allMatch = false
				break
			}
		}
		if allMatch {
			return nil
		}
	}

	// Build set of unavailable channels (case-insensitive) for filtering
	unavailableSet := make(map[string]bool)
	for _, ch := range m.unavailableChannels {
		unavailableSet[strings.ToUpper(strings.TrimSpace(ch))] = true
	}

	var result []string
	for i, roi := range m.rois {
		if m.roiSelected[i] {
			// Filter unavailable channels from ROI channels for CLI
			var filteredChannels []string
			for _, ch := range strings.Split(roi.Channels, ",") {
				ch = strings.TrimSpace(ch)
				if ch != "" && !unavailableSet[strings.ToUpper(ch)] {
					filteredChannels = append(filteredChannels, ch)
				}
			}
			filteredChannelsStr := strings.Join(filteredChannels, ",")
			if filteredChannelsStr != "" {
				result = append(result, fmt.Sprintf("%s:%s", roi.Key, filteredChannelsStr))
			}
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

func (m Model) GetApplicableFeatureFiles() []FeatureFile {
	applicableKeys := make(map[string]bool)

	for i, sel := range m.computationSelected {
		if !sel || i >= len(m.computations) {
			continue
		}
		compKey := m.computations[i].Key
		if features, ok := computationApplicableFeatures[compKey]; ok {
			for _, f := range features {
				applicableKeys[f] = true
			}
		}
	}

	if len(applicableKeys) == 0 {
		return featureFileOptions
	}

	// Filter feature files to only those that are applicable
	var result []FeatureFile
	for _, file := range featureFileOptions {
		if applicableKeys[file.Key] {
			result = append(result, file)
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

func (m Model) plottingNeedsROIs() bool {
	if m.Pipeline != types.PipelinePlotting {
		return true
	}
	selected := m.SelectedPlotIDs()
	if len(selected) == 1 && selected[0] == "band_power_topomaps" {
		return false
	}
	return true
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

func (m Model) SelectedPreprocessingStages() []string {
	var result []string
	for i, sel := range m.prepStageSelected {
		if sel && i < len(m.prepStages) {
			result = append(result, m.prepStages[i].Key)
		}
	}
	sort.Strings(result)
	return result
}

func (m Model) selectedConnectivityMeasures() []string {
	var result []string
	for i, measure := range connectivityMeasures {
		if m.connectivityMeasures[i] {
			result = append(result, measure.Key)
		}
	}
	return result
}

func (m Model) selectedDirectedConnectivityMeasures() []string {
	var result []string
	for i, measure := range directedConnectivityMeasures {
		if m.directedConnMeasures[i] {
			result = append(result, measure.Key)
		}
	}
	return result
}

///////////////////////////////////////////////////////////////////
// Subject Filtering
///////////////////////////////////////////////////////////////////

func (m Model) getFilteredSubjects() []types.SubjectStatus {
	if m.subjectFilter == "" && !m.showOnlyValid {
		return m.subjects
	}

	var filtered []types.SubjectStatus
	filterLower := strings.ToLower(m.subjectFilter)

	for _, s := range m.subjects {
		if m.subjectFilter != "" && !strings.Contains(strings.ToLower(s.ID), filterLower) {
			continue
		}

		if m.showOnlyValid && !m.isSubjectValid(s) {
			continue
		}

		filtered = append(filtered, s)
	}

	return filtered
}

func (m Model) isSubjectValid(s types.SubjectStatus) bool {
	if m.Pipeline == types.PipelinePlotting {
		valid, _ := m.validatePlottingSubject(s)
		return valid
	}
	valid, _ := m.Pipeline.ValidateSubject(s)
	return valid
}

///////////////////////////////////////////////////////////////////
// Command Builder
///////////////////////////////////////////////////////////////////

// argBuilder provides helper methods for building command arguments
type argBuilder struct {
	args []string
}

func newArgBuilder() *argBuilder {
	return &argBuilder{args: make([]string, 0)}
}

func (ab *argBuilder) addIfNonZero(flag string, value float64, format string) {
	if value != 0 {
		ab.args = append(ab.args, flag, fmt.Sprintf(format, value))
	}
}

func (ab *argBuilder) addIfNonZeroInt(flag string, value int) {
	if value != 0 {
		ab.args = append(ab.args, flag, fmt.Sprintf("%d", value))
	}
}

func (ab *argBuilder) addIfNonEmpty(flag string, value string) {
	trimmed := strings.TrimSpace(value)
	if trimmed != "" {
		ab.args = append(ab.args, flag, trimmed)
	}
}

func (ab *argBuilder) addBoolFlag(flag string, value bool) {
	if value {
		ab.args = append(ab.args, flag)
	} else {
		flagName := strings.TrimPrefix(flag, "--")
		ab.args = append(ab.args, "--no-"+flagName)
	}
}

func (ab *argBuilder) addOptionalBoolFlag(flag string, value *bool) {
	if value != nil {
		ab.addBoolFlag(flag, *value)
	}
}

func (ab *argBuilder) addListFlag(flag string, values []string) {
	if len(values) > 0 {
		ab.args = append(ab.args, flag)
		ab.args = append(ab.args, values...)
	}
}

func (ab *argBuilder) addSpaceListFlag(flag string, spec string) {
	trimmed := strings.TrimSpace(spec)
	if trimmed != "" {
		ab.args = append(ab.args, flag)
		ab.args = append(ab.args, splitSpaceList(trimmed)...)
	}
}

func (ab *argBuilder) addSpaceListFlagWithLengthCheck(flag string, spec string, expectedLength int) {
	trimmed := strings.TrimSpace(spec)
	if trimmed != "" {
		vals := splitSpaceList(trimmed)
		if len(vals) == expectedLength {
			ab.args = append(ab.args, flag)
			ab.args = append(ab.args, vals...)
		}
	}
}

func (ab *argBuilder) build() []string {
	return ab.args
}

func (m Model) BuildCommand() string {
	parts := []string{"eeg-pipeline", m.Pipeline.CLICommand()}

	needsMode := m.Pipeline == types.PipelinePreprocessing ||
		m.Pipeline == types.PipelineFeatures ||
		m.Pipeline == types.PipelineBehavior ||
		m.Pipeline == types.PipelinePlotting ||
		m.Pipeline == types.PipelineML ||
		m.Pipeline == types.PipelineFmri ||
		m.Pipeline == types.PipelineFmriAnalysis

	hasValidModeIndex := len(m.modeOptions) > m.modeIndex
	modeToUse := ""
	if needsMode && hasValidModeIndex {
		modeToUse = m.modeOptions[m.modeIndex]

		// Auto-switch to "tfr" mode if TFR plots are selected
		if m.Pipeline == types.PipelinePlotting {
			selectedPlots := m.SelectedPlotIDs()
			for _, plotID := range selectedPlots {
				for _, plot := range m.plotItems {
					if plot.ID == plotID && plot.Group == "tfr" {
						modeToUse = "tfr"
						break
					}
				}
				if modeToUse == "tfr" {
					break
				}
			}
		}

		cliMode := modeToUse
		if m.Pipeline == types.PipelineFmriAnalysis && modeToUse == "trial-signatures" {
			if m.fmriTrialSigMethodIndex%2 == 1 {
				cliMode = "lss"
			} else {
				cliMode = "beta-series"
			}
		}
		parts = append(parts, cliMode)
	}

	if m.Pipeline == types.PipelineML {
		parts = append(parts, "--cv-scope", m.mlScope.CLIValue())
	}

	if m.Pipeline == types.PipelinePlotting {
		if m.plottingScope == PlottingScopeGroup {
			parts = append(parts, "--analysis-scope", m.plottingScope.CLIValue())
		}

		// Always pass --plots to enable independent plot execution
		selectedPlots := m.SelectedPlotIDs()
		if len(selectedPlots) > 0 && len(selectedPlots) < len(m.plotItems) {
			parts = append(parts, "--plots")
			parts = append(parts, selectedPlots...)
		}

		// Per-feature plotter filtering (e.g. select specific Power plots)
		plotters := m.featurePlotterItems()
		if len(plotters) > 0 {
			selected := make([]string, 0, len(plotters))
			for _, p := range plotters {
				if m.featurePlotterSelected[p.ID] {
					selected = append(selected, p.ID)
				}
			}
			if len(selected) > 0 && len(selected) < len(plotters) {
				sort.Strings(selected)
				parts = append(parts, "--feature-plotters")
				parts = append(parts, selected...)
			}
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

		if m.plottingNeedsROIs() {
			// Pass ROI definitions to plotting command
			roiDefs := m.GetROIDefinitions()
			if len(roiDefs) > 0 {
				parts = append(parts, "--rois")
				parts = append(parts, roiDefs...)
			}
		}

		// Pass band selection and definitions to plotting command
		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}

		freqBandDefs := m.GetFrequencyBandDefinitions()
		if len(freqBandDefs) > 0 {
			parts = append(parts, "--frequency-bands")
			parts = append(parts, freqBandDefs...)
		}
	}

	if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		comps := m.SelectedComputations()
		if len(comps) > 0 {
			parts = append(parts, "--computations")
			parts = append(parts, comps...)
		}

		featureFiles := m.SelectedFeatureFiles()
		if len(featureFiles) > 0 && len(featureFiles) < len(m.featureFiles) {
			parts = append(parts, "--feature-files")
			parts = append(parts, featureFiles...)
		}

		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}
	} else if m.Pipeline == types.PipelinePreprocessing {
		mode := m.modeOptions[m.modeIndex]
		if mode == "partial" {
			stages := m.SelectedPreprocessingStages()
			if len(stages) > 0 {
				parts = append(parts, stages...)
			} else {
				parts[len(parts)-1] = "full"
			}
		}
	} else if m.Pipeline != types.PipelinePlotting {
		cats := m.SelectedCategories()
		if len(cats) > 0 && len(cats) < len(m.categories) {
			parts = append(parts, "--categories")
			parts = append(parts, cats...)
		}
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		bands := m.SelectedBands()
		if len(bands) > 0 && len(bands) < len(m.bands) {
			parts = append(parts, "--bands")
			parts = append(parts, bands...)
		}

		freqBandDefs := m.GetFrequencyBandDefinitions()
		if len(freqBandDefs) > 0 {
			parts = append(parts, "--frequency-bands")
			parts = append(parts, freqBandDefs...)
		}

		roiDefs := m.GetROIDefinitions()
		if len(roiDefs) > 0 {
			parts = append(parts, "--rois")
			parts = append(parts, roiDefs...)
		}
	}

	needsPaths := m.Pipeline == types.PipelinePreprocessing ||
		m.Pipeline == types.PipelineFeatures ||
		m.Pipeline == types.PipelineBehavior ||
		m.Pipeline == types.PipelineML ||
		m.Pipeline == types.PipelinePlotting ||
		m.Pipeline == types.PipelineFmri ||
		m.Pipeline == types.PipelineFmriAnalysis

	if needsPaths {
		if m.Pipeline == types.PipelineFmri || m.Pipeline == types.PipelineFmriAnalysis {
			if m.bidsFmriRoot != "" {
				parts = append(parts, "--bids-fmri-root", expandUserPath(m.bidsFmriRoot))
			}
		} else {
			if m.bidsRoot != "" {
				parts = append(parts, "--bids-root", expandUserPath(m.bidsRoot))
			}
		}
		if m.derivRoot != "" {
			parts = append(parts, "--deriv-root", expandUserPath(m.derivRoot))
		}
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		spatial := m.SelectedSpatialModes()
		if len(spatial) > 0 && len(spatial) < len(spatialModes) {
			parts = append(parts, "--spatial")
			parts = append(parts, spatial...)
		}

		for _, tr := range m.TimeRanges {
			tmin := normalizeTimeRangeValue(tr.Tmin)
			tmax := normalizeTimeRangeValue(tr.Tmax)
			parts = append(parts, "--time-range", tr.Name, tmin, tmax)
		}
	}

	if !m.useDefaultAdvanced {
		switch m.Pipeline {
		case types.PipelineFeatures:
			parts = append(parts, m.buildFeaturesAdvancedArgs()...)
		case types.PipelineBehavior:
			parts = append(parts, m.buildBehaviorAdvancedArgs()...)
		case types.PipelinePlotting:
			parts = append(parts, m.buildPlottingAdvancedArgs()...)
		case types.PipelineML:
			parts = append(parts, m.buildMLAdvancedArgs()...)
		case types.PipelinePreprocessing:
			parts = append(parts, m.buildPreprocessingAdvancedArgs()...)
		case types.PipelineFmri:
			parts = append(parts, m.buildFmriAdvancedArgs()...)
		case types.PipelineFmriAnalysis:
			parts = append(parts, m.buildFmriAnalysisAdvancedArgs()...)
		}
	}

	if m.task != "" {
		parts = append(parts, "--task", m.task)
	}

	subjs := m.SelectedSubjectIDs()
	allSubjectsSelected := len(subjs) == 0 || len(subjs) == len(m.subjects)
	if allSubjectsSelected {
		parts = append(parts, "--all-subjects")
	} else {
		for _, s := range subjs {
			parts = append(parts, "--subject", s)
		}
	}

	if m.DryRunMode {
		parts = append(parts, "--dry-run")
	}

	return joinShellCommand(parts)
}

func (m Model) buildPlottingAdvancedArgs() []string {
	ab := newArgBuilder()

	// Plot defaults / styling overrides (mirrors eeg_pipeline/cli/commands/plotting.py)
	ab.addIfNonEmpty("--bbox-inches", m.plotBboxInches)
	ab.addIfNonZero("--pad-inches", m.plotPadInches, "%.4f")

	// Fonts
	ab.addIfNonEmpty("--font-family", m.plotFontFamily)
	ab.addIfNonEmpty("--font-weight", m.plotFontWeight)
	ab.addIfNonZeroInt("--font-size-small", m.plotFontSizeSmall)
	ab.addIfNonZeroInt("--font-size-medium", m.plotFontSizeMedium)
	ab.addIfNonZeroInt("--font-size-large", m.plotFontSizeLarge)
	ab.addIfNonZeroInt("--font-size-title", m.plotFontSizeTitle)
	ab.addIfNonZeroInt("--font-size-annotation", m.plotFontSizeAnnotation)
	ab.addIfNonZeroInt("--font-size-label", m.plotFontSizeLabel)
	ab.addIfNonZeroInt("--font-size-ylabel", m.plotFontSizeYLabel)
	ab.addIfNonZeroInt("--font-size-suptitle", m.plotFontSizeSuptitle)
	ab.addIfNonZeroInt("--font-size-figure-title", m.plotFontSizeFigureTitle)

	// Layout
	ab.addSpaceListFlagWithLengthCheck("--layout-tight-rect", m.plotLayoutTightRectSpec, 4)
	ab.addSpaceListFlagWithLengthCheck("--layout-tight-rect-microstate", m.plotLayoutTightRectMicrostateSpec, 4)
	ab.addSpaceListFlag("--gridspec-width-ratios", m.plotGridSpecWidthRatiosSpec)
	ab.addSpaceListFlag("--gridspec-height-ratios", m.plotGridSpecHeightRatiosSpec)
	ab.addIfNonZero("--gridspec-hspace", m.plotGridSpecHspace, "%.4f")
	ab.addIfNonZero("--gridspec-wspace", m.plotGridSpecWspace, "%.4f")
	ab.addIfNonZero("--gridspec-left", m.plotGridSpecLeft, "%.4f")
	ab.addIfNonZero("--gridspec-right", m.plotGridSpecRight, "%.4f")
	ab.addIfNonZero("--gridspec-top", m.plotGridSpecTop, "%.4f")
	ab.addIfNonZero("--gridspec-bottom", m.plotGridSpecBottom, "%.4f")

	// Figure sizes
	ab.addSpaceListFlagWithLengthCheck("--figure-size-standard", m.plotFigureSizeStandardSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-medium", m.plotFigureSizeMediumSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-small", m.plotFigureSizeSmallSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-square", m.plotFigureSizeSquareSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-wide", m.plotFigureSizeWideSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-tfr", m.plotFigureSizeTFRSpec, 2)
	ab.addSpaceListFlagWithLengthCheck("--figure-size-topomap", m.plotFigureSizeTopomapSpec, 2)

	// Colors
	ab.addIfNonEmpty("--color-condition-2", m.plotColorCondB)
	ab.addIfNonEmpty("--color-condition-1", m.plotColorCondA)
	ab.addIfNonEmpty("--color-significant", m.plotColorSignificant)
	ab.addIfNonEmpty("--color-nonsignificant", m.plotColorNonsignificant)
	ab.addIfNonEmpty("--color-gray", m.plotColorGray)
	ab.addIfNonEmpty("--color-light-gray", m.plotColorLightGray)
	ab.addIfNonEmpty("--color-black", m.plotColorBlack)
	ab.addIfNonEmpty("--color-blue", m.plotColorBlue)
	ab.addIfNonEmpty("--color-red", m.plotColorRed)
	ab.addIfNonEmpty("--color-network-node", m.plotColorNetworkNode)

	// Alpha
	ab.addIfNonZero("--alpha-grid", m.plotAlphaGrid, "%.4f")
	ab.addIfNonZero("--alpha-fill", m.plotAlphaFill, "%.4f")
	ab.addIfNonZero("--alpha-ci", m.plotAlphaCI, "%.4f")
	ab.addIfNonZero("--alpha-ci-line", m.plotAlphaCILine, "%.4f")
	ab.addIfNonZero("--alpha-text-box", m.plotAlphaTextBox, "%.4f")
	ab.addIfNonZero("--alpha-violin-body", m.plotAlphaViolinBody, "%.4f")
	ab.addIfNonZero("--alpha-ridge-fill", m.plotAlphaRidgeFill, "%.4f")

	// Scatter
	ab.addIfNonZeroInt("--scatter-marker-size-small", m.plotScatterMarkerSizeSmall)
	ab.addIfNonZeroInt("--scatter-marker-size-large", m.plotScatterMarkerSizeLarge)
	ab.addIfNonZeroInt("--scatter-marker-size-default", m.plotScatterMarkerSizeDefault)
	ab.addIfNonZero("--scatter-alpha", m.plotScatterAlpha, "%.4f")
	ab.addIfNonEmpty("--scatter-edgecolor", m.plotScatterEdgeColor)
	ab.addIfNonZero("--scatter-edgewidth", m.plotScatterEdgeWidth, "%.4f")

	// Bar
	ab.addIfNonZero("--bar-alpha", m.plotBarAlpha, "%.4f")
	ab.addIfNonZero("--bar-width", m.plotBarWidth, "%.4f")
	ab.addIfNonZeroInt("--bar-capsize", m.plotBarCapsize)
	ab.addIfNonZeroInt("--bar-capsize-large", m.plotBarCapsizeLarge)

	// Line
	ab.addIfNonZero("--line-width-thin", m.plotLineWidthThin, "%.4f")
	ab.addIfNonZero("--line-width-standard", m.plotLineWidthStandard, "%.4f")
	ab.addIfNonZero("--line-width-thick", m.plotLineWidthThick, "%.4f")
	ab.addIfNonZero("--line-width-bold", m.plotLineWidthBold, "%.4f")
	ab.addIfNonZero("--line-alpha-standard", m.plotLineAlphaStandard, "%.4f")
	ab.addIfNonZero("--line-alpha-dim", m.plotLineAlphaDim, "%.4f")
	ab.addIfNonZero("--line-alpha-zero-line", m.plotLineAlphaZeroLine, "%.4f")
	ab.addIfNonZero("--line-alpha-fit-line", m.plotLineAlphaFitLine, "%.4f")
	ab.addIfNonZero("--line-alpha-diagonal", m.plotLineAlphaDiagonal, "%.4f")
	ab.addIfNonZero("--line-alpha-reference", m.plotLineAlphaReference, "%.4f")
	ab.addIfNonZero("--line-regression-width", m.plotLineRegressionWidth, "%.4f")
	ab.addIfNonZero("--line-residual-width", m.plotLineResidualWidth, "%.4f")
	ab.addIfNonZero("--line-qq-width", m.plotLineQQWidth, "%.4f")

	// Histogram
	ab.addIfNonZeroInt("--hist-bins", m.plotHistBins)
	ab.addIfNonZeroInt("--hist-bins-behavioral", m.plotHistBinsBehavioral)
	ab.addIfNonZeroInt("--hist-bins-residual", m.plotHistBinsResidual)
	ab.addIfNonZeroInt("--hist-bins-tfr", m.plotHistBinsTFR)
	ab.addIfNonEmpty("--hist-edgecolor", m.plotHistEdgeColor)
	ab.addIfNonZero("--hist-edgewidth", m.plotHistEdgeWidth, "%.4f")
	ab.addIfNonZero("--hist-alpha", m.plotHistAlpha, "%.4f")
	ab.addIfNonZero("--hist-alpha-residual", m.plotHistAlphaResidual, "%.4f")
	ab.addIfNonZero("--hist-alpha-tfr", m.plotHistAlphaTFR, "%.4f")

	// KDE
	ab.addIfNonZeroInt("--kde-points", m.plotKdePoints)
	ab.addIfNonEmpty("--kde-color", m.plotKdeColor)
	ab.addIfNonZero("--kde-linewidth", m.plotKdeLinewidth, "%.4f")
	ab.addIfNonZero("--kde-alpha", m.plotKdeAlpha, "%.4f")

	// Errorbar
	ab.addIfNonZeroInt("--errorbar-markersize", m.plotErrorbarMarkerSize)
	ab.addIfNonZeroInt("--errorbar-capsize", m.plotErrorbarCapsize)
	ab.addIfNonZeroInt("--errorbar-capsize-large", m.plotErrorbarCapsizeLarge)

	// Text positions
	ab.addIfNonZero("--text-stats-x", m.plotTextStatsX, "%.4f")
	ab.addIfNonZero("--text-stats-y", m.plotTextStatsY, "%.4f")
	ab.addIfNonZero("--text-pvalue-x", m.plotTextPvalueX, "%.4f")
	ab.addIfNonZero("--text-pvalue-y", m.plotTextPvalueY, "%.4f")
	ab.addIfNonZero("--text-bootstrap-x", m.plotTextBootstrapX, "%.4f")
	ab.addIfNonZero("--text-bootstrap-y", m.plotTextBootstrapY, "%.4f")
	ab.addIfNonZero("--text-channel-annotation-x", m.plotTextChannelAnnotationX, "%.4f")
	ab.addIfNonZero("--text-channel-annotation-y", m.plotTextChannelAnnotationY, "%.4f")
	ab.addIfNonZero("--text-title-y", m.plotTextTitleY, "%.4f")
	ab.addIfNonZero("--text-residual-qc-title-y", m.plotTextResidualQcTitleY, "%.4f")

	// Validation
	ab.addIfNonZeroInt("--validation-min-bins-for-calibration", m.plotValidationMinBinsForCalibration)
	ab.addIfNonZeroInt("--validation-max-bins-for-calibration", m.plotValidationMaxBinsForCalibration)
	ab.addIfNonZeroInt("--validation-samples-per-bin", m.plotValidationSamplesPerBin)
	ab.addIfNonZeroInt("--validation-min-rois-for-fdr", m.plotValidationMinRoisForFDR)
	ab.addIfNonZeroInt("--validation-min-pvalues-for-fdr", m.plotValidationMinPvaluesForFDR)

	// Topomap controls
	ab.addIfNonZeroInt("--topomap-contours", m.plotTopomapContours)
	ab.addIfNonEmpty("--topomap-colormap", m.plotTopomapColormap)
	ab.addIfNonZero("--topomap-colorbar-fraction", m.plotTopomapColorbarFraction, "%.4f")
	ab.addIfNonZero("--topomap-colorbar-pad", m.plotTopomapColorbarPad, "%.4f")
	ab.addOptionalBoolFlag("--topomap-diff-annotation-enabled", m.plotTopomapDiffAnnotation)
	ab.addOptionalBoolFlag("--topomap-annotate-descriptive", m.plotTopomapAnnotateDesc)
	ab.addIfNonEmpty("--topomap-sig-mask-marker", m.plotTopomapSigMaskMarker)
	ab.addIfNonEmpty("--topomap-sig-mask-markerfacecolor", m.plotTopomapSigMaskMarkerFaceColor)
	ab.addIfNonEmpty("--topomap-sig-mask-markeredgecolor", m.plotTopomapSigMaskMarkerEdgeColor)
	ab.addIfNonZero("--topomap-sig-mask-linewidth", m.plotTopomapSigMaskLinewidth, "%.4f")
	ab.addIfNonZero("--topomap-sig-mask-markersize", m.plotTopomapSigMaskMarkerSize, "%.4f")

	// TFR controls
	ab.addIfNonZero("--tfr-log-base", m.plotTFRLogBase, "%.4f")
	ab.addIfNonZero("--tfr-percentage-multiplier", m.plotTFRPercentageMultiplier, "%.4f")

	// TFR Topomap controls
	ab.addIfNonZero("--tfr-topomap-window-size-ms", m.plotTFRTopomapWindowSizeMs, "%.1f")
	ab.addIfNonZeroInt("--tfr-topomap-window-count", m.plotTFRTopomapWindowCount)
	ab.addIfNonZero("--tfr-topomap-label-x-position", m.plotTFRTopomapLabelXPosition, "%.2f")
	ab.addIfNonZero("--tfr-topomap-label-y-position-bottom", m.plotTFRTopomapLabelYPositionBottom, "%.2f")
	ab.addIfNonZero("--tfr-topomap-label-y-position", m.plotTFRTopomapLabelYPosition, "%.2f")
	ab.addIfNonZero("--tfr-topomap-title-y", m.plotTFRTopomapTitleY, "%.2f")
	ab.addIfNonZeroInt("--tfr-topomap-title-pad", m.plotTFRTopomapTitlePad)
	ab.addIfNonZero("--tfr-topomap-subplots-right", m.plotTFRTopomapSubplotsRight, "%.2f")
	ab.addIfNonZero("--tfr-topomap-temporal-hspace", m.plotTFRTopomapTemporalHspace, "%.2f")
	ab.addIfNonZero("--tfr-topomap-temporal-wspace", m.plotTFRTopomapTemporalWspace, "%.2f")

	// Sizing controls
	ab.addIfNonZero("--roi-width-per-band", m.plotRoiWidthPerBand, "%.4f")
	ab.addIfNonZero("--roi-width-per-metric", m.plotRoiWidthPerMetric, "%.4f")
	ab.addIfNonZero("--roi-height-per-roi", m.plotRoiHeightPerRoi, "%.4f")
	ab.addIfNonZero("--power-width-per-band", m.plotPowerWidthPerBand, "%.4f")
	ab.addIfNonZero("--power-height-per-segment", m.plotPowerHeightPerSegment, "%.4f")
	ab.addIfNonZero("--itpc-width-per-bin", m.plotItpcWidthPerBin, "%.4f")
	ab.addIfNonZero("--itpc-height-per-band", m.plotItpcHeightPerBand, "%.4f")
	ab.addIfNonZero("--itpc-width-per-band-box", m.plotItpcWidthPerBandBox, "%.4f")
	ab.addIfNonZero("--itpc-height-box", m.plotItpcHeightBox, "%.4f")
	ab.addIfNonEmpty("--pac-cmap", m.plotPacCmap)
	ab.addIfNonZero("--pac-width-per-roi", m.plotPacWidthPerRoi, "%.4f")
	ab.addIfNonZero("--pac-height-box", m.plotPacHeightBox, "%.4f")
	ab.addIfNonZero("--aperiodic-width-per-column", m.plotAperiodicWidthPerColumn, "%.4f")
	ab.addIfNonZero("--aperiodic-height-per-row", m.plotAperiodicHeightPerRow, "%.4f")
	ab.addIfNonZeroInt("--aperiodic-n-perm", m.plotAperiodicNPerm)
	ab.addIfNonZero("--complexity-width-per-measure", m.plotComplexityWidthPerMeasure, "%.4f")
	ab.addIfNonZero("--complexity-height-per-segment", m.plotComplexityHeightPerSegment, "%.4f")
	ab.addIfNonZero("--connectivity-width-per-circle", m.plotConnectivityWidthPerCircle, "%.4f")
	ab.addIfNonZero("--connectivity-width-per-band", m.plotConnectivityWidthPerBand, "%.4f")
	ab.addIfNonZero("--connectivity-height-per-measure", m.plotConnectivityHeightPerMeasure, "%.4f")
	ab.addIfNonZero("--connectivity-circle-top-fraction", m.plotConnectivityCircleTopFraction, "%.4f")
	ab.addIfNonZeroInt("--connectivity-circle-min-lines", m.plotConnectivityCircleMinLines)
	ab.addIfNonZero("--connectivity-network-top-fraction", m.plotConnectivityNetworkTopFraction, "%.4f")
	ab.addIfNonZero("--connectivity-network-top-fraction", m.plotConnectivityNetworkTopFraction, "%.4f")

	// Selection overrides
	ab.addSpaceListFlag("--pac-pairs", m.plotPacPairsSpec)
	measures := m.selectedConnectivityMeasures()
	ab.addListFlag("--connectivity-measures", measures)
	ab.addSpaceListFlag("--spectral-metrics", m.plotSpectralMetricsSpec)
	ab.addSpaceListFlag("--bursts-metrics", m.plotBurstsMetricsSpec)
	ab.addIfNonEmpty("--asymmetry-stat", m.plotAsymmetryStatSpec)
	ab.addSpaceListFlag("--temporal-time-bins", m.plotTemporalTimeBinsSpec)
	ab.addSpaceListFlag("--temporal-time-labels", m.plotTemporalTimeLabelsSpec)

	// Comparisons
	ab.addOptionalBoolFlag("--compare-windows", m.plotCompareWindows)
	ab.addSpaceListFlag("--comparison-windows", m.plotComparisonWindowsSpec)
	ab.addOptionalBoolFlag("--compare-columns", m.plotCompareColumns)
	ab.addIfNonEmpty("--comparison-segment", m.plotComparisonSegment)
	ab.addIfNonEmpty("--comparison-column", m.plotComparisonColumn)
	ab.addSpaceListFlag("--comparison-values", m.plotComparisonValuesSpec)
	ab.addSpaceListFlagWithLengthCheck("--comparison-labels", m.plotComparisonLabelsSpec, 2)
	ab.addSpaceListFlag("--comparison-rois", m.plotComparisonROIsSpec)
	ab.addOptionalBoolFlag("--overwrite", m.plotOverwrite)

	// Per-plot overrides
	ab.args = append(ab.args, m.buildPlotItemConfigArgs()...)

	return ab.build()
}

func (m Model) buildPlotItemConfigArgs() []string {
	var args []string
	plotIDs := m.SelectedPlotIDs()
	for _, plotID := range plotIDs {
		cfg, ok := m.plotItemConfigs[plotID]
		if !ok {
			continue
		}

		if cfg.CompareWindows != nil {
			args = append(args, "--plot-item-config", plotID, "compare_windows", strconv.FormatBool(*cfg.CompareWindows))
		}
		if strings.TrimSpace(cfg.ComparisonWindowsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_windows")
			args = append(args, splitSpaceList(cfg.ComparisonWindowsSpec)...)
		}

		if cfg.CompareColumns != nil {
			args = append(args, "--plot-item-config", plotID, "compare_columns", strconv.FormatBool(*cfg.CompareColumns))
		}
		if strings.TrimSpace(cfg.ComparisonSegment) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_segment", strings.TrimSpace(cfg.ComparisonSegment))
		}
		if strings.TrimSpace(cfg.ComparisonColumn) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_column", strings.TrimSpace(cfg.ComparisonColumn))
		}
		if strings.TrimSpace(cfg.ComparisonValuesSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_values")
			args = append(args, splitSpaceList(cfg.ComparisonValuesSpec)...)
		}
		if strings.TrimSpace(cfg.ComparisonLabelsSpec) != "" {
			vals := splitSpaceList(cfg.ComparisonLabelsSpec)
			if len(vals) == 2 {
				args = append(args, "--plot-item-config", plotID, "comparison_labels")
				args = append(args, vals...)
			}
		}
		if strings.TrimSpace(cfg.ComparisonROIsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_rois")
			args = append(args, splitSpaceList(cfg.ComparisonROIsSpec)...)
		}
		topomapSpec := strings.TrimSpace(cfg.TopomapWindowsSpec)
		if topomapSpec == "" && plotID == "band_power_topomaps" {
			if strings.TrimSpace(cfg.ComparisonWindowsSpec) != "" {
				topomapSpec = strings.TrimSpace(cfg.ComparisonWindowsSpec)
			} else if strings.TrimSpace(m.plotComparisonWindowsSpec) != "" {
				topomapSpec = strings.TrimSpace(m.plotComparisonWindowsSpec)
			}
		}
		if topomapSpec != "" {
			args = append(args, "--plot-item-config", plotID, "topomap_windows")
			args = append(args, splitSpaceList(topomapSpec)...)
		}
		if strings.TrimSpace(cfg.TfrTopomapActiveWindow) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_active_window", strings.TrimSpace(cfg.TfrTopomapActiveWindow))
		}
		if strings.TrimSpace(cfg.TfrTopomapWindowSizeMs) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_window_size_ms", strings.TrimSpace(cfg.TfrTopomapWindowSizeMs))
		}
		if strings.TrimSpace(cfg.TfrTopomapWindowCount) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_window_count", strings.TrimSpace(cfg.TfrTopomapWindowCount))
		}
		if strings.TrimSpace(cfg.TfrTopomapLabelXPosition) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_label_x_position", strings.TrimSpace(cfg.TfrTopomapLabelXPosition))
		}
		if strings.TrimSpace(cfg.TfrTopomapLabelYPositionBottom) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_label_y_position_bottom", strings.TrimSpace(cfg.TfrTopomapLabelYPositionBottom))
		}
		if strings.TrimSpace(cfg.TfrTopomapLabelYPosition) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_label_y_position", strings.TrimSpace(cfg.TfrTopomapLabelYPosition))
		}
		if strings.TrimSpace(cfg.TfrTopomapTitleY) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_title_y", strings.TrimSpace(cfg.TfrTopomapTitleY))
		}
		if strings.TrimSpace(cfg.TfrTopomapTitlePad) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_title_pad", strings.TrimSpace(cfg.TfrTopomapTitlePad))
		}
		if strings.TrimSpace(cfg.TfrTopomapSubplotsRight) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_subplots_right", strings.TrimSpace(cfg.TfrTopomapSubplotsRight))
		}
		if strings.TrimSpace(cfg.TfrTopomapTemporalHspace) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_temporal_hspace", strings.TrimSpace(cfg.TfrTopomapTemporalHspace))
		}
		if strings.TrimSpace(cfg.TfrTopomapTemporalWspace) != "" {
			args = append(args, "--plot-item-config", plotID, "tfr_topomap_temporal_wspace", strings.TrimSpace(cfg.TfrTopomapTemporalWspace))
		}
		if strings.TrimSpace(cfg.ConnectivityCircleTopFraction) != "" {
			args = append(args, "--plot-item-config", plotID, "connectivity_circle_top_fraction", strings.TrimSpace(cfg.ConnectivityCircleTopFraction))
		}
		if strings.TrimSpace(cfg.ConnectivityCircleMinLines) != "" {
			args = append(args, "--plot-item-config", plotID, "connectivity_circle_min_lines", strings.TrimSpace(cfg.ConnectivityCircleMinLines))
		}
		if strings.TrimSpace(cfg.ConnectivityNetworkTopFraction) != "" {
			args = append(args, "--plot-item-config", plotID, "connectivity_network_top_fraction", strings.TrimSpace(cfg.ConnectivityNetworkTopFraction))
		}
		if cfg.ItpcSharedColorbar != nil {
			args = append(args, "--plot-item-config", plotID, "itpc_shared_colorbar", strconv.FormatBool(*cfg.ItpcSharedColorbar))
		}
		// Behavior scatter config
		if strings.TrimSpace(cfg.BehaviorScatterFeaturesSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "scatter_features")
			args = append(args, splitSpaceList(cfg.BehaviorScatterFeaturesSpec)...)
		}
		if strings.TrimSpace(cfg.BehaviorScatterColumnsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "scatter_columns")
			args = append(args, splitSpaceList(cfg.BehaviorScatterColumnsSpec)...)
		}
		if strings.TrimSpace(cfg.BehaviorScatterAggregationModesSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "scatter_aggregation_modes")
			args = append(args, splitSpaceList(cfg.BehaviorScatterAggregationModesSpec)...)
		}
		if strings.TrimSpace(cfg.BehaviorScatterSegmentSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "scatter_segment", strings.TrimSpace(cfg.BehaviorScatterSegmentSpec))
		}
		// Behavior dose-response config
		if strings.TrimSpace(cfg.DoseResponseDoseColumn) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_dose_column", strings.TrimSpace(cfg.DoseResponseDoseColumn))
		}
		if strings.TrimSpace(cfg.DoseResponseResponseColumn) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_response_column")
			args = append(args, splitSpaceList(strings.TrimSpace(cfg.DoseResponseResponseColumn))...)
		}
		if strings.TrimSpace(cfg.DoseResponseBinaryOutcomeColumn) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_binary_outcome_column", strings.TrimSpace(cfg.DoseResponseBinaryOutcomeColumn))
		}
		if strings.TrimSpace(cfg.DoseResponseSegment) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_segment", strings.TrimSpace(cfg.DoseResponseSegment))
		}
		if strings.TrimSpace(cfg.DoseResponseBandsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_bands")
			args = append(args, splitSpaceList(cfg.DoseResponseBandsSpec)...)
		}
		if strings.TrimSpace(cfg.DoseResponseROIsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_rois")
			args = append(args, splitSpaceList(cfg.DoseResponseROIsSpec)...)
		}
		if strings.TrimSpace(cfg.DoseResponseScopesSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_scopes")
			args = append(args, splitSpaceList(cfg.DoseResponseScopesSpec)...)
		}
		if strings.TrimSpace(cfg.DoseResponseStat) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_stat", strings.TrimSpace(cfg.DoseResponseStat))
		}
		if strings.TrimSpace(cfg.BehaviorTemporalStatsFeatureFolder) != "" {
			args = append(
				args,
				"--plot-item-config",
				plotID,
				"temporal_stats_feature_folder",
				strings.TrimSpace(cfg.BehaviorTemporalStatsFeatureFolder),
			)
		}
	}
	return args
}

// buildFeaturesAdvancedArgs returns CLI args for features pipeline advanced options
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

func splitSpaceList(raw string) []string {
	parsed, err := splitShellWords(raw)
	if err == nil && len(parsed) > 0 {
		return parsed
	}
	out := strings.Fields(raw)
	if len(out) == 0 {
		return nil
	}
	return out
}

func splitLooseList(raw string) []string {
	replacer := strings.NewReplacer(
		",", " ",
		";", " ",
		"\t", " ",
		"\n", " ",
		"\r", " ",
	)
	normalized := replacer.Replace(raw)
	return splitSpaceList(normalized)
}

func joinShellCommand(args []string) string {
	if len(args) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, shellQuote(arg))
	}
	return strings.Join(quoted, " ")
}

func shellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if isShellSafe(arg) {
		return arg
	}
	// POSIX-safe single-quote escaping:
	// abc'def -> 'abc'"'"'def'
	return "'" + strings.ReplaceAll(arg, "'", `'"'"'`) + "'"
}

func isShellSafe(arg string) bool {
	for _, r := range arg {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			strings.ContainsRune("@%_+=:,./-~", r) {
			continue
		}
		return false
	}
	return true
}

func splitShellWords(raw string) ([]string, error) {
	type quoteState int
	const (
		stateNone quoteState = iota
		stateSingle
		stateDouble
	)

	var out []string
	var cur strings.Builder
	state := stateNone
	escaped := false

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		out = append(out, cur.String())
		cur.Reset()
	}

	for _, r := range raw {
		if escaped {
			cur.WriteRune(r)
			escaped = false
			continue
		}

		switch state {
		case stateNone:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '\'' {
				state = stateSingle
				continue
			}
			if r == '"' {
				state = stateDouble
				continue
			}
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				flush()
				continue
			}
			cur.WriteRune(r)
		case stateSingle:
			if r == '\'' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		case stateDouble:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		}
	}

	if escaped {
		return nil, fmt.Errorf("unfinished escape sequence")
	}
	if state != stateNone {
		return nil, fmt.Errorf("unterminated quote")
	}
	flush()
	return out, nil
}

// buildMLAdvancedArgs returns CLI args for machine learning pipeline advanced options
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

func normalizeTimeRangeValue(value string) string {
	if value == "" {
		return "none"
	}
	return value
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
