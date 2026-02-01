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
			case "pain_residual":
				result = append(result, key, "temperature_models")
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
// 'pain_residual' includes 'temperature_models'; 'regression' includes 'models' when multi-family enabled.
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
			if key == "pain_residual" && computation == "temperature_models" {
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

		// Pass ROI definitions to plotting command
		roiDefs := m.GetROIDefinitions()
		if len(roiDefs) > 0 {
			parts = append(parts, "--rois")
			parts = append(parts, roiDefs...)
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
	} else if m.Pipeline == types.PipelineMergePsychoPyData || m.Pipeline == types.PipelineRawToBIDS || m.Pipeline == types.PipelineFmriRawToBIDS {
		mode := "merge-psychopy"
		if m.Pipeline == types.PipelineRawToBIDS {
			mode = "raw-to-bids"
		} else if m.Pipeline == types.PipelineFmriRawToBIDS {
			mode = "fmri-raw-to-bids"
		}
		parts = []string{"eeg-pipeline", "utilities", mode}
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
	if m.Pipeline == types.PipelineRawToBIDS || m.Pipeline == types.PipelineMergePsychoPyData || m.Pipeline == types.PipelineFmriRawToBIDS {
		if m.sourceRoot != "" {
			parts = append(parts, "--source-root", expandUserPath(m.sourceRoot))
		}
		if m.Pipeline == types.PipelineFmriRawToBIDS {
			if m.bidsFmriRoot != "" {
				parts = append(parts, "--bids-fmri-root", expandUserPath(m.bidsFmriRoot))
			}
		} else if m.bidsRoot != "" {
			parts = append(parts, "--bids-root", expandUserPath(m.bidsRoot))
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
		case types.PipelineRawToBIDS:
			parts = append(parts, m.buildRawToBidsAdvancedArgs()...)
		case types.PipelineFmriRawToBIDS:
			parts = append(parts, m.buildFmriRawToBidsAdvancedArgs()...)
		case types.PipelineMergePsychoPyData:
			parts = append(parts, m.buildMergeBehaviorAdvancedArgs()...)
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
	ab.addIfNonEmpty("--color-condition-2", m.plotColorPain)
	ab.addIfNonEmpty("--color-condition-1", m.plotColorNonpain)
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
		if strings.TrimSpace(cfg.TopomapWindowsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "topomap_windows")
			args = append(args, splitSpaceList(cfg.TopomapWindowsSpec)...)
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
		if strings.TrimSpace(cfg.DoseResponsePainColumn) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_pain_column", strings.TrimSpace(cfg.DoseResponsePainColumn))
		}
		if strings.TrimSpace(cfg.DoseResponseSegment) != "" {
			args = append(args, "--plot-item-config", plotID, "dose_response_segment", strings.TrimSpace(cfg.DoseResponseSegment))
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
func (m Model) buildFeaturesAdvancedArgs() []string {
	var args []string

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
		if m.aperiodicPsdBandwidth > 0 {
			args = append(args, "--aperiodic-psd-bandwidth", fmt.Sprintf("%.1f", m.aperiodicPsdBandwidth))
		}
		if m.aperiodicMaxRms > 0 {
			args = append(args, "--aperiodic-max-rms", fmt.Sprintf("%.3f", m.aperiodicMaxRms))
		}
		if m.aperiodicMinSegmentSec != 2.0 {
			args = append(args, "--aperiodic-min-segment-sec", fmt.Sprintf("%.1f", m.aperiodicMinSegmentSec))
		}
		if m.aperiodicSubtractEvoked {
			args = append(args, "--aperiodic-subtract-evoked")
		}
	}

	if m.isCategorySelected("itpc") {
		itpcMethods := []string{"global", "fold_global", "loo", "condition"}
		if m.itpcMethod >= 0 && m.itpcMethod < len(itpcMethods) && m.itpcMethod != 0 {
			args = append(args, "--itpc-method", itpcMethods[m.itpcMethod])
		}
		if m.itpcMethod == 3 && strings.TrimSpace(m.itpcConditionColumn) != "" {
			args = append(args, "--itpc-condition-column", strings.TrimSpace(m.itpcConditionColumn))
			if strings.TrimSpace(m.itpcConditionValues) != "" {
				spec := strings.ReplaceAll(m.itpcConditionValues, ",", " ")
				vals := strings.Fields(spec)
				if len(vals) > 0 {
					args = append(args, "--itpc-condition-values")
					args = append(args, vals...)
				}
			}
		}
		if m.itpcMinTrialsPerCondition > 0 && m.itpcMinTrialsPerCondition != 10 {
			args = append(args, "--itpc-min-trials-per-condition", fmt.Sprintf("%d", m.itpcMinTrialsPerCondition))
		}
		if m.itpcNJobs != -1 {
			args = append(args, "--itpc-n-jobs", fmt.Sprintf("%d", m.itpcNJobs))
		}
	}

	if m.spatialTransform != 0 {
		transforms := []string{"none", "csd", "laplacian"}
		args = append(args, "--spatial-transform", transforms[m.spatialTransform])
		if m.spatialTransformLambda2 != 1e-5 {
			args = append(args, "--spatial-transform-lambda2", fmt.Sprintf("%.2e", m.spatialTransformLambda2))
		}
		if m.spatialTransformStiffness != 4.0 {
			args = append(args, "--spatial-transform-stiffness", fmt.Sprintf("%.1f", m.spatialTransformStiffness))
		}
	}

	// Additional connectivity scientific validity options
	if m.isCategorySelected("connectivity") {
		// AEC output format
		switch m.connAECOutput {
		case 1:
			args = append(args, "--aec-output", "z")
		case 2:
			args = append(args, "--aec-output", "r", "z")
		}
		// Force within_epoch for machine learning
		if !m.connForceWithinEpochML {
			args = append(args, "--no-conn-force-within-epoch-for-ml")
		}
	}

	if m.isCategorySelected("directedconnectivity") || m.directedConnEnabled {
		directedMeasures := m.selectedDirectedConnectivityMeasures()
		if len(directedMeasures) > 0 {
			args = append(args, "--directed-connectivity-measures")
			args = append(args, directedMeasures...)
		}
		if m.directedConnOutputLevel == 1 {
			args = append(args, "--directed-conn-output-level", "global_only")
		}
		if m.directedConnMvarOrder != 10 {
			args = append(args, "--directed-conn-mvar-order", fmt.Sprintf("%d", m.directedConnMvarOrder))
		}
		if m.directedConnNFreqs != 16 {
			args = append(args, "--directed-conn-n-freqs", fmt.Sprintf("%d", m.directedConnNFreqs))
		}
		if m.directedConnMinSegSamples != 100 {
			args = append(args, "--directed-conn-min-segment-samples", fmt.Sprintf("%d", m.directedConnMinSegSamples))
		}
	}

	// Source localization options (LCMV, eLORETA)
	if m.isCategorySelected("sourcelocalization") || m.sourceLocEnabled {
		methods := []string{"lcmv", "eloreta"}
		args = append(args, "--source-method", methods[m.sourceLocMethod])

		spacings := []string{"oct5", "oct6", "ico4", "ico5"}
		if m.sourceLocSpacing != 1 { // 1 is oct6 default
			args = append(args, "--source-spacing", spacings[m.sourceLocSpacing])
		}

		parcs := []string{"aparc", "aparc.a2009s", "HCPMMP1"}
		if m.sourceLocParc != 0 {
			args = append(args, "--source-parc", parcs[m.sourceLocParc])
		}

		if m.sourceLocMethod == 0 { // LCMV
			if m.sourceLocReg != 0.05 {
				args = append(args, "--source-reg", fmt.Sprintf("%.3f", m.sourceLocReg))
			}
		} else { // eLORETA
			if m.sourceLocSnr != 3.0 {
				args = append(args, "--source-snr", fmt.Sprintf("%.1f", m.sourceLocSnr))
			}
			if m.sourceLocLoose != 0.2 {
				args = append(args, "--source-loose", fmt.Sprintf("%.2f", m.sourceLocLoose))
			}
			if m.sourceLocDepth != 0.8 {
				args = append(args, "--source-depth", fmt.Sprintf("%.2f", m.sourceLocDepth))
			}
		}

		connMethods := []string{"aec", "wpli", "plv"}
		if m.sourceLocConnMethod != 0 {
			args = append(args, "--source-connectivity-method", connMethods[m.sourceLocConnMethod])
		}

		if m.sourceLocMode == 1 {
			if strings.TrimSpace(m.sourceLocSubject) != "" {
				args = append(args, "--source-subject", strings.TrimSpace(m.sourceLocSubject))
			}
			if m.sourceLocCreateTrans {
				args = append(args, "--source-create-trans")
				if m.sourceLocAllowIdentityTrans {
					args = append(args, "--source-allow-identity-trans")
				}
			}
			if m.sourceLocCreateBemModel {
				args = append(args, "--source-create-bem-model")
			}
			if m.sourceLocCreateBemSolution {
				args = append(args, "--source-create-bem-solution")
			}
			// If not auto-creating, user must provide paths
			if !m.sourceLocCreateTrans && strings.TrimSpace(m.sourceLocTrans) != "" {
				args = append(args, "--source-trans", expandUserPath(strings.TrimSpace(m.sourceLocTrans)))
			}
			if !m.sourceLocCreateBemSolution && strings.TrimSpace(m.sourceLocBem) != "" {
				args = append(args, "--source-bem", expandUserPath(strings.TrimSpace(m.sourceLocBem)))
			}
			if m.sourceLocMindistMm != 5.0 {
				args = append(args, "--source-mindist-mm", fmt.Sprintf("%.1f", m.sourceLocMindistMm))
			}

			fmriEnabled := m.sourceLocFmriEnabled || strings.TrimSpace(m.sourceLocFmriStatsMap) != ""
			if fmriEnabled {
				args = append(args, "--source-fmri")
				if strings.TrimSpace(m.sourceLocFmriStatsMap) != "" {
					args = append(args, "--source-fmri-stats-map", expandUserPath(strings.TrimSpace(m.sourceLocFmriStatsMap)))
				}
				provenances := []string{"independent", "same_dataset"}
				if m.sourceLocFmriProvenance >= 0 && m.sourceLocFmriProvenance < len(provenances) && m.sourceLocFmriProvenance != 0 {
					args = append(args, "--source-fmri-provenance", provenances[m.sourceLocFmriProvenance])
				}
				if !m.sourceLocFmriRequireProv {
					args = append(args, "--no-source-fmri-require-provenance")
				}
				if m.sourceLocFmriThreshold != 3.1 {
					args = append(args, "--source-fmri-threshold", fmt.Sprintf("%.2f", m.sourceLocFmriThreshold))
				}
				if m.sourceLocFmriTail == 1 {
					args = append(args, "--source-fmri-tail", "abs")
				}
				if m.sourceLocFmriMinClusterMM3 > 0 {
					args = append(args, "--source-fmri-cluster-min-mm3", fmt.Sprintf("%.0f", m.sourceLocFmriMinClusterMM3))
				} else if m.sourceLocFmriMinClusterVox != 50 {
					args = append(args, "--source-fmri-cluster-min-voxels", fmt.Sprintf("%d", m.sourceLocFmriMinClusterVox))
				}
				if m.sourceLocFmriMaxClusters != 20 {
					args = append(args, "--source-fmri-max-clusters", fmt.Sprintf("%d", m.sourceLocFmriMaxClusters))
				}
				if m.sourceLocFmriMaxVoxPerClus != 2000 {
					args = append(args, "--source-fmri-max-voxels-per-cluster", fmt.Sprintf("%d", m.sourceLocFmriMaxVoxPerClus))
				}
				if m.sourceLocFmriMaxTotalVox != 20000 {
					args = append(args, "--source-fmri-max-total-voxels", fmt.Sprintf("%d", m.sourceLocFmriMaxTotalVox))
				}
				if m.sourceLocFmriRandomSeed != 0 {
					args = append(args, "--source-fmri-random-seed", fmt.Sprintf("%d", m.sourceLocFmriRandomSeed))
				}

				if strings.TrimSpace(m.sourceLocFmriWindowAName) != "" {
					args = append(args, "--source-fmri-window-a-name", strings.TrimSpace(m.sourceLocFmriWindowAName))
					args = append(args, "--source-fmri-window-a-tmin", fmt.Sprintf("%.3f", m.sourceLocFmriWindowATmin))
					args = append(args, "--source-fmri-window-a-tmax", fmt.Sprintf("%.3f", m.sourceLocFmriWindowATmax))
				}
				if strings.TrimSpace(m.sourceLocFmriWindowBName) != "" {
					args = append(args, "--source-fmri-window-b-name", strings.TrimSpace(m.sourceLocFmriWindowBName))
					args = append(args, "--source-fmri-window-b-tmin", fmt.Sprintf("%.3f", m.sourceLocFmriWindowBTmin))
					args = append(args, "--source-fmri-window-b-tmax", fmt.Sprintf("%.3f", m.sourceLocFmriWindowBTmax))
				}

				// fMRI contrast builder options
				if m.sourceLocFmriContrastEnabled {
					args = append(args, "--source-fmri-contrast-enabled")
					contrastTypes := []string{"t-test", "paired-t-test", "f-test", "custom"}
					args = append(args, "--source-fmri-contrast-type", contrastTypes[m.sourceLocFmriContrastType])
					if m.sourceLocFmriContrastType == 3 { // custom formula
						if strings.TrimSpace(m.sourceLocFmriContrastFormula) != "" {
							args = append(args, "--source-fmri-contrast-formula", strings.TrimSpace(m.sourceLocFmriContrastFormula))
						}
					} else {
						if strings.TrimSpace(m.sourceLocFmriCondAColumn) != "" {
							args = append(args, "--source-fmri-cond-a-column", strings.TrimSpace(m.sourceLocFmriCondAColumn))
						}
						if strings.TrimSpace(m.sourceLocFmriCondAValue) != "" {
							args = append(args, "--source-fmri-cond-a-value", strings.TrimSpace(m.sourceLocFmriCondAValue))
						}
						if strings.TrimSpace(m.sourceLocFmriCondBColumn) != "" {
							args = append(args, "--source-fmri-cond-b-column", strings.TrimSpace(m.sourceLocFmriCondBColumn))
						}
						if strings.TrimSpace(m.sourceLocFmriCondBValue) != "" {
							args = append(args, "--source-fmri-cond-b-value", strings.TrimSpace(m.sourceLocFmriCondBValue))
						}
					}
					if strings.TrimSpace(m.sourceLocFmriContrastName) != "" && m.sourceLocFmriContrastName != "pain_vs_baseline" {
						args = append(args, "--source-fmri-contrast-name", strings.TrimSpace(m.sourceLocFmriContrastName))
					}
					if !m.sourceLocFmriAutoDetectRuns && strings.TrimSpace(m.sourceLocFmriRunsToInclude) != "" {
						args = append(args, "--source-fmri-runs", strings.TrimSpace(m.sourceLocFmriRunsToInclude))
					}
					hrfModels := []string{"spm", "flobs", "fir"}
					if m.sourceLocFmriHrfModel != 0 {
						args = append(args, "--source-fmri-hrf-model", hrfModels[m.sourceLocFmriHrfModel])
					}
					driftModels := []string{"none", "cosine", "polynomial"}
					if m.sourceLocFmriDriftModel != 1 { // cosine is default
						args = append(args, "--source-fmri-drift-model", driftModels[m.sourceLocFmriDriftModel])
					}
					if m.sourceLocFmriHighPassHz != 0.008 {
						args = append(args, "--source-fmri-high-pass", fmt.Sprintf("%.4f", m.sourceLocFmriHighPassHz))
					}
					if m.sourceLocFmriLowPassHz > 0 {
						args = append(args, "--source-fmri-low-pass", fmt.Sprintf("%.2f", m.sourceLocFmriLowPassHz))
					}
					if m.sourceLocFmriClusterCorrection {
						args = append(args, "--source-fmri-cluster-correction")
						if m.sourceLocFmriClusterPThreshold != 0.001 {
							args = append(args, "--source-fmri-cluster-p-threshold", fmt.Sprintf("%.4f", m.sourceLocFmriClusterPThreshold))
						}
					}
					outputTypes := []string{"z-score", "t-stat", "cope", "beta"}
					if m.sourceLocFmriOutputType != 0 {
						args = append(args, "--source-fmri-output-type", outputTypes[m.sourceLocFmriOutputType])
					}
					if !m.sourceLocFmriResampleToFS {
						args = append(args, "--no-source-fmri-resample-to-fs")
					}
				}
			}
		}
	}

	// Ratios options (scientific validity)
	if m.isCategorySelected("ratios") {
		if m.ratioSource == 1 {
			args = append(args, "--ratio-source", "powcorr")
		}
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
		if m.erpSmoothMs > 0 {
			args = append(args, "--erp-smooth-ms", fmt.Sprintf("%.1f", m.erpSmoothMs))
		}
		if m.erpPeakProminenceUv > 0 {
			args = append(args, "--erp-peak-prominence-uv", fmt.Sprintf("%.1f", m.erpPeakProminenceUv))
		}
		if m.erpLowpassHz > 0 {
			args = append(args, "--erp-lowpass-hz", fmt.Sprintf("%.1f", m.erpLowpassHz))
		}
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		methods := []string{"percentile", "zscore", "mad"}
		if m.burstThresholdMethod >= 0 && m.burstThresholdMethod < len(methods) {
			args = append(args, "--burst-threshold-method", methods[m.burstThresholdMethod])
		}
		refs := []string{"trial", "subject", "condition"}
		if m.burstThresholdReference >= 0 && m.burstThresholdReference < len(refs) {
			args = append(args, "--burst-threshold-reference", refs[m.burstThresholdReference])
		}
		if m.burstMinTrialsPerCondition != 10 {
			args = append(args, "--burst-min-trials-per-condition", fmt.Sprintf("%d", m.burstMinTrialsPerCondition))
		}
		if m.burstMinSegmentSec != 2.0 {
			args = append(args, "--burst-min-segment-sec", fmt.Sprintf("%.2f", m.burstMinSegmentSec))
		}
		if m.burstSkipInvalidSegments {
			args = append(args, "--burst-skip-invalid-segments")
		} else {
			args = append(args, "--no-burst-skip-invalid-segments")
		}
		if m.burstThresholdMethod == 0 && m.burstThresholdPercentile > 0 {
			args = append(args, "--burst-threshold-percentile", fmt.Sprintf("%.1f", m.burstThresholdPercentile))
		}
		if m.burstThresholdMethod != 0 {
			args = append(args, "--burst-threshold", fmt.Sprintf("%.2f", m.burstThresholdZ))
		}
		args = append(args, "--burst-min-duration", fmt.Sprintf("%d", m.burstMinDuration))
		if m.burstMinCycles != 3.0 {
			args = append(args, "--burst-min-cycles", fmt.Sprintf("%.1f", m.burstMinCycles))
		}
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
		if m.powerSubtractEvoked {
			args = append(args, "--power-subtract-evoked")
		} else {
			args = append(args, "--no-power-subtract-evoked")
		}
		if m.powerMinTrialsPerCondition != 2 {
			args = append(args, "--power-min-trials-per-condition", fmt.Sprintf("%d", m.powerMinTrialsPerCondition))
		}
		if m.powerExcludeLineNoise {
			args = append(args, "--power-exclude-line-noise")
		} else {
			args = append(args, "--no-power-exclude-line-noise")
		}
		if m.powerLineNoiseFreq != 60.0 {
			args = append(args, "--power-line-noise-freq", fmt.Sprintf("%.0f", m.powerLineNoiseFreq))
		}
		if m.powerLineNoiseWidthHz != 1.0 {
			args = append(args, "--power-line-noise-width-hz", fmt.Sprintf("%.1f", m.powerLineNoiseWidthHz))
		}
		if m.powerLineNoiseHarmonics != 3 {
			args = append(args, "--power-line-noise-harmonics", fmt.Sprintf("%d", m.powerLineNoiseHarmonics))
		}
		if m.powerEmitDb {
			args = append(args, "--power-emit-db")
		} else {
			args = append(args, "--no-power-emit-db")
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
	if m.isCategorySelected("asymmetry") && strings.TrimSpace(m.asymmetryActivationBandsSpec) != "" {
		args = append(args, "--asymmetry-activation-bands")
		args = append(args, splitCSVList(m.asymmetryActivationBandsSpec)...)
	}
	if m.isCategorySelected("asymmetry") {
		if m.asymmetryEmitActivationConvention {
			args = append(args, "--asymmetry-emit-activation-convention")
		} else {
			args = append(args, "--no-asymmetry-emit-activation-convention")
		}
	}

	// TFR parameters
	if m.tfrFreqMin != 1.0 {
		args = append(args, "--tfr-freq-min", fmt.Sprintf("%.1f", m.tfrFreqMin))
	}
	if m.tfrFreqMax != 100.0 {
		args = append(args, "--tfr-freq-max", fmt.Sprintf("%.1f", m.tfrFreqMax))
	}
	if m.tfrNFreqs != 40 {
		args = append(args, "--tfr-n-freqs", fmt.Sprintf("%d", m.tfrNFreqs))
	}
	if m.tfrMinCycles != 3.0 {
		args = append(args, "--tfr-min-cycles", fmt.Sprintf("%.1f", m.tfrMinCycles))
	}
	if m.tfrNCyclesFactor != 2.0 {
		args = append(args, "--tfr-n-cycles-factor", fmt.Sprintf("%.1f", m.tfrNCyclesFactor))
	}
	if m.tfrWorkers != -1 {
		args = append(args, "--tfr-workers", fmt.Sprintf("%d", m.tfrWorkers))
	}
	// TFR advanced
	if m.tfrMaxCycles != 15.0 {
		args = append(args, "--tfr-max-cycles", fmt.Sprintf("%.1f", m.tfrMaxCycles))
	}
	if m.tfrDecimPower != 4 {
		args = append(args, "--tfr-decim-power", fmt.Sprintf("%d", m.tfrDecimPower))
	}
	if m.tfrDecimPhase != 1 {
		args = append(args, "--tfr-decim-phase", fmt.Sprintf("%d", m.tfrDecimPhase))
	}

	// ITPC additional options
	if m.itpcAllowUnsafeLoo {
		args = append(args, "--itpc-allow-unsafe-loo")
	}
	if m.itpcBaselineCorrection != 0 {
		modes := []string{"none", "subtract"}
		args = append(args, "--itpc-baseline-correction", modes[m.itpcBaselineCorrection])
	}

	// Spectral advanced options
	if m.isCategorySelected("spectral") || m.isCategorySelected("ratios") {
		if !m.spectralIncludeLogRatios {
			args = append(args, "--no-spectral-include-log-ratios")
		}
		if m.spectralPsdMethod != 0 {
			args = append(args, "--spectral-psd-method", "welch")
		}
		if m.spectralPsdAdaptive {
			args = append(args, "--spectral-psd-adaptive")
		} else {
			args = append(args, "--no-spectral-psd-adaptive")
		}
		if m.spectralMultitaperAdaptive {
			args = append(args, "--spectral-multitaper-adaptive")
		} else {
			args = append(args, "--no-spectral-multitaper-adaptive")
		}
		if m.spectralFmin != 1.0 {
			args = append(args, "--spectral-fmin", fmt.Sprintf("%.1f", m.spectralFmin))
		}
		if m.spectralFmax != 80.0 {
			args = append(args, "--spectral-fmax", fmt.Sprintf("%.1f", m.spectralFmax))
		}
		if !m.spectralExcludeLineNoise {
			args = append(args, "--no-spectral-exclude-line-noise")
		}
		if m.spectralLineNoiseFreq != 60.0 {
			args = append(args, "--spectral-line-noise-freq", fmt.Sprintf("%.0f", m.spectralLineNoiseFreq))
		}
		if m.spectralLineNoiseWidthHz != 1.0 {
			args = append(args, "--spectral-line-noise-width-hz", fmt.Sprintf("%.1f", m.spectralLineNoiseWidthHz))
		}
		if m.spectralLineNoiseHarmonics != 3 {
			args = append(args, "--spectral-line-noise-harmonics", fmt.Sprintf("%d", m.spectralLineNoiseHarmonics))
		}
		if m.spectralMinSegmentSec != 2.0 {
			args = append(args, "--spectral-min-segment-sec", fmt.Sprintf("%.1f", m.spectralMinSegmentSec))
		}
		if m.spectralMinCyclesAtFmin != 3.0 {
			args = append(args, "--spectral-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.spectralMinCyclesAtFmin))
		}
	}

	// Band envelope options
	if m.bandEnvelopePadSec != 0.5 {
		args = append(args, "--band-envelope-pad-sec", fmt.Sprintf("%.2f", m.bandEnvelopePadSec))
	}
	if m.bandEnvelopePadCycles != 3.0 {
		args = append(args, "--band-envelope-pad-cycles", fmt.Sprintf("%.1f", m.bandEnvelopePadCycles))
	}

	// IAF options
	if m.iafEnabled {
		args = append(args, "--iaf-enabled")
		if m.iafAlphaWidthHz != 2.0 {
			args = append(args, "--iaf-alpha-width-hz", fmt.Sprintf("%.1f", m.iafAlphaWidthHz))
		}
		if m.iafSearchRangeMin != 7.0 || m.iafSearchRangeMax != 13.0 {
			args = append(args, "--iaf-search-range", fmt.Sprintf("%.1f", m.iafSearchRangeMin), fmt.Sprintf("%.1f", m.iafSearchRangeMax))
		}
		if m.iafMinProminence != 0.05 {
			args = append(args, "--iaf-min-prominence", fmt.Sprintf("%.3f", m.iafMinProminence))
		}
		if m.iafMinCyclesAtFmin != 5.0 {
			args = append(args, "--iaf-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.iafMinCyclesAtFmin))
		}
		if m.iafMinBaselineSec != 0.0 {
			args = append(args, "--iaf-min-baseline-sec", fmt.Sprintf("%.2f", m.iafMinBaselineSec))
		}
		if m.iafAllowFullFallback {
			args = append(args, "--iaf-allow-full-fallback")
		} else {
			args = append(args, "--no-iaf-allow-full-fallback")
		}
		if m.iafAllowAllChannelsFallback {
			args = append(args, "--iaf-allow-all-channels-fallback")
		} else {
			args = append(args, "--no-iaf-allow-all-channels-fallback")
		}
		rois := m.SelectedROIs()
		if len(rois) > 0 {
			args = append(args, "--iaf-rois")
			args = append(args, rois...)
		}
	}

	// Aperiodic advanced options
	if m.isCategorySelected("aperiodic") {
		if m.aperiodicModel != 0 {
			args = append(args, "--aperiodic-model", "knee")
		}
		if m.aperiodicPsdMethod != 0 {
			args = append(args, "--aperiodic-psd-method", "welch")
		}
		if !m.aperiodicExcludeLineNoise {
			args = append(args, "--no-aperiodic-exclude-line-noise")
		}
		if m.aperiodicLineNoiseFreq != 60.0 {
			args = append(args, "--aperiodic-line-noise-freq", fmt.Sprintf("%.0f", m.aperiodicLineNoiseFreq))
		}
		if m.aperiodicLineNoiseWidthHz != 1.0 {
			args = append(args, "--aperiodic-line-noise-width-hz", fmt.Sprintf("%.1f", m.aperiodicLineNoiseWidthHz))
		}
		if m.aperiodicLineNoiseHarmonics != 3 {
			args = append(args, "--aperiodic-line-noise-harmonics", fmt.Sprintf("%d", m.aperiodicLineNoiseHarmonics))
		}
	}

	// Connectivity advanced options
	if m.isCategorySelected("connectivity") {
		if m.connGranularity != 0 {
			granularities := []string{"trial", "condition", "subject"}
			args = append(args, "--conn-granularity", granularities[m.connGranularity])
		}
		if m.connGranularity == 1 && strings.TrimSpace(m.connConditionColumn) != "" {
			args = append(args, "--conn-condition-column", strings.TrimSpace(m.connConditionColumn))
			if strings.TrimSpace(m.connConditionValues) != "" {
				spec := strings.ReplaceAll(m.connConditionValues, ",", " ")
				vals := strings.Fields(spec)
				if len(vals) > 0 {
					args = append(args, "--conn-condition-values")
					args = append(args, vals...)
				}
			}
		}
		if m.connMinEpochsPerGroup != 5 {
			args = append(args, "--conn-min-epochs-per-group", fmt.Sprintf("%d", m.connMinEpochsPerGroup))
		}
		if m.connMinCyclesPerBand != 3.0 {
			args = append(args, "--conn-min-cycles-per-band", fmt.Sprintf("%.1f", m.connMinCyclesPerBand))
		}
		if !m.connWarnNoSpatialTransform {
			args = append(args, "--no-conn-warn-no-spatial-transform")
		}
		if m.connPhaseEstimator != 0 {
			args = append(args, "--conn-phase-estimator", "across_epochs")
		}
		if m.connMinSegmentSec != 1.0 {
			args = append(args, "--conn-min-segment-sec", fmt.Sprintf("%.1f", m.connMinSegmentSec))
		}
	}

	// PAC advanced options
	if m.isCategorySelected("pac") {
		if m.pacSource != 0 {
			args = append(args, "--pac-source", "tfr")
		}
		if !m.pacNormalize {
			args = append(args, "--no-pac-normalize")
		}
		if m.pacNSurrogates != 0 {
			args = append(args, "--pac-n-surrogates", fmt.Sprintf("%d", m.pacNSurrogates))
		}
		if m.pacAllowHarmonicOvrlap {
			args = append(args, "--pac-allow-harmonic-overlap")
		}
		if m.pacMaxHarmonic != 6 {
			args = append(args, "--pac-max-harmonic", fmt.Sprintf("%d", m.pacMaxHarmonic))
		}
		if m.pacHarmonicToleranceHz != 1.0 {
			args = append(args, "--pac-harmonic-tolerance-hz", fmt.Sprintf("%.1f", m.pacHarmonicToleranceHz))
		}
		if m.pacComputeWaveformQC {
			args = append(args, "--pac-compute-waveform-qc")
		} else {
			args = append(args, "--no-pac-compute-waveform-qc")
		}
		if m.pacWaveformOffsetMs != 5.0 {
			args = append(args, "--pac-waveform-offset-ms", fmt.Sprintf("%.1f", m.pacWaveformOffsetMs))
		}
	}

	// Complexity advanced options
	if m.isCategorySelected("complexity") {
		bases := []string{"filtered", "envelope"}
		basis := "filtered"
		if m.complexitySignalBasis >= 0 && m.complexitySignalBasis < len(bases) {
			basis = bases[m.complexitySignalBasis]
		}
		if basis != "filtered" {
			args = append(args, "--complexity-signal-basis", basis)
		}
		if m.complexityMinSegmentSec != 2.0 {
			args = append(args, "--complexity-min-segment-sec", fmt.Sprintf("%.2f", m.complexityMinSegmentSec))
		}
		if m.complexityMinSamples != 200 {
			args = append(args, "--complexity-min-samples", fmt.Sprintf("%d", m.complexityMinSamples))
		}
		if !m.complexityZscore {
			args = append(args, "--no-complexity-zscore")
		}
	}

	// Ratios advanced options
	if m.isCategorySelected("ratios") {
		if m.ratiosMinSegmentSec != 1.0 {
			args = append(args, "--ratios-min-segment-sec", fmt.Sprintf("%.2f", m.ratiosMinSegmentSec))
		}
		if m.ratiosMinCyclesAtFmin != 3.0 {
			args = append(args, "--ratios-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.ratiosMinCyclesAtFmin))
		}
		if !m.ratiosSkipInvalidSegments {
			args = append(args, "--no-ratios-skip-invalid-segments")
		}
	}

	// Asymmetry advanced options
	if m.isCategorySelected("asymmetry") {
		if m.asymmetryMinSegmentSec != 1.0 {
			args = append(args, "--asymmetry-min-segment-sec", fmt.Sprintf("%.2f", m.asymmetryMinSegmentSec))
		}
		if m.asymmetryMinCyclesAtFmin != 3.0 {
			args = append(args, "--asymmetry-min-cycles-at-fmin", fmt.Sprintf("%.1f", m.asymmetryMinCyclesAtFmin))
		}
		if !m.asymmetrySkipInvalidSegments {
			args = append(args, "--no-asymmetry-skip-invalid-segments")
		}
	}

	// Quality options
	if m.isCategorySelected("quality") {
		if m.qualityPsdMethod != 0 {
			args = append(args, "--quality-psd-method", "multitaper")
		}
		if m.qualityFmin != 1.0 {
			args = append(args, "--quality-fmin", fmt.Sprintf("%.1f", m.qualityFmin))
		}
		if m.qualityFmax != 100.0 {
			args = append(args, "--quality-fmax", fmt.Sprintf("%.1f", m.qualityFmax))
		}
		if m.qualityNfft != 256 {
			args = append(args, "--quality-n-fft", fmt.Sprintf("%d", m.qualityNfft))
		}
		if !m.qualityExcludeLineNoise {
			args = append(args, "--no-quality-exclude-line-noise")
		}
		if m.qualityLineNoiseFreq != 60.0 {
			args = append(args, "--quality-line-noise-freq", fmt.Sprintf("%.0f", m.qualityLineNoiseFreq))
		}
		if m.qualityLineNoiseWidthHz != 1.0 {
			args = append(args, "--quality-line-noise-width-hz", fmt.Sprintf("%.1f", m.qualityLineNoiseWidthHz))
		}
		if m.qualityLineNoiseHarmonics != 3 {
			args = append(args, "--quality-line-noise-harmonics", fmt.Sprintf("%d", m.qualityLineNoiseHarmonics))
		}
		if m.qualitySnrSignalBandMin != 1.0 || m.qualitySnrSignalBandMax != 30.0 {
			args = append(args, "--quality-snr-signal-band", fmt.Sprintf("%.1f", m.qualitySnrSignalBandMin), fmt.Sprintf("%.1f", m.qualitySnrSignalBandMax))
		}
		if m.qualitySnrNoiseBandMin != 40.0 || m.qualitySnrNoiseBandMax != 80.0 {
			args = append(args, "--quality-snr-noise-band", fmt.Sprintf("%.1f", m.qualitySnrNoiseBandMin), fmt.Sprintf("%.1f", m.qualitySnrNoiseBandMax))
		}
		if m.qualityMuscleBandMin != 30.0 || m.qualityMuscleBandMax != 80.0 {
			args = append(args, "--quality-muscle-band", fmt.Sprintf("%.1f", m.qualityMuscleBandMin), fmt.Sprintf("%.1f", m.qualityMuscleBandMax))
		}
	}

	// ERDS options
	if m.isCategorySelected("erds") {
		if m.erdsUseLogRatio {
			args = append(args, "--erds-use-log-ratio")
		}
		if m.erdsMinBaselinePower != 1.0e-12 {
			args = append(args, "--erds-min-baseline-power", fmt.Sprintf("%.2e", m.erdsMinBaselinePower))
		}
		if m.erdsMinActivePower != 1.0e-12 {
			args = append(args, "--erds-min-active-power", fmt.Sprintf("%.2e", m.erdsMinActivePower))
		}
		if m.erdsMinSegmentSec != 0.5 {
			args = append(args, "--erds-min-segment-sec", fmt.Sprintf("%.2f", m.erdsMinSegmentSec))
		}
		if strings.TrimSpace(m.erdsBandsSpec) != "" && m.erdsBandsSpec != "alpha,beta" {
			args = append(args, "--erds-bands")
			args = append(args, splitCSVList(m.erdsBandsSpec)...)
		}
	}

	// Generic & Validation

	args = append(args, "--min-epochs", fmt.Sprintf("%d", m.minEpochsForFeatures))
	analysisModes := []string{"group_stats", "trial_ml_safe"}
	args = append(args, "--analysis-mode", analysisModes[m.featAnalysisMode])

	// Execution options
	if m.featComputeChangeScores {
		args = append(args, "--compute-change-scores")
	} else {
		args = append(args, "--no-compute-change-scores")
	}
	if m.featSaveTfrWithSidecar {
		args = append(args, "--save-tfr-with-sidecar")
	} else {
		args = append(args, "--no-save-tfr-with-sidecar")
	}
	if m.featNJobsBands != -1 {
		args = append(args, "--n-jobs-bands", fmt.Sprintf("%d", m.featNJobsBands))
	}
	if m.featNJobsConnectivity != -1 {
		args = append(args, "--n-jobs-connectivity", fmt.Sprintf("%d", m.featNJobsConnectivity))
	}
	if m.featNJobsAperiodic != -1 {
		args = append(args, "--n-jobs-aperiodic", fmt.Sprintf("%d", m.featNJobsAperiodic))
	}
	if m.featNJobsComplexity != -1 {
		args = append(args, "--n-jobs-complexity", fmt.Sprintf("%d", m.featNJobsComplexity))
	}

	// Storage options
	if m.saveSubjectLevelFeatures {
		args = append(args, "--save-subject-level-features")
	} else {
		args = append(args, "--no-save-subject-level-features")
	}
	if m.featAlsoSaveCsv {
		args = append(args, "--also-save-csv")
	} else {
		args = append(args, "--no-also-save-csv")
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

	if !m.controlTemperature {
		args = append(args, "--no-control-temperature")
	}

	if !m.controlTrialOrder {
		args = append(args, "--no-control-trial-order")
	}

	// Run adjustment (subject-level; optional)
	if m.runAdjustmentEnabled {
		args = append(args, "--run-adjustment")
		col := strings.TrimSpace(m.runAdjustmentColumn)
		if col != "" && col != "run_id" {
			args = append(args, "--run-adjustment-column", col)
		}
		if !m.runAdjustmentIncludeInCorrelations {
			args = append(args, "--no-run-adjustment-include-in-correlations")
		}
		if m.runAdjustmentMaxDummies != 20 {
			args = append(args, "--run-adjustment-max-dummies", fmt.Sprintf("%d", m.runAdjustmentMaxDummies))
		}
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

	// Output options
	if !m.behaviorOverwrite {
		args = append(args, "--no-overwrite")
	}

	// Trial table / pain residual / diagnostics
	if m.isComputationSelected("trial_table") {
		formats := []string{"parquet", "tsv"}
		if m.trialTableFormat >= 0 && m.trialTableFormat < len(formats) && m.trialTableFormat != 0 {
			args = append(args, "--trial-table-format", formats[m.trialTableFormat])
		}
		if !m.trialTableAddLagFeatures {
			args = append(args, "--no-trial-table-add-lag-features")
		}
		if m.trialOrderMaxMissingFraction != 0.1 {
			args = append(args, "--trial-order-max-missing-fraction", fmt.Sprintf("%.3f", m.trialOrderMaxMissingFraction))
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
			if m.painResidualPolyDegree != 2 {
				args = append(args, "--pain-residual-poly-degree", fmt.Sprintf("%d", m.painResidualPolyDegree))
			}
			if strings.TrimSpace(m.painResidualSplineDfCandidates) != "" && m.painResidualSplineDfCandidates != "3,4,5" {
				args = append(args, "--pain-residual-spline-df-candidates")
				args = append(args, splitCSVList(m.painResidualSplineDfCandidates)...)
			}
		}

		if !m.painResidualModelCompareEnabled {
			args = append(args, "--no-pain-residual-model-compare")
		}
		if strings.TrimSpace(m.painResidualModelComparePolyDegrees) != "" && m.painResidualModelComparePolyDegrees != "2,3" {
			args = append(args, "--pain-residual-model-compare-poly-degrees")
			args = append(args, splitCSVList(m.painResidualModelComparePolyDegrees)...)
		}
		if !m.painResidualBreakpointEnabled {
			args = append(args, "--no-pain-residual-breakpoint-test")
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

		if m.painResidualEnabled && m.painResidualCrossfitEnabled {
			args = append(args, "--pain-residual-crossfit")
			if strings.TrimSpace(m.painResidualCrossfitGroupColumn) != "" {
				args = append(args, "--pain-residual-crossfit-group-column", strings.TrimSpace(m.painResidualCrossfitGroupColumn))
			}
			if m.painResidualCrossfitNSplits != 5 {
				args = append(args, "--pain-residual-crossfit-n-splits", fmt.Sprintf("%d", m.painResidualCrossfitNSplits))
			}
			cfMethods := []string{"spline", "poly"}
			if m.painResidualCrossfitMethod >= 0 && m.painResidualCrossfitMethod < len(cfMethods) && m.painResidualCrossfitMethod != 0 {
				args = append(args, "--pain-residual-crossfit-method", cfMethods[m.painResidualCrossfitMethod])
			}
			if m.painResidualCrossfitMethod == 0 && m.painResidualCrossfitSplineKnots != 5 {
				args = append(args, "--pain-residual-crossfit-spline-n-knots", fmt.Sprintf("%d", m.painResidualCrossfitSplineKnots))
			}
		}
	}

	// Feature QC (optional gating)
	if m.isComputationSelected("correlations") || m.isComputationSelected("multilevel_correlations") {
		if m.featureQCEnabled {
			args = append(args, "--feature-qc-enabled")
			if m.featureQCMaxMissingPct != 0.2 {
				args = append(args, "--feature-qc-max-missing-pct", fmt.Sprintf("%.3f", m.featureQCMaxMissingPct))
			}
			if m.featureQCMinVariance != 1e-10 {
				args = append(args, "--feature-qc-min-variance", fmt.Sprintf("%.6e", m.featureQCMinVariance))
			}
			if !m.featureQCCheckWithinRunVariance {
				args = append(args, "--no-feature-qc-check-within-run-variance")
			}
		} else {
			args = append(args, "--no-feature-qc-enabled")
		}
	}

	// Regression
	if m.isComputationSelected("regression") {
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
		if m.regressionPermutations != 0 {
			args = append(args, "--regression-permutations", fmt.Sprintf("%d", m.regressionPermutations))
		}
		if m.regressionMaxFeatures != 0 {
			args = append(args, "--regression-max-features", fmt.Sprintf("%d", m.regressionMaxFeatures))
		}
	}

	// Models
	if m.isComputationSelected("models") {
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
		if !m.correlationsPreferPainResidual {
			args = append(args, "--no-correlations-prefer-pain-residual")
		}
		if strings.TrimSpace(m.correlationsTypesSpec) != "" && m.correlationsTypesSpec != "partial_cov_temp" {
			args = append(args, "--correlations-types")
			args = append(args, splitCSVList(m.correlationsTypesSpec)...)
		}
		if m.correlationsPrimaryUnit == 1 {
			args = append(args, "--correlations-primary-unit", "run_mean")
		}
		if m.correlationsPermutationPrimary {
			args = append(args, "--correlations-permutation-primary")
		}
		if m.correlationsUseCrossfitResidual {
			args = append(args, "--correlations-use-crossfit-pain-residual")
		}
		if strings.TrimSpace(m.correlationsTargetColumn) != "" {
			args = append(args, "--correlations-target-column", m.correlationsTargetColumn)
		}
	}

	// Multilevel correlations (group-level)
	if m.isComputationSelected("multilevel_correlations") && !m.groupLevelBlockPermutation {
		args = append(args, "--no-group-level-block-permutation")
	}

	// Report
	if m.isComputationSelected("report") && m.reportTopN != 15 {
		args = append(args, "--report-top-n", fmt.Sprintf("%d", m.reportTopN))
	}

	// Pain sensitivity

	// Condition
	if m.isComputationSelected("condition") {
		if strings.TrimSpace(m.conditionCompareColumn) != "" {
			args = append(args, "--condition-compare-column", strings.TrimSpace(m.conditionCompareColumn))
		}
		if strings.TrimSpace(m.conditionCompareValues) != "" {
			args = append(args, "--condition-compare-values")
			args = append(args, splitCSVList(m.conditionCompareValues)...)
		}
		if strings.TrimSpace(m.conditionCompareWindows) != "" {
			args = append(args, "--condition-compare-windows")
			args = append(args, splitSpaceList(m.conditionCompareWindows)...)
		}
		if m.conditionWindowPrimaryUnit == 1 {
			args = append(args, "--condition-window-primary-unit", "run_mean")
		}
		if m.conditionPermutationPrimary {
			args = append(args, "--condition-permutation-primary")
		}
		if !m.conditionFailFast {
			args = append(args, "--no-condition-fail-fast")
		}
		if m.conditionEffectThreshold != 0.5 {
			args = append(args, "--condition-effect-threshold", fmt.Sprintf("%.4f", m.conditionEffectThreshold))
		}
		if m.conditionOverwrite {
			args = append(args, "--condition-overwrite")
		} else {
			args = append(args, "--no-condition-overwrite")
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
		if strings.TrimSpace(m.temporalTargetColumn) != "" {
			args = append(args, "--temporal-target-column", strings.TrimSpace(m.temporalTargetColumn))
		}
		if !m.temporalSplitByCondition {
			args = append(args, "--no-temporal-split-by-condition")
		}
		if strings.TrimSpace(m.temporalConditionColumn) != "" {
			args = append(args, "--temporal-condition-column", strings.TrimSpace(m.temporalConditionColumn))
		}
		if strings.TrimSpace(m.temporalConditionValues) != "" {
			args = append(args, "--temporal-condition-values")
			spec := strings.ReplaceAll(m.temporalConditionValues, ",", " ")
			args = append(args, splitSpaceList(spec)...)
		}
		if !m.temporalIncludeROIAverages {
			args = append(args, "--no-temporal-include-roi-averages")
		}
		if !m.temporalIncludeTFGrid {
			args = append(args, "--no-temporal-include-tf-grid")
		}
		// Temporal feature selection
		if !m.temporalFeaturePowerEnabled {
			args = append(args, "--no-temporal-feature-power")
		}
		if m.temporalFeatureITPCEnabled {
			args = append(args, "--temporal-feature-itpc")
		}
		if m.temporalFeatureERDSEnabled {
			args = append(args, "--temporal-feature-erds")
		}
		// ITPC-specific options (only if ITPC is selected in step 3)
		if m.featureFileSelected["itpc"] || m.temporalFeatureITPCEnabled {
			if !m.temporalITPCBaselineCorrection {
				args = append(args, "--no-temporal-itpc-baseline-correction")
			}
			if m.temporalITPCBaselineMin != -0.5 {
				args = append(args, "--temporal-itpc-baseline-min", fmt.Sprintf("%.2f", m.temporalITPCBaselineMin))
			}
			if m.temporalITPCBaselineMax != -0.01 {
				args = append(args, "--temporal-itpc-baseline-max", fmt.Sprintf("%.2f", m.temporalITPCBaselineMax))
			}
		}
		// ERDS-specific options (only if ERDS is selected in step 3)
		if m.featureFileSelected["erds"] || m.temporalFeatureERDSEnabled {
			if m.temporalERDSBaselineMin != -0.5 {
				args = append(args, "--temporal-erds-baseline-min", fmt.Sprintf("%.2f", m.temporalERDSBaselineMin))
			}
			if m.temporalERDSBaselineMax != -0.1 {
				args = append(args, "--temporal-erds-baseline-max", fmt.Sprintf("%.2f", m.temporalERDSBaselineMax))
			}
			if m.temporalERDSMethod != 0 {
				methods := []string{"percent", "zscore"}
				args = append(args, "--temporal-erds-method", methods[m.temporalERDSMethod])
			}
		}
	}

	// Time-frequency heatmap
	if !m.tfHeatmapEnabled {
		args = append(args, "--no-tf-heatmap-enabled")
	} else {
		if strings.TrimSpace(m.tfHeatmapFreqsSpec) != "" && m.tfHeatmapFreqsSpec != "4,8,13,30,45" {
			args = append(args, "--tf-heatmap-freqs")
			for _, f := range splitCSVList(m.tfHeatmapFreqsSpec) {
				args = append(args, f)
			}
		}
		if m.tfHeatmapTimeResMs != 100 {
			args = append(args, "--tf-heatmap-time-resolution-ms", fmt.Sprintf("%d", m.tfHeatmapTimeResMs))
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
		if strings.TrimSpace(m.clusterConditionColumn) != "" {
			args = append(args, "--cluster-condition-column", strings.TrimSpace(m.clusterConditionColumn))
		}
		if strings.TrimSpace(m.clusterConditionValues) != "" {
			args = append(args, "--cluster-condition-values")
			spec := strings.ReplaceAll(m.clusterConditionValues, ",", " ")
			args = append(args, splitSpaceList(spec)...)
		}
	}

	// Mediation-specific options
	if m.isComputationSelected("mediation") {
		if m.mediationBootstrap != 1000 {
			args = append(args, "--mediation-bootstrap", fmt.Sprintf("%d", m.mediationBootstrap))
		}
		if m.mediationPermutations > 0 {
			args = append(args, "--mediation-permutations", fmt.Sprintf("%d", m.mediationPermutations))
		}
		if m.mediationMinEffect != 0.05 {
			args = append(args, "--mediation-min-effect-size", fmt.Sprintf("%.4f", m.mediationMinEffect))
		}
		if m.mediationMaxMediatorsEnabled {
			if m.mediationMaxMediators != 20 {
				args = append(args, "--mediation-max-mediators", fmt.Sprintf("%d", m.mediationMaxMediators))
			}
		}
	}

	// Moderation-specific options
	if m.isComputationSelected("moderation") {
		if m.moderationPermutations > 0 {
			args = append(args, "--moderation-permutations", fmt.Sprintf("%d", m.moderationPermutations))
		}
		if m.moderationMaxFeaturesEnabled {
			if m.moderationMaxFeatures != 50 {
				args = append(args, "--moderation-max-features", fmt.Sprintf("%d", m.moderationMaxFeatures))
			}
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

	// Output options
	if m.alsoSaveCsv {
		args = append(args, "--also-save-csv")
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
func (m Model) buildMLAdvancedArgs() []string {
	var args []string

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}

	if strings.TrimSpace(m.mlTarget) != "" {
		args = append(args, "--target", strings.TrimSpace(m.mlTarget))
	}

	if strings.EqualFold(strings.TrimSpace(m.mlTarget), "fmri_signature") {
		methods := []string{"beta-series", "lss"}
		args = append(args, "--fmri-signature-method", methods[m.mlFmriSigMethodIndex%len(methods)])

		contrast := strings.TrimSpace(m.mlFmriSigContrastName)
		if contrast != "" && contrast != "pain_vs_nonpain" {
			args = append(args, "--fmri-signature-contrast-name", contrast)
		}

		sigs := []string{"NPS", "SIIPS1"}
		args = append(args, "--fmri-signature-name", sigs[m.mlFmriSigSignatureIndex%len(sigs)])

		metrics := []string{"dot", "cosine", "pearson_r"}
		args = append(args, "--fmri-signature-metric", metrics[m.mlFmriSigMetricIndex%len(metrics)])

		norms := []string{
			"none",
			"zscore_within_run",
			"zscore_within_subject",
			"robust_zscore_within_run",
			"robust_zscore_within_subject",
		}
		if m.mlFmriSigNormalizationIndex%len(norms) != 0 {
			args = append(args, "--fmri-signature-normalization", norms[m.mlFmriSigNormalizationIndex%len(norms)])
		}
		if m.mlFmriSigRoundDecimals != 3 {
			args = append(args, "--fmri-signature-round-decimals", fmt.Sprintf("%d", m.mlFmriSigRoundDecimals))
		}
	}

	if mode == "classify" && m.mlBinaryThresholdEnabled {
		args = append(args, "--binary-threshold", fmt.Sprintf("%.6g", m.mlBinaryThreshold))
	}

	if strings.TrimSpace(m.mlFeatureFamiliesSpec) != "" {
		args = append(args, "--feature-families")
		args = append(args, splitLooseList(m.mlFeatureFamiliesSpec)...)
	}

	if strings.TrimSpace(m.mlFeatureBandsSpec) != "" {
		args = append(args, "--feature-bands")
		args = append(args, splitLooseList(m.mlFeatureBandsSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureSegmentsSpec) != "" {
		args = append(args, "--feature-segments")
		args = append(args, splitLooseList(m.mlFeatureSegmentsSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureScopesSpec) != "" {
		args = append(args, "--feature-scopes")
		args = append(args, splitLooseList(m.mlFeatureScopesSpec)...)
	}
	if strings.TrimSpace(m.mlFeatureStatsSpec) != "" {
		args = append(args, "--feature-stats")
		args = append(args, splitLooseList(m.mlFeatureStatsSpec)...)
	}

	if v := m.mlFeatureHarmonization.CLIValue(); v != "" {
		args = append(args, "--feature-harmonization", v)
	}

	if strings.TrimSpace(m.mlCovariatesSpec) != "" {
		args = append(args, "--covariates")
		args = append(args, splitLooseList(m.mlCovariatesSpec)...)
	}

	if mode == "incremental_validity" && strings.TrimSpace(m.mlBaselinePredictorsSpec) != "" {
		args = append(args, "--baseline-predictors")
		args = append(args, splitLooseList(m.mlBaselinePredictorsSpec)...)
	}

	if mode == "classify" {
		if v := m.mlClassificationModel.CLIValue(); v != "" {
			args = append(args, "--classification-model", v)
		}
	}

	if m.mlRequireTrialMlSafe {
		args = append(args, "--require-trial-ml-safe")
	}

	if mode != "classify" && mode != "timegen" && mode != "model_comparison" && m.mlRegressionModel != MLRegressionElasticNet {
		args = append(args, "--model", m.mlRegressionModel.CLIValue())
	}

	if mode == "uncertainty" && m.mlUncertaintyAlpha != 0.1 {
		args = append(args, "--uncertainty-alpha", fmt.Sprintf("%.6g", m.mlUncertaintyAlpha))
	}

	if mode == "permutation" && m.mlPermNRepeats != 10 {
		args = append(args, "--perm-n-repeats", fmt.Sprintf("%d", m.mlPermNRepeats))
	}

	if m.mlNPerm > 0 {
		args = append(args, "--n-perm", fmt.Sprintf("%d", m.mlNPerm))
	}

	if m.innerSplits != 3 {
		args = append(args, "--inner-splits", fmt.Sprintf("%d", m.innerSplits))
	}

	if m.outerJobs != 1 {
		args = append(args, "--outer-jobs", fmt.Sprintf("%d", m.outerJobs))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	// ElasticNet hyperparameters
	if strings.TrimSpace(m.elasticNetAlphaGrid) != "" && m.elasticNetAlphaGrid != "0.001,0.01,0.1,1,10" {
		args = append(args, "--elasticnet-alpha-grid")
		args = append(args, splitLooseList(m.elasticNetAlphaGrid)...)
	}
	if strings.TrimSpace(m.elasticNetL1RatioGrid) != "" && m.elasticNetL1RatioGrid != "0.2,0.5,0.8" {
		args = append(args, "--elasticnet-l1-ratio-grid")
		args = append(args, splitLooseList(m.elasticNetL1RatioGrid)...)
	}

	// Ridge hyperparameters
	if strings.TrimSpace(m.ridgeAlphaGrid) != "" && m.ridgeAlphaGrid != "0.01,0.1,1,10,100" {
		args = append(args, "--ridge-alpha-grid")
		args = append(args, splitLooseList(m.ridgeAlphaGrid)...)
	}

	// Random Forest hyperparameters
	if m.rfNEstimators != 500 {
		args = append(args, "--rf-n-estimators", fmt.Sprintf("%d", m.rfNEstimators))
	}
	if strings.TrimSpace(m.rfMaxDepthGrid) != "" && m.rfMaxDepthGrid != "5,10,20,null" {
		args = append(args, "--rf-max-depth-grid")
		args = append(args, splitLooseList(m.rfMaxDepthGrid)...)
	}

	if strings.TrimSpace(m.varianceThresholdGrid) != "" && m.varianceThresholdGrid != "0.0,0.01,0.1" {
		args = append(args, "--variance-threshold-grid")
		args = append(args, splitLooseList(m.varianceThresholdGrid)...)
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
	if strings.TrimSpace(m.prepMontage) != "" && m.prepMontage != "easycap-M1" {
		args = append(args, "--montage", m.prepMontage)
	}
	if strings.TrimSpace(m.prepChTypes) != "" && m.prepChTypes != "eeg" {
		args = append(args, "--ch-types", m.prepChTypes)
	}
	if strings.TrimSpace(m.prepEegReference) != "" && m.prepEegReference != "average" {
		args = append(args, "--eeg-reference", m.prepEegReference)
	}
	if strings.TrimSpace(m.prepEogChannels) != "" {
		args = append(args, "--eog-channels", m.prepEogChannels)
	}
	if m.prepRandomState != 42 {
		args = append(args, "--random-state", fmt.Sprintf("%d", m.prepRandomState))
	}
	if m.prepTaskIsRest {
		args = append(args, "--task-is-rest")
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
	if m.prepLineFreq != 0 && m.prepLineFreq != 60 {
		args = append(args, "--line-freq", fmt.Sprintf("%d", m.prepLineFreq))
	}
	if m.prepZaplineFline != 60.0 {
		args = append(args, "--zapline-fline", fmt.Sprintf("%.1f", m.prepZaplineFline))
	}
	if !m.prepFindBreaks {
		args = append(args, "--no-find-breaks")
	}

	// ICA
	if m.prepSpatialFilter != 0 {
		spatialFilterVal := []string{"ica", "ssp"}[m.prepSpatialFilter]
		args = append(args, "--spatial-filter", spatialFilterVal)
	}
	if m.prepICAAlgorithm != 0 {
		icaMethodVal := []string{"extended_infomax", "fastica", "infomax", "picard"}[m.prepICAAlgorithm]
		args = append(args, "--ica-method", icaMethodVal)
	}
	if m.prepICAComp != 0.99 {
		args = append(args, "--ica-components", fmt.Sprintf("%.2f", m.prepICAComp))
	}
	if m.prepICALFreq != 1.0 {
		args = append(args, "--ica-l-freq", fmt.Sprintf("%.1f", m.prepICALFreq))
	}
	if m.prepICARejThresh != 500.0 {
		args = append(args, "--ica-reject", fmt.Sprintf("%.0f", m.prepICARejThresh))
	}
	if m.prepProbThresh != 0.8 {
		args = append(args, "--prob-threshold", fmt.Sprintf("%.1f", m.prepProbThresh))
	}
	if strings.TrimSpace(m.icaLabelsToKeep) != "" && m.icaLabelsToKeep != "brain,other" {
		args = append(args, "--ica-labels-to-keep")
		args = append(args, splitCSVList(m.icaLabelsToKeep)...)
	}
	if m.prepKeepMnebidsBads {
		args = append(args, "--keep-mnebids-bads")
	}

	// PyPREP advanced options
	if !m.prepRansac {
		args = append(args, "--no-ransac")
	}
	if m.prepRepeats != 3 {
		args = append(args, "--repeats", fmt.Sprintf("%d", m.prepRepeats))
	}
	if m.prepAverageReref {
		args = append(args, "--average-reref")
	}
	if strings.TrimSpace(m.prepFileExtension) != "" && m.prepFileExtension != ".vhdr" {
		args = append(args, "--file-extension", m.prepFileExtension)
	}
	if m.prepConsiderPreviousBads {
		args = append(args, "--consider-previous-bads")
	} else {
		args = append(args, "--no-consider-previous-bads")
	}
	if !m.prepOverwriteChansTsv {
		args = append(args, "--no-overwrite-channels-tsv")
	}
	if m.prepDeleteBreaks {
		args = append(args, "--delete-breaks")
	}
	if m.prepBreaksMinLength != 20 {
		args = append(args, "--breaks-min-length", fmt.Sprintf("%d", m.prepBreaksMinLength))
	}
	if m.prepTStartAfterPrevious != 2 {
		args = append(args, "--t-start-after-previous", fmt.Sprintf("%d", m.prepTStartAfterPrevious))
	}
	if m.prepTStopBeforeNext != 2 {
		args = append(args, "--t-stop-before-next", fmt.Sprintf("%d", m.prepTStopBeforeNext))
	}

	// Epoching
	if strings.TrimSpace(m.prepConditions) != "" {
		args = append(args, "--conditions", m.prepConditions)
	}
	if m.prepEpochsTmin != -5.0 {
		args = append(args, "--tmin", fmt.Sprintf("%.1f", m.prepEpochsTmin))
	}
	if m.prepEpochsTmax != 15.0 {
		args = append(args, "--tmax", fmt.Sprintf("%.1f", m.prepEpochsTmax))
	}
	if m.prepEpochsNoBaseline {
		args = append(args, "--no-baseline")
	} else if m.prepEpochsBaselineStart != -2.0 || m.prepEpochsBaselineEnd != 0.0 {
		args = append(args, "--baseline", fmt.Sprintf("%.2f", m.prepEpochsBaselineStart), fmt.Sprintf("%.2f", m.prepEpochsBaselineEnd))
	}
	if m.prepEpochsReject > 0 {
		args = append(args, "--reject", fmt.Sprintf("%.0f", m.prepEpochsReject))
	}
	if m.prepRejectMethod != 1 {
		rejectMethodVal := []string{"none", "autoreject_local", "autoreject_global"}[m.prepRejectMethod]
		args = append(args, "--reject-method", rejectMethodVal)
	}
	if m.prepRunSourceEstimation {
		args = append(args, "--run-source-estimation")
	}

	// Clean events.tsv options
	if !m.prepWriteCleanEvents {
		args = append(args, "--no-write-clean-events")
	}
	if !m.prepOverwriteCleanEvents {
		args = append(args, "--no-overwrite-clean-events")
	}
	if !m.prepCleanEventsStrict {
		args = append(args, "--no-clean-events-strict")
	}

	return args
}

func (m Model) buildFmriAdvancedArgs() []string {
	ab := newArgBuilder()

	// Runtime
	engine := "docker"
	if m.fmriEngineIndex%2 == 1 {
		engine = "apptainer"
	}
	ab.args = append(ab.args, "--engine", engine)
	ab.addIfNonEmpty("--fmriprep-image", m.fmriFmriprepImage)
	ab.addIfNonEmpty("--fmriprep-output-dir", expandUserPath(m.fmriFmriprepOutputDir))
	ab.addIfNonEmpty("--fmriprep-work-dir", expandUserPath(m.fmriFmriprepWorkDir))
	ab.addIfNonEmpty("--fs-license-file", expandUserPath(m.fmriFreesurferLicenseFile))
	ab.addIfNonEmpty("--fs-subjects-dir", expandUserPath(m.fmriFreesurferSubjectsDir))

	// Output
	ab.addSpaceListFlag("--output-spaces", m.fmriOutputSpacesSpec)
	ab.addSpaceListFlag("--ignore", m.fmriIgnoreSpec)
	ab.addIfNonEmpty("--bids-filter-file", expandUserPath(m.fmriBidsFilterFile))

	levelOptions := []string{"full", "resampling", "minimal"}
	if m.fmriLevelIndex > 0 {
		ab.args = append(ab.args, "--level", levelOptions[m.fmriLevelIndex%3])
	}

	ciftiOptions := []string{"", "91k", "170k"}
	if m.fmriCiftiOutputIndex > 0 {
		ab.args = append(ab.args, "--cifti-output", ciftiOptions[m.fmriCiftiOutputIndex%3])
	}

	ab.addIfNonEmpty("--task-id", m.fmriTaskId)

	// Performance
	if m.fmriNThreads > 0 {
		ab.args = append(ab.args, "--nthreads", fmt.Sprintf("%d", m.fmriNThreads))
	}
	if m.fmriOmpNThreads > 0 {
		ab.args = append(ab.args, "--omp-nthreads", fmt.Sprintf("%d", m.fmriOmpNThreads))
	}
	if m.fmriMemMb > 0 {
		ab.args = append(ab.args, "--mem-mb", fmt.Sprintf("%d", m.fmriMemMb))
	}
	if m.fmriLowMem {
		ab.args = append(ab.args, "--low-mem")
	}

	// Anatomical
	if m.fmriSkipReconstruction {
		ab.args = append(ab.args, "--fs-no-reconall")
	}
	if m.fmriLongitudinal {
		ab.args = append(ab.args, "--longitudinal")
	}
	if strings.TrimSpace(m.fmriSkullStripTemplate) != "" && m.fmriSkullStripTemplate != "OASIS30ANTs" {
		ab.args = append(ab.args, "--skull-strip-template", m.fmriSkullStripTemplate)
	}
	if m.fmriSkullStripFixedSeed {
		ab.args = append(ab.args, "--skull-strip-fixed-seed")
	}

	// BOLD processing
	bold2t1wInitOptions := []string{"register", "header"}
	if m.fmriBold2T1wInitIndex == 1 {
		ab.args = append(ab.args, "--bold2t1w-init", bold2t1wInitOptions[1])
	}
	if m.fmriBold2T1wDof != 6 {
		ab.args = append(ab.args, "--bold2t1w-dof", fmt.Sprintf("%d", m.fmriBold2T1wDof))
	}
	if m.fmriSliceTimeRef != 0.5 {
		ab.args = append(ab.args, "--slice-time-ref", fmt.Sprintf("%.2f", m.fmriSliceTimeRef))
	}
	if m.fmriDummyScans > 0 {
		ab.args = append(ab.args, "--dummy-scans", fmt.Sprintf("%d", m.fmriDummyScans))
	}

	// Quality control
	if m.fmriFdSpikeThreshold != 0.5 {
		ab.args = append(ab.args, "--fd-spike-threshold", fmt.Sprintf("%.2f", m.fmriFdSpikeThreshold))
	}
	if m.fmriDvarsSpikeThreshold != 1.5 {
		ab.args = append(ab.args, "--dvars-spike-threshold", fmt.Sprintf("%.2f", m.fmriDvarsSpikeThreshold))
	}

	// Denoising
	if m.fmriUseAroma {
		ab.args = append(ab.args, "--use-aroma")
	}

	// Surface
	if m.fmriMedialSurfaceNan {
		ab.args = append(ab.args, "--medial-surface-nan")
	}
	if m.fmriNoMsm {
		ab.args = append(ab.args, "--no-msm")
	}

	// Multi-echo
	if m.fmriMeOutputEchos {
		ab.args = append(ab.args, "--me-output-echos")
	}

	// Reproducibility
	if m.fmriRandomSeed > 0 {
		ab.args = append(ab.args, "--random-seed", fmt.Sprintf("%d", m.fmriRandomSeed))
	}

	// Validation
	if m.fmriSkipBidsValidation {
		ab.args = append(ab.args, "--skip-bids-validation")
	}
	if m.fmriStopOnFirstCrash {
		ab.args = append(ab.args, "--stop-on-first-crash")
	}
	if !m.fmriCleanWorkdir {
		ab.args = append(ab.args, "--no-clean-workdir")
	}

	// Advanced
	ab.addIfNonEmpty("--fmriprep-extra-args", m.fmriExtraArgs)

	return ab.build()
}

func (m Model) buildFmriAnalysisAdvancedArgs() []string {
	ab := newArgBuilder()

	mode := "first-level"
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	isFirstLevel := mode == "first-level"
	trialMethod := "beta-series"
	if m.fmriTrialSigMethodIndex%2 == 1 {
		trialMethod = "lss"
	}

	inputSource := "fmriprep"
	if m.fmriAnalysisInputSourceIndex%2 == 1 {
		inputSource = "bids_raw"
	}
	ab.args = append(ab.args, "--input-source", inputSource)

	if strings.TrimSpace(m.fmriAnalysisFmriprepSpace) != "" && strings.TrimSpace(m.fmriAnalysisFmriprepSpace) != "T1w" {
		ab.args = append(ab.args, "--fmriprep-space", strings.TrimSpace(m.fmriAnalysisFmriprepSpace))
	}

	if !m.fmriAnalysisRequireFmriprep {
		ab.args = append(ab.args, "--no-require-fmriprep")
	}

	// Runs: accept space-separated ints (e.g. "1 2 3")
	runsSpec := strings.TrimSpace(m.fmriAnalysisRunsSpec)
	if runsSpec != "" {
		runs := splitSpaceList(runsSpec)
		if len(runs) > 0 {
			ab.args = append(ab.args, "--runs")
			ab.args = append(ab.args, runs...)
		}
	}

	// Contrast
	ab.addIfNonEmpty("--contrast-name", strings.TrimSpace(m.fmriAnalysisContrastName))

	if isFirstLevel && m.fmriAnalysisContrastType%2 == 1 {
		ab.args = append(ab.args, "--contrast-type", "custom")
		ab.addIfNonEmpty("--formula", strings.TrimSpace(m.fmriAnalysisFormula))
	} else {
		ab.args = append(ab.args, "--contrast-type", "t-test")
		ab.addIfNonEmpty("--cond-a-column", strings.TrimSpace(m.fmriAnalysisCondAColumn))
		ab.addIfNonEmpty("--cond-a-value", strings.TrimSpace(m.fmriAnalysisCondAValue))
		ab.addIfNonEmpty("--cond-b-column", strings.TrimSpace(m.fmriAnalysisCondBColumn))
		ab.addIfNonEmpty("--cond-b-value", strings.TrimSpace(m.fmriAnalysisCondBValue))
	}

	// GLM
	hrfOptions := []string{"spm", "flobs", "fir"}
	ab.args = append(ab.args, "--hrf-model", hrfOptions[m.fmriAnalysisHrfModel%len(hrfOptions)])
	driftOptions := []string{"none", "cosine", "polynomial"}
	ab.args = append(ab.args, "--drift-model", driftOptions[m.fmriAnalysisDriftModel%len(driftOptions)])

	ab.args = append(ab.args, "--high-pass-hz", fmt.Sprintf("%.6f", m.fmriAnalysisHighPassHz))
	ab.args = append(ab.args, "--low-pass-hz", fmt.Sprintf("%.6f", m.fmriAnalysisLowPassHz))
	ab.args = append(ab.args, "--smoothing-fwhm", fmt.Sprintf("%.1f", m.fmriAnalysisSmoothingFwhm))

	// Confounds / QC
	confoundsOptions := []string{
		"auto",
		"none",
		"motion6",
		"motion12",
		"motion24",
		"motion24+wmcsf",
		"motion24+wmcsf+fd",
	}
	ab.args = append(ab.args, "--confounds-strategy", confoundsOptions[m.fmriAnalysisConfoundsStrategy%len(confoundsOptions)])
	if isFirstLevel {
		ab.addIfNonEmpty("--events-to-model", strings.TrimSpace(m.fmriAnalysisEventsToModel))
	}
	if isFirstLevel && m.fmriAnalysisWriteDesignMatrix {
		ab.args = append(ab.args, "--write-design-matrix")
	}

	// Output
	ab.addIfNonEmpty("--output-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisOutputDir)))

	if isFirstLevel {
		outTypeOptions := []string{"z-score", "t-stat", "cope", "beta"}
		ab.args = append(ab.args, "--output-type", outTypeOptions[m.fmriAnalysisOutputType%len(outTypeOptions)])

		if m.fmriAnalysisResampleToFS {
			ab.args = append(ab.args, "--resample-to-freesurfer")
			ab.addIfNonEmpty("--freesurfer-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisFreesurferDir)))
		}
	}

	// Trial-wise signatures (beta-series / lss)
	if !isFirstLevel {
		if !m.fmriTrialSigIncludeOtherEvents {
			ab.args = append(ab.args, "--no-include-other-events")
		}
		if m.fmriTrialSigMaxTrialsPerRun > 0 {
			ab.args = append(ab.args, "--max-trials-per-run", fmt.Sprintf("%d", m.fmriTrialSigMaxTrialsPerRun))
		}
		weighting := []string{"variance", "mean"}
		if m.fmriTrialSigFixedEffectsWeighting%len(weighting) != 0 {
			ab.args = append(ab.args, "--fixed-effects-weighting", weighting[m.fmriTrialSigFixedEffectsWeighting%len(weighting)])
		}
		if m.fmriTrialSigWriteTrialBetas {
			ab.args = append(ab.args, "--write-trial-betas")
		} else {
			ab.args = append(ab.args, "--no-write-trial-betas")
		}
		if m.fmriTrialSigWriteTrialVariances {
			ab.args = append(ab.args, "--write-trial-variances")
		} else {
			ab.args = append(ab.args, "--no-write-trial-variances")
		}
		if !m.fmriTrialSigWriteConditionBetas {
			ab.args = append(ab.args, "--no-write-condition-betas")
		}

		// Signatures: only emit flag if subset selected
		var sigs []string
		if m.fmriTrialSigSignatureNPS {
			sigs = append(sigs, "NPS")
		}
		if m.fmriTrialSigSignatureSIIPS1 {
			sigs = append(sigs, "SIIPS1")
		}
		if len(sigs) > 0 && len(sigs) < 2 {
			ab.args = append(ab.args, "--signatures")
			ab.args = append(ab.args, sigs...)
		}

		if trialMethod == "lss" && m.fmriTrialSigLssOtherRegressorsIndex%2 == 1 {
			ab.args = append(ab.args, "--lss-other-regressors", "all")
		}

		if strings.TrimSpace(m.fmriAnalysisSignatureDir) != "" {
			ab.args = append(ab.args, "--signature-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisSignatureDir)))
		}
		roiNames := strings.Fields(strings.TrimSpace(m.fmriTrialSigRoiNames))
		if len(roiNames) > 0 {
			ab.args = append(ab.args, "--signature-roi")
			ab.args = append(ab.args, roiNames...)
		}

		// Signature grouping (e.g., temperature levels)
		groupCol := strings.TrimSpace(m.fmriTrialSigGroupColumn)
		groupVals := splitSpaceList(strings.TrimSpace(m.fmriTrialSigGroupValuesSpec))
		if groupCol != "" && len(groupVals) > 0 {
			ab.args = append(ab.args, "--signature-group-column", groupCol)
			ab.args = append(ab.args, "--signature-group-values")
			ab.args = append(ab.args, groupVals...)
			if m.fmriTrialSigGroupScopeIndex%2 == 1 {
				ab.args = append(ab.args, "--signature-group-scope", "per-run")
			}
		}

		return ab.build()
	}

	// Plotting / Report (CLI defaults are off)
	if m.fmriAnalysisPlotsEnabled {
		ab.args = append(ab.args, "--plots")

		if m.fmriAnalysisPlotHTML {
			ab.args = append(ab.args, "--plot-html-report")
		}

		plotSpaceOptions := []string{"both", "native", "mni"}
		ab.args = append(ab.args, "--plot-space", plotSpaceOptions[m.fmriAnalysisPlotSpaceIndex%len(plotSpaceOptions)])

		thresholdModeOptions := []string{"z", "fdr", "none"}
		ab.args = append(ab.args, "--plot-threshold-mode", thresholdModeOptions[m.fmriAnalysisPlotThresholdModeIndex%len(thresholdModeOptions)])

		ab.args = append(ab.args, "--plot-z-threshold", fmt.Sprintf("%.2f", m.fmriAnalysisPlotZThreshold))
		if m.fmriAnalysisPlotThresholdModeIndex%3 == 1 { // fdr
			ab.args = append(ab.args, "--plot-fdr-q", fmt.Sprintf("%.3f", m.fmriAnalysisPlotFdrQ))
		}
		if m.fmriAnalysisPlotClusterMinVoxels > 0 {
			ab.args = append(ab.args, "--plot-cluster-min-voxels", fmt.Sprintf("%d", m.fmriAnalysisPlotClusterMinVoxels))
		}

		vmaxModeOptions := []string{"per-space-robust", "shared-robust", "manual"}
		ab.args = append(ab.args, "--plot-vmax-mode", vmaxModeOptions[m.fmriAnalysisPlotVmaxModeIndex%len(vmaxModeOptions)])
		if m.fmriAnalysisPlotVmaxModeIndex%3 == 2 { // manual
			ab.args = append(ab.args, "--plot-vmax", fmt.Sprintf("%.2f", m.fmriAnalysisPlotVmaxManual))
		}

		if !m.fmriAnalysisPlotIncludeUnthresholded {
			ab.args = append(ab.args, "--no-plot-include-unthresholded")
		}

		if !m.fmriAnalysisPlotEffectSize {
			ab.args = append(ab.args, "--plot-no-effect-size")
		}
		if !m.fmriAnalysisPlotStandardError {
			ab.args = append(ab.args, "--plot-no-standard-error")
		}
		if !m.fmriAnalysisPlotMotionQC {
			ab.args = append(ab.args, "--plot-no-motion-qc")
		}
		if !m.fmriAnalysisPlotCarpetQC {
			ab.args = append(ab.args, "--plot-no-carpet-qc")
		}
		if !m.fmriAnalysisPlotTSNRQC {
			ab.args = append(ab.args, "--plot-no-tsnr-qc")
		}
		if !m.fmriAnalysisPlotDesignQC {
			ab.args = append(ab.args, "--plot-no-design-qc")
		}
		if !m.fmriAnalysisPlotEmbedImages {
			ab.args = append(ab.args, "--plot-no-embed-images")
		}
		if !m.fmriAnalysisPlotSignatures {
			ab.args = append(ab.args, "--plot-no-signatures")
		} else if strings.TrimSpace(m.fmriAnalysisSignatureDir) != "" {
			ab.args = append(ab.args, "--signature-dir", expandUserPath(strings.TrimSpace(m.fmriAnalysisSignatureDir)))
		}

		// Formats: require at least one
		var formats []string
		if m.fmriAnalysisPlotFormatPNG {
			formats = append(formats, "png")
		}
		if m.fmriAnalysisPlotFormatSVG {
			formats = append(formats, "svg")
		}
		if len(formats) > 0 {
			ab.args = append(ab.args, "--plot-formats")
			ab.args = append(ab.args, formats...)
		}

		// Plot types: require at least one
		var plotTypes []string
		if m.fmriAnalysisPlotTypeSlices {
			plotTypes = append(plotTypes, "slices")
		}
		if m.fmriAnalysisPlotTypeGlass {
			plotTypes = append(plotTypes, "glass")
		}
		if m.fmriAnalysisPlotTypeHist {
			plotTypes = append(plotTypes, "hist")
		}
		if m.fmriAnalysisPlotTypeClusters {
			plotTypes = append(plotTypes, "clusters")
		}
		if len(plotTypes) > 0 {
			ab.args = append(ab.args, "--plot-types")
			ab.args = append(ab.args, plotTypes...)
		}
	}

	return ab.build()
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

func (m Model) buildFmriRawToBidsAdvancedArgs() []string {
	var args []string

	if strings.TrimSpace(m.fmriRawSession) != "" {
		args = append(args, "--session", strings.TrimSpace(m.fmriRawSession))
	}
	if strings.TrimSpace(m.fmriRawRestTask) != "" && strings.TrimSpace(m.fmriRawRestTask) != "rest" {
		args = append(args, "--rest-task", strings.TrimSpace(m.fmriRawRestTask))
	}
	if !m.fmriRawIncludeRest {
		args = append(args, "--no-rest")
	}
	if !m.fmriRawIncludeFieldmaps {
		args = append(args, "--no-fieldmaps")
	}

	dicomMode := "symlink"
	switch m.fmriRawDicomModeIndex {
	case 1:
		dicomMode = "copy"
	case 2:
		dicomMode = "skip"
	}
	if dicomMode != "symlink" {
		args = append(args, "--dicom-mode", dicomMode)
	}

	if m.fmriRawOverwrite {
		args = append(args, "--overwrite")
	}
	if !m.fmriRawCreateEvents {
		args = append(args, "--no-events")
	}

	granularity := "phases"
	if m.fmriRawEventGranularity == 1 {
		granularity = "trial"
	}
	if granularity != "phases" {
		args = append(args, "--event-granularity", granularity)
	}

	onsetRef := "as_is"
	switch m.fmriRawOnsetRefIndex {
	case 1:
		onsetRef = "first_iti_start"
	case 2:
		onsetRef = "first_stim_start"
	}
	if onsetRef != "as_is" {
		args = append(args, "--onset-reference", onsetRef)
	}
	if m.fmriRawOnsetOffsetS != 0 {
		args = append(args, "--onset-offset-s", fmt.Sprintf("%.3f", m.fmriRawOnsetOffsetS))
	}
	if strings.TrimSpace(m.fmriRawDcm2niixPath) != "" {
		args = append(args, "--dcm2niix-path", strings.TrimSpace(m.fmriRawDcm2niixPath))
	}
	if strings.TrimSpace(m.fmriRawDcm2niixArgs) != "" {
		for _, tok := range splitListInput(m.fmriRawDcm2niixArgs) {
			args = append(args, "--dcm2niix-arg", tok)
		}
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
