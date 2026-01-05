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
	hasAnyComputation := false

	// Primary computations
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) {
			result = append(result, m.computations[i].Key)
			hasAnyComputation = true
		}
	}
	// Post computations
	for i, sel := range m.postComputationSelected {
		if sel && i < len(m.postComputations) {
			result = append(result, m.postComputations[i].Key)
			hasAnyComputation = true
		}
	}

	// Auto-include trial_table if ANY computation is selected
	// (trial_table is the foundational data structure ALL analyses need)
	if hasAnyComputation {
		result = append(result, "trial_table")
	}

	sort.Strings(result)
	return result
}

// isComputationSelected checks if a specific computation is currently selected
func (m Model) isComputationSelected(computation string) bool {
	// Check primary computations
	for i, sel := range m.computationSelected {
		if sel && i < len(m.computations) && m.computations[i].Key == computation {
			return true
		}
	}
	// Check post computations
	for i, sel := range m.postComputationSelected {
		if sel && i < len(m.postComputations) && m.postComputations[i].Key == computation {
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
		types.PipelineDecoding, types.PipelinePlotting:
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
		case types.PipelinePlotting:
			parts = append(parts, m.buildPlottingAdvancedArgs()...)
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

func (m Model) buildPlottingAdvancedArgs() []string {
	var args []string

	if m.useDefaultAdvanced {
		return args
	}

	// Plot defaults / styling overrides (mirrors eeg_pipeline/cli/commands/plotting.py)
	if strings.TrimSpace(m.plotBboxInches) != "" {
		args = append(args, "--bbox-inches", strings.TrimSpace(m.plotBboxInches))
	}
	if m.plotPadInches != 0 {
		args = append(args, "--pad-inches", fmt.Sprintf("%.4f", m.plotPadInches))
	}

	// Fonts
	if strings.TrimSpace(m.plotFontFamily) != "" {
		args = append(args, "--font-family", strings.TrimSpace(m.plotFontFamily))
	}
	if strings.TrimSpace(m.plotFontWeight) != "" {
		args = append(args, "--font-weight", strings.TrimSpace(m.plotFontWeight))
	}
	if m.plotFontSizeSmall != 0 {
		args = append(args, "--font-size-small", fmt.Sprintf("%d", m.plotFontSizeSmall))
	}
	if m.plotFontSizeMedium != 0 {
		args = append(args, "--font-size-medium", fmt.Sprintf("%d", m.plotFontSizeMedium))
	}
	if m.plotFontSizeLarge != 0 {
		args = append(args, "--font-size-large", fmt.Sprintf("%d", m.plotFontSizeLarge))
	}
	if m.plotFontSizeTitle != 0 {
		args = append(args, "--font-size-title", fmt.Sprintf("%d", m.plotFontSizeTitle))
	}
	if m.plotFontSizeAnnotation != 0 {
		args = append(args, "--font-size-annotation", fmt.Sprintf("%d", m.plotFontSizeAnnotation))
	}
	if m.plotFontSizeLabel != 0 {
		args = append(args, "--font-size-label", fmt.Sprintf("%d", m.plotFontSizeLabel))
	}
	if m.plotFontSizeYLabel != 0 {
		args = append(args, "--font-size-ylabel", fmt.Sprintf("%d", m.plotFontSizeYLabel))
	}
	if m.plotFontSizeSuptitle != 0 {
		args = append(args, "--font-size-suptitle", fmt.Sprintf("%d", m.plotFontSizeSuptitle))
	}
	if m.plotFontSizeFigureTitle != 0 {
		args = append(args, "--font-size-figure-title", fmt.Sprintf("%d", m.plotFontSizeFigureTitle))
	}

	// Layout
	if strings.TrimSpace(m.plotLayoutTightRectSpec) != "" {
		vals := splitSpaceList(m.plotLayoutTightRectSpec)
		if len(vals) == 4 {
			args = append(args, "--layout-tight-rect")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotLayoutTightRectMicrostateSpec) != "" {
		vals := splitSpaceList(m.plotLayoutTightRectMicrostateSpec)
		if len(vals) == 4 {
			args = append(args, "--layout-tight-rect-microstate")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotGridSpecWidthRatiosSpec) != "" {
		args = append(args, "--gridspec-width-ratios")
		args = append(args, splitSpaceList(m.plotGridSpecWidthRatiosSpec)...)
	}
	if strings.TrimSpace(m.plotGridSpecHeightRatiosSpec) != "" {
		args = append(args, "--gridspec-height-ratios")
		args = append(args, splitSpaceList(m.plotGridSpecHeightRatiosSpec)...)
	}
	if m.plotGridSpecHspace != 0 {
		args = append(args, "--gridspec-hspace", fmt.Sprintf("%.4f", m.plotGridSpecHspace))
	}
	if m.plotGridSpecWspace != 0 {
		args = append(args, "--gridspec-wspace", fmt.Sprintf("%.4f", m.plotGridSpecWspace))
	}
	if m.plotGridSpecLeft != 0 {
		args = append(args, "--gridspec-left", fmt.Sprintf("%.4f", m.plotGridSpecLeft))
	}
	if m.plotGridSpecRight != 0 {
		args = append(args, "--gridspec-right", fmt.Sprintf("%.4f", m.plotGridSpecRight))
	}
	if m.plotGridSpecTop != 0 {
		args = append(args, "--gridspec-top", fmt.Sprintf("%.4f", m.plotGridSpecTop))
	}
	if m.plotGridSpecBottom != 0 {
		args = append(args, "--gridspec-bottom", fmt.Sprintf("%.4f", m.plotGridSpecBottom))
	}

	// Figure sizes
	if strings.TrimSpace(m.plotFigureSizeStandardSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeStandardSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-standard")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeMediumSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeMediumSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-medium")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeSmallSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeSmallSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-small")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeSquareSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeSquareSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-square")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeWideSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeWideSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-wide")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeTFRSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeTFRSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-tfr")
			args = append(args, vals...)
		}
	}
	if strings.TrimSpace(m.plotFigureSizeTopomapSpec) != "" {
		vals := splitSpaceList(m.plotFigureSizeTopomapSpec)
		if len(vals) == 2 {
			args = append(args, "--figure-size-topomap")
			args = append(args, vals...)
		}
	}

	// Colors
	if strings.TrimSpace(m.plotColorPain) != "" {
		args = append(args, "--color-pain", strings.TrimSpace(m.plotColorPain))
	}
	if strings.TrimSpace(m.plotColorNonpain) != "" {
		args = append(args, "--color-nonpain", strings.TrimSpace(m.plotColorNonpain))
	}
	if strings.TrimSpace(m.plotColorSignificant) != "" {
		args = append(args, "--color-significant", strings.TrimSpace(m.plotColorSignificant))
	}
	if strings.TrimSpace(m.plotColorNonsignificant) != "" {
		args = append(args, "--color-nonsignificant", strings.TrimSpace(m.plotColorNonsignificant))
	}
	if strings.TrimSpace(m.plotColorGray) != "" {
		args = append(args, "--color-gray", strings.TrimSpace(m.plotColorGray))
	}
	if strings.TrimSpace(m.plotColorLightGray) != "" {
		args = append(args, "--color-light-gray", strings.TrimSpace(m.plotColorLightGray))
	}
	if strings.TrimSpace(m.plotColorBlack) != "" {
		args = append(args, "--color-black", strings.TrimSpace(m.plotColorBlack))
	}
	if strings.TrimSpace(m.plotColorBlue) != "" {
		args = append(args, "--color-blue", strings.TrimSpace(m.plotColorBlue))
	}
	if strings.TrimSpace(m.plotColorRed) != "" {
		args = append(args, "--color-red", strings.TrimSpace(m.plotColorRed))
	}
	if strings.TrimSpace(m.plotColorNetworkNode) != "" {
		args = append(args, "--color-network-node", strings.TrimSpace(m.plotColorNetworkNode))
	}

	// Alpha
	if m.plotAlphaGrid != 0 {
		args = append(args, "--alpha-grid", fmt.Sprintf("%.4f", m.plotAlphaGrid))
	}
	if m.plotAlphaFill != 0 {
		args = append(args, "--alpha-fill", fmt.Sprintf("%.4f", m.plotAlphaFill))
	}
	if m.plotAlphaCI != 0 {
		args = append(args, "--alpha-ci", fmt.Sprintf("%.4f", m.plotAlphaCI))
	}
	if m.plotAlphaCILine != 0 {
		args = append(args, "--alpha-ci-line", fmt.Sprintf("%.4f", m.plotAlphaCILine))
	}
	if m.plotAlphaTextBox != 0 {
		args = append(args, "--alpha-text-box", fmt.Sprintf("%.4f", m.plotAlphaTextBox))
	}
	if m.plotAlphaViolinBody != 0 {
		args = append(args, "--alpha-violin-body", fmt.Sprintf("%.4f", m.plotAlphaViolinBody))
	}
	if m.plotAlphaRidgeFill != 0 {
		args = append(args, "--alpha-ridge-fill", fmt.Sprintf("%.4f", m.plotAlphaRidgeFill))
	}

	// Scatter
	if m.plotScatterMarkerSizeSmall != 0 {
		args = append(args, "--scatter-marker-size-small", fmt.Sprintf("%d", m.plotScatterMarkerSizeSmall))
	}
	if m.plotScatterMarkerSizeLarge != 0 {
		args = append(args, "--scatter-marker-size-large", fmt.Sprintf("%d", m.plotScatterMarkerSizeLarge))
	}
	if m.plotScatterMarkerSizeDefault != 0 {
		args = append(args, "--scatter-marker-size-default", fmt.Sprintf("%d", m.plotScatterMarkerSizeDefault))
	}
	if m.plotScatterAlpha != 0 {
		args = append(args, "--scatter-alpha", fmt.Sprintf("%.4f", m.plotScatterAlpha))
	}
	if strings.TrimSpace(m.plotScatterEdgeColor) != "" {
		args = append(args, "--scatter-edgecolor", strings.TrimSpace(m.plotScatterEdgeColor))
	}
	if m.plotScatterEdgeWidth != 0 {
		args = append(args, "--scatter-edgewidth", fmt.Sprintf("%.4f", m.plotScatterEdgeWidth))
	}

	// Bar
	if m.plotBarAlpha != 0 {
		args = append(args, "--bar-alpha", fmt.Sprintf("%.4f", m.plotBarAlpha))
	}
	if m.plotBarWidth != 0 {
		args = append(args, "--bar-width", fmt.Sprintf("%.4f", m.plotBarWidth))
	}
	if m.plotBarCapsize != 0 {
		args = append(args, "--bar-capsize", fmt.Sprintf("%d", m.plotBarCapsize))
	}
	if m.plotBarCapsizeLarge != 0 {
		args = append(args, "--bar-capsize-large", fmt.Sprintf("%d", m.plotBarCapsizeLarge))
	}

	// Line
	if m.plotLineWidthThin != 0 {
		args = append(args, "--line-width-thin", fmt.Sprintf("%.4f", m.plotLineWidthThin))
	}
	if m.plotLineWidthStandard != 0 {
		args = append(args, "--line-width-standard", fmt.Sprintf("%.4f", m.plotLineWidthStandard))
	}
	if m.plotLineWidthThick != 0 {
		args = append(args, "--line-width-thick", fmt.Sprintf("%.4f", m.plotLineWidthThick))
	}
	if m.plotLineWidthBold != 0 {
		args = append(args, "--line-width-bold", fmt.Sprintf("%.4f", m.plotLineWidthBold))
	}
	if m.plotLineAlphaStandard != 0 {
		args = append(args, "--line-alpha-standard", fmt.Sprintf("%.4f", m.plotLineAlphaStandard))
	}
	if m.plotLineAlphaDim != 0 {
		args = append(args, "--line-alpha-dim", fmt.Sprintf("%.4f", m.plotLineAlphaDim))
	}
	if m.plotLineAlphaZeroLine != 0 {
		args = append(args, "--line-alpha-zero-line", fmt.Sprintf("%.4f", m.plotLineAlphaZeroLine))
	}
	if m.plotLineAlphaFitLine != 0 {
		args = append(args, "--line-alpha-fit-line", fmt.Sprintf("%.4f", m.plotLineAlphaFitLine))
	}
	if m.plotLineAlphaDiagonal != 0 {
		args = append(args, "--line-alpha-diagonal", fmt.Sprintf("%.4f", m.plotLineAlphaDiagonal))
	}
	if m.plotLineAlphaReference != 0 {
		args = append(args, "--line-alpha-reference", fmt.Sprintf("%.4f", m.plotLineAlphaReference))
	}
	if m.plotLineRegressionWidth != 0 {
		args = append(args, "--line-regression-width", fmt.Sprintf("%.4f", m.plotLineRegressionWidth))
	}
	if m.plotLineResidualWidth != 0 {
		args = append(args, "--line-residual-width", fmt.Sprintf("%.4f", m.plotLineResidualWidth))
	}
	if m.plotLineQQWidth != 0 {
		args = append(args, "--line-qq-width", fmt.Sprintf("%.4f", m.plotLineQQWidth))
	}

	// Histogram
	if m.plotHistBins != 0 {
		args = append(args, "--hist-bins", fmt.Sprintf("%d", m.plotHistBins))
	}
	if m.plotHistBinsBehavioral != 0 {
		args = append(args, "--hist-bins-behavioral", fmt.Sprintf("%d", m.plotHistBinsBehavioral))
	}
	if m.plotHistBinsResidual != 0 {
		args = append(args, "--hist-bins-residual", fmt.Sprintf("%d", m.plotHistBinsResidual))
	}
	if m.plotHistBinsTFR != 0 {
		args = append(args, "--hist-bins-tfr", fmt.Sprintf("%d", m.plotHistBinsTFR))
	}
	if strings.TrimSpace(m.plotHistEdgeColor) != "" {
		args = append(args, "--hist-edgecolor", strings.TrimSpace(m.plotHistEdgeColor))
	}
	if m.plotHistEdgeWidth != 0 {
		args = append(args, "--hist-edgewidth", fmt.Sprintf("%.4f", m.plotHistEdgeWidth))
	}
	if m.plotHistAlpha != 0 {
		args = append(args, "--hist-alpha", fmt.Sprintf("%.4f", m.plotHistAlpha))
	}
	if m.plotHistAlphaResidual != 0 {
		args = append(args, "--hist-alpha-residual", fmt.Sprintf("%.4f", m.plotHistAlphaResidual))
	}
	if m.plotHistAlphaTFR != 0 {
		args = append(args, "--hist-alpha-tfr", fmt.Sprintf("%.4f", m.plotHistAlphaTFR))
	}

	// KDE
	if m.plotKdePoints != 0 {
		args = append(args, "--kde-points", fmt.Sprintf("%d", m.plotKdePoints))
	}
	if strings.TrimSpace(m.plotKdeColor) != "" {
		args = append(args, "--kde-color", strings.TrimSpace(m.plotKdeColor))
	}
	if m.plotKdeLinewidth != 0 {
		args = append(args, "--kde-linewidth", fmt.Sprintf("%.4f", m.plotKdeLinewidth))
	}
	if m.plotKdeAlpha != 0 {
		args = append(args, "--kde-alpha", fmt.Sprintf("%.4f", m.plotKdeAlpha))
	}

	// Errorbar
	if m.plotErrorbarMarkerSize != 0 {
		args = append(args, "--errorbar-markersize", fmt.Sprintf("%d", m.plotErrorbarMarkerSize))
	}
	if m.plotErrorbarCapsize != 0 {
		args = append(args, "--errorbar-capsize", fmt.Sprintf("%d", m.plotErrorbarCapsize))
	}
	if m.plotErrorbarCapsizeLarge != 0 {
		args = append(args, "--errorbar-capsize-large", fmt.Sprintf("%d", m.plotErrorbarCapsizeLarge))
	}

	// Text positions
	if m.plotTextStatsX != 0 {
		args = append(args, "--text-stats-x", fmt.Sprintf("%.4f", m.plotTextStatsX))
	}
	if m.plotTextStatsY != 0 {
		args = append(args, "--text-stats-y", fmt.Sprintf("%.4f", m.plotTextStatsY))
	}
	if m.plotTextPvalueX != 0 {
		args = append(args, "--text-pvalue-x", fmt.Sprintf("%.4f", m.plotTextPvalueX))
	}
	if m.plotTextPvalueY != 0 {
		args = append(args, "--text-pvalue-y", fmt.Sprintf("%.4f", m.plotTextPvalueY))
	}
	if m.plotTextBootstrapX != 0 {
		args = append(args, "--text-bootstrap-x", fmt.Sprintf("%.4f", m.plotTextBootstrapX))
	}
	if m.plotTextBootstrapY != 0 {
		args = append(args, "--text-bootstrap-y", fmt.Sprintf("%.4f", m.plotTextBootstrapY))
	}
	if m.plotTextChannelAnnotationX != 0 {
		args = append(args, "--text-channel-annotation-x", fmt.Sprintf("%.4f", m.plotTextChannelAnnotationX))
	}
	if m.plotTextChannelAnnotationY != 0 {
		args = append(args, "--text-channel-annotation-y", fmt.Sprintf("%.4f", m.plotTextChannelAnnotationY))
	}
	if m.plotTextTitleY != 0 {
		args = append(args, "--text-title-y", fmt.Sprintf("%.4f", m.plotTextTitleY))
	}
	if m.plotTextResidualQcTitleY != 0 {
		args = append(args, "--text-residual-qc-title-y", fmt.Sprintf("%.4f", m.plotTextResidualQcTitleY))
	}

	// Validation
	if m.plotValidationMinSamplesForPlot != 0 {
		args = append(args, "--validation-min-samples-for-plot", fmt.Sprintf("%d", m.plotValidationMinSamplesForPlot))
	}
	if m.plotValidationMinSamplesForKDE != 0 {
		args = append(args, "--validation-min-samples-for-kde", fmt.Sprintf("%d", m.plotValidationMinSamplesForKDE))
	}
	if m.plotValidationMinSamplesForFit != 0 {
		args = append(args, "--validation-min-samples-for-fit", fmt.Sprintf("%d", m.plotValidationMinSamplesForFit))
	}
	if m.plotValidationMinSamplesForCalibration != 0 {
		args = append(args, "--validation-min-samples-for-calibration", fmt.Sprintf("%d", m.plotValidationMinSamplesForCalibration))
	}
	if m.plotValidationMinBinsForCalibration != 0 {
		args = append(args, "--validation-min-bins-for-calibration", fmt.Sprintf("%d", m.plotValidationMinBinsForCalibration))
	}
	if m.plotValidationMaxBinsForCalibration != 0 {
		args = append(args, "--validation-max-bins-for-calibration", fmt.Sprintf("%d", m.plotValidationMaxBinsForCalibration))
	}
	if m.plotValidationSamplesPerBin != 0 {
		args = append(args, "--validation-samples-per-bin", fmt.Sprintf("%d", m.plotValidationSamplesPerBin))
	}
	if m.plotValidationMinRoisForFDR != 0 {
		args = append(args, "--validation-min-rois-for-fdr", fmt.Sprintf("%d", m.plotValidationMinRoisForFDR))
	}
	if m.plotValidationMinPvaluesForFDR != 0 {
		args = append(args, "--validation-min-pvalues-for-fdr", fmt.Sprintf("%d", m.plotValidationMinPvaluesForFDR))
	}

	// Topomap controls
	if m.plotTopomapContours > 0 {
		args = append(args, "--topomap-contours", fmt.Sprintf("%d", m.plotTopomapContours))
	}
	if strings.TrimSpace(m.plotTopomapColormap) != "" {
		args = append(args, "--topomap-colormap", m.plotTopomapColormap)
	}
	if m.plotTopomapColorbarFraction != 0 {
		args = append(args, "--topomap-colorbar-fraction", fmt.Sprintf("%.4f", m.plotTopomapColorbarFraction))
	}
	if m.plotTopomapColorbarPad != 0 {
		args = append(args, "--topomap-colorbar-pad", fmt.Sprintf("%.4f", m.plotTopomapColorbarPad))
	}
	if m.plotTopomapDiffAnnotation != nil {
		if *m.plotTopomapDiffAnnotation {
			args = append(args, "--topomap-diff-annotation-enabled")
		} else {
			args = append(args, "--no-topomap-diff-annotation-enabled")
		}
	}
	if m.plotTopomapAnnotateDesc != nil {
		if *m.plotTopomapAnnotateDesc {
			args = append(args, "--topomap-annotate-descriptive")
		} else {
			args = append(args, "--no-topomap-annotate-descriptive")
		}
	}
	if strings.TrimSpace(m.plotTopomapSigMaskMarker) != "" {
		args = append(args, "--topomap-sig-mask-marker", strings.TrimSpace(m.plotTopomapSigMaskMarker))
	}
	if strings.TrimSpace(m.plotTopomapSigMaskMarkerFaceColor) != "" {
		args = append(args, "--topomap-sig-mask-markerfacecolor", strings.TrimSpace(m.plotTopomapSigMaskMarkerFaceColor))
	}
	if strings.TrimSpace(m.plotTopomapSigMaskMarkerEdgeColor) != "" {
		args = append(args, "--topomap-sig-mask-markeredgecolor", strings.TrimSpace(m.plotTopomapSigMaskMarkerEdgeColor))
	}
	if m.plotTopomapSigMaskLinewidth != 0 {
		args = append(args, "--topomap-sig-mask-linewidth", fmt.Sprintf("%.4f", m.plotTopomapSigMaskLinewidth))
	}
	if m.plotTopomapSigMaskMarkerSize != 0 {
		args = append(args, "--topomap-sig-mask-markersize", fmt.Sprintf("%.4f", m.plotTopomapSigMaskMarkerSize))
	}

	// TFR controls
	if m.plotTFRLogBase != 0 {
		args = append(args, "--tfr-log-base", fmt.Sprintf("%.4f", m.plotTFRLogBase))
	}
	if m.plotTFRPercentageMultiplier != 0 {
		args = append(args, "--tfr-percentage-multiplier", fmt.Sprintf("%.4f", m.plotTFRPercentageMultiplier))
	}

	// Sizing controls
	if m.plotRoiWidthPerBand != 0 {
		args = append(args, "--roi-width-per-band", fmt.Sprintf("%.4f", m.plotRoiWidthPerBand))
	}
	if m.plotRoiWidthPerMetric != 0 {
		args = append(args, "--roi-width-per-metric", fmt.Sprintf("%.4f", m.plotRoiWidthPerMetric))
	}
	if m.plotRoiHeightPerRoi != 0 {
		args = append(args, "--roi-height-per-roi", fmt.Sprintf("%.4f", m.plotRoiHeightPerRoi))
	}

	if m.plotPowerWidthPerBand != 0 {
		args = append(args, "--power-width-per-band", fmt.Sprintf("%.4f", m.plotPowerWidthPerBand))
	}
	if m.plotPowerHeightPerSegment != 0 {
		args = append(args, "--power-height-per-segment", fmt.Sprintf("%.4f", m.plotPowerHeightPerSegment))
	}

	if m.plotItpcWidthPerBin != 0 {
		args = append(args, "--itpc-width-per-bin", fmt.Sprintf("%.4f", m.plotItpcWidthPerBin))
	}
	if m.plotItpcHeightPerBand != 0 {
		args = append(args, "--itpc-height-per-band", fmt.Sprintf("%.4f", m.plotItpcHeightPerBand))
	}
	if m.plotItpcWidthPerBandBox != 0 {
		args = append(args, "--itpc-width-per-band-box", fmt.Sprintf("%.4f", m.plotItpcWidthPerBandBox))
	}
	if m.plotItpcHeightBox != 0 {
		args = append(args, "--itpc-height-box", fmt.Sprintf("%.4f", m.plotItpcHeightBox))
	}

	if strings.TrimSpace(m.plotPacCmap) != "" {
		args = append(args, "--pac-cmap", m.plotPacCmap)
	}
	if m.plotPacWidthPerRoi != 0 {
		args = append(args, "--pac-width-per-roi", fmt.Sprintf("%.4f", m.plotPacWidthPerRoi))
	}
	if m.plotPacHeightBox != 0 {
		args = append(args, "--pac-height-box", fmt.Sprintf("%.4f", m.plotPacHeightBox))
	}

	if m.plotAperiodicWidthPerColumn != 0 {
		args = append(args, "--aperiodic-width-per-column", fmt.Sprintf("%.4f", m.plotAperiodicWidthPerColumn))
	}
	if m.plotAperiodicHeightPerRow != 0 {
		args = append(args, "--aperiodic-height-per-row", fmt.Sprintf("%.4f", m.plotAperiodicHeightPerRow))
	}
	if m.plotAperiodicNPerm != 0 {
		args = append(args, "--aperiodic-n-perm", fmt.Sprintf("%d", m.plotAperiodicNPerm))
	}

	if m.plotQualityWidthPerPlot != 0 {
		args = append(args, "--quality-width-per-plot", fmt.Sprintf("%.4f", m.plotQualityWidthPerPlot))
	}
	if m.plotQualityHeightPerPlot != 0 {
		args = append(args, "--quality-height-per-plot", fmt.Sprintf("%.4f", m.plotQualityHeightPerPlot))
	}
	if m.plotQualityDistributionNCols != 0 {
		args = append(args, "--quality-distribution-n-cols", fmt.Sprintf("%d", m.plotQualityDistributionNCols))
	}
	if m.plotQualityDistributionMaxFeatures != 0 {
		args = append(args, "--quality-distribution-max-features", fmt.Sprintf("%d", m.plotQualityDistributionMaxFeatures))
	}
	if m.plotQualityOutlierZThreshold != 0 {
		args = append(args, "--quality-outlier-z-threshold", fmt.Sprintf("%.4f", m.plotQualityOutlierZThreshold))
	}
	if m.plotQualityOutlierMaxFeatures != 0 {
		args = append(args, "--quality-outlier-max-features", fmt.Sprintf("%d", m.plotQualityOutlierMaxFeatures))
	}
	if m.plotQualityOutlierMaxTrials != 0 {
		args = append(args, "--quality-outlier-max-trials", fmt.Sprintf("%d", m.plotQualityOutlierMaxTrials))
	}
	if m.plotQualitySnrThresholdDb != 0 {
		args = append(args, "--quality-snr-threshold-db", fmt.Sprintf("%.4f", m.plotQualitySnrThresholdDb))
	}

	if m.plotComplexityWidthPerMeasure != 0 {
		args = append(args, "--complexity-width-per-measure", fmt.Sprintf("%.4f", m.plotComplexityWidthPerMeasure))
	}
	if m.plotComplexityHeightPerSegment != 0 {
		args = append(args, "--complexity-height-per-segment", fmt.Sprintf("%.4f", m.plotComplexityHeightPerSegment))
	}

	if m.plotConnectivityWidthPerCircle != 0 {
		args = append(args, "--connectivity-width-per-circle", fmt.Sprintf("%.4f", m.plotConnectivityWidthPerCircle))
	}
	if m.plotConnectivityWidthPerBand != 0 {
		args = append(args, "--connectivity-width-per-band", fmt.Sprintf("%.4f", m.plotConnectivityWidthPerBand))
	}
	if m.plotConnectivityHeightPerMeasure != 0 {
		args = append(args, "--connectivity-height-per-measure", fmt.Sprintf("%.4f", m.plotConnectivityHeightPerMeasure))
	}
	if m.plotConnectivityCircleTopFraction != 0 {
		args = append(args, "--connectivity-circle-top-fraction", fmt.Sprintf("%.4f", m.plotConnectivityCircleTopFraction))
	}
	if m.plotConnectivityCircleMinLines != 0 {
		args = append(args, "--connectivity-circle-min-lines", fmt.Sprintf("%d", m.plotConnectivityCircleMinLines))
	}

	// Selection overrides
	if strings.TrimSpace(m.plotPacPairsSpec) != "" {
		args = append(args, "--pac-pairs")
		args = append(args, splitSpaceList(m.plotPacPairsSpec)...)
	}

	measures := m.selectedConnectivityMeasures()
	if len(measures) > 0 {
		args = append(args, "--connectivity-measures")
		args = append(args, measures...)
	}

	if strings.TrimSpace(m.plotSpectralMetricsSpec) != "" {
		args = append(args, "--spectral-metrics")
		args = append(args, splitSpaceList(m.plotSpectralMetricsSpec)...)
	}
	if strings.TrimSpace(m.plotBurstsMetricsSpec) != "" {
		args = append(args, "--bursts-metrics")
		args = append(args, splitSpaceList(m.plotBurstsMetricsSpec)...)
	}
	if strings.TrimSpace(m.plotAsymmetryStatSpec) != "" {
		args = append(args, "--asymmetry-stat", strings.TrimSpace(m.plotAsymmetryStatSpec))
	}
	if strings.TrimSpace(m.plotTemporalTimeBinsSpec) != "" {
		args = append(args, "--temporal-time-bins")
		args = append(args, splitSpaceList(m.plotTemporalTimeBinsSpec)...)
	}
	if strings.TrimSpace(m.plotTemporalTimeLabelsSpec) != "" {
		args = append(args, "--temporal-time-labels")
		args = append(args, splitSpaceList(m.plotTemporalTimeLabelsSpec)...)
	}

	// Per-plot overrides
	args = append(args, m.buildPlotItemConfigArgs()...)

	return args
}

func (m Model) buildPlotItemConfigArgs() []string {
	var args []string
	plotIDs := m.SelectedPlotIDs()
	for _, plotID := range plotIDs {
		cfg, ok := m.plotItemConfigs[plotID]
		if !ok {
			continue
		}

		if strings.TrimSpace(cfg.TfrDefaultBaselineWindowSpec) != "" {
			vals := splitSpaceList(cfg.TfrDefaultBaselineWindowSpec)
			if len(vals) == 2 {
				args = append(args, "--plot-item-config", plotID, "tfr_default_baseline_window")
				args = append(args, vals...)
			}
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
		if strings.TrimSpace(cfg.ComparisonROIsSpec) != "" {
			args = append(args, "--plot-item-config", plotID, "comparison_rois")
			args = append(args, splitSpaceList(cfg.ComparisonROIsSpec)...)
		}
	}
	return args
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
	if m.tfrDecim != 4 {
		args = append(args, "--tfr-decim", fmt.Sprintf("%d", m.tfrDecim))
	}
	if m.tfrWorkers != -1 {
		args = append(args, "--tfr-workers", fmt.Sprintf("%d", m.tfrWorkers))
	}

	// Generic & Validation

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

func splitSpaceList(raw string) []string {
	out := strings.Fields(raw)
	if len(out) == 0 {
		return nil
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

	if m.decodingMinTrialsInner != 3 {
		args = append(args, "--min-trials-inner", fmt.Sprintf("%d", m.decodingMinTrialsInner))
	}

	if m.rngSeed > 0 {
		args = append(args, "--rng-seed", fmt.Sprintf("%d", m.rngSeed))
	}

	if m.skipTimeGen {
		args = append(args, "--skip-time-gen")
	}

	// ElasticNet hyperparameters
	if strings.TrimSpace(m.elasticNetAlphaGrid) != "" && m.elasticNetAlphaGrid != "0.001,0.01,0.1,1,10" {
		args = append(args, "--elasticnet-alpha-grid")
		args = append(args, splitCSVList(m.elasticNetAlphaGrid)...)
	}
	if strings.TrimSpace(m.elasticNetL1RatioGrid) != "" && m.elasticNetL1RatioGrid != "0.2,0.5,0.8" {
		args = append(args, "--elasticnet-l1-ratio-grid")
		args = append(args, splitCSVList(m.elasticNetL1RatioGrid)...)
	}

	// Random Forest hyperparameters
	if m.rfNEstimators != 500 {
		args = append(args, "--rf-n-estimators", fmt.Sprintf("%d", m.rfNEstimators))
	}
	if strings.TrimSpace(m.rfMaxDepthGrid) != "" && m.rfMaxDepthGrid != "5,10,20,null" {
		args = append(args, "--rf-max-depth-grid")
		args = append(args, splitCSVList(m.rfMaxDepthGrid)...)
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
	if strings.TrimSpace(m.icaLabelsToKeep) != "" && m.icaLabelsToKeep != "brain,other" {
		args = append(args, "--ica-labels-to-keep")
		args = append(args, splitCSVList(m.icaLabelsToKeep)...)
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
