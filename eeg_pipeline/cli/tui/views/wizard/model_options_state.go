package wizard

import "github.com/eeg-pipeline/tui/types"

// Plot config options and active-edit cursor state helpers.

func (m Model) getPlotConfigOptions() []optionType {
	options := []optionType{
		optPlotPNG,
		optPlotSVG,
		optPlotPDF,
		optPlotDPI,
		optPlotSaveDPI,
		optPlotOverwrite,
	}

	// Dynamic options based on selected plots/categories
	if m.IsPlotCategorySelected("tfr") || m.IsPlotCategorySelected("features") {
		// ITPC and PAC settings
		options = append(options, optPlotSharedColorbar)
	}

	return options
}

func (m Model) isCurrentlyEditing(opt optionType) bool {
	if !m.editingNumber {
		return false
	}

	// Plotting advanced config uses a mixed row model (per-plot + global options),
	// so the cursor no longer indexes directly into getPlottingOptions().
	if m.Pipeline == types.PipelinePlotting {
		rows := m.getPlottingAdvancedRows()
		if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
			return false
		}
		return rows[m.advancedCursor].kind == plottingRowOption && rows[m.advancedCursor].opt == opt
	}

	var options []optionType
	switch m.Pipeline {
	case types.PipelineFeatures:
		options = m.getFeaturesOptions()
	case types.PipelineBehavior:
		options = m.getBehaviorOptions()
	case types.PipelineML:
		options = m.getMLOptions()
	case types.PipelinePreprocessing:
		options = m.getPreprocessingOptions()
	case types.PipelineFmri:
		options = m.getFmriPreprocessingOptions()
	case types.PipelineFmriAnalysis:
		options = m.getFmriAnalysisOptions()
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
