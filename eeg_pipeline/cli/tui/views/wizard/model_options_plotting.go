package wizard

// Plotting advanced row builders (per-plot rows + global styling groups).

func (m Model) getPlottingAdvancedRows() []plottingAdvancedRow {
	// When using defaults, keep the list to a single actionable row so
	// navigation matches the minimal renderer.
	if m.useDefaultAdvanced {
		return []plottingAdvancedRow{{kind: plottingRowOption, opt: optUseDefaults}}
	}

	rows := make([]plottingAdvancedRow, 0, 128)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: optUseDefaults})
	rows = append(rows, plottingAdvancedRow{kind: plottingRowOption, opt: optConfigSetOverrides})

	// Only show per-plot configs for selected plots
	selectedPlots := m.selectedPlotItemsForConfig()
	if len(selectedPlots) == 0 {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, label: "No plots selected."})
		return rows
	}

	// Per-plot configs only (global styling moved to categories page)
	rows = append(rows, plottingAdvancedRow{kind: plottingRowSection, label: "Plot-Specific Settings"})
	for _, plot := range selectedPlots {
		rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotHeader, plotID: plot.ID})

		fields := m.plotConfigFields(plot)
		if len(fields) == 0 {
			if m.plotItemConfigExpanded[plot.ID] {
				rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotInfo, plotID: plot.ID, label: "No plot-specific settings."})
			}
			continue
		}
		if !m.plotItemConfigExpanded[plot.ID] {
			continue
		}
		for _, f := range fields {
			rows = append(rows, plottingAdvancedRow{kind: plottingRowPlotField, plotID: plot.ID, plotField: f})
		}
	}

	return rows
}

func (m Model) getGlobalStylingOptions() []optionType {
	// Truly global styling options that apply to ALL plots
	options := []optionType{}

	// Defaults & Output
	options = append(options, optPlotGroupDefaults)
	if m.plotGroupDefaultsExpanded {
		options = append(options, optPlotBboxInches, optPlotPadInches)
	}

	// Fonts
	options = append(options, optPlotGroupFonts)
	if m.plotGroupFontsExpanded {
		options = append(options,
			optPlotFontFamily,
			optPlotFontWeight,
			optPlotFontSizeSmall,
			optPlotFontSizeMedium,
			optPlotFontSizeLarge,
			optPlotFontSizeTitle,
			optPlotFontSizeAnnotation,
			optPlotFontSizeLabel,
			optPlotFontSizeYLabel,
			optPlotFontSizeSuptitle,
			optPlotFontSizeFigureTitle,
		)
	}

	// Layout
	options = append(options, optPlotGroupLayout)
	if m.plotGroupLayoutExpanded {
		options = append(options,
			optPlotLayoutTightRect,
			optPlotLayoutTightRectMicrostate,
			optPlotGridSpecWidthRatios,
			optPlotGridSpecHeightRatios,
			optPlotGridSpecHspace,
			optPlotGridSpecWspace,
			optPlotGridSpecLeft,
			optPlotGridSpecRight,
			optPlotGridSpecTop,
			optPlotGridSpecBottom,
		)
	}

	// Figure Sizes
	options = append(options, optPlotGroupFigureSizes)
	if m.plotGroupFigureSizesExpanded {
		options = append(options,
			optPlotFigureSizeStandard,
			optPlotFigureSizeMedium,
			optPlotFigureSizeSmall,
			optPlotFigureSizeSquare,
			optPlotFigureSizeWide,
			optPlotFigureSizeTFR,
			optPlotFigureSizeTopomap,
		)
	}

	// Colors
	options = append(options, optPlotGroupColors)
	if m.plotGroupColorsExpanded {
		options = append(options,
			optPlotColorCondB,
			optPlotColorCondA,
			optPlotColorSignificant,
			optPlotColorNonsignificant,
			optPlotColorGray,
			optPlotColorLightGray,
			optPlotColorBlack,
			optPlotColorBlue,
			optPlotColorRed,
			optPlotColorNetworkNode,
		)
	}

	// Alpha
	options = append(options, optPlotGroupAlpha)
	if m.plotGroupAlphaExpanded {
		options = append(options,
			optPlotAlphaGrid,
			optPlotAlphaFill,
			optPlotAlphaCI,
			optPlotAlphaCILine,
			optPlotAlphaTextBox,
			optPlotAlphaViolinBody,
			optPlotAlphaRidgeFill,
		)
	}

	// Topomap
	options = append(options, optPlotGroupTopomap)
	if m.plotGroupTopomapExpanded {
		options = append(options,
			optPlotTopomapContours,
			optPlotTopomapColormap,
			optPlotTopomapColorbarFraction,
			optPlotTopomapColorbarPad,
			optPlotTopomapDiffAnnotation,
			optPlotTopomapAnnotateDescriptive,
			optPlotTopomapSigMaskMarker,
			optPlotTopomapSigMaskMarkerFaceColor,
			optPlotTopomapSigMaskMarkerEdgeColor,
			optPlotTopomapSigMaskLinewidth,
			optPlotTopomapSigMaskMarkersize,
		)
	}

	// TFR
	options = append(options, optPlotGroupTFR)
	if m.plotGroupTFRExpanded {
		options = append(options,
			optPlotTFRLogBase,
			optPlotTFRPercentageMultiplier,
			optPlotTFRTopomapWindowSizeMs,
			optPlotTFRTopomapWindowCount,
			optPlotTFRTopomapLabelXPosition,
			optPlotTFRTopomapLabelYPositionBottom,
			optPlotTFRTopomapLabelYPosition,
			optPlotTFRTopomapTitleY,
			optPlotTFRTopomapTitlePad,
			optPlotTFRTopomapSubplotsRight,
			optPlotTFRTopomapTemporalHspace,
			optPlotTFRTopomapTemporalWspace,
		)
	}

	// Source Localization
	options = append(options, optPlotGroupSourceLoc)
	if m.plotGroupSourceLocExpanded {
		options = append(options,
			optPlotSourceHemi,
			optPlotSourceViews,
			optPlotSourceCortex,
			optPlotSourceSubjectsDir,
		)
	}

	return options
}
