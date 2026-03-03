package wizard

// Per-plot option-field selection helpers for plotting advanced config.

func (m Model) selectedPlotItemsForConfig() []PlotItem {
	items := make([]PlotItem, 0, len(m.plotItems))
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] {
			continue
		}
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		items = append(items, plot)
	}
	for _, p := range m.featurePlotterItems() {
		if m.featurePlotterSelected[p.ID] {
			items = append(items, PlotItem{ID: p.ID, Group: p.Category, Name: p.Name})
		}
	}
	return items
}

func (m Model) plotSupportsComparisons(plot PlotItem) bool {
	_, ok := comparisonCapablePlotIDs[plot.ID]
	return ok
}

func (m Model) plotConfigFields(plot PlotItem) []plotItemConfigField {
	fields := make([]plotItemConfigField, 0, 16)
	fields = append(fields, preComparisonFieldsByPlotID[plot.ID]...)

	if profile, ok := comparisonFieldProfilesByPlotID[plot.ID]; ok {
		fields = append(fields, profile...)
	} else if m.plotSupportsComparisons(plot) {
		// TFR plots only use column comparisons, not window comparisons
		isTfrPlot := plot.Group == "tfr"

		if !isTfrPlot {
			// Feature plots can use window comparisons
			fields = append(fields,
				plotItemConfigFieldCompareWindows,
				plotItemConfigFieldComparisonWindows,
			)
			// Feature plots use comparison_segment to specify which time window to compare
			fields = append(fields, plotItemConfigFieldComparisonSegment)
		}

		// Column comparison fields (used by both TFR and feature plots)
		fields = append(fields,
			plotItemConfigFieldCompareColumns,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonLabels,
		)

		// ROI field only for feature plots that use ROIs in comparisons
		// TFR topomaps are already spatial, so ROIs don't apply
		if !isTfrPlot {
			if _, excluded := roiComparisonExcludedPlotIDs[plot.ID]; !excluded {
				fields = append(fields, plotItemConfigFieldComparisonROIs)
			}
		}
	}

	fields = append(fields, extraPlotFieldsByPlotID[plot.ID]...)
	return fields
}
