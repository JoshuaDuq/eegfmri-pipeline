package wizard

// Plotting advanced-row model types.

type plottingAdvancedRowKind int

const (
	plottingRowOption plottingAdvancedRowKind = iota
	plottingRowSection
	plottingRowPlotHeader
	plottingRowPlotField
	plottingRowPlotInfo
)

type plottingAdvancedRow struct {
	kind plottingAdvancedRowKind

	// kind == plottingRowOption
	opt optionType

	// kind == plottingRowPlotHeader/plottingRowPlotField/plottingRowPlotInfo
	plotID string

	// kind == plottingRowPlotField
	plotField plotItemConfigField

	// kind == plottingRowSection/plottingRowPlotInfo
	label string
}
