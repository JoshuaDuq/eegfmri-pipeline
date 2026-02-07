package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

const (
	defaultViewHeight        = 40
	headerOverheadLines      = 10
	minimumVisibleLines      = 8
	initialLineCapacity      = 256
	maxAvailableItemsDisplay = 4
)

// plotDefaults holds all default values from eeg_config.yaml plotting section
var plotDefaults = struct {
	// Defaults & Output
	padInches  float64
	bboxInches string

	// Fonts
	fontFamily          string
	fontWeight          string
	fontSizeSmall       int
	fontSizeMedium      int
	fontSizeLarge       int
	fontSizeTitle       int
	fontSizeAnnotation  int
	fontSizeLabel       int
	fontSizeYLabel      int
	fontSizeSuptitle    int
	fontSizeFigureTitle int

	// Layout
	layoutTightRect           string
	layoutTightRectMicrostate string
	gridSpecWidthRatios       string
	gridSpecHeightRatios      string
	gridSpecHspace            float64
	gridSpecWspace            float64
	gridSpecLeft              float64
	gridSpecRight             float64
	gridSpecTop               float64
	gridSpecBottom            float64

	// Figure Sizes
	figureSizeStandard string
	figureSizeMedium   string
	figureSizeSmall    string
	figureSizeSquare   string
	figureSizeWide     string
	figureSizeTFR      string
	figureSizeTopomap  string

	// Colors
	colorPain           string
	colorNonpain        string
	colorSignificant    string
	colorNonsignificant string
	colorGray           string
	colorLightGray      string
	colorBlack          string
	colorBlue           string
	colorRed            string
	colorNetworkNode    string

	// Alpha
	alphaGrid       float64
	alphaFill       float64
	alphaCI         float64
	alphaCILine     float64
	alphaTextBox    float64
	alphaViolinBody float64
	alphaRidgeFill  float64

	// Scatter
	scatterMarkerSizeSmall   int
	scatterMarkerSizeLarge   int
	scatterMarkerSizeDefault int
	scatterAlpha             float64
	scatterEdgewidth         float64
	scatterEdgecolor         string

	// Bar
	barAlpha        float64
	barWidth        float64
	barCapsize      int
	barCapsizeLarge int

	// Line
	lineWidthThin       float64
	lineWidthStandard   float64
	lineWidthThick      float64
	lineWidthBold       float64
	lineAlphaStandard   float64
	lineAlphaDim        float64
	lineAlphaZeroLine   float64
	lineAlphaFitLine    float64
	lineAlphaDiagonal   float64
	lineAlphaReference  float64
	lineRegressionWidth float64
	lineResidualWidth   float64
	lineQQWidth         float64

	// Histogram
	histBins           int
	histBinsBehavioral int
	histBinsResidual   int
	histBinsTFR        int
	histEdgewidth      float64
	histAlpha          float64
	histAlphaResidual  float64
	histAlphaTFR       float64
	histEdgecolor      string

	// KDE
	kdePoints    int
	kdeLinewidth float64
	kdeAlpha     float64
	kdeColor     string

	// Errorbar
	errorbarMarkersize   int
	errorbarCapsize      int
	errorbarCapsizeLarge int

	// Text Positions
	textStatsX             float64
	textStatsY             float64
	textPvalueX            float64
	textPvalueY            float64
	textBootstrapX         float64
	textBootstrapY         float64
	textChannelAnnotationX float64
	textChannelAnnotationY float64
	textTitleY             float64
	textResidualQcTitleY   float64

	// Topomap
	topomapContours               int
	topomapColormap               string
	topomapColorbarFraction       float64
	topomapColorbarPad            float64
	topomapSigMaskMarker          string
	topomapSigMaskMarkerFaceColor string
	topomapSigMaskMarkerEdgeColor string
	topomapSigMaskLinewidth       float64
	topomapSigMaskMarkerSize      float64

	// TFR
	tfrLogBase               float64
	tfrPercentageMultiplier  float64
	tfrDefaultBaselineWindow string

	// TFR Topomap
	tfrTopomapWindowSizeMs         float64
	tfrTopomapWindowCount          int
	tfrTopomapLabelXPosition       float64
	tfrTopomapLabelYPositionBottom float64
	tfrTopomapLabelYPosition       float64
	tfrTopomapTitleY               float64
	tfrTopomapTitlePad             int
	tfrTopomapSubplotsRight        float64
	tfrTopomapTemporalHspace       float64
	tfrTopomapTemporalWspace       float64

	// Comparisons
	comparisonWindows string
	comparisonSegment string
	comparisonColumn  string
	comparisonValues  string
	comparisonLabels  string
	comparisonROIs    string

	// Plot Sizing (from plots section)
	roiWidthPerBand              float64
	roiWidthPerMetric            float64
	roiHeightPerRoi              float64
	powerWidthPerBand            float64
	powerHeightPerSegment        float64
	itpcWidthPerBin              float64
	itpcHeightPerBand            float64
	itpcWidthPerBandBox          float64
	itpcHeightBox                float64
	pacWidthPerRoi               float64
	pacHeightBox                 float64
	aperiodicWidthPerColumn      float64
	aperiodicHeightPerRow        float64
	complexityWidthPerMeasure    float64
	complexityHeightPerSegment   float64
	connectivityWidthPerCircle   float64
	connectivityWidthPerBand     float64
	connectivityHeightPerMeasure float64
	connectivityCircleMinLines   int
}{
	// Defaults & Output
	padInches:  0.02,
	bboxInches: "tight",

	// Fonts
	fontFamily:          "sans-serif",
	fontWeight:          "normal",
	fontSizeSmall:       7,
	fontSizeMedium:      8,
	fontSizeLarge:       9,
	fontSizeTitle:       10,
	fontSizeAnnotation:  4,
	fontSizeLabel:       8,
	fontSizeYLabel:      10,
	fontSizeSuptitle:    11,
	fontSizeFigureTitle: 12,

	// Layout
	layoutTightRect:           "0 0.03 1 1",
	layoutTightRectMicrostate: "0 0.02 1 0.96",
	gridSpecWidthRatios:       "4 1",
	gridSpecHeightRatios:      "1 4",
	gridSpecHspace:            0.15,
	gridSpecWspace:            0.15,
	gridSpecLeft:              0.1,
	gridSpecRight:             0.95,
	gridSpecTop:               0.80,
	gridSpecBottom:            0.12,

	// Figure Sizes
	figureSizeStandard: "10.0 8.0",
	figureSizeMedium:   "9.0 6.0",
	figureSizeSmall:    "7.2 5.4",
	figureSizeSquare:   "3.5 3.5",
	figureSizeWide:     "6.0 2.5",
	figureSizeTFR:      "10.0 10.0",
	figureSizeTopomap:  "7.0 7.0",

	// Colors
	colorPain:           "crimson",
	colorNonpain:        "navy",
	colorSignificant:    "#C42847",
	colorNonsignificant: "#666666",
	colorGray:           "#555555",
	colorLightGray:      "#999999",
	colorBlack:          "k",
	colorBlue:           "#1f77b4",
	colorRed:            "#d62728",
	colorNetworkNode:    "#87CEEB",

	// Alpha
	alphaGrid:       0.3,
	alphaFill:       0.6,
	alphaCI:         0.1,
	alphaCILine:     0.5,
	alphaTextBox:    0.8,
	alphaViolinBody: 0.6,
	alphaRidgeFill:  0.6,

	// Scatter
	scatterMarkerSizeSmall:   3,
	scatterMarkerSizeLarge:   30,
	scatterMarkerSizeDefault: 30,
	scatterAlpha:             0.7,
	scatterEdgewidth:         0.3,
	scatterEdgecolor:         "white",

	// Bar
	barAlpha:        0.8,
	barWidth:        0.6,
	barCapsize:      4,
	barCapsizeLarge: 3,

	// Line
	lineWidthThin:       0.5,
	lineWidthStandard:   0.75,
	lineWidthThick:      1.0,
	lineWidthBold:       1.5,
	lineAlphaStandard:   0.8,
	lineAlphaDim:        0.5,
	lineAlphaZeroLine:   0.3,
	lineAlphaFitLine:    0.8,
	lineAlphaDiagonal:   0.5,
	lineAlphaReference:  0.5,
	lineRegressionWidth: 1.5,
	lineResidualWidth:   1.0,
	lineQQWidth:         0.4,

	// Histogram
	histBins:           30,
	histBinsBehavioral: 15,
	histBinsResidual:   20,
	histBinsTFR:        50,
	histEdgewidth:      0.5,
	histAlpha:          0.7,
	histAlphaResidual:  0.75,
	histAlphaTFR:       0.8,
	histEdgecolor:      "white",

	// KDE
	kdePoints:    100,
	kdeLinewidth: 1.5,
	kdeAlpha:     0.8,
	kdeColor:     "darkblue",

	// Errorbar
	errorbarMarkersize:   4,
	errorbarCapsize:      2,
	errorbarCapsizeLarge: 3,

	// Text Positions
	textStatsX:             0.05,
	textStatsY:             0.95,
	textPvalueX:            0.98,
	textPvalueY:            0.88,
	textBootstrapX:         0.02,
	textBootstrapY:         0.98,
	textChannelAnnotationX: 0.02,
	textChannelAnnotationY: 0.94,
	textTitleY:             0.975,
	textResidualQcTitleY:   1.02,

	// Topomap
	topomapContours:               6,
	topomapColormap:               "RdBu_r",
	topomapColorbarFraction:       0.05,
	topomapColorbarPad:            0.05,
	topomapSigMaskMarker:          "o",
	topomapSigMaskMarkerFaceColor: "none",
	topomapSigMaskMarkerEdgeColor: "g",
	topomapSigMaskLinewidth:       0.8,
	topomapSigMaskMarkerSize:      3,

	// TFR
	tfrLogBase:               10.0,
	tfrPercentageMultiplier:  100.0,
	tfrDefaultBaselineWindow: "-5.0 -0.01",

	// TFR Topomap
	tfrTopomapWindowSizeMs:         100.0,
	tfrTopomapWindowCount:          5,
	tfrTopomapLabelXPosition:       0.5,
	tfrTopomapLabelYPositionBottom: 1.08,
	tfrTopomapLabelYPosition:       1.02,
	tfrTopomapTitleY:               1.04,
	tfrTopomapTitlePad:             4,
	tfrTopomapSubplotsRight:        0.75,
	tfrTopomapTemporalHspace:       0.15,
	tfrTopomapTemporalWspace:       0.8,

	// Comparisons
	comparisonWindows: "",
	comparisonSegment: "",
	comparisonColumn:  "",
	comparisonValues:  "",
	comparisonLabels:  "",
	comparisonROIs:    "",

	// Plot Sizing
	roiWidthPerBand:              3.5,
	roiWidthPerMetric:            3.5,
	roiHeightPerRoi:              4.0,
	powerWidthPerBand:            3.5,
	powerHeightPerSegment:        4.0,
	itpcWidthPerBin:              4.5,
	itpcHeightPerBand:            4.0,
	itpcWidthPerBandBox:          4.0,
	itpcHeightBox:                5.0,
	pacWidthPerRoi:               4.0,
	pacHeightBox:                 5.0,
	aperiodicWidthPerColumn:      5.0,
	aperiodicHeightPerRow:        4.5,
	complexityWidthPerMeasure:    4.5,
	complexityHeightPerSegment:   4.0,
	connectivityWidthPerCircle:   3.5,
	connectivityWidthPerBand:     3.5,
	connectivityHeightPerMeasure: 4.0,
	connectivityCircleMinLines:   0,
}

func (m Model) renderPlottingAdvancedConfigV2() string {
	var builder strings.Builder

	m.renderPlottingAdvancedHeader(&builder)
	if m.useDefaultAdvanced {
		return m.renderMinimalView(&builder)
	}

	m.renderHelpText(&builder)
	lines := m.buildAllLines()
	m.renderLinesWithScrolling(&builder, lines)

	return builder.String()
}

func (m Model) renderPlottingAdvancedHeader(builder *strings.Builder) {
	builder.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")
}

func (m Model) renderMinimalView(builder *strings.Builder) string {
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	builder.WriteString(infoStyle.Render("Using defaults. Space to customize.") + "\n\n")

	const labelWidth = 22
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	isFocused := m.advancedCursor == 0
	cursor := m.renderCursor(isFocused)
	labelStyle := m.buildLabelStyle(labelWidth, isFocused)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	builder.WriteString(cursor + labelStyle.Render("Configuration:") + " " + valueStyle.Render("defaults") + "  " + hintStyle.Render("Space to customize") + "\n")
	return builder.String()
}

func (m Model) renderHelpText(builder *strings.Builder) {
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	var helpText string
	switch {
	case m.editingNumber || m.editingText:
		helpText = "Enter value, Enter to confirm, Esc to cancel"
	case m.expandedOption >= 0:
		helpText = "Space: select  Esc: close list"
	default:
		helpText = "Space: expand/edit  Enter: proceed"
	}

	builder.WriteString(infoStyle.Render(helpText) + "\n")
}

type renderLine struct {
	text string
}

func (m Model) buildAllLines() []renderLine {
	const labelWidth = 28
	lines := make([]renderLine, 0, initialLineCapacity)

	rows := m.getPlottingAdvancedRows()
	plotByID := m.buildPlotMap()

	for i, row := range rows {
		focused := m.advancedCursor == i
		line := m.renderRow(row, plotByID, labelWidth, focused)
		lines = append(lines, line...)
	}

	return lines
}

func (m Model) buildPlotMap() map[string]PlotItem {
	plotByID := make(map[string]PlotItem, len(m.plotItems))
	for _, plot := range m.plotItems {
		plotByID[plot.ID] = plot
	}
	return plotByID
}

func formatTriState(value *bool) string {
	if value == nil {
		return "default"
	}
	if *value {
		return "ON"
	}
	return "OFF"
}

func formatFloatValue(value float64, defaultValue float64, format string) string {
	if value == 0 {
		return fmt.Sprintf(format+" (default)", defaultValue)
	}
	return fmt.Sprintf(format, value)
}

func formatIntValue(value int, defaultValue int) string {
	if value == 0 {
		return fmt.Sprintf("%d (default)", defaultValue)
	}
	return fmt.Sprintf("%d", value)
}

func formatStringValue(value string, defaultValue string) string {
	if strings.TrimSpace(value) == "" {
		if strings.TrimSpace(defaultValue) == "" {
			return "(default)"
		}
		return fmt.Sprintf("(default: %s)", defaultValue)
	}
	return value
}

func buildAvailableHint(prefix string, items []string) string {
	if len(items) == 0 {
		return ""
	}

	maxItems := maxAvailableItemsDisplay
	if len(items) < maxItems {
		maxItems = len(items)
	}

	suffix := ""
	if len(items) > maxItems {
		suffix = fmt.Sprintf(" (+%d)", len(items)-maxItems)
	}

	return fmt.Sprintf("%s: %s%s", prefix, strings.Join(items[:maxItems], " "), suffix)
}

func (m Model) isOptionDisabled(opt optionType) bool {
	return m.useDefaultAdvanced && opt != optUseDefaults
}

func (m Model) renderCursor(isFocused bool) string {
	if isFocused {
		return styles.RenderCursorOptional(m.CursorBlinkVisible())
	}
	return "  "
}

func (m Model) buildLabelStyle(width int, isFocused bool) lipgloss.Style {
	style := lipgloss.NewStyle().Foreground(styles.Text)
	if isFocused {
		style = style.Foreground(styles.Primary).Bold(true)
	}
	return style
}

func (m Model) renderGroupLine(opt optionType, label string, expanded bool, hint string, focused bool) renderLine {
	cursor := m.renderCursor(focused)
	arrow := m.getExpansionArrow(expanded)
	labelStyle := m.buildGroupLabelStyle(focused, m.isOptionDisabled(opt))
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	text := cursor + labelStyle.Render(fmt.Sprintf("%s %s", arrow, label)) + "  " + hintStyle.Render(hint)
	return renderLine{text: styles.TruncateLine(text, m.contentWidth)}
}

func (m Model) renderValueLine(opt optionType, label string, value string, hint string, focused bool, labelWidth int) renderLine {
	cursor := m.renderCursor(focused)
	labelStyle := m.buildLabelStyle(labelWidth, focused)
	valueStyle := m.buildValueStyle(opt)

	displayValue := value
	if strings.TrimSpace(displayValue) == "" {
		displayValue = "(default)"
	}

	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	text := styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(displayValue), hintStyle.Render(hint), labelWidth, m.contentWidth)
	return renderLine{text: text}
}

func (m Model) getExpansionArrow(expanded bool) string {
	if expanded {
		return "▾"
	}
	return "▸"
}

func (m Model) buildGroupLabelStyle(focused bool, disabled bool) lipgloss.Style {
	style := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
	if focused {
		style = style.Foreground(styles.Primary)
	}
	if disabled {
		style = style.Faint(true)
	}
	return style
}

func (m Model) buildValueStyle(opt optionType) lipgloss.Style {
	if m.isOptionDisabled(opt) {
		return lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	}
	return lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
}

func (m Model) renderSectionLine(label string, focused bool) renderLine {
	cursor := m.renderCursor(focused)
	style := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	return renderLine{text: cursor + style.Render(label)}
}

func (m Model) renderPlotHeaderLine(plot PlotItem, expanded bool, focused bool) renderLine {
	cursor := m.renderCursor(focused)
	arrow := m.getExpansionArrow(expanded)
	labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
	if focused {
		labelStyle = labelStyle.Foreground(styles.Primary)
	}
	metaStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	text := cursor + labelStyle.Render(fmt.Sprintf("%s %s", arrow, plot.Name)) + "  " + metaStyle.Render(plot.ID)
	return renderLine{text: text}
}

func (m Model) renderPlotValueLine(label string, value string, hint string, focused bool, labelWidth int) renderLine {
	cursor := m.renderCursor(focused)
	labelStyle := m.buildLabelStyle(labelWidth, focused)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	displayValue := value
	if strings.TrimSpace(displayValue) == "" {
		displayValue = "(default)"
	}

	const indent = "   "
	text := cursor + indent + labelStyle.Render(label+":") + " " + valueStyle.Render(displayValue) + "  " + hintStyle.Render(hint)
	return renderLine{text: text}
}

func (m Model) renderRow(row plottingAdvancedRow, plotByID map[string]PlotItem, labelWidth int, focused bool) []renderLine {
	switch row.kind {
	case plottingRowSection:
		return []renderLine{m.renderSectionLine(row.label, focused)}
	case plottingRowPlotInfo:
		infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
		return []renderLine{{text: "     " + infoStyle.Render(row.label)}}
	case plottingRowPlotHeader:
		plot := plotByID[row.plotID]
		return []renderLine{m.renderPlotHeaderLine(plot, m.plotItemConfigExpanded[row.plotID], focused)}
	case plottingRowPlotField:
		return m.renderPlotField(row, labelWidth, focused)
	case plottingRowOption:
		return m.renderOption(row.opt, labelWidth, focused)
	default:
		return []renderLine{{text: fmt.Sprintf("Unknown row kind: %d", row.kind)}}
	}
}

func (m Model) renderPlotField(row plottingAdvancedRow, labelWidth int, focused bool) []renderLine {
	cfg := m.plotItemConfigs[row.plotID]

	switch row.plotField {
	case plotItemConfigFieldCompareWindows:
		value := formatTriState(cfg.CompareWindows)
		return []renderLine{m.renderPlotValueLine("compare_windows", value, "default/ON/OFF", focused, labelWidth)}
	case plotItemConfigFieldComparisonWindows:
		value := m.getPlotFieldTextValue(cfg.ComparisonWindowsSpec, plotDefaults.comparisonWindows, row, plotItemConfigFieldComparisonWindows)
		hint := m.buildComparisonWindowsHint()
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedPlotComparisonWindows && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonWindows
		lines := []renderLine{m.renderPlotValueLine("windows", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field - use feature-specific windows
		featureGroup := m.getFeatureGroupForPlot(row.plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if isEditing && len(windows) > 0 {
			expandedLines := m.renderExpandedListItems(windows, m.isPlotFieldWindowSelected(cfg.ComparisonWindowsSpec))
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldCompareColumns:
		value := formatTriState(cfg.CompareColumns)
		return []renderLine{m.renderPlotValueLine("compare_columns", value, "default/ON/OFF", focused, labelWidth)}
	case plotItemConfigFieldComparisonSegment:
		value := m.getPlotFieldTextValue(cfg.ComparisonSegment, plotDefaults.comparisonSegment, row, plotItemConfigFieldComparisonSegment)
		hint := m.buildComparisonSegmentHint()
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedPlotComparisonWindows && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonSegment
		lines := []renderLine{m.renderPlotValueLine("segment", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field - use feature-specific windows
		featureGroup := m.getFeatureGroupForPlot(row.plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if isEditing && len(windows) > 0 {
			expandedLines := m.renderExpandedListItems(windows, func(w string) bool {
				return cfg.ComparisonSegment == w
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldComparisonColumn:
		value := m.getPlotFieldTextValue(cfg.ComparisonColumn, plotDefaults.comparisonColumn, row, plotItemConfigFieldComparisonColumn)
		hint := m.buildComparisonColumnHint()
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedPlotComparisonColumn && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonColumn
		lines := []renderLine{m.renderPlotValueLine("column", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field
		plotCols := m.GetPlottingComparisonColumns()
		if isEditing && len(plotCols) > 0 {
			expandedLines := m.renderExpandedListItems(plotCols, func(col string) bool {
				return cfg.ComparisonColumn == col
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldComparisonValues:
		value := m.getPlotFieldTextValue(cfg.ComparisonValuesSpec, plotDefaults.comparisonValues, row, plotItemConfigFieldComparisonValues)
		col := cfg.ComparisonColumn
		if col == "" {
			col = m.plotComparisonColumn
		}
		hint := "e.g. 0 1"
		vals := m.GetPlottingComparisonColumnValues(col)
		if col == "" {
			hint = "select column first"
		} else if len(vals) > 0 {
			hint = fmt.Sprintf("Space to select · %d values", len(vals))
		} else if m.columnDiscoveryError != "" {
			hint = "discovery failed · enter values manually"
		} else if !m.columnDiscoveryDone && !m.trialTableDiscoveryDone {
			hint = "discovering values..."
		} else {
			hint = "Space to select · enter values manually (space-separated)"
		}
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedPlotComparisonValues && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonValues
		lines := []renderLine{m.renderPlotValueLine("values", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field (even if empty - allows manual entry)
		if isEditing {
			if len(vals) > 0 {
				expandedLines := m.renderExpandedListItems(vals, m.isPlotFieldValueSelected(cfg.ComparisonValuesSpec))
				lines = append(lines, expandedLines...)
			} else if col != "" {
				// Show empty state message when column selected but values not discovered
				hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
				hintText := hintStyle.Render(fmt.Sprintf("  No values discovered for %s yet. Enter values manually (space-separated).", col))
				lines = append(lines, renderLine{text: hintText})
			}
		}
		return lines
	case plotItemConfigFieldComparisonLabels:
		value := m.getPlotFieldTextValue(cfg.ComparisonLabelsSpec, plotDefaults.comparisonLabels, row, plotItemConfigFieldComparisonLabels)
		return []renderLine{m.renderPlotValueLine("labels", value, "e.g. condA condB or \"High\" \"Low\"", focused, labelWidth)}
	case plotItemConfigFieldTopomapWindow:
		value := m.getPlotFieldTextValue(cfg.TopomapWindowsSpec, "baseline", row, plotItemConfigFieldTopomapWindow)
		hint := m.buildComparisonSegmentHint()
		isEditing := m.expandedOption == expandedPlotComparisonWindows && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldTopomapWindow
		lines := []renderLine{m.renderPlotValueLine("windows", value, hint, focused || isEditing, labelWidth)}
		plot := m.getPlotByID(row.plotID)
		windows := m.GetPlottingComparisonWindows(plot.Group)
		if isEditing && len(windows) > 0 {
			expandedLines := m.renderExpandedListItems(windows, func(w string) bool {
				return m.isColumnValueSelected(w)
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldConnectivityCircleTopFraction:
		value := m.getPlotFieldTextValue(cfg.ConnectivityCircleTopFraction, "0.1", row, plotItemConfigFieldConnectivityCircleTopFraction)
		return []renderLine{m.renderPlotValueLine("circle_top_fraction", value, "0.0-1.0 (default: 0.1)", focused, labelWidth)}
	case plotItemConfigFieldConnectivityCircleMinLines:
		value := m.getPlotFieldTextValue(cfg.ConnectivityCircleMinLines, "20", row, plotItemConfigFieldConnectivityCircleMinLines)
		return []renderLine{m.renderPlotValueLine("circle_min_lines", value, "integer (default: 20)", focused, labelWidth)}
	case plotItemConfigFieldConnectivityNetworkTopFraction:
		value := m.getPlotFieldTextValue(cfg.ConnectivityNetworkTopFraction, "0.0", row, plotItemConfigFieldConnectivityNetworkTopFraction)
		return []renderLine{m.renderPlotValueLine("network_top_fraction", value, "0.0-1.0 (default: 0.0 = all edges)", focused, labelWidth)}
	case plotItemConfigFieldItpcSharedColorbar:
		value := formatTriState(cfg.ItpcSharedColorbar)
		return []renderLine{m.renderPlotValueLine("shared_colorbar", value, "default/ON/OFF", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapActiveWindow:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapActiveWindow, "3.0 10.5", row, plotItemConfigFieldTfrTopomapActiveWindow)
		return []renderLine{m.renderPlotValueLine("active_window", value, "tmin tmax (default: 3.0 10.5)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapWindowSizeMs:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapWindowSizeMs, "100.0", row, plotItemConfigFieldTfrTopomapWindowSizeMs)
		return []renderLine{m.renderPlotValueLine("window_size_ms", value, "float ms (default: 100.0)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapWindowCount:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapWindowCount, "5", row, plotItemConfigFieldTfrTopomapWindowCount)
		return []renderLine{m.renderPlotValueLine("window_count", value, "int (default: 5)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapLabelXPosition:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapLabelXPosition, "0.5", row, plotItemConfigFieldTfrTopomapLabelXPosition)
		return []renderLine{m.renderPlotValueLine("label_x_position", value, "float (default: 0.5)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapLabelYPositionBottom:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapLabelYPositionBottom, "1.08", row, plotItemConfigFieldTfrTopomapLabelYPositionBottom)
		return []renderLine{m.renderPlotValueLine("label_y_position_bottom", value, "float (default: 1.08)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapLabelYPosition:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapLabelYPosition, "1.02", row, plotItemConfigFieldTfrTopomapLabelYPosition)
		return []renderLine{m.renderPlotValueLine("label_y_position", value, "float (default: 1.02)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapTitleY:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapTitleY, "1.04", row, plotItemConfigFieldTfrTopomapTitleY)
		return []renderLine{m.renderPlotValueLine("title_y", value, "float (default: 1.04)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapTitlePad:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapTitlePad, "4", row, plotItemConfigFieldTfrTopomapTitlePad)
		return []renderLine{m.renderPlotValueLine("title_pad", value, "int (default: 4)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapSubplotsRight:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapSubplotsRight, "0.75", row, plotItemConfigFieldTfrTopomapSubplotsRight)
		return []renderLine{m.renderPlotValueLine("subplots_right", value, "float (default: 0.75)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapTemporalHspace:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapTemporalHspace, "0.15", row, plotItemConfigFieldTfrTopomapTemporalHspace)
		return []renderLine{m.renderPlotValueLine("temporal_hspace", value, "float (default: 0.15)", focused, labelWidth)}
	case plotItemConfigFieldTfrTopomapTemporalWspace:
		value := m.getPlotFieldTextValue(cfg.TfrTopomapTemporalWspace, "0.8", row, plotItemConfigFieldTfrTopomapTemporalWspace)
		return []renderLine{m.renderPlotValueLine("temporal_wspace", value, "float (default: 0.8)", focused, labelWidth)}
	case plotItemConfigFieldComparisonROIs:
		value := m.getPlotFieldTextValue(cfg.ComparisonROIsSpec, plotDefaults.comparisonROIs, row, plotItemConfigFieldComparisonROIs)
		hint := "e.g. all Frontal Sensorimotor_Right ParOccipital_Midline"
		if len(m.discoveredROIs) > 0 {
			hint = fmt.Sprintf("Space to select · %d ROIs available", len(m.discoveredROIs))
		}
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedPlotComparisonROIs && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonROIs
		lines := []renderLine{m.renderPlotValueLine("rois", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field
		if isEditing && len(m.discoveredROIs) > 0 {
			expandedLines := m.renderExpandedListItems(m.discoveredROIs, m.isPlotFieldValueSelected(cfg.ComparisonROIsSpec))
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldBehaviorScatterFeatures:
		value := m.getPlotFieldTextValue(cfg.BehaviorScatterFeaturesSpec, "power", row, plotItemConfigFieldBehaviorScatterFeatures)
		hint := fmt.Sprintf("Space to select · %d feature types", len(behaviorScatterFeatureTypes))
		isEditing := m.expandedOption == expandedBehaviorScatterFeatures && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldBehaviorScatterFeatures
		lines := []renderLine{m.renderPlotValueLine("features", value, hint, focused || isEditing, labelWidth)}
		if isEditing {
			expandedLines := m.renderExpandedListItems(behaviorScatterFeatureTypes, m.isPlotFieldValueSelected(cfg.BehaviorScatterFeaturesSpec))
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldBehaviorScatterColumns:
		value := m.getPlotFieldTextValue(cfg.BehaviorScatterColumnsSpec, "rating", row, plotItemConfigFieldBehaviorScatterColumns)
		hint := "Space to select behavioral columns"
		plotCols := m.GetPlottingComparisonColumns()
		if len(plotCols) > 0 {
			hint = fmt.Sprintf("Space to select · %d columns", len(plotCols))
		}
		isEditing := m.expandedOption == expandedBehaviorScatterColumns && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldBehaviorScatterColumns
		lines := []renderLine{m.renderPlotValueLine("columns", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(plotCols) > 0 {
			expandedLines := m.renderExpandedListItems(plotCols, m.isPlotFieldValueSelected(cfg.BehaviorScatterColumnsSpec))
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldBehaviorScatterAggregationModes:
		value := m.getPlotFieldTextValue(cfg.BehaviorScatterAggregationModesSpec, "roi global", row, plotItemConfigFieldBehaviorScatterAggregationModes)
		hint := fmt.Sprintf("Space to select · %d modes", len(behaviorScatterAggregationModes))
		isEditing := m.expandedOption == expandedBehaviorScatterAggregation && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldBehaviorScatterAggregationModes
		lines := []renderLine{m.renderPlotValueLine("aggregation", value, hint, focused || isEditing, labelWidth)}
		if isEditing {
			expandedLines := m.renderExpandedListItems(behaviorScatterAggregationModes, m.isPlotFieldValueSelected(cfg.BehaviorScatterAggregationModesSpec))
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldBehaviorScatterSegment:
		value := m.getPlotFieldTextValue(cfg.BehaviorScatterSegmentSpec, "", row, plotItemConfigFieldBehaviorScatterSegment)
		hint := m.buildComparisonSegmentHint()
		// When dropdown is expanded for this field, treat as focused
		isEditing := m.expandedOption == expandedBehaviorScatterSegment && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldBehaviorScatterSegment
		lines := []renderLine{m.renderPlotValueLine("segment", value, hint, focused || isEditing, labelWidth)}
		// Show dropdown if expanded for this field - get all available segments
		windows := m.GetPlottingComparisonWindows()
		if isEditing && len(windows) > 0 {
			expandedLines := m.renderExpandedListItems(windows, func(w string) bool {
				return cfg.BehaviorScatterSegmentSpec == w
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
		value := m.getPlotFieldTextValue(cfg.BehaviorTemporalStatsFeatureFolder, "(auto)", row, plotItemConfigFieldBehaviorTemporalStatsFeatureFolder)
		isEditing := m.expandedOption == expandedTemporalTopomapsFeatureDir && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldBehaviorTemporalStatsFeatureFolder

		hint := "Space to select"
		if strings.TrimSpace(m.temporalTopomapsStatsFeatureFoldersError) != "" {
			hint = "Error: " + strings.TrimSpace(m.temporalTopomapsStatsFeatureFoldersError)
		} else if n := len(m.temporalTopomapsStatsFeatureFolders); n > 0 {
			hint = fmt.Sprintf("Space to select · %d folders", n)
		}

		lines := []renderLine{m.renderPlotValueLine("stats_feature_folder", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(m.temporalTopomapsStatsFeatureFolders) > 0 {
			expandedLines := m.renderExpandedListItems(m.temporalTopomapsStatsFeatureFolders, func(folder string) bool {
				return cfg.BehaviorTemporalStatsFeatureFolder == folder
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldDoseResponseDoseColumn:
		value := m.getPlotFieldTextValue(cfg.DoseResponseDoseColumn, "(auto)", row, plotItemConfigFieldDoseResponseDoseColumn)
		plotCols := m.GetPlottingComparisonColumns()
		hint := "Space to select (events.tsv) · or type manually"
		if len(plotCols) > 0 {
			hint = fmt.Sprintf("Space to select · %d columns", len(plotCols))
		}
		isEditing := m.expandedOption == expandedPlotComparisonColumn && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldDoseResponseDoseColumn
		lines := []renderLine{m.renderPlotValueLine("dose_column", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(plotCols) > 0 {
			expandedLines := m.renderExpandedListItems(plotCols, func(col string) bool {
				return cfg.DoseResponseDoseColumn == col
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldDoseResponseResponseColumn:
		value := m.getPlotFieldTextValue(cfg.DoseResponseResponseColumn, "(none)", row, plotItemConfigFieldDoseResponseResponseColumn)
		categories := m.GetTrialTableFeatureCategories()
		hint := "Space to select (trial table feature categories) · or type manually"
		if len(categories) > 0 {
			hint = fmt.Sprintf("Space to select · %d feature categories", len(categories))
		} else if strings.TrimSpace(m.trialTableDiscoveryError) != "" {
			hint = "Trial table discovery error: " + strings.TrimSpace(m.trialTableDiscoveryError)
		} else if len(m.trialTableColumns) == 0 {
			hint = "Trial table not discovered (run trial_table first)"
		}
		isEditing := m.expandedOption == expandedPlotComparisonColumn && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn
		lines := []renderLine{m.renderPlotValueLine("feature_categories", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(categories) > 0 {
			expandedLines := m.renderExpandedListItems(categories, m.isColumnValueSelected)
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldDoseResponsePainColumn:
		value := m.getPlotFieldTextValue(cfg.DoseResponsePainColumn, "(auto)", row, plotItemConfigFieldDoseResponsePainColumn)
		plotCols := m.GetPlottingComparisonColumns()
		hint := "Space to select (events.tsv) · or type manually"
		if len(plotCols) > 0 {
			hint = fmt.Sprintf("Space to select · %d columns", len(plotCols))
		}
		isEditing := m.expandedOption == expandedPlotComparisonColumn && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldDoseResponsePainColumn
		lines := []renderLine{m.renderPlotValueLine("pain_column", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(plotCols) > 0 {
			expandedLines := m.renderExpandedListItems(plotCols, func(col string) bool {
				return cfg.DoseResponsePainColumn == col
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	case plotItemConfigFieldDoseResponseSegment:
		value := m.getPlotFieldTextValue(cfg.DoseResponseSegment, "(default)", row, plotItemConfigFieldDoseResponseSegment)
		windows := m.GetPlottingComparisonWindows()
		hint := "Space to select · or type manually"
		if len(windows) > 0 {
			hint = fmt.Sprintf("Space to select · %d windows", len(windows))
		}
		isEditing := m.expandedOption == expandedPlotComparisonWindows && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldDoseResponseSegment
		lines := []renderLine{m.renderPlotValueLine("segment", value, hint, focused || isEditing, labelWidth)}
		if isEditing && len(windows) > 0 {
			expandedLines := m.renderExpandedListItems(windows, func(w string) bool {
				return cfg.DoseResponseSegment == w
			})
			lines = append(lines, expandedLines...)
		}
		return lines
	default:
		return []renderLine{{text: fmt.Sprintf("Unknown plot field: %d", row.plotField)}}
	}
}

// isPlotFieldWindowSelected returns a function that checks if a window is selected in the given spec
func (m Model) isPlotFieldWindowSelected(spec string) func(string) bool {
	return func(w string) bool {
		if spec == "" {
			return false
		}
		for _, v := range strings.Fields(spec) {
			if v == w {
				return true
			}
		}
		return false
	}
}

// isPlotFieldValueSelected returns a function that checks if a value is selected in the given spec
func (m Model) isPlotFieldValueSelected(spec string) func(string) bool {
	return func(v string) bool {
		if spec == "" {
			return false
		}
		for _, val := range strings.Fields(spec) {
			if val == v {
				return true
			}
		}
		for _, val := range strings.Split(spec, ",") {
			if strings.TrimSpace(val) == v {
				return true
			}
		}
		return false
	}
}

func (m Model) getPlotFieldTextValue(currentValue string, defaultValue string, row plottingAdvancedRow, field plotItemConfigField) string {
	value := formatStringValue(currentValue, defaultValue)
	if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == field {
		value = m.textBuffer + "█"
	}
	return value
}

func (m Model) buildComparisonWindowsHint() string {
	hint := "e.g. baseline active"
	if avail := buildAvailableHint("available", m.availableWindows); avail != "" {
		hint = hint + " · " + avail
	}
	return hint
}

func (m Model) buildComparisonSegmentHint() string {
	hint := "segment name"
	if avail := buildAvailableHint("available", m.availableWindows); avail != "" {
		hint = hint + " · " + avail
	}
	return hint
}

func (m Model) buildComparisonColumnHint() string {
	hint := "events.tsv column"
	if avail := buildAvailableHint("available", m.discoveredColumns); avail != "" {
		hint = hint + " · " + avail
	}
	return hint
}

func (m Model) formatTextFieldWithBuffer(field textField, currentValue string, defaultValue string) string {
	value := formatStringValue(currentValue, defaultValue)
	if m.editingText && m.editingTextField == field {
		value = m.textBuffer + "█"
	}
	return value
}

func (m Model) getFloatFieldValue(opt optionType, currentValue float64, defaultValue float64, format string) string {
	value := formatFloatValue(currentValue, defaultValue, format)
	if m.editingNumber && m.isCurrentlyEditing(opt) {
		value = m.numberBuffer + "█"
	}
	return value
}

func (m Model) getIntFieldValue(opt optionType, currentValue int, defaultValue int) string {
	value := formatIntValue(currentValue, defaultValue)
	if m.editingNumber && m.isCurrentlyEditing(opt) {
		value = m.numberBuffer + "█"
	}
	return value
}

func (m Model) renderOption(opt optionType, labelWidth int, focused bool) []renderLine {
	switch opt {
	case optUseDefaults:
		value := m.boolToOnOff(m.useDefaultAdvanced)
		return []renderLine{m.renderValueLine(opt, "Use Defaults", value, "Skip overrides", focused, labelWidth)}
	case optPlotGroupDefaults:
		return []renderLine{m.renderGroupLine(opt, "Defaults & Output", m.plotGroupDefaultsExpanded, "bbox/padding overrides", focused)}
	case optPlotGroupFonts:
		return []renderLine{m.renderGroupLine(opt, "Fonts", m.plotGroupFontsExpanded, "matplotlib font defaults", focused)}
	case optPlotGroupLayout:
		return []renderLine{m.renderGroupLine(opt, "Layout", m.plotGroupLayoutExpanded, "tight_layout & gridspec", focused)}
	case optPlotGroupFigureSizes:
		return []renderLine{m.renderGroupLine(opt, "Figure Sizes", m.plotGroupFigureSizesExpanded, "default figure sizes", focused)}
	case optPlotGroupColors:
		return []renderLine{m.renderGroupLine(opt, "Colors", m.plotGroupColorsExpanded, "palette overrides", focused)}
	case optPlotGroupAlpha:
		return []renderLine{m.renderGroupLine(opt, "Alpha", m.plotGroupAlphaExpanded, "opacity overrides", focused)}
	case optPlotGroupScatter:
		return []renderLine{m.renderGroupLine(opt, "Scatter", m.plotGroupScatterExpanded, "marker & edge styling", focused)}
	case optPlotGroupBar:
		return []renderLine{m.renderGroupLine(opt, "Bars", m.plotGroupBarExpanded, "bar styling", focused)}
	case optPlotGroupLine:
		return []renderLine{m.renderGroupLine(opt, "Lines", m.plotGroupLineExpanded, "line widths & alpha", focused)}
	case optPlotGroupHistogram:
		return []renderLine{m.renderGroupLine(opt, "Histogram", m.plotGroupHistogramExpanded, "bins & styling", focused)}
	case optPlotGroupKDE:
		return []renderLine{m.renderGroupLine(opt, "KDE", m.plotGroupKDEExpanded, "density styling", focused)}
	case optPlotGroupErrorbar:
		return []renderLine{m.renderGroupLine(opt, "Errorbars", m.plotGroupErrorbarExpanded, "errorbar sizing", focused)}
	case optPlotGroupText:
		return []renderLine{m.renderGroupLine(opt, "Text Positions", m.plotGroupTextExpanded, "annotation placement", focused)}
	case optPlotGroupValidation:
		return []renderLine{m.renderGroupLine(opt, "Validation", m.plotGroupValidationExpanded, "min samples, bins, etc", focused)}
	case optPlotGroupTFRMisc:
		return []renderLine{m.renderGroupLine(opt, "TFR Misc", m.plotGroupTFRMiscExpanded, "baseline defaults", focused)}
	case optPlotGroupTopomap:
		return []renderLine{m.renderGroupLine(opt, "Topomap", m.plotGroupTopomapExpanded, "topomap rendering", focused)}
	case optPlotGroupTFR:
		return []renderLine{m.renderGroupLine(opt, "TFR", m.plotGroupTFRExpanded, "time-frequency controls", focused)}
	case optPlotGroupSizing:
		return []renderLine{m.renderGroupLine(opt, "Plot Sizing", m.plotGroupSizingExpanded, "per-plot sizing", focused)}
	case optPlotGroupSelection:
		return []renderLine{m.renderGroupLine(opt, "Selections", m.plotGroupSelectionExpanded, "metric lists & measures", focused)}
	case optPlotGroupComparisons:
		return []renderLine{m.renderGroupLine(opt, "Comparisons", m.plotGroupComparisonsExpanded, "condition/segment comparisons", focused)}

	case optPlotBboxInches:
		value := m.formatTextFieldWithBuffer(textFieldPlotBboxInches, m.plotBboxInches, plotDefaults.bboxInches)
		return []renderLine{m.renderValueLine(opt, "bbox_inches", value, "e.g. tight", focused, labelWidth)}
	case optPlotFontFamily:
		value := m.formatTextFieldWithBuffer(textFieldPlotFontFamily, m.plotFontFamily, plotDefaults.fontFamily)
		return []renderLine{m.renderValueLine(opt, "font_family", value, "matplotlib font family", focused, labelWidth)}
	case optPlotFontWeight:
		value := m.formatTextFieldWithBuffer(textFieldPlotFontWeight, m.plotFontWeight, plotDefaults.fontWeight)
		return []renderLine{m.renderValueLine(opt, "font_weight", value, "e.g. normal/bold", focused, labelWidth)}
	case optPlotLayoutTightRect:
		value := m.formatTextFieldWithBuffer(textFieldPlotLayoutTightRect, m.plotLayoutTightRectSpec, plotDefaults.layoutTightRect)
		return []renderLine{m.renderValueLine(opt, "tight_rect", value, "left bottom right top", focused, labelWidth)}
	case optPlotLayoutTightRectMicrostate:
		value := m.formatTextFieldWithBuffer(textFieldPlotLayoutTightRectMicrostate, m.plotLayoutTightRectMicrostateSpec, plotDefaults.layoutTightRectMicrostate)
		return []renderLine{m.renderValueLine(opt, "tight_rect_micro", value, "left bottom right top", focused, labelWidth)}
	case optPlotGridSpecWidthRatios:
		value := m.formatTextFieldWithBuffer(textFieldPlotGridSpecWidthRatios, m.plotGridSpecWidthRatiosSpec, plotDefaults.gridSpecWidthRatios)
		return []renderLine{m.renderValueLine(opt, "gridspec_width", value, "space-separated", focused, labelWidth)}
	case optPlotGridSpecHeightRatios:
		value := m.formatTextFieldWithBuffer(textFieldPlotGridSpecHeightRatios, m.plotGridSpecHeightRatiosSpec, plotDefaults.gridSpecHeightRatios)
		return []renderLine{m.renderValueLine(opt, "gridspec_height", value, "space-separated", focused, labelWidth)}
	case optPlotFigureSizeStandard:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeStandard, m.plotFigureSizeStandardSpec, plotDefaults.figureSizeStandard)
		return []renderLine{m.renderValueLine(opt, "figsize_std", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeMedium:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeMedium, m.plotFigureSizeMediumSpec, plotDefaults.figureSizeMedium)
		return []renderLine{m.renderValueLine(opt, "figsize_med", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeSmall:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeSmall, m.plotFigureSizeSmallSpec, plotDefaults.figureSizeSmall)
		return []renderLine{m.renderValueLine(opt, "figsize_small", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeSquare:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeSquare, m.plotFigureSizeSquareSpec, plotDefaults.figureSizeSquare)
		return []renderLine{m.renderValueLine(opt, "figsize_square", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeWide:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeWide, m.plotFigureSizeWideSpec, plotDefaults.figureSizeWide)
		return []renderLine{m.renderValueLine(opt, "figsize_wide", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeTFR:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeTFR, m.plotFigureSizeTFRSpec, plotDefaults.figureSizeTFR)
		return []renderLine{m.renderValueLine(opt, "figsize_tfr", value, "W H", focused, labelWidth)}
	case optPlotFigureSizeTopomap:
		value := m.formatTextFieldWithBuffer(textFieldPlotFigureSizeTopomap, m.plotFigureSizeTopomapSpec, plotDefaults.figureSizeTopomap)
		return []renderLine{m.renderValueLine(opt, "figsize_topomap", value, "W H", focused, labelWidth)}

	case optPlotColorPain:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorPain, m.plotColorPain, plotDefaults.colorPain)
		return []renderLine{m.renderValueLine(opt, "color_condition_2", value, "hex or named color", focused, labelWidth)}
	case optPlotColorNonpain:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorNonpain, m.plotColorNonpain, plotDefaults.colorNonpain)
		return []renderLine{m.renderValueLine(opt, "color_condition_1", value, "hex or named color", focused, labelWidth)}
	case optPlotColorSignificant:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorSignificant, m.plotColorSignificant, plotDefaults.colorSignificant)
		return []renderLine{m.renderValueLine(opt, "color_sig", value, "hex or named color", focused, labelWidth)}
	case optPlotColorNonsignificant:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorNonsignificant, m.plotColorNonsignificant, plotDefaults.colorNonsignificant)
		return []renderLine{m.renderValueLine(opt, "color_nonsig", value, "hex or named color", focused, labelWidth)}
	case optPlotColorGray:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorGray, m.plotColorGray, plotDefaults.colorGray)
		return []renderLine{m.renderValueLine(opt, "color_gray", value, "hex or named color", focused, labelWidth)}
	case optPlotColorLightGray:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorLightGray, m.plotColorLightGray, plotDefaults.colorLightGray)
		return []renderLine{m.renderValueLine(opt, "color_light_gray", value, "hex or named color", focused, labelWidth)}
	case optPlotColorBlack:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorBlack, m.plotColorBlack, plotDefaults.colorBlack)
		return []renderLine{m.renderValueLine(opt, "color_black", value, "hex or named color", focused, labelWidth)}
	case optPlotColorBlue:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorBlue, m.plotColorBlue, plotDefaults.colorBlue)
		return []renderLine{m.renderValueLine(opt, "color_blue", value, "hex or named color", focused, labelWidth)}
	case optPlotColorRed:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorRed, m.plotColorRed, plotDefaults.colorRed)
		return []renderLine{m.renderValueLine(opt, "color_red", value, "hex or named color", focused, labelWidth)}
	case optPlotColorNetworkNode:
		value := m.formatTextFieldWithBuffer(textFieldPlotColorNetworkNode, m.plotColorNetworkNode, plotDefaults.colorNetworkNode)
		return []renderLine{m.renderValueLine(opt, "color_net_node", value, "hex or named color", focused, labelWidth)}
	case optPlotScatterEdgecolor:
		value := m.formatTextFieldWithBuffer(textFieldPlotScatterEdgecolor, m.plotScatterEdgeColor, plotDefaults.scatterEdgecolor)
		return []renderLine{m.renderValueLine(opt, "scatter_edgecolor", value, "hex or named color", focused, labelWidth)}
	case optPlotHistEdgecolor:
		value := m.formatTextFieldWithBuffer(textFieldPlotHistEdgecolor, m.plotHistEdgeColor, plotDefaults.histEdgecolor)
		return []renderLine{m.renderValueLine(opt, "hist_edgecolor", value, "hex or named color", focused, labelWidth)}
	case optPlotKdeColor:
		value := m.formatTextFieldWithBuffer(textFieldPlotKdeColor, m.plotKdeColor, plotDefaults.kdeColor)
		return []renderLine{m.renderValueLine(opt, "kde_color", value, "hex or named color", focused, labelWidth)}
	case optPlotTopomapColormap:
		value := m.formatTextFieldWithBuffer(textFieldPlotTopomapColormap, m.plotTopomapColormap, plotDefaults.topomapColormap)
		return []renderLine{m.renderValueLine(opt, "topomap_cmap", value, "matplotlib cmap", focused, labelWidth)}
	case optPlotTopomapSigMaskMarker:
		value := m.formatTextFieldWithBuffer(textFieldPlotTopomapSigMaskMarker, m.plotTopomapSigMaskMarker, plotDefaults.topomapSigMaskMarker)
		return []renderLine{m.renderValueLine(opt, "sig_mask_marker", value, "e.g. o/x/.", focused, labelWidth)}
	case optPlotTopomapSigMaskMarkerFaceColor:
		value := m.formatTextFieldWithBuffer(textFieldPlotTopomapSigMaskMarkerFaceColor, m.plotTopomapSigMaskMarkerFaceColor, plotDefaults.topomapSigMaskMarkerFaceColor)
		return []renderLine{m.renderValueLine(opt, "sig_mask_face", value, "hex or named color", focused, labelWidth)}
	case optPlotTopomapSigMaskMarkerEdgeColor:
		value := m.formatTextFieldWithBuffer(textFieldPlotTopomapSigMaskMarkerEdgeColor, m.plotTopomapSigMaskMarkerEdgeColor, plotDefaults.topomapSigMaskMarkerEdgeColor)
		return []renderLine{m.renderValueLine(opt, "sig_mask_edge", value, "hex or named color", focused, labelWidth)}
	case optPlotPacCmap:
		value := m.formatTextFieldWithBuffer(textFieldPlotPacCmap, m.plotPacCmap, "magma")
		return []renderLine{m.renderValueLine(opt, "pac_cmap", value, "matplotlib cmap", focused, labelWidth)}

	case optPlotPadInches:
		value := m.getFloatFieldValue(optPlotPadInches, m.plotPadInches, plotDefaults.padInches, "%.4f")
		return []renderLine{m.renderValueLine(opt, "pad_inches", value, "float inches", focused, labelWidth)}
	case optPlotFontSizeSmall:
		value := m.getIntFieldValue(optPlotFontSizeSmall, m.plotFontSizeSmall, plotDefaults.fontSizeSmall)
		return []renderLine{m.renderValueLine(opt, "font_small", value, "int", focused, labelWidth)}
	case optPlotFontSizeMedium:
		value := m.getIntFieldValue(optPlotFontSizeMedium, m.plotFontSizeMedium, plotDefaults.fontSizeMedium)
		return []renderLine{m.renderValueLine(opt, "font_medium", value, "int", focused, labelWidth)}
	case optPlotFontSizeLarge:
		value := m.getIntFieldValue(optPlotFontSizeLarge, m.plotFontSizeLarge, plotDefaults.fontSizeLarge)
		return []renderLine{m.renderValueLine(opt, "font_large", value, "int", focused, labelWidth)}
	case optPlotFontSizeTitle:
		value := m.getIntFieldValue(optPlotFontSizeTitle, m.plotFontSizeTitle, plotDefaults.fontSizeTitle)
		return []renderLine{m.renderValueLine(opt, "font_title", value, "int", focused, labelWidth)}
	case optPlotFontSizeAnnotation:
		value := m.getIntFieldValue(optPlotFontSizeAnnotation, m.plotFontSizeAnnotation, plotDefaults.fontSizeAnnotation)
		return []renderLine{m.renderValueLine(opt, "font_annot", value, "int", focused, labelWidth)}
	case optPlotFontSizeLabel:
		value := m.getIntFieldValue(optPlotFontSizeLabel, m.plotFontSizeLabel, plotDefaults.fontSizeLabel)
		return []renderLine{m.renderValueLine(opt, "font_label", value, "int", focused, labelWidth)}
	case optPlotFontSizeYLabel:
		value := m.getIntFieldValue(optPlotFontSizeYLabel, m.plotFontSizeYLabel, plotDefaults.fontSizeYLabel)
		return []renderLine{m.renderValueLine(opt, "font_ylabel", value, "int", focused, labelWidth)}
	case optPlotFontSizeSuptitle:
		value := m.getIntFieldValue(optPlotFontSizeSuptitle, m.plotFontSizeSuptitle, plotDefaults.fontSizeSuptitle)
		return []renderLine{m.renderValueLine(opt, "font_suptitle", value, "int", focused, labelWidth)}
	case optPlotFontSizeFigureTitle:
		value := m.getIntFieldValue(optPlotFontSizeFigureTitle, m.plotFontSizeFigureTitle, plotDefaults.fontSizeFigureTitle)
		return []renderLine{m.renderValueLine(opt, "font_figtitle", value, "int", focused, labelWidth)}
	case optPlotGridSpecHspace:
		value := m.getFloatFieldValue(optPlotGridSpecHspace, m.plotGridSpecHspace, plotDefaults.gridSpecHspace, "%.4f")
		return []renderLine{m.renderValueLine(opt, "hspace", value, "float", focused, labelWidth)}
	case optPlotGridSpecWspace:
		value := m.getFloatFieldValue(optPlotGridSpecWspace, m.plotGridSpecWspace, plotDefaults.gridSpecWspace, "%.4f")
		return []renderLine{m.renderValueLine(opt, "wspace", value, "float", focused, labelWidth)}
	case optPlotGridSpecLeft:
		value := m.getFloatFieldValue(optPlotGridSpecLeft, m.plotGridSpecLeft, plotDefaults.gridSpecLeft, "%.4f")
		return []renderLine{m.renderValueLine(opt, "left", value, "float [0..1]", focused, labelWidth)}
	case optPlotGridSpecRight:
		value := m.getFloatFieldValue(optPlotGridSpecRight, m.plotGridSpecRight, plotDefaults.gridSpecRight, "%.4f")
		return []renderLine{m.renderValueLine(opt, "right", value, "float [0..1]", focused, labelWidth)}
	case optPlotGridSpecTop:
		value := m.getFloatFieldValue(optPlotGridSpecTop, m.plotGridSpecTop, plotDefaults.gridSpecTop, "%.4f")
		return []renderLine{m.renderValueLine(opt, "top", value, "float [0..1]", focused, labelWidth)}
	case optPlotGridSpecBottom:
		value := m.getFloatFieldValue(optPlotGridSpecBottom, m.plotGridSpecBottom, plotDefaults.gridSpecBottom, "%.4f")
		return []renderLine{m.renderValueLine(opt, "bottom", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaGrid:
		value := m.getFloatFieldValue(optPlotAlphaGrid, m.plotAlphaGrid, plotDefaults.alphaGrid, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_grid", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaFill:
		value := m.getFloatFieldValue(optPlotAlphaFill, m.plotAlphaFill, plotDefaults.alphaFill, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_fill", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaCI:
		value := m.getFloatFieldValue(optPlotAlphaCI, m.plotAlphaCI, plotDefaults.alphaCI, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_ci", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaCILine:
		value := m.getFloatFieldValue(optPlotAlphaCILine, m.plotAlphaCILine, plotDefaults.alphaCILine, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_ci_line", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaTextBox:
		value := m.getFloatFieldValue(optPlotAlphaTextBox, m.plotAlphaTextBox, plotDefaults.alphaTextBox, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_text_box", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaViolinBody:
		value := m.getFloatFieldValue(optPlotAlphaViolinBody, m.plotAlphaViolinBody, plotDefaults.alphaViolinBody, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_violin", value, "float [0..1]", focused, labelWidth)}
	case optPlotAlphaRidgeFill:
		value := m.getFloatFieldValue(optPlotAlphaRidgeFill, m.plotAlphaRidgeFill, plotDefaults.alphaRidgeFill, "%.4f")
		return []renderLine{m.renderValueLine(opt, "alpha_ridge", value, "float [0..1]", focused, labelWidth)}

	case optPlotScatterMarkerSizeSmall:
		value := m.getIntFieldValue(optPlotScatterMarkerSizeSmall, m.plotScatterMarkerSizeSmall, plotDefaults.scatterMarkerSizeSmall)
		return []renderLine{m.renderValueLine(opt, "scatter_ms_small", value, "int", focused, labelWidth)}
	case optPlotScatterMarkerSizeLarge:
		value := m.getIntFieldValue(optPlotScatterMarkerSizeLarge, m.plotScatterMarkerSizeLarge, plotDefaults.scatterMarkerSizeLarge)
		return []renderLine{m.renderValueLine(opt, "scatter_ms_large", value, "int", focused, labelWidth)}
	case optPlotScatterMarkerSizeDefault:
		value := m.getIntFieldValue(optPlotScatterMarkerSizeDefault, m.plotScatterMarkerSizeDefault, plotDefaults.scatterMarkerSizeDefault)
		return []renderLine{m.renderValueLine(opt, "scatter_ms_default", value, "int", focused, labelWidth)}
	case optPlotScatterAlpha:
		value := m.getFloatFieldValue(optPlotScatterAlpha, m.plotScatterAlpha, plotDefaults.scatterAlpha, "%.4f")
		return []renderLine{m.renderValueLine(opt, "scatter_alpha", value, "float [0..1]", focused, labelWidth)}
	case optPlotScatterEdgewidth:
		value := m.getFloatFieldValue(optPlotScatterEdgewidth, m.plotScatterEdgeWidth, plotDefaults.scatterEdgewidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "scatter_edgew", value, "float", focused, labelWidth)}
	case optPlotBarAlpha:
		value := m.getFloatFieldValue(optPlotBarAlpha, m.plotBarAlpha, plotDefaults.barAlpha, "%.4f")
		return []renderLine{m.renderValueLine(opt, "bar_alpha", value, "float [0..1]", focused, labelWidth)}
	case optPlotBarWidth:
		value := m.getFloatFieldValue(optPlotBarWidth, m.plotBarWidth, plotDefaults.barWidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "bar_width", value, "float", focused, labelWidth)}
	case optPlotBarCapsize:
		value := m.getIntFieldValue(optPlotBarCapsize, m.plotBarCapsize, plotDefaults.barCapsize)
		return []renderLine{m.renderValueLine(opt, "bar_capsize", value, "int", focused, labelWidth)}
	case optPlotBarCapsizeLarge:
		value := m.getIntFieldValue(optPlotBarCapsizeLarge, m.plotBarCapsizeLarge, plotDefaults.barCapsizeLarge)
		return []renderLine{m.renderValueLine(opt, "bar_capsize_lg", value, "int", focused, labelWidth)}
	case optPlotLineWidthThin:
		value := m.getFloatFieldValue(optPlotLineWidthThin, m.plotLineWidthThin, plotDefaults.lineWidthThin, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_thin", value, "float", focused, labelWidth)}
	case optPlotLineWidthStandard:
		value := m.getFloatFieldValue(optPlotLineWidthStandard, m.plotLineWidthStandard, plotDefaults.lineWidthStandard, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_std", value, "float", focused, labelWidth)}
	case optPlotLineWidthThick:
		value := m.getFloatFieldValue(optPlotLineWidthThick, m.plotLineWidthThick, plotDefaults.lineWidthThick, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_thick", value, "float", focused, labelWidth)}
	case optPlotLineWidthBold:
		value := m.getFloatFieldValue(optPlotLineWidthBold, m.plotLineWidthBold, plotDefaults.lineWidthBold, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_bold", value, "float", focused, labelWidth)}
	case optPlotLineAlphaStandard:
		value := m.getFloatFieldValue(optPlotLineAlphaStandard, m.plotLineAlphaStandard, plotDefaults.lineAlphaStandard, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_std", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineAlphaDim:
		value := m.getFloatFieldValue(optPlotLineAlphaDim, m.plotLineAlphaDim, plotDefaults.lineAlphaDim, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_dim", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineAlphaZeroLine:
		value := m.getFloatFieldValue(optPlotLineAlphaZeroLine, m.plotLineAlphaZeroLine, plotDefaults.lineAlphaZeroLine, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_zero", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineAlphaFitLine:
		value := m.getFloatFieldValue(optPlotLineAlphaFitLine, m.plotLineAlphaFitLine, plotDefaults.lineAlphaFitLine, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_fit", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineAlphaDiagonal:
		value := m.getFloatFieldValue(optPlotLineAlphaDiagonal, m.plotLineAlphaDiagonal, plotDefaults.lineAlphaDiagonal, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_diag", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineAlphaReference:
		value := m.getFloatFieldValue(optPlotLineAlphaReference, m.plotLineAlphaReference, plotDefaults.lineAlphaReference, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_a_ref", value, "float [0..1]", focused, labelWidth)}
	case optPlotLineRegressionWidth:
		value := m.getFloatFieldValue(optPlotLineRegressionWidth, m.plotLineRegressionWidth, plotDefaults.lineRegressionWidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_reg", value, "float", focused, labelWidth)}
	case optPlotLineResidualWidth:
		value := m.getFloatFieldValue(optPlotLineResidualWidth, m.plotLineResidualWidth, plotDefaults.lineResidualWidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_resid", value, "float", focused, labelWidth)}
	case optPlotLineQQWidth:
		value := m.getFloatFieldValue(optPlotLineQQWidth, m.plotLineQQWidth, plotDefaults.lineQQWidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "line_w_qq", value, "float", focused, labelWidth)}

	case optPlotHistBins:
		value := m.getIntFieldValue(optPlotHistBins, m.plotHistBins, plotDefaults.histBins)
		return []renderLine{m.renderValueLine(opt, "hist_bins", value, "int", focused, labelWidth)}
	case optPlotHistBinsBehavioral:
		value := m.getIntFieldValue(optPlotHistBinsBehavioral, m.plotHistBinsBehavioral, plotDefaults.histBinsBehavioral)
		return []renderLine{m.renderValueLine(opt, "hist_bins_beh", value, "int", focused, labelWidth)}
	case optPlotHistBinsResidual:
		value := m.getIntFieldValue(optPlotHistBinsResidual, m.plotHistBinsResidual, plotDefaults.histBinsResidual)
		return []renderLine{m.renderValueLine(opt, "hist_bins_resid", value, "int", focused, labelWidth)}
	case optPlotHistBinsTFR:
		value := m.getIntFieldValue(optPlotHistBinsTFR, m.plotHistBinsTFR, plotDefaults.histBinsTFR)
		return []renderLine{m.renderValueLine(opt, "hist_bins_tfr", value, "int", focused, labelWidth)}
	case optPlotHistEdgewidth:
		value := m.getFloatFieldValue(optPlotHistEdgewidth, m.plotHistEdgeWidth, plotDefaults.histEdgewidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "hist_edgew", value, "float", focused, labelWidth)}
	case optPlotHistAlpha:
		value := m.getFloatFieldValue(optPlotHistAlpha, m.plotHistAlpha, plotDefaults.histAlpha, "%.4f")
		return []renderLine{m.renderValueLine(opt, "hist_alpha", value, "float [0..1]", focused, labelWidth)}
	case optPlotHistAlphaResidual:
		value := m.getFloatFieldValue(optPlotHistAlphaResidual, m.plotHistAlphaResidual, plotDefaults.histAlphaResidual, "%.4f")
		return []renderLine{m.renderValueLine(opt, "hist_alpha_resid", value, "float [0..1]", focused, labelWidth)}
	case optPlotHistAlphaTFR:
		value := m.getFloatFieldValue(optPlotHistAlphaTFR, m.plotHistAlphaTFR, plotDefaults.histAlphaTFR, "%.4f")
		return []renderLine{m.renderValueLine(opt, "hist_alpha_tfr", value, "float [0..1]", focused, labelWidth)}
	case optPlotKdePoints:
		value := m.getIntFieldValue(optPlotKdePoints, m.plotKdePoints, plotDefaults.kdePoints)
		return []renderLine{m.renderValueLine(opt, "kde_points", value, "int", focused, labelWidth)}
	case optPlotKdeLinewidth:
		value := m.getFloatFieldValue(optPlotKdeLinewidth, m.plotKdeLinewidth, plotDefaults.kdeLinewidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "kde_linew", value, "float", focused, labelWidth)}
	case optPlotKdeAlpha:
		value := m.getFloatFieldValue(optPlotKdeAlpha, m.plotKdeAlpha, plotDefaults.kdeAlpha, "%.4f")
		return []renderLine{m.renderValueLine(opt, "kde_alpha", value, "float [0..1]", focused, labelWidth)}
	case optPlotErrorbarMarkersize:
		value := m.getIntFieldValue(optPlotErrorbarMarkersize, m.plotErrorbarMarkerSize, plotDefaults.errorbarMarkersize)
		return []renderLine{m.renderValueLine(opt, "err_ms", value, "int", focused, labelWidth)}
	case optPlotErrorbarCapsize:
		value := m.getIntFieldValue(optPlotErrorbarCapsize, m.plotErrorbarCapsize, plotDefaults.errorbarCapsize)
		return []renderLine{m.renderValueLine(opt, "err_capsize", value, "int", focused, labelWidth)}
	case optPlotErrorbarCapsizeLarge:
		value := m.getIntFieldValue(optPlotErrorbarCapsizeLarge, m.plotErrorbarCapsizeLarge, plotDefaults.errorbarCapsizeLarge)
		return []renderLine{m.renderValueLine(opt, "err_capsize_lg", value, "int", focused, labelWidth)}
	case optPlotTextStatsX:
		value := m.getFloatFieldValue(optPlotTextStatsX, m.plotTextStatsX, plotDefaults.textStatsX, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_stats_x", value, "float", focused, labelWidth)}
	case optPlotTextStatsY:
		value := m.getFloatFieldValue(optPlotTextStatsY, m.plotTextStatsY, plotDefaults.textStatsY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_stats_y", value, "float", focused, labelWidth)}
	case optPlotTextPvalueX:
		value := m.getFloatFieldValue(optPlotTextPvalueX, m.plotTextPvalueX, plotDefaults.textPvalueX, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_p_x", value, "float", focused, labelWidth)}
	case optPlotTextPvalueY:
		value := m.getFloatFieldValue(optPlotTextPvalueY, m.plotTextPvalueY, plotDefaults.textPvalueY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_p_y", value, "float", focused, labelWidth)}
	case optPlotTextBootstrapX:
		value := m.getFloatFieldValue(optPlotTextBootstrapX, m.plotTextBootstrapX, plotDefaults.textBootstrapX, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_boot_x", value, "float", focused, labelWidth)}
	case optPlotTextBootstrapY:
		value := m.getFloatFieldValue(optPlotTextBootstrapY, m.plotTextBootstrapY, plotDefaults.textBootstrapY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_boot_y", value, "float", focused, labelWidth)}
	case optPlotTextChannelAnnotationX:
		value := m.getFloatFieldValue(optPlotTextChannelAnnotationX, m.plotTextChannelAnnotationX, plotDefaults.textChannelAnnotationX, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_chan_x", value, "float", focused, labelWidth)}
	case optPlotTextChannelAnnotationY:
		value := m.getFloatFieldValue(optPlotTextChannelAnnotationY, m.plotTextChannelAnnotationY, plotDefaults.textChannelAnnotationY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_chan_y", value, "float", focused, labelWidth)}
	case optPlotTextTitleY:
		value := m.getFloatFieldValue(optPlotTextTitleY, m.plotTextTitleY, plotDefaults.textTitleY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_title_y", value, "float", focused, labelWidth)}
	case optPlotTextResidualQcTitleY:
		value := m.getFloatFieldValue(optPlotTextResidualQcTitleY, m.plotTextResidualQcTitleY, plotDefaults.textResidualQcTitleY, "%.4f")
		return []renderLine{m.renderValueLine(opt, "text_residqc_y", value, "float", focused, labelWidth)}
	case optPlotValidationMinBinsForCalibration:
		value := m.getIntFieldValue(optPlotValidationMinBinsForCalibration, m.plotValidationMinBinsForCalibration, 0)
		return []renderLine{m.renderValueLine(opt, "min_bins_cal", value, "int", focused, labelWidth)}
	case optPlotValidationMaxBinsForCalibration:
		value := m.getIntFieldValue(optPlotValidationMaxBinsForCalibration, m.plotValidationMaxBinsForCalibration, 0)
		return []renderLine{m.renderValueLine(opt, "max_bins_cal", value, "int", focused, labelWidth)}
	case optPlotValidationSamplesPerBin:
		value := m.getIntFieldValue(optPlotValidationSamplesPerBin, m.plotValidationSamplesPerBin, 0)
		return []renderLine{m.renderValueLine(opt, "samples_per_bin", value, "int", focused, labelWidth)}
	case optPlotValidationMinRoisForFDR:
		value := m.getIntFieldValue(optPlotValidationMinRoisForFDR, m.plotValidationMinRoisForFDR, 0)
		return []renderLine{m.renderValueLine(opt, "min_rois_fdr", value, "int", focused, labelWidth)}
	case optPlotValidationMinPvaluesForFDR:
		value := m.getIntFieldValue(optPlotValidationMinPvaluesForFDR, m.plotValidationMinPvaluesForFDR, 0)
		return []renderLine{m.renderValueLine(opt, "min_p_fdr", value, "int", focused, labelWidth)}

	case optPlotTopomapContours:
		value := m.getIntFieldValue(optPlotTopomapContours, m.plotTopomapContours, plotDefaults.topomapContours)
		return []renderLine{m.renderValueLine(opt, "topomap_contours", value, "int", focused, labelWidth)}
	case optPlotTopomapColorbarFraction:
		value := m.getFloatFieldValue(optPlotTopomapColorbarFraction, m.plotTopomapColorbarFraction, plotDefaults.topomapColorbarFraction, "%.4f")
		return []renderLine{m.renderValueLine(opt, "cbar_fraction", value, "float [0..1]", focused, labelWidth)}
	case optPlotTopomapColorbarPad:
		value := m.getFloatFieldValue(optPlotTopomapColorbarPad, m.plotTopomapColorbarPad, plotDefaults.topomapColorbarPad, "%.4f")
		return []renderLine{m.renderValueLine(opt, "cbar_pad", value, "float [0..1]", focused, labelWidth)}
	case optPlotTopomapDiffAnnotation:
		value := formatTriState(m.plotTopomapDiffAnnotation)
		return []renderLine{m.renderValueLine(opt, "diff_annotate", value, "tri-state", focused, labelWidth)}
	case optPlotTopomapAnnotateDescriptive:
		value := formatTriState(m.plotTopomapAnnotateDesc)
		return []renderLine{m.renderValueLine(opt, "annotate_desc", value, "tri-state", focused, labelWidth)}
	case optPlotTopomapSigMaskLinewidth:
		value := m.getFloatFieldValue(optPlotTopomapSigMaskLinewidth, m.plotTopomapSigMaskLinewidth, plotDefaults.topomapSigMaskLinewidth, "%.4f")
		return []renderLine{m.renderValueLine(opt, "sig_mask_lw", value, "float", focused, labelWidth)}
	case optPlotTopomapSigMaskMarkersize:
		value := m.getFloatFieldValue(optPlotTopomapSigMaskMarkersize, m.plotTopomapSigMaskMarkerSize, plotDefaults.topomapSigMaskMarkerSize, "%.4f")
		return []renderLine{m.renderValueLine(opt, "sig_mask_ms", value, "float", focused, labelWidth)}
	case optPlotTFRLogBase:
		value := m.getFloatFieldValue(optPlotTFRLogBase, m.plotTFRLogBase, plotDefaults.tfrLogBase, "%.4f")
		return []renderLine{m.renderValueLine(opt, "tfr_log_base", value, "float", focused, labelWidth)}
	case optPlotTFRPercentageMultiplier:
		value := m.getFloatFieldValue(optPlotTFRPercentageMultiplier, m.plotTFRPercentageMultiplier, plotDefaults.tfrPercentageMultiplier, "%.4f")
		return []renderLine{m.renderValueLine(opt, "tfr_pct_mult", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapWindowSizeMs:
		value := m.getFloatFieldValue(optPlotTFRTopomapWindowSizeMs, m.plotTFRTopomapWindowSizeMs, plotDefaults.tfrTopomapWindowSizeMs, "%.1f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_win_ms", value, "float ms", focused, labelWidth)}
	case optPlotTFRTopomapWindowCount:
		value := m.getIntFieldValue(optPlotTFRTopomapWindowCount, m.plotTFRTopomapWindowCount, plotDefaults.tfrTopomapWindowCount)
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_win_cnt", value, "int", focused, labelWidth)}
	case optPlotTFRTopomapLabelXPosition:
		value := m.getFloatFieldValue(optPlotTFRTopomapLabelXPosition, m.plotTFRTopomapLabelXPosition, plotDefaults.tfrTopomapLabelXPosition, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_lbl_x", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapLabelYPositionBottom:
		value := m.getFloatFieldValue(optPlotTFRTopomapLabelYPositionBottom, m.plotTFRTopomapLabelYPositionBottom, plotDefaults.tfrTopomapLabelYPositionBottom, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_lbl_y_bot", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapLabelYPosition:
		value := m.getFloatFieldValue(optPlotTFRTopomapLabelYPosition, m.plotTFRTopomapLabelYPosition, plotDefaults.tfrTopomapLabelYPosition, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_lbl_y", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapTitleY:
		value := m.getFloatFieldValue(optPlotTFRTopomapTitleY, m.plotTFRTopomapTitleY, plotDefaults.tfrTopomapTitleY, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_title_y", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapTitlePad:
		value := m.getIntFieldValue(optPlotTFRTopomapTitlePad, m.plotTFRTopomapTitlePad, plotDefaults.tfrTopomapTitlePad)
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_title_pad", value, "int", focused, labelWidth)}
	case optPlotTFRTopomapSubplotsRight:
		value := m.getFloatFieldValue(optPlotTFRTopomapSubplotsRight, m.plotTFRTopomapSubplotsRight, plotDefaults.tfrTopomapSubplotsRight, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_sub_r", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapTemporalHspace:
		value := m.getFloatFieldValue(optPlotTFRTopomapTemporalHspace, m.plotTFRTopomapTemporalHspace, plotDefaults.tfrTopomapTemporalHspace, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_tmp_h", value, "float", focused, labelWidth)}
	case optPlotTFRTopomapTemporalWspace:
		value := m.getFloatFieldValue(optPlotTFRTopomapTemporalWspace, m.plotTFRTopomapTemporalWspace, plotDefaults.tfrTopomapTemporalWspace, "%.2f")
		return []renderLine{m.renderValueLine(opt, "tfr_topomap_tmp_w", value, "float", focused, labelWidth)}

	case optPlotRoiWidthPerBand:
		value := m.getFloatFieldValue(optPlotRoiWidthPerBand, m.plotRoiWidthPerBand, plotDefaults.roiWidthPerBand, "%.4f")
		return []renderLine{m.renderValueLine(opt, "roi_w_per_band", value, "float", focused, labelWidth)}
	case optPlotRoiWidthPerMetric:
		value := m.getFloatFieldValue(optPlotRoiWidthPerMetric, m.plotRoiWidthPerMetric, plotDefaults.roiWidthPerMetric, "%.4f")
		return []renderLine{m.renderValueLine(opt, "roi_w_per_metric", value, "float", focused, labelWidth)}
	case optPlotRoiHeightPerRoi:
		value := m.getFloatFieldValue(optPlotRoiHeightPerRoi, m.plotRoiHeightPerRoi, plotDefaults.roiHeightPerRoi, "%.4f")
		return []renderLine{m.renderValueLine(opt, "roi_h_per_roi", value, "float", focused, labelWidth)}
	case optPlotPowerWidthPerBand:
		value := m.getFloatFieldValue(optPlotPowerWidthPerBand, m.plotPowerWidthPerBand, plotDefaults.powerWidthPerBand, "%.4f")
		return []renderLine{m.renderValueLine(opt, "power_w_per_band", value, "float", focused, labelWidth)}
	case optPlotPowerHeightPerSegment:
		value := m.getFloatFieldValue(optPlotPowerHeightPerSegment, m.plotPowerHeightPerSegment, plotDefaults.powerHeightPerSegment, "%.4f")
		return []renderLine{m.renderValueLine(opt, "power_h_per_seg", value, "float", focused, labelWidth)}
	case optPlotItpcWidthPerBin:
		value := m.getFloatFieldValue(optPlotItpcWidthPerBin, m.plotItpcWidthPerBin, plotDefaults.itpcWidthPerBin, "%.4f")
		return []renderLine{m.renderValueLine(opt, "itpc_w_per_bin", value, "float", focused, labelWidth)}
	case optPlotItpcHeightPerBand:
		value := m.getFloatFieldValue(optPlotItpcHeightPerBand, m.plotItpcHeightPerBand, plotDefaults.itpcHeightPerBand, "%.4f")
		return []renderLine{m.renderValueLine(opt, "itpc_h_per_band", value, "float", focused, labelWidth)}
	case optPlotItpcWidthPerBandBox:
		value := m.getFloatFieldValue(optPlotItpcWidthPerBandBox, m.plotItpcWidthPerBandBox, plotDefaults.itpcWidthPerBandBox, "%.4f")
		return []renderLine{m.renderValueLine(opt, "itpc_w_box", value, "float", focused, labelWidth)}
	case optPlotItpcHeightBox:
		value := m.getFloatFieldValue(optPlotItpcHeightBox, m.plotItpcHeightBox, plotDefaults.itpcHeightBox, "%.4f")
		return []renderLine{m.renderValueLine(opt, "itpc_h_box", value, "float", focused, labelWidth)}
	case optPlotPacWidthPerRoi:
		value := m.getFloatFieldValue(optPlotPacWidthPerRoi, m.plotPacWidthPerRoi, plotDefaults.pacWidthPerRoi, "%.4f")
		return []renderLine{m.renderValueLine(opt, "pac_w_per_roi", value, "float", focused, labelWidth)}
	case optPlotPacHeightBox:
		value := m.getFloatFieldValue(optPlotPacHeightBox, m.plotPacHeightBox, plotDefaults.pacHeightBox, "%.4f")
		return []renderLine{m.renderValueLine(opt, "pac_h_box", value, "float", focused, labelWidth)}
	case optPlotAperiodicWidthPerColumn:
		value := m.getFloatFieldValue(optPlotAperiodicWidthPerColumn, m.plotAperiodicWidthPerColumn, plotDefaults.aperiodicWidthPerColumn, "%.4f")
		return []renderLine{m.renderValueLine(opt, "aper_w_per_col", value, "float", focused, labelWidth)}
	case optPlotAperiodicHeightPerRow:
		value := m.getFloatFieldValue(optPlotAperiodicHeightPerRow, m.plotAperiodicHeightPerRow, plotDefaults.aperiodicHeightPerRow, "%.4f")
		return []renderLine{m.renderValueLine(opt, "aper_h_per_row", value, "float", focused, labelWidth)}
	case optPlotAperiodicNPerm:
		value := m.getIntFieldValue(optPlotAperiodicNPerm, m.plotAperiodicNPerm, 0)
		return []renderLine{m.renderValueLine(opt, "aper_n_perm", value, "int", focused, labelWidth)}
	case optPlotComplexityWidthPerMeasure:
		value := m.getFloatFieldValue(optPlotComplexityWidthPerMeasure, m.plotComplexityWidthPerMeasure, plotDefaults.complexityWidthPerMeasure, "%.4f")
		return []renderLine{m.renderValueLine(opt, "comp_w_per_meas", value, "float", focused, labelWidth)}
	case optPlotComplexityHeightPerSegment:
		value := m.getFloatFieldValue(optPlotComplexityHeightPerSegment, m.plotComplexityHeightPerSegment, plotDefaults.complexityHeightPerSegment, "%.4f")
		return []renderLine{m.renderValueLine(opt, "comp_h_per_seg", value, "float", focused, labelWidth)}
	case optPlotConnectivityWidthPerCircle:
		value := m.getFloatFieldValue(optPlotConnectivityWidthPerCircle, m.plotConnectivityWidthPerCircle, plotDefaults.connectivityWidthPerCircle, "%.4f")
		return []renderLine{m.renderValueLine(opt, "conn_w_circle", value, "float", focused, labelWidth)}
	case optPlotConnectivityWidthPerBand:
		value := m.getFloatFieldValue(optPlotConnectivityWidthPerBand, m.plotConnectivityWidthPerBand, plotDefaults.connectivityWidthPerBand, "%.4f")
		return []renderLine{m.renderValueLine(opt, "conn_w_band", value, "float", focused, labelWidth)}
	case optPlotConnectivityHeightPerMeasure:
		value := m.getFloatFieldValue(optPlotConnectivityHeightPerMeasure, m.plotConnectivityHeightPerMeasure, plotDefaults.connectivityHeightPerMeasure, "%.4f")
		return []renderLine{m.renderValueLine(opt, "conn_h_meas", value, "float", focused, labelWidth)}
	case optPlotConnectivityCircleTopFraction:
		value := m.getFloatFieldValue(optPlotConnectivityCircleTopFraction, m.plotConnectivityCircleTopFraction, 0, "%.4f")
		return []renderLine{m.renderValueLine(opt, "conn_top_frac", value, "float [0..1]", focused, labelWidth)}
	case optPlotConnectivityCircleMinLines:
		value := m.getIntFieldValue(optPlotConnectivityCircleMinLines, m.plotConnectivityCircleMinLines, plotDefaults.connectivityCircleMinLines)
		return []renderLine{m.renderValueLine(opt, "conn_min_lines", value, "int", focused, labelWidth)}
	case optPlotConnectivityNetworkTopFraction:
		value := m.getFloatFieldValue(optPlotConnectivityNetworkTopFraction, m.plotConnectivityNetworkTopFraction, 0, "%.4f")
		return []renderLine{m.renderValueLine(opt, "conn_net_top_frac", value, "float [0..1]", focused, labelWidth)}

	case optPlotConnectivityMeasures:
		value := m.formatTextFieldWithBuffer(textFieldPlotConnectivityMeasures, m.getTextFieldValue(textFieldPlotConnectivityMeasures), "")
		return []renderLine{m.renderValueLine(opt, "connectivity_measures", value, "space-separated (e.g. aec wpli)", focused, labelWidth)}
	case optPlotPacPairs:
		value := m.formatTextFieldWithBuffer(textFieldPlotPacPairs, m.plotPacPairsSpec, "")
		return []renderLine{m.renderValueLine(opt, "pac_pairs", value, "space-separated", focused, labelWidth)}
	case optPlotSpectralMetrics:
		value := m.formatTextFieldWithBuffer(textFieldPlotSpectralMetrics, m.plotSpectralMetricsSpec, "")
		return []renderLine{m.renderValueLine(opt, "spectral_metrics", value, "space-separated", focused, labelWidth)}
	case optPlotBurstsMetrics:
		value := m.formatTextFieldWithBuffer(textFieldPlotBurstsMetrics, m.plotBurstsMetricsSpec, "")
		return []renderLine{m.renderValueLine(opt, "bursts_metrics", value, "space-separated", focused, labelWidth)}
	case optPlotAsymmetryStat:
		value := m.formatTextFieldWithBuffer(textFieldPlotAsymmetryStat, m.plotAsymmetryStatSpec, "index")
		return []renderLine{m.renderValueLine(opt, "asymmetry_stat", value, "e.g. index", focused, labelWidth)}
	case optPlotTemporalTimeBins:
		value := m.formatTextFieldWithBuffer(textFieldPlotTemporalTimeBins, m.plotTemporalTimeBinsSpec, "")
		return []renderLine{m.renderValueLine(opt, "temporal_bins", value, "space-separated", focused, labelWidth)}
	case optPlotTemporalTimeLabels:
		value := m.formatTextFieldWithBuffer(textFieldPlotTemporalTimeLabels, m.plotTemporalTimeLabelsSpec, "")
		return []renderLine{m.renderValueLine(opt, "temporal_labels", value, "space-separated", focused, labelWidth)}
	case optPlotCompareWindows:
		value := formatTriState(m.plotCompareWindows)
		return []renderLine{m.renderValueLine(opt, "compare_windows", value, "tri-state", focused, labelWidth)}
	case optPlotComparisonWindows:
		return m.renderComparisonWindowsOption(opt, focused, labelWidth)
	case optPlotCompareColumns:
		value := formatTriState(m.plotCompareColumns)
		return []renderLine{m.renderValueLine(opt, "compare_columns", value, "tri-state", focused, labelWidth)}
	case optPlotComparisonSegment:
		value := m.formatTextFieldWithBuffer(textFieldPlotComparisonSegment, m.plotComparisonSegment, plotDefaults.comparisonSegment)
		hint := m.buildComparisonSegmentHint()
		return []renderLine{m.renderValueLine(opt, "comparison_segment", value, hint, focused, labelWidth)}
	case optPlotComparisonColumn:
		return m.renderComparisonColumnOption(opt, focused, labelWidth)
	case optPlotComparisonValues:
		return m.renderComparisonValuesOption(opt, focused, labelWidth)
	case optPlotComparisonLabels:
		value := m.formatTextFieldWithBuffer(textFieldPlotComparisonLabels, m.plotComparisonLabelsSpec, plotDefaults.comparisonLabels)
		return []renderLine{m.renderValueLine(opt, "comparison_labels", value, "2 labels; supports quotes", focused, labelWidth)}
	case optPlotComparisonROIs:
		value := m.formatTextFieldWithBuffer(textFieldPlotComparisonROIs, m.plotComparisonROIsSpec, plotDefaults.comparisonROIs)
		return []renderLine{m.renderValueLine(opt, "comparison_rois", value, "space-separated (empty = all)", focused, labelWidth)}
	case optPlotOverwrite:
		val := "OFF"
		if m.plotOverwrite != nil && *m.plotOverwrite {
			val = "ON"
		}
		return []renderLine{m.renderValueLine(opt, "overwrite", val, "overwrite existing plot files", focused, labelWidth)}

	default:
		// Fallback: keep UI robust even if new options are added without
		// wiring; still show a line so the list remains navigable.
		return []renderLine{m.renderValueLine(opt, fmt.Sprintf("opt_%d", opt), "(unwired)", "Space to edit (TODO)", focused, labelWidth)}
	}
}

func (m Model) renderLinesWithScrolling(builder *strings.Builder, lines []renderLine) {
	effectiveHeight := m.getEffectiveHeight()
	maxLines := m.calculateMaxVisibleLines(effectiveHeight)
	start, end, showScroll := m.calculateScrollBounds(len(lines), maxLines)

	if showScroll && start > 0 {
		scrollStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		builder.WriteString(scrollStyle.Render(fmt.Sprintf("  ↑ %d more items above", start)) + "\n")
	}

	for i := start; i < end; i++ {
		builder.WriteString(lines[i].text + "\n")
	}

	if showScroll && end < len(lines) {
		scrollStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		builder.WriteString(scrollStyle.Render(fmt.Sprintf("  ↓ %d more items below", len(lines)-end)) + "\n")
	}
}

func (m Model) getEffectiveHeight() int {
	if m.height <= 0 {
		return defaultViewHeight
	}
	return m.height
}

func (m Model) calculateMaxVisibleLines(effectiveHeight int) int {
	maxLines := effectiveHeight - headerOverheadLines
	if maxLines < minimumVisibleLines {
		return minimumVisibleLines
	}
	return maxLines
}

func (m Model) calculateScrollBounds(totalLines int, maxLines int) (start int, end int, showScroll bool) {
	if totalLines <= maxLines {
		return 0, totalLines, false
	}

	showScroll = true
	start = m.advancedOffset
	if start < 0 {
		start = 0
	}
	if start > totalLines-maxLines {
		start = totalLines - maxLines
	}
	if start < 0 {
		start = 0
	}

	end = start + maxLines
	if end > totalLines {
		end = totalLines
	}

	return start, end, showScroll
}

func (m Model) renderExpandedListItems(items []string, isSelected func(string) bool) []renderLine {
	lines := make([]renderLine, 0, len(items))
	for j, item := range items {
		isSubFocused := j == m.subCursor
		selected := isSelected(item)
		checkbox := styles.RenderCheckbox(selected, isSubFocused)
		itemStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isSubFocused {
			itemStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		lines = append(lines, renderLine{text: "      " + checkbox + " " + itemStyle.Render(item)})
	}
	return lines
}

func (m Model) renderComparisonWindowsOption(opt optionType, focused bool, labelWidth int) []renderLine {
	value := m.plotComparisonWindowsSpec
	if value == "" {
		value = "(select windows)"
	}
	if m.editingText && m.editingTextField == textFieldPlotComparisonWindows {
		value = m.textBuffer + "█"
	}
	windows := m.GetPlottingComparisonWindows()
	hint := "Space to select"
	if len(windows) > 0 {
		hint = fmt.Sprintf("Space to select · %d windows available", len(windows))
	}
	lines := []renderLine{m.renderValueLine(opt, "comparison_windows", value, hint, focused, labelWidth)}

	if m.expandedOption == expandedPlotComparisonWindows && focused && len(windows) > 0 {
		expandedLines := m.renderExpandedListItems(windows, m.isColumnValueSelected)
		lines = append(lines, expandedLines...)
	}
	return lines
}

func (m Model) renderComparisonColumnOption(opt optionType, focused bool, labelWidth int) []renderLine {
	value := m.plotComparisonColumn
	if value == "" {
		value = "(select column)"
	}
	if m.editingText && m.editingTextField == textFieldPlotComparisonColumn {
		value = m.textBuffer + "█"
	}
	hint := "Space to select"
	if len(m.discoveredColumns) > 0 {
		hint = fmt.Sprintf("Space to select · %d columns available", len(m.discoveredColumns))
	}
	lines := []renderLine{m.renderValueLine(opt, "comparison_column", value, hint, focused, labelWidth)}

	if m.expandedOption == expandedPlotComparisonColumn && focused && len(m.discoveredColumns) > 0 {
		expandedLines := m.renderExpandedListItems(m.discoveredColumns, func(col string) bool {
			return m.plotComparisonColumn == col
		})
		lines = append(lines, expandedLines...)
	}
	return lines
}

func (m Model) renderComparisonValuesOption(opt optionType, focused bool, labelWidth int) []renderLine {
	if m.plotComparisonColumn == "" {
		return []renderLine{m.renderValueLine(opt, "comparison_values", "(select column first)", "requires column selection", focused, labelWidth)}
	}

	value := m.plotComparisonValuesSpec
	if value == "" {
		value = "(select values)"
	}
	if m.editingText && m.editingTextField == textFieldPlotComparisonValues {
		value = m.textBuffer + "█"
	}
	hint := "Space to select"
	vals := m.GetPlottingComparisonColumnValues(m.plotComparisonColumn)
	if len(vals) > 0 {
		hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.plotComparisonColumn)
	}
	lines := []renderLine{m.renderValueLine(opt, "comparison_values", value, hint, focused, labelWidth)}

	if m.expandedOption == expandedPlotComparisonValues && focused && len(vals) > 0 {
		expandedLines := m.renderExpandedListItems(vals, m.isColumnValueSelected)
		lines = append(lines, expandedLines...)
	}
	return lines
}
