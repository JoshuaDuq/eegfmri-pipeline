package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
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
	qualityWidthPerPlot          float64
	qualityHeightPerPlot         float64
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
	qualityWidthPerPlot:          3.0,
	qualityHeightPerPlot:         2.5,
	complexityWidthPerMeasure:    4.5,
	complexityHeightPerSegment:   4.0,
	connectivityWidthPerCircle:   3.5,
	connectivityWidthPerBand:     3.5,
	connectivityHeightPerMeasure: 4.0,
	connectivityCircleMinLines:   0,
}

func (m Model) renderPlottingAdvancedConfigV2() string {
	var b strings.Builder

	accentFrames := []string{"◆", "◇", "◆", "◈"}
	accent := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Render(accentFrames[(m.ticker/3)%len(accentFrames)])
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" ADVANCED PLOT SETTINGS") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	// Minimal view when using defaults
	if m.useDefaultAdvanced {
		b.WriteString(infoStyle.Render("Default configuration will be used for plotting.") + "\n")
		b.WriteString(infoStyle.Render("Press Space to customize settings.") + "\n\n")

		labelWidth := 22
		hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

		isFocused := m.advancedCursor == 0
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		if isFocused {
			labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		b.WriteString(cursor + labelStyle.Render("Configuration:") + " " + valueStyle.Render("Using Defaults") + "  " + hintStyle.Render("Space to customize") + "\n")

		tipStyle := lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).PaddingLeft(4)
		b.WriteString("\n" + tipStyle.Render("Tip: In Custom mode, sections are collapsible for easier navigation.") + "\n")
		return b.String()
	}

	if m.editingNumber || m.editingText {
		b.WriteString(infoStyle.Render("Enter a value, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to expand/edit · ↑↓ to navigate · Enter to proceed") + "\n\n")
	}

	labelWidth := 28
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	type line struct {
		text    string
		focused bool
	}
	lines := make([]line, 0, 256)

	isDisabled := func(opt optionType) bool {
		return m.useDefaultAdvanced && opt != optUseDefaults
	}

	triState := func(v *bool) string {
		if v == nil {
			return "default"
		}
		if *v {
			return "ON"
		}
		return "OFF"
	}

	formatFloat := func(v float64, defaultVal float64, fmtStr string) string {
		if v == 0 {
			return fmt.Sprintf(fmtStr+" (default)", defaultVal)
		}
		return fmt.Sprintf(fmtStr, v)
	}

	formatInt := func(v int, defaultVal int) string {
		if v == 0 {
			return fmt.Sprintf("%d (default)", defaultVal)
		}
		return fmt.Sprintf("%d", v)
	}

	formatString := func(v string, defaultVal string) string {
		if strings.TrimSpace(v) == "" {
			return fmt.Sprintf("(default: %s)", defaultVal)
		}
		return v
	}

	availableHint := func(prefix string, items []string) string {
		if len(items) == 0 {
			return ""
		}
		max := 4
		if len(items) < max {
			max = len(items)
		}
		suffix := ""
		if len(items) > max {
			suffix = fmt.Sprintf(" (+%d)", len(items)-max)
		}
		return fmt.Sprintf("%s: %s%s", prefix, strings.Join(items[:max], " "), suffix)
	}

	groupLine := func(opt optionType, label string, expanded bool, hint string, focused bool) line {
		cursor := "  "
		if focused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		arrow := "▸"
		if expanded {
			arrow = "▾"
		}

		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
		if focused {
			labelStyle = labelStyle.Foreground(styles.Primary)
		}
		if isDisabled(opt) {
			labelStyle = labelStyle.Faint(true)
		}

		return line{
			text: cursor + labelStyle.Render(fmt.Sprintf("%s %s", arrow, label)) + "  " + hintStyle.Render(hint),
		}
	}

	valueLine := func(opt optionType, label string, value string, hint string, focused bool) line {
		cursor := "  "
		if focused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		if focused {
			labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
		}

		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		if isDisabled(opt) {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		}

		displayVal := value
		if strings.TrimSpace(displayVal) == "" {
			displayVal = "(default)"
		}
		return line{
			text: cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(displayVal) + "  " + hintStyle.Render(hint),
		}
	}

	rows := m.getPlottingAdvancedRows()
	plotByID := make(map[string]PlotItem, len(m.plotItems))
	for _, p := range m.plotItems {
		plotByID[p.ID] = p
	}

	sectionLine := func(label string, focused bool) line {
		cursor := "  "
		if focused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}
		style := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		if focused {
			style = style.Underline(true)
		}
		return line{text: cursor + style.Render(label)}
	}

	plotHeaderLine := func(plot PlotItem, expanded bool, focused bool) line {
		cursor := "  "
		if focused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}
		arrow := "▸"
		if expanded {
			arrow = "▾"
		}
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
		if focused {
			labelStyle = labelStyle.Foreground(styles.Primary)
		}
		metaStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		return line{
			text: cursor + labelStyle.Render(fmt.Sprintf("%s %s", arrow, plot.Name)) + "  " + metaStyle.Render(plot.ID),
		}
	}

	plotValueLine := func(label string, value string, hint string, focused bool) line {
		cursor := "  "
		if focused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		if focused {
			labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		displayVal := value
		if strings.TrimSpace(displayVal) == "" {
			displayVal = "(default)"
		}
		indent := "   "
		return line{
			text: cursor + indent + labelStyle.Render(label+":") + " " + valueStyle.Render(displayVal) + "  " + hintStyle.Render(hint),
		}
	}

	for i, row := range rows {
		focused := m.advancedCursor == i

		switch row.kind {
		case plottingRowSection:
			lines = append(lines, sectionLine(row.label, focused))
			continue
		case plottingRowPlotInfo:
			info := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(row.label)
			lines = append(lines, line{text: "     " + info})
			continue
		case plottingRowPlotHeader:
			plot := plotByID[row.plotID]
			lines = append(lines, plotHeaderLine(plot, m.plotItemConfigExpanded[row.plotID], focused))
			continue
		case plottingRowPlotField:
			plot := plotByID[row.plotID]
			cfg := m.plotItemConfigs[row.plotID]
			_ = plot

			switch row.plotField {
			case plotItemConfigFieldTfrDefaultBaselineWindow:
				val := formatString(cfg.TfrDefaultBaselineWindowSpec, plotDefaults.tfrDefaultBaselineWindow)
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldTfrDefaultBaselineWindow {
					val = m.textBuffer + "█"
				}
				lines = append(lines, plotValueLine("tfr_baseline", val, "tmin tmax", focused))
			case plotItemConfigFieldCompareWindows:
				lines = append(lines, plotValueLine("compare_windows", triState(cfg.CompareWindows), "default/ON/OFF", focused))
			case plotItemConfigFieldComparisonWindows:
				val := formatString(cfg.ComparisonWindowsSpec, "baseline active")
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonWindows {
					val = m.textBuffer + "█"
				}
				hint := "e.g. baseline active"
				if avail := availableHint("available", m.availableWindows); avail != "" {
					hint = hint + " · " + avail
				}
				lines = append(lines, plotValueLine("windows", val, hint, focused))
			case plotItemConfigFieldCompareColumns:
				lines = append(lines, plotValueLine("compare_columns", triState(cfg.CompareColumns), "default/ON/OFF", focused))
			case plotItemConfigFieldComparisonSegment:
				val := formatString(cfg.ComparisonSegment, "active")
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonSegment {
					val = m.textBuffer + "█"
				}
				hint := "segment name"
				if avail := availableHint("available", m.availableWindows); avail != "" {
					hint = hint + " · " + avail
				}
				lines = append(lines, plotValueLine("segment", val, hint, focused))
			case plotItemConfigFieldComparisonColumn:
				val := formatString(cfg.ComparisonColumn, "(auto: event_columns.pain_binary)")
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonColumn {
					val = m.textBuffer + "█"
				}
				hint := "events.tsv column"
				if avail := availableHint("available", m.availableColumns); avail != "" {
					hint = hint + " · " + avail
				}
				lines = append(lines, plotValueLine("column", val, hint, focused))
			case plotItemConfigFieldComparisonValues:
				val := formatString(cfg.ComparisonValuesSpec, "0 1")
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonValues {
					val = m.textBuffer + "█"
				}
				lines = append(lines, plotValueLine("values", val, "e.g. 0 1", focused))
			case plotItemConfigFieldComparisonROIs:
				val := formatString(cfg.ComparisonROIsSpec, "(all)")
				if m.editingText && m.editingPlotID == row.plotID && m.editingPlotField == plotItemConfigFieldComparisonROIs {
					val = m.textBuffer + "█"
				}
				lines = append(lines, plotValueLine("rois", val, "e.g. all Frontal Midline_ACC_MCC", focused))
			}
			continue
		}

		opt := row.opt
		switch opt {
		case optUseDefaults:
			lines = append(lines, valueLine(opt, "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip overrides", focused))

		case optPlotGroupDefaults:
			lines = append(lines, groupLine(opt, "Defaults & Output", m.plotGroupDefaultsExpanded, "bbox/padding overrides", focused))
		case optPlotGroupFonts:
			lines = append(lines, groupLine(opt, "Fonts", m.plotGroupFontsExpanded, "matplotlib font defaults", focused))
		case optPlotGroupLayout:
			lines = append(lines, groupLine(opt, "Layout", m.plotGroupLayoutExpanded, "tight_layout & gridspec", focused))
		case optPlotGroupFigureSizes:
			lines = append(lines, groupLine(opt, "Figure Sizes", m.plotGroupFigureSizesExpanded, "default figure sizes", focused))
		case optPlotGroupColors:
			lines = append(lines, groupLine(opt, "Colors", m.plotGroupColorsExpanded, "palette overrides", focused))
		case optPlotGroupAlpha:
			lines = append(lines, groupLine(opt, "Alpha", m.plotGroupAlphaExpanded, "opacity overrides", focused))
		case optPlotGroupScatter:
			lines = append(lines, groupLine(opt, "Scatter", m.plotGroupScatterExpanded, "marker & edge styling", focused))
		case optPlotGroupBar:
			lines = append(lines, groupLine(opt, "Bars", m.plotGroupBarExpanded, "bar styling", focused))
		case optPlotGroupLine:
			lines = append(lines, groupLine(opt, "Lines", m.plotGroupLineExpanded, "line widths & alpha", focused))
		case optPlotGroupHistogram:
			lines = append(lines, groupLine(opt, "Histogram", m.plotGroupHistogramExpanded, "bins & styling", focused))
		case optPlotGroupKDE:
			lines = append(lines, groupLine(opt, "KDE", m.plotGroupKDEExpanded, "density styling", focused))
		case optPlotGroupErrorbar:
			lines = append(lines, groupLine(opt, "Errorbars", m.plotGroupErrorbarExpanded, "errorbar sizing", focused))
		case optPlotGroupText:
			lines = append(lines, groupLine(opt, "Text Positions", m.plotGroupTextExpanded, "annotation placement", focused))
		case optPlotGroupValidation:
			lines = append(lines, groupLine(opt, "Validation", m.plotGroupValidationExpanded, "min samples, bins, etc", focused))
		case optPlotGroupTFRMisc:
			lines = append(lines, groupLine(opt, "TFR Misc", m.plotGroupTFRMiscExpanded, "baseline defaults", focused))

		case optPlotGroupTopomap:
			lines = append(lines, groupLine(opt, "Topomap", m.plotGroupTopomapExpanded, "topomap rendering", focused))
		case optPlotGroupTFR:
			lines = append(lines, groupLine(opt, "TFR", m.plotGroupTFRExpanded, "time-frequency controls", focused))
		case optPlotGroupSizing:
			lines = append(lines, groupLine(opt, "Plot Sizing", m.plotGroupSizingExpanded, "per-plot sizing", focused))
		case optPlotGroupSelection:
			lines = append(lines, groupLine(opt, "Selections", m.plotGroupSelectionExpanded, "metric lists & measures", focused))

		// Defaults / styling (strings)
		case optPlotBboxInches:
			val := formatString(m.plotBboxInches, plotDefaults.bboxInches)
			if m.editingText && m.editingTextField == textFieldPlotBboxInches {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bbox_inches", val, "e.g. tight", focused))
		case optPlotFontFamily:
			val := formatString(m.plotFontFamily, plotDefaults.fontFamily)
			if m.editingText && m.editingTextField == textFieldPlotFontFamily {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_family", val, "matplotlib font family", focused))
		case optPlotFontWeight:
			val := formatString(m.plotFontWeight, plotDefaults.fontWeight)
			if m.editingText && m.editingTextField == textFieldPlotFontWeight {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_weight", val, "e.g. normal/bold", focused))
		case optPlotLayoutTightRect:
			val := formatString(m.plotLayoutTightRectSpec, plotDefaults.layoutTightRect)
			if m.editingText && m.editingTextField == textFieldPlotLayoutTightRect {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "tight_rect", val, "left bottom right top", focused))
		case optPlotLayoutTightRectMicrostate:
			val := formatString(m.plotLayoutTightRectMicrostateSpec, plotDefaults.layoutTightRectMicrostate)
			if m.editingText && m.editingTextField == textFieldPlotLayoutTightRectMicrostate {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "tight_rect_micro", val, "left bottom right top", focused))
		case optPlotGridSpecWidthRatios:
			val := formatString(m.plotGridSpecWidthRatiosSpec, plotDefaults.gridSpecWidthRatios)
			if m.editingText && m.editingTextField == textFieldPlotGridSpecWidthRatios {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "gridspec_width", val, "space-separated", focused))
		case optPlotGridSpecHeightRatios:
			val := formatString(m.plotGridSpecHeightRatiosSpec, plotDefaults.gridSpecHeightRatios)
			if m.editingText && m.editingTextField == textFieldPlotGridSpecHeightRatios {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "gridspec_height", val, "space-separated", focused))
		case optPlotFigureSizeStandard:
			val := formatString(m.plotFigureSizeStandardSpec, plotDefaults.figureSizeStandard)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeStandard {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_std", val, "W H", focused))
		case optPlotFigureSizeMedium:
			val := formatString(m.plotFigureSizeMediumSpec, plotDefaults.figureSizeMedium)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeMedium {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_med", val, "W H", focused))
		case optPlotFigureSizeSmall:
			val := formatString(m.plotFigureSizeSmallSpec, plotDefaults.figureSizeSmall)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeSmall {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_small", val, "W H", focused))
		case optPlotFigureSizeSquare:
			val := formatString(m.plotFigureSizeSquareSpec, plotDefaults.figureSizeSquare)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeSquare {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_square", val, "W H", focused))
		case optPlotFigureSizeWide:
			val := formatString(m.plotFigureSizeWideSpec, plotDefaults.figureSizeWide)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeWide {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_wide", val, "W H", focused))
		case optPlotFigureSizeTFR:
			val := formatString(m.plotFigureSizeTFRSpec, plotDefaults.figureSizeTFR)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeTFR {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_tfr", val, "W H", focused))
		case optPlotFigureSizeTopomap:
			val := formatString(m.plotFigureSizeTopomapSpec, plotDefaults.figureSizeTopomap)
			if m.editingText && m.editingTextField == textFieldPlotFigureSizeTopomap {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "figsize_topomap", val, "W H", focused))

		case optPlotColorPain:
			val := formatString(m.plotColorPain, plotDefaults.colorPain)
			if m.editingText && m.editingTextField == textFieldPlotColorPain {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_pain", val, "hex or named color", focused))
		case optPlotColorNonpain:
			val := formatString(m.plotColorNonpain, plotDefaults.colorNonpain)
			if m.editingText && m.editingTextField == textFieldPlotColorNonpain {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_nonpain", val, "hex or named color", focused))
		case optPlotColorSignificant:
			val := formatString(m.plotColorSignificant, plotDefaults.colorSignificant)
			if m.editingText && m.editingTextField == textFieldPlotColorSignificant {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_sig", val, "hex or named color", focused))
		case optPlotColorNonsignificant:
			val := formatString(m.plotColorNonsignificant, plotDefaults.colorNonsignificant)
			if m.editingText && m.editingTextField == textFieldPlotColorNonsignificant {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_nonsig", val, "hex or named color", focused))
		case optPlotColorGray:
			val := formatString(m.plotColorGray, plotDefaults.colorGray)
			if m.editingText && m.editingTextField == textFieldPlotColorGray {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_gray", val, "hex or named color", focused))
		case optPlotColorLightGray:
			val := formatString(m.plotColorLightGray, plotDefaults.colorLightGray)
			if m.editingText && m.editingTextField == textFieldPlotColorLightGray {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_light_gray", val, "hex or named color", focused))
		case optPlotColorBlack:
			val := formatString(m.plotColorBlack, plotDefaults.colorBlack)
			if m.editingText && m.editingTextField == textFieldPlotColorBlack {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_black", val, "hex or named color", focused))
		case optPlotColorBlue:
			val := formatString(m.plotColorBlue, plotDefaults.colorBlue)
			if m.editingText && m.editingTextField == textFieldPlotColorBlue {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_blue", val, "hex or named color", focused))
		case optPlotColorRed:
			val := formatString(m.plotColorRed, plotDefaults.colorRed)
			if m.editingText && m.editingTextField == textFieldPlotColorRed {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_red", val, "hex or named color", focused))
		case optPlotColorNetworkNode:
			val := formatString(m.plotColorNetworkNode, plotDefaults.colorNetworkNode)
			if m.editingText && m.editingTextField == textFieldPlotColorNetworkNode {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "color_net_node", val, "hex or named color", focused))

		case optPlotScatterEdgecolor:
			val := formatString(m.plotScatterEdgeColor, plotDefaults.scatterEdgecolor)
			if m.editingText && m.editingTextField == textFieldPlotScatterEdgecolor {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_edgecolor", val, "hex or named color", focused))
		case optPlotHistEdgecolor:
			val := formatString(m.plotHistEdgeColor, plotDefaults.histEdgecolor)
			if m.editingText && m.editingTextField == textFieldPlotHistEdgecolor {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_edgecolor", val, "hex or named color", focused))
		case optPlotKdeColor:
			val := formatString(m.plotKdeColor, plotDefaults.kdeColor)
			if m.editingText && m.editingTextField == textFieldPlotKdeColor {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "kde_color", val, "hex or named color", focused))

		case optPlotTopomapColormap:
			val := formatString(m.plotTopomapColormap, plotDefaults.topomapColormap)
			if m.editingText && m.editingTextField == textFieldPlotTopomapColormap {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "topomap_cmap", val, "matplotlib cmap", focused))
		case optPlotTopomapSigMaskMarker:
			val := formatString(m.plotTopomapSigMaskMarker, plotDefaults.topomapSigMaskMarker)
			if m.editingText && m.editingTextField == textFieldPlotTopomapSigMaskMarker {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "sig_mask_marker", val, "e.g. o/x/.", focused))
		case optPlotTopomapSigMaskMarkerFaceColor:
			val := formatString(m.plotTopomapSigMaskMarkerFaceColor, plotDefaults.topomapSigMaskMarkerFaceColor)
			if m.editingText && m.editingTextField == textFieldPlotTopomapSigMaskMarkerFaceColor {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "sig_mask_face", val, "hex or named color", focused))
		case optPlotTopomapSigMaskMarkerEdgeColor:
			val := formatString(m.plotTopomapSigMaskMarkerEdgeColor, plotDefaults.topomapSigMaskMarkerEdgeColor)
			if m.editingText && m.editingTextField == textFieldPlotTopomapSigMaskMarkerEdgeColor {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "sig_mask_edge", val, "hex or named color", focused))
		case optPlotTfrDefaultBaselineWindow:
			val := formatString(m.plotTfrDefaultBaselineWindowSpec, plotDefaults.tfrDefaultBaselineWindow)
			if m.editingText && m.editingTextField == textFieldPlotTfrDefaultBaselineWindow {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "tfr_baseline", val, "tmin tmax", focused))
		case optPlotPacCmap:
			val := formatString(m.plotPacCmap, "magma")
			if m.editingText && m.editingTextField == textFieldPlotPacCmap {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "pac_cmap", val, "matplotlib cmap", focused))

		// Numbers (generic display + buffer override)
		case optPlotPadInches:
			val := formatFloat(m.plotPadInches, plotDefaults.padInches, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotPadInches) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "pad_inches", val, "float inches", focused))
		case optPlotFontSizeSmall:
			val := formatInt(m.plotFontSizeSmall, plotDefaults.fontSizeSmall)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeSmall) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_small", val, "int", focused))
		case optPlotFontSizeMedium:
			val := formatInt(m.plotFontSizeMedium, plotDefaults.fontSizeMedium)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeMedium) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_medium", val, "int", focused))
		case optPlotFontSizeLarge:
			val := formatInt(m.plotFontSizeLarge, plotDefaults.fontSizeLarge)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeLarge) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_large", val, "int", focused))
		case optPlotFontSizeTitle:
			val := formatInt(m.plotFontSizeTitle, plotDefaults.fontSizeTitle)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeTitle) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_title", val, "int", focused))
		case optPlotFontSizeAnnotation:
			val := formatInt(m.plotFontSizeAnnotation, plotDefaults.fontSizeAnnotation)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeAnnotation) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_annot", val, "int", focused))
		case optPlotFontSizeLabel:
			val := formatInt(m.plotFontSizeLabel, plotDefaults.fontSizeLabel)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeLabel) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_label", val, "int", focused))
		case optPlotFontSizeYLabel:
			val := formatInt(m.plotFontSizeYLabel, plotDefaults.fontSizeYLabel)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeYLabel) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_ylabel", val, "int", focused))
		case optPlotFontSizeSuptitle:
			val := formatInt(m.plotFontSizeSuptitle, plotDefaults.fontSizeSuptitle)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeSuptitle) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_suptitle", val, "int", focused))
		case optPlotFontSizeFigureTitle:
			val := formatInt(m.plotFontSizeFigureTitle, plotDefaults.fontSizeFigureTitle)
			if m.editingNumber && m.isCurrentlyEditing(optPlotFontSizeFigureTitle) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "font_figtitle", val, "int", focused))

		case optPlotGridSpecHspace:
			val := formatFloat(m.plotGridSpecHspace, plotDefaults.gridSpecHspace, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecHspace) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hspace", val, "float", focused))
		case optPlotGridSpecWspace:
			val := formatFloat(m.plotGridSpecWspace, plotDefaults.gridSpecWspace, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecWspace) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "wspace", val, "float", focused))
		case optPlotGridSpecLeft:
			val := formatFloat(m.plotGridSpecLeft, plotDefaults.gridSpecLeft, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecLeft) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "left", val, "float [0..1]", focused))
		case optPlotGridSpecRight:
			val := formatFloat(m.plotGridSpecRight, plotDefaults.gridSpecRight, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecRight) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "right", val, "float [0..1]", focused))
		case optPlotGridSpecTop:
			val := formatFloat(m.plotGridSpecTop, plotDefaults.gridSpecTop, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecTop) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "top", val, "float [0..1]", focused))
		case optPlotGridSpecBottom:
			val := formatFloat(m.plotGridSpecBottom, plotDefaults.gridSpecBottom, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotGridSpecBottom) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bottom", val, "float [0..1]", focused))

		case optPlotAlphaGrid:
			val := formatFloat(m.plotAlphaGrid, plotDefaults.alphaGrid, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaGrid) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_grid", val, "float [0..1]", focused))
		case optPlotAlphaFill:
			val := formatFloat(m.plotAlphaFill, plotDefaults.alphaFill, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaFill) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_fill", val, "float [0..1]", focused))
		case optPlotAlphaCI:
			val := formatFloat(m.plotAlphaCI, plotDefaults.alphaCI, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaCI) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_ci", val, "float [0..1]", focused))
		case optPlotAlphaCILine:
			val := formatFloat(m.plotAlphaCILine, plotDefaults.alphaCILine, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaCILine) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_ci_line", val, "float [0..1]", focused))
		case optPlotAlphaTextBox:
			val := formatFloat(m.plotAlphaTextBox, plotDefaults.alphaTextBox, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaTextBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_text_box", val, "float [0..1]", focused))
		case optPlotAlphaViolinBody:
			val := formatFloat(m.plotAlphaViolinBody, plotDefaults.alphaViolinBody, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaViolinBody) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_violin", val, "float [0..1]", focused))
		case optPlotAlphaRidgeFill:
			val := formatFloat(m.plotAlphaRidgeFill, plotDefaults.alphaRidgeFill, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAlphaRidgeFill) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "alpha_ridge", val, "float [0..1]", focused))

		case optPlotScatterMarkerSizeSmall:
			val := formatInt(m.plotScatterMarkerSizeSmall, plotDefaults.scatterMarkerSizeSmall)
			if m.editingNumber && m.isCurrentlyEditing(optPlotScatterMarkerSizeSmall) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_ms_small", val, "int", focused))
		case optPlotScatterMarkerSizeLarge:
			val := formatInt(m.plotScatterMarkerSizeLarge, plotDefaults.scatterMarkerSizeLarge)
			if m.editingNumber && m.isCurrentlyEditing(optPlotScatterMarkerSizeLarge) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_ms_large", val, "int", focused))
		case optPlotScatterMarkerSizeDefault:
			val := formatInt(m.plotScatterMarkerSizeDefault, plotDefaults.scatterMarkerSizeDefault)
			if m.editingNumber && m.isCurrentlyEditing(optPlotScatterMarkerSizeDefault) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_ms_default", val, "int", focused))
		case optPlotScatterAlpha:
			val := formatFloat(m.plotScatterAlpha, plotDefaults.scatterAlpha, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotScatterAlpha) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_alpha", val, "float [0..1]", focused))
		case optPlotScatterEdgewidth:
			val := formatFloat(m.plotScatterEdgeWidth, plotDefaults.scatterEdgewidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotScatterEdgewidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "scatter_edgew", val, "float", focused))

		case optPlotBarAlpha:
			val := formatFloat(m.plotBarAlpha, plotDefaults.barAlpha, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotBarAlpha) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bar_alpha", val, "float [0..1]", focused))
		case optPlotBarWidth:
			val := formatFloat(m.plotBarWidth, plotDefaults.barWidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotBarWidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bar_width", val, "float", focused))
		case optPlotBarCapsize:
			val := formatInt(m.plotBarCapsize, plotDefaults.barCapsize)
			if m.editingNumber && m.isCurrentlyEditing(optPlotBarCapsize) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bar_capsize", val, "int", focused))
		case optPlotBarCapsizeLarge:
			val := formatInt(m.plotBarCapsizeLarge, plotDefaults.barCapsizeLarge)
			if m.editingNumber && m.isCurrentlyEditing(optPlotBarCapsizeLarge) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bar_capsize_lg", val, "int", focused))

		case optPlotLineWidthThin:
			val := formatFloat(m.plotLineWidthThin, plotDefaults.lineWidthThin, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineWidthThin) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_thin", val, "float", focused))
		case optPlotLineWidthStandard:
			val := formatFloat(m.plotLineWidthStandard, plotDefaults.lineWidthStandard, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineWidthStandard) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_std", val, "float", focused))
		case optPlotLineWidthThick:
			val := formatFloat(m.plotLineWidthThick, plotDefaults.lineWidthThick, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineWidthThick) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_thick", val, "float", focused))
		case optPlotLineWidthBold:
			val := formatFloat(m.plotLineWidthBold, plotDefaults.lineWidthBold, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineWidthBold) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_bold", val, "float", focused))
		case optPlotLineAlphaStandard:
			val := formatFloat(m.plotLineAlphaStandard, plotDefaults.lineAlphaStandard, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaStandard) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_std", val, "float [0..1]", focused))
		case optPlotLineAlphaDim:
			val := formatFloat(m.plotLineAlphaDim, plotDefaults.lineAlphaDim, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaDim) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_dim", val, "float [0..1]", focused))
		case optPlotLineAlphaZeroLine:
			val := formatFloat(m.plotLineAlphaZeroLine, plotDefaults.lineAlphaZeroLine, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaZeroLine) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_zero", val, "float [0..1]", focused))
		case optPlotLineAlphaFitLine:
			val := formatFloat(m.plotLineAlphaFitLine, plotDefaults.lineAlphaFitLine, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaFitLine) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_fit", val, "float [0..1]", focused))
		case optPlotLineAlphaDiagonal:
			val := formatFloat(m.plotLineAlphaDiagonal, plotDefaults.lineAlphaDiagonal, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaDiagonal) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_diag", val, "float [0..1]", focused))
		case optPlotLineAlphaReference:
			val := formatFloat(m.plotLineAlphaReference, plotDefaults.lineAlphaReference, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineAlphaReference) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_a_ref", val, "float [0..1]", focused))
		case optPlotLineRegressionWidth:
			val := formatFloat(m.plotLineRegressionWidth, plotDefaults.lineRegressionWidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineRegressionWidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_reg", val, "float", focused))
		case optPlotLineResidualWidth:
			val := formatFloat(m.plotLineResidualWidth, plotDefaults.lineResidualWidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineResidualWidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_resid", val, "float", focused))
		case optPlotLineQQWidth:
			val := formatFloat(m.plotLineQQWidth, plotDefaults.lineQQWidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotLineQQWidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "line_w_qq", val, "float", focused))

		case optPlotHistBins:
			val := formatInt(m.plotHistBins, plotDefaults.histBins)
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistBins) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_bins", val, "int", focused))
		case optPlotHistBinsBehavioral:
			val := formatInt(m.plotHistBinsBehavioral, plotDefaults.histBinsBehavioral)
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistBinsBehavioral) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_bins_beh", val, "int", focused))
		case optPlotHistBinsResidual:
			val := formatInt(m.plotHistBinsResidual, plotDefaults.histBinsResidual)
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistBinsResidual) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_bins_resid", val, "int", focused))
		case optPlotHistBinsTFR:
			val := formatInt(m.plotHistBinsTFR, plotDefaults.histBinsTFR)
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistBinsTFR) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_bins_tfr", val, "int", focused))
		case optPlotHistEdgewidth:
			val := formatFloat(m.plotHistEdgeWidth, plotDefaults.histEdgewidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistEdgewidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_edgew", val, "float", focused))
		case optPlotHistAlpha:
			val := formatFloat(m.plotHistAlpha, plotDefaults.histAlpha, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistAlpha) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_alpha", val, "float [0..1]", focused))
		case optPlotHistAlphaResidual:
			val := formatFloat(m.plotHistAlphaResidual, plotDefaults.histAlphaResidual, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistAlphaResidual) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_alpha_resid", val, "float [0..1]", focused))
		case optPlotHistAlphaTFR:
			val := formatFloat(m.plotHistAlphaTFR, plotDefaults.histAlphaTFR, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotHistAlphaTFR) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "hist_alpha_tfr", val, "float [0..1]", focused))

		case optPlotKdePoints:
			val := formatInt(m.plotKdePoints, plotDefaults.kdePoints)
			if m.editingNumber && m.isCurrentlyEditing(optPlotKdePoints) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "kde_points", val, "int", focused))
		case optPlotKdeLinewidth:
			val := formatFloat(m.plotKdeLinewidth, plotDefaults.kdeLinewidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotKdeLinewidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "kde_linew", val, "float", focused))
		case optPlotKdeAlpha:
			val := formatFloat(m.plotKdeAlpha, plotDefaults.kdeAlpha, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotKdeAlpha) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "kde_alpha", val, "float [0..1]", focused))

		case optPlotErrorbarMarkersize:
			val := formatInt(m.plotErrorbarMarkerSize, plotDefaults.errorbarMarkersize)
			if m.editingNumber && m.isCurrentlyEditing(optPlotErrorbarMarkersize) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "err_ms", val, "int", focused))
		case optPlotErrorbarCapsize:
			val := formatInt(m.plotErrorbarCapsize, plotDefaults.errorbarCapsize)
			if m.editingNumber && m.isCurrentlyEditing(optPlotErrorbarCapsize) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "err_capsize", val, "int", focused))
		case optPlotErrorbarCapsizeLarge:
			val := formatInt(m.plotErrorbarCapsizeLarge, plotDefaults.errorbarCapsizeLarge)
			if m.editingNumber && m.isCurrentlyEditing(optPlotErrorbarCapsizeLarge) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "err_capsize_lg", val, "int", focused))

		case optPlotTextStatsX:
			val := formatFloat(m.plotTextStatsX, plotDefaults.textStatsX, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextStatsX) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_stats_x", val, "float", focused))
		case optPlotTextStatsY:
			val := formatFloat(m.plotTextStatsY, plotDefaults.textStatsY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextStatsY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_stats_y", val, "float", focused))
		case optPlotTextPvalueX:
			val := formatFloat(m.plotTextPvalueX, plotDefaults.textPvalueX, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextPvalueX) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_p_x", val, "float", focused))
		case optPlotTextPvalueY:
			val := formatFloat(m.plotTextPvalueY, plotDefaults.textPvalueY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextPvalueY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_p_y", val, "float", focused))
		case optPlotTextBootstrapX:
			val := formatFloat(m.plotTextBootstrapX, plotDefaults.textBootstrapX, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextBootstrapX) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_boot_x", val, "float", focused))
		case optPlotTextBootstrapY:
			val := formatFloat(m.plotTextBootstrapY, plotDefaults.textBootstrapY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextBootstrapY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_boot_y", val, "float", focused))
		case optPlotTextChannelAnnotationX:
			val := formatFloat(m.plotTextChannelAnnotationX, plotDefaults.textChannelAnnotationX, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextChannelAnnotationX) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_chan_x", val, "float", focused))
		case optPlotTextChannelAnnotationY:
			val := formatFloat(m.plotTextChannelAnnotationY, plotDefaults.textChannelAnnotationY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextChannelAnnotationY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_chan_y", val, "float", focused))
		case optPlotTextTitleY:
			val := formatFloat(m.plotTextTitleY, plotDefaults.textTitleY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextTitleY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_title_y", val, "float", focused))
		case optPlotTextResidualQcTitleY:
			val := formatFloat(m.plotTextResidualQcTitleY, plotDefaults.textResidualQcTitleY, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTextResidualQcTitleY) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "text_residqc_y", val, "float", focused))

		case optPlotValidationMinBinsForCalibration:
			val := formatInt(m.plotValidationMinBinsForCalibration, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotValidationMinBinsForCalibration) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "min_bins_cal", val, "int", focused))
		case optPlotValidationMaxBinsForCalibration:
			val := formatInt(m.plotValidationMaxBinsForCalibration, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotValidationMaxBinsForCalibration) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "max_bins_cal", val, "int", focused))
		case optPlotValidationSamplesPerBin:
			val := formatInt(m.plotValidationSamplesPerBin, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotValidationSamplesPerBin) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "samples_per_bin", val, "int", focused))
		case optPlotValidationMinRoisForFDR:
			val := formatInt(m.plotValidationMinRoisForFDR, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotValidationMinRoisForFDR) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "min_rois_fdr", val, "int", focused))
		case optPlotValidationMinPvaluesForFDR:
			val := formatInt(m.plotValidationMinPvaluesForFDR, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotValidationMinPvaluesForFDR) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "min_p_fdr", val, "int", focused))

		// Plot overrides & selections (existing)
		case optPlotTopomapContours:
			val := formatInt(m.plotTopomapContours, plotDefaults.topomapContours)
			if m.editingNumber && m.isCurrentlyEditing(optPlotTopomapContours) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "topomap_contours", val, "int", focused))
		case optPlotTopomapColorbarFraction:
			val := formatFloat(m.plotTopomapColorbarFraction, plotDefaults.topomapColorbarFraction, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTopomapColorbarFraction) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "cbar_fraction", val, "float [0..1]", focused))
		case optPlotTopomapColorbarPad:
			val := formatFloat(m.plotTopomapColorbarPad, plotDefaults.topomapColorbarPad, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTopomapColorbarPad) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "cbar_pad", val, "float [0..1]", focused))
		case optPlotTopomapDiffAnnotation:
			lines = append(lines, valueLine(opt, "diff_annotate", triState(m.plotTopomapDiffAnnotation), "tri-state", focused))
		case optPlotTopomapAnnotateDescriptive:
			lines = append(lines, valueLine(opt, "annotate_desc", triState(m.plotTopomapAnnotateDesc), "tri-state", focused))
		case optPlotTopomapSigMaskLinewidth:
			val := formatFloat(m.plotTopomapSigMaskLinewidth, plotDefaults.topomapSigMaskLinewidth, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTopomapSigMaskLinewidth) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "sig_mask_lw", val, "float", focused))
		case optPlotTopomapSigMaskMarkersize:
			val := formatFloat(m.plotTopomapSigMaskMarkerSize, plotDefaults.topomapSigMaskMarkerSize, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTopomapSigMaskMarkersize) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "sig_mask_ms", val, "float", focused))

		case optPlotTFRLogBase:
			val := formatFloat(m.plotTFRLogBase, plotDefaults.tfrLogBase, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTFRLogBase) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "tfr_log_base", val, "float", focused))
		case optPlotTFRPercentageMultiplier:
			val := formatFloat(m.plotTFRPercentageMultiplier, plotDefaults.tfrPercentageMultiplier, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotTFRPercentageMultiplier) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "tfr_pct_mult", val, "float", focused))

		case optPlotRoiWidthPerBand:
			val := formatFloat(m.plotRoiWidthPerBand, plotDefaults.roiWidthPerBand, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotRoiWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "roi_w_per_band", val, "float", focused))
		case optPlotRoiWidthPerMetric:
			val := formatFloat(m.plotRoiWidthPerMetric, plotDefaults.roiWidthPerMetric, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotRoiWidthPerMetric) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "roi_w_per_metric", val, "float", focused))
		case optPlotRoiHeightPerRoi:
			val := formatFloat(m.plotRoiHeightPerRoi, plotDefaults.roiHeightPerRoi, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotRoiHeightPerRoi) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "roi_h_per_roi", val, "float", focused))
		case optPlotPowerWidthPerBand:
			val := formatFloat(m.plotPowerWidthPerBand, plotDefaults.powerWidthPerBand, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotPowerWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "power_w_per_band", val, "float", focused))
		case optPlotPowerHeightPerSegment:
			val := formatFloat(m.plotPowerHeightPerSegment, plotDefaults.powerHeightPerSegment, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotPowerHeightPerSegment) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "power_h_per_seg", val, "float", focused))
		case optPlotItpcWidthPerBin:
			val := formatFloat(m.plotItpcWidthPerBin, plotDefaults.itpcWidthPerBin, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotItpcWidthPerBin) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "itpc_w_per_bin", val, "float", focused))
		case optPlotItpcHeightPerBand:
			val := formatFloat(m.plotItpcHeightPerBand, plotDefaults.itpcHeightPerBand, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotItpcHeightPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "itpc_h_per_band", val, "float", focused))
		case optPlotItpcWidthPerBandBox:
			val := formatFloat(m.plotItpcWidthPerBandBox, plotDefaults.itpcWidthPerBandBox, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotItpcWidthPerBandBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "itpc_w_box", val, "float", focused))
		case optPlotItpcHeightBox:
			val := formatFloat(m.plotItpcHeightBox, plotDefaults.itpcHeightBox, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotItpcHeightBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "itpc_h_box", val, "float", focused))
		case optPlotPacWidthPerRoi:
			val := formatFloat(m.plotPacWidthPerRoi, plotDefaults.pacWidthPerRoi, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotPacWidthPerRoi) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "pac_w_per_roi", val, "float", focused))
		case optPlotPacHeightBox:
			val := formatFloat(m.plotPacHeightBox, plotDefaults.pacHeightBox, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotPacHeightBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "pac_h_box", val, "float", focused))
		case optPlotAperiodicWidthPerColumn:
			val := formatFloat(m.plotAperiodicWidthPerColumn, plotDefaults.aperiodicWidthPerColumn, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAperiodicWidthPerColumn) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "aper_w_per_col", val, "float", focused))
		case optPlotAperiodicHeightPerRow:
			val := formatFloat(m.plotAperiodicHeightPerRow, plotDefaults.aperiodicHeightPerRow, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotAperiodicHeightPerRow) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "aper_h_per_row", val, "float", focused))
		case optPlotAperiodicNPerm:
			val := formatInt(m.plotAperiodicNPerm, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotAperiodicNPerm) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "aper_n_perm", val, "int", focused))
		case optPlotQualityWidthPerPlot:
			val := formatFloat(m.plotQualityWidthPerPlot, plotDefaults.qualityWidthPerPlot, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityWidthPerPlot) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_w", val, "float", focused))
		case optPlotQualityHeightPerPlot:
			val := formatFloat(m.plotQualityHeightPerPlot, plotDefaults.qualityHeightPerPlot, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityHeightPerPlot) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_h", val, "float", focused))
		case optPlotQualityDistributionNCols:
			val := formatInt(m.plotQualityDistributionNCols, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityDistributionNCols) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_n_cols", val, "int", focused))
		case optPlotQualityDistributionMaxFeatures:
			val := formatInt(m.plotQualityDistributionMaxFeatures, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityDistributionMaxFeatures) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_max_feat", val, "int", focused))
		case optPlotQualityOutlierZThreshold:
			val := formatFloat(m.plotQualityOutlierZThreshold, 0, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityOutlierZThreshold) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_z_thr", val, "float", focused))
		case optPlotQualityOutlierMaxFeatures:
			val := formatInt(m.plotQualityOutlierMaxFeatures, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityOutlierMaxFeatures) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_out_max_feat", val, "int", focused))
		case optPlotQualityOutlierMaxTrials:
			val := formatInt(m.plotQualityOutlierMaxTrials, 0)
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualityOutlierMaxTrials) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_out_max_trials", val, "int", focused))
		case optPlotQualitySnrThresholdDb:
			val := formatFloat(m.plotQualitySnrThresholdDb, 0, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotQualitySnrThresholdDb) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "quality_snr_db", val, "float", focused))
		case optPlotComplexityWidthPerMeasure:
			val := formatFloat(m.plotComplexityWidthPerMeasure, plotDefaults.complexityWidthPerMeasure, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotComplexityWidthPerMeasure) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "comp_w_per_meas", val, "float", focused))
		case optPlotComplexityHeightPerSegment:
			val := formatFloat(m.plotComplexityHeightPerSegment, plotDefaults.complexityHeightPerSegment, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotComplexityHeightPerSegment) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "comp_h_per_seg", val, "float", focused))
		case optPlotConnectivityWidthPerCircle:
			val := formatFloat(m.plotConnectivityWidthPerCircle, plotDefaults.connectivityWidthPerCircle, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotConnectivityWidthPerCircle) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "conn_w_circle", val, "float", focused))
		case optPlotConnectivityWidthPerBand:
			val := formatFloat(m.plotConnectivityWidthPerBand, plotDefaults.connectivityWidthPerBand, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotConnectivityWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "conn_w_band", val, "float", focused))
		case optPlotConnectivityHeightPerMeasure:
			val := formatFloat(m.plotConnectivityHeightPerMeasure, plotDefaults.connectivityHeightPerMeasure, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotConnectivityHeightPerMeasure) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "conn_h_meas", val, "float", focused))
		case optPlotConnectivityCircleTopFraction:
			val := formatFloat(m.plotConnectivityCircleTopFraction, 0, "%.4f")
			if m.editingNumber && m.isCurrentlyEditing(optPlotConnectivityCircleTopFraction) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "conn_top_frac", val, "float [0..1]", focused))
		case optPlotConnectivityCircleMinLines:
			val := formatInt(m.plotConnectivityCircleMinLines, plotDefaults.connectivityCircleMinLines)
			if m.editingNumber && m.isCurrentlyEditing(optPlotConnectivityCircleMinLines) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "conn_min_lines", val, "int", focused))

		case optPlotConnectivityMeasures:
			val := formatString(m.getTextFieldValue(textFieldPlotConnectivityMeasures), "")
			if m.editingText && m.editingTextField == textFieldPlotConnectivityMeasures {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "connectivity_measures", val, "space-separated (e.g. aec wpli)", focused))

		case optPlotPacPairs:
			val := formatString(m.plotPacPairsSpec, "")
			if m.editingText && m.editingTextField == textFieldPlotPacPairs {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "pac_pairs", val, "space-separated", focused))
		case optPlotSpectralMetrics:
			val := formatString(m.plotSpectralMetricsSpec, "")
			if m.editingText && m.editingTextField == textFieldPlotSpectralMetrics {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "spectral_metrics", val, "space-separated", focused))
		case optPlotBurstsMetrics:
			val := formatString(m.plotBurstsMetricsSpec, "")
			if m.editingText && m.editingTextField == textFieldPlotBurstsMetrics {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "bursts_metrics", val, "space-separated", focused))
		case optPlotAsymmetryStat:
			val := formatString(m.plotAsymmetryStatSpec, "index")
			if m.editingText && m.editingTextField == textFieldPlotAsymmetryStat {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "asymmetry_stat", val, "e.g. index", focused))
		case optPlotTemporalTimeBins:
			val := formatString(m.plotTemporalTimeBinsSpec, "")
			if m.editingText && m.editingTextField == textFieldPlotTemporalTimeBins {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "temporal_bins", val, "space-separated", focused))
		case optPlotTemporalTimeLabels:
			val := formatString(m.plotTemporalTimeLabelsSpec, "")
			if m.editingText && m.editingTextField == textFieldPlotTemporalTimeLabels {
				val = m.textBuffer + "█"
			}
			lines = append(lines, valueLine(opt, "temporal_labels", val, "space-separated", focused))

		default:
			// Fallback: keep UI robust even if new options are added without
			// wiring; still show a line so the list remains navigable.
			lines = append(lines, valueLine(opt, fmt.Sprintf("opt_%d", opt), "(unwired)", "Space to edit (TODO)", focused))
		}
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = 40
	}
	// Overhead: header(4) + title(2) + help(2) + footer(2) = 10
	maxLines := effectiveHeight - 10
	if maxLines < 8 {
		maxLines = 8
	}

	start := 0
	end := len(lines)
	showScroll := false
	if len(lines) > maxLines {
		showScroll = true
		start = m.advancedOffset
		if start < 0 {
			start = 0
		}
		if start > len(lines)-maxLines {
			start = len(lines) - maxLines
		}
		if start < 0 {
			start = 0
		}
		end = start + maxLines
		if end > len(lines) {
			end = len(lines)
		}
	}

	if showScroll && start > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", start)) + "\n")
	}
	for i := start; i < end; i++ {
		b.WriteString(lines[i].text + "\n")
	}
	if showScroll && end < len(lines) {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", len(lines)-end)) + "\n")
	}

	return b.String()
}
