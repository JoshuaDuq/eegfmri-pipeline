// Plot selection, plot config, time range, feature plotter selection.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderPlotSelection() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Plots", m.contentWidth) + "\n")

	visibleItems := []int{}
	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		visibleItems = append(visibleItems, i)
	}

	count := 0
	for _, idx := range visibleItems {
		if m.plotSelected[idx] {
			count++
		}
	}

	b.WriteString(styles.RenderStatusCount(count, len(visibleItems), "selected"))
	b.WriteString("\n\n")

	currentGroup := ""
	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		if plot.Group != currentGroup {
			if currentGroup != "" {
				b.WriteString("\n")
			}
			b.WriteString(styles.RenderDimSectionLabel(strings.ToUpper(plot.Group)) + "\n")
			currentGroup = plot.Group
		}

		isSelected := m.plotSelected[i]
		isFocused := i == m.plotCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		idStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			idStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + idStyle.Render(plot.ID) + nameStyle.Render(" \u2014 "+plot.Name)
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	if m.plotCursor >= 0 && m.plotCursor < len(m.plotItems) {
		plot := m.plotItems[m.plotCursor]
		b.WriteString("\n" + styles.RenderDivider(50) + "\n")
		b.WriteString(styles.RenderDimSectionLabel("Requirements") + "\n")

		reqs := []string{}
		if plot.RequiresEpochs {
			reqs = append(reqs, "epochs")
		}
		if plot.RequiresFeatures {
			reqs = append(reqs, "features")
		}
		if plot.RequiresStats {
			reqs = append(reqs, "stats")
		}
		if len(plot.RequiredFiles) > 0 {
			reqs = append(reqs, plot.RequiredFiles...)
		}
		reqStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
		if len(reqs) > 0 {
			b.WriteString("  " + reqStyle.Render(strings.Join(reqs, ", ")) + "\n")
		} else {
			b.WriteString("  " + reqStyle.Render("base epochs only") + "\n")
		}

		readyCount, totalCount, _ := m.plotAvailabilitySummary(plot)
		if totalCount > 0 {
			statusColor := styles.Success
			if readyCount < totalCount {
				statusColor = styles.Warning
			}
			b.WriteString("  " + lipgloss.NewStyle().Foreground(statusColor).Render(fmt.Sprintf("%d/%d subjects ready", readyCount, totalCount)) + "\n")
		}
	}

	return b.String()
}

func (m Model) renderPlotConfig() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Plot output", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Space: toggle/cycle") + "\n\n")

	options := m.getPlotConfigOptions()
	labelWidth := 16

	for i, opt := range options {
		isFocused := i == m.plotConfigCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			valueStyle = valueStyle.Foreground(styles.Accent).Bold(true)
		}

		var label, value string
		switch opt {
		case optPlotPNG:
			label = "PNG"
			value = m.boolToOnOff(m.plotFormatSelected["png"])
		case optPlotSVG:
			label = "SVG"
			value = m.boolToOnOff(m.plotFormatSelected["svg"])
		case optPlotPDF:
			label = "PDF"
			value = m.boolToOnOff(m.plotFormatSelected["pdf"])
		case optPlotDPI:
			label = "Figure DPI"
			if m.plotDpiIndex >= 0 && m.plotDpiIndex < len(m.plotDpiOptions) {
				value = fmt.Sprintf("%d", m.plotDpiOptions[m.plotDpiIndex])
			} else {
				value = "default"
			}
		case optPlotSaveDPI:
			label = "Savefig DPI"
			if m.plotSavefigDpiIndex >= 0 && m.plotSavefigDpiIndex < len(m.plotDpiOptions) {
				value = fmt.Sprintf("%d", m.plotDpiOptions[m.plotSavefigDpiIndex])
			} else {
				value = "default"
			}
		case optPlotSharedColorbar:
			label = "Shared Colorbar"
			value = m.boolToOnOff(m.plotSharedColorbar)
		case optPlotOverwrite:
			label = "Overwrite"
			val := "OFF"
			if m.plotOverwrite != nil && *m.plotOverwrite {
				val = "ON"
			}
			value = val
		}
		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(value), "", labelWidth, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderTimeRange() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Time range", m.contentWidth) + "\n")

	var tmin, tmax float64
	hasMetadata := false
	for _, s := range m.subjects {
		if len(s.EpochMetadata) > 0 {
			tmin = s.EpochMetadata["tmin"]
			tmax = s.EpochMetadata["tmax"]
			hasMetadata = true
			break
		}
	}

	if hasMetadata {
		metaStyle := lipgloss.NewStyle().Foreground(styles.Accent).Italic(true)
		b.WriteString(fmt.Sprintf("  Epoch Length: %s to %s (seconds)\n\n",
			metaStyle.Render(fmt.Sprintf("%.2f", tmin)),
			metaStyle.Render(fmt.Sprintf("%.2f", tmax))))

		diagramWidth := 50
		epochDuration := tmax - tmin
		if epochDuration > 0 {
			diagram := strings.Builder{}
			diagram.WriteString("  ")
			timelineChars := make([]rune, diagramWidth)
			for i := range timelineChars {
				timelineChars[i] = '─'
			}
			mapToPos := func(value float64, inclusiveEnd bool) int {
				scaled := ((value - tmin) / epochDuration) * float64(diagramWidth-1)
				if inclusiveEnd {
					// Keep right endpoint visually inclusive when value hits epoch tmax.
					scaled += 0.999999
				}
				pos := int(scaled)
				if pos < 0 {
					return 0
				}
				if pos >= diagramWidth {
					return diagramWidth - 1
				}
				return pos
			}
			for _, tr := range m.TimeRanges {
				startVal := parseFloat(tr.Tmin, tmin)
				endVal := parseFloat(tr.Tmax, tmax)
				startPos := mapToPos(startVal, false)
				endPos := mapToPos(endVal, true)
				for i := startPos; i <= endPos; i++ {
					timelineChars[i] = '█'
				}
			}
			timeline := string(timelineChars)
			diagram.WriteString(lipgloss.NewStyle().Foreground(styles.Primary).Render(timeline))
			diagram.WriteString("\n")
			leftLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmin))
			rightLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmax))
			padding := diagramWidth - lipgloss.Width(leftLabel) - lipgloss.Width(rightLabel)
			if padding < 1 {
				padding = 1
			}
			diagram.WriteString("  " + leftLabel + strings.Repeat(" ", padding) + rightLabel + "\n")
			b.WriteString(diagram.String() + "\n")
		}
	}

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  +: add  D: delete  Space/Enter: edit") + "\n\n")

	nameWidth := 15
	valWidth := 10
	headerStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true)
	header := fmt.Sprintf("  %-*s %-*s %-*s", nameWidth, headerStyle.Render("Range name"), valWidth, headerStyle.Render("Start (s)"), valWidth, headerStyle.Render("End (s)"))
	b.WriteString(header + "\n")
	b.WriteString("  " + styles.RenderDivider(nameWidth+valWidth*2+2) + "\n")

	if len(m.TimeRanges) == 0 {
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render("  No time ranges defined. Press [A] to add one.") + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render("  Note: 'baseline' is required for normalization (ERDS, log-ratio).") + "\n")
	}

	for i, tr := range m.TimeRanges {
		isFocused := i == m.timeRangeCursor
		isEditing := i == m.editingRangeIdx
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		tminStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		tmaxStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isEditing {
			switch m.editingField {
			case 0:
				nameStyle = nameStyle.Foreground(styles.Accent).Underline(true)
			case 1:
				tminStyle = tminStyle.Foreground(styles.Accent).Underline(true)
			case 2:
				tmaxStyle = tmaxStyle.Foreground(styles.Accent).Underline(true)
			}
		} else if isFocused {
			nameStyle = nameStyle.Foreground(styles.Primary).Bold(true)
		}

		nameVal := tr.Name
		if nameVal == "" {
			nameVal = "(none)"
		}
		if isEditing && m.editingField == 0 {
			nameVal += "█"
		}
		tminVal := tr.Tmin
		if tminVal == "" {
			tminVal = "default"
		}
		if isEditing && m.editingField == 1 {
			tminVal += "█"
		}
		tmaxVal := tr.Tmax
		if tmaxVal == "" {
			tmaxVal = "default"
		}
		if isEditing && m.editingField == 2 {
			tmaxVal += "█"
		}
		row := fmt.Sprintf("%s%-*s %-*s %-*s",
			cursor,
			nameWidth, nameStyle.Render(nameVal),
			valWidth, tminStyle.Render(tminVal),
			valWidth, tmaxStyle.Render(tmaxVal),
		)
		b.WriteString(row + "\n")
	}

	return b.String()
}

func (m Model) renderFeaturePlotterSelection() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Feature plots", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
		Render("  Space: toggle  A/N: all/none") + "\n\n")

	categories := m.selectedFeaturePlotterCategories()
	if len(categories) == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("  No feature suites selected.\n"))
		return b.String()
	}
	if m.featurePlotters == nil {
		if strings.TrimSpace(m.featurePlotterError) != "" {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("  Failed to load feature plots: " + m.featurePlotterError + "\n"))
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  Proceeding will run all plots.\n"))
		} else {
			b.WriteString("  " + m.plotLoadingSpinner.View() + "\n")
		}
		return b.String()
	}

	items := m.featurePlotterItems()
	if len(items) == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("  No feature plots available.\n"))
		return b.String()
	}

	selectedCount := 0
	for _, p := range items {
		if m.featurePlotterSelected[p.ID] {
			selectedCount++
		}
	}
	b.WriteString(styles.RenderStatusCount(selectedCount, len(items), "selected"))
	b.WriteString("\n\n")

	type listLine struct {
		isHeader bool
		text     string
	}
	var lines []listLine
	cursorLineIdx := 0
	currentCategory := ""
	for i, p := range items {
		if p.Category != currentCategory {
			lines = append(lines, listLine{isHeader: true, text: styles.RenderDimSectionLabel(strings.ToUpper(p.Category))})
			currentCategory = p.Category
		}
		isFocused := i == m.featurePlotterCursor
		if isFocused {
			cursorLineIdx = len(lines)
		}
		isSelected := m.featurePlotterSelected[p.ID]
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		lines = append(lines, listLine{isHeader: false, text: styles.TruncateLine(cursor+checkbox+" "+nameStyle.Render(p.Name), m.contentWidth)})
	}

	layout := styles.CalculateListLayout(m.height, cursorLineIdx, len(lines), 10)
	if layout.ShowScrollUp {
		b.WriteString(styles.RenderScrollUpIndicator(layout.StartIdx) + "\n")
	}
	for i := layout.StartIdx; i < layout.EndIdx; i++ {
		b.WriteString(lines[i].text + "\n")
	}
	if layout.ShowScrollDn {
		remaining := len(lines) - layout.EndIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	return b.String()
}
