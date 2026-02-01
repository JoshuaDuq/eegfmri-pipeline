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

	b.WriteString(styles.SectionTitleStyle.Render("Plots") + "\n\n")

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

	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(visibleItems))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	currentGroup := ""
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		if plot.Group != currentGroup {
			if currentGroup != "" {
				b.WriteString("\n")
			}
			b.WriteString(groupStyle.Render(styles.SelectedMark+" "+strings.ToUpper(plot.Group)) + "\n")
			currentGroup = plot.Group
		}

		isSelected := m.plotSelected[i]
		isFocused := i == m.plotCursor
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		idStyle := lipgloss.NewStyle().Foreground(styles.TextDim).PaddingLeft(1)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			idStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		b.WriteString(checkbox + idStyle.Render(plot.ID) + nameStyle.Render(" — "+plot.Name) + "\n")
	}

	if m.plotCursor >= 0 && m.plotCursor < len(m.plotItems) {
		plot := m.plotItems[m.plotCursor]
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.Repeat("─", 50)) + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Data requirements:") + "\n")

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
		if len(reqs) > 0 {
			reqStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
			b.WriteString(reqStyle.Render(strings.Join(reqs, ", ")) + "\n")
		} else {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).PaddingLeft(2).Render("Base epochs only") + "\n")
		}

		readyCount, totalCount, _ := m.plotAvailabilitySummary(plot)
		if totalCount > 0 {
			statusColor := styles.Success
			if readyCount < totalCount {
				statusColor = styles.Warning
			}
			statusStyle := lipgloss.NewStyle().Foreground(statusColor).PaddingLeft(2)
			b.WriteString(statusStyle.Render(fmt.Sprintf("%d/%d subjects ready", readyCount, totalCount)) + "\n")
		}
	}

	return b.String()
}

func (m Model) renderPlotConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Plot output") + "\n\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Configure output formats and resolution. Use Space to toggle/cycle.\n\n"))

	options := m.getPlotConfigOptions()
	labelWidth := 16

	for i, opt := range options {
		isFocused := i == m.plotConfigCursor
		cursor := ""
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(labelWidth)
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
		b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value) + "\n")
	}

	return b.String()
}

func (m Model) renderTimeRange() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Time range") + "\n\n")

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
			for _, tr := range m.TimeRanges {
				startVal := parseFloat(tr.Tmin, tmin)
				endVal := parseFloat(tr.Tmax, tmax)
				startPos := int(((startVal - tmin) / epochDuration) * float64(diagramWidth-1))
				endPos := int(((endVal - tmin) / epochDuration) * float64(diagramWidth-1))
				if startPos < 0 {
					startPos = 0
				}
				if endPos >= diagramWidth {
					endPos = diagramWidth - 1
				}
				for i := startPos; i <= endPos; i++ {
					timelineChars[i] = '█'
				}
			}
			timeline := string(timelineChars)
			diagram.WriteString(lipgloss.NewStyle().Foreground(styles.Primary).Render(timeline))
			diagram.WriteString("\n")
			diagram.WriteString(fmt.Sprintf("  %-*s%*s\n",
				diagramWidth/2, lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmin)),
				diagramWidth/2, lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmax))))
			b.WriteString(diagram.String() + "\n")
		}
	}

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Define time ranges to be computed separately.") + "\n" +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  [+] Add range  [D] Delete range  [Space/Enter] Edit range") + "\n\n")

	nameWidth := 15
	valWidth := 10
	headerStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true)
	header := fmt.Sprintf("  %-*s %-*s %-*s", nameWidth, headerStyle.Render("Range name"), valWidth, headerStyle.Render("Start (s)"), valWidth, headerStyle.Render("End (s)"))
	b.WriteString(header + "\n")
	b.WriteString("  " + strings.Repeat("─", nameWidth+valWidth*2+2) + "\n")

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
	b.WriteString(styles.SectionTitleStyle.Render("Feature plots") + "\n\n")
	b.WriteString(
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Select which plots to execute within selected Feature suites (e.g., Power).\n",
		) +
			lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
				"  Use ↑/↓ to navigate, Space to toggle, and A/N to select all/none.\n\n",
			),
	)

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
	statusIndicator := lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	if selectedCount > 0 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	}
	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", selectedCount, len(items)),
	))
	if selectedCount == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	type listLine struct {
		isHeader bool
		text     string
	}
	var lines []listLine
	cursorLineIdx := 0
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	currentCategory := ""
	for i, p := range items {
		if p.Category != currentCategory {
			lines = append(lines, listLine{isHeader: true, text: groupStyle.Render(" " + strings.ToUpper(p.Category) + " ")})
			currentCategory = p.Category
		}
		isFocused := i == m.featurePlotterCursor
		if isFocused {
			cursorLineIdx = len(lines)
		}
		isSelected := m.featurePlotterSelected[p.ID]
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		lines = append(lines, listLine{isHeader: false, text: checkbox + nameStyle.Render(p.Name)})
	}

	layout := styles.CalculateListLayout(m.height, cursorLineIdx, len(lines), 10)
	if layout.ShowScrollUp {
		b.WriteString(styles.RenderScrollUpIndicator(layout.StartIdx) + "\n")
	}
	for i := layout.StartIdx; i < layout.EndIdx; i++ {
		b.WriteString(" " + lines[i].text + "\n")
	}
	if layout.ShowScrollDn {
		remaining := len(lines) - layout.EndIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	return b.String()
}
