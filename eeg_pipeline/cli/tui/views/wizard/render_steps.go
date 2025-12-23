package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// Step Rendering
///////////////////////////////////////////////////////////////////

func (m Model) renderConfirmation() string {
	content := strings.Builder{}

	accentFrames := []string{"◆", "◇", "◆", "◈"}
	accent := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Render(accentFrames[(m.ticker/3)%len(accentFrames)])

	content.WriteString(accent + " " + styles.SectionTitleStyle.Render(" CONFIRM EXECUTION ") + "\n\n")

	content.WriteString(lipgloss.NewStyle().Foreground(styles.Text).Render("You are about to execute:") + "\n\n")

	cmdStyle := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Padding(0, 2)
	content.WriteString("  " + cmdStyle.Render(m.BuildCommand()) + "\n\n")

	if len(m.subjectSelected) > 0 {
		selectedCount := 0
		for _, sel := range m.subjectSelected {
			if sel {
				selectedCount++
			}
		}
		if selectedCount > 0 {
			timeInfo := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
				Render(fmt.Sprintf("Processing %d subjects", selectedCount))
			content.WriteString("  " + timeInfo + "\n\n")
		}
	}

	content.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render("Proceed with execution?") + "\n\n")

	yesBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(styles.Success).
		Bold(true).
		Padding(0, 2).
		Render("▶ [Y] Execute")

	noBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FFFFFF")).
		Background(styles.Error).
		Bold(true).
		Padding(0, 2).
		Render("✗ [N] Cancel")

	actions := lipgloss.JoinHorizontal(lipgloss.Left, yesBtn, "   ", noBtn)
	content.WriteString("  " + actions)

	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2)

	return boxStyle.Render(content.String())
}

func (m Model) renderModeSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" MODE SELECTION ") + "\n\n")

	for i, opt := range m.modeOptions {
		isSelected := i == m.modeIndex

		radio := styles.RenderRadio(isSelected, isSelected)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isSelected {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(radio + nameStyle.Render(opt))

		if i < len(m.modeDescriptions) {
			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + m.modeDescriptions[i])
			b.WriteString(desc)
		}

		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderComputationSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" COMPUTATIONS ") + "\n\n")

	count := 0
	for _, sel := range m.computationSelected {
		if sel {
			count++
		}
	}

	// Inline validation indicator
	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}

	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.computations))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	for i, comp := range m.computations {
		isSelected := m.computationSelected[i]
		isFocused := i == m.computationCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(comp.Name))

		desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + comp.Description)
		b.WriteString(desc + "\n")
	}

	return b.String()
}

func (m Model) renderCategorySelection() string {
	var b strings.Builder
	title := " FEATURE CATEGORIES "
	if m.CurrentStep == types.StepSelectPlotCategories {
		title = " PLOT CATEGORIES "
	}
	b.WriteString(styles.SectionTitleStyle.Render(title) + "\n\n")

	count := 0
	for _, sel := range m.selected {
		if sel {
			count++
		}
	}

	// Inline validation indicator
	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}

	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.categories))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	for i, cat := range m.categories {
		isSelected := m.selected[i]
		isFocused := i == m.categoryIndex

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(cat))

		if i < len(m.categoryDescs) {
			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + m.categoryDescs[i])
			b.WriteString(desc)
		}

		if m.featureAvailability != nil {
			if m.featureAvailability[cat] {
				timestamp := m.featureLastModified[cat]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s]", relTime))
					b.WriteString(avail)
				} else {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render("  [DATA]")
					b.WriteString(avail)
				}
			} else {
				unavail := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  [NO DATA]")
				b.WriteString(unavail)
			}
		}

		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderBandSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" FREQUENCY BANDS ") + "\n\n")

	count := 0
	for _, sel := range m.bandSelected {
		if sel {
			count++
		}
	}

	// Inline validation indicator
	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}

	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.bands))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	for i, band := range m.bands {
		isSelected := m.bandSelected[i]
		isFocused := i == m.bandCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(band.Name))

		desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + band.Description)
		b.WriteString(desc)

		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderSpatialSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" SPATIAL AGGREGATION ") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Select how to aggregate features spatially.\n\n"))

	count := 0
	for _, sel := range m.spatialSelected {
		if sel {
			count++
		}
	}
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("  %d of %d selected\n\n", count, len(spatialModes))))

	for i, mode := range spatialModes {
		isSelected := m.spatialSelected[i]
		isFocused := i == m.spatialCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(mode.Name))

		desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + mode.Description)
		b.WriteString(desc + "\n")
	}

	return b.String()
}

func (m Model) renderFeatureFileSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" FEATURE FILES ") + "\n\n")

	instruction := "  Select which feature files to load for analysis.\n"
	if m.Pipeline == types.PipelineCombineFeatures {
		instruction = "  Select which features to aggregate into features_all.tsv.\n"
	}

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		instruction) +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Features marked in green are available; red indicates missing data.\n\n"))

	count := 0
	availableCount := 0
	for key, sel := range m.featureFileSelected {
		if sel {
			count++
		}
		if m.featureAvailability != nil && m.featureAvailability[key] {
			availableCount++
		}
	}
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("  %d of %d selected", count, len(m.featureFiles))))
	if m.featureAvailability != nil {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
			fmt.Sprintf(" | %d available", availableCount)))
	}
	b.WriteString("\n\n")

	for i, file := range m.featureFiles {
		isSelected := m.featureFileSelected[file.Key]
		isFocused := i == m.featureFileCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(file.Name))

		desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + file.Description)
		b.WriteString(desc)

		// Show availability status if we have feature availability data
		if m.featureAvailability != nil {
			if m.featureAvailability[file.Key] {
				// Feature is available - show green checkmark with timestamp if available
				timestamp := m.featureLastModified[file.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s %s]", styles.CheckMark, relTime))
					b.WriteString(avail)
				} else {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + styles.CheckMark + " DATA]")
					b.WriteString(avail)
				}
			} else {
				// Feature not available - show red X
				unavail := lipgloss.NewStyle().Foreground(styles.Error).Render("  [" + styles.CrossMark + " MISSING]")
				b.WriteString(unavail)
			}
		}

		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderPlotSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" PLOT SELECTION ") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Select the plots to generate. Details for the focused item are shown below.\n") +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Use ↑/↓ to navigate, Space to toggle, and A/N to select all/none.\n\n"))

	// Calculate counts
	count := 0
	for _, sel := range m.plotSelected {
		if sel {
			count++
		}
	}

	visibleItems := []int{}
	for i, plot := range m.plotItems {
		if m.IsPlotCategorySelected(plot.Group) {
			visibleItems = append(visibleItems, i)
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

	// Determine visible area for scrolling
	// Total height minus overhead (approx 18-20 lines)
	maxLines := m.height - 18
	if maxLines < 10 {
		maxLines = 10 // Minimum window
	}

	// Group-aware rendering
	currentGroup := ""
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	// Collect all lines to render for the list
	type listLine struct {
		isHeader bool
		text     string
		itemIdx  int // -1 if header
	}
	var lines []listLine

	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}

		if plot.Group != currentGroup {
			lines = append(lines, listLine{true, groupStyle.Render(" " + strings.ToUpper(plot.Group) + " "), -1})
			currentGroup = plot.Group
		}

		isSelected := m.plotSelected[i]
		isFocused := i == m.plotCursor
		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		lines = append(lines, listLine{false, checkbox + nameStyle.Render(plot.Name), i})
	}

	// Ensure offset is valid (don't mutate here, just use it safely)
	offset := m.plotOffset
	if offset > len(lines)-maxLines && len(lines) > maxLines {
		offset = len(lines) - maxLines
	}
	if offset < 0 {
		offset = 0
	}

	// Render the windowed list
	if offset > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ... %d more items above ...", offset)) + "\n")
	} else {
		b.WriteString("\n")
	}

	for i := offset; i < len(lines) && i < offset+maxLines; i++ {
		b.WriteString(" " + lines[i].text + "\n")
	}

	// More indicator
	if offset+maxLines < len(lines) {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ... %d more items below ...", len(lines)-(offset+maxLines))) + "\n")
	} else {
		b.WriteString("\n")
	}

	// Details Pane (for focused item)
	b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.Repeat("─", 60)) + "\n")

	plot := m.plotItems[m.plotCursor]
	b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(" Plot Details: ") +
		lipgloss.NewStyle().Foreground(styles.Text).Render(plot.Name) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).PaddingLeft(2).Render(plot.Description) + "\n")

	if len(plot.RequiredFiles) > 0 {
		reqStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
		b.WriteString(reqStyle.Render("Requires: "+strings.Join(plot.RequiredFiles, ", ")) + "\n")
	}

	return b.String()
}

func (m Model) renderPlotConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" PLOT OUTPUT ") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Configure output formats and resolution. Use Space to toggle/cycle.\n\n"))

	options := m.getPlotConfigOptions()
	labelWidth := 16

	for i, opt := range options {
		isFocused := i == m.plotConfigCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
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
		}

		b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value) + "\n")
	}

	return b.String()
}

func (m Model) renderTimeRange() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" TIME RANGE ") + "\n\n")

	// Find epoch metadata (from first valid subject)
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
	}

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Define time ranges to be computed separately.\n") +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  [A] Add range  [D] Delete range  [Space/Enter] Edit range\n") +
		lipgloss.NewStyle().Foreground(styles.Muted).Render(
			"  Suggested names: baseline, active, n1, n2, p2 (match your paradigm)\n\n"))

	nameWidth := 15
	valWidth := 10

	headerStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true)
	header := fmt.Sprintf("  %-*s %-*s %-*s", nameWidth, headerStyle.Render("Range Name"), valWidth, headerStyle.Render("Start (s)"), valWidth, headerStyle.Render("End (s)"))
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
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		// Styles for fields
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

	// Validation status for required time ranges
	hasBaseline := false
	hasActive := false
	for _, tr := range m.TimeRanges {
		if tr.Name == "baseline" {
			hasBaseline = true
		}
		if tr.Name == "active" {
			hasActive = true
		}
	}

	b.WriteString("\n")
	okStyle := lipgloss.NewStyle().Foreground(styles.Success)
	warnStyle := lipgloss.NewStyle().Foreground(styles.Warning)

	// Show status indicators
	b.WriteString("  " + styles.SectionTitleStyle.Render("Requirements") + "\n")
	if hasBaseline {
		b.WriteString("  " + okStyle.Render("✓") + " baseline - required for normalization\n")
	} else {
		b.WriteString("  " + warnStyle.Render("○") + " baseline - " + warnStyle.Render("not defined (required for ERDS, log-ratio)") + "\n")
	}
	if hasActive {
		b.WriteString("  " + okStyle.Render("✓") + " active - primary analysis window\n")
	} else {
		b.WriteString("  " + warnStyle.Render("○") + " active - " + warnStyle.Render("not defined (recommended)") + "\n")
	}

	// Helpful tip for minimum requirements
	if !hasBaseline || !hasActive {
		b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Tip: Define at least 'baseline' and 'active' for most feature extraction.") + "\n")
	}

	return b.String()
}

func (m Model) renderSubjectSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" SUBJECT SELECTION ") + "\n\n")

	if m.subjectsLoading {
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		frame := frames[m.ticker%len(frames)]
		return b.String() + lipgloss.NewStyle().Foreground(styles.Accent).Render("  "+frame+" Loading subjects...")
	}

	if m.filteringSubject {
		filterBox := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(styles.Accent).
			Padding(0, 1).
			Render("Filter: " + m.subjectFilter + "█")
		b.WriteString("  " + filterBox + "\n\n")
	} else if m.subjectFilter != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Render(
			fmt.Sprintf("  🔍 Filtering: \"%s\"", m.subjectFilter)) + "  " +
			lipgloss.NewStyle().Foreground(styles.Muted).Render("[Esc to clear]") + "\n\n")
	}

	filteredSubjects := m.getFilteredSubjects()

	selectedCount := 0
	validCount := 0
	for subjID, sel := range m.subjectSelected {
		if sel {
			selectedCount++
			for _, s := range m.subjects {
				if s.ID == subjID {
					valid := false
					if m.Pipeline == types.PipelinePlotting {
						valid, _ = m.validatePlottingSubject(s)
					} else if ok, _ := m.Pipeline.ValidateSubject(s); ok {
						valid = true
					}
					if valid {
						validCount++
					}
					break
				}
			}
		}
	}

	// Inline validation indicator
	var statusIndicator string
	if validCount >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}

	summary := fmt.Sprintf("%d selected", selectedCount)
	if validCount < selectedCount {
		summary += fmt.Sprintf(" (%d valid)", validCount)
	}
	if m.subjectFilter != "" {
		summary += fmt.Sprintf(" | %d shown", len(filteredSubjects))
	}
	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(summary))
	if validCount == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	if len(filteredSubjects) == 0 {
		if m.subjectFilter != "" {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("  No subjects match filter"))
		} else {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("  No subjects available"))
		}
		return b.String()
	}

	maxDisplay := styles.MaxVisibleSubjects
	startIdx := 0
	if m.subjectCursor >= maxDisplay {
		startIdx = m.subjectCursor - maxDisplay + 1
	}
	endIdx := startIdx + maxDisplay
	if endIdx > len(filteredSubjects) {
		endIdx = len(filteredSubjects)
	}

	for i := startIdx; i < endIdx; i++ {
		s := filteredSubjects[i]
		isSelected := m.subjectSelected[s.ID]
		isFocused := i == m.subjectCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(s.ID))

		valid, reason := m.Pipeline.ValidateSubject(s)
		if m.Pipeline == types.PipelinePlotting {
			valid, reason = m.validatePlottingSubject(s)
		}
		if !valid {
			indicator := lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark + " " + reason)
			b.WriteString(indicator)
		} else {
			indicator := lipgloss.NewStyle().Foreground(styles.Success).Faint(true).Render(" " + styles.CheckMark)
			b.WriteString(indicator)
		}

		b.WriteString("\n")
	}

	if len(filteredSubjects) > maxDisplay {
		b.WriteString("\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf("  Showing %d-%d of %d", startIdx+1, endIdx, len(filteredSubjects))))
	}

	return b.String()
}

func (m Model) renderReview() string {
	var b strings.Builder

	readyIcon := styles.WarningMark
	headerColor := styles.Warning
	if len(m.validationErrors) == 0 {
		readyIcon = styles.CheckMark
		headerColor = styles.Success
	}

	headerStyle := lipgloss.NewStyle().
		Foreground(headerColor).
		Bold(true)
	b.WriteString(headerStyle.Render(readyIcon+" ") + styles.SectionTitleStyle.Render(" CONFIGURATION REVIEW ") + "\n\n")

	card := strings.Builder{}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(styles.SummaryLabelWidth)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)
	iconStyle := lipgloss.NewStyle().Foreground(styles.Accent)

	card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Pipeline:") +
		lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(m.Pipeline.String()) + "\n")

	modeIcon := "▸"
	if m.modeOptions[m.modeIndex] == styles.ModeVisualize {
		modeIcon = "▹"
	}
	card.WriteString(iconStyle.Render(modeIcon+" ") + labelStyle.Render("Mode:") +
		valueStyle.Render(m.modeOptions[m.modeIndex]) + "\n")

	subjCount := 0
	validCount := 0
	for subjID, sel := range m.subjectSelected {
		if sel {
			subjCount++
			for _, s := range m.subjects {
				if s.ID == subjID {
					if valid, _ := m.Pipeline.ValidateSubject(s); valid {
						validCount++
					}
					break
				}
			}
		}
	}

	var subjectBadge string
	if validCount == subjCount {
		subjectBadge = lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("%d subjects", subjCount))
	} else {
		subjectBadge = lipgloss.NewStyle().Foreground(styles.Warning).Render(
			fmt.Sprintf("%d selected (%d valid)", subjCount, validCount))
	}
	card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Subjects:") + subjectBadge + "\n")

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		cats := m.SelectedCategories()
		var chips string
		for i, cat := range cats {
			chips += lipgloss.NewStyle().
				Foreground(lipgloss.Color("#000000")).
				Background(styles.Accent).
				Padding(0, 1).
				Render(cat)
			if i < len(cats)-1 {
				chips += " "
			}
		}
		card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Features:") + "\n")
		card.WriteString("     " + chips + "\n")

		// Show advanced config summary if not using defaults
		if !m.useDefaultAdvanced {
			configStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Config:") + "\n")

			var configs []string
			if m.isCategorySelected("connectivity") {
				measures := m.selectedConnectivityMeasures()
				if len(measures) > 0 {
					configs = append(configs, fmt.Sprintf("conn=%s", strings.Join(measures, "+")))
				}
			}
			if m.isCategorySelected("pac") {
				configs = append(configs, fmt.Sprintf("pac=%.0f-%.0f/%.0f-%.0fHz",
					m.pacPhaseMin, m.pacPhaseMax, m.pacAmpMin, m.pacAmpMax))
			}
			if m.isCategorySelected("aperiodic") {
				configs = append(configs, fmt.Sprintf("aper=%.0f-%.0fHz", m.aperiodicFmin, m.aperiodicFmax))
			}
			if m.isCategorySelected("complexity") && m.complexityPEOrder != 3 {
				configs = append(configs, fmt.Sprintf("PE=%d", m.complexityPEOrder))
			}

			if len(configs) > 0 {
				card.WriteString("     " + configStyle.Render(strings.Join(configs, ", ")) + "\n")
			}
		}
	} else if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		// Show computations
		comps := m.SelectedComputations()
		card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Analyses:") + valueStyle.Render(strings.Join(comps, ", ")) + "\n")

		// Show selected feature files
		featureFiles := m.SelectedFeatureFiles()
		if len(featureFiles) > 0 {
			var chips string
			for i, file := range featureFiles {
				chips += lipgloss.NewStyle().
					Foreground(lipgloss.Color("#000000")).
					Background(styles.Accent).
					Padding(0, 1).
					Render(file)
				if i < len(featureFiles)-1 {
					chips += " "
				}
			}
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Features:") + "\n")
			card.WriteString("     " + chips + "\n")
		}
	} else if m.Pipeline == types.PipelinePlotting {
		card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Plots:") + "\n")
		grouped := make(map[string][]string)
		for i, plot := range m.plotItems {
			if !m.plotSelected[i] {
				continue
			}
			grouped[plot.Group] = append(grouped[plot.Group], plot.Name)
		}
		var groupOrder []string
		for _, plot := range m.plotItems {
			if _, ok := grouped[plot.Group]; ok {
				found := false
				for _, g := range groupOrder {
					if g == plot.Group {
						found = true
						break
					}
				}
				if !found {
					groupOrder = append(groupOrder, plot.Group)
				}
			}
		}
		for _, group := range groupOrder {
			names := grouped[group]
			if len(names) == 0 {
				continue
			}
			card.WriteString("     " + lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.ToUpper(group)+": ") +
				valueStyle.Render(strings.Join(names, ", ")) + "\n")
		}

		formats := m.SelectedPlotFormats()
		dpi := "default"
		if m.plotDpiIndex >= 0 && m.plotDpiIndex < len(m.plotDpiOptions) {
			dpi = fmt.Sprintf("%d", m.plotDpiOptions[m.plotDpiIndex])
		}
		savefigDpi := "default"
		if m.plotSavefigDpiIndex >= 0 && m.plotSavefigDpiIndex < len(m.plotDpiOptions) {
			savefigDpi = fmt.Sprintf("%d", m.plotDpiOptions[m.plotSavefigDpiIndex])
		}
		card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Output:") +
			valueStyle.Render(fmt.Sprintf("%s | dpi=%s | savefig=%s", strings.Join(formats, ", "), dpi, savefigDpi)) + "\n")
	}

	b.WriteString(styles.CardStyle.Width(m.width-10).Render(card.String()) + "\n\n")

	cmdHeader := lipgloss.JoinHorizontal(lipgloss.Left,
		styles.SectionTitleStyle.Render(" COMMAND "),
		"  ",
		lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("(will be executed)"),
	)
	b.WriteString(cmdHeader + "\n")

	cmdStyle := lipgloss.NewStyle().
		Background(styles.BgBase).
		Foreground(styles.Accent).
		Padding(0, 2).
		MarginLeft(2)
	b.WriteString(cmdStyle.Render(m.BuildCommand()) + "\n\n")

	if len(m.validationErrors) > 0 {
		errPanel := strings.Builder{}
		errHeader := lipgloss.NewStyle().
			Foreground(styles.Error).
			Bold(true).
			Render("⚠ VALIDATION ERRORS")
		errPanel.WriteString(errHeader + "\n\n")

		for _, err := range m.validationErrors {
			errPanel.WriteString(lipgloss.NewStyle().Foreground(styles.Error).Render("  "+styles.CrossMark+" "+err) + "\n")
		}
		errPanel.WriteString("\n")
		errPanel.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  Press Esc to go back and fix issues"))

		b.WriteString(styles.ErrorPanelStyle.Width(m.width-10).Render(errPanel.String()) + "\n\n")
	} else {
		readyMsg := lipgloss.NewStyle().
			Foreground(styles.Success).
			Bold(true).
			Render("✓ Ready to execute")
		hintMsg := lipgloss.NewStyle().
			Foreground(styles.TextDim).
			Render(" — Press Enter to continue")
		b.WriteString("  " + readyMsg + hintMsg + "\n\n")
	}

	return b.String()
}

///////////////////////////////////////////////////////////////////
// Advanced Configuration Rendering
///////////////////////////////////////////////////////////////////

func (m Model) renderAdvancedConfig() string {
	switch m.Pipeline {
	case types.PipelineFeatures:
		return m.renderFeaturesAdvancedConfig()
	case types.PipelineBehavior:
		return m.renderBehaviorAdvancedConfig()
	case types.PipelineDecoding:
		return m.renderDecodingAdvancedConfig()
	default:
		return m.renderDefaultAdvancedConfig()
	}
}

func (m Model) renderFeaturesAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	// Default mode toggle with expand/collapse indicator
	expandIcon := "▶"
	expandHint := "Press Space to customize"
	if !m.useDefaultAdvanced {
		expandIcon = "▼"
		expandHint = "Press Space to use defaults"
	}

	defaultLabel := lipgloss.NewStyle().Foreground(styles.Text).Width(20)
	if m.advancedCursor == 0 && m.expandedOption < 0 {
		defaultLabel = defaultLabel.Foreground(styles.Primary).Bold(true)
	}

	toggleStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	cursor := "  "
	if m.advancedCursor == 0 && m.expandedOption < 0 {
		cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
	}

	b.WriteString(cursor + toggleStyle.Render(expandIcon) + " " + defaultLabel.Render("Configuration"))
	if m.useDefaultAdvanced {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render("Using Defaults"))
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  — " + expandHint))
	} else {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("Custom"))
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  — " + expandHint))
	}
	b.WriteString("\n\n")

	// When using defaults, show minimal summary, otherwise show full config
	if m.useDefaultAdvanced {
		// Collapsed summary - show what defaults will be used
		summaryStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).MarginLeft(4)
		b.WriteString(summaryStyle.Render("Default configuration will be used for all parameters.") + "\n")
		b.WriteString(summaryStyle.Render("Select 'Customize' to modify individual settings.") + "\n")
		return b.String()
	}

	// Expanded - show customization options
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("  Press Space to toggle item, Esc to close.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Press Space to toggle/expand, Enter to proceed.") + "\n\n")
	}

	labelWidth := 22
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build dynamic options list based on selected categories
	// Note: cursor 0 = Configuration toggle (handled above), options start at cursor 1
	options := []struct {
		label      string
		value      string
		hint       string
		expandable bool
		expandIdx  int
	}{}

	// Connectivity options
	if m.isCategorySelected("connectivity") {
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"Connectivity", m.selectedConnectivityDisplay(), "Press Space to expand", true, 4})

		outLevel := "full"
		if m.connOutputLevel == 1 {
			outLevel = "global_only"
		}
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"Conn Output", outLevel, "full / global_only", false, -1})

		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"Graph Metrics", m.boolToOnOff(m.connGraphMetrics), "Enable network analysis", false, -1})

		aecModes := []string{"orth", "none", "sym"}
		aecVal := "unknown"
		if m.connAECMode < len(aecModes) {
			aecVal = aecModes[m.connAECMode]
		}
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"AEC Mode", aecVal, "Orthogonalization mode", false, -1})

	}

	// PAC options
	if m.isCategorySelected("pac") {
		options = append(options,
			struct {
				label      string
				value      string
				hint       string
				expandable bool
				expandIdx  int
			}{"PAC Phase Range", fmt.Sprintf("%.1f-%.1f Hz", m.pacPhaseMin, m.pacPhaseMax), "Low-freq phase band", false, -1},
			struct {
				label      string
				value      string
				hint       string
				expandable bool
				expandIdx  int
			}{"PAC Amp Range", fmt.Sprintf("%.1f-%.1f Hz", m.pacAmpMin, m.pacAmpMax), "High-freq amplitude band", false, -1},
		)
	}

	// Aperiodic options
	if m.isCategorySelected("aperiodic") {
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"Aperiodic Range", fmt.Sprintf("%.1f-%.1f Hz", m.aperiodicFmin, m.aperiodicFmax), "Spectral fit range", false, -1})
	}

	// Complexity options
	if m.isCategorySelected("complexity") {
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"PE Order", fmt.Sprintf("%d", m.complexityPEOrder), "Permutation entropy order (3-7)", false, -1})
	}

	// ERP options
	if m.isCategorySelected("erp") {
		options = append(options,
			struct {
				label      string
				value      string
				hint       string
				expandable bool
				expandIdx  int
			}{"ERP Baseline", m.boolToOnOff(m.erpBaselineCorrection), "Use baseline correction", false, -1},
		)
	}

	// Burst options
	if m.isCategorySelected("bursts") {
		options = append(options,
			struct {
				label      string
				value      string
				hint       string
				expandable bool
				expandIdx  int
			}{"Burst Threshold", fmt.Sprintf("%.1f z", m.burstThresholdZ), "Z-score threshold", false, -1},
		)
	}

	// Power options
	if m.isCategorySelected("power") {
		modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
		modeVal := "unknown"
		if m.powerBaselineMode < len(modes) {
			modeVal = modes[m.powerBaselineMode]
		}

		options = append(options,
			struct {
				label      string
				value      string
				hint       string
				expandable bool
				expandIdx  int
			}{"Power Mode", modeVal, "Normalization method", false, -1},
		)
	}

	// Spectral options
	if m.isCategorySelected("spectral") {
		options = append(options, struct {
			label      string
			value      string
			hint       string
			expandable bool
			expandIdx  int
		}{"Spectral Edge", fmt.Sprintf("%.0f%%", m.spectralEdgePercentile*100), "Percentile for SEF", false, -1})
	}

	for i, opt := range options {
		// Options list starts at cursor position 1 (position 0 is toggle header)
		isFocused := (i+1) == m.advancedCursor && m.expandedOption < 0

		// Build styles based on state
		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}

		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		// Show expand indicator for expandable options
		expandIndicator := ""
		if opt.expandable && m.expandedOption != opt.expandIdx {
			expandIndicator = " [+]"
		} else if opt.expandable && m.expandedOption == opt.expandIdx {
			expandIndicator = " [-]"
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value+expandIndicator))
		if !opt.expandable {
			b.WriteString("  " + hintStyle.Render(opt.hint))
		}
		b.WriteString("\n")

		// Render expanded items if this option is expanded
		if opt.expandable && m.expandedOption == opt.expandIdx {
			b.WriteString(m.renderExpandedItems(opt.expandIdx))
		}
	}

	return b.String()
}

func (m Model) renderExpandedItems(optionIdx int) string {
	var b strings.Builder

	subIndent := "      " // 6 spaces for sub-items

	if optionIdx == 4 { // Connectivity measures
		for i, measure := range connectivityMeasures {
			isSelected := m.connectivityMeasures[i]
			isFocused := i == m.subCursor

			checkbox := styles.RenderCheckbox(isSelected, isFocused)

			nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
			if isFocused {
				nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
			}

			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + measure.Description)
			b.WriteString(subIndent + checkbox + nameStyle.Render(measure.Name) + desc + "\n")
		}
	}

	return b.String()
}

func (m Model) renderBehaviorAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	b.WriteString(infoStyle.Render("  Configure behavior analysis parameters.") + "\n")
	if m.editingNumber {
		b.WriteString(infoStyle.Render("  Type a number, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Press Space to edit values, Enter to proceed.") + "\n\n")
	}

	labelWidth := 22
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build option values, showing input buffer when editing
	bootstrapVal := fmt.Sprintf("%d", m.bootstrapSamples)
	permutationsVal := fmt.Sprintf("%d", m.nPermutations)
	rngSeedVal := m.rngSeedDisplay()

	// Override with input buffer if editing that field
	if m.editingNumber {
		inputDisplay := m.numberBuffer + "█"
		if m.isCurrentlyEditing(optBootstrap) {
			bootstrapVal = inputDisplay
		} else if m.isCurrentlyEditing(optNPerm) {
			permutationsVal = inputDisplay
		} else if m.isCurrentlyEditing(optRNGSeed) {
			rngSeedVal = inputDisplay
		}
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Correlation Method", m.correlationMethod, "spearman (robust) / pearson"},
		{"Bootstrap Samples", bootstrapVal, "Type a number (0=disabled)"},
		{"Permutations", permutationsVal, "Type a number"},
		{"RNG Seed", rngSeedVal, "Type a number (0=default)"},
		{"Control Temperature", m.boolToOnOff(m.controlTemperature), "Partial correlation covariate"},
		{"Control Trial Order", m.boolToOnOff(m.controlTrialOrder), "Order effects covariate"},
		{"FDR Alpha", fmt.Sprintf("%.2f", m.fdrAlpha), "Multiple comparison threshold"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}

		if m.useDefaultAdvanced && i > 0 {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		} else if m.editingNumber && isFocused {
			// Highlight the editing field
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderDecodingAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	b.WriteString(infoStyle.Render("  Configure decoding analysis parameters.") + "\n")
	b.WriteString(infoStyle.Render("  Press Space to toggle/cycle values, Enter to proceed.") + "\n\n")

	labelWidth := 22
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	permutationsVal := fmt.Sprintf("%d", m.decodingNPerm)
	innerSplitsVal := fmt.Sprintf("%d", m.innerSplits)
	rngSeedVal := m.rngSeedDisplay()

	// Override with input buffer if editing that field
	if m.editingNumber {
		inputDisplay := m.numberBuffer + "█"
		if m.isCurrentlyEditing(optDecodingNPerm) {
			permutationsVal = inputDisplay
		} else if m.isCurrentlyEditing(optDecodingInnerSplits) {
			innerSplitsVal = inputDisplay
		} else if m.isCurrentlyEditing(optRNGSeed) {
			rngSeedVal = inputDisplay
		}
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Permutations", permutationsVal, "0=disabled, 100+ for p-values"},
		{"Inner CV Splits", innerSplitsVal, "Cross-validation folds"},
		{"RNG Seed", rngSeedVal, "0=project default"},
		{"Skip Time-Gen", m.boolToOnOff(m.skipTimeGen), "Skip time generalization"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}

		if m.useDefaultAdvanced && i > 0 {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		} else if m.editingNumber && isFocused {
			// Highlight the editing field
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderDefaultAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render("  No advanced options available for this pipeline.") + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render("  Press Enter to continue.") + "\n")
	return b.String()
}

// Helper methods for display

func (m Model) boolToOnOff(val bool) string {
	if val {
		return "ON"
	}
	return "OFF"
}

func (m Model) rngSeedDisplay() string {
	if m.rngSeed == 0 {
		return "0 (default)"
	}
	return fmt.Sprintf("%d", m.rngSeed)
}

func (m Model) selectedConnectivityDisplay() string {
	var selected []string
	for i, sel := range m.connectivityMeasures {
		if sel && i < len(connectivityMeasures) {
			selected = append(selected, connectivityMeasures[i].Key)
		}
	}
	if len(selected) == 0 {
		return "(none)"
	}
	return strings.Join(selected, ", ")
}
