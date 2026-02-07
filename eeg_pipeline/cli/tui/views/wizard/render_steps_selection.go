// Confirmation, mode, computation, category, band, ROI, spatial, feature file, subject selection.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderConfirmation() string {
	content := strings.Builder{}

	content.WriteString(m.renderSectionAccent() + " " + styles.SectionTitleStyle.Render("Review & execute") + "\n\n")

	content.WriteString(
		lipgloss.NewStyle().Foreground(styles.Text).Render("pipeline:") +
			" " +
			lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.Pipeline.String()) +
			"\n",
	)

	selectedCount := countSelectedStringItems(m.subjectSelected)
	if selectedCount > 0 {
		subjectInfo := lipgloss.NewStyle().Foreground(styles.Text).Render("subjects:") +
			" " +
			lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(fmt.Sprintf("%d", selectedCount))
		content.WriteString(subjectInfo + "\n")
	}

	content.WriteString("\n")
	content.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Output locations:") + "\n")
	for _, p := range m.getExpectedOutputPaths() {
		pathLine := lipgloss.NewStyle().Foreground(styles.Muted).Render("  → ") +
			lipgloss.NewStyle().Foreground(styles.Text).Render(p)
		content.WriteString(pathLine + "\n")
	}

	content.WriteString("\n")

	yesBtn := lipgloss.NewStyle().
		Foreground(styles.BgDark).
		Background(styles.Success).
		Bold(true).
		Padding(0, 2).
		Render("[Y] Execute")

	dryRunBtn := lipgloss.NewStyle().
		Foreground(styles.BgDark).
		Background(styles.Warning).
		Bold(true).
		Padding(0, 2).
		Render("[D] Dry-run")

	noBtn := lipgloss.NewStyle().
		Foreground(styles.BgDark).
		Background(styles.Error).
		Bold(true).
		Padding(0, 2).
		Render(styles.CrossMark + " [N] Cancel")

	actions := lipgloss.JoinHorizontal(lipgloss.Left, yesBtn, "  ", dryRunBtn, "  ", noBtn)
	content.WriteString("  " + actions)

	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder()).
		BorderForeground(styles.Border).
		Padding(1, 2)

	return boxStyle.Render(content.String())
}

func (m Model) renderModeSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Mode selection") + "\n\n")

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
	b.WriteString(styles.SectionTitleStyle.Render("Computations") + "\n\n")

	count := 0
	for _, sel := range m.computationSelected {
		if sel {
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
		fmt.Sprintf("%d of %d selected", count, len(m.computations))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	groups := map[string][]Computation{
		"DataPrep": {}, "Core": {}, "Advanced": {}, "Quality": {},
	}
	groupNames := map[string]string{
		"DataPrep": "Data preparation",
		"Core":     "Core analyses",
		"Advanced": "Advanced / causal analyses",
		"Quality":  "Quality & validation",
	}
	groupOrder := []string{"DataPrep", "Core", "Advanced", "Quality"}

	for _, comp := range m.computations {
		if comp.Group != "" {
			groups[comp.Group] = append(groups[comp.Group], comp)
		}
	}

	for _, groupKey := range groupOrder {
		groupComps := groups[groupKey]
		if len(groupComps) == 0 {
			continue
		}
		b.WriteString("\n")
		b.WriteString(styles.SectionTitleStyle.Render(groupNames[groupKey]) + "\n\n")

		for _, comp := range groupComps {
			idx := -1
			for i, c := range m.computations {
				if c.Key == comp.Key {
					idx = i
					break
				}
			}
			if idx == -1 {
				continue
			}

			isSelected := m.computationSelected[idx]
			isFocused := idx == m.computationCursor
			checkbox := styles.RenderCheckbox(isSelected, isFocused)
			nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
			if isFocused {
				nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
			}
			b.WriteString(checkbox + nameStyle.Render(comp.Name))

			if m.computationAvailability != nil && m.computationAvailability[comp.Key] {
				timestamp := m.computationLastModified[comp.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s]", relTime))
					b.WriteString(avail)
				} else {
					b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render("  [DONE]"))
				}
			} else {
				b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  [NO DATA]"))
			}

			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  "+comp.Description) + "\n")
		}
	}

	return b.String()
}

func (m Model) renderCategorySelection() string {
	var b strings.Builder

	title := "Feature categories"
	if m.CurrentStep == types.StepSelectPlotCategories {
		title = "Plot categories"
	}
	b.WriteString(styles.SectionTitleStyle.Render(title) + "\n\n")

	if m.CurrentStep == types.StepSelectPlotCategories {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
			"Toggle a category to include or exclude all plots in that group.\n",
		))
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
			"Press 'g' to configure global styling options.\n\n",
		))
	}

	if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
		return m.renderGlobalStylingPanel()
	}

	count := 0
	for _, sel := range m.selected {
		if sel {
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
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + m.categoryDescs[i]))
		}
		if m.Pipeline == types.PipelinePlotting {
			categories := m.plotCategories
			if len(categories) == 0 {
				categories = defaultPlotCategories
			}
			if i < len(categories) {
				total, selected := m.plotCountsForGroup(categories[i].Key)
				countText := fmt.Sprintf("  [%d/%d plots]", selected, total)
				b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render(countText))
			}
		} else if m.featureAvailability != nil {
			if m.featureAvailability[cat] {
				timestamp := m.featureLastModified[cat]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s]", relTime)))
				} else {
					b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render("  [DATA]"))
				}
			} else {
				b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  [NO DATA]"))
			}
		}
		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderGlobalStylingPanel() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Global styling") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
		"Configure styling options that apply to all plots.\n",
	))
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
		"Press 'g' or Escape to return to categories.\n\n",
	))

	options := m.getGlobalStylingOptions()
	labelWidth := 24
	for i, opt := range options {
		isFocused := i == m.globalStylingCursor
		lines := m.renderOption(opt, labelWidth, isFocused)
		for _, line := range lines {
			b.WriteString(line.text)
			b.WriteString("\n")
		}
	}

	return b.String()
}

func (m Model) renderBandSelection() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Frequency bands") + "\n\n")

	count := 0
	for _, sel := range m.bandSelected {
		if sel {
			count++
		}
	}
	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	b.WriteString("  " + statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.bands))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	instructionStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	b.WriteString(instructionStyle.Render("  Space: toggle • E: edit • +: add band • D: delete band") + "\n\n")

	for i, band := range m.bands {
		isSelected := m.bandSelected[i]
		isFocused := i == m.bandCursor
		isEditing := m.editingBandIdx == i
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		var nameDisplay string
		if isEditing && m.editingBandField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "▌")
		} else {
			nameDisplay = nameStyle.Render(band.Name)
		}

		freqStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			freqStyle = lipgloss.NewStyle().Foreground(styles.Muted)
		}
		var lowHzDisplay, highHzDisplay string
		if isEditing && m.editingBandField == 1 {
			lowHzDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "▌")
		} else {
			lowHzDisplay = freqStyle.Render(fmt.Sprintf("%.1f", band.LowHz))
		}
		if isEditing && m.editingBandField == 2 {
			highHzDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "▌")
		} else {
			highHzDisplay = freqStyle.Render(fmt.Sprintf("%.1f", band.HighHz))
		}
		freqRange := freqStyle.Render(" [") + lowHzDisplay + freqStyle.Render(" - ") + highHzDisplay + freqStyle.Render(" Hz]")
		b.WriteString(checkbox + nameDisplay + freqRange + "\n")
	}

	return b.String()
}

func (m Model) renderROISelection() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Regions of interest") + "\n\n")

	count := 0
	for _, sel := range m.roiSelected {
		if sel {
			count++
		}
	}
	var statusIndicator string
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	b.WriteString("  " + statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.rois))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	instructionStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	b.WriteString(instructionStyle.Render("  Space: toggle • E: edit • +: add ROI • D: delete ROI") + "\n\n")

	for i, roi := range m.rois {
		isSelected := m.roiSelected[i]
		isFocused := i == m.roiCursor
		isEditing := m.editingROIIdx == i
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		var nameDisplay string
		if isEditing && m.editingROIField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditBuffer + "▌")
		} else {
			nameDisplay = nameStyle.Render(roi.Name)
		}
		channelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			channelStyle = lipgloss.NewStyle().Foreground(styles.Muted)
		}
		var channelsDisplay string
		if isEditing && m.editingROIField == 1 {
			channelsDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditBuffer + "▌")
		} else {
			channelsDisplay = m.renderChannelsWithUnavailable(roi.Channels, channelStyle, isFocused)
		}
		b.WriteString(checkbox + nameDisplay + channelStyle.Render(" [") + channelsDisplay + channelStyle.Render("]") + "\n")
	}

	b.WriteString("\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.Repeat("─", 60)) + "\n")

	usedChannels, unusedChannels := m.computeChannelUsage()

	if len(unusedChannels) > 0 {
		unusedLabel := lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render("Unused channels")
		unusedCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(unusedChannels)))
		b.WriteString(unusedLabel + unusedCount + "\n")
		b.WriteString(m.formatChannelList(unusedChannels, styles.Success))
	} else if len(m.availableChannels) > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Italic(true).Render("  All channels assigned to ROIs ✓") + "\n")
	} else {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  No channel information available") + "\n")
	}

	if len(usedChannels) > 0 {
		b.WriteString("\n")
		usedLabel := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render("Used channels")
		usedCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(usedChannels)))
		b.WriteString(usedLabel + usedCount + "\n")
		b.WriteString(m.formatChannelList(usedChannels, styles.Accent))
	}

	if len(m.unavailableChannels) > 0 {
		b.WriteString("\n")
		unavailLabel := lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render("Unavailable channels (bad)")
		unavailCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(m.unavailableChannels)))
		b.WriteString(unavailLabel + unavailCount + "\n")
		b.WriteString(m.formatChannelList(m.unavailableChannels, styles.Error))
	}

	return b.String()
}

func (m Model) computeChannelUsage() (used, unused []string) {
	if len(m.availableChannels) == 0 {
		return nil, nil
	}
	usedSet := make(map[string]bool)
	for _, roi := range m.rois {
		for _, ch := range strings.Split(roi.Channels, ",") {
			ch = strings.TrimSpace(ch)
			if ch != "" {
				usedSet[strings.ToUpper(ch)] = true
			}
		}
	}
	for _, ch := range m.availableChannels {
		if usedSet[strings.ToUpper(ch)] {
			used = append(used, ch)
		} else {
			unused = append(unused, ch)
		}
	}
	return used, unused
}

func (m Model) renderChannelsWithUnavailable(channelsStr string, baseStyle lipgloss.Style, isFocused bool) string {
	if channelsStr == "" {
		return baseStyle.Render(channelsStr)
	}
	unavailableSet := make(map[string]bool)
	for _, ch := range m.unavailableChannels {
		unavailableSet[strings.ToUpper(strings.TrimSpace(ch))] = true
	}
	parts := strings.Split(channelsStr, ",")
	var channels []string
	for _, ch := range parts {
		ch = strings.TrimSpace(ch)
		if ch != "" {
			channels = append(channels, ch)
		}
	}
	rawLength := len(channelsStr)
	needsTruncation := rawLength > 40
	if needsTruncation {
		var truncatedChannels []string
		var truncatedLength int
		for _, ch := range channels {
			chLength := len(ch)
			if len(truncatedChannels) > 0 {
				chLength += 2
			}
			if truncatedLength+chLength > 37 {
				break
			}
			truncatedChannels = append(truncatedChannels, ch)
			truncatedLength += chLength
		}
		channels = truncatedChannels
	}

	var renderedParts []string
	for i, ch := range channels {
		if i > 0 {
			renderedParts = append(renderedParts, baseStyle.Render(", "))
		}
		if unavailableSet[strings.ToUpper(ch)] {
			unavailStyle := lipgloss.NewStyle().Foreground(styles.Error)
			if isFocused {
				unavailStyle = unavailStyle.Bold(true)
			}
			renderedParts = append(renderedParts, unavailStyle.Render(ch))
		} else {
			renderedParts = append(renderedParts, baseStyle.Render(ch))
		}
	}
	result := strings.Join(renderedParts, "")
	if needsTruncation {
		originalCount := 0
		for _, p := range parts {
			if strings.TrimSpace(p) != "" {
				originalCount++
			}
		}
		if len(channels) < originalCount {
			result += baseStyle.Render("...")
		}
	}
	return result
}

func (m Model) formatChannelList(channels []string, color lipgloss.Color) string {
	if len(channels) == 0 {
		return ""
	}
	const maxWidth = 70
	var lines []string
	var currentLine strings.Builder
	currentLine.WriteString("  ")
	for i, ch := range channels {
		sep := ""
		if i > 0 {
			sep = ", "
		}
		entry := sep + ch
		if currentLine.Len()+len(entry) > maxWidth {
			lines = append(lines, currentLine.String())
			currentLine.Reset()
			currentLine.WriteString("  ")
			entry = ch
		}
		currentLine.WriteString(entry)
	}
	if currentLine.Len() > 2 {
		lines = append(lines, currentLine.String())
	}
	channelStyle := lipgloss.NewStyle().Foreground(color).Faint(true)
	var result strings.Builder
	for _, line := range lines {
		result.WriteString(channelStyle.Render(line) + "\n")
	}
	return result.String()
}

func (m Model) renderSpatialSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Spatial aggregation") + "\n\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Select how to aggregate features spatially.") + "\n\n")

	count := 0
	for _, sel := range m.spatialSelected {
		if sel {
			count++
		}
	}
	b.WriteString("  " + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(spatialModes))) + "\n\n")

	for i, mode := range spatialModes {
		isSelected := m.spatialSelected[i]
		isFocused := i == m.spatialCursor
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		b.WriteString(checkbox + nameStyle.Render(mode.Name) + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  "+mode.Description) + "\n")
	}

	return b.String()
}

func (m Model) renderFeatureFileSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Feature files") + "\n\n")

	applicableFeatures := m.GetApplicableFeatureFiles()
	instruction := "  Select which feature files to load for analysis.\n"
	if len(applicableFeatures) < len(featureFileOptions) {
		instruction = "  Showing features applicable for selected computations.\n"
	}
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(instruction) +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Features marked in green are available; red indicates missing data.\n\n"))

	count := 0
	availableCount := 0
	for _, file := range applicableFeatures {
		if m.featureFileSelected[file.Key] {
			count++
		}
		if m.featureAvailability != nil && m.featureAvailability[file.Key] {
			availableCount++
		}
	}
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("  %d of %d selected", count, len(applicableFeatures))))
	if m.featureAvailability != nil {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
			fmt.Sprintf(" | %d available", availableCount)))
	}
	b.WriteString("\n\n")

	displayCursor := m.featureFileCursor
	if displayCursor >= len(applicableFeatures) {
		displayCursor = len(applicableFeatures) - 1
	}
	if displayCursor < 0 {
		displayCursor = 0
	}

	for i, file := range applicableFeatures {
		isSelected := m.featureFileSelected[file.Key]
		isFocused := i == displayCursor
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		b.WriteString(checkbox + nameStyle.Render(file.Name))
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + file.Description))

		if m.featureAvailability != nil {
			if m.featureAvailability[file.Key] {
				timestamp := m.featureLastModified[file.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s %s]", styles.CheckMark, relTime)))
				} else {
					b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + styles.CheckMark + " DATA]"))
				}
			} else {
				b.WriteString(lipgloss.NewStyle().Foreground(styles.Error).Render("  [" + styles.CrossMark + " MISSING]"))
			}
		}
		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderSubjectSelection() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Subject selection") + "\n\n")

	if m.Pipeline == types.PipelineML {
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		hintStyle := lipgloss.NewStyle().Foreground(styles.Muted)

		groupOpt := dimStyle.Render("Group (LOSO)")
		subjectOpt := dimStyle.Render("Subject (within)")
		if m.mlScope == MLCVScopeGroup {
			groupOpt = valueStyle.Render("Group (LOSO)")
		} else {
			subjectOpt = valueStyle.Render("Subject (within)")
		}
		b.WriteString("  " + labelStyle.Render("scope:") + " " + groupOpt + "  " + subjectOpt + "  " +
			hintStyle.Render("[Tab to toggle]") + "\n\n")
	} else if m.Pipeline == types.PipelinePlotting {
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		hintStyle := lipgloss.NewStyle().Foreground(styles.Muted)

		groupOpt := dimStyle.Render("Group")
		subjectOpt := dimStyle.Render("Subject")
		if m.plottingScope == PlottingScopeGroup {
			groupOpt = valueStyle.Render("Group")
		} else {
			subjectOpt = valueStyle.Render("Subject")
		}
		b.WriteString("  " + labelStyle.Render("level:") + " " + groupOpt + "  " + subjectOpt + "  " +
			hintStyle.Render("[Tab to toggle]") + "\n\n")
	}

	if m.subjectsLoading {
		return b.String() + "  " + m.subjectLoadingSpinner.View()
	}

	if m.filteringSubject {
		filterBox := lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(styles.Accent).
			Padding(0, 1).
			Render("filter: " + m.subjectFilter + "█")
		b.WriteString("  " + filterBox + "\n\n")
	} else if m.subjectFilter != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Render(
			fmt.Sprintf("  Filter: \"%s\"", m.subjectFilter)) + "  " +
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

	minValid := 1
	if m.Pipeline == types.PipelineML && m.mlScope == MLCVScopeGroup {
		minValid = 2
	}
	if m.Pipeline == types.PipelinePlotting && m.plottingScope == PlottingScopeGroup {
		minValid = 2
	}

	var statusIndicator string
	if validCount >= minValid {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	countStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	summary := countStyle.Render(fmt.Sprintf("%d", selectedCount)) + dimStyle.Render(" selected")
	if validCount < selectedCount {
		summary += lipgloss.NewStyle().Foreground(styles.Warning).Render(fmt.Sprintf(" (%d valid)", validCount))
	}
	summary += dimStyle.Render(fmt.Sprintf(" of %d total", len(m.subjects)))
	if m.subjectFilter != "" {
		summary += dimStyle.Render(fmt.Sprintf(" | %d shown", len(filteredSubjects)))
	}
	b.WriteString(statusIndicator + summary)
	if validCount < minValid {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(
			fmt.Sprintf(" — select at least %d", minValid)))
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

	overhead := 12
	if m.Pipeline == types.PipelineML || m.Pipeline == types.PipelinePlotting {
		overhead += 2
	}
	layout := styles.CalculateListLayout(m.height, m.subjectCursor, len(filteredSubjects), overhead)
	startIdx := layout.StartIdx
	endIdx := layout.EndIdx

	if layout.ShowScrollUp {
		b.WriteString(styles.RenderScrollUpIndicator(startIdx) + "\n")
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

		var statusBadges []string
		if s.HasSourceData {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("S"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("·"))
		}
		if s.HasBids {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("B"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("·"))
		}
		if s.HasDerivatives {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("D"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("·"))
		}
		badgeStr := lipgloss.NewStyle().Foreground(styles.TextDim).Render(" [") +
			strings.Join(statusBadges, "") +
			lipgloss.NewStyle().Foreground(styles.TextDim).Render("]")
		b.WriteString(badgeStr)
		if !valid {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark + " " + reason))
		}
		b.WriteString("\n")
	}

	if layout.ShowScrollDn {
		remaining := len(filteredSubjects) - endIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}
	if len(filteredSubjects) > layout.MaxItems {
		b.WriteString("\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf("  Showing %d-%d of %d  [↑↓ to scroll]", startIdx+1, endIdx, len(filteredSubjects))))
	}
	b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
		Render("  Legend: [S]=Source Data [B]=BIDS [D]=Derivatives"))

	return b.String()
}
