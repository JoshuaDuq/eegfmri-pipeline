// Confirmation, mode, computation, category, band, ROI, spatial, feature file, subject selection.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderModeSelection() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Mode", m.contentWidth) + "\n")

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for i, opt := range m.modeOptions {
		isSelected := i == m.modeIndex
		cursor := "  "
		if isSelected {
			cursor = styles.RenderCursor()
		}
		radio := styles.RenderRadio(isSelected, isSelected)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isSelected {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + radio + " " + nameStyle.Render(opt)
		if i < len(m.modeDescriptions) {
			line += "  " + descStyle.Render(m.modeDescriptions[i])
		}
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderComputationSelection() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Computations", m.contentWidth) + "\n")

	count := 0
	for _, sel := range m.computationSelected {
		if sel {
			count++
		}
	}

	b.WriteString(styles.RenderStatusCount(count, len(m.computations), "selected"))
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

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for _, groupKey := range groupOrder {
		groupComps := groups[groupKey]
		if len(groupComps) == 0 {
			continue
		}
		b.WriteString("\n")
		b.WriteString(styles.RenderDimSectionLabel(groupNames[groupKey]) + "\n")

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
			cursor := "  "
			if isFocused {
				cursor = styles.RenderCursor()
			}
			checkbox := styles.RenderCheckbox(isSelected, isFocused)
			nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
			if isFocused {
				nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			}
			line := cursor + checkbox + " " + nameStyle.Render(comp.Name)

			if m.computationAvailability != nil && m.computationAvailability[comp.Key] {
				timestamp := m.computationLastModified[comp.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + relTime + "]")
				} else {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + styles.CheckMark + "]")
				}
			} else {
				line += descStyle.Render("  [--]")
			}

			line += descStyle.Render("  " + comp.Description)
			b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
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
	b.WriteString(styles.RenderStepHeader(title, m.contentWidth) + "\n")

	if m.CurrentStep == types.StepSelectPlotCategories {
		hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
		b.WriteString(hintStyle.Render("Toggle categories. Press 'g' for global styling.") + "\n")
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

	b.WriteString(styles.RenderStatusCount(count, len(m.categories), "selected"))
	b.WriteString("\n\n")

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for i, cat := range m.categories {
		isSelected := m.selected[i]
		isFocused := i == m.categoryIndex
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + nameStyle.Render(featureCategoryLabel(cat))

		if i < len(m.categoryDescs) {
			line += "  " + descStyle.Render(m.categoryDescs[i])
		}
		if m.Pipeline == types.PipelinePlotting {
			categories := m.plotCategories
			if len(categories) == 0 {
				categories = defaultPlotCategories
			}
			if i < len(categories) {
				total, selected := m.plotCountsForGroup(categories[i].Key)
				line += descStyle.Render(fmt.Sprintf("  [%d/%d]", selected, total))
			}
		} else if m.featureAvailability != nil {
			if m.featureAvailability[cat] {
				timestamp := m.featureLastModified[cat]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + relTime + "]")
				} else {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + styles.CheckMark + "]")
				}
			} else {
				line += descStyle.Render("  [--]")
			}
		}
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderGlobalStylingPanel() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Global styling", m.contentWidth) + "\n")
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(hintStyle.Render("Configure styling for all plots. Press 'g' or Esc to return.") + "\n\n")

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

	b.WriteString(styles.RenderStepHeader("Frequency bands", m.contentWidth) + "\n")

	count := 0
	for _, sel := range m.bandSelected {
		if sel {
			count++
		}
	}
	b.WriteString(styles.RenderStatusCount(count, len(m.bands), "selected"))
	b.WriteString("\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
		Render("  Space: toggle  E: edit  +: add  D: delete") + "\n\n")

	for i, band := range m.bands {
		isSelected := m.bandSelected[i]
		isFocused := i == m.bandCursor
		isEditing := m.editingBandIdx == i
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}

		var nameDisplay string
		if isEditing && m.editingBandField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "\u258c")
		} else {
			nameDisplay = nameStyle.Render(band.Name)
		}

		freqStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			freqStyle = lipgloss.NewStyle().Foreground(styles.Muted)
		}
		var lowHzDisplay, highHzDisplay string
		if isEditing && m.editingBandField == 1 {
			lowHzDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "\u258c")
		} else {
			lowHzDisplay = freqStyle.Render(fmt.Sprintf("%.1f", band.LowHz))
		}
		if isEditing && m.editingBandField == 2 {
			highHzDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "\u258c")
		} else {
			highHzDisplay = freqStyle.Render(fmt.Sprintf("%.1f", band.HighHz))
		}
		freqRange := freqStyle.Render(" [") + lowHzDisplay + freqStyle.Render(" - ") + highHzDisplay + freqStyle.Render(" Hz]")
		b.WriteString(cursor + checkbox + " " + nameDisplay + freqRange + "\n")
	}

	return b.String()
}

func (m Model) renderROISelection() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Regions of interest", m.contentWidth) + "\n")

	count := 0
	for _, sel := range m.roiSelected {
		if sel {
			count++
		}
	}
	b.WriteString(styles.RenderStatusCount(count, len(m.rois), "selected"))
	b.WriteString("\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
		Render("  Space: toggle  E: edit  Left/Right: move caret  +: add  D: delete") + "\n\n")

	for i, roi := range m.rois {
		isSelected := m.roiSelected[i]
		isFocused := i == m.roiCursor
		isEditing := m.editingROIIdx == i
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		var nameDisplay string
		if isEditing && m.editingROIField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditDisplayValue())
		} else {
			nameDisplay = nameStyle.Render(roi.Name)
		}
		channelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			channelStyle = lipgloss.NewStyle().Foreground(styles.Muted)
		}
		var channelsDisplay string
		if isEditing && m.editingROIField == 1 {
			channelsDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditDisplayValue())
		} else {
			channelsDisplay = m.renderChannelsWithUnavailable(roi.Channels, channelStyle, isFocused)
		}
		line := cursor + checkbox + " " + nameDisplay + channelStyle.Render(" [") + channelsDisplay + channelStyle.Render("]")
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	b.WriteString("\n" + styles.RenderDivider(60) + "\n")

	usedChannels, unusedChannels := m.computeChannelUsage()

	if len(unusedChannels) > 0 {
		b.WriteString(styles.RenderDimSectionLabel("Unused") + lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" %d", len(unusedChannels))) + "\n")
		b.WriteString(m.formatChannelList(unusedChannels, styles.Success))
	} else if len(m.availableChannels) > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Italic(true).Render("  All channels assigned "+styles.CheckMark) + "\n")
	} else {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  No channel info available") + "\n")
	}

	if len(usedChannels) > 0 {
		b.WriteString("\n")
		b.WriteString(styles.RenderDimSectionLabel("Assigned") + lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" %d", len(usedChannels))) + "\n")
		b.WriteString(m.formatChannelList(usedChannels, styles.Accent))
	}

	if len(m.unavailableChannels) > 0 {
		b.WriteString("\n")
		b.WriteString(styles.RenderDimSectionLabel("Bad channels") + lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" %d", len(m.unavailableChannels))) + "\n")
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
	maxWidth := m.contentWidth - 4
	if maxWidth < 30 {
		maxWidth = 30
	}
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
	b.WriteString(styles.RenderStepHeader("Spatial aggregation", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Select how to aggregate features spatially.") + "\n")

	count := 0
	for _, sel := range m.spatialSelected {
		if sel {
			count++
		}
	}
	b.WriteString(styles.RenderStatusCount(count, len(spatialModes), "selected") + "\n\n")

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for i, mode := range spatialModes {
		isSelected := m.spatialSelected[i]
		isFocused := i == m.spatialCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + nameStyle.Render(mode.Name) + "  " + descStyle.Render(mode.Description)
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderFeatureFileSelection() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Feature files", m.contentWidth) + "\n")

	applicableFeatures := m.GetApplicableFeatureFiles()
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if len(applicableFeatures) < len(featureFileOptions) {
		b.WriteString(hintStyle.Render("  Showing applicable features. Green = available, red = missing.") + "\n")
	} else {
		b.WriteString(hintStyle.Render("  Green = available, red = missing data.") + "\n")
	}

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
	b.WriteString(styles.RenderStatusCount(count, len(applicableFeatures), "selected"))
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

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for i, file := range applicableFeatures {
		isSelected := m.featureFileSelected[file.Key]
		isFocused := i == displayCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + nameStyle.Render(file.Name) + "  " + descStyle.Render(file.Description)

		if m.featureAvailability != nil {
			if m.featureAvailability[file.Key] {
				timestamp := m.featureLastModified[file.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + relTime + "]")
				} else {
					line += lipgloss.NewStyle().Foreground(styles.Success).Render("  [" + styles.CheckMark + "]")
				}
			} else {
				line += lipgloss.NewStyle().Foreground(styles.Error).Render("  [" + styles.CrossMark + "]")
			}
		}
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderSubjectSelection() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Subjects", m.contentWidth) + "\n")

	labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	hintStyle := lipgloss.NewStyle().Foreground(styles.Muted)

	switch m.Pipeline {
	case types.PipelineML:
		groupOpt := dimStyle.Render("Group (LOSO)")
		subjectOpt := dimStyle.Render("Subject (within)")
		if m.mlScope == MLCVScopeGroup {
			groupOpt = valueStyle.Render("Group (LOSO)")
		} else {
			subjectOpt = valueStyle.Render("Subject (within)")
		}
		b.WriteString("  " + labelStyle.Render("Scope:") + " " + groupOpt + "  " + subjectOpt + "  " +
			hintStyle.Render("[Tab]") + "\n")
	case types.PipelinePlotting:
		groupOpt := dimStyle.Render("Group")
		subjectOpt := dimStyle.Render("Subject")
		if m.plottingScope == PlottingScopeGroup {
			groupOpt = valueStyle.Render("Group")
		} else {
			subjectOpt = valueStyle.Render("Subject")
		}
		b.WriteString("  " + labelStyle.Render("Level:") + " " + groupOpt + "  " + subjectOpt + "  " +
			hintStyle.Render("[Tab]") + "\n")
	}

	if m.subjectsLoading && len(m.subjects) == 0 {
		return b.String() + "  " + m.subjectLoadingSpinner.View()
	}

	if m.subjectsLoading && len(m.subjects) > 0 {
		b.WriteString("  " + dimStyle.Render("Refreshing subject status... ") + m.subjectLoadingSpinner.View() + "\n")
	}

	if m.filteringSubject {
		filterBox := lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(styles.Accent).
			Padding(0, 1).
			Render("filter: " + m.subjectFilter + "\u2588")
		b.WriteString("  " + filterBox + "\n")
	} else if m.subjectFilter != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Render(
			fmt.Sprintf("  Filter: \"%s\"", m.subjectFilter)) + "  " +
			lipgloss.NewStyle().Foreground(styles.Muted).Render("[Esc to clear]") + "\n")
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

	b.WriteString(styles.RenderStatusCount(validCount, len(m.subjects), "valid"))
	if validCount < selectedCount {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render(fmt.Sprintf(" (%d invalid)", selectedCount-validCount)))
	}
	if m.subjectFilter != "" {
		b.WriteString(dimStyle.Render(fmt.Sprintf(" | %d shown", len(filteredSubjects))))
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

	bracketStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	for i := startIdx; i < endIdx; i++ {
		s := filteredSubjects[i]
		isSelected := m.subjectSelected[s.ID]
		isFocused := i == m.subjectCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + nameStyle.Render(s.ID)

		valid, reason := m.Pipeline.ValidateSubject(s)
		if m.Pipeline == types.PipelinePlotting {
			valid, reason = m.validatePlottingSubject(s)
		}

		var statusBadges []string
		if s.HasSourceData {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("S"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("\u00b7"))
		}
		if s.HasBids {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("B"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("\u00b7"))
		}
		if s.HasDerivatives {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("D"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("\u00b7"))
		}
		line += bracketStyle.Render(" [") + strings.Join(statusBadges, "") + bracketStyle.Render("]")
		if !valid {
			line += lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark + " " + reason)
		}
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
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
