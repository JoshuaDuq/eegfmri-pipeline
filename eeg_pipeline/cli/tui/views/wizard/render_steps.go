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

// renderAnimatedAccent returns an animated accent character based on ticker
func (m Model) renderAnimatedAccent() string {
	accentFrames := []string{"◆", "◇", "◆", "◈"}
	return lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Render(accentFrames[(m.ticker/3)%len(accentFrames)])
}

// calculateScrollWindow calculates visible line range for scrolling
func calculateScrollWindow(totalLines, offset, effectiveHeight, overhead int) (startLine, endLine int, showIndicators bool) {
	maxLines := effectiveHeight - overhead
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}

	if totalLines <= maxLines {
		return 0, totalLines, false
	}

	showIndicators = true
	startLine = offset
	if startLine < 0 {
		startLine = 0
	}
	if startLine > totalLines-maxLines {
		startLine = totalLines - maxLines
	}
	endLine = startLine + maxLines
	return startLine, endLine, showIndicators
}

const (
	defaultLabelWidth     = 22
	defaultLabelWidthWide = 30
	configOverhead        = 10
)

// renderDefaultConfigView renders the default configuration view when useDefaultAdvanced is true
func (m Model) renderDefaultConfigView(configType string) string {
	var b strings.Builder
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render(fmt.Sprintf("Default configuration will be used for %s.", configType)) + "\n")
	b.WriteString(infoStyle.Render("Press Space to customize settings.") + "\n\n")

	labelWidth := defaultLabelWidth
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

func (m Model) renderConfirmation() string {
	content := strings.Builder{}

	accent := m.renderAnimatedAccent()

	content.WriteString(accent + " " + styles.SectionTitleStyle.Render(" REVIEW & EXECUTE ") + "\n\n")

	content.WriteString(
		lipgloss.NewStyle().Foreground(styles.Text).Render("Pipeline:") +
			" " +
			lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.Pipeline.String()) +
			"\n",
	)

	selectedCount := countSelectedStringItems(m.subjectSelected)
	if selectedCount > 0 {
		subjectInfo := lipgloss.NewStyle().Foreground(styles.Text).Render("Subjects:") +
			" " +
			lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(fmt.Sprintf("%d", selectedCount))
		content.WriteString(subjectInfo + "\n")
	}

	// Show output paths
	content.WriteString("\n")
	content.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Underline(true).Render("Output locations:") + "\n")
	outputPaths := m.getExpectedOutputPaths()
	for _, p := range outputPaths {
		pathLine := lipgloss.NewStyle().Foreground(styles.Muted).Render("  → ") +
			lipgloss.NewStyle().Foreground(styles.Text).Render(p)
		content.WriteString(pathLine + "\n")
	}

	content.WriteString("\n")

	// Action buttons
	yesBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(styles.Success).
		Bold(true).
		Padding(0, 2).
		Render("▶ [Y] Execute")

	dryRunBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(styles.Warning).
		Bold(true).
		Padding(0, 2).
		Render("◇ [D] Dry-run")

	noBtn := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FFFFFF")).
		Background(styles.Error).
		Bold(true).
		Padding(0, 2).
		Render("✗ [N] Cancel")

	actions := lipgloss.JoinHorizontal(lipgloss.Left, yesBtn, "  ", dryRunBtn, "  ", noBtn)
	content.WriteString("  " + actions)

	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2)

	return boxStyle.Render(content.String())
}

// getExpectedOutputPaths returns the expected output directories for the current pipeline
func (m Model) getExpectedOutputPaths() []string {
	base := "derivatives/"
	switch m.Pipeline {
	case types.PipelinePreprocessing:
		return []string{base + "preprocessed/eeg/sub-XX/"}
	case types.PipelineFeatures:
		return []string{base + "sub-XX/eeg/features/"}
	case types.PipelineBehavior:
		return []string{base + "sub-XX/eeg/stats/", base + "group/eeg/stats/"}
	case types.PipelineML:
		return []string{base + "machine_learning/"}
	case types.PipelinePlotting:
		return []string{base + "sub-XX/eeg/plots/"}
	case types.PipelineFmri:
		return []string{base + "preprocessed/fmri/fmriprep/sub-XX/"}
	case types.PipelineFmriRawToBIDS:
		return []string{"bids_output/fmri/sub-XX/"}
	case types.PipelineRawToBIDS:
		return []string{"bids_output/eeg/sub-XX/eeg/"}
	default:
		return []string{base}
	}
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

	// Group computations by analysis purpose
	groups := map[string][]Computation{
		"DataPrep": {},
		"Core":     {},
		"Advanced": {},
		"Quality":  {},
	}
	groupNames := map[string]string{
		"DataPrep": "DATA PREPARATION",
		"Core":     "CORE ANALYSES",
		"Advanced": "ADVANCED/CAUSAL ANALYSES",
		"Quality":  "QUALITY & VALIDATION",
	}
	groupOrder := []string{"DataPrep", "Core", "Advanced", "Quality"}

	for _, comp := range m.computations {
		if comp.Group != "" {
			groups[comp.Group] = append(groups[comp.Group], comp)
		}
	}

	// Render each group
	for _, groupKey := range groupOrder {
		groupComps := groups[groupKey]
		if len(groupComps) == 0 {
			continue
		}

		// Group header
		b.WriteString("\n")
		b.WriteString(styles.SectionTitleStyle.Render(" "+groupNames[groupKey]+" ") + "\n\n")

		// Render computations in this group
		for _, comp := range groupComps {
			// Find index in main list
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

			// Show last computed timestamp if available
			if m.computationAvailability != nil && m.computationAvailability[comp.Key] {
				timestamp := m.computationLastModified[comp.Key]
				relTime := formatRelativeTime(timestamp)
				if relTime != "" {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render(fmt.Sprintf("  [%s]", relTime))
					b.WriteString(avail)
				} else {
					avail := lipgloss.NewStyle().Foreground(styles.Success).Render("  [DONE]")
					b.WriteString(avail)
				}
			} else {
				unavail := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  [NO DATA]")
				b.WriteString(unavail)
			}

			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + comp.Description)
			b.WriteString(desc + "\n")
		}
	}

	return b.String()
}

func (m Model) renderCategorySelection() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	title := "FEATURE CATEGORIES"
	if m.CurrentStep == types.StepSelectPlotCategories {
		title = "PLOT CATEGORIES"
	}
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" "+title) + "\n\n")

	if m.CurrentStep == types.StepSelectPlotCategories {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
			"Toggle a category to include or exclude all plots in that group.\n",
		))
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
			"Press 'g' to configure global styling options.\n\n",
		))
	}

	// If showing global styling panel, render that instead
	if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
		return m.renderGlobalStylingPanel()
	}

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

func (m Model) renderGlobalStylingPanel() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" GLOBAL STYLING") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).Render(
		"Configure styling options that apply to ALL plots.\n",
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

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" FREQUENCY BANDS") + "\n\n")

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

	b.WriteString("  " + statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.bands))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	// Instructions
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

		// Band name (editable)
		var nameDisplay string
		if isEditing && m.editingBandField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.bandEditBuffer + "▌")
		} else {
			nameDisplay = nameStyle.Render(band.Name)
		}

		// Frequency range display
		freqStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			freqStyle = lipgloss.NewStyle().Foreground(styles.Secondary)
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

		b.WriteString(checkbox + nameDisplay + freqRange)
		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderROISelection() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" REGIONS OF INTEREST") + "\n\n")

	count := 0
	for _, sel := range m.roiSelected {
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

	b.WriteString("  " + statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d selected", count, len(m.rois))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	// Instructions
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

		// ROI name (editable)
		var nameDisplay string
		if isEditing && m.editingROIField == 0 {
			nameDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditBuffer + "▌")
		} else {
			nameDisplay = nameStyle.Render(roi.Name)
		}

		// Channels display
		channelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			channelStyle = lipgloss.NewStyle().Foreground(styles.Secondary)
		}

		var channelsDisplay string
		if isEditing && m.editingROIField == 1 {
			channelsDisplay = lipgloss.NewStyle().Background(styles.Primary).Foreground(styles.BgDark).Render(m.roiEditBuffer + "▌")
		} else {
			// Render channels with unavailable ones in red
			channelsDisplay = m.renderChannelsWithUnavailable(roi.Channels, channelStyle, isFocused)
		}

		channelInfo := channelStyle.Render(" [") + channelsDisplay + channelStyle.Render("]")

		b.WriteString(checkbox + nameDisplay + channelInfo)
		b.WriteString("\n")
	}

	// Channel reference section
	b.WriteString("\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.Repeat("─", 60)) + "\n")

	// Compute used vs unused channels
	usedChannels, unusedChannels := m.computeChannelUsage()

	// Unused channels (available but not yet assigned to any ROI)
	if len(unusedChannels) > 0 {
		unusedLabel := lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render("Unused Channels")
		unusedCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(unusedChannels)))
		b.WriteString(unusedLabel + unusedCount + "\n")

		channelList := m.formatChannelList(unusedChannels, styles.Success)
		b.WriteString(channelList)
	} else if len(m.availableChannels) > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Italic(true).Render("  All channels assigned to ROIs ✓") + "\n")
	} else {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  No channel information available") + "\n")
	}

	// Used channels (already assigned to one or more ROIs)
	if len(usedChannels) > 0 {
		b.WriteString("\n")
		usedLabel := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render("Used Channels")
		usedCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(usedChannels)))
		b.WriteString(usedLabel + usedCount + "\n")

		channelList := m.formatChannelList(usedChannels, styles.Accent)
		b.WriteString(channelList)
	}

	// Unavailable channels (bad channels from preprocessing)
	if len(m.unavailableChannels) > 0 {
		b.WriteString("\n")
		unavailLabel := lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render("Unavailable Channels (bad)")
		unavailCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf(" (%d)", len(m.unavailableChannels)))
		b.WriteString(unavailLabel + unavailCount + "\n")

		channelList := m.formatChannelList(m.unavailableChannels, styles.Error)
		b.WriteString(channelList)
	}

	return b.String()
}

// computeChannelUsage partitions available channels into used (in ROIs) and unused
func (m Model) computeChannelUsage() (used, unused []string) {
	if len(m.availableChannels) == 0 {
		return nil, nil
	}

	// Build set of channels used across all ROIs
	usedSet := make(map[string]bool)
	for _, roi := range m.rois {
		for _, ch := range strings.Split(roi.Channels, ",") {
			ch = strings.TrimSpace(ch)
			if ch != "" {
				usedSet[strings.ToUpper(ch)] = true
			}
		}
	}

	// Partition available channels
	for _, ch := range m.availableChannels {
		if usedSet[strings.ToUpper(ch)] {
			used = append(used, ch)
		} else {
			unused = append(unused, ch)
		}
	}

	return used, unused
}

// renderChannelsWithUnavailable renders channel list with unavailable channels in red
func (m Model) renderChannelsWithUnavailable(channelsStr string, baseStyle lipgloss.Style, isFocused bool) string {
	if channelsStr == "" {
		return baseStyle.Render(channelsStr)
	}

	// Build unavailable set for quick lookup
	unavailableSet := make(map[string]bool)
	for _, ch := range m.unavailableChannels {
		unavailableSet[strings.ToUpper(strings.TrimSpace(ch))] = true
	}

	// Split channels
	parts := strings.Split(channelsStr, ",")
	var channels []string
	for _, ch := range parts {
		ch = strings.TrimSpace(ch)
		if ch != "" {
			channels = append(channels, ch)
		}
	}

	// Check if we need to truncate (check raw string length, not rendered)
	rawLength := len(channelsStr)
	needsTruncation := rawLength > 40

	if needsTruncation {
		// Truncate the raw channel list first
		var truncatedChannels []string
		var truncatedLength int
		for _, ch := range channels {
			// Account for comma separator
			chLength := len(ch)
			if len(truncatedChannels) > 0 {
				chLength += 2 // ", " separator
			}
			if truncatedLength+chLength > 37 {
				break
			}
			truncatedChannels = append(truncatedChannels, ch)
			truncatedLength += chLength
		}
		channels = truncatedChannels
	}

	// Render each channel with appropriate styling
	var renderedParts []string
	for i, ch := range channels {
		if i > 0 {
			renderedParts = append(renderedParts, baseStyle.Render(", "))
		}

		// Check if this channel is unavailable
		if unavailableSet[strings.ToUpper(ch)] {
			// Render unavailable channel in red
			unavailStyle := lipgloss.NewStyle().Foreground(styles.Error)
			if isFocused {
				unavailStyle = unavailStyle.Bold(true)
			}
			renderedParts = append(renderedParts, unavailStyle.Render(ch))
		} else {
			// Render available channel normally
			renderedParts = append(renderedParts, baseStyle.Render(ch))
		}
	}

	result := strings.Join(renderedParts, "")

	// Add ellipsis if truncated
	if needsTruncation {
		// Count non-empty parts from original
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

// formatChannelList formats a list of channel names into wrapped lines
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
	b.WriteString(styles.SectionTitleStyle.Render(" SPATIAL AGGREGATION ") + "\n\n")

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

		desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + mode.Description)
		b.WriteString(desc + "\n")
	}

	return b.String()
}

func (m Model) renderFeatureFileSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" FEATURE FILES ") + "\n\n")

	// Get only the features applicable for selected computations
	applicableFeatures := m.GetApplicableFeatureFiles()

	instruction := "  Select which feature files to load for analysis.\n"
	if len(applicableFeatures) < len(featureFileOptions) {
		instruction = "  Showing features applicable for selected computations.\n"
	}

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		instruction) +
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

	// Adjust cursor to be within visible range
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

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" PLOTS") + "\n\n")

	// Build visible items list
	visibleItems := []int{}
	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		visibleItems = append(visibleItems, i)
	}

	// Calculate counts
	count := 0
	for _, idx := range visibleItems {
		if m.plotSelected[idx] {
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
		fmt.Sprintf("%d of %d selected", count, len(visibleItems))))
	if count == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	// Simple checkbox list grouped by category
	currentGroup := ""
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}

		// Group header
		if plot.Group != currentGroup {
			if currentGroup != "" {
				b.WriteString("\n")
			}
			b.WriteString(groupStyle.Render("◆ "+strings.ToUpper(plot.Group)) + "\n")
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

		// Show plot ID and name for clarity
		b.WriteString(checkbox + idStyle.Render(plot.ID) + nameStyle.Render(" — "+plot.Name))

		b.WriteString("\n")
	}

	// Data requirements section below the list
	if m.plotCursor >= 0 && m.plotCursor < len(m.plotItems) {
		plot := m.plotItems[m.plotCursor]

		b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Render(strings.Repeat("─", 50)) + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Data Requirements:") + "\n")

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

		// Availability status
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
	b.WriteString(styles.SectionTitleStyle.Render(" PLOT OUTPUT ") + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Configure output formats and resolution. Use Space to toggle/cycle.\n\n"))

	options := m.getPlotConfigOptions()
	labelWidth := 16 // Plot config uses narrower width

	for i, opt := range options {
		isFocused := i == m.plotConfigCursor
		cursor := ""
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
	b.WriteString(styles.SectionTitleStyle.Render(" TIME RANGE ") + "\n\n")

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

func parseFloat(s string, defaultVal float64) float64 {
	if s == "" || s == "default" || s == "none" {
		return defaultVal
	}
	var val float64
	fmt.Sscanf(s, "%f", &val)
	return val
}

func (m Model) renderPlotSelectionSplit() string {
	// For wide screens, just use the same simple layout as narrow
	// This keeps the interface consistent
	return m.renderPlotSelection()
}

func (m Model) renderFeaturePlotterSelection() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" FEATURE PLOTS ") + "\n\n")

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
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  Loading available feature plots...\n"))
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
	cursorLineIdx := 0 // Track which line the cursor is on

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

	// Calculate layout using centralized function
	layout := styles.CalculateListLayout(m.height, cursorLineIdx, len(lines), 10)

	// Show scroll up indicator
	if layout.ShowScrollUp {
		b.WriteString(styles.RenderScrollUpIndicator(layout.StartIdx) + "\n")
	}

	for i := layout.StartIdx; i < layout.EndIdx; i++ {
		b.WriteString(" " + lines[i].text + "\n")
	}

	// Show scroll down indicator
	if layout.ShowScrollDn {
		remaining := len(lines) - layout.EndIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	return b.String()
}

func (m Model) renderSubjectSelection() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" SUBJECT SELECTION") + "\n\n")

	if m.Pipeline == types.PipelineML {
		scopeLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render("Scope:")
		selectedChip := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Padding(0, 1)
		unselectedChip := lipgloss.NewStyle().
			Foreground(styles.TextDim).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(styles.TextDim).
			Padding(0, 1)

		groupChip := unselectedChip.Render("Group (LOSO)")
		subjectChip := unselectedChip.Render("Subject (within)")
		if m.mlScope == MLCVScopeGroup {
			groupChip = selectedChip.Render("Group (LOSO)")
		} else {
			subjectChip = selectedChip.Render("Subject (within)")
		}

		b.WriteString("  " + scopeLabel + " " + groupChip + " " + subjectChip + " " +
			lipgloss.NewStyle().Foreground(styles.Muted).Render("[Tab to toggle]") + "\n\n")
	}

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

	// Calculate responsive layout based on terminal height
	// Overhead: header(4) + title(2) + status(2) + footer(2) + legend(2) = 12
	overhead := 12
	if m.Pipeline == types.PipelineML {
		overhead += 2 // scope line + spacer
	}
	layout := styles.CalculateListLayout(m.height, m.subjectCursor, len(filteredSubjects), overhead)
	startIdx := layout.StartIdx
	endIdx := layout.EndIdx

	// Show scroll up indicator
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
			indicator := lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark + " " + reason)
			b.WriteString(indicator)
		}

		b.WriteString("\n")
	}

	// Show scroll down indicator
	if layout.ShowScrollDn {
		remaining := len(filteredSubjects) - endIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	// Status line with current position
	if len(filteredSubjects) > layout.MaxItems {
		b.WriteString("\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf("  Showing %d-%d of %d  [↑↓ to scroll]", startIdx+1, endIdx, len(filteredSubjects))))
	}

	b.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).
		Render("  Legend: [S]=Source Data [B]=BIDS [D]=Derivatives"))

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
	case types.PipelinePlotting:
		return m.renderPlottingAdvancedConfigV2()
	case types.PipelineML:
		return m.renderMLAdvancedConfig()
	case types.PipelinePreprocessing:
		return m.renderPreprocessingAdvancedConfig()
	case types.PipelineFmri:
		return m.renderFmriAdvancedConfig()
	case types.PipelineRawToBIDS:
		return m.renderRawToBidsAdvancedConfig()
	case types.PipelineFmriRawToBIDS:
		return m.renderFmriRawToBidsAdvancedConfig()
	case types.PipelineMergePsychoPyData:
		return m.renderMergeBehaviorAdvancedConfig()
	default:
		return m.renderDefaultAdvancedConfig()
	}
}

func (m Model) renderFeaturesAdvancedConfig() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" ADVANCED CONFIGURATION") + "\n\n")

	// Contextual help text
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("feature parameters")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter a value, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space to toggle item · Esc to close submenu") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to toggle/expand · ↑↓ to navigate · Enter to proceed") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Prepare values for display
	peOrderVal := fmt.Sprintf("%d", m.complexityPEOrder)
	peDelayVal := fmt.Sprintf("%d", m.complexityPEDelay)
	burstThreshVal := fmt.Sprintf("%.1f z", m.burstThresholdZ)
	burstMinDurVal := fmt.Sprintf("%d ms", m.burstMinDuration)
	erpBaselineVal := m.boolToOnOff(m.erpBaselineCorrection)
	erpAllowNoBaselineVal := m.boolToOnOff(m.erpAllowNoBaseline)
	erpComponentsVal := m.erpComponentsSpec
	if strings.TrimSpace(erpComponentsVal) == "" {
		erpComponentsVal = "(default)"
	}
	powerBaselineVal := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}[m.powerBaselineMode]
	powerRequireBaselineVal := m.boolToOnOff(m.powerRequireBaseline)
	spectralEdgeVal := fmt.Sprintf("%.0f%%", m.spectralEdgePercentile*100)
	spectralRatioPairsVal := m.spectralRatioPairsSpec
	if strings.TrimSpace(spectralRatioPairsVal) == "" {
		spectralRatioPairsVal = "(default)"
	}
	connOutputVal := []string{"full", "global_only"}[m.connOutputLevel]
	connGraphVal := m.boolToOnOff(m.connGraphMetrics)
	connGraphPropVal := fmt.Sprintf("%.2f", m.connGraphProp)
	connWindowLenVal := fmt.Sprintf("%.1f s", m.connWindowLen)
	connWindowStepVal := fmt.Sprintf("%.1f s", m.connWindowStep)
	connAecVal := []string{"orth", "none", "sym"}[m.connAECMode]
	pacPhaseVal := fmt.Sprintf("%.1f-%.1f Hz", m.pacPhaseMin, m.pacPhaseMax)
	pacAmpVal := fmt.Sprintf("%.1f-%.1f Hz", m.pacAmpMin, m.pacAmpMax)
	pacMethodVal := []string{"mvl", "kl", "tort", "ozkurt"}[m.pacMethod]
	pacMinEpochsVal := fmt.Sprintf("%d", m.pacMinEpochs)
	pacPairsVal := m.pacPairsSpec
	if strings.TrimSpace(pacPairsVal) == "" {
		pacPairsVal = "(default)"
	}
	burstBandsVal := m.burstBandsSpec
	if strings.TrimSpace(burstBandsVal) == "" {
		burstBandsVal = "(default)"
	}
	asymPairsVal := m.asymmetryChannelPairsSpec
	if strings.TrimSpace(asymPairsVal) == "" {
		asymPairsVal = "(default)"
	}
	aperiodicPeakZVal := fmt.Sprintf("%.1f", m.aperiodicPeakZ)
	aperiodicR2Val := fmt.Sprintf("%.2f", m.aperiodicMinR2)
	aperiodicPointsVal := fmt.Sprintf("%d", m.aperiodicMinPoints)
	minEpochsVal := fmt.Sprintf("%d", m.minEpochsForFeatures)

	// Input overrides
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optPEOrder):
			peOrderVal = buffer
		case m.isCurrentlyEditing(optPEDelay):
			peDelayVal = buffer
		case m.isCurrentlyEditing(optBurstThreshold):
			burstThreshVal = buffer
		case m.isCurrentlyEditing(optBurstMinDuration):
			burstMinDurVal = buffer
		case m.isCurrentlyEditing(optConnGraphProp):
			connGraphPropVal = buffer
		case m.isCurrentlyEditing(optConnWindowLen):
			connWindowLenVal = buffer
		case m.isCurrentlyEditing(optConnWindowStep):
			connWindowStepVal = buffer
		case m.isCurrentlyEditing(optPACMinEpochs):
			pacMinEpochsVal = buffer
		case m.isCurrentlyEditing(optAperiodicPeakZ):
			aperiodicPeakZVal = buffer
		case m.isCurrentlyEditing(optAperiodicMinR2):
			aperiodicR2Val = buffer
		case m.isCurrentlyEditing(optAperiodicMinPoints):
			aperiodicPointsVal = buffer
		case m.isCurrentlyEditing(optMinEpochs):
			minEpochsVal = buffer
		}
	}
	if m.editingText {
		buffer := m.textBuffer + "█"
		switch m.editingTextField {
		case textFieldPACPairs:
			pacPairsVal = buffer
		case textFieldBurstBands:
			burstBandsVal = buffer
		case textFieldSpectralRatioPairs:
			spectralRatioPairsVal = buffer
		case textFieldAsymmetryChannelPairs:
			asymPairsVal = buffer
		case textFieldERPComponents:
			erpComponentsVal = buffer
		}
	}

	options := m.getFeaturesOptions()

	// Calculate total lines including expanded connectivity measures
	totalLines := len(options)
	if m.expandedOption == expandedConnectivityMeasures {
		totalLines += len(connectivityMeasures)
	}
	if m.expandedOption == expandedDirectedConnMeasures {
		totalLines += len(directedConnectivityMeasures)
	}
	if m.expandedOption == expandedFmriCondAColumn {
		totalLines += len(m.fmriDiscoveredColumns)
	}
	if m.expandedOption == expandedFmriCondAValue {
		totalLines += len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn))
	}
	if m.expandedOption == expandedFmriCondBColumn {
		totalLines += len(m.fmriDiscoveredColumns)
	}
	if m.expandedOption == expandedFmriCondBValue {
		totalLines += len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn))
	}
	if m.expandedOption == expandedItpcConditionColumn {
		totalLines += len(m.availableColumns)
	}
	if m.expandedOption == expandedItpcConditionValues {
		totalLines += len(m.GetDiscoveredColumnValues(m.itpcConditionColumn))
	}
	if m.expandedOption == expandedConnConditionColumn {
		totalLines += len(m.availableColumns)
	}
	if m.expandedOption == expandedConnConditionValues {
		totalLines += len(m.GetDiscoveredColumnValues(m.connConditionColumn))
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	// Show scroll indicator for items above
	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", startLine)) + "\n")
	}

	lineIdx := 0

	for i, opt := range options {
		isFocused := i == m.advancedCursor && m.expandedOption < 0
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}

		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		label := ""
		value := ""
		hint := ""
		expandIndicator := ""

		switch opt {
		case optUseDefaults:
			label = "Configuration"
			if m.useDefaultAdvanced {
				value = "Using Defaults"
				hint = "Press Space to customize"
				expandIndicator = " ▶"
			} else {
				value = "Custom"
				hint = "Press Space to use defaults"
				expandIndicator = " ▼"
			}

		// Section headers - styled as distinct groups with chevron indicators
		case optFeatGroupConnectivity:
			label = "▸ Connectivity"
			hint = "Space to toggle"
			if m.featGroupConnectivityExpanded {
				label = "▾ Connectivity"
				value = ""
				expandIndicator = ""
			} else {
				value = ""
				expandIndicator = ""
			}
			// Use distinct styling for section headers
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupDirectedConnectivity:
			label = "▸ Directed Connectivity"
			hint = "Space to toggle"
			if m.featGroupDirectedConnExpanded {
				label = "▾ Directed Connectivity"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			}
		case optFeatGroupPAC:
			label = "▸ PAC / CFC"
			hint = "Space to toggle"
			if m.featGroupPACExpanded {
				label = "▾ PAC / CFC"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupAperiodic:
			label = "▸ Aperiodic"
			hint = "Space to toggle"
			if m.featGroupAperiodicExpanded {
				label = "▾ Aperiodic"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupComplexity:
			label = "▸ Complexity"
			hint = "Space to toggle"
			if m.featGroupComplexityExpanded {
				label = "▾ Complexity"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupBursts:
			label = "▸ Bursts"
			hint = "Space to toggle"
			if m.featGroupBurstsExpanded {
				label = "▾ Bursts"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupPower:
			label = "▸ Power"
			hint = "Space to toggle"
			if m.featGroupPowerExpanded {
				label = "▾ Power"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupSpectral:
			label = "▸ Spectral"
			hint = "Space to toggle"
			if m.featGroupSpectralExpanded {
				label = "▾ Spectral"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupERP:
			label = "▸ ERP"
			hint = "Space to toggle"
			if m.featGroupERPExpanded {
				label = "▾ ERP"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupRatios:
			label = "▸ Ratios"
			hint = "Space to toggle"
			if m.featGroupRatiosExpanded {
				label = "▾ Ratios"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupAsymmetry:
			label = "▸ Asymmetry"
			hint = "Space to toggle"
			if m.featGroupAsymmetryExpanded {
				label = "▾ Asymmetry"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupSpatialTransform:
			label = "▸ Spatial Transform"
			hint = "Space to toggle"
			if m.featGroupSpatialTransformExpanded {
				label = "▾ Spatial Transform"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optSpatialTransform:
			label = "Transform"
			transforms := []string{"none", "CSD", "Laplacian"}
			value = transforms[m.spatialTransform]
			hint = "volume conduction reduction"
		case optSpatialTransformLambda2:
			label = "Lambda2"
			value = fmt.Sprintf("%.2e", m.spatialTransformLambda2)
			if m.editingNumber && m.isCurrentlyEditing(optSpatialTransformLambda2) {
				value = m.numberBuffer + "█"
			}
			hint = "regularization parameter"
		case optSpatialTransformStiffness:
			label = "Stiffness"
			value = fmt.Sprintf("%.1f", m.spatialTransformStiffness)
			if m.editingNumber && m.isCurrentlyEditing(optSpatialTransformStiffness) {
				value = m.numberBuffer + "█"
			}
			hint = "spline stiffness"
		case optFeatGroupTFR:
			label = "▸ Time-Frequency"
			hint = "Space to toggle"
			if m.featGroupTFRExpanded {
				label = "▾ Time-Frequency"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optTfrFreqMin:
			label = "Freq Min"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMin)
			if m.editingNumber && m.isCurrentlyEditing(optTfrFreqMin) {
				value = m.numberBuffer + "█"
			}
			hint = "min TFR frequency"
		case optTfrFreqMax:
			label = "Freq Max"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMax)
			if m.editingNumber && m.isCurrentlyEditing(optTfrFreqMax) {
				value = m.numberBuffer + "█"
			}
			hint = "max TFR frequency"
		case optTfrNFreqs:
			label = "N Frequencies"
			value = fmt.Sprintf("%d", m.tfrNFreqs)
			if m.editingNumber && m.isCurrentlyEditing(optTfrNFreqs) {
				value = m.numberBuffer + "█"
			}
			hint = "number of freq bins"
		case optTfrMinCycles:
			label = "Min Cycles"
			value = fmt.Sprintf("%.1f", m.tfrMinCycles)
			if m.editingNumber && m.isCurrentlyEditing(optTfrMinCycles) {
				value = m.numberBuffer + "█"
			}
			hint = "wavelet cycles floor"
		case optTfrNCyclesFactor:
			label = "Cycles Factor"
			value = fmt.Sprintf("%.1f", m.tfrNCyclesFactor)
			if m.editingNumber && m.isCurrentlyEditing(optTfrNCyclesFactor) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles per Hz"
		case optTfrWorkers:
			label = "Workers"
			value = fmt.Sprintf("%d", m.tfrWorkers)
			if m.editingNumber && m.isCurrentlyEditing(optTfrWorkers) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optBandEnvelopePadSec:
			label = "Envelope pad (sec)"
			value = fmt.Sprintf("%.2f", m.bandEnvelopePadSec)
			if m.editingNumber && m.isCurrentlyEditing(optBandEnvelopePadSec) {
				value = m.numberBuffer + "█"
			}
			hint = "padding for Hilbert envelopes"
		case optBandEnvelopePadCycles:
			label = "Envelope pad (cycles)"
			value = fmt.Sprintf("%.1f", m.bandEnvelopePadCycles)
			if m.editingNumber && m.isCurrentlyEditing(optBandEnvelopePadCycles) {
				value = m.numberBuffer + "█"
			}
			hint = "padding scaled by fmin"
		case optIAFEnabled:
			label = "IAF enabled"
			value = m.boolToOnOff(m.iafEnabled)
			hint = "individualized alpha frequency"
		case optIAFAlphaWidthHz:
			label = "IAF alpha width"
			value = fmt.Sprintf("%.1f Hz", m.iafAlphaWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optIAFAlphaWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "alpha band half-width"
		case optIAFSearchRangeMin:
			label = "IAF search min"
			value = fmt.Sprintf("%.1f Hz", m.iafSearchRangeMin)
			if m.editingNumber && m.isCurrentlyEditing(optIAFSearchRangeMin) {
				value = m.numberBuffer + "█"
			}
			hint = "search range lower bound"
		case optIAFSearchRangeMax:
			label = "IAF search max"
			value = fmt.Sprintf("%.1f Hz", m.iafSearchRangeMax)
			if m.editingNumber && m.isCurrentlyEditing(optIAFSearchRangeMax) {
				value = m.numberBuffer + "█"
			}
			hint = "search range upper bound"
		case optIAFMinProminence:
			label = "IAF prominence"
			value = fmt.Sprintf("%.3f", m.iafMinProminence)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinProminence) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum PSD peak prominence"
		case optIAFRois:
			label = "IAF ROIs"
			rois := m.SelectedROIs()
			if len(rois) == 0 {
				value = "(select ROIs in ROI step)"
			} else {
				value = strings.Join(rois, ",")
			}
			hint = "derived from ROI selection step"
		case optIAFMinCyclesAtFmin:
			label = "IAF min cycles"
			value = fmt.Sprintf("%.1f", m.iafMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at iaf search fmin"
		case optIAFMinBaselineSec:
			label = "IAF min baseline"
			value = fmt.Sprintf("%.2f s", m.iafMinBaselineSec)
			if m.editingNumber && m.isCurrentlyEditing(optIAFMinBaselineSec) {
				value = m.numberBuffer + "█"
			}
			hint = "additional baseline duration"
		case optIAFAllowFullFallback:
			label = "Allow full fallback"
			value = m.boolToOnOff(m.iafAllowFullFallback)
			hint = "use full segment if baseline missing"
		case optIAFAllowAllChannelsFallback:
			label = "Allow channels fallback"
			value = m.boolToOnOff(m.iafAllowAllChannelsFallback)
			hint = "use all channels if ROIs missing"
		case optFeatGroupStorage:
			label = "▸ Storage"
			hint = "Space to toggle"
			if m.featGroupStorageExpanded {
				label = "▾ Storage"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optSaveSubjectLevelFeatures:
			label = "Save Subject-Level"
			value = m.boolToOnOff(m.saveSubjectLevelFeatures)
			hint = "Space to toggle"
		case optFeatAlsoSaveCsv:
			label = "Also Save CSV"
			value = m.boolToOnOff(m.featAlsoSaveCsv)
			hint = "save feature tables as both parquet and CSV"
		case optFeatGroupExecution:
			label = "▸ Execution"
			hint = "Space to toggle"
			if m.featGroupExecutionExpanded {
				label = "▾ Execution"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optFeatGroupSourceLoc:
			label = "▸ Source Localization"
			hint = "Space to toggle"
			if m.featGroupSourceLocExpanded {
				label = "▾ Source Localization"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		// Connectivity settings
		case optConnectivity:
			label = "Measures"
			value = m.selectedConnectivityDisplay()
			hint = "Select measures"
			if m.expandedOption != expandedConnectivityMeasures {
				expandIndicator = " [+]"
			} else {
				expandIndicator = " [-]"
			}
		case optConnOutputLevel:
			label = "Output Level"
			value = connOutputVal
			hint = "full / global_only"
		case optConnGranularity:
			label = "Granularity"
			granularities := []string{"trial", "condition", "subject"}
			value = granularities[m.connGranularity]
			hint = "trial / condition / subject"
		case optConnConditionColumn:
			label = "Condition Column"
			val := m.connConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldConnConditionColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.availableColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.availableColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedConnConditionColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optConnConditionValues:
			label = "Condition Values"
			if m.connConditionColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.connConditionValues
				if val == "" {
					val = "(select values)"
				}
				if m.editingText && m.editingTextField == textFieldConnConditionValues {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetDiscoveredColumnValues(m.connConditionColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.connConditionColumn)
				}
				hint = "Space to select (others excluded)" + expandIndicatorHint
				if m.expandedOption == expandedConnConditionValues {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optConnPhaseEstimator:
			label = "Phase estimator"
			estimators := []string{"within_epoch", "across_epochs"}
			value = estimators[m.connPhaseEstimator]
			hint = "within_epoch / across_epochs"
		case optConnMinEpochsPerGroup:
			label = "Min epochs/group"
			value = fmt.Sprintf("%d", m.connMinEpochsPerGroup)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinEpochsPerGroup) {
				value = m.numberBuffer + "█"
			}
			hint = "for condition/subject granularity"
		case optConnMinCyclesPerBand:
			label = "Min cycles/band"
			value = fmt.Sprintf("%.1f", m.connMinCyclesPerBand)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinCyclesPerBand) {
				value = m.numberBuffer + "█"
			}
			hint = "recommended cycles at band fmin"
		case optConnMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.1f", m.connMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optConnMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum duration"
		case optConnWarnNoSpatialTransform:
			label = "Warn (no transform)"
			value = m.boolToOnOff(m.connWarnNoSpatialTransform)
			hint = "warn when phase measures lack CSD/Laplacian"
		case optConnGraphMetrics:
			label = "Graph Metrics"
			value = connGraphVal
			hint = "toggles metric computation"
		case optConnGraphProp:
			label = "Graph Threshold"
			value = connGraphPropVal
			hint = "top edges density"
		case optConnWindowLen:
			label = "Window length"
			value = connWindowLenVal
			hint = "seconds per slice"
		case optConnWindowStep:
			label = "Window step"
			value = connWindowStepVal
			hint = "overlap amount"
		case optConnAECMode:
			label = "AEC Mode"
			value = connAecVal
			hint = "orth/none/sym"
		case optConnAECOutput:
			label = "AEC Output"
			switch m.connAECOutput {
			case 1:
				value = "z"
			case 2:
				value = "r+z"
			default:
				value = "r"
			}
			hint = "r (raw), z (Fisher-z), or both"
		case optConnForceWithinEpochML:
			label = "Force within_epoch"
			value = m.boolToOnOff(m.connForceWithinEpochML)
			hint = "CV/ML leakage safety"

		// Directed Connectivity (PSI, DTF, PDC)
		case optDirectedConnMeasures:
			label = "Measures"
			value = m.selectedDirectedConnectivityDisplay()
			hint = "Select directed measures"
			if m.expandedOption != expandedDirectedConnMeasures {
				expandIndicator = " [+]"
			} else {
				expandIndicator = " [-]"
			}
		case optDirectedConnOutputLevel:
			label = "Output Level"
			outputLevels := []string{"full", "global_only"}
			value = outputLevels[m.directedConnOutputLevel]
			hint = "full / global_only"
		case optDirectedConnMvarOrder:
			label = "MVAR Order"
			value = fmt.Sprintf("%d", m.directedConnMvarOrder)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnMvarOrder) {
				value = m.numberBuffer + "█"
			}
			hint = "model order for DTF/PDC"
		case optDirectedConnNFreqs:
			label = "N Freqs"
			value = fmt.Sprintf("%d", m.directedConnNFreqs)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnNFreqs) {
				value = m.numberBuffer + "█"
			}
			hint = "frequency bins"
		case optDirectedConnMinSegSamples:
			label = "Min Seg Samples"
			value = fmt.Sprintf("%d", m.directedConnMinSegSamples)
			if m.editingNumber && m.isCurrentlyEditing(optDirectedConnMinSegSamples) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum samples per segment"

		// Source Localization (LCMV, eLORETA)
		case optSourceLocMode:
			label = "Mode"
			modes := []string{"EEG-only (template)", "fMRI-informed"}
			value = modes[m.sourceLocMode]
			hint = "template or subject-specific (requires fmri-prep)"
		case optSourceLocMethod:
			label = "Method"
			methods := []string{"LCMV", "eLORETA"}
			value = methods[m.sourceLocMethod]
			hint = "beamformer type"
		case optSourceLocSpacing:
			label = "Spacing"
			spacings := []string{"oct5", "oct6", "ico4", "ico5"}
			value = spacings[m.sourceLocSpacing]
			hint = "source space resolution"
		case optSourceLocParc:
			label = "Parcellation"
			parcs := []string{"aparc", "aparc.a2009s", "HCPMMP1"}
			value = parcs[m.sourceLocParc]
			hint = "cortical atlas"
		case optSourceLocReg:
			label = "Regularization"
			value = fmt.Sprintf("%.3f", m.sourceLocReg)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocReg) {
				value = m.numberBuffer + "█"
			}
			hint = "LCMV regularization"
		case optSourceLocSnr:
			label = "SNR"
			value = fmt.Sprintf("%.1f", m.sourceLocSnr)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocSnr) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA signal-to-noise"
		case optSourceLocLoose:
			label = "Loose"
			value = fmt.Sprintf("%.2f", m.sourceLocLoose)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocLoose) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA loose constraint"
		case optSourceLocDepth:
			label = "Depth"
			value = fmt.Sprintf("%.2f", m.sourceLocDepth)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocDepth) {
				value = m.numberBuffer + "█"
			}
			hint = "eLORETA depth weighting"
		case optSourceLocConnMethod:
			label = "Connectivity"
			connMethods := []string{"AEC", "wPLI", "PLV"}
			value = connMethods[m.sourceLocConnMethod]
			hint = "source-space connectivity"
		case optSourceLocSubject:
			label = "FS Subject"
			if strings.TrimSpace(m.sourceLocSubject) == "" {
				value = "(auto)"
			} else {
				value = m.sourceLocSubject
			}
			if m.editingText && m.editingTextField == textFieldSourceLocSubject {
				value = m.textBuffer + "█"
			}
			hint = "FreeSurfer subject name"
		case optSourceLocTrans:
			label = "Coreg trans"
			if strings.TrimSpace(m.sourceLocTrans) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocTrans
			}
			hint = "EEG↔MRI transform .fif"
		case optSourceLocBem:
			label = "BEM sol"
			if strings.TrimSpace(m.sourceLocBem) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocBem
			}
			hint = "BEM solution .fif"
		case optSourceLocMindistMm:
			label = "Mindist (mm)"
			value = fmt.Sprintf("%.1f", m.sourceLocMindistMm)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocMindistMm) {
				value = m.numberBuffer + "█"
			}
			hint = "min distance to inner skull"
		case optSourceLocFmriEnabled:
			label = "fMRI Prior"
			if m.sourceLocFmriEnabled {
				value = "on"
			} else {
				value = "off"
			}
			hint = "restrict volume sources using fMRI"
		case optSourceLocFmriStatsMap:
			label = "fMRI Stats Map"
			if strings.TrimSpace(m.sourceLocFmriStatsMap) == "" {
				value = "(unset)"
			} else {
				value = m.sourceLocFmriStatsMap
			}
			hint = "NIfTI map in FS MRI space"
		case optSourceLocFmriProvenance:
			label = "fMRI Provenance"
			prov := []string{"independent", "same_dataset"}
			if m.sourceLocFmriProvenance >= 0 && m.sourceLocFmriProvenance < len(prov) {
				value = prov[m.sourceLocFmriProvenance]
			} else {
				value = "independent"
			}
			hint = "independent (recommended) vs same_dataset"
		case optSourceLocFmriRequireProvenance:
			label = "Require provenance"
			if m.sourceLocFmriRequireProv {
				value = "on"
			} else {
				value = "off"
			}
			hint = "error if provenance unknown"
		case optSourceLocFmriThreshold:
			label = "fMRI Threshold"
			value = fmt.Sprintf("%.2f", m.sourceLocFmriThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriThreshold) {
				value = m.numberBuffer + "█"
			}
			hint = "e.g., z>=3.10"
		case optSourceLocFmriTail:
			label = "fMRI Tail"
			tails := []string{"pos", "abs"}
			value = tails[m.sourceLocFmriTail]
			hint = "pos or abs"
		case optSourceLocFmriMinClusterVox:
			label = "fMRI Min Voxels"
			value = fmt.Sprintf("%d", m.sourceLocFmriMinClusterVox)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMinClusterVox) {
				value = m.numberBuffer + "█"
			}
			hint = "min cluster size"
		case optSourceLocFmriMaxClusters:
			label = "fMRI Max Clusters"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxClusters)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxClusters) {
				value = m.numberBuffer + "█"
			}
			hint = "limit ROI clusters"
		case optSourceLocFmriMaxVoxPerClus:
			label = "fMRI Max Vox/Cluster"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxVoxPerClus)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxVoxPerClus) {
				value = m.numberBuffer + "█"
			}
			hint = "subsample per cluster"
		case optSourceLocFmriMaxTotalVox:
			label = "fMRI Max Total Vox"
			value = fmt.Sprintf("%d", m.sourceLocFmriMaxTotalVox)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriMaxTotalVox) {
				value = m.numberBuffer + "█"
			}
			hint = "subsample total"
		case optSourceLocFmriRandomSeed:
			label = "fMRI Random Seed"
			value = fmt.Sprintf("%d", m.sourceLocFmriRandomSeed)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriRandomSeed) {
				value = m.numberBuffer + "█"
			}
			hint = "0 = nondeterministic"

		// BEM/Trans generation options (Docker-based)
		case optSourceLocCreateTrans:
			label = "Create Trans"
			if m.sourceLocCreateTrans {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"
		case optSourceLocCreateBemModel:
			label = "Create BEM Model"
			if m.sourceLocCreateBemModel {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"
		case optSourceLocCreateBemSolution:
			label = "Create BEM Solution"
			if m.sourceLocCreateBemSolution {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-generate via Docker (requires Docker)"

		// fMRI GLM Contrast Builder options
		case optSourceLocFmriContrastEnabled:
			label = "Build Contrast"
			if m.sourceLocFmriContrastEnabled {
				value = "on"
			} else {
				value = "off"
			}
			hint = "build from BOLD vs. load pre-computed"
		case optSourceLocFmriContrastType:
			label = "Contrast Type"
			contrastTypes := []string{"t-test", "paired t-test", "F-test", "custom formula"}
			value = contrastTypes[m.sourceLocFmriContrastType]
			hint = "statistical test type"
		case optSourceLocFmriCondAColumn:
			label = "Cond A Column"
			val := m.sourceLocFmriCondAColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriCondAColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.fmriDiscoveredColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.fmriDiscoveredColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriCondAColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriCondAValue:
			label = "Cond A Value"
			if m.sourceLocFmriCondAColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.sourceLocFmriCondAValue
				if val == "" {
					val = "(select value)"
				}
				if m.editingText && m.editingTextField == textFieldSourceLocFmriCondAValue {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.sourceLocFmriCondAColumn)
				}
				hint = "Space to select" + expandIndicatorHint
				if m.expandedOption == expandedFmriCondAValue {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optSourceLocFmriCondBColumn:
			label = "Cond B Column"
			val := m.sourceLocFmriCondBColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriCondBColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.fmriDiscoveredColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.fmriDiscoveredColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedFmriCondBColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optSourceLocFmriCondBValue:
			label = "Cond B Value"
			if m.sourceLocFmriCondBColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.sourceLocFmriCondBValue
				if val == "" {
					val = "(select value)"
				}
				if m.editingText && m.editingTextField == textFieldSourceLocFmriCondBValue {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.sourceLocFmriCondBColumn)
				}
				hint = "Space to select" + expandIndicatorHint
				if m.expandedOption == expandedFmriCondBValue {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optSourceLocFmriContrastFormula:
			label = "Formula"
			if strings.TrimSpace(m.sourceLocFmriContrastFormula) == "" {
				value = "(e.g., pain_high - pain_low)"
			} else {
				value = m.sourceLocFmriContrastFormula
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriContrastFormula {
				value = m.textBuffer + "█"
			}
			hint = "custom contrast formula"
		case optSourceLocFmriContrastName:
			label = "Contrast Name"
			if strings.TrimSpace(m.sourceLocFmriContrastName) == "" {
				value = "pain_vs_baseline"
			} else {
				value = m.sourceLocFmriContrastName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriContrastName {
				value = m.textBuffer + "█"
			}
			hint = "output name for contrast"
		case optSourceLocFmriRunsToInclude:
			label = "Runs"
			if strings.TrimSpace(m.sourceLocFmriRunsToInclude) == "" {
				value = "(auto-detect)"
			} else {
				value = m.sourceLocFmriRunsToInclude
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriRunsToInclude {
				value = m.textBuffer + "█"
			}
			hint = "comma-separated run numbers"
		case optSourceLocFmriAutoDetectRuns:
			label = "Auto-detect Runs"
			if m.sourceLocFmriAutoDetectRuns {
				value = "on"
			} else {
				value = "off"
			}
			hint = "scan BIDS for BOLD runs"
		case optSourceLocFmriHrfModel:
			label = "HRF Model"
			hrfModels := []string{"SPM", "FLOBS", "FIR"}
			value = hrfModels[m.sourceLocFmriHrfModel]
			hint = "hemodynamic response function"
		case optSourceLocFmriDriftModel:
			label = "Drift Model"
			driftModels := []string{"none", "cosine", "polynomial"}
			value = driftModels[m.sourceLocFmriDriftModel]
			hint = "low-frequency drift removal"
		case optSourceLocFmriHighPassHz:
			label = "High-pass (Hz)"
			value = fmt.Sprintf("%.4f", m.sourceLocFmriHighPassHz)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriHighPassHz) {
				value = m.numberBuffer + "█"
			}
			hint = "e.g., 0.008 = 128s period"
		case optSourceLocFmriLowPassHz:
			label = "Low-pass (Hz)"
			value = fmt.Sprintf("%.2f", m.sourceLocFmriLowPassHz)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriLowPassHz) {
				value = m.numberBuffer + "█"
			}
			hint = "temporal smoothing cutoff"
		case optSourceLocFmriClusterCorrection:
			label = "Cluster Correction"
			if m.sourceLocFmriClusterCorrection {
				value = "on"
			} else {
				value = "off"
			}
			hint = "cluster-level FWE correction"
		case optSourceLocFmriClusterPThreshold:
			label = "Cluster p-threshold"
			value = fmt.Sprintf("%.4f", m.sourceLocFmriClusterPThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriClusterPThreshold) {
				value = m.numberBuffer + "█"
			}
			hint = "cluster-forming threshold"
		case optSourceLocFmriOutputType:
			label = "Output Type"
			outputTypes := []string{"z-score", "t-stat", "cope", "beta"}
			value = outputTypes[m.sourceLocFmriOutputType]
			hint = "statistical map type"
		case optSourceLocFmriResampleToFS:
			label = "Resample to FS"
			if m.sourceLocFmriResampleToFS {
				value = "on"
			} else {
				value = "off"
			}
			hint = "auto-resample to FreeSurfer space"
		case optSourceLocFmriWindowAName:
			label = "Window A Name"
			if m.sourceLocFmriWindowAName == "" {
				value = "(not set)"
			} else {
				value = m.sourceLocFmriWindowAName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriWindowAName {
				value = m.textBuffer + "█"
			}
			hint = "e.g., plateau"
		case optSourceLocFmriWindowATmin:
			label = "Window A Tmin"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowATmin)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowATmin) {
				value = m.numberBuffer + "█"
			}
			hint = "start time in seconds"
		case optSourceLocFmriWindowATmax:
			label = "Window A Tmax"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowATmax)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowATmax) {
				value = m.numberBuffer + "█"
			}
			hint = "end time in seconds"
		case optSourceLocFmriWindowBName:
			label = "Window B Name"
			if m.sourceLocFmriWindowBName == "" {
				value = "(not set)"
			} else {
				value = m.sourceLocFmriWindowBName
			}
			if m.editingText && m.editingTextField == textFieldSourceLocFmriWindowBName {
				value = m.textBuffer + "█"
			}
			hint = "e.g., baseline (optional)"
		case optSourceLocFmriWindowBTmin:
			label = "Window B Tmin"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowBTmin)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowBTmin) {
				value = m.numberBuffer + "█"
			}
			hint = "start time in seconds"
		case optSourceLocFmriWindowBTmax:
			label = "Window B Tmax"
			value = fmt.Sprintf("%.2f s", m.sourceLocFmriWindowBTmax)
			if m.editingNumber && m.isCurrentlyEditing(optSourceLocFmriWindowBTmax) {
				value = m.numberBuffer + "█"
			}
			hint = "end time in seconds"

		// ITPC options
		case optFeatGroupITPC:
			label = "▸ ITPC"
			hint = "Space to toggle"
			if m.featGroupITPCExpanded {
				label = "▾ ITPC"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optItpcMethod:
			label = "ITPC Method"
			itpcMethods := []string{"global", "fold_global", "loo", "condition"}
			value = itpcMethods[m.itpcMethod]
			hint = "global/fold_global/loo/condition"
		case optItpcAllowUnsafeLoo:
			label = "Allow unsafe LOO"
			value = m.boolToOnOff(m.itpcAllowUnsafeLoo)
			hint = "unsafe unless computed within CV folds"
		case optItpcBaselineCorrection:
			label = "Baseline correction"
			modes := []string{"none", "subtract"}
			value = modes[m.itpcBaselineCorrection]
			hint = "ITPC baseline correction"
		case optItpcConditionColumn:
			label = "Condition Column"
			val := m.itpcConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldItpcConditionColumn {
				val = m.textBuffer + "█"
			}
			value = val
			expandIndicatorHint := ""
			if len(m.availableColumns) > 0 {
				expandIndicatorHint = fmt.Sprintf(" · %d columns available", len(m.availableColumns))
			}
			hint = "Space to select" + expandIndicatorHint
			if m.expandedOption == expandedItpcConditionColumn {
				expandIndicator = " [-]"
			} else {
				expandIndicator = " [+]"
			}
		case optItpcConditionValues:
			label = "Condition Values"
			if m.itpcConditionColumn == "" {
				value = "(select column first)"
				hint = "requires column selection"
			} else {
				val := m.itpcConditionValues
				if val == "" {
					val = "(select values)"
				}
				if m.editingText && m.editingTextField == textFieldItpcConditionValues {
					val = m.textBuffer + "█"
				}
				value = val
				expandIndicatorHint := ""
				if vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn); len(vals) > 0 {
					expandIndicatorHint = fmt.Sprintf(" · %d values in %s", len(vals), m.itpcConditionColumn)
				}
				hint = "Space to select (others excluded)" + expandIndicatorHint
				if m.expandedOption == expandedItpcConditionValues {
					expandIndicator = " [-]"
				} else {
					expandIndicator = " [+]"
				}
			}
		case optItpcMinTrialsPerCondition:
			label = "Min Trials/Condition"
			value = fmt.Sprintf("%d", m.itpcMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optItpcMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum trials per condition"
		case optItpcNJobs:
			label = "Parallel Jobs"
			value = fmt.Sprintf("%d", m.itpcNJobs)
			if m.editingNumber && m.isCurrentlyEditing(optItpcNJobs) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all CPUs"

		// PAC
		case optPACPhaseRange:
			label = "Phase range"
			value = pacPhaseVal
			hint = "frequencies for phase"
		case optPACAmpRange:
			label = "Amp range"
			value = pacAmpVal
			hint = "frequencies for amplitude"
		case optPACMethod:
			label = "PAC Method"
			value = pacMethodVal
			hint = "algorithm type"
		case optPACMinEpochs:
			label = "Min Epochs"
			value = pacMinEpochsVal
			hint = "minimum required"
		case optPACPairs:
			label = "Band pairs"
			value = pacPairsVal
			hint = "e.g. theta:gamma,alpha:gamma"
		case optPACSource:
			label = "Source"
			sources := []string{"precomputed", "tfr"}
			value = sources[m.pacSource]
			hint = "Hilbert vs wavelet"
		case optPACNormalize:
			label = "Normalize"
			value = m.boolToOnOff(m.pacNormalize)
			hint = "normalize PAC values"
		case optPACNSurrogates:
			label = "N Surrogates"
			value = fmt.Sprintf("%d", m.pacNSurrogates)
			if m.editingNumber && m.isCurrentlyEditing(optPACNSurrogates) {
				value = m.numberBuffer + "█"
			}
			hint = "0=none, >0 for z-scores"
		case optPACAllowHarmonicOverlap:
			label = "Allow Harmonics"
			value = m.boolToOnOff(m.pacAllowHarmonicOvrlap)
			hint = "allow harmonic overlap"
		case optPACMaxHarmonic:
			label = "Max Harmonic"
			value = fmt.Sprintf("%d", m.pacMaxHarmonic)
			if m.editingNumber && m.isCurrentlyEditing(optPACMaxHarmonic) {
				value = m.numberBuffer + "█"
			}
			hint = "upper harmonic to check"
		case optPACHarmonicToleranceHz:
			label = "Harmonic Tol Hz"
			value = fmt.Sprintf("%.1f", m.pacHarmonicToleranceHz)
			if m.editingNumber && m.isCurrentlyEditing(optPACHarmonicToleranceHz) {
				value = m.numberBuffer + "█"
			}
			hint = "tolerance for overlap check"
		case optPACRandomSeed:
			label = "Random Seed"
			value = fmt.Sprintf("%d", m.pacRandomSeed)
			if m.editingNumber && m.isCurrentlyEditing(optPACRandomSeed) {
				value = m.numberBuffer + "█"
			}
			hint = "seed for surrogate testing"
		case optPACComputeWaveformQC:
			label = "Waveform QC"
			value = m.boolToOnOff(m.pacComputeWaveformQC)
			hint = "compute waveform quality"
		case optPACWaveformOffsetMs:
			label = "Waveform Offset"
			value = fmt.Sprintf("%.1f ms", m.pacWaveformOffsetMs)
			if m.editingNumber && m.isCurrentlyEditing(optPACWaveformOffsetMs) {
				value = m.numberBuffer + "█"
			}
			hint = "offset in milliseconds"

		// Aperiodic
		case optAperiodicModel:
			label = "Model"
			models := []string{"fixed", "knee"}
			value = models[m.aperiodicModel]
			hint = "aperiodic model type"
		case optAperiodicPsdMethod:
			label = "PSD method"
			methods := []string{"multitaper", "welch"}
			value = methods[m.aperiodicPsdMethod]
			hint = "PSD estimation method"
		case optAperiodicFmin:
			aperiodicFminVal := fmt.Sprintf("%.1f", m.aperiodicFmin)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicFmin) {
				aperiodicFminVal = m.numberBuffer + "█"
			}
			label = "Fit range (min)"
			value = aperiodicFminVal
			hint = "minimum frequency (Hz)"
		case optAperiodicFmax:
			aperiodicFmaxVal := fmt.Sprintf("%.1f", m.aperiodicFmax)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicFmax) {
				aperiodicFmaxVal = m.numberBuffer + "█"
			}
			label = "Fit range (max)"
			value = aperiodicFmaxVal
			hint = "maximum frequency (Hz)"
		case optAperiodicMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.1f", m.aperiodicMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum duration for stable fits"
		case optAperiodicExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.aperiodicExcludeLineNoise)
			hint = "exclude line-noise bins before fitting"
		case optAperiodicPeakZ:
			label = "Peak Z-thresh"
			value = aperiodicPeakZVal
			hint = "peak rejection"
		case optAperiodicMinR2:
			label = "Min R2"
			value = aperiodicR2Val
			hint = "minimum fit quality"
		case optAperiodicMinPoints:
			label = "Min Points"
			value = aperiodicPointsVal
			hint = "minimum bins required"
		case optAperiodicPsdBandwidth:
			aperiodicBandwidthVal := fmt.Sprintf("%.1f", m.aperiodicPsdBandwidth)
			if m.aperiodicPsdBandwidth == 0 {
				aperiodicBandwidthVal = "(default)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicPsdBandwidth) {
				aperiodicBandwidthVal = m.numberBuffer + "█"
			}
			label = "PSD Bandwidth"
			value = aperiodicBandwidthVal
			hint = "Hz (0=default)"
		case optAperiodicMaxRms:
			aperiodicMaxRmsVal := fmt.Sprintf("%.3f", m.aperiodicMaxRms)
			if m.aperiodicMaxRms == 0 {
				aperiodicMaxRmsVal = "(no limit)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicMaxRms) {
				aperiodicMaxRmsVal = m.numberBuffer + "█"
			}
			label = "Max RMS"
			value = aperiodicMaxRmsVal
			hint = "fit quality limit (0=no limit)"
		case optAperiodicLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.aperiodicLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optAperiodicLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.aperiodicLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optAperiodicLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.aperiodicLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optAperiodicLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"

		// Complexity
		case optPEOrder:
			label = "PE Order"
			value = peOrderVal
			hint = "symbol length (3-7)"
		case optPEDelay:
			label = "PE Delay"
			value = peDelayVal
			hint = "sample lag"
		case optComplexitySignalBasis:
			label = "Signal basis"
			bases := []string{"filtered", "envelope"}
			if m.complexitySignalBasis >= 0 && m.complexitySignalBasis < len(bases) {
				value = bases[m.complexitySignalBasis]
			} else {
				value = "filtered"
			}
			hint = "filtered or envelope"
		case optComplexityMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.complexityMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optComplexityMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "skip short segments"
		case optComplexityMinSamples:
			label = "Min samples"
			value = fmt.Sprintf("%d", m.complexityMinSamples)
			if m.editingNumber && m.isCurrentlyEditing(optComplexityMinSamples) {
				value = m.numberBuffer + "█"
			}
			hint = "skip low-sample segments"
		case optComplexityZscore:
			label = "Z-score"
			if m.complexityZscore {
				value = "on"
			} else {
				value = "off"
			}
			hint = "normalize per-channel"

		// Bursts
		case optBurstThresholdMethod:
			methods := []string{"percentile", "zscore", "mad"}
			methodVal := "percentile"
			if m.burstThresholdMethod >= 0 && m.burstThresholdMethod < len(methods) {
				methodVal = methods[m.burstThresholdMethod]
			}
			label = "Threshold Method"
			value = methodVal
			hint = "percentile / zscore / mad"
		case optBurstThresholdPercentile:
			burstPercentileVal := fmt.Sprintf("%.1f", m.burstThresholdPercentile)
			if m.editingNumber && m.isCurrentlyEditing(optBurstThresholdPercentile) {
				burstPercentileVal = m.numberBuffer + "█"
			}
			label = "Threshold Percentile"
			value = burstPercentileVal
			hint = "percentile threshold"
		case optBurstThreshold:
			label = "Threshold Z"
			value = burstThreshVal
			hint = "amplitude trigger"
		case optBurstThresholdReference:
			label = "Threshold ref"
			refs := []string{"trial", "subject", "condition"}
			value = refs[m.burstThresholdReference]
			hint = "trial/subject/condition"
		case optBurstMinTrialsPerCondition:
			label = "Min trials/cond"
			value = fmt.Sprintf("%d", m.burstMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optBurstMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "used for condition reference"
		case optBurstMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.burstMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optBurstMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "skip short segments"
		case optBurstSkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.burstSkipInvalidSegments)
			hint = "skip invalid segments"
		case optBurstMinDuration:
			label = "Min Duration"
			value = burstMinDurVal
			hint = "minimum length"
		case optBurstBands:
			label = "Burst bands"
			value = burstBandsVal
			hint = "e.g. beta,gamma"

		// Power
		case optPowerRequireBaseline:
			label = "Require baseline"
			value = powerRequireBaselineVal
			hint = "allow raw log power if OFF"
		case optPowerSubtractEvoked:
			label = "Subtract evoked"
			value = m.boolToOnOff(m.powerSubtractEvoked)
			hint = "induced power; CV-safe only with train_mask"
		case optPowerMinTrialsPerCondition:
			label = "Min trials/cond"
			value = fmt.Sprintf("%d", m.powerMinTrialsPerCondition)
			if m.editingNumber && m.isCurrentlyEditing(optPowerMinTrialsPerCondition) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum trials per condition"
		case optPowerExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.powerExcludeLineNoise)
			hint = "exclude line-noise bins"
		case optPowerLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.powerLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optPowerLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.powerLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optPowerLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.powerLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optPowerLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"
		case optPowerEmitDb:
			label = "Emit dB"
			value = m.boolToOnOff(m.powerEmitDb)
			hint = "emit 10*log10 ratios"
		case optPowerBaselineMode:
			label = "Baseline mode"
			value = powerBaselineVal
			hint = "normalization type"

		// Spectral
		case optSpectralEdge:
			label = "Spectral Edge"
			value = spectralEdgeVal
			hint = "percentile for SEF"

		// ERP
		case optERPBaseline:
			label = "ERP baseline"
			value = erpBaselineVal
			hint = "subtract baseline mean"
		case optERPAllowNoBaseline:
			label = "Allow no baseline"
			value = erpAllowNoBaselineVal
			hint = "only used if baseline ON"
		case optERPComponents:
			label = "Components"
			value = erpComponentsVal
			hint = "e.g. n1=0.10-0.20,n2=0.20-0.35"
		case optERPSmoothMs:
			erpSmoothVal := fmt.Sprintf("%.1f ms", m.erpSmoothMs)
			if m.erpSmoothMs == 0 {
				erpSmoothVal = "(no smoothing)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optERPSmoothMs) {
				erpSmoothVal = m.numberBuffer + "█"
			}
			label = "Smooth Window"
			value = erpSmoothVal
			hint = "ms (0=no smoothing)"
		case optERPPeakProminenceUv:
			erpProminenceVal := fmt.Sprintf("%.1f µV", m.erpPeakProminenceUv)
			if m.erpPeakProminenceUv == 0 {
				erpProminenceVal = "(default)"
			}
			if m.editingNumber && m.isCurrentlyEditing(optERPPeakProminenceUv) {
				erpProminenceVal = m.numberBuffer + "█"
			}
			label = "Peak Prominence"
			value = erpProminenceVal
			hint = "µV (0=use default)"
		case optERPLowpassHz:
			erpLowpassVal := fmt.Sprintf("%.1f Hz", m.erpLowpassHz)
			if m.editingNumber && m.isCurrentlyEditing(optERPLowpassHz) {
				erpLowpassVal = m.numberBuffer + "█"
			}
			label = "Low-Pass Filter"
			value = erpLowpassVal
			hint = "Hz before peak detection"

		// Ratios / asymmetry
		case optSpectralRatioPairs:
			label = "Ratio pairs"
			value = spectralRatioPairsVal
			hint = ""
		case optAperiodicSubtractEvoked:
			label = "Induced spectra"
			value = m.boolToOnOff(m.aperiodicSubtractEvoked)
			hint = "subtract evoked for pain"
		case optAsymmetryChannelPairs:
			label = "Channel pairs"
			value = asymPairsVal
			hint = "e.g. F3:F4,C3:C4"
		case optAsymmetryMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.asymmetryMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optAsymmetryMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optAsymmetryMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.asymmetryMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optAsymmetryMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optAsymmetrySkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.asymmetrySkipInvalidSegments)
			hint = "skip invalid segments"
		case optAsymmetryEmitActivationConvention:
			label = "Emit activation"
			value = m.boolToOnOff(m.asymmetryEmitActivationConvention)
			hint = "(R-L)/(R+L) for activation bands"
		case optAsymmetryActivationBands:
			label = "Activation bands"
			val := m.asymmetryActivationBandsSpec
			if strings.TrimSpace(val) == "" {
				val = "(default: alpha)"
			}
			if m.editingText && m.editingTextField == textFieldAsymmetryActivationBands {
				val = m.textBuffer + "█"
			}
			value = val
			hint = "e.g. alpha,beta"

		// Ratios advanced options
		case optRatiosMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.ratiosMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optRatiosMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optRatiosMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.ratiosMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optRatiosMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optRatiosSkipInvalidSegments:
			label = "Skip invalid"
			value = m.boolToOnOff(m.ratiosSkipInvalidSegments)
			hint = "skip invalid segments"

		// Spectral advanced options
		case optSpectralIncludeLogRatios:
			label = "Log ratios"
			value = m.boolToOnOff(m.spectralIncludeLogRatios)
			hint = "include log ratios"
		case optSpectralPsdMethod:
			label = "PSD method"
			methods := []string{"multitaper", "welch"}
			value = methods[m.spectralPsdMethod]
			hint = "multitaper or welch"
		case optSpectralPsdAdaptive:
			label = "PSD adaptive"
			value = m.boolToOnOff(m.spectralPsdAdaptive)
			hint = "adaptive PSD settings"
		case optSpectralMultitaperAdaptive:
			label = "Multitaper adaptive"
			value = m.boolToOnOff(m.spectralMultitaperAdaptive)
			hint = "adaptive multitaper (if supported)"
		case optSpectralFmin:
			label = "Freq min"
			value = fmt.Sprintf("%.1f Hz", m.spectralFmin)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum frequency"
		case optSpectralFmax:
			label = "Freq max"
			value = fmt.Sprintf("%.1f Hz", m.spectralFmax)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralFmax) {
				value = m.numberBuffer + "█"
			}
			hint = "maximum frequency"
		case optSpectralMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.spectralMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optSpectralMinCyclesAtFmin:
			label = "Min cycles"
			value = fmt.Sprintf("%.1f", m.spectralMinCyclesAtFmin)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralMinCyclesAtFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "cycles at lowest freq"
		case optSpectralExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.spectralExcludeLineNoise)
			hint = "exclude line-noise bins"
		case optSpectralLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f", m.spectralLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz (50 or 60)"
		case optSpectralLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f", m.spectralLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "Hz bandwidth to exclude"
		case optSpectralLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.spectralLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optSpectralLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"

		// Quality group header
		case optFeatGroupQuality:
			label = "▸ Quality"
			hint = "Space to toggle"
			if m.featGroupQualityExpanded {
				label = "▾ Quality"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optQualityPsdMethod:
			label = "PSD method"
			methods := []string{"welch", "multitaper"}
			value = methods[m.qualityPsdMethod]
			hint = "welch or multitaper"
		case optQualityFmin:
			label = "Freq min"
			value = fmt.Sprintf("%.1f Hz", m.qualityFmin)
			if m.editingNumber && m.isCurrentlyEditing(optQualityFmin) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum frequency"
		case optQualityFmax:
			label = "Freq max"
			value = fmt.Sprintf("%.1f Hz", m.qualityFmax)
			if m.editingNumber && m.isCurrentlyEditing(optQualityFmax) {
				value = m.numberBuffer + "█"
			}
			hint = "maximum frequency"
		case optQualityNFft:
			label = "N FFT"
			value = fmt.Sprintf("%d", m.qualityNfft)
			if m.editingNumber && m.isCurrentlyEditing(optQualityNFft) {
				value = m.numberBuffer + "█"
			}
			hint = "FFT size"
		case optQualityExcludeLineNoise:
			label = "Exclude line noise"
			value = m.boolToOnOff(m.qualityExcludeLineNoise)
			hint = "remove line noise bins"
		case optQualityLineNoiseFreq:
			label = "Line noise freq"
			value = fmt.Sprintf("%.0f Hz", m.qualityLineNoiseFreq)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseFreq) {
				value = m.numberBuffer + "█"
			}
			hint = "50 or 60 Hz"
		case optQualityLineNoiseWidthHz:
			label = "Line noise width"
			value = fmt.Sprintf("%.1f Hz", m.qualityLineNoiseWidthHz)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseWidthHz) {
				value = m.numberBuffer + "█"
			}
			hint = "exclusion bandwidth"
		case optQualityLineNoiseHarmonics:
			label = "Line noise harmonics"
			value = fmt.Sprintf("%d", m.qualityLineNoiseHarmonics)
			if m.editingNumber && m.isCurrentlyEditing(optQualityLineNoiseHarmonics) {
				value = m.numberBuffer + "█"
			}
			hint = "number of harmonics"
		case optQualitySnrSignalBandMin:
			label = "SNR signal min"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrSignalBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrSignalBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "signal band lower bound"
		case optQualitySnrSignalBandMax:
			label = "SNR signal max"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrSignalBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrSignalBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "signal band upper bound"
		case optQualitySnrNoiseBandMin:
			label = "SNR noise min"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrNoiseBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrNoiseBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "noise band lower bound"
		case optQualitySnrNoiseBandMax:
			label = "SNR noise max"
			value = fmt.Sprintf("%.1f Hz", m.qualitySnrNoiseBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualitySnrNoiseBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "noise band upper bound"
		case optQualityMuscleBandMin:
			label = "Muscle band min"
			value = fmt.Sprintf("%.1f Hz", m.qualityMuscleBandMin)
			if m.editingNumber && m.isCurrentlyEditing(optQualityMuscleBandMin) {
				value = m.numberBuffer + "█"
			}
			hint = "muscle artifact lower"
		case optQualityMuscleBandMax:
			label = "Muscle band max"
			value = fmt.Sprintf("%.1f Hz", m.qualityMuscleBandMax)
			if m.editingNumber && m.isCurrentlyEditing(optQualityMuscleBandMax) {
				value = m.numberBuffer + "█"
			}
			hint = "muscle artifact upper"

		// ERDS group header
		case optFeatGroupERDS:
			label = "▸ ERDS"
			hint = "Space to toggle"
			if m.featGroupERDSExpanded {
				label = "▾ ERDS"
			}
			value, expandIndicator = "", ""
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
		case optERDSUseLogRatio:
			label = "Use log ratio"
			value = m.boolToOnOff(m.erdsUseLogRatio)
			hint = "dB (on) vs percent (off)"
		case optERDSMinBaselinePower:
			label = "Min baseline power"
			value = fmt.Sprintf("%.2e", m.erdsMinBaselinePower)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinBaselinePower) {
				value = m.numberBuffer + "█"
			}
			hint = "clamp baseline power"
		case optERDSMinActivePower:
			label = "Min active power"
			value = fmt.Sprintf("%.2e", m.erdsMinActivePower)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinActivePower) {
				value = m.numberBuffer + "█"
			}
			hint = "clamp active power"
		case optERDSMinSegmentSec:
			label = "Min segment (s)"
			value = fmt.Sprintf("%.2f", m.erdsMinSegmentSec)
			if m.editingNumber && m.isCurrentlyEditing(optERDSMinSegmentSec) {
				value = m.numberBuffer + "█"
			}
			hint = "minimum segment duration"
		case optERDSBands:
			erdsBandsVal := m.erdsBandsSpec
			if strings.TrimSpace(erdsBandsVal) == "" {
				erdsBandsVal = "(default: alpha,beta)"
			}
			if m.editingText && m.editingTextField == textFieldERDSBands {
				erdsBandsVal = m.textBuffer + "█"
			}
			label = "Bands"
			value = erdsBandsVal
			hint = "e.g. alpha,beta"

		// Generic
		case optMinEpochs:
			label = "Min Epochs"
			value = minEpochsVal
			hint = "global minimum required"
		case optFeatAnalysisMode:
			label = "Analysis mode"
			modes := []string{"group_stats", "trial_ml_safe"}
			value = modes[m.featAnalysisMode]
			hint = "group_stats / trial_ml_safe"
		case optFeatComputeChangeScores:
			label = "Change scores"
			value = m.boolToOnOff(m.featComputeChangeScores)
			hint = "add within-subject change columns"
		case optFeatSaveTfrWithSidecar:
			label = "Save TFR sidecar"
			value = m.boolToOnOff(m.featSaveTfrWithSidecar)
			hint = "write TFR arrays for inspection"
		case optFeatNJobsBands:
			label = "n_jobs (bands)"
			value = fmt.Sprintf("%d", m.featNJobsBands)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsBands) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsConnectivity:
			label = "n_jobs (connectivity)"
			value = fmt.Sprintf("%d", m.featNJobsConnectivity)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsConnectivity) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsAperiodic:
			label = "n_jobs (aperiodic)"
			value = fmt.Sprintf("%d", m.featNJobsAperiodic)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsAperiodic) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"
		case optFeatNJobsComplexity:
			label = "n_jobs (complexity)"
			value = fmt.Sprintf("%d", m.featNJobsComplexity)
			if m.editingNumber && m.isCurrentlyEditing(optFeatNJobsComplexity) {
				value = m.numberBuffer + "█"
			}
			hint = "-1 = all cores"

		default:
			label = "Unknown"
			value = ""
			hint = ""
		}

		if lineIdx >= startLine && lineIdx < endLine {
			b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value+expandIndicator))
			b.WriteString("  " + hintStyle.Render(hint))
			b.WriteString("\n")
		}
		lineIdx++

		// Expanded items (connectivity measures)
		if opt == optConnectivity && m.expandedOption == expandedConnectivityMeasures {
			subIndent := "      " // 6 spaces for sub-items
			for j, measure := range connectivityMeasures {
				isSelected := m.connectivityMeasures[j]
				isSubFocused := j == m.subCursor

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + measure.Description)
				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(measure.Name) + desc + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (directed connectivity measures)
		if opt == optDirectedConnMeasures && m.expandedOption == expandedDirectedConnMeasures {
			subIndent := "      " // 6 spaces for sub-items
			for j, measure := range directedConnectivityMeasures {
				isSelected := m.directedConnMeasures[j]
				isSubFocused := j == m.subCursor

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + measure.Description)
				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(measure.Name) + desc + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition A column)
		if opt == optSourceLocFmriCondAColumn && m.expandedOption == expandedFmriCondAColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.fmriDiscoveredColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondAColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition A value)
		if opt == optSourceLocFmriCondAValue && m.expandedOption == expandedFmriCondAValue {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondAValue == v

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition B column)
		if opt == optSourceLocFmriCondBColumn && m.expandedOption == expandedFmriCondBColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.fmriDiscoveredColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondBColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (fMRI condition B value)
		if opt == optSourceLocFmriCondBValue && m.expandedOption == expandedFmriCondBValue {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.sourceLocFmriCondBValue == v

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (ITPC condition column)
		if opt == optItpcConditionColumn && m.expandedOption == expandedItpcConditionColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.availableColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.itpcConditionColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (ITPC condition values)
		if opt == optItpcConditionValues && m.expandedOption == expandedItpcConditionValues {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.isColumnValueSelected(v)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (Connectivity condition column)
		if opt == optConnConditionColumn && m.expandedOption == expandedConnConditionColumn {
			subIndent := "      " // 6 spaces for sub-items
			for j, col := range m.availableColumns {
				isSubFocused := j == m.subCursor
				isSelected := m.connConditionColumn == col

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(col) + "\n")
				}
				lineIdx++
			}
		}

		// Expanded items (Connectivity condition values)
		if opt == optConnConditionValues && m.expandedOption == expandedConnConditionValues {
			subIndent := "      " // 6 spaces for sub-items
			vals := m.GetDiscoveredColumnValues(m.connConditionColumn)
			for j, v := range vals {
				isSubFocused := j == m.subCursor
				isSelected := m.isColumnValueSelected(v)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
				if isSubFocused {
					nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
				}

				if lineIdx >= startLine && lineIdx < endLine {
					b.WriteString(subIndent + checkbox + nameStyle.Render(v) + "\n")
				}
				lineIdx++
			}
		}
	}

	// Show scroll indicator for items below
	if showScrollIndicators && lineIdx > endLine {
		remaining := lineIdx - endLine
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", remaining)) + "\n")
	}

	return b.String()
}

func (m Model) renderBehaviorAdvancedConfig() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" ADVANCED CONFIGURATION") + "\n\n")

	// Contextual help text (same as features pipeline)
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("behavior analysis")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Enter a value, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.expandedOption >= 0 {
		b.WriteString(infoStyle.Render("Space to select item · ↑↓ to navigate · Esc to close list") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to toggle/expand · ↑↓ to navigate · Enter to proceed") + "\n\n")
	}

	labelWidth := defaultLabelWidthWide
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	options := m.getBehaviorOptions()

	getOptionDisplay := func(opt optionType) (string, string, string) {
		numberDisplay := m.numberBuffer + "█"
		textDisplay := m.textBuffer + "█"

		switch opt {
		case optUseDefaults:
			return "Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"
		// Behavior section headers
		case optBehaviorGroupGeneral:
			label := "▸ General"
			if m.behaviorGroupGeneralExpanded {
				label = "▾ General"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupTrialTable:
			label := "▸ Trial Table"
			if m.behaviorGroupTrialTableExpanded {
				label = "▾ Trial Table"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupPainResidual:
			label := "▸ Pain Residual"
			if m.behaviorGroupPainResidualExpanded {
				label = "▾ Pain Residual"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCorrelations:
			label := "▸ Correlations"
			if m.behaviorGroupCorrelationsExpanded {
				label = "▾ Correlations"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupRegression:
			label := "▸ Regression"
			if m.behaviorGroupRegressionExpanded {
				label = "▾ Regression"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupModels:
			label := "▸ Models"
			if m.behaviorGroupModelsExpanded {
				label = "▾ Models"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupStability:
			label := "▸ Stability"
			if m.behaviorGroupStabilityExpanded {
				label = "▾ Stability"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupConsistency:
			label := "▸ Consistency"
			if m.behaviorGroupConsistencyExpanded {
				label = "▾ Consistency"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupInfluence:
			label := "▸ Influence"
			if m.behaviorGroupInfluenceExpanded {
				label = "▾ Influence"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupReport:
			label := "▸ Report"
			if m.behaviorGroupReportExpanded {
				label = "▾ Report"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCondition:
			label := "▸ Condition"
			if m.behaviorGroupConditionExpanded {
				label = "▾ Condition"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupTemporal:
			label := "▸ Temporal"
			if m.behaviorGroupTemporalExpanded {
				label = "▾ Temporal"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupCluster:
			label := "▸ Cluster"
			if m.behaviorGroupClusterExpanded {
				label = "▾ Cluster"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupMediation:
			label := "▸ Mediation"
			if m.behaviorGroupMediationExpanded {
				label = "▾ Mediation"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupModeration:
			label := "▸ Moderation"
			if m.behaviorGroupModerationExpanded {
				label = "▾ Moderation"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupMixedEffects:
			label := "▸ Mixed Effects"
			if m.behaviorGroupMixedEffectsExpanded {
				label = "▾ Mixed Effects"
			}
			return label, "", "Space to toggle"
		case optCorrMethod:
			return "Correlation Method", m.correlationMethod, "spearman / pearson"
		case optRobustCorrelation:
			methods := []string{"none", "percentage_bend", "winsorized", "shepherd"}
			v := "none"
			if m.robustCorrelation >= 0 && m.robustCorrelation < len(methods) {
				v = methods[m.robustCorrelation]
			}
			return "Robust Correlation", v, "robust alternative for outliers"
		case optBootstrap:
			val := fmt.Sprintf("%d", m.bootstrapSamples)
			if m.editingNumber && m.isCurrentlyEditing(optBootstrap) {
				val = numberDisplay
			}
			return "Bootstrap Samples", val, "0=disabled"
		case optNPerm:
			val := fmt.Sprintf("%d", m.nPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optNPerm) {
				val = numberDisplay
			}
			return "Permutations", val, "cluster/global permutations"
		case optRNGSeed:
			val := m.rngSeedDisplay()
			if m.editingNumber && m.isCurrentlyEditing(optRNGSeed) {
				val = numberDisplay
			}
			return "RNG Seed", val, "0=project default"
		case optBehaviorNJobs:
			val := fmt.Sprintf("%d", m.behaviorNJobs)
			if m.editingNumber && m.isCurrentlyEditing(optBehaviorNJobs) {
				val = numberDisplay
			}
			return "N Jobs", val, "-1=all cores"
		case optControlTemp:
			return "Control Temperature", m.boolToOnOff(m.controlTemperature), "partial-correlation covariate"
		case optControlOrder:
			return "Control Trial Order", m.boolToOnOff(m.controlTrialOrder), "partial-correlation covariate"
		case optRunAdjustmentEnabled:
			return "Run Adjustment", m.boolToOnOff(m.runAdjustmentEnabled), "run-aware controls/aggregation"
		case optRunAdjustmentColumn:
			val := m.runAdjustmentColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: run_id)"
			}
			if m.editingText && m.editingTextField == textFieldRunAdjustmentColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if !m.runAdjustmentEnabled {
				hint = "run identifier column (enable Run Adjustment)"
			} else if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Run Column", val, hint
		case optRunAdjustmentIncludeInCorrelations:
			if !m.runAdjustmentEnabled {
				return "Run Dummies in Corr", "N/A", "enable Run Adjustment"
			}
			return "Run Dummies in Corr", m.boolToOnOff(m.runAdjustmentIncludeInCorrelations), "add to partial covariates"
		case optRunAdjustmentMaxDummies:
			if !m.runAdjustmentEnabled {
				return "Max Run Dummies", "N/A", "enable Run Adjustment"
			}
			val := fmt.Sprintf("%d", m.runAdjustmentMaxDummies)
			if m.editingNumber && m.isCurrentlyEditing(optRunAdjustmentMaxDummies) {
				val = numberDisplay
			}
			return "Max Run Dummies", val, "skip if > N levels"
		case optFDRAlpha:
			val := fmt.Sprintf("%.3f", m.fdrAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optFDRAlpha) {
				val = numberDisplay
			}
			return "FDR Alpha", val, "multiple comparison threshold"
		case optComputeChangeScores:
			return "Change Scores", m.boolToOnOff(m.behaviorComputeChangeScores), "Δ rating / Δ temperature"
		case optComputeLosoStability:
			return "LOSO Stability", m.boolToOnOff(m.behaviorComputeLosoStability), "leave-one-out stability"
		case optComputeBayesFactors:
			return "Bayes Factors", m.boolToOnOff(m.behaviorComputeBayesFactors), "optional BF reporting"
		case optFeatureQCEnabled:
			return "Feature QC", m.boolToOnOff(m.featureQCEnabled), "pre-filter features (optional gating)"
		case optFeatureQCMaxMissingPct:
			if !m.featureQCEnabled {
				return "QC Max Missing %", "N/A", "enable Feature QC"
			}
			val := fmt.Sprintf("%.2f", m.featureQCMaxMissingPct)
			if m.editingNumber && m.isCurrentlyEditing(optFeatureQCMaxMissingPct) {
				val = numberDisplay
			}
			return "QC Max Missing %", val, "fraction missing allowed"
		case optFeatureQCMinVariance:
			if !m.featureQCEnabled {
				return "QC Min Variance", "N/A", "enable Feature QC"
			}
			val := fmt.Sprintf("%.2e", m.featureQCMinVariance)
			if m.editingNumber && m.isCurrentlyEditing(optFeatureQCMinVariance) {
				val = numberDisplay
			}
			return "QC Min Variance", val, "drop near-constant features"
		case optFeatureQCCheckWithinRunVariance:
			if !m.featureQCEnabled {
				return "QC Within-Run Var", "N/A", "enable Feature QC"
			}
			return "QC Within-Run Var", m.boolToOnOff(m.featureQCCheckWithinRunVariance), "check per-run variance"

		// Trial table
		case optTrialTableFormat:
			v := "parquet"
			if m.trialTableFormat == 1 {
				v = "tsv"
			}
			return "Trial Table Format", v, "parquet recommended"
		case optTrialTableAddLagFeatures:
			return "Lag/Delta Columns", m.boolToOnOff(m.trialTableAddLagFeatures), "prev_* and delta_*"
		case optTrialOrderMaxMissingFraction:
			if !m.controlTrialOrder {
				return "Trial Order Max Missing", "N/A", "enable Control Trial Order"
			}
			val := fmt.Sprintf("%.2f", m.trialOrderMaxMissingFraction)
			if m.editingNumber && m.isCurrentlyEditing(optTrialOrderMaxMissingFraction) {
				val = numberDisplay
			}
			return "Trial Order Max Missing", val, "disable control if missing > threshold"

		// Pain residual
		case optPainResidualEnabled:
			return "Pain Residual", m.boolToOnOff(m.painResidualEnabled), "rating - f(temp)"
		case optPainResidualMethod:
			v := "spline"
			if m.painResidualMethod == 1 {
				v = "poly"
			}
			return "Residual Method", v, "spline preferred"
		case optPainResidualPolyDegree:
			val := fmt.Sprintf("%d", m.painResidualPolyDegree)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualPolyDegree) {
				val = numberDisplay
			}
			return "Poly Degree", val, "poly fallback degree"
		case optPainResidualSplineDfCandidates:
			val := m.painResidualSplineDfCandidates
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualSplineDfCandidates {
				val = textDisplay
			}
			return "Spline DF Candidates", val, "comma-separated (e.g., 3,4,5)"
		case optPainResidualModelCompare:
			return "Temp Model Compare", m.boolToOnOff(m.painResidualModelCompareEnabled), "non-gating diagnostics"
		case optPainResidualModelComparePolyDegrees:
			val := m.painResidualModelComparePolyDegrees
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualModelComparePolyDegrees {
				val = textDisplay
			}
			return "Model Compare Poly Deg", val, "comma-separated (e.g., 2,3)"
		case optPainResidualBreakpoint:
			return "Breakpoint Test", m.boolToOnOff(m.painResidualBreakpointEnabled), "single-hinge model"
		case optPainResidualBreakpointCandidates:
			val := fmt.Sprintf("%d", m.painResidualBreakpointCandidates)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointCandidates) {
				val = numberDisplay
			}
			return "Breakpoint Candidates", val, "grid size"
		case optPainResidualBreakpointQlow:
			val := fmt.Sprintf("%.2f", m.painResidualBreakpointQlow)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointQlow) {
				val = numberDisplay
			}
			return "Breakpoint Q Low", val, "quantile bound"
		case optPainResidualBreakpointQhigh:
			val := fmt.Sprintf("%.2f", m.painResidualBreakpointQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualBreakpointQhigh) {
				val = numberDisplay
			}
			return "Breakpoint Q High", val, "quantile bound"
		case optPainResidualCrossfitEnabled:
			if !m.painResidualEnabled {
				return "Residual Crossfit", "N/A", "enable Pain Residual"
			}
			return "Residual Crossfit", m.boolToOnOff(m.painResidualCrossfitEnabled), "out-of-run temperature→rating"
		case optPainResidualCrossfitGroupColumn:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Group Col", "N/A", "enable Residual Crossfit"
			}
			val := m.painResidualCrossfitGroupColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: run column)"
			}
			if m.editingText && m.editingTextField == textFieldPainResidualCrossfitGroupColumn {
				val = textDisplay
			}
			return "Crossfit Group Col", val, "GroupKFold column (blank=run column)"
		case optPainResidualCrossfitNSplits:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Splits", "N/A", "enable Residual Crossfit"
			}
			val := fmt.Sprintf("%d", m.painResidualCrossfitNSplits)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualCrossfitNSplits) {
				val = numberDisplay
			}
			return "Crossfit Splits", val, "n_splits (>=2)"
		case optPainResidualCrossfitMethod:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Method", "N/A", "enable Residual Crossfit"
			}
			v := "spline"
			if m.painResidualCrossfitMethod == 1 {
				v = "poly"
			}
			return "Crossfit Method", v, "spline | poly"
		case optPainResidualCrossfitSplineKnots:
			if !m.painResidualEnabled || !m.painResidualCrossfitEnabled {
				return "Crossfit Knots", "N/A", "enable Residual Crossfit"
			}
			if m.painResidualCrossfitMethod == 1 {
				return "Crossfit Knots", "N/A", "poly method"
			}
			val := fmt.Sprintf("%d", m.painResidualCrossfitSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optPainResidualCrossfitSplineKnots) {
				val = numberDisplay
			}
			return "Crossfit Knots", val, "spline knots (>=3)"

		// Regression
		case optRegressionOutcome:
			v := "rating"
			switch m.regressionOutcome {
			case 1:
				v = "pain_residual"
			case 2:
				v = "temperature"
			}
			return "Outcome", v, "dependent variable"
		case optRegressionIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.regressionIncludeTemperature), "add temperature covariate"
		case optRegressionTempControl:
			v := "linear"
			switch m.regressionTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optRegressionTempSplineKnots:
			val := fmt.Sprintf("%d", m.regressionTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optRegressionTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.regressionTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optRegressionTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.regressionTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optRegressionIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.regressionIncludeTrialOrder), "add trial_index covariate"
		case optRegressionIncludePrev:
			return "Prev/Delta Terms", m.boolToOnOff(m.regressionIncludePrev), "use prev_/delta_"
		case optRegressionIncludeRunBlock:
			return "Run/Block Dummies", m.boolToOnOff(m.regressionIncludeRunBlock), "categorical controls"
		case optRegressionIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.regressionIncludeInteraction), "moderation term"
		case optRegressionStandardize:
			return "Standardize", m.boolToOnOff(m.regressionStandardize), "z-score predictors"
		case optRegressionPermutations:
			val := fmt.Sprintf("%d", m.regressionPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionPermutations) {
				val = numberDisplay
			}
			return "Permutations", val, "Freedman–Lane (0=off)"
		case optRegressionMaxFeatures:
			val := fmt.Sprintf("%d", m.regressionMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optRegressionMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"

		// Models
		case optModelsIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.modelsIncludeTemperature), "add temperature covariate"
		case optModelsTempControl:
			v := "linear"
			switch m.modelsTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optModelsTempSplineKnots:
			val := fmt.Sprintf("%d", m.modelsTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optModelsTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.modelsTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optModelsTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.modelsTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optModelsTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optModelsIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.modelsIncludeTrialOrder), "add trial_index covariate"
		case optModelsIncludePrev:
			return "Prev/Delta Terms", m.boolToOnOff(m.modelsIncludePrev), "use prev_/delta_"
		case optModelsIncludeRunBlock:
			return "Run/Block Dummies", m.boolToOnOff(m.modelsIncludeRunBlock), "categorical controls"
		case optModelsIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.modelsIncludeInteraction), "moderation term"
		case optModelsStandardize:
			return "Standardize", m.boolToOnOff(m.modelsStandardize), "z-score predictors"
		case optModelsMaxFeatures:
			val := fmt.Sprintf("%d", m.modelsMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optModelsMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"
		case optModelsOutcomeRating:
			return "Outcome: rating", m.boolToOnOff(m.modelsOutcomeRating), "include rating"
		case optModelsOutcomePainResidual:
			return "Outcome: pain_residual", m.boolToOnOff(m.modelsOutcomePainResidual), "include pain residual"
		case optModelsOutcomeTemperature:
			return "Outcome: temperature", m.boolToOnOff(m.modelsOutcomeTemperature), "include temperature"
		case optModelsOutcomePainBinary:
			return "Outcome: pain_binary", m.boolToOnOff(m.modelsOutcomePainBinary), "include binary outcome"
		case optModelsFamilyOLS:
			return "Family: OLS-HC3", m.boolToOnOff(m.modelsFamilyOLS), "ols_hc3"
		case optModelsFamilyRobust:
			return "Family: Robust", m.boolToOnOff(m.modelsFamilyRobust), "robust_rlm"
		case optModelsFamilyQuantile:
			return "Family: Quantile", m.boolToOnOff(m.modelsFamilyQuantile), "quantile_50"
		case optModelsFamilyLogit:
			return "Family: Logistic", m.boolToOnOff(m.modelsFamilyLogit), "logit"
		case optModelsBinaryOutcome:
			v := "pain_binary"
			if m.modelsBinaryOutcome == 1 {
				v = "rating_median"
			}
			return "Binary Outcome", v, "for logit models"

		// Stability
		case optStabilityMethod:
			v := "spearman"
			if m.stabilityMethod == 1 {
				v = "pearson"
			}
			return "Method", v, "within-group correlation"
		case optStabilityOutcome:
			v := "auto"
			switch m.stabilityOutcome {
			case 1:
				v = "rating"
			case 2:
				v = "pain_residual"
			}
			return "Outcome", v, "auto prefers pain_residual"
		case optStabilityGroupColumn:
			v := "auto"
			switch m.stabilityGroupColumn {
			case 1:
				v = "run"
			case 2:
				v = "block"
			}
			return "Group Column", v, "auto selects run/block"
		case optStabilityPartialTemp:
			return "Partial Temperature", m.boolToOnOff(m.stabilityPartialTemp), "control temperature"
		case optStabilityMaxFeatures:
			val := fmt.Sprintf("%d", m.stabilityMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optStabilityMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "0=no limit"
		case optStabilityAlpha:
			val := fmt.Sprintf("%.3f", m.stabilityAlpha)
			if m.editingNumber && m.isCurrentlyEditing(optStabilityAlpha) {
				val = numberDisplay
			}
			return "Alpha", val, "stability cutoff"

		// Consistency
		case optConsistencyEnabled:
			return "Consistency Summary", m.boolToOnOff(m.consistencyEnabled), "flag sign flips"

		// Influence
		case optInfluenceOutcomeRating:
			return "Outcome: rating", m.boolToOnOff(m.influenceOutcomeRating), "include rating"
		case optInfluenceOutcomePainResidual:
			return "Outcome: pain_residual", m.boolToOnOff(m.influenceOutcomePainResidual), "include residual"
		case optInfluenceOutcomeTemperature:
			return "Outcome: temperature", m.boolToOnOff(m.influenceOutcomeTemperature), "include temperature"
		case optInfluenceMaxFeatures:
			val := fmt.Sprintf("%d", m.influenceMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "top effects to inspect"
		case optInfluenceIncludeTemperature:
			return "Include Temperature", m.boolToOnOff(m.influenceIncludeTemperature), "add covariate"
		case optInfluenceTempControl:
			v := "linear"
			switch m.influenceTempControl {
			case 1:
				v = "rating_hat"
			case 2:
				v = "spline"
			}
			return "Temp Control", v, "linear | rating_hat | spline"
		case optInfluenceTempSplineKnots:
			val := fmt.Sprintf("%d", m.influenceTempSplineKnots)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineKnots) {
				val = numberDisplay
			}
			return "Temp Spline Knots", val, "restricted cubic (>=4)"
		case optInfluenceTempSplineQlow:
			val := fmt.Sprintf("%.3f", m.influenceTempSplineQlow)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineQlow) {
				val = numberDisplay
			}
			return "Spline Q Low", val, "knot quantile"
		case optInfluenceTempSplineQhigh:
			val := fmt.Sprintf("%.3f", m.influenceTempSplineQhigh)
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceTempSplineQhigh) {
				val = numberDisplay
			}
			return "Spline Q High", val, "knot quantile"
		case optInfluenceIncludeTrialOrder:
			return "Include Trial Order", m.boolToOnOff(m.influenceIncludeTrialOrder), "add covariate"
		case optInfluenceIncludeRunBlock:
			return "Include Run/Block", m.boolToOnOff(m.influenceIncludeRunBlock), "categorical controls"
		case optInfluenceIncludeInteraction:
			return "Feature×Temp", m.boolToOnOff(m.influenceIncludeInteraction), "moderation term"
		case optInfluenceStandardize:
			return "Standardize", m.boolToOnOff(m.influenceStandardize), "z-score predictors"
		case optInfluenceCooksThreshold:
			val := "auto"
			if m.influenceCooksThreshold > 0 {
				val = fmt.Sprintf("%.4f", m.influenceCooksThreshold)
			}
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceCooksThreshold) {
				val = numberDisplay
			}
			return "Cook's Threshold", val, "0=auto heuristic"
		case optInfluenceLeverageThreshold:
			val := "auto"
			if m.influenceLeverageThreshold > 0 {
				val = fmt.Sprintf("%.4f", m.influenceLeverageThreshold)
			}
			if m.editingNumber && m.isCurrentlyEditing(optInfluenceLeverageThreshold) {
				val = numberDisplay
			}
			return "Leverage Threshold", val, "0=auto heuristic"

		// Condition
		case optConditionCompareColumn:
			val := m.conditionCompareColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Compare Column", val, hint
		case optConditionCompareWindows:
			val := m.conditionCompareWindows
			if val == "" {
				val = "(select windows)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareWindows {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableWindows) > 0 {
				hint = fmt.Sprintf("Space to select · %d windows available", len(m.availableWindows))
			}
			return "Compare Windows", val, hint
		case optConditionCompareValues:
			if m.conditionCompareColumn == "" {
				return "Compare Values", "(select column first)", "requires column selection"
			}
			val := m.conditionCompareValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldConditionCompareValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.conditionCompareColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.conditionCompareColumn)
			}
			return "Compare Values", val, hint
		case optConditionWindowPrimaryUnit:
			v := "trial"
			if m.conditionWindowPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Window Unit", v, "trial | run_mean"
		case optConditionPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.conditionPermutationPrimary), "within-run/block when available"
		case optConditionFailFast:
			return "Fail Fast", m.boolToOnOff(m.conditionFailFast), "error if split fails"
		case optConditionOverwrite:
			return "Overwrite", m.boolToOnOff(m.conditionOverwrite), "overwrite existing condition effects files"
		case optConditionEffectThreshold:
			val := fmt.Sprintf("%.3f", m.conditionEffectThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optConditionEffectThreshold) {
				val = numberDisplay
			}
			return "Effect Threshold", val, "Cohen's d cutoff"

		// Temporal
		case optTemporalResolutionMs:
			val := fmt.Sprintf("%d", m.temporalResolutionMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalResolutionMs) {
				val = numberDisplay
			}
			return "Time Resolution (ms)", val, "bin size"
		case optTemporalTimeMinMs:
			val := fmt.Sprintf("%d", m.temporalTimeMinMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTimeMinMs) {
				val = numberDisplay
			}
			return "Time Min (ms)", val, "window start"
		case optTemporalTimeMaxMs:
			val := fmt.Sprintf("%d", m.temporalTimeMaxMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTimeMaxMs) {
				val = numberDisplay
			}
			return "Time Max (ms)", val, "window end"
		case optTemporalSmoothMs:
			val := fmt.Sprintf("%d", m.temporalSmoothMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalSmoothMs) {
				val = numberDisplay
			}
			return "Smooth Window (ms)", val, "smoothing length"
		case optTemporalTargetColumn:
			val := m.temporalTargetColumn
			if strings.TrimSpace(val) == "" {
				val = "(default: rating)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalTargetColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Target Column", val, hint
		case optTemporalSplitByCondition:
			return "Split by Condition", m.boolToOnOff(m.temporalSplitByCondition), "separate files per condition"
		case optTemporalConditionColumn:
			val := m.temporalConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalConditionColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Condition Column", val, hint
		case optTemporalConditionValues:
			if !m.temporalSplitByCondition {
				return "Condition Values", "N/A", "enable Split by Condition"
			}
			if m.temporalConditionColumn == "" {
				return "Condition Values", "(select column first)", "requires column selection"
			}
			val := m.temporalConditionValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalConditionValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.temporalConditionColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.temporalConditionColumn)
			}
			return "Condition Values", val, hint
		case optTemporalIncludeROIAverages:
			return "Include ROI Averages", m.boolToOnOff(m.temporalIncludeROIAverages), "add ROI-averaged rows to output"
		case optTemporalIncludeTFGrid:
			return "Include TF Grid", m.boolToOnOff(m.temporalIncludeTFGrid), "add individual frequency rows"

		// Temporal feature selection
		case optTemporalFeaturePower:
			return "Feature: Power", m.boolToOnOff(m.temporalFeaturePowerEnabled), "spectral power in bands"
		case optTemporalFeatureITPC:
			return "Feature: ITPC", m.boolToOnOff(m.temporalFeatureITPCEnabled), "inter-trial phase coherence"
		case optTemporalFeatureERDS:
			return "Feature: ERDS", m.boolToOnOff(m.temporalFeatureERDSEnabled), "event-related desync/sync"

		// ITPC-specific options
		case optTemporalITPCBaselineCorrection:
			return "ITPC Baseline Correction", m.boolToOnOff(m.temporalITPCBaselineCorrection), "subtract baseline ITPC"
		case optTemporalITPCBaselineMin:
			val := fmt.Sprintf("%.2f", m.temporalITPCBaselineMin)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalITPCBaselineMin) {
				val = numberDisplay
			}
			return "ITPC Baseline Start", val, "seconds"
		case optTemporalITPCBaselineMax:
			val := fmt.Sprintf("%.2f", m.temporalITPCBaselineMax)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalITPCBaselineMax) {
				val = numberDisplay
			}
			return "ITPC Baseline End", val, "seconds"

		// ERDS-specific options
		case optTemporalERDSBaselineMin:
			val := fmt.Sprintf("%.2f", m.temporalERDSBaselineMin)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalERDSBaselineMin) {
				val = numberDisplay
			}
			return "ERDS Baseline Start", val, "seconds"
		case optTemporalERDSBaselineMax:
			val := fmt.Sprintf("%.2f", m.temporalERDSBaselineMax)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalERDSBaselineMax) {
				val = numberDisplay
			}
			return "ERDS Baseline End", val, "seconds"
		case optTemporalERDSMethod:
			methods := []string{"percent", "zscore"}
			var v string
			if m.temporalERDSMethod >= 0 && m.temporalERDSMethod < len(methods) {
				v = methods[m.temporalERDSMethod]
			} else {
				v = "percent"
			}
			return "ERDS Method", v, "ERDS normalization"

		// TF Heatmap options
		case optTemporalTfHeatmapEnabled:
			return "TF Heatmap", m.boolToOnOff(m.tfHeatmapEnabled), "time-frequency correlation heatmap"
		case optTemporalTfHeatmapFreqs:
			val := m.tfHeatmapFreqsSpec
			if val == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldTfHeatmapFreqs {
				val = textDisplay
			}
			return "TF Freqs", val, "frequencies for heatmap"
		case optTemporalTfHeatmapTimeResMs:
			val := fmt.Sprintf("%d ms", m.tfHeatmapTimeResMs)
			if m.editingNumber && m.isCurrentlyEditing(optTemporalTfHeatmapTimeResMs) {
				val = numberDisplay
			}
			return "TF Time Res", val, "temporal resolution"

		// Report
		case optReportTopN:
			val := fmt.Sprintf("%d", m.reportTopN)
			if m.editingNumber && m.isCurrentlyEditing(optReportTopN) {
				val = numberDisplay
			}
			return "Top N Rows", val, "per TSV in report"

		// Correlations
		case optCorrelationsTargetRating:
			return "Target: rating", m.boolToOnOff(m.correlationsTargetRating), "include rating"
		case optCorrelationsTargetTemperature:
			return "Target: temperature", m.boolToOnOff(m.correlationsTargetTemperature), "include temperature"
		case optCorrelationsTargetPainResidual:
			return "Target: pain_residual", m.boolToOnOff(m.correlationsTargetPainResidual), "include residual"
		case optCorrelationsPreferPainResidual:
			return "Prefer pain_residual", m.boolToOnOff(m.correlationsPreferPainResidual), "auto target selection"
		case optCorrelationsTypes:
			val := m.correlationsTypesSpec
			if strings.TrimSpace(val) == "" {
				val = "(default: partial_cov_temp)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTypes {
				val = textDisplay
			}
			return "Correlation Types", val, "comma-separated: raw,partial_cov,partial_temp,partial_cov_temp,run_mean"
		case optCorrelationsUseCrossfitPainResidual:
			return "Use pain_residual_cv", m.boolToOnOff(m.correlationsUseCrossfitResidual), "requires residual crossfit"
		case optCorrelationsPrimaryUnit:
			v := "trial"
			if m.correlationsPrimaryUnit == 1 {
				v = "run_mean"
			}
			return "Primary Unit", v, "trial | run_mean"
		case optCorrelationsPermutationPrimary:
			return "Permutation p-primary", m.boolToOnOff(m.correlationsPermutationPrimary), "within-run/block when available"
		case optCorrelationsTargetColumn:
			val := m.correlationsTargetColumn
			if strings.TrimSpace(val) == "" {
				val = "(not set)"
			}
			if m.editingText && m.editingTextField == textFieldCorrelationsTargetColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Custom Target Column", val, hint
		case optCorrelationsMultilevel:
			enabled := m.isComputationSelected("multilevel_correlations")
			val := "No"
			if enabled {
				val = "Yes"
			}
			return "Group Multilevel Correlations", val, "Space to toggle"
		case optGroupLevelBlockPermutation:
			if !m.isComputationSelected("multilevel_correlations") {
				return "Block Permutation", "N/A", "enable Group Multilevel Correlations"
			}
			return "Block Permutation", m.boolToOnOff(m.groupLevelBlockPermutation), "block-restricted when available"

		// Cluster
		case optClusterThreshold:
			val := fmt.Sprintf("%.4f", m.clusterThreshold)
			if m.editingNumber && m.isCurrentlyEditing(optClusterThreshold) {
				val = numberDisplay
			}
			return "Cluster Threshold", val, "forming threshold"
		case optClusterMinSize:
			val := fmt.Sprintf("%d", m.clusterMinSize)
			if m.editingNumber && m.isCurrentlyEditing(optClusterMinSize) {
				val = numberDisplay
			}
			return "Min Cluster Size", val, "minimum cluster size"
		case optClusterTail:
			v := "two-tailed"
			switch m.clusterTail {
			case 1:
				v = "upper"
			case -1:
				v = "lower"
			}
			return "Cluster Tail", v, "test direction"
		case optClusterConditionColumn:
			val := m.clusterConditionColumn
			if val == "" {
				val = "(select column)"
			}
			if m.editingText && m.editingTextField == textFieldClusterConditionColumn {
				val = textDisplay
			}
			hint := "Space to select"
			if len(m.availableColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.availableColumns))
			}
			return "Cluster Column", val, hint
		case optClusterConditionValues:
			if m.clusterConditionColumn == "" {
				return "Cluster Values", "(select column first)", "requires column selection"
			}
			val := m.clusterConditionValues
			if val == "" {
				val = "(select values)"
			}
			if m.editingText && m.editingTextField == textFieldClusterConditionValues {
				val = textDisplay
			}
			hint := "Space to select"
			if vals := m.GetDiscoveredColumnValues(m.clusterConditionColumn); len(vals) > 0 {
				hint = fmt.Sprintf("Space to select · %d values in %s", len(vals), m.clusterConditionColumn)
			}
			return "Cluster Values", val, hint

		// Mediation
		case optMediationBootstrap:
			val := fmt.Sprintf("%d", m.mediationBootstrap)
			if m.editingNumber && m.isCurrentlyEditing(optMediationBootstrap) {
				val = numberDisplay
			}
			return "Mediation Bootstrap", val, "bootstrap iterations"
		case optMediationPermutations:
			val := fmt.Sprintf("%d", m.mediationPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optMediationPermutations) {
				val = numberDisplay
			}
			return "Mediation Permutations", val, "0=disabled"
		case optMediationMinEffect:
			val := fmt.Sprintf("%.3f", m.mediationMinEffect)
			if m.editingNumber && m.isCurrentlyEditing(optMediationMinEffect) {
				val = numberDisplay
			}
			return "Min Effect Size", val, "minimum indirect effect"
		case optMediationMaxMediatorsEnabled:
			return "Limit Max Mediators", m.boolToOnOff(m.mediationMaxMediatorsEnabled), "enable mediator limit"
		case optMediationMaxMediators:
			if !m.mediationMaxMediatorsEnabled {
				return "Max Mediators", "N/A", "limit disabled"
			}
			val := fmt.Sprintf("%d", m.mediationMaxMediators)
			if m.editingNumber && m.isCurrentlyEditing(optMediationMaxMediators) {
				val = numberDisplay
			}
			return "Max Mediators", val, "max mediators tested"

		// Moderation
		case optModerationMaxFeaturesEnabled:
			return "Limit Max Features", m.boolToOnOff(m.moderationMaxFeaturesEnabled), "enable feature limit"
		case optModerationMaxFeatures:
			if !m.moderationMaxFeaturesEnabled {
				return "Max Features", "N/A", "limit disabled"
			}
			val := fmt.Sprintf("%d", m.moderationMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optModerationMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "max features for moderation"
		case optModerationPermutations:
			val := fmt.Sprintf("%d", m.moderationPermutations)
			if m.editingNumber && m.isCurrentlyEditing(optModerationPermutations) {
				val = numberDisplay
			}
			return "Moderation Permutations", val, "0=disabled"

		// Mixed effects
		case optMixedEffectsType:
			v := "intercept"
			if m.mixedEffectsType == 1 {
				v = "intercept_slope"
			}
			return "Random Effects", v, "group-level only"
		case optMixedMaxFeatures:
			val := fmt.Sprintf("%d", m.mixedMaxFeatures)
			if m.editingNumber && m.isCurrentlyEditing(optMixedMaxFeatures) {
				val = numberDisplay
			}
			return "Max Features", val, "max features to include"

		// Output section
		case optBehaviorGroupOutput:
			label := "▸ Output"
			if m.behaviorGroupOutputExpanded {
				label = "▾ Output"
			}
			return label, "", "Space to toggle"
		case optAlsoSaveCsv:
			return "Also Save CSV", m.boolToOnOff(m.alsoSaveCsv), "save tables as both TSV and CSV"
		case optBehaviorOverwrite:
			return "Overwrite Outputs", m.boolToOnOff(m.behaviorOverwrite), "if off, append timestamp to output folders"

		default:
			return "", "", ""
		}
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}

	totalLines := len(options)
	startIdx, endIdx, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	// Show scroll indicator for items above
	if showScrollIndicators && startIdx > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more items above", startIdx)) + "\n")
	}

	for i := startIdx; i < endIdx; i++ {
		opt := options[i]
		isFocused := i == m.advancedCursor
		label, value, hint := getOptionDisplay(opt)

		// Check if this is a section header option
		isSectionHeader := opt >= optBehaviorGroupGeneral && opt <= optBehaviorGroupMixedEffects

		var labelStyle, valueStyle lipgloss.Style
		if isSectionHeader {
			// Section headers get special styling (like features pipeline)
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}
			valueStyle = lipgloss.NewStyle()
		} else if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		if m.useDefaultAdvanced && i > 0 {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		} else if m.editingNumber && isFocused {
			// Highlight the editing field
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		if isSectionHeader {
			// Section headers don't have a colon after the label
			b.WriteString(cursor + labelStyle.Render(label) + "  " + hintStyle.Render(hint) + "\n")
		} else {
			b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value))
			b.WriteString("  " + hintStyle.Render(hint) + "\n")
		}

		// Render expanded column/value list after the relevant option
		if m.shouldRenderExpandedListAfterOption(opt) {
			items := m.getExpandedListItems()
			subIndent := "      " // 6 spaces for sub-items
			for j, item := range items {
				isSubFocused := j == m.subCursor
				isSelected := m.isExpandedItemSelected(j, item)

				checkbox := styles.RenderCheckbox(isSelected, isSubFocused)

				itemStyle := lipgloss.NewStyle().Foreground(styles.Text)
				if isSubFocused {
					itemStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
				}

				b.WriteString(subIndent + checkbox + " " + itemStyle.Render(item) + "\n")
			}
		}
	}

	// Show scroll indicator for items below
	if showScrollIndicators && endIdx < len(options) {
		remaining := len(options) - endIdx
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more items below", remaining)) + "\n")
	}

	return b.String()
}

func (m Model) renderMLAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	b.WriteString(infoStyle.Render("  Configure machine learning analysis parameters.") + "\n")
	b.WriteString(infoStyle.Render("  Press Space to toggle/cycle values, Enter to proceed.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	permutationsVal := fmt.Sprintf("%d", m.mlNPerm)
	innerSplitsVal := fmt.Sprintf("%d", m.innerSplits)
	outerJobsVal := fmt.Sprintf("%d", m.outerJobs)
	rngSeedVal := m.rngSeedDisplay()
	rfNEstimatorsVal := fmt.Sprintf("%d", m.rfNEstimators)

	// Override with input buffer if editing that field
	if m.editingNumber {
		inputDisplay := m.numberBuffer + "█"
		if m.isCurrentlyEditing(optMLNPerm) {
			permutationsVal = inputDisplay
		} else if m.isCurrentlyEditing(optMLInnerSplits) {
			innerSplitsVal = inputDisplay
		} else if m.isCurrentlyEditing(optMLOuterJobs) {
			outerJobsVal = inputDisplay
		} else if m.isCurrentlyEditing(optRNGSeed) {
			rngSeedVal = inputDisplay
		} else if m.isCurrentlyEditing(optRfNEstimators) {
			rfNEstimatorsVal = inputDisplay
		}
	}

	// Text editing for grid values
	elasticNetAlphaVal := m.elasticNetAlphaGrid
	elasticNetL1Val := m.elasticNetL1RatioGrid
	rfMaxDepthVal := m.rfMaxDepthGrid
	if m.editingText {
		inputDisplay := m.textBuffer + "█"
		// The text buffer is shared, check which field is focused
		options := m.getMLOptions()
		if m.advancedCursor < len(options) {
			switch options[m.advancedCursor] {
			case optElasticNetAlphaGrid:
				elasticNetAlphaVal = inputDisplay
			case optElasticNetL1RatioGrid:
				elasticNetL1Val = inputDisplay
			case optRfMaxDepthGrid:
				rfMaxDepthVal = inputDisplay
			}
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
		{"Outer Jobs", outerJobsVal, "Parallel jobs for outer CV"},
		{"RNG Seed", rngSeedVal, "0=project default"},
		{"Skip Time-Gen", m.boolToOnOff(m.skipTimeGen), "Skip time generalization"},
		{"ElasticNet α Grid", elasticNetAlphaVal, "comma-separated alphas"},
		{"ElasticNet L1 Grid", elasticNetL1Val, "comma-separated L1 ratios"},
		{"RF N Estimators", rfNEstimatorsVal, "number of trees"},
		{"RF Max Depth Grid", rfMaxDepthVal, "depths (use 'null' for None)"},
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

func (m Model) renderPreprocessingAdvancedConfig() string {
	var b strings.Builder

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("preprocessing")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Type a number, press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to expand/toggle · ↑↓ navigate · Enter proceed") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build values for display
	nJobsVal := fmt.Sprintf("%d", m.prepNJobs)
	montageVal := m.prepMontage
	if m.editingText && m.editingTextField == textFieldPrepMontage {
		montageVal = m.textBuffer + "█"
	}
	chTypesVal := m.prepChTypes
	if m.editingText && m.editingTextField == textFieldPrepChTypes {
		chTypesVal = m.textBuffer + "█"
	}
	eegRefVal := m.prepEegReference
	if m.editingText && m.editingTextField == textFieldPrepEegReference {
		eegRefVal = m.textBuffer + "█"
	}
	fileExtVal := m.prepFileExtension
	if m.editingText && m.editingTextField == textFieldPrepFileExtension {
		fileExtVal = m.textBuffer + "█"
	}
	renameAnotDictVal := m.prepRenameAnotDict
	if m.editingText && m.editingTextField == textFieldPrepRenameAnotDict {
		renameAnotDictVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(renameAnotDictVal) == "" {
		renameAnotDictVal = "(none)"
	}
	customBadDictVal := m.prepCustomBadDict
	if m.editingText && m.editingTextField == textFieldPrepCustomBadDict {
		customBadDictVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(customBadDictVal) == "" {
		customBadDictVal = "(none)"
	}
	icaLabelsVal := m.icaLabelsToKeep
	if m.editingText && m.editingTextField == textFieldIcaLabelsToKeep {
		icaLabelsVal = m.textBuffer + "█"
	}
	if strings.TrimSpace(icaLabelsVal) == "" {
		icaLabelsVal = "(default)"
	}
	eogChannelsVal := m.prepEogChannels
	if m.editingText && m.editingTextField == textFieldPrepEogChannels {
		eogChannelsVal = m.textBuffer + "█"
	}
	conditionsVal := m.prepConditions
	if m.editingText && m.editingTextField == textFieldPrepConditions {
		conditionsVal = m.textBuffer + "█"
	}
	resampleVal := fmt.Sprintf("%d Hz", m.prepResample)
	lFreqVal := fmt.Sprintf("%.1f Hz", m.prepLFreq)
	hFreqVal := fmt.Sprintf("%.1f Hz", m.prepHFreq)
	notchVal := fmt.Sprintf("%d Hz", m.prepNotch)
	lineFreqVal := fmt.Sprintf("%d Hz", m.prepLineFreq)
	icaCompVal := fmt.Sprintf("%.2f", m.prepICAComp)
	probThreshVal := fmt.Sprintf("%.1f", m.prepProbThresh)
	tminVal := fmt.Sprintf("%.1f s", m.prepEpochsTmin)
	tmaxVal := fmt.Sprintf("%.1f s", m.prepEpochsTmax)
	repeatsVal := fmt.Sprintf("%d", m.prepRepeats)
	breaksMinLenVal := fmt.Sprintf("%d s", m.prepBreaksMinLength)
	tStartPrevVal := fmt.Sprintf("%d s", m.prepTStartAfterPrevious)
	tStopNextVal := fmt.Sprintf("%d s", m.prepTStopBeforeNext)
	randomStateVal := fmt.Sprintf("%d", m.prepRandomState)
	zaplineFlineVal := fmt.Sprintf("%.1f Hz", m.prepZaplineFline)
	icaLFreqVal := fmt.Sprintf("%.1f Hz", m.prepICALFreq)
	icaRejThreshVal := fmt.Sprintf("%.0f µV", m.prepICARejThresh)

	var baselineVal string
	if m.prepEpochsNoBaseline {
		baselineVal = "N/A"
	} else if m.prepEpochsBaselineStart == 0 && m.prepEpochsBaselineEnd == 0 {
		baselineVal = "(default)"
	} else {
		baselineVal = fmt.Sprintf("%.2f to %.2f s", m.prepEpochsBaselineStart, m.prepEpochsBaselineEnd)
	}
	var rejectVal string
	if m.prepEpochsReject == 0 {
		rejectVal = "(none)"
	} else {
		rejectVal = fmt.Sprintf("%.0f µV", m.prepEpochsReject)
	}

	// Input overrides for number editing
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optPrepNJobs):
			nJobsVal = buffer
		case m.isCurrentlyEditing(optPrepResample):
			resampleVal = buffer
		case m.isCurrentlyEditing(optPrepLFreq):
			lFreqVal = buffer
		case m.isCurrentlyEditing(optPrepHFreq):
			hFreqVal = buffer
		case m.isCurrentlyEditing(optPrepNotch):
			notchVal = buffer
		case m.isCurrentlyEditing(optPrepLineFreq):
			lineFreqVal = buffer
		case m.isCurrentlyEditing(optPrepICAComp):
			icaCompVal = buffer
		case m.isCurrentlyEditing(optPrepProbThresh):
			probThreshVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsTmin):
			tminVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsTmax):
			tmaxVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsBaseline):
			baselineVal = buffer
		case m.isCurrentlyEditing(optPrepEpochsReject):
			rejectVal = buffer
		case m.isCurrentlyEditing(optPrepRepeats):
			repeatsVal = buffer
		case m.isCurrentlyEditing(optPrepBreaksMinLength):
			breaksMinLenVal = buffer
		case m.isCurrentlyEditing(optPrepTStartAfterPrevious):
			tStartPrevVal = buffer
		case m.isCurrentlyEditing(optPrepTStopBeforeNext):
			tStopNextVal = buffer
		case m.isCurrentlyEditing(optPrepRandomState):
			randomStateVal = buffer
		case m.isCurrentlyEditing(optPrepZaplineFline):
			zaplineFlineVal = buffer
		case m.isCurrentlyEditing(optPrepICALFreq):
			icaLFreqVal = buffer
		case m.isCurrentlyEditing(optPrepICARejThresh):
			icaRejThreshVal = buffer
		}
	}

	options := m.getPreprocessingOptions()

	// Scrolling support
	totalLines := len(options)
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}
	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more above", startLine)) + "\n")
	}

	for i, opt := range options {
		if i < startLine || i >= endLine {
			continue
		}

		isFocused := i == m.advancedCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}
		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		label := ""
		value := ""
		hint := ""

		switch opt {
		case optUseDefaults:
			label = "Configuration"
			if m.useDefaultAdvanced {
				value = "Using Defaults"
				hint = "Space to customize"
			} else {
				value = "Custom"
				hint = "Space to use defaults"
			}

		// Group headers with chevron indicators
		case optPrepGroupStages:
			if m.prepGroupStagesExpanded {
				label = "▾ Stages"
			} else {
				label = "▸ Stages"
			}
			hint = "Choose preprocessing steps"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optPrepGroupGeneral:
			if m.prepGroupGeneralExpanded {
				label = "▾ General"
			} else {
				label = "▸ General"
			}
			hint = "Montage, parallel jobs, random seed"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optPrepGroupFiltering:
			if m.prepGroupFilteringExpanded {
				label = "▾ Filtering"
			} else {
				label = "▸ Filtering"
			}
			hint = "Resampling, bandpass, notch filters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optPrepGroupPyprep:
			if m.prepGroupPyprepExpanded {
				label = "▾ PyPREP"
			} else {
				label = "▸ PyPREP"
			}
			hint = "Bad channel detection parameters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optPrepGroupICA:
			if m.prepGroupICAExpanded {
				label = "▾ ICA"
			} else {
				label = "▸ ICA"
			}
			hint = "ICA algorithm and parameters"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optPrepGroupEpoching:
			if m.prepGroupEpochingExpanded {
				label = "▾ Epoching"
			} else {
				label = "▸ Epoching"
			}
			hint = "Epoch timing and rejection criteria"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		// Stage toggles (indented under Stages group)
		case optPrepStageBadChannels:
			label = "Bad Channels"
			value = m.boolToOnOff(m.prepStageSelected[0])
			hint = "Automatically detect noisy channels"
		case optPrepStageFiltering:
			label = "Filtering"
			value = m.boolToOnOff(m.prepStageSelected[1])
			hint = "Apply frequency filters to data"
		case optPrepStageICA:
			label = "ICA"
			value = m.boolToOnOff(m.prepStageSelected[2])
			hint = "Remove artifacts via ICA decomposition"
		case optPrepStageEpoching:
			label = "Epoching"
			value = m.boolToOnOff(m.prepStageSelected[3])
			hint = "Extract time-locked epochs from events"

		// General settings (indented)
		case optPrepMontage:
			label = "Montage"
			value = montageVal
			hint = "EEG electrode layout (e.g., easycap-M1)"
		case optPrepChTypes:
			label = "Ch Types"
			value = chTypesVal
			hint = "Channel types to process (e.g., eeg)"
		case optPrepEegReference:
			label = "Reference"
			value = eegRefVal
			hint = "Re-referencing scheme (e.g., average)"
		case optPrepEogChannels:
			label = "EOG Chans"
			value = eogChannelsVal
			if strings.TrimSpace(value) == "" {
				value = "(none)"
			}
			hint = "Eye movement channels (e.g., Fp1,Fp2)"
		case optPrepRandomState:
			label = "Random Seed"
			value = randomStateVal
			hint = "Random seed for reproducible results"
		case optPrepTaskIsRest:
			label = "Resting State"
			value = m.boolToOnOff(m.prepTaskIsRest)
			hint = "Data is resting-state (no events)"
		case optPrepNJobs:
			label = "N Jobs"
			value = nJobsVal
			hint = "Number of parallel processes to use"
		case optPrepUsePyprep:
			label = "Use PyPREP"
			value = m.boolToOnOff(m.prepUsePyprep)
			hint = "Enable PyPREP for bad channel detection"
		case optPrepUseIcalabel:
			label = "Use ICA-Label"
			value = m.boolToOnOff(m.prepUseIcalabel)
			hint = "Auto-classify ICA components as brain/artifact"

		// Filtering settings (indented)
		case optPrepResample:
			label = "Resample"
			value = resampleVal
			hint = "Downsample to this sampling rate"
		case optPrepLFreq:
			label = "High-Pass"
			value = lFreqVal
			hint = "High-pass filter cutoff frequency"
		case optPrepHFreq:
			label = "Low-Pass"
			value = hFreqVal
			hint = "Low-pass filter cutoff frequency"
		case optPrepNotch:
			label = "Notch"
			value = notchVal
			hint = "Remove line noise at this frequency"
		case optPrepLineFreq:
			label = "Line Freq"
			value = lineFreqVal
			hint = "Power line frequency (50 or 60 Hz)"
		case optPrepZaplineFline:
			label = "Zapline"
			value = zaplineFlineVal
			hint = "Zapline: remove power line harmonics"
		case optPrepFindBreaks:
			label = "Find Breaks"
			value = m.boolToOnOff(m.prepFindBreaks)
			hint = "Identify gaps in continuous recording"

		// PyPREP settings (indented)
		case optPrepRansac:
			label = "RANSAC"
			value = m.boolToOnOff(m.prepRansac)
			hint = "RANSAC: robust bad channel detection"
		case optPrepRepeats:
			label = "Repeats"
			value = repeatsVal
			hint = "Number of detection iterations to run"
		case optPrepAverageReref:
			label = "Avg Reref"
			value = m.boolToOnOff(m.prepAverageReref)
			hint = "Average reference before detection"
		case optPrepFileExtension:
			label = "File Ext"
			value = fileExtVal
			hint = "Raw data file extension (e.g., .vhdr)"
		case optPrepConsiderPreviousBads:
			label = "Keep Bads"
			value = m.boolToOnOff(m.prepConsiderPreviousBads)
			hint = "Keep channels marked bad in previous runs"
		case optPrepOverwriteChansTsv:
			label = "Overwrite TSV"
			value = m.boolToOnOff(m.prepOverwriteChansTsv)
			hint = "Update channels.tsv with detected bads"
		case optPrepDeleteBreaks:
			label = "Del Breaks"
			value = m.boolToOnOff(m.prepDeleteBreaks)
			hint = "Remove break periods from data"
		case optPrepBreaksMinLength:
			label = "Break Len"
			value = breaksMinLenVal
			hint = "Minimum duration to qualify as break"
		case optPrepTStartAfterPrevious:
			label = "T Start"
			value = tStartPrevVal
			hint = "Seconds after previous event to include"
		case optPrepTStopBeforeNext:
			label = "T Stop"
			value = tStopNextVal
			hint = "Seconds before next event to include"
		case optPrepRenameAnotDict:
			label = "Rename Anot"
			value = renameAnotDictVal
			hint = "JSON: rename annotations (old:new)"
		case optPrepCustomBadDict:
			label = "Custom Bads"
			value = customBadDictVal
			hint = "JSON: custom bad channels per task/subject"

		// ICA settings (indented)
		case optPrepSpatialFilter:
			spatialFilterVal := []string{"ica", "ssp"}[m.prepSpatialFilter]
			label = "Spatial Filter"
			value = spatialFilterVal
			hint = "Spatial filter: ICA or SSP"
		case optPrepICAAlgorithm:
			icaAlgVal := []string{"extended_infomax", "fastica", "infomax", "picard"}[m.prepICAAlgorithm]
			label = "Algorithm"
			value = icaAlgVal
			hint = "ICA decomposition algorithm"
		case optPrepICAComp:
			label = "Components"
			value = icaCompVal
			hint = "Components: number (int) or variance (0-1)"
		case optPrepICALFreq:
			label = "ICA High-Pass"
			value = icaLFreqVal
			hint = "High-pass filter for ICA preprocessing"
		case optPrepICARejThresh:
			label = "ICA Reject"
			value = icaRejThreshVal
			hint = "Peak-to-peak threshold for ICA epochs (µV)"
		case optPrepProbThresh:
			label = "Prob Thresh"
			value = probThreshVal
			hint = "Minimum probability for IC label acceptance"
		case optPrepKeepMnebidsBads:
			label = "Keep BIDS"
			value = m.boolToOnOff(m.prepKeepMnebidsBads)
			hint = "Keep ICs flagged as bad in MNE-BIDS"
		case optIcaLabelsToKeep:
			label = "Labels Keep"
			value = icaLabelsVal
			hint = "Comma-separated IC labels to keep (e.g., brain,other)"

		// Epoching settings (indented)
		case optPrepConditions:
			condVal := conditionsVal
			if strings.TrimSpace(condVal) == "" {
				condVal = "(default)"
			}
			label = "Conditions"
			value = condVal
			hint = "Event types/triggers to epoch"
		case optPrepEpochsTmin:
			label = "Tmin"
			value = tminVal
			hint = "Epoch start time relative to event (s)"
		case optPrepEpochsTmax:
			label = "Tmax"
			value = tmaxVal
			hint = "Epoch end time relative to event (s)"
		case optPrepEpochsNoBaseline:
			label = "No Baseline"
			value = m.boolToOnOff(m.prepEpochsNoBaseline)
			hint = "Disable baseline correction"
		case optPrepEpochsBaseline:
			label = "Baseline"
			value = baselineVal
			hint = "Baseline correction window (start, end) seconds"
		case optPrepEpochsReject:
			label = "Reject (µV)"
			value = rejectVal
			hint = "Reject epochs exceeding this amplitude (µV)"
		case optPrepRejectMethod:
			rejectMethods := []string{"none", "autoreject_local", "autoreject_global"}
			label = "Reject Method"
			value = rejectMethods[m.prepRejectMethod]
			hint = "Algorithm: none, autoreject_local, autoreject_global"
		case optPrepRunSourceEstimation:
			label = "Source Est"
			value = m.boolToOnOff(m.prepRunSourceEstimation)
			hint = "Run source localization after preprocessing"
		case optPrepWriteCleanEvents:
			label = "Write Clean Events"
			value = m.boolToOnOff(m.prepWriteCleanEvents)
			hint = "Write clean events.tsv aligned to kept epochs"
		case optPrepOverwriteCleanEvents:
			label = "Overwrite Clean Events"
			value = m.boolToOnOff(m.prepOverwriteCleanEvents)
			hint = "Overwrite existing clean events.tsv"
		case optPrepCleanEventsStrict:
			label = "Clean Events Strict"
			value = m.boolToOnOff(m.prepCleanEventsStrict)
			hint = "Fail if clean events.tsv cannot be written"
		}

		b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value))
		if hint != "" {
			b.WriteString("  " + hintStyle.Render(hint))
		}
		b.WriteString("\n")
	}

	if showScrollIndicators && endLine < totalLines {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more below", totalLines-endLine)) + "\n")
	}

	return b.String()
}

func (m Model) renderFmriAdvancedConfig() string {
	var b strings.Builder

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)

	if m.useDefaultAdvanced {
		return m.renderDefaultConfigView("fMRI preprocessing")
	}

	if m.editingNumber {
		b.WriteString(infoStyle.Render("Type a number, press Enter to confirm or Esc to cancel.") + "\n\n")
	} else if m.editingText {
		b.WriteString(infoStyle.Render("Type text, press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("Space to expand/toggle · ↑↓ navigate · Enter proceed") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build values for display
	engineVal := []string{"docker", "apptainer"}[m.fmriEngineIndex%2]

	imageVal := strings.TrimSpace(m.fmriFmriprepImage)
	if m.editingText && m.editingTextField == textFieldFmriFmriprepImage {
		imageVal = m.textBuffer + "█"
	}
	if imageVal == "" {
		imageVal = "(from config)"
	}

	spacesVal := strings.TrimSpace(m.fmriOutputSpacesSpec)
	if m.editingText && m.editingTextField == textFieldFmriOutputSpaces {
		spacesVal = m.textBuffer + "█"
	}
	if spacesVal == "" {
		spacesVal = "(default)"
	}

	ignoreVal := strings.TrimSpace(m.fmriIgnoreSpec)
	if m.editingText && m.editingTextField == textFieldFmriIgnore {
		ignoreVal = m.textBuffer + "█"
	}
	if ignoreVal == "" {
		ignoreVal = "(none)"
	}

	levelOptions := []string{"full", "resampling", "minimal"}
	levelVal := levelOptions[m.fmriLevelIndex%3]

	ciftiOptions := []string{"disabled", "91k", "170k"}
	ciftiVal := ciftiOptions[m.fmriCiftiOutputIndex%3]

	taskIdVal := strings.TrimSpace(m.fmriTaskId)
	if m.editingText && m.editingTextField == textFieldFmriTaskId {
		taskIdVal = m.textBuffer + "█"
	}
	if taskIdVal == "" {
		taskIdVal = "(all tasks)"
	}

	nthreadsVal := fmt.Sprintf("%d", m.fmriNThreads)
	if m.fmriNThreads == 0 {
		nthreadsVal = "(auto)"
	}
	ompNthreadsVal := fmt.Sprintf("%d", m.fmriOmpNThreads)
	if m.fmriOmpNThreads == 0 {
		ompNthreadsVal = "(auto)"
	}
	memVal := fmt.Sprintf("%d", m.fmriMemMb)
	if m.fmriMemMb == 0 {
		memVal = "(auto)"
	}

	skullStripTemplateVal := strings.TrimSpace(m.fmriSkullStripTemplate)
	if m.editingText && m.editingTextField == textFieldFmriSkullStripTemplate {
		skullStripTemplateVal = m.textBuffer + "█"
	}
	if skullStripTemplateVal == "" {
		skullStripTemplateVal = "OASIS30ANTs"
	}

	bold2t1wInitOptions := []string{"register", "header"}
	bold2t1wInitVal := bold2t1wInitOptions[m.fmriBold2T1wInitIndex%2]

	bold2t1wDofVal := fmt.Sprintf("%d", m.fmriBold2T1wDof)
	sliceTimeRefVal := fmt.Sprintf("%.2f", m.fmriSliceTimeRef)
	dummyScansVal := fmt.Sprintf("%d", m.fmriDummyScans)
	if m.fmriDummyScans == 0 {
		dummyScansVal = "(auto)"
	}

	fdSpikeVal := fmt.Sprintf("%.2f", m.fmriFdSpikeThreshold)
	dvarsSpikeVal := fmt.Sprintf("%.2f", m.fmriDvarsSpikeThreshold)

	randomSeedVal := fmt.Sprintf("%d", m.fmriRandomSeed)
	if m.fmriRandomSeed == 0 {
		randomSeedVal = "(random)"
	}

	extraArgsVal := strings.TrimSpace(m.fmriExtraArgs)
	if m.editingText && m.editingTextField == textFieldFmriExtraArgs {
		extraArgsVal = m.textBuffer + "█"
	}
	if extraArgsVal == "" {
		extraArgsVal = "(none)"
	}

	// Handle number editing
	if m.editingNumber {
		buffer := m.numberBuffer + "█"
		switch {
		case m.isCurrentlyEditing(optFmriNThreads):
			nthreadsVal = buffer
		case m.isCurrentlyEditing(optFmriOmpNThreads):
			ompNthreadsVal = buffer
		case m.isCurrentlyEditing(optFmriMemMb):
			memVal = buffer
		case m.isCurrentlyEditing(optFmriBold2T1wDof):
			bold2t1wDofVal = buffer
		case m.isCurrentlyEditing(optFmriSliceTimeRef):
			sliceTimeRefVal = buffer
		case m.isCurrentlyEditing(optFmriDummyScans):
			dummyScansVal = buffer
		case m.isCurrentlyEditing(optFmriFdSpikeThreshold):
			fdSpikeVal = buffer
		case m.isCurrentlyEditing(optFmriDvarsSpikeThreshold):
			dvarsSpikeVal = buffer
		case m.isCurrentlyEditing(optFmriRandomSeed):
			randomSeedVal = buffer
		}
	}

	options := m.getFmriPreprocessingOptions()

	// Scrolling support
	totalLines := len(options)
	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}
	startLine, endLine, showScrollIndicators := calculateScrollWindow(
		totalLines, m.advancedOffset, effectiveHeight, configOverhead)

	if showScrollIndicators && startLine > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↑ %d more above", startLine)) + "\n")
	}

	for i, opt := range options {
		if i < startLine || i >= endLine {
			continue
		}

		isFocused := i == m.advancedCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}
		valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

		label := ""
		value := ""
		hint := ""

		switch opt {
		case optUseDefaults:
			label = "Configuration"
			if m.useDefaultAdvanced {
				value = "Using Defaults"
				hint = "Space to customize"
			} else {
				value = "Custom"
				hint = "Space to use defaults"
			}

		// Group headers with chevron indicators
		case optFmriGroupRuntime:
			if m.fmriGroupRuntimeExpanded {
				label = "▾ Runtime"
			} else {
				label = "▸ Runtime"
			}
			hint = "Container settings"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupOutput:
			if m.fmriGroupOutputExpanded {
				label = "▾ Output"
			} else {
				label = "▸ Output"
			}
			hint = "Output spaces, formats"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupPerformance:
			if m.fmriGroupPerformanceExpanded {
				label = "▾ Performance"
			} else {
				label = "▸ Performance"
			}
			hint = "Threads, memory"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupAnatomical:
			if m.fmriGroupAnatomicalExpanded {
				label = "▾ Anatomical"
			} else {
				label = "▸ Anatomical"
			}
			hint = "FreeSurfer, skull-strip"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupBold:
			if m.fmriGroupBoldExpanded {
				label = "▾ BOLD Processing"
			} else {
				label = "▸ BOLD Processing"
			}
			hint = "Registration, timing"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupQc:
			if m.fmriGroupQcExpanded {
				label = "▾ Quality Control"
			} else {
				label = "▸ Quality Control"
			}
			hint = "Motion thresholds"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupDenoising:
			if m.fmriGroupDenoisingExpanded {
				label = "▾ Denoising"
			} else {
				label = "▸ Denoising"
			}
			hint = "ICA-AROMA"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupSurface:
			if m.fmriGroupSurfaceExpanded {
				label = "▾ Surface"
			} else {
				label = "▸ Surface"
			}
			hint = "Cortical surface options"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupMultiecho:
			if m.fmriGroupMultiechoExpanded {
				label = "▾ Multi-Echo"
			} else {
				label = "▸ Multi-Echo"
			}
			hint = "Multi-echo BOLD"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupRepro:
			if m.fmriGroupReproExpanded {
				label = "▾ Reproducibility"
			} else {
				label = "▸ Reproducibility"
			}
			hint = "Random seeds"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupValidation:
			if m.fmriGroupValidationExpanded {
				label = "▾ Validation"
			} else {
				label = "▸ Validation"
			}
			hint = "BIDS validation, errors"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		case optFmriGroupAdvanced:
			if m.fmriGroupAdvancedExpanded {
				label = "▾ Advanced"
			} else {
				label = "▸ Advanced"
			}
			hint = "Extra CLI arguments"
			if isFocused {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
			}

		// Runtime options (indented)
		case optFmriEngine:
			label = "Engine"
			value = engineVal
			hint = "docker/apptainer"
		case optFmriFmriprepImage:
			label = "Image"
			value = imageVal
			hint = "Container image"

		// Output options (indented)
		case optFmriOutputSpaces:
			label = "Output Spaces"
			value = spacesVal
			hint = "e.g., T1w MNI152..."
		case optFmriIgnore:
			label = "Ignore"
			value = ignoreVal
			hint = "fieldmaps, slicetiming"
		case optFmriLevel:
			label = "Level"
			value = levelVal
			hint = "full/resampling/minimal"
		case optFmriCiftiOutput:
			label = "CIFTI Output"
			value = ciftiVal
			hint = "91k/170k grayordinates"
		case optFmriTaskId:
			label = "Task ID"
			value = taskIdVal
			hint = "Specific task only"

		// Performance options (indented)
		case optFmriNThreads:
			label = "N Threads"
			value = nthreadsVal
			hint = "Max threads (0=auto)"
		case optFmriOmpNThreads:
			label = "OMP Threads"
			value = ompNthreadsVal
			hint = "Per process (0=auto)"
		case optFmriMemMb:
			label = "Mem (MB)"
			value = memVal
			hint = "Memory limit"
		case optFmriLowMem:
			label = "Low Memory"
			value = m.boolToOnOff(m.fmriLowMem)
			hint = "Reduce memory usage"

		// Anatomical options (indented)
		case optFmriSkipReconstruction:
			label = "Skip Recon-All"
			value = m.boolToOnOff(m.fmriSkipReconstruction)
			hint = "No FreeSurfer"
		case optFmriLongitudinal:
			label = "Longitudinal"
			value = m.boolToOnOff(m.fmriLongitudinal)
			hint = "Unbiased template"
		case optFmriSkullStripTemplate:
			label = "Skull Strip Tpl"
			value = skullStripTemplateVal
			hint = "Brain extraction"
		case optFmriSkullStripFixedSeed:
			label = "Fixed Seed"
			value = m.boolToOnOff(m.fmriSkullStripFixedSeed)
			hint = "Reproducible strip"

		// BOLD options (indented)
		case optFmriBold2T1wInit:
			label = "BOLD→T1w Init"
			value = bold2t1wInitVal
			hint = "register/header"
		case optFmriBold2T1wDof:
			label = "BOLD→T1w DOF"
			value = bold2t1wDofVal
			hint = "Degrees freedom"
		case optFmriSliceTimeRef:
			label = "Slice Time Ref"
			value = sliceTimeRefVal
			hint = "0=start, 0.5=mid, 1=end"
		case optFmriDummyScans:
			label = "Dummy Scans"
			value = dummyScansVal
			hint = "Non-steady vols"

		// QC options (indented)
		case optFmriFdSpikeThreshold:
			label = "FD Threshold"
			value = fdSpikeVal
			hint = "mm"
		case optFmriDvarsSpikeThreshold:
			label = "DVARS Threshold"
			value = dvarsSpikeVal
			hint = "Standardized"

		// Denoising options (indented)
		case optFmriUseAroma:
			label = "Use AROMA"
			value = m.boolToOnOff(m.fmriUseAroma)
			hint = "ICA-AROMA"

		// Surface options (indented)
		case optFmriMedialSurfaceNan:
			label = "Medial NaN"
			value = m.boolToOnOff(m.fmriMedialSurfaceNan)
			hint = "Fill medial wall"
		case optFmriNoMsm:
			label = "No MSM"
			value = m.boolToOnOff(m.fmriNoMsm)
			hint = "Disable MSM-Sulc"

		// Multi-echo options (indented)
		case optFmriMeOutputEchos:
			label = "Output Echos"
			value = m.boolToOnOff(m.fmriMeOutputEchos)
			hint = "Each echo separate"

		// Reproducibility options (indented)
		case optFmriRandomSeed:
			label = "Random Seed"
			value = randomSeedVal
			hint = "0=non-deterministic"

		// Validation options (indented)
		case optFmriSkipBidsValidation:
			label = "Skip Validation"
			value = m.boolToOnOff(m.fmriSkipBidsValidation)
			hint = "Skip bids-validator"
		case optFmriStopOnFirstCrash:
			label = "Stop on Crash"
			value = m.boolToOnOff(m.fmriStopOnFirstCrash)
			hint = "Abort on error"
		case optFmriCleanWorkdir:
			label = "Clean Workdir"
			value = m.boolToOnOff(m.fmriCleanWorkdir)
			hint = "Remove on success"

		// Advanced options (indented)
		case optFmriExtraArgs:
			label = "Extra Args"
			value = extraArgsVal
			hint = "Raw CLI args"
		}

		b.WriteString(cursor + labelStyle.Render(label+":") + " " + valueStyle.Render(value))
		if hint != "" {
			b.WriteString("  " + hintStyle.Render(hint))
		}
		b.WriteString("\n")
	}

	if showScrollIndicators && endLine < totalLines {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ↓ %d more below", totalLines-endLine)) + "\n")
	}

	return b.String()
}

func (m Model) renderRawToBidsAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingNumber || m.editingText {
		b.WriteString(infoStyle.Render("  Press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Customize raw-to-BIDS conversion options.") + "\n")
		b.WriteString(infoStyle.Render("  Press Space to toggle/edit, Enter to proceed.") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	montageVal := m.rawMontage
	if m.editingText && m.editingTextField == textFieldRawMontage {
		montageVal = m.textBuffer + "█"
	}

	prefixVal := m.rawEventPrefixes
	if m.editingText && m.editingTextField == textFieldRawEventPrefixes {
		prefixVal = m.textBuffer + "█"
	}

	lineFreqVal := fmt.Sprintf("%d", m.rawLineFreq)
	if m.editingNumber && m.isCurrentlyEditing(optRawLineFreq) {
		lineFreqVal = m.numberBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Montage", montageVal, "Type to edit"},
		{"Line Freq", lineFreqVal, "Type a number (Hz)"},
		{"Overwrite", m.boolToOnOff(m.rawOverwrite), "Replace existing BIDS files"},
		{"Trim to First Volume", m.boolToOnOff(m.rawTrimToFirstVolume), "Trim EEG to fMRI start"},
		{"Event Prefixes", prefixVal, "Comma-separated (optional)"},
		{"Keep Annotations", m.boolToOnOff(m.rawKeepAnnotations), "Keep all raw annotations"},
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
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		displayVal := opt.value
		if displayVal == "" {
			displayVal = "(none)"
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(displayVal))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderFmriRawToBidsAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingNumber || m.editingText {
		b.WriteString(infoStyle.Render("  Press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Customize fMRI DICOM → BIDS conversion options.") + "\n")
		b.WriteString(infoStyle.Render("  Press Space to toggle/edit, Enter to proceed.") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	sessionVal := m.fmriRawSession
	if m.editingText && m.editingTextField == textFieldFmriRawSession {
		sessionVal = m.textBuffer + "█"
	}

	restTaskVal := m.fmriRawRestTask
	if m.editingText && m.editingTextField == textFieldFmriRawRestTask {
		restTaskVal = m.textBuffer + "█"
	}

	dcm2niixPathVal := m.fmriRawDcm2niixPath
	if m.editingText && m.editingTextField == textFieldFmriRawDcm2niixPath {
		dcm2niixPathVal = m.textBuffer + "█"
	}

	dcm2niixArgsVal := m.fmriRawDcm2niixArgs
	if m.editingText && m.editingTextField == textFieldFmriRawDcm2niixArgs {
		dcm2niixArgsVal = m.textBuffer + "█"
	}

	onsetOffsetVal := fmt.Sprintf("%.3f", m.fmriRawOnsetOffsetS)
	if m.editingNumber && m.isCurrentlyEditing(optFmriRawOnsetOffsetS) {
		onsetOffsetVal = m.numberBuffer + "█"
	}

	dicomMode := "symlink"
	if m.fmriRawDicomModeIndex == 1 {
		dicomMode = "copy"
	} else if m.fmriRawDicomModeIndex == 2 {
		dicomMode = "skip"
	}

	granularity := "phases"
	if m.fmriRawEventGranularity == 1 {
		granularity = "trial"
	}

	onsetRef := "as_is"
	if m.fmriRawOnsetRefIndex == 1 {
		onsetRef = "first_iti_start"
	} else if m.fmriRawOnsetRefIndex == 2 {
		onsetRef = "first_stim_start"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Session", sessionVal, "Optional (e.g., 01)"},
		{"Rest Task", restTaskVal, "Task label for resting-state"},
		{"Include Rest", m.boolToOnOff(m.fmriRawIncludeRest), "Convert resting-state series"},
		{"Include Fieldmaps", m.boolToOnOff(m.fmriRawIncludeFieldmaps), "Convert fieldmaps and set IntendedFor"},
		{"DICOM Mode", dicomMode, "sourcedata handling"},
		{"Overwrite", m.boolToOnOff(m.fmriRawOverwrite), "Replace existing BIDS files"},
		{"Create Events", m.boolToOnOff(m.fmriRawCreateEvents), "From PsychoPy TrialSummary.csv"},
		{"Event Granularity", granularity, "stimulation: phases vs trial"},
		{"Onset Reference", onsetRef, "zero onsets within each run"},
		{"Onset Offset (s)", onsetOffsetVal, "additive shift"},
		{"dcm2niix Path", dcm2niixPathVal, "Optional override"},
		{"dcm2niix Args", dcm2niixArgsVal, "Comma-separated extra args"},
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
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		displayVal := opt.value
		if strings.TrimSpace(displayVal) == "" {
			displayVal = "(none)"
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(displayVal))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderMergeBehaviorAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingText {
		b.WriteString(infoStyle.Render("  Press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Configure merge-behavior filters.") + "\n")
		b.WriteString(infoStyle.Render("  Press Space to toggle/edit, Enter to proceed.") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	prefixVal := m.mergeEventPrefixes
	if m.editingText && m.editingTextField == textFieldMergeEventPrefixes {
		prefixVal = m.textBuffer + "█"
	}
	typeVal := m.mergeEventTypes
	if m.editingText && m.editingTextField == textFieldMergeEventTypes {
		typeVal = m.textBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Event Prefixes", prefixVal, "Comma-separated (optional)"},
		{"Event Types", typeVal, "Comma-separated (optional)"},
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
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		displayVal := opt.value
		if displayVal == "" {
			displayVal = "(none)"
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(displayVal))
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

///////////////////////////////////////////////////////////////////
// Preprocessing Step Rendering
///////////////////////////////////////////////////////////////////

func (m Model) renderPreprocessingStageSelection() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" PREPROCESSING STAGES") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Select which preprocessing stages to run.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, A to select all, N to select none.") + "\n\n")

	// Count selected stages
	selectedCount := 0
	for i := range m.prepStages {
		if m.prepStageSelected[i] {
			selectedCount++
		}
	}

	// Inline validation indicator
	var statusIndicator string
	if selectedCount >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}

	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d stages selected", selectedCount, len(m.prepStages))))
	if selectedCount == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	for i, stage := range m.prepStages {
		isSelected := m.prepStageSelected[i]
		isFocused := i == m.prepStageCursor

		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		b.WriteString(checkbox + nameStyle.Render(stage.Name))

		if len(stage.Description) > 0 {
			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + stage.Description)
			b.WriteString(desc)
		}

		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingFiltering() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" FILTERING OPTIONS") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure frequency filtering and resampling.") + "\n")
	b.WriteString(infoStyle.Render("Press Enter to edit a value, Esc to cancel editing.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build display values
	resampleVal := fmt.Sprintf("%d Hz", m.prepResample)
	if m.editingNumber && m.advancedCursor == 0 {
		resampleVal = m.numberBuffer + "█"
	}

	lFreqVal := fmt.Sprintf("%.1f Hz", m.prepLFreq)
	if m.editingNumber && m.advancedCursor == 1 {
		lFreqVal = m.numberBuffer + "█"
	}

	hFreqVal := fmt.Sprintf("%.1f Hz", m.prepHFreq)
	if m.editingNumber && m.advancedCursor == 2 {
		hFreqVal = m.numberBuffer + "█"
	}

	notchVal := fmt.Sprintf("%d Hz", m.prepNotch)
	if m.editingNumber && m.advancedCursor == 3 {
		notchVal = m.numberBuffer + "█"
	}

	lineFreqVal := fmt.Sprintf("%d Hz", m.prepLineFreq)
	if m.editingNumber && m.advancedCursor == 4 {
		lineFreqVal = m.numberBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Resample Freq", resampleVal, "Target sampling rate (0=disable)"},
		{"High-Pass Freq", lFreqVal, "Low frequency cutoff (Hz)"},
		{"Low-Pass Freq", hFreqVal, "High frequency cutoff (Hz)"},
		{"Notch Freq", notchVal, "Line noise notch filter (0=disable)"},
		{"Line Freq", lineFreqVal, "Power line frequency (50 or 60 Hz)"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

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

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingICA() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" ICA OPTIONS") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure independent component analysis.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, Enter to edit values.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build display values
	methods := []string{"fastica", "infomax", "picard"}
	methodVal := methods[m.prepICAAlgorithm]

	compVal := fmt.Sprintf("%.2f", m.prepICAComp)
	if m.editingNumber && m.advancedCursor == 1 {
		compVal = m.numberBuffer + "█"
	}

	probThreshVal := fmt.Sprintf("%.2f", m.prepProbThresh)
	if m.editingNumber && m.advancedCursor == 2 {
		probThreshVal = m.numberBuffer + "█"
	}

	labelsVal := m.icaLabelsToKeep
	if labelsVal == "" {
		labelsVal = "(default)"
	}
	if m.editingText && m.editingTextField == textFieldIcaLabelsToKeep {
		labelsVal = m.textBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use ICALabel", m.boolToOnOff(m.prepUseIcalabel), "Auto-label components"},
		{"ICA Method", methodVal, "fastica, infomax, or picard"},
		{"Components", compVal, "Number or variance fraction"},
		{"Prob Threshold", probThreshVal, "Label probability threshold"},
		{"Labels to Keep", labelsVal, "Comma-separated (e.g., brain,other)"},
		{"Keep MNE-BIDS Bads", m.boolToOnOff(m.prepKeepMnebidsBads), "Keep MNE-BIDS flagged components"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

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

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingEpochs() string {
	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" EPOCH OPTIONS") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure epoch creation.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, Enter to edit values.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	// Build display values
	tminVal := fmt.Sprintf("%.1f s", m.prepEpochsTmin)
	if m.editingNumber && m.advancedCursor == 0 {
		tminVal = m.numberBuffer + "█"
	}

	tmaxVal := fmt.Sprintf("%.1f s", m.prepEpochsTmax)
	if m.editingNumber && m.advancedCursor == 1 {
		tmaxVal = m.numberBuffer + "█"
	}

	baselineVal := "(none)"
	if !m.prepEpochsNoBaseline {
		baselineVal = fmt.Sprintf("[%.1f, %.1f] s", m.prepEpochsBaselineStart, m.prepEpochsBaselineEnd)
	}

	rejectVal := fmt.Sprintf("%.0f µV", m.prepEpochsReject)
	if m.editingNumber && m.advancedCursor == 3 {
		rejectVal = m.numberBuffer + "█"
	}
	if m.prepEpochsReject == 0 {
		rejectVal = "(disabled)"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Tmin", tminVal, "Epoch start time (seconds)"},
		{"Tmax", tmaxVal, "Epoch end time (seconds)"},
		{"Baseline Correction", m.boolToOnOff(!m.prepEpochsNoBaseline), "Apply baseline correction"},
		{"Baseline Window", baselineVal, "Baseline time window"},
		{"Reject Threshold", rejectVal, "Peak-to-peak amplitude (µV, 0=disable)"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

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

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) selectedDirectedConnectivityDisplay() string {
	var selected []string
	for i, sel := range m.directedConnMeasures {
		if sel && i < len(directedConnectivityMeasures) {
			selected = append(selected, directedConnectivityMeasures[i].Key)
		}
	}
	if len(selected) == 0 {
		return "(none)"
	}
	return strings.Join(selected, ", ")
}
