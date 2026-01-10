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
	if startLine < 0 {
		startLine = 0
	}
	endLine = startLine + maxLines
	return startLine, endLine, showIndicators
}

const (
	defaultLabelWidth     = 22
	defaultLabelWidthWide = 30
	configOverhead        = 10
	plotConfigOverhead    = 8
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

		// Estimated time (rough heuristic)
		estMins := m.estimateExecutionTime(selectedCount)
		timeInfo := lipgloss.NewStyle().Foreground(styles.Text).Render("Est. time:") +
			" " +
			lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(fmt.Sprintf("~%d min", estMins))
		content.WriteString(timeInfo + "\n")
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

// estimateExecutionTime returns rough estimate in minutes based on pipeline and subject count
func (m Model) estimateExecutionTime(subjectCount int) int {
	// Rough estimates per subject (in minutes)
	perSubject := 1
	switch m.Pipeline {
	case types.PipelinePreprocessing:
		perSubject = 5
	case types.PipelineFeatures:
		perSubject = 3
	case types.PipelineBehavior:
		perSubject = 1
	case types.PipelineML:
		perSubject = 2
	case types.PipelinePlotting:
		perSubject = 1
	}
	return perSubject * subjectCount
}

// getExpectedOutputPaths returns the expected output directories for the current pipeline
func (m Model) getExpectedOutputPaths() []string {
	base := "derivatives/"
	switch m.Pipeline {
	case types.PipelinePreprocessing:
		return []string{base + "preprocessed/", base + "epochs/"}
	case types.PipelineFeatures:
		return []string{base + "features/"}
	case types.PipelineBehavior:
		return []string{base + "behavior/", base + "stats/"}
	case types.PipelineML:
		return []string{base + "machine_learning/"}
	case types.PipelinePlotting:
		return []string{base + "plots/"}
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

	// Calculate responsive layout based on terminal height
	// Overhead: header(4) + section title(2) + status(2) + footer(2) + spacing(2) = 12
	layout := styles.CalculateListLayout(m.height, m.computationCursor, len(m.computations), 12)

	// Show scroll up indicator
	if layout.ShowScrollUp {
		b.WriteString(styles.RenderScrollUpIndicator(layout.StartIdx) + "\n")
	}

	for i := layout.StartIdx; i < layout.EndIdx; i++ {
		comp := m.computations[i]
		isSelected := m.computationSelected[i]
		isFocused := i == m.computationCursor

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

	// Show scroll down indicator
	if layout.ShowScrollDn {
		remaining := len(m.computations) - layout.EndIdx
		b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	// Post Computations section (only for behavior pipeline)
	if len(m.postComputations) > 0 {
		b.WriteString("\n")
		b.WriteString(styles.SectionTitleStyle.Render(" POST COMPUTATIONS ") + "\n\n")

		postCount := 0
		for _, sel := range m.postComputationSelected {
			if sel {
				postCount++
			}
		}

		// Inline validation indicator for post computations
		var postStatusIndicator string
		if postCount >= 1 {
			postStatusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
		} else {
			postStatusIndicator = lipgloss.NewStyle().Foreground(styles.TextDim).Render("○ ")
		}

		b.WriteString(postStatusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
			fmt.Sprintf("%d of %d selected", postCount, len(m.postComputations))))
		if postCount == 0 {
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render(" — optional"))
		}
		b.WriteString("\n\n")

		// Calculate responsive layout for post computations
		postLayout := styles.CalculateListLayout(m.height/2, m.postComputationCursor, len(m.postComputations), 4)

		// Show scroll up indicator for post computations
		if postLayout.ShowScrollUp {
			b.WriteString(styles.RenderScrollUpIndicator(postLayout.StartIdx) + "\n")
		}

		for i := postLayout.StartIdx; i < postLayout.EndIdx; i++ {
			comp := m.postComputations[i]
			isSelected := m.postComputationSelected[i]
			isFocused := i == m.postComputationCursor

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

		// Show scroll down indicator for post computations
		if postLayout.ShowScrollDn {
			remaining := len(m.postComputations) - postLayout.EndIdx
			b.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
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
			"Toggle a category to include or exclude all plots in that group.\n\n",
		))
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
	b.WriteString(instructionStyle.Render("  Space: edit • A: add band • D: delete band") + "\n\n")

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
	b.WriteString(instructionStyle.Render("  Space: edit • A: add ROI • D: delete ROI") + "\n\n")

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
			// Truncate channels if too long
			channels := roi.Channels
			if len(channels) > 40 {
				channels = channels[:37] + "..."
			}
			channelsDisplay = channelStyle.Render(channels)
		}

		channelInfo := channelStyle.Render(" [") + channelsDisplay + channelStyle.Render("]")

		b.WriteString(checkbox + nameDisplay + channelInfo)
		b.WriteString("\n")
	}

	return b.String()
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
	title := " PLOT SELECTION "
	b.WriteString(styles.SectionTitleStyle.Render(title) + "\n\n")

	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  Select the plots to generate. Details for the focused item are shown below.\n") +
		lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
			"  Use ↑/↓ to navigate, Space to toggle, and A/N to select all/none.\n\n"))

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
	cursorLineIdx := 0 // Track which line the cursor is on

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
		if isFocused {
			cursorLineIdx = len(lines)
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)

		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}

		lines = append(lines, listLine{false, checkbox + nameStyle.Render(plot.Name), i})
	}

	// Calculate layout using centralized function
	layout := styles.CalculateListLayout(m.height, cursorLineIdx, len(lines), 14) // header + title + details pane

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
	if len(reqs) > 0 {
		reqStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
		b.WriteString(reqStyle.Render("Needs: "+strings.Join(reqs, ", ")) + "\n")
	}

	readyCount, totalCount, missing := m.plotAvailabilitySummary(plot)
	if totalCount > 0 {
		statusStyle := lipgloss.NewStyle().Foreground(styles.Warning).PaddingLeft(2)
		if readyCount == totalCount {
			statusStyle = statusStyle.Foreground(styles.Success)
		}
		b.WriteString(statusStyle.Render(fmt.Sprintf("Ready for %d/%d selected subjects", readyCount, totalCount)) + "\n")

		if readyCount < totalCount {
			missingParts := []string{}
			if missing["epochs"] > 0 {
				missingParts = append(missingParts, fmt.Sprintf("epochs=%d", missing["epochs"]))
			}
			if missing["features"] > 0 {
				missingParts = append(missingParts, fmt.Sprintf("features=%d", missing["features"]))
			}
			if missing["stats"] > 0 {
				missingParts = append(missingParts, fmt.Sprintf("stats=%d", missing["stats"]))
			}
			if len(missingParts) > 0 {
				reqStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
				b.WriteString(reqStyle.Render("Missing: "+strings.Join(missingParts, ", ")) + "\n")
			}
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
			"  [A] Add range  [D] Delete range  [Space/Enter] Edit range") + "\n\n")

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
	var b strings.Builder
	title := " PLOT SELECTION "
	b.WriteString(styles.SectionTitleStyle.Render(title) + "\n\n")

	// Left Side: List
	var left strings.Builder
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

	statusIndicator := " "
	if count >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	left.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d/%d selected", count, len(visibleItems))) + "\n\n")

	// List of plots
	currentGroup := ""
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	type listLine struct {
		isHeader bool
		text     string
		itemIdx  int
	}
	var lines []listLine
	cursorLineIdx := 0 // Track which line the cursor is on

	for i, plot := range m.plotItems {
		if !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		if plot.Group != currentGroup {
			lines = append(lines, listLine{true, groupStyle.Render(" " + strings.ToUpper(plot.Group)), -1})
			currentGroup = plot.Group
		}
		isSelected := m.plotSelected[i]
		isFocused := i == m.plotCursor
		if isFocused {
			cursorLineIdx = len(lines)
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		lines = append(lines, listLine{false, checkbox + nameStyle.Render(plot.Name), i})
	}

	// Calculate layout using centralized function
	layout := styles.CalculateListLayout(m.height, cursorLineIdx, len(lines), 10)

	// Show scroll up indicator
	if layout.ShowScrollUp {
		left.WriteString(styles.RenderScrollUpIndicator(layout.StartIdx) + "\n")
	}

	for i := layout.StartIdx; i < layout.EndIdx; i++ {
		left.WriteString(lines[i].text + "\n")
	}

	// Show scroll down indicator
	if layout.ShowScrollDn {
		remaining := len(lines) - layout.EndIdx
		left.WriteString(styles.RenderScrollDownIndicator(remaining) + "\n")
	}

	// Right Side: Details
	var right strings.Builder
	plot := m.plotItems[m.plotCursor]

	detailBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(1, 2).
		Width(m.width / 2)

	right.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render("Plot: ") + plot.Name + "\n")

	categoryBadge := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(styles.Secondary).
		Padding(0, 1).
		Render(plot.Group)
	right.WriteString(categoryBadge + "\n\n")

	right.WriteString(lipgloss.NewStyle().Foreground(styles.Text).Render(plot.Description) + "\n\n")

	right.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Data Requirements:\n"))
	if len(plot.RequiredFiles) > 0 {
		for _, file := range plot.RequiredFiles {
			icon := "  ○ "
			fileStyle := lipgloss.NewStyle().Foreground(styles.Text)
			if strings.Contains(file, "features") {
				icon = "  [F] "
			} else if strings.Contains(file, "epochs") {
				icon = "  [E] "
			} else if strings.Contains(file, "stats") {
				icon = "  [S] "
			}
			right.WriteString(icon + fileStyle.Render(file) + "\n")
		}
	} else {
		right.WriteString("  " + lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("Base epochs only") + "\n")
	}

	readyCount, totalCount, missing := m.plotAvailabilitySummary(plot)
	if totalCount > 0 {
		right.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Availability:\n"))

		readyPct := float64(readyCount) / float64(totalCount) * 100
		statusColor := styles.Success
		statusIcon := "✓"
		if readyPct < 100 {
			statusColor = styles.Warning
			statusIcon = "⚠"
		}
		if readyPct < 50 {
			statusColor = styles.Error
			statusIcon = "⚠"
		}

		statusLine := fmt.Sprintf("  %s %d/%d subjects ready (%.0f%%)",
			statusIcon, readyCount, totalCount, readyPct)
		right.WriteString(lipgloss.NewStyle().Foreground(statusColor).Render(statusLine) + "\n")

		if len(missing) > 0 {
			var mLines []string
			for k, v := range missing {
				if v > 0 {
					mLines = append(mLines, fmt.Sprintf("%s: %d missing", k, v))
				}
			}
			if len(mLines) > 0 {
				right.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(
					"  "+strings.Join(mLines, ", ")) + "\n")
			}
		}
	}

	if len(plot.Dependencies) > 0 {
		right.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render("Dependencies:\n"))
		for _, dep := range plot.Dependencies {
			depIcon := "  → "
			right.WriteString(depIcon + lipgloss.NewStyle().Foreground(styles.Accent).Render(dep) + "\n")
		}
	}

	leftView := lipgloss.NewStyle().Width(m.width / 2).Render(left.String())
	rightView := detailBox.Render(right.String())

	return lipgloss.JoinHorizontal(lipgloss.Top, leftView, rightView)
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
		if s.HasEpochs {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("E"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("·"))
		}
		if s.HasFeatures {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("F"))
		} else {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Muted).Render("·"))
		}
		if s.HasStats {
			statusBadges = append(statusBadges, lipgloss.NewStyle().Foreground(styles.Success).Render("S"))
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
		Render("  Legend: [E]=Epochs [F]=Features [S]=Stats"))

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

	if m.Pipeline == types.PipelineML {
		card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("CV Scope:") +
			valueStyle.Render(m.mlScope.CLIValue()) + "\n")
	}

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

	taskLabel := m.task
	if taskLabel == "" {
		taskLabel = "default"
	}
	card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Task:") + valueStyle.Render(taskLabel) + "\n")

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
				configs = append(configs, fmt.Sprintf("conn_win=%.1fs/%.1fs", m.connWindowLen, m.connWindowStep))
			}
			if m.isCategorySelected("directed_connectivity") {
				measures := m.selectedDirectedConnectivityMeasures()
				if len(measures) > 0 {
					configs = append(configs, fmt.Sprintf("dconn=%s", strings.Join(measures, "+")))
				}
			}
			if m.isCategorySelected("pac") {
				methods := []string{"mvl", "kl", "tort", "ozkurt"}
				configs = append(configs, fmt.Sprintf("pac=%s@%.0f-%.0f/%.0f-%.0fHz",
					methods[m.pacMethod], m.pacPhaseMin, m.pacPhaseMax, m.pacAmpMin, m.pacAmpMax))
			}
			if m.isCategorySelected("aperiodic") {
				configs = append(configs, fmt.Sprintf("aper=%.0f-%.0fHz(z=%.1f)", m.aperiodicFmin, m.aperiodicFmax, m.aperiodicPeakZ))
			}
			if m.isCategorySelected("complexity") {
				configs = append(configs, fmt.Sprintf("PE=%d(d=%d)", m.complexityPEOrder, m.complexityPEDelay))
			}
			if m.isCategorySelected("bursts") {
				configs = append(configs, fmt.Sprintf("bursts=%.1fz/%dms", m.burstThresholdZ, m.burstMinDuration))
			}
			if m.isCategorySelected("power") {
				modes := []string{"logratio", "mean", "ratio", "zscore", "zlogratio"}
				configs = append(configs, fmt.Sprintf("power=%s", modes[m.powerBaselineMode]))
			}
			if m.isCategorySelected("erp") {
				configs = append(configs, fmt.Sprintf("erp_baseline=%v", m.erpBaselineCorrection))
			}
			if m.isCategorySelected("source_localization") {
				methods := []string{"lcmv", "eloreta"}
				parcs := []string{"aparc", "aparc.a2009s", "HCPMMP1"}
				configs = append(configs, fmt.Sprintf("source=%s@%s", methods[m.sourceLocMethod], parcs[m.sourceLocParc]))
			}
			configs = append(configs, fmt.Sprintf("min_epochs=%d", m.minEpochsForFeatures))

			if len(configs) > 0 {
				card.WriteString("     " + configStyle.Render(strings.Join(configs, " | ")) + "\n")
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
			if !m.plotSelected[i] || !m.IsPlotCategorySelected(plot.Group) {
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
	} else if m.Pipeline == types.PipelinePreprocessing {
		if m.bidsRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("BIDS Root:") + valueStyle.Render(m.bidsRoot) + "\n")
		}
		if m.derivRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Deriv Root:") + valueStyle.Render(m.derivRoot) + "\n")
		}

		if !m.useDefaultAdvanced {
			var opts []string
			if !m.prepUsePyprep {
				opts = append(opts, "pyprep=off")
			}
			if !m.prepUseIcalabel {
				opts = append(opts, "icalabel=off")
			}
			if m.prepNJobs != 1 {
				opts = append(opts, fmt.Sprintf("n_jobs=%d", m.prepNJobs))
			}
			if m.prepResample != 500 {
				opts = append(opts, fmt.Sprintf("resample=%dHz", m.prepResample))
			}
			if m.prepLFreq != 0.1 || m.prepHFreq != 100.0 {
				opts = append(opts, fmt.Sprintf("filter=%.1f-%.1fHz", m.prepLFreq, m.prepHFreq))
			}
			if m.prepICAMethod != 0 {
				method := []string{"fastica", "infomax", "picard"}[m.prepICAMethod]
				opts = append(opts, fmt.Sprintf("ica=%s", method))
			}
			if m.prepEpochsTmin != -5.0 || m.prepEpochsTmax != 12.0 {
				opts = append(opts, fmt.Sprintf("epochs=%.1f/%.1fs", m.prepEpochsTmin, m.prepEpochsTmax))
			}

			if len(opts) > 0 {
				card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Config:") +
					valueStyle.Render(strings.Join(opts, ", ")) + "\n")
			}
		}
	} else if m.Pipeline == types.PipelineRawToBIDS {
		if m.sourceRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Source Root:") + valueStyle.Render(m.sourceRoot) + "\n")
		}
		if m.bidsRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("BIDS Root:") + valueStyle.Render(m.bidsRoot) + "\n")
		}
		if !m.useDefaultAdvanced {
			var opts []string
			if m.rawMontage != "" {
				opts = append(opts, fmt.Sprintf("montage=%s", m.rawMontage))
			}
			if m.rawLineFreq != 60 {
				opts = append(opts, fmt.Sprintf("line=%dHz", m.rawLineFreq))
			}
			if m.rawOverwrite {
				opts = append(opts, "overwrite")
			}
			if m.rawZeroBaseOnsets {
				opts = append(opts, "zero-base")
			}
			if m.rawTrimToFirstVolume {
				opts = append(opts, "trim")
			}
			if m.rawEventPrefixes != "" {
				opts = append(opts, fmt.Sprintf("prefixes=%s", m.rawEventPrefixes))
			}
			if m.rawKeepAnnotations {
				opts = append(opts, "keep-annotations")
			}
			if len(opts) > 0 {
				card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Config:") +
					valueStyle.Render(strings.Join(opts, ", ")) + "\n")
			}
		}
	} else if m.Pipeline == types.PipelineMergePsychoPyData {
		if m.sourceRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Source Root:") + valueStyle.Render(m.sourceRoot) + "\n")
		}
		if m.bidsRoot != "" {
			card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("BIDS Root:") + valueStyle.Render(m.bidsRoot) + "\n")
		}
		if !m.useDefaultAdvanced {
			var opts []string
			if m.mergeEventPrefixes != "" {
				opts = append(opts, fmt.Sprintf("prefixes=%s", m.mergeEventPrefixes))
			}
			if m.mergeEventTypes != "" {
				opts = append(opts, fmt.Sprintf("types=%s", m.mergeEventTypes))
			}
			if len(opts) > 0 {
				card.WriteString(iconStyle.Render("▸ ") + labelStyle.Render("Config:") +
					valueStyle.Render(strings.Join(opts, ", ")) + "\n")
			}
		}
	}

	b.WriteString(styles.CardStyle.Width(m.width-10).Render(card.String()) + "\n\n")

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
	case types.PipelinePlotting:
		return m.renderPlottingAdvancedConfig()
	case types.PipelineML:
		return m.renderMLAdvancedConfig()
	case types.PipelinePreprocessing:
		return m.renderPreprocessingAdvancedConfig()
	case types.PipelineRawToBIDS:
		return m.renderRawToBidsAdvancedConfig()
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
	aperiodicRangeVal := fmt.Sprintf("%.1f-%.1f Hz", m.aperiodicFmin, m.aperiodicFmax)
	aperiodicPeakZVal := fmt.Sprintf("%.1f", m.aperiodicPeakZ)
	aperiodicR2Val := fmt.Sprintf("%.2f", m.aperiodicMinR2)
	aperiodicPointsVal := fmt.Sprintf("%d", m.aperiodicMinPoints)
	minEpochsVal := fmt.Sprintf("%d", m.minEpochsForFeatures)
	failOnMissingWindowsVal := m.boolToOnOff(m.failOnMissingWindows)
	failOnMissingNamedVal := m.boolToOnOff(m.failOnMissingNamedWindow)

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
				labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
			} else {
				labelStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Width(labelWidth)
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
			label = "  Transform"
			transforms := []string{"none", "CSD", "Laplacian"}
			value = transforms[m.spatialTransform]
			hint = "volume conduction reduction"
		case optSpatialTransformLambda2:
			label = "  Lambda2"
			value = fmt.Sprintf("%.2e", m.spatialTransformLambda2)
			hint = "regularization parameter"
		case optSpatialTransformStiffness:
			label = "  Stiffness"
			value = fmt.Sprintf("%.1f", m.spatialTransformStiffness)
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
			label = "  Freq Min"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMin)
			hint = "min TFR frequency"
		case optTfrFreqMax:
			label = "  Freq Max"
			value = fmt.Sprintf("%.1f Hz", m.tfrFreqMax)
			hint = "max TFR frequency"
		case optTfrNFreqs:
			label = "  N Frequencies"
			value = fmt.Sprintf("%d", m.tfrNFreqs)
			hint = "number of freq bins"
		case optTfrMinCycles:
			label = "  Min Cycles"
			value = fmt.Sprintf("%.1f", m.tfrMinCycles)
			hint = "wavelet cycles floor"
		case optTfrNCyclesFactor:
			label = "  Cycles Factor"
			value = fmt.Sprintf("%.1f", m.tfrNCyclesFactor)
			hint = "cycles per Hz"
		case optTfrDecim:
			label = "  Decimation"
			value = fmt.Sprintf("%d", m.tfrDecim)
			hint = "temporal downsample"
		case optTfrWorkers:
			label = "  Workers"
			value = fmt.Sprintf("%d", m.tfrWorkers)
			hint = "-1 = all cores"
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
		case optFeatGroupValidation:
			label = "▸ Validation"
			hint = "Space to toggle"
			if m.featGroupValidationExpanded {
				label = "▾ Validation"
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
			hint = "model order for DTF/PDC"
		case optDirectedConnNFreqs:
			label = "N Freqs"
			value = fmt.Sprintf("%d", m.directedConnNFreqs)
			hint = "frequency bins"
		case optDirectedConnMinSegSamples:
			label = "Min Seg Samples"
			value = fmt.Sprintf("%d", m.directedConnMinSegSamples)
			hint = "minimum samples per segment"

		// Source Localization (LCMV, eLORETA)
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
			hint = "LCMV regularization"
		case optSourceLocSnr:
			label = "SNR"
			value = fmt.Sprintf("%.1f", m.sourceLocSnr)
			hint = "eLORETA signal-to-noise"
		case optSourceLocLoose:
			label = "Loose"
			value = fmt.Sprintf("%.2f", m.sourceLocLoose)
			hint = "eLORETA loose constraint"
		case optSourceLocDepth:
			label = "Depth"
			value = fmt.Sprintf("%.2f", m.sourceLocDepth)
			hint = "eLORETA depth weighting"
		case optSourceLocConnMethod:
			label = "Connectivity"
			connMethods := []string{"AEC", "wPLI", "PLV"}
			value = connMethods[m.sourceLocConnMethod]
			hint = "source-space connectivity"

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
			hint = "0=none, >0 for z-scores"
		case optPACAllowHarmonicOverlap:
			label = "Allow Harmonics"
			value = m.boolToOnOff(m.pacAllowHarmonicOvrlap)
			hint = "allow harmonic overlap"
		case optPACMaxHarmonic:
			label = "Max Harmonic"
			value = fmt.Sprintf("%d", m.pacMaxHarmonic)
			hint = "upper harmonic to check"
		case optPACHarmonicToleranceHz:
			label = "Harmonic Tol Hz"
			value = fmt.Sprintf("%.1f", m.pacHarmonicToleranceHz)
			hint = "tolerance for overlap check"

		// Aperiodic
		case optAperiodicRange:
			label = "Fit range"
			value = aperiodicRangeVal
			hint = "frequencies to fit"
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

		// Complexity
		case optPEOrder:
			label = "PE Order"
			value = peOrderVal
			hint = "symbol length (3-7)"
		case optPEDelay:
			label = "PE Delay"
			value = peDelayVal
			hint = "sample lag"

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
			label = "Allow missing baseline"
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
			hint = "e.g. theta:beta,alpha:beta"
		case optAsymmetryChannelPairs:
			label = "Channel pairs"
			value = asymPairsVal
			hint = "e.g. F3:F4,C3:C4"

		// Generic / validation
		case optMinEpochs:
			label = "Min Epochs"
			value = minEpochsVal
			hint = "global minimum required"
		case optFailOnMissingWindows:
			label = "Fail missing windows"
			value = failOnMissingWindowsVal
			hint = "require baseline+active windows"
		case optFailOnMissingNamedWindow:
			label = "Fail missing named"
			value = failOnMissingNamedVal
			hint = "require named window per range"

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
		case optBehaviorGroupCorrelations:
			label := "▸ Correlations"
			if m.behaviorGroupCorrelationsExpanded {
				label = "▾ Correlations"
			}
			return label, "", "Space to toggle"
		case optBehaviorGroupConfounds:
			label := "▸ Confounds"
			if m.behaviorGroupConfoundsExpanded {
				label = "▾ Confounds"
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
			hint := "run identifier column"
			if !m.runAdjustmentEnabled {
				hint = hint + " (enable Run Adjustment)"
			} else if len(m.availableColumns) > 0 {
				max := 4
				if len(m.availableColumns) < max {
					max = len(m.availableColumns)
				}
				suffix := ""
				if len(m.availableColumns) > max {
					suffix = fmt.Sprintf(" (+%d)", len(m.availableColumns)-max)
				}
				hint = fmt.Sprintf("%s · available: %s%s", hint, strings.Join(m.availableColumns[:max], " "), suffix)
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
		case optTrialTableOnlyMode:
			return "Trial-Table Only", m.boolToOnOff(m.trialTableOnly), "skip temporal/cluster stages"
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

		// Trial table
		case optTrialTableFormat:
			v := "parquet"
			if m.trialTableFormat == 1 {
				v = "tsv"
			}
			return "Trial Table Format", v, "parquet recommended"
		case optTrialTableIncludeFeatures:
			return "Include Features", m.boolToOnOff(m.trialTableIncludeFeatures), "merge feature columns"
		case optTrialTableIncludeCovars:
			return "Include Covariates", m.boolToOnOff(m.trialTableIncludeCovars), "merge covariates"
		case optTrialTableIncludeEvents:
			return "Include Events", m.boolToOnOff(m.trialTableIncludeEvents), "merge event metadata"
		case optTrialTableAddLagFeatures:
			return "Lag/Delta Columns", m.boolToOnOff(m.trialTableAddLagFeatures), "prev_* and delta_*"
		case optTrialTableExtraEventCols:
			val := m.trialTableExtraEventCols
			if val == "" {
				val = "(none)"
			}
			if m.editingText && m.editingTextField == textFieldTrialTableExtraEventColumns {
				val = textDisplay
			}
			return "Extra Event Columns", val, "comma-separated"
		case optTrialTableHighMissingFrac:
			val := fmt.Sprintf("%.2f", m.trialTableHighMissingFrac)
			if m.editingNumber && m.isCurrentlyEditing(optTrialTableHighMissingFrac) {
				val = numberDisplay
			}
			return "High Missing Frac", val, "warn if column missing too much"

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

		// Confounds
		case optConfoundsAddAsCovariates:
			return "Add QC Covariates", m.boolToOnOff(m.confoundsAddAsCovariates), "FDR-selected covariates"
		case optConfoundsMaxCovariates:
			val := fmt.Sprintf("%d", m.confoundsMaxCovariates)
			if m.editingNumber && m.isCurrentlyEditing(optConfoundsMaxCovariates) {
				val = numberDisplay
			}
			return "Max QC Covariates", val, "limit added columns"
		case optConfoundsQCColumnPatterns:
			val := m.confoundsQCColumnPatterns
			if val == "" {
				val = "(default)"
			}
			if m.editingText && m.editingTextField == textFieldConfoundsQCColumnPatterns {
				val = textDisplay
			}
			return "QC Column Patterns", val, "comma-separated regexes"

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
			} else if m.editingNumber && m.isCurrentlyEditing(optInfluenceCooksThreshold) {
				val = numberDisplay
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
			if len(m.discoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.discoveredColumns))
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
			if len(m.discoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.discoveredColumns))
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
		case optTemporalFilterValue:
			val := m.temporalFilterValue
			if val == "" {
				val = "(all values)"
			}
			if m.editingText && m.editingTextField == textFieldTemporalFilterValue {
				val = textDisplay
			}
			return "Filter Value", val, "compute only for this value"

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
			if len(m.discoveredColumns) > 0 {
				hint = fmt.Sprintf("Space to select · %d columns available", len(m.discoveredColumns))
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
	b.WriteString(styles.SectionTitleStyle.Render(" ADVANCED CONFIGURATION ") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingNumber {
		b.WriteString(infoStyle.Render("  Type a number, then press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Customize preprocessing controls.") + "\n")
		b.WriteString(infoStyle.Render("  Press Space to toggle/edit, Enter to proceed.") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	groupStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Underline(true)

	// Build values for display
	nJobsVal := fmt.Sprintf("%d", m.prepNJobs)
	montageVal := m.prepMontage
	if m.editingText && m.editingTextField == textFieldPrepMontage {
		montageVal = m.textBuffer + "█"
	}
	resampleVal := fmt.Sprintf("%d Hz", m.prepResample)
	lFreqVal := fmt.Sprintf("%.1f Hz", m.prepLFreq)
	hFreqVal := fmt.Sprintf("%.1f Hz", m.prepHFreq)
	notchVal := fmt.Sprintf("%d Hz", m.prepNotch)
	lineFreqVal := fmt.Sprintf("%d Hz", m.prepLineFreq)
	icaMethodVal := []string{"fastica", "infomax", "picard"}[m.prepICAMethod]
	icaCompVal := fmt.Sprintf("%.2f", m.prepICAComp)
	probThreshVal := fmt.Sprintf("%.1f", m.prepProbThresh)
	tminVal := fmt.Sprintf("%.1f s", m.prepEpochsTmin)
	tmaxVal := fmt.Sprintf("%.1f s", m.prepEpochsTmax)
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

	// Input overrides
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
		}
	}

	options := []struct {
		label string
		value string
		hint  string
		group string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization", ""},

		{"Use PyPREP", m.boolToOnOff(m.prepUsePyprep), "Bad channel detection", "Preprocessing Controls"},
		{"Use ICA Label", m.boolToOnOff(m.prepUseIcalabel), "mne-icalabel classification", ""},
		{"N Jobs", nJobsVal, "Parallel jobs for bad channels", ""},
		{"Montage", montageVal, "EEG montage (e.g., easycap-M1)", ""},

		{"Resample", resampleVal, "Resampling frequency", "Filtering"},
		{"L-Freq", lFreqVal, "High-pass filter", ""},
		{"H-Freq", hFreqVal, "Low-pass filter", ""},
		{"Notch", notchVal, "Line noise filter", ""},
		{"Line Freq", lineFreqVal, "EEG line frequency (50/60)", ""},

		{"ICA Method", icaMethodVal, "Algorithm (fastica/infomax/picard)", "ICA Fitting"},
		{"ICA Components", icaCompVal, "Components or variance fraction", ""},
		{"Prob. Threshold", probThreshVal, "Label classification threshold", ""},
		{"Labels to Keep", m.icaLabelsToKeep, "comma-sep: brain,other,eye...", ""},

		{"Epochs Tmin", tminVal, "Start of epoch", "Epoching"},
		{"Epochs Tmax", tmaxVal, "End of epoch", ""},
		{"No Baseline", m.boolToOnOff(m.prepEpochsNoBaseline), "Disable baseline correction", ""},
		{"Baseline Window", baselineVal, "Baseline start to end (s)", ""},
		{"Reject (µV)", rejectVal, "Peak-to-peak threshold", ""},
	}

	for i, opt := range options {
		if opt.group != "" {
			b.WriteString("\n " + groupStyle.Render(opt.group) + "\n")
		}

		isFocused := i == m.advancedCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

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

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
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
		{"Zero Base Onsets", m.boolToOnOff(m.rawZeroBaseOnsets), "Start events at t=0"},
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

func (m Model) renderPlottingAdvancedConfig() string {
	// New exhaustive plotting advanced config renderer (kept behind a runtime
	// condition so the legacy implementation remains reachable for non-plotting
	// pipelines, avoiding "unreachable code" compiler errors).
	if m.Pipeline == types.PipelinePlotting {
		return m.renderPlottingAdvancedConfigV2()
	}

	var b strings.Builder

	accent := m.renderAnimatedAccent()
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		MarginLeft(1)
	b.WriteString(accent + titleStyle.Render(" ADVANCED PLOT SETTINGS") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	if m.useDefaultAdvanced {
		b.WriteString(infoStyle.Render("Default plotting settings will be used.") + "\n")
		b.WriteString(infoStyle.Render("Press Space to customize plot-specific overrides.") + "\n\n")

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
		return b.String()
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

	options := m.getPlottingOptions()

	// Build visible lines, including expanded connectivity measures
	type line struct {
		text    string
		focused bool
	}
	lines := make([]line, 0, len(options)+len(connectivityMeasures))

	labelWidth := 26
	groupStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	triState := func(v *bool) string {
		if v == nil {
			return "default"
		}
		if *v {
			return "ON"
		}
		return "OFF"
	}
	floatOrDefault := func(v float64, fmtStr string) string {
		if v == 0 {
			return "default"
		}
		return fmt.Sprintf(fmtStr, v)
	}
	intOrDefault := func(v int) string {
		if v == 0 {
			return "default"
		}
		return fmt.Sprintf("%d", v)
	}
	spaceListOrDefault := func(v string) string {
		if strings.TrimSpace(v) == "" {
			return "(default)"
		}
		return v
	}

	for i, opt := range options {
		isFocused := m.advancedCursor == i && m.expandedOption < 0
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("▸ ")
		}

		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(labelWidth)
		valueStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
			valueStyle = valueStyle.Foreground(styles.Accent).Bold(true)
		}

		switch opt {
		case optUseDefaults:
			lines = append(lines, line{
				text: cursor + labelStyle.Render("Configuration:") + " " + valueStyle.Render("Custom") + "  " + hintStyle.Render("Space to reset"),
			})

		case optPlotGroupTopomap:
			chev := "▸"
			if m.plotGroupTopomapExpanded {
				chev = "▾"
			}
			lines = append(lines, line{text: cursor + groupStyle.Render(chev+" Topomap")})
		case optPlotGroupTFR:
			chev := "▸"
			if m.plotGroupTFRExpanded {
				chev = "▾"
			}
			lines = append(lines, line{text: cursor + groupStyle.Render(chev+" TFR")})
		case optPlotGroupSizing:
			chev := "▸"
			if m.plotGroupSizingExpanded {
				chev = "▾"
			}
			lines = append(lines, line{text: cursor + groupStyle.Render(chev+" Sizing")})
		case optPlotGroupSelection:
			chev := "▸"
			if m.plotGroupSelectionExpanded {
				chev = "▾"
			}
			lines = append(lines, line{text: cursor + groupStyle.Render(chev+" Selection")})

		case optPlotTopomapContours:
			val := intOrDefault(m.plotTopomapContours)
			if m.isCurrentlyEditing(optPlotTopomapContours) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Contours:") + " " + valueStyle.Render(val)})
		case optPlotTopomapColormap:
			val := m.plotTopomapColormap
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Colormap:") + " " + valueStyle.Render(val) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotTopomapColorbarFraction:
			val := floatOrDefault(m.plotTopomapColorbarFraction, "%.4f")
			if m.isCurrentlyEditing(optPlotTopomapColorbarFraction) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Cbar fraction:") + " " + valueStyle.Render(val)})
		case optPlotTopomapColorbarPad:
			val := floatOrDefault(m.plotTopomapColorbarPad, "%.4f")
			if m.isCurrentlyEditing(optPlotTopomapColorbarPad) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Cbar pad:") + " " + valueStyle.Render(val)})
		case optPlotTopomapDiffAnnotation:
			lines = append(lines, line{text: cursor + labelStyle.Render("Diff annotate:") + " " + valueStyle.Render(triState(m.plotTopomapDiffAnnotation))})
		case optPlotTopomapAnnotateDescriptive:
			lines = append(lines, line{text: cursor + labelStyle.Render("Annotate desc:") + " " + valueStyle.Render(triState(m.plotTopomapAnnotateDesc))})

		case optPlotTFRLogBase:
			val := floatOrDefault(m.plotTFRLogBase, "%.4f")
			if m.isCurrentlyEditing(optPlotTFRLogBase) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Log base:") + " " + valueStyle.Render(val)})
		case optPlotTFRPercentageMultiplier:
			val := floatOrDefault(m.plotTFRPercentageMultiplier, "%.4f")
			if m.isCurrentlyEditing(optPlotTFRPercentageMultiplier) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Pct multiplier:") + " " + valueStyle.Render(val)})

		case optPlotRoiWidthPerBand:
			val := floatOrDefault(m.plotRoiWidthPerBand, "%.3f")
			if m.isCurrentlyEditing(optPlotRoiWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ROI w/band:") + " " + valueStyle.Render(val)})
		case optPlotRoiWidthPerMetric:
			val := floatOrDefault(m.plotRoiWidthPerMetric, "%.3f")
			if m.isCurrentlyEditing(optPlotRoiWidthPerMetric) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ROI w/metric:") + " " + valueStyle.Render(val)})
		case optPlotRoiHeightPerRoi:
			val := floatOrDefault(m.plotRoiHeightPerRoi, "%.3f")
			if m.isCurrentlyEditing(optPlotRoiHeightPerRoi) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ROI h/roi:") + " " + valueStyle.Render(val)})

		case optPlotPowerWidthPerBand:
			val := floatOrDefault(m.plotPowerWidthPerBand, "%.3f")
			if m.isCurrentlyEditing(optPlotPowerWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Power w/band:") + " " + valueStyle.Render(val)})
		case optPlotPowerHeightPerSegment:
			val := floatOrDefault(m.plotPowerHeightPerSegment, "%.3f")
			if m.isCurrentlyEditing(optPlotPowerHeightPerSegment) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Power h/seg:") + " " + valueStyle.Render(val)})

		case optPlotItpcWidthPerBin:
			val := floatOrDefault(m.plotItpcWidthPerBin, "%.3f")
			if m.isCurrentlyEditing(optPlotItpcWidthPerBin) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ITPC w/bin:") + " " + valueStyle.Render(val)})
		case optPlotItpcHeightPerBand:
			val := floatOrDefault(m.plotItpcHeightPerBand, "%.3f")
			if m.isCurrentlyEditing(optPlotItpcHeightPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ITPC h/band:") + " " + valueStyle.Render(val)})
		case optPlotItpcWidthPerBandBox:
			val := floatOrDefault(m.plotItpcWidthPerBandBox, "%.3f")
			if m.isCurrentlyEditing(optPlotItpcWidthPerBandBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ITPC box w:") + " " + valueStyle.Render(val)})
		case optPlotItpcHeightBox:
			val := floatOrDefault(m.plotItpcHeightBox, "%.3f")
			if m.isCurrentlyEditing(optPlotItpcHeightBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("ITPC box h:") + " " + valueStyle.Render(val)})

		case optPlotPacCmap:
			val := m.plotPacCmap
			if strings.TrimSpace(val) == "" {
				val = "(default)"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("PAC cmap:") + " " + valueStyle.Render(val) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotPacWidthPerRoi:
			val := floatOrDefault(m.plotPacWidthPerRoi, "%.3f")
			if m.isCurrentlyEditing(optPlotPacWidthPerRoi) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("PAC w/roi:") + " " + valueStyle.Render(val)})
		case optPlotPacHeightBox:
			val := floatOrDefault(m.plotPacHeightBox, "%.3f")
			if m.isCurrentlyEditing(optPlotPacHeightBox) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("PAC box h:") + " " + valueStyle.Render(val)})

		case optPlotAperiodicWidthPerColumn:
			val := floatOrDefault(m.plotAperiodicWidthPerColumn, "%.3f")
			if m.isCurrentlyEditing(optPlotAperiodicWidthPerColumn) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Aper w/col:") + " " + valueStyle.Render(val)})
		case optPlotAperiodicHeightPerRow:
			val := floatOrDefault(m.plotAperiodicHeightPerRow, "%.3f")
			if m.isCurrentlyEditing(optPlotAperiodicHeightPerRow) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Aper h/row:") + " " + valueStyle.Render(val)})
		case optPlotAperiodicNPerm:
			val := intOrDefault(m.plotAperiodicNPerm)
			if m.isCurrentlyEditing(optPlotAperiodicNPerm) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Aper nperm:") + " " + valueStyle.Render(val)})

		case optPlotQualityWidthPerPlot:
			val := floatOrDefault(m.plotQualityWidthPerPlot, "%.3f")
			if m.isCurrentlyEditing(optPlotQualityWidthPerPlot) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Quality w:") + " " + valueStyle.Render(val)})
		case optPlotQualityHeightPerPlot:
			val := floatOrDefault(m.plotQualityHeightPerPlot, "%.3f")
			if m.isCurrentlyEditing(optPlotQualityHeightPerPlot) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Quality h:") + " " + valueStyle.Render(val)})
		case optPlotQualityDistributionNCols:
			val := intOrDefault(m.plotQualityDistributionNCols)
			if m.isCurrentlyEditing(optPlotQualityDistributionNCols) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Qual dist cols:") + " " + valueStyle.Render(val)})
		case optPlotQualityDistributionMaxFeatures:
			val := intOrDefault(m.plotQualityDistributionMaxFeatures)
			if m.isCurrentlyEditing(optPlotQualityDistributionMaxFeatures) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Qual dist max:") + " " + valueStyle.Render(val)})
		case optPlotQualityOutlierZThreshold:
			val := floatOrDefault(m.plotQualityOutlierZThreshold, "%.3f")
			if m.isCurrentlyEditing(optPlotQualityOutlierZThreshold) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Outlier z:") + " " + valueStyle.Render(val)})
		case optPlotQualityOutlierMaxFeatures:
			val := intOrDefault(m.plotQualityOutlierMaxFeatures)
			if m.isCurrentlyEditing(optPlotQualityOutlierMaxFeatures) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Outlier max f:") + " " + valueStyle.Render(val)})
		case optPlotQualityOutlierMaxTrials:
			val := intOrDefault(m.plotQualityOutlierMaxTrials)
			if m.isCurrentlyEditing(optPlotQualityOutlierMaxTrials) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Outlier max t:") + " " + valueStyle.Render(val)})
		case optPlotQualitySnrThresholdDb:
			val := floatOrDefault(m.plotQualitySnrThresholdDb, "%.3f")
			if m.isCurrentlyEditing(optPlotQualitySnrThresholdDb) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("SNR thr (dB):") + " " + valueStyle.Render(val)})

		case optPlotComplexityWidthPerMeasure:
			val := floatOrDefault(m.plotComplexityWidthPerMeasure, "%.3f")
			if m.isCurrentlyEditing(optPlotComplexityWidthPerMeasure) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Comp w/meas:") + " " + valueStyle.Render(val)})
		case optPlotComplexityHeightPerSegment:
			val := floatOrDefault(m.plotComplexityHeightPerSegment, "%.3f")
			if m.isCurrentlyEditing(optPlotComplexityHeightPerSegment) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Comp h/seg:") + " " + valueStyle.Render(val)})

		case optPlotConnectivityWidthPerCircle:
			val := floatOrDefault(m.plotConnectivityWidthPerCircle, "%.3f")
			if m.isCurrentlyEditing(optPlotConnectivityWidthPerCircle) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Conn w/circle:") + " " + valueStyle.Render(val)})
		case optPlotConnectivityWidthPerBand:
			val := floatOrDefault(m.plotConnectivityWidthPerBand, "%.3f")
			if m.isCurrentlyEditing(optPlotConnectivityWidthPerBand) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Conn w/band:") + " " + valueStyle.Render(val)})
		case optPlotConnectivityHeightPerMeasure:
			val := floatOrDefault(m.plotConnectivityHeightPerMeasure, "%.3f")
			if m.isCurrentlyEditing(optPlotConnectivityHeightPerMeasure) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Conn h/meas:") + " " + valueStyle.Render(val)})
		case optPlotConnectivityCircleTopFraction:
			val := floatOrDefault(m.plotConnectivityCircleTopFraction, "%.3f")
			if m.isCurrentlyEditing(optPlotConnectivityCircleTopFraction) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Circle top frac:") + " " + valueStyle.Render(val)})
		case optPlotConnectivityCircleMinLines:
			val := intOrDefault(m.plotConnectivityCircleMinLines)
			if m.isCurrentlyEditing(optPlotConnectivityCircleMinLines) {
				val = m.numberBuffer + "█"
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Circle min lines:") + " " + valueStyle.Render(val)})

		case optPlotPacPairs:
			lines = append(lines, line{text: cursor + labelStyle.Render("PAC pairs:") + " " + valueStyle.Render(spaceListOrDefault(m.plotPacPairsSpec)) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotConnectivityMeasures:
			val := "(default)"
			selected := m.selectedConnectivityMeasures()
			if len(selected) > 0 {
				val = strings.Join(selected, " ")
			}
			lines = append(lines, line{text: cursor + labelStyle.Render("Conn measures:") + " " + valueStyle.Render(val) + "  " + hintStyle.Render("Space to expand")})
			if m.expandedOption == expandedConnectivityMeasures && m.advancedCursor == i {
				for j, measure := range connectivityMeasures {
					subFocused := m.subCursor == j
					subCursor := "    "
					subLabel := "  "
					if subFocused {
						subCursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("  ▸ ")
						subLabel = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("  ")
					}
					on := m.connectivityMeasures[j]
					toggle := "OFF"
					if on {
						toggle = "ON"
					}
					lines = append(lines, line{
						text: subCursor + subLabel + measure.Key + ": " + lipgloss.NewStyle().Foreground(styles.Accent).Render(toggle),
					})
				}
			}
		case optPlotSpectralMetrics:
			lines = append(lines, line{text: cursor + labelStyle.Render("Spectral metrics:") + " " + valueStyle.Render(spaceListOrDefault(m.plotSpectralMetricsSpec)) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotBurstsMetrics:
			lines = append(lines, line{text: cursor + labelStyle.Render("Bursts metrics:") + " " + valueStyle.Render(spaceListOrDefault(m.plotBurstsMetricsSpec)) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotAsymmetryStat:
			lines = append(lines, line{text: cursor + labelStyle.Render("Asym stat:") + " " + valueStyle.Render(spaceListOrDefault(m.plotAsymmetryStatSpec)) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotTemporalTimeBins:
			lines = append(lines, line{text: cursor + labelStyle.Render("Temporal bins:") + " " + valueStyle.Render(spaceListOrDefault(m.plotTemporalTimeBinsSpec)) + "  " + hintStyle.Render("Enter to edit")})
		case optPlotTemporalTimeLabels:
			lines = append(lines, line{text: cursor + labelStyle.Render("Temporal labels:") + " " + valueStyle.Render(spaceListOrDefault(m.plotTemporalTimeLabelsSpec)) + "  " + hintStyle.Render("Enter to edit")})
		}
	}

	effectiveHeight := m.height
	if effectiveHeight <= 0 {
		effectiveHeight = defaultTerminalHeight
	}
	start, end, _ := calculateScrollWindow(
		len(lines), m.advancedOffset, effectiveHeight, plotConfigOverhead)

	if start > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ... %d more above ...", start)) + "\n")
	}
	for i := start; i < end; i++ {
		b.WriteString(lines[i].text + "\n")
	}
	if end < len(lines) {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(fmt.Sprintf("  ... %d more below ...", len(lines)-end)) + "\n")
	}

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
