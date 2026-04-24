package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

const (
	reviewPanelMinWidth = 110
	reviewPanelGap      = 2
	reviewPanelWidth    = 48
	reviewLabelWidth    = 11
)

type reviewRow struct {
	label  string
	value  string
	accent bool
}

func (m Model) usesReviewPanel(width int) bool {
	return width >= reviewPanelMinWidth
}

func (m Model) reviewColumnWidths(width int) (int, int) {
	reviewWidth := reviewPanelWidth
	stepWidth := width - reviewWidth - reviewPanelGap
	if stepWidth < minContainerWidth {
		stepWidth = minContainerWidth
		reviewWidth = width - stepWidth - reviewPanelGap
	}
	if reviewWidth < 1 {
		reviewWidth = 1
	}
	return stepWidth, reviewWidth
}

func (m Model) renderReviewPanel(width, height int) string {
	panelStyle := styles.PanelStyle
	// Lip Gloss border width is added outside Width(), so render two cells
	// narrower to keep the framed panel at the requested outer width.
	renderWidth := width - 2
	if renderWidth < 1 {
		renderWidth = 1
	}
	innerWidth := renderWidth - panelStyle.GetHorizontalFrameSize()
	if innerWidth < 1 {
		innerWidth = 1
	}
	innerHeight := height - panelStyle.GetVerticalFrameSize()
	if innerHeight < 1 {
		innerHeight = 1
	}

	body := normalizeContentFrame(m.renderReviewBody(innerWidth, innerHeight), innerWidth, innerHeight)
	return styles.RenderNoWrapBlock(panelStyle.Height(innerHeight), body, renderWidth)
}

func (m Model) renderReviewBody(width, height int) string {
	if height < 12 {
		return m.renderCompactReviewBody(width)
	}

	var b strings.Builder

	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Review", width))
	for _, row := range m.priorityReviewRows(height) {
		b.WriteString("\n")
		b.WriteString(m.renderReviewRow(row, width))
	}

	if rows := m.reviewSelectionRows(); len(rows) > 0 && height >= 20 {
		b.WriteString("\n\n")
		b.WriteString(styles.RenderPreviewSubHeaderWithRule("Selections", width))
		for _, row := range rows {
			b.WriteString("\n")
			b.WriteString(m.renderReviewRow(row, width))
		}
	}

	if validation := m.reviewValidationRows(width); validation != "" {
		b.WriteString("\n\n")
		b.WriteString(validation)
	}

	b.WriteString("\n\n")
	b.WriteString(m.renderNextAction(width))

	b.WriteString("\n\n")
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Command", width))
	for _, line := range m.commandPreviewLines(width, commandPreviewLineBudget(height)) {
		b.WriteString("\n")
		b.WriteString(line)
	}

	return strings.TrimRight(b.String(), "\n")
}

func (m Model) renderCompactReviewBody(width int) string {
	var b strings.Builder

	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Review", width))
	for _, row := range m.compactReviewRows() {
		b.WriteString("\n")
		b.WriteString(m.renderReviewRow(row, width))
	}

	b.WriteString("\n")
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Validation", width))
	errors := m.reviewValidationErrors()
	if len(errors) == 0 {
		b.WriteString("\n")
		b.WriteString(styles.TruncateLine(lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark+" current step ready"), width))
	} else {
		b.WriteString("\n")
		b.WriteString(styles.TruncateLine(lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render(styles.WarningMark+" "+errors[0]), width))
	}

	b.WriteString("\n")
	b.WriteString(m.renderCompactNextAction(width))
	b.WriteString("\n")
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Command", width))
	for _, line := range m.commandPreviewLines(width, 1) {
		b.WriteString("\n")
		b.WriteString(line)
	}

	return b.String()
}

func (m Model) compactReviewRows() []reviewRow {
	rows := m.reviewOverviewRows()
	return []reviewRow{rows[0], rows[3], rows[4], rows[5]}
}

func (m Model) reviewOverviewRows() []reviewRow {
	return []reviewRow{
		{label: "Pipeline", value: m.Pipeline.String(), accent: true},
		{label: "Step", value: m.currentStepLabel()},
		{label: "Mode", value: m.currentModeLabel()},
		{label: "Subjects", value: m.subjectReviewLabel(), accent: countSelectedStringItems(m.subjectSelected) > 0},
		{label: "Task", value: m.taskReviewLabel(), accent: strings.TrimSpace(m.task) != ""},
		{label: "Paths", value: m.pathReviewLabel(), accent: m.requiredPathsConfigured()},
	}
}

func (m Model) priorityReviewRows(height int) []reviewRow {
	rows := m.reviewOverviewRows()
	if height >= 14 {
		return rows
	}
	if height >= 12 {
		return []reviewRow{rows[0], rows[1], rows[3], rows[4], rows[5]}
	}
	if len(rows) > 4 {
		return rows[:4]
	}
	return rows
}

func (m Model) reviewSelectionRows() []reviewRow {
	switch m.Pipeline {
	case types.PipelineBehavior:
		return []reviewRow{
			{label: "Analyses", value: countLabel(countSelectedItems(m.computationSelected), len(m.computations))},
			{label: "Files", value: countLabel(countSelectedStringItems(m.featureFileSelected), len(m.featureFiles))},
			{label: "Bands", value: countLabel(countSelectedItems(m.bandSelected), len(m.bands))},
		}
	case types.PipelineFeatures:
		return []reviewRow{
			{label: "Features", value: countLabel(countSelectedItems(m.selected), len(m.categories))},
			{label: "Bands", value: countLabel(countSelectedItems(m.bandSelected), len(m.bands))},
			{label: "ROIs", value: countLabel(countSelectedItems(m.roiSelected), len(m.rois))},
			{label: "Spatial", value: countLabel(countSelectedItems(m.spatialSelected), len(spatialModes))},
			{label: "Windows", value: fmt.Sprintf("%d", len(m.TimeRanges))},
		}
	case types.PipelinePreprocessing:
		return []reviewRow{
			{label: "Stages", value: countLabel(countSelectedItems(m.prepStageSelected), len(m.prepStages))},
		}
	case types.PipelinePlotting:
		return []reviewRow{
			{label: "Categories", value: countLabel(countSelectedItems(m.selected), len(m.categories))},
			{label: "Plots", value: countLabel(m.countSelectedVisiblePlots(), len(m.plotItems))},
			{label: "Formats", value: countLabel(countSelectedStringItems(m.plotFormatSelected), len(m.plotFormats))},
		}
	default:
		return nil
	}
}

func (m Model) renderReviewRow(row reviewRow, width int) string {
	line := styles.RenderKeyValue(row.label, row.value, reviewLabelWidth)
	if row.accent {
		line = styles.RenderKeyValueAccent(row.label, row.value, reviewLabelWidth)
	}
	return styles.TruncateLine(line, width)
}

func (m Model) reviewValidationRows(width int) string {
	errors := m.reviewValidationErrors()
	if len(errors) == 0 {
		header := styles.RenderPreviewSubHeaderWithRule("Validation", width)
		ok := lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " current step ready")
		return header + "\n" + styles.TruncateLine(ok, width)
	}

	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Validation", width))
	for i, err := range errors {
		if i >= 2 {
			remaining := len(errors) - i
			b.WriteString("\n")
			b.WriteString(styles.TruncateLine(
				lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("+%d more", remaining)),
				width,
			))
			break
		}
		b.WriteString("\n")
		b.WriteString(styles.TruncateLine(
			lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render(styles.WarningMark+" "+err),
			width,
		))
	}
	return b.String()
}

func (m Model) renderNextAction(width int) string {
	action := m.nextActionLabel()
	header := styles.RenderPreviewSubHeaderWithRule("Next", width)
	line := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(action)
	return header + "\n" + styles.TruncateLine(line, width)
}

func (m Model) renderCompactNextAction(width int) string {
	label := lipgloss.NewStyle().Foreground(styles.Muted).Bold(true).Render("NEXT")
	value := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(" " + m.nextActionLabel())
	return styles.TruncateLine(label+value, width)
}

func (m Model) nextActionLabel() string {
	errors := m.reviewValidationErrors()
	if len(errors) > 0 {
		return "Fix: " + errors[0]
	}
	if m.stepIndex >= len(m.steps)-1 {
		return "Enter to run"
	}
	return "Enter to continue"
}

func (m Model) reviewValidationErrors() []string {
	if len(m.validationErrors) > 0 {
		return append([]string(nil), m.validationErrors...)
	}
	copyModel := m
	return copyModel.validateCurrentStep()
}

func (m Model) currentStepLabel() string {
	if m.stepIndex >= 0 && m.stepIndex < len(m.steps) {
		name := stepDisplayNames[m.steps[m.stepIndex]]
		if name != "" {
			return fmt.Sprintf("%d/%d %s", m.stepIndex+1, len(m.steps), name)
		}
	}
	return m.CurrentStep.String()
}

func (m Model) currentModeLabel() string {
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		return m.modeOptions[m.modeIndex]
	}
	return "not set"
}

func (m Model) subjectReviewLabel() string {
	selected := countSelectedStringItems(m.subjectSelected)
	if selected == 0 {
		return "none selected"
	}

	valid := 0
	for _, subject := range m.subjects {
		if !m.subjectSelected[subject.ID] {
			continue
		}
		isValid, _ := m.Pipeline.ValidateSubject(subject)
		if m.Pipeline == types.PipelinePlotting {
			isValid, _ = m.validatePlottingSubject(subject)
		}
		if isValid {
			valid++
		}
	}

	if len(m.subjects) == 0 {
		return fmt.Sprintf("%d selected", selected)
	}
	return fmt.Sprintf("%d selected, %d valid", selected, valid)
}

func (m Model) taskReviewLabel() string {
	task := strings.TrimSpace(m.task)
	if task == "" {
		return "not configured"
	}
	return task
}

func (m Model) pathReviewLabel() string {
	missing := m.missingRequiredPathLabels()
	if len(missing) == 0 {
		return "configured"
	}
	return "missing " + strings.Join(missing, ", ")
}

func (m Model) requiredPathsConfigured() bool {
	return len(m.missingRequiredPathLabels()) == 0
}

func (m Model) missingRequiredPathLabels() []string {
	var missing []string
	if m.Pipeline == types.PipelineFmri || m.Pipeline == types.PipelineFmriAnalysis {
		if strings.TrimSpace(m.bidsFmriRoot) == "" {
			missing = append(missing, "fMRI")
		}
	} else if strings.TrimSpace(m.bidsRoot) == "" {
		missing = append(missing, "BIDS")
	}
	if strings.TrimSpace(m.derivRoot) == "" {
		missing = append(missing, "deriv")
	}
	return missing
}

func countLabel(selected, total int) string {
	if total <= 0 {
		return "0"
	}
	return fmt.Sprintf("%d/%d", selected, total)
}

func commandPreviewLineBudget(height int) int {
	if height >= 20 {
		return 3
	}
	if height >= 14 {
		return 2
	}
	return 1
}

func (m Model) commandPreviewLines(width, maxLines int) []string {
	if width <= 0 || maxLines <= 0 {
		return nil
	}

	command := strings.TrimSpace(m.BuildCommand())
	if command == "" {
		return nil
	}

	words := strings.Fields(command)
	lines := make([]string, 0, maxLines)
	current := ""
	for _, word := range words {
		candidate := word
		if current != "" {
			candidate = current + " " + word
		}
		if lipgloss.Width(candidate) <= width {
			current = candidate
			continue
		}
		if current == "" {
			current = styles.TruncateLine(word, width)
		}
		lines = append(lines, current)
		current = word
		if len(lines) == maxLines {
			break
		}
	}

	if len(lines) < maxLines && current != "" {
		lines = append(lines, current)
	}
	lines = markClippedCommand(lines, words, width)

	style := lipgloss.NewStyle().Foreground(styles.TextDim)
	for i, line := range lines {
		lines[i] = style.Render(styles.TruncateLine(line, width))
	}
	return lines
}

func markClippedCommand(lines, words []string, width int) []string {
	if len(lines) == 0 {
		return lines
	}
	renderedWords := strings.Fields(strings.Join(lines, " "))
	if len(renderedWords) >= len(words) {
		return lines
	}
	last := len(lines) - 1
	lines[last] = styles.TruncateLine(lines[last]+" ...", width)
	return lines
}
