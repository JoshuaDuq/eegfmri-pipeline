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

type rowStatus int

const (
	rowInfo rowStatus = iota
	rowOk
	rowWarn
)

type reviewRow struct {
	label  string
	value  string
	status rowStatus
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
	if height < 8 {
		return m.renderUltraCompactReviewBody(width)
	}

	sep := "\n\n"
	if height < 12 {
		sep = "\n"
	}

	var parts []string
	parts = append(parts, styles.RenderPreviewSubHeaderWithRule("Review", width))
	parts = append(parts, m.renderReviewStatusBanner(width, height >= 12))

	if rows := m.renderReviewRows(m.priorityReviewRows(height), width); rows != "" {
		parts = append(parts, rows)
	}

	if height >= 18 {
		if rows := m.renderReviewRows(m.reviewSelectionRows(), width); rows != "" {
			parts = append(parts, rows)
		}
	}

	parts = append(parts, m.renderReviewCommandBlock(width, height))
	return strings.Join(parts, sep)
}

func (m Model) renderUltraCompactReviewBody(width int) string {
	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Review", width))
	b.WriteString("\n")
	b.WriteString(m.renderReviewStatusBanner(width, false))
	b.WriteString("\n")
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("Command", width))
	for _, line := range m.commandPreviewLines(width, 1) {
		b.WriteString("\n")
		b.WriteString(line)
	}
	return b.String()
}

func (m Model) renderReviewStatusBanner(width int, includeNext bool) string {
	errors := m.reviewValidationErrors()
	var lines []string

	if len(errors) > 0 {
		labelStyle := lipgloss.NewStyle().Foreground(styles.Warning).Bold(true)
		valueStyle := lipgloss.NewStyle().Foreground(styles.Warning)
		lines = append(lines, formatBannerLines("VALIDATION", errors[0], labelStyle, valueStyle, width)...)
		// Show up to one additional concrete error inline; collapse the rest
		// into a quiet count so the banner stays scannable.
		continuationStyle := lipgloss.NewStyle().Foreground(styles.Warning)
		if len(errors) > 1 {
			lines = append(lines, styles.TruncateLine("  "+continuationStyle.Render(errors[1]), width))
		}
		if len(errors) > 2 {
			noun := "issues"
			if len(errors)-2 == 1 {
				noun = "issue"
			}
			more := lipgloss.NewStyle().Foreground(styles.Muted).Render(
				fmt.Sprintf("  +%d more %s", len(errors)-2, noun))
			lines = append(lines, styles.TruncateLine(more, width))
		}
		// When blocked, omit the NEXT row entirely. The user already sees the
		// concrete issues above; a duplicated "Fix: <error>" line just adds
		// noise and crowds the panel.
		return strings.Join(lines, "\n")
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true)
	valueStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	lines = append(lines, formatBannerLines("VALIDATION", "current step ready", labelStyle, valueStyle, width)...)

	if includeNext {
		nextLabelStyle := lipgloss.NewStyle().Foreground(styles.Muted).Bold(true)
		nextValueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		lines = append(lines, formatBannerLines("NEXT", m.nextActionLabel(), nextLabelStyle, nextValueStyle, width)...)
	}

	return strings.Join(lines, "\n")
}

// formatBannerLines emits a single inline line when the full "<label> <value>"
// fits the width budget; otherwise splits onto two lines so long messages
// remain fully readable rather than getting ellipsized. The label is padded
// to reviewLabelWidth so banner labels align with the configuration rows
// below them, and meaning is conveyed by color rather than glyphs.
func formatBannerLines(label, value string, labelStyle, valueStyle lipgloss.Style, width int) []string {
	paddedLabel := labelStyle.Width(reviewLabelWidth).Render(label)
	renderedValue := valueStyle.Render(value)
	inline := "  " + paddedLabel + renderedValue
	if lipgloss.Width(inline) <= width {
		return []string{inline}
	}
	valueIndent := strings.Repeat(" ", 2+reviewLabelWidth)
	lines := []string{styles.TruncateLine("  " + labelStyle.Render(label), width)}
	lines = append(lines, wrapBannerValueLines(valueIndent, renderedValue, width)...)
	return lines
}

func wrapBannerValueLines(indent, renderedValue string, width int) []string {
	available := width - lipgloss.Width(indent)
	if available < 1 {
		return []string{styles.TruncateLine(indent+renderedValue, width)}
	}
	words := strings.Fields(renderedValue)
	if len(words) == 0 {
		return []string{styles.TruncateLine(indent, width)}
	}
	lines := make([]string, 0, 2)
	current := ""
	for _, word := range words {
		candidate := word
		if current != "" {
			candidate = current + " " + word
		}
		if lipgloss.Width(candidate) <= available {
			current = candidate
			continue
		}
		if current == "" {
			lines = append(lines, styles.TruncateLine(indent+word, width))
			continue
		}
		lines = append(lines, styles.TruncateLine(indent+current, width))
		current = word
	}
	if current != "" {
		lines = append(lines, styles.TruncateLine(indent+current, width))
	}
	return lines
}

func (m Model) renderReviewRows(rows []reviewRow, width int) string {
	if len(rows) == 0 {
		return ""
	}
	var b strings.Builder
	for i, row := range rows {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(m.renderReviewRow(row, width))
	}
	return b.String()
}

func (m Model) renderReviewCommandBlock(width, height int) string {
	budget := commandPreviewLineBudget(height)
	all := m.commandPreviewAllLines(width)
	offset, _, _ := clampCommandScroll(m.cmdScrollOffset, len(all), budget)

	var b strings.Builder
	b.WriteString(m.renderCommandPanelHeader(width, offset, len(all), budget))
	style := lipgloss.NewStyle().Foreground(styles.TextDim)
	end := offset + budget
	if end > len(all) {
		end = len(all)
	}
	for i := offset; i < end; i++ {
		b.WriteString("\n")
		b.WriteString(style.Render(all[i]))
	}
	return b.String()
}

// isMouseOverReviewPanel reports whether the absolute mouse X coordinate from
// a tea.MouseMsg falls within the review panel column. Used to route wheel
// events over the panel to command-preview scrolling instead of moving the
// list cursor in the main step content. Returns false whenever the panel is
// not visible (terminal too narrow).
func (m Model) isMouseOverReviewPanel(x int) bool {
	w, h := m.effectiveDimensions()
	containerW, _ := m.containerDimensions(w, h)
	innerW := containerW - containerPadH*2 - containerBorder
	if !m.usesReviewPanel(innerW) {
		return false
	}
	stepWidth, reviewWidth := m.reviewColumnWidths(innerW)
	containerLeft := (w - containerW) / 2
	if containerLeft < 0 {
		containerLeft = 0
	}
	panelLeft := containerLeft + containerBorder/2 + containerPadH + stepWidth + reviewPanelGap
	panelRight := panelLeft + reviewWidth
	return x >= panelLeft && x < panelRight
}

// scrollCommandPreview adjusts the review panel command preview offset by
// `delta` wrapped lines. Negative scrolls toward the top; positive toward the
// bottom. The stored offset is clamped against the live maximum so that
// hammering `]` past the end does not pile up an unreachable value that
// would otherwise require an equal number of `[` presses to "pay back"
// before the visible viewport actually moves.
func (m *Model) scrollCommandPreview(delta int) {
	maxOffset := m.commandPreviewMaxOffset()
	if m.cmdScrollOffset > maxOffset {
		m.cmdScrollOffset = maxOffset
	}
	m.cmdScrollOffset += delta
	if m.cmdScrollOffset < 0 {
		m.cmdScrollOffset = 0
	}
	if m.cmdScrollOffset > maxOffset {
		m.cmdScrollOffset = maxOffset
	}
}

// commandPreviewMaxOffset returns the largest valid scroll offset for the
// command preview given the current viewport dimensions, or 0 when the
// review panel is not visible. Mirrors the width/height math performed by
// renderReviewPanel so the clamp stays consistent with what the render path
// is actually displaying.
func (m Model) commandPreviewMaxOffset() int {
	if !m.usesReviewPanel(m.contentWidth) {
		return 0
	}
	_, reviewWidth := m.reviewColumnWidths(m.contentWidth)
	renderWidth := reviewWidth - 2
	if renderWidth < 1 {
		return 0
	}
	panelInnerWidth := renderWidth - styles.PanelStyle.GetHorizontalFrameSize()
	if panelInnerWidth < 1 {
		return 0
	}
	total := len(m.commandPreviewAllLines(panelInnerWidth))

	mainH := m.availableMainContentHeight()
	panelInnerHeight := mainH - styles.PanelStyle.GetVerticalFrameSize()
	if panelInnerHeight < 1 {
		panelInnerHeight = 1
	}
	budget := commandPreviewLineBudget(panelInnerHeight)

	maxOffset := total - budget
	if maxOffset < 0 {
		maxOffset = 0
	}
	return maxOffset
}

// clampCommandScroll normalizes the requested scroll offset to a valid range
// for the available command preview budget and returns the resolved offset
// plus the count of hidden lines above and below the viewport.
func clampCommandScroll(requested, total, budget int) (offset, above, below int) {
	if budget < 1 {
		budget = 1
	}
	maxOffset := total - budget
	if maxOffset < 0 {
		maxOffset = 0
	}
	offset = requested
	if offset < 0 {
		offset = 0
	}
	if offset > maxOffset {
		offset = maxOffset
	}
	above = offset
	below = total - offset - budget
	if below < 0 {
		below = 0
	}
	return
}

// renderCommandPanelHeader produces the COMMAND header but swaps in a
// transient label when the clipboard copy toast is active so the user
// sees the feedback exactly where the action originated, without
// consuming an extra row in a tight panel.
func (m Model) renderCommandPanelHeader(width, offset, total, budget int) string {
	label := "COMMAND"
	labelStyle := lipgloss.NewStyle().Foreground(styles.Muted).Bold(true)
	switch m.toastType {
	case "clipboard":
		label = "COPIED  command on clipboard"
		labelStyle = lipgloss.NewStyle().Foreground(styles.Success).Bold(true)
	case "clipboard-error":
		label = "COPY FAILED  " + strings.TrimPrefix(m.toastMessage, "Copy failed: ")
		labelStyle = lipgloss.NewStyle().Foreground(styles.Warning).Bold(true)
	}
	return renderCommandHeader(width, offset, total, budget, label, labelStyle)
}

// renderCommandHeader emits the COMMAND sub-header and, when the preview
// is scrollable, a quiet "line A-B/N \u00b7 [ ]" readout on the right side of
// the rule so the user can see their position in the wrapped command and
// discover the keys used to scroll. The label and its style are injected so
// the caller can substitute transient feedback (e.g. clipboard confirm).
func renderCommandHeader(width, offset, total, budget int, labelText string, labelStyle lipgloss.Style) string {
	label := labelStyle.Render(labelText)
	ruleStyle := lipgloss.NewStyle().Foreground(styles.Border)

	if total <= budget {
		// No scroll readout to render; just emit "<LABEL>  <hairline rule>".
		ruleWidth := width - lipgloss.Width(label) - 2
		if ruleWidth < 1 {
			return styles.TruncateLine(label, width)
		}
		rule := ruleStyle.Render("  " + strings.Repeat("\u2500", ruleWidth))
		return label + rule
	}

	first := offset + 1
	last := offset + budget
	if last > total {
		last = total
	}

	position := lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d-%d/%d lines", first, last, total))
	hint := lipgloss.NewStyle().Foreground(styles.Muted).Render("[ ]")
	sep := lipgloss.NewStyle().Foreground(styles.Border).Render(" \u00b7 ")
	indicator := position + sep + hint

	consumed := lipgloss.Width(label) + lipgloss.Width(indicator) + 4
	ruleWidth := width - consumed
	if ruleWidth < 1 {
		// Indicator alone won't fit; fall back to the plain labelled header
		// rather than pushing the title off-screen.
		ruleWidth = width - lipgloss.Width(label) - 2
		if ruleWidth < 1 {
			return styles.TruncateLine(label, width)
		}
		return label + ruleStyle.Render("  "+strings.Repeat("\u2500", ruleWidth))
	}
	rule := ruleStyle.Render("  " + strings.Repeat("\u2500", ruleWidth) + "  ")
	return label + rule + indicator
}

// commandPreviewAllLines returns the full word-wrapped command preview with
// no truncation. Used by the review panel to support scrolling through long
// command lines.
func (m Model) commandPreviewAllLines(width int) []string {
	if width <= 0 {
		return nil
	}
	command := strings.TrimSpace(m.BuildCommand())
	if command == "" {
		return nil
	}
	words := strings.Fields(command)
	var lines []string
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
		if current != "" {
			lines = append(lines, current)
			current = word
		} else {
			// Single word longer than the available width; truncate so it
			// still fits rather than overflowing the panel.
			lines = append(lines, styles.TruncateLine(word, width))
			current = ""
		}
	}
	if current != "" {
		lines = append(lines, current)
	}
	return lines
}

func (m Model) reviewOverviewRows() []reviewRow {
	subjectStatus := rowWarn
	if countSelectedStringItems(m.subjectSelected) > 0 {
		subjectStatus = rowOk
	} else if m.subjectsLoading {
		// Don't shout "none selected" while discovery is still running; the
		// user has not had a chance to choose anything yet.
		subjectStatus = rowInfo
	}
	taskStatus := rowWarn
	if strings.TrimSpace(m.task) != "" {
		taskStatus = rowOk
	}
	pathStatus := rowWarn
	if m.requiredPathsConfigured() {
		pathStatus = rowOk
	}

	return []reviewRow{
		{label: "Pipeline", value: m.Pipeline.String(), accent: true},
		{label: "Mode", value: m.currentModeLabel()},
		{label: "Subjects", value: m.subjectReviewLabel(), status: subjectStatus},
		{label: "Task", value: m.taskReviewLabel(), status: taskStatus},
		{label: "Paths", value: m.pathReviewLabel(), status: pathStatus},
	}
}

func (m Model) priorityReviewRows(height int) []reviewRow {
	rows := m.reviewOverviewRows()
	// rows: 0=Pipeline, 1=Mode, 2=Subjects, 3=Task, 4=Paths
	if height >= 14 {
		return rows
	}
	// Drop Mode (informational) when space is tight; keep readiness rows.
	return []reviewRow{rows[0], rows[2], rows[3], rows[4]}
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
	var lblStyle, valStyle lipgloss.Style
	switch row.status {
	case rowWarn:
		lblStyle = lipgloss.NewStyle().Foreground(styles.Warning).Bold(true)
		valStyle = lipgloss.NewStyle().Foreground(styles.Warning)
	default:
		lblStyle = lipgloss.NewStyle().Foreground(styles.TextDim)
		valStyle = lipgloss.NewStyle().Foreground(styles.Text)
	}
	if row.accent {
		valStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	}
	lbl := lblStyle.Width(reviewLabelWidth).Render(row.label)
	val := valStyle.Render(row.value)
	return styles.TruncateLine("  "+lbl+val, width)
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

func (m Model) currentModeLabel() string {
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		return titleCaseFirst(m.modeOptions[m.modeIndex])
	}
	return "not set"
}

// titleCaseFirst upper-cases the first rune of s without altering the rest.
// Used so display values like "Compute" align with adjacent label values
// ("Features", "Behavior") that come from already-capitalised enum strings.
func titleCaseFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	first := runes[0]
	if first >= 'a' && first <= 'z' {
		runes[0] = first - ('a' - 'A')
	}
	return string(runes)
}

func (m Model) subjectReviewLabel() string {
	selected := countSelectedStringItems(m.subjectSelected)
	if selected == 0 {
		if m.subjectsLoading {
			return "loading\u2026"
		}
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
