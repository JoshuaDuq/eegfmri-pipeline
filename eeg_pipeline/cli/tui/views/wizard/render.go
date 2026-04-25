package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

// footerHint is the wizard-internal hint representation; delegates to the
// shared renderer via toStylesHints + styles.RenderFooterHints.
type footerHint struct {
	key      string
	label    string
	compact  string
	priority int
}

func toStylesHints(hints []footerHint) []styles.FooterHint {
	out := make([]styles.FooterHint, len(hints))
	for i, h := range hints {
		out[i] = styles.FooterHint{Key: h.key, Label: h.label, Compact: h.compact, Priority: h.priority}
	}
	return out
}

const (
	shortHeightThreshold = 25
	minMainContentHeight = 10
	// headerSpacingLines and footerSpacingLines are added to newline counts so
	// mainContentHeight matches the real View() assembly (extra newlines between
	// header ↔ main and main ↔ footer are budgeted here).
	headerSpacingLines    = 5
	footerSpacingLines    = 3
	containerPadH         = 4
	containerPadV         = 3
	containerBorder       = 2
	minContainerWidth     = 60
	minContainerHeight    = 15
	containerWidthPercent = 92
	defaultWidth          = 120
	defaultHeight         = 40
)

var stepDisplayNames = map[types.WizardStep]string{
	types.StepSelectMode:                "Mode",
	types.StepSelectComputations:        "Analyses",
	types.StepSelectFeatureFiles:        "Files",
	types.StepConfigureOptions:          "Features",
	types.StepSelectBands:               "Bands",
	types.StepSelectROIs:                "ROIs",
	types.StepSelectSpatial:             "Spatial",
	types.StepTimeRange:                 "Time",
	types.StepAdvancedConfig:            "Advanced",
	types.StepSelectPlots:               "Plots",
	types.StepSelectFeaturePlotters:     "Feature Plots",
	types.StepSelectPlotCategories:      "Categories",
	types.StepPlotConfig:                "Output",
	types.StepSelectSubjects:            "Subjects",
	types.StepSelectPreprocessingStages: "Stages",
	types.StepPreprocessingFiltering:    "Filtering",
	types.StepPreprocessingICA:          "ICA",
	types.StepPreprocessingEpochs:       "Epochs",
}

func (m Model) View() string {
	if m.showHelp {
		return m.renderHelpOverlay()
	}

	w, h := m.effectiveDimensions()
	containerW, containerH := m.containerDimensions(w, h)
	innerW := containerW - containerPadH*2 - containerBorder
	m.contentWidth = innerW

	header := m.renderHeader(innerW)
	footer := m.renderFooter(innerW)
	mainH := m.mainContentHeight(innerW, containerH, header, footer)
	mainContent := m.renderContent(innerW, mainH)
	mainContent = normalizeContentFrame(mainContent, innerW, mainH)

	mainStyled := styles.RenderNoWrapBlock(lipgloss.NewStyle(), mainContent, innerW)
	// Two blank rows after the wizard chrome (after the stepper bar) so the
	// step content does not crowd the header; one blank row before the footer
	// so the hint row sits slightly below the main block.
	innerView := header + "\n\n\n" + mainStyled + "\n\n" + footer

	containerStyle := lipgloss.NewStyle().
		Height(containerH).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Border).
		Padding(containerPadV, containerPadH)
	container := styles.RenderNoWrapBlock(containerStyle, innerView, containerW)

	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, container)
}

func (m Model) mainContentHeight(contentWidth, containerHeight int, header, footer string) int {
	headerH := strings.Count(header, "\n") + headerSpacingLines
	footerH := strings.Count(footer, "\n") + footerSpacingLines
	return max(containerHeight-containerPadV*2-containerBorder-headerH-footerH, minMainContentHeight)
}

func (m Model) availableMainContentHeight() int {
	w, h := m.effectiveDimensions()
	containerW, containerH := m.containerDimensions(w, h)
	innerW := containerW - containerPadH*2 - containerBorder
	header := m.renderHeader(innerW)
	footer := m.renderFooter(innerW)
	return m.mainContentHeight(innerW, containerH, header, footer)
}

func (m Model) availableAdvancedContentHeight() int {
	return max(m.availableMainContentHeight()-advancedContentOverhead, 1)
}

func normalizeContentFrame(content string, width, height int) string {
	if width <= 0 || height <= 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	framed := make([]string, 0, height)
	for i := 0; i < len(lines) && len(framed) < height; i++ {
		line := styles.TruncateLine(lines[i], width)
		framed = append(framed, styles.PadRight(line, width))
	}
	for len(framed) < height {
		framed = append(framed, strings.Repeat(" ", width))
	}
	return strings.Join(framed, "\n")
}

func (m Model) effectiveDimensions() (int, int) {
	w, h := m.width, m.height
	if w <= 0 {
		w = defaultWidth
	}
	if h <= 0 {
		h = defaultHeight
	}
	return w, h
}

func (m Model) containerDimensions(w, h int) (int, int) {
	cw := w * containerWidthPercent / 100
	if cw < minContainerWidth {
		cw = minContainerWidth
	}
	maxWidth := w - 2
	if maxWidth < 1 {
		maxWidth = 1
	}
	if cw > maxWidth {
		cw = maxWidth
	}

	ch := h - 2
	if ch < minContainerHeight {
		ch = minContainerHeight
	}
	maxHeight := h - 2
	if maxHeight < 1 {
		maxHeight = 1
	}
	if ch > maxHeight {
		ch = maxHeight
	}
	return cw, ch
}

func (m Model) renderHelpOverlay() string {
	w, h := m.effectiveDimensions()
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, m.helpOverlay.View())
}

func (m Model) renderMainContent(isShort bool) string {
	content := m.renderStepContent()
	if len(m.validationErrors) > 0 && !isShort {
		content += "\n" + m.renderValidationErrors()
	}
	return content
}

func (m Model) renderContent(width, height int) string {
	_, terminalHeight := m.effectiveDimensions()
	isShort := terminalHeight < shortHeightThreshold
	content := m.renderMainContent(isShort)
	if !m.usesReviewPanel(width) {
		return content
	}

	stepWidth, reviewWidth := m.reviewColumnWidths(width)
	stepContent := normalizeContentFrame(content, stepWidth, height)
	reviewPanel := m.renderReviewPanel(reviewWidth, height)

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		stepContent,
		strings.Repeat(" ", reviewPanelGap),
		reviewPanel,
	)
}

func (m Model) renderStepContent() string {
	render := m.stepDefinition(m.CurrentStep).render
	if render == nil {
		return ""
	}
	return render(m)
}

func (m Model) renderValidationErrors() string {
	var b strings.Builder
	errStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
	for i, err := range m.validationErrors {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(errStyle.Render(fmt.Sprintf("  %s %s", styles.WarningMark, err)))
	}
	return b.String()
}

func (m Model) renderHeader(width int) string {
	// Title row → blank row → breadcrumb → blank row → stepper bar. The
	// blank rows keep the pipeline title, step rail, and progress gauge
	// from reading as one compressed band at the top of the panel.
	titleRow := m.buildTitleRow(width)
	breadcrumb := m.buildBreadcrumbRow(width)
	progress := m.buildProgressBar(width)
	return titleRow + "\n\n" + breadcrumb + "\n\n" + progress
}

func (m Model) buildTitleRow(width int) string {
	// Whisper-thin accent bar (hairline, not bolded) so it reads as a left
	// gutter marking focus, not as a heavy frame edge. The pipeline name
	// carries the visual weight via Bold + uppercase + tracking.
	bar := lipgloss.NewStyle().Foreground(styles.Primary).Render(styles.SectionIcon)
	pipelineName := strings.ToUpper(m.Pipeline.String())
	title := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render("  " + pipelineName)
	stepPill := m.buildStepPill()
	left := bar + title + "     " + stepPill

	var badges []string
	if badge := m.buildSubjectBadge(); badge != "" {
		badges = append(badges, badge)
	}
	if badge := m.buildPresetBadge(); badge != "" {
		badges = append(badges, badge)
	}

	if len(badges) == 0 {
		return left
	}

	right := strings.Join(badges, "  ")
	gap := width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 2 {
		return left + "  " + right
	}
	return left + strings.Repeat(" ", gap) + right
}

func (m Model) buildStepPill() string {
	if len(m.steps) == 0 {
		return ""
	}
	return styles.RenderStepPill(m.stepIndex+1, len(m.steps))
}

func (m Model) buildProgressBar(width int) string {
	if len(m.steps) == 0 {
		return ""
	}
	filled := m.stepIndex + 1
	if filled < 1 {
		filled = 1
	}
	return styles.RenderStepperBar(filled, len(m.steps), width)
}

func (m Model) buildSubjectBadge() string {
	count := countSelectedStringItems(m.subjectSelected)
	if count > 0 {
		return lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).
			Render(fmt.Sprintf("%d subjects", count))
	}
	if len(m.subjects) > 0 {
		return lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " no subjects")
	}
	return ""
}

func (m Model) buildPresetBadge() string {
	if m.activePreset == "" {
		return ""
	}
	return lipgloss.NewStyle().Foreground(styles.Success).Bold(true).
		Render(styles.CheckMark + " " + m.activePreset)
}

func (m Model) buildBreadcrumbRow(width int) string {
	crumbs := make([]styles.BreadcrumbStep, 0, len(m.steps))
	for _, step := range m.steps {
		name := stepDisplayNames[step]
		if name == "" {
			name = step.String()
		}
		crumbs = append(crumbs, styles.BreadcrumbStep{Name: name})
	}
	return styles.RenderBreadcrumb(crumbs, m.stepIndex, styles.IsNarrowLayout(width))
}

func (m Model) renderFooter(width int) string {
	var hints []footerHint

	switch {
	case m.editingText:
		hints = []footerHint{
			{key: "Type", label: "Edit", compact: "Edit", priority: 0},
			{key: "Enter", label: "Save", compact: "Save", priority: 0},
			{key: "Esc", label: "Cancel", compact: "Cancel", priority: 0},
		}
	case m.editingNumber:
		hints = []footerHint{
			{key: "Type", label: "Enter Number", compact: "Number", priority: 0},
			{key: "Enter", label: "Save", compact: "Save", priority: 0},
			{key: "Esc", label: "Cancel", compact: "Cancel", priority: 0},
		}
	default:
		hints = m.getStepHints()
		if m.usesReviewPanel(m.contentWidth) {
			hints = append(hints,
				footerHint{key: "C", label: "Copy cmd", compact: "Copy", priority: 1},
				footerHint{key: "[ ]", label: "Scroll cmd", compact: "Scroll", priority: 2},
			)
		}
	}

	divider := styles.RenderFooterDivider(width)
	status := m.renderFooterStatus(width)
	barContent := m.renderFooterHints(width, hints)
	footerStyle := styles.FooterStyle
	bar := styles.RenderNoWrapBlock(footerStyle, barContent, width)
	if status == "" {
		return divider + "\n" + bar
	}
	return divider + "\n" + status + "\n" + bar
}

func (m Model) renderFooterHints(width int, hints []footerHint) string {
	return styles.RenderFooterHints(width, toStylesHints(hints))
}

func (m Model) renderFooterStatus(width int) string {
	if len(m.validationErrors) > 0 {
		return m.renderValidationSummary(width)
	}
	if m.toastMessage != "" {
		// Clipboard feedback is rendered inline in the review panel header
		// when that panel is visible, so we suppress the duplicate footer
		// toast in that case. When the panel is not on screen (narrow
		// terminal) the footer remains the canonical surface.
		if m.isClipboardToast() && m.usesReviewPanel(m.contentWidth) {
			return ""
		}
		return m.renderToast(width)
	}
	return ""
}

// isClipboardToast reports whether the active toast belongs to the
// clipboard copy flow and should therefore be routed to the review panel
// instead of the global footer status row.
func (m Model) isClipboardToast() bool {
	return m.toastType == "clipboard" || m.toastType == "clipboard-error"
}

// renderLoadingBanner renders a confident, single-line indeterminate-loading
// affordance: an uppercase title as typographic metadata, then the spinner
// glyph paired with the supplied message on the next line. The spinner sits
// in the gutter (aligned with other content) and the message is rendered in
// the standard text tone — italic + dim made the message read as a sidebar
// caption rather than the primary status, so we keep it upright and at full
// strength. The caller passes the bare spinner glyph; the banner composes
// the message text, so spinners don't double up on their own label.
func (m Model) renderLoadingBanner(title, message, spinner string) string {
	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeader(title) + "\n\n")

	line := "  " + spinner
	if msg := strings.TrimSpace(message); msg != "" {
		line += "  " + lipgloss.NewStyle().Foreground(styles.Text).Render(msg)
	}

	b.WriteString(line + "\n")
	return b.String()
}

func (m Model) renderValidationSummary(width int) string {
	if len(m.validationErrors) == 0 {
		return ""
	}

	summary := m.validationErrors[0]
	if len(m.validationErrors) > 1 {
		summary += fmt.Sprintf(" (+%d more)", len(m.validationErrors)-1)
	}

	text := lipgloss.NewStyle().
		Foreground(styles.Error).
		Bold(true).
		Render(styles.WarningMark + " " + summary)

	return styles.TruncateLine(text, width)
}

func (m Model) renderToast(width int) string {
	type toastDef struct {
		icon string
		fg   lipgloss.Color
	}
	defs := map[string]toastDef{
		"success": {styles.CheckMark, styles.Success},
		"error":   {styles.CrossMark, styles.Error},
		"warning": {styles.WarningMark, styles.Warning},
	}
	d, ok := defs[m.toastType]
	if !ok {
		d = toastDef{styles.ActiveMark, styles.Accent}
	}

	text := lipgloss.NewStyle().
		Foreground(d.fg).Bold(true).
		Render(d.icon + "  " + m.toastMessage)

	return styles.TruncateLine(text, width)
}

func (m Model) getStepHints() []footerHint {
	switch m.CurrentStep {
	case types.StepSelectMode:
		return []footerHint{
			{key: "↑/↓", label: "Navigate", compact: "Nav", priority: 0},
			{key: "Enter", label: "Next", compact: "Next", priority: 0},
			{key: "Esc", label: "Back", compact: "Back", priority: 0},
		}
	case types.StepSelectComputations:
		if m.Pipeline == types.PipelineBehavior {
			return []footerHint{
				{key: "Space", label: "Toggle", compact: "Toggle", priority: 0},
				{key: "A/N", label: "All/None", compact: "All/None", priority: 1},
				{key: "Q", label: "Quick", compact: "Quick", priority: 2},
				{key: "F", label: "Full", compact: "Full", priority: 2},
				{key: "R", label: "Regress", compact: "Regress", priority: 2},
				{key: "T", label: "Temporal", compact: "Temp", priority: 2},
				{key: "Enter", label: "Next", compact: "Next", priority: 0},
			}
		}
		return m.standardSelectionHints()
	case types.StepConfigureOptions:
		if m.Pipeline == types.PipelineFeatures {
			return []footerHint{
				{key: "Space", label: "Toggle", compact: "Toggle", priority: 0},
				{key: "A/N", label: "All/None", compact: "All/None", priority: 1},
				{key: "Enter", label: "Next", compact: "Next", priority: 0},
			}
		}
		return m.standardSelectionHints()
	case types.StepSelectPlotCategories, types.StepSelectBands, types.StepSelectROIs, types.StepSelectSpatial,
		types.StepSelectFeatureFiles, types.StepSelectPlots, types.StepSelectFeaturePlotters,
		types.StepSelectPreprocessingStages:
		return m.standardSelectionHints()
	case types.StepPreprocessingFiltering, types.StepPreprocessingICA, types.StepPreprocessingEpochs:
		if m.editingNumber || m.editingText {
			return []footerHint{
				{key: "Type", label: "Enter Value", compact: "Value", priority: 0},
				{key: "Enter", label: "Confirm", compact: "OK", priority: 0},
				{key: "Esc", label: "Cancel", compact: "Cancel", priority: 0},
			}
		}
		return []footerHint{
			{key: "↑/↓", label: "Navigate", compact: "Nav", priority: 0},
			{key: "Enter", label: "Edit", compact: "Edit", priority: 0},
			{key: "Space", label: "Toggle", compact: "Toggle", priority: 1},
			{key: "Esc", label: "Back", compact: "Back", priority: 0},
		}
	case types.StepPlotConfig:
		return []footerHint{
			{key: "Space", label: "Toggle/Cycle", compact: "Toggle", priority: 0},
			{key: "↑/↓", label: "Navigate", compact: "Nav", priority: 0},
			{key: "Enter", label: "Next", compact: "Next", priority: 0},
			{key: "Esc", label: "Back", compact: "Back", priority: 0},
		}
	case types.StepTimeRange:
		return []footerHint{
			{key: "+", label: "Add", compact: "Add", priority: 1},
			{key: "D", label: "Delete", compact: "Del", priority: 1},
			{key: "Space", label: "Edit", compact: "Edit", priority: 0},
			{key: "Enter", label: "Next", compact: "Next", priority: 0},
			{key: "Esc", label: "Back", compact: "Back", priority: 0},
		}
	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			return []footerHint{
				{key: "Space", label: "Toggle Item", compact: "Toggle", priority: 0},
				{key: "↑/↓", label: "Navigate", compact: "Nav", priority: 0},
				{key: "Esc", label: "Close List", compact: "Close", priority: 0},
			}
		}
		return []footerHint{
			{key: "Space", label: "Toggle/Expand", compact: "Toggle", priority: 0},
			{key: "↑/↓", label: "Navigate", compact: "Nav", priority: 0},
			{key: "Enter", label: "Next", compact: "Next", priority: 0},
			{key: "Esc", label: "Back", compact: "Back", priority: 0},
		}
	case types.StepSelectSubjects:
		return []footerHint{
			{key: "Space", label: "Toggle", compact: "Toggle", priority: 0},
			{key: "A/N", label: "All/None", compact: "All/None", priority: 1},
			{key: "R", label: "Reload", compact: "Reload", priority: 1},
			{key: "Enter", label: "Next", compact: "Next", priority: 0},
		}
	default:
		return []footerHint{}
	}
}

func (m Model) standardSelectionHints() []footerHint {
	return []footerHint{
		{key: "Space", label: "Toggle", compact: "Toggle", priority: 0},
		{key: "A", label: "All", compact: "All", priority: 1},
		{key: "N", label: "None", compact: "None", priority: 1},
		{key: "Enter", label: "Next", compact: "Next", priority: 0},
		{key: "Esc", label: "Back", compact: "Back", priority: 0},
	}
}
