package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

type footerHint struct {
	key      string
	label    string
	compact  string
	priority int
}

const (
	narrowWidthThreshold  = 100
	shortHeightThreshold  = 25
	minMainContentHeight  = 10
	headerSpacingLines    = 3
	footerSpacingLines    = 2
	containerPadH         = 2
	containerPadV         = 1
	containerBorder       = 2
	minContainerWidth     = 60
	minContainerHeight    = 15
	containerWidthPercent = 95
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
	mainContent := m.renderMainContent(h < shortHeightThreshold)
	mainContent = normalizeContentFrame(mainContent, innerW, mainH)

	mainStyled := lipgloss.NewStyle().Width(innerW).Render(mainContent)
	innerView := header + "\n" + mainStyled + "\n" + footer

	container := lipgloss.NewStyle().
		Width(containerW).
		Height(containerH).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Border).
		Padding(containerPadV, containerPadH).
		Render(innerView)

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
	return max(m.availableMainContentHeight()-advancedContentOverhead, minVisibleLines)
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

func (m Model) renderStepContent() string {
	switch m.CurrentStep {
	case types.StepSelectMode:
		return m.renderModeSelection()
	case types.StepSelectComputations:
		return m.renderComputationSelection()
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		return m.renderCategorySelection()
	case types.StepSelectPlots:
		return m.renderPlotSelection()
	case types.StepSelectFeaturePlotters:
		return m.renderFeaturePlotterSelection()
	case types.StepPlotConfig:
		return m.renderPlotConfig()
	case types.StepSelectBands:
		return m.renderBandSelection()
	case types.StepSelectROIs:
		return m.renderROISelection()
	case types.StepSelectFeatureFiles:
		return m.renderFeatureFileSelection()
	case types.StepSelectSpatial:
		return m.renderSpatialSelection()
	case types.StepTimeRange:
		return m.renderTimeRange()
	case types.StepAdvancedConfig:
		return m.renderAdvancedConfig()
	case types.StepSelectSubjects:
		return m.renderSubjectSelection()
	case types.StepSelectPreprocessingStages:
		return m.renderPreprocessingStageSelection()
	case types.StepPreprocessingFiltering:
		return m.renderPreprocessingFiltering()
	case types.StepPreprocessingICA:
		return m.renderPreprocessingICA()
	case types.StepPreprocessingEpochs:
		return m.renderPreprocessingEpochs()
	default:
		return ""
	}
}

func (m Model) renderValidationErrors() string {
	var b strings.Builder
	errStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
	for _, err := range m.validationErrors {
		b.WriteString(errStyle.Render(fmt.Sprintf("  %s %s", styles.WarningMark, err)) + "\n")
	}
	return b.String()
}

func (m Model) renderHeader(width int) string {
	titleRow := m.buildTitleRow(width)
	breadcrumb := m.buildBreadcrumbRow()
	progress := m.buildProgressBar(width)
	centerStyle := lipgloss.NewStyle().Width(width).Align(lipgloss.Center)
	return titleRow + "\n" + centerStyle.Render(breadcrumb) + "\n" + progress
}

func (m Model) buildTitleRow(width int) string {
	bar := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(styles.SectionIcon)
	pipelineName := strings.ToUpper(m.Pipeline.String())
	title := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render(" " + pipelineName)
	stepCount := lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("  %d/%d", m.stepIndex+1, len(m.steps)))

	left := bar + title + stepCount

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

func (m Model) buildProgressBar(width int) string {
	if width <= 0 || len(m.steps) == 0 {
		return ""
	}
	total := len(m.steps)
	filled := m.stepIndex
	barWidth := width
	if barWidth < 4 {
		barWidth = 4
	}

	filledW := barWidth * filled / total
	emptyW := barWidth - filledW

	filledStr := styles.ProgressFilledStyle.Render(strings.Repeat(styles.HeaderSeparatorChar, filledW))
	emptyStr := styles.ProgressEmptyStyle.Render(strings.Repeat(styles.SectionDividerChar, emptyW))
	return filledStr + emptyStr
}

func (m Model) buildSubjectBadge() string {
	count := countSelectedStringItems(m.subjectSelected)
	if count > 0 {
		return m.badge(fmt.Sprintf("%d subjects", count), styles.Accent)
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
	return m.badge(styles.CheckMark+" "+m.activePreset, styles.Success)
}

func (m Model) badge(text string, bg lipgloss.Color) string {
	return lipgloss.NewStyle().
		Foreground(styles.BgDark).
		Background(bg).
		Bold(true).
		Padding(0, 1).
		Render(text)
}

func (m Model) buildBreadcrumbRow() string {
	w, _ := m.effectiveDimensions()
	isCompact := w < narrowWidthThreshold

	var parts []string
	connectorStyle := lipgloss.NewStyle().Foreground(styles.Border)

	for i, step := range m.steps {
		name := stepDisplayNames[step]
		if name == "" {
			name = step.String()
		}

		var segment string
		switch {
		case i < m.stepIndex:
			icon := lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark)
			if isCompact {
				segment = icon
			} else {
				segment = icon + " " + lipgloss.NewStyle().Foreground(styles.TextDim).Render(name)
			}
		case i == m.stepIndex:
			numStyle := lipgloss.NewStyle().Foreground(styles.BgDark).Background(styles.Primary).Bold(true).Padding(0, 1)
			if isCompact {
				segment = numStyle.Render(fmt.Sprintf("%d", i+1))
			} else {
				segment = numStyle.Render(fmt.Sprintf("%d", i+1)) + " " +
					lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(name)
			}
		default:
			numStyle := lipgloss.NewStyle().Foreground(styles.Muted)
			if isCompact {
				segment = numStyle.Render(fmt.Sprintf("%d", i+1))
			} else {
				segment = numStyle.Render(fmt.Sprintf("%d", i+1)) + " " +
					lipgloss.NewStyle().Foreground(styles.Muted).Render(name)
			}
		}
		parts = append(parts, segment)
	}

	connector := connectorStyle.Render(" " + styles.SectionDividerChar + " ")
	if isCompact {
		connector = " "
	}
	return strings.Join(parts, connector)
}

func (m Model) renderFooter(width int) string {
	var hints []footerHint

	switch {
	case m.toastMessage != "":
		return m.renderToast(width)
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
	}

	divider := styles.RenderDivider(width)
	barContent := m.renderFooterHints(width, hints)
	bar := styles.FooterStyle.Width(width).Align(lipgloss.Center).
		Render(barContent)
	return divider + "\n" + bar
}

func (m Model) renderFooterHints(width int, hints []footerHint) string {
	if len(hints) == 0 {
		return ""
	}

	render := func(useCompact bool, maxPriority int) string {
		parts := make([]string, 0, len(hints))
		for _, hint := range hints {
			if hint.priority > maxPriority {
				continue
			}
			label := hint.label
			if useCompact && hint.compact != "" {
				label = hint.compact
			}
			parts = append(parts, styles.RenderKeyHint(hint.key, label))
		}
		return strings.Join(parts, styles.RenderFooterSeparator())
	}

	if full := render(false, 2); lipgloss.Width(full) <= width {
		return full
	}
	if compact := render(true, 2); lipgloss.Width(compact) <= width {
		return compact
	}
	for maxPriority := 1; maxPriority >= 0; maxPriority-- {
		if compact := render(true, maxPriority); compact != "" && lipgloss.Width(compact) <= width {
			return compact
		}
	}
	return render(true, 0)
}

func (m Model) renderToast(width int) string {
	toastStyle := lipgloss.NewStyle().Bold(true)
	switch m.toastType {
	case "success":
		toastStyle = toastStyle.Foreground(styles.Success)
	case "error":
		toastStyle = toastStyle.Foreground(styles.Error)
	case "warning":
		toastStyle = toastStyle.Foreground(styles.Warning)
	default:
		toastStyle = toastStyle.Foreground(styles.Accent)
	}
	divider := styles.RenderDivider(width)
	bar := styles.FooterStyle.Width(width).Align(lipgloss.Center).
		Render(toastStyle.Render(styles.CheckMark + " " + m.toastMessage))
	return divider + "\n" + bar
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
