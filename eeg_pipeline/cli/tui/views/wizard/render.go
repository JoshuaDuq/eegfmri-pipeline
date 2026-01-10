package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

const (
	narrowWidthThreshold   = 100
	shortHeightThreshold   = 25
	minMainContentHeight   = 10
	headerSpacingLines     = 3
	footerSpacingLines     = 2
	breadcrumbFrameDivisor = 3
	footerWidthPadding     = 8
	footerHintSeparator    = "  "
)

///////////////////////////////////////////////////////////////////
// View Entry Point
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	if m.showHelp {
		return m.renderWithHelpOverlay()
	}

	isNarrow := m.width < narrowWidthThreshold
	isShort := m.height < shortHeightThreshold

	header := m.renderHeader()
	headerHeight := calculateHeaderHeight(header)

	footer := m.renderFooter()
	footerHeight := calculateFooterHeight(footer)

	mainHeight := calculateMainContentHeight(m.height, headerHeight, footerHeight)
	mainContent := m.renderMainContent(isNarrow, isShort)

	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent)

	return assembleView(header, mainContentStyled, footer)
}

func calculateHeaderHeight(header string) int {
	return strings.Count(header, "\n") + headerSpacingLines
}

func calculateFooterHeight(footer string) int {
	return strings.Count(footer, "\n") + footerSpacingLines
}

func calculateMainContentHeight(totalHeight, headerHeight, footerHeight int) int {
	availableHeight := totalHeight - headerHeight - footerHeight
	if availableHeight < minMainContentHeight {
		return minMainContentHeight
	}
	return availableHeight
}

func (m Model) renderMainContent(isNarrow, isShort bool) string {
	if m.ConfirmingExecute {
		return m.renderConfirmation()
	}

	mainContent := m.renderStepContent(isNarrow)

	if m.shouldShowValidationErrors(isShort) {
		mainContent += "\n" + m.renderStepValidationErrors()
	}

	return mainContent
}

func (m Model) renderStepContent(isNarrow bool) string {
	switch m.CurrentStep {
	case types.StepSelectMode:
		return m.renderModeSelection()
	case types.StepSelectComputations:
		return m.renderComputationSelection()
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		return m.renderCategorySelection()
	case types.StepSelectPlots:
		if isNarrow {
			return m.renderPlotSelection()
		}
		return m.renderPlotSelectionSplit()
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
	case types.StepReviewExecute:
		return m.renderReview()
	default:
		return ""
	}
}

func (m Model) shouldShowValidationErrors(isShort bool) bool {
	return len(m.validationErrors) > 0 &&
		m.CurrentStep != types.StepReviewExecute &&
		!isShort
}

func assembleView(header, mainContent, footer string) string {
	return header + "\n\n" + mainContent + "\n" + footer
}

func (m Model) renderWithHelpOverlay() string {
	overlay := m.helpOverlay.View()
	return lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		overlay,
	)
}

///////////////////////////////////////////////////////////////////
// Header & Footer Rendering
///////////////////////////////////////////////////////////////////

var stepDisplayNames = map[types.WizardStep]string{
	types.StepSelectMode:            "Mode",
	types.StepSelectComputations:    "Analyses",
	types.StepSelectFeatureFiles:    "Files",
	types.StepConfigureOptions:      "Features",
	types.StepSelectBands:           "Bands",
	types.StepSelectSpatial:         "Spatial",
	types.StepTimeRange:             "Time",
	types.StepAdvancedConfig:        "Advanced",
	types.StepSelectPlots:           "Plots",
	types.StepSelectFeaturePlotters: "Feature Plots",
	types.StepSelectPlotCategories:  "Categories",
	types.StepPlotConfig:            "Output",
	types.StepSelectSubjects:        "Subjects",
	types.StepReviewExecute:         "Review",
}

func (m Model) renderHeader() string {
	var headerBuilder strings.Builder

	titleRow := m.buildHeaderTitleRow()
	headerBuilder.WriteString(titleRow + "\n")

	breadcrumbRow := m.buildBreadcrumbRow()
	headerBuilder.WriteString(breadcrumbRow)

	return headerBuilder.String()
}

func (m Model) buildHeaderTitleRow() string {
	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	pipelineTitle := accentStyle.Render("◆ ") + styles.BrandStyle.Render(strings.ToUpper(m.Pipeline.String()))

	subjectBadge := m.buildSubjectBadge()
	presetBadge := m.buildPresetBadge()

	titleRow := lipgloss.JoinHorizontal(lipgloss.Center, pipelineTitle, "  ", subjectBadge)
	if presetBadge != "" {
		titleRow = lipgloss.JoinHorizontal(lipgloss.Center, titleRow, "  ", presetBadge)
	}

	return titleRow
}

func (m Model) buildSubjectBadge() string {
	selectedCount := m.countSelectedSubjects()
	if selectedCount > 0 {
		return m.renderBadge(fmt.Sprintf("%d subjects", selectedCount), styles.Accent)
	}
	if len(m.subjects) > 0 {
		return lipgloss.NewStyle().Foreground(styles.Warning).Render("⚠ no subjects")
	}
	return ""
}

func (m Model) countSelectedSubjects() int {
	count := 0
	for _, selected := range m.subjectSelected {
		if selected {
			count++
		}
	}
	return count
}

func (m Model) buildPresetBadge() string {
	if m.activePreset == "" {
		return ""
	}
	return m.renderBadge("✓ "+m.activePreset, styles.Success)
}

func (m Model) renderBadge(text string, bgColor lipgloss.Color) string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(bgColor).
		Bold(true).
		Padding(0, 1).
		Render(text)
}

func (m Model) buildBreadcrumbRow() string {
	isCompact := m.width < narrowWidthThreshold
	breadcrumb := m.buildBreadcrumb(isCompact)
	stepCounter := m.buildStepCounter()
	validationIcon := m.buildValidationIcon()

	return lipgloss.JoinHorizontal(lipgloss.Center,
		breadcrumb,
		"    ",
		stepCounter,
		validationIcon,
	)
}

func (m Model) buildBreadcrumb(isCompact bool) string {
	var parts []string
	connector := m.getBreadcrumbConnector(isCompact)

	for i, step := range m.steps {
		icon, text := m.renderBreadcrumbStep(i, step)
		if isCompact {
			parts = append(parts, icon)
		} else {
			parts = append(parts, icon+" "+text)
		}
	}

	return strings.Join(parts, connector)
}

func (m Model) getBreadcrumbConnector(isCompact bool) string {
	if isCompact {
		return " "
	}
	return lipgloss.NewStyle().Foreground(styles.Secondary).Render(" → ")
}

func (m Model) renderBreadcrumbStep(index int, step types.WizardStep) (icon, text string) {
	stepName := m.getStepDisplayName(step)

	if index < m.stepIndex {
		return m.renderCompletedStep(stepName)
	}
	if index == m.stepIndex {
		return m.renderCurrentStep(stepName)
	}
	return m.renderPendingStep(stepName)
}

func (m Model) getStepDisplayName(step types.WizardStep) string {
	if name, exists := stepDisplayNames[step]; exists {
		return name
	}
	return step.String()
}

func (m Model) renderCompletedStep(stepName string) (icon, text string) {
	icon = lipgloss.NewStyle().Foreground(styles.Success).Render("●")
	text = lipgloss.NewStyle().Foreground(styles.TextDim).Render(stepName)
	return icon, text
}

func (m Model) renderCurrentStep(stepName string) (icon, text string) {
	frames := []string{"◉", "●", "◉", "◎"}
	frameIndex := (m.ticker / breadcrumbFrameDivisor) % len(frames)
	currentFrame := frames[frameIndex]

	icon = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(currentFrame)
	text = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(stepName)
	return icon, text
}

func (m Model) renderPendingStep(stepName string) (icon, text string) {
	icon = lipgloss.NewStyle().Foreground(styles.Muted).Render("○")
	text = lipgloss.NewStyle().Foreground(styles.Muted).Render(stepName)
	return icon, text
}

func (m Model) buildStepCounter() string {
	return lipgloss.NewStyle().Foreground(styles.TextDim).
		Render(fmt.Sprintf("Step %d of %d", m.stepIndex+1, len(m.steps)))
}

func (m Model) buildValidationIcon() string {
	stepErrors := m.validateStep()
	if len(stepErrors) == 0 {
		return lipgloss.NewStyle().Foreground(styles.Success).Render(" " + styles.CheckMark)
	}
	return lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark)
}

func (m Model) renderFooter() string {
	if m.toastMessage != "" {
		return m.renderToastFooter()
	}

	if m.editingText {
		return m.renderEditingTextFooter()
	}

	if m.editingNumber {
		return m.renderEditingNumberFooter()
	}

	hints := m.getStepHints()
	return m.formatFooterHints(hints)
}

func (m Model) renderToastFooter() string {
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
	return styles.FooterStyle.Width(m.width - footerWidthPadding).
		Render(toastStyle.Render("✓ " + m.toastMessage))
}

func (m Model) renderEditingTextFooter() string {
	hints := []string{
		styles.RenderKeyHint("Type", "Edit"),
		styles.RenderKeyHint("Enter", "Save"),
		styles.RenderKeyHint("Esc", "Cancel"),
	}
	return m.formatFooterHints(hints)
}

func (m Model) renderEditingNumberFooter() string {
	hints := []string{
		styles.RenderKeyHint("Type", "Enter Number"),
		styles.RenderKeyHint("Enter", "Save"),
		styles.RenderKeyHint("Esc", "Cancel"),
	}
	return m.formatFooterHints(hints)
}

func (m Model) formatFooterHints(hints []string) string {
	return styles.FooterStyle.Width(m.width - footerWidthPadding).
		Render(strings.Join(hints, footerHintSeparator))
}

func (m Model) getStepHints() []string {
	switch m.CurrentStep {
	case types.StepSelectMode:
		return m.getModeSelectionHints()
	case types.StepSelectComputations:
		return m.getComputationSelectionHints()
	case types.StepConfigureOptions:
		return m.getFeatureOptionsHints()
	case types.StepSelectPlotCategories, types.StepSelectBands, types.StepSelectROIs, types.StepSelectSpatial,
		types.StepSelectFeatureFiles, types.StepSelectPlots, types.StepSelectFeaturePlotters:
		return m.getStandardSelectionHints()
	case types.StepPlotConfig:
		return m.getPlotConfigHints()
	case types.StepTimeRange:
		return m.getTimeRangeHints()
	case types.StepAdvancedConfig:
		return m.getAdvancedConfigHints()
	case types.StepSelectSubjects:
		return m.getSubjectSelectionHints()
	case types.StepReviewExecute:
		return m.getReviewExecuteHints()
	default:
		return []string{}
	}
}

func (m Model) getModeSelectionHints() []string {
	return []string{
		styles.RenderKeyHint("↑/↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getComputationSelectionHints() []string {
	if m.Pipeline == types.PipelineBehavior {
		return []string{
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("A/N", "All/None"),
			styles.RenderKeyHint("Q", "Quick"),
			styles.RenderKeyHint("F", "Full"),
			styles.RenderKeyHint("R", "Regress"),
			styles.RenderKeyHint("T", "Temporal"),
			styles.RenderKeyHint("Enter", "Next"),
		}
	}
	return []string{
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A", "All"),
		styles.RenderKeyHint("N", "None"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getFeatureOptionsHints() []string {
	if m.Pipeline == types.PipelineFeatures {
		return []string{
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("A/N", "All/None"),
			styles.RenderKeyHint("Q", "Quick"),
			styles.RenderKeyHint("F", "Full"),
			styles.RenderKeyHint("C", "Connect"),
			styles.RenderKeyHint("S", "Spectral"),
			styles.RenderKeyHint("Enter", "Next"),
		}
	}
	return []string{
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A", "All"),
		styles.RenderKeyHint("N", "None"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getStandardSelectionHints() []string {
	return []string{
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A", "All"),
		styles.RenderKeyHint("N", "None"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getPlotConfigHints() []string {
	return []string{
		styles.RenderKeyHint("Space", "Toggle/Cycle"),
		styles.RenderKeyHint("↑/↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getTimeRangeHints() []string {
	return []string{
		styles.RenderKeyHint("A", "Add"),
		styles.RenderKeyHint("D", "Delete"),
		styles.RenderKeyHint("Space", "Edit"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getAdvancedConfigHints() []string {
	if m.expandedOption >= 0 {
		return []string{
			styles.RenderKeyHint("Space", "Toggle Item"),
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Esc", "Close List"),
		}
	}
	return []string{
		styles.RenderKeyHint("Space", "Toggle/Expand"),
		styles.RenderKeyHint("↑/↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) getSubjectSelectionHints() []string {
	if m.filteringSubject {
		return []string{
			styles.RenderKeyHint("Type", "Filter"),
			styles.RenderKeyHint("Enter", "Apply"),
			styles.RenderKeyHint("Esc", "Clear"),
		}
	}
	return []string{
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A/N", "All/None"),
		styles.RenderKeyHint("/", "Filter"),
		styles.RenderKeyHint("F5", "Refresh"),
		styles.RenderKeyHint("Enter", "Next"),
	}
}

func (m Model) getReviewExecuteHints() []string {
	if len(m.validationErrors) > 0 {
		return []string{
			lipgloss.NewStyle().Foreground(styles.Error).Render("⚠ Fix errors to continue"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	}
	return []string{
		styles.RenderKeyHint("Enter", "EXECUTE"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}

func (m Model) renderStepValidationErrors() string {
	if len(m.validationErrors) == 0 {
		return ""
	}

	var errorBuilder strings.Builder
	errorStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)

	for _, err := range m.validationErrors {
		errorLine := fmt.Sprintf("  %s %s", styles.WarningMark, err)
		errorBuilder.WriteString(errorStyle.Render(errorLine) + "\n")
	}

	return errorBuilder.String()
}
