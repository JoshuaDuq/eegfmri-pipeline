package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// Layout Constants
///////////////////////////////////////////////////////////////////

const (
	narrowWidthThreshold  = 100
	shortHeightThreshold  = 25
	minMainContentHeight  = 10
	headerSpacingLines    = 3
	footerSpacingLines    = 2
	footerHintSeparator   = "  "
	breadcrumbFrameDiv    = 3
	containerPadH         = 2
	containerPadV         = 1
	containerBorder       = 2
	minContainerWidth     = 60
	minContainerHeight    = 15
	containerWidthPercent = 95
	defaultWidth          = 120
	defaultHeight         = 40
)

///////////////////////////////////////////////////////////////////
// View Entry Point
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	if m.showHelp {
		return m.renderHelpOverlay()
	}

	w, h := m.effectiveDimensions()
	containerW, containerH := m.containerDimensions(w, h)
	innerW := containerW - containerPadH*2 - containerBorder

	header := m.renderHeader(innerW)
	footer := m.renderFooter(innerW)
	headerH := strings.Count(header, "\n") + headerSpacingLines
	footerH := strings.Count(footer, "\n") + footerSpacingLines

	mainH := max(containerH-containerPadV*2-containerBorder-headerH-footerH, minMainContentHeight)
	mainContent := m.renderMainContent(w < narrowWidthThreshold, h < shortHeightThreshold)

	mainStyled := lipgloss.NewStyle().Width(innerW).Height(mainH).Render(mainContent)
	innerView := header + "\n\n" + mainStyled + "\n" + footer

	container := lipgloss.NewStyle().
		Width(containerW).
		Height(containerH).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Primary).
		Padding(containerPadV, containerPadH).
		Render(innerView)

	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, container)
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
	ch := h - 2
	if ch < minContainerHeight {
		ch = minContainerHeight
	}
	return cw, ch
}

func (m Model) renderHelpOverlay() string {
	w, h := m.effectiveDimensions()
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, m.helpOverlay.View())
}

///////////////////////////////////////////////////////////////////
// Main Content Routing
///////////////////////////////////////////////////////////////////

func (m Model) renderMainContent(isNarrow, isShort bool) string {
	if m.ConfirmingExecute {
		return m.renderConfirmation()
	}

	content := m.renderStepContent(isNarrow)
	if len(m.validationErrors) > 0 && !isShort {
		content += "\n" + m.renderValidationErrors()
	}
	return content
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

///////////////////////////////////////////////////////////////////
// Header Rendering
///////////////////////////////////////////////////////////////////

var stepDisplayNames = map[types.WizardStep]string{
	types.StepSelectMode:                "Mode",
	types.StepSelectComputations:        "Analyses",
	types.StepSelectFeatureFiles:        "Files",
	types.StepConfigureOptions:          "Features",
	types.StepSelectBands:               "Bands",
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

func (m Model) renderHeader(width int) string {
	title := m.buildTitleRow()
	breadcrumb := m.buildBreadcrumbRow()

	centerStyle := lipgloss.NewStyle().Width(width).Align(lipgloss.Center)
	return centerStyle.Render(title) + "\n" + centerStyle.Render(breadcrumb)
}

func (m Model) buildTitleRow() string {
	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	title := accentStyle.Render("◆ ") + styles.BrandStyle.Render(strings.ToUpper(m.Pipeline.String()))

	parts := []string{title}
	if badge := m.buildSubjectBadge(); badge != "" {
		parts = append(parts, badge)
	}
	if badge := m.buildPresetBadge(); badge != "" {
		parts = append(parts, badge)
	}
	return lipgloss.JoinHorizontal(lipgloss.Center, strings.Join(parts, "  "))
}

func (m Model) buildSubjectBadge() string {
	count := 0
	for _, sel := range m.subjectSelected {
		if sel {
			count++
		}
	}
	if count > 0 {
		return m.badge(fmt.Sprintf("%d subjects", count), styles.Accent)
	}
	if len(m.subjects) > 0 {
		return lipgloss.NewStyle().Foreground(styles.Warning).Render("⚠ no subjects")
	}
	return ""
}

func (m Model) buildPresetBadge() string {
	if m.activePreset == "" {
		return ""
	}
	return m.badge("✓ "+m.activePreset, styles.Success)
}

func (m Model) badge(text string, bg lipgloss.Color) string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#000000")).
		Background(bg).
		Bold(true).
		Padding(0, 1).
		Render(text)
}

func (m Model) buildBreadcrumbRow() string {
	w, _ := m.effectiveDimensions()
	isCompact := w < narrowWidthThreshold

	var parts []string
	connector := lipgloss.NewStyle().Foreground(styles.Secondary).Render(" → ")
	if isCompact {
		connector = " "
	}

	for i, step := range m.steps {
		icon, text := m.breadcrumbStep(i, step)
		if isCompact {
			parts = append(parts, icon)
		} else {
			parts = append(parts, icon+" "+text)
		}
	}

	stepCounter := lipgloss.NewStyle().Foreground(styles.TextDim).
		Render(fmt.Sprintf("Step %d of %d", m.stepIndex+1, len(m.steps)))

	validIcon := lipgloss.NewStyle().Foreground(styles.Success).Render(" " + styles.CheckMark)
	if len(m.validateStep()) > 0 {
		validIcon = lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark)
	}

	return strings.Join(parts, connector) + "    " + stepCounter + validIcon
}

func (m Model) breadcrumbStep(idx int, step types.WizardStep) (icon, text string) {
	name := stepDisplayNames[step]
	if name == "" {
		name = step.String()
	}

	switch {
	case idx < m.stepIndex:
		icon = lipgloss.NewStyle().Foreground(styles.Success).Render("●")
		text = lipgloss.NewStyle().Foreground(styles.TextDim).Render(name)
	case idx == m.stepIndex:
		frames := []string{"◉", "●", "◉", "◎"}
		icon = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(frames[(m.ticker/breadcrumbFrameDiv)%len(frames)])
		text = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(name)
	default:
		icon = lipgloss.NewStyle().Foreground(styles.Muted).Render("○")
		text = lipgloss.NewStyle().Foreground(styles.Muted).Render(name)
	}
	return
}

///////////////////////////////////////////////////////////////////
// Footer Rendering
///////////////////////////////////////////////////////////////////

func (m Model) renderFooter(width int) string {
	var hints []string

	switch {
	case m.toastMessage != "":
		return m.renderToast(width)
	case m.editingText:
		hints = []string{
			styles.RenderKeyHint("Type", "Edit"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
	case m.editingNumber:
		hints = []string{
			styles.RenderKeyHint("Type", "Enter Number"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
	default:
		hints = m.getStepHints()
	}

	return styles.FooterStyle.Width(width).Align(lipgloss.Center).
		Render(strings.Join(hints, footerHintSeparator))
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
	return styles.FooterStyle.Width(width).Align(lipgloss.Center).
		Render(toastStyle.Render("✓ " + m.toastMessage))
}

///////////////////////////////////////////////////////////////////
// Step Hints
///////////////////////////////////////////////////////////////////

func (m Model) getStepHints() []string {
	switch m.CurrentStep {
	case types.StepSelectMode:
		return []string{
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepSelectComputations:
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
		return m.standardSelectionHints()
	case types.StepConfigureOptions:
		if m.Pipeline == types.PipelineFeatures {
			return []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A/N", "All/None"),
				styles.RenderKeyHint("Enter", "Next"),
			}
		}
		return m.standardSelectionHints()
	case types.StepSelectPlotCategories, types.StepSelectBands, types.StepSelectROIs, types.StepSelectSpatial,
		types.StepSelectFeatureFiles, types.StepSelectPlots, types.StepSelectFeaturePlotters,
		types.StepSelectPreprocessingStages:
		return m.standardSelectionHints()
	case types.StepPreprocessingFiltering, types.StepPreprocessingICA, types.StepPreprocessingEpochs:
		if m.editingNumber || m.editingText {
			return []string{
				styles.RenderKeyHint("Type", "Enter Value"),
				styles.RenderKeyHint("Enter", "Confirm"),
				styles.RenderKeyHint("Esc", "Cancel"),
			}
		}
		return []string{
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Edit"),
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepPlotConfig:
		return []string{
			styles.RenderKeyHint("Space", "Toggle/Cycle"),
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepTimeRange:
		return []string{
			styles.RenderKeyHint("A", "Add"),
			styles.RenderKeyHint("D", "Delete"),
			styles.RenderKeyHint("Space", "Edit"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepAdvancedConfig:
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
	case types.StepSelectSubjects:
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
	default:
		return []string{}
	}
}

func (m Model) standardSelectionHints() []string {
	return []string{
		styles.RenderKeyHint("Space", "Toggle"),
		styles.RenderKeyHint("A", "All"),
		styles.RenderKeyHint("N", "None"),
		styles.RenderKeyHint("Enter", "Next"),
		styles.RenderKeyHint("Esc", "Back"),
	}
}
