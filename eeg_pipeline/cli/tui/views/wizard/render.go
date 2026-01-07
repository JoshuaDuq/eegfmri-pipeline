package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// View Entry Point
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	if m.showHelp {
		return m.renderWithHelpOverlay()
	}

	// Responsive flags
	isNarrow := m.width < 100
	isShort := m.height < 25

	// Render header (fixed at top)
	header := m.renderHeader()
	headerHeight := strings.Count(header, "\n") + 3 // +3 for spacing

	// Render footer (fixed at bottom)
	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2 // +2 for spacing

	// Calculate available height for main content
	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	// Render main content
	var mainContent string
	if m.ConfirmingExecute {
		mainContent = m.renderConfirmation()
	} else {
		switch m.CurrentStep {
		case types.StepSelectMode:
			mainContent = m.renderModeSelection()
		case types.StepSelectComputations:
			mainContent = m.renderComputationSelection()
		case types.StepConfigureOptions, types.StepSelectPlotCategories:
			mainContent = m.renderCategorySelection()
		case types.StepSelectPlots:
			if isNarrow {
				mainContent = m.renderPlotSelection()
			} else {
				mainContent = m.renderPlotSelectionSplit()
			}
		case types.StepSelectFeaturePlotters:
			mainContent = m.renderFeaturePlotterSelection()
		case types.StepPlotConfig:
			mainContent = m.renderPlotConfig()
		case types.StepSelectBands:
			mainContent = m.renderBandSelection()
		case types.StepSelectFeatureFiles:
			mainContent = m.renderFeatureFileSelection()
		case types.StepSelectSpatial:
			mainContent = m.renderSpatialSelection()
		case types.StepTimeRange:
			mainContent = m.renderTimeRange()
		case types.StepAdvancedConfig:
			mainContent = m.renderAdvancedConfig()
		case types.StepSelectSubjects:
			mainContent = m.renderSubjectSelection()
		case types.StepReviewExecute:
			mainContent = m.renderReview()
		}

		// Add validation errors if present
		if len(m.validationErrors) > 0 && m.CurrentStep != types.StepReviewExecute {
			if !isShort {
				mainContent += "\n" + m.renderStepValidationErrors()
			}
		}
	}

	// Force main content to fill available height
	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent)

	// Combine: header + main + footer
	return header + "\n\n" + mainContentStyled + "\n" + footer
}

func (m Model) renderWithHelpOverlay() string {
	overlay := m.helpOverlay.View()

	overlayPlaced := lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		overlay,
	)

	return overlayPlaced
}

///////////////////////////////////////////////////////////////////
// Header & Footer Rendering
///////////////////////////////////////////////////////////////////

func (m Model) renderHeader() string {
	stepNames := map[types.WizardStep]string{
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

	var b strings.Builder

	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	pipelineTitle := accentStyle.Render("◆ ") + styles.BrandStyle.Render(strings.ToUpper(m.Pipeline.String()))

	selectedCount := 0
	for _, sel := range m.subjectSelected {
		if sel {
			selectedCount++
		}
	}
	var subjectBadge string
	if selectedCount > 0 {
		subjectBadge = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1).
			Render(fmt.Sprintf("%d subjects", selectedCount))
	} else if len(m.subjects) > 0 {
		subjectBadge = lipgloss.NewStyle().
			Foreground(styles.Warning).
			Render("⚠ no subjects")
	}

	presetBadge := ""
	if m.activePreset != "" {
		presetBadge = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Success).
			Bold(true).
			Padding(0, 1).
			Render("✓ " + m.activePreset)
	}

	titleRow := lipgloss.JoinHorizontal(lipgloss.Center,
		pipelineTitle,
		"  ",
		subjectBadge,
	)
	if presetBadge != "" {
		titleRow = lipgloss.JoinHorizontal(lipgloss.Center, titleRow, "  ", presetBadge)
	}
	b.WriteString(titleRow + "\n")

	var breadcrumbParts []string
	isCompact := m.width < 100

	for i, step := range m.steps {
		stepName := stepNames[step]
		if stepName == "" {
			stepName = step.String()
		}

		var icon, text string
		if i < m.stepIndex {
			icon = lipgloss.NewStyle().Foreground(styles.Success).Render("●")
			text = lipgloss.NewStyle().Foreground(styles.TextDim).Render(stepName)
		} else if i == m.stepIndex {
			frames := []string{"◉", "●", "◉", "◎"}
			frame := frames[(m.ticker/3)%len(frames)]
			icon = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(frame)
			text = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(stepName)
		} else {
			icon = lipgloss.NewStyle().Foreground(styles.Muted).Render("○")
			text = lipgloss.NewStyle().Foreground(styles.Muted).Render(stepName)
		}

		if isCompact {
			breadcrumbParts = append(breadcrumbParts, icon)
		} else {
			breadcrumbParts = append(breadcrumbParts, icon+" "+text)
		}
	}

	connector := lipgloss.NewStyle().Foreground(styles.Secondary).Render(" → ")
	if isCompact {
		connector = " "
	}

	breadcrumb := strings.Join(breadcrumbParts, connector)

	stepCounter := lipgloss.NewStyle().Foreground(styles.TextDim).
		Render(fmt.Sprintf("Step %d of %d", m.stepIndex+1, len(m.steps)))

	var validationIcon string
	stepErrors := m.validateStep()
	if len(stepErrors) == 0 {
		validationIcon = lipgloss.NewStyle().Foreground(styles.Success).Render(" " + styles.CheckMark)
	} else {
		validationIcon = lipgloss.NewStyle().Foreground(styles.Warning).Render(" " + styles.WarningMark)
	}

	breadcrumbRow := lipgloss.JoinHorizontal(lipgloss.Center,
		breadcrumb,
		"    ",
		stepCounter,
		validationIcon,
	)
	b.WriteString(breadcrumbRow)

	return b.String()
}

func (m Model) renderFooter() string {
	var hints []string

	// Toast notification takes precedence
	if m.toastMessage != "" {
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
		return styles.FooterStyle.Width(m.width - 8).Render(toastStyle.Render("✓ " + m.toastMessage))
	}

	if m.editingText {
		hints = []string{
			styles.RenderKeyHint("Type", "Edit"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
		separator := "  "
		return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
	}

	if m.editingNumber {
		hints = []string{
			styles.RenderKeyHint("Type", "Enter Number"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
		separator := "  "
		return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
	}

	switch m.CurrentStep {
	case types.StepSelectMode:
		hints = []string{
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}

	case types.StepSelectComputations:
		if m.Pipeline == types.PipelineBehavior {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A/N", "All/None"),
				styles.RenderKeyHint("Q", "Quick"),
				styles.RenderKeyHint("F", "Full"),
				styles.RenderKeyHint("R", "Regress"),
				styles.RenderKeyHint("T", "Temporal"),
				styles.RenderKeyHint("Enter", "Next"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A", "All"),
				styles.RenderKeyHint("N", "None"),
				styles.RenderKeyHint("Enter", "Next"),
				styles.RenderKeyHint("Esc", "Back"),
			}
		}

	case types.StepConfigureOptions:
		if m.Pipeline == types.PipelineFeatures {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A/N", "All/None"),
				styles.RenderKeyHint("Q", "Quick"),
				styles.RenderKeyHint("F", "Full"),
				styles.RenderKeyHint("C", "Connect"),
				styles.RenderKeyHint("S", "Spectral"),
				styles.RenderKeyHint("Enter", "Next"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A", "All"),
				styles.RenderKeyHint("N", "None"),
				styles.RenderKeyHint("Enter", "Next"),
				styles.RenderKeyHint("Esc", "Back"),
			}
		}

	case types.StepSelectPlotCategories, types.StepSelectBands, types.StepSelectSpatial, types.StepSelectFeatureFiles, types.StepSelectPlots, types.StepSelectFeaturePlotters:
		hints = []string{
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("A", "All"),
			styles.RenderKeyHint("N", "None"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}

	case types.StepPlotConfig:
		hints = []string{
			styles.RenderKeyHint("Space", "Toggle/Cycle"),
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}

	case types.StepTimeRange:
		hints = []string{
			styles.RenderKeyHint("A", "Add"),
			styles.RenderKeyHint("D", "Delete"),
			styles.RenderKeyHint("Space", "Edit"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}

	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle Item"),
				styles.RenderKeyHint("↑/↓", "Navigate"),
				styles.RenderKeyHint("Esc", "Close List"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle/Expand"),
				styles.RenderKeyHint("↑/↓", "Navigate"),
				styles.RenderKeyHint("Enter", "Next"),
				styles.RenderKeyHint("Esc", "Back"),
			}
		}

	case types.StepSelectSubjects:
		if m.filteringSubject {
			hints = []string{
				styles.RenderKeyHint("Type", "Filter"),
				styles.RenderKeyHint("Enter", "Apply"),
				styles.RenderKeyHint("Esc", "Clear"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("Space", "Toggle"),
				styles.RenderKeyHint("A/N", "All/None"),
				styles.RenderKeyHint("/", "Filter"),
				styles.RenderKeyHint("F5", "Refresh"),
				styles.RenderKeyHint("Enter", "Next"),
			}
		}

	case types.StepReviewExecute:
		if len(m.validationErrors) > 0 {
			hints = []string{
				lipgloss.NewStyle().Foreground(styles.Error).Render("⚠ Fix errors to continue"),
				styles.RenderKeyHint("Esc", "Back"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("Enter", "EXECUTE"),
				styles.RenderKeyHint("Esc", "Back"),
			}
		}
	}

	separator := "  "
	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
}
func (m Model) renderStepValidationErrors() string {
	if len(m.validationErrors) == 0 {
		return ""
	}
	var b strings.Builder
	for _, err := range m.validationErrors {
		// Single line error with warning mark
		b.WriteString(lipgloss.NewStyle().
			Foreground(styles.Error).
			Bold(true).
			Render(fmt.Sprintf("  %s %s", styles.WarningMark, err)) + "\n")
	}
	return b.String()
}
