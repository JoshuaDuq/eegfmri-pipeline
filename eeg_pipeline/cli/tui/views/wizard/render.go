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

	// Command preview overlay
	if m.showCommandPreview {
		return m.renderCommandPreviewOverlay()
	}

	var b strings.Builder

	b.WriteString(m.renderHeader())
	b.WriteString("\n\n")

	if m.ConfirmingExecute {
		b.WriteString(m.renderConfirmation())
		return b.String()
	}

	switch m.CurrentStep {
	case types.StepSelectMode:
		b.WriteString(m.renderModeSelection())

	case types.StepSelectComputations:
		b.WriteString(m.renderComputationSelection())
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline or Plotting pipeline categories
		b.WriteString(m.renderCategorySelection())
	case types.StepSelectPlots:
		b.WriteString(m.renderPlotSelection())
	case types.StepPlotConfig:
		b.WriteString(m.renderPlotConfig())
	case types.StepSelectBands:
		b.WriteString(m.renderBandSelection())
	case types.StepSelectFeatureFiles:
		b.WriteString(m.renderFeatureFileSelection())
	case types.StepSelectSpatial:
		b.WriteString(m.renderSpatialSelection())
	case types.StepTimeRange:
		b.WriteString(m.renderTimeRange())
	case types.StepAdvancedConfig:
		b.WriteString(m.renderAdvancedConfig())
	case types.StepSelectSubjects:
		b.WriteString(m.renderSubjectSelection())
	case types.StepReviewExecute:
		b.WriteString(m.renderReview())
	}

	if len(m.validationErrors) > 0 && m.CurrentStep != types.StepReviewExecute {
		b.WriteString("\n")
		b.WriteString(m.renderStepValidationErrors())
	}

	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return b.String()
}

func (m Model) renderCommandPreviewOverlay() string {
	var content strings.Builder

	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("◆ Execution Preview")
	content.WriteString(title + "\n\n")

	// Structured summary
	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(14)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	// Pipeline
	content.WriteString(labelStyle.Render("Pipeline:") + " " + accentStyle.Render(m.Pipeline.String()) + "\n")

	// Mode
	if len(m.modeOptions) > m.modeIndex {
		content.WriteString(labelStyle.Render("Mode:") + " " + valueStyle.Render(m.modeOptions[m.modeIndex]) + "\n")
	}

	// Subject count
	selectedCount := 0
	for _, sel := range m.subjectSelected {
		if sel {
			selectedCount++
		}
	}
	subjColor := styles.Success
	if selectedCount == 0 {
		subjColor = styles.Error
	}
	subjStyle := lipgloss.NewStyle().Foreground(subjColor).Bold(true)
	content.WriteString(labelStyle.Render("Subjects:") + " " + subjStyle.Render(fmt.Sprintf("%d", selectedCount)) + "\n")

	// Categories (for features pipeline)
	if m.Pipeline == types.PipelineFeatures && len(m.selected) > 0 {
		cats := m.SelectedCategories()
		if len(cats) > 0 {
			var chips string
			for i, cat := range cats {
				if i > 0 {
					chips += ", "
				}
				chips += cat
			}
			content.WriteString(labelStyle.Render("Features:") + " " + valueStyle.Render(chips) + "\n")
		}
	}
	if m.Pipeline == types.PipelinePlotting {
		plots := m.SelectedPlotIDs()
		if len(plots) > 0 {
			content.WriteString(labelStyle.Render("Plots:") + " " + valueStyle.Render(fmt.Sprintf("%d selected", len(plots))) + "\n")
		}
	}

	// Divider
	content.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", 40)) + "\n\n")

	// Command
	content.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render("Command:") + "\n")
	cmdStyle := lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Padding(0, 1).
		MarginLeft(2)
	content.WriteString(cmdStyle.Render(m.BuildCommand()) + "\n\n")

	content.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("Press P or Esc to close"))

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Primary).
		Padding(1, 2).
		Width(m.width - 20)

	overlayPlaced := lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		box.Render(content.String()),
	)

	return overlayPlaced
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
		types.StepSelectMode: "Mode",

		types.StepSelectComputations:   "Analyses",
		types.StepSelectFeatureFiles:   "Files",
		types.StepConfigureOptions:     "Features",
		types.StepSelectBands:          "Bands",
		types.StepSelectSpatial:        "Spatial",
		types.StepTimeRange:            "Time",
		types.StepAdvancedConfig:       "Advanced",
		types.StepSelectPlots:          "Plots",
		types.StepSelectPlotCategories: "Categories",
		types.StepPlotConfig:           "Plot Config",
		types.StepSelectSubjects:       "Subjects",
		types.StepReviewExecute:        "Review",
	}

	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	title := accentStyle.Render("◆ ") + styles.BrandStyle.Render(strings.ToUpper(m.Pipeline.String()))

	// Minimal dot-based step indicator
	var stepDots []string
	for i := range m.steps {
		if i < m.stepIndex {
			// Completed step
			stepDots = append(stepDots, lipgloss.NewStyle().Foreground(styles.Success).Render("●"))
		} else if i == m.stepIndex {
			// Current step - subtle pulse
			frames := []string{"◉", "●", "◉", "◎"}
			frame := frames[(m.ticker/3)%len(frames)]
			stepDots = append(stepDots, lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(frame))
		} else {
			// Pending step
			stepDots = append(stepDots, lipgloss.NewStyle().Foreground(styles.Muted).Render("○"))
		}
	}
	stepperLine := strings.Join(stepDots, " ")

	// Current step name and progress
	currentStepName := stepNames[m.CurrentStep]
	stepInfo := lipgloss.NewStyle().
		Foreground(styles.Text).
		Bold(true).
		MarginLeft(2).
		Render(currentStepName)

	stepProgress := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Render(fmt.Sprintf(" (%d/%d)", m.stepIndex+1, len(m.steps)))

	// Subject count badge - show how many subjects are selected
	subjectBadge := ""
	selectedCount := 0
	for _, sel := range m.subjectSelected {
		if sel {
			selectedCount++
		}
	}
	if selectedCount > 0 {
		badgeStyle := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1).
			MarginLeft(2)
		subjectBadge = badgeStyle.Render(fmt.Sprintf("%d subjects", selectedCount))
	} else if len(m.subjects) > 0 {
		badgeStyle := lipgloss.NewStyle().
			Foreground(styles.Warning).
			MarginLeft(2)
		subjectBadge = badgeStyle.Render("(no subjects selected)")
	}

	return lipgloss.JoinHorizontal(lipgloss.Center, title, "  ", stepperLine, stepInfo, stepProgress, subjectBadge)
}

func (m Model) renderFooter() string {
	var hints []string

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

	case types.StepSelectComputations, types.StepConfigureOptions, types.StepSelectPlotCategories, types.StepSelectBands, types.StepSelectSpatial, types.StepSelectFeatureFiles, types.StepSelectPlots:
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
			styles.RenderKeyHint("↑/↓", "Select Field"),
			styles.RenderKeyHint("Type", "Enter Value"),
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
		hints = []string{
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("A", "All"),
			styles.RenderKeyHint("N", "None"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepReviewExecute:
		if len(m.validationErrors) > 0 {
			hints = []string{
				lipgloss.NewStyle().Foreground(styles.Error).Render("Fix errors to continue"),
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
