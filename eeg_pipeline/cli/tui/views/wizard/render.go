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
	case types.StepConfigureOptions:
		// Features pipeline only
		b.WriteString(m.renderCategorySelection())
	case types.StepSelectBands:
		b.WriteString(m.renderBandSelection())
	case types.StepSelectFeatureFiles:
		b.WriteString(m.renderFeatureFileSelection())
	case types.StepSelectSpatial:
		b.WriteString(m.renderSpatialSelection())
	case types.StepTFRVizType:
		b.WriteString(m.renderTFRVizTypeSelection())
	case types.StepTFRChannels:
		b.WriteString(m.renderTFRChannelSelection())
	case types.StepTimeRange:
		b.WriteString(m.renderTimeRange())
	case types.StepAdvancedConfig:
		b.WriteString(m.renderAdvancedConfig())
	case types.StepSelectSubjects:
		b.WriteString(m.renderSubjectSelection())
	case types.StepReviewExecute:
		b.WriteString(m.renderReview())
	}

	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return b.String()
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
		types.StepSelectMode:         "Mode",
		types.StepSelectComputations: "Analyses",
		types.StepSelectFeatureFiles: "Files",
		types.StepConfigureOptions:   "Features",
		types.StepSelectBands:        "Bands",
		types.StepSelectSpatial:      "Spatial",
		types.StepTFRVizType:         "Viz Type",
		types.StepTFRChannels:        "Channels",
		types.StepTimeRange:          "Time",
		types.StepAdvancedConfig:     "Advanced",
		types.StepSelectSubjects:     "Subjects",
		types.StepReviewExecute:      "Review",
	}

	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	title := accentStyle.Render("◆ ") + styles.BrandStyle.Render(strings.ToUpper(m.Pipeline.String()))

	var stepper []string
	for i := range m.steps {
		stepName := stepNames[m.steps[i]]
		if len(stepName) > 8 {
			stepName = stepName[:7] + "."
		}

		if i < m.stepIndex {
			style := lipgloss.NewStyle().
				Foreground(lipgloss.Color("#000000")).
				Background(styles.Success).
				Bold(true).
				Padding(0, 1)
			stepper = append(stepper, style.Render(styles.CheckMark))
		} else if i == m.stepIndex {
			frames := []string{"▸", "▹", "▸", "▹"}
			frame := frames[(m.ticker/2)%len(frames)]
			style := lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFFFFF")).
				Background(styles.Primary).
				Bold(true).
				Padding(0, 1)
			stepper = append(stepper, style.Render(frame+" "+stepName))
		} else {
			style := lipgloss.NewStyle().
				Foreground(styles.TextDim).
				Background(styles.Secondary).
				Padding(0, 1)
			num := fmt.Sprintf(" %d ", i+1)
			stepper = append(stepper, style.Render(num))
		}
	}

	var stepperLine string
	for i, step := range stepper {
		stepperLine += step
		if i < len(stepper)-1 {
			var connector string
			if i < m.stepIndex {
				connector = lipgloss.NewStyle().Foreground(styles.Success).Render("──")
			} else if i == m.stepIndex {
				connector = lipgloss.NewStyle().Foreground(styles.Primary).Render("──")
			} else {
				connector = lipgloss.NewStyle().Foreground(styles.Secondary).Render("──")
			}
			stepperLine += connector
		}
	}

	stepTitle := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Italic(true).
		MarginLeft(2).
		Render("Step " + fmt.Sprintf("%d", m.stepIndex+1) + " of " + fmt.Sprintf("%d", len(m.steps)))

	return lipgloss.JoinHorizontal(lipgloss.Center, title, "  ", stepperLine, stepTitle)
}

func (m Model) renderFooter() string {
	var hints []string

	switch m.CurrentStep {
	case types.StepSelectMode, types.StepTFRVizType:
		hints = []string{
			styles.RenderKeyHint("↑/↓", "Navigate"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepSelectComputations, types.StepConfigureOptions, types.StepSelectBands, types.StepSelectSpatial, types.StepSelectFeatureFiles:
		hints = []string{
			styles.RenderKeyHint("Space", "Toggle"),
			styles.RenderKeyHint("A", "All"),
			styles.RenderKeyHint("N", "None"),
			styles.RenderKeyHint("Enter", "Next"),
			styles.RenderKeyHint("Esc", "Back"),
		}
	case types.StepTFRChannels:
		if m.editingTfrChans {
			hints = []string{
				styles.RenderKeyHint("Type", "Enter Channels"),
				styles.RenderKeyHint("Enter/Esc", "Done"),
			}
		} else {
			hints = []string{
				styles.RenderKeyHint("↑/↓", "Select Mode"),
				styles.RenderKeyHint("Space", "Edit Channels"),
				styles.RenderKeyHint("Enter", "Next"),
				styles.RenderKeyHint("Esc", "Back"),
			}
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
