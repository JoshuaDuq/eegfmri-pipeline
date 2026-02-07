package globalsetup

import (
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

// Rendering helpers for the global setup view.

func (m Model) View() string {
	// Render header
	title := styles.RenderSectionLabel("Global Setup")
	section := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render("  " + m.sections[m.sectionIndex].label)
	header := title + section
	headerHeight := 3

	// Render footer
	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

	// Calculate available height for main content
	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	// Build main content
	var mainContent strings.Builder
	mainContent.WriteString(m.renderFields())

	if m.isLoading {
		mainContent.WriteString("\n  " + m.searchSpinner.View())
	}

	if m.statusMessage != "" {
		color := styles.Success
		if m.statusIsError {
			color = styles.Error
		}
		mainContent.WriteString("\n" + lipgloss.NewStyle().Foreground(color).Render(m.statusMessage))
	}

	if m.isSaving {
		mainContent.WriteString("\n  " + m.saveSpinner.View())
	}

	// Force main content to fill available height
	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent.String())

	return header + "\n\n" + mainContentStyled + "\n" + footer
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑/↓", "Navigate"),
		styles.RenderKeyHint("←/→", "Section"),
		styles.RenderKeyHint("Enter", "Edit"),
		styles.RenderKeyHint("B", "Browse"),
		styles.RenderKeyHint("R", "Reset"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	if m.editingText {
		hints = []string{
			styles.RenderKeyHint("Type", "Edit"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
	}

	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, styles.RenderFooterSeparator()))
}

func (m Model) renderFields() string {
	var b strings.Builder
	section := m.sections[m.sectionIndex]

	b.WriteString(styles.SectionTitleStyle.Render(section.label) + "\n\n")
	if section.description != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(section.description) + "\n\n")
	}

	fields := m.sectionFields(section.key)
	for i, field := range fields {
		isFocused := i == m.fieldCursor
		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(20)
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Bold(true).Width(20)
		}

		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}

		value := m.fieldValue(field.key)
		if m.editingText && m.editingField == field.key {
			value = m.textBuffer + "█"
		}
		if value == "" {
			value = "(not set)"
		}

		valueStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		if value == "(not set)" {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Muted)
		}
		line := cursor + labelStyle.Render(field.label) + " " + valueStyle.Render(value)

		if field.isPath && value != "(not set)" {
			path := m.fieldValue(field.key)
			if pathExists(path) {
				line += lipgloss.NewStyle().Foreground(styles.Success).Render("  " + styles.CheckMark)
			} else {
				line += lipgloss.NewStyle().Foreground(styles.Warning).Render("  " + styles.WarningMark + " not found")
			}
		}

		if field.description != "" {
			line += lipgloss.NewStyle().Foreground(styles.Muted).Render("  " + field.description)
		}

		if field.isPath && isFocused {
			line += lipgloss.NewStyle().Foreground(styles.Border).Render("  [B] browse")
		}

		b.WriteString(line + "\n")
	}

	return b.String()
}
