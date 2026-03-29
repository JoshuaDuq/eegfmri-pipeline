package globalsetup

import (
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

// Rendering helpers for the global setup view.

func (m Model) View() string {
	title := styles.RenderSectionLabel("Global Setup")
	lineWidth := m.width - 8
	if lineWidth < 20 {
		lineWidth = 20
	}
	tabs := m.renderSectionTabs()
	header := title + "\n" + tabs + "\n" + styles.RenderHeaderSeparator(lineWidth)
	headerHeight := strings.Count(header, "\n") + 2

	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	var mainContent strings.Builder
	mainContent.WriteString(m.renderFields(lineWidth))

	if m.isLoading {
		mainContent.WriteString("\n" + styles.TruncateLine("  "+m.searchSpinner.View(), lineWidth))
	}

	if m.statusMessage != "" {
		color := styles.Success
		if m.statusIsError {
			color = styles.Error
		}
		statusLine := lipgloss.NewStyle().Foreground(color).Render(m.statusMessage)
		mainContent.WriteString("\n" + styles.TruncateLine(statusLine, lineWidth))
	}

	if m.isSaving {
		mainContent.WriteString("\n" + styles.TruncateLine("  "+m.saveSpinner.View(), lineWidth))
	}

	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(styles.ClampBlock(mainContent.String(), lineWidth))

	return header + "\n" + mainContentStyled + "\n" + footer
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

	width := m.width - 8
	if width < 20 {
		width = 20
	}
	divider := styles.RenderDivider(width)
	bar := styles.RenderNoWrapBlock(styles.FooterStyle, strings.Join(hints, styles.RenderFooterSeparator()), width)
	return divider + "\n" + bar
}

func (m Model) renderSectionTabs() string {
	var parts []string
	for i, sec := range m.sections {
		if i == m.sectionIndex {
			parts = append(parts, lipgloss.NewStyle().
				Foreground(styles.Primary).
				Bold(true).Underline(true).
				Render(sec.label))
		} else {
			parts = append(parts, lipgloss.NewStyle().
				Foreground(styles.TextDim).
				Render(sec.label))
		}
	}
	sep := lipgloss.NewStyle().Foreground(styles.Border).Render("  ·  ")
	return "  " + strings.Join(parts, sep)
}

func (m Model) renderFields(maxWidth int) string {
	var b strings.Builder
	section := m.sections[m.sectionIndex]

	if section.description != "" {
		description := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render("  " + section.description)
		b.WriteString(styles.TruncateLine(description, maxWidth) + "\n")
	}
	b.WriteString("\n")

	fields := m.sectionFields(section.key)
	for i, field := range fields {
		isFocused := i == m.fieldCursor
		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}

		value := m.fieldValue(field.key)
		if m.editingText && m.editingField == field.key {
			value = m.textBuffer + "█"
		}
		isUnset := value == ""
		if isUnset {
			value = "not set"
		}

		var valueRendered string
		if isUnset {
			valueRendered = lipgloss.NewStyle().
				Foreground(styles.Muted).Italic(true).
				Render("not set")
		} else {
			valueRendered = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(value)
		}
		label := styles.FitLine(labelStyle.Render(field.label), 20)
		prefix := cursor + label + " "

		sep := lipgloss.NewStyle().Foreground(styles.Border).Render(" · ")
		var suffix strings.Builder
		if field.isPath && !isUnset {
			path := m.fieldValue(field.key)
			if pathExists(path) {
				suffix.WriteString(sep)
				suffix.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark))
			} else {
				suffix.WriteString(sep)
				suffix.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " not found"))
			}
		}

		if field.description != "" {
			suffix.WriteString(sep)
			suffix.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(field.description))
		}

		if field.isPath && isFocused {
			suffix.WriteString(sep)
			suffix.WriteString(styles.FooterKeySecondaryStyle.Render("B"))
			suffix.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(" browse"))
		}

		valueWidth := maxWidth - lipgloss.Width(prefix) - lipgloss.Width(suffix.String())
		if valueWidth < 1 {
			valueWidth = 1
		}
		line := prefix + styles.TruncateLine(valueRendered, valueWidth) + suffix.String()
		b.WriteString(styles.TruncateLine(line, maxWidth) + "\n")
	}

	return b.String()
}
