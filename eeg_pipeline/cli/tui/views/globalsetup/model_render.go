package globalsetup

import (
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

// Rendering helpers for the global setup view.

func (m Model) View() string {
	lineWidth := m.width - 8
	if lineWidth < 20 {
		lineWidth = 20
	}
	title := styles.RenderStepHeader("Global Setup", lineWidth)
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
	hints := []styles.FooterHint{
		{Key: "↑/↓", Label: "Navigate", Compact: "Nav", Priority: 0},
		{Key: "←/→", Label: "Section", Compact: "Sec", Priority: 0},
		{Key: "Enter", Label: "Edit", Compact: "Edit", Priority: 0},
		{Key: "B", Label: "Browse", Compact: "Browse", Priority: 1},
		{Key: "R", Label: "Reset", Compact: "Reset", Priority: 1},
		{Key: "Esc", Label: "Back", Compact: "Back", Priority: 0},
	}

	if m.editingText {
		hints = []styles.FooterHint{
			{Key: "Type", Label: "Edit", Compact: "Edit", Priority: 0},
			{Key: "Enter", Label: "Save", Compact: "Save", Priority: 0},
			{Key: "Esc", Label: "Cancel", Compact: "Cancel", Priority: 0},
		}
	}

	width := m.width - 8
	if width < 20 {
		width = 20
	}
	divider := styles.RenderFooterDivider(width)
	bar := styles.RenderNoWrapBlock(styles.FooterStyle, styles.RenderFooterHints(width, hints), width)
	return divider + "\n" + bar
}

func (m Model) renderSectionTabs() string {
	var parts []string
	for i, sec := range m.sections {
		if i == m.sectionIndex {
			parts = append(parts, lipgloss.NewStyle().
				Foreground(styles.Primary).
				Bold(true).Underline(true).
				Render(strings.ToUpper(sec.label)))
		} else {
			parts = append(parts, lipgloss.NewStyle().
				Foreground(styles.TextDim).
				Render(strings.ToUpper(sec.label)))
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
		isEditing := m.editingText && m.editingField == field.key
		isUnset := !isEditing && value == ""

		var valueRendered string
		switch {
		case isEditing:
			// Shared, restrained edit affordance (raised-surface field +
			// underline + caret) rather than an inverted highlight.
			valueRendered = styles.RenderEditingInput(m.textBuffer)
		case isUnset:
			valueRendered = lipgloss.NewStyle().
				Foreground(styles.Muted).Italic(true).
				Render("not set")
		default:
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
			suffix.WriteString(styles.RenderKeyBadge("B", false))
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
