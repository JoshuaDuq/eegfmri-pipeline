// Merge PsychoPy data pipeline advanced configuration.
package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderMergeBehaviorAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingText {
		b.WriteString(infoStyle.Render("  Enter to confirm, Esc to cancel") + "\n")
	} else {
		b.WriteString(infoStyle.Render("  Space: toggle/edit  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	prefixVal := m.mergeEventPrefixes
	if m.editingText && m.editingTextField == textFieldMergeEventPrefixes {
		prefixVal = m.textBuffer + "█"
	}
	typeVal := m.mergeEventTypes
	if m.editingText && m.editingTextField == textFieldMergeEventTypes {
		typeVal = m.textBuffer + "█"
	}
	qcVal := m.mergeQCColumns
	if m.editingText && m.editingTextField == textFieldMergeQCColumns {
		qcVal = m.textBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Event Prefixes", prefixVal, "Comma-separated (optional)"},
		{"Event Types", typeVal, "Comma-separated (optional)"},
		{"QC Columns", qcVal, "Comma-separated (optional)"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}

		if m.useDefaultAdvanced && i > 0 {
			labelStyle = labelStyle.Faint(true)
			valueStyle = lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
		} else {
			valueStyle = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		}

		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}

		displayVal := opt.value
		if displayVal == "" {
			displayVal = "(none)"
		}

		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(opt.label+":"), valueStyle.Render(displayVal), hintStyle.Render(opt.hint), labelWidth, m.contentWidth) + "\n")
	}

	return b.String()
}
