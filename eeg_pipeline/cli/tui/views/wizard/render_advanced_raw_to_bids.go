// EEG raw-to-BIDS pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderRawToBidsAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.RenderStepHeader("Advanced", m.contentWidth) + "\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingNumber || m.editingText {
		b.WriteString(infoStyle.Render("  Enter to confirm, Esc to cancel") + "\n")
	} else {
		b.WriteString(infoStyle.Render("  Space: toggle/edit  Enter: proceed") + "\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	montageVal := m.rawMontage
	if m.editingText && m.editingTextField == textFieldRawMontage {
		montageVal = m.textBuffer + "█"
	}

	prefixVal := m.rawEventPrefixes
	if m.editingText && m.editingTextField == textFieldRawEventPrefixes {
		prefixVal = m.textBuffer + "█"
	}

	lineFreqVal := fmt.Sprintf("%d", m.rawLineFreq)
	if m.editingNumber && m.isCurrentlyEditing(optRawLineFreq) {
		lineFreqVal = m.numberBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Montage", montageVal, "Type to edit"},
		{"Line Freq", lineFreqVal, "Type a number (Hz)"},
		{"Overwrite", m.boolToOnOff(m.rawOverwrite), "Replace existing BIDS files"},
		{"Trim to First Volume", m.boolToOnOff(m.rawTrimToFirstVolume), "Trim EEG to fMRI start"},
		{"Event Prefixes", prefixVal, "Comma-separated (optional)"},
		{"Keep Annotations", m.boolToOnOff(m.rawKeepAnnotations), "Keep all raw annotations"},
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
