// Preprocessing step UIs: stages, filtering, ICA, epochs.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderPreprocessingStageSelection() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Preprocessing stages") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Select which preprocessing stages to run.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, A to select all, N to select none.") + "\n\n")

	selectedCount := 0
	for i := range m.prepStages {
		if m.prepStageSelected[i] {
			selectedCount++
		}
	}

	var statusIndicator string
	if selectedCount >= 1 {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " ")
	} else {
		statusIndicator = lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " ")
	}
	b.WriteString(statusIndicator + lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		fmt.Sprintf("%d of %d stages selected", selectedCount, len(m.prepStages))))
	if selectedCount == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Faint(true).Render(" — select at least 1"))
	}
	b.WriteString("\n\n")

	for i, stage := range m.prepStages {
		isSelected := m.prepStageSelected[i]
		isFocused := i == m.prepStageCursor
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).PaddingLeft(1)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).PaddingLeft(1)
		}
		b.WriteString(checkbox + nameStyle.Render(stage.Name))
		if len(stage.Description) > 0 {
			desc := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + stage.Description)
			b.WriteString(desc)
		}
		b.WriteString("\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingFiltering() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Filtering options") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure frequency filtering and resampling.") + "\n")
	b.WriteString(infoStyle.Render("Press Enter to edit a value, Esc to cancel editing.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	resampleVal := fmt.Sprintf("%d Hz", m.prepResample)
	if m.editingNumber && m.advancedCursor == 0 {
		resampleVal = m.numberBuffer + "█"
	}
	lFreqVal := fmt.Sprintf("%.1f Hz", m.prepLFreq)
	if m.editingNumber && m.advancedCursor == 1 {
		lFreqVal = m.numberBuffer + "█"
	}
	hFreqVal := fmt.Sprintf("%.1f Hz", m.prepHFreq)
	if m.editingNumber && m.advancedCursor == 2 {
		hFreqVal = m.numberBuffer + "█"
	}
	notchVal := fmt.Sprintf("%d Hz", m.prepNotch)
	if m.editingNumber && m.advancedCursor == 3 {
		notchVal = m.numberBuffer + "█"
	}
	lineFreqVal := fmt.Sprintf("%d Hz", m.prepLineFreq)
	if m.editingNumber && m.advancedCursor == 4 {
		lineFreqVal = m.numberBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Resample Freq", resampleVal, "Target sampling rate (0=disable)"},
		{"High-Pass Freq", lFreqVal, "Low frequency cutoff (Hz)"},
		{"Low-Pass Freq", hFreqVal, "High frequency cutoff (Hz)"},
		{"Notch Freq", notchVal, "Line noise notch filter (0=disable)"},
		{"Line Freq", lineFreqVal, "Power line frequency (50 or 60 Hz)"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor
		var labelStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingICA() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("ICA options") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure independent component analysis.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, Enter to edit values.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	methods := []string{"fastica", "infomax", "picard"}
	methodVal := methods[m.prepICAAlgorithm]

	compVal := fmt.Sprintf("%.2f", m.prepICAComp)
	if m.editingNumber && m.advancedCursor == 1 {
		compVal = m.numberBuffer + "█"
	}
	probThreshVal := fmt.Sprintf("%.2f", m.prepProbThresh)
	if m.editingNumber && m.advancedCursor == 2 {
		probThreshVal = m.numberBuffer + "█"
	}
	labelsVal := m.icaLabelsToKeep
	if labelsVal == "" {
		labelsVal = "(default)"
	}
	if m.editingText && m.editingTextField == textFieldIcaLabelsToKeep {
		labelsVal = m.textBuffer + "█"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use ICALabel", m.boolToOnOff(m.prepUseIcalabel), "Auto-label components"},
		{"ICA Method", methodVal, "fastica, infomax, or picard"},
		{"Components", compVal, "Number or variance fraction"},
		{"Prob Threshold", probThreshVal, "Label probability threshold"},
		{"Labels to Keep", labelsVal, "Comma-separated (e.g., brain,other)"},
		{"Keep MNE-BIDS Bads", m.boolToOnOff(m.prepKeepMnebidsBads), "Keep MNE-BIDS flagged components"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor
		var labelStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingEpochs() string {
	var b strings.Builder

	b.WriteString(styles.SectionTitleStyle.Render("Epoch options") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render("Configure epoch creation.") + "\n")
	b.WriteString(infoStyle.Render("Press Space to toggle, Enter to edit values.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	tminVal := fmt.Sprintf("%.1f s", m.prepEpochsTmin)
	if m.editingNumber && m.advancedCursor == 0 {
		tminVal = m.numberBuffer + "█"
	}
	tmaxVal := fmt.Sprintf("%.1f s", m.prepEpochsTmax)
	if m.editingNumber && m.advancedCursor == 1 {
		tmaxVal = m.numberBuffer + "█"
	}
	baselineVal := "(none)"
	if !m.prepEpochsNoBaseline {
		baselineVal = fmt.Sprintf("[%.1f, %.1f] s", m.prepEpochsBaselineStart, m.prepEpochsBaselineEnd)
	}
	rejectVal := fmt.Sprintf("%.0f µV", m.prepEpochsReject)
	if m.editingNumber && m.advancedCursor == 3 {
		rejectVal = m.numberBuffer + "█"
	}
	if m.prepEpochsReject == 0 {
		rejectVal = "(disabled)"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Tmin", tminVal, "Epoch start time (seconds)"},
		{"Tmax", tmaxVal, "Epoch end time (seconds)"},
		{"Baseline Correction", m.boolToOnOff(!m.prepEpochsNoBaseline), "Apply baseline correction"},
		{"Baseline Window", baselineVal, "Baseline time window"},
		{"Reject Threshold", rejectVal, "Peak-to-peak amplitude (µV, 0=disable)"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor
		var labelStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(opt.value))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}
