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

	b.WriteString(styles.RenderStepHeader("Preprocessing stages", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).
		Render("Space: toggle  A: all  N: none") + "\n")

	selectedCount := 0
	for i := range m.prepStages {
		if m.prepStageSelected[i] {
			selectedCount++
		}
	}

	b.WriteString(styles.RenderStatusCount(selectedCount, len(m.prepStages), "stages"))
	b.WriteString("\n\n")

	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	for i, stage := range m.prepStages {
		isSelected := m.prepStageSelected[i]
		isFocused := i == m.prepStageCursor
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursor()
		}
		checkbox := styles.RenderCheckbox(isSelected, isFocused)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		if isFocused {
			nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		}
		line := cursor + checkbox + " " + nameStyle.Render(stage.Name)
		if len(stage.Description) > 0 {
			line += "  " + descStyle.Render(stage.Description)
		}
		b.WriteString(styles.TruncateLine(line, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingFiltering() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Filtering", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).
		Render("Enter: edit  Esc: cancel") + "\n\n")

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
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(opt.label+":"), valueStyle.Render(opt.value), hintStyle.Render(opt.hint), labelWidth, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingICA() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("ICA", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).
		Render("Space: toggle  Enter: edit") + "\n\n")

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
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(opt.label+":"), valueStyle.Render(opt.value), hintStyle.Render(opt.hint), labelWidth, m.contentWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderPreprocessingEpochs() string {
	var b strings.Builder

	b.WriteString(styles.RenderStepHeader("Epochs", m.contentWidth) + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2).
		Render("Space: toggle  Enter: edit") + "\n\n")

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
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text)
		}
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render(opt.label+":"), valueStyle.Render(opt.value), hintStyle.Render(opt.hint), labelWidth, m.contentWidth) + "\n")
	}

	return b.String()
}
