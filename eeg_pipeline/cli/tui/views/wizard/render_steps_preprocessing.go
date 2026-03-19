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
	b.WriteString("\n")

	selectedCount := 0
	for i := range m.prepStages {
		if m.prepStageSelected[i] {
			selectedCount++
		}
	}

	b.WriteString("  " + styles.RenderStatusCount(selectedCount, len(m.prepStages), "stages"))
	b.WriteString("\n")

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
	b.WriteString("\n")

	b.WriteString(styles.RenderStepHeader("Filtering", m.contentWidth) + "\n\n")

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
	b.WriteString("\n")

	b.WriteString(styles.RenderStepHeader("ICA", m.contentWidth) + "\n\n")

	labelWidth := defaultLabelWidth

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

	icaOptions := []struct {
		label string
		value string
		hint  string
	}{
		{"ICA Method", methodVal, "fastica, infomax, or picard"},
		{"Components", compVal, "Number or variance fraction"},
	}
	labelOptions := []struct {
		label string
		value string
		hint  string
	}{
		{"Use ICALabel", m.boolToOnOff(m.prepUseIcalabel), "Auto-label components"},
		{"Prob Threshold", probThreshVal, "Label probability threshold"},
		{"Labels to Keep", labelsVal, "Comma-separated (e.g., brain,other)"},
		{"Keep MNE-BIDS Bads", m.boolToOnOff(m.prepKeepMnebidsBads), "Keep MNE-BIDS flagged components"},
	}

	// flatten with section offsets matching advancedCursor indices (0=UseICALabel,1=Method,2=Comp,3=ProbThresh,4=Labels,5=KeepBads)
	allOptions := []struct {
		label string
		value string
		hint  string
	}{
		{icaOptions[0].label, icaOptions[0].value, icaOptions[0].hint},
		{icaOptions[1].label, icaOptions[1].value, icaOptions[1].hint},
		{labelOptions[0].label, labelOptions[0].value, labelOptions[0].hint},
		{labelOptions[1].label, labelOptions[1].value, labelOptions[1].hint},
		{labelOptions[2].label, labelOptions[2].value, labelOptions[2].hint},
		{labelOptions[3].label, labelOptions[3].value, labelOptions[3].hint},
	}

	b.WriteString(styles.RenderPreviewSubHeader("DECOMPOSITION") + "\n")
	for i := 0; i < 2; i++ {
		opt := allOptions[i]
		isFocused := i+1 == m.advancedCursor // Method=cursor1, Comp=cursor2
		b.WriteString(m.renderConfigRow(opt.label, opt.value, opt.hint, isFocused, labelWidth) + "\n")
	}
	b.WriteString("\n" + styles.RenderPreviewSubHeader("LABELING") + "\n")
	for i := 2; i < len(allOptions); i++ {
		opt := allOptions[i]
		// cursor mapping: UseICALabel=0, Method=1, Comp=2, ProbThresh=3, Labels=4, KeepBads=5 — but original order in m.advancedCursor is 0-5
		// preserve original cursor indices from the original flat list
		cursorIdx := []int{0, 3, 4, 5}[i-2]
		isFocused := cursorIdx == m.advancedCursor
		b.WriteString(m.renderConfigRow(opt.label, opt.value, opt.hint, isFocused, labelWidth) + "\n")
	}

	return b.String()
}

func (m Model) renderConfigRow(label, value, hint string, isFocused bool, labelWidth int) string {
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
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
	return styles.RenderConfigLine(cursor, labelStyle.Render(label+":"), valueStyle.Render(value), hintStyle.Render(hint), labelWidth, m.contentWidth)
}

func (m Model) renderPreprocessingEpochs() string {
	var b strings.Builder
	b.WriteString("\n")

	b.WriteString(styles.RenderStepHeader("Epochs", m.contentWidth) + "\n\n")

	labelWidth := defaultLabelWidth

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

	b.WriteString(styles.RenderPreviewSubHeader("WINDOW") + "\n")
	windowOpts := []struct{ label, value, hint string }{
		{"Tmin", tminVal, "Epoch start time (seconds)"},
		{"Tmax", tmaxVal, "Epoch end time (seconds)"},
	}
	for i, opt := range windowOpts {
		b.WriteString(m.renderConfigRow(opt.label, opt.value, opt.hint, i == m.advancedCursor, labelWidth) + "\n")
	}

	b.WriteString("\n" + styles.RenderPreviewSubHeader("BASELINE") + "\n")
	baselineOpts := []struct{ label, value, hint string }{
		{"Correction", m.boolToOnOff(!m.prepEpochsNoBaseline), "Apply baseline correction"},
		{"Window", baselineVal, "Baseline time window"},
	}
	for i, opt := range baselineOpts {
		b.WriteString(m.renderConfigRow(opt.label, opt.value, opt.hint, i+2 == m.advancedCursor, labelWidth) + "\n")
	}

	b.WriteString("\n" + styles.RenderPreviewSubHeader("REJECTION") + "\n")
	b.WriteString(m.renderConfigRow("Reject Threshold", rejectVal, "Peak-to-peak amplitude (µV, 0=disable)", m.advancedCursor == 4, labelWidth) + "\n")

	return b.String()
}
