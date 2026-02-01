// fMRI raw-to-BIDS pipeline advanced configuration.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

func (m Model) renderFmriRawToBidsAdvancedConfig() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render("Advanced configuration") + "\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	if m.editingNumber || m.editingText {
		b.WriteString(infoStyle.Render("  Press Enter to confirm or Esc to cancel.") + "\n\n")
	} else {
		b.WriteString(infoStyle.Render("  Customize fMRI DICOM → BIDS conversion options.") + "\n")
		b.WriteString(infoStyle.Render("  Press Space to toggle/edit, Enter to proceed.") + "\n\n")
	}

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)

	sessionVal := m.fmriRawSession
	if m.editingText && m.editingTextField == textFieldFmriRawSession {
		sessionVal = m.textBuffer + "█"
	}

	restTaskVal := m.fmriRawRestTask
	if m.editingText && m.editingTextField == textFieldFmriRawRestTask {
		restTaskVal = m.textBuffer + "█"
	}

	dcm2niixPathVal := m.fmriRawDcm2niixPath
	if m.editingText && m.editingTextField == textFieldFmriRawDcm2niixPath {
		dcm2niixPathVal = m.textBuffer + "█"
	}

	dcm2niixArgsVal := m.fmriRawDcm2niixArgs
	if m.editingText && m.editingTextField == textFieldFmriRawDcm2niixArgs {
		dcm2niixArgsVal = m.textBuffer + "█"
	}

	onsetOffsetVal := fmt.Sprintf("%.3f", m.fmriRawOnsetOffsetS)
	if m.editingNumber && m.isCurrentlyEditing(optFmriRawOnsetOffsetS) {
		onsetOffsetVal = m.numberBuffer + "█"
	}

	dicomMode := "symlink"
	if m.fmriRawDicomModeIndex == 1 {
		dicomMode = "copy"
	} else if m.fmriRawDicomModeIndex == 2 {
		dicomMode = "skip"
	}

	granularity := "phases"
	if m.fmriRawEventGranularity == 1 {
		granularity = "trial"
	}

	onsetRef := "as_is"
	if m.fmriRawOnsetRefIndex == 1 {
		onsetRef = "first_iti_start"
	} else if m.fmriRawOnsetRefIndex == 2 {
		onsetRef = "first_stim_start"
	}

	options := []struct {
		label string
		value string
		hint  string
	}{
		{"Use Defaults", m.boolToOnOff(m.useDefaultAdvanced), "Skip customization"},
		{"Session", sessionVal, "Optional (e.g., 01)"},
		{"Rest Task", restTaskVal, "Task label for resting-state"},
		{"Include Rest", m.boolToOnOff(m.fmriRawIncludeRest), "Convert resting-state series"},
		{"Include Fieldmaps", m.boolToOnOff(m.fmriRawIncludeFieldmaps), "Convert fieldmaps and set IntendedFor"},
		{"DICOM Mode", dicomMode, "sourcedata handling"},
		{"Overwrite", m.boolToOnOff(m.fmriRawOverwrite), "Replace existing BIDS files"},
		{"Create Events", m.boolToOnOff(m.fmriRawCreateEvents), "From PsychoPy TrialSummary.csv"},
		{"Event Granularity", granularity, "stimulation: phases vs trial"},
		{"Onset Reference", onsetRef, "zero onsets within each run"},
		{"Onset Offset (s)", onsetOffsetVal, "additive shift"},
		{"dcm2niix Path", dcm2niixPathVal, "Optional override"},
		{"dcm2niix Args", dcm2niixArgsVal, "Comma-separated extra args"},
	}

	for i, opt := range options {
		isFocused := i == m.advancedCursor

		var labelStyle, valueStyle lipgloss.Style
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(labelWidth)
		} else {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
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
		if strings.TrimSpace(displayVal) == "" {
			displayVal = "(none)"
		}

		b.WriteString(cursor + labelStyle.Render(opt.label+":") + " " + valueStyle.Render(displayVal))
		b.WriteString("  " + hintStyle.Render(opt.hint) + "\n")
	}

	return b.String()
}

