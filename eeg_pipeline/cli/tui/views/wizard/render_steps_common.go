// Shared step-rendering helpers, scroll window, default config view, output paths, display helpers.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	"github.com/charmbracelet/lipgloss"
)

const (
	defaultLabelWidth     = 22
	defaultLabelWidthWide = 30
	configOverhead        = 10
)

// renderSectionAccent returns a static accent for section titles (clean, scientific).
func (m Model) renderSectionAccent() string {
	return lipgloss.NewStyle().
		Foreground(styles.Accent).
		Bold(true).
		Render(styles.SelectedMark + " ")
}

// calculateScrollWindow returns the visible line range for scrolling.
func calculateScrollWindow(totalLines, offset, effectiveHeight, overhead int) (startLine, endLine int, showIndicators bool) {
	maxLines := effectiveHeight - overhead
	if maxLines < minVisibleLines {
		maxLines = minVisibleLines
	}
	if totalLines <= maxLines {
		return 0, totalLines, false
	}
	showIndicators = true
	startLine = offset
	if startLine < 0 {
		startLine = 0
	}
	if startLine > totalLines-maxLines {
		startLine = totalLines - maxLines
	}
	endLine = startLine + maxLines
	return startLine, endLine, showIndicators
}

// renderDefaultConfigView renders the default configuration view when useDefaultAdvanced is true.
func (m Model) renderDefaultConfigView(configType string) string {
	var b strings.Builder
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).PaddingLeft(2)
	b.WriteString(infoStyle.Render(fmt.Sprintf("Default configuration will be used for %s.", configType)) + "\n")
	b.WriteString(infoStyle.Render("Press Space to customize settings.") + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	isFocused := m.advancedCursor == 0
	cursor := "  "
	if isFocused {
		cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
	}
	labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(labelWidth)
	if isFocused {
		labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
	}
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	b.WriteString(cursor + labelStyle.Render("configuration:") + " " + valueStyle.Render("using defaults") + "  " + hintStyle.Render("Space to customize") + "\n")
	tipStyle := lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).PaddingLeft(4)
	b.WriteString("\n" + tipStyle.Render("tip: In Custom mode, sections are collapsible for easier navigation.") + "\n")
	return b.String()
}

// getExpectedOutputPaths returns the expected output directories for the current pipeline.
func (m Model) getExpectedOutputPaths() []string {
	base := "derivatives/"
	switch m.Pipeline {
	case types.PipelinePreprocessing:
		return []string{base + "preprocessed/eeg/sub-XX/"}
	case types.PipelineFeatures:
		return []string{base + "sub-XX/eeg/features/"}
	case types.PipelineBehavior:
		return []string{base + "sub-XX/eeg/stats/", base + "group/eeg/stats/"}
	case types.PipelineML:
		return []string{base + "machine_learning/"}
	case types.PipelinePlotting:
		return []string{base + "sub-XX/eeg/plots/"}
	case types.PipelineFmri:
		return []string{base + "preprocessed/fmri/fmriprep/sub-XX/"}
	case types.PipelineFmriAnalysis:
		mode := ""
		if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
			mode = m.modeOptions[m.modeIndex]
		}
		switch mode {
		case "trial-signatures":
			if m.fmriTrialSigMethodIndex%2 == 1 {
				return []string{base + "sub-XX/fmri/lss/task-*/contrast-*/"}
			}
			return []string{base + "sub-XX/fmri/beta_series/task-*/contrast-*/"}
		default:
			return []string{base + "sub-XX/fmri/first_level/task-*/contrast-*/"}
		}
	case types.PipelineFmriRawToBIDS:
		return []string{"bids_output/fmri/sub-XX/"}
	case types.PipelineRawToBIDS:
		return []string{"bids_output/eeg/sub-XX/eeg/"}
	default:
		return []string{base}
	}
}

func parseFloat(s string, defaultVal float64) float64 {
	if s == "" || s == "default" || s == "none" {
		return defaultVal
	}
	var val float64
	fmt.Sscanf(s, "%f", &val)
	return val
}

func (m Model) boolToOnOff(val bool) string {
	if val {
		return "ON"
	}
	return "OFF"
}

func (m Model) rngSeedDisplay() string {
	if m.rngSeed == 0 {
		return "0 (default)"
	}
	return fmt.Sprintf("%d", m.rngSeed)
}

func (m Model) selectedConnectivityDisplay() string {
	var selected []string
	for i, sel := range m.connectivityMeasures {
		if sel && i < len(connectivityMeasures) {
			selected = append(selected, connectivityMeasures[i].Key)
		}
	}
	if len(selected) == 0 {
		return "(none)"
	}
	return strings.Join(selected, ", ")
}

func (m Model) selectedDirectedConnectivityDisplay() string {
	var selected []string
	for i, sel := range m.directedConnMeasures {
		if sel && i < len(directedConnectivityMeasures) {
			selected = append(selected, directedConnectivityMeasures[i].Key)
		}
	}
	if len(selected) == 0 {
		return "(none)"
	}
	return strings.Join(selected, ", ")
}
