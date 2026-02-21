// Shared step-rendering helpers, scroll window, default config view, output paths, display helpers.
package wizard

import (
	"fmt"
	"strings"

	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

const (
	defaultLabelWidth     = 22
	defaultLabelWidthWide = 30
	configOverhead        = 17
)

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
	b.WriteString(infoStyle.Render(fmt.Sprintf("Using defaults for %s. Space to customize.", configType)) + "\n\n")

	labelWidth := defaultLabelWidth
	hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true)
	isFocused := m.advancedCursor == 0
	cursor := "  "
	if isFocused {
		cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
	}
	labelStyle := lipgloss.NewStyle().Foreground(styles.Text)
	if isFocused {
		labelStyle = labelStyle.Foreground(styles.Primary).Bold(true)
	}
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	b.WriteString(styles.RenderConfigLine(cursor, labelStyle.Render("Configuration:"), valueStyle.Render("defaults"), hintStyle.Render("Space to customize"), labelWidth, m.contentWidth) + "\n")
	return b.String()
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
