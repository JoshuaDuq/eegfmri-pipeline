package wizard

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

func (m Model) renderTimeRange() string {
	var builder strings.Builder
	builder.WriteString(styles.RenderStepHeader("Time range", m.contentWidth) + "\n")

	if m.timeRangeShowsRestToggle() {
		isFocused := m.timeRangeCursorOnRestToggle() && m.editingRangeIdx == noRangeEditing
		builder.WriteString(m.renderConfigRow(
			"Resting State", m.boolToOnOff(m.prepTaskIsRest),
			"ON = no task events; power uses raw/log power",
			isFocused, defaultLabelWidth,
		) + "\n\n")
		if m.prepTaskIsRest {
			infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
			builder.WriteString(infoStyle.Render(
				"  The pipeline defaults to a full-epoch analysis window when no explicit time range is provided.",
			) + "\n\n")
		}
	}

	var tmin, tmax float64
	hasMetadata := false
	for _, subject := range m.subjects {
		if len(subject.EpochMetadata) > 0 {
			tmin = subject.EpochMetadata["tmin"]
			tmax = subject.EpochMetadata["tmax"]
			hasMetadata = true
			break
		}
	}

	if hasMetadata {
		metaStyle := lipgloss.NewStyle().Foreground(styles.Accent).Italic(true)
		builder.WriteString(fmt.Sprintf("  Epoch Length: %s to %s (seconds)\n\n",
			metaStyle.Render(fmt.Sprintf("%.2f", tmin)),
			metaStyle.Render(fmt.Sprintf("%.2f", tmax))))

		const diagramWidth = 50
		epochDuration := tmax - tmin
		if epochDuration > 0 {
			var diagram strings.Builder
			diagram.WriteString("  ")
			timelineChars := make([]rune, diagramWidth)
			for index := range timelineChars {
				timelineChars[index] = '─'
			}
			mapToPosition := func(value float64, inclusiveEnd bool) int {
				scaled := ((value - tmin) / epochDuration) * float64(diagramWidth-1)
				if inclusiveEnd {
					scaled += 0.999999
				}
				position := int(scaled)
				if position < 0 {
					return 0
				}
				if position >= diagramWidth {
					return diagramWidth - 1
				}
				return position
			}
			for _, timeRange := range m.TimeRanges {
				startPosition := mapToPosition(parseFloat(timeRange.Tmin, tmin), false)
				endPosition := mapToPosition(parseFloat(timeRange.Tmax, tmax), true)
				for index := startPosition; index <= endPosition; index++ {
					timelineChars[index] = '█'
				}
			}
			diagram.WriteString(lipgloss.NewStyle().Foreground(styles.Primary).Render(string(timelineChars)))
			diagram.WriteString("\n")
			leftLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmin))
			rightLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("%.1fs", tmax))
			padding := diagramWidth - lipgloss.Width(leftLabel) - lipgloss.Width(rightLabel)
			if padding < 1 {
				padding = 1
			}
			diagram.WriteString("  " + leftLabel + strings.Repeat(" ", padding) + rightLabel + "\n")
			builder.WriteString(diagram.String() + "\n")
		}
	}

	builder.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"  +: add  D: delete  Space: toggle/edit") + "\n\n")

	const nameWidth = 15
	const valueWidth = 10
	headerStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true)
	header := fmt.Sprintf("  %-*s %-*s %-*s", nameWidth, headerStyle.Render("Range name"),
		valueWidth, headerStyle.Render("Start (s)"), valueWidth, headerStyle.Render("End (s)"))
	builder.WriteString(header + "\n")
	builder.WriteString("  " + styles.RenderDivider(nameWidth+valueWidth*2+2) + "\n")

	if len(m.TimeRanges) == 0 {
		emptyStateStyle := lipgloss.NewStyle().Foreground(styles.Accent).Italic(true)
		builder.WriteString("\n")
		if m.prepTaskIsRest {
			builder.WriteString(emptyStateStyle.Render("  No explicit time ranges defined. Press [A] to add one.") + "\n")
		} else {
			builder.WriteString(emptyStateStyle.Render("  No time ranges defined. Press [A] to add one.") + "\n")
			builder.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(
				"  Note: 'baseline' is required for normalization (ERDS, log-ratio).") + "\n")
		}
	}

	for index, timeRange := range m.TimeRanges {
		rowCursor := index
		if m.timeRangeShowsRestToggle() {
			rowCursor++
		}
		isFocused := rowCursor == m.timeRangeCursor
		isEditing := index == m.editingRangeIdx
		cursor := "  "
		if isFocused {
			cursor = styles.RenderCursorOptional(m.CursorBlinkVisible())
		}
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		tminStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		tmaxStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		if isEditing {
			switch m.editingField {
			case 0:
				nameStyle = nameStyle.Foreground(styles.Accent).Underline(true)
			case 1:
				tminStyle = tminStyle.Foreground(styles.Accent).Underline(true)
			case 2:
				tmaxStyle = tmaxStyle.Foreground(styles.Accent).Underline(true)
			}
		} else if isFocused {
			nameStyle = nameStyle.Foreground(styles.Primary).Bold(true)
		}

		nameValue := timeRange.Name
		if nameValue == "" {
			nameValue = "(none)"
		}
		if isEditing && m.editingField == 0 {
			nameValue += "█"
		}
		tminValue := timeRange.Tmin
		if tminValue == "" {
			tminValue = "default"
		}
		if isEditing && m.editingField == 1 {
			tminValue += "█"
		}
		tmaxValue := timeRange.Tmax
		if tmaxValue == "" {
			tmaxValue = "default"
		}
		if isEditing && m.editingField == 2 {
			tmaxValue += "█"
		}
		row := fmt.Sprintf("%s%-*s %-*s %-*s", cursor,
			nameWidth, nameStyle.Render(nameValue),
			valueWidth, tminStyle.Render(tminValue),
			valueWidth, tmaxStyle.Render(tmaxValue))
		builder.WriteString(row + "\n")
	}

	return builder.String()
}
