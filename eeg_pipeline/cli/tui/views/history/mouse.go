package history

import (
	"regexp"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var historyMouseANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripHistoryANSI(s string) string {
	return historyMouseANSIPattern.ReplaceAllString(s, "")
}

func (m Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.loading || m.loadError != nil {
		return m, nil
	}

	switch msg.Button {
	case tea.MouseButtonWheelUp:
		if m.cursor > 0 {
			m.cursor--
		}
		return m, nil
	case tea.MouseButtonWheelDown:
		if m.cursor < len(m.records)-1 {
			m.cursor++
		}
		return m, nil
	}

	if msg.Action != tea.MouseActionMotion &&
		(msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft) {
		return m, nil
	}

	line := m.viewLineAt(msg.Y)
	if strings.TrimSpace(stripHistoryANSI(line)) == "" {
		return m, nil
	}

	if idx := m.matchRecordLine(line); idx >= 0 {
		m.cursor = idx
	}

	return m, nil
}

func (m Model) viewLineAt(y int) string {
	if y < 0 {
		return ""
	}
	lines := strings.Split(m.View(), "\n")
	if y >= len(lines) {
		return ""
	}
	return lines[y]
}

func (m Model) matchRecordLine(line string) int {
	normalized := strings.TrimSpace(stripHistoryANSI(line))
	if normalized == "" {
		return -1
	}

	for i, record := range m.records {
		if i >= maxVisibleHistoryRecords {
			break
		}
		parts := []string{
			record.Pipeline,
			record.Mode,
			FormatDurationSeconds(record.Duration),
			FormatTimeAgo(record.StartTime),
		}
		matched := true
		for _, part := range parts {
			if part != "" && !strings.Contains(normalized, part) {
				matched = false
				break
			}
		}
		if matched {
			return i
		}
	}

	return -1
}
