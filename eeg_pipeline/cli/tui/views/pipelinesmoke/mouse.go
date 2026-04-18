package pipelinesmoke

import (
	"regexp"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var pipelineSmokeMouseANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripPipelineSmokeANSI(s string) string {
	return pipelineSmokeMouseANSIPattern.ReplaceAllString(s, "")
}

func (m Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	switch msg.Button {
	case tea.MouseButtonWheelUp:
		if m.cursor > 0 {
			m.cursor--
		} else {
			m.cursor = len(smokeItems) - 1
		}
		return m, nil
	case tea.MouseButtonWheelDown:
		if m.cursor < len(smokeItems)-1 {
			m.cursor++
		} else {
			m.cursor = 0
		}
		return m, nil
	}

	if msg.Action != tea.MouseActionMotion &&
		(msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft) {
		return m, nil
	}

	line := m.viewLineAt(msg.Y)
	if strings.TrimSpace(stripPipelineSmokeANSI(line)) == "" {
		return m, nil
	}

	if idx := m.matchSmokeItemLine(line); idx >= 0 {
		m.cursor = idx
		if msg.Action == tea.MouseActionPress {
			m.toggleCursor()
		}
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

func (m Model) matchSmokeItemLine(line string) int {
	normalized := strings.TrimSpace(stripPipelineSmokeANSI(line))
	for i, item := range smokeItems {
		if strings.Contains(normalized, item.Name) {
			return i
		}
	}
	return -1
}
