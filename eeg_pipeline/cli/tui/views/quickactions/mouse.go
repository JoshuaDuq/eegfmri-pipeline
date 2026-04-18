package quickactions

import (
	"regexp"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var quickActionsMouseANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripQuickActionsANSI(s string) string {
	return quickActionsMouseANSIPattern.ReplaceAllString(s, "")
}

func (m Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if !m.Visible {
		return m, nil
	}

	switch msg.Button {
	case tea.MouseButtonWheelUp:
		if m.cursor > 0 {
			m.cursor--
		} else {
			m.cursor = len(quickActions) - 1
		}
		return m, nil
	case tea.MouseButtonWheelDown:
		if m.cursor < len(quickActions)-1 {
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
	if strings.TrimSpace(stripQuickActionsANSI(line)) == "" {
		return m, nil
	}

	if idx := m.matchActionLine(line); idx >= 0 {
		m.cursor = idx
		if msg.Action == tea.MouseActionPress {
			m.SelectedAction = quickActions[idx].Type
			m.Done = true
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

func (m Model) matchActionLine(line string) int {
	normalized := stripQuickActionsANSI(line)
	for i, action := range quickActions {
		if strings.Contains(normalized, action.Name) {
			return i
		}
	}
	return -1
}
