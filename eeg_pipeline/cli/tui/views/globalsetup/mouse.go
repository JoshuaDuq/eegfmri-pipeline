package globalsetup

import (
	"regexp"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

var globalSetupMouseANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripGlobalSetupANSI(s string) string {
	return globalSetupMouseANSIPattern.ReplaceAllString(s, "")
}

func (m *Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.editingText {
		return m, nil
	}

	switch msg.Button {
	case tea.MouseButtonWheelUp:
		m.moveCursor(-1)
		return m, nil
	case tea.MouseButtonWheelDown:
		m.moveCursor(1)
		return m, nil
	}

	if msg.Action != tea.MouseActionMotion &&
		(msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft) {
		return m, nil
	}

	line := m.viewLineAt(msg.Y)
	normalized := strings.TrimSpace(stripGlobalSetupANSI(line))
	if normalized == "" {
		return m, nil
	}

	if msg.Y == 1 {
		if sectionIndex := m.matchSectionTabLine(normalized, msg.X); sectionIndex >= 0 {
			m.setSection(sectionIndex)
			return m, nil
		}
	}

	if fieldIndex := m.matchFieldLine(msg.Y); fieldIndex >= 0 {
		m.fieldCursor = fieldIndex
		if msg.Action == tea.MouseActionPress {
			return m.activateSelection()
		}
	}

	return m, nil
}

func (m *Model) setSection(index int) {
	if index == m.sectionIndex {
		return
	}
	m.sectionIndex = index
	m.resetSectionState()
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

func (m Model) matchSectionTabLine(line string, x int) int {
	for i, sec := range m.sections {
		label := strings.ToUpper(sec.label)
		start := strings.Index(line, label)
		if start < 0 {
			continue
		}
		if x >= start && x < start+len(label) {
			return i
		}
	}
	return -1
}

func (m Model) matchFieldLine(y int) int {
	start := m.fieldRowStartY()
	if y < start {
		return -1
	}
	fields := m.sectionFields(m.sections[m.sectionIndex].key)
	if y >= start+len(fields) {
		return -1
	}
	return y - start
}

func (m Model) fieldRowStartY() int {
	start := 3
	if m.sections[m.sectionIndex].description != "" {
		start += 2
	} else {
		start++
	}
	return start
}
