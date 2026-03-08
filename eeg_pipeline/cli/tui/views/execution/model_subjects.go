package execution

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

type subjectState string

const (
	subjectPending subjectState = "pending"
	subjectRunning subjectState = "running"
	subjectDone    subjectState = "done"
	subjectFailed  subjectState = "failed"
)

func (m *Model) resetSubjects(subjects []string) {
	m.SubjectOrder = nil
	m.SubjectStatuses = make(map[string]string)
	m.FailedSubjects = nil
	m.SubjectCurrent = 0
	m.CurrentSubject = ""
	m.CurrentOperation = ""
	m.OperationCurrent = 0
	m.OperationTotal = 0
	m.SubjectStartTime = time.Time{}
	m.SubjectDurations = nil
	m.EstimatedRemaining = 0

	for _, subject := range subjects {
		m.trackSubject(subject)
	}
}

func (m *Model) trackSubject(subject string) {
	if subject == "" {
		return
	}
	if m.subjectIndex(subject) >= 0 {
		if _, ok := m.SubjectStatuses[subject]; !ok {
			m.SubjectStatuses[subject] = string(subjectPending)
		}
		return
	}

	m.SubjectOrder = append(m.SubjectOrder, subject)
	if _, ok := m.SubjectStatuses[subject]; !ok {
		m.SubjectStatuses[subject] = string(subjectPending)
	}
}

func (m Model) subjectIndex(subject string) int {
	for idx, candidate := range m.SubjectOrder {
		if candidate == subject {
			return idx
		}
	}
	return -1
}

func (m Model) subjectStatus(subject string) subjectState {
	status, ok := m.SubjectStatuses[subject]
	if !ok || status == "" {
		return subjectPending
	}
	return subjectState(status)
}

func (m *Model) beginSubject(subject string) {
	if subject == "" {
		return
	}
	if m.CurrentSubject != "" && !m.SubjectStartTime.IsZero() {
		m.finishCurrentSubject(subjectDone)
	}

	if m.SubjectCurrent < m.SubjectTotal || m.SubjectTotal == 0 {
		m.SubjectCurrent++
	}

	m.trackSubject(subject)
	m.CurrentSubject = subject
	m.SubjectStartTime = time.Now()
	m.SubjectStatuses[subject] = string(subjectRunning)
	m.CurrentOperation = ""
	m.OperationCurrent = 0
	m.OperationTotal = 0
}

func (m *Model) finishCurrentSubject(status subjectState) {
	subject := m.CurrentSubject
	if subject == "" {
		return
	}

	if !m.SubjectStartTime.IsZero() && status == subjectDone {
		m.SubjectDurations = append(m.SubjectDurations, time.Since(m.SubjectStartTime))
	}

	m.SubjectStatuses[subject] = string(status)
	if status == subjectFailed {
		m.recordFailedSubject(subject)
	}

	m.CurrentSubject = ""
	m.CurrentOperation = ""
	m.OperationCurrent = 0
	m.OperationTotal = 0
	m.SubjectStartTime = time.Time{}
}

func (m *Model) recordFailedSubject(subject string) {
	for _, failed := range m.FailedSubjects {
		if failed == subject {
			return
		}
	}
	m.FailedSubjects = append(m.FailedSubjects, subject)
}

func (m Model) renderSubjectSummary(maxWidth int) string {
	if m.SubjectTotal == 0 {
		return ""
	}

	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	subjectCount := m.SubjectCurrent
	if subjectCount > m.SubjectTotal {
		subjectCount = m.SubjectTotal
	}
	pct := float64(subjectCount) / float64(m.SubjectTotal)

	line := dimStyle.Render("Subjects ") +
		valueStyle.Render(fmt.Sprintf("%d/%d", subjectCount, m.SubjectTotal)) +
		dimStyle.Render(" ") +
		valueStyle.Render(fmt.Sprintf("%.0f%%", pct*100))

	return styles.TruncateLine(line, maxWidth)
}

func (m Model) renderSubjectLane(maxWidth int) string {
	if len(m.SubjectOrder) == 0 || maxWidth <= 0 {
		return ""
	}

	maxVisible := len(m.SubjectOrder)
	if maxVisible > 6 {
		maxVisible = 6
	}

	for maxVisible > 0 {
		start, end := m.visibleSubjectWindow(maxVisible)
		line := m.renderSubjectWindow(start, end)
		if lipgloss.Width(line) <= maxWidth || maxVisible == 1 {
			return styles.TruncateLine(line, maxWidth)
		}
		maxVisible--
	}

	return ""
}

func (m Model) visibleSubjectWindow(maxVisible int) (int, int) {
	total := len(m.SubjectOrder)
	if total <= maxVisible {
		return 0, total
	}

	anchor := m.subjectAnchorIndex()
	start := anchor - maxVisible/2
	if start < 0 {
		start = 0
	}
	end := start + maxVisible
	if end > total {
		end = total
		start = end - maxVisible
	}
	return start, end
}

func (m Model) subjectAnchorIndex() int {
	if idx := m.subjectIndex(m.CurrentSubject); idx >= 0 {
		return idx
	}
	if m.SubjectCurrent <= 0 {
		return 0
	}
	if m.SubjectCurrent >= len(m.SubjectOrder) {
		return len(m.SubjectOrder) - 1
	}
	return m.SubjectCurrent
}

func (m Model) renderSubjectWindow(start, end int) string {
	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	parts := make([]string, 0, end-start+2)

	if start > 0 {
		parts = append(parts, dimStyle.Render(fmt.Sprintf("+%d", start)))
	}
	for _, subject := range m.SubjectOrder[start:end] {
		parts = append(parts, m.renderSubjectChip(subject))
	}
	if end < len(m.SubjectOrder) {
		parts = append(parts, dimStyle.Render(fmt.Sprintf("+%d", len(m.SubjectOrder)-end)))
	}

	return strings.Join(parts, "  ")
}

func (m Model) renderSubjectChip(subject string) string {
	label := subject
	state := m.subjectStatus(subject)

	var icon string
	var style lipgloss.Style

	switch state {
	case subjectRunning:
		icon = "◉"
		style = lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	case subjectDone:
		icon = styles.CheckMark
		style = lipgloss.NewStyle().Foreground(styles.Success)
	case subjectFailed:
		icon = styles.CrossMark
		style = lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
	default:
		icon = styles.PendingMark
		style = lipgloss.NewStyle().Foreground(styles.TextDim)
	}

	return style.Render(icon + " " + label)
}

func (m Model) renderFailureLane(maxWidth int) string {
	if len(m.FailedSubjects) == 0 {
		return ""
	}

	label := lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render("Failed")
	value := lipgloss.NewStyle().Foreground(styles.Error).Render(strings.Join(m.FailedSubjects, ", "))
	return styles.TruncateLine(label+": "+value, maxWidth)
}
