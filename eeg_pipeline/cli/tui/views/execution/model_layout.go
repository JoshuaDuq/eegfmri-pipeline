package execution

import (
	"strings"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func (m *Model) stopResourceMonitoringSafe() {
	if m.stopResourceChan != nil {
		select {
		case <-m.stopResourceChan:
			// Already closed
		default:
			close(m.stopResourceChan)
		}
		m.stopResourceChan = nil
	}
	// Note: resourceUpdateChan is closed by the monitoring goroutine when it exits
	// We just nil our reference to avoid sending to it
	m.resourceUpdateChan = nil
}

// IsDone reports whether the execution has reached a terminal
// state (success, failure or user cancellation).
func (m Model) IsDone() bool {
	return m.Status == StatusSuccess || m.Status == StatusFailed || m.Status == StatusCancelled
}

func (m Model) WasSuccessful() bool {
	return m.Status == StatusSuccess
}

// AddOutput appends a new log line to the model, applying the same
// cleaning, error tracking and viewport updates as streamed output.
func (m *Model) AddOutput(line string) {
	m.addLog(line)
}

// SetStatus forces the execution status; primarily useful in tests
// or higher‑level orchestration code.
func (m *Model) SetStatus(status Status) {
	m.Status = status
}

// SetSize updates the model’s notion of terminal width/height and
// recomputes the log viewport dimensions accordingly.
func (m *Model) SetSize(width, height int) {
	m.width = width
	m.height = height
	m.updateViewportSize()
}

func (m *Model) updateLayout() {
	const (
		minTwoColWidth  = 120
		minTwoColHeight = 28
		minSidebarWidth = 44
	)

	useTwo := m.width >= minTwoColWidth && m.height >= minTwoColHeight
	if !useTwo {
		m.useTwoCol = false
		m.leftWidth = m.width
		m.rightWidth = 0
		return
	}

	gap := m.columnGap
	if gap <= 0 {
		gap = 2
	}

	minLogWidth := styles.MinLogWidth + 8

	available := m.width - gap
	right := int(float64(available) * 0.38)
	if right < minSidebarWidth {
		right = minSidebarWidth
	}
	if right > available-minLogWidth {
		right = available - minLogWidth
	}
	left := available - right
	if left < minLogWidth || right < minSidebarWidth {
		m.useTwoCol = false
		m.leftWidth = m.width
		m.rightWidth = 0
		return
	}

	m.useTwoCol = true
	m.leftWidth = left
	m.rightWidth = right
}

func (m Model) contentWidth() int {
	if m.useTwoCol {
		return m.rightWidth
	}
	return m.width
}

func (m Model) panelWidth() int {
	base := m.contentWidth()
	if m.useTwoCol {
		base -= 4
	} else {
		base -= 10
	}
	if base < 30 {
		base = 30
	}
	return base
}

func (m Model) renderSidebarCard(content string) string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(1, 3)
	return cardStyle.Width(m.panelWidth()).Render(content)
}

func (m Model) copyLogToClipboard() tea.Cmd {
	return func() tea.Msg {
		logContent := "Command: " + m.Command + "\n"
		logContent += "Status: " + m.Status.String() + "\n"
		logContent += "Duration: " + m.getDuration().String() + "\n"
		logContent += "\n--- Log ---\n"
		logContent += strings.Join(m.OutputLines, "\n")

		// Cross-platform clipboard support
		executor.CopyToClipboard(logContent)
		return messages.LogCopiedMsg{}
	}
}
