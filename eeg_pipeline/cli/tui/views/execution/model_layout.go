package execution

import (
	"strings"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
)

// Layout sizing, lifecycle helpers, and clipboard export.

const executionMinVisibleLogLines = 4

func (m Model) compactLogPriorityMode() bool {
	return m.width < styles.MinTerminalWidth || m.height < styles.MinTerminalHeight
}

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
	m.useTwoCol = false
	m.leftWidth = m.width
	m.rightWidth = 0
}

func (m Model) contentWidth() int {
	return m.width
}

func (m Model) panelWidth() int {
	base := m.contentWidth() - 10
	if base < 30 {
		base = 30
	}
	return base
}

func renderedLineCount(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(s, "\n") + 1
}

func clampViewportDimension(available, preferredMin, hardMax int) int {
	if available < 1 {
		return 1
	}
	if available > hardMax {
		return hardMax
	}
	if available < preferredMin {
		return available
	}
	return available
}

func (m Model) stackedReservedHeight() int {
	reserved := renderedLineCount(m.renderHeader()) + renderedLineCount(m.renderFooter())
	reserved += styles.ExecLogTitleLines + styles.ExecViewportBorderLines
	if m.copyMode {
		reserved += styles.ExecCopyModeBannerLines
	}
	for _, section := range m.stackedSupplementarySections() {
		reserved += renderedLineCount(section)
	}
	return reserved
}

func (m Model) stackedSupplementarySections() []string {
	if m.compactLogPriorityMode() {
		return nil
	}

	available := m.height -
		renderedLineCount(m.renderHeader()) -
		renderedLineCount(m.renderFooter()) -
		styles.ExecLogTitleLines -
		styles.ExecViewportBorderLines -
		executionMinVisibleLogLines - 1
	if m.copyMode {
		available -= styles.ExecCopyModeBannerLines
	}
	if available <= 0 {
		return nil
	}

	if m.IsDone() {
		summary := m.renderCompletionSummary()
		if renderedLineCount(summary) <= available {
			return []string{summary}
		}
		return nil
	}

	info := m.renderInfoPanel()
	progress := m.renderSidebarCard(m.renderProgressSection())
	infoLines := renderedLineCount(info)
	progressLines := renderedLineCount(progress)

	if infoLines+progressLines <= available {
		return []string{info, progress}
	}
	if progressLines <= available {
		return []string{progress}
	}
	if infoLines <= available {
		return []string{info}
	}
	return nil
}

func (m Model) sidebarInnerWidth() int {
	w := m.panelWidth() - 6
	if w < 20 {
		w = 20
	}
	return w
}

func (m Model) renderSidebarCard(content string) string {
	return styles.PanelStyle.Width(m.panelWidth()).Render(content)
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
