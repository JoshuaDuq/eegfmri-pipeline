package execution

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/eeg-pipeline/tui/styles"
)

// Log ingestion, cleanup, and viewport maintenance.

func (m *Model) updateViewportSize() {
	m.updateLayout()

	logHeight := clampViewportDimension(m.height-m.stackedReservedHeight(), styles.MinLogHeight, styles.MaxLogHeight)
	logWidth := clampViewportDimension(m.width-8, styles.MinLogWidth, styles.MaxLogWidth)

	m.logViewport.Width = logWidth
	m.logViewport.Height = logHeight
	m.updateLogViewport()
}

///////////////////////////////////////////////////////////////////
// Output Processing
///////////////////////////////////////////////////////////////////

// processOutputLine ingests a single stdout/stderr line from the
// running process, updating progress state when the line contains a
// structured JSON event or appending it to the textual log stream.
func (m *Model) processOutputLine(line string) {
	if strings.HasPrefix(line, "{") {
		var event struct {
			Event         string   `json:"event"`
			Subject       string   `json:"subject"`
			Step          string   `json:"step"`
			Current       int      `json:"current"`
			Total         int      `json:"total"`
			Pct           int      `json:"pct"`
			TotalSubjects int      `json:"total_subjects"`
			Subjects      []string `json:"subjects"`
			Message       string   `json:"message"`
			CPU           float64  `json:"cpu"`
			Memory        float64  `json:"memory"`
			Epoch         string   `json:"epoch"`
		}
		if json.Unmarshal([]byte(line), &event) == nil {
			m.CPUUsage = event.CPU
			m.MemoryUsage = event.Memory
			if event.Epoch != "" {
				m.EpochInfo = event.Epoch
			}
			switch event.Event {
			case "start":
				m.SubjectTotal = event.TotalSubjects
				if m.SubjectTotal == 0 && len(event.Subjects) > 0 {
					m.SubjectTotal = len(event.Subjects)
				}
				m.resetSubjects(event.Subjects)
				m.addLog(fmt.Sprintf("Starting: %d subjects", m.SubjectTotal))
			case "subject_start":
				m.beginSubject(event.Subject)
				if m.SubjectTotal > 0 {
					m.Progress = clampProgress(float64(m.SubjectCurrent-1) / float64(m.SubjectTotal))
				}
				m.calculateETA()
				m.addLog(fmt.Sprintf("[%d/%d] %s", m.SubjectCurrent, m.SubjectTotal, event.Subject))
			case "progress":
				m.CurrentOperation = event.Step
				m.OperationCurrent = event.Current
				m.OperationTotal = event.Total

				hasStepProgress := event.Total > 0
				stepFraction := 0.0
				if hasStepProgress {
					stepFraction = float64(event.Current) / float64(event.Total)
				}
				useSubjectProgress := m.SubjectTotal > 0 && m.SubjectCurrent > 0

				if useSubjectProgress && hasStepProgress {
					subjProgress := float64(m.SubjectCurrent-1) / float64(m.SubjectTotal)
					stepContrib := stepFraction / float64(m.SubjectTotal)
					m.Progress = clampProgress(subjProgress + stepContrib)
				} else if hasStepProgress {
					m.Progress = clampProgress(stepFraction)
				} else if event.Pct > 0 {
					m.Progress = clampProgress(float64(event.Pct) / 100.0)
				}

				if event.Step != "" && event.Current > 0 {
					m.addLog(fmt.Sprintf("  → %s (%d/%d)", event.Step, event.Current, event.Total))
				}
			case "subject_done":
				m.finishCurrentSubject(subjectDone)
				m.calculateETA()
				if m.SubjectTotal > 0 {
					m.Progress = clampProgress(float64(m.SubjectCurrent) / float64(m.SubjectTotal))
				}
			case "subject_failed":
				m.finishCurrentSubject(subjectFailed)
				m.calculateETA()
				if m.SubjectTotal > 0 {
					m.Progress = clampProgress(float64(m.SubjectCurrent) / float64(m.SubjectTotal))
				}
			case "log":
				m.addLog(event.Message)
			case "complete":
				if m.SubjectTotal > 0 && m.SubjectCurrent == 0 {
					m.SubjectCurrent = m.SubjectTotal
				}
				m.Progress = 1.0
			}
			return
		} else {
			m.MalformedJSONCount++
			if m.MalformedJSONCount == 5 {
				m.addLog(lipgloss.NewStyle().Foreground(styles.Warning).Render(
					styles.WarningMark + " Progress reporting may be degraded (malformed JSON events)"))
			}
		}
	}

	m.addLog(line)
}

func clampProgress(p float64) float64 {
	if p < 0.0 {
		return 0.0
	}
	if p > 1.0 {
		return 1.0
	}
	return p
}

var ansiRegex = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|[\x00-\x1f\x7f]|\033\[[0-9;]*m|38;2;[0-9;]+m|\[0m`)

func stripANSI(s string) string {
	return ansiRegex.ReplaceAllString(s, "")
}

func cleanLogLine(line string) string {
	line = stripANSI(line)

	parts := strings.Split(line, " - ")

	if len(parts) >= 3 {
		timestamp := strings.TrimSpace(parts[0])

		if len(timestamp) >= 10 && timestamp[4] == '-' && timestamp[7] == '-' && strings.Contains(timestamp, ":") {
			timeParts := strings.Split(timestamp, " ")
			timeOnly := ""
			if len(timeParts) >= 2 {
				timeOnly = timeParts[1]
			}

			result := ""
			if timeOnly != "" {
				result = lipgloss.NewStyle().Foreground(styles.Muted).Render("["+timeOnly+"]") + " "
			}

			startIndex := 1
			if len(parts) >= 4 {
				startIndex = 2
			}

			remaining := strings.Join(parts[startIndex:], " - ")
			result += remaining

			return result
		}
	}

	return line
}

func (m *Model) addLog(line string) {
	origLine := line
	isError := strings.Contains(strings.ToUpper(origLine), "ERROR") ||
		strings.Contains(strings.ToLower(origLine), "exception") ||
		strings.Contains(strings.ToUpper(origLine), "CRITICAL") ||
		strings.Contains(strings.ToUpper(origLine), "FAILED") ||
		strings.Contains(origLine, "Traceback")

	line = cleanLogLine(line)
	m.OutputLines = append(m.OutputLines, line)

	if isError {
		m.ErrorLines = append(m.ErrorLines, line)
		if len(m.ErrorLines) > styles.MaxRecentErrors {
			m.ErrorLines = m.ErrorLines[1:]
		}
	}

	if m.MaxOutputLines > 0 && len(m.OutputLines) > m.MaxOutputLines {
		m.OutputLines = m.OutputLines[1:]
		if !m.LogTruncated {
			m.LogTruncated = true
			truncWarning := lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark + " Log buffer full - oldest lines discarded")
			m.OutputLines = append(m.OutputLines, truncWarning)
		}
		m.LogTruncatedCount++
	}

	m.updateLogViewport()
}

// wrapText wraps text to fit within the specified width, breaking at word boundaries
// If a word is too long, it will be broken at the width limit
func wrapText(text string, width int) []string {
	if width < 1 {
		return []string{text}
	}

	// Strip ANSI codes to get actual text length
	stripped := stripANSI(text)
	if len(stripped) <= width {
		return []string{text}
	}

	var lines []string
	words := strings.Fields(text)
	if len(words) == 0 {
		// Handle case where text has no words but exceeds width (e.g., long URL)
		for len(stripped) > width {
			lines = append(lines, text[:width])
			text = text[width:]
			stripped = stripANSI(text)
		}
		if len(text) > 0 {
			lines = append(lines, text)
		}
		return lines
	}

	currentLine := words[0]
	for i := 1; i < len(words); i++ {
		testLine := currentLine + " " + words[i]
		if len(stripANSI(testLine)) <= width {
			currentLine = testLine
		} else {
			// Current line is full, add it and start new line
			lines = append(lines, currentLine)
			// Check if the word itself is too long
			wordStripped := stripANSI(words[i])
			if len(wordStripped) > width {
				// Break the long word
				remainingWord := words[i]
				for len(stripANSI(remainingWord)) > width {
					lines = append(lines, remainingWord[:width])
					remainingWord = remainingWord[width:]
				}
				currentLine = remainingWord
			} else {
				currentLine = words[i]
			}
		}
	}
	if currentLine != "" {
		lines = append(lines, currentLine)
	}

	return lines
}

// updateLogViewport rebuilds the viewport contents from the current
// log buffer, applying soft line-wrapping to match the available width.
func (m *Model) updateLogViewport() {
	var filtered []string

	// Calculate available width: viewport width minus border (2) and padding (2)
	contentWidth := m.logViewport.Width - 4
	if contentWidth < 1 {
		contentWidth = 1
	}

	for _, line := range m.OutputLines {
		// Wrap line to fit viewport width
		wrappedLines := wrapText(line, contentWidth)
		filtered = append(filtered, wrappedLines...)
	}

	if len(filtered) == 0 {
		filtered = append(filtered, lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("Log output will appear here..."))
	}

	m.logViewport.SetContent(strings.Join(filtered, "\n"))
	m.logViewport.GotoBottom()
}
