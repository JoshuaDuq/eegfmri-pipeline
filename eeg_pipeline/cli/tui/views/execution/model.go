package execution

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////

type Status int

const (
	StatusPending Status = iota
	StatusRunning
	StatusSuccess
	StatusFailed
	StatusCancelled
)

func (s Status) String() string {
	names := []string{"Pending", "Running", "Success", "Failed", "Cancelled"}
	if int(s) < len(names) {
		return names[s]
	}
	return "Unknown"
}

type CloudStage int

const (
	StageSyncing CloudStage = iota
	StageRunning
	StagePulling
	StageDone
)

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	Command        string
	Status         Status
	Progress       float64
	OutputLines    []string
	MaxOutputLines int
	StartTime      time.Time
	EndTime        time.Time
	ExitCode       int
	Error          error
	IsCloud        bool
	CloudStage     CloudStage
	RepoRoot       string

	// Step-level progress tracking
	CurrentSubject   string
	CurrentOperation string
	SubjectCurrent   int
	SubjectTotal     int
	OperationCurrent int
	OperationTotal   int

	// Error tracking
	ErrorLines         []string
	LogTruncated       bool
	LogTruncatedCount  int
	MalformedJSONCount int
	LastModule         string

	// Scrollable log viewport
	logViewport viewport.Model
	logReady    bool

	// Copy mode - disables mouse capture for text selection
	copyMode       bool
	copyModeNotice string

	// Search/filter mode
	searchMode    bool
	searchQuery   string
	searchMatches int

	// Internal
	cmd        *exec.Cmd
	outputChan chan string
	doneChan   chan error

	width  int
	height int
	ticker int
}

///////////////////////////////////////////////////////////////////
// Constructors
///////////////////////////////////////////////////////////////////

func New(command string) Model {
	vp := viewport.New(80, styles.DefaultLogHeight)
	vp.Style = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(0, 1)
	vp.MouseWheelEnabled = true
	vp.MouseWheelDelta = styles.MouseWheelScrollLines

	return Model{
		Command:        command,
		Status:         StatusPending,
		Progress:       0,
		OutputLines:    []string{},
		MaxOutputLines: styles.MaxScrollbackLines,
		CloudStage:     StageSyncing,
		StartTime:      time.Now(),
		width:          80,
		height:         24,
		logViewport:    vp,
		logReady:       false,
		RepoRoot:       "",
	}
}

func NewWithRoot(command string, repoRoot string) Model {
	m := New(command)
	m.RepoRoot = repoRoot
	return m
}

///////////////////////////////////////////////////////////////////
// Messages
///////////////////////////////////////////////////////////////////

type CommandStartedMsg struct {
	OutputChan chan string
	DoneChan   chan error
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.tick(),
		m.startCommandAsync(),
	)
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*time.Duration(styles.TickIntervalMs), func(t time.Time) tea.Msg {
		return messages.TickMsg{}
	})
}

func (m Model) startCommandAsync() tea.Cmd {
	return func() tea.Msg {
		parts := strings.Fields(m.Command)
		if len(parts) == 0 {
			return messages.CommandDoneMsg{ExitCode: 1}
		}

		var args []string
		if parts[0] == "eeg-pipeline" {
			args = append([]string{"-m", "eeg_pipeline"}, parts[1:]...)
		} else {
			args = parts
		}

		hasInfo := false
		for _, arg := range args {
			if arg == "info" {
				hasInfo = true
				break
			}
		}
		if !hasInfo {
			args = append(args, "--progress-json")
		}

		pyCmd := executor.GetPythonCommand(m.RepoRoot)
		cmd := exec.Command(pyCmd, args...)
		cmd.Dir = m.RepoRoot
		cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			cmd = exec.Command("python3", args...)
			cmd.Dir = m.RepoRoot
			cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")
			stdout, err = cmd.StdoutPipe()
			if err != nil {
				return messages.CommandDoneMsg{ExitCode: 1, Error: err}
			}
		}

		stderr, _ := cmd.StderrPipe()

		if err := cmd.Start(); err != nil {
			return messages.CommandDoneMsg{ExitCode: 1, Error: err}
		}

		outputChan := make(chan string, styles.LogBufferChannels)
		doneChan := make(chan error, 1)

		go func() {
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				outputChan <- scanner.Text()
			}
		}()

		go func() {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				outputChan <- scanner.Text()
			}
		}()

		go func() {
			doneChan <- cmd.Wait()
		}()

		return CommandStartedMsg{OutputChan: outputChan, DoneChan: doneChan}
	}
}

func (m Model) listenForOutput() tea.Cmd {
	if m.outputChan == nil || m.doneChan == nil {
		return nil
	}

	return func() tea.Msg {
		select {
		case line, ok := <-m.outputChan:
			if !ok {
				return nil
			}
			return messages.StreamOutputMsg{Line: line}
		case err := <-m.doneChan:
			exitCode := 0
			success := true
			if err != nil {
				success = false
				if exitErr, ok := err.(*exec.ExitError); ok {
					exitCode = exitErr.ExitCode()
				} else {
					exitCode = 1
				}
			}
			return messages.CommandDoneMsg{ExitCode: exitCode, Success: success}
		}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case messages.TickMsg:
		m.ticker++
		if m.Status == StatusPending || m.Status == StatusRunning {
			m.Status = StatusRunning
			if m.StartTime.IsZero() {
				m.StartTime = time.Now()
			}
			return m, m.tick()
		}
		return m, nil

	case CommandStartedMsg:
		m.outputChan = msg.OutputChan
		m.doneChan = msg.DoneChan
		m.Status = StatusRunning
		return m, m.listenForOutput()

	case messages.StreamOutputMsg:
		m.processOutputLine(msg.Line)
		m.logViewport.SetContent(strings.Join(m.OutputLines, "\n"))
		m.logViewport.GotoBottom()
		return m, m.listenForOutput()

	case messages.LogCopiedMsg:
		// Show a brief notice that log was copied
		m.copyModeNotice = "✓ Log copied to clipboard!"
		m.addLog(lipgloss.NewStyle().Foreground(styles.Success).Render("✓ Log copied to clipboard"))
		return m, nil

	case messages.CommandDoneMsg:
		m.EndTime = time.Now()
		m.ExitCode = msg.ExitCode
		m.Progress = 1.0
		m.CloudStage = StageDone
		if msg.Success {
			m.Status = StatusSuccess
			m.addLog(styles.CheckMark + " Completed successfully")
		} else {
			m.Status = StatusFailed
			errorSummary := m.extractErrorSummary()
			if errorSummary != "" {
				m.addLog(fmt.Sprintf("%s Failed with exit code %d - %s", styles.CrossMark, msg.ExitCode, errorSummary))
			} else {
				m.addLog(fmt.Sprintf("%s Failed with exit code %d", styles.CrossMark, msg.ExitCode))
			}
		}
		return m, nil

	case tea.KeyMsg:
		// Handle search mode first
		if m.searchMode {
			switch msg.String() {
			case "esc":
				m.searchMode = false
				m.searchQuery = ""
				m.updateLogViewport()
			case "enter":
				m.searchMode = false
				// Keep filter active
			case "backspace":
				if len(m.searchQuery) > 0 {
					m.searchQuery = m.searchQuery[:len(m.searchQuery)-1]
					m.updateLogViewport()
				}
			default:
				if len(msg.String()) == 1 {
					m.searchQuery += msg.String()
					m.updateLogViewport()
				}
			}
			return m, nil
		}

		// Check if copy mode is active
		if m.copyMode {
			switch msg.String() {
			case "m", "esc", "q":
				// Exit copy mode, re-enable mouse
				m.copyMode = false
				m.copyModeNotice = ""
				return m, tea.EnableMouseCellMotion
			case "c":
				// Copy while in copy mode
				return m, m.copyLogToClipboard()
			}
			// In copy mode, don't handle other keys to allow selection
			return m, nil
		}

		switch msg.String() {
		case "/":
			// Enter search mode
			m.searchMode = true
			m.searchQuery = ""
			return m, nil
		case "m":
			// Toggle copy mode - disable mouse to allow text selection
			m.copyMode = true
			m.copyModeNotice = "COPY MODE: Select text with mouse, then Cmd+C to copy. Press M or Esc to exit."
			return m, tea.DisableMouse
		case "ctrl+c":
			if m.IsDone() {
				return m, m.copyLogToClipboard()
			}
			m.Status = StatusCancelled
			m.EndTime = time.Now()
			if m.cmd != nil && m.cmd.Process != nil {
				m.cmd.Process.Kill()
			}
			return m, nil
		case "c":
			return m, m.copyLogToClipboard()
		case "j", "down":
			m.logViewport.LineDown(styles.ScrollStepSize)
			return m, nil
		case "k", "up":
			m.logViewport.LineUp(styles.ScrollStepSize)
			return m, nil
		case "g":
			m.logViewport.GotoTop()
			return m, nil
		case "G":
			m.logViewport.GotoBottom()
			return m, nil
		}

	case tea.MouseMsg:
		// In copy mode, don't capture mouse events
		if m.copyMode {
			return m, nil
		}
		var cmd tea.Cmd
		m.logViewport, cmd = m.logViewport.Update(msg)
		return m, cmd

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		logHeight := m.height - 20
		if logHeight < styles.MinLogHeight {
			logHeight = styles.MinLogHeight
		}
		if logHeight > styles.MaxLogHeight {
			logHeight = styles.MaxLogHeight
		}

		logWidth := m.width - 8
		if logWidth < styles.MinLogWidth {
			logWidth = styles.MinLogWidth
		}
		if logWidth > styles.MaxLogWidth {
			logWidth = styles.MaxLogWidth
		}

		m.logViewport.Width = logWidth
		m.logViewport.Height = logHeight
		m.logReady = true
	}

	return m, nil
}

///////////////////////////////////////////////////////////////////
// Output Processing
///////////////////////////////////////////////////////////////////

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
		}
		if json.Unmarshal([]byte(line), &event) == nil {
			switch event.Event {
			case "start":
				m.SubjectTotal = event.TotalSubjects
				if m.SubjectTotal == 0 && len(event.Subjects) > 0 {
					m.SubjectTotal = len(event.Subjects)
				}
				m.addLog(fmt.Sprintf("Starting: %d subjects", m.SubjectTotal))
			case "subject_start":
				if m.SubjectCurrent < m.SubjectTotal || m.SubjectTotal == 0 {
					m.SubjectCurrent++
				}
				m.CurrentSubject = event.Subject
				m.CurrentOperation = ""
				m.OperationCurrent = 0
				m.OperationTotal = 0
				if m.SubjectTotal > 0 {
					m.Progress = clampProgress(float64(m.SubjectCurrent-1) / float64(m.SubjectTotal))
				}
				m.addLog(fmt.Sprintf("[%d/%d] %s", m.SubjectCurrent, m.SubjectTotal, event.Subject))
			case "progress":
				m.CurrentOperation = event.Step
				m.OperationCurrent = event.Current
				m.OperationTotal = event.Total

				// For pipelines with per-subject iteration
				if m.SubjectTotal > 1 && event.Total > 0 {
					if event.Pct > 0 {
						stepPct := float64(event.Pct) / 100.0
						subjProgress := float64(m.SubjectCurrent-1) / float64(m.SubjectTotal)
						stepContrib := stepPct / float64(m.SubjectTotal)
						m.Progress = clampProgress(subjProgress + stepContrib)
					} else {
						subjProgress := float64(m.SubjectCurrent-1) / float64(m.SubjectTotal)
						stepContrib := (float64(event.Current) / float64(event.Total)) / float64(m.SubjectTotal)
						m.Progress = clampProgress(subjProgress + stepContrib)
					}
				} else if event.Total > 0 {
					// For pipelines without per-subject iteration (e.g., decoding, utilities)
					// Progress is based purely on step progress
					m.Progress = clampProgress(float64(event.Current) / float64(event.Total))
				} else if event.Pct > 0 {
					// Fallback to explicit percentage
					m.Progress = clampProgress(float64(event.Pct) / 100.0)
				}

				// Log step changes for visibility
				if event.Step != "" && event.Current > 0 {
					m.addLog(fmt.Sprintf("  → %s (%d/%d)", event.Step, event.Current, event.Total))
				}
			case "subject_done":
				if m.SubjectTotal > 0 {
					m.Progress = clampProgress(float64(m.SubjectCurrent) / float64(m.SubjectTotal))
				}
			case "log":
				m.addLog(event.Message)
			case "complete":
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

	if len(m.OutputLines) > m.MaxOutputLines {
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

func (m *Model) updateLogViewport() {
	if m.searchQuery == "" {
		// No search - show all lines
		m.logViewport.SetContent(strings.Join(m.OutputLines, "\n"))
		m.searchMatches = 0
	} else {
		// Filter lines matching search query
		var filtered []string
		query := strings.ToLower(m.searchQuery)
		highlightStyle := lipgloss.NewStyle().Background(styles.Accent).Foreground(lipgloss.Color("#000000"))

		for _, line := range m.OutputLines {
			if strings.Contains(strings.ToLower(line), query) {
				// Highlight matches
				highlighted := line
				idx := strings.Index(strings.ToLower(line), query)
				if idx >= 0 {
					before := line[:idx]
					match := line[idx : idx+len(m.searchQuery)]
					after := line[idx+len(m.searchQuery):]
					highlighted = before + highlightStyle.Render(match) + after
				}
				filtered = append(filtered, highlighted)
			}
		}

		m.searchMatches = len(filtered)
		if len(filtered) == 0 {
			filtered = append(filtered, lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("No matches found for: "+m.searchQuery))
		}
		m.logViewport.SetContent(strings.Join(filtered, "\n"))
	}
	m.logViewport.GotoBottom()
}

func (m Model) extractErrorSummary() string {
	if len(m.ErrorLines) == 0 {
		return ""
	}

	lastError := m.ErrorLines[len(m.ErrorLines)-1]

	parts := strings.Split(lastError, " - ")
	if len(parts) >= 2 {
		return strings.TrimSpace(strings.Join(parts[1:], " - "))
	}

	if len(lastError) > 80 {
		return lastError[:77] + "..."
	}
	return lastError
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	var b strings.Builder

	// Header
	b.WriteString(m.renderHeader())
	b.WriteString("\n\n")

	// Info Panel
	b.WriteString(m.renderInfoPanel())
	b.WriteString("\n\n")

	// Show completion summary if done
	if m.IsDone() {
		b.WriteString(m.renderCompletionSummary())
		b.WriteString("\n")
	}

	// Progress Section
	b.WriteString(m.renderProgressSection())
	b.WriteString("\n")

	// Log Section
	b.WriteString(m.renderLogSection())
	b.WriteString("\n")

	// Footer
	b.WriteString(m.renderFooter())

	return b.String()
}

func (m Model) renderCompletionSummary() string {
	var b strings.Builder

	// Determine status style and animation
	var icon, statusText string
	var statusColor lipgloss.Color
	var borderColor lipgloss.Color

	switch m.Status {
	case StatusSuccess:
		successFrames := []string{"✓", "✓", "★", "✓"}
		icon = successFrames[m.ticker%len(successFrames)]
		statusText = "COMPLETED SUCCESSFULLY"
		statusColor = styles.Success
		borderColor = styles.Success
	case StatusFailed:
		failFrames := []string{"✗", "✗", "⚠", "✗"}
		icon = failFrames[m.ticker%len(failFrames)]
		statusText = "EXECUTION FAILED"
		statusColor = styles.Error
		borderColor = styles.Error
	case StatusCancelled:
		icon = "⊘"
		statusText = "EXECUTION CANCELLED"
		statusColor = styles.Warning
		borderColor = styles.Warning
	default:
		return ""
	}

	// Status line with icon
	statusLine := lipgloss.NewStyle().
		Bold(true).
		Foreground(statusColor).
		Render(icon + "  " + statusText)
	b.WriteString(statusLine + "\n")

	// Divider line
	divWidth := 40
	div := lipgloss.NewStyle().Foreground(statusColor).Render(strings.Repeat("─", divWidth))
	b.WriteString(div + "\n\n")

	// Stats in a clean layout
	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(14)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	// Duration
	duration := m.getDuration()
	durationStr := formatDuration(duration)
	b.WriteString(labelStyle.Render("Duration:") + valueStyle.Render(durationStr) + "\n")

	// Subjects processed
	if m.SubjectTotal > 0 {
		subjProgress := float64(m.SubjectCurrent) / float64(m.SubjectTotal)
		subjInfo := fmt.Sprintf("%d of %d (%.0f%%)", m.SubjectCurrent, m.SubjectTotal, subjProgress*100)
		b.WriteString(labelStyle.Render("Subjects:") + valueStyle.Render(subjInfo) + "\n")
	}

	// Exit code for failures
	if m.Status == StatusFailed {
		exitStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
		b.WriteString(labelStyle.Render("Exit Code:") + exitStyle.Render(fmt.Sprintf("%d", m.ExitCode)) + "\n")
	}

	// Error count with visual indicator
	if len(m.ErrorLines) > 0 {
		errStyle := lipgloss.NewStyle().Foreground(styles.Warning)
		errText := fmt.Sprintf("%d detected", len(m.ErrorLines))
		if len(m.ErrorLines) > 5 {
			errText += " (scroll log to view)"
		}
		b.WriteString(labelStyle.Render("Errors:") + errStyle.Render(errText) + "\n")
	}

	// Log lines processed
	logCount := fmt.Sprintf("%d lines", len(m.OutputLines))
	if m.LogTruncated {
		logCount += lipgloss.NewStyle().Foreground(styles.Warning).Render(" (truncated)")
	}
	b.WriteString(labelStyle.Render("Log:") + valueStyle.Render(logCount) + "\n")

	// Add quick action hint
	switch m.Status {
	case StatusSuccess:
		b.WriteString("\n")
		hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
		b.WriteString(hintStyle.Render("Press Enter to return to menu, or C to copy log"))
	case StatusFailed:
		b.WriteString("\n")
		hintStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
		b.WriteString(hintStyle.Render("Press R to retry, C to copy log, or Enter to return"))
	}

	// Summary card styling
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(1, 2).
		Width(m.width - 10)

	return cardStyle.Render(b.String())
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1f seconds", d.Seconds())
	} else if d < time.Hour {
		mins := int(d.Minutes())
		secs := int(d.Seconds()) % 60
		return fmt.Sprintf("%d min %d sec", mins, secs)
	}
	hours := int(d.Hours())
	mins := int(d.Minutes()) % 60
	return fmt.Sprintf("%d hr %d min", hours, mins)
}

func (m Model) renderHeader() string {
	title := " PIPELINE EXECUTION "
	if m.IsCloud {
		title = " CLOUD EXECUTION "
	}

	borderColor := styles.Secondary
	switch m.Status {
	case StatusRunning:
		borderColor = styles.Accent
	case StatusSuccess:
		borderColor = styles.Success
	case StatusFailed:
		borderColor = styles.Error
	case StatusCancelled:
		borderColor = styles.Warning
	}

	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Text).
		Background(borderColor).
		Padding(0, 2).
		Render(title)

	return lipgloss.PlaceHorizontal(m.width-4, lipgloss.Center, header)
}

func (m Model) renderInfoPanel() string {
	info := strings.Builder{}

	cmdLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(8).Render("Cmd:")
	cmdValue := lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render(m.Command)
	info.WriteString(cmdLabel + cmdValue + "\n")

	if m.StartTime.Unix() > 0 {
		duration := m.getDuration()
		timeLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(8).Render("Time:")
		timeValue := lipgloss.NewStyle().Foreground(styles.Text).Render(duration.Round(time.Second).String())
		info.WriteString(timeLabel + timeValue)
	}

	return styles.CardStyle.Width(m.width - 10).Render(info.String())
}

func (m Model) renderProgressSection() string {
	var b strings.Builder

	// Section header with animated icon
	var progressIcon string
	var iconStyle lipgloss.Style
	switch m.Status {
	case StatusSuccess:
		progressIcon = styles.CheckMark
		iconStyle = lipgloss.NewStyle().Foreground(styles.Success)
	case StatusFailed:
		progressIcon = styles.CrossMark
		iconStyle = lipgloss.NewStyle().Foreground(styles.Error)
	default:
		progressIcon = []string{"◔", "◑", "◕", "●"}[m.ticker%4]
		iconStyle = lipgloss.NewStyle().Foreground(styles.Accent)
	}

	b.WriteString(iconStyle.Render(progressIcon) + " " + styles.SectionTitleStyle.Render(" PROGRESS ") + "\n\n")

	// Overall progress with animated gradient bar
	progressLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("Overall")
	b.WriteString("  " + progressLabel + m.renderAnimatedProgressBar(m.Progress, m.width-22) + "\n")

	// Subject counter with visual indicator
	if m.SubjectTotal > 0 {
		subjectProgress := float64(m.SubjectCurrent) / float64(m.SubjectTotal)

		// Create a mini visual representation
		var subjectIcons string
		for i := 0; i < m.SubjectTotal && i < 10; i++ {
			if i < m.SubjectCurrent {
				subjectIcons += lipgloss.NewStyle().Foreground(styles.Success).Render("●")
			} else if i == m.SubjectCurrent {
				subjectIcons += lipgloss.NewStyle().Foreground(styles.Accent).Render("○")
			} else {
				subjectIcons += lipgloss.NewStyle().Foreground(styles.Secondary).Render("○")
			}
		}
		if m.SubjectTotal > 10 {
			subjectIcons += lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf(" +%d", m.SubjectTotal-10))
		}

		subjectText := fmt.Sprintf("Subject %d/%d  ", m.SubjectCurrent, m.SubjectTotal)
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).PaddingLeft(12).Render(subjectText))
		b.WriteString(subjectIcons)
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("  %.0f%%", subjectProgress*100)) + "\n")
	}

	// Current step progress with enhanced styling
	if m.CurrentSubject != "" || m.CurrentOperation != "" {
		b.WriteString("\n")
		stepLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("Step")

		if m.OperationTotal > 0 {
			stepProgress := float64(m.OperationCurrent) / float64(m.OperationTotal)
			b.WriteString("  " + stepLabel + m.renderMiniProgressBar(stepProgress, m.width-32))

			// Operation name with step counter
			opText := fmt.Sprintf(" %s (%d/%d)", m.CurrentOperation, m.OperationCurrent, m.OperationTotal)
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(opText) + "\n")
		} else {
			// Just show subject and operation
			b.WriteString("  " + stepLabel)
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.CurrentSubject))
			if m.CurrentOperation != "" {
				b.WriteString(" " + lipgloss.NewStyle().Foreground(styles.TextDim).Render("→ "+m.CurrentOperation))
			}
			b.WriteString("\n")
		}
	}

	// Cloud stages (if cloud mode)
	if m.IsCloud {
		b.WriteString("\n  " + m.renderCloudStages() + "\n")
	}

	// Status Badge with enhanced styling
	b.WriteString("\n  " + m.renderStatus() + "\n")

	return b.String()
}

func (m Model) renderLogSection() string {
	var b strings.Builder

	// Copy mode banner
	if m.copyMode {
		copyBanner := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Padding(0, 2).
			Render("📋 COPY MODE: Select text with mouse, then Cmd+C. Press M or Esc to exit.")
		b.WriteString(copyBanner + "\n\n")
	}

	// Search mode input
	if m.searchMode || m.searchQuery != "" {
		searchIcon := lipgloss.NewStyle().Foreground(styles.Accent).Render("🔍 ")
		inputStyle := lipgloss.NewStyle().
			Background(styles.Secondary).
			Foreground(styles.Text).
			Padding(0, 1)

		query := m.searchQuery
		if m.searchMode {
			query += "▎" // Cursor indicator
		}

		searchInput := searchIcon + inputStyle.Render(query)

		// Show match count if we have results
		if m.searchQuery != "" {
			matchInfo := lipgloss.NewStyle().Foreground(styles.Muted).Render(
				fmt.Sprintf("  %d matches", m.searchMatches))
			searchInput += matchInfo
		}

		b.WriteString(searchInput + "\n\n")
	}

	logHeader := styles.SectionTitleStyle.Render(" LOG ")
	if len(m.OutputLines) > 0 {
		scrollPct := 0
		if m.logViewport.TotalLineCount() > 0 {
			scrollPct = int(float64(m.logViewport.YOffset+m.logViewport.Height) / float64(m.logViewport.TotalLineCount()) * 100)
			if scrollPct > 100 {
				scrollPct = 100
			}
		}

		indicator := lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf(" [%d lines | %d%%]", len(m.OutputLines), scrollPct))
		logHeader += indicator
	}

	b.WriteString(logHeader + "\n")
	b.WriteString(m.logViewport.View())

	return b.String()
}

func (m Model) renderCloudStages() string {
	stages := []struct {
		stage CloudStage
		name  string
	}{
		{StageSyncing, "Sync"},
		{StageRunning, "Run"},
		{StagePulling, "Pull"},
	}

	var parts []string
	for _, s := range stages {
		style := lipgloss.NewStyle().Foreground(styles.Muted)
		marker := "[ ]"

		if s.stage < m.CloudStage {
			style = style.Foreground(styles.Success)
			marker = "[" + styles.CheckMark + "]"
		} else if s.stage == m.CloudStage && m.Status == StatusRunning {
			style = style.Foreground(styles.Accent).Bold(true)
			frames := []string{"-", "\\", "|", "/"}
			marker = "[" + frames[m.ticker%len(frames)] + "]"
		} else if m.CloudStage == StageDone {
			style = style.Foreground(styles.Success)
			marker = "[" + styles.CheckMark + "]"
		}

		parts = append(parts, style.Render(marker+" "+s.name))
	}

	return strings.Join(parts, "   ")
}

func (m Model) renderAnimatedProgressBar(p float64, width int) string {
	if width < styles.MinProgressBarWidth {
		width = styles.MinProgressBarWidth
	}
	if width > styles.MaxProgressBarWidth {
		width = styles.MaxProgressBarWidth
	}

	filled := int(p * float64(width))
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}

	// Create gradient effect with color transition
	var bar string
	for i := 0; i < filled; i++ {
		pct := float64(i) / float64(width)
		if pct < 0.33 {
			bar += lipgloss.NewStyle().Foreground(styles.Accent).Render("█")
		} else if pct < 0.66 {
			bar += lipgloss.NewStyle().Foreground(styles.Primary).Render("█")
		} else {
			bar += lipgloss.NewStyle().Foreground(styles.Success).Render("█")
		}
	}

	// Animated leading edge when not complete
	if filled < width && filled > 0 && m.Status == StatusRunning {
		leadChars := []string{"▓", "▒", "░"}
		leadIdx := m.ticker % len(leadChars)
		bar += lipgloss.NewStyle().Foreground(styles.Accent).Render(leadChars[leadIdx])
		bar += lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("░", width-filled-1))
	} else {
		bar += lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("░", width-filled))
	}

	pct := lipgloss.NewStyle().Bold(true).Foreground(styles.Primary).Render(fmt.Sprintf(" %3.0f%%", p*100))

	return bar + pct
}

func (m Model) renderMiniProgressBar(p float64, width int) string {
	if width < styles.MinProgressBarWidth {
		width = styles.MinProgressBarWidth
	}

	filled := int(p * float64(width))
	bar := lipgloss.NewStyle().Foreground(styles.Accent).Render(strings.Repeat("━", filled))
	empty := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", width-filled))
	return bar + empty
}

func (m Model) renderStatus() string {
	style := lipgloss.NewStyle().Bold(true).Padding(0, 1)
	switch m.Status {
	case StatusRunning:
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		spinner := frames[m.ticker%len(frames)]
		return style.Background(styles.Accent).Foreground(lipgloss.Color("#000000")).Render(spinner + " RUNNING ")
	case StatusSuccess:
		return style.Background(styles.Success).Foreground(lipgloss.Color("#000000")).Render(styles.CheckMark + " SUCCESS ")
	case StatusFailed:
		return style.Background(styles.Error).Foreground(lipgloss.Color("#FFFFFF")).Render(styles.CrossMark + " FAILED ")
	case StatusCancelled:
		return style.Background(styles.Muted).Foreground(lipgloss.Color("#FFFFFF")).Render(" CANCELLED ")
	default:
		return style.Background(styles.Secondary).Foreground(styles.Text).Render(" PENDING ")
	}
}

func (m Model) getDuration() time.Duration {
	if m.Status == StatusRunning {
		return time.Since(m.StartTime)
	}
	if m.EndTime.Unix() > 0 {
		return m.EndTime.Sub(m.StartTime)
	}
	return 0
}

func (m Model) renderFooter() string {
	var hints []string

	// Special footer for search mode
	if m.searchMode {
		hints = []string{
			styles.RenderKeyHint("Esc", "Cancel Search"),
			styles.RenderKeyHint("Enter", "Apply Filter"),
			lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render("Type to search..."),
		}
	} else if m.copyMode {
		// Footer for copy mode
		hints = []string{
			styles.RenderKeyHint("M/Esc", "Exit Copy Mode"),
			styles.RenderKeyHint("C", "Copy All"),
			lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render("Select text with mouse, then Cmd+C"),
		}
	} else if m.IsDone() {
		hints = []string{
			styles.RenderKeyHint("Enter", "Return to Menu"),
			styles.RenderKeyHint("C", "Copy Log"),
			styles.RenderKeyHint("/", "Search"),
			styles.RenderKeyHint("M", "Select Mode"),
			styles.RenderKeyHint("R", "Retry"),
		}
	} else {
		hints = []string{
			styles.RenderKeyHint("Ctrl+C", "Cancel"),
			styles.RenderKeyHint("/", "Search"),
			styles.RenderKeyHint("M", "Select Mode"),
			styles.RenderKeyHint("J/K", "Scroll"),
		}
	}

	separator := "   "
	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
}

///////////////////////////////////////////////////////////////////
// Public Methods
///////////////////////////////////////////////////////////////////

func (m Model) IsDone() bool {
	return m.Status == StatusSuccess || m.Status == StatusFailed || m.Status == StatusCancelled
}

func (m Model) WasSuccessful() bool {
	return m.Status == StatusSuccess
}

func (m *Model) AddOutput(line string) {
	m.addLog(line)
}

func (m *Model) SetStatus(status Status) {
	m.Status = status
}

func (m Model) copyLogToClipboard() tea.Cmd {
	return func() tea.Msg {
		logContent := "Command: " + m.Command + "\n"
		logContent += "Status: " + m.Status.String() + "\n"
		logContent += "Duration: " + m.getDuration().String() + "\n"
		logContent += "\n--- Log ---\n"
		logContent += strings.Join(m.OutputLines, "\n")

		cmd := exec.Command("pbcopy")
		cmd.Stdin = strings.NewReader(logContent)
		cmd.Run()

		return messages.LogCopiedMsg{}
	}
}
