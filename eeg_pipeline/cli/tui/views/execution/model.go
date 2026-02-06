package execution

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// File layout notes:
// - `model.go`: execution model types, constructor, process start, Tea update loop.
// - `model_paths.go`: output-path inference and result-folder opening.
// - `model_logs.go`: log ingest/cleanup/viewport updates.
// - `model_render.go`: status/metrics/log rendering.
// - `model_resources.go`: runtime resource monitoring helpers.
// - `model_layout.go`: shared layout/state helpers and clipboard copy.

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

// Model holds all state required to render and manage a single
// pipeline execution view, including process status, log output,
// resource usage and UI layout.
type Model struct {
	Command        string
	Status         Status
	Progress       float64
	OutputLines    []string
	MaxOutputLines int
	StartTime      time.Time
	EndTime        time.Time
	ExitCode       int
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

	// ETA and timing
	SubjectStartTime   time.Time         // When current subject started
	SubjectDurations   []time.Duration   // History of completed subject durations
	EstimatedRemaining time.Duration     // Calculated ETA
	SubjectStatuses    map[string]string // Per-subject status: "pending", "running", "done", "failed"
	FailedSubjects     []string          // List of failed subject IDs

	// Resource metrics
	CPUUsage      float64
	MemoryUsage   float64
	CPUCoreUsages []float64 // Per-core CPU usage percentages
	NumCPUCores   int       // Total number of CPU cores
	EpochInfo     string

	// Error tracking
	ErrorLines         []string
	LogTruncated       bool
	LogTruncatedCount  int
	MalformedJSONCount int

	// Scrollable log viewport
	logViewport viewport.Model

	// Copy mode - disables mouse capture for text selection
	copyMode bool

	// Internal
	cmd                *exec.Cmd
	cancel             context.CancelFunc
	outputChan         chan string
	doneChan           chan error
	resourceUpdateChan chan messages.ResourceUpdateMsg // Channel for resource updates
	stopResourceChan   chan struct{}                   // Signal to stop resource monitoring

	width      int
	height     int
	useTwoCol  bool
	leftWidth  int
	rightWidth int
	columnGap  int
	ticker     int
	animQueue  animation.Queue
}

///////////////////////////////////////////////////////////////////
// Constructors
///////////////////////////////////////////////////////////////////

// New creates a new execution model for the given shell command,
// initializing log viewport and sensible default view settings.
func New(command string) Model {
	vp := viewport.New(80, styles.DefaultLogHeight)
	vp.Style = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(0, 1)
	vp.MouseWheelEnabled = true
	vp.MouseWheelDelta = styles.MouseWheelScrollLines

	m := Model{
		Command:          command,
		Status:           StatusPending,
		Progress:         0,
		OutputLines:      []string{},
		MaxOutputLines:   styles.MaxScrollbackLines,
		CloudStage:       StageSyncing,
		StartTime:        time.Now(),
		width:            80,
		height:           24,
		columnGap:        2,
		logViewport:      vp,
		RepoRoot:         "",
		SubjectDurations: []time.Duration{},
		SubjectStatuses:  make(map[string]string),
		FailedSubjects:   []string{},
	}
	m.updateLayout()
	m.updateViewportSize()
	m.animQueue.Push(animation.ProgressPulseLoop())
	return m
}

// NewWithRoot creates a new execution model and associates it with
// a specific repository root used for resolving Python entry points
// and output locations.
func NewWithRoot(command string, repoRoot string) Model {
	m := New(command)
	m.RepoRoot = repoRoot
	return m
}

///////////////////////////////////////////////////////////////////
// Messages
///////////////////////////////////////////////////////////////////

// CommandStartedMsg is emitted once the subprocess has started and
// exposes the running command and its output channels to the model.
type CommandStartedMsg struct {
	Cmd        *exec.Cmd
	OutputChan chan string
	DoneChan   chan error
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

// Init implements tea.Model and starts the periodic ticker used
// to drive progress and elapsed time updates.
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.tick(),
	)
}

// tick schedules the next TickMsg using the configured interval.
func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*time.Duration(styles.TickIntervalMs), func(t time.Time) tea.Msg {
		return messages.TickMsg{}
	})
}

// Start creates and launches the underlying Python process for the
// configured command and wires up stdout/stderr streaming.
func (m *Model) Start() tea.Cmd {
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel

	return func() tea.Msg {
		parts, err := splitShellWords(m.Command)
		if err != nil {
			parts = strings.Fields(m.Command)
		}
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
		cmd := exec.CommandContext(ctx, pyCmd, args...)
		cmd.Dir = m.RepoRoot
		cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			// Fallback to python3 if primary command fails
			cmd = exec.CommandContext(ctx, "python3", args...)
			cmd.Dir = m.RepoRoot
			cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")
			cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
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

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				outputChan <- scanner.Text()
			}
		}()

		go func() {
			defer wg.Done()
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				outputChan <- scanner.Text()
			}
		}()

		go func() {
			wg.Wait()
			close(outputChan)
		}()

		go func() {
			doneChan <- cmd.Wait()
		}()

		return CommandStartedMsg{Cmd: cmd, OutputChan: outputChan, DoneChan: doneChan}
	}
}

func splitShellWords(raw string) ([]string, error) {
	type quoteState int
	const (
		stateNone quoteState = iota
		stateSingle
		stateDouble
	)

	var out []string
	var cur strings.Builder
	state := stateNone
	escaped := false

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		out = append(out, cur.String())
		cur.Reset()
	}

	for _, r := range raw {
		if escaped {
			cur.WriteRune(r)
			escaped = false
			continue
		}

		switch state {
		case stateNone:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '\'' {
				state = stateSingle
				continue
			}
			if r == '"' {
				state = stateDouble
				continue
			}
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				flush()
				continue
			}
			cur.WriteRune(r)
		case stateSingle:
			if r == '\'' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		case stateDouble:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		}
	}

	if escaped {
		return nil, fmt.Errorf("unfinished escape sequence")
	}
	if state != stateNone {
		return nil, fmt.Errorf("unterminated quote")
	}
	flush()
	return out, nil
}

// GetContext returns a context that is cancelled when the user cancels the execution
func (m *Model) GetContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	return ctx
}

func (m Model) listenForOutput() tea.Cmd {
	// Don't listen for more output if cancelled or if channels are nil
	if m.Status == StatusCancelled || m.outputChan == nil || m.doneChan == nil {
		return nil
	}

	return func() tea.Msg {
		line, ok := <-m.outputChan
		if !ok {
			return messages.StreamDoneMsg{}
		}
		return messages.StreamOutputMsg{Line: line}
	}
}

func (m Model) listenForDone() tea.Cmd {
	if m.doneChan == nil {
		return nil
	}

	return func() tea.Msg {
		err := <-m.doneChan
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

func (m Model) listenForResourceUpdates() tea.Cmd {
	if m.resourceUpdateChan == nil {
		return nil
	}

	return func() tea.Msg {
		update, ok := <-m.resourceUpdateChan
		if !ok {
			return nil
		}
		return update
	}
}

// Update implements tea.Model and routes incoming messages to update
// execution state, logs, resource metrics and view dimensions.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case messages.TickMsg:
		m.ticker++
		m.animQueue.Tick()
		// Only continue ticking while running
		if m.Status == StatusRunning {
			return m, m.tick()
		}
		if m.Status == StatusPending {
			m.Status = StatusRunning
			if m.StartTime.IsZero() {
				m.StartTime = time.Now()
			}
			return m, m.tick()
		}
		// Done states: no more ticking needed
		return m, nil

	case CommandStartedMsg:
		m.cmd = msg.Cmd
		m.outputChan = msg.OutputChan
		m.doneChan = msg.DoneChan
		m.Status = StatusRunning
		// Start resource monitoring
		m.resourceUpdateChan = make(chan messages.ResourceUpdateMsg, 10)
		m.stopResourceChan = make(chan struct{})
		return m, tea.Batch(m.listenForOutput(), m.listenForDone(), m.startResourceMonitoring(), m.listenForResourceUpdates())

	case messages.StreamOutputMsg:
		// Don't process or listen for more output if cancelled
		if m.Status == StatusCancelled {
			return m, nil
		}
		m.processOutputLine(msg.Line)
		return m, m.listenForOutput()

	case messages.StreamDoneMsg:
		// Stream finished, but we might still be waiting for CommandDoneMsg
		return m, nil

	case messages.LogCopiedMsg:
		m.addLog(lipgloss.NewStyle().Foreground(styles.Success).Render("✓ Log copied to clipboard"))
		return m, nil

	case messages.CommandDoneMsg:
		m.stopResourceMonitoringSafe()
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
		m.updateViewportSize()
		return m, nil

	case messages.ResourceUpdateMsg:
		m.CPUUsage = msg.CPUUsage
		m.MemoryUsage = msg.MemoryUsage
		m.CPUCoreUsages = msg.CPUCoreUsages
		m.NumCPUCores = msg.NumCPUCores
		return m, m.listenForResourceUpdates()

	case executor.FileBrowserResultMsg:
		if msg.Error != nil {
			m.addLog(fmt.Sprintf("%s Failed to open results folder: %v", styles.CrossMark, msg.Error))
		} else {
			m.addLog(styles.CheckMark + " Opened results folder")
		}
		return m, nil

	case tea.KeyMsg:
		// Check if copy mode is active
		if m.copyMode {
			switch msg.String() {
			case "m", "esc", "q":
				m.copyMode = false
				m.updateViewportSize()
				return m, tea.EnableMouseCellMotion
			case "c":
				return m, m.copyLogToClipboard()
			}
			return m, nil
		}

		switch msg.String() {
		case "m":
			m.copyMode = true
			m.updateViewportSize()
			return m, tea.DisableMouse
		case "ctrl+c":
			if m.IsDone() {
				return m, m.copyLogToClipboard()
			}
			m.stopResourceMonitoringSafe()
			m.Status = StatusCancelled
			m.EndTime = time.Now()
			if m.cancel != nil {
				m.cancel()
			}
			if m.cmd != nil && m.cmd.Process != nil {
				syscall.Kill(-m.cmd.Process.Pid, syscall.SIGKILL)
			}
			m.updateViewportSize()
			return m, nil
		case "c":
			return m, m.copyLogToClipboard()
		case "o", "O":
			// Open results folder (only when done successfully)
			if m.IsDone() && m.Status == StatusSuccess {
				return m, m.OpenResultsFolder()
			}
			return m, nil
		case "down":
			m.logViewport.LineDown(styles.ScrollStepSize)
			return m, nil
		case "up":
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
		m.updateViewportSize()
	}

	return m, nil
}
