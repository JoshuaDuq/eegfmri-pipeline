package app

import (
	"time"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/dashboard"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/globalsetup"
	"github.com/eeg-pipeline/tui/views/history"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/quickactions"
	"github.com/eeg-pipeline/tui/views/wizard"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// File layout notes:
// - `model.go`: core app model/type definitions and top-level Tea entry points.
// - `model_persistence.go`: repo root discovery and persisted TUI state.
// - `model_messages.go`: subject/config message handlers and converters.
// - `model_stateflow.go`: per-view update delegation and navigation/execution flow.

// AppState represents the current screen/state of the application
type AppState int

const (
	StateMainMenu AppState = iota
	StatePipelineWizard
	StateExecution
	StateGlobalSetup
	StateDashboard
	StateHistory
)

const (
	maxPipelineIndex = 5 // Maximum valid pipeline index (0-5 for 6 pipelines)
	maxNavDepth      = 3 // Maximum depth to search for repo root
)

// TUIState represents the persistent state of the TUI across sessions
type TUIState struct {
	TimeRanges      []types.TimeRange `json:"time_ranges"`
	LastPipeline    int               `json:"last_pipeline"`
	ROICacheVersion int               `json:"roi_cache_version,omitempty"`

	// Band configuration
	Bands        []BandState `json:"bands,omitempty"`
	BandSelected []bool      `json:"band_selected,omitempty"`

	// ROI configuration
	ROIs        []ROIState `json:"rois,omitempty"`
	ROISelected []bool     `json:"roi_selected,omitempty"`

	// Spatial selection
	SpatialSelected []bool `json:"spatial_selected,omitempty"`

	// Per-pipeline advanced configuration
	// Key is pipeline name (e.g., "features", "behavior", "preprocessing")
	PipelineConfigs map[string]map[string]interface{} `json:"pipeline_configs,omitempty"`
}

// BandState holds serializable band configuration
type BandState struct {
	Key    string  `json:"key"`
	Name   string  `json:"name"`
	LowHz  float64 `json:"low_hz"`
	HighHz float64 `json:"high_hz"`
}

// ROIState holds serializable ROI configuration
type ROIState struct {
	Key      string `json:"key"`
	Name     string `json:"name"`
	Channels string `json:"channels"`
}

// Model is the root application model
type Model struct {
	state    AppState
	navStack []AppState

	// Sub-models
	mainMenu     mainmenu.Model
	wizard       wizard.Model
	execution    execution.Model
	global       globalsetup.Model
	dashboard    dashboard.Model
	historyMdl   history.Model
	quickActions quickactions.Model

	// Execution tracking for history
	execStartTime time.Time
	execCommand   string

	// Shared state
	width    int
	height   int
	task     string
	repoRoot string

	// Selected values
	selectedPipeline types.Pipeline

	// Persistent state
	persistentState TUIState

	// Subject discovery cache (per task + data source) to avoid repeated scans.
	subjectsCache           map[string]messages.SubjectsLoadedMsg
	pendingSubjectsCacheKey string
}

func New() Model {
	repoRoot := findRepoRoot()

	m := Model{
		state:         StateMainMenu,
		navStack:      []AppState{},
		mainMenu:      mainmenu.New(),
		task:          "thermalactive",
		repoRoot:      repoRoot,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
	}

	m.loadState()

	// Restore last selected pipeline cursor position
	if m.isValidPipelineIndex(m.persistentState.LastPipeline) {
		m.mainMenu.SetCursor(m.persistentState.LastPipeline)
	}

	return m
}

// Init implements tea.Model
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		executor.LoadConfigSummary(m.repoRoot),
	)
}

// Update implements tea.Model
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// First handle global messages (window size, subjects loaded, etc.)
	newModel, cmd := m.handleGlobalMessages(msg)
	// Handle both pointer and value types
	if ptrModel, ok := newModel.(*Model); ok {
		m = *ptrModel
	} else {
		m = newModel.(Model)
	}

	// If there's a command, we prioritize it
	if cmd != nil {
		return m, cmd
	}

	// Delegate remaining messages to the current view
	return m.delegateToCurrentView(msg)
}

func (m Model) handleGlobalMessages(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		return m.handleKeyMessage(msg)
	case tea.WindowSizeMsg:
		m.handleWindowSize(msg)
		return m, nil
	case messages.SubjectsLoadedMsg:
		return m.handleSubjectsLoaded(msg)
	case messages.PlottersLoadedMsg:
		m.handlePlottersLoaded(msg)
		return m, nil
	case messages.ColumnsDiscoveredMsg:
		m.handleColumnsDiscovered(msg)
		return m, nil
	case messages.ROIsDiscoveredMsg:
		m.handleROIsDiscovered(msg)
		return m, nil
	case messages.FmriColumnsDiscoveredMsg:
		m.handleFmriColumnsDiscovered(msg)
		return m, nil
	case messages.MultigroupStatsDiscoveredMsg:
		m.handleMultigroupStatsDiscovered(msg)
		return m, nil
	case messages.RefreshSubjectsMsg:
		return m, m.handleRefreshSubjects()
	case messages.ConfigLoadedMsg:
		return m.handleConfigLoaded(msg)
	case messages.TaskUpdatedMsg:
		return m.handleTaskUpdated(msg)
	case messages.ConfigKeysLoadedMsg:
		m.handleConfigKeysLoaded(msg)
		return m, nil
	}
	return m, nil
}

func (m Model) handleKeyMessage(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// When a sub-view is actively editing configuration, defer all key
	// handling to that view so that global bindings like "q" do not
	// trigger application-level exits.
	switch m.state {
	case StatePipelineWizard:
		if m.wizard.IsEditing() {
			return m, nil
		}
	case StateGlobalSetup:
		if m.global.IsEditing() {
			return m, nil
		}
	}

	switch msg.String() {
	case "ctrl+c", "q":
		return m, m.handleQuit()
	case "esc":
		return m.handleEscape()
	case "enter":
		return m, m.handleEnter()
	case "r":
		return m.handleRestart()
	case "p":
		return m.handlePullResults()
	case "d", "D":
		return m.handleOpenDashboard()
	case "h", "H":
		return m.handleOpenHistory()
	case "ctrl+k":
		return m.handleQuickActions()
	}
	return m, nil
}

func (m Model) handleQuit() tea.Cmd {
	if m.state == StateExecution && !m.execution.IsDone() {
		return nil
	}
	return tea.Quit
}

func (m *Model) handleEnter() tea.Cmd {
	if m.state == StateExecution && m.execution.IsDone() {
		m.state = StateMainMenu
		return nil
	}
	return nil
}

func (m Model) handleRestart() (tea.Model, tea.Cmd) {
	if m.state == StateExecution && m.execution.IsDone() {
		return m.startExecution(m.wizard.BuildCommand())
	}
	return m, nil
}

func (m Model) handlePullResults() (tea.Model, tea.Cmd) {
	return m, nil
}

func (m Model) handleOpenDashboard() (tea.Model, tea.Cmd) {
	if m.state != StateMainMenu {
		return m, nil
	}
	m.dashboard = dashboard.New(m.repoRoot)
	m.pushState(StateDashboard)
	return m, tea.Batch(
		m.dashboard.Init(),
		func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
	)
}

func (m Model) handleOpenHistory() (tea.Model, tea.Cmd) {
	if m.state != StateMainMenu {
		return m, nil
	}
	m.historyMdl = history.New(m.repoRoot)
	m.pushState(StateHistory)
	return m, tea.Batch(
		m.historyMdl.Init(),
		func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
	)
}

func (m Model) handleQuickActions() (tea.Model, tea.Cmd) {
	if m.state == StateMainMenu || m.state == StatePipelineWizard {
		m.quickActions.Show()
		return m, m.quickActions.Init()
	}
	return m, nil
}

func (m *Model) handleWindowSize(msg tea.WindowSizeMsg) {
	m.width = msg.Width
	m.height = msg.Height

	// Propagate to all models
	newMenu, _ := m.mainMenu.Update(msg)
	m.mainMenu = newMenu.(mainmenu.Model)

	newWizard, _ := m.wizard.Update(msg)
	m.wizard = newWizard.(wizard.Model)

	newExec, _ := m.execution.Update(msg)
	m.execution = newExec.(execution.Model)

	newGlobal, _ := m.global.Update(msg)
	m.global = *newGlobal.(*globalsetup.Model)
}

func (m Model) View() string {
	if m.width == 0 || m.height == 0 {
		return "Initializing..."
	}

	// Check if terminal is too small to render properly
	if styles.IsTerminalTooSmall(m.width, m.height) {
		return lipgloss.Place(
			m.width, m.height,
			lipgloss.Center, lipgloss.Center,
			styles.RenderTerminalTooSmall(m.width, m.height),
		)
	}

	var content string

	switch m.state {
	case StateMainMenu:
		content = m.mainMenu.View()
	case StatePipelineWizard:
		content = m.wizard.View()
	case StateGlobalSetup:
		content = m.global.View()
	case StateExecution:
		content = m.execution.View()
	case StateDashboard:
		content = m.dashboard.View()
	case StateHistory:
		content = m.historyMdl.View()
	default:
		content = "Unknown state"
	}

	// Render Quick Actions overlay if visible
	if m.quickActions.Visible {
		overlay := m.quickActions.View()
		content = lipgloss.Place(
			m.width, m.height,
			lipgloss.Center, lipgloss.Center,
			overlay,
			lipgloss.WithWhitespaceChars(" "),
			lipgloss.WithWhitespaceForeground(lipgloss.Color("#333333")),
		)
	}

	// Fill entire terminal using Height to force full terminal height
	return lipgloss.NewStyle().
		Width(m.width).
		Height(m.height).
		Render(content)
}
