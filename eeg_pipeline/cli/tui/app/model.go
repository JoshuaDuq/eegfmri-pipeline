package app

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/cloud"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/dashboard"
	"github.com/eeg-pipeline/tui/views/environment"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/globalsetup"
	"github.com/eeg-pipeline/tui/views/history"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/quickactions"
	"github.com/eeg-pipeline/tui/views/wizard"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// AppState represents the current screen/state of the application
type AppState int

const (
	StateEnvSelect AppState = iota
	StateMainMenu
	StatePipelineWizard
	StateExecution
	StateResults
	StateGlobalSetup
	StateDashboard
	StateHistory
)

// TUIState represents the persistent state of the TUI across sessions
type TUIState struct {
	TimeRanges   []types.TimeRange `json:"time_ranges"`
	LastPipeline int               `json:"last_pipeline"`
}

// Model is the root application model
type Model struct {
	state    AppState
	navStack []AppState

	// Sub-models
	envSelect    environment.Model
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
	width       int
	height      int
	task        string
	environment environment.Environment
	cloudConfig cloud.Config
	repoRoot    string

	// Selected values
	selectedPipeline types.Pipeline

	// Persistent state
	persistentState TUIState
}

func New() Model {
	// Determine repo root from executable location or current directory
	exePath, _ := os.Executable()
	repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exePath))))

	// Check if this looks like the repo root (should contain eeg_pipeline folder)
	if _, err := os.Stat(filepath.Join(repoRoot, "eeg_pipeline")); err != nil {
		// Fallback to current working directory
		cwd, err := os.Getwd()
		if err == nil {
			// Check if CWD or one of its parents is the repo root
			checkDir := cwd
			for i := 0; i < 3; i++ {
				if _, err := os.Stat(filepath.Join(checkDir, "eeg_pipeline")); err == nil {
					repoRoot = checkDir
					break
				}
				checkDir = filepath.Dir(checkDir)
			}
		}
	}

	m := Model{
		state:       StateEnvSelect,
		navStack:    []AppState{},
		envSelect:   environment.New(),
		mainMenu:    mainmenu.New(),
		task:        "thermalactive",
		environment: environment.EnvLocal,
		cloudConfig: cloud.DefaultConfig(),
		repoRoot:    repoRoot,
	}

	m.loadState()

	// Restore last selected pipeline cursor position
	if m.persistentState.LastPipeline >= 0 && m.persistentState.LastPipeline < 6 {
		m.mainMenu.SetCursor(m.persistentState.LastPipeline)
	}

	return m
}

func (m *Model) getStatePath() string {
	return filepath.Join(m.repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_state.json")
}

func (m *Model) loadState() {
	path := m.getStatePath()
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var state TUIState
	if err := json.Unmarshal(data, &state); err != nil {
		return
	}

	m.persistentState = state
}

func (m *Model) saveState() {
	path := m.getStatePath()
	// Ensure directory exists
	os.MkdirAll(filepath.Dir(path), 0755)

	data, err := json.MarshalIndent(m.persistentState, "", "  ")
	if err != nil {
		return
	}

	_ = os.WriteFile(path, data, 0644)
}

// Init implements tea.Model
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.envSelect.Init(),
		executor.LoadConfigSummary(m.repoRoot),
	)
}

// Update implements tea.Model
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			if m.state == StateExecution && !m.execution.IsDone() {
				// Let execution handle it
			} else {
				return m, tea.Quit
			}
		case "q":
			if m.state == StateMainMenu || m.state == StateEnvSelect {
				return m, tea.Quit
			}
		case "esc":
			return m.handleEscape()
		case "enter":
			if m.state == StateExecution && m.execution.IsDone() {
				m.state = StateMainMenu
				return m, nil
			}
		case "r":
			if m.state == StateExecution && m.execution.IsDone() {
				return m.startExecution(m.wizard.BuildCommand())
			}
		case "p":
			// Pull results (cloud mode only)
			if m.state == StateExecution && m.execution.IsDone() && m.environment == environment.EnvGoogleCloud {
				dataDir := filepath.Join(m.repoRoot, "eeg_pipeline", "data")
				return m, cloud.PullDerivatives(context.Background(), m.cloudConfig, dataDir)
			}
		case "d", "D":
			// Open Dashboard from main menu
			if m.state == StateMainMenu {
				m.dashboard = dashboard.New(m.repoRoot)
				m.pushState(StateDashboard)
				return m, tea.Batch(
					m.dashboard.Init(),
					func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
				)
			}
		case "h", "H":
			// Open History from main menu
			if m.state == StateMainMenu {
				m.historyMdl = history.New(m.repoRoot)
				m.pushState(StateHistory)
				return m, tea.Batch(
					m.historyMdl.Init(),
					func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
				)
			}
		case "ctrl+k":
			// Quick Actions overlay (available from most states)
			if m.state == StateMainMenu || m.state == StatePipelineWizard {
				m.quickActions.Show()
				return m, m.quickActions.Init()
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Propagate to all models
		newEnv, _ := m.envSelect.Update(msg)
		m.envSelect = newEnv.(environment.Model)

		newMenu, _ := m.mainMenu.Update(msg)
		m.mainMenu = newMenu.(mainmenu.Model)

		newWizard, _ := m.wizard.Update(msg)
		m.wizard = newWizard.(wizard.Model)

		newExec, _ := m.execution.Update(msg)
		m.execution = newExec.(execution.Model)

		newGlobal, _ := m.global.Update(msg)
		m.global = *newGlobal.(*globalsetup.Model)

	case messages.SubjectsLoadedMsg:
		if msg.Error != nil {
			m.execution.AddOutput("Error loading subjects: " + msg.Error.Error())
			// Still notify wizard that loading is done
			m.wizard.SetSubjects(nil)
			return m, nil
		}
		subjects := make([]types.SubjectStatus, len(msg.Subjects))
		for i, s := range msg.Subjects {
			var featAvail *types.FeatureAvailability
			if s.FeatureAvailability != nil {
				featAvail = &types.FeatureAvailability{
					Features:     make(map[string]types.AvailabilityInfo),
					Bands:        make(map[string]types.AvailabilityInfo),
					Computations: make(map[string]types.AvailabilityInfo),
				}
				for k, v := range s.FeatureAvailability.Features {
					lm := ""
					if v.LastModified != nil {
						lm = *v.LastModified
					}
					featAvail.Features[k] = types.AvailabilityInfo{Available: v.Available, LastModified: lm}
				}
				for k, v := range s.FeatureAvailability.Bands {
					lm := ""
					if v.LastModified != nil {
						lm = *v.LastModified
					}
					featAvail.Bands[k] = types.AvailabilityInfo{Available: v.Available, LastModified: lm}
				}
				for k, v := range s.FeatureAvailability.Computations {
					lm := ""
					if v.LastModified != nil {
						lm = *v.LastModified
					}
					featAvail.Computations[k] = types.AvailabilityInfo{Available: v.Available, LastModified: lm}
				}
			}
			subjects[i] = types.SubjectStatus{
				ID:                  s.ID,
				HasEpochs:           s.HasEpochs,
				HasFeatures:         s.HasFeatures,
				HasStats:            s.HasStats,
				AvailableBands:      s.AvailableBands,
				FeatureAvailability: featAvail,
				EpochMetadata:       s.EpochMetadata,
			}
		}
		m.wizard.SetSubjects(subjects)
		m.wizard.SetAvailableMetadata(msg.AvailableWindows, msg.AvailableEventColumns)
		return m, nil

	case messages.PlottersLoadedMsg:
		if msg.Error != nil {
			m.wizard.SetFeaturePlottersError(msg.Error)
			return m, nil
		}
		if msg.FeaturePlotters != nil {
			converted := make(map[string][]wizard.PlotterInfo, len(msg.FeaturePlotters))
			for category, entries := range msg.FeaturePlotters {
				list := make([]wizard.PlotterInfo, 0, len(entries))
				for _, p := range entries {
					list = append(list, wizard.PlotterInfo{ID: p.ID, Category: p.Category, Name: p.Name})
				}
				converted[category] = list
			}
			m.wizard.SetFeaturePlotters(converted)
		}
		return m, nil

	case messages.ColumnsDiscoveredMsg:
		if msg.Error != nil {
			m.wizard.SetColumnsDiscoveryError(msg.Error)
			return m, nil
		}
		m.wizard.SetDiscoveredColumns(msg.Columns, msg.Values, msg.Source)
		return m, nil

	// Cloud operation messages
	case cloud.SyncCompleteMsg:
		if msg.Error != nil {
			m.execution.AddOutput("Sync failed: " + msg.Error.Error())
			m.execution.SetStatus(execution.StatusFailed)
			return m, nil
		}
		m.execution.AddOutput("Sync complete. Running pipeline...")
		m.execution.CloudStage = execution.StageRunning
		// Now run the actual command
		cmd := m.wizard.BuildCommand()
		return m, cloud.RunRemoteCommand(m.execution.GetContext(), m.cloudConfig, cmd)

	case cloud.RunCompleteMsg:
		if msg.Error != nil {
			m.execution.SetStatus(execution.StatusFailed)
			m.execution.CloudStage = execution.StageDone
		} else if msg.ExitCode == 0 {
			m.execution.AddOutput("Pipeline run complete. Pulling results...")
			m.execution.CloudStage = execution.StagePulling
			localDataDir := filepath.Join(m.repoRoot, "eeg_pipeline", "data")
			return m, cloud.PullDerivatives(m.execution.GetContext(), m.cloudConfig, localDataDir)
		} else {
			m.execution.SetStatus(execution.StatusFailed)
			m.execution.CloudStage = execution.StageDone
		}
		return m, nil

	case cloud.PullCompleteMsg:
		m.execution.CloudStage = execution.StageDone
		if msg.Error != nil {
			m.execution.AddOutput("Pull failed: " + msg.Error.Error())
		} else {
			m.execution.AddOutput("Results pulled successfully.")
			m.execution.SetStatus(execution.StatusSuccess)
		}
		return m, nil

	case messages.RefreshSubjectsMsg:
		// Reload subjects when F5 is pressed in wizard
		if m.state == StatePipelineWizard {
			m.wizard.SetSubjectsLoading()
			return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
		}
		return m, nil
	case messages.ConfigLoadedMsg:
		if msg.Error != nil {
			return m, nil
		}
		m.wizard.SetConfigSummary(msg.Summary)
		if msg.Summary.Task != "" && msg.Summary.Task != m.task {
			m.task = msg.Summary.Task
			if m.environment == environment.EnvGoogleCloud {
				m.mainMenu.Task = m.task + " [Cloud]"
			} else {
				m.mainMenu.Task = m.task
			}
			if m.state == StatePipelineWizard {
				m.wizard.SetSubjectsLoading()
				return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
			}
		}
		return m, nil
	case messages.TaskUpdatedMsg:
		if msg.Task != "" && msg.Task != m.task {
			m.task = msg.Task
			if m.environment == environment.EnvGoogleCloud {
				m.mainMenu.Task = m.task + " [Cloud]"
			} else {
				m.mainMenu.Task = m.task
			}
			if m.state == StatePipelineWizard {
				m.wizard.SetSubjectsLoading()
				return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
			}
		}
		return m, nil
	case messages.ConfigKeysLoadedMsg:
		if msg.Error != nil {
			return m, nil
		}
		switch m.state {
		case StateGlobalSetup:
			m.global.SetConfigValues(msg.Values)
		case StatePipelineWizard:
			m.wizard.ApplyConfigKeys(msg.Values)
		}
		return m, nil
	}

	// Delegate to current view
	var cmd tea.Cmd
	switch m.state {
	case StateEnvSelect:
		var newEnvSelect tea.Model
		newEnvSelect, cmd = m.envSelect.Update(msg)
		m.envSelect = newEnvSelect.(environment.Model)

		if m.envSelect.Done {
			m.environment = m.envSelect.Selected
			m.cloudConfig = m.envSelect.CloudConfig
			m.state = StateMainMenu

			// Update main menu with environment info
			if m.environment == environment.EnvGoogleCloud {
				m.mainMenu.Task = m.task + " [Cloud]"
			} else {
				m.mainMenu.Task = m.task
			}
			return m, nil
		}

	case StateMainMenu:
		var newMainMenu tea.Model
		newMainMenu, cmd = m.mainMenu.Update(msg)
		m.mainMenu = newMainMenu.(mainmenu.Model)

		if m.mainMenu.SelectedPipeline >= 0 {
			m.selectedPipeline = types.Pipeline(m.mainMenu.SelectedPipeline)

			// Save last selected pipeline
			m.persistentState.LastPipeline = m.mainMenu.SelectedPipeline
			m.saveState()

			m.wizard = wizard.New(m.selectedPipeline, m.repoRoot)
			// Load persistent time ranges
			m.wizard.SetTimeRanges(m.persistentState.TimeRanges)
			m.wizard.SetSubjectsLoading()
			m.pushState(StatePipelineWizard)
			m.mainMenu.SelectedPipeline = -1
			return m, tea.Batch(
				executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline),
				executor.LoadPlotters(m.repoRoot),
				executor.LoadConfigSummary(m.repoRoot),
				executor.LoadConfigKeys(m.repoRoot, []string{"time_frequency_analysis.bands"}),
				executor.DiscoverColumns(m.repoRoot, m.task),
			)
		}
		if m.mainMenu.SelectedUtility == mainmenu.UtilityGlobalSetup {
			m.mainMenu.SelectedUtility = -1
			m.global = globalsetup.New(m.repoRoot)
			m.global.SetSize(m.width, m.height)
			m.state = StateGlobalSetup
			return m, executor.LoadConfigKeys(m.repoRoot, globalsetup.DefaultConfigKeys())
		}

	case StatePipelineWizard:
		var newWizard tea.Model
		newWizard, cmd = m.wizard.Update(msg)
		m.wizard = newWizard.(wizard.Model)

		// Sync time ranges to persistent state
		m.persistentState.TimeRanges = m.wizard.TimeRanges

		// Only save on key messages to avoid redundant disk I/O on ticks
		if _, ok := msg.(tea.KeyMsg); ok {
			m.saveState()
		}

		if m.wizard.ReadyToExecute {
			command := m.wizard.BuildCommand()
			return m.startExecution(command)
		}

	case StateGlobalSetup:
		var newGlobal tea.Model
		newGlobal, cmd = m.global.Update(msg)
		m.global = *newGlobal.(*globalsetup.Model)
		if m.global.Done {
			m.state = StateMainMenu
			return m, executor.LoadConfigSummary(m.repoRoot)
		}

	case StateExecution:
		var newExec tea.Model
		newExec, cmd = m.execution.Update(msg)
		m.execution = newExec.(execution.Model)

	case StateDashboard:
		var newDash tea.Model
		newDash, cmd = m.dashboard.Update(msg)
		m.dashboard = newDash.(dashboard.Model)

	case StateHistory:
		var newHist tea.Model
		newHist, cmd = m.historyMdl.Update(msg)
		m.historyMdl = newHist.(history.Model)

		// Handle re-run from history
		if m.historyMdl.Done && m.historyMdl.SelectedCommand != "" {
			command := m.historyMdl.SelectedCommand
			m.historyMdl.Done = false
			m.historyMdl.SelectedCommand = ""
			return m.startExecution(command)
		}
	}

	// Handle Quick Actions overlay (can be shown on top of other views)
	if m.quickActions.Visible {
		var newQA tea.Model
		newQA, cmd = m.quickActions.Update(msg)
		m.quickActions = newQA.(quickactions.Model)

		if m.quickActions.Done {
			action := m.quickActions.SelectedAction
			m.quickActions.Reset()
			m.quickActions.Hide()
			return m.handleQuickAction(action)
		}
		return m, cmd
	}

	return m, cmd
}

// handleQuickAction processes the selected quick action
func (m Model) handleQuickAction(action quickactions.ActionType) (tea.Model, tea.Cmd) {
	switch action {
	case quickactions.ActionStats:
		// Open Dashboard
		m.dashboard = dashboard.New(m.repoRoot)
		m.pushState(StateDashboard)
		return m, tea.Batch(
			m.dashboard.Init(),
			func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
		)
	case quickactions.ActionHistory:
		// Open History
		m.historyMdl = history.New(m.repoRoot)
		m.pushState(StateHistory)
		return m, tea.Batch(
			m.historyMdl.Init(),
			func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
		)
	case quickactions.ActionConfig:
		// Open Global Setup
		m.global = globalsetup.New(m.repoRoot)
		m.global.SetSize(m.width, m.height)
		m.pushState(StateGlobalSetup)
		return m, executor.LoadConfigKeys(m.repoRoot, globalsetup.DefaultConfigKeys())
	case quickactions.ActionRefresh:
		// Refresh subjects
		if m.state == StatePipelineWizard {
			m.wizard.SetSubjectsLoading()
			return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
		}
	case quickactions.ActionValidate:
		// Run validation command
		cmd := "eeg-pipeline utilities validate --all-subjects"
		if m.task != "" {
			cmd += " --task " + m.task
		}
		return m.startExecution(cmd)
	case quickactions.ActionExport:
		// Run export command
		cmd := "eeg-pipeline utilities export --all-subjects --format csv"
		if m.task != "" {
			cmd += " --task " + m.task
		}
		return m.startExecution(cmd)
	}
	return m, nil
}

// startExecution starts pipeline execution (local or cloud)
func (m Model) startExecution(command string) (tea.Model, tea.Cmd) {
	m.execution = execution.NewWithRoot(command, m.repoRoot)
	m.execution.IsCloud = m.environment == environment.EnvGoogleCloud
	m.execution.SetSize(m.width, m.height)
	m.pushState(StateExecution)
	m.wizard.ReadyToExecute = false

	// Track execution for history
	m.execStartTime = time.Now()
	m.execCommand = command

	if m.environment == environment.EnvGoogleCloud {
		// Cloud mode: sync first, then run
		m.execution.AddOutput("Syncing code to remote VM...")
		return m, cloud.SyncToRemote(m.execution.GetContext(), m.cloudConfig, m.repoRoot)
	}

	// Local mode: run directly
	return m, m.execution.Start()
}

// handleEscape handles the escape key
func (m Model) handleEscape() (tea.Model, tea.Cmd) {
	switch m.state {
	case StateEnvSelect:
		return m, tea.Quit
	case StateMainMenu:
		m.state = StateEnvSelect
		m.envSelect = environment.New()
		return m, nil
	case StatePipelineWizard:
		if m.wizard.GoBack() {
			return m, tea.ClearScreen
		}
		return m.popState()
	case StateGlobalSetup:
		m.state = StateMainMenu
		return m, nil
	case StateExecution:
		if m.execution.IsDone() {
			// Record execution to history before returning
			m.recordExecutionToHistory()
			return m.popState()
		}
		return m, nil
	case StateDashboard, StateHistory:
		return m.popState()
	default:
		return m.popState()
	}
}

// recordExecutionToHistory saves the completed execution to history
func (m *Model) recordExecutionToHistory() {
	if m.execCommand == "" {
		return
	}

	// Extract pipeline name from command
	parts := strings.Fields(m.execCommand)
	pipeline := "unknown"
	mode := ""
	if len(parts) >= 2 {
		pipeline = parts[1]
	}
	if len(parts) >= 3 {
		mode = parts[2]
	}

	// Determine success from execution status
	success := m.execution.Status == execution.StatusSuccess

	record := history.ExecutionRecord{
		ID:        time.Now().Format("20060102150405"),
		Command:   m.execCommand,
		Pipeline:  pipeline,
		Mode:      mode,
		StartTime: m.execStartTime,
		EndTime:   time.Now(),
		Duration:  time.Since(m.execStartTime).Seconds(),
		ExitCode:  0,
		Success:   success,
	}
	if !success {
		record.ExitCode = 1
	}

	_ = history.AddRecord(m.repoRoot, record)
}

// pushState pushes current state to nav stack
func (m *Model) pushState(newState AppState) {
	m.navStack = append(m.navStack, m.state)
	m.state = newState
}

// popState returns to the previous state
func (m Model) popState() (tea.Model, tea.Cmd) {
	if len(m.navStack) > 0 {
		m.state = m.navStack[len(m.navStack)-1]
		m.navStack = m.navStack[:len(m.navStack)-1]
	}
	return m, nil
}

// View implements tea.Model
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
	case StateEnvSelect:
		content = m.envSelect.View()
	case StateMainMenu:
		content = m.mainMenu.View()
	case StatePipelineWizard:
		content = m.wizard.View()
	case StateGlobalSetup:
		content = m.global.View()
	case StateExecution:
		content = m.execution.View()
	case StateResults:
		content = m.renderResults()
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

func (m Model) renderResults() string {
	return styles.BrandStyle.Render("Results") + "\n\n" +
		lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark+" Pipeline completed") + "\n\n" +
		styles.HelpStyle.Render("[Enter] Return to menu")
}
