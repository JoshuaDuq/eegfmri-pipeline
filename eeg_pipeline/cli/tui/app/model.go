package app

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/eeg-pipeline/tui/cloud"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/environment"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/mainmenu"
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
)

// TUIState represents the persistent state of the TUI across sessions
type TUIState struct {
	TimeRanges []types.TimeRange `json:"time_ranges"`
}

// Model is the root application model
type Model struct {
	state    AppState
	navStack []AppState

	// Sub-models
	envSelect environment.Model
	mainMenu  mainmenu.Model
	wizard    wizard.Model
	execution execution.Model

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

// New creates a new application model
func New() Model {
	// Determine repo root from executable location
	exePath, _ := os.Executable()
	repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exePath))))

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
	return m.envSelect.Init()
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
				return m, cloud.PullDerivatives(m.cloudConfig, dataDir)
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
					Features: make(map[string]types.AvailabilityInfo),
					Bands:    make(map[string]types.AvailabilityInfo),
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
			}
			subjects[i] = types.SubjectStatus{
				ID:                  s.ID,
				HasEpochs:           s.HasEpochs,
				HasFeatures:         s.HasFeatures,
				AvailableBands:      s.AvailableBands,
				FeatureAvailability: featAvail,
				EpochMetadata:       s.EpochMetadata,
			}
		}
		m.wizard.SetSubjects(subjects)
		return m, nil

	// Cloud operation messages
	case cloud.SyncCompleteMsg:
		if msg.Error != nil {
			m.execution.AddOutput("Sync failed: " + msg.Error.Error())
			return m, nil
		}
		m.execution.AddOutput("Sync complete. Running pipeline...")
		// Now run the actual command
		cmd := m.wizard.BuildCommand()
		return m, cloud.RunRemoteCommand(m.cloudConfig, cmd)

	case cloud.RunCompleteMsg:
		if msg.Error != nil {
			m.execution.SetStatus(execution.StatusFailed)
		} else if msg.ExitCode == 0 {
			m.execution.SetStatus(execution.StatusSuccess)
			// Auto-pull if configured
			if m.cloudConfig.AutoPull && m.environment == environment.EnvGoogleCloud {
				dataDir := filepath.Join(m.repoRoot, "eeg_pipeline", "data")
				return m, cloud.PullDerivatives(m.cloudConfig, dataDir)
			}
		} else {
			m.execution.SetStatus(execution.StatusFailed)
		}
		return m, nil

	case cloud.PullCompleteMsg:
		if msg.Error != nil {
			m.execution.AddOutput("Pull failed: " + msg.Error.Error())
		} else {
			m.execution.AddOutput("Results pulled successfully.")
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
			m.wizard = wizard.New(m.selectedPipeline)
			// Load persistent time ranges
			m.wizard.SetTimeRanges(m.persistentState.TimeRanges)
			m.wizard.SetSubjectsLoading()
			m.pushState(StatePipelineWizard)
			m.mainMenu.SelectedPipeline = -1
			return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
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

	case StateExecution:
		var newExec tea.Model
		newExec, cmd = m.execution.Update(msg)
		m.execution = newExec.(execution.Model)
	}

	return m, cmd
}

// startExecution starts pipeline execution (local or cloud)
func (m Model) startExecution(command string) (tea.Model, tea.Cmd) {
	m.execution = execution.NewWithRoot(command, m.repoRoot)
	m.execution.IsCloud = m.environment == environment.EnvGoogleCloud
	m.pushState(StateExecution)
	m.wizard.ReadyToExecute = false

	if m.environment == environment.EnvGoogleCloud {
		// Cloud mode: sync first, then run
		m.execution.AddOutput("Syncing code to remote VM...")
		return m, cloud.SyncToRemote(m.cloudConfig, m.repoRoot)
	}

	// Local mode: run directly
	return m, m.execution.Init()
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
	case StateExecution:
		if m.execution.IsDone() {
			return m.popState()
		}
		return m, nil
	default:
		return m.popState()
	}
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

	var content string

	switch m.state {
	case StateEnvSelect:
		content = m.envSelect.View()
	case StateMainMenu:
		content = m.mainMenu.View()
	case StatePipelineWizard:
		content = m.wizard.View()
	case StateExecution:
		content = m.execution.View()
	case StateResults:
		content = m.renderResults()
	default:
		content = "Unknown state"
	}

	return styles.BoxStyle.
		Width(m.width).
		Height(m.height).
		Render(content)
}

func (m Model) renderResults() string {
	return styles.BrandStyle.Render("Results") + "\n\n" +
		lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark+" Pipeline completed") + "\n\n" +
		styles.HelpStyle.Render("[Enter] Return to menu")
}
