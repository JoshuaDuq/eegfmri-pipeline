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

const (
	maxPipelineIndex = 5 // Maximum valid pipeline index (0-4 for 5 pipelines)
	maxNavDepth      = 3 // Maximum depth to search for repo root
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
	repoRoot := findRepoRoot()

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
	if m.isValidPipelineIndex(m.persistentState.LastPipeline) {
		m.mainMenu.SetCursor(m.persistentState.LastPipeline)
	}

	return m
}

func findRepoRoot() string {
	exePath, err := os.Executable()
	if err == nil {
		repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(exePath))))
		if isValidRepoRoot(repoRoot) {
			return repoRoot
		}
	}

	// Fallback to current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return cwd
	}

	return findRepoRootFromPath(cwd)
}

func findRepoRootFromPath(startPath string) string {
	checkDir := startPath
	for i := 0; i < maxNavDepth; i++ {
		if isValidRepoRoot(checkDir) {
			return checkDir
		}
		checkDir = filepath.Dir(checkDir)
	}
	return startPath
}

func isValidRepoRoot(path string) bool {
	_, err := os.Stat(filepath.Join(path, "eeg_pipeline"))
	return err == nil
}

func (m *Model) isValidPipelineIndex(index int) bool {
	return index >= 0 && index <= maxPipelineIndex
}

func (m *Model) getStatePath() string {
	return filepath.Join(m.repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_state.json")
}

func (m *Model) loadState() {
	path := m.getStatePath()
	data, err := os.ReadFile(path)
	if err != nil {
		// State file doesn't exist or can't be read - use defaults
		return
	}

	var state TUIState
	if err := json.Unmarshal(data, &state); err != nil {
		// Invalid state file - use defaults
		return
	}

	m.persistentState = state
}

func (m *Model) saveState() {
	path := m.getStatePath()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		// Cannot create directory - state won't be saved
		return
	}

	data, err := json.MarshalIndent(m.persistentState, "", "  ")
	if err != nil {
		// Cannot serialize state - state won't be saved
		return
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		// Cannot write state file - state won't be saved
		return
	}
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
	// First handle global messages (window size, subjects loaded, etc.)
	newModel, cmd := m.handleGlobalMessages(msg)
	m = newModel.(Model)

	// If there's a command, we prioritize it
	if cmd != nil {
		return m, cmd
	}

	// For WindowSizeMsg, we've already handled it and propagated it in handleGlobalMessages
	if _, ok := msg.(tea.WindowSizeMsg); ok {
		return m, nil
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
		m.handleSubjectsLoaded(msg)
		return m, nil
	case messages.PlottersLoadedMsg:
		m.handlePlottersLoaded(msg)
		return m, nil
	case messages.ColumnsDiscoveredMsg:
		m.handleColumnsDiscovered(msg)
		return m, nil
	case cloud.SyncCompleteMsg:
		return m.handleCloudSyncComplete(msg)
	case cloud.RunCompleteMsg:
		return m.handleCloudRunComplete(msg)
	case cloud.PullCompleteMsg:
		m.handleCloudPullComplete(msg)
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
	switch msg.String() {
	case "ctrl+c":
		return m, m.handleCtrlC()
	case "q":
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

func (m Model) handleCtrlC() tea.Cmd {
	if m.state == StateExecution && !m.execution.IsDone() {
		return nil // Let execution handle it
	}
	return tea.Quit
}

func (m Model) handleQuit() tea.Cmd {
	if m.state == StateMainMenu || m.state == StateEnvSelect {
		return tea.Quit
	}
	return nil
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
	isCloudExecutionDone := m.state == StateExecution && m.execution.IsDone() && m.environment == environment.EnvGoogleCloud
	if !isCloudExecutionDone {
		return m, nil
	}
	dataDir := filepath.Join(m.repoRoot, "eeg_pipeline", "data")
	return m, cloud.PullDerivatives(context.Background(), m.cloudConfig, dataDir)
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
}

func (m *Model) handleSubjectsLoaded(msg messages.SubjectsLoadedMsg) {
	if msg.Error != nil {
		m.execution.AddOutput("Error loading subjects: " + msg.Error.Error())
		m.wizard.SetSubjects(nil)
		return
	}

	subjects := m.convertSubjects(msg.Subjects)
	m.wizard.SetSubjects(subjects)
	m.wizard.SetAvailableMetadata(msg.AvailableWindows, msg.AvailableEventColumns)
}

func (m *Model) convertSubjects(sourceSubjects []messages.SubjectInfo) []types.SubjectStatus {
	subjects := make([]types.SubjectStatus, len(sourceSubjects))
	for i, s := range sourceSubjects {
		subjects[i] = types.SubjectStatus{
			ID:                  s.ID,
			HasEpochs:           s.HasEpochs,
			HasFeatures:         s.HasFeatures,
			HasStats:            s.HasStats,
			AvailableBands:      s.AvailableBands,
			FeatureAvailability: m.convertFeatureAvailability(s.FeatureAvailability),
			EpochMetadata:       s.EpochMetadata,
		}
	}
	return subjects
}

func (m *Model) convertFeatureAvailability(source *messages.FeatureAvailability) *types.FeatureAvailability {
	if source == nil {
		return nil
	}

	featAvail := &types.FeatureAvailability{
		Features:     make(map[string]types.AvailabilityInfo),
		Bands:        make(map[string]types.AvailabilityInfo),
		Computations: make(map[string]types.AvailabilityInfo),
	}

	for k, v := range source.Features {
		featAvail.Features[k] = m.convertAvailabilityInfo(v)
	}
	for k, v := range source.Bands {
		featAvail.Bands[k] = m.convertAvailabilityInfo(v)
	}
	for k, v := range source.Computations {
		featAvail.Computations[k] = m.convertAvailabilityInfo(v)
	}

	return featAvail
}

func (m *Model) convertAvailabilityInfo(v messages.AvailabilityInfo) types.AvailabilityInfo {
	lastModified := ""
	if v.LastModified != nil {
		lastModified = *v.LastModified
	}
	return types.AvailabilityInfo{
		Available:    v.Available,
		LastModified: lastModified,
	}
}

func (m *Model) handlePlottersLoaded(msg messages.PlottersLoadedMsg) {
	if msg.Error != nil {
		m.wizard.SetFeaturePlottersError(msg.Error)
		return
	}

	if msg.FeaturePlotters == nil {
		return
	}

	converted := m.convertPlotters(msg.FeaturePlotters)
	m.wizard.SetFeaturePlotters(converted)
}

func (m *Model) convertPlotters(source map[string][]messages.PlotterInfo) map[string][]wizard.PlotterInfo {
	converted := make(map[string][]wizard.PlotterInfo, len(source))
	for category, entries := range source {
		list := make([]wizard.PlotterInfo, 0, len(entries))
		for _, p := range entries {
			list = append(list, wizard.PlotterInfo{
				ID:       p.ID,
				Category: p.Category,
				Name:     p.Name,
			})
		}
		converted[category] = list
	}
	return converted
}

func (m *Model) handleColumnsDiscovered(msg messages.ColumnsDiscoveredMsg) {
	if msg.Error != nil {
		m.wizard.SetColumnsDiscoveryError(msg.Error)
		return
	}
	m.wizard.SetDiscoveredColumns(msg.Columns, msg.Values, msg.Source)
}

func (m Model) handleCloudSyncComplete(msg cloud.SyncCompleteMsg) (tea.Model, tea.Cmd) {
	if msg.Error != nil {
		m.execution.AddOutput("Sync failed: " + msg.Error.Error())
		m.execution.SetStatus(execution.StatusFailed)
		return m, nil
	}

	m.execution.AddOutput("Sync complete. Running pipeline...")
	m.execution.CloudStage = execution.StageRunning
	command := m.wizard.BuildCommand()
	return m, cloud.RunRemoteCommand(m.execution.GetContext(), m.cloudConfig, command)
}

func (m Model) handleCloudRunComplete(msg cloud.RunCompleteMsg) (tea.Model, tea.Cmd) {
	if msg.Error != nil {
		m.execution.SetStatus(execution.StatusFailed)
		m.execution.CloudStage = execution.StageDone
		return m, nil
	}

	if msg.ExitCode == 0 {
		m.execution.AddOutput("Pipeline run complete. Pulling results...")
		m.execution.CloudStage = execution.StagePulling
		localDataDir := filepath.Join(m.repoRoot, "eeg_pipeline", "data")
		return m, cloud.PullDerivatives(m.execution.GetContext(), m.cloudConfig, localDataDir)
	}

	m.execution.SetStatus(execution.StatusFailed)
	m.execution.CloudStage = execution.StageDone
	return m, nil
}

func (m *Model) handleCloudPullComplete(msg cloud.PullCompleteMsg) {
	m.execution.CloudStage = execution.StageDone
	if msg.Error != nil {
		m.execution.AddOutput("Pull failed: " + msg.Error.Error())
		return
	}

	m.execution.AddOutput("Results pulled successfully.")
	m.execution.SetStatus(execution.StatusSuccess)
}

func (m Model) handleRefreshSubjects() tea.Cmd {
	if m.state != StatePipelineWizard {
		return nil
	}
	m.wizard.SetSubjectsLoading()
	return executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
}

func (m Model) handleConfigLoaded(msg messages.ConfigLoadedMsg) (tea.Model, tea.Cmd) {
	if msg.Error != nil {
		return m, nil
	}

	m.wizard.SetConfigSummary(msg.Summary)
	if !m.shouldUpdateTask(msg.Summary.Task) {
		return m, nil
	}

	m.updateTask(msg.Summary.Task)
	if m.state == StatePipelineWizard {
		m.wizard.SetSubjectsLoading()
		return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
	}
	return m, nil
}

func (m Model) handleTaskUpdated(msg messages.TaskUpdatedMsg) (tea.Model, tea.Cmd) {
	if !m.shouldUpdateTask(msg.Task) {
		return m, nil
	}

	m.updateTask(msg.Task)
	if m.state == StatePipelineWizard {
		m.wizard.SetSubjectsLoading()
		return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
	}
	return m, nil
}

func (m *Model) shouldUpdateTask(newTask string) bool {
	return newTask != "" && newTask != m.task
}

func (m *Model) updateTask(newTask string) {
	m.task = newTask
	if m.environment == environment.EnvGoogleCloud {
		m.mainMenu.Task = m.task + " [Cloud]"
	} else {
		m.mainMenu.Task = m.task
	}
}

func (m *Model) handleConfigKeysLoaded(msg messages.ConfigKeysLoadedMsg) {
	if msg.Error != nil {
		return
	}

	switch m.state {
	case StateGlobalSetup:
		m.global.SetConfigValues(msg.Values)
	case StatePipelineWizard:
		m.wizard.ApplyConfigKeys(msg.Values)
	}
}

func (m Model) delegateToCurrentView(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch m.state {
	case StateEnvSelect:
		return m.handleEnvSelectUpdate(msg)

	case StateMainMenu:
		return m.handleMainMenuUpdate(msg)

	case StatePipelineWizard:
		return m.handleWizardUpdate(msg)

	case StateGlobalSetup:
		return m.handleGlobalSetupUpdate(msg)
	case StateExecution:
		return m.handleExecutionUpdate(msg)
	case StateDashboard:
		return m.handleDashboardUpdate(msg)
	case StateHistory:
		return m.handleHistoryUpdate(msg)
	}

	// Handle Quick Actions overlay (can be shown on top of other views)
	if m.quickActions.Visible {
		return m.handleQuickActionsOverlay(msg)
	}

	return m, cmd
}

func (m Model) handleEnvSelectUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newEnvSelect tea.Model
	var cmd tea.Cmd
	newEnvSelect, cmd = m.envSelect.Update(msg)
	m.envSelect = newEnvSelect.(environment.Model)

	if !m.envSelect.Done {
		return m, cmd
	}

	m.environment = m.envSelect.Selected
	m.cloudConfig = m.envSelect.CloudConfig
	m.state = StateMainMenu
	m.updateTask(m.task)
	return m, nil
}

func (m Model) handleMainMenuUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newMainMenu tea.Model
	var cmd tea.Cmd
	newMainMenu, cmd = m.mainMenu.Update(msg)
	m.mainMenu = newMainMenu.(mainmenu.Model)

	if m.mainMenu.SelectedPipeline >= 0 {
		return m.handlePipelineSelected()
	}

	if m.mainMenu.SelectedUtility == mainmenu.UtilityGlobalSetup {
		return m.handleGlobalSetupSelected()
	}

	return m, cmd
}

func (m Model) handlePipelineSelected() (tea.Model, tea.Cmd) {
	m.selectedPipeline = types.Pipeline(m.mainMenu.SelectedPipeline)
	m.persistentState.LastPipeline = m.mainMenu.SelectedPipeline
	m.saveState()

	m.wizard = wizard.New(m.selectedPipeline, m.repoRoot)
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

func (m Model) handleGlobalSetupSelected() (tea.Model, tea.Cmd) {
	m.mainMenu.SelectedUtility = -1
	m.global = globalsetup.New(m.repoRoot)
	m.global.SetSize(m.width, m.height)
	m.pushState(StateGlobalSetup)
	return m, executor.LoadConfigKeys(m.repoRoot, globalsetup.DefaultConfigKeys())
}

func (m Model) handleWizardUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newWizard tea.Model
	var cmd tea.Cmd
	newWizard, cmd = m.wizard.Update(msg)
	m.wizard = newWizard.(wizard.Model)

	m.persistentState.TimeRanges = m.wizard.TimeRanges

	// Only save on key messages to avoid redundant disk I/O on ticks
	if _, ok := msg.(tea.KeyMsg); ok {
		m.saveState()
	}

	if m.wizard.ReadyToExecute {
		command := m.wizard.BuildCommand()
		return m.startExecution(command)
	}

	return m, cmd
}

func (m Model) handleGlobalSetupUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newGlobal tea.Model
	var cmd tea.Cmd
	newGlobal, cmd = m.global.Update(msg)
	m.global = *newGlobal.(*globalsetup.Model)

	if !m.global.Done {
		return m, cmd
	}

	m.state = StateMainMenu
	return m, executor.LoadConfigSummary(m.repoRoot)
}

func (m Model) handleExecutionUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newExec tea.Model
	var cmd tea.Cmd
	newExec, cmd = m.execution.Update(msg)
	m.execution = newExec.(execution.Model)
	return m, cmd
}

func (m Model) handleDashboardUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newDash tea.Model
	var cmd tea.Cmd
	newDash, cmd = m.dashboard.Update(msg)
	m.dashboard = newDash.(dashboard.Model)
	return m, cmd
}

func (m Model) handleHistoryUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newHist tea.Model
	var cmd tea.Cmd
	newHist, cmd = m.historyMdl.Update(msg)
	m.historyMdl = newHist.(history.Model)

	if !m.historyMdl.Done || m.historyMdl.SelectedCommand == "" {
		return m, cmd
	}

	command := m.historyMdl.SelectedCommand
	m.historyMdl.Done = false
	m.historyMdl.SelectedCommand = ""
	return m.startExecution(command)
}

func (m Model) handleQuickActionsOverlay(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newQA tea.Model
	var cmd tea.Cmd
	newQA, cmd = m.quickActions.Update(msg)
	m.quickActions = newQA.(quickactions.Model)

	if !m.quickActions.Done {
		return m, cmd
	}

	action := m.quickActions.SelectedAction
	m.quickActions.Reset()
	m.quickActions.Hide()
	return m.handleQuickAction(action)
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

	pipeline, mode := m.extractCommandParts(m.execCommand)
	success := m.execution.Status == execution.StatusSuccess
	exitCode := m.getExitCode(success)
	endTime := time.Now()

	record := history.ExecutionRecord{
		ID:        endTime.Format("20060102150405"),
		Command:   m.execCommand,
		Pipeline:  pipeline,
		Mode:      mode,
		StartTime: m.execStartTime,
		EndTime:   endTime,
		Duration:  time.Since(m.execStartTime).Seconds(),
		ExitCode:  exitCode,
		Success:   success,
	}

	if err := history.AddRecord(m.repoRoot, record); err != nil {
		// History record cannot be saved - continue without it
		return
	}
}

func (m *Model) extractCommandParts(command string) (pipeline, mode string) {
	parts := strings.Fields(command)
	pipeline = "unknown"
	mode = ""

	if len(parts) >= 2 {
		pipeline = parts[1]
	}
	if len(parts) >= 3 {
		mode = parts[2]
	}

	return pipeline, mode
}

func (m *Model) getExitCode(success bool) int {
	if success {
		return 0
	}
	return 1
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
