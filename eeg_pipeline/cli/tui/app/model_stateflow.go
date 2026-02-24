package app

import (
	"fmt"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/dashboard"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/globalsetup"
	"github.com/eeg-pipeline/tui/views/history"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/pipelinesmoke"
	"github.com/eeg-pipeline/tui/views/quickactions"
	"github.com/eeg-pipeline/tui/views/wizard"

	tea "github.com/charmbracelet/bubbletea"
)

// State transition and per-view delegation flow.

func (m Model) delegateToCurrentView(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m.state {
	case StateMainMenu:
		return m.handleMainMenuUpdate(msg)
	case StatePipelineWizard:
		return m.handleWizardUpdate(msg)
	case StatePipelineSmoke:
		return m.handlePipelineSmokeUpdate(msg)
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

	if m.mainMenu.SelectedUtility == mainmenu.UtilityPipelineSmokeTest {
		return m.handlePipelineSmokeSelected()
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
	m.wizard.SetTask(m.task)
	m.wizard.SetTimeRanges(m.persistentState.TimeRanges)
	m.restoreWizardConfig()
	m.wizard.SetSubjectsLoading()
	m.pushState(StatePipelineWizard)
	m.mainMenu.SelectedPipeline = -1

	configKeys := []string{"time_frequency_analysis.bands"}
	if m.selectedPipeline == types.PipelineML {
		configKeys = append(configKeys,
			"machine_learning.targets.regression",
			"machine_learning.targets.classification",
			"machine_learning.targets.binary_threshold",
			"machine_learning.data.feature_families",
			"machine_learning.data.feature_bands",
			"machine_learning.data.feature_segments",
			"machine_learning.data.feature_scopes",
			"machine_learning.data.feature_stats",
			"machine_learning.data.feature_harmonization",
			"machine_learning.data.covariates",
			"machine_learning.data.require_trial_ml_safe",
			"machine_learning.classification.model",
			"machine_learning.cv.inner_splits",
			"machine_learning.models.elasticnet.alpha_grid",
			"machine_learning.models.elasticnet.l1_ratio_grid",
			"machine_learning.models.ridge.alpha_grid",
			"machine_learning.models.random_forest.n_estimators",
			"machine_learning.models.random_forest.max_depth_grid",
			"machine_learning.preprocessing.variance_threshold_grid",
		)
	}
	if m.selectedPipeline == types.PipelineFmri || m.selectedPipeline == types.PipelineFmriAnalysis {
		configKeys = append(configKeys,
			"fmri_preprocessing.engine",
			"fmri_preprocessing.fmriprep.image",
			"fmri_preprocessing.fmriprep.output_dir",
			"fmri_preprocessing.fmriprep.work_dir",
			"fmri_preprocessing.fmriprep.fs_license_file",
			"fmri_preprocessing.fmriprep.fs_subjects_dir",
			"fmri_preprocessing.fmriprep.output_spaces",
			"fmri_preprocessing.fmriprep.ignore",
			"fmri_preprocessing.fmriprep.bids_filter_file",
			"fmri_preprocessing.fmriprep.use_aroma",
			"fmri_preprocessing.fmriprep.skip_bids_validation",
			"fmri_preprocessing.fmriprep.stop_on_first_crash",
			"fmri_preprocessing.fmriprep.clean_workdir",
			"fmri_preprocessing.fmriprep.fs_no_reconall",
			"fmri_preprocessing.fmriprep.nthreads",
			"fmri_preprocessing.fmriprep.omp_nthreads",
			"fmri_preprocessing.fmriprep.mem_mb",
			"fmri_preprocessing.fmriprep.extra_args",
			"paths.freesurfer_license",
			"paths.bids_fmri_root",
		)
	}

	cacheKey := fmt.Sprintf("%s|%s", m.task, m.selectedPipeline.GetDataSource())
	subjectCmd := executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
	if cached, ok := m.subjectsCache[cacheKey]; ok {
		// Serve cached subjects instantly; refresh is available via F5.
		subjectCmd = func() tea.Msg { return cached }
		m.pendingSubjectsCacheKey = ""
	} else {
		m.pendingSubjectsCacheKey = cacheKey
	}

	cmds := []tea.Cmd{
		m.wizard.Init(),
		func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
		subjectCmd,
		executor.LoadPlotters(m.repoRoot),
		executor.LoadConfigSummary(m.repoRoot),
		executor.LoadConfigKeys(m.repoRoot, configKeys),
		executor.DiscoverColumns(m.repoRoot, m.task),
		executor.DiscoverTrialTableColumns(m.repoRoot, m.task),
		executor.DiscoverFmriColumns(m.repoRoot, m.task),
		executor.DiscoverROIs(m.repoRoot, m.task),
	}

	// Also discover condition effects columns for plotting (if subjects available)
	if m.selectedPipeline == types.PipelinePlotting {
		// Will be triggered when subjects are loaded
		cmds = append(cmds, executor.DiscoverConditionEffectsColumns(m.repoRoot, m.task, ""))
	}

	return m, tea.Batch(cmds...)
}

func (m Model) handleGlobalSetupSelected() (tea.Model, tea.Cmd) {
	m.mainMenu.SelectedUtility = -1
	m.global = globalsetup.New(m.repoRoot)
	m.global.SetSize(m.width, m.height)
	m.pushState(StateGlobalSetup)
	return m, tea.Batch(
		m.global.Init(),
		executor.LoadConfigKeys(m.repoRoot, globalsetup.DefaultConfigKeys()),
	)
}

func (m Model) handlePipelineSmokeSelected() (tea.Model, tea.Cmd) {
	m.mainMenu.SelectedUtility = -1
	m.pipelineSmoke = pipelinesmoke.New(m.task)
	m.pushState(StatePipelineSmoke)
	return m, tea.Batch(
		m.pipelineSmoke.Init(),
		func() tea.Msg { return tea.WindowSizeMsg{Width: m.width, Height: m.height} },
	)
}

func (m Model) handlePipelineSmokeUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var updated tea.Model
	var cmd tea.Cmd
	updated, cmd = m.pipelineSmoke.Update(msg)
	m.pipelineSmoke = updated.(pipelinesmoke.Model)

	if m.pipelineSmoke.Done {
		if m.pipelineSmoke.Cancelled {
			m.pipelineSmoke.Reset()
			return m.popState()
		}

		command := m.pipelineSmoke.RunCommand
		m.pipelineSmoke.Reset()
		return m.startExecution(command)
	}

	return m, cmd
}

func (m Model) handleWizardUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	var newWizard tea.Model
	var cmd tea.Cmd
	newWizard, cmd = m.wizard.Update(msg)
	m.wizard = newWizard.(wizard.Model)

	m.persistentState.TimeRanges = m.wizard.TimeRanges

	// Only save on key messages to avoid redundant disk I/O on ticks
	if _, ok := msg.(tea.KeyMsg); ok {
		m.saveWizardConfig()
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
	wasDone := m.execution.IsDone()
	var newExec tea.Model
	var cmd tea.Cmd
	newExec, cmd = m.execution.Update(msg)
	m.execution = newExec.(execution.Model)

	// Record to history when execution completes (transitions from running to done)
	if !wasDone && m.execution.IsDone() {
		m.recordExecutionToHistory()
	}

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
	return m, cmd
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
		return m, tea.Batch(
			m.global.Init(),
			executor.LoadConfigKeys(m.repoRoot, globalsetup.DefaultConfigKeys()),
		)
	case quickactions.ActionRefresh:
		// Refresh subjects
		if m.state == StatePipelineWizard {
			m.wizard.SetSubjectsLoading()
			return m, executor.LoadSubjects(m.repoRoot, m.task, m.selectedPipeline)
		}
	case quickactions.ActionValidate:
		cmd := "eeg-pipeline validate --all-subjects"
		if m.task != "" {
			cmd += " --task " + m.task
		}
		return m.startExecution(cmd)
	case quickactions.ActionExport:
		cmd := "eeg-pipeline info --all-subjects --format csv"
		if m.task != "" {
			cmd += " --task " + m.task
		}
		return m.startExecution(cmd)
	}
	return m, nil
}

// startExecution starts pipeline execution locally.
func (m Model) startExecution(command string) (tea.Model, tea.Cmd) {
	m.execution = execution.NewWithRoot(command, m.repoRoot)
	m.execution.SetSize(m.width, m.height)
	m.pushState(StateExecution)
	m.wizard.ReadyToExecute = false

	// Track execution for history
	m.execStartTime = time.Now()
	m.execCommand = command

	// Local mode: run directly
	return m, m.execution.Start()
}

// handleEscape handles the escape key
func (m Model) handleEscape() (tea.Model, tea.Cmd) {
	switch m.state {
	case StateMainMenu:
		return m, tea.Quit
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
			return m.popState()
		}
		return m, nil
	case StatePipelineSmoke:
		return m.popState()
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
	exitCode := 0
	if !success {
		exitCode = 1
	}

	// Use execution model's StartTime if available (when command actually started),
	// otherwise fall back to execStartTime (when execution view was created)
	startTime := m.execStartTime
	if !m.execution.StartTime.IsZero() {
		startTime = m.execution.StartTime
	}

	endTime := time.Now()
	if !m.execution.EndTime.IsZero() {
		endTime = m.execution.EndTime
	}

	record := history.ExecutionRecord{
		ID:        endTime.Format("20060102150405"),
		Command:   m.execCommand,
		Pipeline:  pipeline,
		Mode:      mode,
		StartTime: startTime,
		EndTime:   endTime,
		Duration:  endTime.Sub(startTime).Seconds(),
		ExitCode:  exitCode,
		Success:   success,
	}

	if err := history.AddRecord(m.repoRoot, record); err != nil {
		// History record cannot be saved - continue without it
		return
	}

	// Clear command to prevent duplicate recording
	m.execCommand = ""
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
