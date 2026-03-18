package app

import (
	"testing"
	"time"

	"github.com/eeg-pipeline/tui/messages"
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

func TestDelegateToCurrentViewRoutesAndOverlay(t *testing.T) {
	repoRoot := t.TempDir()

	t.Run("main menu", func(t *testing.T) {
		m := Model{
			state:            StateMainMenu,
			mainMenu:         mainmenu.New(),
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New(""),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StateMainMenu {
			t.Fatalf("expected state to remain main menu")
		}
	})

	t.Run("quick actions overlay", func(t *testing.T) {
		m := Model{
			state:         AppState(-1),
			repoRoot:      repoRoot,
			subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
			quickActions:   quickactions.New(),
		}
		m.quickActions.Show()
		m.quickActions.Done = true
		m.quickActions.SelectedAction = quickactions.ActionStats

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd == nil {
			t.Fatal("expected non-nil cmd from overlay action")
		}
		updated := next.(Model)
		if updated.state != StateDashboard {
			t.Fatalf("expected dashboard state, got %v", updated.state)
		}
		if updated.quickActions.Visible {
			t.Fatal("expected quick actions overlay to hide after selection")
		}
	})

	t.Run("pipeline smoke", func(t *testing.T) {
		m := Model{
			state:            StatePipelineSmoke,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New("task"),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StatePipelineSmoke {
			t.Fatalf("expected pipeline smoke state to remain active")
		}
	})

	t.Run("global setup", func(t *testing.T) {
		m := Model{
			state:            StateGlobalSetup,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New("task"),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StateGlobalSetup {
			t.Fatalf("expected global setup state to remain active")
		}
	})

	t.Run("execution", func(t *testing.T) {
		m := Model{
			state:            StateExecution,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New("task"),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StateExecution {
			t.Fatalf("expected execution state to remain active")
		}
	})

	t.Run("dashboard", func(t *testing.T) {
		m := Model{
			state:            StateDashboard,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New("task"),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StateDashboard {
			t.Fatalf("expected dashboard state to remain active")
		}
	})

	t.Run("history", func(t *testing.T) {
		m := Model{
			state:            StateHistory,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			pipelineSmoke:    pipelinesmoke.New("task"),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}

		next, cmd := m.delegateToCurrentView(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		if next.(Model).state != StateHistory {
			t.Fatalf("expected history state to remain active")
		}
	})
}

func TestHandlePipelineSmokeUpdate_CompletesOrCancels(t *testing.T) {
	repoRoot := t.TempDir()

	t.Run("cancelled", func(t *testing.T) {
		m := Model{
			state:            StatePipelineSmoke,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			pipelineSmoke:    pipelinesmoke.New("task"),
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}
		m.pipelineSmoke.Done = true
		m.pipelineSmoke.Cancelled = true
		m.pipelineSmoke.RunCommand = "eeg-pipeline validate --all-subjects"

		next, cmd := m.handlePipelineSmokeUpdate(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		updated := next.(Model)
		if updated.state != StateMainMenu {
			t.Fatalf("expected to return to main menu, got %v", updated.state)
		}
	})

	t.Run("start execution", func(t *testing.T) {
		m := Model{
			state:            StatePipelineSmoke,
			navStack:         []AppState{StateMainMenu},
			repoRoot:         repoRoot,
			pipelineSmoke:    pipelinesmoke.New("task"),
			subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
			wizard:           wizard.New(types.PipelineBehavior, repoRoot),
			global:           globalsetup.New(repoRoot),
			execution:        execution.New("echo test"),
			dashboard:        dashboard.New(repoRoot),
			historyMdl:       history.New(repoRoot),
			quickActions:     quickactions.New(),
		}
		m.pipelineSmoke.Done = true
		m.pipelineSmoke.RunCommand = "eeg-pipeline validate --all-subjects"

		next, cmd := m.handlePipelineSmokeUpdate(tea.WindowSizeMsg{Width: 100, Height: 30})
		if cmd == nil {
			t.Fatal("expected execution command")
		}
		updated := next.(Model)
		if updated.state != StateExecution {
			t.Fatalf("expected execution state, got %v", updated.state)
		}
		if updated.execCommand == "" {
			t.Fatal("expected execCommand to be set")
		}
	})
}

func TestHandleWizardUpdate_ReadinessStartsExecution(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:         StatePipelineWizard,
		navStack:      []AppState{StateMainMenu},
		repoRoot:      repoRoot,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
		wizard:        wizard.New(types.PipelineBehavior, repoRoot),
		global:        globalsetup.New(repoRoot),
		execution:     execution.New("echo test"),
		dashboard:     dashboard.New(repoRoot),
		historyMdl:    history.New(repoRoot),
		quickActions:  quickactions.New(),
	}
	m.wizard.ReadyToExecute = true

	next, cmd := m.handleWizardUpdate(tea.WindowSizeMsg{Width: 100, Height: 30})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.state != StateExecution {
		t.Fatalf("expected execution state, got %v", updated.state)
	}
	if updated.execCommand == "" {
		t.Fatal("expected execCommand to be set")
	}
}

func TestHandleGlobalSetupUpdate_DoneReturnsToMainMenu(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:         StateGlobalSetup,
		navStack:      []AppState{StateMainMenu},
		repoRoot:      repoRoot,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
		global:        globalsetup.New(repoRoot),
		wizard:        wizard.New(types.PipelineBehavior, repoRoot),
		execution:     execution.New("echo test"),
		dashboard:     dashboard.New(repoRoot),
		historyMdl:    history.New(repoRoot),
		quickActions:  quickactions.New(),
	}
	m.global.Done = true

	next, cmd := m.handleGlobalSetupUpdate(tea.WindowSizeMsg{Width: 100, Height: 30})
	if cmd == nil {
		t.Fatal("expected config reload command")
	}
	updated := next.(Model)
	if updated.state != StateMainMenu {
		t.Fatalf("expected main menu state, got %v", updated.state)
	}
}

func TestHandleExecutionUpdate_RecordsHistoryOnCompletion(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:         StateExecution,
		navStack:      []AppState{StateMainMenu},
		repoRoot:      repoRoot,
		execution:     execution.New("eeg-pipeline behavior compute"),
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
		wizard:        wizard.New(types.PipelineBehavior, repoRoot),
		global:        globalsetup.New(repoRoot),
		dashboard:     dashboard.New(repoRoot),
		historyMdl:    history.New(repoRoot),
		quickActions:  quickactions.New(),
	}
	m.execCommand = "eeg-pipeline behavior compute"
	m.execution.SetStatus(execution.StatusRunning)
	m.execution.StartTime = time.Now().Add(-2 * time.Minute)

	next, cmd := m.handleExecutionUpdate(messages.CommandDoneMsg{ExitCode: 0, Success: true})
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.execCommand != "" {
		t.Fatalf("expected execCommand to be cleared, got %q", updated.execCommand)
	}

	records, err := history.LoadRecentRecords(repoRoot, 1)
	if err != nil {
		t.Fatalf("LoadRecentRecords error: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 history record, got %d", len(records))
	}
	if records[0].Pipeline != "behavior" {
		t.Fatalf("expected pipeline behavior, got %q", records[0].Pipeline)
	}
	if records[0].Mode != "compute" {
		t.Fatalf("expected mode compute, got %q", records[0].Mode)
	}
	if !records[0].Success {
		t.Fatal("expected successful record")
	}
}
