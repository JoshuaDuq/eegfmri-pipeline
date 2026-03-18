package app

import (
	"errors"
	"testing"

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

func TestHandleKeyMessageRoutesShortcuts(t *testing.T) {
	repoRoot := t.TempDir()
	base := Model{
		repoRoot:      repoRoot,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
		mainMenu:      mainmenu.New(),
		wizard:        wizard.New(types.PipelineBehavior, repoRoot),
		pipelineSmoke: pipelinesmoke.New("task"),
		execution:     execution.New("eeg-pipeline behavior compute"),
		global:        globalsetup.New(repoRoot),
		dashboard:     dashboard.New(repoRoot),
		historyMdl:    history.New(repoRoot),
		quickActions:  quickactions.New(),
	}

	t.Run("quit-returns-command-unless-execution-running", func(t *testing.T) {
		m := base
		m.state = StateMainMenu
		_, cmd := m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")})
		if cmd == nil {
			t.Fatal("expected quit command")
		}

		m.state = StateExecution
		m.execution.SetStatus(execution.StatusRunning)
		_, cmd = m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")})
		if cmd != nil {
			t.Fatalf("expected nil quit command while running, got %T", cmd)
		}
	})

	t.Run("escape-and-enter", func(t *testing.T) {
		m := base
		m.state = StateGlobalSetup
		next, cmd := m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyEsc})
		if cmd != nil {
			t.Fatalf("expected nil cmd from escape, got %T", cmd)
		}
		if next.(Model).state != StateMainMenu {
			t.Fatalf("expected escape to return to main menu")
		}

		m.state = StateExecution
		m.execution.SetStatus(execution.StatusSuccess)
		next, cmd = m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyEnter})
		if cmd != nil {
			t.Fatalf("expected nil cmd from enter, got %T", cmd)
		}
		if next.(Model).state != StateMainMenu {
			t.Fatalf("expected enter to return to main menu")
		}
	})

	t.Run("restart-pull-open-overlay", func(t *testing.T) {
		m := base
		m.state = StateExecution
		m.execution.SetStatus(execution.StatusSuccess)

		next, cmd := m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
		if cmd == nil {
			t.Fatal("expected restart command")
		}
		if next.(Model).state != StateExecution {
			t.Fatalf("expected restart to keep execution state")
		}

		if _, cmd := m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("p")}); cmd != nil {
			t.Fatalf("expected pull-results to be a no-op, got %T", cmd)
		}

		m.state = StateMainMenu
		next, cmd = m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
		if cmd == nil {
			t.Fatal("expected dashboard command")
		}
		if next.(Model).state != StateDashboard {
			t.Fatalf("expected dashboard state, got %v", next.(Model).state)
		}

		m = base
		m.state = StateMainMenu
		next, cmd = m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("h")})
		if cmd == nil {
			t.Fatal("expected history command")
		}
		if next.(Model).state != StateHistory {
			t.Fatalf("expected history state, got %v", next.(Model).state)
		}

		m = base
		m.state = StateMainMenu
		next, cmd = m.handleKeyMessage(tea.KeyMsg{Type: tea.KeyCtrlK})
		if cmd == nil {
			t.Fatal("expected quick actions command")
		}
		if !next.(Model).quickActions.Visible {
			t.Fatal("expected quick actions overlay to become visible")
		}
	})
}

func TestHandleGlobalMessagesRoutesDiscoveryAndConfigUpdates(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:         StatePipelineWizard,
		repoRoot:      repoRoot,
		task:          "task",
		selectedPipeline: types.PipelineBehavior,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
		wizard:        wizard.New(types.PipelineBehavior, repoRoot),
		pipelineSmoke: pipelinesmoke.New("task"),
		execution:     execution.New("echo test"),
		global:        globalsetup.New(repoRoot),
		dashboard:     dashboard.New(repoRoot),
		historyMdl:    history.New(repoRoot),
		quickActions:  quickactions.New(),
	}

	plottersNext, plottersCmd := m.handleGlobalMessages(messages.PlottersLoadedMsg{
		FeaturePlotters: map[string][]messages.PlotterInfo{
			"power": {
				{ID: "power.topo", Category: "power", Name: "Power Topography"},
			},
		},
	})
	if plottersCmd != nil {
		t.Fatalf("expected nil cmd, got %T", plottersCmd)
	}
	_ = plottersNext.(Model)

	m.handleGlobalMessages(messages.ColumnsDiscoveredMsg{
		Source:  "condition_effects",
		Columns: []string{"condition"},
		Values:  map[string][]string{"condition": []string{"A", "B"}},
	})
	m.handleGlobalMessages(messages.ColumnsDiscoveredMsg{
		Source:  "trial_table",
		Columns: []string{"response"},
		Values:  map[string][]string{"response": []string{"1", "0"}},
	})
	m.handleGlobalMessages(messages.FmriColumnsDiscoveredMsg{
		Columns: []string{"trial_type"},
		Values:  map[string][]string{"trial_type": []string{"cue"}},
		Source:  "events",
	})
	m.handleGlobalMessages(messages.ROIsDiscoveredMsg{ROIs: []string{"Frontal"}})
	m.handleGlobalMessages(messages.MultigroupStatsDiscoveredMsg{
		Available:    true,
		Groups:       []string{"control", "treated"},
		NFeatures:    2,
		NSignificant: 1,
		File:         "stats.json",
	})

	next, cmd := m.handleGlobalMessages(messages.ConfigLoadedMsg{
		Summary: messages.ConfigSummary{Task: "rest"},
	})
	if cmd == nil {
		t.Fatal("expected config load to refresh subjects in wizard state")
	}
	if next.(Model).task != "rest" {
		t.Fatalf("expected task to update from config load")
	}

	next, cmd = m.handleGlobalMessages(messages.TaskUpdatedMsg{Task: "new"})
	if cmd == nil {
		t.Fatal("expected task update to refresh subjects in wizard state")
	}
	if next.(Model).task != "new" {
		t.Fatalf("expected task to update from task message")
	}

	next, cmd = m.handleGlobalMessages(messages.ConfigKeysLoadedMsg{
		Values: map[string]interface{}{
			"project.task": "rest",
		},
	})
	if cmd != nil {
		t.Fatalf("expected nil cmd from config-keys load, got %T", cmd)
	}
	if next.(Model).task != "task" {
		t.Fatalf("expected config-keys load to keep current task when not in global setup")
	}

	_, cmd = m.handleGlobalMessages(messages.RefreshSubjectsMsg{})
	if cmd == nil {
		t.Fatal("expected refresh-subjects command in wizard state")
	}

	_, cmd = m.handleGlobalMessages(messages.ConfigLoadedMsg{
		Summary: messages.ConfigSummary{Task: "rest"},
		Error:   errors.New("boom"),
	})
	if cmd != nil {
		t.Fatalf("expected config error to short-circuit with nil cmd, got %T", cmd)
	}
}
