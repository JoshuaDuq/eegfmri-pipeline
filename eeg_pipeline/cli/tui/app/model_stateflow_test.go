package app

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/globalsetup"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/pipelinesmoke"
	"github.com/eeg-pipeline/tui/views/quickactions"
	"github.com/eeg-pipeline/tui/views/wizard"

	tea "github.com/charmbracelet/bubbletea"
)

func TestPushPopState(t *testing.T) {
	m := Model{state: StateMainMenu}

	m.pushState(StateDashboard)
	if m.state != StateDashboard {
		t.Fatalf("expected state=%v, got %v", StateDashboard, m.state)
	}
	if len(m.navStack) != 1 || m.navStack[0] != StateMainMenu {
		t.Fatalf("unexpected nav stack: %+v", m.navStack)
	}

	next, cmd := m.popState()
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.state != StateMainMenu {
		t.Fatalf("expected state=%v, got %v", StateMainMenu, updated.state)
	}
	if len(updated.navStack) != 0 {
		t.Fatalf("expected empty nav stack, got %+v", updated.navStack)
	}
}

func TestHandleMainMenuUpdate_PipelineSelectionDispatchesToWizard(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:         StateMainMenu,
		mainMenu:      mainmenu.New(),
		repoRoot:      repoRoot,
		task:          "stroop",
		width:         120,
		height:        40,
		subjectsCache: make(map[string]messages.SubjectsLoadedMsg),
	}
	m.mainMenu.SetCursor(int(types.PipelineBehavior))

	next, cmd := m.handleMainMenuUpdate(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.state != StatePipelineWizard {
		t.Fatalf("expected state=%v, got %v", StatePipelineWizard, updated.state)
	}
	if updated.selectedPipeline != types.PipelineBehavior {
		t.Fatalf("expected selectedPipeline=%v, got %v", types.PipelineBehavior, updated.selectedPipeline)
	}
	if updated.mainMenu.SelectedPipeline != -1 {
		t.Fatalf("expected main menu SelectedPipeline to be reset, got %d", updated.mainMenu.SelectedPipeline)
	}
	if len(updated.navStack) != 1 || updated.navStack[0] != StateMainMenu {
		t.Fatalf("unexpected nav stack: %+v", updated.navStack)
	}
	expectedCacheKey := fmt.Sprintf("%s|%s", updated.task, updated.selectedPipeline.GetDataSource())
	if updated.pendingSubjectsCacheKey != expectedCacheKey {
		t.Fatalf("expected pendingSubjectsCacheKey=%q, got %q", expectedCacheKey, updated.pendingSubjectsCacheKey)
	}
}

func TestHandleMainMenuUpdate_UtilityDispatchesToSmokeTest(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:     StateMainMenu,
		mainMenu:  mainmenu.New(),
		repoRoot:  repoRoot,
		task:      "task",
		width:     120,
		height:    40,
		execution: execution.New("echo test"),
	}
	m.mainMenu.SelectedUtility = mainmenu.UtilityPipelineSmokeTest

	next, cmd := m.handleMainMenuUpdate(tea.WindowSizeMsg{Width: 120, Height: 40})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.state != StatePipelineSmoke {
		t.Fatalf("expected state=%v, got %v", StatePipelineSmoke, updated.state)
	}
	if updated.mainMenu.SelectedUtility != -1 {
		t.Fatalf("expected main menu SelectedUtility to be reset, got %d", updated.mainMenu.SelectedUtility)
	}
	if len(updated.navStack) != 1 || updated.navStack[0] != StateMainMenu {
		t.Fatalf("unexpected nav stack: %+v", updated.navStack)
	}
	if !strings.Contains(updated.pipelineSmoke.BuildCommand(), "--task task") {
		t.Fatalf("expected pipeline smoke command to include task, got %q", updated.pipelineSmoke.BuildCommand())
	}
}

func TestHandleMainMenuUpdate_UtilityDispatchesToGlobalSetup(t *testing.T) {
	repoRoot := t.TempDir()
	m := Model{
		state:     StateMainMenu,
		mainMenu:  mainmenu.New(),
		repoRoot:  repoRoot,
		task:      "task",
		width:     120,
		height:    40,
		execution: execution.New("echo test"),
	}
	m.mainMenu.SelectedUtility = mainmenu.UtilityGlobalSetup

	next, cmd := m.handleMainMenuUpdate(tea.WindowSizeMsg{Width: 120, Height: 40})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.state != StateGlobalSetup {
		t.Fatalf("expected state=%v, got %v", StateGlobalSetup, updated.state)
	}
	if updated.mainMenu.SelectedUtility != -1 {
		t.Fatalf("expected main menu SelectedUtility to be reset, got %d", updated.mainMenu.SelectedUtility)
	}
	if len(updated.navStack) != 1 || updated.navStack[0] != StateMainMenu {
		t.Fatalf("unexpected nav stack: %+v", updated.navStack)
	}
}

func TestHandleEscape_FromGlobalSetup_ReturnsMainMenu(t *testing.T) {
	m := Model{state: StateGlobalSetup}

	next, cmd := m.handleEscape()
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.state != StateMainMenu {
		t.Fatalf("expected state=%v, got %v", StateMainMenu, updated.state)
	}
}

func TestHandleEscape_FromExecution_OnlyPopsWhenDone(t *testing.T) {
	t.Run("not-done-stays", func(t *testing.T) {
		m := Model{
			state:     StateExecution,
			navStack:  []AppState{StateMainMenu},
			execution: execution.New("echo test"),
		}
		m.execution.SetStatus(execution.StatusRunning)

		next, cmd := m.handleEscape()
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		updated := next.(Model)
		if updated.state != StateExecution {
			t.Fatalf("expected state=%v, got %v", StateExecution, updated.state)
		}
	})

	t.Run("done-pops", func(t *testing.T) {
		m := Model{
			state:     StateExecution,
			navStack:  []AppState{StateMainMenu},
			execution: execution.New("echo test"),
		}
		m.execution.SetStatus(execution.StatusSuccess)

		next, cmd := m.handleEscape()
		if cmd != nil {
			t.Fatalf("expected nil cmd, got %T", cmd)
		}
		updated := next.(Model)
		if updated.state != StateMainMenu {
			t.Fatalf("expected state=%v, got %v", StateMainMenu, updated.state)
		}
	})
}

func TestHandleEscape_FromWizardAtFirstStep_PopsState(t *testing.T) {
	m := Model{
		state:    StatePipelineWizard,
		navStack: []AppState{StateMainMenu},
		wizard:   wizard.New(types.PipelineBehavior, "."),
	}

	next, cmd := m.handleEscape()
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.state != StateMainMenu {
		t.Fatalf("expected state=%v, got %v", StateMainMenu, updated.state)
	}
}

func TestHandleEscape_FromWizardAfterAdvancing_ClearsScreen(t *testing.T) {
	w := wizard.New(types.PipelineBehavior, ".")
	w.SetSubjects([]types.SubjectStatus{
		{ID: "sub-01", HasFeatures: true},
	})
	sized, _ := w.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	w = sized.(wizard.Model)

	advanced, _ := w.Update(tea.KeyMsg{Type: tea.KeyEnter})
	w = advanced.(wizard.Model)

	m := Model{
		state:  StatePipelineWizard,
		wizard: w,
	}

	next, cmd := m.handleEscape()
	if cmd == nil {
		t.Fatal("expected clear-screen cmd")
	}
	updated := next.(Model)
	if updated.state != StatePipelineWizard {
		t.Fatalf("expected state to remain wizard, got %v", updated.state)
	}
}

func TestStartExecutionTransitionsAndTracksCommand(t *testing.T) {
	m := Model{
		state:    StatePipelineWizard,
		navStack: []AppState{StateMainMenu},
		repoRoot: t.TempDir(),
		width:    120,
		height:   40,
		wizard:   wizard.New(types.PipelineBehavior, "."),
	}

	next, cmd := m.startExecution("eeg-pipeline behavior compute")
	if cmd == nil {
		t.Fatal("expected non-nil execution cmd")
	}

	updated := next.(Model)
	if updated.state != StateExecution {
		t.Fatalf("expected state=%v, got %v", StateExecution, updated.state)
	}
	if updated.execCommand != "eeg-pipeline behavior compute" {
		t.Fatalf("expected execCommand to be set, got %q", updated.execCommand)
	}
	if updated.execution.Command != "eeg-pipeline behavior compute" {
		t.Fatalf("expected execution.Command to be set, got %q", updated.execution.Command)
	}
	if updated.execution.RepoRoot != updated.repoRoot {
		t.Fatalf("expected execution.RepoRoot=%q, got %q", updated.repoRoot, updated.execution.RepoRoot)
	}
	if len(updated.navStack) != 2 || updated.navStack[1] != StatePipelineWizard {
		t.Fatalf("expected wizard state to be pushed onto stack, got %+v", updated.navStack)
	}
}

func TestHandleQuickAction_StateTransitions(t *testing.T) {
	base := Model{
		state:    StateMainMenu,
		width:    120,
		height:   40,
		task:     "task",
		repoRoot: t.TempDir(),
	}

	t.Run("opens-dashboard", func(t *testing.T) {
		next, cmd := base.handleQuickAction(quickactions.ActionStats)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StateDashboard {
			t.Fatalf("expected state=%v, got %v", StateDashboard, updated.state)
		}
		if len(updated.navStack) != 1 || updated.navStack[0] != StateMainMenu {
			t.Fatalf("unexpected nav stack: %+v", updated.navStack)
		}
	})

	t.Run("opens-history", func(t *testing.T) {
		next, cmd := base.handleQuickAction(quickactions.ActionHistory)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StateHistory {
			t.Fatalf("expected state=%v, got %v", StateHistory, updated.state)
		}
	})

	t.Run("opens-global-setup", func(t *testing.T) {
		next, cmd := base.handleQuickAction(quickactions.ActionConfig)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StateGlobalSetup {
			t.Fatalf("expected state=%v, got %v", StateGlobalSetup, updated.state)
		}
	})

	t.Run("refresh-only-in-wizard", func(t *testing.T) {
		m := base
		m.state = StatePipelineWizard
		m.selectedPipeline = types.PipelineBehavior
		m.wizard = wizard.New(types.PipelineBehavior, ".")

		next, cmd := m.handleQuickAction(quickactions.ActionRefresh)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StatePipelineWizard {
			t.Fatalf("expected state=%v, got %v", StatePipelineWizard, updated.state)
		}
	})

	t.Run("validate-starts-execution", func(t *testing.T) {
		m := base
		m.task = "stroop"
		m.wizard = wizard.New(types.PipelineBehavior, ".")

		next, cmd := m.handleQuickAction(quickactions.ActionValidate)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StateExecution {
			t.Fatalf("expected state=%v, got %v", StateExecution, updated.state)
		}
		if updated.execCommand == "" {
			t.Fatal("expected execCommand to be set")
		}
	})

	t.Run("export-starts-execution", func(t *testing.T) {
		m := base
		m.task = ""
		m.wizard = wizard.New(types.PipelineBehavior, ".")

		next, cmd := m.handleQuickAction(quickactions.ActionExport)
		if cmd == nil {
			t.Fatal("expected non-nil cmd")
		}
		updated := next.(Model)
		if updated.state != StateExecution {
			t.Fatalf("expected state=%v, got %v", StateExecution, updated.state)
		}
		if updated.execCommand == "" {
			t.Fatal("expected execCommand to be set")
		}
	})
}

func TestHandleGlobalMessages_WindowSizeIsPropagated(t *testing.T) {
	m := Model{
		state:         StateMainMenu,
		mainMenu:      mainmenu.New(),
		wizard:        wizard.New(types.PipelineBehavior, "."),
		pipelineSmoke: pipelinesmoke.New(""),
		execution:     execution.New("echo test"),
		global:        globalsetup.New(t.TempDir()),
	}

	next, cmd := m.handleGlobalMessages(tea.WindowSizeMsg{Width: 101, Height: 29})
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.width != 101 || updated.height != 29 {
		t.Fatalf("expected size 101x29, got %dx%d", updated.width, updated.height)
	}
}

func TestHandleGlobalMessages_RefreshSubjectsOnlyInWizard(t *testing.T) {
	m := Model{
		state:            StatePipelineWizard,
		selectedPipeline: types.PipelineBehavior,
		repoRoot:         t.TempDir(),
		task:             "task",
		wizard:           wizard.New(types.PipelineBehavior, "."),
	}

	next, cmd := m.handleGlobalMessages(messages.RefreshSubjectsMsg{})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.pendingSubjectsCacheKey != "" {
		t.Fatalf("expected pendingSubjectsCacheKey to remain empty (value-receiver), got %q", updated.pendingSubjectsCacheKey)
	}
}

func TestHandleGlobalMessages_TaskUpdateInWizardStartsRefresh(t *testing.T) {
	m := Model{
		state:            StatePipelineWizard,
		selectedPipeline: types.PipelineBehavior,
		repoRoot:         t.TempDir(),
		task:             "old",
		mainMenu:         mainmenu.New(),
		pipelineSmoke:    pipelinesmoke.New("old"),
		wizard:           wizard.New(types.PipelineBehavior, "."),
		subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
	}
	m.mainMenu.Task = "old"

	next, cmd := m.handleGlobalMessages(messages.TaskUpdatedMsg{Task: "new"})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.task != "new" {
		t.Fatalf("expected task=%q, got %q", "new", updated.task)
	}
	if updated.mainMenu.Task != "new" {
		t.Fatalf("expected mainMenu.Task=%q, got %q", "new", updated.mainMenu.Task)
	}
	if !strings.Contains(updated.pipelineSmoke.BuildCommand(), "--task new") {
		t.Fatalf("expected pipeline smoke command to include updated task, got %q", updated.pipelineSmoke.BuildCommand())
	}
	expectedCacheKey := fmt.Sprintf("%s|%s", updated.task, updated.selectedPipeline.GetDataSource())
	if updated.pendingSubjectsCacheKey != expectedCacheKey {
		t.Fatalf("expected pendingSubjectsCacheKey=%q, got %q", expectedCacheKey, updated.pendingSubjectsCacheKey)
	}
}

func TestHandleGlobalMessages_ConfigLoadedTaskChangeInWizardStartsRefresh(t *testing.T) {
	m := Model{
		state:            StatePipelineWizard,
		selectedPipeline: types.PipelineBehavior,
		repoRoot:         t.TempDir(),
		task:             "old",
		mainMenu:         mainmenu.New(),
		pipelineSmoke:    pipelinesmoke.New("old"),
		wizard:           wizard.New(types.PipelineBehavior, "."),
		subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
	}
	m.mainMenu.Task = "old"

	next, cmd := m.handleGlobalMessages(messages.ConfigLoadedMsg{
		Summary: messages.ConfigSummary{Task: "new"},
	})
	if cmd == nil {
		t.Fatal("expected non-nil cmd")
	}
	updated := next.(Model)
	if updated.task != "new" {
		t.Fatalf("expected task=%q, got %q", "new", updated.task)
	}
	expectedCacheKey := fmt.Sprintf("%s|%s", updated.task, updated.selectedPipeline.GetDataSource())
	if updated.pendingSubjectsCacheKey != expectedCacheKey {
		t.Fatalf("expected pendingSubjectsCacheKey=%q, got %q", expectedCacheKey, updated.pendingSubjectsCacheKey)
	}
}

func TestHandleGlobalMessages_SubjectsLoadedCachesAndClearsPendingKey(t *testing.T) {
	m := Model{
		state:                   StatePipelineWizard,
		selectedPipeline:        types.PipelineBehavior,
		repoRoot:                t.TempDir(),
		task:                    "task",
		wizard:                  wizard.New(types.PipelineBehavior, "."),
		execution:               execution.New("echo test"),
		subjectsCache:           make(map[string]messages.SubjectsLoadedMsg),
		pendingSubjectsCacheKey: "task|eeg",
	}

	next, cmd := m.handleGlobalMessages(messages.SubjectsLoadedMsg{
		Subjects: []messages.SubjectInfo{{ID: "sub-01"}},
	})
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.pendingSubjectsCacheKey != "" {
		t.Fatalf("expected pendingSubjectsCacheKey to be cleared, got %q", updated.pendingSubjectsCacheKey)
	}
	if _, ok := updated.subjectsCache["task|eeg"]; !ok {
		t.Fatalf("expected subjectsCache to contain key %q", "task|eeg")
	}
}

func TestHandleGlobalMessages_SubjectsLoadedErrorDoesNotPanic(t *testing.T) {
	m := Model{
		state:            StatePipelineWizard,
		selectedPipeline: types.PipelineBehavior,
		repoRoot:         t.TempDir(),
		task:             "task",
		wizard:           wizard.New(types.PipelineBehavior, "."),
		execution:        execution.New("echo test"),
		subjectsCache:    make(map[string]messages.SubjectsLoadedMsg),
	}

	next, cmd := m.handleGlobalMessages(messages.SubjectsLoadedMsg{
		Error: errors.New("boom"),
	})
	if cmd != nil {
		t.Fatalf("expected nil cmd, got %T", cmd)
	}
	updated := next.(Model)
	if updated.state != StatePipelineWizard {
		t.Fatalf("expected state=%v, got %v", StatePipelineWizard, updated.state)
	}
}
