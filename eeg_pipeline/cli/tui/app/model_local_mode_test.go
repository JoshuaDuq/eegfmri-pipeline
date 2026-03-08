package app

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"
	"github.com/eeg-pipeline/tui/views/execution"
	"github.com/eeg-pipeline/tui/views/mainmenu"
	"github.com/eeg-pipeline/tui/views/wizard"
)

func TestNewStartsInMainMenu(t *testing.T) {
	m := New()
	if m.state != StateMainMenu {
		t.Fatalf("expected initial state %v, got %v", StateMainMenu, m.state)
	}
}

func TestEscapeFromMainMenuQuits(t *testing.T) {
	m := New()
	m.state = StateMainMenu

	next, cmd := m.handleEscape()
	if cmd == nil {
		t.Fatalf("expected quit command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatalf("expected tea.QuitMsg, got %T", cmd())
	}

	nextModel, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if nextModel.state != StateMainMenu {
		t.Fatalf("expected to remain in main menu, got %v", nextModel.state)
	}
}

func TestHandleConfigKeysLoaded_ReappliesPersistedWizardConfig(t *testing.T) {
	m := Model{
		state:            StatePipelineWizard,
		selectedPipeline: types.PipelineFmri,
		wizard:           wizard.New(types.PipelineFmri, t.TempDir()),
		persistentState: TUIState{
			PipelineConfigs: map[string]map[string]interface{}{
				types.PipelineFmri.String(): {
					"fmriFmriprepImage": "persisted/image:latest",
				},
			},
		},
	}

	m.restoreWizardConfig()
	if got := m.wizard.ExportConfig()["fmriFmriprepImage"]; got != "persisted/image:latest" {
		t.Fatalf("expected restored persisted value, got %v", got)
	}

	m.handleConfigKeysLoaded(messages.ConfigKeysLoadedMsg{
		Values: map[string]interface{}{
			"fmri_preprocessing.fmriprep.image": "yaml/default:image",
		},
	})

	if got := m.wizard.ExportConfig()["fmriFmriprepImage"]; got != "persisted/image:latest" {
		t.Fatalf("expected persisted config to win after config hydration, got %v", got)
	}
}

func TestHandlePipelineSmokeUtilityOpensSelector(t *testing.T) {
	m := New()
	m.state = StateMainMenu
	m.task = "task"
	m.mainMenu.SelectedUtility = mainmenu.UtilityPipelineSmokeTest

	next, cmd := m.handleMainMenuUpdate(tea.KeyMsg{})
	if cmd == nil {
		t.Fatalf("expected selector init command")
	}

	updated, ok := next.(Model)
	if !ok {
		t.Fatalf("expected Model, got %T", next)
	}
	if updated.state != StatePipelineSmoke {
		t.Fatalf("expected state %v, got %v", StatePipelineSmoke, updated.state)
	}
}

func TestView_ExecutionBypassesGlobalTooSmallScreen(t *testing.T) {
	m := New()
	m.state = StateExecution
	m.width = 50
	m.height = 16
	m.execution = execution.New("echo test")
	m.execution.SetSize(50, 16)
	m.execution.AddOutput("[12:00:00] processing")
	m.execution.SetStatus(execution.StatusRunning)

	view := m.View()
	if strings.Contains(view, "Terminal too small") {
		t.Fatalf("expected execution view to render instead of global too-small screen\nview:\n%s", view)
	}
	if !strings.Contains(view, "processing") {
		t.Fatalf("expected execution view to keep logs visible\nview:\n%s", view)
	}
}
