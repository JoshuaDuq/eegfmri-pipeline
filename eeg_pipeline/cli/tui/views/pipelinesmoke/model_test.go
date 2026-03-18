package pipelinesmoke

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestNewTrimsTaskAndSelectsAll(t *testing.T) {
	m := New("  task  ")
	if m.task != "task" {
		t.Fatalf("expected trimmed task, got %q", m.task)
	}
	if !m.allSelected() {
		t.Fatal("expected all smoke checks selected by default")
	}
}

func TestBuildCommandInsertsTaskAndPipelines(t *testing.T) {
	m := New("task")
	m.ToggleAll(false)
	m.ToggleByID("features")
	m.ToggleByID("plotting")

	cmd := m.BuildCommand()
	want := "scripts/tui_pipeline_smoke.py --task task --pipelines features,plotting"
	if cmd != want {
		t.Fatalf("expected %q, got %q", want, cmd)
	}
}

func TestUpdateHandlesSelectionExecutionAndCancel(t *testing.T) {
	m := New("task")

	m.cursor = len(smokeItems) - 1
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	got := updated.(Model)
	if got.cursor != 0 {
		t.Fatalf("expected wrap to top, got %d", got.cursor)
	}
	if cmd != nil {
		t.Fatalf("expected no command on navigation, got %#v", cmd)
	}

	got.cursor = 0
	updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeySpace})
	got = updated.(Model)
	if got.selected[smokeItems[0].ID] {
		t.Fatal("expected space to toggle the focused item off")
	}
	if cmd != nil {
		t.Fatalf("expected no command on toggle, got %#v", cmd)
	}

	got.ToggleAll(false)
	updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got = updated.(Model)
	if got.Done {
		t.Fatal("expected enter with no selection to stay open")
	}
	if got.statusLine != "Select at least one smoke check" {
		t.Fatalf("unexpected status line: %q", got.statusLine)
	}
	if cmd != nil {
		t.Fatalf("expected no command with empty selection, got %#v", cmd)
	}

	got.ToggleAll(false)
	got.ToggleByID("features")
	updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got = updated.(Model)
	if !got.Done || got.Cancelled {
		t.Fatalf("expected enter to finish the dialog, got %#v", got)
	}
	if got.RunCommand != "scripts/tui_pipeline_smoke.py --task task --pipelines features" {
		t.Fatalf("unexpected run command: %q", got.RunCommand)
	}
	if cmd != nil {
		t.Fatalf("expected no command on completion, got %#v", cmd)
	}

	got.Done = false
	got.Cancelled = false
	got.RunCommand = "keep"
	updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeyEsc})
	got = updated.(Model)
	if !got.Done || !got.Cancelled {
		t.Fatalf("expected esc to cancel, got %#v", got)
	}
	if got.RunCommand != "" {
		t.Fatalf("expected esc to clear command, got %q", got.RunCommand)
	}
	if cmd != nil {
		t.Fatalf("expected no command on cancel, got %#v", cmd)
	}
}

func TestViewAndReset(t *testing.T) {
	m := New("task")
	view := m.View()
	if !strings.Contains(view, "Pipeline Smoke Test") {
		t.Fatalf("expected title in view, got %q", view)
	}
	if !strings.Contains(view, "Task:") {
		t.Fatalf("expected task line in view, got %q", view)
	}
	if !strings.Contains(view, "Selected: 11/11") {
		t.Fatalf("expected selected count in view, got %q", view)
	}

	m.Done = true
	m.Cancelled = true
	m.RunCommand = "cmd"
	m.statusLine = "status"
	m.Reset()
	if m.Done || m.Cancelled || m.RunCommand != "" || m.statusLine != "" {
		t.Fatalf("expected reset to clear transient state, got %#v", m)
	}
}
