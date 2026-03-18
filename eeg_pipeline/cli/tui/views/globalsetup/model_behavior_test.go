package globalsetup

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
)

func TestBuildOverridesNormalizesProjectValues(t *testing.T) {
	m := New(t.TempDir())
	m.task = "rest"
	m.randomState = "17"
	m.subjectList = "sub-01, sub-02; sub-03"

	overrides := m.buildOverrides()
	project, ok := overrides["project"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected project overrides, got %#v", overrides)
	}

	if project["task"] != "rest" {
		t.Fatalf("expected task override, got %#v", project["task"])
	}

	if got := project["random_state"]; got != 17 {
		t.Fatalf("expected numeric random_state override, got %#v", got)
	}

	subjects, ok := project["subject_list"].([]string)
	if !ok {
		t.Fatalf("expected subject_list slice, got %#v", project["subject_list"])
	}
	if len(subjects) != 3 {
		t.Fatalf("expected 3 subjects, got %#v", subjects)
	}
	if subjects[0] != "sub-01" || subjects[1] != "sub-02" || subjects[2] != "sub-03" {
		t.Fatalf("unexpected subject list: %#v", subjects)
	}
}

func TestBuildOverridesUsesNullSubjectListWhenEmpty(t *testing.T) {
	m := New(t.TempDir())

	overrides := m.buildOverrides()
	project, ok := overrides["project"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected project overrides, got %#v", overrides)
	}
	if project["subject_list"] != nil {
		t.Fatalf("expected nil subject_list, got %#v", project["subject_list"])
	}
}

func TestViewShowsPathStatusAndStatusMessage(t *testing.T) {
	repoRoot := t.TempDir()
	existingPath := filepath.Join(repoRoot, "existing")
	if err := os.MkdirAll(existingPath, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	m := New(repoRoot)
	m.isLoading = false
	m.sectionIndex = int(sectionPaths)
	m.fieldCursor = 0
	m.bidsRoot = existingPath
	m.bidsRestRoot = filepath.Join(repoRoot, "missing")
	m.statusMessage = "Discovery failed"
	m.statusIsError = true

	rendered := m.View()
	if !strings.Contains(rendered, "Global Setup") {
		t.Fatalf("expected title in render, got %q", rendered)
	}
	if !strings.Contains(rendered, styles.CheckMark) {
		t.Fatalf("expected check mark for existing path, got %q", rendered)
	}
	if !strings.Contains(rendered, styles.WarningMark+" not found") {
		t.Fatalf("expected warning for missing path, got %q", rendered)
	}
	if !strings.Contains(rendered, "[B] browse") {
		t.Fatalf("expected browse hint on focused path, got %q", rendered)
	}
	if !strings.Contains(rendered, "Discovery failed") {
		t.Fatalf("expected status message in render, got %q", rendered)
	}
}

func TestRenderFooterUsesEditingHints(t *testing.T) {
	m := New(t.TempDir())
	m.editingText = true

	footer := m.renderFooter()
	if !strings.Contains(footer, "Type") {
		t.Fatalf("expected editing footer hints, got %q", footer)
	}
	if strings.Contains(footer, "Navigate") {
		t.Fatalf("expected editing footer to replace navigation hints, got %q", footer)
	}
}

func TestUpdateHandlesMessageAndKeyBranches(t *testing.T) {
	repoRoot := t.TempDir()

	t.Run("config load error", func(t *testing.T) {
		m := New(repoRoot)
		updated, cmd := m.Update(messages.ConfigKeysLoadedMsg{
			Values: map[string]interface{}{
				"project.task":       "rest",
				"project.subject_list": []interface{}{"sub-01", "sub-02"},
			},
			Error: errors.New("boom"),
		})

		got := updated.(*Model)
		if got.isLoading {
			t.Fatal("expected loading to stop")
		}
		if got.task != "rest" {
			t.Fatalf("expected task to hydrate, got %q", got.task)
		}
		if got.subjectList != "sub-01, sub-02" {
			t.Fatalf("expected subject list to hydrate, got %q", got.subjectList)
		}
		if !got.statusIsError {
			t.Fatal("expected error status flag")
		}
		if !strings.Contains(got.statusMessage, "Discovery failed") {
			t.Fatalf("unexpected status message: %q", got.statusMessage)
		}
		if cmd != nil {
			t.Fatalf("expected no command, got %#v", cmd)
		}
	})

	t.Run("activate selection and edit commit", func(t *testing.T) {
		m := New(repoRoot)
		m.sectionIndex = int(sectionProject)
		m.fieldCursor = 0

		updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
		got := updated.(*Model)
		if !got.editingText || got.editingField != fieldTask {
			t.Fatalf("expected text edit to start, got %#v", got)
		}
		if cmd != nil {
			t.Fatalf("expected no command on activation, got %#v", cmd)
		}

		got.textBuffer = "rest"
		updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeyEnter})
		got = updated.(*Model)
		if !got.isSaving {
			t.Fatal("expected save to start")
		}
		if got.task != "rest" {
			t.Fatalf("expected task to update, got %q", got.task)
		}
		if cmd == nil {
			t.Fatal("expected save command")
		}
	})

	t.Run("browse and reset", func(t *testing.T) {
		m := New(repoRoot)
		m.sectionIndex = int(sectionPaths)
		m.fieldCursor = 0

		updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("b")})
		got := updated.(*Model)
		if cmd == nil {
			t.Fatal("expected browse command for path field")
		}
		if got.bidsRoot != "" {
			t.Fatalf("unexpected bids root mutation before command execution: %q", got.bidsRoot)
		}

		updated, cmd = got.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
		got = updated.(*Model)
		if cmd == nil {
			t.Fatal("expected reset command")
		}
		if got.statusMessage != "Overrides reset to defaults" {
			t.Fatalf("unexpected reset status: %q", got.statusMessage)
		}
	})

	t.Run("window size", func(t *testing.T) {
		m := New(repoRoot)
		updated, cmd := m.Update(tea.WindowSizeMsg{Width: 88, Height: 31})
		got := updated.(*Model)
		if got.width != 88 || got.height != 31 {
			t.Fatalf("expected size update, got %dx%d", got.width, got.height)
		}
		if cmd != nil {
			t.Fatalf("expected no command on resize, got %#v", cmd)
		}
	})
}
