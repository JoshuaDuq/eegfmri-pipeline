package quickactions

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestShowHideReset(t *testing.T) {
	m := New()
	if m.Visible {
		t.Fatalf("expected new model to be hidden")
	}

	m.Show()
	if !m.Visible {
		t.Fatalf("expected Show to set Visible")
	}
	if m.Done {
		t.Fatalf("expected Show to clear Done")
	}
	if m.cursor != 0 {
		t.Fatalf("expected Show to reset cursor, got %d", m.cursor)
	}

	m.Hide()
	if m.Visible {
		t.Fatalf("expected Hide to clear Visible")
	}

	m.Done = true
	m.SelectedAction = ActionExport
	m.Reset()
	if m.Done {
		t.Fatalf("expected Reset to clear Done")
	}
	if m.SelectedAction != ActionStats {
		t.Fatalf("expected Reset to set default action, got %v", m.SelectedAction)
	}
}

func TestUpdateNavigationAndSelection(t *testing.T) {
	m := New()
	m.Visible = true

	m.cursor = 0
	model, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = model.(Model)
	if m.cursor != len(quickActions)-1 {
		t.Fatalf("expected wrap on up, got %d", m.cursor)
	}

	model, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = model.(Model)
	if m.cursor != 0 {
		t.Fatalf("expected wrap on down, got %d", m.cursor)
	}

	model, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = model.(Model)
	if m.cursor != 1 {
		t.Fatalf("expected cursor to advance, got %d", m.cursor)
	}

	model, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = model.(Model)
	if !m.Done {
		t.Fatalf("expected enter to mark Done")
	}
	if m.SelectedAction != quickActions[m.cursor].Type {
		t.Fatalf("expected selection to match cursor")
	}

	m.Done = false
	model, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace})
	m = model.(Model)
	if !m.Done {
		t.Fatalf("expected space to mark Done")
	}
}

func TestUpdateShortcutsAndEsc(t *testing.T) {
	m := New()
	m.Visible = true

	model, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("h")})
	m = model.(Model)
	if !m.Done {
		t.Fatalf("expected shortcut to mark Done")
	}
	if m.SelectedAction != ActionHistory {
		t.Fatalf("expected shortcut to select history, got %v", m.SelectedAction)
	}

	m.Visible = true
	model, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = model.(Model)
	if m.Visible {
		t.Fatalf("expected esc to hide")
	}
}

func TestViewVisibilityAndContent(t *testing.T) {
	m := New()
	if view := m.View(); view != "" {
		t.Fatalf("expected empty view when hidden")
	}

	m.Show()
	view := m.View()
	if !strings.Contains(view, "Quick Actions") {
		t.Fatalf("expected title in view")
	}
	for _, action := range quickActions {
		if !strings.Contains(view, action.Name) {
			t.Fatalf("expected action %q in view", action.Name)
		}
	}
	if !strings.Contains(view, "shortcuts or") {
		t.Fatalf("expected footer in view")
	}
}

func TestRenderActionDescriptionToggles(t *testing.T) {
	m := New()
	action := quickActions[0]

	line := m.renderAction(action, false)
	if strings.Contains(line, action.Description) {
		t.Fatalf("expected non-cursor line to omit description")
	}

	line = m.renderAction(action, true)
	if !strings.Contains(line, action.Description) {
		t.Fatalf("expected cursor line to include description")
	}
}
