package app

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
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
