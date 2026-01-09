package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/eeg-pipeline/tui/app"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	ansiDisableMouseClick    = "\033[?1000l"
	ansiDisableMouseButton   = "\033[?1002l"
	ansiDisableMouseTracking = "\033[?1003l"
	ansiDisableSGRMouse      = "\033[?1006l"
	ansiExitAlternateScreen   = "\033[?1049l"
)

func disableMouseTracking() {
	fmt.Print(ansiDisableMouseClick)
	fmt.Print(ansiDisableMouseButton)
	fmt.Print(ansiDisableMouseTracking)
	fmt.Print(ansiDisableSGRMouse)
}

func exitAlternateScreen() {
	fmt.Print(ansiExitAlternateScreen)
}

func resetTerminalAttributes() {
	cmd := exec.Command("stty", "sane")
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to reset terminal attributes: %v\n", err)
	}
}

func resetTerminal() {
	disableMouseTracking()
	exitAlternateScreen()
	resetTerminalAttributes()
}

func handlePanic() {
	if r := recover(); r != nil {
		resetTerminal()
		fmt.Fprintf(os.Stderr, "TUI crashed: %v\n", r)
		os.Exit(1)
	}
}

func main() {
	defer handlePanic()

	program := tea.NewProgram(
		app.New(),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	defer resetTerminal()

	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}
}
