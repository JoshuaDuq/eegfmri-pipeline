package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/eeg-pipeline/tui/app"

	tea "github.com/charmbracelet/bubbletea"
)

// Version is set at build time via -ldflags "-X main.Version=x.y.z".
var Version = "dev"

const (
	ansiDisableMouseTracking = "\033[?1000l\033[?1002l\033[?1003l\033[?1006l"
	ansiExitAlternateScreen  = "\033[?1049l"
)

func disableMouseTracking() {
	fmt.Print(ansiDisableMouseTracking)
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

	appModel := app.New(Version)
	program := tea.NewProgram(
		appModel,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	_, err := program.Run()
	resetTerminal()

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}

}
