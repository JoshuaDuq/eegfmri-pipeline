package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/eeg-pipeline/tui/app"
	"github.com/eeg-pipeline/tui/cloud"

	tea "github.com/charmbracelet/bubbletea"
)

const (
	ansiDisableMouseClick    = "\033[?1000l"
	ansiDisableMouseButton   = "\033[?1002l"
	ansiDisableMouseTracking = "\033[?1003l"
	ansiDisableSGRMouse      = "\033[?1006l"
	ansiExitAlternateScreen  = "\033[?1049l"
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

func stopVMIfNeeded(finalModel tea.Model) {
	model, ok := finalModel.(app.Model)
	if !ok {
		return
	}

	if !model.IsCloudMode() {
		return
	}

	cfg := model.GetCloudConfig()
	if !cloud.IsVMRunning(cfg) {
		return
	}

	fmt.Println("\n☁️  Stopping Cloud VM... (this may take a moment)")
	if err := cloud.StopVMSync(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  Warning: failed to stop VM: %v\n", err)
	} else {
		fmt.Println("✓  Cloud VM stopped successfully.")
	}
}

func main() {
	defer handlePanic()

	appModel := app.New()
	program := tea.NewProgram(
		appModel,
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	finalModel, err := program.Run()

	// Reset terminal FIRST so messages are visible
	resetTerminal()

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}

	// Stop VM on exit if using cloud environment
	stopVMIfNeeded(finalModel)
}
