package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/eeg-pipeline/tui/app"

	tea "github.com/charmbracelet/bubbletea"
)

// resetTerminal restores the terminal to a clean state
// This handles cleanup if the program exits abnormally
func resetTerminal() {
	// Disable mouse tracking modes
	fmt.Print("\033[?1000l") // Disable mouse click tracking
	fmt.Print("\033[?1002l") // Disable mouse button tracking
	fmt.Print("\033[?1003l") // Disable all mouse tracking
	fmt.Print("\033[?1006l") // Disable SGR mouse mode

	// Exit alternate screen buffer
	fmt.Print("\033[?1049l")

	// Reset terminal attributes using stty
	cmd := exec.Command("stty", "sane")
	cmd.Stdin = os.Stdin
	cmd.Run()
}

func main() {
	// Ensure terminal is reset on panic
	defer func() {
		if r := recover(); r != nil {
			resetTerminal()
			fmt.Fprintf(os.Stderr, "TUI crashed: %v\n", r)
			os.Exit(1)
		}
	}()

	p := tea.NewProgram(
		app.New(),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	_, err := p.Run()

	// Always reset terminal on exit (normal or error)
	resetTerminal()

	if err != nil {
		fmt.Printf("Error running TUI: %v\n", err)
		os.Exit(1)
	}
}
