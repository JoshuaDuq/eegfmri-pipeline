package executor

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
)

func runCommandOutputTrimmed(cmd *exec.Cmd) (string, error) {
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

// CopyToClipboard copies text to the system clipboard in a cross-platform way
func CopyToClipboard(text string) error {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("pbcopy")
	case "linux":
		// Try xclip first, fall back to xsel
		if _, err := exec.LookPath("xclip"); err == nil {
			cmd = exec.Command("xclip", "-selection", "clipboard")
		} else if _, err := exec.LookPath("xsel"); err == nil {
			cmd = exec.Command("xsel", "--clipboard", "--input")
		} else {
			return fmt.Errorf("no clipboard utility found (install xclip or xsel)")
		}
	case "windows":
		cmd = exec.Command("cmd", "/c", "clip")
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	cmd.Stdin = strings.NewReader(text)
	return cmd.Run()
}

// CopyToClipboardCmd returns a tea.Cmd that copies text to clipboard
func CopyToClipboardCmd(text string) tea.Cmd {
	return func() tea.Msg {
		err := CopyToClipboard(text)
		return ClipboardResultMsg{Error: err}
	}
}

// ClipboardResultMsg is sent after a clipboard operation
type ClipboardResultMsg struct {
	Error error
}

// OpenInFileBrowser opens a path in the system file browser
func OpenInFileBrowser(path string) error {
	// Verify path exists before trying to open
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("path does not exist: %s: %w", path, err)
	}

	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "linux":
		cmd = exec.Command("xdg-open", path)
	case "windows":
		cmd = exec.Command("explorer", path)
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	// Start the command and detach it (don't wait for completion)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start file browser: %w", err)
	}

	// Detach the process so it doesn't become a zombie
	go func() {
		_ = cmd.Wait()
	}()

	return nil
}

// OpenInFileBrowserCmd returns a tea.Cmd that opens a path in file browser
func OpenInFileBrowserCmd(path string) tea.Cmd {
	return func() tea.Msg {
		err := OpenInFileBrowser(path)
		return FileBrowserResultMsg{Error: err}
	}
}

// FileBrowserResultMsg is sent after a file browser operation
type FileBrowserResultMsg struct {
	Error error
}

// PickFolderMsg is the message type for folder picker results
type PickFolderMsg struct {
	Path  string
	Error error
	Field string
}

// PickFolder opens a folder picker dialog and returns the selected path
func PickFolder(prompt string, field string) tea.Cmd {
	return func() tea.Msg {
		var path string
		var err error

		switch runtime.GOOS {
		case "darwin":
			cmd := exec.Command("osascript", "-e",
				fmt.Sprintf(`POSIX path of (choose folder with prompt "%s")`, prompt))
			path, err = runCommandOutputTrimmed(cmd)

		case "linux":
			// Try zenity first, then kdialog
			if _, lookErr := exec.LookPath("zenity"); lookErr == nil {
				cmd := exec.Command("zenity", "--file-selection", "--directory", "--title", prompt)
				path, err = runCommandOutputTrimmed(cmd)
			} else if _, lookErr := exec.LookPath("kdialog"); lookErr == nil {
				cmd := exec.Command("kdialog", "--getexistingdirectory", ".", "--title", prompt)
				path, err = runCommandOutputTrimmed(cmd)
			} else {
				// Fallback: prompt user to type path manually
				err = fmt.Errorf("no folder picker available (install zenity or kdialog), please type path manually")
			}

		case "windows":
			// PowerShell folder browser dialog
			psScript := fmt.Sprintf(`
Add-Type -AssemblyName System.Windows.Forms
$browser = New-Object System.Windows.Forms.FolderBrowserDialog
$browser.Description = "%s"
$result = $browser.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $browser.SelectedPath
}
`, prompt)
			cmd := exec.Command("powershell", "-Command", psScript)
			path, err = runCommandOutputTrimmed(cmd)

		default:
			err = fmt.Errorf("unsupported platform: %s", runtime.GOOS)
		}

		return PickFolderMsg{Path: path, Error: err, Field: field}
	}
}

// GetPlatform returns the current platform name
func GetPlatform() string {
	return runtime.GOOS
}

// IsMac returns true if running on macOS
func IsMac() bool {
	return GetPlatform() == "darwin"
}

// IsLinux returns true if running on Linux
func IsLinux() bool {
	return GetPlatform() == "linux"
}

// IsWindows returns true if running on Windows
func IsWindows() bool {
	return GetPlatform() == "windows"
}

// GetHomeDir returns the user's home directory
func GetHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return home
}
