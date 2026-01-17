package executor

import (
	"os"
	"path/filepath"
	"runtime"
)

// GetPythonCommand returns the best python executable to use.
// It looks for virtual environments in the project root.
func GetPythonCommand(repoRoot string) string {
	venvPaths := []string{
		filepath.Join(repoRoot, "eeg_pipeline", ".venv311"),
		filepath.Join(repoRoot, ".venv311"),
		filepath.Join(repoRoot, ".venv"),
		filepath.Join(repoRoot, "venv"),
	}

	for _, venvPath := range venvPaths {
		if pythonPath := findPythonInVenv(venvPath); pythonPath != "" {
			return pythonPath
		}
	}

	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func findPythonInVenv(venvPath string) string {
	info, err := os.Stat(venvPath)
	if err != nil || !info.IsDir() {
		return ""
	}

	binDir := "bin"
	if runtime.GOOS == "windows" {
		binDir = "Scripts"
	}

	executables := []string{"python", "python3"}
	for _, executable := range executables {
		executablePath := filepath.Join(venvPath, binDir, executable)
		if runtime.GOOS == "windows" {
			executablePath += ".exe"
		}

		if _, err := os.Stat(executablePath); err == nil {
			return executablePath
		}
	}

	return ""
}
