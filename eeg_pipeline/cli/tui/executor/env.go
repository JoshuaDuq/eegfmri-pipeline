package executor

import (
	"os"
	"path/filepath"
	"runtime"
)

// GetPythonCommand returns the best python executable to use.
// It looks for virtual environments in the project root.
func GetPythonCommand(repoRoot string) string {
	// 1. Check for .venv311 (user's specific venv)
	venvPath := filepath.Join(repoRoot, ".venv311")
	if isDir(venvPath) {
		pyPath := getVenvPython(venvPath)
		if pyPath != "" {
			return pyPath
		}
	}

	// 2. Check for standard .venv or venv
	venvs := []string{".venv", "venv"}
	for _, v := range venvs {
		venvPath := filepath.Join(repoRoot, v)
		if isDir(venvPath) {
			pyPath := getVenvPython(venvPath)
			if pyPath != "" {
				return pyPath
			}
		}
	}

	// 3. Fallback to system python3 then python
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

func getVenvPython(venvPath string) string {
	binDir := "bin"
	if runtime.GOOS == "windows" {
		binDir = "Scripts"
	}

	pyPath := filepath.Join(venvPath, binDir, "python")
	if runtime.GOOS == "windows" {
		pyPath += ".exe"
	}

	if fileExists(pyPath) {
		return pyPath
	}

	// Try python3 if python doesn't exist
	py3Path := filepath.Join(venvPath, binDir, "python3")
	if runtime.GOOS == "windows" {
		py3Path += ".exe"
	}
	if fileExists(py3Path) {
		return py3Path
	}

	return ""
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
