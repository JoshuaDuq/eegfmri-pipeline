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
	if pythonPath := getVenvPython(venvPath); pythonPath != "" {
		return pythonPath
	}

	// 2. Check for standard .venv or venv in the repo root
	venvDirNames := []string{".venv", "venv"}
	for _, venvDirName := range venvDirNames {
		venvPath := filepath.Join(repoRoot, venvDirName)
		if pythonPath := getVenvPython(venvPath); pythonPath != "" {
			return pythonPath
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
	if !isDir(venvPath) {
		return ""
	}

	binDir := venvBinDir()
	pythonExecutables := []string{"python", "python3"}

	for _, executable := range pythonExecutables {
		executablePath := filepath.Join(venvPath, binDir, executable)
		executablePath = appendExeOnWindows(executablePath)
		if fileExists(executablePath) {
			return executablePath
		}
	}

	return ""
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func venvBinDir() string {
	if runtime.GOOS == "windows" {
		return "Scripts"
	}
	return "bin"
}

func appendExeOnWindows(path string) string {
	if runtime.GOOS == "windows" {
		return path + ".exe"
	}
	return path
}
