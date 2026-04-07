package executor

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

type PythonCommand struct {
	Executable string
	PrefixArgs []string
}

func (cmd PythonCommand) Args(args ...string) []string {
	out := make([]string, 0, len(cmd.PrefixArgs)+len(args))
	out = append(out, cmd.PrefixArgs...)
	out = append(out, args...)
	return out
}

func ResolvePythonCommand(repoRoot string) PythonCommand {
	return resolvePythonCommand(runtime.GOOS, repoRoot, exec.LookPath)
}

func resolvePythonCommand(
	goos string,
	repoRoot string,
	lookPath func(string) (string, error),
) PythonCommand {
	venvPaths := []string{
		filepath.Join(repoRoot, "eeg_pipeline", ".venv311"),
		filepath.Join(repoRoot, ".venv311"),
		filepath.Join(repoRoot, ".venv"),
		filepath.Join(repoRoot, "venv"),
	}

	for _, venvPath := range venvPaths {
		if pythonPath := findPythonInVenv(goos, venvPath); pythonPath != "" {
			return PythonCommand{Executable: pythonPath}
		}
	}

	if goos == "windows" {
		if pythonPath, err := lookPath("python"); err == nil {
			return PythonCommand{Executable: pythonPath}
		}
		if launcherPath, err := lookPath("py"); err == nil {
			return PythonCommand{
				Executable: launcherPath,
				PrefixArgs: []string{"-3"},
			}
		}
		return PythonCommand{Executable: "python"}
	}

	if pythonPath, err := lookPath("python3"); err == nil {
		return PythonCommand{Executable: pythonPath}
	}
	return PythonCommand{Executable: "python3"}
}

func findPythonInVenv(goos string, venvPath string) string {
	info, err := os.Stat(venvPath)
	if err != nil || !info.IsDir() {
		return ""
	}

	binDir := "bin"
	executables := []string{"python", "python3"}
	if goos == "windows" {
		binDir = "Scripts"
		executables = []string{"python.exe"}
	}

	for _, executable := range executables {
		executablePath := filepath.Join(venvPath, binDir, executable)
		if _, err := os.Stat(executablePath); err == nil {
			return executablePath
		}
	}

	return ""
}
