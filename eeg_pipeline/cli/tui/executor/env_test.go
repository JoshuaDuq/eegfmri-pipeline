package executor

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestGetPythonCommandPrefersFirstVenv(t *testing.T) {
	repoRoot := t.TempDir()

	venvPaths := []string{
		filepath.Join(repoRoot, "eeg_pipeline", ".venv311"),
		filepath.Join(repoRoot, ".venv311"),
		filepath.Join(repoRoot, ".venv"),
		filepath.Join(repoRoot, "venv"),
	}

	for i, venvPath := range venvPaths {
		binDir := "bin"
		if runtime.GOOS == "windows" {
			binDir = "Scripts"
		}
		pythonPath := filepath.Join(venvPath, binDir, "python")
		if runtime.GOOS == "windows" {
			pythonPath += ".exe"
		}
		if err := os.MkdirAll(filepath.Dir(pythonPath), 0o755); err != nil {
			t.Fatalf("mkdir %d: %v", i, err)
		}
		if err := os.WriteFile(pythonPath, []byte(""), 0o644); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}

	got := GetPythonCommand(repoRoot)
	want := filepath.Join(venvPaths[0], func() string {
		if runtime.GOOS == "windows" {
			return "Scripts/python.exe"
		}
		return "bin/python"
	}())
	if got != want {
		t.Fatalf("GetPythonCommand() = %q, want %q", got, want)
	}
}

func TestGetPythonCommandFallsBackWhenNoVenvExists(t *testing.T) {
	repoRoot := t.TempDir()

	got := GetPythonCommand(repoRoot)
	if runtime.GOOS == "windows" {
		if got != "python" {
			t.Fatalf("GetPythonCommand() = %q, want python", got)
		}
		return
	}
	if got != "python3" {
		t.Fatalf("GetPythonCommand() = %q, want python3", got)
	}
}

