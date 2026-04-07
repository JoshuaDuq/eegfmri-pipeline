package executor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolvePythonCommandPrefersFirstVenv(t *testing.T) {
	repoRoot := t.TempDir()

	venvPaths := []string{
		filepath.Join(repoRoot, "eeg_pipeline", ".venv311"),
		filepath.Join(repoRoot, ".venv311"),
		filepath.Join(repoRoot, ".venv"),
		filepath.Join(repoRoot, "venv"),
	}

	for i, venvPath := range venvPaths {
		pythonPath := filepath.Join(venvPath, "Scripts", "python.exe")
		if err := os.MkdirAll(filepath.Dir(pythonPath), 0o755); err != nil {
			t.Fatalf("mkdir %d: %v", i, err)
		}
		if err := os.WriteFile(pythonPath, []byte(""), 0o644); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}

	got := resolvePythonCommand("windows", repoRoot, func(name string) (string, error) {
		return "", os.ErrNotExist
	})
	want := filepath.Join(venvPaths[0], "Scripts", "python.exe")
	if got.Executable != want {
		t.Fatalf("resolvePythonCommand().Executable = %q, want %q", got.Executable, want)
	}
	if len(got.PrefixArgs) != 0 {
		t.Fatalf("resolvePythonCommand().PrefixArgs = %#v, want empty", got.PrefixArgs)
	}
}

func TestResolvePythonCommandWindowsFallsBackToPythonThenPyLauncher(t *testing.T) {
	repoRoot := t.TempDir()

	got := resolvePythonCommand("windows", repoRoot, func(name string) (string, error) {
		if name == "python" {
			return "C:\\Python311\\python.exe", nil
		}
		return "", os.ErrNotExist
	})
	if got.Executable != "C:\\Python311\\python.exe" {
		t.Fatalf("resolvePythonCommand().Executable = %q, want python.exe", got.Executable)
	}
	if len(got.PrefixArgs) != 0 {
		t.Fatalf("resolvePythonCommand().PrefixArgs = %#v, want empty", got.PrefixArgs)
	}

	got = resolvePythonCommand("windows", repoRoot, func(name string) (string, error) {
		if name == "py" {
			return "C:\\Windows\\py.exe", nil
		}
		return "", os.ErrNotExist
	})
	if got.Executable != "C:\\Windows\\py.exe" {
		t.Fatalf("resolvePythonCommand().Executable = %q, want py launcher", got.Executable)
	}
	if len(got.PrefixArgs) != 1 || got.PrefixArgs[0] != "-3" {
		t.Fatalf("resolvePythonCommand().PrefixArgs = %#v, want [-3]", got.PrefixArgs)
	}
}

func TestResolvePythonCommandUnixFallsBackToPython3(t *testing.T) {
	repoRoot := t.TempDir()

	got := resolvePythonCommand("darwin", repoRoot, func(name string) (string, error) {
		if name == "python3" {
			return "/usr/bin/python3", nil
		}
		return "", os.ErrNotExist
	})
	if got.Executable != "/usr/bin/python3" {
		t.Fatalf("resolvePythonCommand().Executable = %q, want /usr/bin/python3", got.Executable)
	}
	if len(got.PrefixArgs) != 0 {
		t.Fatalf("resolvePythonCommand().PrefixArgs = %#v, want empty", got.PrefixArgs)
	}
}
