package executor

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunCommandOutputTrimmed(t *testing.T) {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "echo", "  hello  ")
	} else {
		cmd = exec.Command("sh", "-c", "printf '  hello  '")
	}

	got, err := runCommandOutputTrimmed(cmd)
	if err != nil {
		t.Fatalf("runCommandOutputTrimmed() error = %v", err)
	}
	if got != "hello" {
		t.Fatalf("runCommandOutputTrimmed() = %q, want %q", got, "hello")
	}
}

func TestCopyToClipboardUsesWrapper(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	msg, ok := CopyToClipboardCmd("text")().(ClipboardResultMsg)
	if !ok {
		t.Fatalf("expected ClipboardResultMsg, got %T", msg)
	}
	if msg.Error == nil {
		t.Fatal("expected clipboard command to fail without a clipboard utility")
	}
}

func TestOpenInFileBrowserRejectsMissingPath(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "missing")
	err := OpenInFileBrowser(missingPath)
	if err == nil {
		t.Fatal("expected missing path to fail")
	}
	if !strings.Contains(err.Error(), "path does not exist") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOpenInFileBrowserCmdUsesWrapper(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "missing")
	msg, ok := OpenInFileBrowserCmd(missingPath)().(FileBrowserResultMsg)
	if !ok {
		t.Fatalf("expected FileBrowserResultMsg, got %T", msg)
	}
	if msg.Error == nil {
		t.Fatal("expected file browser command to fail for missing path")
	}
}
