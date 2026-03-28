package globalsetup

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteFileAtomicallyReplacesContents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "overrides.json")
	if err := os.WriteFile(path, []byte(`{"old": true}`), 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}

	if err := writeFileAtomically(path, []byte(`{"new": true}`), 0o644); err != nil {
		t.Fatalf("writeFileAtomically: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read final file: %v", err)
	}
	if string(content) != `{"new": true}` {
		t.Fatalf("unexpected file content: %q", string(content))
	}

	matches, err := filepath.Glob(filepath.Join(dir, "overrides.json.tmp*"))
	if err != nil {
		t.Fatalf("glob temp files: %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no leftover temp files, got %v", matches)
	}
}
