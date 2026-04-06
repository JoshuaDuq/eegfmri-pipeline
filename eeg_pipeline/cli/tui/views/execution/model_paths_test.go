package execution

import (
	"path/filepath"
	"testing"
)

func TestExpandParsedPathWithHomeSupportsWindowsAndUnixPrefixes(t *testing.T) {
	home := filepath.Join(string(filepath.Separator), "Users", "tester")

	cases := map[string]string{
		"~":             home,
		"~/results":     filepath.Join(home, "results"),
		"~\\results":    filepath.Join(home, "results"),
		"relative/path": "relative/path",
	}

	for input, want := range cases {
		if got := expandParsedPathWithHome(input, home); got != want {
			t.Fatalf("expandParsedPathWithHome(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestExtractDerivRootExpandsWindowsStyleHomePath(t *testing.T) {
	repoRoot := filepath.Join("workspace", "repo")
	home := filepath.Join(string(filepath.Separator), "Users", "tester")
	model := Model{
		RepoRoot: repoRoot,
		Command:  `eeg_pipeline features --deriv-root "~\derivatives\rest"`,
	}

	got := expandParsedPathWithHome(`~\derivatives\rest`, home)
	want := filepath.Join(home, "derivatives", "rest")
	if got != want {
		t.Fatalf("expected helper expansion to preserve Windows home path, got %q want %q", got, want)
	}

	if extracted := model.extractDerivRootFromCommand(home); extracted != want {
		t.Fatalf("extractDerivRootFromCommand() = %q, want %q", extracted, want)
	}
}

func TestExtractBidsFmriRootExpandsWindowsStyleHomePath(t *testing.T) {
	repoRoot := filepath.Join("workspace", "repo")
	home := filepath.Join(string(filepath.Separator), "Users", "tester")
	model := Model{
		RepoRoot: repoRoot,
		Command:  `eeg_pipeline fmri preprocess --bids-fmri-root "~\bids\fmri"`,
	}

	want := filepath.Join(home, "bids", "fmri")
	if extracted := model.extractBidsFmriRootFromCommand(home); extracted != want {
		t.Fatalf("extractBidsFmriRootFromCommand() = %q, want %q", extracted, want)
	}
}

func TestExtractDerivRootPrefersExplicitArgsForQuotedWindowsPaths(t *testing.T) {
	repoRoot := filepath.Join("workspace", "repo")
	home := filepath.Join(string(filepath.Separator), "Users", "O'Brien")
	model := Model{
		RepoRoot:    repoRoot,
		Command:     `eeg_pipeline features --deriv-root '~/Bob''s Results'`,
		CommandArgs: []string{"eeg-pipeline", "features", "--deriv-root", `~\Bob's Results`},
	}

	want := filepath.Join(home, "Bob's Results")
	if extracted := model.extractDerivRootFromCommand(home); extracted != want {
		t.Fatalf("extractDerivRootFromCommand() = %q, want %q", extracted, want)
	}
}
