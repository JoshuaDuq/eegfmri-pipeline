package executor

import "testing"

func TestJoinCommandUsesPowerShellQuotingOnWindows(t *testing.T) {
	args := []string{
		"eeg-pipeline",
		"features",
		"--deriv-root",
		`C:\Users\Test User\derivatives`,
		"--set",
		`project.note=Bob's dataset`,
	}

	got := JoinCommand("windows", args)
	want := "eeg-pipeline features --deriv-root 'C:\\Users\\Test User\\derivatives' --set 'project.note=Bob''s dataset'"
	if got != want {
		t.Fatalf("JoinCommand(windows) = %q, want %q", got, want)
	}
}
