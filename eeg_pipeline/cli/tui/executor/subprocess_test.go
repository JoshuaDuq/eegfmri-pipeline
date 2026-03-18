package executor

import (
	"testing"

	"github.com/eeg-pipeline/tui/messages"

	tea "github.com/charmbracelet/bubbletea"
)

func TestFlattenConfigValues_FlattensNestedSections(t *testing.T) {
	values := map[string]interface{}{
		"time_frequency_analysis": map[string]interface{}{
			"bands": map[string]interface{}{
				"alpha": []interface{}{8.0, 12.0},
			},
		},
		"fmri_preprocessing": map[string]interface{}{
			"fmriprep": map[string]interface{}{
				"nthreads":     float64(8),
				"omp_nthreads": float64(4),
			},
		},
		"machine_learning.targets.regression": "value",
	}

	got := flattenConfigValues(values)

	if _, ok := got["time_frequency_analysis.bands"]; !ok {
		t.Fatalf("expected flattened bands key to be present")
	}
	if got["fmri_preprocessing.fmriprep.nthreads"] != float64(8) {
		t.Fatalf("expected nthreads=8, got %v", got["fmri_preprocessing.fmriprep.nthreads"])
	}
	if got["fmri_preprocessing.fmriprep.omp_nthreads"] != float64(4) {
		t.Fatalf("expected omp_nthreads=4, got %v", got["fmri_preprocessing.fmriprep.omp_nthreads"])
	}
	if got["machine_learning.targets.regression"] != "value" {
		t.Fatalf("expected dotted key to be preserved, got %v", got["machine_learning.targets.regression"])
	}
}

func TestFlattenConfigValues_HandlesScalarsAndEmptyKeys(t *testing.T) {
	values := map[string]interface{}{
		"":        "ignored",
		"project": "rest",
		"paths": map[string]interface{}{
			"bids_root": "/data/bids",
		},
	}

	got := flattenConfigValues(values)

	if _, ok := got[""]; ok {
		t.Fatal("expected empty key to be ignored")
	}
	if got["project"] != "rest" {
		t.Fatalf("expected scalar project value, got %v", got["project"])
	}
	if got["paths.bids_root"] != "/data/bids" {
		t.Fatalf("expected nested path flattening, got %v", got["paths.bids_root"])
	}
}

func TestProgressStreamerEmptyCommandEmitsDone(t *testing.T) {
	ps := NewProgressStreamer(t.TempDir(), "")
	cmd := ps.Start()
	if msg := cmd(); msg != nil {
		t.Fatalf("expected nil immediate msg, got %T", msg)
	}

	eventCmd := ps.WaitForEvent()
	event := eventCmd()
	done, ok := event.(messages.CommandDoneMsg)
	if !ok {
		t.Fatalf("expected CommandDoneMsg, got %T", event)
	}
	if done.ExitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", done.ExitCode)
	}
}

func TestProgressStreamerWaitForEventWhenClosed(t *testing.T) {
	ps := &ProgressStreamer{Events: make(chan tea.Msg)}
	close(ps.Events)

	msg := ps.WaitForEvent()()
	if msg != nil {
		t.Fatalf("expected nil message when channel closed, got %T", msg)
	}
}
