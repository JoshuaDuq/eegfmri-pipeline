package executor

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
)

func TestJSONLoadersAndProgressStreamer(t *testing.T) {
	repoRoot := writeFakePythonBinary(t)

	t.Run("load config summary", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "config_summary")

		msg := LoadConfigSummary(repoRoot)()
		got, ok := msg.(messages.ConfigLoadedMsg)
		if !ok {
			t.Fatalf("expected ConfigLoadedMsg, got %T", msg)
		}
		if got.Error != nil {
			t.Fatalf("expected no error, got %v", got.Error)
		}
		if got.Summary.Task != "rest" || got.Summary.PreprocessingNJobs != 8 {
			t.Fatalf("unexpected config summary: %#v", got.Summary)
		}
		if got.Summary.BidsRestRoot != "/bids-rest" || got.Summary.SourceRoot != "/source" {
			t.Fatalf("unexpected config summary paths: %#v", got.Summary)
		}
	})

	t.Run("load config keys", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "config_keys")

		msg := LoadConfigKeys(repoRoot, []string{"project.task", "paths.bids_root"})()
		got, ok := msg.(messages.ConfigKeysLoadedMsg)
		if !ok {
			t.Fatalf("expected ConfigKeysLoadedMsg, got %T", msg)
		}
		if got.Error != nil {
			t.Fatalf("expected no error, got %v", got.Error)
		}
		if got.Values["project.task"] != "rest" {
			t.Fatalf("expected flattened task key, got %#v", got.Values["project.task"])
		}
		if got.Values["project.random_state"] != float64(7) {
			t.Fatalf("expected flattened random_state key, got %#v", got.Values["project.random_state"])
		}
		if got.Values["paths.freesurfer_license"] != "license.txt" {
			t.Fatalf("expected flattened license key, got %#v", got.Values["paths.freesurfer_license"])
		}
	})

	t.Run("load subjects and refresh", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "subjects_populated")

		msg := LoadSubjects(repoRoot, "task", types.PipelineFeatures)()
		got, ok := msg.(messages.SubjectsLoadedMsg)
		if !ok {
			t.Fatalf("expected SubjectsLoadedMsg, got %T", msg)
		}
		if got.Error != nil {
			t.Fatalf("expected no error, got %v", got.Error)
		}
		if len(got.Subjects) != 1 || got.Subjects[0].ID != "sub-01" {
			t.Fatalf("unexpected subjects payload: %#v", got.Subjects)
		}
		if len(got.AvailableWindowsByFeature["power"]) != 1 {
			t.Fatalf("expected windows by feature to hydrate, got %#v", got.AvailableWindowsByFeature)
		}

		msg = LoadSubjectsRefresh(repoRoot, "task", types.PipelineFeatures)()
		updated, ok := msg.(messages.SubjectsLoadedMsg)
		if !ok {
			t.Fatalf("expected SubjectsLoadedMsg from refresh, got %T", msg)
		}
		if updated.Error != nil {
			t.Fatalf("expected no refresh error, got %v", updated.Error)
		}
	})

	t.Run("discover columns", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "columns_trial_table")

		msg := DiscoverColumns(repoRoot, "task")()
		got, ok := msg.(messages.ColumnsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected ColumnsDiscoveredMsg, got %T", msg)
		}
		if got.Error != nil {
			t.Fatalf("expected no error, got %v", got.Error)
		}
		if got.Source != "trial_table" {
			t.Fatalf("expected source trial_table, got %q", got.Source)
		}
		if len(got.Columns) != 2 || got.Columns[0] != "trial_type" {
			t.Fatalf("unexpected columns payload: %#v", got.Columns)
		}

		msg = DiscoverTrialTableColumns(repoRoot, "task")()
		trialTable, ok := msg.(messages.ColumnsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected ColumnsDiscoveredMsg, got %T", msg)
		}
		if trialTable.Source != "trial_table" {
			t.Fatalf("expected trial_table source, got %q", trialTable.Source)
		}
	})

	t.Run("discover condition effects", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "columns_condition_effects")

		msg := DiscoverConditionEffectsColumns(repoRoot, "task", "sub-01")()
		got, ok := msg.(messages.ColumnsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected ColumnsDiscoveredMsg, got %T", msg)
		}
		if got.Source != "condition_effects" {
			t.Fatalf("expected condition_effects source, got %q", got.Source)
		}
	})

	t.Run("discover fmri conditions and stats", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "fmri_conditions_error")

		msg := DiscoverFmriConditions(repoRoot, "sub-01", "task", "trial_type")()
		got, ok := msg.(messages.FmriConditionsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected FmriConditionsDiscoveredMsg, got %T", msg)
		}
		if got.Error == nil || got.Error.Error() != "invalid fmri conditions" {
			t.Fatalf("unexpected fmri conditions error: %v", got.Error)
		}

		t.Setenv("FAKE_EXECUTOR_MODE", "multigroup_ok")
		msg = DiscoverMultigroupStats(repoRoot, "sub-01")()
		stats, ok := msg.(messages.MultigroupStatsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected MultigroupStatsDiscoveredMsg, got %T", msg)
		}
		if !stats.Available || len(stats.Groups) != 2 {
			t.Fatalf("unexpected multigroup stats payload: %#v", stats)
		}
	})

	t.Run("discover rois and plotters", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "rois_ok")

		msg := DiscoverROIs(repoRoot, "task")()
		rois, ok := msg.(messages.ROIsDiscoveredMsg)
		if !ok {
			t.Fatalf("expected ROIsDiscoveredMsg, got %T", msg)
		}
		if rois.Error != nil || len(rois.ROIs) != 2 {
			t.Fatalf("unexpected rois payload: %#v", rois)
		}

		t.Setenv("FAKE_EXECUTOR_MODE", "plotters")
		msg = LoadPlotters(repoRoot)()
		plotters, ok := msg.(messages.PlottersLoadedMsg)
		if !ok {
			t.Fatalf("expected PlottersLoadedMsg, got %T", msg)
		}
		if plotters.Error != nil {
			t.Fatalf("expected no plotter error, got %v", plotters.Error)
		}
		if len(plotters.FeaturePlotters["power"]) != 1 {
			t.Fatalf("unexpected plotter payload: %#v", plotters.FeaturePlotters)
		}
	})

	t.Run("run python stderr", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "stderr_error")

		if _, err := runPythonJSONCommand(repoRoot, []string{"-m", "eeg_pipeline", "info", "config"}); err == nil {
			t.Fatal("expected runPythonJSONCommand to fail")
		} else if !strings.Contains(err.Error(), "python failed") {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("progress streamer", func(t *testing.T) {
		t.Setenv("FAKE_EXECUTOR_MODE", "progress")

		ps := NewProgressStreamer(repoRoot, "eeg-pipeline analyze")
		if msg := ps.Start()(); msg != nil {
			t.Fatalf("expected nil from Start cmd, got %T", msg)
		}

		var received []tea.Msg
		for {
			msg := ps.WaitForEvent()()
			if msg == nil {
				break
			}
			received = append(received, msg)
		}

		if len(received) != 8 {
			t.Fatalf("expected 8 events, got %d: %#v", len(received), received)
		}
		if started, ok := received[0].(messages.CommandStartedMsg); !ok || started.Operation != "eeg-pipeline analyze" {
			t.Fatalf("unexpected first event: %#v", received[0])
		}
		if second, ok := received[1].(messages.CommandStartedMsg); !ok || second.TotalSubjects != 1 {
			t.Fatalf("unexpected start event: %#v", received[1])
		}
		if _, ok := received[2].(messages.SubjectStartedMsg); !ok {
			t.Fatalf("expected subject started event, got %#v", received[2])
		}
		if _, ok := received[3].(messages.StepProgressMsg); !ok {
			t.Fatalf("expected progress event, got %#v", received[3])
		}
		if _, ok := received[4].(messages.SubjectDoneMsg); !ok {
			t.Fatalf("expected subject done event, got %#v", received[4])
		}
		if logMsg, ok := received[5].(messages.LogMsg); !ok || logMsg.Message != "done" {
			t.Fatalf("unexpected log event: %#v", received[5])
		}
		if logMsg, ok := received[6].(messages.LogMsg); !ok || logMsg.Message != "plain log line" {
			t.Fatalf("unexpected log event: %#v", received[6])
		}
		if done, ok := received[7].(messages.CommandDoneMsg); !ok || !done.Success {
			t.Fatalf("unexpected command done event: %#v", received[7])
		}
	})
}

func writeFakePythonBinary(t *testing.T) string {
	t.Helper()

	repoRoot := t.TempDir()
	pythonPath := fakePythonPath(repoRoot)
	if err := os.MkdirAll(filepath.Dir(pythonPath), 0o755); err != nil {
		t.Fatalf("mkdir fake python: %v", err)
	}

	buildDir := t.TempDir()
	sourcePath := filepath.Join(buildDir, "main.go")
	if err := os.WriteFile(sourcePath, []byte(fakePythonProgram), 0o644); err != nil {
		t.Fatalf("write fake python source: %v", err)
	}

	cmd := exec.Command("go", "build", "-o", pythonPath, sourcePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build fake python: %v\n%s", err, output)
	}

	return repoRoot
}

func fakePythonPath(repoRoot string) string {
	binDir := "bin"
	executable := "python"
	if runtime.GOOS == "windows" {
		binDir = "Scripts"
		executable = "python.exe"
	}

	return filepath.Join(repoRoot, "eeg_pipeline", ".venv311", binDir, executable)
}

const fakePythonProgram = `package main

import (
	"fmt"
	"os"
)

func printLine(line string) {
	fmt.Fprintln(os.Stdout, line)
}

func main() {
	switch os.Getenv("FAKE_EXECUTOR_MODE") {
	case "config_summary":
		printLine("{\"task\":\"rest\",\"bids_root\":\"/bids\",\"bids_rest_root\":\"/bids-rest\",\"bids_fmri_root\":\"/fmri\",\"deriv_root\":\"/deriv\",\"deriv_rest_root\":\"/deriv-rest\",\"source_root\":\"/source\",\"preprocessing_n_jobs\":8}")
	case "config_keys":
		printLine("{\"project\":{\"task\":\"rest\",\"random_state\":7,\"subject_list\":[\"sub-01\",\"sub-02\"]},\"paths\":{\"bids_root\":\"/bids\",\"freesurfer_license\":\"license.txt\"}}")
	case "subjects_populated":
		printLine("{\"subjects\":[{\"id\":\"sub-01\",\"has_source_data\":true,\"has_bids\":true,\"has_derivatives\":false,\"has_epochs\":false,\"has_features\":false,\"has_stats\":false}],\"available_windows\":[\"window-a\"],\"available_windows_by_feature\":{\"power\":[\"window-a\"]},\"available_event_columns\":[\"trial_type\"],\"available_channels\":[\"Cz\"],\"unavailable_channels\":[\"Pz\"]}")
	case "columns_trial_table":
		printLine("{\"columns\":[\"trial_type\",\"condition\"],\"values\":{\"trial_type\":[\"a\",\"b\"]},\"windows\":[\"win-a\"],\"source\":\"trial_table\"}")
	case "columns_condition_effects":
		printLine("{\"columns\":[\"effect\"],\"values\":{\"effect\":[\"x\"]},\"windows\":[\"win-b\"],\"source\":\"condition_effects\"}")
	case "fmri_conditions_error":
		printLine("{\"error\":\"invalid fmri conditions\"}")
	case "multigroup_ok":
		printLine("{\"available\":true,\"groups\":[\"control\",\"treated\"],\"n_features\":12,\"n_significant\":3,\"file\":\"stats.json\"}")
	case "rois_ok":
		printLine("{\"rois\":[\"roi-1\",\"roi-2\"]}")
	case "plotters":
		printLine("{\"feature_plotters\":{\"power\":[{\"id\":\"power.topo\",\"category\":\"power\",\"name\":\"Power Topography\"}]}}")
	case "progress":
		printLine("{\"event\":\"start\",\"operation\":\"analyze\",\"subjects\":[\"sub-01\"],\"total_subjects\":1}")
		printLine("{\"event\":\"subject_start\",\"subject\":\"sub-01\"}")
		printLine("{\"event\":\"progress\",\"subject\":\"sub-01\",\"step\":\"prep\",\"current\":1,\"total\":2,\"pct\":50}")
		printLine("{\"event\":\"subject_done\",\"subject\":\"sub-01\",\"success\":true}")
		printLine("{\"event\":\"log\",\"level\":\"info\",\"message\":\"done\",\"subject\":\"sub-01\"}")
		printLine("plain log line")
	case "stderr_error":
		fmt.Fprintln(os.Stderr, "python failed")
		os.Exit(1)
	default:
		printLine("{}")
	}
}
`
