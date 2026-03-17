package executor

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/types"
)

// ProgressEvent from CLI --progress-json
type ProgressEvent struct {
	Event         string   `json:"event"`
	Operation     string   `json:"operation,omitempty"`
	Subjects      []string `json:"subjects,omitempty"`
	TotalSubjects int      `json:"total_subjects,omitempty"`
	Subject       string   `json:"subject,omitempty"`
	Step          string   `json:"step,omitempty"`
	Current       int      `json:"current,omitempty"`
	Total         int      `json:"total,omitempty"`
	Pct           int      `json:"pct,omitempty"`
	Level         string   `json:"level,omitempty"`
	Message       string   `json:"message,omitempty"`
	Success       bool     `json:"success,omitempty"`
	Duration      float64  `json:"duration,omitempty"`
	Code          string   `json:"code,omitempty"`
}

// SubjectsResponse from eeg-pipeline info subjects --json
type SubjectsResponse struct {
	Subjects                  []messages.SubjectInfo `json:"subjects"`
	Count                     int                    `json:"count"`
	AvailableWindows          []string               `json:"available_windows"`
	AvailableWindowsByFeature map[string][]string    `json:"available_windows_by_feature"`
	AvailableEventColumns     []string               `json:"available_event_columns"`
	AvailableChannels         []string               `json:"available_channels"`
	UnavailableChannels       []string               `json:"unavailable_channels"`
}

// ConfigSummaryResponse from eeg-pipeline info config --json
type ConfigSummaryResponse struct {
	Task               string `json:"task"`
	BidsRoot           string `json:"bids_root"`
	BidsRestRoot       string `json:"bids_rest_root"`
	BidsFmriRoot       string `json:"bids_fmri_root"`
	DerivRoot          string `json:"deriv_root"`
	DerivRestRoot      string `json:"deriv_rest_root"`
	SourceRoot         string `json:"source_root"`
	PreprocessingNJobs int    `json:"preprocessing_n_jobs"`
}

type PlottersResponse struct {
	FeaturePlotters map[string][]messages.PlotterInfo `json:"feature_plotters"`
}

// DiscoverColumnsResponse from eeg-pipeline discover --json
type DiscoverColumnsResponse struct {
	Columns []string            `json:"columns"`
	Values  map[string][]string `json:"values"`
	Windows []string            `json:"windows,omitempty"`
	Source  string              `json:"source"`
	File    string              `json:"file,omitempty"`
}

// DiscoverFmriConditionsResponse from eeg-pipeline info fmri-conditions --json
type DiscoverFmriConditionsResponse struct {
	Conditions      []string `json:"conditions"`
	ConditionColumn string   `json:"condition_column,omitempty"`
	Subject         string   `json:"subject"`
	Task            string   `json:"task"`
	Error           string   `json:"error,omitempty"`
}

// DiscoverROIsResponse from eeg-pipeline info rois --json
type DiscoverROIsResponse struct {
	ROIs    []string `json:"rois"`
	Subject string   `json:"subject"`
	Source  string   `json:"source,omitempty"`
	Error   string   `json:"error,omitempty"`
}

// DiscoverMultigroupStatsResponse from eeg-pipeline info multigroup-stats --json
type DiscoverMultigroupStatsResponse struct {
	Available    bool     `json:"available"`
	Groups       []string `json:"groups"`
	NFeatures    int      `json:"n_features"`
	NSignificant int      `json:"n_significant"`
	File         string   `json:"file,omitempty"`
	Subject      string   `json:"subject,omitempty"`
	Error        string   `json:"error,omitempty"`
}

// DiscoverFmriConditions runs a Python command to discover available fMRI condition values.
func DiscoverFmriConditions(repoRoot string, subject string, task string, conditionColumn string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline.cli.main", "info", "fmri-conditions", "--json"}
		if subject != "" {
			args = append(args, "--subject", subject)
		}
		if task != "" {
			args = append(args, "--task", task)
		}
		if strings.TrimSpace(conditionColumn) != "" {
			args = append(args, "--condition-column", strings.TrimSpace(conditionColumn))
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.FmriConditionsDiscoveredMsg{
				Conditions: nil,
				Subject:    subject,
				Task:       task,
				Error:      err,
			}
		}

		var response DiscoverFmriConditionsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.FmriConditionsDiscoveredMsg{
				Conditions: nil,
				Subject:    subject,
				Task:       task,
				Error:      err,
			}
		}

		if response.Error != "" {
			return messages.FmriConditionsDiscoveredMsg{
				Conditions: nil,
				Subject:    subject,
				Task:       task,
				Error:      fmt.Errorf("%s", response.Error),
			}
		}

		return messages.FmriConditionsDiscoveredMsg{
			Conditions: response.Conditions,
			Subject:    response.Subject,
			Task:       response.Task,
			Error:      nil,
		}
	}
}

// DiscoverFmriColumns runs eeg-pipeline info fmri-columns --json to find fMRI event columns
func DiscoverFmriColumns(repoRoot string, task string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "fmri-columns", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.FmriColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		var response DiscoverColumnsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.FmriColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		return messages.FmriColumnsDiscoveredMsg{
			Columns: response.Columns,
			Values:  response.Values,
			Source:  response.Source,
			Error:   nil,
		}
	}
}

// DiscoverColumns runs eeg-pipeline info discover --json to find available columns and their values
func DiscoverColumns(repoRoot string, task string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "discover", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		var response DiscoverColumnsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		return messages.ColumnsDiscoveredMsg{
			Columns: response.Columns,
			Values:  response.Values,
			Windows: response.Windows,
			Source:  response.Source,
			Error:   nil,
		}
	}
}

// DiscoverTrialTableColumns runs eeg-pipeline info discover --discover-source trial-table --json
// to find available trial-table columns (including feature columns when present).
func DiscoverTrialTableColumns(repoRoot string, task string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "discover", "--discover-source", "trial-table", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Windows: nil,
				Source:  "trial_table",
				Error:   err,
			}
		}

		var response DiscoverColumnsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Windows: nil,
				Source:  "trial_table",
				Error:   err,
			}
		}

		return messages.ColumnsDiscoveredMsg{
			Columns: response.Columns,
			Values:  response.Values,
			Windows: response.Windows,
			Source:  response.Source,
			Error:   nil,
		}
	}
}

// DiscoverConditionEffectsColumns runs eeg-pipeline info discover --discover-source condition-effects --json
// to find available condition columns and values from condition effects files
func DiscoverConditionEffectsColumns(repoRoot string, task string, subject string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "discover", "--discover-source", "condition-effects", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}
		if subject != "" {
			args = append(args, "--subject", subject)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		var response DiscoverColumnsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.ColumnsDiscoveredMsg{
				Columns: nil,
				Values:  nil,
				Error:   err,
			}
		}

		return messages.ColumnsDiscoveredMsg{
			Columns: response.Columns,
			Values:  response.Values,
			Windows: response.Windows,
			Source:  response.Source,
			Error:   nil,
		}
	}
}

// DiscoverMultigroupStats runs eeg-pipeline info multigroup-stats --json to find available multigroup comparisons
func DiscoverMultigroupStats(repoRoot string, subject string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "multigroup-stats", "--json"}
		if subject != "" {
			args = append(args, "--subject", subject)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.MultigroupStatsDiscoveredMsg{
				Available: false,
				Groups:    nil,
				Error:     err,
			}
		}

		var response DiscoverMultigroupStatsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.MultigroupStatsDiscoveredMsg{
				Available: false,
				Groups:    nil,
				Error:     err,
			}
		}

		if response.Error != "" {
			return messages.MultigroupStatsDiscoveredMsg{
				Available: false,
				Groups:    nil,
				Error:     fmt.Errorf("%s", response.Error),
			}
		}

		return messages.MultigroupStatsDiscoveredMsg{
			Available:    response.Available,
			Groups:       response.Groups,
			NFeatures:    response.NFeatures,
			NSignificant: response.NSignificant,
			File:         response.File,
			Error:        nil,
		}
	}
}

// DiscoverROIs runs eeg-pipeline info rois --json to find available ROIs from feature data
func DiscoverROIs(repoRoot string, task string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "rois", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ROIsDiscoveredMsg{
				ROIs:  nil,
				Error: err,
			}
		}

		var response DiscoverROIsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.ROIsDiscoveredMsg{
				ROIs:  nil,
				Error: err,
			}
		}

		if response.Error != "" {
			return messages.ROIsDiscoveredMsg{
				ROIs:  nil,
				Error: fmt.Errorf("%s", response.Error),
			}
		}

		return messages.ROIsDiscoveredMsg{
			ROIs:  response.ROIs,
			Error: nil,
		}
	}
}

// LoadSubjects runs eeg-pipeline info subjects --json and returns subjects.
// Uses --source all so the TUI always shows every discovered subject (BIDS, epochs, features, source_data);
// pipeline validation still determines which subjects are runnable for the chosen stage.
func LoadSubjects(repoRoot string, task string, pipeline types.Pipeline) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "subjects", "--status", "--json", "--cache"}
		if task != "" {
			args = append(args, "--task", task)
		}
		args = append(args, "--source", "all")

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			// Return actual error instead of silent fallback
			return messages.SubjectsLoadedMsg{
				Subjects: nil,
				Error:    err,
			}
		}

		var response SubjectsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.SubjectsLoadedMsg{
				Subjects: nil,
				Error:    err,
			}
		}

		if len(response.Subjects) == 0 {
			return messages.SubjectsLoadedMsg{
				Subjects: []messages.SubjectInfo{},
				Error:    nil,
			}
		}

		return messages.SubjectsLoadedMsg{
			Subjects:                  response.Subjects,
			AvailableWindows:          response.AvailableWindows,
			AvailableWindowsByFeature: response.AvailableWindowsByFeature,
			AvailableEventColumns:     response.AvailableEventColumns,
			AvailableChannels:         response.AvailableChannels,
			UnavailableChannels:       response.UnavailableChannels,
		}
	}
}

// LoadSubjectsRefresh forces refresh of cached subject discovery/status.
func LoadSubjectsRefresh(repoRoot string, task string, pipeline types.Pipeline) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "subjects", "--status", "--json", "--cache", "--refresh"}
		if task != "" {
			args = append(args, "--task", task)
		}
		args = append(args, "--source", "all")

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.SubjectsLoadedMsg{
				Subjects: nil,
				Error:    err,
			}
		}

		var response SubjectsResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.SubjectsLoadedMsg{
				Subjects: nil,
				Error:    err,
			}
		}

		return messages.SubjectsLoadedMsg{
			Subjects:                  response.Subjects,
			AvailableWindows:          response.AvailableWindows,
			AvailableWindowsByFeature: response.AvailableWindowsByFeature,
			AvailableEventColumns:     response.AvailableEventColumns,
			AvailableChannels:         response.AvailableChannels,
			UnavailableChannels:       response.UnavailableChannels,
		}
	}
}

func LoadPlotters(repoRoot string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "plotters", "--json"}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.PlottersLoadedMsg{Error: err}
		}

		var response PlottersResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.PlottersLoadedMsg{Error: err}
		}

		return messages.PlottersLoadedMsg{FeaturePlotters: response.FeaturePlotters}
	}
}

// LoadConfigSummary runs eeg-pipeline info config --json and returns config summary
func LoadConfigSummary(repoRoot string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "config", "--json"}

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ConfigLoadedMsg{
				Error: err,
			}
		}

		var response ConfigSummaryResponse
		if err := json.Unmarshal(output, &response); err != nil {
			return messages.ConfigLoadedMsg{
				Error: err,
			}
		}

		return messages.ConfigLoadedMsg{
			Summary: messages.ConfigSummary{
				Task:               response.Task,
				BidsRoot:           response.BidsRoot,
				BidsRestRoot:       response.BidsRestRoot,
				BidsFmriRoot:       response.BidsFmriRoot,
				DerivRoot:          response.DerivRoot,
				DerivRestRoot:      response.DerivRestRoot,
				SourceRoot:         response.SourceRoot,
				PreprocessingNJobs: response.PreprocessingNJobs,
			},
			Error: nil,
		}
	}
}

// LoadConfigKeys runs eeg-pipeline info config --json --keys ... and returns values
func LoadConfigKeys(repoRoot string, keys []string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "config", "--json", "--keys"}
		args = append(args, keys...)

		output, err := runPythonJSONCommand(repoRoot, args)
		if err != nil {
			return messages.ConfigKeysLoadedMsg{
				Error: err,
			}
		}

		values := make(map[string]interface{})
		if err := json.Unmarshal(output, &values); err != nil {
			return messages.ConfigKeysLoadedMsg{
				Error: err,
			}
		}
		values = flattenConfigValues(values)

		return messages.ConfigKeysLoadedMsg{
			Values: values,
			Error:  nil,
		}
	}
}

// flattenConfigValues expands nested config objects into dotted keys so
// hydration can bind values regardless of whether config was queried by
// leaf key or section root.
func flattenConfigValues(values map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{})
	var walk func(prefix string, value interface{})
	walk = func(prefix string, value interface{}) {
		if prefix == "" {
			return
		}
		out[prefix] = value
		switch nested := value.(type) {
		case map[string]interface{}:
			for key, child := range nested {
				next := key
				if prefix != "" {
					next = prefix + "." + key
				}
				walk(next, child)
			}
		}
	}

	for key, value := range values {
		if strings.Contains(key, ".") {
			out[key] = value
			continue
		}
		walk(key, value)
	}
	return out
}

type ProgressStreamer struct {
	Command  string
	RepoRoot string
	Events   chan tea.Msg
}

func NewProgressStreamer(repoRoot string, command string) *ProgressStreamer {
	return &ProgressStreamer{
		Command:  command,
		RepoRoot: repoRoot,
		Events:   make(chan tea.Msg, 100),
	}
}

// Start begins streaming progress events
func (ps *ProgressStreamer) Start() tea.Cmd {
	return func() tea.Msg {
		parts := strings.Fields(ps.Command)
		if len(parts) == 0 {
			ps.Events <- messages.CommandDoneMsg{ExitCode: 1}
			close(ps.Events)
			return nil
		}

		var args []string
		if parts[0] == "eeg-pipeline" {
			args = append([]string{"-m", "eeg_pipeline"}, parts[1:]...)
		} else {
			args = parts[1:]
		}
		args = append(args, "--progress-json")

		startTime := time.Now()
		pyCmd := GetPythonCommand(ps.RepoRoot)
		cmd := exec.Command(pyCmd, args...)
		cmd.Dir = ps.RepoRoot
		// Disable colored output and enable unbuffered output
		cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			ps.Events <- messages.CommandDoneMsg{ExitCode: 1, Error: err}
			close(ps.Events)
			return nil
		}

		if err := cmd.Start(); err != nil {
			ps.Events <- messages.CommandDoneMsg{ExitCode: 1, Error: err}
			close(ps.Events)
			return nil
		}

		ps.Events <- messages.CommandStartedMsg{Operation: ps.Command}

		subjectIndex := 0
		totalSubjects := 0

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()

			if strings.HasPrefix(line, "{") {
				var event ProgressEvent
				if json.Unmarshal([]byte(line), &event) == nil {
					switch event.Event {
					case "start":
						totalSubjects = event.TotalSubjects
						ps.Events <- messages.CommandStartedMsg{
							Operation:     event.Operation,
							Subjects:      event.Subjects,
							TotalSubjects: event.TotalSubjects,
						}
					case "subject_start":
						subjectIndex++
						ps.Events <- messages.SubjectStartedMsg{
							Subject: event.Subject,
							Current: subjectIndex,
							Total:   totalSubjects,
						}
					case "progress":
						ps.Events <- messages.StepProgressMsg{
							Subject: event.Subject,
							Step:    event.Step,
							Current: event.Current,
							Total:   event.Total,
							Pct:     event.Pct,
						}
					case "subject_done":
						ps.Events <- messages.SubjectDoneMsg{
							Subject: event.Subject,
							Success: event.Success,
						}
					case "log":
						ps.Events <- messages.LogMsg{
							Level:   event.Level,
							Message: event.Message,
							Subject: event.Subject,
						}
					case "error":
						ps.Events <- messages.LogMsg{
							Level:   "error",
							Message: event.Message,
							Subject: event.Subject,
						}
					}
				}
			} else {
				ps.Events <- messages.LogMsg{
					Level:   "info",
					Message: line,
				}
			}
		}

		err = cmd.Wait()
		duration := time.Since(startTime)

		exitCode := 0
		success := true
		if err != nil {
			success = false
			if exitErr, ok := err.(*exec.ExitError); ok {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 1
			}
		}

		ps.Events <- messages.CommandDoneMsg{
			ExitCode: exitCode,
			Duration: duration,
			Success:  success,
		}
		close(ps.Events)

		return nil
	}
}

// WaitForEvent returns a command that waits for the next event
func (ps *ProgressStreamer) WaitForEvent() tea.Cmd {
	return func() tea.Msg {
		event, ok := <-ps.Events
		if !ok {
			return nil
		}
		return event
	}
}

func runPythonJSONCommand(repoRoot string, args []string) ([]byte, error) {
	pyCmd := GetPythonCommand(repoRoot)
	cmd := exec.Command(pyCmd, args...)
	cmd.Dir = repoRoot
	cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		stderrText := strings.TrimSpace(stderr.String())
		if stderrText != "" {
			return nil, fmt.Errorf("%s", stderrText)
		}
		return nil, err
	}

	return stdout.Bytes(), nil
}
