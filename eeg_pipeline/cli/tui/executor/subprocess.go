package executor

import (
	"bufio"
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
	Subjects              []messages.SubjectInfo `json:"subjects"`
	Count                 int                    `json:"count"`
	AvailableWindows      []string               `json:"available_windows"`
	AvailableEventColumns []string               `json:"available_event_columns"`
}

// ConfigSummaryResponse from eeg-pipeline info config --json
type ConfigSummaryResponse struct {
	Task               string `json:"task"`
	BidsRoot           string `json:"bids_root"`
	DerivRoot          string `json:"deriv_root"`
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
	Source  string              `json:"source"`
	File    string              `json:"file,omitempty"`
}

// DiscoverColumns runs eeg-pipeline discover --json to find available columns and their values
func DiscoverColumns(repoRoot string, task string) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "discover", "all", "--json"}
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
			Source:  response.Source,
			Error:   nil,
		}
	}
}

// LoadSubjects runs eeg-pipeline info subjects --json and returns subjects
func LoadSubjects(repoRoot string, task string, pipeline types.Pipeline) tea.Cmd {
	return func() tea.Msg {
		args := []string{"-m", "eeg_pipeline", "info", "subjects", "--status", "--json"}
		if task != "" {
			args = append(args, "--task", task)
		}

		// Use the pipeline's GetDataSource method for proper source determination
		source := pipeline.GetDataSource()
		args = append(args, "--source", source)

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
			Subjects:              response.Subjects,
			AvailableWindows:      response.AvailableWindows,
			AvailableEventColumns: response.AvailableEventColumns,
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
				DerivRoot:          response.DerivRoot,
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

		return messages.ConfigKeysLoadedMsg{
			Values: values,
			Error:  nil,
		}
	}
}

// RunPipelineCommand executes a pipeline command with progress streaming
func RunPipelineCommand(repoRoot string, command string) tea.Cmd {
	return func() tea.Msg {
		parts, err := splitShellWords(command)
		if err != nil {
			parts = strings.Fields(command)
		}
		if len(parts) == 0 {
			return messages.CommandDoneMsg{ExitCode: 1, Error: nil}
		}

		// Convert eeg-pipeline to python -m eeg_pipeline
		var args []string
		if parts[0] == "eeg-pipeline" {
			args = append([]string{"-m", "eeg_pipeline"}, parts[1:]...)
		} else {
			args = parts[1:]
		}

		// Add --progress-json flag for real-time progress
		args = append(args, "--progress-json")

		startTime := time.Now()
		pyCmd := GetPythonCommand(repoRoot)
		cmd := exec.Command(pyCmd, args...)
		cmd.Dir = repoRoot
		// Disable colored output and enable unbuffered output
		cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return messages.CommandDoneMsg{ExitCode: 1, Error: err, Duration: time.Since(startTime)}
		}

		stderr, _ := cmd.StderrPipe()

		if err := cmd.Start(); err != nil {
			return messages.CommandDoneMsg{ExitCode: 1, Error: err, Duration: time.Since(startTime)}
		}

		// Read stdout for progress JSON
		go func() {
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				line := scanner.Text()
				// Try to parse as JSON progress event
				if strings.HasPrefix(line, "{") {
					var event ProgressEvent
					if json.Unmarshal([]byte(line), &event) == nil {
						// Events would be sent via channel in full implementation
						// For now, just consume
						_ = event
					}
				}
			}
		}()

		// Read stderr
		go func() {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				_ = scanner.Text()
			}
		}()

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

		return messages.CommandDoneMsg{
			ExitCode: exitCode,
			Duration: duration,
			Success:  success,
			Error:    nil,
		}
	}
}

func splitShellWords(raw string) ([]string, error) {
	type quoteState int
	const (
		stateNone quoteState = iota
		stateSingle
		stateDouble
	)

	var out []string
	var cur strings.Builder
	state := stateNone
	escaped := false

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		out = append(out, cur.String())
		cur.Reset()
	}

	for _, r := range raw {
		if escaped {
			cur.WriteRune(r)
			escaped = false
			continue
		}

		switch state {
		case stateNone:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '\'' {
				state = stateSingle
				continue
			}
			if r == '"' {
				state = stateDouble
				continue
			}
			if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
				flush()
				continue
			}
			cur.WriteRune(r)
		case stateSingle:
			if r == '\'' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		case stateDouble:
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				state = stateNone
				continue
			}
			cur.WriteRune(r)
		}
	}

	if escaped {
		return nil, fmt.Errorf("unfinished escape sequence")
	}
	if state != stateNone {
		return nil, fmt.Errorf("unterminated quote")
	}
	flush()
	return out, nil
}

// ProgressStreamCmd creates a command that streams progress events
// This returns the initial command and a channel for progress events
type ProgressStreamer struct {
	Command  string
	RepoRoot string
	Events   chan tea.Msg
	Done     chan bool
}

// NewProgressStreamer creates a streamer for real-time progress
func NewProgressStreamer(repoRoot string, command string) *ProgressStreamer {
	return &ProgressStreamer{
		Command:  command,
		RepoRoot: repoRoot,
		Events:   make(chan tea.Msg, 100),
		Done:     make(chan bool),
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

		// Build command
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

		// Send started message
		ps.Events <- messages.CommandStartedMsg{Operation: ps.Command}

		// Track subjects for progress
		subjectIndex := 0
		totalSubjects := 0

		// Read and parse progress events
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
					case "complete":
						// Will be handled by cmd.Wait()
					case "error":
						ps.Events <- messages.LogMsg{
							Level:   "error",
							Message: event.Message,
							Subject: event.Subject,
						}
					}
				}
			} else {
				// Regular log line
				ps.Events <- messages.LogMsg{
					Level:   "info",
					Message: line,
				}
			}
		}

		// Wait for command to finish
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

// runPythonJSONCommand executes a Python module command in the repository root,
// ensuring consistent environment configuration and a python3 fallback. It
// returns the command's stdout if successful, or an error if both attempts fail.
func runPythonJSONCommand(repoRoot string, args []string) ([]byte, error) {
	pyCmd := GetPythonCommand(repoRoot)
	cmd := exec.Command(pyCmd, args...)
	cmd.Dir = repoRoot
	cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

	output, err := cmd.Output()
	if err == nil {
		return output, nil
	}

	fallbackCmd := exec.Command("python3", args...)
	fallbackCmd.Dir = repoRoot
	fallbackCmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

	output, err = fallbackCmd.Output()
	if err != nil {
		return nil, err
	}

	return output, nil
}
