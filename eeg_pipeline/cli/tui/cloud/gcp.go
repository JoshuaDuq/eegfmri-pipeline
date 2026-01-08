package cloud

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// Config holds GCP cloud configuration
type Config struct {
	// SSH settings
	RemoteHost   string // SSH host alias or IP
	RemoteUser   string // SSH username (optional if using ssh config)
	RemoteBase   string // Remote base path
	VenvActivate string // Path to venv activate script on remote

	// GCP settings for auto-stop
	GCPProject  string
	GCPZone     string
	GCPInstance string

	// Options
	SyncCode bool
	SyncData bool
	AutoPull bool
	AutoStop bool
	NJobs    int
}

// DefaultConfig returns default cloud configuration
func DefaultConfig() Config {
	return Config{
		RemoteHost:   "thermal-gcp",
		RemoteBase:   "/mnt/data/Thermal_Pain_EEG_Pipeline",
		VenvActivate: "/mnt/data/Thermal_Pain_EEG_Pipeline/.venv/bin/activate",
		GCPProject:   "eegpipeline-481605",
		GCPZone:      "us-central1-a",
		GCPInstance:  "eegpipeline",
		SyncCode:     true,
		SyncData:     true,
		AutoPull:     true,
		AutoStop:     false,
		NJobs:        -1,
	}
}

// Messages for cloud operations
type (
	VMStatusMsg struct {
		Running bool
		IP      string
		Error   error
	}
	SyncStartedMsg  struct{}
	SyncCompleteMsg struct {
		Duration time.Duration
		Error    error
	}
	RunStartedMsg struct {
		Command string
	}
	RunOutputMsg struct {
		Line string
	}
	RunCompleteMsg struct {
		ExitCode int
		Duration time.Duration
		Error    error
	}
	PullStartedMsg  struct{}
	PullCompleteMsg struct {
		Duration time.Duration
		Error    error
	}
)

// CheckVMStatus checks if the GCP VM is running
func CheckVMStatus(cfg Config) tea.Cmd {
	return func() tea.Msg {
		// Try SSH connection test
		cmd := exec.Command("ssh", "-o", "ConnectTimeout=5", cfg.getRemoteTarget(), "hostname")
		output, err := cmd.Output()

		if err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		return VMStatusMsg{Running: true, IP: strings.TrimSpace(string(output))}
	}
}

// StartVM starts the GCP instance
func StartVM(cfg Config) tea.Cmd {
	return func() tea.Msg {
		if cfg.GCPProject == "" || cfg.GCPZone == "" || cfg.GCPInstance == "" {
			return VMStatusMsg{Running: false, Error: fmt.Errorf("GCP config not set")}
		}

		cmd := exec.Command("gcloud", "compute", "instances", "start",
			cfg.GCPInstance,
			"--zone", cfg.GCPZone,
			"--project", cfg.GCPProject,
		)

		if err := cmd.Run(); err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		// Wait a bit for SSH to become available
		time.Sleep(10 * time.Second)

		return VMStatusMsg{Running: true}
	}
}

// StopVM stops the GCP instance
func StopVM(cfg Config) tea.Cmd {
	return func() tea.Msg {
		if cfg.GCPProject == "" || cfg.GCPZone == "" || cfg.GCPInstance == "" {
			return VMStatusMsg{Running: true, Error: fmt.Errorf("GCP config not set")}
		}

		cmd := exec.Command("gcloud", "compute", "instances", "stop",
			cfg.GCPInstance,
			"--zone", cfg.GCPZone,
			"--project", cfg.GCPProject,
		)

		if err := cmd.Run(); err != nil {
			return VMStatusMsg{Running: true, Error: err}
		}

		return VMStatusMsg{Running: false}
	}
}

// SyncToRemote syncs code and data to the remote VM
func SyncToRemote(ctx context.Context, cfg Config, repoRoot string) tea.Cmd {
	return func() tea.Msg {
		startTime := time.Now()

		if cfg.SyncCode {
			if err := syncCode(ctx, cfg, repoRoot); err != nil {
				return SyncCompleteMsg{Error: err, Duration: time.Since(startTime)}
			}
		}

		if cfg.SyncData {
			dataDir := filepath.Join(repoRoot, "eeg_pipeline", "data")
			if err := syncData(ctx, cfg, dataDir); err != nil {
				return SyncCompleteMsg{Error: err, Duration: time.Since(startTime)}
			}
		}

		return SyncCompleteMsg{Duration: time.Since(startTime)}
	}
}

// RunRemoteCommand runs a pipeline command on the remote VM
func RunRemoteCommand(ctx context.Context, cfg Config, pipelineCmd string) tea.Cmd {
	return func() tea.Msg {
		startTime := time.Now()

		// Build the remote command
		prefix := fmt.Sprintf("cd %s", cfg.RemoteBase)
		if cfg.VenvActivate != "" {
			prefix += fmt.Sprintf(" && source %s", cfg.VenvActivate)
		}
		prefix += fmt.Sprintf(" && export EEG_PIPELINE_N_JOBS=%d && export MNE_N_JOBS=%d", cfg.NJobs, cfg.NJobs)

		// Strip "eeg-pipeline " prefix - BuildCommand() returns full command
		// but python3 -m eeg_pipeline already provides the entry point
		args := strings.TrimPrefix(pipelineCmd, "eeg-pipeline ")

		fullCmd := fmt.Sprintf("%s && python3 -m eeg_pipeline %s", prefix, args)

		cmd := exec.CommandContext(ctx, "ssh", cfg.getRemoteTarget(), fullCmd)

		if err := cmd.Run(); err != nil {
			exitCode := 0
			if exitErr, ok := err.(*exec.ExitError); ok {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 1
			}
			return RunCompleteMsg{ExitCode: exitCode, Error: err, Duration: time.Since(startTime)}
		}

		return RunCompleteMsg{ExitCode: 0, Duration: time.Since(startTime)}
	}
}

// PullDerivatives pulls results from the remote VM
func PullDerivatives(ctx context.Context, cfg Config, localDataDir string) tea.Cmd {
	return func() tea.Msg {
		startTime := time.Now()

		remote := fmt.Sprintf("%s:%s/eeg_pipeline/data/derivatives/", cfg.getRemoteTarget(), cfg.RemoteBase)
		local := filepath.Join(localDataDir, "derivatives") + "/"

		// Ensure local directory exists
		os.MkdirAll(local, 0755)

		args := []string{
			"-avz", "--partial", "--progress",
			"--exclude", "preprocessed",
			"--exclude", ".DS_Store",
			remote, local,
		}

		cmd := exec.CommandContext(ctx, "rsync", args...)
		if err := cmd.Run(); err != nil {
			return PullCompleteMsg{Error: err, Duration: time.Since(startTime)}
		}

		return PullCompleteMsg{Duration: time.Since(startTime)}
	}
}

// Helper methods
func (cfg Config) getRemoteTarget() string {
	if cfg.RemoteUser != "" {
		return fmt.Sprintf("%s@%s", cfg.RemoteUser, cfg.RemoteHost)
	}
	return cfg.RemoteHost
}

func syncCode(ctx context.Context, cfg Config, repoRoot string) error {
	remote := fmt.Sprintf("%s:%s/", cfg.getRemoteTarget(), cfg.RemoteBase)

	args := []string{
		"-avz", "--delete", "--partial",
		"--exclude", ".git",
		"--exclude", "__pycache__",
		"--exclude", "*.pyc",
		"--exclude", ".venv",
		"--exclude", "venv",
		"--exclude", ".venv311",
		"--exclude", "eeg_pipeline/data",
		"--exclude", ".DS_Store",
		repoRoot + "/", remote,
	}

	cmd := exec.CommandContext(ctx, "rsync", args...)
	return cmd.Run()
}

func syncData(ctx context.Context, cfg Config, dataDir string) error {
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		return nil // Skip if data dir doesn't exist
	}

	remote := fmt.Sprintf("%s:%s/eeg_pipeline/data/", cfg.getRemoteTarget(), cfg.RemoteBase)

	args := []string{
		"-avz", "--delete", "--partial",
		"--exclude", ".DS_Store",
		dataDir + "/", remote,
	}

	cmd := exec.CommandContext(ctx, "rsync", args...)
	return cmd.Run()
}
