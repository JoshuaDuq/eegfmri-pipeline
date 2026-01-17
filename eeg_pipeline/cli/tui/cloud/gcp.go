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

const (
	vmStartupWaitDuration = 10 * time.Second
	defaultDirectoryPerms = 0755
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
	SyncCompleteMsg struct {
		Duration time.Duration
		Error    error
	}
	RunCompleteMsg struct {
		ExitCode int
		Duration time.Duration
		Error    error
	}
	PullCompleteMsg struct {
		Duration time.Duration
		Error    error
	}
)

// CheckVMStatus checks if the GCP VM is running using gcloud API
func CheckVMStatus(cfg Config) tea.Cmd {
	return func() tea.Msg {
		if err := validateGCPConfig(cfg); err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		// Use gcloud to get the actual VM status from GCP API
		cmd := exec.Command("gcloud", "compute", "instances", "describe",
			cfg.GCPInstance,
			"--zone", cfg.GCPZone,
			"--project", cfg.GCPProject,
			"--format", "value(status)",
		)

		output, err := cmd.Output()
		if err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		status := strings.TrimSpace(string(output))
		isRunning := status == "RUNNING"

		return VMStatusMsg{Running: isRunning, IP: cfg.GCPInstance}
	}
}

// StartVM starts the GCP instance
func StartVM(cfg Config) tea.Cmd {
	return func() tea.Msg {
		if err := validateGCPConfig(cfg); err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		cmd := buildGCPStartCommand(cfg)
		if err := cmd.Run(); err != nil {
			return VMStatusMsg{Running: false, Error: err}
		}

		time.Sleep(vmStartupWaitDuration)
		return VMStatusMsg{Running: true}
	}
}

// StopVM stops the GCP instance
func StopVM(cfg Config) tea.Cmd {
	return func() tea.Msg {
		if err := validateGCPConfig(cfg); err != nil {
			return VMStatusMsg{Running: true, Error: err}
		}

		cmd := buildGCPStopCommand(cfg)
		if err := cmd.Run(); err != nil {
			return VMStatusMsg{Running: true, Error: err}
		}

		return VMStatusMsg{Running: false}
	}
}

// StopVMSync stops the GCP instance synchronously (for use in cleanup/exit)
func StopVMSync(cfg Config) error {
	if err := validateGCPConfig(cfg); err != nil {
		return err
	}

	cmd := buildGCPStopCommand(cfg)
	return cmd.Run()
}

// IsVMRunning checks if the VM is currently running (synchronous)
func IsVMRunning(cfg Config) bool {
	if err := validateGCPConfig(cfg); err != nil {
		return false
	}

	cmd := exec.Command("gcloud", "compute", "instances", "describe",
		cfg.GCPInstance,
		"--zone", cfg.GCPZone,
		"--project", cfg.GCPProject,
		"--format", "value(status)",
	)

	output, err := cmd.Output()
	if err != nil {
		return false
	}

	status := strings.TrimSpace(string(output))
	return status == "RUNNING"
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
		remoteCommand := buildRemoteCommand(cfg, pipelineCmd)
		cmd := exec.CommandContext(ctx, "ssh", cfg.getRemoteTarget(), remoteCommand)

		if err := cmd.Run(); err != nil {
			exitCode := extractExitCode(err)
			return RunCompleteMsg{
				ExitCode: exitCode,
				Error:    err,
				Duration: time.Since(startTime),
			}
		}

		return RunCompleteMsg{ExitCode: 0, Duration: time.Since(startTime)}
	}
}

// PullDerivatives pulls results from the remote VM
func PullDerivatives(ctx context.Context, cfg Config, localDataDir string) tea.Cmd {
	return func() tea.Msg {
		startTime := time.Now()
		remotePath := buildRemoteDerivativesPath(cfg)
		localPath := filepath.Join(localDataDir, "derivatives") + "/"

		if err := ensureDirectoryExists(localPath); err != nil {
			return PullCompleteMsg{Error: err, Duration: time.Since(startTime)}
		}

		rsyncArgs := buildPullRsyncArgs(remotePath, localPath)
		cmd := exec.CommandContext(ctx, "rsync", rsyncArgs...)

		if err := cmd.Run(); err != nil {
			return PullCompleteMsg{Error: err, Duration: time.Since(startTime)}
		}

		return PullCompleteMsg{Duration: time.Since(startTime)}
	}
}

func (cfg Config) getRemoteTarget() string {
	if cfg.RemoteUser != "" {
		return fmt.Sprintf("%s@%s", cfg.RemoteUser, cfg.RemoteHost)
	}
	return cfg.RemoteHost
}

func validateGCPConfig(cfg Config) error {
	if cfg.GCPProject == "" || cfg.GCPZone == "" || cfg.GCPInstance == "" {
		return fmt.Errorf("GCP config not set")
	}
	return nil
}

func buildGCPStartCommand(cfg Config) *exec.Cmd {
	return exec.Command("gcloud", "compute", "instances", "start",
		cfg.GCPInstance,
		"--zone", cfg.GCPZone,
		"--project", cfg.GCPProject,
	)
}

func buildGCPStopCommand(cfg Config) *exec.Cmd {
	return exec.Command("gcloud", "compute", "instances", "stop",
		cfg.GCPInstance,
		"--zone", cfg.GCPZone,
		"--project", cfg.GCPProject,
	)
}

func buildRemoteCommand(cfg Config, pipelineCmd string) string {
	commandParts := []string{fmt.Sprintf("cd %s", cfg.RemoteBase)}

	if cfg.VenvActivate != "" {
		commandParts = append(commandParts, fmt.Sprintf("source %s", cfg.VenvActivate))
	}

	envVars := fmt.Sprintf("export EEG_PIPELINE_N_JOBS=%d && export MNE_N_JOBS=%d", cfg.NJobs, cfg.NJobs)
	commandParts = append(commandParts, envVars)

	pipelineArgs := strings.TrimPrefix(pipelineCmd, "eeg-pipeline ")
	pythonCommand := fmt.Sprintf("python3 -m eeg_pipeline %s", pipelineArgs)
	commandParts = append(commandParts, pythonCommand)

	return strings.Join(commandParts, " && ")
}

func extractExitCode(err error) int {
	if exitErr, ok := err.(*exec.ExitError); ok {
		return exitErr.ExitCode()
	}
	return 1
}

func buildRemoteDerivativesPath(cfg Config) string {
	return fmt.Sprintf("%s:%s/eeg_pipeline/data/derivatives/", cfg.getRemoteTarget(), cfg.RemoteBase)
}

func ensureDirectoryExists(path string) error {
	return os.MkdirAll(path, defaultDirectoryPerms)
}

func buildPullRsyncArgs(remotePath, localPath string) []string {
	return []string{
		"-avz", "--partial", "--progress",
		"--exclude", "preprocessed",
		"--exclude", ".DS_Store",
		remotePath, localPath,
	}
}

func syncCode(ctx context.Context, cfg Config, repoRoot string) error {
	remotePath := fmt.Sprintf("%s:%s/", cfg.getRemoteTarget(), cfg.RemoteBase)
	rsyncArgs := buildCodeSyncArgs(repoRoot, remotePath)
	cmd := exec.CommandContext(ctx, "rsync", rsyncArgs...)
	return cmd.Run()
}

func buildCodeSyncArgs(repoRoot, remotePath string) []string {
	return []string{
		"-avz", "--delete", "--partial",
		"--exclude", ".git",
		"--exclude", "__pycache__",
		"--exclude", "*.pyc",
		"--exclude", ".venv",
		"--exclude", "venv",
		"--exclude", "eeg_pipeline/.venv311",
		"--exclude", "eeg_pipeline/data",
		"--exclude", ".DS_Store",
		repoRoot + "/", remotePath,
	}
}

func syncData(ctx context.Context, cfg Config, dataDir string) error {
	if !directoryExists(dataDir) {
		return nil
	}

	remotePath := fmt.Sprintf("%s:%s/eeg_pipeline/data/", cfg.getRemoteTarget(), cfg.RemoteBase)
	rsyncArgs := buildDataSyncArgs(dataDir, remotePath)
	cmd := exec.CommandContext(ctx, "rsync", rsyncArgs...)
	return cmd.Run()
}

func directoryExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func buildDataSyncArgs(dataDir, remotePath string) []string {
	return []string{
		"-avz", "--delete", "--partial",
		"--exclude", ".DS_Store",
		dataDir + "/", remotePath,
	}
}
