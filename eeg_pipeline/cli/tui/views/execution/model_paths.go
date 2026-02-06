package execution

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
)

// Output-path parsing and results-folder opening helpers.

func (m Model) GetOutputPaths() []string {
	if m.RepoRoot == "" {
		return nil
	}

	// Parse --deriv-root from command if present
	base := m.extractDerivRoot()
	if base == "" {
		// Fallback to default location
		base = filepath.Join(m.RepoRoot, "eeg_pipeline", "data", "derivatives")
	}

	// Parse pipeline from command
	cmd := strings.ToLower(m.Command)
	var paths []string

	switch {
	case strings.Contains(cmd, "preprocess"):
		paths = []string{
			filepath.Join(base, "preprocessed", "eeg"),
			filepath.Join(base, "epochs"),
		}
	case strings.Contains(cmd, "features"):
		paths = []string{filepath.Join(base, "features")}
	case strings.Contains(cmd, "behavior"):
		paths = []string{
			filepath.Join(base, "behavior"),
			filepath.Join(base, "stats"),
		}
	case strings.Contains(cmd, "fmri-analysis"):
		paths = []string{filepath.Join(base, "sub-XX", "fmri", "first_level")}
	case strings.Contains(cmd, "fmri") && strings.Contains(cmd, "preprocess"):
		paths = []string{filepath.Join(base, "preprocessed", "fmri", "fmriprep")}
	case strings.Contains(cmd, "fmri-raw-to-bids"):
		// Extract bids-fmri-root from command or use default
		bidsFmriRoot := m.extractBidsFmriRoot()
		if bidsFmriRoot == "" {
			bidsFmriRoot = filepath.Join(m.RepoRoot, "eeg_pipeline", "data", "bids_output", "fmri")
		}
		paths = []string{bidsFmriRoot}
	case strings.Contains(cmd, " ml "):
		paths = []string{filepath.Join(base, "machine_learning")}
	case strings.Contains(cmd, "plot"):
		paths = []string{filepath.Join(base, "plots")}
	default:
		paths = []string{base}
	}

	// Filter to paths that exist
	var existingPaths []string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			existingPaths = append(existingPaths, p)
		}
	}

	if len(existingPaths) == 0 {
		return paths // Return expected paths even if they don't exist yet
	}
	return existingPaths
}

// extractDerivRoot extracts the --deriv-root argument from the command string
func (m Model) extractDerivRoot() string {
	if m.Command == "" {
		return ""
	}

	// Pattern to match --deriv-root with optional equals sign
	// Matches: --deriv-root /path, --deriv-root=/path, --deriv-root "path with spaces"
	// Handles both quoted and unquoted paths
	pattern := regexp.MustCompile(`--deriv-root(?:=|\s+)(?:"([^"]+)"|'([^']+)'|([^\s]+))`)
	matches := pattern.FindStringSubmatch(m.Command)
	if len(matches) > 1 {
		var path string
		// Check which capture group matched (quoted double, quoted single, or unquoted)
		if matches[1] != "" {
			path = matches[1] // Double-quoted
		} else if matches[2] != "" {
			path = matches[2] // Single-quoted
		} else {
			path = matches[3] // Unquoted
		}

		// Expand user home directory if path starts with ~
		if strings.HasPrefix(path, "~") {
			home, err := os.UserHomeDir()
			if err == nil {
				path = filepath.Join(home, strings.TrimPrefix(path, "~"))
			}
		}
		// Convert to absolute path
		if !filepath.IsAbs(path) {
			// If relative, make it relative to repo root
			path = filepath.Join(m.RepoRoot, path)
		}
		return filepath.Clean(path)
	}

	return ""
}

// extractBidsFmriRoot extracts the --bids-fmri-root argument from the command string
func (m Model) extractBidsFmriRoot() string {
	if m.Command == "" {
		return ""
	}

	// Pattern to match --bids-fmri-root with optional equals sign
	pattern := regexp.MustCompile(`--bids-fmri-root(?:=|\s+)(?:"([^"]+)"|'([^']+)'|([^\s]+))`)
	matches := pattern.FindStringSubmatch(m.Command)
	if len(matches) > 1 {
		var path string
		if matches[1] != "" {
			path = matches[1]
		} else if matches[2] != "" {
			path = matches[2]
		} else {
			path = matches[3]
		}

		if strings.HasPrefix(path, "~") {
			home, err := os.UserHomeDir()
			if err == nil {
				path = filepath.Join(home, strings.TrimPrefix(path, "~"))
			}
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(m.RepoRoot, path)
		}
		return filepath.Clean(path)
	}
	return ""
}

// OpenResultsFolder opens the first output path in the system file browser
func (m Model) OpenResultsFolder() tea.Cmd {
	paths := m.GetOutputPaths()
	if len(paths) == 0 {
		m.addLog(fmt.Sprintf("%s No output paths found", styles.CrossMark))
		return nil
	}

	targetPath := paths[0]

	// Ensure the path exists and is a directory
	if info, err := os.Stat(targetPath); err != nil {
		// Path doesn't exist, try parent directory
		parent := filepath.Dir(targetPath)
		if parentInfo, err := os.Stat(parent); err == nil && parentInfo.IsDir() {
			targetPath = parent
		} else {
			m.addLog(fmt.Sprintf("%s Results folder not found: %s", styles.CrossMark, targetPath))
			return nil
		}
	} else if !info.IsDir() {
		// It's a file, use parent directory
		targetPath = filepath.Dir(targetPath)
	}

	// Verify the final path exists and convert to absolute path
	absPath, err := filepath.Abs(targetPath)
	if err != nil {
		m.addLog(fmt.Sprintf("%s Cannot resolve path: %s", styles.CrossMark, targetPath))
		return nil
	}

	if _, err := os.Stat(absPath); err != nil {
		m.addLog(fmt.Sprintf("%s Cannot open folder (does not exist): %s", styles.CrossMark, absPath))
		return nil
	}

	// Log the path we're trying to open for debugging
	m.addLog(fmt.Sprintf("Opening results folder: %s", absPath))
	return executor.OpenInFileBrowserCmd(absPath)
}

// updateViewportSize recalculates log viewport dimensions based on current state
