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
	home, err := os.UserHomeDir()
	if err != nil {
		home = ""
	}
	return m.extractDerivRootFromCommand(home)
}

func (m Model) extractDerivRootFromCommand(home string) string {
	if path, ok := m.extractFlagPath("--deriv-root", home); ok {
		return path
	}
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
			path = strings.ReplaceAll(matches[2], "''", "'") // Single-quoted
		} else {
			path = matches[3] // Unquoted
		}

		return normalizeExtractedPath(path, home, m.RepoRoot)
	}

	return ""
}

// extractBidsFmriRoot extracts the --bids-fmri-root argument from the command string
func (m Model) extractBidsFmriRoot() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = ""
	}
	return m.extractBidsFmriRootFromCommand(home)
}

func (m Model) extractBidsFmriRootFromCommand(home string) string {
	if path, ok := m.extractFlagPath("--bids-fmri-root", home); ok {
		return path
	}
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
			path = strings.ReplaceAll(matches[2], "''", "'")
		} else {
			path = matches[3]
		}

		return normalizeExtractedPath(path, home, m.RepoRoot)
	}
	return ""
}

func (m Model) extractFlagPath(flag string, home string) (string, bool) {
	if len(m.CommandArgs) == 0 {
		return "", false
	}
	parts := m.CommandArgs

	for index := 0; index < len(parts); index++ {
		part := parts[index]
		if part == flag {
			if index+1 >= len(parts) {
				return "", false
			}
			return normalizeExtractedPath(parts[index+1], home, m.RepoRoot), true
		}

		prefix := flag + "="
		if strings.HasPrefix(part, prefix) {
			return normalizeExtractedPath(strings.TrimPrefix(part, prefix), home, m.RepoRoot), true
		}
	}

	return "", false
}

func normalizeExtractedPath(path string, home string, repoRoot string) string {
	path = expandParsedPathWithHome(path, home)
	if !filepath.IsAbs(path) {
		path = filepath.Join(repoRoot, path)
	}
	return filepath.Clean(path)
}

func expandParsedPathWithHome(path string, home string) string {
	if home == "" {
		return path
	}
	if path == "~" {
		return filepath.Clean(home)
	}
	if len(path) >= 2 && (path[1] == '/' || path[1] == '\\') {
		relative := strings.NewReplacer(
			"/", string(filepath.Separator),
			"\\", string(filepath.Separator),
		).Replace(path[2:])
		return filepath.Clean(filepath.Join(home, relative))
	}
	return path
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
