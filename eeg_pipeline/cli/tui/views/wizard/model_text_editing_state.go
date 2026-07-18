package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/messages"
)

// Wizard text-editing state transitions and summary injection.

func (m *Model) SetConfigSummary(summary messages.ConfigSummary) {
	if (m.task == "" || m.task == "task") && summary.Task != "" {
		m.task = summary.Task
	}
	if summary.BidsRoot != "" {
		m.bidsRoot = summary.BidsRoot
	}
	if summary.BidsRestRoot != "" {
		m.bidsRestRoot = summary.BidsRestRoot
	}
	if summary.BidsFmriRoot != "" {
		m.bidsFmriRoot = summary.BidsFmriRoot
	}
	if summary.DerivRoot != "" {
		m.derivRoot = summary.DerivRoot
		m.fmriSecondLevelContrastDiscoveryKey = ""
	}
	if summary.DerivRestRoot != "" {
		m.derivRestRoot = summary.DerivRestRoot
	}
	if summary.SourceRoot != "" {
		m.sourceRoot = summary.SourceRoot
	}
	if summary.PreprocessingNJobs > 0 {
		m.prepNJobs = summary.PreprocessingNJobs
	}
}

func (m *Model) SetTask(task string) {
	task = strings.TrimSpace(task)
	if task == "" {
		return
	}
	m.task = task
	m.fmriSecondLevelContrastDiscoveryKey = ""
}

func (m *Model) SetRepoRoot(repoRoot string) {
	m.repoRoot = repoRoot
}

func (m *Model) startTextEdit(field textField) {
	m.editingTextField = field
	m.textBuffer = m.getTextFieldValue(field)
	m.editingText = true
}

func (m *Model) commitTextInput() {
	m.setTextFieldValue(m.editingTextField, m.textBuffer)
}
