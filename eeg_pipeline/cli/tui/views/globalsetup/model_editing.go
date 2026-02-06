package globalsetup

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/eeg-pipeline/tui/executor"

	tea "github.com/charmbracelet/bubbletea"
)

// Editing, browse, and overrides persistence helpers.

func (m *Model) moveCursor(delta int) {
	section := m.sections[m.sectionIndex].key
	fields := m.sectionFields(section)
	if len(fields) == 0 {
		return
	}
	m.fieldCursor = (m.fieldCursor + delta + len(fields)) % len(fields)
}

func (m *Model) resetSectionState() {
	m.fieldCursor = 0
	m.statusMessage = ""
	m.statusIsError = false
	m.editingText = false
	m.textBuffer = ""
	m.editingField = 0
}

func (m *Model) activateSelection() (tea.Model, tea.Cmd) {
	section := m.sections[m.sectionIndex].key
	fields := m.sectionFields(section)
	if m.fieldCursor < 0 || m.fieldCursor >= len(fields) {
		return m, nil
	}
	field := fields[m.fieldCursor]
	m.startTextEdit(field.key)
	return m, nil
}

func (m *Model) browseCurrentPath() (tea.Model, tea.Cmd) {
	section := m.sections[m.sectionIndex].key
	fields := m.sectionFields(section)
	if m.fieldCursor < 0 || m.fieldCursor >= len(fields) {
		return m, nil
	}
	field := fields[m.fieldCursor]
	if !field.isPath {
		return m, nil
	}

	return m, m.browseForPath(field.key)
}

func (m Model) browseForPath(field fieldKey) tea.Cmd {
	return func() tea.Msg {
		prompt := "Select folder"
		switch field {
		case fieldBidsRoot:
			prompt = "Select BIDS root folder"
		case fieldBidsFmriRoot:
			prompt = "Select BIDS fMRI data folder"
		case fieldDerivRoot:
			prompt = "Select derivatives root folder"
		case fieldSourceRoot:
			prompt = "Select source data folder"
		}

		result := executor.PickFolder(prompt, fmt.Sprintf("%d", field))()
		if msg, ok := result.(executor.PickFolderMsg); ok {
			return pathPickedMsg{field: field, path: msg.Path, err: msg.Error}
		}
		return pathPickedMsg{field: field, err: fmt.Errorf("unexpected result type")}
	}
}

func (m Model) sectionFields(section sectionKey) []fieldDef {
	switch section {
	case sectionProject:
		return []fieldDef{
			{fieldTask, "Task", "BIDS task label", false},
			{fieldRandomState, "Random State", "Random seed for reproducibility", false},
			{fieldSubjectList, "Subject List", "Comma-separated subject IDs (empty = all)", false},
		}
	case sectionPaths:
		return []fieldDef{
			{fieldBidsRoot, "BIDS Root", "Input BIDS dataset", true},
			{fieldBidsFmriRoot, "BIDS fMRI Root", "fMRI data for contrast builder", true},
			{fieldDerivRoot, "Deriv Root", "Derivatives output", true},
			{fieldSourceRoot, "Source Root", "Raw source data", true},
			{fieldFreesurferDir, "FreeSurfer Dir", "FreeSurfer SUBJECTS_DIR", true},
		}
	case sectionEvents:
		return []fieldDef{
			{fieldEventTemp, "Temperature", "Column candidates", false},
			{fieldEventRating, "Rating", "Column candidates", false},
			{fieldEventPain, "Pain Binary", "Column candidates", false},
		}
	default:
		return []fieldDef{}
	}
}

func (m Model) fieldValue(key fieldKey) string {
	switch key {
	case fieldTask:
		return m.task
	case fieldRandomState:
		return m.randomState
	case fieldSubjectList:
		return m.subjectList
	case fieldBidsRoot:
		return m.bidsRoot
	case fieldBidsFmriRoot:
		return m.bidsFmriRoot
	case fieldDerivRoot:
		return m.derivRoot
	case fieldSourceRoot:
		return m.sourceRoot
	case fieldFreesurferDir:
		return m.freesurferDir
	case fieldEventTemp:
		return m.eventTemp
	case fieldEventRating:
		return m.eventRating
	case fieldEventPain:
		return m.eventPain
	default:
		return ""
	}
}

func (m *Model) setFieldValue(key fieldKey, value string) {
	value = strings.TrimSpace(value)
	switch key {
	case fieldTask:
		m.task = value
	case fieldRandomState:
		m.randomState = value
	case fieldSubjectList:
		m.subjectList = value
	case fieldBidsRoot:
		m.bidsRoot = value
	case fieldBidsFmriRoot:
		m.bidsFmriRoot = value
	case fieldDerivRoot:
		m.derivRoot = value
	case fieldSourceRoot:
		m.sourceRoot = value
	case fieldFreesurferDir:
		m.freesurferDir = value
	case fieldEventTemp:
		m.eventTemp = value
	case fieldEventRating:
		m.eventRating = value
	case fieldEventPain:
		m.eventPain = value
	}
}

func (m *Model) handleTextEdit(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.editingText = false
		m.textBuffer = ""
		m.editingField = 0
	case "enter":
		return m, m.commitTextEdit()
	case "backspace":
		if len(m.textBuffer) > 0 {
			m.textBuffer = m.textBuffer[:len(m.textBuffer)-1]
		}
	case "ctrl+u":
		// Clear entire buffer
		m.textBuffer = ""
	case "ctrl+w":
		// Delete word backwards
		if len(m.textBuffer) > 0 {
			// Find last space or start of string
			lastSpace := strings.LastIndex(m.textBuffer[:len(m.textBuffer)-1], " ")
			if lastSpace == -1 {
				m.textBuffer = ""
			} else {
				m.textBuffer = m.textBuffer[:lastSpace+1]
			}
		}
	default:
		// Handle all printable characters and spaces
		if len(msg.String()) == 1 || msg.String() == " " {
			m.textBuffer += msg.String()
		}
	}
	return m, nil
}

func (m *Model) startTextEdit(key fieldKey) {
	m.editingText = true
	m.editingField = key
	m.textBuffer = m.fieldValue(key)
}

func (m *Model) commitTextEdit() tea.Cmd {
	m.setFieldValue(m.editingField, m.textBuffer)
	m.editingText = false
	m.textBuffer = ""
	m.editingField = 0
	return m.saveOverridesBatch()
}

func (m *Model) saveOverridesBatch() tea.Cmd {
	m.isSaving = true
	return func() tea.Msg {
		overrides := m.buildOverrides()
		data, err := json.MarshalIndent(overrides, "", "  ")
		if err != nil {
			return overridesSavedMsg{err: err}
		}
		os.MkdirAll(filepath.Dir(m.overridesPath), 0755)
		err = os.WriteFile(m.overridesPath, data, 0644)
		return overridesSavedMsg{err: err}
	}
}

func (m *Model) resetOverrides() (tea.Model, tea.Cmd) {
	if err := os.Remove(m.overridesPath); err != nil && !os.IsNotExist(err) {
		m.statusMessage = "Failed to remove overrides"
		m.statusIsError = true
		return m, nil
	}

	m.statusMessage = "Overrides reset to defaults"
	m.statusIsError = false
	return m, tea.Batch(
		executor.LoadConfigSummary(m.repoRoot),
		executor.LoadConfigKeys(m.repoRoot, DefaultConfigKeys()),
	)
}

func (m *Model) buildOverrides() map[string]interface{} {
	overrides := map[string]interface{}{}

	project := map[string]interface{}{}
	if m.task != "" {
		project["task"] = m.task
	}
	if strings.TrimSpace(m.randomState) != "" {
		// Try to parse as integer, fallback to string
		if val, err := strconv.Atoi(strings.TrimSpace(m.randomState)); err == nil {
			project["random_state"] = val
		} else {
			project["random_state"] = m.randomState
		}
	}
	if strings.TrimSpace(m.subjectList) != "" {
		// Parse comma-separated list
		subjects := splitList(m.subjectList)
		if len(subjects) > 0 {
			project["subject_list"] = subjects
		}
	} else {
		// Empty string means null (process all subjects)
		project["subject_list"] = nil
	}
	if len(project) > 0 {
		overrides["project"] = project
	}

	paths := map[string]interface{}{}
	if strings.TrimSpace(m.bidsRoot) != "" {
		paths["bids_root"] = m.bidsRoot
	}
	if strings.TrimSpace(m.bidsFmriRoot) != "" {
		paths["bids_fmri_root"] = m.bidsFmriRoot
	}
	if strings.TrimSpace(m.derivRoot) != "" {
		paths["deriv_root"] = m.derivRoot
	}
	if strings.TrimSpace(m.sourceRoot) != "" {
		paths["source_data"] = m.sourceRoot
	}
	if strings.TrimSpace(m.freesurferDir) != "" {
		paths["freesurfer_dir"] = m.freesurferDir
	}
	if len(paths) > 0 {
		overrides["paths"] = paths
	}

	eventCols := map[string]interface{}{}
	if strings.TrimSpace(m.eventTemp) != "" {
		eventCols["temperature"] = splitList(m.eventTemp)
	}
	if strings.TrimSpace(m.eventRating) != "" {
		eventCols["rating"] = splitList(m.eventRating)
	}
	if strings.TrimSpace(m.eventPain) != "" {
		eventCols["pain_binary"] = splitList(m.eventPain)
	}
	if len(eventCols) > 0 {
		overrides["event_columns"] = eventCols
	}

	return overrides
}

func splitList(value string) []string {
	parts := strings.FieldsFunc(value, func(r rune) bool {
		return r == ',' || r == ';'
	})
	var out []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func toString(value interface{}, fallback string) string {
	switch v := value.(type) {
	case string:
		return v
	default:
		if v == nil {
			return fallback
		}
		return fmt.Sprintf("%v", v)
	}
}

func toStringList(value interface{}) []string {
	switch v := value.(type) {
	case []string:
		return v
	case []interface{}:
		var out []string
		for _, item := range v {
			if item == nil {
				continue
			}
			out = append(out, fmt.Sprintf("%v", item))
		}
		return out
	case string:
		return splitList(v)
	}
	return []string{}
}

func pathExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
