package globalsetup

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
)

// File layout notes:
// - `model.go`: global-setup model types, initialization, Tea lifecycle core.
// - `model_render.go`: rendering of header/footer/field list.
// - `model_editing.go`: cursor movement, text editing, browse/save/reset helpers.

type sectionKey int

const (
	sectionProject sectionKey = iota
	sectionPaths
)

type fieldKey int

const (
	fieldTask fieldKey = iota
	fieldRandomState
	fieldSubjectList
	fieldBidsRoot
	fieldBidsFmriRoot
	fieldDerivRoot
	fieldSourceRoot
	fieldFreesurferDir
)

type sectionDef struct {
	key         sectionKey
	label       string
	description string
}

type fieldDef struct {
	key         fieldKey
	label       string
	description string
	isPath      bool
}

type pathPickedMsg struct {
	field fieldKey
	path  string
	err   error
}

type Model struct {
	repoRoot      string
	overridesPath string

	sections     []sectionDef
	sectionIndex int
	fieldCursor  int

	editingText  bool
	textBuffer   string
	editingField fieldKey

	width  int
	height int

	task          string
	randomState   string
	subjectList   string
	bidsRoot      string
	bidsFmriRoot  string
	derivRoot     string
	sourceRoot    string
	freesurferDir string

	isLoading     bool
	isSaving      bool
	statusMessage string
	statusIsError bool

	animQueue     animation.Queue
	searchSpinner components.Spinner
	saveSpinner   components.Spinner

	Done bool
}

type overridesSavedMsg struct {
	err error
}

func DefaultConfigKeys() []string {
	return []string{
		"project.task",
		"project.random_state",
		"project.subject_list",
		"paths.bids_root",
		"paths.bids_fmri_root",
		"paths.deriv_root",
		"paths.source_data",
		"paths.freesurfer_dir",
	}
}

func New(repoRoot string) Model {
	overridesPath := filepath.Join(repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_overrides.json")
	m := Model{
		repoRoot:      repoRoot,
		overridesPath: overridesPath,
		sections: []sectionDef{
			{sectionProject, "Project", "Task, random seed, and subject filter"},
			{sectionPaths, "Paths", "BIDS and data roots"},
		},
		isLoading: true,
	}
	m.animQueue.Push(animation.ProgressPulseLoop())
	m.searchSpinner = components.NewSpinner("Searching for paths and configuration...")
	m.saveSpinner = components.NewSpinner("Saving changes...")
	return m
}

func (m *Model) SetSize(width, height int) {
	m.width = width
	m.height = height
}

func (m *Model) SetConfigValues(values map[string]interface{}) {
	m.isLoading = false
	if v, ok := values["project.task"]; ok {
		m.task = toString(v, m.task)
	}
	if v, ok := values["project.random_state"]; ok {
		m.randomState = toString(v, m.randomState)
	}
	if v, ok := values["project.subject_list"]; ok {
		// subject_list can be null, a list, or a string
		if v == nil {
			m.subjectList = ""
		} else if list, ok := v.([]interface{}); ok {
			if len(list) > 0 {
				var strs []string
				for _, item := range list {
					strs = append(strs, fmt.Sprintf("%v", item))
				}
				m.subjectList = strings.Join(strs, ", ")
			} else {
				m.subjectList = ""
			}
		} else {
			m.subjectList = toString(v, m.subjectList)
		}
	}
	if v, ok := values["paths.bids_root"]; ok {
		m.bidsRoot = toString(v, m.bidsRoot)
	}
	if v, ok := values["paths.bids_fmri_root"]; ok {
		m.bidsFmriRoot = toString(v, m.bidsFmriRoot)
	}
	if v, ok := values["paths.deriv_root"]; ok {
		m.derivRoot = toString(v, m.derivRoot)
	}
	if v, ok := values["paths.source_data"]; ok {
		m.sourceRoot = toString(v, m.sourceRoot)
	}
	if v, ok := values["paths.freesurfer_dir"]; ok {
		m.freesurferDir = toString(v, m.freesurferDir)
	}
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.tick(), m.immediateTick())
}

func (m Model) immediateTick() tea.Cmd {
	return tea.Tick(0, func(t time.Time) tea.Msg { return tickMsg{} })
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

type tickMsg struct{}

// IsEditing reports whether the global setup view is currently editing a
// configuration field. When true, global keybindings like quitting should
// be suppressed so that the text editor can consume the keys.
func (m Model) IsEditing() bool {
	return m.editingText
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.animQueue.Tick()
		m.searchSpinner.Tick()
		m.saveSpinner.Tick()
		return m, m.tick()
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case pathPickedMsg:
		if msg.err == nil && msg.path != "" {
			m.setFieldValue(msg.field, msg.path)
			return m, m.saveOverridesBatch()
		}
	case messages.ConfigKeysLoadedMsg:
		m.isLoading = false
		m.SetConfigValues(msg.Values)
		if msg.Error != nil {
			m.statusMessage = "Discovery failed: " + msg.Error.Error()
			m.statusIsError = true
		}
		return m, nil
	case overridesSavedMsg:
		m.isSaving = false
		if msg.err != nil {
			m.statusMessage = "Save failed: " + msg.err.Error()
			m.statusIsError = true
			return m, nil
		}
		// Refresh discovery after save
		return m, tea.Batch(
			executor.LoadConfigSummary(m.repoRoot),
			executor.LoadConfigKeys(m.repoRoot, DefaultConfigKeys()),
		)
	case tea.KeyMsg:
		if m.editingText {
			return m.handleTextEdit(msg)
		}

		switch msg.String() {
		case "esc":
			m.Done = true
			return m, nil
		case "left", "h":
			m.sectionIndex = (m.sectionIndex - 1 + len(m.sections)) % len(m.sections)
			m.resetSectionState()
			return m, tea.ClearScreen
		case "right", "l":
			m.sectionIndex = (m.sectionIndex + 1) % len(m.sections)
			m.resetSectionState()
			return m, tea.ClearScreen
		case "up", "k":
			m.moveCursor(-1)
		case "down", "j":
			m.moveCursor(1)
		case "enter", " ":
			return m.activateSelection()
		case "b", "B":
			return m.browseCurrentPath()
		case "r", "R":
			return m.resetOverrides()
		}
	}
	return m, nil
}
