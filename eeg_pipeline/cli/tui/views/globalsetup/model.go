package globalsetup

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type sectionKey int

const (
	sectionProject sectionKey = iota
	sectionPaths
	sectionEvents
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
	fieldEventTemp
	fieldEventRating
	fieldEventPain
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

	eventTemp   string
	eventRating string
	eventPain   string

	isLoading     bool
	isSaving      bool
	statusMessage string
	statusIsError bool

	animQueue   animation.Queue
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
		"event_columns.temperature",
		"event_columns.rating",
		"event_columns.pain_binary",
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
			{sectionEvents, "Events", "Behavior columns"},
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
	if v, ok := values["event_columns.temperature"]; ok {
		m.eventTemp = strings.Join(toStringList(v), ", ")
	}
	if v, ok := values["event_columns.rating"]; ok {
		m.eventRating = strings.Join(toStringList(v), ", ")
	}
	if v, ok := values["event_columns.pain_binary"]; ok {
		m.eventPain = strings.Join(toStringList(v), ", ")
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

func (m Model) View() string {
	// Render header
	title := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("Global setup")
	section := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.sections[m.sectionIndex].label)
	header := title + "  " + section
	headerHeight := 3

	// Render footer
	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

	// Calculate available height for main content
	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	// Build main content
	var mainContent strings.Builder
	mainContent.WriteString(m.renderFields())

	if m.isLoading {
		mainContent.WriteString("\n  " + m.searchSpinner.View())
	}

	if m.statusMessage != "" {
		color := styles.Success
		if m.statusIsError {
			color = styles.Error
		}
		mainContent.WriteString("\n" + lipgloss.NewStyle().Foreground(color).Render(m.statusMessage))
	}

	if m.isSaving {
		mainContent.WriteString("\n  " + m.saveSpinner.View())
	}

	// Force main content to fill available height
	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent.String())

	return header + "\n\n" + mainContentStyled + "\n" + footer
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑/↓", "Navigate"),
		styles.RenderKeyHint("←/→", "Section"),
		styles.RenderKeyHint("Enter", "Edit"),
		styles.RenderKeyHint("B", "Browse"),
		styles.RenderKeyHint("R", "Reset"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	if m.editingText {
		hints = []string{
			styles.RenderKeyHint("Type", "Edit"),
			styles.RenderKeyHint("Enter", "Save"),
			styles.RenderKeyHint("Esc", "Cancel"),
		}
	}

	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, styles.RenderFooterSeparator()))
}

func (m Model) renderFields() string {
	var b strings.Builder
	section := m.sections[m.sectionIndex]

	b.WriteString(styles.SectionTitleStyle.Render(" "+section.label+" ") + "\n\n")
	if section.description != "" {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(section.description) + "\n\n")
	}

	fields := m.sectionFields(section.key)
	for i, field := range fields {
		isFocused := i == m.fieldCursor
		labelStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(20)
		if isFocused {
			labelStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Width(20)
		}

		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}

		value := m.fieldValue(field.key)
		if m.editingText && m.editingField == field.key {
			value = m.textBuffer + "█"
		}
		if value == "" {
			value = "(not set)"
		}

		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		line := cursor + labelStyle.Render(field.label+":") + " " + valueStyle.Render(value)

		if field.isPath && value != "(not set)" {
			path := m.fieldValue(field.key)
			if pathExists(path) {
				line += lipgloss.NewStyle().Foreground(styles.Success).Render("  ✓")
			} else {
				line += lipgloss.NewStyle().Foreground(styles.Warning).Render("  ⚠ (Not found! Please check or enter manually)")
			}
		}

		if field.description != "" {
			line += lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  " + field.description)
		}

		if field.isPath {
			line += lipgloss.NewStyle().Foreground(styles.Muted).Render("  (Press B to browse)")
		}

		b.WriteString(line + "\n")
	}

	return b.String()
}

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
