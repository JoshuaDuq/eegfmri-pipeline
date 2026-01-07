package globalsetup

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

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
	sectionBands
	sectionROIs
	sectionReview
)

type fieldKey int

const (
	fieldTask fieldKey = iota
	fieldRandomState
	fieldSubjectList
	fieldBidsRoot
	fieldDerivRoot
	fieldSourceRoot
	fieldEventTemp
	fieldEventRating
	fieldEventPain
	fieldBandEdit
	fieldRoiEdit
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

type BandEntry struct {
	Name string
	Low  float64
	High float64
}

type RoiEntry struct {
	Name     string
	Patterns []string
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

	task        string
	randomState string
	subjectList string
	bidsRoot    string
	derivRoot   string
	sourceRoot  string

	eventTemp   string
	eventRating string
	eventPain   string

	bands      []BandEntry
	rois       []RoiEntry
	bandCursor int
	roiCursor  int

	isLoading     bool
	isSaving      bool
	statusMessage string
	statusIsError bool

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
		"paths.deriv_root",
		"paths.source_data",
		"event_columns.temperature",
		"event_columns.rating",
		"event_columns.pain_binary",
		"time_frequency_analysis.bands",
		"time_frequency_analysis.rois",
	}
}

func New(repoRoot string) Model {
	overridesPath := filepath.Join(repoRoot, "eeg_pipeline", "data", "derivatives", ".tui_overrides.json")
	return Model{
		repoRoot:      repoRoot,
		overridesPath: overridesPath,
		sections: []sectionDef{
			{sectionProject, "Project", "Task, random seed, and subject filter"},
			{sectionPaths, "Paths", "BIDS and data roots"},
			{sectionEvents, "Events", "Behavior columns"},
			{sectionBands, "Bands", "Frequency bands"},
			{sectionROIs, "ROIs", "Region definitions"},
			{sectionReview, "Review", "Current configuration"},
		},
		isLoading: true,
	}
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
	if v, ok := values["paths.deriv_root"]; ok {
		m.derivRoot = toString(v, m.derivRoot)
	}
	if v, ok := values["paths.source_data"]; ok {
		m.sourceRoot = toString(v, m.sourceRoot)
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
	if v, ok := values["time_frequency_analysis.bands"]; ok {
		m.bands = parseBands(v)
	}
	if v, ok := values["time_frequency_analysis.rois"]; ok {
		m.rois = parseRois(v)
	}
}

func (m Model) Init() tea.Cmd {
	return nil
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
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
	case messages.ConfigLoadedMsg:
		// Also handle summary if it comes
		m.isLoading = false
		if msg.Error != nil {
			m.statusMessage = "Discovery failed"
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
			m.fieldCursor = 0
			m.statusMessage = ""
			m.statusIsError = false
			m.editingText = false
			m.textBuffer = ""
			m.editingField = 0
			return m, tea.ClearScreen
		case "right", "l":
			m.sectionIndex = (m.sectionIndex + 1) % len(m.sections)
			m.fieldCursor = 0
			m.statusMessage = ""
			m.statusIsError = false
			m.editingText = false
			m.textBuffer = ""
			m.editingField = 0
			return m, tea.ClearScreen
		case "up", "k":
			m.moveCursor(-1)
		case "down", "j":
			m.moveCursor(1)
		case "enter", " ":
			return m.activateSelection()
		case "a", "A":
			return m.addEntry()
		case "d", "D", "x":
			return m.deleteEntry()
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
	title := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("◆ GLOBAL SETUP")
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
	switch m.sections[m.sectionIndex].key {
	case sectionBands:
		mainContent.WriteString(m.renderBands())
	case sectionROIs:
		mainContent.WriteString(m.renderRois())
	case sectionReview:
		mainContent.WriteString(m.renderReview())
	default:
		mainContent.WriteString(m.renderFields())
	}

	if m.isLoading {
		mainContent.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render("  Searching for paths and configuration..."))
	}

	if m.statusMessage != "" {
		color := styles.Success
		if m.statusIsError {
			color = styles.Error
		}
		mainContent.WriteString("\n" + lipgloss.NewStyle().Foreground(color).Render(m.statusMessage))
	}

	if m.isSaving {
		mainContent.WriteString("\n" + lipgloss.NewStyle().Foreground(styles.TextDim).Render("  Saving changes..."))
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

	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, "  "))
}

func (m Model) renderFields() string {
	var b strings.Builder
	section := m.sections[m.sectionIndex]

	b.WriteString(styles.SectionTitleStyle.Render(" "+strings.ToUpper(section.label)+" ") + "\n\n")
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

func (m Model) renderBands() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" FREQUENCY BANDS ") + "\n\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"Define band name and range. Use A to add, D to delete.") + "\n\n")

	if len(m.bands) == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("No bands defined. Press A to add.") + "\n")
		return b.String()
	}

	for i, band := range m.bands {
		isFocused := i == m.bandCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}
		name := lipgloss.NewStyle().Foreground(styles.Text).Render(band.Name)
		if isFocused {
			name = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(band.Name)
		}
		rangeText := fmt.Sprintf("%s Hz", formatRange(band.Low, band.High))
		b.WriteString(cursor + name + "  " + lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(rangeText) + "\n")
	}
	return b.String()
}

func (m Model) renderRois() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" ROIS ") + "\n\n")

	helpText := "ROI name with regex list. Use A to add, D to delete."
	if m.editingText && m.editingField == fieldRoiEdit {
		helpText = "Editing ROI: name=pattern1 ; pattern2. Press Enter to save, Esc to cancel."
	}
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(helpText) + "\n\n")

	if len(m.rois) == 0 && !m.editingText {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Warning).Render("No ROIs defined. Press A to add.") + "\n")
		return b.String()
	}

	// Show editing input if currently editing
	if m.editingText && m.editingField == fieldRoiEdit {
		cursor := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		editPrompt := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("Editing: ")
		editText := m.textBuffer + "█"
		b.WriteString(cursor + editPrompt + lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(editText) + "\n\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Faint(true).Render("  Format: ROI_name=pattern1 ; pattern2 ; pattern3") + "\n")
		return b.String()
	}

	// Show ROI list
	for i, roi := range m.rois {
		isFocused := i == m.roiCursor
		cursor := "  "
		if isFocused {
			cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("> ")
		}
		name := roi.Name
		if name == "" {
			name = "(unnamed)"
		}
		if isFocused {
			name = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(name)
		} else {
			name = lipgloss.NewStyle().Foreground(styles.Text).Render(name)
		}
		patterns := strings.Join(roi.Patterns, " ; ")
		if patterns == "" {
			patterns = "(none)"
		}
		b.WriteString(cursor + name + "\n")
		b.WriteString("    " + lipgloss.NewStyle().Foreground(styles.TextDim).Render(patterns) + "\n")
	}
	return b.String()
}

func (m Model) renderReview() string {
	var b strings.Builder
	b.WriteString(styles.SectionTitleStyle.Render(" REVIEW CONFIG ") + "\n\n")
	b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true).Render(
		"Configuration is automatically saved. Use R to reset to defaults.") + "\n\n")

	task := m.task
	if task == "" {
		task = "(default)"
	}
	b.WriteString("Task: " + lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(task) + "\n")

	randomState := m.randomState
	if randomState == "" {
		randomState = "(default: 42)"
	}
	b.WriteString("Random State: " + lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(randomState) + "\n")

	subjectList := m.subjectList
	if subjectList == "" {
		subjectList = "(all subjects)"
	}
	b.WriteString("Subject List: " + lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(subjectList) + "\n")

	if m.bidsRoot != "" {
		b.WriteString("BIDS Root: " + m.bidsRoot + "\n")
	}
	if m.derivRoot != "" {
		b.WriteString("Deriv Root: " + m.derivRoot + "\n")
	}
	if m.sourceRoot != "" {
		b.WriteString("Source Root: " + m.sourceRoot + "\n")
	}

	b.WriteString("\n")
	b.WriteString("Bands: " + fmt.Sprintf("%d", len(m.bands)) + "\n")
	b.WriteString("ROIs: " + fmt.Sprintf("%d", len(m.rois)) + "\n")
	return b.String()
}

func (m *Model) moveCursor(delta int) {
	section := m.sections[m.sectionIndex].key
	switch section {
	case sectionBands:
		if len(m.bands) == 0 {
			return
		}
		m.bandCursor = (m.bandCursor + delta + len(m.bands)) % len(m.bands)
	case sectionROIs:
		if len(m.rois) == 0 {
			return
		}
		m.roiCursor = (m.roiCursor + delta + len(m.rois)) % len(m.rois)
	case sectionReview:
		return
	default:
		fields := m.sectionFields(section)
		if len(fields) == 0 {
			return
		}
		m.fieldCursor = (m.fieldCursor + delta + len(fields)) % len(fields)
	}
}

func (m *Model) activateSelection() (tea.Model, tea.Cmd) {
	section := m.sections[m.sectionIndex].key
	switch section {
	case sectionBands:
		return m.editBand()
	case sectionROIs:
		return m.editRoi()
	case sectionReview:
		return m, nil
	default:
		fields := m.sectionFields(section)
		if m.fieldCursor < 0 || m.fieldCursor >= len(fields) {
			return m, nil
		}
		field := fields[m.fieldCursor]
		m.startTextEdit(field.key)
		return m, nil
	}
}

func (m *Model) addEntry() (tea.Model, tea.Cmd) {
	switch m.sections[m.sectionIndex].key {
	case sectionBands:
		m.bands = append(m.bands, BandEntry{Name: "new_band", Low: 1.0, High: 4.0})
		m.bandCursor = len(m.bands) - 1
		model, cmd := m.editBand()
		return model, tea.Batch(cmd, m.saveOverridesBatch())
	case sectionROIs:
		m.rois = append(m.rois, RoiEntry{Name: "NewROI", Patterns: []string{}})
		m.roiCursor = len(m.rois) - 1
		model, cmd := m.editRoi()
		return model, tea.Batch(cmd, m.saveOverridesBatch())
	}
	return m, nil
}

func (m *Model) deleteEntry() (tea.Model, tea.Cmd) {
	switch m.sections[m.sectionIndex].key {
	case sectionBands:
		if len(m.bands) == 0 {
			return m, nil
		}
		idx := m.bandCursor
		m.bands = append(m.bands[:idx], m.bands[idx+1:]...)
		if m.bandCursor >= len(m.bands) {
			m.bandCursor = len(m.bands) - 1
		}
		return m, m.saveOverridesBatch()
	case sectionROIs:
		if len(m.rois) == 0 {
			return m, nil
		}
		idx := m.roiCursor
		m.rois = append(m.rois[:idx], m.rois[idx+1:]...)
		if m.roiCursor >= len(m.rois) {
			m.roiCursor = len(m.rois) - 1
		}
		return m, m.saveOverridesBatch()
	}
	return m, nil
}

func (m *Model) browseCurrentPath() (tea.Model, tea.Cmd) {
	section := m.sections[m.sectionIndex].key
	if section == sectionReview || section == sectionBands || section == sectionROIs {
		return m, nil
	}

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
		if runtime.GOOS != "darwin" {
			return pathPickedMsg{field: field, err: fmt.Errorf("file picker not supported on %s", runtime.GOOS)}
		}

		prompt := "Select folder"
		switch field {
		case fieldBidsRoot:
			prompt = "Select BIDS root folder"
		case fieldDerivRoot:
			prompt = "Select derivatives root folder"
		case fieldSourceRoot:
			prompt = "Select source data folder"
		}

		cmd := exec.Command("osascript", "-e", fmt.Sprintf(`POSIX path of (choose folder with prompt "%s")`, prompt))
		output, err := cmd.Output()
		if err != nil {
			return pathPickedMsg{field: field, err: err}
		}

		path := strings.TrimSpace(string(output))
		return pathPickedMsg{field: field, path: path, err: nil}
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
			{fieldDerivRoot, "Deriv Root", "Derivatives output", true},
			{fieldSourceRoot, "Source Root", "Raw source data", true},
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
	case fieldDerivRoot:
		return m.derivRoot
	case fieldSourceRoot:
		return m.sourceRoot
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
	case fieldDerivRoot:
		m.derivRoot = value
	case fieldSourceRoot:
		m.sourceRoot = value
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
	switch m.editingField {
	case fieldBandEdit:
		m.applyBandEdit(m.textBuffer)
	case fieldRoiEdit:
		m.applyRoiEdit(m.textBuffer)
	default:
		m.setFieldValue(m.editingField, m.textBuffer)
	}
	m.editingText = false
	m.textBuffer = ""
	m.editingField = 0
	return m.saveOverridesBatch()
}

func (m *Model) editBand() (tea.Model, tea.Cmd) {
	if len(m.bands) == 0 {
		return m, nil
	}
	band := m.bands[m.bandCursor]
	m.editingText = true
	m.editingField = fieldBandEdit
	m.textBuffer = fmt.Sprintf("%s=%.2f-%.2f", band.Name, band.Low, band.High)
	return m, nil
}

func (m *Model) editRoi() (tea.Model, tea.Cmd) {
	if len(m.rois) == 0 {
		return m, nil
	}
	roi := m.rois[m.roiCursor]
	patterns := strings.Join(roi.Patterns, " ; ")
	m.editingText = true
	m.editingField = fieldRoiEdit
	m.textBuffer = fmt.Sprintf("%s=%s", roi.Name, patterns)
	return m, nil
}

func (m *Model) applyBandEdit(value string) {
	value = strings.TrimSpace(value)
	if value == "" || len(m.bands) == 0 {
		return
	}

	name, low, high, ok := parseBandLine(value)
	if !ok {
		m.statusMessage = "Invalid band format. Use name=low-high"
		m.statusIsError = true
		return
	}
	m.bands[m.bandCursor] = BandEntry{Name: name, Low: low, High: high}
}

func (m *Model) applyRoiEdit(value string) {
	value = strings.TrimSpace(value)
	if value == "" || len(m.rois) == 0 {
		return
	}
	name, patterns := parseRoiLine(value)
	if name == "" {
		m.statusMessage = "Invalid ROI format. Use name=regex1 ; regex2"
		m.statusIsError = true
		return
	}
	m.rois[m.roiCursor] = RoiEntry{Name: name, Patterns: patterns}
}

func (m *Model) saveOverridesBatch() tea.Cmd {
	m.isSaving = true
	return func() tea.Msg {
		if err := m.validate(); err != nil {
			return overridesSavedMsg{err: err}
		}
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
	if strings.TrimSpace(m.derivRoot) != "" {
		paths["deriv_root"] = m.derivRoot
	}
	if strings.TrimSpace(m.sourceRoot) != "" {
		paths["source_data"] = m.sourceRoot
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

	tfr := map[string]interface{}{}
	if len(m.bands) > 0 {
		bands := map[string]interface{}{}
		for _, band := range m.bands {
			bands[band.Name] = []float64{band.Low, band.High}
		}
		tfr["bands"] = bands
	}
	if len(m.rois) > 0 {
		rois := map[string]interface{}{}
		for _, roi := range m.rois {
			if roi.Name == "" {
				continue
			}
			rois[roi.Name] = roi.Patterns
		}
		tfr["rois"] = rois
	}
	if len(tfr) > 0 {
		overrides["time_frequency_analysis"] = tfr
	}

	return overrides
}

func (m *Model) validate() error {
	for _, band := range m.bands {
		if band.Name == "" || band.Low >= band.High {
			return fmt.Errorf("band ranges must be valid (low < high)")
		}
	}
	for _, roi := range m.rois {
		if roi.Name == "" {
			return fmt.Errorf("ROI names cannot be empty")
		}
	}
	return nil
}

func parseBandLine(value string) (string, float64, float64, bool) {
	if strings.Contains(value, "=") {
		parts := strings.SplitN(value, "=", 2)
		name := strings.TrimSpace(parts[0])
		rangePart := strings.TrimSpace(parts[1])
		if name == "" || rangePart == "" {
			return "", 0, 0, false
		}
		if strings.Contains(rangePart, "-") {
			rparts := strings.SplitN(rangePart, "-", 2)
			low, err1 := strconv.ParseFloat(strings.TrimSpace(rparts[0]), 64)
			high, err2 := strconv.ParseFloat(strings.TrimSpace(rparts[1]), 64)
			if err1 == nil && err2 == nil {
				return name, low, high, true
			}
		}
	}
	return "", 0, 0, false
}

func parseRoiLine(value string) (string, []string) {
	parts := strings.SplitN(value, "=", 2)
	if len(parts) < 2 {
		return "", nil
	}
	name := strings.TrimSpace(parts[0])
	if name == "" {
		return "", nil
	}
	patterns := splitList(parts[1])
	return name, patterns
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}

func formatRange(low, high float64) string {
	return fmt.Sprintf("%s-%s", formatFloat(low), formatFloat(high))
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

func toFloat(value interface{}, fallback float64) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case string:
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			return parsed
		}
	}
	return fallback
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

func parseBands(value interface{}) []BandEntry {
	raw, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	var bands []BandEntry
	for name, entry := range raw {
		switch v := entry.(type) {
		case []interface{}:
			if len(v) >= 2 {
				low := toFloat(v[0], 0)
				high := toFloat(v[1], 0)
				bands = append(bands, BandEntry{Name: name, Low: low, High: high})
			}
		}
	}
	sort.Slice(bands, func(i, j int) bool {
		return bands[i].Name < bands[j].Name
	})
	return bands
}

func parseRois(value interface{}) []RoiEntry {
	raw, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	var rois []RoiEntry
	for name, entry := range raw {
		rois = append(rois, RoiEntry{Name: name, Patterns: toStringList(entry)})
	}
	sort.Slice(rois, func(i, j int) bool {
		return rois[i].Name < rois[j].Name
	})
	return rois
}

func pathExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
