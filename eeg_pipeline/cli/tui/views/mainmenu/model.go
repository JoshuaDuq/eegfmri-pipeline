package mainmenu

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// Pipeline Definition
///////////////////////////////////////////////////////////////////

type pipelineItem struct {
	name        string
	description string
	pipelineIdx int // index into types.Pipeline
	focusAreas  []string
}

// Preprocessing pipelines (EEG and fMRI preprocessing)
var preprocessingPipelines = []pipelineItem{
	{
		name:        "EEG Preprocessing",
		description: "Bad channels, ICA, epochs",
		pipelineIdx: int(types.PipelinePreprocessing),
		focusAreas: []string{
			"Select subjects and preprocessing stages",
			"Adjust filtering, ICA, and epoch parameters",
			"Resume saved selections when reopening the wizard",
		},
	},
	{
		name:        "fMRI Preprocessing",
		description: "Preprocess fMRI (fMRIPrep-style)",
		pipelineIdx: int(types.PipelineFmri),
		focusAreas: []string{
			"Configure container runtime, outputs, and performance limits",
			"Set anatomical, BOLD, QC, and reproducibility options",
			"Resume saved selections when reopening the wizard",
		},
	},
}

// Analysis pipelines
var analysisPipelines = []pipelineItem{
	{
		name:        "Features",
		description: "Extract EEG feature sets",
		pipelineIdx: int(types.PipelineFeatures),
		focusAreas: []string{
			"Choose feature families, bands, ROIs, and spatial aggregation",
			"Define time windows before building commands",
			"Resume saved selections when reopening the wizard",
		},
	},
	{
		name:        "Behavior",
		description: "EEG-behavior analysis",
		pipelineIdx: int(types.PipelineBehavior),
		focusAreas: []string{
			"Select computations and compatible feature inputs",
			"Scope subjects before entering advanced options",
			"Resume saved selections when reopening the wizard",
		},
	},
	{
		name:        "Machine Learning",
		description: "LOSO regression & classification",
		pipelineIdx: int(types.PipelineML),
		focusAreas: []string{
			"Choose regression or classification workflows",
			"Filter feature sets before tuning model options",
			"Resume saved selections when reopening the wizard",
		},
	},
	{
		name:        "fMRI Analysis",
		description: "First-level contrasts + trial-wise signatures",
		pipelineIdx: int(types.PipelineFmriAnalysis),
		focusAreas: []string{
			"Switch between first-level, second-level, and trial-signature modes",
			"Configure contrasts, confounds, and report outputs",
			"Resume saved selections when reopening the wizard",
		},
	},
}

type utilityItem struct {
	name        string
	description string
	scope       string
	command     string
	focusAreas  []string
	pipelineIdx int
}

const (
	UtilityGlobalSetup = iota
	UtilityPlotting
	UtilityPipelineSmokeTest
)

var utilities = []utilityItem{
	{
		name:        "Global Setup",
		description: "Configure project paths and settings",
		scope:       "Project configuration",
		command:     "Overrides editor",
		focusAreas: []string{
			"Set task, derivatives, BIDS, and source-data paths",
			"Persist overrides for the rest of the TUI",
		},
		pipelineIdx: -1,
	},
	{
		name:        "Plotting",
		description: "Curate and export visualization suites",
		scope:       "All derived outputs",
		command:     "eeg-pipeline plotting",
		focusAreas: []string{
			"Choose plot categories, plotters, and output formats",
			"Adjust global styling and per-plot overrides",
			"Resume saved selections when reopening the wizard",
		},
		pipelineIdx: int(types.PipelinePlotting),
	},
	{
		name:        "Pipeline Smoke Test",
		description: "Run quick parser/runtime checks across pipeline commands",
		scope:       "CLI entrypoints",
		command:     "scripts/tui_pipeline_smoke.py",
		focusAreas: []string{
			"Verify parser wiring for major CLI command families",
			"Run lightweight runtime checks without requiring study data",
		},
		pipelineIdx: -1,
	},
}

///////////////////////////////////////////////////////////////////
// Section Constants
///////////////////////////////////////////////////////////////////

const (
	SectionPreprocessing = iota
	SectionAnalysis
	SectionUtilities
)

const (
	mainMenuColumnGap         = 2
	mainMenuMenuMinWidth      = 20
	mainMenuPreviewMinWidth   = 24
	mainMenuPreviewLabelWidth = 10
)

type HomeConfigSummary struct {
	Task               string
	BidsRoot           string
	BidsFmriRoot       string
	DerivRoot          string
	SourceRoot         string
	PreprocessingNJobs int
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	currentSection   int // SectionPreprocessing, SectionAnalysis, or SectionUtilities
	prepCursor       int // cursor within preprocessing section
	analysisCursor   int // cursor within analysis section
	utilityCursor    int // cursor within utilities section
	lastPipelineIdx  int
	SelectedPipeline int
	SelectedUtility  int // -1 means none selected
	width            int
	height           int

	Task    string
	version string

	configSummary HomeConfigSummary

	// Toast notification
	toast components.Toast

	// Animation
	animQueue animation.Queue
}

func New() Model {
	m := Model{
		currentSection:   SectionPreprocessing,
		lastPipelineIdx:  -1,
		SelectedPipeline: -1,
		SelectedUtility:  -1,
	}
	m.animQueue.Push(animation.CursorBlinkLoop())
	return m
}

// SetCursor sets the pipeline cursor position (for restoring last selected pipeline)
func (m *Model) SetCursor(idx int) {
	m.lastPipelineIdx = idx

	// Find which section this pipeline belongs to
	for i, p := range preprocessingPipelines {
		if p.pipelineIdx == idx {
			m.currentSection = SectionPreprocessing
			m.prepCursor = i
			return
		}
	}
	for i, p := range analysisPipelines {
		if p.pipelineIdx == idx {
			m.currentSection = SectionAnalysis
			m.analysisCursor = i
			return
		}
	}
	if idx == int(types.PipelinePlotting) {
		m.currentSection = SectionUtilities
		m.utilityCursor = UtilityPlotting
		return
	}
}

func (m *Model) SetLastPipeline(idx int) {
	m.lastPipelineIdx = idx
}

func (m *Model) SetConfigSummary(summary HomeConfigSummary) {
	m.configSummary = summary
	if task := strings.TrimSpace(summary.Task); task != "" {
		m.Task = task
	}
}

func (m *Model) SetVersion(v string) {
	m.version = v
}

func (m *Model) SetToast(t components.Toast) {
	m.toast = t
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

type tickMsg struct{}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.animQueue.Tick()
		m.toast.Tick()
		return m, m.tick()

	case tea.MouseMsg:
		return m.handleMouse(msg)

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			m.handleUp()
		case "down", "j":
			m.handleDown()
		case "r":
			return m.handleResumeLastSession()
		case "enter", " ":
			return m.handleEnter()
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	}

	return m, nil
}

func (m Model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	switch msg.Button {
	case tea.MouseButtonWheelUp:
		m.handleUp()
		return m, nil
	case tea.MouseButtonWheelDown:
		m.handleDown()
		return m, nil
	}

	if msg.Action != tea.MouseActionMotion && (msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft) {
		return m, nil
	}

	activate := msg.Action == tea.MouseActionPress && msg.Button == tea.MouseButtonLeft
	contentWidth := max(m.width-4, 1)

	leftWidth, _ := m.mainMenuColumnWidths(contentWidth)
	if msg.X >= leftWidth {
		return m, nil
	}

	line := m.viewLineAt(msg.Y)
	if line == "" {
		return m, nil
	}

	if idx := indexOfPipelineLine(line, preprocessingPipelines); idx >= 0 {
		m.currentSection = SectionPreprocessing
		m.prepCursor = idx
		if activate {
			return m.handleEnter()
		}
		return m, nil
	}
	if idx := indexOfPipelineLine(line, analysisPipelines); idx >= 0 {
		m.currentSection = SectionAnalysis
		m.analysisCursor = idx
		if activate {
			return m.handleEnter()
		}
		return m, nil
	}
	if idx := indexOfUtilityLine(line, utilities); idx >= 0 {
		m.currentSection = SectionUtilities
		m.utilityCursor = idx
		if activate {
			return m.handleEnter()
		}
		return m, nil
	}

	return m, nil
}

func (m Model) viewLineAt(y int) string {
	if y < 0 {
		return ""
	}
	lines := strings.Split(m.View(), "\n")
	if y >= len(lines) {
		return ""
	}
	return lines[y]
}

func indexOfPipelineLine(line string, items []pipelineItem) int {
	for i, item := range items {
		if strings.Contains(line, item.name) {
			return i
		}
	}
	return -1
}

func indexOfUtilityLine(line string, items []utilityItem) int {
	for i, item := range items {
		if strings.Contains(line, item.name) {
			return i
		}
	}
	return -1
}

func (m Model) handleResumeLastSession() (tea.Model, tea.Cmd) {
	if !m.hasLastPipeline() {
		m.toast = components.NewToast("No saved session yet", components.ToastWarning, components.ToastDurationMedium)
		return m, nil
	}
	m.SelectedPipeline = m.lastPipelineIdx
	return m, nil
}

func (m *Model) handleUp() {
	switch m.currentSection {
	case SectionPreprocessing:
		if m.prepCursor > 0 {
			m.prepCursor--
		} else {
			m.currentSection = SectionUtilities
			m.utilityCursor = len(utilities) - 1
		}
	case SectionAnalysis:
		if m.analysisCursor > 0 {
			m.analysisCursor--
		} else {
			m.currentSection = SectionPreprocessing
			m.prepCursor = len(preprocessingPipelines) - 1
		}
	case SectionUtilities:
		if m.utilityCursor > 0 {
			m.utilityCursor--
		} else {
			m.currentSection = SectionAnalysis
			m.analysisCursor = len(analysisPipelines) - 1
		}
	}
}

func (m *Model) handleDown() {
	switch m.currentSection {
	case SectionPreprocessing:
		if m.prepCursor < len(preprocessingPipelines)-1 {
			m.prepCursor++
		} else {
			m.currentSection = SectionAnalysis
			m.analysisCursor = 0
		}
	case SectionAnalysis:
		if m.analysisCursor < len(analysisPipelines)-1 {
			m.analysisCursor++
		} else {
			m.currentSection = SectionUtilities
			m.utilityCursor = 0
		}
	case SectionUtilities:
		if m.utilityCursor < len(utilities)-1 {
			m.utilityCursor++
		} else {
			m.currentSection = SectionPreprocessing
			m.prepCursor = 0
		}
	}
}

func (m Model) handleEnter() (tea.Model, tea.Cmd) {
	switch m.currentSection {
	case SectionPreprocessing:
		m.SelectedPipeline = preprocessingPipelines[m.prepCursor].pipelineIdx
	case SectionAnalysis:
		m.SelectedPipeline = analysisPipelines[m.analysisCursor].pipelineIdx
	case SectionUtilities:
		switch m.utilityCursor {
		case UtilityGlobalSetup:
			m.SelectedUtility = UtilityGlobalSetup
			return m, nil
		case UtilityPlotting:
			m.SelectedPipeline = int(types.PipelinePlotting)
			return m, nil
		case UtilityPipelineSmokeTest:
			m.SelectedUtility = UtilityPipelineSmokeTest
			return m, nil
		}
	}
	return m, nil
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	header := m.renderHeader()
	headerHeight := lipgloss.Height(header) + 1

	footer := m.renderFooter()
	footerHeight := lipgloss.Height(footer) + 1

	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	contentWidth := max(m.width-4, 1)
	content := m.renderContent(contentWidth, mainHeight)

	if m.toast.Visible {
		content += "\n" + m.toast.View()
	}

	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(content)

	return header + "\n" + mainContentStyled + "\n" + footer
}

func (m Model) renderHeader() string {
	lineWidth := m.width - 4
	if lineWidth < 0 {
		lineWidth = 0
	}

	// Quiet, single-line brand treatment. No glyph — the name + hairline rule
	// beneath is enough visual anchor for a research-app header. The product
	// mark uses a thin hyphen-separated lowercase wordmark, and the version
	// is rendered as muted metadata, separated by a dim · to read as a
	// structured "name · version" caption rather than two adjacent labels.
	logo := styles.TitleAccentStyle.Render("eegfmri-pipeline")

	v := m.version
	if v == "" {
		v = "dev"
	}
	// Prepend "v" only for release-looking versions (digit-leading); keep
	// placeholder labels like "dev" / "nightly" bare.
	versionLabel := v
	if len(v) > 0 && v[0] >= '0' && v[0] <= '9' {
		versionLabel = "v" + v
	}
	versionSep := lipgloss.NewStyle().Foreground(styles.Border).Render(" · ")
	versionText := styles.MutedTextStyle.Render(versionLabel)

	left := "  " + logo + versionSep + versionText

	right := ""
	if task := strings.TrimSpace(m.Task); task != "" {
		right = styles.RenderLabelValueInline("task", task) + "  "
	} else {
		right = styles.HintStyle.Render("task not configured") + "  "
	}

	spacer := ""
	if lineWidth > lipgloss.Width(left)+lipgloss.Width(right) {
		spacer = strings.Repeat(" ", lineWidth-lipgloss.Width(left)-lipgloss.Width(right))
	}
	titleRow := left + spacer + right

	return titleRow + "\n" + styles.RenderHeaderSeparator(lineWidth)
}

// renderSectionHeader renders a section label in plain uppercase with an
// accent-bar indicator. Active sections use the bright bar + text color;
// inactive sections use a thinner muted bar + muted text. This matches the
// "DETAILS / WORKSPACE / FOCUS" sub-header convention elsewhere in the app.
func (m Model) renderSectionHeader(title string, isActive bool) string {
	upper := strings.ToUpper(title)
	if isActive {
		bar := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(styles.SectionIconActive)
		heading := lipgloss.NewStyle().Foreground(styles.Text).Bold(true).Render(upper)
		return bar + " " + heading
	}
	dimBar := lipgloss.NewStyle().Foreground(styles.Secondary).Render(styles.SectionIcon)
	dimHeading := lipgloss.NewStyle().Foreground(styles.Muted).Bold(true).Render(upper)
	return dimBar + " " + dimHeading
}

// renderSectionHeaderWithCount renders a section header and, when the section
// is currently active, appends a subtle "N/Total" position counter.
func (m Model) renderSectionHeaderWithCount(title string, isActive bool, cursor, total int) string {
	base := m.renderSectionHeader(title, isActive)
	if !isActive || total <= 1 {
		return base
	}
	counter := styles.MutedTextStyle.Render(fmt.Sprintf("  %d/%d", cursor+1, total))
	return base + counter
}

func (m Model) renderItem(name string, selected bool, width int) string {
	// Menu rows now read as a compact index: name only, with state encoded by
	// a quiet left rail and weight. The detail pane carries the explanatory copy.
	if selected {
		line := styles.RenderAccentBar(true) + " " +
			lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(name)
		return styles.TruncateLine(line, width)
	}

	line := "  " + lipgloss.NewStyle().Foreground(styles.TextDim).Render(name)
	return styles.TruncateLine(line, width)
}

func (m Model) renderFooter() string {
	hints := []styles.FooterHint{
		{Key: "↑↓", Label: "Navigate", Compact: "Nav", Priority: 0},
		{Key: "⏎", Label: "Open", Compact: "Open", Priority: 0},
		{Key: "R", Label: "Resume", Compact: "Resume", Priority: 1},
		{Key: "Q", Label: "Quit", Compact: "Quit", Priority: 1},
		{Key: "D", Label: "Dashboard", Compact: "Dash", Priority: 2},
	}

	width := m.width - 4
	if width < 20 {
		width = 20
	}
	divider := styles.RenderFooterDivider(width)
	bar := styles.RenderNoWrapBlock(styles.FooterStyle, styles.RenderFooterHints(width, hints), width)
	return divider + "\n" + bar
}

type detailRow struct {
	label  string
	value  string
	accent bool
}

type selectionDetail struct {
	title       string
	description string
	kind        string
	focusAreas  []string
	lastUsed    bool
	rows        []detailRow
}

func (m Model) renderContent(width, height int) string {
	return m.renderWideContent(width, height)
}

func (m Model) renderWideContent(width, height int) string {
	leftWidth, rightWidth := m.mainMenuColumnWidths(width)

	menuPaneStyle := styles.CardStyleFocused.Height(height)
	menuPane := styles.RenderNoWrapBlock(menuPaneStyle, m.renderMenuPane(leftWidth-6, height-4), leftWidth)
	previewPaneStyle := styles.PanelStyle.Height(height)
	previewPane := styles.RenderNoWrapBlock(previewPaneStyle, m.renderPreviewPane(rightWidth-6), rightWidth)

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		menuPane,
		strings.Repeat(" ", mainMenuColumnGap),
		previewPane,
	)
}

func (m Model) mainMenuColumnWidths(contentWidth int) (int, int) {
	if contentWidth <= 0 {
		return 1, 1
	}

	if contentWidth < mainMenuMenuMinWidth+mainMenuPreviewMinWidth+mainMenuColumnGap {
		usable := max(contentWidth-mainMenuColumnGap, 2)
		leftWidth := usable * 48 / 100
		if leftWidth < 1 {
			leftWidth = 1
		}
		rightWidth := usable - leftWidth
		if rightWidth < 1 {
			rightWidth = 1
		}
		return leftWidth, rightWidth
	}

	leftWidth := contentWidth * 48 / 100
	if leftWidth < mainMenuMenuMinWidth {
		leftWidth = mainMenuMenuMinWidth
	}

	rightWidth := contentWidth - leftWidth - mainMenuColumnGap
	if rightWidth < mainMenuPreviewMinWidth {
		rightWidth = mainMenuPreviewMinWidth
		leftWidth = contentWidth - rightWidth - mainMenuColumnGap
	}

	if leftWidth < 1 {
		leftWidth = 1
	}
	if rightWidth < 1 {
		rightWidth = 1
	}
	return leftWidth, rightWidth
}

func (m Model) renderMenuPane(innerWidth, innerHeight int) string {
	lines, selectedLine := m.buildMenuLines(innerWidth)
	return m.renderMenuViewport(lines, innerHeight, selectedLine)
}

func (m Model) buildMenuLines(width int) ([]string, int) {
	if width < 20 {
		width = 20
	}

	lines := make([]string, 0, 24)
	selectedLine := -1

	lines = m.appendPipelineSectionLines(
		lines,
		&selectedLine,
		"Preprocessing",
		SectionPreprocessing,
		m.prepCursor,
		preprocessingPipelines,
		width,
	)
	lines = append(lines, "")

	lines = m.appendPipelineSectionLines(
		lines,
		&selectedLine,
		"Analysis",
		SectionAnalysis,
		m.analysisCursor,
		analysisPipelines,
		width,
	)
	lines = append(lines, "")

	lines = m.appendUtilitySectionLines(
		lines,
		&selectedLine,
		"Utilities",
		SectionUtilities,
		m.utilityCursor,
		utilities,
		width,
	)

	return lines, selectedLine
}

func (m Model) appendPipelineSectionLines(
	lines []string,
	selectedLine *int,
	title string,
	sectionID int,
	cursor int,
	items []pipelineItem,
	width int,
) []string {
	lines = append(lines, m.renderSectionHeaderWithCount(title, m.currentSection == sectionID, cursor, len(items)))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderItem(item.name, isSelected, width))
	}
	return lines
}

func (m Model) appendUtilitySectionLines(
	lines []string,
	selectedLine *int,
	title string,
	sectionID int,
	cursor int,
	items []utilityItem,
	width int,
) []string {
	lines = append(lines, m.renderSectionHeaderWithCount(title, m.currentSection == sectionID, cursor, len(items)))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderItem(item.name, isSelected, width))
	}
	return lines
}

func (m Model) renderMenuViewport(lines []string, innerHeight, selectedLine int) string {
	if len(lines) == 0 {
		return ""
	}
	if innerHeight <= 0 || len(lines) <= innerHeight {
		return strings.Join(lines, "\n")
	}
	if selectedLine < 0 || selectedLine >= len(lines) {
		selectedLine = 0
	}

	contentHeight := innerHeight - 2
	if contentHeight < 1 {
		contentHeight = 1
	}

	var layout styles.ListLayout
	for {
		layout = styles.CalculateListLayout(contentHeight, selectedLine, len(lines), 0)
		usedHeight := contentHeight
		if layout.ShowScrollUp {
			usedHeight++
		}
		if layout.ShowScrollDn {
			usedHeight++
		}
		if usedHeight >= innerHeight || contentHeight >= innerHeight {
			break
		}
		contentHeight++
	}

	visible := make([]string, 0, innerHeight)
	if layout.ShowScrollUp {
		visible = append(visible, styles.RenderScrollUpIndicator(layout.StartIdx))
	}
	visible = append(visible, lines[layout.StartIdx:layout.EndIdx]...)
	if layout.ShowScrollDn {
		visible = append(visible, styles.RenderScrollDownIndicator(len(lines)-layout.EndIdx))
	}

	return strings.Join(visible, "\n")
}

func (m Model) renderPreviewPane(innerWidth int) string {
	if innerWidth < 24 {
		innerWidth = 24
	}

	detail := m.selectedDetail()
	titleStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	descriptionStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

	kindLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(detail.kind)
	if detail.lastUsed {
		lastUsed := lipgloss.NewStyle().Foreground(styles.Success).Render("  " + styles.CheckMark + " last used")
		kindLabel += lastUsed
	}

	var b strings.Builder
	b.WriteString(titleStyle.Render(detail.title))
	b.WriteString("\n")
	b.WriteString(kindLabel)
	b.WriteString("\n")
	b.WriteString(descriptionStyle.Render(detail.description))
	if details := m.renderPreviewDetailsBlock(detail, innerWidth); details != "" {
		b.WriteString("\n\n")
		b.WriteString(details)
	}
	if workspace := m.renderPreviewWorkspaceBlock(innerWidth); workspace != "" {
		b.WriteString("\n\n")
		b.WriteString(workspace)
	}
	if focus := m.renderPreviewFocusBlock(detail, innerWidth); focus != "" {
		b.WriteString("\n\n")
		b.WriteString(focus)
	}

	return strings.TrimRight(b.String(), "\n")
}

func (m Model) selectedDetail() selectionDetail {
	switch m.currentSection {
	case SectionPreprocessing:
		return m.pipelineDetail(preprocessingPipelines[m.prepCursor])
	case SectionAnalysis:
		return m.pipelineDetail(analysisPipelines[m.analysisCursor])
	default:
		return m.utilityDetail(utilities[m.utilityCursor])
	}
}

func (m Model) pipelineDetail(item pipelineItem) selectionDetail {
	pipeline := types.Pipeline(item.pipelineIdx)
	return selectionDetail{
		title:       item.name,
		description: item.description,
		kind:        "Pipeline",
		focusAreas:  item.focusAreas,
		lastUsed:    m.lastPipelineIdx == item.pipelineIdx,
		rows: []detailRow{
			{label: "Source", value: pipeline.GetDataSource()},
			{label: "Command", value: "eeg-pipeline " + pipeline.CLICommand()},
			{label: "Task", value: m.currentTaskLabel(), accent: m.hasConfiguredTask()},
		},
	}
}

func (m Model) utilityDetail(item utilityItem) selectionDetail {
	if item.pipelineIdx >= 0 {
		pipeline := types.Pipeline(item.pipelineIdx)
		return selectionDetail{
			title:       item.name,
			description: item.description,
			kind:        "Utility",
			focusAreas:  item.focusAreas,
			lastUsed:    m.lastPipelineIdx == item.pipelineIdx,
			rows: []detailRow{
				{label: "Source", value: pipeline.GetDataSource()},
				{label: "Command", value: item.command},
				{label: "Task", value: m.currentTaskLabel(), accent: m.hasConfiguredTask()},
			},
		}
	}

	return selectionDetail{
		title:       item.name,
		description: item.description,
		kind:        "Utility",
		focusAreas:  item.focusAreas,
		rows: []detailRow{
			{label: "Scope", value: item.scope},
			{label: "Entry", value: item.command},
			{label: "Task", value: m.currentTaskLabel(), accent: m.hasConfiguredTask()},
		},
	}
}

func (m Model) hasConfiguredTask() bool {
	return strings.TrimSpace(m.Task) != ""
}

func (m Model) currentTaskLabel() string {
	if !m.hasConfiguredTask() {
		return "not configured"
	}
	return strings.TrimSpace(m.Task)
}

func (m Model) hasLastPipeline() bool {
	return m.lastPipelineIdx >= 0 && m.lastPipelineIdx <= int(types.PipelineFmriAnalysis)
}

func (m Model) lastPipeline() (types.Pipeline, bool) {
	if !m.hasLastPipeline() {
		return 0, false
	}
	return types.Pipeline(m.lastPipelineIdx), true
}

func (m Model) previewWorkspaceRows() []detailRow {
	var rows []detailRow
	if path := m.shortPath(m.configSummary.DerivRoot); path != "" {
		rows = append(rows, detailRow{label: "Deriv", value: path})
	}
	if path := m.shortPath(m.configSummary.BidsRoot); path != "" {
		rows = append(rows, detailRow{label: "BIDS", value: path})
	}
	if path := m.shortPath(m.configSummary.BidsFmriRoot); path != "" {
		rows = append(rows, detailRow{label: "fMRI", value: path})
	}
	if path := m.shortPath(m.configSummary.SourceRoot); path != "" {
		rows = append(rows, detailRow{label: "Source", value: path})
	}
	return rows
}

func (m Model) renderPreviewDetailsBlock(detail selectionDetail, width int) string {
	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("DETAILS", width))
	b.WriteString("\n")
	for _, row := range detail.rows {
		b.WriteString("\n")
		line := styles.RenderKeyValue(row.label, row.value, mainMenuPreviewLabelWidth)
		if row.accent {
			line = styles.RenderKeyValueAccent(row.label, row.value, mainMenuPreviewLabelWidth)
		}
		b.WriteString(styles.TruncateLine(line, width))
	}
	return b.String()
}

func (m Model) renderPreviewWorkspaceBlock(width int) string {
	rows := m.previewWorkspaceRows()
	if len(rows) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("WORKSPACE", width))
	b.WriteString("\n")
	for _, row := range rows {
		b.WriteString("\n")
		line := styles.RenderKeyValue(row.label, row.value, mainMenuPreviewLabelWidth)
		if row.accent {
			line = styles.RenderKeyValueAccent(row.label, row.value, mainMenuPreviewLabelWidth)
		}
		b.WriteString(styles.TruncateLine(line, width))
	}
	return b.String()
}

func (m Model) renderPreviewFocusBlock(detail selectionDetail, width int) string {
	if len(detail.focusAreas) == 0 {
		return ""
	}

	bodyStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	bulletStyle := lipgloss.NewStyle().Foreground(styles.Accent)

	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("FOCUS", width))
	b.WriteString("\n")
	for _, focus := range detail.focusAreas {
		b.WriteString("\n")
		bullet := bulletStyle.Render(styles.BulletMark) + " "
		b.WriteString(styles.TruncateLine(bullet+bodyStyle.Render(focus), width))
	}
	return b.String()
}

func (m Model) shortPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}

	home := homeDir()
	clean := filepath.Clean(path)
	if relative, ok := homeRelativePath(runtime.GOOS, home, clean); ok {
		if relative == "" {
			return "~"
		}
		return filepath.Join("~", relative)
	}
	parent := filepath.Base(filepath.Dir(clean))
	base := filepath.Base(clean)
	if parent == "." || parent == string(filepath.Separator) || parent == base {
		return clean
	}
	return filepath.Join("..", parent, base)
}

func homeDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return ""
}

func homeRelativePath(goos string, home string, path string) (string, bool) {
	if home == "" {
		return "", false
	}

	compareHome := filepath.Clean(home)
	comparePath := filepath.Clean(path)
	if goos == "windows" {
		compareHome = strings.ToLower(compareHome)
		comparePath = strings.ToLower(comparePath)
	}

	relative, err := filepath.Rel(compareHome, comparePath)
	if err != nil {
		return "", false
	}
	if relative == "." {
		return "", true
	}
	if relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", false
	}
	return relative, true
}

///////////////////////////////////////////////////////////////////
