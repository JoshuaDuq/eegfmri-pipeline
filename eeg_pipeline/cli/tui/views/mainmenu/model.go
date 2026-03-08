package mainmenu

import (
	"fmt"
	"path/filepath"
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
	mainMenuWideThreshold            = 118
	mainMenuCompactWidth             = 84
	mainMenuCompactHeight            = 22
	mainMenuStandardTopHeight        = 7
	mainMenuStandardSummaryMinHeight = 28
	mainMenuColumnGap                = 2
	mainMenuPreviewMinWidth          = 42
	mainMenuPreviewLabelWidth        = 10
)

type HomeConfigSummary struct {
	Task               string
	BidsRoot           string
	BidsFmriRoot       string
	DerivRoot          string
	SourceRoot         string
	PreprocessingNJobs int
}

type RecentRunSummary struct {
	Pipeline string
	Mode     string
	Age      string
	Duration string
	Success  bool
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

	Task string

	configSummary     HomeConfigSummary
	recentRuns        []RecentRunSummary
	savedConfigCounts map[int]int

	// Toast notification
	toast components.Toast

	// Animation
	ticker    int
	animQueue animation.Queue
}

func New() Model {
	m := Model{
		currentSection:    SectionPreprocessing,
		prepCursor:        0,
		analysisCursor:    0,
		utilityCursor:     0,
		lastPipelineIdx:   -1,
		SelectedPipeline:  -1,
		SelectedUtility:   -1,
		Task:              "task",
		savedConfigCounts: make(map[int]int),
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

func (m *Model) SetRecentRuns(runs []RecentRunSummary) {
	m.recentRuns = append([]RecentRunSummary(nil), runs...)
}

func (m *Model) SetSavedConfigCounts(counts map[int]int) {
	m.savedConfigCounts = make(map[int]int, len(counts))
	for pipelineIdx, count := range counts {
		m.savedConfigCounts[pipelineIdx] = count
	}
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
		m.ticker++
		m.animQueue.Tick()
		m.toast.Tick()
		return m, m.tick()

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

func (m Model) handleResumeLastSession() (tea.Model, tea.Cmd) {
	if !m.hasLastPipeline() {
		m.toast = components.NewToast("No saved session yet", components.ToastWarning, 24)
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
	headerHeight := strings.Count(header, "\n") + 2

	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

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

	glyph := lipgloss.NewStyle().Foreground(styles.Primary).Render("◆")
	logo := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render("eegfmri-pipeline")
	version := lipgloss.NewStyle().Foreground(styles.Muted).Render("v1.0")
	titleRow := "  " + glyph + " " + logo + "  " + version

	return titleRow + "\n" + styles.RenderHeaderSeparator(lineWidth)
}

func (m Model) renderSectionHeader(title string, isActive bool) string {
	if isActive {
		return styles.RenderActiveSectionLabel(title)
	}
	return styles.RenderDimSectionLabel(title)
}

type sectionRenderConfig struct {
	width            int
	showDescriptions bool
}

type menuPaneConfig struct {
	width            int
	showDescriptions bool
	showTitle        bool
	showSubtitle     bool
	showDividers     bool
}

func (m Model) renderItem(name, description string, selected bool, config sectionRenderConfig) string {
	if selected {
		cursor := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Background(styles.Highlight).Render(styles.SelectedMark + " ")
		nameStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Background(styles.Highlight)
		sepStyle := lipgloss.NewStyle().Foreground(styles.Muted).Background(styles.Highlight)
		descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Background(styles.Highlight)
		var inner string
		if config.showDescriptions {
			inner = cursor + nameStyle.Render(name) + sepStyle.Render(" "+styles.BulletMark+" ") + descStyle.Render(description)
		} else {
			inner = cursor + nameStyle.Render(name)
		}
		inner = styles.TruncateLine(inner, config.width)
		return lipgloss.NewStyle().Width(config.width).Background(styles.Highlight).Render(inner)
	}

	sep := lipgloss.NewStyle().Foreground(styles.Muted).Render(" " + styles.BulletMark + " ")
	nameStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	descStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	var rawLine string
	if config.showDescriptions {
		rawLine = "  " + nameStyle.Render(name) + sep + descStyle.Render(description)
	} else {
		rawLine = "  " + nameStyle.Render(name)
	}
	return styles.TruncateLine(rawLine, config.width)
}

func (m Model) renderPipelineItem(p pipelineItem, selected bool, config sectionRenderConfig) string {
	return m.renderItem(p.name, p.description, selected, config)
}

func (m Model) renderUtilityItem(u utilityItem, selected bool, config sectionRenderConfig) string {
	return m.renderItem(u.name, u.description, selected, config)
}

func (m Model) renderFooter() string {
	hints := []footerHint{
		{key: "↑↓", label: "Navigate", compact: "Nav", priority: 0},
		{key: "⏎", label: "Open", compact: "Open", priority: 0},
		{key: "R", label: "Resume", compact: "Resume", priority: 1},
		{key: "Q", label: "Quit", compact: "Quit", priority: 1},
		{key: "D", label: "Dashboard", compact: "Dash", priority: 2},
		{key: "H", label: "History", compact: "Hist", priority: 2},
	}

	width := m.width - 4
	if width < 20 {
		width = 20
	}
	divider := styles.RenderHeaderSeparator(width)
	bar := styles.FooterStyle.Width(width).Render(m.renderFooterHints(width, hints))
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

type footerHint struct {
	key      string
	label    string
	compact  string
	priority int
}

func (m Model) renderContent(width, height int) string {
	if width >= mainMenuWideThreshold {
		return m.renderWideContent(width, height)
	}
	if m.useCompactLayout(width, height) {
		return m.renderCompactContent(width, height)
	}
	return m.renderStandardContent(width, height)
}

func (m Model) useCompactLayout(width, height int) bool {
	return width < mainMenuCompactWidth || height < mainMenuCompactHeight
}

func (m Model) renderCompactContent(width, height int) string {
	compactWidth := max(width, 1)
	compactHeight := max(height, 1)
	return styles.BoxStyle.Width(compactWidth).Height(compactHeight).Render(
		m.renderCompactMenuPane(compactWidth-4, compactHeight-2),
	)
}

func (m Model) renderWideContent(width, height int) string {
	leftWidth := width * 48 / 100
	if leftWidth < 48 {
		leftWidth = 48
	}
	rightWidth := width - leftWidth - mainMenuColumnGap
	if rightWidth < mainMenuPreviewMinWidth {
		rightWidth = mainMenuPreviewMinWidth
		leftWidth = width - rightWidth - mainMenuColumnGap
	}

	menuPane := styles.CardStyleFocused.Width(leftWidth).Height(height).Render(m.renderMenuPane(leftWidth-6, height-4))
	previewPane := styles.PanelStyle.Width(rightWidth).Height(height).Render(m.renderPreviewPane(rightWidth - 6))

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		menuPane,
		strings.Repeat(" ", mainMenuColumnGap),
		previewPane,
	)
}

func (m Model) renderStandardContent(width, height int) string {
	if width < 20 {
		width = 20
	}
	if height < 12 {
		height = 12
	}
	if height < mainMenuStandardSummaryMinHeight {
		return styles.CardStyle.Width(width).Height(height).Render(m.renderStandardMenuPane(width-6, height-4))
	}

	topHeight := mainMenuStandardTopHeight
	menuHeight := height - topHeight - 1
	if menuHeight < 10 {
		return styles.CardStyle.Width(width).Height(height).Render(m.renderStandardMenuPane(width-6, height-4))
	}

	summaryRow := m.renderStandardSummaryRow(width, topHeight)
	menuPane := styles.CardStyle.Width(width).Height(menuHeight).Render(m.renderStandardMenuPane(width-6, menuHeight-4))

	return summaryRow + "\n" + menuPane
}

func (m Model) renderStandardSummaryRow(width, height int) string {
	leftWidth := width * 46 / 100
	if leftWidth < 36 {
		leftWidth = 36
	}
	rightWidth := width - leftWidth - mainMenuColumnGap
	if rightWidth < 36 {
		rightWidth = 36
		leftWidth = width - rightWidth - mainMenuColumnGap
	}

	sessionPane := styles.PanelStyle.Width(leftWidth).Height(height).Render(m.renderSessionSummaryBlock(leftWidth - 6))
	selectedPane := styles.PanelStyle.Width(rightWidth).Height(height).Render(m.renderSelectedSummaryBlock(rightWidth - 6))

	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		sessionPane,
		strings.Repeat(" ", mainMenuColumnGap),
		selectedPane,
	)
}

func (m Model) renderMenuPane(innerWidth, innerHeight int) string {
	lines, selectedLine := m.buildMenuLines(menuPaneConfig{
		width:            innerWidth,
		showDescriptions: true,
		showTitle:        true,
		showSubtitle:     false,
		showDividers:     false,
	})
	return m.renderMenuViewport(lines, innerHeight, selectedLine)
}

func (m Model) renderStandardMenuPane(innerWidth, innerHeight int) string {
	lines, selectedLine := m.buildMenuLines(menuPaneConfig{
		width:            innerWidth,
		showDescriptions: false,
		showTitle:        true,
		showSubtitle:     false,
		showDividers:     false,
	})
	return m.renderMenuViewport(lines, innerHeight, selectedLine)
}

func (m Model) renderCompactMenuPane(innerWidth, innerHeight int) string {
	lines, selectedLine := m.buildMenuLines(menuPaneConfig{
		width:            innerWidth,
		showDescriptions: innerWidth >= 44,
		showTitle:        false,
		showSubtitle:     false,
		showDividers:     false,
	})
	return m.renderMenuViewport(lines, innerHeight, selectedLine)
}

func (m Model) buildMenuLines(config menuPaneConfig) ([]string, int) {
	if config.width < 20 {
		config.width = 20
	}

	lines := make([]string, 0, 24)
	selectedLine := -1
	subtitleStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	itemConfig := sectionRenderConfig{
		width:            config.width,
		showDescriptions: config.showDescriptions,
	}

	if config.showTitle && config.showSubtitle {
		lines = append(lines, subtitleStyle.Render("Open a pipeline wizard or project utility."), "")
	}

	lines = m.appendPipelineSectionLines(
		lines,
		&selectedLine,
		"Preprocessing",
		SectionPreprocessing,
		m.prepCursor,
		preprocessingPipelines,
		itemConfig,
	)
	if config.showDividers {
		lines = append(lines, styles.RenderDivider(config.width), "")
	} else {
		lines = append(lines, "")
	}

	lines = m.appendPipelineSectionLines(
		lines,
		&selectedLine,
		"Analysis",
		SectionAnalysis,
		m.analysisCursor,
		analysisPipelines,
		itemConfig,
	)
	if config.showDividers {
		lines = append(lines, styles.RenderDivider(config.width), "")
	} else {
		lines = append(lines, "")
	}

	lines = m.appendUtilitySectionLines(
		lines,
		&selectedLine,
		"Utilities",
		SectionUtilities,
		m.utilityCursor,
		utilities,
		itemConfig,
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
	config sectionRenderConfig,
) []string {
	lines = append(lines, m.renderSectionHeader(title, m.currentSection == sectionID))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderPipelineItem(item, isSelected, config))
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
	config sectionRenderConfig,
) []string {
	lines = append(lines, m.renderSectionHeader(title, m.currentSection == sectionID))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderUtilityItem(item, isSelected, config))
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
		lastUsedMark := lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " last used")
		kindLabel += "  " + lastUsedMark
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
		return m.pipelineDetail(preprocessingPipelines[m.prepCursor], "Preprocessing")
	case SectionAnalysis:
		return m.pipelineDetail(analysisPipelines[m.analysisCursor], "Analysis")
	default:
		return m.utilityDetail(utilities[m.utilityCursor])
	}
}

func (m Model) pipelineDetail(item pipelineItem, group string) selectionDetail {
	pipeline := types.Pipeline(item.pipelineIdx)
	return selectionDetail{
		title:       item.name,
		description: item.description,
		kind:        "Pipeline",
		focusAreas:  item.focusAreas,
		lastUsed:    m.lastPipelineIdx == item.pipelineIdx,
		rows: []detailRow{
			{label: "Group", value: group},
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
				{label: "Group", value: "Utilities"},
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
			{label: "Group", value: "Utilities"},
			{label: "Scope", value: item.scope},
			{label: "Entry", value: item.command},
			{label: "Task", value: m.currentTaskLabel(), accent: m.hasConfiguredTask()},
		},
	}
}

func (m Model) hasConfiguredTask() bool {
	task := strings.TrimSpace(m.Task)
	return task != "" && task != "task"
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

func (m Model) savedConfigCount(pipelineIdx int) int {
	if m.savedConfigCounts == nil {
		return 0
	}
	return m.savedConfigCounts[pipelineIdx]
}

func (m Model) renderSessionSummaryBlock(width int) string {
	if width < 20 {
		width = 20
	}

	lines := []string{styles.RenderSectionLabel("Session")}
	lines = append(lines, styles.TruncateLine(m.renderLastSessionLine(), width))
	lines = append(lines, styles.TruncateLine(m.renderSessionStatusLine(), width))
	return strings.Join(lines, "\n")
}

func (m Model) renderSelectedSummaryBlock(width int) string {
	if width < 20 {
		width = 20
	}

	detail := m.selectedDetail()
	lines := []string{styles.RenderSectionLabel("Selected")}
	lines = append(lines, styles.TruncateLine(m.renderSelectedTitleLine(detail), width))
	lines = append(lines, styles.TruncateLine(m.renderSelectedMetaLine(detail), width))
	return strings.Join(lines, "\n")
}

func (m Model) renderLastSessionLine() string {
	if !m.hasLastPipeline() {
		return lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("No saved session yet.")
	}

	pipeline, _ := m.lastPipeline()
	line := lipgloss.NewStyle().Foreground(styles.TextDim).Render("Last ") +
		lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(pipeline.String())
	if count := m.savedConfigCount(m.lastPipelineIdx); count > 0 {
		line += lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("  %d saved", count))
	}
	return line
}

func (m Model) renderSessionStatusLine() string {
	taskLine := styles.RenderKeyValue("Task", m.currentTaskLabel(), 5)
	if m.hasConfiguredTask() {
		taskLine = styles.RenderKeyValueAccent("Task", m.currentTaskLabel(), 5)
	}
	if len(m.recentRuns) == 0 {
		return taskLine
	}

	recentLine := lipgloss.NewStyle().Foreground(styles.Muted).Render("Recent ") + m.renderRecentRunLine(m.recentRuns[0])
	return taskLine + "  " + styles.RenderFooterSeparator() + "  " + recentLine
}

func (m Model) renderSelectedTitleLine(detail selectionDetail) string {
	title := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(detail.title)
	kind := lipgloss.NewStyle().Foreground(styles.Muted).Render(detail.kind)
	line := title + "  " + kind
	if detail.lastUsed {
		lastUsed := lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark + " last used")
		line += "  " + lastUsed
	}
	return line
}

func (m Model) renderSelectedMetaLine(detail selectionDetail) string {
	var parts []string
	if value := m.detailValue(detail, "Command"); value != "" {
		parts = append(parts, value)
	} else if value := m.detailValue(detail, "Entry"); value != "" {
		parts = append(parts, value)
	}
	if value := m.detailValue(detail, "Scope"); value != "" {
		parts = append(parts, value)
	}
	if len(parts) == 0 {
		parts = append(parts, detail.description)
	}
	return strings.Join(parts, "  "+styles.RenderFooterSeparator()+"  ")
}

func (m Model) detailValue(detail selectionDetail, label string) string {
	for _, row := range detail.rows {
		if row.label == label {
			return row.value
		}
	}
	return ""
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
		if row.label == "Group" {
			continue
		}
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
	if len(rows) == 0 && len(m.recentRuns) == 0 {
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
	if len(m.recentRuns) > 0 {
		b.WriteString("\n")
		label := lipgloss.NewStyle().Foreground(styles.TextDim).Width(mainMenuPreviewLabelWidth).Render("Recent")
		b.WriteString(styles.TruncateLine(label+m.renderRecentRunLine(m.recentRuns[0]), width))
	}
	return b.String()
}

func (m Model) renderPreviewFocusBlock(detail selectionDetail, width int) string {
	if len(detail.focusAreas) == 0 {
		return ""
	}

	bodyStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(width)
	bulletStyle := lipgloss.NewStyle().Foreground(styles.Accent)

	var b strings.Builder
	b.WriteString(styles.RenderPreviewSubHeaderWithRule("FOCUS", width))
	b.WriteString("\n")
	for _, focus := range detail.focusAreas {
		b.WriteString("\n")
		b.WriteString(bulletStyle.Render(styles.BulletMark))
		b.WriteString(" ")
		b.WriteString(bodyStyle.Render(focus))
	}
	return b.String()
}

func (m Model) renderRecentRunLine(run RecentRunSummary) string {
	statusIcon := lipgloss.NewStyle().Foreground(styles.Success).Render(styles.CheckMark)
	if !run.Success {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark)
	}

	parts := []string{
		statusIcon,
		lipgloss.NewStyle().Foreground(styles.Text).Render(run.Pipeline),
	}
	if run.Mode != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.TextDim).Render(run.Mode))
	}
	if run.Duration != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.Muted).Render(run.Duration))
	}
	if run.Age != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.Muted).Render(run.Age))
	}

	return strings.Join(parts, "  ")
}

func (m Model) shortPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}

	clean := filepath.Clean(path)
	parent := filepath.Base(filepath.Dir(clean))
	base := filepath.Base(clean)
	if parent == "." || parent == string(filepath.Separator) || parent == base {
		return clean
	}
	return filepath.Join("...", parent, base)
}

func (m Model) renderFooterHints(width int, hints []footerHint) string {
	render := func(useCompact bool, maxPriority int) string {
		parts := make([]string, 0, len(hints))
		for _, hint := range hints {
			if hint.priority > maxPriority {
				continue
			}
			label := hint.label
			if useCompact && hint.compact != "" {
				label = hint.compact
			}
			if hint.priority == 0 {
				parts = append(parts, styles.RenderKeyHint(hint.key, label))
			} else {
				parts = append(parts, styles.RenderKeyHintSecondary(hint.key, label))
			}
		}
		return strings.Join(parts, styles.RenderFooterSeparator())
	}

	if full := render(false, 2); lipgloss.Width(full) <= width {
		return full
	}
	if compact := render(true, 2); lipgloss.Width(compact) <= width {
		return compact
	}
	for maxPriority := 1; maxPriority >= 0; maxPriority-- {
		if compact := render(true, maxPriority); compact != "" && lipgloss.Width(compact) <= width {
			return compact
		}
	}
	return render(true, 0)
}

///////////////////////////////////////////////////////////////////
