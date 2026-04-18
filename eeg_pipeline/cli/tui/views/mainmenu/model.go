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
	mainMenuWideThreshold     = 118
	mainMenuTallNarrowWidth   = 92
	mainMenuTallNarrowHeight  = 28
	mainMenuSplitMinWidth     = 76
	mainMenuSplitMinHeight    = 26
	mainMenuColumnGap         = 2
	mainMenuPreviewMinWidth   = 42
	mainMenuPreviewLabelWidth = 10
	mainMenuCompactDetailMin  = 15
	mainMenuCompactMenuMin    = 6
	mainMenuCompactDetailRows = 6
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

	Task    string
	version string

	configSummary HomeConfigSummary
	recentRuns    []RecentRunSummary

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

func (m *Model) SetRecentRuns(runs []RecentRunSummary) {
	m.recentRuns = append([]RecentRunSummary(nil), runs...)
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

	// Quiet, single-line brand treatment. No glyph — the name + thin rule beneath
	// is enough visual anchor for a research-app header.
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
	versionText := styles.SubtitleStyle.Render(versionLabel)

	left := "  " + logo + "  " + versionText

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
	// Selection is signaled with a steady left-edge accent bar (no blink),
	// the item name in the primary accent color + bold, and a subtle dot
	// separator before the description. Unselected rows are aligned to the
	// same left column using a single space + gutter.
	sep := lipgloss.NewStyle().Foreground(styles.Border).Render(" · ")

	if selected {
		bar := styles.RenderAccentBar(true)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		descStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		var inner string
		if config.showDescriptions {
			inner = bar + " " + nameStyle.Render(name) + sep + descStyle.Render(description)
		} else {
			inner = bar + " " + nameStyle.Render(name)
		}
		return styles.TruncateLine(inner, config.width)
	}

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

func (m Model) renderFooter() string {
	hints := []styles.FooterHint{
		{Key: "↑↓", Label: "Navigate", Compact: "Nav", Priority: 0},
		{Key: "⏎", Label: "Open", Compact: "Open", Priority: 0},
		{Key: "R", Label: "Resume", Compact: "Resume", Priority: 1},
		{Key: "Q", Label: "Quit", Compact: "Quit", Priority: 1},
		{Key: "D", Label: "Dashboard", Compact: "Dash", Priority: 2},
		{Key: "H", Label: "History", Compact: "Hist", Priority: 2},
		{Key: "Ctrl+K", Label: "Quick Actions", Compact: "Quick", Priority: 2},
	}

	width := m.width - 4
	if width < 20 {
		width = 20
	}
	divider := styles.RenderDivider(width)
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
	if m.useWideLayout(width, height) {
		return m.renderWideContent(width, height)
	}
	return m.renderCompactContent(width, height)
}

func (m Model) useWideLayout(width, height int) bool {
	if width >= mainMenuWideThreshold {
		return true
	}
	return width >= mainMenuTallNarrowWidth && height >= mainMenuTallNarrowHeight
}

func (m Model) renderCompactContent(width, height int) string {
	compactWidth := max(width, 1)
	compactHeight := max(height, 1)
	if m.useSplitCompactLayout(compactWidth, compactHeight) {
		return m.renderSplitCompactContent(compactWidth, compactHeight)
	}

	innerWidth := max(compactWidth-4, 1)
	innerHeight := max(compactHeight-2, 1)
	style := styles.BoxStyle.Height(compactHeight)
	return styles.RenderNoWrapBlock(style, m.renderCompactBody(innerWidth, innerHeight), compactWidth)
}

func (m Model) useSplitCompactLayout(width, height int) bool {
	return width >= mainMenuSplitMinWidth && height >= mainMenuSplitMinHeight
}

func (m Model) renderSplitCompactContent(width, height int) string {
	menuPaneHeight, detailPaneHeight := m.compactPanelHeights(height)
	menuPaneStyle := styles.CardStyleFocused.Height(menuPaneHeight)
	menuPane := styles.RenderNoWrapBlock(menuPaneStyle, m.renderCompactMenuPane(width-6, menuPaneHeight-4), width)
	detailPaneStyle := styles.PanelStyle.Height(detailPaneHeight)
	detailPane := styles.RenderNoWrapBlock(detailPaneStyle, m.renderCompactDetailPane(width-6, detailPaneHeight-4), width)

	return lipgloss.JoinVertical(
		lipgloss.Left,
		menuPane,
		"",
		detailPane,
	)
}

func (m Model) compactPanelHeights(totalHeight int) (int, int) {
	detailPaneHeight := min(16, totalHeight*40/100)
	if detailPaneHeight < 12 {
		detailPaneHeight = 12
	}

	menuPaneHeight := totalHeight - detailPaneHeight - 1
	if menuPaneHeight < 13 {
		menuPaneHeight = 13
		detailPaneHeight = totalHeight - menuPaneHeight - 1
	}

	return menuPaneHeight, detailPaneHeight
}

func (m Model) renderCompactBody(innerWidth, innerHeight int) string {
	if innerHeight < mainMenuCompactDetailMin {
		return m.renderCompactMenuPane(innerWidth, innerHeight)
	}

	detailHeight := m.compactDetailHeight(innerHeight)
	if detailHeight < 4 {
		return m.renderCompactMenuPane(innerWidth, innerHeight)
	}

	menuHeight := innerHeight - detailHeight - 1
	if menuHeight < mainMenuCompactMenuMin {
		return m.renderCompactMenuPane(innerWidth, innerHeight)
	}

	menu := m.renderCompactMenuPane(innerWidth, menuHeight)
	detail := m.renderCompactDetailPane(innerWidth, detailHeight)
	return menu + "\n" + styles.RenderDivider(innerWidth) + "\n" + detail
}

func (m Model) compactDetailHeight(innerHeight int) int {
	extraRows := max(innerHeight-mainMenuCompactDetailMin, 0)
	detailHeight := mainMenuCompactDetailRows + extraRows/2
	return min(detailHeight, innerHeight-mainMenuCompactMenuMin-1)
}

func compactFocusContentRows(availableRows, focusAreaCount int) int {
	if focusAreaCount == 0 || availableRows <= 1 {
		return 0
	}

	focusRows := min(focusAreaCount, availableRows-1)
	return 1 + focusRows
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

func (m Model) renderMenuPane(innerWidth, innerHeight int) string {
	lines, selectedLine := m.buildMenuLines(menuPaneConfig{
		width:            innerWidth,
		showDescriptions: true,
		showTitle:        true,
		showSubtitle:     false,
		showDividers:     true,
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
	lines = append(lines, m.renderSectionHeaderWithCount(title, m.currentSection == sectionID, cursor, len(items)))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderItem(item.name, item.description, isSelected, config))
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
	lines = append(lines, m.renderSectionHeaderWithCount(title, m.currentSection == sectionID, cursor, len(items)))
	for idx, item := range items {
		isSelected := m.currentSection == sectionID && idx == cursor
		if isSelected {
			*selectedLine = len(lines)
		}
		lines = append(lines, m.renderItem(item.name, item.description, isSelected, config))
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

func (m Model) renderCompactDetailPane(width, maxLines int) string {
	detail := m.selectedDetail()
	titleStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	descriptionStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	focusStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	bulletStyle := lipgloss.NewStyle().Foreground(styles.Accent)

	lines := []string{
		styles.RenderPreviewSubHeaderWithRule("DETAILS", width),
		styles.TruncateLine(titleStyle.Render(detail.title), width),
		styles.TruncateLine(descriptionStyle.Render(detail.description), width),
	}

	remainingRows := maxLines - len(lines)
	minimumRows := len(detail.rows) +
		compactFocusContentRows(max(remainingRows-len(detail.rows), 0), len(detail.focusAreas))
	if remainingRows > minimumRows+1 {
		lines = append(lines, "")
		remainingRows--
	}

	for _, row := range detail.rows {
		if remainingRows <= 0 {
			break
		}

		line := styles.RenderKeyValue(row.label, row.value, mainMenuPreviewLabelWidth)
		if row.accent {
			line = styles.RenderKeyValueAccent(row.label, row.value, mainMenuPreviewLabelWidth)
		}

		lines = append(lines, styles.TruncateLine(line, width))
		remainingRows--
	}

	focusRows := min(len(detail.focusAreas), max(remainingRows-1, 0))
	if focusRows > 0 && remainingRows > focusRows+1 {
		lines = append(lines, "")
		remainingRows--
	}
	if focusRows > 0 {
		lines = append(lines, styles.RenderPreviewSubHeaderWithRule("FOCUS", width))
		remainingRows--
	}
	for _, focus := range detail.focusAreas {
		if remainingRows <= 0 {
			break
		}

		line := bulletStyle.Render(styles.BulletMark) + " " + focusStyle.Render(focus)
		lines = append(lines, styles.TruncateLine(line, width))
		remainingRows--
	}

	return strings.Join(lines, "\n")
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
	recentLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(mainMenuPreviewLabelWidth).Render("Recent")
	blankLabel := strings.Repeat(" ", mainMenuPreviewLabelWidth)
	for i, run := range m.recentRuns {
		b.WriteString("\n")
		if i == 0 {
			b.WriteString(styles.TruncateLine(recentLabel+m.renderRecentRunLine(run), width))
		} else {
			b.WriteString(styles.TruncateLine(blankLabel+m.renderRecentRunLine(run), width))
		}
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

func (m Model) renderRecentRunLine(run RecentRunSummary) string {
	var statusIcon string
	if run.Success {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render(styles.CheckMark)
	} else {
		statusIcon = lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render(styles.CrossMark)
	}

	pipelineStyle := lipgloss.NewStyle().Foreground(styles.Text).Bold(true)
	parts := []string{
		statusIcon,
		pipelineStyle.Render(run.Pipeline),
	}
	if run.Mode != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.TextDim).Render(run.Mode))
	}
	if run.Age != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.Muted).Render(run.Age))
	}
	if run.Duration != "" {
		parts = append(parts, lipgloss.NewStyle().Foreground(styles.Muted).Render(run.Duration))
	}

	sep := lipgloss.NewStyle().Foreground(styles.Border).Render(" · ")
	return strings.Join(parts, sep)
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
