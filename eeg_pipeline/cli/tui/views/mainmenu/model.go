package mainmenu

import (
	"fmt"
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
}

// Preprocessing pipelines (EEG and fMRI preprocessing)
var preprocessingPipelines = []pipelineItem{
	{"EEG Preprocessing", "Bad channels, ICA, epochs", 0},
	{"fMRI Preprocessing", "Preprocess fMRI (fMRIPrep-style)", 5},
}

// Analysis pipelines
var analysisPipelines = []pipelineItem{
	{"Features", "Extract EEG feature sets", 1},
	{"Behavior", "EEG-behavior analysis", 2},
	{"Machine Learning", "LOSO regression & classification", 3},
	{"Plotting", "Curate and export visualization suites", 4},
	{"fMRI Analysis", "First-level contrasts + trial-wise pain signatures", 9},
}

type utilityItem struct {
	name        string
	description string
}

const (
	UtilityGlobalSetup = iota
	UtilityMergePsychopy
	UtilityRawToBids
	UtilityFmriRawToBids
)

var utilities = []utilityItem{
	{"Global Setup", "Configure project paths and settings"},
	{"Merge PsychoPy Data", "Merge PsychoPy data into BIDS events files"},
	{"EEG Raw to BIDS", "Convert raw EEG data to BIDS format"},
	{"fMRI Raw to BIDS", "Convert raw fMRI DICOM series to BIDS format"},
}

///////////////////////////////////////////////////////////////////
// Section Constants
///////////////////////////////////////////////////////////////////

const (
	SectionPreprocessing = iota
	SectionAnalysis
	SectionUtilities
)

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	currentSection   int // SectionPreprocessing, SectionAnalysis, or SectionUtilities
	prepCursor       int // cursor within preprocessing section
	analysisCursor   int // cursor within analysis section
	utilityCursor    int // cursor within utilities section
	SelectedPipeline int
	SelectedUtility  int // -1 means none selected
	width            int
	height           int

	Task    string
	IsCloud bool

	// Help overlay
	helpOverlay components.HelpOverlay
	showHelp    bool

	// Toast notification
	toast components.Toast

	// Animation
	ticker    int
	animQueue animation.Queue
}

func New() Model {
	help := components.NewHelpOverlay("Keyboard Shortcuts", 50)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓ or j/k", Description: "Move cursor"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Select pipeline"},
		{Key: "?", Description: "Toggle help"},
	})
	help.AddSection("General", []components.HelpItem{
		{Key: "Esc", Description: "Go back"},
		{Key: "q", Description: "Quit application"},
	})

	m := Model{
		currentSection:   SectionPreprocessing,
		prepCursor:       0,
		analysisCursor:   0,
		utilityCursor:    0,
		SelectedPipeline: -1,
		SelectedUtility:  -1,
		Task:             "thermalactive",
		helpOverlay:      help,
	}
	m.animQueue.Push(animation.CursorBlinkLoop())
	return m
}

// SetCursor sets the pipeline cursor position (for restoring last selected pipeline)
func (m *Model) SetCursor(idx int) {
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
		// Handle help overlay first
		if m.showHelp {
			if msg.String() == "?" || msg.String() == "esc" {
				m.showHelp = false
				m.helpOverlay.Visible = false
			}
			return m, nil
		}

		switch msg.String() {
		case "?":
			m.showHelp = true
			m.helpOverlay.Visible = true
		case "up", "k":
			m.handleUp()
		case "down", "j":
			m.handleDown()
		case "enter", " ":
			return m.handleEnter()
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.helpOverlay.Width = min(50, m.width-10)
	}

	return m, nil
}

func (m Model) cursorPrefix(selected bool) string {
	if !selected {
		return "   "
	}
	return styles.RenderCursorOptional(m.animQueue.CursorVisible())
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
		case UtilityMergePsychopy:
			m.SelectedPipeline = int(types.PipelineMergePsychoPyData)
		case UtilityRawToBids:
			m.SelectedPipeline = int(types.PipelineRawToBIDS)
		case UtilityFmriRawToBids:
			m.SelectedPipeline = int(types.PipelineFmriRawToBIDS)
		}
	}
	return m, nil
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	// Help overlay takes precedence
	if m.showHelp {
		return m.renderWithOverlay(m.helpOverlay.View())
	}

	// Render header (fixed at top)
	header := m.renderHeader()
	headerHeight := strings.Count(header, "\n") + 3

	// Render footer (fixed at bottom)
	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

	// Calculate available height for main content
	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < 10 {
		mainHeight = 10
	}

	// Main Content - Three Section Layout
	prepCol := m.renderPreprocessingColumn()
	analysisCol := m.renderAnalysisColumn()
	utilitiesCol := m.renderUtilitiesColumn()

	leftContent := prepCol + "\n" + analysisCol + "\n" + utilitiesCol

	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(max(m.width-10, 52)).Render(leftContent),
	)

	mainContent := content
	if m.toast.Visible {
		mainContent += "\n\n" + m.toast.View()
	}

	// Force main content to fill available height
	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent)

	return header + "\n\n" + mainContentStyled + "\n" + footer
}

func (m Model) renderWithOverlay(overlay string) string {
	baseView := m.renderBaseView()
	overlayPlaced := lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		overlay,
	)
	return overlayPlaced + "\n" + lipgloss.NewStyle().Foreground(styles.Muted).Render(baseView)
}

func (m Model) renderBaseView() string {
	header := m.renderHeader()
	prepCol := m.renderPreprocessingColumn()
	analysisCol := m.renderAnalysisColumn()
	utilitiesCol := m.renderUtilitiesColumn()

	leftContent := prepCol + "\n" + analysisCol + "\n" + utilitiesCol
	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(max(m.width-10, 52)).Render(leftContent),
	)

	return header + "\n\n" + content
}

func (m Model) renderHeader() string {
	// EEG waveform motif
	waveform := lipgloss.NewStyle().Foreground(styles.Border).Render("~∿∿∿~")

	logo := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render(" EEG Pipeline")

	versionBadge := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Background(styles.Border).
		Padding(0, 1).
		Render("v1.0")

	var envBadge string
	if m.IsCloud {
		envBadge = styles.BadgeAccentStyle.Render(" CLOUD ")
	} else {
		envBadge = lipgloss.NewStyle().
			Foreground(styles.Success).
			Background(styles.Surface).
			Padding(0, 1).
			Render(styles.ActiveMark + " Local")
	}

	titleRow := waveform + logo + "  " + versionBadge + "  " + envBadge

	lineWidth := m.width - 4
	if lineWidth < 0 {
		lineWidth = 0
	}

	return lipgloss.JoinVertical(lipgloss.Left,
		titleRow,
		styles.RenderHeaderSeparator(lineWidth),
	)
}

func (m Model) renderSectionHeader(title string, isActive bool) string {
	if isActive {
		return styles.RenderSectionLabel(title)
	}
	return styles.RenderDimSectionLabel(title)
}

func (m Model) renderPreprocessingColumn() string {
	var lines []string
	lines = append(lines, m.renderSectionHeader("Preprocessing", m.currentSection == SectionPreprocessing))
	lines = append(lines, "")

	for i, p := range preprocessingPipelines {
		isSelected := m.currentSection == SectionPreprocessing && i == m.prepCursor
		lines = append(lines, m.renderPipelineItem(p, isSelected))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderAnalysisColumn() string {
	var lines []string
	lines = append(lines, m.renderSectionHeader("Analysis", m.currentSection == SectionAnalysis))
	lines = append(lines, "")

	for i, p := range analysisPipelines {
		isSelected := m.currentSection == SectionAnalysis && i == m.analysisCursor
		lines = append(lines, m.renderPipelineItem(p, isSelected))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderUtilitiesColumn() string {
	var lines []string
	lines = append(lines, m.renderSectionHeader("Utilities", m.currentSection == SectionUtilities))
	lines = append(lines, "")

	for i, u := range utilities {
		isSelected := m.currentSection == SectionUtilities && i == m.utilityCursor
		lines = append(lines, m.renderUtilityItem(u, isSelected))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderItem(name, description string, selected bool) string {
	var item strings.Builder

	cursor := m.cursorPrefix(selected)

	nameStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	descStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	if selected {
		nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
		descStyle = lipgloss.NewStyle().Foreground(styles.TextDim)
	}

	item.WriteString(fmt.Sprintf("%s%s\n", cursor, nameStyle.Render(name)))
	item.WriteString("      " + descStyle.Render(description) + "\n")

	return item.String()
}

func (m Model) renderPipelineItem(p pipelineItem, selected bool) string {
	return m.renderItem(p.name, p.description, selected)
}

func (m Model) renderUtilityItem(u utilityItem, selected bool) string {
	return m.renderItem(u.name, u.description, selected)
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("D", "Dashboard"),
		styles.RenderKeyHint("H", "History"),
		styles.RenderKeyHint("⏎", "Select"),
		styles.RenderKeyHint("Q", "Quit"),
	}

	separator := styles.RenderFooterSeparator()
	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
}
