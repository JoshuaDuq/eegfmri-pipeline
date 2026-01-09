package mainmenu

import (
	"fmt"
	"strings"
	"time"

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
	shortcut    string
}

var pipelines = []pipelineItem{
	{"Preprocessing", "Bad channels, ICA, epochs", "1"},
	{"Features", "Extract EEG feature sets", "2"},
	{"Behavior", "EEG-behavior analysis", "3"},
	{"Machine Learning", "LOSO regression & classification", "4"},
	{"Plotting", "Curate and export visualization suites", "5"},
}

type utilityItem struct {
	name        string
	description string
	shortcut    string
}

const (
	UtilityGlobalSetup = iota
	UtilityMergePsychopy
	UtilityRawToBids
)

var utilities = []utilityItem{
	{"Merge PsychoPy Data", "Merge PsychoPy data into BIDS events files", "M"},
	{"Raw to BIDS", "Convert raw EEG data to BIDS format", "R"},
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	pipelineCursor   int
	utilityCursor    int
	inUtilities      bool // true when browsing utilities section
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
	ticker int
}

func New() Model {
	help := components.NewHelpOverlay("Keyboard Shortcuts", 50)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓ or j/k", Description: "Move cursor"},
		{Key: "1-5", Description: "Quick select pipeline"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Select pipeline"},
		{Key: "G", Description: "Open Global Setup"},
		{Key: "?", Description: "Toggle help"},
	})
	help.AddSection("General", []components.HelpItem{
		{Key: "Esc", Description: "Go back"},
		{Key: "q", Description: "Quit application"},
	})

	return Model{
		pipelineCursor:   0,
		SelectedPipeline: -1,
		SelectedUtility:  -1,
		Task:             "thermalactive",
		helpOverlay:      help,
	}
}

// SetCursor sets the pipeline cursor position (for restoring last selected pipeline)
func (m *Model) SetCursor(idx int) {
	if idx >= 0 && idx < len(pipelines) {
		m.pipelineCursor = idx
		m.inUtilities = false
	}
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

type tickMsg struct{}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
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
		case "tab":
			m.handleTab()
		case "enter", " ":
			return m.handleEnter()
		case "1", "2", "3", "4", "5":
			idx := int(msg.String()[0] - '1')
			if idx >= 0 && idx < len(pipelines) {
				m.inUtilities = false
				m.pipelineCursor = idx
				m.SelectedPipeline = idx
			}
		case "m", "M":
			// Quick select Merge PsychoPy Data utility
			m.inUtilities = true
			m.utilityCursor = UtilityMergePsychopy
			m.SelectedUtility = UtilityMergePsychopy
		case "r", "R":
			// Quick select Raw to BIDS utility
			m.inUtilities = true
			m.utilityCursor = UtilityRawToBids
			m.SelectedUtility = UtilityRawToBids
		case "g", "G":
			// Quick select Global Setup
			m.inUtilities = true
			m.utilityCursor = UtilityGlobalSetup
			m.SelectedUtility = UtilityGlobalSetup
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.helpOverlay.Width = min(50, m.width-10)
	}

	return m, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func animatedCursor(selected bool, ticker int) string {
	if !selected {
		return "   "
	}

	frames := []string{"▸", "▹", "▸", "▹"}
	frame := frames[(ticker/2)%len(frames)]

	return lipgloss.NewStyle().
		Foreground(styles.Primary).
		Bold(true).
		Render(" " + frame + " ")
}

func (m *Model) handleUp() {
	if m.inUtilities {
		if m.utilityCursor > 0 {
			m.utilityCursor--
		} else {
			m.utilityCursor = len(utilities) - 1
		}
	} else {
		if m.pipelineCursor > 0 {
			m.pipelineCursor--
		} else {
			m.pipelineCursor = len(pipelines) - 1
		}
	}
}

func (m *Model) handleDown() {
	if m.inUtilities {
		if m.utilityCursor < len(utilities)-1 {
			m.utilityCursor++
		} else {
			m.utilityCursor = 0
		}
	} else {
		if m.pipelineCursor < len(pipelines)-1 {
			m.pipelineCursor++
		} else {
			m.pipelineCursor = 0
		}
	}
}

func (m *Model) handleTab() {
	m.inUtilities = !m.inUtilities
}

func (m Model) handleEnter() (tea.Model, tea.Cmd) {
	if m.inUtilities {
		// Map utility items to pipeline types for wizard launching
		switch m.utilityCursor {
		case UtilityGlobalSetup:
			m.SelectedUtility = UtilityGlobalSetup
			return m, nil
		case UtilityMergePsychopy:
			m.SelectedPipeline = int(types.PipelineMergePsychoPyData)
		case UtilityRawToBids:
			m.SelectedPipeline = int(types.PipelineRawToBIDS)
		}
	} else {
		m.SelectedPipeline = m.pipelineCursor
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

	// Main Content - Dual Column Layout
	pipelinesCol := m.renderPipelinesColumn()
	utilitiesCol := m.renderUtilitiesColumn()

	leftContent := pipelinesCol + "\n" + utilitiesCol

	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(m.width-10).Render(leftContent),
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
	// Render the base view dimmed
	baseView := m.renderBaseView()

	// Center the overlay
	overlayPlaced := lipgloss.Place(
		m.width, m.height,
		lipgloss.Center, lipgloss.Center,
		overlay,
	)

	// Simple overlay (in a real implementation, we'd dim the background)
	return overlayPlaced + "\n" + lipgloss.NewStyle().Foreground(styles.Muted).Render(baseView)
}

func (m Model) renderBaseView() string {
	var b strings.Builder
	b.WriteString(m.renderHeader())
	b.WriteString("\n\n")

	pipelinesCol := m.renderPipelinesColumn()

	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(m.width-10).Render(pipelinesCol),
	)
	b.WriteString(content)
	return b.String()
}

func (m Model) renderHeader() string {
	// Animated logo accent
	accentChars := []string{"◆", "◇", "◆", "◈"}
	accent := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Accent).
		Render(accentChars[(m.ticker/3)%len(accentChars)])

	// Logo/Brand with gradient-like effect
	logoStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary)

	logo := accent + " " + logoStyle.Render("EEG PIPELINE")
	version := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render(" v1.0")

	subtitle := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Italic(true).
		Render("EEG Pipeline")

	// Environment badge
	var envBadge string
	if m.IsCloud {
		envBadge = "  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1).
			Render("CLOUD")
	} else {
		// Subtle pulse for local indicator
		localFrames := []string{"●", "◉", "●", "◎"}
		localIcon := localFrames[(m.ticker/5)%len(localFrames)]
		envBadge = "  " + lipgloss.NewStyle().
			Foreground(styles.Success).
			Padding(0, 1).
			Render(localIcon+" LOCAL")
	}

	header := lipgloss.JoinVertical(lipgloss.Left,
		logo+version+envBadge,
		"  "+subtitle,
	)

	// Decorative gradient-style line
	lineWidth := m.width - 8
	if lineWidth < 0 {
		lineWidth = 0
	}
	line1 := lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("─", lineWidth/3))
	line2 := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", lineWidth/3))
	line3 := lipgloss.NewStyle().Foreground(styles.Muted).Render(strings.Repeat("─", lineWidth/3+lineWidth%3))
	line := line1 + line2 + line3

	return header + "\n" + line
}

func (m Model) renderPipelinesColumn() string {
	var lines []string

	// Section header with icon
	headerIcon := lipgloss.NewStyle().Foreground(styles.Accent).Render("▸ ")
	if m.inUtilities {
		headerIcon = lipgloss.NewStyle().Foreground(styles.Muted).Render("  ")
	}
	header := lipgloss.JoinHorizontal(lipgloss.Left,
		headerIcon,
		styles.SectionTitleStyle.Render(" PIPELINES "),
	)
	lines = append(lines, header)
	lines = append(lines, "")

	for i, p := range pipelines {
		isSelected := !m.inUtilities && i == m.pipelineCursor
		lines = append(lines, m.renderPipelineItem(p, isSelected))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderUtilitiesColumn() string {
	var lines []string

	// Section header with icon
	headerIcon := lipgloss.NewStyle().Foreground(styles.Accent).Render("▸ ")
	if !m.inUtilities {
		headerIcon = lipgloss.NewStyle().Foreground(styles.Muted).Render("  ")
	}
	header := lipgloss.JoinHorizontal(lipgloss.Left,
		headerIcon,
		styles.SectionTitleStyle.Render(" UTILITIES "),
	)
	lines = append(lines, header)
	lines = append(lines, "")

	for i, u := range utilities {
		isSelected := m.inUtilities && i == m.utilityCursor
		lines = append(lines, m.renderUtilityItem(u, isSelected))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderUtilityItem(u utilityItem, selected bool) string {
	var item strings.Builder

	// Selection indicator
	cursor := animatedCursor(selected, m.ticker)

	// Shortcut badge
	shortcutStyle := lipgloss.NewStyle().Foreground(styles.Muted).Render("[" + u.shortcut + "]")

	// Name with styling
	nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
	if selected {
		nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	}

	// First line: cursor + shortcut + name
	item.WriteString(fmt.Sprintf("%s%s %s\n", cursor, shortcutStyle, nameStyle.Render(u.name)))

	// Second line: description (indented)
	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	item.WriteString("       " + descStyle.Render(u.description) + "\n")

	return item.String()
}

func (m Model) renderPipelineItem(p pipelineItem, selected bool) string {
	var item strings.Builder

	// Selection indicator with animation
	cursor := animatedCursor(selected, m.ticker)

	// Shortcut badge
	shortcutStyle := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render("[" + p.shortcut + "]")

	// Name with styling
	nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
	if selected {
		nameStyle = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)
	}

	// First line: cursor + shortcut + name
	item.WriteString(fmt.Sprintf("%s%s %s\n", cursor, shortcutStyle, nameStyle.Render(p.name)))

	// Second line: description (indented)
	descStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Italic(true)
	item.WriteString("       " + descStyle.Render(p.description) + "\n")

	return item.String()
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("1-5", "Pipeline"),
		styles.RenderKeyHint("D", "Dashboard"),
		styles.RenderKeyHint("H", "History"),
		styles.RenderKeyHint("Enter", "Select"),
		styles.RenderKeyHint("Q", "Quit"),
	}

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	footerContent := strings.Join(hints, separator)

	return styles.FooterStyle.Width(m.width - 8).Render(footerContent)
}
