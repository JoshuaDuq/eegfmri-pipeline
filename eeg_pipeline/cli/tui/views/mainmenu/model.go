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
	icon        string
	shortcut    string
}

var pipelines = []pipelineItem{
	{"Preprocessing", "Bad channels, ICA, epochs", "▸", "1"},
	{"Features", "Extract power, connectivity, dynamics", "▸", "2"},
	{"Behavior", "EEG-behavior correlation analysis", "▸", "3"},
	{"ERP", "Event-related potential statistics", "▸", "4"},
	{"TFR", "Time-frequency representations", "▸", "5"},
	{"Decoding", "LOSO regression & classification", "▸", "6"},
}

type utilityItem struct {
	name        string
	description string
	shortcut    string
}

var utilities = []utilityItem{
	{"Combine Features", "Merge all feature files into features_all.tsv", "C"},
	{"Merge Behavior", "Merge behavioral data into BIDS events files", "M"},
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

	SubjectCount  int
	FeaturesDone  int
	FeaturesTotal int
	Task          string
	IsCloud       bool

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
		{Key: "1-6", Description: "Quick select pipeline"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Select pipeline"},
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
		case "1", "2", "3", "4", "5", "6":
			idx := int(msg.String()[0] - '1')
			if idx >= 0 && idx < len(pipelines) {
				m.inUtilities = false
				m.pipelineCursor = idx
				m.SelectedPipeline = idx
			}
		case "c", "C":
			// Quick select Combine Features utility
			m.inUtilities = true
			m.utilityCursor = 0
			m.SelectedUtility = 0
		case "m", "M":
			// Quick select Merge Behavior utility
			m.inUtilities = true
			m.utilityCursor = 1
			m.SelectedUtility = 1
		case "r", "R":
			// Quick select Raw to BIDS utility
			m.inUtilities = true
			m.utilityCursor = 2
			m.SelectedUtility = 2
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
		case 0:
			m.SelectedPipeline = int(types.PipelineCombineFeatures)
		case 1:
			m.SelectedPipeline = int(types.PipelineMergeBehavior)
		case 2:
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

	var b strings.Builder

	// Header with logo
	b.WriteString(m.renderHeader())
	b.WriteString("\n\n")

	// Main Content - Dual Column Layout
	pipelinesCol := m.renderPipelinesColumn()
	utilitiesCol := m.renderUtilitiesColumn()
	sidebarCol := m.renderSidebar()

	leftContent := pipelinesCol + "\n" + utilitiesCol
	leftWidth := (m.width - 10) * 2 / 3
	if leftWidth < 52 {
		leftWidth = 52
	}
	rightWidth := m.width - leftWidth - 10
	if rightWidth < 26 {
		rightWidth = 26
	}

	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(leftWidth).Render(leftContent),
		"  ",
		styles.CardStyle.Width(rightWidth).Render(sidebarCol),
	)

	b.WriteString(content + "\n\n")

	// Toast notification (if visible)
	if m.toast.Visible {
		b.WriteString(m.toast.View() + "\n\n")
	}

	// Footer
	b.WriteString(m.renderFooter())

	return b.String()
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
	sidebarCol := m.renderSidebar()

	content := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.CardStyle.Width(52).Render(pipelinesCol),
		"  ",
		styles.CardStyle.Width(26).Render(sidebarCol),
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
		Render("Thermal Pain Analysis Suite")

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
			Background(styles.BgBase).
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
		lines = append(lines, m.renderPipelineItem(i, p, isSelected))
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
	cursor := "   "
	if selected {
		frames := []string{"▸", "▹", "▸", "▹"}
		frame := frames[(m.ticker/2)%len(frames)]
		cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(" " + frame + " ")
	}

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

func (m Model) renderPipelineItem(_ int, p pipelineItem, selected bool) string {
	var item strings.Builder

	// Selection indicator with animation
	cursor := "   "
	if selected {
		// Animate the cursor
		frames := []string{"▸", "▹", "▸", "▹"}
		frame := frames[(m.ticker/2)%len(frames)]
		cursor = lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(" " + frame + " ")
	}

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

func (m Model) renderSidebar() string {
	var lines []string

	// Status section
	header := lipgloss.JoinHorizontal(lipgloss.Left,
		lipgloss.NewStyle().Foreground(styles.Accent).Render("▸ "),
		styles.SectionTitleStyle.Render(" STATUS "),
	)
	lines = append(lines, header)
	lines = append(lines, "")

	// Subject count with animated indicator when loading
	subjectValue := m.formatSubjectCount()
	lines = append(lines, m.renderStatusRow("Subjects", subjectValue))

	// Task info
	lines = append(lines, m.renderStatusRow("Task", m.Task))

	lines = append(lines, "")

	// Quick stats section
	statsHeader := lipgloss.JoinHorizontal(lipgloss.Left,
		lipgloss.NewStyle().Foreground(styles.Accent).Render("▸ "),
		styles.SectionTitleStyle.Render(" STATS "),
	)
	lines = append(lines, statsHeader)
	lines = append(lines, "")

	// Feature progress
	if m.FeaturesTotal > 0 {
		pct := float64(m.FeaturesDone) / float64(m.FeaturesTotal) * 100
		progressBar := m.renderMiniProgress(pct / 100)
		lines = append(lines, "  "+progressBar)
		lines = append(lines, m.renderStatusRow("Features", fmt.Sprintf("%.0f%%", pct)))
	} else {
		lines = append(lines, lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("  No cache"))
	}

	return strings.Join(lines, "\n")
}

func (m Model) renderStatusRow(label, value string) string {
	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10)
	return "  " + labelStyle.Render(label) + " " + value
}

func (m Model) formatSubjectCount() string {
	if m.SubjectCount == 0 {
		// Animated loading indicator
		frames := []string{"◐", "◓", "◑", "◒"}
		frame := frames[m.ticker%len(frames)]
		return lipgloss.NewStyle().Foreground(styles.Accent).Render(frame + " probing...")
	}
	return lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render(fmt.Sprintf("%d", m.SubjectCount))
}

func (m Model) renderMiniProgress(pct float64) string {
	width := 16
	filled := int(pct * float64(width))
	if filled > width {
		filled = width
	}

	bar := lipgloss.NewStyle().Foreground(styles.Success).Render(strings.Repeat("█", filled))
	empty := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("░", width-filled))
	return bar + empty
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("Tab", "Switch"),
		styles.RenderKeyHint("1-6", "Pipeline"),
		styles.RenderKeyHint("C", "Combine"),
		styles.RenderKeyHint("M", "Merge"),
		styles.RenderKeyHint("R", "Raw2BIDS"),
		styles.RenderKeyHint("Enter", "Select"),
		styles.RenderKeyHint("Q", "Quit"),
	}

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	footerContent := strings.Join(hints, separator)

	return styles.FooterStyle.Width(m.width - 8).Render(footerContent)
}
