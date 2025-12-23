package dashboard

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

///////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////

type StatsData struct {
	TotalSubjects     int            `json:"total_subjects"`
	BidsSubjects      int            `json:"bids_subjects"`
	EpochsSubjects    int            `json:"epochs_subjects"`
	FeaturesSubjects  int            `json:"features_subjects"`
	EpochsPct         float64        `json:"epochs_pct"`
	FeaturesPct       float64        `json:"features_pct"`
	FeatureCategories map[string]int `json:"feature_categories"`
	Task              string         `json:"task"`
	DerivRoot         string         `json:"deriv_root"`
}

type loadStatsMsg struct {
	Data  StatsData
	Error error
}

type tickMsg struct{}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	stats      StatsData
	loading    bool
	loadError  error
	repoRoot   string
	lastUpdate time.Time

	width  int
	height int
	ticker int
}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New(repoRoot string) Model {
	return Model{
		repoRoot: repoRoot,
		loading:  true,
	}
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.tick(),
		m.loadStats(),
	)
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*150, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) loadStats() tea.Cmd {
	return func() tea.Msg {
		pyCmd := executor.GetPythonCommand(m.repoRoot)
		args := []string{"-m", "eeg_pipeline", "stats", "--json"}

		cmd := exec.Command(pyCmd, args...)
		cmd.Dir = m.repoRoot
		cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return loadStatsMsg{Error: err}
		}

		if err := cmd.Start(); err != nil {
			return loadStatsMsg{Error: err}
		}

		var output strings.Builder
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			output.WriteString(scanner.Text())
		}

		if err := cmd.Wait(); err != nil {
			return loadStatsMsg{Error: err}
		}

		var data StatsData
		if err := json.Unmarshal([]byte(output.String()), &data); err != nil {
			return loadStatsMsg{Error: err}
		}

		return loadStatsMsg{Data: data}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		return m, m.tick()

	case loadStatsMsg:
		m.loading = false
		if msg.Error != nil {
			m.loadError = msg.Error
		} else {
			m.stats = msg.Data
			m.lastUpdate = time.Now()
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "r":
			m.loading = true
			m.loadError = nil
			return m, m.loadStats()
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	}

	return m, nil
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

func (m Model) View() string {
	var b strings.Builder

	// Header
	b.WriteString(m.renderHeader())
	b.WriteString("\n")

	if m.loading {
		b.WriteString(m.renderLoading())
	} else if m.loadError != nil {
		b.WriteString(m.renderError())
	} else {
		b.WriteString(m.renderStats())
	}

	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return styles.BoxStyle.Render(b.String())
}

func (m Model) renderHeader() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("◆ PROJECT DASHBOARD")

	separator := lipgloss.NewStyle().
		Foreground(styles.Secondary).
		Render(strings.Repeat("─", 60))

	return title + "\n" + separator
}

func (m Model) renderLoading() string {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := frames[m.ticker%len(frames)]

	spinner := lipgloss.NewStyle().Foreground(styles.Accent).Render(frame)
	text := lipgloss.NewStyle().Foreground(styles.Text).Render(" Loading project statistics...")

	return "\n\n  " + spinner + text + "\n\n"
}

func (m Model) renderError() string {
	errorStyle := lipgloss.NewStyle().
		Foreground(styles.Error).
		Bold(true)

	msgStyle := lipgloss.NewStyle().
		Foreground(styles.TextDim)

	return "\n\n" +
		errorStyle.Render("  "+styles.CrossMark+" Failed to load statistics") + "\n" +
		msgStyle.Render("  "+m.loadError.Error()) + "\n\n"
}

func (m Model) renderStats() string {
	var b strings.Builder

	// Task info
	b.WriteString("\n")
	taskLine := lipgloss.NewStyle().Foreground(styles.TextDim).Render("  Task: ") +
		lipgloss.NewStyle().Foreground(styles.Accent).Render(m.stats.Task)
	b.WriteString(taskLine + "\n\n")

	// Subject Progress Card
	b.WriteString(m.renderSubjectProgress())
	b.WriteString("\n")

	// Feature Categories Card
	b.WriteString(m.renderFeatureCategories())
	b.WriteString("\n")

	// Last update
	if !m.lastUpdate.IsZero() {
		updateTime := lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render(
			"  Last updated: " + m.lastUpdate.Format("15:04:05"))
		b.WriteString(updateTime + "\n")
	}

	return b.String()
}

func (m Model) renderSubjectProgress() string {
	var b strings.Builder

	// Section header
	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true).
		Render(" SUBJECTS ")
	b.WriteString("  " + header + "\n\n")

	// Stats with mini progress bars
	total := m.stats.TotalSubjects

	items := []struct {
		label string
		count int
		color lipgloss.Color
	}{
		{"Total", total, styles.Text},
		{"BIDS", m.stats.BidsSubjects, styles.Accent},
		{"Epochs", m.stats.EpochsSubjects, styles.Warning},
		{"Features", m.stats.FeaturesSubjects, styles.Success},
	}

	labelWidth := 12
	barWidth := 25

	for _, item := range items {
		label := lipgloss.NewStyle().
			Foreground(styles.TextDim).
			Width(labelWidth).
			Render("  " + item.label)

		pct := 0.0
		if total > 0 && item.label != "Total" {
			pct = float64(item.count) / float64(total)
		} else if item.label == "Total" {
			pct = 1.0
		}

		bar := m.renderMiniBar(pct, barWidth, item.color)

		countStr := lipgloss.NewStyle().
			Foreground(item.color).
			Bold(true).
			Width(5).
			Align(lipgloss.Right).
			Render(fmt.Sprintf("%d", item.count))

		pctStr := ""
		if item.label != "Total" && total > 0 {
			pctStr = lipgloss.NewStyle().
				Foreground(styles.Muted).
				Render(fmt.Sprintf(" (%.0f%%)", pct*100))
		}

		b.WriteString(label + countStr + " " + bar + pctStr + "\n")
	}

	return b.String()
}

func (m Model) renderMiniBar(pct float64, width int, color lipgloss.Color) string {
	filled := int(pct * float64(width))
	if filled > width {
		filled = width
	}

	bar := lipgloss.NewStyle().Foreground(color).Render(strings.Repeat("█", filled))
	empty := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("░", width-filled))

	return bar + empty
}

func (m Model) renderFeatureCategories() string {
	var b strings.Builder

	// Section header
	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true).
		Render(" FEATURE CATEGORIES ")
	b.WriteString("  " + header + "\n\n")

	if len(m.stats.FeatureCategories) == 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).Render("    No feature data available\n"))
		return b.String()
	}

	total := m.stats.FeaturesSubjects
	if total == 0 {
		total = m.stats.TotalSubjects
	}

	categories := []string{
		"power", "connectivity", "aperiodic", "bursts", "complexity",
		"itpc", "pac", "quality", "erds", "spectral", "ratios", "asymmetry",
	}

	for _, cat := range categories {
		count, ok := m.stats.FeatureCategories[cat]
		if !ok {
			count = 0
		}

		pct := 0.0
		if total > 0 {
			pct = float64(count) / float64(total)
		}

		// Icon based on category
		icon := m.getCategoryIcon(cat)

		label := lipgloss.NewStyle().
			Foreground(styles.TextDim).
			Width(14).
			Render("  " + icon + " " + cat)

		bar := m.renderMiniBar(pct, 20, styles.Primary)

		countStr := lipgloss.NewStyle().
			Foreground(styles.Text).
			Render(fmt.Sprintf(" %d/%d", count, total))

		b.WriteString(label + bar + countStr + "\n")
	}

	return b.String()
}

func (m Model) getCategoryIcon(cat string) string {
	icons := map[string]string{
		"power":        "▸",
		"connectivity": "▸",
		"aperiodic":    "▸",

		"bursts":     "▸",
		"complexity": "▸",
		"itpc":       "▸",
		"pac":        "▸",
		"quality":    "▸",
		"erds":       "▸",
		"spectral":   "▸",
		"ratios":     "▸",
		"asymmetry":  "▸",
	}
	if icon, ok := icons[cat]; ok {
		return icon
	}
	return "•"
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("R", "Refresh"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	return styles.FooterStyle.Render(strings.Join(hints, separator))
}

///////////////////////////////////////////////////////////////////
// Public Methods
///////////////////////////////////////////////////////////////////

func (m Model) IsLoading() bool {
	return m.loading
}

func (m Model) HasError() bool {
	return m.loadError != nil
}

func (m Model) GetStats() StatsData {
	return m.stats
}
