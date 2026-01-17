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

const (
	tickIntervalMs     = 150
	headerSeparatorLen = 60
	subjectLabelWidth  = 12
	subjectBarWidth    = 25
	featureLabelWidth  = 14
	featureBarWidth    = 20
)

type StatsData struct {
	TotalSubjects     int            `json:"total_subjects"`
	BidsSubjects      int            `json:"bids_subjects"`
	EegPrepSubjects   int            `json:"eeg_prep_subjects"`
	FmriPrepSubjects  int            `json:"fmri_prep_subjects"`
	EpochsSubjects    int            `json:"epochs_subjects"`
	FeaturesSubjects  int            `json:"features_subjects"`
	FeatureCategories map[string]int `json:"feature_categories"`
	Task              string         `json:"task"`
}

type loadStatsMsg struct {
	Data  StatsData
	Error error
}

type tickMsg struct{}

type Model struct {
	stats      StatsData
	loading    bool
	loadError  error
	repoRoot   string
	lastUpdate time.Time
	ticker     int
}

func New(repoRoot string) Model {
	return Model{
		repoRoot: repoRoot,
		loading:  true,
	}
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.tick(),
		m.loadStats(),
	)
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*tickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

func (m Model) loadStats() tea.Cmd {
	return func() tea.Msg {
		output, err := m.executeStatsCommand()
		if err != nil {
			return loadStatsMsg{Error: err}
		}

		data, err := m.parseStatsOutput(output)
		if err != nil {
			return loadStatsMsg{Error: err}
		}

		return loadStatsMsg{Data: data}
	}
}

func (m Model) executeStatsCommand() (string, error) {
	pythonCmd := executor.GetPythonCommand(m.repoRoot)
	args := []string{"-m", "eeg_pipeline", "stats", "--json"}

	cmd := exec.Command(pythonCmd, args...)
	cmd.Dir = m.repoRoot
	cmd.Env = append(os.Environ(), "NO_COLOR=1", "PYTHONUNBUFFERED=1")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}

	if err := cmd.Start(); err != nil {
		return "", err
	}

	var output strings.Builder
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		output.WriteString(scanner.Text())
	}

	if err := cmd.Wait(); err != nil {
		return "", err
	}

	return output.String(), nil
}

func (m Model) parseStatsOutput(output string) (StatsData, error) {
	var data StatsData
	if err := json.Unmarshal([]byte(output), &data); err != nil {
		return StatsData{}, err
	}
	return data, nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.ticker++
		return m, m.tick()

	case loadStatsMsg:
		return m.handleLoadStatsMsg(msg)

	case tea.KeyMsg:
		return m.handleKeyMsg(msg)

	case tea.WindowSizeMsg:
		return m, nil
	}

	return m, nil
}

func (m Model) handleLoadStatsMsg(msg loadStatsMsg) (tea.Model, tea.Cmd) {
	m.loading = false
	if msg.Error != nil {
		m.loadError = msg.Error
	} else {
		m.stats = msg.Data
		m.lastUpdate = time.Now()
	}
	return m, nil
}

func (m Model) handleKeyMsg(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if msg.String() == "r" {
		m.loading = true
		m.loadError = nil
		return m, m.loadStats()
	}
	return m, nil
}

func (m Model) View() string {
	var b strings.Builder

	b.WriteString(m.renderHeader())
	b.WriteString("\n")
	b.WriteString(m.renderContent())
	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return styles.BoxStyle.Render(b.String())
}

func (m Model) renderContent() string {
	if m.loading {
		return m.renderLoading()
	}
	if m.loadError != nil {
		return m.renderError()
	}
	return m.renderStats()
}

func (m Model) renderHeader() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("◆ PROJECT DASHBOARD")

	separator := lipgloss.NewStyle().
		Foreground(styles.Secondary).
		Render(strings.Repeat("─", headerSeparatorLen))

	return title + "\n" + separator
}

func (m Model) renderLoading() string {
	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frameIndex := m.ticker % len(spinnerFrames)
	currentFrame := spinnerFrames[frameIndex]

	spinner := lipgloss.NewStyle().Foreground(styles.Accent).Render(currentFrame)
	loadingText := lipgloss.NewStyle().Foreground(styles.Text).Render(" Loading project statistics...")

	return "\n\n  " + spinner + loadingText + "\n\n"
}

func (m Model) renderError() string {
	errorStyle := lipgloss.NewStyle().
		Foreground(styles.Error).
		Bold(true)

	messageStyle := lipgloss.NewStyle().
		Foreground(styles.TextDim)

	errorMessage := errorStyle.Render("  " + styles.CrossMark + " Failed to load statistics")
	errorDetails := messageStyle.Render("  " + m.loadError.Error())

	return "\n\n" + errorMessage + "\n" + errorDetails + "\n\n"
}

func (m Model) renderStats() string {
	var b strings.Builder

	b.WriteString("\n")
	b.WriteString(m.renderTaskInfo())
	b.WriteString("\n")
	b.WriteString(m.renderSubjectProgress())
	b.WriteString("\n")
	b.WriteString(m.renderFeatureCategories())
	b.WriteString("\n")
	b.WriteString(m.renderLastUpdate())

	return b.String()
}

func (m Model) renderTaskInfo() string {
	taskLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("  Task: ")
	taskValue := lipgloss.NewStyle().Foreground(styles.Accent).Render(m.stats.Task)
	return taskLabel + taskValue
}

func (m Model) renderLastUpdate() string {
	if m.lastUpdate.IsZero() {
		return ""
	}
	updateTime := m.lastUpdate.Format("15:04:05")
	updateText := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Italic(true).
		Render("  Last updated: " + updateTime)
	return updateText + "\n"
}

func (m Model) renderSubjectProgress() string {
	var b strings.Builder

	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true).
		Render(" SUBJECTS ")
	b.WriteString("  " + header + "\n\n")

	totalSubjects := m.stats.TotalSubjects
	subjectItems := []struct {
		label string
		count int
		color lipgloss.Color
	}{
		{"Total", totalSubjects, styles.Text},
		{"BIDS", m.stats.BidsSubjects, styles.Accent},
		{"EEG Prep", m.stats.EegPrepSubjects, styles.Secondary},
		{"fMRI Prep", m.stats.FmriPrepSubjects, styles.Secondary},
		{"Epochs", m.stats.EpochsSubjects, styles.Warning},
		{"Features", m.stats.FeaturesSubjects, styles.Success},
	}

	for _, item := range subjectItems {
		b.WriteString(m.renderSubjectItem(item.label, item.count, item.color, totalSubjects))
	}

	return b.String()
}

func (m Model) renderSubjectItem(label string, count int, color lipgloss.Color, totalSubjects int) string {
	isTotal := label == "Total"
	percentage := m.calculatePercentage(count, totalSubjects, isTotal)

	labelText := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(subjectLabelWidth).
		Render("  " + label)

	progressBar := m.renderMiniBar(percentage, subjectBarWidth, color)

	countText := lipgloss.NewStyle().
		Foreground(color).
		Bold(true).
		Width(5).
		Align(lipgloss.Right).
		Render(fmt.Sprintf("%d", count))

	percentageText := m.formatPercentageText(percentage, isTotal)

	return labelText + countText + " " + progressBar + percentageText + "\n"
}

func (m Model) calculatePercentage(count, total int, isTotal bool) float64 {
	if isTotal {
		return 1.0
	}
	if total == 0 {
		return 0.0
	}
	return float64(count) / float64(total)
}

func (m Model) formatPercentageText(percentage float64, isTotal bool) string {
	if isTotal {
		return ""
	}
	percentageValue := percentage * 100
	return lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render(fmt.Sprintf(" (%.0f%%)", percentageValue))
}

func (m Model) renderMiniBar(percentage float64, width int, color lipgloss.Color) string {
	filledWidth := int(percentage * float64(width))
	emptyWidth := width - filledWidth

	filledBar := lipgloss.NewStyle().
		Foreground(color).
		Render(strings.Repeat("█", filledWidth))
	emptyBar := lipgloss.NewStyle().
		Foreground(styles.Secondary).
		Render(strings.Repeat("░", emptyWidth))

	return filledBar + emptyBar
}

func (m Model) renderFeatureCategories() string {
	var b strings.Builder

	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Underline(true).
		Render(" FEATURE CATEGORIES ")
	b.WriteString("  " + header + "\n\n")

	if len(m.stats.FeatureCategories) == 0 {
		noDataMessage := lipgloss.NewStyle().
			Foreground(styles.Muted).
			Italic(true).
			Render("    No feature data available\n")
		b.WriteString(noDataMessage)
		return b.String()
	}

	totalSubjects := m.getTotalForFeatureCategories()
	featureCategories := []string{
		"power", "connectivity", "aperiodic", "bursts", "complexity",
		"itpc", "pac", "quality", "erds", "spectral", "ratios", "asymmetry",
	}

	for _, category := range featureCategories {
		b.WriteString(m.renderFeatureCategory(category, totalSubjects))
	}

	return b.String()
}

func (m Model) getTotalForFeatureCategories() int {
	if m.stats.FeaturesSubjects > 0 {
		return m.stats.FeaturesSubjects
	}
	return m.stats.TotalSubjects
}

func (m Model) renderFeatureCategory(category string, totalSubjects int) string {
	count := m.stats.FeatureCategories[category]
	percentage := m.calculatePercentage(count, totalSubjects, false)

	categoryIcon := styles.BulletMark
	label := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(featureLabelWidth).
		Render("  " + categoryIcon + " " + category)

	progressBar := m.renderMiniBar(percentage, featureBarWidth, styles.Primary)

	countText := lipgloss.NewStyle().
		Foreground(styles.Text).
		Render(fmt.Sprintf(" %d/%d", count, totalSubjects))

	return label + progressBar + countText + "\n"
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("R", "Refresh"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	hintSeparator := lipgloss.NewStyle().
		Foreground(styles.Secondary).
		Render("  │  ")
	hintsText := strings.Join(hints, hintSeparator)

	return styles.FooterStyle.Render(hintsText)
}

