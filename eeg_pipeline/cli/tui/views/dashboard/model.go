package dashboard

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	headerSeparatorLen  = 60
	subjectLabelWidth   = 14
	subjectBarWidth     = 25
	featureLabelWidth   = 24
	featureBarWidth     = 20
)

type StatsData struct {
	TotalSubjects          int            `json:"total_subjects"`
	BidsSubjects           int            `json:"bids_subjects"`
	EegPrepSubjects        int            `json:"eeg_prep_subjects"`
	FmriPrepSubjects       int            `json:"fmri_prep_subjects"`
	EpochsSubjects         int            `json:"epochs_subjects"`
	FeaturesSubjects       int            `json:"features_subjects"`
	FmriFirstLevelSubjects int            `json:"fmri_first_level_subjects"`
	FmriBetaSeriesSubjects int            `json:"fmri_beta_series_subjects"`
	FmriLssSubjects        int            `json:"fmri_lss_subjects"`
	FeatureCategories      map[string]int `json:"feature_categories"`
	Task                   string         `json:"task"`
}

type loadStatsMsg struct {
	Data  StatsData
	Error error
}

type Model struct {
	stats      StatsData
	loading    bool
	loadError  error
	repoRoot   string
	lastUpdate time.Time
	animQueue  animation.Queue
	spinner    components.Spinner
}

func New(repoRoot string) Model {
	m := Model{
		repoRoot: repoRoot,
		loading:  true,
		spinner:  components.NewSpinner("Loading project statistics..."),
	}
	m.animQueue.Push(animation.ProgressPulseLoop())
	return m
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.loadStats(), m.tick(), m.immediateTick())
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
		m.animQueue.Tick()
		m.spinner.Tick()
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
		return m, tea.Batch(m.loadStats(), m.tick())
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
		Render("Project dashboard")

	sepLen := headerSeparatorLen
	if sepLen > 60 {
		sepLen = 60
	}
	return title + "\n" + styles.RenderHeaderSeparator(sepLen)
}

func (m Model) renderLoading() string {
	return "\n\n  " + m.spinner.View() + "\n\n"
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
	b.WriteString("\n\n")
	b.WriteString(m.renderEegSection())
	b.WriteString("\n")
	b.WriteString(styles.SectionDividerStyle.Render("  " + strings.Repeat(styles.SectionDividerChar, 36)))
	b.WriteString("\n\n")
	b.WriteString(m.renderFmriSection())
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
		Render("  last updated: " + updateTime)
	return updateText + "\n"
}

func (m Model) subjectRowColor(label string, count, total int) lipgloss.Color {
	if label == "Total" {
		return styles.Muted
	}
	if total == 0 {
		return styles.Muted
	}
	pct := float64(count) / float64(total)
	if pct == 0 {
		return styles.Warning
	}
	if pct >= 1 {
		return styles.Success
	}
	return styles.Primary
}

func (m Model) renderEegSection() string {
	var b strings.Builder

	header := styles.SectionTitleStyle.Render("EEG")
	b.WriteString("  " + header + "\n\n")

	totalSubjects := m.stats.TotalSubjects
	eegRows := []struct {
		label string
		count int
	}{
		{"Total", totalSubjects},
		{"BIDS", m.stats.BidsSubjects},
		{"EEG Prep", m.stats.EegPrepSubjects},
		{"Epochs", m.stats.EpochsSubjects},
		{"Features", m.stats.FeaturesSubjects},
	}

	for _, row := range eegRows {
		color := m.subjectRowColor(row.label, row.count, totalSubjects)
		b.WriteString(m.renderSubjectItem(row.label, row.count, color, totalSubjects))
	}

	b.WriteString("\n")
	b.WriteString(m.renderFeatureCategories())

	return b.String()
}

func (m Model) renderFmriSection() string {
	var b strings.Builder

	header := styles.SectionTitleStyle.Render("fMRI")
	b.WriteString("  " + header + "\n\n")

	totalSubjects := m.stats.TotalSubjects
	fmriRows := []struct {
		label string
		count int
	}{
		{"Total", totalSubjects},
		{"BIDS", m.stats.BidsSubjects},
		{"fMRI Prep", m.stats.FmriPrepSubjects},
		{"First level", m.stats.FmriFirstLevelSubjects},
		{"Beta series", m.stats.FmriBetaSeriesSubjects},
		{"LSS", m.stats.FmriLssSubjects},
	}

	for _, row := range fmriRows {
		color := m.subjectRowColor(row.label, row.count, totalSubjects)
		b.WriteString(m.renderSubjectItem(row.label, row.count, color, totalSubjects))
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
	zeroIndicator := ""
	if !isTotal && totalSubjects > 0 && count == 0 {
		zeroIndicator = " " + lipgloss.NewStyle().Foreground(styles.Warning).Render("—")
	}

	return labelText + countText + " " + progressBar + percentageText + zeroIndicator + "\n"
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

	filledBar := lipgloss.NewStyle().Foreground(color).Render(strings.Repeat("▓", filledWidth))
	emptyBar := lipgloss.NewStyle().Foreground(styles.Muted).Render(strings.Repeat("░", emptyWidth))

	return filledBar + emptyBar
}

func (m Model) featureCategoryRowColor(count, total int) lipgloss.Color {
	if total == 0 {
		return styles.Muted
	}
	pct := float64(count) / float64(total)
	if pct == 0 {
		return styles.Warning
	}
	if pct >= 1 {
		return styles.Success
	}
	return styles.Primary
}

func (m Model) allFeatureCategoriesZero(totalSubjects int) bool {
	for _, count := range m.stats.FeatureCategories {
		if count > 0 {
			return false
		}
	}
	return totalSubjects > 0
}

func (m Model) renderFeatureCategories() string {
	var b strings.Builder

	header := styles.SectionTitleStyle.Render("Feature categories")
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
	keys := make([]string, 0, len(m.stats.FeatureCategories))
	for k := range m.stats.FeatureCategories {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	if m.allFeatureCategoriesZero(totalSubjects) {
		n := len(keys)
		summary := lipgloss.NewStyle().
			Foreground(styles.Warning).
			Render(fmt.Sprintf("    All categories: 0/%d subjects (no data yet)", totalSubjects))
		b.WriteString(summary + "\n")
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).
			Render(fmt.Sprintf("    %d categories: %s", n, strings.Join(keys, ", "))) + "\n")
		return b.String()
	}

	for _, category := range keys {
		if category == "" {
			continue
		}
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
	color := m.featureCategoryRowColor(count, totalSubjects)

	label := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(featureLabelWidth).
		Render("  " + styles.BulletMark + " " + category)

	progressBar := m.renderMiniBar(percentage, featureBarWidth, color)

	countText := lipgloss.NewStyle().
		Foreground(color).
		Render(fmt.Sprintf(" %d/%d", count, totalSubjects))

	zeroIndicator := ""
	if totalSubjects > 0 && count == 0 {
		zeroIndicator = " " + lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.WarningMark)
	}

	return label + progressBar + countText + zeroIndicator + "\n"
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("R", "Refresh"),
		styles.RenderKeyHint("Esc", "Back"),
	}

	hintsText := strings.Join(hints, styles.RenderFooterSeparator())

	return styles.FooterStyle.Render(hintsText)
}
