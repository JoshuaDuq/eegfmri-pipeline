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
	defaultWidth       = 120
	defaultHeight      = 32
	boxFrameWidth      = 6
	dashboardColumnGap = 2
	minSectionWidth    = 28
)

type sectionLayout struct {
	width             int
	countWidth        int
	featureCountWidth int
	subjectLabelWidth int
	subjectBarWidth   int
	featureLabelWidth int
	featureBarWidth   int
	showPercent       bool
	showIndicators    bool
}

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
	width      int
	height     int
}

func New(repoRoot string) Model {
	m := Model{
		repoRoot: repoRoot,
		loading:  true,
		spinner:  components.NewSpinner("Loading project statistics..."),
		width:    defaultWidth,
		height:   defaultHeight,
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
		m.width = msg.Width
		m.height = msg.Height
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

func (m Model) effectiveDimensions() (int, int) {
	width := m.width
	height := m.height
	if width <= 0 {
		width = defaultWidth
	}
	if height <= 0 {
		height = defaultHeight
	}
	return width, height
}

func (m Model) boxWidth() int {
	width, _ := m.effectiveDimensions()
	width -= 2
	if width < 1 {
		return 1
	}
	return width
}

func (m Model) contentWidth() int {
	width, _ := m.effectiveDimensions()
	width -= boxFrameWidth
	if width < 1 {
		return 1
	}
	return width
}

func (m Model) usesTwoColumnLayout(contentWidth int) bool {
	width, height := m.effectiveDimensions()
	if !styles.UseTwoColumnLayout(width, height) {
		return false
	}
	leftWidth, rightWidth := m.statsColumnWidths(contentWidth)
	return leftWidth >= minSectionWidth && rightWidth >= minSectionWidth
}

func (m Model) View() string {
	contentWidth := m.contentWidth()

	var b strings.Builder
	b.WriteString(m.renderHeader(contentWidth))
	b.WriteString(m.renderContent(contentWidth))
	b.WriteString(m.renderFooter(contentWidth))

	return styles.BoxStyle.Width(m.boxWidth()).Render(b.String())
}

func (m Model) renderContent(width int) string {
	if m.loading {
		return m.renderLoading(width)
	}
	if m.loadError != nil {
		return m.renderError(width)
	}
	return m.renderStats(width)
}

func (m Model) renderHeader(width int) string {
	glyph := lipgloss.NewStyle().Foreground(styles.Primary).Render("◈")
	title := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render("Project Dashboard")
	return "  " + glyph + "  " + title + "\n" + styles.RenderDivider(width) + "\n"
}

func (m Model) renderLoading(width int) string {
	line := styles.TruncateLine("  "+m.spinner.View(), width)
	return "\n" + line + "\n"
}

func (m Model) renderError(width int) string {
	errorStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
	messageStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

	header := styles.TruncateLine(errorStyle.Render("  "+styles.CrossMark+" Failed to load statistics"), width)
	message := styles.TruncateLine(messageStyle.Render("  "+m.loadError.Error()), width)
	return "\n" + header + "\n" + message + "\n"
}

func (m Model) renderStats(width int) string {
	var b strings.Builder
	b.WriteString(m.renderSummaryStrip(width))
	b.WriteString("\n")
	b.WriteString(m.renderStatsSections(width))
	if lastUpdate := m.renderLastUpdate(width); lastUpdate != "" {
		b.WriteString("\n")
		b.WriteString(lastUpdate)
		b.WriteString("\n")
	}
	return b.String()
}

func (m Model) renderSummaryStrip(width int) string {
	taskLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("Task")
	taskValue := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.stats.Task)

	totalLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("Subjects")
	totalValue := lipgloss.NewStyle().Foreground(styles.Text).Bold(true).Render(fmt.Sprintf("%d", m.stats.TotalSubjects))

	left := "  " + taskLabel + "  " + taskValue + "    " + totalLabel + "  " + totalValue

	updatedAt := ""
	if !m.lastUpdate.IsZero() {
		updatedAt = lipgloss.NewStyle().Foreground(styles.Muted).Render("Updated " + m.lastUpdate.Format("15:04:05"))
	}

	spacer := lipgloss.NewStyle().Width(max(width-lipgloss.Width(left)-lipgloss.Width(updatedAt), 0)).Render("")
	return left + spacer + updatedAt + "\n" + styles.RenderDivider(width) + "\n"
}

func (m Model) renderLastUpdate(_ int) string {
	return ""
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

func (m Model) statsColumnWidths(width int) (int, int) {
	leftWidth := (width - dashboardColumnGap) * 54 / 100
	rightWidth := width - dashboardColumnGap - leftWidth
	return leftWidth, rightWidth
}

func (m Model) renderStatsSections(width int) string {
	if !m.usesTwoColumnLayout(width) {
		eegSection := strings.TrimRight(m.renderEegSection(width), "\n")
		fmriSection := strings.TrimRight(m.renderFmriSection(width), "\n")
		return eegSection + "\n" + styles.RenderDivider(width) + "\n" + fmriSection
	}

	leftWidth, rightWidth := m.statsColumnWidths(width)
	left := lipgloss.NewStyle().Width(leftWidth).Render(strings.TrimRight(m.renderEegSection(leftWidth), "\n"))
	right := lipgloss.NewStyle().Width(rightWidth).Render(strings.TrimRight(m.renderFmriSection(rightWidth), "\n"))

	return lipgloss.JoinHorizontal(lipgloss.Top, left, strings.Repeat(" ", dashboardColumnGap), right)
}

func (m Model) maxCountWidth() int {
	maxCount := m.stats.TotalSubjects
	for _, count := range m.stats.FeatureCategories {
		if count > maxCount {
			maxCount = count
		}
	}
	width := len(fmt.Sprintf("%d", maxCount))
	if width < 3 {
		width = 3
	}
	return width
}

func (m Model) layoutForSection(width int) sectionLayout {
	countWidth := max(m.maxCountWidth(), 3)
	featureCountWidth := max(len(fmt.Sprintf("%d/%d", m.stats.TotalSubjects, m.stats.TotalSubjects)), 3)
	subjectLabelWidth := min(max(width/4, 10), 14)
	featureLabelWidth := min(max(width/2, 14), 24)
	showPercent := width >= 44
	showIndicators := width >= 34

	percentWidth := 0
	if showPercent {
		percentWidth = len(" (100%)")
	}
	indicatorWidth := 0
	if showIndicators {
		indicatorWidth = 2
	}

	subjectBarWidth := width - subjectLabelWidth - countWidth - 1 - percentWidth - indicatorWidth
	if subjectBarWidth < 6 {
		subjectBarWidth = 6
	}

	featureBarWidth := width - featureLabelWidth - 1 - featureCountWidth - indicatorWidth
	if featureBarWidth < 6 {
		featureBarWidth = 6
	}

	return sectionLayout{
		width:             width,
		countWidth:        countWidth,
		featureCountWidth: featureCountWidth,
		subjectLabelWidth: subjectLabelWidth,
		subjectBarWidth:   subjectBarWidth,
		featureLabelWidth: featureLabelWidth,
		featureBarWidth:   featureBarWidth,
		showPercent:       showPercent,
		showIndicators:    showIndicators,
	}
}

func (m Model) renderPipelineSectionHeader(title string, width int) string {
	bar := lipgloss.NewStyle().Foreground(styles.Primary).Render(styles.SectionIcon)
	label := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render(" " + strings.ToUpper(title))
	totalBadge := lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("  %d subjects", m.stats.TotalSubjects))
	left := "  " + bar + label + totalBadge
	spacer := lipgloss.NewStyle().Width(max(width-lipgloss.Width(left), 0)).Render("")
	return left + spacer + "\n"
}

func (m Model) renderSubSectionHeader(title string, _ int) string {
	return "  " + lipgloss.NewStyle().Foreground(styles.TextDim).Bold(true).Render(title) + "\n"
}

func (m Model) renderEegSection(width int) string {
	var b strings.Builder
	layout := m.layoutForSection(width)

	b.WriteString(m.renderPipelineSectionHeader("EEG", width))

	totalSubjects := m.stats.TotalSubjects
	eegRows := []struct {
		label string
		count int
	}{
		{"BIDS", m.stats.BidsSubjects},
		{"EEG Prep", m.stats.EegPrepSubjects},
		{"Epochs", m.stats.EpochsSubjects},
		{"Features", m.stats.FeaturesSubjects},
	}

	for _, row := range eegRows {
		color := m.subjectRowColor(row.label, row.count, totalSubjects)
		b.WriteString(m.renderSubjectItem(row.label, row.count, color, totalSubjects, layout))
	}

	b.WriteString("\n")
	b.WriteString(m.renderFeatureCategories(width, layout))

	return b.String()
}

func (m Model) renderFmriSection(width int) string {
	var b strings.Builder
	layout := m.layoutForSection(width)

	b.WriteString(m.renderPipelineSectionHeader("fMRI", width))

	totalSubjects := m.stats.TotalSubjects
	fmriRows := []struct {
		label string
		count int
	}{
		{"BIDS", m.stats.BidsSubjects},
		{"fMRI Prep", m.stats.FmriPrepSubjects},
		{"First level", m.stats.FmriFirstLevelSubjects},
		{"Beta series", m.stats.FmriBetaSeriesSubjects},
		{"LSS", m.stats.FmriLssSubjects},
	}

	for _, row := range fmriRows {
		color := m.subjectRowColor(row.label, row.count, totalSubjects)
		b.WriteString(m.renderSubjectItem(row.label, row.count, color, totalSubjects, layout))
	}

	return b.String()
}

func (m Model) renderSubjectItem(label string, count int, color lipgloss.Color, totalSubjects int, layout sectionLayout) string {
	percentage := m.calculatePercentage(count, totalSubjects, false)

	labelText := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(layout.subjectLabelWidth).
		Render(styles.TruncateLine("  "+label, layout.subjectLabelWidth))

	progressBar := m.renderBlockBar(percentage, layout.subjectBarWidth, color)

	fractionText := lipgloss.NewStyle().
		Foreground(color).
		Bold(count > 0).
		Width(layout.countWidth).
		Align(lipgloss.Right).
		Render(fmt.Sprintf("%d/%d", count, totalSubjects))

	percentageText := m.formatPercentageText(percentage, false, layout.showPercent)

	zeroIndicator := ""
	if layout.showIndicators && totalSubjects > 0 && count == 0 {
		zeroIndicator = " " + lipgloss.NewStyle().Foreground(styles.Warning).Render("—")
	}

	line := labelText + fractionText + " " + progressBar + percentageText + zeroIndicator
	return styles.TruncateLine(line, layout.width) + "\n"
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

func (m Model) formatPercentageText(percentage float64, isTotal, showPercent bool) string {
	if isTotal || !showPercent {
		return ""
	}
	percentageValue := percentage * 100
	return lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render(fmt.Sprintf(" (%.0f%%)", percentageValue))
}

func (m Model) renderBlockBar(percentage float64, width int, color lipgloss.Color) string {
	if width <= 0 {
		return ""
	}
	filledWidth := int(percentage * float64(width))
	if filledWidth < 0 {
		filledWidth = 0
	}
	if filledWidth > width {
		filledWidth = width
	}
	emptyWidth := width - filledWidth

	filledBar := lipgloss.NewStyle().Foreground(color).Render(strings.Repeat("█", filledWidth))
	emptyBar := lipgloss.NewStyle().Foreground(styles.Border).Render(strings.Repeat("░", emptyWidth))

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

func (m Model) renderFeatureCategories(width int, layout sectionLayout) string {
	var b strings.Builder

	b.WriteString(m.renderSubSectionHeader("Feature Categories", width))

	if len(m.stats.FeatureCategories) == 0 {
		noDataMessage := lipgloss.NewStyle().
			Foreground(styles.Muted).
			Italic(true).
			Render("    No feature data available")
		b.WriteString(styles.TruncateLine(noDataMessage, width) + "\n")
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
		b.WriteString(styles.TruncateLine(summary, width) + "\n")
		categoryList := lipgloss.NewStyle().Foreground(styles.Muted).Italic(true).
			Render(fmt.Sprintf("    %d categories: %s", n, strings.Join(keys, ", ")))
		b.WriteString(styles.TruncateLine(categoryList, width) + "\n")
		return b.String()
	}

	for _, category := range keys {
		if category == "" {
			continue
		}
		b.WriteString(m.renderFeatureCategory(category, totalSubjects, layout))
	}

	return b.String()
}

func (m Model) getTotalForFeatureCategories() int {
	if m.stats.FeaturesSubjects > 0 {
		return m.stats.FeaturesSubjects
	}
	return m.stats.TotalSubjects
}

func (m Model) renderFeatureCategory(category string, totalSubjects int, layout sectionLayout) string {
	count := m.stats.FeatureCategories[category]
	percentage := m.calculatePercentage(count, totalSubjects, false)
	color := m.featureCategoryRowColor(count, totalSubjects)

	label := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Width(layout.featureLabelWidth).
		Render(styles.TruncateLine("    "+category, layout.featureLabelWidth))

	progressBar := m.renderBlockBar(percentage, layout.featureBarWidth, color)

	fractionText := lipgloss.NewStyle().
		Foreground(color).
		Bold(count > 0).
		Width(layout.featureCountWidth).
		Align(lipgloss.Right).
		Render(fmt.Sprintf("%d/%d", count, totalSubjects))

	zeroIndicator := ""
	if layout.showIndicators && totalSubjects > 0 && count == 0 {
		zeroIndicator = " " + lipgloss.NewStyle().Foreground(styles.Warning).Render("—")
	}

	line := label + fractionText + " " + progressBar + zeroIndicator
	return styles.TruncateLine(line, layout.width) + "\n"
}

func (m Model) renderFooter(width int) string {
	hints := strings.Join([]string{
		styles.RenderKeyHint("R", "Refresh"),
		styles.RenderFooterSeparator(),
		styles.RenderKeyHintSecondary("Esc", "Back"),
	}, "")
	divider := styles.RenderDivider(width)
	bar := styles.FooterStyle.Width(width).Render(hints)
	return divider + "\n" + bar
}
