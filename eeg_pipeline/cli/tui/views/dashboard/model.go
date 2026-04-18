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
	defaultWidth    = 120
	defaultHeight   = 32
	minSectionWidth = 28
)

type sectionLayout struct {
	width             int
	countWidth        int
	featureCountWidth int
	subjectLabelWidth int
	featureLabelWidth int
	showPercent       bool
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
	pythonCmd := executor.ResolvePythonCommand(m.repoRoot)
	args := []string{"-m", "eeg_pipeline", "stats", "--json"}

	cmd := exec.Command(pythonCmd.Executable, pythonCmd.Args(args...)...)
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

// contentWidth is the inner text width inside BoxStyle — must match what
// RenderNoWrapBlock uses when clamping lines. Previously this subtracted a
// fixed constant from the terminal width while the box outer width was
// terminal−2, so every line was ~2 cells too wide and ClampBlock appended
// "..." on every row (a vertical run of ellipses on the right edge).
func (m Model) contentWidth() int {
	outer := m.boxWidth()
	inner := outer - styles.BoxStyle.GetHorizontalFrameSize()
	if inner < 1 {
		return 1
	}
	return inner
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

	// Every physical row must be exactly `contentWidth` cells wide before the
	// outer BoxStyle clamps — otherwise lipgloss join / footer can sit one
	// cell over and ClampBlock turns the overrun into a trailing "..." on the right.
	body := m.fitDashboardContentWidth(b.String(), contentWidth)
	return styles.RenderNoWrapBlock(styles.BoxStyle, body, m.boxWidth())
}

// fitDashboardContentWidth pads or truncates each non-empty line to exactly w
// cells so nested blocks (two-column join, rules) never exceed the box inner width.
func (m Model) fitDashboardContentWidth(s string, w int) string {
	if w <= 0 {
		return s
	}
	lines := strings.Split(s, "\n")
	for i := range lines {
		if lines[i] == "" {
			continue
		}
		lines[i] = styles.FitLine(lines[i], w)
	}
	return strings.Join(lines, "\n")
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
	// Match the app-wide header pattern: hairline accent bar + uppercase
	// title, anchored by the divider beneath. Drops the decorative "◈"
	// ornament in favor of a consistent typographic rhythm.
	bar := lipgloss.NewStyle().Foreground(styles.Primary).Render(styles.SectionIconActive)
	title := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).
		Render(strings.ToUpper("Project Dashboard"))
	headerLine := "  " + bar + " " + title
	sep := styles.RenderHeaderSeparator(width)
	// Blank row between title and full-width hairline so the chrome is not cramped.
	return headerLine + "\n\n" + sep + "\n"
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
	// One quiet row below the title rule so the summary strip is not glued to
	// the header hairline.
	b.WriteString("\n")
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
	taskValue := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.stats.Task)
	taskLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("task  ")

	countValue := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).
		Render(fmt.Sprintf("%d", m.stats.TotalSubjects))
	subjectsLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("  subjects  ")

	left := "  " + taskLabel + taskValue + subjectsLabel + countValue

	updatedAt := ""
	if !m.lastUpdate.IsZero() {
		updatedAt = lipgloss.NewStyle().Foreground(styles.Muted).
			Render("Updated " + m.lastUpdate.Format("15:04:05"))
	}

	spacer := lipgloss.NewStyle().Width(max(width-lipgloss.Width(left)-lipgloss.Width(updatedAt), 0)).Render("")
	line := left + spacer + updatedAt
	line = styles.TruncateLine(line, width)
	return line + "\n" + styles.RenderDivider(width) + "\n"
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

const columnSepWidth = 3 // " │ "

func (m Model) statsColumnWidths(width int) (int, int) {
	// Balanced 50/50 split so EEG vs fMRI blocks sit symmetrically; the old
	// 54/46 split left a wide empty gutter and pushed fMRI content toward the edge.
	usable := width - columnSepWidth
	leftWidth := usable / 2
	rightWidth := usable - leftWidth
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
	sep := lipgloss.NewStyle().Foreground(styles.Border).Render(strings.Repeat(styles.SectionDividerChar, 1))

	joined := lipgloss.JoinHorizontal(lipgloss.Top, left, " "+sep+" ", right)
	return styles.ClampBlock(joined, width)
}

func (m Model) layoutForSection(width int) sectionLayout {
	fractionWidth := max(len(fmt.Sprintf("%d/%d", m.stats.TotalSubjects, m.stats.TotalSubjects)), 3)
	subjectLabelWidth := min(max(width/4, 12), 18)
	featureLabelWidth := min(max(width/3, 20), 32)
	showPercent := width >= 50

	return sectionLayout{
		width:             width,
		countWidth:        fractionWidth,
		featureCountWidth: fractionWidth,
		subjectLabelWidth: subjectLabelWidth,
		featureLabelWidth: featureLabelWidth,
		showPercent:       showPercent,
	}
}

func (m Model) pipelineSectionColor(title string) lipgloss.Color {
	// Keep all pipeline section headers on the same monochrome axis — the
	// title ("EEG" / "fMRI") carries the semantic distinction, not color.
	return styles.Primary
}

func (m Model) renderPipelineSectionHeader(title string, width int) string {
	color := m.pipelineSectionColor(title)
	icon := lipgloss.NewStyle().Foreground(color).Render(styles.SectionIconActive)
	label := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).
		Render(" " + strings.ToUpper(title))
	countValue := lipgloss.NewStyle().Foreground(styles.Text).Bold(true).
		Render(fmt.Sprintf("%d", m.stats.TotalSubjects))
	countLabel := lipgloss.NewStyle().Foreground(styles.Muted).Render(" subjects")
	countText := countValue + countLabel
	left := "  " + icon + label
	spacerW := max(width-lipgloss.Width(left)-lipgloss.Width(countText), 0)
	spacer := lipgloss.NewStyle().Width(spacerW).Render("")
	line := styles.TruncateLine(left+spacer+countText, width)
	return line + "\n"
}

func (m Model) renderSubSectionHeader(title string, width int) string {
	// Uppercase + muted title + trailing hairline rule, matching the
	// "DETAILS / WORKSPACE / FOCUS" sub-header rhythm used elsewhere.
	return "  " + styles.RenderPreviewSubHeaderWithRule(title, width-2) + "\n"
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
	b.WriteString("\n")

	return b.String()
}

func (m Model) renderSubjectItem(label string, count int, color lipgloss.Color, totalSubjects int, layout sectionLayout) string {
	percentage := m.calculatePercentage(count, totalSubjects, false)

	labelText := lipgloss.NewStyle().Foreground(styles.Text).Render(
		styles.PadRight(styles.TruncateLine("  "+label, layout.subjectLabelWidth), layout.subjectLabelWidth),
	)

	fractionStr := fmt.Sprintf("%d/%d", count, totalSubjects)
	fractionText := lipgloss.NewStyle().Foreground(color).Bold(count > 0).Render(
		styles.PadRight(fractionStr, layout.countWidth),
	)

	percentageText := m.formatMutedPercentSuffix(percentage, layout.showPercent)

	line := labelText + "  " + fractionText + percentageText
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

// formatMutedPercentSuffix appends a quiet " · 42%" after the fraction when
// there is room — coverage is encoded typographically, not with block bars.
func (m Model) formatMutedPercentSuffix(percentage float64, showPercent bool) string {
	if !showPercent {
		return ""
	}
	pct := percentage * 100
	// Thin space before the percent keeps it attached to the row without a
	// heavy parenthetical.
	return lipgloss.NewStyle().Foreground(styles.Muted).
		Render(fmt.Sprintf("  · %3.0f%%", pct))
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

	label := lipgloss.NewStyle().Foreground(styles.TextDim).Render(
		styles.PadRight(styles.TruncateLine("    "+category, layout.featureLabelWidth), layout.featureLabelWidth),
	)

	fractionStr := fmt.Sprintf("%d/%d", count, totalSubjects)
	fractionText := lipgloss.NewStyle().Foreground(color).Bold(count > 0).Render(
		styles.PadRight(fractionStr, layout.featureCountWidth),
	)

	percentageText := m.formatMutedPercentSuffix(percentage, layout.showPercent)

	// Zero coverage is visible via fraction styling (Warning); a trailing em
	// dash was removed — it misaligned the right edge when only some rows had one.
	line := label + "  " + fractionText + percentageText
	return styles.TruncateLine(line, layout.width) + "\n"
}

func (m Model) renderFooter(width int) string {
	// Two-hint footers look cramped when both hints cluster at the left edge
	// (the standard "    ·    " separator is only 9 cells wide). Anchor the
	// primary action on the left and the navigation hint on the right so the
	// two read as distinct affordances, mirroring a conventional status bar.
	left := styles.RenderKeyHint("R", "Refresh")
	right := styles.RenderKeyHint("Esc", "Back")
	gap := max(width-lipgloss.Width(left)-lipgloss.Width(right), 1)
	bar := left + lipgloss.NewStyle().Width(gap).Render("") + right
	bar = styles.TruncateLine(bar, width)
	divider := styles.RenderFooterDivider(width)
	return divider + "\n" + styles.FooterStyle.Render(bar)
}
