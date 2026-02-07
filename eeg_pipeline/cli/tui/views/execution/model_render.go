package execution

import (
	"fmt"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

// Rendering helpers for status, progress, metrics, and log panels.

func (m Model) extractErrorSummary() string {
	if len(m.ErrorLines) == 0 {
		return ""
	}

	lastError := m.ErrorLines[len(m.ErrorLines)-1]

	parts := strings.Split(lastError, " - ")
	if len(parts) >= 2 {
		return strings.TrimSpace(strings.Join(parts[1:], " - "))
	}

	if len(lastError) > 80 {
		return lastError[:77] + "..."
	}
	return lastError
}

///////////////////////////////////////////////////////////////////
// View
///////////////////////////////////////////////////////////////////

// View implements tea.Model and composes all execution subviews
// (header, info, progress, logs and footer) into a single string.
func (m Model) View() string {
	var b strings.Builder

	// Header
	b.WriteString(m.renderHeader())
	b.WriteString("\n")

	if m.useTwoCol {
		sidebar := strings.Builder{}
		sidebar.WriteString(m.renderInfoPanel())
		sidebar.WriteString("\n\n")
		if m.IsDone() {
			sidebar.WriteString(m.renderCompletionSummary())
		} else {
			sidebar.WriteString(m.renderSidebarCard(m.renderProgressSection()))
		}

		logView := lipgloss.NewStyle().Width(m.leftWidth).Render(m.renderLogSection())
		sidebarView := lipgloss.NewStyle().Width(m.rightWidth).Render(sidebar.String())
		columns := lipgloss.JoinHorizontal(lipgloss.Top, logView, strings.Repeat(" ", m.columnGap), sidebarView)
		b.WriteString(columns)
		b.WriteString("\n")
	} else {
		// Log Section
		b.WriteString(m.renderLogSection())
		b.WriteString("\n")

		// Info Panel
		b.WriteString(m.renderInfoPanel())
		b.WriteString("\n")

		if m.IsDone() {
			// Show completion summary instead of progress section when done
			b.WriteString(m.renderCompletionSummary())
			b.WriteString("\n")
		} else {
			// Progress Section (only while running)
			b.WriteString(m.renderSidebarCard(m.renderProgressSection()))
			b.WriteString("\n")
		}
	}

	// Footer
	b.WriteString(m.renderFooter())

	return b.String()
}

// renderCompletionSummary produces a polished summary card shown
// once execution has finished, with status banner, metric tiles,
// output paths and action buttons.
func (m Model) renderCompletionSummary() string {
	var b strings.Builder

	var icon, statusText string
	var statusColor lipgloss.Color
	var borderColor lipgloss.Color

	switch m.Status {
	case StatusSuccess:
		icon = styles.CheckMark
		statusText = "Pipeline Complete"
		statusColor = styles.Success
		borderColor = styles.Success
	case StatusFailed:
		icon = styles.CrossMark
		statusText = "Pipeline Failed"
		statusColor = styles.Error
		borderColor = styles.Error
	case StatusCancelled:
		icon = styles.CrossMark
		statusText = "Cancelled"
		statusColor = styles.Warning
		borderColor = styles.Warning
	default:
		return ""
	}

	pw := m.panelWidth()
	iw := pw - 6

	// Status banner — full-width colored bar
	bannerStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.BgDark).
		Background(statusColor).
		Padding(0, 2).
		Width(iw)
	b.WriteString(bannerStyle.Render(icon+"  "+statusText) + "\n\n")

	// Metric tiles row — compact key stats
	duration := m.getDuration()
	tiles := m.buildCompletionTiles(duration)
	b.WriteString(styles.TruncateLine(m.renderMetricTiles(tiles, iw), iw) + "\n")

	// Failed subjects detail
	if len(m.FailedSubjects) > 0 {
		failStyle := lipgloss.NewStyle().Foreground(styles.Error)
		b.WriteString(styles.TruncateLine(failStyle.Render(fmt.Sprintf("  Failed: %s", strings.Join(m.FailedSubjects, ", "))), iw) + "\n")
	}

	// Error summary
	if len(m.ErrorLines) > 0 {
		errStyle := lipgloss.NewStyle().Foreground(styles.Warning)
		errText := fmt.Sprintf("%d error(s) detected", len(m.ErrorLines))
		if len(m.ErrorLines) > 5 {
			errText += " — scroll log to review"
		}
		b.WriteString(styles.TruncateLine("  "+errStyle.Render(styles.WarningMark+" "+errText), iw) + "\n")
	}

	// Output paths (success only)
	if m.Status == StatusSuccess && m.RepoRoot != "" {
		outputPaths := m.GetOutputPaths()
		if len(outputPaths) > 0 {
			b.WriteString("\n")
			pathLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Render("  Output:")
			b.WriteString(pathLabel + "\n")
			for _, p := range outputPaths {
				arrow := lipgloss.NewStyle().Foreground(styles.Accent).Render("  → ")
				path := lipgloss.NewStyle().Foreground(styles.Text).Render(p)
				b.WriteString(styles.TruncateLine(arrow+path, iw) + "\n")
			}
		}
	}

	// Action buttons
	b.WriteString("\n" + m.renderCompletionActions())

	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(1, 2).
		Width(pw)

	return cardStyle.Render(b.String())
}

// metricTile holds a single metric for the completion tile row.
type metricTile struct {
	label string
	value string
	color lipgloss.Color
}

// buildCompletionTiles assembles the metric tiles for the completion summary.
func (m Model) buildCompletionTiles(duration time.Duration) []metricTile {
	tiles := []metricTile{
		{label: "Duration", value: formatDuration(duration), color: styles.Primary},
	}

	if m.SubjectTotal > 0 && m.SubjectCurrent > 0 {
		tiles = append(tiles, metricTile{
			label: "Subjects",
			value: fmt.Sprintf("%d/%d", m.SubjectCurrent, m.SubjectTotal),
			color: styles.Accent,
		})
	}

	if len(m.SubjectDurations) > 0 {
		avgSec := m.averageSubjectDuration().Seconds()
		if avgSec > 0 {
			rate := 3600.0 / avgSec
			rateStr := fmt.Sprintf("%.1f/hr", rate)
			if rate < 1 {
				rateStr = fmt.Sprintf("%.0fm ea", avgSec/60)
			}
			tiles = append(tiles, metricTile{label: "Rate", value: rateStr, color: styles.Primary})
		}
	}

	if m.Status == StatusFailed {
		tiles = append(tiles, metricTile{
			label: "Exit",
			value: fmt.Sprintf("%d", m.ExitCode),
			color: styles.Error,
		})
	}

	logVal := fmt.Sprintf("%d", len(m.OutputLines))
	if m.LogTruncated {
		logVal += "+"
	}
	tiles = append(tiles, metricTile{label: "Log", value: logVal, color: styles.TextDim})

	return tiles
}

// renderMetricTiles renders a horizontal row of compact metric tiles.
func (m Model) renderMetricTiles(tiles []metricTile, maxWidth int) string {
	if len(tiles) == 0 {
		return ""
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

	var parts []string
	for _, t := range tiles {
		val := lipgloss.NewStyle().Foreground(t.color).Bold(true).Render(t.value)
		lbl := labelStyle.Render(t.label)
		parts = append(parts, "  "+lbl+" "+val)
	}

	return strings.Join(parts, lipgloss.NewStyle().Foreground(styles.Border).Render("  │  "))
}

// renderCompletionActions renders the action button bar for the completion card.
func (m Model) renderCompletionActions() string {
	btnStyle := func(bg lipgloss.Color) lipgloss.Style {
		return lipgloss.NewStyle().
			Foreground(styles.BgDark).
			Background(bg).
			Bold(true).
			Padding(0, 1)
	}
	dimBtn := lipgloss.NewStyle().
		Foreground(styles.Text).
		Background(styles.Border).
		Padding(0, 1)

	var buttons []string
	switch m.Status {
	case StatusSuccess:
		buttons = append(buttons,
			btnStyle(styles.Success).Render("[Enter] Menu"),
			btnStyle(styles.Accent).Render("[O] Open Results"),
			dimBtn.Render("[C] Copy Log"),
		)
	case StatusFailed:
		buttons = append(buttons,
			btnStyle(styles.Warning).Render("[R] Retry"),
			btnStyle(styles.Error).Render("[C] Copy Log"),
			dimBtn.Render("[Enter] Menu"),
		)
	case StatusCancelled:
		buttons = append(buttons,
			btnStyle(styles.Accent).Render("[R] Retry"),
			dimBtn.Render("[Enter] Menu"),
		)
	}

	return "  " + strings.Join(buttons, "  ")
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1f seconds", d.Seconds())
	} else if d < time.Hour {
		mins := int(d.Minutes())
		secs := int(d.Seconds()) % 60
		return fmt.Sprintf("%d min %d sec", mins, secs)
	}
	hours := int(d.Hours())
	mins := int(d.Minutes()) % 60
	return fmt.Sprintf("%d hr %d min", hours, mins)
}

// renderHeader renders the top title bar indicating local vs cloud
// execution and colors it according to the current status.
func (m Model) renderHeader() string {
	title := "Pipeline Execution"
	if m.IsCloud {
		title = "Cloud Execution"
	}

	statusColor := styles.Border
	switch m.Status {
	case StatusRunning:
		statusColor = styles.Accent
	case StatusSuccess:
		statusColor = styles.Success
	case StatusFailed:
		statusColor = styles.Error
	case StatusCancelled:
		statusColor = styles.Warning
	}

	bar := lipgloss.NewStyle().Foreground(statusColor).Render(styles.SectionIcon)
	headerText := lipgloss.NewStyle().Bold(true).Foreground(styles.Text).Render(" " + title)
	headerLine := lipgloss.PlaceHorizontal(m.width-4, lipgloss.Center, bar+headerText)

	lineWidth := m.width - 4
	if lineWidth < 0 {
		lineWidth = 0
	}
	return headerLine + "\n" + styles.RenderDivider(lineWidth)
}

// renderInfoPanel renders a compact timing card with elapsed and ETA.
// Subject details are in the progress section.
func (m Model) renderInfoPanel() string {
	var rows []string

	const labelW = 6
	iw := m.sidebarInnerWidth()

	if m.StartTime.Unix() > 0 {
		duration := m.getDuration()
		timeLine := styles.RenderKeyValue("Time", formatDuration(duration), labelW)
		if m.Status == StatusRunning && m.EstimatedRemaining > 0 {
			etaStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			timeLine += lipgloss.NewStyle().Foreground(styles.TextDim).Render("  ETA ") + etaStyle.Render(formatDuration(m.EstimatedRemaining))
		}
		rows = append(rows, styles.TruncateLine(timeLine, iw))
	}

	if len(m.FailedSubjects) > 0 {
		failStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
		line := lipgloss.NewStyle().Foreground(styles.Error).Render(styles.PadRight("Fail", labelW)) +
			failStyle.Render(fmt.Sprintf("%d subject(s)", len(m.FailedSubjects)))
		rows = append(rows, styles.TruncateLine(line, iw))
	}

	content := strings.Join(rows, "\n")
	return m.renderSidebarCard(content)
}

func (m *Model) calculateETA() {
	if len(m.SubjectDurations) == 0 || m.SubjectTotal == 0 {
		m.EstimatedRemaining = 0
		return
	}

	avgDuration := m.averageSubjectDuration()
	remainingSubjects := m.SubjectTotal - m.SubjectCurrent
	if remainingSubjects < 0 {
		remainingSubjects = 0
	}

	m.EstimatedRemaining = time.Duration(remainingSubjects) * avgDuration
}

func (m Model) averageSubjectDuration() time.Duration {
	if len(m.SubjectDurations) == 0 {
		return 0
	}

	var total time.Duration
	for _, d := range m.SubjectDurations {
		total += d
	}
	return total / time.Duration(len(m.SubjectDurations))
}

// renderProgressSection renders the main progress overview including
// overall completion, current subject/step, metrics and cloud stages.
func (m Model) renderProgressSection() string {
	var b strings.Builder

	iw := m.sidebarInnerWidth()

	b.WriteString(styles.TruncateLine(m.renderStatus()+"  "+styles.RenderSectionLabel("Progress"), iw) + "\n")

	// Overall progress bar
	barWidth := iw - 10
	if barWidth < 8 {
		barWidth = 8
	}
	b.WriteString("  " + m.renderAnimatedProgressBar(m.Progress, barWidth) + "\n")

	// Subject tracker: compact dot strip with counter
	if m.SubjectTotal > 0 && m.SubjectCurrent > 0 {
		b.WriteString(styles.TruncateLine("  "+m.renderSubjectTracker(iw-4), iw) + "\n")
	}

	// Current step: inline subject → operation
	if m.CurrentSubject != "" || m.CurrentOperation != "" {
		b.WriteString(styles.TruncateLine("  "+m.renderCurrentStep(iw-4), iw) + "\n")
	}

	// Metrics dashboard (compact heatmap + gauges)
	if m.height >= 28 {
		b.WriteString("\n" + m.renderMetricsDashboard(iw) + "\n")
	}

	// Cloud stages
	if m.IsCloud {
		b.WriteString("\n  " + m.renderCloudStages() + "\n")
	}

	return b.String()
}

// renderSubjectTracker renders a compact subject progress line with dot indicators.
func (m Model) renderSubjectTracker(maxWidth int) string {
	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	pctStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	subjectProgress := float64(m.SubjectCurrent) / float64(m.SubjectTotal)

	// Dot strip: ●●●○○○ — cap at 20 dots for space
	maxDots := m.SubjectTotal
	if maxDots > 20 {
		maxDots = 20
	}
	var dots strings.Builder
	for i := 0; i < maxDots; i++ {
		// Map dot index to subject index for >20 subjects
		mappedIdx := i
		if m.SubjectTotal > 20 {
			mappedIdx = i * m.SubjectTotal / 20
		}
		switch {
		case mappedIdx < m.SubjectCurrent-1:
			// Completed
			status, isFailed := m.SubjectStatuses[m.subjectIDForIndex(mappedIdx)]
			if isFailed && status == "failed" {
				dots.WriteString(lipgloss.NewStyle().Foreground(styles.Error).Render("●"))
			} else {
				dots.WriteString(lipgloss.NewStyle().Foreground(styles.Success).Render("●"))
			}
		case mappedIdx == m.SubjectCurrent-1:
			// Current (running)
			dots.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render("◉"))
		default:
			// Pending
			dots.WriteString(lipgloss.NewStyle().Foreground(styles.Border).Render("○"))
		}
	}
	if m.SubjectTotal > 20 {
		dots.WriteString(dimStyle.Render(fmt.Sprintf(" +%d", m.SubjectTotal-20)))
	}

	counter := dimStyle.Render(fmt.Sprintf("%d/%d ", m.SubjectCurrent, m.SubjectTotal))
	pct := pctStyle.Render(fmt.Sprintf("%.0f%%", subjectProgress*100))

	return counter + dots.String() + " " + pct
}

// subjectIDForIndex returns the subject ID for a given 0-based index, or empty string.
func (m Model) subjectIDForIndex(idx int) string {
	// SubjectStatuses is keyed by subject ID; we don't have an ordered list,
	// so return empty to skip failed-dot coloring when index can't be resolved.
	return ""
}

// renderCurrentStep renders the active subject and operation on one compact line.
func (m Model) renderCurrentStep(maxWidth int) string {
	dimStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	accentStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	var parts []string

	if m.CurrentSubject != "" {
		parts = append(parts, accentStyle.Render(m.CurrentSubject))
	}

	if m.CurrentOperation != "" {
		opText := m.CurrentOperation
		if m.OperationTotal > 0 {
			stepProgress := float64(m.OperationCurrent) / float64(m.OperationTotal)
			opText += fmt.Sprintf(" %d/%d", m.OperationCurrent, m.OperationTotal)
			// Inline mini bar (8 chars)
			barWidth := 8
			filled := int(stepProgress * float64(barWidth))
			bar := lipgloss.NewStyle().Foreground(styles.Primary).Render(strings.Repeat("━", filled))
			empty := lipgloss.NewStyle().Foreground(styles.Border).Render(strings.Repeat("─", barWidth-filled))
			parts = append(parts, dimStyle.Render("→ ")+dimStyle.Render(opText)+" "+bar+empty)
		} else {
			parts = append(parts, dimStyle.Render("→ "+opText))
		}
	}

	return strings.Join(parts, " ")
}

// renderMetricsDashboard renders a compact resource dashboard with
// CPU heatmap, memory gauge, and throughput rate.
func (m Model) renderMetricsDashboard(maxWidth int) string {
	if m.Status == StatusPending {
		return ""
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true)

	var rows []string

	cpuLine := m.renderCPUHeatmap()
	rows = append(rows, styles.TruncateLine(cpuLine, maxWidth))

	memLine := "  " + labelStyle.Render("Mem ") + m.renderMemoryGauge(12) + " " + valueStyle.Render(fmt.Sprintf("%.1f GB", m.MemoryUsage))
	if m.EpochInfo != "" {
		memLine += labelStyle.Render("  Item ") + valueStyle.Render(m.EpochInfo)
	}
	rows = append(rows, styles.TruncateLine(memLine, maxWidth))

	if len(m.SubjectDurations) > 0 && m.SubjectTotal > 0 {
		avgSec := m.averageSubjectDuration().Seconds()
		if avgSec > 0 {
			rate := 3600.0 / avgSec
			rateStr := fmt.Sprintf("%.1f subj/hr", rate)
			if rate < 1 {
				rateStr = fmt.Sprintf("%.0f min/subj", avgSec/60)
			}
			rows = append(rows, styles.TruncateLine("  "+labelStyle.Render("Rate ")+valueStyle.Render(rateStr)+
				labelStyle.Render("  Done ")+valueStyle.Render(fmt.Sprintf("%d/%d", len(m.SubjectDurations), m.SubjectTotal)), maxWidth))
		}
	}

	return strings.Join(rows, "\n")
}

// renderCPUHeatmap renders a compact single-line heatmap of per-core CPU usage
// using Unicode block characters (▁▂▃▄▅▆▇█) colored by utilization level.
func (m Model) renderCPUHeatmap() string {
	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)

	if m.NumCPUCores == 0 || len(m.CPUCoreUsages) == 0 {
		return "  " + labelStyle.Render("Cpu ") + valueStyle.Render(fmt.Sprintf("%.0f%%", m.CPUUsage))
	}

	blocks := []string{"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}

	var heatmap strings.Builder
	for _, usage := range m.CPUCoreUsages {
		u := usage
		if u < 0 {
			u = 0
		}
		if u > 100 {
			u = 100
		}

		idx := int(u / 100.0 * float64(len(blocks)-1))
		if idx >= len(blocks) {
			idx = len(blocks) - 1
		}

		var color lipgloss.Color
		switch {
		case u < 25:
			color = styles.Success
		case u < 50:
			color = styles.Primary
		case u < 75:
			color = styles.Accent
		case u < 90:
			color = styles.Warning
		default:
			color = styles.Error
		}
		heatmap.WriteString(lipgloss.NewStyle().Foreground(color).Render(blocks[idx]))
	}

	// Aggregate stats
	numCores := len(m.CPUCoreUsages)
	avgUsage := 0.0
	for _, u := range m.CPUCoreUsages {
		avgUsage += u
	}
	if numCores > 0 {
		avgUsage /= float64(numCores)
	}

	summary := valueStyle.Render(fmt.Sprintf(" %.0f%%", m.CPUUsage)) +
		labelStyle.Render(fmt.Sprintf(" avg %.0f%% · %dc", avgUsage, numCores))

	return "  " + labelStyle.Render("CPU ") + heatmap.String() + summary
}

// renderMemoryGauge renders a compact memory usage gauge bar.
func (m Model) renderMemoryGauge(width int) string {
	// Estimate usage fraction (cap at 32 GB as "full" for visualization)
	const maxMemGB = 32.0
	fraction := m.MemoryUsage / maxMemGB
	if fraction > 1.0 {
		fraction = 1.0
	}
	if fraction < 0 {
		fraction = 0
	}

	filled := int(fraction * float64(width))
	if filled > width {
		filled = width
	}

	var color lipgloss.Color
	switch {
	case fraction < 0.5:
		color = styles.Success
	case fraction < 0.75:
		color = styles.Accent
	case fraction < 0.9:
		color = styles.Warning
	default:
		color = styles.Error
	}

	bar := lipgloss.NewStyle().Foreground(color).Render(strings.Repeat("█", filled))
	empty := lipgloss.NewStyle().Foreground(styles.Border).Render(strings.Repeat("░", width-filled))
	return bar + empty
}

func (m Model) renderLogSection() string {
	var b strings.Builder
	contentWidth := m.logViewport.Width

	// Copy mode banner
	if m.copyMode {
		copyBanner := lipgloss.NewStyle().
			Bold(true).
			Foreground(styles.BgDark).
			Background(styles.Accent).
			Padding(0, 2).
			Render("Copy mode: Select text with mouse, then Cmd+C. Press M or Esc to exit.")
		b.WriteString(lipgloss.NewStyle().Width(contentWidth).Render(copyBanner) + "\n")
	}

	logHeader := styles.RenderSectionLabel("Log")
	if len(m.OutputLines) > 0 {
		scrollPct := 0
		if m.logViewport.TotalLineCount() > 0 {
			scrollPct = int(float64(m.logViewport.YOffset+m.logViewport.Height) / float64(m.logViewport.TotalLineCount()) * 100)
			if scrollPct > 100 {
				scrollPct = 100
			}
		}

		indicator := lipgloss.NewStyle().Foreground(styles.Muted).Render(
			fmt.Sprintf(" [%d lines | %d%%]", len(m.OutputLines), scrollPct))
		logHeader += indicator
	}

	b.WriteString(lipgloss.NewStyle().Width(contentWidth).Render(logHeader) + "\n")
	b.WriteString(styles.RenderDivider(contentWidth) + "\n")
	b.WriteString(m.logViewport.View())

	return b.String()
}

// renderCloudStages renders a compact multi‑stage indicator used
// during cloud execution (sync, run, pull).
func (m Model) renderCloudStages() string {
	stages := []struct {
		stage CloudStage
		name  string
	}{
		{StageSyncing, "Sync"},
		{StageRunning, "Run"},
		{StagePulling, "Pull"},
	}

	var parts []string
	for _, s := range stages {
		style := lipgloss.NewStyle().Foreground(styles.Muted)
		marker := "[ ]"

		if s.stage < m.CloudStage {
			style = style.Foreground(styles.Success)
			marker = "[" + styles.CheckMark + "]"
		} else if s.stage == m.CloudStage && m.Status == StatusRunning {
			style = style.Foreground(styles.Accent).Bold(true)
			marker = "[" + styles.ActiveMark + "]"
		} else if m.CloudStage == StageDone {
			style = style.Foreground(styles.Success)
			marker = "[" + styles.CheckMark + "]"
		}

		parts = append(parts, style.Render(marker+" "+s.name))
	}

	return strings.Join(parts, "   ")
}

// renderAnimatedProgressBar renders the main progress bar (single accent color; name kept for API).
// When running, the leading edge pulses via the animation queue.
func (m Model) renderAnimatedProgressBar(p float64, width int) string {
	if width < styles.MinProgressBarWidth {
		width = styles.MinProgressBarWidth
	}
	if width > styles.MaxProgressBarWidth {
		width = styles.MaxProgressBarWidth
	}

	filled := int(p * float64(width))
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}

	fillStyle := lipgloss.NewStyle().Foreground(styles.Primary)
	emptyStyle := lipgloss.NewStyle().Foreground(styles.Border)

	var fillBlock string
	if m.Status == StatusRunning {
		kind, progress := m.animQueue.Current()
		if kind == animation.KindProgressPulse && filled > 0 {
			leadPulse := "▓"
			if progress < 0.5 {
				leadPulse = "█"
			}
			fillBlock = fillStyle.Render(strings.Repeat("█", filled-1) + leadPulse)
		} else {
			fillBlock = fillStyle.Render(strings.Repeat("█", filled))
		}
	} else {
		fillBlock = fillStyle.Render(strings.Repeat("█", filled))
	}

	bar := fillBlock + emptyStyle.Render(strings.Repeat("░", width-filled))
	pct := lipgloss.NewStyle().Bold(true).Foreground(styles.Primary).Render(fmt.Sprintf(" %3.0f%%", p*100))
	return bar + pct
}

// renderStatus renders a small status badge summarizing the current
// execution state with an appropriate color and icon.
func (m Model) renderStatus() string {
	style := lipgloss.NewStyle().Bold(true).Padding(0, 1)
	switch m.Status {
	case StatusRunning:
		return style.Background(styles.Primary).Foreground(styles.BgDark).Render(styles.ActiveMark + " Running")
	case StatusSuccess:
		return style.Background(styles.Success).Foreground(styles.BgDark).Render(styles.CheckMark + " Success")
	case StatusFailed:
		return style.Background(styles.Error).Foreground(styles.BgDark).Render(styles.CrossMark + " Failed")
	case StatusCancelled:
		return style.Background(styles.Warning).Foreground(styles.BgDark).Render(styles.CrossMark + " Cancelled")
	default:
		return style.Foreground(styles.Muted).Render(" Pending ")
	}
}

func (m Model) getDuration() time.Duration {
	if m.Status == StatusRunning {
		return time.Since(m.StartTime)
	}
	if m.EndTime.Unix() > 0 {
		return m.EndTime.Sub(m.StartTime)
	}
	return 0
}

func (m Model) renderFooter() string {
	var hints []string

	if m.copyMode {
		hints = []string{
			styles.RenderKeyHint("M/Esc", "Exit Copy Mode"),
			styles.RenderKeyHint("C", "Copy All"),
			lipgloss.NewStyle().Foreground(styles.Accent).Italic(true).Render("Select text with mouse, then Cmd+C"),
		}
	} else if m.IsDone() {
		hints = []string{
			styles.RenderKeyHint("Enter", "Return to Menu"),
			styles.RenderKeyHint("C", "Copy Log"),
			styles.RenderKeyHint("O", "Open Results"),
			styles.RenderKeyHint("R", "Retry"),
		}
	} else {
		hints = []string{
			styles.RenderKeyHint("Ctrl+C", "Cancel"),
			styles.RenderKeyHint("C", "Copy Log"),
			styles.RenderKeyHint("M", "Copy Mode"),
			styles.RenderKeyHint("\u2191/\u2193", "Scroll"),
			styles.RenderKeyHint("G/Shift+G", "Top/Bottom"),
		}
	}

	width := m.width - 8
	if width < 20 {
		width = 20
	}
	divider := styles.RenderDivider(width)
	bar := styles.FooterStyle.Width(width).Render(strings.Join(hints, styles.RenderFooterSeparator()))
	return divider + "\n" + bar
}

///////////////////////////////////////////////////////////////////
// Resource Monitoring
///////////////////////////////////////////////////////////////////

// startResourceMonitoring begins monitoring CPU and memory usage of the process
